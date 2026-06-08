#!/usr/bin/env python3
"""
V4.23 support-normal grasp projection.

定位：
    把“手往上抬一点，给手指闭合留空间”变成通用规则。
    不是手工微调单个 sample。

输入：
    V4.20d selected_topk_feasible.json

对每个 candidate：
    T_prior = T_world_object @ T_object_hand
    T_proj(dz) = T_prior + dz * world_z

对每个 dz：
    1. IK 到 projected q_grasp；
    2. one-time exact setup 到 q_grasp/open，用来评估这个 projected pose 本身是否推物体；
    3. dynamic short close；
    4. 判断 thumb + 至少一根非拇指；
    5. 记录 exact_setup_disp / close_disp / support freeze / ready；
    6. 选出最小改变量下最可靠的 projected pose。

输出：
    projected_candidates_ranked.json
    best_projected_candidate.json
    projection_report.txt

注意：
    V4.23 不是最终执行器；
    V4.23 只负责生成 projected grasp target。
"""

from pathlib import Path
import argparse
import importlib.util
import json
import numpy as np
import mujoco


PROJECT = Path.home() / "Projects/o7_mujoco_sim"
HELPER = PROJECT / "scripts/05_execution_runner/run_v4_17_exact_site_qgrasp_close_lift_debug.py"

spec = importlib.util.spec_from_file_location("v417_helper", str(HELPER))
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)

NON_THUMB = ["index", "middle", "ring", "pinky"]


def resolve(p):
    p = Path(str(p)).expanduser()
    return p if p.is_absolute() else PROJECT / p


def rel(p):
    p = Path(p)
    try:
        return str(p.relative_to(PROJECT))
    except Exception:
        return str(p)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def load_json(path):
    return json.loads(Path(path).read_text())


def side_open_from_close(close_ctrl):
    side = dict(close_ctrl)
    for j in ["index_mcp_pitch", "middle_mcp_pitch", "ring_mcp_pitch", "pinky_mcp_pitch"]:
        side[j] = 0.0
    return side


def real_groups_from_contacts(state, kind, dist_th):
    raw = state.get("object_contacts", []) if kind == "object" else state.get("support_contacts", [])
    groups = {}
    contacts = []
    for c in raw:
        d = float(c.get("dist", 999.0))
        if d <= dist_th:
            g = c.get("group")
            if g is None:
                continue
            groups[g] = groups.get(g, 0) + 1
            contacts.append(c)
    return groups, contacts


def ready_from_groups(groups):
    return ("thumb" in groups) and any(g in groups for g in NON_THUMB)


def make_hand_ctrl(side_open, close_ctrl, group_alpha):
    ctrl = dict(side_open)
    for g, joints in H.FINGER_GROUP_TO_JOINTS.items():
        a = float(group_alpha.get(g, 0.0))
        for j in joints:
            v0 = float(side_open.get(j, close_ctrl.get(j, 0.0)))
            v1 = float(close_ctrl.get(j, v0))
            ctrl[j] = (1.0 - a) * v0 + a * v1
    return ctrl


def step_ctrl(model, data, q_arm, hand_ctrl):
    H.apply_ctrl(model, data, q_arm, hand_ctrl)
    mujoco.mj_step(model, data)


def current_site_error(model, data, site_name, T_target):
    T_cur = H.site_world_T(model, data, site_name)
    _, _, pos_n, rot_n = H.pose_error(T_cur, T_target)
    return float(pos_n), float(rot_n)


def eval_one_projection(args, candidate, dz, rank_from_selector):
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)

    local_idx = int(candidate["valid_local_index"])
    raw_idx = int(candidate.get("raw_sample_index", local_idx))

    sample = H.load_sample(args.npy, local_idx)
    T_object_hand = H.sample_T_object_hand(sample)
    close_ctrl = H.sample_ctrl(sample)
    side_open = side_open_from_close(close_ctrl)

    # scene settle
    H.set_qpos_once(model, data, H.Q_HOME, side_open)
    for _ in range(args.settle_steps):
        step_ctrl(model, data, H.Q_HOME, side_open)

    object_before_setup = H.object_pos(model, data, args.object_body).copy()
    T_world_object = H.body_world_T(model, data, args.object_body)

    T_prior = T_world_object @ T_object_hand
    T_proj = np.array(T_prior, dtype=float)
    T_proj[:3, 3] += np.array([0.0, 0.0, float(dz)], dtype=float)

    ik = H.solve_site_ik(
        model,
        args.target_site,
        T_proj,
        H.Q_HOME,
        max_iters=args.ik_iters,
    )

    row = {
        "rank_from_selector": int(rank_from_selector),
        "valid_local_index": local_idx,
        "raw_sample_index": raw_idx,
        "selector_type": candidate.get("grasp_type", candidate.get("selector_type", "")),
        "selector_score": float(candidate.get("score", 0.0)),
        "dz": float(dz),
        "T_world_object": H.mat_to_dict(T_world_object),
        "T_object_hand": H.mat_to_dict(T_object_hand),
        "T_grasp_prior": H.mat_to_dict(T_prior),
        "T_grasp_projected": H.mat_to_dict(T_proj),
        "ik": ik,
    }

    if not ik["success"]:
        row.update({
            "projection_ready": False,
            "projection_success": False,
            "projection_score": -1e6,
            "failure_reason": "ik_failed",
        })
        return row

    q_grasp = ik["q_arm"]

    # exact setup：只做一次 set_qpos，用来评估 projected q_grasp/open 是否会直接推物体
    H.set_qpos_once(model, data, q_grasp, side_open)

    for _ in range(args.exact_setup_steps):
        step_ctrl(model, data, q_grasp, side_open)

    object_after_setup = H.object_pos(model, data, args.object_body).copy()
    exact_setup_disp = float(np.linalg.norm(object_after_setup - object_before_setup))

    setup_site_pos_err, setup_site_rot_err = current_site_error(
        model, data, args.target_site, T_proj
    )

    object_start_for_close = object_after_setup.copy()

    # open hold，记录 open 是否已经有接触/支撑
    open_state = H.contact_state(model, data, args.object_body)
    open_obj_groups, _ = real_groups_from_contacts(open_state, "object", args.object_ready_dist)
    open_support_groups, _ = real_groups_from_contacts(open_state, "support", args.support_freeze_dist)

    # short dynamic close
    group_alpha = {g: 0.0 for g in H.FINGER_GROUP_TO_JOINTS}
    frozen = {g: False for g in H.FINGER_GROUP_TO_JOINTS}
    freeze_reason = {}
    hand_ctrl = dict(side_open)

    stable_ready = 0
    max_stable_ready = 0
    max_close_disp = 0.0
    max_obj_groups = {}
    max_support_groups = {}
    final_obj_groups = {}
    final_support_groups = {}

    close_rows = []

    for k in range(args.close_steps):
        st_before = H.contact_state(model, data, args.object_body)
        support_before, support_contacts_before = real_groups_from_contacts(
            st_before, "support", args.support_freeze_dist
        )

        for g in H.FINGER_GROUP_TO_JOINTS:
            if frozen[g]:
                continue

            if g in support_before:
                frozen[g] = True
                freeze_reason[g] = {
                    "reason": "support_real_contact_freeze",
                    "step": k,
                    "contacts": [c for c in support_contacts_before if c.get("group") == g],
                }
                continue

            group_alpha[g] = min(
                1.0,
                float(group_alpha[g]) + 1.0 / max(1, args.close_steps)
            )

        hand_ctrl = make_hand_ctrl(side_open, close_ctrl, group_alpha)
        step_ctrl(model, data, q_grasp, hand_ctrl)

        st = H.contact_state(model, data, args.object_body)
        obj_groups, obj_contacts = real_groups_from_contacts(
            st, "object", args.object_ready_dist
        )
        support_groups, support_contacts = real_groups_from_contacts(
            st, "support", args.support_freeze_dist
        )

        ready = ready_from_groups(obj_groups)
        stable_ready = stable_ready + 1 if ready else 0
        max_stable_ready = max(max_stable_ready, stable_ready)

        obj_pos = H.object_pos(model, data, args.object_body)
        close_disp = float(np.linalg.norm(obj_pos - object_start_for_close))
        max_close_disp = max(max_close_disp, close_disp)

        if len(obj_groups) > len(max_obj_groups):
            max_obj_groups = dict(obj_groups)
        if len(support_groups) > len(max_support_groups):
            max_support_groups = dict(support_groups)

        final_obj_groups = obj_groups
        final_support_groups = support_groups

        if k % args.log_every == 0 or k == args.close_steps - 1 or stable_ready >= args.ready_stable_steps:
            close_rows.append({
                "step": k,
                "ready": bool(ready),
                "stable_ready": int(stable_ready),
                "close_disp": close_disp,
                "object_groups": obj_groups,
                "support_groups": support_groups,
                "group_alpha": dict(group_alpha),
                "frozen": dict(frozen),
            })

        if stable_ready >= args.ready_stable_steps:
            break

        if close_disp > args.abort_close_disp:
            break

    # post hold
    post_stable = 0
    max_post_stable = 0
    for _ in range(args.post_steps):
        step_ctrl(model, data, q_grasp, hand_ctrl)

        st = H.contact_state(model, data, args.object_body)
        obj_groups, _ = real_groups_from_contacts(st, "object", args.object_ready_dist)
        support_groups, _ = real_groups_from_contacts(st, "support", args.support_freeze_dist)

        ready = ready_from_groups(obj_groups)
        post_stable = post_stable + 1 if ready else 0
        max_post_stable = max(max_post_stable, post_stable)

        obj_pos = H.object_pos(model, data, args.object_body)
        close_disp = float(np.linalg.norm(obj_pos - object_start_for_close))
        max_close_disp = max(max_close_disp, close_disp)

        if len(obj_groups) > len(max_obj_groups):
            max_obj_groups = dict(obj_groups)
        if len(support_groups) > len(max_support_groups):
            max_support_groups = dict(support_groups)

        final_obj_groups = obj_groups
        final_support_groups = support_groups

    projection_ready = (
        max_stable_ready >= args.ready_stable_steps
        or max_post_stable >= args.ready_stable_steps
        or ready_from_groups(final_obj_groups)
    )

    setup_ok = exact_setup_disp <= args.max_exact_setup_disp
    close_ok = max_close_disp <= args.max_close_disp
    projection_success = bool(projection_ready and setup_ok and close_ok)

    n_obj_groups = len(max_obj_groups)
    n_frozen = sum(1 for v in frozen.values() if v)

    # 分数：优先 ready/success，其次少推物体、小 dz、接触组数量
    score = 0.0
    if projection_ready:
        score += 1000.0
    if projection_success:
        score += 600.0

    score += 100.0 * n_obj_groups
    score += 10.0 * max(max_stable_ready, max_post_stable)
    score += 0.05 * float(candidate.get("score", 0.0))

    score -= 1800.0 * max(0.0, exact_setup_disp - args.max_exact_setup_disp)
    score -= 1500.0 * max(0.0, max_close_disp - args.max_close_disp)
    score -= 250.0 * float(dz)
    score -= 15.0 * n_frozen

    if not setup_ok:
        failure_reason = "exact_setup_push_too_large"
    elif not projection_ready:
        failure_reason = "not_ready"
    elif not close_ok:
        failure_reason = "close_disp_too_large"
    else:
        failure_reason = ""

    row.update({
        "projection_ready": bool(projection_ready),
        "projection_success": bool(projection_success),
        "projection_score": float(score),
        "failure_reason": failure_reason,

        "exact_setup_disp": float(exact_setup_disp),
        "exact_setup_ok": bool(setup_ok),
        "setup_site_pos_err": float(setup_site_pos_err),
        "setup_site_rot_err": float(setup_site_rot_err),

        "open_object_groups": open_obj_groups,
        "open_support_groups": open_support_groups,

        "max_close_disp": float(max_close_disp),
        "close_ok": bool(close_ok),
        "max_stable_ready": int(max_stable_ready),
        "max_post_stable": int(max_post_stable),
        "max_obj_groups": max_obj_groups,
        "max_support_groups": max_support_groups,
        "final_object_groups": final_obj_groups,
        "final_support_groups": final_support_groups,

        "group_alpha": group_alpha,
        "frozen": frozen,
        "freeze_reason": freeze_reason,
        "final_hand_ctrl": hand_ctrl,
        "object_before_setup": object_before_setup.tolist(),
        "object_after_setup": object_after_setup.tolist(),
        "object_start_for_close": object_start_for_close.tolist(),
        "close_rows": close_rows,
    })

    return row


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--npy", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--object-body", default="grasp_object")
    ap.add_argument("--target-site", default="dataset_hand_base_debug")
    ap.add_argument("--max-candidates", type=int, default=0)

    ap.add_argument("--dz-list", default="0,0.004,0.008,0.012,0.016,0.020,0.024,0.028,0.032,0.036")
    ap.add_argument("--settle-steps", type=int, default=800)
    ap.add_argument("--exact-setup-steps", type=int, default=80)
    ap.add_argument("--close-steps", type=int, default=420)
    ap.add_argument("--post-steps", type=int, default=100)

    ap.add_argument("--object-ready-dist", type=float, default=0.0015)
    ap.add_argument("--support-freeze-dist", type=float, default=0.0)
    ap.add_argument("--ready-stable-steps", type=int, default=5)

    ap.add_argument("--max-exact-setup-disp", type=float, default=0.012)
    ap.add_argument("--max-close-disp", type=float, default=0.025)
    ap.add_argument("--abort-close-disp", type=float, default=0.040)

    ap.add_argument("--ik-iters", type=int, default=350)
    ap.add_argument("--log-every", type=int, default=80)
    args = ap.parse_args()

    args.model = str(resolve(args.model))
    args.npy = str(resolve(args.npy))
    args.candidates = str(resolve(args.candidates))
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_json(args.candidates)
    if args.max_candidates and args.max_candidates > 0:
        candidates = candidates[:args.max_candidates]

    dz_values = [float(x) for x in args.dz_list.split(",") if x.strip()]

    print("========== V4.23 SUPPORT-NORMAL GRASP PROJECTION ==========")
    print("model     :", rel(args.model))
    print("npy       :", rel(args.npy))
    print("candidates:", rel(args.candidates))
    print("num cand  :", len(candidates))
    print("dz_values :", dz_values)
    print("out_dir   :", rel(out_dir))

    rows = []

    for ci, cand in enumerate(candidates, start=1):
        print(
            f"\n[CAND] {ci}/{len(candidates)} "
            f"local={cand.get('valid_local_index')} raw={cand.get('raw_sample_index')} "
            f"type={cand.get('grasp_type')} selector_score={cand.get('score')}"
        )

        for dz in dz_values:
            r = eval_one_projection(args, cand, dz, ci)
            rows.append(r)

            print(
                f"[trial] local={r['valid_local_index']:03d} dz={dz:+.4f} "
                f"ready={r.get('projection_ready')} success={r.get('projection_success')} "
                f"score={r.get('projection_score'):.2f} "
                f"setup_disp={r.get('exact_setup_disp',999):.4f} "
                f"close_disp={r.get('max_close_disp',999):.4f} "
                f"obj={r.get('max_obj_groups')} "
                f"frozen={r.get('frozen')} "
                f"reason={r.get('failure_reason')}"
            )

    rows_sorted = sorted(rows, key=lambda x: x.get("projection_score", -1e9), reverse=True)
    best = rows_sorted[0] if rows_sorted else {}

    save_json(out_dir / "projected_candidates_ranked.json", rows_sorted)
    save_json(out_dir / "best_projected_candidate.json", best)

    lines = []
    lines.append("========== V4.23 PROJECTION REPORT ==========")
    lines.append(f"model     : {rel(args.model)}")
    lines.append(f"candidates: {rel(args.candidates)}")
    lines.append(f"num trials: {len(rows_sorted)}")
    lines.append("")
    lines.append("---- ranked top 20 ----")

    for rank, r in enumerate(rows_sorted[:20], start=1):
        lines.append(
            f"rank={rank:02d} "
            f"local={r['valid_local_index']:03d} raw={r['raw_sample_index']:03d} "
            f"type={r.get('selector_type')} dz={r.get('dz'):+.4f} "
            f"score={r.get('projection_score',0):.2f} "
            f"ready={r.get('projection_ready')} success={r.get('projection_success')} "
            f"setup_disp={r.get('exact_setup_disp',999):.4f} "
            f"close_disp={r.get('max_close_disp',999):.4f} "
            f"site_err={r.get('setup_site_pos_err',999):.4f} "
            f"obj={r.get('max_obj_groups')} "
            f"support={r.get('max_support_groups')} "
            f"frozen={r.get('frozen')} "
            f"reason={r.get('failure_reason')}"
        )

    if best:
        lines.append("")
        lines.append("---- selected best projected candidate ----")
        lines.append(f"local={best.get('valid_local_index')} raw={best.get('raw_sample_index')}")
        lines.append(f"type={best.get('selector_type')}")
        lines.append(f"dz={best.get('dz')}")
        lines.append(f"projection_ready={best.get('projection_ready')}")
        lines.append(f"projection_success={best.get('projection_success')}")
        lines.append(f"projection_score={best.get('projection_score')}")
        lines.append(f"exact_setup_disp={best.get('exact_setup_disp')}")
        lines.append(f"max_close_disp={best.get('max_close_disp')}")
        lines.append(f"max_obj_groups={best.get('max_obj_groups')}")
        lines.append(f"frozen={best.get('frozen')}")

    lines.append("")
    lines.append("---- output ----")
    lines.append(f"ranked: {rel(out_dir / 'projected_candidates_ranked.json')}")
    lines.append(f"best  : {rel(out_dir / 'best_projected_candidate.json')}")
    lines.append("===========================================")

    report = "\n".join(lines) + "\n"
    (out_dir / "projection_report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
