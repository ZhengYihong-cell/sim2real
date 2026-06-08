#!/usr/bin/env python3
"""
V4.21 Top-K short dynamic validation.

定位：
    对 V4.20d selector 选出的 feasible Top-K 候选做短动态验证。
    不是最终执行器，不做完整 lift，不救单个 sample。

输入：
    selected_topk_feasible.json

每个候选流程：
    1. 根据当前 scene 中 object pose 得到 T_world_object；
    2. 从 npy 读取 T_object_hand 和 O7 close ctrl；
    3. T_world_hand = T_world_object @ T_object_hand；
    4. site IK 到 dataset_hand_base_debug；
    5. 平滑运动到 q_grasp；
    6. 在 q_grasp 上短 close；
    7. 真实支撑接触的手指冻结；
    8. 统计 thumb + nonthumb、object displacement、support contact；
    9. 输出 ranking。

成功标准：
    short_ready = thumb + 至少一根非拇指稳定接触；
    object displacement 不超过阈值；
    open/move 阶段没有严重推物体。

输出：
    short_dynamic_results.json
    short_dynamic_report.txt
    selected_best_candidate.json
"""

from pathlib import Path
import argparse
import importlib.util
import json
import time
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


def load_candidates(path, max_candidates=0):
    rows = json.loads(Path(path).read_text())
    if max_candidates and max_candidates > 0:
        rows = rows[:max_candidates]
    return rows


def step_ctrl(model, data, q_arm, hand_ctrl):
    H.apply_ctrl(model, data, q_arm, hand_ctrl)
    mujoco.mj_step(model, data)


def real_groups_from_contacts(state, key, dist_th):
    groups = {}
    contacts = []

    if key == "object":
        raw = state.get("object_contacts", [])
    else:
        raw = state.get("support_contacts", [])

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


def side_open_from_close(close_ctrl):
    side = dict(close_ctrl)
    for j in ["index_mcp_pitch", "middle_mcp_pitch", "ring_mcp_pitch", "pinky_mcp_pitch"]:
        side[j] = 0.0
    return side


def current_site_error(model, data, site_name, T_target):
    T_cur = H.site_world_T(model, data, site_name)
    _, _, pos_n, rot_n = H.pose_error(T_cur, T_target)
    return float(pos_n), float(rot_n)


def make_hand_ctrl_from_alpha(side_open, close_ctrl, group_alpha):
    ctrl = dict(side_open)
    for g, joints in H.FINGER_GROUP_TO_JOINTS.items():
        a = float(group_alpha.get(g, 0.0))
        for j in joints:
            v0 = float(side_open.get(j, close_ctrl.get(j, 0.0)))
            v1 = float(close_ctrl.get(j, v0))
            ctrl[j] = (1.0 - a) * v0 + a * v1
    return ctrl


def validate_one_candidate(args, candidate, rank):
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)

    local_idx = int(candidate["valid_local_index"])
    raw_idx = int(candidate.get("raw_sample_index", local_idx))

    sample = H.load_sample(args.npy, local_idx)
    T_object_hand = H.sample_T_object_hand(sample)
    close_ctrl = H.sample_ctrl(sample)
    side_open = side_open_from_close(close_ctrl)

    # 初始 home + open，动态 settle 物体
    H.set_qpos_once(model, data, H.Q_HOME, side_open)

    for _ in range(args.settle_steps):
        step_ctrl(model, data, H.Q_HOME, side_open)

    object_start = H.object_pos(model, data, args.object_body)
    T_world_object = H.body_world_T(model, data, args.object_body)

    T_grasp = T_world_object @ T_object_hand

    ik_grasp = H.solve_site_ik(
        model,
        args.target_site,
        T_grasp,
        H.Q_HOME,
        max_iters=args.ik_iters,
    )

    result = {
        "rank_from_selector": int(rank),
        "valid_local_index": local_idx,
        "raw_sample_index": raw_idx,
        "selector_score": float(candidate.get("score", 0.0)),
        "selector_type": candidate.get("grasp_type", ""),
        "selector_ready_close": bool(candidate.get("proxy_ready_close", False)),
        "selector_open_support_min": candidate.get("open_support_min"),
        "selector_close_support_min": candidate.get("close_support_min"),
        "selector_contact_score": candidate.get("contact_score"),
        "object_start": object_start.tolist(),
        "T_world_object_after_settle": H.mat_to_dict(T_world_object),
        "T_grasp": H.mat_to_dict(T_grasp),
        "ik_grasp": ik_grasp,
    }

    if not ik_grasp["success"]:
        result.update({
            "short_ready": False,
            "short_success": False,
            "failure_reason": "ik_failed",
            "short_score": -1e6,
        })
        return result

    q_grasp = ik_grasp["q_arm"]
    q_current = H.get_joint_values(model, data, H.ARM_JOINTS)

    # 平滑运动到 q_grasp，手保持 open
    move_rows = []
    max_move_object_disp = 0.0
    max_move_support_real = {}
    max_move_object_groups = {}

    for k in range(args.move_steps):
        alpha = k / max(1, args.move_steps - 1)
        q_cmd = H.interp_dict(q_current, q_grasp, alpha, H.ARM_JOINTS)

        step_ctrl(model, data, q_cmd, side_open)

        st = H.contact_state(model, data, args.object_body)
        obj_groups, _ = real_groups_from_contacts(st, "object", args.object_ready_dist)
        sup_groups, _ = real_groups_from_contacts(st, "support", args.support_freeze_dist)

        obj_pos = H.object_pos(model, data, args.object_body)
        disp = float(np.linalg.norm(obj_pos - object_start))
        max_move_object_disp = max(max_move_object_disp, disp)

        if len(obj_groups) > len(max_move_object_groups):
            max_move_object_groups = dict(obj_groups)
        if len(sup_groups) > len(max_move_support_real):
            max_move_support_real = dict(sup_groups)

        if k % args.log_every == 0 or k == args.move_steps - 1:
            move_rows.append({
                "step": k,
                "alpha": alpha,
                "object_disp": disp,
                "object_groups": obj_groups,
                "support_real_groups": sup_groups,
            })

    move_site_pos_err, move_site_rot_err = current_site_error(
        model, data, args.target_site, T_grasp
    )

    exact_setup_disp = 0.0
    exact_setup_site_pos_err = move_site_pos_err
    exact_setup_site_rot_err = move_site_rot_err

    if args.snap_to_qgrasp_for_short_validation:
        # V4.21b:
        # 这里只做一次 qpos 设置，用于短验证候选本身；
        # 后续 close/post 仍然是真实动力学 step，不再每步硬写 qpos。
        object_before_exact_setup = H.object_pos(model, data, args.object_body).copy()

        H.set_qpos_once(model, data, q_grasp, side_open)

        for _ in range(args.exact_grasp_settle_steps):
            step_ctrl(model, data, q_grasp, side_open)

        object_after_exact_setup = H.object_pos(model, data, args.object_body).copy()
        exact_setup_disp = float(np.linalg.norm(object_after_exact_setup - object_before_exact_setup))

        exact_setup_site_pos_err, exact_setup_site_rot_err = current_site_error(
            model, data, args.target_site, T_grasp
        )

        # 后续 close/post 的位移从 exact-qgrasp setup 后重新计，避免 arm tracking 污染候选短验证。
        object_start = object_after_exact_setup.copy()

    result["exact_qgrasp_setup_enabled"] = bool(args.snap_to_qgrasp_for_short_validation)
    result["exact_setup_disp"] = float(exact_setup_disp)
    result["exact_setup_site_pos_err"] = float(exact_setup_site_pos_err)
    result["exact_setup_site_rot_err"] = float(exact_setup_site_rot_err)
    result["object_start_for_close"] = object_start.tolist()

    # q_grasp open hold
    hold_rows = []
    hold_object_disp = 0.0
    for k in range(args.hold_steps):
        step_ctrl(model, data, q_grasp, side_open)

        if k % args.log_every == 0 or k == args.hold_steps - 1:
            st = H.contact_state(model, data, args.object_body)
            obj_groups, _ = real_groups_from_contacts(st, "object", args.object_ready_dist)
            sup_groups, _ = real_groups_from_contacts(st, "support", args.support_freeze_dist)
            obj_pos = H.object_pos(model, data, args.object_body)
            disp = float(np.linalg.norm(obj_pos - object_start))
            hold_object_disp = max(hold_object_disp, disp)
            hold_rows.append({
                "step": k,
                "object_disp": disp,
                "object_groups": obj_groups,
                "support_real_groups": sup_groups,
            })

    # support-aware short close
    group_alpha = {g: 0.0 for g in H.FINGER_GROUP_TO_JOINTS}
    frozen = {g: False for g in H.FINGER_GROUP_TO_JOINTS}
    freeze_reason = {}

    stable_ready = 0
    max_stable_ready = 0
    max_close_object_disp = 0.0
    max_close_object_groups = {}
    max_close_support_real = {}
    close_rows = []
    last_hand_ctrl = dict(side_open)

    for k in range(args.close_steps):
        st_before = H.contact_state(model, data, args.object_body)
        support_real_before, support_contacts_before = real_groups_from_contacts(
            st_before, "support", args.support_freeze_dist
        )

        for g in H.FINGER_GROUP_TO_JOINTS:
            if frozen[g]:
                continue

            if g in support_real_before:
                frozen[g] = True
                freeze_reason[g] = {
                    "reason": "support_real_contact_freeze",
                    "step": k,
                    "support_freeze_dist": args.support_freeze_dist,
                    "contacts": [c for c in support_contacts_before if c.get("group") == g],
                }
                continue

            group_alpha[g] = min(
                1.0,
                float(group_alpha[g]) + 1.0 / max(1, args.close_steps)
            )

        last_hand_ctrl = make_hand_ctrl_from_alpha(side_open, close_ctrl, group_alpha)

        step_ctrl(model, data, q_grasp, last_hand_ctrl)

        st = H.contact_state(model, data, args.object_body)
        obj_groups, obj_contacts = real_groups_from_contacts(
            st, "object", args.object_ready_dist
        )
        sup_groups, sup_contacts = real_groups_from_contacts(
            st, "support", args.support_freeze_dist
        )

        ready = ready_from_groups(obj_groups)
        stable_ready = stable_ready + 1 if ready else 0
        max_stable_ready = max(max_stable_ready, stable_ready)

        obj_pos = H.object_pos(model, data, args.object_body)
        disp = float(np.linalg.norm(obj_pos - object_start))
        max_close_object_disp = max(max_close_object_disp, disp)

        if len(obj_groups) > len(max_close_object_groups):
            max_close_object_groups = dict(obj_groups)
        if len(sup_groups) > len(max_close_support_real):
            max_close_support_real = dict(sup_groups)

        if k % args.log_every == 0 or k == args.close_steps - 1 or stable_ready >= args.ready_stable_steps:
            close_rows.append({
                "step": k,
                "object_disp": disp,
                "object_groups": obj_groups,
                "support_real_groups": sup_groups,
                "ready": bool(ready),
                "stable_ready": int(stable_ready),
                "group_alpha": dict(group_alpha),
                "frozen": dict(frozen),
            })

        if stable_ready >= args.ready_stable_steps:
            break

        if disp > args.abort_object_disp:
            break

    # post hold，确认 ready 是否稳定
    post_rows = []
    post_stable = 0
    max_post_stable = 0
    for k in range(args.post_steps):
        step_ctrl(model, data, q_grasp, last_hand_ctrl)

        st = H.contact_state(model, data, args.object_body)
        obj_groups, _ = real_groups_from_contacts(st, "object", args.object_ready_dist)
        sup_groups, _ = real_groups_from_contacts(st, "support", args.support_freeze_dist)
        ready = ready_from_groups(obj_groups)
        post_stable = post_stable + 1 if ready else 0
        max_post_stable = max(max_post_stable, post_stable)

        obj_pos = H.object_pos(model, data, args.object_body)
        disp = float(np.linalg.norm(obj_pos - object_start))
        max_close_object_disp = max(max_close_object_disp, disp)

        if k % args.log_every == 0 or k == args.post_steps - 1:
            post_rows.append({
                "step": k,
                "object_disp": disp,
                "object_groups": obj_groups,
                "support_real_groups": sup_groups,
                "ready": bool(ready),
                "post_stable": int(post_stable),
            })

    final_state = H.contact_state(model, data, args.object_body)
    final_obj_groups, final_obj_contacts = real_groups_from_contacts(
        final_state, "object", args.object_ready_dist
    )
    final_sup_groups, final_sup_contacts = real_groups_from_contacts(
        final_state, "support", args.support_freeze_dist
    )

    final_pos = H.object_pos(model, data, args.object_body)
    final_disp = float(np.linalg.norm(final_pos - object_start))

    short_ready = (
        max_stable_ready >= args.ready_stable_steps
        or max_post_stable >= args.ready_stable_steps
        or ready_from_groups(final_obj_groups)
    )

    displacement_ok = final_disp <= args.max_final_disp
    move_ok = True if args.snap_to_qgrasp_for_short_validation else (max_move_object_disp <= args.max_move_disp)
    exact_setup_ok = exact_setup_disp <= args.max_exact_setup_disp
    short_success = bool(short_ready and displacement_ok and move_ok and exact_setup_ok)

    # 排序分：短验证优先，selector 分只作为轻微 tie-break
    n_obj_groups = len(max_close_object_groups)
    n_frozen = sum(1 for v in frozen.values() if v)

    short_score = 0.0
    if short_ready:
        short_score += 1000.0
    if short_success:
        short_score += 500.0

    short_score += 80.0 * n_obj_groups
    short_score += 10.0 * max(max_stable_ready, max_post_stable)
    short_score += 0.05 * float(candidate.get("score", 0.0))

    short_score -= 1200.0 * max(0.0, final_disp - args.max_final_disp)
    short_score -= 1200.0 * max(0.0, max_move_object_disp - args.max_move_disp)
    short_score -= 20.0 * n_frozen

    result.update({
        "short_ready": bool(short_ready),
        "short_success": bool(short_success),
        "failure_reason": "" if short_success else (
            "not_ready" if not short_ready else
            "final_disp_too_large" if not displacement_ok else
            "exact_setup_push_too_large" if not exact_setup_ok else
            "move_push_too_large"
        ),
        "move_ok_for_short_validation": bool(move_ok),
        "exact_setup_ok": bool(exact_setup_ok),
        "short_score": float(short_score),

        "move_site_pos_err": float(move_site_pos_err),
        "move_site_rot_err": float(move_site_rot_err),
        "max_move_object_disp": float(max_move_object_disp),
        "hold_object_disp": float(hold_object_disp),
        "max_close_object_disp": float(max_close_object_disp),
        "final_object_disp": float(final_disp),
        "final_object_pos": final_pos.tolist(),

        "max_stable_ready": int(max_stable_ready),
        "max_post_stable": int(max_post_stable),
        "max_close_object_groups": max_close_object_groups,
        "max_close_support_real_groups": max_close_support_real,
        "final_object_groups": final_obj_groups,
        "final_support_real_groups": final_sup_groups,
        "final_object_contacts": final_obj_contacts,
        "final_support_contacts": final_sup_contacts,

        "group_alpha": group_alpha,
        "frozen": frozen,
        "freeze_reason": freeze_reason,
        "final_hand_ctrl": last_hand_ctrl,

        "move_rows": move_rows,
        "hold_rows": hold_rows,
        "close_rows": close_rows,
        "post_rows": post_rows,
    })

    return result


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--npy", required=True)
    ap.add_argument("--candidates", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--object-body", default="grasp_object")
    ap.add_argument("--target-site", default="dataset_hand_base_debug")
    ap.add_argument("--max-candidates", type=int, default=0)

    ap.add_argument("--settle-steps", type=int, default=800)
    ap.add_argument("--move-steps", type=int, default=900)
    ap.add_argument("--hold-steps", type=int, default=120)
    ap.add_argument("--close-steps", type=int, default=450)
    ap.add_argument("--post-steps", type=int, default=120)

    ap.add_argument("--object-ready-dist", type=float, default=0.0015)
    ap.add_argument("--support-freeze-dist", type=float, default=0.0)
    ap.add_argument("--ready-stable-steps", type=int, default=5)

    ap.add_argument("--max-move-disp", type=float, default=0.018)
    ap.add_argument("--max-final-disp", type=float, default=0.025)
    ap.add_argument("--abort-object-disp", type=float, default=0.040)

    ap.add_argument("--snap-to-qgrasp-for-short-validation", action="store_true",
                    help="V4.21b: one-time set qpos to exact q_grasp before close, so short validation tests grasp candidate rather than arm tracking")
    ap.add_argument("--exact-grasp-settle-steps", type=int, default=80)
    ap.add_argument("--max-exact-setup-disp", type=float, default=0.020)

    ap.add_argument("--ik-iters", type=int, default=350)
    ap.add_argument("--log-every", type=int, default=75)
    args = ap.parse_args()

    args.model = str(resolve(args.model))
    args.npy = str(resolve(args.npy))
    args.candidates = str(resolve(args.candidates))
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    candidates = load_candidates(args.candidates, args.max_candidates)

    print("========== V4.21 TOP-K SHORT DYNAMIC VALIDATION ==========")
    print("model     :", rel(args.model))
    print("npy       :", rel(args.npy))
    print("candidates:", rel(args.candidates))
    print("num       :", len(candidates))
    print("out_dir   :", rel(out_dir))

    results = []
    for i, cand in enumerate(candidates, start=1):
        print(
            f"\n[RUN] {i}/{len(candidates)} "
            f"local={cand.get('valid_local_index')} "
            f"raw={cand.get('raw_sample_index')} "
            f"type={cand.get('grasp_type')} "
            f"selector_score={cand.get('score')}"
        )

        r = validate_one_candidate(args, cand, i)
        results.append(r)

        print(
            f"[DONE] local={r['valid_local_index']} "
            f"short_ready={r.get('short_ready')} "
            f"short_success={r.get('short_success')} "
            f"score={r.get('short_score'):.2f} "
            f"disp={r.get('final_object_disp', 999):.4f} "
            f"move_disp={r.get('max_move_object_disp', 999):.4f} "
            f"obj={r.get('max_close_object_groups')} "
            f"frozen={r.get('frozen')} "
            f"reason={r.get('failure_reason')}"
        )

    results_sorted = sorted(results, key=lambda x: x.get("short_score", -1e9), reverse=True)
    best = results_sorted[0] if results_sorted else None

    save_json(out_dir / "short_dynamic_results.json", results_sorted)
    save_json(out_dir / "selected_best_candidate.json", best if best is not None else {})

    lines = []
    lines.append("========== V4.21 SHORT DYNAMIC VALIDATION REPORT ==========")
    lines.append(f"model     : {rel(args.model)}")
    lines.append(f"candidates: {rel(args.candidates)}")
    lines.append(f"num       : {len(results_sorted)}")
    lines.append("")
    lines.append("---- ranked results ----")

    for rank, r in enumerate(results_sorted, start=1):
        lines.append(
            f"rank={rank:02d} "
            f"local={r['valid_local_index']:03d} raw={r['raw_sample_index']:03d} "
            f"type={r.get('selector_type')} "
            f"short_score={r.get('short_score', 0):.2f} "
            f"ready={r.get('short_ready')} success={r.get('short_success')} "
            f"selector={r.get('selector_score'):.2f} "
            f"move_disp={r.get('max_move_object_disp', 999):.4f} "
            f"final_disp={r.get('final_object_disp', 999):.4f} "
            f"site_err={r.get('move_site_pos_err', 999):.4f} "
            f"exact_site_err={r.get('exact_setup_site_pos_err', 999):.4f} "
            f"exact_disp={r.get('exact_setup_disp', 0):.4f} "
            f"obj={r.get('max_close_object_groups')} "
            f"support={r.get('max_close_support_real_groups')} "
            f"frozen={r.get('frozen')} "
            f"reason={r.get('failure_reason')}"
        )

    if best:
        lines.append("")
        lines.append("---- selected best ----")
        lines.append(f"local={best['valid_local_index']} raw={best['raw_sample_index']}")
        lines.append(f"type={best.get('selector_type')}")
        lines.append(f"short_ready={best.get('short_ready')}")
        lines.append(f"short_success={best.get('short_success')}")
        lines.append(f"short_score={best.get('short_score')}")
        lines.append(f"max_close_object_groups={best.get('max_close_object_groups')}")
        lines.append(f"frozen={best.get('frozen')}")
        lines.append(f"final_hand_ctrl={best.get('final_hand_ctrl')}")

    lines.append("")
    lines.append("---- output ----")
    lines.append(f"results: {rel(out_dir / 'short_dynamic_results.json')}")
    lines.append(f"best   : {rel(out_dir / 'selected_best_candidate.json')}")
    lines.append("==========================================================")

    report = "\n".join(lines) + "\n"
    (out_dir / "short_dynamic_report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
