#!/usr/bin/env python3
"""
V4.26 integrated pre-lift validation.

定位：
    对 V4.23 projected candidates 做真实动态 pre-lift 验证。
    不是最终 lift runner。
    目的：自动选出 move/hold/close 都能通过的 best projected candidate。

为什么需要它：
    V4.24 只看 exact short close，可能漏掉动态路径推物体；
    V4.25 只看 open move/hold，可能漏掉 close 阶段把物体碰掉；
    V4.26 把 move + hold + close + post-close 放到同一个真实动态流程里评估。

输入：
    V4.23 projected_candidates_ranked.json

输出：
    integrated_candidates_ranked.json
    best_integrated_projected_candidate.json
    integrated_pre_lift_report.txt

验证流程：
    1. scene settle；
    2. 根据当前 object pose + sample + dz 重算 T_grasp_projected；
    3. site IK；
    4. dynamic smooth move 到 q_grasp，hand open；
    5. servo hold at q_grasp，hand open；
    6. support-aware dynamic close；
    7. post-close hold；
    8. 判断 grip_ready、object displacement、first_ready_step、first_ready_disp；
    9. 自动选择 best。
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


def step_ctrl(model, data, q_arm, hand_ctrl):
    H.apply_ctrl(model, data, q_arm, hand_ctrl)
    mujoco.mj_step(model, data)


def current_site_error(model, data, site_name, T_target):
    T_cur = H.site_world_T(model, data, site_name)
    _, _, pos_n, rot_n = H.pose_error(T_cur, T_target)
    return float(pos_n), float(rot_n)


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


def recompute_projected_T_grasp(row, T_world_object, T_object_hand):
    """
    不直接信旧 JSON 里的绝对 T_grasp_projected。
    每次根据当前 scene settle 后的 object pose 重算：
        T_grasp = T_world_object @ T_object_hand + dz * world_z
    """
    T = T_world_object @ T_object_hand
    dz = float(row.get("dz", 0.0))
    T[:3, 3] += np.array([0.0, 0.0, dz], dtype=float)
    return T


def validate_candidate(args, row, input_rank):
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)

    local_idx = int(row["valid_local_index"])
    raw_idx = int(row.get("raw_sample_index", local_idx))

    sample = H.load_sample(args.npy, local_idx)
    T_object_hand = H.sample_T_object_hand(sample)
    close_ctrl = H.sample_ctrl(sample)
    side_open = side_open_from_close(close_ctrl)

    H.set_qpos_once(model, data, H.Q_HOME, side_open)

    for _ in range(args.settle_steps):
        step_ctrl(model, data, H.Q_HOME, side_open)

    object_start = H.object_pos(model, data, args.object_body).copy()
    T_world_object = H.body_world_T(model, data, args.object_body)

    T_grasp = recompute_projected_T_grasp(row, T_world_object, T_object_hand)

    ik = H.solve_site_ik(
        model,
        args.target_site,
        T_grasp,
        H.Q_HOME,
        max_iters=args.ik_iters,
    )

    out = dict(row)
    out["integrated_input_rank"] = int(input_rank)
    out["integrated_local_index"] = local_idx
    out["integrated_raw_index"] = raw_idx
    out["T_world_object_integrated"] = H.mat_to_dict(T_world_object)
    out["T_object_hand_integrated"] = H.mat_to_dict(T_object_hand)
    out["T_grasp_projected"] = H.mat_to_dict(T_grasp)
    out["integrated_ik"] = ik

    if not ik["success"]:
        out.update({
            "integrated_ready": False,
            "integrated_success": False,
            "integrated_score": -1e6,
            "integrated_failure_reason": "ik_failed",
        })
        return out

    q_current = H.get_joint_values(model, data, H.ARM_JOINTS)
    q_grasp = ik["q_arm"]

    max_move_disp = 0.0
    max_hold_disp = 0.0
    max_close_disp = 0.0
    max_post_disp = 0.0
    move_aborted = False
    hold_aborted = False
    close_aborted = False

    max_move_obj_groups = {}
    max_hold_obj_groups = {}
    max_close_obj_groups = {}
    max_post_obj_groups = {}

    max_move_support_groups = {}
    max_hold_support_groups = {}
    max_close_support_groups = {}
    max_post_support_groups = {}

    min_hold_site_pos_err = 999.0
    min_hold_site_rot_err = 999.0
    final_move_site_pos_err = 999.0
    final_move_site_rot_err = 999.0

    # 1) dynamic move, open hand
    for k in range(args.move_steps):
        alpha = k / max(1, args.move_steps - 1)
        q_cmd = H.interp_dict(q_current, q_grasp, alpha, H.ARM_JOINTS)
        step_ctrl(model, data, q_cmd, side_open)

        obj_pos = H.object_pos(model, data, args.object_body)
        disp = float(np.linalg.norm(obj_pos - object_start))
        max_move_disp = max(max_move_disp, disp)

        st = H.contact_state(model, data, args.object_body)
        obj_groups, _ = real_groups_from_contacts(st, "object", args.object_ready_dist)
        sup_groups, _ = real_groups_from_contacts(st, "support", args.support_freeze_dist)

        if len(obj_groups) > len(max_move_obj_groups):
            max_move_obj_groups = dict(obj_groups)
        if len(sup_groups) > len(max_move_support_groups):
            max_move_support_groups = dict(sup_groups)

        if disp > args.abort_move_disp:
            move_aborted = True
            break

    final_move_site_pos_err, final_move_site_rot_err = current_site_error(
        model, data, args.target_site, T_grasp
    )

    # 2) servo hold at q_grasp, open hand
    if not move_aborted:
        for k in range(args.hold_steps):
            step_ctrl(model, data, q_grasp, side_open)

            obj_pos = H.object_pos(model, data, args.object_body)
            disp = float(np.linalg.norm(obj_pos - object_start))
            max_hold_disp = max(max_hold_disp, disp)

            st = H.contact_state(model, data, args.object_body)
            obj_groups, _ = real_groups_from_contacts(st, "object", args.object_ready_dist)
            sup_groups, _ = real_groups_from_contacts(st, "support", args.support_freeze_dist)

            if len(obj_groups) > len(max_hold_obj_groups):
                max_hold_obj_groups = dict(obj_groups)
            if len(sup_groups) > len(max_hold_support_groups):
                max_hold_support_groups = dict(sup_groups)

            site_pos_err, site_rot_err = current_site_error(model, data, args.target_site, T_grasp)
            min_hold_site_pos_err = min(min_hold_site_pos_err, site_pos_err)
            min_hold_site_rot_err = min(min_hold_site_rot_err, site_rot_err)

            if disp > args.abort_hold_disp:
                hold_aborted = True
                break

    site_ready = (
        min_hold_site_pos_err <= args.site_ready_pos_err
        and min_hold_site_rot_err <= args.site_ready_rot_err
    )

    # 3) support-aware close, same state, no reset/snap
    group_alpha = {g: 0.0 for g in H.FINGER_GROUP_TO_JOINTS}
    frozen = {g: False for g in H.FINGER_GROUP_TO_JOINTS}
    freeze_reason = {}
    last_hand_ctrl = dict(side_open)

    stable_ready = 0
    max_stable_ready = 0
    first_ready_event = None
    final_obj_groups = {}
    final_support_groups = {}

    if not move_aborted and not hold_aborted:
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

            last_hand_ctrl = make_hand_ctrl(side_open, close_ctrl, group_alpha)
            step_ctrl(model, data, q_grasp, last_hand_ctrl)

            st = H.contact_state(model, data, args.object_body)
            obj_groups, obj_contacts = real_groups_from_contacts(
                st, "object", args.object_ready_dist
            )
            sup_groups, sup_contacts = real_groups_from_contacts(
                st, "support", args.support_freeze_dist
            )

            obj_pos = H.object_pos(model, data, args.object_body)
            disp = float(np.linalg.norm(obj_pos - object_start))
            max_close_disp = max(max_close_disp, disp)

            if len(obj_groups) > len(max_close_obj_groups):
                max_close_obj_groups = dict(obj_groups)
            if len(sup_groups) > len(max_close_support_groups):
                max_close_support_groups = dict(sup_groups)

            ready = ready_from_groups(obj_groups)
            stable_ready = stable_ready + 1 if ready else 0
            max_stable_ready = max(max_stable_ready, stable_ready)

            if first_ready_event is None and ready:
                first_ready_event = {
                    "phase": "close",
                    "step": int(k),
                    "step_ratio": float(k / max(1, args.close_steps)),
                    "object_disp": float(disp),
                    "object_groups": obj_groups,
                    "support_groups": sup_groups,
                    "group_alpha": dict(group_alpha),
                    "frozen": dict(frozen),
                }

            final_obj_groups = obj_groups
            final_support_groups = sup_groups

            if disp > args.abort_close_disp:
                close_aborted = True
                break

            if stable_ready >= args.ready_stable_steps and args.stop_close_on_ready:
                break

    # 4) post-close hold
    post_stable = 0
    max_post_stable = 0

    if not move_aborted and not hold_aborted and not close_aborted:
        for k in range(args.post_steps):
            step_ctrl(model, data, q_grasp, last_hand_ctrl)

            st = H.contact_state(model, data, args.object_body)
            obj_groups, _ = real_groups_from_contacts(st, "object", args.object_ready_dist)
            sup_groups, _ = real_groups_from_contacts(st, "support", args.support_freeze_dist)

            obj_pos = H.object_pos(model, data, args.object_body)
            disp = float(np.linalg.norm(obj_pos - object_start))
            max_post_disp = max(max_post_disp, disp)

            if len(obj_groups) > len(max_post_obj_groups):
                max_post_obj_groups = dict(obj_groups)
            if len(sup_groups) > len(max_post_support_groups):
                max_post_support_groups = dict(sup_groups)

            ready = ready_from_groups(obj_groups)
            post_stable = post_stable + 1 if ready else 0
            max_post_stable = max(max_post_stable, post_stable)

            if first_ready_event is None and ready:
                first_ready_event = {
                    "phase": "post_close",
                    "step": int(args.close_steps + k),
                    "step_ratio": float((args.close_steps + k) / max(1, args.close_steps + args.post_steps)),
                    "object_disp": float(disp),
                    "object_groups": obj_groups,
                    "support_groups": sup_groups,
                    "group_alpha": dict(group_alpha),
                    "frozen": dict(frozen),
                }

            final_obj_groups = obj_groups
            final_support_groups = sup_groups

    final_pos = H.object_pos(model, data, args.object_body)
    final_prelift_disp = float(np.linalg.norm(final_pos - object_start))

    grip_ready = (
        max_stable_ready >= args.ready_stable_steps
        or max_post_stable >= args.ready_stable_steps
        or ready_from_groups(final_obj_groups)
    )

    move_ok = max_move_disp <= args.max_move_disp
    hold_ok = max_hold_disp <= args.max_hold_disp
    prelift_ok = final_prelift_disp <= args.max_prelift_disp
    close_ok = max(max_close_disp, max_post_disp, final_prelift_disp) <= args.max_prelift_disp

    integrated_success = bool(
        site_ready
        and grip_ready
        and move_ok
        and hold_ok
        and prelift_ok
        and not move_aborted
        and not hold_aborted
        and not close_aborted
    )

    n_obj_groups = len(max(max_close_obj_groups, max_post_obj_groups, key=len))
    n_frozen = sum(1 for v in frozen.values() if v)
    dz = float(row.get("dz", 0.0))

    if move_aborted:
        reason = "move_push_abort"
    elif hold_aborted:
        reason = "hold_push_abort"
    elif close_aborted:
        reason = "close_push_abort"
    elif not site_ready:
        reason = "site_not_ready"
    elif not grip_ready:
        reason = "grip_not_ready"
    elif not move_ok:
        reason = "move_disp_too_large"
    elif not hold_ok:
        reason = "hold_disp_too_large"
    elif not prelift_ok:
        reason = "prelift_disp_too_large"
    else:
        reason = ""

    old_quality = float(row.get("quality_score", row.get("projection_score", 0.0)))
    old_path = float(row.get("path_quality_score", 0.0))

    score = 0.0
    if integrated_success:
        score += 5000.0
    if grip_ready:
        score += 1200.0
    if site_ready:
        score += 500.0

    score += 120.0 * n_obj_groups
    score += 8.0 * max(max_stable_ready, max_post_stable)

    if first_ready_event is not None:
        score -= args.w_ready_ratio * first_ready_event["step_ratio"]
        score -= args.w_first_ready_disp * max(0.0, first_ready_event["object_disp"] - args.clean_ready_disp)
    else:
        score -= 2000.0

    score -= args.w_move_disp * max(0.0, max_move_disp - args.clean_move_disp)
    score -= args.w_hold_disp * max(0.0, max_hold_disp - args.clean_hold_disp)
    score -= args.w_prelift_disp * max(0.0, final_prelift_disp - args.clean_prelift_disp)
    score -= args.w_dz * dz
    score -= args.w_frozen * n_frozen

    score += args.w_old_quality * old_quality
    score += args.w_old_path * old_path

    out.update({
        "integrated_ready": bool(grip_ready),
        "integrated_success": bool(integrated_success),
        "integrated_score": float(score),
        "integrated_failure_reason": reason,

        "integrated_site_ready": bool(site_ready),
        "integrated_move_ok": bool(move_ok),
        "integrated_hold_ok": bool(hold_ok),
        "integrated_prelift_ok": bool(prelift_ok),
        "integrated_close_ok": bool(close_ok),

        "integrated_move_aborted": bool(move_aborted),
        "integrated_hold_aborted": bool(hold_aborted),
        "integrated_close_aborted": bool(close_aborted),

        "integrated_max_move_disp": float(max_move_disp),
        "integrated_max_hold_disp": float(max_hold_disp),
        "integrated_max_close_disp": float(max_close_disp),
        "integrated_max_post_disp": float(max_post_disp),
        "integrated_final_prelift_disp": float(final_prelift_disp),

        "integrated_final_move_site_pos_err": float(final_move_site_pos_err),
        "integrated_final_move_site_rot_err": float(final_move_site_rot_err),
        "integrated_min_hold_site_pos_err": float(min_hold_site_pos_err),
        "integrated_min_hold_site_rot_err": float(min_hold_site_rot_err),

        "integrated_first_ready_event": first_ready_event,
        "integrated_max_stable_ready": int(max_stable_ready),
        "integrated_max_post_stable": int(max_post_stable),

        "integrated_max_move_obj_groups": max_move_obj_groups,
        "integrated_max_hold_obj_groups": max_hold_obj_groups,
        "integrated_max_close_obj_groups": max_close_obj_groups,
        "integrated_max_post_obj_groups": max_post_obj_groups,
        "integrated_final_obj_groups": final_obj_groups,

        "integrated_max_move_support_groups": max_move_support_groups,
        "integrated_max_hold_support_groups": max_hold_support_groups,
        "integrated_max_close_support_groups": max_close_support_groups,
        "integrated_max_post_support_groups": max_post_support_groups,
        "integrated_final_support_groups": final_support_groups,

        "integrated_group_alpha": group_alpha,
        "integrated_frozen": frozen,
        "integrated_freeze_reason": freeze_reason,
        "integrated_final_hand_ctrl": last_hand_ctrl,
        "integrated_object_start": object_start.tolist(),
        "integrated_final_object_pos": final_pos.tolist(),
    })

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projected-ranked", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--npy", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--object-body", default="grasp_object")
    ap.add_argument("--target-site", default="dataset_hand_base_debug")
    ap.add_argument("--max-candidates", type=int, default=20)

    ap.add_argument("--settle-steps", type=int, default=800)
    ap.add_argument("--move-steps", type=int, default=2600)
    ap.add_argument("--hold-steps", type=int, default=1600)
    ap.add_argument("--close-steps", type=int, default=900)
    ap.add_argument("--post-steps", type=int, default=250)
    ap.add_argument("--ik-iters", type=int, default=350)

    ap.add_argument("--object-ready-dist", type=float, default=0.0015)
    ap.add_argument("--support-freeze-dist", type=float, default=0.0)
    ap.add_argument("--ready-stable-steps", type=int, default=5)

    ap.add_argument("--site-ready-pos-err", type=float, default=0.018)
    ap.add_argument("--site-ready-rot-err", type=float, default=0.12)

    ap.add_argument("--max-move-disp", type=float, default=0.018)
    ap.add_argument("--max-hold-disp", type=float, default=0.022)
    ap.add_argument("--max-prelift-disp", type=float, default=0.030)

    ap.add_argument("--clean-move-disp", type=float, default=0.004)
    ap.add_argument("--clean-hold-disp", type=float, default=0.008)
    ap.add_argument("--clean-ready-disp", type=float, default=0.012)
    ap.add_argument("--clean-prelift-disp", type=float, default=0.018)

    ap.add_argument("--abort-move-disp", type=float, default=0.050)
    ap.add_argument("--abort-hold-disp", type=float, default=0.055)
    ap.add_argument("--abort-close-disp", type=float, default=0.055)

    ap.add_argument("--stop-close-on-ready", action="store_true", default=False)

    ap.add_argument("--w-ready-ratio", type=float, default=700.0)
    ap.add_argument("--w-first-ready-disp", type=float, default=26000.0)
    ap.add_argument("--w-move-disp", type=float, default=32000.0)
    ap.add_argument("--w-hold-disp", type=float, default=26000.0)
    ap.add_argument("--w-prelift-disp", type=float, default=28000.0)
    ap.add_argument("--w-dz", type=float, default=450.0)
    ap.add_argument("--w-frozen", type=float, default=20.0)
    ap.add_argument("--w-old-quality", type=float, default=0.01)
    ap.add_argument("--w-old-path", type=float, default=0.005)

    args = ap.parse_args()

    args.projected_ranked = str(resolve(args.projected_ranked))
    args.model = str(resolve(args.model))
    args.npy = str(resolve(args.npy))
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_json(args.projected_ranked)
    if args.max_candidates and args.max_candidates > 0:
        rows = rows[:args.max_candidates]

    print("========== V4.26 INTEGRATED PRE-LIFT VALIDATION ==========")
    print("projected_ranked:", rel(args.projected_ranked))
    print("model           :", rel(args.model))
    print("npy             :", rel(args.npy))
    print("num             :", len(rows))
    print("out_dir         :", rel(out_dir))

    validated = []

    for i, row in enumerate(rows, start=1):
        print(
            f"\n[INTEGRATED] {i}/{len(rows)} "
            f"local={row.get('valid_local_index')} raw={row.get('raw_sample_index')} "
            f"type={row.get('selector_type')} dz={row.get('dz')} "
            f"proj_score={row.get('projection_score')}"
        )

        r = validate_candidate(args, row, i)
        validated.append(r)

        fre = r.get("integrated_first_ready_event")
        fre_s = None if fre is None else f"{fre.get('phase')}:{fre.get('step')} disp={fre.get('object_disp'):.4f}"

        print(
            f"[DONE] local={r.get('valid_local_index')} dz={r.get('dz')} "
            f"ready={r.get('integrated_ready')} success={r.get('integrated_success')} "
            f"score={r.get('integrated_score'):.2f} "
            f"move={r.get('integrated_max_move_disp',999):.4f} "
            f"hold={r.get('integrated_max_hold_disp',999):.4f} "
            f"prelift={r.get('integrated_final_prelift_disp',999):.4f} "
            f"site={r.get('integrated_min_hold_site_pos_err',999):.4f}/"
            f"{r.get('integrated_min_hold_site_rot_err',999):.4f} "
            f"first_ready={fre_s} "
            f"reason={r.get('integrated_failure_reason')}"
        )

    ranked = sorted(validated, key=lambda x: x.get("integrated_score", -1e9), reverse=True)
    success_rows = [r for r in ranked if r.get("integrated_success")]
    ready_rows = [r for r in ranked if r.get("integrated_ready")]

    if success_rows:
        best = success_rows[0]
        mode = "best_integrated_success"
    elif ready_rows:
        best = ready_rows[0]
        mode = "best_integrated_ready_not_clean"
    else:
        best = ranked[0] if ranked else {}
        mode = "best_available_failed"

    if isinstance(best, dict):
        best["integrated_selection_mode"] = mode

    save_json(out_dir / "integrated_candidates_ranked.json", ranked)
    save_json(out_dir / "integrated_success_candidates.json", success_rows)
    save_json(out_dir / "best_integrated_projected_candidate.json", best)

    lines = []
    lines.append("========== V4.26 INTEGRATED PRE-LIFT VALIDATION REPORT ==========")
    lines.append(f"input : {rel(args.projected_ranked)}")
    lines.append(f"num   : {len(ranked)}")
    lines.append(f"integrated_success_count: {len(success_rows)}")
    lines.append(f"integrated_ready_count  : {len(ready_rows)}")
    lines.append(f"selection_mode          : {mode}")
    lines.append("")
    lines.append("---- ranked ----")

    for idx, r in enumerate(ranked[:20], start=1):
        fre = r.get("integrated_first_ready_event")
        if fre is None:
            fre_txt = "None"
        else:
            fre_txt = f"{fre.get('phase')}:{fre.get('step')} disp={fre.get('object_disp'):.4f}"

        lines.append(
            f"rank={idx:02d} "
            f"local={r.get('valid_local_index'):03d} raw={r.get('raw_sample_index'):03d} "
            f"type={r.get('selector_type')} dz={r.get('dz'):+.4f} "
            f"score={r.get('integrated_score',0):.2f} "
            f"ready={r.get('integrated_ready')} success={r.get('integrated_success')} "
            f"move={r.get('integrated_max_move_disp',999):.4f} "
            f"hold={r.get('integrated_max_hold_disp',999):.4f} "
            f"prelift={r.get('integrated_final_prelift_disp',999):.4f} "
            f"site={r.get('integrated_min_hold_site_pos_err',999):.4f}/"
            f"{r.get('integrated_min_hold_site_rot_err',999):.4f} "
            f"first_ready={fre_txt} "
            f"obj={r.get('integrated_max_close_obj_groups')} "
            f"reason={r.get('integrated_failure_reason')}"
        )

    lines.append("")
    lines.append("---- selected best ----")
    if best:
        lines.append(f"local={best.get('valid_local_index')} raw={best.get('raw_sample_index')}")
        lines.append(f"type={best.get('selector_type')}")
        lines.append(f"dz={best.get('dz')}")
        lines.append(f"selection_mode={best.get('integrated_selection_mode')}")
        lines.append(f"integrated_ready={best.get('integrated_ready')}")
        lines.append(f"integrated_success={best.get('integrated_success')}")
        lines.append(f"integrated_score={best.get('integrated_score')}")
        lines.append(f"move_disp={best.get('integrated_max_move_disp')}")
        lines.append(f"hold_disp={best.get('integrated_max_hold_disp')}")
        lines.append(f"prelift_disp={best.get('integrated_final_prelift_disp')}")
        lines.append(f"first_ready={best.get('integrated_first_ready_event')}")
        lines.append(f"max_close_obj_groups={best.get('integrated_max_close_obj_groups')}")
        lines.append(f"frozen={best.get('integrated_frozen')}")
        lines.append(f"reason={best.get('integrated_failure_reason')}")

    lines.append("")
    lines.append("---- output ----")
    lines.append(f"ranked : {out_dir / 'integrated_candidates_ranked.json'}")
    lines.append(f"success: {out_dir / 'integrated_success_candidates.json'}")
    lines.append(f"best   : {out_dir / 'best_integrated_projected_candidate.json'}")
    lines.append("==============================================================")

    report = "\n".join(lines) + "\n"
    (out_dir / "integrated_pre_lift_report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
