#!/usr/bin/env python3
"""
V4.25 path-aware quality filter.

定位：
    在 V4.24 quality filter 之后，再加入真实动态 move/hold 阶段检查。
    不是最终执行器，不做 close/lift。
    目标是自动排除：
        exact short validation 看起来好，
        但真实动态移动到 q_grasp 时会把物体碰走/碰掉的候选。

输入：
    V4.24 quality_ranked_projected_candidates.json

输出：
    path_aware_candidates_ranked.json
    best_path_aware_projected_candidate.json
    path_aware_filter_report.txt

核心检查：
    1. IK 是否成功；
    2. smooth move 到 q_grasp/open 阶段物体位移是否过大；
    3. servo hold at q_grasp/open 阶段物体位移是否过大；
    4. site error 是否达到 ready 阈值；
    5. 只从 path_ready 的候选里选 best；
       如果没有 path_ready，则选惩罚后最好的，但标注 fallback。
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


def candidate_T_grasp(row, model, data, object_body, npy_path):
    local_idx = int(row["valid_local_index"])
    sample = H.load_sample(npy_path, local_idx)
    T_object_hand = H.sample_T_object_hand(sample)
    T_world_object = H.body_world_T(model, data, object_body)

    if isinstance(row, dict) and "T_grasp_projected" in row:
        tg = row.get("T_grasp_projected", {})
        if isinstance(tg, dict) and "T" in tg:
            return np.asarray(tg["T"], dtype=float), T_object_hand, T_world_object

    return T_world_object @ T_object_hand, T_object_hand, T_world_object


def validate_path(args, row, rank_input):
    model = mujoco.MjModel.from_xml_path(str(args.model))
    data = mujoco.MjData(model)

    local_idx = int(row["valid_local_index"])
    raw_idx = int(row.get("raw_sample_index", local_idx))

    sample = H.load_sample(args.npy, local_idx)
    close_ctrl = H.sample_ctrl(sample)
    side_open = side_open_from_close(close_ctrl)

    H.set_qpos_once(model, data, H.Q_HOME, side_open)

    for _ in range(args.settle_steps):
        step_ctrl(model, data, H.Q_HOME, side_open)

    object_start = H.object_pos(model, data, args.object_body).copy()

    T_grasp, T_object_hand, T_world_object = candidate_T_grasp(
        row, model, data, args.object_body, args.npy
    )

    ik = H.solve_site_ik(
        model,
        args.target_site,
        T_grasp,
        H.Q_HOME,
        max_iters=args.ik_iters,
    )

    out = dict(row)
    out["path_filter_input_rank"] = int(rank_input)
    out["path_filter_local_index"] = local_idx
    out["path_filter_raw_index"] = raw_idx
    out["path_T_grasp"] = H.mat_to_dict(T_grasp)
    out["path_T_world_object"] = H.mat_to_dict(T_world_object)
    out["path_ik"] = ik

    if not ik["success"]:
        out.update({
            "path_ready": False,
            "path_success": False,
            "path_quality_score": -1e6,
            "path_failure_reason": "ik_failed",
        })
        return out

    q_current = H.get_joint_values(model, data, H.ARM_JOINTS)
    q_grasp = ik["q_arm"]

    max_move_disp = 0.0
    max_hold_disp = 0.0
    max_move_obj_groups = {}
    max_move_support_groups = {}
    max_hold_obj_groups = {}
    max_hold_support_groups = {}

    move_aborted = False

    # smooth move: dynamic arm ctrl, hand open
    for k in range(args.move_steps):
        alpha = k / max(1, args.move_steps - 1)
        q_cmd = H.interp_dict(q_current, q_grasp, alpha, H.ARM_JOINTS)
        step_ctrl(model, data, q_cmd, side_open)

        obj_pos = H.object_pos(model, data, args.object_body)
        disp = float(np.linalg.norm(obj_pos - object_start))
        max_move_disp = max(max_move_disp, disp)

        st = H.contact_state(model, data, args.object_body)
        obj_groups, _ = real_groups_from_contacts(st, "object", args.object_ready_dist)
        support_groups, _ = real_groups_from_contacts(st, "support", args.support_freeze_dist)

        if len(obj_groups) > len(max_move_obj_groups):
            max_move_obj_groups = dict(obj_groups)
        if len(support_groups) > len(max_move_support_groups):
            max_move_support_groups = dict(support_groups)

        if disp > args.abort_move_disp:
            move_aborted = True
            break

    move_site_pos_err, move_site_rot_err = current_site_error(
        model, data, args.target_site, T_grasp
    )

    # servo hold at q_grasp/open
    min_hold_site_pos_err = 999.0
    min_hold_site_rot_err = 999.0
    hold_aborted = False

    if not move_aborted:
        for k in range(args.hold_steps):
            step_ctrl(model, data, q_grasp, side_open)

            obj_pos = H.object_pos(model, data, args.object_body)
            disp = float(np.linalg.norm(obj_pos - object_start))
            max_hold_disp = max(max_hold_disp, disp)

            st = H.contact_state(model, data, args.object_body)
            obj_groups, _ = real_groups_from_contacts(st, "object", args.object_ready_dist)
            support_groups, _ = real_groups_from_contacts(st, "support", args.support_freeze_dist)

            if len(obj_groups) > len(max_hold_obj_groups):
                max_hold_obj_groups = dict(obj_groups)
            if len(support_groups) > len(max_hold_support_groups):
                max_hold_support_groups = dict(support_groups)

            site_pos_err, site_rot_err = current_site_error(model, data, args.target_site, T_grasp)
            min_hold_site_pos_err = min(min_hold_site_pos_err, site_pos_err)
            min_hold_site_rot_err = min(min_hold_site_rot_err, site_rot_err)

            if disp > args.abort_hold_disp:
                hold_aborted = True
                break

    final_obj_pos = H.object_pos(model, data, args.object_body)
    final_preclose_disp = float(np.linalg.norm(final_obj_pos - object_start))

    site_ready = (
        min_hold_site_pos_err <= args.site_ready_pos_err
        and min_hold_site_rot_err <= args.site_ready_rot_err
    )

    move_ok = max_move_disp <= args.max_move_disp
    hold_ok = max_hold_disp <= args.max_hold_disp
    path_ready = bool(site_ready and move_ok and hold_ok and not move_aborted and not hold_aborted)

    base_quality = float(row.get("quality_score", row.get("projection_score", 0.0)))

    score = base_quality
    if path_ready:
        score += 1500.0
    else:
        score -= 1500.0

    score -= args.w_move_disp * max(0.0, max_move_disp - args.clean_move_disp)
    score -= args.w_hold_disp * max(0.0, max_hold_disp - args.clean_hold_disp)
    score -= args.w_site_pos * max(0.0, min_hold_site_pos_err - args.site_ready_pos_err)
    score -= args.w_site_rot * max(0.0, min_hold_site_rot_err - args.site_ready_rot_err)

    if move_aborted:
        reason = "move_push_abort"
    elif hold_aborted:
        reason = "hold_push_abort"
    elif not site_ready:
        reason = "site_not_ready"
    elif not move_ok:
        reason = "move_disp_too_large"
    elif not hold_ok:
        reason = "hold_disp_too_large"
    else:
        reason = ""

    out.update({
        "path_ready": bool(path_ready),
        "path_success": bool(path_ready),
        "path_quality_score": float(score),
        "path_failure_reason": reason,

        "path_site_ready": bool(site_ready),
        "path_move_ok": bool(move_ok),
        "path_hold_ok": bool(hold_ok),
        "path_move_aborted": bool(move_aborted),
        "path_hold_aborted": bool(hold_aborted),

        "path_max_move_disp": float(max_move_disp),
        "path_max_hold_disp": float(max_hold_disp),
        "path_final_preclose_disp": float(final_preclose_disp),
        "path_move_site_pos_err": float(move_site_pos_err),
        "path_move_site_rot_err": float(move_site_rot_err),
        "path_min_hold_site_pos_err": float(min_hold_site_pos_err),
        "path_min_hold_site_rot_err": float(min_hold_site_rot_err),

        "path_max_move_object_groups": max_move_obj_groups,
        "path_max_move_support_groups": max_move_support_groups,
        "path_max_hold_object_groups": max_hold_obj_groups,
        "path_max_hold_support_groups": max_hold_support_groups,
    })

    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--quality-ranked", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--npy", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--object-body", default="grasp_object")
    ap.add_argument("--target-site", default="dataset_hand_base_debug")
    ap.add_argument("--max-candidates", type=int, default=20)

    ap.add_argument("--settle-steps", type=int, default=800)
    ap.add_argument("--move-steps", type=int, default=2200)
    ap.add_argument("--hold-steps", type=int, default=1200)
    ap.add_argument("--ik-iters", type=int, default=350)

    ap.add_argument("--object-ready-dist", type=float, default=0.0015)
    ap.add_argument("--support-freeze-dist", type=float, default=0.0)

    ap.add_argument("--site-ready-pos-err", type=float, default=0.018)
    ap.add_argument("--site-ready-rot-err", type=float, default=0.12)

    ap.add_argument("--max-move-disp", type=float, default=0.018)
    ap.add_argument("--max-hold-disp", type=float, default=0.025)
    ap.add_argument("--clean-move-disp", type=float, default=0.006)
    ap.add_argument("--clean-hold-disp", type=float, default=0.012)
    ap.add_argument("--abort-move-disp", type=float, default=0.045)
    ap.add_argument("--abort-hold-disp", type=float, default=0.050)

    ap.add_argument("--w-move-disp", type=float, default=30000.0)
    ap.add_argument("--w-hold-disp", type=float, default=24000.0)
    ap.add_argument("--w-site-pos", type=float, default=12000.0)
    ap.add_argument("--w-site-rot", type=float, default=1500.0)
    args = ap.parse_args()

    args.quality_ranked = str(resolve(args.quality_ranked))
    args.model = str(resolve(args.model))
    args.npy = str(resolve(args.npy))
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_json(args.quality_ranked)
    if args.max_candidates and args.max_candidates > 0:
        rows = rows[:args.max_candidates]

    print("========== V4.25 PATH-AWARE QUALITY FILTER ==========")
    print("quality_ranked:", rel(args.quality_ranked))
    print("model         :", rel(args.model))
    print("npy           :", rel(args.npy))
    print("num           :", len(rows))
    print("out_dir       :", rel(out_dir))

    validated = []

    for i, row in enumerate(rows, start=1):
        print(
            f"\n[PATH] {i}/{len(rows)} "
            f"local={row.get('valid_local_index')} raw={row.get('raw_sample_index')} "
            f"type={row.get('selector_type')} dz={row.get('dz')} "
            f"qlabel={row.get('quality_label')} qscore={row.get('quality_score')}"
        )

        r = validate_path(args, row, i)
        validated.append(r)

        print(
            f"[DONE] local={r.get('valid_local_index')} dz={r.get('dz')} "
            f"path_ready={r.get('path_ready')} pscore={r.get('path_quality_score'):.2f} "
            f"move={r.get('path_max_move_disp',999):.4f} "
            f"hold={r.get('path_max_hold_disp',999):.4f} "
            f"site={r.get('path_min_hold_site_pos_err',999):.4f}/"
            f"{r.get('path_min_hold_site_rot_err',999):.4f} "
            f"reason={r.get('path_failure_reason')}"
        )

    ranked = sorted(validated, key=lambda x: x.get("path_quality_score", -1e9), reverse=True)
    ready_rows = [r for r in ranked if r.get("path_ready")]

    if ready_rows:
        best = ready_rows[0]
        mode = "best_path_ready"
    else:
        best = ranked[0] if ranked else {}
        mode = "best_available_path_failed"

    if isinstance(best, dict):
        best["path_selection_mode"] = mode

    save_json(out_dir / "path_aware_candidates_ranked.json", ranked)
    save_json(out_dir / "path_ready_candidates.json", ready_rows)
    save_json(out_dir / "best_path_aware_projected_candidate.json", best)

    lines = []
    lines.append("========== V4.25 PATH-AWARE QUALITY FILTER REPORT ==========")
    lines.append(f"input : {rel(args.quality_ranked)}")
    lines.append(f"num   : {len(ranked)}")
    lines.append(f"path_ready_count: {len(ready_rows)}")
    lines.append(f"selection_mode  : {mode}")
    lines.append("")
    lines.append("---- ranked ----")

    for i, r in enumerate(ranked[:20], start=1):
        lines.append(
            f"rank={i:02d} "
            f"local={r.get('valid_local_index'):03d} raw={r.get('raw_sample_index'):03d} "
            f"type={r.get('selector_type')} dz={r.get('dz'):+.4f} "
            f"qlabel={r.get('quality_label')} "
            f"qscore={r.get('quality_score',0):.2f} "
            f"pscore={r.get('path_quality_score',0):.2f} "
            f"path_ready={r.get('path_ready')} "
            f"move={r.get('path_max_move_disp',999):.4f} "
            f"hold={r.get('path_max_hold_disp',999):.4f} "
            f"site={r.get('path_min_hold_site_pos_err',999):.4f}/"
            f"{r.get('path_min_hold_site_rot_err',999):.4f} "
            f"reason={r.get('path_failure_reason')}"
        )

    lines.append("")
    lines.append("---- selected best ----")
    if best:
        lines.append(f"local={best.get('valid_local_index')} raw={best.get('raw_sample_index')}")
        lines.append(f"type={best.get('selector_type')}")
        lines.append(f"dz={best.get('dz')}")
        lines.append(f"quality_label={best.get('quality_label')}")
        lines.append(f"path_ready={best.get('path_ready')}")
        lines.append(f"path_selection_mode={best.get('path_selection_mode')}")
        lines.append(f"path_quality_score={best.get('path_quality_score')}")
        lines.append(f"path_max_move_disp={best.get('path_max_move_disp')}")
        lines.append(f"path_max_hold_disp={best.get('path_max_hold_disp')}")
        lines.append(f"path_min_hold_site_pos_err={best.get('path_min_hold_site_pos_err')}")
        lines.append(f"path_failure_reason={best.get('path_failure_reason')}")

    lines.append("")
    lines.append("---- output ----")
    lines.append(f"ranked: {out_dir / 'path_aware_candidates_ranked.json'}")
    lines.append(f"ready : {out_dir / 'path_ready_candidates.json'}")
    lines.append(f"best  : {out_dir / 'best_path_aware_projected_candidate.json'}")
    lines.append("============================================================")

    report = "\n".join(lines) + "\n"
    (out_dir / "path_aware_filter_report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
