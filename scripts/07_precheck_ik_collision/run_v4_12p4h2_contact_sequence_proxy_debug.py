#!/usr/bin/env python3
"""
文件名：
    run_v4_12p4h2_contact_sequence_proxy_debug.py

脚本类别：
    debug / fast-geometric-score / contact-sequence-proxy / force-closure-proxy / candidate-selector

用途：
    本脚本用于 V4.12P4H2 阶段。
    P4H 已经能选出几何 force-line 看起来不错的 hand-local candidate，
    但实际 close 时可能出现“四指侧先碰物体，把物体顺着手内部推走，而 thumb 没有形成对侧约束”的问题。
    本脚本在 P4H Top-K 基础上加入“接触顺序代理评分”，用于淘汰这种假抓握。

核心流程：
    1. 读取 P4H summary.json 中的 Top-K hand-local delta。
    2. 固定当前 P3 best q_grasp，不重新做整臂 IK。
    3. 构造 side_open_ctrl 和 close_target。
    4. 对每个 hand-local delta，按 alpha 从 0 到 1 扫描 hand ctrl：
           hand_ctrl(alpha) = side_open + alpha * (close_target - side_open)
    5. 每个 alpha 下只做 FK + 几何距离判断，不跑动态。
    6. 记录 thumb / index / middle / ring / pinky 第一次进入物体侧壁接触带的 alpha。
    7. 评分时强惩罚：
           - non-thumb 明显早于 thumb；
           - thumb 一直不接触；
           - final close 时少于 2 个 non-thumb 接触；
           - thumb-finger force line 不穿过物体中心；
           - 对握方向 radial_dot 不够负；
           - 接触高度差过大。
    8. 输出新的 best_candidate 和 best_config。
    9. 自动生成只验证 best 一次的 P2/P3/P4J strict viewer 脚本。

输入：
    --model
        MuJoCo XML 场景，建议使用 hard_support 版本。
    --candidate
        原始 candidate JSON。
    --p3-json
        当前 P3 输出 JSON，用于读取 q_grasp。
    --p4h-summary
        P4H 输出 summary.json。
    --best-config
        P4H 输出 best_force_proxy_config.json，主要用于 thumb preshape 先验。
    --object-body
        物体 body 名称，例如 grasp_can。
    --target-body
        hand target body，当前通常是 fr3_link7。

输出：
    --out-dir/summary.json
        P4H2 contact sequence proxy 排序结果。
    --out-dir/topk_summary.txt
        可读排行榜。
    --out-dir/best_candidate.json
        重新 patch 后的 best candidate。
    --out-dir/best_contact_sequence_config.json
        best 的接触顺序和 force proxy 评分细节。
    --out-dir/run_best_contact_sequence_viewer.sh
        只验证 best 一次的 viewer 脚本。

当前流程位置：
    P4H force-proxy Top-K
        -> P4H2 contact-sequence proxy
        -> P2/P3/P4J strict viewer 验证 best

本脚本不负责：
    1. 不对每个候选跑完整动态抓取。
    2. 不对每个候选重新做 IK。
    3. 不替代真正 MuJoCo contact force。
    4. 不允许把“四指先推走物体”当成有效抓握。
"""

from pathlib import Path
import argparse
import importlib.util
import json
import math
import os
import shlex
import numpy as np
import mujoco


PROJECT = Path.home() / "Projects/o7_mujoco_sim"

P4F_PATH = PROJECT / "scripts/05_execution_runner/run_v4_12p4f_target_close_debug.py"
P4H_PATH = PROJECT / "scripts/07_precheck_ik_collision/run_v4_12p4h_force_closure_proxy_hand_refine_debug.py"
P2_SCRIPT = PROJECT / "scripts/07_precheck_ik_collision/run_v4_12p2_pinocchio_multiseed_ik_debug.py"
P3_SCRIPT = PROJECT / "scripts/07_precheck_ik_collision/run_v4_12p3_pinocchio_ik_plus_mujoco_collision_precheck_debug.py"
P4J_STRICT_SCRIPT = PROJECT / "scripts/05_execution_runner/run_v4_12p4j_strict_latch_close_debug.py"
RUN_CLEAN = PROJECT / "run_mujoco_clean.sh"

ARM_JOINTS = [
    "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
    "fr3_joint5", "fr3_joint6", "fr3_joint7",
]

HAND_JOINTS = [
    "thumb_cmc_roll",
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]

NON_THUMB_GROUPS = ["index", "middle", "ring", "pinky"]
ALL_GROUPS = ["thumb", "index", "middle", "ring", "pinky"]


def import_py(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p4f = import_py(P4F_PATH, "p4f")
p4h = import_py(P4H_PATH, "p4h")


def resolve_path(p):
    p = Path(str(p)).expanduser()
    if not p.is_absolute():
        p = PROJECT / p
    return p


def rel(p):
    p = resolve_path(p)
    try:
        return str(p.relative_to(PROJECT))
    except Exception:
        return str(p)


def load_json(p):
    with open(resolve_path(p), "r") as f:
        return json.load(f)


def save_json(p, obj):
    p = resolve_path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(to_jsonable(obj), f, indent=2)


def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


def shell_join(cmd):
    return " ".join(shlex.quote(str(x)) for x in cmd)


def interp_dict(a, b, alpha, keys):
    out = {}
    for k in keys:
        av = float(a.get(k, 0.0))
        bv = float(b.get(k, av))
        out[k] = av + float(alpha) * (bv - av)
    return out


def selected_plan(p3, which):
    item = p3.get(which)
    if item is None:
        raise RuntimeError(f"{which} is None in p3 json")
    for k in ["q_pre", "q_grasp", "q_lift"]:
        if k not in item:
            raise RuntimeError(f"{which} missing {k}")
    return item


def group_contact_candidate(points, obj, args):
    b = p4h.side_surface_best(points, obj, args)
    good = bool(
        b.get("valid", False)
        and b["side_error"] <= args.contact_surface_tol
        and b["z_error"] <= args.contact_z_tol
    )
    b["good"] = good
    return b


def evaluate_one_delta(model, data, args, q_grasp, object_geom, support_top,
                       T_world_target_old, delta_meta, side_open_ctrl, close_target):
    dx, dy, dz = delta_meta["delta_local"]
    yaw = float(delta_meta.get("yaw_deg", 0.0))
    roll = float(delta_meta.get("roll_deg", 0.0))
    Tdl = p4h.T_delta_local(dx, dy, dz, yaw, roll)

    alphas = np.linspace(0.0, 1.0, args.alpha_samples)

    first_alpha = {g: None for g in ALL_GROUPS}
    first_detail = {g: None for g in ALL_GROUPS}
    final_status = {}
    force_pairs_by_alpha = []

    for a in alphas:
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        p4h.apply_arm_q(model, data, q_grasp)
        ctrl = interp_dict(side_open_ctrl, close_target, float(a), HAND_JOINTS)
        p4h.apply_hand_qpos_ctrl(model, data, ctrl)
        mujoco.mj_forward(model, data)

        hand_pts, _ = p4h.collect_hand_points(model, data)

        moved = {}
        for g, pts in hand_pts.items():
            moved[g] = p4h.transform_points(pts, T_world_target_old, Tdl)

        group_status = {}
        for g in ALL_GROUPS:
            group_status[g] = group_contact_candidate(moved[g], object_geom, args)

            if group_status[g]["good"] and first_alpha[g] is None:
                first_alpha[g] = float(a)
                first_detail[g] = group_status[g]

        thumb = group_status["thumb"]
        pair_candidates = []

        if thumb["good"]:
            for g in NON_THUMB_GROUPS:
                fg = group_status[g]
                if not fg["good"]:
                    continue
                lm = p4h.line_metrics(thumb, fg, object_geom)
                pair_ok = bool(
                    lm["line_dist_xy"] <= args.force_line_tol
                    and args.force_alpha_min <= lm["alpha_on_segment"] <= args.force_alpha_max
                    and lm["opposition_dot"] <= args.force_radial_dot_max
                    and lm["z_diff"] <= args.force_z_diff_tol
                )
                pair_candidates.append({
                    "alpha": float(a),
                    "finger_group": g,
                    "thumb": thumb,
                    "finger": fg,
                    "line_metrics": lm,
                    "pair_ok": pair_ok,
                })

        if pair_candidates:
            pair_candidates.sort(
                key=lambda x: (
                    0 if x["pair_ok"] else 1,
                    x["line_metrics"]["line_dist_xy"],
                    x["line_metrics"]["opposition_error"],
                    x["line_metrics"]["z_diff"],
                )
            )
            force_pairs_by_alpha.append(pair_candidates[0])

        if abs(float(a) - 1.0) < 1e-9:
            final_status = group_status

    final_good_non_thumb = [g for g in NON_THUMB_GROUPS if final_status.get(g, {}).get("good", False)]
    first_non_thumb_vals = [first_alpha[g] for g in NON_THUMB_GROUPS if first_alpha[g] is not None]
    first_non_thumb = min(first_non_thumb_vals) if first_non_thumb_vals else None

    first_thumb = first_alpha["thumb"]

    best_final_pair = None
    if final_status.get("thumb", {}).get("good", False):
        pairs = []
        for g in NON_THUMB_GROUPS:
            if not final_status.get(g, {}).get("good", False):
                continue
            lm = p4h.line_metrics(final_status["thumb"], final_status[g], object_geom)
            pairs.append({
                "finger_group": g,
                "thumb": final_status["thumb"],
                "finger": final_status[g],
                "line_metrics": lm,
                "pair_ok": bool(
                    lm["line_dist_xy"] <= args.force_line_tol
                    and args.force_alpha_min <= lm["alpha_on_segment"] <= args.force_alpha_max
                    and lm["opposition_dot"] <= args.force_radial_dot_max
                    and lm["z_diff"] <= args.force_z_diff_tol
                ),
            })
        if pairs:
            pairs.sort(
                key=lambda x: (
                    0 if x["pair_ok"] else 1,
                    x["line_metrics"]["line_dist_xy"],
                    x["line_metrics"]["opposition_error"],
                    x["line_metrics"]["z_diff"],
                )
            )
            best_final_pair = pairs[0]

    earliest_pair = None
    if force_pairs_by_alpha:
        ok_pairs = [p for p in force_pairs_by_alpha if p["pair_ok"]]
        if ok_pairs:
            earliest_pair = ok_pairs[0]
        else:
            earliest_pair = force_pairs_by_alpha[0]

    score = 0.0
    reasons = []

    if first_thumb is None:
        score += args.penalty_no_thumb
        reasons.append("no_thumb_contact_proxy")
    else:
        score += args.w_first_thumb_alpha * first_thumb
        reasons.append(f"first_thumb={first_thumb:.3f}")

    if first_non_thumb is None:
        score += args.penalty_no_non_thumb
        reasons.append("no_non_thumb_contact_proxy")
    else:
        reasons.append(f"first_non_thumb={first_non_thumb:.3f}")

    if first_thumb is not None and first_non_thumb is not None:
        early_gap = first_thumb - first_non_thumb
        if early_gap > args.allowed_non_thumb_early_gap:
            score += args.penalty_non_thumb_too_early + args.w_early_gap * early_gap
            reasons.append(f"non_thumb_too_early_gap={early_gap:.3f}")
        else:
            score -= args.bonus_contact_sequence_ok
            reasons.append("contact_sequence_ok")

    if len(final_good_non_thumb) < args.min_final_non_thumb_groups:
        score += args.penalty_not_enough_final_non_thumb
        reasons.append(f"final_non_thumb_too_few={len(final_good_non_thumb)}")
    else:
        score -= args.bonus_final_multi_finger
        reasons.append(f"final_non_thumb={final_good_non_thumb}")

    if best_final_pair is None:
        score += args.penalty_no_final_force_pair
        reasons.append("no_final_force_pair")
    else:
        lm = best_final_pair["line_metrics"]
        score += args.w_force_line * lm["line_dist_xy"]
        score += args.w_force_opposition * lm["opposition_error"]
        score += args.w_force_z * lm["z_diff"]
        score += args.w_force_alpha * (
            max(0.0, args.force_alpha_min - lm["alpha_on_segment"])
            + max(0.0, lm["alpha_on_segment"] - args.force_alpha_max)
        )
        if best_final_pair["pair_ok"]:
            score -= args.bonus_final_force_pair_ok
            reasons.append(f"final_force_pair_ok={best_final_pair['finger_group']}")
        else:
            reasons.append(f"final_force_pair_bad={best_final_pair['finger_group']}")

    if earliest_pair is None:
        score += args.penalty_no_sweep_force_pair
        reasons.append("no_sweep_force_pair")
    else:
        if earliest_pair["pair_ok"]:
            score -= args.bonus_sweep_force_pair_ok
            reasons.append(f"sweep_force_pair_ok_alpha={earliest_pair['alpha']:.3f}:{earliest_pair['finger_group']}")
        else:
            reasons.append(f"sweep_force_pair_bad_alpha={earliest_pair['alpha']:.3f}:{earliest_pair['finger_group']}")

    delta_norm = float(np.linalg.norm(np.asarray(delta_meta["delta_local"], dtype=float)))
    score += args.w_delta * delta_norm
    score += args.w_yaw * math.radians(abs(yaw))
    score += args.w_roll * math.radians(abs(roll))

    return {
        "score": float(score),
        "reasons": reasons,
        "delta_meta": delta_meta,
        "first_alpha": first_alpha,
        "first_detail": first_detail,
        "final_good_non_thumb": final_good_non_thumb,
        "final_status": final_status,
        "best_final_pair": best_final_pair,
        "earliest_pair": earliest_pair,
        "delta_norm": delta_norm,
    }


def patch_candidate(candidate, model, data, args, q_grasp, best_record):
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    p4h.apply_arm_q(model, data, q_grasp)
    mujoco.mj_forward(model, data)

    T_world_object = p4h.T_body(model, data, args.object_body)
    T_world_target_old = p4h.T_body(model, data, args.target_body)

    d = best_record["delta_meta"]
    dx, dy, dz = d["delta_local"]
    Tdl = p4h.T_delta_local(dx, dy, dz, d.get("yaw_deg", 0.0), d.get("roll_deg", 0.0))

    patched, patch_info = p4h.patch_candidate(
        candidate=candidate,
        T_world_object=T_world_object,
        T_world_target_old=T_world_target_old,
        T_delta=Tdl,
        best_record=best_record,
    )

    return patched, patch_info


def write_viewer_script(args, out_dir):
    out_dir = resolve_path(out_dir)
    script = out_dir / "run_best_contact_sequence_viewer.sh"

    best_candidate = out_dir / "best_candidate.json"
    best_config = out_dir / "best_contact_sequence_config.json"
    p2_json = out_dir / "best_p2.json"
    p3_json = out_dir / "best_p3.json"
    best_plan = out_dir / "best_plan.json"
    out_json = out_dir / "best_p4j_contact_sequence_viewer.json"

    lines = []
    lines.append("#!/usr/bin/env bash")
    lines.append("set -e")
    lines.append("cd ~/Projects/o7_mujoco_sim")
    lines.append("source ~/mujoco_env/bin/activate")
    lines.append("")

    lines.append("echo '===== P2 for P4H2 best candidate ====='")
    lines.append(shell_join([
        "python3",
        rel(P2_SCRIPT),
        "--urdf", args.urdf,
        "--model", args.model,
        "--candidate", rel(best_candidate),
        "--runner-json", args.runner_json,
        "--object-body", args.object_body,
        "--target-frame", args.target_frame,
        "--out", rel(p2_json),
        "--random-seeds", str(args.p2_random_seeds),
        "--random-std", str(args.p2_random_std),
        "--max-iters", str(args.p2_max_iters),
        "--pos-tol", str(args.p2_pos_tol),
        "--rot-tol", str(args.p2_rot_tol),
        "--rot-weight", str(args.p2_rot_weight),
    ]) + f" 2>&1 | tee {shlex.quote(rel(out_dir / 'best_p2.txt'))}")

    lines.append("")
    lines.append("echo '===== P3 for P4H2 best candidate ====='")
    lines.append(shell_join([
        "python3",
        rel(P3_SCRIPT),
        "--p2-json", rel(p2_json),
        "--model", args.model,
        "--candidate", rel(best_candidate),
        "--object-body", args.object_body,
        "--out", rel(p3_json),
        "--best-plan-out", rel(best_plan),
        "--top-per-target", str(args.p3_top_per_target),
        "--max-combos", str(args.p3_max_combos),
        "--path-samples", str(args.p3_path_samples),
        "--min-hand-support-clearance", str(args.p3_min_hand_support_clearance),
        "--min-fr3-object-clearance", str(args.p3_min_fr3_object_clearance),
        "--max-grasp-hand-object-distance", str(args.p3_max_grasp_hand_object_distance),
        "--min-joint-margin", str(args.p3_min_joint_margin),
    ]) + f" 2>&1 | tee {shlex.quote(rel(out_dir / 'best_p3.txt'))}")

    lines.append("")
    lines.append("echo '===== P4J strict viewer for P4H2 best candidate ====='")
    lines.append(shell_join([
        str(RUN_CLEAN),
        rel(P4J_STRICT_SCRIPT),
        "--model", args.model,
        "--candidate", rel(best_candidate),
        "--p3-json", rel(p3_json),
        "--best-config", rel(best_config),
        "--which", "best_available",
        "--object-body", args.object_body,
        "--out", rel(out_json),
        "--viewer",
        "--move-steps", str(args.verify_move_steps),
        "--thumb-preshape-steps", str(args.verify_thumb_preshape_steps),
        "--close-duration", str(args.verify_close_duration),
        "--settle-duration", str(args.verify_settle_duration),
        "--hold-duration", str(args.verify_hold_duration),
        "--lift-duration", str(args.verify_lift_duration),
        "--finger-close-scale", str(args.finger_close_scale),
        "--thumb-pitch-from-finger-gain", str(args.thumb_pitch_from_finger_gain),
        "--safe-object-disp", str(args.verify_safe_object_disp),
        "--hard-object-push-disp", str(args.verify_hard_object_push_disp),
        "--max-support-penetration", str(args.verify_max_support_penetration),
        "--hard-support-penetration", str(args.verify_hard_support_penetration),
        "--force-stable-steps", str(args.verify_force_stable_steps),
        "--min-contact-normal-force", str(args.verify_min_contact_normal_force),
        "--min-latch-alpha", str(args.verify_min_latch_alpha),
        "--min-latch-non-thumb-groups", str(args.verify_min_latch_non_thumb_groups),
        "--force-line-tol", str(args.force_line_tol),
        "--force-alpha-min", str(args.force_alpha_min),
        "--force-alpha-max", str(args.force_alpha_max),
        "--force-radial-dot-max", str(args.force_radial_dot_max),
        "--force-z-diff-tol", str(args.force_z_diff_tol),
        "--keep-viewer-open",
        "--frame-sleep", str(args.verify_frame_sleep),
    ]) + f" 2>&1 | tee {shlex.quote(rel(out_dir / 'best_p4j_viewer.txt'))}")

    with open(script, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")

    os.chmod(script, 0o755)
    return script


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--p3-json", required=True)
    ap.add_argument("--p4h-summary", required=True)
    ap.add_argument("--best-config", required=True)
    ap.add_argument("--which", default="best_available", choices=["best_available", "best_pass"])
    ap.add_argument("--object-body", default="grasp_can")
    ap.add_argument("--target-body", default="fr3_link7")
    ap.add_argument("--target-frame", default="fr3_link7")
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--eval-top-k-from-p4h", type=int, default=20)
    ap.add_argument("--alpha-samples", type=int, default=31)

    ap.add_argument("--finger-close-scale", type=float, default=0.85)
    ap.add_argument("--thumb-pitch-from-finger-gain", type=float, default=0.35)
    ap.add_argument("--preshape-fingers-from-best", action="store_true")

    ap.add_argument("--contact-surface-tol", type=float, default=0.014)
    ap.add_argument("--contact-z-tol", type=float, default=0.035)
    ap.add_argument("--allowed-non-thumb-early-gap", type=float, default=0.08)
    ap.add_argument("--min-final-non-thumb-groups", type=int, default=2)

    ap.add_argument("--z-weight", type=float, default=0.30)
    ap.add_argument("--force-line-tol", type=float, default=0.012)
    ap.add_argument("--force-alpha-min", type=float, default=0.15)
    ap.add_argument("--force-alpha-max", type=float, default=0.85)
    ap.add_argument("--force-radial-dot-max", type=float, default=-0.25)
    ap.add_argument("--force-z-diff-tol", type=float, default=0.035)

    ap.add_argument("--penalty-no-thumb", type=float, default=100.0)
    ap.add_argument("--penalty-no-non-thumb", type=float, default=60.0)
    ap.add_argument("--penalty-non-thumb-too-early", type=float, default=80.0)
    ap.add_argument("--penalty-not-enough-final-non-thumb", type=float, default=70.0)
    ap.add_argument("--penalty-no-final-force-pair", type=float, default=90.0)
    ap.add_argument("--penalty-no-sweep-force-pair", type=float, default=40.0)

    ap.add_argument("--bonus-contact-sequence-ok", type=float, default=20.0)
    ap.add_argument("--bonus-final-multi-finger", type=float, default=15.0)
    ap.add_argument("--bonus-final-force-pair-ok", type=float, default=20.0)
    ap.add_argument("--bonus-sweep-force-pair-ok", type=float, default=10.0)

    ap.add_argument("--w-first-thumb-alpha", type=float, default=2.0)
    ap.add_argument("--w-early-gap", type=float, default=100.0)
    ap.add_argument("--w-force-line", type=float, default=12.0)
    ap.add_argument("--w-force-opposition", type=float, default=5.0)
    ap.add_argument("--w-force-z", type=float, default=4.0)
    ap.add_argument("--w-force-alpha", type=float, default=5.0)
    ap.add_argument("--w-delta", type=float, default=0.6)
    ap.add_argument("--w-yaw", type=float, default=0.4)
    ap.add_argument("--w-roll", type=float, default=0.4)

    ap.add_argument("--urdf", default="models/fr3_o7/fr3_o7_mujoco_ready_obj.urdf")
    ap.add_argument("--runner-json", default="diagnostics/current_v412/v4_12a_can52_strict_clearance_debug_runner.json")

    ap.add_argument("--p2-random-seeds", type=int, default=12)
    ap.add_argument("--p2-random-std", type=float, default=0.6)
    ap.add_argument("--p2-max-iters", type=int, default=300)
    ap.add_argument("--p2-pos-tol", type=float, default=0.00025)
    ap.add_argument("--p2-rot-tol", type=float, default=0.0025)
    ap.add_argument("--p2-rot-weight", type=float, default=0.55)

    ap.add_argument("--p3-top-per-target", type=int, default=6)
    ap.add_argument("--p3-max-combos", type=int, default=216)
    ap.add_argument("--p3-path-samples", type=int, default=32)
    ap.add_argument("--p3-min-hand-support-clearance", type=float, default=0.0)
    ap.add_argument("--p3-min-fr3-object-clearance", type=float, default=0.0)
    ap.add_argument("--p3-max-grasp-hand-object-distance", type=float, default=0.045)
    ap.add_argument("--p3-min-joint-margin", type=float, default=0.0)

    ap.add_argument("--verify-move-steps", type=int, default=100)
    ap.add_argument("--verify-thumb-preshape-steps", type=int, default=100)
    ap.add_argument("--verify-close-duration", type=float, default=1.8)
    ap.add_argument("--verify-settle-duration", type=float, default=0.8)
    ap.add_argument("--verify-hold-duration", type=float, default=0.5)
    ap.add_argument("--verify-lift-duration", type=float, default=2.0)
    ap.add_argument("--verify-safe-object-disp", type=float, default=0.020)
    ap.add_argument("--verify-hard-object-push-disp", type=float, default=0.035)
    ap.add_argument("--verify-max-support-penetration", type=float, default=0.001)
    ap.add_argument("--verify-hard-support-penetration", type=float, default=0.001)
    ap.add_argument("--verify-force-stable-steps", type=int, default=8)
    ap.add_argument("--verify-min-contact-normal-force", type=float, default=0.02)
    ap.add_argument("--verify-min-latch-alpha", type=float, default=0.78)
    ap.add_argument("--verify-min-latch-non-thumb-groups", type=int, default=2)
    ap.add_argument("--verify-frame-sleep", type=float, default=0.002)

    args = ap.parse_args()

    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    for p in [P4F_PATH, P4H_PATH, P2_SCRIPT, P3_SCRIPT, P4J_STRICT_SCRIPT, RUN_CLEAN]:
        if not p.exists():
            raise RuntimeError(f"missing required script: {p}")

    model_path = resolve_path(args.model)
    candidate_path = resolve_path(args.candidate)
    p3_path = resolve_path(args.p3_json)
    p4h_summary_path = resolve_path(args.p4h_summary)
    best_config_path = resolve_path(args.best_config)

    candidate = load_json(candidate_path)
    p3 = load_json(p3_path)
    p4h_summary = load_json(p4h_summary_path)
    best_config = load_json(best_config_path)

    plan = selected_plan(p3, args.which)
    q_grasp = plan["q_grasp"]

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    p4h.apply_arm_q(model, data, q_grasp)
    mujoco.mj_forward(model, data)

    object_points = p4h.collect_object_points(model, data, args.object_body)
    object_geom = p4h.estimate_object_geom(object_points)
    support_top, support_names = p4h.collect_support_top(model, data)
    T_world_target_old = p4h.T_body(model, data, args.target_body)

    candidate_ctrl, candidate_ctrl_source = p4f.extract_candidate_ctrl(candidate, model)
    best_ctrl, best_ctrl_source = p4f.extract_best_config_ctrl(best_config, model)
    open_ctrl = p4f.make_open_ctrl(model)
    side_open_ctrl = p4f.make_side_open_ctrl(model, open_ctrl, candidate_ctrl, best_ctrl, args)
    close_target = p4f.make_close_target(model, side_open_ctrl, candidate_ctrl, args)

    raw_topk = p4h_summary.get("topk", [])[:args.eval_top_k_from_p4h]
    if not raw_topk:
        raise RuntimeError("p4h summary has empty topk")

    print("\n========== V4.12P4H2 CONTACT-SEQUENCE PROXY ==========")
    print("model           :", model_path)
    print("candidate       :", candidate_path)
    print("p3_json         :", p3_path)
    print("p4h_summary     :", p4h_summary_path)
    print("best_config     :", best_config_path)
    print("out_dir         :", out_dir)
    print("eval_top_k      :", len(raw_topk))
    print("alpha_samples   :", args.alpha_samples)
    print("candidate_source:", candidate_ctrl_source)
    print("best_source     :", best_ctrl_source)
    print("side_open_ctrl  :", side_open_ctrl)
    print("close_target    :", close_target)
    print("object_geom     :", object_geom)
    print("support_top     :", support_top, support_names)
    print("======================================================\n")

    records = []

    for i, r in enumerate(raw_topk, 1):
        d = r.get("delta_meta", {})
        if "delta_local" not in d:
            raise RuntimeError(f"topk[{i}] missing delta_meta.delta_local")

        # 保证 json 里的 delta_local 是 list[float]
        d = {
            "delta_local": [float(x) for x in d["delta_local"]],
            "yaw_deg": float(d.get("yaw_deg", 0.0)),
            "roll_deg": float(d.get("roll_deg", 0.0)),
        }

        ev = evaluate_one_delta(
            model=model,
            data=data,
            args=args,
            q_grasp=q_grasp,
            object_geom=object_geom,
            support_top=support_top,
            T_world_target_old=T_world_target_old,
            delta_meta=d,
            side_open_ctrl=side_open_ctrl,
            close_target=close_target,
        )

        ev["p4h_rank"] = i
        ev["p4h_score"] = float(r.get("score", 999.0))
        ev["p4h_original_record"] = r
        records.append(ev)

        print(
            f"[{i:02d}] score={ev['score']:.3f} "
            f"p4h_score={ev['p4h_score']:.3f} "
            f"delta={d['delta_local']} yaw={d['yaw_deg']} roll={d['roll_deg']} "
            f"first={ev['first_alpha']} "
            f"final_non_thumb={ev['final_good_non_thumb']} "
            f"reasons={ev['reasons']}"
        )

    ranked = sorted(records, key=lambda x: float(x["score"]))
    topk = ranked[:min(20, len(ranked))]
    best = topk[0]

    patched_candidate, patch_info = patch_candidate(candidate, model, data, args, q_grasp, best)

    best_candidate_path = out_dir / "best_candidate.json"
    best_config_out_path = out_dir / "best_contact_sequence_config.json"
    summary_path = out_dir / "summary.json"
    top_txt_path = out_dir / "topk_summary.txt"

    save_json(best_candidate_path, patched_candidate)

    best_config_out = {
        "format": "v4_12p4h2_contact_sequence_proxy_best_config",
        "best_record": {
            "hand_config": {
                "ctrl": dict(close_target),
            },
            "contact_sequence_proxy": best,
        },
        "candidate_ctrl_source": candidate_ctrl_source,
        "best_ctrl_source": best_ctrl_source,
        "candidate_ctrl": candidate_ctrl,
        "side_open_ctrl": side_open_ctrl,
        "close_target": close_target,
        "object_geom": object_geom,
        "support_top": support_top,
        "patch_info": patch_info,
        "best_candidate": str(best_candidate_path),
    }
    save_json(best_config_out_path, best_config_out)

    viewer_script = write_viewer_script(args, out_dir)

    summary = {
        "format": "v4_12p4h2_contact_sequence_proxy_debug",
        "model": str(model_path),
        "candidate": str(candidate_path),
        "p3_json": str(p3_path),
        "p4h_summary": str(p4h_summary_path),
        "best_config_in": str(best_config_path),
        "which": args.which,
        "object_body": args.object_body,
        "target_body": args.target_body,
        "args": vars(args),
        "candidate_ctrl_source": candidate_ctrl_source,
        "best_ctrl_source": best_ctrl_source,
        "candidate_ctrl": candidate_ctrl,
        "side_open_ctrl": side_open_ctrl,
        "close_target": close_target,
        "object_geom": object_geom,
        "support_top": support_top,
        "support_names": support_names,
        "num_records": len(records),
        "best_candidate": str(best_candidate_path),
        "best_contact_sequence_config": str(best_config_out_path),
        "viewer_script": str(viewer_script),
        "topk": topk,
    }
    save_json(summary_path, summary)

    with open(top_txt_path, "w") as f:
        f.write("rank,score,p4h_rank,p4h_score,delta,yaw,roll,first_thumb,first_index,first_middle,first_ring,first_pinky,final_non_thumb,reasons\n")
        for k, r in enumerate(topk, 1):
            d = r["delta_meta"]
            fa = r["first_alpha"]
            f.write(
                f"{k},"
                f"{r['score']:.6f},"
                f"{r['p4h_rank']},"
                f"{r['p4h_score']:.6f},"
                f"{d['delta_local']},"
                f"{d['yaw_deg']},"
                f"{d['roll_deg']},"
                f"{fa.get('thumb')},"
                f"{fa.get('index')},"
                f"{fa.get('middle')},"
                f"{fa.get('ring')},"
                f"{fa.get('pinky')},"
                f"{r['final_good_non_thumb']},"
                f"{r['reasons']}\n"
            )

    print("\n========== P4H2 SUMMARY ==========")
    print("summary      :", summary_path)
    print("top_txt      :", top_txt_path)
    print("best_candidate:", best_candidate_path)
    print("best_config  :", best_config_out_path)
    print("viewer_script:", viewer_script)

    for i, r in enumerate(topk[:10], 1):
        d = r["delta_meta"]
        print(
            f"{i:02d}. score={r['score']:.3f} "
            f"p4h_rank={r['p4h_rank']} "
            f"delta={d['delta_local']} yaw={d['yaw_deg']} roll={d['roll_deg']} "
            f"first={r['first_alpha']} "
            f"final_non_thumb={r['final_good_non_thumb']} "
            f"reasons={r['reasons']}"
        )

    print("\nBest viewer command:")
    print(f"bash {viewer_script}")
    print("==================================\n")


if __name__ == "__main__":
    main()
