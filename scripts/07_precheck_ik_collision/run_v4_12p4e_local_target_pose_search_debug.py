#!/usr/bin/env python3
"""
文件名：
    run_v4_12p4e_local_target_pose_search_debug.py

脚本类别：
    debug / local-search / candidate-refinement / precheck-runner

用途：
    本脚本用于 V4.12P4E 阶段。
    它不再使用 P4D 的“物体中心必须落在 thumb-four 中点”的硬约束，
    而是回到更稳妥的方式：
        1. 以当前 candidate 的 target.T_object_target 为基准；
        2. 在该目标位姿附近做小范围 xyz 平移扰动；
        3. 每个扰动 candidate 重新跑 P2 Pinocchio IK；
        4. 每个扰动 candidate 重新跑 P3 MuJoCo 碰撞/路径预检；
        5. 每个扰动 candidate 用 P4C 低/中幅闭合做非 viewer 接触测试；
        6. 根据 thumb + non-thumb 接触、hand-support、object displacement、闭合幅度等评分；
        7. 输出 Top-K 排名，并自动生成最好一次的 viewer 指令。

输入：
    1. --model
       当前 MuJoCo XML 场景。
    2. --candidate
       原始 candidate JSON。
    3. --runner-json
       P2 使用的 runner-json，用于提供参考初始状态。
    4. --urdf
       Pinocchio 使用的 URDF。
    5. --object-body
       物体 body 名，例如 grasp_can。
    6. --dx-list / --dy-list / --dz-list
       世界坐标下目标位姿扰动列表，单位 m。

输出：
    1. out-dir/variants/var_xxx/ 下的 candidate、P2、P3、P4C 结果。
    2. --out-summary 汇总 JSON。
    3. out-dir/topk_summary.txt 可读排行榜。
    4. out-dir/run_best_viewer.sh 最好一次的可视化指令。

当前流程位置：
    dataset/candidate prior
        -> P4E local target pose search
        -> 选出 best local pose
        -> viewer 观察 best
        -> 再进入 P4F/P5：固化局部优化和闭合策略

本脚本不负责：
    1. 不修改原始 candidate。
    2. 不修改物体位置。
    3. 不修改闭合控制器 P4C。
    4. 不做全局路径规划。
    5. 不保证一次搜索必然成功；它负责把局部扰动下“相对最好”的结果找出来。
"""

from pathlib import Path
import argparse
import csv
import json
import os
import shlex
import subprocess
import sys
import numpy as np
import mujoco


PROJECT = Path.home() / "Projects/o7_mujoco_sim"
RUN_CLEAN = PROJECT / "run_mujoco_clean.sh"

P2_SCRIPT = PROJECT / "scripts/07_precheck_ik_collision/run_v4_12p2_pinocchio_multiseed_ik_debug.py"
P3_SCRIPT = PROJECT / "scripts/07_precheck_ik_collision/run_v4_12p3_pinocchio_ik_plus_mujoco_collision_precheck_debug.py"
P4C_SCRIPT = PROJECT / "scripts/05_execution_runner/run_v4_12p4c_opposition_contact_seek_close_debug.py"


def resolve_path(p):
    p = Path(str(p)).expanduser()
    if not p.is_absolute():
        p = PROJECT / p
    return p


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


def parse_float_list(s):
    if s is None or str(s).strip() == "":
        return []
    return [float(x) for x in str(s).replace(",", " ").split()]


def T_inv(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def body_T_world(model, data, body_name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise RuntimeError(f"cannot find body: {body_name}")

    T = np.eye(4)
    T[:3, :3] = data.xmat[bid].reshape(3, 3)
    T[:3, 3] = data.xpos[bid]
    return T


def patch_candidate_by_world_delta(candidate, T_world_object, delta_world, out_path, variant_name):
    if "target" not in candidate or "T_object_target" not in candidate["target"]:
        raise RuntimeError("candidate missing target.T_object_target")

    T_object_target_old = np.asarray(candidate["target"]["T_object_target"], dtype=float)
    T_world_target_old = T_world_object @ T_object_target_old

    T_world_target_new = T_world_target_old.copy()
    T_world_target_new[:3, 3] += np.asarray(delta_world, dtype=float).reshape(3)

    T_object_target_new = T_inv(T_world_object) @ T_world_target_new

    patched = json.loads(json.dumps(candidate))
    patched["target"]["T_object_target"] = T_object_target_new.tolist()

    meta = patched.setdefault("debug_patch_meta", {})
    meta["v4_12p4e_local_target_pose_search"] = {
        "variant_name": variant_name,
        "delta_world": np.asarray(delta_world).tolist(),
        "T_object_target_old": T_object_target_old.tolist(),
        "T_object_target_new": T_object_target_new.tolist(),
    }

    save_json(out_path, patched)

    return {
        "T_object_target_old": T_object_target_old,
        "T_world_target_old": T_world_target_old,
        "T_world_target_new": T_world_target_new,
        "T_object_target_new": T_object_target_new,
    }


def run_cmd(cmd, log_path, cwd=PROJECT):
    log_path = resolve_path(log_path)
    log_path.parent.mkdir(parents=True, exist_ok=True)

    with open(log_path, "w") as f:
        f.write("$ " + " ".join(shlex.quote(str(x)) for x in cmd) + "\n\n")
        f.flush()

        proc = subprocess.run(
            [str(x) for x in cmd],
            cwd=str(cwd),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )

    return proc.returncode


def max_logged_object_disp(p4c):
    max_disp = 0.0
    max_phase = ""
    for row in p4c.get("logs", []):
        disp = row.get("object_disp", None)
        if disp is not None and float(disp) > max_disp:
            max_disp = float(disp)
            max_phase = row.get("phase", "")
    return max_disp, max_phase


def max_logged_contact(p4c, key):
    m = 0
    for row in p4c.get("logs", []):
        c = row.get("contacts", {})
        m = max(m, int(c.get(key, 0)))
    return m


def count_non_thumb(state):
    return sum(1 for g in ["index", "middle", "ring", "pinky"] if state.get(g, False))


def score_p4c_result(p4c, args):
    status = p4c.get("status", "UNKNOWN")
    obj_state = p4c.get("object_contact_state", {}) or {}
    sup_state = p4c.get("support_contact_state", {}) or {}
    final_ctrl = p4c.get("final_ctrl", {}) or {}

    thumb = bool(obj_state.get("thumb", False))
    non_thumb = count_non_thumb(obj_state)
    total_groups = sum(1 for g in ["thumb", "index", "middle", "ring", "pinky"] if obj_state.get(g, False))

    max_disp, max_disp_phase = max_logged_object_disp(p4c)
    max_hand_support = max_logged_contact(p4c, "hand_support")
    max_fr3_object = max_logged_contact(p4c, "fr3_object")

    finger_ctrl_max = max(
        float(final_ctrl.get(j, 0.0))
        for j in ["index_mcp_pitch", "middle_mcp_pitch", "ring_mcp_pitch", "pinky_mcp_pitch"]
    )
    thumb_pitch = float(final_ctrl.get("thumb_cmc_pitch", 0.0))

    score = 0.0
    reasons = []

    if thumb:
        score += 150.0
        reasons.append("thumb_contact")
    else:
        score -= 100.0
        reasons.append("no_thumb")

    if non_thumb > 0:
        score += 350.0 * non_thumb
        reasons.append(f"non_thumb={non_thumb}")
    else:
        score -= 250.0
        reasons.append("no_non_thumb")

    if thumb and non_thumb > 0:
        score += 500.0
        reasons.append("opposition_contact")

    score += 60.0 * total_groups

    if max_hand_support > 0:
        score -= 2000.0
        reasons.append(f"hand_support={max_hand_support}")

    if max_fr3_object > 0:
        score -= 1500.0
        reasons.append(f"fr3_object={max_fr3_object}")

    score -= max_disp * 20000.0
    if max_disp > args.hard_object_push_disp:
        score -= 1000.0
        reasons.append(f"hard_push_disp={max_disp:.5f}")
    elif max_disp > args.soft_object_push_disp:
        score -= 300.0
        reasons.append(f"soft_push_disp={max_disp:.5f}")

    # can 很细，允许中等闭合，但不希望靠 0.55~0.60 的深卷捞物体
    if finger_ctrl_max <= args.preferred_finger_max:
        score += 120.0
        reasons.append("finger_ctrl_reasonable")
    else:
        score -= (finger_ctrl_max - args.preferred_finger_max) * 1000.0
        reasons.append(f"finger_ctrl_too_large={finger_ctrl_max:.3f}")

    if "SUCCESS" in status:
        score += 300.0
        reasons.append(status)
    elif status == "FAIL_CONTACT_GOAL":
        score -= 80.0
    elif status == "FAIL_HARD_GUARD":
        score -= 700.0
    else:
        score -= 30.0

    return {
        "score": score,
        "reasons": reasons,
        "status": status,
        "thumb_contact": thumb,
        "non_thumb_contacts": non_thumb,
        "total_object_groups": total_groups,
        "support_state": sup_state,
        "max_object_disp": max_disp,
        "max_object_disp_phase": max_disp_phase,
        "max_hand_support": max_hand_support,
        "max_fr3_object": max_fr3_object,
        "finger_ctrl_max": finger_ctrl_max,
        "thumb_pitch": thumb_pitch,
    }


def read_json_if_exists(p):
    p = resolve_path(p)
    if not p.exists():
        return None
    try:
        return json.load(open(p))
    except Exception:
        return None


def shell_join(cmd):
    return " ".join(shlex.quote(str(x)) for x in cmd)


def build_p4c_cmd(args, model_path, cand_path, p3_path, out_json, viewer=False):
    cmd = [
        str(RUN_CLEAN),
        str(P4C_SCRIPT.relative_to(PROJECT)),
        "--model", str(Path(model_path).relative_to(PROJECT) if str(model_path).startswith(str(PROJECT)) else model_path),
        "--candidate", str(Path(cand_path).relative_to(PROJECT) if str(cand_path).startswith(str(PROJECT)) else cand_path),
        "--p3-json", str(Path(p3_path).relative_to(PROJECT) if str(p3_path).startswith(str(PROJECT)) else p3_path),
        "--which", "best_available",
        "--object-body", args.object_body,
        "--out", str(Path(out_json).relative_to(PROJECT) if str(out_json).startswith(str(PROJECT)) else out_json),
        "--move-steps", str(args.move_steps),
        "--thumb-preshape-steps", str(args.thumb_preshape_steps),
        "--thumb-roll-preshape", str(args.thumb_roll_preshape),
        "--thumb-yaw-preshape", str(args.thumb_yaw_preshape),
        "--thumb-pitch-open", str(args.thumb_pitch_open),
        "--finger-seek-duration", str(args.finger_seek_duration),
        "--thumb-comp-duration", str(args.thumb_comp_duration),
        "--micro-squeeze-duration", str(args.micro_squeeze_duration),
        "--hold-duration", str(args.hold_duration),
        "--lift-duration", str(args.lift_duration),
        "--finger-seek-speed", str(args.finger_seek_speed),
        "--thumb-comp-speed", str(args.thumb_comp_speed),
        "--micro-finger-speed", str(args.micro_finger_speed),
        "--micro-thumb-speed", str(args.micro_thumb_speed),
        "--soft-object-push-disp", str(args.soft_object_push_disp),
        "--hard-object-push-disp", str(args.hard_object_push_disp),
        "--micro-push-increase-limit", str(args.micro_push_increase_limit),
        "--min-total-object-groups", str(args.min_total_object_groups),
        "--min-non-thumb-groups", str(args.min_non_thumb_groups),
        "--min-lift-rise-success", str(args.min_lift_rise_success),
        "--frame-sleep", str(args.frame_sleep if viewer else 0.0),
    ]

    if args.no_fail_on_hand_support:
        cmd.append("--no-fail-on-hand-support")

    if args.lift_even_if_fail:
        cmd.append("--lift-even-if-fail")

    if viewer:
        cmd.append("--viewer")

    return cmd


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--runner-json", required=True)
    ap.add_argument("--urdf", required=True)
    ap.add_argument("--object-body", default="grasp_can")
    ap.add_argument("--target-frame", default="fr3_link7")

    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--out-summary", required=True)

    ap.add_argument("--dx-list", default="-0.012 0 0.012")
    ap.add_argument("--dy-list", default="-0.012 0 0.012")
    ap.add_argument("--dz-list", default="0 0.006")

    ap.add_argument("--max-variants", type=int, default=9999)

    ap.add_argument("--p2-random-seeds", type=int, default=8)
    ap.add_argument("--p2-random-std", type=float, default=0.6)
    ap.add_argument("--p2-max-iters", type=int, default=280)
    ap.add_argument("--p2-pos-tol", type=float, default=0.0003)
    ap.add_argument("--p2-rot-tol", type=float, default=0.003)
    ap.add_argument("--p2-rot-weight", type=float, default=0.55)

    ap.add_argument("--p3-top-per-target", type=int, default=5)
    ap.add_argument("--p3-max-combos", type=int, default=160)
    ap.add_argument("--p3-path-samples", type=int, default=28)
    ap.add_argument("--p3-min-hand-support-clearance", type=float, default=0.0)
    ap.add_argument("--p3-min-fr3-object-clearance", type=float, default=0.0)
    ap.add_argument("--p3-max-grasp-hand-object-distance", type=float, default=0.040)
    ap.add_argument("--p3-min-joint-margin", type=float, default=0.0)

    ap.add_argument("--move-steps", type=int, default=80)
    ap.add_argument("--thumb-preshape-steps", type=int, default=80)
    ap.add_argument("--thumb-roll-preshape", type=float, default=0.56)
    ap.add_argument("--thumb-yaw-preshape", type=float, default=1.15)
    ap.add_argument("--thumb-pitch-open", type=float, default=0.08)

    ap.add_argument("--finger-seek-duration", type=float, default=1.00)
    ap.add_argument("--thumb-comp-duration", type=float, default=1.20)
    ap.add_argument("--micro-squeeze-duration", type=float, default=0.0)
    ap.add_argument("--hold-duration", type=float, default=0.35)
    ap.add_argument("--lift-duration", type=float, default=0.8)

    ap.add_argument("--finger-seek-speed", type=float, default=0.35)
    ap.add_argument("--thumb-comp-speed", type=float, default=0.25)
    ap.add_argument("--micro-finger-speed", type=float, default=0.0)
    ap.add_argument("--micro-thumb-speed", type=float, default=0.0)

    ap.add_argument("--soft-object-push-disp", type=float, default=0.004)
    ap.add_argument("--hard-object-push-disp", type=float, default=0.012)
    ap.add_argument("--micro-push-increase-limit", type=float, default=0.001)

    ap.add_argument("--preferred-finger-max", type=float, default=0.42)
    ap.add_argument("--min-total-object-groups", type=int, default=2)
    ap.add_argument("--min-non-thumb-groups", type=int, default=1)
    ap.add_argument("--min-lift-rise-success", type=float, default=0.015)

    ap.add_argument("--no-fail-on-hand-support", action="store_true")
    ap.add_argument("--lift-even-if-fail", action="store_true")
    ap.add_argument("--frame-sleep", type=float, default=0.002)

    args = ap.parse_args()

    model_path = resolve_path(args.model)
    candidate_path = resolve_path(args.candidate)
    runner_json = resolve_path(args.runner_json)
    urdf_path = resolve_path(args.urdf)
    out_dir = resolve_path(args.out_dir)
    out_summary = resolve_path(args.out_summary)

    for p in [model_path, candidate_path, runner_json, urdf_path, RUN_CLEAN, P2_SCRIPT, P3_SCRIPT, P4C_SCRIPT]:
        if not p.exists():
            raise RuntimeError(f"missing required path: {p}")

    out_dir.mkdir(parents=True, exist_ok=True)
    variants_dir = out_dir / "variants"
    variants_dir.mkdir(parents=True, exist_ok=True)

    candidate = load_json(candidate_path)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    T_world_object = body_T_world(model, data, args.object_body)

    dxs = parse_float_list(args.dx_list)
    dys = parse_float_list(args.dy_list)
    dzs = parse_float_list(args.dz_list)

    deltas = []
    for dx in dxs:
        for dy in dys:
            for dz in dzs:
                deltas.append(np.array([dx, dy, dz], dtype=float))

    # 优先把 0,0,0 放在第一个，便于对照
    deltas.sort(key=lambda v: (0 if np.linalg.norm(v) < 1e-12 else 1, float(np.linalg.norm(v))))

    if args.max_variants > 0:
        deltas = deltas[:args.max_variants]

    print("\n========== V4.12P4E LOCAL TARGET POSE SEARCH ==========")
    print("model      :", model_path)
    print("candidate  :", candidate_path)
    print("runner_json:", runner_json)
    print("urdf       :", urdf_path)
    print("object_body:", args.object_body)
    print("out_dir    :", out_dir)
    print("num variants:", len(deltas))
    print("dx:", dxs)
    print("dy:", dys)
    print("dz:", dzs)
    print("=======================================================\n")

    records = []

    for i, delta in enumerate(deltas):
        variant_name = f"var_{i:03d}_dx{delta[0]*1000:+.0f}_dy{delta[1]*1000:+.0f}_dz{delta[2]*1000:+.0f}"
        vdir = variants_dir / variant_name
        vdir.mkdir(parents=True, exist_ok=True)

        cand_out = vdir / f"{variant_name}_candidate.json"
        p2_out = vdir / f"{variant_name}_p2.json"
        p3_out = vdir / f"{variant_name}_p3.json"
        best_plan_out = vdir / f"{variant_name}_best_plan.json"
        p4c_out = vdir / f"{variant_name}_p4c.json"

        p2_log = vdir / f"{variant_name}_p2.txt"
        p3_log = vdir / f"{variant_name}_p3.txt"
        p4c_log = vdir / f"{variant_name}_p4c.txt"

        print(f"\n------------------------------------------------")
        print(f"[{i+1:03d}/{len(deltas):03d}] {variant_name} delta={delta}")
        print(f"------------------------------------------------")

        rec = {
            "variant": variant_name,
            "index": i,
            "delta_world": delta.tolist(),
            "candidate": str(cand_out),
            "p2_json": str(p2_out),
            "p3_json": str(p3_out),
            "best_plan_json": str(best_plan_out),
            "p4c_json": str(p4c_out),
            "p2_log": str(p2_log),
            "p3_log": str(p3_log),
            "p4c_log": str(p4c_log),
        }

        try:
            patch_info = patch_candidate_by_world_delta(
                candidate=candidate,
                T_world_object=T_world_object,
                delta_world=delta,
                out_path=cand_out,
                variant_name=variant_name,
            )
            rec["patch_info"] = {
                "target_pos_old_world": patch_info["T_world_target_old"][:3, 3],
                "target_pos_new_world": patch_info["T_world_target_new"][:3, 3],
            }

            p2_cmd = [
                "python3", str(P2_SCRIPT.relative_to(PROJECT)),
                "--urdf", str(urdf_path.relative_to(PROJECT)),
                "--model", str(model_path.relative_to(PROJECT)),
                "--candidate", str(cand_out.relative_to(PROJECT)),
                "--runner-json", str(runner_json.relative_to(PROJECT)),
                "--object-body", args.object_body,
                "--target-frame", args.target_frame,
                "--out", str(p2_out.relative_to(PROJECT)),
                "--random-seeds", str(args.p2_random_seeds),
                "--random-std", str(args.p2_random_std),
                "--max-iters", str(args.p2_max_iters),
                "--pos-tol", str(args.p2_pos_tol),
                "--rot-tol", str(args.p2_rot_tol),
                "--rot-weight", str(args.p2_rot_weight),
            ]
            rc = run_cmd(p2_cmd, p2_log)
            rec["p2_returncode"] = rc
            if rc != 0:
                rec["error"] = "P2 failed"
                rec["score_info"] = {"score": -1e9, "reasons": ["P2 failed"]}
                records.append(rec)
                print("[FAIL] P2 failed")
                continue

            p3_cmd = [
                "python3", str(P3_SCRIPT.relative_to(PROJECT)),
                "--p2-json", str(p2_out.relative_to(PROJECT)),
                "--model", str(model_path.relative_to(PROJECT)),
                "--candidate", str(cand_out.relative_to(PROJECT)),
                "--object-body", args.object_body,
                "--out", str(p3_out.relative_to(PROJECT)),
                "--best-plan-out", str(best_plan_out.relative_to(PROJECT)),
                "--top-per-target", str(args.p3_top_per_target),
                "--max-combos", str(args.p3_max_combos),
                "--path-samples", str(args.p3_path_samples),
                "--min-hand-support-clearance", str(args.p3_min_hand_support_clearance),
                "--min-fr3-object-clearance", str(args.p3_min_fr3_object_clearance),
                "--max-grasp-hand-object-distance", str(args.p3_max_grasp_hand_object_distance),
                "--min-joint-margin", str(args.p3_min_joint_margin),
            ]
            rc = run_cmd(p3_cmd, p3_log)
            rec["p3_returncode"] = rc
            if rc != 0:
                rec["error"] = "P3 failed"
                rec["score_info"] = {"score": -1e9, "reasons": ["P3 failed"]}
                records.append(rec)
                print("[FAIL] P3 failed")
                continue

            p4c_cmd = build_p4c_cmd(
                args=args,
                model_path=model_path,
                cand_path=cand_out,
                p3_path=p3_out,
                out_json=p4c_out,
                viewer=False,
            )
            rc = run_cmd(p4c_cmd, p4c_log)
            rec["p4c_returncode"] = rc
            if rc != 0:
                rec["error"] = "P4C failed"
                rec["score_info"] = {"score": -1e9, "reasons": ["P4C failed"]}
                records.append(rec)
                print("[FAIL] P4C failed")
                continue

            p4c = read_json_if_exists(p4c_out)
            if not p4c:
                rec["error"] = "P4C json missing or invalid"
                rec["score_info"] = {"score": -1e9, "reasons": ["P4C json missing"]}
                records.append(rec)
                print("[FAIL] P4C json missing")
                continue

            score_info = score_p4c_result(p4c, args)
            rec["score_info"] = score_info

            print(
                f"[SCORE] {score_info['score']:+.2f} "
                f"status={score_info['status']} "
                f"thumb={score_info['thumb_contact']} "
                f"non_thumb={score_info['non_thumb_contacts']} "
                f"disp={score_info['max_object_disp']:.5f} "
                f"finger_max={score_info['finger_ctrl_max']:.3f} "
                f"reasons={score_info['reasons']}"
            )

        except Exception as e:
            rec["error"] = repr(e)
            rec["score_info"] = {"score": -1e9, "reasons": [repr(e)]}
            print("[EXCEPTION]", repr(e))

        records.append(rec)
        save_json(out_dir / "partial_summary.json", {"records": records})

    records_sorted = sorted(
        records,
        key=lambda r: float((r.get("score_info") or {}).get("score", -1e9)),
        reverse=True,
    )

    summary = {
        "format": "v4_12p4e_local_target_pose_search_debug",
        "model": str(model_path),
        "candidate": str(candidate_path),
        "runner_json": str(runner_json),
        "urdf": str(urdf_path),
        "object_body": args.object_body,
        "args": vars(args),
        "records": records,
        "topk": records_sorted[:10],
    }
    save_json(out_summary, summary)

    top_txt = out_dir / "topk_summary.txt"
    with open(top_txt, "w") as f:
        f.write("rank,score,variant,delta_world,status,thumb,non_thumb,max_disp,finger_ctrl_max,reasons\n")
        for rank, r in enumerate(records_sorted[:20], 1):
            si = r.get("score_info") or {}
            f.write(
                f"{rank},"
                f"{si.get('score', -1e9):+.2f},"
                f"{r.get('variant')},"
                f"{r.get('delta_world')},"
                f"{si.get('status')},"
                f"{si.get('thumb_contact')},"
                f"{si.get('non_thumb_contacts')},"
                f"{si.get('max_object_disp')},"
                f"{si.get('finger_ctrl_max')},"
                f"{si.get('reasons')}\n"
            )

    csv_path = out_dir / "topk_summary.csv"
    with open(csv_path, "w", newline="") as f:
        writer = csv.writer(f)
        writer.writerow([
            "rank", "score", "variant", "dx", "dy", "dz",
            "status", "thumb", "non_thumb", "max_disp", "finger_ctrl_max",
            "candidate", "p3_json", "p4c_json", "reasons"
        ])
        for rank, r in enumerate(records_sorted, 1):
            si = r.get("score_info") or {}
            d = r.get("delta_world", [None, None, None])
            writer.writerow([
                rank, si.get("score", -1e9), r.get("variant"),
                d[0], d[1], d[2],
                si.get("status"), si.get("thumb_contact"), si.get("non_thumb_contacts"),
                si.get("max_object_disp"), si.get("finger_ctrl_max"),
                r.get("candidate"), r.get("p3_json"), r.get("p4c_json"),
                si.get("reasons"),
            ])

    if records_sorted:
        best = records_sorted[0]
        best_viewer_out = out_dir / "best_viewer_result.json"
        viewer_cmd = build_p4c_cmd(
            args=args,
            model_path=model_path,
            cand_path=resolve_path(best["candidate"]),
            p3_path=resolve_path(best["p3_json"]),
            out_json=best_viewer_out,
            viewer=True,
        )

        viewer_script = out_dir / "run_best_viewer.sh"
        with open(viewer_script, "w") as f:
            f.write("#!/usr/bin/env bash\n")
            f.write("cd ~/Projects/o7_mujoco_sim\n")
            f.write("source ~/mujoco_env/bin/activate\n")
            f.write(shell_join(viewer_cmd))
            f.write(" 2>&1 | tee ")
            f.write(shlex.quote(str((out_dir / "best_viewer_result.txt").relative_to(PROJECT))))
            f.write("\n")
        os.chmod(viewer_script, 0o755)

    print("\n========== P4E SEARCH SUMMARY ==========")
    print("summary:", out_summary)
    print("top txt:", top_txt)
    print("top csv:", csv_path)

    for rank, r in enumerate(records_sorted[:10], 1):
        si = r.get("score_info") or {}
        print(
            f"{rank:02d}. score={si.get('score', -1e9):+.2f} "
            f"variant={r.get('variant')} "
            f"delta={r.get('delta_world')} "
            f"status={si.get('status')} "
            f"thumb={si.get('thumb_contact')} "
            f"non_thumb={si.get('non_thumb_contacts')} "
            f"disp={si.get('max_object_disp')} "
            f"finger={si.get('finger_ctrl_max')} "
            f"reasons={si.get('reasons')}"
        )

    if records_sorted:
        print("\nBest viewer command:")
        print(f"bash {out_dir / 'run_best_viewer.sh'}")

    print("========================================\n")


if __name__ == "__main__":
    main()
