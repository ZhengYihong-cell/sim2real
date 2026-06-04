#!/usr/bin/env python3
"""
文件名：
    run_v4_12p4g_thumb_only_refine_debug.py

脚本类别：
    debug / runner / thumb-only-refinement / local-sweep / viewer

用途：
    本脚本用于 V4.12P4G 阶段。
    当前 P4F 已经证明四指可以闭合到 candidate prior，
    但可视化显示大拇指接触点偏高，导致 lift 时接触容易丢失。
    因此本脚本只围绕大拇指做小范围局部修正，不重新做整臂 IK，不重新做 P3 碰撞搜索。

核心流程：
    1. 读取 P3 best_available / best_pass 中的 q_pre / q_grasp / q_lift。
    2. 读取 candidate hand.o7_active_ctrl 作为四指闭合目标。
    3. 读取 P4E-fast best_config 中的大拇指初始 preshape。
    4. 对 thumb_cmc_roll / thumb_cmc_yaw / thumb_cmc_pitch 做小范围偏移扫描。
    5. 每个组合只跑当前轨迹上的 close + hold，快速统计：
          - thumb 是否接触物体
          - 是否至少有一个非拇指接触物体
          - thumb 接触点高度是否偏高
          - 物体是否被明显推走
          - hand_support / fr3_object 是否出现
    6. 输出 Top-K。
    7. 自动生成 best 一次的 viewer 指令，可选执行 lift。

输入：
    --model
        MuJoCo XML 场景。
    --candidate
        原始 candidate JSON。
    --p3-json
        P3 输出 JSON。
    --best-config
        P4E-fast 输出 best_config.json。
    --object-body
        物体 body 名称，例如 grasp_can。

输出：
    --out-dir/summary.json
        大拇指小范围搜索结果。
    --out-dir/topk_summary.txt
        可读排行榜。
    --out-dir/run_best_thumb_viewer.sh
        最优大拇指参数的 viewer 指令。
    --out-dir/best_thumb_config.json
        最优偏移参数。

当前流程位置：
    P4E-fast / P4F target-close
        -> P4G thumb-only local refinement
        -> best viewer
        -> 后续若可行，再固化进 online controller

本脚本不负责：
    1. 不重新做整臂 IK。
    2. 不重新搜索 handbase 大位姿。
    3. 不改变四指 candidate prior 目标。
    4. 不写死绝对手型；它只在 candidate/best_config 的基础上加小偏移。
    5. 不保证一次搜索必然抓取成功，只负责快速验证 thumb 局部修正方向。
"""

from pathlib import Path
import argparse
import importlib.util
import json
import math
import os
import shlex
import time
import numpy as np
import mujoco

try:
    import mujoco.viewer
except Exception:
    mujoco.viewer = None


PROJECT = Path.home() / "Projects/o7_mujoco_sim"
P4F_PATH = PROJECT / "scripts/05_execution_runner/run_v4_12p4f_target_close_debug.py"
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

FOUR_FINGER_JOINTS = [
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]


def load_p4f():
    spec = importlib.util.spec_from_file_location("p4f", str(P4F_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p4f = load_p4f()


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


def parse_float_list(s):
    return [float(x) for x in str(s).replace(",", " ").split() if x.strip()]


def shell_join(cmd):
    return " ".join(shlex.quote(str(x)) for x in cmd)


def object_center_z(model, data, object_body):
    bid = p4f.body_id(model, object_body)
    return float(data.xpos[bid][2])


def object_pos(model, data, object_body):
    bid = p4f.body_id(model, object_body)
    return data.xpos[bid].copy()


def contact_detail(model, data, sets):
    object_geoms = sets["object_geoms"]
    hand_geoms = sets["hand_geoms"]
    support_geoms = sets["support_geoms"]
    fr3_geoms = sets["fr3_geoms"]

    object_groups = {}
    support_groups = {}
    thumb_contact_positions = []
    non_thumb_contact_positions = []
    pairs = []

    hand_object = 0
    hand_support = 0
    fr3_object = 0
    object_support = 0

    for i in range(data.ncon):
        c = data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)

        g1_obj = g1 in object_geoms
        g2_obj = g2 in object_geoms
        g1_hand = g1 in hand_geoms
        g2_hand = g2 in hand_geoms
        g1_sup = g1 in support_geoms
        g2_sup = g2 in support_geoms
        g1_fr3 = g1 in fr3_geoms
        g2_fr3 = g2 in fr3_geoms

        if (g1_hand and g2_obj) or (g2_hand and g1_obj):
            hand_object += 1
            hg = g1 if g1_hand else g2
            group = p4f.group_of_geom(model, hg)
            if group:
                object_groups[group] = object_groups.get(group, 0) + 1
                if group == "thumb":
                    thumb_contact_positions.append(np.array(c.pos, dtype=float).copy())
                else:
                    non_thumb_contact_positions.append(np.array(c.pos, dtype=float).copy())

        if (g1_hand and g2_sup) or (g2_hand and g1_sup):
            hand_support += 1
            hg = g1 if g1_hand else g2
            group = p4f.group_of_geom(model, hg)
            if group:
                support_groups[group] = support_groups.get(group, 0) + 1

        if (g1_fr3 and g2_obj) or (g2_fr3 and g1_obj):
            fr3_object += 1

        if (g1_obj and g2_sup) or (g2_obj and g1_sup):
            object_support += 1

        pairs.append({
            "geom1": p4f.geom_name(model, g1),
            "body1": p4f.geom_body_name(model, g1),
            "geom2": p4f.geom_name(model, g2),
            "body2": p4f.geom_body_name(model, g2),
            "dist": float(c.dist),
            "pos": np.array(c.pos, dtype=float).copy(),
        })

    return {
        "ncon": int(data.ncon),
        "hand_object": hand_object,
        "hand_support": hand_support,
        "fr3_object": fr3_object,
        "object_support": object_support,
        "object_groups": object_groups,
        "support_groups": support_groups,
        "thumb_contact_positions": thumb_contact_positions,
        "non_thumb_contact_positions": non_thumb_contact_positions,
        "pairs": pairs,
    }


def make_ctrls(model, candidate, best_config, args, offsets):
    candidate_ctrl, candidate_source = p4f.extract_candidate_ctrl(candidate, model)
    best_ctrl, best_source = p4f.extract_best_config_ctrl(best_config, model)

    open_ctrl = p4f.make_open_ctrl(model)
    side_open = p4f.make_side_open_ctrl(model, open_ctrl, candidate_ctrl, best_ctrl, args)

    side_open["thumb_cmc_roll"] = p4f.clamp_ctrl(
        model, "thumb_cmc_roll",
        side_open["thumb_cmc_roll"] + offsets["thumb_roll_offset"],
    )
    side_open["thumb_cmc_yaw"] = p4f.clamp_ctrl(
        model, "thumb_cmc_yaw",
        side_open["thumb_cmc_yaw"] + offsets["thumb_yaw_offset"],
    )
    side_open["thumb_cmc_pitch"] = p4f.clamp_ctrl(
        model, "thumb_cmc_pitch",
        side_open["thumb_cmc_pitch"] + offsets["thumb_pitch_open_offset"],
    )

    close_target = p4f.make_close_target(model, side_open, candidate_ctrl, args)
    close_target["thumb_cmc_roll"] = side_open["thumb_cmc_roll"]
    close_target["thumb_cmc_yaw"] = side_open["thumb_cmc_yaw"]
    close_target["thumb_cmc_pitch"] = p4f.clamp_ctrl(
        model, "thumb_cmc_pitch",
        close_target["thumb_cmc_pitch"] + offsets["thumb_pitch_close_offset"],
    )

    return {
        "candidate_ctrl": candidate_ctrl,
        "candidate_source": candidate_source,
        "best_ctrl": best_ctrl,
        "best_source": best_source,
        "open_ctrl": open_ctrl,
        "side_open_ctrl": side_open,
        "close_target": close_target,
    }


def interp_dict(a, b, alpha, keys):
    out = {}
    for k in keys:
        av = float(a.get(k, 0.0))
        bv = float(b.get(k, av))
        out[k] = av + alpha * (bv - av)
    return out


def simulate_one(model, args, plan, ctrl_pack, offsets, viewer=None, do_lift=False, tag="eval", data=None):
    if data is None:
        data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    q_pre = plan["q_pre"]
    q_grasp = plan["q_grasp"]
    q_lift = plan["q_lift"]

    q_current = {j: p4f.get_joint_qpos(model, data, j) or 0.0 for j in ARM_JOINTS}

    sets = p4f.build_geom_sets(model, args.object_body)
    obj_ref = object_pos(model, data, args.object_body)
    obj_z_ref = object_center_z(model, data, args.object_body)

    logs = []

    def run_steps(label, steps, callback):
        log_every = max(1, int(args.log_dt / model.opt.timestep))
        for k in range(steps + 1):
            alpha = k / max(1, steps)
            t = k * float(model.opt.timestep)

            callback(alpha, t, k)
            mujoco.mj_forward(model, data)

            if k % log_every == 0 or k == steps:
                obj = object_pos(model, data, args.object_body)
                detail = contact_detail(model, data, sets)
                logs.append({
                    "phase": label,
                    "step": int(k),
                    "time": float(t),
                    "alpha": float(alpha),
                    "object_pos": obj,
                    "object_disp": float(np.linalg.norm(obj - obj_ref)),
                    "object_center_z": object_center_z(model, data, args.object_body),
                    "contact_detail": detail,
                    "hand_qpos": p4f.hand_qpos(model, data),
                })

                if viewer is not None:
                    print(
                        f"[{label}] {k:4d}/{steps} "
                        f"disp={np.linalg.norm(obj - obj_ref):.5f} "
                        f"groups={detail['object_groups']} "
                        f"hand_support={detail['hand_support']} "
                        f"object_support={detail['object_support']}"
                    )

            mujoco.mj_step(model, data)

            if viewer is not None:
                viewer.sync()
                if args.frame_sleep > 0:
                    time.sleep(args.frame_sleep)

    run_steps(
        "move_to_pre_open",
        args.move_steps,
        lambda a, t, k: (
            p4f.set_arm_qpos_and_ctrl(model, data, interp_dict(q_current, q_pre, a, ARM_JOINTS)),
            p4f.set_hand_ctrl(model, data, ctrl_pack["open_ctrl"], args.direct_hand_qpos),
        ),
    )

    run_steps(
        "thumb_preshape",
        args.thumb_preshape_steps,
        lambda a, t, k: (
            p4f.set_arm_qpos_and_ctrl(model, data, q_pre),
            p4f.set_hand_ctrl(
                model, data,
                interp_dict(ctrl_pack["open_ctrl"], ctrl_pack["side_open_ctrl"], a, HAND_JOINTS),
                args.direct_hand_qpos,
            ),
        ),
    )

    run_steps(
        "move_to_grasp_side_open",
        args.move_steps,
        lambda a, t, k: (
            p4f.set_arm_qpos_and_ctrl(model, data, interp_dict(q_pre, q_grasp, a, ARM_JOINTS)),
            p4f.set_hand_ctrl(model, data, ctrl_pack["side_open_ctrl"], args.direct_hand_qpos),
        ),
    )

    run_steps(
        "target_close",
        max(1, int(args.close_duration / model.opt.timestep)),
        lambda a, t, k: (
            p4f.set_arm_qpos_and_ctrl(model, data, q_grasp),
            p4f.set_hand_ctrl(
                model, data,
                interp_dict(ctrl_pack["side_open_ctrl"], ctrl_pack["close_target"], a, HAND_JOINTS),
                args.direct_hand_qpos,
            ),
        ),
    )

    run_steps(
        "hold_after_close",
        max(1, int(args.hold_duration / model.opt.timestep)),
        lambda a, t, k: (
            p4f.set_arm_qpos_and_ctrl(model, data, q_grasp),
            p4f.set_hand_ctrl(model, data, ctrl_pack["close_target"], args.direct_hand_qpos),
        ),
    )

    hold_logs = [r for r in logs if r["phase"] == "hold_after_close"]
    final_hold = hold_logs[-1] if hold_logs else logs[-1]
    final_detail = final_hold["contact_detail"]
    object_groups = final_detail["object_groups"]

    has_thumb = "thumb" in object_groups
    non_thumb_groups = [g for g in ["index", "middle", "ring", "pinky"] if g in object_groups]
    has_non_thumb = len(non_thumb_groups) > 0

    thumb_positions = []
    for r in hold_logs:
        thumb_positions.extend(r["contact_detail"]["thumb_contact_positions"])

    if thumb_positions:
        thumb_z = float(np.mean([p[2] for p in thumb_positions]))
        thumb_z_min = float(np.min([p[2] for p in thumb_positions]))
        thumb_z_max = float(np.max([p[2] for p in thumb_positions]))
    else:
        thumb_z = None
        thumb_z_min = None
        thumb_z_max = None

    object_z_mid = float(np.mean([r["object_center_z"] for r in hold_logs])) if hold_logs else obj_z_ref
    thumb_high_error = 0.0 if thumb_z is None else max(0.0, thumb_z - (object_z_mid + args.thumb_target_z_bias))
    thumb_abs_z_error = 999.0 if thumb_z is None else abs(thumb_z - (object_z_mid + args.thumb_target_z_bias))

    max_disp = max(float(r["object_disp"]) for r in logs)
    final_disp = float(final_hold["object_disp"])

    score = 0.0
    reasons = []

    if has_thumb:
        score += 300.0
        reasons.append("thumb")
    else:
        score -= 500.0
        reasons.append("no_thumb")

    if has_non_thumb:
        score += 600.0 + 120.0 * len(non_thumb_groups)
        reasons.append("non_thumb=" + ",".join(non_thumb_groups))
    else:
        score -= 450.0
        reasons.append("no_non_thumb")

    score -= 18000.0 * max_disp
    score -= 12000.0 * final_disp
    score -= 900.0 * thumb_high_error
    score -= 250.0 * thumb_abs_z_error

    if final_detail["hand_support"] > 0:
        score -= 1500.0
        reasons.append(f"hand_support={final_detail['hand_support']}")

    if final_detail["fr3_object"] > 0:
        score -= 1200.0
        reasons.append(f"fr3_object={final_detail['fr3_object']}")

    if max_disp > args.hard_object_push_disp:
        score -= 900.0
        reasons.append(f"hard_push={max_disp:.4f}")

    if do_lift:
        run_steps(
            "lift",
            max(1, int(args.lift_duration / model.opt.timestep)),
            lambda a, t, k: (
                p4f.set_arm_qpos_and_ctrl(model, data, interp_dict(q_grasp, q_lift, a, ARM_JOINTS)),
                p4f.set_hand_ctrl(model, data, ctrl_pack["close_target"], args.direct_hand_qpos),
            ),
        )

    final_counts = contact_detail(model, data, sets)
    final_obj = object_pos(model, data, args.object_body)
    final_rise = float(final_obj[2] - obj_ref[2])

    return {
        "tag": tag,
        "offsets": offsets,
        "score": float(score),
        "reasons": reasons,
        "has_thumb": bool(has_thumb),
        "has_non_thumb": bool(has_non_thumb),
        "non_thumb_groups": non_thumb_groups,
        "object_groups_hold": object_groups,
        "thumb_contact_z_mean": thumb_z,
        "thumb_contact_z_min": thumb_z_min,
        "thumb_contact_z_max": thumb_z_max,
        "object_z_mid_hold": object_z_mid,
        "thumb_high_error": thumb_high_error,
        "thumb_abs_z_error": thumb_abs_z_error,
        "max_object_disp": max_disp,
        "final_hold_disp": final_disp,
        "final_hold_detail": final_detail,
        "final_counts": final_counts,
        "final_rise": final_rise,
        "final_hand_qpos": p4f.hand_qpos(model, data),
        "ctrl_pack": ctrl_pack,
        "logs": logs if args.save_logs or viewer is not None else [],
    }


def write_best_viewer_script(args, out_dir, best):
    out_dir = resolve_path(out_dir)
    script = out_dir / "run_best_thumb_viewer.sh"

    o = best["offsets"]

    cmd = [
        str(RUN_CLEAN),
        "scripts/05_execution_runner/run_v4_12p4g_thumb_only_refine_debug.py",
        "--model", args.model,
        "--candidate", args.candidate,
        "--p3-json", args.p3_json,
        "--best-config", args.best_config,
        "--which", args.which,
        "--object-body", args.object_body,
        "--out-dir", rel(out_dir / "best_viewer"),
        "--eval-single",
        "--viewer",
        "--do-lift",
        "--keep-viewer-open",
        "--thumb-roll-offset", str(o["thumb_roll_offset"]),
        "--thumb-yaw-offset", str(o["thumb_yaw_offset"]),
        "--thumb-pitch-open-offset", str(o["thumb_pitch_open_offset"]),
        "--thumb-pitch-close-offset", str(o["thumb_pitch_close_offset"]),
        "--finger-close-scale", str(args.finger_close_scale),
        "--thumb-pitch-from-finger-gain", str(args.thumb_pitch_from_finger_gain),
        "--move-steps", str(args.move_steps),
        "--thumb-preshape-steps", str(args.thumb_preshape_steps),
        "--close-duration", str(args.close_duration),
        "--hold-duration", str(args.hold_duration),
        "--lift-duration", str(args.lift_duration),
        "--hard-object-push-disp", str(args.hard_object_push_disp),
        "--frame-sleep", str(args.frame_sleep),
        "--log-dt", str(args.log_dt),
        "--save-logs",
    ]

    if args.direct_hand_qpos:
        cmd.append("--direct-hand-qpos")
    if args.preshape_fingers_from_best:
        cmd.append("--preshape-fingers-from-best")

    with open(script, "w") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -e\n")
        f.write("cd ~/Projects/o7_mujoco_sim\n")
        f.write("source ~/mujoco_env/bin/activate\n")
        f.write(shell_join(cmd))
        f.write(" 2>&1 | tee ")
        f.write(shlex.quote(rel(out_dir / "best_thumb_viewer.txt")))
        f.write("\n")

    os.chmod(script, 0o755)
    return script


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--p3-json", required=True)
    ap.add_argument("--best-config", required=True)
    ap.add_argument("--which", default="best_available", choices=["best_available", "best_pass"])
    ap.add_argument("--object-body", default="grasp_can")
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--thumb-roll-offset-list", default="-0.08 -0.04 0 0.04")
    ap.add_argument("--thumb-yaw-offset-list", default="-0.10 -0.05 0 0.05")
    ap.add_argument("--thumb-pitch-open-offset-list", default="-0.02 0")
    ap.add_argument("--thumb-pitch-close-offset-list", default="-0.04 -0.02 0 0.02")

    ap.add_argument("--eval-single", action="store_true")
    ap.add_argument("--thumb-roll-offset", type=float, default=0.0)
    ap.add_argument("--thumb-yaw-offset", type=float, default=0.0)
    ap.add_argument("--thumb-pitch-open-offset", type=float, default=0.0)
    ap.add_argument("--thumb-pitch-close-offset", type=float, default=0.0)

    ap.add_argument("--top-k", type=int, default=12)

    ap.add_argument("--move-steps", type=int, default=80)
    ap.add_argument("--thumb-preshape-steps", type=int, default=80)
    ap.add_argument("--close-duration", type=float, default=1.8)
    ap.add_argument("--hold-duration", type=float, default=0.7)
    ap.add_argument("--lift-duration", type=float, default=2.0)

    ap.add_argument("--finger-close-scale", type=float, default=1.0)
    ap.add_argument("--thumb-pitch-from-finger-gain", type=float, default=0.35)
    ap.add_argument("--preshape-fingers-from-best", action="store_true")
    ap.add_argument("--direct-hand-qpos", action="store_true")

    ap.add_argument("--thumb-target-z-bias", type=float, default=-0.006)
    ap.add_argument("--hard-object-push-disp", type=float, default=0.020)

    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--do-lift", action="store_true")
    ap.add_argument("--keep-viewer-open", action="store_true")
    ap.add_argument("--save-logs", action="store_true")

    ap.add_argument("--log-dt", type=float, default=0.12)
    ap.add_argument("--frame-sleep", type=float, default=0.002)

    args = ap.parse_args()

    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolve_path(args.model)
    candidate_path = resolve_path(args.candidate)
    p3_path = resolve_path(args.p3_json)
    best_config_path = resolve_path(args.best_config)

    for p in [model_path, candidate_path, p3_path, best_config_path, P4F_PATH]:
        if not p.exists():
            raise RuntimeError(f"missing path: {p}")

    candidate = load_json(candidate_path)
    p3 = load_json(p3_path)
    best_config = load_json(best_config_path)
    plan = p4f.selected_plan(p3, args.which)

    model = mujoco.MjModel.from_xml_path(str(model_path))

    if args.eval_single:
        offsets = {
            "thumb_roll_offset": args.thumb_roll_offset,
            "thumb_yaw_offset": args.thumb_yaw_offset,
            "thumb_pitch_open_offset": args.thumb_pitch_open_offset,
            "thumb_pitch_close_offset": args.thumb_pitch_close_offset,
        }

        ctrl_pack = make_ctrls(model, candidate, best_config, args, offsets)

        print("\n========== V4.12P4G SINGLE THUMB VIEWER/EVAL ==========")
        print("offsets:", offsets)
        print("side_open_ctrl:", ctrl_pack["side_open_ctrl"])
        print("close_target:", ctrl_pack["close_target"])
        print("=======================================================\n")

        if args.viewer:
            if mujoco.viewer is None:
                raise RuntimeError("mujoco.viewer not available")
            viewer_data = mujoco.MjData(model)
            with mujoco.viewer.launch_passive(model, viewer_data) as viewer:
                result = simulate_one(
                    model=model,
                    args=args,
                    plan=plan,
                    ctrl_pack=ctrl_pack,
                    offsets=offsets,
                    viewer=viewer,
                    data=viewer_data,
                    do_lift=args.do_lift,
                    tag="single_viewer",
                )
                if args.keep_viewer_open:
                    print("[VIEWER] 播放完成。关闭窗口即可退出。")
                    while viewer.is_running():
                        viewer.sync()
                        time.sleep(0.05)
        else:
            result = simulate_one(
                model=model,
                args=args,
                plan=plan,
                ctrl_pack=ctrl_pack,
                offsets=offsets,
                viewer=None,
                do_lift=args.do_lift,
                tag="single_eval",
            )

        save_json(out_dir / "single_result.json", result)

        print("\n========== P4G SINGLE RESULT ==========")
        print("score:", result["score"])
        print("reasons:", result["reasons"])
        print("object_groups_hold:", result["object_groups_hold"])
        print("thumb_contact_z_mean:", result["thumb_contact_z_mean"])
        print("object_z_mid_hold:", result["object_z_mid_hold"])
        print("max_object_disp:", result["max_object_disp"])
        print("final_rise:", result["final_rise"])
        print("saved:", out_dir / "single_result.json")
        print("=======================================\n")
        return

    roll_list = parse_float_list(args.thumb_roll_offset_list)
    yaw_list = parse_float_list(args.thumb_yaw_offset_list)
    pitch_open_list = parse_float_list(args.thumb_pitch_open_offset_list)
    pitch_close_list = parse_float_list(args.thumb_pitch_close_offset_list)

    print("\n========== V4.12P4G THUMB-ONLY LOCAL SWEEP ==========")
    print("model      :", model_path)
    print("candidate  :", candidate_path)
    print("p3_json    :", p3_path)
    print("best_config:", best_config_path)
    print("which      :", args.which)
    print("object_body:", args.object_body)
    print("out_dir    :", out_dir)
    print("roll_list  :", roll_list)
    print("yaw_list   :", yaw_list)
    print("popen_list :", pitch_open_list)
    print("pclose_list:", pitch_close_list)
    print("num combos :", len(roll_list) * len(yaw_list) * len(pitch_open_list) * len(pitch_close_list))
    print("====================================================\n")

    records = []
    idx = 0

    for ro in roll_list:
        for yo in yaw_list:
            for po in pitch_open_list:
                for pc in pitch_close_list:
                    idx += 1
                    offsets = {
                        "thumb_roll_offset": float(ro),
                        "thumb_yaw_offset": float(yo),
                        "thumb_pitch_open_offset": float(po),
                        "thumb_pitch_close_offset": float(pc),
                    }

                    ctrl_pack = make_ctrls(model, candidate, best_config, args, offsets)

                    result = simulate_one(
                        model=model,
                        args=args,
                        plan=plan,
                        ctrl_pack=ctrl_pack,
                        offsets=offsets,
                        viewer=None,
                        do_lift=False,
                        tag=f"sweep_{idx:03d}",
                    )

                    records.append(result)

                    print(
                        f"[{idx:03d}] score={result['score']:+.3f} "
                        f"offs={offsets} "
                        f"groups={result['object_groups_hold']} "
                        f"thumb_z={result['thumb_contact_z_mean']} "
                        f"obj_z={result['object_z_mid_hold']:.5f} "
                        f"disp={result['max_object_disp']:.5f} "
                        f"reasons={result['reasons']}"
                    )

    ranked = sorted(records, key=lambda r: float(r["score"]), reverse=True)
    topk = ranked[:args.top_k]

    summary = {
        "format": "v4_12p4g_thumb_only_refine_debug",
        "model": str(model_path),
        "candidate": str(candidate_path),
        "p3_json": str(p3_path),
        "best_config": str(best_config_path),
        "which": args.which,
        "object_body": args.object_body,
        "args": vars(args),
        "num_records": len(records),
        "topk": topk,
    }

    save_json(out_dir / "summary.json", summary)
    save_json(out_dir / "best_thumb_config.json", topk[0])

    top_txt = out_dir / "topk_summary.txt"
    with open(top_txt, "w") as f:
        f.write("rank,score,offsets,groups,thumb_z,obj_z,disp,reasons\n")
        for i, r in enumerate(topk, 1):
            f.write(
                f"{i},"
                f"{r['score']:.6f},"
                f"{r['offsets']},"
                f"{r['object_groups_hold']},"
                f"{r['thumb_contact_z_mean']},"
                f"{r['object_z_mid_hold']},"
                f"{r['max_object_disp']},"
                f"{r['reasons']}\n"
            )

    viewer_script = write_best_viewer_script(args, out_dir, topk[0])

    print("\n========== P4G SUMMARY ==========")
    print("summary:", out_dir / "summary.json")
    print("top_txt:", top_txt)
    print("best_config:", out_dir / "best_thumb_config.json")
    print("viewer_script:", viewer_script)

    for i, r in enumerate(topk[:10], 1):
        print(
            f"{i:02d}. score={r['score']:+.3f} "
            f"offs={r['offsets']} "
            f"groups={r['object_groups_hold']} "
            f"thumb_z={r['thumb_contact_z_mean']} "
            f"obj_z={r['object_z_mid_hold']:.5f} "
            f"disp={r['max_object_disp']:.5f} "
            f"reasons={r['reasons']}"
        )

    print("\nBest viewer command:")
    print(f"bash {viewer_script}")
    print("=================================\n")


if __name__ == "__main__":
    main()
