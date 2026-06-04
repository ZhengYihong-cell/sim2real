#!/usr/bin/env python3
"""
文件名：
    run_v4_12p4b_side_grasp_thumb_preshape_debug.py

脚本类别：
    debug / runner / viewer / contact-seeking-close

用途：
    本脚本用于修正 V4.12P4 中侧握 can/bottle 时的拇指策略错误。
    旧 P4 把所有手指从 0 位一起闭合，导致 thumb 不是从对握姿态进入抓取。
    本脚本增加 side-grasp thumb preshape 阶段：
        1. 机械臂到 q_pre，四指保持打开。
        2. 拇指先移动到与四指对握的预张开姿态。
        3. 机械臂到 q_grasp，保持拇指对握预姿态。
        4. contact-seeking close：
            - thumb roll / thumb yaw 保持对握姿态；
            - thumb pitch 负责向物体闭合；
            - 四指 MCP 逐步闭合；
            - 某个 finger group 接触 object 后冻结该组。
        5. hold 后执行 q_grasp -> q_lift。

输入：
    1. MuJoCo XML。
    2. candidate JSON。
    3. P3 JSON 或 P3 plan JSON。
    4. object_body，例如 grasp_can。

输出：
    1. viewer 可视化。
    2. JSON 日志，记录 thumb preshape、finger contact、final lift。
    3. 终端打印 contact-seeking close 的关键状态。

当前流程位置：
    P2 Pinocchio IK
        -> P3 IK 组合与路径预检
        -> P4B 侧握 thumb-preshape contact-seeking close
        -> 后续固化为真实 runner

本脚本不负责：
    1. 不重新求 IK。
    2. 不重新选 candidate。
    3. 不用固定倍率粗暴放大闭合。
    4. 不把数据集 hand ctrl 当最终命令。
"""

from pathlib import Path
import argparse
import json
import time
import sys
import numpy as np
import mujoco

try:
    import mujoco.viewer
except Exception:
    mujoco.viewer = None


PROJECT = Path.home() / "Projects/o7_mujoco_sim"
sys.path.append(str(PROJECT / "scripts/05_execution_runner"))

import run_v4_12p4_contact_seeking_close_debug as p4


def resolve_path(p):
    p = Path(str(p)).expanduser()
    if not p.is_absolute():
        p = PROJECT / p
    return p


def save_json(p, obj):
    p = resolve_path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(p4.to_jsonable(obj), f, indent=2)


def make_side_grasp_open_ctrl(model, data, args):
    """
    侧握开手姿态：
    - 四指 MCP 打开；
    - thumb roll/yaw 先到对握预姿态；
    - thumb pitch 保持较开。
    """
    ctrl = p4.initial_open_ctrl(model, data)

    for j in p4.ACTIVE_HAND_JOINTS:
        ctrl[j] = 0.0

    ctrl["thumb_cmc_roll"] = args.thumb_roll_preshape
    ctrl["thumb_cmc_yaw"] = args.thumb_yaw_preshape
    ctrl["thumb_cmc_pitch"] = args.thumb_pitch_open

    for j in p4.ACTIVE_HAND_JOINTS:
        ctrl[j] = p4.clamp_ctrl(model, j, ctrl[j])

    return ctrl


def make_side_grasp_close_dirs():
    """
    侧握闭合方向：
    - thumb roll/yaw 不继续卷，保持对握姿态；
    - thumb pitch 向内闭合；
    - 四指 MCP 向内闭合。
    """
    dirs = {}
    for j in p4.ACTIVE_HAND_JOINTS:
        dirs[j] = 1.0

    dirs["thumb_cmc_roll"] = 0.0
    dirs["thumb_cmc_yaw"] = 0.0
    dirs["thumb_cmc_pitch"] = 1.0

    return dirs


def play_thumb_preshape(viewer, model, data, q_hold, start_ctrl, target_ctrl, steps, sleep_s, object_body, logs):
    print("\n========== THUMB PRESHAPE ==========")
    print("start_ctrl :", start_ctrl)
    print("target_ctrl:", target_ctrl)

    for i in range(steps + 1):
        s = i / float(max(steps, 1))
        ctrl = p4.interp_qdict(start_ctrl, target_ctrl, s)

        p4.apply_arm_q(model, data, q_hold, also_ctrl=True)
        p4.apply_hand_ctrl(model, data, ctrl)
        mujoco.mj_step(model, data)
        p4.apply_arm_q(model, data, q_hold, also_ctrl=True)
        p4.apply_hand_ctrl(model, data, ctrl)
        mujoco.mj_forward(model, data)

        if i in [0, steps // 2, steps]:
            pose = p4.object_pose(model, data, object_body)
            logs.append({
                "phase": "thumb_preshape",
                "step": i,
                "s": s,
                "ctrl": dict(ctrl),
                "hand_qpos": p4.read_hand_qpos(model, data),
                "object_pos": pose["pos"] if pose else None,
            })
            print(
                f"[thumb_preshape] {i}/{steps} "
                f"thumb=({ctrl.get('thumb_cmc_roll'):.3f}, "
                f"{ctrl.get('thumb_cmc_yaw'):.3f}, "
                f"{ctrl.get('thumb_cmc_pitch'):.3f}) "
                f"hand_qpos={p4.read_hand_qpos(model, data)}"
            )

        if viewer is not None:
            viewer.sync()
            time.sleep(sleep_s)

    print("====================================\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--p3-json", default="")
    ap.add_argument("--plan-json", default="")
    ap.add_argument("--which", default="best_available", choices=["best_available", "best_pass"])
    ap.add_argument("--object-body", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--viewer", action="store_true")

    ap.add_argument("--move-steps", type=int, default=120)
    ap.add_argument("--thumb-preshape-steps", type=int, default=120)
    ap.add_argument("--close-duration", type=float, default=3.0)
    ap.add_argument("--hold-duration", type=float, default=0.6)
    ap.add_argument("--lift-duration", type=float, default=2.0)
    ap.add_argument("--close-speed", type=float, default=0.65)
    ap.add_argument("--frame-sleep", type=float, default=0.002)
    ap.add_argument("--log-dt", type=float, default=0.1)

    # 这一组是侧握 thumb opposition 预姿态，不是最终闭合幅度。
    # 这里使用之前 O7 侧握较自然姿态中出现过的 thumb 对握量级：
    # roll/yaw 先把拇指摆到对握方向，pitch 保持较开。
    ap.add_argument("--thumb-roll-preshape", type=float, default=0.56)
    ap.add_argument("--thumb-yaw-preshape", type=float, default=1.15)
    ap.add_argument("--thumb-pitch-open", type=float, default=0.10)

    ap.add_argument("--min-total-object-groups", type=int, default=2)
    ap.add_argument("--min-non-thumb-groups", type=int, default=1)
    ap.add_argument("--allow-no-thumb", action="store_true")
    ap.add_argument("--max-object-push-disp", type=float, default=0.045)
    ap.add_argument("--min-lift-rise-success", type=float, default=0.015)

    args = ap.parse_args()

    model_path = resolve_path(args.model)
    candidate_path = resolve_path(args.candidate)
    candidate = p4.load_json(candidate_path)

    if args.plan_json:
        src_json_path = resolve_path(args.plan_json)
        src = p4.load_json(src_json_path)
        best, source_kind = p4.selected_plan_from_json(src, args.which)
    elif args.p3_json:
        src_json_path = resolve_path(args.p3_json)
        src = p4.load_json(src_json_path)
        best, source_kind = p4.selected_plan_from_json(src, args.which)
    else:
        raise RuntimeError("pass --p3-json or --plan-json")

    object_body = (
        args.object_body
        or best.get("object_body", "")
        or src.get("object_body", "")
        or (candidate.get("object") or {}).get("body", "")
    )
    if not object_body:
        raise RuntimeError("cannot infer object_body; pass --object-body")

    q_pre = best["q_pre"]
    q_grasp = best["q_grasp"]
    q_lift = best["q_lift"]

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    q_current = p4.current_arm_qdict(model, data)
    geom_sets = p4.collect_geom_sets(model, object_body)

    zero_open_ctrl = p4.initial_open_ctrl(model, data)
    for j in p4.ACTIVE_HAND_JOINTS:
        zero_open_ctrl[j] = 0.0

    side_open_ctrl = make_side_grasp_open_ctrl(model, data, args)
    close_dirs = make_side_grasp_close_dirs()

    viewer_cm = None
    viewer = None
    if args.viewer:
        if mujoco.viewer is None:
            raise RuntimeError("mujoco.viewer not available")
        viewer_cm = mujoco.viewer.launch_passive(model, data)
        viewer = viewer_cm.__enter__()

    logs = []
    status = "UNKNOWN"
    failure_reasons = []

    try:
        print("\n========== V4.12P4B SIDE-GRASP THUMB PRESHAPE ==========")
        print("model      :", model_path)
        print("candidate  :", candidate_path)
        print("source json:", src_json_path)
        print("source kind:", source_kind)
        print("object_body:", object_body)
        print("thumb preshape:")
        print("  roll :", args.thumb_roll_preshape)
        print("  yaw  :", args.thumb_yaw_preshape)
        print("  pitch:", args.thumb_pitch_open)
        print("side_open_ctrl:", side_open_ctrl)
        print("close_dirs    :", close_dirs)
        print("selected seeds:", best.get("pre_seed"), "->", best.get("grasp_seed"), "->", best.get("lift_seed"))
        print("========================================================\n")

        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)

        p4.apply_arm_q(model, data, q_current, also_ctrl=True)
        p4.apply_hand_ctrl(model, data, zero_open_ctrl)
        mujoco.mj_forward(model, data)

        if viewer is not None:
            viewer.sync()
            time.sleep(0.3)

        # 1. 到 q_pre，手先保持完全打开
        p4.move_arm_segment(
            viewer, model, data,
            q_current, q_pre,
            zero_open_ctrl,
            args.move_steps,
            args.frame_sleep,
            "move_to_pre_zero_open",
            object_body,
            logs,
        )

        # 2. 在 q_pre 处先做 thumb opposition preshape
        play_thumb_preshape(
            viewer, model, data,
            q_pre,
            zero_open_ctrl,
            side_open_ctrl,
            args.thumb_preshape_steps,
            args.frame_sleep,
            object_body,
            logs,
        )

        # 3. 从 q_pre 到 q_grasp，保持 side-open thumb 对握预姿态
        p4.move_arm_segment(
            viewer, model, data,
            q_pre, q_grasp,
            side_open_ctrl,
            args.move_steps,
            args.frame_sleep,
            "move_to_grasp_side_open",
            object_body,
            logs,
        )

        # 4. 接触反馈闭合
        close_result = p4.contact_seeking_close(
            viewer, model, data,
            q_grasp,
            object_body,
            geom_sets,
            side_open_ctrl,
            close_dirs,
            args,
            logs,
        )

        if close_result["failure_reason"]:
            status = "FAIL_CLOSE"
            failure_reasons.append(close_result["failure_reason"])
        elif not close_result["success_contact_distribution"]:
            status = "FAIL_NO_CONTACT_DISTRIBUTION"
            failure_reasons.append("required thumb/non-thumb contact distribution not reached")
        else:
            status = "CONTACT_CLOSE_OK"

        final_ctrl = close_result["final_ctrl"]

        p4.hold_phase(
            viewer, model, data,
            q_grasp,
            final_ctrl,
            object_body,
            geom_sets,
            args.hold_duration,
            args.frame_sleep,
            "hold_after_close",
            logs,
        )

        lift_result = p4.lift_phase(
            viewer, model, data,
            q_grasp,
            q_lift,
            final_ctrl,
            object_body,
            geom_sets,
            args.lift_duration,
            args.frame_sleep,
            logs,
        )

        if status == "CONTACT_CLOSE_OK":
            if lift_result["final_rise"] >= args.min_lift_rise_success:
                status = "SUCCESS_LIFT"
            else:
                status = "FAIL_LIFT_RISE_TOO_SMALL"
                failure_reasons.append(
                    f"final_rise {lift_result['final_rise']:.5f} < {args.min_lift_rise_success:.5f}"
                )

        out = {
            "format": "v4_12p4b_side_grasp_thumb_preshape_debug",
            "model": str(model_path),
            "candidate": str(candidate_path),
            "source_json": str(src_json_path),
            "source_kind": source_kind,
            "object_body": object_body,
            "args": vars(args),
            "status": status,
            "failure_reasons": failure_reasons,
            "q_current": q_current,
            "q_pre": q_pre,
            "q_grasp": q_grasp,
            "q_lift": q_lift,
            "zero_open_ctrl": zero_open_ctrl,
            "side_open_ctrl": side_open_ctrl,
            "close_dirs": close_dirs,
            "close_result": close_result,
            "lift_result": lift_result,
            "logs": logs,
        }

        save_json(args.out, out)

        print("\n========== V4.12P4B SUMMARY ==========")
        print("status:", status)
        print("failure_reasons:", failure_reasons)
        print("object_contact_state:", close_result["object_contact_state"])
        print("support_contact_state:", close_result["support_contact_state"])
        print("final_ctrl:", close_result["final_ctrl"])
        print("final_hand_qpos:", close_result["final_hand_qpos"])
        print("final_rise:", lift_result["final_rise"])
        print("final_contacts:", {
            "hand_object": lift_result["final_contacts"]["hand_object"],
            "hand_support": lift_result["final_contacts"]["hand_support"],
            "fr3_object": lift_result["final_contacts"]["fr3_object"],
            "object_support": lift_result["final_contacts"]["object_support"],
            "object_groups": lift_result["final_contacts"]["object_groups"],
        })
        print("saved:", resolve_path(args.out))
        print("======================================\n")

        if args.viewer:
            print("viewer 播放完成。关闭窗口即可退出。")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.03)

    finally:
        if viewer_cm is not None:
            viewer_cm.__exit__(None, None, None)


if __name__ == "__main__":
    main()
