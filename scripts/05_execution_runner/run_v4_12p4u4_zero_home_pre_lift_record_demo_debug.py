#!/usr/bin/env python3
"""
脚本类型：
    debug / runner / recording-demo

用途：
    V4.12P4U4。
    用于录屏展示 FR3 + O7 从初始竖直姿态开始，经过安全 home，再到 q_pre / q_grasp，
    执行 ready-gated snap close、micro-squeeze 和 lift 的完整流程。

输入：
    --model
        已验证成功的 FR3+O7+can MuJoCo XML。
    --candidate
        当前成功候选 best_candidate.json。
    --p3-json
        当前成功 P3 q_pre / q_grasp。
    --best-config
        当前成功 O7 手型参数。
    --out
        输出 JSON 日志。

输出：
    1. MuJoCo viewer 可视化录屏 demo；
    2. JSON 日志，记录各阶段、接触、物体位移、ready gate 和 lift 结果。

当前流程位置：
    只用于录屏展示，不修改 legacy_final_demos，不重新搜索 candidate，不重新做大范围 IK。

核心流程：
    zero_clamped 开场
    → v12_start 安全 home
    → P3 q_pre
    → P3 q_grasp
    → snap close
    → gated micro-squeeze
    → grip_ready 后 fixed-grip world-z lift

不负责：
    1. 不自动生成新的 IK grasp；
    2. 不使用 auto_radial；
    3. 不允许没抓稳就强行 lift；
    4. 不在 lift 阶段继续改变手型。
"""

from pathlib import Path
import argparse
import json
import time
import importlib.util
import numpy as np
import mujoco
import mujoco.viewer


PROJECT = Path.home() / "Projects/o7_mujoco_sim"
P4U1_PATH = PROJECT / "scripts/05_execution_runner/run_v4_12p4u1_precontact_snap_close_debug.py"


def load_p4u1():
    spec = importlib.util.spec_from_file_location("p4u1", str(P4U1_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p4u1 = load_p4u1()

ARM_JOINTS = p4u1.ARM_JOINTS
O7_ACTIVE_JOINTS = p4u1.O7_ACTIVE_JOINTS

V12_START_ARM = {
    "fr3_joint1": 0.00,
    "fr3_joint2": -0.70,
    "fr3_joint3": 0.00,
    "fr3_joint4": -2.20,
    "fr3_joint5": 0.00,
    "fr3_joint6": 1.80,
    "fr3_joint7": 0.80,
}


def resolve_path(p):
    p = Path(str(p)).expanduser()
    if not p.is_absolute():
        p = PROJECT / p
    return p


def save_json(path, obj):
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(p4u1.to_jsonable(obj), f, indent=2)


def clamp_joint_q(model, joint_name, value):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return float(value)
    if bool(model.jnt_limited[jid]):
        lo, hi = model.jnt_range[jid]
        return float(np.clip(float(value), float(lo), float(hi)))
    return float(value)


def make_start_arm(model, mode):
    if mode == "zero_raw":
        return {j: 0.0 for j in ARM_JOINTS}
    if mode == "zero_clamped":
        return {j: clamp_joint_q(model, j, 0.0) for j in ARM_JOINTS}
    if mode == "v12_start":
        return {j: clamp_joint_q(model, j, V12_START_ARM[j]) for j in ARM_JOINTS}
    raise RuntimeError(f"unknown start-arm-mode: {mode}")


def print_result(out_path, result):
    print("\n========== P4U4 ZERO-HOME-PRE-LIFT RECORD RESULT ==========")
    print("out                 :", resolve_path(out_path))
    print("grip_ready          :", result.get("grip_ready"))
    print("stop_reason         :", result.get("stop_reason"))
    print("max_stable_count    :", result.get("max_stable_count"))
    print("final_object_disp   :", result.get("final_object_disp"))
    print("final_object_rise   :", result.get("final_object_rise"))
    print("final_groups        :", result.get("final_groups"))
    print("final_opposition_cos:", result.get("final_opposition_cos"))
    print("===========================================================\n")


def abort_result(args, model_path, runner, q_start, q_home, q_pre, q_grasp, q_lift,
                 arm_source, hand_source, side_open_ctrl, close_target,
                 grip_hold_ctrl, lift_ik_info, stop_reason, object_geoms, model, data):
    final_live = p4u1.collect_live_contact(model, data, object_geoms, args)
    result = {
        "format": "v4_12p4u4_zero_home_pre_lift_record_demo",
        "model": str(model_path),
        "args": vars(args),
        "q_start": q_start,
        "q_home": q_home,
        "q_pre": q_pre,
        "q_grasp": q_grasp,
        "q_lift": q_lift,
        "arm_source": arm_source,
        "hand_source": hand_source,
        "side_open_ctrl": side_open_ctrl,
        "close_target": close_target,
        "grip_ready": False,
        "stop_reason": stop_reason,
        "max_stable_count": runner.max_stable_count,
        "final_object_disp": runner.object_disp(),
        "final_object_rise": runner.object_rise(),
        "final_groups": final_live["groups"],
        "final_opposition_cos": final_live["opposition_cos"],
        "grip_hold_ctrl": grip_hold_ctrl,
        "lift_ik_info": lift_ik_info,
        "rows": runner.rows,
    }
    save_json(args.out, result)
    print_result(args.out, result)
    return result


def run(args):
    model_path = resolve_path(args.model)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    ctrl_map = p4u1.build_ctrl_map(model)

    candidate = p4u1.load_json(args.candidate)
    p3 = p4u1.load_json(args.p3_json)
    best = p4u1.load_json(args.best_config)

    q_pre, q_grasp, arm_source = p4u1.extract_arm_plan(p3, candidate)
    q_start = make_start_arm(model, args.start_arm_mode)
    q_home = make_start_arm(model, "v12_start")

    hand_prior, hand_source = p4u1.extract_hand_ctrl(best, candidate)
    side_open_ctrl, close_target = p4u1.make_side_open_and_close(
        hand_prior,
        finger_scale=args.finger_close_scale,
        thumb_gain=args.thumb_pitch_from_finger_gain,
        thumb_open_pitch=args.thumb_open_pitch,
    )

    obj_bid = p4u1.body_id(model, args.object_body)
    target_bid = p4u1.body_id(model, args.target_body)
    object_geoms = p4u1.geoms_of_body(model, obj_bid)

    mujoco.mj_resetData(model, data)
    p4u1.set_qpos_dict(model, data, q_start)
    p4u1.set_qpos_dict(model, data, side_open_ctrl)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    q_lift = dict(q_grasp)
    lift_ik_info = None
    if args.enable_lift:
        q_lift, lift_ik_info = p4u1.solve_world_z_lift_qpos_dict(
            model=model,
            data=data,
            q_seed=q_grasp,
            target_body_name=args.target_body,
            lift_z=args.lift_z,
            ik_iters=args.lift_ik_iters,
            damping=args.lift_ik_damping,
        )

    print("\n========== V4.12P4U4 ZERO-HOME-PRE-LIFT RECORD DEMO ==========")
    print("model              :", model_path)
    print("candidate          :", resolve_path(args.candidate))
    print("p3_json            :", resolve_path(args.p3_json))
    print("best_config        :", resolve_path(args.best_config))
    print("object_body        :", args.object_body)
    print("target_body        :", args.target_body)
    print("start_arm_mode     :", args.start_arm_mode)
    print("q_start            :", q_start)
    print("q_home             :", q_home)
    print("arm_source         :", arm_source)
    print("q_pre              :", q_pre)
    print("q_grasp            :", q_grasp)
    print("hand_source        :", hand_source)
    print("side_open_ctrl     :", side_open_ctrl)
    print("close_target       :", close_target)
    print("enable_lift        :", args.enable_lift)
    print("lift_z             :", args.lift_z)
    print("lift_ik_info       :", lift_ik_info)
    print("approach_abort_disp:", args.approach_abort_disp)
    print("IMPORTANT          : zero -> v12_safe_home -> q_pre, no auto_radial.")
    print("==============================================================\n")

    runner = p4u1.Runner(model, data, ctrl_map, obj_bid, object_geoms, args)

    def run_approach_phase_checked(label, steps, arm_a, arm_b, hand_ctrl):
        runner.run_phase(
            label,
            steps,
            arm_a,
            arm_b,
            hand_ctrl,
            hand_ctrl,
        )
        disp = runner.object_disp()
        if disp > float(args.approach_abort_disp):
            print(f"\n[ABORT] approach moved object too much: {disp:.5f} > {args.approach_abort_disp:.5f}")
            return False
        return True

    def execute(viewer=None):
        if viewer is not None:
            runner.attach_viewer(viewer)

        dt = float(model.opt.timestep)

        start_hold_steps = max(0, int(float(args.start_hold_duration) / dt))
        start_to_home_steps = max(1, int(float(args.start_to_home_duration) / dt))
        home_hold_steps = max(0, int(float(args.home_hold_duration) / dt))
        home_to_pre_steps = max(1, int(float(args.home_to_pre_duration) / dt))
        pre_hold_steps = max(0, int(float(args.pre_hold_duration) / dt))
        pre_to_grasp_steps = max(1, int(float(args.pre_to_grasp_duration) / dt))
        settle_steps = max(0, int(float(args.grasp_settle_duration) / dt))
        close_steps = max(1, int(float(args.close_duration) / dt))
        post_hold_steps = max(0, int(float(args.post_close_target_hold_duration) / dt))
        micro_steps = max(0, int(float(args.micro_squeeze_duration) / dt))
        lift_steps = max(1, int(float(args.lift_duration) / dt))
        final_hold_steps = max(0, int(float(args.final_hold_duration) / dt))

        runner.run_hold(
            "record_hold_at_zero_start_side_open",
            start_hold_steps,
            q_start,
            side_open_ctrl,
        )

        if not run_approach_phase_checked(
            "record_slow_arm_zero_start_to_v12_safe_home",
            start_to_home_steps,
            q_start,
            q_home,
            side_open_ctrl,
        ):
            return abort_result(args, model_path, runner, q_start, q_home, q_pre, q_grasp, q_lift,
                                arm_source, hand_source, side_open_ctrl, close_target, dict(close_target),
                                lift_ik_info, "abort_zero_to_home_moved_object",
                                object_geoms, model, data)

        runner.run_hold(
            "record_hold_at_v12_safe_home_side_open",
            home_hold_steps,
            q_home,
            side_open_ctrl,
        )

        if not run_approach_phase_checked(
            "record_slow_arm_v12_safe_home_to_p3_pre",
            home_to_pre_steps,
            q_home,
            q_pre,
            side_open_ctrl,
        ):
            return abort_result(args, model_path, runner, q_start, q_home, q_pre, q_grasp, q_lift,
                                arm_source, hand_source, side_open_ctrl, close_target, dict(close_target),
                                lift_ik_info, "abort_home_to_pre_moved_object",
                                object_geoms, model, data)

        runner.run_hold(
            "record_hold_at_p3_pre_side_open",
            pre_hold_steps,
            q_pre,
            side_open_ctrl,
        )

        if not run_approach_phase_checked(
            "record_slow_arm_p3_pre_to_grasp",
            pre_to_grasp_steps,
            q_pre,
            q_grasp,
            side_open_ctrl,
        ):
            return abort_result(args, model_path, runner, q_start, q_home, q_pre, q_grasp, q_lift,
                                arm_source, hand_source, side_open_ctrl, close_target, dict(close_target),
                                lift_ik_info, "abort_pre_to_grasp_moved_object",
                                object_geoms, model, data)

        runner.run_hold(
            "settle_side_open_at_grasp",
            settle_steps,
            q_grasp,
            side_open_ctrl,
        )

        runner.run_phase(
            "snap_close_to_target",
            close_steps,
            q_grasp,
            q_grasp,
            side_open_ctrl,
            close_target,
        )

        runner.run_hold(
            "grip_settle_at_close_target",
            post_hold_steps,
            q_grasp,
            close_target,
        )

        grip_ready = runner.stable_count >= int(args.grip_ready_stable_steps)
        stop_reason = "ready_after_close_target_hold" if grip_ready else "not_ready_after_close_target_hold"
        grip_hold_ctrl = dict(close_target)

        if not grip_ready and micro_steps > 0:
            print(f"\n[PHASE] gated_micro_squeeze, steps={micro_steps}")

            squeeze_dir = {}
            for j in O7_ACTIVE_JOINTS:
                squeeze_dir[j] = float(close_target.get(j, 0.0)) - float(side_open_ctrl.get(j, 0.0))

            start_disp = runner.object_disp()

            for k in range(micro_steps + 1):
                alpha = 1.0 if micro_steps <= 0 else k / float(micro_steps)
                frac = float(args.micro_squeeze_fraction) * p4u1.smoothstep(alpha)

                hand = {}
                for j in O7_ACTIVE_JOINTS:
                    hand[j] = float(close_target.get(j, 0.0)) + frac * squeeze_dir.get(j, 0.0)

                grip_hold_ctrl = dict(hand)
                runner.step_once("gated_micro_squeeze", k, alpha, q_grasp, hand)

                if runner.stable_count >= int(args.grip_ready_stable_steps):
                    grip_ready = True
                    stop_reason = "ready_during_gated_micro_squeeze"
                    print("[GRIP READY] stable opposition reached. Stop squeezing.")
                    break

                disp = runner.object_disp()

                if disp > float(args.max_grip_disp):
                    stop_reason = "fail_object_disp_exceeded_during_micro_squeeze"
                    print(f"[NO GRIP] object disp exceeded: {disp:.5f} > {args.max_grip_disp:.5f}")
                    break

                if disp - start_disp > float(args.max_extra_disp_during_squeeze):
                    stop_reason = "fail_extra_disp_exceeded_during_micro_squeeze"
                    print(f"[NO GRIP] extra object disp exceeded: {disp-start_disp:.5f} > {args.max_extra_disp_during_squeeze:.5f}")
                    break

        if not grip_ready:
            print("\n[NO_LIFT] grip is not ready. Do not lift an ungrasped object.")
            return abort_result(args, model_path, runner, q_start, q_home, q_pre, q_grasp, q_lift,
                                arm_source, hand_source, side_open_ctrl, close_target, grip_hold_ctrl,
                                lift_ik_info, stop_reason, object_geoms, model, data)

        print("\n[GRIP_READY] lift is allowed. Hand ctrl will stay constant during lift.")
        final_live_before_lift = p4u1.collect_live_contact(model, data, object_geoms, args)

        if args.enable_lift:
            runner.run_phase(
                "lift_world_z_with_fixed_grip_ctrl",
                lift_steps,
                q_grasp,
                q_lift,
                grip_hold_ctrl,
                grip_hold_ctrl,
            )

            if final_hold_steps > 0:
                runner.run_hold(
                    "final_air_hold_after_lift",
                    final_hold_steps,
                    q_lift,
                    grip_hold_ctrl,
                )

        final_live = p4u1.collect_live_contact(model, data, object_geoms, args)
        result = {
            "format": "v4_12p4u4_zero_home_pre_lift_record_demo",
            "model": str(model_path),
            "args": vars(args),
            "q_start": q_start,
            "q_home": q_home,
            "q_pre": q_pre,
            "q_grasp": q_grasp,
            "q_lift": q_lift,
            "arm_source": arm_source,
            "hand_source": hand_source,
            "side_open_ctrl": side_open_ctrl,
            "close_target": close_target,
            "grip_ready": True,
            "stop_reason": stop_reason,
            "max_stable_count": runner.max_stable_count,
            "final_object_disp": runner.object_disp(),
            "final_object_rise": runner.object_rise(),
            "final_groups": final_live["groups"],
            "final_opposition_cos": final_live["opposition_cos"],
            "pre_lift_groups": final_live_before_lift["groups"],
            "pre_lift_opposition_cos": final_live_before_lift["opposition_cos"],
            "grip_hold_ctrl": grip_hold_ctrl,
            "lift_ik_info": lift_ik_info,
            "rows": runner.rows,
        }
        save_json(args.out, result)
        print_result(args.out, result)
        return result

    if args.viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
            result = execute(viewer)
            if args.keep_viewer_open:
                print("[VIEWER] keep open. Close viewer window or Ctrl+C in terminal to exit.")
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.02)
            return result

    return execute(None)


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--p3-json", required=True)
    ap.add_argument("--best-config", required=True)
    ap.add_argument("--which", default="best_available")

    ap.add_argument("--object-body", default="grasp_can")
    ap.add_argument("--target-body", default="fr3_link7")
    ap.add_argument("--out", required=True)

    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--keep-viewer-open", action="store_true")
    ap.add_argument("--frame-sleep", type=float, default=0.0015)

    ap.add_argument(
        "--start-arm-mode",
        choices=["zero_clamped", "zero_raw", "v12_start"],
        default="zero_clamped",
    )

    ap.add_argument("--start-hold-duration", type=float, default=1.2)
    ap.add_argument("--start-to-home-duration", type=float, default=3.0)
    ap.add_argument("--home-hold-duration", type=float, default=0.6)
    ap.add_argument("--home-to-pre-duration", type=float, default=4.0)
    ap.add_argument("--pre-hold-duration", type=float, default=0.8)
    ap.add_argument("--pre-to-grasp-duration", type=float, default=2.0)
    ap.add_argument("--grasp-settle-duration", type=float, default=0.35)

    ap.add_argument("--close-duration", type=float, default=0.45)
    ap.add_argument("--post-close-target-hold-duration", type=float, default=0.25)
    ap.add_argument("--micro-squeeze-duration", type=float, default=0.35)
    ap.add_argument("--micro-squeeze-fraction", type=float, default=0.08)

    ap.add_argument("--enable-lift", action="store_true")
    ap.add_argument("--lift-z", type=float, default=0.060)
    ap.add_argument("--lift-duration", type=float, default=3.0)
    ap.add_argument("--final-hold-duration", type=float, default=1.0)
    ap.add_argument("--lift-ik-iters", type=int, default=140)
    ap.add_argument("--lift-ik-damping", type=float, default=1e-4)

    ap.add_argument("--finger-close-scale", type=float, default=0.92)
    ap.add_argument("--thumb-pitch-from-finger-gain", type=float, default=0.24)
    ap.add_argument("--thumb-open-pitch", type=float, default=0.22)

    ap.add_argument("--min-live-non-thumb", type=int, default=1)
    ap.add_argument("--opposition-cos-threshold", type=float, default=-0.30)
    ap.add_argument("--grip-ready-stable-steps", type=int, default=8)

    ap.add_argument("--max-grip-disp", type=float, default=0.006)
    ap.add_argument("--max-extra-disp-during-squeeze", type=float, default=0.003)

    # 接近阶段保护。超过这个位移，说明路径扫到物体，直接中止。
    ap.add_argument("--approach-abort-disp", type=float, default=0.010)

    ap.add_argument("--print-every-steps", type=int, default=100)
    ap.add_argument("--log-every-steps", type=int, default=100)

    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
