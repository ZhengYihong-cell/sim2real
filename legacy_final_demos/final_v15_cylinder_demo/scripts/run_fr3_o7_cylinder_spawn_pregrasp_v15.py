#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import time
from datetime import datetime
import importlib.util
import numpy as np
import mujoco
import mujoco.viewer


PROJECT = Path.home() / "Projects/o7_mujoco_sim"

DEFAULT_MODEL = PROJECT / "models/fr3_o7/fr3_o7_actuated_scene_v13_cylinder.xml"
DEFAULT_TEMPLATE = PROJECT / "records/fr3_o7_grasp_template_cylinder_v1.json"
V12_PATH = PROJECT / "scripts/run_fr3_o7_object_ik_grasp_v12.py"


def load_v12():
    spec = importlib.util.spec_from_file_location("v12", str(V12_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


v12 = load_v12()


def pose_to_T(pos, quat_wxyz):
    T = np.eye(4, dtype=float)
    T[:3, :3] = v12.quat_to_rotmat(quat_wxyz)
    T[:3, 3] = np.asarray(pos, dtype=float)
    return T


def body_pose(model, data, body_name):
    bid = v12.body_id(model, body_name)
    return data.xpos[bid].copy(), data.xquat[bid].copy()


def get_model_initial_object_T(model, object_body):
    """
    关键点：
    物体初始位姿从 XML/model 初始状态读取，
    不再使用模板里保存时的 object pose。
    """
    d0 = mujoco.MjData(model)
    mujoco.mj_resetData(model, d0)
    mujoco.mj_forward(model, d0)

    pos, quat = body_pose(model, d0, object_body)
    return pose_to_T(pos, quat)


def randomize_T(T0, rng, xy_range, yaw_range_deg, z_shift):
    dx = rng.uniform(-xy_range, xy_range)
    dy = rng.uniform(-xy_range, xy_range)
    yaw = np.deg2rad(rng.uniform(-yaw_range_deg, yaw_range_deg))

    T = T0.copy()
    T[:3, 3] = T0[:3, 3] + np.array([dx, dy, z_shift], dtype=float)
    T[:3, :3] = v12.yaw_matrix(yaw)[:3, :3] @ T0[:3, :3]

    pos, quat = v12.T_to_pose(T)

    info = {
        "dx": float(dx),
        "dy": float(dy),
        "z_shift": float(z_shift),
        "yaw_deg": float(np.rad2deg(yaw)),
    }

    return T, pos, quat, info


def run_one_trial(model, data, template, rng, args, trial_idx, viewer=None):
    object_body = args.object_body
    object_tokens = v12.parse_tokens(args.object_token)
    support_tokens = v12.parse_tokens(args.support_tokens)
    target_body = args.target_body

    # 1. 物体初始位姿
    # 对当前竖直圆柱模板，必须从模板里的 T_world_object 开始。
    # 因为模板源记录中圆柱已经被手接触后形成了可抓取的支撑姿态：
    # hand_object=7, object_support=1。
    # 如果改用 XML/model 的初始圆柱位姿，会导致目标手腕位姿整体错位。
    if args.spawn_source == "template":
        T_spawn_base = v12.get_template_world_object(template)
    elif args.spawn_source == "model":
        T_spawn_base = get_model_initial_object_T(model, object_body)
    else:
        raise RuntimeError(f"unknown spawn_source: {args.spawn_source}")

    T_world_object, object_pos, object_quat, random_info = randomize_T(
        T_spawn_base,
        rng,
        args.xy_range,
        args.yaw_range_deg,
        args.z_shift,
    )

    # 2. 由模板相对位姿得到最终 grasp 目标
    T_object_target = v12.get_template_target_transform(template)
    T_world_target = T_world_object @ T_object_target
    target_pos, target_quat = v12.T_to_pose(T_world_target)

    # 3. pregrasp：目标位姿整体上移，避免远距离运动时提前扫倒圆柱
    pregrasp_pos = target_pos.copy()
    pregrasp_pos[2] += args.pregrasp_z
    pregrasp_quat = target_quat.copy()

    seed_q = template["franka_ctrl"]
    o7_ctrl = template["o7_active_ctrl"]

    q_pre, ik_pre_info = v12.solve_franka_ik_dls(
        model,
        seed_q=seed_q,
        target_body=target_body,
        target_pos=pregrasp_pos,
        target_quat=pregrasp_quat,
    )

    q_grasp, ik_grasp_info = v12.solve_franka_ik_dls(
        model,
        seed_q=q_pre,
        target_body=target_body,
        target_pos=target_pos,
        target_quat=target_quat,
    )

    print("\n\n===================================================")
    print(f"[TRIAL {trial_idx}]")
    print("object_body      :", object_body)
    print("random_info      :", random_info)
    print("spawn object pos :", object_pos)
    print("target_body      :", target_body)
    print("pregrasp_pos     :", pregrasp_pos)
    print("grasp_pos        :", target_pos)
    print("ik_pre_info      :", ik_pre_info)
    print("ik_grasp_info    :", ik_grasp_info)
    print("===================================================")

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0

    v12.set_free_body_pose(model, data, object_body, object_pos, object_quat)
    v12.set_arm_qpos_and_ctrl(model, data, v12.START_ARM)
    v12.set_approach_hand_ctrl(model, data, o7_ctrl)
    mujoco.mj_forward(model, data)

    logs = []
    z0 = float(v12.body_pos(model, data, object_body)[2])

    lift_joint = args.lift_joint
    lift_aid = v12.actuator_id(model, lift_joint + "_pos")
    lift_start = q_grasp[lift_joint]
    lift_target = lift_start + args.lift_delta

    def move_pre_cb(alpha, t):
        a = v12.smoothstep(alpha)
        qcmd = v12.interpolate_dict(v12.START_ARM, q_pre, a)
        v12.set_arm_ctrl(model, data, qcmd)
        v12.set_approach_hand_ctrl(model, data, o7_ctrl)

    def descend_cb(alpha, t):
        a = v12.smoothstep(alpha)
        qcmd = v12.interpolate_dict(q_pre, q_grasp, a)
        v12.set_arm_ctrl(model, data, qcmd)
        v12.set_approach_hand_ctrl(model, data, o7_ctrl)

    def close_cb(alpha, t):
        v12.set_arm_ctrl(model, data, q_grasp)
        v12.set_close_hand_ctrl(model, data, o7_ctrl, alpha)

    def lift_cb(alpha, t):
        v12.set_arm_ctrl(model, data, q_grasp)
        v12.set_full_grasp_hand_ctrl(model, data, o7_ctrl)
        a = v12.smoothstep(alpha)
        data.ctrl[lift_aid] = lift_start + a * args.lift_delta

    def hold_cb(alpha, t):
        v12.set_arm_ctrl(model, data, q_grasp)
        v12.set_full_grasp_hand_ctrl(model, data, o7_ctrl)
        data.ctrl[lift_aid] = lift_target

    v12.run_stage(model, data, viewer, "move_to_pregrasp", args.move_duration, args.log_dt, move_pre_cb, logs, object_body, object_tokens, support_tokens)
    v12.run_stage(model, data, viewer, "descend_to_grasp", args.descend_duration, args.log_dt, descend_cb, logs, object_body, object_tokens, support_tokens)
    v12.run_stage(model, data, viewer, "close_hand", args.close_duration, args.log_dt, close_cb, logs, object_body, object_tokens, support_tokens)
    v12.run_stage(model, data, viewer, "lift", args.lift_duration, args.log_dt, lift_cb, logs, object_body, object_tokens, support_tokens)
    v12.run_stage(model, data, viewer, "air_hold", args.hold_duration, args.log_dt, hold_cb, logs, object_body, object_tokens, support_tokens)

    final_counts = v12.contact_counts(model, data, object_tokens, support_tokens)
    final_z = float(v12.body_pos(model, data, object_body)[2])
    max_z = max(row["object_z"] for row in logs)

    lift_rows = [r for r in logs if r["stage"] in ["lift", "air_hold"]]
    min_hand_object_after_lift = min(r["hand_object"] for r in lift_rows) if lift_rows else 0

    success = (
        ik_pre_info.get("success", False)
        and ik_grasp_info.get("success", False)
        and final_counts["hand_object"] >= args.min_final_hand_object
        and final_counts["fr3_object"] == 0
        and final_counts["object_support"] == 0
        and final_z >= z0 + args.min_final_rise
    )

    summary = {
        "trial": trial_idx,
        "random_info": random_info,
        "object_body": object_body,
        "ik_pre_info": ik_pre_info,
        "ik_grasp_info": ik_grasp_info,
        "q_pre": q_pre,
        "q_grasp": q_grasp,
        "target_pos": target_pos.tolist(),
        "target_quat": target_quat.tolist(),
        "pregrasp_pos": pregrasp_pos.tolist(),
        "pregrasp_quat": pregrasp_quat.tolist(),
        "z0": z0,
        "final_z": final_z,
        "max_z": float(max_z),
        "final_rise": float(final_z - z0),
        "max_rise": float(max_z - z0),
        "min_hand_object_after_lift": int(min_hand_object_after_lift),
        "final_counts": final_counts,
        "success": bool(success),
        "logs": logs,
    }

    print("\n========== TRIAL SUMMARY ==========")
    print("success                    :", summary["success"])
    print("z0                         :", summary["z0"])
    print("final_z                    :", summary["final_z"])
    print("final_rise                 :", summary["final_rise"])
    print("max_rise                   :", summary["max_rise"])
    print("min_hand_object_after_lift :", summary["min_hand_object_after_lift"])
    print("final_counts               :", summary["final_counts"])
    print("===================================\n")

    return summary


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--template", default=str(DEFAULT_TEMPLATE))

    ap.add_argument("--object-body", default="grasp_cylinder")
    ap.add_argument("--object-token", default="grasp_cylinder")
    ap.add_argument("--support-tokens", default="pedestal table")
    ap.add_argument("--target-body", default="fr3_link7")

    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--xy-range", type=float, default=0.0)
    ap.add_argument("--yaw-range-deg", type=float, default=0.0)
    ap.add_argument("--z-shift", type=float, default=0.0)

    ap.add_argument("--spawn-source", choices=["template", "model"], default="template")
    ap.add_argument("--pregrasp-z", type=float, default=0.08)

    ap.add_argument("--move-duration", type=float, default=4.0)
    ap.add_argument("--descend-duration", type=float, default=3.0)
    ap.add_argument("--close-duration", type=float, default=3.0)
    ap.add_argument("--lift-duration", type=float, default=5.0)
    ap.add_argument("--hold-duration", type=float, default=4.0)

    ap.add_argument("--lift-joint", default="fr3_joint4")
    ap.add_argument("--lift-delta", type=float, default=0.18)
    ap.add_argument("--log-dt", type=float, default=0.5)

    ap.add_argument("--min-final-hand-object", type=int, default=3)
    ap.add_argument("--min-final-rise", type=float, default=0.005)

    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--out", default="")

    args = ap.parse_args()

    model_path = v12.resolve_path(args.model)
    template_path = v12.resolve_path(args.template)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    template = v12.load_json(template_path)

    rng = np.random.default_rng(args.seed)

    if args.out:
        out_path = v12.resolve_path(args.out)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = PROJECT / "diagnostics" / f"fr3_o7_cylinder_spawn_pregrasp_v15_{stamp}.json"

    print("\n========== FR3 + O7 CYLINDER SPAWN PREGRASP V15 ==========")
    print("model         :", model_path)
    print("template      :", template_path)
    print("object_body   :", args.object_body)
    print("trials        :", args.trials)
    print("xy_range      :", args.xy_range)
    print("yaw_range_deg :", args.yaw_range_deg)
    print("spawn_source  :", args.spawn_source)
    print("pregrasp_z    :", args.pregrasp_z)
    print("lift_delta    :", args.lift_delta)
    print("out           :", out_path)
    print("==========================================================\n")

    results = []

    if args.viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = np.array([0.54, -0.02, 0.25])
            viewer.cam.distance = 1.2
            viewer.cam.azimuth = 125
            viewer.cam.elevation = -18
            viewer.opt.geomgroup[3] = 0
            viewer.opt.geomgroup[4] = 0

            for i in range(args.trials):
                results.append(run_one_trial(model, data, template, rng, args, i, viewer))

            print("[DONE] 关闭 viewer 退出。")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.01)
    else:
        for i in range(args.trials):
            results.append(run_one_trial(model, data, template, rng, args, i, None))

    success_count = sum(1 for r in results if r["success"])

    final = {
        "format": "fr3_o7_cylinder_spawn_pregrasp_v15_result",
        "model": str(model_path),
        "template": str(template_path),
        "args": vars(args),
        "success_count": success_count,
        "trials": args.trials,
        "success_rate": success_count / max(1, args.trials),
        "results": results,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)

    print("\n========== V15 FINAL SUMMARY ==========")
    print("success_count:", success_count)
    print("trials       :", args.trials)
    print("success_rate :", success_count / max(1, args.trials))
    print("saved        :", out_path)
    print("=======================================\n")


if __name__ == "__main__":
    main()
