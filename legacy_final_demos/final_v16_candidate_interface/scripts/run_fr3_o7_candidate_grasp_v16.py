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


def load_candidate(path):
    with open(path, "r") as f:
        c = json.load(f)

    if c.get("format") != "fr3_o7_grasp_candidate_v1":
        raise RuntimeError(f"unsupported candidate format: {c.get('format')}")

    return c


def get_exec_number(args_value, candidate_value):
    return args_value if args_value is not None else candidate_value


def get_exec_string(args_value, candidate_value):
    return args_value if args_value else candidate_value


def run_one_trial(model, data, candidate, rng, args, trial_idx, viewer=None):
    obj_cfg = candidate["object"]
    target_cfg = candidate["target"]
    hand_cfg = candidate["hand"]
    exec_cfg = candidate["execution"]
    val_cfg = candidate.get("validation", {})

    object_body = obj_cfg["body"]
    object_tokens = v12.parse_tokens(obj_cfg["token"])
    support_tokens = v12.parse_tokens(obj_cfg.get("support_tokens", "pedestal table"))
    target_body = target_cfg.get("body", "fr3_link7")

    spawn_source = args.spawn_source or obj_cfg.get("spawn_source", "template")

    if spawn_source == "template":
        T_spawn_base = np.asarray(obj_cfg["T_world_object"], dtype=float)
    elif spawn_source == "model":
        T_spawn_base = get_model_initial_object_T(model, object_body)
    else:
        raise RuntimeError(f"unknown spawn_source: {spawn_source}")

    xy_range = args.xy_range
    yaw_range_deg = args.yaw_range_deg
    z_shift = args.z_shift

    T_world_object, object_pos, object_quat, random_info = randomize_T(
        T_spawn_base,
        rng,
        xy_range,
        yaw_range_deg,
        z_shift,
    )

    T_object_target = np.asarray(target_cfg["T_object_target"], dtype=float)
    T_world_target = T_world_object @ T_object_target
    target_pos, target_quat = v12.T_to_pose(T_world_target)

    pre_cfg = exec_cfg.get("pregrasp", {})
    pregrasp_z = get_exec_number(args.pregrasp_z, float(pre_cfg.get("z_offset", 0.0)))

    pregrasp_pos = target_pos.copy()
    pregrasp_pos[2] += pregrasp_z
    pregrasp_quat = target_quat.copy()

    move_duration = get_exec_number(args.move_duration, float(pre_cfg.get("move_duration", 5.0)))
    descend_duration = get_exec_number(args.descend_duration, float(pre_cfg.get("descend_duration", 0.0)))
    close_duration = get_exec_number(args.close_duration, float(exec_cfg.get("close_duration", 3.0)))

    lift_cfg = exec_cfg["lift"]
    lift_joint = get_exec_string(args.lift_joint, lift_cfg.get("joint", "fr3_joint4"))
    lift_delta = get_exec_number(args.lift_delta, float(lift_cfg.get("delta", 0.18)))
    lift_duration = get_exec_number(args.lift_duration, float(lift_cfg.get("duration", 5.0)))
    hold_duration = get_exec_number(args.hold_duration, float(exec_cfg.get("hold_duration", 6.0)))

    min_final_hand_object = args.min_final_hand_object
    if min_final_hand_object is None:
        min_final_hand_object = int(val_cfg.get("min_final_hand_object", 3))

    min_final_rise = args.min_final_rise
    if min_final_rise is None:
        min_final_rise = float(val_cfg.get("min_final_rise", 0.005))

    o7_ctrl = hand_cfg["o7_active_ctrl"]
    seed_q = candidate.get("arm_seed", {}).get("franka_ctrl", v12.START_ARM)

    if abs(pregrasp_z) > 1e-9 or descend_duration > 1e-9:
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
    else:
        q_grasp, ik_grasp_info = v12.solve_franka_ik_dls(
            model,
            seed_q=seed_q,
            target_body=target_body,
            target_pos=target_pos,
            target_quat=target_quat,
        )
        q_pre = q_grasp.copy()
        ik_pre_info = {
            "success": True,
            "skipped": True,
            "reason": "pregrasp_z=0 and descend_duration=0",
        }

    print("\n\n===================================================")
    print(f"[TRIAL {trial_idx}] candidate={candidate['candidate_name']}")
    print("object_body      :", object_body)
    print("random_info      :", random_info)
    print("spawn_source     :", spawn_source)
    print("target_body      :", target_body)
    print("pregrasp_z       :", pregrasp_z)
    print("pregrasp_pos     :", pregrasp_pos)
    print("grasp_pos        :", target_pos)
    print("lift_joint       :", lift_joint)
    print("lift_delta       :", lift_delta)
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

    lift_aid = v12.actuator_id(model, lift_joint + "_pos")
    lift_start = q_grasp[lift_joint]
    lift_target = lift_start + lift_delta

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

    def move_grasp_cb(alpha, t):
        a = v12.smoothstep(alpha)
        qcmd = v12.interpolate_dict(v12.START_ARM, q_grasp, a)
        v12.set_arm_ctrl(model, data, qcmd)
        v12.set_approach_hand_ctrl(model, data, o7_ctrl)

    def close_cb(alpha, t):
        v12.set_arm_ctrl(model, data, q_grasp)
        v12.set_close_hand_ctrl(model, data, o7_ctrl, alpha)

    def lift_cb(alpha, t):
        v12.set_arm_ctrl(model, data, q_grasp)
        v12.set_full_grasp_hand_ctrl(model, data, o7_ctrl)
        a = v12.smoothstep(alpha)
        data.ctrl[lift_aid] = lift_start + a * lift_delta

    def hold_cb(alpha, t):
        v12.set_arm_ctrl(model, data, q_grasp)
        v12.set_full_grasp_hand_ctrl(model, data, o7_ctrl)
        data.ctrl[lift_aid] = lift_target

    if abs(pregrasp_z) > 1e-9 or descend_duration > 1e-9:
        v12.run_stage(model, data, viewer, "move_to_pregrasp", move_duration, args.log_dt, move_pre_cb, logs, object_body, object_tokens, support_tokens)
        v12.run_stage(model, data, viewer, "descend_to_grasp", descend_duration, args.log_dt, descend_cb, logs, object_body, object_tokens, support_tokens)
    else:
        v12.run_stage(model, data, viewer, "move_to_grasp", move_duration, args.log_dt, move_grasp_cb, logs, object_body, object_tokens, support_tokens)

    v12.run_stage(model, data, viewer, "close_hand", close_duration, args.log_dt, close_cb, logs, object_body, object_tokens, support_tokens)
    v12.run_stage(model, data, viewer, "lift", lift_duration, args.log_dt, lift_cb, logs, object_body, object_tokens, support_tokens)
    v12.run_stage(model, data, viewer, "air_hold", hold_duration, args.log_dt, hold_cb, logs, object_body, object_tokens, support_tokens)

    final_counts = v12.contact_counts(model, data, object_tokens, support_tokens)
    final_z = float(v12.body_pos(model, data, object_body)[2])
    max_z = max(row["object_z"] for row in logs)

    lift_rows = [r for r in logs if r["stage"] in ["lift", "air_hold"]]
    min_hand_object_after_lift = min(r["hand_object"] for r in lift_rows) if lift_rows else 0

    success = (
        ik_pre_info.get("success", False)
        and ik_grasp_info.get("success", False)
        and final_counts["hand_object"] >= min_final_hand_object
        and final_counts["fr3_object"] == 0
        and final_counts["object_support"] == 0
        and final_z >= z0 + min_final_rise
    )

    summary = {
        "trial": trial_idx,
        "candidate_name": candidate["candidate_name"],
        "random_info": random_info,
        "object_body": object_body,
        "spawn_source": spawn_source,
        "ik_pre_info": ik_pre_info,
        "ik_grasp_info": ik_grasp_info,
        "q_pre": q_pre,
        "q_grasp": q_grasp,
        "target_pos": target_pos.tolist(),
        "target_quat": target_quat.tolist(),
        "pregrasp_pos": pregrasp_pos.tolist(),
        "pregrasp_quat": pregrasp_quat.tolist(),
        "lift_joint": lift_joint,
        "lift_delta": lift_delta,
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

    ap.add_argument("--model", required=True)
    ap.add_argument("--candidate", required=True)

    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--xy-range", type=float, default=0.0)
    ap.add_argument("--yaw-range-deg", type=float, default=0.0)
    ap.add_argument("--z-shift", type=float, default=0.0)

    ap.add_argument("--spawn-source", choices=["template", "model"], default="")

    ap.add_argument("--pregrasp-z", type=float, default=None)
    ap.add_argument("--move-duration", type=float, default=None)
    ap.add_argument("--descend-duration", type=float, default=None)
    ap.add_argument("--close-duration", type=float, default=None)
    ap.add_argument("--lift-duration", type=float, default=None)
    ap.add_argument("--hold-duration", type=float, default=None)

    ap.add_argument("--lift-joint", default="")
    ap.add_argument("--lift-delta", type=float, default=None)

    ap.add_argument("--min-final-hand-object", type=int, default=None)
    ap.add_argument("--min-final-rise", type=float, default=None)

    ap.add_argument("--log-dt", type=float, default=0.5)
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--out", default="")

    args = ap.parse_args()

    model_path = v12.resolve_path(args.model)
    candidate_path = v12.resolve_path(args.candidate)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    candidate = load_candidate(candidate_path)

    rng = np.random.default_rng(args.seed)

    if args.out:
        out_path = v12.resolve_path(args.out)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = PROJECT / "diagnostics" / f"fr3_o7_candidate_grasp_v16_{candidate['candidate_name']}_{stamp}.json"

    print("\n========== FR3 + O7 CANDIDATE GRASP V16 ==========")
    print("model         :", model_path)
    print("candidate     :", candidate_path)
    print("candidate_name:", candidate["candidate_name"])
    print("trials        :", args.trials)
    print("xy_range      :", args.xy_range)
    print("yaw_range_deg :", args.yaw_range_deg)
    print("out           :", out_path)
    print("==================================================\n")

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
                results.append(run_one_trial(model, data, candidate, rng, args, i, viewer))

            print("[DONE] 关闭 viewer 退出。")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.01)
    else:
        for i in range(args.trials):
            results.append(run_one_trial(model, data, candidate, rng, args, i, None))

    success_count = sum(1 for r in results if r["success"])

    final = {
        "format": "fr3_o7_candidate_grasp_v16_result",
        "model": str(model_path),
        "candidate": str(candidate_path),
        "candidate_name": candidate["candidate_name"],
        "args": vars(args),
        "success_count": success_count,
        "trials": args.trials,
        "success_rate": success_count / max(1, args.trials),
        "results": results,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(final, f, indent=2)

    print("\n========== V16 FINAL SUMMARY ==========")
    print("success_count:", success_count)
    print("trials       :", args.trials)
    print("success_rate :", success_count / max(1, args.trials))
    print("saved        :", out_path)
    print("=======================================\n")


if __name__ == "__main__":
    main()
