#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import time
from datetime import datetime
import importlib.util
import numpy as np
import mujoco

def quat_to_mat_wxyz_debug(q):
    q = np.asarray(q, dtype=float).copy()
    q = q / (np.linalg.norm(q) + 1e-12)
    mat = np.zeros(9, dtype=float)
    mujoco.mju_quat2Mat(mat, q)
    return mat.reshape(3, 3)


def compute_approach_aware_pregrasp_debug(model, data, target_pos, target_quat, args):
    """
    V4.5 修正：
    palm_axis 不能直接作为通用 approach 方向。
    对 side grasp，更稳定的是 object_center -> target_hand 的水平径向方向。
    
    modes:
      world_z       : 旧版，target_pos + z_clearance
      palm_axis     : 沿掌心反方向退开，保留给调试
      object_radial : 推荐侧抓模式，沿物体到手的水平径向退开
    """
    target_pos = np.asarray(target_pos, dtype=float)
    target_quat = np.asarray(target_quat, dtype=float)

    mode = getattr(args, "approach_mode", "object_radial")
    approach_dist = float(getattr(args, "pregrasp_approach_dist", 0.075))
    z_clearance = float(getattr(args, "pregrasp_z", 0.0))

    R = quat_to_mat_wxyz_debug(target_quat)
    palm_axis = R[:, 0].copy()
    palm_axis = palm_axis / (np.linalg.norm(palm_axis) + 1e-12)

    object_body = getattr(args, "object_body", "grasp_bottle")
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)

    if mode == "world_z":
        pregrasp_pos = target_pos.copy()
        pregrasp_pos[2] += z_clearance
        approach_axis = np.array([0.0, 0.0, -1.0], dtype=float)
        return pregrasp_pos, approach_axis

    if mode == "palm_axis":
        if bid >= 0:
            object_center = data.xpos[bid].copy()
            to_object = object_center - target_pos
            if float(np.dot(palm_axis, to_object)) < 0.0:
                palm_axis *= -1.0

        pregrasp_pos = target_pos - approach_dist * palm_axis
        pregrasp_pos[2] += z_clearance
        return pregrasp_pos, palm_axis

    # 推荐：object_radial
    if bid < 0:
        # 找不到物体时退化到 world_z，避免乱飞
        pregrasp_pos = target_pos.copy()
        pregrasp_pos[2] += z_clearance
        return pregrasp_pos, np.array([0.0, 0.0, -1.0], dtype=float)

    object_center = data.xpos[bid].copy()

    radial = target_pos - object_center
    radial[2] = 0.0

    radial_norm = np.linalg.norm(radial)
    if radial_norm < 1e-6:
        # 如果目标几乎在物体正上方，说明更像 top grasp，用 world_z
        pregrasp_pos = target_pos.copy()
        pregrasp_pos[2] += max(z_clearance, approach_dist)
        return pregrasp_pos, np.array([0.0, 0.0, -1.0], dtype=float)

    radial = radial / radial_norm

    # 从物体外侧接近：pregrasp 在 target 的径向外侧
    pregrasp_pos = target_pos + approach_dist * radial
    pregrasp_pos[2] += z_clearance

    return pregrasp_pos, radial


import mujoco.viewer


PROJECT = Path.home() / "Projects/o7_mujoco_sim"
SITE_RUNNER_PATH = PROJECT / "scripts/run_fr3_o7_candidate_grasp_site_debug.py"


def load_site_runner():
    spec = importlib.util.spec_from_file_location("site_runner", str(SITE_RUNNER_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


site_runner = load_site_runner()
v12 = site_runner.v12


def load_candidate(path):
    with open(path, "r") as f:
        c = json.load(f)
    if c.get("format") != "fr3_o7_grasp_candidate_v1":
        raise RuntimeError(f"unsupported candidate format: {c.get('format')}")
    return c


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

    return T, pos, quat, {
        "dx": float(dx),
        "dy": float(dy),
        "z_shift": float(z_shift),
        "yaw_deg": float(np.rad2deg(yaw)),
    }


def get_arm_qdict(model, data):
    return {j: v12.get_qpos_joint(model, data, j) for j in v12.FRANKA_JOINTS}


def set_arm_qdict_ctrl(model, data, qdict):
    for j, val in qdict.items():
        v12.set_ctrl_joint(model, data, j, val)


def frame_error(model, data, frame_type, frame_name, target_pos, target_quat):
    cur_pos, cur_quat = site_runner.frame_pose(model, data, frame_type, frame_name)
    pos_err = np.asarray(target_pos, dtype=float) - cur_pos
    rot_err = v12.quat_orientation_error(np.asarray(target_quat, dtype=float), cur_quat)

    return {
        "cur_pos": cur_pos.tolist(),
        "cur_quat": cur_quat.tolist(),
        "target_pos": np.asarray(target_pos, dtype=float).tolist(),
        "target_quat": np.asarray(target_quat, dtype=float).tolist(),
        "pos_err_norm": float(np.linalg.norm(pos_err)),
        "rot_err_norm": float(np.linalg.norm(rot_err)),
        "pos_err": pos_err.tolist(),
        "rot_err": rot_err.tolist(),
    }


def site_servo_ctrl_step(
    model,
    data,
    frame_type,
    frame_name,
    target_pos,
    target_quat,
    damping=0.04,
    step_scale=0.45,
    max_step=0.035,
    rot_weight=0.55,
):
    """
    真实物理版 close 控制：
    不直接 set qpos，只根据当前 site 误差计算一个新的 arm position ctrl。
    这相当于简化版 Cartesian/site 伺服。
    """
    mujoco.mj_forward(model, data)

    cur_pos, cur_quat = site_runner.frame_pose(model, data, frame_type, frame_name)

    pos_err = np.asarray(target_pos, dtype=float) - cur_pos
    rot_err = v12.quat_orientation_error(np.asarray(target_quat, dtype=float), cur_quat)
    err = np.concatenate([pos_err, rot_weight * rot_err])

    jacp, jacr = site_runner.frame_jac(model, data, frame_type, frame_name)
    dofs = np.array([v12.dadr(model, j) for j in v12.FRANKA_JOINTS], dtype=int)

    J = np.vstack([
        jacp[:, dofs],
        rot_weight * jacr[:, dofs],
    ])

    A = J @ J.T + (damping ** 2) * np.eye(6)
    dq = J.T @ np.linalg.solve(A, err)
    dq *= step_scale

    dq_norm = float(np.linalg.norm(dq))
    if dq_norm > max_step:
        dq *= max_step / (dq_norm + 1e-12)

    q_now = get_arm_qdict(model, data)
    q_cmd = {}

    for idx, j in enumerate(v12.FRANKA_JOINTS):
        jid = v12.joint_id(model, j)
        q = q_now[j] + dq[idx]

        if model.jnt_limited[jid]:
            lo, hi = model.jnt_range[jid]
            q = float(np.clip(q, lo, hi))

        q_cmd[j] = q

    set_arm_qdict_ctrl(model, data, q_cmd)

    return {
        "pos_err_norm": float(np.linalg.norm(pos_err)),
        "rot_err_norm": float(np.linalg.norm(rot_err)),
        "dq_norm": dq_norm,
        "q_cmd": q_cmd,
    }


def log_row(model, data, stage, t, object_body, object_tokens, support_tokens, frame_type, frame_name, target_pos, target_quat):
    obj_pos = v12.body_pos(model, data, object_body)
    counts = v12.contact_counts(model, data, object_tokens, support_tokens)
    ferr = frame_error(model, data, frame_type, frame_name, target_pos, target_quat)

    row = {
        "stage": stage,
        "t": float(t),
        "object_pos": obj_pos.tolist(),
        "object_z": float(obj_pos[2]),
        "target_frame_error": ferr,
        **counts,
    }

    return row


def run_stage(
    model,
    data,
    viewer,
    label,
    duration,
    log_dt,
    ctrl_cb,
    logs,
    object_body,
    object_tokens,
    support_tokens,
    frame_type,
    frame_name,
    target_pos_for_log,
    target_quat_for_log,
):
    dt = float(model.opt.timestep)
    steps = max(1, int(duration / dt))
    log_every = max(1, int(log_dt / dt))

    print(f"\n[STAGE] {label}, duration={duration:.2f}s")

    for k in range(steps + 1):
        alpha = k / steps
        t = k * dt

        ctrl_cb(alpha, t)

        if k % log_every == 0 or k == steps:
            row = log_row(
                model,
                data,
                label,
                t,
                object_body,
                object_tokens,
                support_tokens,
                frame_type,
                frame_name,
                target_pos_for_log,
                target_quat_for_log,
            )
            logs.append(row)

            ferr = row["target_frame_error"]
            print(
                f"  t={t:6.2f} "
                f"object_z={row['object_z']:.5f} "
                f"hand_object={row['hand_object']:2d} "
                f"fr3_object={row['fr3_object']:2d} "
                f"object_support={row['object_support']:2d} "
                f"pos_err={ferr['pos_err_norm']:.5f} "
                f"rot_err={ferr['rot_err_norm']:.5f}"
            )

        mujoco.mj_step(model, data)

        if viewer is not None:
            viewer.sync()
            time.sleep(dt)


def interp_vec(a, b, alpha):
    return np.asarray(a, dtype=float) * (1.0 - alpha) + np.asarray(b, dtype=float) * alpha


def run_one_trial(model, data, candidate, rng, args, trial_idx, viewer=None):
    obj_cfg = candidate["object"]
    target_cfg = candidate["target"]
    hand_cfg = candidate["hand"]
    exec_cfg = candidate["execution"]
    val_cfg = candidate.get("validation", {})

    object_body = obj_cfg["body"]
    object_tokens = v12.parse_tokens(obj_cfg["token"])
    support_tokens = v12.parse_tokens(obj_cfg.get("support_tokens", "pedestal table"))

    frame_type, frame_name = site_runner.resolve_target_frame(target_cfg)

    spawn_source = args.spawn_source or obj_cfg.get("spawn_source", "model")
    if spawn_source == "model":
        T_spawn_base = get_model_initial_object_T(model, object_body)
    elif spawn_source == "template":
        T_spawn_base = np.asarray(obj_cfg["T_world_object"], dtype=float)
    else:
        raise RuntimeError(f"unknown spawn_source: {spawn_source}")

    T_world_object, object_pos, object_quat, random_info = randomize_T(
        T_spawn_base,
        rng,
        args.xy_range,
        args.yaw_range_deg,
        args.z_shift,
    )

    T_object_target = np.asarray(target_cfg["T_object_target"], dtype=float)
    T_world_target = T_world_object @ T_object_target
    target_pos, target_quat = v12.T_to_pose(T_world_target)

    pre_cfg = exec_cfg.get("pregrasp", {})
    pregrasp_z = args.pregrasp_z if args.pregrasp_z is not None else float(pre_cfg.get("z_offset", 0.08))
    move_duration = args.move_duration if args.move_duration is not None else float(pre_cfg.get("move_duration", 4.0))
    descend_duration = args.descend_duration if args.descend_duration is not None else float(pre_cfg.get("descend_duration", 2.0))
    close_duration = args.close_duration if args.close_duration is not None else float(exec_cfg.get("close_duration", 3.0))
    hold_duration = args.hold_duration if args.hold_duration is not None else float(exec_cfg.get("hold_duration", 3.0))

    pregrasp_pos = target_pos.copy()
    pregrasp_pos[2] += pregrasp_z
    pregrasp_quat = target_quat.copy()

    lift_z = args.lift_z
    lift_mode = getattr(args, "lift_mode", "q_interp")

    min_final_hand_object = args.min_final_hand_object
    if min_final_hand_object is None:
        min_final_hand_object = int(val_cfg.get("min_final_hand_object", 1))

    o7_ctrl = hand_cfg["o7_active_ctrl"]
    seed_q = candidate.get("arm_seed", {}).get("franka_ctrl", v12.START_ARM)


    # V4.5 approach-aware pregrasp:
    # 原始 pregrasp_pos 可能只是 target_pos + world_z。
    # 这里在求 q_pre 前统一覆盖，保证侧抓从掌心反方向外侧接近。
    pregrasp_pos, approach_palm_axis_world = compute_approach_aware_pregrasp_debug(
        model, data, target_pos, target_quat, args
    )
    q_pre, ik_pre_info = site_runner.solve_franka_ik_dls_frame(
        model,
        seed_q=seed_q,
        target_frame_type=frame_type,
        target_frame_name=frame_name,
        target_pos=pregrasp_pos,
        target_quat=pregrasp_quat,
    )

    q_grasp, ik_grasp_info = site_runner.solve_franka_ik_dls_frame(
        model,
        seed_q=q_pre,
        target_frame_type=frame_type,
        target_frame_name=frame_name,
        target_pos=target_pos,
        target_quat=target_quat,
    )

    print("\n\n===================================================")
    print(f"[TRIAL {trial_idx}] candidate={candidate['candidate_name']}")
    print("object_body       :", object_body)
    print("random_info       :", random_info)
    print("spawn_source      :", spawn_source)
    print("target_frame_type :", frame_type)
    print("target_frame_name :", frame_name)
    print("pregrasp_pos      :", pregrasp_pos)
    print("approach_mode      :", getattr(args, "approach_mode", "world_z"))
    print("approach_dist      :", getattr(args, "pregrasp_approach_dist", 0.075))
    print("approach_palm_axis :", approach_palm_axis_world if "approach_palm_axis_world" in locals() else None)
    print("grasp_pos         :", target_pos)
    print("lift_z            :", lift_z)
    print("lift_mode          :", lift_mode)
    print("ik_pre_info       :", ik_pre_info)
    print("ik_grasp_info     :", ik_grasp_info)
    print("===================================================")

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    if model.nu > 0:
        data.ctrl[:] = 0.0

    v12.set_free_body_pose(model, data, object_body, object_pos, object_quat)
    v12.set_arm_qpos_and_ctrl(model, data, v12.START_ARM)
    v12.set_approach_hand_ctrl(model, data, o7_ctrl)
    mujoco.mj_forward(model, data)

    z0 = float(v12.body_pos(model, data, object_body)[2])
    logs = []

    def move_to_pre_cb(alpha, t):
        a = v12.smoothstep(alpha)
        qcmd = v12.interpolate_dict(v12.START_ARM, q_pre, a)
        v12.set_arm_ctrl(model, data, qcmd)
        v12.set_approach_hand_ctrl(model, data, o7_ctrl)

    def descend_servo_cb(alpha, t):
        a = v12.smoothstep(alpha)
        p = interp_vec(pregrasp_pos, target_pos, a)

        site_servo_ctrl_step(
            model,
            data,
            frame_type,
            frame_name,
            p,
            target_quat,
            damping=args.servo_damping,
            step_scale=args.servo_step_scale,
            max_step=args.servo_max_step,
            rot_weight=args.servo_rot_weight,
        )
        v12.set_approach_hand_ctrl(model, data, o7_ctrl)

    def close_servo_cb(alpha, t):
        site_servo_ctrl_step(
            model,
            data,
            frame_type,
            frame_name,
            target_pos,
            target_quat,
            damping=args.servo_damping,
            step_scale=args.servo_step_scale,
            max_step=args.servo_max_step,
            rot_weight=args.servo_rot_weight,
        )
        v12.set_close_hand_ctrl(model, data, o7_ctrl, alpha)

    def grasp_hold_servo_cb(alpha, t):
        site_servo_ctrl_step(
            model,
            data,
            frame_type,
            frame_name,
            target_pos,
            target_quat,
            damping=args.servo_damping,
            step_scale=args.servo_step_scale,
            max_step=args.servo_max_step,
            rot_weight=args.servo_rot_weight,
        )
        v12.set_full_grasp_hand_ctrl(model, data, o7_ctrl)

    lift_runtime = {
        "prepared": False,
        "start_q": None,
        "target_q": None,
        "start_pos": None,
        "start_quat": None,
        "target_pos": None,
        "target_quat": None,
        "ik_info": None,
    }

    def prepare_lift_from_current_site_pose():
        """
        V4.4:
        lift 不再从原始数据集 target_pos 上加 z。
        lift 从 hold 结束后的当前真实 site 位姿开始，求 q_lift，然后整臂 q-space 插值上抬。
        """
        if lift_runtime["prepared"]:
            return

        cur_pos, cur_quat = site_runner.frame_pose(model, data, frame_type, frame_name)
        cur_q = get_arm_qdict(model, data)

        lift_target_pos = np.asarray(cur_pos, dtype=float).copy()
        lift_target_pos[2] += lift_z
        lift_target_quat = np.asarray(cur_quat, dtype=float).copy()

        q_lift, ik_lift_info = site_runner.solve_franka_ik_dls_frame(
            model,
            seed_q=cur_q,
            target_frame_type=frame_type,
            target_frame_name=frame_name,
            target_pos=lift_target_pos,
            target_quat=lift_target_quat,
        )

        lift_runtime["prepared"] = True
        lift_runtime["start_q"] = cur_q
        lift_runtime["target_q"] = q_lift
        lift_runtime["start_pos"] = np.asarray(cur_pos, dtype=float)
        lift_runtime["start_quat"] = np.asarray(cur_quat, dtype=float)
        lift_runtime["target_pos"] = lift_target_pos
        lift_runtime["target_quat"] = lift_target_quat
        lift_runtime["ik_info"] = ik_lift_info

        print("\n[LIFT PREPARE]")
        print("lift_mode          :", lift_mode)
        print("current_site_pos   :", lift_runtime["start_pos"])
        print("lift_target_pos    :", lift_runtime["target_pos"])
        print("current_site_quat  :", lift_runtime["start_quat"])
        print("ik_lift_info       :", ik_lift_info)
        print("q_start            :", cur_q)
        print("q_lift             :", q_lift)
        print("================\n")

    def lift_q_interp_cb(alpha, t):
        a = v12.smoothstep(alpha)
        qcmd = v12.interpolate_dict(lift_runtime["start_q"], lift_runtime["target_q"], a)
        v12.set_arm_ctrl(model, data, qcmd)
        v12.set_full_grasp_hand_ctrl(model, data, o7_ctrl)

    def lift_servo_cb(alpha, t):
        # 旧模式保留：从当前实际site位姿出发做site servo，但默认不再用这个。
        a = v12.smoothstep(alpha)
        p = lift_runtime["start_pos"].copy()
        p[2] += lift_z * a

        site_servo_ctrl_step(
            model,
            data,
            frame_type,
            frame_name,
            p,
            lift_runtime["start_quat"],
            damping=args.servo_damping,
            step_scale=args.servo_step_scale,
            max_step=args.servo_max_step,
            rot_weight=args.servo_rot_weight,
        )
        v12.set_full_grasp_hand_ctrl(model, data, o7_ctrl)

    run_stage(
        model, data, viewer,
        "move_to_pregrasp_ctrl",
        move_duration,
        args.log_dt,
        move_to_pre_cb,
        logs,
        object_body,
        object_tokens,
        support_tokens,
        frame_type,
        frame_name,
        pregrasp_pos,
        pregrasp_quat,
    )

    run_stage(
        model, data, viewer,
        "descend_site_servo",
        descend_duration,
        args.log_dt,
        descend_servo_cb,
        logs,
        object_body,
        object_tokens,
        support_tokens,
        frame_type,
        frame_name,
        target_pos,
        target_quat,
    )

    run_stage(
        model, data, viewer,
        "close_site_servo",
        close_duration,
        args.log_dt,
        close_servo_cb,
        logs,
        object_body,
        object_tokens,
        support_tokens,
        frame_type,
        frame_name,
        target_pos,
        target_quat,
    )

    run_stage(
        model, data, viewer,
        "grasp_hold_site_servo",
        hold_duration,
        args.log_dt,
        grasp_hold_servo_cb,
        logs,
        object_body,
        object_tokens,
        support_tokens,
        frame_type,
        frame_name,
        target_pos,
        target_quat,
    )

    if lift_z > 0:
        prepare_lift_from_current_site_pose()

        if lift_mode == "q_interp":
            lift_cb = lift_q_interp_cb
            lift_stage_name = "lift_q_interp_current_pose"
        else:
            lift_cb = lift_servo_cb
            lift_stage_name = "lift_site_servo_current_pose"

        run_stage(
            model, data, viewer,
            lift_stage_name,
            args.lift_duration,
            args.log_dt,
            lift_cb,
            logs,
            object_body,
            object_tokens,
            support_tokens,
            frame_type,
            frame_name,
            lift_runtime["target_pos"],
            lift_runtime["target_quat"],
        )

    final_counts = v12.contact_counts(model, data, object_tokens, support_tokens)
    final_pos = v12.body_pos(model, data, object_body)
    final_z = float(final_pos[2])
    max_z = max(row["object_z"] for row in logs)

    close_rows = [r for r in logs if r["stage"] == "close_site_servo"]
    hold_rows = [r for r in logs if r["stage"] == "grasp_hold_site_servo"]
    lift_rows = [r for r in logs if r["stage"] in ("lift_site_servo", "lift_site_servo_current_pose", "lift_q_interp_current_pose")]

    close_final = close_rows[-1] if close_rows else None
    hold_final = hold_rows[-1] if hold_rows else None

    max_hand_object_close = max([r["hand_object"] for r in close_rows], default=0)
    max_hand_object_hold = max([r["hand_object"] for r in hold_rows], default=0)
    max_hand_object_lift = max([r["hand_object"] for r in lift_rows], default=0)

    final_frame_target_pos = lift_runtime["target_pos"] if lift_runtime.get("prepared") else target_pos
    final_frame_target_quat = lift_runtime["target_quat"] if lift_runtime.get("prepared") else target_quat
    final_frame_error = frame_error(model, data, frame_type, frame_name, final_frame_target_pos, final_frame_target_quat)

    success = (
        ik_pre_info.get("success", False)
        and ik_grasp_info.get("success", False)
        and max_hand_object_close >= min_final_hand_object
        and max_hand_object_hold >= min_final_hand_object
        and final_counts["fr3_object"] == 0
        and final_counts["object_support"] == 0
    )

    summary = {
        "trial": trial_idx,
        "candidate_name": candidate["candidate_name"],
        "random_info": random_info,
        "object_body": object_body,
        "spawn_source": spawn_source,
        "target_frame_type": frame_type,
        "target_frame_name": frame_name,
        "target_pos": target_pos.tolist(),
        "target_quat": target_quat.tolist(),
        "pregrasp_pos": pregrasp_pos.tolist(),
        "pregrasp_quat": pregrasp_quat.tolist(),
        "ik_pre_info": ik_pre_info,
        "ik_grasp_info": ik_grasp_info,
        "q_pre": q_pre,
        "q_grasp": q_grasp,
        "lift_mode": lift_mode,
        "lift_start_pos": lift_runtime["start_pos"].tolist() if lift_runtime.get("start_pos") is not None else None,
        "lift_target_pos": lift_runtime["target_pos"].tolist() if lift_runtime.get("target_pos") is not None else None,
        "lift_start_quat": lift_runtime["start_quat"].tolist() if lift_runtime.get("start_quat") is not None else None,
        "lift_target_quat": lift_runtime["target_quat"].tolist() if lift_runtime.get("target_quat") is not None else None,
        "ik_lift_info": lift_runtime.get("ik_info"),
        "q_lift": lift_runtime.get("target_q"),
        "z0": z0,
        "final_pos": final_pos.tolist(),
        "final_z": final_z,
        "final_rise": float(final_z - z0),
        "max_z": float(max_z),
        "max_rise": float(max_z - z0),
        "close_final": close_final,
        "hold_final": hold_final,
        "max_hand_object_close": int(max_hand_object_close),
        "max_hand_object_hold": int(max_hand_object_hold),
        "max_hand_object_lift": int(max_hand_object_lift),
        "final_frame_error": final_frame_error,
        "final_counts": final_counts,
        "success": bool(success),
        "logs": logs,
    }

    print("\n========== TRIAL SUMMARY ==========")
    print("success              :", summary["success"])
    print("z0                   :", summary["z0"])
    print("final_z              :", summary["final_z"])
    print("final_rise           :", summary["final_rise"])
    print("max_rise             :", summary["max_rise"])
    print("max_hand_object_close:", summary["max_hand_object_close"])
    print("max_hand_object_hold :", summary["max_hand_object_hold"])
    print("max_hand_object_lift :", summary["max_hand_object_lift"])
    print("final_frame_pos_err  :", final_frame_error["pos_err_norm"])
    print("final_frame_rot_err  :", final_frame_error["rot_err_norm"])
    print("final_counts         :", final_counts)
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
    ap.add_argument("--approach-mode", choices=["world_z", "palm_axis", "object_radial"], default="world_z",
                    help="world_z=旧版竖直pregrasp；palm_axis=沿掌心反方向退开；object_radial=沿物体到手的水平径向退开，推荐侧抓")
    ap.add_argument("--pregrasp-approach-dist", type=float, default=0.075,
                    help="palm_axis模式下，pregrasp沿掌心反方向退开的距离")
    ap.add_argument("--move-duration", type=float, default=None)
    ap.add_argument("--descend-duration", type=float, default=None)
    ap.add_argument("--close-duration", type=float, default=None)
    ap.add_argument("--hold-duration", type=float, default=None)

    ap.add_argument("--lift-z", type=float, default=0.0)
    ap.add_argument("--lift-duration", type=float, default=3.0)
    ap.add_argument("--lift-mode", choices=["q_interp", "site_servo"], default="q_interp",
                    help="q_interp: hold结束后基于当前实际site位姿求q_lift并整臂插值上抬；site_servo:旧的lift site伺服")

    ap.add_argument("--servo-damping", type=float, default=0.04)
    ap.add_argument("--servo-step-scale", type=float, default=0.45)
    ap.add_argument("--servo-max-step", type=float, default=0.035)
    ap.add_argument("--servo-rot-weight", type=float, default=0.55)

    ap.add_argument("--min-final-hand-object", type=int, default=None)

    ap.add_argument("--log-dt", type=float, default=0.25)
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
        out_path = PROJECT / "diagnostics" / f"fr3_o7_candidate_grasp_site_servo_debug_{candidate['candidate_name']}_{stamp}.json"

    print("\n========== FR3 + O7 SITE SERVO DEBUG ==========")
    print("model             :", model_path)
    print("candidate         :", candidate_path)
    print("candidate_name    :", candidate["candidate_name"])
    print("trials            :", args.trials)
    print("servo_damping     :", args.servo_damping)
    print("servo_step_scale  :", args.servo_step_scale)
    print("servo_max_step    :", args.servo_max_step)
    print("servo_rot_weight  :", args.servo_rot_weight)
    print("lift_z            :", args.lift_z)
    print("out               :", out_path)
    print("================================================\n")

    results = []

    if args.viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = np.array([0.48, -0.03, 0.12])
            viewer.cam.distance = 0.8
            viewer.cam.azimuth = 130
            viewer.cam.elevation = -18
            viewer.opt.geomgroup[3] = 0
            viewer.opt.geomgroup[4] = 1

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

    summary = {
        "format": "fr3_o7_candidate_grasp_site_servo_debug_result_v1",
        "model": str(model_path),
        "candidate": str(candidate_path),
        "candidate_name": candidate["candidate_name"],
        "trials": args.trials,
        "success_count": int(success_count),
        "success_rate": float(success_count / max(1, args.trials)),
        "results": results,
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(summary, f, indent=2)

    print("\n========== ALL DONE ==========")
    print("success_count:", success_count)
    print("trials       :", args.trials)
    print("success_rate :", summary["success_rate"])
    print("saved        :", out_path)
    print("==============================\n")


if __name__ == "__main__":
    main()
