#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import time
import numpy as np
import mujoco
import mujoco.viewer


PROJECT = Path.home() / "Projects/o7_mujoco_sim"

DEFAULT_MODEL = PROJECT / "models/fr3_o7/fr3_o7_actuated_scene_v1f_stable_hand.xml"
DEFAULT_TEMPLATE = PROJECT / "records/fr3_o7_grasp_template_v1.json"


FRANKA_JOINTS = [
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
]

O7_ACTIVE_JOINTS = [
    "thumb_cmc_roll",
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]

O7_THUMB_JOINTS = [
    "thumb_cmc_roll",
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
]

O7_FOUR_FINGER_JOINTS = [
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]

START_ARM = {
    "fr3_joint1": 0.00,
    "fr3_joint2": -0.70,
    "fr3_joint3": 0.00,
    "fr3_joint4": -2.20,
    "fr3_joint5": 0.00,
    "fr3_joint6": 1.80,
    "fr3_joint7": 0.80,
}


def resolve_path(p):
    p = Path(p).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


def name(model, objtype, idx):
    return mujoco.mj_id2name(model, objtype, idx) or ""


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def quat_to_rotmat(q_wxyz):
    mat = np.zeros(9, dtype=float)
    mujoco.mju_quat2Mat(mat, np.asarray(q_wxyz, dtype=float))
    return mat.reshape(3, 3)


def rotmat_to_quat(R):
    q = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).reshape(9))
    return q


def pose_to_T(pos, quat_wxyz):
    T = np.eye(4, dtype=float)
    T[:3, :3] = quat_to_rotmat(quat_wxyz)
    T[:3, 3] = np.asarray(pos, dtype=float)
    return T


def T_to_pose(T):
    return T[:3, 3].copy(), rotmat_to_quat(T[:3, :3])


def quat_mul(q1, q2):
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array([
        w1*w2 - x1*x2 - y1*y2 - z1*z2,
        w1*x2 + x1*w2 + y1*z2 - z1*y2,
        w1*y2 - x1*z2 + y1*w2 + z1*x2,
        w1*z2 + x1*y2 - y1*x2 + z1*w2,
    ], dtype=float)


def quat_conj(q):
    return np.array([q[0], -q[1], -q[2], -q[3]], dtype=float)


def quat_orientation_error(q_target, q_current):
    """
    返回小角度旋转误差向量。
    q_target / q_current 都是 wxyz。
    """
    q_err = quat_mul(q_target, quat_conj(q_current))
    if q_err[0] < 0:
        q_err = -q_err
    return 2.0 * q_err[1:4]


def joint_id(model, joint_name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise RuntimeError(f"cannot find joint: {joint_name}")
    return jid


def actuator_id(model, actuator_name):
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, actuator_name)
    if aid < 0:
        raise RuntimeError(f"cannot find actuator: {actuator_name}")
    return aid


def body_id(model, body_name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise RuntimeError(f"cannot find body: {body_name}")
    return bid


def qadr(model, joint_name):
    return int(model.jnt_qposadr[joint_id(model, joint_name)])


def dadr(model, joint_name):
    return int(model.jnt_dofadr[joint_id(model, joint_name)])


def set_qpos_joint(model, data, joint_name, value):
    data.qpos[qadr(model, joint_name)] = float(value)


def get_qpos_joint(model, data, joint_name):
    return float(data.qpos[qadr(model, joint_name)])


def set_ctrl_joint(model, data, joint_name, value):
    aid = actuator_id(model, joint_name + "_pos")
    if model.actuator_ctrllimited[aid]:
        lo, hi = model.actuator_ctrlrange[aid]
        value = float(np.clip(value, lo, hi))
    data.ctrl[aid] = float(value)


def get_ctrl_joint(model, data, joint_name):
    aid = actuator_id(model, joint_name + "_pos")
    return float(data.ctrl[aid])


def set_free_body_pose(model, data, body_name, pos, quat_wxyz):
    bid = body_id(model, body_name)

    free_jid = -1
    for jid in range(model.njnt):
        if int(model.jnt_bodyid[jid]) == bid and int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
            free_jid = jid
            break

    if free_jid < 0:
        raise RuntimeError(f"body {body_name} has no freejoint")

    qa = int(model.jnt_qposadr[free_jid])
    da = int(model.jnt_dofadr[free_jid])

    data.qpos[qa:qa+3] = np.asarray(pos, dtype=float)
    data.qpos[qa+3:qa+7] = np.asarray(quat_wxyz, dtype=float)
    data.qvel[da:da+6] = 0.0


def body_pose(model, data, body_name):
    bid = body_id(model, body_name)
    return data.xpos[bid].copy(), data.xquat[bid].copy()


def contact_counts(model, data):
    hand_box = 0
    fr3_box = 0
    box_table = 0
    box_pedestal = 0
    hand_table = 0

    hand_tokens = ["thumb", "index", "middle", "ring", "pinky", "metacarpals"]

    for i in range(data.ncon):
        c = data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)

        g1_name = name(model, mujoco.mjtObj.mjOBJ_GEOM, g1).lower()
        g2_name = name(model, mujoco.mjtObj.mjOBJ_GEOM, g2).lower()

        b1_name = name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g1])).lower()
        b2_name = name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g2])).lower()

        text = " ".join([g1_name, g2_name, b1_name, b2_name])

        has_box = "grasp_box" in text
        has_hand = any(k in text for k in hand_tokens)
        has_fr3 = "fr3_" in text
        has_table = "table" in text
        has_pedestal = "pedestal" in text

        if has_box and has_hand:
            hand_box += 1
        if has_box and has_fr3:
            fr3_box += 1
        if has_box and has_table:
            box_table += 1
        if has_box and has_pedestal:
            box_pedestal += 1
        if has_hand and has_table:
            hand_table += 1

    return {
        "ncon": int(data.ncon),
        "hand_box": hand_box,
        "fr3_box": fr3_box,
        "box_table": box_table,
        "box_pedestal": box_pedestal,
        "hand_table": hand_table,
    }


def smoothstep(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def set_arm_qpos_and_ctrl(model, data, qdict):
    for j, v in qdict.items():
        set_qpos_joint(model, data, j, v)
        set_ctrl_joint(model, data, j, v)


def set_arm_ctrl(model, data, qdict):
    for j, v in qdict.items():
        set_ctrl_joint(model, data, j, v)


def set_hand_ctrl(model, data, hand_ctrl):
    for j, v in hand_ctrl.items():
        if j in O7_ACTIVE_JOINTS:
            set_ctrl_joint(model, data, j, v)


def open_hand_ctrl(model, data):
    for j in O7_ACTIVE_JOINTS:
        set_ctrl_joint(model, data, j, 0.0)


def set_approach_hand_ctrl(model, data, o7_template_ctrl):
    """
    正确的 approach 手型：
    1. thumb_cmc_roll / thumb_cmc_yaw 先到模板值，用来让大拇指位于正确对握侧；
    2. thumb_cmc_pitch 必须保持 0，避免大拇指提前闭合扫到 box；
    3. 四指保持打开。
    """
    set_ctrl_joint(model, data, "thumb_cmc_roll", o7_template_ctrl["thumb_cmc_roll"])
    set_ctrl_joint(model, data, "thumb_cmc_yaw", o7_template_ctrl["thumb_cmc_yaw"])
    set_ctrl_joint(model, data, "thumb_cmc_pitch", 0.0)

    for j in ["index_mcp_pitch", "middle_mcp_pitch", "ring_mcp_pitch", "pinky_mcp_pitch"]:
        set_ctrl_joint(model, data, j, 0.0)


def set_close_hand_ctrl(model, data, o7_template_ctrl, alpha):
    """
    正确的 close 手型：
    thumb_cmc_roll / thumb_cmc_yaw 始终保持模板值；
    thumb_cmc_pitch 和四指一起从 0 闭合到模板值。
    """
    a = smoothstep(alpha)

    set_ctrl_joint(model, data, "thumb_cmc_roll", o7_template_ctrl["thumb_cmc_roll"])
    set_ctrl_joint(model, data, "thumb_cmc_yaw", o7_template_ctrl["thumb_cmc_yaw"])
    set_ctrl_joint(model, data, "thumb_cmc_pitch", a * o7_template_ctrl["thumb_cmc_pitch"])

    for j in ["index_mcp_pitch", "middle_mcp_pitch", "ring_mcp_pitch", "pinky_mcp_pitch"]:
        set_ctrl_joint(model, data, j, a * o7_template_ctrl[j])


def solve_franka_ik_dls(
    model,
    seed_q,
    target_body,
    target_pos,
    target_quat,
    max_iter=300,
    damping=0.03,
    step_scale=0.7,
    max_step=0.08,
    pos_tol=1e-4,
    rot_tol=2e-3,
    rot_weight=0.6,
):
    """
    只解 Franka 7 个关节。
    target_body 当前默认 fr3_link7。
    """
    data = mujoco.MjData(model)

    for j, v in seed_q.items():
        set_qpos_joint(model, data, j, v)

    mujoco.mj_forward(model, data)

    target_bid = body_id(model, target_body)
    dofs = np.array([dadr(model, j) for j in FRANKA_JOINTS], dtype=int)

    best_err = 1e9
    best_q = None

    for it in range(max_iter):
        mujoco.mj_forward(model, data)

        cur_pos = data.xpos[target_bid].copy()
        cur_quat = data.xquat[target_bid].copy()

        pos_err = np.asarray(target_pos, dtype=float) - cur_pos
        rot_err = quat_orientation_error(np.asarray(target_quat, dtype=float), cur_quat)

        err = np.concatenate([pos_err, rot_weight * rot_err])

        pos_norm = float(np.linalg.norm(pos_err))
        rot_norm = float(np.linalg.norm(rot_err))
        err_norm = float(np.linalg.norm(err))

        if err_norm < best_err:
            best_err = err_norm
            best_q = {j: get_qpos_joint(model, data, j) for j in FRANKA_JOINTS}

        if pos_norm < pos_tol and rot_norm < rot_tol:
            return best_q, {
                "success": True,
                "iter": it,
                "pos_err": pos_norm,
                "rot_err": rot_norm,
                "err_norm": err_norm,
            }

        jacp = np.zeros((3, model.nv), dtype=float)
        jacr = np.zeros((3, model.nv), dtype=float)
        mujoco.mj_jacBody(model, data, jacp, jacr, target_bid)

        J_full = np.vstack([
            jacp[:, dofs],
            rot_weight * jacr[:, dofs],
        ])

        A = J_full @ J_full.T + (damping ** 2) * np.eye(6)
        dq = J_full.T @ np.linalg.solve(A, err)

        dq = step_scale * dq

        dq_norm = float(np.linalg.norm(dq))
        if dq_norm > max_step:
            dq *= max_step / (dq_norm + 1e-12)

        for idx, j in enumerate(FRANKA_JOINTS):
            jid = joint_id(model, j)
            qa = int(model.jnt_qposadr[jid])
            data.qpos[qa] += dq[idx]

            if model.jnt_limited[jid]:
                lo, hi = model.jnt_range[jid]
                data.qpos[qa] = np.clip(data.qpos[qa], lo, hi)

    return best_q, {
        "success": False,
        "iter": max_iter,
        "best_err": best_err,
    }


def interpolate_dict(a, b, alpha):
    out = {}
    for k in sorted(set(a.keys()) | set(b.keys())):
        av = a.get(k, 0.0)
        bv = b.get(k, 0.0)
        out[k] = av * (1.0 - alpha) + bv * alpha
    return out


def run_stage(model, data, viewer, label, duration, log_dt, ctrl_callback):
    dt = float(model.opt.timestep)
    steps = max(1, int(duration / dt))
    log_every = max(1, int(log_dt / dt))

    print(f"\n[STAGE] {label}, duration={duration:.2f}s")

    for k in range(steps + 1):
        alpha = k / steps
        t = k * dt

        ctrl_callback(alpha, t)

        if k % log_every == 0 or k == steps:
            box = body_pose(model, data, "grasp_box")[0]
            target = body_pose(model, data, "fr3_link7")[0]
            counts = contact_counts(model, data)

            print(
                f"  t={t:6.2f} "
                f"box_z={box[2]:.5f} "
                f"fr3_link7={target} "
                f"hand_box={counts['hand_box']:2d} "
                f"fr3_box={counts['fr3_box']:2d} "
                f"box_table={counts['box_table']:2d} "
                f"box_pedestal={counts['box_pedestal']:2d}"
            )

        mujoco.mj_step(model, data)

        if viewer is not None:
            viewer.sync()
            time.sleep(dt)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--template", default=str(DEFAULT_TEMPLATE))
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--no-viewer", action="store_true")
    ap.add_argument("--move-duration", type=float, default=5.0)
    ap.add_argument("--close-duration", type=float, default=3.0)
    ap.add_argument("--lift-duration", type=float, default=5.0)
    ap.add_argument("--hold-duration", type=float, default=6.0)
    ap.add_argument("--lift-delta", type=float, default=0.12)
    ap.add_argument("--log-dt", type=float, default=0.5)
    ap.add_argument("--box-shift", type=str, default="0 0 0")
    ap.add_argument("--ik-seed", choices=["template", "start"], default="template")
    args = ap.parse_args()

    model_path = resolve_path(args.model)
    template_path = resolve_path(args.template)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    template = load_json(template_path)

    target_body = template["target_body"]
    box_body = template["box_body"]

    T_world_box_template = np.asarray(template["T_world_box"], dtype=float)
    T_box_target = np.asarray(template["T_box_target"], dtype=float)

    box_shift = np.fromstring(args.box_shift, sep=" ", dtype=float)
    if box_shift.shape[0] != 3:
        raise RuntimeError("--box-shift should be like: '0 0 0'")

    T_world_box = T_world_box_template.copy()
    T_world_box[:3, 3] += box_shift

    T_world_target = T_world_box @ T_box_target
    target_pos, target_quat = T_to_pose(T_world_target)
    box_pos, box_quat = T_to_pose(T_world_box)

    franka_template_q = template["franka_ctrl"]
    o7_template_ctrl = template["o7_active_ctrl"]

    if args.ik_seed == "template":
        seed_q = franka_template_q.copy()
    else:
        seed_q = START_ARM.copy()

    q_ik, ik_info = solve_franka_ik_dls(
        model=model,
        seed_q=seed_q,
        target_body=target_body,
        target_pos=target_pos,
        target_quat=target_quat,
    )

    print("\n========== FR3 + O7 IK REPLAY V10 ==========")
    print("model        :", model_path)
    print("template     :", template_path)
    print("box body     :", box_body)
    print("target body  :", target_body)
    print("box shift    :", box_shift)
    print("target pos   :", target_pos)
    print("target quat  :", target_quat)
    print("ik info      :", ik_info)
    print()
    print("IK result q:")
    for j in FRANKA_JOINTS:
        print(f"  {j:12s}: {q_ik[j]:+.6f}")
    print()
    print("O7 ctrl from template:")
    for j in O7_ACTIVE_JOINTS:
        print(f"  {j:18s}: {o7_template_ctrl[j]:+.6f}")
    print("============================================\n")

    # 初始化真实仿真 data
    data = mujoco.MjData(model)

    set_free_body_pose(model, data, box_body, box_pos, box_quat)

    set_arm_qpos_and_ctrl(model, data, START_ARM)
    # approach 初始手型：thumb roll/yaw 到位，thumb pitch 和四指打开
    set_approach_hand_ctrl(model, data, o7_template_ctrl)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    start_arm = START_ARM.copy()
    ik_arm = q_ik.copy()

    lift_joint = "fr3_joint4"
    lift_aid = actuator_id(model, lift_joint + "_pos")
    lift_start = ik_arm[lift_joint]
    lift_target = lift_start + args.lift_delta

    def move_cb(alpha, t):
        a = smoothstep(alpha)
        qcmd = interpolate_dict(start_arm, ik_arm, a)
        set_arm_ctrl(model, data, qcmd)
        set_approach_hand_ctrl(model, data, o7_template_ctrl)

    def close_cb(alpha, t):
        set_arm_ctrl(model, data, ik_arm)
        set_close_hand_ctrl(model, data, o7_template_ctrl, alpha)

    def lift_cb(alpha, t):
        set_arm_ctrl(model, data, ik_arm)
        set_hand_ctrl(model, data, o7_template_ctrl)

        a = smoothstep(alpha)
        data.ctrl[lift_aid] = lift_start + a * args.lift_delta

    def hold_cb(alpha, t):
        set_arm_ctrl(model, data, ik_arm)
        set_hand_ctrl(model, data, o7_template_ctrl)
        data.ctrl[lift_aid] = lift_target

    def run_all(viewer=None):
        run_stage(model, data, viewer, "move arm by IK result, thumb yaw/roll set, thumb pitch open", args.move_duration, args.log_dt, move_cb)
        run_stage(model, data, viewer, "close thumb pitch and four fingers together", args.close_duration, args.log_dt, close_cb)
        run_stage(model, data, viewer, "lift by fr3_joint4", args.lift_duration, args.log_dt, lift_cb)
        run_stage(model, data, viewer, "air hold", args.hold_duration, args.log_dt, hold_cb)
        print("\n[IK DEMO DONE] 关闭 viewer 退出。")

    use_viewer = args.viewer or (not args.no_viewer)

    if use_viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = np.array([0.52, 0.02, 0.25])
            viewer.cam.distance = 1.1
            viewer.cam.azimuth = 125
            viewer.cam.elevation = -18
            viewer.opt.geomgroup[3] = 0
            viewer.opt.geomgroup[4] = 1

            run_all(viewer)

            while viewer.is_running():
                viewer.sync()
                time.sleep(0.01)
    else:
        run_all(None)


if __name__ == "__main__":
    main()
