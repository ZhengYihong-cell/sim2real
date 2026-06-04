#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import time
from datetime import datetime
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


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def name(model, objtype, idx):
    return mujoco.mj_id2name(model, objtype, idx) or ""


def quat_to_rotmat(q_wxyz):
    mat = np.zeros(9, dtype=float)
    mujoco.mju_quat2Mat(mat, np.asarray(q_wxyz, dtype=float))
    return mat.reshape(3, 3)


def rotmat_to_quat(R):
    q = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).reshape(9))
    return q


def T_to_pose(T):
    pos = T[:3, 3].copy()
    quat = rotmat_to_quat(T[:3, :3])
    return pos, quat


def yaw_matrix(yaw):
    c = np.cos(yaw)
    s = np.sin(yaw)
    T = np.eye(4)
    T[:3, :3] = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])
    return T


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


def set_arm_qpos_and_ctrl(model, data, qdict):
    for j, v in qdict.items():
        set_qpos_joint(model, data, j, v)
        set_ctrl_joint(model, data, j, v)


def set_arm_ctrl(model, data, qdict):
    for j, v in qdict.items():
        set_ctrl_joint(model, data, j, v)


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


def body_pos(model, data, body_name):
    bid = body_id(model, body_name)
    return data.xpos[bid].copy()


def smoothstep(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def interpolate_dict(a, b, alpha):
    out = {}
    for k in sorted(set(a.keys()) | set(b.keys())):
        out[k] = a.get(k, 0.0) * (1.0 - alpha) + b.get(k, 0.0) * alpha
    return out


def set_approach_hand_ctrl(model, data, o7_ctrl):
    """
    通用 approach 手型：
    thumb_cmc_roll / thumb_cmc_yaw 到模板值；
    thumb_cmc_pitch = 0；
    四指 = 0。
    """
    set_ctrl_joint(model, data, "thumb_cmc_roll", o7_ctrl["thumb_cmc_roll"])
    set_ctrl_joint(model, data, "thumb_cmc_yaw", o7_ctrl["thumb_cmc_yaw"])
    set_ctrl_joint(model, data, "thumb_cmc_pitch", 0.0)

    for j in O7_FOUR_FINGER_JOINTS:
        set_ctrl_joint(model, data, j, 0.0)


def set_close_hand_ctrl(model, data, o7_ctrl, alpha):
    """
    通用 close 手型：
    thumb_cmc_roll / thumb_cmc_yaw 保持模板值；
    thumb_cmc_pitch 和四指一起闭合。
    """
    a = smoothstep(alpha)

    set_ctrl_joint(model, data, "thumb_cmc_roll", o7_ctrl["thumb_cmc_roll"])
    set_ctrl_joint(model, data, "thumb_cmc_yaw", o7_ctrl["thumb_cmc_yaw"])
    set_ctrl_joint(model, data, "thumb_cmc_pitch", a * o7_ctrl["thumb_cmc_pitch"])

    for j in O7_FOUR_FINGER_JOINTS:
        set_ctrl_joint(model, data, j, a * o7_ctrl[j])


def set_full_grasp_hand_ctrl(model, data, o7_ctrl):
    for j in O7_ACTIVE_JOINTS:
        set_ctrl_joint(model, data, j, o7_ctrl[j])


def parse_tokens(s):
    if not s:
        return []
    return [x.strip().lower() for x in s.replace(",", " ").split() if x.strip()]


def contact_counts(model, data, object_tokens, support_tokens):
    hand_object = 0
    fr3_object = 0
    object_support = 0
    hand_support = 0

    support_detail = {tok: 0 for tok in support_tokens}

    hand_tokens = ["thumb", "index", "middle", "ring", "pinky", "metacarpals"]
    fr3_tokens = ["fr3_"]

    for i in range(data.ncon):
        c = data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)

        g1_name = name(model, mujoco.mjtObj.mjOBJ_GEOM, g1).lower()
        g2_name = name(model, mujoco.mjtObj.mjOBJ_GEOM, g2).lower()

        b1_name = name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g1])).lower()
        b2_name = name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g2])).lower()

        text = " ".join([g1_name, g2_name, b1_name, b2_name])

        has_object = any(tok in text for tok in object_tokens)
        has_hand = any(tok in text for tok in hand_tokens)
        has_fr3 = any(tok in text for tok in fr3_tokens)
        has_support = any(tok in text for tok in support_tokens)

        if has_object and has_hand:
            hand_object += 1
        if has_object and has_fr3:
            fr3_object += 1
        if has_object and has_support:
            object_support += 1
            for tok in support_tokens:
                if tok in text:
                    support_detail[tok] += 1
        if has_hand and has_support:
            hand_support += 1

    return {
        "ncon": int(data.ncon),
        "hand_object": hand_object,
        "fr3_object": fr3_object,
        "object_support": object_support,
        "hand_support": hand_support,
        "support_detail": support_detail,
    }


def get_template_matrix(template, generic_key, old_key):
    if generic_key in template:
        return np.asarray(template[generic_key], dtype=float)
    if old_key in template:
        return np.asarray(template[old_key], dtype=float)
    raise RuntimeError(f"template missing {generic_key} / {old_key}")


def get_template_object_body(template):
    return template.get("object_body", template.get("box_body", "grasp_box"))


def get_template_target_transform(template):
    return get_template_matrix(template, "T_object_target", "T_box_target")


def get_template_world_object(template):
    return get_template_matrix(template, "T_world_object", "T_world_box")


def make_random_object_pose(template, rng, xy_range, yaw_range_deg, z_shift=0.0):
    T0 = get_template_world_object(template)

    dx = rng.uniform(-xy_range, xy_range)
    dy = rng.uniform(-xy_range, xy_range)
    yaw = np.deg2rad(rng.uniform(-yaw_range_deg, yaw_range_deg))

    T = T0.copy()
    T[:3, 3] = T0[:3, 3] + np.array([dx, dy, z_shift])
    T[:3, :3] = yaw_matrix(yaw)[:3, :3] @ T0[:3, :3]

    pos, quat = T_to_pose(T)

    return T, pos, quat, {
        "dx": float(dx),
        "dy": float(dy),
        "z_shift": float(z_shift),
        "yaw_deg": float(np.rad2deg(yaw)),
    }


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
    data = mujoco.MjData(model)

    for j, v in seed_q.items():
        set_qpos_joint(model, data, j, v)

    mujoco.mj_forward(model, data)

    target_bid = body_id(model, target_body)
    dofs = np.array([dadr(model, j) for j in FRANKA_JOINTS], dtype=int)

    best_err = 1e9
    best_q = {j: seed_q[j] for j in FRANKA_JOINTS}
    best_info = {}

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
            best_info = {
                "iter": it,
                "pos_err": pos_norm,
                "rot_err": rot_norm,
                "err_norm": err_norm,
            }

        if pos_norm < pos_tol and rot_norm < rot_tol:
            best_info["success"] = True
            return best_q, best_info

        jacp = np.zeros((3, model.nv), dtype=float)
        jacr = np.zeros((3, model.nv), dtype=float)
        mujoco.mj_jacBody(model, data, jacp, jacr, target_bid)

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

        for idx, j in enumerate(FRANKA_JOINTS):
            jid = joint_id(model, j)
            qa = int(model.jnt_qposadr[jid])
            data.qpos[qa] += dq[idx]

            if model.jnt_limited[jid]:
                lo, hi = model.jnt_range[jid]
                data.qpos[qa] = np.clip(data.qpos[qa], lo, hi)

    best_info["success"] = False
    best_info["best_err"] = best_err
    return best_q, best_info


def run_stage(model, data, viewer, label, duration, log_dt, ctrl_callback, logs, object_body, object_tokens, support_tokens):
    dt = float(model.opt.timestep)
    steps = max(1, int(duration / dt))
    log_every = max(1, int(log_dt / dt))

    print(f"\n[STAGE] {label}, duration={duration:.2f}s")

    for k in range(steps + 1):
        alpha = k / steps
        t = k * dt

        ctrl_callback(alpha, t)

        if k % log_every == 0 or k == steps:
            obj = body_pos(model, data, object_body)
            fr3 = body_pos(model, data, "fr3_link7")
            counts = contact_counts(model, data, object_tokens, support_tokens)

            row = {
                "stage": label,
                "t": float(t),
                "object_z": float(obj[2]),
                "fr3_link7": fr3.tolist(),
                **counts,
            }
            logs.append(row)

            print(
                f"  t={t:6.2f} "
                f"object_z={obj[2]:.5f} "
                f"hand_object={counts['hand_object']:2d} "
                f"fr3_object={counts['fr3_object']:2d} "
                f"object_support={counts['object_support']:2d} "
                f"hand_support={counts['hand_support']:2d}"
            )

        mujoco.mj_step(model, data)

        if viewer is not None:
            viewer.sync()
            time.sleep(dt)


def run_one_trial(model, data, template, rng, args, trial_idx, viewer=None):
    object_body = args.object_body or get_template_object_body(template)
    target_body = args.target_body or template["target_body"]

    object_token_string = args.object_token or object_body
    object_tokens = parse_tokens(object_token_string)
    support_tokens = parse_tokens(args.support_tokens)

    T_world_object, object_pos, object_quat, random_info = make_random_object_pose(
        template,
        rng,
        args.xy_range,
        args.yaw_range_deg,
        args.z_shift,
    )

    T_object_target = get_template_target_transform(template)
    T_world_target = T_world_object @ T_object_target
    target_pos, target_quat = T_to_pose(T_world_target)

    seed_q = template["franka_ctrl"]
    o7_ctrl = template["o7_active_ctrl"]

    q_ik, ik_info = solve_franka_ik_dls(
        model,
        seed_q=seed_q,
        target_body=target_body,
        target_pos=target_pos,
        target_quat=target_quat,
    )

    print("\n\n===================================================")
    print(f"[TRIAL {trial_idx}]")
    print("object_body :", object_body)
    print("object_tokens:", object_tokens)
    print("support_tokens:", support_tokens)
    print("random_info :", random_info)
    print("target_body :", target_body)
    print("target_pos  :", target_pos)
    print("ik_info     :", ik_info)
    print("===================================================")

    data.qpos[:] = 0.0
    data.qvel[:] = 0.0
    data.ctrl[:] = 0.0

    set_free_body_pose(model, data, object_body, object_pos, object_quat)
    set_arm_qpos_and_ctrl(model, data, START_ARM)
    set_approach_hand_ctrl(model, data, o7_ctrl)
    mujoco.mj_forward(model, data)

    logs = []
    z0 = float(body_pos(model, data, object_body)[2])

    lift_joint = args.lift_joint
    lift_aid = actuator_id(model, lift_joint + "_pos")
    lift_start = q_ik[lift_joint]
    lift_target = lift_start + args.lift_delta

    def move_cb(alpha, t):
        a = smoothstep(alpha)
        qcmd = interpolate_dict(START_ARM, q_ik, a)
        set_arm_ctrl(model, data, qcmd)
        set_approach_hand_ctrl(model, data, o7_ctrl)

    def close_cb(alpha, t):
        set_arm_ctrl(model, data, q_ik)
        set_close_hand_ctrl(model, data, o7_ctrl, alpha)

    def lift_cb(alpha, t):
        set_arm_ctrl(model, data, q_ik)
        set_full_grasp_hand_ctrl(model, data, o7_ctrl)
        a = smoothstep(alpha)
        data.ctrl[lift_aid] = lift_start + a * args.lift_delta

    def hold_cb(alpha, t):
        set_arm_ctrl(model, data, q_ik)
        set_full_grasp_hand_ctrl(model, data, o7_ctrl)
        data.ctrl[lift_aid] = lift_target

    run_stage(model, data, viewer, "move_ik_approach", args.move_duration, args.log_dt, move_cb, logs, object_body, object_tokens, support_tokens)
    run_stage(model, data, viewer, "close_hand", args.close_duration, args.log_dt, close_cb, logs, object_body, object_tokens, support_tokens)
    run_stage(model, data, viewer, "lift", args.lift_duration, args.log_dt, lift_cb, logs, object_body, object_tokens, support_tokens)
    run_stage(model, data, viewer, "air_hold", args.hold_duration, args.log_dt, hold_cb, logs, object_body, object_tokens, support_tokens)

    final_counts = contact_counts(model, data, object_tokens, support_tokens)
    final_z = float(body_pos(model, data, object_body)[2])
    max_z = max(row["object_z"] for row in logs)
    min_hand_object_after_lift = min(
        row["hand_object"]
        for row in logs
        if row["stage"] in ["lift", "air_hold"]
    )

    success = (
        ik_info.get("success", False)
        and final_counts["hand_object"] >= args.min_final_hand_object
        and final_counts["fr3_object"] == 0
        and final_counts["object_support"] == 0
        and final_z >= z0 + args.min_final_rise
    )

    summary = {
        "trial": trial_idx,
        "object_body": object_body,
        "object_tokens": object_tokens,
        "support_tokens": support_tokens,
        "random_info": random_info,
        "ik_info": ik_info,
        "ik_q": q_ik,
        "target_body": target_body,
        "target_pos": target_pos.tolist(),
        "target_quat": target_quat.tolist(),
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

    # 泛化接口
    ap.add_argument("--object-body", default="", help="object body name; empty means use template object_body/box_body")
    ap.add_argument("--object-token", default="", help="tokens for detecting object contacts; empty means object body name")
    ap.add_argument("--support-tokens", default="pedestal table", help="support tokens, e.g. 'pedestal table'")
    ap.add_argument("--target-body", default="", help="IK target body; empty means use template target_body")

    ap.add_argument("--trials", type=int, default=1)
    ap.add_argument("--seed", type=int, default=42)

    ap.add_argument("--xy-range", type=float, default=0.02)
    ap.add_argument("--yaw-range-deg", type=float, default=10.0)
    ap.add_argument("--z-shift", type=float, default=0.0)

    ap.add_argument("--move-duration", type=float, default=5.0)
    ap.add_argument("--close-duration", type=float, default=3.0)
    ap.add_argument("--lift-duration", type=float, default=5.0)
    ap.add_argument("--hold-duration", type=float, default=6.0)

    ap.add_argument("--lift-joint", default="fr3_joint4")
    ap.add_argument("--lift-delta", type=float, default=0.18)
    ap.add_argument("--log-dt", type=float, default=0.5)

    ap.add_argument("--min-final-hand-object", type=int, default=3)
    ap.add_argument("--min-final-rise", type=float, default=0.005)

    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--out", default="")

    args = ap.parse_args()

    model_path = resolve_path(args.model)
    template_path = resolve_path(args.template)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    template = load_json(template_path)

    rng = np.random.default_rng(args.seed)

    if args.out:
        out_path = resolve_path(args.out)
    else:
        stamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        out_path = PROJECT / "diagnostics" / f"fr3_o7_object_ik_grasp_v12_{stamp}.json"

    print("\n========== FR3 + O7 OBJECT IK GRASP V12 ==========")
    print("model          :", model_path)
    print("template       :", template_path)
    print("object_body arg:", args.object_body)
    print("object_token   :", args.object_token)
    print("support_tokens :", args.support_tokens)
    print("target_body arg:", args.target_body)
    print("trials         :", args.trials)
    print("seed           :", args.seed)
    print("xy_range       :", args.xy_range)
    print("yaw_range_deg  :", args.yaw_range_deg)
    print("lift_joint     :", args.lift_joint)
    print("lift_delta     :", args.lift_delta)
    print("out            :", out_path)
    print("==================================================\n")

    results = []

    if args.viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = np.array([0.52, 0.02, 0.25])
            viewer.cam.distance = 1.1
            viewer.cam.azimuth = 125
            viewer.cam.elevation = -18
            viewer.opt.geomgroup[3] = 0
            viewer.opt.geomgroup[4] = 1

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
        "format": "fr3_o7_object_ik_grasp_v12_result",
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

    print("\n========== V12 FINAL SUMMARY ==========")
    print("success_count:", success_count)
    print("trials       :", args.trials)
    print("success_rate :", success_count / max(1, args.trials))
    print("saved        :", out_path)
    print("=======================================\n")


if __name__ == "__main__":
    main()
