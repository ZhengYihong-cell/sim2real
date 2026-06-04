#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import numpy as np
import mujoco


PROJECT = Path.home() / "Projects/o7_mujoco_sim"

DEFAULT_MODEL = PROJECT / "models/fr3_o7/fr3_o7_actuated_scene_v13_cylinder.xml"
DEFAULT_RECORD = PROJECT / "records/fr3_o7_object_keyboard_pose_v13_20260516_131858_708337.json"
DEFAULT_OUT = PROJECT / "records/fr3_o7_grasp_template_cylinder_v1.json"


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


def resolve_path(p):
    p = Path(p).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


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


def inv_T(T):
    out = np.eye(4, dtype=float)
    R = T[:3, :3]
    p = T[:3, 3]
    out[:3, :3] = R.T
    out[:3, 3] = -R.T @ p
    return out


def body_pose(model, data, body_name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise RuntimeError(f"cannot find body: {body_name}")
    return data.xpos[bid].copy(), data.xquat[bid].copy()


def joint_qpos(model, data, joint_name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise RuntimeError(f"cannot find joint: {joint_name}")
    return float(data.qpos[int(model.jnt_qposadr[jid])])


def actuator_ctrl(model, data, joint_name):
    aid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, joint_name + "_pos")
    if aid < 0:
        raise RuntimeError(f"cannot find actuator: {joint_name}_pos")
    return float(data.ctrl[aid])


def apply_record(model, data, record):
    qpos = np.asarray(record["qpos"], dtype=float)
    ctrl = np.asarray(record["ctrl"], dtype=float)

    if len(qpos) != model.nq:
        raise RuntimeError(f"qpos length mismatch: record={len(qpos)}, model.nq={model.nq}")
    if len(ctrl) != model.nu:
        raise RuntimeError(f"ctrl length mismatch: record={len(ctrl)}, model.nu={model.nu}")

    data.qpos[:] = qpos
    data.ctrl[:] = ctrl
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--record", default=str(DEFAULT_RECORD))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--object-body", default="")
    ap.add_argument("--target-body", default="fr3_link7")
    args = ap.parse_args()

    model_path = resolve_path(args.model)
    record_path = resolve_path(args.record)
    out_path = resolve_path(args.out)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    record = load_json(record_path)
    apply_record(model, data, record)

    object_body = args.object_body or record.get("object_body", "grasp_cylinder")
    target_body = args.target_body

    object_pos, object_quat = body_pose(model, data, object_body)
    target_pos, target_quat = body_pose(model, data, target_body)

    T_world_object = pose_to_T(object_pos, object_quat)
    T_world_target = pose_to_T(target_pos, target_quat)
    T_object_target = inv_T(T_world_object) @ T_world_target

    rel_pos, rel_quat = T_to_pose(T_object_target)

    franka_qpos = {j: joint_qpos(model, data, j) for j in FRANKA_JOINTS}
    franka_ctrl = {j: actuator_ctrl(model, data, j) for j in FRANKA_JOINTS}
    o7_active_qpos = {j: joint_qpos(model, data, j) for j in O7_ACTIVE_JOINTS}
    o7_active_ctrl = {j: actuator_ctrl(model, data, j) for j in O7_ACTIVE_JOINTS}

    template = {
        "format": "fr3_o7_object_grasp_template_v1",
        "source_model": str(model_path),
        "source_record": str(record_path),

        "object_body": object_body,
        "object_token": record.get("object_token", object_body),
        "support_tokens": record.get("support_tokens", "pedestal table"),
        "target_body": target_body,

        "T_world_object": T_world_object.tolist(),
        "T_world_target": T_world_target.tolist(),
        "T_object_target": T_object_target.tolist(),

        "world_object_pose": {
            "body": object_body,
            "pos": object_pos.tolist(),
            "quat_wxyz": object_quat.tolist(),
        },
        "world_target_pose": {
            "body": target_body,
            "pos": target_pos.tolist(),
            "quat_wxyz": target_quat.tolist(),
        },
        "object_to_target_pose": {
            "body": target_body,
            "pos": rel_pos.tolist(),
            "quat_wxyz": rel_quat.tolist(),
        },

        "franka_qpos": franka_qpos,
        "franka_ctrl": franka_ctrl,
        "o7_active_qpos": o7_active_qpos,
        "o7_active_ctrl": o7_active_ctrl,

        "full_qpos": data.qpos.tolist(),
        "full_ctrl": data.ctrl.tolist(),

        "source_contact_summary": record.get("contact_summary", {}),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(template, f, indent=2)

    print("\n========== EXTRACT OBJECT GRASP TEMPLATE V14 ==========")
    print("model        :", model_path)
    print("record       :", record_path)
    print("out          :", out_path)
    print()
    print("object_body  :", object_body)
    print("target_body  :", target_body)
    print("object_pos   :", object_pos)
    print("object_quat  :", object_quat)
    print("target_pos   :", target_pos)
    print("target_quat  :", target_quat)
    print()
    print("T_object_target pos :", rel_pos)
    print("T_object_target quat:", rel_quat)
    print()
    print("Franka ctrl:")
    for k, v in franka_ctrl.items():
        print(f"  {k:12s}: {v:+.6f}")
    print()
    print("O7 active ctrl:")
    for k, v in o7_active_ctrl.items():
        print(f"  {k:18s}: {v:+.6f}")
    print()
    print("source contact summary:")
    print(record.get("contact_summary", {}))
    print("=======================================================\n")


if __name__ == "__main__":
    main()
