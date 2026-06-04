#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import numpy as np
import mujoco


PROJECT = Path.home() / "Projects/o7_mujoco_sim"

DEFAULT_MODEL = PROJECT / "models/fr3_o7/fr3_o7_actuated_scene_v1f_stable_hand.xml"
DEFAULT_RECORD = PROJECT / "records/stable_fr3_o7_grasp_candidate_v1.json"
DEFAULT_OUT = PROJECT / "records/fr3_o7_grasp_template_v1.json"


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


TARGET_BODY_CANDIDATES = [
    # 理想情况：如果 MuJoCo 中保留了手基座
    "hand_base_link",
    "hand_base",
    "o7_hand_base",
    "right_hand_base",

    # Franka 末端/法兰，最适合作为 IK target
    "fr3_link8",
    "fr3_link7",

    # 最后兜底：这个 body 之前已经确认存在，但它更像手部参考点，不是腕部
    "thumb_metacarpals_base1",
]


DEBUG_BODY_CANDIDATES = [
    "thumb_metacarpals_base1",
    "thumb_distal",
    "index_distal",
    "middle_distal",
]


def resolve_path(p):
    p = Path(p).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


def mj_name(model, objtype, idx):
    return mujoco.mj_id2name(model, objtype, idx) or ""


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def list_body_names(model):
    return [
        mj_name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(model.nbody)
    ]


def print_relevant_bodies(model):
    print("\n========== RELEVANT BODY NAMES ==========")
    keywords = [
        "fr3",
        "hand",
        "thumb",
        "index",
        "middle",
        "ring",
        "pinky",
        "base",
        "link7",
        "link8",
    ]

    for i in range(model.nbody):
        bname = mj_name(model, mujoco.mjtObj.mjOBJ_BODY, i)
        low = bname.lower()
        if any(k in low for k in keywords):
            print(f"{i:3d}  {bname}")
    print("=========================================\n")


def pick_body(model, requested, candidates, label):
    body_names = set(list_body_names(model))

    if requested and requested != "auto":
        if requested in body_names:
            return requested

        print_relevant_bodies(model)
        raise RuntimeError(f"cannot find {label} body: {requested}")

    for b in candidates:
        if b in body_names:
            return b

    print_relevant_bodies(model)
    raise RuntimeError(f"cannot auto-pick {label} body from candidates: {candidates}")


def quat_to_rotmat(q_wxyz):
    q = np.asarray(q_wxyz, dtype=float)
    mat = np.zeros(9, dtype=float)
    mujoco.mju_quat2Mat(mat, q)
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
    pos = T[:3, 3].copy()
    quat = rotmat_to_quat(T[:3, :3])
    return pos, quat


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
        print_relevant_bodies(model)
        raise RuntimeError(f"cannot find body: {body_name}")
    return data.xpos[bid].copy(), data.xquat[bid].copy()


def joint_qpos(model, data, joint_name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        raise RuntimeError(f"cannot find joint: {joint_name}")
    qadr = model.jnt_qposadr[jid]
    return float(data.qpos[qadr])


def actuator_ctrl(model, data, joint_name):
    aid = mujoco.mj_name2id(
        model,
        mujoco.mjtObj.mjOBJ_ACTUATOR,
        joint_name + "_pos",
    )
    if aid < 0:
        raise RuntimeError(f"cannot find actuator: {joint_name}_pos")
    return float(data.ctrl[aid])


def apply_record(model, data, record):
    qpos = np.asarray(record["qpos"], dtype=float)
    ctrl = np.asarray(record["ctrl"], dtype=float)

    if len(qpos) != model.nq:
        raise RuntimeError(
            f"qpos length mismatch: record={len(qpos)}, model.nq={model.nq}"
        )

    if len(ctrl) != model.nu:
        raise RuntimeError(
            f"ctrl length mismatch: record={len(ctrl)}, model.nu={model.nu}"
        )

    data.qpos[:] = qpos
    data.ctrl[:] = ctrl
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--record", default=str(DEFAULT_RECORD))
    ap.add_argument("--out", default=str(DEFAULT_OUT))
    ap.add_argument("--box-body", default="grasp_box")

    # 这里默认 auto，不再死写 hand_base_link
    ap.add_argument(
        "--target-body",
        default="auto",
        help="IK target body. auto will try hand_base/fr3_link8/fr3_link7/thumb_metacarpals_base1",
    )
    ap.add_argument(
        "--debug-body",
        default="auto",
        help="debug hand body for relative pose print",
    )
    ap.add_argument("--list-bodies", action="store_true")

    args = ap.parse_args()

    model_path = resolve_path(args.model)
    record_path = resolve_path(args.record)
    out_path = resolve_path(args.out)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    if args.list_bodies:
        print_relevant_bodies(model)

    record = load_json(record_path)
    apply_record(model, data, record)

    target_body = pick_body(
        model,
        args.target_body,
        TARGET_BODY_CANDIDATES,
        "target",
    )

    debug_body = pick_body(
        model,
        args.debug_body,
        DEBUG_BODY_CANDIDATES,
        "debug",
    )

    box_pos, box_quat = body_pose(model, data, args.box_body)
    target_pos, target_quat = body_pose(model, data, target_body)
    debug_pos, debug_quat = body_pose(model, data, debug_body)

    T_world_box = pose_to_T(box_pos, box_quat)
    T_world_target = pose_to_T(target_pos, target_quat)
    T_world_debug = pose_to_T(debug_pos, debug_quat)

    T_box_target = inv_T(T_world_box) @ T_world_target
    T_box_debug = inv_T(T_world_box) @ T_world_debug

    rel_target_pos, rel_target_quat = T_to_pose(T_box_target)
    rel_debug_pos, rel_debug_quat = T_to_pose(T_box_debug)

    franka_qpos = {
        j: joint_qpos(model, data, j)
        for j in FRANKA_JOINTS
    }
    franka_ctrl = {
        j: actuator_ctrl(model, data, j)
        for j in FRANKA_JOINTS
    }

    o7_active_qpos = {
        j: joint_qpos(model, data, j)
        for j in O7_ACTIVE_JOINTS
    }
    o7_active_ctrl = {
        j: actuator_ctrl(model, data, j)
        for j in O7_ACTIVE_JOINTS
    }

    template = {
        "format": "fr3_o7_grasp_template_v1",
        "source_model": str(model_path),
        "source_record": str(record_path),

        "box_body": args.box_body,

        # 注意：这里的 target_body 就是后续 IK 要追踪的 body
        "target_body": target_body,
        "debug_body": debug_body,

        "T_world_box": T_world_box.tolist(),
        "T_world_target": T_world_target.tolist(),
        "T_box_target": T_box_target.tolist(),

        "world_box_pose": {
            "pos": box_pos.tolist(),
            "quat_wxyz": box_quat.tolist(),
        },
        "world_target_pose": {
            "body": target_body,
            "pos": target_pos.tolist(),
            "quat_wxyz": target_quat.tolist(),
        },
        "box_to_target_pose": {
            "body": target_body,
            "pos": rel_target_pos.tolist(),
            "quat_wxyz": rel_target_quat.tolist(),
        },

        "debug_world_pose": {
            "body": debug_body,
            "pos": debug_pos.tolist(),
            "quat_wxyz": debug_quat.tolist(),
        },
        "debug_box_to_body_pose": {
            "body": debug_body,
            "pos": rel_debug_pos.tolist(),
            "quat_wxyz": rel_debug_quat.tolist(),
        },

        "franka_qpos": franka_qpos,
        "franka_ctrl": franka_ctrl,

        # 后续手指不走 IK，直接用这里保存的 7 个主动控制量
        "o7_active_qpos": o7_active_qpos,
        "o7_active_ctrl": o7_active_ctrl,

        "full_qpos": data.qpos.tolist(),
        "full_ctrl": data.ctrl.tolist(),
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(template, f, indent=2)

    print("\n========== EXTRACT FR3+O7 GRASP TEMPLATE V9 ==========")
    print("model :", model_path)
    print("record:", record_path)
    print("out   :", out_path)
    print()
    print("box body              :", args.box_body)
    print("box world pos         :", box_pos)
    print("box world quat        :", box_quat)
    print()
    print("IK target body        :", target_body)
    print("target world pos      :", target_pos)
    print("target world quat     :", target_quat)
    print("T_box_target pos      :", rel_target_pos)
    print("T_box_target quat     :", rel_target_quat)
    print()
    print("debug body            :", debug_body)
    print("debug world pos       :", debug_pos)
    print("T_box_debug pos       :", rel_debug_pos)
    print("T_box_debug quat      :", rel_debug_quat)
    print()
    print("Franka ctrl:")
    for k, v in franka_ctrl.items():
        print(f"  {k:12s}: {v:+.6f}")

    print()
    print("O7 active ctrl:")
    for k, v in o7_active_ctrl.items():
        print(f"  {k:18s}: {v:+.6f}")

    print("=======================================================\n")


if __name__ == "__main__":
    main()
