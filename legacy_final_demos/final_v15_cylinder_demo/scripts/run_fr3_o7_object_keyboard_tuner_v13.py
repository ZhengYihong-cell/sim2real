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
RECORD_DIR = PROJECT / "records"
RECORD_DIR.mkdir(exist_ok=True)

DEFAULT_MODEL = PROJECT / "models/fr3_o7/fr3_o7_actuated_scene_v13_cylinder.xml"


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

MIMIC = {
    "thumb_mcp": ("thumb_cmc_pitch", 1.3898, 0.0),
    "thumb_ip": ("thumb_cmc_pitch", 1.5085, 0.0),

    "index_pip": ("index_mcp_pitch", 1.3462, 0.0),
    "index_dip": ("index_mcp_pitch", 0.4615, 0.0),

    "middle_pip": ("middle_mcp_pitch", 1.3462, 0.0),
    "middle_dip": ("middle_mcp_pitch", 0.4615, 0.0),

    "ring_pip": ("ring_mcp_pitch", 1.3462, 0.0),
    "ring_dip": ("ring_mcp_pitch", 0.4615, 0.0),

    "pinky_pip": ("pinky_mcp_pitch", 1.3462, 0.0),
    "pinky_dip": ("pinky_mcp_pitch", 0.4615, 0.0),
}


# 用 box 成功模板附近的机械臂姿态作为圆柱调姿起点
START_ARM = {
    "fr3_joint1": 0.36,
    "fr3_joint2": -0.04,
    "fr3_joint3": -0.32,
    "fr3_joint4": -2.26,
    "fr3_joint5": 0.08,
    "fr3_joint6": 2.55,
    "fr3_joint7": -0.13,
}


# approach 手型：thumb roll/yaw 到位，thumb pitch 和四指打开
APPROACH_HAND = {
    "thumb_cmc_roll": 0.27,
    "thumb_cmc_yaw": 1.42,
    "thumb_cmc_pitch": 0.00,
    "index_mcp_pitch": 0.00,
    "middle_mcp_pitch": 0.00,
    "ring_mcp_pitch": 0.00,
    "pinky_mcp_pitch": 0.00,
}


# 初始抓握预设，后面可在 MuJoCo Control 面板继续细调
GRASP_PRESET = {
    "thumb_cmc_roll": 0.27,
    "thumb_cmc_yaw": 1.42,
    "thumb_cmc_pitch": 0.31,
    "index_mcp_pitch": 0.23,
    "middle_mcp_pitch": 0.25,
    "ring_mcp_pitch": 0.25,
    "pinky_mcp_pitch": 0.22,
}


selected_kind = "arm"
selected_arm = 0
selected_hand = 0
step = 0.02
paused = False


def resolve_path(p):
    p = Path(p).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


def name(model, objtype, idx):
    return mujoco.mj_id2name(model, objtype, idx) or ""


def parse_tokens(s):
    return [x.strip().lower() for x in s.replace(",", " ").split() if x.strip()]


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


def set_qpos_joint(model, data, joint_name, value):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return
    data.qpos[int(model.jnt_qposadr[jid])] = float(value)


def get_qpos_joint(model, data, joint_name):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return None
    return float(data.qpos[int(model.jnt_qposadr[jid])])


def set_ctrl_joint(model, data, joint_name, value):
    aid = actuator_id(model, joint_name + "_pos")
    if model.actuator_ctrllimited[aid]:
        lo, hi = model.actuator_ctrlrange[aid]
        value = float(np.clip(value, lo, hi))
    data.ctrl[aid] = float(value)


def get_ctrl_joint(model, data, joint_name):
    aid = actuator_id(model, joint_name + "_pos")
    return float(data.ctrl[aid])


def set_arm_qpos_and_ctrl(model, data, qdict):
    for j, v in qdict.items():
        set_qpos_joint(model, data, j, v)
        set_ctrl_joint(model, data, j, v)


def set_hand_active_qpos_and_ctrl(model, data, active_dict):
    for j in O7_ACTIVE_JOINTS:
        v = float(active_dict.get(j, 0.0))
        set_qpos_joint(model, data, j, v)
        set_ctrl_joint(model, data, j, v)

    # 同步 mimic qpos，避免启动瞬间手指像面条一样被 equality 拉动
    for mimic_joint, (parent, mul, offset) in MIMIC.items():
        parent_v = float(active_dict.get(parent, 0.0))
        set_qpos_joint(model, data, mimic_joint, offset + mul * parent_v)


def set_hand_ctrl(model, data, active_dict):
    for j in O7_ACTIVE_JOINTS:
        if j in active_dict:
            set_ctrl_joint(model, data, j, active_dict[j])


def open_hand_keep_thumb_yaw_roll(model, data):
    d = dict(APPROACH_HAND)
    set_hand_ctrl(model, data, d)


def full_grasp_preset(model, data):
    set_hand_ctrl(model, data, GRASP_PRESET)


def body_pose(model, data, body_name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        return None, None
    return data.xpos[bid].copy(), data.xquat[bid].copy()


def contact_counts(model, data, object_tokens, support_tokens):
    hand_object = 0
    fr3_object = 0
    object_support = 0
    hand_support = 0

    hand_tokens = ["thumb", "index", "middle", "ring", "pinky", "metacarpals"]
    support_detail = {tok: 0 for tok in support_tokens}

    contacts = []

    for i in range(data.ncon):
        c = data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)

        g1_name = name(model, mujoco.mjtObj.mjOBJ_GEOM, g1)
        g2_name = name(model, mujoco.mjtObj.mjOBJ_GEOM, g2)

        b1_name = name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g1]))
        b2_name = name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[g2]))

        text = " ".join([g1_name, g2_name, b1_name, b2_name]).lower()

        has_object = any(tok in text for tok in object_tokens)
        has_hand = any(tok in text for tok in hand_tokens)
        has_fr3 = "fr3_" in text
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

        contacts.append({
            "geom1": g1_name,
            "geom2": g2_name,
            "body1": b1_name,
            "body2": b2_name,
            "dist": float(c.dist),
        })

    return {
        "ncon": int(data.ncon),
        "hand_object": hand_object,
        "fr3_object": fr3_object,
        "object_support": object_support,
        "hand_support": hand_support,
        "support_detail": support_detail,
        "contacts": contacts,
    }


def initialize(model, data):
    # 不清空 object freejoint，保留 XML 中圆柱/垫块位置
    set_arm_qpos_and_ctrl(model, data, START_ARM)
    set_hand_active_qpos_and_ctrl(model, data, APPROACH_HAND)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)


def nudge_selected(model, data, delta):
    global selected_kind, selected_arm, selected_hand

    if selected_kind == "arm":
        j = FRANKA_JOINTS[selected_arm]
    else:
        j = O7_ACTIVE_JOINTS[selected_hand]

    old = get_ctrl_joint(model, data, j)
    set_ctrl_joint(model, data, j, old + delta)
    new = get_ctrl_joint(model, data, j)

    print(f"[NUDGE] {j}: {old:+.4f} -> {new:+.4f}")


def print_status(model, data, object_body, object_tokens, support_tokens):
    if selected_kind == "arm":
        selected_joint = FRANKA_JOINTS[selected_arm]
    else:
        selected_joint = O7_ACTIVE_JOINTS[selected_hand]

    obj_pos, obj_quat = body_pose(model, data, object_body)
    fr3_pos, fr3_quat = body_pose(model, data, "fr3_link7")

    fingertip_bodies = [
        "thumb_distal",
        "index_distal",
        "middle_distal",
        "ring_distal",
        "pinky_distal",
    ]

    counts = contact_counts(model, data, object_tokens, support_tokens)

    print("\n========== V13 STATUS ==========")
    print("selected kind :", selected_kind)
    print("selected joint:", selected_joint)
    print("selected ctrl :", get_ctrl_joint(model, data, selected_joint))
    print("keyboard step :", step)
    print()
    print("object_body   :", object_body)
    print("object_pos    :", obj_pos)
    print("object_quat   :", obj_quat)
    print("fr3_link7_pos :", fr3_pos)
    if obj_pos is not None and fr3_pos is not None:
        print("fr3_link7 - object:", fr3_pos - obj_pos)

    print("\nARM ctrl:")
    for j in FRANKA_JOINTS:
        print(f"  {j:12s}: {get_ctrl_joint(model, data, j):+.5f}")

    print("\nO7 active ctrl:")
    for j in O7_ACTIVE_JOINTS:
        print(f"  {j:18s}: {get_ctrl_joint(model, data, j):+.5f}")

    print("\nFINGERTIPS:")
    for b in fingertip_bodies:
        p, _ = body_pose(model, data, b)
        print(f"  {b:14s}: {p}")

    print("\nCONTACT:")
    print("  ncon          :", counts["ncon"])
    print("  hand_object   :", counts["hand_object"])
    print("  fr3_object    :", counts["fr3_object"])
    print("  object_support:", counts["object_support"])
    print("  hand_support  :", counts["hand_support"])
    print("  support_detail:", counts["support_detail"])
    print("================================\n")


def save_record(model, data, model_path, object_body, object_token_string, support_token_string):
    stamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    out = RECORD_DIR / f"fr3_o7_object_keyboard_pose_v13_{stamp}.json"

    object_tokens = parse_tokens(object_token_string)
    support_tokens = parse_tokens(support_token_string)

    body_names = [
        object_body,
        "fr3_link7",
        "thumb_metacarpals_base1",
        "thumb_distal",
        "index_distal",
        "middle_distal",
        "ring_distal",
        "pinky_distal",
    ]

    body_poses = {}
    for b in body_names:
        pos, quat = body_pose(model, data, b)
        if pos is not None:
            body_poses[b] = {
                "pos": pos.tolist(),
                "quat_wxyz": quat.tolist(),
            }

    counts = contact_counts(model, data, object_tokens, support_tokens)

    blob = {
        "format": "fr3_o7_object_keyboard_pose_v13",
        "time": stamp,
        "model_path": str(model_path),
        "object_body": object_body,
        "object_token": object_token_string,
        "support_tokens": support_token_string,

        "qpos": data.qpos.tolist(),
        "qvel": data.qvel.tolist(),
        "ctrl": data.ctrl.tolist(),

        "franka_ctrl": {
            j: get_ctrl_joint(model, data, j)
            for j in FRANKA_JOINTS
        },
        "o7_active_ctrl": {
            j: get_ctrl_joint(model, data, j)
            for j in O7_ACTIVE_JOINTS
        },
        "body_poses": body_poses,
        "contact_summary": {
            k: v for k, v in counts.items()
            if k != "contacts"
        },
        "contacts": counts["contacts"],
    }

    with open(out, "w") as f:
        json.dump(blob, f, indent=2)

    print("\n[SAVED]", out)
    print_status(model, data, object_body, object_tokens, support_tokens)


def main():
    global selected_kind, selected_arm, selected_hand, step, paused

    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default=str(DEFAULT_MODEL))
    ap.add_argument("--object-body", default="grasp_cylinder")
    ap.add_argument("--object-token", default="grasp_cylinder")
    ap.add_argument("--support-tokens", default="pedestal table")
    args = ap.parse_args()

    model_path = resolve_path(args.model)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    object_tokens = parse_tokens(args.object_token)
    support_tokens = parse_tokens(args.support_tokens)

    initialize(model, data)

    def key_callback(keycode):
        global selected_kind, selected_arm, selected_hand, step, paused

        ch = chr(keycode).lower() if 0 <= keycode < 256 else ""

        if ch and ch in "1234567":
            selected_kind = "arm"
            selected_arm = int(ch) - 1
            print("[SELECT ARM]", FRANKA_JOINTS[selected_arm])

        elif ch and ch in "qwertyu":
            idx = "qwertyu".index(ch)
            if idx < len(O7_ACTIVE_JOINTS):
                selected_kind = "hand"
                selected_hand = idx
                print("[SELECT HAND]", O7_ACTIVE_JOINTS[selected_hand])

        elif ch == "[":
            nudge_selected(model, data, -step)

        elif ch == "]":
            nudge_selected(model, data, +step)

        elif ch == "z":
            step = max(0.001, step / 2.0)
            print("[STEP]", step)

        elif ch == "x":
            step = min(0.2, step * 2.0)
            print("[STEP]", step)

        elif ch == "o":
            open_hand_keep_thumb_yaw_roll(model, data)
            print("[HAND] approach hand: thumb roll/yaw kept, pitch and fingers open")

        elif ch == "g":
            full_grasp_preset(model, data)
            print("[HAND] full grasp preset")

        elif ch == "p":
            print_status(model, data, args.object_body, object_tokens, support_tokens)

        elif ch == "s":
            save_record(
                model,
                data,
                model_path,
                args.object_body,
                args.object_token,
                args.support_tokens,
            )

        elif ch == " ":
            paused = not paused
            print("[PAUSED]", paused)

    print("\n========== FR3 + O7 OBJECT KEYBOARD TUNER V13 ==========")
    print("model         :", model_path)
    print("object_body   :", args.object_body)
    print("object_token  :", args.object_token)
    print("support_tokens:", args.support_tokens)
    print("nu / neq      :", model.nu, model.neq)
    print()
    print("用法：")
    print("  左侧 Control 面板：细调所有 actuator")
    print("  1~7            ：选择 Franka 7 轴")
    print("  q/w/e/r/t/y/u  ：选择 O7 7 个主动关节")
    print("  [ / ]          ：当前关节 -/+ step")
    print("  z / x          ：缩小 / 放大 step")
    print("  o              ：approach 手型，thumb roll/yaw 保持，pitch 和四指打开")
    print("  g              ：应用当前抓握预设")
    print("  p              ：打印状态和接触")
    print("  s              ：保存当前姿态 JSON")
    print("  SPACE          ：暂停 / 继续")
    print()
    print("建议：")
    print("  保存圆柱模板时，尽量让圆柱仍在蓝色垫块上，手已经夹住圆柱；")
    print("  目标状态最好 hand_object >= 4，fr3_object = 0，hand_support = 0。")
    print("========================================================\n")

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.cam.lookat[:] = np.array([0.52, 0.02, 0.25])
        viewer.cam.distance = 1.1
        viewer.cam.azimuth = 125
        viewer.cam.elevation = -18
        viewer.opt.geomgroup[3] = 0
        viewer.opt.geomgroup[4] = 1

        while viewer.is_running():
            if not paused:
                mujoco.mj_step(model, data)

            viewer.sync()
            time.sleep(model.opt.timestep)


if __name__ == "__main__":
    main()
