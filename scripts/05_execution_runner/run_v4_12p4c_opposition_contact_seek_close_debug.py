#!/usr/bin/env python3
"""
文件名：
    run_v4_12p4c_opposition_contact_seek_close_debug.py

脚本类别：
    debug / runner / viewer / contact-driven-close

用途：
    本脚本用于 V4.12P4C 阶段，实现“保持对握结构的接触寻优闭合”。
    它解决 P4/P4B 中出现的问题：
        1. 拇指不是从自然对握姿态闭合；
        2. 四指先接触后继续挤压，导致物体被推出抓握范围；
        3. 闭合控制没有根据 object displacement / contact event 切换策略。

核心流程：
    1. 读取 P3 选出的 q_pre / q_grasp / q_lift。
    2. 机械臂从当前姿态移动到 q_pre。
    3. 在 q_pre 处进行 side-grasp thumb opposition preshape：
       thumb roll / yaw 到对握姿态，thumb pitch 保持打开，四指保持打开。
    4. 机械臂移动到 q_grasp，手保持 side-open 对握预姿态。
    5. Phase A：four-finger light contact seek
       四指低速闭合，任一四指 group 接触 object 后冻结该 group。
       如果物体被明显推出，停止四指继续内压，切到 thumb compensation。
    6. Phase B：thumb opposition compensation
       thumb roll/yaw 保持对握，thumb pitch 低速补上对抗接触。
       目标是形成 thumb + 至少一个非拇指 group 的对握接触。
    7. Phase C：micro squeeze
       只对尚未接触的 finger group 做小步补接触；
       已接触 group 基本保持，防止继续把物体挤走。
    8. hold。
    9. 满足接触门控后执行 q_grasp -> q_lift。

输入：
    1. --model
       MuJoCo XML，例如 models/fr3_o7/fr3_o7_can52_upright_tabletop_v47b_xminus008_debug.xml
    2. --candidate
       candidate JSON，用于读取 object.body 和可能的手型先验。
    3. --p3-json 或 --plan-json
       P3 输出 JSON 或 best plan JSON，必须包含 q_pre / q_grasp / q_lift。
    4. --object-body
       物体 body 名称，例如 grasp_can。

输出：
    1. --out 指定的 JSON 结果文件。
    2. 终端日志，包括每个阶段的接触 group、物体位移、冻结状态、最终 lift 结果。
    3. 可选 MuJoCo viewer 可视化。

当前流程位置：
    candidate / dataset prior
        -> P2 Pinocchio IK 多 seed
        -> P3 MuJoCo collision + joint margin precheck
        -> P4C opposition-preserving contact-seeking close
        -> 后续 P5：固化为 selector/runner 的统一执行策略

本脚本不负责：
    1. 不重新求 IK。
    2. 不重新生成候选。
    3. 不做全局碰撞规划。
    4. 不把数据集 hand ctrl 当作最终闭合命令。
    5. 不通过固定倍率试错抓握。
"""

from pathlib import Path
import argparse
import json
import time
import numpy as np
import mujoco

try:
    import mujoco.viewer
except Exception:
    mujoco.viewer = None


PROJECT = Path.home() / "Projects/o7_mujoco_sim"

ARM_JOINTS = [
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
]

THUMB_POSTURE_JOINTS = [
    "thumb_cmc_roll",
    "thumb_cmc_yaw",
]

THUMB_CLOSE_JOINTS = [
    "thumb_cmc_pitch",
]

FOUR_FINGER_GROUPS = {
    "index": ["index_mcp_pitch"],
    "middle": ["middle_mcp_pitch"],
    "ring": ["ring_mcp_pitch"],
    "pinky": ["pinky_mcp_pitch"],
}

FINGER_GROUPS = {
    "thumb": THUMB_CLOSE_JOINTS,
    **FOUR_FINGER_GROUPS,
}

ACTIVE_HAND_JOINTS = [
    "thumb_cmc_roll",
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]

HAND_TOKENS = [
    "thumb",
    "index",
    "middle",
    "ring",
    "pinky",
    "hand",
    "palm",
    "metacarpals",
]

SUPPORT_TOKENS = [
    "object_pedestal",
    "pedestal",
    "support",
    "table",
]

FR3_TOKENS = [
    "fr3_link",
    "fr3_",
]


def resolve_path(p):
    p = Path(str(p)).expanduser()
    if not p.is_absolute():
        p = PROJECT / p
    return p


def load_json(p):
    with open(resolve_path(p), "r") as f:
        return json.load(f)


def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


def save_json(p, obj):
    p = resolve_path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(to_jsonable(obj), f, indent=2)


def mj_name(model, objtype, idx):
    return mujoco.mj_id2name(model, objtype, int(idx)) or ""


def mj_id(model, objtype, name):
    if name is None:
        return -1
    return mujoco.mj_name2id(model, objtype, str(name))


def joint_id(model, name):
    return mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def actuator_id(model, name):
    return mj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def body_id(model, name):
    return mj_id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def geom_name(model, gid):
    n = mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
    return n if n else f"geom_{gid}"


def body_name(model, bid):
    n = mj_name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
    return n if n else f"body_{bid}"


def body_name_of_geom(model, gid):
    return body_name(model, int(model.geom_bodyid[int(gid)]))


def body_is_descendant(model, bid, root_bid):
    if bid < 0 or root_bid < 0:
        return False

    cur = int(bid)
    while cur > 0:
        if cur == int(root_bid):
            return True
        cur = int(model.body_parentid[cur])

    return cur == int(root_bid)


def collect_geom_sets(model, object_body):
    object_bid = body_id(model, object_body)
    if object_bid < 0:
        raise RuntimeError(f"cannot find object body: {object_body}")

    object_geoms = []
    support_geoms = []
    hand_geoms = []
    fr3_geoms = []

    for gid in range(model.ngeom):
        gname = geom_name(model, gid)
        bname = body_name_of_geom(model, gid)
        text = f"{gname} {bname}".lower()
        bid = int(model.geom_bodyid[gid])

        if body_is_descendant(model, bid, object_bid):
            object_geoms.append(gid)
            continue

        if any(tok in text for tok in SUPPORT_TOKENS):
            support_geoms.append(gid)
            continue

        if any(tok in text for tok in HAND_TOKENS):
            hand_geoms.append(gid)
            continue

        if any(tok in text for tok in FR3_TOKENS):
            fr3_geoms.append(gid)
            continue

    if not object_geoms:
        raise RuntimeError(f"no object geoms found under body {object_body}")
    if not hand_geoms:
        print("[WARN] no hand geoms found by token matching")
    if not support_geoms:
        print("[WARN] no support geoms found by token matching")

    return {
        "object_geoms": object_geoms,
        "support_geoms": support_geoms,
        "hand_geoms": hand_geoms,
        "fr3_geoms": fr3_geoms,
    }


def find_actuator_for_joint(model, joint_name):
    candidates = [
        joint_name,
        f"{joint_name}_pos",
        f"{joint_name}_ctrl",
        f"{joint_name}_act",
        f"{joint_name}_motor",
    ]

    for n in candidates:
        aid = actuator_id(model, n)
        if aid >= 0:
            return aid, n

    return -1, ""


def actuator_ctrl_range(model, joint_name):
    aid, act_name = find_actuator_for_joint(model, joint_name)
    if aid < 0:
        return None, act_name

    limited = bool(model.actuator_ctrllimited[aid])
    if limited:
        lo, hi = model.actuator_ctrlrange[aid]
        return (float(lo), float(hi)), act_name

    jid = joint_id(model, joint_name)
    if jid >= 0 and bool(model.jnt_limited[jid]):
        lo, hi = model.jnt_range[jid]
        return (float(lo), float(hi)), act_name

    return (-3.0, 3.0), act_name


def clamp_ctrl(model, joint_name, value):
    cr, _ = actuator_ctrl_range(model, joint_name)
    if cr is None:
        return float(value)
    lo, hi = cr
    return float(np.clip(float(value), lo, hi))


def set_actuator_ctrl(model, data, joint_name, value):
    aid, act_name = find_actuator_for_joint(model, joint_name)
    if aid < 0:
        return False, ""

    data.ctrl[aid] = clamp_ctrl(model, joint_name, value)
    return True, act_name


def set_joint_qpos(model, data, joint_name, value):
    jid = joint_id(model, joint_name)
    if jid < 0:
        return False

    jtype = int(model.jnt_type[jid])
    if jtype not in [
        int(mujoco.mjtJoint.mjJNT_HINGE),
        int(mujoco.mjtJoint.mjJNT_SLIDE),
    ]:
        return False

    qadr = int(model.jnt_qposadr[jid])
    val = float(value)

    if bool(model.jnt_limited[jid]):
        lo, hi = model.jnt_range[jid]
        val = float(np.clip(val, lo, hi))

    data.qpos[qadr] = val
    return True


def get_joint_qpos(model, data, joint_name):
    jid = joint_id(model, joint_name)
    if jid < 0:
        return None

    jtype = int(model.jnt_type[jid])
    if jtype not in [
        int(mujoco.mjtJoint.mjJNT_HINGE),
        int(mujoco.mjtJoint.mjJNT_SLIDE),
    ]:
        return None

    qadr = int(model.jnt_qposadr[jid])
    return float(data.qpos[qadr])


def apply_arm_q(model, data, qdict, also_ctrl=True):
    missing = []
    for j, v in (qdict or {}).items():
        if j not in ARM_JOINTS:
            continue

        ok_q = set_joint_qpos(model, data, j, v)
        ok_c = True
        if also_ctrl:
            ok_c, _ = set_actuator_ctrl(model, data, j, v)

        if not ok_q and not ok_c:
            missing.append(j)

    return missing


def apply_hand_ctrl(model, data, ctrl, direct_qpos=False):
    applied = {}
    missing = {}

    for j, v in (ctrl or {}).items():
        ok_c, act_name = set_actuator_ctrl(model, data, j, v)
        ok_q = True

        if direct_qpos:
            ok_q = set_joint_qpos(model, data, j, v)

        if ok_c:
            applied[j] = act_name
        elif ok_q:
            applied[j] = "qpos_direct"
        else:
            missing[j] = v

    return applied, missing


def read_arm_qpos(model, data):
    out = {}
    for j in ARM_JOINTS:
        v = get_joint_qpos(model, data, j)
        if v is not None:
            out[j] = v
    return out


def read_hand_qpos(model, data):
    out = {}
    for j in ACTIVE_HAND_JOINTS:
        v = get_joint_qpos(model, data, j)
        if v is not None:
            out[j] = v
    return out


def read_hand_ctrl(model, data):
    out = {}
    for j in ACTIVE_HAND_JOINTS:
        aid, _ = find_actuator_for_joint(model, j)
        if aid >= 0:
            out[j] = float(data.ctrl[aid])
    return out


def interp_dict(a, b, s):
    keys = sorted(set((a or {}).keys()) | set((b or {}).keys()))
    out = {}
    for k in keys:
        av = float((a or {}).get(k, (b or {}).get(k, 0.0)))
        bv = float((b or {}).get(k, (a or {}).get(k, 0.0)))
        out[k] = (1.0 - s) * av + s * bv
    return out


def object_pose(model, data, object_body):
    bid = body_id(model, object_body)
    if bid < 0:
        return None

    return {
        "pos": data.xpos[bid].copy(),
        "xmat": data.xmat[bid].reshape(3, 3).copy(),
        "quat": data.xquat[bid].copy(),
    }


def object_disp(model, data, object_body, ref_pos):
    pose = object_pose(model, data, object_body)
    if pose is None:
        return 999.0
    return float(np.linalg.norm(pose["pos"] - np.asarray(ref_pos, dtype=float)))


def group_of_hand_geom(model, gid):
    text = f"{geom_name(model, gid)} {body_name_of_geom(model, gid)}".lower()

    if "thumb" in text:
        return "thumb"
    if "index" in text:
        return "index"
    if "middle" in text:
        return "middle"
    if "ring" in text:
        return "ring"
    if "pinky" in text:
        return "pinky"
    if "palm" in text or "hand" in text:
        return "palm"

    return "unknown"


def contact_summary(model, data, geom_sets):
    object_set = set(geom_sets["object_geoms"])
    support_set = set(geom_sets["support_geoms"])
    hand_set = set(geom_sets["hand_geoms"])
    fr3_set = set(geom_sets["fr3_geoms"])

    out = {
        "ncon": int(data.ncon),
        "hand_object": 0,
        "hand_support": 0,
        "fr3_object": 0,
        "fr3_support": 0,
        "object_support": 0,
        "object_groups": {},
        "support_groups": {},
        "pairs": [],
    }

    def has_pair(a, b, A, B):
        return (a in A and b in B) or (a in B and b in A)

    for ci in range(data.ncon):
        c = data.contact[ci]
        g1 = int(c.geom1)
        g2 = int(c.geom2)

        cls = "other"
        group = "unknown"

        if has_pair(g1, g2, hand_set, object_set):
            cls = "hand_object"
            out["hand_object"] += 1
            hg = g1 if g1 in hand_set else g2
            group = group_of_hand_geom(model, hg)
            out["object_groups"][group] = out["object_groups"].get(group, 0) + 1

        elif has_pair(g1, g2, hand_set, support_set):
            cls = "hand_support"
            out["hand_support"] += 1
            hg = g1 if g1 in hand_set else g2
            group = group_of_hand_geom(model, hg)
            out["support_groups"][group] = out["support_groups"].get(group, 0) + 1

        elif has_pair(g1, g2, fr3_set, object_set):
            cls = "fr3_object"
            out["fr3_object"] += 1

        elif has_pair(g1, g2, fr3_set, support_set):
            cls = "fr3_support"
            out["fr3_support"] += 1

        elif has_pair(g1, g2, object_set, support_set):
            cls = "object_support"
            out["object_support"] += 1

        out["pairs"].append({
            "class": cls,
            "group": group,
            "geom1": geom_name(model, g1),
            "body1": body_name_of_geom(model, g1),
            "geom2": geom_name(model, g2),
            "body2": body_name_of_geom(model, g2),
            "dist": float(c.dist),
            "pos": data.contact[ci].pos.copy(),
        })

    return out


def extract_candidate_ctrl(candidate):
    hand = candidate.get("hand", {}) or {}

    keys = [
        "o7_active_ctrl",
        "active_ctrl",
        "ctrl",
        "target_ctrl",
        "qpos",
        "target_qpos",
    ]

    for key in keys:
        val = hand.get(key, None)
        if isinstance(val, dict):
            picked = {}
            for j in ACTIVE_HAND_JOINTS:
                if j in val:
                    picked[j] = float(val[j])
            if picked:
                return picked, f"hand.{key}"

    for key in keys:
        val = candidate.get(key, None)
        if isinstance(val, dict):
            picked = {}
            for j in ACTIVE_HAND_JOINTS:
                if j in val:
                    picked[j] = float(val[j])
            if picked:
                return picked, key

    return {}, "NOT_FOUND"


def selected_plan_from_json(src, which):
    if all(k in src for k in ["q_pre", "q_grasp", "q_lift"]):
        return src, "plan_json"

    item = src.get(which, None)
    if item is None:
        raise RuntimeError(f"{which} is None in source json")

    if not all(k in item for k in ["q_pre", "q_grasp", "q_lift"]):
        raise RuntimeError(f"{which} does not contain q_pre/q_grasp/q_lift")

    return item, which


def make_zero_open_ctrl(model):
    ctrl = {}
    for j in ACTIVE_HAND_JOINTS:
        ctrl[j] = 0.0
        ctrl[j] = clamp_ctrl(model, j, ctrl[j])
    return ctrl


def make_side_open_ctrl(model, args, candidate_ctrl):
    """
    侧握对握预张开，不是最终闭合。
    thumb roll/yaw 负责把拇指摆到对握侧。
    thumb pitch 保持打开。
    四指保持打开。
    """
    ctrl = make_zero_open_ctrl(model)

    if args.use_candidate_thumb_preshape and candidate_ctrl:
        ctrl["thumb_cmc_roll"] = candidate_ctrl.get("thumb_cmc_roll", args.thumb_roll_preshape)
        ctrl["thumb_cmc_yaw"] = candidate_ctrl.get("thumb_cmc_yaw", args.thumb_yaw_preshape)
    else:
        ctrl["thumb_cmc_roll"] = args.thumb_roll_preshape
        ctrl["thumb_cmc_yaw"] = args.thumb_yaw_preshape

    ctrl["thumb_cmc_pitch"] = args.thumb_pitch_open

    for j in ACTIVE_HAND_JOINTS:
        ctrl[j] = clamp_ctrl(model, j, ctrl[j])

    return ctrl


def ctrl_bound_in_close_direction(model, joint_name, direction=1.0):
    cr, _ = actuator_ctrl_range(model, joint_name)
    if cr is None:
        return 2.0 * float(direction)

    lo, hi = cr
    return hi if direction >= 0.0 else lo


def increment_joint_ctrl(model, ctrl, joint_name, delta, direction=1.0):
    old = float(ctrl.get(joint_name, 0.0))
    target = ctrl_bound_in_close_direction(model, joint_name, direction)
    new = old + float(direction) * float(delta)

    if direction >= 0.0:
        new = min(new, target)
    else:
        new = max(new, target)

    ctrl[joint_name] = clamp_ctrl(model, joint_name, new)
    return ctrl[joint_name] - old


def step_with_hold(model, data, q_arm, hand_ctrl, args, nstep=1):
    for _ in range(max(1, int(nstep))):
        if args.arm_hold_mode == "hard":
            apply_arm_q(model, data, q_arm, also_ctrl=True)
        else:
            for j, v in q_arm.items():
                if j in ARM_JOINTS:
                    set_actuator_ctrl(model, data, j, v)

        apply_hand_ctrl(model, data, hand_ctrl, direct_qpos=args.direct_hand_qpos)
        mujoco.mj_step(model, data)

        if args.arm_hold_mode == "hard":
            apply_arm_q(model, data, q_arm, also_ctrl=True)
            mujoco.mj_forward(model, data)


def sync_viewer(viewer, sleep_s):
    if viewer is not None:
        viewer.sync()
        if sleep_s > 0.0:
            time.sleep(sleep_s)


def move_arm_segment(viewer, model, data, q_from, q_to, hand_ctrl, steps, label, object_body, args, logs):
    print(f"\n[PHASE] {label}, steps={steps}")

    for i in range(steps + 1):
        s = i / float(max(steps, 1))
        q = interp_dict(q_from, q_to, s)

        apply_arm_q(model, data, q, also_ctrl=True)
        apply_hand_ctrl(model, data, hand_ctrl, direct_qpos=args.direct_hand_qpos)
        mujoco.mj_forward(model, data)

        if i in [0, steps // 2, steps]:
            pose = object_pose(model, data, object_body)
            row = {
                "phase": label,
                "step": i,
                "s": s,
                "object_pos": pose["pos"] if pose else None,
                "arm_qpos": read_arm_qpos(model, data),
                "hand_qpos": read_hand_qpos(model, data),
                "hand_ctrl": read_hand_ctrl(model, data),
            }
            logs.append(row)
            print(
                f"  [{label}] {i}/{steps} "
                f"object_pos={pose['pos'] if pose else None} "
                f"hand_qpos={read_hand_qpos(model, data)}"
            )

        sync_viewer(viewer, args.frame_sleep)


def update_contact_states(cs, object_state, support_state, first_object, first_support, ctrl, model, data, t, step):
    events = []

    for group in ["thumb", "index", "middle", "ring", "pinky"]:
        if cs["object_groups"].get(group, 0) > 0:
            if not object_state[group]:
                first_object[group] = {
                    "time": t,
                    "step": step,
                    "ctrl": dict(ctrl),
                    "hand_qpos": read_hand_qpos(model, data),
                }
                events.append({
                    "event": "first_object_contact",
                    "group": group,
                    "time": t,
                    "step": step,
                })
                print(f"[CONTACT object] group={group} t={t:.4f}")

            object_state[group] = True

        if cs["support_groups"].get(group, 0) > 0:
            if not support_state[group]:
                first_support[group] = {
                    "time": t,
                    "step": step,
                    "ctrl": dict(ctrl),
                    "hand_qpos": read_hand_qpos(model, data),
                }
                events.append({
                    "event": "first_support_contact",
                    "group": group,
                    "time": t,
                    "step": step,
                })
                print(f"[CONTACT support] group={group} t={t:.4f}")

            support_state[group] = True

    return events


def non_thumb_contact_count(object_state):
    return sum(1 for g in ["index", "middle", "ring", "pinky"] if object_state.get(g, False))


def total_object_group_count(object_state):
    return sum(1 for g in ["thumb", "index", "middle", "ring", "pinky"] if object_state.get(g, False))


def contact_goal_reached(object_state, args):
    if not args.allow_no_thumb and not object_state.get("thumb", False):
        return False

    if non_thumb_contact_count(object_state) < args.min_non_thumb_groups:
        return False

    if total_object_group_count(object_state) < args.min_total_object_groups:
        return False

    return True


def should_fail_support(cs, args):
    if not args.fail_on_hand_support:
        return False, ""

    if cs["hand_support"] <= 0:
        return False, ""

    return True, f"hand_support_contact: groups={cs['support_groups']}"


def log_close_state(logs, phase, step, t, model, data, object_body, cs, ctrl, frozen, ref_pos, extra=None):
    pose = object_pose(model, data, object_body)
    disp = object_disp(model, data, object_body, ref_pos)

    row = {
        "phase": phase,
        "step": step,
        "time": t,
        "object_pos": pose["pos"] if pose else None,
        "object_disp": disp,
        "contacts": {
            "ncon": cs["ncon"],
            "hand_object": cs["hand_object"],
            "hand_support": cs["hand_support"],
            "fr3_object": cs["fr3_object"],
            "object_support": cs["object_support"],
            "object_groups": dict(cs["object_groups"]),
            "support_groups": dict(cs["support_groups"]),
        },
        "ctrl": dict(ctrl),
        "hand_qpos": read_hand_qpos(model, data),
        "frozen": dict(frozen),
    }

    if extra:
        row.update(extra)

    logs.append(row)


def phase_four_finger_light_seek(viewer, model, data, q_grasp, ctrl, object_body, geom_sets, ref_pos, state, args, logs):
    """
    Phase A:
        四指轻触。
        已接触 group 冻结。
        如果物体位移超过 soft push threshold，停止四指继续内压，切到 thumb compensation。
    """
    dt = float(model.opt.timestep)
    max_steps = int(args.finger_seek_duration / dt)
    log_every = max(1, int(args.log_dt / dt))

    result = {
        "phase": "four_finger_light_seek",
        "stop_reason": "",
        "events": [],
    }

    print("\n========== PHASE A: FOUR-FINGER LIGHT SEEK ==========")

    for step in range(max_steps + 1):
        t = step * dt

        for group, joints in FOUR_FINGER_GROUPS.items():
            if state["frozen"][group]:
                continue

            for j in joints:
                increment_joint_ctrl(
                    model,
                    ctrl,
                    j,
                    delta=args.finger_seek_speed * dt,
                    direction=1.0,
                )

        step_with_hold(model, data, q_grasp, ctrl, args, nstep=1)

        cs = contact_summary(model, data, geom_sets)
        events = update_contact_states(
            cs,
            state["object_contact"],
            state["support_contact"],
            state["first_object_contact"],
            state["first_support_contact"],
            ctrl,
            model,
            data,
            t,
            step,
        )
        result["events"].extend(events)

        for group in ["index", "middle", "ring", "pinky"]:
            if state["object_contact"].get(group, False):
                state["frozen"][group] = True

        fail_support, reason = should_fail_support(cs, args)
        if fail_support:
            result["stop_reason"] = reason
            state["hard_failure"] = reason
            print("[FAIL]", reason)
            break

        disp = object_disp(model, data, object_body, ref_pos)
        if disp > args.hard_object_push_disp:
            reason = f"object hard pushed during finger seek: disp={disp:.5f} > {args.hard_object_push_disp:.5f}"
            result["stop_reason"] = reason
            state["hard_failure"] = reason
            print("[FAIL]", reason)
            break

        if disp > args.soft_object_push_disp:
            result["stop_reason"] = f"soft_push_guard_switch_to_thumb: disp={disp:.5f}"
            print("[SWITCH]", result["stop_reason"])
            break

        if non_thumb_contact_count(state["object_contact"]) >= args.min_non_thumb_groups:
            result["stop_reason"] = "enough_non_thumb_contact"
            print("[OK] enough non-thumb contact, switch to thumb compensation")
            break

        if step % log_every == 0 or step == max_steps:
            log_close_state(
                logs,
                "four_finger_light_seek",
                step,
                t,
                model,
                data,
                object_body,
                cs,
                ctrl,
                state["frozen"],
                ref_pos,
            )
            print(
                f"[finger_seek] t={t:.3f} "
                f"disp={disp:.5f} "
                f"object_groups={cs['object_groups']} "
                f"support_groups={cs['support_groups']} "
                f"frozen={state['frozen']}"
            )

        sync_viewer(viewer, args.frame_sleep)

    if not result["stop_reason"]:
        result["stop_reason"] = "timeout"

    print("PHASE A stop_reason:", result["stop_reason"])
    print("object_contact:", state["object_contact"])
    print("frozen:", state["frozen"])
    print("=====================================================\n")

    return result


def phase_thumb_compensation(viewer, model, data, q_grasp, ctrl, object_body, geom_sets, ref_pos, state, args, logs):
    """
    Phase B:
        四指接触侧保持，不再继续大幅内压。
        thumb roll/yaw 保持对握。
        thumb pitch 低速闭合，补上对抗接触。
    """
    dt = float(model.opt.timestep)
    max_steps = int(args.thumb_comp_duration / dt)
    log_every = max(1, int(args.log_dt / dt))

    result = {
        "phase": "thumb_compensation",
        "stop_reason": "",
        "events": [],
    }

    print("\n========== PHASE B: THUMB COMPENSATION ==========")

    for step in range(max_steps + 1):
        t = step * dt

        if not state["object_contact"].get("thumb", False):
            increment_joint_ctrl(
                model,
                ctrl,
                "thumb_cmc_pitch",
                delta=args.thumb_comp_speed * dt,
                direction=1.0,
            )

        # thumb roll/yaw 保持 side-open 对握姿态，不在这里继续卷。
        ctrl["thumb_cmc_roll"] = clamp_ctrl(model, "thumb_cmc_roll", ctrl["thumb_cmc_roll"])
        ctrl["thumb_cmc_yaw"] = clamp_ctrl(model, "thumb_cmc_yaw", ctrl["thumb_cmc_yaw"])

        step_with_hold(model, data, q_grasp, ctrl, args, nstep=1)

        cs = contact_summary(model, data, geom_sets)
        events = update_contact_states(
            cs,
            state["object_contact"],
            state["support_contact"],
            state["first_object_contact"],
            state["first_support_contact"],
            ctrl,
            model,
            data,
            t,
            step,
        )
        result["events"].extend(events)

        if state["object_contact"].get("thumb", False):
            state["frozen"]["thumb"] = True

        fail_support, reason = should_fail_support(cs, args)
        if fail_support:
            result["stop_reason"] = reason
            state["hard_failure"] = reason
            print("[FAIL]", reason)
            break

        disp = object_disp(model, data, object_body, ref_pos)
        if disp > args.hard_object_push_disp:
            reason = f"object hard pushed during thumb compensation: disp={disp:.5f} > {args.hard_object_push_disp:.5f}"
            result["stop_reason"] = reason
            state["hard_failure"] = reason
            print("[FAIL]", reason)
            break

        if contact_goal_reached(state["object_contact"], args):
            result["stop_reason"] = "contact_goal_reached"
            print("[OK] thumb + non-thumb opposition contact reached")
            break

        if step % log_every == 0 or step == max_steps:
            log_close_state(
                logs,
                "thumb_compensation",
                step,
                t,
                model,
                data,
                object_body,
                cs,
                ctrl,
                state["frozen"],
                ref_pos,
            )
            print(
                f"[thumb_comp] t={t:.3f} "
                f"disp={disp:.5f} "
                f"object_groups={cs['object_groups']} "
                f"support_groups={cs['support_groups']} "
                f"thumb_pitch={ctrl.get('thumb_cmc_pitch', 0.0):.4f}"
            )

        sync_viewer(viewer, args.frame_sleep)

    if not result["stop_reason"]:
        result["stop_reason"] = "timeout"

    print("PHASE B stop_reason:", result["stop_reason"])
    print("object_contact:", state["object_contact"])
    print("frozen:", state["frozen"])
    print("=================================================\n")

    return result


def phase_micro_squeeze(viewer, model, data, q_grasp, ctrl, object_body, geom_sets, ref_pos, state, args, logs):
    """
    Phase C:
        已有 thumb + non-thumb 后，仅做小步补接触。
        已经接触的 group 保持。
        未接触的四指 group 可以小步补接触。
        如果物体位移继续增长，立即停止 micro squeeze。
    """
    dt = float(model.opt.timestep)
    max_steps = int(args.micro_squeeze_duration / dt)
    log_every = max(1, int(args.log_dt / dt))

    result = {
        "phase": "micro_squeeze",
        "stop_reason": "",
        "events": [],
    }

    print("\n========== PHASE C: MICRO SQUEEZE ==========")

    last_disp = object_disp(model, data, object_body, ref_pos)

    for step in range(max_steps + 1):
        t = step * dt

        # 只补没有接触的非拇指 group；已接触 group 不再继续大幅挤压。
        for group, joints in FOUR_FINGER_GROUPS.items():
            if state["object_contact"].get(group, False):
                continue

            for j in joints:
                increment_joint_ctrl(
                    model,
                    ctrl,
                    j,
                    delta=args.micro_finger_speed * dt,
                    direction=1.0,
                )

        # 如果 thumb 尚未接触，继续小步补；如果已经接触，不再继续加压。
        if not state["object_contact"].get("thumb", False):
            increment_joint_ctrl(
                model,
                ctrl,
                "thumb_cmc_pitch",
                delta=args.micro_thumb_speed * dt,
                direction=1.0,
            )

        step_with_hold(model, data, q_grasp, ctrl, args, nstep=1)

        cs = contact_summary(model, data, geom_sets)
        events = update_contact_states(
            cs,
            state["object_contact"],
            state["support_contact"],
            state["first_object_contact"],
            state["first_support_contact"],
            ctrl,
            model,
            data,
            t,
            step,
        )
        result["events"].extend(events)

        for group in ["thumb", "index", "middle", "ring", "pinky"]:
            if state["object_contact"].get(group, False):
                state["frozen"][group] = True

        fail_support, reason = should_fail_support(cs, args)
        if fail_support:
            result["stop_reason"] = reason
            state["hard_failure"] = reason
            print("[FAIL]", reason)
            break

        disp = object_disp(model, data, object_body, ref_pos)

        if disp > args.hard_object_push_disp:
            reason = f"object hard pushed during micro squeeze: disp={disp:.5f} > {args.hard_object_push_disp:.5f}"
            result["stop_reason"] = reason
            state["hard_failure"] = reason
            print("[FAIL]", reason)
            break

        if disp - last_disp > args.micro_push_increase_limit:
            result["stop_reason"] = (
                f"micro_push_increase_guard: disp_inc={disp - last_disp:.5f} "
                f"> {args.micro_push_increase_limit:.5f}"
            )
            print("[STOP]", result["stop_reason"])
            break

        last_disp = disp

        if contact_goal_reached(state["object_contact"], args):
            # 如果目标已达到，不一定马上停，可以多给几步补其他 group；
            # 但达到 min_contact 后超过 min_micro_hold_time 就停。
            if t >= args.min_micro_hold_time:
                result["stop_reason"] = "contact_goal_stable"
                print("[OK] contact goal stable")
                break

        if step % log_every == 0 or step == max_steps:
            log_close_state(
                logs,
                "micro_squeeze",
                step,
                t,
                model,
                data,
                object_body,
                cs,
                ctrl,
                state["frozen"],
                ref_pos,
            )
            print(
                f"[micro] t={t:.3f} "
                f"disp={disp:.5f} "
                f"object_groups={cs['object_groups']} "
                f"support_groups={cs['support_groups']} "
                f"contact_state={state['object_contact']}"
            )

        sync_viewer(viewer, args.frame_sleep)

    if not result["stop_reason"]:
        result["stop_reason"] = "timeout"

    print("PHASE C stop_reason:", result["stop_reason"])
    print("object_contact:", state["object_contact"])
    print("final ctrl:", ctrl)
    print("============================================\n")

    return result


def hold_phase(viewer, model, data, q_hold, ctrl, object_body, geom_sets, duration, phase_name, args, logs):
    dt = float(model.opt.timestep)
    steps = int(duration / dt)
    log_every = max(1, int(args.log_dt / dt))

    print(f"\n[PHASE] {phase_name}, duration={duration}")

    for step in range(steps + 1):
        t = step * dt

        step_with_hold(model, data, q_hold, ctrl, args, nstep=1)

        if step % log_every == 0 or step == steps:
            cs = contact_summary(model, data, geom_sets)
            pose = object_pose(model, data, object_body)

            logs.append({
                "phase": phase_name,
                "step": step,
                "time": t,
                "object_pos": pose["pos"] if pose else None,
                "contacts": {
                    "ncon": cs["ncon"],
                    "hand_object": cs["hand_object"],
                    "hand_support": cs["hand_support"],
                    "fr3_object": cs["fr3_object"],
                    "object_support": cs["object_support"],
                    "object_groups": dict(cs["object_groups"]),
                    "support_groups": dict(cs["support_groups"]),
                },
                "ctrl": dict(ctrl),
                "hand_qpos": read_hand_qpos(model, data),
            })

            print(
                f"[{phase_name}] t={t:.3f} "
                f"object_groups={cs['object_groups']} "
                f"hand_support={cs['hand_support']} "
                f"object_support={cs['object_support']}"
            )

        sync_viewer(viewer, args.frame_sleep)


def lift_phase(viewer, model, data, q_grasp, q_lift, ctrl, object_body, geom_sets, duration, args, logs):
    dt = float(model.opt.timestep)
    steps = int(duration / dt)
    log_every = max(1, int(args.log_dt / dt))

    pose0 = object_pose(model, data, object_body)
    z0 = float(pose0["pos"][2]) if pose0 else 0.0

    print(f"\n[PHASE] lift, duration={duration}, z0={z0:.5f}")

    for step in range(steps + 1):
        t = step * dt
        s = step / float(max(steps, 1))
        q = interp_dict(q_grasp, q_lift, s)

        step_with_hold(model, data, q, ctrl, args, nstep=1)

        if step % log_every == 0 or step == steps:
            cs = contact_summary(model, data, geom_sets)
            pose = object_pose(model, data, object_body)
            z = float(pose["pos"][2]) if pose else 0.0

            logs.append({
                "phase": "lift",
                "step": step,
                "time": t,
                "s": s,
                "object_pos": pose["pos"] if pose else None,
                "rise": z - z0,
                "contacts": {
                    "ncon": cs["ncon"],
                    "hand_object": cs["hand_object"],
                    "hand_support": cs["hand_support"],
                    "fr3_object": cs["fr3_object"],
                    "object_support": cs["object_support"],
                    "object_groups": dict(cs["object_groups"]),
                    "support_groups": dict(cs["support_groups"]),
                },
                "ctrl": dict(ctrl),
                "hand_qpos": read_hand_qpos(model, data),
            })

            print(
                f"[lift] t={t:.3f} "
                f"rise={z - z0:+.5f} "
                f"hand_object={cs['hand_object']} "
                f"object_groups={cs['object_groups']} "
                f"object_support={cs['object_support']}"
            )

        sync_viewer(viewer, args.frame_sleep)

    pose1 = object_pose(model, data, object_body)
    z1 = float(pose1["pos"][2]) if pose1 else z0
    cs1 = contact_summary(model, data, geom_sets)

    return {
        "initial_z": z0,
        "final_z": z1,
        "final_rise": z1 - z0,
        "final_contacts": cs1,
    }


def run_p4c(model, data, viewer, args, candidate, src, best, source_kind, source_json_path):
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

    geom_sets = collect_geom_sets(model, object_body)
    candidate_ctrl, candidate_ctrl_source = extract_candidate_ctrl(candidate)

    zero_open_ctrl = make_zero_open_ctrl(model)
    side_open_ctrl = make_side_open_ctrl(model, args, candidate_ctrl)

    logs = []

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    q_current = read_arm_qpos(model, data)

    print("\n========== V4.12P4C OPPOSITION CONTACT-SEEK CLOSE ==========")
    print("model                 :", resolve_path(args.model))
    print("candidate             :", resolve_path(args.candidate))
    print("source_json           :", source_json_path)
    print("source_kind           :", source_kind)
    print("object_body           :", object_body)
    print("candidate_ctrl_source :", candidate_ctrl_source)
    print("candidate_ctrl        :", candidate_ctrl)
    print("zero_open_ctrl        :", zero_open_ctrl)
    print("side_open_ctrl        :", side_open_ctrl)
    print("q seeds/status        :", best.get("pre_seed"), "->", best.get("grasp_seed"), "->", best.get("lift_seed"))
    print("arm_hold_mode         :", args.arm_hold_mode)
    print("direct_hand_qpos      :", args.direct_hand_qpos)
    print("geom_sets sizes       :", {k: len(v) for k, v in geom_sets.items()})
    print("=============================================================\n")

    apply_arm_q(model, data, q_current, also_ctrl=True)
    apply_hand_ctrl(model, data, zero_open_ctrl, direct_qpos=args.direct_hand_qpos)
    mujoco.mj_forward(model, data)
    sync_viewer(viewer, 0.3 if viewer is not None else 0.0)

    move_arm_segment(
        viewer,
        model,
        data,
        q_current,
        q_pre,
        zero_open_ctrl,
        args.move_steps,
        "move_to_pre_zero_open",
        object_body,
        args,
        logs,
    )

    # thumb opposition preshape at q_pre
    print("\n[PHASE] thumb_opposition_preshape")
    for i in range(args.thumb_preshape_steps + 1):
        s = i / float(max(args.thumb_preshape_steps, 1))
        ctrl = interp_dict(zero_open_ctrl, side_open_ctrl, s)

        step_with_hold(model, data, q_pre, ctrl, args, nstep=1)

        if i in [0, args.thumb_preshape_steps // 2, args.thumb_preshape_steps]:
            pose = object_pose(model, data, object_body)
            logs.append({
                "phase": "thumb_opposition_preshape",
                "step": i,
                "s": s,
                "object_pos": pose["pos"] if pose else None,
                "ctrl": dict(ctrl),
                "hand_qpos": read_hand_qpos(model, data),
            })
            print(
                f"  [thumb_preshape] {i}/{args.thumb_preshape_steps} "
                f"thumb_roll={ctrl['thumb_cmc_roll']:.4f} "
                f"thumb_yaw={ctrl['thumb_cmc_yaw']:.4f} "
                f"thumb_pitch={ctrl['thumb_cmc_pitch']:.4f} "
                f"hand_qpos={read_hand_qpos(model, data)}"
            )

        sync_viewer(viewer, args.frame_sleep)

    move_arm_segment(
        viewer,
        model,
        data,
        q_pre,
        q_grasp,
        side_open_ctrl,
        args.move_steps,
        "move_to_grasp_side_open",
        object_body,
        args,
        logs,
    )

    pose_close0 = object_pose(model, data, object_body)
    close_ref_pos = pose_close0["pos"].copy() if pose_close0 else np.zeros(3)

    ctrl = dict(side_open_ctrl)

    state = {
        "object_contact": {g: False for g in ["thumb", "index", "middle", "ring", "pinky"]},
        "support_contact": {g: False for g in ["thumb", "index", "middle", "ring", "pinky"]},
        "frozen": {g: False for g in ["thumb", "index", "middle", "ring", "pinky"]},
        "first_object_contact": {},
        "first_support_contact": {},
        "hard_failure": "",
    }

    # 读取一开始是否已经有接触
    mujoco.mj_forward(model, data)
    cs0 = contact_summary(model, data, geom_sets)
    update_contact_states(
        cs0,
        state["object_contact"],
        state["support_contact"],
        state["first_object_contact"],
        state["first_support_contact"],
        ctrl,
        model,
        data,
        0.0,
        0,
    )

    for g in ["thumb", "index", "middle", "ring", "pinky"]:
        if state["object_contact"][g]:
            state["frozen"][g] = True

    phase_results = []

    r1 = phase_four_finger_light_seek(
        viewer,
        model,
        data,
        q_grasp,
        ctrl,
        object_body,
        geom_sets,
        close_ref_pos,
        state,
        args,
        logs,
    )
    phase_results.append(r1)

    if not state["hard_failure"]:
        r2 = phase_thumb_compensation(
            viewer,
            model,
            data,
            q_grasp,
            ctrl,
            object_body,
            geom_sets,
            close_ref_pos,
            state,
            args,
            logs,
        )
        phase_results.append(r2)

    if not state["hard_failure"]:
        r3 = phase_micro_squeeze(
            viewer,
            model,
            data,
            q_grasp,
            ctrl,
            object_body,
            geom_sets,
            close_ref_pos,
            state,
            args,
            logs,
        )
        phase_results.append(r3)

    close_success = contact_goal_reached(state["object_contact"], args) and not state["hard_failure"]

    hold_phase(
        viewer,
        model,
        data,
        q_grasp,
        ctrl,
        object_body,
        geom_sets,
        args.hold_duration,
        "hold_after_close",
        args,
        logs,
    )

    lift_result = None

    if close_success or args.lift_even_if_fail:
        lift_result = lift_phase(
            viewer,
            model,
            data,
            q_grasp,
            q_lift,
            ctrl,
            object_body,
            geom_sets,
            args.lift_duration,
            args,
            logs,
        )
    else:
        print("[SKIP LIFT] close contact goal not reached. Use --lift-even-if-fail to visualize lift anyway.")
        cs = contact_summary(model, data, geom_sets)
        pose = object_pose(model, data, object_body)
        lift_result = {
            "initial_z": float(pose["pos"][2]) if pose else 0.0,
            "final_z": float(pose["pos"][2]) if pose else 0.0,
            "final_rise": 0.0,
            "final_contacts": cs,
            "skipped": True,
        }

    if state["hard_failure"]:
        status = "FAIL_HARD_GUARD"
        failure_reasons = [state["hard_failure"]]
    elif not close_success:
        status = "FAIL_CONTACT_GOAL"
        failure_reasons = ["thumb + non-thumb opposition contact goal not reached"]
    elif lift_result["final_rise"] < args.min_lift_rise_success:
        status = "FAIL_LIFT_RISE_TOO_SMALL"
        failure_reasons = [
            f"final_rise {lift_result['final_rise']:.5f} < {args.min_lift_rise_success:.5f}"
        ]
    else:
        status = "SUCCESS_LIFT"

    final_contacts = lift_result["final_contacts"]

    result = {
        "format": "v4_12p4c_opposition_contact_seek_close_debug",
        "model": str(resolve_path(args.model)),
        "candidate": str(resolve_path(args.candidate)),
        "source_json": str(source_json_path),
        "source_kind": source_kind,
        "object_body": object_body,
        "args": vars(args),
        "status": status,
        "failure_reasons": failure_reasons,
        "q_current": q_current,
        "q_pre": q_pre,
        "q_grasp": q_grasp,
        "q_lift": q_lift,
        "candidate_ctrl_source": candidate_ctrl_source,
        "candidate_ctrl": candidate_ctrl,
        "zero_open_ctrl": zero_open_ctrl,
        "side_open_ctrl": side_open_ctrl,
        "final_ctrl": dict(ctrl),
        "final_hand_qpos": read_hand_qpos(model, data),
        "close_ref_pos": close_ref_pos,
        "object_contact_state": dict(state["object_contact"]),
        "support_contact_state": dict(state["support_contact"]),
        "frozen_state": dict(state["frozen"]),
        "first_object_contact": state["first_object_contact"],
        "first_support_contact": state["first_support_contact"],
        "phase_results": phase_results,
        "lift_result": lift_result,
        "logs": logs,
    }

    save_json(args.out, result)

    print("\n========== V4.12P4C SUMMARY ==========")
    print("status:", status)
    print("failure_reasons:", failure_reasons)
    print("object_contact_state:", state["object_contact"])
    print("support_contact_state:", state["support_contact"])
    print("frozen_state:", state["frozen"])
    print("final_ctrl:", ctrl)
    print("final_hand_qpos:", read_hand_qpos(model, data))
    print("final_rise:", lift_result["final_rise"])
    print("final_contacts:", {
        "hand_object": final_contacts["hand_object"],
        "hand_support": final_contacts["hand_support"],
        "fr3_object": final_contacts["fr3_object"],
        "object_support": final_contacts["object_support"],
        "object_groups": final_contacts["object_groups"],
        "support_groups": final_contacts["support_groups"],
    })
    print("saved:", resolve_path(args.out))
    print("======================================\n")

    return result


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

    ap.add_argument("--arm-hold-mode", choices=["hard", "ctrl"], default="hard")
    ap.add_argument("--direct-hand-qpos", action="store_true")

    ap.add_argument("--move-steps", type=int, default=120)
    ap.add_argument("--thumb-preshape-steps", type=int, default=120)

    ap.add_argument("--thumb-roll-preshape", type=float, default=0.56)
    ap.add_argument("--thumb-yaw-preshape", type=float, default=1.15)
    ap.add_argument("--thumb-pitch-open", type=float, default=0.08)
    ap.add_argument("--use-candidate-thumb-preshape", action="store_true")

    ap.add_argument("--finger-seek-duration", type=float, default=1.4)
    ap.add_argument("--thumb-comp-duration", type=float, default=1.8)
    ap.add_argument("--micro-squeeze-duration", type=float, default=0.8)
    ap.add_argument("--hold-duration", type=float, default=0.6)
    ap.add_argument("--lift-duration", type=float, default=2.0)

    ap.add_argument("--finger-seek-speed", type=float, default=0.45)
    ap.add_argument("--thumb-comp-speed", type=float, default=0.35)
    ap.add_argument("--micro-finger-speed", type=float, default=0.12)
    ap.add_argument("--micro-thumb-speed", type=float, default=0.10)

    ap.add_argument("--soft-object-push-disp", type=float, default=0.006)
    ap.add_argument("--hard-object-push-disp", type=float, default=0.025)
    ap.add_argument("--micro-push-increase-limit", type=float, default=0.003)

    ap.add_argument("--min-total-object-groups", type=int, default=2)
    ap.add_argument("--min-non-thumb-groups", type=int, default=1)
    ap.add_argument("--allow-no-thumb", action="store_true")

    ap.add_argument("--fail-on-hand-support", action="store_true", default=True)
    ap.add_argument("--no-fail-on-hand-support", dest="fail_on_hand_support", action="store_false")

    ap.add_argument("--min-micro-hold-time", type=float, default=0.20)
    ap.add_argument("--min-lift-rise-success", type=float, default=0.015)
    ap.add_argument("--lift-even-if-fail", action="store_true")

    ap.add_argument("--frame-sleep", type=float, default=0.002)
    ap.add_argument("--log-dt", type=float, default=0.1)

    args = ap.parse_args()

    model_path = resolve_path(args.model)
    candidate_path = resolve_path(args.candidate)

    candidate = load_json(candidate_path)

    if args.plan_json:
        source_json_path = resolve_path(args.plan_json)
        src = load_json(source_json_path)
        best, source_kind = selected_plan_from_json(src, args.which)
    elif args.p3_json:
        source_json_path = resolve_path(args.p3_json)
        src = load_json(source_json_path)
        best, source_kind = selected_plan_from_json(src, args.which)
    else:
        raise RuntimeError("pass --p3-json or --plan-json")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    viewer_cm = None
    viewer = None

    try:
        if args.viewer:
            if mujoco.viewer is None:
                raise RuntimeError("mujoco.viewer is not available")
            viewer_cm = mujoco.viewer.launch_passive(model, data)
            viewer = viewer_cm.__enter__()

            # 给一个比较稳定的默认观察视角；不影响仿真。
            viewer.cam.lookat[:] = np.array([0.48, 0.0, 0.25])
            viewer.cam.distance = 0.95
            viewer.cam.azimuth = 130
            viewer.cam.elevation = -25

        run_p4c(
            model=model,
            data=data,
            viewer=viewer,
            args=args,
            candidate=candidate,
            src=src,
            best=best,
            source_kind=source_kind,
            source_json_path=source_json_path,
        )

        if args.viewer:
            print("[VIEWER] 播放完成。关闭窗口即可退出。")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.03)

    finally:
        if viewer_cm is not None:
            viewer_cm.__exit__(None, None, None)


if __name__ == "__main__":
    main()
