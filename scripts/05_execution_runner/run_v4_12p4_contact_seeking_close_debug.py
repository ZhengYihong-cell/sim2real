#!/usr/bin/env python3
"""
文件名：
    run_v4_12p4_contact_seeking_close_debug.py

脚本类别：
    debug / runner / contact-seeking-close

用途：
    本脚本用于 V4.12P4 阶段，在 P3 已经选出较平滑机械臂 IK 轨迹的基础上，
    实现“接触反馈驱动的手指闭合”，而不是把数据集 hand ctrl 当作最终闭合命令。
    它会让机械臂到达 q_pre 和 q_grasp，然后保持 q_grasp，逐步闭合 O7 手指。
    每个 finger group 一旦接触 object，就冻结该组，不再继续内压。
    未接触的 finger group 继续低速闭合，直到满足接触分布或超时。
    最后保持当前接触 ctrl，并执行 q_grasp -> q_lift。

输入：
    1. MuJoCo XML 模型文件。
    2. candidate JSON，用于读取 object.body 和候选手型趋势。
    3. P3 JSON 或 P3 plan JSON，用于读取 q_pre、q_grasp、q_lift。
    4. object_body，例如 grasp_can。

输出：
    1. JSON 结果文件，记录每个阶段的 object pose、contact 统计、finger group 接触事件。
    2. 可选 viewer 动画。
    3. 终端打印最终是否形成 thumb + non-thumb 接触，以及是否 lift。

当前流程位置：
    candidate
        -> P2 Pinocchio 多 seed IK
        -> P3 MuJoCo 路径预检选 q_pre/q_grasp/q_lift
        -> P4 本脚本执行 contact-seeking close
        -> 后续固化为真正 runner / selector 闭环

本脚本不负责：
    1. 不重新求 IK。
    2. 不重新做 P3 组合搜索。
    3. 不做大范围候选生成。
    4. 不把闭合幅度写死成某个倍率。
    5. 不保证最终实机可执行，只用于验证 contact-driven close 逻辑。
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
    "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
    "fr3_joint5", "fr3_joint6", "fr3_joint7",
]

FINGER_GROUPS = {
    "thumb": [
        "thumb_cmc_roll",
        "thumb_cmc_yaw",
        "thumb_cmc_pitch",
    ],
    "index": [
        "index_mcp_pitch",
    ],
    "middle": [
        "middle_mcp_pitch",
    ],
    "ring": [
        "ring_mcp_pitch",
    ],
    "pinky": [
        "pinky_mcp_pitch",
    ],
}

ACTIVE_HAND_JOINTS = []
for _g, _js in FINGER_GROUPS.items():
    ACTIVE_HAND_JOINTS.extend(_js)

HAND_TOKENS = ["thumb", "index", "middle", "ring", "pinky", "hand", "palm"]
SUPPORT_TOKENS = ["object_pedestal", "pedestal", "support", "table"]
FR3_TOKENS = ["fr3_link", "fr3_"]


def resolve_path(p):
    p = Path(str(p)).expanduser()
    if not p.is_absolute():
        p = PROJECT / p
    return p


def load_json(p):
    with open(resolve_path(p), "r") as f:
        return json.load(f)


def save_json(p, obj):
    p = resolve_path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(to_jsonable(obj), f, indent=2)


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


def mj_id(model, objtype, name):
    if name is None:
        return -1
    return mujoco.mj_name2id(model, objtype, str(name))


def body_id(model, name):
    return mj_id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def joint_id(model, name):
    return mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def actuator_id(model, name):
    return mj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def geom_name(model, gid):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid))
    return name or f"geom_{gid}"


def body_name(model, bid):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(bid))
    return name or f"body_{bid}"


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
        text = (gname + " " + bname).lower()
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
        f"{joint_name}_act",
        f"{joint_name}_ctrl",
        f"{joint_name}_motor",
    ]
    for name in candidates:
        aid = actuator_id(model, name)
        if aid >= 0:
            return aid, name
    return -1, ""


def get_joint_qpos(model, data, name):
    jid = joint_id(model, name)
    if jid < 0:
        return None
    adr = int(model.jnt_qposadr[jid])
    jtype = int(model.jnt_type[jid])
    if jtype in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
        return float(data.qpos[adr])
    return None


def set_joint_qpos(model, data, name, value):
    jid = joint_id(model, name)
    if jid < 0:
        return False
    adr = int(model.jnt_qposadr[jid])
    jtype = int(model.jnt_type[jid])
    if jtype in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
        data.qpos[adr] = float(value)
        return True
    return False


def set_actuator_ctrl(model, data, name, value):
    aid, act_name = find_actuator_for_joint(model, name)
    if aid < 0:
        return False, ""
    data.ctrl[aid] = float(value)
    return True, act_name


def actuator_ctrl_range(model, joint_name):
    aid, act_name = find_actuator_for_joint(model, joint_name)
    if aid < 0:
        return None, act_name

    limited = bool(model.actuator_ctrllimited[aid])
    lo, hi = model.actuator_ctrlrange[aid]

    if not limited:
        lo, hi = -3.0, 3.0

    return (float(lo), float(hi)), act_name


def clamp_ctrl(model, joint_name, value):
    cr, _ = actuator_ctrl_range(model, joint_name)
    if cr is None:
        return float(value)
    lo, hi = cr
    return float(np.clip(float(value), lo, hi))


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


def apply_hand_ctrl(model, data, ctrl):
    applied = {}
    missing = {}
    for j, v in (ctrl or {}).items():
        ok, act_name = set_actuator_ctrl(model, data, j, v)
        if ok:
            applied[j] = act_name
        else:
            missing[j] = v
    return applied, missing


def current_arm_qdict(model, data):
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


def interp_qdict(qa, qb, s):
    keys = sorted(set((qa or {}).keys()) | set((qb or {}).keys()))
    out = {}
    for k in keys:
        a = float((qa or {}).get(k, (qb or {}).get(k, 0.0)))
        b = float((qb or {}).get(k, (qa or {}).get(k, 0.0)))
        out[k] = (1.0 - s) * a + s * b
    return out


def extract_candidate_hand_trend(candidate):
    hand = candidate.get("hand", {}) or {}

    candidate_keys = [
        "o7_active_ctrl",
        "active_ctrl",
        "ctrl",
        "target_ctrl",
        "qpos",
        "target_qpos",
    ]

    for key in candidate_keys:
        val = hand.get(key, None)
        if isinstance(val, dict):
            picked = {}
            for j in ACTIVE_HAND_JOINTS:
                if j in val:
                    picked[j] = float(val[j])
            if picked:
                return picked, f"hand.{key}"

    for key in candidate_keys:
        val = candidate.get(key, None)
        if isinstance(val, dict):
            picked = {}
            for j in ACTIVE_HAND_JOINTS:
                if j in val:
                    picked[j] = float(val[j])
            if picked:
                return picked, key

    return {}, "NOT_FOUND"


def infer_close_direction(model, candidate_trend):
    directions = {}
    details = {}

    for j in ACTIVE_HAND_JOINTS:
        trend = float(candidate_trend.get(j, 0.0))
        cr, act_name = actuator_ctrl_range(model, j)

        if abs(trend) > 1e-5:
            direction = 1.0 if trend > 0.0 else -1.0
            source = "candidate_trend_sign"
        else:
            if cr is None:
                direction = 1.0
                source = "default_positive_no_actuator"
            else:
                lo, hi = cr
                if abs(hi) >= abs(lo):
                    direction = 1.0
                else:
                    direction = -1.0
                source = "ctrlrange_larger_abs_bound"

        directions[j] = direction
        details[j] = {
            "candidate_trend": trend,
            "direction": direction,
            "ctrlrange": cr,
            "actuator_name": act_name,
            "source": source,
        }

    return directions, details


def ctrl_target_bound(model, joint_name, direction):
    cr, _ = actuator_ctrl_range(model, joint_name)
    if cr is None:
        return 2.0 * float(direction)
    lo, hi = cr
    return hi if direction >= 0 else lo


def initial_open_ctrl(model, data):
    ctrl = {}
    for j in ACTIVE_HAND_JOINTS:
        aid, _ = find_actuator_for_joint(model, j)
        if aid >= 0:
            ctrl[j] = float(data.ctrl[aid])
        else:
            ctrl[j] = 0.0
    return ctrl


def group_of_geom(model, gid):
    text = (geom_name(model, gid) + " " + body_name_of_geom(model, gid)).lower()
    for group in FINGER_GROUPS.keys():
        if group in text:
            return group
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
        "object_support": 0,
        "fr3_support": 0,
        "other": 0,
        "object_groups": {},
        "support_groups": {},
        "pairs": [],
    }

    def has(a, b, A, B):
        return (a in A and b in B) or (a in B and b in A)

    for i in range(data.ncon):
        c = data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)

        cls = "other"
        group = "unknown"

        if has(g1, g2, hand_set, object_set):
            cls = "hand_object"
            out["hand_object"] += 1
            hand_geom = g1 if g1 in hand_set else g2
            group = group_of_geom(model, hand_geom)
            out["object_groups"][group] = out["object_groups"].get(group, 0) + 1

        elif has(g1, g2, hand_set, support_set):
            cls = "hand_support"
            out["hand_support"] += 1
            hand_geom = g1 if g1 in hand_set else g2
            group = group_of_geom(model, hand_geom)
            out["support_groups"][group] = out["support_groups"].get(group, 0) + 1

        elif has(g1, g2, fr3_set, object_set):
            cls = "fr3_object"
            out["fr3_object"] += 1

        elif has(g1, g2, object_set, support_set):
            cls = "object_support"
            out["object_support"] += 1

        elif has(g1, g2, fr3_set, support_set):
            cls = "fr3_support"
            out["fr3_support"] += 1

        else:
            out["other"] += 1

        out["pairs"].append({
            "class": cls,
            "group": group,
            "geom1": geom_name(model, g1),
            "body1": body_name_of_geom(model, g1),
            "geom2": geom_name(model, g2),
            "body2": body_name_of_geom(model, g2),
            "dist": float(c.dist),
        })

    return out


def object_pose(model, data, object_body):
    bid = body_id(model, object_body)
    if bid < 0:
        return None
    return {
        "pos": data.xpos[bid].copy(),
        "xmat": data.xmat[bid].reshape(3, 3).copy(),
    }


def object_disp(model, data, object_body, object_initial_pos):
    pose = object_pose(model, data, object_body)
    if pose is None:
        return 999.0
    return float(np.linalg.norm(pose["pos"] - object_initial_pos))


def selected_plan_from_json(plan_or_p3, which):
    if "q_pre" in plan_or_p3 and "q_grasp" in plan_or_p3 and "q_lift" in plan_or_p3:
        return plan_or_p3, "plan_json"

    item = plan_or_p3.get(which, None)
    if item is None:
        raise RuntimeError(f"{which} is None in p3 json")
    return item, which


def step_with_arm_hold(model, data, q_hold, hand_ctrl, nstep=1):
    for _ in range(max(1, nstep)):
        apply_arm_q(model, data, q_hold, also_ctrl=True)
        apply_hand_ctrl(model, data, hand_ctrl)
        mujoco.mj_step(model, data)
        apply_arm_q(model, data, q_hold, also_ctrl=True)
        apply_hand_ctrl(model, data, hand_ctrl)
        mujoco.mj_forward(model, data)


def sync_viewer(viewer, sleep_s):
    if viewer is not None:
        viewer.sync()
        if sleep_s > 0:
            time.sleep(sleep_s)


def move_arm_segment(viewer, model, data, q_from, q_to, hand_ctrl, steps, sleep_s, label, object_body, logs):
    for i in range(steps + 1):
        s = i / float(max(steps, 1))
        q = interp_qdict(q_from, q_to, s)

        apply_arm_q(model, data, q, also_ctrl=True)
        apply_hand_ctrl(model, data, hand_ctrl)
        mujoco.mj_forward(model, data)

        if i in [0, steps // 2, steps]:
            pose = object_pose(model, data, object_body)
            logs.append({
                "phase": label,
                "step": i,
                "s": s,
                "object_pos": pose["pos"] if pose else None,
                "hand_qpos": read_hand_qpos(model, data),
                "hand_ctrl": read_hand_ctrl(model, data),
            })
            print(f"[{label}] {i}/{steps} object_pos={pose['pos'] if pose else None}")

        sync_viewer(viewer, sleep_s)


def has_required_contact(contact_state, require_thumb=True, min_non_thumb_groups=1, min_total_groups=2):
    groups = set(k for k, v in contact_state.items() if v)
    non_thumb = groups - {"thumb"}

    if require_thumb and "thumb" not in groups:
        return False

    if len(non_thumb) < min_non_thumb_groups:
        return False

    if len(groups) < min_total_groups:
        return False

    return True


def contact_seeking_close(
    viewer,
    model,
    data,
    q_grasp,
    object_body,
    geom_sets,
    open_ctrl,
    close_dirs,
    args,
    logs,
):
    dt = float(model.opt.timestep)
    if dt <= 0:
        dt = 0.002

    ctrl = dict(open_ctrl)
    frozen_groups = {g: False for g in FINGER_GROUPS.keys()}
    object_contact_state = {g: False for g in FINGER_GROUPS.keys()}
    support_contact_state = {g: False for g in FINGER_GROUPS.keys()}
    first_object_contact = {}
    first_support_contact = {}
    close_events = []

    pose0 = object_pose(model, data, object_body)
    object_initial_pos = pose0["pos"].copy() if pose0 else np.zeros(3)

    max_steps = int(args.close_duration / dt)
    log_interval = max(1, int(args.log_dt / dt))

    failure_reason = ""
    reached_contact_goal = False

    print("\n========== CONTACT-SEEKING CLOSE ==========")
    print("dt:", dt, "max_steps:", max_steps)
    print("open_ctrl:", open_ctrl)
    print("close_dirs:", close_dirs)
    print("object_initial_pos:", object_initial_pos)

    for step in range(max_steps + 1):
        t = step * dt

        # 每个未冻结 group 继续低速向闭合方向移动
        for group, joints in FINGER_GROUPS.items():
            if frozen_groups[group]:
                continue

            for j in joints:
                direction = float(close_dirs.get(j, 1.0))
                target_bound = ctrl_target_bound(model, j, direction)
                current = float(ctrl.get(j, 0.0))

                delta = direction * float(args.close_speed) * dt
                nxt = current + delta

                if direction >= 0:
                    nxt = min(nxt, target_bound)
                else:
                    nxt = max(nxt, target_bound)

                ctrl[j] = clamp_ctrl(model, j, nxt)

        step_with_arm_hold(model, data, q_grasp, ctrl, nstep=1)
        cs = contact_summary(model, data, geom_sets)

        # 更新 object contact
        for group in FINGER_GROUPS.keys():
            if cs["object_groups"].get(group, 0) > 0:
                if not object_contact_state[group]:
                    first_object_contact[group] = {
                        "time": t,
                        "step": step,
                        "ctrl": dict(ctrl),
                        "hand_qpos": read_hand_qpos(model, data),
                    }
                    close_events.append({
                        "event": "first_object_contact",
                        "group": group,
                        "time": t,
                        "step": step,
                    })
                    print(f"[CONTACT object] group={group} t={t:.3f}")

                object_contact_state[group] = True
                frozen_groups[group] = True

        # 更新 support contact，hand-support 直接失败
        for group in FINGER_GROUPS.keys():
            if cs["support_groups"].get(group, 0) > 0:
                if not support_contact_state[group]:
                    first_support_contact[group] = {
                        "time": t,
                        "step": step,
                        "ctrl": dict(ctrl),
                        "hand_qpos": read_hand_qpos(model, data),
                    }
                    close_events.append({
                        "event": "first_support_contact",
                        "group": group,
                        "time": t,
                        "step": step,
                    })
                    print(f"[CONTACT support] group={group} t={t:.3f}")

                support_contact_state[group] = True

        if cs["hand_support"] > 0:
            failure_reason = f"hand_support_contact during close at t={t:.3f}"
            print("[FAIL]", failure_reason)
            break

        disp = object_disp(model, data, object_body, object_initial_pos)
        if disp > args.max_object_push_disp:
            failure_reason = f"object_pushed_too_far during close disp={disp:.5f} > {args.max_object_push_disp:.5f}"
            print("[FAIL]", failure_reason)
            break

        reached_contact_goal = has_required_contact(
            object_contact_state,
            require_thumb=not args.allow_no_thumb,
            min_non_thumb_groups=args.min_non_thumb_groups,
            min_total_groups=args.min_total_object_groups,
        )

        if reached_contact_goal:
            print("[OK] required contact distribution reached at t=", round(t, 4))
            break

        if step % log_interval == 0 or step == max_steps:
            pose = object_pose(model, data, object_body)
            logs.append({
                "phase": "contact_seeking_close",
                "step": step,
                "time": t,
                "ctrl": dict(ctrl),
                "hand_qpos": read_hand_qpos(model, data),
                "contacts": {
                    "ncon": cs["ncon"],
                    "hand_object": cs["hand_object"],
                    "hand_support": cs["hand_support"],
                    "fr3_object": cs["fr3_object"],
                    "object_support": cs["object_support"],
                    "object_groups": dict(cs["object_groups"]),
                    "support_groups": dict(cs["support_groups"]),
                },
                "object_pos": pose["pos"] if pose else None,
                "object_disp": disp,
                "frozen_groups": dict(frozen_groups),
            })
            print(
                f"[close] t={t:.3f} "
                f"groups={cs['object_groups']} "
                f"support={cs['support_groups']} "
                f"disp={disp:.5f} "
                f"ctrl={ {k: round(v,3) for k,v in ctrl.items()} }"
            )

        sync_viewer(viewer, args.frame_sleep)

    print("final object_contact_state:", object_contact_state)
    print("final support_contact_state:", support_contact_state)
    print("final ctrl:", ctrl)
    print("final hand_qpos:", read_hand_qpos(model, data))
    print("failure_reason:", failure_reason)
    print("==========================================\n")

    return {
        "success_contact_distribution": bool(reached_contact_goal and not failure_reason),
        "failure_reason": failure_reason,
        "final_ctrl": ctrl,
        "final_hand_qpos": read_hand_qpos(model, data),
        "object_contact_state": object_contact_state,
        "support_contact_state": support_contact_state,
        "first_object_contact": first_object_contact,
        "first_support_contact": first_support_contact,
        "close_events": close_events,
    }


def hold_phase(viewer, model, data, q_hold, hand_ctrl, object_body, geom_sets, duration, sleep_s, label, logs):
    dt = float(model.opt.timestep)
    steps = int(duration / dt)
    log_interval = max(1, int(0.1 / dt))

    for step in range(steps + 1):
        step_with_arm_hold(model, data, q_hold, hand_ctrl, nstep=1)

        if step % log_interval == 0 or step == steps:
            cs = contact_summary(model, data, geom_sets)
            pose = object_pose(model, data, object_body)
            logs.append({
                "phase": label,
                "step": step,
                "time": step * dt,
                "contacts": {
                    "ncon": cs["ncon"],
                    "hand_object": cs["hand_object"],
                    "hand_support": cs["hand_support"],
                    "fr3_object": cs["fr3_object"],
                    "object_support": cs["object_support"],
                    "object_groups": dict(cs["object_groups"]),
                    "support_groups": dict(cs["support_groups"]),
                },
                "object_pos": pose["pos"] if pose else None,
                "hand_qpos": read_hand_qpos(model, data),
                "hand_ctrl": read_hand_ctrl(model, data),
            })
            print(f"[{label}] step={step}/{steps} contacts={cs['object_groups']} support={cs['support_groups']}")

        sync_viewer(viewer, sleep_s)


def lift_phase(viewer, model, data, q_grasp, q_lift, hand_ctrl, object_body, geom_sets, duration, sleep_s, logs):
    dt = float(model.opt.timestep)
    steps = int(duration / dt)
    log_interval = max(1, int(0.1 / dt))

    pose0 = object_pose(model, data, object_body)
    z0 = float(pose0["pos"][2]) if pose0 else 0.0

    for step in range(steps + 1):
        s = step / float(max(steps, 1))
        q = interp_qdict(q_grasp, q_lift, s)

        step_with_arm_hold(model, data, q, hand_ctrl, nstep=1)

        if step % log_interval == 0 or step == steps:
            cs = contact_summary(model, data, geom_sets)
            pose = object_pose(model, data, object_body)
            z = float(pose["pos"][2]) if pose else 0.0
            logs.append({
                "phase": "lift",
                "step": step,
                "time": step * dt,
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
                "hand_qpos": read_hand_qpos(model, data),
                "hand_ctrl": read_hand_ctrl(model, data),
            })
            print(
                f"[lift] step={step}/{steps} "
                f"rise={z-z0:+.5f} "
                f"hand_object={cs['hand_object']} "
                f"groups={cs['object_groups']} "
                f"object_support={cs['object_support']}"
            )

        sync_viewer(viewer, sleep_s)

    pose1 = object_pose(model, data, object_body)
    z1 = float(pose1["pos"][2]) if pose1 else z0
    cs1 = contact_summary(model, data, geom_sets)

    return {
        "initial_z": z0,
        "final_z": z1,
        "final_rise": z1 - z0,
        "final_contacts": cs1,
    }


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

    ap.add_argument("--move-steps", type=int, default=120)
    ap.add_argument("--close-duration", type=float, default=2.5)
    ap.add_argument("--hold-duration", type=float, default=0.5)
    ap.add_argument("--lift-duration", type=float, default=2.0)
    ap.add_argument("--close-speed", type=float, default=0.65)
    ap.add_argument("--frame-sleep", type=float, default=0.0)
    ap.add_argument("--log-dt", type=float, default=0.1)

    ap.add_argument("--min-total-object-groups", type=int, default=2)
    ap.add_argument("--min-non-thumb-groups", type=int, default=1)
    ap.add_argument("--allow-no-thumb", action="store_true")
    ap.add_argument("--max-object-push-disp", type=float, default=0.045)
    ap.add_argument("--min-lift-rise-success", type=float, default=0.015)

    args = ap.parse_args()

    model_path = resolve_path(args.model)
    candidate_path = resolve_path(args.candidate)

    candidate = load_json(candidate_path)

    if args.plan_json:
        src_json_path = resolve_path(args.plan_json)
        src = load_json(src_json_path)
        best, source_kind = selected_plan_from_json(src, args.which)
    elif args.p3_json:
        src_json_path = resolve_path(args.p3_json)
        src = load_json(src_json_path)
        best, source_kind = selected_plan_from_json(src, args.which)
    else:
        raise RuntimeError("pass --p3-json or --plan-json")

    object_body = args.object_body or best.get("object_body", "") or src.get("object_body", "") or (candidate.get("object") or {}).get("body", "")
    if not object_body:
        raise RuntimeError("cannot infer object_body; pass --object-body")

    q_pre = best["q_pre"]
    q_grasp = best["q_grasp"]
    q_lift = best["q_lift"]

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    q_current = current_arm_qdict(model, data)
    geom_sets = collect_geom_sets(model, object_body)

    candidate_trend, trend_source = extract_candidate_hand_trend(candidate)
    close_dirs, close_dir_details = infer_close_direction(model, candidate_trend)

    open_ctrl = initial_open_ctrl(model, data)
    # debug 阶段明确将 active hand ctrl 归零，避免继承旧状态
    for j in ACTIVE_HAND_JOINTS:
        open_ctrl[j] = 0.0

    viewer_cm = None
    viewer = None

    if args.viewer:
        if mujoco.viewer is None:
            raise RuntimeError("mujoco.viewer not available")
        viewer_cm = mujoco.viewer.launch_passive(model, data)
        viewer = viewer_cm.__enter__()

    logs = []
    status = "UNKNOWN"
    failure_reasons = []

    try:
        print("\n========== V4.12P4 CONTACT-SEEKING CLOSE ==========")
        print("model      :", model_path)
        print("candidate  :", candidate_path)
        print("source json:", src_json_path)
        print("source kind:", source_kind)
        print("object_body:", object_body)
        print("trend_source:", trend_source)
        print("candidate_trend:", candidate_trend)
        print("close_dir_details:")
        for j, d in close_dir_details.items():
            print(" ", j, d)
        print("q seeds/status:", best.get("pre_seed"), "->", best.get("grasp_seed"), "->", best.get("lift_seed"))
        print("===================================================\n")

        # reset and open hand
        mujoco.mj_resetData(model, data)
        mujoco.mj_forward(model, data)
        apply_arm_q(model, data, q_current, also_ctrl=True)
        apply_hand_ctrl(model, data, open_ctrl)
        mujoco.mj_forward(model, data)
        sync_viewer(viewer, 0.2 if args.viewer else 0.0)

        move_arm_segment(
            viewer, model, data,
            q_current, q_pre,
            open_ctrl,
            args.move_steps,
            args.frame_sleep,
            "move_to_pre",
            object_body,
            logs,
        )

        move_arm_segment(
            viewer, model, data,
            q_pre, q_grasp,
            open_ctrl,
            args.move_steps,
            args.frame_sleep,
            "move_to_grasp",
            object_body,
            logs,
        )

        close_result = contact_seeking_close(
            viewer, model, data,
            q_grasp,
            object_body,
            geom_sets,
            open_ctrl,
            close_dirs,
            args,
            logs,
        )

        if close_result["failure_reason"]:
            status = "FAIL_CLOSE"
            failure_reasons.append(close_result["failure_reason"])
        elif not close_result["success_contact_distribution"]:
            status = "FAIL_NO_CONTACT_DISTRIBUTION"
            failure_reasons.append("required thumb/non-thumb contact distribution not reached")
        else:
            status = "CONTACT_CLOSE_OK"

        final_ctrl = close_result["final_ctrl"]

        hold_phase(
            viewer, model, data,
            q_grasp,
            final_ctrl,
            object_body,
            geom_sets,
            args.hold_duration,
            args.frame_sleep,
            "hold_after_close",
            logs,
        )

        lift_result = lift_phase(
            viewer, model, data,
            q_grasp,
            q_lift,
            final_ctrl,
            object_body,
            geom_sets,
            args.lift_duration,
            args.frame_sleep,
            logs,
        )

        if status == "CONTACT_CLOSE_OK":
            if lift_result["final_rise"] >= args.min_lift_rise_success:
                status = "SUCCESS_LIFT"
            else:
                status = "FAIL_LIFT_RISE_TOO_SMALL"
                failure_reasons.append(
                    f"final_rise {lift_result['final_rise']:.5f} < {args.min_lift_rise_success:.5f}"
                )

        out = {
            "format": "v4_12p4_contact_seeking_close_debug",
            "model": str(model_path),
            "candidate": str(candidate_path),
            "source_json": str(src_json_path),
            "source_kind": source_kind,
            "object_body": object_body,
            "args": vars(args),
            "status": status,
            "failure_reasons": failure_reasons,
            "q_current": q_current,
            "q_pre": q_pre,
            "q_grasp": q_grasp,
            "q_lift": q_lift,
            "candidate_trend_source": trend_source,
            "candidate_trend": candidate_trend,
            "close_dir_details": close_dir_details,
            "open_ctrl": open_ctrl,
            "close_result": close_result,
            "lift_result": lift_result,
            "logs": logs,
        }

        save_json(args.out, out)

        print("\n========== V4.12P4 SUMMARY ==========")
        print("status:", status)
        print("failure_reasons:", failure_reasons)
        print("object_contact_state:", close_result["object_contact_state"])
        print("support_contact_state:", close_result["support_contact_state"])
        print("final_ctrl:", close_result["final_ctrl"])
        print("final_hand_qpos:", close_result["final_hand_qpos"])
        print("final_rise:", lift_result["final_rise"])
        print("final_contacts:", {
            "hand_object": lift_result["final_contacts"]["hand_object"],
            "hand_support": lift_result["final_contacts"]["hand_support"],
            "fr3_object": lift_result["final_contacts"]["fr3_object"],
            "object_support": lift_result["final_contacts"]["object_support"],
            "object_groups": lift_result["final_contacts"]["object_groups"],
        })
        print("saved:", resolve_path(args.out))
        print("=====================================\n")

        if args.viewer:
            print("viewer 播放完成。关闭窗口即可退出。")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.03)

    finally:
        if viewer_cm is not None:
            viewer_cm.__exit__(None, None, None)


if __name__ == "__main__":
    main()
