#!/usr/bin/env python3
"""
文件名：
    run_v4_12p4f_target_close_debug.py

脚本类别：
    debug / runner / target-close / viewer

用途：
    本脚本用于 V4.12P4F 阶段，专门修复 P4C 中“四指没有真正闭合到目标”的问题。
    它不再使用 finger_seek_duration × finger_seek_speed 来间接决定四指闭合量，
    而是显式构造 close_target：
        四指 close target = candidate prior hand ctrl * finger_close_scale
        thumb roll/yaw = P4E-fast best_config 中的对握姿态，若没有则使用 candidate prior
        thumb pitch = max(candidate prior thumb pitch, thumb_open + finger_prior_scale * gain)

输入：
    1. --model
       MuJoCo XML 场景。
    2. --candidate
       candidate JSON，必须包含 hand.o7_active_ctrl。
    3. --p3-json
       P3 输出 JSON，读取 best_available / best_pass 中的 q_pre/q_grasp/q_lift。
    4. --best-config
       P4E-fast 输出的 best_config.json，可选但推荐，用于读取局部修正后的 thumb preshape。
    5. --object-body
       物体 body 名称，例如 grasp_can。

输出：
    1. --out 指定的 JSON，记录各阶段接触、ctrl、qpos、object displacement。
    2. viewer 可视化过程。

当前流程位置：
    P4E-fast best candidate / P3 best plan
        -> P4F target-close runner
        -> 检查四指是否真的闭合到 candidate prior
        -> 检查是否出现 thumb + non-thumb 接触

本脚本不负责：
    1. 不重新做 IK。
    2. 不重新做 P3 碰撞搜索。
    3. 不做新的局部几何搜索。
    4. 不判断最终方案已经泛化成功。
    5. 不把四指角度写死；四指目标来自 candidate prior 和 finger_close_scale。
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

HAND_JOINTS = [
    "thumb_cmc_roll",
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]

FOUR_FINGER_JOINTS = [
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]


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
    return mujoco.mj_name2id(model, objtype, str(name))


def mj_name(model, objtype, idx):
    return mujoco.mj_id2name(model, objtype, int(idx)) or ""


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


def geom_body_name(model, gid):
    return body_name(model, int(model.geom_bodyid[int(gid)]))


def body_is_descendant(model, bid, root_bid):
    cur = int(bid)
    while cur > 0:
        if cur == int(root_bid):
            return True
        cur = int(model.body_parentid[cur])
    return cur == int(root_bid)


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
            return aid
    return -1


def ctrl_range(model, joint_name):
    aid = find_actuator_for_joint(model, joint_name)
    if aid >= 0 and bool(model.actuator_ctrllimited[aid]):
        lo, hi = model.actuator_ctrlrange[aid]
        return float(lo), float(hi)

    jid = joint_id(model, joint_name)
    if jid >= 0 and bool(model.jnt_limited[jid]):
        lo, hi = model.jnt_range[jid]
        return float(lo), float(hi)

    return -3.0, 3.0


def clamp_ctrl(model, joint_name, value):
    lo, hi = ctrl_range(model, joint_name)
    return float(np.clip(float(value), lo, hi))


def set_joint_qpos(model, data, joint_name, value):
    jid = joint_id(model, joint_name)
    if jid < 0:
        return False
    if int(model.jnt_type[jid]) not in [
        int(mujoco.mjtJoint.mjJNT_HINGE),
        int(mujoco.mjtJoint.mjJNT_SLIDE),
    ]:
        return False
    qadr = int(model.jnt_qposadr[jid])
    data.qpos[qadr] = clamp_ctrl(model, joint_name, value)
    return True


def get_joint_qpos(model, data, joint_name):
    jid = joint_id(model, joint_name)
    if jid < 0:
        return None
    if int(model.jnt_type[jid]) not in [
        int(mujoco.mjtJoint.mjJNT_HINGE),
        int(mujoco.mjtJoint.mjJNT_SLIDE),
    ]:
        return None
    qadr = int(model.jnt_qposadr[jid])
    return float(data.qpos[qadr])


def set_actuator_ctrl(model, data, joint_name, value):
    aid = find_actuator_for_joint(model, joint_name)
    if aid < 0:
        return False
    data.ctrl[aid] = clamp_ctrl(model, joint_name, value)
    return True


def set_arm_qpos_and_ctrl(model, data, q):
    for j in ARM_JOINTS:
        if j not in q:
            continue
        v = float(q[j])
        set_joint_qpos(model, data, j, v)
        set_actuator_ctrl(model, data, j, v)


def set_hand_ctrl(model, data, ctrl, direct_qpos=False):
    for j in HAND_JOINTS:
        if j not in ctrl:
            continue
        v = clamp_ctrl(model, j, ctrl[j])
        set_actuator_ctrl(model, data, j, v)
        if direct_qpos:
            set_joint_qpos(model, data, j, v)


def hand_qpos(model, data):
    out = {}
    for j in HAND_JOINTS:
        v = get_joint_qpos(model, data, j)
        if v is not None:
            out[j] = v
    return out


def interp_dict(a, b, alpha, keys):
    out = {}
    for k in keys:
        av = float(a.get(k, 0.0))
        bv = float(b.get(k, av))
        out[k] = av + float(alpha) * (bv - av)
    return out


def selected_plan(p3, which):
    item = p3.get(which)
    if item is None:
        raise RuntimeError(f"{which} is None in p3 json")
    for k in ["q_pre", "q_grasp", "q_lift"]:
        if k not in item:
            raise RuntimeError(f"{which} missing {k}")
    return item


def extract_candidate_ctrl(candidate, model):
    hand = candidate.get("hand", {}) or {}
    for key in ["o7_active_ctrl", "active_ctrl", "ctrl", "target_ctrl", "qpos"]:
        val = hand.get(key, None)
        if isinstance(val, dict):
            out = {}
            for j in HAND_JOINTS:
                if j in val:
                    out[j] = clamp_ctrl(model, j, val[j])
            if out:
                return out, f"hand.{key}"

    for key in ["o7_active_ctrl", "active_ctrl", "ctrl", "target_ctrl", "qpos"]:
        val = candidate.get(key, None)
        if isinstance(val, dict):
            out = {}
            for j in HAND_JOINTS:
                if j in val:
                    out[j] = clamp_ctrl(model, j, val[j])
            if out:
                return out, key

    raise RuntimeError("cannot extract candidate hand ctrl")


def extract_best_config_ctrl(best_config, model):
    if not best_config:
        return {}, ""

    rec = best_config.get("best_record", {}) or {}
    hc = rec.get("hand_config", {}) or {}
    ctrl = hc.get("ctrl", {}) or {}

    out = {}
    for j in HAND_JOINTS:
        if j in ctrl:
            out[j] = clamp_ctrl(model, j, ctrl[j])

    return out, "best_config.best_record.hand_config.ctrl" if out else ""


def make_open_ctrl(model):
    return {j: clamp_ctrl(model, j, 0.0) for j in HAND_JOINTS}


def make_side_open_ctrl(model, open_ctrl, candidate_ctrl, best_ctrl, args):
    ctrl = dict(open_ctrl)

    source = best_ctrl if best_ctrl else candidate_ctrl

    ctrl["thumb_cmc_roll"] = clamp_ctrl(
        model,
        "thumb_cmc_roll",
        source.get("thumb_cmc_roll", candidate_ctrl.get("thumb_cmc_roll", 0.0)),
    )
    ctrl["thumb_cmc_yaw"] = clamp_ctrl(
        model,
        "thumb_cmc_yaw",
        source.get("thumb_cmc_yaw", candidate_ctrl.get("thumb_cmc_yaw", 0.0)),
    )

    # thumb pitch 初始仍保持较打开，但来自 best/candidate，不写死绝对角。
    ctrl["thumb_cmc_pitch"] = clamp_ctrl(
        model,
        "thumb_cmc_pitch",
        source.get("thumb_cmc_pitch", candidate_ctrl.get("thumb_cmc_pitch", 0.0)),
    )

    if args.preshape_fingers_from_best and best_ctrl:
        for j in FOUR_FINGER_JOINTS:
            ctrl[j] = clamp_ctrl(model, j, best_ctrl.get(j, 0.0))
    else:
        for j in FOUR_FINGER_JOINTS:
            ctrl[j] = clamp_ctrl(model, j, 0.0)

    return ctrl


def make_close_target(model, side_open_ctrl, candidate_ctrl, args):
    target = dict(side_open_ctrl)

    for j in FOUR_FINGER_JOINTS:
        prior = float(candidate_ctrl.get(j, 0.0))
        target[j] = clamp_ctrl(model, j, prior * args.finger_close_scale)

    max_four_prior = max(abs(float(candidate_ctrl.get(j, 0.0))) for j in FOUR_FINGER_JOINTS)

    adaptive_thumb_pitch = (
        float(side_open_ctrl.get("thumb_cmc_pitch", 0.0))
        + args.thumb_pitch_from_finger_gain * max_four_prior
    )

    target["thumb_cmc_pitch"] = clamp_ctrl(
        model,
        "thumb_cmc_pitch",
        max(float(candidate_ctrl.get("thumb_cmc_pitch", 0.0)), adaptive_thumb_pitch),
    )

    target["thumb_cmc_roll"] = side_open_ctrl["thumb_cmc_roll"]
    target["thumb_cmc_yaw"] = side_open_ctrl["thumb_cmc_yaw"]

    return target


def object_pos(model, data, object_body):
    bid = body_id(model, object_body)
    if bid < 0:
        raise RuntimeError(f"cannot find object body: {object_body}")
    return data.xpos[bid].copy()


def group_of_geom(model, gid):
    text = f"{geom_name(model, gid)} {geom_body_name(model, gid)}".lower()
    for g in ["thumb", "index", "middle", "ring", "pinky"]:
        if g in text:
            return g
    return ""


def build_geom_sets(model, object_body):
    obj_bid = body_id(model, object_body)
    if obj_bid < 0:
        raise RuntimeError(f"cannot find object body: {object_body}")

    object_geoms = set()
    support_geoms = set()
    hand_geoms = set()
    fr3_geoms = set()

    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        gname = geom_name(model, gid).lower()
        bname = geom_body_name(model, gid).lower()
        text = gname + " " + bname

        if body_is_descendant(model, bid, obj_bid):
            object_geoms.add(gid)

        if ("object_pedestal" in text) or ("pedestal" in text) or ("support" in text) or (gname == "table"):
            support_geoms.add(gid)

        if group_of_geom(model, gid):
            hand_geoms.add(gid)

        if "fr3_link" in text or "fr3_" in text:
            fr3_geoms.add(gid)

    return {
        "object_geoms": object_geoms,
        "support_geoms": support_geoms,
        "hand_geoms": hand_geoms,
        "fr3_geoms": fr3_geoms,
    }


def contact_counts(model, data, sets):
    object_geoms = sets["object_geoms"]
    support_geoms = sets["support_geoms"]
    hand_geoms = sets["hand_geoms"]
    fr3_geoms = sets["fr3_geoms"]

    out = {
        "ncon": int(data.ncon),
        "hand_object": 0,
        "hand_support": 0,
        "fr3_object": 0,
        "object_support": 0,
        "fr3_support": 0,
        "object_groups": {},
        "support_groups": {},
        "pairs": [],
    }

    for i in range(data.ncon):
        c = data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)

        pair = {
            "geom1": geom_name(model, g1),
            "body1": geom_body_name(model, g1),
            "geom2": geom_name(model, g2),
            "body2": geom_body_name(model, g2),
            "dist": float(c.dist),
        }

        g1_obj = g1 in object_geoms
        g2_obj = g2 in object_geoms
        g1_sup = g1 in support_geoms
        g2_sup = g2 in support_geoms
        g1_hand = g1 in hand_geoms
        g2_hand = g2 in hand_geoms
        g1_fr3 = g1 in fr3_geoms
        g2_fr3 = g2 in fr3_geoms

        if (g1_hand and g2_obj) or (g2_hand and g1_obj):
            out["hand_object"] += 1
            hg = g1 if g1_hand else g2
            group = group_of_geom(model, hg)
            if group:
                out["object_groups"][group] = out["object_groups"].get(group, 0) + 1

        if (g1_hand and g2_sup) or (g2_hand and g1_sup):
            out["hand_support"] += 1
            hg = g1 if g1_hand else g2
            group = group_of_geom(model, hg)
            if group:
                out["support_groups"][group] = out["support_groups"].get(group, 0) + 1

        if (g1_fr3 and g2_obj) or (g2_fr3 and g1_obj):
            out["fr3_object"] += 1

        if (g1_obj and g2_sup) or (g2_obj and g1_sup):
            out["object_support"] += 1

        if (g1_fr3 and g2_sup) or (g2_fr3 and g1_sup):
            out["fr3_support"] += 1

        out["pairs"].append(pair)

    return out


def phase_steps(model, data, viewer, label, steps, callback, logs, args, sets, object_body, obj_ref):
    log_every = max(1, int(args.log_dt / model.opt.timestep))

    print(f"\n[PHASE] {label}, steps={steps}")

    for k in range(steps + 1):
        alpha = k / max(1, steps)
        t = k * float(model.opt.timestep)

        callback(alpha, t, k)

        mujoco.mj_forward(model, data)

        if k % log_every == 0 or k == steps:
            obj = object_pos(model, data, object_body)
            disp = float(np.linalg.norm(obj - obj_ref))
            counts = contact_counts(model, data, sets)
            row = {
                "phase": label,
                "step": int(k),
                "time": float(t),
                "alpha": float(alpha),
                "object_pos": obj,
                "object_disp": disp,
                "contacts": counts,
                "hand_qpos": hand_qpos(model, data),
            }
            logs.append(row)

            print(
                f"[{label}] {k:4d}/{steps} "
                f"disp={disp:.5f} "
                f"groups={counts['object_groups']} "
                f"hand_support={counts['hand_support']} "
                f"object_support={counts['object_support']} "
                f"hand_qpos={row['hand_qpos']}"
            )

        mujoco.mj_step(model, data)

        if viewer is not None:
            viewer.sync()
            if args.frame_sleep > 0:
                time.sleep(args.frame_sleep)


def run(args):
    model_path = resolve_path(args.model)
    candidate_path = resolve_path(args.candidate)
    p3_path = resolve_path(args.p3_json)
    best_config_path = resolve_path(args.best_config) if args.best_config else None

    candidate = load_json(candidate_path)
    p3 = load_json(p3_path)
    best_config = load_json(best_config_path) if best_config_path and best_config_path.exists() else {}

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    plan = selected_plan(p3, args.which)
    q_pre = plan["q_pre"]
    q_grasp = plan["q_grasp"]
    q_lift = plan["q_lift"]

    candidate_ctrl, candidate_ctrl_source = extract_candidate_ctrl(candidate, model)
    best_ctrl, best_ctrl_source = extract_best_config_ctrl(best_config, model)

    open_ctrl = make_open_ctrl(model)
    side_open_ctrl = make_side_open_ctrl(model, open_ctrl, candidate_ctrl, best_ctrl, args)
    close_target = make_close_target(model, side_open_ctrl, candidate_ctrl, args)

    sets = build_geom_sets(model, args.object_body)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)
    q_current = {j: get_joint_qpos(model, data, j) or 0.0 for j in ARM_JOINTS}
    obj_ref = object_pos(model, data, args.object_body)

    logs = []

    print("\n========== V4.12P4F TARGET CLOSE DEBUG ==========")
    print("model                :", model_path)
    print("candidate            :", candidate_path)
    print("p3_json              :", p3_path)
    print("best_config          :", best_config_path)
    print("which                :", args.which)
    print("object_body          :", args.object_body)
    print("candidate_ctrl_source:", candidate_ctrl_source)
    print("best_ctrl_source     :", best_ctrl_source)
    print("candidate_ctrl       :", candidate_ctrl)
    print("best_ctrl            :", best_ctrl)
    print("side_open_ctrl       :", side_open_ctrl)
    print("close_target         :", close_target)
    print("finger_close_scale   :", args.finger_close_scale)
    print("thumb_pitch_gain     :", args.thumb_pitch_from_finger_gain)
    print("direct_hand_qpos     :", args.direct_hand_qpos)
    print("geom_sets sizes      :", {k: len(v) for k, v in sets.items()})
    print("=================================================\n")

    def run_with_viewer(viewer=None):
        phase_steps(
            model, data, viewer,
            "move_to_pre_open",
            args.move_steps,
            lambda a, t, k: (
                set_arm_qpos_and_ctrl(model, data, interp_dict(q_current, q_pre, a, ARM_JOINTS)),
                set_hand_ctrl(model, data, open_ctrl, args.direct_hand_qpos),
            ),
            logs, args, sets, args.object_body, obj_ref,
        )

        phase_steps(
            model, data, viewer,
            "thumb_preshape",
            args.thumb_preshape_steps,
            lambda a, t, k: (
                set_arm_qpos_and_ctrl(model, data, q_pre),
                set_hand_ctrl(model, data, interp_dict(open_ctrl, side_open_ctrl, a, HAND_JOINTS), args.direct_hand_qpos),
            ),
            logs, args, sets, args.object_body, obj_ref,
        )

        phase_steps(
            model, data, viewer,
            "move_to_grasp_side_open",
            args.move_steps,
            lambda a, t, k: (
                set_arm_qpos_and_ctrl(model, data, interp_dict(q_pre, q_grasp, a, ARM_JOINTS)),
                set_hand_ctrl(model, data, side_open_ctrl, args.direct_hand_qpos),
            ),
            logs, args, sets, args.object_body, obj_ref,
        )

        close_steps = max(1, int(args.close_duration / model.opt.timestep))
        frozen = {g: False for g in ["thumb", "index", "middle", "ring", "pinky"]}
        first_object_contact = {}

        print("\n[PHASE] target_close_to_candidate_prior")

        # P4F fix:
        # close 阶段一旦因为接触冻结、object push 阈值、或者自然结束而停下，
        # 后续 hold/lift 必须沿用 close 结束那一刻的 ctrl。
        # 不能再强行切回 close_target，否则大拇指/四指会在 hold 阶段突然继续猛闭合，
        # 把物体压进支撑块，或者造成 thumb 与四指自碰。
        close_end_ctrl = dict(side_open_ctrl)

        for k in range(close_steps + 1):
            alpha = k / close_steps
            t = k * float(model.opt.timestep)

            ctrl = interp_dict(side_open_ctrl, close_target, alpha, HAND_JOINTS)

            if args.freeze_contacted_groups:
                for group in frozen:
                    if frozen[group]:
                        for j in HAND_JOINTS:
                            if group in j:
                                ctrl[j] = get_joint_qpos(model, data, j) or ctrl[j]

            set_arm_qpos_and_ctrl(model, data, q_grasp)
            set_hand_ctrl(model, data, ctrl, args.direct_hand_qpos)
            mujoco.mj_forward(model, data)

            # 记录 close 阶段真实执行到的最后 ctrl。
            # 如果启用了 freeze_contacted_groups，这里保存的是冻结后的 ctrl，
            # 不是原始 close_target。
            close_end_ctrl = dict(ctrl)

            counts = contact_counts(model, data, sets)
            obj = object_pos(model, data, args.object_body)
            disp = float(np.linalg.norm(obj - obj_ref))

            for group in counts["object_groups"]:
                if group not in first_object_contact:
                    first_object_contact[group] = {
                        "time": float(t),
                        "step": int(k),
                        "ctrl": dict(ctrl),
                        "hand_qpos": hand_qpos(model, data),
                    }
                if args.freeze_contacted_groups:
                    frozen[group] = True

            if k % max(1, int(args.log_dt / model.opt.timestep)) == 0 or k == close_steps:
                row = {
                    "phase": "target_close_to_candidate_prior",
                    "step": int(k),
                    "time": float(t),
                    "alpha": float(alpha),
                    "ctrl": dict(ctrl),
                    "hand_qpos": hand_qpos(model, data),
                    "object_pos": obj,
                    "object_disp": disp,
                    "contacts": counts,
                    "frozen": dict(frozen),
                }
                logs.append(row)

                print(
                    f"[target_close] t={t:.3f} "
                    f"disp={disp:.5f} "
                    f"groups={counts['object_groups']} "
                    f"support_groups={counts['support_groups']} "
                    f"ctrl_four={{"
                    f"i:{ctrl['index_mcp_pitch']:.3f}, "
                    f"m:{ctrl['middle_mcp_pitch']:.3f}, "
                    f"r:{ctrl['ring_mcp_pitch']:.3f}, "
                    f"p:{ctrl['pinky_mcp_pitch']:.3f}"
                    f"}}"
                )

            if disp > args.hard_object_push_disp:
                print(f"[STOP] hard object push disp {disp:.5f} > {args.hard_object_push_disp:.5f}")
                break

            mujoco.mj_step(model, data)
            if viewer is not None:
                viewer.sync()
                if args.frame_sleep > 0:
                    time.sleep(args.frame_sleep)

        final_counts = contact_counts(model, data, sets)
        object_contact_state = {g: (g in final_counts["object_groups"]) for g in ["thumb", "index", "middle", "ring", "pinky"]}
        has_thumb = object_contact_state["thumb"]
        has_non_thumb = any(object_contact_state[g] for g in ["index", "middle", "ring", "pinky"])
        close_success = bool(has_thumb and has_non_thumb)

        print("\n========== CLOSE SUMMARY ==========")
        print("object_contact_state:", object_contact_state)
        print("final_counts:", final_counts)
        print("final_hand_qpos:", hand_qpos(model, data))
        print("first_object_contact:", first_object_contact)
        print("close_success:", close_success)
        print("===================================")

        held_ctrl = dict(close_end_ctrl)
        print("[HELD CTRL] hold/lift will use close_end_ctrl instead of close_target:", held_ctrl)

        phase_steps(
            model, data, viewer,
            "hold_after_close",
            max(1, int(args.hold_duration / model.opt.timestep)),
            lambda a, t, k: (
                set_arm_qpos_and_ctrl(model, data, q_grasp),
                set_hand_ctrl(model, data, held_ctrl, args.direct_hand_qpos),
            ),
            logs, args, sets, args.object_body, obj_ref,
        )

        if close_success or args.lift_even_if_fail:
            phase_steps(
                model, data, viewer,
                "lift",
                max(1, int(args.lift_duration / model.opt.timestep)),
                lambda a, t, k: (
                    set_arm_qpos_and_ctrl(model, data, interp_dict(q_grasp, q_lift, a, ARM_JOINTS)),
                    set_hand_ctrl(model, data, held_ctrl, args.direct_hand_qpos),
                ),
                logs, args, sets, args.object_body, obj_ref,
            )
        else:
            print("[SKIP LIFT] close contact goal not reached.")

        return {
            "close_success": close_success,
            "object_contact_state": object_contact_state,
            "final_counts": final_counts,
            "first_object_contact": first_object_contact,
        }

    if args.viewer:
        if mujoco.viewer is None:
            raise RuntimeError("mujoco.viewer is not available")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            result = run_with_viewer(viewer)
            print("[VIEWER] 播放完成。关闭窗口即可退出。")
            while viewer.is_running() and args.keep_viewer_open:
                viewer.sync()
                time.sleep(0.05)
    else:
        result = run_with_viewer(None)

    final_obj = object_pos(model, data, args.object_body)
    final_rise = float(final_obj[2] - obj_ref[2])
    final_counts = contact_counts(model, data, sets)

    status = "SUCCESS" if result["close_success"] and final_rise >= args.min_lift_rise_success else "FAIL_CONTACT_OR_LIFT"
    if result["close_success"] and not (final_rise >= args.min_lift_rise_success):
        status = "SUCCESS_CLOSE_ONLY"

    out = {
        "format": "v4_12p4f_target_close_debug",
        "model": str(model_path),
        "candidate": str(candidate_path),
        "p3_json": str(p3_path),
        "best_config": str(best_config_path) if best_config_path else "",
        "which": args.which,
        "object_body": args.object_body,
        "args": vars(args),
        "candidate_ctrl_source": candidate_ctrl_source,
        "best_ctrl_source": best_ctrl_source,
        "candidate_ctrl": candidate_ctrl,
        "best_ctrl": best_ctrl,
        "open_ctrl": open_ctrl,
        "side_open_ctrl": side_open_ctrl,
        "close_target": close_target,
        "held_ctrl": held_ctrl,
        "q_pre": q_pre,
        "q_grasp": q_grasp,
        "q_lift": q_lift,
        "status": status,
        "close_result": result,
        "final_rise": final_rise,
        "final_counts": final_counts,
        "final_hand_qpos": hand_qpos(model, data),
        "logs": logs,
    }

    save_json(args.out, out)

    print("\n========== V4.12P4F TARGET CLOSE RESULT ==========")
    print("status:", status)
    print("close_success:", result["close_success"])
    print("object_contact_state:", result["object_contact_state"])
    print("final_rise:", final_rise)
    print("final_counts:", final_counts)
    print("final_hand_qpos:", hand_qpos(model, data))
    print("saved:", resolve_path(args.out))
    print("=================================================\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--p3-json", required=True)
    ap.add_argument("--best-config", default="")
    ap.add_argument("--which", default="best_available", choices=["best_available", "best_pass"])
    ap.add_argument("--object-body", default="grasp_can")
    ap.add_argument("--out", required=True)

    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--keep-viewer-open", action="store_true")
    ap.add_argument("--direct-hand-qpos", action="store_true")

    ap.add_argument("--move-steps", type=int, default=100)
    ap.add_argument("--thumb-preshape-steps", type=int, default=100)
    ap.add_argument("--close-duration", type=float, default=1.6)
    ap.add_argument("--hold-duration", type=float, default=0.6)
    ap.add_argument("--lift-duration", type=float, default=1.5)

    ap.add_argument("--finger-close-scale", type=float, default=1.0)
    ap.add_argument("--thumb-pitch-from-finger-gain", type=float, default=0.35)
    ap.add_argument("--preshape-fingers-from-best", action="store_true")
    ap.add_argument("--freeze-contacted-groups", action="store_true")

    ap.add_argument("--hard-object-push-disp", type=float, default=0.020)
    ap.add_argument("--min-lift-rise-success", type=float, default=0.015)
    ap.add_argument("--lift-even-if-fail", action="store_true")

    ap.add_argument("--log-dt", type=float, default=0.1)
    ap.add_argument("--frame-sleep", type=float, default=0.002)

    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
