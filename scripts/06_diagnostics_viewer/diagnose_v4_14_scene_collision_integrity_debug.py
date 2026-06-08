#!/usr/bin/env python3
"""
脚本类型：
    debug / diagnostic / collision-integrity / v4.14

用途：
    诊断当前 V4.14 site-target 场景中 object / support / hand 三者是否存在碰撞层问题。
    重点检查：
        1. object 是否初始穿入 pedestal/support；
        2. object-support 是否真正开启碰撞；
        3. hand-support 是否真正开启碰撞；
        4. q_grasp / close / squeeze / q_lift 姿态下，手指是否已经穿支撑块；
        5. 当前 runner 是否因为强制写 qpos 而允许手指穿支撑。

输入：
    --model        当前 scene.xml
    --result-json  V4.14 result.json
    --object-body  物体 body，例如 grasp_object

输出：
    out_dir/collision_integrity_report.txt
    out_dir/collision_integrity_summary.json

当前流程位置：
    V4.14 site-target IK 已证明 frame 正确
        -> 本脚本检查碰撞层和支撑穿透
        -> 通过后才允许写正式 dynamic runner

不负责：
    1. 不运行抓取；
    2. 不修改 scene；
    3. 不修改 P4U1/P4U6；
    4. 不做姿态微调；
    5. 不把 force-lift 当成功。
"""

from pathlib import Path
import argparse
import json
import math
import numpy as np
import mujoco


PROJECT = Path.home() / "Projects/o7_mujoco_sim"

ARM_JOINTS = [
    "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
    "fr3_joint5", "fr3_joint6", "fr3_joint7",
]

HAND_TOKENS = ["thumb", "index", "middle", "ring", "pinky", "palm", "hand"]
SUPPORT_TOKENS = ["object_pedestal", "pedestal", "support", "table"]
BAD_SUPPORT_TOKENS = ["world_plane", "floor"]


def resolve(p):
    p = Path(str(p)).expanduser()
    if not p.is_absolute():
        p = PROJECT / p
    return p


def rel(p):
    p = Path(p)
    try:
        return str(p.relative_to(PROJECT))
    except Exception:
        return str(p)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def geom_name(model, gid):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or f"geom_{gid}"


def body_name(model, bid):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(bid)) or f"body_{bid}"


def geom_body_name(model, gid):
    return body_name(model, int(model.geom_bodyid[int(gid)]))


def name2id(model, objtype, name):
    return mujoco.mj_name2id(model, objtype, name)


def joint_qpos_addr(model, joint_name):
    jid = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return None
    return int(model.jnt_qposadr[jid])


def joint_dof_addr(model, joint_name):
    jid = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return None
    return int(model.jnt_dofadr[jid])


def actuator_for_joint(model, joint_name):
    jid = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return None
    for aid in range(model.nu):
        if int(model.actuator_trnid[aid, 0]) == jid:
            return int(aid)
    return None


def set_joint_qpos(model, data, joint_name, value):
    adr = joint_qpos_addr(model, joint_name)
    if adr is None:
        return False
    jid = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    v = float(value)
    if jid >= 0 and int(model.jnt_limited[jid]):
        lo, hi = model.jnt_range[jid]
        v = float(np.clip(v, lo, hi))
    data.qpos[adr] = v
    return True


def set_joint_ctrl(model, data, joint_name, value):
    aid = actuator_for_joint(model, joint_name)
    if aid is None:
        return False
    v = float(value)
    if int(model.actuator_ctrllimited[aid]):
        lo, hi = model.actuator_ctrlrange[aid]
        v = float(np.clip(v, lo, hi))
    data.ctrl[aid] = v
    return True


def quat_from_R(R):
    q = np.zeros(4, dtype=float)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).reshape(9))
    return q


def set_free_body_pose_from_T(model, data, body, T):
    bid = name2id(model, mujoco.mjtObj.mjOBJ_BODY, body)
    if bid < 0:
        raise RuntimeError(f"missing body: {body}")

    # 找这个 body 的 free joint
    free_jid = None
    for jid in range(model.njnt):
        if int(model.jnt_bodyid[jid]) == bid and int(model.jnt_type[jid]) == int(mujoco.mjtJoint.mjJNT_FREE):
            free_jid = jid
            break

    if free_jid is None:
        return False

    adr = int(model.jnt_qposadr[free_jid])
    R = np.asarray(T[:3, :3], dtype=float)
    p = np.asarray(T[:3, 3], dtype=float)
    q = quat_from_R(R)

    data.qpos[adr:adr + 3] = p
    data.qpos[adr + 3:adr + 7] = q
    return True


def set_arm_and_hand(model, data, q_arm, hand_ctrl):
    for j, v in (q_arm or {}).items():
        set_joint_qpos(model, data, j, v)
        set_joint_ctrl(model, data, j, v)
        da = joint_dof_addr(model, j)
        if da is not None:
            data.qvel[da] = 0.0

    for j, v in (hand_ctrl or {}).items():
        set_joint_qpos(model, data, j, v)
        set_joint_ctrl(model, data, j, v)
        da = joint_dof_addr(model, j)
        if da is not None:
            data.qvel[da] = 0.0

    mujoco.mj_forward(model, data)


def geom_mask_allowed(model, g1, g2):
    c1 = int(model.geom_contype[g1])
    a1 = int(model.geom_conaffinity[g1])
    c2 = int(model.geom_contype[g2])
    a2 = int(model.geom_conaffinity[g2])
    return bool((c1 & a2) or (c2 & a1))


def geom_info(model, gid):
    return {
        "id": int(gid),
        "name": geom_name(model, gid),
        "body": geom_body_name(model, gid),
        "type": int(model.geom_type[gid]),
        "contype": int(model.geom_contype[gid]),
        "conaffinity": int(model.geom_conaffinity[gid]),
        "group": int(model.geom_group[gid]),
        "size": np.asarray(model.geom_size[gid]).tolist(),
    }


def collect_object_geoms(model, object_body):
    bid = name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
    if bid < 0:
        raise RuntimeError(f"missing object body: {object_body}")
    return [
        gid for gid in range(model.ngeom)
        if int(model.geom_bodyid[gid]) == bid
    ]


def collect_support_geoms(model):
    out = []
    for gid in range(model.ngeom):
        s = (geom_name(model, gid) + " " + geom_body_name(model, gid)).lower()
        if any(bad in s for bad in BAD_SUPPORT_TOKENS):
            continue
        if any(tok in s for tok in SUPPORT_TOKENS):
            out.append(gid)
    return out


def collect_hand_geoms(model):
    out = []
    for gid in range(model.ngeom):
        s = (geom_name(model, gid) + " " + geom_body_name(model, gid)).lower()
        if any(tok in s for tok in SUPPORT_TOKENS + ["object", "grasp"]):
            continue
        if any(tok in s for tok in HAND_TOKENS):
            out.append(gid)
    return out


def signed_geom_distance(model, data, g1, g2, distmax=0.30):
    fromto = np.zeros(6, dtype=float)
    try:
        d = float(mujoco.mj_geomDistance(model, data, int(g1), int(g2), float(distmax), fromto))
        return d
    except Exception as e:
        return None


def pair_report(model, data, geoms_a, geoms_b, label, top_n=12):
    rows = []
    for ga in geoms_a:
        for gb in geoms_b:
            if ga == gb:
                continue
            d = signed_geom_distance(model, data, ga, gb)
            rows.append({
                "label": label,
                "geom_a": geom_info(model, ga),
                "geom_b": geom_info(model, gb),
                "mask_allowed": geom_mask_allowed(model, ga, gb),
                "signed_distance": d,
            })

    def key(r):
        d = r["signed_distance"]
        if d is None:
            return 1e9
        return float(d)

    rows.sort(key=key)
    return rows[:top_n], rows


def current_contacts(model, data, object_geoms, support_geoms, hand_geoms):
    object_set = set(object_geoms)
    support_set = set(support_geoms)
    hand_set = set(hand_geoms)

    out = []
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        tags = []
        if g1 in object_set or g2 in object_set:
            tags.append("object")
        if g1 in support_set or g2 in support_set:
            tags.append("support")
        if g1 in hand_set or g2 in hand_set:
            tags.append("hand")
        out.append({
            "geom1": geom_name(model, g1),
            "geom2": geom_name(model, g2),
            "dist": float(c.dist),
            "tags": tags,
        })
    return out


def analyze_state(model, data, state_name, q_arm, hand_ctrl, T_object, object_body):
    if T_object is not None:
        set_free_body_pose_from_T(model, data, object_body, np.asarray(T_object, dtype=float))
    data.qvel[:] = 0.0
    set_arm_and_hand(model, data, q_arm, hand_ctrl)

    object_geoms = collect_object_geoms(model, object_body)
    support_geoms = collect_support_geoms(model)
    hand_geoms = collect_hand_geoms(model)

    obj_sup_top, obj_sup_all = pair_report(model, data, object_geoms, support_geoms, "object-support")
    hand_sup_top, hand_sup_all = pair_report(model, data, hand_geoms, support_geoms, "hand-support")
    hand_obj_top, hand_obj_all = pair_report(model, data, hand_geoms, object_geoms, "hand-object")

    contacts = current_contacts(model, data, object_geoms, support_geoms, hand_geoms)

    def min_dist(rows):
        valid = [r for r in rows if r["signed_distance"] is not None]
        if not valid:
            return None
        return min(float(r["signed_distance"]) for r in valid)

    def min_mask_allowed(rows):
        valid = [r for r in rows if r["signed_distance"] is not None]
        if not valid:
            return None
        best = min(valid, key=lambda r: float(r["signed_distance"]))
        return bool(best["mask_allowed"])

    return {
        "state": state_name,
        "object_support_min_signed_distance": min_dist(obj_sup_all),
        "object_support_min_pair_mask_allowed": min_mask_allowed(obj_sup_all),
        "hand_support_min_signed_distance": min_dist(hand_sup_all),
        "hand_support_min_pair_mask_allowed": min_mask_allowed(hand_sup_all),
        "hand_object_min_signed_distance": min_dist(hand_obj_all),
        "hand_object_min_pair_mask_allowed": min_mask_allowed(hand_obj_all),
        "top_object_support_pairs": obj_sup_top,
        "top_hand_support_pairs": hand_sup_top,
        "top_hand_object_pairs": hand_obj_top,
        "ncon": int(data.ncon),
        "contacts": contacts[:40],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--result-json", required=True)
    ap.add_argument("--object-body", default="grasp_object")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    model_path = resolve(args.model)
    result_path = resolve(args.result_json)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    result = json.loads(result_path.read_text())

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    object_geoms = collect_object_geoms(model, args.object_body)
    support_geoms = collect_support_geoms(model)
    hand_geoms = collect_hand_geoms(model)

    T_object = result.get("T_world_object_after_settle", {}).get("T")
    q_grasp = result.get("ik_grasp", {}).get("q_arm", {})
    q_lift = result.get("ik_lift", {}).get("q_arm", {})
    side_open = result.get("side_open_ctrl", {})
    close = result.get("close_ctrl", {})
    squeeze = result.get("squeeze_ctrl", {})

    states = []
    states.append(analyze_state(model, data, "q_grasp_side_open", q_grasp, side_open, T_object, args.object_body))
    states.append(analyze_state(model, data, "q_grasp_close", q_grasp, close, T_object, args.object_body))
    states.append(analyze_state(model, data, "q_grasp_squeeze", q_grasp, squeeze, T_object, args.object_body))
    states.append(analyze_state(model, data, "q_lift_squeeze", q_lift, squeeze, T_object, args.object_body))

    summary = {
        "format": "v4_14_collision_integrity_diagnostic_v1",
        "model": rel(model_path),
        "result_json": rel(result_path),
        "object_body": args.object_body,
        "object_geoms": [geom_info(model, g) for g in object_geoms],
        "support_geoms": [geom_info(model, g) for g in support_geoms],
        "hand_geoms_count": len(hand_geoms),
        "states": states,
        "diagnosis_notes": [
            "signed_distance < 0 means geometric penetration/intersection.",
            "mask_allowed=False means MuJoCo collision pair is disabled by contype/conaffinity.",
            "If signed_distance is negative and runner hard-sets qpos every step, physics cannot resolve penetration because qpos is overwritten.",
        ],
    }

    save_json(out_dir / "collision_integrity_summary.json", summary)

    lines = []
    lines.append("========== V4.14 COLLISION INTEGRITY DIAGNOSTIC ==========")
    lines.append(f"model      : {rel(model_path)}")
    lines.append(f"result_json: {rel(result_path)}")
    lines.append(f"object_body: {args.object_body}")
    lines.append("")
    lines.append("---- geom counts ----")
    lines.append(f"object_geoms : {len(object_geoms)}")
    lines.append(f"support_geoms: {len(support_geoms)}")
    lines.append(f"hand_geoms   : {len(hand_geoms)}")
    lines.append("")

    lines.append("---- support geoms ----")
    for g in support_geoms:
        gi = geom_info(model, g)
        lines.append(
            f"{gi['name']} body={gi['body']} contype={gi['contype']} "
            f"conaffinity={gi['conaffinity']} group={gi['group']} size={gi['size']}"
        )
    lines.append("")

    for st in states:
        lines.append(f"---- state: {st['state']} ----")
        lines.append(f"object-support min signed distance : {st['object_support_min_signed_distance']}")
        lines.append(f"object-support nearest mask allowed: {st['object_support_min_pair_mask_allowed']}")
        lines.append(f"hand-support min signed distance   : {st['hand_support_min_signed_distance']}")
        lines.append(f"hand-support nearest mask allowed  : {st['hand_support_min_pair_mask_allowed']}")
        lines.append(f"hand-object min signed distance    : {st['hand_object_min_signed_distance']}")
        lines.append(f"hand-object nearest mask allowed   : {st['hand_object_min_pair_mask_allowed']}")
        lines.append(f"ncon                               : {st['ncon']}")

        lines.append("  top object-support pairs:")
        for r in st["top_object_support_pairs"][:5]:
            lines.append(
                f"    d={r['signed_distance']} mask={r['mask_allowed']} "
                f"{r['geom_a']['name']} <-> {r['geom_b']['name']}"
            )

        lines.append("  top hand-support pairs:")
        for r in st["top_hand_support_pairs"][:8]:
            lines.append(
                f"    d={r['signed_distance']} mask={r['mask_allowed']} "
                f"{r['geom_a']['name']} <-> {r['geom_b']['name']}"
            )

        lines.append("  top hand-object pairs:")
        for r in st["top_hand_object_pairs"][:5]:
            lines.append(
                f"    d={r['signed_distance']} mask={r['mask_allowed']} "
                f"{r['geom_a']['name']} <-> {r['geom_b']['name']}"
            )
        lines.append("")

    lines.append("---- interpretation ----")
    lines.append("1. object-support min distance < 0：物体初始就穿垫块，builder/放置高度/mesh bbox 需要修。")
    lines.append("2. hand-support min distance < 0：该抓握姿态本身穿垫块，必须在 selector/P3 前过滤。")
    lines.append("3. mask_allowed=False：即使几何穿透，MuJoCo 也不会产生碰撞反力，XML 碰撞 mask 错。")
    lines.append("4. mask_allowed=True 但仍穿透：runner 硬写 qpos 会把手强塞进垫块，正式 runner 必须只发 ctrl，不得每步覆盖 qpos。")
    lines.append("==========================================================")

    report = "\n".join(lines) + "\n"
    (out_dir / "collision_integrity_report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
