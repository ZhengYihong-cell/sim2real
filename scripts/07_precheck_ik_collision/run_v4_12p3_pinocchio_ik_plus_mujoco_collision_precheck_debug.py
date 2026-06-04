#!/usr/bin/env python3
"""
文件名：
    run_v4_12p3_pinocchio_ik_plus_mujoco_collision_precheck_debug.py

脚本类别：
    debug / precheck / diagnostic

用途：
    本脚本用于 V4.12P3 阶段，把 V4.12P2 生成的 Pinocchio 多 seed IK 结果
    放回 MuJoCo 模型中做路径碰撞预检和组合评分。

输入：
    1. V4.12P2 输出的 JSON，里面包含 pre / grasp / lift 三个目标的多组 IK 解。
    2. MuJoCo XML，用于 forward kinematics 和 mj_geomDistance 几何距离检测。
    3. candidate JSON，用于读取 O7 手部闭合 ctrl 和 object.body。
    4. object_body，例如 grasp_can。

输出：
    1. 每组 pre-grasp-lift IK 组合的路径最小距离。
    2. hand-support、fr3-object、hand-object 的关键 clearance。
    3. PASS_PRECHECK / FAIL_PRECHECK 判定。
    4. 最优 IK 组合 plan JSON，可供后续 runner 使用。

当前流程位置：
    candidate JSON
        -> V4.12P2 Pinocchio 多 seed IK
        -> V4.12P3 MuJoCo 路径碰撞预检与组合筛选
        -> 后续 V4.12P4/P5 动态 runner 使用预检通过的 q_pre/q_grasp/q_lift

本脚本不负责：
    1. 不重新求 Pinocchio IK。
    2. 不启动 viewer。
    3. 不执行动态抓取。
    4. 不修改 candidate。
    5. 不判断最终是否真实 lift 成功，只判断进入动态 runner 前的路径安全性。
"""

from pathlib import Path
import argparse
import itertools
import json
import math
import numpy as np
import mujoco


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


def mj_name(model, objtype, idx):
    if idx < 0:
        return None
    return mujoco.mj_id2name(model, objtype, int(idx))


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
    return mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"


def body_name(model, bid):
    return mj_name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body_{bid}"


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


def get_joint_value(model, data, name):
    jid = joint_id(model, name)
    if jid < 0:
        return None
    adr = int(model.jnt_qposadr[jid])
    jtype = int(model.jnt_type[jid])
    if jtype in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
        return float(data.qpos[adr])
    return None


def current_arm_qdict(model, data):
    out = {}
    for j in ARM_JOINTS:
        v = get_joint_value(model, data, j)
        if v is not None:
            out[j] = v
    return out


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
    for nm in [name, f"{name}_pos", f"{name}_act", f"{name}_ctrl"]:
        aid = actuator_id(model, nm)
        if aid >= 0:
            data.ctrl[aid] = float(value)
            return True
    return False


def apply_qdict(model, data, qdict, also_ctrl=True):
    set_joints = []
    set_ctrls = []
    missing = []

    for k, v in (qdict or {}).items():
        ok_j = set_joint_qpos(model, data, k, v)
        ok_c = False
        if also_ctrl:
            ok_c = set_actuator_ctrl(model, data, k, v)

        if ok_j:
            set_joints.append(k)
        if ok_c:
            set_ctrls.append(k)
        if not ok_j and not ok_c:
            missing.append(k)

    mujoco.mj_forward(model, data)

    return {
        "set_joints": set_joints,
        "set_ctrls": set_ctrls,
        "missing": missing,
    }


def extract_hand_ctrl(candidate):
    hand = candidate.get("hand", {}) or {}
    ctrl = hand.get("o7_active_ctrl", None)
    if isinstance(ctrl, dict):
        return {str(k): float(v) for k, v in ctrl.items()}
    return {}


def approx_geom_distance(model, data, g1, g2):
    fromto = np.zeros(6, dtype=float)
    try:
        d = mujoco.mj_geomDistance(model, data, int(g1), int(g2), 1.0, fromto)
        return float(d), fromto.copy(), "mj_geomDistance"
    except Exception:
        p1 = data.geom_xpos[int(g1)]
        p2 = data.geom_xpos[int(g2)]
        r1 = float(model.geom_rbound[int(g1)])
        r2 = float(model.geom_rbound[int(g2)])
        d = float(np.linalg.norm(p1 - p2) - r1 - r2)
        fromto[:3] = p1
        fromto[3:] = p2
        return d, fromto.copy(), "rbound_fallback"


def min_pair_distance(model, data, a_geoms, b_geoms):
    best = None

    for g1 in a_geoms:
        for g2 in b_geoms:
            d, fromto, method = approx_geom_distance(model, data, g1, g2)
            item = {
                "distance": float(d),
                "geom1": geom_name(model, g1),
                "body1": body_name_of_geom(model, g1),
                "geom2": geom_name(model, g2),
                "body2": body_name_of_geom(model, g2),
                "method": method,
                "fromto": fromto,
            }

            if best is None or item["distance"] < best["distance"]:
                best = item

    if best is None:
        return {
            "distance": 999.0,
            "geom1": "",
            "body1": "",
            "geom2": "",
            "body2": "",
            "method": "empty_geom_set",
            "fromto": np.zeros(6),
        }

    return best


def classify_contact_pair(g1, g2, geom_sets):
    obj = set(geom_sets["object_geoms"])
    sup = set(geom_sets["support_geoms"])
    hand = set(geom_sets["hand_geoms"])
    fr3 = set(geom_sets["fr3_geoms"])

    a = int(g1)
    b = int(g2)

    def has(S1, S2):
        return (a in S1 and b in S2) or (a in S2 and b in S1)

    if has(hand, obj):
        return "hand_object"
    if has(hand, sup):
        return "hand_support"
    if has(fr3, obj):
        return "fr3_object"
    if has(obj, sup):
        return "object_support"
    if has(fr3, sup):
        return "fr3_support"
    return "other"


def contact_summary(model, data, geom_sets):
    out = {
        "ncon": int(data.ncon),
        "hand_object": 0,
        "hand_support": 0,
        "fr3_object": 0,
        "object_support": 0,
        "fr3_support": 0,
        "other": 0,
        "pairs": [],
    }

    for i in range(data.ncon):
        c = data.contact[i]
        cls = classify_contact_pair(int(c.geom1), int(c.geom2), geom_sets)
        out[cls] += 1
        out["pairs"].append({
            "class": cls,
            "geom1": geom_name(model, int(c.geom1)),
            "body1": body_name_of_geom(model, int(c.geom1)),
            "geom2": geom_name(model, int(c.geom2)),
            "body2": body_name_of_geom(model, int(c.geom2)),
            "dist": float(c.dist),
        })

    return out


def interp_qdict(qa, qb, s):
    keys = sorted(set(qa.keys()) | set(qb.keys()))
    out = {}
    for k in keys:
        a = float(qa.get(k, qb.get(k, 0.0)))
        b = float(qb.get(k, qa.get(k, 0.0)))
        out[k] = (1.0 - s) * a + s * b
    return out


def qdict_distance(qa, qb):
    vals = []
    for k in ARM_JOINTS:
        if k in qa and k in qb:
            vals.append(float(qa[k]) - float(qb[k]))
    return float(np.linalg.norm(vals))


def reset_and_apply(model, data, base_qpos, base_ctrl, qdict, hand_ctrl=None):
    data.qpos[:] = base_qpos
    data.ctrl[:] = base_ctrl
    mujoco.mj_forward(model, data)

    apply_qdict(model, data, qdict, also_ctrl=True)

    if hand_ctrl:
        apply_qdict(model, data, hand_ctrl, also_ctrl=True)

    mujoco.mj_forward(model, data)


def eval_state(model, data, base_qpos, base_ctrl, geom_sets, qdict, hand_ctrl=None):
    reset_and_apply(model, data, base_qpos, base_ctrl, qdict, hand_ctrl)

    return {
        "min_hand_support": min_pair_distance(model, data, geom_sets["hand_geoms"], geom_sets["support_geoms"]),
        "min_fr3_object": min_pair_distance(model, data, geom_sets["fr3_geoms"], geom_sets["object_geoms"]),
        "min_hand_object": min_pair_distance(model, data, geom_sets["hand_geoms"], geom_sets["object_geoms"]),
        "contacts": contact_summary(model, data, geom_sets),
    }


def eval_path(model, data, base_qpos, base_ctrl, geom_sets, qa, qb, path_name, samples, hand_ctrl=None):
    min_hand_support = None
    min_fr3_object = None
    min_hand_object = None
    worst_contacts = None
    points_brief = []

    for i in range(samples + 1):
        s = i / float(samples)
        q = interp_qdict(qa, qb, s)

        reset_and_apply(model, data, base_qpos, base_ctrl, q, hand_ctrl)

        hs = min_pair_distance(model, data, geom_sets["hand_geoms"], geom_sets["support_geoms"])
        fo = min_pair_distance(model, data, geom_sets["fr3_geoms"], geom_sets["object_geoms"])
        ho = min_pair_distance(model, data, geom_sets["hand_geoms"], geom_sets["object_geoms"])
        cs = contact_summary(model, data, geom_sets)

        if min_hand_support is None or hs["distance"] < min_hand_support["distance"]:
            min_hand_support = {**hs, "index": i, "s": s}
            worst_contacts = cs

        if min_fr3_object is None or fo["distance"] < min_fr3_object["distance"]:
            min_fr3_object = {**fo, "index": i, "s": s}

        if min_hand_object is None or ho["distance"] < min_hand_object["distance"]:
            min_hand_object = {**ho, "index": i, "s": s}

        if i in [0, samples // 2, samples]:
            points_brief.append({
                "index": i,
                "s": s,
                "min_hand_support_distance": hs["distance"],
                "min_fr3_object_distance": fo["distance"],
                "min_hand_object_distance": ho["distance"],
                "contacts": {
                    "ncon": cs["ncon"],
                    "hand_object": cs["hand_object"],
                    "hand_support": cs["hand_support"],
                    "fr3_object": cs["fr3_object"],
                    "object_support": cs["object_support"],
                    "fr3_support": cs["fr3_support"],
                },
            })

    return {
        "path": path_name,
        "num_samples": samples + 1,
        "min_hand_support": min_hand_support,
        "min_fr3_object": min_fr3_object,
        "min_hand_object": min_hand_object,
        "contacts_at_min_hand_support": worst_contacts,
        "points_brief": points_brief,
    }


def top_solutions(p2_data, target_name, top_n, require_success=True):
    arr = p2_data.get("results", {}).get(target_name, []) or []
    if require_success:
        arr = [x for x in arr if x.get("success", False)]
    arr = sorted(arr, key=lambda x: float(x.get("score", 1e9)))
    return arr[:top_n]


def min_margin_of_combo(pre, grasp, lift):
    vals = [
        float(pre.get("min_joint_limit_margin", 0.0)),
        float(grasp.get("min_joint_limit_margin", 0.0)),
        float(lift.get("min_joint_limit_margin", 0.0)),
    ]
    return min(vals)


def evaluate_combo(model, data, base_qpos, base_ctrl, geom_sets, q_current, hand_ctrl, combo, args):
    pre, grasp, lift = combo["pre"], combo["grasp"], combo["lift"]

    q_pre = pre["q"]
    q_grasp = grasp["q"]
    q_lift = lift["q"]

    path_current_pre = eval_path(
        model, data, base_qpos, base_ctrl, geom_sets,
        q_current, q_pre,
        "q_current_to_q_pre_open_hand",
        args.path_samples,
        hand_ctrl=None,
    )

    path_pre_grasp = eval_path(
        model, data, base_qpos, base_ctrl, geom_sets,
        q_pre, q_grasp,
        "q_pre_to_q_grasp_open_hand",
        args.path_samples,
        hand_ctrl=None,
    )

    path_grasp_lift = eval_path(
        model, data, base_qpos, base_ctrl, geom_sets,
        q_grasp, q_lift,
        "q_grasp_to_q_lift_closed_hand",
        args.path_samples,
        hand_ctrl=hand_ctrl,
    )

    static_grasp_open = eval_state(
        model, data, base_qpos, base_ctrl, geom_sets,
        q_grasp,
        hand_ctrl=None,
    )

    static_grasp_closed = eval_state(
        model, data, base_qpos, base_ctrl, geom_sets,
        q_grasp,
        hand_ctrl=hand_ctrl,
    )

    paths = [path_current_pre, path_pre_grasp, path_grasp_lift]

    min_hand_support = min(p["min_hand_support"]["distance"] for p in paths)
    min_fr3_object = min(p["min_fr3_object"]["distance"] for p in paths)
    min_hand_object_path = min(p["min_hand_object"]["distance"] for p in paths)

    grasp_hand_object = static_grasp_closed["min_hand_object"]["distance"]
    grasp_hand_support = static_grasp_closed["min_hand_support"]["distance"]
    grasp_fr3_object = static_grasp_closed["min_fr3_object"]["distance"]

    hard_reasons = []

    for p in paths:
        hs = p["min_hand_support"]["distance"]
        fo = p["min_fr3_object"]["distance"]

        if hs < args.min_hand_support_clearance:
            hard_reasons.append(
                f"{p['path']}: hand-support clearance {hs:.5f} < {args.min_hand_support_clearance:.5f}"
            )

        if fo < args.min_fr3_object_clearance:
            hard_reasons.append(
                f"{p['path']}: fr3-object clearance {fo:.5f} < {args.min_fr3_object_clearance:.5f}"
            )

    if grasp_hand_object > args.max_grasp_hand_object_distance:
        hard_reasons.append(
            f"q_grasp_closed: hand-object distance {grasp_hand_object:.5f} > {args.max_grasp_hand_object_distance:.5f}"
        )

    if grasp_hand_support < args.min_hand_support_clearance:
        hard_reasons.append(
            f"q_grasp_closed: hand-support clearance {grasp_hand_support:.5f} < {args.min_hand_support_clearance:.5f}"
        )

    if grasp_fr3_object < args.min_fr3_object_clearance:
        hard_reasons.append(
            f"q_grasp_closed: fr3-object clearance {grasp_fr3_object:.5f} < {args.min_fr3_object_clearance:.5f}"
        )

    combo_joint_margin = min_margin_of_combo(pre, grasp, lift)
    if combo_joint_margin < args.min_joint_margin:
        hard_reasons.append(
            f"combo joint margin {combo_joint_margin:.5f} < {args.min_joint_margin:.5f}"
        )

    smooth_cost = (
        qdict_distance(q_current, q_pre)
        + qdict_distance(q_pre, q_grasp)
        + qdict_distance(q_grasp, q_lift)
    )

    ik_score = (
        float(pre.get("score", 0.0))
        + float(grasp.get("score", 0.0))
        + float(lift.get("score", 0.0))
    )

    support_violation = max(0.0, args.min_hand_support_clearance - min_hand_support)
    fr3_violation = max(0.0, args.min_fr3_object_clearance - min_fr3_object)
    grasp_far = max(0.0, grasp_hand_object - args.max_grasp_hand_object_distance)
    margin_violation = max(0.0, args.min_joint_margin - combo_joint_margin)

    score = 0.0
    score += 100000.0 * support_violation
    score += 100000.0 * fr3_violation
    score += 50000.0 * grasp_far
    score += 10000.0 * margin_violation
    score += 1.0 * ik_score
    score += args.smooth_weight * smooth_cost
    score -= args.clearance_reward_weight * max(0.0, min_hand_support)
    score -= 0.5 * args.clearance_reward_weight * max(0.0, min_fr3_object)

    if hard_reasons:
        precheck_status = "FAIL_PRECHECK"
        score += 1_000_000.0
    else:
        precheck_status = "PASS_PRECHECK"

    return {
        "combo_id": combo["combo_id"],
        "pre_seed": pre.get("seed_name", ""),
        "grasp_seed": grasp.get("seed_name", ""),
        "lift_seed": lift.get("seed_name", ""),
        "pre_rank": combo["pre_rank"],
        "grasp_rank": combo["grasp_rank"],
        "lift_rank": combo["lift_rank"],
        "precheck_status": precheck_status,
        "hard_reasons": hard_reasons,
        "score": float(score),
        "ik_score_sum": float(ik_score),
        "smooth_cost": float(smooth_cost),
        "combo_min_joint_margin": float(combo_joint_margin),
        "min_path_hand_support_clearance": float(min_hand_support),
        "min_path_fr3_object_clearance": float(min_fr3_object),
        "min_path_hand_object_distance": float(min_hand_object_path),
        "static_grasp_closed_hand_object_distance": float(grasp_hand_object),
        "static_grasp_closed_hand_support_clearance": float(grasp_hand_support),
        "static_grasp_closed_fr3_object_clearance": float(grasp_fr3_object),
        "q_pre": q_pre,
        "q_grasp": q_grasp,
        "q_lift": q_lift,
        "path_precheck": paths,
        "static_precheck": {
            "q_grasp_open": static_grasp_open,
            "q_grasp_closed": static_grasp_closed,
        },
        "source_solutions": {
            "pre": {
                "seed_name": pre.get("seed_name"),
                "score": pre.get("score"),
                "pos_err": pre.get("pos_err"),
                "rot_err": pre.get("rot_err"),
                "min_joint_limit_margin": pre.get("min_joint_limit_margin"),
            },
            "grasp": {
                "seed_name": grasp.get("seed_name"),
                "score": grasp.get("score"),
                "pos_err": grasp.get("pos_err"),
                "rot_err": grasp.get("rot_err"),
                "min_joint_limit_margin": grasp.get("min_joint_limit_margin"),
            },
            "lift": {
                "seed_name": lift.get("seed_name"),
                "score": lift.get("score"),
                "pos_err": lift.get("pos_err"),
                "rot_err": lift.get("rot_err"),
                "min_joint_limit_margin": lift.get("min_joint_limit_margin"),
            },
        },
    }


def make_plan(best, p2_data, candidate_path, model_path, object_body, args):
    return {
        "format": "v4_12p3_pinocchio_mujoco_precheck_plan_debug",
        "meaning": "Precomputed q_pre/q_grasp/q_lift selected by Pinocchio IK plus MuJoCo collision precheck.",
        "model": str(model_path),
        "candidate": str(candidate_path),
        "object_body": object_body,
        "target_frame": p2_data.get("target_frame", "fr3_link7"),
        "precheck_status": best.get("precheck_status"),
        "score": best.get("score"),
        "selected_seeds": {
            "pre": best.get("pre_seed"),
            "grasp": best.get("grasp_seed"),
            "lift": best.get("lift_seed"),
        },
        "q_pre": best.get("q_pre"),
        "q_grasp": best.get("q_grasp"),
        "q_lift": best.get("q_lift"),
        "metrics": {
            "combo_min_joint_margin": best.get("combo_min_joint_margin"),
            "min_path_hand_support_clearance": best.get("min_path_hand_support_clearance"),
            "min_path_fr3_object_clearance": best.get("min_path_fr3_object_clearance"),
            "static_grasp_closed_hand_object_distance": best.get("static_grasp_closed_hand_object_distance"),
            "static_grasp_closed_hand_support_clearance": best.get("static_grasp_closed_hand_support_clearance"),
            "smooth_cost": best.get("smooth_cost"),
            "hard_reasons": best.get("hard_reasons"),
        },
        "args": vars(args),
        "note": "This plan is not yet executed. A later runner must accept these precomputed q states explicitly.",
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p2-json", required=True)
    ap.add_argument("--model", default="")
    ap.add_argument("--candidate", default="")
    ap.add_argument("--object-body", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--best-plan-out", default="")

    ap.add_argument("--top-per-target", type=int, default=8)
    ap.add_argument("--max-combos", type=int, default=512)
    ap.add_argument("--path-samples", type=int, default=40)

    ap.add_argument("--min-hand-support-clearance", type=float, default=0.005)
    ap.add_argument("--min-fr3-object-clearance", type=float, default=0.005)
    ap.add_argument("--max-grasp-hand-object-distance", type=float, default=0.030)
    ap.add_argument("--min-joint-margin", type=float, default=0.001)

    ap.add_argument("--smooth-weight", type=float, default=0.15)
    ap.add_argument("--clearance-reward-weight", type=float, default=10.0)

    ap.add_argument("--require-ik-success", action="store_true", default=True)

    args = ap.parse_args()

    p2_path = resolve_path(args.p2_json)
    p2_data = load_json(p2_path)

    model_path = resolve_path(args.model or p2_data["model"])
    candidate_path = resolve_path(args.candidate or p2_data["candidate"])

    candidate = load_json(candidate_path)
    object_body = args.object_body or p2_data.get("object_body", "") or (candidate.get("object") or {}).get("body", "")
    if not object_body:
        raise RuntimeError("cannot infer object_body; pass --object-body")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    base_qpos = data.qpos.copy()
    base_ctrl = data.ctrl.copy()

    q_current = current_arm_qdict(model, data)
    hand_ctrl = extract_hand_ctrl(candidate)
    geom_sets = collect_geom_sets(model, object_body)

    pre_sols = top_solutions(p2_data, "pre", args.top_per_target, args.require_ik_success)
    grasp_sols = top_solutions(p2_data, "grasp", args.top_per_target, args.require_ik_success)
    lift_sols = top_solutions(p2_data, "lift", args.top_per_target, args.require_ik_success)

    combos = []
    combo_id = 0

    for (pi, pre), (gi, grasp), (li, lift) in itertools.product(
        enumerate(pre_sols, 1),
        enumerate(grasp_sols, 1),
        enumerate(lift_sols, 1),
    ):
        combo_id += 1
        combos.append({
            "combo_id": combo_id,
            "pre": pre,
            "grasp": grasp,
            "lift": lift,
            "pre_rank": pi,
            "grasp_rank": gi,
            "lift_rank": li,
        })

    combos = combos[:args.max_combos]

    print("\n========== V4.12P3 PINOCCHIO IK + MUJOCO COLLISION PRECHECK ==========")
    print("p2_json      :", p2_path)
    print("model        :", model_path)
    print("candidate    :", candidate_path)
    print("object_body  :", object_body)
    print("top_per_target:", args.top_per_target)
    print("num combos   :", len(combos))
    print("path_samples :", args.path_samples)
    print("thresholds   :")
    print("  min hand-support clearance:", args.min_hand_support_clearance)
    print("  min fr3-object clearance  :", args.min_fr3_object_clearance)
    print("  max grasp hand-object dist :", args.max_grasp_hand_object_distance)
    print("  min joint margin           :", args.min_joint_margin)
    print("======================================================================\n")

    results = []

    for idx, combo in enumerate(combos, 1):
        item = evaluate_combo(
            model, data, base_qpos, base_ctrl,
            geom_sets,
            q_current,
            hand_ctrl,
            combo,
            args,
        )
        results.append(item)

        print(
            f"[{idx:03d}/{len(combos):03d}] "
            f"status={item['precheck_status']} "
            f"score={item['score']:.3f} "
            f"pre={item['pre_seed']} "
            f"grasp={item['grasp_seed']} "
            f"lift={item['lift_seed']} "
            f"HS={item['min_path_hand_support_clearance']:+.5f} "
            f"FO={item['min_path_fr3_object_clearance']:+.5f} "
            f"GO={item['static_grasp_closed_hand_object_distance']:+.5f} "
            f"margin={item['combo_min_joint_margin']:+.5f} "
            f"smooth={item['smooth_cost']:.3f}"
        )

        if item["hard_reasons"]:
            for r in item["hard_reasons"][:3]:
                print("    -", r)

    ranked = sorted(results, key=lambda x: float(x.get("score", 1e18)))
    passed = [r for r in ranked if r.get("precheck_status") == "PASS_PRECHECK"]

    best_available = ranked[0] if ranked else None
    best_pass = passed[0] if passed else None

    out = {
        "format": "v4_12p3_pinocchio_ik_plus_mujoco_collision_precheck_debug",
        "meaning": "Score combinations of P2 Pinocchio IK solutions by MuJoCo path collision clearance.",
        "p2_json": str(p2_path),
        "model": str(model_path),
        "candidate": str(candidate_path),
        "object_body": object_body,
        "args": vars(args),
        "geom_counts": {
            "object_geoms": len(geom_sets["object_geoms"]),
            "support_geoms": len(geom_sets["support_geoms"]),
            "hand_geoms": len(geom_sets["hand_geoms"]),
            "fr3_geoms": len(geom_sets["fr3_geoms"]),
        },
        "num_combos": len(combos),
        "num_pass": len(passed),
        "best_pass": best_pass,
        "best_available": best_available,
        "ranked": ranked,
    }

    save_json(args.out, out)

    if args.best_plan_out:
        if best_pass is not None:
            plan = make_plan(best_pass, p2_data, candidate_path, model_path, object_body, args)
        elif best_available is not None:
            plan = make_plan(best_available, p2_data, candidate_path, model_path, object_body, args)
        else:
            plan = {
                "format": "v4_12p3_empty_plan_debug",
                "precheck_status": "NO_COMBO",
                "note": "No IK solution combination was evaluated.",
            }
        save_json(args.best_plan_out, plan)

    print("\n========== V4.12P3 SUMMARY ==========")
    print("num_combos:", len(combos))
    print("num_pass  :", len(passed))
    print("out       :", resolve_path(args.out))
    if args.best_plan_out:
        print("best_plan :", resolve_path(args.best_plan_out))

    for key, b in [("best_pass", best_pass), ("best_available", best_available)]:
        print(f"\n[{key}]")
        if b is None:
            print("  None")
            continue

        print("  status:", b["precheck_status"])
        print("  score :", b["score"])
        print("  seeds :", b["pre_seed"], "->", b["grasp_seed"], "->", b["lift_seed"])
        print("  min_path_hand_support_clearance:", b["min_path_hand_support_clearance"])
        print("  min_path_fr3_object_clearance  :", b["min_path_fr3_object_clearance"])
        print("  static_grasp_closed_hand_object:", b["static_grasp_closed_hand_object_distance"])
        print("  static_grasp_closed_hand_support:", b["static_grasp_closed_hand_support_clearance"])
        print("  combo_min_joint_margin:", b["combo_min_joint_margin"])
        print("  smooth_cost:", b["smooth_cost"])
        print("  reasons:")
        for r in b["hard_reasons"][:10]:
            print("   -", r)

    print("=====================================\n")


if __name__ == "__main__":
    main()
