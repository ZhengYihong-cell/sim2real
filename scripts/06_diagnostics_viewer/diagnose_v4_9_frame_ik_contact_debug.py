#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import math
import re
import numpy as np
import mujoco


PROJECT = Path.home() / "Projects/o7_mujoco_sim"


HAND_TOKENS = ["thumb", "index", "middle", "ring", "pinky", "palm", "hand_base", "hand"]
FINGER_GROUPS = ["thumb", "index", "middle", "ring", "pinky"]
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


def site_id(model, name):
    return mj_id(model, mujoco.mjtObj.mjOBJ_SITE, name)


def geom_id(model, name):
    return mj_id(model, mujoco.mjtObj.mjOBJ_GEOM, name)


def joint_id(model, name):
    return mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def actuator_id(model, name):
    return mj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    return v / max(float(np.linalg.norm(v)), eps)


def T_from_Rp(R, p):
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(p, dtype=float).reshape(3)
    return T


def T_inv(T):
    T = np.asarray(T, dtype=float)
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def rot_angle(R):
    R = np.asarray(R, dtype=float).reshape(3, 3)
    c = (np.trace(R) - 1.0) * 0.5
    c = float(np.clip(c, -1.0, 1.0))
    return float(math.acos(c))


def frame_T(model, data, name):
    bid = body_id(model, name)
    if bid >= 0:
        return T_from_Rp(data.xmat[bid].reshape(3, 3), data.xpos[bid]), "body"

    sid = site_id(model, name)
    if sid >= 0:
        return T_from_Rp(data.site_xmat[sid].reshape(3, 3), data.site_xpos[sid]), "site"

    return None, None


def frame_error(T_cur, T_target):
    ep = T_target[:3, 3] - T_cur[:3, 3]
    er = rot_angle(T_cur[:3, :3].T @ T_target[:3, :3])
    return {
        "pos_err_norm": float(np.linalg.norm(ep)),
        "rot_err_angle": float(er),
        "pos_err": ep,
    }


def set_joint_qpos(model, data, name, value):
    jid = joint_id(model, name)
    if jid < 0:
        return False

    adr = int(model.jnt_qposadr[jid])
    jtype = int(model.jnt_type[jid])

    # hinge / slide
    if jtype in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
        data.qpos[adr] = float(value)
        return True

    return False


def set_actuator_ctrl(model, data, name, value):
    names = [name, f"{name}_pos", f"{name}_act", f"{name}_ctrl"]
    for nm in names:
        aid = actuator_id(model, nm)
        if aid >= 0:
            data.ctrl[aid] = float(value)
            return True
    return False


def apply_qdict(model, data, qdict):
    if not qdict:
        return {"set_joints": [], "set_ctrls": [], "missing": []}

    set_joints = []
    set_ctrls = []
    missing = []

    for k, v in qdict.items():
        ok_j = set_joint_qpos(model, data, k, v)
        ok_c = set_actuator_ctrl(model, data, k, v)

        if ok_j:
            set_joints.append(k)
        if ok_c:
            set_ctrls.append(k)
        if not ok_j and not ok_c:
            missing.append(k)

    return {
        "set_joints": set_joints,
        "set_ctrls": set_ctrls,
        "missing": missing,
    }


def extract_hand_ctrl(candidate):
    hand = candidate.get("hand", {}) or {}
    ctrl = hand.get("o7_active_ctrl", None)
    if isinstance(ctrl, dict):
        return ctrl
    return {}


def body_is_descendant(model, bid, root_bid):
    if bid < 0 or root_bid < 0:
        return False
    cur = int(bid)
    while cur > 0:
        if cur == int(root_bid):
            return True
        cur = int(model.body_parentid[cur])
    return cur == int(root_bid)


def collect_geoms(model, object_body):
    object_bid = body_id(model, object_body)

    object_geoms = []
    support_geoms = []
    hand_geoms = []
    fr3_geoms = []
    all_named_geoms = []

    for gid in range(model.ngeom):
        name = mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"
        lname = name.lower()
        bid = int(model.geom_bodyid[gid])
        bname = mj_name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""

        all_named_geoms.append((gid, name, bname))

        if body_is_descendant(model, bid, object_bid):
            object_geoms.append(gid)
            continue

        if any(tok in lname for tok in SUPPORT_TOKENS):
            support_geoms.append(gid)
            continue

        if any(tok in lname for tok in HAND_TOKENS) and not any(tok in lname for tok in SUPPORT_TOKENS):
            hand_geoms.append(gid)
            continue

        if bname.lower().startswith("fr3") or any(tok in lname for tok in FR3_TOKENS):
            fr3_geoms.append(gid)

    return {
        "object_geoms": object_geoms,
        "support_geoms": support_geoms,
        "hand_geoms": hand_geoms,
        "fr3_geoms": fr3_geoms,
        "all_named_geoms": all_named_geoms,
    }


def geom_name(model, gid):
    return mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"


def body_name_of_geom(model, gid):
    bid = int(model.geom_bodyid[gid])
    return mj_name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or f"body_{bid}"


def approx_geom_distance(model, data, g1, g2):
    # 优先用 MuJoCo 的 geomDistance；若 mesh 类型不支持，则退化为 bounding-sphere 估计。
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


def closest_pairs(model, data, a_geoms, b_geoms, limit=10):
    pairs = []
    for g1 in a_geoms:
        for g2 in b_geoms:
            d, fromto, method = approx_geom_distance(model, data, g1, g2)
            pairs.append({
                "distance": d,
                "geom1": geom_name(model, g1),
                "body1": body_name_of_geom(model, g1),
                "geom2": geom_name(model, g2),
                "body2": body_name_of_geom(model, g2),
                "method": method,
                "fromto": fromto,
            })
    pairs.sort(key=lambda x: x["distance"])
    return pairs[:limit]


def group_hand_geoms(model, hand_geoms):
    groups = {g: [] for g in FINGER_GROUPS}
    groups["other_hand"] = []

    for gid in hand_geoms:
        name = geom_name(model, gid).lower()
        placed = False
        for g in FINGER_GROUPS:
            if g in name:
                groups[g].append(gid)
                placed = True
                break
        if not placed:
            groups["other_hand"].append(gid)

    return groups


def classify_contact_pair(g1, g2, geom_sets):
    obj = set(geom_sets["object_geoms"])
    sup = set(geom_sets["support_geoms"])
    hand = set(geom_sets["hand_geoms"])
    fr3 = set(geom_sets["fr3_geoms"])

    a, b = int(g1), int(g2)

    def has(A, B, S1, S2):
        return (A in S1 and B in S2) or (A in S2 and B in S1)

    if has(a, b, hand, obj):
        return "hand_object"
    if has(a, b, hand, sup):
        return "hand_support"
    if has(a, b, fr3, obj):
        return "fr3_object"
    if has(a, b, obj, sup):
        return "object_support"
    if has(a, b, fr3, sup):
        return "fr3_support"
    return "other"


def contact_summary(model, data, geom_sets):
    counts = {
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
        g1 = int(c.geom1)
        g2 = int(c.geom2)
        cls = classify_contact_pair(g1, g2, geom_sets)
        counts[cls] += 1
        counts["pairs"].append({
            "class": cls,
            "geom1": geom_name(model, g1),
            "body1": body_name_of_geom(model, g1),
            "geom2": geom_name(model, g2),
            "body2": body_name_of_geom(model, g2),
            "dist": float(c.dist),
        })

    return counts


def scenario_diagnosis(model, data, scenario_name, target_frame_name, T_world_target, geom_sets):
    T_cur, frame_type = frame_T(model, data, target_frame_name)
    if T_cur is None:
        ferr = None
    else:
        ferr = frame_error(T_cur, T_world_target)

    contacts = contact_summary(model, data, geom_sets)

    hand_groups = group_hand_geoms(model, geom_sets["hand_geoms"])

    group_to_object = {}
    group_to_support = {}

    for group, geoms in hand_groups.items():
        if geoms:
            group_to_object[group] = closest_pairs(model, data, geoms, geom_sets["object_geoms"], limit=3)
            group_to_support[group] = closest_pairs(model, data, geoms, geom_sets["support_geoms"], limit=3)

    all_hand_to_object = closest_pairs(model, data, geom_sets["hand_geoms"], geom_sets["object_geoms"], limit=12)
    all_hand_to_support = closest_pairs(model, data, geom_sets["hand_geoms"], geom_sets["support_geoms"], limit=12)

    return {
        "scenario": scenario_name,
        "target_frame": target_frame_name,
        "target_frame_type_found": frame_type,
        "frame_error_to_target": ferr,
        "contact_summary": contacts,
        "closest_hand_to_object": all_hand_to_object,
        "closest_hand_to_support": all_hand_to_support,
        "finger_group_to_object": group_to_object,
        "finger_group_to_support": group_to_support,
    }


def print_scenario_brief(diag):
    print(f"\n========== SCENARIO: {diag['scenario']} ==========")
    ferr = diag.get("frame_error_to_target")
    if ferr is None:
        print("frame_error: target frame not found")
    else:
        print(f"frame pos_err = {ferr['pos_err_norm']:.5f} m")
        print(f"frame rot_err = {ferr['rot_err_angle']:.5f} rad")
        print(f"frame pos_err xyz = {np.asarray(ferr['pos_err'])}")

    c = diag["contact_summary"]
    print(
        "contacts:",
        f"ncon={c['ncon']}",
        f"hand_object={c['hand_object']}",
        f"hand_support={c['hand_support']}",
        f"fr3_object={c['fr3_object']}",
        f"object_support={c['object_support']}",
    )

    ho = diag["closest_hand_to_object"][:5]
    hs = diag["closest_hand_to_support"][:5]

    print("\nclosest hand -> object:")
    for p in ho:
        print(f"  d={p['distance']:+.5f}  {p['body1']}/{p['geom1']}  <->  {p['body2']}/{p['geom2']}  [{p['method']}]")

    print("\nclosest hand -> support:")
    for p in hs:
        print(f"  d={p['distance']:+.5f}  {p['body1']}/{p['geom1']}  <->  {p['body2']}/{p['geom2']}  [{p['method']}]")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--runner-json", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--target-frame", default="")
    ap.add_argument("--object-body", default="")
    ap.add_argument("--settle-steps", type=int, default=0)
    args = ap.parse_args()

    model_path = resolve_path(args.model)
    candidate_path = resolve_path(args.candidate)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    candidate = load_json(candidate_path)

    object_body = args.object_body or candidate.get("object", {}).get("body", "")
    target_frame = args.target_frame or candidate.get("target", {}).get("body", "fr3_link7")
    T_object_target = np.asarray(candidate["target"]["T_object_target"], dtype=float)

    if not object_body:
        raise RuntimeError("cannot infer object_body; please pass --object-body")

    T_world_object, object_frame_type = frame_T(model, data, object_body)
    if T_world_object is None:
        raise RuntimeError(f"cannot find object body/site: {object_body}")

    T_world_target = T_world_object @ T_object_target

    geom_sets = collect_geoms(model, object_body)

    runner = None
    runner_result = None
    if args.runner_json:
        runner = load_json(args.runner_json)
        if runner.get("results"):
            runner_result = runner["results"][0]

    q_pre = (runner_result or {}).get("q_pre", {})
    q_grasp = (runner_result or {}).get("q_grasp", {})
    q_lift = (runner_result or {}).get("q_lift", {})
    hand_ctrl = extract_hand_ctrl(candidate)

    base_qpos = data.qpos.copy()
    base_ctrl = data.ctrl.copy()

    def reset_to_base():
        data.qpos[:] = base_qpos
        data.ctrl[:] = base_ctrl
        mujoco.mj_forward(model, data)

    def apply_state(name, qdict=None, close_hand=False):
        reset_to_base()
        set_info = {"set_joints": [], "set_ctrls": [], "missing": []}

        if qdict:
            set_info = apply_qdict(model, data, qdict)

        hand_info = None
        if close_hand and hand_ctrl:
            hand_info = apply_qdict(model, data, hand_ctrl)

        mujoco.mj_forward(model, data)

        if args.settle_steps > 0:
            for _ in range(args.settle_steps):
                mujoco.mj_step(model, data)

        diag = scenario_diagnosis(model, data, name, target_frame, T_world_target, geom_sets)
        diag["set_info"] = set_info
        diag["hand_close_info"] = hand_info
        return diag

    scenarios = []
    scenarios.append(apply_state("model_default", None, close_hand=False))

    if q_pre:
        scenarios.append(apply_state("q_pre_arm_only", q_pre, close_hand=False))
        scenarios.append(apply_state("q_pre_with_candidate_hand_ctrl", q_pre, close_hand=True))

    if q_grasp:
        scenarios.append(apply_state("q_grasp_arm_only", q_grasp, close_hand=False))
        scenarios.append(apply_state("q_grasp_with_candidate_hand_ctrl", q_grasp, close_hand=True))

    if q_lift:
        scenarios.append(apply_state("q_lift_arm_only", q_lift, close_hand=False))

    # frame offset diagnostics
    reset_to_base()
    frame_names_to_check = [
        "fr3_link7",
        "fr3_link8",
        "hand_base_link",
        "dataset_hand_base_debug",
        "o7_hand_base",
        "palm",
    ]

    frames = {}
    for nm in frame_names_to_check:
        T, typ = frame_T(model, data, nm)
        if T is not None:
            frames[nm] = {
                "type": typ,
                "pos": T[:3, 3],
                "R": T[:3, :3],
            }

    relative_frames = {}
    if "fr3_link7" in frames:
        T7 = T_from_Rp(frames["fr3_link7"]["R"], frames["fr3_link7"]["pos"])
        for nm, f in frames.items():
            if nm == "fr3_link7":
                continue
            Tn = T_from_Rp(f["R"], f["pos"])
            T_7_n = T_inv(T7) @ Tn
            relative_frames[f"T_fr3_link7_to_{nm}"] = {
                "translation": T_7_n[:3, 3],
                "R": T_7_n[:3, :3],
                "rot_angle_from_identity": rot_angle(T_7_n[:3, :3]),
            }

    runner_target_compare = None
    if runner_result:
        rt = np.asarray(runner_result.get("target_pos", [np.nan, np.nan, np.nan]), dtype=float)
        if np.all(np.isfinite(rt)):
            runner_target_compare = {
                "runner_target_pos": rt,
                "computed_target_pos": T_world_target[:3, 3],
                "pos_diff_norm": float(np.linalg.norm(rt - T_world_target[:3, 3])),
                "pos_diff": rt - T_world_target[:3, 3],
            }

    out = {
        "format": "v4_9_frame_ik_contact_diagnosis_debug",
        "model": str(model_path),
        "candidate": str(candidate_path),
        "runner_json": str(resolve_path(args.runner_json)) if args.runner_json else "",
        "object_body": object_body,
        "object_frame_type": object_frame_type,
        "target_frame": target_frame,
        "T_object_target": T_object_target,
        "T_world_object": T_world_object,
        "T_world_target": T_world_target,
        "target_world_pos": T_world_target[:3, 3],
        "candidate_hand_ctrl": hand_ctrl,
        "runner_ik_info": {
            "ik_pre_info": (runner_result or {}).get("ik_pre_info"),
            "ik_grasp_info": (runner_result or {}).get("ik_grasp_info"),
            "ik_lift_info": (runner_result or {}).get("ik_lift_info"),
            "runner_success": (runner_result or {}).get("success"),
            "final_counts": (runner_result or {}).get("final_counts"),
            "final_rise": (runner_result or {}).get("final_rise"),
            "max_hand_object_close": (runner_result or {}).get("max_hand_object_close"),
            "max_hand_object_hold": (runner_result or {}).get("max_hand_object_hold"),
            "max_hand_object_lift": (runner_result or {}).get("max_hand_object_lift"),
        },
        "runner_target_compare": runner_target_compare,
        "available_frames": frames,
        "relative_frames_from_fr3_link7": relative_frames,
        "geom_set_counts": {
            "object_geoms": len(geom_sets["object_geoms"]),
            "support_geoms": len(geom_sets["support_geoms"]),
            "hand_geoms": len(geom_sets["hand_geoms"]),
            "fr3_geoms": len(geom_sets["fr3_geoms"]),
        },
        "geom_sets_preview": {
            "object_geoms": [geom_name(model, g) for g in geom_sets["object_geoms"][:50]],
            "support_geoms": [geom_name(model, g) for g in geom_sets["support_geoms"][:50]],
            "hand_geoms": [geom_name(model, g) for g in geom_sets["hand_geoms"][:80]],
            "fr3_geoms": [geom_name(model, g) for g in geom_sets["fr3_geoms"][:80]],
        },
        "scenarios": scenarios,
    }

    save_json(args.out, out)

    print("\n========== V4.9A FRAME / IK / CONTACT DIAGNOSIS ==========")
    print("model        :", model_path)
    print("candidate    :", candidate_path)
    print("runner_json  :", resolve_path(args.runner_json) if args.runner_json else "")
    print("object_body  :", object_body)
    print("target_frame :", target_frame)
    print("target_world_pos:", T_world_target[:3, 3])
    print("geom counts  :", out["geom_set_counts"])

    if runner_target_compare:
        print("\n[CHECK] candidate T_object_target vs runner target_pos")
        print("computed target:", runner_target_compare["computed_target_pos"])
        print("runner target  :", runner_target_compare["runner_target_pos"])
        print("diff norm      :", runner_target_compare["pos_diff_norm"])

    print("\n[RUNNER IK INFO]")
    rik = out["runner_ik_info"]
    print("ik_pre_info   :", rik["ik_pre_info"])
    print("ik_grasp_info :", rik["ik_grasp_info"])
    print("ik_lift_info  :", rik["ik_lift_info"])
    print("final_counts  :", rik["final_counts"])
    print("final_rise    :", rik["final_rise"])

    print("\n[RELATIVE FRAMES FROM fr3_link7]")
    for k, v in relative_frames.items():
        print(k)
        print("  translation:", np.asarray(v["translation"]))
        print("  rot_angle  :", v["rot_angle_from_identity"])

    for sc in scenarios:
        print_scenario_brief(sc)

    print("\n[SAVED]", resolve_path(args.out))
    print("==========================================================\n")


if __name__ == "__main__":
    main()
