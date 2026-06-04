#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import os
import subprocess
import numpy as np
import mujoco


PROJECT = Path.home() / "Projects/o7_mujoco_sim"
RUN_CLEAN = PROJECT / "run_mujoco_clean.sh"
RUNNER = PROJECT / "scripts/run_fr3_o7_candidate_grasp_site_servo_debug.py"

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
        lname = (gname + " " + bname).lower()
        bid = int(model.geom_bodyid[gid])

        if body_is_descendant(model, bid, object_bid):
            object_geoms.append(gid)
            continue

        if any(tok in lname for tok in SUPPORT_TOKENS):
            support_geoms.append(gid)
            continue

        if any(tok in lname for tok in HAND_TOKENS):
            hand_geoms.append(gid)
            continue

        if any(tok in lname for tok in FR3_TOKENS):
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


def apply_qdict(model, data, qdict):
    set_joints = []
    set_ctrls = []
    missing = []

    for k, v in (qdict or {}).items():
        ok_j = set_joint_qpos(model, data, k, v)
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
        return ctrl
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


def current_arm_qdict(model, data):
    out = {}
    for j in ARM_JOINTS:
        v = get_joint_value(model, data, j)
        if v is not None:
            out[j] = v
    return out


def interp_qdict(qa, qb, s):
    keys = sorted(set(qa.keys()) | set(qb.keys()))
    out = {}
    for k in keys:
        a = float(qa.get(k, qb.get(k, 0.0)))
        b = float(qb.get(k, qa.get(k, 0.0)))
        out[k] = (1.0 - s) * a + s * b
    return out


def eval_static_state(model, data, base_qpos, base_ctrl, geom_sets, qdict, hand_ctrl=None):
    data.qpos[:] = base_qpos
    data.ctrl[:] = base_ctrl
    mujoco.mj_forward(model, data)

    apply_qdict(model, data, qdict)

    if hand_ctrl:
        apply_qdict(model, data, hand_ctrl)

    mujoco.mj_forward(model, data)

    return {
        "min_hand_support": min_pair_distance(model, data, geom_sets["hand_geoms"], geom_sets["support_geoms"]),
        "min_fr3_object": min_pair_distance(model, data, geom_sets["fr3_geoms"], geom_sets["object_geoms"]),
        "min_hand_object": min_pair_distance(model, data, geom_sets["hand_geoms"], geom_sets["object_geoms"]),
        "contacts": contact_summary(model, data, geom_sets),
    }


def eval_path(model, data, base_qpos, base_ctrl, geom_sets, qa, qb, path_name, samples, hand_ctrl=None):
    points = []
    min_hand_support = None
    min_fr3_object = None
    min_hand_object = None
    first_hand_support_violation = None
    first_fr3_object_collision = None

    for i in range(samples + 1):
        s = i / float(samples)
        q = interp_qdict(qa, qb, s)

        data.qpos[:] = base_qpos
        data.ctrl[:] = base_ctrl
        mujoco.mj_forward(model, data)

        apply_qdict(model, data, q)

        if hand_ctrl:
            apply_qdict(model, data, hand_ctrl)

        mujoco.mj_forward(model, data)

        hs = min_pair_distance(model, data, geom_sets["hand_geoms"], geom_sets["support_geoms"])
        fo = min_pair_distance(model, data, geom_sets["fr3_geoms"], geom_sets["object_geoms"])
        ho = min_pair_distance(model, data, geom_sets["hand_geoms"], geom_sets["object_geoms"])
        cs = contact_summary(model, data, geom_sets)

        pt = {
            "path": path_name,
            "index": i,
            "s": s,
            "q": q,
            "min_hand_support": hs,
            "min_fr3_object": fo,
            "min_hand_object": ho,
            "contacts": cs,
        }
        points.append(pt)

        if min_hand_support is None or hs["distance"] < min_hand_support["distance"]:
            min_hand_support = {**hs, "index": i, "s": s}
        if min_fr3_object is None or fo["distance"] < min_fr3_object["distance"]:
            min_fr3_object = {**fo, "index": i, "s": s}
        if min_hand_object is None or ho["distance"] < min_hand_object["distance"]:
            min_hand_object = {**ho, "index": i, "s": s}

    return {
        "path": path_name,
        "num_samples": samples + 1,
        "min_hand_support": min_hand_support,
        "min_fr3_object": min_fr3_object,
        "min_hand_object": min_hand_object,
        "points": points,
    }


def run_runner_if_needed(args, runner_json):
    runner_json = resolve_path(runner_json)
    if args.runner_json:
        return resolve_path(args.runner_json)

    cmd = [
        str(RUN_CLEAN),
        str(RUNNER),
        "--model", str(resolve_path(args.model)),
        "--candidate", str(resolve_path(args.candidate)),
        "--spawn-source", args.spawn_source,
        "--out", str(runner_json),
        "--approach-mode", args.approach_mode,
        "--pregrasp-z", str(args.pregrasp_z),
        "--pregrasp-approach-dist", str(args.pregrasp_approach_dist),
        "--move-duration", str(args.move_duration),
        "--descend-duration", str(args.descend_duration),
        "--close-duration", str(args.close_duration),
        "--hold-duration", str(args.hold_duration),
        "--lift-z", str(args.lift_z),
        "--lift-duration", str(args.lift_duration),
        "--lift-mode", args.lift_mode,
        "--min-final-hand-object", str(args.min_final_hand_object),
        "--log-dt", str(args.log_dt),
    ]

    log_path = runner_json.with_suffix(".txt")
    with open(log_path, "w") as lf:
        ret = subprocess.run(
            cmd,
            cwd=PROJECT,
            stdout=lf,
            stderr=subprocess.STDOUT,
            env={**os.environ, "PYTHONUNBUFFERED": "1"},
        )

    if ret.returncode != 0:
        raise RuntimeError(f"runner failed, see {log_path}")

    return runner_json


def build_status(path_infos, static_infos, args):
    reasons = []

    for p in path_infos:
        hs = p["min_hand_support"]["distance"]
        fo = p["min_fr3_object"]["distance"]

        if hs < args.min_hand_support_clearance:
            reasons.append(
                f"{p['path']}: hand-support clearance {hs:.5f} < {args.min_hand_support_clearance:.5f}"
            )

        if fo < args.min_fr3_object_clearance:
            reasons.append(
                f"{p['path']}: fr3-object clearance {fo:.5f} < {args.min_fr3_object_clearance:.5f}"
            )

    grasp = static_infos.get("q_grasp_closed", {})
    if grasp:
        ho = grasp["min_hand_object"]["distance"]
        hs = grasp["min_hand_support"]["distance"]

        if ho > args.max_grasp_hand_object_distance:
            reasons.append(
                f"q_grasp_closed: hand-object distance {ho:.5f} > {args.max_grasp_hand_object_distance:.5f}"
            )

        if hs < args.min_hand_support_clearance:
            reasons.append(
                f"q_grasp_closed: hand-support clearance {hs:.5f} < {args.min_hand_support_clearance:.5f}"
            )

    if reasons:
        return "FAIL_PRECHECK", reasons

    return "PASS_PRECHECK", []


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--runner-json", default="")
    ap.add_argument("--object-body", default="")

    ap.add_argument("--path-samples", type=int, default=30)
    ap.add_argument("--min-hand-support-clearance", type=float, default=0.005)
    ap.add_argument("--min-fr3-object-clearance", type=float, default=0.000)
    ap.add_argument("--max-grasp-hand-object-distance", type=float, default=0.030)

    ap.add_argument("--spawn-source", default="model")
    ap.add_argument("--approach-mode", default="world_z")
    ap.add_argument("--pregrasp-z", type=float, default=0.085)
    ap.add_argument("--pregrasp-approach-dist", type=float, default=0.075)
    ap.add_argument("--move-duration", type=float, default=1.50)
    ap.add_argument("--descend-duration", type=float, default=0.90)
    ap.add_argument("--close-duration", type=float, default=1.05)
    ap.add_argument("--hold-duration", type=float, default=0.75)
    ap.add_argument("--lift-z", type=float, default=0.120)
    ap.add_argument("--lift-duration", type=float, default=2.00)
    ap.add_argument("--lift-mode", default="q_interp")
    ap.add_argument("--min-final-hand-object", type=int, default=2)
    ap.add_argument("--log-dt", type=float, default=0.25)

    args = ap.parse_args()

    out_path = resolve_path(args.out)
    runner_json = out_path.parent / (out_path.stem + "_runner.json")
    runner_json = run_runner_if_needed(args, runner_json)

    candidate = load_json(args.candidate)
    runner_data = load_json(runner_json)
    result = runner_data["results"][0]

    object_body = args.object_body or candidate.get("object", {}).get("body", "")
    if not object_body:
        raise RuntimeError("cannot infer object body; pass --object-body")

    model = mujoco.MjModel.from_xml_path(str(resolve_path(args.model)))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    base_qpos = data.qpos.copy()
    base_ctrl = data.ctrl.copy()

    geom_sets = collect_geom_sets(model, object_body)
    q_current = current_arm_qdict(model, data)
    q_pre = result.get("q_pre", {}) or {}
    q_grasp = result.get("q_grasp", {}) or {}
    q_lift = result.get("q_lift", {}) or {}
    hand_ctrl = extract_hand_ctrl(candidate)

    path_infos = []
    static_infos = {}

    if q_pre:
        path_infos.append(
            eval_path(
                model, data, base_qpos, base_ctrl, geom_sets,
                q_current, q_pre,
                "q_current_to_q_pre_open_hand",
                args.path_samples,
                hand_ctrl=None,
            )
        )

    if q_pre and q_grasp:
        path_infos.append(
            eval_path(
                model, data, base_qpos, base_ctrl, geom_sets,
                q_pre, q_grasp,
                "q_pre_to_q_grasp_open_hand",
                args.path_samples,
                hand_ctrl=None,
            )
        )

    if q_grasp:
        static_infos["q_grasp_open"] = eval_static_state(
            model, data, base_qpos, base_ctrl, geom_sets,
            q_grasp,
            hand_ctrl=None,
        )
        static_infos["q_grasp_closed"] = eval_static_state(
            model, data, base_qpos, base_ctrl, geom_sets,
            q_grasp,
            hand_ctrl=hand_ctrl,
        )

    if q_lift and q_grasp:
        path_infos.append(
            eval_path(
                model, data, base_qpos, base_ctrl, geom_sets,
                q_grasp, q_lift,
                "q_grasp_to_q_lift_closed_hand",
                args.path_samples,
                hand_ctrl=hand_ctrl,
            )
        )

    status, reasons = build_status(path_infos, static_infos, args)

    out = {
        "format": "v4_12a_mujoco_fk_collision_baseline_debug",
        "meaning": "Precheck candidate using runner IK result plus MuJoCo FK and geomDistance path clearance.",
        "model": str(resolve_path(args.model)),
        "candidate": str(resolve_path(args.candidate)),
        "runner_json": str(runner_json),
        "object_body": object_body,
        "args": vars(args),
        "geom_counts": {
            "object_geoms": len(geom_sets["object_geoms"]),
            "support_geoms": len(geom_sets["support_geoms"]),
            "hand_geoms": len(geom_sets["hand_geoms"]),
            "fr3_geoms": len(geom_sets["fr3_geoms"]),
        },
        "runner_ik": {
            "ik_pre_info": result.get("ik_pre_info"),
            "ik_grasp_info": result.get("ik_grasp_info"),
            "ik_lift_info": result.get("ik_lift_info"),
            "runner_success": result.get("success"),
            "final_counts": result.get("final_counts"),
            "final_rise": result.get("final_rise"),
        },
        "q_current": q_current,
        "q_pre": q_pre,
        "q_grasp": q_grasp,
        "q_lift": q_lift,
        "hand_ctrl": hand_ctrl,
        "path_precheck": path_infos,
        "static_precheck": static_infos,
        "precheck_status": status,
        "precheck_reasons": reasons,
    }

    save_json(out_path, out)

    print("\n========== V4.12A MUJOCO FK COLLISION BASELINE ==========")
    print("model       :", resolve_path(args.model))
    print("candidate   :", resolve_path(args.candidate))
    print("runner_json :", runner_json)
    print("object_body :", object_body)
    print("geom_counts :", out["geom_counts"])
    print("status      :", status)

    if reasons:
        print("\n[REASONS]")
        for r in reasons:
            print(" -", r)

    print("\n[RUNNER IK]")
    print("ik_pre_info  :", result.get("ik_pre_info"))
    print("ik_grasp_info:", result.get("ik_grasp_info"))
    print("ik_lift_info :", result.get("ik_lift_info"))

    print("\n[PATH MIN DISTANCES]")
    for p in path_infos:
        print("\n", p["path"])
        hs = p["min_hand_support"]
        fo = p["min_fr3_object"]
        ho = p["min_hand_object"]
        print(f"  min hand-support: {hs['distance']:+.5f}  {hs['body1']}/{hs['geom1']} <-> {hs['body2']}/{hs['geom2']}  at s={hs.get('s')}")
        print(f"  min fr3-object  : {fo['distance']:+.5f}  {fo['body1']}/{fo['geom1']} <-> {fo['body2']}/{fo['geom2']}  at s={fo.get('s')}")
        print(f"  min hand-object : {ho['distance']:+.5f}  {ho['body1']}/{ho['geom1']} <-> {ho['body2']}/{ho['geom2']}  at s={ho.get('s')}")

    print("\n[STATIC GRASP]")
    for k, v in static_infos.items():
        hs = v["min_hand_support"]
        ho = v["min_hand_object"]
        print(f"\n {k}")
        print(f"  min hand-support: {hs['distance']:+.5f}  {hs['body1']}/{hs['geom1']} <-> {hs['body2']}/{hs['geom2']}")
        print(f"  min hand-object : {ho['distance']:+.5f}  {ho['body1']}/{ho['geom1']} <-> {ho['body2']}/{ho['geom2']}")
        print(f"  contacts        : {v['contacts']}")

    print("\n[SAVED]", out_path)
    print("=========================================================\n")


if __name__ == "__main__":
    main()
