#!/usr/bin/env python3
from pathlib import Path
import argparse
import json
import math
import numpy as np
import mujoco
import pinocchio as pin


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
    return mujoco.mj_name2id(model, objtype, str(name))


def mj_joint_id(model, name):
    return mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def mj_body_id(model, name):
    return mj_id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def mj_site_id(model, name):
    return mj_id(model, mujoco.mjtObj.mjOBJ_SITE, name)


def T_from_Rp(R, p):
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(p, dtype=float).reshape(3)
    return T


def rot_angle(R):
    R = np.asarray(R, dtype=float).reshape(3, 3)
    c = 0.5 * (np.trace(R) - 1.0)
    c = float(np.clip(c, -1.0, 1.0))
    return float(math.acos(c))


def set_mujoco_joint_qpos(model, data, name, value):
    jid = mj_joint_id(model, name)
    if jid < 0:
        return False

    adr = int(model.jnt_qposadr[jid])
    jtype = int(model.jnt_type[jid])

    if jtype in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
        data.qpos[adr] = float(value)
        return True

    return False


def get_mujoco_joint_qpos(model, data, name):
    jid = mj_joint_id(model, name)
    if jid < 0:
        return None

    adr = int(model.jnt_qposadr[jid])
    jtype = int(model.jnt_type[jid])

    if jtype in [mujoco.mjtJoint.mjJNT_HINGE, mujoco.mjtJoint.mjJNT_SLIDE]:
        return float(data.qpos[adr])

    return None


def apply_mujoco_qdict(model, data, qdict):
    applied = {}
    missing = []
    for k, v in qdict.items():
        ok = set_mujoco_joint_qpos(model, data, k, v)
        if ok:
            applied[k] = float(v)
        else:
            missing.append(k)
    mujoco.mj_forward(model, data)
    return applied, missing


def mujoco_frame_T(model, data, frame_name):
    bid = mj_body_id(model, frame_name)
    if bid >= 0:
        return T_from_Rp(data.xmat[bid].reshape(3, 3), data.xpos[bid]), "body"

    sid = mj_site_id(model, frame_name)
    if sid >= 0:
        return T_from_Rp(data.site_xmat[sid].reshape(3, 3), data.site_xpos[sid]), "site"

    return None, None


def pin_find_frame_id(pin_model, frame_name):
    # 优先精确匹配 frame
    for i, f in enumerate(pin_model.frames):
        if f.name == frame_name:
            return i, f.name

    # 其次宽松匹配，便于排查 fr3_link7 / link7 / frame 名称差异
    candidates = []
    for i, f in enumerate(pin_model.frames):
        if frame_name in f.name or f.name in frame_name:
            candidates.append((i, f.name))

    if candidates:
        return candidates[0]

    return None, None


def pin_q_from_qdict(pin_model, qdict):
    q = pin.neutral(pin_model)
    applied = {}
    missing = []

    for jn, val in qdict.items():
        if not pin_model.existJointName(jn):
            missing.append(jn)
            continue

        jid = pin_model.getJointId(jn)
        nq = int(pin_model.nqs[jid])
        idx = int(pin_model.idx_qs[jid])

        if nq != 1:
            missing.append(jn)
            continue

        q[idx] = float(val)
        applied[jn] = float(val)

    return q, applied, missing


def pin_frame_T(pin_model, pin_data, q, frame_name):
    fid, matched_name = pin_find_frame_id(pin_model, frame_name)
    if fid is None:
        return None, None

    pin.forwardKinematics(pin_model, pin_data, q)
    pin.updateFramePlacements(pin_model, pin_data)

    M = pin_data.oMf[fid]
    return T_from_Rp(M.rotation, M.translation), matched_name


def compare_T(T_pin, T_mj):
    dp = T_pin[:3, 3] - T_mj[:3, 3]
    dR = T_mj[:3, :3].T @ T_pin[:3, :3]
    return {
        "pos_err_norm": float(np.linalg.norm(dp)),
        "pos_err_xyz": dp,
        "rot_err_rad": rot_angle(dR),
        "pin_pos": T_pin[:3, 3],
        "mujoco_pos": T_mj[:3, 3],
        "pin_R": T_pin[:3, :3],
        "mujoco_R": T_mj[:3, :3],
    }


def make_state_qdicts(mj_model, mj_data, runner_json):
    states = {}

    default_q = {}
    for j in ARM_JOINTS:
        v = get_mujoco_joint_qpos(mj_model, mj_data, j)
        if v is not None:
            default_q[j] = v
    states["model_default"] = default_q

    if runner_json:
        d = load_json(runner_json)
        if d.get("results"):
            r = d["results"][0]
            for key in ["q_pre", "q_grasp", "q_lift"]:
                qd = r.get(key, {}) or {}
                if qd:
                    states[key] = {k: float(v) for k, v in qd.items() if k in ARM_JOINTS}

    return states


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default="models/fr3_o7/fr3_o7_mujoco_ready_obj.urdf")
    ap.add_argument("--model", required=True)
    ap.add_argument("--runner-json", default="")
    ap.add_argument("--out", required=True)
    ap.add_argument("--frames", default="fr3_link7 fr3_link8 hand_base_link")
    args = ap.parse_args()

    urdf_path = resolve_path(args.urdf)
    mj_model_path = resolve_path(args.model)

    if not urdf_path.exists():
        raise RuntimeError(f"URDF not found: {urdf_path}")
    if not mj_model_path.exists():
        raise RuntimeError(f"MuJoCo XML not found: {mj_model_path}")

    pin_model = pin.buildModelFromUrdf(str(urdf_path))
    pin_data = pin_model.createData()

    mj_model = mujoco.MjModel.from_xml_path(str(mj_model_path))
    mj_data = mujoco.MjData(mj_model)
    mujoco.mj_resetData(mj_model, mj_data)
    mujoco.mj_forward(mj_model, mj_data)

    frames = args.frames.split()
    states = make_state_qdicts(mj_model, mj_data, args.runner_json)

    all_results = []

    print("\n========== V4.12P1 PINOCCHIO FK vs MUJOCO FK ==========")
    print("URDF :", urdf_path)
    print("MJCF :", mj_model_path)
    print("Pin model nq/nv:", pin_model.nq, pin_model.nv)
    print("Mj model nq/nv :", mj_model.nq, mj_model.nv)
    print("frames:", frames)
    print("states:", list(states.keys()))
    print("======================================================\n")

    for state_name, qdict in states.items():
        print(f"\n----- STATE: {state_name} -----")

        # MuJoCo 设定关节
        mujoco.mj_resetData(mj_model, mj_data)
        mujoco.mj_forward(mj_model, mj_data)
        mj_applied, mj_missing = apply_mujoco_qdict(mj_model, mj_data, qdict)

        # Pinocchio 设定关节
        q_pin, pin_applied, pin_missing = pin_q_from_qdict(pin_model, qdict)

        state_result = {
            "state": state_name,
            "qdict": qdict,
            "mujoco_applied": mj_applied,
            "mujoco_missing": mj_missing,
            "pin_applied": pin_applied,
            "pin_missing": pin_missing,
            "frames": {},
        }

        print("mujoco missing:", mj_missing)
        print("pin missing   :", pin_missing)

        for frame in frames:
            T_mj, mj_type = mujoco_frame_T(mj_model, mj_data, frame)
            T_pin, pin_frame_name = pin_frame_T(pin_model, pin_data, q_pin, frame)

            if T_mj is None or T_pin is None:
                item = {
                    "frame": frame,
                    "mujoco_found": T_mj is not None,
                    "mujoco_type": mj_type,
                    "pin_found": T_pin is not None,
                    "pin_frame_name": pin_frame_name,
                    "compare_ok": False,
                }
                state_result["frames"][frame] = item
                print(f"[{frame}] missing: mujoco={T_mj is not None}, pin={T_pin is not None}")
                continue

            cmp = compare_T(T_pin, T_mj)
            item = {
                "frame": frame,
                "mujoco_type": mj_type,
                "pin_frame_name": pin_frame_name,
                "compare": cmp,
                "compare_ok": bool(cmp["pos_err_norm"] < 1e-4 and cmp["rot_err_rad"] < 1e-3),
            }
            state_result["frames"][frame] = item

            print(
                f"[{frame}] pin_frame={pin_frame_name} "
                f"pos_err={cmp['pos_err_norm']:.8f} m "
                f"rot_err={cmp['rot_err_rad']:.8f} rad"
            )
            print("  pin_pos   :", cmp["pin_pos"])
            print("  mujoco_pos:", cmp["mujoco_pos"])
            print("  pos_diff  :", cmp["pos_err_xyz"])

        all_results.append(state_result)

    available_pin_frames = [f.name for f in pin_model.frames]
    available_mj_bodies = [
        mujoco.mj_id2name(mj_model, mujoco.mjtObj.mjOBJ_BODY, i)
        for i in range(mj_model.nbody)
    ]

    out = {
        "format": "v4_12p1_pinocchio_fk_compare_mujoco_debug",
        "urdf": str(urdf_path),
        "model": str(mj_model_path),
        "runner_json": str(resolve_path(args.runner_json)) if args.runner_json else "",
        "frames": frames,
        "pin_model": {
            "nq": pin_model.nq,
            "nv": pin_model.nv,
            "joint_names": list(pin_model.names),
        },
        "mujoco_model": {
            "nq": mj_model.nq,
            "nv": mj_model.nv,
        },
        "results": all_results,
        "available_pin_frames_preview": available_pin_frames[:200],
        "available_mujoco_bodies_preview": [x for x in available_mj_bodies if x][:200],
    }

    save_json(args.out, out)

    print("\n[SAVED]", resolve_path(args.out))
    print("======================================================\n")


if __name__ == "__main__":
    main()
