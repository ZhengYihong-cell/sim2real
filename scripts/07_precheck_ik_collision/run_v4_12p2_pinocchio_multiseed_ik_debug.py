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

# FR3 常用中性/自然 seed，不是最终强约束，只是多 seed 初值之一
FR3_NOMINAL = {
    "fr3_joint1": 0.0,
    "fr3_joint2": -0.785,
    "fr3_joint3": 0.0,
    "fr3_joint4": -2.356,
    "fr3_joint5": 0.0,
    "fr3_joint6": 1.571,
    "fr3_joint7": 0.785,
}


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


def mj_body_id(model, name):
    return mj_id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def mj_joint_id(model, name):
    return mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def T_from_Rp(R, p):
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(p, dtype=float).reshape(3)
    return T


def T_inv(T):
    T = np.asarray(T, dtype=float)
    out = np.eye(4, dtype=float)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def rot_angle(R):
    R = np.asarray(R, dtype=float).reshape(3, 3)
    c = 0.5 * (np.trace(R) - 1.0)
    c = float(np.clip(c, -1.0, 1.0))
    return float(math.acos(c))


def object_T_from_mujoco(model, data, object_body):
    bid = mj_body_id(model, object_body)
    if bid < 0:
        raise RuntimeError(f"cannot find object body: {object_body}")

    return T_from_Rp(data.xmat[bid].reshape(3, 3), data.xpos[bid])


def target_T_from_candidate(model_path, candidate_path, object_body):
    model = mujoco.MjModel.from_xml_path(str(resolve_path(model_path)))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    cand = load_json(candidate_path)
    T_object_target = np.asarray(cand["target"]["T_object_target"], dtype=float)

    T_world_object = object_T_from_mujoco(model, data, object_body)
    T_world_target = T_world_object @ T_object_target
    return T_world_target


def add_world_z(T, dz):
    out = np.array(T, dtype=float).copy()
    out[:3, 3] += np.array([0.0, 0.0, float(dz)])
    return out


def pin_find_frame_id(model, frame_name):
    for i, f in enumerate(model.frames):
        if f.name == frame_name:
            return i
    raise RuntimeError(f"cannot find Pinocchio frame: {frame_name}")


def pin_q_from_qdict(model, qdict):
    q = pin.neutral(model)
    for jn, val in qdict.items():
        if not model.existJointName(jn):
            continue
        jid = model.getJointId(jn)
        if int(model.nqs[jid]) != 1:
            continue
        idx = int(model.idx_qs[jid])
        q[idx] = float(val)
    return q


def qdict_from_pin_q(model, q):
    out = {}
    for jn in ARM_JOINTS:
        if not model.existJointName(jn):
            continue
        jid = model.getJointId(jn)
        if int(model.nqs[jid]) != 1:
            continue
        idx = int(model.idx_qs[jid])
        out[jn] = float(q[idx])
    return out


def arm_v_indices(model):
    out = []
    for jn in ARM_JOINTS:
        if not model.existJointName(jn):
            raise RuntimeError(f"Pinocchio model missing joint: {jn}")
        jid = model.getJointId(jn)
        out.append(int(model.idx_vs[jid]))
    return out


def get_mujoco_default_arm_q(model_path):
    model = mujoco.MjModel.from_xml_path(str(resolve_path(model_path)))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    out = {}
    for j in ARM_JOINTS:
        jid = mj_joint_id(model, j)
        if jid < 0:
            continue
        adr = int(model.jnt_qposadr[jid])
        out[j] = float(data.qpos[adr])
    return out


def load_runner_seed_qdicts(runner_json):
    if not runner_json:
        return {}

    p = resolve_path(runner_json)
    if not p.exists():
        return {}

    d = load_json(p)
    if not d.get("results"):
        return {}

    r = d["results"][0]
    out = {}
    for key in ["q_pre", "q_grasp", "q_lift"]:
        qd = r.get(key, {}) or {}
        if qd:
            out[key] = {k: float(v) for k, v in qd.items() if k in ARM_JOINTS}
    return out


def frame_pose(model, data, q, frame_id):
    pin.forwardKinematics(model, data, q)
    pin.updateFramePlacements(model, data)
    M = data.oMf[frame_id]
    return M.rotation.copy(), M.translation.copy()


def orientation_error_world(R_current, R_target):
    # 世界系姿态误差，适合搭配 LOCAL_WORLD_ALIGNED Jacobian 的 angular 部分
    return 0.5 * (
        np.cross(R_current[:, 0], R_target[:, 0])
        + np.cross(R_current[:, 1], R_target[:, 1])
        + np.cross(R_current[:, 2], R_target[:, 2])
    )


def pose_error(model, data, q, frame_id, T_target, rot_weight):
    R_cur, p_cur = frame_pose(model, data, q, frame_id)
    R_t = T_target[:3, :3]
    p_t = T_target[:3, 3]

    e_pos = p_t - p_cur
    e_rot = orientation_error_world(R_cur, R_t)

    e = np.concatenate([e_pos, float(rot_weight) * e_rot])
    return e, e_pos, e_rot, R_cur, p_cur


def clamp_q_to_limits(model, q, margin=1e-6):
    q = np.array(q, dtype=float).copy()

    for jn in ARM_JOINTS:
        if not model.existJointName(jn):
            continue
        jid = model.getJointId(jn)
        if int(model.nqs[jid]) != 1:
            continue

        qi = int(model.idx_qs[jid])
        lo = float(model.lowerPositionLimit[qi])
        hi = float(model.upperPositionLimit[qi])

        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            q[qi] = float(np.clip(q[qi], lo + margin, hi - margin))

    return q


def joint_limit_margin(model, q):
    margins = {}
    min_margin = 999.0

    for jn in ARM_JOINTS:
        if not model.existJointName(jn):
            continue
        jid = model.getJointId(jn)
        qi = int(model.idx_qs[jid])
        lo = float(model.lowerPositionLimit[qi])
        hi = float(model.upperPositionLimit[qi])
        val = float(q[qi])

        if np.isfinite(lo) and np.isfinite(hi) and lo < hi:
            m = min(val - lo, hi - val)
        else:
            m = 999.0

        margins[jn] = {
            "q": val,
            "lower": lo,
            "upper": hi,
            "margin": float(m),
        }
        min_margin = min(min_margin, float(m))

    return float(min_margin), margins


def seed_distance(model, q, q_seed):
    qd = qdict_from_pin_q(model, q)
    sd = qdict_from_pin_q(model, q_seed)
    vals = []
    for j in ARM_JOINTS:
        vals.append(float(qd[j] - sd[j]))
    return float(np.linalg.norm(vals))


def solve_ik_one(model, frame_id, T_target, q_seed, args):
    data = model.createData()
    q = np.array(q_seed, dtype=float).copy()
    v_idx = arm_v_indices(model)

    best = None

    for it in range(args.max_iters):
        e, e_pos, e_rot, R_cur, p_cur = pose_error(
            model, data, q, frame_id, T_target, args.rot_weight
        )

        pos_err = float(np.linalg.norm(e_pos))
        rot_err = float(np.linalg.norm(e_rot))
        err_norm = float(np.linalg.norm(e))

        if best is None or err_norm < best["err_norm"]:
            best = {
                "iter": it,
                "q": q.copy(),
                "pos_err": pos_err,
                "rot_err": rot_err,
                "err_norm": err_norm,
            }

        if pos_err < args.pos_tol and rot_err < args.rot_tol:
            break

        J = pin.computeFrameJacobian(
            model,
            data,
            q,
            frame_id,
            pin.ReferenceFrame.LOCAL_WORLD_ALIGNED,
        )

        J = np.asarray(J, dtype=float)
        J_arm = J[:, v_idx].copy()
        J_arm[3:6, :] *= float(args.rot_weight)

        A = J_arm @ J_arm.T + (args.damping ** 2) * np.eye(6)
        dq_arm = J_arm.T @ np.linalg.solve(A, e)

        step_norm = float(np.linalg.norm(dq_arm))
        if step_norm > args.max_step:
            dq_arm *= float(args.max_step / (step_norm + 1e-12))

        dq_full = np.zeros(model.nv)
        for k, vi in enumerate(v_idx):
            dq_full[vi] = float(args.step_scale) * float(dq_arm[k])

        q = pin.integrate(model, q, dq_full)
        q = clamp_q_to_limits(model, q)

    q_best = best["q"]
    e, e_pos, e_rot, R_cur, p_cur = pose_error(
        model, data, q_best, frame_id, T_target, args.rot_weight
    )
    min_margin, margins = joint_limit_margin(model, q_best)

    success = bool(best["pos_err"] < args.pos_tol and best["rot_err"] < args.rot_tol)

    return {
        "success": success,
        "iter": int(best["iter"]),
        "pos_err": float(best["pos_err"]),
        "rot_err": float(best["rot_err"]),
        "err_norm": float(best["err_norm"]),
        "q": qdict_from_pin_q(model, q_best),
        "min_joint_limit_margin": min_margin,
        "joint_limit_margins": margins,
        "target_pos": T_target[:3, 3].copy(),
        "final_pos": p_cur.copy(),
    }


def generate_seeds(pin_model, base_seed_dicts, args):
    seeds = []

    for name, qd in base_seed_dicts.items():
        seeds.append((name, pin_q_from_qdict(pin_model, qd)))

    rng = np.random.default_rng(args.seed)
    nominal = pin_q_from_qdict(pin_model, FR3_NOMINAL)

    for i in range(args.random_seeds):
        noise = rng.normal(0.0, args.random_std, size=len(ARM_JOINTS))
        qd = dict(FR3_NOMINAL)
        for j, n in zip(ARM_JOINTS, noise):
            qd[j] = float(qd[j] + n)
        seeds.append((f"random_nominal_{i:02d}", pin_q_from_qdict(pin_model, qd)))

    # 去重：只按 arm q 四舍五入
    unique = []
    seen = set()
    for name, q in seeds:
        qd = qdict_from_pin_q(pin_model, q)
        key = tuple(round(qd[j], 4) for j in ARM_JOINTS)
        if key in seen:
            continue
        seen.add(key)
        unique.append((name, q))

    return unique


def solution_score(sol, seed_dist, args):
    # 这里只是 IK 层评分；碰撞分数在 P3 加
    limit_penalty = 0.0
    if sol["min_joint_limit_margin"] < args.limit_margin_soft:
        limit_penalty = args.limit_margin_soft - sol["min_joint_limit_margin"]

    return float(
        1000.0 * sol["pos_err"]
        + 5.0 * sol["rot_err"]
        + args.seed_dist_weight * seed_dist
        + args.limit_penalty_weight * limit_penalty
        + (0.0 if sol["success"] else 100.0)
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--urdf", default="models/fr3_o7/fr3_o7_mujoco_ready_obj.urdf")
    ap.add_argument("--model", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--runner-json", default="")
    ap.add_argument("--object-body", default="")
    ap.add_argument("--target-frame", default="fr3_link7")
    ap.add_argument("--out", required=True)

    ap.add_argument("--pregrasp-z", type=float, default=0.085)
    ap.add_argument("--lift-z", type=float, default=0.120)

    ap.add_argument("--max-iters", type=int, default=300)
    ap.add_argument("--pos-tol", type=float, default=2e-4)
    ap.add_argument("--rot-tol", type=float, default=2e-3)
    ap.add_argument("--rot-weight", type=float, default=0.55)
    ap.add_argument("--damping", type=float, default=1e-3)
    ap.add_argument("--step-scale", type=float, default=0.65)
    ap.add_argument("--max-step", type=float, default=0.08)

    ap.add_argument("--random-seeds", type=int, default=16)
    ap.add_argument("--random-std", type=float, default=0.7)
    ap.add_argument("--seed", type=int, default=7)

    ap.add_argument("--limit-margin-soft", type=float, default=0.08)
    ap.add_argument("--seed-dist-weight", type=float, default=0.02)
    ap.add_argument("--limit-penalty-weight", type=float, default=3.0)

    args = ap.parse_args()

    urdf_path = resolve_path(args.urdf)
    model_path = resolve_path(args.model)
    cand_path = resolve_path(args.candidate)

    cand = load_json(cand_path)
    object_body = args.object_body or (cand.get("object") or {}).get("body", "")
    if not object_body:
        raise RuntimeError("cannot infer object body; pass --object-body")

    pin_model = pin.buildModelFromUrdf(str(urdf_path))
    frame_id = pin_find_frame_id(pin_model, args.target_frame)

    T_grasp = target_T_from_candidate(model_path, cand_path, object_body)
    T_pre = add_world_z(T_grasp, args.pregrasp_z)
    T_lift = add_world_z(T_grasp, args.lift_z)

    runner_seeds = load_runner_seed_qdicts(args.runner_json)
    default_seed = get_mujoco_default_arm_q(model_path)

    base_seed_dicts = {
        "model_default": default_seed,
        "fr3_nominal": FR3_NOMINAL,
    }

    for k, v in runner_seeds.items():
        base_seed_dicts[f"runner_{k}"] = v

    seeds = generate_seeds(pin_model, base_seed_dicts, args)

    targets = {
        "pre": T_pre,
        "grasp": T_grasp,
        "lift": T_lift,
    }

    print("\n========== V4.12P2 PINOCCHIO MULTI-SEED IK ==========")
    print("urdf        :", urdf_path)
    print("model       :", model_path)
    print("candidate   :", cand_path)
    print("object_body :", object_body)
    print("target_frame:", args.target_frame)
    print("num_seeds   :", len(seeds))
    print("target grasp pos:", T_grasp[:3, 3])
    print("target pre pos  :", T_pre[:3, 3])
    print("target lift pos :", T_lift[:3, 3])
    print("=====================================================\n")

    all_results = {}

    for target_name, T_target in targets.items():
        print(f"\n----- TARGET: {target_name} -----")
        target_results = []

        for seed_name, q_seed in seeds:
            sol = solve_ik_one(pin_model, frame_id, T_target, q_seed, args)
            sd = seed_distance(pin_model, pin_q_from_qdict(pin_model, sol["q"]), q_seed)
            score = solution_score(sol, sd, args)

            item = {
                "target": target_name,
                "seed_name": seed_name,
                "score": score,
                "seed_distance": sd,
                **sol,
            }
            target_results.append(item)

        target_results.sort(key=lambda x: x["score"])
        all_results[target_name] = target_results

        for i, r in enumerate(target_results[:8], 1):
            print(
                f"{i:02d}. seed={r['seed_name']} "
                f"success={int(r['success'])} "
                f"score={r['score']:.5f} "
                f"pos={r['pos_err']:.6f} "
                f"rot={r['rot_err']:.6f} "
                f"margin={r['min_joint_limit_margin']:.4f} "
                f"seedDist={r['seed_distance']:.4f}"
            )
            print("    q:", {k: round(v, 4) for k, v in r["q"].items()})

    out = {
        "format": "v4_12p2_pinocchio_multiseed_ik_debug",
        "urdf": str(urdf_path),
        "model": str(model_path),
        "candidate": str(cand_path),
        "runner_json": str(resolve_path(args.runner_json)) if args.runner_json else "",
        "object_body": object_body,
        "target_frame": args.target_frame,
        "args": vars(args),
        "target_poses": {
            "pre": T_pre,
            "grasp": T_grasp,
            "lift": T_lift,
        },
        "base_seed_dicts": base_seed_dicts,
        "num_seeds": len(seeds),
        "results": all_results,
        "best": {
            k: v[0] if v else None for k, v in all_results.items()
        },
    }

    save_json(args.out, out)

    print("\n[SAVED]", resolve_path(args.out))
    print("=====================================================\n")


if __name__ == "__main__":
    main()
