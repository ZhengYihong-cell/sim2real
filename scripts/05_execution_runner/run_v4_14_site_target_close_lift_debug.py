#!/usr/bin/env python3
"""
脚本类型：
    debug / execution-runner / v4.14 / site-target-close-lift

用途：
    验证单个数据集 sample 在正确 frame 链路下能否动态 close 和 lift。
    本脚本直接使用数据集 hand_pose：
        T_world_dataset_hand_base_debug = T_world_object @ T_object_hand_base_link
    然后对 MuJoCo site `dataset_hand_base_debug` 做 IK，得到 q_grasp。
    这避免旧 V4.13 中错误的 fr3_link7 target 转换。

输入：
    --model          当前 scene.xml
    --npy            object.npy
    --sample-index   valid local sample index
    --object-body    MuJoCo object body，例如 grasp_object
    --target-site    dataset_hand_base_debug

输出：
    out_dir/result.json
    out_dir/terminal.txt

当前流程位置：
    frame 验证已证明 site IK 正确
        -> 本脚本验证该 sample 在 q_grasp 处能否 close/lift
        -> 若成功，再把 P2/P3/runner 主线改成 site-target 版本

不负责：
    1. 不做旧 fr3_link7 target；
    2. 不做完整路径规划；
    3. 不做 selector 排序；
    4. 不做沿轴人工微调；
    5. 不修改 legacy_final_demos。
"""

from pathlib import Path
import argparse
import json
import math
import time

import numpy as np
import mujoco
import mujoco.viewer



LIVE_VIEWER = None
LIVE_SLEEP = 0.0015

def live_sync():
    global LIVE_VIEWER, LIVE_SLEEP
    if LIVE_VIEWER is not None:
        try:
            if LIVE_VIEWER.is_running():
                LIVE_VIEWER.sync()
                time.sleep(float(LIVE_SLEEP))
        except Exception:
            pass

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

O7_ACTIVE_JOINTS = [
    "thumb_cmc_roll",
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]

NON_THUMB = ["index", "middle", "ring", "pinky"]

Q_HOME = {
    "fr3_joint1": 0.0,
    "fr3_joint2": -0.7,
    "fr3_joint3": 0.0,
    "fr3_joint4": -2.2,
    "fr3_joint5": 0.0,
    "fr3_joint6": 1.8,
    "fr3_joint7": 0.8,
}


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


def normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n


def robust_rot6d_to_R(r6):
    r6 = np.asarray(r6, dtype=float).reshape(6)
    x_raw = r6[0:3]
    y_raw = r6[3:6]

    x = normalize(x_raw)
    y = normalize(y_raw)

    middle = normalize(x + y)
    orthmid = normalize(x - y)

    x = normalize(middle + orthmid)
    y = normalize(middle - orthmid)
    z = normalize(np.cross(x, y))

    return np.stack([x, y, z], axis=1)


def T_from_Rp(R, p):
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(p, dtype=float)
    return T


def mat_to_dict(T):
    return {
        "pos": T[:3, 3].tolist(),
        "R": T[:3, :3].tolist(),
        "T": T.tolist(),
    }


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


def clamp_ctrl(model, aid, value):
    value = float(value)
    if aid is None:
        return value
    limited = int(model.actuator_ctrllimited[aid])
    if limited:
        lo, hi = model.actuator_ctrlrange[aid]
        value = float(np.clip(value, lo, hi))
    return value


def clamp_joint(model, joint_name, value):
    jid = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return float(value)
    if int(model.jnt_limited[jid]):
        lo, hi = model.jnt_range[jid]
        return float(np.clip(value, lo, hi))
    return float(value)


def set_joint_qpos(model, data, joint_name, value):
    adr = joint_qpos_addr(model, joint_name)
    if adr is None:
        return False
    data.qpos[adr] = clamp_joint(model, joint_name, value)
    return True


def set_joint_ctrl(model, data, joint_name, value):
    aid = actuator_for_joint(model, joint_name)
    if aid is None:
        return False
    data.ctrl[aid] = clamp_ctrl(model, aid, value)
    return True


def set_arm_qpos_ctrl(model, data, q_arm):
    for j, v in q_arm.items():
        set_joint_qpos(model, data, j, v)
        set_joint_ctrl(model, data, j, v)
        da = joint_dof_addr(model, j)
        if da is not None:
            data.qvel[da] = 0.0
    mujoco.mj_forward(model, data)


def set_hand_qpos_ctrl(model, data, ctrl, set_qpos=True):
    for j, v in ctrl.items():
        if set_qpos:
            set_joint_qpos(model, data, j, v)
        set_joint_ctrl(model, data, j, v)
    mujoco.mj_forward(model, data)


def get_joint_values(model, data, names):
    out = {}
    for n in names:
        adr = joint_qpos_addr(model, n)
        if adr is not None:
            out[n] = float(data.qpos[adr])
    return out


def interp_dict(a, b, alpha):
    out = {}
    keys = set(a.keys()) | set(b.keys())
    for k in keys:
        va = float(a.get(k, b.get(k, 0.0)))
        vb = float(b.get(k, a.get(k, 0.0)))
        out[k] = (1.0 - alpha) * va + alpha * vb
    return out


def load_sample(npy, idx):
    arr = np.load(npy, allow_pickle=True)
    if idx < 0 or idx >= len(arr):
        raise RuntimeError(f"sample index out of range: {idx}, n={len(arr)}")
    s = arr[idx].item() if hasattr(arr[idx], "item") else arr[idx]
    if not isinstance(s, dict):
        raise RuntimeError(f"sample is not dict: {type(s)}")
    return s


def sample_T_object_hand(sample):
    hp = np.asarray(sample["hand_pose"], dtype=float)
    t = hp[0:3]
    R = robust_rot6d_to_R(hp[3:9])
    return T_from_Rp(R, t)


def sample_ctrl(sample):
    hp = np.asarray(sample["hand_pose"], dtype=float)
    if hp.shape[0] < 16:
        raise RuntimeError("hand_pose does not contain 7 active O7 ctrl values")
    return {j: float(v) for j, v in zip(O7_ACTIVE_JOINTS, hp[9:16])}


def side_open_from_close(close):
    side = dict(close)
    for j in ["index_mcp_pitch", "middle_mcp_pitch", "ring_mcp_pitch", "pinky_mcp_pitch"]:
        side[j] = 0.0
    return side


def squeeze_from_close(close, finger_scale=1.08, thumb_pitch_extra=0.03):
    out = dict(close)
    for j in ["index_mcp_pitch", "middle_mcp_pitch", "ring_mcp_pitch", "pinky_mcp_pitch"]:
        out[j] = float(out[j]) * finger_scale
    if "thumb_cmc_pitch" in out:
        out["thumb_cmc_pitch"] = float(out["thumb_cmc_pitch"]) + float(thumb_pitch_extra)
    return out


def body_world_T(model, data, body_name):
    bid = name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise RuntimeError(f"missing body: {body_name}")
    R = np.asarray(data.xmat[bid], dtype=float).reshape(3, 3)
    p = np.asarray(data.xpos[bid], dtype=float)
    return T_from_Rp(R, p)


def site_world_T(model, data, site_name):
    sid = name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0:
        available = []
        for i in range(model.nsite):
            n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
            if n:
                available.append(n)
        raise RuntimeError(f"missing site: {site_name}, available={available}")
    R = np.asarray(data.site_xmat[sid], dtype=float).reshape(3, 3)
    p = np.asarray(data.site_xpos[sid], dtype=float)
    return T_from_Rp(R, p)


def pose_error(T_cur, T_tar):
    pc = T_cur[:3, 3]
    pt = T_tar[:3, 3]
    Rc = T_cur[:3, :3]
    Rt = T_tar[:3, :3]
    pos_err = pt - pc
    rot_err = 0.5 * (
        np.cross(Rc[:, 0], Rt[:, 0]) +
        np.cross(Rc[:, 1], Rt[:, 1]) +
        np.cross(Rc[:, 2], Rt[:, 2])
    )
    return pos_err, rot_err, float(np.linalg.norm(pos_err)), float(np.linalg.norm(rot_err))


def solve_site_ik(model, data, site_name, T_target, q_seed=None,
                  max_iters=350, damping=1e-4, step_scale=0.85,
                  rot_weight=0.65, pos_tol=8e-4, rot_tol=8e-3):
    if q_seed:
        set_arm_qpos_ctrl(model, data, q_seed)

    sid = name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0:
        raise RuntimeError(f"missing site: {site_name}")

    dof_ids = []
    for j in ARM_JOINTS:
        da = joint_dof_addr(model, j)
        if da is None:
            raise RuntimeError(f"missing arm joint: {j}")
        dof_ids.append(da)
    dof_ids = np.asarray(dof_ids, dtype=int)

    history = []
    success = False

    for it in range(max_iters):
        mujoco.mj_forward(model, data)
        T_cur = site_world_T(model, data, site_name)
        pos_err, rot_err, pos_n, rot_n = pose_error(T_cur, T_target)

        history.append({
            "iter": it,
            "pos_err_norm": pos_n,
            "rot_err_norm": rot_n,
        })

        if pos_n < pos_tol and rot_n < rot_tol:
            success = True
            break

        jacp = np.zeros((3, model.nv), dtype=float)
        jacr = np.zeros((3, model.nv), dtype=float)
        mujoco.mj_jacSite(model, data, jacp, jacr, sid)

        J = np.vstack([jacp[:, dof_ids], rot_weight * jacr[:, dof_ids]])
        e = np.concatenate([pos_err, rot_weight * rot_err])

        A = J @ J.T + damping * np.eye(6)
        dq = J.T @ np.linalg.solve(A, e)

        n = float(np.linalg.norm(dq))
        if n > 0.10:
            dq *= 0.10 / n
        dq *= step_scale

        for k, j in enumerate(ARM_JOINTS):
            adr = joint_qpos_addr(model, j)
            data.qpos[adr] = clamp_joint(model, j, float(data.qpos[adr] + dq[k]))

    mujoco.mj_forward(model, data)
    T_final = site_world_T(model, data, site_name)
    _, _, pos_n, rot_n = pose_error(T_final, T_target)

    return {
        "success": bool(success),
        "iters": len(history),
        "final_pos_err_norm": pos_n,
        "final_rot_err_norm": rot_n,
        "q_arm": get_joint_values(model, data, ARM_JOINTS),
        "history_tail": history[-10:],
    }


def geom_name(model, gid):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or f"geom_{gid}"


def geom_body_name(model, gid):
    bid = int(model.geom_bodyid[int(gid)])
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""


def object_geom_ids(model, object_body):
    bid = name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
    if bid < 0:
        raise RuntimeError(f"missing object body: {object_body}")
    return {gid for gid in range(model.ngeom) if int(model.geom_bodyid[gid]) == bid}


def classify_hand_group(name):
    s = str(name).lower()
    if "thumb" in s:
        return "thumb"
    if "index" in s:
        return "index"
    if "middle" in s:
        return "middle"
    if "ring" in s:
        return "ring"
    if "pinky" in s:
        return "pinky"
    if "palm" in s or "hand" in s:
        return "palm"
    return None


def contact_groups(model, data, object_body):
    obj_geoms = object_geom_ids(model, object_body)
    groups = {}
    contacts = []

    for i in range(data.ncon):
        c = data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)

        if g1 in obj_geoms:
            other = g2
            objg = g1
        elif g2 in obj_geoms:
            other = g1
            objg = g2
        else:
            continue

        name = geom_name(model, other) + " " + geom_body_name(model, other)
        grp = classify_hand_group(name)
        if grp is None:
            continue

        groups[grp] = groups.get(grp, 0) + 1
        contacts.append({
            "group": grp,
            "hand_geom": geom_name(model, other),
            "object_geom": geom_name(model, objg),
            "dist": float(c.dist),
        })

    return groups, contacts


def ready_from_groups(groups):
    if "thumb" not in groups:
        return False
    return any(g in groups for g in NON_THUMB)


def object_pos(model, data, object_body):
    bid = name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
    return np.asarray(data.xpos[bid], dtype=float).copy()


def step_with_targets(model, data, q_arm, hand_ctrl, steps, object_body,
                      rows, phase, print_every=100, stable_counter_start=0,
                      ready_stable_steps=6):
    stable = int(stable_counter_start)
    last_ready = False

    for k in range(steps + 1):
        alpha = k / max(steps, 1)

        set_arm_qpos_ctrl(model, data, q_arm)
        set_hand_qpos_ctrl(model, data, hand_ctrl, set_qpos=False)

        mujoco.mj_step(model, data)
        live_sync()
        groups, contacts = contact_groups(model, data, object_body)
        ready = ready_from_groups(groups)

        if ready:
            stable += 1
        else:
            stable = 0

        last_ready = ready

        if k % print_every == 0 or k == steps:
            print(
                f"[{phase}] {k:5d}/{steps} alpha={alpha:.3f} "
                f"groups={groups} ready={ready} stable={stable}/{ready_stable_steps}"
            )

        rows.append({
            "phase": phase,
            "step": k,
            "alpha": alpha,
            "time": float(data.time),
            "object_pos": object_pos(model, data, object_body).tolist(),
            "groups": groups,
            "contacts": contacts[:20],
            "ready": bool(ready),
            "stable_count": int(stable),
            "hand_ctrl": dict(hand_ctrl),
        })

    return stable, last_ready


def step_interpolated(model, data, q0, q1, hand0, hand1, steps, object_body,
                      rows, phase, print_every=100, stable_counter_start=0,
                      ready_stable_steps=6):
    stable = int(stable_counter_start)
    last_ready = False

    for k in range(steps + 1):
        alpha = k / max(steps, 1)
        q_arm = interp_dict(q0, q1, alpha)
        hand = interp_dict(hand0, hand1, alpha)

        set_arm_qpos_ctrl(model, data, q_arm)
        set_hand_qpos_ctrl(model, data, hand, set_qpos=False)

        mujoco.mj_step(model, data)
        live_sync()
        groups, contacts = contact_groups(model, data, object_body)
        ready = ready_from_groups(groups)
        if ready:
            stable += 1
        else:
            stable = 0
        last_ready = ready

        if k % print_every == 0 or k == steps:
            print(
                f"[{phase}] {k:5d}/{steps} alpha={alpha:.3f} "
                f"groups={groups} ready={ready} stable={stable}/{ready_stable_steps}"
            )

        rows.append({
            "phase": phase,
            "step": k,
            "alpha": alpha,
            "time": float(data.time),
            "object_pos": object_pos(model, data, object_body).tolist(),
            "groups": groups,
            "contacts": contacts[:20],
            "ready": bool(ready),
            "stable_count": int(stable),
            "hand_ctrl": dict(hand),
        })

    return stable, last_ready


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--npy", required=True)
    ap.add_argument("--sample-index", type=int, required=True)
    ap.add_argument("--object-body", default="grasp_object")
    ap.add_argument("--target-site", default="dataset_hand_base_debug")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--force-lift", action="store_true",
                    help="debug only: run lift even if grip_ready is false, so we can visually inspect whether the grasp can carry the object")
    ap.add_argument("--live-sleep", type=float, default=0.0015,
                    help="sleep time after each viewer sync in live viewer mode")

    ap.add_argument("--settle-steps", type=int, default=500)
    ap.add_argument("--grasp-hold-steps", type=int, default=250)
    ap.add_argument("--close-steps", type=int, default=650)
    ap.add_argument("--post-close-steps", type=int, default=350)
    ap.add_argument("--squeeze-steps", type=int, default=450)
    ap.add_argument("--lift-steps", type=int, default=1800)
    ap.add_argument("--final-hold-steps", type=int, default=500)

    ap.add_argument("--lift-z", type=float, default=0.09)
    ap.add_argument("--finger-squeeze-scale", type=float, default=1.08)
    ap.add_argument("--thumb-pitch-extra", type=float, default=0.03)
    ap.add_argument("--ready-stable-steps", type=int, default=6)
    ap.add_argument("--print-every", type=int, default=100)

    args = ap.parse_args()

    model_path = resolve(args.model)
    npy_path = resolve(args.npy)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    global LIVE_VIEWER, LIVE_SLEEP
    LIVE_SLEEP = float(args.live_sleep)
    if args.viewer:
        LIVE_VIEWER = mujoco.viewer.launch_passive(model, data)
        LIVE_VIEWER.cam.lookat[:] = [0.455, 0.0, 0.30]
        LIVE_VIEWER.cam.distance = 0.75
        LIVE_VIEWER.cam.azimuth = 130
        LIVE_VIEWER.cam.elevation = -25
        live_sync()

    sample = load_sample(npy_path, args.sample_index)
    T_object_hand = sample_T_object_hand(sample)
    close_ctrl = sample_ctrl(sample)
    side_open_ctrl = side_open_from_close(close_ctrl)
    squeeze_ctrl = squeeze_from_close(
        close_ctrl,
        finger_scale=args.finger_squeeze_scale,
        thumb_pitch_extra=args.thumb_pitch_extra,
    )

    rows = []

    print("========== V4.14 SITE-TARGET CLOSE/LIFT ==========")
    print("model      :", rel(model_path))
    print("npy        :", rel(npy_path))
    print("sample_idx :", args.sample_index)
    print("object_body:", args.object_body)
    print("target_site:", args.target_site)

    # 初始放到安全 home，手打开，让物体先稳定。
    set_arm_qpos_ctrl(model, data, Q_HOME)
    set_hand_qpos_ctrl(model, data, side_open_ctrl, set_qpos=True)

    print("\n[PHASE] settle object at home/side-open")
    stable = 0
    for k in range(args.settle_steps):
        set_arm_qpos_ctrl(model, data, Q_HOME)
        set_hand_qpos_ctrl(model, data, side_open_ctrl, set_qpos=False)
        mujoco.mj_step(model, data)
        live_sync()
    object_pos_after_settle = object_pos(model, data, args.object_body)
    T_world_object = body_world_T(model, data, args.object_body)
    T_world_target = T_world_object @ T_object_hand

    print("object_pos_after_settle:", object_pos_after_settle.tolist())
    print("target_site_pos        :", T_world_target[:3, 3].tolist())

    # 解 q_grasp。
    ik_grasp = solve_site_ik(
        model, data, args.target_site, T_world_target,
        q_seed=Q_HOME,
        max_iters=350,
        damping=1e-4,
        step_scale=0.85,
        rot_weight=0.65,
    )

    q_grasp = ik_grasp["q_arm"]

    print("\n[IK grasp]")
    print("success:", ik_grasp["success"])
    print("final_pos_err:", ik_grasp["final_pos_err_norm"])
    print("final_rot_err:", ik_grasp["final_rot_err_norm"])
    print("q_grasp:", q_grasp)

    # 解 q_lift，目标是同一个 site target 世界 z 上抬。
    T_lift_target = np.array(T_world_target, dtype=float)
    T_lift_target[2, 3] += float(args.lift_z)

    ik_lift = solve_site_ik(
        model, data, args.target_site, T_lift_target,
        q_seed=q_grasp,
        max_iters=350,
        damping=1e-4,
        step_scale=0.85,
        rot_weight=0.65,
    )

    q_lift = ik_lift["q_arm"]

    print("\n[IK lift]")
    print("success:", ik_lift["success"])
    print("final_pos_err:", ik_lift["final_pos_err_norm"])
    print("final_rot_err:", ik_lift["final_rot_err_norm"])
    print("q_lift:", q_lift)

    # 直接进入 q_grasp，验证 close/lift，不验证 approach path。
    set_arm_qpos_ctrl(model, data, q_grasp)
    set_hand_qpos_ctrl(model, data, side_open_ctrl, set_qpos=True)
    mujoco.mj_forward(model, data)

    grasp_object_start_pos = object_pos(model, data, args.object_body)

    print("\n[PHASE] hold at q_grasp side-open")
    stable, _ = step_with_targets(
        model, data, q_grasp, side_open_ctrl,
        args.grasp_hold_steps,
        args.object_body,
        rows,
        "hold_at_site_grasp_side_open",
        print_every=args.print_every,
        stable_counter_start=stable,
        ready_stable_steps=args.ready_stable_steps,
    )

    print("\n[PHASE] close to dataset sample ctrl")
    stable, _ = step_interpolated(
        model, data, q_grasp, q_grasp,
        side_open_ctrl, close_ctrl,
        args.close_steps,
        args.object_body,
        rows,
        "close_to_dataset_ctrl",
        print_every=args.print_every,
        stable_counter_start=stable,
        ready_stable_steps=args.ready_stable_steps,
    )

    print("\n[PHASE] post-close hold")
    stable, _ = step_with_targets(
        model, data, q_grasp, close_ctrl,
        args.post_close_steps,
        args.object_body,
        rows,
        "post_close_hold",
        print_every=args.print_every,
        stable_counter_start=stable,
        ready_stable_steps=args.ready_stable_steps,
    )

    print("\n[PHASE] gated micro squeeze")
    stable, _ = step_interpolated(
        model, data, q_grasp, q_grasp,
        close_ctrl, squeeze_ctrl,
        args.squeeze_steps,
        args.object_body,
        rows,
        "gated_micro_squeeze",
        print_every=args.print_every,
        stable_counter_start=stable,
        ready_stable_steps=args.ready_stable_steps,
    )

    groups_now, contacts_now = contact_groups(model, data, args.object_body)
    grip_ready = ready_from_groups(groups_now) and stable >= args.ready_stable_steps

    print("\n[READY CHECK]")
    print("groups:", groups_now)
    print("stable:", stable)
    print("grip_ready:", grip_ready)

    lifted = False
    if args.force_lift and not grip_ready:
        print("\n[FORCE_LIFT_DEBUG] grip_ready is False, but force-lift is enabled for visual diagnosis.")

    if (grip_ready or args.force_lift) and ik_lift["success"]:
        print("\n[PHASE] lift with fixed grip")
        stable, _ = step_interpolated(
            model, data, q_grasp, q_lift,
            squeeze_ctrl, squeeze_ctrl,
            args.lift_steps,
            args.object_body,
            rows,
            "fixed_grip_lift",
            print_every=args.print_every,
            stable_counter_start=stable,
            ready_stable_steps=args.ready_stable_steps,
        )
        lifted = True

        print("\n[PHASE] final hold")
        stable, _ = step_with_targets(
            model, data, q_lift, squeeze_ctrl,
            args.final_hold_steps,
            args.object_body,
            rows,
            "final_hold_after_lift",
            print_every=args.print_every,
            stable_counter_start=stable,
            ready_stable_steps=args.ready_stable_steps,
        )
    else:
        print("\n[NO_LIFT] grip not ready or lift IK failed.")

    final_pos = object_pos(model, data, args.object_body)
    final_groups, final_contacts = contact_groups(model, data, args.object_body)

    final_rise = float(final_pos[2] - grasp_object_start_pos[2])
    final_disp = float(np.linalg.norm(final_pos - grasp_object_start_pos))

    success = bool(grip_ready and lifted and final_rise > 0.03)

    result = {
        "format": "v4_14_site_target_close_lift_debug_v1",
        "model": rel(model_path),
        "npy": rel(npy_path),
        "sample_index_valid_local": args.sample_index,
        "object_body": args.object_body,
        "target_site": args.target_site,
        "T_world_object_after_settle": mat_to_dict(T_world_object),
        "T_object_hand_base_from_dataset": mat_to_dict(T_object_hand),
        "T_world_site_target": mat_to_dict(T_world_target),
        "ik_grasp": ik_grasp,
        "ik_lift": ik_lift,
        "side_open_ctrl": side_open_ctrl,
        "close_ctrl": close_ctrl,
        "squeeze_ctrl": squeeze_ctrl,
        "object_pos_after_settle": object_pos_after_settle.tolist(),
        "grasp_object_start_pos": grasp_object_start_pos.tolist(),
        "final_object_pos": final_pos.tolist(),
        "final_object_disp": final_disp,
        "final_object_rise": final_rise,
        "grip_ready": grip_ready,
        "force_lift_debug": bool(args.force_lift),
        "lifted": lifted,
        "success": success,
        "final_groups": final_groups,
        "final_contacts": final_contacts[:30],
        "stable_count_final": stable,
        "rows": rows,
    }

    save_json(out_dir / "result.json", result)

    print("\n========== V4.14 SITE-TARGET CLOSE/LIFT RESULT ==========")
    print("out              :", rel(out_dir / "result.json"))
    print("ik_grasp_success :", ik_grasp["success"])
    print("ik_lift_success  :", ik_lift["success"])
    print("grip_ready       :", grip_ready)
    print("lifted           :", lifted)
    print("success          :", success)
    print("final_object_rise:", final_rise)
    print("final_object_disp:", final_disp)
    print("final_groups     :", final_groups)
    print("=========================================================")

    if args.viewer:
        print("[VIEWER] live run finished. Keep window open; close viewer window or Ctrl+C.")
        while LIVE_VIEWER is not None and LIVE_VIEWER.is_running():
            live_sync()


if __name__ == "__main__":
    main()
