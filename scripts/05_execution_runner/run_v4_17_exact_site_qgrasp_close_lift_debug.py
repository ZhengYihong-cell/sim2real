#!/usr/bin/env python3
"""
脚本类型：
    debug / execution-runner / v4.17 / exact-site-qgrasp-close-lift

用途：
    验证“已知物体位姿 + 数据集先验手位姿”后，直接 site IK 到 q_grasp，
    然后在 q_grasp 上闭合并 lift。

    本脚本用于废弃 V4.16 中错误的 pregrasp/contact-seek/q_hold 逻辑。
    V4.16 的错误是：
        1. 算出了正确 ik_grasp；
        2. 但 approach 阶段一检测到 object near/contact 就提前停止；
        3. 后续 close 使用提前停止的 q_hold，而不是 ik_grasp.q_arm；
        4. 导致视觉上手距/高度不对。

    V4.17 改为：
        1. 使用 T_world_object @ T_object_hand_base_link 作为 dataset_hand_base_debug 目标；
        2. site IK 得到 q_grasp；
        3. 从当前 home 平滑运动到 q_grasp，不提前停止；
        4. 在 q_grasp 保持一段时间；
        5. 手指闭合阶段只冻结碰到 support/垫块的 finger group；
        6. 碰到 object 的手指不冻结，继续参与夹持；
        7. thumb + 至少一根非拇指 object 接触稳定后，执行 lift。

输入：
    --model          推荐使用 scene_v415_stiff_contact.xml
    --npy            object.npy
    --sample-index   valid local sample index
    --object-body    grasp_object
    --target-site    dataset_hand_base_debug

输出：
    out_dir/result.json
    out_dir/terminal.txt

当前流程位置：
    V4.14 已验证 site target frame 正确；
    V4.15 stiff-contact scene 已修正 object-support 动态穿透；
    V4.17 只验证 exact q_grasp close/lift，不引入 pregrasp/contact seek。

不负责：
    1. 不走旧 fr3_link7 target；
    2. 不做 pregrasp；
    3. 不做 object-contact 提前停止；
    4. 不做 RRT/Pinocchio 全路径规划；
    5. 不换 sample；
    6. 不做沿某个固定轴的人工微调。
"""

from pathlib import Path
import argparse
import json
import time
import numpy as np
import mujoco
import mujoco.viewer


PROJECT = Path.home() / "Projects/o7_mujoco_sim"

ARM_JOINTS = [
    "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
    "fr3_joint5", "fr3_joint6", "fr3_joint7",
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

FINGER_GROUP_TO_JOINTS = {
    "thumb": ["thumb_cmc_roll", "thumb_cmc_yaw", "thumb_cmc_pitch"],
    "index": ["index_mcp_pitch"],
    "middle": ["middle_mcp_pitch"],
    "ring": ["ring_mcp_pitch"],
    "pinky": ["pinky_mcp_pitch"],
}

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


def clamp_joint(model, joint_name, value):
    jid = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    value = float(value)
    if jid >= 0 and int(model.jnt_limited[jid]):
        lo, hi = model.jnt_range[jid]
        value = float(np.clip(value, lo, hi))
    return value


def clamp_ctrl(model, aid, value):
    value = float(value)
    if aid is not None and int(model.actuator_ctrllimited[aid]):
        lo, hi = model.actuator_ctrlrange[aid]
        value = float(np.clip(value, lo, hi))
    return value


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


def set_qpos_once(model, data, q_arm, hand_ctrl):
    for j, v in q_arm.items():
        set_joint_qpos(model, data, j, v)
        da = joint_dof_addr(model, j)
        if da is not None:
            data.qvel[da] = 0.0

    for j, v in hand_ctrl.items():
        set_joint_qpos(model, data, j, v)
        da = joint_dof_addr(model, j)
        if da is not None:
            data.qvel[da] = 0.0

    mujoco.mj_forward(model, data)


def apply_ctrl(model, data, q_arm, hand_ctrl):
    for j, v in q_arm.items():
        set_joint_ctrl(model, data, j, v)
    for j, v in hand_ctrl.items():
        set_joint_ctrl(model, data, j, v)


def get_joint_values(model, data, names):
    out = {}
    for n in names:
        adr = joint_qpos_addr(model, n)
        if adr is not None:
            out[n] = float(data.qpos[adr])
    return out


def interp_dict(a, b, alpha, keys):
    out = {}
    for k in keys:
        v0 = float(a.get(k, b.get(k, 0.0)))
        v1 = float(b.get(k, v0))
        out[k] = (1.0 - alpha) * v0 + alpha * v1
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


def body_world_T(model, data, body_name):
    bid = name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise RuntimeError(f"missing body: {body_name}")
    R = np.asarray(data.xmat[bid], dtype=float).reshape(3, 3)
    p = np.asarray(data.xpos[bid], dtype=float)
    return T_from_Rp(R, p)


def object_pos(model, data, object_body):
    bid = name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
    return np.asarray(data.xpos[bid], dtype=float).copy()


def site_world_T(model, data, site_name):
    sid = name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0:
        names = []
        for i in range(model.nsite):
            n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
            if n:
                names.append(n)
        raise RuntimeError(f"missing site: {site_name}; available={names}")
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


def solve_site_ik(model, site_name, T_target, q_seed,
                  max_iters=350, damping=1e-4, step_scale=0.85,
                  rot_weight=0.65, pos_tol=8e-4, rot_tol=8e-3):
    data = mujoco.MjData(model)
    for j, v in q_seed.items():
        set_joint_qpos(model, data, j, v)
    mujoco.mj_forward(model, data)

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
        history.append({"iter": it, "pos_err_norm": pos_n, "rot_err_norm": rot_n})

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


def support_geom_ids(model):
    out = set()
    for gid in range(model.ngeom):
        s = (geom_name(model, gid) + " " + geom_body_name(model, gid)).lower()
        if "world_plane" in s or "floor" in s:
            continue
        if "object_pedestal" in s or "pedestal" in s or "support" in s:
            out.add(gid)
    return out


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
    return None


def contact_state(model, data, object_body):
    obj = object_geom_ids(model, object_body)
    sup = support_geom_ids(model)

    object_groups = {}
    support_groups = {}
    object_contacts = []
    support_contacts = []

    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)

        kind = None
        other = None

        if g1 in obj:
            kind = "object"
            other = g2
        elif g2 in obj:
            kind = "object"
            other = g1
        elif g1 in sup:
            kind = "support"
            other = g2
        elif g2 in sup:
            kind = "support"
            other = g1

        if kind is None:
            continue

        grp = classify_hand_group(geom_name(model, other) + " " + geom_body_name(model, other))
        if grp is None:
            continue

        row = {
            "group": grp,
            "kind": kind,
            "dist": float(c.dist),
            "hand_geom": geom_name(model, other),
            "other_geom_1": geom_name(model, g1),
            "other_geom_2": geom_name(model, g2),
        }

        if kind == "object":
            object_groups[grp] = object_groups.get(grp, 0) + 1
            object_contacts.append(row)
        else:
            support_groups[grp] = support_groups.get(grp, 0) + 1
            support_contacts.append(row)

    return {
        "object_groups": object_groups,
        "support_groups": support_groups,
        "object_contacts": object_contacts,
        "support_contacts": support_contacts,
        "ncon": int(data.ncon),
    }


def ready_from_groups(groups):
    return ("thumb" in groups) and any(g in groups for g in NON_THUMB)


def live_sync(viewer, sleep_s):
    if viewer is not None and viewer.is_running():
        viewer.sync()
        time.sleep(float(sleep_s))


def step_ctrl(model, data, q_arm, hand_ctrl, viewer, live_sleep):
    apply_ctrl(model, data, q_arm, hand_ctrl)
    mujoco.mj_step(model, data)
    live_sync(viewer, live_sleep)


def move_smooth_to_qgrasp(model, data, viewer, args, q_start, q_grasp, hand_ctrl, object_start):
    rows = []

    print("\n[PHASE] exact smooth move current -> q_grasp")
    for k in range(args.move_steps):
        alpha = k / max(1, args.move_steps - 1)
        q_cmd = interp_dict(q_start, q_grasp, alpha, ARM_JOINTS)

        step_ctrl(model, data, q_cmd, hand_ctrl, viewer, args.live_sleep)

        st = contact_state(model, data, args.object_body)
        obj_p = object_pos(model, data, args.object_body)
        obj_disp = float(np.linalg.norm(obj_p - object_start))

        T_site = site_world_T(model, data, args.target_site)

        if k % args.print_every == 0 or k == args.move_steps - 1:
            print(
                f"[move] {k}/{args.move_steps} alpha={alpha:.3f} "
                f"obj_disp={obj_disp:.5f} obj={st['object_groups']} support={st['support_groups']}"
            )

        rows.append({
            "phase": "move_to_exact_qgrasp",
            "step": k,
            "alpha": alpha,
            "object_pos": obj_p.tolist(),
            "object_disp": obj_disp,
            "object_groups": st["object_groups"],
            "support_groups": st["support_groups"],
            "site_pos": T_site[:3, 3].tolist(),
        })

    return rows


def hold_at_qgrasp(model, data, viewer, args, q_grasp, hand_ctrl, object_start, phase, steps):
    rows = []
    stable = 0
    max_stable = 0

    print(f"\n[PHASE] {phase}")
    for k in range(steps):
        step_ctrl(model, data, q_grasp, hand_ctrl, viewer, args.live_sleep)

        st = contact_state(model, data, args.object_body)
        ready = ready_from_groups(st["object_groups"])
        stable = stable + 1 if ready else 0
        max_stable = max(max_stable, stable)

        obj_p = object_pos(model, data, args.object_body)
        obj_disp = float(np.linalg.norm(obj_p - object_start))

        if k % args.print_every == 0 or k == steps - 1:
            print(
                f"[{phase}] {k}/{steps} obj_disp={obj_disp:.5f} "
                f"obj={st['object_groups']} support={st['support_groups']} ready={ready} stable={stable}"
            )

        rows.append({
            "phase": phase,
            "step": k,
            "object_pos": obj_p.tolist(),
            "object_disp": obj_disp,
            "object_groups": st["object_groups"],
            "support_groups": st["support_groups"],
            "ready": bool(ready),
            "stable": int(stable),
        })

    return rows, stable, max_stable


def support_aware_close(model, data, viewer, args, q_grasp, side_open, close_ctrl, object_start):
    group_alpha = {g: 0.0 for g in FINGER_GROUP_TO_JOINTS}
    frozen = {g: False for g in FINGER_GROUP_TO_JOINTS}
    freeze_reason = {}

    hand_ctrl = dict(side_open)
    rows = []
    stable = 0
    max_stable = 0

    print("\n[PHASE] close at exact q_grasp, only support contact freezes finger")

    for k in range(args.close_steps):
        st_before = contact_state(model, data, args.object_body)

        for g in FINGER_GROUP_TO_JOINTS:
            if frozen[g]:
                continue

            if g in st_before["support_groups"]:
                frozen[g] = True
                freeze_reason[g] = {
                    "reason": "support_contact_freeze",
                    "step": k,
                    "support_groups": dict(st_before["support_groups"]),
                }
                continue

            group_alpha[g] = min(1.0, group_alpha[g] + 1.0 / max(1, args.close_steps))

        hand_ctrl = dict(side_open)
        for g, joints in FINGER_GROUP_TO_JOINTS.items():
            a = float(group_alpha[g])
            for j in joints:
                v0 = float(side_open.get(j, close_ctrl.get(j, 0.0)))
                v1 = float(close_ctrl.get(j, v0))
                hand_ctrl[j] = (1.0 - a) * v0 + a * v1

        step_ctrl(model, data, q_grasp, hand_ctrl, viewer, args.live_sleep)

        st = contact_state(model, data, args.object_body)
        ready = ready_from_groups(st["object_groups"])
        stable = stable + 1 if ready else 0
        max_stable = max(max_stable, stable)

        obj_p = object_pos(model, data, args.object_body)
        obj_disp = float(np.linalg.norm(obj_p - object_start))

        if k % args.print_every == 0 or k == args.close_steps - 1:
            print(
                f"[close] {k}/{args.close_steps} alpha={group_alpha} frozen={frozen} "
                f"obj_disp={obj_disp:.5f} obj={st['object_groups']} support={st['support_groups']} "
                f"ready={ready} stable={stable}"
            )

        rows.append({
            "phase": "support_aware_close_at_exact_qgrasp",
            "step": k,
            "object_pos": obj_p.tolist(),
            "object_disp": obj_disp,
            "object_groups": st["object_groups"],
            "support_groups": st["support_groups"],
            "group_alpha": dict(group_alpha),
            "frozen": dict(frozen),
            "freeze_reason": dict(freeze_reason),
            "hand_ctrl": dict(hand_ctrl),
            "ready": bool(ready),
            "stable": int(stable),
        })

        if stable >= args.ready_stable_steps:
            print(f"[READY] stable={stable}; stop close and start lift.")
            break

    return {
        "rows": rows,
        "stable": stable,
        "max_stable": max_stable,
        "group_alpha": group_alpha,
        "frozen": frozen,
        "freeze_reason": freeze_reason,
        "hand_ctrl": hand_ctrl,
        "last_state": contact_state(model, data, args.object_body),
    }


def lift_with_fixed_grip(model, data, viewer, args, q_grasp, q_lift, hand_ctrl, object_start):
    rows = []
    stable = 0
    max_stable = 0

    print("\n[PHASE] lift with exact q_grasp -> q_lift and fixed hand ctrl")
    for k in range(args.lift_steps):
        alpha = k / max(1, args.lift_steps - 1)
        q_cmd = interp_dict(q_grasp, q_lift, alpha, ARM_JOINTS)

        step_ctrl(model, data, q_cmd, hand_ctrl, viewer, args.live_sleep)

        st = contact_state(model, data, args.object_body)
        ready = ready_from_groups(st["object_groups"])
        stable = stable + 1 if ready else 0
        max_stable = max(max_stable, stable)

        obj_p = object_pos(model, data, args.object_body)
        rise = float(obj_p[2] - object_start[2])
        disp = float(np.linalg.norm(obj_p - object_start))

        if k % args.print_every == 0 or k == args.lift_steps - 1:
            print(
                f"[lift] {k}/{args.lift_steps} alpha={alpha:.3f} rise={rise:.5f} disp={disp:.5f} "
                f"obj={st['object_groups']} support={st['support_groups']} ready={ready} stable={stable}"
            )

        rows.append({
            "phase": "lift_exact_qgrasp_to_qlift",
            "step": k,
            "alpha": alpha,
            "object_pos": obj_p.tolist(),
            "rise": rise,
            "disp": disp,
            "object_groups": st["object_groups"],
            "support_groups": st["support_groups"],
            "ready": bool(ready),
            "stable": int(stable),
        })

    return rows, stable, max_stable


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--npy", required=True)
    ap.add_argument("--sample-index", type=int, required=True)
    ap.add_argument("--object-body", default="grasp_object")
    ap.add_argument("--target-site", default="dataset_hand_base_debug")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--live-sleep", type=float, default=0.002)

    ap.add_argument("--settle-steps", type=int, default=1000)
    ap.add_argument("--move-steps", type=int, default=1200)
    ap.add_argument("--grasp-hold-steps", type=int, default=500)
    ap.add_argument("--close-steps", type=int, default=900)
    ap.add_argument("--post-close-steps", type=int, default=350)
    ap.add_argument("--lift-steps", type=int, default=1500)
    ap.add_argument("--final-hold-steps", type=int, default=400)

    ap.add_argument("--lift-z", type=float, default=0.09)
    ap.add_argument("--ready-stable-steps", type=int, default=5)
    ap.add_argument("--print-every", type=int, default=100)
    args = ap.parse_args()

    model_path = resolve(args.model)
    npy_path = resolve(args.npy)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    sample = load_sample(npy_path, args.sample_index)
    T_object_hand = sample_T_object_hand(sample)
    close_ctrl = sample_ctrl(sample)
    side_open = side_open_from_close(close_ctrl)

    print("========== V4.17 EXACT SITE QGRASP CLOSE/LIFT ==========")
    print("model       :", rel(model_path))
    print("npy         :", rel(npy_path))
    print("sample_index:", args.sample_index)
    print("object_body :", args.object_body)
    print("target_site :", args.target_site)

    set_qpos_once(model, data, Q_HOME, side_open)

    viewer = None
    if args.viewer:
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.lookat[:] = [0.455, 0.0, 0.30]
        viewer.cam.distance = 0.75
        viewer.cam.azimuth = 130
        viewer.cam.elevation = -25
        live_sync(viewer, args.live_sleep)

    print("\n[PHASE] settle object with stiff-contact scene")
    for k in range(args.settle_steps):
        step_ctrl(model, data, Q_HOME, side_open, viewer, args.live_sleep)
        if k % args.print_every == 0 or k == args.settle_steps - 1:
            print(f"[settle] {k}/{args.settle_steps} object_pos={object_pos(model, data, args.object_body).tolist()}")

    object_start = object_pos(model, data, args.object_body)
    T_world_object = body_world_T(model, data, args.object_body)
    T_grasp = T_world_object @ T_object_hand

    T_lift = np.array(T_grasp, dtype=float)
    T_lift[2, 3] += float(args.lift_z)

    print("\n[IK exact target]")
    ik_grasp = solve_site_ik(model, args.target_site, T_grasp, Q_HOME)
    ik_lift = solve_site_ik(model, args.target_site, T_lift, ik_grasp["q_arm"])

    print("T_world_object.pos:", T_world_object[:3, 3].tolist())
    print("T_object_hand.pos :", T_object_hand[:3, 3].tolist())
    print("T_grasp.pos       :", T_grasp[:3, 3].tolist())
    print("ik_grasp_success  :", ik_grasp["success"], "pos_err:", ik_grasp["final_pos_err_norm"], "rot_err:", ik_grasp["final_rot_err_norm"])
    print("ik_lift_success   :", ik_lift["success"], "pos_err:", ik_lift["final_pos_err_norm"], "rot_err:", ik_lift["final_rot_err_norm"])

    if not ik_grasp["success"]:
        raise RuntimeError("ik_grasp failed; cannot execute.")

    q_current = get_joint_values(model, data, ARM_JOINTS)

    move_rows = move_smooth_to_qgrasp(
        model, data, viewer, args,
        q_current,
        ik_grasp["q_arm"],
        side_open,
        object_start,
    )

    # 到 q_grasp 后保持一段时间，让你能看清楚手是否到了 V14 验证的目标位姿。
    hold_rows, hold_stable, hold_max_stable = hold_at_qgrasp(
        model, data, viewer, args,
        ik_grasp["q_arm"],
        side_open,
        object_start,
        "hold_exact_qgrasp_side_open",
        args.grasp_hold_steps,
    )

    close_info = support_aware_close(
        model, data, viewer, args,
        ik_grasp["q_arm"],
        side_open,
        close_ctrl,
        object_start,
    )

    post_rows, post_stable, post_max_stable = hold_at_qgrasp(
        model, data, viewer, args,
        ik_grasp["q_arm"],
        close_info["hand_ctrl"],
        object_start,
        "post_close_hold_exact_qgrasp",
        args.post_close_steps,
    )

    grip_ready = (
        close_info["stable"] >= args.ready_stable_steps
        or post_stable >= args.ready_stable_steps
        or ready_from_groups(contact_state(model, data, args.object_body)["object_groups"])
    )

    lifted = False
    lift_rows = []
    lift_stable = 0
    lift_max_stable = 0

    if grip_ready and ik_lift["success"]:
        lift_rows, lift_stable, lift_max_stable = lift_with_fixed_grip(
            model, data, viewer, args,
            ik_grasp["q_arm"],
            ik_lift["q_arm"],
            close_info["hand_ctrl"],
            object_start,
        )
        lifted = True

        final_hold_rows, final_stable, final_max_stable = hold_at_qgrasp(
            model, data, viewer, args,
            ik_lift["q_arm"],
            close_info["hand_ctrl"],
            object_start,
            "final_hold_after_lift",
            args.final_hold_steps,
        )
    else:
        print("[NO_LIFT] grip_ready false or ik_lift failed.")
        final_hold_rows = []
        final_stable = 0
        final_max_stable = 0

    final_pos = object_pos(model, data, args.object_body)
    final_state = contact_state(model, data, args.object_body)
    final_rise = float(final_pos[2] - object_start[2])
    final_disp = float(np.linalg.norm(final_pos - object_start))
    final_groups = final_state["object_groups"]

    success = bool(lifted and final_rise > 0.03 and ready_from_groups(final_groups))

    result = {
        "format": "v4_17_exact_site_qgrasp_close_lift_debug_v1",
        "model": rel(model_path),
        "npy": rel(npy_path),
        "sample_index_valid_local": args.sample_index,
        "object_body": args.object_body,
        "target_site": args.target_site,
        "T_world_object_after_settle": mat_to_dict(T_world_object),
        "T_object_hand_base_from_dataset": mat_to_dict(T_object_hand),
        "T_grasp": mat_to_dict(T_grasp),
        "T_lift": mat_to_dict(T_lift),
        "ik_grasp": ik_grasp,
        "ik_lift": ik_lift,
        "side_open_ctrl": side_open,
        "close_ctrl": close_ctrl,
        "final_hand_ctrl": close_info["hand_ctrl"],
        "group_alpha": close_info["group_alpha"],
        "frozen": close_info["frozen"],
        "freeze_reason": close_info["freeze_reason"],
        "object_start": object_start.tolist(),
        "final_object_pos": final_pos.tolist(),
        "final_object_rise": final_rise,
        "final_object_disp": final_disp,
        "final_object_groups": final_groups,
        "final_support_groups": final_state["support_groups"],
        "grip_ready": grip_ready,
        "lifted": lifted,
        "success": success,
        "move_rows": move_rows,
        "hold_rows": hold_rows,
        "close_rows": close_info["rows"],
        "post_rows": post_rows,
        "lift_rows": lift_rows,
        "final_hold_rows": final_hold_rows,
    }

    save_json(out_dir / "result.json", result)

    print("\n========== V4.17 RESULT ==========")
    print("out:", rel(out_dir / "result.json"))
    print("grip_ready:", grip_ready)
    print("lifted:", lifted)
    print("success:", success)
    print("final_object_rise:", final_rise)
    print("final_object_disp:", final_disp)
    print("final_object_groups:", final_groups)
    print("final_support_groups:", final_state["support_groups"])
    print("frozen:", close_info["frozen"])
    print("freeze_reason:", close_info["freeze_reason"])
    print("==================================")

    if args.viewer:
        print("[VIEWER] live run finished. Keep open; close viewer or Ctrl+C.")
        while viewer is not None and viewer.is_running():
            live_sync(viewer, args.live_sleep)


if __name__ == "__main__":
    main()
