#!/usr/bin/env python3
"""
脚本类型：
    debug / execution-runner / v4.15 / site-target-contact-aware-close-lift

用途：
    在正确 site-target frame 链路下，验证单个 sample 的接触感知闭合和 lift。
    本脚本直接使用：
        T_world_dataset_hand_base_debug = T_world_object @ T_object_hand_base_link
    对 dataset_hand_base_debug 做 IK，得到 q_grasp / q_lift。

    与 V4.14 的关键区别：
        1. 动态阶段不再每步强行 set qpos；
        2. 手指按 group 独立闭合；
        3. 某个手指 group 碰到 support，则该 group 冻结；
        4. 某个手指 group 碰到 object，也可冻结，防止继续硬压；
        5. 其他未冻结手指继续闭合；
        6. thumb + 任意非拇指形成 object 接触后，尝试 lift。

输入：
    --model          推荐使用 scene_v415_stiff_contact.xml
    --npy            object.npy
    --sample-index   valid local sample index
    --object-body    MuJoCo object body，例如 grasp_object
    --target-site    dataset_hand_base_debug

输出：
    out_dir/result.json
    out_dir/terminal.txt

当前流程位置：
    V4.14 已证明 site-target frame 正确；
    V4.15 stiff-contact scene 已证明 object-support 动态 settle 不再穿透；
    本脚本验证 contact-aware close/lift。

不负责：
    1. 不走旧 fr3_link7 target；
    2. 不做完整 approach path；
    3. 不做 selector；
    4. 不换 sample；
    5. 不做沿轴人工微调。
"""

from pathlib import Path
import argparse
import json
import math
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


def hand_group_geom_ids(model):
    groups = {g: set() for g in FINGER_GROUP_TO_JOINTS.keys()}
    for gid in range(model.ngeom):
        name = geom_name(model, gid) + " " + geom_body_name(model, gid)
        grp = classify_hand_group(name)
        if grp in groups:
            groups[grp].add(gid)
    return groups


def min_geom_distance(model, data, geoms_a, geoms_b, distmax=0.20):
    best = None
    fromto = np.zeros(6, dtype=float)
    for ga in geoms_a:
        for gb in geoms_b:
            try:
                d = float(mujoco.mj_geomDistance(model, data, int(ga), int(gb), float(distmax), fromto))
            except Exception:
                continue
            if best is None or d < best["distance"]:
                best = {
                    "distance": d,
                    "geom_a": geom_name(model, ga),
                    "geom_b": geom_name(model, gb),
                }
    return best


def contact_state(model, data, object_body):
    obj = object_geom_ids(model, object_body)
    sup = support_geom_ids(model)
    hgroups = hand_group_geom_ids(model)

    object_groups = {}
    support_groups = {}
    object_contacts = []
    support_contacts = []

    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)

        other = None
        if g1 in obj:
            other = g2
            kind = "object"
        elif g2 in obj:
            other = g1
            kind = "object"
        elif g1 in sup:
            other = g2
            kind = "support"
        elif g2 in sup:
            other = g1
            kind = "support"
        else:
            continue

        grp = classify_hand_group(geom_name(model, other) + " " + geom_body_name(model, other))
        if grp is None:
            continue

        row = {
            "group": grp,
            "dist": float(c.dist),
            "hand_geom": geom_name(model, other),
            "kind": kind,
        }

        if kind == "object":
            object_groups[grp] = object_groups.get(grp, 0) + 1
            object_contacts.append(row)
        else:
            support_groups[grp] = support_groups.get(grp, 0) + 1
            support_contacts.append(row)

    # 几何距离兜底：防止 margin/contact 设置导致 ncon 不稳定
    object_min = {}
    support_min = {}
    for grp, geoms in hgroups.items():
        bo = min_geom_distance(model, data, geoms, obj)
        bs = min_geom_distance(model, data, geoms, sup)
        if bo:
            object_min[grp] = bo
        if bs:
            support_min[grp] = bs

    return {
        "object_groups": object_groups,
        "support_groups": support_groups,
        "object_contacts": object_contacts,
        "support_contacts": support_contacts,
        "object_min_distance": object_min,
        "support_min_distance": support_min,
    }


def ready_from_object_groups(groups):
    return ("thumb" in groups) and any(g in groups for g in NON_THUMB)


def init_hand_ctrl(side_open):
    return dict(side_open)


def group_alpha_ctrl(side_open, close_ctrl, alpha):
    ctrl = dict(side_open)
    for g, joints in FINGER_GROUP_TO_JOINTS.items():
        for j in joints:
            a = float(side_open.get(j, close_ctrl.get(j, 0.0)))
            b = float(close_ctrl.get(j, a))
            ctrl[j] = (1.0 - alpha) * a + alpha * b
    return ctrl


def update_contact_aware_ctrl(model, data, object_body, side_open, close_ctrl,
                              group_alpha, frozen, freeze_reason,
                              alpha_step, support_stop_dist, object_stop_dist,
                              freeze_on_object):
    st = contact_state(model, data, object_body)

    for grp in FINGER_GROUP_TO_JOINTS.keys():
        if frozen.get(grp, False):
            continue

        support_near = st["support_min_distance"].get(grp)
        object_near = st["object_min_distance"].get(grp)

        support_d = None if support_near is None else support_near["distance"]
        object_d = None if object_near is None else object_near["distance"]

        if support_d is not None and support_d <= support_stop_dist:
            frozen[grp] = True
            freeze_reason[grp] = {
                "reason": "support_contact_or_near",
                "distance": support_d,
                "pair": support_near,
            }
            continue

        if freeze_on_object and object_d is not None and object_d <= object_stop_dist:
            frozen[grp] = True
            freeze_reason[grp] = {
                "reason": "object_contact_or_near",
                "distance": object_d,
                "pair": object_near,
            }
            continue

        group_alpha[grp] = min(1.0, float(group_alpha.get(grp, 0.0)) + alpha_step)

    ctrl = dict(side_open)
    for grp, joints in FINGER_GROUP_TO_JOINTS.items():
        a = float(group_alpha.get(grp, 0.0))
        for j in joints:
            v0 = float(side_open.get(j, close_ctrl.get(j, 0.0)))
            v1 = float(close_ctrl.get(j, v0))
            ctrl[j] = (1.0 - a) * v0 + a * v1

    return ctrl, st, group_alpha, frozen, freeze_reason


def interp_arm(q0, q1, alpha):
    out = {}
    for j in ARM_JOINTS:
        a = float(q0.get(j, q1.get(j, 0.0)))
        b = float(q1.get(j, a))
        out[j] = (1.0 - alpha) * a + alpha * b
    return out


def live_setup(args, model, data):
    if not args.viewer:
        return None
    viewer = mujoco.viewer.launch_passive(model, data)
    viewer.cam.lookat[:] = [0.455, 0.0, 0.30]
    viewer.cam.distance = 0.75
    viewer.cam.azimuth = 130
    viewer.cam.elevation = -25
    return viewer


def live_sync(viewer, sleep_s):
    if viewer is not None and viewer.is_running():
        viewer.sync()
        time.sleep(sleep_s)


def step_ctrl(model, data, q_arm_ctrl, hand_ctrl, viewer, args):
    apply_ctrl(model, data, q_arm_ctrl, hand_ctrl)
    mujoco.mj_step(model, data)
    live_sync(viewer, args.live_sleep)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--npy", required=True)
    ap.add_argument("--sample-index", type=int, required=True)
    ap.add_argument("--object-body", default="grasp_object")
    ap.add_argument("--target-site", default="dataset_hand_base_debug")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--live-sleep", type=float, default=0.0015)

    ap.add_argument("--settle-steps", type=int, default=1200)
    ap.add_argument("--grasp-hold-steps", type=int, default=300)
    ap.add_argument("--close-steps", type=int, default=900)
    ap.add_argument("--post-close-steps", type=int, default=350)
    ap.add_argument("--lift-steps", type=int, default=1800)
    ap.add_argument("--final-hold-steps", type=int, default=500)
    ap.add_argument("--lift-z", type=float, default=0.09)

    ap.add_argument("--support-stop-dist", type=float, default=0.0008)
    ap.add_argument("--object-stop-dist", type=float, default=0.0008)
    ap.add_argument("--freeze-on-object-contact", action="store_true")
    ap.add_argument("--ready-stable-steps", type=int, default=4)
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

    print("========== V4.15 SITE-TARGET CONTACT-AWARE CLOSE/LIFT ==========")
    print("model       :", rel(model_path))
    print("npy         :", rel(npy_path))
    print("sample_index:", args.sample_index)
    print("object_body :", args.object_body)
    print("target_site :", args.target_site)

    # 初始化：只设置一次 qpos，让物体动态 settle
    set_qpos_once(model, data, Q_HOME, side_open)
    viewer = live_setup(args, model, data)

    print("\n[PHASE] settle object with stiff contact scene")
    for k in range(args.settle_steps):
        step_ctrl(model, data, Q_HOME, side_open, viewer, args)
        if k % args.print_every == 0 or k == args.settle_steps - 1:
            st = contact_state(model, data, args.object_body)
            print(f"[settle] {k}/{args.settle_steps} object_pos={object_pos(model, data, args.object_body).tolist()} support_groups={st['support_groups']}")

    T_world_object = body_world_T(model, data, args.object_body)
    T_world_target = T_world_object @ T_object_hand

    ik_grasp = solve_site_ik(model, args.target_site, T_world_target, Q_HOME)
    q_grasp = ik_grasp["q_arm"]

    T_lift = np.array(T_world_target, dtype=float)
    T_lift[2, 3] += float(args.lift_z)
    ik_lift = solve_site_ik(model, args.target_site, T_lift, q_grasp)
    q_lift = ik_lift["q_arm"]

    print("\n[IK]")
    print("ik_grasp_success:", ik_grasp["success"], "pos_err:", ik_grasp["final_pos_err_norm"], "rot_err:", ik_grasp["final_rot_err_norm"])
    print("ik_lift_success :", ik_lift["success"], "pos_err:", ik_lift["final_pos_err_norm"], "rot_err:", ik_lift["final_rot_err_norm"])

    # 只设置一次 q_grasp + open hand，然后后续只发 ctrl
    set_qpos_once(model, data, q_grasp, side_open)

    object_start = object_pos(model, data, args.object_body)
    rows = []
    stable_ready = 0
    max_ready = 0

    print("\n[PHASE] hold at q_grasp side-open, ctrl only")
    for k in range(args.grasp_hold_steps):
        step_ctrl(model, data, q_grasp, side_open, viewer, args)
        st = contact_state(model, data, args.object_body)
        ready = ready_from_object_groups(st["object_groups"])
        stable_ready = stable_ready + 1 if ready else 0
        max_ready = max(max_ready, stable_ready)

        if k % args.print_every == 0 or k == args.grasp_hold_steps - 1:
            print(f"[hold] {k}/{args.grasp_hold_steps} object_groups={st['object_groups']} support_groups={st['support_groups']} ready={ready} stable={stable_ready}")

        rows.append({
            "phase": "hold_side_open",
            "step": k,
            "object_pos": object_pos(model, data, args.object_body).tolist(),
            "state": st,
            "ready": bool(ready),
            "stable_ready": int(stable_ready),
        })

    group_alpha = {g: 0.0 for g in FINGER_GROUP_TO_JOINTS}
    frozen = {g: False for g in FINGER_GROUP_TO_JOINTS}
    freeze_reason = {}
    last_ctrl = dict(side_open)
    alpha_step = 1.0 / max(1, args.close_steps)

    print("\n[PHASE] contact-aware close")
    for k in range(args.close_steps):
        last_ctrl, st, group_alpha, frozen, freeze_reason = update_contact_aware_ctrl(
            model, data, args.object_body,
            side_open, close_ctrl,
            group_alpha, frozen, freeze_reason,
            alpha_step,
            args.support_stop_dist,
            args.object_stop_dist,
            args.freeze_on_object_contact,
        )

        step_ctrl(model, data, q_grasp, last_ctrl, viewer, args)

        st = contact_state(model, data, args.object_body)
        ready = ready_from_object_groups(st["object_groups"])
        stable_ready = stable_ready + 1 if ready else 0
        max_ready = max(max_ready, stable_ready)

        if k % args.print_every == 0 or k == args.close_steps - 1:
            print(
                f"[close] {k}/{args.close_steps} alpha={group_alpha} frozen={frozen} "
                f"obj={st['object_groups']} support={st['support_groups']} ready={ready} stable={stable_ready}"
            )

        rows.append({
            "phase": "contact_aware_close",
            "step": k,
            "object_pos": object_pos(model, data, args.object_body).tolist(),
            "state": st,
            "group_alpha": dict(group_alpha),
            "frozen": dict(frozen),
            "freeze_reason": dict(freeze_reason),
            "hand_ctrl": dict(last_ctrl),
            "ready": bool(ready),
            "stable_ready": int(stable_ready),
        })

        if stable_ready >= args.ready_stable_steps:
            print(f"[READY] stable_ready reached {stable_ready}; stop close and start lift.")
            break

    print("\n[PHASE] post-close hold")
    for k in range(args.post_close_steps):
        step_ctrl(model, data, q_grasp, last_ctrl, viewer, args)
        st = contact_state(model, data, args.object_body)
        ready = ready_from_object_groups(st["object_groups"])
        stable_ready = stable_ready + 1 if ready else 0
        max_ready = max(max_ready, stable_ready)

        if k % args.print_every == 0 or k == args.post_close_steps - 1:
            print(f"[post] {k}/{args.post_close_steps} obj={st['object_groups']} support={st['support_groups']} ready={ready} stable={stable_ready}")

        rows.append({
            "phase": "post_close_hold",
            "step": k,
            "object_pos": object_pos(model, data, args.object_body).tolist(),
            "state": st,
            "frozen": dict(frozen),
            "freeze_reason": dict(freeze_reason),
            "hand_ctrl": dict(last_ctrl),
            "ready": bool(ready),
            "stable_ready": int(stable_ready),
        })

    final_close_state = contact_state(model, data, args.object_body)
    grip_ready = stable_ready >= args.ready_stable_steps and ready_from_object_groups(final_close_state["object_groups"])

    print("\n[READY CHECK]")
    print("grip_ready:", grip_ready)
    print("object_groups:", final_close_state["object_groups"])
    print("support_groups:", final_close_state["support_groups"])
    print("frozen:", frozen)
    print("freeze_reason:", freeze_reason)

    lifted = False
    if grip_ready and ik_lift["success"]:
        print("\n[PHASE] lift, ctrl only, frozen hand ctrl")
        for k in range(args.lift_steps):
            alpha = k / max(1, args.lift_steps - 1)
            q_arm = interp_arm(q_grasp, q_lift, alpha)
            step_ctrl(model, data, q_arm, last_ctrl, viewer, args)

            st = contact_state(model, data, args.object_body)
            ready = ready_from_object_groups(st["object_groups"])
            stable_ready = stable_ready + 1 if ready else 0
            max_ready = max(max_ready, stable_ready)

            if k % args.print_every == 0 or k == args.lift_steps - 1:
                print(f"[lift] {k}/{args.lift_steps} alpha={alpha:.3f} obj={st['object_groups']} support={st['support_groups']} ready={ready} stable={stable_ready}")

            rows.append({
                "phase": "lift",
                "step": k,
                "alpha": alpha,
                "object_pos": object_pos(model, data, args.object_body).tolist(),
                "state": st,
                "hand_ctrl": dict(last_ctrl),
                "ready": bool(ready),
                "stable_ready": int(stable_ready),
            })

        lifted = True

        print("\n[PHASE] final hold")
        for k in range(args.final_hold_steps):
            step_ctrl(model, data, q_lift, last_ctrl, viewer, args)
            st = contact_state(model, data, args.object_body)

            if k % args.print_every == 0 or k == args.final_hold_steps - 1:
                print(f"[final] {k}/{args.final_hold_steps} obj={st['object_groups']} support={st['support_groups']} object_pos={object_pos(model, data, args.object_body).tolist()}")

            rows.append({
                "phase": "final_hold",
                "step": k,
                "object_pos": object_pos(model, data, args.object_body).tolist(),
                "state": st,
                "hand_ctrl": dict(last_ctrl),
            })
    else:
        print("[NO_LIFT] grip not ready or lift IK failed. This is a real failure, not force-lift.")

    final_pos = object_pos(model, data, args.object_body)
    final_state = contact_state(model, data, args.object_body)
    final_rise = float(final_pos[2] - object_start[2])
    final_disp = float(np.linalg.norm(final_pos - object_start))
    success = bool(lifted and final_rise > 0.03 and ready_from_object_groups(final_state["object_groups"]))

    result = {
        "format": "v4_15_site_target_contact_aware_close_lift_debug_v1",
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
        "side_open_ctrl": side_open,
        "close_ctrl": close_ctrl,
        "final_hand_ctrl": last_ctrl,
        "group_alpha": group_alpha,
        "frozen": frozen,
        "freeze_reason": freeze_reason,
        "grip_ready": grip_ready,
        "lifted": lifted,
        "success": success,
        "object_start": object_start.tolist(),
        "final_object_pos": final_pos.tolist(),
        "final_object_rise": final_rise,
        "final_object_disp": final_disp,
        "final_state": final_state,
        "max_ready_stable_count": max_ready,
        "rows": rows,
    }

    save_json(out_dir / "result.json", result)

    print("\n========== V4.15 RESULT ==========")
    print("out:", rel(out_dir / "result.json"))
    print("grip_ready:", grip_ready)
    print("lifted:", lifted)
    print("success:", success)
    print("final_object_rise:", final_rise)
    print("final_object_disp:", final_disp)
    print("final_object_groups:", final_state["object_groups"])
    print("final_support_groups:", final_state["support_groups"])
    print("frozen:", frozen)
    print("freeze_reason:", freeze_reason)
    print("==================================")

    if args.viewer:
        print("[VIEWER] live run finished. Keep open; close viewer or Ctrl+C.")
        while viewer is not None and viewer.is_running():
            live_sync(viewer, args.live_sleep)


if __name__ == "__main__":
    main()
