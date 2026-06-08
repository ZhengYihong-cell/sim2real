#!/usr/bin/env python3
"""
V4.20c real-hand-keypoint proxy selector.

定位：
    lightweight prior selector / keypoint-proxy calibration / Top-K high recall.
    不是最终执行器，不负责 close/lift，不救单个 sample。

核心目的：
    修正 V4.20b 的最大问题：
        hardcoded hand proxy 与真实 O7 手几何/坐标系可能不一致，
        导致所有 sample 都被判定 thumb_far / nonthumb_far。

做法：
    1. 读取 FR3+O7 MuJoCo model；
    2. 对每个 dataset sample：
        T_world_hand = T_world_object @ T_object_hand
        site IK 到 dataset_hand_base_debug；
        设置 O7 active hand qpos/ctrl；
        从真实 MuJoCo hand geoms 中提取 finger/palm surface proxy；
    3. 用真实 hand geom surface 到 object AABB 的距离估计接触潜力；
    4. 同时计算旧 hardcoded proxy，输出对比；
    5. 输出：
        ranked_real_proxy_candidates.json
        selected_topk_overall.json
        selected_topk_diverse.json
        selected_topk_feasible.json
        real_proxy_report.txt
        real_proxy_compare_debug.json

注意：
    V4.20c 的目标是让 Top-K 里面包含可抓候选；
    不是要求 Top-1 必然成功。
"""

from pathlib import Path
import argparse
import csv
import json
import math
import numpy as np
import mujoco


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
    return p if p.is_absolute() else PROJECT / p


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


def parse_vec2(s):
    v = [float(x) for x in str(s).replace(",", " ").split()]
    if len(v) == 2:
        return np.asarray(v, dtype=float)
    if len(v) == 3:
        return np.asarray(v[:2], dtype=float)
    raise RuntimeError(f"expected 2 or 3 values, got {s}")


def parse_vec3(s):
    v = [float(x) for x in str(s).replace(",", " ").split()]
    if len(v) != 3:
        raise RuntimeError(f"expected 3 values, got {s}")
    return np.asarray(v, dtype=float)


def parse_quat_wxyz(s):
    q = np.asarray([float(x) for x in str(s).replace(",", " ").split()], dtype=float)
    if q.shape[0] != 4:
        raise RuntimeError(f"expected quat wxyz 4 values, got {s}")
    return q / max(float(np.linalg.norm(q)), 1e-12)


def quat_wxyz_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z, 2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [2*x*y + 2*z*w,     1 - 2*x*x - 2*z*z, 2*y*z - 2*x*w],
        [2*x*z - 2*y*w,     2*y*z + 2*x*w,     1 - 2*x*x - 2*y*y],
    ], dtype=float)


def normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    n = float(np.linalg.norm(v))
    if n < eps:
        return v * 0.0
    return v / n


def robust_rot6d_to_R(r6):
    r6 = np.asarray(r6, dtype=float).reshape(6)
    x_raw = r6[:3]
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


def invert_T(T):
    T = np.asarray(T, dtype=float)
    Ti = np.eye(4, dtype=float)
    Ti[:3, :3] = T[:3, :3].T
    Ti[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return Ti


def transform_points(T, pts):
    pts = np.asarray(pts, dtype=float)
    return (T[:3, :3] @ pts.T).T + T[:3, 3]


def read_obj_vertices(path):
    verts = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                ps = line.strip().split()
                if len(ps) >= 4:
                    try:
                        verts.append([float(ps[1]), float(ps[2]), float(ps[3])])
                    except Exception:
                        pass
    if not verts:
        raise RuntimeError(f"no vertices in obj: {path}")
    return np.asarray(verts, dtype=float)


def compute_object_pos_on_support(verts_scaled, R_obj, support_center_xy, support_top_z, clearance):
    verts_rot = (R_obj @ verts_scaled.T).T
    min_z = float(verts_rot[:, 2].min())
    return np.asarray([
        float(support_center_xy[0]),
        float(support_center_xy[1]),
        float(support_top_z + clearance - min_z),
    ], dtype=float)


def load_samples(npy_path):
    arr = np.load(npy_path, allow_pickle=True)
    out = []
    for i in range(len(arr)):
        s = arr[i].item() if hasattr(arr[i], "item") else arr[i]
        if isinstance(s, dict) and "hand_pose" in s:
            out.append((i, s))
    return out


def sample_scale(sample):
    for k in ["scale", "object_scale", "mesh_scale"]:
        if k in sample:
            try:
                return float(sample[k])
            except Exception:
                pass
    return 1.0


def sample_raw_index(local_i, sample):
    for k in ["sample_index", "raw_sample_index", "index", "idx"]:
        if k in sample:
            try:
                return int(sample[k])
            except Exception:
                pass
    return int(local_i)


def extract_T_object_hand(sample):
    hp = np.asarray(sample["hand_pose"], dtype=float)
    if hp.shape[0] < 9:
        raise RuntimeError(f"bad hand_pose shape={hp.shape}")
    return T_from_Rp(robust_rot6d_to_R(hp[3:9]), hp[:3])


def extract_ctrl(sample):
    hp = np.asarray(sample["hand_pose"], dtype=float)
    if hp.shape[0] >= 16:
        return {j: float(v) for j, v in zip(O7_ACTIVE_JOINTS, hp[9:16])}
    return {j: 0.0 for j in O7_ACTIVE_JOINTS}


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


def body_name(model, bid):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(bid)) or f"body_{bid}"


def geom_body_name(model, gid):
    return body_name(model, int(model.geom_bodyid[int(gid)]))


def classify_group_from_name(s):
    s = str(s).lower()

    if "grasp_object" in s or "object_pedestal" in s or "pedestal" in s:
        return None
    if "table" in s or "floor" in s or "plane" in s or "support" in s:
        return None
    if "fr3_" in s or "panda" in s:
        return None

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

    if "hand_base" in s or "palm" in s or "base_link" in s:
        return "palm"

    return None


def geom_effective_radius(model, gid):
    gid = int(gid)
    size = np.asarray(model.geom_size[gid], dtype=float)
    rbound = float(model.geom_rbound[gid]) if hasattr(model, "geom_rbound") else 0.0

    gtype = int(model.geom_type[gid])

    if gtype == int(mujoco.mjtGeom.mjGEOM_SPHERE):
        r = float(size[0])
    elif gtype == int(mujoco.mjtGeom.mjGEOM_CAPSULE):
        r = float(size[0])
    elif gtype == int(mujoco.mjtGeom.mjGEOM_CYLINDER):
        r = float(size[0])
    elif gtype == int(mujoco.mjtGeom.mjGEOM_BOX):
        r = float(np.linalg.norm(size))
    else:
        nz = size[np.isfinite(size)]
        r = float(np.max(np.abs(nz))) if nz.size else 0.0
        if r < 1e-6 and rbound > 0:
            r = rbound

    if not np.isfinite(r) or r <= 0:
        r = 0.008

    return float(np.clip(r, 0.002, 0.040))


def collect_real_hand_geom_records(model, data, use_collision_only=True):
    records = []
    group_counts = {g: 0 for g in ["thumb", "index", "middle", "ring", "pinky", "palm"]}

    for gid in range(model.ngeom):
        gname = geom_name(model, gid)
        bname = geom_body_name(model, gid)
        full = f"{gname} {bname}"

        grp = classify_group_from_name(full)
        if grp is None:
            continue

        if use_collision_only:
            contype = int(model.geom_contype[gid])
            conaff = int(model.geom_conaffinity[gid])
            if contype == 0 and conaff == 0:
                continue

        center = np.asarray(data.geom_xpos[gid], dtype=float).copy()
        radius = geom_effective_radius(model, gid)

        records.append({
            "geom_id": int(gid),
            "geom_name": gname,
            "body_name": bname,
            "group": grp,
            "center_world": center.tolist(),
            "radius": radius,
            "contype": int(model.geom_contype[gid]),
            "conaffinity": int(model.geom_conaffinity[gid]),
            "geom_type": int(model.geom_type[gid]),
        })
        group_counts[grp] += 1

    return records, group_counts


def point_to_aabb_distance(p, bmin, bmax):
    p = np.asarray(p, dtype=float)
    bmin = np.asarray(bmin, dtype=float)
    bmax = np.asarray(bmax, dtype=float)

    outside = np.maximum(np.maximum(bmin - p, p - bmax), 0.0)
    outside_norm = float(np.linalg.norm(outside))
    if outside_norm > 0:
        return outside_norm

    inside_margins = np.minimum(p - bmin, bmax - p)
    return -float(np.min(inside_margins))


def surface_distances_to_object_aabb(records, T_world_object, bbox_min_obj, bbox_max_obj):
    T_obj_world = invert_T(T_world_object)

    group_min = {}
    group_best = {}

    for r in records:
        grp = r["group"]
        c_w = np.asarray(r["center_world"], dtype=float)
        c_obj = transform_points(T_obj_world, c_w.reshape(1, 3))[0]
        center_d = point_to_aabb_distance(c_obj, bbox_min_obj, bbox_max_obj)
        surf_d = float(center_d - float(r["radius"]))

        if grp not in group_min or surf_d < group_min[grp]:
            group_min[grp] = surf_d
            rr = dict(r)
            rr["center_object"] = c_obj.tolist()
            rr["center_to_object_aabb_distance"] = float(center_d)
            rr["surface_to_object_aabb_distance"] = float(surf_d)
            group_best[grp] = rr

    return group_min, group_best


def support_clearance_real(records, support_center_xy, support_half_size,
                           pedestal_top_z, table_top_z, pedestal_xy_margin):
    hx, hy, _ = support_half_size

    min_clear = 999.0
    group_min = {}
    best = {}

    for r in records:
        grp = r["group"]
        c = np.asarray(r["center_world"], dtype=float)
        radius = float(r["radius"])

        inside_ped_xy = (
            abs(c[0] - support_center_xy[0]) <= hx + pedestal_xy_margin and
            abs(c[1] - support_center_xy[1]) <= hy + pedestal_xy_margin
        )

        top_z = float(pedestal_top_z if inside_ped_xy else table_top_z)
        support_name = "pedestal" if inside_ped_xy else "table"

        clear = float(c[2] - radius - top_z)

        if clear < min_clear:
            min_clear = clear

        if grp not in group_min or clear < group_min[grp]:
            group_min[grp] = clear
            rr = dict(r)
            rr["support_name"] = support_name
            rr["surface_clearance_to_support"] = clear
            best[grp] = rr

    return float(min_clear), group_min, best


def make_hardcoded_proxy_points(T_object_hand, ctrl):
    R = T_object_hand[:3, :3]
    p = T_object_hand[:3, 3]
    X, Y, Z = R[:, 0], R[:, 1], R[:, 2]

    pts = {
        "hand_base": p,
        "palm_center": p + 0.030 * Z - 0.018 * X,
        "palm_low": p + 0.020 * Z - 0.055 * X,
    }

    for g, yoff in {
        "index": -0.030,
        "middle": -0.010,
        "ring": 0.010,
        "pinky": 0.030,
    }.items():
        q = float(ctrl.get(f"{g}_mcp_pitch", 0.0))
        bend = np.clip(q / 1.2, 0.0, 1.0)

        pts[f"{g}_base"] = p + 0.045 * Z + yoff * Y - 0.020 * X
        pts[f"{g}_mid"] = p + (0.075 - 0.015*bend) * Z + yoff * Y - (0.025 + 0.025*bend) * X
        pts[f"{g}_tip"] = p + (0.110 - 0.035*bend) * Z + yoff * Y - (0.030 + 0.060*bend) * X

    thumb_pitch = float(ctrl.get("thumb_cmc_pitch", 0.0))
    tb = np.clip((thumb_pitch + 0.2) / 1.3, 0.0, 1.0)
    pts["thumb_base"] = p + 0.025 * Z + 0.055 * Y - 0.020 * X
    pts["thumb_mid"] = p + (0.050 - 0.010*tb) * Z + 0.075 * Y - (0.030 + 0.025*tb) * X
    pts["thumb_tip"] = p + (0.070 - 0.020*tb) * Z + 0.090 * Y - (0.040 + 0.055*tb) * X

    return pts


def hardcoded_group_distances(proxy_obj, bbox_min, bbox_max):
    groups = {g: [] for g in ["thumb", "index", "middle", "ring", "pinky", "palm"]}
    for name, p in proxy_obj.items():
        matched = False
        for g in ["thumb", "index", "middle", "ring", "pinky"]:
            if name.startswith(g):
                groups[g].append(p)
                matched = True
                break
        if not matched:
            groups["palm"].append(p)

    out = {}
    for g, pts in groups.items():
        if not pts:
            continue
        out[g] = float(min(point_to_aabb_distance(p, bbox_min, bbox_max) for p in pts))
    return out


def hardcoded_real_delta_summary(proxy_obj, real_records, T_world_object):
    T_obj_world = invert_T(T_world_object)
    real_by_group = {g: [] for g in ["thumb", "index", "middle", "ring", "pinky", "palm"]}

    for r in real_records:
        grp = r["group"]
        c_w = np.asarray(r["center_world"], dtype=float)
        c_obj = transform_points(T_obj_world, c_w.reshape(1, 3))[0]
        real_by_group.setdefault(grp, []).append(c_obj)

    proxy_by_group = {g: [] for g in ["thumb", "index", "middle", "ring", "pinky", "palm"]}
    for name, p in proxy_obj.items():
        matched = False
        for g in ["thumb", "index", "middle", "ring", "pinky"]:
            if name.startswith(g):
                proxy_by_group[g].append(np.asarray(p, dtype=float))
                matched = True
                break
        if not matched:
            proxy_by_group["palm"].append(np.asarray(p, dtype=float))

    out = {}
    for g in ["thumb", "index", "middle", "ring", "pinky", "palm"]:
        P = proxy_by_group.get(g, [])
        R = real_by_group.get(g, [])
        if not P or not R:
            out[g] = {"available": False}
            continue

        ds = []
        for p in P:
            ds.append(min(float(np.linalg.norm(p - r)) for r in R))
        out[g] = {
            "available": True,
            "proxy_to_nearest_real_mean": float(np.mean(ds)),
            "proxy_to_nearest_real_min": float(np.min(ds)),
            "proxy_to_nearest_real_max": float(np.max(ds)),
            "num_proxy": int(len(P)),
            "num_real": int(len(R)),
        }
    return out


def approach_risk_world(T_world_hand, T_world_object, support_top_z):
    R = T_world_hand[:3, :3]
    hp = T_world_hand[:3, 3]
    op = T_world_object[:3, 3]

    palm_out = normalize(R[:, 0])
    approach_dir = -palm_out
    dot_up = float(np.dot(approach_dir, np.array([0.0, 0.0, 1.0])))

    risk = 0.0
    reasons = []

    hand_below_support = hp[2] < support_top_z + 0.020
    hand_below_object = hp[2] < op[2] - 0.010

    # 这里不再把 dot_up<0 简单当成坏，因为 top approach 往往是向下接近。
    # 真正高风险主要是：手在低位，并且接近方向带明显向上分量，容易从支撑下面钻入。
    if hand_below_support:
        risk += 80.0
        reasons.append(f"hand_low_near_support hp_z={hp[2]:.4f}")
    elif hand_below_object and dot_up > 0.25:
        risk += 60.0
        reasons.append(f"approach_likely_from_below dot_up={dot_up:.3f}")
    elif abs(dot_up) < 0.10:
        risk += 5.0
        reasons.append(f"approach_nearly_horizontal dot_up={dot_up:.3f}")
    else:
        reasons.append(f"approach_record dot_up={dot_up:.3f}")

    return float(risk), reasons, float(dot_up)


def object_proxy_penetration_penalty(group_dists, pen_tol=-0.006):
    penalty = 0.0
    reasons = []
    for g, d in group_dists.items():
        if d < pen_tol:
            p = min(120.0, 2000.0 * abs(float(d) - pen_tol))
            penalty += p
            reasons.append(f"{g}_deep_inside_object_aabb d={d:.4f} penalty={p:.1f}")
    return float(penalty), reasons


def classify_grasp(T_world_hand, T_world_object,
                   object_world_bbox_min, object_world_bbox_max,
                   support_top_z, support_min_clearance,
                   bbox_obj_min, bbox_obj_max,
                   approach_dot_up):
    hp = T_world_hand[:3, 3]
    obj_center = 0.5 * (object_world_bbox_min + object_world_bbox_max)
    obj_size_w = object_world_bbox_max - object_world_bbox_min
    obj_height = max(float(obj_size_w[2]), 1e-6)

    rel = hp - obj_center
    rel_z = float(rel[2])
    horiz_norm = float(np.linalg.norm([rel[0], rel[1], 0.0]))

    obj_size_obj = bbox_obj_max - bbox_obj_min
    major_idx = int(np.argmax(obj_size_obj))
    major_axis_obj = np.zeros(3)
    major_axis_obj[major_idx] = 1.0
    major_axis_w = normalize(T_world_object[:3, :3] @ major_axis_obj)

    half_len = max(0.5 * float(obj_size_obj[major_idx]), 1e-6)
    endness = abs(float(np.dot(rel, major_axis_w))) / half_len
    above_ratio = rel_z / obj_height

    reasons = []

    if support_min_clearance < -0.004:
        reasons.append(f"real_hand_below_support={support_min_clearance:.5f}")
        return "under_or_low_side_grasp", reasons, above_ratio, endness, horiz_norm

    if hp[2] < support_top_z + 0.020:
        reasons.append(f"handbase_too_low_to_support={hp[2] - support_top_z:.5f}")
        return "under_or_low_side_grasp", reasons, above_ratio, endness, horiz_norm

    if endness > 0.65:
        reasons.append(f"near_object_end endness={endness:.3f}")
        return "end_grasp", reasons, above_ratio, endness, horiz_norm

    # 不能只看 hand 在上方就判 top。
    # top 需要：明显在物体上方 + approach 有明显竖直成分。
    if above_ratio > 0.75 and abs(approach_dot_up) > 0.25 and horiz_norm < 0.12:
        reasons.append(f"top_like above_ratio={above_ratio:.3f} dot_up={approach_dot_up:.3f}")
        return "top_grasp", reasons, above_ratio, endness, horiz_norm

    if horiz_norm > 0.025 and abs(above_ratio) < 4.0:
        reasons.append(f"side_like horiz={horiz_norm:.3f} above_ratio={above_ratio:.3f} dot_up={approach_dot_up:.3f}")
        return "side_grasp", reasons, above_ratio, endness, horiz_norm

    if above_ratio > 0.75:
        reasons.append(f"high_ambiguous above_ratio={above_ratio:.3f} dot_up={approach_dot_up:.3f}")
        return "ambiguous_grasp", reasons, above_ratio, endness, horiz_norm

    reasons.append("fallback_ambiguous")
    return "ambiguous_grasp", reasons, above_ratio, endness, horiz_norm


def score_candidate(grasp_type, real_group_dists, support_min_clearance,
                    object_world_size, T_world_hand, T_world_object,
                    approach_risk, approach_reasons, type_reasons):
    obj_diag = max(float(np.linalg.norm(object_world_size)), 1e-6)

    # real geom surface proxy 已经减了 geom radius，所以阈值比 V4.20b 更合理。
    thumb_th = float(np.clip(0.10 * obj_diag, 0.012, 0.030))
    non_th = float(np.clip(0.09 * obj_diag, 0.010, 0.028))

    thumb_d = float(real_group_dists.get("thumb", 999.0))
    non_d = {g: float(real_group_dists.get(g, 999.0)) for g in NON_THUMB}
    best_non = min(non_d.values()) if non_d else 999.0

    near_thumb = thumb_d <= thumb_th
    near_non = [g for g, d in non_d.items() if d <= non_th]

    contact_score = 0.0
    contact_reasons = []

    if near_thumb:
        contact_score += 45.0
        contact_reasons.append(f"thumb_near_object d={thumb_d:.4f} th={thumb_th:.4f}")
    else:
        contact_score -= min(70.0, 900.0 * max(0.0, thumb_d - thumb_th))
        contact_reasons.append(f"thumb_far d={thumb_d:.4f} th={thumb_th:.4f}")

    if near_non:
        contact_score += 35.0 + 10.0 * len(near_non)
        contact_reasons.append(f"nonthumb_near={near_non} best={best_non:.4f} th={non_th:.4f}")
    else:
        contact_score -= min(80.0, 1000.0 * max(0.0, best_non - non_th))
        contact_reasons.append(f"nonthumb_far best={best_non:.4f} th={non_th:.4f}")

    if near_thumb and near_non:
        contact_score += 90.0
        contact_reasons.append("thumb_plus_nonthumb_real_proxy_ready")

    support_risk = 0.0
    support_reasons = []

    if support_min_clearance < -0.006:
        support_risk += 300.0
        support_reasons.append(f"hard_support_penetration={support_min_clearance:.5f}")
    elif support_min_clearance < -0.001:
        support_risk += 120.0
        support_reasons.append(f"soft_support_penetration={support_min_clearance:.5f}")
    elif support_min_clearance < 0.004:
        support_risk += 35.0
        support_reasons.append(f"near_support={support_min_clearance:.5f}")
    elif support_min_clearance < 0.010:
        support_risk += 10.0
        support_reasons.append(f"support_margin_small={support_min_clearance:.5f}")
    else:
        support_reasons.append(f"support_clear={support_min_clearance:.5f}")

    type_score = {
        "side_grasp": 28.0,
        "top_grasp": 22.0,
        "end_grasp": 18.0,
        "ambiguous_grasp": -8.0,
        "under_or_low_side_grasp": -260.0,
    }.get(grasp_type, -20.0)

    object_penalty, pen_reasons = object_proxy_penetration_penalty(real_group_dists)

    hp = T_world_hand[:3, 3]
    op = T_world_object[:3, 3]
    hand_center_dist = float(np.linalg.norm(hp - op))
    distance_bad = 0.0
    if hand_center_dist > 3.0 * obj_diag:
        distance_bad = 45.0
    elif hand_center_dist > 2.3 * obj_diag:
        distance_bad = 18.0

    final_score = (
        contact_score
        + type_score
        - support_risk
        - approach_risk
        - object_penalty
        - distance_bad
    )

    reasons = []
    reasons.extend(contact_reasons)
    reasons.extend(support_reasons)
    reasons.append(f"type_score={type_score:.1f} type={grasp_type}")
    reasons.extend(approach_reasons)
    reasons.extend(pen_reasons)
    if distance_bad > 0:
        reasons.append(f"handbase_far_from_object_center={hand_center_dist:.4f} distance_bad={distance_bad:.1f}")
    reasons.extend(type_reasons)

    return {
        "score": float(final_score),
        "contact_score": float(contact_score),
        "support_risk": float(support_risk),
        "approach_risk": float(approach_risk),
        "object_penetration_penalty": float(object_penalty),
        "distance_bad": float(distance_bad),
        "type_score": float(type_score),
        "thumb_near_threshold": float(thumb_th),
        "nonthumb_near_threshold": float(non_th),
        "near_thumb": bool(near_thumb),
        "near_nonthumb": near_non,
        "proxy_ready": bool(near_thumb and len(near_non) > 0),
        "rank_reason_short": "; ".join(reasons[:5]),
        "score_reasons": reasons,
    }


def select_diverse(rows, top_k, max_per_type):
    selected = []
    per_type = {}

    for r in rows:
        gt = r["grasp_type"]
        if per_type.get(gt, 0) >= max_per_type:
            continue
        selected.append(r)
        per_type[gt] = per_type.get(gt, 0) + 1
        if len(selected) >= top_k:
            return selected

    existing = {r["valid_local_index"] for r in selected}
    for r in rows:
        if r["valid_local_index"] in existing:
            continue
        selected.append(r)
        if len(selected) >= top_k:
            break

    return selected


def select_feasible(rows, top_k):
    feasible = []
    for r in rows:
        if r["grasp_type"] == "under_or_low_side_grasp":
            continue
        if r["support_risk"] >= 120.0:
            continue
        if r["approach_risk"] >= 80.0:
            continue
        feasible.append(r)
        if len(feasible) >= top_k:
            break
    return feasible


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object-code", required=True)
    ap.add_argument("--npy", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--model", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--object-body", default="grasp_object")
    ap.add_argument("--target-site", default="dataset_hand_base_debug")
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--max-per-type", type=int, default=6)

    ap.add_argument("--support-center-xy", default="0.455 0.0")
    ap.add_argument("--support-half-size", default="0.045 0.045 0.115")
    ap.add_argument("--support-top-z", type=float, default=0.23)
    ap.add_argument("--table-top-z", type=float, default=0.0)
    ap.add_argument("--pedestal-xy-margin", type=float, default=0.025)
    ap.add_argument("--object-clearance", type=float, default=0.003)

    ap.add_argument("--object-pos", default="")
    ap.add_argument("--object-quat-wxyz", default="1 0 0 0")
    ap.add_argument("--object-scale", default="", help="optional fixed object scale; empty means use sample scale")
    ap.add_argument("--support-json", default="", help="reserved; V4.20c keeps single table+pedestal interface")

    ap.add_argument("--use-collision-only", action="store_true", default=True)
    ap.add_argument("--all-geoms", action="store_true", help="use visual+collision geoms if collision geoms are missing")
    ap.add_argument("--debug-first-n", type=int, default=0, help="0 means all samples")
    args = ap.parse_args()

    npy_path = resolve(args.npy)
    mesh_path = resolve(args.mesh)
    model_path = resolve(args.model)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    support_center_xy = parse_vec2(args.support_center_xy)
    support_half_size = parse_vec3(args.support_half_size)
    support_top_z = float(args.support_top_z)
    table_top_z = float(args.table_top_z)

    verts = read_obj_vertices(mesh_path)
    samples = load_samples(npy_path)
    if args.debug_first_n and args.debug_first_n > 0:
        samples = samples[:args.debug_first_n]

    if not samples:
        raise RuntimeError(f"no valid samples in {npy_path}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    rows = []
    compare_debug = []

    print("========== V4.20c REAL HAND KEYPOINT PROXY SELECTOR ==========")
    print("object_code :", args.object_code)
    print("npy         :", rel(npy_path))
    print("mesh        :", rel(mesh_path))
    print("model       :", rel(model_path))
    print("samples     :", len(samples))
    print("target_site :", args.target_site)
    print("support     :", support_center_xy.tolist(), support_half_size.tolist(), "top_z=", support_top_z)
    print("table_top_z :", table_top_z)

    for count, (local_i, sample) in enumerate(samples, start=1):
        raw_i = sample_raw_index(local_i, sample)

        if args.object_scale.strip():
            scale = float(args.object_scale)
        else:
            scale = sample_scale(sample)

        verts_scaled = verts * scale
        bbox_min = verts_scaled.min(axis=0)
        bbox_max = verts_scaled.max(axis=0)
        bbox_size = bbox_max - bbox_min

        R_obj = quat_wxyz_to_R(parse_quat_wxyz(args.object_quat_wxyz))
        if args.object_pos.strip():
            object_pos = parse_vec3(args.object_pos)
        else:
            object_pos = compute_object_pos_on_support(
                verts_scaled=verts_scaled,
                R_obj=R_obj,
                support_center_xy=support_center_xy,
                support_top_z=support_top_z,
                clearance=args.object_clearance,
            )

        T_world_object = T_from_Rp(R_obj, object_pos)
        T_object_hand = extract_T_object_hand(sample)
        T_world_hand = T_world_object @ T_object_hand
        ctrl = extract_ctrl(sample)

        ik = solve_site_ik(model, args.target_site, T_world_hand, Q_HOME)

        row_base = {
            "object_code": args.object_code,
            "valid_local_index": int(local_i),
            "raw_sample_index": int(raw_i),
            "scale": float(scale),
            "T_object_hand_pos": T_object_hand[:3, 3].tolist(),
            "T_world_hand_pos": T_world_hand[:3, 3].tolist(),
            "T_world_object_pos": T_world_object[:3, 3].tolist(),
            "object_bbox_size_scaled": bbox_size.tolist(),
            "ik_success": bool(ik["success"]),
            "ik_pos_err": float(ik["final_pos_err_norm"]),
            "ik_rot_err": float(ik["final_rot_err_norm"]),
            "ctrl": ctrl,
        }

        if not ik["success"]:
            r = dict(row_base)
            r.update({
                "grasp_type": "ik_failed",
                "score": -1e6,
                "proxy_ready": False,
                "near_thumb": False,
                "near_nonthumb": [],
                "contact_score": -1e6,
                "support_risk": 1e6,
                "approach_risk": 1e6,
                "object_penetration_penalty": 0.0,
                "distance_bad": 0.0,
                "type_score": -1e6,
                "rank_reason_short": "ik_failed",
                "score_reasons": ["ik_failed"],
            })
            rows.append(r)
            print(f"[{count:03d}/{len(samples):03d}] local={local_i:03d} IK failed pos_err={ik['final_pos_err_norm']:.5f}")
            continue

        set_qpos_once(model, data, ik["q_arm"], ctrl)

        use_collision_only = bool(args.use_collision_only and not args.all_geoms)
        real_records, group_counts = collect_real_hand_geom_records(
            model, data, use_collision_only=use_collision_only
        )

        if sum(group_counts.values()) == 0 and use_collision_only:
            real_records, group_counts = collect_real_hand_geom_records(
                model, data, use_collision_only=False
            )

        if sum(group_counts.get(g, 0) for g in ["thumb", "index", "middle", "ring", "pinky"]) == 0:
            r = dict(row_base)
            r.update({
                "grasp_type": "no_hand_geoms_found",
                "score": -1e6,
                "proxy_ready": False,
                "near_thumb": False,
                "near_nonthumb": [],
                "contact_score": -1e6,
                "support_risk": 1e6,
                "approach_risk": 1e6,
                "object_penetration_penalty": 0.0,
                "distance_bad": 0.0,
                "type_score": -1e6,
                "rank_reason_short": "no_hand_geoms_found",
                "score_reasons": ["no_hand_geoms_found", f"group_counts={group_counts}"],
                "real_group_counts": group_counts,
            })
            rows.append(r)
            print(f"[{count:03d}/{len(samples):03d}] local={local_i:03d} no hand geoms found group_counts={group_counts}")
            continue

        real_group_dists, real_best = surface_distances_to_object_aabb(
            real_records, T_world_object, bbox_min, bbox_max
        )

        support_min_clearance, support_group_min, support_best = support_clearance_real(
            records=real_records,
            support_center_xy=support_center_xy,
            support_half_size=support_half_size,
            pedestal_top_z=support_top_z,
            table_top_z=table_top_z,
            pedestal_xy_margin=args.pedestal_xy_margin,
        )

        object_world_verts = transform_points(T_world_object, verts_scaled)
        ow_min = object_world_verts.min(axis=0)
        ow_max = object_world_verts.max(axis=0)
        ow_size = ow_max - ow_min

        approach_risk, approach_reasons, approach_dot_up = approach_risk_world(
            T_world_hand, T_world_object, support_top_z
        )

        grasp_type, type_reasons, above_ratio, endness, horiz_norm = classify_grasp(
            T_world_hand=T_world_hand,
            T_world_object=T_world_object,
            object_world_bbox_min=ow_min,
            object_world_bbox_max=ow_max,
            support_top_z=support_top_z,
            support_min_clearance=support_min_clearance,
            bbox_obj_min=bbox_min,
            bbox_obj_max=bbox_max,
            approach_dot_up=approach_dot_up,
        )

        score_info = score_candidate(
            grasp_type=grasp_type,
            real_group_dists=real_group_dists,
            support_min_clearance=support_min_clearance,
            object_world_size=ow_size,
            T_world_hand=T_world_hand,
            T_world_object=T_world_object,
            approach_risk=approach_risk,
            approach_reasons=approach_reasons,
            type_reasons=type_reasons,
        )

        hardcoded_proxy_obj = make_hardcoded_proxy_points(T_object_hand, ctrl)
        hardcoded_dists = hardcoded_group_distances(hardcoded_proxy_obj, bbox_min, bbox_max)
        delta_summary = hardcoded_real_delta_summary(hardcoded_proxy_obj, real_records, T_world_object)

        best_nonthumb = min(real_group_dists.get(g, 999.0) for g in NON_THUMB)

        row = dict(row_base)
        row.update({
            "grasp_type": grasp_type,
            "score": float(score_info["score"]),
            "final_score": float(score_info["score"]),
            "contact_score": float(score_info["contact_score"]),
            "support_risk": float(score_info["support_risk"]),
            "approach_risk": float(score_info["approach_risk"]),
            "approach_dot_up": float(approach_dot_up),
            "object_penetration_penalty": float(score_info["object_penetration_penalty"]),
            "distance_bad": float(score_info["distance_bad"]),
            "type_score": float(score_info["type_score"]),
            "proxy_ready": bool(score_info["proxy_ready"]),
            "near_thumb": bool(score_info["near_thumb"]),
            "near_nonthumb": score_info["near_nonthumb"],
            "thumb_near_threshold": float(score_info["thumb_near_threshold"]),
            "nonthumb_near_threshold": float(score_info["nonthumb_near_threshold"]),
            "thumb_surface_dist": float(real_group_dists.get("thumb", 999.0)),
            "best_nonthumb_surface_dist": float(best_nonthumb),
            "real_group_object_surface_distances": real_group_dists,
            "real_group_best_geoms": real_best,
            "real_group_counts": group_counts,
            "support_min_clearance": float(support_min_clearance),
            "support_group_min_clearance": support_group_min,
            "support_best_geoms": support_best,
            "hardcoded_group_object_distances": hardcoded_dists,
            "hardcoded_to_real_delta_summary": delta_summary,
            "above_ratio": float(above_ratio),
            "endness": float(endness),
            "horiz_norm": float(horiz_norm),
            "type_reasons": type_reasons,
            "rank_reason_short": score_info["rank_reason_short"],
            "score_reasons": score_info["score_reasons"],
        })

        rows.append(row)

        compare_debug.append({
            "valid_local_index": int(local_i),
            "raw_sample_index": int(raw_i),
            "grasp_type": grasp_type,
            "score": row["score"],
            "proxy_ready": row["proxy_ready"],
            "real_group_object_surface_distances": real_group_dists,
            "hardcoded_group_object_distances": hardcoded_dists,
            "hardcoded_to_real_delta_summary": delta_summary,
            "real_group_counts": group_counts,
            "support_min_clearance": float(support_min_clearance),
            "ik": ik,
        })

        print(
            f"[{count:03d}/{len(samples):03d}] "
            f"local={local_i:03d} raw={raw_i:03d} "
            f"type={grasp_type:24s} score={row['score']:+8.2f} "
            f"ready={row['proxy_ready']} "
            f"thumb={row['thumb_surface_dist']:+.4f} "
            f"non={row['best_nonthumb_surface_dist']:+.4f} "
            f"support={support_min_clearance:+.4f} "
            f"ik={ik['success']}"
        )

    rows_sorted = sorted(rows, key=lambda x: x["score"], reverse=True)

    selected_overall = rows_sorted[:args.top_k]
    selected_diverse = select_diverse(rows_sorted, args.top_k, args.max_per_type)
    selected_feasible = select_feasible(rows_sorted, args.top_k)

    save_json(out_dir / "ranked_real_proxy_candidates.json", rows_sorted)
    save_json(out_dir / "selected_topk_overall.json", selected_overall)
    save_json(out_dir / "selected_topk_diverse.json", selected_diverse)
    save_json(out_dir / "selected_topk_feasible.json", selected_feasible)
    save_json(out_dir / "real_proxy_compare_debug.json", compare_debug)

    (out_dir / "selected_valid_local_indices_overall.txt").write_text(
        "\n".join(str(r["valid_local_index"]) for r in selected_overall) + "\n"
    )
    (out_dir / "selected_valid_local_indices_diverse.txt").write_text(
        "\n".join(str(r["valid_local_index"]) for r in selected_diverse) + "\n"
    )
    (out_dir / "selected_valid_local_indices_feasible.txt").write_text(
        "\n".join(str(r["valid_local_index"]) for r in selected_feasible) + "\n"
    )

    csv_path = out_dir / "ranked_real_proxy_candidates.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "local", "raw", "type", "score",
            "ready", "contact_score", "support_risk", "approach_risk",
            "object_penetration_penalty", "distance_bad", "type_score",
            "thumb_surface_dist", "best_nonthumb_surface_dist",
            "support_min_clearance", "ik_success", "ik_pos_err",
            "approach_dot_up", "above_ratio", "endness", "horiz_norm",
            "rank_reason_short",
        ])
        for rank, r in enumerate(rows_sorted, start=1):
            w.writerow([
                rank,
                r["valid_local_index"],
                r["raw_sample_index"],
                r["grasp_type"],
                f"{r['score']:.6f}",
                r["proxy_ready"],
                f"{r['contact_score']:.6f}",
                f"{r['support_risk']:.6f}",
                f"{r['approach_risk']:.6f}",
                f"{r['object_penetration_penalty']:.6f}",
                f"{r['distance_bad']:.6f}",
                f"{r['type_score']:.6f}",
                f"{r.get('thumb_surface_dist', 999.0):.6f}",
                f"{r.get('best_nonthumb_surface_dist', 999.0):.6f}",
                f"{r.get('support_min_clearance', 999.0):.6f}",
                r.get("ik_success"),
                f"{r.get('ik_pos_err', 999.0):.6f}",
                f"{r.get('approach_dot_up', 999.0):.6f}",
                f"{r.get('above_ratio', 999.0):.6f}",
                f"{r.get('endness', 999.0):.6f}",
                f"{r.get('horiz_norm', 999.0):.6f}",
                r.get("rank_reason_short", ""),
            ])

    type_counts = {}
    ready_count = 0
    ik_fail_count = 0
    for r in rows_sorted:
        type_counts[r["grasp_type"]] = type_counts.get(r["grasp_type"], 0) + 1
        if r.get("proxy_ready"):
            ready_count += 1
        if not r.get("ik_success", False):
            ik_fail_count += 1

    lines = []
    lines.append("========== V4.20c REAL HAND KEYPOINT PROXY SELECTOR ==========")
    lines.append("定位：real MuJoCo hand geom proxy；目标是 Top-K 高召回，不是最终执行器。")
    lines.append(f"object_code : {args.object_code}")
    lines.append(f"npy         : {rel(npy_path)}")
    lines.append(f"mesh        : {rel(mesh_path)}")
    lines.append(f"model       : {rel(model_path)}")
    lines.append(f"samples     : {len(rows_sorted)}")
    lines.append(f"top_k       : {args.top_k}")
    lines.append(f"proxy_ready_count: {ready_count}")
    lines.append(f"ik_fail_count    : {ik_fail_count}")
    lines.append("")
    lines.append("---- type counts ----")
    for k, v in sorted(type_counts.items()):
        lines.append(f"{k}: {v}")

    def append_top(title, selected):
        lines.append("")
        lines.append(f"---- {title} ----")
        for rank, r in enumerate(selected, start=1):
            lines.append(
                f"rank={rank:02d} "
                f"local={r['valid_local_index']:03d} raw={r['raw_sample_index']:03d} "
                f"type={r['grasp_type']} score={r['score']:.2f} ready={r['proxy_ready']} "
                f"C={r['contact_score']:.1f} S={r['support_risk']:.1f} "
                f"A={r['approach_risk']:.1f} Pen={r['object_penetration_penalty']:.1f} "
                f"D={r['distance_bad']:.1f} T={r['type_score']:.1f} "
                f"thumb={r.get('thumb_surface_dist', 999.0):+.4f} "
                f"non={r.get('best_nonthumb_surface_dist', 999.0):+.4f} "
                f"support={r.get('support_min_clearance', 999.0):+.4f} "
                f"ik={r.get('ik_success')}"
            )
            lines.append(f"  - {r.get('rank_reason_short', '')}")

    append_top("selected_topk_overall", selected_overall)
    append_top("selected_topk_diverse", selected_diverse)
    append_top("selected_topk_feasible", selected_feasible)

    lines.append("")
    lines.append("---- output ----")
    lines.append(f"ranked json    : {rel(out_dir / 'ranked_real_proxy_candidates.json')}")
    lines.append(f"ranked csv     : {rel(csv_path)}")
    lines.append(f"overall topk   : {rel(out_dir / 'selected_topk_overall.json')}")
    lines.append(f"diverse topk   : {rel(out_dir / 'selected_topk_diverse.json')}")
    lines.append(f"feasible topk  : {rel(out_dir / 'selected_topk_feasible.json')}")
    lines.append(f"compare debug  : {rel(out_dir / 'real_proxy_compare_debug.json')}")
    lines.append("===============================================================")

    report = "\n".join(lines) + "\n"
    (out_dir / "real_proxy_report.txt").write_text(report, encoding="utf-8")
    print(report)


if __name__ == "__main__":
    main()
