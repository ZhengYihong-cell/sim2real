#!/usr/bin/env python3
"""
文件名：
    run_v4_12p4e_fast_hand_local_pose_score_debug.py

脚本类别：
    debug / fast-geometric-score / hand-local-refinement / candidate-patch

用途：
    本脚本用于 V4.12P4E-fast 阶段。
    它只做“手局部位姿 + 自适应手型”的快速几何代理评分，不对每个扰动点重复跑完整
    P2/P3/P4C。它用于替代慢速 local target search。

核心流程：
    1. 读取当前 P3 best q_grasp。
    2. 在 MuJoCo 中设置 q_grasp。
    3. 从 candidate 中读取 O7 hand ctrl/qpos 作为手型先验。
    4. 用 MuJoCo actuator ctrlrange / joint range 作为手型边界。
    5. 构造自适应 hand preshape：
           hand_ctrl = open_ctrl + ratio * (candidate_prior - open_ctrl)
       而不是写死 thumb_roll / thumb_yaw / finger_preclose。
    6. 在 target frame 局部坐标系下采样小范围 Δx/Δy/Δz/yaw。
    7. 对每个局部扰动做纯几何评分：
           - thumb 是否靠近物体侧壁
           - 四指是否靠近物体另一侧侧壁
           - thumb 与四指是否形成对握
           - 手掌/手指是否过低碰支撑面
           - 是否需要过大的 finger preclose
    8. 输出 Top-K 和 best candidate。
    9. 自动生成只验证 best 的 P2/P3/P4C viewer 脚本。

输入：
    1. --model
       当前 MuJoCo XML。
    2. --candidate
       原始 candidate JSON，必须包含 target.T_object_target。
    3. --p3-json
       当前 P3 输出 JSON，用于读取 best_available / best_pass 的 q_grasp。
    4. --object-body
       物体 body 名称，例如 grasp_can。
    5. --target-body
       hand target 对应的 MuJoCo body，当前通常是 fr3_link7。

输出：
    1. --out-dir/summary.json
       Top-K 几何评分结果。
    2. --out-dir/best_candidate.json
       只对 best local correction patch 后的 candidate。
    3. --out-dir/best_config.json
       best 的局部位姿修正和自适应 preshape 参数。
    4. --out-dir/run_best_once.sh
       只跑 best 一次 P2/P3/P4C viewer 的验证脚本。

当前流程位置：
    candidate prior / P3 best arm pose
        -> P4E-fast 几何代理评分
        -> 只对 best 1 个修正跑 P2/P3/P4C viewer
        -> 后续固化到在线 selector/runner

本脚本不负责：
    1. 不对每个扰动点跑 IK。
    2. 不对每个扰动点跑动态闭合。
    3. 不做全局路径规划。
    4. 不把 thumb/finger 姿态写死为固定角度。
    5. 不保证 best 一定抓取成功，只负责快速筛出更合理的局部修正。
"""

from pathlib import Path
import argparse
import json
import math
import os
import shlex
import numpy as np
import mujoco


PROJECT = Path.home() / "Projects/o7_mujoco_sim"
RUN_CLEAN = PROJECT / "run_mujoco_clean.sh"

P2_SCRIPT = PROJECT / "scripts/07_precheck_ik_collision/run_v4_12p2_pinocchio_multiseed_ik_debug.py"
P3_SCRIPT = PROJECT / "scripts/07_precheck_ik_collision/run_v4_12p3_pinocchio_ik_plus_mujoco_collision_precheck_debug.py"
P4C_SCRIPT = PROJECT / "scripts/05_execution_runner/run_v4_12p4c_opposition_contact_seek_close_debug.py"

ARM_JOINTS = [
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
]

ACTIVE_HAND_JOINTS = [
    "thumb_cmc_roll",
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]

THUMB_POSTURE_JOINTS = [
    "thumb_cmc_roll",
    "thumb_cmc_yaw",
]

THUMB_CLOSE_JOINTS = [
    "thumb_cmc_pitch",
]

FOUR_FINGER_JOINTS = [
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]

HAND_GROUP_TOKENS = {
    "thumb": ["thumb"],
    "index": ["index"],
    "middle": ["middle"],
    "ring": ["ring"],
    "pinky": ["pinky"],
    "palm": ["palm", "hand_base", "hand"],
}

SUPPORT_TOKENS = ["object_pedestal", "pedestal", "support", "table"]


def resolve_path(p):
    p = Path(str(p)).expanduser()
    if not p.is_absolute():
        p = PROJECT / p
    return p


def rel(p):
    p = resolve_path(p)
    try:
        return str(p.relative_to(PROJECT))
    except Exception:
        return str(p)


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


def parse_float_list(s):
    return [float(x) for x in str(s).replace(",", " ").split() if x.strip()]


def mj_id(model, objtype, name):
    return mujoco.mj_name2id(model, objtype, str(name))


def mj_name(model, objtype, idx):
    return mujoco.mj_id2name(model, objtype, int(idx)) or ""


def body_id(model, name):
    return mj_id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def joint_id(model, name):
    return mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def actuator_id(model, name):
    return mj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def geom_name(model, gid):
    n = mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)
    return n if n else f"geom_{gid}"


def body_name(model, bid):
    n = mj_name(model, mujoco.mjtObj.mjOBJ_BODY, bid)
    return n if n else f"body_{bid}"


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


def T_delta_local(dx, dy, dz, yaw_deg):
    yaw = math.radians(float(yaw_deg))
    c = math.cos(yaw)
    s = math.sin(yaw)

    T = np.eye(4)
    T[:3, :3] = np.array([
        [c, -s, 0.0],
        [s,  c, 0.0],
        [0.0, 0.0, 1.0],
    ])
    T[:3, 3] = np.array([dx, dy, dz], dtype=float)
    return T


def transform_points_world_by_target_delta(points, T_world_target, T_delta):
    if len(points) == 0:
        return points

    T_map = T_world_target @ T_delta @ T_inv(T_world_target)
    pts = np.asarray(points, dtype=float)
    ph = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    out = (T_map @ ph.T).T[:, :3]
    return out


def object_T_world(model, data, object_body):
    bid = body_id(model, object_body)
    if bid < 0:
        raise RuntimeError(f"cannot find object body: {object_body}")
    return T_from_Rp(data.xmat[bid].reshape(3, 3), data.xpos[bid])


def target_T_world(model, data, target_body):
    bid = body_id(model, target_body)
    if bid < 0:
        raise RuntimeError(f"cannot find target body: {target_body}")
    return T_from_Rp(data.xmat[bid].reshape(3, 3), data.xpos[bid])


def get_joint_qpos(model, data, name):
    jid = joint_id(model, name)
    if jid < 0:
        return None

    jtype = int(model.jnt_type[jid])
    if jtype not in [
        int(mujoco.mjtJoint.mjJNT_HINGE),
        int(mujoco.mjtJoint.mjJNT_SLIDE),
    ]:
        return None

    qadr = int(model.jnt_qposadr[jid])
    return float(data.qpos[qadr])


def set_joint_qpos(model, data, name, value):
    jid = joint_id(model, name)
    if jid < 0:
        return False

    jtype = int(model.jnt_type[jid])
    if jtype not in [
        int(mujoco.mjtJoint.mjJNT_HINGE),
        int(mujoco.mjtJoint.mjJNT_SLIDE),
    ]:
        return False

    qadr = int(model.jnt_qposadr[jid])
    val = float(value)

    if bool(model.jnt_limited[jid]):
        lo, hi = model.jnt_range[jid]
        val = float(np.clip(val, lo, hi))

    data.qpos[qadr] = val
    return True


def find_actuator_for_joint(model, joint_name):
    names = [
        joint_name,
        f"{joint_name}_pos",
        f"{joint_name}_ctrl",
        f"{joint_name}_act",
        f"{joint_name}_motor",
    ]

    for n in names:
        aid = actuator_id(model, n)
        if aid >= 0:
            return aid, n

    return -1, ""


def actuator_or_joint_range(model, joint_name):
    aid, act_name = find_actuator_for_joint(model, joint_name)
    if aid >= 0:
        limited = bool(model.actuator_ctrllimited[aid])
        if limited:
            lo, hi = model.actuator_ctrlrange[aid]
            return (float(lo), float(hi)), "actuator", act_name

    jid = joint_id(model, joint_name)
    if jid >= 0 and bool(model.jnt_limited[jid]):
        lo, hi = model.jnt_range[jid]
        return (float(lo), float(hi)), "joint", act_name

    return (-3.0, 3.0), "fallback", act_name


def clamp_ctrl(model, joint_name, value):
    cr, _, _ = actuator_or_joint_range(model, joint_name)
    lo, hi = cr
    return float(np.clip(float(value), lo, hi))


def set_actuator_ctrl(model, data, joint_name, value):
    aid, act_name = find_actuator_for_joint(model, joint_name)
    if aid < 0:
        return False, ""

    data.ctrl[aid] = clamp_ctrl(model, joint_name, value)
    return True, act_name


def apply_arm_q(model, data, qdict):
    for j, v in (qdict or {}).items():
        if j not in ARM_JOINTS:
            continue
        set_joint_qpos(model, data, j, v)
        set_actuator_ctrl(model, data, j, v)


def apply_hand_ctrl_as_qpos(model, data, ctrl):
    for j, v in (ctrl or {}).items():
        if j not in ACTIVE_HAND_JOINTS:
            continue
        val = clamp_ctrl(model, j, v)
        set_joint_qpos(model, data, j, val)
        set_actuator_ctrl(model, data, j, val)


def selected_best(p3, which):
    item = p3.get(which)
    if item is None:
        raise RuntimeError(f"{which} is None in p3 json")
    if "q_grasp" not in item:
        raise RuntimeError(f"{which} missing q_grasp")
    return item


def extract_candidate_prior(candidate, model):
    hand = candidate.get("hand", {}) or {}
    keys = [
        "o7_active_ctrl",
        "active_ctrl",
        "ctrl",
        "target_ctrl",
        "qpos",
        "target_qpos",
    ]

    for key in keys:
        val = hand.get(key, None)
        if isinstance(val, dict):
            out = {}
            for j in ACTIVE_HAND_JOINTS:
                if j in val:
                    out[j] = clamp_ctrl(model, j, float(val[j]))
            if out:
                return out, f"hand.{key}"

    for key in keys:
        val = candidate.get(key, None)
        if isinstance(val, dict):
            out = {}
            for j in ACTIVE_HAND_JOINTS:
                if j in val:
                    out[j] = clamp_ctrl(model, j, float(val[j]))
            if out:
                return out, key

    # fallback 只在 candidate 没有手型时使用；边界来自模型 ctrlrange/joint range，不写固定角度。
    out = {}
    for j in ACTIVE_HAND_JOINTS:
        cr, _, _ = actuator_or_joint_range(model, j)
        lo, hi = cr
        if abs(hi) >= abs(lo):
            out[j] = 0.5 * hi
        else:
            out[j] = 0.5 * lo

    return out, "fallback_from_ctrlrange"


def make_open_ctrl(model):
    out = {}
    for j in ACTIVE_HAND_JOINTS:
        # 当前 O7 模型中 0 通常是张开位；若后续模型有 nonzero default，可在这里替换为 default ctrl/qpos。
        out[j] = clamp_ctrl(model, j, 0.0)
    return out


def make_adaptive_ctrl(model, open_ctrl, prior_ctrl, thumb_alpha, finger_beta, thumb_pitch_beta):
    ctrl = dict(open_ctrl)

    for j in THUMB_POSTURE_JOINTS:
        op = float(open_ctrl.get(j, 0.0))
        pr = float(prior_ctrl.get(j, op))
        ctrl[j] = clamp_ctrl(model, j, op + float(thumb_alpha) * (pr - op))

    for j in THUMB_CLOSE_JOINTS:
        op = float(open_ctrl.get(j, 0.0))
        pr = float(prior_ctrl.get(j, op))
        ctrl[j] = clamp_ctrl(model, j, op + float(thumb_pitch_beta) * (pr - op))

    for j in FOUR_FINGER_JOINTS:
        op = float(open_ctrl.get(j, 0.0))
        pr = float(prior_ctrl.get(j, op))
        ctrl[j] = clamp_ctrl(model, j, op + float(finger_beta) * (pr - op))

    return ctrl


def geom_world_points(model, data, gid, max_points_per_geom=40):
    gid = int(gid)
    gtype = int(model.geom_type[gid])
    pos = data.geom_xpos[gid].copy()
    R = data.geom_xmat[gid].reshape(3, 3).copy()

    if gtype == int(mujoco.mjtGeom.mjGEOM_MESH):
        mid = int(model.geom_dataid[gid])
        if mid >= 0:
            adr = int(model.mesh_vertadr[mid])
            num = int(model.mesh_vertnum[mid])
            verts = np.asarray(model.mesh_vert[adr:adr + num], dtype=float)
            if len(verts) > max_points_per_geom:
                stride = max(1, len(verts) // max_points_per_geom)
                verts = verts[::stride][:max_points_per_geom]
            return (R @ verts.T).T + pos

    # primitive fallback：用中心和轴向偏移点近似
    size = np.asarray(model.geom_size[gid], dtype=float).reshape(-1)
    pts_local = [np.zeros(3)]
    for ax in range(3):
        mag = float(size[min(ax, len(size) - 1)]) if len(size) else 0.02
        pts_local.append(np.eye(3)[ax] * mag)
        pts_local.append(-np.eye(3)[ax] * mag)

    pts_local = np.asarray(pts_local, dtype=float)
    return (R @ pts_local.T).T + pos


def collect_object_points(model, data, object_body):
    root = body_id(model, object_body)
    if root < 0:
        raise RuntimeError(f"cannot find object body: {object_body}")

    pts = []
    geoms = []

    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        if body_is_descendant(model, bid, root):
            geoms.append(gid)
            pts.append(geom_world_points(model, data, gid, max_points_per_geom=200))

    if not pts:
        raise RuntimeError(f"no object geoms found for body: {object_body}")

    pts = np.concatenate(pts, axis=0)
    return pts, geoms


def collect_support_top(model, data):
    max_z = None
    support_geoms = []

    for gid in range(model.ngeom):
        text = f"{geom_name(model, gid)} {body_name_of_geom(model, gid)}".lower()
        if any(t in text for t in SUPPORT_TOKENS):
            support_geoms.append(gid)
            pts = geom_world_points(model, data, gid, max_points_per_geom=80)
            z = float(np.max(pts[:, 2]))
            if max_z is None or z > max_z:
                max_z = z

    if max_z is None:
        max_z = 0.0

    return float(max_z), support_geoms


def classify_hand_group(model, gid):
    text = f"{geom_name(model, gid)} {body_name_of_geom(model, gid)}".lower()

    for group, toks in HAND_GROUP_TOKENS.items():
        if any(t in text for t in toks):
            return group

    return ""


def collect_hand_points_by_group(model, data):
    groups = {
        "thumb": [],
        "index": [],
        "middle": [],
        "ring": [],
        "pinky": [],
        "palm": [],
    }

    names = {g: [] for g in groups}

    for gid in range(model.ngeom):
        group = classify_hand_group(model, gid)
        if not group:
            continue

        pts = geom_world_points(model, data, gid, max_points_per_geom=30)
        groups[group].append(pts)
        names[group].append({
            "geom": geom_name(model, gid),
            "body": body_name_of_geom(model, gid),
            "npts": int(len(pts)),
        })

    out = {}
    for g, arrs in groups.items():
        if arrs:
            out[g] = np.concatenate(arrs, axis=0)
        else:
            out[g] = np.zeros((0, 3))

    return out, names


def estimate_upright_object_geometry(object_points):
    pts = np.asarray(object_points, dtype=float)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    center = 0.5 * (mn + mx)

    # 对 upright can/cylinder，xy 半径用 bbox x/y 的较大半宽。
    radius_x = 0.5 * float(mx[0] - mn[0])
    radius_y = 0.5 * float(mx[1] - mn[1])
    radius_xy = max(radius_x, radius_y)

    return {
        "bbox_min": mn,
        "bbox_max": mx,
        "center": center,
        "center_xy": center[:2],
        "z_min": float(mn[2]),
        "z_max": float(mx[2]),
        "height": float(mx[2] - mn[2]),
        "radius_xy": float(radius_xy),
        "radius_x": radius_x,
        "radius_y": radius_y,
    }


def side_surface_best(points, obj_geom, z_weight):
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] == 0:
        return {
            "valid": False,
            "score": 999.0,
            "point": np.zeros(3),
            "radial_unit": np.zeros(2),
            "radial": 999.0,
            "side_error": 999.0,
            "z_error": 999.0,
        }

    cxy = np.asarray(obj_geom["center_xy"], dtype=float)
    r = float(obj_geom["radius_xy"])
    zmin = float(obj_geom["z_min"])
    zmax = float(obj_geom["z_max"])

    xy = pts[:, :2]
    z = pts[:, 2]

    radial_vec = xy - cxy.reshape(1, 2)
    radial = np.linalg.norm(radial_vec, axis=1)

    side_error = np.abs(radial - r)

    # can 是细长物体，不强制固定抓取高度；只惩罚明显离开物体高度范围的点。
    z_low_err = np.maximum(0.0, zmin - z)
    z_high_err = np.maximum(0.0, z - zmax)
    z_error = z_low_err + z_high_err

    score = side_error + float(z_weight) * z_error

    idx = int(np.argmin(score))
    ru = radial_vec[idx] / (radial[idx] + 1e-12)

    return {
        "valid": True,
        "score": float(score[idx]),
        "point": pts[idx],
        "radial_unit": ru,
        "radial": float(radial[idx]),
        "side_error": float(side_error[idx]),
        "z_error": float(z_error[idx]),
    }


def score_one(hand_pts, obj_geom, support_top, ctrl, sample_meta, args):
    thumb = side_surface_best(hand_pts["thumb"], obj_geom, args.z_weight)

    four_candidates = {}
    for g in ["index", "middle", "ring", "pinky"]:
        four_candidates[g] = side_surface_best(hand_pts[g], obj_geom, args.z_weight)

    valid_four = {g: v for g, v in four_candidates.items() if v["valid"]}
    if not thumb["valid"] or not valid_four:
        return {
            "score": 1e9,
            "reasons": ["missing thumb or finger points"],
        }

    best_four_group, best_four = min(valid_four.items(), key=lambda kv: kv[1]["score"])

    good_four_groups = [
        g for g, v in valid_four.items()
        if v["side_error"] <= args.good_surface_tol and v["z_error"] <= args.good_z_tol
    ]

    dot = float(np.dot(thumb["radial_unit"], best_four["radial_unit"]))
    opposition_error = abs(dot + 1.0)

    all_hand = []
    for g in ["thumb", "index", "middle", "ring", "pinky", "palm"]:
        if len(hand_pts[g]) > 0:
            all_hand.append(hand_pts[g])
    all_hand = np.concatenate(all_hand, axis=0) if all_hand else np.zeros((0, 3))

    min_hand_z = float(np.min(all_hand[:, 2])) if len(all_hand) else 999.0
    support_clearance = min_hand_z - float(support_top)
    support_penalty = max(0.0, args.min_support_clearance - support_clearance)

    finger_ctrl_max = max(abs(float(ctrl.get(j, 0.0))) for j in FOUR_FINGER_JOINTS)
    thumb_posture_mag = math.sqrt(
        float(ctrl.get("thumb_cmc_roll", 0.0)) ** 2
        + float(ctrl.get("thumb_cmc_yaw", 0.0)) ** 2
    )

    local_delta_norm = float(np.linalg.norm(sample_meta["delta_local"]))
    yaw_abs = abs(float(sample_meta["yaw_deg"]))

    score = 0.0
    score += args.w_thumb_surface * thumb["score"]
    score += args.w_four_surface * best_four["score"]
    score += args.w_opposition * opposition_error
    score += args.w_support * support_penalty
    score += args.w_finger_curl * max(0.0, finger_ctrl_max - args.preferred_finger_max)
    score += args.w_local_delta * local_delta_norm
    score += args.w_yaw * math.radians(yaw_abs)
    score -= args.w_good_four_group * len(good_four_groups)

    # 轻度鼓励 thumb 和至少一个四指都接近侧壁，不能只靠一个点好。
    if thumb["side_error"] <= args.good_surface_tol and best_four["side_error"] <= args.good_surface_tol:
        score -= args.w_opposition_bonus

    reasons = []
    reasons.append(f"thumb_side={thumb['side_error']:.4f}")
    reasons.append(f"four_side={best_four['side_error']:.4f}:{best_four_group}")
    reasons.append(f"dot={dot:.3f}")
    reasons.append(f"good_four={len(good_four_groups)}")
    reasons.append(f"support_clear={support_clearance:.4f}")
    reasons.append(f"finger_max={finger_ctrl_max:.3f}")

    return {
        "score": float(score),
        "thumb": thumb,
        "best_four_group": best_four_group,
        "best_four": best_four,
        "four_candidates": four_candidates,
        "good_four_groups": good_four_groups,
        "opposition_dot": dot,
        "opposition_error": opposition_error,
        "min_hand_z": min_hand_z,
        "support_clearance": support_clearance,
        "support_penalty": support_penalty,
        "finger_ctrl_max": finger_ctrl_max,
        "thumb_posture_mag": thumb_posture_mag,
        "reasons": reasons,
    }


def patch_candidate_target(candidate, T_world_object, T_world_target_old, T_delta, best_meta):
    T_world_target_new = T_world_target_old @ T_delta
    T_object_target_new = T_inv(T_world_object) @ T_world_target_new

    patched = json.loads(json.dumps(candidate))
    patched["target"]["T_object_target"] = T_object_target_new.tolist()

    meta = patched.setdefault("debug_patch_meta", {})
    meta["v4_12p4e_fast_hand_local_pose_score"] = to_jsonable(best_meta)
    meta["v4_12p4e_fast_hand_local_pose_score"]["T_object_target_new"] = T_object_target_new.tolist()

    return patched, {
        "T_world_target_new": T_world_target_new,
        "T_object_target_new": T_object_target_new,
    }


def shell_join(cmd):
    return " ".join(shlex.quote(str(x)) for x in cmd)


def write_best_scripts(args, out_dir, best_record):
    out_dir = resolve_path(out_dir)

    p2_json = out_dir / "best_p2.json"
    p3_json = out_dir / "best_p3.json"
    best_plan = out_dir / "best_plan.json"
    p4c_json = out_dir / "best_p4c_viewer.json"

    best_candidate = out_dir / "best_candidate.json"
    cfg = best_record["hand_config"]["ctrl"]

    thumb_roll = float(cfg.get("thumb_cmc_roll", 0.0))
    thumb_yaw = float(cfg.get("thumb_cmc_yaw", 0.0))
    thumb_pitch = float(cfg.get("thumb_cmc_pitch", 0.0))

    # P4C 当前没有单独的 finger preclose 输入，所以用 best 几何评分中的 finger_ctrl_max
    # 推导一个有限 finger_seek_duration，避免四指无限深卷。
    finger_max = float(best_record["score_detail"].get("finger_ctrl_max", 0.0))
    verify_finger_speed = float(args.verify_finger_seek_speed)
    if verify_finger_speed <= 1e-6:
        verify_finger_duration = float(args.verify_finger_seek_duration)
    else:
        verify_finger_duration = max(
            args.verify_finger_seek_duration_min,
            min(args.verify_finger_seek_duration_max, finger_max / verify_finger_speed),
        )

    run_script = out_dir / "run_best_once.sh"

    lines = []
    lines.append("#!/usr/bin/env bash")
    lines.append("set -e")
    lines.append("cd ~/Projects/o7_mujoco_sim")
    lines.append("source ~/mujoco_env/bin/activate")
    lines.append("")
    lines.append("echo '===== P2 best only ====='")
    lines.append(shell_join([
        "python3",
        rel(P2_SCRIPT),
        "--urdf", args.urdf,
        "--model", args.model,
        "--candidate", rel(best_candidate),
        "--runner-json", args.runner_json,
        "--object-body", args.object_body,
        "--target-frame", args.target_frame,
        "--out", rel(p2_json),
        "--random-seeds", str(args.p2_random_seeds),
        "--random-std", str(args.p2_random_std),
        "--max-iters", str(args.p2_max_iters),
        "--pos-tol", str(args.p2_pos_tol),
        "--rot-tol", str(args.p2_rot_tol),
        "--rot-weight", str(args.p2_rot_weight),
    ]) + f" 2>&1 | tee {shlex.quote(rel(out_dir / 'best_p2.txt'))}")
    lines.append("")
    lines.append("echo '===== P3 best only ====='")
    lines.append(shell_join([
        "python3",
        rel(P3_SCRIPT),
        "--p2-json", rel(p2_json),
        "--model", args.model,
        "--candidate", rel(best_candidate),
        "--object-body", args.object_body,
        "--out", rel(p3_json),
        "--best-plan-out", rel(best_plan),
        "--top-per-target", str(args.p3_top_per_target),
        "--max-combos", str(args.p3_max_combos),
        "--path-samples", str(args.p3_path_samples),
        "--min-hand-support-clearance", str(args.p3_min_hand_support_clearance),
        "--min-fr3-object-clearance", str(args.p3_min_fr3_object_clearance),
        "--max-grasp-hand-object-distance", str(args.p3_max_grasp_hand_object_distance),
        "--min-joint-margin", str(args.p3_min_joint_margin),
    ]) + f" 2>&1 | tee {shlex.quote(rel(out_dir / 'best_p3.txt'))}")
    lines.append("")
    lines.append("echo '===== P4C best viewer ====='")
    p4c_cmd = [
        str(RUN_CLEAN),
        rel(P4C_SCRIPT),
        "--model", args.model,
        "--candidate", rel(best_candidate),
        "--p3-json", rel(p3_json),
        "--which", "best_available",
        "--object-body", args.object_body,
        "--out", rel(p4c_json),
        "--viewer",
        "--move-steps", str(args.verify_move_steps),
        "--thumb-preshape-steps", str(args.verify_thumb_preshape_steps),
        "--thumb-roll-preshape", f"{thumb_roll:.8f}",
        "--thumb-yaw-preshape", f"{thumb_yaw:.8f}",
        "--thumb-pitch-open", f"{thumb_pitch:.8f}",
        "--finger-seek-duration", f"{verify_finger_duration:.6f}",
        "--thumb-comp-duration", str(args.verify_thumb_comp_duration),
        "--micro-squeeze-duration", str(args.verify_micro_squeeze_duration),
        "--hold-duration", str(args.verify_hold_duration),
        "--lift-duration", str(args.verify_lift_duration),
        "--finger-seek-speed", str(args.verify_finger_seek_speed),
        "--thumb-comp-speed", str(args.verify_thumb_comp_speed),
        "--micro-finger-speed", str(args.verify_micro_finger_speed),
        "--micro-thumb-speed", str(args.verify_micro_thumb_speed),
        "--soft-object-push-disp", str(args.verify_soft_object_push_disp),
        "--hard-object-push-disp", str(args.verify_hard_object_push_disp),
        "--micro-push-increase-limit", str(args.verify_micro_push_increase_limit),
        "--min-total-object-groups", str(args.verify_min_total_object_groups),
        "--min-non-thumb-groups", str(args.verify_min_non_thumb_groups),
        "--min-lift-rise-success", str(args.verify_min_lift_rise_success),
        "--frame-sleep", str(args.verify_frame_sleep),
    ]

    if args.verify_no_fail_on_hand_support:
        p4c_cmd.append("--no-fail-on-hand-support")

    lines.append(shell_join(p4c_cmd) + f" 2>&1 | tee {shlex.quote(rel(out_dir / 'best_p4c_viewer.txt'))}")

    with open(run_script, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")

    os.chmod(run_script, 0o755)

    return {
        "run_script": str(run_script),
        "p2_json": str(p2_json),
        "p3_json": str(p3_json),
        "best_plan": str(best_plan),
        "p4c_json": str(p4c_json),
        "verify_thumb_roll": thumb_roll,
        "verify_thumb_yaw": thumb_yaw,
        "verify_thumb_pitch_open": thumb_pitch,
        "verify_finger_seek_duration": verify_finger_duration,
    }


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--p3-json", required=True)
    ap.add_argument("--which", default="best_available", choices=["best_available", "best_pass"])
    ap.add_argument("--object-body", default="grasp_can")
    ap.add_argument("--target-body", default="fr3_link7")
    ap.add_argument("--target-frame", default="fr3_link7")

    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--local-dx-list", default="-0.012 -0.006 0 0.006 0.012")
    ap.add_argument("--local-dy-list", default="-0.012 -0.006 0 0.006 0.012")
    ap.add_argument("--local-dz-list", default="-0.004 0 0.006")
    ap.add_argument("--local-yaw-deg-list", default="-5 0 5")

    ap.add_argument("--thumb-alpha-list", default="0.35 0.55 0.75 0.95")
    ap.add_argument("--finger-beta-list", default="0.00 0.25 0.50")
    ap.add_argument("--thumb-pitch-beta-list", default="0.00 0.20 0.40")

    ap.add_argument("--top-k", type=int, default=20)

    ap.add_argument("--z-weight", type=float, default=0.35)
    ap.add_argument("--good-surface-tol", type=float, default=0.018)
    ap.add_argument("--good-z-tol", type=float, default=0.030)
    ap.add_argument("--min-support-clearance", type=float, default=0.006)
    ap.add_argument("--preferred-finger-max", type=float, default=0.42)

    ap.add_argument("--w-thumb-surface", type=float, default=4.0)
    ap.add_argument("--w-four-surface", type=float, default=5.0)
    ap.add_argument("--w-opposition", type=float, default=2.0)
    ap.add_argument("--w-support", type=float, default=30.0)
    ap.add_argument("--w-finger-curl", type=float, default=3.0)
    ap.add_argument("--w-local-delta", type=float, default=0.7)
    ap.add_argument("--w-yaw", type=float, default=0.3)
    ap.add_argument("--w-good-four-group", type=float, default=0.12)
    ap.add_argument("--w-opposition-bonus", type=float, default=0.25)

    # 下面这些只用于生成 best 一次验证脚本，不在 fast scoring 中使用。
    ap.add_argument("--urdf", default="models/fr3_o7/fr3_o7_mujoco_ready_obj.urdf")
    ap.add_argument("--runner-json", default="diagnostics/current_v412/v4_12a_can52_strict_clearance_debug_runner.json")

    ap.add_argument("--p2-random-seeds", type=int, default=12)
    ap.add_argument("--p2-random-std", type=float, default=0.6)
    ap.add_argument("--p2-max-iters", type=int, default=300)
    ap.add_argument("--p2-pos-tol", type=float, default=0.00025)
    ap.add_argument("--p2-rot-tol", type=float, default=0.0025)
    ap.add_argument("--p2-rot-weight", type=float, default=0.55)

    ap.add_argument("--p3-top-per-target", type=int, default=6)
    ap.add_argument("--p3-max-combos", type=int, default=216)
    ap.add_argument("--p3-path-samples", type=int, default=32)
    ap.add_argument("--p3-min-hand-support-clearance", type=float, default=0.0)
    ap.add_argument("--p3-min-fr3-object-clearance", type=float, default=0.0)
    ap.add_argument("--p3-max-grasp-hand-object-distance", type=float, default=0.045)
    ap.add_argument("--p3-min-joint-margin", type=float, default=0.0)

    ap.add_argument("--verify-move-steps", type=int, default=100)
    ap.add_argument("--verify-thumb-preshape-steps", type=int, default=100)
    ap.add_argument("--verify-finger-seek-speed", type=float, default=0.35)
    ap.add_argument("--verify-finger-seek-duration-min", type=float, default=0.35)
    ap.add_argument("--verify-finger-seek-duration-max", type=float, default=1.05)
    ap.add_argument("--verify-finger-seek-duration", type=float, default=0.75)
    ap.add_argument("--verify-thumb-comp-duration", type=float, default=1.2)
    ap.add_argument("--verify-micro-squeeze-duration", type=float, default=0.0)
    ap.add_argument("--verify-hold-duration", type=float, default=0.5)
    ap.add_argument("--verify-lift-duration", type=float, default=1.5)
    ap.add_argument("--verify-thumb-comp-speed", type=float, default=0.25)
    ap.add_argument("--verify-micro-finger-speed", type=float, default=0.0)
    ap.add_argument("--verify-micro-thumb-speed", type=float, default=0.0)
    ap.add_argument("--verify-soft-object-push-disp", type=float, default=0.004)
    ap.add_argument("--verify-hard-object-push-disp", type=float, default=0.012)
    ap.add_argument("--verify-micro-push-increase-limit", type=float, default=0.001)
    ap.add_argument("--verify-min-total-object-groups", type=int, default=2)
    ap.add_argument("--verify-min-non-thumb-groups", type=int, default=1)
    ap.add_argument("--verify-min-lift-rise-success", type=float, default=0.015)
    ap.add_argument("--verify-frame-sleep", type=float, default=0.002)
    ap.add_argument("--verify-no-fail-on-hand-support", action="store_true")

    args = ap.parse_args()

    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolve_path(args.model)
    candidate_path = resolve_path(args.candidate)
    p3_path = resolve_path(args.p3_json)

    for p in [model_path, candidate_path, p3_path, P2_SCRIPT, P3_SCRIPT, P4C_SCRIPT, RUN_CLEAN]:
        if not p.exists():
            raise RuntimeError(f"missing path: {p}")

    candidate = load_json(candidate_path)
    p3 = load_json(p3_path)
    best = selected_best(p3, args.which)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    q_grasp = best["q_grasp"]
    apply_arm_q(model, data, q_grasp)
    mujoco.mj_forward(model, data)

    T_world_object = object_T_world(model, data, args.object_body)
    T_world_target_old = target_T_world(model, data, args.target_body)

    object_points, object_geoms = collect_object_points(model, data, args.object_body)
    object_geom = estimate_upright_object_geometry(object_points)
    support_top, support_geoms = collect_support_top(model, data)

    prior_ctrl, prior_source = extract_candidate_prior(candidate, model)
    open_ctrl = make_open_ctrl(model)

    dxs = parse_float_list(args.local_dx_list)
    dys = parse_float_list(args.local_dy_list)
    dzs = parse_float_list(args.local_dz_list)
    yaws = parse_float_list(args.local_yaw_deg_list)

    thumb_alphas = parse_float_list(args.thumb_alpha_list)
    finger_betas = parse_float_list(args.finger_beta_list)
    thumb_pitch_betas = parse_float_list(args.thumb_pitch_beta_list)

    print("\n========== V4.12P4E-FAST HAND LOCAL POSE SCORE ==========")
    print("model        :", model_path)
    print("candidate    :", candidate_path)
    print("p3_json      :", p3_path)
    print("which        :", args.which)
    print("object_body  :", args.object_body)
    print("target_body  :", args.target_body)
    print("out_dir      :", out_dir)
    print("prior_source :", prior_source)
    print("prior_ctrl   :", prior_ctrl)
    print("open_ctrl    :", open_ctrl)
    print("object_geom  :", object_geom)
    print("support_top  :", support_top)
    print("object_geoms :", [geom_name(model, g) for g in object_geoms])
    print("support_geoms:", [geom_name(model, g) for g in support_geoms])
    print("local grid   :", len(dxs), len(dys), len(dzs), len(yaws))
    print("hand grid    :", len(thumb_alphas), len(finger_betas), len(thumb_pitch_betas))
    print("=========================================================\n")

    records = []
    total = (
        len(dxs) * len(dys) * len(dzs) * len(yaws)
        * len(thumb_alphas) * len(finger_betas) * len(thumb_pitch_betas)
    )
    counter = 0

    for ta in thumb_alphas:
        for fb in finger_betas:
            for tb in thumb_pitch_betas:
                ctrl = make_adaptive_ctrl(
                    model=model,
                    open_ctrl=open_ctrl,
                    prior_ctrl=prior_ctrl,
                    thumb_alpha=ta,
                    finger_beta=fb,
                    thumb_pitch_beta=tb,
                )

                mujoco.mj_resetData(model, data)
                mujoco.mj_forward(model, data)
                apply_arm_q(model, data, q_grasp)
                apply_hand_ctrl_as_qpos(model, data, ctrl)
                mujoco.mj_forward(model, data)

                base_hand_pts, hand_geom_names = collect_hand_points_by_group(model, data)

                for dx in dxs:
                    for dy in dys:
                        for dz in dzs:
                            for yaw in yaws:
                                counter += 1

                                Tdl = T_delta_local(dx, dy, dz, yaw)

                                moved_pts = {}
                                for g, pts in base_hand_pts.items():
                                    moved_pts[g] = transform_points_world_by_target_delta(
                                        pts,
                                        T_world_target_old,
                                        Tdl,
                                    )

                                sample_meta = {
                                    "delta_local": np.array([dx, dy, dz], dtype=float),
                                    "yaw_deg": float(yaw),
                                    "thumb_alpha": float(ta),
                                    "finger_beta": float(fb),
                                    "thumb_pitch_beta": float(tb),
                                }

                                detail = score_one(
                                    hand_pts=moved_pts,
                                    obj_geom=object_geom,
                                    support_top=support_top,
                                    ctrl=ctrl,
                                    sample_meta=sample_meta,
                                    args=args,
                                )

                                rec = {
                                    "score": detail["score"],
                                    "sample_meta": sample_meta,
                                    "hand_config": {
                                        "thumb_alpha": float(ta),
                                        "finger_beta": float(fb),
                                        "thumb_pitch_beta": float(tb),
                                        "ctrl": dict(ctrl),
                                    },
                                    "score_detail": detail,
                                }
                                records.append(rec)

    records_sorted = sorted(records, key=lambda r: float(r["score"]))
    topk = records_sorted[:args.top_k]
    best_record = records_sorted[0]

    T_best_delta = T_delta_local(
        best_record["sample_meta"]["delta_local"][0],
        best_record["sample_meta"]["delta_local"][1],
        best_record["sample_meta"]["delta_local"][2],
        best_record["sample_meta"]["yaw_deg"],
    )

    patched_candidate, patch_info = patch_candidate_target(
        candidate=candidate,
        T_world_object=T_world_object,
        T_world_target_old=T_world_target_old,
        T_delta=T_best_delta,
        best_meta={
            "best_record": best_record,
            "prior_source": prior_source,
            "prior_ctrl": prior_ctrl,
            "open_ctrl": open_ctrl,
            "object_geom": object_geom,
            "support_top": support_top,
        },
    )

    best_candidate_path = out_dir / "best_candidate.json"
    best_config_path = out_dir / "best_config.json"
    summary_path = out_dir / "summary.json"
    top_txt_path = out_dir / "topk_summary.txt"

    save_json(best_candidate_path, patched_candidate)

    best_config = {
        "format": "v4_12p4e_fast_best_config_debug",
        "best_record": best_record,
        "prior_source": prior_source,
        "prior_ctrl": prior_ctrl,
        "open_ctrl": open_ctrl,
        "object_geom": object_geom,
        "support_top": support_top,
        "patch_info": patch_info,
        "best_candidate": str(best_candidate_path),
    }
    save_json(best_config_path, best_config)

    script_info = write_best_scripts(args, out_dir, best_record)

    summary = {
        "format": "v4_12p4e_fast_hand_local_pose_score_debug",
        "model": str(model_path),
        "candidate": str(candidate_path),
        "p3_json": str(p3_path),
        "which": args.which,
        "object_body": args.object_body,
        "target_body": args.target_body,
        "args": vars(args),
        "num_samples": len(records),
        "prior_source": prior_source,
        "prior_ctrl": prior_ctrl,
        "open_ctrl": open_ctrl,
        "object_geom": object_geom,
        "support_top": support_top,
        "best_candidate": str(best_candidate_path),
        "best_config": str(best_config_path),
        "script_info": script_info,
        "topk": topk,
    }
    save_json(summary_path, summary)

    with open(top_txt_path, "w") as f:
        f.write("rank,score,delta_local,yaw,thumb_alpha,finger_beta,thumb_pitch_beta,reasons,ctrl\n")
        for i, r in enumerate(topk, 1):
            sm = r["sample_meta"]
            hc = r["hand_config"]
            sd = r["score_detail"]
            f.write(
                f"{i},"
                f"{r['score']:.6f},"
                f"{to_jsonable(sm['delta_local'])},"
                f"{sm['yaw_deg']},"
                f"{sm['thumb_alpha']},"
                f"{sm['finger_beta']},"
                f"{sm['thumb_pitch_beta']},"
                f"{sd.get('reasons')},"
                f"{hc['ctrl']}\n"
            )

    print("\n========== P4E-FAST SUMMARY ==========")
    print("num_samples:", len(records))
    print("summary    :", summary_path)
    print("top_txt    :", top_txt_path)
    print("best_candidate:", best_candidate_path)
    print("best_config   :", best_config_path)
    print("run_best_once :", script_info["run_script"])

    for i, r in enumerate(topk[:10], 1):
        sm = r["sample_meta"]
        sd = r["score_detail"]
        hc = r["hand_config"]
        print(
            f"{i:02d}. score={r['score']:.6f} "
            f"delta={to_jsonable(sm['delta_local'])} "
            f"yaw={sm['yaw_deg']:+.1f} "
            f"alpha={sm['thumb_alpha']:.2f} "
            f"beta={sm['finger_beta']:.2f} "
            f"tp_beta={sm['thumb_pitch_beta']:.2f} "
            f"thumb_side={sd['thumb']['side_error']:.4f} "
            f"four={sd['best_four_group']}:{sd['best_four']['side_error']:.4f} "
            f"dot={sd['opposition_dot']:.3f} "
            f"support={sd['support_clearance']:.4f} "
            f"finger_max={sd['finger_ctrl_max']:.3f} "
            f"ctrl={ {k: round(v, 4) for k, v in hc['ctrl'].items()} }"
        )

    print("\nBest verify command:")
    print(f"bash {script_info['run_script']}")
    print("======================================\n")


if __name__ == "__main__":
    main()
