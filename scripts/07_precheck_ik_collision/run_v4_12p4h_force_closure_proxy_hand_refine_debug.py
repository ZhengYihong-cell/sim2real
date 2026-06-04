#!/usr/bin/env python3
"""
文件名：
    run_v4_12p4h_force_closure_proxy_hand_refine_debug.py

脚本类别：
    debug / fast-geometric-score / force-closure-proxy / hand-local-refinement

用途：
    本脚本用于 V4.12P4H 阶段。
    当前 P4G 已经证明：只调大拇指无法形成稳定抓握，物体没有落在大拇指与四指的对握力线上。
    因此本脚本引入力闭合代理评分，用于快速选择更合理的 hand-local 局部修正。

核心思想：
    不对每个局部扰动点跑 P2 / P3 / P4F。
    而是在当前 q_grasp 和 candidate hand prior close target 下，做纯几何代理评分：

        1. thumb 接触候选点是否接近物体侧壁；
        2. 至少一个非拇指 finger 接触候选点是否接近物体另一侧侧壁；
        3. 物体中心轴是否接近 thumb-finger 力线；
        4. thumb 与 finger 的径向方向是否对抗；
        5. thumb 与 finger 接触高度是否接近；
        6. hand 是否过低接近支撑块；
        7. hand local 修正量是否过大。

    评分最低的局部修正会被写入 best_candidate.json，并自动生成只验证 best 一次的脚本。

输入：
    1. --model
       MuJoCo XML 场景。
    2. --candidate
       原始 candidate JSON，必须包含 target.T_object_target 和 hand/o7 ctrl。
    3. --p3-json
       当前 P3 输出 JSON，用于读取 best_available / best_pass 的 q_grasp。
    4. --object-body
       物体 body 名称，例如 grasp_can。
    5. --target-body
       hand target body，当前通常是 fr3_link7。

输出：
    1. --out-dir/summary.json
       全部局部扰动的 Top-K 评分结果。
    2. --out-dir/topk_summary.txt
       可读排行榜。
    3. --out-dir/best_candidate.json
       根据 best hand-local delta patch 后的 candidate。
    4. --out-dir/best_force_proxy_config.json
       best 的局部修正、手型、力闭合代理指标。
    5. --out-dir/run_best_force_proxy_viewer.sh
       只对 best 跑一次 P2 / P3 / P4F viewer 的验证脚本。

当前流程位置：
    candidate prior
        -> P3 best q_grasp
        -> P4H force-closure-proxy hand-local refinement
        -> 只验证 best 一次
        -> 若仍不行，再进入接触阶段小范围力闭合微调

本脚本不负责：
    1. 不对每个扰动点重新做 IK。
    2. 不对每个扰动点重新做 P3 碰撞搜索。
    3. 不跑每个扰动点的动态抓取。
    4. 不做完整 wrench-space force closure 求解。
    5. 不大范围修改整臂位姿。
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
P4F_SCRIPT = PROJECT / "scripts/05_execution_runner/run_v4_12p4f_target_close_debug.py"

ARM_JOINTS = [
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
]

HAND_JOINTS = [
    "thumb_cmc_roll",
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]

FOUR_FINGER_JOINTS = [
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]

FINGER_GROUPS = ["index", "middle", "ring", "pinky"]


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


def shell_join(cmd):
    return " ".join(shlex.quote(str(x)) for x in cmd)


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


def geom_body_name(model, gid):
    return body_name(model, int(model.geom_bodyid[int(gid)]))


def body_is_descendant(model, bid, root_bid):
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


def T_body(model, data, body_name_):
    bid = body_id(model, body_name_)
    if bid < 0:
        raise RuntimeError(f"cannot find body: {body_name_}")
    return T_from_Rp(data.xmat[bid].reshape(3, 3), data.xpos[bid])


def T_inv(T):
    T = np.asarray(T, dtype=float)
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def T_delta_local(dx, dy, dz, yaw_deg, roll_deg):
    T = np.eye(4)
    R = Rz(math.radians(yaw_deg)) @ Rx(math.radians(roll_deg))
    T[:3, :3] = R
    T[:3, 3] = np.array([dx, dy, dz], dtype=float)
    return T


def transform_points(points, T_world_target, T_delta):
    pts = np.asarray(points, dtype=float)
    if pts.size == 0:
        return pts.reshape(0, 3)
    T_map = T_world_target @ T_delta @ T_inv(T_world_target)
    ph = np.concatenate([pts, np.ones((pts.shape[0], 1))], axis=1)
    return (T_map @ ph.T).T[:, :3]


def ctrl_range(model, joint_name):
    for nm in [joint_name, f"{joint_name}_pos", f"{joint_name}_ctrl", f"{joint_name}_act", f"{joint_name}_motor"]:
        aid = actuator_id(model, nm)
        if aid >= 0 and bool(model.actuator_ctrllimited[aid]):
            lo, hi = model.actuator_ctrlrange[aid]
            return float(lo), float(hi)

    jid = joint_id(model, joint_name)
    if jid >= 0 and bool(model.jnt_limited[jid]):
        lo, hi = model.jnt_range[jid]
        return float(lo), float(hi)

    return -3.0, 3.0


def clamp_ctrl(model, joint_name, value):
    lo, hi = ctrl_range(model, joint_name)
    return float(np.clip(float(value), lo, hi))


def set_joint_qpos(model, data, joint_name, value):
    jid = joint_id(model, joint_name)
    if jid < 0:
        return False
    if int(model.jnt_type[jid]) not in [
        int(mujoco.mjtJoint.mjJNT_HINGE),
        int(mujoco.mjtJoint.mjJNT_SLIDE),
    ]:
        return False
    qadr = int(model.jnt_qposadr[jid])
    data.qpos[qadr] = clamp_ctrl(model, joint_name, value)
    return True


def set_actuator_ctrl(model, data, joint_name, value):
    for nm in [joint_name, f"{joint_name}_pos", f"{joint_name}_ctrl", f"{joint_name}_act", f"{joint_name}_motor"]:
        aid = actuator_id(model, nm)
        if aid >= 0:
            data.ctrl[aid] = clamp_ctrl(model, joint_name, value)
            return True
    return False


def apply_arm_q(model, data, q):
    for j in ARM_JOINTS:
        if j in q:
            set_joint_qpos(model, data, j, q[j])
            set_actuator_ctrl(model, data, j, q[j])


def apply_hand_qpos_ctrl(model, data, ctrl):
    for j in HAND_JOINTS:
        if j in ctrl:
            set_joint_qpos(model, data, j, ctrl[j])
            set_actuator_ctrl(model, data, j, ctrl[j])


def selected_plan(p3, which):
    item = p3.get(which)
    if item is None:
        raise RuntimeError(f"{which} is None in p3 json")
    for k in ["q_pre", "q_grasp", "q_lift"]:
        if k not in item:
            raise RuntimeError(f"{which} missing {k}")
    return item


def extract_candidate_ctrl(candidate, model):
    hand = candidate.get("hand", {}) or {}

    for key in ["o7_active_ctrl", "active_ctrl", "ctrl", "target_ctrl", "qpos", "target_qpos"]:
        val = hand.get(key)
        if isinstance(val, dict):
            out = {}
            for j in HAND_JOINTS:
                if j in val:
                    out[j] = clamp_ctrl(model, j, val[j])
            if out:
                return out, f"hand.{key}"

    for key in ["o7_active_ctrl", "active_ctrl", "ctrl", "target_ctrl", "qpos", "target_qpos"]:
        val = candidate.get(key)
        if isinstance(val, dict):
            out = {}
            for j in HAND_JOINTS:
                if j in val:
                    out[j] = clamp_ctrl(model, j, val[j])
            if out:
                return out, key

    raise RuntimeError("candidate has no usable O7 hand ctrl/qpos")


def extract_best_config_ctrl(best_config, model):
    if not best_config:
        return {}, ""
    rec = best_config.get("best_record", {}) or {}
    hc = rec.get("hand_config", {}) or {}
    ctrl = hc.get("ctrl", {}) or {}

    out = {}
    for j in HAND_JOINTS:
        if j in ctrl:
            out[j] = clamp_ctrl(model, j, ctrl[j])
    return out, "best_record.hand_config.ctrl" if out else ""


def make_open_ctrl(model):
    return {j: clamp_ctrl(model, j, 0.0) for j in HAND_JOINTS}


def make_close_ctrl(model, open_ctrl, candidate_ctrl, best_ctrl, args):
    ctrl = dict(open_ctrl)

    thumb_source = best_ctrl if best_ctrl else candidate_ctrl

    ctrl["thumb_cmc_roll"] = clamp_ctrl(
        model,
        "thumb_cmc_roll",
        thumb_source.get("thumb_cmc_roll", candidate_ctrl.get("thumb_cmc_roll", 0.0)),
    )
    ctrl["thumb_cmc_yaw"] = clamp_ctrl(
        model,
        "thumb_cmc_yaw",
        thumb_source.get("thumb_cmc_yaw", candidate_ctrl.get("thumb_cmc_yaw", 0.0)),
    )

    max_four_prior = max(abs(float(candidate_ctrl.get(j, 0.0))) for j in FOUR_FINGER_JOINTS)

    ctrl["thumb_cmc_pitch"] = clamp_ctrl(
        model,
        "thumb_cmc_pitch",
        max(
            candidate_ctrl.get("thumb_cmc_pitch", 0.0),
            thumb_source.get("thumb_cmc_pitch", 0.0) + args.thumb_pitch_from_finger_gain * max_four_prior,
        ),
    )

    for j in FOUR_FINGER_JOINTS:
        ctrl[j] = clamp_ctrl(model, j, candidate_ctrl.get(j, 0.0) * args.finger_close_scale)

    return ctrl


def geom_world_points(model, data, gid, max_mesh_points=80):
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
            if len(verts) > max_mesh_points:
                stride = max(1, len(verts) // max_mesh_points)
                verts = verts[::stride][:max_mesh_points]
            return (R @ verts.T).T + pos

    size = np.asarray(model.geom_size[gid], dtype=float).reshape(-1)
    mags = [
        float(size[0]) if len(size) > 0 else 0.02,
        float(size[1]) if len(size) > 1 else float(size[0]) if len(size) > 0 else 0.02,
        float(size[2]) if len(size) > 2 else float(size[0]) if len(size) > 0 else 0.02,
    ]

    pts = [np.zeros(3)]
    axes = np.eye(3)
    for i in range(3):
        pts.append(axes[i] * mags[i])
        pts.append(-axes[i] * mags[i])

    pts = np.asarray(pts, dtype=float)
    return (R @ pts.T).T + pos


def collect_object_points(model, data, object_body):
    root = body_id(model, object_body)
    if root < 0:
        raise RuntimeError(f"cannot find object body: {object_body}")

    pts = []
    for gid in range(model.ngeom):
        bid = int(model.geom_bodyid[gid])
        if body_is_descendant(model, bid, root):
            pts.append(geom_world_points(model, data, gid, max_mesh_points=200))

    if not pts:
        raise RuntimeError(f"no object geoms found for {object_body}")

    return np.concatenate(pts, axis=0)


def estimate_object_geom(points):
    pts = np.asarray(points, dtype=float)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    center = 0.5 * (mn + mx)

    rx = 0.5 * float(mx[0] - mn[0])
    ry = 0.5 * float(mx[1] - mn[1])
    rz = 0.5 * float(mx[2] - mn[2])

    return {
        "bbox_min": mn,
        "bbox_max": mx,
        "center": center,
        "center_xy": center[:2],
        "z_min": float(mn[2]),
        "z_max": float(mx[2]),
        "height": float(mx[2] - mn[2]),
        "radius_xy": float(max(rx, ry)),
        "rx": rx,
        "ry": ry,
        "rz": rz,
    }


def hand_group_of_geom(model, gid):
    text = f"{geom_name(model, gid)} {geom_body_name(model, gid)}".lower()

    if "thumb" in text:
        return "thumb"
    if "index" in text:
        return "index"
    if "middle" in text:
        return "middle"
    if "ring" in text:
        return "ring"
    if "pinky" in text:
        return "pinky"
    if "palm" in text or "hand" in text:
        return "palm"

    return ""


def collect_hand_points(model, data):
    groups = {
        "thumb": [],
        "index": [],
        "middle": [],
        "ring": [],
        "pinky": [],
        "palm": [],
    }

    meta = {k: [] for k in groups}

    for gid in range(model.ngeom):
        g = hand_group_of_geom(model, gid)
        if not g:
            continue
        pts = geom_world_points(model, data, gid, max_mesh_points=50)
        groups[g].append(pts)
        meta[g].append({
            "geom": geom_name(model, gid),
            "body": geom_body_name(model, gid),
            "npts": int(len(pts)),
        })

    out = {}
    for g in groups:
        if groups[g]:
            out[g] = np.concatenate(groups[g], axis=0)
        else:
            out[g] = np.zeros((0, 3), dtype=float)

    return out, meta


def collect_support_top(model, data):
    max_z = 0.0
    names = []

    for gid in range(model.ngeom):
        text = f"{geom_name(model, gid)} {geom_body_name(model, gid)}".lower()
        if "pedestal" in text or "support" in text or "table" in text:
            pts = geom_world_points(model, data, gid, max_mesh_points=80)
            max_z = max(max_z, float(np.max(pts[:, 2])))
            names.append(geom_name(model, gid))

    return max_z, names


def side_surface_best(points, obj, args):
    pts = np.asarray(points, dtype=float)
    if pts.shape[0] == 0:
        return {
            "valid": False,
            "point": np.zeros(3),
            "side_error": 999.0,
            "z_error": 999.0,
            "radial_unit": np.zeros(2),
            "score": 999.0,
        }

    cxy = np.asarray(obj["center_xy"], dtype=float)
    r = float(obj["radius_xy"])
    zmin = float(obj["z_min"])
    zmax = float(obj["z_max"])

    xy = pts[:, :2]
    z = pts[:, 2]

    rv = xy - cxy.reshape(1, 2)
    radial = np.linalg.norm(rv, axis=1)

    side_error = np.abs(radial - r)

    z_error = np.maximum(0.0, zmin - z) + np.maximum(0.0, z - zmax)

    score = side_error + args.z_weight * z_error
    idx = int(np.argmin(score))

    ru = rv[idx] / (radial[idx] + 1e-12)

    return {
        "valid": True,
        "point": pts[idx],
        "side_error": float(side_error[idx]),
        "z_error": float(z_error[idx]),
        "radial": float(radial[idx]),
        "radial_unit": ru,
        "score": float(score[idx]),
    }


def line_metrics(thumb, finger, obj):
    pt = np.asarray(thumb["point"], dtype=float)
    pf = np.asarray(finger["point"], dtype=float)

    cxy = np.asarray(obj["center_xy"], dtype=float)

    vt = pf[:2] - pt[:2]
    nvt = float(np.linalg.norm(vt))

    if nvt < 1e-9:
        line_dist_xy = 999.0
        alpha = -999.0
    else:
        w = cxy - pt[:2]
        alpha = float(np.dot(w, vt) / (nvt * nvt))
        cross = abs(vt[0] * w[1] - vt[1] * w[0])
        line_dist_xy = float(cross / nvt)

    z_diff = abs(float(pt[2] - pf[2]))
    dot = float(np.dot(thumb["radial_unit"], finger["radial_unit"]))

    alpha_penalty = max(0.0, -alpha) + max(0.0, alpha - 1.0)

    return {
        "line_dist_xy": line_dist_xy,
        "alpha_on_segment": alpha,
        "alpha_penalty": alpha_penalty,
        "z_diff": z_diff,
        "opposition_dot": dot,
        "opposition_error": abs(dot + 1.0),
    }


def score_one(moved_hand, obj, support_top, delta_meta, close_ctrl, args):
    thumb = side_surface_best(moved_hand["thumb"], obj, args)

    finger_bests = {}
    for g in FINGER_GROUPS:
        finger_bests[g] = side_surface_best(moved_hand[g], obj, args)

    valid_fingers = {g: v for g, v in finger_bests.items() if v["valid"]}

    if not thumb["valid"] or not valid_fingers:
        return {
            "score": 1e9,
            "reasons": ["missing hand points"],
        }

    candidates = []

    for g, finger in valid_fingers.items():
        lm = line_metrics(thumb, finger, obj)

        good_thumb = thumb["side_error"] <= args.good_surface_tol and thumb["z_error"] <= args.good_z_tol
        good_finger = finger["side_error"] <= args.good_surface_tol and finger["z_error"] <= args.good_z_tol
        good_line = lm["line_dist_xy"] <= args.good_line_tol
        good_opp = lm["opposition_dot"] <= args.max_opposition_dot
        good_z = lm["z_diff"] <= args.good_height_diff

        pair_score = 0.0
        pair_score += args.w_thumb_surface * thumb["side_error"]
        pair_score += args.w_finger_surface * finger["side_error"]
        pair_score += args.w_line * lm["line_dist_xy"]
        pair_score += args.w_opposition * lm["opposition_error"]
        pair_score += args.w_height * lm["z_diff"]
        pair_score += args.w_alpha * lm["alpha_penalty"]
        pair_score += args.w_z_outside * (thumb["z_error"] + finger["z_error"])

        if good_thumb:
            pair_score -= args.bonus_good_thumb
        if good_finger:
            pair_score -= args.bonus_good_finger
        if good_line and good_opp and good_z:
            pair_score -= args.bonus_force_proxy_pair

        candidates.append({
            "finger_group": g,
            "finger": finger,
            "line_metrics": lm,
            "pair_score": float(pair_score),
            "flags": {
                "good_thumb": good_thumb,
                "good_finger": good_finger,
                "good_line": good_line,
                "good_opp": good_opp,
                "good_z": good_z,
            },
        })

    best_pair = min(candidates, key=lambda x: x["pair_score"])

    all_pts = []
    for g in ["thumb", "index", "middle", "ring", "pinky", "palm"]:
        if moved_hand[g].shape[0] > 0:
            all_pts.append(moved_hand[g])
    all_pts = np.concatenate(all_pts, axis=0) if all_pts else np.zeros((0, 3))
    min_hand_z = float(np.min(all_pts[:, 2])) if all_pts.shape[0] else 999.0
    support_clearance = min_hand_z - float(support_top)
    support_penalty = max(0.0, args.min_support_clearance - support_clearance)

    delta_norm = float(np.linalg.norm(delta_meta["delta_local"]))
    yaw_abs = abs(float(delta_meta["yaw_deg"]))
    roll_abs = abs(float(delta_meta["roll_deg"]))

    finger_ctrl_max = max(abs(float(close_ctrl.get(j, 0.0))) for j in FOUR_FINGER_JOINTS)

    score = float(best_pair["pair_score"])
    score += args.w_support * support_penalty
    score += args.w_delta * delta_norm
    score += args.w_yaw * math.radians(yaw_abs)
    score += args.w_roll * math.radians(roll_abs)
    score += args.w_finger_curl * max(0.0, finger_ctrl_max - args.preferred_finger_max)

    reasons = [
        f"finger={best_pair['finger_group']}",
        f"thumb_side={thumb['side_error']:.4f}",
        f"finger_side={best_pair['finger']['side_error']:.4f}",
        f"line={best_pair['line_metrics']['line_dist_xy']:.4f}",
        f"dot={best_pair['line_metrics']['opposition_dot']:.3f}",
        f"z_diff={best_pair['line_metrics']['z_diff']:.4f}",
        f"alpha={best_pair['line_metrics']['alpha_on_segment']:.3f}",
        f"support_clear={support_clearance:.4f}",
    ]

    return {
        "score": float(score),
        "reasons": reasons,
        "thumb": thumb,
        "finger_bests": finger_bests,
        "best_pair": best_pair,
        "support_clearance": support_clearance,
        "support_penalty": support_penalty,
        "min_hand_z": min_hand_z,
        "delta_norm": delta_norm,
        "finger_ctrl_max": finger_ctrl_max,
    }


def patch_candidate(candidate, T_world_object, T_world_target_old, T_delta, best_record):
    T_world_target_new = T_world_target_old @ T_delta
    T_object_target_new = T_inv(T_world_object) @ T_world_target_new

    patched = json.loads(json.dumps(candidate))
    patched["target"]["T_object_target"] = T_object_target_new.tolist()

    meta = patched.setdefault("debug_patch_meta", {})
    meta["v4_12p4h_force_closure_proxy"] = to_jsonable({
        "best_record": best_record,
        "T_object_target_new": T_object_target_new,
        "T_world_target_new": T_world_target_new,
    })

    return patched, {
        "T_world_target_new": T_world_target_new,
        "T_object_target_new": T_object_target_new,
    }


def write_best_script(args, out_dir):
    out_dir = resolve_path(out_dir)

    script = out_dir / "run_best_force_proxy_viewer.sh"

    p2_json = out_dir / "best_p2.json"
    p3_json = out_dir / "best_p3.json"
    best_plan_json = out_dir / "best_plan.json"
    p4f_json = out_dir / "best_p4f_force_proxy_viewer.json"

    best_candidate = out_dir / "best_candidate.json"
    best_config = out_dir / "best_force_proxy_config.json"

    lines = []
    lines.append("#!/usr/bin/env bash")
    lines.append("set -e")
    lines.append("cd ~/Projects/o7_mujoco_sim")
    lines.append("source ~/mujoco_env/bin/activate")
    lines.append("")
    lines.append("echo '===== P2 best force-proxy candidate ====='")
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
    lines.append("echo '===== P3 best force-proxy candidate ====='")
    lines.append(shell_join([
        "python3",
        rel(P3_SCRIPT),
        "--p2-json", rel(p2_json),
        "--model", args.model,
        "--candidate", rel(best_candidate),
        "--object-body", args.object_body,
        "--out", rel(p3_json),
        "--best-plan-out", rel(best_plan_json),
        "--top-per-target", str(args.p3_top_per_target),
        "--max-combos", str(args.p3_max_combos),
        "--path-samples", str(args.p3_path_samples),
        "--min-hand-support-clearance", str(args.p3_min_hand_support_clearance),
        "--min-fr3-object-clearance", str(args.p3_min_fr3_object_clearance),
        "--max-grasp-hand-object-distance", str(args.p3_max_grasp_hand_object_distance),
        "--min-joint-margin", str(args.p3_min_joint_margin),
    ]) + f" 2>&1 | tee {shlex.quote(rel(out_dir / 'best_p3.txt'))}")
    lines.append("")
    lines.append("echo '===== P4F viewer best force-proxy candidate ====='")
    cmd = [
        str(RUN_CLEAN),
        rel(P4F_SCRIPT),
        "--model", args.model,
        "--candidate", rel(best_candidate),
        "--p3-json", rel(p3_json),
        "--best-config", rel(best_config),
        "--which", "best_available",
        "--object-body", args.object_body,
        "--out", rel(p4f_json),
        "--viewer",
        "--move-steps", str(args.verify_move_steps),
        "--thumb-preshape-steps", str(args.verify_thumb_preshape_steps),
        "--close-duration", str(args.verify_close_duration),
        "--hold-duration", str(args.verify_hold_duration),
        "--lift-duration", str(args.verify_lift_duration),
        "--finger-close-scale", str(args.finger_close_scale),
        "--thumb-pitch-from-finger-gain", str(args.thumb_pitch_from_finger_gain),
        "--hard-object-push-disp", str(args.verify_hard_object_push_disp),
        "--min-lift-rise-success", str(args.verify_min_lift_rise_success),
        "--lift-even-if-fail",
        "--keep-viewer-open",
        "--frame-sleep", str(args.verify_frame_sleep),
    ]

    lines.append(shell_join(cmd) + f" 2>&1 | tee {shlex.quote(rel(out_dir / 'best_p4f_viewer.txt'))}")

    with open(script, "w") as f:
        f.write("\n".join(lines))
        f.write("\n")

    os.chmod(script, 0o755)

    return script


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--p3-json", required=True)
    ap.add_argument("--best-config", default="")
    ap.add_argument("--which", default="best_available", choices=["best_available", "best_pass"])
    ap.add_argument("--object-body", default="grasp_can")
    ap.add_argument("--target-body", default="fr3_link7")
    ap.add_argument("--target-frame", default="fr3_link7")
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--local-dx-list", default="-0.012 -0.006 0 0.006 0.012")
    ap.add_argument("--local-dy-list", default="-0.012 -0.006 0 0.006 0.012")
    ap.add_argument("--local-dz-list", default="-0.006 0 0.006")
    ap.add_argument("--local-yaw-deg-list", default="-5 0 5")
    ap.add_argument("--local-roll-deg-list", default="-4 0 4")

    ap.add_argument("--finger-close-scale", type=float, default=1.0)
    ap.add_argument("--thumb-pitch-from-finger-gain", type=float, default=0.35)

    ap.add_argument("--top-k", type=int, default=20)

    ap.add_argument("--z-weight", type=float, default=0.30)
    ap.add_argument("--good-surface-tol", type=float, default=0.018)
    ap.add_argument("--good-z-tol", type=float, default=0.035)
    ap.add_argument("--good-line-tol", type=float, default=0.010)
    ap.add_argument("--good-height-diff", type=float, default=0.030)
    ap.add_argument("--max-opposition-dot", type=float, default=-0.35)
    ap.add_argument("--min-support-clearance", type=float, default=0.004)
    ap.add_argument("--preferred-finger-max", type=float, default=0.62)

    ap.add_argument("--w-thumb-surface", type=float, default=6.0)
    ap.add_argument("--w-finger-surface", type=float, default=6.0)
    ap.add_argument("--w-line", type=float, default=10.0)
    ap.add_argument("--w-opposition", type=float, default=4.0)
    ap.add_argument("--w-height", type=float, default=3.0)
    ap.add_argument("--w-alpha", type=float, default=4.0)
    ap.add_argument("--w-z-outside", type=float, default=3.0)
    ap.add_argument("--w-support", type=float, default=20.0)
    ap.add_argument("--w-delta", type=float, default=0.8)
    ap.add_argument("--w-yaw", type=float, default=0.5)
    ap.add_argument("--w-roll", type=float, default=0.5)
    ap.add_argument("--w-finger-curl", type=float, default=1.0)

    ap.add_argument("--bonus-good-thumb", type=float, default=0.15)
    ap.add_argument("--bonus-good-finger", type=float, default=0.20)
    ap.add_argument("--bonus-force-proxy-pair", type=float, default=0.45)

    # 只用于自动生成 best 一次验证脚本
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
    ap.add_argument("--verify-close-duration", type=float, default=1.8)
    ap.add_argument("--verify-hold-duration", type=float, default=0.8)
    ap.add_argument("--verify-lift-duration", type=float, default=2.0)
    ap.add_argument("--verify-hard-object-push-disp", type=float, default=0.020)
    ap.add_argument("--verify-min-lift-rise-success", type=float, default=0.015)
    ap.add_argument("--verify-frame-sleep", type=float, default=0.002)

    args = ap.parse_args()

    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model_path = resolve_path(args.model)
    candidate_path = resolve_path(args.candidate)
    p3_path = resolve_path(args.p3_json)
    best_config_path = resolve_path(args.best_config) if args.best_config else None

    for p in [model_path, candidate_path, p3_path, P2_SCRIPT, P3_SCRIPT, P4F_SCRIPT, RUN_CLEAN]:
        if not p.exists():
            raise RuntimeError(f"missing path: {p}")

    candidate = load_json(candidate_path)
    p3 = load_json(p3_path)
    best_config = load_json(best_config_path) if best_config_path and best_config_path.exists() else {}

    plan = selected_plan(p3, args.which)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    apply_arm_q(model, data, plan["q_grasp"])

    candidate_ctrl, candidate_source = extract_candidate_ctrl(candidate, model)
    best_ctrl, best_source = extract_best_config_ctrl(best_config, model)

    open_ctrl = make_open_ctrl(model)
    close_ctrl = make_close_ctrl(model, open_ctrl, candidate_ctrl, best_ctrl, args)

    apply_hand_qpos_ctrl(model, data, close_ctrl)
    mujoco.mj_forward(model, data)

    T_world_object = T_body(model, data, args.object_body)
    T_world_target_old = T_body(model, data, args.target_body)

    object_points = collect_object_points(model, data, args.object_body)
    object_geom = estimate_object_geom(object_points)

    base_hand_points, hand_meta = collect_hand_points(model, data)
    support_top, support_names = collect_support_top(model, data)

    dxs = parse_float_list(args.local_dx_list)
    dys = parse_float_list(args.local_dy_list)
    dzs = parse_float_list(args.local_dz_list)
    yaws = parse_float_list(args.local_yaw_deg_list)
    rolls = parse_float_list(args.local_roll_deg_list)

    print("\n========== V4.12P4H FORCE-CLOSURE PROXY ==========")
    print("model           :", model_path)
    print("candidate       :", candidate_path)
    print("p3_json         :", p3_path)
    print("best_config     :", best_config_path)
    print("which           :", args.which)
    print("object_body     :", args.object_body)
    print("target_body     :", args.target_body)
    print("out_dir         :", out_dir)
    print("candidate_source:", candidate_source)
    print("best_source     :", best_source)
    print("candidate_ctrl  :", candidate_ctrl)
    print("close_ctrl      :", close_ctrl)
    print("object_geom     :", object_geom)
    print("support_top     :", support_top, support_names)
    print("grid sizes      :", len(dxs), len(dys), len(dzs), len(yaws), len(rolls))
    print("num samples     :", len(dxs) * len(dys) * len(dzs) * len(yaws) * len(rolls))
    print("==================================================\n")

    records = []
    idx = 0

    for dx in dxs:
        for dy in dys:
            for dz in dzs:
                for yaw in yaws:
                    for roll in rolls:
                        idx += 1

                        Tdl = T_delta_local(dx, dy, dz, yaw, roll)
                        moved = {}
                        for g, pts in base_hand_points.items():
                            moved[g] = transform_points(pts, T_world_target_old, Tdl)

                        delta_meta = {
                            "delta_local": np.array([dx, dy, dz], dtype=float),
                            "yaw_deg": float(yaw),
                            "roll_deg": float(roll),
                        }

                        detail = score_one(moved, object_geom, support_top, delta_meta, close_ctrl, args)

                        rec = {
                            "index": idx,
                            "score": detail["score"],
                            "delta_meta": delta_meta,
                            "score_detail": detail,
                            "close_ctrl": dict(close_ctrl),
                        }

                        records.append(rec)

    ranked = sorted(records, key=lambda r: float(r["score"]))
    topk = ranked[:args.top_k]
    best = topk[0]

    T_best_delta = T_delta_local(
        best["delta_meta"]["delta_local"][0],
        best["delta_meta"]["delta_local"][1],
        best["delta_meta"]["delta_local"][2],
        best["delta_meta"]["yaw_deg"],
        best["delta_meta"]["roll_deg"],
    )

    patched_candidate, patch_info = patch_candidate(candidate, T_world_object, T_world_target_old, T_best_delta, best)

    best_candidate_path = out_dir / "best_candidate.json"
    best_config_path_out = out_dir / "best_force_proxy_config.json"
    summary_path = out_dir / "summary.json"
    top_txt_path = out_dir / "topk_summary.txt"

    save_json(best_candidate_path, patched_candidate)

    best_config_out = {
        "format": "v4_12p4h_force_closure_proxy_best_config",
        "best_record": {
            "hand_config": {
                "ctrl": dict(close_ctrl),
            },
            "force_proxy": best,
        },
        "candidate_source": candidate_source,
        "best_source": best_source,
        "candidate_ctrl": candidate_ctrl,
        "close_ctrl": close_ctrl,
        "object_geom": object_geom,
        "support_top": support_top,
        "patch_info": patch_info,
        "best_candidate": str(best_candidate_path),
    }

    save_json(best_config_path_out, best_config_out)

    viewer_script = write_best_script(args, out_dir)

    summary = {
        "format": "v4_12p4h_force_closure_proxy_hand_refine_debug",
        "model": str(model_path),
        "candidate": str(candidate_path),
        "p3_json": str(p3_path),
        "best_config_in": str(best_config_path) if best_config_path else "",
        "which": args.which,
        "object_body": args.object_body,
        "target_body": args.target_body,
        "args": vars(args),
        "candidate_source": candidate_source,
        "best_source": best_source,
        "candidate_ctrl": candidate_ctrl,
        "close_ctrl": close_ctrl,
        "object_geom": object_geom,
        "support_top": support_top,
        "support_names": support_names,
        "hand_meta": hand_meta,
        "num_records": len(records),
        "best_candidate": str(best_candidate_path),
        "best_force_proxy_config": str(best_config_path_out),
        "viewer_script": str(viewer_script),
        "topk": topk,
    }

    save_json(summary_path, summary)

    with open(top_txt_path, "w") as f:
        f.write("rank,score,delta_local,yaw,roll,finger,line,dot,z_diff,alpha,thumb_side,finger_side,support_clear,reasons\n")
        for i, r in enumerate(topk, 1):
            d = r["delta_meta"]
            sd = r["score_detail"]
            bp = sd["best_pair"]
            lm = bp["line_metrics"]
            f.write(
                f"{i},"
                f"{r['score']:.6f},"
                f"{to_jsonable(d['delta_local'])},"
                f"{d['yaw_deg']},"
                f"{d['roll_deg']},"
                f"{bp['finger_group']},"
                f"{lm['line_dist_xy']:.6f},"
                f"{lm['opposition_dot']:.6f},"
                f"{lm['z_diff']:.6f},"
                f"{lm['alpha_on_segment']:.6f},"
                f"{sd['thumb']['side_error']:.6f},"
                f"{bp['finger']['side_error']:.6f},"
                f"{sd['support_clearance']:.6f},"
                f"{sd['reasons']}\n"
            )

    print("\n========== P4H SUMMARY ==========")
    print("summary      :", summary_path)
    print("top_txt      :", top_txt_path)
    print("best_candidate:", best_candidate_path)
    print("best_config  :", best_config_path_out)
    print("viewer_script:", viewer_script)

    for i, r in enumerate(topk[:10], 1):
        d = r["delta_meta"]
        sd = r["score_detail"]
        bp = sd["best_pair"]
        lm = bp["line_metrics"]
        print(
            f"{i:02d}. score={r['score']:.6f} "
            f"delta={to_jsonable(d['delta_local'])} "
            f"yaw={d['yaw_deg']:+.1f} roll={d['roll_deg']:+.1f} "
            f"finger={bp['finger_group']} "
            f"line={lm['line_dist_xy']:.4f} "
            f"dot={lm['opposition_dot']:.3f} "
            f"z_diff={lm['z_diff']:.4f} "
            f"alpha={lm['alpha_on_segment']:.3f} "
            f"thumb_side={sd['thumb']['side_error']:.4f} "
            f"finger_side={bp['finger']['side_error']:.4f} "
            f"support={sd['support_clearance']:.4f} "
            f"reasons={sd['reasons']}"
        )

    print("\nBest viewer command:")
    print(f"bash {viewer_script}")
    print("=================================\n")


if __name__ == "__main__":
    main()
