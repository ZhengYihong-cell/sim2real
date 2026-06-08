#!/usr/bin/env python3
"""
V4.20b lightweight prior selector.

目标：
    Top-K 高召回，不是 Top-1 必然成功。
    只负责从数据集先验中选出“值得进入 IK/path/short-dynamic validation”的候选。

修复点：
    1. support_center_xy 支持 2/3 维输入；
    2. 物体旋转后按旋转 mesh 最低点放置；
    3. 加入 approach_risk；
    4. 加入 object AABB 深穿透惩罚；
    5. 接触距离阈值按物体尺寸自适应；
    6. end_grasp 用半长归一化；
    7. 输出拆分评分项，方便后续调参。
"""

from pathlib import Path
import argparse
import csv
import json
import math
import numpy as np


PROJECT = Path.home() / "Projects/o7_mujoco_sim"

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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False))


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
    return v * 0.0 if n < eps else v / n


def robust_rot6d_to_R(r6):
    r6 = np.asarray(r6, dtype=float).reshape(6)
    x_raw, y_raw = r6[:3], r6[3:6]
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
    return np.array([
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


def make_hand_proxy_points(T_object_hand, ctrl):
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


def grouped_points(proxy):
    groups = {g: [] for g in ["thumb", "index", "middle", "ring", "pinky", "palm"]}
    for name, p in proxy.items():
        for g in groups:
            if name.startswith(g):
                groups[g].append(p)
                break
        else:
            groups["palm"].append(p)
    return groups


def group_object_distances(proxy_groups, bbox_min, bbox_max):
    out = {}
    for g, pts in proxy_groups.items():
        if pts:
            out[g] = float(min(point_to_aabb_distance(p, bbox_min, bbox_max) for p in pts))
    return out


def support_clearance_world(T_world_object, proxy_points_obj, support_center_xy, support_half_size, support_top_z):
    hx, hy, _ = support_half_size
    names = list(proxy_points_obj.keys())
    pts_w = transform_points(T_world_object, np.asarray(list(proxy_points_obj.values()), dtype=float))

    vals = []
    detail = []
    for name, pw in zip(names, pts_w):
        x, y, z = pw
        inside_xy = (
            abs(x - support_center_xy[0]) <= hx + 0.025 and
            abs(y - support_center_xy[1]) <= hy + 0.025
        )
        clearance = float(z - support_top_z)
        if inside_xy:
            vals.append(clearance)
        detail.append({
            "name": name,
            "world": pw.tolist(),
            "inside_or_near_support_xy": bool(inside_xy),
            "clearance_to_support_top": clearance,
        })
    return (float(min(vals)) if vals else 999.0), detail


def approach_risk_world(T_world_hand, support_normal=np.array([0.0, 0.0, 1.0])):
    R = T_world_hand[:3, :3]
    palm_out = normalize(R[:, 0])
    approach_dir = -palm_out
    dot_up = float(np.dot(approach_dir, support_normal))

    risk = 0.0
    reasons = []
    if dot_up < -0.35:
        risk += 80.0
        reasons.append(f"approach_from_below_or_downward dot_up={dot_up:.3f}")
    elif dot_up < -0.05:
        risk += 25.0
        reasons.append(f"approach_low_margin dot_up={dot_up:.3f}")
    else:
        reasons.append(f"approach_ok dot_up={dot_up:.3f}")
    return risk, reasons, dot_up


def object_proxy_penetration_penalty(group_dists, pen_tol=-0.006):
    penalty = 0.0
    reasons = []
    for g, d in group_dists.items():
        if d < pen_tol:
            p = min(120.0, 2000.0 * abs(float(d) - pen_tol))
            penalty += p
            reasons.append(f"{g}_deep_inside_object_aabb d={d:.4f} penalty={p:.1f}")
    return penalty, reasons


def classify_grasp(T_world_hand, T_world_object, object_world_bbox_min, object_world_bbox_max,
                   support_top_z, support_min_clearance, bbox_obj_min, bbox_obj_max):
    hp = T_world_hand[:3, 3]
    op = T_world_object[:3, 3]
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
        reasons.append(f"proxy_points_below_support={support_min_clearance:.5f}")
        return "under_or_low_side_grasp", reasons

    if hp[2] < support_top_z + 0.025:
        reasons.append(f"handbase_too_low_to_support={hp[2] - support_top_z:.5f}")
        return "under_or_low_side_grasp", reasons

    if above_ratio > 0.75:
        reasons.append(f"hand_above_object above_ratio={above_ratio:.3f}")
        return "top_grasp", reasons

    if endness > 0.65:
        reasons.append(f"near_object_end endness={endness:.3f}")
        return "end_grasp", reasons

    if horiz_norm > 0.025 and abs(above_ratio) < 1.4:
        reasons.append(f"horizontal_relation horiz={horiz_norm:.3f} above_ratio={above_ratio:.3f}")
        return "side_grasp", reasons

    reasons.append("fallback_ambiguous")
    return "ambiguous_grasp", reasons


def score_candidate(grasp_type, group_dists, support_min_clearance, type_reasons,
                    object_world_size, T_world_hand, T_world_object):
    obj_diag = max(float(np.linalg.norm(object_world_size)), 1e-6)
    thumb_th = float(np.clip(0.08 * obj_diag, 0.010, 0.025))
    non_th = float(np.clip(0.07 * obj_diag, 0.008, 0.022))

    thumb_d = float(group_dists.get("thumb", 999.0))
    non_d = {g: float(group_dists.get(g, 999.0)) for g in NON_THUMB}
    best_non = min(non_d.values()) if non_d else 999.0
    near_thumb = thumb_d <= thumb_th
    near_non = [g for g, d in non_d.items() if d <= non_th]

    contact_score = 0.0
    contact_reasons = []
    if near_thumb:
        contact_score += 35.0
        contact_reasons.append(f"thumb_near_object d={thumb_d:.4f} th={thumb_th:.4f}")
    else:
        contact_score -= min(60.0, 800.0 * max(0.0, thumb_d - thumb_th))
        contact_reasons.append(f"thumb_far d={thumb_d:.4f} th={thumb_th:.4f}")

    if near_non:
        contact_score += 25.0 + 8.0 * len(near_non)
        contact_reasons.append(f"nonthumb_near={near_non} best={best_non:.4f} th={non_th:.4f}")
    else:
        contact_score -= min(70.0, 900.0 * max(0.0, best_non - non_th))
        contact_reasons.append(f"nonthumb_far best={best_non:.4f} th={non_th:.4f}")

    if near_thumb and near_non:
        contact_score += 60.0
        contact_reasons.append("thumb_plus_nonthumb_proxy_ready")

    support_risk = 0.0
    support_reasons = []
    if support_min_clearance < -0.004:
        support_risk += 300.0
        support_reasons.append(f"hard_support_penetration={support_min_clearance:.5f}")
    elif support_min_clearance < 0.0:
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
        "side_grasp": 25.0,
        "top_grasp": 20.0,
        "end_grasp": 18.0,
        "ambiguous_grasp": -10.0,
        "under_or_low_side_grasp": -260.0,
    }.get(grasp_type, -20.0)

    approach_risk, approach_reasons, approach_dot_up = approach_risk_world(T_world_hand)
    object_penalty, pen_reasons = object_proxy_penetration_penalty(group_dists)

    hp = T_world_hand[:3, 3]
    op = T_world_object[:3, 3]
    hand_center_dist = float(np.linalg.norm(hp - op))
    distance_bad = 0.0
    if hand_center_dist > 3.0 * obj_diag:
        distance_bad = 40.0
    elif hand_center_dist > 2.3 * obj_diag:
        distance_bad = 15.0

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

    rank_reason_short = "; ".join(reasons[:4])

    return {
        "final_score": float(final_score),
        "contact_score": float(contact_score),
        "support_risk": float(support_risk),
        "approach_risk": float(approach_risk),
        "approach_dot_up": float(approach_dot_up),
        "object_penetration_penalty": float(object_penalty),
        "distance_bad": float(distance_bad),
        "type_score": float(type_score),
        "thumb_near_threshold": thumb_th,
        "nonthumb_near_threshold": non_th,
        "near_thumb": bool(near_thumb),
        "near_nonthumb": near_non,
        "proxy_ready": bool(near_thumb and len(near_non) > 0),
        "rank_reason_short": rank_reason_short,
        "score_reasons": reasons,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object-code", required=True)
    ap.add_argument("--npy", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-k", type=int, default=20)
    ap.add_argument("--support-center-xy", default="0.455 0.0")
    ap.add_argument("--support-half-size", default="0.045 0.045 0.115")
    ap.add_argument("--support-top-z", type=float, default=0.23)
    ap.add_argument("--object-clearance", type=float, default=0.003)
    ap.add_argument("--object-pos", default="")
    ap.add_argument("--object-quat-wxyz", default="1 0 0 0")
    ap.add_argument("--max-per-type", type=int, default=6)
    ap.add_argument("--support-json", default="", help="reserved for future multi-support scene description")
    args = ap.parse_args()

    npy_path = resolve(args.npy)
    mesh_path = resolve(args.mesh)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    support_center_xy = parse_vec2(args.support_center_xy)
    support_half_size = parse_vec3(args.support_half_size)
    support_top_z = float(args.support_top_z)

    verts = read_obj_vertices(mesh_path)
    samples = load_samples(npy_path)
    if not samples:
        raise RuntimeError(f"no valid samples in {npy_path}")

    rows = []

    for local_i, sample in samples:
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

        proxy = make_hand_proxy_points(T_object_hand, ctrl)
        groups = grouped_points(proxy)
        group_dists = group_object_distances(groups, bbox_min, bbox_max)

        support_min_clearance, support_detail = support_clearance_world(
            T_world_object, proxy, support_center_xy, support_half_size, support_top_z
        )

        object_world_verts = transform_points(T_world_object, verts_scaled)
        ow_min = object_world_verts.min(axis=0)
        ow_max = object_world_verts.max(axis=0)
        ow_size = ow_max - ow_min

        grasp_type, type_reasons = classify_grasp(
            T_world_hand=T_world_hand,
            T_world_object=T_world_object,
            object_world_bbox_min=ow_min,
            object_world_bbox_max=ow_max,
            support_top_z=support_top_z,
            support_min_clearance=support_min_clearance,
            bbox_obj_min=bbox_min,
            bbox_obj_max=bbox_max,
        )

        score_info = score_candidate(
            grasp_type=grasp_type,
            group_dists=group_dists,
            support_min_clearance=support_min_clearance,
            type_reasons=type_reasons,
            object_world_size=ow_size,
            T_world_hand=T_world_hand,
            T_world_object=T_world_object,
        )

        row = {
            "object_code": args.object_code,
            "valid_local_index": int(local_i),
            "raw_sample_index": sample_raw_index(local_i, sample),
            "scale": float(scale),
            "grasp_type": grasp_type,
            "score": score_info["final_score"],
            **score_info,
            "thumb_bbox_dist": float(group_dists.get("thumb", 999.0)),
            "best_nonthumb_bbox_dist": float(min(group_dists.get(g, 999.0) for g in NON_THUMB)),
            "group_object_distances": group_dists,
            "support_min_clearance": float(support_min_clearance),
            "T_object_hand_pos": T_object_hand[:3, 3].tolist(),
            "T_world_hand_pos": T_world_hand[:3, 3].tolist(),
            "T_world_object_pos": T_world_object[:3, 3].tolist(),
            "object_bbox_size_scaled": bbox_size.tolist(),
            "object_world_bbox_min": ow_min.tolist(),
            "object_world_bbox_max": ow_max.tolist(),
            "ctrl": ctrl,
            "type_reasons": type_reasons,
        }
        rows.append(row)

    rows_sorted = sorted(rows, key=lambda r: r["score"], reverse=True)

    selected = []
    per_type = {}
    for r in rows_sorted:
        gt = r["grasp_type"]
        if per_type.get(gt, 0) >= args.max_per_type:
            continue
        selected.append(r)
        per_type[gt] = per_type.get(gt, 0) + 1
        if len(selected) >= args.top_k:
            break

    if len(selected) < args.top_k:
        existing = {r["valid_local_index"] for r in selected}
        for r in rows_sorted:
            if r["valid_local_index"] in existing:
                continue
            selected.append(r)
            existing.add(r["valid_local_index"])
            if len(selected) >= args.top_k:
                break

    save_json(out_dir / "ranked_candidates.json", rows_sorted)
    save_json(out_dir / "selected_topk_compact.json", selected)

    with (out_dir / "ranked_candidates.csv").open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "local", "raw", "type", "score",
            "contact_score", "support_risk", "approach_risk",
            "object_penetration_penalty", "distance_bad", "type_score",
            "proxy_ready", "thumb_d", "non_d", "support_clearance",
            "rank_reason_short",
        ])
        for rank, r in enumerate(rows_sorted, start=1):
            w.writerow([
                rank, r["valid_local_index"], r["raw_sample_index"], r["grasp_type"],
                f"{r['score']:.6f}",
                f"{r['contact_score']:.6f}",
                f"{r['support_risk']:.6f}",
                f"{r['approach_risk']:.6f}",
                f"{r['object_penetration_penalty']:.6f}",
                f"{r['distance_bad']:.6f}",
                f"{r['type_score']:.6f}",
                r["proxy_ready"],
                f"{r['thumb_bbox_dist']:.6f}",
                f"{r['best_nonthumb_bbox_dist']:.6f}",
                f"{r['support_min_clearance']:.6f}",
                r["rank_reason_short"],
            ])

    (out_dir / "selected_valid_local_indices.txt").write_text(
        "\n".join(str(r["valid_local_index"]) for r in selected) + "\n"
    )

    lines = []
    lines.append("========== V4.20b GENERAL GRASP TYPE SELECTOR ==========")
    lines.append("定位：lightweight prior selector；目标是 Top-K 高召回，不是最终执行器。")
    lines.append(f"object_code : {args.object_code}")
    lines.append(f"npy         : {rel(npy_path)}")
    lines.append(f"mesh        : {rel(mesh_path)}")
    lines.append(f"samples     : {len(rows)}")
    lines.append(f"top_k       : {args.top_k}")
    lines.append("")
    lines.append("---- type counts ----")
    type_counts = {}
    for r in rows_sorted:
        type_counts[r["grasp_type"]] = type_counts.get(r["grasp_type"], 0) + 1
    for k, v in sorted(type_counts.items()):
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("---- selected top-k ----")
    for rank, r in enumerate(selected, start=1):
        lines.append(
            f"rank={rank:02d} local={r['valid_local_index']:03d} raw={r['raw_sample_index']:03d} "
            f"type={r['grasp_type']} score={r['score']:.2f} ready={r['proxy_ready']} "
            f"C={r['contact_score']:.1f} S={r['support_risk']:.1f} "
            f"A={r['approach_risk']:.1f} Pen={r['object_penetration_penalty']:.1f} "
            f"D={r['distance_bad']:.1f} T={r['type_score']:.1f} "
            f"thumb_d={r['thumb_bbox_dist']:.4f} non_d={r['best_nonthumb_bbox_dist']:.4f} "
            f"support={r['support_min_clearance']:.4f}"
        )
        lines.append(f"  - {r['rank_reason_short']}")
    lines.append("")
    lines.append("---- output ----")
    lines.append(f"ranked json : {rel(out_dir / 'ranked_candidates.json')}")
    lines.append(f"ranked csv  : {rel(out_dir / 'ranked_candidates.csv')}")
    lines.append(f"topk json   : {rel(out_dir / 'selected_topk_compact.json')}")
    lines.append("=========================================================")

    report = "\n".join(lines) + "\n"
    (out_dir / "v420b_selector_report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
