#!/usr/bin/env python3
"""
脚本类型：
    debug / v4.20 / general-grasp-selector / grasp-type-classifier

用途：
    面向“泛化抓取”的通用抓握选择器第一版。
    输入一个数据集物体的所有先验 sample，自动完成：
        1. 数据集 hand_pose 解析；
        2. T_object_hand -> T_world_hand；
        3. 抓型分类：top / side / end / under_or_low_side / ambiguous；
        4. 支撑环境风险估计；
        5. thumb + 非拇指接触潜力估计；
        6. Top-K 排序输出。

    重点：
        不救某一个 sample；
        不做 dz/radial 手工微调；
        不跑动态执行；
        只做“从所有先验中选择哪些值得进入 IK/动态验证”。

输入：
    --object-code
    --npy
    --mesh
    --out-dir

输出：
    out_dir/
        ranked_candidates.json
        ranked_candidates.csv
        selected_topk_compact.json
        selected_valid_local_indices.txt
        v420_selector_report.txt

当前流程位置：
    数据集先验
        -> V4.20 通用抓型分类 + 候选排序
        -> 后续 site IK / path / short dynamic validation

不负责：
    1. 不执行抓取；
    2. 不做完整 IK 路径；
    3. 不修改 scene；
    4. 不做单个 sample 的局部姿态修正。
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


def parse_vec3(s):
    v = [float(x) for x in str(s).replace(",", " ").split()]
    if len(v) != 3:
        raise RuntimeError(f"expected 3 values, got {s}")
    return np.asarray(v, dtype=float)


def parse_quat_wxyz(s):
    v = [float(x) for x in str(s).replace(",", " ").split()]
    if len(v) != 4:
        raise RuntimeError(f"expected 4 values, got {s}")
    q = np.asarray(v, dtype=float)
    q = q / max(np.linalg.norm(q), 1e-12)
    return q


def quat_wxyz_to_R(q):
    w, x, y, z = q
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,       2*x*z + 2*y*w],
        [2*x*y + 2*z*w,         1 - 2*x*x - 2*z*z,   2*y*z - 2*x*w],
        [2*x*z - 2*y*w,         2*y*z + 2*x*w,       1 - 2*x*x - 2*y*y],
    ], dtype=float)


def normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
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
        raise RuntimeError(f"no vertices read from obj: {path}")
    return np.asarray(verts, dtype=float)


def load_sample_array(npy_path):
    arr = np.load(npy_path, allow_pickle=True)
    samples = []
    for i in range(len(arr)):
        s = arr[i].item() if hasattr(arr[i], "item") else arr[i]
        if not isinstance(s, dict):
            continue
        if "hand_pose" not in s:
            continue
        samples.append((i, s))
    return samples


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

    p = hp[:3]
    R = robust_rot6d_to_R(hp[3:9])
    return T_from_Rp(R, p)


def extract_ctrl(sample):
    hp = np.asarray(sample["hand_pose"], dtype=float)
    if hp.shape[0] >= 16:
        return {j: float(v) for j, v in zip(O7_ACTIVE_JOINTS, hp[9:16])}
    return {j: 0.0 for j in O7_ACTIVE_JOINTS}


def point_to_aabb_distance(p, bmin, bmax):
    """
    AABB signed-ish distance:
      >0 outside distance
      <=0 inside, negative nearest exit distance
    """
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
    """
    用简化手关键点代理做选择器评分。
    不是精确碰撞，只用于 Top-K 排序。

    O7 数据集坐标约定：
      Z: 手指方向
      X: 掌心外法向
      Y: 四指 -> 拇指方向
    """
    R = T_object_hand[:3, :3]
    p = T_object_hand[:3, 3]

    X = R[:, 0]
    Y = R[:, 1]
    Z = R[:, 2]

    pts = {}

    pts["hand_base"] = p
    pts["palm_center"] = p + 0.030 * Z - 0.018 * X
    pts["palm_low"] = p + 0.020 * Z - 0.055 * X

    y_offsets = {
        "index": -0.030,
        "middle": -0.010,
        "ring": 0.010,
        "pinky": 0.030,
    }

    for g, yoff in y_offsets.items():
        q = float(ctrl.get(f"{g}_mcp_pitch", 0.0))
        bend = np.clip(q / 1.2, 0.0, 1.0)

        base = p + 0.045 * Z + yoff * Y - 0.020 * X
        mid = p + (0.075 - 0.015 * bend) * Z + yoff * Y - (0.025 + 0.025 * bend) * X
        tip = p + (0.110 - 0.035 * bend) * Z + yoff * Y - (0.030 + 0.060 * bend) * X

        pts[f"{g}_base"] = base
        pts[f"{g}_mid"] = mid
        pts[f"{g}_tip"] = tip

    thumb_pitch = float(ctrl.get("thumb_cmc_pitch", 0.0))
    thumb_bend = np.clip((thumb_pitch + 0.2) / 1.3, 0.0, 1.0)

    pts["thumb_base"] = p + 0.025 * Z + 0.055 * Y - 0.020 * X
    pts["thumb_mid"] = p + (0.050 - 0.010 * thumb_bend) * Z + 0.075 * Y - (0.030 + 0.025 * thumb_bend) * X
    pts["thumb_tip"] = p + (0.070 - 0.020 * thumb_bend) * Z + 0.090 * Y - (0.040 + 0.055 * thumb_bend) * X

    return pts


def grouped_points(proxy):
    groups = {g: [] for g in ["thumb", "index", "middle", "ring", "pinky", "palm"]}
    for name, p in proxy.items():
        if name.startswith("thumb"):
            groups["thumb"].append(p)
        elif name.startswith("index"):
            groups["index"].append(p)
        elif name.startswith("middle"):
            groups["middle"].append(p)
        elif name.startswith("ring"):
            groups["ring"].append(p)
        elif name.startswith("pinky"):
            groups["pinky"].append(p)
        else:
            groups["palm"].append(p)
    return groups


def group_object_distances(proxy_groups, bbox_min, bbox_max):
    out = {}
    for g, pts in proxy_groups.items():
        if not pts:
            continue
        ds = [point_to_aabb_distance(p, bbox_min, bbox_max) for p in pts]
        out[g] = float(min(ds))
    return out


def support_clearance_world(T_world_object, proxy_points_obj, support_center_xy, support_half_size, support_top_z):
    support_center_xy = np.asarray(support_center_xy, dtype=float)
    hx, hy, hz = support_half_size

    pts_obj = np.asarray(list(proxy_points_obj.values()), dtype=float)
    pts_w = transform_points(T_world_object, pts_obj)

    values = []
    detail = []

    for name, pw in zip(proxy_points_obj.keys(), pts_w):
        x, y, z = pw
        inside_xy = (
            abs(x - support_center_xy[0]) <= hx + 0.025
            and abs(y - support_center_xy[1]) <= hy + 0.025
        )

        clearance = float(z - support_top_z)

        if inside_xy:
            values.append(clearance)

        detail.append({
            "name": name,
            "world": pw.tolist(),
            "inside_or_near_support_xy": bool(inside_xy),
            "clearance_to_support_top": clearance,
        })

    if values:
        min_clear = float(min(values))
    else:
        min_clear = 999.0

    return min_clear, detail


def classify_grasp(T_world_hand, T_world_object, object_world_bbox_min, object_world_bbox_max,
                   support_top_z, support_min_clearance, bbox_obj_min, bbox_obj_max):
    hp = T_world_hand[:3, 3]
    op = T_world_object[:3, 3]
    Rw = T_world_hand[:3, :3]

    finger_axis = normalize(Rw[:, 2])
    palm_axis = normalize(Rw[:, 0])

    obj_center = 0.5 * (object_world_bbox_min + object_world_bbox_max)
    obj_size = object_world_bbox_max - object_world_bbox_min
    obj_height = max(float(obj_size[2]), 1e-6)

    rel = hp - obj_center
    rel_z = float(rel[2])
    horiz = np.array([rel[0], rel[1], 0.0], dtype=float)
    horiz_norm = float(np.linalg.norm(horiz))

    # 物体主轴，先用 object-frame bbox 最长轴，然后转到 world
    obj_size_obj = bbox_obj_max - bbox_obj_min
    major_idx = int(np.argmax(obj_size_obj))
    major_axis_obj = np.zeros(3)
    major_axis_obj[major_idx] = 1.0
    major_axis_w = normalize(T_world_object[:3, :3] @ major_axis_obj)

    endness = abs(float(np.dot(rel, major_axis_w))) / max(float(obj_size_obj[major_idx]), 1e-6)
    above_ratio = rel_z / obj_height

    to_object = normalize(obj_center - hp)
    finger_to_object_align = float(np.dot(finger_axis, to_object))
    palm_to_object_align = float(np.dot(-palm_axis, to_object))

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

    if endness > 0.42:
        reasons.append(f"near_object_end endness={endness:.3f}")
        return "end_grasp", reasons

    if horiz_norm > 0.025 and abs(above_ratio) < 1.4:
        reasons.append(f"horizontal_relation horiz={horiz_norm:.3f} above_ratio={above_ratio:.3f}")
        return "side_grasp", reasons

    reasons.append("fallback_ambiguous")
    return "ambiguous_grasp", reasons


def score_candidate(grasp_type, group_dists, support_min_clearance, type_reasons,
                    object_world_size, T_world_hand, T_world_object):
    score = 0.0
    reasons = []

    thumb_d = group_dists.get("thumb", 999.0)
    non_d = {g: group_dists.get(g, 999.0) for g in NON_THUMB}
    best_non = min(non_d.values()) if non_d else 999.0
    near_non = [g for g, d in non_d.items() if d <= 0.018]
    near_thumb = thumb_d <= 0.020

    # 接触潜力
    if near_thumb:
        score += 35.0
        reasons.append(f"thumb_near_object d={thumb_d:.4f}")
    else:
        score -= min(60.0, 800.0 * max(0.0, thumb_d - 0.020))
        reasons.append(f"thumb_far d={thumb_d:.4f}")

    if near_non:
        score += 25.0 + 8.0 * len(near_non)
        reasons.append(f"nonthumb_near={near_non} best={best_non:.4f}")
    else:
        score -= min(70.0, 900.0 * max(0.0, best_non - 0.018))
        reasons.append(f"nonthumb_far best={best_non:.4f}")

    if near_thumb and near_non:
        score += 60.0
        reasons.append("thumb_plus_nonthumb_proxy_ready")

    # 支撑风险
    if support_min_clearance < -0.004:
        score -= 300.0
        reasons.append(f"hard_support_penetration={support_min_clearance:.5f}")
    elif support_min_clearance < 0.0:
        score -= 120.0
        reasons.append(f"soft_support_penetration={support_min_clearance:.5f}")
    elif support_min_clearance < 0.004:
        score -= 35.0
        reasons.append(f"near_support={support_min_clearance:.5f}")
    elif support_min_clearance < 0.010:
        score -= 10.0
        reasons.append(f"support_margin_small={support_min_clearance:.5f}")
    else:
        score += 10.0
        reasons.append(f"support_clear={support_min_clearance:.5f}")

    # 抓型偏好：不是写死，而是桌面环境下的先验偏好
    if grasp_type == "under_or_low_side_grasp":
        score -= 260.0
        reasons.append("type_penalty_under_or_low_side")
    elif grasp_type == "top_grasp":
        score += 20.0
        reasons.append("type_bonus_top")
    elif grasp_type == "side_grasp":
        score += 25.0
        reasons.append("type_bonus_side")
    elif grasp_type == "end_grasp":
        score += 18.0
        reasons.append("type_bonus_end")
    else:
        score -= 10.0
        reasons.append("type_ambiguous")

    # 过远惩罚：handbase 离 object center 过远，往往是看起来悬空/抓不到
    hp = T_world_hand[:3, 3]
    op = T_world_object[:3, 3]
    obj_diag = max(float(np.linalg.norm(object_world_size)), 1e-6)
    hand_center_dist = float(np.linalg.norm(hp - op))
    if hand_center_dist > 3.0 * obj_diag:
        score -= 40.0
        reasons.append(f"handbase_far_from_object_center={hand_center_dist:.4f}")

    # 合并分类理由
    reasons.extend(type_reasons)

    return score, reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object-code", required=True)
    ap.add_argument("--npy", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-k", type=int, default=20)

    # 当前场景默认与之前桌面/垫块一致；后续可以从 scene 自动读取
    ap.add_argument("--support-center-xy", default="0.455 0.0")
    ap.add_argument("--support-half-size", default="0.045 0.045 0.115")
    ap.add_argument("--support-top-z", type=float, default=0.23)
    ap.add_argument("--object-clearance", type=float, default=0.003)

    # 当前物体位姿；默认由 mesh bottom 放在支撑面上
    ap.add_argument("--object-pos", default="")
    ap.add_argument("--object-quat-wxyz", default="1 0 0 0")

    ap.add_argument("--max-per-type", type=int, default=6)
    args = ap.parse_args()

    npy_path = resolve(args.npy)
    mesh_path = resolve(args.mesh)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    support_center_xy = parse_vec3(args.support_center_xy)[:2]
    support_half_size = parse_vec3(args.support_half_size)
    support_top_z = float(args.support_top_z)

    verts = read_obj_vertices(mesh_path)
    samples = load_sample_array(npy_path)

    if not samples:
        raise RuntimeError(f"no valid samples in {npy_path}")

    rows = []

    for local_i, sample in samples:
        scale = sample_scale(sample)
        verts_scaled = verts * scale
        bbox_min = verts_scaled.min(axis=0)
        bbox_max = verts_scaled.max(axis=0)
        bbox_size = bbox_max - bbox_min

        if args.object_pos.strip():
            object_pos = parse_vec3(args.object_pos)
        else:
            object_pos = np.array([
                support_center_xy[0],
                support_center_xy[1],
                support_top_z + args.object_clearance - float(bbox_min[2])
            ], dtype=float)

        R_obj = quat_wxyz_to_R(parse_quat_wxyz(args.object_quat_wxyz))
        T_world_object = T_from_Rp(R_obj, object_pos)

        T_object_hand = extract_T_object_hand(sample)
        T_world_hand = T_world_object @ T_object_hand
        ctrl = extract_ctrl(sample)

        proxy = make_hand_proxy_points(T_object_hand, ctrl)
        groups = grouped_points(proxy)

        group_dists = group_object_distances(groups, bbox_min, bbox_max)

        support_min_clearance, support_detail = support_clearance_world(
            T_world_object,
            proxy,
            support_center_xy=support_center_xy,
            support_half_size=support_half_size,
            support_top_z=support_top_z,
        )

        object_world_verts = transform_points(T_world_object, verts_scaled)
        object_world_bbox_min = object_world_verts.min(axis=0)
        object_world_bbox_max = object_world_verts.max(axis=0)
        object_world_size = object_world_bbox_max - object_world_bbox_min

        grasp_type, type_reasons = classify_grasp(
            T_world_hand=T_world_hand,
            T_world_object=T_world_object,
            object_world_bbox_min=object_world_bbox_min,
            object_world_bbox_max=object_world_bbox_max,
            support_top_z=support_top_z,
            support_min_clearance=support_min_clearance,
            bbox_obj_min=bbox_min,
            bbox_obj_max=bbox_max,
        )

        score, score_reasons = score_candidate(
            grasp_type=grasp_type,
            group_dists=group_dists,
            support_min_clearance=support_min_clearance,
            type_reasons=type_reasons,
            object_world_size=object_world_size,
            T_world_hand=T_world_hand,
            T_world_object=T_world_object,
        )

        near_thumb = group_dists.get("thumb", 999.0) <= 0.020
        near_non = [g for g in NON_THUMB if group_dists.get(g, 999.0) <= 0.018]
        proxy_ready = bool(near_thumb and len(near_non) > 0)

        row = {
            "object_code": args.object_code,
            "valid_local_index": int(local_i),
            "raw_sample_index": sample_raw_index(local_i, sample),
            "scale": float(scale),
            "grasp_type": grasp_type,
            "score": float(score),
            "proxy_ready": proxy_ready,
            "near_thumb": bool(near_thumb),
            "near_nonthumb": near_non,
            "thumb_bbox_dist": float(group_dists.get("thumb", 999.0)),
            "best_nonthumb_bbox_dist": float(min([group_dists.get(g, 999.0) for g in NON_THUMB])),
            "group_object_distances": group_dists,
            "support_min_clearance": float(support_min_clearance),
            "T_object_hand_pos": T_object_hand[:3, 3].tolist(),
            "T_world_hand_pos": T_world_hand[:3, 3].tolist(),
            "T_world_object_pos": T_world_object[:3, 3].tolist(),
            "object_bbox_size_scaled": bbox_size.tolist(),
            "object_world_bbox_min": object_world_bbox_min.tolist(),
            "object_world_bbox_max": object_world_bbox_max.tolist(),
            "ctrl": ctrl,
            "type_reasons": type_reasons,
            "score_reasons": score_reasons,
        }

        rows.append(row)

    rows_sorted = sorted(rows, key=lambda r: r["score"], reverse=True)

    # Top-K 保留抓型多样性，避免一种类型占满
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

    # 保存 JSON
    save_json(out_dir / "ranked_candidates.json", rows_sorted)
    save_json(out_dir / "selected_topk_compact.json", selected)

    # 保存 CSV
    csv_path = out_dir / "ranked_candidates.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank",
            "valid_local_index",
            "raw_sample_index",
            "grasp_type",
            "score",
            "proxy_ready",
            "near_thumb",
            "near_nonthumb",
            "thumb_bbox_dist",
            "best_nonthumb_bbox_dist",
            "support_min_clearance",
            "T_world_hand_pos",
            "score_reasons",
        ])

        for rank, r in enumerate(rows_sorted, start=1):
            w.writerow([
                rank,
                r["valid_local_index"],
                r["raw_sample_index"],
                r["grasp_type"],
                f"{r['score']:.6f}",
                r["proxy_ready"],
                r["near_thumb"],
                "|".join(r["near_nonthumb"]),
                f"{r['thumb_bbox_dist']:.6f}",
                f"{r['best_nonthumb_bbox_dist']:.6f}",
                f"{r['support_min_clearance']:.6f}",
                json.dumps(r["T_world_hand_pos"]),
                " ; ".join(r["score_reasons"][:8]),
            ])

    (out_dir / "selected_valid_local_indices.txt").write_text(
        "\n".join(str(r["valid_local_index"]) for r in selected) + "\n"
    )
    (out_dir / "selected_raw_sample_indices.txt").write_text(
        "\n".join(str(r["raw_sample_index"]) for r in selected) + "\n"
    )

    # 报告
    lines = []
    lines.append("========== V4.20 GENERAL GRASP TYPE SELECTOR ==========")
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
            f"rank={rank:02d} "
            f"local={r['valid_local_index']:03d} raw={r['raw_sample_index']:03d} "
            f"type={r['grasp_type']} score={r['score']:.3f} "
            f"ready={r['proxy_ready']} "
            f"thumb_d={r['thumb_bbox_dist']:.4f} "
            f"non_d={r['best_nonthumb_bbox_dist']:.4f} "
            f"support={r['support_min_clearance']:.4f} "
            f"near_non={r['near_nonthumb']}"
        )
        for rr in r["score_reasons"][:5]:
            lines.append(f"  - {rr}")
        lines.append("")

    lines.append("---- output ----")
    lines.append(f"ranked json : {rel(out_dir / 'ranked_candidates.json')}")
    lines.append(f"ranked csv  : {rel(csv_path)}")
    lines.append(f"topk json   : {rel(out_dir / 'selected_topk_compact.json')}")
    lines.append("========================================================")
    report = "\n".join(lines) + "\n"

    (out_dir / "v420_selector_report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
