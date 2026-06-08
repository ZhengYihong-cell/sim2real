#!/usr/bin/env python3
"""
V4.20d real-proxy support-fix selector.

定位：
    lightweight prior selector / Top-K high recall.
    不是最终执行器，不负责 close/lift，不救单个 sample。

相对 V4.20c 的修复：
    1. real MuJoCo hand geom proxy 保留；
    2. support 风险拆成 open_support 与 close_support；
    3. support 距离使用 mujoco.mj_geomDistance；
    4. 抓型分类主要使用 open_support，不再让 close 阶段支撑接近直接决定 under；
    5. 输出 overall / diverse / feasible 三类 Top-K，全部排除 ik_failed；
    6. feasible 目标是送入 V4.21 IK/path/short-dynamic validation 的候选，不代表最终成功。
"""

from pathlib import Path
import argparse
import importlib.util
import csv
import json
import numpy as np
import mujoco


PROJECT = Path.home() / "Projects/o7_mujoco_sim"
HELPER = PROJECT / "scripts/07_precheck_ik_collision/select_v4_20c_real_hand_keypoint_proxy_debug.py"

spec = importlib.util.spec_from_file_location("v420c_helper", str(HELPER))
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)

NON_THUMB = ["index", "middle", "ring", "pinky"]


def body_world_T(model, data, body_name):
    bid = H.name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if bid < 0:
        raise RuntimeError(f"missing body: {body_name}")
    R = np.asarray(data.xmat[bid], dtype=float).reshape(3, 3)
    p = np.asarray(data.xpos[bid], dtype=float)
    return H.T_from_Rp(R, p)


def side_open_from_close(close_ctrl):
    side = dict(close_ctrl)
    for j in ["index_mcp_pitch", "middle_mcp_pitch", "ring_mcp_pitch", "pinky_mcp_pitch"]:
        side[j] = 0.0
    return side


def collect_support_geom_ids(model):
    ids = []
    for gid in range(model.ngeom):
        s = (H.geom_name(model, gid) + " " + H.geom_body_name(model, gid)).lower()

        if "grasp_object" in s:
            continue

        is_support = (
            "object_pedestal" in s
            or "pedestal" in s
            or "support" in s
            or "table" in s
            or "floor" in s
            or "plane" in s
        )

        if is_support:
            ids.append(gid)

    return ids


def geom_distance(model, data, g1, g2, distmax=0.25):
    fromto = np.zeros(6, dtype=float)
    try:
        return float(mujoco.mj_geomDistance(model, data, int(g1), int(g2), float(distmax), fromto))
    except Exception:
        return None


def support_distance_by_geom(model, data, hand_records, support_gids, fallback_table_top_z=0.0):
    """
    返回真实 hand geom 到支撑 geom 的 signed distance。
    如果模型里没有支撑 geom，则退化为 center_z - radius - table_top_z。
    """
    group_min = {}
    group_best = {}
    global_min = 999.0

    if support_gids:
        for r in hand_records:
            gid = int(r["geom_id"])
            grp = r["group"]

            for sgid in support_gids:
                d = geom_distance(model, data, gid, sgid)
                if d is None:
                    continue

                if d < global_min:
                    global_min = d

                if grp not in group_min or d < group_min[grp]:
                    group_min[grp] = float(d)
                    rr = dict(r)
                    rr["support_geom_id"] = int(sgid)
                    rr["support_geom_name"] = H.geom_name(model, sgid)
                    rr["support_body_name"] = H.geom_body_name(model, sgid)
                    rr["support_distance"] = float(d)
                    group_best[grp] = rr
    else:
        for r in hand_records:
            grp = r["group"]
            c = np.asarray(r["center_world"], dtype=float)
            d = float(c[2] - float(r["radius"]) - fallback_table_top_z)

            if d < global_min:
                global_min = d

            if grp not in group_min or d < group_min[grp]:
                group_min[grp] = d
                rr = dict(r)
                rr["support_geom_name"] = "fallback_table_plane"
                rr["support_distance"] = d
                group_best[grp] = rr

    return float(global_min), group_min, group_best


def object_proxy_penetration_penalty(group_dists, pen_tol=-0.006):
    penalty = 0.0
    reasons = []
    for g, d in group_dists.items():
        d = float(d)
        if d < pen_tol:
            p = min(120.0, 2000.0 * abs(d - pen_tol))
            penalty += p
            reasons.append(f"{g}_deep_inside_object_aabb d={d:.4f} penalty={p:.1f}")
    return float(penalty), reasons


def support_risk_from_distance(d, phase):
    """
    open 阶段严格；close 阶段相对宽松。
    因为 close 后碰垫块的手指可以在后续 contact-aware close 里冻结。
    """
    d = float(d)
    reasons = []

    if phase == "open":
        if d < -0.004:
            return 300.0, [f"open_hard_support_penetration={d:.5f}"]
        if d < -0.001:
            return 150.0, [f"open_soft_support_penetration={d:.5f}"]
        if d < 0.002:
            return 45.0, [f"open_near_support={d:.5f}"]
        if d < 0.008:
            return 12.0, [f"open_support_margin_small={d:.5f}"]
        return 0.0, [f"open_support_clear={d:.5f}"]

    if d < -0.010:
        return 180.0, [f"close_hard_support_penetration={d:.5f}"]
    if d < -0.004:
        return 70.0, [f"close_soft_support_penetration={d:.5f}"]
    if d < 0.000:
        return 25.0, [f"close_touching_support={d:.5f}"]
    if d < 0.006:
        return 8.0, [f"close_near_support={d:.5f}"]
    return 0.0, [f"close_support_clear={d:.5f}"]


def approach_risk_world(T_world_hand, T_world_object, support_top_z):
    R = T_world_hand[:3, :3]
    hp = T_world_hand[:3, 3]
    op = T_world_object[:3, 3]

    palm_out = H.normalize(R[:, 0])
    approach_dir = -palm_out
    dot_up = float(np.dot(approach_dir, np.array([0.0, 0.0, 1.0])))

    risk = 0.0
    reasons = []

    if hp[2] < support_top_z + 0.020:
        risk += 80.0
        reasons.append(f"hand_low_near_support hp_z={hp[2]:.4f}")
    elif hp[2] < op[2] - 0.010 and dot_up > 0.25:
        risk += 60.0
        reasons.append(f"approach_likely_from_below dot_up={dot_up:.3f}")
    elif abs(dot_up) < 0.10:
        risk += 5.0
        reasons.append(f"approach_nearly_horizontal dot_up={dot_up:.3f}")
    else:
        reasons.append(f"approach_record dot_up={dot_up:.3f}")

    return float(risk), reasons, dot_up


def classify_grasp_open_support(T_world_hand, T_world_object,
                                object_world_bbox_min, object_world_bbox_max,
                                support_top_z, open_support_min,
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
    major_axis_w = H.normalize(T_world_object[:3, :3] @ major_axis_obj)

    half_len = max(0.5 * float(obj_size_obj[major_idx]), 1e-6)
    endness = abs(float(np.dot(rel, major_axis_w))) / half_len
    above_ratio = rel_z / obj_height

    reasons = []

    # 只用 open 阶段支撑风险决定是否 blocked/under
    if open_support_min < -0.004:
        reasons.append(f"open_hand_below_support={open_support_min:.5f}")
        return "under_or_low_side_grasp", reasons, above_ratio, endness, horiz_norm

    if hp[2] < support_top_z + 0.020:
        reasons.append(f"handbase_too_low_to_support={hp[2] - support_top_z:.5f}")
        return "under_or_low_side_grasp", reasons, above_ratio, endness, horiz_norm

    if endness > 0.65:
        reasons.append(f"near_object_end endness={endness:.3f}")
        return "end_grasp", reasons, above_ratio, endness, horiz_norm

    # top 不能只看 z，高位侧抓不能误判成 top
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


def contact_score_from_close(close_dists, object_world_size):
    obj_diag = max(float(np.linalg.norm(object_world_size)), 1e-6)
    thumb_th = float(np.clip(0.10 * obj_diag, 0.012, 0.030))
    non_th = float(np.clip(0.09 * obj_diag, 0.010, 0.028))

    thumb_d = float(close_dists.get("thumb", 999.0))
    non_d = {g: float(close_dists.get(g, 999.0)) for g in NON_THUMB}
    best_non = min(non_d.values()) if non_d else 999.0

    near_thumb = thumb_d <= thumb_th
    near_non = [g for g, d in non_d.items() if d <= non_th]

    score = 0.0
    reasons = []

    if near_thumb:
        score += 45.0
        reasons.append(f"thumb_near_object d={thumb_d:.4f} th={thumb_th:.4f}")
    else:
        score -= min(70.0, 900.0 * max(0.0, thumb_d - thumb_th))
        reasons.append(f"thumb_far d={thumb_d:.4f} th={thumb_th:.4f}")

    if near_non:
        score += 35.0 + 10.0 * len(near_non)
        reasons.append(f"nonthumb_near={near_non} best={best_non:.4f} th={non_th:.4f}")
    else:
        score -= min(80.0, 1000.0 * max(0.0, best_non - non_th))
        reasons.append(f"nonthumb_far best={best_non:.4f} th={non_th:.4f}")

    if near_thumb and near_non:
        score += 90.0
        reasons.append("thumb_plus_nonthumb_close_proxy_ready")

    return {
        "contact_score": float(score),
        "near_thumb": bool(near_thumb),
        "near_nonthumb": near_non,
        "proxy_ready_close": bool(near_thumb and len(near_non) > 0),
        "thumb_near_threshold": thumb_th,
        "nonthumb_near_threshold": non_th,
        "thumb_surface_dist": thumb_d,
        "best_nonthumb_surface_dist": best_non,
        "contact_reasons": reasons,
    }


def type_score_value(grasp_type):
    return {
        "side_grasp": 28.0,
        "top_grasp": 22.0,
        "end_grasp": 18.0,
        "ambiguous_grasp": -8.0,
        "under_or_low_side_grasp": -260.0,
    }.get(grasp_type, -20.0)


def select_overall(rows, k):
    return [r for r in rows if r.get("ik_success", False)][:k]


def select_diverse(rows, top_k, max_per_type):
    selected = []
    per_type = {}
    for r in rows:
        if not r.get("ik_success", False):
            continue
        gt = r["grasp_type"]
        if per_type.get(gt, 0) >= max_per_type:
            continue
        selected.append(r)
        per_type[gt] = per_type.get(gt, 0) + 1
        if len(selected) >= top_k:
            return selected

    existing = {r["valid_local_index"] for r in selected}
    for r in rows:
        if not r.get("ik_success", False):
            continue
        if r["valid_local_index"] in existing:
            continue
        selected.append(r)
        if len(selected) >= top_k:
            break
    return selected


def select_feasible(rows, top_k):
    selected = []
    for r in rows:
        if not r.get("ik_success", False):
            continue
        if r["grasp_type"] == "under_or_low_side_grasp":
            continue
        if r["open_support_risk"] >= 120.0:
            continue
        if r["approach_risk"] >= 100.0:
            continue
        if not r["proxy_ready_close"]:
            continue
        # close support 太严重的先别送执行；轻中度接触留给 contact-aware close。
        if r["close_support_risk"] >= 180.0:
            continue
        selected.append(r)
        if len(selected) >= top_k:
            break
    return selected


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
    ap.add_argument("--object-clearance", type=float, default=0.003)

    ap.add_argument("--object-pos", default="")
    ap.add_argument("--object-quat-wxyz", default="1 0 0 0")
    ap.add_argument("--object-scale", default="", help="fixed object scale for all priors")
    ap.add_argument("--fixed-object-scale-sample-index", type=int, default=-1,
                    help="use this sample's scale as current object scale for all candidates")
    ap.add_argument("--use-scene-object-pose", action="store_true",
                    help="read T_world_object from --model object body instead of recomputing from support")
    ap.add_argument("--all-geoms", action="store_true")
    ap.add_argument("--debug-first-n", type=int, default=0)
    args = ap.parse_args()

    npy_path = H.resolve(args.npy)
    mesh_path = H.resolve(args.mesh)
    model_path = H.resolve(args.model)
    out_dir = H.resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    support_center_xy = H.parse_vec2(args.support_center_xy)
    support_half_size = H.parse_vec3(args.support_half_size)
    support_top_z = float(args.support_top_z)
    table_top_z = float(args.table_top_z)

    verts = H.read_obj_vertices(mesh_path)
    samples_all = H.load_samples(npy_path)
    samples = samples_all[:args.debug_first_n] if args.debug_first_n > 0 else samples_all

    fixed_scale = None
    if args.object_scale.strip():
        fixed_scale = float(args.object_scale)
    elif args.fixed_object_scale_sample_index >= 0:
        found = False
        for li, s in samples_all:
            if li == args.fixed_object_scale_sample_index:
                fixed_scale = H.sample_scale(s)
                found = True
                break
        if not found:
            raise RuntimeError(f"fixed scale sample not found: {args.fixed_object_scale_sample_index}")

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    support_gids = collect_support_geom_ids(model)

    if args.use_scene_object_pose:
        T_world_object_scene = body_world_T(model, data, args.object_body)
    else:
        T_world_object_scene = None

    rows = []

    print("========== V4.20d REAL PROXY SUPPORT FIX SELECTOR ==========")
    print("object_code :", args.object_code)
    print("npy         :", H.rel(npy_path))
    print("mesh        :", H.rel(mesh_path))
    print("model       :", H.rel(model_path))
    print("samples     :", len(samples))
    print("target_site :", args.target_site)
    print("support_gids:", [(int(g), H.geom_name(model, g), H.geom_body_name(model, g)) for g in support_gids])
    print("fixed_scale :", fixed_scale)
    print("use_scene_object_pose:", args.use_scene_object_pose)

    for count, (local_i, sample) in enumerate(samples, start=1):
        raw_i = H.sample_raw_index(local_i, sample)

        scale = float(fixed_scale if fixed_scale is not None else H.sample_scale(sample))
        verts_scaled = verts * scale
        bbox_min = verts_scaled.min(axis=0)
        bbox_max = verts_scaled.max(axis=0)
        bbox_size = bbox_max - bbox_min

        R_obj = H.quat_wxyz_to_R(H.parse_quat_wxyz(args.object_quat_wxyz))

        if T_world_object_scene is not None:
            T_world_object = np.array(T_world_object_scene, dtype=float)
        else:
            if args.object_pos.strip():
                object_pos = H.parse_vec3(args.object_pos)
            else:
                object_pos = H.compute_object_pos_on_support(
                    verts_scaled=verts_scaled,
                    R_obj=R_obj,
                    support_center_xy=support_center_xy,
                    support_top_z=support_top_z,
                    clearance=args.object_clearance,
                )
            T_world_object = H.T_from_Rp(R_obj, object_pos)

        T_object_hand = H.extract_T_object_hand(sample)
        T_world_hand = T_world_object @ T_object_hand
        close_ctrl = H.extract_ctrl(sample)
        open_ctrl = side_open_from_close(close_ctrl)

        object_world_verts = H.transform_points(T_world_object, verts_scaled)
        ow_min = object_world_verts.min(axis=0)
        ow_max = object_world_verts.max(axis=0)
        ow_size = ow_max - ow_min

        ik = H.solve_site_ik(model, args.target_site, T_world_hand, H.Q_HOME)

        base = {
            "object_code": args.object_code,
            "valid_local_index": int(local_i),
            "raw_sample_index": int(raw_i),
            "scale_used": float(scale),
            "T_object_hand_pos": T_object_hand[:3, 3].tolist(),
            "T_world_hand_pos": T_world_hand[:3, 3].tolist(),
            "T_world_object_pos": T_world_object[:3, 3].tolist(),
            "object_bbox_size_scaled": bbox_size.tolist(),
            "ik_success": bool(ik["success"]),
            "ik_pos_err": float(ik["final_pos_err_norm"]),
            "ik_rot_err": float(ik["final_rot_err_norm"]),
            "close_ctrl": close_ctrl,
            "open_ctrl": open_ctrl,
        }

        if not ik["success"]:
            row = dict(base)
            row.update({
                "grasp_type": "ik_failed",
                "score": -1e6,
                "proxy_ready_close": False,
                "contact_score": -1e6,
                "open_support_risk": 1e6,
                "close_support_risk": 1e6,
                "approach_risk": 1e6,
                "type_score": -1e6,
                "rank_reason_short": "ik_failed",
                "score_reasons": ["ik_failed"],
            })
            rows.append(row)
            print(f"[{count:03d}/{len(samples):03d}] local={local_i:03d} IK_FAIL pos_err={ik['final_pos_err_norm']:.5f}")
            continue

        # ---------- open stage ----------
        H.set_qpos_once(model, data, ik["q_arm"], open_ctrl)
        open_records, open_counts = H.collect_real_hand_geom_records(
            model, data, use_collision_only=not args.all_geoms
        )
        if sum(open_counts.values()) == 0 and not args.all_geoms:
            open_records, open_counts = H.collect_real_hand_geom_records(
                model, data, use_collision_only=False
            )

        open_obj_dists, open_obj_best = H.surface_distances_to_object_aabb(
            open_records, T_world_object, bbox_min, bbox_max
        )
        open_support_min, open_support_group_min, open_support_best = support_distance_by_geom(
            model, data, open_records, support_gids, fallback_table_top_z=table_top_z
        )

        # ---------- close stage ----------
        H.set_qpos_once(model, data, ik["q_arm"], close_ctrl)
        close_records, close_counts = H.collect_real_hand_geom_records(
            model, data, use_collision_only=not args.all_geoms
        )
        if sum(close_counts.values()) == 0 and not args.all_geoms:
            close_records, close_counts = H.collect_real_hand_geom_records(
                model, data, use_collision_only=False
            )

        close_obj_dists, close_obj_best = H.surface_distances_to_object_aabb(
            close_records, T_world_object, bbox_min, bbox_max
        )
        close_support_min, close_support_group_min, close_support_best = support_distance_by_geom(
            model, data, close_records, support_gids, fallback_table_top_z=table_top_z
        )

        approach_risk, approach_reasons, approach_dot_up = approach_risk_world(
            T_world_hand, T_world_object, support_top_z
        )

        grasp_type, type_reasons, above_ratio, endness, horiz_norm = classify_grasp_open_support(
            T_world_hand=T_world_hand,
            T_world_object=T_world_object,
            object_world_bbox_min=ow_min,
            object_world_bbox_max=ow_max,
            support_top_z=support_top_z,
            open_support_min=open_support_min,
            bbox_obj_min=bbox_min,
            bbox_obj_max=bbox_max,
            approach_dot_up=approach_dot_up,
        )

        contact = contact_score_from_close(close_obj_dists, ow_size)

        open_support_risk, open_support_reasons = support_risk_from_distance(open_support_min, "open")
        close_support_risk, close_support_reasons = support_risk_from_distance(close_support_min, "close")

        object_penalty, pen_reasons = object_proxy_penetration_penalty(close_obj_dists)

        type_score = type_score_value(grasp_type)

        final_score = (
            contact["contact_score"]
            + type_score
            - open_support_risk
            - 0.45 * close_support_risk
            - approach_risk
            - object_penalty
        )

        score_reasons = []
        score_reasons.extend(contact["contact_reasons"])
        score_reasons.extend(open_support_reasons)
        score_reasons.extend(close_support_reasons)
        score_reasons.append(f"type_score={type_score:.1f} type={grasp_type}")
        score_reasons.extend(approach_reasons)
        score_reasons.extend(pen_reasons)
        score_reasons.extend(type_reasons)

        row = dict(base)
        row.update({
            "grasp_type": grasp_type,
            "score": float(final_score),
            "final_score": float(final_score),

            "proxy_ready_open": False,
            "proxy_ready_close": bool(contact["proxy_ready_close"]),
            "near_thumb_close": bool(contact["near_thumb"]),
            "near_nonthumb_close": contact["near_nonthumb"],

            "contact_score": float(contact["contact_score"]),
            "open_support_risk": float(open_support_risk),
            "close_support_risk": float(close_support_risk),
            "approach_risk": float(approach_risk),
            "object_penetration_penalty": float(object_penalty),
            "type_score": float(type_score),

            "thumb_near_threshold": float(contact["thumb_near_threshold"]),
            "nonthumb_near_threshold": float(contact["nonthumb_near_threshold"]),

            "open_thumb_dist": float(open_obj_dists.get("thumb", 999.0)),
            "open_best_nonthumb_dist": float(min(open_obj_dists.get(g, 999.0) for g in NON_THUMB)),
            "close_thumb_dist": float(contact["thumb_surface_dist"]),
            "close_best_nonthumb_dist": float(contact["best_nonthumb_surface_dist"]),

            "open_group_object_surface_distances": open_obj_dists,
            "close_group_object_surface_distances": close_obj_dists,
            "open_group_best_geoms": open_obj_best,
            "close_group_best_geoms": close_obj_best,

            "open_support_min": float(open_support_min),
            "close_support_min": float(close_support_min),
            "open_support_group_min": open_support_group_min,
            "close_support_group_min": close_support_group_min,
            "open_support_best_geoms": open_support_best,
            "close_support_best_geoms": close_support_best,

            "open_real_group_counts": open_counts,
            "close_real_group_counts": close_counts,

            "approach_dot_up": float(approach_dot_up),
            "above_ratio": float(above_ratio),
            "endness": float(endness),
            "horiz_norm": float(horiz_norm),

            "type_reasons": type_reasons,
            "rank_reason_short": "; ".join(score_reasons[:6]),
            "score_reasons": score_reasons,
        })

        rows.append(row)

        print(
            f"[{count:03d}/{len(samples):03d}] "
            f"local={local_i:03d} raw={raw_i:03d} "
            f"type={grasp_type:24s} score={final_score:+8.2f} "
            f"readyC={contact['proxy_ready_close']} "
            f"openS={open_support_min:+.4f} closeS={close_support_min:+.4f} "
            f"thumbC={contact['thumb_surface_dist']:+.4f} "
            f"nonC={contact['best_nonthumb_surface_dist']:+.4f} "
            f"ik=True"
        )

    rows_sorted = sorted(rows, key=lambda x: x["score"], reverse=True)

    selected_overall = select_overall(rows_sorted, args.top_k)
    selected_diverse = select_diverse(rows_sorted, args.top_k, args.max_per_type)
    selected_feasible = select_feasible(rows_sorted, args.top_k)

    H.save_json(out_dir / "ranked_v420d_candidates.json", rows_sorted)
    H.save_json(out_dir / "selected_topk_overall.json", selected_overall)
    H.save_json(out_dir / "selected_topk_diverse.json", selected_diverse)
    H.save_json(out_dir / "selected_topk_feasible.json", selected_feasible)

    (out_dir / "selected_valid_local_indices_overall.txt").write_text(
        "\n".join(str(r["valid_local_index"]) for r in selected_overall) + "\n"
    )
    (out_dir / "selected_valid_local_indices_diverse.txt").write_text(
        "\n".join(str(r["valid_local_index"]) for r in selected_diverse) + "\n"
    )
    (out_dir / "selected_valid_local_indices_feasible.txt").write_text(
        "\n".join(str(r["valid_local_index"]) for r in selected_feasible) + "\n"
    )

    csv_path = out_dir / "ranked_v420d_candidates.csv"
    with csv_path.open("w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "local", "raw", "type", "score",
            "ready_close", "contact_score",
            "open_support_risk", "close_support_risk",
            "approach_risk", "object_penalty", "type_score",
            "open_support_min", "close_support_min",
            "close_thumb_dist", "close_best_nonthumb_dist",
            "ik_success", "ik_pos_err",
            "rank_reason_short",
        ])
        for rank, r in enumerate(rows_sorted, start=1):
            w.writerow([
                rank,
                r["valid_local_index"],
                r["raw_sample_index"],
                r["grasp_type"],
                f"{r['score']:.6f}",
                r.get("proxy_ready_close"),
                f"{r.get('contact_score', 0):.6f}",
                f"{r.get('open_support_risk', 0):.6f}",
                f"{r.get('close_support_risk', 0):.6f}",
                f"{r.get('approach_risk', 0):.6f}",
                f"{r.get('object_penetration_penalty', 0):.6f}",
                f"{r.get('type_score', 0):.6f}",
                f"{r.get('open_support_min', 999):.6f}",
                f"{r.get('close_support_min', 999):.6f}",
                f"{r.get('close_thumb_dist', 999):.6f}",
                f"{r.get('close_best_nonthumb_dist', 999):.6f}",
                r.get("ik_success"),
                f"{r.get('ik_pos_err', 999):.6f}",
                r.get("rank_reason_short", ""),
            ])

    type_counts = {}
    ready_close_count = 0
    feasible_count = len(selected_feasible)
    ik_fail_count = 0

    for r in rows_sorted:
        type_counts[r["grasp_type"]] = type_counts.get(r["grasp_type"], 0) + 1
        if r.get("proxy_ready_close", False):
            ready_close_count += 1
        if not r.get("ik_success", False):
            ik_fail_count += 1

    lines = []
    lines.append("========== V4.20d REAL PROXY SUPPORT FIX SELECTOR ==========")
    lines.append("定位：real hand proxy + open/close support split；目标 Top-K 高召回，不是最终执行器。")
    lines.append(f"object_code : {args.object_code}")
    lines.append(f"npy         : {H.rel(npy_path)}")
    lines.append(f"mesh        : {H.rel(mesh_path)}")
    lines.append(f"model       : {H.rel(model_path)}")
    lines.append(f"samples     : {len(rows_sorted)}")
    lines.append(f"top_k       : {args.top_k}")
    lines.append(f"fixed_scale : {fixed_scale}")
    lines.append(f"use_scene_object_pose: {args.use_scene_object_pose}")
    lines.append(f"support_gids: {[(int(g), H.geom_name(model,g), H.geom_body_name(model,g)) for g in support_gids]}")
    lines.append(f"ready_close_count: {ready_close_count}")
    lines.append(f"ik_fail_count    : {ik_fail_count}")
    lines.append(f"feasible_count   : {feasible_count}")
    lines.append("")
    lines.append("---- type counts ----")
    for k, v in sorted(type_counts.items()):
        lines.append(f"{k}: {v}")

    def append_top(title, selected):
        lines.append("")
        lines.append(f"---- {title} ----")
        if not selected:
            lines.append("(empty)")
            return
        for rank, r in enumerate(selected, start=1):
            lines.append(
                f"rank={rank:02d} "
                f"local={r['valid_local_index']:03d} raw={r['raw_sample_index']:03d} "
                f"type={r['grasp_type']} score={r['score']:.2f} readyC={r.get('proxy_ready_close')} "
                f"C={r.get('contact_score',0):.1f} "
                f"openS={r.get('open_support_risk',0):.1f}/{r.get('open_support_min',999):+.4f} "
                f"closeS={r.get('close_support_risk',0):.1f}/{r.get('close_support_min',999):+.4f} "
                f"A={r.get('approach_risk',0):.1f} "
                f"thumbC={r.get('close_thumb_dist',999):+.4f} "
                f"nonC={r.get('close_best_nonthumb_dist',999):+.4f} "
                f"ik={r.get('ik_success')}"
            )
            lines.append(f"  - {r.get('rank_reason_short','')}")

    append_top("selected_topk_overall", selected_overall)
    append_top("selected_topk_diverse", selected_diverse)
    append_top("selected_topk_feasible", selected_feasible)

    lines.append("")
    lines.append("---- output ----")
    lines.append(f"ranked json   : {H.rel(out_dir / 'ranked_v420d_candidates.json')}")
    lines.append(f"ranked csv    : {H.rel(csv_path)}")
    lines.append(f"overall topk  : {H.rel(out_dir / 'selected_topk_overall.json')}")
    lines.append(f"diverse topk  : {H.rel(out_dir / 'selected_topk_diverse.json')}")
    lines.append(f"feasible topk : {H.rel(out_dir / 'selected_topk_feasible.json')}")
    lines.append("=============================================================")

    report = "\n".join(lines) + "\n"
    (out_dir / "v420d_selector_report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
