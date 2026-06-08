#!/usr/bin/env python3
"""
脚本类型：
    debug / selector / scene-aware-general-grasp / V4.13

用途：
    通用抓握选择器 V1。
    面向“桌面/支撑物上随机放置一个数据集物体”的在线泛化流程：
        1. 读取该物体的五类先验文件：
           object.npy
           object_metrics_valid.json
           object_metrics_valid_per_sample.npz
           object_validation_flags.npz
           object_validation_meta.json
        2. 对每个 valid grasp prior 计算 object-only 分数；
        3. 根据 hand_pose 相对物体的位置粗分类抓型；
        4. 如果给了 scene.xml，则解析 object pose 与 support top，做 scene-aware access 粗过滤；
        5. 输出 Top-K 候选及原因，供后续 candidate/scene 生成、P2/P3、P4U6/P4U1 使用。

输入：
    --object-code
    --prior-dir 或五个显式文件路径
    --mesh-root 或 --object-mesh
    可选 --scene，用于解析当前物体 pose 与 support top
    可选 --support-top-z，用于覆盖/指定支撑面高度

输出：
    out-dir/
        general_grasp_select_summary.json
        general_grasp_select_report.txt
        selected_sample_indices.txt
        selected_valid_local_indices.txt
        selected_topk_compact.json

当前流程位置：
    dataset prior / five prior files
        -> 本脚本通用场景感知筛选
        -> 后续只对 Top-1/Top-3 做 candidate/scene、P2/P3、P4U6/P4U1

不负责：
    1. 不运行 P2/P3；
    2. 不运行 viewer；
    3. 不修改 legacy_final_demos；
    4. 不替代 P4U6/P4U1 执行器；
    5. 不保证 V1 的粗几何判断等价于完整接触仿真。
"""

from __future__ import annotations

import argparse
import json
import math
import os
import re
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import numpy as np


PROJECT = Path.home() / "Projects/o7_mujoco_sim"

ACTIVE_JOINTS = [
    "thumb_cmc_roll",
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]


def resolve_path(p: Optional[str], project: Path = PROJECT) -> Optional[Path]:
    if p is None or str(p).strip() == "":
        return None
    q = Path(str(p)).expanduser()
    if not q.is_absolute():
        q = project / q
    return q


def rel(p: Path, project: Path = PROJECT) -> str:
    try:
        return str(Path(p).resolve().relative_to(project.resolve()))
    except Exception:
        return str(p)


def load_json(path: Path) -> Dict[str, Any]:
    return json.loads(path.read_text())


def save_json(path: Path, obj: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def parse_float_list(s: str, n: int) -> Optional[np.ndarray]:
    try:
        xs = [float(x) for x in str(s).replace(",", " ").split()]
        if len(xs) != n:
            return None
        return np.asarray(xs, dtype=float)
    except Exception:
        return None


def quat_to_R_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=float)
    if q.shape != (4,):
        return np.eye(3)
    n = np.linalg.norm(q)
    if n < 1e-12:
        return np.eye(3)
    w, x, y, z = q / n
    return np.array([
        [1 - 2*y*y - 2*z*z,     2*x*y - 2*z*w,     2*x*z + 2*y*w],
        [    2*x*y + 2*z*w, 1 - 2*x*x - 2*z*z,     2*y*z - 2*x*w],
        [    2*x*z - 2*y*w,     2*y*z + 2*x*w, 1 - 2*x*x - 2*y*y],
    ], dtype=float)


def euler_xyz_to_R(euler: np.ndarray) -> np.ndarray:
    rx, ry, rz = [float(x) for x in euler]
    cx, sx = math.cos(rx), math.sin(rx)
    cy, sy = math.cos(ry), math.sin(ry)
    cz, sz = math.cos(rz), math.sin(rz)
    Rx = np.array([[1,0,0],[0,cx,-sx],[0,sx,cx]], dtype=float)
    Ry = np.array([[cy,0,sy],[0,1,0],[-sy,0,cy]], dtype=float)
    Rz = np.array([[cz,-sz,0],[sz,cz,0],[0,0,1]], dtype=float)
    return Rz @ Ry @ Rx


def safe_num(x: Any, default: float = float("nan")) -> float:
    try:
        y = float(x)
        if math.isfinite(y):
            return y
        return default
    except Exception:
        return default


def robust_scores_low_good(vals: np.ndarray) -> np.ndarray:
    vals = np.asarray(vals, dtype=float)
    out = np.zeros_like(vals, dtype=float)
    mask = np.isfinite(vals)
    if np.count_nonzero(mask) < 2:
        return out
    med = np.nanmedian(vals[mask])
    q25 = np.nanpercentile(vals[mask], 25)
    q75 = np.nanpercentile(vals[mask], 75)
    scale = max(q75 - q25, 1e-9)
    z = (vals - med) / scale
    out[mask] = np.clip(-z, -3.0, 3.0)
    return out


def robust_scores_high_good(vals: np.ndarray) -> np.ndarray:
    return -robust_scores_low_good(vals)


def read_obj_bbox(obj_path: Optional[Path]) -> Dict[str, Any]:
    if obj_path is None or not obj_path.exists():
        return {"ok": False, "reason": "missing_mesh", "path": str(obj_path)}
    verts = []
    with obj_path.open("r", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except Exception:
                pass
    if not verts:
        return {"ok": False, "reason": "no_vertices", "path": str(obj_path)}
    pts = np.asarray(verts, dtype=float)
    mn, mx = pts.min(axis=0), pts.max(axis=0)
    size = mx - mn
    return {
        "ok": True,
        "path": str(obj_path),
        "bbox_min": mn.tolist(),
        "bbox_max": mx.tolist(),
        "bbox_size": size.tolist(),
        "bbox_center": ((mn + mx) * 0.5).tolist(),
        "long_axis": int(np.argmax(size)),
        "short_axis": int(np.argmin(size)),
    }


def discover_files(args: argparse.Namespace, project: Path) -> Dict[str, Path]:
    object_code = args.object_code
    prior_dir = resolve_path(args.prior_dir, project)

    paths = {
        "npy": resolve_path(args.npy, project),
        "metrics_json": resolve_path(args.metrics_json, project),
        "per_sample_npz": resolve_path(args.per_sample_npz, project),
        "flags_npz": resolve_path(args.flags_npz, project),
        "meta_json": resolve_path(args.meta_json, project),
    }

    if prior_dir is not None:
        defaults = {
            "npy": prior_dir / f"{object_code}.npy",
            "metrics_json": prior_dir / f"{object_code}_metrics_valid.json",
            "per_sample_npz": prior_dir / f"{object_code}_metrics_valid_per_sample.npz",
            "flags_npz": prior_dir / f"{object_code}_validation_flags.npz",
            "meta_json": prior_dir / f"{object_code}_validation_meta.json",
        }
        for k, p in defaults.items():
            if paths[k] is None:
                paths[k] = p

    missing = [k for k, p in paths.items() if p is None or not p.exists()]
    if missing:
        detail = "\n".join(f"{k}: {paths.get(k)}" for k in missing)
        raise FileNotFoundError("missing required prior files:\n" + detail)

    return paths  # type: ignore


def get_sample_array(npy_path: Path) -> List[Dict[str, Any]]:
    arr = np.load(npy_path, allow_pickle=True)
    samples = []
    for x in arr:
        d = x.item() if hasattr(x, "item") and not isinstance(x, dict) else x
        if not isinstance(d, dict):
            raise RuntimeError(f"unexpected npy item type: {type(d)}")
        samples.append(d)
    return samples


def get_hand_translation(sample: Dict[str, Any]) -> Optional[np.ndarray]:
    hp = sample.get("hand_pose")
    if hp is None:
        return None
    arr = np.asarray(hp, dtype=float).reshape(-1)
    if arr.size < 3:
        return None
    return arr[:3].copy()


def get_active_ctrl(sample: Dict[str, Any]) -> Dict[str, float]:
    qpos = sample.get("qpos", {})
    if not isinstance(qpos, dict):
        return {}
    out = {}
    for j in ACTIVE_JOINTS:
        if j in qpos:
            out[j] = safe_num(qpos[j])
    return out


def parse_scene(scene_path: Optional[Path], object_body: str) -> Dict[str, Any]:
    info: Dict[str, Any] = {
        "scene_given": scene_path is not None,
        "scene_ok": False,
        "object_body": object_body,
        "object_pos_world": [0.0, 0.0, 0.0],
        "R_world_object": np.eye(3).tolist(),
        "support_top_z": None,
        "support_candidates": [],
    }
    if scene_path is None:
        return info
    if not scene_path.exists():
        info["error"] = f"scene missing: {scene_path}"
        return info

    try:
        root = ET.parse(str(scene_path)).getroot()
        # MuJoCo worldbody body pose. This is coarse; nested body transforms are not accumulated in V1.
        for b in root.iter("body"):
            if b.attrib.get("name") == object_body:
                pos = parse_float_list(b.attrib.get("pos", "0 0 0"), 3)
                if pos is not None:
                    info["object_pos_world"] = pos.tolist()
                if "quat" in b.attrib:
                    q = parse_float_list(b.attrib["quat"], 4)
                    if q is not None:
                        info["R_world_object"] = quat_to_R_wxyz(q).tolist()
                elif "euler" in b.attrib:
                    e = parse_float_list(b.attrib["euler"], 3)
                    if e is not None:
                        info["R_world_object"] = euler_xyz_to_R(e).tolist()
                break

        support_candidates = []
        for g in root.iter("geom"):
            name = (g.attrib.get("name") or "").lower()
            gtype = g.attrib.get("type", "")
            if not any(tok in name for tok in ["support", "pedestal", "table", "desk", "plane"]):
                continue
            pos = parse_float_list(g.attrib.get("pos", "0 0 0"), 3)
            size = parse_float_list(g.attrib.get("size", "0 0 0"), 3)
            if pos is None:
                pos = np.zeros(3)
            top_z = None
            if gtype == "box" and size is not None and size.size >= 3:
                top_z = float(pos[2] + size[2])
            elif gtype == "plane":
                top_z = float(pos[2])
            elif size is not None and size.size >= 1:
                top_z = float(pos[2] + size[-1])
            support_candidates.append({
                "name": g.attrib.get("name", ""),
                "type": gtype,
                "pos": pos.tolist(),
                "size": size.tolist() if size is not None else None,
                "top_z": top_z,
            })

        valid_tops = [x["top_z"] for x in support_candidates if x.get("top_z") is not None]
        if valid_tops:
            info["support_top_z"] = float(max(valid_tops))
        info["support_candidates"] = support_candidates
        info["scene_ok"] = True
    except Exception as e:
        info["error"] = repr(e)
    return info


def classify_grasp(sample: Dict[str, Any], bbox: Dict[str, Any], scene_info: Dict[str, Any]) -> Tuple[str, List[str], Dict[str, Any]]:
    p = get_hand_translation(sample)
    if p is None:
        return "unknown", ["missing hand_pose[:3]"], {}

    scale = safe_num(sample.get("scale"), 1.0)
    bbox_size = np.asarray(bbox.get("bbox_size", [1.0, 1.0, 1.0]), dtype=float)
    if not bbox.get("ok"):
        bbox_size = np.ones(3)

    obj_size = np.maximum(bbox_size * max(scale, 1e-9), 1e-6)
    Rwo = np.asarray(scene_info.get("R_world_object", np.eye(3)), dtype=float)
    support_normal_world = np.array([0.0, 0.0, 1.0], dtype=float)
    support_normal_obj = Rwo.T @ support_normal_world
    support_axis = int(np.argmax(np.abs(support_normal_obj)))
    sign = float(np.sign(support_normal_obj[support_axis]) or 1.0)

    h = 0.5 * obj_size
    norm_p = p / np.maximum(h, 1e-6)
    vertical_component = sign * norm_p[support_axis]
    lateral_axes = [i for i in range(3) if i != support_axis]
    lateral_norm = float(np.linalg.norm(norm_p[lateral_axes]))
    long_axis = int(np.argmax(obj_size))
    end_component = abs(norm_p[long_axis])

    reasons = [
        f"p_object={p.tolist()}",
        f"scale={scale:.5f}",
        f"object_size≈{obj_size.tolist()}",
        f"support_axis_obj={support_axis}",
        f"vertical_component={vertical_component:.3f}",
        f"lateral_norm={lateral_norm:.3f}",
        f"end_component_long_axis={end_component:.3f}",
    ]

    if vertical_component > 1.05:
        gt = "top_grasp"
    elif vertical_component < -0.35:
        gt = "under_or_low_side_grasp"
    elif end_component > 1.05 and long_axis != support_axis:
        gt = "end_grasp"
    elif lateral_norm > 0.80:
        gt = "side_grasp"
    else:
        gt = "ambiguous_grasp"

    features = {
        "p_object": p.tolist(),
        "scale": scale,
        "object_size_est": obj_size.tolist(),
        "support_axis_obj": support_axis,
        "support_axis_sign": sign,
        "vertical_component": vertical_component,
        "lateral_norm": lateral_norm,
        "end_component": end_component,
        "long_axis": long_axis,
    }
    return gt, reasons, features


def object_only_score(sample: Dict[str, Any], idx: int, arrays: Dict[str, np.ndarray]) -> Tuple[float, List[str]]:
    reasons = []
    score = 0.0

    low_good_keys = [
        "energy", "E_fc", "E_dis", "E_pen", "E_spen", "E_opp", "E_syn",
        "metric_pen_depth", "metric_synergy_err",
    ]
    high_good_keys = ["metric_thumb_contact"]

    for k in low_good_keys:
        vals = arrays.get(k)
        if vals is not None and idx < len(vals):
            s = float(robust_scores_low_good(vals)[idx])
            score += 4.0 * s
            reasons.append(f"{k}_low_good_zscore={s:.3f}")
    for k in high_good_keys:
        vals = arrays.get(k)
        if vals is not None and idx < len(vals):
            s = float(robust_scores_high_good(vals)[idx])
            score += 8.0 * s
            reasons.append(f"{k}_high_good_zscore={s:.3f}")

    # q1 含义在不同管线可能不同，V1 只弱加权，不作为硬判断。
    vals = arrays.get("metric_q1")
    if vals is not None and idx < len(vals):
        s = float(robust_scores_low_good(vals)[idx])
        score += 1.0 * s
        reasons.append(f"metric_q1_weak_low_good_zscore={s:.3f}")

    ctrl = get_active_ctrl(sample)
    if len(ctrl) == len(ACTIVE_JOINTS):
        fingers = [ctrl.get(j, 0.0) for j in ["index_mcp_pitch", "middle_mcp_pitch", "ring_mcp_pitch", "pinky_mcp_pitch"]]
        finger_mean = float(np.mean(fingers))
        thumb_yaw = ctrl.get("thumb_cmc_yaw", 0.0)
        thumb_roll = ctrl.get("thumb_cmc_roll", 0.0)
        if 0.15 <= finger_mean <= 0.95:
            score += 10.0
            reasons.append(f"finger_mean_ok={finger_mean:.3f}")
        else:
            score -= 8.0
            reasons.append(f"finger_mean_out={finger_mean:.3f}")
        if abs(thumb_yaw) > 0.20 or abs(thumb_roll) > 0.20:
            score += 5.0
            reasons.append(f"thumb_active_ok roll={thumb_roll:.3f} yaw={thumb_yaw:.3f}")
        else:
            score -= 4.0
            reasons.append(f"thumb_not_active roll={thumb_roll:.3f} yaw={thumb_yaw:.3f}")
    else:
        score -= 10.0
        reasons.append("active_ctrl_incomplete")

    return float(score), reasons


def scene_score_for_grasp(
    grasp_type: str,
    features: Dict[str, Any],
    sample: Dict[str, Any],
    scene_info: Dict[str, Any],
    object_top_z: Optional[float],
    args: argparse.Namespace,
) -> Tuple[float, bool, str, List[str]]:
    reasons: List[str] = []
    score = 0.0
    hard_fail = False
    hard_reason = ""

    if not scene_info.get("scene_given") and args.support_top_z is None:
        return 0.0, False, "", ["scene not given: scene-aware access is neutral"]

    p_obj = np.asarray(features.get("p_object", [0, 0, 0]), dtype=float)
    Rwo = np.asarray(scene_info.get("R_world_object", np.eye(3)), dtype=float)
    obj_pos = np.asarray(scene_info.get("object_pos_world", [0, 0, 0]), dtype=float)
    p_world = obj_pos + Rwo @ p_obj

    support_top = args.support_top_z
    if support_top is None:
        support_top = scene_info.get("support_top_z")
    if support_top is None:
        return 0.0, False, "", ["support_top_z unknown: scene-aware access is neutral"]

    clearance = float(p_world[2] - support_top)
    reasons.append(f"hand_target_world_z={p_world[2]:.5f}")
    reasons.append(f"support_top_z={float(support_top):.5f}")
    reasons.append(f"hand_target_above_support={clearance:.5f}")

    # 这是粗代理：真正 thumb/finger swept volume 后面再用 MuJoCo/P3/P4 检。
    if grasp_type in ["side_grasp", "under_or_low_side_grasp", "end_grasp"]:
        if clearance < args.hard_block_margin:
            hard_fail = True
            hard_reason = "HAND_TARGET_BELOW_SUPPORT_HARD_BLOCK"
            score -= 100.0
            reasons.append("hard fail: target below support; likely thumb/finger access blocked")
        elif clearance < args.soft_touch_margin:
            score -= 25.0
            reasons.append("soft risk: target very close to support; allow only if later contact check passes")
        else:
            score += 20.0
            reasons.append("side/end access target above support")

    if grasp_type == "top_grasp":
        if object_top_z is not None:
            dz = float(p_world[2] - object_top_z)
            reasons.append(f"object_top_z≈{object_top_z:.5f}")
            reasons.append(f"hand_target_above_object_top≈{dz:.5f}")
            if dz < -0.02:
                score -= 40.0
                reasons.append("top grasp target is below object top: suspicious")
            elif dz > 0.22:
                score -= 20.0
                reasons.append("top grasp target too high: may miss object")
            else:
                score += 25.0
                reasons.append("top access height reasonable")
        else:
            score += 5.0
            reasons.append("top grasp kept, but object_top_z unknown")

    return float(score), hard_fail, hard_reason, reasons


def estimate_object_top_z(bbox: Dict[str, Any], scene_info: Dict[str, Any], sample_scale: float) -> Optional[float]:
    if not bbox.get("ok") or not scene_info.get("scene_given"):
        return None
    obj_pos = np.asarray(scene_info.get("object_pos_world", [0, 0, 0]), dtype=float)
    Rwo = np.asarray(scene_info.get("R_world_object", np.eye(3)), dtype=float)
    mn = np.asarray(bbox["bbox_min"], dtype=float) * sample_scale
    mx = np.asarray(bbox["bbox_max"], dtype=float) * sample_scale
    corners = []
    for x in [mn[0], mx[0]]:
        for y in [mn[1], mx[1]]:
            for z in [mn[2], mx[2]]:
                corners.append([x, y, z])
    pts = obj_pos[None, :] + np.asarray(corners) @ Rwo.T
    return float(np.max(pts[:, 2]))


def build_arrays(samples: List[Dict[str, Any]], per_npz: Dict[str, np.ndarray]) -> Dict[str, np.ndarray]:
    arrays: Dict[str, np.ndarray] = {}
    keys = [
        "energy", "E_fc", "E_dis", "E_pen", "E_spen", "E_opp", "E_syn",
        "metric_q1", "metric_pen_depth", "metric_thumb_contact", "metric_synergy_err",
    ]
    for k in keys:
        vals = []
        for s in samples:
            vals.append(safe_num(s.get(k), float("nan")))
        arrays[k] = np.asarray(vals, dtype=float)

    # 如果 per_sample 有更明确的 valid 顺序指标，则覆盖/补充。
    mapping = {
        "q1": "metric_q1",
        "pen_depth": "metric_pen_depth",
        "thumb_contact": "metric_thumb_contact",
        "synergy_err": "metric_synergy_err",
    }
    n = len(samples)
    for src, dst in mapping.items():
        if src in per_npz:
            a = np.asarray(per_npz[src], dtype=float)
            if a.shape[0] == n:
                arrays[dst] = a
    return arrays


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=str(PROJECT))
    ap.add_argument("--object-code", required=True)
    ap.add_argument("--prior-dir", default="")
    ap.add_argument("--npy", default="")
    ap.add_argument("--metrics-json", default="")
    ap.add_argument("--per-sample-npz", default="")
    ap.add_argument("--flags-npz", default="")
    ap.add_argument("--meta-json", default="")
    ap.add_argument("--mesh-root", default="dataset/meshdata")
    ap.add_argument("--object-mesh", default="")
    ap.add_argument("--scene", default="")
    ap.add_argument("--object-body", default="grasp_object")
    ap.add_argument("--support-top-z", type=float, default=None)
    ap.add_argument("--out-dir", default="diagnostics/current_v413/general_grasp_select_debug")
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--soft-touch-margin", type=float, default=0.015)
    ap.add_argument("--hard-block-margin", type=float, default=-0.006)
    args = ap.parse_args()

    project = Path(args.project_root).expanduser().resolve()
    files = discover_files(args, project)

    mesh_path = resolve_path(args.object_mesh, project)
    if mesh_path is None:
        mesh_root = resolve_path(args.mesh_root, project)
        mesh_path = mesh_root / args.object_code / "coacd" / "decomposed.obj" if mesh_root else None
    bbox = read_obj_bbox(mesh_path)

    scene_path = resolve_path(args.scene, project)
    scene_info = parse_scene(scene_path, args.object_body)

    out_dir = resolve_path(args.out_dir, project)
    assert out_dir is not None
    out_dir.mkdir(parents=True, exist_ok=True)

    samples = get_sample_array(files["npy"])
    metrics = load_json(files["metrics_json"])
    meta = load_json(files["meta_json"])
    per_npz = dict(np.load(files["per_sample_npz"], allow_pickle=True))
    flags_npz = dict(np.load(files["flags_npz"], allow_pickle=True))
    arrays = build_arrays(samples, per_npz)

    valid_indices = per_npz.get("valid_indices")
    if valid_indices is not None and len(valid_indices) == len(samples):
        raw_indices = [int(x) for x in valid_indices.tolist()]
    else:
        raw_indices = list(range(len(samples)))

    rows = []
    for i, sample in enumerate(samples):
        grasp_type, gt_reasons, features = classify_grasp(sample, bbox, scene_info)
        obj_score, obj_reasons = object_only_score(sample, i, arrays)

        scale = safe_num(sample.get("scale"), 1.0)
        obj_top_z = estimate_object_top_z(bbox, scene_info, scale)
        sc_score, hard_fail, hard_reason, sc_reasons = scene_score_for_grasp(
            grasp_type, features, sample, scene_info, obj_top_z, args
        )

        final = obj_score + sc_score
        if hard_fail:
            final -= 200.0

        qpos = sample.get("qpos", {}) if isinstance(sample.get("qpos"), dict) else {}
        active_ctrl = {j: safe_num(qpos.get(j)) for j in ACTIVE_JOINTS if j in qpos}

        row = {
            "valid_local_index": i,
            "raw_sample_index": raw_indices[i],
            "object_code": args.object_code,
            "scale": scale,
            "grasp_type": grasp_type,
            "final_score": float(final),
            "object_only_score": float(obj_score),
            "scene_score": float(sc_score),
            "hard_fail": bool(hard_fail),
            "hard_fail_reason": hard_reason,
            "features": features,
            "active_ctrl": active_ctrl,
            "metrics": {
                "energy": safe_num(sample.get("energy")),
                "E_fc": safe_num(sample.get("E_fc")),
                "E_dis": safe_num(sample.get("E_dis")),
                "E_pen": safe_num(sample.get("E_pen")),
                "E_spen": safe_num(sample.get("E_spen")),
                "E_opp": safe_num(sample.get("E_opp")),
                "E_syn": safe_num(sample.get("E_syn")),
                "metric_q1": safe_num(sample.get("metric_q1")),
                "metric_pen_depth": safe_num(sample.get("metric_pen_depth")),
                "metric_thumb_contact": safe_num(sample.get("metric_thumb_contact")),
                "metric_synergy_err": safe_num(sample.get("metric_synergy_err")),
            },
            "reasons": {
                "grasp_type": gt_reasons,
                "object_only": obj_reasons,
                "scene": sc_reasons,
            },
        }

        if hard_fail:
            row["decision"] = "REJECT_SCENE_HARD_BLOCK"
        elif final >= 35:
            row["decision"] = "TRY_TOPK_P2P3"
        elif final >= 10:
            row["decision"] = "KEEP_AS_BACKUP"
        else:
            row["decision"] = "REJECT_LOW_SCORE"

        rows.append(row)

    rows_sorted = sorted(rows, key=lambda r: r["final_score"], reverse=True)
    selected = [r for r in rows_sorted if not r["hard_fail"]][:args.top_k]

    summary = {
        "format": "v4_13_scene_aware_general_grasp_select_debug_v1",
        "object_code": args.object_code,
        "project_root": str(project),
        "files": {k: rel(v, project) for k, v in files.items()},
        "mesh": bbox,
        "metrics_valid": metrics,
        "validation_meta": meta,
        "scene_info": {
            **{k: v for k, v in scene_info.items() if k != "R_world_object"},
            "R_world_object": scene_info.get("R_world_object"),
        },
        "num_samples": len(samples),
        "num_selected": len(selected),
        "selected": selected,
        "rows_sorted": rows_sorted,
    }
    save_json(out_dir / "general_grasp_select_summary.json", summary)

    (out_dir / "selected_sample_indices.txt").write_text(
        "\n".join(str(r["raw_sample_index"]) for r in selected) + "\n"
    )
    (out_dir / "selected_valid_local_indices.txt").write_text(
        "\n".join(str(r["valid_local_index"]) for r in selected) + "\n"
    )
    save_json(out_dir / "selected_topk_compact.json", selected)

    lines: List[str] = []
    lines.append("========== V4.13 SCENE-AWARE GENERAL GRASP SELECT ==========")
    lines.append(f"object_code: {args.object_code}")
    lines.append(f"npy samples: {len(samples)}")
    lines.append(f"metrics n_valid: {metrics.get('n_valid')}")
    lines.append(f"meta selection: {meta.get('selection_definition')}")
    lines.append(f"mesh_ok: {bbox.get('ok')} mesh: {bbox.get('path')}")
    lines.append(f"scene: {rel(scene_path, project) if scene_path else 'None'}")
    lines.append(f"support_top_z: {args.support_top_z if args.support_top_z is not None else scene_info.get('support_top_z')}")
    lines.append("")
    lines.append("---- TOP SELECTED ----")
    for rank, r in enumerate(selected, start=1):
        lines.append(
            f"rank={rank:02d} raw={r['raw_sample_index']:03d} local={r['valid_local_index']:03d} "
            f"type={r['grasp_type']} decision={r['decision']} final={r['final_score']:.3f} "
            f"object={r['object_only_score']:.3f} scene={r['scene_score']:.3f} "
            f"scale={r['scale']:.5f} hard={r['hard_fail_reason']}"
        )
        m = r["metrics"]
        lines.append(
            f"  metrics: E_fc={m['E_fc']:.6g} E_pen={m['E_pen']:.6g} "
            f"E_opp={m['E_opp']:.6g} thumb={m['metric_thumb_contact']:.4f} "
            f"syn={m['metric_synergy_err']:.6g}"
        )
        for rr in (r["reasons"]["grasp_type"][:4] + r["reasons"]["scene"][:4]):
            lines.append(f"  - {rr}")
    lines.append("")
    lines.append("---- ALL TOP 20 ----")
    for rank, r in enumerate(rows_sorted[:20], start=1):
        lines.append(
            f"{rank:02d}. raw={r['raw_sample_index']:03d} local={r['valid_local_index']:03d} "
            f"type={r['grasp_type']} decision={r['decision']} score={r['final_score']:.3f} "
            f"hard={r['hard_fail_reason']}"
        )
    lines.append("============================================================")
    report = "\n".join(lines) + "\n"
    (out_dir / "general_grasp_select_report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
