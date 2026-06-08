#!/usr/bin/env python3
"""
脚本类型：
    debug / v4.13e / support-aware-selector-rerank

用途：
    对 V4.13 selector 的 object-only Top-K 结果进行支撑面感知重排。
    当前 V4.13D 失败原因是 selector 把 under_or_low_side_grasp 排在前面，
    但该类抓型在桌面/蓝色支撑块场景中容易从支撑面下方进入，导致 P3 无组合或 hand-support 碰撞。

    本脚本不做任何人工微调，只做抓握选择层面的通用约束：
        1. 桌面支撑场景下，under_or_low_side_grasp 强降权；
        2. 计算 sample hand_pose 中 handbase 在当前 object placement 下相对 support_top_z 的高度；
        3. 优先选择 side_grasp / end_grasp / 合理 top_grasp；
        4. 保留多抓型配额，避免 Top-K 被同一种失败抓型占满；
        5. 输出新的 selected_topk_compact.json，供 V4.13D 继续 generic builder + P2/P3。

输入：
    1. V4.13 selector 生成的 selected_topk_compact.json，建议 top-k 足够大，比如 80；
    2. object.npy；
    3. object mesh；
    4. support_top_z / support footprint 参数。

输出：
    out_dir/
        selected_topk_compact.json
        selected_sample_indices.txt
        selected_valid_local_indices.txt
        v4_13e_support_aware_report.txt
        v4_13e_support_aware_summary.json

当前流程位置：
    V4.13 selector object-only
        -> 本脚本 support-aware rerank
        -> V4.13D generic builder + P2/P3
        -> viewer

不负责：
    1. 不跑 P2/P3；
    2. 不跑 viewer；
    3. 不修改 P4U1/P4U6；
    4. 不做沿轴人工微调；
    5. 不写死某个 sample。
"""

from pathlib import Path
import argparse
import json
import math
import numpy as np


PROJECT = Path.home() / "Projects/o7_mujoco_sim"


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


def load_json(path):
    return json.loads(Path(path).read_text())


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))


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
        raise RuntimeError(f"no vertices in mesh: {path}")
    return np.asarray(verts, dtype=float)


def load_sample(npy_path, local_idx):
    arr = np.load(npy_path, allow_pickle=True)
    if local_idx < 0 or local_idx >= len(arr):
        raise RuntimeError(f"local index out of range: {local_idx}, n={len(arr)}")
    sample = arr[local_idx].item() if hasattr(arr[local_idx], "item") else arr[local_idx]
    if not isinstance(sample, dict):
        raise RuntimeError(f"sample is not dict: {type(sample)}")
    return sample


def score_type(grasp_type, handbase_clearance, object_top_z, support_top_z):
    """
    只做选择层通用约束，不做姿态微调。
    """
    gt = str(grasp_type)
    reasons = []
    score = 0.0
    decision = "KEEP"

    if gt == "under_or_low_side_grasp":
        score -= 80
        reasons.append("under_or_low_side_grasp_downrank_for_support_scene")
        if finite(handbase_clearance) and handbase_clearance < 0.035:
            score -= 80
            decision = "REJECT_SUPPORT_UNDER"
            reasons.append(f"handbase_too_close_to_support={handbase_clearance:.5f}")
        return score, decision, reasons

    if gt == "side_grasp":
        score += 35
        reasons.append("side_grasp_preferred_for_elongated_object")
        if finite(handbase_clearance):
            if handbase_clearance >= 0.045:
                score += 25
                reasons.append(f"side_handbase_clearance_good={handbase_clearance:.5f}")
            elif handbase_clearance >= 0.025:
                score += 5
                reasons.append(f"side_handbase_clearance_marginal={handbase_clearance:.5f}")
            else:
                score -= 45
                reasons.append(f"side_handbase_too_low={handbase_clearance:.5f}")
        return score, decision, reasons

    if gt == "end_grasp":
        score += 30
        reasons.append("end_grasp_allowed")
        if finite(handbase_clearance) and handbase_clearance < 0.025:
            score -= 30
            reasons.append(f"end_handbase_low={handbase_clearance:.5f}")
        return score, decision, reasons

    if gt == "top_grasp":
        score += 15
        reasons.append("top_grasp_allowed_but_support_checked_by_p3")
        if finite(handbase_clearance):
            # top grasp 的 handbase 往往更高，但太低说明其实不是上抓。
            if handbase_clearance >= 0.070:
                score += 15
                reasons.append(f"top_handbase_height_reasonable={handbase_clearance:.5f}")
            else:
                score -= 25
                reasons.append(f"top_handbase_height_low={handbase_clearance:.5f}")
        return score, decision, reasons

    if gt == "ambiguous_grasp":
        score += 0
        reasons.append("ambiguous_keep_as_fallback")
        if finite(handbase_clearance) and handbase_clearance < 0.025:
            score -= 30
            reasons.append(f"ambiguous_handbase_low={handbase_clearance:.5f}")
        return score, decision, reasons

    score -= 10
    reasons.append(f"unknown_type={gt}")
    return score, decision, reasons


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--selector-json", required=True)
    ap.add_argument("--npy", required=True)
    ap.add_argument("--mesh", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--top-k", type=int, default=10)
    ap.add_argument("--support-top-z", type=float, default=0.23)
    ap.add_argument("--object-clearance", type=float, default=0.003)
    args = ap.parse_args()

    selector_json = resolve(args.selector_json)
    npy_path = resolve(args.npy)
    mesh_path = resolve(args.mesh)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_json(selector_json)
    if isinstance(rows, dict):
        rows = rows.get("selected") or rows.get("rows_sorted") or []

    verts = read_obj_vertices(mesh_path)

    scored = []

    for r in rows:
        row = dict(r)
        local = int(row["valid_local_index"])
        sample = load_sample(npy_path, local)

        scale = float(sample.get("scale", 1.0))
        hp = np.asarray(sample["hand_pose"], dtype=float)

        verts_scaled = verts * scale
        mesh_min = verts_scaled.min(axis=0)
        mesh_max = verts_scaled.max(axis=0)
        mesh_size = mesh_max - mesh_min

        # generic builder 里 object 默认不旋转，object body z 由 mesh bottom 对齐支撑面计算。
        object_pos_z = args.support_top_z + args.object_clearance - float(mesh_min[2])
        object_top_z = object_pos_z + float(mesh_max[2])

        # hand_pose[0:3] 是 object frame 下 hand_base_link 位置，这里只估计高度，不改变姿态。
        handbase_world_z = object_pos_z + float(hp[2])
        handbase_clearance = handbase_world_z - args.support_top_z

        selector_score = row.get("final_score")
        base = float(selector_score) if finite(selector_score) else 0.0

        type_score, decision, type_reasons = score_type(
            row.get("grasp_type"),
            handbase_clearance,
            object_top_z,
            args.support_top_z,
        )

        # 支撑感知最终分数：保留原先验分，但让场景约束能真正改变排序。
        final = 0.35 * base + type_score

        row["v413e_support_score"] = float(final)
        row["v413e_base_selector_score"] = base
        row["v413e_type_score"] = float(type_score)
        row["v413e_decision"] = decision
        row["v413e_reasons"] = type_reasons
        row["v413e_geometry"] = {
            "sample_scale": scale,
            "mesh_bbox_min_scaled": mesh_min.tolist(),
            "mesh_bbox_max_scaled": mesh_max.tolist(),
            "mesh_bbox_size_scaled": mesh_size.tolist(),
            "object_pos_z": object_pos_z,
            "object_top_z": object_top_z,
            "support_top_z": args.support_top_z,
            "handbase_world_z_est": handbase_world_z,
            "handbase_clearance_to_support_est": handbase_clearance,
            "hand_pose_translation_object_frame": hp[:3].tolist(),
        }

        scored.append(row)

    scored_sorted = sorted(scored, key=lambda x: x.get("v413e_support_score", -1e9), reverse=True)

    # 多抓型配额，避免 top-k 又被一种抓型占满。
    selected = []
    per_type_count = {}
    max_per_type = 3

    for r in scored_sorted:
        if r.get("v413e_decision", "") == "REJECT_SUPPORT_UNDER":
            continue
        gt = str(r.get("grasp_type"))
        if per_type_count.get(gt, 0) >= max_per_type:
            continue
        selected.append(r)
        per_type_count[gt] = per_type_count.get(gt, 0) + 1
        if len(selected) >= args.top_k:
            break

    # 兜底：如果被过滤太多，就加回分数最高的非重复项，但 under 仍放最后。
    if len(selected) < args.top_k:
        existing = {int(r["valid_local_index"]) for r in selected}
        for r in scored_sorted:
            idx = int(r["valid_local_index"])
            if idx in existing:
                continue
            selected.append(r)
            existing.add(idx)
            if len(selected) >= args.top_k:
                break

    save_json(out_dir / "v4_13e_support_aware_summary.json", {
        "format": "v4_13e_support_aware_rerank_debug_v1",
        "selector_json": rel(selector_json),
        "npy": rel(npy_path),
        "mesh": rel(mesh_path),
        "support_top_z": args.support_top_z,
        "object_clearance": args.object_clearance,
        "rows_sorted": scored_sorted,
        "selected": selected,
    })

    # V4.13D 兼容输入
    compact = []
    for i, r in enumerate(selected, start=1):
        rr = dict(r)
        rr["rank"] = i
        rr["final_score"] = r["v413e_support_score"]
        rr["decision"] = r["v413e_decision"]
        compact.append(rr)

    save_json(out_dir / "selected_topk_compact.json", compact)

    (out_dir / "selected_valid_local_indices.txt").write_text(
        "\n".join(str(int(r["valid_local_index"])) for r in compact) + "\n"
    )
    (out_dir / "selected_sample_indices.txt").write_text(
        "\n".join(str(int(r.get("raw_sample_index", r["valid_local_index"]))) for r in compact) + "\n"
    )

    lines = []
    lines.append("========== V4.13E SUPPORT-AWARE RERANK REPORT ==========")
    lines.append(f"selector_json: {rel(selector_json)}")
    lines.append(f"npy          : {rel(npy_path)}")
    lines.append(f"mesh         : {rel(mesh_path)}")
    lines.append(f"support_top : {args.support_top_z}")
    lines.append("")

    lines.append("---- selected ----")
    for i, r in enumerate(compact, start=1):
        geom = r["v413e_geometry"]
        lines.append(
            f"rank={i:02d} local={r.get('valid_local_index')} raw={r.get('raw_sample_index')} "
            f"type={r.get('grasp_type')} score={r.get('v413e_support_score'):.3f} "
            f"base={r.get('v413e_base_selector_score'):.3f} type_score={r.get('v413e_type_score'):.3f} "
            f"decision={r.get('v413e_decision')} "
            f"hand_clear={geom.get('handbase_clearance_to_support_est'):.5f} "
            f"object_top={geom.get('object_top_z'):.5f}"
        )
        for rr in r.get("v413e_reasons", []):
            lines.append(f"  - {rr}")
        lines.append("")

    lines.append("---- rejected / downranked examples ----")
    for r in scored_sorted[:20]:
        if r in compact:
            continue
        geom = r["v413e_geometry"]
        lines.append(
            f"local={r.get('valid_local_index')} raw={r.get('raw_sample_index')} "
            f"type={r.get('grasp_type')} score={r.get('v413e_support_score'):.3f} "
            f"decision={r.get('v413e_decision')} "
            f"hand_clear={geom.get('handbase_clearance_to_support_est'):.5f}"
        )
        for rr in r.get("v413e_reasons", [])[:3]:
            lines.append(f"  - {rr}")
        lines.append("")

    lines.append("输出给 V4.13D 的文件:")
    lines.append(f"{rel(out_dir / 'selected_topk_compact.json')}")
    lines.append("========================================================")
    txt = "\n".join(lines) + "\n"
    (out_dir / "v4_13e_support_aware_report.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
