#!/usr/bin/env python3
"""
脚本类型：
    debug / v4.13 / asset-quality-gate / object-selection

用途：
    在通用抓握 selector 前，对数据集物体资产做质量筛选。
    当前 core-bottle-3b0e... 虽然标签是 bottle，但可用 COACD mesh 显示只是小短柱，
    不适合作为“瓶子泛化 demo”继续测试。
    
    本脚本执行：
        1. 扫描 validate_results 中的 object .npy；
        2. 读取 meshdata 下对应 COACD mesh；
        3. 根据 sample scale 计算真实缩放后 bbox；
        4. 计算尺寸、长宽比、valid sample 数、是否存在 visual/original mesh；
        5. 给出 asset quality 排名；
        6. 生成 Top-N preview XML，方便快速挑选下一个物体。

输入：
    dataset/O7_Full_V8BestBaseline_165objs_20260422_084834/validate_results/seed1
    dataset/meshdata

输出：
    diagnostics/current_v413/object_asset_quality_audit_debug/
        asset_quality_report.txt
        asset_quality_summary.json
        asset_quality_preview.xml

当前流程位置：
    物体选择 / 随机物体输入
        -> asset quality gate
        -> V4.13 selector
        -> generic builder
        -> P2/P3
        -> viewer

不负责：
    1. 不运行 P2/P3；
    2. 不运行抓握 viewer；
    3. 不做任何沿轴微调；
    4. 不修改 legacy demo；
    5. 不修改 P4U1/P4U6。
"""

from pathlib import Path
import argparse
import json
import math
import xml.etree.ElementTree as ET

import numpy as np
import mujoco


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


def read_obj_vertices(path):
    verts = []
    faces = 0
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                ps = line.strip().split()
                if len(ps) >= 4:
                    try:
                        verts.append([float(ps[1]), float(ps[2]), float(ps[3])])
                    except Exception:
                        pass
            elif line.startswith("f "):
                faces += 1
    if not verts:
        return None, faces
    return np.asarray(verts, dtype=float), faces


def bbox_info(verts, scale):
    vv = np.asarray(verts, dtype=float) * float(scale)
    mn = vv.min(axis=0)
    mx = vv.max(axis=0)
    size = mx - mn
    sorted_size = sorted([float(x) for x in size], reverse=True)
    long_axis = int(np.argmax(size))
    long_size = sorted_size[0]
    mid_size = sorted_size[1]
    short_size = sorted_size[2]
    aspect_long_short = long_size / max(short_size, 1e-9)
    aspect_long_mid = long_size / max(mid_size, 1e-9)
    return {
        "bbox_min": mn.tolist(),
        "bbox_max": mx.tolist(),
        "bbox_size": size.tolist(),
        "long_axis": long_axis,
        "long_size": long_size,
        "mid_size": mid_size,
        "short_size": short_size,
        "aspect_long_short": aspect_long_short,
        "aspect_long_mid": aspect_long_mid,
    }


def load_scales(npy_path):
    arr = np.load(npy_path, allow_pickle=True)
    scales = []
    for i in range(len(arr)):
        x = arr[i].item() if hasattr(arr[i], "item") else arr[i]
        if isinstance(x, dict) and "scale" in x:
            try:
                scales.append(float(x["scale"]))
            except Exception:
                pass
    if not scales:
        scales = [1.0]
    return {
        "n_samples": int(len(arr)),
        "scale_min": float(np.min(scales)),
        "scale_max": float(np.max(scales)),
        "scale_mean": float(np.mean(scales)),
        "scale_median": float(np.median(scales)),
        "first_scale": float(scales[0]),
    }


def read_valid_count(metrics_path):
    if not metrics_path.exists():
        return None, {}
    try:
        d = json.loads(metrics_path.read_text())
        return d.get("n_samples"), d
    except Exception:
        return None, {}


def object_category(object_code):
    parts = object_code.split("-")
    if len(parts) >= 2:
        return parts[1]
    return "unknown"


def find_mesh_files(object_dir):
    coacd = object_dir / "coacd"
    decomposed = coacd / "decomposed.obj"
    pieces = sorted(coacd.glob("coacd_convex_piece_*.obj"))

    all_objs = sorted(object_dir.rglob("*.obj"))
    visual_like = []
    for p in all_objs:
        low = str(p).lower()
        if "/coacd/" in low or "\\coacd\\" in low:
            continue
        if "__unused__" in low:
            continue
        visual_like.append(p)

    return {
        "decomposed": decomposed if decomposed.exists() else None,
        "pieces": pieces,
        "visual_like": visual_like,
        "all_objs": all_objs,
    }


def score_asset(row, category_filter):
    score = 0.0
    reasons = []

    if not row.get("mesh_ok"):
        row["decision"] = "REJECT_NO_MESH"
        row["asset_score"] = -1e9
        row["asset_reasons"] = ["missing mesh"]
        return row

    b = row["bbox_scaled"]
    long_size = b["long_size"]
    mid_size = b["mid_size"]
    short_size = b["short_size"]
    aspect_ls = b["aspect_long_short"]
    aspect_lm = b["aspect_long_mid"]

    # 尺寸优先：太小的物体不适合当前 FR3+O7 + pedestal demo。
    if long_size < 0.08:
        score -= 80
        reasons.append(f"too_small_for_demo_long={long_size:.4f}")
    elif long_size <= 0.18:
        score += 35
        reasons.append(f"good_size_long={long_size:.4f}")
    elif long_size <= 0.28:
        score += 20
        reasons.append(f"large_but_ok_long={long_size:.4f}")
    else:
        score -= 20
        reasons.append(f"too_large_long={long_size:.4f}")

    # bottle 类最好有明显长轴。
    cat = row.get("category", "")
    if category_filter == "bottle" or cat == "bottle":
        if aspect_ls >= 2.4:
            score += 35
            reasons.append(f"bottle_like_aspect_good={aspect_ls:.2f}")
        elif aspect_ls >= 1.8:
            score += 10
            reasons.append(f"bottle_like_aspect_weak={aspect_ls:.2f}")
        else:
            score -= 35
            reasons.append(f"not_bottle_like_aspect={aspect_ls:.2f}")
    else:
        if aspect_ls >= 1.3:
            score += 10
            reasons.append(f"has_shape_anisotropy={aspect_ls:.2f}")

    # 不要太薄或太小，避免看起来像碎片。
    if mid_size < 0.025 or short_size < 0.020:
        score -= 25
        reasons.append(f"too_thin_mid={mid_size:.4f}_short={short_size:.4f}")

    # valid 样本越多越好，但不能压过几何质量。
    nvalid = row.get("valid_n_samples")
    if isinstance(nvalid, int):
        score += min(nvalid, 80) * 0.25
        reasons.append(f"valid_samples={nvalid}")

    if row.get("n_pieces", 0) >= 2:
        score += 5
        reasons.append(f"coacd_pieces={row.get('n_pieces')}")

    if row.get("has_visual_like"):
        score += 20
        reasons.append("has_visual_like_obj")
    else:
        reasons.append("no_visual_like_obj_only_coacd")

    if score >= 35:
        decision = "GOOD_FOR_NEXT_TEST"
    elif score >= 0:
        decision = "MAYBE"
    else:
        decision = "SKIP_FOR_VISUAL_DEMO"

    row["asset_score"] = float(score)
    row["asset_reasons"] = reasons
    row["decision"] = decision
    return row


def make_preview_xml(rows, out_xml, top_n):
    root = ET.Element("mujoco", model="v413_asset_quality_preview")
    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", gravity="0 0 -9.81")

    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "material", name="orange", rgba="0.95 0.45 0.10 1")
    ET.SubElement(asset, "material", name="blue", rgba="0.10 0.32 0.58 0.65")
    ET.SubElement(asset, "material", name="dark", rgba="0.25 0.28 0.32 1")

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 -1.8 2.5", dir="0 1 -1", diffuse="1 1 1")
    ET.SubElement(world, "camera", name="overview", pos="0 -1.4 0.9", xyaxes="1 0 0 0 0.45 1")

    ET.SubElement(
        world,
        "geom",
        name="world_plane",
        type="plane",
        pos="0 0 0",
        size="2 2 0.02",
        material="dark",
    )

    shown = [r for r in rows if r.get("mesh_ok")][:top_n]
    spacing = 0.18
    start_x = -0.5 * spacing * max(0, len(shown) - 1)
    support_top_z = 0.23
    clearance = 0.003

    for i, r in enumerate(shown):
        x = start_x + i * spacing

        ET.SubElement(
            world,
            "geom",
            name=f"pedestal_{i}",
            type="box",
            pos=f"{x} 0 {support_top_z * 0.5}",
            size="0.055 0.055 0.115",
            material="blue",
        )

        mesh_name = f"mesh_{i}"
        mesh_path = Path(r["mesh_path"]).resolve()
        scale = r["scale_used"]

        ET.SubElement(
            asset,
            "mesh",
            name=mesh_name,
            file=str(mesh_path),
            scale=f"{scale:.12g} {scale:.12g} {scale:.12g}",
        )

        mn = np.asarray(r["bbox_scaled"]["bbox_min"], dtype=float)
        obj_z = support_top_z + clearance - float(mn[2])

        body = ET.SubElement(
            world,
            "body",
            name=f"body_{i}_{r['object_code']}",
            pos=f"{x} 0 {obj_z}",
        )

        ET.SubElement(
            body,
            "geom",
            name=f"geom_{i}_{r['object_code']}",
            type="mesh",
            mesh=mesh_name,
            material="orange",
            contype="0",
            conaffinity="0",
        )

    out_xml.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(out_xml, encoding="utf-8", xml_declaration=True)

    model = mujoco.MjModel.from_xml_path(str(out_xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    return {
        "preview_xml": rel(out_xml),
        "compiled_ok": True,
        "shown": [r["object_code"] for r in shown],
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--validate-dir", default="dataset/O7_Full_V8BestBaseline_165objs_20260422_084834/validate_results/seed1")
    ap.add_argument("--mesh-root", default="dataset/meshdata")
    ap.add_argument("--category", default="bottle", help="例如 bottle/can/jar；空字符串表示所有类别")
    ap.add_argument("--out-dir", default="diagnostics/current_v413/object_asset_quality_audit_debug")
    ap.add_argument("--top-n-preview", type=int, default=8)
    args = ap.parse_args()

    validate_dir = resolve(args.validate_dir)
    mesh_root = resolve(args.mesh_root)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if args.category.strip():
        npys = sorted(validate_dir.glob(f"core-{args.category.strip()}-*.npy"))
    else:
        npys = sorted(validate_dir.glob("*.npy"))

    rows = []

    for npy in npys:
        object_code = npy.stem
        category = object_category(object_code)
        object_dir = mesh_root / object_code
        metrics_path = validate_dir / f"{object_code}_metrics_valid.json"

        row = {
            "object_code": object_code,
            "category": category,
            "npy": rel(npy),
            "object_dir": rel(object_dir),
            "mesh_ok": False,
        }

        try:
            scales = load_scales(npy)
            row.update(scales)

            nvalid, metrics = read_valid_count(metrics_path)
            row["valid_n_samples"] = nvalid
            row["metrics_path"] = rel(metrics_path) if metrics_path.exists() else None

            mesh_files = find_mesh_files(object_dir)
            row["n_pieces"] = len(mesh_files["pieces"])
            row["has_visual_like"] = bool(mesh_files["visual_like"])
            row["visual_like_files"] = [rel(p) for p in mesh_files["visual_like"][:10]]

            mesh_path = mesh_files["decomposed"]
            if mesh_path is None:
                row["error"] = "missing coacd/decomposed.obj"
                rows.append(score_asset(row, args.category))
                continue

            verts, faces = read_obj_vertices(mesh_path)
            if verts is None:
                row["error"] = "mesh has no vertices"
                rows.append(score_asset(row, args.category))
                continue

            scale = row["scale_median"]
            row["mesh_ok"] = True
            row["mesh_path"] = rel(mesh_path)
            row["mesh_faces"] = int(faces)
            row["mesh_vertices"] = int(len(verts))
            row["scale_used"] = float(scale)
            row["bbox_scaled"] = bbox_info(verts, scale)

        except Exception as e:
            row["error"] = repr(e)

        rows.append(score_asset(row, args.category))

    rows_sorted = sorted(rows, key=lambda r: r.get("asset_score", -1e9), reverse=True)

    preview = make_preview_xml(rows_sorted, out_dir / "asset_quality_preview.xml", args.top_n_preview)

    summary = {
        "format": "v4_13_object_asset_quality_audit_debug_v1",
        "validate_dir": rel(validate_dir),
        "mesh_root": rel(mesh_root),
        "category": args.category,
        "num_objects": len(rows_sorted),
        "preview": preview,
        "rows_sorted": rows_sorted,
    }
    save_json(out_dir / "asset_quality_summary.json", summary)

    lines = []
    lines.append("========== V4.13 OBJECT ASSET QUALITY AUDIT ==========")
    lines.append(f"validate_dir: {rel(validate_dir)}")
    lines.append(f"mesh_root   : {rel(mesh_root)}")
    lines.append(f"category    : {args.category}")
    lines.append(f"num_objects : {len(rows_sorted)}")
    lines.append(f"preview_xml : {preview['preview_xml']}")
    lines.append("")

    for i, r in enumerate(rows_sorted, start=1):
        b = r.get("bbox_scaled", {})
        lines.append(
            f"rank={i:02d} score={r.get('asset_score'):.2f} decision={r.get('decision')} "
            f"object={r.get('object_code')} valid={r.get('valid_n_samples')} "
            f"scale={r.get('scale_used')} size={b.get('bbox_size')} "
            f"long={b.get('long_size')} aspect={b.get('aspect_long_short')} "
            f"pieces={r.get('n_pieces')} visual={r.get('has_visual_like')} "
            f"err={r.get('error')}"
        )
        for rr in r.get("asset_reasons", []):
            lines.append(f"  - {rr}")
        lines.append(f"  mesh: {r.get('mesh_path')}")
        lines.append("")

    lines.append("---- 使用建议 ----")
    lines.append("1. 先从 decision=GOOD_FOR_NEXT_TEST 的物体里继续 V4.13 selector。")
    lines.append("2. decision=SKIP_FOR_VISUAL_DEMO 的物体不是不能抓，而是不适合作为当前 demo 物体。")
    lines.append("3. 这里没有做任何抓握姿态微调，只做资产质量门控。")
    lines.append("======================================================")

    report = "\n".join(lines) + "\n"
    (out_dir / "asset_quality_report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
