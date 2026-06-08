#!/usr/bin/env python3
"""
脚本类型：
    debug / diagnostic / object-mesh-integrity / visual-preview

用途：
    检查 V4.13 泛化流程中某个 object_code 的 mesh 是否真实符合物体外观。
    当前 core-bottle 在 generic builder 中显示不像 bottle，本脚本用于区分：
        1. decomposed.obj 本身是否只是简化碰撞代理；
        2. coacd_convex_piece_*.obj 组合后是否更像目标物体；
        3. 当前 sample scale 缩放后尺寸是否合理；
        4. mesh 文件目录中是否存在其他可用 visual mesh。

输入：
    object_code
    object npy
    valid local sample index
    mesh root

输出：
    out_dir/
        mesh_variant_report.txt
        mesh_variant_summary.json
        mesh_variant_preview.xml

当前流程位置：
    V4.13 selector 已能读五文件先验
        -> 本脚本检查 object mesh / scale / visual proxy 是否正确
        -> 确认无误后再写 generic builder v2
        -> 后续再 P2/P3 / viewer

不负责：
    1. 不跑 P2/P3；
    2. 不跑 grasp viewer；
    3. 不修改 builder；
    4. 不做任何抓握姿态微调；
    5. 不把某个物体写死。
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


def load_sample_scale(npy_path, sample_index):
    arr = np.load(npy_path, allow_pickle=True)
    if sample_index < 0 or sample_index >= len(arr):
        raise RuntimeError(f"sample_index out of range: {sample_index}, n={len(arr)}")
    sample = arr[sample_index].item() if hasattr(arr[sample_index], "item") else arr[sample_index]
    if not isinstance(sample, dict):
        raise RuntimeError(f"sample is not dict: {type(sample)}")
    return float(sample.get("scale", 1.0)), sample


def read_obj_vertices(path):
    verts = []
    faces = 0
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if line.startswith("v "):
                parts = line.strip().split()
                if len(parts) >= 4:
                    try:
                        verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
                    except Exception:
                        pass
            elif line.startswith("f "):
                faces += 1
    verts = np.asarray(verts, dtype=float)
    if verts.size == 0:
        return None, faces
    return verts, faces


def bbox_info_from_vertices(verts, scale):
    v = np.asarray(verts, dtype=float) * float(scale)
    mn = v.min(axis=0)
    mx = v.max(axis=0)
    size = mx - mn
    center = 0.5 * (mn + mx)
    long_axis = int(np.argmax(size))
    return {
        "bbox_min": mn.tolist(),
        "bbox_max": mx.tolist(),
        "bbox_size": size.tolist(),
        "bbox_center": center.tolist(),
        "long_axis": long_axis,
        "long_axis_size": float(size[long_axis]),
    }


def collect_variants(object_dir):
    coacd_dir = object_dir / "coacd"

    variants = []

    decomposed = coacd_dir / "decomposed.obj"
    if decomposed.exists():
        variants.append({
            "name": "coacd_decomposed_single",
            "mode": "single",
            "files": [decomposed],
        })

    pieces = sorted(coacd_dir.glob("coacd_convex_piece_*.obj"))
    if pieces:
        variants.append({
            "name": "coacd_convex_pieces_combined",
            "mode": "multi",
            "files": pieces,
        })
        for p in pieces[:8]:
            variants.append({
                "name": f"single_{p.stem}",
                "mode": "single",
                "files": [p],
            })

    # 查找可能存在的 visual/original mesh。排除 coacd 中已经加入的重复项。
    all_objs = sorted(object_dir.rglob("*.obj"))
    seen = {p.resolve() for v in variants for p in v["files"]}
    extra = []
    for p in all_objs:
        if p.resolve() in seen:
            continue
        low = str(p).lower()
        if "__unused__" in low:
            continue
        extra.append(p)

    for p in extra[:12]:
        variants.append({
            "name": f"extra_{p.stem}",
            "mode": "single",
            "files": [p],
        })

    return variants


def summarize_variant(v, scale):
    all_verts = []
    file_infos = []

    for f in v["files"]:
        verts, faces = read_obj_vertices(f)
        if verts is None:
            file_infos.append({
                "file": rel(f),
                "ok": False,
                "n_vertices": 0,
                "n_faces": faces,
            })
            continue

        info = bbox_info_from_vertices(verts, scale)
        info.update({
            "file": rel(f),
            "ok": True,
            "n_vertices": int(len(verts)),
            "n_faces": int(faces),
        })
        file_infos.append(info)
        all_verts.append(verts)

    if all_verts:
        combined = np.vstack(all_verts)
        combined_info = bbox_info_from_vertices(combined, scale)
        ok = True
    else:
        combined_info = {}
        ok = False

    return {
        "name": v["name"],
        "mode": v["mode"],
        "files": [rel(f) for f in v["files"]],
        "ok": ok,
        "combined": combined_info,
        "file_infos": file_infos,
    }


def make_preview_xml(out_xml, variant_summaries, scale, support_top_z=0.23, clearance=0.003):
    root = ET.Element("mujoco", model="v413_mesh_variant_preview")

    ET.SubElement(root, "compiler", angle="radian", autolimits="true")
    ET.SubElement(root, "option", timestep="0.002", gravity="0 0 -9.81")

    asset = ET.SubElement(root, "asset")
    ET.SubElement(asset, "material", name="orange", rgba="0.95 0.45 0.10 1")
    ET.SubElement(asset, "material", name="blue", rgba="0.10 0.32 0.58 0.65")
    ET.SubElement(asset, "material", name="dark", rgba="0.25 0.28 0.32 1")

    world = ET.SubElement(root, "worldbody")
    ET.SubElement(world, "light", pos="0 -1.5 2.5", dir="0 1 -1", diffuse="1 1 1")
    ET.SubElement(world, "camera", name="overview", pos="0 -1.0 0.8", xyaxes="1 0 0 0 0.5 1")

    ET.SubElement(
        world,
        "geom",
        name="world_plane",
        type="plane",
        pos="0 0 0",
        size="1.5 1.5 0.02",
        material="dark",
    )

    usable = [v for v in variant_summaries if v["ok"]]
    spacing = 0.22
    start_x = -0.5 * spacing * max(0, len(usable) - 1)

    for i, v in enumerate(usable):
        x = start_x + i * spacing
        name = v["name"]

        # 只预览前 8 个，避免太挤
        if i >= 8:
            break

        support_half = [0.055, 0.055, support_top_z * 0.5]
        ET.SubElement(
            world,
            "geom",
            name=f"pedestal_{i}_{name}",
            type="box",
            pos=f"{x} 0 {support_top_z * 0.5}",
            size=f"{support_half[0]} {support_half[1]} {support_half[2]}",
            material="blue",
        )

        bbox_min = np.asarray(v["combined"]["bbox_min"], dtype=float)
        obj_z = support_top_z + clearance - float(bbox_min[2])

        body = ET.SubElement(
            world,
            "body",
            name=f"body_{i}_{name}",
            pos=f"{x} 0 {obj_z}",
        )

        for j, f in enumerate(v["files"]):
            mesh_name = f"mesh_{i}_{j}_{name}"
            ET.SubElement(
                asset,
                "mesh",
                name=mesh_name,
                file=str(Path(f).resolve()),
                scale=f"{scale:.12g} {scale:.12g} {scale:.12g}",
            )
            ET.SubElement(
                body,
                "geom",
                name=f"geom_{i}_{j}_{name}",
                type="mesh",
                mesh=mesh_name,
                material="orange",
                contype="0",
                conaffinity="0",
            )

    out_xml.parent.mkdir(parents=True, exist_ok=True)
    ET.ElementTree(root).write(out_xml, encoding="utf-8", xml_declaration=True)

    # 编译验证
    model = mujoco.MjModel.from_xml_path(str(out_xml))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    return {
        "preview_xml": rel(out_xml),
        "compiled_ok": True,
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "shown_variants": [v["name"] for v in usable[:8]],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object-code", required=True)
    ap.add_argument("--npy", required=True)
    ap.add_argument("--sample-index", type=int, required=True)
    ap.add_argument("--mesh-root", default="dataset/meshdata")
    ap.add_argument("--out-dir", required=True)
    args = ap.parse_args()

    npy = resolve(args.npy)
    mesh_root = resolve(args.mesh_root)
    object_dir = mesh_root / args.object_code
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    if not object_dir.exists():
        raise FileNotFoundError(object_dir)

    scale, sample = load_sample_scale(npy, args.sample_index)

    variants = collect_variants(object_dir)
    summaries = [summarize_variant(v, scale) for v in variants]

    preview_info = make_preview_xml(out_dir / "mesh_variant_preview.xml", summaries, scale)

    log_path = object_dir / "coacd/decomposed_log.txt"
    log_head = ""
    if log_path.exists():
        log_head = log_path.read_text(errors="ignore")[:3000]

    final = {
        "format": "v4_13_object_mesh_variant_inspect_debug_v1",
        "object_code": args.object_code,
        "object_dir": rel(object_dir),
        "npy": rel(npy),
        "sample_index_valid_local": args.sample_index,
        "sample_scale": scale,
        "variants": summaries,
        "preview": preview_info,
        "coacd_log_head": log_head,
    }

    save_json(out_dir / "mesh_variant_summary.json", final)

    lines = []
    lines.append("========== V4.13 OBJECT MESH VARIANT INSPECT ==========")
    lines.append(f"object_code : {args.object_code}")
    lines.append(f"object_dir  : {rel(object_dir)}")
    lines.append(f"npy         : {rel(npy)}")
    lines.append(f"sample_idx  : {args.sample_index}")
    lines.append(f"sample_scale: {scale}")
    lines.append("")
    lines.append("---- variants ----")

    for i, v in enumerate(summaries):
        c = v.get("combined", {})
        lines.append(
            f"[{i:02d}] {v['name']} mode={v['mode']} ok={v['ok']} "
            f"files={len(v['files'])}"
        )
        if v["ok"]:
            lines.append(f"     bbox_size_scaled={c.get('bbox_size')}")
            lines.append(f"     bbox_min_scaled ={c.get('bbox_min')}")
            lines.append(f"     bbox_max_scaled ={c.get('bbox_max')}")
            lines.append(f"     long_axis={c.get('long_axis')} long_axis_size={c.get('long_axis_size')}")
        for f in v["files"]:
            lines.append(f"       file: {rel(f)}")
        lines.append("")

    lines.append("---- preview ----")
    lines.append(f"preview_xml: {preview_info['preview_xml']}")
    lines.append(f"shown      : {preview_info['shown_variants']}")
    lines.append("")
    lines.append("结论判断：")
    lines.append("1. 如果 coacd_convex_pieces_combined 比 coacd_decomposed_single 更像物体，generic builder 应改成加载 piece 组合。")
    lines.append("2. 如果所有 variant 都不像 bottle，说明这个 object_code 的 mesh 本身就不是你期望的 bottle 外形，应该换物体或找 visual mesh。")
    lines.append("3. 这里不做任何抓握姿态微调，只确认物体资产是否正确。")
    lines.append("========================================================")

    report = "\n".join(lines) + "\n"
    (out_dir / "mesh_variant_report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
