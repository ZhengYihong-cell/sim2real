#!/usr/bin/env python3
"""
脚本类型：
    debug / locator / visual-mesh-source / generic-builder-precheck

用途：
    定位 V4.13 泛化流程中真实应该使用的 object visual mesh。
    当前问题：
        之前用户确认物体模型显示正常；
        但临时 generic builder 使用 dataset/meshdata/<object>/coacd/decomposed.obj 后，
        物体显示成小短柱/简化碰撞体。
    
    本脚本用于查找：
        1. 工程中是否存在 object_code 对应的原始 model.obj / visual mesh；
        2. 旧 scene / legacy demo / diagnostics 中是否引用过正确 mesh；
        3. 当前 object_code 的 hash 是否在其他目录出现；
        4. 相关 mesh 的 bbox 尺寸，便于确认哪个才是正常视觉模型。

输入：
    object_code
    工程目录 ~/Projects/o7_mujoco_sim

输出：
    diagnostics/current_v413/locate_real_visual_mesh_source_debug/
        visual_mesh_source_report.txt
        visual_mesh_source_summary.json

当前流程位置：
    V4.13 generic builder 发现物体显示异常
        -> 本脚本定位真实 visual mesh 来源
        -> 修正 generic builder 的 mesh 选择逻辑
        -> 再回到 selector / P2 / P3 / viewer

不负责：
    1. 不运行 P2/P3；
    2. 不运行 viewer；
    3. 不修改 builder；
    4. 不做任何姿态微调。
"""

from pathlib import Path
import argparse
import json
import re
import xml.etree.ElementTree as ET

import numpy as np


PROJECT = Path.home() / "Projects/o7_mujoco_sim"
OUT = PROJECT / "diagnostics/current_v413/locate_real_visual_mesh_source_debug"


SEARCH_ROOTS = [
    "dataset",
    "assets",
    "models",
    "diagnostics/current_v412",
    "diagnostics/current_v413",
    "legacy_final_demos",
]


def rel(p):
    p = Path(p)
    try:
        return str(p.relative_to(PROJECT))
    except Exception:
        return str(p)


def read_obj_vertices(path):
    verts = []
    faces = 0
    try:
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
    except Exception as e:
        return None, 0, repr(e)

    if not verts:
        return None, faces, "no vertices"
    return np.asarray(verts, dtype=float), faces, None


def obj_bbox(path):
    verts, faces, err = read_obj_vertices(path)
    if verts is None:
        return {
            "ok": False,
            "error": err,
            "n_vertices": 0,
            "n_faces": faces,
        }
    mn = verts.min(axis=0)
    mx = verts.max(axis=0)
    size = mx - mn
    return {
        "ok": True,
        "n_vertices": int(len(verts)),
        "n_faces": int(faces),
        "bbox_min": mn.tolist(),
        "bbox_max": mx.tolist(),
        "bbox_size_raw": size.tolist(),
        "long_axis": int(np.argmax(size)),
        "long_size_raw": float(np.max(size)),
    }


def file_text_head(path, limit=200000):
    try:
        return path.read_text(errors="ignore")[:limit]
    except Exception:
        return ""


def parse_xml_mesh_refs(path):
    refs = []
    try:
        tree = ET.parse(str(path))
        root = tree.getroot()
    except Exception:
        return refs

    for m in root.findall(".//mesh"):
        refs.append({
            "mesh_name": m.get("name"),
            "file": m.get("file"),
            "scale": m.get("scale"),
        })
    return refs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object-code", required=True)
    ap.add_argument("--out-dir", default=str(OUT))
    args = ap.parse_args()

    object_code = args.object_code
    hash_part = object_code.split("-")[-1]
    category_part = object_code.split("-")[1] if len(object_code.split("-")) > 2 else ""

    out_dir = Path(args.out_dir)
    if not out_dir.is_absolute():
        out_dir = PROJECT / out_dir
    out_dir.mkdir(parents=True, exist_ok=True)

    keywords = [
        object_code,
        hash_part,
        f"{category_part}-{hash_part}" if category_part else hash_part,
        f"{category_part}_{hash_part}" if category_part else hash_part,
    ]
    keywords = [k for k in keywords if k]

    candidate_files = []
    xml_refs = []
    text_hits = []

    for root_s in SEARCH_ROOTS:
        root = PROJECT / root_s
        if not root.exists():
            continue

        # 1. 文件路径中直接包含 hash/object_code 的 obj/stl/dae
        for ext in ["*.obj", "*.stl", "*.dae", "*.ply", "*.urdf", "*.xml", "*.json"]:
            for p in root.rglob(ext):
                ps = str(p)
                if any(k in ps for k in keywords):
                    candidate_files.append(p)

        # 2. XML 中 mesh file 引用包含 hash/object_code
        for p in root.rglob("*.xml"):
            txt = file_text_head(p, limit=200000)
            if any(k in txt for k in keywords):
                refs = parse_xml_mesh_refs(p)
                xml_refs.append({
                    "xml": rel(p),
                    "mesh_refs": refs,
                    "matched_keywords": [k for k in keywords if k in txt],
                })

        # 3. Python / shell / json 中提到该 object
        for ext in ["*.py", "*.sh", "*.json", "*.txt"]:
            for p in root.rglob(ext):
                if p.stat().st_size > 5_000_000:
                    continue
                txt = file_text_head(p, limit=200000)
                if any(k in txt for k in keywords):
                    text_hits.append({
                        "file": rel(p),
                        "matched_keywords": [k for k in keywords if k in txt],
                    })

    # 去重
    unique = []
    seen = set()
    for p in candidate_files:
        rp = str(p.resolve())
        if rp not in seen:
            seen.add(rp)
            unique.append(p)

    mesh_rows = []
    for p in unique:
        suffix = p.suffix.lower()
        row = {
            "path": rel(p),
            "suffix": suffix,
            "size_bytes": p.stat().st_size,
        }
        if suffix == ".obj":
            row.update(obj_bbox(p))
        mesh_rows.append(row)

    # 按“更可能是 visual mesh”排序：非 coacd、文件名 model.obj、尺寸大、顶点多
    def rank_mesh(r):
        path = r["path"].lower()
        score = 0
        if r.get("ok"):
            score += 10
            score += min(r.get("n_vertices", 0), 10000) / 1000
            score += min(r.get("long_size_raw", 0), 5.0) * 5
        if "/coacd/" not in path:
            score += 30
        if path.endswith("/model.obj") or path.endswith("model.obj"):
            score += 40
        if "visual" in path or "textured" in path or "meshes" in path:
            score += 10
        if "decomposed" in path or "convex_piece" in path:
            score -= 20
        return score

    mesh_rows_sorted = sorted(mesh_rows, key=rank_mesh, reverse=True)

    summary = {
        "format": "v4_13_real_visual_mesh_source_locator_debug_v1",
        "object_code": object_code,
        "hash_part": hash_part,
        "keywords": keywords,
        "mesh_candidates_sorted": mesh_rows_sorted,
        "xml_refs": xml_refs[:80],
        "text_hits": text_hits[:120],
    }

    (out_dir / "visual_mesh_source_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    lines = []
    lines.append("========== V4.13 REAL VISUAL MESH SOURCE LOCATOR ==========")
    lines.append(f"object_code: {object_code}")
    lines.append(f"hash_part  : {hash_part}")
    lines.append(f"keywords   : {keywords}")
    lines.append("")

    lines.append("---- mesh/file candidates ----")
    if not mesh_rows_sorted:
        lines.append("NONE")
    for i, r in enumerate(mesh_rows_sorted[:80], start=1):
        lines.append(
            f"[{i:02d}] {r['path']} suffix={r['suffix']} size={r['size_bytes']} "
            f"ok={r.get('ok')} verts={r.get('n_vertices')} faces={r.get('n_faces')} "
            f"bbox_raw={r.get('bbox_size_raw')} long={r.get('long_size_raw')}"
        )
        if r.get("error"):
            lines.append(f"     error={r.get('error')}")
    lines.append("")

    lines.append("---- XML refs containing object/hash ----")
    if not xml_refs:
        lines.append("NONE")
    for x in xml_refs[:40]:
        lines.append(f"xml: {x['xml']} matched={x['matched_keywords']}")
        for m in x["mesh_refs"][:20]:
            lines.append(f"  mesh name={m.get('mesh_name')} file={m.get('file')} scale={m.get('scale')}")
        lines.append("")

    lines.append("---- text hits ----")
    if not text_hits:
        lines.append("NONE")
    for x in text_hits[:80]:
        lines.append(f"{x['file']} matched={x['matched_keywords']}")

    lines.append("")
    lines.append("结论使用：")
    lines.append("1. 如果找到非 coacd 的 model.obj / visual mesh，generic builder 应改用该 visual mesh。")
    lines.append("2. 如果只有 coacd/decomposed.obj，说明当前本地资产目录确实缺 visual mesh。")
    lines.append("3. 如果旧 scene 引用过正确 mesh，后续直接复用旧 scene 的 mesh 路径规则。")
    lines.append("===========================================================")

    report = "\n".join(lines) + "\n"
    (out_dir / "visual_mesh_source_report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
