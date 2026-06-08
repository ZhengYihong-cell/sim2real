#!/usr/bin/env python3
"""
脚本类型：
    debug / locator / pipeline-precheck

用途：
    为 V4.13 通用抓握流程定位当前工程中可复用的 candidate / scene / P2 / P3 脚本。
    当前已有 V4.13 selector 输出 Top-K valid_local_index，
    下一步需要把这些 sample 转成 FR3+O7 candidate 和 scene，再跑轻量 P2/P3。

输入：
    当前工程目录 ~/Projects/o7_mujoco_sim

输出：
    diagnostics/current_v413/locate_v4_13_pipeline_debug/
        locate_report.txt
        locate_summary.json

当前流程位置：
    V4.13 selector
        -> 本脚本定位实际可复用文件
        -> 后续生成 V4.13b topk candidate/scene + light P2/P3 runner

不负责：
    1. 不运行 P2/P3；
    2. 不运行 viewer；
    3. 不修改 legacy_final_demos；
    4. 不修改任何已有脚本。
"""

from pathlib import Path
import json
import os
import re

PROJECT = Path.home() / "Projects/o7_mujoco_sim"
OUT = PROJECT / "diagnostics/current_v413/locate_v4_13_pipeline_debug"
OUT.mkdir(parents=True, exist_ok=True)

SEARCH_DIRS = [
    PROJECT / "scripts",
    PROJECT / "models",
    PROJECT / "legacy_final_demos",
    PROJECT / "diagnostics/current_v412",
    PROJECT / "diagnostics/current_v413",
]

KEYWORDS = {
    "candidate_scene_builder": [
        "candidate", "scene", "dataset", "sample", "build", "initial"
    ],
    "dataset_to_candidate": [
        "dataset", "candidate", "hand_pose", "T_object", "o7_active_ctrl"
    ],
    "p2_ik": [
        "p2", "pinocchio", "multiseed", "ik"
    ],
    "p3_precheck": [
        "p3", "mujoco", "collision", "precheck"
    ],
    "p4u6_runner": [
        "p4u6", "path", "record", "demo"
    ],
    "p4u1_close": [
        "p4u1", "snap", "close"
    ],
}

IMPORTANT_PATTERNS = [
    "*candidate*scene*.py",
    "*dataset*candidate*.py",
    "*bottle*candidate*.py",
    "*initial*candidate*.py",
    "*p2*ik*.py",
    "*p3*collision*.py",
    "*p4u6*.py",
    "*p4u1*.py",
    "*.xml",
    "*.urdf",
]

def rel(p):
    try:
        return str(p.relative_to(PROJECT))
    except Exception:
        return str(p)

def read_head(path, n=12000):
    try:
        return path.read_text(errors="ignore")[:n]
    except Exception:
        return ""

def score_file(path, keys):
    name = path.name.lower()
    text = read_head(path).lower() if path.suffix in [".py", ".sh", ".xml", ".urdf", ".json"] else ""
    score = 0
    hits = []
    for k in keys:
        kk = k.lower()
        if kk in name:
            score += 4
            hits.append(f"name:{k}")
        if kk in text:
            score += 1
            hits.append(f"text:{k}")
    return score, hits

def main():
    all_files = []
    for root in SEARCH_DIRS:
        if not root.exists():
            continue
        for pat in IMPORTANT_PATTERNS:
            all_files.extend(root.rglob(pat))

    # 去重
    seen = set()
    files = []
    for p in all_files:
        if p in seen or not p.is_file():
            continue
        seen.add(p)
        files.append(p)

    result = {}
    for group, keys in KEYWORDS.items():
        rows = []
        for p in files:
            score, hits = score_file(p, keys)
            if score > 0:
                rows.append({
                    "path": rel(p),
                    "score": score,
                    "hits": hits[:20],
                    "size": p.stat().st_size,
                })
        rows.sort(key=lambda x: x["score"], reverse=True)
        result[group] = rows[:20]

    # 额外检查几个必须文件
    must_check = {
        "urdf_candidates": sorted([rel(p) for p in (PROJECT / "models").rglob("*fr3*o7*.urdf")]) if (PROJECT / "models").exists() else [],
        "scene_candidates": sorted([rel(p) for p in (PROJECT / "models").rglob("*bottle*scene*.xml")]) if (PROJECT / "models").exists() else [],
        "current_v413_selector_outputs": sorted([rel(p) for p in (PROJECT / "diagnostics/current_v413").rglob("*select*")]) if (PROJECT / "diagnostics/current_v413").exists() else [],
    }

    summary = {
        "format": "locate_v4_13_pipeline_debug_v1",
        "project": str(PROJECT),
        "groups": result,
        "must_check": must_check,
    }

    (OUT / "locate_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    lines = []
    lines.append("========== LOCATE V4.13 PIPELINE REPORT ==========")
    lines.append(f"project: {PROJECT}")
    lines.append("")

    for group, rows in result.items():
        lines.append(f"---- {group} ----")
        if not rows:
            lines.append("  NONE")
        for r in rows[:10]:
            lines.append(f"  score={r['score']:02d} {r['path']}")
            lines.append(f"    hits={r['hits']}")
        lines.append("")

    lines.append("---- must_check ----")
    for k, vals in must_check.items():
        lines.append(f"{k}:")
        for v in vals[:20]:
            lines.append(f"  {v}")
        if not vals:
            lines.append("  NONE")
        lines.append("")

    lines.append("==================================================")
    txt = "\n".join(lines) + "\n"
    (OUT / "locate_report.txt").write_text(txt)
    print(txt)

if __name__ == "__main__":
    main()
