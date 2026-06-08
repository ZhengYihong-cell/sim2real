#!/usr/bin/env python3
"""
脚本类型：
    debug / diagnostic / target-selector

用途：
    在 dataset/meshdata 和 O7 validate_results 中自动搜索下一个适合泛化验证的 can / SodaCan 目标物体。
    目标是从已有数据集中挑出与已成功 sem-SodaCan-16526 不同的 can 类物体，
    用作第二轮泛化测试。

输入：
    1. --dataset-root
       O7 数据集结果目录。
    2. --mesh-root
       meshdata 目录。
    3. --out-dir
       输出诊断结果目录。

输出：
    1. next_can_target_report.txt
    2. next_can_target_summary.json

当前流程位置：
    第一轮 SodaCan 泛化成功之后：
        自动筛选第二个 can-like 目标
        -> 后续再对选定目标生成 candidate / scene
        -> P2/P3
        -> P4U6/P4U1

不负责：
    1. 不运行 P2/P3；
    2. 不运行 MuJoCo；
    3. 不生成 candidate；
    4. 不修改 legacy_final_demos；
    5. 不覆盖已固化的 can52 或 sodacan165 demo。
"""

from pathlib import Path
import argparse
import json
import math
import numpy as np


PROJECT = Path.home() / "Projects/o7_mujoco_sim"

DEFAULT_DATASET_ROOT = "dataset/O7_Full_V8BestBaseline_165objs_20260422_084834"
DEFAULT_MESH_ROOT = "dataset/meshdata"
DEFAULT_OUT_DIR = "diagnostics/current_v412/next_can_target_select_debug"

ALREADY_DONE = {
    "sem-SodaCan-16526d147e837c386829bf9ee210f5e7",
    "core-can-52e295024593705fb00c487926b62c9",
}


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


def read_obj_bbox(obj_path):
    obj_path = resolve_path(obj_path)
    verts = []
    if not obj_path.exists():
        return {"exists": False, "path": rel(obj_path), "ok": False}

    with open(obj_path, "r", errors="ignore") as f:
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
        return {"exists": True, "path": rel(obj_path), "ok": False, "reason": "no vertices"}

    pts = np.asarray(verts, dtype=float)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    size = mx - mn
    return {
        "exists": True,
        "path": rel(obj_path),
        "ok": True,
        "num_vertices": int(len(pts)),
        "bbox_min": mn.tolist(),
        "bbox_max": mx.tolist(),
        "bbox_size": size.tolist(),
        "bbox_center": (0.5 * (mn + mx)).tolist(),
        "height_z": float(size[2]),
        "diameter_xy": float(max(size[0], size[1])),
        "slender_ratio_z_over_xy": float(size[2] / max(max(size[0], size[1]), 1e-9)),
    }


def find_can_objects(mesh_root):
    mesh_root = resolve_path(mesh_root)
    objects = []
    if not mesh_root.exists():
        return objects

    for d in sorted(mesh_root.iterdir()):
        if not d.is_dir():
            continue
        name = d.name
        low = name.lower()
        if "can" in low or "sodacan" in low:
            objects.append(name)
    return objects


def find_npy_files(dataset_root, target):
    dataset_root = resolve_path(dataset_root)
    files = []
    for pat in [
        f"validate_results/seed*/{target}.npy",
        f"validate_results_friction_mu*/seed*/{target}.npy",
        f"results/seed*/{target}.npy",
    ]:
        files.extend(dataset_root.glob(pat))
    return sorted(set(files))


def summarize_sample_keys(npy_path, preview=1):
    try:
        arr = np.load(npy_path, allow_pickle=True)
    except Exception as e:
        return {"load_ok": False, "error": repr(e)}

    out = {
        "load_ok": True,
        "path": rel(npy_path),
        "num_samples": int(len(arr)),
        "shape": list(getattr(arr, "shape", [])),
        "dtype": str(getattr(arr, "dtype", "")),
    }

    if len(arr) > 0:
        try:
            s = arr[0].item() if hasattr(arr[0], "item") else arr[0]
            if isinstance(s, dict):
                out["sample0_keys"] = list(s.keys())
                if "hand_pose" in s:
                    out["hand_pose_shape"] = list(np.asarray(s["hand_pose"]).shape)
                if "qpos" in s and isinstance(s["qpos"], dict):
                    out["qpos_num_keys"] = len(s["qpos"])
                if "scale" in s:
                    out["scale0"] = float(s["scale"])
        except Exception as e:
            out["sample0_error"] = repr(e)

    return out


def summarize_flags_and_metrics(npy_path, target, n_samples):
    parent = npy_path.parent
    flags_path = parent / f"{target}_validation_flags.npz"
    metrics_path = parent / f"{target}_metrics_valid_per_sample.npz"
    metrics_json = parent / f"{target}_metrics_valid.json"

    out = {
        "flags_path": rel(flags_path),
        "flags_exists": flags_path.exists(),
        "metrics_npz_path": rel(metrics_path),
        "metrics_npz_exists": metrics_path.exists(),
        "metrics_json_path": rel(metrics_json),
        "metrics_json_exists": metrics_json.exists(),
        "score": 0.0,
        "top_indices": [],
        "flag_summary": {},
        "metric_summary": {},
    }

    scores = np.zeros(n_samples, dtype=float)
    evidence = [[] for _ in range(n_samples)]

    if flags_path.exists():
        try:
            f = np.load(flags_path, allow_pickle=True)
            for k in f.keys():
                arr = np.asarray(f[k])
                item = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
                if arr.ndim == 1 and arr.shape[0] == n_samples and arr.dtype == np.bool_:
                    cnt = int(np.count_nonzero(arr))
                    item["num_true"] = cnt
                    item["true_preview"] = np.where(arr)[0][:20].astype(int).tolist()

                    name = k.lower()
                    positive = any(t in name for t in ["valid", "success", "pass", "stable", "lift", "simulated"])
                    negative = any(t in name for t in ["fail", "collision", "penetr", "invalid", "slip", "drop"])

                    if positive and not negative:
                        idxs = np.where(arr)[0]
                        scores[idxs] += 2.0
                        for idx in idxs:
                            evidence[int(idx)].append("+" + k)
                    elif negative:
                        idxs = np.where(arr)[0]
                        scores[idxs] -= 2.0
                        for idx in idxs:
                            evidence[int(idx)].append("-" + k)

                out["flag_summary"][k] = item
        except Exception as e:
            out["flags_error"] = repr(e)

    if metrics_path.exists():
        try:
            m = np.load(metrics_path, allow_pickle=True)
            for k in m.keys():
                arr = np.asarray(m[k])
                item = {"shape": list(arr.shape), "dtype": str(arr.dtype)}
                if arr.ndim == 1 and arr.shape[0] == n_samples and np.issubdtype(arr.dtype, np.number):
                    vals = arr.astype(float)
                    finite = vals[np.isfinite(vals)]
                    if finite.size:
                        item["min"] = float(finite.min())
                        item["max"] = float(finite.max())
                        item["mean"] = float(finite.mean())

                        med = np.nanmedian(vals)
                        std = np.nanstd(vals) + 1e-9
                        norm = np.clip((vals - med) / std, -3.0, 3.0)

                        name = k.lower()
                        if any(t in name for t in ["success", "rise", "lift", "contact", "hand_object", "force"]):
                            scores += 0.2 * norm
                        if any(t in name for t in ["penetr", "collision", "slip", "drop", "fail", "energy", "loss"]):
                            scores -= 0.2 * norm

                out["metric_summary"][k] = item
        except Exception as e:
            out["metrics_error"] = repr(e)

    order = np.argsort(-scores)
    top = []
    for idx in order[:10]:
        top.append({
            "sample_index": int(idx),
            "heuristic_score": float(scores[idx]),
            "evidence": evidence[int(idx)][:12],
        })

    out["score"] = float(scores[order[0]]) if n_samples > 0 else 0.0
    out["top_indices"] = top
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--mesh-root", default=DEFAULT_MESH_ROOT)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    args = ap.parse_args()

    dataset_root = resolve_path(args.dataset_root)
    mesh_root = resolve_path(args.mesh_root)
    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    objects = find_can_objects(mesh_root)

    for obj in objects:
        mesh_path = mesh_root / obj / "coacd/decomposed.obj"
        bbox = read_obj_bbox(mesh_path)
        npy_files = find_npy_files(dataset_root, obj)

        row = {
            "object": obj,
            "already_done": obj in ALREADY_DONE,
            "mesh": bbox,
            "npy_files": [],
            "best_npy": None,
            "object_score": -999.0,
        }

        best_score = -999.0
        best_item = None

        for npy_path in npy_files:
            ns = summarize_sample_keys(npy_path)
            if not ns.get("load_ok"):
                item = {"npy": rel(npy_path), "load_ok": False, "error": ns.get("error")}
            else:
                n = int(ns.get("num_samples", 0))
                fm = summarize_flags_and_metrics(npy_path, obj, n)
                item = {
                    "npy": rel(npy_path),
                    "load_ok": True,
                    "sample_summary": ns,
                    "flags_metrics": fm,
                }

                score = fm.get("score", 0.0)
                # validate_results/seed1/2/3 优先于 raw results
                p = str(npy_path)
                if "/validate_results/seed1/" in p:
                    score += 0.3
                elif "/validate_results/seed2/" in p:
                    score += 0.2
                elif "/validate_results/seed3/" in p:
                    score += 0.1

                # 已做过的不推荐
                if obj in ALREADY_DONE:
                    score -= 100.0

                # mesh 太离谱的先降权，但不删除
                if bbox.get("ok"):
                    ratio = bbox.get("slender_ratio_z_over_xy", 0.0)
                    if 1.0 <= ratio <= 2.5:
                        score += 0.5
                    else:
                        score -= 0.5

                item["selection_score"] = float(score)

                if score > best_score:
                    best_score = score
                    best_item = item

            row["npy_files"].append(item)

        if best_item is not None:
            row["best_npy"] = best_item
            row["object_score"] = float(best_score)

        rows.append(row)

    rows_sorted = sorted(rows, key=lambda r: r.get("object_score", -999), reverse=True)

    summary = {
        "format": "next_can_target_select_debug_v1",
        "dataset_root": rel(dataset_root),
        "mesh_root": rel(mesh_root),
        "out_dir": rel(out_dir),
        "already_done": sorted(ALREADY_DONE),
        "rows_sorted": rows_sorted,
    }

    (out_dir / "next_can_target_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    lines = []
    lines.append("========== NEXT CAN GENERALIZATION TARGET REPORT ==========")
    lines.append(f"dataset_root: {rel(dataset_root)}")
    lines.append(f"mesh_root   : {rel(mesh_root)}")
    lines.append(f"num objects : {len(rows_sorted)}")
    lines.append("")

    for i, r in enumerate(rows_sorted[:20]):
        obj = r["object"]
        mesh = r.get("mesh", {})
        best = r.get("best_npy")
        lines.append(f"[{i:02d}] object={obj}")
        lines.append(f"     already_done={r.get('already_done')} object_score={r.get('object_score')}")
        if mesh.get("ok"):
            lines.append(
                f"     bbox_size={mesh.get('bbox_size')} "
                f"height_z={mesh.get('height_z'):.6f} "
                f"diameter_xy={mesh.get('diameter_xy'):.6f} "
                f"ratio={mesh.get('slender_ratio_z_over_xy'):.3f}"
            )
        else:
            lines.append(f"     mesh_ok=False path={mesh.get('path')}")

        if best:
            ss = best.get("sample_summary", {})
            fm = best.get("flags_metrics", {})
            lines.append(f"     best_npy={best.get('npy')}")
            lines.append(f"     num_samples={ss.get('num_samples')} selection_score={best.get('selection_score')}")
            lines.append("     top sample indices:")
            for t in fm.get("top_indices", [])[:8]:
                lines.append(
                    f"       sample={t['sample_index']:03d} "
                    f"score={t['heuristic_score']:.3f} "
                    f"evidence={t['evidence']}"
                )
        else:
            lines.append("     best_npy=None")

        lines.append("")

    lines.append("---- 推荐 ----")
    recommended = None
    for r in rows_sorted:
        if not r.get("already_done") and r.get("best_npy"):
            recommended = r
            break

    if recommended is None:
        lines.append("没有找到未完成且带 npy 的 can-like 目标。")
    else:
        best = recommended["best_npy"]
        fm = best["flags_metrics"]
        top = fm.get("top_indices", [])
        lines.append(f"next_object: {recommended['object']}")
        lines.append(f"next_npy   : {best['npy']}")
        if top:
            lines.append("next_samples: " + " ".join(str(x["sample_index"]) for x in top[:10]))

    lines.append("==========================================================")

    report = "\n".join(lines) + "\n"
    (out_dir / "next_can_target_report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
