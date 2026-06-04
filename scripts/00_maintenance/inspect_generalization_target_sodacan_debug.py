#!/usr/bin/env python3
"""
脚本类型：
    debug / diagnostic / generalization-precheck

用途：
    对第一轮泛化目标物体做只读体检。
    默认目标为：
        sem-SodaCan-16526d147e837c386829bf9ee210f5e7

输入：
    1. --target
       数据集物体完整代码。
    2. --dataset-root
       O7 数据集结果目录。
    3. --mesh-root
       meshdata 根目录。
    4. --out-dir
       输出诊断结果目录。

输出：
    1. inspect_summary.json
       机器可读诊断摘要。
    2. inspect_report.txt
       人可读诊断报告。
    3. 终端打印关键结果。

当前流程位置：
    泛化流程最前置体检：
        object mesh / dataset npy / metrics / flags
        -> 判断能否进入 candidate 生成、P4E/P4H/P4H2、P2/P3、P4U6/P4U1。

不负责：
    1. 不修改任何 XML；
    2. 不生成 candidate；
    3. 不运行 IK；
    4. 不运行 MuJoCo viewer；
    5. 不修改 legacy_final_demos；
    6. 不替换 can52 成功 demo。
"""

from pathlib import Path
import argparse
import json
import math
import traceback
import numpy as np


PROJECT = Path.home() / "Projects/o7_mujoco_sim"

DEFAULT_TARGET = "sem-SodaCan-16526d147e837c386829bf9ee210f5e7"
DEFAULT_DATASET_ROOT = "dataset/O7_Full_V8BestBaseline_165objs_20260422_084834"
DEFAULT_MESH_ROOT = "dataset/meshdata"
DEFAULT_OUT_DIR = "diagnostics/current_v412/sodacan165_generalization_inspect_debug"


def resolve_path(p):
    p = Path(str(p)).expanduser()
    if not p.is_absolute():
        p = PROJECT / p
    return p


def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


def save_json(path, obj):
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_jsonable(obj), f, indent=2, ensure_ascii=False)


def write_text(path, text):
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


def rel(path):
    path = resolve_path(path)
    try:
        return str(path.relative_to(PROJECT))
    except Exception:
        return str(path)


def read_obj_bbox(obj_path, max_vertices=None):
    obj_path = resolve_path(obj_path)
    verts = []
    n_vertex_lines = 0

    with open(obj_path, "r", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            n_vertex_lines += 1
            if max_vertices is not None and len(verts) >= int(max_vertices):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except Exception:
                continue

    if not verts:
        return {
            "exists": obj_path.exists(),
            "path": rel(obj_path),
            "num_vertex_lines": n_vertex_lines,
            "num_loaded_vertices": 0,
            "ok": False,
            "reason": "no vertices parsed",
        }

    pts = np.asarray(verts, dtype=float)
    mn = pts.min(axis=0)
    mx = pts.max(axis=0)
    size = mx - mn
    center = 0.5 * (mn + mx)

    return {
        "exists": obj_path.exists(),
        "path": rel(obj_path),
        "num_vertex_lines": n_vertex_lines,
        "num_loaded_vertices": int(len(pts)),
        "ok": True,
        "bbox_min": mn,
        "bbox_max": mx,
        "bbox_size": size,
        "bbox_center": center,
        "height_z": float(size[2]),
        "radius_xy_bbox": float(0.5 * max(size[0], size[1])),
        "diameter_x": float(size[0]),
        "diameter_y": float(size[1]),
    }


def safe_load_json(path):
    path = resolve_path(path)
    if not path.exists():
        return None, {"exists": False, "path": rel(path)}
    try:
        with open(path, "r") as f:
            obj = json.load(f)
        return obj, {"exists": True, "path": rel(path), "load_ok": True}
    except Exception as e:
        return None, {
            "exists": True,
            "path": rel(path),
            "load_ok": False,
            "error": repr(e),
        }


def safe_np_load(path):
    path = resolve_path(path)
    try:
        return np.load(path, allow_pickle=True), None
    except Exception as e:
        return None, repr(e)


def as_sample(x):
    try:
        if hasattr(x, "item"):
            y = x.item()
            if isinstance(y, dict):
                return y
    except Exception:
        pass
    if isinstance(x, dict):
        return x
    return x


def short_value_desc(v):
    if isinstance(v, dict):
        return {
            "type": "dict",
            "num_keys": len(v),
            "keys_preview": list(v.keys())[:20],
        }
    if isinstance(v, (list, tuple)):
        return {
            "type": type(v).__name__,
            "len": len(v),
        }
    if isinstance(v, np.ndarray):
        return {
            "type": "ndarray",
            "shape": list(v.shape),
            "dtype": str(v.dtype),
        }
    return {
        "type": type(v).__name__,
        "repr": repr(v)[:120],
    }


def summarize_sample(sample):
    if not isinstance(sample, dict):
        return {
            "type": type(sample).__name__,
            "repr": repr(sample)[:300],
        }

    out = {
        "type": "dict",
        "keys": list(sample.keys()),
        "fields": {},
    }

    for k in sample.keys():
        out["fields"][k] = short_value_desc(sample[k])

    hp = sample.get("hand_pose", None)
    if hp is not None:
        hp_arr = np.asarray(hp)
        out["hand_pose_shape"] = list(hp_arr.shape)
        if hp_arr.size > 0:
            out["hand_pose_first_values"] = hp_arr.reshape(-1)[:16].astype(float).tolist()

    qpos = sample.get("qpos", None)
    if isinstance(qpos, dict):
        out["qpos_num_keys"] = len(qpos)
        out["qpos_keys_preview"] = list(qpos.keys())[:30]

    if "scale" in sample:
        try:
            out["scale"] = float(sample["scale"])
        except Exception:
            out["scale_repr"] = repr(sample["scale"])

    return out


def summarize_npy(path, preview_samples=3):
    path = resolve_path(path)
    arr, err = safe_np_load(path)
    if err is not None:
        return {
            "path": rel(path),
            "exists": path.exists(),
            "load_ok": False,
            "error": err,
        }

    summary = {
        "path": rel(path),
        "exists": True,
        "load_ok": True,
        "type": type(arr).__name__,
        "shape": list(getattr(arr, "shape", [])),
        "dtype": str(getattr(arr, "dtype", "")),
    }

    try:
        n = len(arr)
    except Exception:
        n = 0

    summary["num_samples"] = int(n)

    sample_summaries = []
    scales = []
    hp_shapes = {}
    qpos_key_counts = []
    qpos_key_union = set()

    for i in range(n):
        s = as_sample(arr[i])
        if i < int(preview_samples):
            sample_summaries.append({
                "index": i,
                "summary": summarize_sample(s),
            })

        if isinstance(s, dict):
            if "scale" in s:
                try:
                    scales.append(float(s["scale"]))
                except Exception:
                    pass

            hp = s.get("hand_pose", None)
            if hp is not None:
                shp = tuple(np.asarray(hp).shape)
                hp_shapes[str(shp)] = hp_shapes.get(str(shp), 0) + 1

            qpos = s.get("qpos", None)
            if isinstance(qpos, dict):
                qpos_key_counts.append(len(qpos))
                for k in qpos.keys():
                    qpos_key_union.add(str(k))

    if scales:
        scales_arr = np.asarray(scales, dtype=float)
        summary["scale_stats"] = {
            "count": int(len(scales_arr)),
            "min": float(scales_arr.min()),
            "max": float(scales_arr.max()),
            "mean": float(scales_arr.mean()),
            "unique_preview": sorted([float(x) for x in set(np.round(scales_arr, 8))])[:20],
        }

    summary["hand_pose_shapes"] = hp_shapes

    if qpos_key_counts:
        qkc = np.asarray(qpos_key_counts, dtype=float)
        summary["qpos_key_count_stats"] = {
            "count": int(len(qkc)),
            "min": int(qkc.min()),
            "max": int(qkc.max()),
            "mean": float(qkc.mean()),
        }
        summary["qpos_key_union_preview"] = sorted(qpos_key_union)[:80]

    summary["sample_preview"] = sample_summaries

    return summary


def find_npy_files(dataset_root, target):
    dataset_root = resolve_path(dataset_root)
    patterns = [
        f"validate_results/seed*/{target}.npy",
        f"validate_results_friction_mu*/seed*/{target}.npy",
        f"validate_results_perturb*/seed*/{target}.npy",
        f"results/seed*/{target}.npy",
    ]

    files = []
    for pat in patterns:
        files.extend(dataset_root.glob(pat))

    # 备用旧目录
    files.extend((PROJECT / "dataset/meshdata/__unused__20260105/dexgraspnet").glob(f"{target}.npy"))

    # 去重保序
    out = []
    seen = set()
    for f in sorted(files):
        key = str(f.resolve())
        if key in seen:
            continue
        seen.add(key)
        out.append(f)
    return out


def candidate_sidecar_paths(npy_path, target):
    p = resolve_path(npy_path)
    parent = p.parent
    return {
        "metrics_valid_json": parent / f"{target}_metrics_valid.json",
        "metrics_raw_json": parent / f"{target}_metrics_raw.json",
        "metrics_valid_per_sample_npz": parent / f"{target}_metrics_valid_per_sample.npz",
        "metrics_raw_per_sample_npz": parent / f"{target}_metrics_raw_per_sample.npz",
        "validation_flags_npz": parent / f"{target}_validation_flags.npz",
        "validation_meta_json": parent / f"{target}_validation_meta.json",
    }


def collect_numeric_lists(obj, n, prefix=""):
    out = []
    if isinstance(obj, dict):
        for k, v in obj.items():
            p = f"{prefix}.{k}" if prefix else str(k)
            out.extend(collect_numeric_lists(v, n, p))
    elif isinstance(obj, list):
        if len(obj) == n:
            vals = []
            ok = True
            for x in obj:
                if isinstance(x, bool):
                    vals.append(float(x))
                elif isinstance(x, (int, float)) and math.isfinite(float(x)):
                    vals.append(float(x))
                else:
                    ok = False
                    break
            if ok:
                arr = np.asarray(vals, dtype=float)
                out.append({
                    "path": prefix,
                    "len": int(len(arr)),
                    "min": float(arr.min()),
                    "max": float(arr.max()),
                    "mean": float(arr.mean()),
                    "num_nonzero": int(np.count_nonzero(arr)),
                    "first_values": arr[:10].tolist(),
                })
        for i, v in enumerate(obj[:5]):
            p = f"{prefix}[{i}]"
            out.extend(collect_numeric_lists(v, n, p))
    return out


def summarize_json_file(path, n_samples):
    obj, info = safe_load_json(path)
    if obj is None:
        return info

    out = dict(info)
    out["top_type"] = type(obj).__name__

    if isinstance(obj, dict):
        out["top_keys"] = list(obj.keys())[:80]
        out["num_top_keys"] = len(obj)

    numeric_lists = collect_numeric_lists(obj, n_samples)
    out["numeric_lists_len_eq_num_samples"] = numeric_lists[:80]

    return out


def summarize_npz_file(path, n_samples):
    path = resolve_path(path)
    if not path.exists():
        return {"exists": False, "path": rel(path)}

    try:
        data = np.load(path, allow_pickle=True)
    except Exception as e:
        return {
            "exists": True,
            "path": rel(path),
            "load_ok": False,
            "error": repr(e),
        }

    out = {
        "exists": True,
        "path": rel(path),
        "load_ok": True,
        "keys": list(data.keys()),
        "arrays": {},
    }

    for k in data.keys():
        arr = np.asarray(data[k])
        item = {
            "shape": list(arr.shape),
            "dtype": str(arr.dtype),
        }

        if arr.ndim == 1 and arr.shape[0] == n_samples:
            if arr.dtype == np.bool_:
                item["num_true"] = int(np.count_nonzero(arr))
                item["num_false"] = int(arr.size - np.count_nonzero(arr))
                item["true_indices_preview"] = np.where(arr)[0][:30].astype(int).tolist()
            elif np.issubdtype(arr.dtype, np.number):
                vals = arr.astype(float)
                finite = vals[np.isfinite(vals)]
                if finite.size:
                    item["min"] = float(finite.min())
                    item["max"] = float(finite.max())
                    item["mean"] = float(finite.mean())
                    item["first_values"] = vals[:10].tolist()
        elif arr.ndim > 0:
            item["first_values_flat_preview"] = arr.reshape(-1)[:10].tolist()

        out["arrays"][k] = item

    return out


def heuristic_rank_from_sidecars(sidecars, n_samples):
    """
    只做启发式排名，不作为最终结论。
    目的是帮助快速选几个 sample 进入下一步静态/可视化验证。
    """
    score = np.zeros(n_samples, dtype=float)
    evidence = [[] for _ in range(n_samples)]

    # flags 中名字像 positive 的 bool 加分，名字像 negative 的 bool 扣分。
    flags_path = sidecars.get("validation_flags_npz")
    if flags_path is not None and resolve_path(flags_path).exists():
        try:
            npz = np.load(resolve_path(flags_path), allow_pickle=True)
            for k in npz.keys():
                arr = np.asarray(npz[k])
                if arr.ndim != 1 or arr.shape[0] != n_samples or arr.dtype != np.bool_:
                    continue

                name = k.lower()
                positive = any(t in name for t in ["valid", "success", "pass", "stable", "lift"])
                negative = any(t in name for t in ["fail", "collision", "penetr", "invalid", "slip", "drop"])

                if positive and not negative:
                    idxs = np.where(arr)[0]
                    score[idxs] += 1.0
                    for i in idxs[:5000]:
                        evidence[int(i)].append(f"+{k}")
                elif negative:
                    idxs = np.where(arr)[0]
                    score[idxs] -= 1.0
                    for i in idxs[:5000]:
                        evidence[int(i)].append(f"-{k}")
        except Exception:
            pass

    # per_sample npz 中常见数值名做轻量启发。
    for key in ["metrics_valid_per_sample_npz", "metrics_raw_per_sample_npz"]:
        p = sidecars.get(key)
        if p is None or not resolve_path(p).exists():
            continue
        try:
            npz = np.load(resolve_path(p), allow_pickle=True)
            for k in npz.keys():
                arr = np.asarray(npz[k])
                if arr.ndim != 1 or arr.shape[0] != n_samples or not np.issubdtype(arr.dtype, np.number):
                    continue

                vals = arr.astype(float)
                finite = np.isfinite(vals)
                if not np.any(finite):
                    continue

                name = k.lower()
                v = vals.copy()
                med = np.nanmedian(v[finite])
                std = np.nanstd(v[finite]) + 1e-9
                norm = np.zeros_like(v)
                norm[finite] = (v[finite] - med) / std
                norm = np.clip(norm, -3.0, 3.0)

                if any(t in name for t in ["success", "rise", "lift", "contact", "hand_object", "force"]):
                    score += 0.10 * norm
                if any(t in name for t in ["penetr", "collision", "slip", "drop", "fail", "energy", "loss"]):
                    score -= 0.10 * norm
        except Exception:
            pass

    order = np.argsort(-score)
    top = []
    for idx in order[:20]:
        top.append({
            "sample_index": int(idx),
            "heuristic_score": float(score[idx]),
            "evidence_preview": evidence[int(idx)][:20],
        })
    return top


def inspect_one_npy(npy_path, target, preview_samples):
    summary = summarize_npy(npy_path, preview_samples=preview_samples)
    n = int(summary.get("num_samples", 0))

    sidecars = candidate_sidecar_paths(npy_path, target)
    sidecar_summary = {}

    for k, p in sidecars.items():
        if p.suffix == ".json":
            sidecar_summary[k] = summarize_json_file(p, n)
        elif p.suffix == ".npz":
            sidecar_summary[k] = summarize_npz_file(p, n)
        else:
            sidecar_summary[k] = {"path": rel(p), "exists": p.exists()}

    htop = heuristic_rank_from_sidecars(sidecars, n) if n > 0 else []

    summary["sidecars"] = sidecar_summary
    summary["heuristic_top_sample_indices"] = htop
    summary["note"] = (
        "heuristic_top_sample_indices 只是根据 flags/metrics 名称做的启发式排序，"
        "不是最终候选结论。后续仍需做 candidate 生成、P4E/P4H/P4H2、P2/P3 和动态验证。"
    )
    return summary


def choose_recommended_npy(npy_summaries):
    """
    简单推荐优先入口：
    1. validate_results/seed1
    2. validate_results/seed2
    3. validate_results/seed3
    4. 其他
    """
    if not npy_summaries:
        return None

    def key(item):
        p = item.get("path", "")
        if "/validate_results/seed1/" in p:
            return 0
        if "/validate_results/seed2/" in p:
            return 1
        if "/validate_results/seed3/" in p:
            return 2
        if "/validate_results_friction_mu" in p:
            return 3
        if "/results/seed" in p:
            return 4
        return 9

    arr = sorted(npy_summaries, key=key)
    return arr[0]


def make_report(summary):
    lines = []
    lines.append("========== GENERALIZATION TARGET INSPECT REPORT ==========")
    lines.append(f"target      : {summary['target']}")
    lines.append(f"project     : {summary['project']}")
    lines.append(f"mesh_root   : {summary['mesh_root']}")
    lines.append(f"dataset_root: {summary['dataset_root']}")
    lines.append("")

    lines.append("---- MESH ----")
    mesh = summary["mesh"]
    lines.append(f"mesh path      : {mesh.get('path')}")
    lines.append(f"mesh ok        : {mesh.get('ok')}")
    if mesh.get("ok"):
        lines.append(f"bbox size xyz  : {mesh.get('bbox_size')}")
        lines.append(f"bbox center    : {mesh.get('bbox_center')}")
        lines.append(f"height_z       : {mesh.get('height_z')}")
        lines.append(f"radius_xy_bbox : {mesh.get('radius_xy_bbox')}")
        lines.append(f"num vertices   : {mesh.get('num_vertex_lines')}")
    else:
        lines.append(f"mesh reason    : {mesh.get('reason')}")
    lines.append(f"convex pieces  : {summary.get('num_convex_pieces')}")
    lines.append("")

    lines.append("---- NPY FILES ----")
    lines.append(f"num npy files: {len(summary['npy_files'])}")
    for i, item in enumerate(summary["npy_files"]):
        lines.append("")
        lines.append(f"[{i}] {item.get('path')}")
        lines.append(f"    load_ok     : {item.get('load_ok')}")
        lines.append(f"    shape/dtype : {item.get('shape')} / {item.get('dtype')}")
        lines.append(f"    num_samples : {item.get('num_samples')}")
        lines.append(f"    scale_stats : {item.get('scale_stats')}")
        lines.append(f"    hand_pose_shapes : {item.get('hand_pose_shapes')}")
        lines.append(f"    qpos_key_count_stats : {item.get('qpos_key_count_stats')}")

        htop = item.get("heuristic_top_sample_indices", [])
        if htop:
            lines.append("    heuristic top sample indices:")
            for x in htop[:10]:
                lines.append(
                    f"      sample={x['sample_index']:4d} "
                    f"score={x['heuristic_score']:.4f} "
                    f"evidence={x['evidence_preview'][:6]}"
                )

        sidecars = item.get("sidecars", {})
        for sk in [
            "metrics_valid_json",
            "metrics_valid_per_sample_npz",
            "validation_flags_npz",
            "validation_meta_json",
        ]:
            s = sidecars.get(sk, {})
            lines.append(f"    {sk}: exists={s.get('exists')} path={s.get('path')}")
            if sk.endswith("_npz") and s.get("exists") and s.get("load_ok"):
                keys = s.get("keys", [])
                lines.append(f"        keys preview: {keys[:20]}")
            if sk.endswith("_json") and s.get("exists") and s.get("load_ok"):
                lines.append(f"        top keys preview: {s.get('top_keys', [])[:20]}")

    rec = summary.get("recommended_npy")
    lines.append("")
    lines.append("---- RECOMMENDED START POINT ----")
    if rec:
        lines.append(f"recommended npy: {rec.get('path')}")
        htop = rec.get("heuristic_top_sample_indices", [])
        if htop:
            lines.append("recommended first sample indices to inspect:")
            lines.append(", ".join(str(x["sample_index"]) for x in htop[:10]))
        else:
            lines.append("no heuristic sample ranking found; use metrics/flags details above.")
    else:
        lines.append("no npy recommended because no npy file was loaded.")

    lines.append("")
    lines.append("---- NEXT STEP ----")
    lines.append("把本报告发回后，下一步再生成 candidate / scene / P2-P3 / P4U6 的实际运行脚本。")
    lines.append("=========================================================")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=DEFAULT_TARGET)
    ap.add_argument("--dataset-root", default=DEFAULT_DATASET_ROOT)
    ap.add_argument("--mesh-root", default=DEFAULT_MESH_ROOT)
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)
    ap.add_argument("--preview-samples", type=int, default=3)
    args = ap.parse_args()

    out_dir = resolve_path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    target = args.target
    mesh_root = resolve_path(args.mesh_root)
    dataset_root = resolve_path(args.dataset_root)

    mesh_dir = mesh_root / target / "coacd"
    mesh_path = mesh_dir / "decomposed.obj"
    convex_pieces = sorted(mesh_dir.glob("coacd_convex_piece_*.obj"))

    summary = {
        "format": "generalization_target_inspect_debug_v1",
        "target": target,
        "project": str(PROJECT),
        "mesh_root": rel(mesh_root),
        "dataset_root": rel(dataset_root),
        "out_dir": rel(out_dir),
        "mesh": {},
        "num_convex_pieces": len(convex_pieces),
        "convex_piece_preview": [rel(p) for p in convex_pieces[:20]],
        "npy_files": [],
        "recommended_npy": None,
    }

    try:
        summary["mesh"] = read_obj_bbox(mesh_path)
    except Exception as e:
        summary["mesh"] = {
            "path": rel(mesh_path),
            "exists": mesh_path.exists(),
            "ok": False,
            "error": repr(e),
            "traceback": traceback.format_exc(),
        }

    npy_files = find_npy_files(dataset_root, target)

    for p in npy_files:
        try:
            item = inspect_one_npy(p, target, preview_samples=args.preview_samples)
        except Exception as e:
            item = {
                "path": rel(p),
                "exists": p.exists(),
                "load_ok": False,
                "error": repr(e),
                "traceback": traceback.format_exc(),
            }
        summary["npy_files"].append(item)

    summary["recommended_npy"] = choose_recommended_npy(summary["npy_files"])

    json_path = out_dir / "inspect_summary.json"
    txt_path = out_dir / "inspect_report.txt"

    save_json(json_path, summary)
    report = make_report(summary)
    write_text(txt_path, report)

    print(report)
    print("saved json:", rel(json_path))
    print("saved txt :", rel(txt_path))


if __name__ == "__main__":
    main()
