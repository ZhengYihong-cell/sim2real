#!/usr/bin/env python3
"""
V4.24 grasp-quality filter.

定位：
    对 V4.23 projected_candidates_ranked.json 做质量过滤与重排序。
    不是执行器，不修改抓握位姿，不救单个 sample。

目的：
    V4.23 已经能找到 support-normal projected grasp 并成功 lift，
    但可能出现 push-assisted / edge-assisted 抓握。
    V4.24 增加质量指标，自动选择更干净的 projected candidate。

输入：
    projected_candidates_ranked.json

输出：
    quality_ranked_projected_candidates.json
    best_quality_projected_candidate.json
    clean_projected_candidates.json
    quality_filter_report.txt

核心指标：
    first_ready_step_ratio
    first_ready_disp
    max_close_disp
    exact_setup_disp
    dz
    frozen_count
    object_group_count
    quality_label:
        clean
        acceptable
        push_assisted
        failed
"""

from pathlib import Path
import argparse
import json


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def load_json(path):
    return json.loads(Path(path).read_text())


def rel(path):
    try:
        return str(Path(path).resolve().relative_to(Path.cwd().resolve()))
    except Exception:
        return str(path)


def find_first_ready(row):
    close_rows = row.get("close_rows", []) or []
    for cr in close_rows:
        if cr.get("ready", False):
            return {
                "step": int(cr.get("step", -1)),
                "close_disp": float(cr.get("close_disp", 999.0)),
                "object_groups": cr.get("object_groups", {}),
                "support_groups": cr.get("support_groups", {}),
                "group_alpha": cr.get("group_alpha", {}),
                "frozen": cr.get("frozen", {}),
            }
    return None


def count_true(d):
    if not isinstance(d, dict):
        return 0
    return sum(1 for v in d.values() if bool(v))


def object_group_count(groups):
    if not isinstance(groups, dict):
        return 0
    return len(groups)


def quality_label(metrics, args):
    if not metrics["projection_ready"]:
        return "failed"

    if (
        metrics["projection_success"]
        and metrics["exact_setup_disp"] <= args.clean_setup_disp
        and metrics["first_ready_disp"] <= args.clean_ready_disp
        and metrics["max_close_disp"] <= args.clean_close_disp
        and metrics["first_ready_step_ratio"] <= args.clean_ready_ratio
    ):
        return "clean"

    if (
        metrics["exact_setup_disp"] <= args.accept_setup_disp
        and metrics["first_ready_disp"] <= args.accept_ready_disp
        and metrics["max_close_disp"] <= args.accept_close_disp
        and metrics["first_ready_step_ratio"] <= args.accept_ready_ratio
    ):
        return "acceptable"

    return "push_assisted"


def compute_metrics(row, close_steps_default):
    first = find_first_ready(row)
    projection_ready = bool(row.get("projection_ready", False))
    projection_success = bool(row.get("projection_success", False))

    close_steps = close_steps_default
    if row.get("close_rows"):
        # V4.23 的 close_rows 不是每一步都有日志，所以 close_steps 仍使用命令行传入。
        close_steps = close_steps_default

    if first is None:
        first_ready_step = 999999
        first_ready_ratio = 1.0
        first_ready_disp = 999.0
        first_ready_groups = {}
        first_ready_alpha = {}
    else:
        first_ready_step = int(first["step"])
        first_ready_ratio = float(first_ready_step / max(1, close_steps))
        first_ready_disp = float(first.get("close_disp", 999.0))
        first_ready_groups = first.get("object_groups", {})
        first_ready_alpha = first.get("group_alpha", {})

    frozen = row.get("frozen", {}) or {}
    max_obj_groups = row.get("max_obj_groups", {}) or {}
    final_obj_groups = row.get("final_object_groups", {}) or {}

    metrics = {
        "projection_ready": projection_ready,
        "projection_success": projection_success,
        "dz": float(row.get("dz", 0.0)),
        "exact_setup_disp": float(row.get("exact_setup_disp", 999.0)),
        "max_close_disp": float(row.get("max_close_disp", 999.0)),
        "first_ready_step": first_ready_step,
        "first_ready_step_ratio": first_ready_ratio,
        "first_ready_disp": first_ready_disp,
        "first_ready_object_groups": first_ready_groups,
        "first_ready_group_alpha": first_ready_alpha,
        "max_obj_group_count": object_group_count(max_obj_groups),
        "final_obj_group_count": object_group_count(final_obj_groups),
        "frozen_count": count_true(frozen),
        "selector_score": float(row.get("selector_score", 0.0)),
        "projection_score_old": float(row.get("projection_score", 0.0)),
        "failure_reason_old": row.get("failure_reason", ""),
    }
    return metrics


def compute_quality_score(metrics, label, args):
    score = 0.0

    if metrics["projection_ready"]:
        score += 1000.0
    if metrics["projection_success"]:
        score += 500.0

    if label == "clean":
        score += 800.0
    elif label == "acceptable":
        score += 450.0
    elif label == "push_assisted":
        score += 100.0
    else:
        score -= 1000.0

    score += 120.0 * metrics["max_obj_group_count"]
    score += 50.0 * metrics["final_obj_group_count"]

    # 越早 ready 越好
    score -= args.w_ready_ratio * metrics["first_ready_step_ratio"]

    # ready 时物体位移越小越好
    score -= args.w_ready_disp * max(0.0, metrics["first_ready_disp"] - args.clean_ready_disp)

    # close 全程最大位移越小越好
    score -= args.w_close_disp * max(0.0, metrics["max_close_disp"] - args.clean_close_disp)

    # exact setup 推物体越小越好
    score -= args.w_setup_disp * max(0.0, metrics["exact_setup_disp"] - args.clean_setup_disp)

    # dz 不是越大越好。能不偏离 prior 就不要偏离。
    score -= args.w_dz * metrics["dz"]

    # 冻结太多手指说明依赖支撑/碰撞，扣一点
    score -= args.w_frozen * metrics["frozen_count"]

    # 原 projection score 作为弱 tie-break
    score += args.w_old_score * metrics["projection_score_old"]

    return float(score)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--projected-ranked", required=True)
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--close-steps", type=int, default=420)

    # clean 阈值：更严格
    ap.add_argument("--clean-setup-disp", type=float, default=0.008)
    ap.add_argument("--clean-ready-disp", type=float, default=0.010)
    ap.add_argument("--clean-close-disp", type=float, default=0.016)
    ap.add_argument("--clean-ready-ratio", type=float, default=0.75)

    # acceptable 阈值：能接受，但标记不是最干净
    ap.add_argument("--accept-setup-disp", type=float, default=0.014)
    ap.add_argument("--accept-ready-disp", type=float, default=0.018)
    ap.add_argument("--accept-close-disp", type=float, default=0.026)
    ap.add_argument("--accept-ready-ratio", type=float, default=0.90)

    # 权重
    ap.add_argument("--w-ready-ratio", type=float, default=550.0)
    ap.add_argument("--w-ready-disp", type=float, default=30000.0)
    ap.add_argument("--w-close-disp", type=float, default=22000.0)
    ap.add_argument("--w-setup-disp", type=float, default=25000.0)
    ap.add_argument("--w-dz", type=float, default=350.0)
    ap.add_argument("--w-frozen", type=float, default=20.0)
    ap.add_argument("--w-old-score", type=float, default=0.02)

    args = ap.parse_args()

    projected_path = Path(args.projected_ranked)
    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = load_json(projected_path)
    quality_rows = []

    for row in rows:
        metrics = compute_metrics(row, args.close_steps)
        label = quality_label(metrics, args)
        qscore = compute_quality_score(metrics, label, args)

        r = dict(row)
        r["quality_label"] = label
        r["quality_score"] = qscore
        r["quality_metrics"] = metrics
        r["quality_reason_short"] = (
            f"label={label}; "
            f"ready={metrics['projection_ready']}; "
            f"success={metrics['projection_success']}; "
            f"dz={metrics['dz']:.4f}; "
            f"setup={metrics['exact_setup_disp']:.4f}; "
            f"first_ready_step={metrics['first_ready_step']}; "
            f"ready_ratio={metrics['first_ready_step_ratio']:.3f}; "
            f"ready_disp={metrics['first_ready_disp']:.4f}; "
            f"max_close_disp={metrics['max_close_disp']:.4f}; "
            f"groups={metrics['max_obj_group_count']}; "
            f"frozen={metrics['frozen_count']}"
        )
        quality_rows.append(r)

    quality_rows = sorted(quality_rows, key=lambda x: x.get("quality_score", -1e9), reverse=True)

    clean_rows = [r for r in quality_rows if r.get("quality_label") == "clean"]
    acceptable_rows = [r for r in quality_rows if r.get("quality_label") in ["clean", "acceptable"]]
    ready_rows = [r for r in quality_rows if r.get("projection_ready")]

    if clean_rows:
        best = clean_rows[0]
        best_selection_mode = "best_clean"
    elif acceptable_rows:
        best = acceptable_rows[0]
        best_selection_mode = "best_acceptable"
    elif ready_rows:
        best = ready_rows[0]
        best_selection_mode = "best_ready_push_assisted"
    else:
        best = quality_rows[0] if quality_rows else {}
        best_selection_mode = "best_available_failed"

    if isinstance(best, dict):
        best["quality_selection_mode"] = best_selection_mode

    save_json(out_dir / "quality_ranked_projected_candidates.json", quality_rows)
    save_json(out_dir / "clean_projected_candidates.json", clean_rows)
    save_json(out_dir / "acceptable_projected_candidates.json", acceptable_rows)
    save_json(out_dir / "best_quality_projected_candidate.json", best)

    label_counts = {}
    for r in quality_rows:
        label_counts[r["quality_label"]] = label_counts.get(r["quality_label"], 0) + 1

    lines = []
    lines.append("========== V4.24 GRASP QUALITY FILTER ==========")
    lines.append(f"input : {projected_path}")
    lines.append(f"num   : {len(quality_rows)}")
    lines.append(f"best_selection_mode: {best_selection_mode}")
    lines.append("")
    lines.append("---- label counts ----")
    for k, v in sorted(label_counts.items()):
        lines.append(f"{k}: {v}")

    lines.append("")
    lines.append("---- quality ranked top 20 ----")
    for i, r in enumerate(quality_rows[:20], start=1):
        m = r["quality_metrics"]
        lines.append(
            f"rank={i:02d} "
            f"local={r.get('valid_local_index'):03d} raw={r.get('raw_sample_index'):03d} "
            f"type={r.get('selector_type')} dz={r.get('dz'):+.4f} "
            f"label={r.get('quality_label')} qscore={r.get('quality_score'):.2f} "
            f"ready={m['projection_ready']} success={m['projection_success']} "
            f"setup={m['exact_setup_disp']:.4f} "
            f"ready_step={m['first_ready_step']} ratio={m['first_ready_step_ratio']:.3f} "
            f"ready_disp={m['first_ready_disp']:.4f} "
            f"close_disp={m['max_close_disp']:.4f} "
            f"groups={m['max_obj_group_count']} frozen={m['frozen_count']} "
            f"old_score={m['projection_score_old']:.2f}"
        )

    lines.append("")
    lines.append("---- selected best ----")
    if best:
        m = best.get("quality_metrics", {})
        lines.append(f"local={best.get('valid_local_index')} raw={best.get('raw_sample_index')}")
        lines.append(f"type={best.get('selector_type')}")
        lines.append(f"dz={best.get('dz')}")
        lines.append(f"quality_label={best.get('quality_label')}")
        lines.append(f"quality_score={best.get('quality_score')}")
        lines.append(f"selection_mode={best.get('quality_selection_mode')}")
        lines.append(f"projection_ready={best.get('projection_ready')}")
        lines.append(f"projection_success={best.get('projection_success')}")
        lines.append(f"exact_setup_disp={best.get('exact_setup_disp')}")
        lines.append(f"first_ready_step={m.get('first_ready_step')}")
        lines.append(f"first_ready_step_ratio={m.get('first_ready_step_ratio')}")
        lines.append(f"first_ready_disp={m.get('first_ready_disp')}")
        lines.append(f"max_close_disp={best.get('max_close_disp')}")
        lines.append(f"max_obj_groups={best.get('max_obj_groups')}")
        lines.append(f"frozen={best.get('frozen')}")

    lines.append("")
    lines.append("---- output ----")
    lines.append(f"quality ranked: {out_dir / 'quality_ranked_projected_candidates.json'}")
    lines.append(f"clean candidates: {out_dir / 'clean_projected_candidates.json'}")
    lines.append(f"acceptable candidates: {out_dir / 'acceptable_projected_candidates.json'}")
    lines.append(f"best quality: {out_dir / 'best_quality_projected_candidate.json'}")
    lines.append("================================================")

    report = "\n".join(lines) + "\n"
    (out_dir / "quality_filter_report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
