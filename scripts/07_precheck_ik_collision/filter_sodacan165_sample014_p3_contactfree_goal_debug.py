#!/usr/bin/env python3
"""
脚本类型：
    debug / diagnostic / p3-plan-filter

用途：
    从 sem-SodaCan-16526 sample014 的 P3 结果中，筛选一个更适合 P4U6 路径规划的 PASS 组合。
    具体目标是避免 P4U6 在 pre_to_grasp 阶段因为 q_grasp 已经接触而报：
        goal invalid: {'reason': 'contact', 'min_distance': -1.0, 'pair': None}

输入：
    1. --p3-json
       原始 sample014_p3.json。
    2. --out
       输出过滤后的 p3 json。
    3. --min-go
       要求 static_grasp_closed_hand_object_distance 至少大于该值。
       这代表 q_grasp 闭手状态与物体之间留一点点正间隙，便于 P4U6 无接触规划到目标。
    4. --max-go
       最大允许间隙，避免离物体太远。
    5. --min-hsc
       q_grasp_closed 对支撑物的最小 clearance。

输出：
    1. 过滤后的 p3 json；
    2. filter_report.txt；
    3. 终端打印选择结果。

当前流程位置：
    P3 已完成且有很多 PASS
        -> 过滤出适合 P4U6 的 contact-free q_grasp
        -> P4U6 路径规划
        -> P4U1 ready-gated snap close

不负责：
    1. 不重新运行 P2；
    2. 不重新运行 P3；
    3. 不修改 P4U6/P4U1 源码；
    4. 不修改 legacy demo；
    5. 不保证最终动态抓取成功，只解决 P4U6 目标点已有接触导致的路径规划拒绝问题。
"""

from pathlib import Path
import argparse
import copy
import json
import math


PROJECT = Path.home() / "Projects/o7_mujoco_sim"


def resolve_path(p):
    p = Path(str(p)).expanduser()
    if not p.is_absolute():
        p = PROJECT / p
    return p


def is_number(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def walk_records(obj, path="root"):
    records = []

    if isinstance(obj, dict):
        status = obj.get("precheck_status")
        has_q = any(k in obj for k in ["q_pre", "q_grasp", "q_lift"])
        has_score = "score" in obj

        if status is not None or has_q or has_score:
            rec = dict(obj)
            rec["_debug_path"] = path
            records.append(rec)

        for k, v in obj.items():
            records.extend(walk_records(v, f"{path}.{k}"))

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            records.extend(walk_records(v, f"{path}[{i}]"))

    return records


def get_float(d, key, default=None):
    v = d.get(key, default)
    if is_number(v):
        return float(v)
    return default


def record_ok(r, args):
    if r.get("precheck_status") != "PASS_PRECHECK":
        return False

    go = get_float(r, "static_grasp_closed_hand_object_distance")
    hsc = get_float(r, "static_grasp_closed_hand_support_clearance")
    hs = get_float(r, "min_path_hand_support_clearance")
    fo = get_float(r, "min_path_fr3_object_clearance")

    if go is None:
        return False
    if go < args.min_go or go > args.max_go:
        return False

    if hsc is not None and hsc < args.min_hsc:
        return False
    if hs is not None and hs < args.min_path_hs:
        return False
    if fo is not None and fo < args.min_path_fo:
        return False

    return True


def sort_key(r):
    score = get_float(r, "score", 1e99)
    go = get_float(r, "static_grasp_closed_hand_object_distance", 1e99)
    hsc = get_float(r, "static_grasp_closed_hand_support_clearance", 0.0)
    margin = get_float(r, "combo_min_joint_margin", 0.0)

    # 优先低 score，其次 q_grasp 与物体间隙不要太大，再其次支撑 clearance 和关节 margin 更大。
    return (score, abs(go - 0.003), -hsc, -margin)


def brief(r):
    keys = [
        "_debug_path",
        "precheck_status",
        "score",
        "pre_seed",
        "grasp_seed",
        "lift_seed",
        "min_path_hand_support_clearance",
        "min_path_fr3_object_clearance",
        "static_grasp_closed_hand_object_distance",
        "static_grasp_closed_hand_support_clearance",
        "combo_min_joint_margin",
        "smooth_cost",
    ]
    return {k: r.get(k) for k in keys if k in r}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--p3-json",
        default="diagnostics/current_v412/sodacan165_topk_p2p3_batch_debug/sample014/sample014_p3.json",
    )
    ap.add_argument(
        "--out",
        default="diagnostics/current_v412/sodacan165_sample014_p4u6_viewer_debug/sample014_p3_contactfree_goal.json",
    )
    ap.add_argument("--min-go", type=float, default=0.0020)
    ap.add_argument("--max-go", type=float, default=0.0100)
    ap.add_argument("--min-hsc", type=float, default=0.0)
    ap.add_argument("--min-path-hs", type=float, default=0.0)
    ap.add_argument("--min-path-fo", type=float, default=0.0)
    args = ap.parse_args()

    p3_path = resolve_path(args.p3_json)
    out_path = resolve_path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)

    data = json.loads(p3_path.read_text())
    records = walk_records(data)

    pass_records = [r for r in records if r.get("precheck_status") == "PASS_PRECHECK"]
    candidates = [r for r in pass_records if record_ok(r, args)]
    candidates = sorted(candidates, key=sort_key)

    report_lines = []
    report_lines.append("========== FILTER P3 CONTACT-FREE GOAL REPORT ==========")
    report_lines.append(f"input p3     : {p3_path}")
    report_lines.append(f"output p3    : {out_path}")
    report_lines.append(f"num records  : {len(records)}")
    report_lines.append(f"num pass     : {len(pass_records)}")
    report_lines.append(f"num selected : {len(candidates)}")
    report_lines.append(f"criteria     : min_go={args.min_go}, max_go={args.max_go}, min_hsc={args.min_hsc}")
    report_lines.append("")

    if not candidates:
        report_lines.append("[ERROR] no contact-free PASS candidate found.")
        report_lines.append("Top PASS records preview:")
        for r in sorted(pass_records, key=lambda x: get_float(x, "score", 1e99))[:20]:
            report_lines.append(json.dumps(brief(r), indent=2, ensure_ascii=False))
        txt = "\n".join(report_lines) + "\n"
        (out_path.parent / "filter_report.txt").write_text(txt)
        print(txt)
        raise SystemExit(2)

    selected = copy.deepcopy(candidates[0])
    selected.pop("_debug_path", None)

    report_lines.append("---- selected ----")
    report_lines.append(json.dumps(brief(candidates[0]), indent=2, ensure_ascii=False))
    report_lines.append("")
    report_lines.append("---- top selected preview ----")
    for r in candidates[:15]:
        report_lines.append(json.dumps(brief(r), indent=2, ensure_ascii=False))

    new_data = copy.deepcopy(data)
    new_data["best_pass"] = selected
    new_data["best_available"] = selected
    new_data["filter_note"] = {
        "type": "contactfree_goal_for_p4u6_debug",
        "reason": "P4U6 pre_to_grasp planner rejects q_grasp if already in contact.",
        "source_p3_json": str(p3_path),
        "criteria": {
            "min_go": args.min_go,
            "max_go": args.max_go,
            "min_hsc": args.min_hsc,
            "min_path_hs": args.min_path_hs,
            "min_path_fo": args.min_path_fo,
        },
        "selected_record_debug": brief(candidates[0]),
    }

    out_path.write_text(json.dumps(new_data, indent=2, ensure_ascii=False))
    txt = "\n".join(report_lines) + "\n"
    (out_path.parent / "filter_report.txt").write_text(txt)

    print(txt)


if __name__ == "__main__":
    main()
