#!/usr/bin/env python3
"""
文件名：
    run_v4_12p3a_audit_distance_and_margin_debug.py

脚本类别：
    debug / diagnostic / precheck-audit

用途：
    本脚本用于审计 V4.12P3 输出结果中的距离判据是否可靠。
    重点检查 mj_geomDistance 返回的 distance=0.0 是否真的表示碰撞，还是 fromto 两点其实有明显距离。

输入：
    1. V4.12P3 输出的 JSON。
       例如 diagnostics/current_v412/v4_12p3_can52_ik_collision_combo_debug.json

输出：
    1. 对每个组合重新计算 effective distance。
    2. 统计在修正 distance 后，有多少组合满足几何 clearance。
    3. 分别统计严格关节余量和放宽关节余量下的通过情况。
    4. 输出新的审计 JSON 和终端摘要。

当前流程位置：
    V4.12P3 组合预检结果
        -> P3A 距离/关节余量审计
        -> 决定是否修正 P3 距离函数，或回到 P2 强化关节限位约束

本脚本不负责：
    1. 不重新跑 Pinocchio IK。
    2. 不重新跑 MuJoCo FK。
    3. 不启动 viewer。
    4. 不生成最终 runner plan。
"""

from pathlib import Path
import argparse
import json
import numpy as np


PROJECT = Path.home() / "Projects/o7_mujoco_sim"


def resolve_path(p):
    p = Path(str(p)).expanduser()
    if not p.is_absolute():
        p = PROJECT / p
    return p


def load_json(p):
    with open(resolve_path(p), "r") as f:
        return json.load(f)


def save_json(p, obj):
    p = resolve_path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)


def fromto_norm(item):
    ft = np.asarray(item.get("fromto", [0, 0, 0, 0, 0, 0]), dtype=float)
    if ft.shape[0] != 6:
        return 0.0
    return float(np.linalg.norm(ft[:3] - ft[3:]))


def effective_distance(item, zero_eps=1e-9, fromto_eps=1e-6):
    """
    修正规则：
    1. 如果 mj_geomDistance 返回非零，先用原值。
    2. 如果返回 0，但 fromto 两点距离明显大于 0，则使用 fromto 两点距离。
    3. 如果返回 0 且 fromto 也近似 0，才保留 0。
    """
    d = float(item.get("distance", 999.0))
    n = fromto_norm(item)

    if abs(d) <= zero_eps and n > fromto_eps:
        return n

    return d


def get_path_metric(path, key):
    return effective_distance(path[key])


def get_static_metric(static_state, key):
    return effective_distance(static_state[key])


def audit_one_combo(r, args):
    paths = r.get("path_precheck", []) or []
    static = r.get("static_precheck", {}) or {}
    q_grasp_closed = static.get("q_grasp_closed", {}) or {}

    min_hs = 999.0
    min_fo = 999.0
    min_ho = 999.0

    bad_reasons = []
    original_zero_suspicious = []

    for p in paths:
        pname = p.get("path", "")

        hs_raw = float((p.get("min_hand_support") or {}).get("distance", 999.0))
        fo_raw = float((p.get("min_fr3_object") or {}).get("distance", 999.0))
        ho_raw = float((p.get("min_hand_object") or {}).get("distance", 999.0))

        hs_eff = get_path_metric(p, "min_hand_support")
        fo_eff = get_path_metric(p, "min_fr3_object")
        ho_eff = get_path_metric(p, "min_hand_object")

        min_hs = min(min_hs, hs_eff)
        min_fo = min(min_fo, fo_eff)
        min_ho = min(min_ho, ho_eff)

        for label, raw, eff in [
            ("hand-support", hs_raw, hs_eff),
            ("fr3-object", fo_raw, fo_eff),
            ("hand-object", ho_raw, ho_eff),
        ]:
            if abs(raw) <= 1e-9 and eff > args.suspicious_fromto_threshold:
                original_zero_suspicious.append(
                    f"{pname}: {label} raw=0 but effective={eff:.5f}"
                )

        if hs_eff < args.min_hand_support_clearance:
            bad_reasons.append(
                f"{pname}: effective hand-support {hs_eff:.5f} < {args.min_hand_support_clearance:.5f}"
            )

        if fo_eff < args.min_fr3_object_clearance:
            bad_reasons.append(
                f"{pname}: effective fr3-object {fo_eff:.5f} < {args.min_fr3_object_clearance:.5f}"
            )

    if q_grasp_closed:
        go_eff = get_static_metric(q_grasp_closed, "min_hand_object")
        gs_eff = get_static_metric(q_grasp_closed, "min_hand_support")
        gf_eff = get_static_metric(q_grasp_closed, "min_fr3_object")
    else:
        go_eff = 999.0
        gs_eff = 999.0
        gf_eff = 999.0

    if go_eff > args.max_grasp_hand_object_distance:
        bad_reasons.append(
            f"q_grasp_closed: effective hand-object {go_eff:.5f} > {args.max_grasp_hand_object_distance:.5f}"
        )

    if gs_eff < args.min_hand_support_clearance:
        bad_reasons.append(
            f"q_grasp_closed: effective hand-support {gs_eff:.5f} < {args.min_hand_support_clearance:.5f}"
        )

    if gf_eff < args.min_fr3_object_clearance:
        bad_reasons.append(
            f"q_grasp_closed: effective fr3-object {gf_eff:.5f} < {args.min_fr3_object_clearance:.5f}"
        )

    margin = float(r.get("combo_min_joint_margin", 0.0))
    if margin < args.min_joint_margin:
        bad_reasons.append(
            f"combo joint margin {margin:.6f} < {args.min_joint_margin:.6f}"
        )

    if margin < args.loose_joint_margin:
        loose_bad = list(bad_reasons)
    else:
        loose_bad = [x for x in bad_reasons if not x.startswith("combo joint margin")]

    geom_bad = [x for x in bad_reasons if not x.startswith("combo joint margin")]

    return {
        "combo_id": r.get("combo_id"),
        "original_status": r.get("precheck_status"),
        "pre_seed": r.get("pre_seed"),
        "grasp_seed": r.get("grasp_seed"),
        "lift_seed": r.get("lift_seed"),
        "original_score": r.get("score"),
        "effective_min_path_hand_support": min_hs,
        "effective_min_path_fr3_object": min_fo,
        "effective_min_path_hand_object": min_ho,
        "effective_static_grasp_hand_object": go_eff,
        "effective_static_grasp_hand_support": gs_eff,
        "effective_static_grasp_fr3_object": gf_eff,
        "combo_min_joint_margin": margin,
        "geometry_pass": len(geom_bad) == 0,
        "strict_pass": len(bad_reasons) == 0,
        "loose_margin_pass": len(loose_bad) == 0,
        "bad_reasons": bad_reasons,
        "geometry_bad_reasons": geom_bad,
        "original_zero_suspicious": original_zero_suspicious,
        "q_pre": r.get("q_pre"),
        "q_grasp": r.get("q_grasp"),
        "q_lift": r.get("q_lift"),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--p3-json", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--min-hand-support-clearance", type=float, default=0.005)
    ap.add_argument("--min-fr3-object-clearance", type=float, default=0.005)
    ap.add_argument("--max-grasp-hand-object-distance", type=float, default=0.030)
    ap.add_argument("--min-joint-margin", type=float, default=0.001)
    ap.add_argument("--loose-joint-margin", type=float, default=0.0)
    ap.add_argument("--suspicious-fromto-threshold", type=float, default=0.003)

    args = ap.parse_args()

    p3 = load_json(args.p3_json)
    ranked = p3.get("ranked", []) or []

    audited = [audit_one_combo(r, args) for r in ranked]

    geometry_pass = [r for r in audited if r["geometry_pass"]]
    strict_pass = [r for r in audited if r["strict_pass"]]
    loose_pass = [r for r in audited if r["loose_margin_pass"]]

    suspicious_count = sum(1 for r in audited if r["original_zero_suspicious"])

    # 排序：先 geometry pass，再有效 hand-support/fr3-object 大，再 grasp object 小，再 joint margin 大
    audited_sorted = sorted(
        audited,
        key=lambda r: (
            not r["geometry_pass"],
            -r["effective_min_path_hand_support"],
            -r["effective_min_path_fr3_object"],
            r["effective_static_grasp_hand_object"],
            -r["combo_min_joint_margin"],
        ),
    )

    out = {
        "format": "v4_12p3a_audit_distance_and_margin_debug",
        "meaning": "Audit P3 result by replacing suspicious distance=0 with norm(fromto) when fromto is nonzero.",
        "source_p3_json": str(resolve_path(args.p3_json)),
        "args": vars(args),
        "source_num_combos": p3.get("num_combos"),
        "source_num_pass": p3.get("num_pass"),
        "audited_num": len(audited),
        "num_strict_pass": len(strict_pass),
        "num_geometry_pass": len(geometry_pass),
        "num_loose_margin_pass": len(loose_pass),
        "num_with_suspicious_zero": suspicious_count,
        "best_strict": strict_pass[0] if strict_pass else None,
        "best_geometry": geometry_pass[0] if geometry_pass else None,
        "best_loose_margin": loose_pass[0] if loose_pass else None,
        "ranked_audit": audited_sorted,
    }

    save_json(args.out, out)

    print("\n========== V4.12P3A DISTANCE + MARGIN AUDIT ==========")
    print("source:", resolve_path(args.p3_json))
    print("out   :", resolve_path(args.out))
    print("source_num_combos:", p3.get("num_combos"))
    print("source_num_pass  :", p3.get("num_pass"))
    print("audited_num      :", len(audited))
    print("num_strict_pass  :", len(strict_pass))
    print("num_geometry_pass:", len(geometry_pass))
    print("num_loose_margin_pass:", len(loose_pass))
    print("num_with_suspicious_zero:", suspicious_count)

    print("\n----- TOP 10 AUDITED -----")
    for i, r in enumerate(audited_sorted[:10], 1):
        print(
            f"{i:02d}. combo={r['combo_id']} "
            f"geom_pass={int(r['geometry_pass'])} "
            f"strict={int(r['strict_pass'])} "
            f"loose={int(r['loose_margin_pass'])} "
            f"HS={r['effective_min_path_hand_support']:+.5f} "
            f"FO={r['effective_min_path_fr3_object']:+.5f} "
            f"GO={r['effective_static_grasp_hand_object']:+.5f} "
            f"GS={r['effective_static_grasp_hand_support']:+.5f} "
            f"margin={r['combo_min_joint_margin']:+.6f} "
            f"seeds={r['pre_seed']}->{r['grasp_seed']}->{r['lift_seed']}"
        )
        for rr in r["bad_reasons"][:4]:
            print("    -", rr)
        if r["original_zero_suspicious"][:2]:
            print("    suspicious:")
            for ss in r["original_zero_suspicious"][:2]:
                print("      *", ss)

    print("======================================================\n")


if __name__ == "__main__":
    main()
