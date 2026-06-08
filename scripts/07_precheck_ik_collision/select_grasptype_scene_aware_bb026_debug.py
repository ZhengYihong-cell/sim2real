#!/usr/bin/env python3
"""
脚本类型：
    debug / selector / grasp-type-aware / scene-conditioned

用途：
    对 BB026 SodaCan 候选做“抓型 + 场景可执行性”快速筛选。
    重点解决当前问题：
        某些候选对物体本身看起来像好抓握，但因为蓝色支撑块挡住大拇指/手指进入通道，
        实际无法抓取。
    
    本脚本把评分拆成两层：
        1. object-only grasp quality
           只看物体局部抓握是否自然、是否靠近物体、是否有可能形成对抗。
        2. scene-conditioned feasibility
           看支撑物是否阻挡 thumb/finger access corridor。
    
    对 side grasp，如果检测到 thumb/finger access 被 support block，
    不再只是扣分，而是直接硬淘汰，并建议切换 top grasp 或 end grasp。

输入：
    diagnostics/current_v412/sodacan_bb026_topk_p2p3_batch_debug/sample*/
    diagnostics/current_v412/sodacan_bb026_fast_generalization_debug/fast_rank_before_p2p3.json
    diagnostics/current_v412/sodacan_bb026_fast_generalization_debug/fast_final_summary.json
    diagnostics/current_v412/sodacan_bb026_grasptype_scene_audit_debug/audit_summary.json

输出：
    diagnostics/current_v412/sodacan_bb026_grasptype_scene_aware_select_debug/
        scene_aware_select_report.txt
        scene_aware_select_summary.json

当前流程位置：
    Top-K dataset prior
        -> 抓型分类 side/top/end
        -> support-aware hard filter
        -> 决定是否继续 P2/P3/P4U6，或者切换抓型

不负责：
    1. 不运行 P2/P3；
    2. 不运行 viewer；
    3. 不生成新的 top grasp；
    4. 不修改 legacy_final_demos；
    5. 不修改 P4U1/P4U6 源码。
"""

from pathlib import Path
import json
import math
import numpy as np


PROJECT = Path.home() / "Projects/o7_mujoco_sim"

INROOT = PROJECT / "diagnostics/current_v412/sodacan_bb026_topk_p2p3_batch_debug"
FAST_RANK_JSON = PROJECT / "diagnostics/current_v412/sodacan_bb026_fast_generalization_debug/fast_rank_before_p2p3.json"
FAST_FINAL_JSON = PROJECT / "diagnostics/current_v412/sodacan_bb026_fast_generalization_debug/fast_final_summary.json"
AUDIT_JSON = PROJECT / "diagnostics/current_v412/sodacan_bb026_grasptype_scene_audit_debug/audit_summary.json"
OUTDIR = PROJECT / "diagnostics/current_v412/sodacan_bb026_grasptype_scene_aware_select_debug"

OBJECT_BODY = "grasp_can"

# 这几个 margin 可以后续调，但先给一个保守规则。
SUPPORT_CLEAR_MARGIN = 0.006
SIDE_BLOCK_HS_THRESH = -0.006
SIDE_HARD_BLOCK_HS_THRESH = -0.012
GOOD_GO_MIN = 0.001
GOOD_GO_MAX = 0.012


def rel(p):
    p = Path(p)
    try:
        return str(p.relative_to(PROJECT))
    except Exception:
        return str(p)


def load_json(path):
    path = Path(path)
    if not path.exists():
        return None
    return json.loads(path.read_text())


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def find_candidate(sample_dir, sid):
    p = sample_dir / f"initial_debug/candidates/sample{sid}_candidate.json"
    if p.exists():
        return p
    xs = sorted((sample_dir / "initial_debug/candidates").glob("*.json"))
    return xs[0] if xs else None


def get_nested(d, path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def get_T_object_target(candidate):
    for path in [
        ("target", "T_object_target"),
        ("target", "T_object_fr3_link7"),
        ("T_object_target",),
        ("T_object_fr3_link7",),
    ]:
        arr = get_nested(candidate, path)
        if arr is None:
            continue
        try:
            T = np.asarray(arr, dtype=float)
            if T.shape == (4, 4):
                return T
        except Exception:
            pass
    return None


def get_ctrl(candidate):
    ctrl = get_nested(candidate, ("hand", "o7_active_ctrl"), {})
    if not isinstance(ctrl, dict):
        ctrl = {}
    out = {}
    for k, v in ctrl.items():
        try:
            out[k] = float(v)
        except Exception:
            pass
    return out


def load_fast_rank_rows():
    d = load_json(FAST_RANK_JSON)
    out = {}
    if not d:
        return out
    for r in d.get("rows_sorted", []):
        sid = str(r.get("sample")).zfill(3)
        out[sid] = r
    return out


def load_fast_final_rows():
    d = load_json(FAST_FINAL_JSON)
    out = {}
    if not d:
        return out
    for r in d.get("rows_sorted", []):
        sid = str(r.get("sample")).zfill(3)
        out[sid] = r
    return out


def load_audit_rows():
    d = load_json(AUDIT_JSON)
    out = {}
    if not d:
        return out
    for r in d.get("rows_sorted", []):
        sid = str(r.get("sample")).zfill(3)
        out[sid] = r
    return out


def old_p3_from_row(row):
    """
    兼容 fast_rank / audit / fast_final 中的 p3 字段。
    """
    p3 = row.get("p3", {})
    if not isinstance(p3, dict):
        return {}

    return {
        "num_pass": p3.get("num_pass"),
        "HS": p3.get("HS", p3.get("min_path_hand_support_clearance")),
        "FO": p3.get("FO", p3.get("min_path_fr3_object_clearance")),
        "GO": p3.get("GO", p3.get("static_grasp_closed_hand_object_distance")),
        "HSc": p3.get("HSc", p3.get("static_grasp_closed_hand_support_clearance")),
        "margin": p3.get("margin", p3.get("combo_min_joint_margin")),
        "hard_reasons": p3.get("hard_reasons", []),
        "best_pass_exists": p3.get("best_pass_exists"),
        "status": p3.get("status", p3.get("best_available_status")),
    }


def classify_grasp_type_from_features(features):
    """
    这里只用 wrist/target 相对物体的位置做粗分类。
    不是最终抓取判断，只决定用哪套 scene-aware hard rule。
    """
    if not features:
        return "unknown", ["missing_features"]

    p = np.asarray(features.get("p", [0, 0, 0]), dtype=float)
    radial_xy = float(features.get("radial_xy", np.linalg.norm(p[:2])))
    z = float(features.get("z", p[2]))

    reasons = []
    reasons.append(f"radial_xy={radial_xy:.5f}")
    reasons.append(f"target_z={z:.5f}")

    # wrist 在物体侧方，通常是 side 或 end。
    # z 很高且 radial 不特别大时，更接近 top。
    if z > 0.12 and radial_xy < 0.17:
        return "top_like", reasons + ["rule=z_high_and_radial_moderate"]
    if radial_xy > 0.18 and abs(z) < 0.16:
        return "side_like", reasons + ["rule=large_radial_and_mid_z"]
    if radial_xy > 0.18 and z < -0.16:
        return "side_low_or_under_like", reasons + ["rule=large_radial_but_low_z"]
    if radial_xy > 0.18 and z > 0.16:
        return "side_high_or_top_transition", reasons + ["rule=large_radial_and_high_z"]
    if radial_xy <= 0.18 and z > 0.12:
        return "top_like", reasons + ["rule=small_radial_and_high_z"]
    return "ambiguous", reasons + ["rule=ambiguous"]


def object_only_score(features, p3):
    """
    物体局部抓握质量：不考虑支撑物阻挡。
    这个分数高，只能说明候选像一个抓握；不能说明场景可执行。
    """
    score = 0.0
    reasons = []

    if features:
        radial_xy = float(features.get("radial_xy", 0.0))
        z = float(features.get("z", 0.0))
        finger_mean = features.get("finger_mean")

        if radial_xy > 0.12:
            score += 15
            reasons.append(f"radial_ok={radial_xy:.5f}")
        else:
            score -= 10
            reasons.append(f"radial_too_small={radial_xy:.5f}")

        if abs(z) < 0.25:
            score += 8
            reasons.append(f"z_not_extreme={z:.5f}")
        else:
            score -= 8
            reasons.append(f"z_extreme={z:.5f}")

        if isinstance(finger_mean, (int, float)) and 0.35 <= float(finger_mean) <= 0.85:
            score += 8
            reasons.append(f"finger_mean_ok={finger_mean:.3f}")
        else:
            score -= 5
            reasons.append(f"finger_mean_bad={finger_mean}")

    GO = p3.get("GO")
    FO = p3.get("FO")

    if isinstance(GO, (int, float)):
        if GOOD_GO_MIN <= GO <= GOOD_GO_MAX:
            score += 20
            reasons.append(f"GO_good={GO:.5f}")
        elif GO < 0:
            score -= 12
            reasons.append(f"GO_penetration={GO:.5f}")
        else:
            reasons.append(f"GO_out={GO:.5f}")

    if isinstance(FO, (int, float)):
        if FO >= 0.003:
            score += 10
            reasons.append(f"FO_clear={FO:.5f}")
        elif FO >= 0:
            score += 3
            reasons.append(f"FO_near={FO:.5f}")
        else:
            score -= 12
            reasons.append(f"FO_collision={FO:.5f}")

    return score, reasons


def scene_feasibility_score(grasp_type, p3, audit):
    """
    场景条件可执行性，重点看 support block。
    对 side grasp，hand-support collision 不是普通扣分，而是硬失败。
    """
    score = 0.0
    reasons = []
    hard_fail = False
    hard_fail_reason = None
    recommended_fallback = None

    HS = p3.get("HS")
    HSc = p3.get("HSc")
    num_pass = p3.get("num_pass")
    hard_reasons = p3.get("hard_reasons") or []

    if isinstance(num_pass, int) and num_pass > 0:
        score += 40
        reasons.append(f"p3_pass={num_pass}")
    else:
        reasons.append("p3_pass_zero")

    if isinstance(HS, (int, float)):
        if HS >= SUPPORT_CLEAR_MARGIN:
            score += 30
            reasons.append(f"support_clear_good={HS:.5f}")
        elif HS >= 0:
            score += 15
            reasons.append(f"support_clear_nonnegative={HS:.5f}")
        elif HS >= SIDE_BLOCK_HS_THRESH:
            score -= 10
            reasons.append(f"support_mild_negative={HS:.5f}")
        elif HS >= SIDE_HARD_BLOCK_HS_THRESH:
            score -= 35
            reasons.append(f"support_blocked={HS:.5f}")
        else:
            score -= 80
            reasons.append(f"support_hard_blocked={HS:.5f}")

    if isinstance(HSc, (int, float)) and HSc < SIDE_HARD_BLOCK_HS_THRESH:
        score -= 40
        reasons.append(f"closed_hand_support_blocked={HSc:.5f}")

    hard_text = " | ".join(str(x) for x in hard_reasons)
    if "q_grasp_closed" in hard_text and "hand-support" in hard_text:
        reasons.append("q_grasp_closed_hits_support")
    if "q_pre_to_q_grasp_open_hand" in hard_text and "hand-support" in hard_text:
        reasons.append("approach_corridor_hits_support")

    # side-like 的核心硬规则：
    # 侧抓如果闭合时手/大拇指/手指需要进入支撑物空间，则直接不是当前场景的好抓握。
    if grasp_type in ["side_like", "side_low_or_under_like", "side_high_or_top_transition"]:
        if isinstance(HS, (int, float)) and HS < SIDE_HARD_BLOCK_HS_THRESH:
            hard_fail = True
            hard_fail_reason = "SIDE_GRASP_SUPPORT_BLOCKED"
            recommended_fallback = "try_top_or_end_grasp"
        if isinstance(HSc, (int, float)) and HSc < SIDE_HARD_BLOCK_HS_THRESH:
            hard_fail = True
            hard_fail_reason = "SIDE_GRASP_CLOSED_HAND_SUPPORT_BLOCKED"
            recommended_fallback = "try_top_or_end_grasp"

    # top-like 不因为 side support 阻挡直接淘汰，但后续必须单独检查 palm/finger 下探 clearance。
    if grasp_type == "top_like":
        if isinstance(HS, (int, float)) and HS < -0.03:
            hard_fail = True
            hard_fail_reason = "TOP_GRASP_TOO_DEEP_IN_SUPPORT"
            recommended_fallback = "try_end_grasp_or_raise_contact_band"
        else:
            reasons.append("top_like_needs_top_access_check")

    settle = audit.get("settle", {}) if isinstance(audit, dict) else {}
    if settle:
        rise = settle.get("object_rise")
        disp = settle.get("object_disp")
        reasons.append(f"settle_rise={rise}")
        reasons.append(f"settle_disp={disp}")

    return score, reasons, hard_fail, hard_fail_reason, recommended_fallback


def final_decision(object_score, scene_score, hard_fail, grasp_type):
    if hard_fail:
        return "REJECT_SCENE_BLOCKED"
    total = object_score + scene_score
    if total >= 70:
        return "TRY_VIEWER"
    if total >= 35:
        return "TRY_LIGHT_P2P3_FIRST"
    if grasp_type == "top_like":
        return "KEEP_FOR_TOP_CHECK"
    return "REJECT_LOW_SCORE"


def main():
    OUTDIR.mkdir(parents=True, exist_ok=True)

    fast_rank = load_fast_rank_rows()
    fast_final = load_fast_final_rows()
    audit_rows = load_audit_rows()

    sample_ids = sorted(
        set(fast_rank.keys()) |
        set(fast_final.keys()) |
        set(audit_rows.keys()) |
        {p.name.replace("sample", "") for p in INROOT.glob("sample*") if p.is_dir()}
    )

    rows = []

    for sid in sample_ids:
        sid = str(sid).zfill(3)
        sample_dir = INROOT / f"sample{sid}"
        cand_path = find_candidate(sample_dir, sid)

        row = {
            "sample": sid,
            "sample_dir": rel(sample_dir),
            "candidate": rel(cand_path) if cand_path else None,
            "fast_rank": fast_rank.get(sid, {}),
            "fast_final": fast_final.get(sid, {}),
            "audit": audit_rows.get(sid, {}),
        }

        features = None
        if fast_rank.get(sid, {}).get("features"):
            features = fast_rank[sid]["features"]
        else:
            if cand_path and cand_path.exists():
                cand = load_json(cand_path)
                T = get_T_object_target(cand)
                ctrl = get_ctrl(cand)
                if T is not None:
                    p = T[:3, 3]
                    fingers = [
                        ctrl.get("index_mcp_pitch"),
                        ctrl.get("middle_mcp_pitch"),
                        ctrl.get("ring_mcp_pitch"),
                        ctrl.get("pinky_mcp_pitch"),
                    ]
                    fingers = [x for x in fingers if isinstance(x, (int, float))]
                    features = {
                        "p": p.tolist(),
                        "radial_xy": float(np.linalg.norm(p[:2])),
                        "z": float(p[2]),
                        "finger_mean": float(np.mean(fingers)) if fingers else None,
                        "thumb_yaw": ctrl.get("thumb_cmc_yaw"),
                        "thumb_roll": ctrl.get("thumb_cmc_roll"),
                        "thumb_pitch": ctrl.get("thumb_cmc_pitch"),
                    }

        row["features"] = features

        # 优先用 fast_final 的 fast_p3，其次 fast_rank 的旧 p3，其次 audit p3。
        p3 = {}
        if fast_final.get(sid, {}).get("fast_p3"):
            p3 = fast_final[sid]["fast_p3"]
        elif fast_rank.get(sid, {}).get("p3"):
            p3 = old_p3_from_row(fast_rank[sid])
        elif audit_rows.get(sid, {}).get("p3"):
            p3 = old_p3_from_row(audit_rows[sid])
        row["p3_used"] = p3

        grasp_type, gt_reasons = classify_grasp_type_from_features(features)
        obj_score, obj_reasons = object_only_score(features, p3)
        scene_score, scene_reasons, hard_fail, hard_reason, fallback = scene_feasibility_score(
            grasp_type, p3, row["audit"]
        )
        decision = final_decision(obj_score, scene_score, hard_fail, grasp_type)

        row["grasp_type"] = grasp_type
        row["grasp_type_reasons"] = gt_reasons
        row["object_only_score"] = float(obj_score)
        row["object_only_reasons"] = obj_reasons
        row["scene_feasibility_score"] = float(scene_score)
        row["scene_feasibility_reasons"] = scene_reasons
        row["hard_fail"] = bool(hard_fail)
        row["hard_fail_reason"] = hard_reason
        row["recommended_fallback"] = fallback
        row["final_score"] = float(obj_score + scene_score)
        row["decision"] = decision

        rows.append(row)

    rows_sorted = sorted(
        rows,
        key=lambda r: (
            r["decision"] in ["TRY_VIEWER", "TRY_LIGHT_P2P3_FIRST", "KEEP_FOR_TOP_CHECK"],
            r["final_score"],
        ),
        reverse=True,
    )

    summary = {
        "format": "sodacan_bb026_grasptype_scene_aware_select_debug_v1",
        "rules": {
            "SUPPORT_CLEAR_MARGIN": SUPPORT_CLEAR_MARGIN,
            "SIDE_BLOCK_HS_THRESH": SIDE_BLOCK_HS_THRESH,
            "SIDE_HARD_BLOCK_HS_THRESH": SIDE_HARD_BLOCK_HS_THRESH,
            "GOOD_GO_MIN": GOOD_GO_MIN,
            "GOOD_GO_MAX": GOOD_GO_MAX,
            "note": "side grasp support block is a hard reject, not just a score penalty",
        },
        "rows_sorted": rows_sorted,
    }

    save_json(OUTDIR / "scene_aware_select_summary.json", summary)

    lines = []
    lines.append("========== BB026 GRASP-TYPE + SCENE-AWARE SELECT REPORT ==========")
    lines.append("")
    lines.append("核心规则：")
    lines.append("  object-only good 不等于 scene-conditioned good。")
    lines.append("  对 side grasp，如果 hand-support clearance 明显为负，则认为 thumb/finger access 被 support block，直接硬淘汰。")
    lines.append("")

    for r in rows_sorted:
        p3 = r.get("p3_used", {})
        lines.append(
            f"sample={r['sample']} "
            f"type={r['grasp_type']} "
            f"decision={r['decision']} "
            f"final={r['final_score']:.2f} "
            f"object={r['object_only_score']:.2f} "
            f"scene={r['scene_feasibility_score']:.2f} "
            f"hard={r['hard_fail_reason']} "
            f"fallback={r['recommended_fallback']} "
            f"HS={p3.get('HS')} FO={p3.get('FO')} GO={p3.get('GO')} HSc={p3.get('HSc')}"
        )
        lines.append("  grasp_type:")
        for x in r["grasp_type_reasons"]:
            lines.append(f"    - {x}")
        lines.append("  object_only:")
        for x in r["object_only_reasons"]:
            lines.append(f"    - {x}")
        lines.append("  scene_feasibility:")
        for x in r["scene_feasibility_reasons"]:
            lines.append(f"    - {x}")
        lines.append("")

    n_side_blocked = sum(
        1 for r in rows_sorted
        if r["grasp_type"] in ["side_like", "side_low_or_under_like", "side_high_or_top_transition"]
        and r["hard_fail"]
    )
    n_side_total = sum(
        1 for r in rows_sorted
        if r["grasp_type"] in ["side_like", "side_low_or_under_like", "side_high_or_top_transition"]
    )
    n_top_like = sum(1 for r in rows_sorted if r["grasp_type"] == "top_like")

    lines.append("---- 结论 ----")
    lines.append(f"side candidates blocked: {n_side_blocked}/{n_side_total}")
    lines.append(f"top-like candidates: {n_top_like}")

    if n_side_total > 0 and n_side_blocked == n_side_total:
        lines.append("结论：当前 BB026 的 side grasp 候选整体被 support block，不应该继续侧抓 viewer。")
        lines.append("建议：切换 top grasp 或 end grasp，并为 top/end 生成新的 candidate。")
    else:
        lines.append("结论：仍存在未被 support block 的 side candidate，可继续轻量 P2/P3 或 viewer。")

    lines.append("===================================================================")

    report = "\n".join(lines) + "\n"
    (OUTDIR / "scene_aware_select_report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
