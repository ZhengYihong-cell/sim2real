#!/usr/bin/env python3
"""
脚本类型：
    debug / diagnostic / grasp-type-and-scene-audit

用途：
    审计 sem-SodaCan-bb0262... 第二轮泛化候选失败原因。
    当前 sample009 viewer 显示抓型不对，且日志表明物体在抓取前已经下落约 18.8mm。
    本脚本用于区分：
        1. scene 初始支撑是否稳定；
        2. P3 best_available 是否只是 hand-support 轻微失败；
        3. 候选是否存在明显 top-poke / q_pre=q_grasp / 无真实 approach 风险；
        4. 哪些 sample 值得继续跑 viewer。

输入：
    diagnostics/current_v412/sodacan_bb026_topk_p2p3_batch_debug/sample*/

输出：
    diagnostics/current_v412/sodacan_bb026_grasptype_scene_audit_debug/
        audit_report.txt
        audit_summary.json

当前流程位置：
    BB026 Top-K P2/P3 已跑完
        -> 本脚本审计 grasp-type / scene-stability
        -> 选择真正适合 P4U6/P4U1 的候选

不负责：
    1. 不修改 scene；
    2. 不重新运行 P2/P3；
    3. 不运行 viewer；
    4. 不修改 can52 / sodacan165 固化 demo；
    5. 不把某个 sample 写死为最终结果。
"""

from pathlib import Path
import json
import math
import traceback

import mujoco
import numpy as np


PROJECT = Path.home() / "Projects/o7_mujoco_sim"
INROOT = PROJECT / "diagnostics/current_v412/sodacan_bb026_topk_p2p3_batch_debug"
OUTDIR = PROJECT / "diagnostics/current_v412/sodacan_bb026_grasptype_scene_audit_debug"
OUTDIR.mkdir(parents=True, exist_ok=True)

OBJECT_BODY = "grasp_can"
SETTLE_STEPS = 1500


def rel(p: Path) -> str:
    try:
        return str(p.relative_to(PROJECT))
    except Exception:
        return str(p)


def load_json(path: Path):
    return json.loads(path.read_text())


def find_scene(sample_dir: Path):
    scene_dir = sample_dir / "initial_debug/scene"
    xs = sorted(scene_dir.glob("*.xml"))
    return xs[0] if xs else None


def find_candidate(sample_dir: Path, sid: str):
    p = sample_dir / f"initial_debug/candidates/sample{sid}_candidate.json"
    if p.exists():
        return p
    xs = sorted((sample_dir / "initial_debug/candidates").glob("*.json"))
    return xs[0] if xs else None


def object_settle_check(scene_path: Path):
    out = {
        "ok": False,
        "error": None,
        "object_pos0": None,
        "object_pos1": None,
        "object_disp": None,
        "object_rise": None,
        "ncon_final": None,
        "object_support_contacts_final": None,
    }

    try:
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)

        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY)
        if bid < 0:
            raise RuntimeError(f"missing body: {OBJECT_BODY}")

        mujoco.mj_forward(model, data)
        p0 = np.array(data.xpos[bid], dtype=float)

        for _ in range(SETTLE_STEPS):
            mujoco.mj_step(model, data)

        p1 = np.array(data.xpos[bid], dtype=float)

        object_support = 0
        for i in range(data.ncon):
            c = data.contact[i]
            g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
            g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
            s = (g1 + " " + g2).lower()
            if ("support" in s or "pedestal" in s or "table" in s) and ("can" in s or "grasp" in s):
                object_support += 1

        out.update({
            "ok": True,
            "object_pos0": p0.tolist(),
            "object_pos1": p1.tolist(),
            "object_disp": float(np.linalg.norm(p1 - p0)),
            "object_rise": float(p1[2] - p0[2]),
            "ncon_final": int(data.ncon),
            "object_support_contacts_final": int(object_support),
        })

    except Exception as e:
        out["error"] = repr(e)
        out["traceback"] = traceback.format_exc()

    return out


def summarize_p3(p3_path: Path):
    if not p3_path.exists():
        return {"exists": False}

    d = load_json(p3_path)
    ba = d.get("best_available") or {}
    bp = d.get("best_pass")

    return {
        "exists": True,
        "num_combos": d.get("num_combos"),
        "num_pass": d.get("num_pass"),
        "best_pass_exists": bp is not None,
        "best_available_status": ba.get("precheck_status"),
        "best_available_score": ba.get("score"),
        "min_path_hand_support_clearance": ba.get("min_path_hand_support_clearance"),
        "min_path_fr3_object_clearance": ba.get("min_path_fr3_object_clearance"),
        "static_grasp_closed_hand_object_distance": ba.get("static_grasp_closed_hand_object_distance"),
        "static_grasp_closed_hand_support_clearance": ba.get("static_grasp_closed_hand_support_clearance"),
        "combo_min_joint_margin": ba.get("combo_min_joint_margin"),
        "hard_reasons": ba.get("hard_reasons") or [],
    }


def summarize_candidate(candidate_path: Path):
    if not candidate_path or not candidate_path.exists():
        return {"exists": False}

    d = load_json(candidate_path)
    hand = d.get("hand", {})
    ctrl = hand.get("o7_active_ctrl") or d.get("o7_active_ctrl") or {}

    # 尽量兼容不同 candidate 格式
    hand_pos = None
    for key in ["hand_pos", "position", "pos"]:
        if key in hand and isinstance(hand[key], list) and len(hand[key]) == 3:
            hand_pos = hand[key]
            break
        if key in d and isinstance(d[key], list) and len(d[key]) == 3:
            hand_pos = d[key]
            break

    return {
        "exists": True,
        "hand_pos": hand_pos,
        "ctrl": ctrl,
    }


def classify_row(row):
    reasons = []
    score = 0.0

    settle = row.get("settle", {})
    p3 = row.get("p3", {})

    if not settle.get("ok"):
        reasons.append("settle_check_failed")
        score -= 100
    else:
        rise = settle.get("object_rise")
        disp = settle.get("object_disp")
        if abs(rise) > 0.005 or disp > 0.006:
            reasons.append(f"scene_unstable_object_moves_before_grasp rise={rise:.5f} disp={disp:.5f}")
            score -= 50
        else:
            reasons.append("scene_stable")
            score += 20

    num_pass = p3.get("num_pass")
    if isinstance(num_pass, int) and num_pass > 0:
        reasons.append(f"p3_has_pass={num_pass}")
        score += 30
    else:
        reasons.append("p3_pass_zero")
        score -= 5

    hs = p3.get("min_path_hand_support_clearance")
    fo = p3.get("min_path_fr3_object_clearance")
    go = p3.get("static_grasp_closed_hand_object_distance")

    if isinstance(fo, (int, float)) and fo >= 0.003:
        reasons.append(f"fr3_object_clearance_ok={fo:.5f}")
        score += 8
    elif isinstance(fo, (int, float)) and fo >= 0.0:
        reasons.append(f"fr3_object_clearance_near={fo:.5f}")
        score += 3
    elif isinstance(fo, (int, float)):
        reasons.append(f"fr3_object_collision={fo:.5f}")
        score -= 15

    if isinstance(go, (int, float)) and 0.0005 <= go <= 0.012:
        reasons.append(f"hand_object_distance_reasonable={go:.5f}")
        score += 8
    elif isinstance(go, (int, float)) and go < 0.0:
        reasons.append(f"static_penetration_or_too_deep={go:.5f}")
        score -= 10
    elif isinstance(go, (int, float)):
        reasons.append(f"hand_object_distance={go:.5f}")

    if isinstance(hs, (int, float)):
        if hs >= 0.0:
            reasons.append(f"hand_support_clear={hs:.5f}")
            score += 10
        elif hs >= -0.018:
            reasons.append(f"mild_hand_support_penetration={hs:.5f}")
            score -= 2
        else:
            reasons.append(f"large_hand_support_penetration={hs:.5f}")
            score -= 12

    hard = " | ".join(p3.get("hard_reasons") or [])
    if "q_pre_to_q_grasp_open_hand" in hard:
        reasons.append("approach_has_collision_risk")
        score -= 10
    if "q_grasp_closed" in hard and "hand-support" in hard:
        reasons.append("closed_hand_hits_support")
        score -= 6

    # 只给建议，不做最终判断
    if any("scene_unstable" in r for r in reasons):
        decision = "REJECT_UNTIL_SCENE_STABLE"
    elif score >= 35:
        decision = "TRY_VIEWER"
    elif score >= 15:
        decision = "NEAR_TRY_AFTER_FILTER"
    else:
        decision = "REJECT_GRASPTYPE_OR_COLLISION"

    return score, decision, reasons


def main():
    rows = []

    sample_dirs = sorted([p for p in INROOT.glob("sample*") if p.is_dir()])
    for sdir in sample_dirs:
        sid = sdir.name.replace("sample", "")
        scene = find_scene(sdir)
        cand = find_candidate(sdir, sid)
        p3_path = sdir / f"sample{sid}_p3.json"

        row = {
            "sample": sid,
            "sample_index": int(sid),
            "sample_dir": rel(sdir),
            "scene": rel(scene) if scene else None,
            "candidate": rel(cand) if cand else None,
            "p3_json": rel(p3_path) if p3_path.exists() else None,
            "settle": object_settle_check(scene) if scene else {"ok": False, "error": "missing scene"},
            "candidate_summary": summarize_candidate(cand) if cand else {"exists": False},
            "p3": summarize_p3(p3_path),
        }

        score, decision, reasons = classify_row(row)
        row["audit_score"] = float(score)
        row["decision"] = decision
        row["audit_reasons"] = reasons
        rows.append(row)

    rows_sorted = sorted(rows, key=lambda r: r["audit_score"], reverse=True)

    summary = {
        "format": "sodacan_bb026_grasptype_scene_audit_debug_v1",
        "input_root": rel(INROOT),
        "out_dir": rel(OUTDIR),
        "object_body": OBJECT_BODY,
        "settle_steps": SETTLE_STEPS,
        "rows_sorted": rows_sorted,
    }

    (OUTDIR / "audit_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

    lines = []
    lines.append("========== SODACAN BB026 GRASP-TYPE + SCENE AUDIT ==========")
    lines.append(f"inroot       : {rel(INROOT)}")
    lines.append(f"settle_steps : {SETTLE_STEPS}")
    lines.append("")

    for r in rows_sorted:
        st = r["settle"]
        p3 = r["p3"]
        lines.append(
            f"sample={r['sample']} decision={r['decision']} score={r['audit_score']:.2f} "
            f"settle_rise={st.get('object_rise')} settle_disp={st.get('object_disp')} "
            f"p3_pass={p3.get('num_pass')} "
            f"HS={p3.get('min_path_hand_support_clearance')} "
            f"FO={p3.get('min_path_fr3_object_clearance')} "
            f"GO={p3.get('static_grasp_closed_hand_object_distance')}"
        )
        for rr in r["audit_reasons"]:
            lines.append(f"  - {rr}")
        lines.append("")

    lines.append("---- 结论提示 ----")
    lines.append("1. 若大多数 sample 都是 REJECT_UNTIL_SCENE_STABLE，说明 BB026 scene 初始物体摆放需要先修正，不能继续 viewer。")
    lines.append("2. 若 scene stable 但 p3_pass=0，优先找 mild_hand_support_penetration 且 fr3_object_clearance_ok 的 sample。")
    lines.append("3. 若出现 q_pre_to_q_grasp 碰撞或 q_pre=q_grasp 视觉上像上方戳取，要进入 grasp-type selector，而不是调 close 参数。")
    lines.append("============================================================")

    txt = "\n".join(lines) + "\n"
    (OUTDIR / "audit_report.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
