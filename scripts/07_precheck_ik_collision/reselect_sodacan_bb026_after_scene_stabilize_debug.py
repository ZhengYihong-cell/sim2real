#!/usr/bin/env python3
"""
脚本类型：
    debug / selector / scene-stabilize-and-p2p3-reselect

用途：
    对 sem-SodaCan-bb0262... 的已有 Top-K 候选重新选择最合适抓握姿态。
    当前问题不是 sample009 单点失败，而是 BB026 多个 sample 的 object 初始 scene 不稳定，
    物体在抓取前会下落 9~19mm，导致后续 P2/P3 和 P4U6 目标错位。

    本脚本执行：
        1. 读取已有 Top-K sample 目录；
        2. 对每个 sample 的 scene 做 object 初始稳定化；
        3. 使用稳定化后的 scene 重新跑 P2；
        4. 使用稳定化后的 scene 重新跑 P3；
        5. 汇总排序，选择最适合继续 P4U6 viewer 的姿态。

输入：
    diagnostics/current_v412/sodacan_bb026_topk_p2p3_batch_debug/sample*/

输出：
    diagnostics/current_v412/sodacan_bb026_stable_scene_reselect_p2p3_debug/
        sampleXXX/
            stable_scene.xml
            settle_before.json
            settle_after.json
            sampleXXX_p2.json
            sampleXXX_p3.json
            sampleXXX_best_plan.json
            terminal_p2.txt
            terminal_p3.txt
        reselect_summary.txt
        reselect_summary.json

当前流程位置：
    BB026 Top-K 初筛失败
        -> scene 稳定化
        -> 重新 P2/P3
        -> 自动选择最适合姿态
        -> 后续再 P4U6/P4U1 viewer

不负责：
    1. 不运行 viewer；
    2. 不执行动态抓取；
    3. 不修改 legacy_final_demos；
    4. 不修改 P4U1/P4U6 源码；
    5. 不在单个 sample 上手调参数。
"""

from pathlib import Path
import argparse
import json
import subprocess
import traceback
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


PROJECT = Path.home() / "Projects/o7_mujoco_sim"

DEFAULT_INROOT = PROJECT / "diagnostics/current_v412/sodacan_bb026_topk_p2p3_batch_debug"
DEFAULT_OUTROOT = PROJECT / "diagnostics/current_v412/sodacan_bb026_stable_scene_reselect_p2p3_debug"

OBJECT_BODY = "grasp_can"
SETTLE_STEPS = 1500

P2_SCRIPT = PROJECT / "scripts/07_precheck_ik_collision/run_v4_12p2_pinocchio_multiseed_ik_debug.py"
P3_SCRIPT = PROJECT / "scripts/07_precheck_ik_collision/run_v4_12p3_pinocchio_ik_plus_mujoco_collision_precheck_debug.py"
URDF = PROJECT / "models/fr3_o7/fr3_o7_mujoco_ready_obj.urdf"
RUNNER_JSON = PROJECT / "diagnostics/current_v412/v4_12a_can52_strict_clearance_debug_runner.json"


def rel(p):
    p = Path(p)
    try:
        return str(p.relative_to(PROJECT))
    except Exception:
        return str(p)


def save_json(path, obj):
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def run_cmd(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return proc.returncode


def find_scene(sample_dir):
    xs = sorted((sample_dir / "initial_debug/scene").glob("*.xml"))
    return xs[0] if xs else None


def find_candidate(sample_dir, sid):
    p = sample_dir / f"initial_debug/candidates/sample{sid}_candidate.json"
    if p.exists():
        return p
    xs = sorted((sample_dir / "initial_debug/candidates").glob("*.json"))
    return xs[0] if xs else None


def settle_scene(scene_path):
    out = {
        "ok": False,
        "scene": rel(scene_path),
        "object_pos0": None,
        "object_pos1": None,
        "delta": None,
        "object_disp": None,
        "object_rise": None,
        "ncon_final": None,
        "object_support_contacts_final": None,
        "error": None,
    }

    try:
        model = mujoco.MjModel.from_xml_path(str(scene_path))
        data = mujoco.MjData(model)

        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, OBJECT_BODY)
        if bid < 0:
            raise RuntimeError(f"missing body {OBJECT_BODY}")

        mujoco.mj_forward(model, data)
        p0 = np.array(data.xpos[bid], dtype=float)

        for _ in range(SETTLE_STEPS):
            mujoco.mj_step(model, data)

        p1 = np.array(data.xpos[bid], dtype=float)
        delta = p1 - p0

        object_support = 0
        for i in range(data.ncon):
            c = data.contact[i]
            g1 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom1) or ""
            g2 = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, c.geom2) or ""
            s = (g1 + " " + g2).lower()
            has_obj = ("can" in s or "grasp" in s)
            has_support = ("support" in s or "pedestal" in s or "table" in s)
            if has_obj and has_support:
                object_support += 1

        out.update({
            "ok": True,
            "object_pos0": p0.tolist(),
            "object_pos1": p1.tolist(),
            "delta": delta.tolist(),
            "object_disp": float(np.linalg.norm(delta)),
            "object_rise": float(delta[2]),
            "ncon_final": int(data.ncon),
            "object_support_contacts_final": int(object_support),
        })

    except Exception as e:
        out["error"] = repr(e)
        out["traceback"] = traceback.format_exc()

    return out


def patch_body_pos_by_delta(scene_in, scene_out, delta):
    tree = ET.parse(str(scene_in))
    root = tree.getroot()

    body = None
    for b in root.iter("body"):
        if b.attrib.get("name") == OBJECT_BODY:
            body = b
            break

    if body is None:
        raise RuntimeError(f"cannot find body name={OBJECT_BODY}")

    old_pos_str = body.attrib.get("pos", "0 0 0")
    old_pos = np.array([float(x) for x in old_pos_str.split()], dtype=float)
    if old_pos.shape[0] != 3:
        raise RuntimeError(f"bad body pos: {old_pos_str}")

    new_pos = old_pos + np.array(delta, dtype=float)
    body.set("pos", f"{new_pos[0]:.12g} {new_pos[1]:.12g} {new_pos[2]:.12g}")

    scene_out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(scene_out), encoding="utf-8", xml_declaration=True)

    return {
        "old_body_pos": old_pos.tolist(),
        "delta": list(map(float, delta)),
        "new_body_pos": new_pos.tolist(),
        "scene_out": rel(scene_out),
    }


def summarize_p3(p3_path):
    if not p3_path.exists():
        return {"exists": False}

    try:
        d = json.loads(p3_path.read_text())
    except Exception as e:
        return {"exists": True, "parse_ok": False, "error": repr(e)}

    ba = d.get("best_available") or {}
    bp = d.get("best_pass")
    chosen = bp if bp is not None else ba

    return {
        "exists": True,
        "parse_ok": True,
        "num_combos": d.get("num_combos"),
        "num_pass": d.get("num_pass"),
        "best_pass_exists": bp is not None,
        "best_available_status": ba.get("precheck_status"),
        "chosen_status": chosen.get("precheck_status") if isinstance(chosen, dict) else None,
        "score": chosen.get("score") if isinstance(chosen, dict) else None,
        "min_path_hand_support_clearance": chosen.get("min_path_hand_support_clearance") if isinstance(chosen, dict) else None,
        "min_path_fr3_object_clearance": chosen.get("min_path_fr3_object_clearance") if isinstance(chosen, dict) else None,
        "static_grasp_closed_hand_object_distance": chosen.get("static_grasp_closed_hand_object_distance") if isinstance(chosen, dict) else None,
        "static_grasp_closed_hand_support_clearance": chosen.get("static_grasp_closed_hand_support_clearance") if isinstance(chosen, dict) else None,
        "combo_min_joint_margin": chosen.get("combo_min_joint_margin") if isinstance(chosen, dict) else None,
        "hard_reasons": chosen.get("hard_reasons") if isinstance(chosen, dict) else None,
    }


def score_row(row):
    score = 0.0
    reasons = []

    after = row.get("settle_after", {})
    p3 = row.get("p3", {})

    if after.get("ok") and abs(after.get("object_rise", 999)) <= 0.002 and after.get("object_disp", 999) <= 0.003:
        score += 50
        reasons.append("stable_scene_ok")
    else:
        score -= 100
        reasons.append("stable_scene_still_moves")

    num_pass = p3.get("num_pass")
    if isinstance(num_pass, int):
        score += min(num_pass, 300) * 0.5
        reasons.append(f"num_pass={num_pass}")

    hs = p3.get("min_path_hand_support_clearance")
    fo = p3.get("min_path_fr3_object_clearance")
    go = p3.get("static_grasp_closed_hand_object_distance")
    hsc = p3.get("static_grasp_closed_hand_support_clearance")
    margin = p3.get("combo_min_joint_margin")

    if isinstance(hs, (int, float)):
        score += 20.0 * hs
        if hs >= 0:
            score += 10
            reasons.append(f"hand_support_clear={hs:.5f}")
        elif hs >= -0.010:
            score -= 3
            reasons.append(f"mild_hand_support={hs:.5f}")
        else:
            score -= 12
            reasons.append(f"bad_hand_support={hs:.5f}")

    if isinstance(fo, (int, float)):
        score += 10.0 * fo
        if fo >= 0.003:
            score += 8
            reasons.append(f"fr3_object_clear={fo:.5f}")
        elif fo >= 0:
            score += 2
            reasons.append(f"fr3_object_near={fo:.5f}")
        else:
            score -= 20
            reasons.append(f"fr3_object_collision={fo:.5f}")

    if isinstance(go, (int, float)):
        if 0.001 <= go <= 0.012:
            score += 10
            reasons.append(f"good_hand_object_gap={go:.5f}")
        elif go < 0:
            score -= 10
            reasons.append(f"penetration={go:.5f}")
        else:
            reasons.append(f"hand_object_gap={go:.5f}")

    if isinstance(hsc, (int, float)):
        if hsc >= 0:
            score += 8
        elif hsc < -0.020:
            score -= 8

    if isinstance(margin, (int, float)):
        score += min(max(margin, 0.0), 1.0) * 5

    if p3.get("best_pass_exists"):
        score += 30
        reasons.append("has_best_pass")

    row["selector_score"] = float(score)
    row["selector_reasons"] = reasons

    if score >= 80 and p3.get("best_pass_exists"):
        row["decision"] = "TRY_VIEWER_FIRST"
    elif score >= 40:
        row["decision"] = "TRY_AFTER_FILTER_OR_VIEWER"
    else:
        row["decision"] = "REJECT_FOR_NOW"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inroot", default=str(DEFAULT_INROOT))
    ap.add_argument("--outroot", default=str(DEFAULT_OUTROOT))
    ap.add_argument("--samples", default="")
    args = ap.parse_args()

    inroot = Path(args.inroot).expanduser()
    if not inroot.is_absolute():
        inroot = PROJECT / inroot

    outroot = Path(args.outroot).expanduser()
    if not outroot.is_absolute():
        outroot = PROJECT / outroot

    outroot.mkdir(parents=True, exist_ok=True)

    if args.samples.strip():
        sample_dirs = []
        for x in args.samples.replace(",", " ").split():
            sid = f"{int(x):03d}"
            sample_dirs.append(inroot / f"sample{sid}")
    else:
        sample_dirs = sorted([p for p in inroot.glob("sample*") if p.is_dir()])

    rows = []

    for sdir in sample_dirs:
        sid = sdir.name.replace("sample", "")
        odir = outroot / f"sample{sid}"
        odir.mkdir(parents=True, exist_ok=True)

        row = {
            "sample": sid,
            "sample_dir": rel(sdir),
            "out_dir": rel(odir),
        }

        try:
            scene = find_scene(sdir)
            candidate = find_candidate(sdir, sid)

            if scene is None:
                raise RuntimeError("missing initial scene")
            if candidate is None:
                raise RuntimeError("missing candidate")

            row["initial_scene"] = rel(scene)
            row["candidate"] = rel(candidate)

            settle_before = settle_scene(scene)
            save_json(odir / "settle_before.json", settle_before)
            row["settle_before"] = settle_before

            if not settle_before.get("ok"):
                raise RuntimeError("settle before failed")

            stable_scene = odir / "stable_scene.xml"
            patch_info = patch_body_pos_by_delta(scene, stable_scene, settle_before["delta"])
            save_json(odir / "scene_patch_info.json", patch_info)
            row["patch_info"] = patch_info
            row["stable_scene"] = rel(stable_scene)

            settle_after = settle_scene(stable_scene)
            save_json(odir / "settle_after.json", settle_after)
            row["settle_after"] = settle_after

            p2_json = odir / f"sample{sid}_p2.json"
            p3_json = odir / f"sample{sid}_p3.json"
            plan_json = odir / f"sample{sid}_best_plan.json"

            cmd_p2 = [
                "python3", str(P2_SCRIPT.relative_to(PROJECT)),
                "--urdf", str(URDF.relative_to(PROJECT)),
                "--model", str(stable_scene.relative_to(PROJECT)),
                "--candidate", str(candidate.relative_to(PROJECT)),
                "--runner-json", str(RUNNER_JSON.relative_to(PROJECT)) if RUNNER_JSON.exists() else "",
                "--object-body", OBJECT_BODY,
                "--target-frame", "fr3_link7",
                "--out", str(p2_json.relative_to(PROJECT)),
                "--random-seeds", "12",
                "--random-std", "0.6",
                "--max-iters", "350",
                "--pos-tol", "0.00035",
                "--rot-tol", "0.0035",
                "--rot-weight", "0.55",
            ]

            rc2 = run_cmd(cmd_p2, odir / "terminal_p2.txt")
            row["p2_return_code"] = rc2

            if rc2 != 0:
                raise RuntimeError(f"P2 failed rc={rc2}")

            cmd_p3 = [
                "python3", str(P3_SCRIPT.relative_to(PROJECT)),
                "--p2-json", str(p2_json.relative_to(PROJECT)),
                "--model", str(stable_scene.relative_to(PROJECT)),
                "--candidate", str(candidate.relative_to(PROJECT)),
                "--object-body", OBJECT_BODY,
                "--out", str(p3_json.relative_to(PROJECT)),
                "--best-plan-out", str(plan_json.relative_to(PROJECT)),
                "--top-per-target", "8",
                "--max-combos", "512",
                "--path-samples", "40",
                "--min-hand-support-clearance", "0.0",
                "--min-fr3-object-clearance", "0.0",
                "--max-grasp-hand-object-distance", "0.050",
                "--min-joint-margin", "0.0",
            ]

            rc3 = run_cmd(cmd_p3, odir / "terminal_p3.txt")
            row["p3_return_code"] = rc3

            if rc3 != 0:
                row["p3_error_note"] = f"P3 failed rc={rc3}"

            row["p2_json"] = rel(p2_json)
            row["p3_json"] = rel(p3_json)
            row["plan_json"] = rel(plan_json)
            row["p3"] = summarize_p3(p3_json)

        except Exception as e:
            row["error"] = repr(e)
            row["traceback"] = traceback.format_exc()
            row.setdefault("p3", {"exists": False})

        score_row(row)
        rows.append(row)

    rows_sorted = sorted(rows, key=lambda r: r.get("selector_score", -999999), reverse=True)

    summary = {
        "format": "sodacan_bb026_stable_scene_reselect_p2p3_debug_v1",
        "inroot": rel(inroot),
        "outroot": rel(outroot),
        "rows_sorted": rows_sorted,
    }

    save_json(outroot / "reselect_summary.json", summary)

    lines = []
    lines.append("========== SODACAN BB026 STABLE-SCENE RESELECT SUMMARY ==========")
    lines.append(f"inroot : {rel(inroot)}")
    lines.append(f"outroot: {rel(outroot)}")
    lines.append("")

    for r in rows_sorted:
        before = r.get("settle_before", {})
        after = r.get("settle_after", {})
        p3 = r.get("p3", {})
        lines.append(
            f"sample={r['sample']} decision={r.get('decision')} score={r.get('selector_score'):.2f} "
            f"before_rise={before.get('object_rise')} after_rise={after.get('object_rise')} "
            f"pass={p3.get('num_pass')} status={p3.get('chosen_status')} "
            f"HS={p3.get('min_path_hand_support_clearance')} "
            f"FO={p3.get('min_path_fr3_object_clearance')} "
            f"GO={p3.get('static_grasp_closed_hand_object_distance')} "
            f"HSc={p3.get('static_grasp_closed_hand_support_clearance')} "
            f"margin={p3.get('combo_min_joint_margin')}"
        )
        for rr in r.get("selector_reasons", []):
            lines.append(f"  - {rr}")
        if r.get("error"):
            lines.append(f"  ERROR: {r.get('error')}")
        lines.append("")

    lines.append("---- next ----")
    lines.append("选择排序第一且 decision 不是 REJECT_FOR_NOW 的 sample，再进入 P4U6 viewer。")
    lines.append("如果所有 sample 仍然 REJECT_FOR_NOW，则 BB026 这批 dataset prior 抓型不适合当前 tabletop，需要换候选或换物体。")
    lines.append("=================================================================")

    txt = "\n".join(lines) + "\n"
    (outroot / "reselect_summary.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
