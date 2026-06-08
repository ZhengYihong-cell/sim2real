#!/usr/bin/env python3
"""
脚本类型：
    debug / v4.13d / generic-object-generalization / topk-light-p2p3

用途：
    对一个“几何上正常”的数据集物体执行通用泛化流程：
        1. 读取 V4.13 selector 输出的 Top-K 先验；
        2. 使用 build_v4_13_generic_candidate_scene_debug.py 生成 scene/candidate；
        3. 对 Top-K 执行轻量 P2 IK；
        4. 执行轻量 P3 MuJoCo collision precheck；
        5. 自动排序并生成 rank1 viewer 脚本。

输入：
    diagnostics/current_v413/<object>_selector_debug/selected_topk_compact.json
    dataset/.../validate_results/seed1/<object_code>.npy
    dataset/meshdata/<object_code>/coacd/decomposed.obj
    scripts/07_precheck_ik_collision/build_v4_13_generic_candidate_scene_debug.py
    P2/P3/P4U6 既有脚本

输出：
    diagnostics/current_v413/<object>_v413d_generic_p2p3_debug/
        localXXX_rawYYY_<type>/
            scene.xml
            candidate.json
            integrity_report.txt
            p2.json
            p3.json
            plan.json
            terminal_build.txt
            terminal_p2.txt
            terminal_p3.txt
        v4_13d_summary.txt
        v4_13d_summary.json
    scripts/05_execution_runner/run_v4_13d_selected_viewer_debug.sh

当前流程位置：
    资产质量筛选
        -> V4.13 selector
        -> 本脚本 generic builder + light P2/P3
        -> P4U6/P4U1 viewer

不负责：
    1. 不做沿某个轴的人工微调；
    2. 不把某个物体写死成 can/bottle 特例；
    3. 不修改 legacy_final_demos；
    4. 不修改 P4U1/P4U6 源码；
    5. 不做全量慢筛。
"""

from pathlib import Path
import argparse
import json
import math
import subprocess
import traceback


PROJECT = Path.home() / "Projects/o7_mujoco_sim"

BUILDER = PROJECT / "scripts/07_precheck_ik_collision/build_v4_13_generic_candidate_scene_debug.py"
P2_SCRIPT = PROJECT / "scripts/07_precheck_ik_collision/run_v4_12p2_pinocchio_multiseed_ik_debug.py"
P3_SCRIPT = PROJECT / "scripts/07_precheck_ik_collision/run_v4_12p3_pinocchio_ik_plus_mujoco_collision_precheck_debug.py"
P4U6_SCRIPT = PROJECT / "scripts/05_execution_runner/run_v4_12p4u6_ik_path_record_demo_debug.py"

URDF = PROJECT / "models/fr3_o7/fr3_o7_mujoco_ready_obj.urdf"
RUNNER_JSON = PROJECT / "diagnostics/current_v412/v4_12a_can52_strict_clearance_debug_runner.json"

O7_ACTIVE_JOINTS = [
    "thumb_cmc_roll",
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]


def rel(p):
    p = Path(p)
    try:
        return str(p.relative_to(PROJECT))
    except Exception:
        return str(p)


def resolve(p):
    p = Path(str(p)).expanduser()
    if not p.is_absolute():
        p = PROJECT / p
    return p


def load_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def run_cmd(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        p = subprocess.run(
            cmd,
            cwd=str(PROJECT),
            stdout=f,
            stderr=subprocess.STDOUT,
            text=True,
        )
    return p.returncode


def safe_name(s):
    return "".join(c if c.isalnum() or c in "_-." else "_" for c in str(s))


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def summarize_p3(path):
    path = Path(path)
    if not path.exists():
        return {"exists": False}

    try:
        d = load_json(path)
    except Exception as e:
        return {"exists": True, "parse_ok": False, "error": repr(e)}

    bp = d.get("best_pass")
    ba = d.get("best_available") or {}
    chosen = bp if bp is not None else ba

    out = {
        "exists": True,
        "parse_ok": True,
        "num_combos": d.get("num_combos"),
        "num_pass": d.get("num_pass"),
        "best_pass_exists": bp is not None,
        "chosen_is": "best_pass" if bp is not None else "best_available",
    }

    if isinstance(chosen, dict):
        out.update({
            "status": chosen.get("precheck_status"),
            "score": chosen.get("score"),
            "HS": chosen.get("min_path_hand_support_clearance"),
            "FO": chosen.get("min_path_fr3_object_clearance"),
            "GO": chosen.get("static_grasp_closed_hand_object_distance"),
            "HSc": chosen.get("static_grasp_closed_hand_support_clearance"),
            "margin": chosen.get("combo_min_joint_margin"),
            "hard_reasons": chosen.get("hard_reasons") or [],
        })

    return out


def score_row(row):
    p3 = row.get("p3", {})
    selector_score = row.get("selector_final_score")

    s = 0.0
    reasons = []

    if finite(selector_score):
        s += 0.05 * float(selector_score)
        reasons.append(f"selector_score={float(selector_score):.3f}")

    if row.get("build_return_code") != 0:
        s -= 400
        reasons.append("build_failed")

    if row.get("p2_return_code") not in [0, None]:
        s -= 250
        reasons.append("p2_failed")

    if row.get("p3_return_code") not in [0, None]:
        s -= 150
        reasons.append("p3_failed")

    if not p3.get("exists"):
        s -= 200
        reasons.append("no_p3_json")
        row["v413d_score"] = float(s)
        row["v413d_score_reasons"] = reasons
        return row

    if p3.get("best_pass_exists"):
        s += 160
        reasons.append("best_pass_exists")
    else:
        reasons.append("no_best_pass")

    n = p3.get("num_pass")
    if isinstance(n, int):
        s += min(n, 50) * 4.0
        reasons.append(f"num_pass={n}")

    go = p3.get("GO")
    if finite(go):
        go = float(go)
        if -0.010 <= go <= 0.012:
            s += 35
            reasons.append(f"GO_near_or_contact={go:.5f}")
        elif 0.012 < go <= 0.030:
            s -= 10
            reasons.append(f"GO_gap_somewhat_large={go:.5f}")
        elif go > 0.030:
            s -= 55
            reasons.append(f"GO_too_far={go:.5f}")
        else:
            s -= 25
            reasons.append(f"GO_too_deep={go:.5f}")

    for key in ["HS", "HSc"]:
        c = p3.get(key)
        if finite(c):
            c = float(c)
            if c >= 0.006:
                s += 18
                reasons.append(f"{key}_clear_good={c:.5f}")
            elif c >= 0.0:
                s += 10
                reasons.append(f"{key}_nonnegative={c:.5f}")
            elif c >= -0.006:
                s += 2
                reasons.append(f"{key}_soft_touch={c:.5f}")
            elif c >= -0.018:
                s -= 18
                reasons.append(f"{key}_medium_penetration={c:.5f}")
            else:
                s -= 70
                reasons.append(f"{key}_hard_penetration={c:.5f}")

    fo = p3.get("FO")
    if finite(fo):
        fo = float(fo)
        if fo >= 0.006:
            s += 20
            reasons.append(f"FO_clear={fo:.5f}")
        elif fo >= 0.0:
            s += 6
            reasons.append(f"FO_nonnegative={fo:.5f}")
        elif fo >= -0.006:
            s -= 15
            reasons.append(f"FO_slight_collision={fo:.5f}")
        else:
            s -= 60
            reasons.append(f"FO_bad_collision={fo:.5f}")

    margin = p3.get("margin")
    if finite(margin):
        s += min(max(float(margin), 0.0), 1.0) * 4.0
        reasons.append(f"joint_margin={float(margin):.3f}")

    row["v413d_score"] = float(s)
    row["v413d_score_reasons"] = reasons
    return row


def get_candidate_ctrl(candidate_path):
    d = load_json(candidate_path)
    ctrl = d.get("hand", {}).get("o7_active_ctrl")
    if not isinstance(ctrl, dict):
        ctrl = d.get("o7_active_ctrl")
    if not isinstance(ctrl, dict):
        raise RuntimeError(f"candidate missing o7_active_ctrl: {candidate_path}")

    return {j: float(ctrl[j]) for j in O7_ACTIVE_JOINTS}


def make_best_config(candidate_path, out_path):
    base_candidates = [
        PROJECT / "legacy_final_demos/final_sodacan165_sample014_p4u6_ready_gated_snap_demo/inputs/best_config.json",
        PROJECT / "legacy_final_demos/final_can52_p4u1_ready_gated_snap_demo/inputs/best_config.json",
        PROJECT / "diagnostics/current_v412/v4_12p4h2_can52_contact_sequence_proxy_debug/best_contact_sequence_config_ctrlsplit_debug.json",
    ]

    base_path = next((p for p in base_candidates if p.exists()), None)
    if base_path is None:
        raise FileNotFoundError("cannot find base best_config")

    cfg = load_json(base_path)
    ctrl = get_candidate_ctrl(candidate_path)

    cfg.setdefault("best_record", {})
    cfg["best_record"].setdefault("hand_config", {})
    cfg["best_record"]["hand_config"]["ctrl"] = ctrl
    cfg["best_record"]["hand_config"]["source"] = "v4_13d_generic_candidate_ctrl"

    cfg["v4_13d_debug_note"] = {
        "type": "generic_object_generalization_best_config",
        "base_config": rel(base_path),
        "candidate": rel(candidate_path),
        "reason": "runner reads best_record.hand_config.ctrl; use selected generic candidate O7 active ctrl.",
    }

    save_json(out_path, cfg)
    return base_path, ctrl


def make_viewer_script(best, best_config_path, outroot):
    script = PROJECT / "scripts/05_execution_runner/run_v4_13d_selected_viewer_debug.sh"
    viewer_out = outroot / "viewer_selected"

    scene = best["scene"]
    candidate = best["candidate"]
    p3_json = best["p3_json"]
    object_body = "grasp_object"

    text = f'''#!/usr/bin/env bash
# 脚本类型：
#   debug / execution-runner / v4.13d-selected-viewer
#
# 用途：
#   运行 V4.13d 通用泛化流程自动选出的 rank1 candidate。
#
# 输出：
#   {rel(viewer_out)}/terminal.txt
#   {rel(viewer_out)}/result.json
#   {rel(viewer_out)}/path_plan.json
#
# 不负责：
#   不重新筛选，不修改 legacy demo，不修改 P4U6/P4U1 源码。

set -euo pipefail

cd "$HOME/Projects/o7_mujoco_sim"
source "$HOME/mujoco_env/bin/activate"

OUTDIR="{rel(viewer_out)}"
mkdir -p "$OUTDIR"

LOG="$OUTDIR/terminal.txt"
RESULT_JSON="$OUTDIR/result.json"
PATH_PLAN_JSON="$OUTDIR/path_plan.json"

exec > >(tee "$LOG") 2>&1

echo "========== V4.13D SELECTED VIEWER =========="
echo "tag          : {best.get('tag')}"
echo "local index  : {best.get('valid_local_index')}"
echo "raw index    : {best.get('raw_sample_index')}"
echo "grasp_type   : {best.get('grasp_type')}"
echo "score        : {best.get('v413d_score')}"
echo "scene        : {scene}"
echo "candidate    : {candidate}"
echo "p3_json      : {p3_json}"
echo "object_body  : {object_body}"
echo "best_config  : {rel(best_config_path)}"
echo

./run_mujoco_clean.sh scripts/05_execution_runner/run_v4_12p4u6_ik_path_record_demo_debug.py \\
  --model "{scene}" \\
  --candidate "{candidate}" \\
  --p3-json "{p3_json}" \\
  --best-config "{rel(best_config_path)}" \\
  --which best_available \\
  --object-body "{object_body}" \\
  --target-body fr3_link7 \\
  --out "$RESULT_JSON" \\
  --plan-out "$PATH_PLAN_JSON" \\
  --viewer \\
  --keep-viewer-open \\
  --start-arm-mode zero_clamped \\
  --start-hold-duration 0.8 \\
  --home-hold-duration 0.3 \\
  --pre-hold-duration 0.35 \\
  --grasp-settle-duration 0.25 \\
  --close-duration 0.45 \\
  --post-close-target-hold-duration 0.25 \\
  --micro-squeeze-duration 0.40 \\
  --micro-squeeze-fraction 0.04 \\
  --finger-close-scale 1.16 \\
  --thumb-pitch-from-finger-gain 0.24 \\
  --grip-ready-stable-steps 6 \\
  --min-live-non-thumb 1 \\
  --opposition-cos-threshold -0.30 \\
  --max-grip-disp 0.024 \\
  --max-extra-disp-during-squeeze 0.006 \\
  --approach-abort-disp 0.030 \\
  --approach-min-clearance 0.002 \\
  --grasp-path-min-clearance 0.001 \\
  --plan-attempts 3 \\
  --rrt-max-iters 1800 \\
  --rrt-step 0.30 \\
  --edge-step 0.045 \\
  --goal-bias 0.20 \\
  --shortcut-iters 60 \\
  --joint-speed-rad-s 0.85 \\
  --min-segment-duration 0.20 \\
  --hard-servo-approach \\
  --enable-lift \\
  --lift-z 0.090 \\
  --lift-duration 2.6 \\
  --final-hold-duration 0.9 \\
  --print-every-steps 100 \\
  --log-every-steps 100 \\
  --frame-sleep 0.0015

echo
echo "========== RESULT QUICK VIEW =========="
python3 - <<'R'
import json
from pathlib import Path

p = Path("{rel(viewer_out)}/result.json")
if p.exists():
    d = json.loads(p.read_text())
    for k in [
        "success", "stop_reason", "grip_ready",
        "final_object_disp", "final_object_rise", "max_object_rise",
        "final_groups", "final_opposition_cos", "max_stable_count"
    ]:
        if k in d:
            print(k, ":", d[k])
R
'''
    script.write_text(text)
    script.chmod(0o755)
    return script


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object-code", required=True)
    ap.add_argument("--prior-dir", required=True)
    ap.add_argument("--mesh-root", required=True)
    ap.add_argument("--selector-json", required=True)
    ap.add_argument("--outroot", required=True)
    ap.add_argument("--max-samples", type=int, default=5)
    ap.add_argument("--support-top-z", type=float, default=0.23)
    ap.add_argument("--support-center-xy", default="0.455 0.0")
    ap.add_argument("--support-half-size", default="0.045 0.045 0.115")
    ap.add_argument("--object-clearance", type=float, default=0.003)
    args = ap.parse_args()

    prior_dir = resolve(args.prior_dir)
    mesh_root = resolve(args.mesh_root)
    selector_json = resolve(args.selector_json)
    outroot = resolve(args.outroot)
    outroot.mkdir(parents=True, exist_ok=True)

    npy_path = prior_dir / f"{args.object_code}.npy"
    mesh_path = mesh_root / args.object_code / "coacd" / "decomposed.obj"

    required = [
        BUILDER, P2_SCRIPT, P3_SCRIPT, P4U6_SCRIPT,
        URDF, selector_json, npy_path, mesh_path
    ]
    missing = [rel(p) for p in required if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("missing required files:\\n" + "\\n".join(missing))

    rows_in = load_json(selector_json)
    if isinstance(rows_in, dict):
        rows_in = rows_in.get("selected") or rows_in.get("rows_sorted") or []
    rows_in = rows_in[: args.max_samples]

    rows = []

    for r in rows_in:
        local = int(r["valid_local_index"])
        raw = int(r.get("raw_sample_index", local))
        grasp_type = safe_name(r.get("grasp_type", "unknown"))
        selector_final_score = r.get("final_score")

        tag = f"local{local:03d}_raw{raw:03d}_{grasp_type}"
        odir = outroot / tag
        odir.mkdir(parents=True, exist_ok=True)

        row = {
            "tag": tag,
            "valid_local_index": local,
            "raw_sample_index": raw,
            "grasp_type": grasp_type,
            "selector_final_score": selector_final_score,
            "out_dir": rel(odir),
        }

        try:
            build_cmd = [
                "python3", rel(BUILDER),
                "--object-code", args.object_code,
                "--npy", rel(npy_path),
                "--sample-index", str(local),
                "--object-mesh", rel(mesh_path),
                "--out-dir", rel(odir),
                "--object-body", "grasp_object",
                "--support-center-xy", args.support_center_xy,
                "--support-half-size", args.support_half_size,
                "--support-top-z", str(args.support_top_z),
                "--object-clearance", str(args.object_clearance),
            ]

            rc_build = run_cmd(build_cmd, odir / "terminal_build.txt")
            row["build_return_code"] = rc_build

            scene = odir / "scene.xml"
            candidate = odir / "candidate.json"
            integrity = odir / "integrity_summary.json"

            if rc_build != 0 or not scene.exists() or not candidate.exists():
                row["error"] = f"build failed rc={rc_build}"
                rows.append(score_row(row))
                continue

            row["scene"] = rel(scene)
            row["candidate"] = rel(candidate)
            row["integrity"] = rel(integrity) if integrity.exists() else None

            p2_json = odir / "p2.json"
            p3_json = odir / "p3.json"
            plan_json = odir / "plan.json"

            p2_cmd = [
                "python3", rel(P2_SCRIPT),
                "--urdf", rel(URDF),
                "--model", rel(scene),
                "--candidate", rel(candidate),
            ]
            if RUNNER_JSON.exists():
                p2_cmd += ["--runner-json", rel(RUNNER_JSON)]
            p2_cmd += [
                "--object-body", "grasp_object",
                "--target-frame", "fr3_link7",
                "--out", rel(p2_json),
                "--random-seeds", "4",
                "--random-std", "0.50",
                "--max-iters", "260",
                "--pos-tol", "0.0006",
                "--rot-tol", "0.006",
                "--rot-weight", "0.48",
            ]

            rc_p2 = run_cmd(p2_cmd, odir / "terminal_p2.txt")
            row["p2_return_code"] = rc_p2
            row["p2_json"] = rel(p2_json)

            if rc_p2 != 0:
                row["error"] = f"P2 failed rc={rc_p2}"
                rows.append(score_row(row))
                continue

            p3_cmd = [
                "python3", rel(P3_SCRIPT),
                "--p2-json", rel(p2_json),
                "--model", rel(scene),
                "--candidate", rel(candidate),
                "--object-body", "grasp_object",
                "--out", rel(p3_json),
                "--best-plan-out", rel(plan_json),
                "--top-per-target", "4",
                "--max-combos", "128",
                "--path-samples", "18",
                "--min-hand-support-clearance", "0.0",
                "--min-fr3-object-clearance", "0.0",
                "--max-grasp-hand-object-distance", "0.070",
                "--min-joint-margin", "0.0",
            ]

            rc_p3 = run_cmd(p3_cmd, odir / "terminal_p3.txt")
            row["p3_return_code"] = rc_p3
            row["p3_json"] = rel(p3_json)
            row["plan_json"] = rel(plan_json)

            if rc_p3 != 0:
                row["error"] = f"P3 failed rc={rc_p3}"

            row["p3"] = summarize_p3(p3_json)

        except Exception as e:
            row["error"] = repr(e)
            row["traceback"] = traceback.format_exc()

        rows.append(score_row(row))

    rows_sorted = sorted(rows, key=lambda x: x.get("v413d_score", -1e9), reverse=True)

    executable = []
    for r in rows_sorted:
        p3 = r.get("p3", {})
        if not (r.get("scene") and r.get("candidate") and r.get("p3_json")):
            continue
        if p3.get("best_pass_exists"):
            executable.append(r)
            continue
        if (
            finite(p3.get("GO")) and -0.010 <= float(p3["GO"]) <= 0.012
            and finite(p3.get("HS")) and float(p3["HS"]) >= -0.006
            and finite(p3.get("FO")) and float(p3["FO"]) >= -0.006
        ):
            executable.append(r)

    best = executable[0] if executable else None

    if best:
        best_config = outroot / "best_config_selected.json"
        base_cfg, ctrl = make_best_config(PROJECT / best["candidate"], best_config)
        best["best_config"] = rel(best_config)
        best["best_config_base"] = rel(base_cfg)
        best["best_config_ctrl"] = ctrl
        viewer_script = make_viewer_script(best, best_config, outroot)
        best["viewer_script"] = rel(viewer_script)

    summary = {
        "format": "v4_13d_generic_topk_light_p2p3_debug_v1",
        "object_code": args.object_code,
        "selector_json": rel(selector_json),
        "npy": rel(npy_path),
        "mesh": rel(mesh_path),
        "outroot": rel(outroot),
        "rows_sorted": rows_sorted,
        "best": best,
    }

    save_json(outroot / "v4_13d_summary.json", summary)

    lines = []
    lines.append("========== V4.13D GENERIC TOPK -> LIGHT P2/P3 SUMMARY ==========")
    lines.append(f"object_code  : {args.object_code}")
    lines.append(f"selector_json: {rel(selector_json)}")
    lines.append(f"npy          : {rel(npy_path)}")
    lines.append(f"mesh         : {rel(mesh_path)}")
    lines.append(f"outroot      : {rel(outroot)}")
    lines.append("")

    for i, r in enumerate(rows_sorted, start=1):
        p3 = r.get("p3", {})
        lines.append(
            f"rank={i:02d} tag={r.get('tag')} "
            f"score={r.get('v413d_score'):.3f} "
            f"type={r.get('grasp_type')} local={r.get('valid_local_index')} raw={r.get('raw_sample_index')} "
            f"pass={p3.get('num_pass')} status={p3.get('status')} "
            f"HS={p3.get('HS')} HSc={p3.get('HSc')} FO={p3.get('FO')} GO={p3.get('GO')} "
            f"build_rc={r.get('build_return_code')} p2_rc={r.get('p2_return_code')} p3_rc={r.get('p3_return_code')} "
            f"err={r.get('error')}"
        )
        for rr in r.get("v413d_score_reasons", [])[:10]:
            lines.append(f"  - {rr}")
        for rr in (p3.get("hard_reasons") or [])[:3]:
            lines.append(f"  hard: {rr}")
        lines.append("")

    lines.append("---- BEST ----")
    if best:
        lines.append(f"best_tag: {best.get('tag')}")
        lines.append(f"scene: {best.get('scene')}")
        lines.append(f"candidate: {best.get('candidate')}")
        lines.append(f"p3_json: {best.get('p3_json')}")
        lines.append(f"best_config: {best.get('best_config')}")
        lines.append(f"viewer_script: {best.get('viewer_script')}")
        lines.append("")
        lines.append("run viewer:")
        lines.append(f"./{best.get('viewer_script')}")
    else:
        lines.append("best: None")
        lines.append("没有 PASS/near-pass；先看各 rank 的 terminal_p2.txt / terminal_p3.txt。")

    lines.append("=================================================================")
    txt = "\\n".join(lines) + "\\n"
    (outroot / "v4_13d_summary.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
