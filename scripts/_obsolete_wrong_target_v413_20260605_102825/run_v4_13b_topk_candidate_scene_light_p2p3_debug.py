#!/usr/bin/env python3
"""
脚本类型：
    debug / v4.13b-fixed / topk-prior-to-light-p2p3

用途：
    将 V4.13 通用 selector 输出的 Top-K 先验样本转成 candidate/scene，
    然后运行轻量 P2 IK 与 P3 collision precheck，最后按 P3 结果选择 rank1。
    本版本修复：
        1. --runner-json 插入位置错误导致 P2 参数错位；
        2. 当候选没有生成 p3_json 时仍强行生成 viewer 的 KeyError。

输入：
    selected_topk_compact.json
    object.npy
    object mesh
    build_sodacan165_initial_candidate_scene_debug.py
    run_v4_12p2_pinocchio_multiseed_ik_debug.py
    run_v4_12p3_pinocchio_ik_plus_mujoco_collision_precheck_debug.py

输出：
    diagnostics/current_v413/core_bottle_v413b_topk_p2p3_debug/
        */initial_debug/
        */p2.json
        */p3.json
        */plan.json
        v4_13b_summary.txt
        v4_13b_summary.json
    若存在可执行候选：
        scripts/05_execution_runner/run_v4_13b_selected_viewer_debug.sh

当前流程位置：
    五文件先验包 -> V4.13 selector -> 本脚本 Top-K light P2/P3 -> P4U6/P4U1 viewer

不负责：
    1. 不修改 legacy_final_demos；
    2. 不修改 P4U1/P4U6 源码；
    3. 不做全量慢筛；
    4. 不把 can 特例迁移写进通用流程。
"""

from pathlib import Path
import argparse
import copy
import json
import math
import subprocess
import traceback
import xml.etree.ElementTree as ET

PROJECT = Path.home() / "Projects/o7_mujoco_sim"

BUILDER = PROJECT / "scripts/00_maintenance/build_sodacan165_initial_candidate_scene_debug.py"
P2_SCRIPT = PROJECT / "scripts/07_precheck_ik_collision/run_v4_12p2_pinocchio_multiseed_ik_debug.py"
P3_SCRIPT = PROJECT / "scripts/07_precheck_ik_collision/run_v4_12p3_pinocchio_ik_plus_mujoco_collision_precheck_debug.py"
P4U6_SCRIPT = PROJECT / "scripts/05_execution_runner/run_v4_12p4u6_ik_path_record_demo_debug.py"

URDF = PROJECT / "models/fr3_o7/fr3_o7_mujoco_ready_obj.urdf"
RUNNER_JSON = PROJECT / "diagnostics/current_v412/v4_12a_can52_strict_clearance_debug_runner.json"

DEFAULT_TOPK_JSON = PROJECT / "diagnostics/current_v413/core_bottle_general_select_debug/selected_topk_compact.json"
DEFAULT_OUTROOT = PROJECT / "diagnostics/current_v413/core_bottle_v413b_topk_p2p3_debug"

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


def load_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, obj):
    path = Path(path)
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


def safe_name(s):
    return "".join(c if c.isalnum() or c in "_-." else "_" for c in str(s))


def find_first_scene(initial_dir):
    xs = sorted((initial_dir / "scene").glob("*.xml"))
    return xs[0] if xs else None


def find_first_candidate(initial_dir, sid):
    cand_dir = initial_dir / "candidates"
    exact = cand_dir / f"sample{sid}_candidate.json"
    if exact.exists():
        return exact
    xs = sorted(cand_dir.glob("*.json"))
    return xs[0] if xs else None


def detect_object_body(scene_path):
    try:
        tree = ET.parse(str(scene_path))
        root = tree.getroot()
        names = [b.attrib.get("name", "") for b in root.iter("body")]
    except Exception:
        return "grasp_can"

    for n in ["grasp_object", "grasp_bottle", "grasp_can", "grasp_mug", "grasp_box"]:
        if n in names:
            return n

    for n in names:
        if n.startswith("grasp_"):
            return n

    return "grasp_can"


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def summarize_p3(p3_path):
    p3_path = Path(p3_path)
    if not p3_path.exists():
        return {"exists": False}

    try:
        d = load_json(p3_path)
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
        s -= 500
        reasons.append("build_failed")
    if row.get("p2_return_code") not in [0, None]:
        s -= 300
        reasons.append("p2_failed")
    if row.get("p3_return_code") not in [0, None]:
        s -= 200
        reasons.append("p3_failed")

    if not p3.get("exists"):
        s -= 250
        reasons.append("no_p3_json")
        row["v413b_score"] = float(s)
        row["v413b_score_reasons"] = reasons
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

    row["v413b_score"] = float(s)
    row["v413b_score_reasons"] = reasons
    return row


def get_candidate_ctrl(candidate_path):
    d = load_json(candidate_path)
    ctrl = d.get("hand", {}).get("o7_active_ctrl")
    if not isinstance(ctrl, dict):
        ctrl = d.get("o7_active_ctrl")
    if not isinstance(ctrl, dict):
        raise RuntimeError(f"candidate missing o7_active_ctrl: {candidate_path}")

    out = {}
    for j in O7_ACTIVE_JOINTS:
        if j not in ctrl:
            raise RuntimeError(f"candidate ctrl missing {j}: {candidate_path}")
        out[j] = float(ctrl[j])
    return out


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
    cfg["best_record"]["hand_config"]["source"] = "v4_13b_candidate_ctrl_fixed"

    cfg["v4_13b_debug_note"] = {
        "type": "topk_general_prior_candidate_best_config_fixed",
        "base_config": rel(base_path),
        "candidate": rel(candidate_path),
        "reason": "runner reads best_record.hand_config.ctrl; use selected candidate O7 active ctrl.",
    }

    save_json(out_path, cfg)
    return base_path, ctrl


def make_viewer_script(best, best_config_path, outroot):
    script = PROJECT / "scripts/05_execution_runner/run_v4_13b_selected_viewer_debug.sh"
    viewer_out = outroot / "viewer_selected"

    model = best["scene"]
    candidate = best["candidate"]
    p3_json = best["p3_json"]
    object_body = best["object_body"]

    text = f'''#!/usr/bin/env bash
# 脚本类型：
#   debug / execution-runner / v4.13b-selected-viewer
#
# 用途：
#   运行 V4.13b 通用选择器选出的 rank1 candidate。
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

echo "========== V4.13B SELECTED VIEWER =========="
echo "valid_local_index : {best.get('valid_local_index')}"
echo "raw_sample_index  : {best.get('raw_sample_index')}"
echo "grasp_type        : {best.get('grasp_type')}"
echo "v413b_score       : {best.get('v413b_score')}"
echo "model             : {model}"
echo "candidate         : {candidate}"
echo "p3_json           : {p3_json}"
echo "object_body       : {object_body}"
echo "best_config       : {rel(best_config_path)}"
echo

./run_mujoco_clean.sh scripts/05_execution_runner/run_v4_12p4u6_ik_path_record_demo_debug.py \\
  --model "{model}" \\
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
  --micro-squeeze-duration 0.35 \\
  --micro-squeeze-fraction 0.02 \\
  --finger-close-scale 1.14 \\
  --thumb-pitch-from-finger-gain 0.24 \\
  --grip-ready-stable-steps 5 \\
  --min-live-non-thumb 1 \\
  --opposition-cos-threshold -0.30 \\
  --max-grip-disp 0.024 \\
  --max-extra-disp-during-squeeze 0.005 \\
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
    ap.add_argument("--prior-dir", default="dataset/O7_Full_V8BestBaseline_165objs_20260422_084834/validate_results/seed1")
    ap.add_argument("--mesh-root", default="dataset/meshdata")
    ap.add_argument("--topk-json", default=str(DEFAULT_TOPK_JSON))
    ap.add_argument("--outroot", default=str(DEFAULT_OUTROOT))
    ap.add_argument("--max-samples", type=int, default=5)
    args = ap.parse_args()

    prior_dir = PROJECT / args.prior_dir
    mesh_root = PROJECT / args.mesh_root
    topk_json = Path(args.topk_json)
    if not topk_json.is_absolute():
        topk_json = PROJECT / topk_json

    outroot = Path(args.outroot)
    if not outroot.is_absolute():
        outroot = PROJECT / outroot
    outroot.mkdir(parents=True, exist_ok=True)

    npy = prior_dir / f"{args.object_code}.npy"
    mesh = mesh_root / args.object_code / "coacd/decomposed.obj"

    required = [BUILDER, P2_SCRIPT, P3_SCRIPT, P4U6_SCRIPT, URDF, topk_json, npy, mesh]
    missing = [rel(p) for p in required if not Path(p).exists()]
    if missing:
        raise FileNotFoundError("missing required files:\\n" + "\\n".join(missing))

    rows_in = load_json(topk_json)
    if isinstance(rows_in, dict):
        rows_in = rows_in.get("selected") or rows_in.get("rows_sorted") or []
    rows_in = rows_in[: args.max_samples]

    rows = []

    for r in rows_in:
        local = int(r["valid_local_index"])
        raw = int(r.get("raw_sample_index", local))
        grasp_type = safe_name(r.get("grasp_type", "unknown"))

        sid = f"{local:03d}"
        tag = f"local{sid}_raw{raw:03d}_{grasp_type}"
        odir = outroot / tag
        initial_dir = odir / "initial_debug"
        odir.mkdir(parents=True, exist_ok=True)

        row = {
            "tag": tag,
            "valid_local_index": local,
            "raw_sample_index": raw,
            "grasp_type": grasp_type,
            "selector_final_score": r.get("final_score"),
            "selector_decision": r.get("decision"),
            "out_dir": rel(odir),
        }

        try:
            build_cmd = [
                "python3", rel(BUILDER),
                "--target", args.object_code,
                "--npy", rel(npy),
                "--object-mesh", rel(mesh),
                "--sample-indices", str(local),
                "--out-dir", rel(initial_dir),
            ]

            rc_build = run_cmd(build_cmd, odir / "terminal_build.txt")
            row["build_return_code"] = rc_build

            if rc_build != 0:
                row["error"] = f"build failed rc={rc_build}"
                rows.append(score_row(row))
                continue

            scene = find_first_scene(initial_dir)
            candidate = find_first_candidate(initial_dir, sid)

            if scene is None:
                raise RuntimeError("scene not generated")
            if candidate is None:
                raise RuntimeError("candidate not generated")

            object_body = detect_object_body(scene)

            row["scene"] = rel(scene)
            row["candidate"] = rel(candidate)
            row["object_body"] = object_body

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
                "--object-body", object_body,
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
                "--object-body", object_body,
                "--out", rel(p3_json),
                "--best-plan-out", rel(plan_json),
                "--top-per-target", "4",
                "--max-combos", "128",
                "--path-samples", "18",
                "--min-hand-support-clearance", "0.0",
                "--min-fr3-object-clearance", "0.0",
                "--max-grasp-hand-object-distance", "0.065",
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

    rows_sorted = sorted(rows, key=lambda x: x.get("v413b_score", -1e9), reverse=True)

    executable = [
        r for r in rows_sorted
        if r.get("scene") and r.get("candidate") and r.get("p3_json") and Path(PROJECT / r["p3_json"]).exists()
    ]
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
        "format": "v4_13b_topk_candidate_scene_light_p2p3_debug_fixed_v1",
        "object_code": args.object_code,
        "npy": rel(npy),
        "mesh": rel(mesh),
        "topk_json": rel(topk_json),
        "outroot": rel(outroot),
        "rows_sorted": rows_sorted,
        "best": best,
    }
    save_json(outroot / "v4_13b_summary.json", summary)

    lines = []
    lines.append("========== V4.13B FIXED TOPK -> CANDIDATE/SCENE -> LIGHT P2/P3 SUMMARY ==========")
    lines.append(f"object_code: {args.object_code}")
    lines.append(f"topk_json  : {rel(topk_json)}")
    lines.append(f"outroot    : {rel(outroot)}")
    lines.append("")

    for i, r in enumerate(rows_sorted, start=1):
        p3 = r.get("p3", {})
        lines.append(
            f"rank={i:02d} tag={r.get('tag')} "
            f"v413b_score={r.get('v413b_score'):.3f} "
            f"type={r.get('grasp_type')} "
            f"local={r.get('valid_local_index')} raw={r.get('raw_sample_index')} "
            f"pass={p3.get('num_pass')} status={p3.get('status')} "
            f"HS={p3.get('HS')} HSc={p3.get('HSc')} FO={p3.get('FO')} GO={p3.get('GO')} "
            f"build_rc={r.get('build_return_code')} p2_rc={r.get('p2_return_code')} p3_rc={r.get('p3_return_code')} "
            f"err={r.get('error')}"
        )
        for rr in r.get("v413b_score_reasons", [])[:10]:
            lines.append(f"  - {rr}")
        hard = p3.get("hard_reasons") or []
        for rr in hard[:3]:
            lines.append(f"  hard: {rr}")
        lines.append("")

    lines.append("---- BEST ----")
    if best:
        lines.append(f"best_tag: {best.get('tag')}")
        lines.append(f"scene: {best.get('scene')}")
        lines.append(f"candidate: {best.get('candidate')}")
        lines.append(f"object_body: {best.get('object_body')}")
        lines.append(f"p3_json: {best.get('p3_json')}")
        lines.append(f"best_config: {best.get('best_config')}")
        lines.append(f"viewer_script: {best.get('viewer_script')}")
        lines.append("")
        lines.append("run viewer:")
        lines.append(f"./{best.get('viewer_script')}")
    else:
        lines.append("best: None")
        lines.append("没有候选成功生成 scene + candidate + p3_json；先查看各目录 terminal_build.txt / terminal_p2.txt。")

    lines.append("===============================================================================")
    txt = "\\n".join(lines) + "\\n"
    (outroot / "v4_13b_summary.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
