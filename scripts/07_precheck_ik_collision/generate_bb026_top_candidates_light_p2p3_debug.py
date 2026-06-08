#!/usr/bin/env python3
"""
脚本类型：
    debug / generator / top-grasp / light-p2p3

用途：
    根据 scene-aware selector 的结果，对 BB026 SodaCan 生成 top grasp 候选。
    当前结论是：
        side grasp 候选整体被 support block；
        sample010 / sample014 是 top_like，可作为上抓 wrist 姿态种子。

    本脚本执行：
        1. 读取 scene_aware_select_summary.json；
        2. 只取 decision=KEEP_FOR_TOP_CHECK 的 top_like 样本；
        3. 基于这些 top_like candidate 的 wrist rotation 和 hand ctrl，
           生成少量不同 top approach 高度 / 横向偏移的 candidate；
        4. 对这些 candidate 运行轻量 P2/P3；
        5. 输出排序结果；
        6. 如果有 P3 pass，自动生成 viewer 脚本。

输入：
    diagnostics/current_v412/sodacan_bb026_grasptype_scene_aware_select_debug/scene_aware_select_summary.json
    diagnostics/current_v412/sodacan_bb026_topk_p2p3_batch_debug/sample010 或 sample014 下的 scene/candidate

输出：
    diagnostics/current_v412/sodacan_bb026_top_grasp_light_debug/
        candidates/
        scenes/
        p2p3/
        top_grasp_summary.txt
        top_grasp_summary.json
    scripts/05_execution_runner/run_sodacan_bb026_top_selected_viewer_debug.sh

当前流程位置：
    side grasp 被 support block
        -> top grasp candidate generator
        -> light P2/P3
        -> viewer 验证

不负责：
    1. 不修改 legacy_final_demos；
    2. 不修改 P4U1/P4U6 源码；
    3. 不跑全量慢筛选；
    4. 不保证第一次 top candidate 一定抓起；
    5. 不把 sample010/014 当成最终姿态，只把它们作为 top-like wrist 种子。
"""

from pathlib import Path
import copy
import json
import math
import os
import subprocess
import traceback
import xml.etree.ElementTree as ET

import numpy as np


PROJECT = Path.home() / "Projects/o7_mujoco_sim"

SUMMARY_JSON = PROJECT / "diagnostics/current_v412/sodacan_bb026_grasptype_scene_aware_select_debug/scene_aware_select_summary.json"
OUTROOT = PROJECT / "diagnostics/current_v412/sodacan_bb026_top_grasp_light_debug"

P2_SCRIPT = PROJECT / "scripts/07_precheck_ik_collision/run_v4_12p2_pinocchio_multiseed_ik_debug.py"
P3_SCRIPT = PROJECT / "scripts/07_precheck_ik_collision/run_v4_12p3_pinocchio_ik_plus_mujoco_collision_precheck_debug.py"

URDF = PROJECT / "models/fr3_o7/fr3_o7_mujoco_ready_obj.urdf"
RUNNER_JSON = PROJECT / "diagnostics/current_v412/v4_12a_can52_strict_clearance_debug_runner.json"

OBJECT_BODY = "grasp_can"
TARGET_BODY = "fr3_link7"

# 控制速度：默认只测 12 个 top candidate。
MAX_CANDIDATES = int(os.environ.get("MAX_TOP_CANDIDATES", "12"))


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
        p = subprocess.run(cmd, cwd=str(PROJECT), stdout=f, stderr=subprocess.STDOUT, text=True)
    return p.returncode


def find_scene(sample_dir):
    xs = sorted((sample_dir / "initial_debug/scene").glob("*.xml"))
    return xs[0] if xs else None


def find_candidate(sample_dir, sid):
    p = sample_dir / f"initial_debug/candidates/sample{sid}_candidate.json"
    if p.exists():
        return p
    xs = sorted((sample_dir / "initial_debug/candidates").glob("*.json"))
    return xs[0] if xs else None


def get_T(candidate):
    paths = [
        ("target", "T_object_target"),
        ("target", "T_object_fr3_link7"),
        ("T_object_target",),
        ("T_object_fr3_link7",),
    ]
    for path in paths:
        cur = candidate
        ok = True
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok:
            arr = np.asarray(cur, dtype=float)
            if arr.shape == (4, 4):
                return arr, path
    raise RuntimeError("candidate missing target T_object_target / T_object_fr3_link7")


def set_T(candidate, path, T):
    cur = candidate
    for k in path[:-1]:
        cur = cur[k]
    cur[path[-1]] = T.tolist()


def patch_scene_to_settled_pose(scene_in, scene_out, audit_row):
    """
    使用 audit 中 object_pos1 - object_pos0 的 delta，把 object body 初始位置 patch 到稳定后位置。
    """
    settle = audit_row.get("settle", {})
    p0 = settle.get("object_pos0")
    p1 = settle.get("object_pos1")

    if not p0 or not p1:
        scene_out.write_text(scene_in.read_text())
        return {"patched": False, "reason": "no settle p0/p1"}

    delta = np.asarray(p1, dtype=float) - np.asarray(p0, dtype=float)

    tree = ET.parse(str(scene_in))
    root = tree.getroot()

    body = None
    for b in root.iter("body"):
        if b.attrib.get("name") == OBJECT_BODY:
            body = b
            break

    if body is None:
        raise RuntimeError(f"cannot find object body: {OBJECT_BODY}")

    old_pos = np.asarray([float(x) for x in body.attrib.get("pos", "0 0 0").split()], dtype=float)
    new_pos = old_pos + delta

    body.set("pos", f"{new_pos[0]:.12g} {new_pos[1]:.12g} {new_pos[2]:.12g}")

    scene_out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(scene_out), encoding="utf-8", xml_declaration=True)

    return {
        "patched": True,
        "old_pos": old_pos.tolist(),
        "delta": delta.tolist(),
        "new_pos": new_pos.tolist(),
    }


def Rz(yaw):
    c = math.cos(yaw)
    s = math.sin(yaw)
    R = np.eye(3)
    R[0, 0] = c
    R[0, 1] = -s
    R[1, 0] = s
    R[1, 1] = c
    return R


def generate_top_variants(seed_row):
    """
    使用 top_like seed 的 wrist rotation，不凭空猜 hand frame；
    只改变 T_object_target 的平移，轻微 yaw sweep。
    """
    sid = str(seed_row["sample"]).zfill(3)
    sample_dir = PROJECT / seed_row["sample_dir"]
    cand_path = Path(seed_row["candidate"])
    if not cand_path.is_absolute():
        cand_path = PROJECT / cand_path

    base = load_json(cand_path)
    T0, t_path = get_T(base)

    p0 = T0[:3, 3].copy()
    R0 = T0[:3, :3].copy()

    xy = p0[:2]
    norm_xy = float(np.linalg.norm(xy))
    if norm_xy < 1e-6:
        direction = np.array([1.0, 0.0])
    else:
        direction = xy / norm_xy

    # top grasp：目标点保持在物体上方，radial 不要太大，z 比原 top_like 更可控。
    radial_values = [0.035, 0.070, 0.105]
    z_values = [0.245, 0.285, 0.325]
    yaw_values_deg = [0.0, 35.0, -35.0]

    variants = []
    for radial in radial_values:
        for z in z_values:
            for yaw_deg in yaw_values_deg:
                cand = copy.deepcopy(base)

                T = T0.copy()
                T[:3, 3] = np.array([direction[0] * radial, direction[1] * radial, z], dtype=float)
                T[:3, :3] = Rz(math.radians(yaw_deg)) @ R0

                set_T(cand, t_path, T)

                cand["candidate_name"] = (
                    f"bb026_top_from_sample{sid}_"
                    f"r{radial:.3f}_z{z:.3f}_yaw{int(yaw_deg):+d}_debug"
                )
                cand.setdefault("source", {})
                cand["source"]["type"] = "bb026_top_grasp_light_generator_debug"
                cand["source"]["seed_sample"] = int(sid)
                cand["source"]["top_variant"] = {
                    "radial": radial,
                    "z": z,
                    "yaw_deg": yaw_deg,
                    "seed_T_object_target_translation": p0.tolist(),
                    "meaning": "reuse top-like wrist rotation and hand ctrl; sweep top approach translation/yaw",
                }

                cand.setdefault("execution", {})
                cand["execution"]["grasp_type"] = "top_grasp_candidate_debug"

                # 启发式：优先中等高度和中等 radial。
                heuristic = abs(radial - 0.070) + 0.8 * abs(z - 0.285) + 0.001 * abs(yaw_deg)

                variants.append({
                    "seed_sample": sid,
                    "radial": radial,
                    "z": z,
                    "yaw_deg": yaw_deg,
                    "heuristic": float(heuristic),
                    "candidate": cand,
                })

    return sorted(variants, key=lambda x: x["heuristic"])


def summarize_p3(path):
    if not path.exists():
        return {"exists": False}

    try:
        d = load_json(path)
    except Exception as e:
        return {"exists": True, "parse_ok": False, "error": repr(e)}

    bp = d.get("best_pass")
    ba = d.get("best_available") or {}
    chosen = bp if bp is not None else ba

    return {
        "exists": True,
        "parse_ok": True,
        "num_combos": d.get("num_combos"),
        "num_pass": d.get("num_pass"),
        "best_pass_exists": bp is not None,
        "status": chosen.get("precheck_status") if isinstance(chosen, dict) else None,
        "score": chosen.get("score") if isinstance(chosen, dict) else None,
        "HS": chosen.get("min_path_hand_support_clearance") if isinstance(chosen, dict) else None,
        "FO": chosen.get("min_path_fr3_object_clearance") if isinstance(chosen, dict) else None,
        "GO": chosen.get("static_grasp_closed_hand_object_distance") if isinstance(chosen, dict) else None,
        "HSc": chosen.get("static_grasp_closed_hand_support_clearance") if isinstance(chosen, dict) else None,
        "margin": chosen.get("combo_min_joint_margin") if isinstance(chosen, dict) else None,
        "hard_reasons": chosen.get("hard_reasons") if isinstance(chosen, dict) else None,
    }


def score_result(row):
    p3 = row.get("p3", {})
    s = 0.0

    if p3.get("best_pass_exists"):
        s += 100
    n = p3.get("num_pass")
    if isinstance(n, int):
        s += min(n, 100)

    HS = p3.get("HS")
    FO = p3.get("FO")
    GO = p3.get("GO")
    HSc = p3.get("HSc")
    margin = p3.get("margin")

    if isinstance(HS, (int, float)):
        if HS >= 0:
            s += 30
        elif HS >= -0.006:
            s += 5
        else:
            s -= 30

    if isinstance(HSc, (int, float)):
        if HSc >= 0:
            s += 20
        elif HSc < -0.012:
            s -= 20

    if isinstance(FO, (int, float)):
        if FO >= 0.003:
            s += 20
        elif FO >= 0:
            s += 5
        else:
            s -= 20

    if isinstance(GO, (int, float)):
        if 0.001 <= GO <= 0.06:
            s += 20
        elif GO < 0:
            s -= 10

    if isinstance(margin, (int, float)):
        s += min(max(margin, 0.0), 1.0) * 5

    s -= 20.0 * row.get("heuristic", 0.0)
    return float(s)


def make_best_config_from_candidate(candidate_path, out_path):
    base_candidates = [
        PROJECT / "legacy_final_demos/final_sodacan165_sample014_p4u6_ready_gated_snap_demo/inputs/best_config.json",
        PROJECT / "diagnostics/current_v412/v4_12p4h2_can52_contact_sequence_proxy_debug/best_contact_sequence_config_ctrlsplit_debug.json",
    ]

    base_path = next((p for p in base_candidates if p.exists()), None)
    if base_path is None:
        raise FileNotFoundError("cannot find base best_config")

    cfg = load_json(base_path)
    cand = load_json(candidate_path)
    ctrl = cand.get("hand", {}).get("o7_active_ctrl")
    if not isinstance(ctrl, dict):
        raise RuntimeError("candidate missing hand.o7_active_ctrl")

    cfg.setdefault("best_record", {})
    cfg["best_record"].setdefault("hand_config", {})
    cfg["best_record"]["hand_config"]["ctrl"] = {k: float(v) for k, v in ctrl.items()}
    cfg["best_record"]["hand_config"]["source"] = "bb026_top_candidate_ctrl_debug"
    cfg["generalization_debug_note"] = {
        "type": "bb026_top_best_config_from_candidate",
        "base_config": rel(base_path),
        "candidate": rel(candidate_path),
    }

    save_json(out_path, cfg)
    return base_path


def make_viewer_script(best, best_config):
    script = PROJECT / "scripts/05_execution_runner/run_sodacan_bb026_top_selected_viewer_debug.sh"

    text = f'''#!/usr/bin/env bash
# 脚本类型：
#   debug / execution-runner / top-grasp-viewer
#
# 用途：
#   对 BB026 top grasp generator 选出的候选运行 P4U6 + P4U1 viewer。
#
# 输入：
#   top generator 生成的 stable_scene、candidate、p3、best_config。
#
# 输出：
#   diagnostics/current_v412/sodacan_bb026_top_grasp_light_debug/viewer_selected/
#       terminal.txt
#       result.json
#       path_plan.json
#
# 不负责：
#   不重新筛选，不修改 legacy demo，不修改 P4U1/P4U6 源码。

set -euo pipefail

cd "$HOME/Projects/o7_mujoco_sim"
source "$HOME/mujoco_env/bin/activate"

OUTDIR="diagnostics/current_v412/sodacan_bb026_top_grasp_light_debug/viewer_selected"
mkdir -p "$OUTDIR"

LOG="$OUTDIR/terminal.txt"
RESULT_JSON="$OUTDIR/result.json"
PATH_PLAN_JSON="$OUTDIR/path_plan.json"

exec > >(tee "$LOG") 2>&1

echo "========== BB026 TOP SELECTED VIEWER =========="
echo "seed_sample : {best['seed_sample']}"
echo "candidate   : {best['candidate_path']}"
echo "model       : {best['stable_scene']}"
echo "p3_json     : {best['p3_json']}"
echo "best_config : {rel(best_config)}"
echo

./run_mujoco_clean.sh scripts/05_execution_runner/run_v4_12p4u6_ik_path_record_demo_debug.py \\
  --model "{best['stable_scene']}" \\
  --candidate "{best['candidate_path']}" \\
  --p3-json "{best['p3_json']}" \\
  --best-config "{rel(best_config)}" \\
  --which best_available \\
  --object-body grasp_can \\
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
  --micro-squeeze-fraction 0.00 \\
  --finger-close-scale 1.12 \\
  --thumb-pitch-from-finger-gain 0.24 \\
  --grip-ready-stable-steps 5 \\
  --min-live-non-thumb 1 \\
  --opposition-cos-threshold -0.30 \\
  --max-grip-disp 0.022 \\
  --max-extra-disp-during-squeeze 0.004 \\
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
p = Path("diagnostics/current_v412/sodacan_bb026_top_grasp_light_debug/viewer_selected/result.json")
if p.exists():
    d = json.loads(p.read_text())
    for k in ["success","stop_reason","grip_ready","final_object_disp","final_object_rise","max_object_rise","final_groups","final_opposition_cos","max_stable_count"]:
        if k in d:
            print(f"{{k}}: {{d[k]}}")
R
'''
    script.write_text(text)
    script.chmod(0o755)
    return script


def main():
    OUTROOT.mkdir(parents=True, exist_ok=True)

    if not SUMMARY_JSON.exists():
        raise FileNotFoundError(SUMMARY_JSON)

    for f in [P2_SCRIPT, P3_SCRIPT, URDF]:
        if not f.exists():
            raise FileNotFoundError(f)

    summary = load_json(SUMMARY_JSON)

    seed_rows = []
    for r in summary.get("rows_sorted", []):
        if r.get("decision") == "KEEP_FOR_TOP_CHECK" and r.get("grasp_type") == "top_like":
            seed_rows.append(r)

    if not seed_rows:
        raise RuntimeError("no top_like KEEP_FOR_TOP_CHECK rows found")

    # 先为每个 seed 准备 stable scene
    stable_scenes = {}
    for r in seed_rows:
        sid = str(r["sample"]).zfill(3)
        sample_dir = PROJECT / r["sample_dir"]
        scene_in = find_scene(sample_dir)
        if scene_in is None:
            raise RuntimeError(f"missing scene for sample {sid}")

        scene_out = OUTROOT / "scenes" / f"stable_scene_seed{sid}.xml"
        patch_info = patch_scene_to_settled_pose(scene_in, scene_out, r.get("audit", {}))
        stable_scenes[sid] = scene_out
        save_json(OUTROOT / "scenes" / f"stable_scene_seed{sid}_patch_info.json", patch_info)

    # 生成候选，按启发式只保留前 MAX_CANDIDATES
    variants = []
    for r in seed_rows:
        variants.extend(generate_top_variants(r))

    variants = sorted(variants, key=lambda x: x["heuristic"])[:MAX_CANDIDATES]

    rows = []

    for i, v in enumerate(variants):
        idx = i + 1
        sid = v["seed_sample"]

        tag = f"{idx:03d}_seed{sid}_r{v['radial']:.3f}_z{v['z']:.3f}_yaw{int(v['yaw_deg']):+d}"
        cand_path = OUTROOT / "candidates" / f"{tag}.json"
        p2_json = OUTROOT / "p2p3" / tag / "p2.json"
        p3_json = OUTROOT / "p2p3" / tag / "p3.json"
        plan_json = OUTROOT / "p2p3" / tag / "plan.json"
        log_p2 = OUTROOT / "p2p3" / tag / "terminal_p2.txt"
        log_p3 = OUTROOT / "p2p3" / tag / "terminal_p3.txt"

        save_json(cand_path, v["candidate"])

        row = {
            "tag": tag,
            "seed_sample": sid,
            "radial": v["radial"],
            "z": v["z"],
            "yaw_deg": v["yaw_deg"],
            "heuristic": v["heuristic"],
            "candidate_path": rel(cand_path),
            "stable_scene": rel(stable_scenes[sid]),
            "p2_json": rel(p2_json),
            "p3_json": rel(p3_json),
            "plan_json": rel(plan_json),
        }

        try:
            cmd_p2 = [
                "python3", rel(P2_SCRIPT),
                "--urdf", rel(URDF),
                "--model", rel(stable_scenes[sid]),
                "--candidate", rel(cand_path),
                "--runner-json", rel(RUNNER_JSON) if RUNNER_JSON.exists() else "",
                "--object-body", OBJECT_BODY,
                "--target-frame", TARGET_BODY,
                "--out", rel(p2_json),
                "--random-seeds", "3",
                "--random-std", "0.45",
                "--max-iters", "220",
                "--pos-tol", "0.0007",
                "--rot-tol", "0.007",
                "--rot-weight", "0.45",
            ]

            rc2 = run_cmd(cmd_p2, log_p2)
            row["p2_return_code"] = rc2

            if rc2 != 0:
                row["error"] = f"P2 failed rc={rc2}"
                rows.append(row)
                continue

            cmd_p3 = [
                "python3", rel(P3_SCRIPT),
                "--p2-json", rel(p2_json),
                "--model", rel(stable_scenes[sid]),
                "--candidate", rel(cand_path),
                "--object-body", OBJECT_BODY,
                "--out", rel(p3_json),
                "--best-plan-out", rel(plan_json),
                "--top-per-target", "3",
                "--max-combos", "48",
                "--path-samples", "10",
                "--min-hand-support-clearance", "0.0",
                "--min-fr3-object-clearance", "0.0",
                "--max-grasp-hand-object-distance", "0.075",
                "--min-joint-margin", "0.0",
            ]

            rc3 = run_cmd(cmd_p3, log_p3)
            row["p3_return_code"] = rc3
            row["p3"] = summarize_p3(p3_json)

        except Exception as e:
            row["error"] = repr(e)
            row["traceback"] = traceback.format_exc()

        row["selector_score"] = score_result(row)
        rows.append(row)

    rows_sorted = sorted(rows, key=lambda x: x.get("selector_score", -1e9), reverse=True)

    best = rows_sorted[0] if rows_sorted else None
    viewer_script = None
    best_config = None

    if best and best.get("candidate_path") and best.get("p3_json"):
        # 即使没有 pass，也生成 best_config，但只有 pass>0 时才建议 viewer。
        best_config = OUTROOT / "best_config_from_top_candidate.json"
        make_best_config_from_candidate(PROJECT / best["candidate_path"], best_config)
        best["best_config"] = rel(best_config)

        p3 = best.get("p3", {})
        if p3.get("best_pass_exists") or (isinstance(p3.get("num_pass"), int) and p3.get("num_pass") > 0):
            viewer_script = make_viewer_script(best, best_config)
            best["viewer_script"] = rel(viewer_script)

    final = {
        "format": "bb026_top_grasp_light_debug_v1",
        "max_candidates": MAX_CANDIDATES,
        "seed_samples": [str(r["sample"]).zfill(3) for r in seed_rows],
        "rows_sorted": rows_sorted,
        "best": best,
    }

    save_json(OUTROOT / "top_grasp_summary.json", final)

    lines = []
    lines.append("========== BB026 TOP GRASP LIGHT SUMMARY ==========")
    lines.append(f"seed_samples: {[str(r['sample']).zfill(3) for r in seed_rows]}")
    lines.append(f"max_candidates: {MAX_CANDIDATES}")
    lines.append("")

    for r in rows_sorted:
        p3 = r.get("p3", {})
        lines.append(
            f"tag={r.get('tag')} "
            f"score={r.get('selector_score')} "
            f"seed={r.get('seed_sample')} "
            f"radial={r.get('radial')} z={r.get('z')} yaw={r.get('yaw_deg')} "
            f"pass={p3.get('num_pass')} status={p3.get('status')} "
            f"HS={p3.get('HS')} FO={p3.get('FO')} GO={p3.get('GO')} HSc={p3.get('HSc')} "
            f"p2_rc={r.get('p2_return_code')} p3_rc={r.get('p3_return_code')} "
            f"err={r.get('error')}"
        )
        reasons = p3.get("hard_reasons") or []
        for rr in reasons[:3]:
            lines.append(f"  - {rr}")

    lines.append("")
    lines.append("---- BEST ----")
    if best:
        lines.append(f"best_tag: {best.get('tag')}")
        lines.append(f"candidate: {best.get('candidate_path')}")
        lines.append(f"scene: {best.get('stable_scene')}")
        lines.append(f"p3: {best.get('p3_json')}")
        lines.append(f"best_config: {best.get('best_config')}")
        if best.get("viewer_script"):
            lines.append(f"viewer_script: {best.get('viewer_script')}")
            lines.append("run viewer:")
            lines.append(f"./{best.get('viewer_script')}")
        else:
            lines.append("viewer_script: None")
            lines.append("no P3 pass yet; send this summary back for next adjustment.")
    else:
        lines.append("best: None")

    lines.append("===================================================")

    txt = "\n".join(lines) + "\n"
    (OUTROOT / "top_grasp_summary.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
