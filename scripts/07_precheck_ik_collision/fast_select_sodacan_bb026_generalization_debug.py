#!/usr/bin/env python3
"""
脚本类型：
    debug / fast-selector / generalization

用途：
    对 sem-SodaCan-bb0262... 做快速泛化筛选。
    目标是不再对所有 sample 跑完整慢 P2/P3，而是：
        1. 读取已有 Top-K candidate / P3 / audit；
        2. 用成功 SodaCan165 demo 作为 can-like 抓型参考；
        3. 对 BB026 candidate 做抓型相似度 + 支撑风险 + 已有 P3 快速评分；
        4. 只对 Top-N 做稳定 scene patch；
        5. 只对 Top-N 做轻量 P2/P3；
        6. 生成一个可直接 viewer 验证的 run 脚本。

输入：
    diagnostics/current_v412/sodacan_bb026_topk_p2p3_batch_debug/sample*/
    diagnostics/current_v412/sodacan_bb026_grasptype_scene_audit_debug/audit_summary.json
    legacy_final_demos/final_sodacan165_sample014_p4u6_ready_gated_snap_demo/inputs/candidate.json

输出：
    diagnostics/current_v412/sodacan_bb026_fast_generalization_debug/
        fast_rank_before_p2p3.txt
        fast_rank_before_p2p3.json
        sampleXXX/
            stable_scene.xml
            sampleXXX_fast_p2.json
            sampleXXX_fast_p3.json
            sampleXXX_fast_plan.json
            terminal_p2.txt
            terminal_p3.txt
        fast_final_summary.txt
        fast_final_summary.json
        best_config_from_candidate.json
    scripts/05_execution_runner/run_sodacan_bb026_fast_selected_viewer_debug.sh

当前流程位置：
    快速泛化主线：
        Top-K candidate
        -> fast grasp-type / support-aware ranking
        -> Top-N light P2/P3
        -> P4U6/P4U1 viewer

不负责：
    1. 不修改 legacy_final_demos；
    2. 不修改 P4U1/P4U6 源码；
    3. 不做全量慢筛选；
    4. 不保证第一个选出的 sample 一定成功，只负责快速缩小搜索范围。
"""

from pathlib import Path
import copy
import json
import math
import subprocess
import traceback
import xml.etree.ElementTree as ET

import numpy as np


PROJECT = Path.home() / "Projects/o7_mujoco_sim"

INROOT = PROJECT / "diagnostics/current_v412/sodacan_bb026_topk_p2p3_batch_debug"
AUDIT_JSON = PROJECT / "diagnostics/current_v412/sodacan_bb026_grasptype_scene_audit_debug/audit_summary.json"
OUTROOT = PROJECT / "diagnostics/current_v412/sodacan_bb026_fast_generalization_debug"

REFERENCE_CANDIDATE = PROJECT / "legacy_final_demos/final_sodacan165_sample014_p4u6_ready_gated_snap_demo/inputs/candidate.json"

P2_SCRIPT = PROJECT / "scripts/07_precheck_ik_collision/run_v4_12p2_pinocchio_multiseed_ik_debug.py"
P3_SCRIPT = PROJECT / "scripts/07_precheck_ik_collision/run_v4_12p3_pinocchio_ik_plus_mujoco_collision_precheck_debug.py"
P4U6_SCRIPT = PROJECT / "scripts/05_execution_runner/run_v4_12p4u6_ik_path_record_demo_debug.py"

URDF = PROJECT / "models/fr3_o7/fr3_o7_mujoco_ready_obj.urdf"
RUNNER_JSON = PROJECT / "diagnostics/current_v412/v4_12a_can52_strict_clearance_debug_runner.json"

OBJECT_BODY = "grasp_can"
TARGET_BODY = "fr3_link7"

TOP_N_LIGHT_P2P3 = 3

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
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(path).write_text(json.dumps(obj, indent=2, ensure_ascii=False))


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


def get_T_object_target(candidate):
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
                return arr
    return None


def get_ctrl(candidate):
    hand = candidate.get("hand", {})
    ctrl = hand.get("o7_active_ctrl")
    if not isinstance(ctrl, dict):
        ctrl = candidate.get("o7_active_ctrl", {})
    return {j: float(ctrl[j]) for j in O7_ACTIVE_JOINTS if j in ctrl}


def rot_distance(R1, R2):
    try:
        x = (np.trace(R1.T @ R2) - 1.0) * 0.5
        x = float(np.clip(x, -1.0, 1.0))
        return float(math.acos(x))
    except Exception:
        return math.pi


def candidate_features(candidate):
    T = get_T_object_target(candidate)
    if T is None:
        return None

    p = T[:3, 3]
    R = T[:3, :3]
    ctrl = get_ctrl(candidate)

    fingers = [ctrl.get(j, 0.0) for j in ["index_mcp_pitch", "middle_mcp_pitch", "ring_mcp_pitch", "pinky_mcp_pitch"]]

    return {
        "p": p,
        "R": R,
        "radial_xy": float(np.linalg.norm(p[:2])),
        "z": float(p[2]),
        "ctrl": ctrl,
        "finger_mean": float(np.mean(fingers)) if fingers else None,
        "thumb_yaw": ctrl.get("thumb_cmc_yaw"),
        "thumb_roll": ctrl.get("thumb_cmc_roll"),
        "thumb_pitch": ctrl.get("thumb_cmc_pitch"),
    }


def summarize_p3(p3_path):
    if not p3_path.exists():
        return {}
    try:
        d = load_json(p3_path)
    except Exception:
        return {}

    ba = d.get("best_available") or {}
    bp = d.get("best_pass")
    chosen = bp if bp is not None else ba

    return {
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


def load_audit_rows():
    if not AUDIT_JSON.exists():
        return {}
    d = load_json(AUDIT_JSON)
    out = {}
    for r in d.get("rows_sorted", []):
        out[str(r.get("sample")).zfill(3)] = r
    return out


def score_fast(row, ref_feat):
    score = 0.0
    reasons = []

    feat = row.get("features")
    p3 = row.get("p3", {})
    audit = row.get("audit", {})

    if feat is None:
        row["fast_score"] = -1e9
        row["fast_reasons"] = ["missing_T_object_target"]
        return

    # 1. can-like 抓型参考：优先和已成功 sodacan165 sample014 的 hand target 相似。
    if ref_feat is not None:
        pos_err = float(np.linalg.norm(feat["p"] - ref_feat["p"]))
        rot_err = rot_distance(feat["R"], ref_feat["R"])
        radial_err = abs(feat["radial_xy"] - ref_feat["radial_xy"])
        z_err = abs(feat["z"] - ref_feat["z"])

        score += max(0.0, 35.0 - 350.0 * pos_err)
        score += max(0.0, 20.0 - 10.0 * rot_err)
        score += max(0.0, 10.0 - 200.0 * radial_err)
        score += max(0.0, 10.0 - 200.0 * z_err)

        reasons.append(f"ref_pos_err={pos_err:.5f}")
        reasons.append(f"ref_rot_err={rot_err:.3f}")
        reasons.append(f"ref_radial_err={radial_err:.5f}")
        reasons.append(f"ref_z_err={z_err:.5f}")

    # 2. 不鼓励太上方戳取：hand target radial 太小或 z 过高都扣分。
    if feat["radial_xy"] < 0.03:
        score -= 30
        reasons.append(f"reject_like_top_poke_radial={feat['radial_xy']:.5f}")
    else:
        score += 10
        reasons.append(f"side_radial_ok={feat['radial_xy']:.5f}")

    if abs(feat["z"]) > 0.12:
        score -= 10
        reasons.append(f"target_z_large={feat['z']:.5f}")
    else:
        score += 5
        reasons.append(f"target_z_reasonable={feat['z']:.5f}")

    # 3. hand prior 不要太离谱。
    fm = feat.get("finger_mean")
    if fm is not None:
        if 0.35 <= fm <= 0.85:
            score += 8
            reasons.append(f"finger_mean_ok={fm:.3f}")
        else:
            score -= 8
            reasons.append(f"finger_mean_out={fm:.3f}")

    # 4. 用已有 P3 信息快速加权，但不把 pass=0 作为绝对淘汰。
    num_pass = p3.get("num_pass")
    if isinstance(num_pass, int) and num_pass > 0:
        score += min(num_pass, 200) * 0.4 + 20
        reasons.append(f"old_p3_pass={num_pass}")
    else:
        reasons.append("old_p3_pass_zero")

    FO = p3.get("FO")
    GO = p3.get("GO")
    HS = p3.get("HS")
    HSc = p3.get("HSc")
    margin = p3.get("margin")

    if isinstance(FO, (int, float)):
        if FO >= 0.003:
            score += 15
            reasons.append(f"FO_good={FO:.5f}")
        elif FO >= 0:
            score += 4
            reasons.append(f"FO_near={FO:.5f}")
        else:
            score -= 20
            reasons.append(f"FO_bad={FO:.5f}")

    if isinstance(GO, (int, float)):
        if 0.001 <= GO <= 0.012:
            score += 15
            reasons.append(f"GO_good={GO:.5f}")
        elif GO < 0:
            score -= 12
            reasons.append(f"GO_penetration={GO:.5f}")
        else:
            reasons.append(f"GO={GO:.5f}")

    if isinstance(HS, (int, float)):
        if HS >= 0:
            score += 12
            reasons.append(f"HS_clear={HS:.5f}")
        elif HS >= -0.018:
            score -= 2
            reasons.append(f"HS_mild={HS:.5f}")
        else:
            score -= 12
            reasons.append(f"HS_bad={HS:.5f}")

    if isinstance(HSc, (int, float)) and HSc < -0.025:
        score -= 8
        reasons.append(f"HSc_bad={HSc:.5f}")

    if isinstance(margin, (int, float)):
        score += min(max(margin, 0.0), 1.0) * 5
        reasons.append(f"margin={margin:.3f}")

    # 5. audit 只用于提示 scene 稳定化，不直接淘汰。
    settle = audit.get("settle", {})
    if settle:
        rise = settle.get("object_rise")
        disp = settle.get("object_disp")
        reasons.append(f"audit_settle_rise={rise}")
        reasons.append(f"audit_settle_disp={disp}")

    row["fast_score"] = float(score)
    row["fast_reasons"] = reasons


def patch_scene_by_audit_delta(scene_in, scene_out, audit_row):
    settle = audit_row.get("settle", {})
    p0 = settle.get("object_pos0")
    p1 = settle.get("object_pos1")

    if not p0 or not p1:
        # 不知道 delta 就直接复制
        scene_out.write_text(scene_in.read_text())
        return {"patched": False, "reason": "no audit p0/p1"}

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
        "old_body_pos": old_pos.tolist(),
        "delta": delta.tolist(),
        "new_body_pos": new_pos.tolist(),
    }


def make_best_config_from_candidate(candidate_path, out_path):
    base_candidates = [
        PROJECT / "legacy_final_demos/final_sodacan165_sample014_p4u6_ready_gated_snap_demo/inputs/best_config.json",
        PROJECT / "diagnostics/current_v412/v4_12p4h2_can52_contact_sequence_proxy_debug/best_contact_sequence_config_ctrlsplit_debug.json",
    ]
    base = None
    base_path = None
    for p in base_candidates:
        if p.exists():
            base = load_json(p)
            base_path = p
            break

    if base is None:
        raise FileNotFoundError("cannot find base best_config")

    cand = load_json(candidate_path)
    ctrl = get_ctrl(cand)
    if len(ctrl) < 7:
        raise RuntimeError(f"candidate ctrl incomplete: {ctrl}")

    cfg = copy.deepcopy(base)
    cfg.setdefault("best_record", {})
    cfg["best_record"].setdefault("hand_config", {})
    cfg["best_record"]["hand_config"]["ctrl"] = ctrl
    cfg["best_record"]["hand_config"]["source"] = "bb026_fast_selected_candidate_ctrl"

    cfg["generalization_debug_note"] = {
        "type": "bb026_fast_selected_best_config",
        "base_config": rel(base_path),
        "candidate": rel(candidate_path),
        "ctrl": ctrl,
    }

    save_json(out_path, cfg)
    return ctrl, base_path


def make_viewer_script(best_row, best_config_path):
    script = PROJECT / "scripts/05_execution_runner/run_sodacan_bb026_fast_selected_viewer_debug.sh"

    model = best_row["stable_scene"]
    candidate = best_row["candidate"]
    p3_json = best_row["fast_p3_json"]

    text = f'''#!/usr/bin/env bash
# 脚本类型：
#   debug / execution-runner / fast-selected-viewer
#
# 用途：
#   对 BB026 fast selector 选出的候选运行 P4U6 + P4U1 viewer。
#
# 输入：
#   fast selector 生成的 stable_scene、candidate、fast_p3、best_config。
#
# 输出：
#   diagnostics/current_v412/sodacan_bb026_fast_generalization_debug/viewer_selected/
#       terminal.txt
#       result.json
#       path_plan.json
#
# 不负责：
#   不重新筛选，不修改 legacy demo，不修改 P4U1/P4U6 源码。

set -euo pipefail

cd "$HOME/Projects/o7_mujoco_sim"
source "$HOME/mujoco_env/bin/activate"

OUTDIR="diagnostics/current_v412/sodacan_bb026_fast_generalization_debug/viewer_selected"
mkdir -p "$OUTDIR"

LOG="$OUTDIR/terminal.txt"
RESULT_JSON="$OUTDIR/result.json"
PATH_PLAN_JSON="$OUTDIR/path_plan.json"

exec > >(tee "$LOG") 2>&1

echo "========== BB026 FAST SELECTED VIEWER =========="
echo "sample      : {best_row['sample']}"
echo "model       : {model}"
echo "candidate   : {candidate}"
echo "p3_json     : {p3_json}"
echo "best_config : {rel(best_config_path)}"
echo

./run_mujoco_clean.sh scripts/05_execution_runner/run_v4_12p4u6_ik_path_record_demo_debug.py \\
  --model "{model}" \\
  --candidate "{candidate}" \\
  --p3-json "{p3_json}" \\
  --best-config "{rel(best_config_path)}" \\
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
  --max-grip-disp 0.020 \\
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
p = Path("diagnostics/current_v412/sodacan_bb026_fast_generalization_debug/viewer_selected/result.json")
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

    if not INROOT.exists():
        raise FileNotFoundError(INROOT)
    if not REFERENCE_CANDIDATE.exists():
        print("[WARN] reference candidate missing:", REFERENCE_CANDIDATE)

    ref_feat = None
    if REFERENCE_CANDIDATE.exists():
        ref_feat = candidate_features(load_json(REFERENCE_CANDIDATE))

    audit_rows = load_audit_rows()

    rows = []
    for sdir in sorted(INROOT.glob("sample*")):
        if not sdir.is_dir():
            continue

        sid = sdir.name.replace("sample", "")
        scene = find_scene(sdir)
        candidate_path = find_candidate(sdir, sid)
        p3_path = sdir / f"sample{sid}_p3.json"

        row = {
            "sample": sid,
            "sample_dir": rel(sdir),
            "scene": rel(scene) if scene else None,
            "candidate": rel(candidate_path) if candidate_path else None,
            "old_p3_json": rel(p3_path) if p3_path.exists() else None,
            "audit": audit_rows.get(sid, {}),
            "p3": summarize_p3(p3_path),
        }

        try:
            if candidate_path is None:
                raise RuntimeError("missing candidate")
            cand = load_json(candidate_path)
            feat = candidate_features(cand)
            row["features"] = None
            if feat is not None:
                row["features"] = {
                    "p": feat["p"].tolist(),
                    "radial_xy": feat["radial_xy"],
                    "z": feat["z"],
                    "finger_mean": feat["finger_mean"],
                    "thumb_yaw": feat["thumb_yaw"],
                    "thumb_roll": feat["thumb_roll"],
                    "thumb_pitch": feat["thumb_pitch"],
                }
                row["_feature_obj"] = feat
        except Exception as e:
            row["feature_error"] = repr(e)

        # 内部保留 numpy feature
        if "_feature_obj" in row:
            feat_obj = row["_feature_obj"]
            row["features_numpy_ready"] = True
            row["features_internal"] = feat_obj

        # score 需要 numpy feature，这里临时还原
        tmp = copy.deepcopy(row)
        if "features_internal" in row:
            tmp["features"] = row["features_internal"]
        else:
            tmp["features"] = None

        score_fast(tmp, ref_feat)
        row["fast_score"] = tmp["fast_score"]
        row["fast_reasons"] = tmp["fast_reasons"]

        row.pop("_feature_obj", None)
        row.pop("features_internal", None)
        rows.append(row)

    rows_sorted = sorted(rows, key=lambda r: r.get("fast_score", -1e9), reverse=True)

    # 保存轻量排序
    rank_json_rows = copy.deepcopy(rows_sorted)
    save_json(OUTROOT / "fast_rank_before_p2p3.json", {"rows_sorted": rank_json_rows})

    lines = []
    lines.append("========== BB026 FAST RANK BEFORE LIGHT P2/P3 ==========")
    for r in rows_sorted:
        lines.append(
            f"sample={r['sample']} fast_score={r['fast_score']:.2f} "
            f"old_pass={r.get('p3',{}).get('num_pass')} "
            f"FO={r.get('p3',{}).get('FO')} GO={r.get('p3',{}).get('GO')} HS={r.get('p3',{}).get('HS')} "
            f"features={r.get('features')}"
        )
        for rr in r.get("fast_reasons", [])[:10]:
            lines.append(f"  - {rr}")
        lines.append("")
    (OUTROOT / "fast_rank_before_p2p3.txt").write_text("\n".join(lines) + "\n")
    print("\n".join(lines))

    selected = rows_sorted[:TOP_N_LIGHT_P2P3]

    final_rows = []
    for r in selected:
        sid = r["sample"]
        odir = OUTROOT / f"sample{sid}"
        odir.mkdir(parents=True, exist_ok=True)

        row = copy.deepcopy(r)
        try:
            scene_path = PROJECT / r["scene"]
            candidate_path = PROJECT / r["candidate"]
            stable_scene = odir / "stable_scene.xml"

            patch_info = patch_scene_by_audit_delta(scene_path, stable_scene, r.get("audit", {}))
            save_json(odir / "stable_scene_patch_info.json", patch_info)

            p2_json = odir / f"sample{sid}_fast_p2.json"
            p3_json = odir / f"sample{sid}_fast_p3.json"
            plan_json = odir / f"sample{sid}_fast_plan.json"

            cmd_p2 = [
                "python3", rel(P2_SCRIPT),
                "--urdf", rel(URDF),
                "--model", rel(stable_scene),
                "--candidate", rel(candidate_path),
                "--runner-json", rel(RUNNER_JSON) if RUNNER_JSON.exists() else "",
                "--object-body", OBJECT_BODY,
                "--target-frame", TARGET_BODY,
                "--out", rel(p2_json),
                "--random-seeds", "4",
                "--random-std", "0.45",
                "--max-iters", "220",
                "--pos-tol", "0.0006",
                "--rot-tol", "0.006",
                "--rot-weight", "0.45",
            ]
            rc2 = run_cmd(cmd_p2, odir / "terminal_p2.txt")

            row["stable_scene"] = rel(stable_scene)
            row["fast_p2_json"] = rel(p2_json)
            row["p2_return_code"] = rc2

            if rc2 != 0:
                row["light_error"] = f"P2 failed rc={rc2}"
                final_rows.append(row)
                continue

            cmd_p3 = [
                "python3", rel(P3_SCRIPT),
                "--p2-json", rel(p2_json),
                "--model", rel(stable_scene),
                "--candidate", rel(candidate_path),
                "--object-body", OBJECT_BODY,
                "--out", rel(p3_json),
                "--best-plan-out", rel(plan_json),
                "--top-per-target", "3",
                "--max-combos", "64",
                "--path-samples", "12",
                "--min-hand-support-clearance", "0.0",
                "--min-fr3-object-clearance", "0.0",
                "--max-grasp-hand-object-distance", "0.055",
                "--min-joint-margin", "0.0",
            ]
            rc3 = run_cmd(cmd_p3, odir / "terminal_p3.txt")

            row["fast_p3_json"] = rel(p3_json)
            row["fast_plan_json"] = rel(plan_json)
            row["p3_return_code"] = rc3
            row["fast_p3"] = summarize_p3(p3_json)

        except Exception as e:
            row["light_error"] = repr(e)
            row["traceback"] = traceback.format_exc()

        final_rows.append(row)

    def final_score(r):
        base = r.get("fast_score", -1e9)
        p3 = r.get("fast_p3", {})
        s = base

        if p3.get("best_pass_exists"):
            s += 80
        npass = p3.get("num_pass")
        if isinstance(npass, int):
            s += min(npass, 100) * 1.0

        FO = p3.get("FO")
        GO = p3.get("GO")
        HS = p3.get("HS")
        if isinstance(FO, (int, float)) and FO >= 0:
            s += min(FO, 0.05) * 200
        if isinstance(GO, (int, float)) and 0.001 <= GO <= 0.015:
            s += 20
        if isinstance(HS, (int, float)):
            if HS >= 0:
                s += 20
            elif HS >= -0.015:
                s -= 5
            else:
                s -= 20
        return s

    for r in final_rows:
        r["final_fast_score"] = float(final_score(r))

    final_sorted = sorted(final_rows, key=lambda r: r.get("final_fast_score", -1e9), reverse=True)

    best = final_sorted[0] if final_sorted else None

    if best is not None and best.get("candidate") and best.get("fast_p3_json"):
        best_config_path = OUTROOT / "best_config_from_candidate.json"
        ctrl, base_cfg = make_best_config_from_candidate(PROJECT / best["candidate"], best_config_path)
        best["best_config"] = rel(best_config_path)
        best["best_config_base"] = rel(base_cfg)
        best["best_config_ctrl"] = ctrl
        viewer_script = make_viewer_script(best, best_config_path)
        best["viewer_script"] = rel(viewer_script)

    save_json(OUTROOT / "fast_final_summary.json", {"rows_sorted": final_sorted, "best": best})

    lines = []
    lines.append("========== BB026 FAST FINAL SUMMARY ==========")
    for r in final_sorted:
        fp3 = r.get("fast_p3", {})
        lines.append(
            f"sample={r['sample']} final_score={r.get('final_fast_score'):.2f} "
            f"fast_score={r.get('fast_score'):.2f} "
            f"pass={fp3.get('num_pass')} status={fp3.get('chosen_status')} "
            f"HS={fp3.get('HS')} FO={fp3.get('FO')} GO={fp3.get('GO')} HSc={fp3.get('HSc')} "
            f"p2_rc={r.get('p2_return_code')} p3_rc={r.get('p3_return_code')} "
            f"error={r.get('light_error')}"
        )
    lines.append("")
    lines.append("---- BEST ----")
    if best:
        lines.append(f"best_sample: {best.get('sample')}")
        lines.append(f"stable_scene: {best.get('stable_scene')}")
        lines.append(f"candidate: {best.get('candidate')}")
        lines.append(f"fast_p3_json: {best.get('fast_p3_json')}")
        lines.append(f"best_config: {best.get('best_config')}")
        lines.append(f"viewer_script: {best.get('viewer_script')}")
        lines.append("")
        lines.append("run viewer:")
        lines.append(f"./{best.get('viewer_script')}")
    else:
        lines.append("no best selected")

    txt = "\n".join(lines) + "\n"
    (OUTROOT / "fast_final_summary.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
