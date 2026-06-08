#!/usr/bin/env python3
"""
脚本类型：
    debug / rerank / soft-support / top-grasp

用途：
    对 BB026 top grasp 已生成候选重新排序。
    当前旧排序错误地偏向“离支撑远、离物体也远”的候选，导致 viewer 中完全没有物体接触。
    本脚本改用更接近实机逻辑的软支撑规则：
        1. 轻微 support 接触可以接受；
        2. 大拇指/手指大穿透 support 不能接受；
        3. GO 太大说明离物体太远，直接降权；
        4. GO 略负或接近 0 说明可能已有/即将形成物体接触，可以优先；
        5. thumb + 至少一根非拇指对抗由后续 P4U1 grip gate 判断，这里只做候选排序。

输入：
    diagnostics/current_v412/sodacan_bb026_top_grasp_light_debug/top_grasp_summary.json
    diagnostics/current_v412/sodacan_bb026_top_grasp_light_debug/candidates/*.json
    diagnostics/current_v412/sodacan_bb026_top_grasp_light_debug/scenes/*.xml
    diagnostics/current_v412/sodacan_bb026_top_grasp_light_debug/p2p3/*/p3.json

输出：
    diagnostics/current_v412/sodacan_bb026_top_soft_rerank_debug/
        soft_rerank_summary.txt
        soft_rerank_summary.json
        best_config_rankXX.json
    scripts/05_execution_runner/run_sodacan_bb026_top_soft_rank01_viewer_debug.sh
    scripts/05_execution_runner/run_sodacan_bb026_top_soft_rank02_viewer_debug.sh
    scripts/05_execution_runner/run_sodacan_bb026_top_soft_rank03_viewer_debug.sh

当前流程位置：
    top grasp candidates 已生成
        -> soft-support rerank
        -> 依次 viewer 验证 rank1/rank2/rank3

不负责：
    1. 不重新跑 P2/P3；
    2. 不重新生成 candidate；
    3. 不修改 P4U1/P4U6；
    4. 不修改 legacy_final_demos。
"""

from pathlib import Path
import json
import copy
import math


PROJECT = Path.home() / "Projects/o7_mujoco_sim"
SUMMARY_JSON = PROJECT / "diagnostics/current_v412/sodacan_bb026_top_grasp_light_debug/top_grasp_summary.json"
OUTROOT = PROJECT / "diagnostics/current_v412/sodacan_bb026_top_soft_rerank_debug"
RUNNER_DIR = PROJECT / "scripts/05_execution_runner"

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


def finite_number(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def soft_support_score(clearance, name, reasons):
    """
    clearance >= 0: 好
    -0.006 ~ 0: 轻微碰，允许
    -0.015 ~ -0.006: 明显碰，扣分但不一定淘汰
    < -0.015: 大穿透，强扣分
    """
    if not finite_number(clearance):
        reasons.append(f"{name}=None")
        return -20

    c = float(clearance)
    if c >= 0.006:
        reasons.append(f"{name}_clear_good={c:.5f}")
        return 30
    if c >= 0.0:
        reasons.append(f"{name}_touch_or_clear={c:.5f}")
        return 20
    if c >= -0.006:
        reasons.append(f"{name}_soft_touch_allow={c:.5f}")
        return 8
    if c >= -0.015:
        reasons.append(f"{name}_medium_support_touch={c:.5f}")
        return -15

    reasons.append(f"{name}_hard_support_penetration={c:.5f}")
    return -80


def object_gap_score(go, reasons):
    """
    GO 是闭合手与物体距离。
    之前选中的 0.022m 太远，viewer 完全没接触，所以必须强降权。
    """
    if not finite_number(go):
        reasons.append("GO=None")
        return -50

    go = float(go)

    if -0.012 <= go <= -0.001:
        reasons.append(f"GO_contact_like_good={go:.5f}")
        return 45
    if -0.001 < go <= 0.006:
        reasons.append(f"GO_near_contact_good={go:.5f}")
        return 50
    if 0.006 < go <= 0.012:
        reasons.append(f"GO_small_gap_ok={go:.5f}")
        return 25
    if 0.012 < go <= 0.020:
        reasons.append(f"GO_gap_large={go:.5f}")
        return -20
    if go > 0.020:
        reasons.append(f"GO_too_far_no_contact={go:.5f}")
        return -90

    reasons.append(f"GO_too_deep_penetration={go:.5f}")
    return -35


def fr3_object_score(fo, reasons):
    if not finite_number(fo):
        reasons.append("FO=None")
        return -20

    fo = float(fo)
    if fo >= 0.01:
        reasons.append(f"FO_clear={fo:.5f}")
        return 20
    if fo >= 0:
        reasons.append(f"FO_touch_or_clear={fo:.5f}")
        return 8
    if fo >= -0.006:
        reasons.append(f"FO_slight_collision={fo:.5f}")
        return -15

    reasons.append(f"FO_bad_collision={fo:.5f}")
    return -70


def score_row(row):
    p3 = row.get("p3", {})
    reasons = []
    score = 0.0

    num_pass = p3.get("num_pass")
    if isinstance(num_pass, int) and num_pass > 0:
        score += 15 + min(num_pass, 5) * 8
        reasons.append(f"p3_pass={num_pass}")
    else:
        reasons.append("p3_pass_zero")

    go = p3.get("GO")
    hs = p3.get("HS")
    hsc = p3.get("HSc")
    fo = p3.get("FO")

    score += object_gap_score(go, reasons)
    score += soft_support_score(hs, "HS_path", reasons)
    score += soft_support_score(hsc, "HSc_closed", reasons)
    score += fr3_object_score(fo, reasons)

    # 稍微偏向有支撑安全但不能牺牲物体接触。
    margin = p3.get("margin")
    if finite_number(margin):
        score += min(max(float(margin), 0.0), 1.0) * 3
        reasons.append(f"joint_margin={float(margin):.3f}")

    # 轻微扣 heuristic，不让它压过接触逻辑。
    h = row.get("heuristic")
    if finite_number(h):
        score -= 5.0 * float(h)
        reasons.append(f"heuristic={float(h):.5f}")

    # 强制：如果 GO 太远，就算 clearance 再好也不应第一。
    if finite_number(go) and float(go) > 0.020:
        score -= 60
        reasons.append("hard_downrank_reason=too_far_from_object")

    # 强制：如果 support 大穿透，不能 viewer 优先。
    if (finite_number(hs) and float(hs) < -0.020) or (finite_number(hsc) and float(hsc) < -0.020):
        score -= 60
        reasons.append("hard_downrank_reason=large_support_penetration")

    row["soft_score"] = float(score)
    row["soft_reasons"] = reasons
    return row


def make_best_config(candidate_path, out_path):
    base_candidates = [
        PROJECT / "legacy_final_demos/final_sodacan165_sample014_p4u6_ready_gated_snap_demo/inputs/best_config.json",
        PROJECT / "diagnostics/current_v412/sodacan_bb026_top_grasp_light_debug/best_config_from_top_candidate.json",
        PROJECT / "diagnostics/current_v412/v4_12p4h2_can52_contact_sequence_proxy_debug/best_contact_sequence_config_ctrlsplit_debug.json",
    ]

    base_path = next((p for p in base_candidates if p.exists()), None)
    if base_path is None:
        raise FileNotFoundError("cannot find base best_config")

    cfg = load_json(base_path)
    cand = load_json(candidate_path)
    ctrl = cand.get("hand", {}).get("o7_active_ctrl")
    if not isinstance(ctrl, dict):
        raise RuntimeError(f"candidate missing hand.o7_active_ctrl: {candidate_path}")

    ctrl = {j: float(ctrl[j]) for j in O7_ACTIVE_JOINTS if j in ctrl}
    if len(ctrl) != len(O7_ACTIVE_JOINTS):
        raise RuntimeError(f"incomplete ctrl: {ctrl}")

    cfg.setdefault("best_record", {})
    cfg["best_record"].setdefault("hand_config", {})
    cfg["best_record"]["hand_config"]["ctrl"] = ctrl
    cfg["best_record"]["hand_config"]["source"] = "bb026_top_soft_rerank_candidate_ctrl"
    cfg["generalization_debug_note"] = {
        "type": "bb026_top_soft_rerank_best_config",
        "base_config": rel(base_path),
        "candidate": rel(candidate_path),
        "reason": "soft-support rerank: prioritize object contact, allow slight support touch, reject large support penetration",
    }

    save_json(out_path, cfg)


def make_viewer_script(rank, row, best_config):
    script = RUNNER_DIR / f"run_sodacan_bb026_top_soft_rank{rank:02d}_viewer_debug.sh"

    outdir = f"diagnostics/current_v412/sodacan_bb026_top_soft_rerank_debug/viewer_rank{rank:02d}"

    text = f'''#!/usr/bin/env bash
# 脚本类型：
#   debug / execution-runner / soft-rerank-top-viewer
#
# 用途：
#   运行 BB026 top grasp soft-support rerank 的第 {rank} 名候选。
#
# 输入：
#   candidate / scene / p3 / best_config 均来自 soft rerank。
#
# 输出：
#   {outdir}/terminal.txt
#   {outdir}/result.json
#   {outdir}/path_plan.json
#
# 不负责：
#   不重新筛选，不修改 legacy demo，不修改 P4U1/P4U6 源码。

set -euo pipefail

cd "$HOME/Projects/o7_mujoco_sim"
source "$HOME/mujoco_env/bin/activate"

OUTDIR="{outdir}"
mkdir -p "$OUTDIR"

LOG="$OUTDIR/terminal.txt"
RESULT_JSON="$OUTDIR/result.json"
PATH_PLAN_JSON="$OUTDIR/path_plan.json"

exec > >(tee "$LOG") 2>&1

echo "========== BB026 TOP SOFT RANK {rank:02d} VIEWER =========="
echo "tag         : {row.get('tag')}"
echo "soft_score  : {row.get('soft_score')}"
echo "candidate   : {row.get('candidate_path')}"
echo "model       : {row.get('stable_scene')}"
echo "p3_json     : {row.get('p3_json')}"
echo "best_config : {rel(best_config)}"
echo

./run_mujoco_clean.sh scripts/05_execution_runner/run_v4_12p4u6_ik_path_record_demo_debug.py \\
  --model "{row.get('stable_scene')}" \\
  --candidate "{row.get('candidate_path')}" \\
  --p3-json "{row.get('p3_json')}" \\
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
  --micro-squeeze-fraction 0.04 \\
  --finger-close-scale 1.16 \\
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
p = Path("$RESULT_JSON")
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
    RUNNER_DIR.mkdir(parents=True, exist_ok=True)

    if not SUMMARY_JSON.exists():
        raise FileNotFoundError(SUMMARY_JSON)

    data = load_json(SUMMARY_JSON)
    rows = data.get("rows_sorted", [])

    scored = [score_row(copy.deepcopy(r)) for r in rows]
    scored = sorted(scored, key=lambda x: x.get("soft_score", -1e9), reverse=True)

    for i, r in enumerate(scored[:3], start=1):
        cfg_path = OUTROOT / f"best_config_rank{i:02d}.json"
        make_best_config(PROJECT / r["candidate_path"], cfg_path)
        r["soft_best_config"] = rel(cfg_path)
        script = make_viewer_script(i, r, cfg_path)
        r["soft_viewer_script"] = rel(script)

    save_json(OUTROOT / "soft_rerank_summary.json", {
        "format": "bb026_top_soft_rerank_debug_v1",
        "rows_sorted": scored,
    })

    lines = []
    lines.append("========== BB026 TOP SOFT-SUPPORT RERANK SUMMARY ==========")
    lines.append("规则：GO 太远强降权；轻微 support 接触允许；大 support 穿透强降权。")
    lines.append("")

    for i, r in enumerate(scored, start=1):
        p3 = r.get("p3", {})
        lines.append(
            f"rank={i:02d} tag={r.get('tag')} soft_score={r.get('soft_score'):.3f} "
            f"old_score={r.get('selector_score')} pass={p3.get('num_pass')} "
            f"HS={p3.get('HS')} HSc={p3.get('HSc')} FO={p3.get('FO')} GO={p3.get('GO')} "
            f"candidate={r.get('candidate_path')}"
        )
        for rr in r.get("soft_reasons", []):
            lines.append(f"  - {rr}")
        if r.get("soft_viewer_script"):
            lines.append(f"  viewer: ./{r.get('soft_viewer_script')}")
        lines.append("")

    lines.append("---- 建议运行顺序 ----")
    for r in scored[:3]:
        if r.get("soft_viewer_script"):
            lines.append(f"./{r.get('soft_viewer_script')}")
    lines.append("===========================================================")

    txt = "\\n".join(lines) + "\\n"
    (OUTROOT / "soft_rerank_summary.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
