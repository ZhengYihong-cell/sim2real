#!/usr/bin/env bash
# 脚本类型：
#   debug / execution-runner / v4.13d-selected-viewer
#
# 用途：
#   运行 V4.13d 通用泛化流程自动选出的 rank1 candidate。
#
# 输出：
#   diagnostics/current_v413/core-bottle-9da48acd8046be50307e049494951dab_v413d_after_v413e_p2p3_debug/viewer_selected/terminal.txt
#   diagnostics/current_v413/core-bottle-9da48acd8046be50307e049494951dab_v413d_after_v413e_p2p3_debug/viewer_selected/result.json
#   diagnostics/current_v413/core-bottle-9da48acd8046be50307e049494951dab_v413d_after_v413e_p2p3_debug/viewer_selected/path_plan.json
#
# 不负责：
#   不重新筛选，不修改 legacy demo，不修改 P4U6/P4U1 源码。

set -euo pipefail

cd "$HOME/Projects/o7_mujoco_sim"
source "$HOME/mujoco_env/bin/activate"

OUTDIR="diagnostics/current_v413/core-bottle-9da48acd8046be50307e049494951dab_v413d_after_v413e_p2p3_debug/viewer_selected"
mkdir -p "$OUTDIR"

LOG="$OUTDIR/terminal.txt"
RESULT_JSON="$OUTDIR/result.json"
PATH_PLAN_JSON="$OUTDIR/path_plan.json"

exec > >(tee "$LOG") 2>&1

echo "========== V4.13D SELECTED VIEWER =========="
echo "tag          : local023_raw062_top_grasp"
echo "local index  : 23"
echo "raw index    : 62"
echo "grasp_type   : top_grasp"
echo "score        : 48.80476567709407"
echo "scene        : diagnostics/current_v413/core-bottle-9da48acd8046be50307e049494951dab_v413d_after_v413e_p2p3_debug/local023_raw062_top_grasp/scene.xml"
echo "candidate    : diagnostics/current_v413/core-bottle-9da48acd8046be50307e049494951dab_v413d_after_v413e_p2p3_debug/local023_raw062_top_grasp/candidate.json"
echo "p3_json      : diagnostics/current_v413/core-bottle-9da48acd8046be50307e049494951dab_v413d_after_v413e_p2p3_debug/local023_raw062_top_grasp/p3.json"
echo "object_body  : grasp_object"
echo "best_config  : diagnostics/current_v413/core-bottle-9da48acd8046be50307e049494951dab_v413d_after_v413e_p2p3_debug/best_config_selected.json"
echo

./run_mujoco_clean.sh scripts/05_execution_runner/run_v4_12p4u6_ik_path_record_demo_debug.py \
  --model "diagnostics/current_v413/core-bottle-9da48acd8046be50307e049494951dab_v413d_after_v413e_p2p3_debug/local023_raw062_top_grasp/scene.xml" \
  --candidate "diagnostics/current_v413/core-bottle-9da48acd8046be50307e049494951dab_v413d_after_v413e_p2p3_debug/local023_raw062_top_grasp/candidate.json" \
  --p3-json "diagnostics/current_v413/core-bottle-9da48acd8046be50307e049494951dab_v413d_after_v413e_p2p3_debug/local023_raw062_top_grasp/p3.json" \
  --best-config "diagnostics/current_v413/core-bottle-9da48acd8046be50307e049494951dab_v413d_after_v413e_p2p3_debug/best_config_selected.json" \
  --which best_available \
  --object-body "grasp_object" \
  --target-body fr3_link7 \
  --out "$RESULT_JSON" \
  --plan-out "$PATH_PLAN_JSON" \
  --viewer \
  --keep-viewer-open \
  --start-arm-mode zero_clamped \
  --start-hold-duration 0.8 \
  --home-hold-duration 0.3 \
  --pre-hold-duration 0.35 \
  --grasp-settle-duration 0.25 \
  --close-duration 0.45 \
  --post-close-target-hold-duration 0.25 \
  --micro-squeeze-duration 0.40 \
  --micro-squeeze-fraction 0.04 \
  --finger-close-scale 1.16 \
  --thumb-pitch-from-finger-gain 0.24 \
  --grip-ready-stable-steps 6 \
  --min-live-non-thumb 1 \
  --opposition-cos-threshold -0.30 \
  --max-grip-disp 0.024 \
  --max-extra-disp-during-squeeze 0.006 \
  --approach-abort-disp 0.030 \
  --approach-min-clearance 0.002 \
  --grasp-path-min-clearance 0.001 \
  --plan-attempts 3 \
  --rrt-max-iters 1800 \
  --rrt-step 0.30 \
  --edge-step 0.045 \
  --goal-bias 0.20 \
  --shortcut-iters 60 \
  --joint-speed-rad-s 0.85 \
  --min-segment-duration 0.20 \
  --hard-servo-approach \
  --enable-lift \
  --lift-z 0.090 \
  --lift-duration 2.6 \
  --final-hold-duration 0.9 \
  --print-every-steps 100 \
  --log-every-steps 100 \
  --frame-sleep 0.0015

echo
echo "========== RESULT QUICK VIEW =========="
python3 - <<'R'
import json
from pathlib import Path

p = Path("diagnostics/current_v413/core-bottle-9da48acd8046be50307e049494951dab_v413d_after_v413e_p2p3_debug/viewer_selected/result.json")
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
