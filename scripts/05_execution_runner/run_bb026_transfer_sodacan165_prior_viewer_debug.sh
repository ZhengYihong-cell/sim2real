#!/usr/bin/env bash
# 脚本类型：
#   debug / execution-runner / transferred-prior-viewer
#
# 用途：
#   运行 BB026 直接迁移 SodaCan165 成功抓握先验后的 P4U6/P4U1 viewer。
#
# 输入：
#   diagnostics/current_v412/bb026_transfer_sodacan165_prior_debug/candidate.json
#   diagnostics/current_v412/bb026_transfer_sodacan165_prior_debug/best_config.json
#   diagnostics/current_v412/bb026_transfer_sodacan165_prior_debug/p3.json
#
# 输出：
#   diagnostics/current_v412/bb026_transfer_sodacan165_prior_debug/viewer/
#     terminal.txt
#     result.json
#     path_plan.json

set -euo pipefail

cd "$HOME/Projects/o7_mujoco_sim"
source "$HOME/mujoco_env/bin/activate"

OUTDIR="diagnostics/current_v412/bb026_transfer_sodacan165_prior_debug/viewer"
mkdir -p "$OUTDIR"

LOG="$OUTDIR/terminal.txt"
RESULT_JSON="$OUTDIR/result.json"
PATH_PLAN_JSON="$OUTDIR/path_plan.json"

exec > >(tee "$LOG") 2>&1

./run_mujoco_clean.sh scripts/05_execution_runner/run_v4_12p4u6_ik_path_record_demo_debug.py \
  --model diagnostics/current_v412/sodacan_bb026_top_grasp_light_debug/scenes/stable_scene_seed014.xml \
  --candidate diagnostics/current_v412/bb026_transfer_sodacan165_prior_debug/candidate.json \
  --p3-json diagnostics/current_v412/bb026_transfer_sodacan165_prior_debug/p3.json \
  --best-config diagnostics/current_v412/bb026_transfer_sodacan165_prior_debug/best_config.json \
  --which best_available \
  --object-body grasp_can \
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
  --micro-squeeze-duration 0.35 \
  --micro-squeeze-fraction 0.00 \
  --finger-close-scale 1.12 \
  --thumb-pitch-from-finger-gain 0.24 \
  --grip-ready-stable-steps 5 \
  --min-live-non-thumb 1 \
  --opposition-cos-threshold -0.30 \
  --max-grip-disp 0.020 \
  --max-extra-disp-during-squeeze 0.004 \
  --approach-abort-disp 0.030 \
  --approach-min-clearance 0.002 \
  --grasp-path-min-clearance 0.001 \
  --plan-attempts 4 \
  --rrt-max-iters 2200 \
  --rrt-step 0.28 \
  --edge-step 0.035 \
  --goal-bias 0.20 \
  --shortcut-iters 100 \
  --joint-speed-rad-s 0.80 \
  --min-segment-duration 0.25 \
  --hard-servo-approach \
  --enable-lift \
  --lift-z 0.090 \
  --lift-duration 2.8 \
  --final-hold-duration 1.0 \
  --print-every-steps 100 \
  --log-every-steps 100 \
  --frame-sleep 0.0015

echo
echo "========== RESULT QUICK VIEW =========="
python3 - <<'PY'
import json
from pathlib import Path
p = Path("diagnostics/current_v412/bb026_transfer_sodacan165_prior_debug/viewer/result.json")
if p.exists():
    d = json.loads(p.read_text())
    for k in ["success","stop_reason","grip_ready","final_object_disp","final_object_rise","max_object_rise","final_groups","final_opposition_cos","max_stable_count"]:
        if k in d:
            print(f"{k}: {d[k]}")
PY
