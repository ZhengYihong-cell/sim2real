#!/usr/bin/env bash
# 脚本类型：final-demo / viewer-runner
# 用途：复现 SodaCan sample014 泛化成功抓取 demo。
# 输入：本目录 inputs 下固化的 scene、candidate、P3 和 best_config。
# 输出：records/latest_viewer 下的 terminal/result/path_plan。
# 不负责：不重新搜索候选，不重新跑 P2/P3，不修改任何 legacy demo。

set -euo pipefail

DEMO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "$DEMO_DIR/../.." && pwd)"
cd "$PROJECT_ROOT"

source "$HOME/mujoco_env/bin/activate"

mkdir -p "$DEMO_DIR/records/latest_viewer"
LOG="$DEMO_DIR/records/latest_viewer/terminal.txt"
RESULT_JSON="$DEMO_DIR/records/latest_viewer/result.json"
PATH_PLAN_JSON="$DEMO_DIR/records/latest_viewer/path_plan.json"

exec > >(tee "$LOG") 2>&1

./run_mujoco_clean.sh scripts/05_execution_runner/run_v4_12p4u6_ik_path_record_demo_debug.py \
  --model "$DEMO_DIR/inputs/scene.xml" \
  --candidate "$DEMO_DIR/inputs/candidate.json" \
  --p3-json "$DEMO_DIR/inputs/p3_contactfree_goal.json" \
  --best-config "$DEMO_DIR/inputs/best_config.json" \
  --which best_available \
  --object-body grasp_can \
  --target-body fr3_link7 \
  --out "$RESULT_JSON" \
  --plan-out "$PATH_PLAN_JSON" \
  --viewer \
  --keep-viewer-open \
  --start-arm-mode zero_clamped \
  --start-hold-duration 1.2 \
  --home-hold-duration 0.6 \
  --pre-hold-duration 0.8 \
  --grasp-settle-duration 0.35 \
  --close-duration 0.45 \
  --post-close-target-hold-duration 0.25 \
  --micro-squeeze-duration 0.35 \
  --micro-squeeze-fraction 0.00 \
  --finger-close-scale 1.12 \
  --thumb-pitch-from-finger-gain 0.24 \
  --grip-ready-stable-steps 5 \
  --min-live-non-thumb 1 \
  --opposition-cos-threshold -0.30 \
  --max-grip-disp 0.018 \
  --max-extra-disp-during-squeeze 0.004 \
  --approach-abort-disp 0.025 \
  --approach-min-clearance 0.003 \
  --grasp-path-min-clearance 0.001 \
  --plan-attempts 5 \
  --rrt-max-iters 2500 \
  --rrt-step 0.28 \
  --edge-step 0.035 \
  --goal-bias 0.20 \
  --shortcut-iters 100 \
  --joint-speed-rad-s 0.75 \
  --min-segment-duration 0.25 \
  --hard-servo-approach \
  --enable-lift \
  --lift-z 0.090 \
  --lift-duration 3.0 \
  --final-hold-duration 1.2 \
  --print-every-steps 100 \
  --log-every-steps 100 \
  --frame-sleep 0.0015
