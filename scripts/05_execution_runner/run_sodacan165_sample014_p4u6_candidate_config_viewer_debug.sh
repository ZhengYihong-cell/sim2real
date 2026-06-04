#!/usr/bin/env bash
# 脚本类型：
#   debug / execution-runner / p4u6-viewer-launcher
#
# 用途：
#   使用 sem-SodaCan-16526 sample014 的：
#     1. contact-free P3 结果；
#     2. sodacan candidate 自己生成的 best_config；
#   重新运行 P4U6 + P4U1 动态可视化抓握。
#
# 输入：
#   diagnostics/current_v412/sodacan165_sample014_p4u6_viewer_debug/sample014_p3_contactfree_goal.json
#   diagnostics/current_v412/sodacan165_sample014_p4u6_viewer_debug/sodacan_sample014_best_config_from_candidate_debug.json
#
# 输出：
#   diagnostics/current_v412/sodacan165_sample014_p4u6_candidate_config_viewer_debug/
#     terminal.txt
#     result.json
#     path_plan.json
#
# 当前流程位置：
#   P3 contact-free candidate
#     -> sodacan-specific hand config
#     -> P4U6 collision-aware approach
#     -> P4U1 ready-gated snap close
#     -> lift
#
# 不负责：
#   1. 不重新搜索候选；
#   2. 不修改 P4U1/P4U6 源码；
#   3. 不修改 can52 legacy demo；
#   4. 不手工调整单个关节参数。

set -euo pipefail

cd ~/Projects/o7_mujoco_sim

if [[ -f "$HOME/mujoco_env/bin/activate" ]]; then
  source "$HOME/mujoco_env/bin/activate"
fi

OUTDIR="diagnostics/current_v412/sodacan165_sample014_p4u6_candidate_config_viewer_debug"
mkdir -p "$OUTDIR"

LOG="$OUTDIR/terminal.txt"
exec > >(tee "$LOG") 2>&1

P4U6="scripts/05_execution_runner/run_v4_12p4u6_ik_path_record_demo_debug.py"

MODEL="diagnostics/current_v412/sodacan165_topk_p2p3_batch_debug/sample014/initial_debug/scene/sodacan165_sample014_from_can52_scene_debug.xml"
CANDIDATE="diagnostics/current_v412/sodacan165_topk_p2p3_batch_debug/sample014/initial_debug/candidates/sample014_candidate.json"
P3_JSON="diagnostics/current_v412/sodacan165_sample014_p4u6_viewer_debug/sample014_p3_contactfree_goal.json"
BEST_CONFIG="diagnostics/current_v412/sodacan165_sample014_p4u6_viewer_debug/sodacan_sample014_best_config_from_candidate_debug.json"

RESULT_JSON="$OUTDIR/result.json"
PATH_PLAN_JSON="$OUTDIR/path_plan.json"

echo "========== SODACAN165 SAMPLE014 P4U6 WITH CANDIDATE CONFIG =========="
echo "project     : $(pwd)"
echo "p4u6        : $P4U6"
echo "model       : $MODEL"
echo "candidate   : $CANDIDATE"
echo "p3_json     : $P3_JSON"
echo "best_config : $BEST_CONFIG"
echo "outdir      : $OUTDIR"
echo

for f in "$P4U6" "$MODEL" "$CANDIDATE" "$P3_JSON" "$BEST_CONFIG"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] missing file: $f"
    exit 2
  fi
done

python3 - <<PY
import json
from pathlib import Path

p3 = json.loads(Path("$P3_JSON").read_text())
cfg = json.loads(Path("$BEST_CONFIG").read_text())

print("========== INPUT QUICK CHECK ==========")
print("p3 best_available:", (p3.get("best_available") or {}).get("precheck_status"))
print("p3 GO:", (p3.get("best_available") or {}).get("static_grasp_closed_hand_object_distance"))
print("p3 score:", (p3.get("best_available") or {}).get("score"))
print("best_config ctrl:", cfg.get("best_record", {}).get("hand_config", {}).get("ctrl"))
print("=======================================")
PY

if [[ -x "./run_mujoco_clean.sh" ]]; then
  RUN_PREFIX=(./run_mujoco_clean.sh)
else
  RUN_PREFIX=(python3)
fi

"${RUN_PREFIX[@]}" "$P4U6" \
  --model "$MODEL" \
  --candidate "$CANDIDATE" \
  --p3-json "$P3_JSON" \
  --best-config "$BEST_CONFIG" \
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
  --micro-squeeze-fraction 0.08 \
  --finger-close-scale 0.92 \
  --thumb-pitch-from-finger-gain 0.24 \
  --grip-ready-stable-steps 8 \
  --min-live-non-thumb 1 \
  --opposition-cos-threshold -0.30 \
  --max-grip-disp 0.006 \
  --max-extra-disp-during-squeeze 0.003 \
  --approach-abort-disp 0.015 \
  --approach-min-clearance 0.003 \
  --grasp-path-min-clearance 0.001 \
  --plan-attempts 10 \
  --rrt-max-iters 4000 \
  --rrt-step 0.28 \
  --edge-step 0.035 \
  --goal-bias 0.20 \
  --shortcut-iters 400 \
  --joint-speed-rad-s 0.75 \
  --min-segment-duration 0.35 \
  --hard-servo-approach \
  --enable-lift \
  --lift-z 0.060 \
  --lift-duration 3.0 \
  --final-hold-duration 1.0 \
  --print-every-steps 100 \
  --log-every-steps 100 \
  --frame-sleep 0.0015

echo
echo "========== DONE =========="
echo "terminal : $LOG"
echo "result   : $RESULT_JSON"
echo "path_plan: $PATH_PLAN_JSON"

if [[ -f "$RESULT_JSON" ]]; then
  echo
  echo "========== RESULT QUICK VIEW =========="
  python3 - <<PY
import json
from pathlib import Path

p = Path("$RESULT_JSON")
d = json.loads(p.read_text())
for k in [
    "success",
    "stop_reason",
    "grip_ready",
    "final_object_rise",
    "max_object_rise",
    "final_counts",
    "max_hand_object",
    "max_hand_object_lift",
]:
    if k in d:
        print(f"{k}: {d[k]}")
PY
fi
