#!/usr/bin/env bash
# 脚本类型：
#   debug / execution-runner / viewer-validation
#
# 用途：
#   对 sem-SodaCan-16526 sample014 的 close sweep 最优参数进行 viewer 复查。
#   当前 sweep 已证明 grip_ready=True，并形成 thumb + pinky 对抗；
#   本脚本用于肉眼确认 lift 阶段物体是否真正离开支撑块，还是只是贴着/蹭着垫块。
#
# 输入：
#   1. sample014 scene
#   2. sample014 candidate
#   3. contact-free P3 json
#   4. sodacan sample014 专属 best_config
#
# 输出：
#   diagnostics/current_v412/sodacan165_sample014_best_sweep_viewer_debug/
#     terminal.txt
#     result.json
#     path_plan.json
#
# 当前流程位置：
#   P2/P3 已通过
#     -> P4U6 path 已通过
#     -> close sweep 已找到 grip_ready
#     -> viewer 复查 lift 是否真实成立
#
# 不负责：
#   1. 不修改 P4U1/P4U6 源码；
#   2. 不修改 legacy_final_demos；
#   3. 不重新跑 P2/P3；
#   4. 不手工写死最终 demo 参数；
#   5. 不判断其他物体泛化。

set -euo pipefail

cd ~/Projects/o7_mujoco_sim

if [[ -f "$HOME/mujoco_env/bin/activate" ]]; then
  source "$HOME/mujoco_env/bin/activate"
fi

OUTDIR="diagnostics/current_v412/sodacan165_sample014_best_sweep_viewer_debug"
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

# sweep 最优组：run_10 / run_11 / run_12
FINGER_CLOSE_SCALE="${FINGER_CLOSE_SCALE:-1.12}"
MICRO_SQUEEZE_FRACTION="${MICRO_SQUEEZE_FRACTION:-0.00}"
MAX_GRIP_DISP="${MAX_GRIP_DISP:-0.018}"

echo "========== SODACAN165 SAMPLE014 BEST SWEEP VIEWER =========="
echo "finger_close_scale     : $FINGER_CLOSE_SCALE"
echo "micro_squeeze_fraction : $MICRO_SQUEEZE_FRACTION"
echo "max_grip_disp          : $MAX_GRIP_DISP"
echo "model                  : $MODEL"
echo "candidate              : $CANDIDATE"
echo "p3_json                : $P3_JSON"
echo "best_config            : $BEST_CONFIG"
echo "outdir                 : $OUTDIR"
echo

for f in "$P4U6" "$MODEL" "$CANDIDATE" "$P3_JSON" "$BEST_CONFIG"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] missing file: $f"
    exit 2
  fi
done

./run_mujoco_clean.sh "$P4U6" \
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
  --micro-squeeze-fraction "$MICRO_SQUEEZE_FRACTION" \
  --finger-close-scale "$FINGER_CLOSE_SCALE" \
  --thumb-pitch-from-finger-gain 0.24 \
  --grip-ready-stable-steps 5 \
  --min-live-non-thumb 1 \
  --opposition-cos-threshold -0.30 \
  --max-grip-disp "$MAX_GRIP_DISP" \
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
    "final_object_disp",
    "final_object_rise",
    "max_object_rise",
    "final_groups",
    "final_opposition_cos",
    "max_stable_count",
    "final_counts",
    "max_hand_object",
    "max_hand_object_lift",
]:
    if k in d:
        print(f"{k}: {d[k]}")
PY
fi
