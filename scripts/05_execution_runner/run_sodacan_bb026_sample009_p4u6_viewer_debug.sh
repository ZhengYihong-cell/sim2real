#!/usr/bin/env bash
# 脚本类型：
#   debug / execution-runner / p4u6-viewer-validation
#
# 用途：
#   对第二个 SodaCan 泛化目标 sem-SodaCan-bb0262... 的 sample009 执行 P4U6 + P4U1 viewer 验证。
#   当前 P3 严格预检 pass=0，但 sample009 主要只因为闭合阶段 hand-support 轻微为负；
#   FR3-object clearance 为正，hand-object distance 合理，因此进入 viewer 观察是否能实际抓起。
#
# 输入：
#   1. sample009 scene
#   2. sample009 candidate
#   3. sample009 P3 best_available
#   4. sample009 专属 best_config
#
# 输出：
#   diagnostics/current_v412/sodacan_bb026_sample009_p4u6_viewer_debug/
#     terminal.txt
#     result.json
#     path_plan.json
#
# 当前流程位置：
#   第二个 SodaCan 泛化：
#     P2/P3 strict check
#     -> support-relaxed visual validation
#     -> ready-gated snap close
#     -> lift
#
# 不负责：
#   1. 不修改 can52 或 sodacan165 固化 demo；
#   2. 不重新生成 P2/P3；
#   3. 不手调 sample 的位姿；
#   4. 不把 support collision 直接判定为成功，最终以 viewer 和 result.json 为准。

set -euo pipefail

cd "$HOME/Projects/o7_mujoco_sim"

if [[ -f "$HOME/mujoco_env/bin/activate" ]]; then
  source "$HOME/mujoco_env/bin/activate"
fi

OUTDIR="diagnostics/current_v412/sodacan_bb026_sample009_p4u6_viewer_debug"
mkdir -p "$OUTDIR"

LOG="$OUTDIR/terminal.txt"
RESULT_JSON="$OUTDIR/result.json"
PATH_PLAN_JSON="$OUTDIR/path_plan.json"

exec > >(tee "$LOG") 2>&1

P4U6="scripts/05_execution_runner/run_v4_12p4u6_ik_path_record_demo_debug.py"
MODEL="diagnostics/current_v412/sodacan_bb026_topk_p2p3_batch_debug/sample009/initial_debug/scene/sodacan165_sample009_from_can52_scene_debug.xml"
CANDIDATE="diagnostics/current_v412/sodacan_bb026_topk_p2p3_batch_debug/sample009/initial_debug/candidates/sample009_candidate.json"
P3_JSON="diagnostics/current_v412/sodacan_bb026_topk_p2p3_batch_debug/sample009/sample009_p3.json"
BEST_CONFIG="diagnostics/current_v412/sodacan_bb026_sample009_p4u6_viewer_debug/bb026_sample009_best_config_from_candidate_debug.json"

echo "========== SODACAN BB026 SAMPLE009 P4U6 VIEWER =========="
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

set +e
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
RET=$?
set -e

echo
echo "========== RUN RETURN =========="
echo "return_code: $RET"
echo "terminal   : $LOG"
echo "result     : $RESULT_JSON"
echo "path_plan  : $PATH_PLAN_JSON"

if [[ -f "$RESULT_JSON" ]]; then
  echo
  echo "========== RESULT QUICK VIEW =========="
  python3 -c "import json; d=json.load(open('$RESULT_JSON')); keys=['success','stop_reason','grip_ready','final_object_disp','final_object_rise','max_object_rise','final_groups','final_opposition_cos','max_stable_count','final_counts','max_hand_object','max_hand_object_lift']; [print(f'{k}: {d[k]}') for k in keys if k in d]"
fi

exit "$RET"
