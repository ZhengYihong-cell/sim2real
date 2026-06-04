#!/usr/bin/env bash
# 脚本类型：
#   debug / execution-runner / p4u6-viewer-launcher
#
# 用途：
#   使用 sem-SodaCan-16526 的 P2/P3 批量筛选结果，运行 sample014 的完整动态可视化抓握：
#     P4U6 collision-aware approach path
#     -> P4U1 ready-gated snap close
#     -> grip_ready
#     -> fixed-grip lift
#
# 输入：
#   默认读取：
#     diagnostics/current_v412/sodacan165_topk_p2p3_batch_debug/sample014/
#   其中包括：
#     1. sample014 专属 MuJoCo XML
#     2. sample014_candidate.json
#     3. sample014_p3.json
#
# 输出：
#   diagnostics/current_v412/sodacan165_sample014_p4u6_viewer_debug/
#     terminal.txt
#     result.json
#     path_plan.json
#     p4u6_help.txt
#
# 当前流程位置：
#   Top-K prior -> P2/P3 已通过
#     -> P4U6 approach path
#     -> P4U1 ready-gated snap close
#     -> dynamic lift validation
#
# 不负责：
#   1. 不重新搜索数据集候选；
#   2. 不修改 legacy_final_demos；
#   3. 不修改 P4U1/P4U6 源码；
#   4. 不修改 can52 成功 demo；
#   5. 不写死新的抓握参数；
#   6. 不替代后续 P4H/P4H2 手型优化。

set -euo pipefail

cd ~/Projects/o7_mujoco_sim

if [[ -f "$HOME/mujoco_env/bin/activate" ]]; then
  source "$HOME/mujoco_env/bin/activate"
fi

RAW_SAMPLE="${SAMPLE_IDX:-014}"
printf -v SID "%03d" "$((10#$RAW_SAMPLE))"

OUTDIR="diagnostics/current_v412/sodacan165_sample${SID}_p4u6_viewer_debug"
mkdir -p "$OUTDIR"

LOG="$OUTDIR/terminal.txt"
exec > >(tee "$LOG") 2>&1

P4U6="scripts/05_execution_runner/run_v4_12p4u6_ik_path_record_demo_debug.py"
if [[ ! -f "$P4U6" ]]; then
  P4U6="$(find scripts legacy_final_demos -type f -name '*.py' 2>/dev/null | grep -Ei 'p4u6|ik_path|record_demo' | head -1 || true)"
fi

BATCH_ROOT="diagnostics/current_v412/sodacan165_topk_p2p3_batch_debug"
SAMPLE_DIR="$BATCH_ROOT/sample${SID}"

MODEL="$SAMPLE_DIR/initial_debug/scene/sodacan165_sample${SID}_from_can52_scene_debug.xml"
CANDIDATE="$SAMPLE_DIR/initial_debug/candidates/sample${SID}_candidate.json"
P3_JSON="$SAMPLE_DIR/sample${SID}_p3.json"

RESULT_JSON="$OUTDIR/result.json"
PATH_PLAN_JSON="$OUTDIR/path_plan.json"
HELP_TXT="$OUTDIR/p4u6_help.txt"

BEST_CONFIG="diagnostics/current_v412/v4_12p4h2_can52_contact_sequence_proxy_debug/best_contact_sequence_config_ctrlsplit_debug.json"
if [[ ! -f "$BEST_CONFIG" ]]; then
  BEST_CONFIG="$(find diagnostics records legacy_final_demos -type f -name '*config*.json' 2>/dev/null | grep -Ei 'contact_sequence|ctrlsplit|best' | head -1 || true)"
fi

echo "========== SODACAN165 SAMPLE${SID} P4U6 VIEWER =========="
echo "project     : $(pwd)"
echo "sample      : $SID"
echo "p4u6        : ${P4U6:-MISSING}"
echo "model       : $MODEL"
echo "candidate   : $CANDIDATE"
echo "p3_json     : $P3_JSON"
echo "best_config : ${BEST_CONFIG:-MISSING}"
echo "outdir      : $OUTDIR"
echo

for f in "$P4U6" "$MODEL" "$CANDIDATE" "$P3_JSON"; do
  if [[ -z "$f" || ! -f "$f" ]]; then
    echo "[ERROR] missing required file: $f"
    exit 2
  fi
done

python3 - <<PY
import json
from pathlib import Path

p = Path("$P3_JSON")
d = json.loads(p.read_text())
print("========== P3 QUICK CHECK ==========")
print("num_combos:", d.get("num_combos"))
print("num_pass  :", d.get("num_pass"))
ba = d.get("best_available") or {}
bp = d.get("best_pass")
print("best_available_status:", ba.get("precheck_status"))
print("best_available_score :", ba.get("score"))
print("best_pass_exists     :", bp is not None)
print("====================================")
if not d.get("num_pass", 0):
    raise SystemExit("[ERROR] P3 has no pass; do not run P4U6.")
PY

python3 "$P4U6" --help > "$HELP_TXT" 2>&1 || true

echo
echo "========== P4U6 HELP KEY FLAGS =========="
grep -E -- "--model|--candidate|--p3-json|--best-config|--which|--viewer|--keep-viewer-open|--enable-lift|--hard-servo-approach" "$HELP_TXT" || true
echo "help saved: $HELP_TXT"
echo

if [[ -x "./run_mujoco_clean.sh" ]]; then
  RUN_PREFIX=(./run_mujoco_clean.sh)
else
  RUN_PREFIX=(python3)
fi

ARGS=(
  "$P4U6"
  --model "$MODEL"
  --candidate "$CANDIDATE"
  --p3-json "$P3_JSON"
  --which best_available
  --object-body grasp_can
  --target-body fr3_link7
  --out "$RESULT_JSON"
  --plan-out "$PATH_PLAN_JSON"
  --viewer
  --keep-viewer-open
  --start-arm-mode zero_clamped
  --start-hold-duration 1.2
  --home-hold-duration 0.6
  --pre-hold-duration 0.8
  --grasp-settle-duration 0.35
  --close-duration 0.45
  --post-close-target-hold-duration 0.25
  --micro-squeeze-duration 0.35
  --micro-squeeze-fraction 0.08
  --finger-close-scale 0.92
  --thumb-pitch-from-finger-gain 0.24
  --grip-ready-stable-steps 8
  --min-live-non-thumb 1
  --opposition-cos-threshold -0.30
  --max-grip-disp 0.006
  --max-extra-disp-during-squeeze 0.003
  --approach-abort-disp 0.015
  --approach-min-clearance 0.003
  --grasp-path-min-clearance 0.001
  --plan-attempts 10
  --rrt-max-iters 4000
  --rrt-step 0.28
  --edge-step 0.035
  --goal-bias 0.20
  --shortcut-iters 400
  --joint-speed-rad-s 0.75
  --min-segment-duration 0.35
  --hard-servo-approach
  --enable-lift
  --lift-z 0.060
  --lift-duration 3.0
  --final-hold-duration 1.0
  --print-every-steps 100
  --log-every-steps 100
  --frame-sleep 0.0015
)

if grep -q -- "--best-config" "$HELP_TXT"; then
  if [[ -z "${BEST_CONFIG:-}" || ! -f "$BEST_CONFIG" ]]; then
    echo "[ERROR] P4U6 supports/requires --best-config, but no config json found."
    echo "Try:"
    echo "find diagnostics records legacy_final_demos -type f -name '*config*.json' | grep -Ei 'contact_sequence|ctrlsplit|best'"
    exit 3
  fi
  ARGS+=(--best-config "$BEST_CONFIG")
else
  echo "[WARN] P4U6 help does not show --best-config; skip best-config."
fi

echo
echo "========== RUN COMMAND =========="
printf '%q ' "${RUN_PREFIX[@]}" "${ARGS[@]}"
echo
echo "================================="
echo

"${RUN_PREFIX[@]}" "${ARGS[@]}"

echo
echo "========== DONE =========="
echo "terminal : $LOG"
echo "result   : $RESULT_JSON"
echo "path plan: $PATH_PLAN_JSON"

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
