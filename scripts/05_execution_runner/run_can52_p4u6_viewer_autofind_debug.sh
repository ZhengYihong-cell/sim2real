#!/usr/bin/env bash
# 脚本类型：
#   debug / launcher / viewer-autofind
#
# 用途：
#   自动定位当前 can52 P4U6 可视化抓握所需文件，并运行一次 viewer。
#
# 输入：
#   默认从 ~/Projects/o7_mujoco_sim 下自动寻找：
#     1. P4U6 runner
#     2. P4U1 runner 的固定 import 路径
#     3. can52 old ellipsoid proxy XML
#     4. best_candidate.json
#     5. best_p3.json
#     6. best contact sequence config
#
# 输出：
#   diagnostics/current_v412/run_can52_p4u6_viewer_autofind_terminal.txt
#   diagnostics/current_v412/run_can52_p4u6_viewer_autofind_result.json
#   diagnostics/current_v412/run_can52_p4u6_viewer_autofind_plan.json
#
# 当前流程位置：
#   用于验证 can52 的完整可视化路径：
#   collision-aware approach path -> P4U1 ready-gated snap close -> fixed-grip lift。
#
# 不负责：
#   1. 不重新搜索 grasp；
#   2. 不修改 legacy_final_demos；
#   3. 不修改 P4U1/P4U6 源码；
#   4. 不创建兼容软链接；
#   5. 不泛化到新物体。

set -euo pipefail

cd ~/Projects/o7_mujoco_sim

if [[ -f "$HOME/mujoco_env/bin/activate" ]]; then
  source "$HOME/mujoco_env/bin/activate"
fi

mkdir -p diagnostics/current_v412

LOG="diagnostics/current_v412/run_can52_p4u6_viewer_autofind_terminal.txt"
exec > >(tee "$LOG") 2>&1

echo "========== CAN52 P4U6 VIEWER AUTOFIND =========="
echo "project: $(pwd)"
echo "log    : $LOG"
echo

pick_first_file() {
  for p in "$@"; do
    if [[ -f "$p" ]]; then
      echo "$p"
      return 0
    fi
  done
  return 1
}

find_by_name_content() {
  local nameglob="$1"
  shift
  local musts=("$@")

  for root in diagnostics legacy_final_demos models records scripts; do
    [[ -d "$root" ]] || continue

    while IFS= read -r f; do
      local ok=1
      for m in "${musts[@]}"; do
        [[ -z "$m" ]] && continue
        if ! grep -q -- "$m" "$f" 2>/dev/null; then
          ok=0
          break
        fi
      done

      if [[ "$ok" == "1" ]]; then
        echo "$f"
        return 0
      fi
    done < <(find "$root" -type f -name "$nameglob" 2>/dev/null | sort)
  done

  return 1
}

P4U1_EXPECTED="scripts/05_execution_runner/run_v4_12p4u1_precontact_snap_close_debug.py"
P4U6_EXPECTED="scripts/05_execution_runner/run_v4_12p4u6_ik_path_record_demo_debug.py"

MODEL_EXPECTED="diagnostics/current_v412/v4_12p4t2_scene_can52_contact_stable_old_ellipsoid_proxy.xml"
CAND_EXPECTED="diagnostics/current_v412/v4_12p4h2_can52_contact_sequence_proxy_debug/best_candidate.json"
P3_EXPECTED="diagnostics/current_v412/v4_12p4h2_can52_contact_sequence_proxy_debug/best_p3.json"
CFG_EXPECTED="diagnostics/current_v412/v4_12p4h2_can52_contact_sequence_proxy_debug/best_contact_sequence_config_ctrlsplit_debug.json"

P4U6="${P4U6:-}"
MODEL="${MODEL:-}"
CANDIDATE="${CANDIDATE:-}"
P3_JSON="${P3_JSON:-}"
BEST_CONFIG="${BEST_CONFIG:-}"

if [[ -z "$P4U6" ]]; then
  P4U6="$(pick_first_file "$P4U6_EXPECTED" 2>/dev/null || true)"
fi
if [[ -z "$P4U6" ]]; then
  P4U6="$(find_by_name_content "*.py" "V4.12P4U6" "collision-aware" 2>/dev/null || true)"
fi

if [[ -z "$MODEL" ]]; then
  MODEL="$(pick_first_file "$MODEL_EXPECTED" 2>/dev/null || true)"
fi
if [[ -z "$MODEL" ]]; then
  MODEL="$(find_by_name_content "*.xml" "grasp_can" "ellipsoid" 2>/dev/null || true)"
fi

if [[ -z "$CANDIDATE" ]]; then
  CANDIDATE="$(pick_first_file "$CAND_EXPECTED" 2>/dev/null || true)"
fi
if [[ -z "$CANDIDATE" ]]; then
  CANDIDATE="$(find_by_name_content "*candidate*.json" "T_object_target" "grasp_can" 2>/dev/null || true)"
fi

if [[ -z "$P3_JSON" ]]; then
  P3_JSON="$(pick_first_file "$P3_EXPECTED" 2>/dev/null || true)"
fi
if [[ -z "$P3_JSON" ]]; then
  P3_JSON="$(find_by_name_content "*p3*.json" "q_grasp" "fr3_joint7" 2>/dev/null || true)"
fi
if [[ -z "$P3_JSON" ]]; then
  P3_JSON="$(find_by_name_content "*.json" "q_grasp" "fr3_joint7" 2>/dev/null || true)"
fi

if [[ -z "$BEST_CONFIG" ]]; then
  BEST_CONFIG="$(pick_first_file "$CFG_EXPECTED" 2>/dev/null || true)"
fi
if [[ -z "$BEST_CONFIG" ]]; then
  BEST_CONFIG="$(find_by_name_content "*config*.json" "thumb_cmc_roll" "index_mcp_pitch" 2>/dev/null || true)"
fi

echo "P4U1 expected : $P4U1_EXPECTED"
echo "P4U6          : ${P4U6:-MISSING}"
echo "MODEL         : ${MODEL:-MISSING}"
echo "CANDIDATE     : ${CANDIDATE:-MISSING}"
echo "P3_JSON       : ${P3_JSON:-MISSING}"
echo "BEST_CONFIG   : ${BEST_CONFIG:-MISSING}"
echo

if [[ ! -f "$P4U1_EXPECTED" ]]; then
  echo "[ERROR] P4U6 脚本内部固定 import 这个 P4U1 路径，但当前不存在："
  echo "        $P4U1_EXPECTED"
  echo
  echo "你可以先运行下面命令把实际 P4U1 找出来："
  echo "find scripts legacy_final_demos -type f -name '*.py' | grep -i 'p4u1\\|snap\\|ready'"
  exit 2
fi

missing=0
for f in "$P4U6" "$MODEL" "$CANDIDATE" "$P3_JSON" "$BEST_CONFIG"; do
  if [[ -z "$f" || ! -f "$f" ]]; then
    missing=1
  fi
done

if [[ "$missing" == "1" ]]; then
  echo "[ERROR] 自动定位失败。请把上面 MISSING 的项截图或终端输出发我。"
  echo
  echo "辅助定位命令："
  echo "find diagnostics legacy_final_demos models records scripts -type f \\( -name '*.json' -o -name '*.xml' -o -name '*.py' \\) | grep -Ei 'can52|p4u6|p4u1|candidate|best_p3|config|proxy|ellipsoid|snap'"
  exit 3
fi

OUT_JSON="diagnostics/current_v412/run_can52_p4u6_viewer_autofind_result.json"
PLAN_JSON="diagnostics/current_v412/run_can52_p4u6_viewer_autofind_plan.json"

echo "========== RUN P4U6 VIEWER =========="
echo "out      : $OUT_JSON"
echo "plan_out : $PLAN_JSON"
echo

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
  --out "$OUT_JSON" \
  --plan-out "$PLAN_JSON" \
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
echo "terminal log: $LOG"
echo "result json : $OUT_JSON"
echo "plan json   : $PLAN_JSON"
