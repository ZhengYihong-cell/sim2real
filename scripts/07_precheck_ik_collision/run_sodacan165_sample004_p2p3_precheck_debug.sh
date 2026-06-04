#!/usr/bin/env bash
# 脚本类型：
#   debug / precheck / p2-p3-launcher
#
# 用途：
#   对 sem-SodaCan-16526 的 sample004 candidate 执行：
#     P2 Pinocchio 多 seed IK
#     P3 MuJoCo 路径碰撞预检
#
# 输入：
#   1. sample004 专属 MuJoCo scene XML
#   2. sample004_candidate.json
#   3. FR3+O7 URDF
#   4. 可选 runner_json seed
#
# 输出：
#   diagnostics/current_v412/sodacan165_sample004_p2p3_precheck_debug/
#     sample004_p2.json
#     sample004_p3.json
#     sample004_best_plan.json
#     terminal.txt
#
# 当前流程位置：
#   dataset prior candidate
#     -> P2 IK
#     -> P3 MuJoCo collision precheck
#     -> 后续 P4E/P4H/P4H2 或 P4U6/P4U1
#
# 不负责：
#   1. 不运行 viewer；
#   2. 不执行动态抓取；
#   3. 不修改 can52 legacy demo；
#   4. 不修改 candidate；
#   5. 不判断最终 lift 是否成功。

set -euo pipefail

cd ~/Projects/o7_mujoco_sim

if [[ -f "$HOME/mujoco_env/bin/activate" ]]; then
  source "$HOME/mujoco_env/bin/activate"
fi

OUTDIR="diagnostics/current_v412/sodacan165_sample004_p2p3_precheck_debug"
mkdir -p "$OUTDIR"

LOG="$OUTDIR/terminal.txt"
exec > >(tee "$LOG") 2>&1

URDF="models/fr3_o7/fr3_o7_mujoco_ready_obj.urdf"
MODEL="diagnostics/current_v412/sodacan165_sample004_initial_debug/scene/sodacan165_sample004_from_can52_scene_debug.xml"
CANDIDATE="diagnostics/current_v412/sodacan165_sample004_initial_debug/candidates/sample004_candidate.json"

P2_SCRIPT="scripts/07_precheck_ik_collision/run_v4_12p2_pinocchio_multiseed_ik_debug.py"
P3_SCRIPT="scripts/07_precheck_ik_collision/run_v4_12p3_pinocchio_ik_plus_mujoco_collision_precheck_debug.py"

P2_JSON="$OUTDIR/sample004_p2.json"
P3_JSON="$OUTDIR/sample004_p3.json"
PLAN_JSON="$OUTDIR/sample004_best_plan.json"

RUNNER_JSON="diagnostics/current_v412/v4_12a_can52_strict_clearance_debug_runner.json"
if [[ ! -f "$RUNNER_JSON" ]]; then
  RUNNER_JSON="$(find diagnostics legacy_final_demos records -type f -name '*runner*.json' 2>/dev/null | grep -Ei 'can52|strict|v4_12a' | head -1 || true)"
fi

echo "========== SODACAN165 SAMPLE004 P2/P3 PRECHECK =========="
echo "project     : $(pwd)"
echo "urdf        : $URDF"
echo "model       : $MODEL"
echo "candidate   : $CANDIDATE"
echo "runner_json : ${RUNNER_JSON:-NONE}"
echo "outdir      : $OUTDIR"
echo

for f in "$URDF" "$MODEL" "$CANDIDATE" "$P2_SCRIPT" "$P3_SCRIPT"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] missing file: $f"
    exit 2
  fi
done

echo
echo "========== RUN P2: PINOCCHIO MULTI-SEED IK =========="
python3 "$P2_SCRIPT" \
  --urdf "$URDF" \
  --model "$MODEL" \
  --candidate "$CANDIDATE" \
  --runner-json "${RUNNER_JSON:-}" \
  --object-body grasp_can \
  --target-frame fr3_link7 \
  --out "$P2_JSON" \
  --random-seeds 16 \
  --random-std 0.6 \
  --max-iters 350 \
  --pos-tol 0.00035 \
  --rot-tol 0.0035 \
  --rot-weight 0.55

echo
echo "========== RUN P3: MUJOCO COLLISION PRECHECK =========="
python3 "$P3_SCRIPT" \
  --p2-json "$P2_JSON" \
  --model "$MODEL" \
  --candidate "$CANDIDATE" \
  --object-body grasp_can \
  --out "$P3_JSON" \
  --best-plan-out "$PLAN_JSON" \
  --top-per-target 8 \
  --max-combos 512 \
  --path-samples 40 \
  --min-hand-support-clearance 0.0 \
  --min-fr3-object-clearance 0.0 \
  --max-grasp-hand-object-distance 0.050 \
  --min-joint-margin 0.0

echo
echo "========== QUICK SUMMARY =========="
python3 - <<'PY'
import json
from pathlib import Path

p3 = Path("diagnostics/current_v412/sodacan165_sample004_p2p3_precheck_debug/sample004_p3.json")
plan = Path("diagnostics/current_v412/sodacan165_sample004_p2p3_precheck_debug/sample004_best_plan.json")

d = json.loads(p3.read_text())
print("num_combos:", d.get("num_combos"))
print("num_pass  :", d.get("num_pass"))

for key in ["best_pass", "best_available"]:
    item = d.get(key)
    print()
    print(key + ":")
    if item is None:
        print("  None")
        continue
    print("  precheck_status:", item.get("precheck_status"))
    print("  score:", item.get("score"))
    print("  pre_seed:", item.get("pre_seed"))
    print("  grasp_seed:", item.get("grasp_seed"))
    print("  lift_seed:", item.get("lift_seed"))
    print("  min_path_hand_support_clearance:", item.get("min_path_hand_support_clearance"))
    print("  min_path_fr3_object_clearance:", item.get("min_path_fr3_object_clearance"))
    print("  static_grasp_closed_hand_object_distance:", item.get("static_grasp_closed_hand_object_distance"))
    print("  combo_min_joint_margin:", item.get("combo_min_joint_margin"))
    print("  hard_reasons:", item.get("hard_reasons"))

if plan.exists():
    q = json.loads(plan.read_text())
    print()
    print("best_plan:", plan)
    print("best_plan_status:", q.get("precheck_status"))
PY

echo
echo "========== DONE =========="
echo "terminal: $LOG"
echo "p2      : $P2_JSON"
echo "p3      : $P3_JSON"
echo "plan    : $PLAN_JSON"
