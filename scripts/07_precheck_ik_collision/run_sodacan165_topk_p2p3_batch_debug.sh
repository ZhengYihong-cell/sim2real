#!/usr/bin/env bash
# 脚本类型：
#   debug / precheck / batch-p2-p3-screening
#
# 用途：
#   对 sem-SodaCan-16526 的 Top-K 数据集候选批量执行：
#     1. candidate + scene 生成
#     2. P2 Pinocchio 多 seed IK
#     3. P3 MuJoCo 碰撞预检
#   用于从多个数据集先验中筛出更适合当前 tabletop/support/FR3 场景的候选。
#
# 输入：
#   默认 sample 列表：
#     4 14 23 17 22 5 2 6 1 3
#
# 输出：
#   diagnostics/current_v412/sodacan165_topk_p2p3_batch_debug/
#     sampleXXX/
#       candidate + scene
#       sampleXXX_p2.json
#       sampleXXX_p3.json
#       sampleXXX_best_plan.json
#       terminal.txt
#     batch_summary.json
#     batch_summary.txt
#     terminal_all.txt
#
# 当前流程位置：
#   Top-K dataset prior
#     -> candidate / scene
#     -> P2 IK
#     -> P3 collision precheck
#     -> 选择可进入 P4E/P4H/P4H2 或 P4U6 的候选。
#
# 不负责：
#   1. 不运行 viewer；
#   2. 不执行动态抓取；
#   3. 不修改 can52 legacy demo；
#   4. 不手工调单个 sample；
#   5. 不修改 P4U1/P4U6 成功逻辑。

set -euo pipefail

cd ~/Projects/o7_mujoco_sim

if [[ -f "$HOME/mujoco_env/bin/activate" ]]; then
  source "$HOME/mujoco_env/bin/activate"
fi

OUTROOT="diagnostics/current_v412/sodacan165_topk_p2p3_batch_debug"
mkdir -p "$OUTROOT"

LOG_ALL="$OUTROOT/terminal_all.txt"
exec > >(tee "$LOG_ALL") 2>&1

SAMPLES="${SAMPLES:-4 14 23 17 22 5 2 6 1 3}"

BUILDER="scripts/00_maintenance/build_sodacan165_initial_candidate_scene_debug.py"
P2_SCRIPT="scripts/07_precheck_ik_collision/run_v4_12p2_pinocchio_multiseed_ik_debug.py"
P3_SCRIPT="scripts/07_precheck_ik_collision/run_v4_12p3_pinocchio_ik_plus_mujoco_collision_precheck_debug.py"

URDF="models/fr3_o7/fr3_o7_mujoco_ready_obj.urdf"
RUNNER_JSON="diagnostics/current_v412/v4_12a_can52_strict_clearance_debug_runner.json"

for f in "$BUILDER" "$P2_SCRIPT" "$P3_SCRIPT" "$URDF"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] missing file: $f"
    exit 2
  fi
done

if [[ ! -f "$RUNNER_JSON" ]]; then
  RUNNER_JSON="$(find diagnostics legacy_final_demos records -type f -name '*runner*.json' 2>/dev/null | grep -Ei 'can52|strict|v4_12a' | head -1 || true)"
fi

echo "========== SODACAN165 TOP-K P2/P3 BATCH =========="
echo "project     : $(pwd)"
echo "samples     : $SAMPLES"
echo "outroot     : $OUTROOT"
echo "runner_json : ${RUNNER_JSON:-NONE}"
echo

for IDX in $SAMPLES; do
  printf -v SID "%03d" "$IDX"

  echo
  echo "============================================================"
  echo "SAMPLE $SID"
  echo "============================================================"

  SDIR="$OUTROOT/sample${SID}"
  mkdir -p "$SDIR"

  {
    echo "========== BUILD CANDIDATE + SCENE: sample${SID} =========="
    python3 "$BUILDER" \
      --sample-indices "$IDX" \
      --out-dir "$SDIR/initial_debug"

    MODEL="$SDIR/initial_debug/scene/sodacan165_sample${SID}_from_can52_scene_debug.xml"
    CANDIDATE="$SDIR/initial_debug/candidates/sample${SID}_candidate.json"

    P2_JSON="$SDIR/sample${SID}_p2.json"
    P3_JSON="$SDIR/sample${SID}_p3.json"
    PLAN_JSON="$SDIR/sample${SID}_best_plan.json"

    echo
    echo "========== RUN P2: sample${SID} =========="
    python3 "$P2_SCRIPT" \
      --urdf "$URDF" \
      --model "$MODEL" \
      --candidate "$CANDIDATE" \
      --runner-json "${RUNNER_JSON:-}" \
      --object-body grasp_can \
      --target-frame fr3_link7 \
      --out "$P2_JSON" \
      --random-seeds 12 \
      --random-std 0.6 \
      --max-iters 350 \
      --pos-tol 0.00035 \
      --rot-tol 0.0035 \
      --rot-weight 0.55

    echo
    echo "========== RUN P3: sample${SID} =========="
    python3 "$P3_SCRIPT" \
      --p2-json "$P2_JSON" \
      --model "$MODEL" \
      --candidate "$CANDIDATE" \
      --object-body grasp_can \
      --out "$P3_JSON" \
      --best-plan-out "$PLAN_JSON" \
      --top-per-target 6 \
      --max-combos 216 \
      --path-samples 36 \
      --min-hand-support-clearance 0.0 \
      --min-fr3-object-clearance 0.0 \
      --max-grasp-hand-object-distance 0.050 \
      --min-joint-margin 0.0

  } 2>&1 | tee "$SDIR/terminal.txt" || {
    echo "[WARN] sample${SID} failed during build/P2/P3, continue."
  }

done

echo
echo "========== COLLECT BATCH SUMMARY =========="

python3 - <<'PY'
import json
from pathlib import Path

outroot = Path("diagnostics/current_v412/sodacan165_topk_p2p3_batch_debug")
rows = []

for sdir in sorted(outroot.glob("sample*")):
    sid = sdir.name.replace("sample", "")
    p3 = sdir / f"sample{sid}_p3.json"
    plan = sdir / f"sample{sid}_best_plan.json"

    row = {
        "sample": sid,
        "sample_index": int(sid),
        "p3_exists": p3.exists(),
        "plan_exists": plan.exists(),
    }

    if p3.exists():
        try:
            d = json.loads(p3.read_text())
            row["num_combos"] = d.get("num_combos")
            row["num_pass"] = d.get("num_pass")
            bp = d.get("best_pass")
            ba = d.get("best_available")
            row["best_pass_status"] = None if bp is None else bp.get("precheck_status")
            if ba is not None:
                row["best_available_status"] = ba.get("precheck_status")
                row["best_available_score"] = ba.get("score")
                row["min_path_hand_support_clearance"] = ba.get("min_path_hand_support_clearance")
                row["min_path_fr3_object_clearance"] = ba.get("min_path_fr3_object_clearance")
                row["static_grasp_closed_hand_object_distance"] = ba.get("static_grasp_closed_hand_object_distance")
                row["static_grasp_closed_hand_support_clearance"] = ba.get("static_grasp_closed_hand_support_clearance")
                row["combo_min_joint_margin"] = ba.get("combo_min_joint_margin")
                row["hard_reasons"] = ba.get("hard_reasons")
        except Exception as e:
            row["p3_parse_error"] = repr(e)

    rows.append(row)

def sort_key(r):
    num_pass = r.get("num_pass")
    if num_pass is None:
        num_pass = -1
    hs = r.get("min_path_hand_support_clearance")
    if hs is None:
        hs = -999
    fo = r.get("min_path_fr3_object_clearance")
    if fo is None:
        fo = -999
    score = r.get("best_available_score")
    if score is None:
        score = 1e99
    return (-num_pass, -hs, -fo, score)

rows_sorted = sorted(rows, key=sort_key)

summary = {
    "format": "sodacan165_topk_p2p3_batch_summary_debug_v1",
    "outroot": str(outroot),
    "rows_sorted": rows_sorted,
}

(outroot / "batch_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

lines = []
lines.append("========== SODACAN165 TOP-K P2/P3 BATCH SUMMARY ==========")
for r in rows_sorted:
    lines.append(
        f"sample={r['sample']} "
        f"pass={r.get('num_pass')} "
        f"status={r.get('best_available_status')} "
        f"score={r.get('best_available_score')} "
        f"HS={r.get('min_path_hand_support_clearance')} "
        f"FO={r.get('min_path_fr3_object_clearance')} "
        f"GO={r.get('static_grasp_closed_hand_object_distance')} "
        f"HSc={r.get('static_grasp_closed_hand_support_clearance')} "
        f"margin={r.get('combo_min_joint_margin')}"
    )
    reasons = r.get("hard_reasons") or []
    for rr in reasons[:3]:
        lines.append(f"  - {rr}")
lines.append("===========================================================")

txt = "\n".join(lines) + "\n"
(outroot / "batch_summary.txt").write_text(txt)
print(txt)
PY

echo
echo "========== DONE =========="
echo "all log : $LOG_ALL"
echo "summary : $OUTROOT/batch_summary.txt"
