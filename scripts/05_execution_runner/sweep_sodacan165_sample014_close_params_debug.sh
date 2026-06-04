#!/usr/bin/env bash
# 脚本类型：
#   debug / execution-runner / close-parameter-sweep
#
# 用途：
#   对 sem-SodaCan-16526 sample014 做小范围闭合参数扫描。
#   当前路径规划和姿态已经基本可用，但 P4U1 只检测到 thumb 接触，
#   没有形成 thumb + finger 对抗，因此不允许 lift。
#
# 输入：
#   1. sample014 MuJoCo scene
#   2. sample014 candidate
#   3. contact-free P3 json
#   4. sodacan candidate-derived best_config
#
# 输出：
#   diagnostics/current_v412/sodacan165_sample014_close_sweep_debug/
#     run_*/
#       terminal.txt
#       result.json
#       path_plan.json
#     sweep_summary.txt
#     sweep_summary.json
#
# 当前流程位置：
#   P4U6 路径已通过
#     -> P4U1 close 参数小范围扫描
#     -> 找到 thumb + finger 对抗
#     -> 再 viewer 复查和 lift
#
# 不负责：
#   1. 不修改 P4U1/P4U6 源码；
#   2. 不修改 can52 legacy demo；
#   3. 不重新跑 P2/P3；
#   4. 不对单个 sample 写死最终参数；
#   5. 不替代后续 P4H/P4H2 正式手型优化。

set -euo pipefail

cd ~/Projects/o7_mujoco_sim

if [[ -f "$HOME/mujoco_env/bin/activate" ]]; then
  source "$HOME/mujoco_env/bin/activate"
fi

OUTROOT="diagnostics/current_v412/sodacan165_sample014_close_sweep_debug"
mkdir -p "$OUTROOT"

LOG_ALL="$OUTROOT/terminal_all.txt"
exec > >(tee "$LOG_ALL") 2>&1

P4U6="scripts/05_execution_runner/run_v4_12p4u6_ik_path_record_demo_debug.py"
MODEL="diagnostics/current_v412/sodacan165_topk_p2p3_batch_debug/sample014/initial_debug/scene/sodacan165_sample014_from_can52_scene_debug.xml"
CANDIDATE="diagnostics/current_v412/sodacan165_topk_p2p3_batch_debug/sample014/initial_debug/candidates/sample014_candidate.json"
P3_JSON="diagnostics/current_v412/sodacan165_sample014_p4u6_viewer_debug/sample014_p3_contactfree_goal.json"
BEST_CONFIG="diagnostics/current_v412/sodacan165_sample014_p4u6_viewer_debug/sodacan_sample014_best_config_from_candidate_debug.json"

for f in "$P4U6" "$MODEL" "$CANDIDATE" "$P3_JSON" "$BEST_CONFIG"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] missing file: $f"
    exit 2
  fi
done

echo "========== SODACAN165 SAMPLE014 CLOSE SWEEP =========="
echo "outroot: $OUTROOT"
echo

RUN_ID=0

# 先小范围扫，不要太大，避免把物体推飞。
FINGER_SCALES="0.96 1.00 1.06 1.12"
MICRO_FRACS="0.00 0.04 0.08"
MAX_DISPS="0.018"

for FS in $FINGER_SCALES; do
  for MF in $MICRO_FRACS; do
    for MD in $MAX_DISPS; do
      RUN_ID=$((RUN_ID + 1))
      printf -v RID "%02d" "$RUN_ID"
      RDIR="$OUTROOT/run_${RID}_fs${FS}_mf${MF}_md${MD}"
      mkdir -p "$RDIR"

      echo
      echo "============================================================"
      echo "RUN $RID | finger_close_scale=$FS micro_squeeze_fraction=$MF max_grip_disp=$MD"
      echo "============================================================"

      set +e
      ./run_mujoco_clean.sh "$P4U6" \
        --model "$MODEL" \
        --candidate "$CANDIDATE" \
        --p3-json "$P3_JSON" \
        --best-config "$BEST_CONFIG" \
        --which best_available \
        --object-body grasp_can \
        --target-body fr3_link7 \
        --out "$RDIR/result.json" \
        --plan-out "$RDIR/path_plan.json" \
        --start-arm-mode zero_clamped \
        --start-hold-duration 0.4 \
        --home-hold-duration 0.2 \
        --pre-hold-duration 0.25 \
        --grasp-settle-duration 0.25 \
        --close-duration 0.45 \
        --post-close-target-hold-duration 0.25 \
        --micro-squeeze-duration 0.35 \
        --micro-squeeze-fraction "$MF" \
        --finger-close-scale "$FS" \
        --thumb-pitch-from-finger-gain 0.24 \
        --grip-ready-stable-steps 5 \
        --min-live-non-thumb 1 \
        --opposition-cos-threshold -0.30 \
        --max-grip-disp "$MD" \
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
        --lift-z 0.060 \
        --lift-duration 2.0 \
        --final-hold-duration 0.4 \
        --print-every-steps 200 \
        --log-every-steps 200 \
        > "$RDIR/terminal.txt" 2>&1
      RET=$?
      set -e

      echo "return_code=$RET" | tee "$RDIR/return_code.txt"
      tail -35 "$RDIR/terminal.txt" || true
    done
  done
done

echo
echo "========== COLLECT SWEEP SUMMARY =========="

python3 - <<'PY'
import json
from pathlib import Path

outroot = Path("diagnostics/current_v412/sodacan165_sample014_close_sweep_debug")
rows = []

for rdir in sorted(outroot.glob("run_*")):
    row = {"run": rdir.name}
    result = rdir / "result.json"
    term = rdir / "terminal.txt"

    if result.exists():
        try:
            d = json.loads(result.read_text())
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
                row[k] = d.get(k)
        except Exception as e:
            row["parse_error"] = repr(e)
    else:
        row["missing_result"] = True

    if term.exists():
        txt = term.read_text(errors="ignore")
        if "[NO_LIFT]" in txt:
            row["no_lift"] = True
        if "grip_ready          : True" in txt or "grip_ready: True" in txt:
            row["grip_ready_text"] = True

    rows.append(row)

def has_non_thumb(row):
    g = row.get("final_groups")
    if not isinstance(g, dict):
        return 0
    return sum(v for k, v in g.items() if k != "thumb" and isinstance(v, int))

def score(row):
    ready = 1 if row.get("grip_ready") else 0
    non_thumb = has_non_thumb(row)
    rise = row.get("final_object_rise")
    if not isinstance(rise, (int, float)):
        rise = -999
    disp = row.get("final_object_disp")
    if not isinstance(disp, (int, float)):
        disp = 999
    stable = row.get("max_stable_count")
    if not isinstance(stable, (int, float)):
        stable = 0
    return (-ready, -non_thumb, -stable, -rise, disp)

rows_sorted = sorted(rows, key=score)

summary = {
    "format": "sodacan165_sample014_close_sweep_summary_debug_v1",
    "rows_sorted": rows_sorted,
}

(outroot / "sweep_summary.json").write_text(json.dumps(summary, indent=2, ensure_ascii=False))

lines = []
lines.append("========== SODACAN165 SAMPLE014 CLOSE SWEEP SUMMARY ==========")
for r in rows_sorted:
    lines.append(
        f"{r.get('run')} "
        f"ready={r.get('grip_ready')} "
        f"success={r.get('success')} "
        f"stop={r.get('stop_reason')} "
        f"groups={r.get('final_groups')} "
        f"opp={r.get('final_opposition_cos')} "
        f"disp={r.get('final_object_disp')} "
        f"rise={r.get('final_object_rise')} "
        f"stable={r.get('max_stable_count')}"
    )
lines.append("==============================================================")
txt = "\n".join(lines) + "\n"
(outroot / "sweep_summary.txt").write_text(txt)
print(txt)
PY

echo
echo "========== DONE =========="
echo "summary: $OUTROOT/sweep_summary.txt"
