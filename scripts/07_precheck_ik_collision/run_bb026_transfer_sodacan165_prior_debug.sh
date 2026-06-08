#!/usr/bin/env bash
# 脚本类型：
#   debug / transfer-test / prior-reuse / light-p2p3
#
# 用途：
#   不再继续 top generator。
#   直接把已经成功的 SodaCan165 sample014 抓握先验迁移到 BB026：
#     1. 复制成功 demo 的 T_object_target；
#     2. 复制成功 demo 的 hand ctrl / best_config；
#     3. 放到 BB026 stable scene 上；
#     4. 轻量 P2/P3；
#     5. 生成 viewer 运行脚本。
#
# 输入：
#   legacy_final_demos/final_sodacan165_sample014_p4u6_ready_gated_snap_demo/inputs/candidate.json
#   legacy_final_demos/final_sodacan165_sample014_p4u6_ready_gated_snap_demo/inputs/best_config.json
#   diagnostics/current_v412/sodacan_bb026_topk_p2p3_batch_debug/sample014/initial_debug/candidates/sample014_candidate.json
#   diagnostics/current_v412/sodacan_bb026_top_grasp_light_debug/scenes/stable_scene_seed014.xml
#
# 输出：
#   diagnostics/current_v412/bb026_transfer_sodacan165_prior_debug/
#     candidate.json
#     best_config.json
#     p2.json
#     p3.json
#     plan.json
#     summary.txt
#   scripts/05_execution_runner/run_bb026_transfer_sodacan165_prior_viewer_debug.sh
#
# 当前流程位置：
#   BB026 top/soft rerank 已证明抓型生成不对
#     -> 直接迁移成功 SodaCan165 prior
#     -> 验证是 selector/generator 问题，还是 object-frame/mesh 不一致问题。
#
# 不负责：
#   1. 不修改 legacy_final_demos；
#   2. 不修改 P4U1/P4U6 源码；
#   3. 不继续调 top generator；
#   4. 不把该迁移结果直接当最终泛化算法。

set -euo pipefail

cd "$HOME/Projects/o7_mujoco_sim"

if [[ -f "$HOME/mujoco_env/bin/activate" ]]; then
  source "$HOME/mujoco_env/bin/activate"
fi

OUTROOT="diagnostics/current_v412/bb026_transfer_sodacan165_prior_debug"
mkdir -p "$OUTROOT"

SRC_CAND="legacy_final_demos/final_sodacan165_sample014_p4u6_ready_gated_snap_demo/inputs/candidate.json"
SRC_BEST="legacy_final_demos/final_sodacan165_sample014_p4u6_ready_gated_snap_demo/inputs/best_config.json"

BB026_BASE_CAND="diagnostics/current_v412/sodacan_bb026_topk_p2p3_batch_debug/sample014/initial_debug/candidates/sample014_candidate.json"
BB026_SCENE="diagnostics/current_v412/sodacan_bb026_top_grasp_light_debug/scenes/stable_scene_seed014.xml"

OUT_CAND="$OUTROOT/candidate.json"
OUT_BEST="$OUTROOT/best_config.json"
P2_JSON="$OUTROOT/p2.json"
P3_JSON="$OUTROOT/p3.json"
PLAN_JSON="$OUTROOT/plan.json"

P2_SCRIPT="scripts/07_precheck_ik_collision/run_v4_12p2_pinocchio_multiseed_ik_debug.py"
P3_SCRIPT="scripts/07_precheck_ik_collision/run_v4_12p3_pinocchio_ik_plus_mujoco_collision_precheck_debug.py"
URDF="models/fr3_o7/fr3_o7_mujoco_ready_obj.urdf"
RUNNER_JSON="diagnostics/current_v412/v4_12a_can52_strict_clearance_debug_runner.json"

for f in "$SRC_CAND" "$SRC_BEST" "$BB026_BASE_CAND" "$BB026_SCENE" "$P2_SCRIPT" "$P3_SCRIPT" "$URDF"; do
  if [[ ! -f "$f" ]]; then
    echo "[ERROR] missing file: $f"
    exit 2
  fi
done

python3 - <<'PY'
from pathlib import Path
import json
import copy

PROJECT = Path.home() / "Projects/o7_mujoco_sim"

SRC_CAND = PROJECT / "legacy_final_demos/final_sodacan165_sample014_p4u6_ready_gated_snap_demo/inputs/candidate.json"
SRC_BEST = PROJECT / "legacy_final_demos/final_sodacan165_sample014_p4u6_ready_gated_snap_demo/inputs/best_config.json"
BB026_BASE = PROJECT / "diagnostics/current_v412/sodacan_bb026_topk_p2p3_batch_debug/sample014/initial_debug/candidates/sample014_candidate.json"

OUT_CAND = PROJECT / "diagnostics/current_v412/bb026_transfer_sodacan165_prior_debug/candidate.json"
OUT_BEST = PROJECT / "diagnostics/current_v412/bb026_transfer_sodacan165_prior_debug/best_config.json"

JOINTS = [
    "thumb_cmc_roll",
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]

def load(p):
    return json.loads(Path(p).read_text())

def save(p, d):
    Path(p).parent.mkdir(parents=True, exist_ok=True)
    Path(p).write_text(json.dumps(d, indent=2, ensure_ascii=False))

def find_T_path(d):
    paths = [
        ("target", "T_object_target"),
        ("target", "T_object_fr3_link7"),
        ("T_object_target",),
        ("T_object_fr3_link7",),
    ]
    for path in paths:
        cur = d
        ok = True
        for k in path:
            if isinstance(cur, dict) and k in cur:
                cur = cur[k]
            else:
                ok = False
                break
        if ok:
            return path, cur
    raise RuntimeError("cannot find T_object_target / T_object_fr3_link7")

def set_path(d, path, val):
    cur = d
    for k in path[:-1]:
        cur = cur[k]
    cur[path[-1]] = val

def get_ctrl(d):
    ctrl = d.get("hand", {}).get("o7_active_ctrl")
    if not isinstance(ctrl, dict):
        raise RuntimeError("source candidate missing hand.o7_active_ctrl")
    return {j: float(ctrl[j]) for j in JOINTS}

src_cand = load(SRC_CAND)
src_best = load(SRC_BEST)
bb = load(BB026_BASE)

src_T_path, src_T = find_T_path(src_cand)
bb_T_path, old_bb_T = find_T_path(bb)
src_ctrl = get_ctrl(src_cand)

# 关键：只把成功 can 的 object-frame hand target 迁移到 BB026，
# 其余 object/body/scene 信息沿用 BB026 candidate。
new_cand = copy.deepcopy(bb)
set_path(new_cand, bb_T_path, src_T)
new_cand.setdefault("hand", {})
new_cand["hand"]["o7_active_ctrl"] = src_ctrl
new_cand["candidate_name"] = "bb026_transfer_sodacan165_sample014_prior_debug"
new_cand["transfer_debug_note"] = {
    "type": "transfer_successful_sodacan165_prior_to_bb026",
    "src_candidate": str(SRC_CAND),
    "src_T_path": list(src_T_path),
    "bb026_base_candidate": str(BB026_BASE),
    "bb026_T_path": list(bb_T_path),
    "reason": "BB026 top generator produced no contact; test whether successful can-like object-frame grasp transfers directly.",
}
save(OUT_CAND, new_cand)

# best_config 直接复制成功 demo，但显式保证 runner 读取的 ctrl 是成功 prior 的 ctrl。
new_best = copy.deepcopy(src_best)
hc = new_best.setdefault("best_record", {}).setdefault("hand_config", {})
before = hc.get("ctrl", {})
hc["ctrl"] = src_ctrl
hc["source"] = "transferred_sodacan165_success_ctrl_to_bb026"
new_best["bb026_transfer_debug_note"] = {
    "type": "best_config_from_successful_sodacan165_demo",
    "src_best_config": str(SRC_BEST),
    "before_ctrl": before,
    "after_ctrl": src_ctrl,
}
save(OUT_BEST, new_best)

print("========== TRANSFER PRIOR DONE ==========")
print("out_candidate:", OUT_CAND)
print("out_best     :", OUT_BEST)
print("src_T_path   :", src_T_path)
print("bb_T_path    :", bb_T_path)
print("src_ctrl     :", src_ctrl)
PY

echo
echo "========== RUN LIGHT P2 =========="
python3 "$P2_SCRIPT" \
  --urdf "$URDF" \
  --model "$BB026_SCENE" \
  --candidate "$OUT_CAND" \
  --runner-json "$RUNNER_JSON" \
  --object-body grasp_can \
  --target-frame fr3_link7 \
  --out "$P2_JSON" \
  --random-seeds 8 \
  --random-std 0.55 \
  --max-iters 300 \
  --pos-tol 0.0005 \
  --rot-tol 0.005 \
  --rot-weight 0.50 \
  2>&1 | tee "$OUTROOT/terminal_p2.txt"

echo
echo "========== RUN LIGHT P3 =========="
python3 "$P3_SCRIPT" \
  --p2-json "$P2_JSON" \
  --model "$BB026_SCENE" \
  --candidate "$OUT_CAND" \
  --object-body grasp_can \
  --out "$P3_JSON" \
  --best-plan-out "$PLAN_JSON" \
  --top-per-target 4 \
  --max-combos 128 \
  --path-samples 18 \
  --min-hand-support-clearance 0.0 \
  --min-fr3-object-clearance 0.0 \
  --max-grasp-hand-object-distance 0.060 \
  --min-joint-margin 0.0 \
  2>&1 | tee "$OUTROOT/terminal_p3.txt"

python3 - <<'PY'
from pathlib import Path
import json

PROJECT = Path.home() / "Projects/o7_mujoco_sim"
outroot = PROJECT / "diagnostics/current_v412/bb026_transfer_sodacan165_prior_debug"
p3 = outroot / "p3.json"
summary = outroot / "summary.txt"

d = json.loads(p3.read_text())
bp = d.get("best_pass")
ba = d.get("best_available") or {}
chosen = bp if bp is not None else ba

lines = []
lines.append("========== BB026 TRANSFER SODACAN165 PRIOR SUMMARY ==========")
lines.append(f"num_combos: {d.get('num_combos')}")
lines.append(f"num_pass  : {d.get('num_pass')}")
lines.append(f"best_pass_exists: {bp is not None}")
for k in [
    "precheck_status",
    "score",
    "min_path_hand_support_clearance",
    "min_path_fr3_object_clearance",
    "static_grasp_closed_hand_object_distance",
    "static_grasp_closed_hand_support_clearance",
    "combo_min_joint_margin",
    "hard_reasons",
]:
    lines.append(f"{k}: {chosen.get(k)}")
lines.append("")
if d.get("num_pass", 0) > 0:
    lines.append("viewer_script: scripts/05_execution_runner/run_bb026_transfer_sodacan165_prior_viewer_debug.sh")
    lines.append("run viewer:")
    lines.append("./scripts/05_execution_runner/run_bb026_transfer_sodacan165_prior_viewer_debug.sh")
else:
    lines.append("NO PASS: do not viewer yet. Send this summary back.")
lines.append("============================================================")

summary.write_text("\n".join(lines) + "\n")
print("\n".join(lines))
PY

cat <<'RUN' > scripts/05_execution_runner/run_bb026_transfer_sodacan165_prior_viewer_debug.sh
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
RUN

chmod +x scripts/05_execution_runner/run_bb026_transfer_sodacan165_prior_viewer_debug.sh

echo
echo "========== DONE =========="
echo "summary:"
echo "cat $OUTROOT/summary.txt"
