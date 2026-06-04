#!/usr/bin/env bash
set -euo pipefail
shopt -s nullglob

MODE="${1:-dryrun}"

if [[ "$MODE" != "dryrun" && "$MODE" != "apply" ]]; then
  echo "Usage: bash scripts/00_maintenance/organize_project_for_v412_debug.sh [dryrun|apply]"
  exit 1
fi

ROOT="$(pwd -P)"
STAMP="$(date +%Y%m%d_%H%M%S)"
ARCHIVE_DIR="_archive_after_v410_${STAMP}"

if [[ ! -d "$ROOT/scripts" || ! -d "$ROOT/models" || ! -d "$ROOT/diagnostics" ]]; then
  echo "[ERROR] 请在 ~/Projects/o7_mujoco_sim 项目根目录执行"
  exit 1
fi

echo "========== PROJECT ORGANIZE FOR V4.12 =========="
echo "ROOT        : $ROOT"
echo "MODE        : $MODE"
echo "ARCHIVE_DIR : $ARCHIVE_DIR"
echo "================================================"

do_mkdir() {
  local d="$1"
  if [[ "$MODE" == "dryrun" ]]; then
    echo "[DRYRUN] mkdir -p $d"
  else
    mkdir -p "$d"
  fi
}

move_keep_link() {
  local src="$1"
  local dst_dir="$2"

  if [[ ! -e "$src" && ! -L "$src" ]]; then
    return 0
  fi

  local base
  base="$(basename "$src")"
  local dst="${dst_dir}/${base}"

  if [[ "$src" == "$dst" ]]; then
    return 0
  fi

  do_mkdir "$dst_dir"

  if [[ "$MODE" == "dryrun" ]]; then
    echo "[DRYRUN] mv $src -> $dst"
    echo "[DRYRUN] ln -s $ROOT/$dst $src"
    return 0
  fi

  if [[ -L "$src" ]]; then
    echo "[SKIP] $src already symlink"
    return 0
  fi

  if [[ -e "$dst" ]]; then
    echo "[WARN] destination exists, skip moving: $dst"
    return 0
  fi

  mv "$src" "$dst"
  ln -s "$ROOT/$dst" "$src"
  echo "[MOVED+LINK] $src -> $dst"
}

archive_path() {
  local src="$1"

  if [[ ! -e "$src" && ! -L "$src" ]]; then
    return 0
  fi

  local dst="${ARCHIVE_DIR}/${src}"
  local dst_parent
  dst_parent="$(dirname "$dst")"

  if [[ "$MODE" == "dryrun" ]]; then
    echo "[DRYRUN] archive $src -> $dst"
    return 0
  fi

  mkdir -p "$dst_parent"
  mv "$src" "$dst"
  echo "[ARCHIVED] $src -> $dst"
}

# 1. 创建目录
for d in \
  docs/00_project_record \
  docs/10_method_notes \
  docs/20_run_guides \
  scripts/00_maintenance \
  scripts/01_scene_modeling \
  scripts/02_dataset_candidate \
  scripts/03_candidate_scoring \
  scripts/04_planning_legacy_v4_v11 \
  scripts/05_execution_runner \
  scripts/06_diagnostics_viewer \
  scripts/07_precheck_ik_collision \
  scripts/90_legacy_misc \
  diagnostics/current_v412 \
  diagnostics/archive_readme \
  records/README_keep \
  records/candidates \
  records/precheck \
  records/runs
do
  do_mkdir "$d"
done

# 2. 移动 docs
move_keep_link "docs/07_grasp_demo_stage_record.md" "docs/00_project_record"

# 3. 场景 / 模型 / patch 类脚本
for f in \
  scripts/make_*_debug.py \
  scripts/make_*_v*.py \
  scripts/patch_bottle_scene_*_debug.py \
  scripts/patch_tabletop_*_debug.py \
  scripts/patch_scene_support_collision_debug.py \
  scripts/patch_bottle_scene_free_dataset_debug.py \
  scripts/patch_bottle_scene_upright_tabletop_debug.py \
  scripts/patch_bottle_scene_upright_tabletop_scaled_debug.py \
  scripts/patch_candidate_radial_outward_debug.py \
  scripts/patch_candidate_safe_wrist_orientation_debug.py \
  scripts/add_dataset_hand_base_site_debug.py \
  scripts/convert_fr3_o7_dae_to_obj.py
do
  move_keep_link "$f" "scripts/01_scene_modeling"
done

# 4. 数据集到 candidate
for f in \
  scripts/dataset_bottle_to_candidate_debug.py \
  scripts/dataset_to_candidate_debug.py \
  scripts/make_v4_7_can_scene_and_candidates_debug.py \
  scripts/make_v4_7b_can_tabletop_safe_candidates_debug.py \
  scripts/make_v4_4_tabletop_side_body_topk_debug.py \
  scripts/make_v4_2_diverse_topk_debug.py
do
  move_keep_link "$f" "scripts/02_dataset_candidate"
done

# 5. 候选评分 / 筛选 / Top-K
for f in \
  scripts/apply_v4_3_grasp_type_consistency_debug.py \
  scripts/classify_tabletop_support_safety_debug.py \
  scripts/evaluate_tabletop_candidate_quality_debug.py \
  scripts/evaluate_tabletop_candidate_quality_v2_debug.py \
  scripts/evaluate_v4_5_site_servo_topk_debug.py \
  scripts/evaluate_v4_7_strict_collision_topk_debug.py \
  scripts/filter_bottle_dataset_tabletop_candidates_debug.py \
  scripts/generate_grasp_type_hypotheses_v4_debug.py \
  scripts/inspect_upright_tabletop_sample_contacts_debug.py \
  scripts/select_tabletop_grasp_candidates_v2_debug.py \
  scripts/select_v4_online_topk_o7_dexrep_lite_debug.py
do
  move_keep_link "$f" "scripts/03_candidate_scoring"
done

# 6. V4.0~V4.11 旧规划/搜索/优化
for f in \
  scripts/generate_support_aware_prior_plans_debug.py \
  scripts/generate_grasp_execution_plan_debug.py \
  scripts/optimize_prior_guided_grasp_debug.py \
  scripts/optimize_source_truth_prior_grasp_debug.py \
  scripts/optimize_source_truth_prior_grasp_v2_debug.py \
  scripts/optimize_source_truth_prior_grasp_v3_debug.py \
  scripts/refine_v3_from_contact_rich_rollout_debug.py \
  scripts/refine_v4_top_pose_variants_geometry_debug.py \
  scripts/run_v4_*_debug.py \
  scripts/run_v4_branch_search_debug.py \
  scripts/run_v4_online_topk_short_rollout_debug.py \
  scripts/run_v4_10_support_clearance_search_debug.py \
  scripts/patch_v4_11_thumb_close_candidate_debug.py
do
  move_keep_link "$f" "scripts/04_planning_legacy_v4_v11"
done

# 7. 执行 runner
for f in \
  scripts/run_fr3_o7_candidate_grasp_site_servo_debug.py \
  scripts/run_fr3_o7_candidate_grasp_site_servo_fixed_object_debug.py \
  scripts/run_fr3_o7_candidate_grasp_v16.py \
  scripts/run_fr3_o7_candidate_grasp_with_plan_debug.py \
  scripts/run_fr3_o7_lift_demo_v8.py \
  scripts/run_fr3_o7_object_ik_grasp_v12.py \
  scripts/run_fr3_o7_object_reach_tune_v13.py \
  scripts/run_fr3_o7_prior_contact_seek_debug.py \
  scripts/run_source_truth_prior_optimized_result_debug.py
do
  move_keep_link "$f" "scripts/05_execution_runner"
done

# 8. 诊断 / 可视化
for f in \
  scripts/diagnose_v4_9_frame_ik_contact_debug.py \
  scripts/preview_*_debug.py \
  scripts/verify_fr3_o7_handbase_alignment_debug.py \
  scripts/view_*_debug.py \
  scripts/view_*_pose_debug.py \
  scripts/view_*_candidate_*_debug.py \
  scripts/clean_o7_versions_safe.py
do
  move_keep_link "$f" "scripts/06_diagnostics_viewer"
done

# 9. 旧 diagnostics 归档，不删除
for p in \
  diagnostics/v4_5_site_servo_topk_eval_debug \
  diagnostics/v4_6_small_pose_perturb_debug \
  diagnostics/v4_6_medium_pose_perturb_debug \
  diagnostics/v4_7_can52_site_servo_eval_debug \
  diagnostics/v4_7_can52_x90_site_servo_eval_debug \
  diagnostics/v4_7_can52_tabletop_safe_eval_debug \
  diagnostics/v4_7_can52_strict_collision_eval_debug \
  diagnostics/v4_8_can52_support_aware_pose_search_debug \
  diagnostics/v4_8b_can52_alignment_aware_pose_search_debug \
  diagnostics/v4_9b_can52_source_s030_runner_debug.txt \
  diagnostics/v4_9b_can52_source_s030_runner_debug.json \
  diagnostics/v4_9b_can52_source_s030_frame_ik_contact_diag_debug.txt \
  diagnostics/v4_9b_can52_source_s030_frame_ik_contact_diag_debug.json \
  diagnostics/v4_10_can52_s030_support_clearance_search_debug \
  diagnostics/v4_10_can52_s030_support_clearance_search_debug.txt \
  diagnostics/v4_10_can52_s030_support_clearance_search_debug.json \
  diagnostics/view_v4_10_can52_s030_best_debug.txt \
  diagnostics/view_v4_10_can52_s030_best_debug.json \
  diagnostics/view_v4_11_can52_s030_thumb_palm_axis_debug.txt \
  diagnostics/view_v4_11_can52_s030_thumb_palm_axis_debug.json \
  diagnostics/prior_contact_seek_debug \
  diagnostics/prior_contact_seek_safe_wrist_debug \
  diagnostics/radial_out_sample008_debug \
  diagnostics/support_safety_debug \
  diagnostics/with_plan_debug \
  diagnostics/with_plan_v2_debug \
  diagnostics/eval_quality_scale006_tscale10_debug \
  diagnostics/eval_quality_v2_scale006_tscale10_debug \
  diagnostics/dynamic_scale006_tscale10_batch_debug
do
  archive_path "$p"
done

# 10. 写 README
if [[ "$MODE" == "apply" ]]; then
cat > docs/PROJECT_STRUCTURE_V412.md <<'MD'
# O7 MuJoCo 项目结构说明（V4.12 起）

当前路线已经从“继续调单个抓取姿态”切换为：

1. 数据集先验只负责提供候选抓握方向、抓型、手型趋势。
2. IK 模块负责判断 FR3 是否能自然到达目标位姿。
3. 碰撞预检模块负责判断路径是否会碰支撑台、物体或机器人自身。
4. MuJoCo runner 只负责最终动态验证，不再承担 IK 和碰撞筛选职责。

## 目录

- `scripts/01_scene_modeling/`  
  生成或 patch MuJoCo XML、支撑台、物体、URDF/MJCF 转换等。

- `scripts/02_dataset_candidate/`  
  数据集 sample 转 candidate，生成候选 JSON。

- `scripts/03_candidate_scoring/`  
  抓型一致性、支撑安全、Top-K、O7-DexRep-lite 类评分。

- `scripts/04_planning_legacy_v4_v11/`  
  V4.0 到 V4.11 的旧位姿搜索、support-aware search、thumb patch 等实验脚本。保留用于追溯，不作为后续主线。

- `scripts/05_execution_runner/`  
  真正执行 grasp / close / hold / lift 的 MuJoCo runner。

- `scripts/06_diagnostics_viewer/`  
  frame、IK、contact、viewer 诊断脚本。

- `scripts/07_precheck_ik_collision/`  
  V4.12 新主线：Pinocchio IK、多 seed IK、路径碰撞预检、FCL/MuJoCo clearance 检查。

- `diagnostics/current_v412/`  
  V4.12 起的新日志和结果优先放这里。

- `_archive_after_v410_*/`  
  V4.10 以前的大量中间 diagnostics 归档目录。不是删除。
MD

cat > scripts/README.md <<'MD'
# scripts 目录说明

从 V4.12 开始，脚本不再全部堆在 `scripts/` 根目录。

旧路径保留为软链接，避免之前的命令立即失效；真正文件已按功能分类。

## 当前主线

后续新增脚本优先放入：

- `07_precheck_ik_collision/`  
  IK、碰撞预检、路径 clearance 检查。

- `05_execution_runner/`  
  动态执行 runner。

- `06_diagnostics_viewer/`  
  可视化与诊断。

## 旧实验

- `04_planning_legacy_v4_v11/`  
  V4.0~V4.11 的位姿搜索和调参脚本，只用于回溯，不再作为主线继续堆功能。
MD

cat > scripts/07_precheck_ik_collision/README.md <<'MD'
# V4.12 IK + Collision Precheck

本目录用于承接新的主线。

目标：

1. 使用 Pinocchio 或公司内部 IK 求解器生成多组 FR3 IK 解。
2. 对 `q_current -> q_pre -> q_grasp -> q_lift` 做路径离散采样。
3. 使用 FCL / hpp-fcl / MuJoCo geomDistance 做碰撞与 clearance 预检。
4. 只有通过预检的 candidate 才允许进入 MuJoCo 动态 runner。

预检输出字段建议：

- `ik_success`
- `pose_error`
- `joint_limit_margin`
- `min_hand_support_clearance`
- `min_fr3_object_clearance`
- `min_hand_object_distance_at_grasp`
- `first_collision_pair`
- `precheck_status`
MD

cat > diagnostics/README.md <<'MD'
# diagnostics 目录说明

V4.12 之前的中间日志很多已经归档到 `_archive_after_v410_*`。

从 V4.12 开始，新结果优先放入：

- `diagnostics/current_v412/`

命名建议：

- `v4_12a_pinocchio_ik_diag_debug.json`
- `v4_12b_path_collision_precheck_debug.json`
- `view_v4_12_pass_precheck_best_debug.json`
MD

cat > records/README.md <<'MD'
# records 目录说明

- `records/candidates/`  
  保存候选抓握 JSON。

- `records/precheck/`  
  保存 IK + 碰撞预检后的 candidate 结果。

- `records/runs/`  
  保存最终 runner 可复现实验记录。
MD
fi

echo
echo "========== DONE =========="
echo "MODE=$MODE"
if [[ "$MODE" == "dryrun" ]]; then
  echo "这只是预览，没有移动文件。确认后运行："
  echo "bash scripts/00_maintenance/organize_project_for_v412_debug.sh apply"
else
  echo "整理完成。旧 diagnostics 已归档到：$ARCHIVE_DIR"
  echo "旧脚本路径已尽量保留软链接，避免旧命令立刻失效。"
fi
