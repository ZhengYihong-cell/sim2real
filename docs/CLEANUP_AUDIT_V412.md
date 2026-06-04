# O7 MuJoCo Cleanup Audit V4.12

审计日期：2026-06-02  
范围：`scripts/`, `diagnostics/`, `records/`, `models/fr3_o7/`, `docs/`

本文件是第一阶段只读结构审计。未删除、未移动、未修改任何既有文件；仅新增本审计文档。

## 审计方法

- 用 `find` / `rg --files` 扫描目标目录。
- 用 `rg` 搜索每个 Python 脚本的文件名和模块名，检查是否被其他脚本 import、`importlib.util.spec_from_file_location` 动态加载、或通过 `subprocess` / `run_mujoco_clean.sh` 调用。
- 用 `rg` 搜索 `models/fr3_o7/*.xml` 文件名，检查 XML 是否仍被 diagnostics、records、docs 或 scripts 引用。
- 未扫描 `dataset/`，并且本轮不建议删除 `dataset/` 下任何内容。

## 扫描概览

- `scripts/`：70 个 Python 源文件，不含 `__pycache__`。
- `diagnostics/`：约 521 个文件，主要是 V4.0~V4.11 的 JSON/TXT 结果和 runner logs。
- `records/`：约 477 个文件，主要是 candidates、plans、stable templates。
- `models/fr3_o7/`：约 56 个文件，其中 XML 场景、URDF、OBJ/MTL 转换网格并存。
- `docs/`：2 个既有 markdown 文件：`PROJECT_STRUCTURE_V412.md` 与 `o7_grasp_demo_stage_record.md`。

## 重要保护结论

- 不删除 `dataset/`。
- 不删除 `run_mujoco_clean.sh`。
- 不删除当前 runner：`scripts/05_execution_runner/run_fr3_o7_candidate_grasp_site_servo_debug.py`，以及兼容软链接 `scripts/run_fr3_o7_candidate_grasp_site_servo_debug.py`。
- 不删除 `scripts/run_fr3_o7_candidate_grasp_site_debug.py`。虽然它在 `scripts/` 根目录且不是软链接，但当前 runner 动态加载它。
- 不删除 `scripts/05_execution_runner/run_fr3_o7_object_ik_grasp_v12.py`。它被 `run_fr3_o7_candidate_grasp_site_debug.py` 和 V16 runner 动态加载。
- 不删除 `scripts/06_diagnostics_viewer/diagnose_v4_9_frame_ik_contact_debug.py`，以及兼容软链接。
- 不删除 `scripts/07_precheck_ik_collision/`。它是 V4.12 IK / collision precheck 主线目录，目前只有 README。
- 不删除 `models/fr3_o7/` 下当前高引用 XML，尤其：
  - `fr3_o7_bottle_scene_handbase_upright_tabletop_scale006_debug.xml`
  - `fr3_o7_can52_upright_tabletop_v47b_debug.xml`
  - `fr3_o7_can52_upright_tabletop_x90_debug.xml`
  - `fr3_o7_can52_upright_tabletop_debug.xml`
  - `fr3_o7_actuated_scene_v1f_stable_hand.xml`
  - `fr3_o7_actuated_scene_v13_cylinder.xml`
- 不删除任何用途不确定的文件；对旧结果只建议归档，不建议直接删除。

## Python 脚本功能分类

### 场景/模型生成

主要职责：生成或 patch MuJoCo XML、URDF/MJCF、支撑台、物体、候选姿态小修正。

- `scripts/01_scene_modeling/add_dataset_hand_base_site_debug.py`
- `scripts/01_scene_modeling/convert_fr3_o7_dae_to_obj.py`
- `scripts/01_scene_modeling/make_bottle_scene_debug.py`
- `scripts/01_scene_modeling/make_fr3_o7_actuated_scene_v1d_urdf_coupled.py`
- `scripts/01_scene_modeling/make_fr3_o7_actuated_scene_v1e_contact_raised.py`
- `scripts/01_scene_modeling/make_fr3_o7_actuated_scene_v1f_stable_hand.py`
- `scripts/01_scene_modeling/make_fr3_o7_cylinder_scene_v13.py`
- `scripts/01_scene_modeling/make_v4_2_diverse_topk_debug.py`
- `scripts/01_scene_modeling/make_v4_4_tabletop_side_body_topk_debug.py`
- `scripts/01_scene_modeling/make_v4_7_can_scene_and_candidates_debug.py`
- `scripts/01_scene_modeling/make_v4_7b_can_tabletop_safe_candidates_debug.py`
- `scripts/01_scene_modeling/patch_bottle_scene_free_dataset_debug.py`
- `scripts/01_scene_modeling/patch_bottle_scene_upright_tabletop_debug.py`
- `scripts/01_scene_modeling/patch_bottle_scene_upright_tabletop_scaled_debug.py`
- `scripts/01_scene_modeling/patch_candidate_radial_outward_debug.py`
- `scripts/01_scene_modeling/patch_candidate_safe_wrist_orientation_debug.py`
- `scripts/01_scene_modeling/patch_scene_support_collision_debug.py`
- `scripts/01_scene_modeling/patch_tabletop_support_small_solid_debug.py`
- `scripts/01_scene_modeling/patch_tabletop_support_solid_debug.py`
- `scripts/01_scene_modeling/patch_tabletop_support_zshift_debug.py`

引用情况：多数没有被其他 Python 直接 import；主要作为一次性生成脚本使用。`convert_fr3_o7_dae_to_obj.py` 自身使用 `subprocess` 调外部转换工具。整理脚本 `scripts/00_maintenance/organize_project_for_v412_debug.sh` 引用了这些文件名用于分层迁移，不是运行期依赖。

### 数据集 sample 转 candidate

- `scripts/02_dataset_candidate/dataset_to_candidate_debug.py`
- `scripts/dataset_bottle_to_candidate_site_debug.py`
- `scripts/01_scene_modeling/make_v4_7_can_scene_and_candidates_debug.py`
- `scripts/01_scene_modeling/make_v4_7b_can_tabletop_safe_candidates_debug.py`

引用情况：
- `dataset_to_candidate_debug.py` 在 `records/candidates/from_dataset_debug_candidate.json` 和脚本输出 note 中留下记录，未发现被其他 Python import。
- `dataset_bottle_to_candidate_site_debug.py` 在 `records/candidates/bottle_dataset_site_sample7_debug.json` 中留下 `type` 记录，未发现被其他 Python import。
- `make_v4_7*_scene_and_candidates*` 同时生成场景和候选，属于旧 V4.7 实验链，建议归档但保留。

### candidate 筛选/评分/top-k

- `scripts/03_candidate_scoring/apply_v4_3_grasp_type_consistency_debug.py`
- `scripts/03_candidate_scoring/classify_tabletop_support_safety_debug.py`
- `scripts/03_candidate_scoring/evaluate_tabletop_candidate_quality_debug.py`
- `scripts/03_candidate_scoring/evaluate_tabletop_candidate_quality_v2_debug.py`
- `scripts/03_candidate_scoring/evaluate_v4_5_site_servo_topk_debug.py`
- `scripts/03_candidate_scoring/filter_bottle_dataset_tabletop_candidates_debug.py`
- `scripts/03_candidate_scoring/generate_grasp_type_hypotheses_v4_debug.py`
- `scripts/03_candidate_scoring/inspect_upright_tabletop_sample_contacts_debug.py`
- `scripts/03_candidate_scoring/select_tabletop_grasp_candidates_v2_debug.py`
- `scripts/03_candidate_scoring/select_v4_online_topk_o7_dexrep_lite_debug.py`
- `scripts/evaluate_v4_7c_strict_collision_topk_debug.py`
- `scripts/refine_v4_3_from_contact_rich_rollout_debug.py`
- `scripts/refine_v4_topk_pose_variants_geometry_debug.py`

引用情况：
- `classify_tabletop_support_safety_debug.py`, `evaluate_tabletop_candidate_quality*.py`, `filter_bottle_dataset_tabletop_candidates_debug.py`, `inspect_upright_tabletop_sample_contacts_debug.py` 都通过 `importlib.util.spec_from_file_location` 动态加载 `scripts/run_fr3_o7_candidate_grasp_site_debug.py`。
- `evaluate_v4_5_site_servo_topk_debug.py` 和 `evaluate_v4_7c_strict_collision_topk_debug.py` 使用 `subprocess` + `run_mujoco_clean.sh` 调 runner。
- `filter_bottle_dataset_tabletop_candidates_debug.py` 在大量 `records/candidates/upright_tabletop*` JSON 中留下 `type` 记录。
- `select_tabletop_grasp_candidates_v2_debug.py` 在 `diagnostics/select_tabletop_candidates_v2_scale006_debug.json` 中留下 format 记录。
- `refine_v4_*` 位于 `scripts/` 根目录且不是软链接，未发现被其他 Python 直接调用；更像 V4.2/V4.3 旧 refinement 实验脚本，建议归档保留。

### 动态 runner

- `scripts/05_execution_runner/run_fr3_o7_candidate_grasp_site_servo_debug.py`
- `scripts/05_execution_runner/run_fr3_o7_candidate_grasp_site_servo_fixed_object_debug.py`
- `scripts/05_execution_runner/run_fr3_o7_candidate_grasp_v16.py`
- `scripts/05_execution_runner/run_fr3_o7_candidate_grasp_with_plan_debug.py`
- `scripts/05_execution_runner/run_fr3_o7_lift_demo_v8.py`
- `scripts/05_execution_runner/run_fr3_o7_object_ik_grasp_v12.py`
- `scripts/05_execution_runner/run_fr3_o7_prior_contact_seek_debug.py`
- `scripts/05_execution_runner/run_source_truth_prior_optimized_result_debug.py`
- `scripts/run_fr3_o7_candidate_grasp_site_debug.py`
- `scripts/run_fr3_o7_object_keyboard_tuner_v13.py`
- `scripts/run_support_aware_prior_plan_debug.py`

引用情况：
- 当前主 runner `run_fr3_o7_candidate_grasp_site_servo_debug.py` 动态加载 `scripts/run_fr3_o7_candidate_grasp_site_debug.py`。
- `run_fr3_o7_candidate_grasp_site_debug.py` 动态加载 `scripts/run_fr3_o7_object_ik_grasp_v12.py`。
- `run_fr3_o7_candidate_grasp_site_servo_fixed_object_debug.py`, `run_fr3_o7_candidate_grasp_with_plan_debug.py`, `run_fr3_o7_prior_contact_seek_debug.py`, `run_support_aware_prior_plan_debug.py` 都动态加载当前 site-servo runner。
- `run_fr3_o7_candidate_grasp_v16.py` 动态加载 V12 runner。
- `run_source_truth_prior_optimized_result_debug.py` 动态加载 `run_fr3_o7_candidate_grasp_site_debug.py`。
- 大量 diagnostics 明确记录 `run_mujoco_clean.sh scripts/run_fr3_o7_candidate_grasp_site_servo_debug.py ...` 命令，当前 runner 必须保留。
- `run_fr3_o7_object_keyboard_tuner_v13.py` 未发现当前代码直接调用；作为旧交互调参工具建议归档保留。

### viewer/diagnostics

- `scripts/06_diagnostics_viewer/diagnose_v4_9_frame_ik_contact_debug.py`
- `scripts/06_diagnostics_viewer/preview_upright_tabletop_sample_static_debug.py`
- `scripts/06_diagnostics_viewer/preview_upright_tabletop_scene_debug.py`
- `scripts/06_diagnostics_viewer/verify_fr3_o7_handbase_alignment_debug.py`
- `scripts/06_diagnostics_viewer/view_o7_axis_convention_sweep_debug.py`
- `scripts/06_diagnostics_viewer/view_o7_close_at_optimized_pose_debug.py`
- `scripts/06_diagnostics_viewer/view_source_truth_candidate_pose_debug.py`
- `scripts/06_diagnostics_viewer/view_source_truth_prior_optimized_result_debug.py`
- `scripts/06_diagnostics_viewer/view_source_truth_prior_v2_result_debug.py`
- `scripts/06_diagnostics_viewer/view_v4_rollout_candidate_debug.py`
- `scripts/06_diagnostics_viewer/view_v4_topk_sequence_debug.py`

引用情况：
- 多数 viewer 动态加载 `scripts/run_fr3_o7_candidate_grasp_site_debug.py`。
- `view_source_truth_prior_v2_result_debug.py` 动态加载 `scripts/optimize_source_truth_prior_grasp_debug.py`。
- `view_v4_topk_sequence_debug.py` 通过 `subprocess` 调 `scripts/view_v4_rollout_candidate_debug.py`。
- `view_v4_rollout_candidate_debug.py` 使用 `run_mujoco_clean.sh` 调 runner。
- `verify_fr3_o7_handbase_alignment_debug.py` 在 diagnostics 中有结果记录。
- `diagnose_v4_9_frame_ik_contact_debug.py` 是明确保护文件，保留。

### V4.0~V4.11 旧实验脚本

- `scripts/04_planning_legacy_v4_v11/generate_grasp_execution_plan_debug.py`
- `scripts/04_planning_legacy_v4_v11/generate_support_aware_prior_plans_debug.py`
- `scripts/04_planning_legacy_v4_v11/optimize_prior_guided_grasp_debug.py`
- `scripts/04_planning_legacy_v4_v11/optimize_source_truth_prior_grasp_debug.py`
- `scripts/04_planning_legacy_v4_v11/optimize_source_truth_prior_grasp_v2_debug.py`
- `scripts/04_planning_legacy_v4_v11/optimize_source_truth_prior_grasp_v3_debug.py`
- `scripts/04_planning_legacy_v4_v11/patch_v4_11_thumb_close_candidate_debug.py`
- `scripts/04_planning_legacy_v4_v11/run_v4_10_support_clearance_search_debug.py`
- `scripts/04_planning_legacy_v4_v11/run_v4_6_small_pose_perturb_debug.py`
- `scripts/04_planning_legacy_v4_v11/run_v4_8_support_aware_pose_search_debug.py`
- `scripts/04_planning_legacy_v4_v11/run_v4_branch_search_debug.py`
- `scripts/04_planning_legacy_v4_v11/run_v4_online_topk_short_rollout_debug.py`
- `scripts/archive_after_v16_clean.py`

引用情况：
- `optimize_source_truth_prior_grasp_v2_debug.py` 动态加载 V1。
- `optimize_source_truth_prior_grasp_v3_debug.py` 动态加载 V2。
- `run_v4_branch_search_debug.py` 通过 `subprocess` + `run_mujoco_clean.sh` 调 V3 优化脚本。
- `run_v4_6_small_pose_perturb_debug.py`, `run_v4_8_support_aware_pose_search_debug.py`, `run_v4_10_support_clearance_search_debug.py` 都通过 `subprocess` + `run_mujoco_clean.sh` 调 runner。
- `run_v4_online_topk_short_rollout_debug.py` 动态加载 site-servo runner。
- `generate_grasp_execution_plan_debug.py` 在 `records/plans/upright_tabletop_scale006_debug/*.json` 中留下 planner 记录。
- `archive_after_v16_clean.py` 是旧清理脚本，含删除/归档意图；第一阶段不要执行，建议归档保留。

### V4.12 以后还可能保留的主线脚本/目录

- `scripts/07_precheck_ik_collision/README.md`：V4.12 新主线占位，后续 IK、collision precheck、clearance 检查应在此扩展。
- `scripts/05_execution_runner/run_fr3_o7_candidate_grasp_site_servo_debug.py`：当前动态验证 runner。
- `scripts/run_fr3_o7_candidate_grasp_site_debug.py`：当前 runner 的底层 site runner 依赖。
- `scripts/05_execution_runner/run_fr3_o7_object_ik_grasp_v12.py`：当前底层 IK/工具函数依赖。
- `scripts/06_diagnostics_viewer/diagnose_v4_9_frame_ik_contact_debug.py`：保留为 frame / IK / contact 诊断。
- `scripts/05_execution_runner/run_fr3_o7_candidate_grasp_site_servo_fixed_object_debug.py`：当前 runner 的固定物体调试变体。
- `scripts/05_execution_runner/run_fr3_o7_candidate_grasp_with_plan_debug.py`：plan 驱动执行入口，后续 precheck plan 可能复用。
- `scripts/05_execution_runner/run_fr3_o7_prior_contact_seek_debug.py` 与 `scripts/run_support_aware_prior_plan_debug.py`：仍可能作为 V4.12 前后过渡参考，但建议新功能逐步迁入 `07_precheck_ik_collision/`。

## 直接引用与调用关系

以下为 `rg` 检出的代码级依赖，不含仅在 JSON/TXT 里作为历史结果 format 出现的弱引用。

| 被引用脚本 | 引用方式 | 引用方 |
|---|---|---|
| `scripts/run_fr3_o7_candidate_grasp_site_debug.py` | `importlib.util.spec_from_file_location` | 当前 site-servo runner、source-truth runner、多个 scoring/viewer/legacy optimizer |
| `scripts/run_fr3_o7_object_ik_grasp_v12.py` | `importlib.util.spec_from_file_location` | `run_fr3_o7_candidate_grasp_site_debug.py`, `run_fr3_o7_candidate_grasp_v16.py` |
| `scripts/run_fr3_o7_candidate_grasp_site_servo_debug.py` | `importlib.util.spec_from_file_location` | fixed-object runner、with-plan runner、prior-contact-seek runner、support-aware runner、V4 online top-k runner |
| `scripts/run_fr3_o7_candidate_grasp_site_servo_debug.py` | `subprocess` / `run_mujoco_clean.sh` | V4.5 top-k eval、V4.6 perturb、V4.7c strict collision eval、V4.8/V4.10 support-aware search、diagnostics command logs |
| `scripts/optimize_source_truth_prior_grasp_debug.py` | `importlib.util.spec_from_file_location` | V2 optimizer、`view_source_truth_prior_v2_result_debug.py` |
| `scripts/optimize_source_truth_prior_grasp_v2_debug.py` | `importlib.util.spec_from_file_location` | V3 optimizer |
| `scripts/optimize_source_truth_prior_grasp_v3_debug.py` | `subprocess` / `run_mujoco_clean.sh` | `run_v4_branch_search_debug.py` |
| `scripts/view_v4_rollout_candidate_debug.py` | `subprocess` | `view_v4_topk_sequence_debug.py` |
| `run_mujoco_clean.sh` | command wrapper | 多个 eval/search/view 脚本与大量 diagnostics 记录 |

注意：当前大量代码仍硬编码 `PROJECT / "scripts/foo.py"`。根目录兼容软链接仍有价值，不能在未修路径前删除。

## models/fr3_o7 审计

### 必须保留的模型/资源

- `converted_meshes_obj/*.obj` 与 `*.mtl`：由 MuJoCo XML/URDF 资产链使用，不能删。
- `fr3_o7_raw.urdf`, `fr3_o7_abs.urdf`, `fr3_o7_mujoco_ready_obj.urdf`：转换来源和中间结果，保留。
- `fr3_o7_actuated_scene_v1f_stable_hand.xml`：稳定手模型，高引用。
- `fr3_o7_actuated_scene_v13_cylinder.xml`：V13/V16 圆柱实验仍有记录引用。
- `fr3_o7_bottle_scene_handbase_upright_tabletop_scale006_debug.xml`：最高引用当前瓶子 tabletop 模型。
- `fr3_o7_can52_upright_tabletop_debug.xml`, `fr3_o7_can52_upright_tabletop_x90_debug.xml`, `fr3_o7_can52_upright_tabletop_v47b_debug.xml`：V4.7~V4.11 can52 主实验模型，仍被大量 diagnostics 引用。
- `fr3_o7_bottle_scene_handbase_*` 其他 XML：虽引用数较少，但作为场景演化链保留，不能直接删除。

### 可疑中间模型

- `models/fr3_o7/__v46_tmp_trial000_*.xml` 到 `__v46_tmp_trial019_*.xml`
  - 判断：V4.6 perturb 临时 XML。
  - 引用：仍被 `diagnostics/v4_6_small_pose_perturb_debug.json`, `diagnostics/v4_6b_medium_pose_perturb_debug.json`, runner logs/results 引用。
  - 建议：不要直接删除；若要清理，先整体归档到模型 archive，并在 diagnostics 中保留路径映射。
  - 替代：基础模型 `fr3_o7_bottle_scene_handbase_upright_tabletop_scale006_debug.xml` + V4.6 perturb 脚本/参数可重建，但当前审计未验证完全可重现。

## diagnostics 审计

- `diagnostics/current_v412/`：V4.12 新结果目录，必须保留。
- `diagnostics/archive_readme/`：归档说明目录，保留。
- `diagnostics/v4_search_all_debug/`, `diagnostics/v4_search_dataset_first_debug/`：V4 搜索批量结果，建议归档保留。
- `diagnostics/v4_6b_medium_pose_perturb_debug/`：包含 runner logs/results/tmp_candidates，与 `models/fr3_o7/__v46_tmp_trial*.xml` 对应，建议归档保留。
- `diagnostics/v4_7a_can52_x90_site_servo_eval_debug/`, `diagnostics/v4_7b_can52_tabletop_safe_eval_debug/`, `diagnostics/v4_7c_can52_strict_collision_eval_debug/`：V4.7 can52 评测结果，建议归档保留。
- 根目录大量 `*.json` / `*.txt`：多数是旧实验结果或环境/NVIDIA 修复记录。不能作为主线依赖，但可作为实验追溯材料整体归档。

## records 审计

- `records/candidates/`：保存候选 JSON，不能直接删除。
- `records/precheck/`：V4.12 precheck 输出目录，必须保留。
- `records/runs/`：V4.12 runner 复现实验目录，必须保留。
- `records/plans/`：旧 plan 输出，建议归档保留。
- `records/stable_*`, `records/fr3_o7_grasp_template*.json`, `records/o7_mount_transform_debug.json`：稳定候选和模板，保留。
- `records/candidates/v4_2_geometry_refined_bottle_debug/` 与 `v4_3_contact_lift_refined_bottle_debug/`：数量较大，是旧 top-k/refinement 产物，建议归档保留。
- `records/candidates/v4_7*_can52*`：与 can52 XML 和 diagnostics 对应，保留或归档保留。

## 清单

### KEEP_MAINLINE

必须保留，不能删：

- `run_mujoco_clean.sh`
- `scripts/README.md`
- `scripts/00_maintenance/organize_project_for_v412_debug.sh`
- `scripts/07_precheck_ik_collision/`
- `scripts/05_execution_runner/run_fr3_o7_candidate_grasp_site_servo_debug.py`
- `scripts/run_fr3_o7_candidate_grasp_site_servo_debug.py`，兼容软链接
- `scripts/run_fr3_o7_candidate_grasp_site_debug.py`
- `scripts/05_execution_runner/run_fr3_o7_object_ik_grasp_v12.py`
- `scripts/run_fr3_o7_object_ik_grasp_v12.py`，兼容软链接
- `scripts/05_execution_runner/run_fr3_o7_candidate_grasp_site_servo_fixed_object_debug.py`
- `scripts/05_execution_runner/run_fr3_o7_candidate_grasp_with_plan_debug.py`
- `scripts/05_execution_runner/run_fr3_o7_prior_contact_seek_debug.py`
- `scripts/run_support_aware_prior_plan_debug.py`
- `scripts/06_diagnostics_viewer/diagnose_v4_9_frame_ik_contact_debug.py`
- `scripts/06_diagnostics_viewer/verify_fr3_o7_handbase_alignment_debug.py`
- `scripts/06_diagnostics_viewer/preview_upright_tabletop_sample_static_debug.py`
- `scripts/06_diagnostics_viewer/preview_upright_tabletop_scene_debug.py`
- `scripts/03_candidate_scoring/filter_bottle_dataset_tabletop_candidates_debug.py`
- `scripts/03_candidate_scoring/select_tabletop_grasp_candidates_v2_debug.py`
- `scripts/03_candidate_scoring/select_v4_online_topk_o7_dexrep_lite_debug.py`
- `scripts/02_dataset_candidate/dataset_to_candidate_debug.py`
- `scripts/dataset_bottle_to_candidate_site_debug.py`
- `scripts/01_scene_modeling/add_dataset_hand_base_site_debug.py`
- `scripts/01_scene_modeling/patch_bottle_scene_upright_tabletop_scaled_debug.py`
- `scripts/01_scene_modeling/patch_scene_support_collision_debug.py`
- `models/fr3_o7/converted_meshes_obj/`
- `models/fr3_o7/fr3_o7_raw.urdf`
- `models/fr3_o7/fr3_o7_abs.urdf`
- `models/fr3_o7/fr3_o7_mujoco_ready_obj.urdf`
- `models/fr3_o7/fr3_o7_actuated_scene_v1f_stable_hand.xml`
- `models/fr3_o7/fr3_o7_actuated_scene_v13_cylinder.xml`
- `models/fr3_o7/fr3_o7_bottle_scene_handbase_upright_tabletop_scale006_debug.xml`
- `models/fr3_o7/fr3_o7_can52_upright_tabletop_debug.xml`
- `models/fr3_o7/fr3_o7_can52_upright_tabletop_x90_debug.xml`
- `models/fr3_o7/fr3_o7_can52_upright_tabletop_v47b_debug.xml`
- `diagnostics/current_v412/`
- `records/precheck/`
- `records/runs/`
- `records/candidates/`
- `records/stable_fr3_o7_grasp_candidate_v1.json`
- `records/stable_fr3_o7_cylinder_grasp_candidate_v1.json`
- `records/stable_real_contact_grasp_v8_18b_valid_mu1p2_box6cm.json`
- `docs/PROJECT_STRUCTURE_V412.md`
- `docs/o7_grasp_demo_stage_record.md`

### ARCHIVE_LEGACY

旧 demo / 旧实验，可以归档但暂不删除：

- `scripts/04_planning_legacy_v4_v11/` 全目录。
- `scripts/05_execution_runner/run_fr3_o7_lift_demo_v8.py`
- `scripts/05_execution_runner/run_fr3_o7_candidate_grasp_v16.py`
- `scripts/run_fr3_o7_object_keyboard_tuner_v13.py`
- `scripts/archive_after_v16_clean.py`
- `scripts/refine_v4_3_from_contact_rich_rollout_debug.py`
- `scripts/refine_v4_topk_pose_variants_geometry_debug.py`
- `scripts/evaluate_v4_7c_strict_collision_topk_debug.py`
- `scripts/01_scene_modeling/make_v4_2_diverse_topk_debug.py`
- `scripts/01_scene_modeling/make_v4_4_tabletop_side_body_topk_debug.py`
- `scripts/01_scene_modeling/make_v4_7_can_scene_and_candidates_debug.py`
- `scripts/01_scene_modeling/make_v4_7b_can_tabletop_safe_candidates_debug.py`
- `scripts/03_candidate_scoring/apply_v4_3_grasp_type_consistency_debug.py`
- `scripts/03_candidate_scoring/evaluate_v4_5_site_servo_topk_debug.py`
- `diagnostics/v4_search_all_debug/`
- `diagnostics/v4_search_dataset_first_debug/`
- `diagnostics/v4_6b_medium_pose_perturb_debug/`
- `diagnostics/v4_7a_can52_x90_site_servo_eval_debug/`
- `diagnostics/v4_7b_can52_tabletop_safe_eval_debug/`
- `diagnostics/v4_7c_can52_strict_collision_eval_debug/`
- `records/plans/`
- `records/candidates/v4_grasp_type_hypotheses_debug/`
- `records/candidates/v4_2_geometry_refined_bottle_debug/`
- `records/candidates/v4_3_contact_lift_refined_bottle_debug/`
- `records/candidates/v4_7_can52_debug/`
- `records/candidates/v4_7a_can52_x90_debug/`
- `records/candidates/v4_7b_can52_tabletop_safe_debug/`

归档建议：移动到 `_archive_after_v412_YYYYMMDD/` 或保持现状，且保留相对路径。不要只移动 JSON/TXT 而不移动它们引用的 tmp XML，否则旧实验不可复现。

### DELETE_CANDIDATE

第一阶段不执行删除。以下仅是候选，必须二次确认。

| 路径/模式 | 删除理由 | 替代/恢复方式 | 风险 |
|---|---|---|---|
| `scripts/__pycache__/` | Python 运行缓存，不是源文件 | 重新运行脚本会自动生成 | 低 |
| `models/fr3_o7/__v46_tmp_trial*.xml` | V4.6 perturb 中间 XML，命名含 `tmp`，不是当前主线模型 | 基础模型 `fr3_o7_bottle_scene_handbase_upright_tabletop_scale006_debug.xml` + V4.6 perturb 参数可能重建 | 中；旧 diagnostics 明确引用，建议先归档 |
| `diagnostics/v4_6b_medium_pose_perturb_debug/tmp_candidates/` | V4.6 runner 临时候选副本 | 对应正式候选在 `records/candidates/v4_2_geometry_refined_bottle_debug/` 或 diagnostics 汇总中可追溯 | 中；需逐项核对 candidate stem |
| `diagnostics/*nvidia*debug.txt`, `diagnostics/*reboot*debug.txt`, `diagnostics/*glx*debug.txt` | 环境修复记录，不属于抓取主线 | 无需替代；可移动到环境日志归档 | 低到中；保留一份归档更稳 |
| 根目录旧 diagnostics 单文件 `v4_*.json` / `v4_*.txt` | 大量旧实验结果，主线不再直接读取 | 对应实验脚本与 records 可重新生成部分结果 | 中；建议整体归档，不建议直接删除 |

不建议列为删除候选：

- 根目录 `scripts/*.py` 软链接：当前许多脚本硬编码 `PROJECT / "scripts/foo.py"`，删除会破坏兼容。
- `records/candidates/`：候选数据仍是 runner/precheck 输入。
- `models/fr3_o7/fr3_o7_*debug.xml`：即使少引用，也可能是场景演化链。
- `diagnostics/current_v412/`, `records/precheck/`, `records/runs/`：V4.12 新主线目录。

## 最后建议的目录结构

只建议，不执行：

```text
docs/
  PROJECT_STRUCTURE_V412.md
  CLEANUP_AUDIT_V412.md
  00_project_record/
    o7_grasp_demo_stage_record.md

scripts/
  README.md
  00_maintenance/
    organize_project_for_v412_debug.sh
  01_scene_modeling/
  02_dataset_candidate/
  03_candidate_scoring/
  04_planning_legacy_v4_v11/
  05_execution_runner/
  06_diagnostics_viewer/
  07_precheck_ik_collision/
    README.md
    ik_solvers/
    collision_precheck/
    clearance_eval/
  90_legacy_misc/
    archive_after_v16_clean.py
    run_fr3_o7_object_keyboard_tuner_v13.py
    refine_v4_*.py
  *.py -> compatibility symlinks only

models/fr3_o7/
  converted_meshes_obj/
  fr3_o7_*.urdf
  fr3_o7_actuated_scene_*.xml
  fr3_o7_bottle_scene_*.xml
  fr3_o7_can52_*.xml
  archive_tmp_v46/
    __v46_tmp_trial*.xml

records/
  README.md
  candidates/
  precheck/
  runs/
  plans/
  archive_legacy_v4/

diagnostics/
  README.md
  current_v412/
  archive_legacy_v4/
  archive_env_logs/
```

## 下一步建议

1. 先修正仍硬编码 `PROJECT / "scripts/foo.py"` 的动态加载路径，或明确保留根目录兼容软链接。
2. 在 `scripts/07_precheck_ik_collision/` 新增真实 IK / collision precheck 脚本后，再重新跑一次引用审计。
3. 只做归档迁移，不做删除；归档后用当前 runner 和至少一个 can52 / bottle candidate 做 smoke test。
