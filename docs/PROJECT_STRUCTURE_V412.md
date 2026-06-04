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
