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
