# final_sodacan165_sample014_p4u6_ready_gated_snap_demo

这是 sem-SodaCan-16526 的第一轮泛化成功 demo。

## 流程：
dataset prior sample014 -> candidate/scene -> P2/P3 -> contact-free P3 goal -> sodacan-specific hand_config -> P4U6 path -> P4U1 ready-gated snap close -> fixed-grip lift

## 成功指标：
grip_ready=True
final_object_rise≈0.0656 m
final_groups={'thumb': 1, 'middle': 1, 'pinky': 1}
final_opposition_cos≈-0.755
三段 path 均 success=True

## 运行：
cd ~/Projects/o7_mujoco_sim
./legacy_final_demos/final_sodacan165_sample014_p4u6_ready_gated_snap_demo/run_demo_viewer.sh

## 无界面快速回归：
cd ~/Projects/o7_mujoco_sim
./legacy_final_demos/final_sodacan165_sample014_p4u6_ready_gated_snap_demo/run_demo_headless.sh

## 注意：
不要用本 demo 覆盖 can52 legacy demo。
不要把 sample014 参数当成所有物体的固定参数。
后续新物体仍然要走 Top-K -> P2/P3 -> object-specific hand_config -> P4U6 -> P4U1。



## 固化文件说明
inputs/scene.xml
    固化后的 SodaCan MuJoCo scene，object mesh 已 patch 到 demo 内部 object_mesh。

inputs/candidate.json
    sample014 数据集先验转换得到的 candidate。

inputs/p3_contactfree_goal.json
    从 P3 PASS 组合中筛出的 contact-free q_grasp 版本，避免 P4U6 pre_to_grasp 目标点已有接触。

inputs/best_config.json
    使用 sample014 candidate 自己的 O7 active ctrl 生成的 sodacan 专属 hand_config。

inputs/object_mesh/decomposed.obj
    SodaCan object mesh。

records/result_success.json
    成功 viewer 运行结果。

records/path_plan_success.json
    成功 viewer 对应路径规划结果。

records/terminal_success.txt
    成功 viewer 对应完整终端日志。

records/sweep_summary.txt / sweep_summary.json
    close sweep 结果。

scripts_snapshot/
    成功时相关 runner 源码快照，仅用于记录，不作为默认运行入口。

## 关键参数
finger_close_scale=1.12
micro_squeeze_fraction=0.00
max_grip_disp=0.018
grip_ready_stable_steps=5
lift_z=0.090
lift_duration=3.0
final_hold_duration=1.2