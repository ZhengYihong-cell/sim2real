# O7 抓取仿真阶段性操作记录

## 0. 当前阶段结论

当前已经完成 O7 手在 MuJoCo 中的真实接触参数验证，并得到一个可复现的有效抓握候选姿态。

有效抓握文件：

```bash
records/stable_real_contact_grasp_v8_18b_valid_mu1p2_box6cm.json

核心模型文件：
models/o7_grasp_scene_v8_16_real_contact_mu1p2.xml
models/o7_grasp_scene_v8_18b_real_contact_tune_visible.xml

核心结论：
1. 当前抓握不是 V8.13 的极限参数假稳定；
2. 当前抓握不是几何死卡；
3. 当前抓握依赖摩擦；
4. 临界摩擦系数大约在 0.035 ~ 0.04 之间；
5. mu=1.2 下 30 秒保持不掉落，但存在缓慢下滑；
6. 当前手-only 最小 demo 已经能完成接近、闭合、抬升、保持、下放、释放流程，但抓握姿态仍受固定接近方向影响。

快速验证：
./run_mujoco_clean.sh scripts/validate_saved_grasp_mu_sweep.py \
  --record records/stable_real_contact_grasp_v8_18b_valid_mu1p2_box6cm.json \
  --models models/o7_grasp_scene_v8_16_real_contact_mu1p2.xml \
  --duration 3 \
  --log-dt 1.0 \
  --out diagnostics/quick_check_after_cleanup.json

1.MoveIt 控制流程(Franka 机械臂 + O7 手控制流程)
#检查URDF 
check_urdf ~/Projects/o7_minimal_sim/ros2_ws/src/fr3_o7_description/urdf/fr3_o7.urdf


终端 1：启动 robot_state_publisher 发布 FR3+O7 模型
cd ~/Projects/o7_minimal_sim/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run robot_state_publisher robot_state_publisher \
  ~/Projects/o7_minimal_sim/ros2_ws/src/fr3_o7_description/urdf/fr3_o7.urdf

终端 2：启动 joint_state_publisher_gui 手动拖关节
cd ~/Projects/o7_minimal_sim/ros2_ws
source /opt/ros/humble/setup.bash
source install/setup.bash
ros2 run joint_state_publisher_gui joint_state_publisher_gui

终端 3：启动 RViz / MoveIt MotionPlanning
source /opt/ros/humble/setup.bash
rviz2

RViz 中设置：
Fixed Frame: base 或 fr3_link0
Add -> RobotModel
Description Topic: /robot_description
Add -> TF


2.MuJoCo 中手动调整 O7 抓握姿态与保存流程
#激活环境
cd ~/Projects/o7_mujoco_sim
source ~/mujoco_env/bin/activate

#启动 O7 手动调姿工具
./run_mujoco_clean.sh scripts/run_o7_joint_tuner_v8_18_real_contact.py

保存 V  看接触点 C

3. MuJoCo 中复现 O7 手有效抓握的指令
# 打开 viewer 查看保存抓握
./run_mujoco_clean.sh scripts/inspect_saved_grasp_in_viewer.py \
  --model models/o7_grasp_scene_v8_18b_real_contact_tune_visible.xml \
  --record records/stable_real_contact_grasp_v8_18b_valid_mu1p2_box6cm.json \
  --zero-qvel

# viewer 操作：
SPACE：暂停 / 播放
P：打印当前接触点
R：重置到保存姿态
L：诊断用锁住手指 qpos，不作为真实结果

# 真实接触参数下 30 秒保持验证
./run_mujoco_clean.sh scripts/validate_saved_grasp_mu_sweep.py \
  --record records/stable_real_contact_grasp_v8_18b_valid_mu1p2_box6cm.json \
  --models models/o7_grasp_scene_v8_16_real_contact_mu1p2.xml \
  --duration 30 \
  --log-dt 1.0 \
  --out diagnostics/final_hold_30s_v8_18b_mu1p2.json

# 手-only 最小 demo 闭环复现
./run_mujoco_clean.sh scripts/run_o7_minimal_demo_from_valid_snapshot.py \
  --model models/o7_grasp_scene_v8_18b_real_contact_tune_visible.xml \
  --record records/stable_real_contact_grasp_v8_18b_valid_mu1p2_box6cm.json \
  --table-box-z 0.0315 \
  --pre-offset "0 0.12 0.04" \
  --lift-height 0.22 \
  --finger-boost 0.08

# demo流程
1. box 放在桌面；
2. 大拇指提前预成形；
3. 手接近 box；
4. 四指闭合；
5. 抬升；
6. 空中保持；
7. 下放；
8. 张开释放。

# MuJoco机械臂+O7抓握demo复现
./run_mujoco_clean.sh scripts/run_fr3_o7_lift_demo_v8.py \
  --record records/stable_fr3_o7_grasp_candidate_v1.json \
  --joint fr3_joint4 \
  --delta 0.15 \
  --settle 1.0 \
  --lift-duration 5.0 \
  --hold-duration 8.0

阶段 A：单模板 + MuJoCo 真值位姿
目标：验证随机 box 位姿下，IK 自动抓取是否可行。

阶段 B：多模板库
目标：保存多个不同方向的 T_box_target，例如前抓、侧抓、偏左、偏右、不同 yaw。

阶段 C：数据集/抓握生成模型
目标：对不同物体自动选择或生成 T_box_hand。

阶段 D：视觉接入
目标：实机获得 T_world_object。

阶段 E：实机执行
目标：MoveIt / 控制器执行 IK 轨迹 + O7 手指闭合。

# Mujoco 圆柱体
cd ~/Projects/o7_mujoco_sim
source ~/mujoco_env/bin/activate

./run_mujoco_clean.sh scripts/run_fr3_o7_cylinder_spawn_pregrasp_v15.py \
  --model models/fr3_o7/fr3_o7_actuated_scene_v13_cylinder.xml \
  --template records/fr3_o7_grasp_template_cylinder_v1.json \
  --trials 1 \
  --seed 1 \
  --spawn-source template \
  --xy-range 0.0 \
  --yaw-range-deg 0.0 \
  --pregrasp-z 0.08 \
  --lift-joint fr3_joint3 \
  --lift-delta 0.18 \
  --viewer



最终路线应该变成：

新物体/新场景
→ 从数据集中检索多个可能的拟人抓握候选
→ 根据当前场景做可行性筛选
→ 为每个候选自动生成 approach 路径
→ 仿真/几何快速验证
→ 选择最优候选执行

也就是说：

数据集提供“像人一样怎么抓”的先验；
规划器负责“在当前真实环境中怎么接近、怎么闭合、怎么避障”。


基线demo：
cd ~/Projects/o7_mujoco_sim
./run_viewer_current_good_demo_debug.sh