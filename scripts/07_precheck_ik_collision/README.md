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
