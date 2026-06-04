#  文件说明

旧 ellipsoid proxy 碰撞层：

v4_12p4t2_scene_can52_contact_stable_old_ellipsoid_proxy.xml

从旧 O7-only 模型迁移过来的红色椭球碰撞体版本。这个前提很重要，之前 raw mesh collision 的接触法向太乱，物体很容易被单侧接触推倒


## 细节说明
1.“慢慢磨进去闭合”改成了 snap close。慢速闭合时，单根手指会先碰到 can，然后持续推它，导致 can 往手心倒。side-open 到位 → 0.45s 快速同步闭合到 close_target

2.保持 close_target 控制目标。也就是手指执行器继续给夹持力，而不是停在当前 qpos 上。

3.加了 gated micro-squeeze。作用是：snap close 后如果已经接近抓住，但接触还不够稳，就再给一点点夹紧量。并且设置了保护：
--max-grip-disp 0.006
--max-extra-disp-during-squeeze 0.003

4.lift 被 ready gate 控制。只有 thumb + 至少一根四指形成稳定对抗接触并且连续达到 grip_ready_stable_steps才允许 lift

