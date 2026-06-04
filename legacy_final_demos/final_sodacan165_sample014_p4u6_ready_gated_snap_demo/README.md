# final_sodacan165_sample014_p4u6_ready_gated_snap_demo

这是 sem-SodaCan-16526 的第一轮泛化成功 demo。

流程：
dataset prior sample014 -> candidate/scene -> P2/P3 -> contact-free P3 goal -> sodacan-specific hand_config -> P4U6 path -> P4U1 ready-gated snap close -> fixed-grip lift

成功指标：
grip_ready=True
final_object_rise≈0.0656 m
final_groups={'thumb': 1, 'middle': 1, 'pinky': 1}
final_opposition_cos≈-0.755
三段 path 均 success=True

运行：
cd ~/Projects/o7_mujoco_sim
./legacy_final_demos/final_sodacan165_sample014_p4u6_ready_gated_snap_demo/run_demo_viewer.sh

无界面快速回归：
cd ~/Projects/o7_mujoco_sim
./legacy_final_demos/final_sodacan165_sample014_p4u6_ready_gated_snap_demo/run_demo_headless.sh

注意：
不要用本 demo 覆盖 can52 legacy demo。
不要把 sample014 参数当成所有物体的固定参数。
后续新物体仍然要走 Top-K -> P2/P3 -> object-specific hand_config -> P4U6 -> P4U1。
