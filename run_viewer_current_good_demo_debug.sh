#!/usr/bin/env bash
set -e

cd ~/Projects/o7_mujoco_sim
source ~/mujoco_env/bin/activate

python3 scripts/view_v4_rollout_candidate_debug.py \
  --rollout-json diagnostics/v4_2_geometry_refined_diverse_short_rollout_liftstrict_debug.json \
  --mode contact \
  --approach-mode world_z \
  --pregrasp-z 0.085 \
  --move-duration 1.50 \
  --descend-duration 0.90 \
  --close-duration 1.05 \
  --hold-duration 0.75 \
  --lift-z 0.120 \
  --lift-duration 2.00 \
  --slowdown 1.0 \
  2>&1 | tee diagnostics/view_current_good_demo_debug.txt
