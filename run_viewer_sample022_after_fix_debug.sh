#!/usr/bin/env bash
set -e

cd ~/Projects/o7_mujoco_sim
source ~/mujoco_env/bin/activate

unset MUJOCO_GL
unset PYOPENGL_PLATFORM
unset LD_PRELOAD

./run_mujoco_clean.sh scripts/run_fr3_o7_prior_contact_seek_debug.py \
  --viewer \
  --model models/fr3_o7/fr3_o7_bottle_scene_handbase_upright_tabletop_scale006_debug.xml \
  --candidate records/candidates/prior_contact_seek_safe_wrist_debug/bottle_tabletop_sample022_safe_wrist_palm_in_up_debug.json \
  --trials 1 \
  --seed 42 \
  --xy-range 0.0 \
  --yaw-range-deg 0.0 \
  --z-shift 0.0 \
  --spawn-source model \
  --pregrasp-distance 0.08 \
  --min-side-clearance 0.065 \
  --min-handbase-radial-dist 0.075 \
  --handbase-surface-margin 0.045 \
  --preshape-alpha 0.0 \
  --move-duration 3.0 \
  --mid-duration 1.5 \
  --seek-duration 3.0 \
  --close-duration 3.0 \
  --hold-duration 1.5 \
  --lift-z 0.025 \
  --lift-duration 2.5 \
  --min-final-hand-object 2 \
  --out diagnostics/prior_contact_seek_safe_wrist_debug/sample022_safe_wrist_contact_seek_viewer_after_fix_debug.json
