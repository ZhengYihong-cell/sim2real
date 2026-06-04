#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

if [ -d "${MUJOCO_ENV:-$HOME/mujoco_env}" ]; then
  source "${MUJOCO_ENV:-$HOME/mujoco_env}/bin/activate"
fi

./run_mujoco_clean.sh scripts/05_execution_runner/run_v4_12p4u1_precontact_snap_close_debug.py \
  --model diagnostics/current_v412/v4_12p4t2_scene_can52_contact_stable_old_ellipsoid_proxy.xml \
  --candidate diagnostics/current_v412/v4_12p4h2_can52_contact_sequence_proxy_debug/best_candidate.json \
  --p3-json diagnostics/current_v412/v4_12p4h2_can52_contact_sequence_proxy_debug/best_p3.json \
  --best-config diagnostics/current_v412/v4_12p4h2_can52_contact_sequence_proxy_debug/best_contact_sequence_config_ctrlsplit_debug.json \
  --which best_available \
  --object-body grasp_can \
  --target-body fr3_link7 \
  --out outputs/can52_p4u1_ready_gated_snap_micro_squeeze_lift_headless.json \
  --move-steps 80 \
  --thumb-preshape-steps 80 \
  --close-duration 0.45 \
  --post-close-target-hold-duration 0.25 \
  --micro-squeeze-duration 0.35 \
  --micro-squeeze-fraction 0.08 \
  --finger-close-scale 0.92 \
  --thumb-pitch-from-finger-gain 0.24 \
  --grip-ready-stable-steps 8 \
  --min-live-non-thumb 1 \
  --opposition-cos-threshold -0.30 \
  --max-grip-disp 0.006 \
  --max-extra-disp-during-squeeze 0.003 \
  --enable-lift \
  --lift-mode world_z \
  --lift-z 0.060 \
  --lift-duration 2.0 \
  --print-every-steps 30 \
  --log-every-steps 30 \
  2>&1 | tee outputs/can52_p4u1_ready_gated_snap_micro_squeeze_lift_headless_terminal.txt
