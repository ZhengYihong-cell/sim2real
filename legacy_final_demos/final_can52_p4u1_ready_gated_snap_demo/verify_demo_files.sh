#!/usr/bin/env bash
set -euo pipefail

cd "$(dirname "$0")"

echo "========== CHECK FILES =========="

REQ=(
  "scripts/05_execution_runner/run_v4_12p4u1_precontact_snap_close_debug.py"
  "diagnostics/current_v412/v4_12p4t2_scene_can52_contact_stable_old_ellipsoid_proxy.xml"
  "diagnostics/current_v412/v4_12p4h2_can52_contact_sequence_proxy_debug/best_candidate.json"
  "diagnostics/current_v412/v4_12p4h2_can52_contact_sequence_proxy_debug/best_p3.json"
  "diagnostics/current_v412/v4_12p4h2_can52_contact_sequence_proxy_debug/best_contact_sequence_config_ctrlsplit_debug.json"
  "assets"
  "run_demo_viewer.sh"
  "run_demo_headless.sh"
)

for f in "${REQ[@]}"; do
  if [ ! -e "$f" ]; then
    echo "[MISSING] $f"
    exit 1
  fi
  echo "[OK] $f"
done

echo
echo "========== PY COMPILE =========="
python3 -m py_compile scripts/05_execution_runner/run_v4_12p4u1_precontact_snap_close_debug.py

echo
echo "========== MUJOCO MODEL LOAD =========="
python3 - <<'PY'
import mujoco
p = "diagnostics/current_v412/v4_12p4t2_scene_can52_contact_stable_old_ellipsoid_proxy.xml"
m = mujoco.MjModel.from_xml_path(p)
print("[OK] loaded:", p)
print("nbody:", m.nbody)
print("ngeom:", m.ngeom)
print("nu:", m.nu)
PY

echo
echo "[OK] demo verified"
