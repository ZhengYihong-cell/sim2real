#!/usr/bin/env bash
set -e

# 清理 ROS / MoveIt / conda 相关变量，避免污染 MuJoCo
unset ROS_DISTRO
unset ROS_VERSION
unset ROS_PACKAGE_PATH
unset AMENT_PREFIX_PATH
unset CMAKE_PREFIX_PATH
unset COLCON_PREFIX_PATH
unset PYTHONPATH
unset CONDA_PREFIX
unset CONDA_DEFAULT_ENV
unset CONDA_SHLVL

export PATH=$(echo "$PATH" | tr ':' '\n' \
  | grep -v "miniconda3" \
  | grep -v "Moveit2_ws" \
  | grep -v "o7_minimal_sim" \
  | paste -sd:)

export LD_LIBRARY_PATH=$(echo "${LD_LIBRARY_PATH:-}" | tr ':' '\n' \
  | grep -v "miniconda3" \
  | grep -v "Moveit2_ws" \
  | grep -v "o7_minimal_sim" \
  | paste -sd:)

export MUJOCO_GL=glfw

exec /home/zyh/mujoco_env/bin/python "$@"
