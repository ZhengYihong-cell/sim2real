#!/usr/bin/env bash
# 脚本类型：
#   debug / maintenance / file-discovery
#
# 用途：
#   为第一轮 O7 抓取泛化定位候选物体相关文件。
#   优先检查 sem-SodaCan / core-can / core-jar / core-bottle 是否在 dataset、records、diagnostics 中存在 mesh、npy、json、xml。
#
# 输入：
#   无命令行参数。默认在 ~/Projects/o7_mujoco_sim 下搜索。
#
# 输出：
#   diagnostics/current_v412/generalization_target_discovery_debug.txt
#
# 当前流程位置：
#   泛化前的文件定位阶段。
#
# 不负责：
#   1. 不修改模型；
#   2. 不生成 candidate；
#   3. 不运行 IK；
#   4. 不运行 viewer；
#   5. 不动 legacy_final_demos。

set -euo pipefail

cd ~/Projects/o7_mujoco_sim
mkdir -p diagnostics/current_v412

OUT="diagnostics/current_v412/generalization_target_discovery_debug.txt"
exec > >(tee "$OUT") 2>&1

echo "========== GENERALIZATION TARGET DISCOVERY DEBUG =========="
echo "project: $(pwd)"
echo "out    : $OUT"
echo

TARGETS=(
  "sem-SodaCan-16526d147e837c386829bf9ee210f5e7"
  "sem-SodaCan-343287cd508a798d38df439574e01b2"
  "sem-SodaCan-3c8af6b0aeaf13c2abf4b6b757f4f768"
  "sem-SodaCan-bb0262b63f857b24dc4868f575aa7e3c"
  "sem-SodaCan-cbb5347c6da1d885c617fcca80b33ab4"
  "core-can-10c9a321485711a88051229d056d81db"
  "core-can-af444e72a44e6db153c22afefaf6f2a4"
  "core-can-eac30c41aad2ff27c0ca8d7a07be3be"
  "core-jar-1168c9e9db2c1c5066639e628d6519b6"
  "core-jar-166c3012a4b35768f51f77a6d7299806"
  "core-bottle-1071fa4cddb2da2fc8724d5673a063a6"
  "core-bottle-3b0e35ff08f09a85f0d11ae402ef940e"
)

for obj in "${TARGETS[@]}"; do
  echo
  echo "------------------------------------------------------------"
  echo "OBJECT: $obj"
  echo "------------------------------------------------------------"

  echo "[mesh/data dirs]"
  find dataset assets models records diagnostics -type d -iname "*$obj*" 2>/dev/null | sort | head -40 || true

  echo
  echo "[mesh files]"
  find dataset assets models records diagnostics -type f \
    \( -iname "*.obj" -o -iname "*.stl" -o -iname "*.ply" -o -iname "*.xml" \) 2>/dev/null \
    | grep -F "$obj" | sort | head -60 || true

  echo
  echo "[candidate / npy / json files]"
  find dataset assets models records diagnostics -type f \
    \( -iname "*.json" -o -iname "*.npy" -o -iname "*.npz" -o -iname "*.txt" \) 2>/dev/null \
    | grep -F "$obj" | sort | head -80 || true
done

echo
echo "========== SCRIPT CANDIDATE GENERATORS =========="
find scripts -type f -iname "*.py" 2>/dev/null \
  | grep -Ei "candidate|dataset|source|select|p4e|p4h|p4h2|scene|mesh|object" \
  | sort

echo
echo "========== DONE =========="
echo "saved: $OUT"
