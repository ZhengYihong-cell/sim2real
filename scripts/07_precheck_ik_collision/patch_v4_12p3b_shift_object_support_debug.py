#!/usr/bin/env python3
"""
文件名：
    patch_v4_12p3b_shift_object_support_debug.py

脚本类别：
    debug / scene-patch / precheck-support-tool

用途：
    本脚本用于 V4.12P3B 阶段，快速验证“物体位置是否导致 FR3 joint7 贴近极限”。
    它会把指定物体 body 和蓝色支撑垫块 geom/body 一起做世界坐标平移。
    典型用法是把 can 和 object_pedestal 一起沿 x 负方向拉近 Franka 基座。

输入：
    1. 一个 MuJoCo XML 场景文件。
    2. object body 名称，例如 grasp_can。
    3. support 名称关键词，例如 object_pedestal。
    4. dx/dy/dz 平移量。

输出：
    1. 一个新的 MuJoCo XML，不覆盖原始模型。
    2. 终端打印哪些 body/geom 被移动。

当前流程位置：
    当前 XML 场景
        -> 平移 object + support
        -> V4.12P2 Pinocchio 多 seed IK
        -> V4.12P3/P3A 碰撞与关节余量预检

本脚本不负责：
    1. 不修改 dataset。
    2. 不修改 candidate。
    3. 不重新求 IK。
    4. 不启动 viewer。
    5. 不判断抓取是否成功。
"""

from pathlib import Path
import argparse
import xml.etree.ElementTree as ET


PROJECT = Path.home() / "Projects/o7_mujoco_sim"


def resolve_path(p):
    p = Path(str(p)).expanduser()
    if not p.is_absolute():
        p = PROJECT / p
    return p


def parse_vec(s):
    vals = [float(x) for x in str(s).split()]
    if len(vals) != 3:
        raise ValueError(f"pos must have 3 values, got: {s}")
    return vals


def vec_to_str(v):
    return f"{v[0]:.9g} {v[1]:.9g} {v[2]:.9g}"


def shift_element_pos(elem, delta):
    old = parse_vec(elem.get("pos", "0 0 0"))
    new = [old[i] + delta[i] for i in range(3)]
    elem.set("pos", vec_to_str(new))
    return old, new


def name_matches(name, tokens):
    name = str(name or "").lower()
    return any(t.lower() in name for t in tokens)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--object-body", default="grasp_can")
    ap.add_argument("--support-token", default="object_pedestal")
    ap.add_argument("--dx", type=float, default=-0.08)
    ap.add_argument("--dy", type=float, default=0.0)
    ap.add_argument("--dz", type=float, default=0.0)
    args = ap.parse_args()

    src = resolve_path(args.input)
    out = resolve_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    delta = [args.dx, args.dy, args.dz]

    tree = ET.parse(src)
    root = tree.getroot()

    moved = []

    # 1. 移动物体 body，例如 grasp_can。
    for body in root.iter("body"):
        name = body.get("name", "")
        if name == args.object_body:
            old, new = shift_element_pos(body, delta)
            moved.append({
                "type": "body",
                "name": name,
                "old_pos": old,
                "new_pos": new,
                "reason": "object_body",
            })

    # 2. 移动支撑垫块 geom，例如 object_pedestal。
    support_tokens = [args.support_token, "pedestal"]
    for geom in root.iter("geom"):
        name = geom.get("name", "")
        if name_matches(name, support_tokens):
            old, new = shift_element_pos(geom, delta)
            moved.append({
                "type": "geom",
                "name": name,
                "old_pos": old,
                "new_pos": new,
                "reason": "support_geom",
            })

    # 3. 如果支撑垫块本身是 body，也移动。
    for body in root.iter("body"):
        name = body.get("name", "")
        if name_matches(name, support_tokens) and name != args.object_body:
            old, new = shift_element_pos(body, delta)
            moved.append({
                "type": "body",
                "name": name,
                "old_pos": old,
                "new_pos": new,
                "reason": "support_body",
            })

    if not moved:
        raise RuntimeError(
            "No object/support element moved. Check --object-body and --support-token."
        )

    tree.write(out, encoding="utf-8", xml_declaration=True)

    print("\n========== V4.12P3B SHIFT OBJECT + SUPPORT ==========")
    print("input :", src)
    print("output:", out)
    print("delta :", delta)
    print("\nMoved elements:")
    for m in moved:
        print(
            f"  {m['type']:>4} {m['name']:<32} "
            f"{m['old_pos']} -> {m['new_pos']}  reason={m['reason']}"
        )
    print("=====================================================\n")


if __name__ == "__main__":
    main()
