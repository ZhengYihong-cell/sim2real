#!/usr/bin/env python3
"""
文件名：
    patch_v4_12p4c_high_friction_scene_debug.py

脚本类别：
    debug / scene-patch / friction-test

用途：
    本脚本用于 V4.12P4C 阶段的高摩擦隔离实验。
    它只修改 MuJoCo XML 中指定 object body 及蓝色支撑垫块相关 geom 的 friction / condim，
    用于验证“物体被挤出抓握范围”是否与 object-support 摩擦不足有关。

输入：
    1. --input
       原始 MuJoCo XML 场景。
    2. --output
       输出的新 MuJoCo XML 场景。
    3. --object-body
       物体 body 名，例如 grasp_can。
    4. --support-token
       支撑垫块名称关键词，例如 object_pedestal。
    5. --friction
       MuJoCo geom friction 三元组：滑动摩擦、扭转摩擦、滚动摩擦。
       例如 "5 1 1"。
    6. --condim
       接触维度。设置为 6 时会启用更完整的摩擦模型。

输出：
    1. 新 XML 文件，不覆盖原始模型。
    2. 终端打印被修改的 geom 名称、旧 friction、新 friction。

当前流程位置：
    原始 can52 tabletop XML
        -> 本脚本生成 high-friction XML
        -> 重新运行 P4C
        -> 对比 object displacement / hand_object / object_support

本脚本不负责：
    1. 不修改 candidate。
    2. 不修改 IK。
    3. 不修改 P4C 闭合控制。
    4. 不判断抓取成功，只生成高摩擦测试场景。
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


def find_body(root, body_name):
    for body in root.iter("body"):
        if body.get("name", "") == body_name:
            return body
    return None


def collect_geoms_recursive(elem, out):
    for child in list(elem):
        if child.tag == "geom":
            out.append(child)
        collect_geoms_recursive(child, out)


def patch_geom(geom, friction, condim, reason):
    old_friction = geom.get("friction", "")
    old_condim = geom.get("condim", "")

    geom.set("friction", friction)

    if condim is not None:
        geom.set("condim", str(condim))

    return {
        "name": geom.get("name", ""),
        "reason": reason,
        "old_friction": old_friction,
        "new_friction": friction,
        "old_condim": old_condim,
        "new_condim": str(condim) if condim is not None else old_condim,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--input", required=True)
    ap.add_argument("--output", required=True)
    ap.add_argument("--object-body", default="grasp_can")
    ap.add_argument("--support-token", default="object_pedestal")
    ap.add_argument("--friction", default="5 1 1")
    ap.add_argument("--condim", type=int, default=6)
    ap.add_argument("--patch-object", action="store_true")
    ap.add_argument("--patch-support", action="store_true")
    args = ap.parse_args()

    src = resolve_path(args.input)
    out = resolve_path(args.output)
    out.parent.mkdir(parents=True, exist_ok=True)

    if not args.patch_object and not args.patch_support:
        # 默认两者都改，做一次“强摩擦上限测试”
        args.patch_object = True
        args.patch_support = True

    tree = ET.parse(src)
    root = tree.getroot()

    patched = []

    if args.patch_object:
        body = find_body(root, args.object_body)
        if body is None:
            raise RuntimeError(f"cannot find object body: {args.object_body}")

        obj_geoms = []
        collect_geoms_recursive(body, obj_geoms)

        for geom in obj_geoms:
            patched.append(
                patch_geom(
                    geom,
                    args.friction,
                    args.condim,
                    reason=f"object_body={args.object_body}",
                )
            )

    if args.patch_support:
        token = args.support_token.lower()
        for geom in root.iter("geom"):
            name = geom.get("name", "")
            if token in name.lower() or "pedestal" in name.lower() or "support" in name.lower():
                patched.append(
                    patch_geom(
                        geom,
                        args.friction,
                        args.condim,
                        reason=f"support_token={args.support_token}",
                    )
                )

    if not patched:
        raise RuntimeError("No geom patched. Check --object-body and --support-token.")

    tree.write(out, encoding="utf-8", xml_declaration=True)

    print("\n========== V4.12P4C HIGH FRICTION PATCH ==========")
    print("input   :", src)
    print("output  :", out)
    print("friction:", args.friction)
    print("condim  :", args.condim)
    print("patched geoms:")
    for item in patched:
        print(
            f"  {item['name']:<36} "
            f"reason={item['reason']:<28} "
            f"friction: '{item['old_friction']}' -> '{item['new_friction']}', "
            f"condim: '{item['old_condim']}' -> '{item['new_condim']}'"
        )
    print("==================================================\n")


if __name__ == "__main__":
    main()
