#!/usr/bin/env python3
"""
文件名：
    patch_v4_12p4k_hard_support_contact_debug.py

脚本类别：
    debug / model-patch / hard-contact / support-contact

用途：
    本脚本用于 V4.12P4K 阶段。
    它不负责抓取控制，也不负责判断抓握成功失败。
    它只负责把 MuJoCo XML 场景中的“物体-蓝色支撑块”接触改成更硬的接触模型，
    让物体在仿真中尽量不能穿入蓝色垫块。

输入：
    --in-xml
        原始 MuJoCo XML 场景。
    --out-xml
        输出的硬接触 MuJoCo XML 场景。
    --object-body
        被抓物体 body 名称，例如 grasp_can。
    --support-token
        支撑块 geom/body 名称关键词，例如 pedestal/support/table。

输出：
    一个新的 XML 文件。该 XML 会：
        1. 给 object geom 和 support geom 设置更硬的 solref / solimp；
        2. 给 object-support 添加显式 contact pair；
        3. 提高接触维度 condim 和摩擦参数；
        4. 保留原模型主体结构。

当前流程位置：
    原始 tabletop scene
        -> hard support contact scene
        -> 再运行 P4H/P4J/P4K 抓取验证

本脚本不负责：
    1. 不运行仿真；
    2. 不修改 IK；
    3. 不修改候选抓握；
    4. 不把“穿透超过阈值失败”当成解决方案；
    5. 不保证所有 penetration 绝对为 0，因为 MuJoCo 接触求解仍可能有极小数值残差。
"""

from pathlib import Path
import argparse
import xml.etree.ElementTree as ET


def find_body(elem, name):
    if elem.tag == "body" and elem.get("name") == name:
        return elem
    for child in list(elem):
        found = find_body(child, name)
        if found is not None:
            return found
    return None


def collect_geom_names_under(elem):
    names = []
    for e in elem.iter():
        if e.tag == "geom" and e.get("name"):
            names.append(e.get("name"))
    return names


def get_or_make(root, tag):
    e = root.find(tag)
    if e is None:
        e = ET.SubElement(root, tag)
    return e


def remove_existing_pair(contact, g1, g2):
    for pair in list(contact):
        if pair.tag != "pair":
            continue
        a = pair.get("geom1")
        b = pair.get("geom2")
        if {a, b} == {g1, g2}:
            contact.remove(pair)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-xml", required=True)
    ap.add_argument("--out-xml", required=True)
    ap.add_argument("--object-body", default="grasp_can")
    ap.add_argument("--support-token", default="pedestal,support,table")
    ap.add_argument("--solref", default="-50000 -1000")
    ap.add_argument("--solimp", default="0.995 0.999 0.0001 0.5 2")
    ap.add_argument("--friction", default="3.0 0.5 0.5")
    ap.add_argument("--condim", default="4")
    ap.add_argument("--margin", default="0")
    ap.add_argument("--gap", default="0")
    args = ap.parse_args()

    in_xml = Path(args.in_xml).expanduser()
    out_xml = Path(args.out_xml).expanduser()
    out_xml.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(in_xml)
    root = tree.getroot()

    object_body = find_body(root, args.object_body)
    if object_body is None:
        raise RuntimeError(f"cannot find object body: {args.object_body}")

    object_geoms = collect_geom_names_under(object_body)
    if not object_geoms:
        raise RuntimeError(f"no geom under object body: {args.object_body}")

    tokens = [x.strip().lower() for x in args.support_token.split(",") if x.strip()]
    support_geoms = []

    for geom in root.iter("geom"):
        name = geom.get("name", "")
        text = name.lower()
        if any(t in text for t in tokens):
            support_geoms.append(name)

    if not support_geoms:
        raise RuntimeError(f"cannot find support geom by tokens: {tokens}")

    # 给 object geom 和 support geom 本身也设置硬接触参数。
    hard_geom_names = set(object_geoms + support_geoms)
    for geom in root.iter("geom"):
        name = geom.get("name", "")
        if name in hard_geom_names:
            geom.set("solref", args.solref)
            geom.set("solimp", args.solimp)
            geom.set("friction", args.friction)
            geom.set("condim", args.condim)
            geom.set("margin", args.margin)
            geom.set("gap", args.gap)

    # 显式添加 object-support contact pair。
    contact = get_or_make(root, "contact")

    for sg in support_geoms:
        for og in object_geoms:
            remove_existing_pair(contact, sg, og)
            pair = ET.SubElement(contact, "pair")
            pair.set("geom1", sg)
            pair.set("geom2", og)
            pair.set("solref", args.solref)
            pair.set("solimp", args.solimp)
            pair.set("friction", args.friction)
            pair.set("condim", args.condim)
            pair.set("margin", args.margin)
            pair.set("gap", args.gap)

    # 提高求解迭代，但不改 timestep，避免 runner 时间尺度变化。
    option = get_or_make(root, "option")
    option.set("iterations", "100")
    option.set("ls_iterations", "50")

    tree.write(out_xml, encoding="utf-8", xml_declaration=True)

    print("========== HARD SUPPORT CONTACT PATCH ==========")
    print("in_xml       :", in_xml)
    print("out_xml      :", out_xml)
    print("object_body  :", args.object_body)
    print("object_geoms :", object_geoms)
    print("support_geoms:", support_geoms)
    print("solref       :", args.solref)
    print("solimp       :", args.solimp)
    print("friction     :", args.friction)
    print("condim       :", args.condim)
    print("===============================================")


if __name__ == "__main__":
    main()
