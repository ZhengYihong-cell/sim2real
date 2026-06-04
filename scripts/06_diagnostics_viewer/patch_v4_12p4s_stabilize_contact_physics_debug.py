#!/usr/bin/env python3
"""
文件名：
    patch_v4_12p4s_stabilize_contact_physics_debug.py

脚本类别：
    debug / patcher / physics-stabilization

用途：
    本脚本用于 V4.12P4S 阶段。
    当前 can 在 MuJoCo 中被手指轻微接触后出现明显不符合真实世界的移动/倾倒/弹飞。
    本脚本生成一个更稳定的接触物理 XML，用于验证问题是否来自底层 contact 参数。

它会修改：
    1. option：
        timestep、solver、iterations、tolerance；
    2. object geom / support geom / hand geom：
        friction、solref、solimp、condim、margin、gap；
    3. 可选：
        给 object body 写入显式 inertial，避免物体质量过轻或惯量异常。

输入：
    --in-xml
        原始 XML。
    --out-xml
        输出 XML。
    --object-body
        物体 body 名，例如 grasp_can。
    --support-tokens
        支撑块名称关键词。
    --object-mass
        若 >0，则给 object body 写入显式 inertial 质量和近似圆柱惯量。

输出：
    新 XML 文件。

当前流程位置：
    P4S 体检之后，用新 XML 复跑 P4P3/P4R，确认是否物理参数导致不真实移动。

本脚本不负责：
    1. 不修改抓握控制逻辑；
    2. 不判定抓取成功；
    3. 不永久覆盖原模型；
    4. 不把仿真调成“作弊固定物体”，只是增强真实桌面接触稳定性。
"""

from pathlib import Path
import argparse
import xml.etree.ElementTree as ET
import math


PROJECT = Path.home() / "Projects/o7_mujoco_sim"


def resolve_path(p):
    p = Path(str(p)).expanduser()
    if not p.is_absolute():
        p = PROJECT / p
    return p


def parse_vec(s, default):
    if s is None:
        return list(default)
    vals = [float(x) for x in str(s).split()]
    if not vals:
        return list(default)
    return vals


def vec_to_str(v):
    return " ".join(f"{float(x):.9g}" for x in v)


def find_body(elem, name):
    if elem.tag == "body" and elem.get("name") == name:
        return elem
    for c in list(elem):
        r = find_body(c, name)
        if r is not None:
            return r
    return None


def body_contains_geom_name(body, keyword_list):
    for g in body.iter("geom"):
        name = (g.get("name") or "").lower()
        if any(k in name for k in keyword_list):
            return True
    return False


def is_hand_geom(elem):
    text = ((elem.get("name") or "") + " " + (elem.get("class") or "")).lower()
    return any(k in text for k in ["thumb", "index", "middle", "ring", "pinky", "palm", "hand"])


def collect_parent_map(root):
    parent = {}
    for p in root.iter():
        for c in list(p):
            parent[c] = p
    return parent


def inherited_body_name(elem, parent_map):
    cur = elem
    while cur in parent_map:
        cur = parent_map[cur]
        if cur.tag == "body":
            return cur.get("name") or ""
    return ""


def patch_geom(g, friction, solref, solimp, condim, margin, gap):
    g.set("friction", friction)
    g.set("solref", solref)
    g.set("solimp", solimp)
    g.set("condim", str(condim))
    g.set("margin", str(margin))
    g.set("gap", str(gap))


def set_object_inertial(body, mass, radius, height):
    # 近似实心圆柱惯量。对罐子来说不完全真实，但比质量/惯量异常更稳定。
    Ixx = (1.0 / 12.0) * mass * (3.0 * radius * radius + height * height)
    Iyy = Ixx
    Izz = 0.5 * mass * radius * radius

    inertial = None
    for c in list(body):
        if c.tag == "inertial":
            inertial = c
            break

    if inertial is None:
        inertial = ET.Element("inertial")
        body.insert(0, inertial)

    inertial.set("pos", "0 0 0")
    inertial.set("quat", "1 0 0 0")
    inertial.set("mass", f"{mass:.9g}")
    inertial.set("diaginertia", f"{Ixx:.9g} {Iyy:.9g} {Izz:.9g}")

    return {
        "mass": mass,
        "radius": radius,
        "height": height,
        "diaginertia": [Ixx, Iyy, Izz],
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-xml", required=True)
    ap.add_argument("--out-xml", required=True)
    ap.add_argument("--object-body", default="grasp_can")
    ap.add_argument("--support-tokens", default="object_pedestal,pedestal,support,table")

    ap.add_argument("--friction-object", default="2.0 0.20 0.02")
    ap.add_argument("--friction-support", default="2.5 0.25 0.03")
    ap.add_argument("--friction-hand", default="1.8 0.15 0.02")

    ap.add_argument("--solref", default="0.004 1.2")
    ap.add_argument("--solimp", default="0.98 0.995 0.0005")
    ap.add_argument("--condim", type=int, default=6)
    ap.add_argument("--margin", default="0")
    ap.add_argument("--gap", default="0")

    ap.add_argument("--timestep", default="0.001")
    ap.add_argument("--iterations", default="100")
    ap.add_argument("--ls-iterations", default="50")
    ap.add_argument("--tolerance", default="1e-10")
    ap.add_argument("--solver", default="Newton")
    ap.add_argument("--integrator", default="implicitfast")

    ap.add_argument("--object-mass", type=float, default=0.12)
    ap.add_argument("--object-radius", type=float, default=0.022)
    ap.add_argument("--object-height", type=float, default=0.115)

    args = ap.parse_args()

    in_xml = resolve_path(args.in_xml)
    out_xml = resolve_path(args.out_xml)
    out_xml.parent.mkdir(parents=True, exist_ok=True)

    tree = ET.parse(in_xml)
    root = tree.getroot()
    parent_map = collect_parent_map(root)

    option = root.find("option")
    if option is None:
        option = ET.Element("option")
        root.insert(0, option)

    option.set("timestep", args.timestep)
    option.set("iterations", args.iterations)
    option.set("ls_iterations", args.ls_iterations)
    option.set("tolerance", args.tolerance)
    option.set("solver", args.solver)
    option.set("integrator", args.integrator)

    object_body = find_body(root, args.object_body)
    if object_body is None:
        raise RuntimeError(f"cannot find object body: {args.object_body}")

    support_tokens = [x.strip().lower() for x in args.support_tokens.split(",") if x.strip()]

    patched_object = []
    patched_support = []
    patched_hand = []

    for g in root.iter("geom"):
        name = (g.get("name") or "").lower()
        body_name = inherited_body_name(g, parent_map).lower()

        in_object = body_name == args.object_body.lower()
        in_support = any(t in name or t in body_name for t in support_tokens)
        in_hand = is_hand_geom(g) or any(k in body_name for k in ["thumb", "index", "middle", "ring", "pinky", "palm", "hand"])

        if in_object:
            patch_geom(g, args.friction_object, args.solref, args.solimp, args.condim, args.margin, args.gap)
            patched_object.append(g.get("name"))
        elif in_support:
            patch_geom(g, args.friction_support, args.solref, args.solimp, args.condim, args.margin, args.gap)
            patched_support.append(g.get("name"))
        elif in_hand:
            patch_geom(g, args.friction_hand, args.solref, args.solimp, args.condim, args.margin, args.gap)
            patched_hand.append(g.get("name"))

    inertial_info = None
    if args.object_mass > 0:
        inertial_info = set_object_inertial(
            object_body,
            args.object_mass,
            args.object_radius,
            args.object_height,
        )

    tree.write(out_xml, encoding="utf-8", xml_declaration=True)

    print("========== P4S CONTACT PHYSICS PATCH ==========")
    print("in_xml :", in_xml)
    print("out_xml:", out_xml)
    print("patched_object_geoms:", patched_object)
    print("patched_support_geoms:", patched_support)
    print("patched_hand_geoms_count:", len(patched_hand))
    print("object_inertial:", inertial_info)
    print("option solver/timestep:", args.solver, args.timestep)
    print("friction object/support/hand:", args.friction_object, args.friction_support, args.friction_hand)
    print("solref:", args.solref)
    print("solimp:", args.solimp)
    print("condim:", args.condim)
    print("===============================================")


if __name__ == "__main__":
    main()
