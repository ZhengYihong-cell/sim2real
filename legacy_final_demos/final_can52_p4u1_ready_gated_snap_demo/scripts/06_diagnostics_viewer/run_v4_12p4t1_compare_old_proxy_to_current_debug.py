#!/usr/bin/env python3
"""
脚本类型：
    debug / diagnostic / collision-proxy-compare

用途：
    V4.12P4T1。
    对比旧 O7-only stable grasp 场景和当前 FR3+O7 场景的 O7 手部碰撞层。
    重点检查：
        1. 旧模型中的 ellipsoid proxy 数量、名称、body、pos、quat、size；
        2. 当前 FR3+O7 中是否存在同名 body；
        3. 当前模型中 O7 手部 mesh collision 数量；
        4. 判断 ellipsoid proxy 是否可以迁移到当前模型。

输入：
    --old-xml      旧 O7-only stable XML
    --current-xml  当前 FR3+O7 XML
    --out          输出 JSON
    --report       输出 TXT 报告

输出：
    JSON + TXT 对比报告。

不负责：
    不修改 XML；
    不运行抓取；
    不加入 sensor；
    不做 IK。
"""

from pathlib import Path
import argparse
import json
import xml.etree.ElementTree as ET


PROJECT = Path.home() / "Projects/o7_mujoco_sim"


def resolve_path(p):
    p = Path(str(p)).expanduser()
    if not p.is_absolute():
        p = PROJECT / p
    return p


def parent_map(root):
    m = {}
    for p in root.iter():
        for c in list(p):
            m[c] = p
    return m


def nearest_body(elem, pm):
    cur = elem
    while cur in pm:
        cur = pm[cur]
        if cur.tag == "body":
            return cur
    return None


def body_names(root):
    return {b.get("name") for b in root.iter("body") if b.get("name")}


def is_o7_body_name(name):
    if not name:
        return False
    t = name.lower()
    return any(k in t for k in [
        "thumb", "index", "middle", "ring", "pinky",
        "hand_base", "palm", "metacarpal", "proximal", "distal"
    ])


def geom_record(g, pm):
    b = nearest_body(g, pm)
    bname = b.get("name") if b is not None else None
    return {
        "geom_name": g.get("name"),
        "body": bname,
        "type": g.get("type", "mesh"),
        "pos": g.get("pos"),
        "quat": g.get("quat"),
        "size": g.get("size"),
        "mesh": g.get("mesh"),
        "class": g.get("class"),
        "contype": g.get("contype"),
        "conaffinity": g.get("conaffinity"),
        "group": g.get("group"),
        "friction": g.get("friction"),
        "solref": g.get("solref"),
        "solimp": g.get("solimp"),
        "condim": g.get("condim"),
        "rgba": g.get("rgba"),
        "material": g.get("material"),
    }


def collect_geoms(root):
    pm = parent_map(root)
    out = []
    for g in root.iter("geom"):
        rec = geom_record(g, pm)
        out.append(rec)
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-xml", required=True)
    ap.add_argument("--current-xml", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    args = ap.parse_args()

    old_xml = resolve_path(args.old_xml)
    cur_xml = resolve_path(args.current_xml)

    old_root = ET.parse(old_xml).getroot()
    cur_root = ET.parse(cur_xml).getroot()

    old_bodies = body_names(old_root)
    cur_bodies = body_names(cur_root)

    old_geoms = collect_geoms(old_root)
    cur_geoms = collect_geoms(cur_root)

    old_ellipsoids = [
        g for g in old_geoms
        if g["type"] == "ellipsoid" and is_o7_body_name(g["body"])
    ]

    old_o7_mesh = [
        g for g in old_geoms
        if g["type"] == "mesh" and is_o7_body_name(g["body"])
    ]

    cur_o7_mesh_collision = [
        g for g in cur_geoms
        if g["type"] == "mesh"
        and is_o7_body_name(g["body"])
        and str(g.get("contype")) not in ["0", "None"]
    ]

    missing_bodies = sorted({
        g["body"] for g in old_ellipsoids
        if g["body"] not in cur_bodies
    })

    matched = []
    for g in old_ellipsoids:
        matched.append({
            **g,
            "body_exists_in_current": g["body"] in cur_bodies,
        })

    result = {
        "old_xml": str(old_xml),
        "current_xml": str(cur_xml),
        "num_old_ellipsoid_proxy": len(old_ellipsoids),
        "num_old_o7_mesh": len(old_o7_mesh),
        "num_current_o7_mesh_collision": len(cur_o7_mesh_collision),
        "missing_bodies_for_old_ellipsoid": missing_bodies,
        "old_ellipsoid_proxy": matched,
        "current_o7_mesh_collision": cur_o7_mesh_collision,
    }

    out = resolve_path(args.out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2, ensure_ascii=False))

    report = resolve_path(args.report)
    report.parent.mkdir(parents=True, exist_ok=True)

    with open(report, "w") as f:
        f.write("========== V4.12P4T1 OLD PROXY VS CURRENT ==========\n\n")
        f.write(f"old_xml: {old_xml}\n")
        f.write(f"current_xml: {cur_xml}\n\n")
        f.write(f"num_old_ellipsoid_proxy: {len(old_ellipsoids)}\n")
        f.write(f"num_old_o7_mesh: {len(old_o7_mesh)}\n")
        f.write(f"num_current_o7_mesh_collision: {len(cur_o7_mesh_collision)}\n")
        f.write(f"missing_bodies_for_old_ellipsoid: {missing_bodies}\n\n")

        f.write("---- OLD ELLIPSOID PROXY ----\n")
        for g in matched:
            f.write(
                f"body={g['body']:<28s} "
                f"name={str(g['geom_name']):<36s} "
                f"exists={g['body_exists_in_current']} "
                f"pos={g['pos']} quat={g['quat']} size={g['size']} "
                f"friction={g['friction']} condim={g['condim']} "
                f"contype={g['contype']} conaffinity={g['conaffinity']}\n"
            )

        f.write("\n---- CURRENT O7 MESH COLLISION ----\n")
        for g in cur_o7_mesh_collision:
            f.write(
                f"body={g['body']:<28s} "
                f"name={str(g['geom_name']):<36s} "
                f"mesh={g['mesh']} "
                f"contype={g['contype']} conaffinity={g['conaffinity']} "
                f"group={g['group']}\n"
            )

    print("[SAVED]", out)
    print("[REPORT]", report)
    print("num_old_ellipsoid_proxy:", len(old_ellipsoids))
    print("num_current_o7_mesh_collision:", len(cur_o7_mesh_collision))
    print("missing_bodies:", missing_bodies)


if __name__ == "__main__":
    main()
