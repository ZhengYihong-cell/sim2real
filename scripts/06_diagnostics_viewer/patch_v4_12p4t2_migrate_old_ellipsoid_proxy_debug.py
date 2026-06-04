#!/usr/bin/env python3
"""
脚本类型：
    debug / patcher / collision-proxy-migration

用途：
    V4.12P4T2。
    将旧 O7-only stable grasp 模型中的 ellipsoid collision proxy 迁移到当前 FR3+O7 场景中。

背景：
    当前 FR3+O7 场景中 O7 手部碰撞主要是 mesh collision。
    旧 O7-only stable 模型中存在已经调过的 ellipsoid proxy 碰撞层。
    当前 can 抓取中出现“接触合力方向不对、物体被推倒/推入手心”的现象，
    很可能与 raw mesh collision 接触法向不稳定有关。
    本脚本用于把旧 ellipsoid proxy 迁移到当前模型中，验证是否恢复更稳定的接触。

输入：
    --old-proxy-xml
        旧 O7-only XML，例如 models/o7_grasp_scene_v8_18b_real_contact_tune_visible.xml。
    --current-xml
        当前 FR3+O7/can 场景 XML，例如 P4S contact stable 场景。
    --out-xml
        输出迁移后的新 XML。

输出：
    一个新的 MuJoCo XML：
        1. 当前 O7 mesh collision 被禁用；
        2. 旧 ellipsoid proxy 被复制到当前同名 body 下；
        3. proxy 作为新的 O7 collision 使用；
        4. visual mesh 保留；
        5. 不加入 touch sensor。

当前流程位置：
    P4S 物理体检之后，P4P/P4R 继续抓取之前。
    这是恢复旧稳定 O7 contact layer 的第一步。

不负责：
    1. 不重新做 IK；
    2. 不运行抓取；
    3. 不加入触觉 sensor；
    4. 不覆盖原 XML；
    5. 不迁移 hand_base_link 上的 palm/root pad，除非当前模型存在同名 body。
"""

from pathlib import Path
import argparse
import xml.etree.ElementTree as ET
import copy


PROJECT = Path.home() / "Projects/o7_mujoco_sim"


O7_BODY_KEYS = [
    "thumb", "index", "middle", "ring", "pinky",
    "metacarpals", "proximal", "distal",
]


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


def body_name_of_geom(g, pm):
    b = nearest_body(g, pm)
    if b is None:
        return None
    return b.get("name")


def collect_bodies(root):
    out = {}
    for b in root.iter("body"):
        name = b.get("name")
        if name:
            out[name] = b
    return out


def is_o7_finger_body(name):
    if not name:
        return False
    t = name.lower()
    return any(k in t for k in O7_BODY_KEYS)


def is_active_mesh_collision(g, body_name):
    if g.tag != "geom":
        return False
    if g.get("type", "mesh") != "mesh":
        return False
    if not is_o7_finger_body(body_name):
        return False
    contype = g.get("contype")
    conaffinity = g.get("conaffinity")
    if contype == "0" and conaffinity == "0":
        return False
    return True


def is_old_ellipsoid_proxy(g, body_name):
    if g.tag != "geom":
        return False
    if g.get("type") != "ellipsoid":
        return False
    if not is_o7_finger_body(body_name) and body_name != "hand_base_link":
        return False
    return True


def patch_proxy_geom(g, args, index):
    new_g = copy.deepcopy(g)

    old_name = new_g.get("name") or f"old_ellipsoid_proxy_{index:03d}"
    new_g.set("name", f"p4t2_{old_name}")

    new_g.set("type", "ellipsoid")
    new_g.set("contype", "1")
    new_g.set("conaffinity", "1")
    new_g.set("group", str(args.proxy_group))
    new_g.set("condim", str(args.condim))
    new_g.set("friction", args.friction)
    new_g.set("solref", args.solref)
    new_g.set("solimp", args.solimp)
    new_g.set("margin", args.margin)
    new_g.set("gap", args.gap)

    # 让 proxy 可视化时明显可见；正式使用时可通过 group 隐藏。
    new_g.set("rgba", args.proxy_rgba)

    # 避免旧 material 在当前 XML 中不存在。
    if "material" in new_g.attrib:
        del new_g.attrib["material"]

    return new_g


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--old-proxy-xml", required=True)
    ap.add_argument("--current-xml", required=True)
    ap.add_argument("--out-xml", required=True)

    ap.add_argument("--proxy-group", type=int, default=4)
    ap.add_argument("--disabled-old-mesh-group", type=int, default=5)

    ap.add_argument("--friction", default="2.0 0.20 0.02")
    ap.add_argument("--solref", default="0.004 1.2")
    ap.add_argument("--solimp", default="0.98 0.995 0.0005")
    ap.add_argument("--condim", type=int, default=6)
    ap.add_argument("--margin", default="0")
    ap.add_argument("--gap", default="0")
    ap.add_argument("--proxy-rgba", default="1 0 0 0.55")
    ap.add_argument("--disabled-mesh-rgba", default="1 0 1 0.10")

    args = ap.parse_args()

    old_xml = resolve_path(args.old_proxy_xml)
    cur_xml = resolve_path(args.current_xml)
    out_xml = resolve_path(args.out_xml)
    out_xml.parent.mkdir(parents=True, exist_ok=True)

    old_tree = ET.parse(old_xml)
    old_root = old_tree.getroot()
    old_pm = parent_map(old_root)

    cur_tree = ET.parse(cur_xml)
    cur_root = cur_tree.getroot()
    cur_pm = parent_map(cur_root)
    cur_bodies = collect_bodies(cur_root)

    disabled_mesh = []
    skipped_mesh = []

    # 1. 禁用当前 O7 finger mesh collision。
    for g in cur_root.iter("geom"):
        bname = body_name_of_geom(g, cur_pm)
        if is_active_mesh_collision(g, bname):
            disabled_mesh.append({
                "body": bname,
                "name": g.get("name"),
                "mesh": g.get("mesh"),
            })
            g.set("contype", "0")
            g.set("conaffinity", "0")
            g.set("group", str(args.disabled_old_mesh_group))
            g.set("rgba", args.disabled_mesh_rgba)
        elif g.get("type", "mesh") == "mesh" and is_o7_finger_body(bname):
            skipped_mesh.append({
                "body": bname,
                "name": g.get("name"),
                "mesh": g.get("mesh"),
                "contype": g.get("contype"),
                "conaffinity": g.get("conaffinity"),
            })

    # 2. 从旧模型复制 ellipsoid proxy 到当前同名 body。
    migrated = []
    skipped_missing_body = []

    idx = 0
    for g in old_root.iter("geom"):
        old_bname = body_name_of_geom(g, old_pm)
        if not is_old_ellipsoid_proxy(g, old_bname):
            continue

        if old_bname not in cur_bodies:
            skipped_missing_body.append({
                "body": old_bname,
                "name": g.get("name"),
                "pos": g.get("pos"),
                "size": g.get("size"),
            })
            continue

        new_g = patch_proxy_geom(g, args, idx)
        cur_bodies[old_bname].append(new_g)
        migrated.append({
            "body": old_bname,
            "old_name": g.get("name"),
            "new_name": new_g.get("name"),
            "pos": new_g.get("pos"),
            "size": new_g.get("size"),
        })
        idx += 1

    cur_tree.write(out_xml, encoding="utf-8", xml_declaration=True)

    print("========== V4.12P4T2 MIGRATE OLD ELLIPSOID PROXY ==========")
    print("old_proxy_xml:", old_xml)
    print("current_xml  :", cur_xml)
    print("out_xml      :", out_xml)
    print("disabled current O7 mesh collision:", len(disabled_mesh))
    print("migrated ellipsoid proxy:", len(migrated))
    print("skipped missing body:", len(skipped_missing_body))
    print()
    print("missing body skipped:")
    for x in skipped_missing_body:
        print("  ", x)
    print("============================================================")


if __name__ == "__main__":
    main()
