#!/usr/bin/env python3
"""
脚本类型：
    debug / v4.15 / scene-patch / object-support-clearance

用途：
    修复 scene 中物体初始压进 object_pedestal 的问题。
    本脚本不换 sample，不改抓握姿态，不做手工微调抓握。
    只做一件事：
        根据 MuJoCo 编译后的真实 object-support signed distance，
        将 object body 沿 world-z 抬高到不穿透垫块。

输入：
    --model        原始 scene.xml
    --object-body  物体 body，例如 grasp_object
    --out          输出修正后的 scene xml

输出：
    修正后的 scene XML
    patch report JSON/TXT

当前流程位置：
    V4.14 已证明 site target 正确
        -> 先修 object-support 初始穿透
        -> 后续进入 V4.15 contact-aware close/lift runner

不负责：
    1. 不换 sample；
    2. 不修改手姿态；
    3. 不运行抓取；
    4. 不做 selector；
    5. 不处理手指闭合逻辑。
"""

from pathlib import Path
import argparse
import json
import xml.etree.ElementTree as ET
import numpy as np
import mujoco


PROJECT = Path.home() / "Projects/o7_mujoco_sim"


def resolve(p):
    p = Path(str(p)).expanduser()
    if not p.is_absolute():
        p = PROJECT / p
    return p


def rel(p):
    p = Path(p)
    try:
        return str(p.relative_to(PROJECT))
    except Exception:
        return str(p)


def geom_name(model, gid):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid)) or f"geom_{gid}"


def body_name(model, bid):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(bid)) or f"body_{bid}"


def collect_object_geoms(model, object_body):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
    if bid < 0:
        raise RuntimeError(f"missing object body: {object_body}")
    return [gid for gid in range(model.ngeom) if int(model.geom_bodyid[gid]) == bid]


def collect_support_geoms(model):
    out = []
    for gid in range(model.ngeom):
        g = geom_name(model, gid).lower()
        b = body_name(model, model.geom_bodyid[gid]).lower()
        s = g + " " + b
        if "world_plane" in s or "floor" in s:
            continue
        if "object_pedestal" in s or "pedestal" in s or "support" in s:
            out.append(gid)
    return out


def min_pair_distance(model, data, geoms_a, geoms_b, distmax=0.30):
    best = None
    fromto = np.zeros(6, dtype=float)
    for ga in geoms_a:
        for gb in geoms_b:
            try:
                d = float(mujoco.mj_geomDistance(model, data, int(ga), int(gb), float(distmax), fromto))
            except Exception:
                continue
            row = {
                "distance": d,
                "geom_a": geom_name(model, ga),
                "geom_b": geom_name(model, gb),
            }
            if best is None or d < best["distance"]:
                best = row
    return best


def current_object_support_distance(xml_path, object_body):
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    obj = collect_object_geoms(model, object_body)
    sup = collect_support_geoms(model)
    best = min_pair_distance(model, data, obj, sup)
    return best


def patch_body_z(xml_in, xml_out, object_body, dz):
    tree = ET.parse(str(xml_in))
    root = tree.getroot()

    target = None
    for b in root.iter("body"):
        if b.attrib.get("name") == object_body:
            target = b
            break

    if target is None:
        raise RuntimeError(f"cannot find body in XML: {object_body}")

    old_pos = [float(x) for x in target.attrib.get("pos", "0 0 0").split()]
    if len(old_pos) != 3:
        raise RuntimeError(f"bad body pos: {target.attrib.get('pos')}")

    new_pos = old_pos[:]
    new_pos[2] += float(dz)
    target.set("pos", f"{new_pos[0]:.12g} {new_pos[1]:.12g} {new_pos[2]:.12g}")

    xml_out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(xml_out), encoding="utf-8", xml_declaration=True)

    return old_pos, new_pos


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--object-body", default="grasp_object")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clearance", type=float, default=0.002)
    ap.add_argument("--max-iters", type=int, default=5)
    args = ap.parse_args()

    xml_in = resolve(args.model)
    xml_out = resolve(args.out)
    report_json = xml_out.with_suffix(".patch_report.json")
    report_txt = xml_out.with_suffix(".patch_report.txt")

    work_in = xml_in
    total_dz = 0.0
    steps = []

    before = current_object_support_distance(work_in, args.object_body)
    if before is None:
        raise RuntimeError("cannot compute object-support distance")

    current_dist = before["distance"]

    # 如果当前已经满足 clearance，也仍然复制一份输出 scene。
    if current_dist >= args.clearance:
        patch_body_z(work_in, xml_out, args.object_body, 0.0)
        after = current_object_support_distance(xml_out, args.object_body)
    else:
        tmp = xml_out
        for it in range(args.max_iters):
            need = args.clearance - current_dist
            dz = max(float(need), 0.0)

            old_pos, new_pos = patch_body_z(work_in, tmp, args.object_body, dz)
            total_dz += dz

            after_step = current_object_support_distance(tmp, args.object_body)
            steps.append({
                "iter": it,
                "input": rel(work_in),
                "output": rel(tmp),
                "old_distance": current_dist,
                "dz": dz,
                "old_pos": old_pos,
                "new_pos": new_pos,
                "new_distance": None if after_step is None else after_step["distance"],
                "pair": after_step,
            })

            if after_step is None:
                break

            current_dist = after_step["distance"]
            work_in = tmp

            if current_dist >= args.clearance:
                break

        after = current_object_support_distance(xml_out, args.object_body)

    report = {
        "format": "v4_15_object_support_clearance_patch_debug_v1",
        "model_in": rel(xml_in),
        "model_out": rel(xml_out),
        "object_body": args.object_body,
        "clearance": args.clearance,
        "before": before,
        "after": after,
        "total_dz": total_dz,
        "steps": steps,
        "ok": after is not None and after["distance"] >= args.clearance,
    }

    report_json.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    lines = []
    lines.append("========== V4.15 OBJECT-SUPPORT CLEARANCE PATCH ==========")
    lines.append(f"model_in : {rel(xml_in)}")
    lines.append(f"model_out: {rel(xml_out)}")
    lines.append(f"object   : {args.object_body}")
    lines.append(f"clearance: {args.clearance}")
    lines.append("")
    lines.append(f"before distance: {before}")
    lines.append(f"after distance : {after}")
    lines.append(f"total dz       : {total_dz}")
    lines.append(f"ok             : {report['ok']}")
    lines.append("==========================================================")
    txt = "\n".join(lines) + "\n"
    report_txt.write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
