#!/usr/bin/env python3
"""
脚本类型：
    debug / v4.15 / scene-patch / stiff-contact / dynamic-settle-check

用途：
    修复 V4.14 中“XML 初始物体没穿透，但动态 settle 后物体压进垫块”的问题。
    本脚本不换 sample、不改抓握姿态、不沿轴微调 grasp。
    只做场景物理接触参数修正：
        1. 加强 object_pedestal / grasp_object / 手部碰撞 geom 的 contact stiffness；
        2. 给 object-support 增加接触 margin，让接触提前发生；
        3. 编译 patched scene 后动态 settle，验证 object-support 是否仍然穿透。

输入：
    --model 原始 scene.xml
    --object-body 物体 body，例如 grasp_object
    --out 输出 stiff-contact scene.xml

输出：
    scene_v415_stiff_contact.xml
    scene_v415_stiff_contact.patch_report.txt/json

当前流程位置：
    site-target frame 已验证正确
        -> 本脚本修 scene 动态接触底座
        -> 后续再写 contact-aware close/lift runner

不负责：
    1. 不换 sample；
    2. 不修改手姿态；
    3. 不运行抓取；
    4. 不做 selector；
    5. 不把 force-lift 当成功。
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
        s = (geom_name(model, gid) + " " + body_name(model, model.geom_bodyid[gid])).lower()
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


def object_support_distance(model, data, object_body):
    obj = collect_object_geoms(model, object_body)
    sup = collect_support_geoms(model)
    return min_pair_distance(model, data, obj, sup)


def set_actuator_ctrl_to_qpos(model, data):
    """
    防止 settle 时机械臂乱掉。这里不是抓取控制，只是场景 settle 检查。
    """
    for aid in range(model.nu):
        jid = int(model.actuator_trnid[aid, 0])
        if jid < 0:
            continue
        qadr = int(model.jnt_qposadr[jid])
        if qadr < len(data.qpos):
            val = float(data.qpos[qadr])
            if int(model.actuator_ctrllimited[aid]):
                lo, hi = model.actuator_ctrlrange[aid]
                val = float(np.clip(val, lo, hi))
            data.ctrl[aid] = val


def settle_and_measure(xml_path, object_body, steps):
    model = mujoco.MjModel.from_xml_path(str(xml_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    before = object_support_distance(model, data, object_body)

    for _ in range(steps):
        set_actuator_ctrl_to_qpos(model, data)
        mujoco.mj_step(model, data)

    after = object_support_distance(model, data, object_body)

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
    obj_pos = data.xpos[bid].copy().tolist() if bid >= 0 else None

    return {
        "before": before,
        "after_settle": after,
        "object_pos_after_settle": obj_pos,
        "ncon_after_settle": int(data.ncon),
    }


def patch_xml(xml_in, xml_out, object_body, solref, solimp, object_margin, support_margin, hand_margin):
    tree = ET.parse(str(xml_in))
    root = tree.getroot()

    # timestep 稍微减小，减少动态穿透；不改抓握逻辑。
    opt = root.find("option")
    if opt is None:
        opt = ET.SubElement(root, "option")
    opt.set("timestep", "0.001")

    patched = []

    for g in root.iter("geom"):
        name = g.attrib.get("name", "")
        body = ""
        # xml 层不好直接拿 parent，这里按 geom 名称规则处理。
        low = name.lower()

        is_object = name.startswith(object_body) or "grasp_object" in low
        is_support = "object_pedestal" in low or "pedestal" in low or "support" in low
        is_hand_collision = (
            name.endswith("_auto_geom_00")
            or ("thumb" in low or "index" in low or "middle" in low or "ring" in low or "pinky" in low)
        )

        if is_object:
            g.set("solref", solref)
            g.set("solimp", solimp)
            g.set("margin", str(object_margin))
            g.set("gap", "0")
            patched.append({"geom": name, "role": "object", "margin": object_margin})

        elif is_support:
            g.set("solref", solref)
            g.set("solimp", solimp)
            g.set("margin", str(support_margin))
            g.set("gap", "0")
            patched.append({"geom": name, "role": "support", "margin": support_margin})

        elif is_hand_collision:
            # 只对已经开启碰撞的手部 geom 有效；visual/off geom 即使设置也不会产生碰撞。
            g.set("solref", solref)
            g.set("solimp", solimp)
            g.set("margin", str(hand_margin))
            g.set("gap", "0")
            patched.append({"geom": name, "role": "hand_like", "margin": hand_margin})

    xml_out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(xml_out), encoding="utf-8", xml_declaration=True)
    return patched


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--object-body", default="grasp_object")
    ap.add_argument("--out", required=True)
    ap.add_argument("--settle-steps", type=int, default=2000)
    ap.add_argument("--solref", default="0.001 1")
    ap.add_argument("--solimp", default="0.99 0.999 0.0005 0.5 2")
    ap.add_argument("--object-margin", type=float, default=0.004)
    ap.add_argument("--support-margin", type=float, default=0.004)
    ap.add_argument("--hand-margin", type=float, default=0.0015)
    ap.add_argument("--pass-min-after-settle", type=float, default=0.0005)
    args = ap.parse_args()

    xml_in = resolve(args.model)
    xml_out = resolve(args.out)

    before = settle_and_measure(xml_in, args.object_body, args.settle_steps)
    patched_geoms = patch_xml(
        xml_in, xml_out, args.object_body,
        args.solref, args.solimp,
        args.object_margin, args.support_margin, args.hand_margin,
    )
    after = settle_and_measure(xml_out, args.object_body, args.settle_steps)

    ok = (
        after["after_settle"] is not None
        and after["after_settle"]["distance"] is not None
        and after["after_settle"]["distance"] >= args.pass_min_after_settle
    )

    report = {
        "format": "v4_15_stiff_contact_scene_patch_debug_v1",
        "model_in": rel(xml_in),
        "model_out": rel(xml_out),
        "object_body": args.object_body,
        "settle_steps": args.settle_steps,
        "solref": args.solref,
        "solimp": args.solimp,
        "object_margin": args.object_margin,
        "support_margin": args.support_margin,
        "hand_margin": args.hand_margin,
        "pass_min_after_settle": args.pass_min_after_settle,
        "before_patch_settle": before,
        "after_patch_settle": after,
        "patched_geoms_count": len(patched_geoms),
        "patched_geoms_head": patched_geoms[:40],
        "ok": ok,
    }

    json_path = xml_out.with_suffix(".patch_report.json")
    txt_path = xml_out.with_suffix(".patch_report.txt")
    json_path.write_text(json.dumps(report, indent=2, ensure_ascii=False))

    lines = []
    lines.append("========== V4.15 STIFF CONTACT SCENE PATCH ==========")
    lines.append(f"model_in : {rel(xml_in)}")
    lines.append(f"model_out: {rel(xml_out)}")
    lines.append(f"object   : {args.object_body}")
    lines.append("")
    lines.append("---- before patch dynamic settle ----")
    lines.append(f"before initial distance : {before['before']}")
    lines.append(f"before after_settle     : {before['after_settle']}")
    lines.append(f"before obj_pos_settle   : {before['object_pos_after_settle']}")
    lines.append("")
    lines.append("---- after patch dynamic settle ----")
    lines.append(f"after initial distance  : {after['before']}")
    lines.append(f"after after_settle      : {after['after_settle']}")
    lines.append(f"after obj_pos_settle    : {after['object_pos_after_settle']}")
    lines.append("")
    lines.append(f"patched_geoms_count     : {len(patched_geoms)}")
    lines.append(f"ok                      : {ok}")
    lines.append("=====================================================")

    txt = "\n".join(lines) + "\n"
    txt_path.write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
