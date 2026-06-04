#!/usr/bin/env python3
"""
文件名：
    run_v4_12p4s_physics_sanity_check_debug.py

脚本类别：
    debug / diagnostic / physics-sanity-check

用途：
    本脚本用于 V4.12P4S 阶段。
    当前抓握调试中发现：细长 can 物体被手指轻微接触后会异常移动、倾倒、甚至像“弹飞”。
    这说明继续调手指闭合角度已经没有意义，必须先检查 MuJoCo 底层接触物理参数是否合理。

本脚本会检查：
    1. MuJoCo option 参数：timestep、solver、iterations、integrator、cone 等；
    2. object body 的质量、惯量、自由度；
    3. object/support/hand geom 的 friction、solref、solimp、condim、margin、gap；
    4. 物体在无手指接触时，单独放在支撑块上的稳定性；
    5. object-support 接触数量、接触穿透 dist、法向力、物体漂移量；
    6. 是否存在支撑接触太软、摩擦太低、condim 太低、质量异常等问题。

输入：
    --model
        MuJoCo XML 场景。
    --object-body
        被抓物体 body 名，例如 grasp_can。
    --support-tokens
        用于识别支撑块 geom 的名称关键词。
    --out
        输出 JSON。
    --report
        输出 TXT 报告。

输出：
    JSON + TXT 报告。

当前流程位置：
    P4P/P4R 继续调手指前，先做 P4S 物理体检。

本脚本不负责：
    1. 不控制机械臂抓取；
    2. 不修改 XML；
    3. 不做 lift；
    4. 不用抓取评分掩盖物理问题。
"""

from pathlib import Path
import argparse
import json
import numpy as np
import mujoco


PROJECT = Path.home() / "Projects/o7_mujoco_sim"


def resolve_path(p):
    p = Path(str(p)).expanduser()
    if not p.is_absolute():
        p = PROJECT / p
    return p


def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


def save_json(path, obj):
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_jsonable(obj), f, indent=2)


def obj_name(model, objtype, idx):
    name = mujoco.mj_id2name(model, objtype, int(idx))
    return name if name else f"unnamed_{idx}"


def body_id(model, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise RuntimeError(f"cannot find body: {name}")
    return bid


def geom_name(model, gid):
    return obj_name(model, mujoco.mjtObj.mjOBJ_GEOM, gid)


def body_name(model, bid):
    return obj_name(model, mujoco.mjtObj.mjOBJ_BODY, bid)


def geom_body_name(model, gid):
    return body_name(model, model.geom_bodyid[int(gid)])


def support_geom_ids(model, support_tokens):
    tokens = [x.strip().lower() for x in support_tokens.split(",") if x.strip()]
    out = set()
    for gid in range(model.ngeom):
        n = geom_name(model, gid).lower()
        b = geom_body_name(model, gid).lower()
        if any(t in n or t in b for t in tokens):
            out.add(gid)
    return out


def object_geom_ids(model, object_body):
    bid = body_id(model, object_body)
    out = set()
    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) == bid:
            out.add(gid)
    return out


def hand_geom_ids(model):
    keys = ["thumb", "index", "middle", "ring", "pinky", "hand", "palm"]
    out = set()
    for gid in range(model.ngeom):
        text = (geom_name(model, gid) + " " + geom_body_name(model, gid)).lower()
        if any(k in text for k in keys):
            out.add(gid)
    return out


def geom_info(model, gid):
    return {
        "gid": int(gid),
        "name": geom_name(model, gid),
        "body": geom_body_name(model, gid),
        "type": int(model.geom_type[gid]),
        "size": model.geom_size[gid].copy(),
        "pos": model.geom_pos[gid].copy(),
        "friction": model.geom_friction[gid].copy(),
        "solref": model.geom_solref[gid].copy(),
        "solimp": model.geom_solimp[gid].copy(),
        "condim": int(model.geom_condim[gid]),
        "priority": int(model.geom_priority[gid]),
        "margin": float(model.geom_margin[gid]),
        "gap": float(model.geom_gap[gid]),
        "rgba": model.geom_rgba[gid].copy(),
    }


def body_info(model, bid):
    return {
        "bid": int(bid),
        "name": body_name(model, bid),
        "mass": float(model.body_mass[bid]),
        "inertia": model.body_inertia[bid].copy(),
        "ipos": model.body_ipos[bid].copy(),
        "iquat": model.body_iquat[bid].copy(),
        "parent": body_name(model, model.body_parentid[bid]) if bid != 0 else "world",
    }


def joint_info_for_body(model, bid):
    out = []
    for jid in range(model.njnt):
        if int(model.jnt_bodyid[jid]) == int(bid):
            out.append({
                "jid": int(jid),
                "name": obj_name(model, mujoco.mjtObj.mjOBJ_JOINT, jid),
                "type": int(model.jnt_type[jid]),
                "qposadr": int(model.jnt_qposadr[jid]),
                "dofadr": int(model.jnt_dofadr[jid]),
                "limited": bool(model.jnt_limited[jid]),
                "range": model.jnt_range[jid].copy(),
                "damping": float(model.dof_damping[int(model.jnt_dofadr[jid])]) if int(model.jnt_dofadr[jid]) < model.nv else None,
                "armature": float(model.dof_armature[int(model.jnt_dofadr[jid])]) if int(model.jnt_dofadr[jid]) < model.nv else None,
            })
    return out


def contact_force6(model, data, ci):
    f = np.zeros(6)
    try:
        mujoco.mj_contactForce(model, data, int(ci), f)
    except Exception:
        pass
    return f


def collect_contacts(model, data, object_geoms, support_geoms, hand_geoms):
    obj_support = []
    obj_hand = []
    all_obj = []

    for i in range(data.ncon):
        c = data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)
        f6 = contact_force6(model, data, i)

        g1_obj = g1 in object_geoms
        g2_obj = g2 in object_geoms
        g1_sup = g1 in support_geoms
        g2_sup = g2 in support_geoms
        g1_hand = g1 in hand_geoms
        g2_hand = g2 in hand_geoms

        item = {
            "ci": int(i),
            "geom1": geom_name(model, g1),
            "geom2": geom_name(model, g2),
            "body1": geom_body_name(model, g1),
            "body2": geom_body_name(model, g2),
            "dist": float(c.dist),
            "pos": np.array(c.pos).copy(),
            "normal_force": float(max(0.0, f6[0])),
            "force6": f6.copy(),
        }

        if g1_obj or g2_obj:
            all_obj.append(item)
        if (g1_obj and g2_sup) or (g2_obj and g1_sup):
            obj_support.append(item)
        if (g1_obj and g2_hand) or (g2_obj and g1_hand):
            obj_hand.append(item)

    return {
        "all_object_contacts": all_obj,
        "object_support_contacts": obj_support,
        "object_hand_contacts": obj_hand,
    }


def run(args):
    model_path = resolve_path(args.model)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    obj_bid = body_id(model, args.object_body)
    object_geoms = object_geom_ids(model, args.object_body)
    support_geoms = support_geom_ids(model, args.support_tokens)
    hand_geoms = hand_geom_ids(model)

    print("\n========== V4.12P4S PHYSICS SANITY CHECK ==========")
    print("model:", model_path)
    print("object_body:", args.object_body)
    print("object_geoms:", [geom_name(model, g) for g in sorted(object_geoms)])
    print("support_geoms:", [geom_name(model, g) for g in sorted(support_geoms)])
    print("===================================================\n")

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    obj_pos0 = data.xpos[obj_bid].copy()

    samples = []
    for k in range(args.rest_steps + 1):
        mujoco.mj_step(model, data)
        if k % args.log_every == 0 or k == args.rest_steps:
            contacts = collect_contacts(model, data, object_geoms, support_geoms, hand_geoms)
            obj_pos = data.xpos[obj_bid].copy()
            obj_quat = data.xquat[obj_bid].copy()
            support_dists = [c["dist"] for c in contacts["object_support_contacts"]]
            support_forces = [c["normal_force"] for c in contacts["object_support_contacts"]]
            samples.append({
                "step": int(k),
                "time": float(data.time),
                "object_pos": obj_pos,
                "object_quat": obj_quat,
                "object_disp_from_initial": float(np.linalg.norm(obj_pos - obj_pos0)),
                "num_object_support_contacts": len(contacts["object_support_contacts"]),
                "num_object_hand_contacts": len(contacts["object_hand_contacts"]),
                "min_support_dist": min(support_dists) if support_dists else None,
                "max_support_normal_force": max(support_forces) if support_forces else 0.0,
                "sum_support_normal_force": float(sum(support_forces)),
                "contacts": contacts,
            })

    object_geom_infos = [geom_info(model, g) for g in sorted(object_geoms)]
    support_geom_infos = [geom_info(model, g) for g in sorted(support_geoms)]
    hand_geom_infos = [geom_info(model, g) for g in sorted(hand_geoms)]

    result = {
        "format": "v4_12p4s_physics_sanity_check_debug",
        "model": str(model_path),
        "option": {
            "timestep": float(model.opt.timestep),
            "iterations": int(model.opt.iterations),
            "ls_iterations": int(model.opt.ls_iterations),
            "tolerance": float(model.opt.tolerance),
            "solver": int(model.opt.solver),
            "integrator": int(model.opt.integrator),
            "cone": int(model.opt.cone),
            "gravity": model.opt.gravity.copy(),
        },
        "object_body": body_info(model, obj_bid),
        "object_joints": joint_info_for_body(model, obj_bid),
        "object_geoms": object_geom_infos,
        "support_geoms": support_geom_infos,
        "hand_geom_count": len(hand_geom_infos),
        "hand_geom_sample": hand_geom_infos[:80],
        "rest_samples": samples,
    }

    save_json(args.out, result)

    report_path = resolve_path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)

    with open(report_path, "w") as f:
        f.write("========== V4.12P4S PHYSICS SANITY CHECK REPORT ==========\n\n")
        f.write(f"model: {model_path}\n")
        f.write(f"object_body: {args.object_body}\n\n")

        f.write("---- OPTION ----\n")
        for k, v in result["option"].items():
            f.write(f"{k}: {v}\n")

        f.write("\n---- OBJECT BODY ----\n")
        f.write(json.dumps(to_jsonable(result["object_body"]), indent=2) + "\n")
        f.write("\n---- OBJECT JOINTS ----\n")
        f.write(json.dumps(to_jsonable(result["object_joints"]), indent=2) + "\n")

        f.write("\n---- OBJECT GEOMS ----\n")
        for g in object_geom_infos:
            f.write(json.dumps(to_jsonable(g), indent=2) + "\n")

        f.write("\n---- SUPPORT GEOMS ----\n")
        for g in support_geom_infos:
            f.write(json.dumps(to_jsonable(g), indent=2) + "\n")

        f.write("\n---- REST STABILITY SAMPLES ----\n")
        for s in samples:
            f.write(
                f"step={s['step']} time={s['time']:.4f} "
                f"disp={s['object_disp_from_initial']:.6f} "
                f"support_contacts={s['num_object_support_contacts']} "
                f"hand_contacts={s['num_object_hand_contacts']} "
                f"min_support_dist={s['min_support_dist']} "
                f"sum_support_force={s['sum_support_normal_force']:.6f}\n"
            )
            for c in s["contacts"]["object_support_contacts"][:10]:
                f.write(
                    f"    SUPPORT {c['geom1']} <-> {c['geom2']} "
                    f"dist={c['dist']:.8f} fn={c['normal_force']:.6f}\n"
                )
            for c in s["contacts"]["object_hand_contacts"][:10]:
                f.write(
                    f"    HAND {c['geom1']} <-> {c['geom2']} "
                    f"dist={c['dist']:.8f} fn={c['normal_force']:.6f}\n"
                )

    print("[SAVED]", resolve_path(args.out))
    print("[REPORT]", report_path)
    print("\nQuick summary:")
    print("object mass:", model.body_mass[obj_bid])
    print("object inertia:", model.body_inertia[obj_bid])
    for g in object_geom_infos:
        print("[object geom]", g["name"], "friction=", g["friction"], "solref=", g["solref"], "solimp=", g["solimp"], "condim=", g["condim"])
    for g in support_geom_infos:
        print("[support geom]", g["name"], "friction=", g["friction"], "solref=", g["solref"], "solimp=", g["solimp"], "condim=", g["condim"])
    print("====================================================\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--object-body", default="grasp_can")
    ap.add_argument("--support-tokens", default="object_pedestal,pedestal,support,table")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)
    ap.add_argument("--rest-steps", type=int, default=800)
    ap.add_argument("--log-every", type=int, default=80)
    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
