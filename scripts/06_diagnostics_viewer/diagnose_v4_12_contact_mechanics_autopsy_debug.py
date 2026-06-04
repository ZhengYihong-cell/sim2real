#!/usr/bin/env python3
"""
文件名：
    diagnose_v4_12_contact_mechanics_autopsy_debug.py

脚本类别：
    debug / diagnostic / contact-mechanics-autopsy

用途：
    本脚本用于 V4.12 阶段的“接触力学尸检”。
    当前问题不是继续评分，而是必须查清楚为什么抓握过程中物体会被推向外、向下或滑入手内部。
    本脚本会复现 open -> preshape -> move_to_grasp -> close 的过程，并逐帧记录：
        1. 手指控制量 side_open / close_target 是否语义错误；
        2. 哪个手部 link 最先接触物体；
        3. 接触来自 thumb / index / middle / ring / pinky / palm 中哪一类；
        4. 接触 geom 是 distal 指腹、proximal 近端，还是 hand/palm 内侧；
        5. 物体开始明显移动时的接触组合；
        6. 接触力大小、接触点、物体位移方向；
        7. 是否出现“大拇指提前闭合成推板”的现象。

输入：
    --model
        MuJoCo XML 场景，建议使用 hard_support 版本。
    --candidate
        当前要尸检的 candidate JSON。
    --p3-json
        当前用于执行的 P3 JSON。
    --best-config
        当前传给 runner 的 best_config JSON。
    --object-body
        被抓物体 body 名，例如 grasp_can。
    --out-prefix
        输出文件前缀。

输出：
    <out-prefix>.json
        完整逐帧诊断数据。
    <out-prefix>.txt
        人可读诊断报告。

当前流程位置：
    P4H2 / P4J 出现异常抓握后
        -> 本脚本复现 close 过程
        -> 定位接触和控制语义根因

本脚本不负责：
    1. 不选择 best；
    2. 不修改候选；
    3. 不做 lift；
    4. 不判断抓握成功；
    5. 不继续调评分。
"""

from pathlib import Path
import argparse
import importlib.util
import json
import math
import time
import numpy as np
import mujoco

try:
    import mujoco.viewer
except Exception:
    mujoco.viewer = None


PROJECT = Path.home() / "Projects/o7_mujoco_sim"
P4F_PATH = PROJECT / "scripts/05_execution_runner/run_v4_12p4f_target_close_debug.py"

ARM_JOINTS = [
    "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
    "fr3_joint5", "fr3_joint6", "fr3_joint7",
]

HAND_JOINTS = [
    "thumb_cmc_roll",
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]

NON_THUMB = ["index", "middle", "ring", "pinky"]


def load_p4f():
    spec = importlib.util.spec_from_file_location("p4f", str(P4F_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p4f = load_p4f()


def resolve_path(p):
    p = Path(str(p)).expanduser()
    if not p.is_absolute():
        p = PROJECT / p
    return p


def load_json(p):
    with open(resolve_path(p), "r") as f:
        return json.load(f)


def save_json(p, obj):
    p = resolve_path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(to_jsonable(obj), f, indent=2)


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


def interp_dict(a, b, alpha, keys):
    out = {}
    for k in keys:
        av = float(a.get(k, 0.0))
        bv = float(b.get(k, av))
        out[k] = av + alpha * (bv - av)
    return out


def geom_name(model, gid):
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid))
    return n if n else f"geom_{gid}"


def body_name(model, bid):
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(bid))
    return n if n else f"body_{bid}"


def geom_body_name(model, gid):
    return body_name(model, int(model.geom_bodyid[int(gid)]))


def object_center(model, data, object_body):
    bid = p4f.body_id(model, object_body)
    return data.xpos[bid].copy()


def contact_force6(model, data, contact_index):
    f = np.zeros(6, dtype=float)
    try:
        mujoco.mj_contactForce(model, data, int(contact_index), f)
    except Exception:
        pass
    return f


def classify_hand_part(geom, body):
    text = f"{geom} {body}".lower()
    if "thumb" in text:
        group = "thumb"
    elif "index" in text:
        group = "index"
    elif "middle" in text:
        group = "middle"
    elif "ring" in text:
        group = "ring"
    elif "pinky" in text:
        group = "pinky"
    elif "palm" in text or "hand" in text:
        group = "palm"
    else:
        group = "unknown"

    if "distal" in text:
        segment = "distal"
    elif "middle" in text:
        segment = "middle"
    elif "proximal" in text:
        segment = "proximal"
    elif "palm" in text or "hand" in text:
        segment = "palm_or_hand"
    else:
        segment = "unknown"

    return group, segment


def collect_contacts(model, data, sets, object_body):
    object_geoms = sets["object_geoms"]
    hand_geoms = sets["hand_geoms"]
    support_geoms = sets["support_geoms"]
    fr3_geoms = sets["fr3_geoms"]

    c = object_center(model, data, object_body)
    contacts = []
    groups = {}
    segments = {}
    object_support = []
    hand_support = []
    fr3_object = []

    for i in range(data.ncon):
        con = data.contact[i]
        g1 = int(con.geom1)
        g2 = int(con.geom2)

        g1_obj = g1 in object_geoms
        g2_obj = g2 in object_geoms
        g1_hand = g1 in hand_geoms
        g2_hand = g2 in hand_geoms
        g1_sup = g1 in support_geoms
        g2_sup = g2 in support_geoms
        g1_fr3 = g1 in fr3_geoms
        g2_fr3 = g2 in fr3_geoms

        f6 = contact_force6(model, data, i)

        item = {
            "index": int(i),
            "geom1": geom_name(model, g1),
            "body1": geom_body_name(model, g1),
            "geom2": geom_name(model, g2),
            "body2": geom_body_name(model, g2),
            "dist": float(con.dist),
            "pos": np.array(con.pos, dtype=float).copy(),
            "normal_force": float(max(0.0, f6[0])),
            "force6": f6.copy(),
            "rel_pos_from_object_center": np.array(con.pos, dtype=float).copy() - c,
            "type": "other",
        }

        if (g1_hand and g2_obj) or (g2_hand and g1_obj):
            hg = g1 if g1_hand else g2
            hgeom = geom_name(model, hg)
            hbody = geom_body_name(model, hg)
            group, segment = classify_hand_part(hgeom, hbody)
            item["type"] = "hand_object"
            item["hand_geom"] = hgeom
            item["hand_body"] = hbody
            item["hand_group"] = group
            item["hand_segment"] = segment
            groups[group] = groups.get(group, 0) + 1
            segments[f"{group}:{segment}"] = segments.get(f"{group}:{segment}", 0) + 1

        elif (g1_obj and g2_sup) or (g2_obj and g1_sup):
            item["type"] = "object_support"
            object_support.append(item)

        elif (g1_hand and g2_sup) or (g2_hand and g1_sup):
            item["type"] = "hand_support"
            hand_support.append(item)

        elif (g1_fr3 and g2_obj) or (g2_fr3 and g1_obj):
            item["type"] = "fr3_object"
            fr3_object.append(item)

        contacts.append(item)

    hand_object = [x for x in contacts if x["type"] == "hand_object"]

    return {
        "contacts": contacts,
        "hand_object": hand_object,
        "object_support": object_support,
        "hand_support": hand_support,
        "fr3_object": fr3_object,
        "groups": groups,
        "segments": segments,
        "min_object_support_dist": min([x["dist"] for x in object_support], default=None),
    }


def summarize_ctrl(candidate_ctrl, best_ctrl, side_open, close_target):
    rows = []
    for j in HAND_JOINTS:
        cc = candidate_ctrl.get(j, None)
        bc = best_ctrl.get(j, None)
        so = side_open.get(j, None)
        ct = close_target.get(j, None)
        rows.append({
            "joint": j,
            "candidate": cc,
            "best_config": bc,
            "side_open": so,
            "close_target": ct,
            "close_minus_side": None if so is None or ct is None else float(ct - so),
        })
    return rows


def detect_ctrl_bug(ctrl_rows):
    warnings = []
    by = {r["joint"]: r for r in ctrl_rows}

    thumb = by.get("thumb_cmc_pitch")
    if thumb:
        cand = thumb["candidate"]
        best = thumb["best_config"]
        side = thumb["side_open"]
        close = thumb["close_target"]

        if best is not None and cand is not None and best > cand + 0.15:
            warnings.append(
                f"thumb_cmc_pitch best_config={best:.4f} 明显大于 candidate={cand:.4f}，"
                f"best_config 可能已经是 close_target，却又被当成 preshape/side_open 使用。"
            )

        if side is not None and close is not None and side > 0.35 and close > 0.55:
            warnings.append(
                f"thumb_cmc_pitch side_open={side:.4f}, close_target={close:.4f}，"
                f"大拇指可能在接近阶段已经过度闭合，后续继续闭合会变成推板。"
            )

        if side is not None and cand is not None and side > cand + 0.15:
            warnings.append(
                f"thumb_cmc_pitch side_open={side:.4f} 明显大于 candidate={cand:.4f}，"
                f"说明 side_open 不是打开姿态。"
            )

    return warnings


def row_summary(row):
    return (
        f"{row['phase']} step={row['step']} alpha={row['alpha']:.3f} "
        f"disp={row['object_disp']:.5f} "
        f"obj_delta={np.array(row['object_delta']).round(5).tolist()} "
        f"groups={row['contact']['groups']} "
        f"segments={row['contact']['segments']} "
        f"support={row['contact']['min_object_support_dist']}"
    )


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--p3-json", required=True)
    ap.add_argument("--best-config", required=True)
    ap.add_argument("--which", default="best_available", choices=["best_available", "best_pass"])
    ap.add_argument("--object-body", default="grasp_can")
    ap.add_argument("--out-prefix", required=True)
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--move-steps", type=int, default=100)
    ap.add_argument("--thumb-preshape-steps", type=int, default=100)
    ap.add_argument("--close-duration", type=float, default=1.8)
    ap.add_argument("--finger-close-scale", type=float, default=0.85)
    ap.add_argument("--thumb-pitch-from-finger-gain", type=float, default=0.35)
    ap.add_argument("--preshape-fingers-from-best", action="store_true")
    ap.add_argument("--direct-hand-qpos", action="store_true")
    ap.add_argument("--log-dt", type=float, default=0.04)
    ap.add_argument("--frame-sleep", type=float, default=0.001)
    args = ap.parse_args()

    model_path = resolve_path(args.model)
    candidate_path = resolve_path(args.candidate)
    p3_path = resolve_path(args.p3_json)
    best_config_path = resolve_path(args.best_config)

    candidate = load_json(candidate_path)
    p3 = load_json(p3_path)
    best_config = load_json(best_config_path)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    plan = p4f.selected_plan(p3, args.which)
    q_pre = plan["q_pre"]
    q_grasp = plan["q_grasp"]

    candidate_ctrl, candidate_ctrl_source = p4f.extract_candidate_ctrl(candidate, model)
    best_ctrl, best_ctrl_source = p4f.extract_best_config_ctrl(best_config, model)

    open_ctrl = p4f.make_open_ctrl(model)
    side_open_ctrl = p4f.make_side_open_ctrl(model, open_ctrl, candidate_ctrl, best_ctrl, args)
    close_target = p4f.make_close_target(model, side_open_ctrl, candidate_ctrl, args)

    ctrl_rows = summarize_ctrl(candidate_ctrl, best_ctrl, side_open_ctrl, close_target)
    ctrl_warnings = detect_ctrl_bug(ctrl_rows)

    sets = p4f.build_geom_sets(model, args.object_body)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    q_current = {j: p4f.get_joint_qpos(model, data, j) or 0.0 for j in ARM_JOINTS}
    obj0 = p4f.object_pos(model, data, args.object_body)

    logs = []
    first_hand_contact = None
    first_thumb = None
    first_non_thumb = None
    first_large_motion = None
    first_proximal_or_palm = None

    log_every = max(1, int(args.log_dt / model.opt.timestep))

    def step_and_log(phase, step, total_steps, alpha, arm_q, hand_ctrl, force_log=False):
        nonlocal first_hand_contact, first_thumb, first_non_thumb, first_large_motion, first_proximal_or_palm

        p4f.set_arm_qpos_and_ctrl(model, data, arm_q)
        p4f.set_hand_ctrl(model, data, hand_ctrl, args.direct_hand_qpos)
        mujoco.mj_step(model, data)

        obj = p4f.object_pos(model, data, args.object_body)
        delta = obj - obj0
        contact = collect_contacts(model, data, sets, args.object_body)

        row = {
            "phase": phase,
            "step": int(step),
            "total_steps": int(total_steps),
            "alpha": float(alpha),
            "object_pos": obj.copy(),
            "object_delta": delta.copy(),
            "object_disp": float(np.linalg.norm(delta)),
            "hand_qpos": p4f.hand_qpos(model, data),
            "cmd_ctrl": dict(hand_ctrl),
            "contact": contact,
        }

        if contact["hand_object"] and first_hand_contact is None:
            first_hand_contact = row

        if "thumb" in contact["groups"] and first_thumb is None:
            first_thumb = row

        if any(g in contact["groups"] for g in NON_THUMB) and first_non_thumb is None:
            first_non_thumb = row

        if row["object_disp"] > 0.010 and first_large_motion is None:
            first_large_motion = row

        for item in contact["hand_object"]:
            seg = item.get("hand_segment", "")
            grp = item.get("hand_group", "")
            if seg in ["proximal", "palm_or_hand"] or grp == "palm":
                if first_proximal_or_palm is None:
                    first_proximal_or_palm = row

        if force_log or step % log_every == 0 or step == total_steps:
            logs.append(row)
            print(row_summary(row))

        return row

    def sequence(viewer=None):
        print("\n[PHASE] move_to_pre_open")
        for k in range(args.move_steps + 1):
            a = k / max(1, args.move_steps)
            arm_q = interp_dict(q_current, q_pre, a, ARM_JOINTS)
            step_and_log("move_to_pre_open", k, args.move_steps, a, arm_q, open_ctrl)
            if viewer is not None:
                viewer.sync()
                time.sleep(args.frame_sleep)

        print("\n[PHASE] thumb_preshape")
        for k in range(args.thumb_preshape_steps + 1):
            a = k / max(1, args.thumb_preshape_steps)
            h = interp_dict(open_ctrl, side_open_ctrl, a, HAND_JOINTS)
            step_and_log("thumb_preshape", k, args.thumb_preshape_steps, a, q_pre, h)
            if viewer is not None:
                viewer.sync()
                time.sleep(args.frame_sleep)

        print("\n[PHASE] move_to_grasp_side_open")
        for k in range(args.move_steps + 1):
            a = k / max(1, args.move_steps)
            arm_q = interp_dict(q_pre, q_grasp, a, ARM_JOINTS)
            step_and_log("move_to_grasp_side_open", k, args.move_steps, a, arm_q, side_open_ctrl)
            if viewer is not None:
                viewer.sync()
                time.sleep(args.frame_sleep)

        close_steps = max(1, int(args.close_duration / model.opt.timestep))
        print("\n[PHASE] close_autopsy")
        for k in range(close_steps + 1):
            a = k / max(1, close_steps)
            h = interp_dict(side_open_ctrl, close_target, a, HAND_JOINTS)
            row = step_and_log("close_autopsy", k, close_steps, a, q_grasp, h)
            if viewer is not None:
                viewer.sync()
                time.sleep(args.frame_sleep)

            # 位移已经超过 5 cm 后继续跑意义不大，保留现场。
            if row["object_disp"] > 0.050:
                print("[STOP AUTOPSY] object displacement > 0.05m, stop close replay.")
                break

    print("\n========== V4.12 CONTACT MECHANICS AUTOPSY ==========")
    print("model                :", model_path)
    print("candidate            :", candidate_path)
    print("p3_json              :", p3_path)
    print("best_config          :", best_config_path)
    print("candidate_ctrl_source:", candidate_ctrl_source)
    print("best_ctrl_source     :", best_ctrl_source)
    print("candidate_ctrl       :", candidate_ctrl)
    print("best_ctrl            :", best_ctrl)
    print("side_open_ctrl       :", side_open_ctrl)
    print("close_target         :", close_target)
    print("ctrl_warnings        :", ctrl_warnings)
    print("=====================================================\n")

    if args.viewer:
        if mujoco.viewer is None:
            raise RuntimeError("mujoco.viewer unavailable")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            sequence(viewer)
            print("[VIEWER] 诊断播放完成，关闭窗口退出。")
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.05)
    else:
        sequence(None)

    diagnosis = {
        "format": "v4_12_contact_mechanics_autopsy_debug",
        "model": str(model_path),
        "candidate": str(candidate_path),
        "p3_json": str(p3_path),
        "best_config": str(best_config_path),
        "which": args.which,
        "object_body": args.object_body,
        "candidate_ctrl_source": candidate_ctrl_source,
        "best_ctrl_source": best_ctrl_source,
        "candidate_ctrl": candidate_ctrl,
        "best_ctrl": best_ctrl,
        "side_open_ctrl": side_open_ctrl,
        "close_target": close_target,
        "ctrl_rows": ctrl_rows,
        "ctrl_warnings": ctrl_warnings,
        "first_hand_contact": first_hand_contact,
        "first_thumb": first_thumb,
        "first_non_thumb": first_non_thumb,
        "first_large_motion": first_large_motion,
        "first_proximal_or_palm": first_proximal_or_palm,
        "logs": logs,
    }

    out_prefix = resolve_path(args.out_prefix)
    out_prefix.parent.mkdir(parents=True, exist_ok=True)
    save_json(str(out_prefix) + ".json", diagnosis)

    txt_path = Path(str(out_prefix) + ".txt")
    with open(txt_path, "w") as f:
        f.write("========== V4.12 CONTACT MECHANICS AUTOPSY REPORT ==========\n\n")
        f.write(f"model: {model_path}\n")
        f.write(f"candidate: {candidate_path}\n")
        f.write(f"p3_json: {p3_path}\n")
        f.write(f"best_config: {best_config_path}\n\n")

        f.write("---- CTRL PIPELINE ----\n")
        for r in ctrl_rows:
            f.write(
                f"{r['joint']}: candidate={r['candidate']} "
                f"best_config={r['best_config']} "
                f"side_open={r['side_open']} "
                f"close_target={r['close_target']} "
                f"close_minus_side={r['close_minus_side']}\n"
            )
        f.write("\n")

        f.write("---- CTRL WARNINGS ----\n")
        if ctrl_warnings:
            for w in ctrl_warnings:
                f.write(f"[WARNING] {w}\n")
        else:
            f.write("[OK] no obvious ctrl semantic warning detected\n")
        f.write("\n")

        f.write("---- FIRST EVENTS ----\n")
        for name, row in [
            ("first_hand_contact", first_hand_contact),
            ("first_thumb", first_thumb),
            ("first_non_thumb", first_non_thumb),
            ("first_large_motion_gt_1cm", first_large_motion),
            ("first_proximal_or_palm_contact", first_proximal_or_palm),
        ]:
            f.write(f"\n{name}:\n")
            if row is None:
                f.write("  None\n")
            else:
                f.write("  " + row_summary(row) + "\n")
                for c in row["contact"]["hand_object"]:
                    f.write(
                        f"    hand_object: group={c.get('hand_group')} "
                        f"segment={c.get('hand_segment')} "
                        f"hand_geom={c.get('hand_geom')} "
                        f"dist={c['dist']:.6f} "
                        f"normal_force={c['normal_force']:.6f} "
                        f"rel={np.array(c['rel_pos_from_object_center']).round(5).tolist()}\n"
                    )
        f.write("\n")

        f.write("---- KEY LOGS ----\n")
        for row in logs:
            if row["contact"]["hand_object"] or row["object_disp"] > 0.005:
                f.write(row_summary(row) + "\n")
                for c in row["contact"]["hand_object"]:
                    f.write(
                        f"    {c.get('hand_group')}:{c.get('hand_segment')} "
                        f"{c.get('hand_geom')} "
                        f"dist={c['dist']:.6f} "
                        f"fN={c['normal_force']:.6f} "
                        f"rel={np.array(c['rel_pos_from_object_center']).round(5).tolist()}\n"
                    )

    print("\n========== AUTOPSY SAVED ==========")
    print("json:", str(out_prefix) + ".json")
    print("txt :", txt_path)
    print("===================================\n")


if __name__ == "__main__":
    main()
