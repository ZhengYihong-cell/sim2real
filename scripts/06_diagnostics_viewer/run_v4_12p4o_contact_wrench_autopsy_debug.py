#!/usr/bin/env python3
"""
文件名：
    run_v4_12p4o_contact_wrench_autopsy_debug.py

脚本类别：
    debug / diagnostic / contact-wrench-autopsy / adaptive-pose-suggestion

用途：
    本脚本用于 V4.12P4O 阶段。
    当前已经证明单纯手指闭合不足以稳定侧握细长物体。
    本脚本不继续调手指角度，而是复现慢速闭合过程，并计算每一帧手-物体接触对物体造成的：
        1. 单个接触力 f_i；
        2. 单个接触力矩 tau_i = (p_i - p_obj) x f_i；
        3. 合力 F_net；
        4. 合力矩 Tau_net；
        5. thumb 与非拇指接触力是否形成对抗；
        6. 当前 handbase 应该往哪个局部方向微调。

核心思想：
    如果合力/合力矩不能保证抓握，就不能继续闭合。
    应该先调整 handbase / wrist 位姿，使合力趋向稳定抓握，再小幅 squeeze。

输入：
    --model
        MuJoCo XML 场景，建议使用 hard_support 版本或 P4N 找到的相对较好 offset 场景。
    --candidate
        当前 candidate JSON。
    --p3-json
        当前 P3 JSON。
    --best-config
        已修正 ctrl semantics 的 best_config。
    --object-body
        被抓物体 body 名，例如 grasp_can。
    --out
        输出 JSON。
    --report
        输出可读 txt 报告。

输出：
    1. JSON：逐帧接触力、合力、合力矩、建议 hand-local delta。
    2. TXT：关键事件摘要和修正建议。
    3. 可选 viewer：同步观察画面。

当前流程位置：
    P4M3 live-contact gate 证明当前闭合不能稳定夹持
        -> P4O 接触 wrench 尸检
        -> 根据建议生成 hand-local 小位姿修正
        -> 再回到 P2/P3/P4M3 或后续 P4P 自适应 close

本脚本不负责：
    1. 不重新做 IK；
    2. 不直接移动 handbase；
    3. 不直接 lift；
    4. 不把接触历史当作抓住；
    5. 不用评分掩盖接触力/力矩问题。
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

NON_THUMB_GROUPS = ["index", "middle", "ring", "pinky"]


def import_py(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p4f = import_py(P4F_PATH, "p4f")


def resolve_path(p):
    p = Path(str(p)).expanduser()
    if not p.is_absolute():
        p = PROJECT / p
    return p


def save_json(path, obj):
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_jsonable(obj), f, indent=2)


def load_json(path):
    with open(resolve_path(path), "r") as f:
        return json.load(f)


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
        out[k] = av + float(alpha) * (bv - av)
    return out


def geom_name(model, gid):
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid))
    return n if n else f"geom_{gid}"


def body_name(model, bid):
    n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(bid))
    return n if n else f"body_{bid}"


def geom_body_name(model, gid):
    return body_name(model, int(model.geom_bodyid[int(gid)]))


def classify_hand_group(model, gid):
    text = f"{geom_name(model, gid)} {geom_body_name(model, gid)}".lower()
    if "thumb" in text:
        return "thumb"
    if "index" in text:
        return "index"
    if "middle" in text:
        return "middle"
    if "ring" in text:
        return "ring"
    if "pinky" in text:
        return "pinky"
    if "palm" in text or "hand" in text:
        return "palm"
    return "unknown"


def classify_segment(model, gid):
    text = f"{geom_name(model, gid)} {geom_body_name(model, gid)}".lower()
    if "distal" in text:
        return "distal"
    if "middle" in text:
        return "middle"
    if "proximal" in text:
        return "proximal"
    if "palm" in text or "hand" in text:
        return "palm_or_hand"
    return "unknown"


def object_body_id(model, object_body):
    bid = p4f.body_id(model, object_body)
    if bid < 0:
        raise RuntimeError(f"cannot find object body: {object_body}")
    return bid


def object_state(model, data, object_body):
    bid = object_body_id(model, object_body)
    return {
        "pos": data.xpos[bid].copy(),
        "xmat": data.xmat[bid].reshape(3, 3).copy(),
        "velp": data.cvel[bid][3:6].copy(),
        "velr": data.cvel[bid][0:3].copy(),
    }


def body_transform(model, data, body_name_str):
    bid = p4f.body_id(model, body_name_str)
    if bid < 0:
        raise RuntimeError(f"cannot find body: {body_name_str}")
    T = np.eye(4)
    T[:3, :3] = data.xmat[bid].reshape(3, 3)
    T[:3, 3] = data.xpos[bid]
    return T


def contact_force6(model, data, ci):
    f = np.zeros(6, dtype=float)
    try:
        mujoco.mj_contactForce(model, data, int(ci), f)
    except Exception:
        pass
    return f


def contact_force_world_candidates(con, f6):
    """
    MuJoCo 的 contact.frame 是接触坐标系。
    不同版本/理解中 frame 的行列约定容易混淆。
    这里同时返回：
        normal_force_world：只用法向力和 frame[0:3]，最稳定；
        full_force_world_rows：把 frame 当作 row axes；
        full_force_world_cols：把 frame 当作 column axes。
    实际建议主要基于法向力 + 接触点几何方向。
    """
    frame = np.array(con.frame, dtype=float).reshape(3, 3)
    normal_axis = frame[0].copy()
    normal_force_world = normal_axis * float(f6[0])
    full_rows = frame.T @ np.asarray(f6[:3], dtype=float)
    full_cols = frame @ np.asarray(f6[:3], dtype=float)
    return normal_force_world, full_rows, full_cols


def orient_force_toward_object_center(raw_force, contact_pos, obj_center):
    """
    接触力方向符号在 MuJoCo 接触对中容易混淆。
    对于“手推物体”的诊断，我们将力方向调整为大致指向物体中心的一侧。
    这样可以得到“该接触对物体施加的压入方向”的稳定近似。
    """
    f = np.asarray(raw_force, dtype=float).copy()
    inward = np.asarray(obj_center, dtype=float) - np.asarray(contact_pos, dtype=float)
    if np.linalg.norm(f) < 1e-12 or np.linalg.norm(inward) < 1e-12:
        return f

    if float(np.dot(f, inward)) < 0.0:
        f = -f
    return f


def collect_wrench(model, data, sets, object_body, target_body):
    object_geoms = sets["object_geoms"]
    hand_geoms = sets["hand_geoms"]
    support_geoms = sets["support_geoms"]

    obj = object_state(model, data, object_body)
    obj_center = obj["pos"]
    T_world_hand = body_transform(model, data, target_body)
    R_world_hand = T_world_hand[:3, :3]

    hand_object_contacts = []
    object_support_contacts = []
    groups = {}
    segments = {}

    F_net = np.zeros(3)
    Tau_net = np.zeros(3)

    F_by_group = {}
    Tau_by_group = {}

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

        f6 = contact_force6(model, data, i)
        fn = float(max(0.0, f6[0]))

        if (g1_obj and g2_sup) or (g2_obj and g1_sup):
            object_support_contacts.append({
                "contact_index": int(i),
                "geom1": geom_name(model, g1),
                "geom2": geom_name(model, g2),
                "dist": float(con.dist),
                "normal_force": fn,
                "pos": np.array(con.pos, dtype=float).copy(),
            })

        if not ((g1_hand and g2_obj) or (g2_hand and g1_obj)):
            continue

        hg = g1 if g1_hand else g2
        group = classify_hand_group(model, hg)
        segment = classify_segment(model, hg)

        normal_world, full_rows, full_cols = contact_force_world_candidates(con, f6)

        # 主要用于诊断的近似 object force：
        # 取法向力，并把方向调整为指向物体中心，表示该手部接触对物体的压入力方向。
        f_obj_world = orient_force_toward_object_center(normal_world, con.pos, obj_center)

        r = np.asarray(con.pos, dtype=float) - obj_center
        tau = np.cross(r, f_obj_world)

        F_net += f_obj_world
        Tau_net += tau

        F_by_group[group] = F_by_group.get(group, np.zeros(3)) + f_obj_world
        Tau_by_group[group] = Tau_by_group.get(group, np.zeros(3)) + tau

        groups[group] = groups.get(group, 0) + 1
        segments[f"{group}:{segment}"] = segments.get(f"{group}:{segment}", 0) + 1

        hand_object_contacts.append({
            "contact_index": int(i),
            "group": group,
            "segment": segment,
            "hand_geom": geom_name(model, hg),
            "hand_body": geom_body_name(model, hg),
            "dist": float(con.dist),
            "normal_force": fn,
            "pos": np.array(con.pos, dtype=float).copy(),
            "r_from_obj": r,
            "f_obj_world_normal_oriented": f_obj_world,
            "f_world_normal_raw": normal_world,
            "f_world_full_rows": full_rows,
            "f_world_full_cols": full_cols,
            "tau_obj_world": tau,
        })

    F_thumb = F_by_group.get("thumb", np.zeros(3))
    F_non = np.zeros(3)
    for g in NON_THUMB_GROUPS:
        F_non += F_by_group.get(g, np.zeros(3))

    thumb_norm = float(np.linalg.norm(F_thumb))
    non_norm = float(np.linalg.norm(F_non))
    if thumb_norm > 1e-9 and non_norm > 1e-9:
        opposition_cos = float(np.dot(F_thumb, F_non) / (thumb_norm * non_norm))
    else:
        opposition_cos = None

    # hand-local 表达：用于给出位姿微调方向。
    F_hand = R_world_hand.T @ F_net
    Tau_hand = R_world_hand.T @ Tau_net

    min_support_dist = min([x["dist"] for x in object_support_contacts], default=None)

    return {
        "groups": groups,
        "segments": segments,
        "hand_object_contacts": hand_object_contacts,
        "object_support_contacts": object_support_contacts,
        "min_object_support_dist": min_support_dist,
        "F_net_world": F_net,
        "Tau_net_world": Tau_net,
        "F_by_group_world": F_by_group,
        "Tau_by_group_world": Tau_by_group,
        "F_thumb_world": F_thumb,
        "F_non_thumb_world": F_non,
        "thumb_non_thumb_opposition_cos": opposition_cos,
        "F_net_hand": F_hand,
        "Tau_net_hand": Tau_hand,
    }


def hand_qpos(model, data):
    return p4f.hand_qpos(model, data)


def make_correction_suggestion(wrench, obj_delta, args):
    """
    这个建议不是最终控制律，而是诊断方向。
    原则：
        F_net_hand 指向哪里，说明手正在把物体往哪里推；
        handbase 应该朝相反方向小幅移动，让接触力更对称；
        Tau_net_hand 哪个轴大，说明物体有绕该轴翻倒趋势，需要 wrist roll/yaw/pitch 微调。
    """
    Fh = np.asarray(wrench["F_net_hand"], dtype=float)
    Th = np.asarray(wrench["Tau_net_hand"], dtype=float)
    Fnorm = float(np.linalg.norm(Fh))
    Tnorm = float(np.linalg.norm(Th))

    if Fnorm > 1e-9:
        # handbase 朝反合力方向微调，幅值限制在 max_delta。
        dpos = -args.pose_gain_force * Fh / (Fnorm + 1e-9)
    else:
        dpos = np.zeros(3)

    max_d = abs(args.max_suggest_delta)
    n = float(np.linalg.norm(dpos))
    if n > max_d:
        dpos = dpos / n * max_d

    if Tnorm > 1e-9:
        drot = -args.pose_gain_torque * Th / (Tnorm + 1e-9)
    else:
        drot = np.zeros(3)

    max_r = math.radians(abs(args.max_suggest_rot_deg))
    rn = float(np.linalg.norm(drot))
    if rn > max_r:
        drot = drot / rn * max_r

    # 更易读的判断标签。
    notes = []
    if Fnorm < args.small_force_norm:
        notes.append("net_force_small_or_no_reliable_contact")
    else:
        idx = int(np.argmax(np.abs(Fh)))
        axis = ["hand_x", "hand_y", "hand_z"][idx]
        sign = "+" if Fh[idx] > 0 else "-"
        notes.append(f"net_force_main_axis={sign}{axis}")

    if Tnorm < args.small_torque_norm:
        notes.append("net_torque_small")
    else:
        idx = int(np.argmax(np.abs(Th)))
        axis = ["hand_roll_x", "hand_pitch_y", "hand_yaw_z"][idx]
        sign = "+" if Th[idx] > 0 else "-"
        notes.append(f"net_torque_main_axis={sign}{axis}")

    opp = wrench.get("thumb_non_thumb_opposition_cos")
    if opp is None:
        notes.append("no_thumb_non_thumb_opposition_pair")
    elif opp < -0.3:
        notes.append(f"thumb_non_thumb_opposition_ok_cos={opp:.3f}")
    else:
        notes.append(f"thumb_non_thumb_not_opposed_cos={opp:.3f}")

    return {
        "suggest_delta_local_xyz": dpos,
        "suggest_rotvec_local_xyz_rad": drot,
        "suggest_rotvec_local_xyz_deg": np.degrees(drot),
        "notes": notes,
        "F_norm": Fnorm,
        "Tau_norm": Tnorm,
    }


def log_row_text(row):
    w = row["wrench"]
    s = row["suggestion"]
    return (
        f"{row['phase']} step={row['step']} alpha={row['alpha']:.3f} "
        f"disp={row['object_disp']:.5f} "
        f"groups={w['groups']} "
        f"Fh={np.array(w['F_net_hand']).round(4).tolist()} "
        f"Th={np.array(w['Tau_net_hand']).round(4).tolist()} "
        f"opp={w['thumb_non_thumb_opposition_cos']} "
        f"suggest_d={np.array(s['suggest_delta_local_xyz']).round(5).tolist()} "
        f"suggest_rdeg={np.array(s['suggest_rotvec_local_xyz_deg']).round(3).tolist()} "
        f"notes={s['notes']}"
    )


def run(args):
    model = mujoco.MjModel.from_xml_path(str(resolve_path(args.model)))
    data = mujoco.MjData(model)

    candidate = load_json(args.candidate)
    p3 = load_json(args.p3_json)
    best_config = load_json(args.best_config)

    plan = p4f.selected_plan(p3, args.which)
    q_pre = plan["q_pre"]
    q_grasp = plan["q_grasp"]

    candidate_ctrl, candidate_ctrl_source = p4f.extract_candidate_ctrl(candidate, model)
    best_ctrl, best_ctrl_source = p4f.extract_best_config_ctrl(best_config, model)

    open_ctrl = p4f.make_open_ctrl(model)
    side_open_ctrl = p4f.make_side_open_ctrl(model, open_ctrl, candidate_ctrl, best_ctrl, args)
    close_target = p4f.make_close_target(model, side_open_ctrl, candidate_ctrl, args)

    sets = p4f.build_geom_sets(model, args.object_body)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    obj0 = object_state(model, data, args.object_body)["pos"].copy()
    q_current = {j: p4f.get_joint_qpos(model, data, j) or 0.0 for j in ARM_JOINTS}

    log_every = max(1, int(args.log_dt / model.opt.timestep))
    rows = []
    key_events = []

    print("\n========== V4.12P4O CONTACT WRENCH AUTOPSY ==========")
    print("model                :", resolve_path(args.model))
    print("candidate            :", resolve_path(args.candidate))
    print("p3_json              :", resolve_path(args.p3_json))
    print("best_config          :", resolve_path(args.best_config))
    print("candidate_ctrl_source:", candidate_ctrl_source)
    print("best_ctrl_source     :", best_ctrl_source)
    print("side_open_ctrl       :", side_open_ctrl)
    print("close_target         :", close_target)
    print("target_body          :", args.target_body)
    print("=====================================================\n")

    def step_with_state(phase, step, total_steps, alpha, arm_q, hand_ctrl):
        p4f.set_arm_qpos_and_ctrl(model, data, arm_q)
        p4f.set_hand_ctrl(model, data, hand_ctrl, args.direct_hand_qpos)
        mujoco.mj_step(model, data)

        obj = object_state(model, data, args.object_body)
        obj_delta = obj["pos"] - obj0
        wrench = collect_wrench(model, data, sets, args.object_body, args.target_body)
        suggestion = make_correction_suggestion(wrench, obj_delta, args)

        row = {
            "phase": phase,
            "step": int(step),
            "total_steps": int(total_steps),
            "alpha": float(alpha),
            "object_pos": obj["pos"],
            "object_delta": obj_delta,
            "object_disp": float(np.linalg.norm(obj_delta)),
            "object_velp": obj["velp"],
            "object_velr": obj["velr"],
            "hand_qpos": hand_qpos(model, data),
            "cmd_ctrl": dict(hand_ctrl),
            "wrench": wrench,
            "suggestion": suggestion,
        }

        interesting = bool(wrench["hand_object_contacts"]) or row["object_disp"] > args.interesting_disp

        if interesting:
            key_events.append(row)

        if interesting or step % log_every == 0 or step == total_steps:
            rows.append(row)
            print(log_row_text(row))

        return row

    def run_phase(viewer, phase, steps, cb):
        print(f"\n[PHASE] {phase}, steps={steps}")
        for k in range(steps + 1):
            a = k / max(1, steps)
            arm_q, hand_ctrl = cb(a, k)
            row = step_with_state(phase, k, steps, a, arm_q, hand_ctrl)

            if viewer is not None:
                viewer.sync()
                if args.frame_sleep > 0:
                    time.sleep(args.frame_sleep)

            if row["object_disp"] > args.stop_disp:
                print(f"[STOP] object_disp>{args.stop_disp}: {row['object_disp']:.5f}")
                return False
        return True

    def sequence(viewer=None):
        ok = run_phase(
            viewer,
            "move_to_pre_open",
            args.move_steps,
            lambda a, k: (
                interp_dict(q_current, q_pre, a, ARM_JOINTS),
                open_ctrl,
            ),
        )
        if not ok:
            return

        ok = run_phase(
            viewer,
            "thumb_preshape",
            args.thumb_preshape_steps,
            lambda a, k: (
                q_pre,
                interp_dict(open_ctrl, side_open_ctrl, a, HAND_JOINTS),
            ),
        )
        if not ok:
            return

        ok = run_phase(
            viewer,
            "move_to_grasp_side_open",
            args.move_steps,
            lambda a, k: (
                interp_dict(q_pre, q_grasp, a, ARM_JOINTS),
                side_open_ctrl,
            ),
        )
        if not ok:
            return

        close_steps = max(1, int(args.close_duration / model.opt.timestep))
        run_phase(
            viewer,
            "slow_close_wrench_autopsy",
            close_steps,
            lambda a, k: (
                q_grasp,
                interp_dict(side_open_ctrl, close_target, a, HAND_JOINTS),
            ),
        )

    if args.viewer:
        if mujoco.viewer is None:
            raise RuntimeError("mujoco.viewer unavailable")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            # P4O viewer force visualization:
            # 打开 MuJoCo 内置接触点/接触力显示。
            # 注意：这只是可视化 contact force，不改变仿真动力学。
            try:
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = True
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = True
                viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = True

                # Force arrows were too large and blocked the scene.
                # These visual parameters only affect display, not physics.
                try:
                    model.vis.map.force = float(args.vis_force_scale)
                    model.vis.map.torque = float(args.vis_torque_scale)
                except Exception as e:
                    print("[VIEWER OPT WARNING] cannot set vis.map force/torque:", e)

                for attr, val in [
                    ("forcewidth", args.vis_force_width),
                    ("contactwidth", args.vis_contact_width),
                    ("contactheight", args.vis_contact_height),
                ]:
                    try:
                        setattr(model.vis.scale, attr, float(val))
                    except Exception as e:
                        print(f"[VIEWER OPT WARNING] cannot set vis.scale.{attr}:", e)

                print(
                    "[VIEWER OPT] contact point / force enabled, "
                    f"force_scale={args.vis_force_scale}, "
                    f"force_width={args.vis_force_width}, "
                    f"contact_width={args.vis_contact_width}"
                )
            except Exception as e:
                print("[VIEWER OPT WARNING] cannot enable contact force flags:", e)

            sequence(viewer)
            print("[VIEWER] P4O 播放完成，关闭窗口退出。")
            if args.keep_viewer_open:
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.05)
    else:
        sequence(None)

    # 统计主要建议：只取有手物接触的帧。
    contact_rows = [r for r in key_events if r["wrench"]["hand_object_contacts"]]
    if contact_rows:
        mean_d = np.mean([r["suggestion"]["suggest_delta_local_xyz"] for r in contact_rows], axis=0)
        mean_r = np.mean([r["suggestion"]["suggest_rotvec_local_xyz_rad"] for r in contact_rows], axis=0)
        mean_F = np.mean([r["wrench"]["F_net_hand"] for r in contact_rows], axis=0)
        mean_T = np.mean([r["wrench"]["Tau_net_hand"] for r in contact_rows], axis=0)
    else:
        mean_d = np.zeros(3)
        mean_r = np.zeros(3)
        mean_F = np.zeros(3)
        mean_T = np.zeros(3)

    result = {
        "format": "v4_12p4o_contact_wrench_autopsy_debug",
        "model": str(resolve_path(args.model)),
        "candidate": str(resolve_path(args.candidate)),
        "p3_json": str(resolve_path(args.p3_json)),
        "best_config": str(resolve_path(args.best_config)),
        "args": vars(args),
        "candidate_ctrl_source": candidate_ctrl_source,
        "best_ctrl_source": best_ctrl_source,
        "side_open_ctrl": side_open_ctrl,
        "close_target": close_target,
        "mean_contact_F_net_hand": mean_F,
        "mean_contact_Tau_net_hand": mean_T,
        "mean_suggest_delta_local_xyz": mean_d,
        "mean_suggest_rotvec_local_xyz_rad": mean_r,
        "mean_suggest_rotvec_local_xyz_deg": np.degrees(mean_r),
        "num_rows": len(rows),
        "num_contact_rows": len(contact_rows),
        "key_events": key_events,
        "rows": rows,
    }

    save_json(args.out, result)

    report_path = resolve_path(args.report)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    with open(report_path, "w") as f:
        f.write("========== V4.12P4O CONTACT WRENCH AUTOPSY REPORT ==========\n\n")
        f.write(f"model: {resolve_path(args.model)}\n")
        f.write(f"candidate: {resolve_path(args.candidate)}\n")
        f.write(f"p3_json: {resolve_path(args.p3_json)}\n")
        f.write(f"best_config: {resolve_path(args.best_config)}\n\n")
        f.write("---- CTRL ----\n")
        f.write(f"side_open_ctrl: {side_open_ctrl}\n")
        f.write(f"close_target  : {close_target}\n\n")
        f.write("---- MEAN CONTACT WRENCH IN HAND FRAME ----\n")
        f.write(f"mean F_hand: {mean_F.tolist()}\n")
        f.write(f"mean Tau_hand: {mean_T.tolist()}\n")
        f.write(f"mean suggested delta local xyz: {mean_d.tolist()}\n")
        f.write(f"mean suggested rotvec local deg: {np.degrees(mean_r).tolist()}\n\n")
        f.write("---- KEY EVENTS ----\n")
        for r in key_events[:200]:
            f.write(log_row_text(r) + "\n")
            for c in r["wrench"]["hand_object_contacts"]:
                f.write(
                    f"    contact group={c['group']} seg={c['segment']} geom={c['hand_geom']} "
                    f"fN={c['normal_force']:.4f} "
                    f"pos={np.array(c['pos']).round(5).tolist()} "
                    f"f_obj_world={np.array(c['f_obj_world_normal_oriented']).round(5).tolist()} "
                    f"tau={np.array(c['tau_obj_world']).round(5).tolist()}\n"
                )

    print("\n========== P4O RESULT ==========")
    print("out:", resolve_path(args.out))
    print("report:", report_path)
    print("num_contact_rows:", len(contact_rows))
    print("mean_contact_F_net_hand:", mean_F)
    print("mean_contact_Tau_net_hand:", mean_T)
    print("mean_suggest_delta_local_xyz:", mean_d)
    print("mean_suggest_rotvec_local_xyz_deg:", np.degrees(mean_r))
    print("================================\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--p3-json", required=True)
    ap.add_argument("--best-config", required=True)
    ap.add_argument("--which", default="best_available", choices=["best_available", "best_pass"])
    ap.add_argument("--object-body", default="grasp_can")
    ap.add_argument("--target-body", default="fr3_link7")
    ap.add_argument("--out", required=True)
    ap.add_argument("--report", required=True)

    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--keep-viewer-open", action="store_true")
    ap.add_argument("--direct-hand-qpos", action="store_true")

    ap.add_argument("--move-steps", type=int, default=80)
    ap.add_argument("--thumb-preshape-steps", type=int, default=80)
    ap.add_argument("--close-duration", type=float, default=1.2)
    ap.add_argument("--finger-close-scale", type=float, default=0.85)
    ap.add_argument("--thumb-pitch-from-finger-gain", type=float, default=0.35)
    ap.add_argument("--preshape-fingers-from-best", action="store_true")

    ap.add_argument("--pose-gain-force", type=float, default=0.004)
    ap.add_argument("--pose-gain-torque", type=float, default=math.radians(2.0))
    ap.add_argument("--max-suggest-delta", type=float, default=0.006)
    ap.add_argument("--max-suggest-rot-deg", type=float, default=3.0)
    ap.add_argument("--small-force-norm", type=float, default=0.05)
    ap.add_argument("--small-torque-norm", type=float, default=0.002)

    ap.add_argument("--interesting-disp", type=float, default=0.003)
    ap.add_argument("--stop-disp", type=float, default=0.035)
    ap.add_argument("--log-dt", type=float, default=0.05)
    ap.add_argument("--frame-sleep", type=float, default=0.001)

    # viewer force visualization tuning
    ap.add_argument("--vis-force-scale", type=float, default=0.00018)
    ap.add_argument("--vis-torque-scale", type=float, default=0.00008)
    ap.add_argument("--vis-force-width", type=float, default=0.00006)
    ap.add_argument("--vis-contact-width", type=float, default=0.00035)
    ap.add_argument("--vis-contact-height", type=float, default=0.00012)

    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
