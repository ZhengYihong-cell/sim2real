#!/usr/bin/env python3
"""
文件名：
    run_v4_12p4m_touch_freeze_then_squeeze_debug.py

脚本类别：
    debug / runner / touch-freeze / contact-mechanics-diagnostic

用途：
    本脚本用于 V4.12P4M 阶段。
    当前问题不是继续评分，而是要验证一个更直接的接触控制策略：
        手指缓慢闭合；
        哪个手指先碰物体，就冻结该手指；
        等 thumb 和若干非拇指都冻结在物体周围后；
        再从冻结姿态一起小幅闭合，观察能否形成稳定抓握。

核心思想：
    不让某一根手指继续把物体推飞。
    接触即冻结，先形成“围住物体”的姿态，再统一 squeeze。

输入：
    --model
        MuJoCo XML 场景，建议使用 hard_support 版本。
    --candidate
        当前 candidate JSON。
    --p3-json
        当前 P3 JSON。
    --best-config
        已修正 ctrl semantics 的 best_config。
    --object-body
        被抓物体 body 名，例如 grasp_can。

输出：
    --out
        JSON 记录冻结顺序、每个冻结时刻、物体位移、最终接触状态。
    viewer
        可视化 touch-freeze-then-squeeze 过程。

当前流程位置：
    ctrl semantics 修正
        -> touch-freeze diagnostic
        -> 判断当前 hand pose 是否合理
        -> 再决定是否继续 hand-local pose 调整

本脚本不负责：
    1. 不重新做 IK；
    2. 不重新选 best；
    3. 不做完整 force closure；
    4. 不直接 lift；
    5. 不把物体推飞后的接触当成成功。
"""

from pathlib import Path
import argparse
import importlib.util
import json
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

GROUP_TO_JOINT = {
    "thumb": "thumb_cmc_pitch",
    "index": "index_mcp_pitch",
    "middle": "middle_mcp_pitch",
    "ring": "ring_mcp_pitch",
    "pinky": "pinky_mcp_pitch",
}

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
    return ""


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


def object_pos(model, data, object_body):
    bid = p4f.body_id(model, object_body)
    if bid < 0:
        raise RuntimeError(f"cannot find object body: {object_body}")
    return data.xpos[bid].copy()


def contact_force6(model, data, ci):
    f = np.zeros(6, dtype=float)
    try:
        mujoco.mj_contactForce(model, data, int(ci), f)
    except Exception:
        pass
    return f


def collect_hand_object_contacts(model, data, sets):
    object_geoms = sets["object_geoms"]
    hand_geoms = sets["hand_geoms"]
    support_geoms = sets["support_geoms"]

    groups = {}
    contacts = []
    object_support = []

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
        normal_force = float(max(0.0, f6[0]))

        if (g1_obj and g2_sup) or (g2_obj and g1_sup):
            object_support.append({
                "geom1": geom_name(model, g1),
                "geom2": geom_name(model, g2),
                "dist": float(con.dist),
                "normal_force": normal_force,
                "pos": np.array(con.pos, dtype=float).copy(),
            })

        if not ((g1_hand and g2_obj) or (g2_hand and g1_obj)):
            continue

        hg = g1 if g1_hand else g2
        group = classify_hand_group(model, hg)
        seg = classify_segment(model, hg)

        item = {
            "contact_index": int(i),
            "group": group,
            "segment": seg,
            "hand_geom": geom_name(model, hg),
            "hand_body": geom_body_name(model, hg),
            "dist": float(con.dist),
            "normal_force": normal_force,
            "pos": np.array(con.pos, dtype=float).copy(),
        }
        contacts.append(item)

        if group:
            groups[group] = groups.get(group, 0) + 1

    min_support_dist = min([x["dist"] for x in object_support], default=None)

    return {
        "groups": groups,
        "contacts": contacts,
        "object_support": object_support,
        "min_object_support_dist": min_support_dist,
    }


def hand_qpos(model, data):
    return p4f.hand_qpos(model, data)


def current_joint(model, data, joint_name):
    v = p4f.get_joint_qpos(model, data, joint_name)
    return 0.0 if v is None else float(v)


def make_ctrl_with_frozen(base_ctrl, frozen_ctrl):
    ctrl = dict(base_ctrl)
    for g, val in frozen_ctrl.items():
        j = GROUP_TO_JOINT.get(g)
        if j:
            ctrl[j] = float(val)
    return ctrl


def freeze_new_groups(model, data, contact, frozen_ctrl, freeze_events, phase, step, alpha, obj_ref, args):
    groups = contact["groups"]
    obj = object_pos(model, data, args.object_body)
    disp = float(np.linalg.norm(obj - obj_ref))

    for g in ["thumb", "index", "middle", "ring", "pinky"]:
        if g not in groups:
            continue
        if g in frozen_ctrl:
            continue

        j = GROUP_TO_JOINT[g]
        val = current_joint(model, data, j)
        frozen_ctrl[g] = val

        detail = [c for c in contact["contacts"] if c["group"] == g]

        event = {
            "group": g,
            "joint": j,
            "frozen_value": val,
            "phase": phase,
            "step": int(step),
            "alpha": float(alpha),
            "object_disp": disp,
            "object_pos": obj.copy(),
            "contacts": detail,
            "all_groups": dict(groups),
            "min_object_support_dist": contact["min_object_support_dist"],
        }
        freeze_events.append(event)

        print(
            f"[FREEZE] {g:6s} {j}={val:.5f} "
            f"phase={phase} step={step} alpha={alpha:.3f} "
            f"disp={disp:.5f} contacts={[(x['hand_geom'], x['segment'], x['normal_force']) for x in detail]}"
        )


def run_phase(model, data, viewer, args, phase, steps, callback, logs, sets, obj_ref):
    log_every = max(1, int(args.log_dt / model.opt.timestep))

    print(f"\n[PHASE] {phase}, steps={steps}")

    for k in range(steps + 1):
        alpha = k / max(1, steps)
        callback(alpha, k)

        mujoco.mj_step(model, data)

        if k % log_every == 0 or k == steps:
            obj = object_pos(model, data, args.object_body)
            contact = collect_hand_object_contacts(model, data, sets)
            row = {
                "phase": phase,
                "step": int(k),
                "alpha": float(alpha),
                "object_pos": obj.copy(),
                "object_disp": float(np.linalg.norm(obj - obj_ref)),
                "hand_qpos": hand_qpos(model, data),
                "contact": contact,
            }
            logs.append(row)
            print(
                f"[{phase}] {k:4d}/{steps} "
                f"disp={row['object_disp']:.5f} "
                f"groups={contact['groups']} "
                f"support={contact['min_object_support_dist']}"
            )

        if viewer is not None:
            viewer.sync()
            if args.frame_sleep > 0:
                time.sleep(args.frame_sleep)


def run(args):
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

    sets = p4f.build_geom_sets(model, args.object_body)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    q_current = {j: p4f.get_joint_qpos(model, data, j) or 0.0 for j in ARM_JOINTS}
    obj_ref = object_pos(model, data, args.object_body)

    logs = []
    freeze_events = []
    frozen_ctrl = {}

    print("\n========== V4.12P4M TOUCH-FREEZE-THEN-SQUEEZE ==========")
    print("model                :", model_path)
    print("candidate            :", candidate_path)
    print("p3_json              :", p3_path)
    print("best_config          :", best_config_path)
    print("candidate_ctrl_source:", candidate_ctrl_source)
    print("best_ctrl_source     :", best_ctrl_source)
    print("side_open_ctrl       :", side_open_ctrl)
    print("close_target         :", close_target)
    print("probe_duration       :", args.probe_duration)
    print("squeeze_fraction     :", args.squeeze_fraction)
    print("min_freeze_non_thumb :", args.min_freeze_non_thumb)
    print("max_probe_disp       :", args.max_probe_disp)
    print("========================================================\n")

    stop_reason = ""
    frozen_ready_step = None

    def sequence(viewer=None):
        nonlocal stop_reason, frozen_ready_step

        run_phase(
            model, data, viewer, args,
            "move_to_pre_open",
            args.move_steps,
            lambda a, k: (
                p4f.set_arm_qpos_and_ctrl(model, data, interp_dict(q_current, q_pre, a, ARM_JOINTS)),
                p4f.set_hand_ctrl(model, data, open_ctrl, args.direct_hand_qpos),
            ),
            logs, sets, obj_ref,
        )

        run_phase(
            model, data, viewer, args,
            "thumb_preshape",
            args.thumb_preshape_steps,
            lambda a, k: (
                p4f.set_arm_qpos_and_ctrl(model, data, q_pre),
                p4f.set_hand_ctrl(model, data, interp_dict(open_ctrl, side_open_ctrl, a, HAND_JOINTS), args.direct_hand_qpos),
            ),
            logs, sets, obj_ref,
        )

        run_phase(
            model, data, viewer, args,
            "move_to_grasp_side_open",
            args.move_steps,
            lambda a, k: (
                p4f.set_arm_qpos_and_ctrl(model, data, interp_dict(q_pre, q_grasp, a, ARM_JOINTS)),
                p4f.set_hand_ctrl(model, data, side_open_ctrl, args.direct_hand_qpos),
            ),
            logs, sets, obj_ref,
        )

        probe_steps = max(1, int(args.probe_duration / model.opt.timestep))
        log_every = max(1, int(args.log_dt / model.opt.timestep))

        print(f"\n[PHASE] slow_probe_close_and_freeze, steps={probe_steps}")

        for k in range(probe_steps + 1):
            a = k / probe_steps
            target = interp_dict(side_open_ctrl, close_target, a, HAND_JOINTS)
            ctrl = make_ctrl_with_frozen(target, frozen_ctrl)

            p4f.set_arm_qpos_and_ctrl(model, data, q_grasp)
            p4f.set_hand_ctrl(model, data, ctrl, args.direct_hand_qpos)

            mujoco.mj_step(model, data)

            contact = collect_hand_object_contacts(model, data, sets)
            obj = object_pos(model, data, args.object_body)
            disp = float(np.linalg.norm(obj - obj_ref))

            freeze_new_groups(model, data, contact, frozen_ctrl, freeze_events, "slow_probe_close_and_freeze", k, a, obj_ref, args)

            non_thumb_frozen = [g for g in NON_THUMB if g in frozen_ctrl]
            ready = ("thumb" in frozen_ctrl) and (len(non_thumb_frozen) >= args.min_freeze_non_thumb)

            if k % log_every == 0 or k == probe_steps or ready:
                row = {
                    "phase": "slow_probe_close_and_freeze",
                    "step": int(k),
                    "alpha": float(a),
                    "object_pos": obj.copy(),
                    "object_disp": disp,
                    "cmd_ctrl_before_freeze_override": target,
                    "cmd_ctrl_after_freeze_override": ctrl,
                    "frozen_ctrl": dict(frozen_ctrl),
                    "non_thumb_frozen": non_thumb_frozen,
                    "ready": bool(ready),
                    "hand_qpos": hand_qpos(model, data),
                    "contact": contact,
                }
                logs.append(row)
                print(
                    f"[probe] {k:4d}/{probe_steps} alpha={a:.3f} "
                    f"disp={disp:.5f} groups={contact['groups']} "
                    f"frozen={frozen_ctrl} ready={ready}"
                )

            if disp > args.max_probe_disp and not ready:
                stop_reason = "object_moved_too_much_before_fingers_surround_it"
                print(f"[STOP] {stop_reason}: disp={disp:.5f}, frozen={frozen_ctrl}")
                break

            if ready:
                frozen_ready_step = int(k)
                print(f"[READY] thumb + {len(non_thumb_frozen)} non-thumb groups frozen. Enter squeeze.")
                break

            if viewer is not None:
                viewer.sync()
                if args.frame_sleep > 0:
                    time.sleep(args.frame_sleep)

        if frozen_ready_step is None:
            if not stop_reason:
                stop_reason = "not_enough_fingers_frozen_after_probe"
            print(f"[NO SQUEEZE] {stop_reason}")
            return

        squeeze_steps = max(1, int(args.squeeze_duration / model.opt.timestep))
        print(f"\n[PHASE] synchronized_micro_squeeze, steps={squeeze_steps}")

        frozen_base_ctrl = dict(side_open_ctrl)
        for g, val in frozen_ctrl.items():
            j = GROUP_TO_JOINT[g]
            frozen_base_ctrl[j] = float(val)

        for k in range(squeeze_steps + 1):
            a = k / squeeze_steps
            squeeze_ctrl = dict(frozen_base_ctrl)

            for g, val in frozen_ctrl.items():
                j = GROUP_TO_JOINT[g]
                final = float(close_target[j])
                start = float(val)
                squeeze_ctrl[j] = start + args.squeeze_fraction * a * (final - start)

            p4f.set_arm_qpos_and_ctrl(model, data, q_grasp)
            p4f.set_hand_ctrl(model, data, squeeze_ctrl, args.direct_hand_qpos)

            mujoco.mj_step(model, data)

            obj = object_pos(model, data, args.object_body)
            disp = float(np.linalg.norm(obj - obj_ref))
            contact = collect_hand_object_contacts(model, data, sets)

            if k % log_every == 0 or k == squeeze_steps:
                row = {
                    "phase": "synchronized_micro_squeeze",
                    "step": int(k),
                    "alpha": float(a),
                    "object_pos": obj.copy(),
                    "object_disp": disp,
                    "squeeze_ctrl": squeeze_ctrl,
                    "frozen_ctrl": dict(frozen_ctrl),
                    "hand_qpos": hand_qpos(model, data),
                    "contact": contact,
                }
                logs.append(row)
                print(
                    f"[squeeze] {k:4d}/{squeeze_steps} alpha={a:.3f} "
                    f"disp={disp:.5f} groups={contact['groups']} "
                    f"support={contact['min_object_support_dist']}"
                )

            if disp > args.max_squeeze_disp:
                stop_reason_local = "object_moved_too_much_during_squeeze"
                print(f"[STOP] {stop_reason_local}: disp={disp:.5f}")
                break

            if viewer is not None:
                viewer.sync()
                if args.frame_sleep > 0:
                    time.sleep(args.frame_sleep)

        run_phase(
            model, data, viewer, args,
            "hold_after_squeeze",
            max(1, int(args.hold_duration / model.opt.timestep)),
            lambda a, k: (
                p4f.set_arm_qpos_and_ctrl(model, data, q_grasp),
                p4f.set_hand_ctrl(model, data, squeeze_ctrl, args.direct_hand_qpos),
            ),
            logs, sets, obj_ref,
        )

    if args.viewer:
        if mujoco.viewer is None:
            raise RuntimeError("mujoco.viewer unavailable")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            sequence(viewer)
            print("[VIEWER] 播放完成，关闭窗口退出。")
            if args.keep_viewer_open:
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.05)
    else:
        sequence(None)

    final_obj = object_pos(model, data, args.object_body)
    final_contact = collect_hand_object_contacts(model, data, sets)

    out = {
        "format": "v4_12p4m_touch_freeze_then_squeeze_debug",
        "model": str(model_path),
        "candidate": str(candidate_path),
        "p3_json": str(p3_path),
        "best_config": str(best_config_path),
        "args": vars(args),
        "side_open_ctrl": side_open_ctrl,
        "close_target": close_target,
        "frozen_ctrl": frozen_ctrl,
        "freeze_events": freeze_events,
        "frozen_ready_step": frozen_ready_step,
        "stop_reason": stop_reason,
        "final_object_pos": final_obj,
        "final_object_disp": float(np.linalg.norm(final_obj - obj_ref)),
        "final_contact": final_contact,
        "final_hand_qpos": hand_qpos(model, data),
        "logs": logs,
    }

    save_json(args.out, out)

    print("\n========== P4M RESULT ==========")
    print("frozen_ctrl:", frozen_ctrl)
    print("freeze_events:", [(e["group"], e["step"], e["alpha"], e["object_disp"]) for e in freeze_events])
    print("frozen_ready_step:", frozen_ready_step)
    print("stop_reason:", stop_reason)
    print("final_disp:", out["final_object_disp"])
    print("final_groups:", final_contact["groups"])
    print("saved:", resolve_path(args.out))
    print("===============================\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--p3-json", required=True)
    ap.add_argument("--best-config", required=True)
    ap.add_argument("--which", default="best_available", choices=["best_available", "best_pass"])
    ap.add_argument("--object-body", default="grasp_can")
    ap.add_argument("--out", required=True)

    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--keep-viewer-open", action="store_true")
    ap.add_argument("--direct-hand-qpos", action="store_true")

    ap.add_argument("--move-steps", type=int, default=100)
    ap.add_argument("--thumb-preshape-steps", type=int, default=100)
    ap.add_argument("--probe-duration", type=float, default=2.5)
    ap.add_argument("--squeeze-duration", type=float, default=0.8)
    ap.add_argument("--hold-duration", type=float, default=0.8)

    ap.add_argument("--finger-close-scale", type=float, default=0.85)
    ap.add_argument("--thumb-pitch-from-finger-gain", type=float, default=0.35)
    ap.add_argument("--preshape-fingers-from-best", action="store_true")

    ap.add_argument("--min-freeze-non-thumb", type=int, default=2)
    ap.add_argument("--squeeze-fraction", type=float, default=0.25)
    ap.add_argument("--max-probe-disp", type=float, default=0.020)
    ap.add_argument("--max-squeeze-disp", type=float, default=0.035)

    ap.add_argument("--log-dt", type=float, default=0.05)
    ap.add_argument("--frame-sleep", type=float, default=0.002)

    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
