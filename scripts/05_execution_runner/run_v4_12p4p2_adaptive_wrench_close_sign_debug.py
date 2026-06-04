#!/usr/bin/env python3
"""
文件名：
    run_v4_12p4p2_adaptive_wrench_close_sign_debug.py

脚本类别：
    debug / runner / adaptive-contact-wrench-close / wrist-assisted-close

用途：
    本脚本用于 V4.12P4P2 阶段。
    当前已经确认：单纯按预设手指角度闭合无法稳定侧握细长物体。
    本脚本尝试把 P4O 中的接触合力/合力矩诊断用于在线调节：
        1. 手指慢速闭合；
        2. 实时读取 hand-object contact wrench；
        3. 如果合力/力矩显示物体被推偏，就通过 fr3_link7 Jacobian 对 handbase 做小范围位姿修正；
        4. 如果只有 thumb 接触，暂停 thumb 继续闭合，让四指侧追上；
        5. 如果只有非拇指接触，暂停非拇指继续闭合，让 thumb 侧追上；
        6. 只有实时出现 thumb + 非拇指对抗接触，并连续稳定，才允许 squeeze / hold / lift。

输入：
    --model
        MuJoCo XML 场景。
    --candidate
        当前 candidate JSON。
    --p3-json
        当前 P3 JSON。
    --best-config
        已修正 ctrl semantics 的 best_config。
    --object-body
        被抓物体 body 名，例如 grasp_can。
    --target-body
        用于 Jacobian 微调的 body，默认 fr3_link7。

输出：
    --out
        JSON 记录自适应过程、每帧 wrench、handbase 微调量、live contact gate 状态。

当前流程位置：
    P4O contact wrench autopsy
        -> P4P adaptive wrench close
        -> 如果有效，再固化为正式 contact-seeking close 策略

本脚本不负责：
    1. 不做大范围 IK；
    2. 不重新选择候选；
    3. 不离线优化 Top-K；
    4. 不把历史接触当作成功；
    5. 不保证实机控制，只用于验证“合力不对则微调位姿”的控制思想。
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
P4O_PATH = PROJECT / "scripts/06_diagnostics_viewer/run_v4_12p4o_contact_wrench_autopsy_debug.py"

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


def import_py(path, name):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p4f = import_py(P4F_PATH, "p4f")
p4o = import_py(P4O_PATH, "p4o")


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


def get_joint_value(model, data, name):
    v = p4f.get_joint_qpos(model, data, name)
    return 0.0 if v is None else float(v)


def clamp_joint(model, joint_name, q):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return q
    if bool(model.jnt_limited[jid]):
        lo, hi = model.jnt_range[jid]
        return float(np.clip(q, lo, hi))
    return float(q)


def arm_joint_dofs(model):
    dofs = []
    qpos_addrs = []
    for jn in ARM_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jn)
        if jid < 0:
            raise RuntimeError(f"missing arm joint: {jn}")
        dofs.append(int(model.jnt_dofadr[jid]))
        qpos_addrs.append(int(model.jnt_qposadr[jid]))
    return dofs, qpos_addrs


def body_pose(model, data, body):
    bid = p4f.body_id(model, body)
    if bid < 0:
        raise RuntimeError(f"missing body: {body}")
    R = data.xmat[bid].reshape(3, 3).copy()
    p = data.xpos[bid].copy()
    return p, R


def apply_body_delta_by_jacobian(model, data, arm_q, target_body, dpos_local, drot_local, args):
    """
    用 MuJoCo body Jacobian 对 fr3_link7 做微小位姿修正。
    这里只更新 7 个 arm joint，不动手指。
    dpos_local / drot_local 是 target_body 局部坐标下的小修正。
    """
    dpos_local = np.asarray(dpos_local, dtype=float).reshape(3)
    drot_local = np.asarray(drot_local, dtype=float).reshape(3)

    if np.linalg.norm(dpos_local) < 1e-12 and np.linalg.norm(drot_local) < 1e-12:
        return arm_q, np.zeros(7)

    bid = p4f.body_id(model, target_body)
    if bid < 0:
        raise RuntimeError(f"missing target body: {target_body}")

    _, R = body_pose(model, data, target_body)
    dpos_world = R @ dpos_local
    drot_world = R @ drot_local

    jacp = np.zeros((3, model.nv))
    jacr = np.zeros((3, model.nv))
    mujoco.mj_jacBody(model, data, jacp, jacr, bid)

    dofs, _ = arm_joint_dofs(model)
    J = np.vstack([jacp[:, dofs], jacr[:, dofs]])
    twist = np.concatenate([dpos_world, drot_world])

    lam = float(args.jacobian_damping)
    A = J @ J.T + (lam * lam) * np.eye(6)

    try:
        dq = J.T @ np.linalg.solve(A, twist)
    except np.linalg.LinAlgError:
        dq = J.T @ np.linalg.pinv(A) @ twist

    max_dq = float(args.max_joint_delta_per_step)
    dq = np.clip(dq, -max_dq, max_dq)

    out = dict(arm_q)
    for i, jn in enumerate(ARM_JOINTS):
        out[jn] = clamp_joint(model, jn, float(out[jn]) + float(dq[i]))

    return out, dq


def limit_vec(v, max_norm):
    v = np.asarray(v, dtype=float).copy()
    n = float(np.linalg.norm(v))
    if n > max_norm > 0:
        return v / n * max_norm
    return v


def make_ctrl_with_pauses(base_ctrl, pause_ctrl):
    out = dict(base_ctrl)
    for group, value in pause_ctrl.items():
        j = GROUP_TO_JOINT.get(group)
        if j is not None:
            out[j] = float(value)
    return out


def live_contact_state(wrench, args):
    groups = wrench["groups"]
    has_thumb = "thumb" in groups
    live_non = [g for g in NON_THUMB if g in groups]
    opp = wrench.get("thumb_non_thumb_opposition_cos")
    opp_ok = (opp is not None) and (opp <= args.opposition_cos_threshold)
    enough = has_thumb and (len(live_non) >= args.min_live_non_thumb)
    return {
        "groups": dict(groups),
        "has_thumb": has_thumb,
        "live_non_thumb": live_non,
        "num_live_non_thumb": len(live_non),
        "opposition_cos": opp,
        "opposition_ok": bool(opp_ok),
        "contact_ready_now": bool(enough and opp_ok),
    }


def setup_viewer_options(model, viewer, args):
    """
    P4P viewer display setup.

    用途：
        彻底关闭 MuJoCo 内置 contact force / contact point 可视化。
        本函数只影响 viewer 显示，不改变仿真动力学。
        后续自适应调节以 terminal/json 中的 wrench、contact group、opposition、stable_count 为准。
    """
    try:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
    except Exception as e:
        print("[VIEWER WARNING] cannot set contact display flags:", e)

    print("========== VIEWER VISUAL PARAMS ==========")
    try:
        print("mjVIS_CONTACTPOINT:", bool(viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT]))
        print("mjVIS_CONTACTFORCE:", bool(viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE]))
        print("mjVIS_TRANSPARENT:", bool(viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT]))
    except Exception as e:
        print("[VIEWER WARNING] cannot print viewer flags:", e)
    print("contact force/point visualization disabled")
    print("==========================================")


def run(args):
    model = mujoco.MjModel.from_xml_path(str(resolve_path(args.model)))
    data = mujoco.MjData(model)

    candidate = load_json(args.candidate)
    p3 = load_json(args.p3_json)
    best_config = load_json(args.best_config)

    plan = p4f.selected_plan(p3, args.which)
    q_pre = plan["q_pre"]
    q_grasp = plan["q_grasp"]
    q_lift = plan.get("q_lift", q_grasp)

    candidate_ctrl, candidate_ctrl_source = p4f.extract_candidate_ctrl(candidate, model)
    best_ctrl, best_ctrl_source = p4f.extract_best_config_ctrl(best_config, model)

    open_ctrl = p4f.make_open_ctrl(model)
    side_open_ctrl = p4f.make_side_open_ctrl(model, open_ctrl, candidate_ctrl, best_ctrl, args)
    close_target = p4f.make_close_target(model, side_open_ctrl, candidate_ctrl, args)

    sets = p4f.build_geom_sets(model, args.object_body)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    q_current = {j: get_joint_value(model, data, j) for j in ARM_JOINTS}
    arm_q = dict(q_grasp)

    obj0 = p4o.object_state(model, data, args.object_body)["pos"].copy()

    rows = []
    stable_count = 0
    pause_ctrl = {}
    cum_dpos_local = np.zeros(3)
    cum_drot_local = np.zeros(3)
    final_ctrl = dict(side_open_ctrl)
    final_arm_q = dict(arm_q)
    stop_reason = ""
    adaptive_ready = False

    log_every = max(1, int(args.log_dt / model.opt.timestep))

    print("\n========== V4.12P4P2 ADAPTIVE WRENCH CLOSE SIGN TEST ==========")
    print("model                :", resolve_path(args.model))
    print("candidate            :", resolve_path(args.candidate))
    print("p3_json              :", resolve_path(args.p3_json))
    print("best_config          :", resolve_path(args.best_config))
    print("candidate_ctrl_source:", candidate_ctrl_source)
    print("best_ctrl_source     :", best_ctrl_source)
    print("side_open_ctrl       :", side_open_ctrl)
    print("close_target         :", close_target)
    print("target_body          :", args.target_body)
    print("adapt_sign           :", args.adapt_sign)
    print("====================================================\n")

    def step_common(phase, step, total_steps, alpha, arm_cmd, hand_ctrl, adapt=False):
        nonlocal arm_q, stable_count, pause_ctrl, cum_dpos_local, cum_drot_local
        nonlocal final_ctrl, final_arm_q, stop_reason, adaptive_ready

        p4f.set_arm_qpos_and_ctrl(model, data, arm_cmd)
        p4f.set_hand_ctrl(model, data, hand_ctrl, args.direct_hand_qpos)
        mujoco.mj_step(model, data)

        obj = p4o.object_state(model, data, args.object_body)
        obj_delta = obj["pos"] - obj0
        obj_disp = float(np.linalg.norm(obj_delta))

        wrench = p4o.collect_wrench(model, data, sets, args.object_body, args.target_body)
        suggestion = p4o.make_correction_suggestion(wrench, obj_delta, args)
        live = live_contact_state(wrench, args)

        if adapt and wrench["hand_object_contacts"]:
            has_thumb = live["has_thumb"]
            n_non = live["num_live_non_thumb"]

            # 单侧接触时，暂停导致单侧推物体的一侧继续闭合。
            if has_thumb and n_non == 0:
                pause_ctrl["thumb"] = float(hand_ctrl["thumb_cmc_pitch"])
            elif (not has_thumb) and n_non > 0:
                for g in live["live_non_thumb"]:
                    pause_ctrl[g] = float(hand_ctrl[GROUP_TO_JOINT[g]])
            elif has_thumb and n_non > 0:
                # 一旦形成两侧接触，允许小幅继续闭合，但不立刻完全释放 pause。
                pass

            # 只有没达到稳定对抗时，才做 handbase 微伺服。
            if not live["contact_ready_now"]:
                raw_dpos = np.asarray(suggestion["suggest_delta_local_xyz"], dtype=float)
                raw_drot = np.asarray(suggestion["suggest_rotvec_local_xyz_rad"], dtype=float)

                # P4P2 关键修正：
                # P4O 的 suggestion 方向来自接触力符号约定，当前日志显示该方向可能反了。
                # 所以这里通过 adapt_sign 显式测试方向。
                dpos = float(args.adapt_sign) * raw_dpos
                drot = float(args.adapt_sign) * raw_drot

                dpos = limit_vec(dpos, args.max_adapt_delta_per_step)
                drot = limit_vec(drot, math.radians(args.max_adapt_rot_deg_per_step))

                # 累积位姿微调上限，避免越调越远。
                new_cum_dpos = cum_dpos_local + dpos
                if np.linalg.norm(new_cum_dpos) > args.max_total_adapt_delta:
                    dpos = limit_vec(new_cum_dpos, args.max_total_adapt_delta) - cum_dpos_local

                new_cum_drot = cum_drot_local + drot
                max_total_rot = math.radians(args.max_total_adapt_rot_deg)
                if np.linalg.norm(new_cum_drot) > max_total_rot:
                    drot = limit_vec(new_cum_drot, max_total_rot) - cum_drot_local

                arm_q, dq = apply_body_delta_by_jacobian(
                    model, data, arm_q, args.target_body, dpos, drot, args
                )

                cum_dpos_local += dpos
                cum_drot_local += drot
            else:
                dq = np.zeros(7)
        else:
            dq = np.zeros(7)

        if live["contact_ready_now"] and obj_disp <= args.max_ready_disp:
            stable_count += 1
        else:
            stable_count = 0

        ready = stable_count >= args.live_ready_stable_steps
        if ready:
            adaptive_ready = True

        final_ctrl = dict(hand_ctrl)
        final_arm_q = dict(arm_q)

        row = {
            "phase": phase,
            "step": int(step),
            "total_steps": int(total_steps),
            "alpha": float(alpha),
            "object_pos": obj["pos"],
            "object_delta": obj_delta,
            "object_disp": obj_disp,
            "arm_q": dict(arm_q),
            "hand_ctrl": dict(hand_ctrl),
            "pause_ctrl": dict(pause_ctrl),
            "adapt_sign": float(args.adapt_sign),
            "cum_dpos_local": cum_dpos_local.copy(),
            "cum_drot_local_rad": cum_drot_local.copy(),
            "cum_drot_local_deg": np.degrees(cum_drot_local),
            "jacobian_dq": dq,
            "wrench": wrench,
            "suggestion": suggestion,
            "live": live,
            "stable_count": int(stable_count),
            "ready": bool(ready),
        }

        interesting = bool(wrench["hand_object_contacts"]) or obj_disp > args.interesting_disp
        if interesting or step % log_every == 0 or step == total_steps or ready:
            rows.append(row)
            print(
                f"[{phase}] {step:4d}/{total_steps} alpha={alpha:.3f} "
                f"disp={obj_disp:.5f} groups={live['groups']} "
                f"opp={live['opposition_cos']} ready={ready} "
                f"stable={stable_count}/{args.live_ready_stable_steps} "
                f"cum_d={np.round(cum_dpos_local,5).tolist()} "
                f"cum_rdeg={np.round(np.degrees(cum_drot_local),3).tolist()} "
                f"pause={pause_ctrl}"
            )

        if obj_disp > args.stop_disp:
            stop_reason = f"object_disp_exceeded_{args.stop_disp}"
            print("[STOP]", stop_reason, "disp=", obj_disp)

        return ready

    def run_phase(viewer, phase, steps, cb):
        for k in range(steps + 1):
            a = k / max(1, steps)
            arm_cmd, hand_cmd, adapt = cb(a, k)
            ready = step_common(phase, k, steps, a, arm_cmd, hand_cmd, adapt=adapt)
            if viewer is not None:
                viewer.sync()
                if args.frame_sleep > 0:
                    time.sleep(args.frame_sleep)
            if stop_reason:
                return False
            if ready and phase == "adaptive_wrench_close":
                return True
        return True

    def sequence(viewer=None):
        nonlocal arm_q, pause_ctrl, final_ctrl, final_arm_q

        print("\n[PHASE] move_to_pre_open")
        ok = run_phase(
            viewer,
            "move_to_pre_open",
            args.move_steps,
            lambda a, k: (
                interp_dict(q_current, q_pre, a, ARM_JOINTS),
                open_ctrl,
                False,
            ),
        )
        if not ok:
            return

        print("\n[PHASE] thumb_preshape")
        ok = run_phase(
            viewer,
            "thumb_preshape",
            args.thumb_preshape_steps,
            lambda a, k: (
                q_pre,
                interp_dict(open_ctrl, side_open_ctrl, a, HAND_JOINTS),
                False,
            ),
        )
        if not ok:
            return

        print("\n[PHASE] move_to_grasp_side_open")
        ok = run_phase(
            viewer,
            "move_to_grasp_side_open",
            args.move_steps,
            lambda a, k: (
                interp_dict(q_pre, q_grasp, a, ARM_JOINTS),
                side_open_ctrl,
                False,
            ),
        )
        if not ok:
            return

        arm_q = dict(q_grasp)
        pause_ctrl = {}

        close_steps = max(1, int(args.close_duration / model.opt.timestep))
        print("\n[PHASE] adaptive_wrench_close")

        def adaptive_cb(a, k):
            base = interp_dict(side_open_ctrl, close_target, a, HAND_JOINTS)
            ctrl = make_ctrl_with_pauses(base, pause_ctrl)
            return arm_q, ctrl, True

        ok = run_phase(viewer, "adaptive_wrench_close", close_steps, adaptive_cb)
        if not ok:
            return

        if not adaptive_ready:
            print("[NO SQUEEZE] live contact gate not satisfied.")
            return

        # 微 squeeze：只在 live gate 通过后做很小幅度收紧。
        squeeze_ctrl = dict(final_ctrl)
        for j in HAND_JOINTS:
            start = float(final_ctrl.get(j, 0.0))
            end = float(close_target.get(j, start))
            squeeze_ctrl[j] = start + args.squeeze_fraction * (end - start)

        print("\n[PHASE] squeeze_after_adaptive_ready")
        squeeze_steps = max(1, int(args.squeeze_duration / model.opt.timestep))
        ok = run_phase(
            viewer,
            "squeeze_after_adaptive_ready",
            squeeze_steps,
            lambda a, k: (
                final_arm_q,
                interp_dict(final_ctrl, squeeze_ctrl, a, HAND_JOINTS),
                False,
            ),
        )
        if not ok:
            return

        print("\n[PHASE] hold_after_squeeze")
        hold_steps = max(1, int(args.hold_duration / model.opt.timestep))
        ok = run_phase(
            viewer,
            "hold_after_squeeze",
            hold_steps,
            lambda a, k: (
                final_arm_q,
                squeeze_ctrl,
                False,
            ),
        )
        if not ok:
            return

        if args.enable_lift:
            print("\n[PHASE] lift_after_adaptive_close")
            lift_steps = max(1, int(args.lift_duration / model.opt.timestep))
            run_phase(
                viewer,
                "lift_after_adaptive_close",
                lift_steps,
                lambda a, k: (
                    interp_dict(final_arm_q, q_lift, a, ARM_JOINTS),
                    squeeze_ctrl,
                    False,
                ),
            )

    if args.viewer:
        if mujoco.viewer is None:
            raise RuntimeError("mujoco.viewer unavailable")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            setup_viewer_options(model, viewer, args)
            sequence(viewer)
            print("[VIEWER] P4P 播放完成，关闭窗口退出。")
            if args.keep_viewer_open:
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.05)
    else:
        sequence(None)

    final_obj = p4o.object_state(model, data, args.object_body)
    final_wrench = p4o.collect_wrench(model, data, sets, args.object_body, args.target_body)

    result = {
        "format": "v4_12p4p2_adaptive_wrench_close_sign_debug",
        "model": str(resolve_path(args.model)),
        "candidate": str(resolve_path(args.candidate)),
        "p3_json": str(resolve_path(args.p3_json)),
        "best_config": str(resolve_path(args.best_config)),
        "args": vars(args),
        "adapt_sign": float(args.adapt_sign),
        "side_open_ctrl": side_open_ctrl,
        "close_target": close_target,
        "adaptive_ready": adaptive_ready,
        "stable_count": stable_count,
        "stop_reason": stop_reason,
        "cum_dpos_local": cum_dpos_local,
        "cum_drot_local_rad": cum_drot_local,
        "cum_drot_local_deg": np.degrees(cum_drot_local),
        "final_object_pos": final_obj["pos"],
        "final_object_disp": float(np.linalg.norm(final_obj["pos"] - obj0)),
        "final_wrench_groups": final_wrench["groups"],
        "final_opposition_cos": final_wrench.get("thumb_non_thumb_opposition_cos"),
        "rows": rows,
    }

    save_json(args.out, result)

    print("\n========== P4P2 RESULT ==========")
    print("out:", resolve_path(args.out))
    print("adaptive_ready:", adaptive_ready)
    print("stable_count:", stable_count)
    print("stop_reason:", stop_reason)
    print("final_object_disp:", result["final_object_disp"])
    print("final_wrench_groups:", result["final_wrench_groups"])
    print("final_opposition_cos:", result["final_opposition_cos"])
    print("cum_dpos_local:", cum_dpos_local)
    print("cum_drot_local_deg:", np.degrees(cum_drot_local))
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

    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--keep-viewer-open", action="store_true")
    ap.add_argument("--direct-hand-qpos", action="store_true")

    ap.add_argument("--move-steps", type=int, default=80)
    ap.add_argument("--thumb-preshape-steps", type=int, default=80)
    ap.add_argument("--close-duration", type=float, default=1.2)
    ap.add_argument("--squeeze-duration", type=float, default=0.4)
    ap.add_argument("--hold-duration", type=float, default=0.5)
    ap.add_argument("--enable-lift", action="store_true")
    ap.add_argument("--lift-duration", type=float, default=1.2)

    ap.add_argument("--finger-close-scale", type=float, default=0.85)
    ap.add_argument("--thumb-pitch-from-finger-gain", type=float, default=0.35)
    ap.add_argument("--preshape-fingers-from-best", action="store_true")

    # P4O suggestion gains, deliberately smaller than autopsy defaults.
    ap.add_argument("--pose-gain-force", type=float, default=0.0010)
    ap.add_argument("--pose-gain-torque", type=float, default=math.radians(0.50))

    # P4P2 sign test:
    # +1 表示沿用 P4P 原来的微调方向；
    # -1 表示把 P4O suggestion 方向整体反过来。
    # 当前日志显示原方向很可能在 thumb 单侧接触时把手往错误方向调，
    # 所以本脚本建议先用 -1 测试。
    ap.add_argument("--adapt-sign", type=float, default=-1.0)
    ap.add_argument("--max-suggest-delta", type=float, default=0.0010)
    ap.add_argument("--max-suggest-rot-deg", type=float, default=0.50)
    ap.add_argument("--small-force-norm", type=float, default=0.05)
    ap.add_argument("--small-torque-norm", type=float, default=0.002)

    # Adaptive Jacobian servo limits.
    ap.add_argument("--max-adapt-delta-per-step", type=float, default=0.00035)
    ap.add_argument("--max-adapt-rot-deg-per-step", type=float, default=0.10)
    ap.add_argument("--max-total-adapt-delta", type=float, default=0.006)
    ap.add_argument("--max-total-adapt-rot-deg", type=float, default=4.0)
    ap.add_argument("--jacobian-damping", type=float, default=0.08)
    ap.add_argument("--max-joint-delta-per-step", type=float, default=0.004)

    # Live contact gate.
    ap.add_argument("--min-live-non-thumb", type=int, default=1)
    ap.add_argument("--opposition-cos-threshold", type=float, default=-0.30)
    ap.add_argument("--live-ready-stable-steps", type=int, default=40)
    ap.add_argument("--max-ready-disp", type=float, default=0.014)
    ap.add_argument("--squeeze-fraction", type=float, default=0.20)

    ap.add_argument("--interesting-disp", type=float, default=0.003)
    ap.add_argument("--stop-disp", type=float, default=0.030)
    ap.add_argument("--log-dt", type=float, default=0.05)
    ap.add_argument("--frame-sleep", type=float, default=0.001)

    # Viewer force display.
    ap.add_argument("--vis-force-scale", type=float, default=0.00018)
    ap.add_argument("--vis-torque-scale", type=float, default=0.00008)
    ap.add_argument("--vis-force-width", type=float, default=0.00006)
    ap.add_argument("--vis-contact-width", type=float, default=0.00035)
    ap.add_argument("--vis-contact-height", type=float, default=0.00012)

    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
