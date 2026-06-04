#!/usr/bin/env python3
"""
文件名：
    run_v4_12p4i_force_proxy_latch_close_debug.py

脚本类别：
    debug / runner / force-closure-proxy-latch / support-aware-close / viewer

用途：
    本脚本用于 V4.12P4I 阶段，修复“只要 thumb + 任意非拇指接触就冻结”的错误逻辑。
    真实情况中，thumb 可能先推动物体，使物体被动碰到某根四指，但这不代表已经形成抓握。
    因此本脚本只在“简单力闭合代理分数达标”后冻结 held_ctrl。

核心逻辑：
    1. 按 P3 best 的 q_pre / q_grasp / q_lift 执行。
    2. 按 candidate prior 生成 close_target，不写死四指角度。
    3. close 阶段逐步闭合。
    4. 每一步检查 hand-object 接触。
    5. 对 thumb contact 与 non-thumb contact 两两配对，计算 force-closure proxy：
        - 物体中心到 thumb-finger 连线距离 line_dist_xy；
        - 物体中心投影是否落在线段中间 alpha；
        - thumb 与 finger 是否在物体两侧 radial_dot；
        - thumb/finger 接触高度差 z_diff；
        - object-support 穿透是否过大；
        - object 位移是否过大。
    6. 只有 force proxy 达标才 latch，并用当前真实 hand qpos 作为 held_ctrl。
    7. latch 后 hold/lift 不再继续猛闭合。
    8. 如果没达到 force proxy，但物体开始被压进支撑块或位移过大，则回退 last_safe_ctrl。

输入：
    --model
        MuJoCo XML 场景。
    --candidate
        candidate JSON，包含 hand.o7_active_ctrl。
    --p3-json
        P3 输出 JSON，读取 q_pre / q_grasp / q_lift。
    --best-config
        P4H / P4E-fast 输出的 best_config，用于读取 thumb preshape。
    --object-body
        物体 body 名称，例如 grasp_can。

输出：
    --out
        JSON 记录 close / hold / lift 的接触、force proxy、held_ctrl、最终结果。
    viewer
        可视化 force-proxy latch close 过程。

当前流程位置：
    P4H force-closure proxy 选出 hand-local best
        -> P4I force-proxy latch close
        -> 如果 force proxy 达标，再冻结并 lift

本脚本不负责：
    1. 不重新做 IK。
    2. 不重新做 P3 碰撞预检。
    3. 不重新做 hand local 几何搜索。
    4. 不做完整 wrench-space force closure。
    5. 不把“单纯接触”当成“已经抓住”。
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

NON_THUMB_GROUPS = ["index", "middle", "ring", "pinky"]


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


def current_hand_qpos_as_ctrl(model, data):
    out = {}
    for j in HAND_JOINTS:
        v = p4f.get_joint_qpos(model, data, j)
        if v is not None:
            out[j] = float(v)
    return out


def set_arm_and_hand(model, data, arm_q, hand_ctrl, direct_hand_qpos):
    p4f.set_arm_qpos_and_ctrl(model, data, arm_q)
    p4f.set_hand_ctrl(model, data, hand_ctrl, direct_hand_qpos)


def object_center(model, data, object_body):
    bid = p4f.body_id(model, object_body)
    if bid < 0:
        raise RuntimeError(f"cannot find object body: {object_body}")
    return data.xpos[bid].copy()


def classify_contact_pair(model, g1, g2, sets):
    object_geoms = sets["object_geoms"]
    hand_geoms = sets["hand_geoms"]
    support_geoms = sets["support_geoms"]
    fr3_geoms = sets["fr3_geoms"]

    g1_obj = g1 in object_geoms
    g2_obj = g2 in object_geoms
    g1_hand = g1 in hand_geoms
    g2_hand = g2 in hand_geoms
    g1_sup = g1 in support_geoms
    g2_sup = g2 in support_geoms
    g1_fr3 = g1 in fr3_geoms
    g2_fr3 = g2 in fr3_geoms

    hand_object = (g1_hand and g2_obj) or (g2_hand and g1_obj)
    object_support = (g1_obj and g2_sup) or (g2_obj and g1_sup)
    hand_support = (g1_hand and g2_sup) or (g2_hand and g1_sup)
    fr3_object = (g1_fr3 and g2_obj) or (g2_fr3 and g1_obj)

    hand_geom = None
    object_geom = None
    support_geom = None

    if hand_object:
        hand_geom = g1 if g1_hand else g2
        object_geom = g1 if g1_obj else g2

    if object_support:
        object_geom = g1 if g1_obj else g2
        support_geom = g1 if g1_sup else g2

    return {
        "hand_object": hand_object,
        "object_support": object_support,
        "hand_support": hand_support,
        "fr3_object": fr3_object,
        "hand_geom": hand_geom,
        "object_geom": object_geom,
        "support_geom": support_geom,
    }


def collect_contacts_with_proxy(model, data, sets, object_body, args):
    c_obj = object_center(model, data, object_body)
    cxy = c_obj[:2]

    thumb_contacts = []
    non_thumb_contacts = []
    object_support_contacts = []
    hand_support_count = 0
    fr3_object_count = 0
    object_groups = {}
    pairs = []

    for i in range(data.ncon):
        con = data.contact[i]
        g1 = int(con.geom1)
        g2 = int(con.geom2)
        cls = classify_contact_pair(model, g1, g2, sets)

        pair = {
            "geom1": p4f.geom_name(model, g1),
            "body1": p4f.geom_body_name(model, g1),
            "geom2": p4f.geom_name(model, g2),
            "body2": p4f.geom_body_name(model, g2),
            "dist": float(con.dist),
            "pos": np.array(con.pos, dtype=float).copy(),
        }
        pairs.append(pair)

        if cls["hand_support"]:
            hand_support_count += 1

        if cls["fr3_object"]:
            fr3_object_count += 1

        if cls["object_support"]:
            object_support_contacts.append(pair)

        if cls["hand_object"]:
            hg = cls["hand_geom"]
            group = p4f.group_of_geom(model, hg)
            if group:
                object_groups[group] = object_groups.get(group, 0) + 1

            item = {
                "group": group,
                "pos": np.array(con.pos, dtype=float).copy(),
                "dist": float(con.dist),
                "hand_geom": p4f.geom_name(model, hg),
                "object_geom": p4f.geom_name(model, cls["object_geom"]),
                "pair": pair,
            }

            if group == "thumb":
                thumb_contacts.append(item)
            elif group in NON_THUMB_GROUPS:
                non_thumb_contacts.append(item)

    min_object_support_dist = None
    if object_support_contacts:
        min_object_support_dist = min(float(x["dist"]) for x in object_support_contacts)

    proxy_candidates = []

    for tc in thumb_contacts:
        for fc in non_thumb_contacts:
            pt = np.asarray(tc["pos"], dtype=float)
            pf = np.asarray(fc["pos"], dtype=float)

            v = pf[:2] - pt[:2]
            vnorm = float(np.linalg.norm(v))
            if vnorm < 1e-9:
                line_dist = 999.0
                alpha = -999.0
            else:
                w = cxy - pt[:2]
                alpha = float(np.dot(w, v) / (vnorm * vnorm))
                cross = abs(v[0] * w[1] - v[1] * w[0])
                line_dist = float(cross / vnorm)

            rt = pt[:2] - cxy
            rf = pf[:2] - cxy
            rt_norm = float(np.linalg.norm(rt))
            rf_norm = float(np.linalg.norm(rf))
            if rt_norm < 1e-9 or rf_norm < 1e-9:
                radial_dot = 1.0
            else:
                radial_dot = float(np.dot(rt / rt_norm, rf / rf_norm))

            z_diff = float(abs(pt[2] - pf[2]))

            alpha_ok = args.force_alpha_min <= alpha <= args.force_alpha_max
            line_ok = line_dist <= args.force_line_tol
            radial_ok = radial_dot <= args.force_radial_dot_max
            z_ok = z_diff <= args.force_z_diff_tol

            score = (
                args.w_force_line * line_dist
                + args.w_force_radial * max(0.0, radial_dot - args.force_radial_dot_target)
                + args.w_force_z * z_diff
                + args.w_force_alpha * (max(0.0, args.force_alpha_min - alpha) + max(0.0, alpha - args.force_alpha_max))
            )

            ok = bool(alpha_ok and line_ok and radial_ok and z_ok)

            proxy_candidates.append({
                "ok": ok,
                "score": float(score),
                "thumb": tc,
                "finger": fc,
                "finger_group": fc["group"],
                "line_dist_xy": line_dist,
                "alpha": alpha,
                "radial_dot": radial_dot,
                "z_diff": z_diff,
                "flags": {
                    "alpha_ok": bool(alpha_ok),
                    "line_ok": bool(line_ok),
                    "radial_ok": bool(radial_ok),
                    "z_ok": bool(z_ok),
                },
            })

    proxy_candidates.sort(key=lambda x: x["score"])
    best_proxy = proxy_candidates[0] if proxy_candidates else None
    force_proxy_ok = bool(best_proxy is not None and best_proxy["ok"])

    if min_object_support_dist is not None and min_object_support_dist < -abs(args.max_support_penetration):
        force_proxy_ok = False

    return {
        "ncon": int(data.ncon),
        "object_groups": object_groups,
        "thumb_contacts": thumb_contacts,
        "non_thumb_contacts": non_thumb_contacts,
        "object_support_contacts": object_support_contacts,
        "min_object_support_dist": min_object_support_dist,
        "hand_support_count": hand_support_count,
        "fr3_object_count": fr3_object_count,
        "pairs": pairs,
        "proxy_candidates": proxy_candidates[:10],
        "best_proxy": best_proxy,
        "force_proxy_ok": force_proxy_ok,
    }


def brief_proxy(proxy):
    bp = proxy.get("best_proxy")
    if not bp:
        return "no_proxy"
    return (
        f"ok={bp['ok']} finger={bp['finger_group']} "
        f"line={bp['line_dist_xy']:.4f} "
        f"alpha={bp['alpha']:.3f} "
        f"radial_dot={bp['radial_dot']:.3f} "
        f"z_diff={bp['z_diff']:.4f}"
    )


def run_phase(model, data, viewer, args, phase, steps, callback, logs, sets, obj_ref):
    log_every = max(1, int(args.log_dt / model.opt.timestep))

    print(f"\n[PHASE] {phase}, steps={steps}")

    for k in range(steps + 1):
        alpha = k / max(1, steps)
        callback(alpha, k)
        mujoco.mj_forward(model, data)

        if k % log_every == 0 or k == steps:
            proxy = collect_contacts_with_proxy(model, data, sets, args.object_body, args)
            obj = p4f.object_pos(model, data, args.object_body)
            row = {
                "phase": phase,
                "step": int(k),
                "alpha": float(alpha),
                "object_pos": obj,
                "object_disp": float(np.linalg.norm(obj - obj_ref)),
                "proxy": proxy,
                "hand_qpos": p4f.hand_qpos(model, data),
            }
            logs.append(row)
            print(
                f"[{phase}] {k:4d}/{steps} "
                f"disp={row['object_disp']:.5f} "
                f"groups={proxy['object_groups']} "
                f"support_dist={proxy['min_object_support_dist']} "
                f"force_ok={proxy['force_proxy_ok']} "
                f"{brief_proxy(proxy)}"
            )

        mujoco.mj_step(model, data)

        if viewer is not None:
            viewer.sync()
            if args.frame_sleep > 0:
                time.sleep(args.frame_sleep)


def force_proxy_latch_close(model, data, viewer, args, q_grasp, side_open_ctrl, close_target, sets, obj_ref, logs):
    close_steps = max(1, int(args.close_duration / model.opt.timestep))
    settle_steps = max(1, int(args.settle_duration / model.opt.timestep))
    log_every = max(1, int(args.log_dt / model.opt.timestep))

    last_safe_ctrl = current_hand_qpos_as_ctrl(model, data)
    held_ctrl = dict(last_safe_ctrl)
    latched = False
    latch_info = None
    stop_reason = ""

    print("\n[PHASE] close_until_force_proxy")

    for phase_name, steps, use_interpolation in [
        ("close_until_force_proxy", close_steps, True),
        ("settle_until_force_proxy", settle_steps, False),
    ]:
        print(f"\n[SUBPHASE] {phase_name}, steps={steps}")

        for k in range(steps + 1):
            alpha = k / max(1, steps)
            if use_interpolation:
                ctrl = interp_dict(side_open_ctrl, close_target, alpha, HAND_JOINTS)
            else:
                ctrl = dict(close_target)

            set_arm_and_hand(model, data, q_grasp, ctrl, args.direct_hand_qpos)
            mujoco.mj_forward(model, data)

            obj = p4f.object_pos(model, data, args.object_body)
            disp = float(np.linalg.norm(obj - obj_ref))
            proxy = collect_contacts_with_proxy(model, data, sets, args.object_body, args)

            safe_support = (
                proxy["min_object_support_dist"] is None
                or proxy["min_object_support_dist"] >= -abs(args.max_support_penetration)
            )
            safe_disp = disp <= args.safe_object_disp

            if safe_support and safe_disp:
                last_safe_ctrl = current_hand_qpos_as_ctrl(model, data)

            if proxy["force_proxy_ok"] and safe_support and disp <= args.hard_object_push_disp:
                latched = True
                held_ctrl = current_hand_qpos_as_ctrl(model, data)
                latch_info = {
                    "phase": phase_name,
                    "step": int(k),
                    "alpha": float(alpha),
                    "ctrl": dict(ctrl),
                    "held_ctrl": dict(held_ctrl),
                    "object_disp": disp,
                    "proxy": proxy,
                }
                print("[LATCH] force proxy reached:", brief_proxy(proxy))
                break

            hard_push = disp > args.hard_object_push_disp
            hard_support = (
                proxy["min_object_support_dist"] is not None
                and proxy["min_object_support_dist"] < -abs(args.hard_support_penetration)
            )

            if hard_push or hard_support:
                stop_reason = "hard_push_or_support_before_force_proxy"
                held_ctrl = dict(last_safe_ctrl)
                print(
                    f"[STOP] before force proxy. "
                    f"disp={disp:.5f}, support_dist={proxy['min_object_support_dist']}, "
                    f"use last_safe_ctrl"
                )
                break

            if k % log_every == 0 or k == steps:
                row = {
                    "phase": phase_name,
                    "step": int(k),
                    "alpha": float(alpha),
                    "ctrl": dict(ctrl),
                    "object_pos": obj,
                    "object_disp": disp,
                    "proxy": proxy,
                    "hand_qpos": p4f.hand_qpos(model, data),
                }
                logs.append(row)
                print(
                    f"[{phase_name}] {k:4d}/{steps} "
                    f"disp={disp:.5f} groups={proxy['object_groups']} "
                    f"support_dist={proxy['min_object_support_dist']} "
                    f"force_ok={proxy['force_proxy_ok']} "
                    f"{brief_proxy(proxy)}"
                )

            mujoco.mj_step(model, data)

            if viewer is not None:
                viewer.sync()
                if args.frame_sleep > 0:
                    time.sleep(args.frame_sleep)

        if latched or stop_reason:
            break

    if not latched and not stop_reason:
        stop_reason = "no_force_proxy_after_close_and_settle"
        held_ctrl = current_hand_qpos_as_ctrl(model, data)

    print("\n========== FORCE PROXY LATCH SUMMARY ==========")
    print("latched:", latched)
    print("stop_reason:", stop_reason)
    print("held_ctrl:", held_ctrl)
    print("latch_info:", latch_info)
    print("===============================================")

    return {
        "latched": bool(latched),
        "stop_reason": stop_reason,
        "held_ctrl": held_ctrl,
        "latch_info": latch_info,
    }


def run(args):
    model_path = resolve_path(args.model)
    candidate_path = resolve_path(args.candidate)
    p3_path = resolve_path(args.p3_json)
    best_config_path = resolve_path(args.best_config) if args.best_config else None

    candidate = load_json(candidate_path)
    p3 = load_json(p3_path)
    best_config = load_json(best_config_path) if best_config_path and best_config_path.exists() else {}

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    plan = p4f.selected_plan(p3, args.which)
    q_pre = plan["q_pre"]
    q_grasp = plan["q_grasp"]
    q_lift = plan["q_lift"]

    candidate_ctrl, candidate_ctrl_source = p4f.extract_candidate_ctrl(candidate, model)
    best_ctrl, best_ctrl_source = p4f.extract_best_config_ctrl(best_config, model)

    open_ctrl = p4f.make_open_ctrl(model)
    side_open_ctrl = p4f.make_side_open_ctrl(model, open_ctrl, candidate_ctrl, best_ctrl, args)
    close_target = p4f.make_close_target(model, side_open_ctrl, candidate_ctrl, args)

    sets = p4f.build_geom_sets(model, args.object_body)

    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    q_current = {j: p4f.get_joint_qpos(model, data, j) or 0.0 for j in ARM_JOINTS}
    obj_ref = p4f.object_pos(model, data, args.object_body)

    logs = []

    print("\n========== V4.12P4I FORCE-PROXY LATCH CLOSE ==========")
    print("model                :", model_path)
    print("candidate            :", candidate_path)
    print("p3_json              :", p3_path)
    print("best_config          :", best_config_path)
    print("candidate_ctrl_source:", candidate_ctrl_source)
    print("best_ctrl_source     :", best_ctrl_source)
    print("side_open_ctrl       :", side_open_ctrl)
    print("close_target         :", close_target)
    print("force_line_tol       :", args.force_line_tol)
    print("force_alpha_range    :", args.force_alpha_min, args.force_alpha_max)
    print("force_radial_dot_max :", args.force_radial_dot_max)
    print("force_z_diff_tol     :", args.force_z_diff_tol)
    print("max_support_pen      :", args.max_support_penetration)
    print("======================================================\n")

    def sequence(viewer=None):
        run_phase(
            model, data, viewer, args,
            "move_to_pre_open",
            args.move_steps,
            lambda a, k: set_arm_and_hand(
                model, data,
                interp_dict(q_current, q_pre, a, ARM_JOINTS),
                open_ctrl,
                args.direct_hand_qpos,
            ),
            logs, sets, obj_ref,
        )

        run_phase(
            model, data, viewer, args,
            "thumb_preshape",
            args.thumb_preshape_steps,
            lambda a, k: set_arm_and_hand(
                model, data,
                q_pre,
                interp_dict(open_ctrl, side_open_ctrl, a, HAND_JOINTS),
                args.direct_hand_qpos,
            ),
            logs, sets, obj_ref,
        )

        run_phase(
            model, data, viewer, args,
            "move_to_grasp_side_open",
            args.move_steps,
            lambda a, k: set_arm_and_hand(
                model, data,
                interp_dict(q_pre, q_grasp, a, ARM_JOINTS),
                side_open_ctrl,
                args.direct_hand_qpos,
            ),
            logs, sets, obj_ref,
        )

        latch = force_proxy_latch_close(
            model, data, viewer, args,
            q_grasp,
            side_open_ctrl,
            close_target,
            sets,
            obj_ref,
            logs,
        )

        held_ctrl = latch["held_ctrl"]

        run_phase(
            model, data, viewer, args,
            "hold_with_force_proxy_ctrl",
            max(1, int(args.hold_duration / model.opt.timestep)),
            lambda a, k: set_arm_and_hand(
                model, data,
                q_grasp,
                held_ctrl,
                args.direct_hand_qpos,
            ),
            logs, sets, obj_ref,
        )

        if latch["latched"] or args.lift_even_if_fail:
            run_phase(
                model, data, viewer, args,
                "lift",
                max(1, int(args.lift_duration / model.opt.timestep)),
                lambda a, k: set_arm_and_hand(
                    model, data,
                    interp_dict(q_grasp, q_lift, a, ARM_JOINTS),
                    held_ctrl,
                    args.direct_hand_qpos,
                ),
                logs, sets, obj_ref,
            )
        else:
            print("[SKIP LIFT] force proxy not reached.")

        return latch

    if args.viewer:
        if mujoco.viewer is None:
            raise RuntimeError("mujoco.viewer is not available")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            latch = sequence(viewer)
            print("[VIEWER] 播放完成。关闭窗口即可退出。")
            if args.keep_viewer_open:
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.05)
    else:
        latch = sequence(None)

    final_obj = p4f.object_pos(model, data, args.object_body)
    final_rise = float(final_obj[2] - obj_ref[2])
    final_proxy = collect_contacts_with_proxy(model, data, sets, args.object_body, args)
    status = "SUCCESS" if latch["latched"] and final_rise >= args.min_lift_rise_success else "FAIL"

    out = {
        "format": "v4_12p4i_force_proxy_latch_close_debug",
        "model": str(model_path),
        "candidate": str(candidate_path),
        "p3_json": str(p3_path),
        "best_config": str(best_config_path) if best_config_path else "",
        "which": args.which,
        "object_body": args.object_body,
        "args": vars(args),
        "candidate_ctrl_source": candidate_ctrl_source,
        "best_ctrl_source": best_ctrl_source,
        "candidate_ctrl": candidate_ctrl,
        "best_ctrl": best_ctrl,
        "open_ctrl": open_ctrl,
        "side_open_ctrl": side_open_ctrl,
        "close_target": close_target,
        "latch": latch,
        "status": status,
        "final_rise": final_rise,
        "final_proxy": final_proxy,
        "final_hand_qpos": p4f.hand_qpos(model, data),
        "logs": logs,
    }

    save_json(args.out, out)

    print("\n========== V4.12P4I FORCE-PROXY RESULT ==========")
    print("status:", status)
    print("latched:", latch["latched"])
    print("stop_reason:", latch["stop_reason"])
    print("final_rise:", final_rise)
    print("final_force_proxy_ok:", final_proxy["force_proxy_ok"])
    print("final_proxy:", brief_proxy(final_proxy))
    print("saved:", resolve_path(args.out))
    print("=================================================\n")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--p3-json", required=True)
    ap.add_argument("--best-config", default="")
    ap.add_argument("--which", default="best_available", choices=["best_available", "best_pass"])
    ap.add_argument("--object-body", default="grasp_can")
    ap.add_argument("--out", required=True)

    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--keep-viewer-open", action="store_true")
    ap.add_argument("--direct-hand-qpos", action="store_true")

    ap.add_argument("--move-steps", type=int, default=100)
    ap.add_argument("--thumb-preshape-steps", type=int, default=100)
    ap.add_argument("--close-duration", type=float, default=1.8)
    ap.add_argument("--settle-duration", type=float, default=0.8)
    ap.add_argument("--hold-duration", type=float, default=0.5)
    ap.add_argument("--lift-duration", type=float, default=2.0)

    ap.add_argument("--finger-close-scale", type=float, default=0.85)
    ap.add_argument("--thumb-pitch-from-finger-gain", type=float, default=0.35)
    ap.add_argument("--preshape-fingers-from-best", action="store_true")

    ap.add_argument("--safe-object-disp", type=float, default=0.006)
    ap.add_argument("--hard-object-push-disp", type=float, default=0.014)
    ap.add_argument("--max-support-penetration", type=float, default=0.003)
    ap.add_argument("--hard-support-penetration", type=float, default=0.008)
    ap.add_argument("--min-lift-rise-success", type=float, default=0.015)
    ap.add_argument("--lift-even-if-fail", action="store_true")

    ap.add_argument("--force-line-tol", type=float, default=0.012)
    ap.add_argument("--force-alpha-min", type=float, default=0.15)
    ap.add_argument("--force-alpha-max", type=float, default=0.85)
    ap.add_argument("--force-radial-dot-max", type=float, default=-0.25)
    ap.add_argument("--force-radial-dot-target", type=float, default=-0.75)
    ap.add_argument("--force-z-diff-tol", type=float, default=0.035)

    ap.add_argument("--w-force-line", type=float, default=10.0)
    ap.add_argument("--w-force-radial", type=float, default=4.0)
    ap.add_argument("--w-force-z", type=float, default=3.0)
    ap.add_argument("--w-force-alpha", type=float, default=4.0)

    ap.add_argument("--log-dt", type=float, default=0.1)
    ap.add_argument("--frame-sleep", type=float, default=0.002)

    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
