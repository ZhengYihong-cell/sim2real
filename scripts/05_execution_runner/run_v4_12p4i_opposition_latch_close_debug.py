#!/usr/bin/env python3
"""
文件名：
    run_v4_12p4i_opposition_latch_close_debug.py

脚本类别：
    debug / runner / opposition-latch-close / support-aware-close / viewer

用途：
    本脚本用于 V4.12P4I 阶段，修复 P4F/P4H 后出现的两个执行问题：
    1. 单独 thumb 一碰就冻结，导致四指尚未形成抓握时就停止有效闭合；
    2. hold 阶段继续猛闭合，导致物体被压进支撑块。

核心思想：
    不再使用“任意手指接触就冻结”的策略。
    而是采用 opposition latch：
        只有当 thumb + 至少一个非拇指 finger 同时接触物体时，
        才认为形成了对握接触，并锁定当前真实 hand qpos 作为 held_ctrl。
    如果还没有形成对握，则继续朝 candidate prior close target 缓慢闭合。
    如果物体位移超过阈值，则回退到上一帧安全 hand qpos，避免继续压物体。

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
        JSON 记录 close / settle / hold / lift 的接触状态、held_ctrl、最终结果。
    viewer
        可视化当前 opposition-latch close 过程。

当前流程位置：
    P4H force-closure proxy 选出 hand-local best
        -> P4I opposition-latch close
        -> 如果能形成 thumb + non-thumb 对握，再 lift

本脚本不负责：
    1. 不重新做 IK。
    2. 不重新做 P3 碰撞预检。
    3. 不重新做几何搜索。
    4. 不单独冻结 thumb。
    5. 不做完整 wrench-space force closure，只执行对握接触锁定。
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


def has_opposition(counts):
    groups = counts.get("object_groups", {})
    has_thumb = "thumb" in groups
    nts = [g for g in NON_THUMB if g in groups]
    return has_thumb and len(nts) > 0, has_thumb, nts


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


def log_row(model, data, phase, step, alpha, object_body, sets, obj_ref):
    obj = p4f.object_pos(model, data, object_body)
    counts = p4f.contact_counts(model, data, sets)
    opp, has_thumb, nts = has_opposition(counts)
    return {
        "phase": phase,
        "step": int(step),
        "alpha": float(alpha),
        "object_pos": obj,
        "object_disp": float(np.linalg.norm(obj - obj_ref)),
        "contacts": counts,
        "opposition": bool(opp),
        "has_thumb": bool(has_thumb),
        "non_thumb_groups": nts,
        "hand_qpos": p4f.hand_qpos(model, data),
    }


def run_phase(model, data, viewer, args, phase, steps, callback, logs, sets, obj_ref):
    log_every = max(1, int(args.log_dt / model.opt.timestep))

    print(f"\n[PHASE] {phase}, steps={steps}")

    for k in range(steps + 1):
        alpha = k / max(1, steps)
        callback(alpha, k)

        mujoco.mj_forward(model, data)

        if k % log_every == 0 or k == steps:
            row = log_row(model, data, phase, k, alpha, args.object_body, sets, obj_ref)
            logs.append(row)
            print(
                f"[{phase}] {k:4d}/{steps} "
                f"disp={row['object_disp']:.5f} "
                f"groups={row['contacts']['object_groups']} "
                f"support={row['contacts']['object_support']} "
                f"opp={row['opposition']} "
                f"hand_qpos={row['hand_qpos']}"
            )

        mujoco.mj_step(model, data)

        if viewer is not None:
            viewer.sync()
            if args.frame_sleep > 0:
                time.sleep(args.frame_sleep)


def opposition_latch_close(model, data, viewer, args, q_grasp, side_open_ctrl, close_target, sets, obj_ref, logs):
    close_steps = max(1, int(args.close_duration / model.opt.timestep))
    settle_steps = max(1, int(args.settle_duration / model.opt.timestep))
    log_every = max(1, int(args.log_dt / model.opt.timestep))

    latched = False
    latch_reason = ""
    held_ctrl = current_hand_qpos_as_ctrl(model, data)
    last_safe_ctrl = dict(held_ctrl)
    first_opposition = None
    first_thumb = None
    first_non_thumb = None

    def maybe_print(phase, k, steps, alpha, ctrl):
        row = log_row(model, data, phase, k, alpha, args.object_body, sets, obj_ref)
        row["ctrl"] = dict(ctrl)
        logs.append(row)
        print(
            f"[{phase}] {k:4d}/{steps} "
            f"disp={row['object_disp']:.5f} "
            f"groups={row['contacts']['object_groups']} "
            f"support={row['contacts']['object_support']} "
            f"opp={row['opposition']} "
            f"ctrl_thumb_pitch={ctrl.get('thumb_cmc_pitch', 0.0):.3f} "
            f"ctrl_four={{"
            f"i:{ctrl.get('index_mcp_pitch', 0.0):.3f}, "
            f"m:{ctrl.get('middle_mcp_pitch', 0.0):.3f}, "
            f"r:{ctrl.get('ring_mcp_pitch', 0.0):.3f}, "
            f"p:{ctrl.get('pinky_mcp_pitch', 0.0):.3f}"
            f"}}"
        )
        return row

    print("\n[PHASE] target_close_until_opposition")

    for k in range(close_steps + 1):
        alpha = k / close_steps
        ctrl = interp_dict(side_open_ctrl, close_target, alpha, HAND_JOINTS)

        set_arm_and_hand(model, data, q_grasp, ctrl, args.direct_hand_qpos)
        mujoco.mj_forward(model, data)

        counts = p4f.contact_counts(model, data, sets)
        opp, has_thumb, nts = has_opposition(counts)
        obj = p4f.object_pos(model, data, args.object_body)
        disp = float(np.linalg.norm(obj - obj_ref))

        if disp <= args.safe_object_disp:
            last_safe_ctrl = current_hand_qpos_as_ctrl(model, data)

        if has_thumb and first_thumb is None:
            first_thumb = {
                "phase": "target_close_until_opposition",
                "step": int(k),
                "alpha": float(alpha),
                "ctrl": dict(ctrl),
                "hand_qpos": current_hand_qpos_as_ctrl(model, data),
                "groups": dict(counts.get("object_groups", {})),
            }

        if nts and first_non_thumb is None:
            first_non_thumb = {
                "phase": "target_close_until_opposition",
                "step": int(k),
                "alpha": float(alpha),
                "ctrl": dict(ctrl),
                "hand_qpos": current_hand_qpos_as_ctrl(model, data),
                "groups": dict(counts.get("object_groups", {})),
            }

        if opp:
            latched = True
            latch_reason = "opposition_in_close"
            held_ctrl = current_hand_qpos_as_ctrl(model, data)
            first_opposition = {
                "phase": "target_close_until_opposition",
                "step": int(k),
                "alpha": float(alpha),
                "ctrl": dict(ctrl),
                "hand_qpos": dict(held_ctrl),
                "groups": dict(counts.get("object_groups", {})),
                "disp": disp,
            }
            print("[LATCH] opposition in close:", first_opposition)
            maybe_print("target_close_until_opposition", k, close_steps, alpha, ctrl)
            break

        if disp > args.hard_object_push_disp:
            latched = False
            latch_reason = "hard_push_before_opposition"
            held_ctrl = dict(last_safe_ctrl)
            print(f"[STOP] hard push before opposition: disp={disp:.5f}, use last_safe_ctrl")
            maybe_print("target_close_until_opposition", k, close_steps, alpha, ctrl)
            break

        if k % log_every == 0 or k == close_steps:
            maybe_print("target_close_until_opposition", k, close_steps, alpha, ctrl)

        mujoco.mj_step(model, data)

        if viewer is not None:
            viewer.sync()
            if args.frame_sleep > 0:
                time.sleep(args.frame_sleep)

    if not latched and latch_reason != "hard_push_before_opposition":
        print("\n[PHASE] settle_until_opposition")
        for k in range(settle_steps + 1):
            alpha = k / settle_steps

            # 继续给 close_target，但不再增加目标，等待执行器慢慢追上。
            ctrl = dict(close_target)
            set_arm_and_hand(model, data, q_grasp, ctrl, args.direct_hand_qpos)
            mujoco.mj_forward(model, data)

            counts = p4f.contact_counts(model, data, sets)
            opp, has_thumb, nts = has_opposition(counts)
            obj = p4f.object_pos(model, data, args.object_body)
            disp = float(np.linalg.norm(obj - obj_ref))

            if disp <= args.safe_object_disp:
                last_safe_ctrl = current_hand_qpos_as_ctrl(model, data)

            if has_thumb and first_thumb is None:
                first_thumb = {
                    "phase": "settle_until_opposition",
                    "step": int(k),
                    "alpha": float(alpha),
                    "ctrl": dict(ctrl),
                    "hand_qpos": current_hand_qpos_as_ctrl(model, data),
                    "groups": dict(counts.get("object_groups", {})),
                }

            if nts and first_non_thumb is None:
                first_non_thumb = {
                    "phase": "settle_until_opposition",
                    "step": int(k),
                    "alpha": float(alpha),
                    "ctrl": dict(ctrl),
                    "hand_qpos": current_hand_qpos_as_ctrl(model, data),
                    "groups": dict(counts.get("object_groups", {})),
                }

            if opp:
                latched = True
                latch_reason = "opposition_in_settle"
                held_ctrl = current_hand_qpos_as_ctrl(model, data)
                first_opposition = {
                    "phase": "settle_until_opposition",
                    "step": int(k),
                    "alpha": float(alpha),
                    "ctrl": dict(ctrl),
                    "hand_qpos": dict(held_ctrl),
                    "groups": dict(counts.get("object_groups", {})),
                    "disp": disp,
                }
                print("[LATCH] opposition in settle:", first_opposition)
                maybe_print("settle_until_opposition", k, settle_steps, alpha, ctrl)
                break

            if disp > args.hard_object_push_disp:
                latch_reason = "hard_push_in_settle"
                held_ctrl = dict(last_safe_ctrl)
                print(f"[STOP] hard push in settle: disp={disp:.5f}, use last_safe_ctrl")
                maybe_print("settle_until_opposition", k, settle_steps, alpha, ctrl)
                break

            if k % log_every == 0 or k == settle_steps:
                maybe_print("settle_until_opposition", k, settle_steps, alpha, ctrl)

            mujoco.mj_step(model, data)

            if viewer is not None:
                viewer.sync()
                if args.frame_sleep > 0:
                    time.sleep(args.frame_sleep)

    if not latched and latch_reason == "":
        latch_reason = "no_opposition_after_close_and_settle"
        held_ctrl = current_hand_qpos_as_ctrl(model, data)

    print("\n========== OPPOSITION LATCH SUMMARY ==========")
    print("latched:", latched)
    print("latch_reason:", latch_reason)
    print("held_ctrl:", held_ctrl)
    print("first_thumb:", first_thumb)
    print("first_non_thumb:", first_non_thumb)
    print("first_opposition:", first_opposition)
    print("==============================================")

    return {
        "latched": bool(latched),
        "latch_reason": latch_reason,
        "held_ctrl": held_ctrl,
        "first_thumb": first_thumb,
        "first_non_thumb": first_non_thumb,
        "first_opposition": first_opposition,
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

    print("\n========== V4.12P4I OPPOSITION-LATCH CLOSE ==========")
    print("model                :", model_path)
    print("candidate            :", candidate_path)
    print("p3_json              :", p3_path)
    print("best_config          :", best_config_path)
    print("candidate_ctrl_source:", candidate_ctrl_source)
    print("best_ctrl_source     :", best_ctrl_source)
    print("side_open_ctrl       :", side_open_ctrl)
    print("close_target         :", close_target)
    print("finger_close_scale   :", args.finger_close_scale)
    print("thumb_pitch_gain     :", args.thumb_pitch_from_finger_gain)
    print("safe_object_disp     :", args.safe_object_disp)
    print("hard_object_push_disp:", args.hard_object_push_disp)
    print("=====================================================\n")

    def run_sequence(viewer=None):
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

        latch = opposition_latch_close(
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
            "hold_with_latched_ctrl",
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
            print("[SKIP LIFT] no opposition latch and lift_even_if_fail=False")

        return latch

    if args.viewer:
        if mujoco.viewer is None:
            raise RuntimeError("mujoco.viewer is not available")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            latch = run_sequence(viewer)
            print("[VIEWER] 播放完成。关闭窗口即可退出。")
            if args.keep_viewer_open:
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.05)
    else:
        latch = run_sequence(None)

    final_obj = p4f.object_pos(model, data, args.object_body)
    final_rise = float(final_obj[2] - obj_ref[2])
    final_counts = p4f.contact_counts(model, data, sets)
    final_opp, final_has_thumb, final_nts = has_opposition(final_counts)

    status = "SUCCESS" if latch["latched"] and final_rise >= args.min_lift_rise_success else "FAIL"

    out = {
        "format": "v4_12p4i_opposition_latch_close_debug",
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
        "final_counts": final_counts,
        "final_opposition": bool(final_opp),
        "final_non_thumb_groups": final_nts,
        "final_hand_qpos": p4f.hand_qpos(model, data),
        "logs": logs,
    }

    save_json(args.out, out)

    print("\n========== V4.12P4I RESULT ==========")
    print("status:", status)
    print("latched:", latch["latched"])
    print("latch_reason:", latch["latch_reason"])
    print("final_rise:", final_rise)
    print("final_counts:", final_counts)
    print("final_opposition:", final_opp, final_nts)
    print("saved:", resolve_path(args.out))
    print("=====================================\n")


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
    ap.add_argument("--min-lift-rise-success", type=float, default=0.015)
    ap.add_argument("--lift-even-if-fail", action="store_true")

    ap.add_argument("--log-dt", type=float, default=0.1)
    ap.add_argument("--frame-sleep", type=float, default=0.002)

    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
