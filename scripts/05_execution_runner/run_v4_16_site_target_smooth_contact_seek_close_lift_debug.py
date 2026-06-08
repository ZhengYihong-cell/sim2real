#!/usr/bin/env python3
"""
脚本类型：
    debug / execution-runner / v4.16 / site-target-smooth-contact-seek-close-lift

用途：
    修正 V4.15 的错误动态逻辑：
        1. 不再从 home 瞬移到 q_grasp；
        2. 先用数据集先验得到 dataset_hand_base_debug 的目标位姿；
        3. 生成 pregrasp，并平滑运动到 pregrasp；
        4. 再从 pregrasp 平滑接近 grasp，接触物体或物体明显被推动时停止 approach；
        5. close 阶段只冻结碰到 support/垫块的手指；
        6. 碰到 object 的手指不冻结，继续参与夹持；
        7. 形成 thumb + 至少一根非拇指接触后尝试 lift。

输入：
    --model          推荐 scene_v415_stiff_contact.xml
    --npy            object.npy
    --sample-index   valid local sample index
    --object-body    grasp_object
    --target-site    dataset_hand_base_debug

输出：
    out_dir/result.json
    out_dir/terminal.txt

当前流程位置：
    V4.15 stiff-contact scene 已修正 object-support 动态穿透；
    本脚本用于替代错误的 V4.15 runner，验证 smooth approach + support-aware close/lift。

不负责：
    1. 不走旧 fr3_link7 target；
    2. 不做全局 RRT/Pinocchio 路径规划；
    3. 不做 selector；
    4. 不换 sample；
    5. 不做沿某个固定轴的人工微调。
"""

from pathlib import Path
import argparse
import importlib.util
import json
import time
import numpy as np
import mujoco
import mujoco.viewer


PROJECT = Path.home() / "Projects/o7_mujoco_sim"
HELPER = PROJECT / "scripts/05_execution_runner/run_v4_15_site_target_contact_aware_close_lift_debug.py"


spec = importlib.util.spec_from_file_location("v415_helper", str(HELPER))
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)


def live_sync(viewer, sleep_s):
    if viewer is not None and viewer.is_running():
        viewer.sync()
        time.sleep(float(sleep_s))


def real_object_groups(state, contact_dist):
    groups = {}
    for c in state.get("object_contacts", []):
        if float(c.get("dist", 999.0)) <= contact_dist:
            g = c.get("group")
            groups[g] = groups.get(g, 0) + 1

    for g, row in state.get("object_min_distance", {}).items():
        if float(row.get("distance", 999.0)) <= contact_dist:
            groups[g] = max(groups.get(g, 0), 1)

    return groups


def ready_from_groups(groups):
    return ("thumb" in groups) and any(g in groups for g in H.NON_THUMB)


def interp_hand(a, b, alpha):
    out = {}
    for j in H.O7_ACTIVE_JOINTS:
        v0 = float(a.get(j, b.get(j, 0.0)))
        v1 = float(b.get(j, v0))
        out[j] = (1.0 - alpha) * v0 + alpha * v1
    return out


def support_contact_groups(state):
    groups = {}
    for c in state.get("support_contacts", []):
        g = c.get("group")
        groups[g] = groups.get(g, 0) + 1
    return groups


def move_arm_smooth(model, data, viewer, args, q_from, q_to, hand_ctrl,
                    object_start, object_body, phase,
                    stop_on_contact=False):
    stop_reason = None
    last_q = dict(q_from)
    last_state = None

    for k in range(args.steps_move):
        alpha = k / max(1, args.steps_move - 1)
        q_cmd = H.interp_arm(q_from, q_to, alpha)

        H.apply_ctrl(model, data, q_cmd, hand_ctrl)
        mujoco.mj_step(model, data)
        live_sync(viewer, args.live_sleep)

        last_q = H.get_joint_values(model, data, H.ARM_JOINTS)
        st = H.contact_state(model, data, object_body)
        last_state = st

        obj_p = H.object_pos(model, data, object_body)
        obj_disp = float(np.linalg.norm(obj_p - object_start))
        obj_groups = real_object_groups(st, args.object_contact_dist)
        sup_groups = support_contact_groups(st)

        if k % args.print_every == 0 or k == args.steps_move - 1:
            print(
                f"[{phase}] {k}/{args.steps_move} alpha={alpha:.3f} "
                f"obj_disp={obj_disp:.5f} obj={obj_groups} support={sup_groups}"
            )

        if stop_on_contact:
            if obj_disp > args.approach_abort_disp:
                stop_reason = f"object_pushed_disp_{obj_disp:.5f}"
                print(f"[APPROACH_STOP] {stop_reason}")
                break

            if sup_groups:
                stop_reason = f"support_contact_{sup_groups}"
                print(f"[APPROACH_STOP] {stop_reason}")
                break

            if obj_groups:
                stop_reason = f"object_contact_{obj_groups}"
                print(f"[APPROACH_STOP] {stop_reason}")
                break

    return {
        "q_stop": last_q,
        "state_stop": last_state,
        "stop_reason": stop_reason,
    }


def close_support_aware(model, data, viewer, args, q_hold, side_open, close_ctrl,
                        object_start, object_body):
    group_alpha = {g: 0.0 for g in H.FINGER_GROUP_TO_JOINTS}
    frozen = {g: False for g in H.FINGER_GROUP_TO_JOINTS}
    freeze_reason = {}
    hand_ctrl = dict(side_open)
    stable = 0
    max_stable = 0
    rows = []

    print("\n[PHASE] support-aware close, object contact does NOT freeze fingers")

    for k in range(args.steps_close):
        st_before = H.contact_state(model, data, object_body)
        sup_groups = support_contact_groups(st_before)

        for g in H.FINGER_GROUP_TO_JOINTS:
            if frozen[g]:
                continue
            if g in sup_groups:
                frozen[g] = True
                freeze_reason[g] = {
                    "reason": "support_contact",
                    "support_groups": sup_groups,
                    "step": k,
                }
                continue

            group_alpha[g] = min(1.0, group_alpha[g] + 1.0 / max(1, args.steps_close))

        hand_ctrl = dict(side_open)
        for g, joints in H.FINGER_GROUP_TO_JOINTS.items():
            a = float(group_alpha[g])
            for j in joints:
                v0 = float(side_open.get(j, close_ctrl.get(j, 0.0)))
                v1 = float(close_ctrl.get(j, v0))
                hand_ctrl[j] = (1.0 - a) * v0 + a * v1

        H.apply_ctrl(model, data, q_hold, hand_ctrl)
        mujoco.mj_step(model, data)
        live_sync(viewer, args.live_sleep)

        st = H.contact_state(model, data, object_body)
        obj_groups = real_object_groups(st, args.object_contact_dist)
        sup_groups = support_contact_groups(st)

        ready = ready_from_groups(obj_groups)
        stable = stable + 1 if ready else 0
        max_stable = max(max_stable, stable)

        obj_p = H.object_pos(model, data, object_body)
        obj_disp = float(np.linalg.norm(obj_p - object_start))

        if k % args.print_every == 0 or k == args.steps_close - 1:
            print(
                f"[close] {k}/{args.steps_close} alpha={group_alpha} frozen={frozen} "
                f"obj={obj_groups} support={sup_groups} disp={obj_disp:.5f} "
                f"ready={ready} stable={stable}"
            )

        rows.append({
            "phase": "support_aware_close",
            "step": k,
            "object_pos": obj_p.tolist(),
            "object_disp": obj_disp,
            "object_groups": obj_groups,
            "support_groups": sup_groups,
            "group_alpha": dict(group_alpha),
            "frozen": dict(frozen),
            "freeze_reason": dict(freeze_reason),
            "hand_ctrl": dict(hand_ctrl),
            "ready": bool(ready),
            "stable": int(stable),
        })

        if obj_disp > args.close_abort_disp:
            print(f"[CLOSE_ABORT] object pushed too much: {obj_disp:.5f}")
            break

        if stable >= args.ready_stable_steps:
            print(f"[READY] stable={stable}; close stops and lift starts.")
            break

    return {
        "hand_ctrl": hand_ctrl,
        "group_alpha": group_alpha,
        "frozen": frozen,
        "freeze_reason": freeze_reason,
        "stable": stable,
        "max_stable": max_stable,
        "rows": rows,
        "last_state": H.contact_state(model, data, object_body),
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--npy", required=True)
    ap.add_argument("--sample-index", type=int, required=True)
    ap.add_argument("--object-body", default="grasp_object")
    ap.add_argument("--target-site", default="dataset_hand_base_debug")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--live-sleep", type=float, default=0.002)

    ap.add_argument("--pregrasp-dist", type=float, default=0.09)
    ap.add_argument("--lift-z", type=float, default=0.09)

    ap.add_argument("--steps-settle", type=int, default=1000)
    ap.add_argument("--steps-move", type=int, default=900)
    ap.add_argument("--steps-close", type=int, default=900)
    ap.add_argument("--steps-post-hold", type=int, default=250)
    ap.add_argument("--steps-lift", type=int, default=1400)
    ap.add_argument("--steps-final-hold", type=int, default=400)

    ap.add_argument("--approach-abort-disp", type=float, default=0.010)
    ap.add_argument("--close-abort-disp", type=float, default=0.018)
    ap.add_argument("--object-contact-dist", type=float, default=0.0045)
    ap.add_argument("--ready-stable-steps", type=int, default=5)
    ap.add_argument("--print-every", type=int, default=100)
    args = ap.parse_args()

    model_path = H.resolve(args.model)
    npy_path = H.resolve(args.npy)
    out_dir = H.resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    sample = H.load_sample(npy_path, args.sample_index)
    T_object_hand = H.sample_T_object_hand(sample)
    close_ctrl = H.sample_ctrl(sample)
    side_open = H.side_open_from_close(close_ctrl)

    print("========== V4.16 SITE-TARGET SMOOTH CONTACT-SEEK CLOSE/LIFT ==========")
    print("model       :", H.rel(model_path))
    print("npy         :", H.rel(npy_path))
    print("sample_index:", args.sample_index)
    print("object_body :", args.object_body)
    print("target_site :", args.target_site)

    H.set_qpos_once(model, data, H.Q_HOME, side_open)
    viewer = None
    if args.viewer:
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.lookat[:] = [0.455, 0.0, 0.30]
        viewer.cam.distance = 0.75
        viewer.cam.azimuth = 130
        viewer.cam.elevation = -25

    print("\n[PHASE] settle object")
    for k in range(args.steps_settle):
        H.apply_ctrl(model, data, H.Q_HOME, side_open)
        mujoco.mj_step(model, data)
        live_sync(viewer, args.live_sleep)

        if k % args.print_every == 0 or k == args.steps_settle - 1:
            print(f"[settle] {k}/{args.steps_settle} object_pos={H.object_pos(model, data, args.object_body).tolist()}")

    object_start = H.object_pos(model, data, args.object_body)
    T_world_object = H.body_world_T(model, data, args.object_body)
    T_grasp = T_world_object @ T_object_hand

    approach_dir = T_grasp[:3, 3] - T_world_object[:3, 3]
    n = float(np.linalg.norm(approach_dir))
    if n < 1e-8:
        approach_dir = np.array([0.0, 0.0, 1.0])
    else:
        approach_dir = approach_dir / n

    T_pre = np.array(T_grasp, dtype=float)
    T_pre[:3, 3] = T_grasp[:3, 3] + approach_dir * float(args.pregrasp_dist)

    T_lift = np.array(T_grasp, dtype=float)
    T_lift[2, 3] += float(args.lift_z)

    print("\n[IK]")
    ik_pre = H.solve_site_ik(model, args.target_site, T_pre, H.Q_HOME)
    ik_grasp = H.solve_site_ik(model, args.target_site, T_grasp, ik_pre["q_arm"])
    ik_lift = H.solve_site_ik(model, args.target_site, T_lift, ik_grasp["q_arm"])

    print("ik_pre   :", ik_pre["success"], ik_pre["final_pos_err_norm"], ik_pre["final_rot_err_norm"])
    print("ik_grasp :", ik_grasp["success"], ik_grasp["final_pos_err_norm"], ik_grasp["final_rot_err_norm"])
    print("ik_lift  :", ik_lift["success"], ik_lift["final_pos_err_norm"], ik_lift["final_rot_err_norm"])

    if not (ik_pre["success"] and ik_grasp["success"] and ik_lift["success"]):
        raise RuntimeError("IK failed; do not execute dynamic close/lift.")

    q_current = H.get_joint_values(model, data, H.ARM_JOINTS)

    print("\n[PHASE] smooth move home -> pregrasp")
    pre_move = move_arm_smooth(
        model, data, viewer, args,
        q_current, ik_pre["q_arm"],
        side_open,
        object_start,
        args.object_body,
        "move_to_pregrasp",
        stop_on_contact=False,
    )

    print("\n[PHASE] smooth approach pregrasp -> grasp, stop on first real contact/push")
    approach = move_arm_smooth(
        model, data, viewer, args,
        pre_move["q_stop"], ik_grasp["q_arm"],
        side_open,
        object_start,
        args.object_body,
        "approach_to_grasp",
        stop_on_contact=True,
    )

    q_hold = approach["q_stop"]
    print("approach_stop_reason:", approach["stop_reason"])
    print("q_hold:", q_hold)

    close_info = close_support_aware(
        model, data, viewer, args,
        q_hold, side_open, close_ctrl,
        object_start, args.object_body,
    )

    grip_ready = close_info["stable"] >= args.ready_stable_steps and ready_from_groups(
        real_object_groups(close_info["last_state"], args.object_contact_dist)
    )

    print("\n[PHASE] post-close hold")
    stable = close_info["stable"]
    for k in range(args.steps_post_hold):
        H.apply_ctrl(model, data, q_hold, close_info["hand_ctrl"])
        mujoco.mj_step(model, data)
        live_sync(viewer, args.live_sleep)

        st = H.contact_state(model, data, args.object_body)
        obj_groups = real_object_groups(st, args.object_contact_dist)
        ready = ready_from_groups(obj_groups)
        stable = stable + 1 if ready else 0

        if k % args.print_every == 0 or k == args.steps_post_hold - 1:
            print(f"[post] {k}/{args.steps_post_hold} obj={obj_groups} ready={ready} stable={stable}")

    grip_ready = stable >= args.ready_stable_steps

    lifted = False
    lift_rows = []

    if grip_ready:
        print("\n[PHASE] lift with fixed hand ctrl")
        q_lift = ik_lift["q_arm"]
        for k in range(args.steps_lift):
            alpha = k / max(1, args.steps_lift - 1)
            q_cmd = H.interp_arm(q_hold, q_lift, alpha)
            H.apply_ctrl(model, data, q_cmd, close_info["hand_ctrl"])
            mujoco.mj_step(model, data)
            live_sync(viewer, args.live_sleep)

            st = H.contact_state(model, data, args.object_body)
            obj_groups = real_object_groups(st, args.object_contact_dist)
            obj_p = H.object_pos(model, data, args.object_body)
            rise = float(obj_p[2] - object_start[2])

            if k % args.print_every == 0 or k == args.steps_lift - 1:
                print(f"[lift] {k}/{args.steps_lift} alpha={alpha:.3f} rise={rise:.5f} obj={obj_groups}")

            lift_rows.append({
                "phase": "lift",
                "step": k,
                "alpha": alpha,
                "object_pos": obj_p.tolist(),
                "rise": rise,
                "object_groups": obj_groups,
            })

        lifted = True

        print("\n[PHASE] final hold")
        for k in range(args.steps_final_hold):
            H.apply_ctrl(model, data, ik_lift["q_arm"], close_info["hand_ctrl"])
            mujoco.mj_step(model, data)
            live_sync(viewer, args.live_sleep)

            if k % args.print_every == 0 or k == args.steps_final_hold - 1:
                print(f"[final] {k}/{args.steps_final_hold} object_pos={H.object_pos(model, data, args.object_body).tolist()}")
    else:
        print("[NO_LIFT] grip_ready is false, so lift is not executed.")

    final_pos = H.object_pos(model, data, args.object_body)
    final_state = H.contact_state(model, data, args.object_body)
    final_groups = real_object_groups(final_state, args.object_contact_dist)

    final_rise = float(final_pos[2] - object_start[2])
    final_disp = float(np.linalg.norm(final_pos - object_start))
    success = bool(lifted and final_rise > 0.03 and ready_from_groups(final_groups))

    result = {
        "format": "v4_16_site_target_smooth_contact_seek_close_lift_debug_v1",
        "model": H.rel(model_path),
        "npy": H.rel(npy_path),
        "sample_index_valid_local": args.sample_index,
        "object_body": args.object_body,
        "target_site": args.target_site,
        "T_world_object_after_settle": H.mat_to_dict(T_world_object),
        "T_object_hand_base_from_dataset": H.mat_to_dict(T_object_hand),
        "T_pre": H.mat_to_dict(T_pre),
        "T_grasp": H.mat_to_dict(T_grasp),
        "T_lift": H.mat_to_dict(T_lift),
        "approach_dir": approach_dir.tolist(),
        "ik_pre": ik_pre,
        "ik_grasp": ik_grasp,
        "ik_lift": ik_lift,
        "approach_stop_reason": approach["stop_reason"],
        "q_hold": q_hold,
        "side_open_ctrl": side_open,
        "close_ctrl": close_ctrl,
        "final_hand_ctrl": close_info["hand_ctrl"],
        "group_alpha": close_info["group_alpha"],
        "frozen": close_info["frozen"],
        "freeze_reason": close_info["freeze_reason"],
        "grip_ready": grip_ready,
        "lifted": lifted,
        "success": success,
        "object_start": object_start.tolist(),
        "final_object_pos": final_pos.tolist(),
        "final_object_rise": final_rise,
        "final_object_disp": final_disp,
        "final_object_groups": final_groups,
        "final_state": final_state,
        "close_rows": close_info["rows"],
        "lift_rows": lift_rows,
    }

    H.save_json(out_dir / "result.json", result)

    print("\n========== V4.16 RESULT ==========")
    print("out:", H.rel(out_dir / "result.json"))
    print("approach_stop_reason:", approach["stop_reason"])
    print("grip_ready:", grip_ready)
    print("lifted:", lifted)
    print("success:", success)
    print("final_object_rise:", final_rise)
    print("final_object_disp:", final_disp)
    print("final_object_groups:", final_groups)
    print("frozen:", close_info["frozen"])
    print("freeze_reason:", close_info["freeze_reason"])
    print("==================================")

    if args.viewer:
        print("[VIEWER] live run finished. Keep open; close viewer or Ctrl+C.")
        while viewer is not None and viewer.is_running():
            live_sync(viewer, args.live_sleep)


if __name__ == "__main__":
    main()
