#!/usr/bin/env python3
"""
脚本类型：
    debug / execution-runner / v4.18 / exact-qgrasp-real-contact-close-lift

用途：
    在 V4.17 exact site q_grasp 逻辑正确的基础上，修正两个问题：
        1. 不再把 MuJoCo margin 内的 support contact 直接当成真实碰撞；
        2. 只有 support contact 的 dist <= support-freeze-dist 时才冻结对应手指。

    本脚本保持：
        T_world_dataset_hand_base_debug = T_world_object @ T_object_hand_base_link
        site IK -> q_grasp
        平滑运动到 q_grasp
        在 q_grasp 上闭合
        碰到垫块的手指冻结
        碰到物体的手指不冻结，继续夹持
        ready 后 lift

输入：
    --model          推荐 scene_v418_low_margin_contact.xml
    --npy            object.npy
    --sample-index   valid local sample index
    --object-body    grasp_object
    --target-site    dataset_hand_base_debug

输出：
    out_dir/result.json
    out_dir/terminal.txt

当前流程位置：
    替代 V4.17 中“margin contact 提前冻结”的问题；
    不负责全局路径规划，不负责换 sample，不负责修改数据集先验。
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
HELPER = PROJECT / "scripts/05_execution_runner/run_v4_17_exact_site_qgrasp_close_lift_debug.py"

spec = importlib.util.spec_from_file_location("v417_helper", str(HELPER))
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)


def live_sync(viewer, sleep_s):
    if viewer is not None and viewer.is_running():
        viewer.sync()
        time.sleep(float(sleep_s))


def real_support_groups(state, freeze_dist):
    groups = {}
    contacts = []
    for c in state.get("support_contacts", []):
        d = float(c.get("dist", 999.0))
        if d <= freeze_dist:
            g = c.get("group")
            groups[g] = groups.get(g, 0) + 1
            contacts.append(c)
    return groups, contacts


def real_object_groups(state, ready_dist):
    groups = {}
    contacts = []
    for c in state.get("object_contacts", []):
        d = float(c.get("dist", 999.0))
        if d <= ready_dist:
            g = c.get("group")
            groups[g] = groups.get(g, 0) + 1
            contacts.append(c)
    return groups, contacts


def ready_from_groups(groups):
    return ("thumb" in groups) and any(g in groups for g in H.NON_THUMB)


def step_ctrl(model, data, q_arm, hand_ctrl, viewer, live_sleep):
    H.apply_ctrl(model, data, q_arm, hand_ctrl)
    mujoco.mj_step(model, data)
    live_sync(viewer, live_sleep)


def move_to_qgrasp(model, data, viewer, args, q_start, q_grasp, hand_ctrl, object_start):
    rows = []
    print("\n[PHASE] smooth move current -> exact q_grasp")

    for k in range(args.move_steps):
        alpha = k / max(1, args.move_steps - 1)
        q_cmd = H.interp_dict(q_start, q_grasp, alpha, H.ARM_JOINTS)
        step_ctrl(model, data, q_cmd, hand_ctrl, viewer, args.live_sleep)

        st = H.contact_state(model, data, args.object_body)
        real_sup, _ = real_support_groups(st, args.support_freeze_dist)
        obj_groups, _ = real_object_groups(st, args.object_ready_dist)

        obj_p = H.object_pos(model, data, args.object_body)
        obj_disp = float(np.linalg.norm(obj_p - object_start))

        if k % args.print_every == 0 or k == args.move_steps - 1:
            print(
                f"[move] {k}/{args.move_steps} alpha={alpha:.3f} "
                f"obj_disp={obj_disp:.5f} obj={obj_groups} "
                f"support_margin={st['support_groups']} support_real={real_sup}"
            )

        rows.append({
            "phase": "move_to_exact_qgrasp",
            "step": k,
            "alpha": alpha,
            "object_pos": obj_p.tolist(),
            "object_disp": obj_disp,
            "object_groups": obj_groups,
            "support_margin_groups": st["support_groups"],
            "support_real_groups": real_sup,
        })

    return rows


def hold_at_q(model, data, viewer, args, q_arm, hand_ctrl, object_start, phase, steps):
    rows = []
    stable = 0
    max_stable = 0

    print(f"\n[PHASE] {phase}")
    for k in range(steps):
        step_ctrl(model, data, q_arm, hand_ctrl, viewer, args.live_sleep)

        st = H.contact_state(model, data, args.object_body)
        obj_groups, _ = real_object_groups(st, args.object_ready_dist)
        real_sup, _ = real_support_groups(st, args.support_freeze_dist)
        ready = ready_from_groups(obj_groups)

        stable = stable + 1 if ready else 0
        max_stable = max(max_stable, stable)

        obj_p = H.object_pos(model, data, args.object_body)
        obj_disp = float(np.linalg.norm(obj_p - object_start))

        if k % args.print_every == 0 or k == steps - 1:
            print(
                f"[{phase}] {k}/{steps} obj_disp={obj_disp:.5f} "
                f"obj={obj_groups} support_margin={st['support_groups']} "
                f"support_real={real_sup} ready={ready} stable={stable}"
            )

        rows.append({
            "phase": phase,
            "step": k,
            "object_pos": obj_p.tolist(),
            "object_disp": obj_disp,
            "object_groups": obj_groups,
            "support_margin_groups": st["support_groups"],
            "support_real_groups": real_sup,
            "ready": bool(ready),
            "stable": int(stable),
        })

    return rows, stable, max_stable


def support_aware_close_real(model, data, viewer, args, q_grasp, side_open, close_ctrl, object_start):
    group_alpha = {g: 0.0 for g in H.FINGER_GROUP_TO_JOINTS}
    frozen = {g: False for g in H.FINGER_GROUP_TO_JOINTS}
    freeze_reason = {}
    hand_ctrl = dict(side_open)

    rows = []
    stable = 0
    max_stable = 0

    print("\n[PHASE] close at q_grasp, freeze only real support contact")

    for k in range(args.close_steps):
        st_before = H.contact_state(model, data, args.object_body)
        real_sup_before, real_sup_contacts_before = real_support_groups(st_before, args.support_freeze_dist)

        for g in H.FINGER_GROUP_TO_JOINTS:
            if frozen[g]:
                continue

            if g in real_sup_before:
                frozen[g] = True
                freeze_reason[g] = {
                    "reason": "real_support_contact_freeze",
                    "step": k,
                    "support_freeze_dist": args.support_freeze_dist,
                    "support_contacts": [c for c in real_sup_contacts_before if c.get("group") == g],
                }
                continue

            group_alpha[g] = min(1.0, group_alpha[g] + 1.0 / max(1, args.close_steps))

        hand_ctrl = dict(side_open)
        for g, joints in H.FINGER_GROUP_TO_JOINTS.items():
            a = float(group_alpha[g])
            for j in joints:
                v0 = float(side_open.get(j, close_ctrl.get(j, 0.0)))
                v1 = float(close_ctrl.get(j, v0))
                hand_ctrl[j] = (1.0 - a) * v0 + a * v1

        step_ctrl(model, data, q_grasp, hand_ctrl, viewer, args.live_sleep)

        st = H.contact_state(model, data, args.object_body)
        obj_groups, obj_contacts = real_object_groups(st, args.object_ready_dist)
        real_sup, real_sup_contacts = real_support_groups(st, args.support_freeze_dist)

        ready = ready_from_groups(obj_groups)
        stable = stable + 1 if ready else 0
        max_stable = max(max_stable, stable)

        obj_p = H.object_pos(model, data, args.object_body)
        obj_disp = float(np.linalg.norm(obj_p - object_start))

        if k % args.print_every == 0 or k == args.close_steps - 1:
            print(
                f"[close] {k}/{args.close_steps} alpha={group_alpha} frozen={frozen} "
                f"obj_disp={obj_disp:.5f} obj={obj_groups} "
                f"support_margin={st['support_groups']} support_real={real_sup} "
                f"ready={ready} stable={stable}"
            )

        rows.append({
            "phase": "real_contact_support_aware_close",
            "step": k,
            "object_pos": obj_p.tolist(),
            "object_disp": obj_disp,
            "object_groups": obj_groups,
            "object_contacts": obj_contacts,
            "support_margin_groups": st["support_groups"],
            "support_real_groups": real_sup,
            "support_real_contacts": real_sup_contacts,
            "group_alpha": dict(group_alpha),
            "frozen": dict(frozen),
            "freeze_reason": dict(freeze_reason),
            "hand_ctrl": dict(hand_ctrl),
            "ready": bool(ready),
            "stable": int(stable),
        })

        if stable >= args.ready_stable_steps:
            print(f"[READY] stable={stable}; stop close and start lift.")
            break

    return {
        "rows": rows,
        "stable": stable,
        "max_stable": max_stable,
        "group_alpha": group_alpha,
        "frozen": frozen,
        "freeze_reason": freeze_reason,
        "hand_ctrl": hand_ctrl,
        "last_state": H.contact_state(model, data, args.object_body),
    }


def lift_with_fixed_grip(model, data, viewer, args, q_grasp, q_lift, hand_ctrl, object_start):
    rows = []
    stable = 0
    max_stable = 0

    print("\n[PHASE] lift with fixed hand ctrl")
    for k in range(args.lift_steps):
        alpha = k / max(1, args.lift_steps - 1)
        q_cmd = H.interp_dict(q_grasp, q_lift, alpha, H.ARM_JOINTS)
        step_ctrl(model, data, q_cmd, hand_ctrl, viewer, args.live_sleep)

        st = H.contact_state(model, data, args.object_body)
        obj_groups, _ = real_object_groups(st, args.object_ready_dist)
        real_sup, _ = real_support_groups(st, args.support_freeze_dist)
        ready = ready_from_groups(obj_groups)

        stable = stable + 1 if ready else 0
        max_stable = max(max_stable, stable)

        obj_p = H.object_pos(model, data, args.object_body)
        rise = float(obj_p[2] - object_start[2])
        disp = float(np.linalg.norm(obj_p - object_start))

        if k % args.print_every == 0 or k == args.lift_steps - 1:
            print(
                f"[lift] {k}/{args.lift_steps} alpha={alpha:.3f} "
                f"rise={rise:.5f} disp={disp:.5f} obj={obj_groups} "
                f"support_real={real_sup} ready={ready} stable={stable}"
            )

        rows.append({
            "phase": "lift",
            "step": k,
            "alpha": alpha,
            "object_pos": obj_p.tolist(),
            "rise": rise,
            "disp": disp,
            "object_groups": obj_groups,
            "support_real_groups": real_sup,
            "ready": bool(ready),
            "stable": int(stable),
        })

    return rows, stable, max_stable


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

    ap.add_argument("--settle-steps", type=int, default=1000)
    ap.add_argument("--move-steps", type=int, default=1200)
    ap.add_argument("--grasp-hold-steps", type=int, default=300)
    ap.add_argument("--close-steps", type=int, default=900)
    ap.add_argument("--post-close-steps", type=int, default=120)
    ap.add_argument("--lift-steps", type=int, default=1500)
    ap.add_argument("--final-hold-steps", type=int, default=250)

    ap.add_argument("--lift-z", type=float, default=0.09)
    ap.add_argument("--support-freeze-dist", type=float, default=0.0)
    ap.add_argument("--object-ready-dist", type=float, default=0.0015)
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

    print("========== V4.18 EXACT QGRASP REAL-CONTACT CLOSE/LIFT ==========")
    print("model       :", H.rel(model_path))
    print("npy         :", H.rel(npy_path))
    print("sample_index:", args.sample_index)
    print("object_body :", args.object_body)
    print("target_site :", args.target_site)
    print("support_freeze_dist:", args.support_freeze_dist)
    print("object_ready_dist   :", args.object_ready_dist)

    H.set_qpos_once(model, data, H.Q_HOME, side_open)

    viewer = None
    if args.viewer:
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.lookat[:] = [0.455, 0.0, 0.30]
        viewer.cam.distance = 0.75
        viewer.cam.azimuth = 130
        viewer.cam.elevation = -25
        live_sync(viewer, args.live_sleep)

    print("\n[PHASE] settle object")
    for k in range(args.settle_steps):
        step_ctrl(model, data, H.Q_HOME, side_open, viewer, args.live_sleep)
        if k % args.print_every == 0 or k == args.settle_steps - 1:
            print(f"[settle] {k}/{args.settle_steps} object_pos={H.object_pos(model, data, args.object_body).tolist()}")

    object_start = H.object_pos(model, data, args.object_body)
    T_world_object = H.body_world_T(model, data, args.object_body)
    T_grasp = T_world_object @ T_object_hand
    T_lift = np.array(T_grasp, dtype=float)
    T_lift[2, 3] += float(args.lift_z)

    print("\n[IK exact target]")
    ik_grasp = H.solve_site_ik(model, args.target_site, T_grasp, H.Q_HOME)
    ik_lift = H.solve_site_ik(model, args.target_site, T_lift, ik_grasp["q_arm"])

    print("T_world_object.pos:", T_world_object[:3, 3].tolist())
    print("T_object_hand.pos :", T_object_hand[:3, 3].tolist())
    print("T_grasp.pos       :", T_grasp[:3, 3].tolist())
    print("ik_grasp_success  :", ik_grasp["success"], "pos_err:", ik_grasp["final_pos_err_norm"], "rot_err:", ik_grasp["final_rot_err_norm"])
    print("ik_lift_success   :", ik_lift["success"], "pos_err:", ik_lift["final_pos_err_norm"], "rot_err:", ik_lift["final_rot_err_norm"])

    if not ik_grasp["success"]:
        raise RuntimeError("ik_grasp failed")

    q_current = H.get_joint_values(model, data, H.ARM_JOINTS)

    move_rows = move_to_qgrasp(model, data, viewer, args, q_current, ik_grasp["q_arm"], side_open, object_start)

    hold_rows, hold_stable, hold_max_stable = hold_at_q(
        model, data, viewer, args,
        ik_grasp["q_arm"], side_open, object_start,
        "hold_exact_qgrasp_side_open",
        args.grasp_hold_steps,
    )

    close_info = support_aware_close_real(
        model, data, viewer, args,
        ik_grasp["q_arm"], side_open, close_ctrl, object_start
    )

    post_rows, post_stable, post_max_stable = hold_at_q(
        model, data, viewer, args,
        ik_grasp["q_arm"], close_info["hand_ctrl"], object_start,
        "post_close_hold",
        args.post_close_steps,
    )

    grip_ready = (
        close_info["max_stable"] >= args.ready_stable_steps
        or post_max_stable >= args.ready_stable_steps
    )

    lifted = False
    lift_rows = []
    if grip_ready and ik_lift["success"]:
        lift_rows, lift_stable, lift_max_stable = lift_with_fixed_grip(
            model, data, viewer, args,
            ik_grasp["q_arm"], ik_lift["q_arm"], close_info["hand_ctrl"], object_start
        )
        lifted = True

        final_rows, final_stable, final_max_stable = hold_at_q(
            model, data, viewer, args,
            ik_lift["q_arm"], close_info["hand_ctrl"], object_start,
            "final_hold_after_lift",
            args.final_hold_steps,
        )
    else:
        print("[NO_LIFT] grip_ready false or ik_lift failed.")
        final_rows = []

    final_pos = H.object_pos(model, data, args.object_body)
    final_state = H.contact_state(model, data, args.object_body)
    final_obj_groups, _ = real_object_groups(final_state, args.object_ready_dist)
    final_real_sup, _ = real_support_groups(final_state, args.support_freeze_dist)

    final_rise = float(final_pos[2] - object_start[2])
    final_disp = float(np.linalg.norm(final_pos - object_start))
    success = bool(lifted and final_rise > 0.03 and ready_from_groups(final_obj_groups))

    result = {
        "format": "v4_18_exact_qgrasp_real_contact_close_lift_debug_v1",
        "model": H.rel(model_path),
        "npy": H.rel(npy_path),
        "sample_index_valid_local": args.sample_index,
        "object_body": args.object_body,
        "target_site": args.target_site,
        "support_freeze_dist": args.support_freeze_dist,
        "object_ready_dist": args.object_ready_dist,
        "T_world_object_after_settle": H.mat_to_dict(T_world_object),
        "T_object_hand_base_from_dataset": H.mat_to_dict(T_object_hand),
        "T_grasp": H.mat_to_dict(T_grasp),
        "T_lift": H.mat_to_dict(T_lift),
        "ik_grasp": ik_grasp,
        "ik_lift": ik_lift,
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
        "final_object_groups": final_obj_groups,
        "final_support_real_groups": final_real_sup,
        "final_support_margin_groups": final_state["support_groups"],
        "move_rows": move_rows,
        "hold_rows": hold_rows,
        "close_rows": close_info["rows"],
        "post_rows": post_rows,
        "lift_rows": lift_rows,
        "final_rows": final_rows,
    }

    H.save_json(out_dir / "result.json", result)

    print("\n========== V4.18 RESULT ==========")
    print("out:", H.rel(out_dir / "result.json"))
    print("grip_ready:", grip_ready)
    print("lifted:", lifted)
    print("success:", success)
    print("final_object_rise:", final_rise)
    print("final_object_disp:", final_disp)
    print("final_object_groups:", final_obj_groups)
    print("final_support_real_groups:", final_real_sup)
    print("final_support_margin_groups:", final_state["support_groups"])
    print("frozen:", close_info["frozen"])
    print("freeze_reason:", close_info["freeze_reason"])
    print("==================================")

    if args.viewer:
        print("[VIEWER] live run finished. Keep open; close viewer or Ctrl+C.")
        while viewer is not None and viewer.is_running():
            live_sync(viewer, args.live_sleep)


if __name__ == "__main__":
    main()
