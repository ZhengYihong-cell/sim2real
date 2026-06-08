#!/usr/bin/env python3
"""
V4.22 best-candidate full close/lift runner.

定位：
    对 V4.21b 选出的 best candidate 做完整动态 close + lift + viewer 验证。
    这不是 selector，不再调 sample，不再局部微调位姿。

流程：
    1. 读取 selected_best_candidate.json，得到 valid_local_index；
    2. 根据 scene 中 settle 后的 object pose 计算 T_world_hand；
    3. site IK 到 dataset_hand_base_debug；
    4. 从 home/open 平滑运动到 q_grasp；
    5. 在 q_grasp 处长时间 servo hold，让 arm 尽量跟上；
    6. support-aware close：碰到 support 的手指冻结，碰到 object 的手指继续夹持；
    7. thumb + 至少一根非拇指 ready 后 lift；
    8. 输出 result.json 和 terminal 日志。

注意：
    不使用 snap-to-qgrasp。
    不每步硬写 qpos。
    动态阶段只发 ctrl。
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

NON_THUMB = ["index", "middle", "ring", "pinky"]


def resolve(p):
    p = Path(str(p)).expanduser()
    return p if p.is_absolute() else PROJECT / p


def rel(p):
    p = Path(p)
    try:
        return str(p.relative_to(PROJECT))
    except Exception:
        return str(p)


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def load_best(path):
    d = json.loads(Path(path).read_text())
    if "valid_local_index" not in d:
        raise RuntimeError(f"best candidate missing valid_local_index: {path}")
    return d


def side_open_from_close(close_ctrl):
    side = dict(close_ctrl)
    for j in ["index_mcp_pitch", "middle_mcp_pitch", "ring_mcp_pitch", "pinky_mcp_pitch"]:
        side[j] = 0.0
    return side


def live_sync(viewer, sleep_s):
    if viewer is not None and viewer.is_running():
        viewer.sync()
        time.sleep(float(sleep_s))


def step_ctrl(model, data, q_arm, hand_ctrl, viewer, sleep_s):
    H.apply_ctrl(model, data, q_arm, hand_ctrl)
    mujoco.mj_step(model, data)
    live_sync(viewer, sleep_s)


def interp_arm(q0, q1, alpha):
    out = {}
    for j in H.ARM_JOINTS:
        a = float(q0.get(j, q1.get(j, 0.0)))
        b = float(q1.get(j, a))
        out[j] = (1.0 - alpha) * a + alpha * b
    return out


def real_groups_from_contacts(state, kind, dist_th):
    raw = state.get("object_contacts", []) if kind == "object" else state.get("support_contacts", [])
    groups = {}
    contacts = []
    for c in raw:
        d = float(c.get("dist", 999.0))
        if d <= dist_th:
            g = c.get("group")
            if g is None:
                continue
            groups[g] = groups.get(g, 0) + 1
            contacts.append(c)
    return groups, contacts


def ready_from_groups(groups):
    return ("thumb" in groups) and any(g in groups for g in NON_THUMB)


def current_site_error(model, data, site_name, T_target):
    T_cur = H.site_world_T(model, data, site_name)
    _, _, pos_n, rot_n = H.pose_error(T_cur, T_target)
    return float(pos_n), float(rot_n)


def make_hand_ctrl(side_open, close_ctrl, group_alpha):
    ctrl = dict(side_open)
    for g, joints in H.FINGER_GROUP_TO_JOINTS.items():
        a = float(group_alpha.get(g, 0.0))
        for j in joints:
            v0 = float(side_open.get(j, close_ctrl.get(j, 0.0)))
            v1 = float(close_ctrl.get(j, v0))
            ctrl[j] = (1.0 - a) * v0 + a * v1
    return ctrl


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--npy", required=True)
    ap.add_argument("--best-candidate", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--object-body", default="grasp_object")
    ap.add_argument("--target-site", default="dataset_hand_base_debug")
    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--live-sleep", type=float, default=0.002)

    ap.add_argument("--settle-steps", type=int, default=1000)
    ap.add_argument("--move-steps", type=int, default=3200)
    ap.add_argument("--servo-hold-steps", type=int, default=2500)
    ap.add_argument("--close-steps", type=int, default=1000)
    ap.add_argument("--post-close-steps", type=int, default=300)
    ap.add_argument("--lift-steps", type=int, default=2200)
    ap.add_argument("--final-hold-steps", type=int, default=800)

    ap.add_argument("--lift-z", type=float, default=0.09)
    ap.add_argument("--object-ready-dist", type=float, default=0.0015)
    ap.add_argument("--support-freeze-dist", type=float, default=0.0)
    ap.add_argument("--ready-stable-steps", type=int, default=5)

    ap.add_argument("--site-ready-pos-err", type=float, default=0.018)
    ap.add_argument("--site-ready-rot-err", type=float, default=0.12)
    ap.add_argument("--print-every", type=int, default=120)

    ap.add_argument("--max-final-disp-before-lift", type=float, default=0.035)
    ap.add_argument("--success-rise", type=float, default=0.03)
    args = ap.parse_args()

    model_path = resolve(args.model)
    npy_path = resolve(args.npy)
    best_path = resolve(args.best_candidate)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    best = load_best(best_path)
    sample_idx = int(best["valid_local_index"])
    raw_idx = int(best.get("raw_sample_index", sample_idx))

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    sample = H.load_sample(str(npy_path), sample_idx)
    T_object_hand = H.sample_T_object_hand(sample)
    close_ctrl = H.sample_ctrl(sample)
    side_open = side_open_from_close(close_ctrl)

    print("========== V4.22 BEST CANDIDATE FULL CLOSE/LIFT ==========")
    print("model          :", rel(model_path))
    print("npy            :", rel(npy_path))
    print("best_candidate :", rel(best_path))
    print("sample local/raw:", sample_idx, raw_idx)
    print("selector_type  :", best.get("selector_type"))
    print("short_ready    :", best.get("short_ready"))
    print("short_success  :", best.get("short_success"))

    H.set_qpos_once(model, data, H.Q_HOME, side_open)

    viewer = None
    if args.viewer:
        viewer = mujoco.viewer.launch_passive(model, data)
        viewer.cam.lookat[:] = [0.455, 0.0, 0.30]
        viewer.cam.distance = 0.75
        viewer.cam.azimuth = 130
        viewer.cam.elevation = -25
        live_sync(viewer, args.live_sleep)

    print("\n[PHASE] settle")
    for k in range(args.settle_steps):
        step_ctrl(model, data, H.Q_HOME, side_open, viewer, args.live_sleep)
        if k % args.print_every == 0 or k == args.settle_steps - 1:
            print(f"[settle] {k}/{args.settle_steps} object_pos={H.object_pos(model, data, args.object_body).tolist()}")

    object_start = H.object_pos(model, data, args.object_body)
    T_world_object = H.body_world_T(model, data, args.object_body)
    T_grasp = T_world_object @ T_object_hand

    T_lift = np.array(T_grasp, dtype=float)
    T_lift[2, 3] += float(args.lift_z)

    print("\n[IK]")
    ik_grasp = H.solve_site_ik(model, args.target_site, T_grasp, H.Q_HOME)
    ik_lift = H.solve_site_ik(model, args.target_site, T_lift, ik_grasp["q_arm"])

    print("T_world_object.pos:", T_world_object[:3, 3].tolist())
    print("T_object_hand.pos :", T_object_hand[:3, 3].tolist())
    print("T_grasp.pos       :", T_grasp[:3, 3].tolist())
    print("ik_grasp:", ik_grasp["success"], "pos_err=", ik_grasp["final_pos_err_norm"], "rot_err=", ik_grasp["final_rot_err_norm"])
    print("ik_lift :", ik_lift["success"], "pos_err=", ik_lift["final_pos_err_norm"], "rot_err=", ik_lift["final_rot_err_norm"])

    if not ik_grasp["success"]:
        raise RuntimeError("ik_grasp failed; cannot execute V4.22")

    q_current = H.get_joint_values(model, data, H.ARM_JOINTS)
    q_grasp = ik_grasp["q_arm"]
    q_lift = ik_lift["q_arm"] if ik_lift["success"] else None

    rows = []
    max_move_disp = 0.0

    print("\n[PHASE] smooth move current -> q_grasp, hand open")
    for k in range(args.move_steps):
        alpha = k / max(1, args.move_steps - 1)
        q_cmd = interp_arm(q_current, q_grasp, alpha)
        step_ctrl(model, data, q_cmd, side_open, viewer, args.live_sleep)

        obj_pos = H.object_pos(model, data, args.object_body)
        disp = float(np.linalg.norm(obj_pos - object_start))
        max_move_disp = max(max_move_disp, disp)

        st = H.contact_state(model, data, args.object_body)
        obj_groups, _ = real_groups_from_contacts(st, "object", args.object_ready_dist)
        sup_groups, _ = real_groups_from_contacts(st, "support", args.support_freeze_dist)
        site_pos_err, site_rot_err = current_site_error(model, data, args.target_site, T_grasp)

        if k % args.print_every == 0 or k == args.move_steps - 1:
            print(
                f"[move] {k}/{args.move_steps} alpha={alpha:.3f} "
                f"site=({site_pos_err:.4f},{site_rot_err:.4f}) disp={disp:.4f} "
                f"obj={obj_groups} support={sup_groups}"
            )

        rows.append({
            "phase": "move",
            "step": k,
            "alpha": alpha,
            "site_pos_err": site_pos_err,
            "site_rot_err": site_rot_err,
            "object_disp": disp,
            "object_groups": obj_groups,
            "support_groups": sup_groups,
        })

    print("\n[PHASE] servo hold at q_grasp, hand open")
    site_ready = False
    best_site_pos_err = 999.0
    best_site_rot_err = 999.0
    max_hold_disp = max_move_disp

    for k in range(args.servo_hold_steps):
        step_ctrl(model, data, q_grasp, side_open, viewer, args.live_sleep)

        obj_pos = H.object_pos(model, data, args.object_body)
        disp = float(np.linalg.norm(obj_pos - object_start))
        max_hold_disp = max(max_hold_disp, disp)

        st = H.contact_state(model, data, args.object_body)
        obj_groups, _ = real_groups_from_contacts(st, "object", args.object_ready_dist)
        sup_groups, _ = real_groups_from_contacts(st, "support", args.support_freeze_dist)
        site_pos_err, site_rot_err = current_site_error(model, data, args.target_site, T_grasp)

        best_site_pos_err = min(best_site_pos_err, site_pos_err)
        best_site_rot_err = min(best_site_rot_err, site_rot_err)

        if site_pos_err <= args.site_ready_pos_err and site_rot_err <= args.site_ready_rot_err:
            site_ready = True

        if k % args.print_every == 0 or k == args.servo_hold_steps - 1:
            print(
                f"[hold] {k}/{args.servo_hold_steps} "
                f"site=({site_pos_err:.4f},{site_rot_err:.4f}) ready={site_ready} "
                f"disp={disp:.4f} obj={obj_groups} support={sup_groups}"
            )

        rows.append({
            "phase": "servo_hold_open",
            "step": k,
            "site_pos_err": site_pos_err,
            "site_rot_err": site_rot_err,
            "site_ready": bool(site_ready),
            "object_disp": disp,
            "object_groups": obj_groups,
            "support_groups": sup_groups,
        })

    print("\n[PHASE] support-aware close at q_grasp")
    group_alpha = {g: 0.0 for g in H.FINGER_GROUP_TO_JOINTS}
    frozen = {g: False for g in H.FINGER_GROUP_TO_JOINTS}
    freeze_reason = {}
    last_hand_ctrl = dict(side_open)

    stable_ready = 0
    max_stable_ready = 0
    max_close_disp = 0.0
    max_obj_groups = {}
    max_support_groups = {}

    for k in range(args.close_steps):
        st_before = H.contact_state(model, data, args.object_body)
        support_before, support_contacts_before = real_groups_from_contacts(st_before, "support", args.support_freeze_dist)

        for g in H.FINGER_GROUP_TO_JOINTS:
            if frozen[g]:
                continue
            if g in support_before:
                frozen[g] = True
                freeze_reason[g] = {
                    "reason": "support_real_contact_freeze",
                    "step": k,
                    "contacts": [c for c in support_contacts_before if c.get("group") == g],
                }
                continue
            group_alpha[g] = min(1.0, float(group_alpha[g]) + 1.0 / max(1, args.close_steps))

        last_hand_ctrl = make_hand_ctrl(side_open, close_ctrl, group_alpha)
        step_ctrl(model, data, q_grasp, last_hand_ctrl, viewer, args.live_sleep)

        st = H.contact_state(model, data, args.object_body)
        obj_groups, obj_contacts = real_groups_from_contacts(st, "object", args.object_ready_dist)
        sup_groups, sup_contacts = real_groups_from_contacts(st, "support", args.support_freeze_dist)

        ready = ready_from_groups(obj_groups)
        stable_ready = stable_ready + 1 if ready else 0
        max_stable_ready = max(max_stable_ready, stable_ready)

        if len(obj_groups) > len(max_obj_groups):
            max_obj_groups = dict(obj_groups)
        if len(sup_groups) > len(max_support_groups):
            max_support_groups = dict(sup_groups)

        obj_pos = H.object_pos(model, data, args.object_body)
        disp = float(np.linalg.norm(obj_pos - object_start))
        max_close_disp = max(max_close_disp, disp)

        site_pos_err, site_rot_err = current_site_error(model, data, args.target_site, T_grasp)

        if k % args.print_every == 0 or k == args.close_steps - 1 or stable_ready >= args.ready_stable_steps:
            print(
                f"[close] {k}/{args.close_steps} ready={ready} stable={stable_ready} "
                f"site=({site_pos_err:.4f},{site_rot_err:.4f}) disp={disp:.4f} "
                f"obj={obj_groups} support={sup_groups} alpha={group_alpha} frozen={frozen}"
            )

        rows.append({
            "phase": "close",
            "step": k,
            "ready": bool(ready),
            "stable_ready": int(stable_ready),
            "site_pos_err": site_pos_err,
            "site_rot_err": site_rot_err,
            "object_disp": disp,
            "object_groups": obj_groups,
            "object_contacts": obj_contacts,
            "support_groups": sup_groups,
            "support_contacts": sup_contacts,
            "group_alpha": dict(group_alpha),
            "frozen": dict(frozen),
            "hand_ctrl": dict(last_hand_ctrl),
        })

        if stable_ready >= args.ready_stable_steps:
            print(f"[READY] stable_ready={stable_ready}; close stops.")
            break

        if disp > args.max_final_disp_before_lift:
            print(f"[WARN] object disp before lift too large: {disp:.4f}")
            break

    print("\n[PHASE] post-close hold")
    post_stable = 0
    max_post_stable = 0

    for k in range(args.post_close_steps):
        step_ctrl(model, data, q_grasp, last_hand_ctrl, viewer, args.live_sleep)

        st = H.contact_state(model, data, args.object_body)
        obj_groups, obj_contacts = real_groups_from_contacts(st, "object", args.object_ready_dist)
        sup_groups, _ = real_groups_from_contacts(st, "support", args.support_freeze_dist)

        ready = ready_from_groups(obj_groups)
        post_stable = post_stable + 1 if ready else 0
        max_post_stable = max(max_post_stable, post_stable)

        obj_pos = H.object_pos(model, data, args.object_body)
        disp = float(np.linalg.norm(obj_pos - object_start))
        max_close_disp = max(max_close_disp, disp)

        if k % args.print_every == 0 or k == args.post_close_steps - 1:
            print(
                f"[post] {k}/{args.post_close_steps} ready={ready} post_stable={post_stable} "
                f"disp={disp:.4f} obj={obj_groups} support={sup_groups}"
            )

        rows.append({
            "phase": "post_close",
            "step": k,
            "ready": bool(ready),
            "post_stable": int(post_stable),
            "object_disp": disp,
            "object_groups": obj_groups,
            "object_contacts": obj_contacts,
            "support_groups": sup_groups,
        })

    final_close_state = H.contact_state(model, data, args.object_body)
    final_close_obj_groups, _ = real_groups_from_contacts(final_close_state, "object", args.object_ready_dist)

    grip_ready = (
        max_stable_ready >= args.ready_stable_steps
        or max_post_stable >= args.ready_stable_steps
        or ready_from_groups(final_close_obj_groups)
    )

    lifted = False
    print("\n[READY CHECK]")
    print("site_ready:", site_ready, "best_site_pos_err:", best_site_pos_err, "best_site_rot_err:", best_site_rot_err)
    print("grip_ready:", grip_ready)
    print("max_obj_groups:", max_obj_groups)
    print("frozen:", frozen)
    print("freeze_reason:", freeze_reason)

    if grip_ready and ik_lift["success"]:
        print("\n[PHASE] lift with fixed hand ctrl")
        for k in range(args.lift_steps):
            alpha = k / max(1, args.lift_steps - 1)
            q_cmd = interp_arm(q_grasp, q_lift, alpha)
            step_ctrl(model, data, q_cmd, last_hand_ctrl, viewer, args.live_sleep)

            st = H.contact_state(model, data, args.object_body)
            obj_groups, _ = real_groups_from_contacts(st, "object", args.object_ready_dist)
            sup_groups, _ = real_groups_from_contacts(st, "support", args.support_freeze_dist)

            ready = ready_from_groups(obj_groups)
            obj_pos = H.object_pos(model, data, args.object_body)
            rise = float(obj_pos[2] - object_start[2])
            disp = float(np.linalg.norm(obj_pos - object_start))

            if k % args.print_every == 0 or k == args.lift_steps - 1:
                print(
                    f"[lift] {k}/{args.lift_steps} alpha={alpha:.3f} "
                    f"rise={rise:.4f} disp={disp:.4f} ready={ready} obj={obj_groups} support={sup_groups}"
                )

            rows.append({
                "phase": "lift",
                "step": k,
                "alpha": alpha,
                "rise": rise,
                "object_disp": disp,
                "ready": bool(ready),
                "object_groups": obj_groups,
                "support_groups": sup_groups,
            })

        lifted = True

        print("\n[PHASE] final hold")
        for k in range(args.final_hold_steps):
            step_ctrl(model, data, q_lift, last_hand_ctrl, viewer, args.live_sleep)

            st = H.contact_state(model, data, args.object_body)
            obj_groups, _ = real_groups_from_contacts(st, "object", args.object_ready_dist)
            obj_pos = H.object_pos(model, data, args.object_body)
            rise = float(obj_pos[2] - object_start[2])
            disp = float(np.linalg.norm(obj_pos - object_start))

            if k % args.print_every == 0 or k == args.final_hold_steps - 1:
                print(
                    f"[final] {k}/{args.final_hold_steps} rise={rise:.4f} disp={disp:.4f} obj={obj_groups}"
                )

            rows.append({
                "phase": "final_hold",
                "step": k,
                "rise": rise,
                "object_disp": disp,
                "object_groups": obj_groups,
            })
    else:
        print("[NO_LIFT] grip not ready or lift IK failed.")

    final_pos = H.object_pos(model, data, args.object_body)
    final_state = H.contact_state(model, data, args.object_body)
    final_obj_groups, _ = real_groups_from_contacts(final_state, "object", args.object_ready_dist)
    final_sup_groups, _ = real_groups_from_contacts(final_state, "support", args.support_freeze_dist)

    final_rise = float(final_pos[2] - object_start[2])
    final_disp = float(np.linalg.norm(final_pos - object_start))
    success = bool(lifted and final_rise >= args.success_rise and ready_from_groups(final_obj_groups))

    result = {
        "format": "v4_22_best_candidate_full_close_lift_debug_v1",
        "model": rel(model_path),
        "npy": rel(npy_path),
        "best_candidate": rel(best_path),
        "sample_index_valid_local": sample_idx,
        "raw_sample_index": raw_idx,
        "selector_type": best.get("selector_type"),
        "short_ready_from_v421b": best.get("short_ready"),
        "short_success_from_v421b": best.get("short_success"),
        "T_world_object_after_settle": H.mat_to_dict(T_world_object),
        "T_object_hand": H.mat_to_dict(T_object_hand),
        "T_grasp": H.mat_to_dict(T_grasp),
        "T_lift": H.mat_to_dict(T_lift),
        "ik_grasp": ik_grasp,
        "ik_lift": ik_lift,
        "site_ready": site_ready,
        "best_site_pos_err": best_site_pos_err,
        "best_site_rot_err": best_site_rot_err,
        "grip_ready": grip_ready,
        "lifted": lifted,
        "success": success,
        "object_start": object_start.tolist(),
        "final_object_pos": final_pos.tolist(),
        "final_object_rise": final_rise,
        "final_object_disp": final_disp,
        "final_object_groups": final_obj_groups,
        "final_support_groups": final_sup_groups,
        "max_move_disp": max_move_disp,
        "max_hold_disp": max_hold_disp,
        "max_close_disp": max_close_disp,
        "max_stable_ready": max_stable_ready,
        "max_post_stable": max_post_stable,
        "max_obj_groups": max_obj_groups,
        "max_support_groups": max_support_groups,
        "group_alpha": group_alpha,
        "frozen": frozen,
        "freeze_reason": freeze_reason,
        "final_hand_ctrl": last_hand_ctrl,
        "rows": rows,
    }

    save_json(out_dir / "result.json", result)

    print("\n========== V4.22 RESULT ==========")
    print("out:", rel(out_dir / "result.json"))
    print("sample:", sample_idx, "raw:", raw_idx)
    print("site_ready:", site_ready, "best_site_pos_err:", best_site_pos_err)
    print("grip_ready:", grip_ready)
    print("lifted:", lifted)
    print("success:", success)
    print("final_object_rise:", final_rise)
    print("final_object_disp:", final_disp)
    print("final_object_groups:", final_obj_groups)
    print("final_support_groups:", final_sup_groups)
    print("frozen:", frozen)
    print("freeze_reason:", freeze_reason)
    print("==================================")

    if args.viewer:
        print("[VIEWER] finished. Close viewer or Ctrl+C.")
        while viewer is not None and viewer.is_running():
            live_sync(viewer, args.live_sleep)


if __name__ == "__main__":
    main()
