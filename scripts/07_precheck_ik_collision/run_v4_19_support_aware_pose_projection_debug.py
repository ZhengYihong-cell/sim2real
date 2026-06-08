#!/usr/bin/env python3
"""
脚本类型：
    debug / v4.19 / support-aware-pose-projection / grasp-pose-selector

用途：
    对一个已选中的数据集抓握先验 T_prior 做通用可行域投影。
    不是手动微调某个 sample，而是自动搜索最小位姿修正，使抓握姿态同时满足：
        1. 尽量接近数据集先验；
        2. 避免手指/手掌进入支撑台 margin 区域；
        3. 闭合后能够形成 thumb + 至少一根四指的物体接触；
        4. 不明显推走物体。

输入：
    --model        已修复 contact 的 scene，例如 scene_v418_low_margin_contact.xml
    --npy          object.npy
    --sample-index valid local sample index
    --object-body  grasp_object
    --target-site  dataset_hand_base_debug
    --out-dir      输出目录

输出：
    out_dir/v419_pose_projection_result.json
    out_dir/v419_pose_projection_summary.txt

当前流程位置：
    V4.18 已证明 IK 和 close 执行链路可运行，但 exact prior 在桌面上可能过低/过近支撑。
    本脚本将 prior 自动投影到 support-aware 可行区域，供后续完整 close/lift runner 使用。

不负责：
    1. 不换 sample；
    2. 不修改数据集；
    3. 不做人工指定的固定 z 微调；
    4. 不替代完整动态 lift，只做短 close 预评估和姿态选择。
"""

from pathlib import Path
import argparse
import importlib.util
import json
import math
import numpy as np
import mujoco


PROJECT = Path.home() / "Projects/o7_mujoco_sim"
HELPER = PROJECT / "scripts/05_execution_runner/run_v4_17_exact_site_qgrasp_close_lift_debug.py"

spec = importlib.util.spec_from_file_location("v417_helper", str(HELPER))
H = importlib.util.module_from_spec(spec)
spec.loader.exec_module(H)


NON_THUMB = ["index", "middle", "ring", "pinky"]


def save_json(path, data):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data, indent=2, ensure_ascii=False))


def real_groups_from_contacts(contacts, dist_th):
    groups = {}
    used = []
    for c in contacts:
        d = float(c.get("dist", 999.0))
        if d <= dist_th:
            g = c.get("group")
            groups[g] = groups.get(g, 0) + 1
            used.append(c)
    return groups, used


def object_ready(groups):
    return ("thumb" in groups) and any(g in groups for g in NON_THUMB)


def mat_to_quat_wxyz(R):
    q = np.zeros(4)
    mujoco.mju_mat2Quat(q, np.asarray(R, dtype=float).reshape(9))
    return q.tolist()


def set_arm_and_hand(model, data, q_arm, hand_ctrl):
    H.apply_ctrl(model, data, q_arm, hand_ctrl)
    mujoco.mj_forward(model, data)


def step_arm_and_hand(model, data, q_arm, hand_ctrl, n=1):
    for _ in range(n):
        H.apply_ctrl(model, data, q_arm, hand_ctrl)
        mujoco.mj_step(model, data)


def settle_model(model, data, q_arm, hand_ctrl, object_body, steps):
    for _ in range(steps):
        step_arm_and_hand(model, data, q_arm, hand_ctrl, 1)
    obj_pos = H.object_pos(model, data, object_body)
    T_obj = H.body_world_T(model, data, object_body)
    return obj_pos, T_obj


def interp_ctrl(side_open, close_ctrl, alpha):
    out = dict(side_open)
    for k, v1 in close_ctrl.items():
        v0 = float(side_open.get(k, v1))
        out[k] = (1.0 - alpha) * v0 + alpha * float(v1)
    return out


def evaluate_pose(model_path, args, q_home, side_open, close_ctrl, T_candidate, object_start, T_world_object):
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    H.set_qpos_once(model, data, q_home, side_open)
    for _ in range(args.settle_steps):
        step_arm_and_hand(model, data, q_home, side_open, 1)

    ik = H.solve_site_ik(model, args.target_site, T_candidate, q_home)
    if not ik["success"]:
        return {
            "ik_success": False,
            "score": 1e9,
            "reason": "ik_failed",
            "ik": ik,
        }

    q_arm = ik["q_arm"]

    # 到 q_arm，side-open
    set_arm_and_hand(model, data, q_arm, side_open)
    for _ in range(args.hold_steps):
        step_arm_and_hand(model, data, q_arm, side_open, 1)

    st_open = H.contact_state(model, data, args.object_body)
    open_obj_groups, _ = real_groups_from_contacts(st_open.get("object_contacts", []), args.object_ready_dist)
    open_support_real, _ = real_groups_from_contacts(st_open.get("support_contacts", []), args.support_real_dist)
    open_support_margin = st_open.get("support_groups", {})

    # 短 close 预检
    max_obj_groups = {}
    max_obj_count = 0
    max_ready_stable = 0
    ready_stable = 0
    max_support_margin_count = len(open_support_margin)
    max_support_real_count = len(open_support_real)

    for k in range(args.close_probe_steps):
        alpha = (k + 1) / max(1, args.close_probe_steps)
        hand_ctrl = interp_ctrl(side_open, close_ctrl, alpha)
        step_arm_and_hand(model, data, q_arm, hand_ctrl, 1)

        st = H.contact_state(model, data, args.object_body)
        obj_groups, _ = real_groups_from_contacts(st.get("object_contacts", []), args.object_ready_dist)
        support_real, _ = real_groups_from_contacts(st.get("support_contacts", []), args.support_real_dist)
        support_margin = st.get("support_groups", {})

        if len(obj_groups) > max_obj_count:
            max_obj_count = len(obj_groups)
            max_obj_groups = dict(obj_groups)

        max_support_margin_count = max(max_support_margin_count, len(support_margin))
        max_support_real_count = max(max_support_real_count, len(support_real))

        if object_ready(obj_groups):
            ready_stable += 1
            max_ready_stable = max(max_ready_stable, ready_stable)
        else:
            ready_stable = 0

    final_state = H.contact_state(model, data, args.object_body)
    final_obj_groups, _ = real_groups_from_contacts(final_state.get("object_contacts", []), args.object_ready_dist)
    final_support_real, _ = real_groups_from_contacts(final_state.get("support_contacts", []), args.support_real_dist)
    final_support_margin = final_state.get("support_groups", {})

    final_pos = H.object_pos(model, data, args.object_body)
    obj_disp = float(np.linalg.norm(final_pos - object_start))

    ready = object_ready(final_obj_groups) or object_ready(max_obj_groups)

    # score 越小越好
    # 解释：
    #   ready 奖励最大；
    #   support_real 真实穿透惩罚极大；
    #   support_margin 是靠近/擦到支撑，惩罚中等；
    #   object displacement 惩罚；
    #   prior_offset 惩罚由外部填。
    score = 0.0
    if ready:
        score -= 1000.0
    score -= 120.0 * max_obj_count
    score -= 50.0 * max_ready_stable
    score += 500.0 * max_support_real_count
    score += 30.0 * max_support_margin_count
    score += 1000.0 * obj_disp

    return {
        "ik_success": True,
        "score_without_prior": float(score),
        "ready": bool(ready),
        "max_ready_stable": int(max_ready_stable),
        "max_obj_groups": max_obj_groups,
        "final_obj_groups": final_obj_groups,
        "open_support_margin": open_support_margin,
        "open_support_real": open_support_real,
        "final_support_margin": final_support_margin,
        "final_support_real": final_support_real,
        "max_support_margin_count": int(max_support_margin_count),
        "max_support_real_count": int(max_support_real_count),
        "final_object_pos": final_pos.tolist(),
        "object_disp": obj_disp,
        "ik": ik,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--npy", required=True)
    ap.add_argument("--sample-index", type=int, required=True)
    ap.add_argument("--object-body", default="grasp_object")
    ap.add_argument("--target-site", default="dataset_hand_base_debug")
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--settle-steps", type=int, default=800)
    ap.add_argument("--hold-steps", type=int, default=80)
    ap.add_argument("--close-probe-steps", type=int, default=260)

    ap.add_argument("--object-ready-dist", type=float, default=0.0015)
    ap.add_argument("--support-real-dist", type=float, default=0.0)

    ap.add_argument("--dz-list", default="0,0.004,0.008,0.012,0.016,0.020,0.024,0.028,0.032")
    ap.add_argument("--radial-list", default="-0.012,-0.008,-0.004,0,0.004,0.008,0.012")

    args = ap.parse_args()

    model_path = H.resolve(args.model)
    npy_path = H.resolve(args.npy)
    out_dir = H.resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    sample = H.load_sample(npy_path, args.sample_index)
    T_object_hand = H.sample_T_object_hand(sample)
    close_ctrl = H.sample_ctrl(sample)
    side_open = H.side_open_from_close(close_ctrl)

    # 初始 settle，得到真实物体世界位姿
    model0 = mujoco.MjModel.from_xml_path(str(model_path))
    data0 = mujoco.MjData(model0)
    H.set_qpos_once(model0, data0, H.Q_HOME, side_open)

    object_start, T_world_object = settle_model(
        model0, data0, H.Q_HOME, side_open, args.object_body, args.settle_steps
    )

    T_prior = T_world_object @ T_object_hand
    object_pos = T_world_object[:3, 3].copy()
    hand_pos = T_prior[:3, 3].copy()

    # 支撑法向，当前桌面/垫块统一为 world +z
    support_normal = np.array([0.0, 0.0, 1.0], dtype=float)

    # radial 方向不是手工轴，而是“当前 handbase 到 object center 的水平投影方向”
    # radial > 0 表示手往物体中心靠近；radial < 0 表示手离物体中心远一点。
    radial_vec = object_pos - hand_pos
    radial_vec[2] = 0.0
    n = np.linalg.norm(radial_vec)
    if n < 1e-9:
        radial_dir = np.array([0.0, 0.0, 0.0], dtype=float)
    else:
        radial_dir = radial_vec / n

    dz_values = [float(x) for x in args.dz_list.split(",") if x.strip()]
    radial_values = [float(x) for x in args.radial_list.split(",") if x.strip()]

    print("========== V4.19 SUPPORT-AWARE POSE PROJECTION ==========")
    print("model       :", H.rel(model_path))
    print("npy         :", H.rel(npy_path))
    print("sample_index:", args.sample_index)
    print("object_start:", object_start.tolist())
    print("T_prior.pos :", T_prior[:3, 3].tolist())
    print("support_normal:", support_normal.tolist())
    print("radial_dir     :", radial_dir.tolist())
    print("dz_values      :", dz_values)
    print("radial_values  :", radial_values)

    trials = []
    best = None

    for dz in dz_values:
        for radial in radial_values:
            T = np.array(T_prior, dtype=float)
            delta = support_normal * dz + radial_dir * radial
            T[:3, 3] += delta

            ev = evaluate_pose(
                model_path=model_path,
                args=args,
                q_home=H.Q_HOME,
                side_open=side_open,
                close_ctrl=close_ctrl,
                T_candidate=T,
                object_start=object_start,
                T_world_object=T_world_object,
            )

            prior_penalty = 2000.0 * float(np.linalg.norm(delta))
            score = float(ev.get("score_without_prior", 1e9) + prior_penalty)
            ev["score"] = score
            ev["dz"] = dz
            ev["radial"] = radial
            ev["delta"] = delta.tolist()
            ev["T_candidate"] = {
                "pos": T[:3, 3].tolist(),
                "quat_wxyz": mat_to_quat_wxyz(T[:3, :3]),
                "R": T[:3, :3].tolist(),
                "T": T.tolist(),
            }

            trials.append(ev)

            print(
                f"[trial] dz={dz:+.4f} radial={radial:+.4f} "
                f"score={score:+.2f} ik={ev.get('ik_success')} "
                f"ready={ev.get('ready')} obj={ev.get('max_obj_groups')} "
                f"support_margin_max={ev.get('max_support_margin_count')} "
                f"support_real_max={ev.get('max_support_real_count')} "
                f"disp={ev.get('object_disp')}"
            )

            if best is None or score < best["score"]:
                best = ev

    result = {
        "format": "v4_19_support_aware_pose_projection_debug_v1",
        "model": H.rel(model_path),
        "npy": H.rel(npy_path),
        "sample_index_valid_local": args.sample_index,
        "object_body": args.object_body,
        "target_site": args.target_site,
        "object_start": object_start.tolist(),
        "T_world_object": {
            "pos": T_world_object[:3, 3].tolist(),
            "R": T_world_object[:3, :3].tolist(),
            "T": T_world_object.tolist(),
        },
        "T_object_hand_from_dataset": {
            "pos": T_object_hand[:3, 3].tolist(),
            "R": T_object_hand[:3, :3].tolist(),
            "T": T_object_hand.tolist(),
        },
        "T_prior": {
            "pos": T_prior[:3, 3].tolist(),
            "quat_wxyz": mat_to_quat_wxyz(T_prior[:3, :3]),
            "R": T_prior[:3, :3].tolist(),
            "T": T_prior.tolist(),
        },
        "radial_dir": radial_dir.tolist(),
        "support_normal": support_normal.tolist(),
        "best": best,
        "trials": sorted(trials, key=lambda x: x["score"]),
    }

    save_json(out_dir / "v419_pose_projection_result.json", result)

    summary = []
    summary.append("========== V4.19 BEST ==========")
    summary.append(f"best dz      : {best.get('dz')}")
    summary.append(f"best radial  : {best.get('radial')}")
    summary.append(f"best score   : {best.get('score')}")
    summary.append(f"ready        : {best.get('ready')}")
    summary.append(f"max_obj      : {best.get('max_obj_groups')}")
    summary.append(f"final_obj    : {best.get('final_obj_groups')}")
    summary.append(f"support_margin_max: {best.get('max_support_margin_count')}")
    summary.append(f"support_real_max  : {best.get('max_support_real_count')}")
    summary.append(f"object_disp  : {best.get('object_disp')}")
    summary.append(f"T_candidate.pos: {best['T_candidate']['pos']}")
    summary.append("================================")
    (out_dir / "v419_pose_projection_summary.txt").write_text("\n".join(summary), encoding="utf-8")

    print("\n".join(summary))
    print("result:", H.rel(out_dir / "v419_pose_projection_result.json"))


if __name__ == "__main__":
    main()
