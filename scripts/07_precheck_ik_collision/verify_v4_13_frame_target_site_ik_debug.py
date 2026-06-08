#!/usr/bin/env python3
"""
脚本类型：
    debug / diagnostic / frame-target / site-ik / v4.13

用途：
    验证 V4.13 当前 generic candidate 链路的根本问题：
        已知 object pose 和数据集先验 hand_pose 后，
        T_world_hand = T_world_object @ T_object_hand_base_link
        是否能通过 MuJoCo site IK 让 dataset_hand_base_debug 到达目标。

    本脚本直接使用数据集 sample 的 hand_pose，不再把目标先转成 fr3_link7。
    它会同时对比：
        1. site IK 解出来的 q 是否能让 dataset_hand_base_debug 对准目标；
        2. 当前 P4U6 result.json 里的 q_grasp 是否其实离目标很远；
        3. 两种姿态下手到物体最近距离是多少。

输入：
    --model       当前 scene.xml
    --npy         当前 object.npy
    --sample-index valid local sample index
    --object-body MuJoCo 物体 body 名称，例如 grasp_object
    --target-site dataset_hand_base_debug
    --result-json 可选，当前 viewer 的 result.json，用于对比旧 q_grasp

输出：
    out_dir/frame_target_site_ik_report.txt
    out_dir/frame_target_site_ik_summary.json

当前流程位置：
    V4.13 selector / generic builder 已生成 scene 和 sample
        -> 本脚本验证 frame target 是否正确
        -> 若 site IK 成功而旧 q_grasp 失败，则修 P2/P3 target frame
        -> 后续再回到 planner / viewer

不负责：
    1. 不运行 P2/P3；
    2. 不运行 P4U6；
    3. 不修改 legacy demo；
    4. 不做抓握姿态微调；
    5. 不按某个轴手动移动目标。
"""

from pathlib import Path
import argparse
import json
import math
import time

import numpy as np
import mujoco


PROJECT = Path.home() / "Projects/o7_mujoco_sim"

ARM_JOINTS = [
    "fr3_joint1",
    "fr3_joint2",
    "fr3_joint3",
    "fr3_joint4",
    "fr3_joint5",
    "fr3_joint6",
    "fr3_joint7",
]

O7_ACTIVE_JOINTS = [
    "thumb_cmc_roll",
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]

Q_HOME = {
    "fr3_joint1": 0.0,
    "fr3_joint2": -0.7,
    "fr3_joint3": 0.0,
    "fr3_joint4": -2.2,
    "fr3_joint5": 0.0,
    "fr3_joint6": 1.8,
    "fr3_joint7": 0.8,
}


def resolve(p):
    p = Path(str(p)).expanduser()
    if not p.is_absolute():
        p = PROJECT / p
    return p


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


def normalize(v, eps=1e-12):
    v = np.asarray(v, dtype=float)
    n = np.linalg.norm(v)
    if n < eps:
        return v * 0.0
    return v / n


def robust_rot6d_to_R(r6):
    r6 = np.asarray(r6, dtype=float).reshape(6)
    x_raw = r6[0:3]
    y_raw = r6[3:6]

    x = normalize(x_raw)
    y = normalize(y_raw)

    middle = normalize(x + y)
    orthmid = normalize(x - y)

    x = normalize(middle + orthmid)
    y = normalize(middle - orthmid)
    z = normalize(np.cross(x, y))

    return np.stack([x, y, z], axis=1)


def T_from_Rp(R, p):
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(p, dtype=float)
    return T


def mat4_to_dict(T):
    return {
        "pos": T[:3, 3].tolist(),
        "R": T[:3, :3].tolist(),
        "T": T.tolist(),
    }


def load_sample(npy, idx):
    arr = np.load(npy, allow_pickle=True)
    if idx < 0 or idx >= len(arr):
        raise RuntimeError(f"sample index out of range: {idx}, n={len(arr)}")
    s = arr[idx].item() if hasattr(arr[idx], "item") else arr[idx]
    if not isinstance(s, dict):
        raise RuntimeError(f"sample is not dict: {type(s)}")
    return s


def extract_sample_T_object_hand(sample):
    hp = np.asarray(sample["hand_pose"], dtype=float)
    if hp.shape[0] < 9:
        raise RuntimeError(f"hand_pose too short: shape={hp.shape}")
    t = hp[0:3]
    R = robust_rot6d_to_R(hp[3:9])
    return T_from_Rp(R, t)


def extract_sample_ctrl(sample):
    hp = np.asarray(sample["hand_pose"], dtype=float)
    if hp.shape[0] >= 16:
        return {j: float(v) for j, v in zip(O7_ACTIVE_JOINTS, hp[9:16])}
    return {}


def name2id(model, objtype, name):
    return mujoco.mj_name2id(model, objtype, name)


def list_site_names(model):
    out = []
    for i in range(model.nsite):
        n = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_SITE, i)
        if n:
            out.append(n)
    return out


def joint_qpos_addr(model, joint_name):
    jid = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return None
    return int(model.jnt_qposadr[jid])


def joint_dof_addr(model, joint_name):
    jid = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return None
    return int(model.jnt_dofadr[jid])


def set_joint_values(model, data, qdict, strict=False):
    missing = []
    for name, val in qdict.items():
        adr = joint_qpos_addr(model, name)
        if adr is None:
            missing.append(name)
            continue
        data.qpos[adr] = float(val)
    mujoco.mj_forward(model, data)
    if strict and missing:
        raise RuntimeError(f"missing joints: {missing}")
    return missing


def get_joint_values(model, data, names):
    out = {}
    for n in names:
        adr = joint_qpos_addr(model, n)
        if adr is not None:
            out[n] = float(data.qpos[adr])
    return out


def clamp_joint(model, joint_name, q):
    jid = name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return q
    limited = int(model.jnt_limited[jid])
    if not limited:
        return q
    lo, hi = model.jnt_range[jid]
    return float(np.clip(q, lo, hi))


def object_world_T(model, data, object_body):
    bid = name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
    if bid < 0:
        raise RuntimeError(f"missing object body: {object_body}")
    R = np.asarray(data.xmat[bid], dtype=float).reshape(3, 3)
    p = np.asarray(data.xpos[bid], dtype=float)
    return T_from_Rp(R, p)


def site_world_T(model, data, site_name):
    sid = name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0:
        raise RuntimeError(f"missing target site: {site_name}; available sites={list_site_names(model)}")
    R = np.asarray(data.site_xmat[sid], dtype=float).reshape(3, 3)
    p = np.asarray(data.site_xpos[sid], dtype=float)
    return T_from_Rp(R, p)


def pose_error(T_cur, T_tar):
    pc = T_cur[:3, 3]
    pt = T_tar[:3, 3]
    Rc = T_cur[:3, :3]
    Rt = T_tar[:3, :3]
    pos_err = pt - pc
    rot_vec = 0.5 * (
        np.cross(Rc[:, 0], Rt[:, 0]) +
        np.cross(Rc[:, 1], Rt[:, 1]) +
        np.cross(Rc[:, 2], Rt[:, 2])
    )
    return pos_err, rot_vec, float(np.linalg.norm(pos_err)), float(np.linalg.norm(rot_vec))


def solve_site_ik(model, data, site_name, T_target, *,
                  max_iters=300, damping=1e-4, step_scale=0.8,
                  rot_weight=0.65, pos_tol=8e-4, rot_tol=8e-3):
    sid = name2id(model, mujoco.mjtObj.mjOBJ_SITE, site_name)
    if sid < 0:
        raise RuntimeError(f"missing target site: {site_name}; available sites={list_site_names(model)}")

    dof_ids = []
    qpos_addrs = []
    for j in ARM_JOINTS:
        da = joint_dof_addr(model, j)
        qa = joint_qpos_addr(model, j)
        if da is None or qa is None:
            raise RuntimeError(f"missing arm joint in model: {j}")
        dof_ids.append(da)
        qpos_addrs.append(qa)

    dof_ids = np.asarray(dof_ids, dtype=int)

    hist = []
    success = False

    for it in range(max_iters):
        mujoco.mj_forward(model, data)

        T_cur = site_world_T(model, data, site_name)
        pos_err, rot_err, pos_norm, rot_norm = pose_error(T_cur, T_target)

        hist.append({
            "iter": it,
            "pos_err_norm": pos_norm,
            "rot_err_norm": rot_norm,
        })

        if pos_norm < pos_tol and rot_norm < rot_tol:
            success = True
            break

        jacp = np.zeros((3, model.nv), dtype=float)
        jacr = np.zeros((3, model.nv), dtype=float)
        mujoco.mj_jacSite(model, data, jacp, jacr, sid)

        J = np.vstack([
            jacp[:, dof_ids],
            rot_weight * jacr[:, dof_ids],
        ])
        e = np.concatenate([pos_err, rot_weight * rot_err])

        A = J @ J.T + damping * np.eye(6)
        dq = J.T @ np.linalg.solve(A, e)

        max_step = 0.10
        n = float(np.linalg.norm(dq))
        if n > max_step:
            dq *= max_step / n

        dq *= step_scale

        for k, jname in enumerate(ARM_JOINTS):
            qa = qpos_addrs[k]
            data.qpos[qa] = clamp_joint(model, jname, float(data.qpos[qa] + dq[k]))

    mujoco.mj_forward(model, data)
    T_final = site_world_T(model, data, site_name)
    pos_err, rot_err, pos_norm, rot_norm = pose_error(T_final, T_target)

    return {
        "success": bool(success),
        "iters": len(hist),
        "final_pos_err_norm": pos_norm,
        "final_rot_err_norm": rot_norm,
        "q_arm": get_joint_values(model, data, ARM_JOINTS),
        "history_tail": hist[-10:],
    }


def geom_body_name(model, gid):
    bid = int(model.geom_bodyid[gid])
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, bid) or ""


def geom_name(model, gid):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or f"geom_{gid}"


def collect_object_geoms(model, object_body):
    bid = name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
    if bid < 0:
        raise RuntimeError(f"missing object body: {object_body}")
    return [gid for gid in range(model.ngeom) if int(model.geom_bodyid[gid]) == bid]


def collect_hand_geoms(model):
    tokens = ["thumb", "index", "middle", "ring", "pinky", "palm", "hand"]
    bad = ["object", "pedestal", "support", "world_plane", "floor", "table"]
    out = []
    for gid in range(model.ngeom):
        gn = geom_name(model, gid).lower()
        bn = geom_body_name(model, gid).lower()
        s = gn + " " + bn
        if any(b in s for b in bad):
            continue
        if any(t in s for t in tokens):
            out.append(gid)
    return out


def min_geom_distance(model, data, hand_geoms, object_geoms, distmax=0.20):
    best = None
    fromto = np.zeros(6, dtype=float)

    for hg in hand_geoms:
        for og in object_geoms:
            try:
                d = float(mujoco.mj_geomDistance(model, data, int(hg), int(og), float(distmax), fromto))
            except Exception:
                continue
            if best is None or d < best["distance"]:
                best = {
                    "distance": d,
                    "hand_geom": geom_name(model, hg),
                    "object_geom": geom_name(model, og),
                }

    return best or {"distance": None, "hand_geom": None, "object_geom": None}


def load_result_q_grasp(path):
    if not path:
        return None
    p = resolve(path)
    if not p.exists():
        return None
    d = json.loads(p.read_text())
    q = d.get("q_grasp")
    if isinstance(q, dict):
        return q
    return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--npy", required=True)
    ap.add_argument("--sample-index", type=int, required=True)
    ap.add_argument("--object-body", default="grasp_object")
    ap.add_argument("--target-site", default="dataset_hand_base_debug")
    ap.add_argument("--result-json", default="")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--viewer", action="store_true")
    args = ap.parse_args()

    model_path = resolve(args.model)
    npy_path = resolve(args.npy)
    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    sample = load_sample(npy_path, args.sample_index)
    T_object_hand = extract_sample_T_object_hand(sample)
    sample_ctrl = extract_sample_ctrl(sample)

    T_world_object = object_world_T(model, data, args.object_body)
    T_world_hand_target = T_world_object @ T_object_hand

    object_geoms = collect_object_geoms(model, args.object_body)
    hand_geoms = collect_hand_geoms(model)

    # 1. 先对比当前 result.json 里的 q_grasp，如果提供了。
    old_compare = None
    old_q = load_result_q_grasp(args.result_json)
    if old_q:
        set_joint_values(model, data, Q_HOME)
        set_joint_values(model, data, old_q)
        set_joint_values(model, data, sample_ctrl)
        T_old_site = site_world_T(model, data, args.target_site)
        pos_e, rot_e, pos_n, rot_n = pose_error(T_old_site, T_world_hand_target)
        old_dist = min_geom_distance(model, data, hand_geoms, object_geoms)
        old_compare = {
            "q_grasp_source": rel(resolve(args.result_json)),
            "site_pos_err_norm_to_dataset_target": pos_n,
            "site_rot_err_norm_to_dataset_target": rot_n,
            "min_hand_object_distance": old_dist,
            "q_grasp": old_q,
            "site_pose": mat4_to_dict(T_old_site),
        }

    # 2. 直接做 dataset_hand_base_debug site IK。
    set_joint_values(model, data, Q_HOME)
    set_joint_values(model, data, sample_ctrl)
    ik_info = solve_site_ik(
        model, data, args.target_site, T_world_hand_target,
        max_iters=350,
        damping=1e-4,
        step_scale=0.85,
        rot_weight=0.65,
        pos_tol=8e-4,
        rot_tol=8e-3,
    )
    set_joint_values(model, data, sample_ctrl)

    T_ik_site = site_world_T(model, data, args.target_site)
    ik_pos_e, ik_rot_e, ik_pos_n, ik_rot_n = pose_error(T_ik_site, T_world_hand_target)
    ik_dist = min_geom_distance(model, data, hand_geoms, object_geoms)

    summary = {
        "format": "v4_13_frame_target_site_ik_verify_debug_v1",
        "model": rel(model_path),
        "npy": rel(npy_path),
        "sample_index_valid_local": args.sample_index,
        "object_body": args.object_body,
        "target_site": args.target_site,
        "available_sites": list_site_names(model),
        "T_world_object": mat4_to_dict(T_world_object),
        "T_object_hand_base_from_dataset": mat4_to_dict(T_object_hand),
        "T_world_hand_base_target": mat4_to_dict(T_world_hand_target),
        "sample_ctrl": sample_ctrl,
        "hand_geoms_count": len(hand_geoms),
        "object_geoms_count": len(object_geoms),
        "old_q_grasp_compare": old_compare,
        "site_ik": ik_info,
        "site_ik_final_pose": mat4_to_dict(T_ik_site),
        "site_ik_final_pos_err_norm": ik_pos_n,
        "site_ik_final_rot_err_norm": ik_rot_n,
        "site_ik_min_hand_object_distance": ik_dist,
        "diagnosis": {},
    }

    if old_compare:
        old_d = old_compare["min_hand_object_distance"]["distance"]
        summary["diagnosis"]["old_q_grasp_far_from_dataset_target"] = (
            old_compare["site_pos_err_norm_to_dataset_target"] > 0.015
            or (old_d is not None and old_d > 0.015)
        )

    ik_d = ik_dist["distance"]
    summary["diagnosis"]["site_ik_reaches_dataset_hand_target"] = (
        ik_info["success"] and ik_pos_n < 0.002 and ik_rot_n < 0.02
    )
    summary["diagnosis"]["site_ik_hand_near_object"] = (
        ik_d is not None and ik_d < 0.020
    )

    save_json(out_dir / "frame_target_site_ik_summary.json", summary)

    lines = []
    lines.append("========== V4.13 FRAME TARGET SITE IK VERIFY ==========")
    lines.append(f"model       : {rel(model_path)}")
    lines.append(f"npy         : {rel(npy_path)}")
    lines.append(f"sample_index: {args.sample_index}  # valid local index")
    lines.append(f"object_body : {args.object_body}")
    lines.append(f"target_site : {args.target_site}")
    lines.append("")
    lines.append("---- dataset target ----")
    lines.append(f"T_world_object.pos        : {T_world_object[:3, 3].tolist()}")
    lines.append(f"T_object_hand_base.pos    : {T_object_hand[:3, 3].tolist()}")
    lines.append(f"T_world_hand_target.pos   : {T_world_hand_target[:3, 3].tolist()}")
    lines.append("")

    if old_compare:
        lines.append("---- current P4U6 q_grasp compare ----")
        lines.append(f"site_pos_err_to_dataset_target : {old_compare['site_pos_err_norm_to_dataset_target']:.6f} m")
        lines.append(f"site_rot_err_to_dataset_target : {old_compare['site_rot_err_norm_to_dataset_target']:.6f}")
        lines.append(f"min_hand_object_distance       : {old_compare['min_hand_object_distance']}")
        lines.append("")

    lines.append("---- direct site IK to dataset_hand target ----")
    lines.append(f"ik_success          : {ik_info['success']}")
    lines.append(f"iters               : {ik_info['iters']}")
    lines.append(f"final_pos_err_norm  : {ik_pos_n:.6f} m")
    lines.append(f"final_rot_err_norm  : {ik_rot_n:.6f}")
    lines.append(f"q_arm               : {ik_info['q_arm']}")
    lines.append(f"min_hand_object_dist: {ik_dist}")
    lines.append("")

    lines.append("---- diagnosis ----")
    for k, v in summary["diagnosis"].items():
        lines.append(f"{k}: {v}")
    lines.append("")
    lines.append("判断标准：")
    lines.append("1. 如果 direct site IK 成功且 min_hand_object_distance 很小，说明数据集先验目标本身可达。")
    lines.append("2. 如果 current P4U6 q_grasp 离 dataset target 很远，说明 fr3_link7 target 转换/P2-P3 链路错。")
    lines.append("3. 这一步不做任何姿态微调，只验证 frame 和 target。")
    lines.append("=======================================================")

    report = "\n".join(lines) + "\n"
    (out_dir / "frame_target_site_ik_report.txt").write_text(report)
    print(report)

    if args.viewer:
        print("[VIEWER] showing direct site IK solution. Close viewer window to exit.")
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.cam.lookat[:] = np.asarray(T_world_object[:3, 3], dtype=float)
            viewer.cam.distance = 0.9
            viewer.cam.azimuth = 130
            viewer.cam.elevation = -25
            while viewer.is_running():
                viewer.sync()
                time.sleep(0.01)


if __name__ == "__main__":
    main()
