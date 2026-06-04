#!/usr/bin/env python3
"""
文件名：
    patch_v4_12p4d_opposition_corridor_candidate_debug.py

脚本类别：
    debug / candidate-patch / grasp-pose-refinement

用途：
    本脚本用于 V4.12P4D 阶段，根据当前 P3 best q_grasp 下的 O7 手与物体相对几何，
    自动修正 candidate 的 target.T_object_target 平移部分。
    修正目标不是固定 XYZ 偏移，而是让物体中心进入“大拇指—四指”的自然对握通道。

核心思想：
    1. 在 MuJoCo 中设置 q_grasp 和 side-open 手型。
    2. 计算 thumb 侧代表点 p_thumb。
    3. 计算四指侧代表点 p_four。
    4. 对握通道中心点 p_mid = (p_thumb + p_four) / 2。
    5. 物体中心 c_obj 应该接近 p_mid。
    6. 若 c_obj 明显偏向 thumb 或 four-finger 一侧，则平移 hand target：
           delta_world = c_obj - p_mid
       使新的对握通道中心更接近物体中心。

输入：
    1. --model
       当前 MuJoCo XML。
    2. --candidate
       原 candidate JSON，必须包含 target.T_object_target。
    3. --p3-json
       P3 输出 JSON，用于读取 best_available / best_pass 的 q_grasp。
    4. --object-body
       物体 body 名称，例如 grasp_can。
    5. side-open thumb preshape 参数。

输出：
    1. 修正后的 candidate JSON。
    2. 诊断 JSON，记录 thumb/four/object 的几何关系和 delta。
    3. 终端打印修正前后的对握通道误差。

当前流程位置：
    P3 best q_grasp
        -> P4D opposition corridor candidate patch
        -> 重新运行 P2/P3/P4C
        -> 检查四指轻闭合是否能接触物体

本脚本不负责：
    1. 不重新求 IK。
    2. 不执行动态抓取。
    3. 不修改物体位置。
    4. 不修改闭合控制逻辑。
    5. 不保证一次修正成功，只生成几何依据明确的新 candidate。
"""

from pathlib import Path
import argparse
import json
import numpy as np
import mujoco


PROJECT = Path.home() / "Projects/o7_mujoco_sim"

ARM_JOINTS = [
    "fr3_joint1", "fr3_joint2", "fr3_joint3", "fr3_joint4",
    "fr3_joint5", "fr3_joint6", "fr3_joint7",
]

ACTIVE_HAND_JOINTS = [
    "thumb_cmc_roll",
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]


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


def mj_id(model, objtype, name):
    return mujoco.mj_name2id(model, objtype, str(name))


def body_id(model, name):
    return mj_id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def joint_id(model, name):
    return mj_id(model, mujoco.mjtObj.mjOBJ_JOINT, name)


def actuator_id(model, name):
    return mj_id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def geom_name(model, gid):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, int(gid))
    return name or f"geom_{gid}"


def body_name(model, bid):
    name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(bid))
    return name or f"body_{bid}"


def body_name_of_geom(model, gid):
    return body_name(model, int(model.geom_bodyid[int(gid)]))


def set_joint_qpos(model, data, name, value):
    jid = joint_id(model, name)
    if jid < 0:
        return False
    jtype = int(model.jnt_type[jid])
    if jtype not in [int(mujoco.mjtJoint.mjJNT_HINGE), int(mujoco.mjtJoint.mjJNT_SLIDE)]:
        return False
    adr = int(model.jnt_qposadr[jid])
    data.qpos[adr] = float(value)
    return True


def set_actuator_ctrl(model, data, name, value):
    names = [name, f"{name}_pos", f"{name}_ctrl", f"{name}_act", f"{name}_motor"]
    for nm in names:
        aid = actuator_id(model, nm)
        if aid >= 0:
            data.ctrl[aid] = float(value)
            return True
    return False


def apply_qdict(model, data, qdict, also_ctrl=True):
    for k, v in (qdict or {}).items():
        set_joint_qpos(model, data, k, v)
        if also_ctrl:
            set_actuator_ctrl(model, data, k, v)


def selected_best(p3, which):
    item = p3.get(which)
    if item is None:
        raise RuntimeError(f"{which} is None in p3 json")
    if "q_grasp" not in item:
        raise RuntimeError(f"{which} has no q_grasp")
    return item


def T_from_Rp(R, p):
    T = np.eye(4)
    T[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(p, dtype=float).reshape(3)
    return T


def T_inv(T):
    out = np.eye(4)
    out[:3, :3] = T[:3, :3].T
    out[:3, 3] = -T[:3, :3].T @ T[:3, 3]
    return out


def object_T_world(model, data, object_body):
    bid = body_id(model, object_body)
    if bid < 0:
        raise RuntimeError(f"cannot find object body: {object_body}")
    return T_from_Rp(data.xmat[bid].reshape(3, 3), data.xpos[bid])


def classify_geom_group(model, gid):
    text = (geom_name(model, gid) + " " + body_name_of_geom(model, gid)).lower()
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
    return ""


def representative_points(model, data, object_body):
    T_obj = object_T_world(model, data, object_body)
    c_obj = T_obj[:3, 3].copy()

    groups = {
        "thumb": [],
        "index": [],
        "middle": [],
        "ring": [],
        "pinky": [],
    }

    for gid in range(model.ngeom):
        group = classify_geom_group(model, gid)
        if not group:
            continue

        pos = data.geom_xpos[gid].copy()
        dist = float(np.linalg.norm(pos - c_obj))
        groups[group].append({
            "geom_id": int(gid),
            "geom": geom_name(model, gid),
            "body": body_name_of_geom(model, gid),
            "pos": pos,
            "dist_to_object_center": dist,
        })

    for g in groups:
        groups[g].sort(key=lambda x: x["dist_to_object_center"])

    if not groups["thumb"]:
        raise RuntimeError("no thumb geoms found")
    if not any(groups[g] for g in ["index", "middle", "ring", "pinky"]):
        raise RuntimeError("no four-finger geoms found")

    # thumb 代表点：离物体中心最近的 thumb geom 中心
    thumb_rep = groups["thumb"][0]

    # 四指代表点：每个四指 group 取最近 geom，再求平均
    four_reps = []
    for g in ["index", "middle", "ring", "pinky"]:
        if groups[g]:
            four_reps.append(groups[g][0])

    p_thumb = thumb_rep["pos"]
    p_four = np.mean([x["pos"] for x in four_reps], axis=0)

    return {
        "object_center": c_obj,
        "thumb_rep": thumb_rep,
        "four_reps": four_reps,
        "p_thumb": p_thumb,
        "p_four": p_four,
        "groups_preview": {
            g: groups[g][:5] for g in groups
        },
    }


def compute_corridor_delta(rep, args):
    c = rep["object_center"]
    p_thumb = rep["p_thumb"]
    p_four = rep["p_four"]

    v = p_four - p_thumb
    vv = float(np.dot(v, v))
    if vv < 1e-12:
        raise RuntimeError("thumb-four vector too small")

    alpha = float(np.dot(c - p_thumb, v) / vv)
    p_mid = 0.5 * (p_thumb + p_four)

    raw_delta = c - p_mid

    if args.project == "xy":
        raw_delta[2] = 0.0
    elif args.project == "opposition_axis":
        axis = v / (np.linalg.norm(v) + 1e-12)
        raw_delta = axis * float(np.dot(raw_delta, axis))
        raw_delta[2] = 0.0
    elif args.project == "xyz":
        pass
    else:
        raise RuntimeError(f"unknown project mode: {args.project}")

    delta = float(args.gain) * raw_delta
    n = float(np.linalg.norm(delta))
    if n > args.max_shift:
        delta = delta * (args.max_shift / (n + 1e-12))

    corrected_mid = p_mid + delta
    err_before = c - p_mid
    err_after = c - corrected_mid

    return {
        "alpha_object_on_thumb_four_axis": alpha,
        "p_mid": p_mid,
        "err_before": err_before,
        "err_before_norm": float(np.linalg.norm(err_before)),
        "raw_delta_world": raw_delta,
        "delta_world": delta,
        "delta_world_norm": float(np.linalg.norm(delta)),
        "corrected_mid": corrected_mid,
        "err_after": err_after,
        "err_after_norm": float(np.linalg.norm(err_after)),
    }


def patch_candidate_target(candidate, model, data, object_body, delta_world):
    if "target" not in candidate or "T_object_target" not in candidate["target"]:
        raise RuntimeError("candidate missing target.T_object_target")

    T_object_target_old = np.asarray(candidate["target"]["T_object_target"], dtype=float)
    T_world_object = object_T_world(model, data, object_body)
    T_world_target_old = T_world_object @ T_object_target_old

    T_world_target_new = T_world_target_old.copy()
    T_world_target_new[:3, 3] += np.asarray(delta_world, dtype=float).reshape(3)

    T_object_target_new = T_inv(T_world_object) @ T_world_target_new

    patched = json.loads(json.dumps(candidate))
    patched["target"]["T_object_target"] = T_object_target_new.tolist()

    meta = patched.setdefault("debug_patch_meta", {})
    meta["v4_12p4d_opposition_corridor_patch"] = {
        "delta_world": np.asarray(delta_world).tolist(),
        "T_object_target_old": T_object_target_old.tolist(),
        "T_object_target_new": T_object_target_new.tolist(),
    }

    return patched, {
        "T_world_object": T_world_object,
        "T_world_target_old": T_world_target_old,
        "T_world_target_new": T_world_target_new,
        "T_object_target_old": T_object_target_old,
        "T_object_target_new": T_object_target_new,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--p3-json", required=True)
    ap.add_argument("--which", default="best_available", choices=["best_available", "best_pass"])
    ap.add_argument("--object-body", default="grasp_can")
    ap.add_argument("--out-candidate", required=True)
    ap.add_argument("--out-diagnostic", required=True)

    ap.add_argument("--thumb-roll-preshape", type=float, default=0.56)
    ap.add_argument("--thumb-yaw-preshape", type=float, default=1.15)
    ap.add_argument("--thumb-pitch-open", type=float, default=0.08)

    ap.add_argument("--project", default="xy", choices=["xy", "xyz", "opposition_axis"])
    ap.add_argument("--gain", type=float, default=0.8)
    ap.add_argument("--max-shift", type=float, default=0.035)

    args = ap.parse_args()

    model_path = resolve_path(args.model)
    cand_path = resolve_path(args.candidate)
    p3_path = resolve_path(args.p3_json)

    candidate = load_json(cand_path)
    p3 = load_json(p3_path)
    best = selected_best(p3, args.which)

    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    mujoco.mj_resetData(model, data)
    mujoco.mj_forward(model, data)

    q_grasp = best["q_grasp"]
    apply_qdict(model, data, q_grasp, also_ctrl=True)

    side_open_ctrl = {
        "thumb_cmc_roll": args.thumb_roll_preshape,
        "thumb_cmc_yaw": args.thumb_yaw_preshape,
        "thumb_cmc_pitch": args.thumb_pitch_open,
        "index_mcp_pitch": 0.0,
        "middle_mcp_pitch": 0.0,
        "ring_mcp_pitch": 0.0,
        "pinky_mcp_pitch": 0.0,
    }
    apply_qdict(model, data, side_open_ctrl, also_ctrl=True)
    mujoco.mj_forward(model, data)

    rep = representative_points(model, data, args.object_body)
    delta_info = compute_corridor_delta(rep, args)

    patched, patch_info = patch_candidate_target(
        candidate,
        model,
        data,
        args.object_body,
        delta_info["delta_world"],
    )

    save_json(args.out_candidate, patched)

    diagnostic = {
        "format": "v4_12p4d_opposition_corridor_candidate_patch_debug",
        "model": str(model_path),
        "candidate": str(cand_path),
        "p3_json": str(p3_path),
        "which": args.which,
        "object_body": args.object_body,
        "args": vars(args),
        "q_grasp": q_grasp,
        "side_open_ctrl": side_open_ctrl,
        "representative_points": rep,
        "delta_info": delta_info,
        "patch_info": patch_info,
        "out_candidate": str(resolve_path(args.out_candidate)),
    }
    save_json(args.out_diagnostic, diagnostic)

    print("\n========== V4.12P4D OPPOSITION CORRIDOR PATCH ==========")
    print("model        :", model_path)
    print("candidate    :", cand_path)
    print("p3_json      :", p3_path)
    print("object_body  :", args.object_body)
    print("out_candidate:", resolve_path(args.out_candidate))
    print("out_diag     :", resolve_path(args.out_diagnostic))
    print()
    print("object_center:", rep["object_center"])
    print("p_thumb      :", rep["p_thumb"])
    print("p_four       :", rep["p_four"])
    print("p_mid        :", delta_info["p_mid"])
    print("alpha        :", delta_info["alpha_object_on_thumb_four_axis"])
    print("err_before   :", delta_info["err_before"], "norm=", delta_info["err_before_norm"])
    print("delta_world  :", delta_info["delta_world"], "norm=", delta_info["delta_world_norm"])
    print("err_after    :", delta_info["err_after"], "norm=", delta_info["err_after_norm"])
    print()
    print("thumb_rep:", rep["thumb_rep"]["geom"], rep["thumb_rep"]["body"], rep["thumb_rep"]["pos"])
    print("four_reps:")
    for x in rep["four_reps"]:
        print("  ", x["geom"], x["body"], x["pos"], "dist=", x["dist_to_object_center"])
    print("========================================================\n")


if __name__ == "__main__":
    main()
