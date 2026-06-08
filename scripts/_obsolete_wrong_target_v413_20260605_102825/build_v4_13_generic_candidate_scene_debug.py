#!/usr/bin/env python3
"""
脚本类型：
    debug / v4.13 / generic-builder / candidate-scene / integrity-check

用途：
    通用生成任意数据集物体的 FR3+O7 MuJoCo scene 和 candidate。
    解决旧 build_sodacan165_initial_candidate_scene_debug.py 复用 can 模板导致：
        1. 随机物体 mesh 不完整显示；
        2. object frame / mesh bbox / support top 对齐不可靠；
        3. table 被误当成真实支撑面；
        4. 后续 selector/P2/P3 在错误 scene 上工作。

输入：
    1. object_code
    2. object.npy
    3. object mesh
    4. valid local sample index
    5. FR3+O7 template XML
    6. support pedestal 参数

输出：
    out_dir/
        scene.xml
        candidate.json
        integrity_report.txt
        integrity_summary.json

当前流程位置：
    V4.13 selector 选出 Top-K prior
        -> 本脚本生成真实通用 scene/candidate
        -> 后续 P2/P3
        -> 后续 P4U6/P4U1 viewer

不负责：
    1. 不跑 P2/P3；
    2. 不跑 viewer；
    3. 不修改 legacy_final_demos；
    4. 不修改 P4U1/P4U6；
    5. 不做沿某个轴的人工微调；
    6. 不替某个 sample 手工调姿态。
"""

from pathlib import Path
import argparse
import json
import math
import xml.etree.ElementTree as ET

import numpy as np
import mujoco


PROJECT = Path.home() / "Projects/o7_mujoco_sim"

O7_ACTIVE_JOINTS = [
    "thumb_cmc_roll",
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]


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


def parse_vec3(s):
    vals = [float(x) for x in str(s).replace(",", " ").split()]
    if len(vals) != 3:
        raise RuntimeError(f"need 3 values, got: {s}")
    return np.asarray(vals, dtype=float)


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


def Rx(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=float)


def Ry(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=float)


def Rz(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=float)


def euler_xyz_to_R(rx, ry, rz):
    return Rz(rz) @ Ry(ry) @ Rx(rx)


def mat_to_quat_wxyz(R):
    R = np.asarray(R, dtype=float)
    tr = float(np.trace(R))

    if tr > 0:
        s = math.sqrt(tr + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s

    q = np.asarray([w, x, y, z], dtype=float)
    return q / (np.linalg.norm(q) + 1e-12)


def T_from_Rp(R, p):
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R, dtype=float)
    T[:3, 3] = np.asarray(p, dtype=float)
    return T


def load_sample(npy_path, sample_index):
    arr = np.load(npy_path, allow_pickle=True)
    if sample_index < 0 or sample_index >= len(arr):
        raise RuntimeError(f"sample_index out of range: {sample_index}, n={len(arr)}")
    sample = arr[sample_index].item() if hasattr(arr[sample_index], "item") else arr[sample_index]
    if not isinstance(sample, dict):
        raise RuntimeError(f"sample is not dict: {type(sample)}")
    return arr, sample


def extract_active_ctrl(sample):
    hp = np.asarray(sample["hand_pose"], dtype=float)
    if hp.shape[0] >= 16:
        return {j: float(v) for j, v in zip(O7_ACTIVE_JOINTS, hp[9:16])}

    qpos = sample.get("qpos", {})
    if isinstance(qpos, dict) and all(j in qpos for j in O7_ACTIVE_JOINTS):
        return {j: float(qpos[j]) for j in O7_ACTIVE_JOINTS}

    raise RuntimeError("cannot extract O7 active ctrl from sample")


def read_obj_vertices(path):
    verts = []
    with open(path, "r", errors="ignore") as f:
        for line in f:
            if not line.startswith("v "):
                continue
            parts = line.strip().split()
            if len(parts) < 4:
                continue
            try:
                verts.append([float(parts[1]), float(parts[2]), float(parts[3])])
            except Exception:
                pass
    if not verts:
        raise RuntimeError(f"no vertices found in obj: {path}")
    return np.asarray(verts, dtype=float)


def ensure_material(asset, name, rgba):
    for m in list(asset.findall("material")):
        if m.get("name") == name:
            asset.remove(m)
    ET.SubElement(
        asset,
        "material",
        name=name,
        rgba=rgba,
        specular="0.2",
        shininess="0.3",
    )


def clean_old_scene_objects(worldbody, object_body):
    remove_body_names = {
        object_body,
        "grasp_object",
        "grasp_can",
        "grasp_bottle",
        "grasp_mug",
        "grasp_box",
        "dataset_object",
        "object_pedestal_body",
    }

    for b in list(worldbody.findall("body")):
        name = b.get("name", "")
        if name in remove_body_names or name.startswith("grasp_") or name.startswith("dataset_object"):
            worldbody.remove(b)

    # 只移除 worldbody 直连的旧支撑/旧桌面，不碰机器人内部 geom。
    for g in list(worldbody.findall("geom")):
        name = (g.get("name") or "").lower()
        if any(k in name for k in ["table", "support", "pedestal", "floor", "ground"]):
            worldbody.remove(g)


def clean_old_object_assets(asset):
    for m in list(asset.findall("mesh")):
        name = (m.get("name") or "").lower()
        file = (m.get("file") or "").lower()
        if (
            name.startswith("v413_object")
            or name.startswith("dataset_object")
            or name.startswith("grasp_object")
            or "bottle_piece" in name
            or "can" in name
            or "sodacan" in file
            or "bottle" in file
        ):
            asset.remove(m)


def bbox_from_world_verts(verts_world):
    return verts_world.min(axis=0), verts_world.max(axis=0)


def mj_geom_bbox_world(model, data, gid):
    typ = int(model.geom_type[gid])
    p = np.asarray(data.geom_xpos[gid], dtype=float)
    R = np.asarray(data.geom_xmat[gid], dtype=float).reshape(3, 3)
    size = np.asarray(model.geom_size[gid], dtype=float)

    if typ == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(model.geom_dataid[gid])
        if mid >= 0:
            adr = int(model.mesh_vertadr[mid])
            num = int(model.mesh_vertnum[mid])
            verts = np.asarray(model.mesh_vert[adr:adr + num], dtype=float)
            pts = verts @ R.T + p
            return pts.min(axis=0), pts.max(axis=0)

    if typ == mujoco.mjtGeom.mjGEOM_BOX:
        corners = []
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                for sz in [-1, 1]:
                    corners.append([sx * size[0], sy * size[1], sz * size[2]])
        pts = np.asarray(corners, dtype=float) @ R.T + p
        return pts.min(axis=0), pts.max(axis=0)

    radius = float(size[0]) if len(size) > 0 else 0.0
    half_len = float(size[1]) if len(size) > 1 else 0.0
    extent = radius + half_len
    return p - extent, p + extent


def integrity_check(scene_path, object_body):
    model = mujoco.MjModel.from_xml_path(str(scene_path))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
    if bid < 0:
        raise RuntimeError(f"missing object body in compiled model: {object_body}")

    obj_mn_list = []
    obj_mx_list = []
    obj_geoms = []

    support_candidates = []

    for gid in range(model.ngeom):
        gname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""
        bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])) or ""
        mn, mx = mj_geom_bbox_world(model, data, gid)

        if int(model.geom_bodyid[gid]) == bid:
            obj_mn_list.append(mn)
            obj_mx_list.append(mx)
            obj_geoms.append(gname)

        low = gname.lower()
        if any(k in low for k in ["object_pedestal", "pedestal", "support"]):
            support_candidates.append({
                "geom_name": gname,
                "body_name": bname,
                "bbox_min": mn.tolist(),
                "bbox_max": mx.tolist(),
                "top_z": float(mx[2]),
            })

    if not obj_mn_list:
        raise RuntimeError(f"object body has no geoms: {object_body}")

    obj_mn = np.vstack(obj_mn_list).min(axis=0)
    obj_mx = np.vstack(obj_mx_list).max(axis=0)

    support_top = None
    if support_candidates:
        support_top = max(x["top_z"] for x in support_candidates)

    return {
        "compiled_ok": True,
        "nbody": int(model.nbody),
        "ngeom": int(model.ngeom),
        "object_body": object_body,
        "object_body_xpos": data.xpos[bid].tolist(),
        "object_geoms": obj_geoms,
        "object_bbox_min": obj_mn.tolist(),
        "object_bbox_max": obj_mx.tolist(),
        "object_bbox_size": (obj_mx - obj_mn).tolist(),
        "support_candidates": support_candidates,
        "support_top_z": support_top,
        "object_bottom_minus_support_top": None if support_top is None else float(obj_mn[2] - support_top),
    }


def pick_template(user_template):
    if user_template:
        p = resolve(user_template)
        if not p.exists():
            raise FileNotFoundError(p)
        return p

    candidates = [
        PROJECT / "models/fr3_o7/main_xml/fr3_o7_bottle_scene_handbase_upright_tabletop_scale006_debug.xml",
        PROJECT / "models/fr3_o7/fr3_o7_bottle_scene_handbase_upright_tabletop_scale006_debug.xml",
        PROJECT / "models/fr3_o7/fr3_o7_bottle_scene_debug.xml",
        PROJECT / "models/fr3_o7/fr3_o7_actuated_scene_v1f_stable_hand.xml",
    ]
    for p in candidates:
        if p.exists():
            return p
    raise FileNotFoundError("cannot find FR3+O7 template XML")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--object-code", required=True)
    ap.add_argument("--npy", required=True)
    ap.add_argument("--sample-index", type=int, required=True, help="valid local index inside current .npy")
    ap.add_argument("--object-mesh", required=True)
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--template", default="")
    ap.add_argument("--object-body", default="grasp_object")
    ap.add_argument("--object-token", default="grasp_object")

    ap.add_argument("--support-center-xy", default="0.455 0.0")
    ap.add_argument("--support-half-size", default="0.045 0.045 0.115")
    ap.add_argument("--support-top-z", type=float, default=0.23)
    ap.add_argument("--object-clearance", type=float, default=0.003)

    ap.add_argument("--object-euler", default="0 0 0", help="world object orientation xyz euler, radians")
    ap.add_argument("--mesh-scale-override", type=float, default=0.0)

    ap.add_argument("--fr3-to-handbase-pos", default="0 0 0.107")
    ap.add_argument("--fr3-to-handbase-euler", default="0 0 3.141592653589793")

    args = ap.parse_args()

    out_dir = resolve(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    template = pick_template(args.template)
    npy_path = resolve(args.npy)
    object_mesh = resolve(args.object_mesh)

    _, sample = load_sample(npy_path, args.sample_index)

    hp = np.asarray(sample["hand_pose"], dtype=float)
    sample_scale = float(sample.get("scale", 1.0))
    mesh_scale = float(args.mesh_scale_override) if args.mesh_scale_override > 0 else sample_scale

    R_dataset = robust_rot6d_to_R(hp[3:9])
    t_dataset = hp[0:3]
    T_object_handbase = T_from_Rp(R_dataset, t_dataset)

    fr3_pos = parse_vec3(args.fr3_to_handbase_pos)
    fr3_euler = parse_vec3(args.fr3_to_handbase_euler)
    T_fr3_handbase = T_from_Rp(euler_xyz_to_R(*fr3_euler), fr3_pos)

    T_object_fr3 = T_object_handbase @ np.linalg.inv(T_fr3_handbase)
    o7_ctrl = extract_active_ctrl(sample)

    object_euler = parse_vec3(args.object_euler)
    R_world_object = euler_xyz_to_R(*object_euler)
    quat_world_object = mat_to_quat_wxyz(R_world_object)

    verts = read_obj_vertices(object_mesh)
    verts_scaled = verts * mesh_scale
    verts_world_rel = verts_scaled @ R_world_object.T
    rel_mn, rel_mx = bbox_from_world_verts(verts_world_rel)

    support_center_xy_vals = [float(x) for x in str(args.support_center_xy).replace(",", " ").split()]
    if len(support_center_xy_vals) != 2:
        raise RuntimeError("--support-center-xy needs 2 values")
    support_center_xy = np.asarray(support_center_xy_vals, dtype=float)

    support_half = parse_vec3(args.support_half_size)
    support_half[2] = args.support_top_z * 0.5

    object_pos = np.array([
        support_center_xy[0],
        support_center_xy[1],
        args.support_top_z + args.object_clearance - rel_mn[2],
    ], dtype=float)

    T_world_object = T_from_Rp(R_world_object, object_pos)

    tree = ET.parse(str(template))
    root = tree.getroot()

    asset = root.find("asset")
    if asset is None:
        asset = ET.SubElement(root, "asset")

    worldbody = root.find("worldbody")
    if worldbody is None:
        raise RuntimeError("template has no worldbody")

    ensure_material(asset, "v413_object_orange", "0.95 0.45 0.10 1")
    ensure_material(asset, "v413_pedestal_blue", "0.10 0.32 0.58 0.70")
    ensure_material(asset, "v413_floor_dark", "0.25 0.28 0.32 1")

    clean_old_object_assets(asset)
    clean_old_scene_objects(worldbody, args.object_body)

    mesh_name = "v413_object_mesh"
    ET.SubElement(
        asset,
        "mesh",
        name=mesh_name,
        file=str(object_mesh),
        scale=f"{mesh_scale:.12g} {mesh_scale:.12g} {mesh_scale:.12g}",
    )

    # 地面只用于视觉和基础碰撞，不命名为 table/support，避免 P3 把巨大地面当物体支撑块。
    ET.SubElement(
        worldbody,
        "geom",
        name="world_plane",
        type="plane",
        pos="0 0 0",
        size="1.5 1.5 0.02",
        material="v413_floor_dark",
        friction="1.2 0.005 0.0001",
        condim="3",
        contype="1",
        conaffinity="1",
    )

    ET.SubElement(
        worldbody,
        "geom",
        name="object_pedestal",
        type="box",
        pos=f"{support_center_xy[0]:.12g} {support_center_xy[1]:.12g} {args.support_top_z * 0.5:.12g}",
        size=f"{support_half[0]:.12g} {support_half[1]:.12g} {support_half[2]:.12g}",
        material="v413_pedestal_blue",
        group="0",
        condim="3",
        friction="1.2 0.005 0.0001",
        margin="0",
        gap="0",
        contype="1",
        conaffinity="1",
    )

    obj_body = ET.SubElement(
        worldbody,
        "body",
        name=args.object_body,
        pos=f"{object_pos[0]:.12g} {object_pos[1]:.12g} {object_pos[2]:.12g}",
        quat=f"{quat_world_object[0]:.12g} {quat_world_object[1]:.12g} {quat_world_object[2]:.12g} {quat_world_object[3]:.12g}",
    )

    ET.SubElement(obj_body, "freejoint", name=f"{args.object_body}_freejoint")

    ET.SubElement(
        obj_body,
        "geom",
        name=f"{args.object_body}_geom_0",
        type="mesh",
        mesh=mesh_name,
        material="v413_object_orange",
        friction="1.2 0.005 0.0001",
        condim="3",
        margin="0",
        gap="0",
        contype="2",
        conaffinity="5",
        density="300",
    )

    scene_path = out_dir / "scene.xml"
    tree.write(scene_path, encoding="utf-8", xml_declaration=True)

    integrity = integrity_check(scene_path, args.object_body)

    candidate = {
        "format": "fr3_o7_grasp_candidate_v413_generic",
        "candidate_name": f"{args.object_code}_sample{args.sample_index:03d}_generic_v413",
        "source": {
            "type": "build_v4_13_generic_candidate_scene_debug",
            "object_code": args.object_code,
            "npy": rel(npy_path),
            "sample_index_valid_local": int(args.sample_index),
            "sample_scale": sample_scale,
            "mesh_scale_used": mesh_scale,
            "object_mesh": rel(object_mesh),
            "template": rel(template),
            "rot6d": "robust_compute_rotation_matrix_from_ortho6d_numpy",
            "meaning": "hand_pose is T_object_hand_base_link, converted to T_object_fr3_link7",
        },
        "object": {
            "body": args.object_body,
            "token": args.object_token,
            "support_tokens": "object_pedestal pedestal support",
            "spawn_source": "generic_v413_builder",
            "T_world_object": T_world_object.tolist(),
            "object_pose": {
                "pos": object_pos.tolist(),
                "quat_wxyz": quat_world_object.tolist(),
                "euler_xyz": object_euler.tolist(),
            },
            "support": {
                "top_z": args.support_top_z,
                "clearance": args.object_clearance,
                "center_xy": support_center_xy.tolist(),
                "half_size": support_half.tolist(),
            },
        },
        "target": {
            "body": "fr3_link7",
            "T_object_target": T_object_fr3.tolist(),
        },
        "hand": {
            "type": "o7_active_ctrl",
            "o7_active_ctrl": o7_ctrl,
        },
        "execution": {
            "pregrasp": {
                "mode": "planner_generated_or_p2p3",
            },
            "close_duration": 0.45,
            "lift": {
                "mode": "world_z",
            },
        },
        "validation": {
            "min_final_hand_object": 2,
            "min_final_rise": 0.005,
            "need_thumb_plus_one_non_thumb": True,
        },
    }

    candidate_path = out_dir / "candidate.json"
    save_json(candidate_path, candidate)

    summary = {
        "scene": rel(scene_path),
        "candidate": rel(candidate_path),
        "integrity": integrity,
        "mesh_local_bbox_scaled_min": (verts * mesh_scale).min(axis=0).tolist(),
        "mesh_local_bbox_scaled_max": (verts * mesh_scale).max(axis=0).tolist(),
        "rotated_rel_bbox_min": rel_mn.tolist(),
        "rotated_rel_bbox_max": rel_mx.tolist(),
        "object_pos": object_pos.tolist(),
        "object_quat_wxyz": quat_world_object.tolist(),
        "T_object_fr3_link7": T_object_fr3.tolist(),
        "o7_active_ctrl": o7_ctrl,
    }
    save_json(out_dir / "integrity_summary.json", summary)

    lines = []
    lines.append("========== V4.13 GENERIC CANDIDATE/SCENE BUILDER ==========")
    lines.append(f"object_code      : {args.object_code}")
    lines.append(f"sample_index     : {args.sample_index}  # valid local index")
    lines.append(f"template         : {rel(template)}")
    lines.append(f"object_mesh      : {rel(object_mesh)}")
    lines.append(f"sample_scale     : {sample_scale}")
    lines.append(f"mesh_scale_used  : {mesh_scale}")
    lines.append(f"scene            : {rel(scene_path)}")
    lines.append(f"candidate        : {rel(candidate_path)}")
    lines.append("")
    lines.append("---- object placement ----")
    lines.append(f"support_top_z    : {args.support_top_z}")
    lines.append(f"object_clearance : {args.object_clearance}")
    lines.append(f"object_pos       : {object_pos.tolist()}")
    lines.append(f"object_quat_wxyz : {quat_world_object.tolist()}")
    lines.append("")
    lines.append("---- integrity ----")
    lines.append(f"compiled_ok      : {integrity['compiled_ok']}")
    lines.append(f"object_body      : {integrity['object_body']}")
    lines.append(f"object_bbox_min  : {integrity['object_bbox_min']}")
    lines.append(f"object_bbox_max  : {integrity['object_bbox_max']}")
    lines.append(f"object_bbox_size : {integrity['object_bbox_size']}")
    lines.append(f"support_top_z    : {integrity['support_top_z']}")
    lines.append(f"bottom-support   : {integrity['object_bottom_minus_support_top']}")
    lines.append("")
    lines.append("---- support candidates ----")
    for s in integrity["support_candidates"]:
        lines.append(f"  {s['geom_name']} top_z={s['top_z']} bbox_min={s['bbox_min']} bbox_max={s['bbox_max']}")
    lines.append("")
    lines.append("---- ctrl ----")
    for k, v in o7_ctrl.items():
        lines.append(f"{k:18s}: {v:+.6f}")
    lines.append("===========================================================")

    report = "\n".join(lines) + "\n"
    (out_dir / "integrity_report.txt").write_text(report)
    print(report)


if __name__ == "__main__":
    main()
