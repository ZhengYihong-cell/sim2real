#!/usr/bin/env python3
"""
脚本类型：
    debug / diagnostic / candidate-builder / scene-patcher

用途：
    为第一轮泛化目标 sem-SodaCan-16526... 生成初始 candidate JSON，
    并基于 can52 已验证稳定的 MuJoCo XML 复制/patch 出一个 sodacan 初始场景。

输入：
    1. --npy
       validate_results/seed1 下的 sodacan npy。
    2. --sample-indices
       要生成 candidate 的 sample index 列表。
    3. --base-scene
       can52 成功场景 XML，默认从 diagnostics/current_v412 自动查找。
    4. --object-mesh
       sodacan decomposed.obj。
    5. --object-body
       复用 can52 场景中的 object body 名称，默认 grasp_can。
    6. --out-dir
       输出目录。

输出：
    1. candidates/sampleXXX_candidate.json
    2. candidates/summary.json
    3. scene/sodacan165_sampleXXX_from_can52_scene_debug.xml
    4. build_report.txt

当前流程位置：
    数据集先验 -> 初始 candidate / 初始场景
    后续才进入 P4E/P4H/P4H2、P2/P3、P4U6/P4U1。

不负责：
    1. 不运行 IK；
    2. 不运行 P4E/P4H/P4H2；
    3. 不运行 viewer；
    4. 不修改 can52 成功 demo；
    5. 不修改 legacy_final_demos；
    6. 不保证该 candidate 最终一定能抓，只负责生成可进入下一阶段的初始输入。
"""

from pathlib import Path
import argparse
import json
import math
import shutil
import xml.etree.ElementTree as ET
import numpy as np


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

DEFAULT_TARGET = "sem-SodaCan-16526d147e837c386829bf9ee210f5e7"
DEFAULT_NPY = (
    "dataset/O7_Full_V8BestBaseline_165objs_20260422_084834/"
    "validate_results/seed1/sem-SodaCan-16526d147e837c386829bf9ee210f5e7.npy"
)
DEFAULT_OBJECT_MESH = (
    "dataset/meshdata/sem-SodaCan-16526d147e837c386829bf9ee210f5e7/"
    "coacd/decomposed.obj"
)
DEFAULT_OUT_DIR = "diagnostics/current_v412/sodacan165_sample014_initial_debug"


def resolve_path(p):
    p = Path(str(p)).expanduser()
    if not p.is_absolute():
        p = PROJECT / p
    return p


def rel(p):
    p = resolve_path(p)
    try:
        return str(p.relative_to(PROJECT))
    except Exception:
        return str(p)


def save_json(path, obj):
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_jsonable(obj), f, indent=2, ensure_ascii=False)


def write_text(path, text):
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        f.write(text)


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


def normalize(v, eps=1e-8):
    v = np.asarray(v, dtype=float)
    return v / max(float(np.linalg.norm(v)), eps)


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

    R = np.stack([x, y, z], axis=1)
    return R


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


def T_from_Rp(R, p):
    T = np.eye(4, dtype=float)
    T[:3, :3] = np.asarray(R, dtype=float).reshape(3, 3)
    T[:3, 3] = np.asarray(p, dtype=float).reshape(3)
    return T


def parse_vec3(s):
    vals = [float(x) for x in str(s).replace(",", " ").split() if x.strip()]
    if len(vals) != 3:
        raise RuntimeError(f"vec3 must have 3 numbers, got: {s}")
    return np.asarray(vals, dtype=float)


def parse_indices(s):
    return [int(x) for x in str(s).replace(",", " ").split() if x.strip()]


def load_sample(npy_path, sample_index):
    arr = np.load(resolve_path(npy_path), allow_pickle=True)
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


def candidate_from_sample(args, npy_path, sample_index, out_path):
    _, sample = load_sample(npy_path, sample_index)

    hp = np.asarray(sample["hand_pose"], dtype=float)
    if hp.shape[0] < 16:
        raise RuntimeError(f"sample {sample_index} hand_pose dim < 16: {hp.shape}")

    scale = float(sample.get("scale", 1.0))

    R_dataset = robust_rot6d_to_R(hp[3:9])
    t_dataset = hp[0:3]
    T_dataset_object_handbase = T_from_Rp(R_dataset, t_dataset)

    obj_pos = parse_vec3(args.object_frame_pos)
    obj_euler = parse_vec3(args.object_frame_euler)
    T_object_frame_correction = T_from_Rp(euler_xyz_to_R(*obj_euler), obj_pos)

    if args.mount_json:
        mount_path = resolve_path(args.mount_json)
        with open(mount_path, "r") as f:
            mount_data = json.load(f)
        T_fr3_handbase = np.asarray(mount_data["T_fr3_link7_hand_base_link"], dtype=float)
        mount_source = rel(mount_path)
    else:
        fr3_pos = parse_vec3(args.fr3_to_handbase_pos)
        fr3_euler = parse_vec3(args.fr3_to_handbase_euler)
        T_fr3_handbase = T_from_Rp(euler_xyz_to_R(*fr3_euler), fr3_pos)
        mount_source = "manual_default"

    T_object_handbase = T_object_frame_correction @ T_dataset_object_handbase
    T_object_fr3 = T_object_handbase @ np.linalg.inv(T_fr3_handbase)

    o7_ctrl = extract_active_ctrl(sample)

    candidate = {
        "format": "fr3_o7_grasp_candidate_v1",
        "candidate_name": f"{args.target}_sample{sample_index:03d}_initial_debug",
        "source": {
            "type": "build_sodacan165_initial_candidate_scene_debug",
            "target": args.target,
            "npy": rel(npy_path),
            "sample_index": int(sample_index),
            "scale": scale,
            "rot6d": "robust_compute_rotation_matrix_from_ortho6d_numpy",
            "meaning": "hand_pose is T_object_hand_base_link, converted to T_object_fr3_link7",
            "mount_source": mount_source,
            "fr3_to_handbase_pos": args.fr3_to_handbase_pos,
            "fr3_to_handbase_euler": args.fr3_to_handbase_euler,
        },
        "object": {
            "body": args.object_body,
            "token": args.object_body,
            "support_tokens": args.support_tokens,
            "spawn_source": "model",
            "T_world_object": np.eye(4).tolist(),
        },
        "target": {
            "body": args.target_body,
            "T_object_target": T_object_fr3.tolist(),
        },
        "hand": {
            "type": "o7_active_ctrl",
            "approach_policy": "side_open_then_ready_gated_snap_close",
            "close_policy": "P4U1_ready_gated_snap_close",
            "o7_active_ctrl": o7_ctrl,
        },
        "execution": {
            "pregrasp": {
                "mode": "world_z_offset",
                "z_offset": float(args.pregrasp_z),
            },
            "note": "This candidate only provides prior. Final execution should use P4E/P4H/P4H2 + P2/P3 + P4U6/P4U1.",
        },
        "validation": {
            "min_final_hand_object": 3,
            "min_final_rise": 0.005,
        },
    }

    save_json(out_path, candidate)

    return {
        "sample_index": int(sample_index),
        "scale": scale,
        "hand_pose_translation": t_dataset,
        "hand_pose_rot6d": hp[3:9],
        "o7_active_ctrl": o7_ctrl,
        "candidate": rel(out_path),
        "T_object_fr3_link7": T_object_fr3,
    }


def find_base_scene():
    candidates = [
        PROJECT / "diagnostics/current_v412/v4_12p4t2_scene_can52_contact_stable_old_ellipsoid_proxy.xml",
    ]

    for p in candidates:
        if p.exists():
            return p

    hits = []
    for root in ["diagnostics", "legacy_final_demos", "models", "records"]:
        rp = PROJECT / root
        if not rp.exists():
            continue
        for p in rp.rglob("*.xml"):
            name = str(p).lower()
            if "can52" in name and ("ellipsoid" in name or "contact_stable" in name or "p4t2" in name):
                hits.append(p)

    if hits:
        return sorted(hits, key=lambda x: len(str(x)))[0]

    raise RuntimeError("cannot find can52 stable base scene xml")


def collect_body_mesh_refs(body_elem):
    refs = set()
    for geom in body_elem.iter("geom"):
        mesh = geom.attrib.get("mesh", "")
        if mesh:
            refs.add(mesh)
    return refs


def find_body(root, body_name):
    for b in root.iter("body"):
        if b.attrib.get("name") == body_name:
            return b
    return None


def patch_scene_for_sodacan(args, sample_info, out_scene):
    base_scene = resolve_path(args.base_scene) if args.base_scene else find_base_scene()
    object_mesh = resolve_path(args.object_mesh)

    if not base_scene.exists():
        raise RuntimeError(f"base scene not found: {base_scene}")
    if not object_mesh.exists():
        raise RuntimeError(f"object mesh not found: {object_mesh}")

    tree = ET.parse(str(base_scene))
    root = tree.getroot()

    object_body = find_body(root, args.object_body)
    if object_body is None:
        raise RuntimeError(f"cannot find object body in base scene: {args.object_body}")

    mesh_refs = collect_body_mesh_refs(object_body)

    mesh_assets = {}
    for m in root.iter("mesh"):
        name = m.attrib.get("name", "")
        if name:
            mesh_assets[name] = m

    patched_mesh_assets = []

    # 优先 patch object body 内 geom 引用的 mesh asset。
    for ref in sorted(mesh_refs):
        m = mesh_assets.get(ref)
        if m is None:
            continue
        old_file = m.attrib.get("file", "")
        old_scale = m.attrib.get("scale", "")
        m.set("file", str(object_mesh))
        m.set("scale", f"{sample_info['scale']} {sample_info['scale']} {sample_info['scale']}")
        patched_mesh_assets.append({
            "mesh_name": ref,
            "old_file": old_file,
            "new_file": str(object_mesh),
            "old_scale": old_scale,
            "new_scale": f"{sample_info['scale']} {sample_info['scale']} {sample_info['scale']}",
        })

    # 兜底：如果 object body 没有 mesh geom，就 patch 名称里像 object/can/grasp 的 mesh。
    if not patched_mesh_assets:
        for name, m in mesh_assets.items():
            low = (name + " " + m.attrib.get("file", "")).lower()
            if "can" in low or "grasp" in low or "object" in low:
                old_file = m.attrib.get("file", "")
                old_scale = m.attrib.get("scale", "")
                m.set("file", str(object_mesh))
                m.set("scale", f"{sample_info['scale']} {sample_info['scale']} {sample_info['scale']}")
                patched_mesh_assets.append({
                    "mesh_name": name,
                    "old_file": old_file,
                    "new_file": str(object_mesh),
                    "old_scale": old_scale,
                    "new_scale": f"{sample_info['scale']} {sample_info['scale']} {sample_info['scale']}",
                    "fallback": True,
                })

    if not patched_mesh_assets:
        raise RuntimeError(
            "no mesh asset patched. Base scene may use primitive object geom; need inspect XML manually."
        )

    out_scene = resolve_path(out_scene)
    out_scene.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(out_scene), encoding="utf-8", xml_declaration=True)

    verify = verify_mujoco_scene(out_scene, args.object_body)

    return {
        "base_scene": rel(base_scene),
        "out_scene": rel(out_scene),
        "object_mesh": rel(object_mesh),
        "object_body": args.object_body,
        "patched_mesh_assets": patched_mesh_assets,
        "mujoco_verify": verify,
    }


def verify_mujoco_scene(scene_path, object_body):
    try:
        import mujoco
        model = mujoco.MjModel.from_xml_path(str(resolve_path(scene_path)))
        data = mujoco.MjData(model)
        mujoco.mj_forward(model, data)

        bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, object_body)
        out = {
            "load_ok": True,
            "nbody": int(model.nbody),
            "ngeom": int(model.ngeom),
            "nu": int(model.nu),
            "object_body_id": int(bid),
        }

        if bid >= 0:
            out["object_xpos"] = data.xpos[bid].copy()

        return out

    except Exception as e:
        return {
            "load_ok": False,
            "error": repr(e),
        }


def read_metric_for_samples(npy_path, target, indices):
    p = resolve_path(npy_path)
    parent = p.parent

    sidecars = {
        "metrics_npz": parent / f"{target}_metrics_valid_per_sample.npz",
        "flags_npz": parent / f"{target}_validation_flags.npz",
        "meta_json": parent / f"{target}_validation_meta.json",
        "metrics_json": parent / f"{target}_metrics_valid.json",
    }

    out = {
        "sidecars": {k: rel(v) for k, v in sidecars.items() if v.exists()},
        "samples": {},
    }

    if sidecars["metrics_npz"].exists():
        data = np.load(sidecars["metrics_npz"], allow_pickle=True)
        for idx in indices:
            item = out["samples"].setdefault(str(idx), {})
            for k in data.keys():
                arr = np.asarray(data[k])
                if arr.ndim >= 1 and idx < arr.shape[0]:
                    try:
                        item[k] = arr[idx].item() if hasattr(arr[idx], "item") else arr[idx]
                    except Exception:
                        item[k] = str(arr[idx])

    if sidecars["flags_npz"].exists():
        data = np.load(sidecars["flags_npz"], allow_pickle=True)
        for idx in indices:
            item = out["samples"].setdefault(str(idx), {})
            flags = {}
            for k in data.keys():
                arr = np.asarray(data[k])
                if arr.ndim >= 1 and idx < arr.shape[0]:
                    try:
                        val = arr[idx].item() if hasattr(arr[idx], "item") else arr[idx]
                    except Exception:
                        val = str(arr[idx])
                    flags[k] = val
            item["flags"] = flags

    return out


def make_report(summary):
    lines = []
    lines.append("========== SODACAN165 INITIAL CANDIDATE / SCENE BUILD REPORT ==========")
    lines.append(f"target       : {summary['target']}")
    lines.append(f"npy          : {summary['npy']}")
    lines.append(f"sample list  : {summary['sample_indices']}")
    lines.append(f"out_dir      : {summary['out_dir']}")
    lines.append("")

    lines.append("---- generated candidates ----")
    for item in summary["candidate_infos"]:
        lines.append(
            f"sample {item['sample_index']:03d} | "
            f"scale={item['scale']:.8f} | "
            f"candidate={item['candidate']}"
        )
        lines.append(f"  hand_pos={np.asarray(item['hand_pose_translation']).tolist()}")
        lines.append(f"  ctrl={item['o7_active_ctrl']}")

    lines.append("")
    lines.append("---- patched scene ----")
    scene = summary.get("scene_patch", {})
    lines.append(f"base_scene : {scene.get('base_scene')}")
    lines.append(f"out_scene  : {scene.get('out_scene')}")
    lines.append(f"verify     : {scene.get('mujoco_verify')}")
    lines.append("patched mesh assets:")
    for x in scene.get("patched_mesh_assets", []):
        lines.append(f"  {x}")

    lines.append("")
    lines.append("---- metrics for selected samples ----")
    for k, v in summary.get("metrics", {}).get("samples", {}).items():
        lines.append(f"sample {k}: {v}")

    lines.append("")
    lines.append("---- next ----")
    lines.append("把本报告发回后，再进入 P4E/P4H/P4H2 或 P2/P3。")
    lines.append("======================================================================")
    return "\n".join(lines) + "\n"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--target", default=DEFAULT_TARGET)
    ap.add_argument("--npy", default=DEFAULT_NPY)
    ap.add_argument("--sample-indices", default="14 4 23 17 22")
    ap.add_argument("--object-mesh", default=DEFAULT_OBJECT_MESH)
    ap.add_argument("--base-scene", default="")
    ap.add_argument("--object-body", default="grasp_can")
    ap.add_argument("--target-body", default="fr3_link7")
    ap.add_argument("--support-tokens", default="pedestal table support object_pedestal")
    ap.add_argument("--out-dir", default=DEFAULT_OUT_DIR)

    ap.add_argument("--object-frame-pos", default="0 0 0")
    ap.add_argument("--object-frame-euler", default="0 0 0")

    ap.add_argument("--mount-json", default="")
    ap.add_argument("--fr3-to-handbase-pos", default="0 0 0.107")
    ap.add_argument("--fr3-to-handbase-euler", default="0 0 3.141592653589793")

    ap.add_argument("--pregrasp-z", type=float, default=0.08)

    args = ap.parse_args()

    out_dir = resolve_path(args.out_dir)
    cand_dir = out_dir / "candidates"
    scene_dir = out_dir / "scene"
    cand_dir.mkdir(parents=True, exist_ok=True)
    scene_dir.mkdir(parents=True, exist_ok=True)

    npy_path = resolve_path(args.npy)
    sample_indices = parse_indices(args.sample_indices)

    summary = {
        "format": "sodacan165_initial_candidate_scene_debug_v1",
        "target": args.target,
        "npy": rel(npy_path),
        "sample_indices": sample_indices,
        "out_dir": rel(out_dir),
        "candidate_infos": [],
        "scene_patch": {},
        "metrics": {},
    }

    for idx in sample_indices:
        out_candidate = cand_dir / f"sample{idx:03d}_candidate.json"
        info = candidate_from_sample(args, npy_path, idx, out_candidate)
        summary["candidate_infos"].append(info)

    # 第一版场景使用第一个 sample 的 scale。
    first = summary["candidate_infos"][0]
    out_scene = scene_dir / f"sodacan165_sample{first['sample_index']:03d}_from_can52_scene_debug.xml"
    summary["scene_patch"] = patch_scene_for_sodacan(args, first, out_scene)

    summary["metrics"] = read_metric_for_samples(npy_path, args.target, sample_indices)

    save_json(out_dir / "summary.json", summary)
    report = make_report(summary)
    write_text(out_dir / "build_report.txt", report)

    print(report)
    print("saved summary:", rel(out_dir / "summary.json"))
    print("saved report :", rel(out_dir / "build_report.txt"))


if __name__ == "__main__":
    main()
