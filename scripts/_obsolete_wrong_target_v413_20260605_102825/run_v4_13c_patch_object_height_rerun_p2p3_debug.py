#!/usr/bin/env python3
"""
脚本类型：
    debug / v4.13c / scene-patch / object-height-align / light-p2p3

用途：
    修复 V4.13b 中随机物体 scene 生成后“物体没有完整露出支撑块”的问题。
    具体做法：
        1. 读取 V4.13b summary 中已生成的 Top-K scene/candidate；
        2. 用 MuJoCo 计算 object body 下所有 geom 的 world bbox；
        3. 自动检测 support/table/pedestal 这类支撑几何的 top_z；
        4. 将 object body 的 XML pos.z 平移，使 object_bbox_min_z 对齐到 support_top_z + clearance；
        5. 使用 patch 后的 scene 重新跑轻量 P2/P3；
        6. 只在有 PASS 或 near-pass 时生成 viewer 脚本。

输入：
    diagnostics/current_v413/core_bottle_v413b_topk_p2p3_debug/v4_13b_summary.json

输出：
    diagnostics/current_v413/core_bottle_v413c_height_patch_p2p3_debug/
        localXXX_.../
            scene_patched.xml
            scene_patch_info.json
            p2.json
            p3.json
            plan.json
            terminal_p2.txt
            terminal_p3.txt
        v4_13c_summary.txt
        v4_13c_summary.json
    scripts/05_execution_runner/run_v4_13c_selected_viewer_debug.sh

当前流程位置：
    V4.13b 已完成 Top-K candidate/scene/P2/P3
        -> 本脚本修正 object 初始高度
        -> 重新轻量 P2/P3
        -> 再决定是否 viewer

不负责：
    1. 不修改 legacy_final_demos；
    2. 不修改原始 builder；
    3. 不修改 P4U1/P4U6 源码；
    4. 不做单物体手工调参；
    5. 不把 can 迁移逻辑写入通用流程。
"""

from pathlib import Path
import argparse
import copy
import json
import math
import subprocess
import traceback
import xml.etree.ElementTree as ET

import mujoco
import numpy as np


PROJECT = Path.home() / "Projects/o7_mujoco_sim"

DEFAULT_V413B_SUMMARY = PROJECT / "diagnostics/current_v413/core_bottle_v413b_topk_p2p3_debug/v4_13b_summary.json"
DEFAULT_OUTROOT = PROJECT / "diagnostics/current_v413/core_bottle_v413c_height_patch_p2p3_debug"

P2_SCRIPT = PROJECT / "scripts/07_precheck_ik_collision/run_v4_12p2_pinocchio_multiseed_ik_debug.py"
P3_SCRIPT = PROJECT / "scripts/07_precheck_ik_collision/run_v4_12p3_pinocchio_ik_plus_mujoco_collision_precheck_debug.py"
P4U6_SCRIPT = PROJECT / "scripts/05_execution_runner/run_v4_12p4u6_ik_path_record_demo_debug.py"
URDF = PROJECT / "models/fr3_o7/fr3_o7_mujoco_ready_obj.urdf"
RUNNER_JSON = PROJECT / "diagnostics/current_v412/v4_12a_can52_strict_clearance_debug_runner.json"

CLEARANCE = 0.003

O7_ACTIVE_JOINTS = [
    "thumb_cmc_roll",
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]


def rel(p):
    p = Path(p)
    try:
        return str(p.relative_to(PROJECT))
    except Exception:
        return str(p)


def load_json(path):
    return json.loads(Path(path).read_text())


def save_json(path, obj):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, indent=2, ensure_ascii=False))


def run_cmd(cmd, log_path):
    log_path.parent.mkdir(parents=True, exist_ok=True)
    with log_path.open("w") as f:
        proc = subprocess.run(cmd, cwd=str(PROJECT), stdout=f, stderr=subprocess.STDOUT, text=True)
    return proc.returncode


def finite(x):
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def body_id(model, name):
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)


def geom_name(model, gid):
    return mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, gid) or ""


def mesh_vertices_world(model, data, gid):
    mid = int(model.geom_dataid[gid])
    if mid < 0:
        return None
    adr = int(model.mesh_vertadr[mid])
    num = int(model.mesh_vertnum[mid])
    verts = np.array(model.mesh_vert[adr:adr + num], dtype=float)
    R = np.array(data.geom_xmat[gid], dtype=float).reshape(3, 3)
    p = np.array(data.geom_xpos[gid], dtype=float)
    return verts @ R.T + p


def geom_bbox_world(model, data, gid):
    typ = int(model.geom_type[gid])
    p = np.array(data.geom_xpos[gid], dtype=float)
    R = np.array(data.geom_xmat[gid], dtype=float).reshape(3, 3)
    size = np.array(model.geom_size[gid], dtype=float)

    # mesh
    if typ == mujoco.mjtGeom.mjGEOM_MESH:
        verts = mesh_vertices_world(model, data, gid)
        if verts is not None and len(verts) > 0:
            return verts.min(axis=0), verts.max(axis=0)

    # box
    if typ == mujoco.mjtGeom.mjGEOM_BOX:
        corners = []
        for sx in [-1, 1]:
            for sy in [-1, 1]:
                for sz in [-1, 1]:
                    corners.append([sx * size[0], sy * size[1], sz * size[2]])
        pts = np.asarray(corners, dtype=float) @ R.T + p
        return pts.min(axis=0), pts.max(axis=0)

    # sphere / capsule / cylinder fallback
    radius = float(size[0]) if len(size) > 0 else 0.0
    half_len = float(size[1]) if len(size) > 1 else 0.0
    extent = radius + half_len
    mn = p - extent
    mx = p + extent
    return mn, mx


def object_bbox(model, data, object_body):
    bid = body_id(model, object_body)
    if bid < 0:
        raise RuntimeError(f"missing object body: {object_body}")

    mns = []
    mxs = []
    geoms = []

    for gid in range(model.ngeom):
        if int(model.geom_bodyid[gid]) == bid:
            mn, mx = geom_bbox_world(model, data, gid)
            mns.append(mn)
            mxs.append(mx)
            geoms.append(geom_name(model, gid))

    if not mns:
        raise RuntimeError(f"object body {object_body} has no direct geoms")

    return np.min(np.vstack(mns), axis=0), np.max(np.vstack(mxs), axis=0), geoms


def support_top_z(model, data, object_body):
    candidates = []
    for gid in range(model.ngeom):
        name = geom_name(model, gid).lower()
        if any(k in name for k in ["support", "pedestal", "table", "floor"]):
            # 排除物体自身
            bname = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, int(model.geom_bodyid[gid])) or ""
            if bname == object_body:
                continue
            mn, mx = geom_bbox_world(model, data, gid)
            candidates.append({
                "geom_id": int(gid),
                "geom_name": geom_name(model, gid),
                "top_z": float(mx[2]),
                "bbox_min": mn.tolist(),
                "bbox_max": mx.tolist(),
            })

    # 优先选 top_z 高于地面的局部支撑块，而不是地板
    filtered = [c for c in candidates if c["top_z"] > 0.02]
    if filtered:
        chosen = max(filtered, key=lambda c: c["top_z"])
    elif candidates:
        chosen = max(candidates, key=lambda c: c["top_z"])
    else:
        # 兜底：没有找到支撑几何就用当前物体最低点附近，这时不 patch
        chosen = None

    return chosen, candidates


def patch_scene_z(scene_in, scene_out, object_body, clearance):
    scene_in = Path(scene_in)
    scene_out = Path(scene_out)

    model = mujoco.MjModel.from_xml_path(str(scene_in))
    data = mujoco.MjData(model)
    mujoco.mj_forward(model, data)

    obj_mn, obj_mx, obj_geoms = object_bbox(model, data, object_body)
    support, support_candidates = support_top_z(model, data, object_body)

    if support is None:
        dz = 0.0
        target_min_z = float(obj_mn[2])
        support_z = None
    else:
        support_z = float(support["top_z"])
        target_min_z = support_z + clearance
        dz = target_min_z - float(obj_mn[2])

    tree = ET.parse(str(scene_in))
    root = tree.getroot()

    body = None
    for b in root.iter("body"):
        if b.attrib.get("name") == object_body:
            body = b
            break

    if body is None:
        raise RuntimeError(f"cannot find XML body: {object_body}")

    old_pos = np.array([float(x) for x in body.attrib.get("pos", "0 0 0").split()], dtype=float)
    new_pos = old_pos.copy()
    new_pos[2] += dz
    body.set("pos", f"{new_pos[0]:.12g} {new_pos[1]:.12g} {new_pos[2]:.12g}")

    scene_out.parent.mkdir(parents=True, exist_ok=True)
    tree.write(str(scene_out), encoding="utf-8", xml_declaration=True)

    # 验证 patch 后 bbox
    model2 = mujoco.MjModel.from_xml_path(str(scene_out))
    data2 = mujoco.MjData(model2)
    mujoco.mj_forward(model2, data2)
    obj_mn2, obj_mx2, _ = object_bbox(model2, data2, object_body)

    return {
        "scene_in": rel(scene_in),
        "scene_out": rel(scene_out),
        "object_body": object_body,
        "object_geoms": obj_geoms,
        "object_bbox_before_min": obj_mn.tolist(),
        "object_bbox_before_max": obj_mx.tolist(),
        "object_bbox_after_min": obj_mn2.tolist(),
        "object_bbox_after_max": obj_mx2.tolist(),
        "support_top_z": support_z,
        "support_chosen": support,
        "support_candidates": support_candidates,
        "clearance": clearance,
        "target_object_min_z": target_min_z,
        "old_body_pos": old_pos.tolist(),
        "new_body_pos": new_pos.tolist(),
        "dz": float(dz),
    }


def summarize_p3(path):
    path = Path(path)
    if not path.exists():
        return {"exists": False}
    try:
        d = load_json(path)
    except Exception as e:
        return {"exists": True, "parse_ok": False, "error": repr(e)}

    bp = d.get("best_pass")
    ba = d.get("best_available") or {}
    chosen = bp if bp is not None else ba
    out = {
        "exists": True,
        "parse_ok": True,
        "num_combos": d.get("num_combos"),
        "num_pass": d.get("num_pass"),
        "best_pass_exists": bp is not None,
        "chosen_is": "best_pass" if bp is not None else "best_available",
    }
    if isinstance(chosen, dict):
        out.update({
            "status": chosen.get("precheck_status"),
            "score": chosen.get("score"),
            "HS": chosen.get("min_path_hand_support_clearance"),
            "FO": chosen.get("min_path_fr3_object_clearance"),
            "GO": chosen.get("static_grasp_closed_hand_object_distance"),
            "HSc": chosen.get("static_grasp_closed_hand_support_clearance"),
            "margin": chosen.get("combo_min_joint_margin"),
            "hard_reasons": chosen.get("hard_reasons") or [],
        })
    return out


def score_row(row):
    p3 = row.get("p3", {})
    s = 0.0
    reasons = []

    if p3.get("best_pass_exists"):
        s += 160
        reasons.append("best_pass_exists")
    else:
        reasons.append("no_best_pass")

    n = p3.get("num_pass")
    if isinstance(n, int):
        s += min(n, 50) * 4
        reasons.append(f"num_pass={n}")

    go = p3.get("GO")
    if finite(go):
        go = float(go)
        if -0.010 <= go <= 0.012:
            s += 35
            reasons.append(f"GO_near_or_contact={go:.5f}")
        elif go > 0.030:
            s -= 55
            reasons.append(f"GO_too_far={go:.5f}")

    for key in ["HS", "HSc"]:
        c = p3.get(key)
        if finite(c):
            c = float(c)
            if c >= 0.006:
                s += 18
                reasons.append(f"{key}_clear_good={c:.5f}")
            elif c >= 0:
                s += 10
                reasons.append(f"{key}_nonnegative={c:.5f}")
            elif c >= -0.006:
                s += 2
                reasons.append(f"{key}_soft_touch={c:.5f}")
            elif c >= -0.018:
                s -= 18
                reasons.append(f"{key}_medium_penetration={c:.5f}")
            else:
                s -= 70
                reasons.append(f"{key}_hard_penetration={c:.5f}")

    fo = p3.get("FO")
    if finite(fo):
        fo = float(fo)
        if fo >= 0.006:
            s += 20
            reasons.append(f"FO_clear={fo:.5f}")
        elif fo >= 0:
            s += 6
            reasons.append(f"FO_nonnegative={fo:.5f}")
        elif fo >= -0.006:
            s -= 15
            reasons.append(f"FO_slight_collision={fo:.5f}")
        else:
            s -= 60
            reasons.append(f"FO_bad_collision={fo:.5f}")

    margin = p3.get("margin")
    if finite(margin):
        s += min(max(float(margin), 0), 1) * 4
        reasons.append(f"joint_margin={float(margin):.3f}")

    row["v413c_score"] = float(s)
    row["v413c_score_reasons"] = reasons
    return row


def get_candidate_ctrl(candidate_path):
    d = load_json(candidate_path)
    ctrl = d.get("hand", {}).get("o7_active_ctrl")
    if not isinstance(ctrl, dict):
        ctrl = d.get("o7_active_ctrl")
    if not isinstance(ctrl, dict):
        raise RuntimeError(f"candidate missing ctrl: {candidate_path}")
    return {j: float(ctrl[j]) for j in O7_ACTIVE_JOINTS}


def make_best_config(candidate_path, out_path):
    base_candidates = [
        PROJECT / "legacy_final_demos/final_sodacan165_sample014_p4u6_ready_gated_snap_demo/inputs/best_config.json",
        PROJECT / "legacy_final_demos/final_can52_p4u1_ready_gated_snap_demo/inputs/best_config.json",
        PROJECT / "diagnostics/current_v412/v4_12p4h2_can52_contact_sequence_proxy_debug/best_contact_sequence_config_ctrlsplit_debug.json",
    ]
    base_path = next((p for p in base_candidates if p.exists()), None)
    if base_path is None:
        raise FileNotFoundError("cannot find base best_config")

    cfg = load_json(base_path)
    ctrl = get_candidate_ctrl(candidate_path)
    cfg.setdefault("best_record", {})
    cfg["best_record"].setdefault("hand_config", {})
    cfg["best_record"]["hand_config"]["ctrl"] = ctrl
    cfg["best_record"]["hand_config"]["source"] = "v4_13c_candidate_ctrl_height_patched"
    cfg["v4_13c_debug_note"] = {
        "type": "height_patched_scene_candidate_best_config",
        "base_config": rel(base_path),
        "candidate": rel(candidate_path),
    }
    save_json(out_path, cfg)
    return base_path, ctrl


def make_viewer_script(best, best_config, outroot):
    script = PROJECT / "scripts/05_execution_runner/run_v4_13c_selected_viewer_debug.sh"
    viewer_out = outroot / "viewer_selected"

    text = f'''#!/usr/bin/env bash
# 脚本类型：
#   debug / execution-runner / v4.13c-selected-viewer
#
# 用途：
#   运行 V4.13c object-height-patched scene 的 rank1 candidate。
#
# 输出：
#   {rel(viewer_out)}/terminal.txt
#   {rel(viewer_out)}/result.json
#   {rel(viewer_out)}/path_plan.json
#
# 不负责：
#   不重新筛选，不修改 legacy demo，不修改 P4U6/P4U1 源码。

set -euo pipefail

cd "$HOME/Projects/o7_mujoco_sim"
source "$HOME/mujoco_env/bin/activate"

OUTDIR="{rel(viewer_out)}"
mkdir -p "$OUTDIR"

LOG="$OUTDIR/terminal.txt"
RESULT_JSON="$OUTDIR/result.json"
PATH_PLAN_JSON="$OUTDIR/path_plan.json"

exec > >(tee "$LOG") 2>&1

echo "========== V4.13C SELECTED VIEWER =========="
echo "tag          : {best.get('tag')}"
echo "grasp_type   : {best.get('grasp_type')}"
echo "scene        : {best.get('scene_patched')}"
echo "candidate    : {best.get('candidate')}"
echo "p3_json      : {best.get('p3_json')}"
echo "object_body  : {best.get('object_body')}"
echo "best_config  : {rel(best_config)}"
echo

./run_mujoco_clean.sh scripts/05_execution_runner/run_v4_12p4u6_ik_path_record_demo_debug.py \\
  --model "{best.get('scene_patched')}" \\
  --candidate "{best.get('candidate')}" \\
  --p3-json "{best.get('p3_json')}" \\
  --best-config "{rel(best_config)}" \\
  --which best_available \\
  --object-body "{best.get('object_body')}" \\
  --target-body fr3_link7 \\
  --out "$RESULT_JSON" \\
  --plan-out "$PATH_PLAN_JSON" \\
  --viewer \\
  --keep-viewer-open \\
  --start-arm-mode zero_clamped \\
  --start-hold-duration 0.8 \\
  --home-hold-duration 0.3 \\
  --pre-hold-duration 0.35 \\
  --grasp-settle-duration 0.25 \\
  --close-duration 0.45 \\
  --post-close-target-hold-duration 0.25 \\
  --micro-squeeze-duration 0.45 \\
  --micro-squeeze-fraction 0.05 \\
  --finger-close-scale 1.18 \\
  --thumb-pitch-from-finger-gain 0.24 \\
  --grip-ready-stable-steps 8 \\
  --min-live-non-thumb 1 \\
  --opposition-cos-threshold -0.30 \\
  --max-grip-disp 0.024 \\
  --max-extra-disp-during-squeeze 0.006 \\
  --approach-abort-disp 0.030 \\
  --approach-min-clearance 0.002 \\
  --grasp-path-min-clearance 0.001 \\
  --plan-attempts 3 \\
  --rrt-max-iters 1800 \\
  --rrt-step 0.30 \\
  --edge-step 0.045 \\
  --goal-bias 0.20 \\
  --shortcut-iters 60 \\
  --joint-speed-rad-s 0.85 \\
  --min-segment-duration 0.20 \\
  --hard-servo-approach \\
  --enable-lift \\
  --lift-z 0.090 \\
  --lift-duration 2.6 \\
  --final-hold-duration 0.9 \\
  --print-every-steps 100 \\
  --log-every-steps 100 \\
  --frame-sleep 0.0015

echo
echo "========== RESULT QUICK VIEW =========="
python3 - <<'R'
import json
from pathlib import Path
p = Path("{rel(viewer_out)}/result.json")
if p.exists():
    d = json.loads(p.read_text())
    for k in ["success","stop_reason","grip_ready","final_object_disp","final_object_rise","max_object_rise","final_groups","final_opposition_cos","max_stable_count"]:
        if k in d:
            print(k, ":", d[k])
R
'''
    script.write_text(text)
    script.chmod(0o755)
    return script


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--v413b-summary", default=str(DEFAULT_V413B_SUMMARY))
    ap.add_argument("--outroot", default=str(DEFAULT_OUTROOT))
    ap.add_argument("--max-samples", type=int, default=5)
    ap.add_argument("--clearance", type=float, default=CLEARANCE)
    args = ap.parse_args()

    src = Path(args.v413b_summary)
    if not src.is_absolute():
        src = PROJECT / src

    outroot = Path(args.outroot)
    if not outroot.is_absolute():
        outroot = PROJECT / outroot
    outroot.mkdir(parents=True, exist_ok=True)

    data = load_json(src)
    rows_in = data.get("rows_sorted", [])[: args.max_samples]

    rows = []

    for r in rows_in:
        tag = r["tag"]
        odir = outroot / tag
        odir.mkdir(parents=True, exist_ok=True)

        row = copy.deepcopy(r)
        row["out_dir_v413c"] = rel(odir)

        try:
            scene_in = PROJECT / r["scene"]
            candidate = PROJECT / r["candidate"]
            object_body = r["object_body"]
            scene_patched = odir / "scene_patched.xml"

            patch_info = patch_scene_z(scene_in, scene_patched, object_body, args.clearance)
            save_json(odir / "scene_patch_info.json", patch_info)

            row["scene_patched"] = rel(scene_patched)
            row["scene_patch_info"] = patch_info

            p2_json = odir / "p2.json"
            p3_json = odir / "p3.json"
            plan_json = odir / "plan.json"

            p2_cmd = [
                "python3", rel(P2_SCRIPT),
                "--urdf", rel(URDF),
                "--model", rel(scene_patched),
                "--candidate", rel(candidate),
            ]
            if RUNNER_JSON.exists():
                p2_cmd += ["--runner-json", rel(RUNNER_JSON)]
            p2_cmd += [
                "--object-body", object_body,
                "--target-frame", "fr3_link7",
                "--out", rel(p2_json),
                "--random-seeds", "4",
                "--random-std", "0.50",
                "--max-iters", "260",
                "--pos-tol", "0.0006",
                "--rot-tol", "0.006",
                "--rot-weight", "0.48",
            ]

            rc2 = run_cmd(p2_cmd, odir / "terminal_p2.txt")
            row["p2_return_code_v413c"] = rc2
            row["p2_json_v413c"] = rel(p2_json)

            if rc2 != 0:
                row["error_v413c"] = f"P2 failed rc={rc2}"
                rows.append(score_row(row))
                continue

            p3_cmd = [
                "python3", rel(P3_SCRIPT),
                "--p2-json", rel(p2_json),
                "--model", rel(scene_patched),
                "--candidate", rel(candidate),
                "--object-body", object_body,
                "--out", rel(p3_json),
                "--best-plan-out", rel(plan_json),
                "--top-per-target", "4",
                "--max-combos", "128",
                "--path-samples", "18",
                "--min-hand-support-clearance", "0.0",
                "--min-fr3-object-clearance", "0.0",
                "--max-grasp-hand-object-distance", "0.070",
                "--min-joint-margin", "0.0",
            ]

            rc3 = run_cmd(p3_cmd, odir / "terminal_p3.txt")
            row["p3_return_code_v413c"] = rc3
            row["p3_json"] = rel(p3_json)
            row["plan_json_v413c"] = rel(plan_json)
            if rc3 != 0:
                row["error_v413c"] = f"P3 failed rc={rc3}"

            row["p3"] = summarize_p3(p3_json)

        except Exception as e:
            row["error_v413c"] = repr(e)
            row["traceback_v413c"] = traceback.format_exc()

        rows.append(score_row(row))

    rows_sorted = sorted(rows, key=lambda x: x.get("v413c_score", -1e9), reverse=True)

    # 只有 PASS 或 near-pass 才生成 viewer
    executable = []
    for r in rows_sorted:
        p3 = r.get("p3", {})
        if not (r.get("scene_patched") and r.get("candidate") and r.get("p3_json")):
            continue
        if p3.get("best_pass_exists") or (
            finite(p3.get("GO")) and -0.010 <= float(p3["GO"]) <= 0.012
            and finite(p3.get("HS")) and float(p3["HS"]) >= -0.006
            and finite(p3.get("FO")) and float(p3["FO"]) >= -0.003
        ):
            executable.append(r)

    best = executable[0] if executable else None

    if best:
        best_config = outroot / "best_config_selected.json"
        base_cfg, ctrl = make_best_config(PROJECT / best["candidate"], best_config)
        best["best_config"] = rel(best_config)
        best["best_config_base"] = rel(base_cfg)
        best["best_config_ctrl"] = ctrl
        viewer_script = make_viewer_script(best, best_config, outroot)
        best["viewer_script"] = rel(viewer_script)

    summary = {
        "format": "v4_13c_height_patch_p2p3_debug_v1",
        "v413b_summary": rel(src),
        "outroot": rel(outroot),
        "clearance": args.clearance,
        "rows_sorted": rows_sorted,
        "best": best,
    }
    save_json(outroot / "v4_13c_summary.json", summary)

    lines = []
    lines.append("========== V4.13C OBJECT HEIGHT PATCH + LIGHT P2/P3 SUMMARY ==========")
    lines.append(f"v413b_summary: {rel(src)}")
    lines.append(f"outroot      : {rel(outroot)}")
    lines.append(f"clearance    : {args.clearance}")
    lines.append("")

    for i, r in enumerate(rows_sorted, start=1):
        p3 = r.get("p3", {})
        patch = r.get("scene_patch_info", {})
        lines.append(
            f"rank={i:02d} tag={r.get('tag')} score={r.get('v413c_score'):.3f} "
            f"type={r.get('grasp_type')} local={r.get('valid_local_index')} raw={r.get('raw_sample_index')} "
            f"dz={patch.get('dz')} obj_min_before={patch.get('object_bbox_before_min')} obj_min_after={patch.get('object_bbox_after_min')} "
            f"support_top={patch.get('support_top_z')} "
            f"pass={p3.get('num_pass')} status={p3.get('status')} "
            f"HS={p3.get('HS')} HSc={p3.get('HSc')} FO={p3.get('FO')} GO={p3.get('GO')} "
            f"p2_rc={r.get('p2_return_code_v413c')} p3_rc={r.get('p3_return_code_v413c')} err={r.get('error_v413c')}"
        )
        for rr in r.get("v413c_score_reasons", [])[:10]:
            lines.append(f"  - {rr}")
        for rr in (p3.get("hard_reasons") or [])[:3]:
            lines.append(f"  hard: {rr}")
        lines.append("")

    lines.append("---- BEST ----")
    if best:
        lines.append(f"best_tag: {best.get('tag')}")
        lines.append(f"scene_patched: {best.get('scene_patched')}")
        lines.append(f"candidate: {best.get('candidate')}")
        lines.append(f"object_body: {best.get('object_body')}")
        lines.append(f"p3_json: {best.get('p3_json')}")
        lines.append(f"best_config: {best.get('best_config')}")
        lines.append(f"viewer_script: {best.get('viewer_script')}")
        lines.append("")
        lines.append("run viewer:")
        lines.append(f"./{best.get('viewer_script')}")
    else:
        lines.append("best: None")
        lines.append("没有 PASS/near-pass；不要 viewer，先看 patch 后物体是否完整露出。")

    lines.append("======================================================================")
    txt = "\n".join(lines) + "\n"
    (outroot / "v4_13c_summary.txt").write_text(txt)
    print(txt)


if __name__ == "__main__":
    main()
