#!/usr/bin/env python3
"""
脚本类型：
    debug / viewer / geom-group-inspector

用途：
    用于检查 MuJoCo XML 中视觉 mesh、碰撞 mesh、ellipsoid proxy、COACD proxy 是否是旧版调好的手部模型。
    本脚本只负责加载模型并提供按键切换显示组，不做抓取、不做控制、不改 XML。

输入：
    --model
        要查看的 MuJoCo XML。
    --qpos-json
        可选。如果提供，会尝试读取其中 qpos / q_grasp / q_pre / q_lift 字段并设置关节角。
    --which
        从 qpos-json 中选择 qpos / q_pre / q_grasp / q_lift。
    --alpha
        默认透明度，用于看内部碰撞体。

输出：
    打开 MuJoCo viewer。

按键：
    1 : 只看 group 0
    2 : 只看 group 1
    3 : 只看 group 2
    4 : 只看 group 3
    5 : 只看 group 4
    6 : 只看 group 5
    0 : 显示所有 group
    v : 尝试只看 visual，也就是 group 1/2/3
    c : 尝试只看 collision/proxy，也就是 group 0/4/5
    t : 开关透明
    r : reset
    h : 打印帮助
    Esc : 退出

注意：
    不同 XML 的 group 编号可能不同。
    旧 O7-only v8/v18b 场景里 ellipsoid proxy 很可能不一定是 group 4。
    所以最可靠的方式是逐个按 1~6 看哪一组显示的是手指表面/碰撞体。
"""

from pathlib import Path
import argparse
import json
import time
import numpy as np
import mujoco
import mujoco.viewer


PROJECT = Path.home() / "Projects/o7_mujoco_sim"


def resolve_path(p):
    p = Path(str(p)).expanduser()
    if not p.is_absolute():
        p = PROJECT / p
    return p


def print_help():
    print("""
========== VIEWER KEYS ==========
1 : show only geom group 0
2 : show only geom group 1
3 : show only geom group 2
4 : show only geom group 3
5 : show only geom group 4
6 : show only geom group 5
0 : show all geom groups
v : visual guess, show groups 1/2/3
c : collision/proxy guess, show groups 0/4/5
t : toggle transparent
r : reset qpos
h : print help
Esc : quit
=================================
""")


def set_group(viewer, groups):
    for i in range(len(viewer.opt.geomgroup)):
        viewer.opt.geomgroup[i] = 1 if i in groups else 0
    print("[GEOM GROUP]", list(groups))


def load_qpos_dict(path, which):
    if not path:
        return None
    p = resolve_path(path)
    obj = json.load(open(p, "r"))

    if which in obj and isinstance(obj[which], dict):
        return obj[which]

    if "qpos" in obj and isinstance(obj["qpos"], dict):
        return obj["qpos"]

    if "q_grasp" in obj and isinstance(obj["q_grasp"], dict):
        return obj["q_grasp"]

    if "plan" in obj and isinstance(obj["plan"], dict):
        if which in obj["plan"] and isinstance(obj["plan"][which], dict):
            return obj["plan"][which]

    return None


def apply_qpos_dict(model, data, qpos_dict):
    if not qpos_dict:
        return []

    applied = []
    for name, val in qpos_dict.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        qadr = int(model.jnt_qposadr[jid])
        if qadr < model.nq:
            data.qpos[qadr] = float(val)
            applied.append(name)

    mujoco.mj_forward(model, data)
    return applied


def print_geom_summary(model):
    counts = {}
    by_group = {}
    for gid in range(model.ngeom):
        typ = int(model.geom_type[gid])
        group = int(model.geom_group[gid])
        counts[typ] = counts.get(typ, 0) + 1
        by_group[group] = by_group.get(group, 0) + 1

    type_names = {
        int(mujoco.mjtGeom.mjGEOM_PLANE): "plane",
        int(mujoco.mjtGeom.mjGEOM_SPHERE): "sphere",
        int(mujoco.mjtGeom.mjGEOM_CAPSULE): "capsule",
        int(mujoco.mjtGeom.mjGEOM_ELLIPSOID): "ellipsoid",
        int(mujoco.mjtGeom.mjGEOM_CYLINDER): "cylinder",
        int(mujoco.mjtGeom.mjGEOM_BOX): "box",
        int(mujoco.mjtGeom.mjGEOM_MESH): "mesh",
    }

    print("========== GEOM SUMMARY ==========")
    print("ngeom:", model.ngeom)
    print("by type:")
    for k, v in sorted(counts.items()):
        print(" ", type_names.get(k, str(k)), v)
    print("by group:")
    for k, v in sorted(by_group.items()):
        print(" ", k, v)
    print("==================================")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--qpos-json", default="")
    ap.add_argument("--which", default="q_grasp")
    ap.add_argument("--alpha", type=float, default=0.65)
    args = ap.parse_args()

    model_path = resolve_path(args.model)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)

    mujoco.mj_resetData(model, data)

    qpos_dict = load_qpos_dict(args.qpos_json, args.which)
    applied = apply_qpos_dict(model, data, qpos_dict)

    print("model:", model_path)
    print("qpos_json:", args.qpos_json)
    print("which:", args.which)
    print("applied joints:", applied)
    print_geom_summary(model)
    print_help()

    transparent = False

    def key_callback(keycode):
        nonlocal transparent

        # GLFW key codes for number row and letters.
        if keycode == 256:  # Esc
            return

        ch = None
        try:
            ch = chr(keycode).lower()
        except Exception:
            pass

        if ch == "1":
            set_group(viewer, [0])
        elif ch == "2":
            set_group(viewer, [1])
        elif ch == "3":
            set_group(viewer, [2])
        elif ch == "4":
            set_group(viewer, [3])
        elif ch == "5":
            set_group(viewer, [4])
        elif ch == "6":
            set_group(viewer, [5])
        elif ch == "0":
            set_group(viewer, range(len(viewer.opt.geomgroup)))
        elif ch == "v":
            set_group(viewer, [1, 2, 3])
        elif ch == "c":
            set_group(viewer, [0, 4, 5])
        elif ch == "t":
            transparent = not transparent
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = transparent
            print("[TRANSPARENT]", transparent)
        elif ch == "r":
            mujoco.mj_resetData(model, data)
            apply_qpos_dict(model, data, qpos_dict)
            print("[RESET]")
        elif ch == "h":
            print_help()

    with mujoco.viewer.launch_passive(model, data, key_callback=key_callback) as viewer:
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = False
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = False

        set_group(viewer, range(len(viewer.opt.geomgroup)))

        while viewer.is_running():
            viewer.sync()
            time.sleep(0.01)


if __name__ == "__main__":
    main()
