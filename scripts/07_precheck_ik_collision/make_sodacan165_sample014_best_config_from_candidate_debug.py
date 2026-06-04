#!/usr/bin/env python3
"""
脚本类型：
    debug / config-builder / generalization-adapter

用途：
    为 sem-SodaCan-16526 sample014 生成专属 best_config。
    原因是 P4U6 会从 --best-config 读取 best_record.hand_config.ctrl；
    如果继续使用 can52 的 best_config，会导致 P4U6 在 q_grasp 路径规划阶段使用 can52 手型，
    从而可能出现 pre_to_grasp goal invalid/contact。

输入：
    1. --base-config
       can52 已验证的 best_contact_sequence_config_ctrlsplit_debug.json，用作结构模板。
    2. --candidate
       sodacan sample014_candidate.json，读取 hand.o7_active_ctrl。
    3. --out
       输出 sodacan 专属 best_config。

输出：
    1. sodacan_sample014_best_config_from_candidate_debug.json
    2. 终端打印替换前后信息。

当前流程位置：
    P3 已通过并筛出 contact-free goal
        -> 生成新物体专属 hand_config
        -> P4U6 approach path
        -> P4U1 ready-gated snap close

不负责：
    1. 不修改 can52 legacy demo；
    2. 不修改 P4U1/P4U6 源码；
    3. 不重新运行 P2/P3；
    4. 不做手型优化，只把新物体 candidate 的先验 ctrl 接入 P4U6。
"""

from pathlib import Path
import argparse
import json
import copy


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


def resolve_path(p):
    p = Path(str(p)).expanduser()
    if not p.is_absolute():
        p = PROJECT / p
    return p


def load_json(path):
    path = resolve_path(path)
    with open(path, "r") as f:
        return json.load(f)


def save_json(path, obj):
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(obj, f, indent=2, ensure_ascii=False)


def extract_candidate_ctrl(candidate):
    hand = candidate.get("hand", {})
    ctrl = hand.get("o7_active_ctrl", None)

    if not isinstance(ctrl, dict):
        raise RuntimeError("candidate does not contain hand.o7_active_ctrl")

    missing = [j for j in O7_ACTIVE_JOINTS if j not in ctrl]
    if missing:
        raise RuntimeError(f"candidate ctrl missing joints: {missing}")

    return {j: float(ctrl[j]) for j in O7_ACTIVE_JOINTS}


def replace_ctrl_nodes(obj, ctrl, stats, path="root"):
    if isinstance(obj, dict):
        # 情况 1：标准结构 hand_config.ctrl
        if isinstance(obj.get("ctrl"), dict):
            before = {j: obj["ctrl"].get(j) for j in O7_ACTIVE_JOINTS if j in obj["ctrl"]}
            for j, v in ctrl.items():
                obj["ctrl"][j] = v
            stats.append({
                "path": path + ".ctrl",
                "type": "dict_ctrl",
                "before": before,
                "after": ctrl,
            })

        # 情况 2：某些节点自己就是 joint->value 字典
        if any(j in obj for j in O7_ACTIVE_JOINTS):
            before = {j: obj.get(j) for j in O7_ACTIVE_JOINTS if j in obj}
            for j, v in ctrl.items():
                if j in obj:
                    obj[j] = v
            if before:
                stats.append({
                    "path": path,
                    "type": "joint_value_dict",
                    "before": before,
                    "after_existing_keys": {j: obj.get(j) for j in before.keys()},
                })

        for k, v in list(obj.items()):
            replace_ctrl_nodes(v, ctrl, stats, path + "." + str(k))

    elif isinstance(obj, list):
        for i, v in enumerate(obj):
            replace_ctrl_nodes(v, ctrl, stats, path + f"[{i}]")


def ensure_best_record(data, ctrl):
    if not isinstance(data, dict):
        raise RuntimeError("base config top-level is not dict")

    if "best_record" not in data or not isinstance(data["best_record"], dict):
        data["best_record"] = {}

    br = data["best_record"]

    if "hand_config" not in br or not isinstance(br["hand_config"], dict):
        br["hand_config"] = {}

    br["hand_config"]["ctrl"] = copy.deepcopy(ctrl)
    br["hand_config"]["source"] = "sodacan165_sample014_candidate_o7_active_ctrl_debug"

    data["generalization_debug_note"] = {
        "type": "sodacan165_sample014_best_config_from_candidate",
        "reason": "P4U6 reads best_record.hand_config.ctrl. This file replaces can52 ctrl with sodacan candidate ctrl.",
        "o7_active_joints": O7_ACTIVE_JOINTS,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument(
        "--base-config",
        default="diagnostics/current_v412/v4_12p4h2_can52_contact_sequence_proxy_debug/best_contact_sequence_config_ctrlsplit_debug.json",
    )
    ap.add_argument(
        "--candidate",
        default="diagnostics/current_v412/sodacan165_topk_p2p3_batch_debug/sample014/initial_debug/candidates/sample014_candidate.json",
    )
    ap.add_argument(
        "--out",
        default="diagnostics/current_v412/sodacan165_sample014_p4u6_viewer_debug/sodacan_sample014_best_config_from_candidate_debug.json",
    )
    args = ap.parse_args()

    base_config_path = resolve_path(args.base_config)
    candidate_path = resolve_path(args.candidate)
    out_path = resolve_path(args.out)

    if not base_config_path.exists():
        raise FileNotFoundError(base_config_path)
    if not candidate_path.exists():
        raise FileNotFoundError(candidate_path)

    base = load_json(base_config_path)
    cand = load_json(candidate_path)
    ctrl = extract_candidate_ctrl(cand)

    new_config = copy.deepcopy(base)
    stats = []
    replace_ctrl_nodes(new_config, ctrl, stats)
    ensure_best_record(new_config, ctrl)

    new_config["generalization_debug_note"]["base_config"] = str(base_config_path)
    new_config["generalization_debug_note"]["candidate"] = str(candidate_path)
    new_config["generalization_debug_note"]["out"] = str(out_path)
    new_config["generalization_debug_note"]["num_replaced_nodes"] = len(stats)
    new_config["generalization_debug_note"]["replaced_nodes_preview"] = stats[:20]

    save_json(out_path, new_config)

    print("========== MAKE SODACAN BEST_CONFIG FROM CANDIDATE ==========")
    print("base_config:", base_config_path)
    print("candidate  :", candidate_path)
    print("out        :", out_path)
    print("ctrl       :", ctrl)
    print("num replaced nodes:", len(stats))
    for item in stats[:10]:
        print("replaced:", item["path"], item["type"])
    print("=============================================================")


if __name__ == "__main__":
    main()
