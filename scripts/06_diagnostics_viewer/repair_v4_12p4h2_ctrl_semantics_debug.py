#!/usr/bin/env python3
"""
文件名：
    repair_v4_12p4h2_ctrl_semantics_debug.py

脚本类别：
    debug / repair / config-semantics-fix

用途：
    本脚本用于 V4.12 阶段，修复 P4H2 best_config 中 hand_config.ctrl 的语义错误。
    当前问题是 best_config.best_record.hand_config.ctrl 已经像是 close_target，
    但 P4F/P4J runner 会把它当成 side_open / preshape 输入，然后再次生成 close_target。
    这会导致 thumb_cmc_pitch 从 0.425 再被推到 0.630，使大拇指提前变成推板。

输入：
    --in-config
        原始 P4H2 best_contact_sequence_config.json。
    --out-config
        修正后的 best_config。
    --thumb-joint
        默认 thumb_cmc_pitch。

输出：
    修正后的 best_config。它会：
        1. 保留原始 final close target；
        2. 把 best_record.hand_config.ctrl 中的 thumb_cmc_pitch 回退到 side_open 输入值；
        3. 记录 repair_info，便于后续追溯；
        4. 不改 candidate，不改 P3，不改模型。

当前流程位置：
    P4H2 best_config
        -> 本脚本修正 ctrl 语义
        -> contact autopsy 重新检查
        -> 再决定是否继续 P4J

本脚本不负责：
    1. 不跑仿真；
    2. 不重新选 best；
    3. 不修改手腕位姿；
    4. 不判断抓握成功；
    5. 不做 lift。
"""

from pathlib import Path
import argparse
import copy
import json


HAND_JOINTS = [
    "thumb_cmc_roll",
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]


def load_json(p):
    with open(Path(p).expanduser(), "r") as f:
        return json.load(f)


def save_json(p, obj):
    p = Path(p).expanduser()
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)


def get_ctrl(cfg):
    try:
        return cfg["best_record"]["hand_config"]["ctrl"]
    except Exception as e:
        raise RuntimeError("cannot find best_record.hand_config.ctrl") from e


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--in-config", required=True)
    ap.add_argument("--out-config", required=True)
    ap.add_argument("--thumb-joint", default="thumb_cmc_pitch")
    ap.add_argument("--thumb-min", type=float, default=0.0)
    ap.add_argument("--thumb-max", type=float, default=0.5146)
    args = ap.parse_args()

    cfg = load_json(args.in_config)
    out = copy.deepcopy(cfg)

    ctrl = get_ctrl(out)
    old_ctrl = copy.deepcopy(ctrl)

    top_side = cfg.get("side_open_ctrl", {})
    top_close = cfg.get("close_target", {})

    tj = args.thumb_joint
    if tj not in ctrl:
        raise RuntimeError(f"{tj} not in best_record.hand_config.ctrl")
    if tj not in top_side or tj not in top_close:
        raise RuntimeError(f"{tj} not in top-level side_open_ctrl / close_target")

    old_best_thumb = float(ctrl[tj])
    old_side_thumb = float(top_side[tj])
    old_close_thumb = float(top_close[tj])
    runner_added_delta = old_close_thumb - old_side_thumb

    fixed_side_thumb = old_best_thumb - runner_added_delta
    fixed_side_thumb = max(args.thumb_min, min(args.thumb_max, fixed_side_thumb))

    ctrl[tj] = fixed_side_thumb

    hand_config = out["best_record"]["hand_config"]
    hand_config["ctrl_semantics"] = "side_open_input_for_runner"
    hand_config["original_ctrl_before_semantics_repair"] = old_ctrl
    hand_config["intended_close_target_ctrl_before_semantics_repair"] = top_close
    hand_config["repaired_side_open_input_ctrl"] = copy.deepcopy(ctrl)

    out["repair_info"] = {
        "format": "v4_12p4h2_ctrl_semantics_repair_debug",
        "reason": "best_config.hand_config.ctrl was being used as side_open input although it already behaved like close_target; thumb pitch was added twice by runner.",
        "thumb_joint": tj,
        "old_best_thumb": old_best_thumb,
        "old_top_side_thumb": old_side_thumb,
        "old_top_close_thumb": old_close_thumb,
        "runner_added_delta": runner_added_delta,
        "fixed_side_thumb": fixed_side_thumb,
        "expected_runner_close_thumb_after_fix": fixed_side_thumb + runner_added_delta,
        "note": "After this fix, autopsy should show side_open thumb around fixed_side_thumb and close_target thumb around old_best_thumb, not old_top_close_thumb.",
    }

    save_json(args.out_config, out)

    print("========== V4.12P4H2 CTRL SEMANTICS REPAIR ==========")
    print("in_config :", args.in_config)
    print("out_config:", args.out_config)
    print("thumb_joint:", tj)
    print("old best_config thumb:", old_best_thumb)
    print("old top side thumb   :", old_side_thumb)
    print("old top close thumb  :", old_close_thumb)
    print("runner added delta   :", runner_added_delta)
    print("fixed side thumb     :", fixed_side_thumb)
    print("expected close thumb :", fixed_side_thumb + runner_added_delta)
    print("====================================================")


if __name__ == "__main__":
    main()
