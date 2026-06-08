#!/usr/bin/env python3
"""
脚本类型：
    debug / repair / best-config-ctrl

用途：
    修复 BB026 top soft rerank 生成的 best_config_rank01/02/03。
    当前问题是 best_record.hand_config.ctrl 仍保留 base config 旧 ctrl，
    而 candidate_ctrl / side_open_ctrl 才是当前 rank 对应 candidate 的手型。
    P4U6/P4U1 runner 会读取 best_record.hand_config.ctrl，因此必须修正。

输入：
    diagnostics/current_v412/sodacan_bb026_top_soft_rerank_debug/best_config_rank01.json
    diagnostics/current_v412/sodacan_bb026_top_soft_rerank_debug/best_config_rank02.json
    diagnostics/current_v412/sodacan_bb026_top_soft_rerank_debug/best_config_rank03.json

输出：
    原地覆盖上述 3 个 json，并生成 .bak 备份。

不负责：
    不修改 candidate，不修改 P3，不修改 P4U1/P4U6，不修改 legacy demo。
"""

from pathlib import Path
import json
import shutil

PROJECT = Path.home() / "Projects/o7_mujoco_sim"
ROOT = PROJECT / "diagnostics/current_v412/sodacan_bb026_top_soft_rerank_debug"

JOINTS = [
    "thumb_cmc_roll",
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]

def main():
    for i in [1, 2, 3]:
        p = ROOT / f"best_config_rank{i:02d}.json"
        if not p.exists():
            print("[WARN] missing:", p)
            continue

        bak = p.with_suffix(".json.bak")
        if not bak.exists():
            shutil.copy2(p, bak)

        d = json.loads(p.read_text())

        ctrl = d.get("candidate_ctrl") or d.get("side_open_ctrl")
        if not isinstance(ctrl, dict):
            raise RuntimeError(f"{p} missing candidate_ctrl / side_open_ctrl")

        ctrl = {j: float(ctrl[j]) for j in JOINTS}

        hc = d.setdefault("best_record", {}).setdefault("hand_config", {})
        before = hc.get("ctrl", {})
        hc["ctrl"] = ctrl
        hc["ctrl_semantics"] = "candidate_ctrl_as_runner_side_open_input_repaired"
        hc["source"] = "bb026_top_soft_rerank_candidate_ctrl_repaired"

        d["bb026_soft_rerank_ctrl_repair"] = {
            "reason": "runner reads best_record.hand_config.ctrl; replace stale base ctrl with candidate_ctrl",
            "backup": str(bak),
            "before_ctrl": before,
            "after_ctrl": ctrl,
        }

        p.write_text(json.dumps(d, indent=2, ensure_ascii=False))

        print(f"[OK] repaired rank{i:02d}: {p}")
        print("  before:", before)
        print("  after :", ctrl)

if __name__ == "__main__":
    main()
