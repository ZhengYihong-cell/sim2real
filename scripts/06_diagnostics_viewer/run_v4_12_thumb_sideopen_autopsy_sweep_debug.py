#!/usr/bin/env python3
"""
文件名：
    run_v4_12_thumb_sideopen_autopsy_sweep_debug.py

脚本类别：
    debug / diagnostic / thumb-sideopen-sweep / contact-mechanics-autopsy

用途：
    本脚本用于 V4.12 阶段继续定位抓握异常根因。
    当前已经确认 side_open / close_target 双重叠加问题被修正，
    但大拇指仍然在 close 很早期单侧接触物体，导致物体倾倒或滑走。
    本脚本通过扫描 thumb_cmc_pitch 的 side_open 值，判断问题到底来自：
        A. 大拇指 side_open 仍然太闭；
        B. handbase 位姿本身太靠近 thumb 侧；
        C. 四指闭合轨迹主要由 middle/ring 近端 link 推物体，而不是 distal 指腹夹持。

输入：
    --base-config
        已修正过 ctrl semantics 的 best_config。
    --model
        MuJoCo XML，建议 hard_support 版本。
    --candidate
        当前 candidate。
    --p3-json
        当前 P3 JSON。
    --thumb-list
        要扫描的 thumb_cmc_pitch side_open 值。

输出：
    --out-dir/summary.txt
        每个 thumb side_open 的第一接触、第一次非拇指接触、第一次大位移事件。
    --out-dir/summary.json
        结构化结果。
    --out-dir/configs/
        每个 thumb side_open 对应的临时 best_config。
    --out-dir/autopsy/
        每次尸检的 json/txt/terminal log。

当前流程位置：
    ctrl semantics 修正
        -> thumb side_open sweep autopsy
        -> 判断是否需要打开大拇指，还是重新修 hand local pose

本脚本不负责：
    1. 不选择 best；
    2. 不跑 lift；
    3. 不判断抓握成功；
    4. 不修改原始 candidate；
    5. 不继续用评分掩盖接触力学问题。
"""

from pathlib import Path
import argparse
import copy
import json
import subprocess
import sys


PROJECT = Path.home() / "Projects/o7_mujoco_sim"
AUTOPSY = PROJECT / "scripts/06_diagnostics_viewer/diagnose_v4_12_contact_mechanics_autopsy_debug.py"

HAND_JOINTS = [
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


def rel(p):
    p = resolve_path(p)
    try:
        return str(p.relative_to(PROJECT))
    except Exception:
        return str(p)


def load_json(p):
    with open(resolve_path(p), "r") as f:
        return json.load(f)


def save_json(p, obj):
    p = resolve_path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)


def parse_float_list(s):
    return [float(x) for x in str(s).replace(",", " ").split() if x.strip()]


def get_nested(d, path, default=None):
    cur = d
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return default
        cur = cur[k]
    return cur


def event_brief(row):
    if row is None:
        return {
            "exists": False,
            "phase": None,
            "step": None,
            "alpha": None,
            "disp": None,
            "groups": {},
            "segments": {},
            "first_contact_detail": [],
        }

    contacts = []
    for c in row.get("contact", {}).get("hand_object", []):
        contacts.append({
            "group": c.get("hand_group"),
            "segment": c.get("hand_segment"),
            "geom": c.get("hand_geom"),
            "dist": c.get("dist"),
            "normal_force": c.get("normal_force"),
            "rel": c.get("rel_pos_from_object_center"),
        })

    return {
        "exists": True,
        "phase": row.get("phase"),
        "step": row.get("step"),
        "alpha": row.get("alpha"),
        "disp": row.get("object_disp"),
        "delta": row.get("object_delta"),
        "groups": row.get("contact", {}).get("groups", {}),
        "segments": row.get("contact", {}).get("segments", {}),
        "support": row.get("contact", {}).get("min_object_support_dist"),
        "first_contact_detail": contacts,
    }


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--p3-json", required=True)
    ap.add_argument("--base-config", required=True)
    ap.add_argument("--which", default="best_available")
    ap.add_argument("--object-body", default="grasp_can")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--thumb-list", default="0.00 0.04 0.08 0.12 0.16 0.20 0.22")
    ap.add_argument("--close-duration", type=float, default=1.0)
    ap.add_argument("--move-steps", type=int, default=80)
    ap.add_argument("--thumb-preshape-steps", type=int, default=80)
    ap.add_argument("--finger-close-scale", type=float, default=0.85)
    ap.add_argument("--keep-final-thumb-close", action="store_true")
    ap.add_argument("--desired-final-thumb", type=float, default=-1.0)
    ap.add_argument("--log-dt", type=float, default=0.04)
    args = ap.parse_args()

    out_dir = resolve_path(args.out_dir)
    config_dir = out_dir / "configs"
    autopsy_dir = out_dir / "autopsy"
    config_dir.mkdir(parents=True, exist_ok=True)
    autopsy_dir.mkdir(parents=True, exist_ok=True)

    base_cfg = load_json(args.base_config)
    candidate_ctrl = base_cfg.get("candidate_ctrl", {})
    if not candidate_ctrl:
        raise RuntimeError("base_config missing candidate_ctrl")

    max_four = max(float(candidate_ctrl[j]) for j in [
        "index_mcp_pitch",
        "middle_mcp_pitch",
        "ring_mcp_pitch",
        "pinky_mcp_pitch",
    ])

    base_close = base_cfg.get("close_target", {})
    base_side = base_cfg.get("side_open_ctrl", {})
    default_desired_final = float(base_close.get("thumb_cmc_pitch", 0.4251184210181236))
    desired_final = args.desired_final_thumb if args.desired_final_thumb >= 0 else default_desired_final

    thumbs = parse_float_list(args.thumb_list)
    results = []

    print("========== THUMB SIDE_OPEN AUTOPSY SWEEP ==========")
    print("model       :", args.model)
    print("candidate   :", args.candidate)
    print("p3_json     :", args.p3_json)
    print("base_config :", args.base_config)
    print("out_dir     :", out_dir)
    print("thumbs      :", thumbs)
    print("max_four    :", max_four)
    print("desired_final_thumb:", desired_final)
    print("keep_final_thumb_close:", args.keep_final_thumb_close)
    print("==================================================")

    for th in thumbs:
        cfg = copy.deepcopy(base_cfg)
        ctrl = cfg["best_record"]["hand_config"]["ctrl"]
        ctrl["thumb_cmc_pitch"] = float(th)

        cfg["best_record"]["hand_config"]["ctrl_semantics"] = "side_open_input_for_runner_thumb_sweep"
        cfg["thumb_sideopen_sweep_info"] = {
            "thumb_side_open": float(th),
            "base_side_open_thumb": base_side.get("thumb_cmc_pitch"),
            "base_close_thumb": base_close.get("thumb_cmc_pitch"),
            "keep_final_thumb_close": bool(args.keep_final_thumb_close),
            "desired_final_thumb": desired_final,
        }

        cfg_path = config_dir / f"best_config_thumb_sideopen_{th:.3f}.json"
        save_json(cfg_path, cfg)

        if args.keep_final_thumb_close:
            gain = max(0.0, (desired_final - float(th)) / max_four)
        else:
            gain = 0.35

        prefix = autopsy_dir / f"autopsy_thumb_sideopen_{th:.3f}"

        cmd = [
            sys.executable,
            str(AUTOPSY),
            "--model", args.model,
            "--candidate", args.candidate,
            "--p3-json", args.p3_json,
            "--best-config", rel(cfg_path),
            "--which", args.which,
            "--object-body", args.object_body,
            "--out-prefix", rel(prefix),
            "--move-steps", str(args.move_steps),
            "--thumb-preshape-steps", str(args.thumb_preshape_steps),
            "--close-duration", str(args.close_duration),
            "--finger-close-scale", str(args.finger_close_scale),
            "--thumb-pitch-from-finger-gain", str(gain),
            "--log-dt", str(args.log_dt),
        ]

        print("\n[RUN]", " ".join(cmd))
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT),
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
        )

        term_path = Path(str(prefix) + "_terminal.txt")
        term_path.write_text(proc.stdout)

        if proc.returncode != 0:
            print(proc.stdout)
            raise RuntimeError(f"autopsy failed for thumb={th}, see {term_path}")

        out_json = Path(str(prefix) + ".json")
        d = load_json(out_json)

        item = {
            "thumb_side_open": float(th),
            "thumb_pitch_gain": float(gain),
            "expected_close_thumb": float(th) + gain * max_four,
            "config": str(cfg_path),
            "autopsy_json": str(out_json),
            "autopsy_txt": str(prefix) + ".txt",
            "terminal": str(term_path),
            "first_hand_contact": event_brief(d.get("first_hand_contact")),
            "first_thumb": event_brief(d.get("first_thumb")),
            "first_non_thumb": event_brief(d.get("first_non_thumb")),
            "first_large_motion_gt_1cm": event_brief(d.get("first_large_motion")),
            "first_proximal_or_palm": event_brief(d.get("first_proximal_or_palm")),
            "ctrl_warnings": d.get("ctrl_warnings", []),
        }
        results.append(item)

        fh = item["first_hand_contact"]
        fn = item["first_non_thumb"]
        fl = item["first_large_motion_gt_1cm"]

        print(
            f"[SUMMARY thumb={th:.3f}] "
            f"gain={gain:.3f} close≈{item['expected_close_thumb']:.3f} | "
            f"first_hand={fh['groups']} alpha={fh['alpha']} disp={fh['disp']} | "
            f"first_non_thumb={fn['groups']} alpha={fn['alpha']} disp={fn['disp']} | "
            f"large_motion={fl['groups']} alpha={fl['alpha']} disp={fl['disp']}"
        )

    summary = {
        "format": "v4_12_thumb_sideopen_autopsy_sweep_debug",
        "args": vars(args),
        "max_four": max_four,
        "desired_final_thumb": desired_final,
        "results": results,
    }

    save_json(out_dir / "summary.json", summary)

    txt_path = out_dir / "summary.txt"
    with open(txt_path, "w") as f:
        f.write("thumb_side_open,gain,expected_close,first_hand_groups,first_hand_alpha,first_hand_disp,first_non_thumb_groups,first_non_thumb_alpha,first_non_thumb_disp,large_motion_groups,large_motion_alpha,large_motion_disp,notes\n")
        for r in results:
            fh = r["first_hand_contact"]
            fn = r["first_non_thumb"]
            fl = r["first_large_motion_gt_1cm"]

            notes = []
            if fh["groups"] == {"thumb": 1}:
                notes.append("thumb_first_single")
            if fl["exists"] and (not fl["groups"]):
                notes.append("large_motion_after_transient_push_no_current_contact")
            if fn["exists"]:
                segs = fn["segments"]
                if any(("proximal" in k or ":middle" in k) for k in segs.keys()):
                    notes.append("non_thumb_not_distal")

            f.write(
                f"{r['thumb_side_open']:.4f},"
                f"{r['thumb_pitch_gain']:.4f},"
                f"{r['expected_close_thumb']:.4f},"
                f"{fh['groups']},"
                f"{fh['alpha']},"
                f"{fh['disp']},"
                f"{fn['groups']},"
                f"{fn['alpha']},"
                f"{fn['disp']},"
                f"{fl['groups']},"
                f"{fl['alpha']},"
                f"{fl['disp']},"
                f"{'|'.join(notes)}\n"
            )

    print("\n========== SWEEP SAVED ==========")
    print("summary json:", out_dir / "summary.json")
    print("summary txt :", txt_path)
    print("===============================\n")


if __name__ == "__main__":
    main()
