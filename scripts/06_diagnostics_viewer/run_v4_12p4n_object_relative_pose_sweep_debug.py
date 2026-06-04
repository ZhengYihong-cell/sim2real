#!/usr/bin/env python3
"""
文件名：
    run_v4_12p4n_object_relative_pose_sweep_debug.py

脚本类别：
    debug / diagnostic / object-relative-pose-sweep / hand-pose-root-cause

用途：
    本脚本用于 V4.12P4N 阶段。
    当前 P4M 已证明：慢速接触冻结策略仍然失败，原因更可能是手相对物体的位姿不对。
    为了快速定位“手应该相对物体往哪个方向修”，本脚本不重新做 IK，
    而是固定当前手臂 q_grasp，把物体和蓝色支撑块整体在世界 XY 平面做小范围平移。
    这在诊断意义上等价于测试 hand-local 平移误差方向。

核心思想：
    原始场景：
        手固定，物体在原始位置，一闭合就被 thumb 推走。
    扫描场景：
        物体+支撑块整体 dx/dy 平移。
        如果某个 dx/dy 下接触顺序变成：
            thumb 不再单独很早推物体；
            非拇指不再是 proximal 先顶；
            thumb + 至少两根非拇指能在小位移内冻结；
        说明当前 hand pose 需要朝相反方向修正。

输入：
    --base-model
        原始 hard_support XML。
    --candidate
        当前 candidate。
    --p3-json
        当前 P3 JSON。
    --best-config
        已修正 ctrl semantics 的 best_config。
    --object-body
        物体 body 名，例如 grasp_can。
    --out-dir
        输出目录。

输出：
    --out-dir/models/
        每个 dx/dy 对应的临时 XML。
    --out-dir/results/
        每个 dx/dy 对应的 P4M JSON 和 terminal log。
    --out-dir/summary.csv
        汇总每个 offset 的冻结顺序、停止原因、最终位移、最终接触。
    --out-dir/summary.json
        结构化汇总。
    --out-dir/run_best_viewer.sh
        用当前排序最好的 offset 打开 viewer。

当前流程位置：
    P4M 证明手位姿异常
        -> P4N object-relative sweep 找方向
        -> 再把等价方向转成 hand-local pose 修正
        -> 再重新 P2/P3/P4M/P4J

本脚本不负责：
    1. 不重新做 IK；
    2. 不把物体偏移当最终方案；
    3. 不做 lift；
    4. 不选择正式 demo；
    5. 不用评分掩盖接触问题，只用于定位 hand pose 错误方向。
"""

from pathlib import Path
import argparse
import copy
import csv
import json
import os
import subprocess
import sys
import xml.etree.ElementTree as ET


PROJECT = Path.home() / "Projects/o7_mujoco_sim"
P4M_SCRIPT = PROJECT / "scripts/05_execution_runner/run_v4_12p4m_touch_freeze_then_squeeze_debug.py"
RUN_CLEAN = PROJECT / "run_mujoco_clean.sh"


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


def parse_vec3(s, default=(0.0, 0.0, 0.0)):
    if s is None:
        return list(default)
    vals = [float(x) for x in str(s).split()]
    if len(vals) == 0:
        return list(default)
    while len(vals) < 3:
        vals.append(0.0)
    return vals[:3]


def vec3_to_str(v):
    return f"{v[0]:.9g} {v[1]:.9g} {v[2]:.9g}"


def find_body(elem, name):
    if elem.tag == "body" and elem.get("name") == name:
        return elem
    for c in list(elem):
        r = find_body(c, name)
        if r is not None:
            return r
    return None


def patch_scene_xy(base_xml, out_xml, object_body, dx, dy, support_tokens):
    tree = ET.parse(base_xml)
    root = tree.getroot()

    obj_body = find_body(root, object_body)
    if obj_body is None:
        raise RuntimeError(f"cannot find object body: {object_body}")

    p = parse_vec3(obj_body.get("pos"), (0, 0, 0))
    p[0] += dx
    p[1] += dy
    obj_body.set("pos", vec3_to_str(p))

    tokens = [x.strip().lower() for x in support_tokens.split(",") if x.strip()]
    moved_support = []

    # 支撑块可能是 world 下 geom，也可能在 body 里。
    for elem in root.iter():
        if elem.tag not in ["geom", "body"]:
            continue
        name = elem.get("name", "")
        if not name:
            continue
        if any(t in name.lower() for t in tokens):
            p = parse_vec3(elem.get("pos"), (0, 0, 0))
            p[0] += dx
            p[1] += dy
            elem.set("pos", vec3_to_str(p))
            moved_support.append((elem.tag, name))

    out_xml = Path(out_xml)
    out_xml.parent.mkdir(parents=True, exist_ok=True)
    tree.write(out_xml, encoding="utf-8", xml_declaration=True)

    return {
        "object_body": object_body,
        "dx": dx,
        "dy": dy,
        "moved_support": moved_support,
    }


def load_json(p):
    with open(resolve_path(p), "r") as f:
        return json.load(f)


def save_json(p, obj):
    p = resolve_path(p)
    p.parent.mkdir(parents=True, exist_ok=True)
    with open(p, "w") as f:
        json.dump(obj, f, indent=2)


def parse_list(s):
    return [float(x) for x in str(s).replace(",", " ").split() if x.strip()]


def event_summary(d):
    fe = d.get("freeze_events", [])
    groups = [e.get("group") for e in fe]
    first = fe[0] if fe else None
    second = fe[1] if len(fe) > 1 else None
    non_thumb = [g for g in groups if g in ["index", "middle", "ring", "pinky"]]

    contacts = []
    for e in fe:
        detail = []
        for c in e.get("contacts", []):
            detail.append({
                "geom": c.get("hand_geom"),
                "segment": c.get("segment"),
                "force": c.get("normal_force"),
            })
        contacts.append({
            "group": e.get("group"),
            "alpha": e.get("alpha"),
            "disp": e.get("object_disp"),
            "contacts": detail,
        })

    return {
        "freeze_groups": groups,
        "num_non_thumb": len(set(non_thumb)),
        "first_group": first.get("group") if first else None,
        "first_alpha": first.get("alpha") if first else None,
        "first_disp": first.get("object_disp") if first else None,
        "second_group": second.get("group") if second else None,
        "second_alpha": second.get("alpha") if second else None,
        "second_disp": second.get("object_disp") if second else None,
        "contacts": contacts,
        "ready": d.get("frozen_ready_step") is not None,
        "stop_reason": d.get("stop_reason"),
        "final_disp": d.get("final_object_disp"),
        "final_groups": d.get("final_contact", {}).get("groups", {}),
    }


def score_summary(s):
    score = 0.0

    if s["ready"]:
        score -= 100.0
    else:
        score += 50.0

    if s["first_group"] == "thumb":
        score += 20.0
    elif s["first_group"] is None:
        score += 30.0
    else:
        score -= 10.0

    if s["num_non_thumb"] >= 2:
        score -= 40.0
    else:
        score += 30.0

    if s["second_disp"] is not None:
        score += 100.0 * float(s["second_disp"])
    if s["final_disp"] is not None:
        score += 30.0 * float(s["final_disp"])

    # 近端接触惩罚：只看 geom 名，避免 segment 分类被 middle/proximal 名称干扰。
    for item in s["contacts"]:
        for c in item["contacts"]:
            geom = str(c.get("geom", "")).lower()
            if "proximal" in geom:
                score += 15.0
            if "distal" in geom:
                score -= 5.0

    return score


def run_one(args, model_xml, out_json, terminal_log):
    cmd = [
        str(RUN_CLEAN),
        rel(P4M_SCRIPT),
        "--model", rel(model_xml),
        "--candidate", args.candidate,
        "--p3-json", args.p3_json,
        "--best-config", args.best_config,
        "--which", args.which,
        "--object-body", args.object_body,
        "--out", rel(out_json),
        "--move-steps", str(args.move_steps),
        "--thumb-preshape-steps", str(args.thumb_preshape_steps),
        "--probe-duration", str(args.probe_duration),
        "--squeeze-duration", str(args.squeeze_duration),
        "--hold-duration", str(args.hold_duration),
        "--finger-close-scale", str(args.finger_close_scale),
        "--thumb-pitch-from-finger-gain", str(args.thumb_pitch_from_finger_gain),
        "--min-freeze-non-thumb", str(args.min_freeze_non_thumb),
        "--squeeze-fraction", str(args.squeeze_fraction),
        "--max-probe-disp", str(args.max_probe_disp),
        "--max-squeeze-disp", str(args.max_squeeze_disp),
        "--log-dt", str(args.log_dt),
        "--frame-sleep", str(args.frame_sleep),
    ]

    proc = subprocess.run(
        cmd,
        cwd=str(PROJECT),
        stdout=subprocess.PIPE,
        stderr=subprocess.STDOUT,
        text=True,
    )
    Path(terminal_log).write_text(proc.stdout)

    if proc.returncode != 0:
        raise RuntimeError(f"P4M failed: {terminal_log}")

    return cmd


def write_best_viewer(args, best_record, out_dir):
    script = resolve_path(out_dir) / "run_best_viewer.sh"
    model_xml = best_record["model_xml"]
    out_json = resolve_path(out_dir) / "best_viewer.json"

    cmd = [
        str(RUN_CLEAN),
        rel(P4M_SCRIPT),
        "--model", rel(model_xml),
        "--candidate", args.candidate,
        "--p3-json", args.p3_json,
        "--best-config", args.best_config,
        "--which", args.which,
        "--object-body", args.object_body,
        "--out", rel(out_json),
        "--viewer",
        "--keep-viewer-open",
        "--move-steps", str(args.move_steps),
        "--thumb-preshape-steps", str(args.thumb_preshape_steps),
        "--probe-duration", str(args.probe_duration),
        "--squeeze-duration", str(args.squeeze_duration),
        "--hold-duration", str(args.hold_duration),
        "--finger-close-scale", str(args.finger_close_scale),
        "--thumb-pitch-from-finger-gain", str(args.thumb_pitch_from_finger_gain),
        "--min-freeze-non-thumb", str(args.min_freeze_non_thumb),
        "--squeeze-fraction", str(args.squeeze_fraction),
        "--max-probe-disp", str(args.max_probe_disp),
        "--max-squeeze-disp", str(args.max_squeeze_disp),
        "--log-dt", str(args.log_dt),
        "--frame-sleep", str(args.frame_sleep),
    ]

    with open(script, "w") as f:
        f.write("#!/usr/bin/env bash\n")
        f.write("set -e\n")
        f.write("cd ~/Projects/o7_mujoco_sim\n")
        f.write("source ~/mujoco_env/bin/activate\n")
        f.write(" ".join(cmd) + f" 2>&1 | tee {rel(resolve_path(out_dir) / 'best_viewer_terminal.txt')}\n")

    os.chmod(script, 0o755)
    return script


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--base-model", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--p3-json", required=True)
    ap.add_argument("--best-config", required=True)
    ap.add_argument("--which", default="best_available")
    ap.add_argument("--object-body", default="grasp_can")
    ap.add_argument("--support-tokens", default="object_pedestal,pedestal,support,table")
    ap.add_argument("--out-dir", required=True)

    ap.add_argument("--dx-list", default="-0.015 -0.010 -0.005 0.000 0.005 0.010 0.015")
    ap.add_argument("--dy-list", default="-0.020 -0.015 -0.010 -0.005 0.000 0.005 0.010 0.015 0.020")

    ap.add_argument("--move-steps", type=int, default=80)
    ap.add_argument("--thumb-preshape-steps", type=int, default=80)
    ap.add_argument("--probe-duration", type=float, default=1.2)
    ap.add_argument("--squeeze-duration", type=float, default=0.5)
    ap.add_argument("--hold-duration", type=float, default=0.4)

    ap.add_argument("--finger-close-scale", type=float, default=0.85)
    ap.add_argument("--thumb-pitch-from-finger-gain", type=float, default=0.35)
    ap.add_argument("--min-freeze-non-thumb", type=int, default=2)
    ap.add_argument("--squeeze-fraction", type=float, default=0.25)
    ap.add_argument("--max-probe-disp", type=float, default=0.020)
    ap.add_argument("--max-squeeze-disp", type=float, default=0.035)
    ap.add_argument("--log-dt", type=float, default=0.08)
    ap.add_argument("--frame-sleep", type=float, default=0.001)

    args = ap.parse_args()

    base_model = resolve_path(args.base_model)
    out_dir = resolve_path(args.out_dir)
    model_dir = out_dir / "models"
    result_dir = out_dir / "results"
    model_dir.mkdir(parents=True, exist_ok=True)
    result_dir.mkdir(parents=True, exist_ok=True)

    dxs = parse_list(args.dx_list)
    dys = parse_list(args.dy_list)

    records = []

    print("========== V4.12P4N OBJECT-RELATIVE POSE SWEEP ==========")
    print("base_model:", base_model)
    print("out_dir   :", out_dir)
    print("dxs       :", dxs)
    print("dys       :", dys)
    print("=========================================================")

    for dx in dxs:
        for dy in dys:
            tag = f"dx{dx:+.3f}_dy{dy:+.3f}".replace("+", "p").replace("-", "m").replace(".", "p")
            model_xml = model_dir / f"scene_{tag}.xml"
            out_json = result_dir / f"p4m_{tag}.json"
            term_log = result_dir / f"p4m_{tag}_terminal.txt"

            patch_info = patch_scene_xy(
                base_xml=base_model,
                out_xml=model_xml,
                object_body=args.object_body,
                dx=dx,
                dy=dy,
                support_tokens=args.support_tokens,
            )

            run_one(args, model_xml, out_json, term_log)

            d = load_json(out_json)
            summ = event_summary(d)
            score = score_summary(summ)

            rec = {
                "dx": dx,
                "dy": dy,
                "score": score,
                "model_xml": str(model_xml),
                "out_json": str(out_json),
                "terminal_log": str(term_log),
                "patch_info": patch_info,
                "summary": summ,
            }
            records.append(rec)

            print(
                f"[dx={dx:+.3f}, dy={dy:+.3f}] "
                f"score={score:.3f} ready={summ['ready']} "
                f"first={summ['first_group']}@{summ['first_alpha']} "
                f"second={summ['second_group']}@{summ['second_alpha']} "
                f"second_disp={summ['second_disp']} "
                f"final_disp={summ['final_disp']} "
                f"groups={summ['freeze_groups']} "
                f"stop={summ['stop_reason']}"
            )

    ranked = sorted(records, key=lambda r: r["score"])
    summary_json = out_dir / "summary.json"
    summary_csv = out_dir / "summary.csv"

    save_json(summary_json, {
        "format": "v4_12p4n_object_relative_pose_sweep_debug",
        "args": vars(args),
        "ranked": ranked,
    })

    with open(summary_csv, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow([
            "rank", "score", "dx", "dy", "ready",
            "freeze_groups", "first_group", "first_alpha", "first_disp",
            "second_group", "second_alpha", "second_disp",
            "num_non_thumb", "final_disp", "final_groups", "stop_reason",
            "model_xml", "out_json",
        ])
        for i, r in enumerate(ranked, 1):
            s = r["summary"]
            w.writerow([
                i, r["score"], r["dx"], r["dy"], s["ready"],
                s["freeze_groups"], s["first_group"], s["first_alpha"], s["first_disp"],
                s["second_group"], s["second_alpha"], s["second_disp"],
                s["num_non_thumb"], s["final_disp"], s["final_groups"], s["stop_reason"],
                r["model_xml"], r["out_json"],
            ])

    best_script = write_best_viewer(args, ranked[0], out_dir)

    print("\n========== P4N SUMMARY ==========")
    print("summary_json:", summary_json)
    print("summary_csv :", summary_csv)
    print("best_script :", best_script)
    print("\nTop 10:")
    for i, r in enumerate(ranked[:10], 1):
        s = r["summary"]
        print(
            f"{i:02d}. score={r['score']:.3f} dx={r['dx']:+.3f} dy={r['dy']:+.3f} "
            f"ready={s['ready']} groups={s['freeze_groups']} "
            f"first={s['first_group']}@{s['first_alpha']} "
            f"second={s['second_group']}@{s['second_alpha']} second_disp={s['second_disp']} "
            f"final_disp={s['final_disp']} stop={s['stop_reason']}"
        )
    print("\nBest viewer:")
    print(f"bash {best_script}")
    print("=================================\n")


if __name__ == "__main__":
    main()
