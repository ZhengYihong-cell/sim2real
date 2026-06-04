#!/usr/bin/env python3
from pathlib import Path
import shutil
from datetime import datetime

PROJECT = Path.home() / "Projects/o7_mujoco_sim"
ARCHIVE = PROJECT / f"_archive_debug_versions_{datetime.now().strftime('%Y%m%d_%H%M%S')}"

KEEP_MODELS = {
    "o7_grasp_scene_v8_13_solver_extreme.xml",
    "o7_grasp_scene_v8_13_solver_extreme_dt0005.xml",
    "o7_grasp_scene_v8_10_highfric_condim6.xml",
    "o7_grasp_scene_v8_7_palm_strip_proxy.xml",
    "o7_right_source.xml",
}

KEEP_SCRIPTS = {
    "run_mujoco_clean.sh",
    "run_o7_saved_snapshot_v8_13.py",
    "run_o7_joint_tuner_v8_13_solver_extreme.py",
    "run_o7_joint_tuner_v8_10_highfric_condim6.py",
    "make_o7_grasp_scene_v8_13_solver_extreme.py",
    "make_o7_grasp_scene_v8_10_highfric_condim6.py",
    "make_o7_grasp_scene_v8_7_palm_strip_proxy.py",
    "diagnose_v8_9_free_vs_locked_hand.py",
    "diagnose_v8_11_box_true_force.py",
    "diagnose_v8_12b_slip_velocity_response_fixed.py",
}

KEEP_RECORDS_EXACT = {
    "stable_air_grasp_v8_13_box6cm.json",
    "o7_full_snapshot_v8_7_20260512_172018_324629.json",
}

def move_to_archive(path: Path):
    rel = path.relative_to(PROJECT)
    dst = ARCHIVE / rel
    dst.parent.mkdir(parents=True, exist_ok=True)
    print(f"[MOVE] {rel}  ->  {dst.relative_to(PROJECT)}")
    shutil.move(str(path), str(dst))

def main():
    ARCHIVE.mkdir(parents=True, exist_ok=True)

    # 1. models：只保留最终和关键上游版本
    models_dir = PROJECT / "models"
    if models_dir.exists():
        for p in models_dir.iterdir():
            if p.is_file() and p.suffix == ".xml":
                if p.name not in KEEP_MODELS:
                    move_to_archive(p)

    # 2. scripts：只保留最终运行、最终生成、关键诊断脚本
    scripts_dir = PROJECT / "scripts"
    if scripts_dir.exists():
        for p in scripts_dir.iterdir():
            if not p.is_file():
                continue
            if p.name == "__pycache__":
                continue

            # 只整理 py 和 bak，其他不动
            if p.suffix not in {".py", ".bak"}:
                continue

            if p.name not in KEEP_SCRIPTS:
                move_to_archive(p)

    # 3. records：保留稳定姿态、源空中姿态、最新 v8_13_loaded
    records_dir = PROJECT / "records"
    if records_dir.exists():
        latest_loaded = None
        loaded_files = sorted(
            records_dir.glob("o7_full_snapshot_v8_13_loaded_*.json"),
            key=lambda x: x.stat().st_mtime,
            reverse=True,
        )
        if loaded_files:
            latest_loaded = loaded_files[0].name

        for p in records_dir.iterdir():
            if not p.is_file():
                continue

            if p.name in KEEP_RECORDS_EXACT:
                continue

            if latest_loaded and p.name == latest_loaded:
                continue

            # 保留非 json/txt 的未知文件
            if p.suffix not in {".json", ".txt"}:
                continue

            move_to_archive(p)

    # 4. diagnostics：保留 V8.13 相关，其余归档
    diagnostics_dir = PROJECT / "diagnostics"
    if diagnostics_dir.exists():
        for p in diagnostics_dir.iterdir():
            if not p.is_file():
                continue

            if "v8_13" in p.name:
                continue

            move_to_archive(p)

    print()
    print("[DONE]")
    print("Archive directory:")
    print(ARCHIVE)
    print()
    print("你现在只是把旧文件移动到了归档目录，没有彻底删除。确认项目正常后再考虑 rm -rf 这个归档目录。")

if __name__ == "__main__":
    main()
