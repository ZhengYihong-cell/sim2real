#!/usr/bin/env python3
"""Aggressive V4.12 cleanup planner/applier.

Default dry-run prints the cleanup plan and does not mutate the tree.
Use --dry-run --emit-docs to refresh docs/CLEANUP_PLAN... and manifest.
Use --apply to execute, after typing YES.
"""

from __future__ import annotations

import argparse
import json
import os
import shutil
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Iterable


ROOT = Path(__file__).resolve().parents[2]
MANIFEST_PATH = ROOT / "docs/CLEANUP_MANIFEST_V412_AGGRESSIVE.json"
PLAN_DOC_PATH = ROOT / "docs/CLEANUP_PLAN_V412_AGGRESSIVE.md"

PROTECTED_TOP_LEVEL = {
    "assets",
    "dataset",
    ".git",
    ".codex",
    ".agents",
}

FINAL_DEMOS = [
    "final_v11_demo",
    "final_v15_cylinder_demo",
    "final_v16_candidate_interface",
]

MANDATORY_ROOT_FILES = {
    "run_mujoco_clean.sh",
}

MANDATORY_SCRIPT_PATHS = {
    "scripts/00_maintenance/aggressive_cleanup_for_v412.py",
    "scripts/00_maintenance/organize_project_for_v412_debug.sh",
    "scripts/05_execution_runner/run_fr3_o7_candidate_grasp_site_servo_debug.py",
    "scripts/05_execution_runner/run_fr3_o7_object_ik_grasp_v12.py",
    "scripts/run_fr3_o7_candidate_grasp_site_servo_debug.py",
    "scripts/run_fr3_o7_candidate_grasp_site_debug.py",
    "scripts/run_fr3_o7_object_ik_grasp_v12.py",
    "scripts/06_diagnostics_viewer/diagnose_v4_9_frame_ik_contact_debug.py",
    "scripts/07_precheck_ik_collision/README.md",
    "scripts/README.md",
}

MANDATORY_MODEL_XML = {
    "fr3_o7_actuated_scene_v1f_stable_hand.xml",
    "fr3_o7_bottle_scene_handbase_upright_tabletop_scale006_debug.xml",
    "fr3_o7_can52_upright_tabletop_v47b_debug.xml",
}

SMOKE_CANDIDATE_HINTS = {
    "bottle": [
        "records/candidates/bottle_dataset_site_sample7_debug.json",
        "records/candidates/bottle_dataset_candidate_debug.json",
        "records/candidates/from_dataset_debug_candidate.json",
    ],
    "can52": [
        "records/candidates/v4_7b_can52_tabletop_safe_debug/v47b_can_core-can-52e295024593705fb00c487926b62c9_s030_tabletop_side_body_debug.json",
        "records/candidates/v4_7_can52_debug/v47_can_core-can-52e295024593705fb00c487926b62c9_s001_side_body_debug.json",
    ],
    "cylinder": [
        "records/candidates/cylinder_candidate_v1.json",
        "records/stable_fr3_o7_cylinder_grasp_candidate_v1.json",
    ],
}


@dataclass
class Operation:
    action: str
    path: Path
    target: Path | None
    reason: str


class CleanupPlan:
    def __init__(self) -> None:
        self.entries: dict[tuple[str, str, str], dict] = {}
        self.ops: list[Operation] = []
        self.warnings: list[str] = []

    def rel(self, path: Path) -> str:
        try:
            return path.relative_to(ROOT).as_posix()
        except ValueError:
            return path.as_posix()

    def size_bytes(self, path: Path) -> int:
        try:
            if path.is_symlink():
                return path.lstat().st_size
            if path.is_file():
                return path.stat().st_size
        except OSError:
            return 0
        return 0

    def add_entry(
        self,
        path: Path,
        action: str,
        reason: str,
        target: Path | None = None,
        force: bool = False,
    ) -> None:
        rel_path = self.rel(path)
        rel_target = self.rel(target) if target else ""
        key = (rel_path, action, rel_target)
        if key in self.entries and not force:
            return
        self.entries[key] = {
            "path": rel_path,
            "action": action,
            "target": rel_target,
            "reason": reason,
            "size_bytes": self.size_bytes(path),
        }

    def add_op(
        self,
        action: str,
        path: Path,
        reason: str,
        target: Path | None = None,
    ) -> None:
        self.ops.append(Operation(action=action, path=path, target=target, reason=reason))

    def keep(self, path: Path, reason: str) -> None:
        self.add_entry(path, "keep", reason)

    def mkdir(self, path: Path, reason: str) -> None:
        self.add_entry(path, "keep", reason)
        self.add_op("mkdir", path, reason)

    def delete_path(self, path: Path, reason: str, skip: set[Path] | None = None) -> None:
        if not path.exists() and not path.is_symlink():
            return
        skip_resolved = {p.resolve() for p in (skip or set()) if p.exists() or p.is_symlink()}
        if path.is_file() or path.is_symlink():
            if path.resolve() in skip_resolved:
                return
            self.add_entry(path, "delete", reason, force=True)
            self.add_op("delete", path, reason)
            return

        for item in sorted(path.rglob("*")):
            if item.is_dir() and not item.is_symlink():
                continue
            try:
                if item.resolve() in skip_resolved:
                    continue
            except OSError:
                pass
            self.add_entry(item, "delete", reason, force=True)
        self.add_entry(path, "delete", reason, force=True)
        self.add_op("delete", path, reason)

    def move_path(self, src: Path, dst: Path, reason: str) -> None:
        if not src.exists() and not src.is_symlink():
            return
        if src.is_file() or src.is_symlink():
            self.add_entry(src, "move", reason, dst, force=True)
        else:
            for item in sorted(src.rglob("*")):
                if item.is_dir() and not item.is_symlink():
                    continue
                self.add_entry(item, "move", reason, dst / item.relative_to(src), force=True)
            self.add_entry(src, "move", reason, dst, force=True)
        self.add_op("move", src, reason, dst)

    def symlink(self, link: Path, target: Path, reason: str) -> None:
        self.add_entry(link, "keep", reason, target, force=True)
        self.add_op("symlink", link, reason, target)


def iter_files(path: Path) -> Iterable[Path]:
    if not path.exists() and not path.is_symlink():
        return []
    if path.is_file() or path.is_symlink():
        return [path]
    return [p for p in sorted(path.rglob("*")) if p.is_file() or p.is_symlink()]


def first_existing(paths: list[str]) -> Path | None:
    for rel in paths:
        p = ROOT / rel
        if p.exists() or p.is_symlink():
            return p
    return None


def looks_like_throwaway_xml(name: str) -> bool:
    lowered = name.lower()
    tokens = ["__v46_tmp_trial", "tmp", "trial", "perturb", "viewer", "no_support", "free_preview"]
    return any(tok in lowered for tok in tokens)


def add_static_keeps(plan: CleanupPlan) -> None:
    for name in sorted(PROTECTED_TOP_LEVEL):
        p = ROOT / name
        if p.exists():
            plan.keep(p, "protected top-level directory; cleanup script never mutates it")

    for rel in sorted(MANDATORY_ROOT_FILES):
        p = ROOT / rel
        if p.exists():
            plan.keep(p, "mandatory root executable for clean MuJoCo environment")

    for rel in sorted(MANDATORY_SCRIPT_PATHS):
        p = ROOT / rel
        if p.exists() or p.is_symlink():
            plan.keep(p, "mandatory V4.12 execution/diagnostic/precheck script")

    for rel in [
        "diagnostics/README.md",
        "diagnostics/current_v412",
        "records/README.md",
        "records/precheck",
        "records/runs",
        "docs/PROJECT_STRUCTURE_V412.md",
        "docs/CLEANUP_AUDIT_V412.md",
        "docs/CLEANUP_PLAN_V412_AGGRESSIVE.md",
        "docs/CLEANUP_MANIFEST_V412_AGGRESSIVE.json",
    ]:
        p = ROOT / rel
        if p.exists():
            plan.keep(p, "kept V4.12 structure, records, diagnostics, or cleanup documentation")


def plan_directories(plan: CleanupPlan) -> None:
    for rel in [
        "legacy_final_demos",
        "models/fr3_o7/main_xml",
        "models/fr3_o7/archive_xml",
        "scripts/07_precheck_ik_collision/ik_solvers",
        "scripts/07_precheck_ik_collision/collision_precheck",
        "scripts/07_precheck_ik_collision/clearance_eval",
        "records/precheck",
        "records/runs",
        "records/smoke_test_candidates",
        "diagnostics/current_v412",
        "docs/archive",
    ]:
        plan.mkdir(ROOT / rel, "target V4.12 cleanup directory")


def plan_final_demo_moves(plan: CleanupPlan) -> None:
    for name in FINAL_DEMOS:
        src = ROOT / name
        dst = ROOT / "legacy_final_demos" / name
        if src.exists():
            plan.move_path(src, dst, "move final demo bundle under legacy_final_demos")
        elif dst.exists():
            plan.keep(dst, "legacy final demo already moved")


def plan_docs(plan: CleanupPlan) -> None:
    old_demo = ROOT / "docs/o7_grasp_demo_stage_record.md"
    if old_demo.exists():
        plan.move_path(old_demo, ROOT / "docs/archive/o7_grasp_demo_stage_record.md", "old demo stage record; archive under docs/archive")
    for rel in ["docs/00_project_record", "docs/10_method_notes", "docs/20_run_guides"]:
        p = ROOT / rel
        if p.exists():
            plan.delete_path(p, "empty or legacy docs staging directory not in aggressive V4.12 target")


def plan_models(plan: CleanupPlan) -> None:
    model_dir = ROOT / "models/fr3_o7"
    if not model_dir.exists():
        plan.warnings.append("models/fr3_o7 does not exist")
        return

    mesh_dir = model_dir / "converted_meshes_obj"
    if mesh_dir.exists():
        for p in iter_files(mesh_dir):
            plan.keep(p, "protected converted OBJ/MTL mesh asset")

    for p in sorted(model_dir.glob("*.urdf")):
        plan.keep(p, "protected URDF asset")

    for p in sorted(model_dir.glob("*.xml")):
        if p.name in MANDATORY_MODEL_XML:
            dst = model_dir / "main_xml" / p.name
            plan.move_path(p, dst, "current mainline XML moved to main_xml; root compatibility symlink will be maintained")
            plan.symlink(p, dst, "compatibility symlink for existing scripts/defaults")
        elif looks_like_throwaway_xml(p.name):
            plan.delete_path(p, "old tmp/trial/viewer/no_support/free_preview XML")
        else:
            plan.move_path(p, model_dir / "archive_xml" / p.name, "non-mainline XML kept out of root as archive_xml")


def plan_diagnostics(plan: CleanupPlan) -> None:
    diag = ROOT / "diagnostics"
    if not diag.exists():
        return
    for child in sorted(diag.iterdir()):
        if child.name in {"README.md", "current_v412"}:
            continue
        plan.delete_path(child, "old V4.0-V4.11 diagnostics/logs/results removed for V4.12 reset")


def plan_records(plan: CleanupPlan) -> None:
    records = ROOT / "records"
    if not records.exists():
        return

    smoke_sources: dict[str, Path] = {}
    for label, hints in SMOKE_CANDIDATE_HINTS.items():
        p = first_existing(hints)
        if p is None and label == "can52":
            matches = sorted((ROOT / "records/candidates").glob("**/*can*.json"))
            p = matches[0] if matches else None
        if p is not None:
            smoke_sources[label] = p
            dst = ROOT / "records/smoke_test_candidates" / f"{label}_{p.name}"
            plan.move_path(p, dst, f"minimal {label} smoke test candidate for V4.12 precheck/runner")

    skip = set(smoke_sources.values())

    for p in sorted(records.glob("stable_*.json")):
        plan.keep(p, "stable grasp record kept as minimal known-good reference")
    omt = records / "o7_mount_transform_debug.json"
    if omt.exists():
        plan.keep(omt, "mount transform reference kept for V4.12 frame checks")

    for child in sorted(records.iterdir()):
        if child.name in {"README.md", "precheck", "runs", "smoke_test_candidates"}:
            continue
        if child.name.startswith("stable_") or child.name == "o7_mount_transform_debug.json":
            continue
        plan.delete_path(child, "old candidates/plans/templates/results removed; smoke candidates moved separately", skip=skip)


def plan_scripts(plan: CleanupPlan) -> None:
    scripts = ROOT / "scripts"
    if not scripts.exists():
        return

    for cache_dir in sorted(scripts.rglob("__pycache__")):
        plan.delete_path(cache_dir, "Python bytecode cache removed")

    # Delete known legacy trees first.
    for rel in [
        "scripts/01_scene_modeling",
        "scripts/02_dataset_candidate",
        "scripts/03_candidate_scoring",
        "scripts/04_planning_legacy_v4_v11",
        "scripts/90_legacy_misc",
    ]:
        plan.delete_path(ROOT / rel, "legacy generation/scoring/search/cache scripts removed for V4.12 mainline")

    # Keep only mandatory execution runner files in 05.
    runner_dir = ROOT / "scripts/05_execution_runner"
    if runner_dir.exists():
        keep = {
            runner_dir / "run_fr3_o7_candidate_grasp_site_servo_debug.py",
            runner_dir / "run_fr3_o7_object_ik_grasp_v12.py",
        }
        for child in sorted(runner_dir.iterdir()):
            if child in keep:
                plan.keep(child, "mandatory V4.12 execution dependency")
            else:
                plan.delete_path(child, "old runner/demo/optimized-result script removed")

    viewer_dir = ROOT / "scripts/06_diagnostics_viewer"
    if viewer_dir.exists():
        keep = viewer_dir / "diagnose_v4_9_frame_ik_contact_debug.py"
        for child in sorted(viewer_dir.iterdir()):
            if child == keep:
                plan.keep(child, "mandatory frame/IK/contact diagnostic")
            else:
                plan.delete_path(child, "old viewer/preview script removed")

    # Root compatibility layer: keep mandatory root wrappers, remove old symlinks/files.
    keep_root = {
        ROOT / "scripts/README.md",
        ROOT / "scripts/run_fr3_o7_candidate_grasp_site_servo_debug.py",
        ROOT / "scripts/run_fr3_o7_candidate_grasp_site_debug.py",
        ROOT / "scripts/run_fr3_o7_object_ik_grasp_v12.py",
    }
    for child in sorted(scripts.iterdir()):
        if child.is_dir() and not child.is_symlink():
            continue
        if child in keep_root:
            plan.keep(child, "mandatory root compatibility entry")
        elif child.name.endswith(".py") or child.is_symlink():
            plan.delete_path(child, "old root compatibility symlink/script removed")

    plan.symlink(
        ROOT / "scripts/run_fr3_o7_candidate_grasp_site_servo_debug.py",
        ROOT / "scripts/05_execution_runner/run_fr3_o7_candidate_grasp_site_servo_debug.py",
        "repair/maintain mandatory current runner compatibility symlink",
    )
    plan.symlink(
        ROOT / "scripts/run_fr3_o7_object_ik_grasp_v12.py",
        ROOT / "scripts/05_execution_runner/run_fr3_o7_object_ik_grasp_v12.py",
        "repair/maintain mandatory V12 runner compatibility symlink",
    )


def plan_old_archives(plan: CleanupPlan) -> None:
    for child in sorted(ROOT.iterdir()):
        if child.name.startswith("_archive_after_"):
            plan.delete_path(child, "old conservative archive removed during aggressive V4.12 reset")


def build_plan() -> CleanupPlan:
    plan = CleanupPlan()
    add_static_keeps(plan)
    plan_directories(plan)
    plan_final_demo_moves(plan)
    plan_docs(plan)
    plan_models(plan)
    plan_diagnostics(plan)
    plan_records(plan)
    plan_scripts(plan)
    plan_old_archives(plan)

    for rel in sorted(MANDATORY_ROOT_FILES):
        p = ROOT / rel
        if not p.exists():
            plan.warnings.append(f"mandatory file missing: {rel}")
    for rel in sorted(MANDATORY_SCRIPT_PATHS):
        p = ROOT / rel
        if not (p.exists() or p.is_symlink()):
            plan.warnings.append(f"mandatory script missing before cleanup: {rel}")
    return plan


def write_docs(plan: CleanupPlan) -> None:
    MANIFEST_PATH.parent.mkdir(parents=True, exist_ok=True)
    entries = sorted(plan.entries.values(), key=lambda e: (e["action"], e["path"]))
    MANIFEST_PATH.write_text(json.dumps(entries, indent=2, ensure_ascii=False) + "\n")

    counts: dict[str, int] = {}
    bytes_by_action: dict[str, int] = {}
    for e in entries:
        counts[e["action"]] = counts.get(e["action"], 0) + 1
        bytes_by_action[e["action"]] = bytes_by_action.get(e["action"], 0) + int(e["size_bytes"])

    lines = [
        "# Aggressive Cleanup Plan V4.12",
        "",
        "Generated by `scripts/00_maintenance/aggressive_cleanup_for_v412.py --dry-run --emit-docs`.",
        "",
        "This plan is intentionally aggressive: old V4.0-V4.11 logs, candidates, viewers, demos, search scripts, and conservative archives are deleted or moved out of the mainline.",
        "",
        "Protected invariants:",
        "",
        "- `dataset/` is never mutated.",
        "- `assets/` is never mutated.",
        "- `models/fr3_o7/**/*.obj`, `*.mtl`, and `*.urdf` are kept.",
        "- Mandatory current runner, V12 dependency, diagnose_v4_9 script, and `scripts/07_precheck_ik_collision/` are kept.",
        "- `--apply` requires typing `YES`.",
        "",
        "Manifest:",
        "",
        f"- JSON manifest: `docs/{MANIFEST_PATH.name}`",
        f"- Entries: {len(entries)}",
    ]
    for action in sorted(counts):
        lines.append(f"- {action}: {counts[action]} entries, {bytes_by_action[action]} bytes")

    lines.extend([
        "",
        "Major actions:",
        "",
        "- Move final demo bundles into `legacy_final_demos/`.",
        "- Move old demo notes into `docs/archive/`.",
        "- Move the three required XML scenes into `models/fr3_o7/main_xml/` and maintain root compatibility symlinks.",
        "- Move non-mainline XML to `models/fr3_o7/archive_xml/`; delete temporary V4.6 trial XML.",
        "- Delete old diagnostics except `diagnostics/current_v412/` and `diagnostics/README.md`.",
        "- Move one bottle, one can52, and one cylinder candidate into `records/smoke_test_candidates/`, then delete old candidates/plans/templates.",
        "- Delete legacy V4.0-V4.11 scripts and root compatibility symlinks that no longer point to kept mainline code.",
        "",
        "Post-apply checks performed by the script:",
        "",
        "1. `dataset/` exists.",
        "2. `assets/` exists.",
        "3. `run_mujoco_clean.sh` exists.",
        "4. Current runner exists.",
        "5. `diagnose_v4_9_frame_ik_contact_debug.py` exists.",
        "6. `scripts/07_precheck_ik_collision/` exists.",
        "7. `models/fr3_o7` still contains URDF and OBJ files.",
        "8. Remaining file count and disk usage are printed.",
        "",
    ])
    if plan.warnings:
        lines.append("Warnings:")
        lines.append("")
        for w in plan.warnings:
            lines.append(f"- {w}")
        lines.append("")

    PLAN_DOC_PATH.write_text("\n".join(lines))


def print_plan(plan: CleanupPlan) -> None:
    entries = sorted(plan.entries.values(), key=lambda e: (e["action"], e["path"]))
    counts: dict[str, int] = {}
    for e in entries:
        counts[e["action"]] = counts.get(e["action"], 0) + 1
    print("========== AGGRESSIVE CLEANUP PLAN V4.12 ==========")
    print(f"ROOT: {ROOT}")
    for action in sorted(counts):
        print(f"{action:>6}: {counts[action]} entries")
    if plan.warnings:
        print("WARNINGS:")
        for w in plan.warnings:
            print(f"  - {w}")
    print()
    for e in entries:
        target = f" -> {e['target']}" if e["target"] else ""
        print(f"[{e['action'].upper()}] {e['path']}{target} :: {e['reason']}")


def remove_empty_dirs(start: Path) -> None:
    if not start.exists() or not start.is_dir():
        return
    for d in sorted([p for p in start.rglob("*") if p.is_dir()], reverse=True):
        try:
            d.rmdir()
        except OSError:
            pass


def apply_ops(plan: CleanupPlan) -> None:
    for op in plan.ops:
        if op.action == "mkdir":
            op.path.mkdir(parents=True, exist_ok=True)

    for op in plan.ops:
        if op.action != "move":
            continue
        if not op.path.exists() and not op.path.is_symlink():
            continue
        assert op.target is not None
        if op.target.exists() or op.target.is_symlink():
            raise RuntimeError(f"target already exists: {op.target}")
        op.target.parent.mkdir(parents=True, exist_ok=True)
        shutil.move(str(op.path), str(op.target))

    for op in plan.ops:
        if op.action != "symlink":
            continue
        assert op.target is not None
        op.path.parent.mkdir(parents=True, exist_ok=True)
        desired = op.target
        if op.path.is_symlink():
            current = op.path.resolve()
            if current == desired.resolve():
                continue
            op.path.unlink()
        elif op.path.exists():
            raise RuntimeError(f"cannot replace non-symlink with symlink: {op.path}")
        os.symlink(desired, op.path)

    for op in plan.ops:
        if op.action != "delete":
            continue
        p = op.path
        if p.is_symlink() or p.is_file():
            try:
                p.unlink()
            except FileNotFoundError:
                pass
        elif p.is_dir():
            shutil.rmtree(p, ignore_errors=True)

    for rel in ["scripts", "records", "diagnostics", "models/fr3_o7", "docs"]:
        remove_empty_dirs(ROOT / rel)


def count_files() -> int:
    total = 0
    for p in ROOT.rglob("*"):
        rel = p.relative_to(ROOT)
        if rel.parts and rel.parts[0] == ".git":
            continue
        if p.is_file() or p.is_symlink():
            total += 1
    return total


def disk_usage_bytes(path: Path) -> int:
    total = 0
    for p in path.rglob("*"):
        try:
            if p.is_file() and not p.is_symlink():
                total += p.stat().st_size
            elif p.is_symlink():
                total += p.lstat().st_size
        except OSError:
            pass
    return total


def post_apply_checks() -> None:
    checks = [
        (ROOT / "dataset", "dataset/ exists"),
        (ROOT / "assets", "assets/ exists"),
        (ROOT / "run_mujoco_clean.sh", "run_mujoco_clean.sh exists"),
        (ROOT / "scripts/05_execution_runner/run_fr3_o7_candidate_grasp_site_servo_debug.py", "current runner exists"),
        (ROOT / "scripts/06_diagnostics_viewer/diagnose_v4_9_frame_ik_contact_debug.py", "diagnose_v4_9 exists"),
        (ROOT / "scripts/07_precheck_ik_collision", "precheck directory exists"),
    ]
    ok = True
    for path, label in checks:
        exists = path.exists() or path.is_symlink()
        print(f"[{'OK' if exists else 'FAIL'}] {label}")
        ok = ok and exists

    urdf_exists = any((ROOT / "models/fr3_o7").glob("*.urdf"))
    obj_exists = any((ROOT / "models/fr3_o7").glob("**/*.obj"))
    print(f"[{'OK' if urdf_exists else 'FAIL'}] models/fr3_o7 has URDF")
    print(f"[{'OK' if obj_exists else 'FAIL'}] models/fr3_o7 has OBJ")
    ok = ok and urdf_exists and obj_exists

    print(f"remaining_file_count: {count_files()}")
    print(f"disk_usage_bytes    : {disk_usage_bytes(ROOT)}")
    if not ok:
        raise RuntimeError("post-apply checks failed")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    mode = ap.add_mutually_exclusive_group(required=True)
    mode.add_argument("--dry-run", action="store_true", help="print plan only; no cleanup mutation")
    mode.add_argument("--apply", action="store_true", help="execute cleanup after YES confirmation")
    ap.add_argument("--emit-docs", action="store_true", help="with --dry-run, refresh plan markdown and manifest json")
    args = ap.parse_args()

    plan = build_plan()
    print_plan(plan)

    if args.dry_run:
        if args.emit_docs:
            write_docs(plan)
            print(f"\nWrote {PLAN_DOC_PATH.relative_to(ROOT)}")
            print(f"Wrote {MANIFEST_PATH.relative_to(ROOT)}")
        return 0

    if args.emit_docs:
        print("--emit-docs is only accepted with --dry-run", file=sys.stderr)
        return 2

    answer = input("\nType YES to execute aggressive cleanup: ")
    if answer != "YES":
        print("Aborted; no changes made.")
        return 1

    write_docs(plan)
    apply_ops(plan)
    post_apply_checks()
    print("Aggressive cleanup complete.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
