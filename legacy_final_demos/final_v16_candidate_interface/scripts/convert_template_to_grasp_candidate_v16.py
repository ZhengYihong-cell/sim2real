#!/usr/bin/env python3
from pathlib import Path
import argparse
import json


PROJECT = Path.home() / "Projects/o7_mujoco_sim"


def resolve_path(p):
    p = Path(p).expanduser()
    if not p.is_absolute():
        p = Path.cwd() / p
    return p


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def get_matrix(template, new_key, old_key):
    if new_key in template:
        return template[new_key]
    if old_key in template:
        return template[old_key]
    raise RuntimeError(f"template missing {new_key} / {old_key}")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--template", required=True)
    ap.add_argument("--out", required=True)

    ap.add_argument("--candidate-name", required=True)
    ap.add_argument("--model-hint", default="")

    ap.add_argument("--object-body", default="")
    ap.add_argument("--object-token", default="")
    ap.add_argument("--support-tokens", default="")
    ap.add_argument("--target-body", default="")

    ap.add_argument("--spawn-source", choices=["template", "model"], default="template")

    ap.add_argument("--pregrasp-z", type=float, default=0.0)
    ap.add_argument("--move-duration", type=float, default=5.0)
    ap.add_argument("--descend-duration", type=float, default=0.0)
    ap.add_argument("--close-duration", type=float, default=3.0)
    ap.add_argument("--lift-duration", type=float, default=5.0)
    ap.add_argument("--hold-duration", type=float, default=6.0)

    ap.add_argument("--lift-mode", choices=["joint_delta"], default="joint_delta")
    ap.add_argument("--lift-joint", default="fr3_joint4")
    ap.add_argument("--lift-delta", type=float, default=0.18)

    ap.add_argument("--min-final-hand-object", type=int, default=3)
    ap.add_argument("--min-final-rise", type=float, default=0.005)

    args = ap.parse_args()

    template_path = resolve_path(args.template)
    out_path = resolve_path(args.out)

    t = load_json(template_path)

    object_body = args.object_body or t.get("object_body", t.get("box_body", ""))
    object_token = args.object_token or t.get("object_token", object_body)
    support_tokens = args.support_tokens or t.get("support_tokens", "pedestal table")
    target_body = args.target_body or t.get("target_body", "fr3_link7")

    candidate = {
        "format": "fr3_o7_grasp_candidate_v1",
        "candidate_name": args.candidate_name,
        "source": {
            "type": "converted_from_template",
            "template_path": str(template_path),
            "model_hint": args.model_hint,
        },

        "object": {
            "body": object_body,
            "token": object_token,
            "support_tokens": support_tokens,
            "spawn_source": args.spawn_source,
            "T_world_object": get_matrix(t, "T_world_object", "T_world_box"),
        },

        "target": {
            "body": target_body,
            "T_object_target": get_matrix(t, "T_object_target", "T_box_target"),
        },

        "hand": {
            "type": "o7_active_ctrl",
            "approach_policy": "thumb_roll_yaw_to_template_pitch_and_fingers_open",
            "close_policy": "thumb_pitch_and_four_fingers_close_together",
            "o7_active_ctrl": t["o7_active_ctrl"],
        },

        "arm_seed": {
            "franka_ctrl": t["franka_ctrl"],
        },

        "execution": {
            "pregrasp": {
                "mode": "world_z_offset",
                "z_offset": args.pregrasp_z,
                "move_duration": args.move_duration,
                "descend_duration": args.descend_duration,
            },
            "close_duration": args.close_duration,
            "lift": {
                "mode": args.lift_mode,
                "joint": args.lift_joint,
                "delta": args.lift_delta,
                "duration": args.lift_duration,
            },
            "hold_duration": args.hold_duration,
        },

        "validation": {
            "min_final_hand_object": args.min_final_hand_object,
            "min_final_rise": args.min_final_rise,
        },

        "metadata": {
            "note": "This candidate is executable by run_fr3_o7_candidate_grasp_v16.py",
        },
    }

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        json.dump(candidate, f, indent=2)

    print("\n========== CONVERT TEMPLATE TO CANDIDATE V16 ==========")
    print("template      :", template_path)
    print("out           :", out_path)
    print("candidate_name:", args.candidate_name)
    print("object_body   :", object_body)
    print("object_token  :", object_token)
    print("support_tokens:", support_tokens)
    print("target_body   :", target_body)
    print("spawn_source  :", args.spawn_source)
    print("pregrasp_z    :", args.pregrasp_z)
    print("lift_joint    :", args.lift_joint)
    print("lift_delta    :", args.lift_delta)
    print("======================================================\n")


if __name__ == "__main__":
    main()
