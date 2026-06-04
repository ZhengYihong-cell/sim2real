#!/usr/bin/env python3
"""
脚本类型：
    debug / runner / clean-rewrite

用途：
    V4.12P4U1 ready-gated snap close + micro-squeeze + gated world-z lift。

输入：
    --model        MuJoCo XML 场景
    --candidate    候选抓握 JSON
    --p3-json      P3 结果 JSON，提供 FR3 q_grasp / q_pre
    --best-config  P4H/P4E/P4H2 结果 JSON，提供 O7 hand_config.ctrl
    --out          输出 JSON

输出：
    调试 JSON，记录 snap、grip_settle、micro_squeeze、lift 各阶段接触状态。

当前流程位置：
    P4T2 已恢复旧 ellipsoid proxy 后，用于验证 can52 侧握是否能形成稳定对握。

核心逻辑：
    1. FR3 到达 q_grasp，手保持 side-open；
    2. snap close 到 close_target；
    3. 保持 close_target 一小段时间，让执行器追上，而不是立刻冻结真实 qpos；
    4. 如果未形成稳定 thumb-finger 对抗，执行小幅 gated micro-squeeze；
    5. 只有稳定对握达到 --grip-ready-stable-steps，才允许 lift；
    6. lift 阶段保持固定 grip_hold_ctrl，不再继续闭合；
    7. 如果没抓紧，直接 NO_LIFT，不抬。

不负责：
    1. 不重新做大范围 IK；
    2. 不加入 touch sensor；
    3. 不把没有 ready 的状态强行 lift；
    4. 不在 lift 阶段继续改变手指目标。
"""

from pathlib import Path
import argparse
import json
import time
import numpy as np
import mujoco
import mujoco.viewer


PROJECT = Path.home() / "Projects/o7_mujoco_sim"

ARM_JOINTS = [f"fr3_joint{i}" for i in range(1, 8)]
O7_ACTIVE_JOINTS = [
    "thumb_cmc_roll",
    "thumb_cmc_yaw",
    "thumb_cmc_pitch",
    "index_mcp_pitch",
    "middle_mcp_pitch",
    "ring_mcp_pitch",
    "pinky_mcp_pitch",
]
FINGER_JOINTS = ["index_mcp_pitch", "middle_mcp_pitch", "ring_mcp_pitch", "pinky_mcp_pitch"]


def resolve_path(p):
    p = Path(str(p)).expanduser()
    if not p.is_absolute():
        p = PROJECT / p
    return p


def load_json(path):
    with open(resolve_path(path), "r") as f:
        return json.load(f)


def to_jsonable(x):
    if isinstance(x, dict):
        return {str(k): to_jsonable(v) for k, v in x.items()}
    if isinstance(x, (list, tuple)):
        return [to_jsonable(v) for v in x]
    if isinstance(x, np.ndarray):
        return x.tolist()
    if isinstance(x, np.generic):
        return x.item()
    if isinstance(x, Path):
        return str(x)
    return x


def save_json(path, obj):
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(to_jsonable(obj), f, indent=2)


def get_nested(obj, path):
    cur = obj
    for k in path:
        if not isinstance(cur, dict) or k not in cur:
            return None
        cur = cur[k]
    return cur


def has_keys(d, keys):
    return isinstance(d, dict) and all(k in d for k in keys)


def find_dict_with_keys(obj, keys):
    if has_keys(obj, keys):
        return obj
    if isinstance(obj, dict):
        for v in obj.values():
            r = find_dict_with_keys(v, keys)
            if r is not None:
                return r
    if isinstance(obj, list):
        for v in obj:
            r = find_dict_with_keys(v, keys)
            if r is not None:
                return r
    return None


def extract_arm_plan(p3, candidate):
    q_grasp = None
    q_pre = None
    source = {}

    paths_grasp = [
        ["best_pass", "q_grasp"],
        ["best", "q_grasp"],
        ["plan", "q_grasp"],
        ["q_grasp"],
        ["target", "q_grasp"],
    ]

    for path in paths_grasp:
        x = get_nested(p3, path)
        if has_keys(x, ARM_JOINTS):
            q_grasp = x
            source["q_grasp"] = "p3." + ".".join(path)
            break

    if q_grasp is None:
        x = find_dict_with_keys(p3, ARM_JOINTS)
        if x is not None:
            q_grasp = x
            source["q_grasp"] = "p3.recursive"
        else:
            x = find_dict_with_keys(candidate, ARM_JOINTS)
            if x is not None:
                q_grasp = x
                source["q_grasp"] = "candidate.recursive"

    if q_grasp is None:
        raise RuntimeError("cannot find q_grasp with fr3_joint1..fr3_joint7")

    paths_pre = [
        ["best_pass", "q_pre"],
        ["best", "q_pre"],
        ["plan", "q_pre"],
        ["q_pre"],
        ["target", "q_pre"],
    ]
    for path in paths_pre:
        x = get_nested(p3, path)
        if has_keys(x, ARM_JOINTS):
            q_pre = x
            source["q_pre"] = "p3." + ".".join(path)
            break

    if q_pre is None:
        q_pre = dict(q_grasp)
        source["q_pre"] = "fallback=q_grasp"

    q_grasp = {j: float(q_grasp[j]) for j in ARM_JOINTS}
    q_pre = {j: float(q_pre[j]) for j in ARM_JOINTS}
    return q_pre, q_grasp, source


def extract_hand_ctrl(best_config, candidate):
    paths = [
        ["best_record", "hand_config", "ctrl"],
        ["best", "best_record", "hand_config", "ctrl"],
        ["best", "hand_config", "ctrl"],
        ["hand_config", "ctrl"],
        ["ctrl"],
        ["o7_ctrl"],
        ["hand_ctrl"],
        ["target", "o7_ctrl"],
    ]

    for obj, root_name in [(best_config, "best"), (candidate, "candidate")]:
        for path in paths:
            x = get_nested(obj, path)
            if has_keys(x, O7_ACTIVE_JOINTS):
                return {j: float(x[j]) for j in O7_ACTIVE_JOINTS}, root_name + "." + ".".join(path)

        x = find_dict_with_keys(obj, O7_ACTIVE_JOINTS)
        if x is not None:
            return {j: float(x[j]) for j in O7_ACTIVE_JOINTS}, root_name + ".recursive"

    raise RuntimeError("cannot find O7 active hand ctrl")


def mj_name(model, objtype, idx, fallback):
    name = mujoco.mj_id2name(model, objtype, int(idx))
    return name if name else f"{fallback}_{idx}"


def body_name(model, bid):
    return mj_name(model, mujoco.mjtObj.mjOBJ_BODY, bid, "body")


def geom_name(model, gid):
    return mj_name(model, mujoco.mjtObj.mjOBJ_GEOM, gid, "geom")


def joint_name(model, jid):
    return mj_name(model, mujoco.mjtObj.mjOBJ_JOINT, jid, "joint")


def actuator_name(model, aid):
    return mj_name(model, mujoco.mjtObj.mjOBJ_ACTUATOR, aid, "actuator")


def build_ctrl_map(model):
    out = {}
    for aid in range(model.nu):
        aname = actuator_name(model, aid)
        out[aname] = aid
        if aname.endswith("_pos"):
            out[aname[:-4]] = aid

        jid = int(model.actuator_trnid[aid, 0])
        if 0 <= jid < model.njnt:
            jname = joint_name(model, jid)
            out[jname] = aid
            out[jname + "_pos"] = aid
    return out


def clamp_ctrl(model, aid, value):
    value = float(value)
    if bool(model.actuator_ctrllimited[aid]):
        lo, hi = model.actuator_ctrlrange[aid]
        value = min(max(value, float(lo)), float(hi))
    return value


def set_ctrl_dict(model, data, ctrl_map, ctrl_dict):
    for name, val in ctrl_dict.items():
        if name not in ctrl_map:
            continue
        aid = ctrl_map[name]
        data.ctrl[aid] = clamp_ctrl(model, aid, val)


def set_qpos_dict(model, data, qdict):
    for name, val in qdict.items():
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        qadr = int(model.jnt_qposadr[jid])
        if 0 <= qadr < model.nq:
            data.qpos[qadr] = float(val)


def read_qpos_dict(model, data, joints):
    out = {}
    for name in joints:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
        if jid < 0:
            continue
        qadr = int(model.jnt_qposadr[jid])
        if 0 <= qadr < model.nq:
            out[name] = float(data.qpos[qadr])
    return out


def interp_dict(a, b, alpha):
    out = {}
    keys = set(a.keys()) | set(b.keys())
    for k in keys:
        av = float(a.get(k, b.get(k, 0.0)))
        bv = float(b.get(k, a.get(k, 0.0)))
        out[k] = (1.0 - alpha) * av + alpha * bv
    return out


def make_side_open_and_close(hand_prior, finger_scale, thumb_gain, thumb_open_pitch):
    side = {}
    side["thumb_cmc_roll"] = float(hand_prior["thumb_cmc_roll"])
    side["thumb_cmc_yaw"] = float(hand_prior["thumb_cmc_yaw"])
    side["thumb_cmc_pitch"] = min(float(hand_prior["thumb_cmc_pitch"]), float(thumb_open_pitch))
    for j in FINGER_JOINTS:
        side[j] = 0.0

    close = {}
    close["thumb_cmc_roll"] = float(hand_prior["thumb_cmc_roll"])
    close["thumb_cmc_yaw"] = float(hand_prior["thumb_cmc_yaw"])

    finger_vals = []
    for j in FINGER_JOINTS:
        v = max(0.0, float(hand_prior[j]) * float(finger_scale))
        close[j] = v
        finger_vals.append(v)

    mean_finger = float(np.mean(finger_vals)) if finger_vals else 0.0
    close["thumb_cmc_pitch"] = side["thumb_cmc_pitch"] + float(thumb_gain) * mean_finger
    return side, close


def body_id(model, name):
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if bid < 0:
        raise RuntimeError(f"cannot find body: {name}")
    return bid


def geoms_of_body(model, bid):
    return {gid for gid in range(model.ngeom) if int(model.geom_bodyid[gid]) == int(bid)}


def hand_group_from_geom(model, gid):
    text = (geom_name(model, gid) + " " + body_name(model, model.geom_bodyid[gid])).lower()
    for g in ["thumb", "index", "middle", "ring", "pinky"]:
        if g in text:
            return g
    if "hand" in text or "palm" in text:
        return "palm"
    return None


def collect_live_contact(model, data, object_geoms, args):
    groups = {}
    dirs = {}

    for i in range(data.ncon):
        c = data.contact[i]
        g1 = int(c.geom1)
        g2 = int(c.geom2)
        g1_obj = g1 in object_geoms
        g2_obj = g2 in object_geoms

        if not (g1_obj or g2_obj):
            continue

        hand_gid = g2 if g1_obj else g1
        group = hand_group_from_geom(model, hand_gid)
        if group is None or group == "palm":
            continue

        groups[group] = groups.get(group, 0) + 1

        n = np.array(c.frame[:3], dtype=float)
        n_norm = np.linalg.norm(n)
        if n_norm > 1e-9:
            n = n / n_norm

            # MuJoCo normal is along contact frame x-axis. Direction sign convention is not
            # important for same-side counts, but matters for opposition. This convention
            # matches the intended "force on object" approximation.
            if g1_obj:
                d = -n
            else:
                d = n
            dirs.setdefault(group, []).append(d)

    thumb_dirs = dirs.get("thumb", [])
    non_dirs = []
    for g in ["index", "middle", "ring", "pinky"]:
        non_dirs.extend(dirs.get(g, []))

    opposition_cos = None
    if thumb_dirs and non_dirs:
        vals = []
        for a in thumb_dirs:
            for b in non_dirs:
                vals.append(float(np.dot(a, b)))
        opposition_cos = min(vals) if vals else None

    non_count = sum(groups.get(g, 0) for g in ["index", "middle", "ring", "pinky"])
    ready = (
        groups.get("thumb", 0) > 0
        and non_count >= int(args.min_live_non_thumb)
        and opposition_cos is not None
        and opposition_cos <= float(args.opposition_cos_threshold)
    )

    return {
        "groups": groups,
        "opposition_cos": opposition_cos,
        "ready": bool(ready),
        "non_thumb_contact_count": int(non_count),
    }


def solve_world_z_lift_qpos_dict(model, data, q_seed, target_body_name, lift_z,
                                 ik_iters=140, damping=1e-4, step_scale=0.65):
    q_out = dict(q_seed)
    bid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, target_body_name)
    if bid < 0:
        return q_out, {"success": False, "reason": "target_body_not_found", "target_body": target_body_name}

    qpos_backup = data.qpos.copy()
    qvel_backup = data.qvel.copy()

    set_qpos_dict(model, data, q_seed)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    start_pos = data.xpos[bid].copy()
    target_pos = start_pos + np.array([0.0, 0.0, float(lift_z)])

    dof_ids = []
    qadr_ids = []
    jids = []
    for jname in ARM_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            continue
        dof = int(model.jnt_dofadr[jid])
        qadr = int(model.jnt_qposadr[jid])
        dof_ids.append(dof)
        qadr_ids.append(qadr)
        jids.append(jid)

    for _ in range(int(ik_iters)):
        mujoco.mj_forward(model, data)
        err = target_pos - data.xpos[bid]
        if np.linalg.norm(err) < 5e-4:
            break

        jacp = np.zeros((3, model.nv))
        jacr = np.zeros((3, model.nv))
        mujoco.mj_jacBody(model, data, jacp, jacr, bid)

        J = jacp[:, dof_ids]
        A = J @ J.T + float(damping) * np.eye(3)
        dq = J.T @ np.linalg.solve(A, err)
        dq *= float(step_scale)

        n = float(np.linalg.norm(dq))
        if n > 0.08:
            dq = dq / n * 0.08

        for qadr, jid, delta in zip(qadr_ids, jids, dq):
            data.qpos[qadr] += float(delta)
            if bool(model.jnt_limited[jid]):
                lo, hi = model.jnt_range[jid]
                data.qpos[qadr] = min(max(data.qpos[qadr], lo), hi)

    mujoco.mj_forward(model, data)
    final_pos = data.xpos[bid].copy()
    final_err = float(np.linalg.norm(target_pos - final_pos))
    actual_rise = float(final_pos[2] - start_pos[2])

    for jname in ARM_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, jname)
        if jid < 0:
            continue
        qadr = int(model.jnt_qposadr[jid])
        q_out[jname] = float(data.qpos[qadr])

    info = {
        "success": bool(final_err < 0.003),
        "target_body": target_body_name,
        "requested_lift_z": float(lift_z),
        "actual_target_body_rise": actual_rise,
        "final_err": final_err,
        "start_pos": start_pos,
        "target_pos": target_pos,
        "final_pos": final_pos,
    }

    data.qpos[:] = qpos_backup
    data.qvel[:] = qvel_backup
    mujoco.mj_forward(model, data)
    return q_out, info


class Runner:
    def __init__(self, model, data, ctrl_map, obj_bid, object_geoms, args):
        self.model = model
        self.data = data
        self.ctrl_map = ctrl_map
        self.obj_bid = obj_bid
        self.object_geoms = object_geoms
        self.args = args
        self.rows = []
        self.stable_count = 0
        self.max_stable_count = 0
        self.viewer = None
        self.obj_pos0 = data.xpos[obj_bid].copy()
        self.obj_z0 = float(self.obj_pos0[2])

    def attach_viewer(self, viewer):
        self.viewer = viewer

    def object_disp(self):
        return float(np.linalg.norm(self.data.xpos[self.obj_bid] - self.obj_pos0))

    def object_rise(self):
        return float(self.data.xpos[self.obj_bid][2] - self.obj_z0)

    def step_once(self, phase, step, alpha, arm_ctrl, hand_ctrl):
        set_ctrl_dict(self.model, self.data, self.ctrl_map, arm_ctrl)
        set_ctrl_dict(self.model, self.data, self.ctrl_map, hand_ctrl)
        mujoco.mj_step(self.model, self.data)

        live = collect_live_contact(self.model, self.data, self.object_geoms, self.args)
        if live["ready"]:
            self.stable_count += 1
        else:
            self.stable_count = 0
        self.max_stable_count = max(self.max_stable_count, self.stable_count)

        row = {
            "phase": phase,
            "step": int(step),
            "time": float(self.data.time),
            "alpha": float(alpha),
            "object_pos": self.data.xpos[self.obj_bid].copy(),
            "object_disp": self.object_disp(),
            "object_rise": self.object_rise(),
            "groups": live["groups"],
            "opposition_cos": live["opposition_cos"],
            "ready": live["ready"],
            "stable_count": int(self.stable_count),
            "hand_ctrl": dict(hand_ctrl),
        }

        if step % int(self.args.print_every_steps) == 0 or step == 0:
            print(
                f"[{phase}] {step:5d} alpha={alpha:.3f} "
                f"disp={row['object_disp']:.5f} rise={row['object_rise']:.5f} "
                f"groups={row['groups']} opp={row['opposition_cos']} "
                f"ready={row['ready']} stable={self.stable_count}/{self.args.grip_ready_stable_steps}"
            )

        if step % int(self.args.log_every_steps) == 0 or step == 0:
            self.rows.append(row)

        if self.viewer is not None:
            self.viewer.sync()
            if self.args.frame_sleep > 0:
                time.sleep(float(self.args.frame_sleep))

        return live

    def run_phase(self, phase, steps, arm_start, arm_end, hand_start, hand_end):
        print(f"\n[PHASE] {phase}, steps={steps}")
        last_live = None
        for k in range(int(steps) + 1):
            alpha = 1.0 if steps <= 0 else k / float(steps)
            arm = interp_dict(arm_start, arm_end, alpha)
            hand = interp_dict(hand_start, hand_end, alpha)
            last_live = self.step_once(phase, k, alpha, arm, hand)
        return last_live

    def run_hold(self, phase, steps, arm_ctrl, hand_ctrl):
        print(f"\n[PHASE] {phase}, steps={steps}")
        last_live = None
        for k in range(int(steps) + 1):
            alpha = 1.0 if steps <= 0 else k / float(steps)
            last_live = self.step_once(phase, k, alpha, arm_ctrl, hand_ctrl)
        return last_live


def run(args):
    model_path = resolve_path(args.model)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    ctrl_map = build_ctrl_map(model)

    candidate = load_json(args.candidate)
    p3 = load_json(args.p3_json)
    best = load_json(args.best_config)

    q_pre, q_grasp, arm_source = extract_arm_plan(p3, candidate)
    hand_prior, hand_source = extract_hand_ctrl(best, candidate)
    side_open_ctrl, close_target = make_side_open_and_close(
        hand_prior,
        finger_scale=args.finger_close_scale,
        thumb_gain=args.thumb_pitch_from_finger_gain,
        thumb_open_pitch=args.thumb_open_pitch,
    )

    obj_bid = body_id(model, args.object_body)
    target_bid = body_id(model, args.target_body)
    object_geoms = geoms_of_body(model, obj_bid)

    mujoco.mj_resetData(model, data)
    set_qpos_dict(model, data, q_grasp)
    set_qpos_dict(model, data, side_open_ctrl)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    q_lift = dict(q_grasp)
    lift_ik_info = None
    if args.enable_lift and args.lift_mode == "world_z":
        q_lift, lift_ik_info = solve_world_z_lift_qpos_dict(
            model=model,
            data=data,
            q_seed=q_grasp,
            target_body_name=args.target_body,
            lift_z=args.lift_z,
            ik_iters=args.lift_ik_iters,
            damping=args.lift_ik_damping,
        )

    print("\n========== V4.12P4U1 READY-GATED SNAP CLOSE ==========")
    print("model              :", model_path)
    print("candidate          :", resolve_path(args.candidate))
    print("p3_json            :", resolve_path(args.p3_json))
    print("best_config        :", resolve_path(args.best_config))
    print("object_body        :", args.object_body)
    print("target_body        :", args.target_body)
    print("arm_source         :", arm_source)
    print("hand_source        :", hand_source)
    print("side_open_ctrl     :", side_open_ctrl)
    print("close_target       :", close_target)
    print("enable_lift        :", args.enable_lift)
    print("lift_mode          :", args.lift_mode)
    print("lift_z             :", args.lift_z)
    print("lift_ik_info       :", lift_ik_info)
    print("grip_ready_steps   :", args.grip_ready_stable_steps)
    print("micro_squeeze_frac :", args.micro_squeeze_fraction)
    print("max_grip_disp      :", args.max_grip_disp)
    print("IMPORTANT          : lift only if grip_ready=True. Otherwise NO_LIFT.")
    print("=======================================================\n")

    runner = Runner(model, data, ctrl_map, obj_bid, object_geoms, args)

    def execute(viewer=None):
        if viewer is not None:
            runner.attach_viewer(viewer)

        dt = float(model.opt.timestep)
        settle_steps = int(args.move_steps + args.thumb_preshape_steps)
        close_steps = max(1, int(float(args.close_duration) / dt))
        post_hold_steps = max(0, int(float(args.post_close_target_hold_duration) / dt))
        micro_steps = max(0, int(float(args.micro_squeeze_duration) / dt))
        lift_steps = max(1, int(float(args.lift_duration) / dt))

        runner.run_hold(
            "settle_side_open_at_grasp",
            settle_steps,
            q_grasp,
            side_open_ctrl,
        )

        runner.run_phase(
            "snap_close_to_target",
            close_steps,
            q_grasp,
            q_grasp,
            side_open_ctrl,
            close_target,
        )

        runner.run_hold(
            "grip_settle_at_close_target",
            post_hold_steps,
            q_grasp,
            close_target,
        )

        grip_ready = runner.stable_count >= int(args.grip_ready_stable_steps)
        stop_reason = "ready_after_close_target_hold" if grip_ready else "not_ready_after_close_target_hold"

        grip_hold_ctrl = dict(close_target)

        if not grip_ready and micro_steps > 0:
            print(f"\n[PHASE] gated_micro_squeeze, steps={micro_steps}")

            squeeze_dir = {}
            for j in O7_ACTIVE_JOINTS:
                squeeze_dir[j] = float(close_target.get(j, 0.0)) - float(side_open_ctrl.get(j, 0.0))

            start_disp = runner.object_disp()
            for k in range(micro_steps + 1):
                alpha = 1.0 if micro_steps <= 0 else k / float(micro_steps)
                frac = float(args.micro_squeeze_fraction) * alpha

                hand = {}
                for j in O7_ACTIVE_JOINTS:
                    hand[j] = float(close_target.get(j, 0.0)) + frac * squeeze_dir.get(j, 0.0)

                grip_hold_ctrl = dict(hand)
                live = runner.step_once("gated_micro_squeeze", k, alpha, q_grasp, hand)

                if runner.stable_count >= int(args.grip_ready_stable_steps):
                    grip_ready = True
                    stop_reason = "ready_during_gated_micro_squeeze"
                    print("[GRIP READY] stable opposition reached. Stop squeezing.")
                    break

                disp = runner.object_disp()
                if disp > float(args.max_grip_disp):
                    stop_reason = "fail_object_disp_exceeded_during_micro_squeeze"
                    print(f"[NO GRIP] object disp exceeded: {disp:.5f} > {args.max_grip_disp:.5f}")
                    break

                if disp - start_disp > float(args.max_extra_disp_during_squeeze):
                    stop_reason = "fail_extra_disp_exceeded_during_micro_squeeze"
                    print(f"[NO GRIP] extra object disp exceeded: {disp-start_disp:.5f} > {args.max_extra_disp_during_squeeze:.5f}")
                    break

        if not grip_ready:
            print("\n[NO_LIFT] grip is not ready. Do not lift an ungrasped object.")
            final_live = collect_live_contact(model, data, object_geoms, args)
            result = {
                "format": "v4_12p4u1_ready_gated_snap_close",
                "model": str(model_path),
                "args": vars(args),
                "arm_source": arm_source,
                "hand_source": hand_source,
                "side_open_ctrl": side_open_ctrl,
                "close_target": close_target,
                "grip_ready": False,
                "stop_reason": stop_reason,
                "max_stable_count": runner.max_stable_count,
                "final_object_disp": runner.object_disp(),
                "final_object_rise": runner.object_rise(),
                "final_groups": final_live["groups"],
                "final_opposition_cos": final_live["opposition_cos"],
                "grip_hold_ctrl": grip_hold_ctrl,
                "lift_ik_info": lift_ik_info,
                "rows": runner.rows,
            }
            save_json(args.out, result)
            print_result(args.out, result)
            return result

        print("\n[GRIP_READY] lift is allowed. Hand ctrl will stay constant during lift.")
        final_live_before_lift = collect_live_contact(model, data, object_geoms, args)

        if args.enable_lift:
            runner.run_phase(
                "lift_world_z_with_fixed_grip_ctrl",
                lift_steps,
                q_grasp,
                q_lift,
                grip_hold_ctrl,
                grip_hold_ctrl,
            )
        else:
            stop_reason = "grip_ready_no_lift_requested"

        final_live = collect_live_contact(model, data, object_geoms, args)
        result = {
            "format": "v4_12p4u1_ready_gated_snap_close",
            "model": str(model_path),
            "args": vars(args),
            "arm_source": arm_source,
            "hand_source": hand_source,
            "side_open_ctrl": side_open_ctrl,
            "close_target": close_target,
            "grip_ready": True,
            "stop_reason": stop_reason,
            "max_stable_count": runner.max_stable_count,
            "final_object_disp": runner.object_disp(),
            "final_object_rise": runner.object_rise(),
            "final_groups": final_live["groups"],
            "final_opposition_cos": final_live["opposition_cos"],
            "pre_lift_groups": final_live_before_lift["groups"],
            "pre_lift_opposition_cos": final_live_before_lift["opposition_cos"],
            "grip_hold_ctrl": grip_hold_ctrl,
            "lift_ik_info": lift_ik_info,
            "rows": runner.rows,
        }
        save_json(args.out, result)
        print_result(args.out, result)
        return result

    if args.viewer:
        with mujoco.viewer.launch_passive(model, data) as viewer:
            viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_TRANSPARENT] = False
            result = execute(viewer)
            if args.keep_viewer_open:
                print("[VIEWER] keep open. Close viewer window or Ctrl+C in terminal to exit.")
                while viewer.is_running():
                    viewer.sync()
                    time.sleep(0.02)
            return result
    else:
        return execute(None)


def print_result(out_path, result):
    print("\n========== P4U1 READY-GATED RESULT ==========")
    print("out                 :", resolve_path(out_path))
    print("grip_ready          :", result.get("grip_ready"))
    print("stop_reason         :", result.get("stop_reason"))
    print("max_stable_count    :", result.get("max_stable_count"))
    print("final_object_disp   :", result.get("final_object_disp"))
    print("final_object_rise   :", result.get("final_object_rise"))
    print("final_groups        :", result.get("final_groups"))
    print("final_opposition_cos:", result.get("final_opposition_cos"))
    print("grip_hold_ctrl      :", result.get("grip_hold_ctrl"))
    print("=============================================\n")


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument("--model", required=True)
    ap.add_argument("--candidate", required=True)
    ap.add_argument("--p3-json", required=True)
    ap.add_argument("--best-config", required=True)
    ap.add_argument("--which", default="best_available")

    ap.add_argument("--object-body", default="grasp_can")
    ap.add_argument("--target-body", default="fr3_link7")
    ap.add_argument("--out", required=True)

    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--keep-viewer-open", action="store_true")
    ap.add_argument("--frame-sleep", type=float, default=0.001)

    ap.add_argument("--move-steps", type=int, default=80)
    ap.add_argument("--thumb-preshape-steps", type=int, default=80)
    ap.add_argument("--close-duration", type=float, default=0.45)

    ap.add_argument("--post-close-target-hold-duration", type=float, default=0.25)
    ap.add_argument("--micro-squeeze-duration", type=float, default=0.35)
    ap.add_argument("--micro-squeeze-fraction", type=float, default=0.08)

    ap.add_argument("--enable-lift", action="store_true")
    ap.add_argument("--lift-mode", choices=["world_z"], default="world_z")
    ap.add_argument("--lift-z", type=float, default=0.060)
    ap.add_argument("--lift-duration", type=float, default=2.0)
    ap.add_argument("--lift-ik-iters", type=int, default=140)
    ap.add_argument("--lift-ik-damping", type=float, default=1e-4)

    ap.add_argument("--finger-close-scale", type=float, default=0.92)
    ap.add_argument("--thumb-pitch-from-finger-gain", type=float, default=0.24)
    ap.add_argument("--thumb-open-pitch", type=float, default=0.22)

    ap.add_argument("--min-live-non-thumb", type=int, default=1)
    ap.add_argument("--opposition-cos-threshold", type=float, default=-0.30)

    # 旧参数保留兼容，但不再拿 999 来阻止 ready gate。
    ap.add_argument("--live-ready-stable-steps", type=int, default=999)
    ap.add_argument("--grip-ready-stable-steps", type=int, default=8)

    ap.add_argument("--max-grip-disp", type=float, default=0.006)
    ap.add_argument("--max-extra-disp-during-squeeze", type=float, default=0.003)

    ap.add_argument("--print-every-steps", type=int, default=30)
    ap.add_argument("--log-every-steps", type=int, default=30)

    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
