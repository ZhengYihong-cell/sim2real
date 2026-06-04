#!/usr/bin/env python3
"""
脚本类型：
    debug / runner / collision-aware-recording-demo

用途：
    V4.12P4U6。
    已知目标 q_pre / q_grasp 和物体位置后，先规划一条不会提前碰物体的机械臂接近路径，
    再接入已经验证成功的 P4U1 ready-gated snap close + micro-squeeze + lift。

输入：
    --model
        已验证成功的 FR3+O7+can MuJoCo XML。
    --candidate
        当前成功候选 best_candidate.json。
    --p3-json
        当前成功 P3 机械臂 q_pre / q_grasp。
    --best-config
        当前成功 O7 手型参数。
    --out
        输出运行日志 JSON。
    --plan-out
        输出规划路径 JSON。

输出：
    1. collision-aware approach path；
    2. viewer 中从初始位姿到抓取 lift 的完整录屏流程；
    3. JSON 日志。

当前流程位置：
    只用于录屏展示与路径验证。
    不修改 legacy_final_demos。
    不重新搜索 grasp。
    不重新改抓握逻辑。

核心逻辑：
    1. 从 zero_clamped 起始姿态开场；
    2. 机械臂移动到 v12_safe_home；
    3. 使用 RRT-Connect 搜索 v12_safe_home -> q_pre 的无碰路径；
    4. 检查 q_pre -> q_grasp，如果直接路径不安全，也用 RRT；
    5. approach 阶段手保持 side-open；
    6. approach 执行时可使用 hard-servo，避免控制滞后造成偏离规划路径；
    7. 到 q_grasp 后，沿用 P4U1 成功抓握逻辑；
    8. 只有 grip_ready 后才 lift；
    9. lift 阶段固定 grip_hold_ctrl，不继续改变手型。

不负责：
    1. 不做新的 grasp 搜索；
    2. 不使用 auto_radial；
    3. 不允许没抓稳就强行 lift；
    4. 不在接近阶段闭合手指；
    5. 不把裸关节直线插值当路径规划。
"""

from pathlib import Path
import argparse
import json
import time
import importlib.util
import numpy as np
import mujoco
import mujoco.viewer


PROJECT = Path.home() / "Projects/o7_mujoco_sim"
P4U1_PATH = PROJECT / "scripts/05_execution_runner/run_v4_12p4u1_precontact_snap_close_debug.py"


def load_p4u1():
    spec = importlib.util.spec_from_file_location("p4u1", str(P4U1_PATH))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


p4u1 = load_p4u1()

ARM_JOINTS = p4u1.ARM_JOINTS
O7_ACTIVE_JOINTS = p4u1.O7_ACTIVE_JOINTS

V12_START_ARM = {
    "fr3_joint1": 0.00,
    "fr3_joint2": -0.70,
    "fr3_joint3": 0.00,
    "fr3_joint4": -2.20,
    "fr3_joint5": 0.00,
    "fr3_joint6": 1.80,
    "fr3_joint7": 0.80,
}


def resolve_path(p):
    p = Path(str(p)).expanduser()
    if not p.is_absolute():
        p = PROJECT / p
    return p


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


def smoothstep(x):
    x = float(np.clip(x, 0.0, 1.0))
    return x * x * (3.0 - 2.0 * x)


def clamp_joint_q(model, joint_name, value):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return float(value)
    if bool(model.jnt_limited[jid]):
        lo, hi = model.jnt_range[jid]
        return float(np.clip(float(value), float(lo), float(hi)))
    return float(value)


def make_arm_pose(model, mode):
    if mode == "zero_raw":
        return {j: 0.0 for j in ARM_JOINTS}
    if mode == "zero_clamped":
        return {j: clamp_joint_q(model, j, 0.0) for j in ARM_JOINTS}
    if mode == "v12_start":
        return {j: clamp_joint_q(model, j, V12_START_ARM[j]) for j in ARM_JOINTS}
    raise RuntimeError(f"unknown arm pose mode: {mode}")


def dict_to_vec(q):
    return np.array([float(q[j]) for j in ARM_JOINTS], dtype=float)


def vec_to_dict(v):
    return {j: float(v[i]) for i, j in enumerate(ARM_JOINTS)}


def interp_vec(a, b, alpha, smooth=True):
    aa = smoothstep(alpha) if smooth else float(alpha)
    return (1.0 - aa) * np.asarray(a, dtype=float) + aa * np.asarray(b, dtype=float)


def interp_dict_local(a, b, alpha, smooth=True):
    return vec_to_dict(interp_vec(dict_to_vec(a), dict_to_vec(b), alpha, smooth=smooth))


def joint_bounds(model):
    lo = []
    hi = []
    for j in ARM_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if jid >= 0 and bool(model.jnt_limited[jid]):
            a, b = model.jnt_range[jid]
            lo.append(float(a))
            hi.append(float(b))
        else:
            lo.append(-np.pi)
            hi.append(np.pi)
    return np.array(lo, dtype=float), np.array(hi, dtype=float)


def path_len(path):
    if len(path) < 2:
        return 0.0
    return float(sum(np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1)))


def robot_geom_ids(model, object_geoms):
    ids = []
    tokens = [
        "fr3_",
        "thumb",
        "index",
        "middle",
        "ring",
        "pinky",
        "hand",
        "metacarpals",
    ]

    for gid in range(model.ngeom):
        if gid in object_geoms:
            continue

        if int(model.geom_contype[gid]) == 0 and int(model.geom_conaffinity[gid]) == 0:
            continue

        gname = p4u1.geom_name(model, gid).lower()
        bname = p4u1.body_name(model, int(model.geom_bodyid[gid])).lower()
        text = gname + " " + bname

        if any(tok in text for tok in tokens):
            ids.append(gid)

    return ids


class StaticCollisionChecker:
    def __init__(self, model, object_body, side_open_ctrl, min_clearance):
        self.model = model
        self.data = mujoco.MjData(model)

        mujoco.mj_resetData(self.model, self.data)
        self.data.qvel[:] = 0.0
        if self.model.nu > 0:
            self.data.ctrl[:] = 0.0
        mujoco.mj_forward(self.model, self.data)

        self.base_qpos = self.data.qpos.copy()
        self.base_qvel = self.data.qvel.copy()

        self.object_bid = p4u1.body_id(model, object_body)
        self.object_geoms = p4u1.geoms_of_body(model, self.object_bid)
        self.robot_geoms = robot_geom_ids(model, self.object_geoms)
        self.side_open_ctrl = dict(side_open_ctrl)
        self.min_clearance = float(min_clearance)
        self.check_count = 0

    def set_config(self, q_vec):
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.base_qpos
        self.data.qvel[:] = 0.0
        if self.model.nu > 0:
            self.data.ctrl[:] = 0.0

        p4u1.set_qpos_dict(self.model, self.data, vec_to_dict(q_vec))
        p4u1.set_qpos_dict(self.model, self.data, self.side_open_ctrl)

        mujoco.mj_forward(self.model, self.data)

    def has_contact(self):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)

            robot_obj = (
                (g1 in self.robot_geoms and g2 in self.object_geoms)
                or (g2 in self.robot_geoms and g1 in self.object_geoms)
            )
            if robot_obj:
                return True

        return False

    def min_dist(self):
        min_d = 1e9
        min_pair = None
        fromto = np.zeros(6, dtype=float)

        for rg in self.robot_geoms:
            for og in self.object_geoms:
                try:
                    d = float(mujoco.mj_geomDistance(self.model, self.data, int(rg), int(og), 0.25, fromto))
                except Exception:
                    continue

                if d < min_d:
                    min_d = d
                    min_pair = (
                        p4u1.geom_name(self.model, rg),
                        p4u1.geom_name(self.model, og),
                    )

        return min_d, min_pair

    def valid_config(self, q_vec, clearance=None):
        self.check_count += 1
        if clearance is None:
            clearance = self.min_clearance

        self.set_config(q_vec)

        if self.has_contact():
            return False, {
                "reason": "contact",
                "min_distance": -1.0,
                "pair": None,
            }

        d, pair = self.min_dist()
        if d < float(clearance):
            return False, {
                "reason": "clearance",
                "min_distance": d,
                "pair": pair,
            }

        return True, {
            "reason": "ok",
            "min_distance": d,
            "pair": pair,
        }

    def valid_edge(self, qa, qb, edge_step, clearance=None):
        qa = np.asarray(qa, dtype=float)
        qb = np.asarray(qb, dtype=float)

        dist = float(np.linalg.norm(qb - qa))
        n = max(2, int(np.ceil(dist / float(edge_step))) + 1)

        worst_d = 1e9
        worst_pair = None
        worst_alpha = 0.0

        for k in range(n):
            alpha = k / float(n - 1)
            q = interp_vec(qa, qb, alpha, smooth=False)
            ok, info = self.valid_config(q, clearance=clearance)

            d = float(info.get("min_distance", 1e9))
            if d < worst_d:
                worst_d = d
                worst_pair = info.get("pair")
                worst_alpha = alpha

            if not ok:
                return False, {
                    "reason": info.get("reason"),
                    "alpha": alpha,
                    "min_distance": d,
                    "pair": info.get("pair"),
                }

        return True, {
            "reason": "ok",
            "alpha": worst_alpha,
            "min_distance": worst_d,
            "pair": worst_pair,
        }


class Tree:
    def __init__(self, root):
        self.qs = [np.asarray(root, dtype=float)]
        self.parents = [-1]

    def nearest(self, q):
        q = np.asarray(q, dtype=float)
        ds = [float(np.linalg.norm(x - q)) for x in self.qs]
        return int(np.argmin(ds))

    def add(self, q, parent):
        self.qs.append(np.asarray(q, dtype=float))
        self.parents.append(int(parent))
        return len(self.qs) - 1

    def path_to_root(self, idx):
        out = []
        while idx >= 0:
            out.append(self.qs[idx])
            idx = self.parents[idx]
        return list(reversed(out))


def steer(qa, qb, step):
    qa = np.asarray(qa, dtype=float)
    qb = np.asarray(qb, dtype=float)
    d = qb - qa
    n = float(np.linalg.norm(d))

    if n < 1e-12:
        return qb.copy()
    if n <= float(step):
        return qb.copy()

    return qa + d / n * float(step)


def rrt_connect(start, goal, lo, hi, checker, args, seed):
    rng = np.random.default_rng(seed)

    tree_a = Tree(start)
    tree_b = Tree(goal)
    a_is_start = True

    def grow(tree, target):
        idx = tree.nearest(target)
        q_near = tree.qs[idx]
        q_new = steer(q_near, target, args.rrt_step)

        ok, info = checker.valid_edge(
            q_near,
            q_new,
            edge_step=args.edge_step,
            clearance=args.approach_min_clearance,
        )

        if not ok:
            return False, idx, info

        new_idx = tree.add(q_new, idx)
        return True, new_idx, info

    def connect(tree, target):
        last_idx = None
        last_info = None

        for _ in range(1000):
            idx = tree.nearest(target)
            q_near = tree.qs[idx]
            q_new = steer(q_near, target, args.rrt_step)

            ok, info = checker.valid_edge(
                q_near,
                q_new,
                edge_step=args.edge_step,
                clearance=args.approach_min_clearance,
            )
            last_info = info

            if not ok:
                return False, last_idx, last_info

            new_idx = tree.add(q_new, idx)
            last_idx = new_idx

            if np.linalg.norm(q_new - target) < 1e-8:
                return True, new_idx, info

            if np.linalg.norm(q_new - target) <= args.rrt_step:
                ok2, info2 = checker.valid_edge(
                    q_new,
                    target,
                    edge_step=args.edge_step,
                    clearance=args.approach_min_clearance,
                )
                if ok2:
                    target_idx = tree.add(target, new_idx)
                    return True, target_idx, info2
                return False, new_idx, info2

        return False, last_idx, last_info

    for it in range(int(args.rrt_max_iters)):
        if rng.random() < float(args.goal_bias):
            q_rand = goal.copy()
        else:
            q_rand = rng.uniform(lo, hi)

        ok, idx_new, info = grow(tree_a, q_rand)

        if ok:
            q_new = tree_a.qs[idx_new]
            reached, idx_other, info_other = connect(tree_b, q_new)

            if reached:
                path_a = tree_a.path_to_root(idx_new)
                path_b = tree_b.path_to_root(idx_other)

                if a_is_start:
                    full = path_a + list(reversed(path_b))[1:]
                else:
                    full = path_b + list(reversed(path_a))[1:]

                return {
                    "success": True,
                    "iterations": it + 1,
                    "path": full,
                    "path_length": path_len(full),
                    "checks": checker.check_count,
                }

        tree_a, tree_b = tree_b, tree_a
        a_is_start = not a_is_start

    return {
        "success": False,
        "iterations": int(args.rrt_max_iters),
        "path": [],
        "path_length": None,
        "checks": checker.check_count,
    }


def shortcut_path(path, checker, args, clearance):
    if len(path) <= 2:
        return path

    path = [np.asarray(x, dtype=float).copy() for x in path]
    rng = np.random.default_rng(int(args.seed) + 12345)

    for _ in range(int(args.shortcut_iters)):
        if len(path) <= 2:
            break

        i = int(rng.integers(0, len(path) - 2))
        j = int(rng.integers(i + 2, len(path)))

        ok, _ = checker.valid_edge(
            path[i],
            path[j],
            edge_step=args.edge_step,
            clearance=clearance,
        )

        if ok:
            path = path[: i + 1] + path[j:]

    changed = True
    while changed:
        changed = False
        new_path = [path[0]]
        i = 0

        while i < len(path) - 1:
            best_j = i + 1

            for j in range(len(path) - 1, i + 1, -1):
                ok, _ = checker.valid_edge(
                    path[i],
                    path[j],
                    edge_step=args.edge_step,
                    clearance=clearance,
                )
                if ok:
                    best_j = j
                    break

            if best_j > i + 1:
                changed = True

            new_path.append(path[best_j])
            i = best_j

        path = new_path

    return path


def plan_segment(name, q_a_dict, q_b_dict, checker, lo, hi, args, clearance):
    print(f"\n========== PLAN SEGMENT: {name} ==========")

    qa = dict_to_vec(q_a_dict)
    qb = dict_to_vec(q_b_dict)

    ok_a, info_a = checker.valid_config(qa, clearance=clearance)
    ok_b, info_b = checker.valid_config(qb, clearance=clearance)

    print("start valid:", ok_a, info_a)
    print("goal  valid:", ok_b, info_b)

    if not ok_a:
        raise RuntimeError(f"{name}: start invalid: {info_a}")
    if not ok_b:
        raise RuntimeError(f"{name}: goal invalid: {info_b}")

    ok_direct, direct_info = checker.valid_edge(
        qa,
        qb,
        edge_step=args.edge_step,
        clearance=clearance,
    )

    print("direct edge:", ok_direct, direct_info)

    if ok_direct:
        path = [qa, qb]
        return {
            "name": name,
            "success": True,
            "method": "direct",
            "path": path,
            "path_length": path_len(path),
            "direct_info": direct_info,
            "num_waypoints": len(path),
        }

    best = None

    for attempt in range(int(args.plan_attempts)):
        seed = int(args.seed + 101 * attempt)
        checks_before = checker.check_count

        r = rrt_connect(
            start=qa,
            goal=qb,
            lo=lo,
            hi=hi,
            checker=checker,
            args=args,
            seed=seed,
        )

        r["attempt"] = attempt
        r["seed"] = seed
        r["checks_this_attempt"] = checker.check_count - checks_before

        if not r["success"]:
            print(f"[MISS] attempt={attempt} seed={seed} iters={r['iterations']}")
            continue

        raw_len = r["path_length"]
        short = shortcut_path(r["path"], checker, args, clearance=clearance)
        short_len = path_len(short)

        r["raw_path_length"] = raw_len
        r["path"] = short
        r["path_length"] = short_len
        r["num_waypoints"] = len(short)

        print(
            f"[FOUND] attempt={attempt} seed={seed} "
            f"raw_len={raw_len:.4f} short_len={short_len:.4f} waypoints={len(short)}"
        )

        if best is None or short_len < best["path_length"]:
            best = r

    if best is None:
        return {
            "name": name,
            "success": False,
            "method": "rrt_connect",
            "path": [],
            "path_length": None,
            "direct_info": direct_info,
            "num_waypoints": 0,
        }

    best["name"] = name
    best["method"] = "rrt_connect_shortcut"
    best["direct_info"] = direct_info
    return best


def execute_joint_path(runner, path, label, hand_ctrl, args):
    if len(path) < 2:
        return True

    for si in range(len(path) - 1):
        qa = np.asarray(path[si], dtype=float)
        qb = np.asarray(path[si + 1], dtype=float)
        seg_len = float(np.linalg.norm(qb - qa))

        duration = max(
            float(args.min_segment_duration),
            seg_len / max(float(args.joint_speed_rad_s), 1e-6),
        )
        steps = max(1, int(duration / float(runner.model.opt.timestep)))

        print(
            f"\n[PATH EXEC] {label} seg={si + 1}/{len(path) - 1} "
            f"len={seg_len:.4f} duration={duration:.2f}s steps={steps}"
        )

        for k in range(steps + 1):
            alpha = k / float(steps)
            qv = interp_vec(qa, qb, alpha, smooth=True)
            qdict = vec_to_dict(qv)

            if args.hard_servo_approach:
                p4u1.set_qpos_dict(runner.model, runner.data, qdict)
                p4u1.set_qpos_dict(runner.model, runner.data, hand_ctrl)
                runner.data.qvel[:] = 0.0
                mujoco.mj_forward(runner.model, runner.data)

            runner.step_once(
                phase=f"{label}_seg{si:02d}",
                step=k,
                alpha=alpha,
                arm_ctrl=qdict,
                hand_ctrl=hand_ctrl,
            )

            if runner.object_disp() > float(args.approach_abort_disp):
                print(
                    f"[ABORT] approach moved object too much: "
                    f"{runner.object_disp():.5f} > {args.approach_abort_disp:.5f}"
                )
                return False

    return True


def make_result(args, model_path, runner, q_start, q_home, q_pre, q_grasp, q_lift,
                arm_source, hand_source, side_open_ctrl, close_target,
                grip_ready, stop_reason, grip_hold_ctrl, lift_ik_info,
                plan_result, object_geoms, model, data):
    final_live = p4u1.collect_live_contact(model, data, object_geoms, args)

    return {
        "format": "v4_12p4u6_ik_path_record_demo",
        "model": str(model_path),
        "args": vars(args),
        "q_start": q_start,
        "q_home": q_home,
        "q_pre": q_pre,
        "q_grasp": q_grasp,
        "q_lift": q_lift,
        "arm_source": arm_source,
        "hand_source": hand_source,
        "side_open_ctrl": side_open_ctrl,
        "close_target": close_target,
        "grip_ready": bool(grip_ready),
        "stop_reason": stop_reason,
        "max_stable_count": runner.max_stable_count,
        "final_object_disp": runner.object_disp(),
        "final_object_rise": runner.object_rise(),
        "final_groups": final_live["groups"],
        "final_opposition_cos": final_live["opposition_cos"],
        "grip_hold_ctrl": grip_hold_ctrl,
        "lift_ik_info": lift_ik_info,
        "plan_result": plan_result,
        "rows": runner.rows,
    }


def print_result(out_path, result):
    print("\n========== P4U6 IK-PATH RECORD RESULT ==========")
    print("out                 :", resolve_path(out_path))
    print("grip_ready          :", result.get("grip_ready"))
    print("stop_reason         :", result.get("stop_reason"))
    print("max_stable_count    :", result.get("max_stable_count"))
    print("final_object_disp   :", result.get("final_object_disp"))
    print("final_object_rise   :", result.get("final_object_rise"))
    print("final_groups        :", result.get("final_groups"))
    print("final_opposition_cos:", result.get("final_opposition_cos"))

    plan = result.get("plan_result", {})
    for name, seg in plan.get("segments", {}).items():
        print(
            f"plan {name:14s}: success={seg.get('success')} "
            f"method={seg.get('method')} len={seg.get('path_length')} "
            f"waypoints={len(seg.get('path_dicts', []))}"
        )

    print("================================================\n")


def run(args):
    model_path = resolve_path(args.model)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    ctrl_map = p4u1.build_ctrl_map(model)

    candidate = p4u1.load_json(args.candidate)
    p3 = p4u1.load_json(args.p3_json)
    best_config = p4u1.load_json(args.best_config)

    q_pre, q_grasp, arm_source = p4u1.extract_arm_plan(p3, candidate)
    q_start = make_arm_pose(model, args.start_arm_mode)
    q_home = make_arm_pose(model, "v12_start")

    hand_prior, hand_source = p4u1.extract_hand_ctrl(best_config, candidate)
    side_open_ctrl, close_target = p4u1.make_side_open_and_close(
        hand_prior,
        finger_scale=args.finger_close_scale,
        thumb_gain=args.thumb_pitch_from_finger_gain,
        thumb_open_pitch=args.thumb_open_pitch,
    )

    obj_bid = p4u1.body_id(model, args.object_body)
    object_geoms = p4u1.geoms_of_body(model, obj_bid)

    checker = StaticCollisionChecker(
        model=model,
        object_body=args.object_body,
        side_open_ctrl=side_open_ctrl,
        min_clearance=args.approach_min_clearance,
    )

    lo, hi = joint_bounds(model)

    print("\n========== V4.12P4U6 IK-PATH PLANNING ==========")
    print("model                 :", model_path)
    print("object_body           :", args.object_body)
    print("target_body           :", args.target_body)
    print("q_start               :", q_start)
    print("q_home                :", q_home)
    print("q_pre                 :", q_pre)
    print("q_grasp               :", q_grasp)
    print("arm_source            :", arm_source)
    print("hand_source           :", hand_source)
    print("approach_min_clearance:", args.approach_min_clearance)
    print("grasp_path_clearance  :", args.grasp_path_min_clearance)
    print("hard_servo_approach   :", args.hard_servo_approach)
    print("================================================\n")

    seg_start_home = plan_segment(
        name="start_to_home",
        q_a_dict=q_start,
        q_b_dict=q_home,
        checker=checker,
        lo=lo,
        hi=hi,
        args=args,
        clearance=args.approach_min_clearance,
    )

    if not seg_start_home["success"]:
        raise RuntimeError("failed to find path: start_to_home")

    seg_home_pre = plan_segment(
        name="home_to_pre",
        q_a_dict=q_home,
        q_b_dict=q_pre,
        checker=checker,
        lo=lo,
        hi=hi,
        args=args,
        clearance=args.approach_min_clearance,
    )

    if not seg_home_pre["success"]:
        raise RuntimeError("failed to find path: home_to_pre")

    seg_pre_grasp = plan_segment(
        name="pre_to_grasp",
        q_a_dict=q_pre,
        q_b_dict=q_grasp,
        checker=checker,
        lo=lo,
        hi=hi,
        args=args,
        clearance=args.grasp_path_min_clearance,
    )

    if not seg_pre_grasp["success"]:
        raise RuntimeError("failed to find path: pre_to_grasp")

    plan_result = {
        "format": "v4_12p4u6_ik_path_plan",
        "segments": {
            "start_to_home": {
                **{k: v for k, v in seg_start_home.items() if k != "path"},
                "path_dicts": [vec_to_dict(q) for q in seg_start_home["path"]],
            },
            "home_to_pre": {
                **{k: v for k, v in seg_home_pre.items() if k != "path"},
                "path_dicts": [vec_to_dict(q) for q in seg_home_pre["path"]],
            },
            "pre_to_grasp": {
                **{k: v for k, v in seg_pre_grasp.items() if k != "path"},
                "path_dicts": [vec_to_dict(q) for q in seg_pre_grasp["path"]],
            },
        },
        "checker_total_checks": checker.check_count,
    }

    save_json(args.plan_out, plan_result)

    q_lift = dict(q_grasp)
    lift_ik_info = None

    mujoco.mj_resetData(model, data)
    p4u1.set_qpos_dict(model, data, q_grasp)
    p4u1.set_qpos_dict(model, data, side_open_ctrl)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    if args.enable_lift:
        q_lift, lift_ik_info = p4u1.solve_world_z_lift_qpos_dict(
            model=model,
            data=data,
            q_seed=q_grasp,
            target_body_name=args.target_body,
            lift_z=args.lift_z,
            ik_iters=args.lift_ik_iters,
            damping=args.lift_ik_damping,
        )

    mujoco.mj_resetData(model, data)
    p4u1.set_qpos_dict(model, data, q_start)
    p4u1.set_qpos_dict(model, data, side_open_ctrl)
    data.qvel[:] = 0.0
    mujoco.mj_forward(model, data)

    runner = p4u1.Runner(model, data, ctrl_map, obj_bid, object_geoms, args)

    def execute(viewer=None):
        if viewer is not None:
            runner.attach_viewer(viewer)

        dt = float(model.opt.timestep)

        start_hold_steps = max(0, int(float(args.start_hold_duration) / dt))
        home_hold_steps = max(0, int(float(args.home_hold_duration) / dt))
        pre_hold_steps = max(0, int(float(args.pre_hold_duration) / dt))
        grasp_settle_steps = max(0, int(float(args.grasp_settle_duration) / dt))
        close_steps = max(1, int(float(args.close_duration) / dt))
        post_hold_steps = max(0, int(float(args.post_close_target_hold_duration) / dt))
        micro_steps = max(0, int(float(args.micro_squeeze_duration) / dt))
        lift_steps = max(1, int(float(args.lift_duration) / dt))
        final_hold_steps = max(0, int(float(args.final_hold_duration) / dt))

        runner.run_hold(
            "record_hold_at_start_side_open",
            start_hold_steps,
            q_start,
            side_open_ctrl,
        )

        ok = execute_joint_path(
            runner,
            seg_start_home["path"],
            "record_path_start_to_home",
            side_open_ctrl,
            args,
        )
        if not ok:
            result = make_result(args, model_path, runner, q_start, q_home, q_pre, q_grasp, q_lift,
                                 arm_source, hand_source, side_open_ctrl, close_target,
                                 False, "abort_start_to_home_moved_object", dict(close_target),
                                 lift_ik_info, plan_result, object_geoms, model, data)
            save_json(args.out, result)
            print_result(args.out, result)
            return result

        runner.run_hold(
            "record_hold_at_v12_safe_home",
            home_hold_steps,
            q_home,
            side_open_ctrl,
        )

        ok = execute_joint_path(
            runner,
            seg_home_pre["path"],
            "record_path_home_to_pre",
            side_open_ctrl,
            args,
        )
        if not ok:
            result = make_result(args, model_path, runner, q_start, q_home, q_pre, q_grasp, q_lift,
                                 arm_source, hand_source, side_open_ctrl, close_target,
                                 False, "abort_home_to_pre_moved_object", dict(close_target),
                                 lift_ik_info, plan_result, object_geoms, model, data)
            save_json(args.out, result)
            print_result(args.out, result)
            return result

        runner.run_hold(
            "record_hold_at_p3_pre_side_open",
            pre_hold_steps,
            q_pre,
            side_open_ctrl,
        )

        ok = execute_joint_path(
            runner,
            seg_pre_grasp["path"],
            "record_path_pre_to_grasp",
            side_open_ctrl,
            args,
        )
        if not ok:
            result = make_result(args, model_path, runner, q_start, q_home, q_pre, q_grasp, q_lift,
                                 arm_source, hand_source, side_open_ctrl, close_target,
                                 False, "abort_pre_to_grasp_moved_object", dict(close_target),
                                 lift_ik_info, plan_result, object_geoms, model, data)
            save_json(args.out, result)
            print_result(args.out, result)
            return result

        runner.run_hold(
            "settle_side_open_at_grasp",
            grasp_settle_steps,
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
                frac = float(args.micro_squeeze_fraction) * smoothstep(alpha)

                hand = {}
                for j in O7_ACTIVE_JOINTS:
                    hand[j] = float(close_target.get(j, 0.0)) + frac * squeeze_dir.get(j, 0.0)

                grip_hold_ctrl = dict(hand)
                runner.step_once("gated_micro_squeeze", k, alpha, q_grasp, hand)

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
                    print(f"[NO GRIP] extra object disp exceeded: {disp - start_disp:.5f} > {args.max_extra_disp_during_squeeze:.5f}")
                    break

        if not grip_ready:
            print("\n[NO_LIFT] grip is not ready. Do not lift an ungrasped object.")
            result = make_result(args, model_path, runner, q_start, q_home, q_pre, q_grasp, q_lift,
                                 arm_source, hand_source, side_open_ctrl, close_target,
                                 False, stop_reason, grip_hold_ctrl,
                                 lift_ik_info, plan_result, object_geoms, model, data)
            save_json(args.out, result)
            print_result(args.out, result)
            return result

        print("\n[GRIP_READY] lift is allowed. Hand ctrl will stay constant during lift.")

        if args.enable_lift:
            runner.run_phase(
                "lift_world_z_with_fixed_grip_ctrl",
                lift_steps,
                q_grasp,
                q_lift,
                grip_hold_ctrl,
                grip_hold_ctrl,
            )

            if final_hold_steps > 0:
                runner.run_hold(
                    "final_air_hold_after_lift",
                    final_hold_steps,
                    q_lift,
                    grip_hold_ctrl,
                )

        result = make_result(args, model_path, runner, q_start, q_home, q_pre, q_grasp, q_lift,
                             arm_source, hand_source, side_open_ctrl, close_target,
                             True, stop_reason, grip_hold_ctrl,
                             lift_ik_info, plan_result, object_geoms, model, data)
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

    return execute(None)


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
    ap.add_argument("--plan-out", default="diagnostics/current_v412/v4_12p4u6_ik_path_plan_debug.json")

    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--keep-viewer-open", action="store_true")
    ap.add_argument("--frame-sleep", type=float, default=0.0015)

    ap.add_argument("--start-arm-mode", choices=["zero_clamped", "zero_raw", "v12_start"], default="zero_clamped")

    ap.add_argument("--start-hold-duration", type=float, default=1.2)
    ap.add_argument("--home-hold-duration", type=float, default=0.6)
    ap.add_argument("--pre-hold-duration", type=float, default=0.8)
    ap.add_argument("--grasp-settle-duration", type=float, default=0.35)

    ap.add_argument("--close-duration", type=float, default=0.45)
    ap.add_argument("--post-close-target-hold-duration", type=float, default=0.25)
    ap.add_argument("--micro-squeeze-duration", type=float, default=0.35)
    ap.add_argument("--micro-squeeze-fraction", type=float, default=0.08)

    ap.add_argument("--enable-lift", action="store_true")
    ap.add_argument("--lift-z", type=float, default=0.060)
    ap.add_argument("--lift-duration", type=float, default=3.0)
    ap.add_argument("--final-hold-duration", type=float, default=1.0)
    ap.add_argument("--lift-ik-iters", type=int, default=140)
    ap.add_argument("--lift-ik-damping", type=float, default=1e-4)

    ap.add_argument("--finger-close-scale", type=float, default=0.92)
    ap.add_argument("--thumb-pitch-from-finger-gain", type=float, default=0.24)
    ap.add_argument("--thumb-open-pitch", type=float, default=0.22)

    ap.add_argument("--min-live-non-thumb", type=int, default=1)
    ap.add_argument("--opposition-cos-threshold", type=float, default=-0.30)
    ap.add_argument("--grip-ready-stable-steps", type=int, default=8)

    ap.add_argument("--max-grip-disp", type=float, default=0.006)
    ap.add_argument("--max-extra-disp-during-squeeze", type=float, default=0.003)
    ap.add_argument("--approach-abort-disp", type=float, default=0.015)

    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--plan-attempts", type=int, default=10)
    ap.add_argument("--rrt-max-iters", type=int, default=4000)
    ap.add_argument("--rrt-step", type=float, default=0.28)
    ap.add_argument("--edge-step", type=float, default=0.035)
    ap.add_argument("--goal-bias", type=float, default=0.20)
    ap.add_argument("--shortcut-iters", type=int, default=400)
    ap.add_argument("--approach-min-clearance", type=float, default=0.003)
    ap.add_argument("--grasp-path-min-clearance", type=float, default=0.001)

    ap.add_argument("--joint-speed-rad-s", type=float, default=0.75)
    ap.add_argument("--min-segment-duration", type=float, default=0.35)
    ap.add_argument("--hard-servo-approach", action="store_true", default=True)

    ap.add_argument("--print-every-steps", type=int, default=100)
    ap.add_argument("--log-every-steps", type=int, default=100)

    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
