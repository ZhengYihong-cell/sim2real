#!/usr/bin/env python3
"""
脚本类型：
    debug / runner / collision-aware-recording-demo

用途：
    V4.12P4U5。
    为录屏 demo 搜索一条从安全 home 到 q_pre/q_grasp 的无碰路径，
    然后接入已经验证成功的 P4U1 ready-gated snap close + micro-squeeze + lift。

输入：
    --model
        已验证成功的 FR3+O7+can MuJoCo XML。
    --candidate
        当前成功候选 best_candidate.json。
    --p3-json
        当前成功 P3 q_pre / q_grasp。
    --best-config
        当前成功 O7 手型参数。
    --out
        输出本次运行 JSON 日志。
    --plan-out
        输出搜索得到的 approach path JSON。

输出：
    1. 一条 collision-aware approach path；
    2. viewer 中完整录屏流程；
    3. JSON 日志，包含路径搜索结果、抓握 ready gate 和 lift 结果。

当前流程位置：
    用于录屏展示，不修改 legacy_final_demos，不重新搜索抓握候选，不重新改抓握逻辑。

核心流程：
    1. zero_clamped 开场展示；
    2. zero_clamped -> v12_safe_home 直接插值；
    3. v12_safe_home -> q_pre 使用 RRT 找无碰路径；
    4. q_pre -> q_grasp 若直接路径不安全，也使用 RRT；
    5. q_grasp 后沿用 P4U1 成功抓握逻辑；
    6. grip_ready 后 fixed-grip world-z lift。

不负责：
    1. 不重新生成 grasp；
    2. 不使用 auto_radial；
    3. 不在未抓紧时强行 lift；
    4. 不在 lift 阶段继续改变手型。
"""

from pathlib import Path
import argparse
import json
import time
import random
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


def save_json(path, obj):
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        json.dump(p4u1.to_jsonable(obj), f, indent=2)


def clamp_joint_q(model, joint_name, value):
    jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if jid < 0:
        return float(value)
    if bool(model.jnt_limited[jid]):
        lo, hi = model.jnt_range[jid]
        return float(np.clip(float(value), float(lo), float(hi)))
    return float(value)


def make_start_arm(model, mode):
    if mode == "zero_raw":
        return {j: 0.0 for j in ARM_JOINTS}
    if mode == "zero_clamped":
        return {j: clamp_joint_q(model, j, 0.0) for j in ARM_JOINTS}
    if mode == "v12_start":
        return {j: clamp_joint_q(model, j, V12_START_ARM[j]) for j in ARM_JOINTS}
    raise RuntimeError(f"unknown start-arm-mode: {mode}")


def dict_to_vec(q):
    return np.array([float(q[j]) for j in ARM_JOINTS], dtype=float)


def vec_to_dict(v):
    return {j: float(v[i]) for i, j in enumerate(ARM_JOINTS)}


def joint_bounds(model):
    lows = []
    highs = []
    for j in ARM_JOINTS:
        jid = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, j)
        if jid >= 0 and bool(model.jnt_limited[jid]):
            lo, hi = model.jnt_range[jid]
            lows.append(float(lo))
            highs.append(float(hi))
        else:
            lows.append(-np.pi)
            highs.append(np.pi)
    return np.array(lows, dtype=float), np.array(highs, dtype=float)


def path_length(path):
    if len(path) <= 1:
        return 0.0
    return float(sum(np.linalg.norm(path[i + 1] - path[i]) for i in range(len(path) - 1)))


def robot_geom_ids(model, object_geoms):
    ids = []
    robot_tokens = [
        "fr3_",
        "thumb", "index", "middle", "ring", "pinky",
        "hand", "metacarpals",
    ]

    for gid in range(model.ngeom):
        if gid in object_geoms:
            continue

        # 只检查实际参与碰撞的 geom；visual / disabled collision 不参与路径约束。
        if int(model.geom_contype[gid]) == 0 and int(model.geom_conaffinity[gid]) == 0:
            continue

        gname = p4u1.geom_name(model, gid).lower()
        bname = p4u1.body_name(model, int(model.geom_bodyid[gid])).lower()
        text = gname + " " + bname

        if any(tok in text for tok in robot_tokens):
            ids.append(gid)

    return ids


class CollisionChecker:
    def __init__(self, model, object_body, side_open_ctrl, min_clearance):
        self.model = model
        self.data = mujoco.MjData(model)

        # 关键修正：
        # 不允许在静态碰撞检查里 data.qpos[:] = 0。
        # 物体是 freejoint，qpos 清零会把物体位置/四元数也清掉，
        # 导致 checker 在错误世界状态下误报 robot-object contact。
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

    def set_static_q(self, q_vec):
        # 关键修正：
        # 每次检查都恢复 XML 默认场景状态，尤其保留 can 的 freejoint pose。
        # 然后只覆盖 FR3 arm qpos 和 O7 side-open qpos。
        mujoco.mj_resetData(self.model, self.data)
        self.data.qpos[:] = self.base_qpos
        self.data.qvel[:] = 0.0
        if self.model.nu > 0:
            self.data.ctrl[:] = 0.0

        qdict = vec_to_dict(q_vec)
        p4u1.set_qpos_dict(self.model, self.data, qdict)
        p4u1.set_qpos_dict(self.model, self.data, self.side_open_ctrl)

        mujoco.mj_forward(self.model, self.data)

    def min_robot_object_distance(self):
        min_d = 1e9
        min_pair = None
        fromto = np.zeros(6, dtype=float)

        for rg in self.robot_geoms:
            for og in self.object_geoms:
                try:
                    d = float(mujoco.mj_geomDistance(self.model, self.data, int(rg), int(og), 0.20, fromto))
                except Exception:
                    continue

                if d < min_d:
                    min_d = d
                    min_pair = (
                        p4u1.geom_name(self.model, rg),
                        p4u1.geom_name(self.model, og),
                    )

        return min_d, min_pair

    def has_robot_object_contact(self):
        for i in range(self.data.ncon):
            c = self.data.contact[i]
            g1 = int(c.geom1)
            g2 = int(c.geom2)
            if (g1 in self.object_geoms and g2 in self.robot_geoms) or (g2 in self.object_geoms and g1 in self.robot_geoms):
                return True
        return False

    def config_valid(self, q_vec, clearance=None):
        self.check_count += 1
        if clearance is None:
            clearance = self.min_clearance

        self.set_static_q(q_vec)

        if self.has_robot_object_contact():
            return False, {"reason": "contact", "min_distance": -1.0, "pair": None}

        min_d, pair = self.min_robot_object_distance()
        if min_d < float(clearance):
            return False, {"reason": "clearance", "min_distance": min_d, "pair": pair}

        return True, {"reason": "ok", "min_distance": min_d, "pair": pair}

    def edge_valid(self, qa, qb, edge_step, clearance=None):
        dist = float(np.linalg.norm(qb - qa))
        n = max(2, int(np.ceil(dist / float(edge_step))) + 1)

        worst_d = 1e9
        worst_pair = None
        worst_alpha = 0.0

        for k in range(n):
            alpha = k / float(n - 1)
            q = (1.0 - alpha) * qa + alpha * qb
            ok, info = self.config_valid(q, clearance=clearance)

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
            "min_distance": worst_d,
            "pair": worst_pair,
            "alpha": worst_alpha,
        }


class Tree:
    def __init__(self, root_q, root_label):
        self.qs = [np.asarray(root_q, dtype=float)]
        self.parents = [-1]
        self.root_label = root_label

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
    d = qb - qa
    n = float(np.linalg.norm(d))
    if n < 1e-12:
        return qb.copy()
    if n <= step:
        return qb.copy()
    return qa + d / n * float(step)


def rrt_connect_once(start, goal, bounds_lo, bounds_hi, checker, args, seed):
    rng = np.random.default_rng(seed)
    random.seed(seed)

    tree_a = Tree(start, "start")
    tree_b = Tree(goal, "goal")

    def grow(tree, target):
        idx = tree.nearest(target)
        q_near = tree.qs[idx]
        q_new = steer(q_near, target, args.rrt_step)

        ok, info = checker.edge_valid(
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

            ok, info = checker.edge_valid(
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

            if np.linalg.norm(q_new - target) < 1e-6:
                return True, new_idx, info

            if np.linalg.norm(q_new - target) <= args.rrt_step:
                ok2, info2 = checker.edge_valid(
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

    swapped = False

    for it in range(int(args.rrt_max_iters)):
        if rng.random() < args.goal_bias:
            q_rand = goal.copy()
        else:
            q_rand = rng.uniform(bounds_lo, bounds_hi)

        ok, idx_new, info = grow(tree_a, q_rand)
        if ok:
            q_new = tree_a.qs[idx_new]
            reached, idx_b, info_b = connect(tree_b, q_new)

            if reached:
                path_a = tree_a.path_to_root(idx_new)
                path_b = tree_b.path_to_root(idx_b)

                if tree_a.root_label == "start":
                    path_start_to_mid = path_a
                    path_goal_to_mid = path_b
                else:
                    path_start_to_mid = path_b
                    path_goal_to_mid = path_a

                full = path_start_to_mid + list(reversed(path_goal_to_mid))[1:]

                return {
                    "success": True,
                    "iterations": it + 1,
                    "path": full,
                    "path_length": path_length(full),
                    "checks": checker.check_count,
                }

        tree_a, tree_b = tree_b, tree_a
        swapped = not swapped

    return {
        "success": False,
        "iterations": int(args.rrt_max_iters),
        "path": [],
        "path_length": None,
        "checks": checker.check_count,
    }


def shortcut_path(path, checker, args):
    if len(path) <= 2:
        return path

    path = [p.copy() for p in path]
    rng = np.random.default_rng(args.seed + 999)

    for _ in range(int(args.shortcut_iters)):
        if len(path) <= 2:
            break

        i = int(rng.integers(0, len(path) - 2))
        j = int(rng.integers(i + 2, len(path)))

        ok, _ = checker.edge_valid(
            path[i],
            path[j],
            edge_step=args.edge_step,
            clearance=args.approach_min_clearance,
        )

        if ok:
            path = path[: i + 1] + path[j:]

    # 再做一轮贪心最远连接
    improved = True
    while improved:
        improved = False
        i = 0
        new_path = [path[0]]

        while i < len(path) - 1:
            best_j = i + 1
            for j in range(len(path) - 1, i + 1, -1):
                ok, _ = checker.edge_valid(
                    path[i],
                    path[j],
                    edge_step=args.edge_step,
                    clearance=args.approach_min_clearance,
                )
                if ok:
                    best_j = j
                    break

            new_path.append(path[best_j])
            if best_j > i + 1:
                improved = True
            i = best_j

        path = new_path

    return path


def plan_segment(name, q_a, q_b, checker, bounds_lo, bounds_hi, args, clearance):
    print(f"\n========== PLAN SEGMENT: {name} ==========")

    va = dict_to_vec(q_a)
    vb = dict_to_vec(q_b)

    ok_a, info_a = checker.config_valid(va, clearance=clearance)
    ok_b, info_b = checker.config_valid(vb, clearance=clearance)

    print("start valid:", ok_a, info_a)
    print("goal  valid:", ok_b, info_b)

    if not ok_a:
        raise RuntimeError(f"{name}: start config invalid: {info_a}")
    if not ok_b:
        raise RuntimeError(f"{name}: goal config invalid: {info_b}")

    ok_direct, info_direct = checker.edge_valid(
        va,
        vb,
        edge_step=args.edge_step,
        clearance=clearance,
    )
    print("direct edge:", ok_direct, info_direct)

    if ok_direct:
        path = [va, vb]
        return {
            "name": name,
            "success": True,
            "method": "direct",
            "path": path,
            "path_length": path_length(path),
            "direct_info": info_direct,
        }

    best = None

    for attempt in range(int(args.plan_attempts)):
        seed = int(args.seed + attempt * 101)
        before_checks = checker.check_count

        r = rrt_connect_once(
            start=va,
            goal=vb,
            bounds_lo=bounds_lo,
            bounds_hi=bounds_hi,
            checker=checker,
            args=args,
            seed=seed,
        )

        r["attempt"] = attempt
        r["seed"] = seed
        r["checks_this_attempt"] = checker.check_count - before_checks

        if r["success"]:
            raw_len = r["path_length"]
            short_path = shortcut_path(r["path"], checker, args)
            short_len = path_length(short_path)

            r["raw_path_length"] = raw_len
            r["path"] = short_path
            r["path_length"] = short_len
            r["num_waypoints"] = len(short_path)

            print(
                f"[FOUND] attempt={attempt} seed={seed} "
                f"raw_len={raw_len:.4f} short_len={short_len:.4f} waypoints={len(short_path)}"
            )

            if best is None or short_len < best["path_length"]:
                best = r
        else:
            print(f"[MISS] attempt={attempt} seed={seed} iters={r['iterations']}")

    if best is None:
        return {
            "name": name,
            "success": False,
            "method": "rrt_connect",
            "path": [],
            "path_length": None,
            "direct_info": info_direct,
        }

    best["name"] = name
    best["method"] = "rrt_connect_shortcut"
    return best


def execute_path(runner, path, label, hand_ctrl, args):
    if len(path) < 2:
        return True

    for i in range(len(path) - 1):
        qa = vec_to_dict(path[i])
        qb = vec_to_dict(path[i + 1])
        seg_len = float(np.linalg.norm(path[i + 1] - path[i]))
        duration = max(float(args.min_segment_duration), seg_len / max(float(args.joint_speed_rad_s), 1e-6))
        steps = max(1, int(duration / float(runner.model.opt.timestep)))

        print(f"\n[PATH EXEC] {label} segment {i+1}/{len(path)-1}, len={seg_len:.4f}, duration={duration:.2f}s")

        for k in range(steps + 1):
            alpha = k / float(steps)
            a = p4u1.smoothstep(alpha)
            q = p4u1.interp_dict(qa, qb, a)
            runner.step_once(f"{label}_seg{i:02d}", k, alpha, q, hand_ctrl)

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
    result = {
        "format": "v4_12p4u5_collision_aware_record_demo",
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
    save_json(args.out, result)
    return result


def print_result(out_path, result):
    print("\n========== P4U5 COLLISION-AWARE RECORD RESULT ==========")
    print("out                 :", resolve_path(out_path))
    print("grip_ready          :", result.get("grip_ready"))
    print("stop_reason         :", result.get("stop_reason"))
    print("max_stable_count    :", result.get("max_stable_count"))
    print("final_object_disp   :", result.get("final_object_disp"))
    print("final_object_rise   :", result.get("final_object_rise"))
    print("final_groups        :", result.get("final_groups"))
    print("final_opposition_cos:", result.get("final_opposition_cos"))

    plan = result.get("plan_result", {})
    for k, v in plan.get("segments", {}).items():
        print(f"plan {k:14s}: success={v.get('success')} method={v.get('method')} len={v.get('path_length')} waypoints={len(v.get('path_dicts', []))}")

    print("========================================================\n")


def run(args):
    model_path = resolve_path(args.model)
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    ctrl_map = p4u1.build_ctrl_map(model)

    candidate = p4u1.load_json(args.candidate)
    p3 = p4u1.load_json(args.p3_json)
    best_config = p4u1.load_json(args.best_config)

    q_pre, q_grasp, arm_source = p4u1.extract_arm_plan(p3, candidate)
    q_start = make_start_arm(model, args.start_arm_mode)
    q_home = make_start_arm(model, "v12_start")

    hand_prior, hand_source = p4u1.extract_hand_ctrl(best_config, candidate)
    side_open_ctrl, close_target = p4u1.make_side_open_and_close(
        hand_prior,
        finger_scale=args.finger_close_scale,
        thumb_gain=args.thumb_pitch_from_finger_gain,
        thumb_open_pitch=args.thumb_open_pitch,
    )

    obj_bid = p4u1.body_id(model, args.object_body)
    object_geoms = p4u1.geoms_of_body(model, obj_bid)

    checker = CollisionChecker(
        model=model,
        object_body=args.object_body,
        side_open_ctrl=side_open_ctrl,
        min_clearance=args.approach_min_clearance,
    )

    lo, hi = joint_bounds(model)

    print("\n========== V4.12P4U5 COLLISION-AWARE PATH PLANNING ==========")
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
    print("rrt_step              :", args.rrt_step)
    print("edge_step             :", args.edge_step)
    print("plan_attempts         :", args.plan_attempts)
    print("=============================================================\n")

    seg_home_pre = plan_segment(
        name="home_to_pre",
        q_a=q_home,
        q_b=q_pre,
        checker=checker,
        bounds_lo=lo,
        bounds_hi=hi,
        args=args,
        clearance=args.approach_min_clearance,
    )

    if not seg_home_pre["success"]:
        raise RuntimeError("failed to find collision-free path: home_to_pre")

    seg_pre_grasp = plan_segment(
        name="pre_to_grasp",
        q_a=q_pre,
        q_b=q_grasp,
        checker=checker,
        bounds_lo=lo,
        bounds_hi=hi,
        args=args,
        clearance=args.grasp_path_min_clearance,
    )

    if not seg_pre_grasp["success"]:
        raise RuntimeError("failed to find collision-free path: pre_to_grasp")

    plan_result = {
        "format": "v4_12p4u5_approach_plan",
        "segments": {
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
        start_to_home_steps = max(1, int(float(args.start_to_home_duration) / dt))
        home_hold_steps = max(0, int(float(args.home_hold_duration) / dt))
        pre_hold_steps = max(0, int(float(args.pre_hold_duration) / dt))
        grasp_settle_steps = max(0, int(float(args.grasp_settle_duration) / dt))
        close_steps = max(1, int(float(args.close_duration) / dt))
        post_hold_steps = max(0, int(float(args.post_close_target_hold_duration) / dt))
        micro_steps = max(0, int(float(args.micro_squeeze_duration) / dt))
        lift_steps = max(1, int(float(args.lift_duration) / dt))
        final_hold_steps = max(0, int(float(args.final_hold_duration) / dt))

        runner.run_hold(
            "record_hold_at_zero_start_side_open",
            start_hold_steps,
            q_start,
            side_open_ctrl,
        )

        runner.run_phase(
            "record_slow_arm_zero_start_to_v12_safe_home",
            start_to_home_steps,
            q_start,
            q_home,
            side_open_ctrl,
            side_open_ctrl,
        )

        if runner.object_disp() > float(args.approach_abort_disp):
            result = make_result(
                args, model_path, runner, q_start, q_home, q_pre, q_grasp, q_lift,
                arm_source, hand_source, side_open_ctrl, close_target,
                False, "abort_zero_to_home_moved_object", dict(close_target), lift_ik_info,
                plan_result, object_geoms, model, data,
            )
            print_result(args.out, result)
            return result

        runner.run_hold(
            "record_hold_at_v12_safe_home_side_open",
            home_hold_steps,
            q_home,
            side_open_ctrl,
        )

        ok = execute_path(
            runner=runner,
            path=seg_home_pre["path"],
            label="record_collision_aware_home_to_pre",
            hand_ctrl=side_open_ctrl,
            args=args,
        )

        if not ok:
            result = make_result(
                args, model_path, runner, q_start, q_home, q_pre, q_grasp, q_lift,
                arm_source, hand_source, side_open_ctrl, close_target,
                False, "abort_collision_aware_home_to_pre_moved_object", dict(close_target), lift_ik_info,
                plan_result, object_geoms, model, data,
            )
            print_result(args.out, result)
            return result

        runner.run_hold(
            "record_hold_at_p3_pre_side_open",
            pre_hold_steps,
            q_pre,
            side_open_ctrl,
        )

        ok = execute_path(
            runner=runner,
            path=seg_pre_grasp["path"],
            label="record_collision_aware_pre_to_grasp",
            hand_ctrl=side_open_ctrl,
            args=args,
        )

        if not ok:
            result = make_result(
                args, model_path, runner, q_start, q_home, q_pre, q_grasp, q_lift,
                arm_source, hand_source, side_open_ctrl, close_target,
                False, "abort_collision_aware_pre_to_grasp_moved_object", dict(close_target), lift_ik_info,
                plan_result, object_geoms, model, data,
            )
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
                frac = float(args.micro_squeeze_fraction) * p4u1.smoothstep(alpha)

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
                    print(f"[NO GRIP] extra object disp exceeded: {disp-start_disp:.5f} > {args.max_extra_disp_during_squeeze:.5f}")
                    break

        if not grip_ready:
            print("\n[NO_LIFT] grip is not ready. Do not lift an ungrasped object.")
            result = make_result(
                args, model_path, runner, q_start, q_home, q_pre, q_grasp, q_lift,
                arm_source, hand_source, side_open_ctrl, close_target,
                False, stop_reason, grip_hold_ctrl, lift_ik_info,
                plan_result, object_geoms, model, data,
            )
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

        result = make_result(
            args, model_path, runner, q_start, q_home, q_pre, q_grasp, q_lift,
            arm_source, hand_source, side_open_ctrl, close_target,
            True, stop_reason, grip_hold_ctrl, lift_ik_info,
            plan_result, object_geoms, model, data,
        )
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
    ap.add_argument("--plan-out", default="diagnostics/current_v412/v4_12p4u5_collision_aware_approach_plan_debug.json")

    ap.add_argument("--viewer", action="store_true")
    ap.add_argument("--keep-viewer-open", action="store_true")
    ap.add_argument("--frame-sleep", type=float, default=0.0015)

    ap.add_argument(
        "--start-arm-mode",
        choices=["zero_clamped", "zero_raw", "v12_start"],
        default="zero_clamped",
    )

    ap.add_argument("--start-hold-duration", type=float, default=1.2)
    ap.add_argument("--start-to-home-duration", type=float, default=3.0)
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

    # Path planning parameters.
    ap.add_argument("--seed", type=int, default=7)
    ap.add_argument("--plan-attempts", type=int, default=8)
    ap.add_argument("--rrt-max-iters", type=int, default=3000)
    ap.add_argument("--rrt-step", type=float, default=0.28)
    ap.add_argument("--edge-step", type=float, default=0.035)
    ap.add_argument("--goal-bias", type=float, default=0.20)
    ap.add_argument("--shortcut-iters", type=int, default=350)
    ap.add_argument("--approach-min-clearance", type=float, default=0.003)
    ap.add_argument("--grasp-path-min-clearance", type=float, default=0.001)

    # Execution speed. Shorter planned path will automatically execute faster.
    ap.add_argument("--joint-speed-rad-s", type=float, default=0.75)
    ap.add_argument("--min-segment-duration", type=float, default=0.35)

    ap.add_argument("--print-every-steps", type=int, default=100)
    ap.add_argument("--log-every-steps", type=int, default=100)

    args = ap.parse_args()
    run(args)


if __name__ == "__main__":
    main()
