import argparse
import csv
from pathlib import Path
from typing import Iterable, List

import gymnasium as gym
import numpy as np
import torch

from envs import custom_env  # noqa: F401
from robot.crane_x7 import CraneX7


ALL_JOINTS: List[str] = list(CraneX7.arm_joint_names + CraneX7.gripper_joint_names)
JOINT_INDEX = {name: idx for idx, name in enumerate(ALL_JOINTS)}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Step-response logger to inspect CRANE-X7 PD gains."
    )
    parser.add_argument("--control-mode", type=str, default="pd_joint_pos")
    parser.add_argument("--sim-backend", type=str, default="gpu")
    parser.add_argument("--render-backend", type=str, default="gpu")
    parser.add_argument("--render-mode", type=str, default="rgb_array")
    parser.add_argument("--obs-mode", type=str, default="state")
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--steps", type=int, default=600)
    parser.add_argument("--step-delay", type=int, default=50)
    parser.add_argument("--joint", action="append", default=[])
    parser.add_argument("--delta", type=float, default=0.3)
    parser.add_argument("--gripper-delta", type=float, default=0.015)
    parser.add_argument("--log-path", type=str, default="logs/gain_probe.csv")
    parser.add_argument("--steady-window", type=int, default=100)
    return parser.parse_args()


def infer_control_dt(env) -> float | None:
    for attr in ("control_timestep", "_control_timestep"):
        if hasattr(env, attr):
            value = getattr(env, attr)
            if value:
                return float(value)
    freq = getattr(getattr(env, "agent", None), "_control_freq", None)
    if freq:
        return 1.0 / float(freq)
    return None


def joints_from_args(names: Iterable[str]) -> List[str]:
    if not names:
        return list(CraneX7.arm_joint_names)
    resolved = []
    for name in names:
        if name not in JOINT_INDEX:
            raise ValueError(f"Unknown joint name '{name}'.")
        resolved.append(name)
    return resolved


def build_targets(joints: List[str], delta: float, gripper_delta: float) -> np.ndarray:
    rest = np.asarray(CraneX7.keyframes["rest"].qpos, dtype=np.float32)
    target = rest.copy()
    for name in joints:
        offset = gripper_delta if name in CraneX7.gripper_joint_names else delta
        target[JOINT_INDEX[name]] = rest[JOINT_INDEX[name]] + offset
    return rest, target


def controller_action_from_qpos(controller, qpos: np.ndarray, low: np.ndarray, high: np.ndarray) -> np.ndarray:
    if not hasattr(controller, "from_qpos"):
        raise RuntimeError("Controller does not expose from_qpos, cannot compute action.")
    action = controller.from_qpos(qpos)
    if isinstance(action, torch.Tensor):
        action = action.detach().cpu().numpy()
    action = np.asarray(action, dtype=np.float32)
    return np.clip(action, low, high)


def to_scalar(value):
    if isinstance(value, torch.Tensor):
        value = value.detach().cpu().numpy()
    if isinstance(value, np.ndarray):
        return value.reshape(-1)[0]
    return value


def write_csv(rows: List[dict], path: Path) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    fieldnames = ["step", "time", "phase"]
    for prefix in ("desired", "qpos", "qvel"):
        fieldnames.extend(f"{prefix}_{name}" for name in ALL_JOINTS)
    with path.open("w", newline="") as fp:
        writer = csv.DictWriter(fp, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def compute_summary(times: np.ndarray, qpos: np.ndarray, rest: np.ndarray, target: np.ndarray, step_delay: int, steady_window: int):
    metrics = []
    if qpos.size == 0:
        return metrics
    signed_deltas = target - rest
    start_time = times[step_delay] if step_delay < len(times) else times[-1]
    steady_slice = qpos[-steady_window:] if steady_window > 0 else qpos[-1:]
    for idx, name in enumerate(ALL_JOINTS):
        delta = signed_deltas[idx]
        if abs(delta) < 1e-6:
            continue
        signal = (qpos[:, idx] - rest[idx]) * np.sign(delta)
        target_mag = abs(delta)
        rise_idx = np.where(signal >= 0.9 * target_mag)[0]
        rise_time = times[rise_idx[0]] - start_time if rise_idx.size else None
        overshoot = max(signal.max() - target_mag, 0.0) / target_mag * 100.0
        steady_err = steady_slice[:, idx].mean() - target[idx] if steady_slice.size else np.nan
        metrics.append(
            dict(
                joint=name,
                commanded=target[idx],
                rise_time=rise_time,
                overshoot_pct=overshoot,
                steady_error=steady_err,
            )
        )
    return metrics


def main():
    args = parse_args()
    if args.step_delay >= args.steps:
        raise ValueError("--step-delay must be smaller than --steps.")

    joints = joints_from_args(args.joint)
    rest_qpos, target_qpos = build_targets(joints, args.delta, args.gripper_delta)
    env = gym.make(
        "PickPlace-CRANE-X7",
        control_mode=args.control_mode,
        render_mode=args.render_mode,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        robot_uids=CraneX7.uid,
        obs_mode=args.obs_mode,
    )

    log_rows: List[dict] = []
    qpos_history: List[np.ndarray] = []
    times: List[float] = []
    dt = infer_control_dt(env)
    single_space = env.single_action_space
    low = np.asarray(single_space.low, dtype=np.float32)
    high = np.asarray(single_space.high, dtype=np.float32)

    try:
        env.reset(seed=args.seed)
        base_action = controller_action_from_qpos(env.agent.controller, rest_qpos, low, high)
        step_action = controller_action_from_qpos(env.agent.controller, target_qpos, low, high)

        for step in range(args.steps):
            desired = target_qpos if step >= args.step_delay else rest_qpos
            action = step_action if step >= args.step_delay else base_action
            obs, reward, terminated, truncated, info = env.step(action)

            qpos = np.asarray(env.agent.robot.get_qpos(), dtype=np.float32)[: len(ALL_JOINTS)]
            qvel = np.asarray(env.agent.robot.get_qvel(), dtype=np.float32)[: len(ALL_JOINTS)]
            qpos_history.append(qpos)
            current_time = step * dt if dt is not None else float(step)
            times.append(current_time)

            row = {
                "step": step,
                "time": current_time,
                "phase": "post" if step >= args.step_delay else "pre",
            }
            for idx, name in enumerate(ALL_JOINTS):
                row[f"desired_{name}"] = float(desired[idx])
                row[f"qpos_{name}"] = float(qpos[idx])
                row[f"qvel_{name}"] = float(qvel[idx])
            log_rows.append(row)

            done = bool(to_scalar(terminated))
            trunc = bool(to_scalar(truncated))
            if done or trunc:
                break
    finally:
        env.close()

    output_path = Path(args.log_path)
    write_csv(log_rows, output_path)

    times_arr = np.asarray(times, dtype=np.float32)
    qpos_arr = np.asarray(qpos_history, dtype=np.float32)
    metrics = compute_summary(times_arr, qpos_arr, rest_qpos, target_qpos, args.step_delay, args.steady_window)

    print(f"Logged {len(log_rows)} control steps to {output_path}.")
    if dt is not None:
        print(f"Control period: {dt * 1000:.2f} ms")
    else:
        print("Control period could not be inferred; times are expressed in steps.")
    if not metrics:
        print("No joints were perturbed, nothing to summarize.")
        return
    print("\nJoint response summary:")
    for metric in metrics:
        rise = "N/A" if metric["rise_time"] is None else f"{metric['rise_time']:.4f}"
        print(
            f"  {metric['joint']}: rise={rise}, overshoot={metric['overshoot_pct']:.2f}% "
            f"steady_error={metric['steady_error']:.4f} rad (target {metric['commanded']:.4f})"
        )


if __name__ == "__main__":
    main()
