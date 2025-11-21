import argparse
import os
from typing import Optional

import gymnasium as gym
import imageio.v2 as imageio
import numpy as np
import torch

from train import dreamer_pickplace as dp


PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
os.sys.modules["__main__"] = dp


def load_agent(checkpoint: str) -> Optional[dp.Agent]:
    if not os.path.exists(checkpoint):
        return None
    obj = torch.load(checkpoint, map_location="cpu")
    if obj is None:
        return None
    if isinstance(obj, torch.nn.Module):
        obj.to("cpu")
        obj.eval()
    return obj


def to_frame_array(frame) -> np.ndarray:
    if isinstance(frame, (list, tuple)):
        frame = frame[0]
    if hasattr(frame, "squeeze"):
        frame = frame.squeeze(0)
    if isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    return np.asarray(frame)


def record_episode(agent: Optional[dp.Agent], output: str, steps: int, fps: int) -> None:
    base_env = gym.make(
        "PickPlace-CRANE-X7",
        control_mode="pd_joint_pos",
        render_mode="rgb_array",
        sim_backend="cpu",
        render_backend="cpu",
        robot_uids="CRANE-X7",
        obs_mode="rgb",
    )
    env = dp.HandCameraWrapper(base_env, image_size=64)
    obs, info = env.reset()

    if agent is not None:
        agent.reset()

    frames = []
    frames.append(to_frame_array(base_env.render()))

    for _ in range(steps):
        if agent is None:
            action = base_env.action_space.sample()
        else:
            action, _ = agent(obs, eval=True)
        obs, reward, terminated, truncated, info = env.step(action)
        frames.append(to_frame_array(base_env.render()))
        if terminated or truncated:
            break

    output_dir = os.path.dirname(output)
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    with imageio.get_writer(output, fps=fps) as writer:
        for frame in frames:
            writer.append_data(frame)

    env.close()


def parse_args():
    parser = argparse.ArgumentParser(description="Record PickPlace-CRANE-X7 policy rollout.")
    parser.add_argument("--checkpoint", type=str, default="dreamer_agent_iter10000.pth")
    parser.add_argument("--output", type=str, default="policy_videos/policy_rollout.mp4")
    parser.add_argument("--steps", type=int, default=10000000)
    parser.add_argument("--fps", type=int, default=30)
    return parser.parse_args()


def main():
    args = parse_args()
    tmp_dir = os.environ.get("TMPDIR")
    if not tmp_dir:
        os.environ["TMPDIR"] = str(PROJECT_ROOT)

    agent = load_agent(args.checkpoint)
    record_episode(agent, args.output, steps=args.steps, fps=args.fps)


if __name__ == "__main__":
    main()
