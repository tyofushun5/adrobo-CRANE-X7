import argparse
from pathlib import Path

import numpy as np
import torch

from dreamer_v2.config import Config
from simulation.envs.custom_env import Environment
from simulation.train.train import capture_observation

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_PATH = ROOT_DIR / "simulation" / "train" / "dreamer_agent.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genesis 環境で CRANE-X7 のロールアウトを動画保存します。")
    parser.add_argument("--output", type=str, default="videos/preview.mp4")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--show_viewer", action="store_true")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINT_PATH.with_name("dreamer_agent.pth")),
        help="読み込むポリシーのパス。デフォルトは学習スクリプトのsave_pathと同じファイル。",
    )
    return parser.parse_args()


def record_episode(output: str, steps: int, fps: int, device: str, show_viewer: bool, checkpoint: str) -> None:
    env = Environment(
        num_envs=1,
        max_steps=steps,
        control_mode="discrete_xyz",
        device=device,
        show_viewer=show_viewer,
        record=True,
        video_path=output,
        fps=fps,
    )

    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")

    policy = torch.load(str(ckpt_path), map_location=device)
    if policy is None:
        raise RuntimeError(f"Checkpoint '{ckpt_path}' did not contain a policy object.")
    policy.to(torch.device(device))
    policy.reset()

    obs, _ = env.reset()
    cfg = Config()
    obs = capture_observation(env, cfg.image_size)

    for _ in range(steps):
        action, _ = policy(obs)
        _, reward, terminated, truncated, info = env.step(action[None, :])
        obs = capture_observation(env, cfg.image_size)
        if bool(terminated[0]) or bool(truncated[0]):
            break

    env.close()
    print(f"Saved video to: {output}")


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record_episode(
        str(output_path),
        steps=args.steps,
        fps=args.fps,
        device=args.device,
        show_viewer=args.show_viewer,
        checkpoint=args.checkpoint,
    )


if __name__ == "__main__":
    main()
