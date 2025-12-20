import argparse
from pathlib import Path

import numpy as np
import torch
from torch.serialization import add_safe_globals

from dreamer_v2.config import Config
from dreamer_v2.agent import Agent
from simulation.envs.custom_env import Environment
from simulation.train.train import capture_observation

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_PATH = ROOT_DIR / "simulation" / "train" / "dreamer_agent.pth"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genesis 環境で CRANE-X7 のロールアウトを動画保存します。")
    parser.add_argument("--output", type=str, default="videos/preview.mp4")
    parser.add_argument("--steps", type=int, default=1000)
    parser.add_argument("--fps", type=int, default=60)
    parser.add_argument("--device", type=str, default="cpu")
    parser.add_argument("--show_viewer", action="store_true")
    parser.add_argument("--deterministic", action="store_true", help="確率的方策ではなく決定的に行動を選択します。")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=str(CHECKPOINT_PATH.with_name("dreamer_agent.pth")),
        help="読み込むポリシーのパス。デフォルトは学習スクリプトのsave_pathと同じファイル。",
    )
    return parser.parse_args()


def record_episode(
    output: str,
    steps: int,
    fps: int,
    device: str,
    show_viewer: bool,
    checkpoint: str,
    deterministic: bool,
) -> None:
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


    add_safe_globals([Agent])
    policy = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if policy is None:
        raise RuntimeError(f"Checkpoint '{ckpt_path}' did not contain a policy object.")
    policy.to(torch.device(device))
    policy.reset()

    obs, _ = env.reset()
    cfg = Config()
    obs = capture_observation(env, cfg.image_size)

    for _ in range(steps):
        action, _ = policy(obs, eval=deterministic)
        action_np = np.asarray(action).squeeze()
        if action_np.ndim == 0:
            action_index = int(action_np)
        elif action_np.ndim == 1 and action_np.size == env.action_space.n:
            action_index = int(np.argmax(action_np))
        elif action_np.ndim == 1 and action_np.size == 1:
            action_index = int(action_np[0])
        else:
            raise ValueError(
                f"Unexpected action shape {action_np.shape} for Discrete({env.action_space.n})"
            )
        _, reward, terminated, truncated, info = env.step(action_index)
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
        deterministic=args.deterministic,
    )


if __name__ == "__main__":
    main()
