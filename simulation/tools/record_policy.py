import argparse
from pathlib import Path

import numpy as np

from simulation.envs.custom_env import Environment


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Genesis 環境で CRANE-X7 のロールアウトを動画保存します。")
    parser.add_argument("--output", type=str, default="videos/preview.mp4", help="保存先の動画パス")
    parser.add_argument("--steps", type=int, default=300, help="シミュレーションステップ数")
    parser.add_argument("--fps", type=int, default=30, help="動画のFPS")
    parser.add_argument("--device", type=str, default="cpu", help="cpu か gpu")
    parser.add_argument("--show_viewer", action="store_true", help="Genesis ビューアを表示する場合に指定")
    return parser.parse_args()


def record_episode(output: str, steps: int, fps: int, device: str, show_viewer: bool) -> None:
    # Environment 側で録画をオンにする
    env = Environment(
        num_envs=1,
        max_steps=steps,
        control_mode="delta_xy",
        device=device,
        show_viewer=show_viewer,
        record=True,
        video_path=output,
        fps=fps,
    )

    obs, _ = env.reset()

    for _ in range(steps):
        action = env.action_space.sample().astype(np.float32)
        obs, reward, terminated, truncated, info = env.step(action[None, :])
        if bool(terminated[0]) or bool(truncated[0]):
            break

    env.close()
    print(f"Saved video to: {output}")


def main():
    args = parse_args()
    output_path = Path(args.output)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record_episode(str(output_path), steps=args.steps, fps=args.fps, device=args.device, show_viewer=args.show_viewer)


if __name__ == "__main__":
    main()
