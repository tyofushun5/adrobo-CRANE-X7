from pathlib import Path
import sys
from typing import Any, Mapping

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np

try:
    import torch
except ImportError:  # torch is optional for just visualizing frames
    torch = None


PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from environment import environment  # noqa: F401
from robot.crane_x7 import CraneX7


def to_hwc_image(frame: Any) -> np.ndarray:
    """Convert ManiSkill render output into an image Matplotlib can show."""
    if isinstance(frame, (list, tuple)):
        frame = frame[0]

    if torch is not None and isinstance(frame, torch.Tensor):
        frame = frame.detach().cpu().numpy()
    else:
        frame = np.asarray(frame)

    if frame.ndim == 4:
        frame = frame[0]

    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))

    if frame.ndim == 3 and frame.shape[2] == 1:
        frame = frame[:, :, 0]

    return frame


def get_hand_camera_rgb(obs: Mapping[str, Any]) -> np.ndarray:
    """Extract the CRANE-X7 hand camera RGB observation."""
    sensor_data = obs.get("sensor_data", {})
    if "hand_camera" not in sensor_data:
        raise KeyError("hand_camera data not found in observation.")
    rgb = sensor_data["hand_camera"].get("rgb")
    if rgb is None:
        raise KeyError("RGB image missing from hand_camera observation.")
    return to_hwc_image(rgb)


def main():
    CraneX7.mjcf_path = str((PROJECT_ROOT / "robot" / "crane_x7.xml").resolve())
    env = gym.make(
        "PickPlace-CRANE-X7",
        render_mode="rgb_array",
        sim_backend="cpu",
        render_backend="cpu",
        robot_uids=CraneX7.uid,
        obs_mode="rgb",
    )

    try:
        reset_result = env.reset()
        if isinstance(reset_result, tuple):
            obs, _ = reset_result
        else:
            obs = reset_result

        obs_image = get_hand_camera_rgb(obs)
        render_image = to_hwc_image(env.render())

        fig, (ax_obs, ax_render) = plt.subplots(1, 2, figsize=(10, 5))
        ax_obs.imshow(obs_image)
        ax_obs.set_title("Observation: hand_camera RGB")
        ax_obs.axis("off")

        ax_render.imshow(render_image)
        ax_render.set_title("Env Render")
        ax_render.axis("off")

        plt.tight_layout()
        plt.show()
    finally:
        env.close()


if __name__ == "__main__":
    main()
