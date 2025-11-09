import os
from typing import Optional, Mapping, Any

import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

import custom_env
from robot.crane_x7 import CraneX7

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))

def to_hwc_image(frame: Any):
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


def get_hand_camera_rgb(obs: Mapping[str, Any]):
    sensor_data = obs.get("sensor_data", {})
    if "hand_camera" not in sensor_data:
        raise KeyError("hand_camera data not found in observation.")
    rgb = sensor_data["hand_camera"].get("rgb")
    if rgb is None:
        raise KeyError("RGB image missing from hand_camera observation.")
    return to_hwc_image(rgb)


def main():
    CraneX7.mjcf_path = os.path.abspath(os.path.join(PROJECT_ROOT, "robot", "crane_x7.xml"))
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
