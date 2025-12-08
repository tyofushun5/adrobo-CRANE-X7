import os
from datetime import datetime
import gymnasium as gym
import matplotlib.pyplot as plt
import numpy as np
import torch

from simulation.entity.crane_x7 import CraneX7

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), os.pardir))
OUTPUT_DIR = os.path.join(PROJECT_ROOT, "tmp")

def to_hwc_image(frame):
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


def get_hand_camera_rgb(obs):
    sensor_data = obs.get("sensor_data", {})
    rgb = sensor_data["base_camera"].get("rgb")
    return to_hwc_image(rgb)


def save_images(obs_image: np.ndarray, render_image: np.ndarray, fig: plt.Figure):
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    obs_path = os.path.join(OUTPUT_DIR, f"env_test_obs_{timestamp}.png")
    render_path = os.path.join(OUTPUT_DIR, f"env_test_render_{timestamp}.png")
    compare_path = os.path.join(OUTPUT_DIR, f"env_test_compare_{timestamp}.png")

    plt.imsave(obs_path, obs_image)
    plt.imsave(render_path, render_image)
    fig.savefig(compare_path)

    print(f"Observation image saved to: {obs_path}")
    print(f"Render image saved to: {render_path}")
    print(f"Comparison figure saved to: {compare_path}")


def main():
    CraneX7.urdf_path = os.path.abspath(
        os.path.join(
            PROJECT_ROOT,
            "entity",
            "crane_x7_description",
            "urdf",
            "crane_x7.urdf",
        )
    )
    env = gym.make(
        "PickPlace-CRANE-X7",
        render_mode="rgb_array",
        sim_backend="cpu",
        render_backend="gpu",
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
        ax_obs.set_title("Observation: camera RGB")
        ax_obs.axis("off")

        ax_render.imshow(render_image)
        ax_render.set_title("Env Render")
        ax_render.axis("off")

        plt.tight_layout()
        save_images(obs_image, render_image, fig)
        plt.show()
    finally:
        env.close()


if __name__ == "__main__":
    main()
