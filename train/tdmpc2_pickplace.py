import mani_skill.envs
import gymnasium as gym
import matplotlib.pyplot as plt

from robot.crane_x7 import CraneX7
from envs import custom_env

env = gym.make(
    "PickPlace-CRANE-X7",
    control_mode="pd_joint_pos",
    render_mode="rgb_array",
    robot_uids=CraneX7.uid,
    obs_mode="rgb",
    num_envs=1,
)

obs, info = env.reset()
rgb = obs["sensor_data"]["base_camera"]["rgb"]
if isinstance(rgb, list):
    rgb = rgb[0]
if hasattr(rgb, "detach"):
    rgb = rgb.detach().cpu().numpy()
if rgb.ndim == 4:
    rgb = rgb[0]

plt.imshow(rgb)
plt.axis("off")
plt.show()
