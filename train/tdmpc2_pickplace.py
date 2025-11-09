import mani_skill.envs
import gymnasium as gym

from robot.crane_x7 import CraneX7
from envs import custom_env as pickplace_env

env = gym.make("PickPlace-CRANE-X7")

base_env = gym.make(
                "PickPlace-CRANE-X7",
                render_mode="rgb_array",
                robot_uids=CraneX7.uid,
                obs_mode="rgb",
                num_envs=1,
            )

