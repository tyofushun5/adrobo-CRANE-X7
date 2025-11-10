import mani_skill.envs
import gymnasium as gym

from robot.crane_x7 import CraneX7
from envs import custom_env

env = gym.make("PickPlace-CRANE-X7")

base_env = gym.make(
                "PickPlace-CRANE-X7",
                control_mode="pd_joint_pos",
                render_mode="rgb_array",
                robot_uids=CraneX7.uid,
                obs_mode="rgb",
                num_envs=1,
            )

