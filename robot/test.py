import gymnasium as gym
import mani_skill.envs  # noqa: F401, needed for registration
import envs.custom_env  # noqa: F401, registers PickPlace-CRANE-X7
import numpy as np

# Headless check: run a few steps and print EE pose in base frame to verify clipping.
env = gym.make(
    "PickPlace-CRANE-X7",
    robot_uids="CRANE-X7",
    control_mode="pd_ee_delta_pos_xy_clip",
    sim_backend="cpu",
    render_mode=None,
)
env.reset(seed=0)
# unwrap to get the underlying ManiSkill env (TimeLimitWrapper etc. are on top)
base_env = env.unwrapped if hasattr(env, "unwrapped") else env

for i in range(5):
    env.step(env.action_space.sample())
    agent = getattr(base_env, "agent", None)
    if agent is None:
        print("No agent found on base_env")
        break
    # Inspect link keys and pick base/ee links
    link_keys = list(agent.robot.links_map.keys())
    print("link keys (first 5):", link_keys[:5])
    base_link = agent.robot.get_links()[0]
    ee_link = agent.robot.links_map["crane_x7_gripper_base_link"]
    base = base_link.pose
    ee = ee_link.pose
    ee_local = base.inv() * ee
    print(f"step {i}: ee_local {np.round(ee_local.p,4)} anchor {agent._ee_anchor_local}")
env.close()
print("Done headless EE clamp check.")
