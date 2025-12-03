import numpy as np
import torch


def preprocess_obs(obs):
    image = obs["image"].astype(np.float32)
    image = np.nan_to_num(image, nan=0.0, posinf=255.0, neginf=0.0)
    image = image / 255.0 - 0.5
    joint = obs["joint_pos"].astype(np.float32)
    joint = np.nan_to_num(joint, nan=0.0, posinf=0.0, neginf=0.0)
    return {"image": image, "joint_pos": joint}


def calculate_lambda_target(rewards: torch.Tensor, discounts: torch.Tensor, values: torch.Tensor, lambda_: float):
    V_lambda = torch.zeros_like(rewards)
    for t in reversed(range(rewards.shape[0])):
        if t == rewards.shape[0] - 1:
            V_lambda[t] = rewards[t] + discounts[t] * values[t]
        else:
            V_lambda[t] = rewards[t] + discounts[t] * (
                (1 - lambda_) * values[t + 1] + lambda_ * V_lambda[t + 1]
            )
    return V_lambda
