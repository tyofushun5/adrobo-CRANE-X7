import argparse
import math
import os
import random
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, Tuple, List

import gymnasium as gym
import numpy as np
import torch
import torch.distributions as td
import torch.nn.functional as F
from torch import nn
from torch.distributions import OneHotCategoricalStraightThrough
from torch.distributions.kl import kl_divergence
from torch.nn.utils import clip_grad_norm_

from robot.crane_x7 import CraneX7
from envs import custom_env as pickplace_env

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Import registers the custom PickPlace envs.
TMP_DIR = PROJECT_ROOT / "tmp"
if not TMP_DIR.exists():
    try:
        TMP_DIR.mkdir()
    except PermissionError:
        TMP_DIR = PROJECT_ROOT
for env_key in ("TMPDIR", "TEMP", "TMP"):
    os.environ.setdefault(env_key, str(TMP_DIR))


def to_numpy(array: Any) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


class DeltaPosGripperWrapper(gym.ActionWrapper):
    """Reduce action space to (dx, dy, dz, gripper) in [-1, 1].

    The wrapped env expects a longer delta-EE action (pos + rot [+ gripper]).
    We zero rotation, scale position deltas by `max_delta_m`, and map the last
    element to the gripper channel if present.
    """

    def __init__(self, env: gym.Env, max_delta_m: float = 0.02):
        super().__init__(env)
        self.max_delta_m = max_delta_m
        # Expose a compact 4-D Box action space.
        self.action_space = gym.spaces.Box(
            low=-1.0, high=1.0, shape=(4,), dtype=np.float32
        )

    def action(self, action: np.ndarray) -> np.ndarray:
        # Expected input: [dx, dy, dz, gripper] in [-1, 1].
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        a = np.clip(a, -1.0, 1.0)
        dx_dy_dz = a[:3] * self.max_delta_m
        grip = a[3] if a.size > 3 else 0.0

        # Build the underlying env action: zeros except scaled position deltas.
        raw = self.env.action_space.sample() * 0.0
        if raw.shape[0] >= 3:
            raw[:3] = dx_dy_dz
        # Zero rotation if present.
        if raw.shape[0] >= 6:
            raw[3:6] = 0.0
        # Map gripper to last element if available.
        raw[-1] = grip if raw.size > 0 else 0.0
        return raw


class HandCameraWrapper(gym.Wrapper):

    def __init__(self, env: gym.Env, image_size: int = 64):
        super().__init__(env)
        self.image_size = image_size
        sample_obs, _ = self.env.reset()
        self.joint_dim = self._extract_joint(sample_obs).shape[0]
        self.observation_space = gym.spaces.Dict(
            {
                "image": gym.spaces.Box(
                    low=0, high=255, shape=(image_size, image_size, 3), dtype=np.uint8
                ),
                "joint_pos": gym.spaces.Box(
                    low=-np.inf, high=np.inf, shape=(self.joint_dim,), dtype=np.float32
                ),
            }
        )

    def _extract_rgb(self, obs: Dict[str, Any]) -> np.ndarray:
        camera_dict = obs["sensor_data"]["base_camera"]
        rgb = to_numpy(camera_dict["rgb"])
        if rgb.ndim == 4:
            rgb = rgb[0]
        if rgb.dtype != np.uint8:
            rgb = np.clip(rgb, 0, 1)
            rgb = (rgb * 255).astype(np.uint8)
        rgb_tensor = torch.from_numpy(rgb).permute(2, 0, 1).unsqueeze(0).float()
        rgb_tensor = F.interpolate(
            rgb_tensor, size=(self.image_size, self.image_size), mode="bilinear", align_corners=False
        )
        return rgb_tensor.squeeze(0).permute(1, 2, 0).byte().numpy()

    def _extract_joint(self, obs: Dict[str, Any]) -> np.ndarray:
        joint = obs.get("agent", {}).get("joint_pos", None)
        if joint is None:
            joint = self.env.agent.robot.get_qpos()
        joint = to_numpy(joint).astype(np.float32).reshape(-1)
        joint = np.nan_to_num(joint, nan=0.0, posinf=0.0, neginf=0.0)
        return joint

    def reset(self, **kwargs):
        obs, info = self.env.reset(**kwargs)
        return {
            "image": self._extract_rgb(obs),
            "joint_pos": self._extract_joint(obs),
        }, self._convert_info(info)

    def step(self, action):
        obs, reward, terminated, truncated, info = self.env.step(action)
        reward_value = float(reward.item() if isinstance(reward, torch.Tensor) else reward)
        converted_info = self._convert_info(info)
        return (
            {
                "image": self._extract_rgb(obs),
                "joint_pos": self._extract_joint(obs),
            },
            reward_value,
            bool(terminated),
            bool(truncated),
            converted_info,
        )

    @staticmethod
    def _convert_info(info: Dict[str, Any]) -> Dict[str, Any]:
        converted = {}
        for key, value in info.items():
            if isinstance(value, torch.Tensor):
                if value.numel() == 1:
                    converted[key] = value.item()
                else:
                    converted[key] = value.cpu().numpy()
            else:
                converted[key] = value
        return converted


def make_env(seed: int, image_size: int, sim_backend: str, render_backend: str) -> gym.Env:
    """Create a PickPlace envs, gracefully falling back between GPU/CPU backends."""

    def normalize(value: str | None) -> str:
        return (value or "auto").lower()

    requested_sim = normalize(sim_backend)
    requested_render = normalize(render_backend)

    def resolve_render(sim_choice: str) -> str:
        if requested_render == "auto":
            return "gpu" if sim_choice not in {"cpu", "physx_cpu"} else "cpu"
        return requested_render

    candidate_pairs: List[Tuple[str, str]] = []
    seen_pairs = set()

    def add_candidate(sim_choice: str, render_choice: str) -> None:
        sim_key = normalize(sim_choice)
        render_key = normalize(render_choice)
        if render_key == "auto":
            render_key = resolve_render(sim_key)
        pair = (sim_key, render_key)
        if pair not in seen_pairs:
            seen_pairs.add(pair)
            candidate_pairs.append(pair)

    if requested_sim == "auto":
        for sim_option in ("gpu", "cuda", "cpu"):
            add_candidate(sim_option, resolve_render(sim_option))
    else:
        add_candidate(requested_sim, resolve_render(requested_sim))
        if requested_sim not in {"cpu", "physx_cpu"}:
            add_candidate("cpu", "cpu")

    errors: List[Tuple[str, str, Exception]] = []
    for sim_option, render_option in candidate_pairs:
        try:
            base_env = gym.make(
                "PickPlace-CRANE-X7",
                control_mode="pd_ee_delta_pos",
                render_mode="rgb_array",
                sim_backend=sim_option,
                render_backend=render_option,
                robot_uids=CraneX7.uid,
                obs_mode="rgb",
            )
            wrapped_env = DeltaPosGripperWrapper(base_env, max_delta_m=0.02)
            wrapped_env = HandCameraWrapper(wrapped_env, image_size=image_size)
            wrapped_env.reset(seed=seed)
            setattr(wrapped_env, "resolved_sim_backend", sim_option)
            setattr(wrapped_env, "resolved_render_backend", render_option)

            if requested_sim == "auto":
                print(f"Using ManiSkill backend sim={sim_option}, render={render_option} (auto).")
            else:
                requested_pair = (requested_sim, resolve_render(requested_sim))
                if (sim_option, render_option) != requested_pair:
                    print(
                        "Requested ManiSkill backend "
                        f"sim_backend={sim_backend} render_backend={render_backend} unavailable; "
                        f"falling back to sim_backend={sim_option} render_backend={render_option}."
                    )
                else:
                    print(f"Using ManiSkill backend sim={sim_option}, render={render_option}.")

            return wrapped_env
        except Exception as exc:
            errors.append((sim_option, render_option, exc))

    attempted = "\n".join(
        f"  sim_backend={sim} render_backend={render}: {repr(err)}" for sim, render, err in errors
    )
    error_message = (
        "Failed to initialize ManiSkill PickPlace envs.\n"
        "Attempted backend combinations:\n"
        f"{attempted}"
    )
    last_exception = errors[-1][2] if errors else None
    raise RuntimeError(error_message) from last_exception


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class MSE(td.Normal):
    def __init__(self, loc, validate_args=None):
        super().__init__(loc, 1.0, validate_args=validate_args)

    @property
    def mode(self):
        return self.mean

    def sample(self, sample_shape=torch.Size()):
        return self.rsample(sample_shape)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return -((value - self.loc) ** 2) / 2


# Portions adapted from https://github.com/toshas/torch_truncnorm
class TruncatedStandardNormal(td.Distribution):
    arg_constraints = {"a": td.constraints.real, "b": td.constraints.real}
    has_rsample = True

    def __init__(self, a, b, validate_args=None):
        self.a, self.b = td.utils.broadcast_all(a, b)
        if isinstance(a, (int, float)) and isinstance(b, (int, float)):
            batch_shape = torch.Size()
        else:
            batch_shape = self.a.size()
        super().__init__(batch_shape, validate_args=validate_args)
        if self.a.dtype != self.b.dtype:
            raise ValueError("Truncation bounds types are different")
        eps = torch.finfo(self.a.dtype).eps
        if torch.any((self.a >= self.b).view(-1)):
            raise ValueError("Incorrect truncation range")
        self._dtype_min_gt_0 = eps
        self._dtype_max_lt_1 = 1 - eps
        self._little_phi_a = self._little_phi(self.a)
        self._little_phi_b = self._little_phi(self.b)
        self._big_phi_a = self._big_phi(self.a)
        self._big_phi_b = self._big_phi(self.b)
        self._Z = (self._big_phi_b - self._big_phi_a).clamp_min(eps)
        self._log_Z = self._Z.log()
        self._lpbb_m_lpaa_d_Z = (self._little_phi_b * self.b - self._little_phi_a * self.a) / self._Z
        self._mean = -(self._little_phi_b - self._little_phi_a) / self._Z
        self._mode = torch.clamp(torch.zeros_like(self.a), self.a, self.b)
        self._variance = 1 - self._lpbb_m_lpaa_d_Z - ((self._little_phi_b - self._little_phi_a) / self._Z) ** 2
        two_pi_e = torch.tensor(2 * math.pi * math.e, device=self.a.device, dtype=self.a.dtype)
        self._entropy = 0.5 * torch.log(two_pi_e) + self._log_Z - 0.5 * self._lpbb_m_lpaa_d_Z

    @staticmethod
    def _little_phi(x):
        two_pi = torch.tensor(2 * math.pi, device=x.device, dtype=x.dtype)
        return (-(x**2) * 0.5).exp() / torch.sqrt(two_pi)

    @staticmethod
    def _big_phi(x):
        sqrt_two = torch.tensor(math.sqrt(2.0), device=x.device, dtype=x.dtype)
        return 0.5 * (1 + (x / sqrt_two).erf())

    @staticmethod
    def _inv_big_phi(x):
        sqrt_two = torch.tensor(math.sqrt(2.0), device=x.device, dtype=x.dtype)
        return sqrt_two * (2 * x - 1).erfinv()

    @property
    def support(self):
        return td.constraints.interval(self.a, self.b)

    @property
    def mean(self):
        return self._mean

    @property
    def mode(self):
        return self._mode

    @property
    def variance(self):
        return self._variance

    def entropy(self):
        return self._entropy

    def cdf(self, value):
        if self._validate_args:
            self._validate_sample(value)
        return ((self._big_phi(value) - self._big_phi_a) / self._Z).clamp(0, 1)

    def icdf(self, value):
        return self._inv_big_phi(self._big_phi_a + value * self._Z)

    def log_prob(self, value):
        if self._validate_args:
            self._validate_sample(value)
        two_pi = torch.tensor(2 * math.pi, device=value.device, dtype=value.dtype)
        return -self._log_Z - 0.5 * (value**2) - 0.5 * torch.log(two_pi)

    def rsample(self, sample_shape=torch.Size()):
        shape = self._extended_shape(sample_shape)
        uniform = torch.empty(shape, device=self.a.device).uniform_(self._dtype_min_gt_0, self._dtype_max_lt_1)
        return self.icdf(uniform)


class TruncatedNormal(TruncatedStandardNormal):
    has_rsample = True

    def __init__(self, loc, scale, low, high, validate_args=None):
        self.loc, self.scale, low, high = td.utils.broadcast_all(loc, scale, low, high)
        standardized_low = (low - self.loc) / self.scale
        standardized_high = (high - self.loc) / self.scale
        super().__init__(standardized_low, standardized_high, validate_args=validate_args)
        self._log_scale = self.scale.log()
        self._mean = self._mean * self.scale + self.loc
        self._mode = torch.clamp(self.loc, low, high)
        self._variance = self._variance * self.scale**2
        self._entropy += self._log_scale

    def cdf(self, value):
        return super().cdf((value - self.loc) / self.scale)

    def icdf(self, value):
        return super().icdf(value) * self.scale + self.loc

    def log_prob(self, value):
        return super().log_prob((value - self.loc) / self.scale) - self._log_scale


class TruncNormalDist(TruncatedNormal):
    def __init__(self, loc, scale, low, high, clip=1e-6, mult=1):
        super().__init__(loc, scale, low, high)
        self._clip = clip
        self._mult = mult
        self.low = low
        self.high = high

    def sample(self, *args, **kwargs):
        event = super().rsample(*args, **kwargs)
        if self._clip:
            clipped = torch.clamp(event, self.low + self._clip, self.high - self._clip)
            event = event - event.detach() + clipped.detach()
        if self._mult:
            event *= self._mult
        return event


class RSSM(nn.Module):
    def __init__(
        self,
        mlp_hidden_dim: int,
        rnn_hidden_dim: int,
        state_dim: int,
        num_classes: int,
        action_dim: int,
        obs_embed_dim: int,
    ):
        super().__init__()
        self.rnn_hidden_dim = rnn_hidden_dim
        self.state_dim = state_dim
        self.num_classes = num_classes

        self.transition_hidden = nn.Linear(state_dim * num_classes + action_dim, mlp_hidden_dim)
        self.transition = nn.GRUCell(mlp_hidden_dim, rnn_hidden_dim)

        self.prior_hidden = nn.Linear(rnn_hidden_dim, mlp_hidden_dim)
        self.prior_logits = nn.Linear(mlp_hidden_dim, state_dim * num_classes)

        self.posterior_hidden = nn.Linear(rnn_hidden_dim + obs_embed_dim, mlp_hidden_dim)
        self.posterior_logits = nn.Linear(mlp_hidden_dim, state_dim * num_classes)

    def recurrent(self, state: torch.Tensor, action: torch.Tensor, rnn_hidden: torch.Tensor):
        hidden = F.elu(self.transition_hidden(torch.cat([state, action], dim=1)))
        return self.transition(hidden, rnn_hidden)

    def get_prior(self, rnn_hidden: torch.Tensor, detach: bool = False):
        hidden = F.elu(self.prior_hidden(rnn_hidden))
        logits = self.prior_logits(hidden).reshape(-1, self.state_dim, self.num_classes)
        prior_dist = td.Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
        if detach:
            detach_dist = td.Independent(OneHotCategoricalStraightThrough(logits=logits.detach()), 1)
            return prior_dist, detach_dist
        return prior_dist

    def get_posterior(self, rnn_hidden: torch.Tensor, embedded_obs: torch.Tensor, detach: bool = False):
        hidden = F.elu(self.posterior_hidden(torch.cat([rnn_hidden, embedded_obs], dim=1)))
        logits = self.posterior_logits(hidden).reshape(-1, self.state_dim, self.num_classes)
        posterior_dist = td.Independent(OneHotCategoricalStraightThrough(logits=logits), 1)
        if detach:
            detach_dist = td.Independent(OneHotCategoricalStraightThrough(logits=logits.detach()), 1)
            return posterior_dist, detach_dist
        return posterior_dist


class Encoder(nn.Module):
    def __init__(self, joint_dim: int, joint_embed_dim: int = 128):
        super().__init__()
        self.conv1 = nn.Conv2d(3, 48, kernel_size=4, stride=2)
        self.conv2 = nn.Conv2d(48, 96, kernel_size=4, stride=2)
        self.conv3 = nn.Conv2d(96, 192, kernel_size=4, stride=2)
        self.conv4 = nn.Conv2d(192, 384, kernel_size=4, stride=2)
        self.joint_embed = nn.Sequential(
            nn.Linear(joint_dim, joint_embed_dim),
            nn.ELU(),
            nn.Linear(joint_embed_dim, joint_embed_dim),
            nn.ELU(),
        )
        self.joint_dim = joint_dim
        self.output_dim = 1536 + joint_embed_dim

    def forward(self, images: torch.Tensor, joints: torch.Tensor):
        hidden = F.elu(self.conv1(images))
        hidden = F.elu(self.conv2(hidden))
        hidden = F.elu(self.conv3(hidden))
        image_embedding = self.conv4(hidden).reshape(images.size(0), -1)
        joint_embedding = self.joint_embed(joints)
        return torch.cat([image_embedding, joint_embedding], dim=1)


class Decoder(nn.Module):
    def __init__(self, rnn_hidden_dim: int, state_dim: int, num_classes: int):
        super().__init__()
        self.fc = nn.Linear(state_dim * num_classes + rnn_hidden_dim, 1536)
        self.dc1 = nn.ConvTranspose2d(1536, 192, kernel_size=5, stride=2)
        self.dc2 = nn.ConvTranspose2d(192, 96, kernel_size=5, stride=2)
        self.dc3 = nn.ConvTranspose2d(96, 48, kernel_size=6, stride=2)
        self.dc4 = nn.ConvTranspose2d(48, 3, kernel_size=6, stride=2)

    def forward(self, state: torch.Tensor, rnn_hidden: torch.Tensor):
        hidden = self.fc(torch.cat([state, rnn_hidden], dim=1)).view(-1, 1536, 1, 1)
        hidden = F.elu(self.dc1(hidden))
        hidden = F.elu(self.dc2(hidden))
        hidden = F.elu(self.dc3(hidden))
        mean = self.dc4(hidden)
        return td.Independent(MSE(mean), 3)


class RewardModel(nn.Module):
    def __init__(self, hidden_dim: int, rnn_hidden_dim: int, state_dim: int, num_classes: int):
        super().__init__()
        input_dim = state_dim * num_classes + rnn_hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, rnn_hidden: torch.Tensor):
        hidden = F.elu(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        hidden = F.elu(self.fc2(hidden))
        hidden = F.elu(self.fc3(hidden))
        mean = self.fc4(hidden)
        return td.Independent(MSE(mean), 1)


class Actor(nn.Module):
    def __init__(self, action_dim: int, hidden_dim: int, rnn_hidden_dim: int, state_dim: int, num_classes: int):
        super().__init__()
        input_dim = state_dim * num_classes + rnn_hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.mean = nn.Linear(hidden_dim, action_dim)
        self.std = nn.Linear(hidden_dim, action_dim)
        self.min_stddev = 0.1
        self.init_stddev = np.log(np.exp(5.0) - 1)

    def forward(self, state: torch.Tensor, rnn_hidden: torch.Tensor, eval: bool = False):
        hidden = F.elu(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        hidden = F.elu(self.fc2(hidden))
        hidden = F.elu(self.fc3(hidden))
        hidden = F.elu(self.fc4(hidden))
        mean = torch.tanh(self.mean(hidden))
        stddev = 2 * torch.sigmoid((self.std(hidden) + self.init_stddev) / 2) + self.min_stddev
        if eval:
            return mean, None, None
        dist = td.Independent(TruncNormalDist(mean, stddev, -1, 1), 1)
        action = dist.sample()
        return action, dist.log_prob(action), dist.entropy()


class Critic(nn.Module):
    def __init__(self, hidden_dim: int, rnn_hidden_dim: int, state_dim: int, num_classes: int):
        super().__init__()
        input_dim = state_dim * num_classes + rnn_hidden_dim
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, hidden_dim)
        self.fc3 = nn.Linear(hidden_dim, hidden_dim)
        self.fc4 = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Linear(hidden_dim, 1)

    def forward(self, state: torch.Tensor, rnn_hidden: torch.Tensor):
        hidden = F.elu(self.fc1(torch.cat([state, rnn_hidden], dim=1)))
        hidden = F.elu(self.fc2(hidden))
        hidden = F.elu(self.fc3(hidden))
        hidden = F.elu(self.fc4(hidden))
        return self.out(hidden)


class ReplayBuffer:
    def __init__(
        self,
        capacity: int,
        image_shape: Tuple[int, int, int],
        joint_dim: int,
        action_dim: int,
    ):
        self.capacity = capacity
        self.image_shape = image_shape
        self.joint_dim = joint_dim
        self.images = np.zeros((capacity, *image_shape), dtype=np.float32)
        self.joints = np.zeros((capacity, joint_dim), dtype=np.float32)
        self.actions = np.zeros((capacity, action_dim), dtype=np.float32)
        self.rewards = np.zeros((capacity, 1), dtype=np.float32)
        self.done = np.zeros((capacity, 1), dtype=bool)
        self.index = 0
        self.is_filled = False

    def push(self, observation, action, reward, done):
        self.images[self.index] = observation["image"]
        self.joints[self.index] = observation["joint_pos"]
        self.actions[self.index] = action
        self.rewards[self.index] = reward
        self.done[self.index] = done
        if self.index == self.capacity - 1:
            self.is_filled = True
        self.index = (self.index + 1) % self.capacity

    def sample(self, batch_size, chunk_length):
        episode_ends = np.where(self.done)[0]
        sampled_indexes = []
        max_index = len(self)
        for _ in range(batch_size):
            while True:
                initial_index = np.random.randint(0, max_index - chunk_length)
                final_index = initial_index + chunk_length - 1
                if not np.logical_and(initial_index <= episode_ends, episode_ends < final_index).any():
                    break
            sampled_indexes.extend(range(initial_index, final_index + 1))

        sample_shape = (batch_size, chunk_length)
        sampled_images = self.images[sampled_indexes].reshape(*sample_shape, *self.image_shape)
        sampled_joints = self.joints[sampled_indexes].reshape(*sample_shape, self.joint_dim)
        sampled_actions = self.actions[sampled_indexes].reshape(batch_size, chunk_length, -1)
        sampled_rewards = self.rewards[sampled_indexes].reshape(batch_size, chunk_length, 1)
        sampled_done = self.done[sampled_indexes].reshape(batch_size, chunk_length, 1)
        observations = {"image": sampled_images, "joint_pos": sampled_joints}
        return observations, sampled_actions, sampled_rewards, sampled_done

    def __len__(self):
        return self.capacity if self.is_filled else self.index


def preprocess_obs(obs: Dict[str, np.ndarray]):
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


class Agent(nn.Module):
    def __init__(self, encoder, decoder, rssm, action_model):
        super().__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.rssm = rssm
        self.action_model = action_model
        self.device = next(self.action_model.parameters()).device
        self.rnn_hidden = torch.zeros(1, rssm.rnn_hidden_dim, device=self.device)

    def __call__(self, obs, eval: bool = True):
        obs = preprocess_obs(obs)
        image = torch.as_tensor(obs["image"], device=self.device)
        image = image.transpose(1, 2).transpose(0, 1).unsqueeze(0)
        joint = torch.as_tensor(obs["joint_pos"], device=self.device).unsqueeze(0)
        with torch.no_grad():
            state_prior = self.rssm.get_prior(self.rnn_hidden)
            state = state_prior.sample().flatten(1)
            obs_dist = self.decoder(state, self.rnn_hidden)
            obs_pred = obs_dist.mean

            embedded_obs = self.encoder(image, joint)
            state_posterior = self.rssm.get_posterior(self.rnn_hidden, embedded_obs)
            state = state_posterior.sample().flatten(1)
            action, _, _ = self.action_model(state, self.rnn_hidden, eval=eval)
            self.rnn_hidden = self.rssm.recurrent(state, action, self.rnn_hidden)

        reconstructed = (obs_pred.squeeze().cpu().numpy().transpose(1, 2, 0) + 0.5).clip(0.0, 1.0)
        return action.squeeze().cpu().numpy(), reconstructed

    def reset(self):
        self.rnn_hidden = torch.zeros_like(self.rnn_hidden)

    def to(self, device):
        self.device = device
        self.encoder.to(device)
        self.decoder.to(device)
        self.rssm.to(device)
        self.action_model.to(device)
        self.rnn_hidden = self.rnn_hidden.to(device)


@dataclass
class Config:
    buffer_size: int = 200_000
    batch_size: int = 32
    seq_length: int = 50
    imagination_horizon: int = 15
    state_dim: int = 20
    num_classes: int = 20
    rnn_hidden_dim: int = 200
    mlp_hidden_dim: int = 200
    model_lr: float = 6e-4
    actor_lr: float = 8e-5
    critic_lr: float = 8e-5
    epsilon: float = 1e-5
    weight_decay: float = 1e-6
    gradient_clipping: float = 50.0
    kl_scale: float = 0.2
    kl_balance: float = 0.8
    actor_entropy_scale: float = 3e-3
    slow_critic_update: int = 100
    reward_loss_scale: float = 1.0
    discount: float = 0.995
    lambda_: float = 0.95
    pretrain_iters: int = 200
    iter: int = 80000
    seed_iter: int = 1000
    eval_freq: int = 10
    log_freq: int = 100
    eval_episodes: int = 5
    image_size: int = 64
    seed: int = 1
    sim_backend: str = "cpu"
    render_backend: str = "cpu"
    device: str = "auto"
    save_path: str = "dreamer_agent.pth"
    checkpoint_freq: int = 5000


def evaluation(eval_env: gym.Env, policy: Agent, cfg: Config):
    successes = []
    with torch.no_grad():
        for _ in range(cfg.eval_episodes):
            obs, _ = eval_env.reset()
            policy.reset()
            done = False
            truncated = False
            info = {}
            while not done and not truncated:
                action, _ = policy(obs)
                obs, reward, done, truncated, info = eval_env.step(action)
            success = float(info.get("success", 0.0))
            successes.append(success)
    mean_success = np.mean(successes) if successes else 0.0
    print(f"Eval success rate: {mean_success:.3f}")
    return mean_success


def train(cfg: Config):
    device_str = cfg.device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    set_seed(cfg.seed)

    env = make_env(cfg.seed, cfg.image_size, cfg.sim_backend, cfg.render_backend)
    eval_env = make_env(cfg.seed + 1, cfg.image_size, cfg.sim_backend, cfg.render_backend)

    action_dim = env.action_space.shape[0]
    image_shape = env.observation_space["image"].shape
    joint_dim = env.observation_space["joint_pos"].shape[0]

    replay_buffer = ReplayBuffer(cfg.buffer_size, image_shape, joint_dim, action_dim)

    encoder = Encoder(joint_dim).to(device)
    rssm = RSSM(
        cfg.mlp_hidden_dim,
        cfg.rnn_hidden_dim,
        cfg.state_dim,
        cfg.num_classes,
        action_dim,
        encoder.output_dim,
    ).to(device)
    decoder = Decoder(cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device)
    reward_model = RewardModel(cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device)
    actor = Actor(action_dim, cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device)
    critic = Critic(cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device)
    target_critic = Critic(cfg.mlp_hidden_dim, cfg.rnn_hidden_dim, cfg.state_dim, cfg.num_classes).to(device)
    target_critic.load_state_dict(critic.state_dict())

    wm_params = list(rssm.parameters()) + list(encoder.parameters()) + list(decoder.parameters()) + list(
        reward_model.parameters()
    )

    wm_optimizer = torch.optim.Adam(wm_params, lr=cfg.model_lr, eps=cfg.epsilon, weight_decay=cfg.weight_decay)
    actor_optimizer = torch.optim.Adam(actor.parameters(), lr=cfg.actor_lr, eps=cfg.epsilon, weight_decay=cfg.weight_decay)
    critic_optimizer = torch.optim.Adam(critic.parameters(), lr=cfg.critic_lr, eps=cfg.epsilon, weight_decay=cfg.weight_decay)

    obs, _ = env.reset(seed=cfg.seed)
    done = False
    truncated = False
    print("Collecting seed experience...")
    for _ in range(cfg.seed_iter):
        action = env.action_space.sample()
        next_obs, reward, done, truncated, info = env.step(action)
        terminated = done or truncated
        replay_buffer.push(preprocess_obs(obs), action, reward, terminated)
        if terminated:
            obs, _ = env.reset()
            done = False
            truncated = False
        else:
            obs = next_obs

    policy = Agent(encoder, decoder, rssm, actor)
    policy.to(device)
    checkpoint_base = Path(cfg.save_path)
    checkpoint_base.parent.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(suffix: str | None = None):
        target_path = checkpoint_base
        if suffix:
            target_path = checkpoint_base.with_name(f"{checkpoint_base.stem}_{suffix}{checkpoint_base.suffix}")
        torch.save(policy.to("cpu"), str(target_path))
        policy.to(device)

    print("Starting world model pretraining...")
    for iteration in range(cfg.pretrain_iters):
        observations, actions, rewards, done_flags = replay_buffer.sample(cfg.batch_size, cfg.seq_length)
        done_flags = 1 - done_flags

        obs_images = torch.permute(torch.as_tensor(observations["image"], device=device), (1, 0, 4, 2, 3))
        obs_joints = torch.as_tensor(observations["joint_pos"], device=device).transpose(0, 1)
        actions = torch.as_tensor(actions, device=device).transpose(0, 1)
        rewards = torch.as_tensor(rewards, device=device).transpose(0, 1)
        done_flags = torch.as_tensor(done_flags, device=device).transpose(0, 1).float()

        emb_observations = encoder(
            obs_images.reshape(-1, 3, cfg.image_size, cfg.image_size),
            obs_joints.reshape(-1, joint_dim),
        ).view(cfg.seq_length, cfg.batch_size, -1)

        state = torch.zeros(cfg.batch_size, cfg.state_dim * cfg.num_classes, device=device)
        rnn_hidden = torch.zeros(cfg.batch_size, cfg.rnn_hidden_dim, device=device)
        states = torch.zeros(cfg.seq_length, cfg.batch_size, cfg.state_dim * cfg.num_classes, device=device)
        rnn_hiddens = torch.zeros(cfg.seq_length, cfg.batch_size, cfg.rnn_hidden_dim, device=device)
        kl_loss = 0
        for i in range(cfg.seq_length - 1):
            rnn_hidden = rssm.recurrent(state, actions[i], rnn_hidden)
            prior, detach_prior = rssm.get_prior(rnn_hidden, detach=True)
            posterior, detach_posterior = rssm.get_posterior(rnn_hidden, emb_observations[i + 1], detach=True)
            state = posterior.rsample().flatten(1)
            rnn_hiddens[i + 1] = rnn_hidden
            states[i + 1] = state
            kl_loss += cfg.kl_balance * torch.mean(kl_divergence(detach_posterior, prior)) + (
                1 - cfg.kl_balance
            ) * torch.mean(kl_divergence(posterior, detach_prior))

        kl_loss /= (cfg.seq_length - 1)
        rnn_hiddens = rnn_hiddens[1:]
        states = states[1:]

        flatten_rnn_hiddens = rnn_hiddens.view(-1, cfg.rnn_hidden_dim)
        flatten_states = states.view(-1, cfg.state_dim * cfg.num_classes)
        obs_dist = decoder(flatten_states, flatten_rnn_hiddens)
        reward_dist = reward_model(flatten_states, flatten_rnn_hiddens)

        C, H, W = obs_images.shape[2:]
        obs_loss = -torch.mean(obs_dist.log_prob(obs_images[1:].reshape(-1, C, H, W)))
        reward_loss = -torch.mean(reward_dist.log_prob(rewards[:-1].reshape(-1, 1)))
        wm_loss = obs_loss + cfg.reward_loss_scale * reward_loss + cfg.kl_scale * kl_loss
        obs_loss_value = obs_loss.item()
        reward_loss_value = reward_loss.item()
        kl_loss_value = kl_loss.item()
        wm_loss_value = wm_loss.item()

        wm_optimizer.zero_grad()
        wm_loss.backward()
        clip_grad_norm_(wm_params, cfg.gradient_clipping)
        wm_optimizer.step()

        if (iteration + 1) % cfg.log_freq == 0 or iteration == cfg.pretrain_iters - 1:
            print(
                f"[Pretrain {iteration + 1}/{cfg.pretrain_iters}] "
                f"wm={wm_loss_value:.4f} obs={obs_loss_value:.4f} "
                f"reward={reward_loss_value:.4f} kl={kl_loss_value:.4f}"
            )

    print("Main training loop...")
    obs, _ = env.reset(seed=cfg.seed + 123)
    done = False
    truncated = False
    episode_returns = []
    episode_successes = []
    current_return = 0.0
    last_wm_metrics = None
    last_actor_loss = None
    last_critic_loss = None
    last_entropy = None
    best_success = -np.inf
    for iteration in range(cfg.iter):
        with torch.no_grad():
            action, _ = policy(obs, eval=False)
            next_obs, reward, done, truncated, info = env.step(action)
            terminated = done or truncated
            replay_buffer.push(preprocess_obs(obs), action, reward, terminated)
            current_return += float(reward)
            if terminated:
                episode_returns.append(current_return)
                if len(episode_returns) > 100:
                    episode_returns.pop(0)
                episode_successes.append(float(info.get("success", 0.0)))
                if len(episode_successes) > 100:
                    episode_successes.pop(0)
                current_return = 0.0
                obs, _ = env.reset()
                done = False
                truncated = False
            else:
                obs = next_obs

        if (iteration + 1) % cfg.batch_size == 0:
            observations, actions, rewards, done_flags = replay_buffer.sample(cfg.batch_size, cfg.seq_length)
            done_flags = 1 - done_flags
            obs_images = torch.permute(torch.as_tensor(observations["image"], device=device), (1, 0, 4, 2, 3))
            obs_joints = torch.as_tensor(observations["joint_pos"], device=device).transpose(0, 1)
            actions = torch.as_tensor(actions, device=device).transpose(0, 1)
            rewards = torch.as_tensor(rewards, device=device).transpose(0, 1)
            done_flags = torch.as_tensor(done_flags, device=device).transpose(0, 1).float()

            emb_observations = encoder(
                obs_images.reshape(-1, 3, cfg.image_size, cfg.image_size),
                obs_joints.reshape(-1, joint_dim),
            ).view(cfg.seq_length, cfg.batch_size, -1)
            state = torch.zeros(cfg.batch_size, cfg.state_dim * cfg.num_classes, device=device)
            rnn_hidden = torch.zeros(cfg.batch_size, cfg.rnn_hidden_dim, device=device)
            states = torch.zeros(cfg.seq_length, cfg.batch_size, cfg.state_dim * cfg.num_classes, device=device)
            rnn_hiddens = torch.zeros(cfg.seq_length, cfg.batch_size, cfg.rnn_hidden_dim, device=device)
            kl_loss = 0
            for i in range(cfg.seq_length - 1):
                rnn_hidden = rssm.recurrent(state, actions[i], rnn_hidden)
                prior, detach_prior = rssm.get_prior(rnn_hidden, detach=True)
                posterior, detach_posterior = rssm.get_posterior(rnn_hidden, emb_observations[i + 1], detach=True)
                state = posterior.rsample().flatten(1)
                rnn_hiddens[i + 1] = rnn_hidden
                states[i + 1] = state
                kl_loss += cfg.kl_balance * torch.mean(kl_divergence(detach_posterior, prior)) + (
                    1 - cfg.kl_balance
                ) * torch.mean(kl_divergence(posterior, detach_prior))
            kl_loss /= (cfg.seq_length - 1)

            rnn_hiddens = rnn_hiddens[1:]
            states = states[1:]
            flatten_rnn_hiddens = rnn_hiddens.view(-1, cfg.rnn_hidden_dim)
            flatten_states = states.view(-1, cfg.state_dim * cfg.num_classes)

            obs_dist = decoder(flatten_states, flatten_rnn_hiddens)
            reward_dist = reward_model(flatten_states, flatten_rnn_hiddens)
            C, H, W = obs_images.shape[2:]
            obs_loss = -torch.mean(obs_dist.log_prob(obs_images[1:].reshape(-1, C, H, W)))
            reward_loss = -torch.mean(reward_dist.log_prob(rewards[:-1].reshape(-1, 1)))
            wm_loss = obs_loss + cfg.reward_loss_scale * reward_loss + cfg.kl_scale * kl_loss
            obs_loss_value = obs_loss.item()
            reward_loss_value = reward_loss.item()
            kl_loss_value = kl_loss.item()
            wm_loss_value = wm_loss.item()

            wm_optimizer.zero_grad()
            wm_loss.backward()
            clip_grad_norm_(wm_params, cfg.gradient_clipping)
            wm_optimizer.step()

            flatten_rnn_hiddens = flatten_rnn_hiddens.detach()
            flatten_states = flatten_states.detach()

            imagined_states = torch.zeros(
                cfg.imagination_horizon + 1, *flatten_states.shape, device=device
            )
            imagined_rnn_hiddens = torch.zeros(
                cfg.imagination_horizon + 1, *flatten_rnn_hiddens.shape, device=device
            )
            imagined_action_log_probs = torch.zeros(
                cfg.imagination_horizon, cfg.batch_size * (cfg.seq_length - 1), device=device
            )
            imagined_action_entropys = torch.zeros(
                cfg.imagination_horizon, cfg.batch_size * (cfg.seq_length - 1), device=device
            )

            imagined_states[0] = flatten_states
            imagined_rnn_hiddens[0] = flatten_rnn_hiddens
            for i in range(1, cfg.imagination_horizon + 1):
                actions, action_log_probs, action_entropys = actor(flatten_states, flatten_rnn_hiddens)
                flatten_rnn_hiddens = rssm.recurrent(flatten_states, actions, flatten_rnn_hiddens)
                flatten_states_prior = rssm.get_prior(flatten_rnn_hiddens)
                flatten_states = flatten_states_prior.rsample().flatten(1)

                imagined_rnn_hiddens[i] = flatten_rnn_hiddens
                imagined_states[i] = flatten_states
                imagined_action_log_probs[i - 1] = action_log_probs
                imagined_action_entropys[i - 1] = action_entropys

            imagined_states = imagined_states[1:]
            imagined_rnn_hiddens = imagined_rnn_hiddens[1:]

            flatten_imagined_states = imagined_states.view(-1, cfg.state_dim * cfg.num_classes)
            flatten_imagined_rnn_hiddens = imagined_rnn_hiddens.view(-1, cfg.rnn_hidden_dim)

            imagined_rewards = reward_model(flatten_imagined_states, flatten_imagined_rnn_hiddens).mean.view(
                cfg.imagination_horizon, -1
            )
            target_values = (
                target_critic(flatten_imagined_states, flatten_imagined_rnn_hiddens)
                .view(cfg.imagination_horizon, -1)
                .detach()
            )
            discount_arr = (cfg.discount * torch.ones_like(imagined_rewards)).to(device)
            initial_done = done_flags[1:].reshape(1, -1)
            discount_arr[0] = cfg.discount * initial_done

            lambda_target = calculate_lambda_target(imagined_rewards, discount_arr, target_values, cfg.lambda_)
            weights = torch.cumprod(
                torch.cat([torch.ones_like(discount_arr[:1]), discount_arr[:-1]], dim=0), dim=0
            )
            weights[-1] = 0.0
            objective = lambda_target + cfg.actor_entropy_scale * imagined_action_entropys
            actor_loss = -(weights * objective).mean()
            actor_loss_value = actor_loss.item()

            actor_optimizer.zero_grad()
            actor_loss.backward()
            clip_grad_norm_(actor.parameters(), cfg.gradient_clipping)
            actor_optimizer.step()

            value_mean = critic(flatten_imagined_states.detach(), flatten_imagined_rnn_hiddens.detach()).view(
                cfg.imagination_horizon, -1
            )
            value_dist = MSE(value_mean)
            critic_loss = -(weights.detach() * value_dist.log_prob(lambda_target.detach())).mean()
            critic_loss_value = critic_loss.item()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            clip_grad_norm_(critic.parameters(), cfg.gradient_clipping)
            critic_optimizer.step()

            last_wm_metrics = {
                "wm": wm_loss_value,
                "obs": obs_loss_value,
                "reward": reward_loss_value,
                "kl": kl_loss_value,
            }
            last_actor_loss = actor_loss_value
            last_critic_loss = critic_loss_value
            last_entropy = imagined_action_entropys.mean().item()

            if (iteration + 1) % cfg.slow_critic_update == 0:
                target_critic.load_state_dict(critic.state_dict())

        if (iteration + 1) % cfg.log_freq == 0 and last_wm_metrics is not None:
            recent_return = float(np.mean(episode_returns[-10:])) if episode_returns else 0.0
            recent_success = float(np.mean(episode_successes[-10:])) if episode_successes else 0.0
            print(
                f"[Iter {iteration + 1}/{cfg.iter}] "
                f"wm={last_wm_metrics['wm']:.4f} obs={last_wm_metrics['obs']:.4f} "
                f"reward={last_wm_metrics['reward']:.4f} kl={last_wm_metrics['kl']:.4f} "
                f"actor={last_actor_loss:.4f} critic={last_critic_loss:.4f} "
                f"entropy={last_entropy:.4f} return@10={recent_return:.2f} "
                f"success@10={recent_success:.2f} buffer={len(replay_buffer)}/{cfg.buffer_size}"
            )

        if (iteration + 1) % cfg.eval_freq == 0:
            success = evaluation(eval_env, policy, cfg)
            if success > best_success:
                best_success = success
                save_checkpoint()

        if cfg.checkpoint_freq and (iteration + 1) % cfg.checkpoint_freq == 0:
            save_checkpoint(f"iter{iteration + 1}")

    env.close()
    eval_env.close()
    print("Training completed.")
    return policy


def parse_args():
    parser = argparse.ArgumentParser(description="Dreamer training on ManiSkill PickPlace (CRANE-X7 hand camera).")
    parser.add_argument("--seed", type=int, default=1234)
    parser.add_argument("--total-iters", type=int, default=80000)
    parser.add_argument("--seed-iters", type=int, default=1000)
    parser.add_argument("--pretrain-iters", type=int, default=200)
    parser.add_argument("--batch-size", type=int, default=32)
    parser.add_argument("--buffer-size", type=int, default=200_000)
    parser.add_argument("--imagination-horizon", type=int, default=15)
    parser.add_argument("--model-lr", type=float, default=6e-4)
    parser.add_argument("--actor-lr", type=float, default=8e-5)
    parser.add_argument("--critic-lr", type=float, default=8e-5)
    parser.add_argument("--log-freq", type=int, default=100)
    parser.add_argument("--device", type=str, default="auto")
    parser.add_argument("--save-path", type=str, default="dreamer_agent.pth")
    parser.add_argument("--sim-backend", type=str, default="auto")
    parser.add_argument("--render-backend", type=str, default="auto")
    parser.add_argument("--image-size", type=int, default=64)
    parser.add_argument("--checkpoint-freq", type=int, default=5000)
    return parser.parse_args()


def main():
    args = parse_args()
    cfg = Config(
        seed=args.seed,
        buffer_size=args.buffer_size,
        batch_size=args.batch_size,
        imagination_horizon=args.imagination_horizon,
        model_lr=args.model_lr,
        actor_lr=args.actor_lr,
        critic_lr=args.critic_lr,
        iter=args.total_iters,
        seed_iter=args.seed_iters,
        pretrain_iters=args.pretrain_iters,
        log_freq=args.log_freq,
        device=args.device,
        save_path=args.save_path,
        sim_backend=args.sim_backend,
        render_backend=args.render_backend,
        image_size=args.image_size,
        checkpoint_freq=args.checkpoint_freq,
    )
    train(cfg)


if __name__ == "__main__":
    main()
