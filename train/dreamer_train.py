import sys
from pathlib import Path
from typing import Any, Dict, Tuple, List

import gymnasium as gym
import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from torch.nn.utils import clip_grad_norm_

PROJECT_ROOT = Path(__file__).resolve().parents[1]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

from envs import custom_env as pickplace_env
from robot.crane_x7 import CraneX7

from dreamer_v2 import (
    Agent,
    Actor,
    Config,
    Critic,
    Decoder,
    Encoder,
    ReplayBuffer,
    RewardModel,
    RSSM,
    calculate_lambda_target,
    preprocess_obs,
)

from dreamer_v2.distributions import MSE
from dreamer_v2.tools.set_seed import set_seed


def to_numpy(array: Any) -> np.ndarray:
    if isinstance(array, torch.Tensor):
        return array.detach().cpu().numpy()
    return np.asarray(array)


def transform_reward(reward: float) -> float:
    return float(np.tanh(reward))


class DeltaPosGripperWrapper(gym.ActionWrapper):
    """Reduce action space to (dx, dy, dz, gripper) in [-1, 1]."""

    def __init__(self, env: gym.Env, max_delta_m: float = 0.02):
        super().__init__(env)
        self.max_delta_m = max_delta_m
        self.action_space = gym.spaces.Box(low=-1.0, high=1.0, shape=(4,), dtype=np.float32)

    def action(self, action: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=np.float32).reshape(-1)
        a = np.clip(a, -1.0, 1.0)
        dx_dy_dz = a[:3] * self.max_delta_m
        grip = a[3] if a.size > 3 else 0.0

        raw = self.env.action_space.sample() * 0.0
        if raw.shape[0] >= 3:
            raw[:3] = dx_dy_dz
        if raw.shape[0] >= 6:
            raw[3:6] = 0.0
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
        replay_buffer.push(preprocess_obs(obs), action, transform_reward(reward), terminated)
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
            transformed_reward = transform_reward(reward)
            replay_buffer.push(preprocess_obs(obs), action, transformed_reward, terminated)
            current_return += float(transformed_reward)
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

        if (iteration + 1) % cfg.update_freq == 0:
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


def main():
    cfg = Config()
    train(cfg)


if __name__ == "__main__":
    main()
