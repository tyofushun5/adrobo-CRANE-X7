from __future__ import annotations

from pathlib import Path
from typing import Dict

import numpy as np
import torch
import torch.nn.functional as F
from torch.distributions.kl import kl_divergence
from torch.nn.utils import clip_grad_norm_

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
from simulation.envs.custom_env import Environment
from simulation.reward import transform_reward


def to_hwc_image(frame) -> np.ndarray:
    """Convert various camera outputs into HWC RGB."""
    frame = np.asarray(frame)
    if frame.ndim == 4:
        frame = frame[0]
    if frame.ndim == 3 and frame.shape[0] in (1, 3, 4):
        frame = np.transpose(frame, (1, 2, 0))
    if frame.ndim == 3 and frame.shape[2] == 1:
        frame = frame[:, :, 0]
    return frame


def prepare_image(frame, target_size: int) -> np.ndarray:
    """Ensure uint8 HWC image resized to target_size."""
    image = to_hwc_image(frame)
    image = np.ascontiguousarray(image)
    if image.dtype != np.uint8:
        image = np.clip(image, 0.0, 1.0)
        image = (image * 255.0).astype(np.uint8)
    if image.shape[0] != target_size or image.shape[1] != target_size:
        tensor = torch.from_numpy(image).permute(2, 0, 1).unsqueeze(0).float()
        tensor = F.interpolate(tensor, size=(target_size, target_size), mode="bilinear", align_corners=False)
        image = tensor.squeeze(0).permute(1, 2, 0).byte().numpy()
    return image


def capture_observation(env: Environment, image_size: int) -> Dict[str, np.ndarray]:
    rgb = env.unit.obs_camera.get_image()
    image = prepare_image(rgb, image_size)
    joint = env.crane_x7.get_joint_positions(envs_idx=0, include_gripper=True)
    joint = np.asarray(joint[0], dtype=np.float32).reshape(-1)
    return {"image": image, "joint_pos": joint}


def make_env(cfg) -> Environment:
    return Environment(
        num_envs=1,
        max_steps=cfg.env_max_steps,
        control_mode=cfg.control_mode,
        device=cfg.sim_device,
        show_viewer=cfg.show_viewer,
        record=cfg.record,
        video_path=cfg.video_path,
        fps=cfg.fps,
        obs_cam_res=cfg.obs_cam_res,
        obs_cam_pos=cfg.obs_cam_pos,
        obs_cam_lookat=cfg.obs_cam_lookat,
        obs_cam_fov=cfg.obs_cam_fov,
        substeps=cfg.substeps,
    )


def evaluation(eval_env: Environment, policy: Agent, cfg) -> float:
    returns = []
    with torch.no_grad():
        for ep in range(cfg.eval_episodes):
            eval_env.reset(seed=cfg.seed + 1234 + ep)
            policy.reset()
            obs = capture_observation(eval_env, cfg.image_size)
            done = False
            truncated = False
            episode_return = 0.0
            while not done and not truncated:
                action, _ = policy(obs)
                _, reward, terminated, truncated_arr, _ = eval_env.step(action)
                done = bool(terminated[0])
                truncated = bool(truncated_arr[0])
                episode_return += float(reward[0])
                obs = capture_observation(eval_env, cfg.image_size)
            returns.append(episode_return)
    mean_return = float(np.mean(returns)) if returns else 0.0
    print(f"Eval mean return: {mean_return:.3f}")
    return mean_return


def build_config() -> Config:
    return Config()


def train(cfg: Config):
    device_str = cfg.device
    if device_str == "auto":
        device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    else:
        device = torch.device(device_str)

    set_seed(cfg.seed)

    env = make_env(cfg)
    eval_env = make_env(cfg)

    obs = capture_observation(env, cfg.image_size)
    image_shape = obs["image"].shape
    joint_dim = obs["joint_pos"].shape[0]
    action_dim = env.action_space.shape[0]

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

    env.reset(seed=cfg.seed)
    obs = capture_observation(env, cfg.image_size)
    print("Collecting seed experience...")
    for _ in range(cfg.seed_iter):
        action = env.action_space.sample()
        _, reward, terminated, truncated_arr, _ = env.step(action)
        done = bool(terminated[0])
        truncated = bool(truncated_arr[0])
        replay_buffer.push(preprocess_obs(obs), action, transform_reward(reward[0]), done or truncated)
        obs = capture_observation(env, cfg.image_size)

    policy = Agent(encoder, decoder, rssm, actor)
    policy.to(device)
    policy.reset()
    checkpoint_base = Path(cfg.save_path)
    checkpoint_base.parent.mkdir(parents=True, exist_ok=True)

    def save_checkpoint(suffix: str | None = None):
        target_path = checkpoint_base
        if suffix:
            target_path = checkpoint_base.with_name(f"{checkpoint_base.stem}_{suffix}{checkpoint_base.suffix}")
        torch.save(policy.to("cpu"), str(target_path))
        policy.to(device)

    img_h, img_w = image_shape[0], image_shape[1]

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
            obs_images.reshape(-1, 3, img_h, img_w),
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

        wm_optimizer.zero_grad()
        wm_loss.backward()
        clip_grad_norm_(wm_params, cfg.gradient_clipping)
        wm_optimizer.step()

        if (iteration + 1) % cfg.log_freq == 0 or iteration == cfg.pretrain_iters - 1:
            print(
                f"[Pretrain {iteration + 1}/{cfg.pretrain_iters}] "
                f"wm={wm_loss.item():.4f} obs={obs_loss.item():.4f} "
                f"reward={reward_loss.item():.4f} kl={kl_loss.item():.4f}"
            )

    print("Main training loop...")
    env.reset(seed=cfg.seed + 123)
    obs = capture_observation(env, cfg.image_size)
    episode_returns = []
    current_return = 0.0
    last_wm_metrics = None
    last_actor_loss = None
    last_critic_loss = None
    last_entropy = None
    best_eval = -np.inf

    for iteration in range(cfg.iter):
        with torch.no_grad():
            action, _ = policy(obs, eval=False)
            _, reward, terminated, truncated_arr, _ = env.step(action)
            done = bool(terminated[0])
            truncated = bool(truncated_arr[0])
            transformed_reward = transform_reward(reward[0])
            replay_buffer.push(preprocess_obs(obs), action, transformed_reward, done or truncated)
            current_return += float(transformed_reward)
            obs = capture_observation(env, cfg.image_size)
            if done or truncated:
                episode_returns.append(current_return)
                if len(episode_returns) > 100:
                    episode_returns.pop(0)
                current_return = 0.0
                policy.reset()

        if (iteration + 1) % cfg.update_freq == 0:
            observations, actions, rewards, done_flags = replay_buffer.sample(cfg.batch_size, cfg.seq_length)
            done_flags = 1 - done_flags
            obs_images = torch.permute(torch.as_tensor(observations["image"], device=device), (1, 0, 4, 2, 3))
            obs_joints = torch.as_tensor(observations["joint_pos"], device=device).transpose(0, 1)
            actions = torch.as_tensor(actions, device=device).transpose(0, 1)
            rewards = torch.as_tensor(rewards, device=device).transpose(0, 1)
            done_flags = torch.as_tensor(done_flags, device=device).transpose(0, 1).float()

            emb_observations = encoder(
                obs_images.reshape(-1, 3, img_h, img_w),
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
                actions_imagined, action_log_probs, action_entropys = actor(flatten_states, flatten_rnn_hiddens)
                flatten_rnn_hiddens = rssm.recurrent(flatten_states, actions_imagined, flatten_rnn_hiddens)
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

            actor_optimizer.zero_grad()
            actor_loss.backward()
            clip_grad_norm_(actor.parameters(), cfg.gradient_clipping)
            actor_optimizer.step()

            value_mean = critic(flatten_imagined_states.detach(), flatten_imagined_rnn_hiddens.detach()).view(
                cfg.imagination_horizon, -1
            )
            value_dist = MSE(value_mean)
            critic_loss = -(weights.detach() * value_dist.log_prob(lambda_target.detach())).mean()

            critic_optimizer.zero_grad()
            critic_loss.backward()
            clip_grad_norm_(critic.parameters(), cfg.gradient_clipping)
            critic_optimizer.step()

            last_wm_metrics = {
                "wm": wm_loss.item(),
                "obs": obs_loss.item(),
                "reward": reward_loss.item(),
                "kl": kl_loss.item(),
            }
            last_actor_loss = actor_loss.item()
            last_critic_loss = critic_loss.item()
            last_entropy = imagined_action_entropys.mean().item()

            if (iteration + 1) % cfg.slow_critic_update == 0:
                target_critic.load_state_dict(critic.state_dict())

        if (iteration + 1) % cfg.log_freq == 0 and last_wm_metrics is not None:
            recent_return = float(np.mean(episode_returns[-10:])) if episode_returns else 0.0
            print(
                f"[Iter {iteration + 1}/{cfg.iter}] "
                f"wm={last_wm_metrics['wm']:.4f} obs={last_wm_metrics['obs']:.4f} "
                f"reward={last_wm_metrics['reward']:.4f} kl={last_wm_metrics['kl']:.4f} "
                f"actor={last_actor_loss:.4f} critic={last_critic_loss:.4f} "
                f"entropy={last_entropy:.4f} return@10={recent_return:.2f} "
                f"buffer={len(replay_buffer)}/{cfg.buffer_size}"
            )

        if (iteration + 1) % cfg.eval_freq == 0:
            eval_return = evaluation(eval_env, policy, cfg)
            if eval_return > best_eval:
                best_eval = eval_return
                save_checkpoint()

        if cfg.checkpoint_freq and (iteration + 1) % cfg.checkpoint_freq == 0:
            save_checkpoint(f"iter{iteration + 1}")

    env.close()
    eval_env.close()
    print("Training completed.")
    return policy


def main():
    cfg = build_config()
    save_path = Path(cfg.save_path)
    if save_path.parent and not save_path.parent.exists():
        save_path.parent.mkdir(parents=True, exist_ok=True)

    print("Launching DreamerV2 with config:")
    for key, value in cfg.__dict__.items():
        print(f"  {key}: {value}")

    train(cfg)


if __name__ == "__main__":
    main()
