from pathlib import Path

import numpy as np
import torch
from torch.serialization import add_safe_globals

from dreamerv2.config import Config
from dreamerv2.agent import Agent
from dreamerv2.utils import preprocess_obs
from simulation.envs.custom_env import Environment
from simulation.train.train import capture_observation

ROOT_DIR = Path(__file__).resolve().parent.parent.parent
CHECKPOINT_DIR = ROOT_DIR / "simulation" / "train" / "model"
CHECKPOINT_PATH = CHECKPOINT_DIR / "dreamer_agent.pth"
DEFAULT_OUTPUT = "videos/preview.mp4"
DEFAULT_STEPS = 200
DEFAULT_FPS = 60
DEFAULT_DEVICE = "cpu"
DEFAULT_SHOW_VIEWER = False
DEFAULT_DETERMINISTIC = False
DEFAULT_CHECKPOINT = str(CHECKPOINT_DIR / "dreamer_agent_iter10000.pth")


def save_video(frames: list[np.ndarray], output: str, fps: int) -> None:
    try:
        import imageio.v2 as imageio
    except Exception as exc:
        raise RuntimeError("imageio is required to write imagined videos.") from exc
    writer = imageio.get_writer(output, fps=fps, quality=8)
    try:
        for frame in frames:
            writer.append_data(frame)
    finally:
        writer.close()


def decode_frame(mean_tensor: torch.Tensor) -> np.ndarray:
    frame = mean_tensor.squeeze(0).permute(1, 2, 0).cpu().numpy()
    frame = (frame + 0.5).clip(0.0, 1.0)
    return (frame * 255.0).astype(np.uint8)


def imagined_output_path(output: str) -> str:
    path = Path(output)
    if path.suffix:
        return str(path.with_name(f"{path.stem}_imagined{path.suffix}"))
    return str(path.with_name(f"{path.name}_imagined"))


def record_episode(
    output: str,
    steps: int,
    fps: int,
    device: str,
    show_viewer: bool,
    checkpoint: str,
    deterministic: bool,
) -> None:
    env = Environment(
        num_envs=1,
        max_steps=steps,
        control_mode="discrete_xyz",
        device=device,
        show_viewer=show_viewer,
        record=True,
        video_path=output,
        fps=fps,
        record_cam_res=(64, 64),
    )

    ckpt_path = Path(checkpoint)
    if not ckpt_path.exists():
        raise FileNotFoundError(f"Checkpoint not found: {ckpt_path}")


    add_safe_globals([Agent])
    policy = torch.load(str(ckpt_path), map_location=device, weights_only=False)
    if policy is None:
        raise RuntimeError(f"Checkpoint '{ckpt_path}' did not contain a policy object.")
    policy.to(torch.device(device))
    policy.reset()

    obs, _ = env.reset()
    cfg = Config()
    obs = capture_observation(env, cfg.image_size)

    obs_proc = preprocess_obs(obs)
    image = torch.as_tensor(obs_proc["image"], device=policy.device)
    image = image.transpose(1, 2).transpose(0, 1).unsqueeze(0)
    joint = torch.as_tensor(obs_proc["joint_pos"], device=policy.device).unsqueeze(0)
    with torch.no_grad():
        rnn_hidden = torch.zeros(1, policy.rssm.rnn_hidden_dim, device=policy.device)
        embedded = policy.encoder(image, joint)
        posterior = policy.rssm.get_posterior(rnn_hidden, embedded)
        state = posterior.sample().flatten(1)

        imagined_frames: list[np.ndarray] = []
        obs_dist = policy.decoder(state, rnn_hidden)
        imagined_frames.append(decode_frame(obs_dist.mean))

        for _ in range(steps - 1):
            action, _, _ = policy.action_model(state, rnn_hidden, eval=deterministic)
            rnn_hidden = policy.rssm.recurrent(state, action, rnn_hidden)
            prior = policy.rssm.get_prior(rnn_hidden)
            state = prior.sample().flatten(1)
            obs_dist = policy.decoder(state, rnn_hidden)
            imagined_frames.append(decode_frame(obs_dist.mean))

    for _ in range(steps):
        action, _ = policy(obs, eval=deterministic)
        action_np = np.asarray(action).squeeze()
        if action_np.ndim == 0:
            action_index = int(action_np)
        elif action_np.ndim == 1 and action_np.size == env.action_space.n:
            action_index = int(np.argmax(action_np))
        elif action_np.ndim == 1 and action_np.size == 1:
            action_index = int(action_np[0])
        else:
            raise ValueError(
                f"Unexpected action shape {action_np.shape} for Discrete({env.action_space.n})"
            )
        _, reward, terminated, truncated, info = env.step(action_index)
        obs = capture_observation(env, cfg.image_size)
        if bool(terminated[0]) or bool(truncated[0]):
            break

    env.close()
    save_video(imagined_frames, imagined_output_path(output), fps)
    print(f"Saved video to: {output}")
    print(f"Saved imagined video to: {imagined_output_path(output)}")


def main():
    output_path = Path(DEFAULT_OUTPUT)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    record_episode(
        str(output_path),
        steps=DEFAULT_STEPS,
        fps=DEFAULT_FPS,
        device=DEFAULT_DEVICE,
        show_viewer=DEFAULT_SHOW_VIEWER,
        checkpoint=DEFAULT_CHECKPOINT,
        deterministic=DEFAULT_DETERMINISTIC,
    )


if __name__ == "__main__":
    main()
