from dataclasses import dataclass


@dataclass
class Config:
    buffer_size: int = 100_000
    batch_size: int = 50
    seq_length: int = 50
    imagination_horizon: int = 15

    state_dim: int = 32
    num_classes: int = 32
    rnn_hidden_dim: int = 600
    mlp_hidden_dim: int = 400

    model_lr: float = 2e-4
    actor_lr: float = 4e-5
    critic_lr: float = 1e-4
    epsilon: float = 1e-5
    weight_decay: float = 1e-6
    gradient_clipping: float = 100.0
    kl_scale: float = 0.1
    kl_balance: float = 0.8
    actor_entropy_scale: float = 1e-3
    slow_critic_update: int = 100
    reward_loss_scale: float = 1.0
    discount_loss_scale: float = 1.0
    update_freq: int = 4

    discount: float = 0.995
    lambda_: float = 0.95

    iter: int = 80000
    seed_iter: int = 1000
    eval_freq: int = 10
    eval_episodes: int = 5
    pretrain_iters: int = 200
    log_freq: int = 100
    image_size: int = 64
    seed: int = 1
    sim_backend: str = "cpu"
    render_backend: str = "gpu"
    device: str = "auto"
    save_path: str = "dreamer_agent.pth"
    checkpoint_freq: int = 5000


cfg = Config()
