import numpy as np
from gymnasium import spaces, Env
import genesis as gs
import torch

from simulation.entity.crane_x7 import CraneX7, _default_scene
from simulation.entity.table import TABLE_HEIGHT, add_table


class Environment(Env):
    def __init__(
        self,
        num_envs: int = 1,
        max_steps: int = 300,
        control_mode: str = "delta_xy",
        device: str = "cpu",
        show_viewer: bool = False,
        record: bool = False,
        video_path: str = "videos/preview.mp4",
        fps: int = 60,
        cam_res=(1280, 960),
        cam_pos=(1.3, 0.0, 0.9),
        cam_lookat=(0.35, 0.0, 0.2),
        cam_fov: float = 60.0,
        success_threshold: float = 0.02,
        substeps: int = 10,
    ):
        if control_mode not in ("delta_xy", "delta_xyz"):
            raise ValueError(f"Unsupported control_mode '{control_mode}'")

        self.control_mode = control_mode
        action_dim = 2 if control_mode == "delta_xy" else 3

        gs.init(
            seed=None,
            precision="32",
            debug=False,
            eps=1e-12,
            logging_level="warning",
            backend=gs.cpu if device == "cpu" else gs.gpu,
            theme="dark",
            logger_verbose_time=False,
        )

        self.scene = _default_scene(num_envs=num_envs, show_viewer=show_viewer)
        self.plane = self.scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, -TABLE_HEIGHT)))
        self.table = add_table(self.scene)
        self.crane = CraneX7(self.scene, num_envs=num_envs, root_fixed=True)
        self.crane.create()

        obs_low = np.concatenate(
            [
                self.crane.workspace_min + self.crane.workspace_margin,
                self.crane.workspace_min + self.crane.workspace_margin,
            ]
        ).astype(np.float32)
        obs_high = np.concatenate(
            [
                self.crane.workspace_max - self.crane.workspace_margin,
                self.crane.workspace_max - self.crane.workspace_margin,
            ]
        ).astype(np.float32)

        self.single_action_space = spaces.Box(
            low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32
        )
        self.single_observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            dtype=np.float32,
        )

        # gym.Env expects these attributes
        self.action_space = self.single_action_space
        self.observation_space = self.single_observation_space

        super().__init__()
        self.device = torch.device(device)
        self.num_envs = num_envs
        self.env_ids = torch.arange(self.num_envs, device=self.device)
        self.max_steps = max_steps
        self.substeps = substeps
        self.success_threshold = success_threshold
        self.step_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.targets = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

        self._record = record
        self._video_path = video_path
        self._fps = fps
        self._cam = None
        self._recording = False
        self.cam_res = cam_res
        self.cam_pos = cam_pos
        self.cam_lookat = cam_lookat
        self.cam_fov = cam_fov

        if self._record:
            from pathlib import Path

            Path(self._video_path).parent.mkdir(parents=True, exist_ok=True)
            self._cam = self.scene.add_camera(
                res=self.cam_res,
                pos=self.cam_pos,
                lookat=self.cam_lookat,
                fov=self.cam_fov,
                GUI=False,
            )

        self.scene.build(n_envs=self.num_envs, env_spacing=(2.0, 3.0))
        self.crane.set_gain()
        self.crane.init_pose()

        self.dt_phys = self.scene.dt * self.substeps

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count.zero_()
        self.scene.reset()
        self.crane.init_pose(envs_idx=self.env_ids.cpu().numpy())
        self.targets[:] = self._sample_targets(self.num_envs)

        if self._record and self._cam is not None:
            self._cam.start_recording()
            self._recording = True

        observation = self._get_obs()
        infos = [{} for _ in range(self.num_envs)]
        return observation.cpu().numpy(), infos

    def _sample_targets(self, batch: int):
        low = torch.as_tensor(
            self.crane.workspace_min + self.crane.workspace_margin,
            dtype=torch.float32,
            device=self.device,
        )
        high = torch.as_tensor(
            self.crane.workspace_max - self.crane.workspace_margin,
            dtype=torch.float32,
            device=self.device,
        )

        if self.control_mode == "delta_xy":
            xy = low[:2] + torch.rand((batch, 2), device=self.device) * (high[:2] - low[:2])
            z = torch.full((batch, 1), fill_value=self.crane.table_z, device=self.device)
            return torch.cat([xy, z], dim=1)

        rand = torch.rand((batch, 3), device=self.device)
        return low + rand * (high - low)

    @staticmethod
    def _ensure_pose_import():
        # Work around Genesis Pose missing on rigid_link in some versions.
        try:
            import genesis.engine.entities.rigid_entity.rigid_link as _rl  # type: ignore
            from genesis.ext.pyrender.interaction.vec3 import Pose as _GsPose  # type: ignore
            if not hasattr(_rl, "Pose"):
                _rl.Pose = _GsPose
        except Exception:
            pass

    def _get_end_effector_position(self):
        self._ensure_pose_import()
        try:
            ee_link = self.crane.crane_x7.get_link("crane_x7_gripper_base_link")
            pos = np.asarray(ee_link.pose.p, dtype=np.float32)
            if pos.ndim == 1:
                pos = pos.reshape(1, -1)
        except Exception:
            pos = np.zeros((self.num_envs, 3), dtype=np.float32)
        return torch.as_tensor(pos, dtype=torch.float32, device=self.device)

    def _get_obs(self, ee_pos: torch.Tensor | None = None):
        if ee_pos is None:
            ee_pos = self._get_end_effector_position()
        return torch.cat([ee_pos, self.targets], dim=1)

    def _reset_indices(self, env_ids):
        env_ids = np.r_[env_ids]
        if env_ids.size == 0:
            return
        self.crane.init_pose(envs_idx=env_ids.tolist())
        self.targets[env_ids] = self._sample_targets(len(env_ids))
        self.step_count[env_ids] = 0

    def step(self, action):
        infos = [{} for _ in range(self.num_envs)]

        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        if action.ndim == 1:
            action = action.unsqueeze(0)
        action = action[: self.num_envs]

        self.crane.action(
            target=action.cpu().numpy(),
            envs_idx=self.env_ids.cpu().numpy(),
            control_mode=self.control_mode,
        )

        self.scene.step(self.substeps)

        if self._record and self._cam is not None:
            self._cam.render()

        ee_pos = self._get_end_effector_position()
        dist = torch.linalg.norm(ee_pos - self.targets, dim=1)
        reward = 1.0 - torch.tanh(dist * 5.0)
        success = dist <= self.success_threshold

        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        terminated = success

        observation = self._get_obs(ee_pos)

        done_mask = torch.logical_or(terminated, truncated)
        if done_mask.any():
            done_ids = torch.nonzero(done_mask, as_tuple=False).view(-1)
            self._reset_indices(done_ids.cpu().numpy())

        for idx in range(self.num_envs):
            infos[idx]["distance"] = float(dist[idx].cpu())

        return (
            observation.cpu().numpy(),
            reward.cpu().numpy(),
            terminated.cpu().numpy(),
            truncated.cpu().numpy(),
            infos,
        )

    def render(self):
        pass

    def close(self):
        if self._record and self._cam is not None and self._recording:
            try:
                self._cam.stop_recording(save_to_filename=self._video_path, fps=self._fps)
            finally:
                self._recording = False
