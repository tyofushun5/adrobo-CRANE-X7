from __future__ import annotations

import numpy as np
from gymnasium import spaces
import gymnasium as gym
import genesis as gs
import torch

from simulation.config.genesis_init import GenesisConfig
from simulation.entity.unit import Unit


class Environment(gym.Env):
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
        cam_res=(128, 128),
        cam_pos=(1.0, 1.0, 0.10),
        cam_lookat=(0.150, 0.0, 0.10),
        cam_fov: float = 30.0,
        record_cam_res=None,
        record_cam_pos=None,
        record_cam_lookat=None,
        record_cam_fov: float | None = None,
        success_threshold: float = 0.02,
        substeps: int = 10,
    ):

        action_dim = 2 if control_mode == "delta_xy" else 3

        self.num_envs = num_envs
        self.device = torch.device(device)
        self.env_ids = torch.arange(self.num_envs, device=self.device)
        self.control_mode = control_mode
        self.max_steps = max_steps
        self.substeps = substeps
        self.success_threshold = success_threshold
        self.step_count = torch.zeros(self.num_envs, dtype=torch.int32, device=self.device)
        self.targets = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)
        self.ee_cache = torch.zeros((self.num_envs, 3), dtype=torch.float32, device=self.device)

        self._record = record
        self._video_path = video_path
        self._fps = fps
        self._cam = None
        self._recording = False
        self.cam_res = cam_res
        self.cam_pos = cam_pos
        self.cam_lookat = cam_lookat
        self.cam_fov = cam_fov
        self.record_cam_res = record_cam_res or (1024, 1024)
        self.record_cam_pos = record_cam_pos or cam_pos
        self.record_cam_lookat = record_cam_lookat or cam_lookat
        self.record_cam_fov = record_cam_fov or cam_fov

        self.genesis_cfg = GenesisConfig(
            num_envs=self.num_envs,
            device=device,
            logging_level="warning",
            show_viewer=show_viewer,
            record=record,
            video_path=video_path,
            fps=fps,
            cam_res=cam_res,
            cam_pos=cam_pos,
            cam_lookat=cam_lookat,
            cam_fov=cam_fov,
        )
        self.genesis_cfg.gs_init()
        self.scene = self.genesis_cfg.scene

        self.unit = Unit(
            self.scene,
            num_envs=self.num_envs,
            obs_cam_res=self.cam_res,
            obs_cam_pos=self.cam_pos,
            obs_cam_lookat=self.cam_lookat,
            obs_cam_fov=self.cam_fov,
            render_cam_res=self.record_cam_res,
            render_cam_pos=self.record_cam_pos,
            render_cam_lookat=self.record_cam_lookat,
            render_cam_fov=self.record_cam_fov,
        )
        self.unit.create(enable_render_camera=self._record)
        self.crane = self.unit.crane_x7
        self.workspace = self.unit.workspace
        self.table_z = float(self.crane.table_z)
        self.max_delta = float(self.crane.num_delta)

        # Cube setup (3cm edge). Success if lifted above mid-height of workspace.
        self.cube = self.unit.cube
        self.cube_size = self.cube.size
        self.cube_half = self.cube_size * 0.5
        ws_min = self.workspace.workspace_min
        ws_max = self.workspace.workspace_max
        self.success_height = float(ws_min[2] + 0.5 * (ws_max[2] - ws_min[2]))

        self.scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, -self.unit.table.table_height)))

        workspace_min = self.workspace.workspace_min + self.workspace.workspace_margin
        workspace_max = self.workspace.workspace_max - self.workspace.workspace_margin

        obs_low = np.concatenate([workspace_min, workspace_min]).astype(np.float32)
        obs_high = np.concatenate([workspace_max, workspace_max]).astype(np.float32)

        self.single_action_space = spaces.Box(low=-1.0, high=1.0, shape=(action_dim,), dtype=np.float32)
        self.single_observation_space = spaces.Box(low=obs_low, high=obs_high, dtype=np.float32)

        self.action_space = self.single_action_space
        self.observation_space = self.single_observation_space

        if self._record:
            from pathlib import Path

            Path(self._video_path).parent.mkdir(parents=True, exist_ok=True)
            self._cam = self.unit.render_camera.cam

        self.scene.build(n_envs=self.num_envs, env_spacing=(2.0, 3.0))
        self.crane.set_gain()
        self.crane.reset(envs_idx=self.env_ids.cpu().numpy())

        self.dt_phys = self.scene.dt * self.substeps

        self.reset()

    def reset(self, seed=None, options=None):
        super().reset(seed=seed)
        self.step_count.zero_()
        self.scene.reset()
        self.crane.reset(envs_idx=self.env_ids.cpu().numpy())
        self._init_cache()
        cube_centers = self._sample_targets(self.num_envs)
        self.targets[:] = cube_centers
        self._set_cube_pose(cube_centers)

        for _ in range(10):
            self.scene.step()

        if self._record and self._cam is not None:
            self._cam.start_recording()
            self._recording = True

        observation = self._get_obs()
        infos = [{} for _ in range(self.num_envs)]
        return observation.cpu().numpy(), infos

    def _sample_targets(self, batch: int):
        low = torch.as_tensor(
            self.workspace.workspace_min + self.workspace.workspace_margin,
            dtype=torch.float32,
            device=self.device,
        )
        high = torch.as_tensor(
            self.workspace.workspace_max - self.workspace.workspace_margin,
            dtype=torch.float32,
            device=self.device,
        )
        xy = low[:2] + torch.rand((batch, 2), device=self.device) * (high[:2] - low[:2])
        z = torch.full((batch, 1), fill_value=self.table_z + self.cube_half + 1e-3, device=self.device)
        return torch.cat([xy, z], dim=1)

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

    def _init_cache(self, env_ids=None):
        if env_ids is None:
            env_ids = np.arange(self.num_envs)
        env_ids = np.r_[env_ids]

        midpoint = 0.5 * (self.workspace.workspace_min + self.workspace.workspace_max)
        midpoint[2] = self.table_z
        cache = np.tile(midpoint, (len(env_ids), 1))
        self.ee_cache[env_ids] = torch.as_tensor(cache, dtype=torch.float32, device=self.device)

    def _set_cube_pose(self, centers: torch.Tensor, env_ids=None):
        if env_ids is None:
            env_ids = self.env_ids.cpu().numpy()
        centers_np = centers.detach().cpu().numpy()
        self.cube.reset(center=centers_np, envs_idx=env_ids)

    def _get_cube_position(self):
        self._ensure_pose_import()
        try:
            pos = np.asarray(self.cube.cube.pose.p, dtype=np.float32)
            if pos.ndim == 1:
                pos = pos.reshape(1, -1)
        except Exception:
            pos = np.zeros((self.num_envs, 3), dtype=np.float32)
        return torch.as_tensor(pos, dtype=torch.float32, device=self.device)

    def _get_end_effector_position(self):
        self._ensure_pose_import()
        try:
            ee_link = self.crane.crane_x7.get_link(self.crane.ee_link_name)
            pos = np.asarray(ee_link.pose.p, dtype=np.float32)
            if pos.ndim == 1:
                pos = pos.reshape(1, -1)
        except Exception:
            pos = np.zeros((self.num_envs, 3), dtype=np.float32)
        return torch.as_tensor(pos, dtype=torch.float32, device=self.device)

    def _get_obs(self, ee_pos: torch.Tensor | None = None, cube_pos: torch.Tensor | None = None):
        if ee_pos is None:
            ee_pos = self._get_end_effector_position()
        if cube_pos is None:
            cube_pos = self._get_cube_position()
        # Return EE position and cube position (same shape as original ee+target layout).
        return torch.cat([ee_pos, cube_pos], dim=1)

    def _reset_indices(self, env_ids):
        env_ids = np.r_[env_ids]
        if env_ids.size == 0:
            return
        self.crane.reset(envs_idx=env_ids.tolist())
        self._init_cache(env_ids)
        cube_centers = self._sample_targets(len(env_ids))
        self.targets[env_ids] = cube_centers
        self._set_cube_pose(cube_centers, env_ids)
        self.step_count[env_ids] = 0

    def _apply_action(self, action: torch.Tensor) -> None:
        if action.ndim == 1:
            action = action.unsqueeze(0)
        action = action[: self.num_envs]

        # Action: delta xyz + gripper open/close flag in the last dim.
        # If control_mode == delta_xy, z delta is zeroed.
        delta = torch.zeros((action.shape[0], 3), device=self.device, dtype=torch.float32)
        if self.control_mode == "delta_xy":
            delta[:, :2] = torch.clamp(action[:, :2], -1.0, 1.0)
        else:
            delta[:, :3] = torch.clamp(action[:, :3], -1.0, 1.0)

        # Gripper command: >0 => open, <=0 => close
        grip_cmd = action[:, -1]
        open_mask = grip_cmd > 0

        delta_np = delta.cpu().numpy() * self.max_delta

        base_pos = np.zeros(3, dtype=np.float64)
        base_quat = np.array([0.0, 1.0, 0.0, 0.0], dtype=np.float64)
        ws_min = self.workspace.workspace_min + self.workspace.workspace_margin
        ws_max = self.workspace.workspace_max - self.workspace.workspace_margin

        targets_world = []
        for env_idx, d in enumerate(delta_np):
            current = self.ee_cache[env_idx].cpu().numpy()
            target = current + d
            target = np.clip(target, ws_min, ws_max)
            self.ee_cache[env_idx] = torch.as_tensor(target, device=self.device, dtype=torch.float32)
            target_world = base_pos + self.crane.rotate_vec(base_quat, target)
            targets_world.append(target_world)

        targets_world = np.asarray(targets_world, dtype=np.float64)
        self.crane.ik(targets_world, envs_idx=self.env_ids.cpu().numpy(), target_quat=self.crane.default_ee_quat)

        # Apply gripper command.
        if open_mask.any():
            target = np.tile(np.array([[1.57, 1.57]], dtype=np.float64), (open_mask.sum().item(), 1))
            self.crane.crane_x7.control_dofs_position(target, self.crane.gripper_joint_dofs_idx, np.nonzero(open_mask.cpu())[0])
            self.crane.is_open_gripper = True
        if (~open_mask).any():
            target = np.tile(np.array([[-0.0873, -0.0873]], dtype=np.float64), ((~open_mask).sum().item(), 1))
            self.crane.crane_x7.control_dofs_position(target, self.crane.gripper_joint_dofs_idx, np.nonzero((~open_mask).cpu())[0])
            self.crane.is_open_gripper = False

    def step(self, action):
        infos = [{} for _ in range(self.num_envs)]

        action = torch.as_tensor(action, dtype=torch.float32, device=self.device)
        self._apply_action(action)

        self.scene.step(self.substeps)

        if self._record and self._cam is not None:
            self._cam.render()

        ee_pos = self._get_end_effector_position()
        cube_pos = self._get_cube_position()
        cube_height = cube_pos[:, 2]
        success = cube_height >= self.success_height
        reward = success.float()

        self.step_count += 1
        truncated = self.step_count >= self.max_steps
        terminated = success

        self.targets = cube_pos
        observation = self._get_obs(ee_pos, cube_pos)

        done_mask = torch.logical_or(terminated, truncated)
        if done_mask.any():
            done_ids = torch.nonzero(done_mask, as_tuple=False).view(-1)
            self._reset_indices(done_ids.cpu().numpy())

        for idx in range(self.num_envs):
            infos[idx]["cube_height"] = float(cube_height[idx].cpu())
            infos[idx]["success"] = bool(success[idx].cpu())

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
