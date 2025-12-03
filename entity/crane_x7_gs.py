import os
import sys
from pathlib import Path
from typing import Optional

import numpy as np
import genesis as gs


try:
    from entity.table import TABLE_HEIGHT, add_table
except ModuleNotFoundError:
    from table import TABLE_HEIGHT, add_table


script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)

URDF_PATH = os.path.join(repo_root, "crane_x7_description", "urdf", "crane_x7_d435.urdf")


class CraneX7(object):
    def __init__(self, scene: gs.Scene = None, num_envs: int = 1, urdf_path: Optional[str] = None, root_fixed: bool = True):
        super().__init__()
        self.agent = None
        self.num_envs = num_envs
        self.arm_joint_names = [
            "crane_x7_shoulder_fixed_part_pan_joint",
            "crane_x7_shoulder_revolute_part_tilt_joint",
            "crane_x7_upper_arm_revolute_part_twist_joint",
            "crane_x7_upper_arm_revolute_part_rotate_joint",
            "crane_x7_lower_arm_fixed_part_joint",
            "crane_x7_lower_arm_revolute_part_joint",
            "crane_x7_wrist_joint",
        ]
        self.gripper_joint_names = [
            "crane_x7_gripper_finger_a_joint",
            "crane_x7_gripper_finger_b_joint",
        ]
        self.wheel_dofs = None
        self.pipe_dof = None
        self.scene = scene
        self.urdf_path = urdf_path or URDF_PATH
        self.root_fixed = root_fixed
        self.surfaces = gs.surfaces.Default(
            color=(0.0, 0.0, 0.0),
            opacity=1.0,
            roughness=0.5,
            metallic=0.0,
            emissive=None
        )

    def create(self, urdf_path: Optional[str] = None, root_fixed: Optional[bool] = None):
        urdf_path = self.urdf_path if urdf_path is None else urdf_path
        root_fixed = self.root_fixed if root_fixed is None else root_fixed

        morph = gs.morphs.URDF(
            file=urdf_path,
            decimate=True,
            decimate_face_num=2000,
            decimate_aggressiveness=5,
            convexify=True,
            visualization=True,
            collision=True,
            requires_jac_and_IK=True,
            fixed=root_fixed,
        )

        self.agent = self.scene.add_entity(
            morph=morph,
            material=None,
            surface=None,
            visualize_contact=False,
            vis_mode="visual",
        )
        # self.wheel_dofs = [
        #     self.agent.get_joint(name).dof_idx_local
        #     for name in self.wheel_joints
        # ]
        #
        # pipe_joint = self.agent.get_joint("pipe")
        # self.pipe_dof = pipe_joint.dof_idx_local

        return self.agent

    def action(self, velocity_right, velocity_left, envs_idx=None):
        vel_cmd = np.stack([velocity_right * 3.0, velocity_left * -3.0], axis=1).astype(np.float64)

        if envs_idx is not None:
            idx = np.r_[envs_idx].tolist()
            vel_cmd = vel_cmd[idx]

        self.agent.control_dofs_velocity(vel_cmd, self.wheel_dofs, envs_idx)

    def init_pose(self, envs_idx=None):
        """
        Apply ManiSkill-like 'rest' joint pose after scene.build().
        """
        if envs_idx is None:
            n_envs = getattr(self.scene, "n_envs", self.num_envs)
            envs_idx = np.arange(n_envs)
        envs_idx = np.r_[envs_idx]

        joint_order = self.arm_joint_names + self.gripper_joint_names
        joint_indices = [self.agent.get_joint(name).dof_idx_local for name in joint_order]
        rest_qpos = np.array(
            [0.0, np.pi / 8, 0.0, -np.pi * 5 / 8, 0.0, -np.pi / 2, np.pi / 2, 0.0, 0.0],
            dtype=np.float64,
        )
        rest_qpos = np.tile(rest_qpos, (len(envs_idx), 1))
        self.agent.set_dofs_position(rest_qpos, joint_indices, zero_velocity=True, envs_idx=envs_idx.tolist())

if __name__ == "__main__":

    gs.init(
        seed = None,
        precision = '32',
        debug = False,
        eps = 1e-12,
        backend = gs.gpu,
        theme = 'dark',
        logger_verbose_time = False
    )

    num_envs = 2

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(
            dt=0.01,
            gravity=(0, 0, -9.81),
        ),
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=True,
            enable_collision=True,
            constraint_solver=gs.constraint_solver.Newton,
            iterations=150,
            tolerance=1e-6,
            contact_resolve_time=0.01,
            use_contact_island=False,
            use_hibernation=False
        ),
        show_viewer=True,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(3.5, 0.0, 2.5),
            camera_lookat=(0.0, 0.0, 0.5),
            camera_fov=35,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=True,
            world_frame_size=1.0,
            show_link_frame=False,
            show_cameras=False,
            plane_reflection=True,
            shadow=True,
            background_color=(0.02, 0.04, 0.08),
            ambient_light=(0.12, 0.12, 0.12),
            lights=[
                {"type": "directional", "dir": (-0.6, -0.7, -1.0), "color": (1.0, 0.98, 0.95), "intensity": 3.0},
                {"type": "directional", "dir": (0.4, 0.1, -1.0), "color": (0.9, 0.95, 1.0), "intensity": 1.5},
            ],
            rendered_envs_idx=list(range(num_envs)),
        ),
        renderer=gs.renderers.Rasterizer(),
    )

    plane = scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, -TABLE_HEIGHT)))
    table = add_table(scene)
    crane_x7= CraneX7(scene, num_envs=num_envs, root_fixed=True)
    crane_x7.create()

    scene.build(n_envs=num_envs, env_spacing=(2.0, 3.0))
    crane_x7.init_pose()

    for i in range(100000):
        scene.step()
