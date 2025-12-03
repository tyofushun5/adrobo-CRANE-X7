import abc
import os
import math
from typing import Union, Tuple, Dict, Any, Optional

import numpy as np
import torch

import genesis as gs


script_dir = os.path.dirname(os.path.abspath(__file__))
MJCF_PATH = os.path.join(script_dir, "crane_x7.xml")


class CraneX7(object):
    def __init__(self, scene: gs.Scene = None, num_envs: int = 1):
        super().__init__()
        self.agent = None
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
        self.surfaces = gs.surfaces.Default(
            color=(0.0, 0.0, 0.0),
            opacity=1.0,
            roughness=0.5,
            metallic=0.0,
            emissive=None
        )

    def create(self):
        self.agent = self.scene.add_entity(
            morph=gs.morphs.MJCF(
                file=MJCF_PATH,
                decimate=False,
                # scale=1.0,
                # pos=(0.0, 0.0, 0.0),
                # euler=None,
                convexify=False,
                visualization=True,
                collision=True,
                requires_jac_and_IK=True,
            ),
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

if __name__ == "__main__":

    gs.init(
        seed = None,
        precision = '32',
        debug = False,
        eps = 1e-12,
        backend = gs.cpu,
        theme = 'dark',
        logger_verbose_time = False
    )

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
            rendered_envs_idx=[0],
        ),
        renderer=gs.renderers.Rasterizer(),
    )

    plane = scene.add_entity(gs.morphs.Plane())
    num = 1
    inverted_pendulum = CraneX7(scene, num_envs=num)
    inverted_pendulum.create()

    scene.build(n_envs=num, env_spacing=(0.5, 0.5))
    # cam.start_recording()

    for i in range(100000):
        scene.step()
    #     cam.set_pose(
    #         pos    = (3.0 * np.sin(i / 60), 3.0 * np.cos(i / 60), 2.5),
    #         lookat = (0, 0, 0.5),
    #     )
    #     cam.render()
    # cam.stop_recording(save_to_filename='video.mp4', fps=60)
