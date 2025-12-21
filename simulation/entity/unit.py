import os
import numpy as np
import genesis as gs

from simulation.entity.crane_x7 import CraneX7
from simulation.entity.table import Table
from simulation.entity.workspace import Workspace
from simulation.entity.camera import ObsCamera, RenderCamera
from simulation.entity.cube import Cube


script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))


class Unit(object):
    def __init__(
        self,
        scene,
        num_envs=1,
        is_table=False,
        is_workspace=False,
        obs_cam_res=(64, 64),
        obs_cam_pos=(1.0, 1.0, 0.10),
        obs_cam_lookat=(0.150, 0.0, 0.10),
        obs_cam_fov=30.0,
        record_cam_res=(2048, 2048),
        record_cam_pos=(1.0, 1.0, 0.10),
        record_cam_lookat=(0.150, 0.0, 0.10),
        record_cam_fov=30.0
    ):

        self.scene = scene
        self.num_envs = num_envs
        self.is_table = is_table
        self.is_workspace = is_workspace
        self.obs_cam_res = obs_cam_res
        self.obs_cam_pos = obs_cam_pos
        self.obs_cam_lookat = obs_cam_lookat
        self.obs_cam_fov = obs_cam_fov
        self.record_cam_res = record_cam_res
        self.record_cam_pos = record_cam_pos
        self.record_cam_lookat = record_cam_lookat
        self.record_cam_fov = record_cam_fov


        self.obs_camera = ObsCamera(
            scene=self.scene,
            res=self.obs_cam_res,
            pos=self.obs_cam_pos,
            lookat=self.obs_cam_lookat,
            fov=self.obs_cam_fov
        )

        self.render_camera = RenderCamera(
            scene=self.scene,
            res=self.record_cam_res,
            pos=self.record_cam_pos,
            lookat=self.record_cam_lookat,
            fov=self.record_cam_fov
        )


        self.crane_x7 = CraneX7(scene=self.scene, num_envs=self.num_envs, root_fixed=True)
        self.table = Table(self.scene)
        self.workspace = Workspace(self.scene)

        self.cube_center = (
            self.workspace.workspace_min[0],
            self.workspace.workspace_min[1],
            self.crane_x7.table_z + 0.03 * 0.5,
        )

        self.cube = Cube(scene=self.scene, center=self.cube_center, fixed=False)
        self.cube_half = self.cube.size * 0.5

    def create(self, enable_render_camera: bool = False):
        self.crane_x7.create()
        self.obs_camera.create()
        self.cube.create()

        if self.is_table:
            self.table.create()

        if self.is_workspace:
            self.workspace.create()

        if enable_render_camera:
            self.render_camera.create()

    def step(self, *args, **kwargs):
        self.crane_x7.action(*args, **kwargs)

    def get_obs(self):
        self.crane_x7.get_joint_positions()
        self.obs_camera.get_image()

    def reset(self):
        self.crane_x7.set_gain()
        self.crane_x7.reset()

        low = self.workspace.workspace_min + self.workspace.workspace_margin
        high = self.workspace.workspace_max - self.workspace.workspace_margin
        if self.num_envs == 1:
            xy = low[:2] + np.random.rand(2) * (high[:2] - low[:2])
            z = self.crane_x7.table_z + self.cube_half
            center = np.array([xy[0], xy[1], z], dtype=np.float64)
        else:
            xy = low[:2] + np.random.rand(self.num_envs, 2) * (high[:2] - low[:2])
            z = np.full((self.num_envs, 1), self.crane_x7.table_z + self.cube_half, dtype=np.float64)
            center = np.concatenate([xy, z], axis=1)
        self.cube.reset(center=center, envs_idx=np.arange(self.num_envs))

        for _ in range(100):
            self.scene.step()


if __name__ == "__main__":
    num_envs = 1
    mode = "discrete_xyz"

    gs.init(
        seed=None,
        precision="32",
        debug=False,
        eps=1e-12,
        backend=gs.cpu,
        theme="dark",
        logger_verbose_time=False,
    )

    scene = gs.Scene(
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, -9.81)),
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=True,
            enable_collision=True,
            constraint_solver=gs.constraint_solver.Newton,
            iterations=500,
            tolerance=1e-6,
            contact_resolve_time=0.01,
            use_contact_island=False,
            use_hibernation=False,
        ),
        show_viewer=True,
        viewer_options=gs.options.ViewerOptions(
            camera_pos=(1.0, 1.0, 0.10),
            camera_lookat=(0.200, 0.0, 0.10),
            camera_fov=30,
        ),
        vis_options=gs.options.VisOptions(
            show_world_frame=False,
            world_frame_size=1.0,
            show_link_frame=False,
            show_cameras=False,
            plane_reflection=False,
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


    unit = Unit(scene=scene, num_envs=num_envs, is_workspace=True, is_table=True)
    unit.create()

    if unit.is_table:
        plane = scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, -unit.table.table_height)))
    else:
        plane = scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, 0.0)))

    scene.build(n_envs=num_envs, env_spacing=(2.0, 3.0))

    unit.reset()
    num_steps = 10000
    rng = np.random.default_rng(0)
    for _ in range(num_steps):
        action = rng.integers(0, 8, size=())
        unit.step(action=action)
        unit.get_obs()
        scene.step()
