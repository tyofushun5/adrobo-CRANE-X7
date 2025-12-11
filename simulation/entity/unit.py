import os
import numpy as np
import genesis as gs

from simulation.entity.crane_x7 import CraneX7
from simulation.entity.table import Table
from simulation.entity.workspace import Workspace
from simulation.entity.camera import Camera


script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))


class Unit(object):
    def __init__(self, scene, num_envs=1):
        self.scene = scene
        self.num_envs = num_envs

        self.crane_x7 = CraneX7(scene=self.scene, num_envs=self.num_envs, root_fixed=True)
        self.table = Table(scene)
        self.workspace = Workspace(scene)

    def create(self):
        self.crane_x7.create()
        self.table.create()
        self.workspace.create()

    def step(self, *args, **kwargs):
        self.crane_x7.action(*args, **kwargs)

    def get_obs(self):
        self.crane_x7.get_joint_positions()

    def reset(self):
        self.crane_x7.set_gain()
        self.crane_x7.reset()

        for _ in range(100):
            self.scene.step()


if __name__ == "__main__":
    num_envs = 1
    mode = "delta_xy"
    draw_workspace_bounds = True

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
        sim_options=gs.options.SimOptions(dt=0.01, gravity=(0.0, 0.0, 0.0)),
        rigid_options=gs.options.RigidOptions(
            enable_joint_limit=True,
            enable_collision=True,
            constraint_solver=gs.constraint_solver.Newton,
            iterations=150,
            tolerance=1e-6,
            contact_resolve_time=0.01,
            use_contact_island=False,
            use_hibernation=False,
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


    unit = Unit(scene, num_envs=num_envs)
    unit.create()

    plane = scene.add_entity(gs.morphs.Plane(pos=(0.0, 0.0, -unit.table.table_height)))

    scene.build(n_envs=num_envs, env_spacing=(2.0, 3.0))

    unit.reset()

    if mode in ("delta_xy", "delta_xyz"):
        num_steps = 10000
        rng = np.random.default_rng(0)
        for _ in range(num_steps):
            if mode == "delta_xy":
                action = rng.uniform(-1.0, 1.0, size=(2,))
            else:
                action = rng.uniform(-1.0, 1.0, size=(3,))
            unit.step(
                target=action,
                target_quat=None,
            )
            scene.step()
    elif mode == "workspace_vis":
        while scene.viewer.is_alive():
            scene.step()
