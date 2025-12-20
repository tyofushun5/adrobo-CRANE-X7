import os
import numpy as np
import genesis as gs

from simulation.entity.entity import Entity


script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))


class Cube(Entity):

    def __init__(
        self,
        scene=None,
        surface=None,
        center=(0.30, 0.0, 0.05),
        size=0.03,
        quat=(1.0, 0.0, 0.0, 0.0),
        color=(1.0, 0.9, 0.0),
        fixed=False,
    ):
        super().__init__(scene=scene, surface=surface)
        self.scene = scene
        self.surface = gs.surfaces.Default(
            color=color,
            opacity=1.0,
            roughness=0.4,
            metallic=0.0,
            emissive=None,
        )
        self.center = np.array(center, dtype=np.float64)
        self.size = float(size)
        self.quat = np.array(quat, dtype=np.float64)
        self.fixed = fixed
        self.cube = None

    def create(self):
        half = self.size * 0.5
        origin = self.center - half
        morph = gs.morphs.Box(
            size=(self.size, self.size, self.size),
            pos=tuple(origin),
            quat=tuple(self.quat),
            visualization=True,
            collision=True,
            fixed=self.fixed,
        )

        self.cube = self.scene.add_entity(
            morph=morph,
            material=gs.materials.Rigid(),
            surface=self.surface,
            visualize_contact=False,
            vis_mode="visual",
        )
        return self.cube

    def reset(self, center=None, quat=None, envs_idx=None):
        if center is not None:
            self.center = np.array(center, dtype=np.float64)
        if quat is not None:
            self.quat = np.array(quat, dtype=np.float64)
        if self.cube is not None:
            half = self.size * 0.5
            if self.center.ndim == 2 and envs_idx is not None:
                envs_idx = np.r_[envs_idx]
                for idx, env_id in enumerate(envs_idx):
                    pos = self.center[idx] - half
                    self.cube.set_pos(pos=pos, envs_idx=env_id)
                    if hasattr(self.cube, "set_quat"):
                        self.cube.set_quat(quat=self.quat, envs_idx=env_id)
            else:
                self.cube.set_pos(pos=self.center - half, envs_idx=envs_idx)
                if hasattr(self.cube, "set_quat"):
                    self.cube.set_quat(quat=self.quat, envs_idx=envs_idx)

    @property
    def pose(self):
        return self.cube.pose
