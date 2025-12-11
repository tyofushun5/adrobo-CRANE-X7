import os

import numpy as np
import genesis as gs

from simulation.entity.entity import Entity


script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))

class Workspace(Entity):
    def __init__(self, scene = None, surface=None):
        super().__init__()
        self.scene = scene
        self.surface = surface

        self.workspace_surface = None

        self.point = gs.surfaces.Default(color=(0.0, 0.3, 1.0), opacity=0.6)
        self.surface_edge = gs.surfaces.Default(color=(0.0, 1.0, 0.0), opacity=0.5)

        self.__workspace_min = np.array([0.120, -0.160, 0.070], dtype=np.float64)
        self.__workspace_max = np.array([0.360, 0.160, 0.300], dtype=np.float64)

        self.workspace_margin = 0.0

        self.workspace_min_box = self.__workspace_min + self.workspace_margin
        self.workspace_max_box = self.__workspace_max - self.workspace_margin

    def create(self):
        morph = gs.morphs.Box(
            lower=tuple(self.__workspace_min),
            upper=tuple(self.__workspace_max),
            visualization=True,
            collision=False,
            fixed=True,
        )

        self.workspace_surface = self.scene.add_entity(
            morph=morph,
            material=None,
            surface=self.surface_edge
        )

    @property
    def workspace_min(self):
        return self.__workspace_min

    @property
    def workspace_max(self):
        return self.__workspace_max
