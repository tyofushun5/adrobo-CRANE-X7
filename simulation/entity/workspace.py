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

    def create(self):
        pass
