import os
import numpy as np
import genesis as gs

from simulation.entity.entity import Entity


script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))

class Camera(Entity):
    def __init__(self, scene = None, surface=None):
        super().__init__()
        self.scene = scene
        self.surface = surface
        self.cam = None

    def create(self):
        self.cam = self.scene.add_camera(
            res=(128, 128),
            pos=(1.0, 1.0, 0.10),
            lookat=(0.150, 0.0, 0.10),
            fov=30,
            GUI=True
        )

    def get_image(self):
        rgb, depth, segmentation, normal = self.cam.render(depth=False, segmentation=False, normal=False)
        return rgb