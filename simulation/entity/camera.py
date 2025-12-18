import os
import numpy as np
import genesis as gs

from simulation.entity.entity import Entity


script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))

class ObsCamera(Entity):
    def __init__(
        self,
        scene=None,
        surface=None,
        res=(64, 64),
        pos=(1.0, 1.0, 0.10),
        lookat=(0.100, 0.0, 0.10),
        fov=30,
    ):
        super().__init__()
        self.scene = scene
        self.surface = surface
        self.cam = None
        self.res = res
        self.pos = pos
        self.lookat = lookat
        self.fov = fov

    def create(self):
        self.cam = self.scene.add_camera(
            res=self.res,
            pos=self.pos,
            lookat=self.lookat,
            fov=self.fov,
            GUI=False
        )

    def get_image(self):
        rgb, depth, segmentation, normal = self.cam.render(depth=False, segmentation=False, normal=False)
        return rgb


class RenderCamera(Entity):
    def __init__(
        self,
        scene=None,
        surface=None,
        res=(1024, 1024),
        pos=(1.0, 1.0, 0.10),
        lookat=(0.100, 0.0, 0.10),
        fov=30,
    ):
        super().__init__()
        self.scene = scene
        self.surface = surface
        self.cam = None
        self.res = res
        self.pos = pos
        self.lookat = lookat
        self.fov = fov

    def create(self):
        self.cam = self.scene.add_camera(
            res=self.res,
            pos=self.pos,
            lookat=self.lookat,
            fov=self.fov,
            GUI=False
        )

    def get_image(self):
        rgb, depth, segmentation, normal = self.cam.render(depth=False, segmentation=False, normal=False)
        return rgb
