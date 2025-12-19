import math
import os

import genesis as gs


script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.abspath(os.path.join(script_dir, os.pardir, os.pardir))

class Table(object):
    def __init__(self, scene = None, surface=None, offset=(0.5, 0.0, -0.9196429), scale= 1.75):
        self.scene = scene
        self.surface = surface
        self.offset = offset
        self.scale = scale

        self.table = None
        self.table_path = os.path.join(repo_root,
                                        "ManiSkill",
                                        "mani_skill",
                                        "utils",
                                        "scene_builder",
                                        "table",
                                        "assets",
                                        "table.glb",)

        self.__table_height = 0.9196429
        self.quat = math.cos(-math.pi / 2), 0.0, 0.0, math.sin(-math.pi / 2)

    def create(self):
        visual = gs.morphs.Mesh(
            file=self.table_path,
            scale=self.scale,
            pos=self.offset,
            quat=self.quat,
            fixed=True,
            collision=False,
            parse_glb_with_zup=False,
        )
        self.scene.add_entity(
            morph=visual,
            material=None,
            surface=self.surface,
            visualize_contact=False,
            vis_mode="visual",
        )

        half_x = 1.209 / 2
        half_y = 2.418 / 2
        half_z = self.__table_height / 2
        lower = (
            self.offset[0] - half_x,
            self.offset[1] - half_y,
            self.offset[2],
        )
        upper = (
            self.offset[0] + half_x,
            self.offset[1] + half_y,
            self.offset[2] + self.__table_height,
        )
        collision = gs.morphs.Box(
            lower=lower,
            upper=upper,
            visualization=False,
            collision=True,
            fixed=True,
        )
        self.table = self.scene.add_entity(
            morph=collision,
            material=None,
            surface=None,
            visualize_contact=False,
            vis_mode="visual",
        )
        return self.table

    @property
    def pose(self):
        return self.table.pose

    @property
    def table_height(self):
        return self.__table_height
