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
        self.__table_height = 0.9196429
        self.quat = math.cos(-math.pi / 2), 0.0, 0.0, math.sin(-math.pi / 2)
        self.table_path = os.path.join(repo_root,
                                        "ManiSkill",
                                        "mani_skill",
                                        "utils",
                                        "scene_builder",
                                        "table",
                                        "assets",
                                        "table.glb",)

    def create(self):
        morph = gs.morphs.Mesh(
            file=self.table_path,
            scale=self.scale,
            pos=self.offset,
            quat=self.quat,
            fixed=True,
            parse_glb_with_zup=False,
        )

        self.table = self.scene.add_entity(
            morph=morph,
            material=None,
            surface=self.surface,
            visualize_contact=False,
            vis_mode="visual",
        )
        return self.table

    @property
    def pose(self):
        return self.table.pose

    @property
    def table_height(self):
        return self.table_height

