import math
import os
from typing import Optional, Tuple

import genesis as gs

script_dir = os.path.dirname(os.path.abspath(__file__))
repo_root = os.path.dirname(script_dir)

TABLE_PATH = os.path.join(
    repo_root,
    "ManiSkill",
    "mani_skill",
    "utils",
    "scene_builder",
    "table",
    "assets",
    "table.glb",
)

TABLE_HEIGHT = 0.9196429
TABLE_OFFSET = (0.5, 0.0, -TABLE_HEIGHT)
TABLE_SCALE = 1.75


def _table_quat() -> Tuple[float, float, float, float]:
    """Match ManiSkill table orientation (-90 deg around z)."""
    half = -math.pi / 2
    return math.cos(half), 0.0, 0.0, math.sin(half)


class Table:
    def __init__(
        self,
        surface: Optional[gs.surfaces.Surface] = None,
        offset: Tuple[float, float, float] = TABLE_OFFSET,
        scale: float = TABLE_SCALE,
        quat: Optional[Tuple[float, float, float, float]] = None,
        path: str = TABLE_PATH,
    ):
        self.surface = surface
        self.offset = offset
        self.scale = scale
        self.quat = _table_quat() if quat is None else quat
        self.path = path

    def create(self, scene: gs.Scene):
        return scene.add_entity(
            morph=gs.morphs.Mesh(
                file=self.path,
                scale=self.scale,
                pos=self.offset,
                quat=self.quat,
                fixed=True,
                parse_glb_with_zup=False,
            ),
            material=None,
            surface=self.surface,
            visualize_contact=False,
            vis_mode="visual",
        )


def add_table(scene: gs.Scene, surface: Optional[gs.surfaces.Surface] = None):
    return Table(surface=surface).create(scene)
