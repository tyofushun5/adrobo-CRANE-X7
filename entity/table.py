import os
import math
from typing import Union, Tuple, Dict, Any, Optional

import numpy as np
import torch
import genesis as gs

TABLE_PATH = os.path.join(repo_root, "ManiSkill", "mani_skill", "utils", "scene_builder", "table", "assets", "table.glb")

TABLE_HEIGHT = 0.9196429
TABLE_OFFSET = (0.5, 0.0, -TABLE_HEIGHT)
TABLE_SCALE = 1.75