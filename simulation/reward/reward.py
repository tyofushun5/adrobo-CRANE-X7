from __future__ import annotations

import numpy as np


def transform_reward(reward: float | np.ndarray) -> float:
    """Clamp and squash raw reward into a stable training range."""
    return float(np.tanh(reward))
