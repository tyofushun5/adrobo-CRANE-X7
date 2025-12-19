from __future__ import annotations

import numpy as np


def transform_reward(reward: float | np.ndarray) -> float:
    return float(np.tanh(reward))
