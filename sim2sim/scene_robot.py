"""Fallback Dex3 scene parameters in canonical G1 joint order.

A supplied policy always overrides these tables with its exported actuator contract.
"""
import numpy as np

HERO_KP: np.ndarray = np.array(
    [100, 100, 100, 200, 20, 20, 100, 100, 100, 200, 20, 20, 300, 300, 300, 90, 60, 20, 60, 4, 4, 4, 90, 60, 20, 60, 4, 4, 4], dtype=np.float64
)

HERO_KD: np.ndarray = np.array(
    [2.5, 2.5, 2.5, 5, 0.2, 0.1, 2.5, 2.5, 2.5, 5, 0.2, 0.1, 5.0, 5.0, 5.0, 2.0, 1.0, 0.4, 1.0, 0.2, 0.2, 0.2, 2.0, 1.0, 0.4, 1.0, 0.2, 0.2, 0.2],
    dtype=np.float64,
)

HERO_DEFAULT_DOF_POS: np.ndarray = np.array(
    [-0.1, 0.0, 0.0, 0.3, -0.2, 0.0, -0.1, 0.0, 0.0, 0.3, -0.2, 0.0] + [0.0] * 17, dtype=np.float64
)

HERO_EFFORT_LIMIT: np.ndarray = np.array(
    [88, 88, 88, 139, 50, 50, 88, 88, 88, 139, 50, 50, 88, 50, 50, 25, 25, 25, 25, 25, 5, 5, 25, 25, 25, 25, 25, 5, 5], dtype=np.float64
)

def hero_to_holo(values):
    """These HERO and canonical G1 joint layouts have the same ordering."""
    return np.asarray(values)
