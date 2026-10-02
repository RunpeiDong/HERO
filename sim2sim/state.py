"""Native robot-state snapshots for HERO inference and parity verification."""
from __future__ import annotations
from dataclasses import dataclass
from typing import Any
import numpy as np
from sim2sim.plant import MujocoPlant

@dataclass
class RobotState:
    """Robot state read from the plant BEFORE a control step (all quaternions xyzw, joints in holosoma order)."""

    root_pos: np.ndarray
    root_quat: np.ndarray
    root_ang_vel_b: np.ndarray
    dof_pos: np.ndarray
    dof_vel: np.ndarray
    palm_pos_w: np.ndarray  # (2, 3) [left, right] palm points
    palm_quat_w: np.ndarray  # (2, 4)
    # Object channels for compatible exports; inactive in the browser reference.
    object_pos_w: np.ndarray | None = None
    object_quat_w: np.ndarray | None = None
    # Root linear velocity in the world frame, for exports containing h23.
    root_lin_vel_w: np.ndarray | None = None
    # Optional pose estimate exposing pos_w, quat_xyzw, and lin_vel_w.
    odom: Any = None


def read_state(plant: MujocoPlant) -> RobotState:
    """Immutable snapshot: the plant accessors return views of MuJoCo's live ``qpos`` / ``qvel``, copied here so
    controllers may keep the arrays (history buffers) across physics steps."""
    palm_p, palm_q = plant.palm_pose()
    lin_w = getattr(plant, "root_lin_vel_w", None)
    return RobotState(
        plant.root_pos.copy(), plant.root_quat.copy(), plant.root_ang_vel_b.copy(), plant.dof_pos.copy(), plant.dof_vel.copy(), palm_p, palm_q,
        root_lin_vel_w=(np.array(lin_w, dtype=np.float64, copy=True).reshape(3) if lin_w is not None else None),
    )
