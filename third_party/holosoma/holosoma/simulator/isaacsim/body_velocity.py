"""Explicit link-origin velocity access across IsaacLab versions (no Isaac imports).

IsaacLab's legacy ``body_pos_w`` is a link-origin position, whereas its legacy
``body_lin_vel_w`` is a COM velocity.  They are not a position/derivative pair.
Use the explicit link API when available; older releases expose the COM offset
as ``com_pos_b`` and can be converted with rigid-body kinematics.  The offset is
read on every call because domain randomization may change it after a reset.
"""

from __future__ import annotations

from typing import Any

import torch


def body_link_linear_velocity_w(data: Any) -> torch.Tensor:
    """Return ``[N, B, 3]`` WORLD velocity at each link origin.

    ``data`` is IsaacLab ArticulationData (quaternions are **wxyz** here).
    Never fall back to treating COM velocity as link velocity.  This helper does
    not alter the legacy buffers, so older checkpoints retain their baseline.
    """
    direct = getattr(data, "body_link_lin_vel_w", None)
    if direct is not None:
        return direct

    com_offset = getattr(data, "body_com_pos_b", None)
    if com_offset is None:
        com_offset = getattr(data, "com_pos_b", None)
    if com_offset is None:
        raise RuntimeError(
            "IsaacLab must expose body_link_lin_vel_w or a body COM offset "
            "(body_com_pos_b/com_pos_b); COM velocity cannot substitute for link velocity"
        )
    com_velocity = getattr(data, "body_com_lin_vel_w", None)
    if com_velocity is None:
        com_velocity = data.body_lin_vel_w
    omega = data.body_ang_vel_w
    quat = data.body_quat_w
    # R(q) r = r + 2 [q_w (q_xyz x r) + q_xyz x (q_xyz x r)].
    # Articulation poses have unit quaternions; use no in-place tensor writes.
    uv = torch.linalg.cross(quat[..., 1:], com_offset, dim=-1)
    offset_w = com_offset + 2.0 * (quat[..., :1] * uv + torch.linalg.cross(quat[..., 1:], uv, dim=-1))
    return com_velocity - torch.linalg.cross(omega, offset_w, dim=-1)
