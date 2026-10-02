"""HERO residual end-effector pose mathematics using xyzw quaternions."""

from __future__ import annotations

import torch

# --------------------------------------------------------------------------------------------------
# quaternion primitives (xyzw), written out so the module has no framework dependency
# --------------------------------------------------------------------------------------------------


def quat_conjugate_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Conjugate (= inverse for unit quaternions) of xyzw quaternions ``[..., 4]``."""
    return torch.cat([-q[..., :3], q[..., 3:4]], dim=-1)


def quat_mul_xyzw(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Hamilton product ``a ⊗ b`` for xyzw quaternions ``[..., 4]`` (broadcasting)."""
    ax, ay, az, aw = a.unbind(-1)
    bx, by, bz, bw = b.unbind(-1)
    x = aw * bx + ax * bw + ay * bz - az * by
    y = aw * by - ax * bz + ay * bw + az * bx
    z = aw * bz + ax * by - ay * bx + az * bw
    w = aw * bw - ax * bx - ay * by - az * bz
    return torch.stack([x, y, z, w], dim=-1)


def quat_rotate_xyzw(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors ``v [..., 3]`` by xyzw quaternions ``q [..., 4]`` (R(q) v), broadcasting."""
    xyz = q[..., :3]
    w = q[..., 3:4]
    t = 2.0 * torch.cross(xyz, v, dim=-1)
    return v + w * t + torch.cross(xyz, t, dim=-1)


def quat_rotate_inverse_xyzw(q: torch.Tensor, v: torch.Tensor) -> torch.Tensor:
    """Rotate vectors by the inverse quaternion (R(q)^T v), broadcasting."""
    return quat_rotate_xyzw(quat_conjugate_xyzw(q), v)


def quat_to_matrix_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Rotation matrices ``[..., 3, 3]`` from xyzw quaternions (normalised internally)."""
    q = q / q.norm(dim=-1, keepdim=True).clamp(min=1e-12)
    x, y, z, w = q.unbind(-1)
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    m = torch.stack(
        [
            1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy),
            2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx),
            2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy),
        ],
        dim=-1,
    )
    return m.reshape(q.shape[:-1] + (3, 3))


def quat_angle_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Rotation angle in ``[0, π]`` of xyzw quaternions ``[..., 4]`` (= ‖axis-angle‖).

    Used by the EE rotation reward; sign-canonicalised so w >= 0,
    with angle = 2 atan2(‖xyz‖, w)."""
    mag = torch.linalg.norm(q[..., :3], dim=-1)
    return 2.0 * torch.atan2(mag, q[..., 3].abs())


# --------------------------------------------------------------------------------------------------
# rot6d helpers
# --------------------------------------------------------------------------------------------------


def rot6d_from_matrix(mat: torch.Tensor) -> torch.Tensor:
    """First two COLUMNS of ``mat [..., 3, 3]`` flattened row-major -> ``[..., 6]``.

    Exactly HERO's ``dif_mat[..., :2].reshape(N, -1)``: per matrix ``[R00, R01, R10, R11, R20, R21]``."""
    return mat[..., :2].reshape(mat.shape[:-2] + (6,))


def rot6d_from_quat_xyzw(q: torch.Tensor) -> torch.Tensor:
    """rot6d (first two columns) of the rotation given by xyzw quaternions ``[..., 4]`` -> ``[..., 6]``."""
    return rot6d_from_matrix(quat_to_matrix_xyzw(q))


# --------------------------------------------------------------------------------------------------
# HERO ΔE
# --------------------------------------------------------------------------------------------------


def palm_point_world(
    wrist_pos_w: torch.Tensor, wrist_quat_w: torch.Tensor, offset_local: torch.Tensor
) -> torch.Tensor:
    """World position of the palm point: ``p_wrist + R(q_wrist) offset``.

    Args:
        wrist_pos_w: ``[N, K, 3]`` wrist_yaw link positions (world).
        wrist_quat_w: ``[N, K, 4]`` xyzw wrist_yaw orientations (world).
        offset_local: ``[K, 3]`` (or broadcastable) palm offset in the wrist_yaw frame (``PALM_OFFSET``)."""
    return wrist_pos_w + quat_rotate_xyzw(wrist_quat_w, offset_local.to(wrist_pos_w).expand_as(wrist_pos_w))


def ee_pose_in_root(
    root_pos_w: torch.Tensor, root_quat_w: torch.Tensor, ee_pos_w: torch.Tensor, ee_quat_w: torch.Tensor
) -> tuple[torch.Tensor, torch.Tensor]:
    """Express EE poses in the root (pelvis) frame using the FULL root orientation.

    Args:
        root_pos_w: ``[N, 3]``; root_quat_w: ``[N, 4]`` xyzw.
        ee_pos_w: ``[N, K, 3]``; ee_quat_w: ``[N, K, 4]`` xyzw.
    Returns:
        ``(p_local [N, K, 3], q_local [N, K, 4] xyzw)`` with ``p_local = R_root^T (p_ee - p_root)``,
        ``q_local = q_root^-1 ⊗ q_ee``."""
    root_q = root_quat_w[:, None, :]
    p_local = quat_rotate_inverse_xyzw(root_q, ee_pos_w - root_pos_w[:, None, :])
    q_local = quat_mul_xyzw(quat_conjugate_xyzw(root_q).expand_as(ee_quat_w), ee_quat_w)
    return p_local, q_local


def ee_delta_pos(p_cur_local: torch.Tensor, p_ref_local: torch.Tensor) -> torch.Tensor:
    """HERO translation residual ``dp = p_cur_local - p_ref_local`` (``[N, K, 3]``)."""
    return p_cur_local - p_ref_local


def ee_delta_quat(q_cur_local: torch.Tensor, q_ref_local: torch.Tensor) -> torch.Tensor:
    """HERO rotation residual ``q_diff = q_cur_local^-1 ⊗ q_ref_local`` (xyzw, ``[N, K, 4]``)."""
    return quat_mul_xyzw(quat_conjugate_xyzw(q_cur_local), q_ref_local)


def hero_ee_residual(
    root_pos_w: torch.Tensor,
    root_quat_w: torch.Tensor,
    ee_pos_w: torch.Tensor,
    ee_quat_w: torch.Tensor,
    ref_pos_local: torch.Tensor,
    ref_quat_local: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Full HERO ΔE for K hands (all quaternions xyzw).

    Args:
        root_pos_w ``[N,3]``, root_quat_w ``[N,4]``: robot pelvis pose (world).
        ee_pos_w ``[N,K,3]``, ee_quat_w ``[N,K,4]``: current EE (palm point) poses (world).
        ref_pos_local ``[N,K,3]``, ref_quat_local ``[N,K,4]``: reference EE poses in the clip's pelvis frame.
    Returns:
        ``dp [N,K,3]`` (obs ``dif_local_rigid_body_pos_ee`` flattens to ``[N, 3K]``),
        ``rot6d [N,K,6]`` (obs ``dif_local_rigid_body_rot_ee`` flattens to ``[N, 6K]``),
        ``q_diff [N,K,4]`` xyzw (used by the EE rotation reward through :func:`quat_angle_xyzw`)."""
    p_cur, q_cur = ee_pose_in_root(root_pos_w, root_quat_w, ee_pos_w, ee_quat_w)
    dp = ee_delta_pos(p_cur, ref_pos_local)
    q_diff = ee_delta_quat(q_cur, ref_quat_local)
    return dp, rot6d_from_quat_xyzw(q_diff), q_diff


__all__ = [
    "ee_delta_pos",
    "ee_delta_quat",
    "ee_pose_in_root",
    "hero_ee_residual",
    "palm_point_world",
    "quat_angle_xyzw",
    "quat_conjugate_xyzw",
    "quat_mul_xyzw",
    "quat_rotate_inverse_xyzw",
    "quat_rotate_xyzw",
    "quat_to_matrix_xyzw",
    "rot6d_from_matrix",
    "rot6d_from_quat_xyzw",
]
