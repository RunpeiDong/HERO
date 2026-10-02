"""numpy quaternion / rotation helpers (batched over leading dims).

Convention: every quaternion inside ``sim2sim`` is **xyzw** (holosoma runtime convention) unless the name says
``wxyz``.  MuJoCo (``qpos[3:7]``, ``xquat``) and npz files store wxyz -> convert at the boundary with
:func:`wxyz_to_xyzw` / :func:`xyzw_to_wxyz`.  The formulas mirror ``holosoma.utils.rotations`` (``quat_mul``,
``quat_apply``, ``quaternion_to_matrix``, ``yaw_quat``, ``quat_to_angle_axis``) so the parity tests can compare
bit-for-bit up to float32 rounding.
"""

from __future__ import annotations

import numpy as np

_EPS = 1.0e-12


def wxyz_to_xyzw(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q[..., [1, 2, 3, 0]]


def xyzw_to_wxyz(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q[..., [3, 0, 1, 2]]


def quat_normalize(q: np.ndarray) -> np.ndarray:
    q = np.asarray(q, dtype=np.float64)
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), _EPS)


def quat_conj(q: np.ndarray) -> np.ndarray:
    """Conjugate (== inverse for unit quaternions), xyzw."""
    q = np.asarray(q, dtype=np.float64)
    return q * np.array([-1.0, -1.0, -1.0, 1.0])


quat_inv = quat_conj


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b`` of xyzw quaternions (broadcasting)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    ax, ay, az, aw = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bx, by, bz, bw = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
            aw * bw - ax * bx - ay * by - az * bz,
        ],
        axis=-1,
    )


def quat_apply(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors ``v`` (..., 3) by xyzw quaternions ``q`` (..., 4): ``R(q) v`` (holosoma ``quat_apply``)."""
    q = np.asarray(q, dtype=np.float64)
    v = np.asarray(v, dtype=np.float64)
    xyz = q[..., :3]
    w = q[..., 3:4]
    t = np.cross(xyz, v) * 2.0
    return v + w * t + np.cross(xyz, t)


def quat_apply_inv(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``R(q)^T v`` (holosoma ``quat_rotate_inverse`` / ``quat_apply(quat_inverse(q), v)``)."""
    return quat_apply(quat_conj(q), v)


def quat_to_mat(q: np.ndarray) -> np.ndarray:
    """Rotation matrices (..., 3, 3) from xyzw quaternions (holosoma ``quaternion_to_matrix(w_last=True)``)."""
    q = np.asarray(q, dtype=np.float64)
    i, j, k, r = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    two_s = 2.0 / np.maximum((q * q).sum(-1), _EPS)
    m = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    m[..., 0, 0] = 1 - two_s * (j * j + k * k)
    m[..., 0, 1] = two_s * (i * j - k * r)
    m[..., 0, 2] = two_s * (i * k + j * r)
    m[..., 1, 0] = two_s * (i * j + k * r)
    m[..., 1, 1] = 1 - two_s * (i * i + k * k)
    m[..., 1, 2] = two_s * (j * k - i * r)
    m[..., 2, 0] = two_s * (i * k - j * r)
    m[..., 2, 1] = two_s * (j * k + i * r)
    m[..., 2, 2] = 1 - two_s * (i * i + j * j)
    return m


def rot6d_from_quat(q: np.ndarray) -> np.ndarray:
    """First two matrix COLUMNS flattened row-major ``[m00 m01 m10 m11 m20 m21]`` -> (..., 6) (HERO rot6d)."""
    m = quat_to_mat(q)
    return m[..., :2].reshape(m.shape[:-2] + (6,))


def yaw_quat(q: np.ndarray) -> np.ndarray:
    """Heading-only quaternion (holosoma ``yaw_quat(w_last=True)``)."""
    q = np.asarray(q, dtype=np.float64)
    qx, qy, qz, qw = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    yaw = np.arctan2(2.0 * (qw * qz + qx * qy), 1.0 - 2.0 * (qy * qy + qz * qz))
    out = np.zeros_like(q)
    out[..., 2] = np.sin(yaw / 2.0)
    out[..., 3] = np.cos(yaw / 2.0)
    return quat_normalize(out)


def quat_angle(q: np.ndarray) -> np.ndarray:
    """Rotation angle in ``[0, pi]``: ``2 atan2(|xyz|, |w|)`` (``hero_isaacsim.utils.ee_residual.quat_angle_xyzw``;
    identical to holosoma ``quat_to_angle_axis`` after its ``w >= 0`` canonicalisation)."""
    q = np.asarray(q, dtype=np.float64)
    return 2.0 * np.arctan2(np.linalg.norm(q[..., :3], axis=-1), np.abs(q[..., 3]))


def quat_error_magnitude(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """holosoma ``quat_error_magnitude``: angle of ``q1 * conj(q2)``."""
    return quat_angle(quat_mul(q1, quat_conj(q2)))


def quat_from_euler_xyz(roll, pitch, yaw) -> np.ndarray:
    """xyzw quaternion from roll/pitch/yaw (holosoma ``quat_from_euler_xyz``; scalar or broadcastable arrays)."""
    roll, pitch, yaw = np.asarray(roll, dtype=np.float64), np.asarray(pitch, dtype=np.float64), np.asarray(yaw, dtype=np.float64)
    cy, sy = np.cos(yaw * 0.5), np.sin(yaw * 0.5)
    cr, sr = np.cos(roll * 0.5), np.sin(roll * 0.5)
    cp, sp = np.cos(pitch * 0.5), np.sin(pitch * 0.5)
    qw = cy * cr * cp + sy * sr * sp
    qx = cy * sr * cp - sy * cr * sp
    qy = cy * cr * sp + sy * sr * cp
    qz = sy * cr * cp - cy * sr * sp
    return np.stack([qx, qy, qz, qw], axis=-1)


def express_in_frame(p_w: np.ndarray, q_w: np.ndarray, p_ref: np.ndarray, q_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """World poses -> poses in the ``(p_ref, q_ref)`` frame: ``p = R(q_ref)^T (p_w - p_ref)``, ``q = q_ref^-1 * q_w``."""
    p_w = np.asarray(p_w, dtype=np.float64)
    p_ref = np.asarray(p_ref, dtype=np.float64)
    q_ref = np.asarray(q_ref, dtype=np.float64)
    return quat_apply_inv(q_ref, p_w - p_ref), quat_mul(quat_conj(q_ref), q_w)


__all__ = [
    "express_in_frame",
    "quat_angle",
    "quat_apply",
    "quat_apply_inv",
    "quat_conj",
    "quat_error_magnitude",
    "quat_from_euler_xyz",
    "quat_inv",
    "quat_mul",
    "quat_normalize",
    "quat_to_mat",
    "rot6d_from_quat",
    "wxyz_to_xyzw",
    "xyzw_to_wxyz",
    "yaw_quat",
]
