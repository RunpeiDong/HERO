"""HERO policy observations and optional delta-anchor feedback.

Residual end-effector poses are expressed in the reference pelvis frame.
Delta-anchor terms use future root references and reset-aware odometry noise.
Runtime quaternions use xyzw."""

from __future__ import annotations

import math
from typing import Any, Sequence

import torch

from holosoma.managers.observation.terms.wbt import get_base_ang_vel, get_base_lin_vel, get_projected_gravity

from hero_isaacsim.utils.training import training_noise_active
from hero_isaacsim.utils import hero_constants
from hero_isaacsim.utils.ee_residual import (
    hero_ee_residual,
    palm_point_world,
    quat_conjugate_xyzw,
    quat_mul_xyzw,
    quat_rotate_inverse_xyzw,
    quat_to_matrix_xyzw,
    rot6d_from_quat_xyzw,
)

_C = hero_constants()

# --------------------------------------------------------------------------------------------------
# accessors / cached env-side indices
# --------------------------------------------------------------------------------------------------


def _mc(env: Any):
    mc = env.command_manager.get_state("motion_command")
    assert mc is not None, "motion_command not found in command manager"
    return mc


def _zero_waist_when_walking(mc: Any, override: bool | None) -> bool:
    if override is not None:
        return bool(override)
    cfg = getattr(mc, "motion_cfg", None)
    return bool(getattr(cfg, "zero_waist_when_walking", False))


def _walking_mask(mc: Any) -> torch.Tensor:
    """[N] bool, True for walking envs (HERO ``commands[:, 4] == 1``)."""
    return mc.stand_flag.reshape(-1) > 0.5


def _resolve_body_indices(env: Any, names, attr: str, required: bool = True) -> torch.Tensor | None:
    """Indices of ``names`` in ``env.simulator.body_names`` (cached on ``env.<attr>``)."""
    cached = getattr(env, attr, None)
    if cached is not None:
        return cached
    body_names = list(env.simulator.body_names)
    if not all(n in body_names for n in names):
        if required:
            raise KeyError(f"bodies {names} not all present in simulator.body_names={body_names}")
        setattr(env, attr, None)
        return None
    idx = torch.tensor([body_names.index(n) for n in names], dtype=torch.long, device=env.device)
    setattr(env, attr, idx)
    return idx


def ee_body_indices(env: Any) -> torch.Tensor:
    """[2] long: ``[left_wrist_yaw_link, right_wrist_yaw_link]`` rigid-body indices (cached ``env.ee_body_indices``)."""
    return _resolve_body_indices(env, _C.EE_BODY_NAMES, "ee_body_indices")


def palm_body_indices(env: Any) -> torch.Tensor | None:
    """[2] long palm-link indices if the preset has them (34-body Dex3 preset), else None."""
    return _resolve_body_indices(env, _C.PALM_BODY_NAMES, "palm_body_indices", required=False)


def ee_palm_offset(env: Any) -> torch.Tensor:
    """[2, 3] palm point offset in the wrist_yaw frame (``PALM_OFFSET`` left, right); cached ``env.ee_palm_offset``."""
    cached = getattr(env, "ee_palm_offset", None)
    if cached is None:
        cached = torch.tensor(
            [list(_C.PALM_OFFSET["left"]), list(_C.PALM_OFFSET["right"])], dtype=torch.float, device=env.device
        )
        env.ee_palm_offset = cached
    return cached


def current_ee_pose_w(env: Any, ee_source: str = "wrist_offset") -> tuple[torch.Tensor, torch.Tensor]:
    """Current EE (palm point) poses in world: ``(pos [N,2,3], quat_xyzw [N,2,4])``, order [left, right]."""
    if ee_source == "palm_body":
        idx = palm_body_indices(env)
        if idx is None:
            raise KeyError("ee_source='palm_body' requires left/right_hand_palm_link in simulator.body_names")
        return env.simulator._rigid_body_pos[:, idx, :], env.simulator._rigid_body_rot[:, idx, :]
    if ee_source != "wrist_offset":
        raise ValueError(f"unknown ee_source {ee_source!r}")
    idx = ee_body_indices(env)
    wrist_pos = env.simulator._rigid_body_pos[:, idx, :]
    wrist_quat = env.simulator._rigid_body_rot[:, idx, :]
    return palm_point_world(wrist_pos, wrist_quat, ee_palm_offset(env)), wrist_quat


def hero_ee_reference(env: Any, zero_waist_when_walking: bool | None = None) -> tuple[torch.Tensor, torch.Tensor]:
    """Reference palm poses in the clip's pelvis frame ``(pos [N,2,3], quat_xyzw [N,2,4])``.

    With ``zero_waist_when_walking=True``, walking environments use the zero-waist FK reference.
    False keeps the full-waist reference; ``None`` reads ``motion_cfg.zero_waist_when_walking``.
    Idempotent w.r.t. a command term that already performed the swap."""
    mc = _mc(env)
    pos, quat = mc.ref_ee_pos_pelvis, mc.ref_ee_quat_pelvis
    if _zero_waist_when_walking(mc, zero_waist_when_walking):
        walking = _walking_mask(mc)[:, None, None]
        pos = torch.where(walking, mc.ref_ee_pos_pelvis_zero_waist, pos)
        quat = torch.where(walking, mc.ref_ee_quat_pelvis_zero_waist, quat)
    return pos, quat


def hero_ee_residual_from_env(
    env: Any, zero_waist_when_walking: bool | None = None, ee_source: str = "wrist_offset"
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """HERO ΔE from live env state: ``(dp [N,2,3], rot6d [N,2,6], q_diff xyzw [N,2,4])`` (shared by obs & reward)."""
    root = env.simulator.robot_root_states
    ee_pos_w, ee_quat_w = current_ee_pose_w(env, ee_source)
    ref_pos, ref_quat = hero_ee_reference(env, zero_waist_when_walking)
    return hero_ee_residual(root[:, 0:3], root[:, 3:7], ee_pos_w, ee_quat_w, ref_pos, ref_quat)


def _train_noise(env: Any, out: torch.Tensor, half_width: float, name: str) -> torch.Tensor:
    """``out + U(-w, w)`` (elementwise, out of place) while training; ``out`` itself when ``w == 0`` or evaluating."""
    w = float(half_width)
    if w < 0.0:
        raise ValueError(f"{name} must be >= 0 (uniform half-width), got {w}")
    if w == 0.0 or not training_noise_active(env):
        return out
    return out + (torch.rand_like(out) * 2.0 - 1.0) * w


def euler_roll_pitch_xyzw(q: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """(roll, pitch) in (-π, π] / [-π/2, π/2] from xyzw quaternions ``[N, 4]`` (ZYX intrinsic / xyz extrinsic)."""
    x, y, z, w = q.unbind(-1)
    roll = torch.atan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = torch.asin(torch.clamp(2.0 * (w * y - z * x), -1.0, 1.0))
    return roll, pitch


# --------------------------------------------------------------------------------------------------
# Actor terms (alphabetical == HERO order)
# --------------------------------------------------------------------------------------------------


def h00_actions(env: Any) -> torch.Tensor:
    """Last raw policy actions (29). HERO ``actions``."""
    return env.action_manager.action


def h01_base_ang_vel(env: Any, noise_rad_s: float = 0.0) -> torch.Tensor:
    """Pelvis angular velocity in the pelvis frame (3). HERO ``base_ang_vel`` (config scale 0.25).

    ``noise_rad_s``: training-only uniform ``+-`` noise (rad/s, before the 0.25 scale); 0.0 = bit-exact."""
    return _train_noise(env, get_base_ang_vel(env), noise_rad_s, "noise_rad_s")


def h02_command_ang_vel(env: Any) -> torch.Tensor:
    """Yaw-rate command wz (1). HERO ``command_ang_vel`` = commands[:, 2]."""
    return _mc(env).vel_cmd[:, 2:3]


def h03_command_base_height(env: Any) -> torch.Tensor:
    """Pelvis height command h_cmd (1). HERO ``command_base_height`` (config scale 2.0)."""
    return _mc(env).h_cmd.reshape(env.num_envs, 1)


def h04_command_lin_vel(env: Any) -> torch.Tensor:
    """Planar velocity command (vx, vy) in the heading frame (2). HERO ``command_lin_vel``."""
    return _mc(env).vel_cmd[:, 0:2]


def h05_command_stand(env: Any) -> torch.Tensor:
    """Walk flag (1): 1.0 = walking, 0.0 = stance (HERO ``command_stand`` — the name is inverted)."""
    return _mc(env).stand_flag.reshape(env.num_envs, 1)


def h06_command_waist_dofs(env: Any, zero_waist_when_walking: bool | None = None) -> torch.Tensor:
    """Reference waist (yaw, roll, pitch) (3). HERO ``command_waist_dofs``; zeroed for walking when ``zero_waist_when_walking`` is enabled."""
    mc = _mc(env)
    waist = mc.ref_upper_dof_pos[:, 0:3]
    if _zero_waist_when_walking(mc, zero_waist_when_walking):
        waist = torch.where(_walking_mask(mc)[:, None], torch.zeros_like(waist), waist)
    return waist


def h07_dif_local_rigid_body_pos_ee(
    env: Any, zero_waist_when_walking: bool | None = None, ee_source: str = "wrist_offset", noise_pos_m: float = 0.0
) -> torch.Tensor:
    """Δp = R_root^T(p_EE − p_root) − p_ref_local, [Lx,Ly,Lz,Rx,Ry,Rz] (6)..

    ``noise_pos_m``: training-only uniform ``+-`` noise (m) on the 6 position dims; 0.0 = bit-exact."""
    dp, _, _ = hero_ee_residual_from_env(env, zero_waist_when_walking, ee_source)
    return _train_noise(env, dp.reshape(env.num_envs, -1), noise_pos_m, "noise_pos_m")


def h08_dif_local_rigid_body_rot_ee(
    env: Any, zero_waist_when_walking: bool | None = None, ee_source: str = "wrist_offset", noise_rot6d: float = 0.0
) -> torch.Tensor:
    """6D(R_cur^T R_ref) per hand, first two columns flattened, [left(6), right(6)] (12)..

    ``noise_rot6d``: training-only uniform ``+-`` noise on the 12 rot6d entries; 0.0 = bit-exact."""
    _, rot6d, _ = hero_ee_residual_from_env(env, zero_waist_when_walking, ee_source)
    return _train_noise(env, rot6d.reshape(env.num_envs, -1), noise_rot6d, "noise_rot6d")


def h09_dof_pos(env: Any, noise_rad: float = 0.0) -> torch.Tensor:
    """q − q_default (29). HERO ``dof_pos``.  ``noise_rad``: training-only ``+-`` noise (rad)."""
    return _train_noise(env, env.simulator.dof_pos - env.default_dof_pos, noise_rad, "noise_rad")


def h10_dof_vel(env: Any, noise_rad_s: float = 0.0) -> torch.Tensor:
    """q̇ (29). HERO ``dof_vel`` (config scale 0.05).

    ``noise_rad_s``: training-only ``+-`` noise (rad/s, before the 0.05 scale)."""
    return _train_noise(env, env.simulator.dof_vel, noise_rad_s, "noise_rad_s")


def h11_projected_gravity(env: Any, noise_unit: float = 0.0) -> torch.Tensor:
    """R_root^T (0,0,-1) (3), with training-only uniform noise on each component.

    ``noise_unit`` is the half-width before observation scaling. This parameter
    is separate from holosoma's group-level noise, which also applies in evaluation."""
    return _train_noise(env, get_projected_gravity(env), noise_unit, "noise_unit")


def h12_ref_upper_dof_pos(env: Any, noise_rad: float = 0.0) -> torch.Tensor:
    """Reference ARM joint angles (14, absolute, holosoma order L arm 7 then R arm 7). HERO ``ref_upper_dof_pos``.

    ``noise_rad``: training-only ``+-`` noise on the reference (rad)."""
    return _train_noise(env, _mc(env).ref_upper_dof_pos[:, 3:17], noise_rad, "noise_rad")


def h13_roll_and_pitch(env: Any, noise_rad: float = 0.0) -> torch.Tensor:
    """``noise_rad``: training-only ``+-`` noise (rad)."""
    roll, pitch = euler_roll_pitch_xyzw(env.base_quat)
    return _train_noise(env, torch.stack([roll, pitch], dim=-1), noise_rad, "noise_rad")


# --------------------------------------------------------------------------------------------------
# Additional lower-body references from the clip
# --------------------------------------------------------------------------------------------------


def h14_ref_lower_dof_pos(env: Any, noise_rad: float = 0.0) -> torch.Tensor:
    """Reference LEG joint angles (12, absolute, holosoma order) from the clip.

    ``noise_rad``: training-only ``+-`` noise on the reference (rad)."""
    mc = _mc(env)
    idx = getattr(env, "hero_leg_dof_idx", None)
    if idx is None:
        idx = torch.tensor(list(_C.LEG_DOF_IDX), dtype=torch.long, device=env.device)
        env.hero_leg_dof_idx = idx
    return _train_noise(env, mc.joint_pos[:, idx], noise_rad, "noise_rad")


def h15_ref_root_pitch_roll(env: Any) -> torch.Tensor:
    """Reference pelvis Euler angles (2), ordered **(roll, pitch)** exactly like ``h13_roll_and_pitch`` so the two
    can be differenced directly."""
    roll, pitch = euler_roll_pitch_xyzw(_mc(env).root_quat_w)
    return torch.stack([roll, pitch], dim=-1)


def h16_ref_body_pos_b(
    env: Any,
    body_names: tuple[str, ...] = (
        "left_ankle_roll_link",
        "right_ankle_roll_link",
        "left_knee_link",
        "right_knee_link",
    ),
) -> torch.Tensor:
    """Reference ankle/knee positions relative to the reference pelvis, in the reference pelvis frame (12).

    ``body_names`` must be in ``motion_cfg.body_names_to_track``; output order follows ``body_names``."""
    mc = _mc(env)
    key = "_hero_ref_body_idx_" + "_".join(body_names)
    idx = getattr(mc, key, None)
    if idx is None:
        tracked = list(mc.motion_cfg.body_names_to_track)
        idx = torch.tensor([tracked.index(n) for n in body_names], dtype=torch.long, device=env.device)
        setattr(mc, key, idx)
    body_pos = mc.body_pos_w[:, idx, :]  # [N, B, 3] (world, incl. env origins — same as root_pos_w)
    root_pos = mc.root_pos_w[:, None, :]
    root_quat = mc.root_quat_w[:, None, :]
    pos_b = quat_rotate_inverse_xyzw(root_quat, body_pos - root_pos)
    return pos_b.reshape(env.num_envs, -1)


def _env_has_object_mask(env: Any, mc: Any) -> torch.Tensor:
    mask = getattr(mc, "env_has_object", None)
    if mask is not None:
        return mask.reshape(-1).to(torch.bool)
    has = bool(getattr(getattr(mc, "motion", None), "has_object", False))
    return torch.full((env.num_envs,), has, dtype=torch.bool, device=env.device)


def _object_pose_w(env: Any, mc: Any) -> tuple[torch.Tensor, torch.Tensor] | None:
    if not bool(getattr(getattr(mc, "motion", None), "has_object", False)):
        return None
    return mc.simulator_object_pos_w, mc.simulator_object_quat_w


def h17_obj_pos_b(env: Any) -> torch.Tensor:
    """Object position in the robot pelvis frame, R_root^T (p_obj − p_root), × env_has_object (3)."""
    mc = _mc(env)
    pose = _object_pose_w(env, mc)
    if pose is None:
        return torch.zeros(env.num_envs, 3, device=env.device)
    root = env.simulator.robot_root_states
    pos_b = quat_rotate_inverse_xyzw(root[:, 3:7], pose[0] - root[:, 0:3])
    return pos_b * _env_has_object_mask(env, mc)[:, None].float()


def h18_obj_ori_b(env: Any) -> torch.Tensor:
    """Object orientation relative to the robot pelvis as rot6d(q_root^-1 ⊗ q_obj), × env_has_object (6)."""
    mc = _mc(env)
    pose = _object_pose_w(env, mc)
    if pose is None:
        return torch.zeros(env.num_envs, 6, device=env.device)
    root_quat = env.simulator.robot_root_states[:, 3:7]
    rel = quat_mul_xyzw(quat_conjugate_xyzw(root_quat), pose[1])
    return rot6d_from_quat_xyzw(rel) * _env_has_object_mask(env, mc)[:, None].float()


def h19_has_object_flag(env: Any) -> torch.Tensor:
    """Per-env has-object flag (1)."""
    return _env_has_object_mask(env, _mc(env)).float().reshape(env.num_envs, 1)


# --------------------------------------------------------------------------------------------------
# Planar delta-anchor feedback (h20) and odometry noise
# --------------------------------------------------------------------------------------------------

#: Future reference frames of h20 in control steps (50 Hz: 0 / 0.1 / 0.2 / 0.3 / 0.4 s).  Mirrored torch-free in
#: ``config_values.observation.H20_FUTURE_STEPS``.
H20_FUTURE_STEPS: tuple[int, ...] = (0, 5, 10, 15, 20)
#: Per-frame width of h20: ``[dx, dy, sin(dyaw), cos(dyaw)]`` in the robot heading frame.
H20_FRAME_DIM = 4
H20_DIM = H20_FRAME_DIM * len(H20_FUTURE_STEPS)  # 20
#: Odometry-noise parameters of h20 (all ``noise*`` so provenance / ``noise_params`` pick them up); metres / radians.
H20_NOISE_PARAMS: tuple[str, ...] = (
    "noise_odom_bias_xy_m",
    "noise_odom_bias_yaw_rad",
    "noise_odom_walk_xy_m",
    "noise_odom_walk_yaw_rad",
)


H21_FRAME_DIM = 6
H21_DIM = H21_FRAME_DIM * len(H20_FUTURE_STEPS)  # 30
H22_FRAME_DIM = 1
H22_DIM = H22_FRAME_DIM * len(H20_FUTURE_STEPS)  # 5
H23_DIM = 3
#: Odometry-noise parameters of root orientation, height, and velocity terms (all ``noise*``): h21 roll / pitch bias + the h20 yaw knobs (shared yaw
#: channel), h22 height bias + walk, h23 per-call Gaussian velocity noise.
H21_NOISE_PARAMS: tuple[str, ...] = ("noise_odom_bias_roll_pitch_rad", "noise_odom_bias_yaw_rad", "noise_odom_walk_yaw_rad")
H22_NOISE_PARAMS: tuple[str, ...] = ("noise_odom_bias_z_m", "noise_odom_walk_z_m")
H23_NOISE_PARAMS: tuple[str, ...] = ("noise_odom_vel_m_s",)


def yaw_from_quat_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Heading (yaw) of xyzw quaternions ``[..., 4]``: ``atan2`` of the rotated x-axis (same as
    ``HeroMotionCommand._yaw_of`` / holosoma ``yaw_quat``): ``atan2(2(xy + wz), 1 - 2(y^2 + z^2))``."""
    x, y, z, w = q.unbind(-1)
    return torch.atan2(2.0 * (x * y + w * z), 1.0 - 2.0 * (y * y + z * z))


def ref_root_pose_heading_frame(
    robot_xy: torch.Tensor, robot_yaw: torch.Tensor, ref_xy: torch.Tensor, ref_yaw: torch.Tensor
) -> torch.Tensor:
    """Pure h20 math: ``[N, F, 4]`` = ``[R_yaw(robot)^T (p_ref - p_robot)_xy, sin(yaw_ref - yaw_robot), cos(...)]``."""
    d = ref_xy - robot_xy[:, None, :]
    c, s_ = torch.cos(robot_yaw)[:, None], torch.sin(robot_yaw)[:, None]
    dx = c * d[..., 0] + s_ * d[..., 1]
    dy = -s_ * d[..., 0] + c * d[..., 1]
    dyaw = ref_yaw - robot_yaw[:, None]
    return torch.stack([dx, dy, torch.sin(dyaw), torch.cos(dyaw)], dim=-1)


def rot6d_column_major(mat: torch.Tensor) -> torch.Tensor:
    """h21 flattening: the first two COLUMNS of ``mat [..., 3, 3]`` concatenated column-major -> ``[..., 6]`` =
    ``[R00, R10, R20, R01, R11, R21]`` (column 0 = the rotated x axis, column 1 = the rotated y axis; identity ->
    ``[1, 0, 0, 0, 1, 0]``).  NOT ``utils.ee_residual.rot6d_from_matrix`` (HERO's h08 / h18 row-major interleave
    ``[R00, R01, R10, R11, R20, R21]``, identity ``[1, 0, 0, 1, 0, 0]``)."""
    return mat[..., :, :2].transpose(-1, -2).reshape(mat.shape[:-2] + (6,))


def ref_root_rot6d_root_frame(robot_quat: torch.Tensor, ref_quat: torch.Tensor) -> torch.Tensor:
    """Pure h21 math: ``[N, F, 6]`` = :func:`rot6d_column_major` of ``R_rel = R(q_robot)^T R(q_ref) = R(q_robot^-1 ⊗ q_ref)`` --
    the reference root rotation expressed in the robot ROOT frame (full 3D incl. roll / pitch / yaw); identity ->
    ``[1, 0, 0, 0, 1, 0]``.  ``robot_quat [N, 4]`` (the robot's orientation estimate), ``ref_quat [N, F, 4]``, both xyzw.
    Invariant to a common rigid rotation of both frames; sim2sim rebuilds h21 with it (mind MuJoCo's wxyz)."""
    rel = quat_mul_xyzw(quat_conjugate_xyzw(robot_quat)[:, None, :], ref_quat)
    return rot6d_column_major(quat_to_matrix_xyzw(rel))


def ref_root_height_rel(robot_z: torch.Tensor, ref_z: torch.Tensor) -> torch.Tensor:
    """Pure h22 math: ``[N, F]`` = ``z_ref - z_robot`` (``robot_z [N]`` the robot's height estimate, ``ref_z [N, F]`` the
    reference root heights in the SAME frame, so a common vertical offset -- env origin -- cancels)."""
    return ref_z - robot_z[:, None]


def lin_vel_heading_frame(quat: torch.Tensor, lin_vel_w: torch.Tensor) -> torch.Tensor:
    """Pure h23 math: ``[N, 3]`` = ``R_yaw(quat)^T v_w`` -- the world-frame root velocity rotated into the robot HEADING frame
    (yaw of :func:`yaw_from_quat_xyzw` only: roll / pitch do not enter, z is kept)."""
    yaw = yaw_from_quat_xyzw(quat)
    c, s_ = torch.cos(yaw), torch.sin(yaw)
    vx, vy, vz = lin_vel_w.unbind(-1)
    return torch.stack([c * vx + s_ * vy, -s_ * vx + c * vy, vz], dim=-1)


def _quat_about_axis_xyzw(axis: int, angle: torch.Tensor) -> torch.Tensor:
    """xyzw quaternion ``[N, 4]`` of a rotation by ``angle [N]`` about axis ``axis`` (0 = x, 1 = y, 2 = z)."""
    q = torch.zeros(angle.shape[0], 4, dtype=angle.dtype, device=angle.device)
    q[:, axis] = torch.sin(0.5 * angle)
    q[:, 3] = torch.cos(0.5 * angle)
    return q


def perturbed_root_quat_xyzw(q: torch.Tensor, droll: torch.Tensor, dpitch: torch.Tensor, dyaw: torch.Tensor) -> torch.Tensor:
    """Robot orientation estimate ``q_z(dyaw) ⊗ q ⊗ q_y(dpitch) ⊗ q_x(droll)`` (xyzw ``[N, 4]``; angles ``[N]``): the yaw
    error is a WORLD-frame rotation about +z (odometry heading drift -- the same ``dyaw`` :meth:`HeroOdometryNoiseState.perturb`
    adds to the h20 yaw), the roll / pitch error a BODY-frame tilt (IMU attitude error) in ZYX order, so for an upright robot
    the estimate's Euler roll / pitch (:func:`euler_roll_pitch_xyzw`) are ``droll`` / ``dpitch`` exactly."""
    out = quat_mul_xyzw(_quat_about_axis_xyzw(2, dyaw), q)
    tilt = quat_mul_xyzw(_quat_about_axis_xyzw(1, dpitch), _quat_about_axis_xyzw(0, droll))
    return quat_mul_xyzw(out, tilt)


def _clip_last_frame_index(mc: Any) -> torch.Tensor:
    """[N] last valid absolute frame of every env's current clip (``motion_end_idx`` is exclusive)."""
    ts = mc.time_steps
    end_idx = mc.motion.motion_end_idx.to(ts.device)
    motion_ids = getattr(mc, "motion_ids", None)
    if motion_ids is None or end_idx.numel() == 1:
        per_env_end = end_idx.reshape(-1)[:1].expand(ts.shape[0])
    else:
        per_env_end = end_idx[motion_ids.to(ts.device)]
    return per_env_end - 1


def _check_future_steps(future_steps: Sequence[int]) -> list[int]:
    steps = [int(s) for s in future_steps]
    if not steps or any(s < 0 for s in steps):
        raise ValueError(f"future_steps must be a non-empty sequence of ints >= 0, got {list(future_steps)!r}")
    return steps


H20_FRAME_CACHE_ATTR = "_hero_h20_frame_cache"
"""Attribute on the motion command holding the per-step cache of :func:`hero_ref_root_future_pose_w` (see there)."""


def _h20_cache_key(mc: Any, steps: Sequence[int]) -> tuple | None:
    """Invalidation key of the h20 reference gather, or None when it cannot be signalled safely.

    ``(id(time_steps), time_steps._version, id(motion), timeline versions, steps)`` -- the ``MotionCommand._motion_frames``
    recipe (holosoma wbt.py): ``time_steps`` is rebound by ``init_buffers`` (identity) and written in place once per env
    step / reset (version), so both actor and critic copies of h20 within one step share the gather while any mutation
    invalidates it.  The timeline tensors' versions are included WHEN they track one (ordinary tensors: the mocks / tools
    that edit ``motion.body_pos_w`` in place) and skipped when they are inference tensors (the corpus built under Isaac's
    ``torch.inference_mode`` setup; immutable during training by contract).  An inference-tensor ``time_steps`` (no version
    counter) disables the cache -- every call gathers fresh, exactly like the stock cache."""
    ts = mc.time_steps
    motion = mc.motion
    try:
        key: tuple = (id(ts), ts._version, id(motion), tuple(int(s) for s in steps))
    except RuntimeError:  # inference tensor: no version counter, no safe invalidation signal
        return None
    try:
        return key + (motion.body_pos_w._version, motion.body_quat_w._version)
    except (RuntimeError, AttributeError):  # inference-tensor / exotic timelines: keyed on time_steps only (stock behaviour)
        return key


def _gather_ref_root_future_pose(mc: Any, steps: Sequence[int]) -> tuple[torch.Tensor, torch.Tensor]:
    """The uncached gather: ``(pos [N, F, 3] WITHOUT env origins, quat_xyzw [N, F, 4])`` of npz body 0 at ``min(t + s, clip_last)``."""
    ts = mc.time_steps
    n, f = ts.shape[0], len(steps)
    offs = torch.tensor(list(steps), dtype=ts.dtype, device=ts.device)
    idx = torch.minimum(ts[:, None] + offs[None, :], _clip_last_frame_index(mc)[:, None])  # [N, F]
    flat = idx.reshape(-1)
    pos = mc.motion.frames("body_pos_w", flat)[:, 0].reshape(n, f, 3)
    quat = mc.motion.frames("body_quat_w", flat)[:, 0].reshape(n, f, 4)
    return pos, quat


def reference_origins(env: Any, mc: Any = None) -> torch.Tensor:
    """Reference origins."""
    mc = _mc(env) if mc is None else mc
    origins = getattr(mc, "reference_origins", None)
    if origins is not None and torch.is_tensor(origins) and origins.ndim == 2 and origins.shape[-1] == 3:
        return origins
    return env.simulator.scene.env_origins


def hero_ref_root_future_pos_clip(env: Any, future_steps: Sequence[int] = H20_FUTURE_STEPS) -> tuple[torch.Tensor, torch.Tensor]:
    """The cached future-frame gather WITHOUT any origin: ``(pos_clip [N, F, 3] in the clip frame, quat_xyzw [N, F, 4])`` of npz body 0
    at ``min(time_steps + s, clip_last)``.  Same cache as :func:`hero_ref_root_future_pose_w` (read-only by contract).  For consumers
    that add their own per-frame origin: critic future frames query the terrain at each frame's reference xy, while the actor
    keeps the shared current-frame dz of :func:`hero_ref_root_future_pose_w`."""
    mc = _mc(env)
    steps = _check_future_steps(future_steps)
    return _cached_ref_root_future_pose(mc, steps)


def _cached_ref_root_future_pose(mc: Any, steps: tuple[int, ...]) -> tuple[torch.Tensor, torch.Tensor]:
    """The per-step cached gather behind :func:`hero_ref_root_future_pose_w` (no origins added)."""
    key = _h20_cache_key(mc, steps)
    if key is None:
        return _gather_ref_root_future_pose(mc, steps)
    cache = getattr(mc, H20_FRAME_CACHE_ATTR, None)
    if not isinstance(cache, dict):
        cache = {}
        try:
            setattr(mc, H20_FRAME_CACHE_ATTR, cache)
        except AttributeError:  # slotted / frozen command double: no cache
            cache = None
    if cache is not None and cache.get("key") == key:
        return cache["pos"], cache["quat"]
    pos, quat = _gather_ref_root_future_pose(mc, steps)
    if cache is not None:
        cache.clear()
        cache.update(key=key, pos=pos, quat=quat)
    return pos, quat


def hero_ref_root_future_pose_w(env: Any, future_steps: Sequence[int] = H20_FUTURE_STEPS) -> tuple[torch.Tensor, torch.Tensor]:
    """Cached per env step on the motion command (:data:`H20_FRAME_CACHE_ATTR`, key :func:`_h20_cache_key`): the gather
    reads ``N x F`` full 32-body rows of the (CPU-resident / mmapped) corpus, and h20 is computed twice per step (actor
    copy with odometry noise + critic copy), so the second call -- and holosoma's pre-reset final-observation pass --
    reuse the first gather instead of re-reading ~10x the stock per-step body traffic.  The cached tensors are read-only
    by contract (consumers must ``.clone()`` before mutating; the h20 math only reads them).  The reference origins are added
    per call (never cached), so a moved origin is always honoured."""
    mc = _mc(env)
    steps = _check_future_steps(future_steps)
    pos, quat = _cached_ref_root_future_pose(mc, steps)
    return pos + reference_origins(env, mc)[:, None, :], quat


class _OdomNoiseChannel:
    """One odometry-error channel of :class:`HeroOdometryNoiseState`: per-episode bias ``[N, d]`` (re-sampled for every env whose
    ``episode_length_buf`` is 0 when the channel is advanced = the first observation after ANY reset, consecutive first-step
    failures included -- a clip rollover does not, odometry is continuous on a real robot), random walk ``[N, d]`` with Wiener
    increments ``N(0, w^2 dt)`` once per CONTROL STEP, and its OWN clock, so channels advanced by different terms (h20: xy +
    yaw, h21: yaw + rp, h22: z) never double-step and a shared channel (yaw) advances once per step whichever term runs first."""

    def __init__(
        self,
        num_envs: int,
        dim: int,
        device: Any,
        bias_std: float,
        ep_len: torch.Tensor | None,
        step_counter: int | None = None,
        reset_generation: torch.Tensor | None = None,
    ):
        with torch.inference_mode(False):
            self.bias = torch.randn(num_envs, dim, device=device) * float(bias_std)
            self.walk = torch.zeros(num_envs, dim, device=device)
            self.last_ep_len = (
                ep_len.detach().clone().to(device=device, dtype=torch.long)
                if ep_len is not None
                else torch.zeros(num_envs, dtype=torch.long, device=device)
            )
            # the constructor's sample IS this step's bias: no walk step and no re-sample until the counter moves on
            self.last_step_counter: int | None = None if step_counter is None else int(step_counter)
            self.last_fresh_step = torch.full(
                (num_envs,), -1 if step_counter is None else int(step_counter), dtype=torch.long, device=device
            )
            # the reset generation this channel is synchronised to: the constructor's sample IS the current generation's bias,
            # so a reset marked before the channel existed is not re-sampled a second time
            self.last_gen = (
                reset_generation.detach().clone().to(device=device, dtype=torch.long)
                if reset_generation is not None
                else torch.zeros(num_envs, dtype=torch.long, device=device)
            )
        self.bias_std = float(bias_std)

    def _resample(self, fresh: torch.Tensor, bias_std: float) -> None:
        """New bias ``N(0, bias_std^2)`` and a zero walk for the envs ``fresh`` (bool ``[N]``)."""
        if bool(fresh.any()):
            k = int(fresh.sum())
            self.bias[fresh] = torch.randn(k, self.bias.shape[1], device=self.bias.device) * float(bias_std)
            self.walk[fresh] = 0.0

    def _reset_marked(self, reset_generation: torch.Tensor | None) -> torch.Tensor | None:
        """Bool ``[N]``: the envs whose reset generation moved since this channel last synchronised (None without one)."""
        if reset_generation is None:
            return None
        return reset_generation.to(device=self.last_gen.device, dtype=torch.long) != self.last_gen

    def advance(
        self,
        ep_len: torch.Tensor | None,
        dt: float,
        bias_std: float,
        walk_std: float,
        step_counter: int | None = None,
        reset_generation: torch.Tensor | None = None,
    ) -> None:
        """One control step: when ``step_counter`` differs from the last advance, every env gets one Wiener increment
        ``N(0, walk_std^2 dt)`` on the walk; every env with ``ep_len == 0`` (fresh episode) whose bias was not already re-sampled
        this step, and every env whose ``reset_generation`` moved since this channel's last advance (an explicit reset --
        ``reset_all()`` included, whatever ``ep_len`` shows), gets a new bias ``N(0, bias_std^2)`` and a zero walk.  A channel
        created bias-free by another term (e.g. h21 before h20 in the very first step) samples its bias the first time a positive
        ``bias_std`` arrives.  Without a ``step_counter`` (or on a channel created without one) the legacy
        ``episode_length_buf``-changed detector runs instead (the generation rule still applies there)."""
        n, d = self.bias.shape
        dev = self.bias.device
        with torch.inference_mode(False):
            if float(bias_std) > 0.0 and self.bias_std == 0.0:
                self.bias = torch.randn(n, d, device=dev) * float(bias_std)
                self.bias_std = float(bias_std)
            marked = self._reset_marked(reset_generation)
            if ep_len is None:
                # no episode clock at all (bare doubles): only an explicit reset starts a new episode
                if marked is not None:
                    self._resample(marked, bias_std)
                    self.last_gen.copy_(reset_generation)
                return
            ep_len = ep_len.to(device=self.last_ep_len.device, dtype=torch.long)
            if step_counter is None or self.last_step_counter is None:
                self._advance_legacy(ep_len, dt, bias_std, walk_std, marked)
            else:
                step = int(step_counter)
                if step != self.last_step_counter:
                    if float(walk_std) > 0.0:
                        self.walk += torch.randn(n, d, device=dev) * (float(walk_std) * math.sqrt(float(dt)))
                    self.last_step_counter = step
                    self.last_ep_len.copy_(ep_len)
                fresh = (ep_len == 0) & (self.last_fresh_step != step)
                if marked is not None:
                    fresh |= marked
                self._resample(fresh, bias_std)
                self.last_fresh_step[fresh] = step
            if marked is not None:
                self.last_gen.copy_(reset_generation)

    def _advance_legacy(
        self, ep_len: torch.Tensor, dt: float, bias_std: float, walk_std: float, marked: torch.Tensor | None = None
    ) -> None:
        """Advance legacy."""
        n, d = self.bias.shape
        dev = self.bias.device
        changed = ep_len != self.last_ep_len
        fresh = changed & (ep_len == 0)
        if marked is not None:
            fresh = fresh | marked
        if not bool(changed.any()) and not bool(fresh.any()):
            return
        if float(walk_std) > 0.0 and bool(changed.any()):
            self.walk += changed[:, None] * torch.randn(n, d, device=dev) * (float(walk_std) * math.sqrt(float(dt)))
        self._resample(fresh, bias_std)
        self.last_ep_len.copy_(ep_len)

    @property
    def error(self) -> torch.Tensor:
        """``bias + walk`` ``[N, d]``: the odometry error currently added to the true quantity."""
        return self.bias + self.walk


class HeroOdometryNoiseState:
    """Per-env odometry-error state shared by the delta-anchor terms (``env.hero_odom_noise``): channels ``xy`` ``[N, 2]`` and
    ``yaw`` ``[N, 1]`` (h20; ``yaw`` also rotates the h21 reference), ``rp`` ``[N, 2]`` (h21 roll / pitch, bias only) and ``z``
    ``[N, 1]`` (h22), each a :class:`_OdomNoiseChannel` (per-episode bias + Wiener walk + own control-step clock).  ``xy`` /
    ``yaw`` are created with the state (the h20 constructor signature), ``rp`` / ``z`` on first use (:meth:`channel`).
    ``step_counter`` = ``env.control_step_counter`` at creation (None -> legacy ``episode_length_buf`` detector).
    ``reset_generation [N]`` counts the explicit resets of every env (:meth:`mark_reset`, called from
    ``HeroTrackingManager._reset_buffers_callback`` through :func:`mark_odom_noise_reset`); each channel re-samples the envs whose
    generation moved since its own last advance, including when ``reset_all()`` first observes ``episode_length_buf == 1``. Legacy h20
    views: ``bias_xy`` / ``walk_xy`` ``[N, 2]``, ``bias_yaw`` / ``walk_yaw`` ``[N]``, ``last_ep_len`` = the xy channel's
    ``episode_length_buf`` snapshot; :meth:`advance` / :meth:`perturb` are the h20 (xy + yaw) operations."""

    CHANNEL_DIMS: dict[str, int] = {"xy": 2, "yaw": 1, "rp": 2, "z": 1}

    def __init__(
        self,
        num_envs: int,
        device: Any,
        bias_xy_m: float = 0.0,
        bias_yaw_rad: float = 0.0,
        ep_len: torch.Tensor | None = None,
        step_counter: int | None = None,
    ):
        self._num_envs = int(num_envs)
        self.device = device
        with torch.inference_mode(False):


            self.reset_generation = torch.zeros(self._num_envs, dtype=torch.long, device=device)
        self._channels: dict[str, _OdomNoiseChannel] = {}
        self.channel("xy", bias_xy_m, ep_len, step_counter)
        self.channel("yaw", bias_yaw_rad, ep_len, step_counter)

    @property
    def num_envs(self) -> int:
        return self._num_envs

    def channel(
        self, name: str, bias_std: float = 0.0, ep_len: torch.Tensor | None = None, step_counter: int | None = None
    ) -> _OdomNoiseChannel:
        """The channel ``name`` (:data:`CHANNEL_DIMS`), created on first use with ``bias_std``, the ``ep_len`` snapshot and the
        current ``step_counter`` (the creation step counts as advanced)."""
        ch = self._channels.get(name)
        if ch is None:
            ch = _OdomNoiseChannel(
                self._num_envs, self.CHANNEL_DIMS[name], self.device, bias_std, ep_len, step_counter, self.reset_generation
            )
            self._channels[name] = ch
        return ch

    def has_channel(self, name: str) -> bool:
        return name in self._channels

    def mark_reset(self, env_ids=None) -> int:
        """Record an explicit RESET of ``env_ids`` (None = every env; long indices, a bool mask or any index-like): their
        ``reset_generation`` moves by one, so every channel re-samples their bias and zeroes their walk on its next advance --
        whatever ``episode_length_buf`` shows at that time (``reset_all()`` materialises its observation after one step, so the
        terms see 1).  Marking twice before an advance is one new episode.  Returns the number of indices marked (0 for an empty
        selection).  Not called for clip-rollover soft resets: odometry is continuous on a real robot."""
        with torch.inference_mode(False):
            gen = self.reset_generation
            if env_ids is None:
                gen += 1
                return self._num_envs
            if isinstance(env_ids, torch.Tensor) and env_ids.dtype == torch.bool:
                ids = env_ids.to(device=gen.device).nonzero(as_tuple=False).flatten()
            else:
                ids = torch.as_tensor(env_ids, dtype=torch.long, device=gen.device).reshape(-1)
            if ids.numel() == 0:
                return 0
            gen[ids] += 1
            return int(ids.numel())

    # -- legacy h20 views ---------------------------------------------------------------------------------------
    @property
    def bias_xy(self) -> torch.Tensor:
        return self._channels["xy"].bias

    @property
    def walk_xy(self) -> torch.Tensor:
        return self._channels["xy"].walk

    @property
    def bias_yaw(self) -> torch.Tensor:
        return self._channels["yaw"].bias[:, 0]

    @property
    def walk_yaw(self) -> torch.Tensor:
        return self._channels["yaw"].walk[:, 0]

    @property
    def last_ep_len(self) -> torch.Tensor:
        return self._channels["xy"].last_ep_len

    # -- orientation and height views ------------------------------------------------------------------------------------------
    @property
    def bias_rp(self) -> torch.Tensor:
        """``[N, 2]`` roll / pitch bias of the robot attitude estimate (h21; zeros until the rp channel is advanced)."""
        return self.channel("rp").bias

    @property
    def bias_z(self) -> torch.Tensor:
        return self.channel("z").bias[:, 0]

    @property
    def walk_z(self) -> torch.Tensor:
        return self.channel("z").walk[:, 0]

    # -- operations ---------------------------------------------------------------------------------------------
    def advance_channel(
        self,
        name: str,
        ep_len: torch.Tensor | None,
        dt: float,
        bias_std: float,
        walk_std: float = 0.0,
        step_counter: int | None = None,
    ) -> None:
        """Advance ``name`` by one control step when ``step_counter`` moved since that channel's last advance, re-sampling the
        bias of every env with ``ep_len == 0`` once per step and of every env reset since that channel's last advance
        (:meth:`mark_reset`; :meth:`_OdomNoiseChannel.advance`)."""
        self.channel(name, bias_std, ep_len, step_counter).advance(
            ep_len, dt, bias_std, walk_std, step_counter, reset_generation=self.reset_generation
        )

    def advance(
        self,
        ep_len: torch.Tensor | None,
        dt: float,
        bias_xy_m: float,
        bias_yaw_rad: float,
        walk_xy_m: float,
        walk_yaw_rad: float,
        step_counter: int | None = None,
    ) -> None:
        """h20: advance the ``xy`` and ``yaw`` channels (see :meth:`_OdomNoiseChannel.advance`)."""
        self.advance_channel("xy", ep_len, dt, bias_xy_m, walk_xy_m, step_counter)
        self.advance_channel("yaw", ep_len, dt, bias_yaw_rad, walk_yaw_rad, step_counter)

    def yaw_error(self) -> torch.Tensor:
        """``[N]`` the yaw error shared by h20 and h21 (bias + walk of the ``yaw`` channel)."""
        return self._channels["yaw"].error[:, 0]

    def perturb(self, xy: torch.Tensor, yaw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """h20 odometry estimate = true pose + bias + walk (world xy, yaw)."""
        return xy + self._channels["xy"].error, yaw + self.yaw_error()

    def perturb_quat(self, q: torch.Tensor) -> torch.Tensor:
        """h21 orientation estimate: :func:`perturbed_root_quat_xyzw` with the ``rp`` bias and the shared ``yaw`` error."""
        rp = self.bias_rp
        return perturbed_root_quat_xyzw(q, rp[:, 0], rp[:, 1], self.yaw_error())

    def perturb_z(self, z: torch.Tensor) -> torch.Tensor:
        """h22 height estimate = true root z + bias + walk of the ``z`` channel."""
        return z + self.channel("z").error[:, 0]


def _control_step_counter(env: Any) -> int | None:
    """``env.control_step_counter`` as an int (``HeroTrackingManager``: +1 per ``_post_physics_step`` before termination /
    reset / observations), or None when the env lacks it -> the channels fall back to the legacy
    ``episode_length_buf``-changed detector."""
    c = getattr(env, "control_step_counter", None)
    return None if c is None else int(c)


def _odom_noise_state(env: Any, bias_xy_m: float, bias_yaw_rad: float) -> HeroOdometryNoiseState:
    st = getattr(env, "hero_odom_noise", None)
    if st is None or st.num_envs != int(env.num_envs):
        st = HeroOdometryNoiseState(
            int(env.num_envs),
            env.device,
            bias_xy_m,
            bias_yaw_rad,
            getattr(env, "episode_length_buf", None),
            _control_step_counter(env),
        )
        env.hero_odom_noise = st
    return st


def mark_odom_noise_reset(env: Any, env_ids=None) -> int:
    """Mark odom noise reset."""
    st = getattr(env, "hero_odom_noise", None)
    if st is None or int(st.num_envs) != int(env.num_envs):
        return 0
    return st.mark_reset(env_ids)


def _check_noise_stds(**stds: float) -> bool:
    """Validate the odometry-noise knobs (``>= 0``; ``ValueError`` otherwise) and return True when any is ``> 0``."""
    for name, value in stds.items():
        if float(value) < 0.0:
            raise ValueError(f"{name} must be >= 0 (std), got {value}")
    return any(float(v) > 0.0 for v in stds.values())


def _train_gaussian_noise(env: Any, out: torch.Tensor, std: float, name: str) -> torch.Tensor:
    """``out + N(0, std^2)`` (elementwise, out of place, drawn per call) while training; ``out`` itself when ``std == 0`` or
    evaluating -- the Gaussian counterpart of :func:`_train_noise` (h23 odometry-velocity noise).  Negative stds raise."""
    s = float(std)
    if s < 0.0:
        raise ValueError(f"{name} must be >= 0 (std), got {s}")
    if s == 0.0 or not training_noise_active(env):
        return out
    return out + torch.randn_like(out) * s


def robot_odometry_pose(
    env: Any,
    noise_odom_bias_xy_m: float = 0.0,
    noise_odom_bias_yaw_rad: float = 0.0,
    noise_odom_walk_xy_m: float = 0.0,
    noise_odom_walk_yaw_rad: float = 0.0,
) -> tuple[torch.Tensor, torch.Tensor]:
    """The robot root ``(xy [N, 2], yaw [N])`` as h20 sees it: exact, or -- while training with a non-zero knob --
    perturbed by the per-episode bias + random walk of :class:`HeroOdometryNoiseState` (``xy`` / ``yaw`` channels, advanced here)."""
    active = _check_noise_stds(
        noise_odom_bias_xy_m=noise_odom_bias_xy_m,
        noise_odom_bias_yaw_rad=noise_odom_bias_yaw_rad,
        noise_odom_walk_xy_m=noise_odom_walk_xy_m,
        noise_odom_walk_yaw_rad=noise_odom_walk_yaw_rad,
    )
    rs = env.simulator.robot_root_states
    xy, yaw = rs[:, 0:2], yaw_from_quat_xyzw(rs[:, 3:7])
    if not active or not training_noise_active(env):
        return xy, yaw
    st = _odom_noise_state(env, noise_odom_bias_xy_m, noise_odom_bias_yaw_rad)
    st.advance(
        getattr(env, "episode_length_buf", None),
        float(getattr(env, "dt", 0.02)),
        noise_odom_bias_xy_m,
        noise_odom_bias_yaw_rad,
        noise_odom_walk_xy_m,
        noise_odom_walk_yaw_rad,
        _control_step_counter(env),
    )
    return st.perturb(xy, yaw)


def h20_ref_root_pose_b(
    env: Any,
    future_steps: Sequence[int] = H20_FUTURE_STEPS,
    noise_odom_bias_xy_m: float = 0.0,
    noise_odom_bias_yaw_rad: float = 0.0,
    noise_odom_walk_xy_m: float = 0.0,
    noise_odom_walk_yaw_rad: float = 0.0,
) -> torch.Tensor:
    """Delta-anchor drift feedback (20): reference ROOT pose relative to the robot root in the robot HEADING frame for the
    frames ``future_steps`` ahead, ``[dx, dy, sin(dyaw), cos(dyaw)]`` per frame.  ``[0, 0, 0, 1]``
    per frame == on track.  The ``noise_odom_*`` knobs (std, m / rad; 0.0 = exact) perturb the ROBOT pose estimate
    (:func:`robot_odometry_pose`) while training only."""
    pos_w, quat_w = hero_ref_root_future_pose_w(env, future_steps)
    robot_xy, robot_yaw = robot_odometry_pose(
        env, noise_odom_bias_xy_m, noise_odom_bias_yaw_rad, noise_odom_walk_xy_m, noise_odom_walk_yaw_rad
    )
    out = ref_root_pose_heading_frame(robot_xy, robot_yaw, pos_w[..., 0:2], yaw_from_quat_xyzw(quat_w))
    return out.reshape(env.num_envs, -1)


# --------------------------------------------------------------------------------------------------
# Root orientation, height, and velocity feedback (h21 / h22 / h23)
# --------------------------------------------------------------------------------------------------


def robot_odometry_quat(
    env: Any,
    noise_odom_bias_roll_pitch_rad: float = 0.0,
    noise_odom_bias_yaw_rad: float = 0.0,
    noise_odom_walk_yaw_rad: float = 0.0,
) -> torch.Tensor:
    """The robot root orientation (xyzw ``[N, 4]``) as h21 sees it: exact, or -- while training with a non-zero knob -- the
    estimate :func:`perturbed_root_quat_xyzw` built from the shared :class:`HeroOdometryNoiseState`: the ``yaw`` channel (bias
    + walk -- the SAME error h20 adds to its yaw) and the ``rp`` channel (per-episode roll / pitch bias, no walk: IMU-grade)."""
    active = _check_noise_stds(
        noise_odom_bias_roll_pitch_rad=noise_odom_bias_roll_pitch_rad,
        noise_odom_bias_yaw_rad=noise_odom_bias_yaw_rad,
        noise_odom_walk_yaw_rad=noise_odom_walk_yaw_rad,
    )
    q = env.simulator.robot_root_states[:, 3:7]
    if not active or not training_noise_active(env):
        return q
    st = _odom_noise_state(env, 0.0, noise_odom_bias_yaw_rad)
    ep_len, dt, step = getattr(env, "episode_length_buf", None), float(getattr(env, "dt", 0.02)), _control_step_counter(env)
    st.advance_channel("yaw", ep_len, dt, noise_odom_bias_yaw_rad, noise_odom_walk_yaw_rad, step)
    st.advance_channel("rp", ep_len, dt, noise_odom_bias_roll_pitch_rad, 0.0, step)
    return st.perturb_quat(q)


def robot_odometry_height(env: Any, noise_odom_bias_z_m: float = 0.0, noise_odom_walk_z_m: float = 0.0) -> torch.Tensor:
    """The robot root height ``[N]`` (world z) as h22 sees it: exact, or -- while training with a non-zero knob -- perturbed
    by the ``z`` channel of :class:`HeroOdometryNoiseState` (per-episode bias + random walk, advanced here)."""
    active = _check_noise_stds(noise_odom_bias_z_m=noise_odom_bias_z_m, noise_odom_walk_z_m=noise_odom_walk_z_m)
    z = env.simulator.robot_root_states[:, 2]
    if not active or not training_noise_active(env):
        return z
    st = _odom_noise_state(env, 0.0, 0.0)
    st.advance_channel(
        "z",
        getattr(env, "episode_length_buf", None),
        float(getattr(env, "dt", 0.02)),
        noise_odom_bias_z_m,
        noise_odom_walk_z_m,
        _control_step_counter(env),
    )
    return st.perturb_z(z)


def h21_ref_root_rot_b(
    env: Any,
    future_steps: Sequence[int] = H20_FUTURE_STEPS,
    noise_odom_bias_roll_pitch_rad: float = 0.0,
    noise_odom_bias_yaw_rad: float = 0.0,
    noise_odom_walk_yaw_rad: float = 0.0,
) -> torch.Tensor:
    """Root-orientation feedback (30): reference ROOT orientation relative to the robot root, ``R_rel = R_robot^T R_ref`` as its first two
    columns concatenated COLUMN-MAJOR ``[R00, R10, R20, R01, R11, R21]`` (:func:`rot6d_column_major`; identity ->
    ``[1, 0, 0, 0, 1, 0]`` per frame; NOT the h08 / h18 row-major interleave), for the frames ``future_steps`` ahead.  Realisable on the robot from the IMU roll / pitch + odometry yaw
    and the clip aligned at episode start.  The ``noise_odom_*`` knobs (std, rad; 0.0 = exact) perturb the ROBOT orientation
    estimate (:func:`robot_odometry_quat`) while training only; the yaw knobs must equal h20's (one shared yaw channel)."""
    _, quat_w = hero_ref_root_future_pose_w(env, future_steps)
    q_robot = robot_odometry_quat(env, noise_odom_bias_roll_pitch_rad, noise_odom_bias_yaw_rad, noise_odom_walk_yaw_rad)
    return ref_root_rot6d_root_frame(q_robot, quat_w).reshape(env.num_envs, -1)


def h22_ref_root_height_b(
    env: Any, future_steps: Sequence[int] = H20_FUTURE_STEPS, noise_odom_bias_z_m: float = 0.0, noise_odom_walk_z_m: float = 0.0
) -> torch.Tensor:
    """Root-height feedback (5): ``z_ref_root - z_robot_root`` (m) for the frames ``future_steps`` ahead; env-origin invariant (both
    heights include the origin).  Realisable from odometry z and the aligned clip.  ``noise_odom_bias_z_m``
    / ``noise_odom_walk_z_m`` (std, m; 0.0 = exact) perturb the ROBOT height estimate (:func:`robot_odometry_height`) while
    training only."""
    pos_w, _ = hero_ref_root_future_pose_w(env, future_steps)
    z_robot = robot_odometry_height(env, noise_odom_bias_z_m, noise_odom_walk_z_m)
    return ref_root_height_rel(z_robot, pos_w[..., 2])


def h23_base_lin_vel_odom(env: Any, noise_odom_vel_m_s: float = 0.0) -> torch.Tensor:
    """Root-velocity feedback (3): robot root linear velocity in the robot HEADING frame (current step; odometry-derived velocity
    feedback -- HERO's original actor observation has no base linear velocity).  ``noise_odom_vel_m_s``: training-only
    zero-mean Gaussian (std, m/s; drawn per call, no bias); 0.0 = bit-exact."""
    rs = env.simulator.robot_root_states
    v = lin_vel_heading_frame(rs[:, 3:7], rs[:, 7:10])
    return _train_gaussian_noise(env, v, noise_odom_vel_m_s, "noise_odom_vel_m_s")


# --------------------------------------------------------------------------------------------------
# critic extras (history 1)
# --------------------------------------------------------------------------------------------------


def base_lin_vel(env: Any) -> torch.Tensor:
    """Pelvis linear velocity in the pelvis frame (3); HERO critic ``base_lin_vel`` (config scale 2.0)."""
    return get_base_lin_vel(env)


def base_orientation(env: Any) -> torch.Tensor:
    """Pelvis quaternion xyzw (4); HERO critic ``base_orientation``."""
    return env.base_quat


# --------------------------------------------------------------------------------------------------
# term tables (for config authors and tests)
# --------------------------------------------------------------------------------------------------

_MOD = "hero_isaacsim.managers.observation.hero"

#: Actor terms: name -> (dim, scale). Sorted order == HERO alphabetical order; dims sum to 135.
HERO_H1_ACTOR_TERMS: dict[str, tuple[int, float]] = {
    "h00_actions": (29, 1.0),
    "h01_base_ang_vel": (3, 0.25),
    "h02_command_ang_vel": (1, 1.0),
    "h03_command_base_height": (1, 2.0),
    "h04_command_lin_vel": (2, 1.0),
    "h05_command_stand": (1, 1.0),
    "h06_command_waist_dofs": (3, 1.0),
    "h07_dif_local_rigid_body_pos_ee": (6, 1.0),
    "h08_dif_local_rigid_body_rot_ee": (12, 1.0),
    "h09_dof_pos": (29, 1.0),
    "h10_dof_vel": (29, 0.05),
    "h11_projected_gravity": (3, 1.0),
    "h12_ref_upper_dof_pos": (14, 1.0),
    "h13_roll_and_pitch": (2, 1.0),
}
HERO_H2_EXTRA_TERMS: dict[str, tuple[int, float]] = {
    "h14_ref_lower_dof_pos": (12, 1.0),
    "h15_ref_root_pitch_roll": (2, 1.0),
    "h16_ref_body_pos_b": (12, 1.0),
}
HERO_OBJECT_TERMS: dict[str, tuple[int, float]] = {
    "h17_obj_pos_b": (3, 1.0),
    "h18_obj_ori_b": (6, 1.0),
    "h19_has_object_flag": (1, 1.0),
}
#: Delta-anchor drift-feedback term (sorts after h19, before the critic ``p*`` terms).
HERO_ANCHOR_TERMS: dict[str, tuple[int, float]] = {
    "h20_ref_root_pose_b": (H20_DIM, 1.0),
}


HERO_ANCHOR2_TERMS: dict[str, tuple[int, float]] = {
    "h21_ref_root_rot_b": (H21_DIM, 1.0),
    "h22_ref_root_height_b": (H22_DIM, 1.0),
}
#: Root-velocity feedback and the combined orientation, height, and velocity terms.
HERO_ANCHOR2_VELOCITY_TERMS: dict[str, tuple[int, float]] = {"h23_base_lin_vel_odom": (H23_DIM, 1.0)}
HERO_ANCHOR2V_TERMS: dict[str, tuple[int, float]] = {**HERO_ANCHOR2_TERMS, **HERO_ANCHOR2_VELOCITY_TERMS}
HERO_CRITIC_EXTRA_TERMS: dict[str, tuple[int, float]] = {
    "base_lin_vel": (3, 2.0),
    "base_orientation": (4, 1.0),
}
#: HERO's original (un-prefixed) key names in its alphabetical concatenation order.
HERO_ORIGINAL_ACTOR_KEYS: tuple[str, ...] = (
    "actions",
    "base_ang_vel",
    "command_ang_vel",
    "command_base_height",
    "command_lin_vel",
    "command_stand",
    "command_waist_dofs",
    "dif_local_rigid_body_pos_ee",
    "dif_local_rigid_body_rot_ee",
    "dof_pos",
    "dof_vel",
    "projected_gravity",
    "ref_upper_dof_pos",
    "roll_and_pitch",
)
H1_ACTOR_DIM = sum(d for d, _ in HERO_H1_ACTOR_TERMS.values())  # 135

#: Terms that accept a training-only noise parameter -> the parameter's name (uniform half-width, term units,
#: applied before ``ObsTermCfg.scale``).
HERO_TERM_NOISE_PARAM: dict[str, str] = {
    "h01_base_ang_vel": "noise_rad_s",
    "h07_dif_local_rigid_body_pos_ee": "noise_pos_m",
    "h08_dif_local_rigid_body_rot_ee": "noise_rot6d",
    "h09_dof_pos": "noise_rad",
    "h10_dof_vel": "noise_rad_s",
    "h11_projected_gravity": "noise_unit",
    "h12_ref_upper_dof_pos": "noise_rad",
    "h13_roll_and_pitch": "noise_rad",
    "h14_ref_lower_dof_pos": "noise_rad",
}


def hero_actor_term_cfgs(
    variant: str = "h1",
    with_object: bool = False,
    zero_waist_when_walking: bool | None = None,
    with_anchor: bool = False,
    with_anchor2: bool = False,
    with_anchor2v: bool = False,
):
    """Build ``{term_name: ObsTermCfg}`` for the actor group (``variant`` "h1" or "h2"; ``with_object`` adds h17-h19;
    ``with_anchor`` adds the delta-anchor h20 term with its default future steps and no odometry noise; ``with_anchor2`` adds the
    h21 / h22 terms the same way (implies ``with_anchor``);
    ``with_anchor2v`` adds h21 / h22 / h23 (implies ``with_anchor2``)).

    ``zero_waist_when_walking`` is forwarded to the terms that depend on it (``None`` = read the command config)."""
    from holosoma.config_types.observation import ObsTermCfg  # noqa: PLC0415

    table = dict(HERO_H1_ACTOR_TERMS)
    if variant == "h2":
        table.update(HERO_H2_EXTRA_TERMS)
    elif variant != "h1":
        raise ValueError(f"unknown variant {variant!r}")
    if with_object:
        table.update(HERO_OBJECT_TERMS)
    if with_anchor or with_anchor2 or with_anchor2v:
        table.update(HERO_ANCHOR_TERMS)
    if with_anchor2 or with_anchor2v:
        table.update(HERO_ANCHOR2_TERMS)
    if with_anchor2v:
        table.update(HERO_ANCHOR2_VELOCITY_TERMS)
    cfgs = {}
    for name, (_dim, scale) in table.items():
        params: dict[str, Any] = {}
        if zero_waist_when_walking is not None and name in (
            "h06_command_waist_dofs",
            "h07_dif_local_rigid_body_pos_ee",
            "h08_dif_local_rigid_body_rot_ee",
        ):
            params["zero_waist_when_walking"] = zero_waist_when_walking
        cfgs[name] = ObsTermCfg(func=f"{_MOD}:{name}", params=params, scale=scale)
    return cfgs


def hero_critic_term_cfgs(
    variant: str = "h1",
    with_object: bool = False,
    zero_waist_when_walking: bool | None = None,
    with_anchor: bool = False,
    with_anchor2: bool = False,
    with_anchor2v: bool = False,
):
    """Actor terms + critic extras (``base_lin_vel`` ×2.0, ``base_orientation``) as ``{name: ObsTermCfg}``."""
    from holosoma.config_types.observation import ObsTermCfg  # noqa: PLC0415

    cfgs = hero_actor_term_cfgs(variant, with_object, zero_waist_when_walking, with_anchor, with_anchor2, with_anchor2v)
    for name, (_dim, scale) in HERO_CRITIC_EXTRA_TERMS.items():
        cfgs[name] = ObsTermCfg(func=f"{_MOD}:{name}", scale=scale)
    return cfgs


__all__ = [
    "H1_ACTOR_DIM",
    "H20_DIM",
    "H20_FRAME_DIM",
    "H20_FUTURE_STEPS",
    "H20_NOISE_PARAMS",
    "H21_DIM",
    "H21_FRAME_DIM",
    "H21_NOISE_PARAMS",
    "H22_DIM",
    "H22_FRAME_DIM",
    "H22_NOISE_PARAMS",
    "H23_DIM",
    "H23_NOISE_PARAMS",
    "HERO_ANCHOR_TERMS",
    "HERO_ANCHOR2_TERMS",
    "HERO_ANCHOR2_VELOCITY_TERMS",
    "HERO_ANCHOR2V_TERMS",
    "HERO_CRITIC_EXTRA_TERMS",
    "HeroOdometryNoiseState",
    "HERO_H1_ACTOR_TERMS",
    "HERO_H2_EXTRA_TERMS",
    "HERO_OBJECT_TERMS",
    "HERO_ORIGINAL_ACTOR_KEYS",
    "HERO_TERM_NOISE_PARAM",
    "base_lin_vel",
    "base_orientation",
    "current_ee_pose_w",
    "ee_body_indices",
    "ee_palm_offset",
    "euler_roll_pitch_xyzw",
    "h00_actions",
    "h01_base_ang_vel",
    "h02_command_ang_vel",
    "h03_command_base_height",
    "h04_command_lin_vel",
    "h05_command_stand",
    "h06_command_waist_dofs",
    "h07_dif_local_rigid_body_pos_ee",
    "h08_dif_local_rigid_body_rot_ee",
    "h09_dof_pos",
    "h10_dof_vel",
    "h11_projected_gravity",
    "h12_ref_upper_dof_pos",
    "h13_roll_and_pitch",
    "h14_ref_lower_dof_pos",
    "h15_ref_root_pitch_roll",
    "h16_ref_body_pos_b",
    "h17_obj_pos_b",
    "h18_obj_ori_b",
    "h19_has_object_flag",
    "h20_ref_root_pose_b",
    "h21_ref_root_rot_b",
    "h22_ref_root_height_b",
    "h23_base_lin_vel_odom",
    "hero_actor_term_cfgs",
    "hero_critic_term_cfgs",
    "hero_ee_reference",
    "hero_ee_residual_from_env",
    "hero_ref_root_future_pos_clip",
    "hero_ref_root_future_pose_w",
    "reference_origins",
    "lin_vel_heading_frame",
    "mark_odom_noise_reset",
    "palm_body_indices",
    "perturbed_root_quat_xyzw",
    "ref_root_height_rel",
    "ref_root_pose_heading_frame",
    "ref_root_rot6d_root_frame",
    "robot_odometry_height",
    "rot6d_column_major",
    "robot_odometry_pose",
    "robot_odometry_quat",
    "training_noise_active",
    "yaw_from_quat_xyzw",
]


# Critic-extra aliases: presets name these ``h01a_``/``h01b_`` so they sort right after ``h01_base_ang_vel``
# (holosoma concatenates group terms alphabetically; HERO's critic layout is actions, base_ang_vel, base_lin_vel,
# base_orientation, command_*...). Same callables as ``base_lin_vel`` / ``base_orientation``.
h01a_base_lin_vel = base_lin_vel
h01b_base_orientation = base_orientation
