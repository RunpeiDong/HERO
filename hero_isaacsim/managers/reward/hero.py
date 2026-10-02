"""HERO paper reward terms and separate upper/lower-body reward accounting.

Rewards are multiplied by the 0.02 s control interval. Runtime quaternions
use xyzw; command targets share the reference pelvis frame."""

from __future__ import annotations

from types import SimpleNamespace

from typing import Any

import torch

from holosoma.managers.observation.terms.wbt import (
    get_base_ang_vel,
    get_base_lin_vel,
    get_projected_gravity,
    gravity_vector,
)

from holosoma.managers.reward.base import RewardTermBase

from holosoma.managers.reward.manager import RewardManager

from holosoma.managers.reward.terms.locomotion import (
    penalty_ang_vel_xy,
    termination,
)

from holosoma.managers.reward.terms.locomotion import penalty_feet_ori as _loco_penalty_feet_ori

from holosoma.managers.reward.terms.wbt import limits_dof_pos, penalty_action_rate

from holosoma.utils.rotations import quat_error_magnitude, yaw_quat

from hero_isaacsim.managers.observation.hero import hero_ee_residual_from_env

from hero_isaacsim.managers.termination.hero import (
    invalidate_terrain_cache,
    terrain_ground_z,
    upright_reference_mask,
)

from hero_isaacsim.utils import hero_constants

from hero_isaacsim.utils.ee_residual import (
    quat_angle_xyzw,
    quat_rotate_inverse_xyzw,
    quat_rotate_xyzw,
)

_C = hero_constants()

def _mc(env: Any):
    mc = env.command_manager.get_state("motion_command")
    assert mc is not None, "motion_command not found in command manager"
    return mc

def _walk(env: Any) -> torch.Tensor:
    """[N] float, 1.0 for walking envs (HERO commands[:, 4])."""
    return _mc(env).stand_flag.reshape(-1).float()

def _vel_cmd(env: Any) -> torch.Tensor:
    return _mc(env).vel_cmd

def _h_cmd(env: Any) -> torch.Tensor:
    return _mc(env).h_cmd.reshape(-1)

def _idx(env: Any, attr: str, values) -> torch.Tensor:
    cached = getattr(env, attr, None)
    if cached is None:
        cached = torch.tensor(list(values), dtype=torch.long, device=env.device)
        setattr(env, attr, cached)
    return cached

def _lower_idx(env: Any) -> torch.Tensor:  # legs + waist (15) — HERO WBC ``lower_dof_indices``
    return _idx(env, "hero_lower_dof_idx", _C.LOWER_DOF_IDX)

def _upper_idx(env: Any) -> torch.Tensor:  # arms (14) — HERO ``upper_dof_indices``
    return _idx(env, "hero_upper_dof_idx", _C.UPPER_DOF_IDX)

def _env_origin_z(env: Any) -> torch.Tensor | float:
    scene = getattr(env.simulator, "scene", None)
    origins = getattr(scene, "env_origins", None) if scene is not None else None
    if origins is None:
        return 0.0
    return origins[:, 2]

def _root_height(env: Any, terrain_relative: bool = False) -> torch.Tensor:
    """Robot pelvis height above the env origin, or (``terrain_relative``) above the terrain under the base."""
    if terrain_relative:
        return env.simulator.robot_root_states[:, 2] - terrain_ground_z(env)
    return env.simulator.robot_root_states[:, 2] - _env_origin_z(env)

def _feet_height(env: Any, terrain_relative: bool = False) -> torch.Tensor:
    """``[N, 2]`` ankle_roll heights above the env origin, or (``terrain_relative``) above the terrain under each foot."""
    z = env.simulator._rigid_body_pos[:, env.feet_indices, 2]
    if terrain_relative:
        ground = terrain_ground_z(env, feet=True)
        if ground.shape != z.shape:  # terrain term tracks another foot set -> ground under the base for both feet
            ground = terrain_ground_z(env)[:, None].expand_as(z)
        return z - ground
    origin = _env_origin_z(env)
    return z - (origin[:, None] if torch.is_tensor(origin) else origin)

def _feet_contact(env: Any, threshold: float = 1.0) -> torch.Tensor:
    """[N, 2] bool, vertical contact force > threshold (HERO ``contact_forces[:, feet, 2] > 1``)."""
    return env.simulator.contact_forces[:, env.feet_indices, 2] > threshold

def _feet_contact_norm(env: Any, threshold: float = 1.0) -> torch.Tensor:
    """[N, 2] bool, contact force NORM > threshold (HERO swing-height / contact_no_vel variant)."""
    return torch.norm(env.simulator.contact_forces[:, env.feet_indices, :3], dim=2) > threshold

def _joint_action_term(env: Any):
    term = getattr(env, "hero_joint_action_term", None)
    if term is None:
        for _name, t in env.action_manager.iter_terms():
            if hasattr(t, "torques"):
                term = t
                break
        if term is None:
            raise RuntimeError("no action term with a `torques` buffer found (need JointPositionActionTerm)")
        env.hero_joint_action_term = term
    return term

def _torques(env: Any) -> torch.Tensor:
    return _joint_action_term(env).torques

def _last_dof_vel(env: Any) -> torch.Tensor:
    """Control-step-rate previous joint velocity (``env.last_dof_vel``, maintained by HeroTrackingManager)."""
    last = getattr(env, "last_dof_vel", None)
    if last is None:
        last = torch.zeros_like(env.simulator.dof_vel)
        env.last_dof_vel = last
    return last

def _actions(env: Any) -> tuple[torch.Tensor, torch.Tensor]:
    return env.action_manager.action, env.action_manager.prev_action

def _soft_dof_pos_limits(env: Any, soft_dof_pos_limit: float) -> tuple[torch.Tensor, torch.Tensor]:
    hard = env.simulator.hard_dof_pos_limits
    m = (hard[:, 0] + hard[:, 1]) / 2
    r = hard[:, 1] - hard[:, 0]
    return m - 0.5 * r * soft_dof_pos_limit, m + 0.5 * r * soft_dof_pos_limit

def _ref_root_quat(env: Any) -> torch.Tensor:
    return _mc(env).root_quat_w  # xyzw at runtime

def _ref_torso_quat(env: Any) -> torch.Tensor:
    mc = _mc(env)
    tracked = list(getattr(getattr(mc, "motion_cfg", None), "body_names_to_track", []) or [])
    if _C.TORSO_BODY_NAME in tracked:
        return mc.body_quat_w[:, tracked.index(_C.TORSO_BODY_NAME), :]
    return mc.root_quat_w

_UPRIGHT_CACHE_ATTR = "_hero_upright_ref_cache"

UPRIGHT_REF_MIN_HEIGHT = 0.55

UPRIGHT_REF_MAX_TILT_RAD = 0.6

def _upright_ref(
    env: Any, min_ref_height: float = UPRIGHT_REF_MIN_HEIGHT, max_ref_tilt_rad: float = UPRIGHT_REF_MAX_TILT_RAD
) -> torch.Tensor:
    """[N] float: 1.0 where the REFERENCE pelvis is upright (height above origin ≥ ``min_ref_height`` AND tilt from
    vertical ≤ ``max_ref_tilt_rad``), else 0.0. From ``HeroMotionCommand.root_pos_w / root_quat_w``.

    Computed once per step: the mask is memoised on ``env`` (per threshold pair) until
    :func:`invalidate_upright_ref_cache` is called — :class:`GroupedRewardManager.compute` and
    ``HeroTrackingManager._check_termination / _compute_reward`` do so at their start (explicit invalidation, no
    reliance on tensor version counters). Anything that mutates the reference and re-reads the gate outside those
    hooks must invalidate itself."""
    params = (float(min_ref_height), float(max_ref_tilt_rad))
    cache = getattr(env, _UPRIGHT_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(env, _UPRIGHT_CACHE_ATTR, cache)
    mask = cache.get(params)
    if mask is None:
        mask = upright_reference_mask(env, min_ref_height, max_ref_tilt_rad).float()
        cache[params] = mask
    return mask

def invalidate_upright_ref_cache(env: Any) -> None:
    """Drop the memoised :func:`_upright_ref` masks (call once per step before the first gated term is evaluated); also
    drops the terrain ground-height cache (``termination.hero.invalidate_terrain_cache``) so the terrain-aware terms
    re-sample the ground once per compute."""
    cache = getattr(env, _UPRIGHT_CACHE_ATTR, None)
    if cache:
        cache.clear()
    invalidate_terrain_cache(env)

def _gate_upright(env: Any, value: torch.Tensor, upright_ref_only: bool) -> torch.Tensor:
    """``value * _upright_ref(env)`` when ``upright_ref_only`` else ``value`` (weights unchanged for upright refs)."""
    return value * _upright_ref(env) if upright_ref_only else value

def _root_body_index(env: Any) -> int:
    """Simulator body index of the pelvis (``hero_constants.PELVIS_BODY_NAME``, the articulation root link whose origin
    ``robot_root_states[:, :3]`` reports), memoised on ``env.hero_root_body_index``."""
    cached = getattr(env, "hero_root_body_index", None)
    if cached is None:
        cached = list(env.simulator.body_names).index(_C.PELVIS_BODY_NAME)
        env.hero_root_body_index = cached
    return cached

def root_link_lin_vel_w(env: Any) -> torch.Tensor:
    """``[N, 3]`` WORLD velocity of the pelvis LINK ORIGIN -- the derivative of ``robot_root_states[:, :3]`` and the robot-side
    counterpart of the clip's ``body_lin_vel_w[:, pelvis]``.

    Read from ``simulator._rigid_body_link_vel`` (IsaacLab ``body_link_lin_vel_w``, or the COM-to-link conversion
    ``v_link = v_com - omega x R r_com``; ``holosoma/simulator/isaacsim/body_velocity.py``) at the pelvis row -- IsaacLab's
    ``root_link_lin_vel_w`` is that row.  A simulator without the buffer raises: the legacy COM buffer
    ``robot_root_states[:, 7:10]`` is NOT a substitute and a link offset must never be added to it."""
    link_vel = getattr(env.simulator, "_rigid_body_link_vel", None)
    if link_vel is None:
        raise RuntimeError(
            "link_velocity=True requires simulator._rigid_body_link_vel (explicit link-origin velocity: IsaacLab body_link_lin_vel_w "
            "or the COM -> link conversion of holosoma/simulator/isaacsim/body_velocity.py); the COM buffer robot_root_states[:, 7:10] "
            "cannot substitute for it"
        )
    return link_vel[:, _root_body_index(env), :]

def _robot_lin_vel(env: Any, heading_frame: bool, link_velocity: bool = False) -> torch.Tensor:
    """Robot lin vel."""
    if link_velocity:
        q = env.simulator.robot_root_states[:, 3:7]
        return quat_rotate_inverse_xyzw(yaw_quat(q, w_last=True) if heading_frame else q, root_link_lin_vel_w(env))
    if not heading_frame:
        return get_base_lin_vel(env)
    root = env.simulator.robot_root_states
    return quat_rotate_inverse_xyzw(yaw_quat(root[:, 3:7], w_last=True), root[:, 7:10])

def _robot_yaw_rate(env: Any, heading_frame: bool) -> torch.Tensor:
    """Pelvis yaw rate ``[N]``: pelvis-frame ω_z (HERO) or, with ``heading_frame``, world ω_z (== yaw-only-frame ω_z),
    matching the ``from_clip`` command ``root_ang_vel_w[:, 2]``."""
    if not heading_frame:
        return get_base_ang_vel(env)[:, 2]
    return env.simulator.robot_root_states[:, 12]

def tracking_lin_vel_x(env: Any, sigma: float = 0.25, heading_frame: bool = False, link_velocity: bool = False) -> torch.Tensor:
    """Tracking lin vel x."""
    err = torch.square(_vel_cmd(env)[:, 0] - _robot_lin_vel(env, heading_frame, link_velocity)[:, 0])
    return torch.exp(-err / sigma)

def tracking_lin_vel_y(env: Any, sigma: float = 0.25, heading_frame: bool = False, link_velocity: bool = False) -> torch.Tensor:
    """exp(-(vy_cmd - vy)^2 / sigma); ``heading_frame`` / ``link_velocity`` as in :func:`tracking_lin_vel_x`."""
    err = torch.square(_vel_cmd(env)[:, 1] - _robot_lin_vel(env, heading_frame, link_velocity)[:, 1])
    return torch.exp(-err / sigma)

def tracking_ang_vel(env: Any, sigma: float = 0.25, heading_frame: bool = False) -> torch.Tensor:
    """exp(-(wz_cmd - wz)^2 / sigma); ``heading_frame``: world ω_z like the ``from_clip`` yaw-rate
    command (:func:`_robot_yaw_rate`), False = HERO pelvis-frame ω_z."""
    err = torch.square(_vel_cmd(env)[:, 2] - _robot_yaw_rate(env, heading_frame))
    return torch.exp(-err / sigma)

def _base_height_exp(env: Any, sigma: float, terrain_relative: bool = False) -> torch.Tensor:
    return torch.exp(-torch.abs(_h_cmd(env) - _root_height(env, terrain_relative)) / sigma)

def tracking_walk_base_height(env: Any, sigma: float = 0.05, terrain_relative: bool = False) -> torch.Tensor:
    """exp(-|h_cmd - z_root| / sigma) * walk; ``terrain_relative``: z_root above the terrain."""
    return _base_height_exp(env, sigma, terrain_relative) * _walk(env)

def tracking_stance_base_height(env: Any, sigma: float = 0.05, terrain_relative: bool = False) -> torch.Tensor:
    """exp(-|h_cmd - z_root| / sigma) * (1 - walk); ``terrain_relative`` as above."""
    return _base_height_exp(env, sigma, terrain_relative) * (1.0 - _walk(env))

def _waist_error(env: Any) -> torch.Tensor:
    idx = _idx(env, "hero_waist_dof_idx", _C.WAIST_DOF_IDX)
    ref = _mc(env).ref_upper_dof_pos[:, 0:3]
    return torch.sum(torch.square(env.simulator.dof_pos[:, idx] - ref), dim=1)

def tracking_waist_dofs_tapping(env: Any, sigma: float = 0.05) -> torch.Tensor:
    """exp(-Σ(q_waist - waist_cmd)^2 / sigma) * walk."""
    return torch.exp(-_waist_error(env) / sigma) * _walk(env)

def tracking_waist_dofs_stance(env: Any, sigma: float = 0.05) -> torch.Tensor:
    """exp(-Σ(q_waist - waist_cmd)^2 / sigma) * (1 - walk)."""
    return torch.exp(-_waist_error(env) / sigma) * (1.0 - _walk(env))

def penalty_lin_vel_z(
    env: Any, relative_to_reference: bool = False, walk_only: bool = True, link_velocity: bool = False
) -> torch.Tensor:
    """Penalty lin vel z."""
    if link_velocity:
        vz = quat_rotate_inverse_xyzw(env.simulator.robot_root_states[:, 3:7], root_link_lin_vel_w(env))[:, 2]
    else:
        vz = get_base_lin_vel(env)[:, 2]
    if relative_to_reference:
        mc = _mc(env)
        vz_ref = quat_rotate_inverse_xyzw(mc.root_quat_w, mc.root_lin_vel_w)[:, 2]
        vz = vz - vz_ref
    pen = torch.square(vz)
    return pen * _walk(env) if walk_only else pen

def penalty_orientation(env: Any, relative_to_reference: bool = False) -> torch.Tensor:
    """HERO absolute form Σ g_xy^2 of the pelvis projected gravity.

    ``relative_to_reference=True``: 1 - cos(angle) between the robot's and the reference pelvis' projected
    gravity directions (= 1 - g_robot·g_ref), i.e. the paper's "1 - cos θ" made reference-relative."""
    g = get_projected_gravity(env)
    if not relative_to_reference:
        return torch.sum(torch.square(g[:, :2]), dim=1)
    g_ref = quat_rotate_inverse_xyzw(_ref_root_quat(env), gravity_vector(env))
    return 1.0 - torch.sum(g * g_ref, dim=1)

def penalty_torso_orientation(env: Any, relative_to_reference: bool = False) -> torch.Tensor:
    """HERO absolute form Σ |g_xy| of the torso projected gravity (L1).

    ``relative_to_reference=True``: 1 - g_torso·g_ref_torso (reference torso from the tracked bodies, falls
    back to the reference pelvis if ``torso_link`` is not tracked)."""
    torso_quat = env.simulator._rigid_body_rot[:, env.torso_index]
    g = quat_rotate_inverse_xyzw(torso_quat, gravity_vector(env))
    if not relative_to_reference:
        return torch.sum(torch.abs(g[:, :2]), dim=1)
    g_ref = quat_rotate_inverse_xyzw(_ref_torso_quat(env), gravity_vector(env))
    return 1.0 - torch.sum(g * g_ref, dim=1)

def penalty_ang_vel_xy_torso(env: Any) -> torch.Tensor:
    """Σ ω_xy^2 of the torso angular velocity in the torso frame."""
    torso_quat = env.simulator._rigid_body_rot[:, env.torso_index]
    w = quat_rotate_inverse_xyzw(torso_quat, env.simulator._rigid_body_ang_vel[:, env.torso_index])
    return torch.sum(torch.square(w[:, :2]), dim=1)

def base_height(env: Any, stance_penalty_scale: float = 5.0, terrain_relative: bool = False) -> torch.Tensor:
    """(z_root - h_cmd)^2, multiplied by ``stance_penalty_scale`` in stance; ``terrain_relative``: z_root
    above the terrain under the base."""
    pen = torch.square(_root_height(env, terrain_relative) - _h_cmd(env))
    walk = _walk(env)
    return pen * (walk + (1.0 - walk) * stance_penalty_scale)

def _sq_sum(x: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
    return torch.sum(torch.square(x[:, idx]), dim=1)

def penalty_lower_body_torques(env: Any) -> torch.Tensor:
    """Σ τ^2 over legs + waist."""
    return _sq_sum(_torques(env), _lower_idx(env))

def penalty_lower_body_dof_vel(env: Any) -> torch.Tensor:
    """Σ q̇^2 over legs + waist."""
    return _sq_sum(env.simulator.dof_vel, _lower_idx(env))

def penalty_lower_body_dof_acc(env: Any) -> torch.Tensor:
    """Σ ((q̇_prev - q̇)/dt)^2 over legs + waist."""
    return _sq_sum((_last_dof_vel(env) - env.simulator.dof_vel) / env.dt, _lower_idx(env))

def penalty_lower_body_action_rate(env: Any) -> torch.Tensor:
    """Σ (a_prev - a)^2 over legs + waist."""
    a, prev = _actions(env)
    return _sq_sum(prev - a, _lower_idx(env))

def limits_dof_pos_subset(env: Any, dof_idx: torch.Tensor, soft_dof_pos_limit: float = 0.95) -> torch.Tensor:
    """Σ distance outside the soft (``soft_dof_pos_limit`` of hard range) position limits over ``dof_idx``."""
    lo, hi = _soft_dof_pos_limits(env, soft_dof_pos_limit)
    q = env.simulator.dof_pos[:, dof_idx]
    out = -(q - lo[dof_idx]).clip(max=0.0) + (q - hi[dof_idx]).clip(min=0.0)
    return torch.sum(out, dim=1)

def limits_lower_body_dof_pos(env: Any, soft_dof_pos_limit: float = 0.95) -> torch.Tensor:
    """Soft position-limit violation over legs + waist."""
    return limits_dof_pos_subset(env, _lower_idx(env), soft_dof_pos_limit)

def penalty_stance_dof(env: Any, upright_ref_only: bool = False) -> torch.Tensor:
    """Penalty stance dof."""
    pen = _sq_sum(env.simulator.dof_vel, _lower_idx(env)) * (1.0 - _walk(env))
    return _gate_upright(env, pen, upright_ref_only)

def _feet_air_state(env: Any):
    st = getattr(env, "hero_feet_air_state", None)
    n = env.num_envs
    if st is None or st.feet_air_time.shape[0] != n:
        st = SimpleNamespace(
            feet_air_time=torch.zeros(n, 2, device=env.device),
            last_feet_air_time=torch.zeros(n, 2, device=env.device),
            last_contacts=torch.zeros(n, 2, dtype=torch.bool, device=env.device),
        )
        env.hero_feet_air_state = st
    return st

class FeetAirTime(RewardTermBase):
    """Σ_feet (t_air - threshold_time) * first_contact, only if ‖v_cmd_xy‖ > min_cmd_vel."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        self.threshold_time = float(cfg.params.get("threshold_time", 0.5))
        self.min_cmd_vel = float(cfg.params.get("min_cmd_vel", 0.1))
        self.upright_ref_only = bool(cfg.params.get("upright_ref_only", False))
        _feet_air_state(env)

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        st = _feet_air_state(env)
        contact = _feet_contact(env)
        contact_filt = contact | st.last_contacts
        st.last_contacts = contact
        first_contact = (st.feet_air_time > 0.0) & contact_filt
        st.feet_air_time += env.dt
        st.last_feet_air_time[first_contact] = st.feet_air_time[first_contact]
        rew = torch.sum((st.feet_air_time - self.threshold_time) * first_contact.float(), dim=1)
        rew = rew * (torch.norm(_vel_cmd(env)[:, :2], dim=1) > self.min_cmd_vel).float()
        st.feet_air_time *= (~contact_filt).float()
        return _gate_upright(env, rew, self.upright_ref_only)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        st = _feet_air_state(self.env)
        if env_ids is None:
            st.feet_air_time.zero_()
            st.last_feet_air_time.zero_()
            st.last_contacts.zero_()
        else:
            st.feet_air_time[env_ids] = 0.0
            st.last_feet_air_time[env_ids] = 0.0
            st.last_contacts[env_ids] = False

def penalty_diff_feet_air_time(env: Any, upright_ref_only: bool = False) -> torch.Tensor:
    """Penalty diff feet air time."""
    st = _feet_air_state(env)
    pen = torch.abs(st.last_feet_air_time[:, 0] - st.last_feet_air_time[:, 1])
    return _gate_upright(env, pen, upright_ref_only)

def penalty_contact_no_vel(env: Any, stance_scale: float = 10.0, upright_ref_only: bool = False) -> torch.Tensor:
    """Penalty contact no vel."""
    contact = _feet_contact_norm(env)
    feet_vel = env.simulator._rigid_body_vel[:, env.feet_indices]
    pen = torch.sum(torch.square(feet_vel * contact.unsqueeze(-1)), dim=(1, 2))
    walk = _walk(env)
    return _gate_upright(env, pen * (walk + (1.0 - walk) * stance_scale), upright_ref_only)

def penalty_feet_height(
    env: Any, target: float = 0.11, tolerance: float = 0.02, upright_ref_only: bool = False, terrain_relative: bool = False
) -> torch.Tensor:
    """Penalty feet height."""
    dif = torch.abs(_feet_height(env, terrain_relative) - target).min(dim=1).values
    return _gate_upright(env, torch.clip(dif - tolerance, min=0.0) * _walk(env), upright_ref_only)

def penalty_feet_swing_height(
    env: Any,
    target_walk: float = 0.11,
    target_stance: float = 0.025,
    upright_ref_only: bool = False,
    terrain_relative: bool = False,
) -> torch.Tensor:
    """Penalty feet swing height."""
    contact = _feet_contact_norm(env)
    walk = _walk(env)[:, None]
    target = target_walk * walk + target_stance * (1.0 - walk)
    err = torch.square(_feet_height(env, terrain_relative) - target) * (~contact).float()
    return _gate_upright(env, torch.sum(err, dim=1), upright_ref_only)

def _wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    """Pure wrap to (-π, π] (holosoma's ``wrap_to_pi`` is in-place and mis-wraps (-2π, -π))."""
    return torch.atan2(torch.sin(angles), torch.cos(angles))

def feet_heading_alignment(env: Any, upright_ref_only: bool = False) -> torch.Tensor:
    """Feet heading alignment."""
    fwd = torch.tensor([1.0, 0.0, 0.0], device=env.device).expand(env.num_envs, 3)

    def heading(q):
        f = quat_rotate_xyzw(q, fwd)
        return torch.atan2(f[:, 1], f[:, 0])

    h_root = heading(env.base_quat)
    h_l = heading(env.simulator._rigid_body_rot[:, env.feet_indices[0]])
    h_r = heading(env.simulator._rigid_body_rot[:, env.feet_indices[1]])
    pen = torch.abs(_wrap_to_pi(h_l - h_root)) + torch.abs(_wrap_to_pi(h_r - h_root))
    return _gate_upright(env, pen, upright_ref_only)

def penalty_close_feet_xy(env: Any, close_feet_threshold: float = 0.17, upright_ref_only: bool = False) -> torch.Tensor:
    """Penalty close feet xy."""
    l = env.simulator._rigid_body_pos[:, env.feet_indices[0], :2]
    r = env.simulator._rigid_body_pos[:, env.feet_indices[1], :2]
    return _gate_upright(env, (torch.norm(l - r, dim=1) < close_feet_threshold).float(), upright_ref_only)

def penalty_hip_pos(
    env: Any, stance_scale: float = 1.0, use_h_cmd_quirk: bool = False, upright_ref_only: bool = False
) -> torch.Tensor:
    """Σ q^2 over hip roll & yaw (both legs)."""
    idx = _idx(env, "hero_hip_roll_yaw_idx", (1, 2, 7, 8))
    pen = _sq_sum(env.simulator.dof_pos, idx)
    walk = _walk(env)
    scale = _h_cmd(env) if use_h_cmd_quirk else stance_scale
    return _gate_upright(env, pen * (walk + (1.0 - walk) * scale), upright_ref_only)

def penalty_stance_tap_feet(env: Any, upright_ref_only: bool = False) -> torch.Tensor:
    """Penalty stance tap feet."""
    feet_diff = torch.abs(
        env.simulator._rigid_body_pos[:, env.feet_indices[0], :3] - env.simulator._rigid_body_pos[:, env.feet_indices[1], :3]
    )
    projected = quat_rotate_inverse_xyzw(env.base_quat, feet_diff)
    stance_tap = _walk(env) * (torch.abs(_vel_cmd(env)[:, 0]) > 0.0).float()
    return _gate_upright(env, torch.abs(projected[:, 0]) * (1.0 - stance_tap), upright_ref_only)

def penalty_stance_root(env: Any, upright_ref_only: bool = False) -> torch.Tensor:
    """Penalty stance root."""
    feet_mid = 0.5 * (
        env.simulator._rigid_body_pos[:, env.feet_indices[0], :3] + env.simulator._rigid_body_pos[:, env.feet_indices[1], :3]
    )
    root_pos = env.simulator.robot_root_states[:, 0:3]
    projected = quat_rotate_inverse_xyzw(env.base_quat, root_pos - feet_mid)
    return _gate_upright(env, torch.abs(projected[:, 1]) * (1.0 - _walk(env)), upright_ref_only)

def penalty_stand_still(env: Any, force_threshold: float = 0.1, upright_ref_only: bool = False) -> torch.Tensor:
    """Penalty stand still."""
    no_contact = torch.sum((env.simulator.contact_forces[:, env.feet_indices, 2] < force_threshold).int(), dim=1) > 0
    return _gate_upright(env, no_contact.float() * (1.0 - _walk(env)), upright_ref_only)

def penalty_stance_symmetry(
    env: Any,
    left_no: tuple[int, ...] = (0, 3, 4),
    left_op: tuple[int, ...] = (1, 2, 5),
    right_no: tuple[int, ...] = (6, 9, 10),
    right_op: tuple[int, ...] = (7, 8, 11),
    upright_ref_only: bool = False,
) -> torch.Tensor:
    """Penalty stance symmetry."""
    q = env.simulator.dof_pos
    ln = _idx(env, "hero_sym_left_no", left_no)
    lo = _idx(env, "hero_sym_left_op", left_op)
    rn = _idx(env, "hero_sym_right_no", right_no)
    ro = _idx(env, "hero_sym_right_op", right_op)
    pen = torch.sum(torch.abs(q[:, ln] - q[:, rn]) + torch.abs(q[:, lo] + q[:, ro]), dim=1)
    return _gate_upright(env, pen * (1.0 - _walk(env)), upright_ref_only)

def penalty_ankle_roll(env: Any, upright_ref_only: bool = False) -> torch.Tensor:
    """Penalty ankle roll."""
    idx = _idx(env, "hero_ankle_roll_idx", (5, 11))
    return _gate_upright(env, torch.sum(torch.abs(env.simulator.dof_pos[:, idx]), dim=1), upright_ref_only)

def penalty_contact(env: Any, moving_threshold: float = 0.01, upright_ref_only: bool = False) -> torch.Tensor:
    """Penalty contact."""
    walk = _walk(env) > 0.5
    n_contact = _feet_contact(env).sum(dim=1)
    cmd = _vel_cmd(env)
    moving = (torch.norm(cmd[:, :2], dim=1) >= moving_threshold) | (torch.abs(cmd[:, 2]) >= moving_threshold)
    res = (~walk & (n_contact < 2)) | (walk & ((n_contact == 2) | (n_contact == 0)) & moving)
    return _gate_upright(env, res.float(), upright_ref_only)

def penalty_negative_knee_joint(env: Any, knee_min: float = 0.2, upright_ref_only: bool = False) -> torch.Tensor:
    """Penalty negative knee joint."""
    idx = _idx(env, "hero_knee_idx", (3, 9))
    return _gate_upright(env, torch.sum((env.simulator.dof_pos[:, idx] < knee_min).float(), dim=1), upright_ref_only)

def penalty_shift_in_zero_command(env: Any, cmd_threshold: float = 0.2, upright_ref_only: bool = False) -> torch.Tensor:
    """Penalty shift in zero command."""
    v = torch.norm(env.simulator.robot_root_states[:, 7:9], dim=-1)
    zero_cmd = (torch.norm(_vel_cmd(env)[:, :2], dim=1) < cmd_threshold).float()
    return _gate_upright(env, v * zero_cmd * _walk(env), upright_ref_only)

def penalty_ang_shift_in_zero_command(env: Any, cmd_threshold: float = 0.1, upright_ref_only: bool = False) -> torch.Tensor:
    """Penalty ang shift in zero command."""
    w = torch.abs(env.simulator.robot_root_states[:, 12])
    zero_cmd = (torch.abs(_vel_cmd(env)[:, 2]) < cmd_threshold).float()
    return _gate_upright(env, w * zero_cmd * _walk(env), upright_ref_only)

def penalty_feet_ori(env: Any, upright_ref_only: bool = False) -> torch.Tensor:
    """Penalty feet ori."""
    return _gate_upright(env, _loco_penalty_feet_ori(env), upright_ref_only)

def tracking_upper_body_dofs(env: Any, sigma: float = 0.1) -> torch.Tensor:
    """exp(-Σ_{14 arms}(q - q_ref)^2 / sigma)."""
    idx = _upper_idx(env)
    ref = _mc(env).ref_upper_dof_pos[:, 3:17]
    err = torch.sum(torch.square(env.simulator.dof_pos[:, idx] - ref), dim=1)
    return torch.exp(-err / sigma)

def penalty_upper_body_torques(env: Any) -> torch.Tensor:
    """Σ τ^2 over arms."""
    return _sq_sum(_torques(env), _upper_idx(env))

def penalty_upper_body_dof_vel(env: Any) -> torch.Tensor:
    """Σ q̇^2 over arms."""
    return _sq_sum(env.simulator.dof_vel, _upper_idx(env))

def penalty_upper_body_dof_acc(env: Any) -> torch.Tensor:
    """Σ ((q̇_prev - q̇)/dt)^2 over arms."""
    return _sq_sum((_last_dof_vel(env) - env.simulator.dof_vel) / env.dt, _upper_idx(env))

def penalty_upper_body_action_rate(env: Any) -> torch.Tensor:
    """Σ (a_prev - a)^2 over arms."""
    a, prev = _actions(env)
    return _sq_sum(prev - a, _upper_idx(env))

def limits_upper_body_dof_pos(env: Any, soft_dof_pos_limit: float = 0.95) -> torch.Tensor:
    """Soft position-limit violation over arms."""
    return limits_dof_pos_subset(env, _upper_idx(env), soft_dof_pos_limit)

def limits_upper_body_dof_vel(env: Any) -> torch.Tensor:
    """Σ clip(|q̇| - q̇_lim, 0, 1) over arms, raw URDF velocity limits."""
    idx = _upper_idx(env)
    return torch.sum((torch.abs(env.simulator.dof_vel[:, idx]) - env.dof_vel_limits[idx]).clip(min=0.0, max=1.0), dim=1)

def limits_upper_body_torque(env: Any, soft_torque_limit: float = 0.9) -> torch.Tensor:
    """Σ clip(|τ| - soft_torque_limit * τ_lim, 0) over arms."""
    idx = _upper_idx(env)
    return torch.sum((torch.abs(_torques(env)[:, idx]) - env.torque_limits[idx] * soft_torque_limit).clip(min=0.0), dim=1)

def _ee_vel_body_indices(env: Any) -> torch.Tensor:
    """Rigid bodies whose velocities define the EE velocity: palms if present else wrist_yaw links (``env.ee_vel_body_indices``)."""
    idx = getattr(env, "ee_vel_body_indices", None)
    if idx is None:
        names = list(env.simulator.body_names)
        chosen = _C.PALM_BODY_NAMES if all(n in names for n in _C.PALM_BODY_NAMES) else _C.EE_BODY_NAMES
        idx = torch.tensor([names.index(n) for n in chosen], dtype=torch.long, device=env.device)
        env.ee_vel_body_indices = idx
    return idx

def _last_ee_vel(env: Any, attr: str, like: torch.Tensor) -> torch.Tensor:
    last = getattr(env, attr, None)
    if last is None:
        last = torch.zeros_like(like)
        setattr(env, attr, last)
    return last

def penalty_ee_lin_acc(env: Any) -> torch.Tensor:
    """Σ_hands ‖v_ee - v_ee,prev‖ (finite difference, not / dt) with per-hand buffers
    ``env.last_ee_lin_vel [N,2,3]`` (left <- left, right <- right)."""
    v = env.simulator._rigid_body_vel[:, _ee_vel_body_indices(env), :]
    acc = v - _last_ee_vel(env, "last_ee_lin_vel", v)
    return torch.sum(torch.norm(acc, dim=2), dim=1)

def penalty_ee_ang_acc(env: Any) -> torch.Tensor:
    """Σ_hands ‖ω_ee - ω_ee,prev‖ with ``env.last_ee_ang_vel [N,2,3]``."""
    w = env.simulator._rigid_body_ang_vel[:, _ee_vel_body_indices(env), :]
    acc = w - _last_ee_vel(env, "last_ee_ang_vel", w)
    return torch.sum(torch.norm(acc, dim=2), dim=1)

def _gate(env: Any, gate: str) -> torch.Tensor | float:
    if gate == "stance":
        return 1.0 - _walk(env)
    if gate == "walk":
        return _walk(env)
    if gate == "none":
        return 1.0
    raise ValueError(f"gate must be 'stance' | 'walk' | 'none', got {gate!r}")

def tracking_ee_pos(
    env: Any, sigma: float = 0.05, gate: str = "stance", zero_waist_when_walking: bool | None = None
) -> torch.Tensor:
    """exp(-Σ_6 Δp^2 / sigma) * gate, Δp = HERO pelvis-frame EE residual. ``gate`` in stance|walk|none."""
    dp, _, _ = hero_ee_residual_from_env(env, zero_waist_when_walking)
    err = torch.sum(torch.square(dp.reshape(env.num_envs, -1)), dim=1)
    return torch.exp(-err / sigma) * _gate(env, gate)

def tracking_ee_rot(
    env: Any, sigma: float = 0.10, gate: str = "stance", zero_waist_when_walking: bool | None = None
) -> torch.Tensor:
    """exp(-mean_hands ‖log(q_cur^-1 q_ref)‖ / sigma) * gate — axis-angle norm, linear in the exp."""
    _, _, q_diff = hero_ee_residual_from_env(env, zero_waist_when_walking)
    err = quat_angle_xyzw(q_diff).mean(dim=1).clamp_min(0.0)
    return torch.exp(-err / sigma) * _gate(env, gate)

def tracking_stance_ee_pos(
    env: Any, sigma: float = 0.05, zero_waist_when_walking: bool | None = None, stance_only: bool = True
) -> torch.Tensor:
    """``tracking_stance_ee_pos`` = :func:`tracking_ee_pos` gated to stance."""
    return tracking_ee_pos(env, sigma, "stance" if stance_only else "none", zero_waist_when_walking)

def tracking_stance_ee_rot(
    env: Any, sigma: float = 0.10, zero_waist_when_walking: bool | None = None, stance_only: bool = True
) -> torch.Tensor:
    """``tracking_stance_ee_rot`` = :func:`tracking_ee_rot` gated to stance (``stance_only``
    as in :func:`tracking_stance_ee_pos`)."""
    return tracking_ee_rot(env, sigma, "stance" if stance_only else "none", zero_waist_when_walking)

_MOD = "hero_isaacsim.managers.reward.hero"

_LOCO = "holosoma.managers.reward.terms.locomotion"

LOWER, UPPER, PC = "lower_body", "upper_body", "penalty_curriculum"

class GroupedRewardManager(RewardManager):
    """``RewardManager`` that also tracks per-step term rewards and per-tag group sums.

    ``group_tags`` (default ``("lower_body", "upper_body")``) name the groups; every active term contributes
    its weighted, dt-scaled step reward to each group whose tag it carries. Terms carrying none of the group
    tags are collected under ``ungrouped_terms`` and reported in the ``"ungrouped"`` group (they still count
    towards the total). The total reward is *identical* to ``RewardManager.compute`` (incl. the optional
    ``only_positive_rewards`` clip, which is applied to the total only — the groups are left unclipped).

    Attributes after each :meth:`compute`:
        ``group_rewards``: ``{group: Tensor[num_envs]}`` step rewards per group;
        ``term_step_rewards``: ``{term: Tensor[num_envs]}`` weighted step reward of every term;
        ``episode_group_sums``: ``{group: Tensor[num_envs]}`` running per-episode sums (reset per env in
        :meth:`reset`, exported as ``extras["episode"]["rew_group_<group>"]`` like the term sums).

    Use :meth:`wrap` to upgrade an already-constructed ``RewardManager`` in place (keeps the resolved term
    functions / stateful instances / episode sums, so it is safe to call from ``_init_buffers`` before the
    curriculum manager rescales any weights)."""

    DEFAULT_GROUP_TAGS: tuple[str, ...] = (LOWER, UPPER)
    UNGROUPED = "ungrouped"

    def __init__(self, cfg: Any, env: Any, device: str, group_tags: tuple[str, ...] | None = None):
        super().__init__(cfg, env, device)
        self._init_groups(group_tags)

    @classmethod
    def wrap(cls, manager: RewardManager, group_tags: tuple[str, ...] | None = None) -> "GroupedRewardManager":
        """Return ``manager`` itself if already grouped, else a GroupedRewardManager sharing its state."""
        if isinstance(manager, cls):
            return manager
        wrapped = cls.__new__(cls)
        wrapped.__dict__.update(manager.__dict__)
        wrapped._init_groups(group_tags)
        return wrapped

    # -- setup -------------------------------------------------------------------------------------
    def _init_groups(self, group_tags: tuple[str, ...] | None) -> None:
        self.group_tags: tuple[str, ...] = tuple(group_tags) if group_tags else self.DEFAULT_GROUP_TAGS
        n = self.env.num_envs
        self._term_groups: dict[str, tuple[str, ...]] = {}
        self.ungrouped_terms: list[str] = []
        for name, cfg in zip(self._term_names, self._term_cfgs):
            groups = tuple(g for g in self.group_tags if g in cfg.tags)
            if not groups:
                groups = (self.UNGROUPED,)
                self.ungrouped_terms.append(name)
            self._term_groups[name] = groups
        group_names = list(self.group_tags) + ([self.UNGROUPED] if self.ungrouped_terms else [])
        self.group_rewards: dict[str, torch.Tensor] = {
            g: torch.zeros(n, dtype=torch.float, device=self.device) for g in group_names
        }
        self.episode_group_sums: dict[str, torch.Tensor] = {
            g: torch.zeros(n, dtype=torch.float, device=self.device) for g in group_names
        }
        self.term_step_rewards: dict[str, torch.Tensor] = {
            name: torch.zeros(n, dtype=torch.float, device=self.device) for name in self._term_names
        }
        if self.ungrouped_terms and self.logger is not None:
            self.logger.warning(
                "GroupedRewardManager: terms without a group tag %s -> group '%s': %s",
                self.group_tags, self.UNGROUPED, self.ungrouped_terms,
            )

    def term_groups(self, name: str) -> tuple[str, ...]:
        """Groups a term contributes to."""
        return self._term_groups[name]

    def terms_in_group(self, group: str) -> list[str]:
        return [n for n in self._term_names if group in self._term_groups[n]]

    # -- compute -----------------------------------------------------------------------------------
    def compute(self, dt: float) -> torch.Tensor:
        invalidate_upright_ref_cache(self.env)
        self._reward_buf[:] = 0.0
        for buf in self.group_rewards.values():
            buf[:] = 0.0
        for term_name, term_cfg in zip(self._term_names, self._term_cfgs):
            if term_name in self._term_instances:
                rew_raw = self._term_instances[term_name](self.env, **term_cfg.params)
            else:
                rew_raw = self._term_funcs[term_name](self.env, **term_cfg.params)
            if rew_raw.shape[0] != self.env.num_envs:
                raise ValueError(
                    f"Reward term '{term_name}' returned wrong shape. Expected [{self.env.num_envs}], got {rew_raw.shape}"
                )
            rew_scaled = rew_raw * term_cfg.weight * dt
            self._reward_buf += rew_scaled
            self.term_step_rewards[term_name][:] = rew_scaled
            for g in self._term_groups[term_name]:
                self.group_rewards[g] += rew_scaled
                self.episode_group_sums[g] += rew_scaled
            self._episode_sums[term_name] += rew_scaled
            self._episode_sums_raw[term_name] += rew_raw
        if self.cfg.only_positive_rewards:
            self._reward_buf[:] = torch.clip(self._reward_buf, min=0.0)
        return self._reward_buf

    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, dict[str, torch.Tensor]]:
        extras = super().reset(env_ids)
        if env_ids is None:
            sl: slice | torch.Tensor = slice(None)
        else:
            sl = env_ids if isinstance(env_ids, torch.Tensor) else torch.as_tensor(env_ids, device=self.device)
            sl = sl.to(device=self.device, dtype=torch.long)
        for g, sums in self.episode_group_sums.items():
            rew_all = sums / self.env.max_episode_length_s
            extras["episode_all"][f"rew_group_{g}"] = rew_all.detach().clone()
            extras["episode"][f"rew_group_{g}"] = rew_all.detach().clone() if env_ids is None else rew_all[sl].detach().clone()
            sums[sl] = 0.0
        return extras

HERO_109_REWARD_TABLE: dict[str, tuple[float, tuple[str, ...], str, dict[str, Any]]] = {
    # ---- lower body (38) ----
    "tracking_lin_vel_x": (2.0, (LOWER,), f"{_MOD}:tracking_lin_vel_x", {"sigma": 0.25}),
    "tracking_lin_vel_y": (1.5, (LOWER,), f"{_MOD}:tracking_lin_vel_y", {"sigma": 0.25}),
    "tracking_ang_vel": (4.0, (LOWER,), f"{_MOD}:tracking_ang_vel", {"sigma": 0.25}),
    "tracking_walk_base_height": (1.0, (LOWER,), f"{_MOD}:tracking_walk_base_height", {"sigma": 0.05}),
    "tracking_stance_base_height": (4.0, (LOWER,), f"{_MOD}:tracking_stance_base_height", {"sigma": 0.05}),
    "tracking_waist_dofs_tapping": (0.5, (LOWER,), f"{_MOD}:tracking_waist_dofs_tapping", {"sigma": 0.05}),
    "tracking_waist_dofs_stance": (3.0, (LOWER,), f"{_MOD}:tracking_waist_dofs_stance", {"sigma": 0.05}),
    "penalty_lin_vel_z": (-2.0, (LOWER,), f"{_MOD}:penalty_lin_vel_z", {}),
    "penalty_ang_vel_xy": (-0.05, (LOWER,), f"{_LOCO}:penalty_ang_vel_xy", {}),
    "penalty_orientation": (-1.5, (LOWER,), f"{_MOD}:penalty_orientation", {}),
    "penalty_torso_orientation": (-1.0, (LOWER,), f"{_MOD}:penalty_torso_orientation", {}),
    "penalty_lower_body_torques": (-1.0e-5, (LOWER, PC), f"{_MOD}:penalty_lower_body_torques", {}),
    "penalty_lower_body_dof_vel": (-1.0e-3, (LOWER, PC), f"{_MOD}:penalty_lower_body_dof_vel", {}),
    "penalty_lower_body_dof_acc": (-2.5e-7, (LOWER, PC), f"{_MOD}:penalty_lower_body_dof_acc", {}),
    "penalty_lower_body_action_rate": (-0.1, (LOWER, PC), f"{_MOD}:penalty_lower_body_action_rate", {}),
    "penalty_contact_no_vel": (-0.2, (LOWER,), f"{_MOD}:penalty_contact_no_vel", {}),
    "penalty_feet_ori": (-2.0, (LOWER,), f"{_LOCO}:penalty_feet_ori", {}),
    "limits_lower_body_dof_pos": (-5.0, (LOWER, PC), f"{_MOD}:limits_lower_body_dof_pos", {"soft_dof_pos_limit": 0.95}),
    "feet_air_time": (4.0, (LOWER,), f"{_MOD}:FeetAirTime", {"threshold_time": 0.5, "min_cmd_vel": 0.1}),
    "penalty_diff_feet_air_time": (-5.0, (LOWER,), f"{_MOD}:penalty_diff_feet_air_time", {}),
    "base_height": (-10.0, (LOWER,), f"{_MOD}:base_height", {"stance_penalty_scale": 5.0}),
    "termination": (-250.0, (LOWER,), f"{_LOCO}:termination", {}),
    "penalty_feet_height": (-5.0, (LOWER,), f"{_MOD}:penalty_feet_height", {"target": 0.11}),
    "penalty_feet_swing_height": (-20.0, (LOWER,), f"{_MOD}:penalty_feet_swing_height", {"target_walk": 0.11, "target_stance": 0.025}),
    "feet_heading_alignment": (-0.25, (LOWER,), f"{_MOD}:feet_heading_alignment", {}),
    "penalty_close_feet_xy": (-10.0, (LOWER,), f"{_MOD}:penalty_close_feet_xy", {"close_feet_threshold": 0.17}),
    "penalty_hip_pos": (-2.5, (LOWER,), f"{_MOD}:penalty_hip_pos", {}),
    "penalty_ang_vel_xy_torso": (-1.0, (LOWER,), f"{_MOD}:penalty_ang_vel_xy_torso", {}),
    "penalty_stance_dof": (-1.0e-3, (LOWER,), f"{_MOD}:penalty_stance_dof", {}),
    "penalty_stance_tap_feet": (-5.0, (LOWER,), f"{_MOD}:penalty_stance_tap_feet", {}),
    "penalty_stance_root": (-5.0, (LOWER,), f"{_MOD}:penalty_stance_root", {}),
    "penalty_stand_still": (-0.15, (LOWER,), f"{_MOD}:penalty_stand_still", {}),
    "penalty_stance_symmetry": (-0.5, (LOWER,), f"{_MOD}:penalty_stance_symmetry", {}),
    "penalty_ankle_roll": (-2.0, (LOWER,), f"{_MOD}:penalty_ankle_roll", {}),
    "penalty_contact": (-4.0, (LOWER,), f"{_MOD}:penalty_contact", {}),
    "penalty_negative_knee_joint": (-1.0, (LOWER,), f"{_MOD}:penalty_negative_knee_joint", {"knee_min": 0.2}),
    "penalty_shift_in_zero_command": (-1.5, (LOWER,), f"{_MOD}:penalty_shift_in_zero_command", {}),
    "penalty_ang_shift_in_zero_command": (-1.5, (LOWER,), f"{_MOD}:penalty_ang_shift_in_zero_command", {}),
    # ---- upper body (12) ----
    "tracking_upper_body_dofs": (4.0, (UPPER,), f"{_MOD}:tracking_upper_body_dofs", {"sigma": 0.1}),
    "penalty_upper_body_torques": (-1.0e-5, (UPPER, PC), f"{_MOD}:penalty_upper_body_torques", {}),
    "penalty_upper_body_dof_vel": (-1.0e-3, (UPPER, PC), f"{_MOD}:penalty_upper_body_dof_vel", {}),
    "penalty_upper_body_dof_acc": (-2.5e-7, (UPPER, PC), f"{_MOD}:penalty_upper_body_dof_acc", {}),
    "penalty_upper_body_action_rate": (-0.1, (UPPER, PC), f"{_MOD}:penalty_upper_body_action_rate", {}),
    "limits_upper_body_dof_pos": (-5.0, (UPPER, PC), f"{_MOD}:limits_upper_body_dof_pos", {"soft_dof_pos_limit": 0.95}),
    "limits_upper_body_dof_vel": (-5.0, (UPPER, PC), f"{_MOD}:limits_upper_body_dof_vel", {}),
    "limits_upper_body_torque": (-0.1, (UPPER, PC), f"{_MOD}:limits_upper_body_torque", {"soft_torque_limit": 0.9}),
    "penalty_ee_lin_acc": (-0.2, (UPPER,), f"{_MOD}:penalty_ee_lin_acc", {}),
    "penalty_ee_ang_acc": (-0.02, (UPPER,), f"{_MOD}:penalty_ee_ang_acc", {}),
    "tracking_stance_ee_pos": (2.0, (UPPER,), f"{_MOD}:tracking_stance_ee_pos", {"sigma": 0.05}),
    "tracking_stance_ee_rot": (2.0, (UPPER,), f"{_MOD}:tracking_stance_ee_rot", {"sigma": 0.10}),
}

feet_air_time = FeetAirTime

def build_hero_reward_terms(variant="h1", overrides=None):
    """Construct the paper's grouped reward terms, with optional parameter overrides."""
    from holosoma.config_types.reward import RewardTermCfg
    if variant != "h1":
        raise ValueError("HERO training uses the paper's reward terms.")
    return {name: RewardTermCfg(func=func, params={**params, **(overrides or {}).get(name, {})},
                               weight=weight, tags=list(tags))
            for name, (weight, tags, func, params) in HERO_109_REWARD_TABLE.items()}


def build_hero_reward_manager_cfg(variant="h1", overrides=None):
    """Wrap the grouped paper rewards in the training manager configuration."""
    from holosoma.config_types.reward import RewardManagerCfg
    return RewardManagerCfg(terms=build_hero_reward_terms(variant, overrides), only_positive_rewards=False)
