"""HERO tracking environment with grouped rewards, velocity histories and metrics.

Odometry anchors reset only on episode resets or clip rollovers, so external
perturbations remain visible as drift. Runtime quaternions use xyzw."""

from __future__ import annotations

import copy
import math
import os
from typing import Any, Sequence

import torch
from loguru import logger

from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.utils.average_meters import RatioScalar
from holosoma.utils.rotations import quat_error_magnitude

from hero_isaacsim.managers.command.final_observation import advance_reference_for_final_observation, bootstrap_env_ids
from hero_isaacsim.managers.observation.hero import hero_ee_residual_from_env, mark_odom_noise_reset
from hero_isaacsim.managers.reward.hero import GroupedRewardManager, invalidate_upright_ref_cache
from hero_isaacsim.managers.termination.hero import (
    GRACE_ACTIVE_ATTR,
    check_with_causes,
    invalidate_terrain_cache,
    reference_ground_z,
    split_done_flags,
)
from hero_isaacsim.utils import hero_constants
from hero_isaacsim.utils.ee_residual import quat_angle_xyzw

_C = hero_constants()
_RAD2DEG = 180.0 / math.pi

# ==================================================================================================
# Tensor helpers independent of the environment.
# ==================================================================================================


def wrap_to_pi(angles: torch.Tensor) -> torch.Tensor:
    """Wrap angles to (-π, π] — pure (no in-place mutation) and correct for any range.

    holosoma's ``utils.rotations.wrap_to_pi`` mutates its input and, with torch's truncated ``%=``, leaves angles in (-2π, -π) unwrapped; the metrics here need exact wrapping (e.g. −358° -> +2°)."""
    return torch.atan2(torch.sin(angles), torch.cos(angles))


def yaw_from_quat_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Heading angle (rad, (-π, π]) of xyzw quaternions ``[..., 4]`` (ZYX convention, yaw about world z)."""
    x, y, z, w = q.unbind(-1)
    return torch.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def odom_anchor(pos_w: torch.Tensor, quat_xyzw: torch.Tensor) -> torch.Tensor:
    """``[N, 3]`` odometry anchor ``(x, y, yaw)`` of a world pose (captured at episode start)."""
    return torch.stack([pos_w[:, 0], pos_w[:, 1], yaw_from_quat_xyzw(quat_xyzw)], dim=-1)


def _displacement_in_anchor_frame(pos_w: torch.Tensor, anchor: torch.Tensor) -> torch.Tensor:
    """Planar displacement ``R(-yaw0) (p_xy - p0_xy)`` -> ``[N, 2]``."""
    d = pos_w[:, :2] - anchor[:, :2]
    c, s = torch.cos(anchor[:, 2]), torch.sin(anchor[:, 2])
    return torch.stack([c * d[:, 0] + s * d[:, 1], -s * d[:, 0] + c * d[:, 1]], dim=-1)


def odometry_errors(
    robot_pos_w: torch.Tensor,
    robot_quat_xyzw: torch.Tensor,
    ref_pos_w: torch.Tensor,
    ref_quat_xyzw: torch.Tensor,
    robot_anchor: torch.Tensor,
    ref_anchor: torch.Tensor,
    elapsed_s: torch.Tensor,
) -> dict[str, torch.Tensor]:
    """The robot's planar displacement since the anchor, expressed in the robot's *start* heading frame, is
    compared with the reference root's displacement in the reference's start heading frame; the yaw error is
    the difference of the accumulated heading changes. When the robot is reset onto the reference pose (the
    holosoma default) both anchors coincide and this reduces to the plain world-frame root error; when the
    initial pose is randomised it still measures pure drift.

    Returns per-env tensors: ``odom_xy_err_m``, ``odom_yaw_err_rad``, ``height_err_m`` (|Δz|),
    ``drift_rate_m_per_s`` (= xy error / max(elapsed_s, tiny))."""
    d_rob = _displacement_in_anchor_frame(robot_pos_w, robot_anchor)
    d_ref = _displacement_in_anchor_frame(ref_pos_w, ref_anchor)
    xy_err = torch.norm(d_rob - d_ref, dim=-1)
    dyaw_rob = wrap_to_pi(yaw_from_quat_xyzw(robot_quat_xyzw) - robot_anchor[:, 2])
    dyaw_ref = wrap_to_pi(yaw_from_quat_xyzw(ref_quat_xyzw) - ref_anchor[:, 2])
    yaw_err = torch.abs(wrap_to_pi(dyaw_rob - dyaw_ref))
    height_err = torch.abs(robot_pos_w[:, 2] - ref_pos_w[:, 2])
    drift_rate = xy_err / elapsed_s.clamp(min=1e-6)
    return {
        "odom_xy_err_m": xy_err,
        "odom_yaw_err_rad": yaw_err,
        "height_err_m": height_err,
        "drift_rate_m_per_s": drift_rate,
    }


def ee_errors(dp: torch.Tensor, q_diff_xyzw: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """Per-hand EE errors from the HERO residual: ``(pos_err_m [N,K], rot_err_rad [N,K])``."""
    return torch.norm(dp, dim=-1), quat_angle_xyzw(q_diff_xyzw)


def split_means(values: torch.Tensor, walk_mask: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
    """``(stance_mean, walk_mean)`` of ``values [N]`` over the two command regimes, sync-free.

    An empty split (no env in that regime this step) falls back to the overall mean instead of NaN / 0 so the
    interval reduction done by the logger stays finite and unbiased in the common all-one-regime case."""
    values = values.float()
    walk = walk_mask.to(torch.bool)
    overall = values.mean()
    n_walk = walk.sum()
    n_stance = (~walk).sum()
    walk_mean = torch.where(n_walk > 0, (values * walk).sum() / n_walk.clamp(min=1), overall)
    stance_mean = torch.where(n_stance > 0, (values * ~walk).sum() / n_stance.clamp(min=1), overall)
    return stance_mean, walk_mean


def _quantile(values: torch.Tensor, q: float) -> torch.Tensor:
    v = values.reshape(-1).float()
    if v.numel() == 0:
        return torch.zeros((), device=values.device)
    return torch.quantile(v, q)


def odom_elapsed_s(env: Any) -> torch.Tensor:
    """Seconds since the odometry anchors were captured (hard reset or clip rollover), ``>= 1`` control step.

    Uses ``env._odom_anchor_step`` when the env keeps it (``HeroTrackingManager``), else the episode length."""
    steps = env.episode_length_buf
    anchor_step = getattr(env, "_odom_anchor_step", None)
    if anchor_step is not None and anchor_step.shape == steps.shape:
        steps = steps - anchor_step
    return steps.clamp(min=1).float() * float(env.dt)


def hero_log_metrics(env: Any, *, with_quantiles: bool = True, low_posture_h: float = 0.6) -> dict[str, torch.Tensor]:
    """Values are 0-dim tensors except ``termination/cause_*_frac_of_done`` (per-done indicator vectors, see
    :func:`termination_cause_metrics`).  Needs on ``env``: ``simulator`` tensors,
    ``command_manager.get_state("motion_command")`` (HeroMotionCommand contract), ``episode_length_buf``, ``dt``;
    optional ``_odom_anchor_robot/_odom_anchor_ref [N,3]`` (else the current poses are used as anchors -> zero
    drift), ``_odom_anchor_step [N]``, ``_termination_causes {name: bool[N]}``.  ``motion/source_frac_<tag>`` is
    only emitted when the command term's own ``metrics`` do not carry it (``env.log_dict`` is NOT consulted: it
    persists across steps and would freeze the fallback at its first value)."""
    mc = env.command_manager.get_state("motion_command")
    n = env.num_envs
    out: dict[str, torch.Tensor] = {}

    # ---- per-hand EE error (HERO residual: clip-pelvis-frame reference, full pelvis orientation) -------------
    walk = mc.stand_flag.reshape(-1) > 0.5
    dp, _, q_diff = hero_ee_residual_from_env(env)
    pos_err, rot_err = ee_errors(dp, q_diff)  # [N,2]
    rot_err_deg = rot_err * _RAD2DEG
    for i, side in enumerate(("left", "right")):
        for key, val in ((f"ee/{side}_pos_err_m", pos_err[:, i]), (f"ee/{side}_rot_err_deg", rot_err_deg[:, i])):
            out[key] = val.mean()
            stance_mean, walk_mean = split_means(val, walk)
            out[key + "_stance"] = stance_mean
            out[key + "_walk"] = walk_mean
    out["ee/pos_err_m"] = pos_err.mean()
    out["ee/rot_err_deg"] = rot_err_deg.mean()
    if with_quantiles:
        out["ee/pos_err_m_p50"] = _quantile(pos_err, 0.5)
        out["ee/pos_err_m_p90"] = _quantile(pos_err, 0.9)
        out["ee/rot_err_deg_p90"] = _quantile(rot_err_deg, 0.9)

    # ---- global base odometry -----------------------------------------------------------------------------
    root = env.simulator.robot_root_states
    robot_pos, robot_quat = root[:, 0:3], root[:, 3:7]
    ref_pos, ref_quat = mc.root_pos_w, mc.root_quat_w
    anchor_rob = getattr(env, "_odom_anchor_robot", None)
    anchor_ref = getattr(env, "_odom_anchor_ref", None)
    if anchor_rob is None or anchor_ref is None or anchor_rob.shape[0] != n:
        anchor_rob, anchor_ref = odom_anchor(robot_pos, robot_quat), odom_anchor(ref_pos, ref_quat)
    od = odometry_errors(robot_pos, robot_quat, ref_pos, ref_quat, anchor_rob, anchor_ref, odom_elapsed_s(env))
    out["base/odom_xy_err_m"] = od["odom_xy_err_m"].mean()
    out["base/odom_yaw_err_deg"] = (od["odom_yaw_err_rad"] * _RAD2DEG).mean()
    out["base/height_err_m"] = od["height_err_m"].mean()
    out["base/drift_rate_m_per_s"] = od["drift_rate_m_per_s"].mean()
    if with_quantiles:
        out["base/odom_xy_err_m_p90"] = _quantile(od["odom_xy_err_m"], 0.9)

    # ---- command regime / posture -------------------------------------------------------------------------
    h_cmd = mc.h_cmd.reshape(-1)
    out["cmd/walk_frac"] = walk.float().mean()
    out["cmd/h_cmd_mean"] = h_cmd.mean()
    out["motion/low_posture_frac"] = (h_cmd < low_posture_h).float().mean()
    fix_mask = getattr(mc, "fix_upper_body_mask", None)
    if fix_mask is not None:
        out["cmd/fix_upper_body_frac"] = fix_mask.reshape(-1).float().mean()

    # ---- upper-body joint error (arms 14 vs clip reference) -----------------------------------------------
    arm_idx = getattr(env, "hero_upper_dof_idx", None)
    if arm_idx is None:
        arm_idx = torch.tensor(list(_C.UPPER_DOF_IDX), dtype=torch.long, device=env.device)
        env.hero_upper_dof_idx = arm_idx
    ref_arms = mc.ref_upper_dof_pos[:, -len(_C.UPPER_DOF_IDX):]
    out["upper/arm_joint_err_rad"] = torch.abs(env.simulator.dof_pos[:, arm_idx] - ref_arms).mean()

    # ---- termination causes ---------------------------------------------------------------------------------
    out.update(termination_cause_metrics(env))


    already = set(getattr(mc, "metrics", {}).keys())
    tag_ids = getattr(mc, "source_tag_id", None)
    names = (
        getattr(mc, "source_tag_names", None)
        or getattr(mc, "source_tags", None)
        or getattr(getattr(mc, "motion", None), "source_tags", None)
    )
    if tag_ids is not None and names:
        ids = tag_ids.reshape(-1)
        for i, tag in enumerate(list(names)):
            key = f"motion/source_frac_{tag}"
            if key not in already:
                out[key] = (ids == i).float().mean()


    return out


def termination_cause_metrics(env: Any) -> dict[str, torch.Tensor]:
    """``termination/cause_<name>_rate`` (0-dim) and ``_frac_of_done`` from ``env._termination_causes`` (check_with_causes)."""
    out: dict[str, torch.Tensor] = {}
    causes = getattr(env, "_termination_causes", None) or {}
    done = getattr(env, "reset_buf", None)
    done_mask = done.reshape(-1).bool() if done is not None else None
    for name, mask in causes.items():
        m = mask.reshape(-1).float()
        out[f"termination/cause_{name}_rate"] = m.mean()
        if done_mask is not None:
            out[f"termination/cause_{name}_frac_of_done"] = m[done_mask]
    grace = getattr(env, GRACE_ACTIVE_ATTR, None)  # Share of environments in reset grace.
    if grace is not None and torch.is_tensor(grace):
        out["termination/reset_grace_active_frac"] = grace.reshape(-1).float().mean()
    return out


A_MAJOR_KEYS: tuple[tuple[str, str, float], ...] = (

    ("average_episode_length", "episode_len", 1.0),
    ("episode_len_true", "episode_len_true", 1.0),      # exact counter (no init_at_random_ep_len artefact)
    ("episode_success_frac", "episode_success_frac", 1.0),  # ended by clip end / timeout, not by a failure
    ("ee/left_pos_err_m", "ee_left_pos_err_cm", 100.0),
    ("ee/right_pos_err_m", "ee_right_pos_err_cm", 100.0),
    ("ee/pos_err_m", "ee_pos_err_cm", 100.0),
    ("ee/pos_err_m_p90", "ee_pos_err_p90_cm", 100.0),
    ("ee/left_pos_err_global_m", "ee_left_pos_err_global_cm", 100.0),   # world frame (incl. root drift)
    ("ee/right_pos_err_global_m", "ee_right_pos_err_global_cm", 100.0),
    ("ee/pos_err_global_m", "ee_pos_err_global_cm", 100.0),
    ("ee/left_rot_err_deg", "ee_left_rot_err_deg", 1.0),
    ("ee/right_rot_err_deg", "ee_right_rot_err_deg", 1.0),
    ("ee/rot_err_deg", "ee_rot_err_deg", 1.0),
    ("base/odom_xy_err_m", "odom_xy_err_cm", 100.0),
    ("base/odom_yaw_err_deg", "odom_yaw_err_deg", 1.0),
    ("base/height_err_m", "base_height_err_cm", 100.0),
    ("base/anchor_pos_err_m", "anchor_pos_err_cm", 100.0),
    ("upper/arm_joint_err_rad", "arm_joint_err_rad", 1.0),
    ("motion/error_joint_pos", "motion_joint_pos_err", 1.0),
    ("motion/error_body_pos", "motion_body_pos_err_cm", 100.0),
    ("motion/error_body_rot", "motion_body_rot_err", 1.0),
    ("motion/error_root_xy", "motion_root_xy_err_cm", 100.0),
    ("motion/error_root_z", "motion_root_z_err_cm", 100.0),
    ("motion/error_wrist_pos", "motion_wrist_pos_err_cm", 100.0),
    ("termination/failure_rate", "failure_rate", 1.0),

    ("obj/pos_err_m", "obj_pos_err_cm", 100.0),
    ("obj/ori_err_deg", "obj_ori_err_deg", 1.0),
    ("obj/lifted_frac", "obj_lifted_frac", 1.0),
    ("motion/env_has_object_frac", "env_has_object_frac", 1.0),
)


def a_major_metrics(log_dict: dict[str, Any]) -> dict[str, Any]:
    """``{"A_MAJOR/<name>": value * scale}`` for every source key present in ``log_dict`` (metres -> cm where named)."""
    out: dict[str, Any] = {}
    for src, name, scale in A_MAJOR_KEYS:
        if src in log_dict:
            v = log_dict[src]
            out[f"A_MAJOR/{name}"] = v * scale if scale != 1.0 else v
    return out


# ==================================================================================================
# the env
# ==================================================================================================


FOOT_ORDER_SIDES: tuple[str, str] = ("left", "right")


def foot_order_receipt(body_names: Sequence[str], feet_height_indices: Sequence[int], sides: Sequence[str] = FOOT_ORDER_SIDES) -> tuple[bool, str]:
    """Describe the foot-body order used by the simulator."""
    names = [str(body_names[int(i)]) for i in feet_height_indices]
    got = [next((side for side in sides if side in n), "?") for n in names]
    ok = got == list(sides)
    line = (
        f"[TERRAIN] foot order: feet_height_indices={[int(i) for i in feet_height_indices]} bodies={names} sides={got} "
        f"{'==' if ok else '!='} FOOT_BODIES order {list(sides)} -> {'ok' if ok else 'MISMATCH (foot-side terms assume left, right)'}"
    )
    return ok, line

class HeroTrackingManager(WholeBodyTrackingManager):
    """Track motion references with grouped rewards and reset-aware histories."""

    TASK_NAME = "hero_tracking"
    PROVENANCE_LOG_DIR_ENV = "HERO_LOG_DIR"

    # The motion configuration is taken as given: ``HeroMotionConfig`` refuses the backend's absolute clip cap in
    # per-clip mode at construction (the HERO sampler bounds clips relative to their prior share), so no corpus-size
    # fix-up of the cap is needed or performed here.

    # ------------------------------------------------------------------------------------------
    # naming / body indices
    # ------------------------------------------------------------------------------------------
    def _get_task_name(self) -> str:
        training_task_name = getattr(self.training_config, "task_name", None)
        if isinstance(training_task_name, str) and training_task_name:
            return training_task_name
        return self.TASK_NAME

    @staticmethod
    def _indices_by_substring(body_names: list[str], key: str) -> list[int]:
        return [i for i, name in enumerate(body_names) if key in name]

    def _setup_robot_body_indices(self) -> None:
        names = list(self.body_names)
        rc = self.robot_config
        dev = self.device

        def as_idx(idx: list[int]) -> torch.Tensor:
            return torch.tensor(idx, dtype=torch.long, device=dev)

        feet = self._indices_by_substring(names, rc.foot_body_name)
        if len(feet) != 2:
            raise ValueError(f"expected 2 feet matching {rc.foot_body_name!r}, got {[names[i] for i in feet]}")
        self.feet_indices = as_idx(feet)  # [left, right] by body order
        feet_height = self._indices_by_substring(names, rc.foot_height_name)
        if len(feet_height) == 2:
            self.feet_height_indices = as_idx(feet_height)
        else:
            logger.warning(
                "HeroTrackingManager: no '{}' bodies in the asset; feet_height_indices fall back to the feet "
                "(ankle_roll origin is {} m above the sole)",
                rc.foot_height_name,
                -_C.FOOT_CONTACT_POINT_OFFSET[2],
            )
            self.feet_height_indices = self.feet_indices.clone()
        ok, receipt = foot_order_receipt(names, self.feet_height_indices.tolist())
        (logger.info if ok else logger.warning)(receipt)
        term_contact: list[int] = []
        for key in rc.terminate_after_contacts_on:
            term_contact.extend(self._indices_by_substring(names, key))
        self.termination_contact_indices = as_idx(sorted(set(term_contact)))
        self.knee_indices = as_idx(self._indices_by_substring(names, rc.knee_name))
        if getattr(rc, "has_torso", True):
            self.torso_name = rc.torso_name
            self.torso_index = names.index(self.torso_name)
        else:
            self.torso_name = _C.TORSO_BODY_NAME
            self.torso_index = names.index(_C.TORSO_BODY_NAME) if _C.TORSO_BODY_NAME in names else 0


        self.ee_body_indices = as_idx([names.index(n) for n in _C.EE_BODY_NAMES])
        self.palm_body_indices = (
            as_idx([names.index(n) for n in _C.PALM_BODY_NAMES]) if all(n in names for n in _C.PALM_BODY_NAMES) else None
        )
        self.ee_vel_body_indices = self.palm_body_indices if self.palm_body_indices is not None else self.ee_body_indices
        self.ee_palm_offset = torch.tensor(
            [list(_C.PALM_OFFSET["left"]), list(_C.PALM_OFFSET["right"])], dtype=torch.float, device=dev
        )
        self.hero_lower_dof_idx = torch.tensor(list(_C.LOWER_DOF_IDX), dtype=torch.long, device=dev)
        self.hero_upper_dof_idx = torch.tensor(list(_C.UPPER_DOF_IDX), dtype=torch.long, device=dev)

    # ------------------------------------------------------------------------------------------
    # buffers
    # ------------------------------------------------------------------------------------------
    def _init_buffers(self) -> None:
        super()._init_buffers()
        n, dev = self.num_envs, self.device
        # Velocity histories for finite-difference penalties.
        self.last_dof_vel = torch.zeros(n, self.num_dof, dtype=torch.float, device=dev)
        self.last_ee_lin_vel = torch.zeros(n, 2, 3, dtype=torch.float, device=dev)
        self.last_ee_ang_vel = torch.zeros(n, 2, 3, dtype=torch.float, device=dev)
        # odometry anchors (x, y, yaw) captured after each reset / clip rollover, plus the episode step they were
        # captured at (elapsed time for drift_rate).  ``_odom_anchor_pending`` starts all-True so the first refresh
        # after construction / reset_all() captures every env (reset_all resets, then step() increments
        # episode_length_buf BEFORE the refresh, so an episode_length_buf == 0 test would miss it).
        self._odom_anchor_robot = torch.zeros(n, 3, dtype=torch.float, device=dev)
        self._odom_anchor_ref = torch.zeros(n, 3, dtype=torch.float, device=dev)
        self._odom_anchor_step = torch.zeros(n, dtype=torch.long, device=dev)
        self._odom_anchor_pending = torch.ones(n, dtype=torch.bool, device=dev)
        self._termination_causes: dict[str, torch.Tensor] = {}


        self._provenance_synced = False
        # exact per-env step counter: PPO's init_at_random_ep_len randomises episode_length_buf (desynchronised
        # timeouts), which inflates the first episode lengths after (re)start; this counter starts at 0 and is
        # zeroed on every reset, so A_MAJOR/episode_len_true has no start-up artefact.
        self._ep_steps = torch.zeros(n, dtype=torch.long, device=dev)
        # A monotonic clock lets observation noise advance once per control step
        # even when an episode terminates on its first step.
        self.control_step_counter = 0
        self._ep_len_true_ema = torch.zeros((), dtype=torch.float, device=dev)
        self._ep_len_true_count = 0
        # grouped rewards for PPODual (wrap BEFORE curriculum setup so weight rescaling hits the wrapped manager)
        self._ensure_grouped_reward_manager()

    def _ensure_grouped_reward_manager(self) -> None:
        rm = getattr(self, "reward_manager", None)
        if rm is None or hasattr(rm, "group_rewards"):
            return
        self.reward_manager = GroupedRewardManager.wrap(rm)
        if self.reward_manager.ungrouped_terms:
            logger.warning(
                "HeroTrackingManager: reward terms without lower_body/upper_body tag: {}",
                self.reward_manager.ungrouped_terms,
            )

    # ------------------------------------------------------------------------------------------
    # step hooks
    # ------------------------------------------------------------------------------------------
    def _update_counters_each_step(self) -> None:
        super()._update_counters_each_step()
        self.control_step_counter += 1

    def _compute_final_observations(self, env_ids=None):
        """Compute final observations."""
        if env_ids is None or getattr(self, "_update_tasks_before_termination", False):
            return super()._compute_final_observations(env_ids)
        peek_ids = bootstrap_env_ids(self, env_ids)
        if peek_ids.numel() == 0:
            return super()._compute_final_observations(env_ids)
        with advance_reference_for_final_observation(self, peek_ids):
            return super()._compute_final_observations(env_ids)

    def _check_termination(self) -> None:
        invalidate_upright_ref_cache(self)
        self.reset_buf[:] = 0
        self.time_out_buf[:] = 0
        if self.termination_manager is None:
            return
        reset_flags, timeout_flags, causes = check_with_causes(self.termination_manager, self)
        self._termination_causes = causes


        failed, timed_out_only, overlap = split_done_flags(reset_flags, timeout_flags)
        self._failure_and_timeout = overlap
        self.reset_buf |= failed.to(dtype=self.reset_buf.dtype)
        self.time_out_buf |= timed_out_only
        self.termination_manager.time_outs = self.time_out_buf.clone()
        self.reset_buf |= self.time_out_buf

    def _compute_reward(self) -> None:
        invalidate_upright_ref_cache(self)
        rm = self.reward_manager
        self.rew_buf[:] = rm.compute(self.dt)
        self.episode_sums = getattr(rm, "episode_sums", {})
        self.episode_sums_raw = getattr(rm, "episode_sums_raw", {})
        groups = getattr(rm, "group_rewards", None)
        if groups is not None:
            self.extras["rewards_by_group"] = {g: r.clone() for g, r in groups.items()}
        self._update_velocity_history()

    def _current_ee_velocities(self) -> tuple[torch.Tensor, torch.Tensor]:
        idx = self.ee_vel_body_indices
        return self.simulator._rigid_body_vel[:, idx, :], self.simulator._rigid_body_ang_vel[:, idx, :]

    def _update_velocity_history(self, env_ids: torch.Tensor | None = None) -> None:
        """Copy the current joint / EE velocities into the ``last_*`` buffers (all envs or ``env_ids``)."""
        lin, ang = self._current_ee_velocities()
        if env_ids is None:
            self.last_dof_vel.copy_(self.simulator.dof_vel)
            self.last_ee_lin_vel.copy_(lin)
            self.last_ee_ang_vel.copy_(ang)
        else:
            self.last_dof_vel[env_ids] = self.simulator.dof_vel[env_ids]
            self.last_ee_lin_vel[env_ids] = lin[env_ids]
            self.last_ee_ang_vel[env_ids] = ang[env_ids]

    def _capture_odom_anchors(self, env_ids: torch.Tensor) -> None:
        root = self.simulator.robot_root_states
        mc = self.command_manager.get_state("motion_command")
        self._odom_anchor_robot[env_ids] = odom_anchor(root[env_ids, 0:3], root[env_ids, 3:7])
        self._odom_anchor_ref[env_ids] = odom_anchor(mc.root_pos_w[env_ids], mc.root_quat_w[env_ids])
        anchor_step = getattr(self, "_odom_anchor_step", None)
        if anchor_step is not None:
            anchor_step[env_ids] = self.episode_length_buf[env_ids]

    def _reset_buffers_callback(self, env_ids, target_buf=None):
        super()._reset_buffers_callback(env_ids, target_buf)
        # WBT's callback is the only reset-side writer of need_to_refresh_envs (_push_robots is the other writer);
        # remember which of the refreshed envs were actually reset so only those get new odometry anchors.
        self._odom_anchor_pending[self._ensure_long_tensor(env_ids)] = True


        mark_odom_noise_reset(self, env_ids)


    def _compute_observations(self) -> None:
        """Refresh the terrain cache after resets move the robot.

        Terrain-aware observations must sample the ground at the new position
        rather than reuse the cached height from the reward pass."""
        invalidate_terrain_cache(self)
        super()._compute_observations()


    def _refresh_envs_after_reset(self, env_ids) -> None:


        super()._refresh_envs_after_reset(env_ids)  # writes states, refreshes sim tensors
        env_ids = self._ensure_long_tensor(env_ids)
        if env_ids.numel() == 0:
            return
        self._update_velocity_history(env_ids)  # harmless for pushed envs (same values as the reward step wrote)
        pending = env_ids[self._odom_anchor_pending[env_ids]]
        if pending.numel() > 0:
            self._capture_odom_anchors(pending)
            self._odom_anchor_pending[pending] = False

    def _post_soft_reset(self, env_ids) -> None:
        """Re-seed the per-env history after a clip rollover (called by
        ``HeroMotionCommand._soft_reset_ended_clips``).

        The soft reset teleports the robot onto a new clip inside ``command_manager.step()`` and writes the simulator
        state itself, so ``_refresh_envs_after_reset`` never sees these envs: without this hook the next step's
        dof / EE acceleration penalties difference the pre-teleport velocities and the odometry anchors keep the
        previous clip's start pose.  The sim tensors are already refreshed when this runs."""
        env_ids = self._ensure_long_tensor(env_ids)
        if env_ids.numel() == 0:
            return
        self._update_velocity_history(env_ids)
        self._capture_odom_anchors(env_ids)
        self._odom_anchor_pending[env_ids] = False


    # ------------------------------------------------------------------------------------------
    # curriculum plumbing
    # ------------------------------------------------------------------------------------------
    @property
    def average_episode_length(self) -> float:
        """EMA of finished-episode lengths (steps), read by holosoma's ``PenaltyCurriculum`` / level curricula.

        ``LocomotionManager`` exposes this property; ``WholeBodyTrackingManager`` only logs it, so the HERO env
         provides it here. 0.0 until the
        ``average_episode_tracker`` curriculum term exists."""
        tracker = self.curriculum_manager.get_term("average_episode_tracker")
        if tracker is None:
            return 0.0
        return float(tracker.get_average().detach().cpu().item())

    # logging
    # ------------------------------------------------------------------------------------------
    def _update_log_dict(self) -> None:
        super()._update_log_dict()  # avg episode length, done/timeout rates, motion metrics, object SR
        self._update_true_episode_length()
        try:
            mc = self.command_manager.get_state("motion_command")
            metrics = hero_log_metrics(self)
            self.log_dict.update(metrics)
            self.log_dict.update(a_major_metrics(self.log_dict))
        except Exception as exc:  # noqa: BLE001 — metrics must never kill training
            if not getattr(self, "_metrics_warned", False):
                self._metrics_warned = True
                logger.warning("HeroTrackingManager: metrics skipped ({}); further warnings suppressed", exc)
        self._sync_provenance_once()

    def _update_true_episode_length(self) -> None:
        """EMA (over finished episodes) of the exact episode length; ``episode_success_frac`` = share of finished
        episodes that ended by clip end / timeout rather than a failure (unaffected by clip-length ceilings)."""
        self._ep_steps += 1
        done = self.reset_buf.bool()
        n_done = int(done.sum().item())
        if n_done > 0:
            mean_len = self._ep_steps[done].float().mean()
            w = min(n_done / 10000.0, 1.0)
            if self._ep_len_true_count == 0:
                self._ep_len_true_ema = mean_len
            else:
                self._ep_len_true_ema = self._ep_len_true_ema * (1.0 - w) + mean_len * w
            self._ep_len_true_count += n_done
            self._ep_steps[done] = 0
        self.log_dict["episode_len_true"] = self._ep_len_true_ema.detach()
        tf = self.log_dict.get("termination/timeout_frac_of_done")
        if tf is not None:
            self.log_dict["episode_success_frac"] = tf
        overlap = getattr(self, "_failure_and_timeout", None)
        if overlap is not None:
            self.log_dict["termination/failure_and_timeout_frac_of_done"] = RatioScalar(overlap.bool().sum(), done.sum())

    def _action_contract_string(self) -> str | None:
        term = getattr(self, "hero_joint_action_term", None)
        contract = getattr(term, "action_contract", None)
        return str(contract) if contract is not None else None

    @property
    def action_contract(self) -> str | None:
        """Action contract used in ONNX metadata, the export sidecar, and training logs.

        Returns None when no HERO action term is configured."""
        return self._action_contract_string()

    # ------------------------------------------------------------------------------------------
    # checkpoint env state
    # ------------------------------------------------------------------------------------------
    def _adaptive_sampler_state_problem(self, sampler_state: Any) -> str | None:
        """Why the live command cannot load ``sampler_state`` (None if it can).

        Validates on a probe (shallow copy of the sampler with cloned count tensors): the stock loader validates
        the registry and the sampling policy before touching the table and the HERO subclass only adds read-only
        checks, so the probe reproduces every refusal without mutating the live sampler."""
        mc = self.command_manager.get_state("motion_command")
        if mc is None or not getattr(getattr(mc, "motion_cfg", None), "use_adaptive_timesteps_sampler", False):
            return "the live task has the adaptive sampler disabled"
        sampler = getattr(mc, "adaptive_timesteps_sampler", None)
        if sampler is None:
            return "the live command has no adaptive sampler"
        probe = copy.copy(sampler)
        for name in ("bin_failed_count", "current_bin_failed_count"):
            buf = getattr(sampler, name, None)
            if isinstance(buf, torch.Tensor):
                setattr(probe, name, buf.clone())
        try:
            probe.load_state_dict(sampler_state)
        except (ValueError, KeyError, TypeError) as exc:
            return f"{type(exc).__name__}: {exc}"
        return None


    H_CURRICULUM_STATE_KEY = "hero_h_curriculum_scale"
    """Checkpoint key of the motion command's per-environment height-offset curriculum scale."""

    def _motion_command(self) -> Any:
        try:
            return self.command_manager.get_state("motion_command")
        except Exception:  # noqa: BLE001
            return None

    def reset_all(self):
        """The forced all-env reset (agent construction, ``learn()`` entry) is not an episode end for the height-offset
        curriculum either: flag the motion command so its next ``_update_h_curriculum`` leaves the per-env scale alone
        (the WBT base already suppresses the episode-length tracker and the sampler table for the same reset)."""
        motion_command = self._motion_command()
        if motion_command is not None and hasattr(motion_command, "_skip_h_curriculum_update_once"):
            motion_command._skip_h_curriculum_update_once = True
        return super().reset_all()

    def get_checkpoint_state(self) -> dict[str, Any]:
        """The WBT state (episode-length tracker, curriculum terms, sampler table) plus the height-offset curriculum."""
        state = dict(super().get_checkpoint_state())
        scale = getattr(self._motion_command(), "h_curriculum_scale", None)
        if isinstance(scale, torch.Tensor):
            state[self.H_CURRICULUM_STATE_KEY] = scale.detach().cpu().clone()
        return state

    def _reset_sampler_on_resume(self) -> bool:
        cfg = getattr(self._motion_command(), "motion_cfg", None)
        return bool(getattr(cfg, "reset_sampler_on_resume", False))

    def _restore_h_curriculum_scale(self, saved: Any) -> None:
        """Copy the saved per-env scale into the live buffer; a changed environment count gets the saved mean everywhere."""
        live = getattr(self._motion_command(), "h_curriculum_scale", None)
        if not isinstance(live, torch.Tensor):
            raise ValueError(
                f"checkpoint carries {self.H_CURRICULUM_STATE_KEY} but the live motion command has no height-offset curriculum"
            )
        saved = torch.as_tensor(saved, dtype=live.dtype).reshape(-1)
        if saved.numel() == 0 or not bool(torch.isfinite(saved).all()):
            raise ValueError(f"checkpoint {self.H_CURRICULUM_STATE_KEY} must be a non-empty finite vector")
        if saved.numel() != live.numel():
            mean = float(saved.mean().item())
            logger.warning(
                "{}: checkpoint has {} environments, this run {} -> every environment starts at the saved mean {:.4f}",
                self.H_CURRICULUM_STATE_KEY, saved.numel(), live.numel(), mean,
            )
            saved = torch.full((live.numel(),), mean, dtype=live.dtype)
        # Defensive: the in-place copy works whether or not the buffer was created under inference mode (an ordinary
        # tensor accepts it inside the region too; an inference tensor only inside it).
        with torch.inference_mode():
            live.copy_(saved.to(device=live.device))

    def load_checkpoint_state(self, state: dict[str, Any] | None) -> None:
        """Restore environment state; an adaptive sampler table the live sampler cannot load is an error.

        A table written under another sampling rule, with other settings or for another clip registry would be
        restarted from zeros, i.e. the run would silently continue on a different training distribution. Only
        ``reset_sampler_on_resume`` (``scripts/train.py --reset-sampler-on-resume``) permits that, with a warning."""
        if not state:
            return
        state = dict(state)  # non-empty: a resume (the warning below keys off that, not off what survives the sampler pop)
        sampler_state = state.get("adaptive_timesteps_sampler")
        if sampler_state is not None:
            problem = self._adaptive_sampler_state_problem(sampler_state)
            if problem is not None:
                if not self._reset_sampler_on_resume():
                    raise ValueError(
                        "the checkpoint's adaptive sampler state cannot be restored by the live configuration -- "
                        f"{problem}. Resuming would silently train on a different clip distribution; pass "
                        "--reset-sampler-on-resume to scripts/train.py to restart the failure table from zeros instead."
                    )
                logger.warning("Restarting the adaptive sampler from zeros (reset_sampler_on_resume): {}", problem)
                state.pop("adaptive_timesteps_sampler")
        h_scale = state.pop(self.H_CURRICULUM_STATE_KEY, None)
        super().load_checkpoint_state(state)
        if h_scale is not None:
            self._restore_h_curriculum_scale(h_scale)
        else:
            self._warn_h_curriculum_restart()

    def _warn_h_curriculum_restart(self) -> None:
        """A checkpoint written before the height-offset curriculum was persisted carries the tracker / sampler table but
        no per-environment scale: every environment restarts at ``h_curriculum_init``. Say so (the pre-flight in
        ``scripts/train.py`` reports the same finding before Isaac Sim starts)."""
        motion_command = self._motion_command()
        live = getattr(motion_command, "h_curriculum_scale", None)
        if not isinstance(live, torch.Tensor):
            return
        cfg = getattr(motion_command, "hero_cfg", None) or getattr(motion_command, "motion_cfg", None)
        init = getattr(cfg, "h_curriculum_init", None)
        init = float(init) if isinstance(init, (int, float)) else float(live.float().mean().item())
        logger.warning(
            "checkpoint env_state has no '{}' (written before the height-offset curriculum was checkpointed); the "
            "per-environment height-offset curriculum restarts at h_curriculum_init={:g}",
            self.H_CURRICULUM_STATE_KEY, init,
        )

    def _command_provenance_block(self) -> tuple[dict[str, Any], str | None]:
        """``(HeroMotionCommand.provenance copy, its first manifest path)`` -- ``({}, None)`` for the stock command or
        before the command's setup.  Never raises."""
        try:
            mc = self.command_manager.get_state("motion_command")
        except Exception:  # noqa: BLE001
            return {}, None
        prov = getattr(mc, "provenance", None)
        block = dict(prov) if isinstance(prov, dict) else {}
        manifest = getattr(mc, "manifest_path", None)
        return block, (str(manifest) if manifest else None)

    def _sync_provenance_once(self) -> None:
        """Push run provenance to wandb (+provenance.json) once per process; best effort.  Includes the HERO command's
        ``provenance`` block under ``command`` and falls back to its manifest path when ``$CORPUS_MANIFEST`` is unset."""
        if self._provenance_synced:
            return
        self._provenance_synced = True
        try:
            from hero_isaacsim.logging import provenance as P  # noqa: PLC0415

            prov = getattr(self, "provenance", None)
            if prov is None:
                extra: dict[str, Any] = {
                    "task_name": self._get_task_name(),
                    "reward_terms": list(getattr(self.reward_manager, "active_terms", [])),
                    "reward_groups": {
                        g: self.reward_manager.terms_in_group(g)
                        for g in getattr(self.reward_manager, "group_tags", ())
                    },
                    "termination_terms": list(getattr(self.termination_manager, "_term_names", [])),
                    "obs_groups": {
                        g: sorted(cfg.terms.keys()) for g, cfg in self.observation_manager.cfg.groups.items()
                    },
                    "history_length": dict(getattr(self, "history_length", {})),
                    "robot_preset": getattr(self.robot_config.asset, "urdf_file", None),
                }


                command_block, command_manifest = self._command_provenance_block()
                if command_block:
                    extra["command"] = command_block
                world_size = int(os.environ.get("WORLD_SIZE", "1") or 1)
                kwargs: dict[str, Any] = dict(
                    arm=os.environ.get("HERO_ARM", self._get_task_name()),
                    corpus_manifest=os.environ.get("CORPUS_MANIFEST") or command_manifest,
                    seed=int(getattr(self.training_config, "seed", 0) or 0),
                    global_num_envs=self.num_envs * world_size,
                    envs_per_rank=self.num_envs,
                    extra=extra,
                )
                contract = self._action_contract_string()
                if contract is not None:
                    kwargs["action_contract"] = contract
                prov = P.collect_provenance(**kwargs)
            log_dir = getattr(self, "provenance_log_dir", None) or os.environ.get(self.PROVENANCE_LOG_DIR_ENV)
            P.sync_once(prov, log_dir)
        except Exception as exc:  # noqa: BLE001
            logger.warning("HeroTrackingManager: provenance sync skipped: {}", exc)

