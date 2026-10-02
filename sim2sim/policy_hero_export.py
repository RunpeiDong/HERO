"""Read and execute a HERO dual-actor ONNX export.

The matching JSON sidecar or embedded ONNX metadata defines the observation
terms, history order, actuator parameters, residual reference, and anchor
horizon. Both anchor and no-anchor models use this implementation. Graph input
widths are checked against the exported observation layout. Object channels are supported for exports that include them."""

from __future__ import annotations

import json
import os
import re
from collections import deque
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hero_isaacsim import constants as HC
from sim2sim import plant_params as PP
from sim2sim.mathutil import quat_apply_inv, quat_conj, quat_from_euler_xyz, quat_mul, quat_to_mat, yaw_quat
from sim2sim.observation_math import ee_residual_parts, object_state_10
#: Root-pose sources of the anchor terms: exact simulator state, a leg-odometry estimate, or the LiDAR-inertial odometry model of the
#: benchmark runner (``sim2sim.bench.odometry``; ``RobotState.odom`` exposes ``pos_w`` / ``quat_xyzw`` / ``lin_vel_w`` for either estimator).
ODOM_SOURCES = ("truth", "leg", "so")

# ================================================================================================ contract tables
SIDECAR_SUFFIX = "_hero.json"
SIDECAR_SCHEMA = "hero_export_v1"  # == ppo_dual.export.HERO_SIDECAR_SCHEMA (a different value only warns: the fields decide)
SIDECAR_SCHEMAS_KNOWN: tuple[str, ...] = (SIDECAR_SCHEMA, "hero_dual_export_v1")  # Accepted sidecar schema identifiers.
CONTROL_DT = 0.02  # HERO / holosoma G1 policy step (50 Hz); the sidecar ``policy_dt`` is cross-checked against it
HISTORY_LAYOUT_FRAME_MAJOR = "frame_major_hero_v1"
HISTORY_LAYOUT_TERM_MAJOR = "term_major_holosoma_v1"
HISTORY_LAYOUTS: tuple[str, ...] = (HISTORY_LAYOUT_FRAME_MAJOR, HISTORY_LAYOUT_TERM_MAJOR)
CONTRACT_RESIDUAL_UPPER = "hero_residual_upper_v1"
CONTRACT_RESIDUAL_ALL29 = "hero_residual_all29_v1"
CONTRACTS: tuple[str, ...] = (CONTRACT_RESIDUAL_UPPER, CONTRACT_RESIDUAL_ALL29)
ONNX_INPUT_NAMES: tuple[str, str] = ("actor_obs_lower_body", "actor_obs_upper_body")
ONNX_OUTPUT_NAME = "action"
ACTOR_GROUP = "actor_obs"
HERO_OBS_CLIP = 100.0  # make_hero_observation(clip_observations=100.0)
HERO_ACTION_CLIP = 100.0  # Default G1 policy action clipping limit.
HERO_ACTION_SCALE_BASE = 0.25  # per-joint action_scales = 0.25 * effort_limit / kp
H_CMD_MODES: tuple[str, ...] = ("auto", "clip", "fixed")
HOLD_MODES: tuple[str, ...] = ("zero_action", "clip")
OBJECT_OBS_MODES: tuple[str, ...] = ("auto", "on", "off")
#: ``--hero-export-object-rule``: auto = apply the size / start-height rule when the command carries it, on = require it,
#: off = the clip's raw ``has_object``.
OBJECT_RULE_MODES: tuple[str, ...] = ("auto", "on", "off")
DEFAULT_BOX_SIZE_M = 0.30  # Fallback side length for clips without box_size.
OBJECT_RULE_FIELDS: tuple[str, ...] = ("object_box_side", "object_box_tol", "object_box_size_missing_ok", "max_start_bottom_z_m")
TAG_PREFIX = "hero_export"

#: Single-frame dims of the HERO actor terms (== hero_isaacsim.config_values.observation.HERO_TERM_DIMS; asserted in tests).
HERO_TERM_DIMS: dict[str, int] = {
    "h00_actions": 29,
    "h01_base_ang_vel": 3,
    "h02_command_ang_vel": 1,
    "h03_command_base_height": 1,
    "h04_command_lin_vel": 2,
    "h05_command_stand": 1,
    "h06_command_waist_dofs": 3,
    "h07_dif_local_rigid_body_pos_ee": 6,
    "h08_dif_local_rigid_body_rot_ee": 12,
    "h09_dof_pos": 29,
    "h10_dof_vel": 29,
    "h11_projected_gravity": 3,
    "h12_ref_upper_dof_pos": 14,
    "h13_roll_and_pitch": 2,
    "h14_ref_lower_dof_pos": 12,
    "h15_ref_root_pitch_roll": 2,
    "h16_ref_body_pos_b": 12,
    "h17_obj_pos_b": 3,
    "h18_obj_ori_b": 6,
    "h19_has_object_flag": 1,
    "h20_ref_root_pose_b": 20,
    "h21_ref_root_rot_b": 30,
    "h22_ref_root_height_b": 5,
    "h23_base_lin_vel_odom": 3,
}
#: ObsTermCfg scales (== HERO_TERM_SCALES of the config module; 1.0 elsewhere).
HERO_TERM_SCALES: dict[str, float] = {"h01_base_ang_vel": 0.25, "h03_command_base_height": 2.0, "h10_dof_vel": 0.05}
H1_TERMS: tuple[str, ...] = (
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
)
#: HERO's residual end-effector feedback (palm position / rotation errors); absent from the ``without_delta_ee`` ablation exports.
DELTA_EE_TERMS: tuple[str, ...] = ("h07_dif_local_rigid_body_pos_ee", "h08_dif_local_rigid_body_rot_ee")
#: Terms every HERO export must carry (H1 minus the optional feedback pair).
H1_REQUIRED_TERMS: tuple[str, ...] = tuple(t for t in H1_TERMS if t not in DELTA_EE_TERMS)
H2_EXTRA_TERMS: tuple[str, ...] = ("h14_ref_lower_dof_pos", "h15_ref_root_pitch_roll", "h16_ref_body_pos_b")
OBJECT_TERMS: tuple[str, ...] = ("h17_obj_pos_b", "h18_obj_ori_b", "h19_has_object_flag")
#: Planar delta-anchor term: reference root pose relative to the robot root in the robot heading frame.
ANCHOR_TERM = "h20_ref_root_pose_b"
ANCHOR_TERMS: tuple[str, ...] = (ANCHOR_TERM,)
#: Future frames of h20 (control steps ahead of the stock frame ``t``; 0-0.4 s at 50 Hz) and the per-frame layout.
H20_FUTURE_STEPS: tuple[int, ...] = (0, 5, 10, 15, 20)
H20_PER_FRAME: tuple[str, ...] = ("dx", "dy", "sin_dyaw", "cos_dyaw")

#: Relative root orientation and height use future frames; velocity uses the current heading frame.
#: h21 / h22 share h20's future frames (``HeroCommandParams.ref_root_pose_future_steps``).
ANCHOR2_ROT_TERM = "h21_ref_root_rot_b"
ANCHOR2_HEIGHT_TERM = "h22_ref_root_height_b"
ANCHOR2_VEL_TERM = "h23_base_lin_vel_odom"
#: The optional delta anchor adds root orientation and height feedback.
ANCHOR2_TERMS: tuple[str, ...] = (ANCHOR2_ROT_TERM, ANCHOR2_HEIGHT_TERM)
#: Additional root feedback terms are built only when present in the exported layout.
ANCHOR2V_TERMS: tuple[str, ...] = (ANCHOR2_ROT_TERM, ANCHOR2_HEIGHT_TERM, ANCHOR2_VEL_TERM)

#: Diagnostic on-track inputs retain the trained input width: h20 [0,0,0,1],
#: h21 identity rotation, h22 zero height error. This is not a no-anchor model.
ANCHOR_OBS_MODES: tuple[str, ...] = ("live", "ontrack")
#: All supported root-feedback terms in sorted order.
HERO_PLUS_TERMS: tuple[str, ...] = (ANCHOR_TERM, *ANCHOR2V_TERMS)
#: Per-future-frame layout of h21: the first two columns of ``R_rel`` COLUMN-major (identity -> [1 0 0 0 1 0]); h22 is one metre per frame.
H21_PER_FRAME: tuple[str, ...] = ("r00", "r10", "r20", "r01", "r11", "r21")
H21_FRAME_DIM = len(H21_PER_FRAME)  # 6
H22_FRAME_DIM = 1
#: Per-frame widths of the root-feedback terms with a horizon; h23 has none.
HERO_PLUS_PER_FRAME_DIM: dict[str, int] = {ANCHOR_TERM: len(H20_PER_FRAME), ANCHOR2_ROT_TERM: H21_FRAME_DIM, ANCHOR2_HEIGHT_TERM: H22_FRAME_DIM}
#: Sidecar / ONNX-metadata keys of the per-term horizons (== export.HERO_FUTURE_STEPS_KEYS).
HERO_PLUS_FUTURE_STEPS_KEYS: dict[str, str] = {ANCHOR_TERM: "h20_future_steps", ANCHOR2_ROT_TERM: "h21_future_steps", ANCHOR2_HEIGHT_TERM: "h22_future_steps"}
ANCHOR_TERMS_KEY = "anchor_terms"  # == export.HERO_ANCHOR_TERMS_KEY (per-term provenance block)

ODOM_PLANAR_FIELDS: tuple[str, ...] = ("bias_xy_m", "bias_yaw_rad", "walk_xy_m", "walk_yaw_rad")
ODOM_ANCHOR2_FIELDS: tuple[str, ...] = ("bias_rp_rad", "bias_z_m", "walk_z_m", "vel_std_m_s")
#: h16 bodies (in order) -- default of ``h16_ref_body_pos_b``.
H16_BODY_NAMES: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link", "left_knee_link", "right_knee_link")
GRAVITY_DOWN = np.array([0.0, 0.0, -1.0])
_LEG = np.asarray(HC.LEG_DOF_IDX, dtype=np.int64)
_WAIST = np.asarray(HC.WAIST_DOF_IDX, dtype=np.int64)
_ARM = np.asarray(HC.ARM_DOF_IDX, dtype=np.int64)
_UPPER17 = np.asarray(HC.UPPER_REF_DOF_IDX, dtype=np.int64)
_SIDECAR_RE = re.compile(r"^(?P<stem>.+)_hero\.json$")
PADDLE_URDF_FILE_NAME = "g1_29dof_paddle3box_hero.urdf"  # Filename recognized for compatible paddle-hand exports.


# ================================================================================================ command params
@dataclass(frozen=True)
class HeroCommandParams:
    """Motion-command parameters used to reconstruct the exported policy inputs."""

    arm: str = "h1"
    ref_lookahead_frames: int = 1
    h_cmd_default: float = 0.75
    h_cmd_min: float = 0.15
    stand_speed_thr: float = 0.15
    stand_yaw_rate_thr: float = 0.2
    stand_window_s: float = 0.5
    h_offset_from_clip: bool = False  # True for clip-driven commands.
    stand_flag_mode: str = "bernoulli_walk_0p6"  # "from_clip" for clip-driven commands.
    zero_waist_when_walking: bool = True  # False for clip-driven commands.
    # ---- object rule ( None = that rule is off, both None = raw has_object) ----
    object_box_side: float | None = None
    object_box_tol: float = 0.02
    object_box_size_missing_ok: bool = True
    max_start_bottom_z_m: float | None = None
    # ---- h20 (HERO): future frames of the reference-root-pose term, control steps ahead of the stock frame ``t`` ----
    ref_root_pose_future_steps: tuple[int, ...] = H20_FUTURE_STEPS

    def __post_init__(self) -> None:
        if self.object_box_side is not None and not float(self.object_box_side) > 0.0:
            raise ValueError("object_box_side must be > 0 (m) or None")
        if float(self.object_box_tol) < 0.0:
            raise ValueError("object_box_tol must be >= 0")
        steps = tuple(int(v) for v in np.asarray(self.ref_root_pose_future_steps, dtype=np.int64).reshape(-1).tolist())
        if not steps or any(v < 0 for v in steps) or len(set(steps)) != len(steps):
            raise ValueError(f"ref_root_pose_future_steps must be distinct non-negative ints, got {self.ref_root_pose_future_steps!r}")
        object.__setattr__(self, "ref_root_pose_future_steps", steps)

    @classmethod
    def for_arm(cls, arm: str, **overrides: Any) -> "HeroCommandParams":
        if arm not in ("h1", "h2"):
            raise ValueError(f"arm must be 'h1' or 'h2', got {arm!r}")
        kw: dict[str, Any] = dict(arm=arm)
        if arm == "h2":
            kw.update(h_offset_from_clip=True, stand_flag_mode="from_clip", zero_waist_when_walking=False)
        kw.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**kw)

    @property
    def object_rule_active(self) -> bool:
        """True when at least one of the size / start-height rules is set (``_compute_clip_object_effective`` semantics)."""
        return self.object_box_side is not None or self.max_start_bottom_z_m is not None

    def object_rule(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in OBJECT_RULE_FIELDS} | {"active": self.object_rule_active}

    def stand_half_window(self, fps: float) -> int:
        """``HeroMotionCommand._prepare_clip_kinematic_stats``: ``max(int(round(0.5 * window_s * fps)), 0)`` (Python round)."""
        return max(int(round(0.5 * float(self.stand_window_s) * float(fps))), 0)

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__}


# ================================================================================================ term math (numpy)
def euler_roll_pitch(q_xyzw: np.ndarray) -> np.ndarray:
    """``managers.observation.hero.euler_roll_pitch_xyzw``: (roll, pitch) from an xyzw quaternion (ZYX; pitch = asin(clamp))."""
    x, y, z, w = (float(v) for v in np.asarray(q_xyzw, dtype=np.float64).reshape(4))
    roll = np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y))
    pitch = np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0))
    return np.array([roll, pitch], dtype=np.float64)


def projected_gravity(root_quat: np.ndarray) -> np.ndarray:
    """holosoma ``get_projected_gravity``: ``R_root^T (0, 0, -1)``."""
    return quat_apply_inv(np.asarray(root_quat, dtype=np.float64).reshape(4), GRAVITY_DOWN)


def ee_residual_terms(root_pos, root_quat, palm_pos_w, palm_quat_w, ref_pos_local, ref_quat_local) -> tuple[np.ndarray, np.ndarray]:
    """h07 (6) / h08 (12): ``hero_ee_residual`` of the palm points [left, right] (``sim2sim.observation_math.ee_residual_parts``)."""
    dp, rot6, _ = ee_residual_parts(root_pos, root_quat, palm_pos_w, palm_quat_w, ref_pos_local, ref_quat_local)
    return dp.reshape(6), rot6.reshape(12)


def ref_body_pos_b(body_pos_w: np.ndarray, ref_root_pos: np.ndarray, ref_root_quat: np.ndarray) -> np.ndarray:
    """h16: ``R(q_ref_root)^T (p_body - p_ref_root)`` per body, flattened (B*3)."""
    b = np.asarray(body_pos_w, dtype=np.float64).reshape(-1, 3)
    rel = quat_apply_inv(np.asarray(ref_root_quat, dtype=np.float64).reshape(1, 4), b - np.asarray(ref_root_pos, dtype=np.float64).reshape(1, 3))
    return rel.reshape(-1)


def object_terms(root_pos, root_quat, obj_pos_w, obj_quat_w, has_object: bool) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """h17 (3) / h18 (6) / h19 (1) -- ``object_state_10`` split (all zero incl. the flag when ``has_object`` is False)."""
    s = object_state_10(root_pos, root_quat, obj_pos_w, obj_quat_w, has_object).astype(np.float64)
    return s[0:3], s[3:9], s[9:10]


def vel_cmd_from_clip(root_quat_w: np.ndarray, root_lin_vel_w: np.ndarray, root_ang_vel_w: np.ndarray) -> np.ndarray:
    """``_refresh_hero_refs`` (from_clip): ``[R(yaw_quat(q))^T v]_xy`` and ``w_z`` -> (vx, vy, wz) before the walking mask."""
    q = np.asarray(root_quat_w, dtype=np.float64).reshape(4)
    v_heading = quat_apply_inv(yaw_quat(q), np.asarray(root_lin_vel_w, dtype=np.float64).reshape(3))
    return np.array([v_heading[0], v_heading[1], float(np.asarray(root_ang_vel_w, dtype=np.float64).reshape(3)[2])], dtype=np.float64)


def heading_yaw(q_xyzw: np.ndarray) -> np.ndarray:
    """Yaw of xyzw quaternions ``(..., 4)`` -> ``(...)``: the angle holosoma ``yaw_quat`` keeps (``atan2(2(wz + xy), 1 - 2(y^2 + z^2))``;
    == ``HeroMotionCommand._yaw_of``'s forward-vector atan2)."""
    q = np.asarray(q_xyzw, dtype=np.float64)
    x, y, z, w = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    return np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def wrap_to_pi(a: np.ndarray) -> np.ndarray:
    """Angles wrapped into ``[-pi, pi)`` (holosoma ``wrap_to_pi`` up to the sign of the boundary itself)."""
    return (np.asarray(a, dtype=np.float64) + np.pi) % (2.0 * np.pi) - np.pi


def quat_from_yaw(yaw: float) -> np.ndarray:
    """xyzw quaternion of a rotation about +z."""
    return np.array([0.0, 0.0, np.sin(0.5 * float(yaw)), np.cos(0.5 * float(yaw))], dtype=np.float64)


def ref_root_pose_b(robot_root_pos: np.ndarray, robot_root_quat: np.ndarray, ref_root_pos_w: np.ndarray, ref_root_quat_w: np.ndarray) -> np.ndarray:
    """h20 core: reference root pose(s) relative to the robot root in the robot HEADING frame -> ``(F, 4)``.

    ``[dx, dy] = [R(yaw_quat(q_robot))^T (p_ref - p_robot)]_xy`` (holosoma ``quat_rotate_inverse(yaw_quat(q), .)``),
    ``dyaw = wrap(yaw(q_ref) - yaw(q_robot))`` as ``[sin, cos]``.  ``ref_root_*`` are ``(F, 3)`` / ``(F, 4)`` (or a single
    pose); the robot pose is one ``(3,)`` / ``(4,)``.  Height is deliberately NOT part of the term (h03 carries the height
    command; odometry z is the least reliable channel).  Invariant to a common SE(2) transform of robot + reference."""
    p_r = np.asarray(robot_root_pos, dtype=np.float64).reshape(3)
    q_r = np.asarray(robot_root_quat, dtype=np.float64).reshape(4)
    p_f = np.asarray(ref_root_pos_w, dtype=np.float64).reshape(-1, 3)
    q_f = np.asarray(ref_root_quat_w, dtype=np.float64).reshape(-1, 4)
    if p_f.shape[0] != q_f.shape[0]:
        raise ValueError(f"ref_root_pos_w / ref_root_quat_w frame counts differ: {p_f.shape[0]} vs {q_f.shape[0]}")
    q_yaw = np.broadcast_to(yaw_quat(q_r), q_f.shape)
    d = quat_apply_inv(q_yaw, p_f - p_r[None])
    dyaw = wrap_to_pi(heading_yaw(q_f) - heading_yaw(q_r))
    return np.stack([d[:, 0], d[:, 1], np.sin(dyaw), np.cos(dyaw)], axis=-1)


def h20_frame_indices(ref: Any, t: int, steps: Sequence[int] = H20_FUTURE_STEPS) -> np.ndarray:
    """Clip frames of the h20 future poses: ``clamp(t + f, T-1)`` per ``f`` (holosoma ``clamp(time_steps + f, max=last)``;
    the stock frame ``t`` is the base, like h14-h16 -- NOT the lookahead ``r``).  A padded reference repeats its last real
    pose over the pad, so clamping at ``T-1`` equals Isaac's clamp at the last REAL frame."""
    T = int(ref.T)
    idx = int(t) + np.asarray(list(steps), dtype=np.int64)
    return np.clip(idx, 0, T - 1)


def rot6d_cols(R: np.ndarray) -> np.ndarray:
    """6D rotation representation (Zhou et al. 2019) of rotation matrices ``(..., 3, 3)``: the first two COLUMNS flattened
    COLUMN-major ``[r00 r10 r20 | r01 r11 r21]`` -> ``(..., 6)``; identity -> ``[1 0 0 0 1 0]``.  NOT the HERO rot6d
    of h08 / h18 (``mathutil.rot6d_from_quat``: the same two columns ROW-major, identity ``[1 0 0 1 0 0]``)."""
    R = np.asarray(R, dtype=np.float64)
    return np.concatenate([R[..., :, 0], R[..., :, 1]], axis=-1)


def ref_root_rot_b(robot_root_quat: np.ndarray, ref_root_quat_w: np.ndarray) -> np.ndarray:
    """h21 core: reference root orientation(s) expressed in the robot ROOT frame -> ``(F, 6)``.

    ``R_rel = R(q_robot)^T R(q_ref) = R(q_robot^-1 * q_ref)`` (holosoma ``quat_mul(quat_conjugate(q_robot), q_ref)`` ->
    ``quaternion_to_matrix``), returned as :func:`rot6d_cols` per frame; ``[1 0 0 0 1 0]`` == on track.  Full 3D (roll /
    pitch / yaw) -- unlike h20's heading-only ``dyaw``.  Quaternions xyzw (``ClipReference`` / ``RobotState`` convention; the
    npz / MuJoCo wxyz are converted at those boundaries).  Invariant to a common rotation + translation of robot + reference
    (an odometry-frame quantity: robot roll / pitch from the IMU, yaw from odometry, the clip aligned at episode start)."""
    q_r = np.asarray(robot_root_quat, dtype=np.float64).reshape(4)
    q_f = np.asarray(ref_root_quat_w, dtype=np.float64).reshape(-1, 4)
    rel = quat_mul(np.broadcast_to(quat_conj(q_r), q_f.shape), q_f)
    return rot6d_cols(quat_to_mat(rel))


def ref_root_height_b(robot_root_pos: np.ndarray, ref_root_pos_w: np.ndarray) -> np.ndarray:
    """h22 core: ``z_ref(f) - z_robot`` (m) per reference frame -> ``(F,)``.  Invariant to the env origin / floor height and to
    the planar pose; the complement of h03 (the height COMMAND) -- this is the height ERROR the policy can close."""
    z_r = float(np.asarray(robot_root_pos, dtype=np.float64).reshape(3)[2])
    return np.asarray(ref_root_pos_w, dtype=np.float64).reshape(-1, 3)[:, 2] - z_r


def base_lin_vel_odom(robot_root_quat: np.ndarray, root_lin_vel_w: np.ndarray) -> np.ndarray:
    """h23 core: the robot root linear velocity in the robot HEADING frame ``R(yaw_quat(q_robot))^T v_w`` -> ``(3,)`` (holosoma
    ``quat_rotate_inverse(yaw_quat(q), v)``; only the heading enters, the tilt does not -- unlike the critic's root-frame
    ``h01a_base_lin_vel``).  ``v_w`` = MuJoCo free-joint ``qvel[0:3]`` == Isaac ``robot_root_states[:, 7:10]``; on the robot the
    odometry velocity (a common yaw offset of the odometry frame cancels, so the yaw estimate error does not enter)."""
    q = np.asarray(robot_root_quat, dtype=np.float64).reshape(4)
    return quat_apply_inv(yaw_quat(q), np.asarray(root_lin_vel_w, dtype=np.float64).reshape(3))


@dataclass(frozen=True)
class OdometryNoise:
    """Noise applied to root-state estimates used by delta-anchor observations.

    Each reset draws per-episode biases using ``seed + reset_count``. Planar
    position, yaw, and height can accumulate Gaussian random walks with
    increments scaled by ``sqrt(dt)``. Roll/pitch use body-frame bias, while
    velocity uses independent Gaussian noise each control step. Orientation
    and planar observations share the same yaw error. Zero values give exact
    estimates; noise fields require the corresponding exported input terms.
    """

    bias_xy_m: float = 0.0
    bias_yaw_rad: float = 0.0
    walk_xy_m: float = 0.0
    walk_yaw_rad: float = 0.0
    seed: int = 0
    # ---- orientation, height, and velocity noise ----
    bias_rp_rad: float = 0.0
    bias_z_m: float = 0.0
    walk_z_m: float = 0.0
    vel_std_m_s: float = 0.0

    def __post_init__(self) -> None:
        for k in ODOM_PLANAR_FIELDS + ODOM_ANCHOR2_FIELDS:
            if float(getattr(self, k)) < 0.0:
                raise ValueError(f"OdometryNoise.{k} must be >= 0")

    @property
    def active(self) -> bool:
        return any(float(getattr(self, k)) > 0.0 for k in ODOM_PLANAR_FIELDS + ODOM_ANCHOR2_FIELDS)

    @property
    def planar_active(self) -> bool:
        """xy / yaw noise (acts on h20, and through the yaw on h21)."""
        return any(float(getattr(self, k)) > 0.0 for k in ODOM_PLANAR_FIELDS)

    @property
    def yaw_active(self) -> bool:
        return float(self.bias_yaw_rad) > 0.0 or float(self.walk_yaw_rad) > 0.0

    @property
    def rot_active(self) -> bool:
        """Anything that moves the ORIENTATION estimate h21 is built from (yaw shared with h20 + the roll / pitch bias)."""
        return self.yaw_active or float(self.bias_rp_rad) > 0.0

    @property
    def height_active(self) -> bool:
        return float(self.bias_z_m) > 0.0 or float(self.walk_z_m) > 0.0

    @property
    def vel_active(self) -> bool:
        return float(self.vel_std_m_s) > 0.0

    @classmethod
    def coerce(cls, value: "OdometryNoise | Mapping[str, Any] | None") -> "OdometryNoise":
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        return cls(**{k: value[k] for k in cls.__dataclass_fields__ if k in value})

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__} | {"active": self.active}


class OdometryNoiseState:
    """Reusable root-state noise with the same draw order as :class:`HeroExportPolicy`.

    Construction counts as the first reset. Each reset reseeds with
    ``seed + resets``; each walk advances once per control step. All-zero
    settings return unchanged estimates without drawing random numbers.
    """

    def __init__(self, noise: OdometryNoise | Mapping[str, Any] | None = None):
        self.noise = OdometryNoise.coerce(noise)
        self.resets = 0
        self._rng = np.random.default_rng(int(self.noise.seed))
        self.reset()

    # ---- episode state ------------------------------------------------------------------------------------------------------------
    def reset(self) -> None:
        """New episode: ``seed + resets`` -> bias ``N(0, bias_*)`` on (dx, dy, dyaw), roll / pitch and z when their field is set, the first
        velocity draw; the walk restarts from the bias."""
        self.resets += 1
        self.offset = np.zeros(3, dtype=np.float64)  # [dx, dy, dyaw] of the planar estimate (world xy, yaw about +z)
        self.rp_bias = np.zeros(2, dtype=np.float64)  # body-frame roll / pitch bias of the orientation estimate (IMU grade, no walk)
        self.z_offset = 0.0  # height estimate error (bias + walk)
        self.vel_noise = np.zeros(3, dtype=np.float64)  # this step's heading-frame velocity noise
        n = self.noise
        if not n.active:
            return
        self._rng = np.random.default_rng(int(n.seed) + self.resets)
        self.offset[0:2] = self._rng.normal(0.0, 1.0, size=2) * float(n.bias_xy_m)
        self.offset[2] = self._rng.normal(0.0, 1.0) * float(n.bias_yaw_rad)
        if float(n.bias_rp_rad) > 0.0:
            self.rp_bias = self._rng.normal(0.0, 1.0, size=2) * float(n.bias_rp_rad)
        if float(n.bias_z_m) > 0.0:
            self.z_offset = float(self._rng.normal(0.0, 1.0)) * float(n.bias_z_m)
        self._draw_velocity_noise()

    def _draw_velocity_noise(self) -> None:
        if self.noise.vel_active:
            self.vel_noise = self._rng.normal(0.0, 1.0, size=3) * float(self.noise.vel_std_m_s)

    def walk(self) -> None:
        """One control step of the random walk (increment std ``walk_* * sqrt(CONTROL_DT)``) + the next velocity draw; no-op when inactive.
        Call AFTER the frame's estimates were read (first frame after a reset = bias only), like ``HeroExportPolicy.anchor_terms``."""
        n = self.noise
        if not n.active:
            return
        s = np.sqrt(CONTROL_DT)
        if n.walk_xy_m != 0.0 or n.walk_yaw_rad != 0.0:
            self.offset[0:2] += self._rng.normal(0.0, 1.0, size=2) * float(n.walk_xy_m) * s
            self.offset[2] += self._rng.normal(0.0, 1.0) * float(n.walk_yaw_rad) * s
        if float(n.walk_z_m) > 0.0:
            self.z_offset += float(self._rng.normal(0.0, 1.0)) * float(n.walk_z_m) * s
        self._draw_velocity_noise()

    # ---- estimates ----------------------------------------------------------------------------------------------------------------
    def pose_estimate(self, root_pos: np.ndarray, root_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The planar pose h20 / ``anchor_ref_root_pose_b`` are built from: ``p + [dx, dy, 0]``, ``Rz(dyaw) * q``; unchanged when inactive."""
        p = np.asarray(root_pos, dtype=np.float64).reshape(3)
        q = np.asarray(root_quat, dtype=np.float64).reshape(4)
        if not self.noise.active:
            return p, q
        return p + np.array([self.offset[0], self.offset[1], 0.0]), quat_mul(quat_from_yaw(self.offset[2]), q)

    def orientation_estimate(self, root_quat: np.ndarray) -> np.ndarray:
        """The orientation h21 / ``anchor_ref_root_rot_b`` are built from: ``Rz(dyaw) * q * R_xyz(droll, dpitch, 0)`` (the SAME yaw error as
        the planar estimate + the body-frame roll / pitch bias); ``q`` itself when the rotation noise is off."""
        q = np.asarray(root_quat, dtype=np.float64).reshape(4)
        if not self.noise.rot_active:
            return q
        q_est = q
        if float(self.noise.bias_rp_rad) > 0.0:
            q_est = quat_mul(q_est, quat_from_euler_xyz(self.rp_bias[0], self.rp_bias[1], 0.0))
        if self.noise.yaw_active:
            q_est = quat_mul(quat_from_yaw(self.offset[2]), q_est)
        return q_est

    def height_estimate(self, root_pos: np.ndarray) -> float:
        """The height h22 / ``anchor_ref_root_height_b`` are built from: ``z + (bias + walk)``; exact when the height noise is off."""
        z = float(np.asarray(root_pos, dtype=np.float64).reshape(3)[2])
        return z + (float(self.z_offset) if self.noise.height_active else 0.0)

    def velocity_estimate(self, v_heading: np.ndarray) -> np.ndarray:
        """The heading-frame velocity h23 is built from: ``v`` + this step's Gaussian draw; exact when ``vel_std_m_s`` is 0."""
        v = np.asarray(v_heading, dtype=np.float64).reshape(3)
        return v + self.vel_noise if self.noise.vel_active else v

    def snapshot(self) -> dict[str, Any]:
        """The current error state (per-frame provenance): ``odom_offset`` [dx, dy, dyaw], ``odom_rp_bias``, ``odom_z_offset``, ``odom_vel_noise``."""
        return {"odom_offset": self.offset.copy(), "odom_rp_bias": self.rp_bias.copy(), "odom_z_offset": float(self.z_offset), "odom_vel_noise": self.vel_noise.copy()}


def walking_flags_from_clip(speed: np.ndarray, yaw_rate: np.ndarray, *, half: int, speed_thr: float, yaw_thr: float) -> np.ndarray:
    """``_clip_walking_flag`` for every frame of one clip: (T,) float, 1.0 = walking, 0.0 = stance.

    ``speed`` = |v_xy| of the clip pelvis, ``yaw_rate`` = |w_z| (T,); window ``[max(i-half, 0), min(i+half, T-1)]``,
    stance iff BOTH window means are below their thresholds (prefix sums like the torch implementation)."""
    speed = np.asarray(speed, dtype=np.float64).reshape(-1)
    yaw_rate = np.asarray(yaw_rate, dtype=np.float64).reshape(-1)
    T = speed.shape[0]
    cs = np.concatenate([[0.0], np.cumsum(speed)])
    cy = np.concatenate([[0.0], np.cumsum(yaw_rate)])
    idx = np.arange(T)
    lo = np.maximum(idx - int(half), 0)
    hi = np.minimum(idx + int(half), T - 1)
    count = (hi - lo + 1).astype(np.float64)
    mean_speed = (cs[hi + 1] - cs[lo]) / count
    mean_yaw = (cy[hi + 1] - cy[lo]) / count
    stance = (mean_speed < float(speed_thr)) & (mean_yaw < float(yaw_thr))
    return (~stance).astype(np.float64)


def h_cmd_from_clip(h_ref: float, params: HeroCommandParams, walking: float = 0.0) -> float:
    """Return the clipped reference height for clip-driven commands, or the fixed height."""
    if params.h_offset_from_clip:
        return max(float(h_ref), float(params.h_cmd_min))
    return float(params.h_cmd_default) + 0.0 * (1.0 - float(walking))


def clip_object_decision(ref: Any, params: HeroCommandParams, default_box_size_m: float = DEFAULT_BOX_SIZE_M) -> dict[str, Any]:
    """``HeroMotionCommand._compute_clip_object_effective`` for ONE clip (numpy): ``effective`` = the clip has an object
    track AND size rule (``|max(box_size) - object_box_side| <= object_box_tol``; a clip without ``box_size`` is the
    ``default_box_size_m`` cube when ``object_box_size_missing_ok``, else rejected) AND start-height rule (frame-0 box
    bottom ``object_pos_w[0, 2] - size_z / 2 <= max_start_bottom_z_m``).  Rules that are None are skipped; both None ->
    the raw flag.  Returns the decision with its inputs (``box_edge_m``, ``bottom_z0_m``, ``size_ok``, ``height_ok``)."""
    has = bool(getattr(ref, "has_object", False)) and getattr(ref, "object_pos_w", None) is not None
    out: dict[str, Any] = {
        "has_object": has, "rule_active": params.object_rule_active, "effective": has,
        "box_edge_m": None, "box_size_missing": None, "bottom_z0_m": None, "size_ok": None, "height_ok": None,
    }
    if not has:
        return out
    box = getattr(ref, "box_size", None)
    missing = box is None
    size = np.full(3, float(default_box_size_m)) if missing else np.asarray(box, dtype=np.float64).reshape(3)
    edge = float(np.max(size))
    bottom = float(np.asarray(ref.object_pos_w[0], dtype=np.float64)[2] - 0.5 * size[2])
    out.update(box_edge_m=edge, box_size_missing=missing, bottom_z0_m=bottom)
    eff = True
    if params.object_box_side is not None:
        ok = abs(edge - float(params.object_box_side)) <= float(params.object_box_tol) + 1.0e-6
        if missing and not params.object_box_size_missing_ok:
            ok = False
        out["size_ok"] = bool(ok)
        eff = eff and bool(ok)
    if params.max_start_bottom_z_m is not None:
        ok = bool(np.isfinite(bottom)) and bottom <= float(params.max_start_bottom_z_m)
        out["height_ok"] = bool(ok)
        eff = eff and bool(ok)
    out["effective"] = bool(eff)
    return out


def object_rule_from_preset(preset: str | None) -> dict[str, Any] | None:
    """Compatibility hook; object rules are read from export metadata."""
    return None


# ================================================================================================ layout / history
@dataclass(frozen=True)
class HeroExportLayout:
    """Resolved actor group layout (``ppo_dual.layout.ActorObsLayout`` semantics, numpy side)."""

    terms: tuple[str, ...]
    term_dims: tuple[int, ...]
    history_length: int
    history_layout: str = HISTORY_LAYOUT_FRAME_MAJOR
    group_name: str = ACTOR_GROUP

    def __post_init__(self) -> None:
        if len(self.terms) != len(self.term_dims):
            raise ValueError("terms / term_dims length mismatch")
        if list(self.terms) != sorted(self.terms):
            raise ValueError(f"terms must be sorted (holosoma / HERO concatenation order), got {list(self.terms)}")
        if len(set(self.terms)) != len(self.terms):
            raise ValueError(f"duplicate terms {list(self.terms)}")
        if self.history_length < 1:
            raise ValueError("history_length must be >= 1")
        if self.history_layout not in HISTORY_LAYOUTS:
            raise ValueError(f"history_layout must be one of {HISTORY_LAYOUTS}, got {self.history_layout!r}")
        unknown = [t for t in self.terms if t not in HERO_TERM_DIMS]
        if unknown:
            raise ValueError(f"actor terms this runner cannot build: {unknown} (known: {sorted(HERO_TERM_DIMS)})")
        bad = [(t, d, HERO_TERM_DIMS[t]) for t, d in zip(self.terms, self.term_dims) if int(d) != HERO_TERM_DIMS[t]]
        if bad:
            raise ValueError(f"term dims disagree with the HERO tables (term, sidecar, table): {bad}")
        missing = [t for t in H1_REQUIRED_TERMS if t not in self.terms]
        if missing:
            raise ValueError(f"actor group lacks required HERO terms {missing}")
        # the residual EE feedback h07 / h08 is all-or-nothing: the paper layouts carry both, the without_delta_ee ablation none
        partial_ee = [t for t in DELTA_EE_TERMS if t in self.terms]
        if partial_ee and len(partial_ee) != len(DELTA_EE_TERMS):
            raise ValueError(f"actor group carries a partial residual EE feedback {partial_ee}; expected both of {list(DELTA_EE_TERMS)} or none")


        partial = [t for t in H2_EXTRA_TERMS if t in self.terms]
        if partial and len(partial) != len(H2_EXTRA_TERMS):
            raise ValueError(f"actor group carries a partial lower-body reference {partial}; expected all of {list(H2_EXTRA_TERMS)} or none")

    @property
    def frame_dim(self) -> int:
        return int(sum(self.term_dims))

    @property
    def total_dim(self) -> int:
        return self.frame_dim * self.history_length

    @property
    def offsets(self) -> dict[str, int]:
        out, acc = {}, 0
        for t, d in zip(self.terms, self.term_dims):
            out[t] = acc
            acc += int(d)
        return out

    @property
    def arm(self) -> str:
        """HERO runtime helper."""
        return "h2" if all(t in self.terms for t in H2_EXTRA_TERMS) else "h1"

    @property
    def has_lower_body_reference(self) -> bool:
        """True when the actor carries any of the lower-body reference terms h14-h16."""
        return self.arm == "h2"

    @property
    def object_mode(self) -> str | None:
        """``pose_flag`` (h17-h19), ``flag`` (h19 only) or None -- ``config_values.observation.make_actor_group`` modes."""
        has = [t for t in OBJECT_TERMS if t in self.terms]
        if not has:
            return None
        if has == list(OBJECT_TERMS):
            return "pose_flag"
        if has == ["h19_has_object_flag"]:
            return "flag"
        raise ValueError(f"unsupported object term subset {has}")

    @property
    def has_object_terms(self) -> bool:
        return self.object_mode is not None

    @property
    def has_delta_ee_terms(self) -> bool:
        """True when the actor carries the residual EE feedback h07 / h08 (False for ``without_delta_ee`` exports)."""
        return all(t in self.terms for t in DELTA_EE_TERMS)

    @property
    def has_anchor_term(self) -> bool:
        """True when the actor carries ``h20_ref_root_pose_b``."""
        return ANCHOR_TERM in self.terms

    @property
    def anchor2_present(self) -> tuple[str, ...]:
        """Orientation, height, and velocity feedback terms present in the actor layout."""
        return tuple(t for t in ANCHOR2V_TERMS if t in self.terms)

    @property
    def has_anchor2_terms(self) -> bool:
        return bool(self.anchor2_present)

    @property
    def has_hero_plus_terms(self) -> bool:
        """h20 and / or any anchor2 term: the terms built from the reference root trajectory / odometry."""
        return self.has_anchor_term or self.has_anchor2_terms

    def frame(self, terms: Mapping[str, np.ndarray], scales: Mapping[str, float] = HERO_TERM_SCALES) -> np.ndarray:
        """One frame: sorted terms x scale (float64, ``frame_dim``)."""
        parts = []
        for name, d in zip(self.terms, self.term_dims):
            v = np.asarray(terms[name], dtype=np.float64).reshape(-1)
            if v.shape[0] != int(d):
                raise ValueError(f"term {name}: expected {d} values, got {v.shape[0]}")
            parts.append(v * float(scales.get(name, 1.0)))
        return np.concatenate(parts)

    @classmethod
    def from_metadata(cls, meta: Mapping[str, Any], group_name: str = ACTOR_GROUP) -> "HeroExportLayout":
        """Read ``actor_obs_layout`` (+ top-level ``history_layout``) of the sidecar / ONNX metadata."""
        lay = meta.get("actor_obs_layout")
        if not isinstance(lay, Mapping) or not lay.get("groups"):
            raise ValueError("metadata lacks actor_obs_layout.groups")
        groups = list(lay["groups"])
        group = next((g for g in groups if g.get("name") == group_name), groups[0] if len(groups) == 1 else None)
        if group is None:
            raise ValueError(f"actor_obs_layout has no group {group_name!r}: {[g.get('name') for g in groups]}")
        terms = tuple(str(t) for t in group["terms"])
        dims_raw = group.get("term_dims")
        if dims_raw is None:
            dims_raw = lay.get("term_dims")
        if dims_raw is None and meta.get("actor_obs_term_dims") and list(meta.get("actor_obs_terms") or terms) == list(terms):
            dims_raw = meta["actor_obs_term_dims"]  # Flattened sidecar copy, in actor_obs_terms order.
        if isinstance(dims_raw, Mapping):
            dims = tuple(int(dims_raw[t]) for t in terms)
        elif dims_raw is not None:
            dims = tuple(int(d) for d in dims_raw)
        else:
            unknown = [t for t in terms if t not in HERO_TERM_DIMS]
            if unknown:
                raise ValueError(f"no term_dims in the metadata and unknown terms {unknown}")
            dims = tuple(HERO_TERM_DIMS[t] for t in terms)
        hist = int(group.get("history_length", lay.get("history_length", 5)))
        layout = str(lay.get("history_layout") or meta.get("history_layout") or HISTORY_LAYOUT_FRAME_MAJOR)
        out = cls(terms=terms, term_dims=dims, history_length=hist, history_layout=layout, group_name=str(group.get("name", group_name)))
        dim = group.get("dim")
        if dim is not None and int(dim) != out.total_dim:
            raise ValueError(f"actor group dim {dim} != history {hist} x frame {out.frame_dim} = {out.total_dim}")
        return out

    def describe(self) -> dict[str, Any]:
        return {
            "group": self.group_name,
            "terms": list(self.terms),
            "term_dims": list(self.term_dims),
            "history_length": self.history_length,
            "history_layout": self.history_layout,
            "frame_dim": self.frame_dim,
            "total_dim": self.total_dim,
            "arm": self.arm,
            "lower_body_reference": self.has_lower_body_reference,
            "object_mode": self.object_mode,
            "anchor_term": self.has_anchor_term,
            "anchor2_terms": list(self.anchor2_present),
        }


class HeroFrameHistory:
    """History buffer of ``H`` frames (oldest first, zero frames after reset -- holosoma ``zeros_after_reset``) flattened
    frame-major (HERO export) or term-major (holosoma native) according to the layout."""

    def __init__(self, layout: HeroExportLayout):
        self.layout = layout
        self._frames: deque[np.ndarray] = deque(maxlen=int(layout.history_length))
        self.count = 0

    def reset(self) -> None:
        self._frames.clear()
        self.count = 0

    def push(self, frame: np.ndarray) -> None:
        f = np.array(frame, dtype=np.float64, copy=True).reshape(-1)
        if f.shape[0] != self.layout.frame_dim:
            raise ValueError(f"frame has {f.shape[0]} values, expected {self.layout.frame_dim}")
        self._frames.append(f)
        self.count += 1

    def frames(self) -> np.ndarray:
        """``(H, D)`` oldest first with zero rows in front while filling."""
        H, D = self.layout.history_length, self.layout.frame_dim
        out = np.zeros((H, D), dtype=np.float64)
        hist = list(self._frames)
        if hist:
            out[H - len(hist):] = np.stack(hist, axis=0)
        return out

    def flat(self) -> np.ndarray:
        fr = self.frames()
        if self.layout.history_layout == HISTORY_LAYOUT_FRAME_MAJOR:
            return fr.reshape(-1)
        # term-major: [term_0 f0..fH-1 | term_1 ...]
        parts = []
        for name, d in zip(self.layout.terms, self.layout.term_dims):
            o = self.layout.offsets[name]
            parts.append(fr[:, o : o + int(d)].reshape(-1))
        return np.concatenate(parts)

    def flat32(self) -> np.ndarray:
        return self.flat().astype(np.float32)


# ================================================================================================ files / metadata
def find_hero_sidecars(onnx_dir: str | os.PathLike) -> list[Path]:
    return sorted(Path(onnx_dir).glob(f"*{SIDECAR_SUFFIX}"))


def _iter_key(p: Path, suffix: str):
    m = re.search(r"(\d+)" + re.escape(suffix) + r"$", p.name)
    return (int(m.group(1)) if m else -1, p.name)


def pick_latest_hero_sidecar(onnx_dir: str | os.PathLike) -> Path:
    """Highest ``model_XXXXX_hero.json`` in ``onnx_dir`` (numeric iteration sort, else lexicographic)."""
    cands = find_hero_sidecars(onnx_dir)
    if not cands:
        raise FileNotFoundError(f"no *{SIDECAR_SUFFIX} sidecar in {onnx_dir}")
    return max(cands, key=lambda p: _iter_key(p, SIDECAR_SUFFIX))


def pick_latest_dual_onnx(onnx_dir: str | os.PathLike) -> Path:
    """Highest ``model_XXXXX.onnx`` by iteration; separate encoder/decoder graphs are skipped."""
    cands = [p for p in sorted(Path(onnx_dir).glob("*.onnx")) if not p.name.endswith(("_encoder.onnx", "_decoder.onnx"))]
    if not cands:
        raise FileNotFoundError(f"no model_*.onnx in {onnx_dir}")
    return max(cands, key=lambda p: _iter_key(p, ".onnx"))


def read_onnx_metadata(onnx_path: str | os.PathLike, session: Any | None = None) -> dict[str, Any]:
    """``metadata_props`` of the graph, JSON-decoded per key (holosoma ``attach_onnx_metadata`` stores JSON strings)."""
    if session is None:
        import onnxruntime as ort

        so = ort.SessionOptions()
        so.log_severity_level = 3
        session = ort.InferenceSession(str(onnx_path), so, providers=["CPUExecutionProvider"])
    raw = dict(session.get_modelmeta().custom_metadata_map or {})
    out: dict[str, Any] = {}
    for k, v in raw.items():
        try:
            out[k] = json.loads(v)
        except (TypeError, ValueError):
            out[k] = v
    return out


def resolve_hero_export(
    onnx_dir: str | os.PathLike | None = None, *, sidecar: str | os.PathLike | None = None, onnx: str | os.PathLike | None = None
) -> tuple[Path, Path | None]:
    """``(onnx_path, sidecar_path | None)`` from any of ``--onnx-dir`` / ``--sidecar`` / an explicit ``.onnx``.

    sidecar given -> its ``onnx_file`` / ``files.onnx`` / ``hero_files.onnx`` entry, else ``<stem>.onnx`` beside it;
    onnx given -> ``<stem>_hero.json`` beside it when present; onnx_dir -> the highest ``*_hero.json`` (its ONNX), else
    the highest ``model_*.onnx`` (metadata-only mode)."""
    if sidecar is not None:
        sc = Path(sidecar).expanduser()
        if not sc.is_file():
            raise FileNotFoundError(sc)
        meta = json.loads(sc.read_text())
        name = meta.get("onnx_file") or (meta.get("files") or {}).get("onnx") or (meta.get("hero_files") or {}).get("onnx")
        if onnx is not None:
            path = Path(onnx).expanduser()
        elif name:
            path = sc.parent / str(name)
        else:
            m = _SIDECAR_RE.match(sc.name)
            path = sc.parent / ((m.group("stem") if m else sc.stem) + ".onnx")
        if not path.is_file():
            raise FileNotFoundError(f"ONNX for sidecar {sc}: {path}")
        return path.resolve(), sc.resolve()
    if onnx is not None:
        path = Path(onnx).expanduser()
        if path.is_dir():
            return resolve_hero_export(path)
        if not path.is_file():
            raise FileNotFoundError(path)
        sc = path.with_name(path.stem + SIDECAR_SUFFIX)
        return path.resolve(), (sc.resolve() if sc.is_file() else None)
    if onnx_dir is None:
        raise ValueError("give onnx_dir, sidecar or onnx")
    d = Path(onnx_dir).expanduser()
    if d.is_file():
        return resolve_hero_export(onnx=d)
    if find_hero_sidecars(d):
        return resolve_hero_export(sidecar=pick_latest_hero_sidecar(d))
    path = pick_latest_dual_onnx(d)
    return path.resolve(), None


# ================================================================================================ contract
@dataclass
class HeroExportContract:
    """Everything the controller needs, merged from the sidecar (wins) and the ONNX metadata (fills gaps)."""

    layout: HeroExportLayout
    dof_names: tuple[str, ...]
    kp: np.ndarray
    kd: np.ndarray
    action_scale: np.ndarray  # (29,) per joint
    default_dof_pos: np.ndarray
    effort_limit: np.ndarray
    effort_limit_source: str
    action_contract: str
    residual_dof_idx: np.ndarray
    ref_attr: str
    action_clip: float | None
    obs_clip: float | None
    onnx_inputs: tuple[str, ...]
    onnx_output: str
    action_split: tuple[int, int] | None
    command: HeroCommandParams
    preset: str | None = None
    has_object: bool | None = None
    iteration: int | None = None
    robot_urdf_path: str | None = None
    object_urdf_path: str | None = None
    policy_dt: float | None = None
    wandb_run_path: str | None = None
    schema: str | None = None
    source: str = "onnx_metadata"
    fields_used: dict[str, str] = field(default_factory=dict)
    warnings: list[str] = field(default_factory=list)

    @property
    def preferred_urdf_alias(self) -> str | None:
        """``"paddle"`` when the export was trained on the paddle-hand URDF (its basename), else None (Dex3 default)."""
        base = os.path.basename(str(self.robot_urdf_path or ""))
        return "paddle" if base == PADDLE_URDF_FILE_NAME or "paddle" in base.lower() else None

    def describe(self) -> dict[str, Any]:
        return {
            "schema": self.schema,
            "source": self.source,
            "preset": self.preset,
            "iteration": self.iteration,
            "has_object": self.has_object,
            "layout": self.layout.describe(),
            "obs_clip": self.obs_clip,
            "term_scales": dict(HERO_TERM_SCALES),
            "action_contract": self.action_contract,
            "residual_dof_indices": self.residual_dof_idx.tolist(),
            "ref_attr": self.ref_attr,
            "action_clip": self.action_clip,
            "action_scale": self.action_scale.tolist(),
            "action_split": list(self.action_split) if self.action_split else None,
            "kp": self.kp.tolist(),
            "kd": self.kd.tolist(),
            "effort_limit": self.effort_limit.tolist(),
            "effort_limit_source": self.effort_limit_source,
            "default_dof_pos": self.default_dof_pos.tolist(),
            "onnx_inputs": list(self.onnx_inputs),
            "onnx_output": self.onnx_output,
            "command": self.command.as_dict(),
            "robot_urdf_path": self.robot_urdf_path,
            "object_urdf_path": self.object_urdf_path,
            "policy_dt": self.policy_dt,
            "preferred_urdf_alias": self.preferred_urdf_alias,
            "wandb_run_path": self.wandb_run_path,
            "fields_used": dict(self.fields_used),
            "warnings": list(self.warnings),
        }


def _pick(name: str, sidecar: Mapping[str, Any], meta: Mapping[str, Any], used: dict[str, str], default: Any = None) -> Any:
    if name in sidecar and sidecar[name] is not None:
        used[name] = "sidecar"
        return sidecar[name]
    if name in meta and meta[name] is not None:
        used[name] = "onnx_metadata"
        return meta[name]
    used[name] = "default"
    return default


def contract_from_metadata(
    sidecar: Mapping[str, Any] | None,
    meta: Mapping[str, Any] | None,
    *,
    command_overrides: Mapping[str, Any] | None = None,
    obs_clip: float | None | str = "contract",
    object_rule_lookup: bool = True,
) -> HeroExportContract:
    """Merge the sidecar and ONNX metadata into :class:`HeroExportContract`.

    Sidecar values take precedence. ``object_rule_lookup`` enables a
    compatibility hook when object-rule fields are absent; without rules,
    the raw clip ``has_object`` flag is used.
    """
    sc: Mapping[str, Any] = dict(sidecar or {})
    md: Mapping[str, Any] = dict(meta or {})
    if not sc and not md:
        raise ValueError("neither a sidecar nor ONNX metadata is available")
    used: dict[str, str] = {}
    warns: list[str] = []
    source = "sidecar+onnx_metadata" if (sc and md) else ("sidecar" if sc else "onnx_metadata")

    schema = sc.get("schema") or sc.get("schema_hero")
    if sc and schema is not None and str(schema) not in SIDECAR_SCHEMAS_KNOWN:
        warns.append(f"sidecar schema {schema!r} != expected {SIDECAR_SCHEMA!r}; reading the fields anyway")

    # ---- layout ----------------------------------------------------------------------------------------------------
    lay_src = sc if isinstance(sc.get("actor_obs_layout"), Mapping) else md
    used["actor_obs_layout"] = "sidecar" if lay_src is sc else "onnx_metadata"
    lay_meta = dict(lay_src)
    if "history_layout" not in (lay_meta.get("actor_obs_layout") or {}) and "history_layout" not in lay_meta:
        hl = _pick("history_layout", sc, md, used, HISTORY_LAYOUT_FRAME_MAJOR)
        lay_meta["history_layout"] = hl
    layout = HeroExportLayout.from_metadata(lay_meta)
    # Flattened sidecar fields must agree with the resolved actor layout.
    for key, want in (
        ("hero_arm", layout.arm),
        ("actor_obs_terms", list(layout.terms)),
        ("actor_obs_term_dims", list(layout.term_dims)),
        ("actor_history_length", layout.history_length),
        ("actor_frame_dim", layout.frame_dim),
        ("actor_obs_dim", layout.total_dim),
    ):
        got = sc.get(key)
        if got is None:
            continue
        got_n = [int(x) for x in got] if isinstance(got, (list, tuple)) and key != "actor_obs_terms" else (list(got) if isinstance(got, (list, tuple)) else (int(got) if isinstance(want, int) else str(got)))
        if got_n != want:
            warns.append(f"sidecar {key}={got!r} disagrees with the actor_obs_layout ({want!r}); the layout decides")

    # ---- tables ----------------------------------------------------------------------------------------------------
    dof_names = tuple(str(n) for n in _pick("dof_names", sc, md, used, list(HC.DOF_NAMES)))
    if dof_names != tuple(HC.DOF_NAMES):
        raise ValueError("export dof_names differ from hero_isaacsim.constants.DOF_NAMES (the MuJoCo plant joint order)")
    n = len(dof_names)
    kp = np.asarray(_pick("kp", sc, md, used, PP.KP), dtype=np.float64).reshape(n)
    kd = np.asarray(_pick("kd", sc, md, used, PP.KD), dtype=np.float64).reshape(n)
    default_dof_pos = np.asarray(_pick("default_dof_pos", sc, md, used, PP.DEFAULT_DOF_POS), dtype=np.float64).reshape(n)
    detail = _pick("action_contract_detail", sc, md, used, None) or {}
    scale_raw = detail.get("action_scales") if isinstance(detail, Mapping) and detail.get("action_scales") is not None else None
    if scale_raw is not None:
        used["action_scale"] = used.get("action_contract_detail", "default") + ":action_scales"
    else:
        scale_raw = _pick("action_scale", sc, md, used, PP.ACTION_SCALE)
    scale_arr = np.asarray(scale_raw, dtype=np.float64).reshape(-1)
    action_scale = np.full(n, float(scale_arr[0])) if scale_arr.shape[0] == 1 else scale_arr.reshape(n)
    scale_base = float(detail.get("action_scale", HERO_ACTION_SCALE_BASE)) if isinstance(detail, Mapping) else HERO_ACTION_SCALE_BASE
    eff_raw = _pick("effort_limit", sc, md, used, None)
    if eff_raw is not None:
        effort_limit = np.asarray(eff_raw, dtype=np.float64).reshape(n)
        eff_src = used["effort_limit"]
    elif (not isinstance(detail, Mapping) or detail.get("action_scales_by_effort_limit_over_p_gain", True)) and scale_arr.shape[0] == n and scale_base > 0:
        effort_limit = action_scale * kp / scale_base  # action_scale = base * effort / kp
        eff_src = f"derived:action_scale*kp/{scale_base}"
    else:
        effort_limit = np.asarray(PP.EFFORT_LIMIT, dtype=np.float64).reshape(n)
        eff_src = "plant_params.EFFORT_LIMIT"
    if not np.all(np.isfinite(effort_limit)) or np.any(effort_limit <= 0):
        warns.append("derived effort limits are degenerate; using plant_params.EFFORT_LIMIT")
        effort_limit = np.asarray(PP.EFFORT_LIMIT, dtype=np.float64).reshape(n)
        eff_src = "plant_params.EFFORT_LIMIT"

    # ---- action contract -------------------------------------------------------------------------------------------
    contract_raw = str(_pick("action_contract", sc, md, used, CONTRACT_RESIDUAL_UPPER))
    contract = contract_raw.split(":", 1)[0].strip()
    if isinstance(detail, Mapping) and detail.get("action_contract"):
        contract = str(detail["action_contract"])
    if contract not in CONTRACTS:
        raise ValueError(f"unsupported action contract {contract!r} (need one of {CONTRACTS})")
    if isinstance(detail, Mapping) and detail.get("residual_upper_body_action") is False:
        residual_idx = np.zeros(0, dtype=np.int64)
    elif isinstance(detail, Mapping) and detail.get("residual_dof_indices") is not None:
        residual_idx = np.asarray(detail["residual_dof_indices"], dtype=np.int64).reshape(-1)
    elif isinstance(detail, Mapping) and detail.get("residual_dof_names"):
        residual_idx = np.asarray([dof_names.index(str(x)) for x in detail["residual_dof_names"]], dtype=np.int64)
    else:
        residual_idx = np.arange(n, dtype=np.int64) if contract == CONTRACT_RESIDUAL_ALL29 else _ARM.copy()
    ref_attr = str(detail.get("ref_attr", "ref_upper_dof_pos")) if isinstance(detail, Mapping) else "ref_upper_dof_pos"
    if ref_attr not in ("ref_upper_dof_pos", "joint_pos"):
        raise ValueError(f"unsupported residual ref_attr {ref_attr!r}")
    if ref_attr == "ref_upper_dof_pos" and not set(residual_idx.tolist()) <= set(_UPPER17.tolist()):
        raise ValueError(f"residual joints {residual_idx.tolist()} are not covered by ref_upper_dof_pos (waist + arms)")
    if isinstance(detail, Mapping) and "action_clip_value" in detail:
        clip_v = detail["action_clip_value"]
        action_clip = float(clip_v) if clip_v is not None else None
    else:
        action_clip = float(_pick("action_clip_value", sc, md, used, HERO_ACTION_CLIP))

    # ---- misc ----------------------------------------------------------------------------------------------------
    if obs_clip == "contract":
        oc = _pick("clip_observations", sc, md, used, HERO_OBS_CLIP)
        obs_clip_v = float(oc) if oc is not None else None
    else:
        obs_clip_v = None if obs_clip is None else float(obs_clip)
    inputs = tuple(str(x) for x in _pick("onnx_inputs", sc, md, used, list(ONNX_INPUT_NAMES)))
    output = str(_pick("onnx_output", sc, md, used, ONNX_OUTPUT_NAME))
    split_raw = _pick("action_split", sc, md, used, None)
    split = (int(split_raw[0]), int(split_raw[1])) if split_raw is not None and len(split_raw) == 2 else None
    cmd_raw = _pick("hero_command", sc, md, used, None) or {}
    cmd_kw = {k: cmd_raw[k] for k in HeroCommandParams.__dataclass_fields__ if k in cmd_raw and k != "arm"}
    preset_raw = _pick("preset", sc, md, used, None)
    # Object rules come from hero_command, then the compatibility fallback.
    # Explicit command_overrides take precedence.
    ov = {k: v for k, v in (command_overrides or {}).items() if v is not None}
    if any(k in cmd_kw for k in OBJECT_RULE_FIELDS):
        used["object_rule"] = "hero_command"
    elif not any(k in ov for k in OBJECT_RULE_FIELDS):
        from_preset = object_rule_from_preset(preset_raw if preset_raw is not None else None) if object_rule_lookup else None
        if from_preset:
            cmd_kw.update(from_preset)
            used["object_rule"] = f"preset:{preset_raw}"
        else:
            used["object_rule"] = "default:off"
            if preset_raw is not None and object_rule_lookup:
                warns.append(f"sidecar hero_command lacks the object-rule fields and preset {preset_raw!r} could not be resolved: object rule OFF (raw has_object)")
    if any(k in ov for k in OBJECT_RULE_FIELDS):
        used["object_rule"] = (used.get("object_rule", "default") + "+overrides")
    # Horizon precedence: explicit override, h20_future_steps metadata,
    # hero_command.ref_root_pose_future_steps, then the default with a warning.
    # Reject dimensions that disagree with the training horizon.
    h20_raw = _pick("h20_future_steps", sc, md, used, None)
    legacy_steps = cmd_kw.get("ref_root_pose_future_steps")
    if h20_raw is not None:
        h20_steps = tuple(int(v) for v in np.asarray(h20_raw, dtype=np.int64).reshape(-1).tolist())
        if layout.has_anchor_term:
            h20_dim = int(layout.term_dims[layout.terms.index(ANCHOR_TERM)])
            if len(h20_steps) * len(H20_PER_FRAME) != h20_dim:
                raise ValueError(
                    f"sidecar h20_future_steps={list(h20_steps)} ({len(h20_steps)} frames x {len(H20_PER_FRAME)}) disagree with the "
                    f"exported {ANCHOR_TERM} dim {h20_dim} ({used['h20_future_steps']})"
                )
            if legacy_steps is not None and tuple(int(v) for v in legacy_steps) != h20_steps:
                warns.append(f"sidecar hero_command.ref_root_pose_future_steps={list(legacy_steps)} disagrees with h20_future_steps={list(h20_steps)}; the term's own horizon decides")
            cmd_kw["ref_root_pose_future_steps"] = h20_steps
        else:
            warns.append(f"sidecar h20_future_steps={list(h20_steps)} but the actor group has no {ANCHOR_TERM}; ignored")
    elif layout.has_anchor_term and legacy_steps is not None:
        used["h20_future_steps"] = "hero_command"
    elif layout.has_anchor_term and "ref_root_pose_future_steps" not in ov:
        warns.append(
            f"export carries {ANCHOR_TERM} but neither h20_future_steps nor hero_command.ref_root_pose_future_steps is in the sidecar / "
            f"metadata (export without horizon metadata): assuming the default {list(H20_FUTURE_STEPS)}"
        )
    if layout.has_anchor_term and "ref_root_pose_future_steps" in ov:
        used["h20_future_steps"] = used.get("h20_future_steps", "default") + "+override"
    # h21 / h22 share h20's horizon.  PPODual writes ``h21_future_steps`` / ``h22_future_steps`` next to
    # ``h20_future_steps`` (export.HERO_FUTURE_STEPS_KEYS); a value that disagrees with the resolved h20 horizon is an error (the
    # graph was trained on ONE set of frames), a value on an export without the term is ignored with a warning, and an export
    # whose layout has h21 / h22 but no h20 takes the horizon from these keys.  h23 has no horizon.
    for term in (ANCHOR2_ROT_TERM, ANCHOR2_HEIGHT_TERM):
        key = HERO_PLUS_FUTURE_STEPS_KEYS[term]
        raw = _pick(key, sc, md, used, None)
        if raw is None:
            continue
        steps = tuple(int(v) for v in np.asarray(raw, dtype=np.int64).reshape(-1).tolist())
        if term not in layout.terms:
            warns.append(f"sidecar {key}={list(steps)} but the actor group has no {term}; ignored")
            continue
        if "ref_root_pose_future_steps" in ov:  # the CLI override wins over every sidecar horizon (recorded like h20's)
            if tuple(int(v) for v in ov["ref_root_pose_future_steps"]) != steps:
                warns.append(f"CLI override ref_root_pose_future_steps={list(ov['ref_root_pose_future_steps'])} beats the sidecar {key}={list(steps)}")
            continue
        cur = cmd_kw.get("ref_root_pose_future_steps")
        if cur is None:
            cmd_kw["ref_root_pose_future_steps"] = steps  # no h20 horizon resolved: this term names the shared horizon
            used["h20_future_steps"] = used.get("h20_future_steps", "default") + f"<-{key}"
        elif tuple(int(v) for v in cur) != steps:
            raise ValueError(
                f"sidecar {key}={list(steps)} disagrees with the h20 horizon {list(cur)} ({term} shares h20_ref_root_pose_b's future frames by contract)"
            )
    cmd_kw.update(ov)
    if layout.has_hero_plus_terms and not layout.has_lower_body_reference and not any(k in cmd_kw for k in ("stand_flag_mode", "h_offset_from_clip", "zero_waist_when_walking")):

        # Missing command metadata falls back to stance, zero velocity, and
        # 0.75 m height. Report that change to clip-driven commands.
        warns.append("export carries HERO terms without the lower-body reference h14-h16 and the sidecar has no hero_command block: using fixed command defaults (stance / zero velocity / h_cmd 0.75); provide matching command metadata")
    command = HeroCommandParams.for_arm(layout.arm, **cmd_kw)
    steps_final = tuple(command.ref_root_pose_future_steps)
    for term, per_frame in ((ANCHOR2_ROT_TERM, H21_FRAME_DIM), (ANCHOR2_HEIGHT_TERM, H22_FRAME_DIM)):
        if term in layout.terms:
            dim = int(layout.term_dims[layout.terms.index(term)])
            if len(steps_final) * per_frame != dim:
                raise ValueError(
                    f"exported {term} dim {dim} != {len(steps_final)} future frames x {per_frame} (horizon {list(steps_final)}, "
                    f"source {used.get('h20_future_steps', 'default')})"
                )
    # the per-term provenance block PPODual writes (``anchor_terms``: dim / per_frame_dim / future_steps / noise per HERO term);
    # disagreement with the layout or the resolved horizon = a stale / edited sidecar -> warnings (the layout + horizon decide)
    block = _pick(ANCHOR_TERMS_KEY, sc, md, used, None)
    if isinstance(block, Mapping):
        for term, entry in block.items():
            if term not in layout.terms:
                warns.append(f"sidecar {ANCHOR_TERMS_KEY} lists {term} but the actor group lacks it; ignored")
                continue
            if not isinstance(entry, Mapping):
                continue
            dim = int(layout.term_dims[layout.terms.index(str(term))])
            if entry.get("dim") is not None and int(entry["dim"]) != dim:
                warns.append(f"sidecar {ANCHOR_TERMS_KEY}[{term}].dim={entry['dim']} != the layout dim {dim}; the layout decides")
            fs = entry.get("future_steps")
            if fs is not None and term in HERO_PLUS_PER_FRAME_DIM and tuple(int(v) for v in fs) != steps_final:
                warns.append(f"sidecar {ANCHOR_TERMS_KEY}[{term}].future_steps={list(fs)} != the resolved horizon {list(steps_final)}; the horizon decides")
        missing = [t for t in HERO_PLUS_TERMS if t in layout.terms and t not in block]
        if missing:
            warns.append(f"sidecar {ANCHOR_TERMS_KEY} lacks the exported HERO terms {missing}")
    has_object = _pick("has_object", sc, md, used, None)
    has_object = bool(has_object) if has_object is not None else None
    if has_object is not None and has_object != layout.has_object_terms:
        warns.append(f"sidecar has_object={has_object} but the actor group {'has' if layout.has_object_terms else 'lacks'} h17-h19; the terms decide")
    it = _pick("iteration", sc, md, used, None)
    try:
        iteration = int(it) if it is not None else None
    except (TypeError, ValueError):
        iteration = None
    urdf_raw = _pick("robot_urdf_path", sc, md, used, None)
    obj_urdf_raw = _pick("object_urdf_path", sc, md, used, None)
    dt_raw = _pick("policy_dt", sc, md, used, None)
    try:
        policy_dt = float(dt_raw) if dt_raw is not None else None
    except (TypeError, ValueError):
        policy_dt = None
    if policy_dt is not None and abs(policy_dt - CONTROL_DT) > 1e-6:
        warns.append(f"sidecar policy_dt={policy_dt} != the {CONTROL_DT} s control step this runner uses")
    return HeroExportContract(
        layout=layout,
        dof_names=dof_names,
        kp=kp,
        kd=kd,
        action_scale=action_scale,
        default_dof_pos=default_dof_pos,
        effort_limit=effort_limit,
        effort_limit_source=eff_src,
        action_contract=contract,
        residual_dof_idx=residual_idx,
        ref_attr=ref_attr,
        action_clip=action_clip,
        obs_clip=obs_clip_v,
        onnx_inputs=inputs,
        onnx_output=output,
        action_split=split,
        command=command,
        preset=(str(preset_raw) if preset_raw is not None else None),
        has_object=has_object,
        iteration=iteration,
        robot_urdf_path=(str(urdf_raw) if urdf_raw else None),
        object_urdf_path=(str(obj_urdf_raw) if obj_urdf_raw else None),
        policy_dt=policy_dt,
        wandb_run_path=_pick("wandb_run_path", sc, md, used, None),
        schema=(str(schema) if schema is not None else None),
        source=source,
        fields_used=used,
        warnings=warns,
    )


# ================================================================================================ clip-side cache
class _ClipStats:
    """Per-clip root heights, walking flags, and palm references with a neutral waist."""

    def __init__(self, ref: Any, params: HeroCommandParams):
        T = int(ref.T)
        self.T = T
        self.ref = ref  # strong reference: the cache is validated by identity (``stats``), so the id cannot be recycled meanwhile
        self.key = (id(ref), T)
        keys: dict[str, np.ndarray] = {}
        path = getattr(ref, "path", None)
        if path is not None and Path(path).is_file():
            with np.load(Path(path), allow_pickle=False) as d:
                for k in ("h_ref", "ee_pos_pelvis_zero_waist", "ee_quat_pelvis_zero_waist"):
                    if k in d.files:
                        keys[k] = np.asarray(d[k], dtype=np.float64)
        T0 = int(getattr(ref, "T_original", T))

        def fit(a: np.ndarray | None) -> np.ndarray | None:
            if a is None or a.shape[0] not in (T, T0):
                return None
            if a.shape[0] < T:  # padded reference: repeat the last frame like ClipReference.padded
                a = np.concatenate([a, np.repeat(a[-1:], T - a.shape[0], axis=0)], axis=0)
            return np.ascontiguousarray(a)

        h = fit(keys.get("h_ref"))
        self.h_ref = h if h is not None else np.ascontiguousarray(ref.root_pos_w[:, 2].astype(np.float64))
        self.h_ref_source = "npz:h_ref" if h is not None else "pelvis_z"
        zw_p = fit(keys.get("ee_pos_pelvis_zero_waist"))
        zw_q = fit(keys.get("ee_quat_pelvis_zero_waist"))
        if zw_p is not None and zw_q is not None:
            from sim2sim.mathutil import quat_normalize, wxyz_to_xyzw

            self.ee_pos_zero_waist: np.ndarray | None = zw_p
            self.ee_quat_zero_waist: np.ndarray | None = quat_normalize(wxyz_to_xyzw(zw_q))
        else:
            self.ee_pos_zero_waist = self.ee_quat_zero_waist = None
        # Isaac's clip-end hold clamps ``time_steps`` to the last REAL frame (holosoma wbt.py ``per_motion_end - 1``): the
        # stand-flag window and the velocity command never see the zero-velocity pad of ``ClipReference.padded`` -- build
        # them on the T_original frames and hold the last value over the pad.
        self.T0 = T0
        speed = np.linalg.norm(np.asarray(ref.root_lin_vel_w, dtype=np.float64)[:T0, :2], axis=-1)
        yaw_rate = np.abs(np.asarray(ref.root_ang_vel_w, dtype=np.float64)[:T0, 2])
        self.half = params.stand_half_window(getattr(ref, "fps", 50))
        walking = walking_flags_from_clip(speed, yaw_rate, half=self.half, speed_thr=params.stand_speed_thr, yaw_thr=params.stand_yaw_rate_thr)
        self.walking = fit(walking) if walking.shape[0] < T else walking
        self.object = clip_object_decision(ref, params)


# ================================================================================================ policy
class HeroExportPolicy:
    """Stateful ONNX policy with observation history and joint-target reconstruction.

    Actuator tables ``kp``, ``kd``, ``effort_limit``, and ``default_dof_pos``
    configure the plant. ``preferred_urdf_alias`` identifies exports requiring
    a matching hand model.
    """

    kind = "hero_export_onnx"
    extra_dim = 0

    def __init__(
        self,
        onnx_dir: str | os.PathLike | None = None,
        *,
        sidecar: str | os.PathLike | None = None,
        onnx: str | os.PathLike | None = None,
        h_cmd: str = "auto",
        h_fixed: float | None = None,
        stand_flag: float | None = None,
        ref_lookahead: int | None = None,
        hold_mode: str = "zero_action",
        object_obs: str = "auto",
        object_rule: str = "auto",
        object_box_side: float | None = None,
        object_box_tol: float | None = None,
        max_start_bottom_z_m: float | None = None,
        delay_steps: int = 0,
        obs_clip: float | None | str = "contract",
        providers: Sequence[str] = ("CPUExecutionProvider",),
        intra_op_threads: int = 1,
        tag: str | None = None,
        h20_future_steps: Sequence[int] | None = None,
        odom_noise: OdometryNoise | Mapping[str, Any] | None = None,
        odom_source: str = "truth",
        anchor_obs: str = "live",
    ):
        """Initialize a policy from its export contract.

        ``object_rule`` applies metadata rules (auto), requires them (on), or
        uses the raw clip flag (off). Size and start-height arguments override
        metadata. ``delay_steps`` delays raw actions. ``h20_future_steps``
        overrides the exported reference horizon, with dimension validation.
        ``odom_noise`` perturbs the root-state observations; ``odom_source``
        selects exact simulator state (truth) or ``RobotState.odom`` (leg).
        """
        import onnxruntime as ort

        if h_cmd not in H_CMD_MODES:
            raise ValueError(f"h_cmd must be one of {H_CMD_MODES}, got {h_cmd!r}")
        if hold_mode not in HOLD_MODES:
            raise ValueError(f"hold_mode must be one of {HOLD_MODES}, got {hold_mode!r}")
        if object_obs not in OBJECT_OBS_MODES:
            raise ValueError(f"object_obs must be one of {OBJECT_OBS_MODES}, got {object_obs!r}")
        if object_rule not in OBJECT_RULE_MODES:
            raise ValueError(f"object_rule must be one of {OBJECT_RULE_MODES}, got {object_rule!r}")
        self.onnx_path, self.sidecar_path = resolve_hero_export(onnx_dir, sidecar=sidecar, onnx=onnx)
        so = ort.SessionOptions()
        so.intra_op_num_threads = int(intra_op_threads)
        so.inter_op_num_threads = 1
        so.log_severity_level = 3
        self.sess = ort.InferenceSession(str(self.onnx_path), so, providers=list(providers))
        self.onnx_meta = read_onnx_metadata(self.onnx_path, self.sess)
        self.sidecar_meta: dict[str, Any] | None = json.loads(self.sidecar_path.read_text()) if self.sidecar_path is not None else None
        overrides: dict[str, Any] = {"ref_lookahead_frames": int(ref_lookahead) if ref_lookahead is not None else None}
        if h_fixed is not None:
            overrides["h_cmd_default"] = float(h_fixed)
        if object_rule != "off":
            overrides.update(object_box_side=object_box_side, object_box_tol=object_box_tol, max_start_bottom_z_m=max_start_bottom_z_m)
        if h20_future_steps is not None:
            overrides["ref_root_pose_future_steps"] = tuple(int(v) for v in h20_future_steps)
        self.contract = contract_from_metadata(self.sidecar_meta, self.onnx_meta, command_overrides=overrides, obs_clip=obs_clip, object_rule_lookup=(object_rule != "off"))
        self._init_tables(h_cmd=h_cmd, stand_flag=stand_flag, hold_mode=hold_mode, object_obs=object_obs, object_rule=object_rule, delay_steps=delay_steps, odom_noise=odom_noise, odom_source=odom_source, anchor_obs=anchor_obs)
        self._check_graph()
        stem = self.onnx_path.stem
        self.tag = tag or f"{TAG_PREFIX}_{stem}"
        self.reset()

    # ------------------------------------------------------------------------------------------ setup
    def _init_tables(
        self, *, h_cmd: str, stand_flag: float | None, hold_mode: str, object_obs: str, object_rule: str = "auto", delay_steps: int = 0,
        odom_noise: OdometryNoise | Mapping[str, Any] | None = None, odom_source: str = "truth", anchor_obs: str = "live",
    ) -> None:
        c = self.contract
        if odom_source not in ODOM_SOURCES:
            raise ValueError(f"odom_source must be one of {ODOM_SOURCES}, got {odom_source!r}")
        self.odom_source = str(odom_source)
        if anchor_obs not in ANCHOR_OBS_MODES:
            raise ValueError(f"anchor_obs must be one of {ANCHOR_OBS_MODES}, got {anchor_obs!r}")
        self.anchor_obs = str(anchor_obs)  # :data:`ANCHOR_OBS_MODES`; "ontrack" = the drift-blind ablation of the HERO terms
        self.layout = c.layout
        self.cmd = c.command
        self.h_cmd_mode = h_cmd
        self.stand_flag_override = None if stand_flag is None else float(stand_flag)
        self.hold_mode = hold_mode
        self.object_obs = object_obs
        if object_obs == "on" and not self.layout.has_object_terms:
            raise ValueError(f"object_obs='on' but the export has no object terms (terms {list(self.layout.terms)})")
        self.feed_object = bool(self.layout.has_object_terms and object_obs != "off")
        # object rule: off -> strip the fields (raw has_object); on -> the command must carry a rule
        self.object_rule_mode = object_rule
        if object_rule == "off" and self.cmd.object_rule_active:
            from dataclasses import replace

            self.cmd = replace(self.cmd, object_box_side=None, max_start_bottom_z_m=None)
        if object_rule == "on" and not self.cmd.object_rule_active:
            raise ValueError("object_rule='on' but neither the sidecar (hero_command / preset) nor the overrides carry object_box_side / max_start_bottom_z_m")
        self.delay_steps = int(delay_steps)
        if self.delay_steps < 0:
            raise ValueError(f"delay_steps must be >= 0, got {delay_steps}")
        self._action_queue: deque[np.ndarray] = deque()
        self.dof_names = c.dof_names
        self.kp = c.kp.astype(np.float64)
        self.kd = c.kd.astype(np.float64)
        self.effort_limit = c.effort_limit.astype(np.float64)
        self.default_dof_pos = c.default_dof_pos.astype(np.float64)
        self.action_scale = c.action_scale.astype(np.float64)
        self.action_clip = c.action_clip
        self.residual_dof_idx = c.residual_dof_idx
        self.ref_attr = c.ref_attr
        self.obs_clip = c.obs_clip
        self.iteration = c.iteration
        self.history = HeroFrameHistory(self.layout)
        self.palm_offset = np.asarray([HC.PALM_OFFSET[s] for s in HC.EE_SIDES], dtype=np.float64)
        self._h16_slots = np.asarray([HC.HOLOSOMA_BODY_NAMES_32.index(n) for n in H16_BODY_NAMES], dtype=np.int64)
        self._stats: _ClipStats | None = None
        # h20 (HERO): future frames from the command block / override; MuJoCo-side odometry-noise perturbation (off by default)
        self.h20_future_steps: tuple[int, ...] = tuple(self.cmd.ref_root_pose_future_steps)
        if self.layout.has_anchor_term and len(self.h20_future_steps) * len(H20_PER_FRAME) != HERO_TERM_DIMS[ANCHOR_TERM]:
            raise ValueError(f"h20 needs {HERO_TERM_DIMS[ANCHOR_TERM] // len(H20_PER_FRAME)} future frames, got {self.h20_future_steps}")
        # h21 / h22 share the h20 horizon; h23 has none. The contract validates dimensions.
        self.anchor2_terms_present: tuple[str, ...] = self.layout.anchor2_present
        for term, per_frame in ((ANCHOR2_ROT_TERM, H21_FRAME_DIM), (ANCHOR2_HEIGHT_TERM, H22_FRAME_DIM)):
            if term in self.layout.terms and len(self.h20_future_steps) * per_frame != HERO_TERM_DIMS[term]:
                raise ValueError(f"{term} needs {HERO_TERM_DIMS[term] // per_frame} future frames, got {self.h20_future_steps}")
        self.odom_noise = OdometryNoise.coerce(odom_noise)
        n = self.odom_noise
        if n.planar_active and not self.layout.has_hero_plus_terms:
            raise ValueError("odom_noise only acts on the h20 term; this export has no h20_ref_root_pose_b")
        if n.planar_active and not self.layout.has_anchor_term and not n.yaw_active:
            raise ValueError("odom_noise xy fields act on h20 only; this export has no h20_ref_root_pose_b (yaw fields would act on h21)")
        if float(n.bias_rp_rad) > 0.0 and ANCHOR2_ROT_TERM not in self.layout.terms:
            raise ValueError(f"odom_noise.bias_rp_rad acts on {ANCHOR2_ROT_TERM}; this export has no such term")
        if n.height_active and ANCHOR2_HEIGHT_TERM not in self.layout.terms:
            raise ValueError(f"odom_noise.bias_z_m / walk_z_m act on {ANCHOR2_HEIGHT_TERM}; this export has no such term")
        if n.vel_active and ANCHOR2_VEL_TERM not in self.layout.terms:
            raise ValueError(f"odom_noise.vel_std_m_s acts on {ANCHOR2_VEL_TERM}; this export has no such term")
        self.resets = 0
        self._odom_rng = np.random.default_rng(int(self.odom_noise.seed))
        self._odom_offset = np.zeros(3, dtype=np.float64)  # [dx, dy, dyaw] of the pose ESTIMATE vs the true robot root
        self._odom_rp_bias = np.zeros(2, dtype=np.float64)  # [droll, dpitch] of the orientation estimate (body frame; h21)
        self._odom_z_offset = 0.0  # height estimate error (bias + walk; h22)
        self._odom_vel_noise = np.zeros(3, dtype=np.float64)  # per-step Gaussian on the heading-frame velocity (h23)
        #: the HERO terms the odometry estimate (leg) feeds (empty under ``odom_source="truth"`` or without HERO terms)
        self.odom_feeds_terms: tuple[str, ...] = tuple(t for t in HERO_PLUS_TERMS if t in self.layout.terms) if self.odom_source != "truth" else ()
        if self.odom_source != "truth" and not self.layout.has_hero_plus_terms:
            c.warnings.append(f"odom_source={self.odom_source!r} but the export has no HERO term (h20-h23): the estimator only records its error, the policy is unaffected")
        self.odom_estimate_steps = 0  # frames whose anchor terms were built from the odometry estimate (leg)
        if self.anchor_obs != "live" and not self.layout.has_hero_plus_terms:
            c.warnings.append(f"anchor_obs={self.anchor_obs!r} but the export has no HERO term (h20-h22): nothing to freeze, the policy is unaffected")

    def _check_graph(self) -> None:
        ins = {i.name: i for i in self.sess.get_inputs()}
        outs = self.sess.get_outputs()
        want = [n for n in self.contract.onnx_inputs if n in ins]
        if not want:
            if len(ins) in (1, 2):
                want = list(ins)
            else:
                raise ValueError(f"{self.onnx_path}: inputs {sorted(ins)} do not match the contract {self.contract.onnx_inputs}")
        self.input_names = tuple(want)
        for name in self.input_names:
            shape = list(ins[name].shape)
            if int(shape[-1]) != self.layout.total_dim:
                raise ValueError(f"{self.onnx_path}: input {name} width {shape[-1]} != layout {self.layout.total_dim} ({self.layout.history_length} x {self.layout.frame_dim})")
        out_name = self.contract.onnx_output if any(o.name == self.contract.onnx_output for o in outs) else outs[0].name
        self.output_name = out_name
        out_shape = list(next(o for o in outs if o.name == out_name).shape)
        if int(out_shape[-1]) != len(self.dof_names):
            raise ValueError(f"{self.onnx_path}: output {out_name} width {out_shape[-1]} != {len(self.dof_names)} dofs")

    @classmethod
    def tables_only(cls, contract: HeroExportContract, **kw: Any) -> "HeroExportPolicy":
        """Contract-only instance (no ONNX session): observation / command / action mapping work, ``act`` does not."""
        self = cls.__new__(cls)
        self.onnx_path = None
        self.sidecar_path = None
        self.sess = None
        self.onnx_meta = {}
        self.sidecar_meta = None
        self.contract = contract
        self._init_tables(
            h_cmd=kw.get("h_cmd", "auto"), stand_flag=kw.get("stand_flag"), hold_mode=kw.get("hold_mode", "zero_action"), object_obs=kw.get("object_obs", "auto"),
            object_rule=kw.get("object_rule", "auto"), delay_steps=int(kw.get("delay_steps", 0)), odom_noise=kw.get("odom_noise"), odom_source=kw.get("odom_source", "truth"),
            anchor_obs=kw.get("anchor_obs", "live"),
        )
        self.input_names = tuple(contract.onnx_inputs)
        self.output_name = contract.onnx_output
        self.tag = kw.get("tag") or f"{TAG_PREFIX}_tables"
        self.reset()
        return self

    # ------------------------------------------------------------------------------------------ protocol
    @property
    def has_object_inputs(self) -> bool:
        return self.layout.has_object_terms

    @property
    def has_anchor_inputs(self) -> bool:
        """True when the export consumes the h20 drift-feedback term."""
        return self.layout.has_anchor_term

    @property
    def has_anchor2_inputs(self) -> bool:
        """True when the export consumes orientation, height, or velocity feedback."""
        return self.layout.has_anchor2_terms

    @property
    def preferred_urdf_alias(self) -> str | None:
        return self.contract.preferred_urdf_alias

    @property
    def delay_applied_to(self) -> str:
        """Where a ``--delay-steps`` perturbation acts for this controller: the raw ACTION (holosoma semantics)."""
        return "action"

    def set_action_delay(self, delay_steps: int) -> None:
        """Change the control delay (raw-action FIFO, zeros after reset); takes effect at the next :meth:`reset`."""
        n = int(delay_steps)
        if n < 0:
            raise ValueError(f"delay_steps must be >= 0, got {delay_steps}")
        self.delay_steps = n
        self._prime_action_queue()

    def _prime_action_queue(self) -> None:
        self._action_queue.clear()
        for _ in range(self.delay_steps):
            self._action_queue.append(np.zeros(len(self.dof_names), dtype=np.float64))

    def _delayed_action(self, a: np.ndarray) -> np.ndarray:
        """holosoma ``_apply_action_delay``: push the new action, pop the one ``delay_steps`` control steps old."""
        if self.delay_steps <= 0:
            return a
        self._action_queue.append(np.array(a, dtype=np.float64, copy=True))
        return self._action_queue.popleft()

    def reset(self) -> None:
        self.history.reset()
        self.last_action = np.zeros(len(self.dof_names), dtype=np.float64)
        self.last_action_applied = np.zeros(len(self.dof_names), dtype=np.float64)
        self.steps = 0
        self.object_flag_steps = 0
        self.object_masked_by_rule_steps = 0
        self.odom_estimate_steps = 0
        self._stats = None
        self.last_obs: np.ndarray | None = None
        self.last_terms: dict[str, Any] | None = None
        self._prime_action_queue()
        self._reset_odometry_noise()

    def _reset_odometry_noise(self) -> None:
        """New episode: ``seed + reset_count`` -> bias ``N(0, bias_*)`` on (dx, dy, dyaw); the walk restarts from it.  anchor2:
        the roll / pitch bias (h21), the height bias (h22) and the first per-step velocity draw (h23) follow -- drawn only when
        their field is set, so the xy / yaw streams of a planar-only knob are unchanged."""
        self.resets = int(getattr(self, "resets", 0)) + 1
        self._odom_offset = np.zeros(3, dtype=np.float64)
        self._odom_rp_bias = np.zeros(2, dtype=np.float64)
        self._odom_z_offset = 0.0
        self._odom_vel_noise = np.zeros(3, dtype=np.float64)
        n = getattr(self, "odom_noise", None)
        if n is None or not n.active:
            return
        self._odom_rng = np.random.default_rng(int(n.seed) + self.resets)
        self._odom_offset[0:2] = self._odom_rng.normal(0.0, 1.0, size=2) * float(n.bias_xy_m)
        self._odom_offset[2] = self._odom_rng.normal(0.0, 1.0) * float(n.bias_yaw_rad)
        if float(n.bias_rp_rad) > 0.0:
            self._odom_rp_bias = self._odom_rng.normal(0.0, 1.0, size=2) * float(n.bias_rp_rad)
        if float(n.bias_z_m) > 0.0:
            self._odom_z_offset = float(self._odom_rng.normal(0.0, 1.0)) * float(n.bias_z_m)
        self._draw_velocity_noise()

    def _draw_velocity_noise(self) -> None:
        """h23: a fresh zero-mean Gaussian ``N(0, vel_std_m_s^2)`` per control step (no bias, no walk); zeros when off."""
        n = self.odom_noise
        if n.vel_active:
            self._odom_vel_noise = self._odom_rng.normal(0.0, 1.0, size=3) * float(n.vel_std_m_s)

    def _walk_odometry_noise(self) -> None:
        """One control step of the random walk (increment std ``walk_* * sqrt(dt)``); no-op when inactive.  Also advances the
        height walk (h22) and draws the next step's velocity noise (h23) when those fields are set."""
        n = self.odom_noise
        if not n.active:
            return
        s = np.sqrt(CONTROL_DT)
        if n.walk_xy_m != 0.0 or n.walk_yaw_rad != 0.0:
            self._odom_offset[0:2] += self._odom_rng.normal(0.0, 1.0, size=2) * float(n.walk_xy_m) * s
            self._odom_offset[2] += self._odom_rng.normal(0.0, 1.0) * float(n.walk_yaw_rad) * s
        if float(n.walk_z_m) > 0.0:
            self._odom_z_offset += float(self._odom_rng.normal(0.0, 1.0)) * float(n.walk_z_m) * s
        self._draw_velocity_noise()

    def odometry_orientation_estimate(self, root_quat: np.ndarray) -> np.ndarray:
        """The robot orientation h21 is built from: the yaw error of the shared planar estimate (``quat_from_yaw(dyaw)``, WORLD
        z -- the same rotation :meth:`odometry_pose_estimate` applies for h20) and the roll / pitch bias in the BODY frame
        (``q_est = Rz(dyaw) * q_true * R_xyz(droll, dpitch, 0)``, an IMU tilt error); identity when the rotation noise is off."""
        q = np.asarray(root_quat, dtype=np.float64).reshape(4)
        if not self.odom_noise.rot_active:
            return q
        q_est = q
        if float(self.odom_noise.bias_rp_rad) > 0.0:
            q_est = quat_mul(q_est, quat_from_euler_xyz(self._odom_rp_bias[0], self._odom_rp_bias[1], 0.0))
        if self.odom_noise.yaw_active:
            q_est = quat_mul(quat_from_yaw(self._odom_offset[2]), q_est)
        return q_est

    def odometry_height_estimate(self, root_pos: np.ndarray) -> float:
        """The robot root height h22 is built from: ``z_true + (bias + walk)``; exact when the height noise is off."""
        return float(np.asarray(root_pos, dtype=np.float64).reshape(3)[2]) + (float(self._odom_z_offset) if self.odom_noise.height_active else 0.0)

    def odometry_velocity_estimate(self, v_heading: np.ndarray) -> np.ndarray:
        """The heading-frame velocity h23 is built from: the true value plus this step's Gaussian draw (a common yaw offset of the
        odometry frame cancels in ``R_yaw^T v``, so the yaw error does not enter); exact when ``vel_std_m_s`` is 0."""
        v = np.asarray(v_heading, dtype=np.float64).reshape(3)
        return v + self._odom_vel_noise if self.odom_noise.vel_active else v

    def odometry_pose_estimate(self, root_pos: np.ndarray, root_quat: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """The robot planar pose h20 is built from: the true root pose shifted by the current odometry offset
        (``[dx, dy]`` in the WORLD frame, ``dyaw`` about +z); identity when ``odom_noise`` is inactive."""
        p = np.asarray(root_pos, dtype=np.float64).reshape(3)
        q = np.asarray(root_quat, dtype=np.float64).reshape(4)
        if not self.odom_noise.active:
            return p, q
        from sim2sim.mathutil import quat_mul

        p_est = p + np.array([self._odom_offset[0], self._odom_offset[1], 0.0])
        q_est = quat_mul(quat_from_yaw(self._odom_offset[2]), q)
        return p_est, q_est

    def reset_target(self, q_ref0: np.ndarray) -> np.ndarray:
        """holosoma ``reset_all``: one zero-action step -> ``q_default`` + the arm residual of the GIVEN reference pose
        (``zero_action``); ``clip`` holds that pose instead.  :meth:`reset_target_for` picks the Isaac lookahead frame."""
        q0 = np.asarray(q_ref0, dtype=np.float64).reshape(len(self.dof_names))
        if self.hold_mode == "clip":
            return q0.copy()
        return self.q_target(np.zeros(len(self.dof_names)), q0)

    def reset_target_for(self, ref: Any) -> np.ndarray:
        """Reset-step target from the clip: ``_refresh_hero_refs`` runs at reset with ``time_steps = 0``, so the residual
        of the zero-action step reads ``ref_upper_dof_pos`` of frame ``ref_lookahead_frames`` (``joint_pos`` refs and
        ``hold_mode='clip'`` stay at frame 0)."""
        if self.hold_mode == "clip" or self.ref_attr == "joint_pos":
            return self.reset_target(ref.joint_pos[0])
        return self.reset_target(ref.joint_pos[ref.clamp(int(self.cmd.ref_lookahead_frames))])

    def control(self, ref: Any, t: int, st: Any) -> np.ndarray:
        terms = self.frame_terms(ref, t, st)
        obs = self.observe(terms)
        raw = self.act(obs)
        self.last_action = raw.astype(np.float64)  # h00 = UNdelayed, UNclipped ActionManager.action
        self.last_obs, self.last_terms = obs, terms
        self.steps += 1
        a = self._delayed_action(self.last_action)  # control delay acts on the action; the residual uses the CURRENT ref
        self.last_action_applied = a
        return self.q_target(a, terms["q_ref_residual"])

    # ------------------------------------------------------------------------------------------ commands
    def stats(self, ref: Any) -> _ClipStats:
        """Cache one clip by object identity, retaining a strong reference to prevent ID reuse."""
        if self._stats is None or self._stats.ref is not ref or self._stats.T != int(ref.T):
            self._stats = _ClipStats(ref, self.cmd)
        return self._stats

    def ref_frame(self, ref: Any, t: int) -> int:
        return ref.clamp(int(t) + int(self.cmd.ref_lookahead_frames))

    def walking_flag(self, ref: Any, r: int) -> float:
        if self.stand_flag_override is not None:
            return float(self.stand_flag_override)
        if self.cmd.stand_flag_mode == "from_clip":
            return float(self.stats(ref).walking[r])
        return 0.0  # bernoulli_walk_0p6: stance in evaluation

    def h_command(self, ref: Any, r: int, walking: float) -> float:
        if self.h_cmd_mode == "fixed":
            return float(self.cmd.h_cmd_default)
        if self.h_cmd_mode == "clip":
            return max(float(self.stats(ref).h_ref[r]), float(self.cmd.h_cmd_min))
        return h_cmd_from_clip(float(self.stats(ref).h_ref[r]), self.cmd, walking)

    def commands_from_clip(self, ref: Any, t: int) -> dict[str, Any]:
        """HERO command / reference terms: lookahead frame ``r`` for the H command, frame ``t`` for the stock refs."""
        t = ref.clamp(t)
        r = self.ref_frame(ref, t)
        walking = self.walking_flag(ref, r)
        jp_r = np.asarray(ref.joint_pos[r], dtype=np.float64)
        waist = jp_r[_WAIST].copy()
        ee_p, ee_q = self._ee_reference(ref, r)
        if self.cmd.zero_waist_when_walking and walking > 0.5:
            waist[:] = 0.0
            st = self.stats(ref)
            if st.ee_pos_zero_waist is not None:
                ee_p, ee_q = st.ee_pos_zero_waist[r].copy(), st.ee_quat_zero_waist[r].copy()
        if self.cmd.stand_flag_mode == "from_clip":
            r_v = min(r, self.stats(ref).T0 - 1)  # padded tail: hold the last real frame's velocity (Isaac clamp)
            vel = vel_cmd_from_clip(ref.root_quat_w[r_v], ref.root_lin_vel_w[r_v], ref.root_ang_vel_w[r_v]) * walking
        else:
            vel = np.zeros(3)  # evaluation: zero linear command, heading controller idle (x walking = 0)
        ref_upper17 = jp_r[_UPPER17].copy()
        if self.cmd.zero_waist_when_walking and walking > 0.5:
            ref_upper17[0:3] = 0.0
        if self.ref_attr == "joint_pos":
            q_ref_res = np.asarray(ref.joint_pos[t], dtype=np.float64)[self.residual_dof_idx]
        else:
            cols = [list(_UPPER17).index(int(i)) for i in self.residual_dof_idx]
            q_ref_res = ref_upper17[cols]
        return {
            "frame": int(t),
            "frame_cmd": int(r),
            "frame_cmd_vel": int(min(r, self.stats(ref).T0 - 1)),
            "walking": float(walking),
            "h02_command_ang_vel": vel[2:3].copy(),
            "h03_command_base_height": np.array([self.h_command(ref, r, walking)]),
            "h04_command_lin_vel": vel[0:2].copy(),
            "h05_command_stand": np.array([walking]),
            "h06_command_waist_dofs": waist,
            "h12_ref_upper_dof_pos": jp_r[_ARM].copy(),
            "ref_ee_pos_pelvis": ee_p,
            "ref_ee_quat_pelvis": ee_q,
            "q_ref_residual": q_ref_res,
        }

    def _ee_reference(self, ref: Any, r: int) -> tuple[np.ndarray, np.ndarray]:
        """Clip palm pose in the clip's OWN pelvis frame ((2,3), (2,4) xyzw): npz ``ee_pos/quat_pelvis`` when loaded, else FK."""
        p = getattr(ref, "ee_pos_pelvis", None)
        q = getattr(ref, "ee_quat_pelvis", None)
        if p is not None and q is not None and p.shape[0] == ref.T:
            return np.asarray(p[r], dtype=np.float64), np.asarray(q[r], dtype=np.float64)
        pp, qq = ref.palm_pose_own_pelvis(r)
        return np.asarray(pp, dtype=np.float64), np.asarray(qq, dtype=np.float64)

    # ------------------------------------------------------------------------------------------ observation
    def proprio_terms(self, root_quat, root_ang_vel_b, dof_pos, dof_vel) -> dict[str, np.ndarray]:
        return {
            "h01_base_ang_vel": np.array(root_ang_vel_b, dtype=np.float64, copy=True).reshape(3),
            "h09_dof_pos": np.asarray(dof_pos, dtype=np.float64).reshape(-1) - self.default_dof_pos,
            "h10_dof_vel": np.array(dof_vel, dtype=np.float64, copy=True).reshape(-1),
            "h11_projected_gravity": projected_gravity(root_quat),
            "h13_roll_and_pitch": euler_roll_pitch(root_quat),
        }

    def h2_terms(self, ref: Any, t: int) -> dict[str, np.ndarray]:
        """h14-h16 from the stock MotionCommand frame ``t`` (no lookahead)."""
        t = ref.clamp(t)
        root_p, root_q = ref.root_pos_w[t], ref.root_quat_w[t]
        return {
            "h14_ref_lower_dof_pos": np.asarray(ref.joint_pos[t], dtype=np.float64)[_LEG].copy(),
            "h15_ref_root_pitch_roll": euler_roll_pitch(root_q),
            "h16_ref_body_pos_b": ref_body_pos_b(ref.body_pos_w[t, self._h16_slots], root_p, root_q),
        }

    def object_rule_decision(self, ref: Any) -> dict[str, Any]:
        """:func:`clip_object_decision` of this clip under the export's command (cached per clip)."""
        return dict(self.stats(ref).object)

    def clip_object_effective(self, ref: Any) -> bool:
        """True when an object track passes the exported size and start-height rules."""
        return bool(self.stats(ref).object["effective"])

    def object_terms(self, ref: Any, st: Any) -> dict[str, np.ndarray]:
        obj_p = getattr(st, "object_pos_w", None)
        dec = self.stats(ref).object
        has_object = bool(self.feed_object and obj_p is not None and dec["effective"])
        if self.feed_object and obj_p is not None and dec["has_object"] and not dec["effective"]:
            self.object_masked_by_rule_steps += 1
        p, o, f = object_terms(st.root_pos, st.root_quat, obj_p, getattr(st, "object_quat_w", None), has_object)
        self.object_flag_steps += int(has_object)
        out = {"h17_obj_pos_b": p, "h18_obj_ori_b": o, "h19_has_object_flag": f}
        return {k: v for k, v in out.items() if k in self.layout.terms}

    def anchor_terms(self, ref: Any, t: int, st: Any) -> dict[str, Any]:
        """The HERO terms of one frame, built from ONE odometry estimate (then the noise walk advances once):

        * h20 (20): :func:`ref_root_pose_b` of the clip root at :func:`h20_frame_indices` against the robot root of ``st``
          (its planar odometry estimate under ``odom_noise``), flattened frame-major ``[f0: dx dy sin cos | f1 ... ]``;
        * h21 (30): :func:`ref_root_rot_b` of the same clip frames against the orientation estimate
          (:meth:`odometry_orientation_estimate`: the SAME yaw error as h20 + the roll / pitch bias), frame-major ``[f0: 6D | f1 ...]``;
        * h22 (5): :func:`ref_root_height_b` against the height estimate (:meth:`odometry_height_estimate`);
        * h23 (3): :func:`base_lin_vel_odom` of ``st.root_lin_vel_w`` (+ this step's Gaussian, :meth:`odometry_velocity_estimate`).

        Source of the robot pose / velocity (:meth:`odometry_source_state`): the exact ``st.root_pos / root_quat / root_lin_vel_w`` under
        ``odom_source="truth"``, the odometry estimate ``st.odom`` (``pos_w / quat_xyzw / lin_vel_w``: the leg-odometry state under ``"leg"``).

        Bookkeeping: ``frames_h20`` (the clip frames used by h20 / h21 / h22), ``odom_source``, ``odom_offset`` (the planar estimate's
        [dx, dy, dyaw] error), ``odom_rp_bias`` / ``odom_z_offset`` / ``odom_vel_noise`` (zeros when exact).  Only the terms the
        layout carries are returned."""
        idx = h20_frame_indices(ref, ref.clamp(t), self.h20_future_steps)
        root_pos, root_quat, root_vel_w = self.odometry_source_state(st)
        out: dict[str, Any] = {"frames_h20": idx.tolist(), "odom_source": self.odom_source, "odom_offset": self._odom_offset.copy()}
        if self.layout.has_anchor_term:
            p_est, q_est = self.odometry_pose_estimate(root_pos, root_quat)
            out[ANCHOR_TERM] = ref_root_pose_b(p_est, q_est, ref.root_pos_w[idx], ref.root_quat_w[idx]).reshape(-1)
        if self.layout.has_anchor2_terms:
            out.update(odom_rp_bias=self._odom_rp_bias.copy(), odom_z_offset=float(self._odom_z_offset), odom_vel_noise=self._odom_vel_noise.copy())
            if ANCHOR2_ROT_TERM in self.layout.terms:
                out[ANCHOR2_ROT_TERM] = ref_root_rot_b(self.odometry_orientation_estimate(root_quat), ref.root_quat_w[idx]).reshape(-1)
            if ANCHOR2_HEIGHT_TERM in self.layout.terms:
                z_est = self.odometry_height_estimate(root_pos)
                out[ANCHOR2_HEIGHT_TERM] = ref_root_height_b(np.array([0.0, 0.0, z_est]), ref.root_pos_w[idx]).reshape(-1)
            if ANCHOR2_VEL_TERM in self.layout.terms:
                if root_vel_w is None:
                    raise ValueError(f"{ANCHOR2_VEL_TERM} needs RobotState.root_lin_vel_w (the MuJoCo root linear velocity, rollout.read_state); got None")
                out[ANCHOR2_VEL_TERM] = self.odometry_velocity_estimate(base_lin_vel_odom(root_quat, root_vel_w)).reshape(-1)
        if self.anchor_obs == "ontrack":  # ablation (ANCHOR_OBS_MODES): tell the policy it is exactly on the reference root; h23 stays live
            n = int(idx.shape[0])
            if ANCHOR_TERM in out:
                out[ANCHOR_TERM] = np.tile(np.array([0.0, 0.0, 0.0, 1.0]), n)
            if ANCHOR2_ROT_TERM in out:
                out[ANCHOR2_ROT_TERM] = np.tile(np.array([1.0, 0.0, 0.0, 0.0, 1.0, 0.0]), n)
            if ANCHOR2_HEIGHT_TERM in out:
                out[ANCHOR2_HEIGHT_TERM] = np.zeros(n, dtype=np.float64)
        out["anchor_obs"] = self.anchor_obs
        self._walk_odometry_noise()  # the walk advances AFTER the frame is built (first frame after reset = bias only)
        return out

    def odometry_source_state(self, st: Any) -> tuple[np.ndarray, np.ndarray, np.ndarray | None]:
        """Root position, XYZW orientation, and velocity used by the anchor terms.

        ``truth`` uses simulator state; ``leg`` requires an estimate in
        ``st.odom``. Missing estimates raise an error.
        """
        if self.odom_source == "truth":
            return np.asarray(st.root_pos, dtype=np.float64).reshape(3), np.asarray(st.root_quat, dtype=np.float64).reshape(4), getattr(st, "root_lin_vel_w", None)
        od = getattr(st, "odom", None)
        if od is None:
            raise ValueError(f"odom_source={self.odom_source!r} needs RobotState.odom (an estimate exposing pos_w, quat_xyzw and lin_vel_w)")
        self.odom_estimate_steps += 1
        return np.asarray(od.pos_w, dtype=np.float64).reshape(3), np.asarray(od.quat_xyzw, dtype=np.float64).reshape(4), np.asarray(od.lin_vel_w, dtype=np.float64).reshape(3)

    @property
    def odom_leg_steps(self) -> int:
        """Number of frames whose anchor terms used the leg-odometry estimate."""
        return int(self.odom_estimate_steps)

    def frame_terms(self, ref: Any, t: int, st: Any) -> dict[str, Any]:
        """All unscaled terms of the frame (+ bookkeeping keys ``frame`` / ``frame_cmd`` / ``walking`` / ``q_ref_residual`` /
        ``frames_h20`` / ``odom_source`` / ``odom_offset`` / ``odom_rp_bias`` / ``odom_z_offset`` / ``odom_vel_noise``)."""
        cmd = self.commands_from_clip(ref, t)
        terms: dict[str, Any] = {"h00_actions": self.last_action.copy()}
        terms.update(self.proprio_terms(st.root_quat, st.root_ang_vel_b, st.dof_pos, st.dof_vel))
        dp6, rot12 = ee_residual_terms(st.root_pos, st.root_quat, st.palm_pos_w, st.palm_quat_w, cmd["ref_ee_pos_pelvis"], cmd["ref_ee_quat_pelvis"])
        terms["h07_dif_local_rigid_body_pos_ee"] = dp6
        terms["h08_dif_local_rigid_body_rot_ee"] = rot12
        for k in ("h02_command_ang_vel", "h03_command_base_height", "h04_command_lin_vel", "h05_command_stand", "h06_command_waist_dofs", "h12_ref_upper_dof_pos"):
            terms[k] = cmd[k]
        if self.layout.has_lower_body_reference:  # Build h14-h16 only when present in the exported layout.
            terms.update(self.h2_terms(ref, t))
        if self.layout.has_object_terms:
            terms.update(self.object_terms(ref, st))
        if self.layout.has_hero_plus_terms:
            terms.update(self.anchor_terms(ref, t, st))
        for k in ("frame", "frame_cmd", "frame_cmd_vel", "walking", "q_ref_residual", "ref_ee_pos_pelvis", "ref_ee_quat_pelvis"):
            terms[k] = cmd[k]
        return terms

    def observe(self, terms: Mapping[str, Any]) -> np.ndarray:
        """Scale + sort into one frame, clip, push into the history, return the flattened float32 vector."""
        frame = self.layout.frame(terms)
        if self.obs_clip is not None:
            frame = np.clip(frame, -self.obs_clip, self.obs_clip)
        self.history.push(frame)
        return self.history.flat32()

    # ------------------------------------------------------------------------------------------ inference / action
    def act(self, obs: np.ndarray) -> np.ndarray:
        """Raw graph output (29,) float64; the same vector feeds every actor input."""
        if self.sess is None:
            raise RuntimeError("tables_only instance has no ONNX session")
        x = np.asarray(obs, dtype=np.float32).reshape(1, self.layout.total_dim)
        out = self.sess.run([self.output_name], {name: x for name in self.input_names})[0]
        return np.asarray(out, dtype=np.float64).reshape(len(self.dof_names))

    def clip_action(self, action: np.ndarray) -> np.ndarray:
        a = np.asarray(action, dtype=np.float64).reshape(len(self.dof_names))
        return np.clip(a, -self.action_clip, self.action_clip) if self.action_clip is not None else a

    def q_target(self, action: np.ndarray, q_ref_residual: np.ndarray) -> np.ndarray:
        """``q_default + action_scales * clip(a)`` + ``(q_ref - q_default)`` on the residual joints (holosoma order)."""
        scaled = self.action_scale * self.clip_action(action)
        q = self.default_dof_pos + scaled
        if self.residual_dof_idx.size:
            q_ref = np.asarray(q_ref_residual, dtype=np.float64).reshape(-1)
            if q_ref.shape[0] == len(self.dof_names) and self.residual_dof_idx.size != len(self.dof_names):
                q_ref = q_ref[self.residual_dof_idx]
            q[self.residual_dof_idx] += q_ref - self.default_dof_pos[self.residual_dof_idx]
        return q

    # ------------------------------------------------------------------------------------------ meta
    def object_rule_summary(self) -> dict[str, Any]:
        """The object rule this controller applies (series meta ``object_rule``): mode, fields, where they came from."""
        return {"mode": self.object_rule_mode, **self.cmd.object_rule(), "source": self.contract.fields_used.get("object_rule", "default:off"), "default_box_size_m": DEFAULT_BOX_SIZE_M}

    def object_obs_summary(self) -> dict[str, Any]:
        out = {
            "mode": self.object_obs,
            "policy_has_object_inputs": bool(self.layout.has_object_terms),
            "fed": self.feed_object,
            "steps": int(self.steps),
            "flag_steps": int(self.object_flag_steps),
        }
        if self.layout.has_object_terms:
            # steps whose clip has an object track and a box in the state but the Isaac object rule masked it (flag 0)
            out["masked_by_rule_steps"] = int(self.object_masked_by_rule_steps)
            out["clip_object_effective"] = (bool(self._stats.object["effective"]) if self._stats is not None else None)
        return out

    def describe(self) -> dict[str, Any]:
        d = {
            "kind": self.kind,
            "tag": self.tag,
            "onnx": str(self.onnx_path) if self.onnx_path else None,
            "onnx_bytes": (self.onnx_path.stat().st_size if self.onnx_path else None),
            "sidecar": str(self.sidecar_path) if self.sidecar_path else None,
            "iteration": self.iteration,
            "onnx_inputs_used": list(self.input_names),
            "onnx_output_used": self.output_name,
            "same_obs_to_both_inputs": True,
            "obs_dim": self.layout.total_dim,
            "frame_dim": self.layout.frame_dim,
            "history_length": self.layout.history_length,
            "history_layout": self.layout.history_layout,
            "history_padding": "zeros_after_reset",
            "arm": self.layout.arm,
            "lower_body_reference": self.layout.has_lower_body_reference,
            "object_mode": self.layout.object_mode,
            "object_obs": self.object_obs,
            "h_cmd_mode": self.h_cmd_mode,
            "stand_flag_override": self.stand_flag_override,
            "hold_mode": self.hold_mode,
            "ref_lookahead": int(self.cmd.ref_lookahead_frames),
            "h16_body_names": list(H16_BODY_NAMES) if self.layout.has_lower_body_reference else None,
            "control_hz": 50,
            "delay_steps": int(self.delay_steps),
            "delay_applied_to": self.delay_applied_to,
            "object_rule": self.object_rule_summary(),
            "anchor_term": self.layout.has_anchor_term,
            "h20": (
                {"term": ANCHOR_TERM, "future_steps": list(self.h20_future_steps), "per_frame": list(H20_PER_FRAME), "frame": "robot_heading",
                 "base_frame": "stock_t", "dims": HERO_TERM_DIMS[ANCHOR_TERM],
                 "future_steps_source": self.contract.fields_used.get("h20_future_steps", "default")}
                if self.layout.has_anchor_term else None
            ),
            "odom_noise": self.odom_noise.as_dict(),
            "odom_source": self.odom_source,
            "odom_feeds_terms": list(self.odom_feeds_terms),
            "anchor_obs": self.anchor_obs,
            "anchor2_terms": list(self.anchor2_terms_present),
            "h21": (
                {"term": ANCHOR2_ROT_TERM, "future_steps": list(self.h20_future_steps), "per_frame": list(H21_PER_FRAME), "frame": "robot_root",
                 "rep": "rot6d_first_two_columns_column_major", "identity": [1, 0, 0, 0, 1, 0], "base_frame": "stock_t", "dims": HERO_TERM_DIMS[ANCHOR2_ROT_TERM]}
                if ANCHOR2_ROT_TERM in self.layout.terms else None
            ),
            "h22": (
                {"term": ANCHOR2_HEIGHT_TERM, "future_steps": list(self.h20_future_steps), "per_frame": ["dz"], "unit": "m", "base_frame": "stock_t",
                 "dims": HERO_TERM_DIMS[ANCHOR2_HEIGHT_TERM]}
                if ANCHOR2_HEIGHT_TERM in self.layout.terms else None
            ),
            "h23": (
                {"term": ANCHOR2_VEL_TERM, "frame": "robot_heading", "source": "root_lin_vel_w (MuJoCo qvel[0:3])", "dims": HERO_TERM_DIMS[ANCHOR2_VEL_TERM]}
                if ANCHOR2_VEL_TERM in self.layout.terms else None
            ),
        }
        d.update({k: v for k, v in self.contract.describe().items() if k not in d})
        return d


__all__ = [
    "ACTOR_GROUP",
    "ANCHOR_TERM",
    "ANCHOR_TERMS",
    "ANCHOR2_HEIGHT_TERM",
    "ANCHOR2_ROT_TERM",
    "ANCHOR2_TERMS",
    "ANCHOR2V_TERMS",
    "ANCHOR2_VEL_TERM",
    "ANCHOR_TERMS_KEY",
    "H20_FUTURE_STEPS",
    "H20_PER_FRAME",
    "H21_FRAME_DIM",
    "H21_PER_FRAME",
    "H22_FRAME_DIM",
    "HERO_PLUS_FUTURE_STEPS_KEYS",
    "HERO_PLUS_PER_FRAME_DIM",
    "ANCHOR_OBS_MODES",
    "HERO_PLUS_TERMS",
    "ODOM_ANCHOR2_FIELDS",
    "ODOM_PLANAR_FIELDS",
    "ODOM_SOURCES",
    "OdometryNoise",
    "OdometryNoiseState",
    "base_lin_vel_odom",
    "ref_root_height_b",
    "ref_root_rot_b",
    "rot6d_cols",
    "h20_frame_indices",
    "heading_yaw",
    "quat_from_yaw",
    "ref_root_pose_b",
    "wrap_to_pi",
    "CONTRACTS",
    "CONTRACT_RESIDUAL_ALL29",
    "CONTRACT_RESIDUAL_UPPER",
    "DEFAULT_BOX_SIZE_M",
    "OBJECT_RULE_FIELDS",
    "OBJECT_RULE_MODES",
    "clip_object_decision",
    "object_rule_from_preset",
    "H16_BODY_NAMES",
    "H1_TERMS",
    "H1_REQUIRED_TERMS",
    "DELTA_EE_TERMS",
    "H2_EXTRA_TERMS",
    "HERO_ACTION_CLIP",
    "HERO_OBS_CLIP",
    "HERO_TERM_DIMS",
    "HERO_TERM_SCALES",
    "CONTROL_DT",
    "HISTORY_LAYOUTS",
    "HISTORY_LAYOUT_FRAME_MAJOR",
    "HISTORY_LAYOUT_TERM_MAJOR",
    "HOLD_MODES",
    "H_CMD_MODES",
    "OBJECT_OBS_MODES",
    "OBJECT_TERMS",
    "ONNX_INPUT_NAMES",
    "ONNX_OUTPUT_NAME",
    "PADDLE_URDF_FILE_NAME",
    "SIDECAR_SCHEMA",
    "SIDECAR_SCHEMAS_KNOWN",
    "SIDECAR_SUFFIX",
    "HeroCommandParams",
    "HeroExportContract",
    "HeroExportLayout",
    "HeroExportPolicy",
    "HeroFrameHistory",
    "contract_from_metadata",
    "ee_residual_terms",
    "euler_roll_pitch",
    "find_hero_sidecars",
    "h_cmd_from_clip",
    "object_terms",
    "pick_latest_dual_onnx",
    "pick_latest_hero_sidecar",
    "projected_gravity",
    "read_onnx_metadata",
    "ref_body_pos_b",
    "resolve_hero_export",
    "vel_cmd_from_clip",
    "walking_flags_from_clip",
]
