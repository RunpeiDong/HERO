"""Would-terminate causes of the training termination set, evaluated in numpy (the rollout is never reset on failure).

Mirror of the training termination terms (``hero_isaacsim/managers/termination``); anchor = the pelvis.

* ``anchor_pos``      ``|z_ref - z_robot| > 0.15`` (adaptive: 0.75 when the REFERENCE root height < 0.5)
* ``anchor_ori_full`` ``quat_error_magnitude(q_ref, q_robot)^2 > 0.2`` rad^2 (full orientation)
* ``ee_body_pos``     any of ankle_roll L/R, wrist_yaw L/R with ``|z_ref_reanchored - z_robot| > 0.15`` (adaptive)
* ``foot_pos_xyz``    any ankle_roll with ``||p_ref_reanchored - p_robot|| > 0.2``
* ``anchor_xy``       ``||(p_ref - p_robot)_xy|| > 0.5``
* ``fall``            evaluator-only guard: robot pelvis z < 0.30 while the reference pelvis z > 0.50
* ``fall_low``        (hero_bench_v1 low-posture layers; ``TerminationConfig.fall_low_ref_margin_m``) -- the low-posture extension of ``fall``,
                      OFF by default (None = every legacy number unchanged): when the REFERENCE pelvis is at or below ``fall_ref_pelvis_z_min``
                      (0.50 m: low tables, floor picks, where the plain rule is suppressed) the robot pelvis below ``ref pelvis - margin``
                      (default margin 0.20 m) also counts as ``fall``.  Continuous with the plain rule at 0.50 m (0.50 - 0.20 = 0.30).  It is an
                      alias, not a new cause: ``--fail-causes fall_low`` (:func:`parse_fail_causes`) enables the margin and counts ``fall``;
                      the flags keep :data:`CAUSE_ORDER`.

Re-anchoring (the training command's per-step rule): the reference bodies are moved to the ROBOT anchor xy + heading
(yaw only), keeping the height difference: ``delta_q = yaw_quat(q_robot_anchor * inv(q_ref_anchor))``,
``p_rel = p_robot_anchor + [0, 0, (p_ref_anchor - p_robot_anchor)_z] + R(delta_q) (p_body_ref - p_ref_anchor)``.
Training computes it with the robot pose from BEFORE the physics step of the frame it belongs to; the rollout mirrors that by
re-anchoring right after advancing the reference frame and checking after the physics step.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Mapping, Sequence

import numpy as np

from sim2sim.mathutil import quat_apply, quat_conj, quat_error_magnitude, quat_mul, yaw_quat

HEIGHT_BODIES: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link")
FOOT_BODIES: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link")
CAUSE_ORDER: tuple[str, ...] = ("anchor_pos", "anchor_ori_full", "ee_body_pos", "foot_pos_xyz", "anchor_xy", "fall")
TRAINING_CAUSES: tuple[str, ...] = CAUSE_ORDER[:-1]


@dataclass(frozen=True)
class TerminationConfig:
    anchor_height_threshold: float = 0.15
    threshold_adaptive: bool = True
    down_threshold: float = 0.75
    root_height_threshold: float = 0.5
    anchor_ori_threshold_rad2: float = 0.2
    body_height_threshold: float = 0.15
    body_pos_threshold: float = 0.2
    anchor_xy_threshold: float = 0.5
    fall_pelvis_z: float = 0.30
    fall_ref_pelvis_z_min: float = 0.50
    #: hero_bench_v1 ``fall_low`` (module docstring): with the reference pelvis <= ``fall_ref_pelvis_z_min`` the robot pelvis below
    #: ``ref pelvis - this`` is a fall too; None (default) = the plain rule only (legacy numbers unchanged)
    fall_low_ref_margin_m: float | None = None
    height_bodies: tuple[str, ...] = HEIGHT_BODIES
    foot_bodies: tuple[str, ...] = FOOT_BODIES
    # Causes that count towards ``first_fail_step`` (all causes are always evaluated and reported per clip).  The
    # default is the full training set; a policy that never received a pelvis-pose reference (HERO's upper-body ONNX
    # only gets h + upper-body joints + deltaEE) is fairly judged with ``("fall", "anchor_xy")`` = fell / walked away.
    fail_causes: tuple[str, ...] = CAUSE_ORDER

    def __post_init__(self) -> None:
        bad = [c for c in self.fail_causes if c not in CAUSE_ORDER]
        if bad:
            raise ValueError(f"unknown termination cause(s) {bad}; known: {CAUSE_ORDER}")
        if self.fall_low_ref_margin_m is not None and not (float(self.fall_low_ref_margin_m) > 0.0):
            raise ValueError(f"fall_low_ref_margin_m must be > 0 m or None, got {self.fall_low_ref_margin_m!r}")

    @property
    def tracked_bodies(self) -> tuple[str, ...]:
        seen: list[str] = []
        for n in self.height_bodies + self.foot_bodies:
            if n not in seen:
                seen.append(n)
        return tuple(seen)


def reanchor_body_pos(
    ref_anchor_pos: np.ndarray,
    ref_anchor_quat: np.ndarray,
    robot_anchor_pos: np.ndarray,
    robot_anchor_quat: np.ndarray,
    ref_body_pos: np.ndarray,
) -> np.ndarray:
    """``body_pos_relative_w`` for ``ref_body_pos (B, 3)`` (see module docstring)."""
    delta_q = yaw_quat(quat_mul(np.asarray(robot_anchor_quat), quat_conj(np.asarray(ref_anchor_quat))))
    delta_pos = np.asarray(ref_anchor_pos, dtype=np.float64) - np.asarray(robot_anchor_pos, dtype=np.float64)
    delta_pos = np.array([0.0, 0.0, delta_pos[2]])
    return np.asarray(robot_anchor_pos)[None, :] + delta_pos[None, :] + quat_apply(delta_q[None, :], np.asarray(ref_body_pos) - np.asarray(ref_anchor_pos)[None, :])


def _height_threshold(cfg: TerminationConfig, base: float, ref_root_height: float) -> float:
    if cfg.threshold_adaptive and ref_root_height < cfg.root_height_threshold:
        return cfg.down_threshold
    return base


def check_terminations(
    cfg: TerminationConfig,
    *,
    ref_root_pos: np.ndarray,
    ref_root_quat: np.ndarray,
    robot_root_pos: np.ndarray,
    robot_root_quat: np.ndarray,
    rel_body_pos: Mapping[str, np.ndarray],
    robot_body_pos: Mapping[str, np.ndarray],
) -> dict[str, bool]:
    """Per-cause flags in :data:`CAUSE_ORDER`.  ``rel_body_pos`` = re-anchored reference body positions (world),
    ``robot_body_pos`` = robot body positions (world), both keyed by body name."""
    ref_root_height = float(ref_root_pos[2])
    robot_root_height = float(robot_root_pos[2])
    out: dict[str, bool] = {}
    thr_anchor = _height_threshold(cfg, cfg.anchor_height_threshold, ref_root_height)
    out["anchor_pos"] = bool(abs(ref_root_pos[2] - robot_root_pos[2]) > thr_anchor)
    ang = float(quat_error_magnitude(ref_root_quat, robot_root_quat))
    out["anchor_ori_full"] = bool(ang * ang > cfg.anchor_ori_threshold_rad2)
    thr_body = _height_threshold(cfg, cfg.body_height_threshold, ref_root_height)
    out["ee_body_pos"] = bool(any(abs(rel_body_pos[n][2] - robot_body_pos[n][2]) > thr_body for n in cfg.height_bodies))
    out["foot_pos_xyz"] = bool(any(np.linalg.norm(rel_body_pos[n] - robot_body_pos[n]) > cfg.body_pos_threshold for n in cfg.foot_bodies))
    out["anchor_xy"] = bool(np.linalg.norm(np.asarray(ref_root_pos[:2]) - np.asarray(robot_root_pos[:2])) > cfg.anchor_xy_threshold)
    fall = robot_root_height < cfg.fall_pelvis_z and ref_root_height > cfg.fall_ref_pelvis_z_min
    if cfg.fall_low_ref_margin_m is not None and ref_root_height <= cfg.fall_ref_pelvis_z_min:
        # fall_low (module docstring): low reference -> the robot pelvis more than the margin below the reference pelvis is a fall
        fall = fall or robot_root_height < ref_root_height - float(cfg.fall_low_ref_margin_m)
    out["fall"] = bool(fall)
    return out


# ------------------------------------------------------------------------------------------------ --fail-causes parsing (runner + tests)
#: default ``fall_low`` margin (m) enabled by the ``fall_low`` alias
FALL_LOW_DEFAULT_MARGIN_M: float = 0.20
FALL_LOW_ALIAS: str = "fall_low"


def parse_fail_causes(text: str | None, default: Sequence[str] = CAUSE_ORDER) -> tuple[tuple[str, ...], float | None]:
    """``--fail-causes`` text -> ``(causes, fall_low_margin_m)``.  Comma list of :data:`CAUSE_ORDER` names plus the alias ``fall_low`` =
    count ``fall`` AND enable the low-posture margin (:data:`FALL_LOW_DEFAULT_MARGIN_M`); ``fall_low:<m>`` sets the margin.  Order is kept,
    duplicates dropped; empty / None -> ``default`` with no margin.  Unknown names are left for :class:`TerminationConfig` to reject."""
    items = [c.strip() for c in str(text or "").split(",") if c.strip()]
    if not items:
        return tuple(default), None
    causes: list[str] = []
    margin: float | None = None
    for c in items:
        name, _, arg = c.partition(":")
        if name == FALL_LOW_ALIAS:
            margin = float(arg) if arg else FALL_LOW_DEFAULT_MARGIN_M
            name = "fall"
        if name not in causes:
            causes.append(name)
    return tuple(causes), margin


def termination_config_from_causes(text: str | None, **overrides) -> TerminationConfig:
    """:func:`parse_fail_causes` -> :class:`TerminationConfig` (``fail_causes`` + ``fall_low_ref_margin_m``; ``overrides`` = other fields)."""
    causes, margin = parse_fail_causes(text)
    return TerminationConfig(fail_causes=causes, fall_low_ref_margin_m=margin, **overrides)


__all__ = ["CAUSE_ORDER", "FALL_LOW_ALIAS", "FALL_LOW_DEFAULT_MARGIN_M", "FOOT_BODIES", "HEIGHT_BODIES", "TRAINING_CAUSES", "TerminationConfig",
           "check_terminations", "parse_fail_causes", "reanchor_body_pos", "termination_config_from_causes"]
