"""Closed-loop replanning + goal adjustment for the benchmark rollouts (HERO, arXiv 2602.16705, Sec. "Replanning and
Goal Adjustment").

What HERO does (paper + the deployed benchmark script's ``_update_grasp2root_from_odometry``):

* the goal is fixed in the WORLD; at every planner update the goal is re-expressed in the current base frame through
  odometry and the planner replans the remaining upper-body reference from the CURRENT joint state;
* goal adjustment: the commanded goal ``g`` (initialised to the true goal) is nudged against the measured world-frame EE
  error ``e = p_EE - T`` -- ``g <- g - beta e`` with ``beta = SHIFT_GAIN = 0.6``, translation only (``ROT_GAIN = 0``),
  only while ``|e| < REPLAN_DISTANCE = 15 cm`` (or after the 3rd replan), per-update shift clipped to ``MAX_STEP_SHIFT =
  1 cm`` (the paper text says 5 mm), skipped below ``STOP_SHIFT_GAIN_THRESHOLD = 2 cm``, and everything stops once
  ``|e| <= STAY_THRESHOLD = 1.75 cm`` ("stay there": the reference freezes).  The deployed code re-derives ``g`` from the
  world goal at every update before applying ONE shift (``adjust_mode="reset"``); the paper's formula accumulates
  (``adjust_mode="accumulate"``, default here).

What this module does (simulator ground truth as odometry / FK):

* :class:`GoalAdjuster` -- the pure decision logic above (stay / skip / replan + the goal update);
* :class:`BenchReplanner` -- at every replan event it reads the live robot state (root pose, 29 joints, both feet) and
  re-solves the whole-body reference with the benchmark's mink IK (``data_tools.hero_reach_generator`` model, solver,
  foot / posture / support machinery): a straight palm path from the current palm pose to the (adjusted) world goal,
  feet flat where they are, the base either driven back to the ORIGINAL plan's end state (``base_mode="clip"``) or held
  at its CURRENT pose (``base_mode="current"``: HERO plans arms + waist only, its legs are its own -- the cuRobo 17-DoF
  analogue).  The solved frames are FK'd to the 32-body layout (``data_tools.fk_mujoco``) and returned as an in-memory
  :class:`ClipReference` whose frame 0 is the robot's own state, so every observation term is consistent with where the
  robot actually is.

Metrics are NOT affected: the rollout keeps scoring against the ORIGINAL clip on its own timeline (world-fixed goal),
see :func:`sim2sim.bench.rollout.rollout_clip`.

Handover smoothing.  At the swap the new reference's frame 0 IS the robot, so the EE-error / joint-error observations collapse
to ~0 and the reference palm velocity restarts from zero while the robot is mid-motion -> the policy relaxes, then re-accelerates
(a visible jerk at every event).  Three independent :class:`ReplanConfig` knobs, all OFF by default (byte-identical plans / series):

* ``handover_blend_s`` -- for the first ``blend_s`` seconds after the swap the reference the policy sees is a crossfade
  ``old_ref(t_old + i) -> new_ref(i)`` on the shared 50 Hz grid, weight ``i / n_blend`` (frame 0 == the old reference's
  frame at the swap instant, frame ``n_blend`` == the new plan's own frame); positions / dofs linear, quaternions slerp;
  the blended frames are BAKED into the returned :class:`ClipReference` (:func:`blend_handover_frames`).  The crossfade removes
  the frame-0 POSITION jump only; velocity continuity needs ``start_pose="reference"`` (+ ``start_velocity="match"``);
* ``start_velocity="match"`` -- the new palm path leaves its start point with the OLD reference's palm velocity at the swap
  instant (central difference of ``palm_pose_w``, :func:`ref_palm_velocity_w`) on a quintic (minimum-jerk) profile
  ``p0 + (p1 - p0) S5(u) + v0 T H1(u)`` (:func:`palm_path`); ``"zero"`` keeps the original cubic smoothstep from rest;
* ``start_pose="reference"`` -- the new palm path starts at the OLD reference's current world palm pose (not the robot's)
  and the IK is seeded with the old reference's arm joints; the base / legs / waist still come from the robot.  The start
  configuration is first settled onto the frame-0 targets (``_settle_start``) so frame 0 lands on the old palm.

The knobs need the reference the controller is following at the swap: :class:`BenchReplanner` tracks it itself
(``reference=`` = the rollout's (padded) clip; every returned plan becomes the live reference at its step).

hero_bench_v1: a manifest row may carry ``table: None`` (floor picks: the IK model is built without a slab) and clips of the retract
layer carry their OWN retract segment -- :func:`clip_replan_config` gives such a row ``ReplanConfig.stop_frame = hold_end_frame``
(``retract_start_frame`` when present).  At ``stop_frame`` the replanner stops issuing replans and hands the controller back to the ORIGINAL
clip resumed at that frame (:func:`resume_reference`, crossfaded over ``handover_blend_s``) so the clip's own retract runs;
``summary()["stop"]`` records it.  Default ``stop_frame=None`` = the plain closed loop.
"""

from __future__ import annotations

import dataclasses
import json
import math
import os
import time
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hero_isaacsim import constants as HC
from sim2sim.mathutil import quat_to_mat, xyzw_to_wxyz
from sim2sim.reference import ClipReference

FPS: int = 50
SIDES: tuple[str, str] = ("left", "right")
FIRST_EVENTS: tuple[str, ...] = ("reach_end", "start")
BASE_MODES: tuple[str, ...] = ("clip", "current")
ADJUST_MODES: tuple[str, ...] = ("accumulate", "reset")
START_VELOCITIES: tuple[str, ...] = ("zero", "match")   # palm path leaves its start at rest | with the old reference's palm velocity
START_POSES: tuple[str, ...] = ("robot", "reference")    # palm path starts at the robot's palm | at the OLD reference's current palm pose
_LEG_IDX = {"left": (0, 3, 4), "right": (6, 9, 10)}  # hip_pitch, knee, ankle_pitch in DOF_NAMES order
_WAIST_YAW, _WAIST_PITCH = 12, 14


# ================================================================================================ configuration
@dataclasses.dataclass(frozen=True)
class ReplanConfig:
    period_s: float = 1.0                 # replan cadence after the first event
    first_event: str = "reach_end"        # anchor of the first event: the original plan's reach_end frame, or step 1
    first_delay_s: float = 0.0            # extra delay after the anchor
    base_mode: str = "clip"               # "clip" (drive the base to the original plan's end state) | "current" (hold it)
    hold_s: float = 12.0                  # hold appended after the replanned reach (last solved frame repeated)
    reach_min_s: float = 0.5
    reach_max_s: float = 3.0
    hold_solved_frames: int = 5           # hold frames actually solved by the QP before repeating the last one
    stay_threshold_m: float = 0.0175      # HERO STAY_THRESHOLD: below this the reference freezes for good
    skip_below_m: float = 0.02            # HERO STOP_SHIFT_GAIN_THRESHOLD: no replan for such a small error
    max_replans: int = 20                 # HERO MAX_REPLAN_NUMBER
    goal_adjust: bool = False
    adjust_gain: float = 0.6              # HERO SHIFT_GAIN (paper beta)
    adjust_max_step_m: float = 0.01       # HERO MAX_STEP_SHIFT (paper text: 5 mm)
    adjust_gate_m: float = 0.15           # HERO REPLAN_DISTANCE: adjust only when |e| < 15 cm ...
    adjust_after_replans: int = 2         # ... or after this many replans (``replan_times > 2``)
    adjust_mode: str = "accumulate"       # paper formula (accumulate) | deployed code (reset from the world goal each update)
    # ---- handover smoothing (module docstring; defaults = the original hard swap) ----
    handover_blend_s: float = 0.0         # crossfade old -> new reference over this long after the swap (0 = hard swap)
    start_velocity: str = "zero"          # "zero": cubic smoothstep from rest | "match": quintic leaving with the old reference's palm velocity
    start_pose: str = "robot"             # "robot": palm path starts at the robot's palm | "reference": at the OLD reference's current palm pose
    # ---- hero_bench_v1: clips that carry their OWN retract segment ----
    stop_frame: int | None = None         # stop issuing replans at this step (= the row's hold_end_frame for a retract clip) and hand the
                                          # controller back to the ORIGINAL clip resumed at that frame (crossfaded over handover_blend_s), so the
                                          # clip's own retract segment runs; None (default) = never

    def __post_init__(self) -> None:
        if self.first_event not in FIRST_EVENTS:
            raise ValueError(f"first_event must be one of {FIRST_EVENTS}, got {self.first_event!r}")
        if self.base_mode not in BASE_MODES:
            raise ValueError(f"base_mode must be one of {BASE_MODES}, got {self.base_mode!r}")
        if self.adjust_mode not in ADJUST_MODES:
            raise ValueError(f"adjust_mode must be one of {ADJUST_MODES}, got {self.adjust_mode!r}")
        if self.start_velocity not in START_VELOCITIES:
            raise ValueError(f"start_velocity must be one of {START_VELOCITIES}, got {self.start_velocity!r}")
        if self.start_pose not in START_POSES:
            raise ValueError(f"start_pose must be one of {START_POSES}, got {self.start_pose!r}")
        if self.period_s <= 0 or self.hold_s <= 0 or self.reach_min_s <= 0 or self.reach_max_s < self.reach_min_s:
            raise ValueError("period_s / hold_s / reach_min_s must be > 0 and reach_max_s >= reach_min_s")
        if not (self.handover_blend_s >= 0.0):
            raise ValueError(f"handover_blend_s must be >= 0, got {self.handover_blend_s!r}")
        if self.stop_frame is not None and (int(self.stop_frame) != self.stop_frame or int(self.stop_frame) < 1):
            raise ValueError(f"stop_frame must be a positive frame index or None, got {self.stop_frame!r}")

    @property
    def period_frames(self) -> int:
        return max(int(round(self.period_s * FPS)), 1)

    @property
    def blend_frames(self) -> int:
        """Crossfade length in frames (``handover_blend_s`` on the 50 Hz grid; 0 = hard swap)."""
        return max(int(round(self.handover_blend_s * FPS)), 0)

    @property
    def needs_live_reference(self) -> bool:
        """True when any handover knob is on: the replanner must know the reference the controller follows at the swap."""
        return self.blend_frames > 0 or self.start_velocity != "zero" or self.start_pose != "robot"

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# ================================================================================================ goal adjustment
class GoalAdjuster:
    """HERO ``_update_grasp2root_from_odometry`` decision logic on the WORLD-frame EE error of the reaching hand."""

    def __init__(self, goal_pos_w: np.ndarray, cfg: ReplanConfig):
        self.cfg = cfg
        self.goal_orig = np.asarray(goal_pos_w, dtype=np.float64).reshape(3).copy()
        self.goal = self.goal_orig.copy()
        self.n_replans = 0
        self.stayed_at: int | None = None
        self.events: list[dict[str, Any]] = []

    @property
    def stayed(self) -> bool:
        return self.stayed_at is not None

    def decide(self, palm_pos_w: np.ndarray, step: int) -> str:
        """``"stay"`` (converged, frozen for good) | ``"skip"`` (no replan this update) | ``"replan"`` (``self.goal`` is the
        target to plan to; adjusted first when enabled)."""
        cfg = self.cfg
        e = np.asarray(palm_pos_w, dtype=np.float64).reshape(3) - self.goal_orig
        n = float(np.linalg.norm(e))
        shift_cm = 0.0
        if self.stayed:
            action = "stay"
        elif n <= cfg.stay_threshold_m:
            self.stayed_at = int(step)
            action = "stay"
        elif n < cfg.skip_below_m or self.n_replans >= cfg.max_replans:
            action = "skip"
        else:
            if cfg.goal_adjust and (n < cfg.adjust_gate_m or self.n_replans > cfg.adjust_after_replans):
                shift = cfg.adjust_gain * e
                m = float(np.linalg.norm(shift))
                if m > cfg.adjust_max_step_m:
                    shift *= cfg.adjust_max_step_m / m
                base = self.goal if cfg.adjust_mode == "accumulate" else self.goal_orig
                self.goal = base - shift
                shift_cm = float(np.linalg.norm(shift)) * 100.0
            self.n_replans += 1
            action = "replan"
        self.events.append({
            "step": int(step),
            "err_cm": n * 100.0,
            "action": action,
            "shift_cm": shift_cm,
            "goal_offset_cm": float(np.linalg.norm(self.goal - self.goal_orig)) * 100.0,
        })
        return action

    def summary(self) -> dict[str, Any]:
        return {
            "n_replans": int(self.n_replans),
            "stayed_at_step": self.stayed_at,
            "goal_offset_cm": float(np.linalg.norm(self.goal - self.goal_orig)) * 100.0,
            "goal_offset_m": (self.goal - self.goal_orig).tolist(),
            "events": list(self.events),
        }


# ================================================================================================ bench goal
class ManifestRows(dict):
    """``{file stem: manifest row}`` plus the manifest ``path`` it was read from (error messages) and the once-per-manifest legacy-fallback flag."""

    path: str | None = None
    legacy_fallback_warned: bool = False


def load_bench_manifest(path: str | os.PathLike) -> ManifestRows:
    """``{file stem: row}`` of a BENCH_MANIFEST.json (a list of rows or ``{"clips": [...]}``); the returned mapping remembers its ``path``."""
    obj = json.loads(Path(path).read_text())
    rows = obj.get("clips") if isinstance(obj, dict) else obj
    if not isinstance(rows, list):
        raise ValueError(f"{path}: expected a list of clip rows or {{'clips': [...]}}")
    out = ManifestRows()
    out.path = str(path)
    for e in rows:
        key = str(e.get("file") or e.get("clip_id"))
        if key.endswith(".npz"):
            key = key[:-4]
        out[key] = dict(e)
    return out


def manifest_is_legacy(manifest: Mapping[str, Mapping[str, Any]]) -> bool:
    """True for a legacy manifest: no row carries a ``stratum`` field (every hero_bench_v1 row does)."""
    return not any(isinstance(v, Mapping) and v.get("stratum") is not None for v in manifest.values())


def manifest_entry(manifest: Mapping[str, dict[str, Any]], clip_name: str) -> dict[str, Any]:
    """The manifest row of a clip.  hero_bench_v1 manifests (rows with ``stratum``): the EXACT file stem, else a KeyError naming the clip and
    the manifest -- the re-timed twins ``slow_x2__<id>`` / ``hold6__<id>`` / ``fast_x0p75__<id>`` share their paper-protocol ``clip_id``, so a partial match
    would silently replan a clip toward its twin's goal / frames.  Legacy manifests (no ``stratum`` anywhere): the old fallback -- the part after
    ``__`` or the ``clip_id`` -- with ONE RuntimeWarning per manifest."""
    stem = os.path.basename(str(clip_name))
    if stem.endswith(".npz"):
        stem = stem[:-4]
    if stem in manifest:
        return manifest[stem]
    where = f" ({manifest.path})" if getattr(manifest, "path", None) else ""
    if not manifest_is_legacy(manifest):
        raise KeyError(f"clip {clip_name!r}: no manifest row with the file stem {stem!r} in the bench manifest{where} -- hero_bench_v1 rows are matched by "
                       "exact file name (re-timed twins share a clip_id, so no partial match is attempted); is this the manifest of the clip's tier?")
    tail = stem.split("__", 1)[-1]
    for k, v in manifest.items():
        if k.split("__", 1)[-1] == tail or str(v.get("clip_id")) == tail:
            if not getattr(manifest, "legacy_fallback_warned", False):
                warnings.warn(f"legacy bench manifest{where} (no stratum field): clips without an exact file-stem row are matched by their name tail / clip_id",
                              RuntimeWarning, stacklevel=2)
                try:
                    manifest.legacy_fallback_warned = True   # type: ignore[attr-defined]
                except AttributeError:   # a plain dict: the warnings module de-duplicates the (constant) message instead
                    pass
            return v
    raise KeyError(f"clip {clip_name!r} not in the bench manifest{where}")


def reach_end_frame_of(e: Mapping[str, Any]) -> int:
    """``reach_end_frame`` of a manifest row (fallback ``n_frames - hold_frames``)."""
    if e.get("reach_end_frame") is not None:
        return int(e["reach_end_frame"])
    return int(e["n_frames"]) - int(e["hold_frames"])


def hold_end_frame_of(e: Mapping[str, Any]) -> int | None:
    """``hold_end_frame`` of a manifest row (hero_bench_v1 field; fallback ``reach_end_frame + hold_frames``; None when neither is known)."""
    if e.get("hold_end_frame") is not None:
        return int(e["hold_end_frame"])
    if e.get("hold_frames") is not None and (e.get("reach_end_frame") is not None or e.get("n_frames") is not None):
        return reach_end_frame_of(e) + int(e["hold_frames"])
    return None


def clip_has_own_retract(e: Mapping[str, Any]) -> bool:
    """True when the manifest row says the clip carries its own retract segment (hero_bench_v1 retract layer: ``has_retract`` /
    ``retract_start_frame`` / a ``retract`` block / ``stratum == "retract"``)."""
    if e.get("has_retract"):
        return True
    if e.get("retract_start_frame") is not None:
        return True
    rb = e.get("retract")
    if isinstance(rb, Mapping) and rb:
        return True
    return str(e.get("stratum") or "") == "retract"


def clip_replan_config(cfg: ReplanConfig, e: Mapping[str, Any]) -> ReplanConfig:
    """The per-clip :class:`ReplanConfig` of a manifest row: a clip with its OWN retract segment gets ``stop_frame = hold_end_frame`` (replans
    stop there, the controller is handed back to the clip for its retract).  Other rows: the same ``cfg`` object (byte-identical runs)."""
    if not clip_has_own_retract(e):
        return cfg
    name = str(e.get("file") or e.get("clip_id") or "?")
    stop = e.get("retract_start_frame")
    if stop is None:
        stop = hold_end_frame_of(e)
    if stop is None:
        warnings.warn(f"{name}: retract clip without hold_end_frame / retract_start_frame -- replans are not stopped", RuntimeWarning, stacklevel=2)
        return cfg
    if cfg.stop_frame is None or int(stop) < int(cfg.stop_frame):
        return dataclasses.replace(cfg, stop_frame=int(stop))
    return cfg


def resume_reference(ref: ClipReference, k: int, name: str | None = None) -> ClipReference:
    """The reference ``ref`` resumed at frame ``k`` as a NEW reference whose frame 0 is ``ref``'s frame ``k`` (holosoma arrays through
    :meth:`ClipReference.from_arrays`; velocities by finite differences).  The hand-back target of ``ReplanConfig.stop_frame``."""
    k = ref.clamp(int(k))
    k = min(k, ref.T - 2) if ref.T >= 2 else 0   # >= 2 frames so the finite differences are defined
    jp = np.concatenate([np.asarray(ref.root_pos_from_joint_pos[k:], dtype=np.float64), xyzw_to_wxyz(np.asarray(ref.root_quat_from_joint_pos[k:], dtype=np.float64)),
                         np.asarray(ref.joint_pos[k:], dtype=np.float64)], axis=1)
    bp = np.array(ref.body_pos_w[k:], dtype=np.float64, copy=True)
    bq = xyzw_to_wxyz(np.asarray(ref.body_quat_w[k:], dtype=np.float64))
    return ClipReference.from_arrays(fps=int(ref.fps), joint_pos=jp, body_pos_w=bp, body_quat_w=bq, name=name or ref.name)


@dataclasses.dataclass(frozen=True)
class BenchGoal:
    """One benchmark goal (manifest row): world palm target, reaching hand, table slab (None = no table: floor picks), original base plan."""

    clip_name: str
    hand: str                      # "left" | "right"
    pos_w: np.ndarray              # (3,)
    quat_wxyz: np.ndarray          # (4,) palm orientation target
    table: dict[str, Any] | None   # {"center", "half_size", "surface_z"} (generator TableBox) | None (no table geometry)
    reach_end_frame: int
    pelvis_drop: float             # original plan: m below STANDING_ROOT_Z
    pelvis_pitch: float            # rad
    waist_pitch: float             # rad
    base_family: str
    height_label: str

    @property
    def hand_index(self) -> int:
        return 0 if self.hand == "left" else 1

    @property
    def active_hands(self) -> tuple[bool, bool]:
        return (self.hand == "left", self.hand == "right")

    @classmethod
    def from_manifest_entry(cls, e: Mapping[str, Any]) -> "BenchGoal":
        from data_tools import reach_specs as rs

        hand = str(e["hand"]).lower()
        if hand not in SIDES:
            raise ValueError(f"manifest hand must be left/right, got {hand!r}")
        yaw, pitch, roll = (math.radians(float(e[k])) for k in ("target_yaw_deg", "target_pitch_deg", "target_roll_deg"))
        table = e.get("table")
        if not table:
            # legacy rows (no ``table`` block) rebuild the slab from table_edge_x / table_top_z; a row with ``table: None`` and no edge is a
            # table-free goal (hero_bench_v1 floor_pick): the IK model is built without a slab
            if e.get("table_edge_x") is not None and e.get("table_top_z") is not None:
                edge_x = float(e["table_edge_x"])
                depth = float(e.get("table_depth", 0.6))
                top = float(e["table_top_z"])
                table = {"center": [edge_x + depth / 2.0, 0.5 * float(e["target_pos_w"][1]), top - 0.02], "half_size": [depth / 2.0, 0.7, 0.02], "surface_z": top}
            else:
                table = None
        reach_end = reach_end_frame_of(e)
        return cls(
            clip_name=str(e.get("file") or e.get("clip_id")),
            hand=hand,
            pos_w=np.asarray(e["target_pos_w"], dtype=np.float64).reshape(3),
            quat_wxyz=np.asarray(rs.mat_to_quat_wxyz(rs.palm_rotation(yaw, pitch, roll)), dtype=np.float64),
            table=(dict(table) if table else None),
            reach_end_frame=int(reach_end),
            pelvis_drop=float(e.get("pelvis_drop", 0.0)),
            pelvis_pitch=math.radians(float(e.get("pelvis_pitch_deg", 0.0))),
            waist_pitch=math.radians(float(e.get("waist_pitch_deg", 0.0))),
            base_family=str(e.get("base_family", "stand")),
            height_label=str(e.get("height_label") or e.get("source_tag") or ""),
        )


# ================================================================================================ geometry helpers
def _yaw_pitch_roll(R: np.ndarray) -> tuple[float, float, float]:
    """ZYX Euler angles of a rotation matrix (``R = Rz(yaw) Ry(pitch) Rx(roll)``)."""
    yaw = math.atan2(R[1, 0], R[0, 0])
    pitch = math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0]))
    roll = math.atan2(R[2, 1], R[2, 2])
    return yaw, pitch, roll


def _wrap(a: float) -> float:
    return (a + math.pi) % (2.0 * math.pi) - math.pi


def _quat_angle_wxyz(q0: np.ndarray, q1: np.ndarray) -> float:
    d = abs(float(np.dot(q0, q1)))
    return 2.0 * math.acos(min(1.0, d))


# ================================================================================================ handover smoothing
def quintic_step(u: float) -> float:
    """Minimum-jerk step ``S5(u) = 10u^3 - 15u^4 + 6u^5`` on [0, 1]: zero velocity AND acceleration at both ends (the cubic
    ``reach_specs.smoothstep`` has a non-zero acceleration at its ends)."""
    u = min(1.0, max(0.0, float(u)))
    return u * u * u * (10.0 + u * (-15.0 + 6.0 * u))


def quintic_velocity_carry(u: float) -> float:
    """Quintic Hermite basis of the START velocity: ``H1(u) = u - 6u^3 + 8u^4 - 3u^5`` -- ``H1(0) = 0, H1'(0) = 1, H1''(0) = 0``
    and ``H1(1) = H1'(1) = H1''(1) = 0``.  A path ``p0 + (p1 - p0) S5(u) + v0 T H1(u)`` (``u = t / T``) leaves ``p0`` with velocity
    ``v0`` and zero acceleration and arrives at ``p1`` at rest with zero acceleration (peak carry ``~0.2 |v0| T``)."""
    u = min(1.0, max(0.0, float(u)))
    return u * (1.0 + u * u * (-6.0 + u * (8.0 - 3.0 * u)))


def palm_path(p0: np.ndarray, q0_wxyz: np.ndarray, p1: np.ndarray, q1_wxyz: np.ndarray, n_reach: int, n_total: int, *, start_velocity: str = "zero",
              v0_w: np.ndarray | None = None, fps: int = FPS) -> tuple[np.ndarray, np.ndarray]:
    """Per-frame palm targets ``((n_total, 3), (n_total, 4) wxyz)`` of a replanned reach from ``(p0, q0)`` to ``(p1, q1)`` over
    ``n_reach`` frames, then held.  ``start_velocity="zero"``: the original cubic smoothstep from rest; ``"match"``: quintic profile
    carrying the start velocity ``v0_w`` (m/s, world) -- see :func:`quintic_velocity_carry`; the orientation slerps on the same scalar
    profile (no angular-velocity carry)."""
    from data_tools import reach_specs as rs

    if start_velocity not in START_VELOCITIES:
        raise ValueError(f"start_velocity must be one of {START_VELOCITIES}, got {start_velocity!r}")
    p0 = np.asarray(p0, dtype=np.float64).reshape(3)
    p1 = np.asarray(p1, dtype=np.float64).reshape(3)
    n_reach = max(int(n_reach), 1)
    pos = np.empty((int(n_total), 3), dtype=np.float64)
    quat = np.empty((int(n_total), 4), dtype=np.float64)
    T_s = n_reach / float(fps)
    v0 = np.zeros(3) if v0_w is None else np.asarray(v0_w, dtype=np.float64).reshape(3)
    for t in range(int(n_total)):
        u = min(t / n_reach, 1.0)
        if start_velocity == "zero":
            s = rs.smoothstep(u)
            pos[t] = p0 + (p1 - p0) * s
        else:
            s = quintic_step(u)
            pos[t] = p0 + (p1 - p0) * s + v0 * (T_s * quintic_velocity_carry(u))
        quat[t] = rs.quat_slerp_wxyz(q0_wxyz, q1_wxyz, s)
    return pos, quat


def slerp_wxyz(q0: np.ndarray, q1: np.ndarray, w: float) -> np.ndarray:
    """Vectorised slerp of wxyz quaternions ``(..., 4)`` at weight ``w`` (0 -> exactly ``q0``, 1 -> exactly ``q1``; the shorter arc is
    taken in between, nlerp below 1.8 deg like ``reach_specs.quat_slerp_wxyz``)."""
    q0 = np.asarray(q0, dtype=np.float64)
    q1 = np.asarray(q1, dtype=np.float64)
    if w <= 0.0:
        return q0.copy()
    if w >= 1.0:
        return q1.copy()
    a = q0 / np.maximum(np.linalg.norm(q0, axis=-1, keepdims=True), 1e-12)
    b = q1 / np.maximum(np.linalg.norm(q1, axis=-1, keepdims=True), 1e-12)
    d = np.sum(a * b, axis=-1, keepdims=True)
    b = np.where(d < 0.0, -b, b)
    d = np.abs(d)
    lin = (1.0 - w) * a + w * b
    lin /= np.maximum(np.linalg.norm(lin, axis=-1, keepdims=True), 1e-12)
    th = np.arccos(np.clip(d, -1.0, 1.0))
    s = np.sin(th)
    with np.errstate(divide="ignore", invalid="ignore"):
        sl = (np.sin((1.0 - w) * th) / s) * a + (np.sin(w * th) / s) * b
    return np.where(d > 0.9995, lin, sl)


def ref_palm_velocity_w(ref: ClipReference, t: int) -> np.ndarray:
    """World palm velocity ``(2, 3)`` m/s of a reference at frame ``t``: central difference of ``palm_pose_w`` over the neighbouring
    frames (one-sided at the clip ends, zero on a held / padded frame)."""
    t0, t1 = ref.clamp(int(t) - 1), ref.clamp(int(t) + 1)
    if t1 <= t0:
        return np.zeros((2, 3), dtype=np.float64)
    p0, _ = ref.palm_pose_w(t0)
    p1, _ = ref.palm_pose_w(t1)
    return (np.asarray(p1, dtype=np.float64) - np.asarray(p0, dtype=np.float64)) * (float(ref.fps) / float(t1 - t0))


@dataclasses.dataclass(frozen=True)
class Handover:
    """The reference the controller followed up to the swap, sampled at the frame it would have shown THIS step: palm poses /
    velocities (world) and joints -- the inputs of the handover knobs."""

    ref: ClipReference
    t: int
    palm_pos_w: np.ndarray       # (2, 3)
    palm_quat_wxyz: np.ndarray   # (2, 4)
    palm_vel_w: np.ndarray       # (2, 3) m/s
    dof_pos: np.ndarray          # (29,)

    @classmethod
    def at(cls, ref: ClipReference, t: int) -> "Handover":
        t = ref.clamp(int(t))
        p, q = ref.palm_pose_w(t)
        return cls(ref=ref, t=t, palm_pos_w=np.asarray(p, dtype=np.float64).copy(), palm_quat_wxyz=xyzw_to_wxyz(np.asarray(q, dtype=np.float64)).copy(),
                   palm_vel_w=ref_palm_velocity_w(ref, t), dof_pos=np.asarray(ref.joint_pos[t], dtype=np.float64).copy())

    def summary(self, hand_index: int) -> dict[str, Any]:
        return {"frame": int(self.t), "ref_name": str(self.ref.name), "palm_speed_cm_s": float(np.linalg.norm(self.palm_vel_w[hand_index])) * 100.0}


def blend_handover_frames(old_ref: ClipReference, old_t: int, joint_pos: np.ndarray, body_pos_w: np.ndarray, body_quat_wxyz: np.ndarray, n_blend: int) -> int:
    """IN PLACE crossfade of frames ``0..n_blend`` of a new reference (holosoma layout: ``joint_pos (T, 36)`` = root xyz + root quat
    WXYZ + 29 dofs, ``body_pos_w (T, 32, 3)``, ``body_quat_wxyz (T, 32, 4)``) from the OLD reference (its frame ``old_t + i`` on the
    shared grid, held past its end) to the new plan: weight ``w_i = i / n_blend`` -- frame 0 IS the old reference's frame ``old_t``,
    frame ``n_blend`` the new plan's own frame; positions / dofs linear, quaternions slerp.  Returns the number of frames touched."""
    n_blend = int(n_blend)
    if n_blend <= 0:
        return 0
    T = int(joint_pos.shape[0])
    last = min(n_blend, T - 1)
    for i in range(last + 1):
        w = i / n_blend
        to = old_ref.clamp(int(old_t) + i)
        old_root_p = np.asarray(old_ref.root_pos_from_joint_pos[to], dtype=np.float64)
        old_root_q = xyzw_to_wxyz(np.asarray(old_ref.root_quat_from_joint_pos[to], dtype=np.float64))
        old_dofs = np.asarray(old_ref.joint_pos[to], dtype=np.float64)
        joint_pos[i, 0:3] = (1.0 - w) * old_root_p + w * joint_pos[i, 0:3]
        joint_pos[i, 3:7] = slerp_wxyz(old_root_q, joint_pos[i, 3:7], w)
        joint_pos[i, 7:] = (1.0 - w) * old_dofs + w * joint_pos[i, 7:]
        body_pos_w[i] = (1.0 - w) * np.asarray(old_ref.body_pos_w[to], dtype=np.float64) + w * body_pos_w[i]
        body_quat_wxyz[i] = slerp_wxyz(xyzw_to_wxyz(np.asarray(old_ref.body_quat_w[to], dtype=np.float64)), body_quat_wxyz[i], w)
    return last + 1


# ================================================================================================ replanner
class BenchReplanner:
    """Online whole-body replanner for one bench clip (build once per clip; ``reset()`` per rollout)."""

    def __init__(self, goal: BenchGoal, cfg: ReplanConfig, *, mjcf_path: str | os.PathLike | None = None, solver_cfg: Any | None = None,
                 fk_scene_xml: str | os.PathLike | None = None, arm_posture: np.ndarray | None = None, reference: ClipReference | None = None):
        """``arm_posture`` (29,) -- joint posture prior for BOTH arms (normally the original clip's joints at its reach_end
        frame): the inactive arm is held there (cost 3.0) and the active arm is weakly (0.3) attracted to the original IK
        solution's arm configuration, so replans keep the clip's elbow configuration and the inactive arm cannot drift
        replan after replan.  ``None`` -> the robot's current arm joints are used instead.

        ``reference`` -- the reference the rollout starts on (the PADDED clip the controller follows, ``rollout_clip(ref=...)``):
        the live reference the handover knobs (``handover_blend_s`` / ``start_velocity`` / ``start_pose``) read at the first event;
        later events read the previous plan.  Required when any knob is on; ignored (bookkeeping only) otherwise."""
        from data_tools import fk_mujoco as fkm
        from data_tools import hero_reach_generator as gen

        self.gen = gen
        self.goal = goal
        self.cfg = cfg
        self.reference = reference
        if cfg.needs_live_reference and reference is None:
            raise ValueError("ReplanConfig handover knobs (handover_blend_s > 0 / start_velocity='match' / start_pose='reference') need "
                             "BenchReplanner(reference=<the rollout's ClipReference>) to know the reference the controller follows at the swap")
        self.arm_posture = None if arm_posture is None else np.asarray(arm_posture, dtype=np.float64).reshape(HC.NUM_DOF).copy()
        self.solver_cfg = solver_cfg or gen.SolverConfig()
        table = gen.TableBox.from_dict(goal.table) if goal.table else None   # None (floor pick): the IK model has no slab
        self.bundle = gen.build_model([table], Path(mjcf_path) if mjcf_path else gen.DEFAULT_MJCF)
        self.solver = gen.MinkSolver(self.bundle, self.solver_cfg)
        self.fk = fkm.get_fk(fk_scene_xml) if fk_scene_xml else fkm.get_fk()
        b = self.bundle
        self._lo = b.jnt_range[:, 0] + np.maximum(self.solver_cfg.limit_min_margin, 0.5 * (1.0 - self.solver_cfg.limit_utilization) * (b.jnt_range[:, 1] - b.jnt_range[:, 0]))
        self._hi = b.jnt_range[:, 1] - np.maximum(self.solver_cfg.limit_min_margin, 0.5 * (1.0 - self.solver_cfg.limit_utilization) * (b.jnt_range[:, 1] - b.jnt_range[:, 0]))
        self.reset()

    # ------------------------------------------------------------------------------------------ schedule
    def reset(self) -> None:
        cfg = self.cfg
        self.adjuster = GoalAdjuster(self.goal.pos_w, cfg)
        anchor = self.goal.reach_end_frame if cfg.first_event == "reach_end" else 1
        self.next_event = max(int(anchor + round(cfg.first_delay_s * FPS)), 1)
        self.plans: list[dict[str, Any]] = []
        self.ik_time_s = 0.0
        # the reference the controller follows and the step at which its frame 0 was shown (rollout: the original clip shows
        # frame k at step k -> k0 = 0; a plan returned at step k_s shows frame k - k_s afterwards)
        self.live_ref: ClipReference | None = self.reference
        self.live_k0: int = 0
        # stop_frame (hero_bench_v1 retract clips): replans stop there; the controller is handed back to the original clip once
        self.stopped_at: int | None = None
        self.handback_info: dict[str, Any] | None = None

    def live_at(self, k: int) -> tuple[ClipReference | None, int]:
        """``(reference, frame)`` the controller would follow at step ``k`` before this step's event -- mirrors the rollout's
        ``(ref_live, t_live)`` (``t_k = clamp(k)`` on the original clip, ``clamp(k - k_swap)`` on a plan).  ``(None, 0)`` without one."""
        if self.live_ref is None:
            return None, 0
        return self.live_ref, self.live_ref.clamp(int(k) - self.live_k0)

    def step(self, k: int, st: Any, plant: Any) -> ClipReference | None:
        """Called once per control step BEFORE the policy; returns a fresh reference when this step replans."""
        cfg = self.cfg
        if cfg.stop_frame is not None and k >= cfg.stop_frame:
            if self.stopped_at is None:
                self.stopped_at = int(k)
                return self.hand_back(k)
            return None
        if k < self.next_event:
            return None
        while self.next_event <= k:
            self.next_event += self.cfg.period_frames
        action = self.adjuster.decide(st.palm_pos_w[self.goal.hand_index], k)
        if action != "replan":
            return None
        old_ref, old_t = self.live_at(k) if self.cfg.needs_live_reference else (None, 0)
        t0 = time.perf_counter()
        ref, info = self.plan_from_state(st, plant, self.adjuster.goal, old_ref=old_ref, old_t=old_t)
        dt_ = time.perf_counter() - t0
        self.ik_time_s += dt_
        info.update({"step": int(k), "ik_s": dt_, "goal_offset_cm": float(np.linalg.norm(self.adjuster.goal - self.goal.pos_w)) * 100.0})
        self.plans.append(info)
        self.live_ref, self.live_k0 = ref, int(k)
        return ref

    def summary(self) -> dict[str, Any]:
        s = self.adjuster.summary()
        s.update({
            "plans": list(self.plans),
            "n_plans": len(self.plans),
            "ik_time_s": self.ik_time_s,
            "mean_terminal_ik_err_cm": float(np.mean([p["terminal_palm_err_cm"] for p in self.plans])) if self.plans else None,
            "config": self.cfg.as_dict(),
        })
        if self.cfg.stop_frame is not None:   # new key only with a stop frame (hero_bench_v1 retract clips)
            s["stop"] = {"stop_frame": int(self.cfg.stop_frame), "stopped_at_step": self.stopped_at, "handback": self.handback_info}
        return s

    # ------------------------------------------------------------------------------------------ stop frame (own-retract clips)
    def hand_back(self, k: int) -> ClipReference | None:
        """At ``stop_frame``: if the controller follows a replanned reference, return the ORIGINAL clip resumed at frame ``k``
        (:func:`resume_reference`; crossfaded from the live plan over ``handover_blend_s`` like every swap) so the clip's own retract
        segment runs; None when the controller still follows the original clip (nothing to hand back)."""
        if self.reference is None or self.live_ref is None or self.live_ref is self.reference:
            self.handback_info = {"step": int(k), "handed_back": False, "reason": "controller already on the original clip"}
            return None
        old_ref, old_t = self.live_at(k)
        ref = resume_reference(self.reference, k)
        hand_i = self.goal.hand_index
        p_old, _ = old_ref.palm_pose_w(old_t)
        p_new, _ = ref.palm_pose_w(0)
        n_blended = 0
        if self.cfg.blend_frames > 0:
            jp = np.concatenate([np.asarray(ref.root_pos_from_joint_pos, dtype=np.float64), xyzw_to_wxyz(np.asarray(ref.root_quat_from_joint_pos, dtype=np.float64)),
                                 np.asarray(ref.joint_pos, dtype=np.float64)], axis=1)
            bp = np.array(ref.body_pos_w, dtype=np.float64, copy=True)
            bq = xyzw_to_wxyz(np.asarray(ref.body_quat_w, dtype=np.float64))
            n_blended = blend_handover_frames(old_ref, old_t, jp, bp, bq, self.cfg.blend_frames)
            ref = ClipReference.from_arrays(fps=int(ref.fps), joint_pos=jp, body_pos_w=bp, body_quat_w=bq, name=ref.name)
        self.handback_info = {"step": int(k), "handed_back": True, "frames": int(ref.T), "blend_frames": int(self.cfg.blend_frames), "blended_frames": int(n_blended),
                              "palm_jump_cm": float(np.linalg.norm(np.asarray(p_new, dtype=np.float64)[hand_i] - np.asarray(p_old, dtype=np.float64)[hand_i])) * 100.0}
        self.live_ref, self.live_k0 = ref, int(k)
        return ref

    # ------------------------------------------------------------------------------------------ state ingestion
    def state_from_robot(self, st: Any, plant: Any):
        """Live robot -> generator ``ClipState`` (feet flat where they are, base / waist / legs as measured)."""
        gen, rs = self.gen, self._rs()
        R = quat_to_mat(np.asarray(st.root_quat, dtype=np.float64))
        yaw_p, pitch_p, _ = _yaw_pitch_roll(R)
        ankle_p, ankle_q = plant.body_pose_by_name([f"{s}_ankle_roll_link" for s in SIDES])
        yaws = [_yaw_pitch_roll(quat_to_mat(q))[0] for q in ankle_q]
        yaw0 = math.atan2(sum(math.sin(y) for y in yaws), sum(math.cos(y) for y in yaws))
        Rz = rs.rot_z(yaw0)
        foot_pos = {s: np.array([ankle_p[i, 0], ankle_p[i, 1], rs.SOLE_BELOW_ANKLE], dtype=np.float64) for i, s in enumerate(SIDES)}
        toe_pos = {s: foot_pos[s] + Rz @ np.asarray(rs.TOE_LOCAL, dtype=np.float64) for s in SIDES}
        center = 0.5 * (foot_pos["left"][:2] + foot_pos["right"][:2])
        root = np.asarray(st.root_pos, dtype=np.float64)
        shift = Rz[:2, :2].T @ (root[:2] - center)
        q = np.asarray(st.dof_pos, dtype=np.float64)
        leg = {s: {"hip_pitch": float(q[i[0]]), "knee": float(q[i[1]]), "ankle_pitch": float(q[i[2]])} for s, i in _LEG_IDX.items()}
        return gen.ClipState(yaw0=yaw0, foot_pos=foot_pos, toe_pos=toe_pos, stance_center=center, pelvis_height=float(root[2]),
                             pelvis_pitch=float(pitch_p), pelvis_yaw_delta=_wrap(yaw_p - yaw0), pelvis_shift=shift,
                             waist_pitch=float(q[_WAIST_PITCH]), waist_yaw=float(q[_WAIST_YAW]), leg_prior=leg,
                             foot_mode={s: "flat" for s in SIDES}, heel_pitch={s: 0.0 for s in SIDES}, kneel_side="")

    def base_for(self, state):
        """End state of the base: the original plan (``clip``) or the current pose (``current``)."""
        rs = self._rs()
        g = self.goal
        if self.cfg.base_mode == "clip":
            return rs.BaseStrategy(family=g.base_family, drop_mode="replan", pelvis_drop=g.pelvis_drop, pelvis_pitch=g.pelvis_pitch,
                                   waist_pitch=g.waist_pitch, waist_pitch_fraction=0.6, waist_yaw=0.0, pelvis_shift=(0.0, 0.0), pelvis_yaw_delta=0.0,
                                   foot_mode="flat", heel_lift_pitch=0.0, kneel_side="", leg_prior=None, base_timing="sync", forced_variant=False)
        return rs.BaseStrategy(family="current", drop_mode="replan", pelvis_drop=float(rs.STANDING_ROOT_Z - state.pelvis_height), pelvis_pitch=state.pelvis_pitch,
                               waist_pitch=state.waist_pitch, waist_pitch_fraction=0.6, waist_yaw=state.waist_yaw, pelvis_shift=tuple(float(v) for v in state.pelvis_shift),
                               pelvis_yaw_delta=state.pelvis_yaw_delta, foot_mode="flat", heel_lift_pitch=0.0, kneel_side="", leg_prior=None,
                               base_timing="sync", forced_variant=False)

    def q_start_from_robot(self, st: Any) -> np.ndarray:
        b = self.bundle
        q = np.zeros(b.nq, dtype=np.float64)
        q[0:3] = np.asarray(st.root_pos, dtype=np.float64)
        q[3:7] = xyzw_to_wxyz(np.asarray(st.root_quat, dtype=np.float64))
        q[3:7] /= np.linalg.norm(q[3:7])
        q[b.dof_qpos_idx] = np.clip(np.asarray(st.dof_pos, dtype=np.float64), self._lo, self._hi)  # inside the QP's 98 % limits
        return q

    # ------------------------------------------------------------------------------------------ planning
    def build_plan(self, state, base, q_start: np.ndarray, goal_pos_w: np.ndarray, goal_quat_wxyz: np.ndarray, handover: Handover | None = None,
                   p0_robot: np.ndarray | None = None):
        """Straight palm path from the current palm pose to the goal; base / posture blend to ``base``; feet flat.

        ``handover`` (the old reference at the swap instant) enables the handover knobs: ``start_pose="reference"`` starts the
        palm path at the old reference's palm pose instead of the robot's FK, ``start_velocity="match"`` carries its palm velocity
        (:func:`palm_path`).  ``None`` -> the plain plan whatever the config says.  ``p0_robot`` -- the ROBOT's palm position
        (generator FK of the unseeded start configuration) for the ``start_pose_offset_cm`` diagnostic; ``None`` -> the FK palm
        of ``q_start``."""
        gen, rs, cfg, scfg = self.gen, self._rs(), self.cfg, self.solver_cfg
        active = self.goal.active_hands
        hand_i = self.goal.hand_index
        palms, _shoulders = gen._palm_start(self.bundle, q_start)
        p0, q0 = palms[self.goal.hand]
        p0_robot = np.asarray(p0 if p0_robot is None else p0_robot, dtype=np.float64).copy()
        start_velocity, v0 = "zero", None
        if handover is not None and cfg.start_pose == "reference":
            p0, q0 = handover.palm_pos_w[hand_i].copy(), handover.palm_quat_wxyz[hand_i].copy()
        if handover is not None and cfg.start_velocity == "match":
            start_velocity, v0 = "match", handover.palm_vel_w[hand_i].copy()
        goal_pos_w = np.asarray(goal_pos_w, dtype=np.float64)
        dist = float(np.linalg.norm(goal_pos_w - p0))
        ang = _quat_angle_wxyz(q0, goal_quat_wxyz)
        if cfg.base_mode == "clip":
            leg_new = {s: gen._leg_prior_for(base, s, state) for s in SIDES}
        else:
            leg_new = {s: dict(state.leg_prior[s]) for s in SIDES}
        h_new, pitch_new, ydel_new = base.pelvis_height, base.pelvis_pitch, base.pelvis_yaw_delta
        shift_new = np.asarray(base.pelvis_shift, dtype=np.float64)
        travel = 4.0 * abs(h_new - state.pelvis_height) + abs(pitch_new - state.pelvis_pitch) + abs(base.waist_pitch - state.waist_pitch) + abs(base.waist_yaw - state.waist_yaw)
        T_reach = float(np.clip(0.4 + 1.2 * dist + 0.5 * ang / math.pi + 0.35 * travel, cfg.reach_min_s, cfg.reach_max_s))
        n_reach = max(int(round(T_reach * FPS)), 2)
        n_total = n_reach + int(cfg.hold_solved_frames)
        t_idx = np.arange(n_total, dtype=np.float64)
        s_base = np.array([rs.smoothstep(min(t / n_reach, 1.0)) for t in t_idx])
        Rz = rs.rot_z(state.yaw0)
        pelvis_pos = np.empty((n_total, 3))
        pelvis_rpy = np.empty((n_total, 3))
        posture_q = np.empty((n_total, 29))
        dofs0 = q_start[self.bundle.dof_qpos_idx]
        q_from = gen._posture_from(state.leg_prior, state.waist_pitch, state.waist_yaw, state.pelvis_height, state.pelvis_pitch)
        q_to = gen._posture_from(leg_new, base.waist_pitch, base.waist_yaw, h_new, pitch_new)
        arms = list(HC.ARM_DOF_IDX)
        q_from[arms] = dofs0[arms]       # start where the arms are ...
        q_to[arms] = dofs0[arms] if self.arm_posture is None else self.arm_posture[arms]   # ... end at the clip's arm posture (inactive arm held there)
        if cfg.base_mode == "current":   # legs: exactly the measured joints, not the closed-form box
            for s, idx in _LEG_IDX.items():
                for j, key in zip(idx, ("hip_pitch", "knee", "ankle_pitch")):
                    q_from[j] = q_to[j] = state.leg_prior[s][key]
        for t in range(n_total):
            s = float(s_base[t])
            h = state.pelvis_height + (h_new - state.pelvis_height) * s
            pitch = state.pelvis_pitch + (pitch_new - state.pelvis_pitch) * s
            ydel = state.pelvis_yaw_delta + (ydel_new - state.pelvis_yaw_delta) * s
            shift = state.pelvis_shift + (shift_new - state.pelvis_shift) * s
            xy = state.stance_center + Rz[:2, :2] @ shift
            pelvis_pos[t] = [xy[0], xy[1], h]
            pelvis_rpy[t] = [0.0, pitch, state.yaw0 + ydel]
            posture_q[t] = q_from + (q_to - q_from) * s
        posture_cost = np.repeat(gen._posture_cost(scfg, active, "")[None], n_total, axis=0)
        palm_pos = np.empty((n_total, 2, 3))
        palm_quat = np.empty((n_total, 2, 4))
        palm_cost = np.zeros((n_total, 2))
        for i, side in enumerate(SIDES):
            pi, qi = palms[side]
            palm_pos[:, i] = pi
            palm_quat[:, i] = qi
        path_pos, path_quat = palm_path(p0, q0, goal_pos_w, goal_quat_wxyz, n_reach, n_total, start_velocity=start_velocity, v0_w=v0)
        palm_pos[:, hand_i] = path_pos
        palm_quat[:, hand_i] = path_quat
        palm_cost[:, hand_i] = scfg.palm_pos_cost
        feet = gen._foot_plans(state, base, n_reach, n_total, s_base, False)
        flat_pts = {s: gen._flat_sole_points(state.foot_pos[s], state.yaw0) for s in SIDES}
        support = [gen._support_points(feet, t, flat_pts) for t in range(n_total)]
        tp = np.repeat(palm_pos[-1][None], n_total, axis=0)
        tq = np.repeat(palm_quat[-1][None], n_total, axis=0)
        tp[:, hand_i] = goal_pos_w
        tq[:, hand_i] = goal_quat_wxyz
        plan = gen.SegmentPlan(frames=n_total, reach_frames=n_reach, hold_frames=int(cfg.hold_solved_frames), palm_pos=palm_pos, palm_quat=palm_quat,
                               palm_cost=palm_cost, pelvis_pos=pelvis_pos, pelvis_rpy=pelvis_rpy, posture_q=posture_q, posture_cost=posture_cost, feet=feet,
                               target_pos_frame=tp, target_quat_frame=tq, target_final_pos=tp[-1].copy(), target_final_quat=tq[-1].copy(),
                               retarget_frame=-1, kind="reach", support_points=support)
        info = {"dist_cm": dist * 100.0, "ang_deg": math.degrees(ang), "n_reach": n_reach, "reach_s": n_reach / FPS}
        if handover is not None:
            info.update({
                "start_pose": cfg.start_pose, "start_velocity": start_velocity,
                "start_pose_offset_cm": float(np.linalg.norm(np.asarray(p0, dtype=np.float64) - p0_robot)) * 100.0,  # old reference palm vs robot palm
                "start_speed_cm_s": (float(np.linalg.norm(v0)) * 100.0 if v0 is not None else 0.0),
            })
        return plan, info

    def plan_from_state(self, st: Any, plant: Any, goal_pos_w: np.ndarray, *, old_ref: ClipReference | None = None, old_t: int = 0) -> tuple[ClipReference, dict[str, Any]]:
        """Solve a fresh whole-body reference from the live robot state to ``goal_pos_w`` (palm orientation = bench goal).

        ``old_ref`` / ``old_t`` -- the reference the controller followed and the frame it would show this step (``live_at``):
        the handover knobs of the config act on it (``start_pose`` / ``start_velocity`` shape the plan, ``handover_blend_s``
        crossfades the first frames of the solved reference from it).  ``None`` (or a config without knobs) -> the plain plan
        whose frame 0 is the robot's own state."""
        b, cfg = self.bundle, self.cfg
        state = self.state_from_robot(st, plant)
        base = self.base_for(state)
        q_start = self.q_start_from_robot(st)
        handover = Handover.at(old_ref, old_t) if (old_ref is not None and cfg.needs_live_reference) else None
        p0_robot = None
        if handover is not None and cfg.start_pose == "reference":  # seed the IK with the old reference's arms (frame 0 -> its palm pose, not the robot's)
            p0_robot = self.gen._palm_start(b, q_start)[0][self.goal.hand][0].copy()   # the ROBOT's palm (before the seed) for start_pose_offset_cm
            arms = list(HC.ARM_DOF_IDX)
            q_start[np.asarray(b.dof_qpos_idx)[arms]] = np.clip(handover.dof_pos[arms], self._lo[arms], self._hi[arms])
        plan, info = self.build_plan(state, base, q_start, goal_pos_w, self.goal.quat_wxyz, handover=handover, p0_robot=p0_robot)
        self._prime_tasks(q_start)
        if handover is not None and cfg.start_pose == "reference":
            # the old reference's arm joints only reproduce its palm pose with ITS base / waist; the robot's differ, and the QP's per-frame
            # velocity cap (0.08 rad) would leave frame 0 short of the old palm -> converge the start configuration onto the frame-0 targets first
            q_start, settle_err = self._settle_start(plan, q_start)
            info["settle_palm_err_cm"] = settle_err * 100.0
            self._prime_tasks(q_start)
        qs, resid = self.solver.solve_plan(plan, 0, q_start, self.goal.active_hands, q_before_start=None)
        jp = np.concatenate([qs[:, :7], qs[:, b.dof_qpos_idx]], axis=1)
        n_pad = max(int(round(cfg.hold_s * FPS)) - int(cfg.hold_solved_frames), 0)
        if n_pad:
            jp = np.concatenate([jp, np.repeat(jp[-1:], n_pad, axis=0)], axis=0)
        fk = self.fk.fk_all(jp)
        body_pos, body_quat = fk["body_pos_w"], fk["body_quat_w"]
        term_err = float(np.linalg.norm(fk["palm_pos_w"][-1, self.goal.hand_index] - np.asarray(goal_pos_w)))
        pelvis_z_start = float(jp[0, 2])   # the PLAN's start (IK frame 0) -- read before the crossfade rewrites frame 0 to the old reference
        n_blended = 0
        if handover is not None and cfg.blend_frames > 0:  # crossfade baked into the first frames (the plan's own frames beyond)
            body_pos, body_quat = np.array(body_pos, dtype=np.float64, copy=True), np.array(body_quat, dtype=np.float64, copy=True)
            n_blended = blend_handover_frames(handover.ref, handover.t, jp, body_pos, body_quat, cfg.blend_frames)
        ref = ClipReference.from_arrays(fps=FPS, joint_pos=jp, body_pos_w=body_pos, body_quat_w=body_quat, name=self.goal.clip_name)
        info.update({
            "frames": int(jp.shape[0]),
            "terminal_palm_err_cm": term_err * 100.0,
            "max_palm_path_err_cm": float(resid[: plan.reach_frames, 0].max()) * 100.0 if plan.reach_frames else 0.0,
            "qp_failures": int(self.solver.qp_failures),
            "pelvis_z_start": pelvis_z_start,
            "pelvis_z_end": float(jp[-1, 2]),
        })
        if handover is not None:   # blend_frames = the config's n_blend; blended_frames = frames rewritten (0..n_blend inclusive)
            info.update({"blend_frames": int(cfg.blend_frames), "blended_frames": int(n_blended), "handover": handover.summary(self.goal.hand_index)})
        return ref, info

    def _settle_start(self, plan, q_start: np.ndarray, n_settle: int = 10) -> tuple[np.ndarray, float]:
        """Solve the plan's FRAME 0 targets repeated ``n_settle`` times from ``q_start`` (the QP's velocity limiter allows 0.08 rad per
        frame) and return the last configuration + its palm position residual (m): the IK start for ``start_pose="reference"``,
        whose palm sits at the old reference's palm while base / feet are the robot's."""
        gen = self.gen

        def rep(a):
            return np.repeat(np.asarray(a)[:1], n_settle, axis=0)

        feet = {s: dataclasses.replace(fp, pivot_pos=rep(fp.pivot_pos), foot_rot=rep(fp.foot_rot), pitch=rep(fp.pitch), knee_pos=(rep(fp.knee_pos) if fp.knee_pos is not None else None),
                                       contact=rep(fp.contact), knee_cost=rep(fp.knee_cost), pivot_cost=rep(fp.pivot_cost), ori_cost=rep(fp.ori_cost)) for s, fp in plan.feet.items()}
        settle = gen.SegmentPlan(frames=n_settle, reach_frames=n_settle, hold_frames=0, palm_pos=rep(plan.palm_pos), palm_quat=rep(plan.palm_quat), palm_cost=rep(plan.palm_cost),
                                 pelvis_pos=rep(plan.pelvis_pos), pelvis_rpy=rep(plan.pelvis_rpy), posture_q=rep(plan.posture_q), posture_cost=rep(plan.posture_cost), feet=feet,
                                 target_pos_frame=rep(plan.target_pos_frame), target_quat_frame=rep(plan.target_quat_frame), target_final_pos=plan.target_pos_frame[0].copy(),
                                 target_final_quat=plan.target_quat_frame[0].copy(), retarget_frame=-1, kind="reach", support_points=[plan.support_points[0]] * n_settle)
        qs, resid = self.solver.solve_plan(settle, 0, q_start, self.goal.active_hands, q_before_start=None)
        return qs[-1].copy(), float(resid[-1, 0])

    def _prime_tasks(self, q_start: np.ndarray) -> None:
        """Every FrameTask needs a target before mink evaluates it (``generate_clip`` gets that from
        ``solve_initial_stance``); zero-cost tasks (inactive palm, knees, toes) are primed from the start configuration."""
        sol = self.solver
        sol.q.update(q_start.copy())
        for task in (sol.pelvis, *sol.palm.values(), *sol.ankle.values(), *sol.toe.values(), *sol.knee.values()):
            task.set_target_from_configuration(sol.q)
        sol.posture.set_target(q_start.copy())
        sol.prev.set_target(q_start.copy())

    # ------------------------------------------------------------------------------------------ misc
    @staticmethod
    def _rs():
        from data_tools import reach_specs as rs

        return rs


__all__ = ["ADJUST_MODES", "BASE_MODES", "FIRST_EVENTS", "FPS", "SIDES", "START_POSES", "START_VELOCITIES", "BenchGoal", "BenchReplanner", "GoalAdjuster", "Handover", "ReplanConfig",
           "blend_handover_frames", "clip_has_own_retract", "clip_replan_config", "hold_end_frame_of", "ManifestRows", "load_bench_manifest", "manifest_entry", "manifest_is_legacy", "palm_path", "quintic_step",
           "quintic_velocity_carry", "reach_end_frame_of", "ref_palm_velocity_w", "resume_reference", "slerp_wxyz"]
