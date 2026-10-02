"""Time-stretched copy of a HERO reaching bench (``hero_bench_v1`` layout): slower (or faster) reaches, same goals.

Why: the bench reach is fast (~2-3 s), so a policy's world-frame error at the goal is dominated by tracking lag and the closed-loop
replanner only acts at the end.  Re-timed copies -- the same goals reached slower or faster, or held longer -- separate the speed and
hold axes of the protocol while the evaluators keep consuming ``<dir>/*.npz`` + ``BENCH_MANIFEST.json``; the hero_bench_v1 strata
``slow_x2`` / ``hold6`` / ``fast_x0p75`` are built this way (``data_tools.build_hero_bench``).

    python -m data_tools.retime_bench --src data/corpus/hero_bench_v1 --out data/corpus/hero_bench_v1_x2 --factor 2.0 [--hold-s keep|<s>]

What is written (per clip, the same 32-body holosoma / HERO npz layout the training corpus uses, see ``data_tools.schema``):

* time axis: new frame count ``N = round(F x T)``; new frame ``j`` samples the source at the fractional frame
  ``u_j = j (T-1)/(N-1)`` (endpoint-preserving: frame 0 and the last frame -- the settled goal pose -- are the source frames; the
  effective stretch ``(N-1)/(T-1)`` is recorded).  ``fps`` stays 50;
* positions (``joint_pos`` root xyz + 29 dof, ``body_pos_w``, ``ee_pos_pelvis*``, ``h_ref``, ``object_pos_w``) -- linear interpolation;
  quaternions (``joint_pos`` root quat, ``body_quat_w``, ``ee_quat_pelvis*``, ``object_quat_w``; all wxyz) -- shortest-arc slerp
  between the two neighbouring source frames, source values taken verbatim when ``u_j`` is an integer;
* EVERY velocity array is recomputed from the resampled signals with the corpus converter's own estimators
  (``fk_mujoco.finite_difference`` = central differences, one-sided at the ends; ``fk_mujoco.angular_velocity_world`` = the world-frame
  increment ``q_{k+1} q_{k-1}^-1`` over ``2/fps``, ``q_{k+1} q_k^-1`` at the ends): ``joint_vel`` = [root lin vel | root ang vel |
  dof vel], ``body_lin_vel_w`` / ``body_ang_vel_w``, ``object_lin_vel_w`` / ``object_ang_vel_w`` (``boxcarry_gen`` clips).  Nothing is
  divided by the factor.  The frame of the root angular velocity (``joint_vel[:, 3:6]``) is auto-detected from the source clip (the IK
  banks store it in the WORLD frame, the corpus converter's recompute path in the pelvis frame; both are stripped by the loaders) so
  ``--factor 1.0`` reproduces the source.  An unknown per-frame float key whose name looks like a velocity / acceleration
  (``VELOCITY_NAME_PATTERN``) is REFUSED (``ValueError``): interpolating it would keep the source speed on the slowed clip;
* contact / flag / id arrays and any integer / bool / string per-frame key -- nearest-neighbour;
* per-clip keys (``fps``, names, ``source_tag``, ``parent_id``, ``license_class``, ``has_object``, ``box_size``) -- copied;
* ``--hold-s keep`` (default) stretches the hold phase with the rest; ``--hold-s <s>`` stretches only ``[0, reach_end_frame]`` by the
  factor and gives the hold a fixed duration (the hold is a static pose, so its own time axis is resampled to ``round(s x fps)`` frames).

``BENCH_MANIFEST.json``: every clip row keeps its goal pose / labels; the frame indices (``settle_frames``, ``reach_start_frame``,
``reach_end_frame``, ``reach_frames``, ``hold_frames``, ``n_frames``, ``duration_s``) are mapped through the same time warp, rounded
towards the phase they name (``MANIFEST_INDEX_SIDES``): ``reach_end_frame`` = the generator's FIRST HOLD FRAME maps with ceil to the
first new frame whose source sample is at / after it (so the pose there IS the goal; a plain round would land inside the last reach
step for most real clips, e.g. 266 frames / reach_end 116 at x2: exact 232.44 -> 233, not 232), which is what ``sim2sim.replan``
anchors its first event on, ``run_hero_mujoco`` takes the arm posture from and the summaries' hold windows start at;
``settle_frames`` / ``reach_start_frame`` = the reach's t=0 frame (the last frame still at the start pose) map with floor.
``reach_frames`` / ``hold_frames`` follow from the mapped indices.  ``sha256`` / ``bytes`` / ``max_qdot_rad_s`` / ``h_ref_*`` are
recomputed; a ``retime`` block records the mapping.  Top level: ``per_height`` duration statistics, ``verification.clip_lengths`` and
``protocol.timing`` (``time_scale``, ``hold_s``) follow; a ``retime`` block points at the source.  ``RETIME_RECEIPT.json``: factor,
source dir / manifest sha, per-clip source sha and old / new frame counts and indices, ``ok`` (verification passed).

Optional (hero_bench_v1 re-timed strata; all default to off, which keeps the behaviour above byte for byte): ``gate`` = ``{"max_qdot_rad_s":
8.0, "max_qddot_rad_s2": 100.0}`` re-measures the dof speed / acceleration of every RETIMED clip (central differences of ``joint_pos[:, 7:]`` at
``fps``, the generator's definition) and DROPS clips over either bound (not written, no manifest row; listed in the receipt's ``gate_dropped``
with the measured values -- the per-clip receipts always carry ``max_qdot_rad_s`` / ``max_qddot_rad_s2``); ``rename`` maps a source file name
to the output file name (e.g. ``h050__x.npz`` -> ``slow_x2__x.npz``; the manifest row's ``file`` follows, ``retime.src_file`` keeps the source
name); ``source_tag`` overrides the per-clip ``source_tag`` key of the output (the evaluators group by the file-name prefix, the loaders by
this key -- keep them equal).

Guards: ``--factor 1.0`` reproduces the source arrays to float32 rounding (test); every output is checked with
``data_tools.schema.validate_holosoma_npz`` and, when ``sim2sim`` is importable, loaded through ``sim2sim.reference.ClipReference``
(``--no-verify`` skips the loader pass).  When verification FAILS the manifest / receipt are written as ``BENCH_MANIFEST.FAILED.json``
/ ``RETIME_RECEIPT.FAILED.json`` (``ok: false``) and the tool exits non-zero, so a broken bench never carries the canonical file
names that the evaluators (``<motion dir>/BENCH_MANIFEST.json``) and the hero_bench_v1 builder (receipt) key on.

Run tests: ``python -m pytest tests/test_hero_bench_build.py -q`` from the release root.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import os
import platform
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from data_tools import fk_mujoco as fkm
from data_tools.schema import EXTENSION_KEYS, REQUIRED_KEYS, read_npz, scalar_fps, validate_holosoma_npz

TOOL_VERSION = "1.2.0"   # 1.2.0: optional gate / rename / source_tag (hero_bench_v1 re-timed strata); defaults byte-identical to 1.1.0
RECEIPT_NAME = "RETIME_RECEIPT.json"
MANIFEST_NAME = "BENCH_MANIFEST.json"
#: names used INSTEAD of the canonical ones when the output fails verification (downstream keys on the canonical names)
RECEIPT_FAILED_NAME = "RETIME_RECEIPT.FAILED.json"
MANIFEST_FAILED_NAME = "BENCH_MANIFEST.FAILED.json"
RECEIPT_SCHEMA = "hero_bench_retime_receipt_v1"
REPO_ROOT = Path(__file__).resolve().parents[1]

#: keys that are never per-frame (copied verbatim even when a dimension happens to equal T)
PER_CLIP_KEYS: frozenset[str] = frozenset({"fps", "body_names", "joint_names", "source_tag", "parent_id", "license_class", "has_object", "box_size"})
#: per-frame wxyz quaternion arrays (slerp); ``joint_pos[:, 3:7]`` is handled inside the joint_pos rule
QUAT_KEYS: frozenset[str] = frozenset({"body_quat_w", "ee_quat_pelvis", "ee_quat_pelvis_zero_waist", "object_quat_w"})
#: per-frame linear-interpolation keys
LINEAR_KEYS: frozenset[str] = frozenset({"body_pos_w", "ee_pos_pelvis", "ee_pos_pelvis_zero_waist", "h_ref", "object_pos_w"})
#: velocity arrays and the resampled signal they are recomputed from (``object_ang_vel_w``: boxcarry_gen clips, world frame like the
#: generator's ``fkm.angular_velocity_world(object_quat_w)``)
VELOCITY_KEYS: dict[str, str] = {"joint_vel": "joint_pos", "body_lin_vel_w": "body_pos_w", "body_ang_vel_w": "body_quat_w",
                                 "object_lin_vel_w": "object_pos_w", "object_ang_vel_w": "object_quat_w"}
#: unknown float per-frame keys whose name carries one of these are nearest-neighbour resampled (flags / ids / contacts)
NEAREST_NAME_HINTS: tuple[str, ...] = ("contact", "flag", "phase", "mask", "active", "valid", "_id", "label")
#: an unknown per-frame FLOAT key matching this is a velocity / acceleration the tool cannot recompute -> refused (never interpolated)
VELOCITY_NAME_PATTERN = re.compile(r"vel|velocity|omega|qd|acc")
ROOT_ANG_FRAMES: tuple[str, ...] = ("auto", "world", "pelvis")
#: manifest frame-index fields mapped through the warp and the side they round to: ``reach_end_frame`` is the FIRST hold frame (ceil:
#: the first new frame at / after the source boundary, so its pose is the goal); ``settle_frames`` == ``reach_start_frame`` is the
#: reach's t=0 frame, the LAST frame at the start pose (floor)
MANIFEST_INDEX_SIDES: dict[str, str] = {"settle_frames": "floor", "reach_start_frame": "floor", "reach_end_frame": "ceil"}
MANIFEST_INDEX_FIELDS: tuple[str, ...] = tuple(MANIFEST_INDEX_SIDES)
INDEX_SIDES: tuple[str, ...] = ("round", "floor", "ceil")


# ================================================================================================ time warp
@dataclasses.dataclass(frozen=True)
class TimeWarp:
    """Monotone map new frame -> fractional source frame, piecewise linear over ``segments`` ``(old_a, old_b, new_a, new_b)``."""

    old_frames: int
    new_frames: int
    old_index: np.ndarray                      # (new_frames,) fractional source frame of every new frame
    segments: tuple[tuple[int, int, int, int], ...]
    mode: str                                  # "uniform" | "hold_fixed"

    @property
    def factor_effective(self) -> float:
        """Overall stretch of the clip duration ``(N-1)/(T-1)``."""
        return float(self.new_frames - 1) / float(max(self.old_frames - 1, 1))

    def map_index_exact(self, k: int) -> float:
        """Source frame index -> the (fractional) new frame position sampling it exactly (segment boundaries are integers)."""
        k = int(k)
        if k <= 0:
            return 0.0
        if k >= self.old_frames - 1:
            return float(self.new_frames - 1 + (k - (self.old_frames - 1)))  # beyond the end: keep the offset (n_frames -> N)
        for old_a, old_b, new_a, new_b in self.segments:
            if old_a <= k <= old_b:
                if old_b == old_a:
                    return float(new_a)
                return float(new_a + (k - old_a) * (new_b - new_a) / float(old_b - old_a))
        raise ValueError(f"frame {k} outside the warp segments {self.segments}")

    def map_index(self, k: int, side: str = "round") -> int:
        """Source frame index -> a new frame index: ``round`` (nearest), ``ceil`` (the first new frame whose source sample is at / after
        ``k`` -- use for the FIRST frame of a phase, e.g. ``reach_end_frame``) or ``floor`` (the last new frame at / before ``k`` -- the
        last frame of a phase, e.g. ``reach_start_frame`` = reach t=0).  Segment boundaries and ``factor 1.0`` map exactly for all sides."""
        if side not in INDEX_SIDES:
            raise ValueError(f"side must be one of {INDEX_SIDES}, got {side!r}")
        exact = self.map_index_exact(k)
        if side == "ceil":
            return int(np.ceil(exact - 1e-9))
        if side == "floor":
            return int(np.floor(exact + 1e-9))
        return int(np.floor(exact + 0.5))


def _segment_index(old_a: int, old_b: int, new_a: int, new_b: int) -> np.ndarray:
    n = new_b - new_a + 1
    if n <= 1 or old_b == old_a:
        return np.full(n, float(old_a), dtype=np.float64)
    return old_a + np.arange(n, dtype=np.float64) * (old_b - old_a) / float(new_b - new_a)


def build_warp(old_frames: int, factor: float, *, reach_end_frame: int | None = None, hold_frames_new: int | None = None) -> TimeWarp:
    """``hold_frames_new is None`` -> uniform stretch (``N = round(F T)``); else ``[0, reach_end]`` is stretched by ``F`` and the hold
    ``[reach_end, T-1]`` is resampled onto ``hold_frames_new`` frames (frames ``[R_new, N)`` hold the goal)."""
    T = int(old_frames)
    if T < 2:
        raise ValueError(f"need >= 2 frames, got {T}")
    if not np.isfinite(factor) or factor <= 0.0:
        raise ValueError(f"factor must be positive, got {factor!r}")
    if hold_frames_new is None or reach_end_frame is None:
        N = int(np.floor(factor * T + 0.5))
        if N < 2:
            raise ValueError(f"factor {factor} on {T} frames gives {N} frame(s)")
        seg = (0, T - 1, 0, N - 1)
        return TimeWarp(T, N, _segment_index(*seg), (seg,), "uniform")
    re_ = int(reach_end_frame)
    if not 1 <= re_ <= T - 1:
        raise ValueError(f"reach_end_frame {re_} outside [1, {T - 1}]")
    H = int(hold_frames_new)
    if H < 1:
        raise ValueError(f"hold_frames_new must be >= 1, got {H}")
    R = max(int(np.floor(factor * re_ + 0.5)), 1)      # first hold frame of the new clip
    N = R + H
    seg1 = (0, re_, 0, R)
    seg2 = (re_, T - 1, R, N - 1)
    idx = np.concatenate([_segment_index(*seg1), _segment_index(*seg2)[1:]])
    assert idx.shape[0] == N, (idx.shape, N)
    return TimeWarp(T, N, idx, (seg1, seg2), "hold_fixed")


# ================================================================================================ resampling
def _neighbours(old_index: np.ndarray, T: int) -> tuple[np.ndarray, np.ndarray]:
    u = np.clip(np.asarray(old_index, dtype=np.float64), 0.0, float(T - 1))
    i0 = np.clip(np.floor(u).astype(np.int64), 0, max(T - 2, 0))
    frac = u - i0
    return i0, frac


def resample_linear(x: np.ndarray, old_index: np.ndarray) -> np.ndarray:
    """Piecewise-linear resampling of ``x`` (T, ...) at fractional source frames (exact at integer indices)."""
    x = np.asarray(x, dtype=np.float64)
    T = x.shape[0]
    if T == 1:
        return np.repeat(x, old_index.shape[0], axis=0)
    i0, frac = _neighbours(old_index, T)
    f = frac.reshape((-1,) + (1,) * (x.ndim - 1))
    return x[i0] * (1.0 - f) + x[i0 + 1] * f


def resample_quat(q: np.ndarray, old_index: np.ndarray) -> np.ndarray:
    """Shortest-arc slerp of wxyz quaternions ``q`` (T, ..., 4) at fractional source frames.  Integer indices return the source
    quaternion verbatim (sign and rounding preserved); interior samples take the sign of the earlier neighbour."""
    q = np.asarray(q, dtype=np.float64)
    T = q.shape[0]
    if T == 1:
        return np.repeat(q, old_index.shape[0], axis=0)
    i0, frac = _neighbours(old_index, T)
    q0 = q[i0]
    q1 = q[i0 + 1]
    flip = np.sum(q0 * q1, axis=-1, keepdims=True) < 0.0
    q1s = np.where(flip, -q1, q1)
    f = frac.reshape((-1,) + (1,) * (q.ndim - 2))
    out = fkm.quat_slerp(q0, q1s, np.broadcast_to(f, q0.shape[:-1]))
    at0 = (frac == 0.0).reshape((-1,) + (1,) * (q.ndim - 1))
    at1 = (frac == 1.0).reshape((-1,) + (1,) * (q.ndim - 1))
    return np.where(at0, q0, np.where(at1, q1, out))


def resample_nearest(x: np.ndarray, old_index: np.ndarray) -> np.ndarray:
    x = np.asarray(x)
    T = x.shape[0]
    idx = np.clip(np.floor(np.asarray(old_index, dtype=np.float64) + 0.5).astype(np.int64), 0, T - 1)
    return x[idx]


# ================================================================================================ velocities
def detect_root_ang_frame(joint_pos: np.ndarray, joint_vel: np.ndarray, fps: float) -> str:
    """``world`` or ``pelvis``: whichever expression of the root angular velocity finite-differenced from ``joint_pos[:, 3:7]`` is closer
    to the stored ``joint_vel[:, 3:6]`` (ties, e.g. no root rotation -> ``world``, the IK-bank convention)."""
    jp = np.asarray(joint_pos, dtype=np.float64)
    jv = np.asarray(joint_vel, dtype=np.float64)
    ang_w = fkm.angular_velocity_world(jp[:, 3:7], fps)
    ang_b = fkm.quat_rotate_inverse(jp[:, 3:7], ang_w)
    err_w = float(np.abs(ang_w - jv[:, 3:6]).max())
    err_b = float(np.abs(ang_b - jv[:, 3:6]).max())
    return "world" if err_w <= err_b else "pelvis"


def joint_vel_from_joint_pos(joint_pos: np.ndarray, fps: float, root_ang_frame: str = "world") -> np.ndarray:
    """(T, 35) = [root lin vel (world) | root ang vel (``root_ang_frame``) | dof vel], central differences of ``joint_pos``."""
    if root_ang_frame not in ("world", "pelvis"):
        raise ValueError(f"root_ang_frame must be world|pelvis, got {root_ang_frame!r}")
    jp = np.asarray(joint_pos, dtype=np.float64)
    lin = fkm.finite_difference(jp[:, :3], fps)
    ang = fkm.angular_velocity_world(jp[:, 3:7], fps)
    if root_ang_frame == "pelvis":
        ang = fkm.quat_rotate_inverse(jp[:, 3:7], ang)
    dof = fkm.finite_difference(jp[:, 7:], fps)
    return np.concatenate([lin, ang, dof], axis=1)


def recompute_velocities(arrays: Mapping[str, np.ndarray], fps: float, root_ang_frame: str) -> dict[str, np.ndarray]:
    """Every velocity key of ``VELOCITY_KEYS`` whose source signal is present, recomputed from the (resampled) signals."""
    out: dict[str, np.ndarray] = {}
    if "joint_pos" in arrays:
        out["joint_vel"] = joint_vel_from_joint_pos(arrays["joint_pos"], fps, root_ang_frame)
    if "body_pos_w" in arrays:
        out["body_lin_vel_w"] = fkm.finite_difference(arrays["body_pos_w"], fps)
    if "body_quat_w" in arrays:
        out["body_ang_vel_w"] = fkm.angular_velocity_world(arrays["body_quat_w"], fps)
    if "object_pos_w" in arrays:
        out["object_lin_vel_w"] = fkm.finite_difference(arrays["object_pos_w"], fps)
    if "object_quat_w" in arrays:
        out["object_ang_vel_w"] = fkm.angular_velocity_world(arrays["object_quat_w"], fps)
    return out


#: accepted spellings of the two gate bounds (plan: ``max_qdot_rad_s`` / ``max_qddot_rad_s2``)
GATE_KEYS: dict[str, tuple[str, ...]] = {"max_qdot_rad_s": ("max_qdot_rad_s", "qdot_max", "qdot_max_rad_s"),
                                         "max_qddot_rad_s2": ("max_qddot_rad_s2", "qddot_max", "qddot_max_rad_s2")}


def normalize_gate(gate: Mapping[str, float] | None) -> dict[str, float] | None:
    """``{"max_qdot_rad_s": 8.0, "max_qddot_rad_s2": 100.0}`` (either bound may be omitted; alternative spellings in ``GATE_KEYS``)."""
    if gate is None:
        return None
    out: dict[str, float] = {}
    for canon, names in GATE_KEYS.items():
        for n in names:
            if n in gate and gate[n] is not None:
                v = float(gate[n])
                if not np.isfinite(v) or v <= 0.0:
                    raise ValueError(f"gate {canon} must be a positive finite bound, got {gate[n]!r}")
                out[canon] = v
                break
    unknown = set(gate) - {n for names in GATE_KEYS.values() for n in names}
    if unknown:
        raise ValueError(f"unknown gate keys {sorted(unknown)}; accepted: {GATE_KEYS}")
    return out


def dof_speed_limits(joint_pos: np.ndarray, fps: float) -> dict[str, float]:
    """``max |qdot|`` (rad/s) and ``max |qddot|`` (rad/s^2) of the 29 dof columns of ``joint_pos`` (T, 36): central differences at ``fps``
    (one-sided at the ends), the acceleration differenced from the velocity -- the generator audit's definition (``np.gradient`` twice)."""
    q = np.asarray(joint_pos, dtype=np.float64)[:, 7:]
    if q.shape[0] < 2:
        return {"max_qdot_rad_s": 0.0, "max_qddot_rad_s2": 0.0}
    qd = fkm.finite_difference(q, fps)
    qdd = fkm.finite_difference(qd, fps)
    return {"max_qdot_rad_s": float(np.abs(qd).max()), "max_qddot_rad_s2": float(np.abs(qdd).max())}


def gate_check(limits: Mapping[str, float], gate: Mapping[str, float] | None) -> list[str]:
    """Names of the gate bounds ``limits`` exceeds (empty = passes; ``gate`` None = no gate)."""
    if not gate:
        return []
    return [k for k, bound in gate.items() if float(limits.get(k, 0.0)) > float(bound)]


# ================================================================================================ one clip
def _is_per_frame(key: str, value: np.ndarray, T: int) -> bool:
    return key not in PER_CLIP_KEYS and value.ndim >= 1 and value.shape[0] == T


def retime_arrays(src: Mapping[str, np.ndarray], warp: TimeWarp, *, root_ang_frame: str = "auto") -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Time-warp one clip's arrays.  Returns ``(arrays, info)``; ``info`` records the root angular frame and how unknown keys were handled."""
    if root_ang_frame not in ROOT_ANG_FRAMES:
        raise ValueError(f"root_ang_frame must be one of {ROOT_ANG_FRAMES}")
    jp_src = np.asarray(src["joint_pos"])
    T = int(jp_src.shape[0])
    if T != warp.old_frames:
        raise ValueError(f"warp built for {warp.old_frames} frames, clip has {T}")
    fps = scalar_fps(src["fps"]) if "fps" in src else 50.0
    u = warp.old_index
    out: dict[str, np.ndarray] = {}
    info: dict[str, Any] = {"unknown_linear": [], "unknown_nearest": [], "copied": [], "nearest": [], "recomputed": []}

    frame = root_ang_frame
    if frame == "auto":
        frame = detect_root_ang_frame(jp_src, src["joint_vel"], fps) if ("joint_vel" in src and np.asarray(src["joint_vel"]).shape == (T, jp_src.shape[1] - 1)) else "world"
    info["root_ang_frame"] = frame

    jp = np.asarray(jp_src, dtype=np.float64)
    out["joint_pos"] = np.concatenate([resample_linear(jp[:, :3], u), resample_quat(jp[:, 3:7], u), resample_linear(jp[:, 7:], u)], axis=1)
    for key, value in src.items():
        if key == "joint_pos" or key in VELOCITY_KEYS:
            continue
        value = np.asarray(value)
        if not _is_per_frame(key, value, T):
            out[key] = value
            info["copied"].append(key)
            continue
        if key in QUAT_KEYS:
            out[key] = resample_quat(value, u)
        elif key in LINEAR_KEYS:
            out[key] = resample_linear(value, u)
        elif value.dtype.kind in "biuUSO" or any(h in key.lower() for h in NEAREST_NAME_HINTS):
            out[key] = resample_nearest(value, u)
            info["nearest"].append(key)
            if value.dtype.kind == "f":
                info["unknown_nearest"].append(key)
        elif VELOCITY_NAME_PATTERN.search(key.lower()):
            raise ValueError(f"per-frame float key {key!r} {value.shape} looks like a velocity / acceleration the tool cannot recompute "
                             f"(known: {sorted(VELOCITY_KEYS)}); interpolating it would keep the SOURCE speed on the retimed clip -- "
                             f"add it to VELOCITY_KEYS with its source signal or drop it from the clip")
        else:
            out[key] = resample_linear(value, u)
            info["unknown_linear"].append(key)
    # velocities: recomputed from the resampled signals (never divided); keys absent from the source are not invented
    vel = recompute_velocities(out, fps, frame)
    for key in VELOCITY_KEYS:
        if key in src:
            if key not in vel:
                raise ValueError(f"cannot recompute {key}: its source signal {VELOCITY_KEYS[key]!r} is missing")
            out[key] = vel[key]
            info["recomputed"].append(key)
    # dtypes follow the source (float arrays back to their stored precision)
    for key, value in src.items():
        if key in out and np.asarray(value).dtype.kind == "f" and out[key].dtype != np.asarray(value).dtype:
            out[key] = out[key].astype(np.asarray(value).dtype)
    return out, info


# ================================================================================================ manifest
def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    if os.environ.get("HERO_GIT_SHA"):
        return os.environ["HERO_GIT_SHA"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


def _stats(values: Sequence[float]) -> dict[str, float]:
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if x.size == 0:
        return {"n": 0, "min": float("nan"), "max": float("nan"), "mean": float("nan"), "p50": float("nan"), "p95": float("nan")}
    return {"n": int(x.size), "min": float(x.min()), "max": float(x.max()), "mean": float(x.mean()), "p50": float(np.quantile(x, 0.5)), "p95": float(np.quantile(x, 0.95))}


def load_manifest_rows(path: Path) -> tuple[dict[str, Any] | None, dict[str, dict[str, Any]]]:
    """``(top-level dict or None, {file name: row})``; accepts ``{'clips': [...]}``, a bare list or ``{stem: row}``."""
    if not path.is_file():
        return None, {}
    obj = json.loads(path.read_text())
    if isinstance(obj, dict) and isinstance(obj.get("clips"), list):
        rows = obj["clips"]
        top: dict[str, Any] | None = obj
    elif isinstance(obj, list):
        rows, top = obj, {"clips": obj}
    elif isinstance(obj, dict):
        rows, top = list(obj.values()), {"clips": list(obj.values())}
    else:
        raise ValueError(f"{path}: unrecognised manifest layout")
    by_file: dict[str, dict[str, Any]] = {}
    for e in rows:
        name = str(e.get("file") or f"{e.get('clip_id')}.npz")
        if not name.endswith(".npz"):
            name += ".npz"
        by_file[name] = dict(e)
    return top, by_file


def reach_end_of(entry: Mapping[str, Any] | None) -> int | None:
    if not entry:
        return None
    if entry.get("reach_end_frame") is not None:
        return int(entry["reach_end_frame"])
    if entry.get("n_frames") is not None and entry.get("hold_frames") is not None:
        return int(entry["n_frames"]) - int(entry["hold_frames"])
    return None


def scale_manifest_entry(entry: Mapping[str, Any], warp: TimeWarp, arrays: Mapping[str, np.ndarray], out_path: Path, fps: float) -> dict[str, Any]:
    """Manifest row of the retimed clip: goal pose / labels unchanged, frame indices mapped through the warp, file facts recomputed."""
    e = dict(entry)
    N = warp.new_frames
    old = {k: e.get(k) for k in ("settle_frames", "reach_start_frame", "reach_end_frame", "reach_frames", "hold_frames", "n_frames", "duration_s")}
    for k, side in MANIFEST_INDEX_SIDES.items():
        if e.get(k) is not None:
            e[k] = warp.map_index(int(e[k]), side)
    e["n_frames"] = int(N)
    e["duration_s"] = float(N / fps)
    if e.get("reach_end_frame") is not None:
        e["hold_frames"] = int(N - int(e["reach_end_frame"]))
        if e.get("reach_start_frame") is not None:
            e["reach_frames"] = int(e["reach_end_frame"]) - int(e["reach_start_frame"])
    elif e.get("hold_frames") is not None and old["n_frames"] is not None:
        e["hold_frames"] = int(N - warp.map_index(int(old["n_frames"]) - int(old["hold_frames"]), "ceil"))
    e["file"] = out_path.name
    e["fps"] = int(round(fps))
    e["sha256"] = _sha256(out_path)
    e["bytes"] = int(out_path.stat().st_size)
    if "joint_vel" in arrays and "max_qdot_rad_s" in e:
        e["max_qdot_rad_s"] = float(np.abs(np.asarray(arrays["joint_vel"], dtype=np.float64)[:, 6:]).max())
    if "h_ref" in arrays:
        e["h_ref_min_m"] = float(np.asarray(arrays["h_ref"]).min())
        e["h_ref_max_m"] = float(np.asarray(arrays["h_ref"]).max())
    e["retime"] = {
        "factor_effective": warp.factor_effective, "mode": warp.mode, "segments_old_new": [list(s) for s in warp.segments],
        "src": {k: v for k, v in old.items() if v is not None}, "src_file": entry.get("file"), "src_sha256": entry.get("sha256"),
    }
    return e


def retime_manifest_top(top: Mapping[str, Any] | None, clips: Sequence[dict[str, Any]], *, src_dir: Path, src_manifest_sha: str | None, factor: float,
                        hold_s: float | None, fps: float, missing_files: Sequence[str], unlisted_files: Sequence[str],
                        verification: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Top-level manifest of the retimed bench: the source's provenance / protocol / selection blocks with the timing, per-height
    durations and verification facts replaced, plus a ``retime`` block."""
    out: dict[str, Any] = json.loads(json.dumps(top or {}, default=str))
    out["created_utc"] = time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime())
    out["git_sha"] = _git_sha()
    out["host"] = platform.node()
    out["python"] = sys.version.split()[0]
    out["n_clips"] = len(clips)
    labels = out.get("height_labels") or sorted({c.get("height_label") for c in clips if c.get("height_label")})
    if labels and isinstance(out.get("per_height"), dict):
        for lab in labels:
            cs = [c for c in clips if c.get("height_label") == lab]
            ph = dict(out["per_height"].get(lab) or {})
            ph["n_clips"] = len(cs)
            ph["duration_s"] = _stats([c["duration_s"] for c in cs])
            if all(c.get("reach_frames") is not None for c in cs):
                ph["reach_duration_s"] = _stats([c["reach_frames"] / fps for c in cs])
            out["per_height"][lab] = ph
    timing = dict((out.get("protocol") or {}).get("timing") or {})
    if timing or "protocol" in out:
        timing["fps"] = int(round(fps))
        timing["time_scale"] = float(timing.get("time_scale", 1.0)) * float(factor)
        if hold_s is not None:
            timing["hold_s"] = float(hold_s)
        elif timing.get("hold_s") is not None:
            timing["hold_s"] = float(timing["hold_s"]) * float(factor)
        if clips and all(c.get("settle_frames") is not None for c in clips):
            settles = sorted({int(c["settle_frames"]) for c in clips})
            timing["settle_frames"] = settles[0] if len(settles) == 1 else settles
        timing["retime"] = f"time axis stretched x{factor:g} by data_tools.retime_bench ({'hold stretched too' if hold_s is None else f'hold fixed at {hold_s:g} s'}); reach_end_frame = first hold frame of the retimed clip"
        out.setdefault("protocol", {})["timing"] = timing
    ver = dict(out.get("verification") or {})
    ver["clip_lengths"] = {c["file"]: int(c["n_frames"]) for c in clips}
    ver["total_frames"] = int(sum(int(c["n_frames"]) for c in clips))
    ver["n_files"] = len(clips)
    ver["missing_files"] = list(missing_files)
    ver["unexpected_files"] = list(unlisted_files)
    ver["removed_stale_files"] = []
    ver["mode"] = "retime_bench"
    ver["verifier"] = "data_tools.retime_bench.verify_outputs (schema.validate_holosoma_npz + sim2sim.reference.ClipReference when importable)"
    if verification is not None:
        ver["n_loaded"] = int(verification.get("loader_loaded") or 0) if verification.get("loader") and "skipped" not in str(verification.get("loader")) else int(verification.get("schema_checked") or 0)
        ver["n_skipped"] = len(verification.get("schema_problems") or {}) + len(verification.get("loader_errors") or {})
        ver["skipped"] = sorted(set(verification.get("schema_problems") or {}) | set(verification.get("loader_errors") or {}))
        ver["notes"] = [f"loader: {verification.get('loader') or 'skipped'}"]
    out["verification"] = ver
    out["retime"] = {
        "tool": "data_tools.retime_bench", "tool_version": TOOL_VERSION, "factor": float(factor), "hold_s": ("keep" if hold_s is None else float(hold_s)),
        "src_dir": str(src_dir), "src_manifest_sha256": src_manifest_sha, "src_schema": (top or {}).get("schema"),
        "note": "same goals (target_pos_w / target_*_deg / table) as the source; frame indices mapped through the per-clip time warp",
    }
    out["clips"] = list(clips)
    return out


# ================================================================================================ verification
def verify_outputs(out_dir: Path, files: Sequence[str], expected_frames: Mapping[str, int], *, loader: bool = True) -> dict[str, Any]:
    """Schema check of every output (always) + ``sim2sim.reference.ClipReference`` load (when importable and ``loader``)."""
    result: dict[str, Any] = {"schema_checked": 0, "schema_problems": {}, "loader": None, "loader_loaded": 0, "loader_errors": {}}
    for name in files:
        arrays = read_npz(out_dir / name)
        require_ext = all(k in arrays for k in EXTENSION_KEYS)
        problems = validate_holosoma_npz(arrays, require_extension=require_ext, expected_fps=None)
        if int(arrays["joint_pos"].shape[0]) != int(expected_frames[name]):
            problems.append(f"frames {arrays['joint_pos'].shape[0]} != expected {expected_frames[name]}")
        result["schema_checked"] += 1
        if problems:
            result["schema_problems"][name] = problems
    if loader:
        try:
            from sim2sim.reference import ClipReference, discover_clips  # noqa: PLC0415
        except Exception as exc:  # noqa: BLE001
            result["loader"] = f"sim2sim.reference not importable ({type(exc).__name__}: {exc}); loader pass skipped"
        else:
            result["loader"] = "sim2sim.reference.ClipReference"
            found = {p.name for p in discover_clips(str(out_dir))}
            for name in files:
                if name not in found:
                    result["loader_errors"][name] = "not returned by discover_clips"
                    continue
                try:
                    ref = ClipReference(out_dir / name)
                    if ref.T != int(expected_frames[name]):
                        raise ValueError(f"ClipReference.T {ref.T} != {expected_frames[name]}")
                    ref.palm_pose_w(ref.T - 1)
                    result["loader_loaded"] += 1
                except Exception as exc:  # noqa: BLE001
                    result["loader_errors"][name] = f"{type(exc).__name__}: {exc}"
    return result


# ================================================================================================ driver
def _parse_hold(text: str) -> float | None:
    t = str(text).strip().lower()
    if t in ("keep", "", "none"):
        return None
    v = float(t)
    if not np.isfinite(v) or v <= 0.0:
        raise ValueError(f"--hold-s must be 'keep' or a positive duration in seconds, got {text!r}")
    return v


def retime_bench(src_dir: str | os.PathLike, out_dir: str | os.PathLike, factor: float, *, hold_s: float | None = None, root_ang_frame: str = "auto",
                 pattern: str = "*.npz", overwrite: bool = False, verify: bool = True, quiet: bool = False,
                 gate: Mapping[str, float] | None = None, rename: Callable[[str], str] | None = None, source_tag: str | None = None) -> dict[str, Any]:
    """Write the retimed bench; returns the receipt (also written to ``<out>/RETIME_RECEIPT.json``).

    ``gate`` / ``rename`` / ``source_tag`` (all optional, see the module docstring): drop clips whose retimed dof speed / acceleration exceed
    the bounds, rename the outputs, override the per-clip ``source_tag`` key.  Defaults keep the original behaviour exactly."""
    src = Path(src_dir)
    out = Path(out_dir)
    gate_n = normalize_gate(gate)
    if not src.is_dir():
        raise FileNotFoundError(f"--src {src} is not a directory")
    if factor <= 0.0 or not np.isfinite(factor):
        raise ValueError(f"--factor must be positive, got {factor!r}")
    if out.exists() and any(out.iterdir()) and not overwrite:
        raise FileExistsError(f"--out {out} exists and is not empty (use --overwrite)")
    if src.resolve() == out.resolve():
        raise ValueError("--out must differ from --src")
    files = sorted(p.name for p in src.glob(pattern) if p.is_file())
    if not files:
        raise FileNotFoundError(f"no {pattern} in {src}")
    top, rows = load_manifest_rows(src / MANIFEST_NAME)
    src_manifest_sha = _sha256(src / MANIFEST_NAME) if (src / MANIFEST_NAME).is_file() else None
    if hold_s is not None and not rows:
        raise ValueError("--hold-s <s> needs a BENCH_MANIFEST.json with reach_end_frame per clip (none found)")
    out.mkdir(parents=True, exist_ok=True)
    t0 = time.time()
    receipts: list[dict[str, Any]] = []
    clips_new: list[dict[str, Any]] = []
    expected: dict[str, int] = {}
    fps_seen: set[float] = set()
    frames_seen: dict[str, int] = {}
    written: list[str] = []                 # output file names (after ``rename``) that passed the gate
    gate_dropped: list[dict[str, Any]] = []
    for i, name in enumerate(files):
        arrays = read_npz(src / name)
        T = int(arrays["joint_pos"].shape[0])
        fps = scalar_fps(arrays["fps"]) if "fps" in arrays else 50.0
        fps_seen.add(fps)
        row = rows.get(name)
        re_ = reach_end_of(row)
        hold_new = None
        if hold_s is not None:
            if re_ is None or not (1 <= re_ <= T - 1):
                raise ValueError(f"{name}: --hold-s needs a valid reach_end_frame in the manifest (got {re_!r} for {T} frames)")
            hold_new = int(np.floor(hold_s * fps + 0.5))
        warp = build_warp(T, factor, reach_end_frame=re_, hold_frames_new=hold_new)
        new_arrays, info = retime_arrays(arrays, warp, root_ang_frame=root_ang_frame)
        if source_tag is not None:
            new_arrays["source_tag"] = np.asarray(str(source_tag))
        out_name = name if rename is None else str(rename(name))
        if not out_name.endswith(".npz") or "/" in out_name or out_name in expected:
            raise ValueError(f"rename({name!r}) -> {out_name!r}: must be a unique *.npz file name")
        limits = dof_speed_limits(new_arrays["joint_pos"], fps)
        exceeded = gate_check(limits, gate_n)
        re_new = warp.map_index(re_, MANIFEST_INDEX_SIDES["reach_end_frame"]) if re_ is not None else None  # first hold frame of the new clip
        rec = {
            "file": out_name, "src_file": name, "src_sha256": _sha256(src / name), "sha256": None, "old_frames": T, "new_frames": warp.new_frames,
            "factor_effective": warp.factor_effective, "mode": warp.mode, "root_ang_frame": info["root_ang_frame"],
            "old_reach_end_frame": re_, "new_reach_end_frame": re_new,
            "old_hold_frames": (T - re_ if re_ is not None else None), "new_hold_frames": (warp.new_frames - re_new if re_new is not None else None),
            "in_manifest": row is not None, "unknown_linear": info["unknown_linear"], "unknown_nearest": info["unknown_nearest"],
            "max_qdot_rad_s": limits["max_qdot_rad_s"], "max_qddot_rad_s2": limits["max_qddot_rad_s2"], "gate_exceeded": exceeded,
        }
        if exceeded:
            # over the speed / acceleration gate: nothing written, no manifest row -- the receipt keeps the evidence
            rec["written"] = False
            gate_dropped.append({"file": out_name, "src_file": name, "exceeded": exceeded, **limits, "gate": dict(gate_n or {})})
            receipts.append(rec)
            if not quiet:
                print(f"[retime_bench] GATE DROP {name}: {', '.join(f'{k} {limits[k]:.2f} > {gate_n[k]:g}' for k in exceeded)}", flush=True)
            continue
        out_path = out / out_name
        tmp = out / (out_name + ".tmp.npz")
        np.savez_compressed(tmp, **new_arrays)
        os.replace(tmp, out_path)
        rec["sha256"] = _sha256(out_path)
        rec["written"] = True
        expected[out_name] = warp.new_frames
        written.append(out_name)
        receipts.append(rec)
        if row is not None:
            clips_new.append(scale_manifest_entry(row, warp, new_arrays, out_path, fps))
        frames_seen[out_name] = warp.new_frames
        if not quiet and (i % 50 == 0 or i == len(files) - 1):
            print(f"[retime_bench] {i + 1}/{len(files)} {name}: {T} -> {warp.new_frames} frames (x{warp.factor_effective:.4f}, root ang {info['root_ang_frame']})", flush=True)
    missing = sorted(set(rows) - set(files))
    unlisted = sorted(set(files) - set(rows))
    fps_out = sorted(fps_seen)[0] if len(fps_seen) == 1 else float("nan")
    verification = verify_outputs(out, written, expected, loader=verify)
    ok = not verification["schema_problems"] and not verification["loader_errors"]
    # a bench that fails verification must not carry the canonical names: the evaluators pick up <motion dir>/BENCH_MANIFEST.json and
    # the corpus builder keys on RETIME_RECEIPT.json -- the .FAILED.json copies keep the evidence without being consumable
    manifest_name = MANIFEST_NAME if ok else MANIFEST_FAILED_NAME
    receipt_name = RECEIPT_NAME if ok else RECEIPT_FAILED_NAME
    for stale in (MANIFEST_NAME, RECEIPT_NAME, MANIFEST_FAILED_NAME, RECEIPT_FAILED_NAME):
        if (out / stale).exists():
            (out / stale).unlink()  # --overwrite of an earlier (good or failed) build: never leave both variants behind
    manifest_sha = None
    if rows:
        manifest = retime_manifest_top(top, clips_new, src_dir=src, src_manifest_sha=src_manifest_sha, factor=factor, hold_s=hold_s, fps=fps_out,
                                       missing_files=missing, unlisted_files=unlisted, verification=verification)
        (out / manifest_name).write_text(json.dumps(manifest, indent=1, sort_keys=True, default=str))
        manifest_sha = _sha256(out / manifest_name)
    receipt = {
        "schema": RECEIPT_SCHEMA, "tool_version": TOOL_VERSION, "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git_sha": _git_sha(),
        "host": platform.node(), "python": sys.version.split()[0], "wall_s": time.time() - t0, "ok": bool(ok),
        "src_dir": str(src), "out_dir": str(out), "factor": float(factor), "hold_s": ("keep" if hold_s is None else float(hold_s)), "root_ang_frame": root_ang_frame,
        "fps": fps_out, "src_manifest_sha256": src_manifest_sha, "manifest_file": (manifest_name if rows else None), "manifest_sha256": manifest_sha,
        "receipt_file": receipt_name, "n_clips": len(files), "n_written": len(written), "n_in_manifest": len(clips_new),
        "manifest_missing_files": missing[:200], "n_manifest_missing_files": len(missing), "unlisted_files": unlisted[:200],
        "frames": {"old_total": int(sum(r["old_frames"] for r in receipts if r["written"])), "new_total": int(sum(r["new_frames"] for r in receipts if r["written"])),
                   "old_max": int(max([r["old_frames"] for r in receipts if r["written"]] or [0])), "new_max": int(max([r["new_frames"] for r in receipts if r["written"]] or [0]))},
        "root_ang_frame_counts": {f: sum(1 for r in receipts if r["root_ang_frame"] == f) for f in ("world", "pelvis")},
        "index_sides": dict(MANIFEST_INDEX_SIDES), "verification": verification,
        "gate": gate_n, "n_gate_dropped": len(gate_dropped), "gate_dropped": gate_dropped,
        "renamed": rename is not None, "source_tag_override": source_tag, "clips": receipts,
    }
    (out / receipt_name).write_text(json.dumps(receipt, indent=1, default=str))
    if not ok:
        raise RuntimeError(f"retimed bench failed verification (written as {manifest_name} / {receipt_name}, NOT usable): "
                           f"schema {len(verification['schema_problems'])} clip(s) {list(verification['schema_problems'].items())[:2]}; "
                           f"loader {len(verification['loader_errors'])} clip(s) {list(verification['loader_errors'].items())[:2]}")
    if not quiet:
        fr = receipt["frames"]
        print(f"[retime_bench] wrote {len(written)} / {len(files)} clips to {out} (x{factor:g}, hold {'keep' if hold_s is None else f'{hold_s:g} s'}"
              f"{'' if not gate_dropped else f', {len(gate_dropped)} dropped by the gate'}): frames {fr['old_total']} -> {fr['new_total']} "
              f"(longest {fr['old_max']} -> {fr['new_max']} = {fr['new_max'] / fps_out:.2f} s); manifest rows {len(clips_new)}"
              f"{'' if not missing else f'; {len(missing)} manifest rows have no file here'}; verification: schema {verification['schema_checked']} ok, "
              f"loader {verification['loader'] or 'skipped'} {verification['loader_loaded']} loaded", flush=True)
    return receipt


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--src", required=True, help="bench dir (<h>__<clip>.npz + BENCH_MANIFEST.json)")
    ap.add_argument("--out", required=True, help="output dir (must not exist / be empty unless --overwrite)")
    ap.add_argument("--factor", type=float, required=True, help="time stretch F: new frame count = round(F x old); 2.0 = twice as slow")
    ap.add_argument("--hold-s", default="keep", help="'keep' (stretch the hold with the rest, default) or a fixed hold duration in seconds")
    ap.add_argument("--root-ang-frame", choices=ROOT_ANG_FRAMES, default="auto", help="frame of joint_vel[:, 3:6] (auto = detect from the source clip)")
    ap.add_argument("--pattern", default="*.npz")
    ap.add_argument("--overwrite", action="store_true")
    ap.add_argument("--no-verify", action="store_true", help="skip the sim2sim ClipReference loader pass (the schema check always runs)")
    ap.add_argument("--quiet", action="store_true")
    a = ap.parse_args(argv)
    try:
        hold = _parse_hold(a.hold_s)
    except ValueError as exc:
        ap.error(str(exc))
    retime_bench(a.src, a.out, a.factor, hold_s=hold, root_ang_frame=a.root_ang_frame, pattern=a.pattern, overwrite=a.overwrite, verify=not a.no_verify, quiet=a.quiet)
    return 0


__all__ = ["INDEX_SIDES", "LINEAR_KEYS", "MANIFEST_FAILED_NAME", "MANIFEST_INDEX_FIELDS", "MANIFEST_INDEX_SIDES", "MANIFEST_NAME", "PER_CLIP_KEYS", "QUAT_KEYS",
           "RECEIPT_FAILED_NAME", "RECEIPT_NAME", "ROOT_ANG_FRAMES", "TOOL_VERSION", "TimeWarp", "VELOCITY_KEYS", "VELOCITY_NAME_PATTERN",
           "GATE_KEYS", "build_warp", "detect_root_ang_frame", "dof_speed_limits", "gate_check", "joint_vel_from_joint_pos", "load_manifest_rows",
           "normalize_gate", "reach_end_of", "recompute_velocities", "resample_linear", "resample_nearest", "resample_quat", "retime_arrays", "retime_bench",
           "retime_manifest_top", "scale_manifest_entry", "verify_outputs"]


if __name__ == "__main__":
    sys.exit(main())
