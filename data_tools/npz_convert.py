"""Convert G1 motion arrays into the HERO-extended HoloSoma NPZ format.

The converter accepts root xyz + wxyz quaternion + 29 joint angles, stored as
joint_pos or qpos. It can also read precomputed HoloSoma body channels. Positions
and rotations are resampled to 50 Hz, body poses and palm references are computed
with MuJoCo forward kinematics, and velocities are finite differences. Optional
object channels are supported for reading compatible files; AMASS preparation
always removes them by supplying only robot poses.

For the AMASS-only Quick Start corpus, use scripts/prepare_amass.py after external
G1 retargeting. For an additional motion source, this lower-level converter
accepts --source-tag, --license-class, and optional format conversion controls.

All quaternion arrays use wxyz. Conversion receipts record source hashes,
frame counts, source frame rate, and processing errors."""

from __future__ import annotations

from data_tools import _threads  # noqa: F401  -- single-threaded BLAS/OpenMP before numpy and MuJoCo load

import argparse
import concurrent.futures as cf
import hashlib
import json
import os
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from data_tools import fk_mujoco as fkm
from data_tools.schema import (
    ATTACHED_MASK_KEY,
    DEFAULT_STANDING_LEG_ANGLES,
    DEFAULT_STANDING_ROOT_Z,
    DOF_NAMES,
    HOLOSOMA_BODY_NAMES_32,
    LEG_DOF_IDX,
    LICENSE_CLASSES,
    ClipMeta,
    SchemaError,
    assert_valid,
    decode_names,
    make_parent_id,
    read_npz,
    scalar_fps,
)

FAMILIES: tuple[str, ...] = ("S1", "S2", "S3", "S4", "S5", "S6", "THIN")
_S5_KEYS: frozenset[str] = frozenset({"scene_xml", "spec", "dof_names"})
#: Authored IK palm targets [left, right] in world coordinates, shape (T, 2, 3).
HAND_TARGETS_KEY: str = "authored_hand_targets_w"


class UnsupportedFamilyError(ValueError):
    """Raised when the key set of a clip matches no known schema family."""


# --------------------------------------------------------------------------------------
# family detection + canonical extraction
# --------------------------------------------------------------------------------------


def detect_family(data: Mapping[str, Any]) -> str:
    """Classify a loaded NPZ dictionary into one of :data:`FAMILIES`."""
    keys = set(data.keys())
    if "qpos43" in keys or ("qpos" in keys and "joint_pos" not in keys):
        return "S3"
    if "joint_pos" not in keys:
        raise UnsupportedFamilyError(f"no joint_pos/qpos43/qpos key; keys={sorted(keys)}")
    if "body_pos_w" in keys and "body_names" in keys:
        names = decode_names(data["body_names"])
        if set(HOLOSOMA_BODY_NAMES_32) <= set(names):
            return "S6"
        if len(names) <= 2:
            return "S2"
        return "S1"
    if keys & {"ee_pos_w", "target_ee_pos_w", "table_height_m", "hand_mode"}:
        return "S4"
    if keys & _S5_KEYS:
        return "S5"
    return "THIN"


@dataclass
class CanonicalInput:
    """Family-independent view of a clip before FK (float64, wxyz)."""

    family: str
    fps: float
    joint_pos: np.ndarray  # (T, 36)
    joint_vel: np.ndarray | None = None  # (T, 35) as stored (None -> finite difference)
    object_pos_w: np.ndarray | None = None  # (T, 3) dynamic object only
    object_quat_w: np.ndarray | None = None  # (T, 4) wxyz
    box_size: np.ndarray | None = None  # (3,)
    body_pos_w: np.ndarray | None = None  # (T, 32, 3) in canonical body order
    body_quat_w: np.ndarray | None = None
    body_lin_vel_w: np.ndarray | None = None
    body_ang_vel_w: np.ndarray | None = None
    attached_mask: np.ndarray | None = None
    hand_targets_w: np.ndarray | None = None  # (T, 2, 3) authored palm targets of the source (checked against FK, never written)
    notes: list[str] = field(default_factory=list)

    @property
    def frames(self) -> int:
        return int(self.joint_pos.shape[0])


def _optional_phase_arrays(data: Mapping[str, Any], T: int, notes: list[str]) -> tuple[np.ndarray | None, np.ndarray | None]:
    """``(attached_mask uint8 (T,), authored_hand_targets_w (T, 2, 3))`` of the source when present and well-formed."""
    mask = None
    if ATTACHED_MASK_KEY in data:
        am = np.asarray(data[ATTACHED_MASK_KEY])
        if am.shape == (T,) and am.dtype.kind in "biuf" and np.isfinite(am.astype(np.float64)).all():
            mask = (am.astype(np.float64) != 0.0).astype(np.uint8)
        else:
            notes.append(f"{ATTACHED_MASK_KEY} shape {am.shape} dtype {am.dtype} unusable -> ignored")
    targets = None
    if HAND_TARGETS_KEY in data:
        ht = np.asarray(data[HAND_TARGETS_KEY], dtype=np.float64)
        if ht.shape == (T, 2, 3) and np.isfinite(ht).all():
            targets = ht
        else:
            notes.append(f"{HAND_TARGETS_KEY} shape {ht.shape} unusable -> ignored")
    return mask, targets


def _permute_joints(joint_pos: np.ndarray, joint_vel: np.ndarray | None, names: list[str], notes: list[str]):
    """Reorder dof columns into DOF_NAMES order when the file's joint_names are a permutation."""
    if names == list(DOF_NAMES):
        return joint_pos, joint_vel
    if set(names) != set(DOF_NAMES) or len(names) != 29:
        raise SchemaError(f"joint_names are not the 29 G1 dofs: {names}")
    perm = np.asarray([names.index(n) for n in DOF_NAMES], dtype=np.int64)
    joint_pos = np.concatenate([joint_pos[:, :7], joint_pos[:, 7:][:, perm]], axis=1)
    if joint_vel is not None:
        joint_vel = np.concatenate([joint_vel[:, :6], joint_vel[:, 6:][:, perm]], axis=1)
    notes.append("joint columns permuted into DOF_NAMES order")
    return joint_pos, joint_vel


def _dynamic_object(data: Mapping[str, Any], notes: list[str], T: int) -> tuple[np.ndarray | None, np.ndarray | None]:
    """Return (object_pos_w, object_quat_w) if the clip carries a per-frame object, else (None, None)."""
    if "object_pos_w" not in data:
        return None, None
    op = np.asarray(data["object_pos_w"], dtype=np.float64)
    if op.ndim == 1:
        notes.append("static object_pos_w (3,) target dropped")
        return None, None
    if op.shape != (T, 3):
        raise SchemaError(f"object_pos_w shape {op.shape} != {(T, 3)}")
    if "object_quat_w" not in data:
        raise SchemaError("object_pos_w (T,3) present but object_quat_w missing")
    oq = np.asarray(data["object_quat_w"], dtype=np.float64)
    if oq.shape != (T, 4):
        raise SchemaError(f"object_quat_w shape {oq.shape} != {(T, 4)}")
    return op, fkm.quat_normalize(oq)


def _root_quat_check(joint_pos: np.ndarray) -> np.ndarray:
    jp = np.asarray(joint_pos, dtype=np.float64).copy()
    norms = np.linalg.norm(jp[:, 3:7], axis=1)
    if not np.isfinite(jp).all():
        raise SchemaError("joint_pos contains NaN/Inf")
    if np.max(np.abs(norms - 1.0)) > 1.0e-2:
        raise SchemaError("joint_pos[:, 3:7] is not a unit wxyz quaternion (max |norm-1| > 1e-2)")
    jp[:, 3:7] = jp[:, 3:7] / norms[:, None]
    return jp


def extract_canonical(data: Mapping[str, Any], family: str | None = None) -> CanonicalInput:
    """Map any supported family onto :class:`CanonicalInput` (no resampling, no FK)."""
    family = family or detect_family(data)
    notes: list[str] = []

    if family == "S3":
        q = np.asarray(data["qpos43"] if "qpos43" in data else data["qpos"], dtype=np.float64)
        if q.ndim != 2 or q.shape[1] not in (36, 43):
            raise SchemaError(f"qpos43/qpos must be (T,43) or (T,36), got {q.shape}")
        T = q.shape[0]
        joint_pos = _root_quat_check(q[:, :36])
        obj_p = obj_q = None
        if q.shape[1] == 43:
            obj_p = q[:, 36:39].copy()
            obj_q = fkm.quat_normalize(q[:, 39:43])
        fps = scalar_fps(data.get("fps"), default=50.0)
        if "fps" not in data:
            notes.append("fps absent -> assumed 50")
        box = np.asarray(data["box_size"], dtype=np.float64) if "box_size" in data else None
        return CanonicalInput(family, fps, joint_pos, None, obj_p, obj_q, box, notes=notes)

    joint_pos = np.asarray(data["joint_pos"], dtype=np.float64)
    if joint_pos.ndim != 2 or joint_pos.shape[1] != 36:
        raise SchemaError(f"joint_pos must be (T,36), got {joint_pos.shape}")
    T = joint_pos.shape[0]
    joint_vel = None
    if "joint_vel" in data:
        joint_vel = np.asarray(data["joint_vel"], dtype=np.float64)
        if joint_vel.shape != (T, 35) or not np.isfinite(joint_vel).all():
            notes.append(f"stored joint_vel {joint_vel.shape} unusable -> recomputed")
            joint_vel = None
    if "joint_names" in data:
        joint_pos, joint_vel = _permute_joints(joint_pos, joint_vel, decode_names(data["joint_names"]), notes)
    joint_pos = _root_quat_check(joint_pos)
    fps = scalar_fps(data.get("fps"), default=50.0)
    if "fps" not in data:
        notes.append("fps absent -> assumed 50")
    obj_p, obj_q = _dynamic_object(data, notes, T)
    box = np.asarray(data["box_size"], dtype=np.float64) if "box_size" in data else None

    if family == "S5":
        dropped = sorted(k for k in ("scene_xml", "spec") if k in data)
        notes.append(f"stairs analytic clip: {dropped} not carried (flat-terrain tracker npz)")
    canon = CanonicalInput(family, fps, joint_pos, joint_vel, obj_p, obj_q, box, notes=notes)
    canon.attached_mask, canon.hand_targets_w = _optional_phase_arrays(data, T, notes)
    if family == "S6":
        names = decode_names(data["body_names"])
        idx = np.asarray([names.index(n) for n in HOLOSOMA_BODY_NAMES_32], dtype=np.int64)
        bp = np.asarray(data["body_pos_w"], dtype=np.float64)
        bq = np.asarray(data["body_quat_w"], dtype=np.float64)
        ok = bp.shape == (T, len(names), 3) and bq.shape == (T, len(names), 4)
        ok = ok and np.isfinite(bp).all() and np.isfinite(bq).all()
        if ok:
            canon.body_pos_w = bp[:, idx]
            canon.body_quat_w = fkm.quat_normalize(bq[:, idx])
            for key, attr in (("body_lin_vel_w", "body_lin_vel_w"), ("body_ang_vel_w", "body_ang_vel_w")):
                if key in data:
                    arr = np.asarray(data[key], dtype=np.float64)
                    if arr.shape == (T, len(names), 3) and np.isfinite(arr).all():
                        setattr(canon, attr, arr[:, idx])
            if len(names) != 32:
                notes.append(f"{len(names)}-body file reduced to the 32 canonical bodies")
        else:
            notes.append("stored body arrays malformed -> FK recomputed")
    return canon


# --------------------------------------------------------------------------------------
# resampling + derivatives
# --------------------------------------------------------------------------------------


def resample_grid(frames_in: int, fps_in: float, fps_out: float) -> np.ndarray:
    """Output sample times (s) covering ``[0, (frames_in-1)/fps_in]`` on a ``1/fps_out`` grid."""
    if frames_in < 2:
        return np.zeros(1, dtype=np.float64)
    duration = (frames_in - 1) / float(fps_in)
    n_out = int(np.floor(duration * float(fps_out) + 1.0e-9)) + 1
    return np.arange(n_out, dtype=np.float64) / float(fps_out)


def interp_linear(x: np.ndarray, t_in: np.ndarray, t_out: np.ndarray) -> np.ndarray:
    """Piecewise-linear resampling of ``x`` (T, ...) along axis 0."""
    x = np.asarray(x, dtype=np.float64)
    flat = x.reshape(x.shape[0], -1)
    out = np.empty((t_out.shape[0], flat.shape[1]), dtype=np.float64)
    for c in range(flat.shape[1]):
        out[:, c] = np.interp(t_out, t_in, flat[:, c])
    return out.reshape((t_out.shape[0],) + x.shape[1:])


def interp_quat(q: np.ndarray, t_in: np.ndarray, t_out: np.ndarray) -> np.ndarray:
    """Slerp resampling of wxyz quaternions ``q`` (T, ..., 4) along axis 0 (sign-continuous)."""
    q = fkm.quat_normalize(q)
    # make the sequence sign-continuous first so interior slerps are short-arc
    q = q.copy()
    for t in range(1, q.shape[0]):
        flip = np.sum(q[t] * q[t - 1], axis=-1) < 0.0
        q[t][flip] = -q[t][flip]
    idx = np.clip(np.searchsorted(t_in, t_out, side="right") - 1, 0, q.shape[0] - 2)
    t0 = t_in[idx]
    t1 = t_in[idx + 1]
    frac = np.clip((t_out - t0) / np.maximum(t1 - t0, 1.0e-12), 0.0, 1.0)
    shape = (t_out.shape[0],) + (1,) * (q.ndim - 2)
    return fkm.quat_slerp(q[idx], q[idx + 1], frac.reshape(shape))


def resample_to_fps(canon: CanonicalInput, fps_out: float) -> CanonicalInput:
    """Resample joint_pos (linear; slerp on the root quaternion) and the object track to ``fps_out``.

    Velocities and body arrays are dropped (recomputed by FK/finite differences later).
    Duration is preserved to within one output frame.
    """
    if np.isclose(canon.fps, fps_out, rtol=0.0, atol=1.0e-6):
        return canon
    T = canon.frames
    t_in = np.arange(T, dtype=np.float64) / canon.fps
    t_out = resample_grid(T, canon.fps, fps_out)
    jp = canon.joint_pos
    jp_out = np.concatenate(
        [interp_linear(jp[:, :3], t_in, t_out), interp_quat(jp[:, 3:7], t_in, t_out), interp_linear(jp[:, 7:], t_in, t_out)],
        axis=1,
    )
    op = oq = None
    if canon.object_pos_w is not None and canon.object_quat_w is not None:
        op = interp_linear(canon.object_pos_w, t_in, t_out)
        oq = interp_quat(canon.object_quat_w, t_in, t_out)
    notes = canon.notes + [f"resampled {canon.fps:g} -> {fps_out:g} fps ({T} -> {t_out.shape[0]} frames)"]
    out = CanonicalInput(canon.family, float(fps_out), jp_out, None, op, oq, canon.box_size, notes=notes)
    if canon.attached_mask is not None:  # nearest source frame (a label, not a signal)
        idx = np.clip(np.rint(t_out * canon.fps).astype(np.int64), 0, T - 1)
        out.attached_mask = canon.attached_mask[idx].astype(np.uint8)
    if canon.hand_targets_w is not None:
        out.hand_targets_w = interp_linear(canon.hand_targets_w, t_in, t_out)
    return out


def joint_vel_from_joint_pos(joint_pos: np.ndarray, fps: float) -> np.ndarray:
    """(T, 35) = [root lin vel (world), root ang vel (pelvis frame, MuJoCo qvel), 29 dof vel].

    Central differences (one-sided at the ends). The root angular velocity is expressed in
    the pelvis frame to match MuJoCo ``qvel`` and the holosoma retargeting writer
    (``convert_data_format_mj.py`` stores ``robot_data.qvel``). The holosoma loader strips
    the 6 root columns; the pelvis velocity it uses comes from ``body_*_vel_w``.
    """
    jp = np.asarray(joint_pos, dtype=np.float64)
    lin = fkm.finite_difference(jp[:, :3], fps)
    ang_w = fkm.angular_velocity_world(jp[:, 3:7], fps)
    ang_b = fkm.quat_rotate_inverse(jp[:, 3:7], ang_w)
    dof = fkm.finite_difference(jp[:, 7:], fps)
    return np.concatenate([lin, ang_b, dof], axis=1)


def body_velocities(body_pos_w: np.ndarray, body_quat_w: np.ndarray, fps: float) -> tuple[np.ndarray, np.ndarray]:
    """World-frame linear (gradient) and angular (quaternion increment) body velocities."""
    lin = fkm.finite_difference(body_pos_w, fps)
    ang = fkm.angular_velocity_world(body_quat_w, fps)
    return lin, ang


_LEG_COLS = np.asarray([fkm.ROOT_QPOS + i for i in LEG_DOF_IDX], dtype=np.int64)


def apply_legs_default_standing(joint_pos: np.ndarray) -> np.ndarray:
    """Re-pose a clip to the default stance: legs = holosoma defaults, root z = 0.76, root quat yaw-only.

    Root x/y and heading are preserved (HERO IK clips have root at the origin anyway); the
    waist and arms are untouched. Returns a new (T, 36) float64 array (wxyz root quaternion).
    """
    jp = np.asarray(joint_pos, dtype=np.float64).copy()
    jp[:, 2] = DEFAULT_STANDING_ROOT_Z
    q = fkm.quat_normalize(jp[:, 3:7])
    yaw = np.arctan2(2.0 * (q[:, 0] * q[:, 3] + q[:, 1] * q[:, 2]), 1.0 - 2.0 * (q[:, 2] ** 2 + q[:, 3] ** 2))
    jp[:, 3:7] = np.stack([np.cos(yaw / 2), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2)], axis=1)
    jp[:, _LEG_COLS] = np.asarray(DEFAULT_STANDING_LEG_ANGLES, dtype=np.float64)[None]
    return jp


# --------------------------------------------------------------------------------------
# payload conversion
# --------------------------------------------------------------------------------------


def _str_arr(s: str) -> np.ndarray:
    return np.asarray(str(s))


def convert_payload(
    data: Mapping[str, Any],
    meta: ClipMeta,
    *,
    parent_id: str,
    fk: fkm.G1Dex3FK | None = None,
    validate: bool = True,
) -> tuple[dict[str, np.ndarray], dict[str, Any]]:
    """Convert a loaded clip dict -> (output arrays, receipt). Pure function apart from the FK object."""
    t_start = time.perf_counter()
    fk = fk or fkm.get_fk(floor_z=meta.floor_z)
    family = detect_family(data)
    canon = extract_canonical(data, family)
    frames_in, fps_in = canon.frames, canon.fps
    if frames_in < 2:
        raise SchemaError(f"clip has {frames_in} frame(s); need >= 2")
    canon = resample_to_fps(canon, float(meta.target_fps))
    fps = float(meta.target_fps)
    if meta.legs_default_standing:
        canon.joint_pos = apply_legs_default_standing(canon.joint_pos)
        canon.joint_vel = None  # leg columns changed -> recompute
        canon.body_pos_w = canon.body_quat_w = canon.body_lin_vel_w = canon.body_ang_vel_w = None
        canon.notes.append(f"legs re-posed to default stance, root z={DEFAULT_STANDING_ROOT_Z:g} yaw-only (legs_default_standing)")
    jp = canon.joint_pos
    T = canon.frames

    t_fk = time.perf_counter()
    fk_full = fk.fk_all(jp)
    passthrough = canon.body_pos_w is not None and canon.body_quat_w is not None
    if passthrough:
        body_pos_w, body_quat_w = canon.body_pos_w.copy(), canon.body_quat_w.copy()
        # Derived bodies (foot contact points) follow the training-URDF offset in every clip;
        # native holosoma clips store them coincident with the ankle (loader aliases them anyway).
        stored_derived = body_pos_w[:, fk.derived_slots].copy()
        fk.apply_derived(body_pos_w, body_quat_w)
        derived_shift = float(np.max(np.linalg.norm(body_pos_w[:, fk.derived_slots] - stored_derived, axis=-1))) if len(fk.derived_slots) else 0.0
        if derived_shift > 1.0e-4:
            canon.notes.append(f"{'/'.join(fk.derived_body_names)} re-derived from ankle pose (moved {derived_shift:.4f} m)")
        if canon.body_lin_vel_w is not None and canon.body_ang_vel_w is not None:
            body_lin_vel_w, body_ang_vel_w = canon.body_lin_vel_w.copy(), canon.body_ang_vel_w.copy()
            if derived_shift > 1.0e-4:
                lin_d, ang_d = body_velocities(body_pos_w[:, fk.derived_slots], body_quat_w[:, fk.derived_slots], fps)
                body_lin_vel_w[:, fk.derived_slots] = lin_d
                body_ang_vel_w[:, fk.derived_slots] = ang_d
        else:
            body_lin_vel_w, body_ang_vel_w = body_velocities(body_pos_w, body_quat_w, fps)
            canon.notes.append("body velocities recomputed by finite differences")
        fk_pelvis_err = float(
            np.max(np.linalg.norm(fk_full["body_pos_w"][:, fk.shared_slots] - body_pos_w[:, fk.shared_slots], axis=-1))
        )
    else:
        body_pos_w, body_quat_w = fk_full["body_pos_w"], fk_full["body_quat_w"]
        body_lin_vel_w, body_ang_vel_w = body_velocities(body_pos_w, body_quat_w, fps)
        fk_pelvis_err = 0.0
    joint_vel = canon.joint_vel if canon.joint_vel is not None else joint_vel_from_joint_pos(jp, fps)
    ext = fk.hero_extension_keys(jp, fk=fk_full)
    fk_seconds = time.perf_counter() - t_fk

    hand_err_max: list[float] | None = None
    hand_err_mean: list[float] | None = None
    if canon.hand_targets_w is not None and not meta.legs_default_standing:
        err = np.linalg.norm(fk_full["palm_pos_w"] - canon.hand_targets_w, axis=-1)  # (T, 2)
        hand_err_max = [float(v) for v in err.max(axis=0)]
        hand_err_mean = [float(v) for v in err.mean(axis=0)]
        if meta.hand_target_tol_m is not None and max(hand_err_max) > float(meta.hand_target_tol_m):
            raise SchemaError(
                f"FK Dex3 palm point deviates from {HAND_TARGETS_KEY} by up to {max(hand_err_max):.4f} m "
                f"(left {hand_err_max[0]:.4f}, right {hand_err_max[1]:.4f}) > hand_target_tol_m {meta.hand_target_tol_m:g}"
            )
    elif canon.hand_targets_w is not None:
        canon.notes.append(f"{HAND_TARGETS_KEY} check skipped (legs_default_standing re-poses the root)")
    elif meta.hand_target_tol_m is not None:
        canon.notes.append(f"{HAND_TARGETS_KEY} absent -> hand-target check skipped")

    has_object = bool(meta.keep_object and canon.object_pos_w is not None)
    if canon.object_pos_w is not None and not meta.keep_object:
        canon.notes.append("dynamic object track dropped (keep_object=False)")
    write_mask = bool(has_object and meta.keep_attached_mask and canon.attached_mask is not None)
    if meta.keep_attached_mask and not write_mask:
        canon.notes.append(
            f"{ATTACHED_MASK_KEY} requested but " + ("absent in the source" if canon.attached_mask is None else "no kept dynamic object") + " -> not written"
        )

    out: dict[str, np.ndarray] = {
        "fps": np.asarray(int(meta.target_fps), dtype=np.int64),
        "joint_pos": jp.astype(np.float32),
        "joint_vel": np.asarray(joint_vel).astype(np.float32),
        "body_pos_w": np.asarray(body_pos_w).astype(np.float32),
        "body_quat_w": np.asarray(body_quat_w).astype(np.float32),
        "body_lin_vel_w": np.asarray(body_lin_vel_w).astype(np.float32),
        "body_ang_vel_w": np.asarray(body_ang_vel_w).astype(np.float32),
        "body_names": np.asarray(HOLOSOMA_BODY_NAMES_32),
        "joint_names": np.asarray(DOF_NAMES),
        "ee_pos_pelvis": ext["ee_pos_pelvis"].astype(np.float32),
        "ee_quat_pelvis": ext["ee_quat_pelvis"].astype(np.float32),
        "ee_pos_pelvis_zero_waist": ext["ee_pos_pelvis_zero_waist"].astype(np.float32),
        "ee_quat_pelvis_zero_waist": ext["ee_quat_pelvis_zero_waist"].astype(np.float32),
        "h_ref": ext["h_ref"].astype(np.float32),
        "source_tag": _str_arr(meta.source_tag),
        "parent_id": _str_arr(parent_id),
        "license_class": _str_arr(meta.license_class),
        "has_object": np.asarray(has_object, dtype=np.bool_),
    }
    if has_object:
        assert canon.object_pos_w is not None and canon.object_quat_w is not None
        out["object_pos_w"] = canon.object_pos_w.astype(np.float32)
        out["object_quat_w"] = canon.object_quat_w.astype(np.float32)
        out["object_lin_vel_w"] = fkm.finite_difference(canon.object_pos_w, fps).astype(np.float32)
        if canon.box_size is not None:
            out["box_size"] = np.asarray(canon.box_size, dtype=np.float32).reshape(3)
        if write_mask:
            assert canon.attached_mask is not None
            out[ATTACHED_MASK_KEY] = canon.attached_mask.astype(np.uint8).reshape(T)
    if validate:
        assert_valid(out, expected_fps=meta.target_fps)

    attached_range = None
    if write_mask:
        on = np.flatnonzero(out[ATTACHED_MASK_KEY])
        attached_range = [int(on[0]), int(on[-1]) + 1] if on.size else None
    receipt: dict[str, Any] = {
        "family": family,
        "frames_in": frames_in,
        "fps_in": fps_in,
        "frames_out": T,
        "fps_out": int(meta.target_fps),
        "has_object": has_object,
        "body_passthrough": passthrough,
        "fk_vs_stored_body_pos_max_m": fk_pelvis_err if passthrough else None,
        "h_ref_min": float(ext["h_ref"].min()),
        "h_ref_max": float(ext["h_ref"].max()),
        "source_tag": meta.source_tag,
        "parent_id": parent_id,
        "license_class": meta.license_class,
        "legs_default_standing": bool(meta.legs_default_standing),
        "attached_mask": write_mask,
        "attached_range": attached_range,  # [start, end) frames of the written mask (None = not written / never on)
        "hand_target_err_max_m": hand_err_max,  # [left, right] or None when the source has no authored_hand_targets_w
        "hand_target_err_mean_m": hand_err_mean,
        "hand_target_tol_m": meta.hand_target_tol_m,
        "notes": list(canon.notes),
        "fk_seconds": round(fk_seconds, 4),
        "seconds": round(time.perf_counter() - t_start, 4),
    }
    return out, receipt


def sha256_file(path: str | Path, chunk: int = 1 << 20) -> str:
    """Hex sha256 of a file (for provenance and duplicate detection)."""
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


_META_FIELDS = (
    "source_tag", "license_class", "parent_id", "keep_object", "target_fps", "floor_z", "legs_default_standing",
    "keep_attached_mask", "hand_target_tol_m",
)


def coerce_meta(meta: ClipMeta | Mapping[str, Any]) -> ClipMeta:
    """Accept ClipMeta or a dictionary of metadata; retain extra caller fields in ClipMeta.extra."""
    if isinstance(meta, ClipMeta):
        return meta
    if not isinstance(meta, Mapping):
        raise TypeError(f"meta must be ClipMeta or a mapping, got {type(meta).__name__}")
    kw: dict[str, Any] = {k: meta[k] for k in _META_FIELDS if k in meta and meta[k] is not None}
    if "keep_object" not in kw and "object_policy" in meta:
        kw["keep_object"] = str(meta["object_policy"]) == "keep"
    if "target_fps" in kw:
        kw["target_fps"] = int(round(float(kw["target_fps"])))
    kw["extra"] = {k: v for k, v in meta.items() if k not in _META_FIELDS and k != "object_policy"}
    return ClipMeta(**kw)


def convert_clip(
    path: str | Path,
    out_path: str | Path,
    meta: ClipMeta | Mapping[str, Any],
    *,
    fk: fkm.G1Dex3FK | None = None,
    overwrite: bool = False,
) -> dict[str, Any]:
    """Convert one file -> ``out_path`` (``np.savez_compressed``, atomic replace). Returns a receipt dict.

    ``meta.parent_id`` None -> the input file stem. Existing outputs are skipped unless
    ``overwrite`` (receipt ``status="skipped"``).
    """
    path = Path(path)
    out_path = Path(out_path)
    meta = coerce_meta(meta)
    if out_path.exists() and not overwrite:
        return {"source": str(path), "output": str(out_path), "status": "skipped"}
    data = read_npz(path)
    parent_id = meta.parent_id if meta.parent_id else path.stem
    payload, receipt = convert_payload(data, meta, parent_id=parent_id, fk=fk)
    receipt["source_sha256"] = sha256_file(path)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    tmp = out_path.with_name(f".{out_path.stem}.tmp-{os.getpid()}.npz")
    try:
        np.savez_compressed(tmp, **payload)
        os.replace(tmp, out_path)
    finally:
        if tmp.exists():
            tmp.unlink()
    receipt.update({"source": str(path), "output": str(out_path), "status": "ok", "bytes": out_path.stat().st_size})
    return receipt


# --------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------

_WORKER: dict[str, Any] = {}


def _worker_init(scene_xml: str, floor_z: float) -> None:
    _WORKER["fk"] = fkm.get_fk(scene_xml, floor_z)


def _worker_convert(job: tuple[str, str, dict[str, Any], str, bool]) -> dict[str, Any]:
    src, dst, meta_kwargs, parent_rule, overwrite = job
    try:
        meta = ClipMeta(**meta_kwargs)
        meta.parent_id = make_parent_id(Path(src).stem, parent_rule)
        fk = _WORKER.get("fk")
        return convert_clip(src, dst, meta, fk=fk, overwrite=overwrite)
    except Exception as exc:  # noqa: BLE001 - per-clip report, keep the batch going
        return {"source": src, "output": dst, "status": "error", "error": f"{type(exc).__name__}: {exc}"}


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    single = p.add_argument_group("single clip")
    single.add_argument("--clip", type=Path)
    single.add_argument("--out", type=Path)
    batch = p.add_argument_group("directory")
    batch.add_argument("--in-dir", type=Path)
    batch.add_argument("--out-dir", type=Path)
    batch.add_argument("--glob", default="*.npz", help="glob relative to --in-dir (e.g. 'shards/*/clips/*.npz')")
    batch.add_argument("--jobs", type=int, default=1)
    batch.add_argument("--limit", type=int, default=None, help="convert at most N clips (smoke runs)")
    p.add_argument("--source-tag", required=True)
    p.add_argument("--license-class", default="unknown", choices=LICENSE_CLASSES)
    p.add_argument("--parent-id-rule", default="stem", help="'stem' or 'strip_suffix:<regex>'")
    p.add_argument("--keep-object", action="store_true", help="keep dynamic object tracks (has_object=True)")
    p.add_argument(
        "--legs-default-standing",
        action="store_true",
        help="re-pose legs to the default stance + root z=0.76 before FK (HERO-style IK clips, source_tag hero_ik)",
    )
    p.add_argument("--keep-attached-mask", action="store_true", help="pass the source's attached_mask (T,) through as uint8 (kept dynamic object only)")
    p.add_argument("--hand-target-tol", type=float, default=None, metavar="M", help="fail a clip whose FK palm point deviates from authored_hand_targets_w by more than M metres")
    p.add_argument("--overwrite", action="store_true")
    p.add_argument("--target-fps", type=int, default=50)
    p.add_argument("--floor-z", type=float, default=0.0)
    p.add_argument("--scene-xml", type=Path, default=fkm.DEFAULT_SCENE_XML)
    p.add_argument("--receipts", type=Path, default=None, help="write JSON-lines receipts here (default: stdout)")
    p.add_argument("--summary-json", type=Path, default=None, help="write one aggregate JSON summary of the batch here")
    return p


def summarize_receipts(receipts: list[dict[str, Any]], *, seconds: float, args: argparse.Namespace | None = None) -> dict[str, Any]:
    """Aggregate per-clip receipts into the batch summary written by ``--summary-json``."""
    counts = {"ok": 0, "skipped": 0, "error": 0}
    families: dict[str, int] = {}
    frames_out = 0
    fk_seconds = 0.0
    has_object = 0
    errors: list[dict[str, str]] = []
    for r in receipts:
        counts[r.get("status", "error")] = counts.get(r.get("status", "error"), 0) + 1
        if r.get("status") == "ok":
            families[r["family"]] = families.get(r["family"], 0) + 1
            frames_out += int(r.get("frames_out", 0))
            fk_seconds += float(r.get("fk_seconds", 0.0))
            has_object += int(bool(r.get("has_object", False)))
        elif r.get("status") == "error":
            errors.append({"source": r.get("source", ""), "error": r.get("error", "")})
    out: dict[str, Any] = {
        "total": len(receipts),
        "counts": counts,
        "families": families,
        "frames_out": frames_out,
        "hours_out": round(frames_out / 50.0 / 3600.0, 4),
        "has_object": has_object,
        "fk_seconds_sum": round(fk_seconds, 3),
        "wall_seconds": round(seconds, 3),
        "errors": errors[:200],
    }
    if args is not None:
        out["args"] = {
            k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items() if k not in ("receipts", "summary_json")
        }
    return out


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    meta_kwargs = dict(
        source_tag=args.source_tag,
        license_class=args.license_class,
        keep_object=bool(args.keep_object),
        target_fps=int(args.target_fps),
        floor_z=float(args.floor_z),
        legs_default_standing=bool(args.legs_default_standing),
        keep_attached_mask=bool(args.keep_attached_mask),
        hand_target_tol_m=args.hand_target_tol,
    )
    make_parent_id("probe", args.parent_id_rule)  # fail fast on a bad rule
    sink = open(args.receipts, "w") if args.receipts else sys.stdout

    all_receipts: list[dict[str, Any]] = []

    def emit(r: dict[str, Any]) -> None:
        all_receipts.append(r)
        print(json.dumps(r), file=sink, flush=True)

    def write_summary(seconds: float) -> None:
        if args.summary_json is not None:
            args.summary_json.parent.mkdir(parents=True, exist_ok=True)
            args.summary_json.write_text(json.dumps(summarize_receipts(all_receipts, seconds=seconds, args=args), indent=1))

    t0 = time.perf_counter()
    try:
        if args.clip is not None:
            if args.out is None:
                raise SystemExit("--clip requires --out")
            _worker_init(str(args.scene_xml), float(args.floor_z))
            r = _worker_convert((str(args.clip), str(args.out), meta_kwargs, args.parent_id_rule, args.overwrite))
            emit(r)
            write_summary(time.perf_counter() - t0)
            return 0 if r["status"] != "error" else 1

        if args.in_dir is None or args.out_dir is None:
            raise SystemExit("pass --clip/--out or --in-dir/--out-dir")
        clips = sorted(args.in_dir.glob(args.glob))
        if args.limit is not None:
            clips = clips[: args.limit]
        if not clips:
            raise SystemExit(f"no clips matching {args.glob!r} under {args.in_dir}")
        jobs = [
            (str(c), str(args.out_dir / c.relative_to(args.in_dir)), meta_kwargs, args.parent_id_rule, args.overwrite)
            for c in clips
        ]
        counts = {"ok": 0, "skipped": 0, "error": 0}
        if args.jobs <= 1:
            _worker_init(str(args.scene_xml), float(args.floor_z))
            for job in jobs:
                r = _worker_convert(job)
                counts[r["status"]] += 1
                emit(r)
        else:
            with cf.ProcessPoolExecutor(
                max_workers=args.jobs, initializer=_worker_init, initargs=(str(args.scene_xml), float(args.floor_z))
            ) as pool:
                for r in pool.map(_worker_convert, jobs, chunksize=4):
                    counts[r["status"]] += 1
                    emit(r)
        dt = time.perf_counter() - t0
        write_summary(dt)
        print(
            f"npz_convert: {counts['ok']} ok, {counts['skipped']} skipped, {counts['error']} failed "
            f"of {len(jobs)} in {dt:.1f}s -> {args.out_dir}",
            file=sys.stderr,
        )
        return 1 if counts["error"] else 0
    finally:
        if sink is not sys.stdout:
            sink.close()


if __name__ == "__main__":
    sys.exit(main())
