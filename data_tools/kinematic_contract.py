"""Explicit link-origin velocity contract for opt-in, separately built corpora.

Input clips may contain MuJoCo/Isaac centre-of-mass velocities beside
link-origin positions. Converting
to this contract derives velocities from the saved poses, preserving the motion
and avoiding guesses about each source model's inertias or derivative stencil.
Pose/FK disagreement is rejected rather than silently repaired.
"""
from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from data_tools import fk_mujoco as fkm
from data_tools.schema import DOF_NAMES, HOLOSOMA_BODY_NAMES_32, SchemaError, decode_names, scalar_fps

VERSION = 1
CONVENTIONS = {
    "kinematic_body_position_frame": "world_link_origin",
    "kinematic_body_linear_velocity_frame": "world_link_origin",
    "kinematic_body_angular_velocity_frame": "world",
    "kinematic_root_angular_velocity_frame": "pelvis",
    "kinematic_velocity_scheme": "central_fd_one_sided_edges",
}
POSE_KEYS = ("fps", "joint_names", "body_names", "joint_pos", "body_pos_w", "body_quat_w")


def pose_sha256(data: Mapping[str, Any]) -> str:
    """Bind the FK check to the exact saved pose and name arrays, including dtype."""
    digest = hashlib.sha256()
    for key in POSE_KEYS:
        a = np.ascontiguousarray(data[key])
        digest.update(f"{key}:{a.dtype.str}:{a.shape}:".encode())
        digest.update(a.tobytes())
    return digest.hexdigest()


def derive_velocities(data: Mapping[str, Any]) -> dict[str, np.ndarray]:
    # Local import avoids a converter/schema import cycle.
    from data_tools.npz_convert import body_velocities, joint_vel_from_joint_pos
    fps = scalar_fps(data["fps"])
    bp, bq = np.asarray(data["body_pos_w"], dtype=np.float64), np.asarray(data["body_quat_w"], dtype=np.float64)
    lv, av = body_velocities(bp, bq, fps)
    return {"joint_vel": joint_vel_from_joint_pos(np.asarray(data["joint_pos"], dtype=np.float64), fps).astype(np.float32),
            "body_lin_vel_w": lv.astype(np.float32), "body_ang_vel_w": av.astype(np.float32)}


def contract_problems(data: Mapping[str, Any], *, require: bool = True) -> list[str]:
    """Verify stamps, their exact pose binding, and all three velocity arrays.

    No MuJoCo instance is needed. This catches stale stamps when a downstream
    edit copies metadata without recomputing/rechecking the kinematics.
    """
    if "kinematic_contract_version" not in data:
        return ["missing kinematic_contract_version"] if require else []
    errors = []
    try:
        if np.asarray(data["kinematic_contract_version"]).item() != VERSION:
            errors.append("unsupported kinematic_contract_version")
        for key, value in CONVENTIONS.items():
            if key not in data or np.asarray(data[key]).item() != value:
                errors.append(f"{key} must be {value!r}")
        if decode_names(data["joint_names"]) != list(DOF_NAMES) or decode_names(data["body_names"]) != list(HOLOSOMA_BODY_NAMES_32):
            errors.append("kinematic contract requires canonical joint/body order")
        if str(np.asarray(data.get("kinematic_pose_sha256", "")).item()) != pose_sha256(data):
            errors.append("kinematic pose digest mismatch; FK stamp is stale")
        scene_sha = str(np.asarray(data.get("kinematic_fk_scene_sha256", "")).item())
        if len(scene_sha) != 64 or any(c not in "0123456789abcdef" for c in scene_sha):
            errors.append("invalid kinematic_fk_scene_sha256")
        for kind in ("position", "rotation"):
            unit = "m" if kind == "position" else "rad"
            value = float(np.asarray(data[f"kinematic_fk_{kind}_max_error_{unit}"]).item())
            tol = float(np.asarray(data[f"kinematic_fk_{kind}_tolerance_{unit}"]).item())
            if not np.isfinite([value, tol]).all() or value < 0 or tol <= 0 or value > tol:
                errors.append(f"kinematic FK {kind} did not pass its recorded tolerance")
        for key, expected in derive_velocities(data).items():
            value = np.asarray(data[key])
            if value.shape != expected.shape or not np.allclose(value, expected, rtol=2e-6, atol=2e-6):
                errors.append(f"{key} violates central finite-difference contract")
    except (KeyError, TypeError, ValueError, IndexError) as exc:
        errors.append(f"malformed kinematic contract: {exc}")
    return errors


def canonicalize_payload(data: Mapping[str, Any], fk: Any, *,
                         fk_position_tolerance_m: float = 0.002,
                         fk_rotation_tolerance_rad: float = 0.02) -> tuple[dict[str, Any], dict[str, Any]]:
    """Keep every pose/metadata array and replace only the three robot velocities.

    An inconsistent contract can be rebuilt by this operation, but
    body/joint pose disagreement fails the gate. No terminal zeroing or footlock
    occurs. The resulting speeds follow the source's actual terminal pose step.
    """
    from data_tools.schema import validate_holosoma_npz
    tolerances = (fk_position_tolerance_m, fk_rotation_tolerance_rad)
    if not np.isfinite(tolerances).all() or min(tolerances) <= 0:
        raise ValueError("FK tolerances must be positive and finite")
    source = {k: v for k, v in data.items() if not k.startswith("kinematic_")}
    problems = validate_holosoma_npz(source, require_extension=False, expected_fps=None)
    if problems:
        raise SchemaError("; ".join(problems))
    if decode_names(source["joint_names"]) != list(DOF_NAMES) or decode_names(source["body_names"]) != list(HOLOSOMA_BODY_NAMES_32):
        raise SchemaError("canonicalization requires canonical joint/body order")
    jp = np.asarray(source["joint_pos"], dtype=np.float64)
    bp, bq = np.asarray(source["body_pos_w"], dtype=np.float64), np.asarray(source["body_quat_w"], dtype=np.float64)
    expected_p, expected_q = fk.body_poses(jp)
    pe = float(np.linalg.norm(bp - expected_p, axis=-1).max())
    dot = np.abs(np.sum(fkm.quat_normalize(bq) * fkm.quat_normalize(expected_q), axis=-1))
    re = float((2 * np.arccos(np.clip(dot, 0, 1))).max())
    if pe > fk_position_tolerance_m or re > fk_rotation_tolerance_rad:
        raise SchemaError(f"pose/FK gate failed: position {pe:.6g} m, rotation {re:.6g} rad; no pose changed")
    out = dict(source)
    derived = derive_velocities(source)
    delta = {key: {"rms": float(np.sqrt(np.mean(np.square(np.asarray(source[key], dtype=float) - val)))),
                   "max_abs": float(np.max(np.abs(np.asarray(source[key], dtype=float) - val)))} for key, val in derived.items()}
    out.update(derived)
    out.update({key: np.asarray(value) for key, value in CONVENTIONS.items()})
    out.update(kinematic_contract_version=np.asarray(VERSION),
               kinematic_pose_sha256=np.asarray(pose_sha256(out)),
               kinematic_fk_scene_sha256=np.asarray(hashlib.sha256(Path(fk.scene_xml).read_bytes()).hexdigest()),
               kinematic_fk_position_max_error_m=np.asarray(pe),
               kinematic_fk_rotation_max_error_rad=np.asarray(re),
               kinematic_fk_position_tolerance_m=np.asarray(fk_position_tolerance_m),
               kinematic_fk_rotation_tolerance_rad=np.asarray(fk_rotation_tolerance_rad))
    problems = contract_problems(out)
    if problems:
        raise SchemaError("; ".join(problems))
    return out, {"contract_version": VERSION, "frames": len(jp), "fps": scalar_fps(out["fps"]),
                 "pose_sha256": pose_sha256(out), "fk_position_max_error_m": pe, "fk_rotation_max_error_rad": re,
                 "input_contract_version": np.asarray(data.get("kinematic_contract_version", 0)).item(),
                 "velocity_change": delta, "pose_arrays_unchanged": True}
