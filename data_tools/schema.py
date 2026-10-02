"""HERO-extended HoloSoma motion NPZ schema.

Required motion arrays are fps, joint_pos (T,36), joint_vel (T,35), the four
world-frame body pose/velocity arrays, body_names, and joint_names. Body arrays
use the 32-body order in hero_isaacsim.constants. All stored quaternions are wxyz;
runtime tensors convert to xyzw when loaded.

HERO adds palm poses in the pelvis frame, zero-waist palm poses, reference root
height, source_tag, parent_id, license_class, and has_object. Optional object
channels are retained for format compatibility. The default AMASS preparation
writes only robot motions, with has_object=False.

An optional kinematic contract certifies link-origin world velocities and
finite-difference conventions. Pose edits must update its hashes and stamps."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np


from hero_isaacsim.constants import (  # noqa: E402
    DOF_NAMES,
    EE_BODY_NAMES,
    FOOT_CONTACT_POINT_BODY_NAMES,
    FOOT_CONTACT_POINT_OFFSET,
    FOOT_CONTACT_POINT_PARENTS,
    HOLOSOMA_BODY_NAMES_32,
    INIT_POS_Z,
    LEG_DOF_IDX,
    LOWER_DOF_IDX,
    PALM_BODY_NAMES,
    PALM_OFFSET,
    UPPER_DOF_IDX,
    WAIST_DOF_IDX,
)

#: ``left/right_foot_contact_point`` -> parent ankle link. The Dex3 FK scenes in this repo have no
#: such bodies, so the converter derives them from ``FOOT_CONTACT_POINT_OFFSET`` (holosoma
#: g1_29dof.urdf: ``pos="0 0 -0.037"``). The holosoma loader aliases them to the ankle anyway.
FOOT_CONTACT_POINT_PARENT: dict[str, str] = dict(zip(FOOT_CONTACT_POINT_BODY_NAMES, FOOT_CONTACT_POINT_PARENTS))

DEFAULT_STANDING_LEG_ANGLES: tuple[float, ...] = (
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # left  hip_pitch, hip_roll, hip_yaw, knee, ankle_pitch, ankle_roll
    -0.312, 0.0, 0.0, 0.669, -0.363, 0.0,  # right
)
DEFAULT_STANDING_ROOT_Z: float = float(INIT_POS_Z)
assert len(DEFAULT_STANDING_LEG_ANGLES) == len(LEG_DOF_IDX) == 12

# --------------------------------------------------------------------------------------
# npz key sets
# --------------------------------------------------------------------------------------

TARGET_FPS: int = 50

#: Mirrors ``holosoma.managers.command.terms.wbt.MotionLoader._REQUIRED_KEYS``.
REQUIRED_KEYS: frozenset[str] = frozenset(
    {
        "fps",
        "joint_pos",
        "joint_vel",
        "body_pos_w",
        "body_quat_w",
        "body_lin_vel_w",
        "body_ang_vel_w",
        "body_names",
        "joint_names",
    }
)
#: The loader reads all three unconditionally as soon as ``object_pos_w`` exists.
OBJECT_KEYS: tuple[str, ...] = ("object_pos_w", "object_quat_w", "object_lin_vel_w")
EXTENSION_KEYS: tuple[str, ...] = (
    "ee_pos_pelvis",
    "ee_quat_pelvis",
    "ee_pos_pelvis_zero_waist",
    "ee_quat_pelvis_zero_waist",
    "h_ref",
    "source_tag",
    "parent_id",
    "license_class",
    "has_object",
)
LICENSE_CLASSES: tuple[str, ...] = ("apache", "cc-by", "research-only", "nc", "unknown")
ATTACHED_MASK_KEY: str = "attached_mask"
OPTIONAL_OBJECT_KEYS: tuple[str, ...] = (ATTACHED_MASK_KEY,)

ALL_OUTPUT_KEYS: frozenset[str] = REQUIRED_KEYS | set(EXTENSION_KEYS) | set(OBJECT_KEYS) | {"box_size"} | set(OPTIONAL_OBJECT_KEYS)


class SchemaError(ValueError):
    """Raised when an npz payload violates the HERO-extended holosoma schema."""


@dataclass
class ClipMeta:
    """Per-clip metadata supplied by the caller (CLI or a corpus converter)."""

    source_tag: str
    license_class: str = "unknown"
    parent_id: str | None = None  # None -> file stem at convert time
    keep_object: bool = False  # keep a *dynamic* object trajectory if the source has one
    target_fps: int = TARGET_FPS
    floor_z: float = 0.0  # h_ref = pelvis z - floor_z
    #: Overwrite the 12 leg joints with the default stance and root z with 0.76 before FK (hero_ik clips)
    legs_default_standing: bool = False
    keep_attached_mask: bool = False
    hand_target_tol_m: float | None = None
    extra: dict[str, Any] = field(default_factory=dict)  # free-form, not written to the npz

    def __post_init__(self) -> None:
        if not self.source_tag or not isinstance(self.source_tag, str):
            raise SchemaError("ClipMeta.source_tag must be a non-empty string")
        if self.license_class not in LICENSE_CLASSES:
            raise SchemaError(f"license_class {self.license_class!r} not in {LICENSE_CLASSES}")
        if int(self.target_fps) != self.target_fps or self.target_fps <= 0:
            raise SchemaError(f"target_fps must be a positive integer, got {self.target_fps!r}")
        self.target_fps = int(self.target_fps)
        if self.hand_target_tol_m is not None:
            tol = float(self.hand_target_tol_m)
            if not np.isfinite(tol) or tol <= 0.0:
                raise SchemaError(f"hand_target_tol_m must be a positive finite distance in metres, got {self.hand_target_tol_m!r}")
            self.hand_target_tol_m = tol


# --------------------------------------------------------------------------------------
# helpers
# --------------------------------------------------------------------------------------


def decode_names(values: Any) -> list[str]:
    """Decode a str/bytes array to ``list[str]`` (same rule as holosoma ``_decode_npz_names``)."""
    raw = values.tolist() if hasattr(values, "tolist") else list(values)
    if isinstance(raw, (str, bytes)):
        raw = [raw]
    return [v.decode("utf-8") if isinstance(v, bytes) else str(v) for v in raw]


def npz_scalar_str(value: Any) -> str:
    """Return the python ``str`` held by a 0-d ``<U``/``S`` array (or a plain str)."""
    arr = np.asarray(value)
    if arr.size != 1:
        raise SchemaError(f"expected a scalar string, got shape {arr.shape}")
    item = arr.reshape(()).item()
    return item.decode("utf-8") if isinstance(item, bytes) else str(item)


def scalar_fps(value: Any, *, default: float | None = None) -> float:
    """Return ``fps`` as a float from ``()``, ``(1,)`` arrays or python numbers.

    ``default`` is used when ``value`` is None (qpos43 thin files omit fps -> 50).
    """
    if value is None:
        if default is None:
            raise SchemaError("fps missing and no default given")
        return float(default)
    arr = np.asarray(value)
    if arr.size != 1:
        raise SchemaError(f"fps must be a scalar, got shape {arr.shape}")
    fps = float(arr.reshape(()))
    if not np.isfinite(fps) or fps <= 0.0:
        raise SchemaError(f"fps must be finite and positive, got {fps!r}")
    return fps


def read_npz(path: str | Path) -> dict[str, np.ndarray]:
    """Load every array of an npz into a dict (pickle disabled unless the file needs it)."""
    path = Path(path)
    try:
        with np.load(path, allow_pickle=False) as z:
            return {k: z[k] for k in z.files}
    except ValueError:
        # Object-array metadata requires pickle decoding; load only trusted NPZ files.
        with np.load(path, allow_pickle=True) as z:
            return {k: z[k] for k in z.files}


def _is_unit_quat(q: np.ndarray, atol: float = 1.0e-3) -> bool:
    return bool(np.all(np.abs(np.linalg.norm(q, axis=-1) - 1.0) <= atol))


def _finite(a: np.ndarray) -> bool:
    return bool(np.isfinite(np.asarray(a, dtype=np.float64)).all())


# --------------------------------------------------------------------------------------
# validator
# --------------------------------------------------------------------------------------


def validate_holosoma_npz(
    data: Mapping[str, Any],
    *,
    require_extension: bool = True,
    expected_fps: float | None = TARGET_FPS,
    robot_body_names: Iterable[str] | None = HOLOSOMA_BODY_NAMES_32,
    robot_joint_names: Iterable[str] | None = DOF_NAMES,
    quat_atol: float = 1.0e-3,
) -> list[str]:
    """Return a list of schema violations (empty list == valid).

    Replicates the checks of ``MotionLoader._load_data_from_motion_npz`` (required keys,
    scalar positive fps equal to the control fps, ``joint_pos == len(joint_names)+7``,
    ``joint_vel == len(joint_names)+6``, body count == ``len(body_names)``, every robot
    body/joint name present in the file, object triple present together) and adds the
    HERO extension-key contract when ``require_extension`` is True.
    """
    problems: list[str] = []
    keys = set(data.keys())

    missing = REQUIRED_KEYS - keys
    if missing:
        problems.append(f"missing required keys: {sorted(missing)}")
        return problems  # nothing else can be checked reliably

    # --- fps ------------------------------------------------------------------------
    try:
        fps = scalar_fps(data["fps"])
        if expected_fps is not None and not np.isclose(fps, float(expected_fps), rtol=0.0, atol=1.0e-4):
            problems.append(f"fps={fps:g} != expected {float(expected_fps):g}")
    except SchemaError as exc:
        problems.append(str(exc))

    # --- names -----------------------------------------------------------------------
    body_names = decode_names(data["body_names"])
    joint_names = decode_names(data["joint_names"])
    if robot_body_names is not None:
        absent = [n for n in robot_body_names if n not in body_names]
        if absent:
            problems.append(f"body_names lacks robot bodies {absent}")
    if robot_joint_names is not None:
        absent = [n for n in robot_joint_names if n not in joint_names]
        if absent:
            problems.append(f"joint_names lacks robot joints {absent}")
        if list(robot_joint_names) != joint_names and not absent:
            problems.append("joint_names are a permutation of DOF_NAMES (loader tolerates, contract does not)")

    # --- column / shape checks --------------------------------------------------------
    jp = np.asarray(data["joint_pos"])
    jv = np.asarray(data["joint_vel"])
    bp = np.asarray(data["body_pos_w"])
    bq = np.asarray(data["body_quat_w"])
    blv = np.asarray(data["body_lin_vel_w"])
    bav = np.asarray(data["body_ang_vel_w"])
    if jp.ndim != 2:
        problems.append(f"joint_pos must be 2-D, got {jp.shape}")
        return problems
    T = jp.shape[0]
    if T < 2:
        problems.append(f"clip has {T} frame(s); need >= 2")
    if jp.shape[1] != len(joint_names) + 7:
        problems.append(f"joint_pos columns {jp.shape[1]} != len(joint_names)+7 = {len(joint_names) + 7}")
    if jv.shape != (T, len(joint_names) + 6):
        problems.append(f"joint_vel shape {jv.shape} != {(T, len(joint_names) + 6)}")
    B = len(body_names)
    for name, arr, last in (("body_pos_w", bp, 3), ("body_quat_w", bq, 4), ("body_lin_vel_w", blv, 3), ("body_ang_vel_w", bav, 3)):
        if arr.shape != (T, B, last):
            problems.append(f"{name} shape {arr.shape} != {(T, B, last)}")
    for name in ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w"):
        if not _finite(data[name]):
            problems.append(f"{name} has NaN/Inf")
    if jp.shape[1] >= 7 and _finite(jp) and not _is_unit_quat(jp[:, 3:7], quat_atol):
        problems.append("joint_pos[:, 3:7] root quaternion is not unit (wxyz expected)")
    if bq.ndim == 3 and bq.shape[-1] == 4 and _finite(bq) and not _is_unit_quat(bq, quat_atol):
        problems.append("body_quat_w is not unit (wxyz expected)")
    if "pelvis" in body_names and bp.ndim == 3 and bp.shape[0] == T and jp.shape[1] >= 3:
        pel = body_names.index("pelvis")
        if bp.shape[1] > pel and not np.allclose(bp[:, pel], jp[:, :3], atol=1.0e-4):
            problems.append("body_pos_w[pelvis] != joint_pos[:, :3] (root pos)")

    # --- object channel -----------------------------------------------------------------
    has_object_key = "object_pos_w" in keys
    if has_object_key:
        op = np.asarray(data["object_pos_w"])
        if op.ndim == 1:
            problems.append("object_pos_w is a static (3,) target; must be stripped (has_object=False)")
        else:
            absent = [k for k in OBJECT_KEYS if k not in keys]
            if absent:
                problems.append(f"object_pos_w present but {absent} missing (loader reads all three)")
            else:
                oq = np.asarray(data["object_quat_w"])
                ov = np.asarray(data["object_lin_vel_w"])
                if op.shape != (T, 3):
                    problems.append(f"object_pos_w shape {op.shape} != {(T, 3)}")
                if oq.shape != (T, 4):
                    problems.append(f"object_quat_w shape {oq.shape} != {(T, 4)}")
                elif _finite(oq) and not _is_unit_quat(oq, quat_atol):
                    problems.append("object_quat_w is not unit (wxyz expected)")
                if ov.shape != (T, 3):
                    problems.append(f"object_lin_vel_w shape {ov.shape} != {(T, 3)}")
                for k in OBJECT_KEYS:
                    if not _finite(data[k]):
                        problems.append(f"{k} has NaN/Inf")
    if "box_size" in keys:
        bs = np.asarray(data["box_size"], dtype=np.float64)
        if bs.shape != (3,) or not np.all(np.isfinite(bs)) or np.any(bs <= 0):
            problems.append(f"box_size must be a positive (3,) vector, got {bs!r}")
    if ATTACHED_MASK_KEY in keys:
        am = np.asarray(data[ATTACHED_MASK_KEY])
        if am.shape != (T,):
            problems.append(f"{ATTACHED_MASK_KEY} shape {am.shape} != {(T,)}")
        elif am.dtype.kind not in "biu":
            problems.append(f"{ATTACHED_MASK_KEY} dtype {am.dtype} must be bool / integer (uint8 {{0, 1}})")
        elif not np.all(np.isin(am, (0, 1))):
            problems.append(f"{ATTACHED_MASK_KEY} values must be 0 / 1")
        if not has_object_key or (has_object_key and np.asarray(data["object_pos_w"]).ndim == 1):
            problems.append(f"{ATTACHED_MASK_KEY} present without a dynamic object track")

    if not problems and any(k.startswith("kinematic_") for k in keys):
        from data_tools.kinematic_contract import contract_problems
        problems.extend(contract_problems(data))

    if not require_extension:
        return problems

    # --- HERO extension keys --------------------------------------------------------------
    missing_ext = [k for k in EXTENSION_KEYS if k not in keys]
    if missing_ext:
        problems.append(f"missing extension keys: {missing_ext}")
        return problems
    for name, last in (("ee_pos_pelvis", 3), ("ee_quat_pelvis", 4), ("ee_pos_pelvis_zero_waist", 3), ("ee_quat_pelvis_zero_waist", 4)):
        arr = np.asarray(data[name])
        if arr.shape != (T, 2, last):
            problems.append(f"{name} shape {arr.shape} != {(T, 2, last)}")
        elif not _finite(arr):
            problems.append(f"{name} has NaN/Inf")
        elif last == 4 and not _is_unit_quat(arr, quat_atol):
            problems.append(f"{name} is not unit (wxyz expected)")
    h = np.asarray(data["h_ref"])
    if h.shape != (T,):
        problems.append(f"h_ref shape {h.shape} != {(T,)}")
    elif not _finite(h):
        problems.append("h_ref has NaN/Inf")
    for name in ("source_tag", "parent_id", "license_class"):
        try:
            s = npz_scalar_str(data[name])
            if not s:
                problems.append(f"{name} is empty")
            if name == "license_class" and s not in LICENSE_CLASSES:
                problems.append(f"license_class {s!r} not in {LICENSE_CLASSES}")
        except SchemaError as exc:
            problems.append(f"{name}: {exc}")
    ho = np.asarray(data["has_object"])
    if ho.shape != () or ho.dtype != np.bool_:
        problems.append(f"has_object must be a 0-d bool, got shape {ho.shape} dtype {ho.dtype}")
    else:
        if bool(ho) != has_object_key:
            problems.append(f"has_object={bool(ho)} but object_pos_w present={has_object_key}")
        if not bool(ho) and "box_size" in keys:
            problems.append("box_size present while has_object=False")
        if not bool(ho) and ATTACHED_MASK_KEY in keys:
            problems.append(f"{ATTACHED_MASK_KEY} present while has_object=False")
    return problems


def assert_valid(data: Mapping[str, Any], **kwargs: Any) -> None:
    """Raise :class:`SchemaError` listing every violation found by :func:`validate_holosoma_npz`."""
    problems = validate_holosoma_npz(data, **kwargs)
    if problems:
        raise SchemaError("; ".join(problems))


_STRIP_RULE = re.compile(r"^strip_suffix:(?P<regex>.+)$", re.DOTALL)


def make_parent_id(stem: str, rule: str = "stem") -> str:
    """Derive ``parent_id`` from a file stem.

    rules: ``"stem"`` (identity) or ``"strip_suffix:<regex>"`` (``re.sub(regex, "", stem)``,
    falling back to the stem when the result is empty).
    """
    if rule == "stem":
        return stem
    m = _STRIP_RULE.match(rule)
    if not m:
        raise SchemaError(f"unknown parent-id rule {rule!r}; use 'stem' or 'strip_suffix:<regex>'")
    out = re.sub(m.group("regex"), "", stem)
    return out or stem
