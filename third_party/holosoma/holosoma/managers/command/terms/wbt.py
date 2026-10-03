from __future__ import annotations

import json
import os
import re
import zipfile
from pathlib import Path
from typing import Any, List, Mapping, Sequence

import numpy as np
import torch
from loguru import logger

from holosoma.config_types.command import MotionConfig, NoiseToInitialPoseConfig
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.managers.command.base import CommandTermBase
from holosoma.utils.file_cache import cached_open
from holosoma.utils.path import resolve_data_file_path
from holosoma.utils.rotations import (
    get_euler_xyz,
    quat_apply,
    quat_conjugate,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_inverse,
    quat_mul,
    quat_to_angle_axis,
    slerp,
    yaw_quat,
)
from holosoma.utils.simulator_config import SimulatorType


#: ``CORPUS_MANIFEST.json`` ``kinematic_contract.schema`` prefix of a certified LINK-origin corpus
#: (``data_tools.corpus_contract.CORPUS_CONTRACT_SCHEMA`` = ``hero_link_origin_corpus/v1``).  Checked here with the
#: stdlib only: holosoma must not import hero_isaacsim / data_tools.
LINK_ORIGIN_CORPUS_SCHEMA_PREFIX = "hero_link_origin_corpus"
CORPUS_MANIFEST_NAME = "CORPUS_MANIFEST.json"
#: Held-out clips retain their velocity convention and do not inherit the parent's certificate.
HELD_OUT_DIR_NAME = "held_out"
#: The per-clip kinematic stamp every certified LINK-origin clip carries INSIDE its NPZ
#: (``data_tools.kinematic_contract.CONVENTIONS["kinematic_body_linear_velocity_frame"]``, written by
#: ``canonicalize_payload``); a legacy clip carries no such key.  Read only for clips no manifest decides
#: (:func:`clip_reference_root_velocity_frame`).
LINK_ORIGIN_CLIP_STAMP_KEY = "kinematic_body_linear_velocity_frame"
LINK_ORIGIN_CLIP_STAMP_VALUE = "world_link_origin"
#: What the reference root linear velocity (``MotionCommand.root_lin_vel_w``) measures: ``"com"`` = the body's
#: centre-of-mass velocity (legacy holosoma converter, ``mj_objectVelocity(mjOBJ_BODY)``), ``"link"`` = the link-origin
#: velocity (certified corpora).  IsaacLab's ``write_root_velocity_to_sim`` expects the COM velocity.
REFERENCE_ROOT_VELOCITY_FRAMES = ("com", "link")
#: The command-level frame of a motion source whose clips come in BOTH conventions (opt-in only, see
#: :data:`ALLOW_MIXED_VELOCITY_FRAMES_ENV`): the frame is then decided PER CLIP and only LINK clips are converted.
MIXED_REFERENCE_ROOT_VELOCITY_FRAME = "mixed"
#: ``=1`` allows a non-contract command to mix LINK-origin and COM clips.
#: Otherwise setup rejects mixed sources; contract commands always reject them.
ALLOW_MIXED_VELOCITY_FRAMES_ENV = "HERO_ALLOW_MIXED_VELOCITY_FRAMES"


def manifest_reference_root_velocity_frame(manifest: Path) -> str | None:
    """``"link"`` / ``"com"`` from ONE manifest file; ``None`` when there is no such file."""
    if not manifest.is_file():
        return None
    try:
        with manifest.open("r", encoding="utf-8") as stream:
            data = json.load(stream)
    except (OSError, ValueError) as exc:
        logger.warning("Unreadable {} ({}: {}); treating {} as a legacy (COM velocity) corpus", manifest, type(exc).__name__, exc, manifest.parent)
        return "com"
    contract = data.get("kinematic_contract") if isinstance(data, dict) else None
    schema = contract.get("schema") if isinstance(contract, dict) else None
    return "link" if isinstance(schema, str) and schema.startswith(LINK_ORIGIN_CORPUS_SCHEMA_PREFIX) else "com"


def clip_reference_root_velocity_frame(clip_file: str | Path) -> str | None:
    """``"link"`` / ``"com"`` from the clip's OWN NPZ kinematic stamp; ``None`` when the file cannot be read as an NPZ.

    A certified LINK-origin clip carries :data:`LINK_ORIGIN_CLIP_STAMP_KEY` = :data:`LINK_ORIGIN_CLIP_STAMP_VALUE` (the
    certified build stamps every convention key into the payload); a legacy clip carries no such key -> ``"com"``.  Only the
    zip directory and that one scalar member are read, never the motion arrays.  A stamp with another value is a convention
    the reset writers do not know: reported and passed through as ``"com"``.
    """
    path = os.path.expanduser(str(clip_file))
    try:
        with np.load(path, allow_pickle=False) as archive:
            if LINK_ORIGIN_CLIP_STAMP_KEY not in archive.files:
                return "com"
            value = str(np.asarray(archive[LINK_ORIGIN_CLIP_STAMP_KEY]).item())
    except (OSError, ValueError, EOFError, zipfile.BadZipFile) as exc:
        logger.warning("Cannot read the kinematic stamp of {} ({}: {})", path, type(exc).__name__, exc)
        return None
    if value == LINK_ORIGIN_CLIP_STAMP_VALUE:
        return "link"
    logger.warning(
        "{}: {} = {!r} is not a root-velocity convention the reset writers know; treating the clip as legacy (COM velocity)",
        path,
        LINK_ORIGIN_CLIP_STAMP_KEY,
        value,
    )
    return "com"


def top_level_clip_names(directory: Path) -> frozenset[str]:
    """The ``*.npz`` file names directly inside ``directory`` (empty when it cannot be listed).

    For a certified corpus this IS the certificate's inventory: ``kinematic_contract.scope == "top_level_training_clips"``
    and ``data_tools.corpus_contract.verify_corpus_contract`` hashes exactly these names into ``train_registry_sha256``;
    clips in sub-directories are outside the certificate.
    """
    try:
        with os.scandir(directory) as entries:
            return frozenset(entry.name for entry in entries if entry.name.endswith(".npz") and entry.is_file())
    except OSError:
        return frozenset()


def reference_root_velocity_frame_candidates(directory: Path) -> list[Path]:
    """The manifests that decide the velocity frame of the clips stored in ``directory``, most specific first.

    Check the directory's ``CORPUS_MANIFEST.json`` before its parent's. ``held_out/``
    uses only its own manifest, defaulting to ``"com"`` when absent. A parent's
    certificate applies only to clips it lists (:class:`_DirectoryFrameRule`).
    """
    candidates = [directory / CORPUS_MANIFEST_NAME]
    if directory.name != HELD_OUT_DIR_NAME:
        candidates.append(directory.parent / CORPUS_MANIFEST_NAME)
    return candidates


class _DirectoryFrameRule:
    """Resolve the velocity frame of clips stored in one directory.

    ``frame`` set: a manifest decides the whole directory -- its own ``CORPUS_MANIFEST.json`` (``"link"`` / ``"com"``), or
    ``held_out/`` without one (``"com"``, the build's policy).  ``frame`` None: no own manifest; a clip is ``"link"`` when its
    NAME is in ``inventory`` -- the top-level clips of the certified PARENT ``parent`` (a copied subset of the corpus; this is
    the only way the parent's certificate is inherited) -- and otherwise judged by its OWN NPZ kinematic stamp
    (:func:`clip_reference_root_velocity_frame`; unreadable -> ``"com"``).  Without a certified parent ``inventory`` is empty,
    so every clip is stamp-decided (a manifest-less directory of legacy clips stays ``"com"``; one of copied certified clips
    becomes ``"link"``). ``empty_frame`` is the frame of a directory
    with no clips at all (nothing to convert): the certified parent's ``"link"``, else ``"com"``.
    """

    __slots__ = ("frame", "inventory", "parent", "empty_frame")

    def __init__(self, frame: str | None, inventory: frozenset[str] = frozenset(), parent: Path | None = None, empty_frame: str = "com"):
        self.frame = frame
        self.inventory = inventory
        self.parent = parent
        self.empty_frame = empty_frame

    def clip_frame(self, path: Path) -> str:
        if self.frame is not None:
            return self.frame
        if path.name in self.inventory:
            return "link"
        stamped = clip_reference_root_velocity_frame(path)
        return "com" if stamped is None else stamped

    def directory_frame(self, directory: Path) -> str:
        """ONE frame for the whole directory (per motion-source ENTRY, when the loader exposes no clip list)."""
        if self.frame is not None:
            return self.frame
        names = sorted(top_level_clip_names(directory))
        if not names:
            return self.empty_frame
        frames = {name: self.clip_frame(directory / name) for name in names}
        distinct = set(frames.values())
        if len(distinct) == 1:
            return distinct.pop()
        n_link = sum(1 for f in frames.values() if f == "link")
        where = f"a sub-directory of the certified corpus {self.parent}" if self.parent is not None else "a directory without a CORPUS_MANIFEST.json"
        raise RuntimeError(
            f"{directory} ({where}) holds {n_link} LINK-origin clips (in the certificate's inventory or stamped) and "
            f"{len(frames) - n_link} legacy COM clips, so its reference root velocity frame cannot be decided per motion-source "
            "entry; use a loader that exposes its clip list (the frame is then decided per clip) or split the directory"
        )


def _directory_frame_rule(directory: Path, cache: dict[Path, _DirectoryFrameRule]) -> _DirectoryFrameRule:
    rule = cache.get(directory)
    if rule is None:
        candidates = reference_root_velocity_frame_candidates(directory)
        own = manifest_reference_root_velocity_frame(candidates[0])
        if own is not None:
            rule = _DirectoryFrameRule(own)
        elif len(candidates) == 1:  # held_out/: never inherits, legacy convention by the build's policy
            rule = _DirectoryFrameRule("com")
        else:
            parent = manifest_reference_root_velocity_frame(candidates[1])
            if parent == "link":  # certified parent: its certificate covers only the clips it lists
                rule = _DirectoryFrameRule(None, top_level_clip_names(directory.parent), directory.parent, empty_frame="link")
            elif parent is None:  # no manifest anywhere above: every clip is judged by its own stamp
                rule = _DirectoryFrameRule(None)
            else:  # a legacy parent declares its sub-directories legacy too
                rule = _DirectoryFrameRule("com")
        cache[directory] = rule
    return rule


def _real_clip_path(clip_file: str, real_dirs: dict[str, Path]) -> Path:
    """Resolve clip symlinks to the corpus containing their manifest.

    Directories are resolved once; the resolved filename is matched against the inventory.
    """
    path = os.path.expanduser(str(clip_file))
    if os.path.islink(path):
        return Path(os.path.realpath(path))
    parent, name = os.path.split(path)
    real = real_dirs.get(parent)
    if real is None:
        real = Path(os.path.realpath(parent)) if parent else Path(os.path.realpath(os.curdir))
        real_dirs[parent] = real
    return real / name


class ReferenceRootVelocityFrames:
    """Result of :func:`resolve_reference_root_velocity_frames`.

    ``frame``: ``"link"`` / ``"com"`` when every clip / entry agrees, :data:`MIXED_REFERENCE_ROOT_VELOCITY_FRAME` otherwise.
    ``by_directory``: ``{directory: frame}`` -- the motion-source entries as given, or the REAL clip directories when the
    frames were decided per clip; ``"mixed"`` for a directory whose own clips disagree (inventory / stamped LINK clips next
    to legacy ones).  ``per_clip``: the frame of every clip in clip order (None without ``clip_files``).
    ``clips_by_directory``: how many clips / entries each directory contributed; ``frame_counts``: the same split per
    frame (``{directory: {frame: n}}``, for the refusal message).
    """

    __slots__ = ("frame", "by_directory", "per_clip", "clips_by_directory", "frame_counts")

    def __init__(
        self,
        frame: str,
        by_directory: dict[str, str],
        per_clip: tuple[str, ...] | None,
        clips_by_directory: dict[str, int],
        frame_counts: dict[str, dict[str, int]] | None = None,
    ):
        self.frame = frame
        self.by_directory = by_directory
        self.per_clip = per_clip
        self.clips_by_directory = clips_by_directory
        self.frame_counts = frame_counts if frame_counts is not None else {d: {f: clips_by_directory.get(d, 0)} for d, f in by_directory.items()}

    @property
    def mixed(self) -> bool:
        return self.frame == MIXED_REFERENCE_ROOT_VELOCITY_FRAME

    def describe(self) -> str:
        """``link: <dir> (n clips), ...; com: <dir> (n clips), ...`` (``entries`` when resolved per motion-source entry); a
        directory whose clips disagree is listed under both frames with its per-frame counts."""
        unit = "clips" if self.per_clip is not None else "entries"
        parts = []
        for frame in REFERENCE_ROOT_VELOCITY_FRAMES[::-1]:  # link first
            items = [(d, counts[frame]) for d, counts in self.frame_counts.items() if counts.get(frame)]
            if items:
                parts.append(f"{frame}: " + ", ".join(f"{d} ({n} {unit})" for d, n in items))
        return "; ".join(parts) or "no motion source"


def mixed_velocity_frames_allowed() -> bool:
    """``HERO_ALLOW_MIXED_VELOCITY_FRAMES=1`` in the environment."""
    return os.environ.get(ALLOW_MIXED_VELOCITY_FRAMES_ENV, "").strip() == "1"


def resolve_reference_root_velocity_frames(
    motion_source: str | None,
    clip_files: Sequence[str] | None = None,
    *,
    allow_mixed: bool | None = None,
) -> ReferenceRootVelocityFrames:
    """Resolve the reference root velocity frames stored in ``motion_source``.

    ``motion_source`` is ``MotionConfig.motion_dir`` (one directory or a comma list) or ``motion_file`` (one clip).  With
    ``clip_files`` (the loader's file list in clip order) the frame is decided PER CLIP through the clip's REAL path
    (:func:`_real_clip_path`); otherwise per entry (directories as given, a ``.npz`` entry as one clip).  The rule per
    directory (:class:`_DirectoryFrameRule`): its own ``CORPUS_MANIFEST.json`` decides the whole directory (``"link"`` when
    ``kinematic_contract.schema`` starts with :data:`LINK_ORIGIN_CORPUS_SCHEMA_PREFIX`, else ``"com"``); ``held_out/`` without
    one is ``"com"`` (the build's policy); a sub-directory of a certified corpus inherits the certificate ONLY for the clips
    the certificate lists (the parent's top-level names, :func:`top_level_clip_names`); every other clip -- another
    directory under a certified corpus (``heldout/``, ``bench_local/``...), or any directory without a manifest -- is judged
    by its own NPZ kinematic stamp (:func:`clip_reference_root_velocity_frame`: certified clips are stamped, legacy clips
    are not).  A directory whose clips disagree can only be decided per clip (per entry it is refused).

    A source that mixes both conventions is REFUSED (``RuntimeError`` naming the LINK and COM directories) unless
    ``allow_mixed`` (default: :func:`mixed_velocity_frames_allowed`, env ``HERO_ALLOW_MIXED_VELOCITY_FRAMES=1``) -- then
    ``frame`` is ``"mixed"`` and ``per_clip`` tells the reset writers which rows to convert; without ``clip_files`` the
    per-clip decision is impossible and a mixed source is refused even with the opt-in.
    """
    rule_cache: dict[Path, _DirectoryFrameRule] = {}
    frame_counts: dict[str, dict[str, int]] = {}
    per_clip: list[str] | None = None

    def record(directory: str, frame: str) -> None:
        counts = frame_counts.setdefault(directory, {})
        counts[frame] = counts.get(frame, 0) + 1

    if clip_files is not None:
        per_clip = []
        real_dirs: dict[str, Path] = {}
        for clip in clip_files:
            path = _real_clip_path(clip, real_dirs)
            frame = _directory_frame_rule(path.parent, rule_cache).clip_frame(path)
            record(str(path.parent), frame)
            per_clip.append(frame)
    else:
        for entry in [entry.strip() for entry in str(motion_source or "").split(",") if entry.strip()]:
            path = Path(os.path.expanduser(entry))
            if path.suffix == ".npz" or path.is_file():
                record(str(path.parent), _directory_frame_rule(path.parent, rule_cache).clip_frame(path))
            else:
                record(str(path), _directory_frame_rule(path, rule_cache).directory_frame(path))
    by_directory = {d: next(iter(counts)) if len(counts) == 1 else MIXED_REFERENCE_ROOT_VELOCITY_FRAME for d, counts in frame_counts.items()}
    clips_by_directory = {d: sum(counts.values()) for d, counts in frame_counts.items()}
    frames = {frame for counts in frame_counts.values() for frame in counts}
    per_clip_out = None if per_clip is None else tuple(per_clip)
    if not frames:
        return ReferenceRootVelocityFrames("com", by_directory, per_clip_out, clips_by_directory, frame_counts)
    if len(frames) == 1:
        return ReferenceRootVelocityFrames(frames.pop(), by_directory, per_clip_out, clips_by_directory, frame_counts)
    result = ReferenceRootVelocityFrames(MIXED_REFERENCE_ROOT_VELOCITY_FRAME, by_directory, per_clip_out, clips_by_directory, frame_counts)
    if allow_mixed is None:
        allow_mixed = mixed_velocity_frames_allowed()
    if not allow_mixed:
        raise RuntimeError(
            f"motion source {motion_source!r} mixes certified LINK-origin and legacy COM root-velocity conventions "
            f"({result.describe()}); refusing: the reset writer would hand the LINK clips' velocity to the simulator's COM "
            f"setter unconverted.  Split the corpora, or set {ALLOW_MIXED_VELOCITY_FRAMES_ENV}=1 to decide the frame per "
            "clip (only LINK clips are converted); the contract commands refuse mixed inputs regardless"
        )
    if per_clip is None:
        raise RuntimeError(
            f"motion source {motion_source!r} mixes LINK-origin and COM root-velocity conventions ({result.describe()}) and "
            f"{ALLOW_MIXED_VELOCITY_FRAMES_ENV}=1 is set, but the loader exposes no per-clip file list, so the frame cannot be "
            "decided per clip; refusing"
        )
    return result


def reference_root_velocity_frame(
    motion_source: str | None, clip_files: Sequence[str] | None = None, *, allow_mixed: bool | None = None
) -> str:
    """``"link"`` / ``"com"`` / ``"mixed"`` of :func:`resolve_reference_root_velocity_frames` (same refusal rules)."""
    return resolve_reference_root_velocity_frames(motion_source, clip_files, allow_mixed=allow_mixed).frame


def reference_clip_files(motion: Any) -> list[str] | None:
    """The loader's per-clip file list in clip order, or None when it has none / it is not aligned with ``num_motions``.

    ``clip_files`` (``hero_isaacsim`` ``HeroMultiMotionLoader``), ``motion_files`` (:class:`MultiMotionLoader`) or
    ``motion_file`` (:class:`MotionLoader`, one clip).  A misaligned list (a pseudo-clip the loader did not register) is
    ignored so the caller falls back to the per-entry resolution instead of mis-assigning frames to clips.
    """
    if motion is None:
        return None
    num_motions = getattr(motion, "num_motions", None)
    for attr in ("clip_files", "motion_files"):
        files = getattr(motion, attr, None)
        if files is None:
            continue
        files = [str(f) for f in files]
        if num_motions is not None and len(files) != int(num_motions):
            logger.warning(
                "{}.{} lists {} clips but num_motions is {}; the reference root velocity frame is resolved per motion source "
                "entry instead of per clip",
                type(motion).__name__,
                attr,
                len(files),
                int(num_motions),
            )
            return None
        return files
    motion_file = getattr(motion, "motion_file", None)
    if isinstance(motion_file, str) and motion_file:
        return [motion_file]
    return None


#########################################################################################################
## MotionLoader and AdaptiveTimestepsSampler
#########################################################################################################
def _decode_npz_names(values: Any) -> list[str]:
    """Decode string arrays from NPZ files without turning bytes into ``b'...'``."""
    raw_values = values.tolist() if hasattr(values, "tolist") else list(values)
    return [value.decode("utf-8") if isinstance(value, bytes) else str(value) for value in raw_values]


class MotionTimebaseError(ValueError):
    """Raised when motion frames cannot be advanced one-for-one with the control loop."""


def _validated_motion_fps(raw_fps: Any, *, source: str, expected_fps: float | None = None) -> float:
    """Return a finite scalar FPS and enforce the one-frame-per-control-step contract.

    ``MotionCommand.step`` advances its reference by exactly one frame on every
    control step.  A clip sampled at any other rate would therefore be played at
    the wrong speed while its stored velocity channels kept their original time
    units.  Resampling is intentionally an offline data operation; runtime must
    fail loudly rather than silently change the task dynamics.
    """
    fps_array = np.asarray(raw_fps)
    if fps_array.size != 1:
        raise MotionTimebaseError(
            f"Motion clip '{source}' has non-scalar fps metadata with shape {fps_array.shape}; "
            "expected one positive scalar value."
        )
    fps = float(fps_array.reshape(()))
    if not np.isfinite(fps) or fps <= 0.0:
        raise MotionTimebaseError(
            f"Motion clip '{source}' has invalid fps={fps!r}; expected a finite positive value."
        )
    if expected_fps is not None:
        expected = float(expected_fps)
        if not np.isfinite(expected) or expected <= 0.0:
            raise MotionTimebaseError(
                f"Invalid environment control fps={expected!r}; expected a finite positive value."
            )
        if not np.isclose(fps, expected, rtol=0.0, atol=1.0e-4):
            raise MotionTimebaseError(
                f"Motion/control timebase mismatch for '{source}': motion fps={fps:g}, "
                f"control fps={expected:g}. MotionCommand advances exactly one frame per control step; "
                f"resample this clip to {expected:g} Hz before training or evaluation."
            )
    return fps


class MotionLoader:
    def __init__(
        self,
        motion_file: str,
        robot_body_names: list[str],
        robot_joint_names: list[str],
        device: str = "cpu",
        expected_fps: float | None = None,
        storage_device: str | None = None,
    ):
        # Resolve the motion file path using importlib.resources
        motion_file = resolve_data_file_path(motion_file)
        # The clip held by this loader, used to detect its velocity frame.
        self.motion_file: str = str(motion_file)

        # Bulk motion tensors can live on a different device than the consumers
        # (``storage_device="cpu"`` keeps ten-thousand-clip datasets out of GPU
        # memory; MotionCommand transfers the frames it needs per step).
        self.device = device
        self.storage_device = storage_device if storage_device else device

        logger.info(f"Loading motion file: {motion_file}")
        body_names_in_motion_data, joint_names_in_motion_data = self._load_data_from_motion_npz(
            motion_file, self.storage_device, expected_fps=expected_fps
        )
        body_indexes = self._get_index_of_a_in_b(robot_body_names, body_names_in_motion_data, self.storage_device)
        joint_indexes = self._get_index_of_a_in_b(robot_joint_names, joint_names_in_motion_data, self.storage_device)

        # Canonicalize body and joint ordering once so property reads can return tensor views.


        self._joint_pos = self._joint_pos[:, joint_indexes].contiguous()
        self._joint_vel = self._joint_vel[:, joint_indexes].contiguous()
        self._body_pos_w = self._body_pos_w[:, body_indexes].contiguous()
        self._body_quat_w = self._body_quat_w[:, body_indexes].contiguous()
        self._body_lin_vel_w = self._body_lin_vel_w[:, body_indexes].contiguous()
        self._body_ang_vel_w = self._body_ang_vel_w[:, body_indexes].contiguous()

        # Post-canonicalization both index maps are the identity.  They are kept
        # because MotionCommand consumes them for robot<->motion order mapping.
        self._joint_indexes = torch.arange(len(robot_joint_names), dtype=torch.long, device=self.storage_device)
        self._body_indexes = torch.arange(len(robot_body_names), dtype=torch.long, device=self.storage_device)
        self.time_step_total = self._joint_pos.shape[0]

    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)

    # Expected holosoma NPZ keys
    _REQUIRED_KEYS = {
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

    def _load_data_from_motion_npz(
        self, motion_file: str, device: str, *, expected_fps: float | None = None
    ) -> tuple[list[str], list[str]]:
        with cached_open(motion_file, "rb") as f, np.load(f) as data:
            # Sanity check: warn if not in expected holosoma format
            keys = set(data.files)
            missing = self._REQUIRED_KEYS - keys
            if missing:
                logger.warning(
                    f"Motion NPZ '{motion_file}' is missing expected holosoma keys: {missing}. "
                    f"All motion data should be in holosoma format (with body_names, joint_names, "
                    f"and root DOFs in joint_pos). Convert from TML/BeyondMimic first."
                )
                raise ValueError(
                    f"Unsupported motion format in '{motion_file}': missing keys {missing}. "
                    f"Please convert to holosoma format."
                )

            self.fps = _validated_motion_fps(data["fps"], source=motion_file, expected_fps=expected_fps)

            body_names = _decode_npz_names(data["body_names"])
            joint_names = _decode_npz_names(data["joint_names"])

            joint_pos_raw = data["joint_pos"]
            joint_vel_raw = data["joint_vel"]
            body_pos_w_raw = data["body_pos_w"]
            body_quat_w_raw = data["body_quat_w"]
            body_lin_vel_w_raw = data["body_lin_vel_w"]
            body_ang_vel_w_raw = data["body_ang_vel_w"]

            # Holosoma format: joint_pos includes root DOFs [xyz, wxyz] as first 7 values
            # joint_vel includes root velocity [vel_xyz, vel_wxyz] as first 6 values
            num_joint_cols = joint_pos_raw.shape[1]
            num_vel_cols = joint_vel_raw.shape[1]
            num_bodies = body_pos_w_raw.shape[1]

            if num_joint_cols != len(joint_names) + 7:
                logger.warning(
                    f"Unexpected joint_pos columns: got {num_joint_cols}, expected {len(joint_names) + 7} "
                    f"(= {len(joint_names)} joints + 7 root DOFs). File: {motion_file}"
                )
            if num_vel_cols != len(joint_names) + 6:
                logger.warning(
                    f"Unexpected joint_vel columns: got {num_vel_cols}, expected {len(joint_names) + 6} "
                    f"(= {len(joint_names)} joints + 6 root DOFs). File: {motion_file}"
                )
            if num_bodies != len(body_names):
                logger.warning(
                    f"Body count mismatch: body_pos_w has {num_bodies} bodies but body_names has "
                    f"{len(body_names)}. File: {motion_file}"
                )

            # Strip root DOFs
            self._joint_pos = torch.tensor(joint_pos_raw[:, 7:], dtype=torch.float32, device=device)
            self._joint_vel = torch.tensor(joint_vel_raw[:, 6:], dtype=torch.float32, device=device)

            assert len(joint_names) == self._joint_pos.shape[1], (
                f"Joint names ({len(joint_names)}) != joint_pos columns ({self._joint_pos.shape[1]}) in {motion_file}"
            )
            assert len(body_names) == body_pos_w_raw.shape[1], (
                f"Body names ({len(body_names)}) != body_pos_w bodies ({body_pos_w_raw.shape[1]}) in {motion_file}"
            )

            self._body_pos_w = torch.tensor(body_pos_w_raw, dtype=torch.float32, device=device)

            # NOTE: wxyz after loading from npz
            body_quat_w_wxyz = torch.tensor(body_quat_w_raw, dtype=torch.float32, device=device)  # This is wxyz
            self._body_quat_w = body_quat_w_wxyz[:, :, [1, 2, 3, 0]]  # Change to xyzw

            self._body_lin_vel_w = torch.tensor(body_lin_vel_w_raw, dtype=torch.float32, device=device)
            self._body_ang_vel_w = torch.tensor(body_ang_vel_w_raw, dtype=torch.float32, device=device)

            # add object pos and quat
            self.has_object = "object_pos_w" in data
            if self.has_object:
                self._object_pos_w = torch.tensor(data["object_pos_w"], dtype=torch.float32, device=device)
                # NOTE: wxyz after loading from npz
                object_quat_w = torch.tensor(data["object_quat_w"], dtype=torch.float32, device=device)
                self._object_quat_w = object_quat_w[:, [1, 2, 3, 0]]  # Change to xyzw
                self._object_lin_vel_w = torch.tensor(data["object_lin_vel_w"], dtype=torch.float32, device=device)
            else:
                self._object_pos_w = torch.zeros(0, 3, device=device)
                self._object_quat_w = torch.zeros(0, 4, device=device)
                self._object_lin_vel_w = torch.zeros(0, 3, device=device)
        return body_names, joint_names

    # Storage is canonicalized to robot body/joint order at load time, so the
    # bulk-tensor properties return the raw storage directly (the historical
    # per-access ``[:, indexes]`` gather copied the full timeline every read).
    @property
    def joint_pos(self) -> torch.Tensor:
        return self._joint_pos

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._joint_vel

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w

    @property
    def object_pos_w(self) -> torch.Tensor:
        return self._object_pos_w

    @property
    def object_quat_w(self) -> torch.Tensor:
        return self._object_quat_w

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        return self._object_lin_vel_w

    @property
    def num_motions(self) -> int:
        return 1

    @property
    def motion_start_idx(self) -> torch.Tensor:
        return torch.tensor([0], dtype=torch.long, device=self.device)

    @property
    def motion_end_idx(self) -> torch.Tensor:
        return torch.tensor([self.time_step_total], dtype=torch.long, device=self.device)

    def frames(self, name: str, idx: torch.Tensor) -> torch.Tensor:
        """Gather motion frames by index, bridging a CPU storage device.

        ``idx`` lives on the compute device; when the bulk tensors are stored on
        CPU (``storage_device="cpu"``) the gather runs on CPU and only the
        selected frames are transferred.
        """
        source = getattr(self, name)
        if source.device == idx.device:
            return source[idx]
        return source[idx.to(source.device)].to(idx.device, non_blocking=True)

    def extend_with_segments(self, segments: dict[str, torch.Tensor], prepend: bool) -> MotionLoader:
        """Merge interpolated segments with motion data, mutating this MotionLoader."""
        concat_targets = [
            ("joint_pos", "_joint_pos"),
            ("joint_vel", "_joint_vel"),
            ("body_pos", "_body_pos_w"),
            ("body_quat", "_body_quat_w"),
            ("body_lin_vel", "_body_lin_vel_w"),
            ("body_ang_vel", "_body_ang_vel_w"),
        ]
        if self.has_object:
            concat_targets.extend(
                [
                    ("object_pos", "_object_pos_w"),
                    ("object_quat", "_object_quat_w"),
                    ("object_lin_vel", "_object_lin_vel_w"),
                ]
            )

        for seg_key, attr_name in concat_targets:
            existing = getattr(self, attr_name)
            tensors = (segments[seg_key], existing) if prepend else (existing, segments[seg_key])
            setattr(self, attr_name, torch.cat(tensors, dim=0))

        self.time_step_total = self._joint_pos.shape[0]
        return self


class MultiMotionLoader:
    """Loads multiple NPZ motion files from a directory and concatenates them at runtime.

    Tracks per-motion boundaries so environments can sample within individual clips.
    Compatible with the same interface as MotionLoader.
    """

    def __init__(
        self,
        motion_dir: str,
        robot_body_names: list[str],
        robot_joint_names: list[str],
        device: str = "cpu",
        expected_fps: float | None = None,
        storage_device: str | None = None,
    ):
        self.device = device
        self.storage_device = storage_device if storage_device else device
        # Support comma-separated directories for combining multiple datasets
        dirs = [d.strip() for d in motion_dir.split(",")]
        motion_files = []
        for d in dirs:
            expanded = os.path.expanduser(d)
            files = sorted(str(p) for p in Path(expanded).glob("*.npz"))
            logger.info(f"MultiMotionLoader: found {len(files)} .npz files in {expanded}")
            motion_files.extend(files)
        assert len(motion_files) > 0, f"No .npz files found in {motion_dir}"
        logger.info(f"MultiMotionLoader: loading {len(motion_files)} total motion files")

        loaders = []
        loaded_motion_files = []
        skipped = 0
        for mf in motion_files:
            try:
                loader = MotionLoader(
                    mf,
                    robot_body_names,
                    robot_joint_names,
                    device=device,
                    expected_fps=expected_fps,
                    storage_device=self.storage_device,
                )
                loaders.append(loader)
                loaded_motion_files.append(mf)
            except MotionTimebaseError:
                # A timebase mismatch is not a malformed clip that can be skipped:
                # silently dropping it changes the requested dataset and can hide a
                # mixed-FPS directory.  Propagate with the actionable source path.
                raise
            except (KeyError, AssertionError, ValueError) as e:  # noqa: PERF203
                # Skip files with incompatible format (e.g., missing body_names, wrong body count)
                skipped += 1
                if skipped <= 3:
                    logger.warning(f"MultiMotionLoader: skipping {mf}: {e}")
        if skipped > 3:
            logger.warning(f"MultiMotionLoader: skipped {skipped} files total due to format issues")
        assert len(loaders) > 0, f"No compatible motion files found (skipped {skipped})"

        # Even without an environment-provided expected rate, concatenated clips
        # must share one timebase.  MultiMotionLoader advances every clip by one
        # frame per command step and cannot represent per-clip rates.
        first_fps = loaders[0].fps
        mixed_fps = [
            (path, loader.fps)
            for path, loader in zip(loaded_motion_files, loaders)
            if not np.isclose(loader.fps, first_fps, rtol=0.0, atol=1.0e-4)
        ]
        if mixed_fps:
            details = ", ".join(
                [f"{loaded_motion_files[0]}={first_fps:g}Hz"]
                + [f"{path}={fps:g}Hz" for path, fps in mixed_fps[:5]]
            )
            if len(mixed_fps) > 5:
                details += f", ... ({len(mixed_fps)} mismatched clips total)"
            raise MotionTimebaseError(
                "MultiMotionLoader cannot mix clip frame rates because references advance one frame per "
                f"control step: {details}. Resample all clips to one common FPS before loading."
            )

        # Track per-motion boundaries.  These small index tensors stay on the
        # compute device even when the bulk data is stored on CPU: samplers and
        # clip-end checks read them every step.
        lengths = [loader.time_step_total for loader in loaders]
        cumulative = torch.tensor(lengths, dtype=torch.long, device=device).cumsum(dim=0)
        self._motion_start_idx = torch.cat([torch.tensor([0], dtype=torch.long, device=device), cumulative[:-1]])
        self._motion_end_idx = cumulative
        self._num_motions = len(loaders)
        # Loaded clip files in motion order, used to detect per-clip velocity frames.
        self.motion_files: list[str] = list(loaded_motion_files)

        # Concatenate all motion data (already canonicalized to robot order and
        # resident on storage_device by each MotionLoader).
        self._joint_pos = torch.cat([ld._joint_pos for ld in loaders], dim=0)
        self._joint_vel = torch.cat([ld._joint_vel for ld in loaders], dim=0)
        self._body_pos_w = torch.cat([ld._body_pos_w for ld in loaders], dim=0)
        self._body_quat_w = torch.cat([ld._body_quat_w for ld in loaders], dim=0)
        self._body_lin_vel_w = torch.cat([ld._body_lin_vel_w for ld in loaders], dim=0)
        self._body_ang_vel_w = torch.cat([ld._body_ang_vel_w for ld in loaders], dim=0)

        # Use indexes from first loader (all loaders share the same robot).
        # Post-canonicalization these are identity maps (see MotionLoader).
        self._joint_indexes = loaders[0]._joint_indexes
        self._body_indexes = loaders[0]._body_indexes
        self.fps = first_fps
        self.time_step_total = self._joint_pos.shape[0]

        # Require consistent object-track availability across clips. A mixed corpus
        # would otherwise drop valid object references and score object rewards against zeros.


        object_flags = [ld.has_object for ld in loaders]
        if any(object_flags) and not all(object_flags):
            missing = [i for i, flag in enumerate(object_flags) if not flag]
            raise ValueError(
                f"motion corpus mixes {sum(object_flags)} clips WITH an object track and "
                f"{len(missing)} without (loader indices {missing[:8]}"
                f"{'...' if len(missing) > 8 else ''}). Object references would be dropped for the "
                "entire corpus. Split the corpora, or add object tracks to the clips that lack them."
            )
        self.has_object = all(object_flags)
        if self.has_object:
            self._object_pos_w = torch.cat([ld._object_pos_w for ld in loaders], dim=0)
            self._object_quat_w = torch.cat([ld._object_quat_w for ld in loaders], dim=0)
            self._object_lin_vel_w = torch.cat([ld._object_lin_vel_w for ld in loaders], dim=0)
        else:
            self._object_pos_w = torch.zeros(0, 3, device=self.storage_device)
            self._object_quat_w = torch.zeros(0, 4, device=self.storage_device)
            self._object_lin_vel_w = torch.zeros(0, 3, device=self.storage_device)

        # Log how many clips contain object tracks so missing task data is visible at startup.


        logger.info(
            f"MultiMotionLoader: {self._num_motions} motions, {self.time_step_total} total frames, "
            f"object tracks {sum(object_flags)}/{len(object_flags)} clips -> has_object="
            f"{self.has_object}"
            + (f", object frames {self._object_pos_w.shape[0]}" if self.has_object else
               " (object references are ZERO for the whole corpus)")
        )

    @property
    def num_motions(self) -> int:
        return self._num_motions

    @property
    def motion_start_idx(self) -> torch.Tensor:
        return self._motion_start_idx

    @property
    def motion_end_idx(self) -> torch.Tensor:
        return self._motion_end_idx

    # Storage is canonicalized at load time (see MotionLoader), so these return
    # the raw concatenated tensors without a per-access gather copy.
    @property
    def joint_pos(self) -> torch.Tensor:
        return self._joint_pos

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._joint_vel

    @property
    def body_pos_w(self) -> torch.Tensor:
        return self._body_pos_w

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._body_quat_w

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._body_lin_vel_w

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._body_ang_vel_w

    @property
    def object_pos_w(self) -> torch.Tensor:
        return self._object_pos_w

    @property
    def object_quat_w(self) -> torch.Tensor:
        return self._object_quat_w

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        return self._object_lin_vel_w

    frames = MotionLoader.frames

    def extend_with_segments(self, segments: dict[str, torch.Tensor], prepend: bool) -> MultiMotionLoader:
        """Merge interpolated segments with motion data, mutating this MultiMotionLoader."""
        concat_targets = [
            ("joint_pos", "_joint_pos"),
            ("joint_vel", "_joint_vel"),
            ("body_pos", "_body_pos_w"),
            ("body_quat", "_body_quat_w"),
            ("body_lin_vel", "_body_lin_vel_w"),
            ("body_ang_vel", "_body_ang_vel_w"),
        ]
        if self.has_object:
            concat_targets.extend(
                [
                    ("object_pos", "_object_pos_w"),
                    ("object_quat", "_object_quat_w"),
                    ("object_lin_vel", "_object_lin_vel_w"),
                ]
            )

        added_frames = 0
        for seg_key, attr_name in concat_targets:
            existing = getattr(self, attr_name)
            tensors = (segments[seg_key], existing) if prepend else (existing, segments[seg_key])
            setattr(self, attr_name, torch.cat(tensors, dim=0))
            if added_frames == 0:
                added_frames = segments[seg_key].shape[0]

        # Update boundaries — shift all motion boundaries if prepending
        files = getattr(self, "motion_files", None)
        if files is not None and len(files) == int(self._num_motions):  # the pseudo-clip inherits its neighbour's file (frame)
            if prepend:
                files.insert(0, files[0])
            else:
                files.append(files[-1])
        if prepend:
            self._motion_start_idx = self._motion_start_idx + added_frames
            self._motion_end_idx = self._motion_end_idx + added_frames
            dev = self._motion_start_idx.device
            self._motion_start_idx = torch.cat(
                [torch.tensor([0], dtype=torch.long, device=dev), self._motion_start_idx]
            )
            self._motion_end_idx = torch.cat(
                [torch.tensor([added_frames], dtype=torch.long, device=dev), self._motion_end_idx]
            )
        else:
            old_total = self.time_step_total
            dev = self._motion_start_idx.device
            self._motion_start_idx = torch.cat(
                [self._motion_start_idx, torch.tensor([old_total], dtype=torch.long, device=dev)]
            )
            self._motion_end_idx = torch.cat(
                [self._motion_end_idx, torch.tensor([old_total + added_frames], dtype=torch.long, device=dev)]
            )

        self.time_step_total = self._joint_pos.shape[0]
        self._num_motions = len(self._motion_start_idx)
        return self

    def extend_each_clip_with_segments(
        self, per_clip_segments: list[dict[str, torch.Tensor]], prepend: bool
    ) -> "MultiMotionLoader":
        """Insert each clip's default-pose transition within its own phase range.

        per_clip_segments[c] uses the same tensor keys as extend_with_segments.
        Segments are prepended or appended to clip c without creating new motion IDs."""
        assert len(per_clip_segments) == self._num_motions, (
            f"need one segment dict per clip ({self._num_motions}), got {len(per_clip_segments)}"
        )
        concat_targets = [
            ("joint_pos", "_joint_pos"), ("joint_vel", "_joint_vel"),
            ("body_pos", "_body_pos_w"), ("body_quat", "_body_quat_w"),
            ("body_lin_vel", "_body_lin_vel_w"), ("body_ang_vel", "_body_ang_vel_w"),
        ]
        if self.has_object:
            concat_targets += [
                ("object_pos", "_object_pos_w"), ("object_quat", "_object_quat_w"),
                ("object_lin_vel", "_object_lin_vel_w"),
            ]
        starts = self._motion_start_idx.tolist()
        ends = self._motion_end_idx.tolist()
        dev = self._motion_start_idx.device

        new_arrays: dict[str, list[torch.Tensor]] = {attr: [] for _, attr in concat_targets}
        new_lengths: list[int] = []
        for c in range(self._num_motions):
            s, e = int(starts[c]), int(ends[c])
            seg = per_clip_segments[c]
            add = int(seg[concat_targets[0][0]].shape[0])
            for seg_key, attr in concat_targets:
                clip_slice = getattr(self, attr)[s:e]
                pieces = (seg[seg_key], clip_slice) if prepend else (clip_slice, seg[seg_key])
                new_arrays[attr].append(torch.cat(pieces, dim=0))
            new_lengths.append((e - s) + add)

        for _, attr in concat_targets:
            setattr(self, attr, torch.cat(new_arrays[attr], dim=0))
        cum = torch.tensor(new_lengths, dtype=torch.long, device=dev).cumsum(0)
        self._motion_start_idx = torch.cat([torch.tensor([0], dtype=torch.long, device=dev), cum[:-1]])
        self._motion_end_idx = cum
        self.time_step_total = self._joint_pos.shape[0]
        # _num_motions UNCHANGED — transitions are folded into existing clips, not new clips.
        return self


class AdaptiveTimestepsSampler:
    """Prioritizes training on motion segments where the robot fails most often.

    Three modes (single clip: all equivalent):
    - legacy (default): one bin table over the CONCATENATED timeline; the sampled bin is
      reinterpreted as a phase of whichever clip the env is independently assigned to —
      failure stats are smeared across unrelated clips in multi-task.
    - phase_binning: one bin table over normalized per-clip phase [0,1] — coherent phase
      curriculum, but still shared across clips and clip choice stays uniform.
    - per_clip: a (num_clips, num_bins) failure registry; (clip, phase) sampled jointly
      proportional to failure. Clip-local curriculum + failure-weighted clip selection.

    In per_clip mode, optional clip-marginal temperature and hard-cap shaping is
    applied after phase smoothing. It limits cross-clip collapse without changing
    the learned conditional phase distribution inside any clip. Defaults preserve
    the historical per_clip_v1 distribution exactly.
    """

    def __init__(
        self,
        motion_time_step_total: int,
        device: str,
        env_fps: int,
        adaptive_kernel_size: int = 1,
        adaptive_lambda: float = 0.8,
        adaptive_uniform_ratio: float = 0.1,
        adaptive_alpha: float = 0.001,
        phase_binning: bool = False,
        max_clip_time_step: int | None = None,
        per_clip: bool = False,
        num_clips: int = 1,
        adaptive_clip_temperature: float = 1.0,
        adaptive_clip_max_probability: float = 1.0,
    ):
        self.device = device
        # length of the motion in rl environment time steps
        self.motion_time_step_total = motion_time_step_total
        # fps of the rl environment
        self.env_fps = env_fps

        self.adaptive_kernel_size = adaptive_kernel_size
        self.adaptive_lambda = adaptive_lambda
        self.adaptive_uniform_ratio = float(adaptive_uniform_ratio)
        self.adaptive_clip_temperature = float(adaptive_clip_temperature)
        self.adaptive_clip_max_probability = float(adaptive_clip_max_probability)

        if not np.isfinite(self.adaptive_uniform_ratio) or not 0.0 <= self.adaptive_uniform_ratio <= 1.0:
            raise ValueError(
                "adaptive_uniform_ratio must be finite and in [0, 1], "
                f"got {adaptive_uniform_ratio!r}"
            )
        if not np.isfinite(self.adaptive_clip_temperature) or self.adaptive_clip_temperature <= 0.0:
            raise ValueError(
                "adaptive_clip_temperature must be finite and positive, "
                f"got {adaptive_clip_temperature!r}"
            )
        if (
            not np.isfinite(self.adaptive_clip_max_probability)
            or not 0.0 < self.adaptive_clip_max_probability <= 1.0
        ):
            raise ValueError(
                "adaptive_clip_max_probability must be finite and in (0, 1], "
                f"got {adaptive_clip_max_probability!r}"
            )
        self.adaptive_alpha = adaptive_alpha

        self.per_clip = per_clip
        self.num_clips = max(int(num_clips), 1) if per_clip else 1
        if self.per_clip and self.num_clips > 1:
            feasible_floor = 1.0 / float(self.num_clips)
            if self.adaptive_clip_max_probability < feasible_floor - 1.0e-12:
                raise ValueError(
                    "adaptive_clip_max_probability is infeasible for this registry: "
                    f"max_probability={self.adaptive_clip_max_probability:g}, num_clips={self.num_clips}, "
                    f"minimum feasible value={feasible_floor:g}"
                )
        # Phase bins cover normalized per-clip time, at a resolution based on the longest clip.
        # Without phase binning, bins cover the concatenated timeline.


        self.phase_binning = phase_binning or per_clip
        if self.phase_binning:
            ref_len = max_clip_time_step if max_clip_time_step is not None else motion_time_step_total
            self.num_bins = int(ref_len // max(self.env_fps, 1)) + 1
        else:
            # Match BeyondMimic binning: ~1 second bins at env FPS, with +1 tail bin.
            self.num_bins = int(self.motion_time_step_total // max(self.env_fps, 1)) + 1

        # Match BeyondMimic non-causal kernel.
        self.kernel = torch.tensor(
            [self.adaptive_lambda**i for i in range(self.adaptive_kernel_size)],
            device=self.device,
        )
        self.kernel = self.kernel / self.kernel.sum()

        # key data: failure counts
        self.init_buffers()
        # metrics
        self.metrics: dict[str, torch.Tensor] = {}
        # get_stats() runs conv1d smoothing + entropy + (in per_clip mode) a
        # sort/water-fill projection over the full (clips, bins) table.  It is
        # called from update_metrics() every env step, where the table only
        # changes at episode boundaries — recompute at a fixed step interval
        # and serve cached metrics in between.
        self._stats_interval = max(int(os.environ.get("WBT_SAMPLER_STATS_INTERVAL", "24")), 1)
        self._stats_countdown = 0

    def init_buffers(self):
        # Registry rows: one per clip in per_clip mode, a single shared row otherwise.
        rows = self.num_clips
        self.current_bin_failed_count = torch.zeros(rows, self.num_bins, dtype=torch.float, device=self.device)
        self.bin_failed_count = torch.zeros(rows, self.num_bins, dtype=torch.float, device=self.device)

    def sampling_policy(self) -> dict[str, Any]:
        """Settings that turn the failure table into draw probabilities.

        Stored next to the table by :meth:`state_dict` and compared entry by entry
        by :meth:`load_state_dict`, so a table cannot resume under a different
        sampling rule.  Subclasses that change the composition add their own
        entries (numbers are compared with a tolerance, everything else exactly).
        """

        return {
            "adaptive_uniform_ratio": float(self.adaptive_uniform_ratio),
            "adaptive_clip_temperature": float(self.adaptive_clip_temperature),
            "adaptive_clip_max_probability": float(self.adaptive_clip_max_probability),
        }

    @staticmethod
    def sampling_policy_mismatches(saved: Mapping[str, Any] | None, expected: Mapping[str, Any]) -> dict[str, Any]:
        """``{key: (saved, expected)}`` for every expected policy entry that ``saved`` lacks or disagrees with."""

        if not isinstance(saved, Mapping):
            raise TypeError("adaptive sampler sampling_policy must be a mapping")
        mismatches: dict[str, Any] = {}
        for key, value in expected.items():
            if key not in saved:
                mismatches[key] = (None, value)
                continue
            have = saved[key]
            numeric = isinstance(value, (int, float)) and not isinstance(value, bool)
            if numeric:
                same = (
                    isinstance(have, (int, float))
                    and not isinstance(have, bool)
                    and bool(np.isclose(float(have), float(value), rtol=0.0, atol=1.0e-12))
                )
            else:
                same = have == value
            if not same:
                mismatches[key] = (have, value)
        return mismatches

    def state_dict(self) -> dict[str, Any]:
        """Return the persistent adaptive curriculum state.

        The configuration fields are deliberately stored alongside the tensors.
        Loading a failure table against a different clip registry, binning mode,
        or prepend-modified timeline would silently assign difficulty to the wrong
        phase, so such resumes must fail loudly rather than reshape the table.
        """

        return {
            "schema": "holosoma_adaptive_timesteps_sampler_v1",
            "motion_time_step_total": int(self.motion_time_step_total),
            "env_fps": int(self.env_fps),
            "num_bins": int(self.num_bins),
            "num_clips": int(self.num_clips),
            "phase_binning": bool(self.phase_binning),
            "per_clip": bool(self.per_clip),
            "sampling_policy": self.sampling_policy(),
            "bin_failed_count": self.bin_failed_count.detach().cpu().clone(),
            "current_bin_failed_count": self.current_bin_failed_count.detach().cpu().clone(),
        }

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        """Restore a checkpointed curriculum after validating its support."""

        if not isinstance(state, Mapping):
            raise TypeError("adaptive sampler checkpoint must be a mapping")
        if state.get("schema") != "holosoma_adaptive_timesteps_sampler_v1":
            raise ValueError(f"unsupported adaptive sampler checkpoint schema: {state.get('schema')!r}")

        expected = {
            "motion_time_step_total": int(self.motion_time_step_total),
            "env_fps": int(self.env_fps),
            "num_bins": int(self.num_bins),
            "num_clips": int(self.num_clips),
            "phase_binning": bool(self.phase_binning),
            "per_clip": bool(self.per_clip),
        }
        mismatches = {
            key: (state.get(key), value)
            for key, value in expected.items()
            if state.get(key) != value
        }
        if mismatches:
            raise ValueError(
                "adaptive sampler checkpoint support differs from the live motion registry: "
                f"{mismatches}"
            )

        expected_policy = self.sampling_policy()
        checkpoint_policy = state.get("sampling_policy")
        if checkpoint_policy is None:
            # Checkpoints created before capped/tempered sampling had exactly
            # these implicit settings. They remain loadable under the legacy
            # defaults, but cannot silently resume into a different curriculum.
            checkpoint_policy = {
                "adaptive_uniform_ratio": 0.1,
                "adaptive_clip_temperature": 1.0,
                "adaptive_clip_max_probability": 1.0,
            }
        policy_mismatches = self.sampling_policy_mismatches(checkpoint_policy, expected_policy)
        if policy_mismatches:
            raise ValueError(
                "adaptive sampler checkpoint sampling policy differs from the live configuration: "
                f"{policy_mismatches}"
            )

        shape = (self.num_clips, self.num_bins)
        restored: dict[str, torch.Tensor] = {}
        for key in ("bin_failed_count", "current_bin_failed_count"):
            if key not in state:
                raise KeyError(f"adaptive sampler checkpoint is missing {key!r}")
            value = torch.as_tensor(state[key], dtype=self.bin_failed_count.dtype, device=self.device)
            if tuple(value.shape) != shape:
                raise ValueError(
                    f"adaptive sampler {key} shape {tuple(value.shape)} != live support {shape}"
                )
            if not bool(torch.isfinite(value).all().item()) or bool((value < 0).any().item()):
                raise ValueError(f"adaptive sampler {key} must be finite and non-negative")
            restored[key] = value

        self.bin_failed_count.copy_(restored["bin_failed_count"])
        self.current_bin_failed_count.copy_(restored["current_bin_failed_count"])

    def synchronize_distributed(self, *, world_size: int) -> None:
        """Average rank-local failure EMAs at a rollout boundary.

        Each PPO rank intentionally explores independently during its local
        rollout.  The EMA update is linear, therefore averaging the resulting
        tables at the rollout boundary is equivalent to applying the EMA to the
        rank-mean failure counts at every step, without an NCCL collective in the
        simulator hot path.  Averaging (rather than summing) also keeps the
        single-rank and multi-rank uniform-floor scale comparable.
        """

        if world_size <= 1:
            return
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError(
                "adaptive sampler requested multi-rank synchronization without an initialized process group"
            )
        actual_world_size = int(torch.distributed.get_world_size())
        if actual_world_size != int(world_size):
            raise RuntimeError(
                "adaptive sampler distributed world-size mismatch: "
                f"process_group={actual_world_size} requested={world_size}"
            )
        # MotionCommand is constructed while Isaac's environment setup runs in
        # ``torch.inference_mode``.  Its persistent buffers are therefore
        # inference tensors: mutating them later from PPO's ordinary Python
        # context raises "Inplace update to inference tensor outside
        # InferenceMode".  Keep both the collective's in-place write and the
        # rank-mean normalization in an explicit inference-mode region.
        with torch.inference_mode():
            for value in (self.bin_failed_count, self.current_bin_failed_count):
                torch.distributed.all_reduce(value, op=torch.distributed.ReduceOp.SUM)
                value.div_(float(world_size))

    def update_current_bin_failed_count(
        self,
        failed_at_time_step: torch.Tensor,
        failed_at_phase: torch.Tensor | None = None,
        failed_clip_ids: torch.Tensor | None = None,
    ):
        """Update the current bin failed count with terminated time steps.

        Legacy (phase_binning=False): bin by absolute concatenated timestep.
        phase_binning=True: bin by normalized per-clip phase in [0,1] (failed_at_phase).
        per_clip=True: additionally route each failure to its own clip's row (failed_clip_ids)."""
        if self.phase_binning and failed_at_phase is not None:
            failed_bin = torch.clamp((failed_at_phase * self.num_bins).long(), 0, self.num_bins - 1)
        else:
            failed_bin = torch.clamp(
                (failed_at_time_step * self.num_bins) // max(self.motion_time_step_total, 1),
                0,
                self.num_bins - 1,
            ).long()
        assert failed_bin.min() >= 0 and failed_bin.max() < self.num_bins, "Failed bin is out of range"
        if self.per_clip and failed_clip_ids is not None:
            flat = failed_clip_ids.long() * self.num_bins + failed_bin
            counts = torch.bincount(flat, minlength=self.num_clips * self.num_bins)
            self.current_bin_failed_count[:] = counts.view(self.num_clips, self.num_bins).float()
        else:
            self.current_bin_failed_count[:] = torch.bincount(failed_bin, minlength=self.num_bins).view(
                1, self.num_bins
            )

    def update_bin_failed_count(self):
        """At every rl environment step, update the failed count with the current bin failed count."""
        self.bin_failed_count = (self.adaptive_alpha * self.current_bin_failed_count) + (
            1 - self.adaptive_alpha
        ) * self.bin_failed_count
        self.current_bin_failed_count.zero_()

    @staticmethod
    def _project_with_probability_cap(probabilities: torch.Tensor, max_prob: float) -> torch.Tensor:
        """KL-project a positive categorical distribution onto ``q_i <= max_prob``.

        The solution has the water-filling form ``q_i = min(max_prob, scale * p_i)``.
        A vectorized sorted-support calculation avoids per-clip Python loops and GPU
        synchronization. Callers validate feasibility before reaching this helper.
        """

        num_items = int(probabilities.numel())
        if num_items <= 1 or max_prob >= 1.0:
            return probabilities
        uniform_prob = 1.0 / float(num_items)
        if max_prob <= uniform_prob + 1.0e-12:
            return torch.full_like(probabilities, uniform_prob)


        eps = torch.finfo(probabilities.dtype).tiny
        positive = probabilities.clamp_min(eps)
        positive = positive / positive.sum()
        sorted_prob, _ = torch.sort(positive, descending=True)
        tail_mass = torch.flip(torch.cumsum(torch.flip(sorted_prob, dims=(0,)), dim=0), dims=(0,))
        capped_count = torch.arange(num_items, device=probabilities.device, dtype=probabilities.dtype)
        remaining_mass = 1.0 - capped_count * float(max_prob)
        scale = remaining_mass / tail_mass.clamp_min(eps)

        # For k capped entries, the largest still-free entry (index k) must fit
        # under the cap. The first such k is the water-filling active set.
        valid = (remaining_mass > 0.0) & (scale * sorted_prob <= float(max_prob) + 1.0e-7)
        # Feasibility plus the near-uniform branch above guarantees at least one
        # valid active set. argmax keeps this hot path entirely on the device;
        # torch.nonzero/.item would introduce a CUDA synchronization per metric update.
        chosen_index = torch.argmax(valid.to(dtype=torch.int64))
        chosen_scale = scale[chosen_index]
        projected = torch.minimum(positive * chosen_scale, torch.full_like(positive, float(max_prob)))
        # The analytical result sums to one; normalize only for roundoff.
        projected = projected / projected.sum()
        return projected

    def _shape_clip_marginal(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Temper/cap the clip marginal without changing phase sampling inside a clip."""

        if (
            not self.per_clip
            or self.num_clips <= 1
            or (
                self.adaptive_clip_temperature == 1.0
                and self.adaptive_clip_max_probability >= 1.0
            )
        ):
            # Preserve the historical numerical path exactly at default settings.
            return probabilities

        table = probabilities.view(self.num_clips, self.num_bins)
        raw_clip_prob = table.sum(dim=1)
        positive_clip_prob = raw_clip_prob.clamp_min(torch.finfo(table.dtype).tiny)
        tempered_clip_prob = positive_clip_prob.pow(1.0 / self.adaptive_clip_temperature)
        tempered_clip_prob = tempered_clip_prob / tempered_clip_prob.sum()
        shaped_clip_prob = self._project_with_probability_cap(
            tempered_clip_prob, self.adaptive_clip_max_probability
        )

        # A row can be exactly zero only when uniform_ratio=0 and it has never
        # failed. If capping allocates it mass, sample its phase uniformly.
        uniform_phase = torch.full_like(table, 1.0 / float(self.num_bins))
        conditional_phase = torch.where(
            raw_clip_prob[:, None] > 0.0,
            table / raw_clip_prob[:, None].clamp_min(torch.finfo(table.dtype).tiny),
            uniform_phase,
        )
        shaped = conditional_phase * shaped_clip_prob[:, None]
        return shaped.reshape(-1)

    @property
    def sampling_probabilities(self) -> torch.Tensor:
        """Row-wise smoothed sampling table, normalized over ALL (row, bin) entries.

        Shape (num_rows * num_bins,). In shared-row modes this is the original 1-D
        distribution; in per_clip mode row r holds clip r's phase distribution and the
        global normalization makes clip selection itself failure-weighted."""
        total_cells = self.bin_failed_count.numel()
        if self.per_clip and self.num_clips > 1:
            # Normalize failure counts before mixing with uniform sampling.
            # The configured uniform probability remains independent of failure-count magnitude.


            fail = self.bin_failed_count
            fail_sum = fail.sum()
            fail_dist = fail / fail_sum if fail_sum > 0 else torch.full_like(fail, 1.0 / total_cells)
            uniform = torch.full_like(fail, 1.0 / total_cells)
            sampling_probabilities = (
                (1.0 - self.adaptive_uniform_ratio) * fail_dist + self.adaptive_uniform_ratio * uniform
            )
        else:
            # Legacy path — byte-identical to the original (single-clip / shared-row training).
            sampling_probabilities = self.bin_failed_count + self.adaptive_uniform_ratio / float(total_cells)
        # conv1d over the bin axis per row (batch=rows); replicate-pad only the bin axis.
        sampling_probabilities = torch.nn.functional.pad(
            sampling_probabilities.unsqueeze(1),
            (0, self.adaptive_kernel_size - 1),  # Non-causal kernel
            mode="replicate",
        )
        sampling_probabilities = torch.nn.functional.conv1d(sampling_probabilities, self.kernel.view(1, 1, -1)).view(-1)
        sampling_probabilities = sampling_probabilities / sampling_probabilities.sum()
        return self._shape_clip_marginal(sampling_probabilities)

    def sample(self, num_samples: int) -> torch.Tensor:
        sampled_bins = torch.multinomial(self.sampling_probabilities, num_samples, replacement=True)
        # In per_clip mode the table is flat (clip, bin); reduce to the bin within the clip so the
        # returned value is always a valid phase in [0, 1).
        sampled_bins = sampled_bins % self.num_bins
        # inside of each bin, randomly sample a time step, ignoring the borders
        return (sampled_bins + torch.rand(num_samples, device=self.device)) / self.num_bins

    def sample_clip_phase(self, num_samples: int) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint (clip, phase) draw from the per-(clip, bin) table. per_clip mode only."""
        assert self.per_clip, "sample_clip_phase requires per_clip mode"
        flat = torch.multinomial(self.sampling_probabilities, num_samples, replacement=True)
        clip_ids = flat // self.num_bins
        bins = flat % self.num_bins
        phase = (bins.float() + torch.rand(num_samples, device=self.device)) / self.num_bins
        return clip_ids, phase

    def get_stats(self):
        # Serve cached metrics between recompute intervals (see __init__).  The
        # first call always computes so metrics exist before the first log.
        if self._stats_countdown > 0 and self.metrics:
            self._stats_countdown -= 1
            return
        self._stats_countdown = self._stats_interval - 1
        # Metrics
        prob = self.sampling_probabilities
        H = -(prob * (prob + 1e-12).log()).sum()
        # A sub-second clip has one bin.  Its unique distribution is maximally
        # uniform by definition; dividing by log(1) would report NaN.
        H_norm = torch.ones_like(H) if prob.numel() == 1 else H / np.log(prob.numel())
        pmax, imax = prob.max(dim=0)
        self.metrics["sampling_entropy"] = H_norm
        self.metrics["sampling_top1_prob"] = pmax
        self.metrics["sampling_top1_bin"] = (imax % self.num_bins).float() / self.num_bins
        if self.per_clip:
            # Marginal clip distribution: how unevenly the curriculum focuses across clips.
            clip_marginal = prob.view(self.num_clips, self.num_bins).sum(dim=1)
            Hc = -(clip_marginal * (clip_marginal + 1e-12).log()).sum()
            self.metrics["clip_sampling_entropy"] = Hc / np.log(max(self.num_clips, 2))
            clip_top1_prob, clip_top1_id = clip_marginal.max(dim=0)
            self.metrics["clip_top1_prob"] = clip_top1_prob
            # HoloSoma's episode metric accumulator computes ``Tensor.mean``
            # on every value.  Keep the diagnostic ID numerically integral,
            # but expose it in the same floating dtype as the other metrics so
            # the first logging interval cannot fail on a Long tensor.
            self.metrics["clip_top1_id"] = clip_top1_id.to(dtype=prob.dtype)
            self.metrics["clip_effective_count"] = Hc.exp()


#########################################################################################################
## Helper functions
#########################################################################################################
FAKE_BODY_NAME_ALIASES: dict[str, str] = {
    # Fake foot contact bodies are authored in the URDF purely for height computation.
    # They do not exist in the motion-capture dataset, so we alias them back to the
    # closest real body when indexing into motion data. These are not actually used in training.
    "left_foot_contact_point": "left_ankle_roll_link",
    "right_foot_contact_point": "right_ankle_roll_link",
}


def get_filtered_body_names(body_list: List[str], pattern: str) -> List[str]:
    return [body_name for body_name in body_list if re.match(pattern, body_name)]


def _npz_has_object_flag(path: str) -> bool:
    """The npz ``has_object`` scalar (DATA meaning of the object track, ``data_tools.add_object_channel``); a file WITHOUT the key is
    treated as a real object (conservative).  Reads one zip member only."""
    try:
        with np.load(path, allow_pickle=False) as z:
            return bool(z["has_object"]) if "has_object" in z.files else True
    except Exception:  # noqa: BLE001 -- unreadable file: conservative
        return True


def clip_object_flags(motion: Any) -> list[bool] | None:
    """Per-clip "carries a REAL object" flags: the HERO loader's ``clip_has_object`` when present, else the npz ``has_object`` scalars of
    the stock loader's ``motion_files``; ``None`` when neither source exists (the caller then keeps the stock failure)."""
    flags = getattr(motion, "clip_has_object", None)
    if flags is not None:
        return [bool(x) for x in torch.as_tensor(flags).reshape(-1).tolist()]
    files = getattr(motion, "motion_files", None)
    if not files:
        return None
    return [_npz_has_object_flag(str(f)) for f in files]


def resolve_object_track_actor(motion: Any, simulator: Any, object_name: str = "object") -> tuple[Any, bool]:
    """Return ``(actor indices, ignored)`` for a corpus carrying an object track.

    The loader sets ``has_object`` from TRACK presence; ``data_tools.add_object_channel`` writes a PARKED dummy track
    (``has_object=False``, object 5 m below the floor) next to object-free clips so one corpus can drive an object scene.  On a scene
    Without an object actor, the lookup raises ``KeyError``; when every clip is object-free
    (``motion.clip_has_object`` all False) the track is useless there and is dropped: ``motion.has_object`` is set False, a WARNING is
    logged and ``(None, True)`` is returned.  A REAL object clip (or a loader without the per-clip flag) re-raises: those clips need an
    object scene, not a bypass.  With the actor present the indices are returned unchanged ``(indices, False)``."""
    try:
        return simulator.get_actor_indices(object_name, env_ids=None), False
    except KeyError:
        flags = clip_object_flags(motion)
        if flags is None:
            raise
        flags = torch.as_tensor(flags, dtype=torch.bool).reshape(-1)
        n_real = int(flags.sum().item())
        if n_real > 0:
            raise KeyError(
                f"scene has no '{object_name}' actor but {n_real} clip(s) carry a REAL object (has_object=True): evaluate / train them "
                f"on an object preset or drop them from the corpus; only parked dummy tracks (has_object=False) are ignored"
            )
        motion.has_object = False
        logger.warning(
            "MotionCommand: corpus object track present but every clip is object-free (parked dummy, has_object=False) and the scene has "
            "no '{}' actor -> object track ignored ({} clips)", object_name, int(flags.numel())
        )
        return None, True


class MotionCommand(CommandTermBase):
    def __init__(self, cfg: Any, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)

        self._env = env
        # Evaluation normally starts every clip at phase zero.  Evaluators that
        # need random-phase coverage can opt in without setting
        # ``env.is_evaluating=False`` (which would also enable training-only
        # augmentation/randomization paths).
        self.evaluation_random_phase = False
        # Per-step cache for motion frame gathers, keyed on time_steps identity
        # and version (see _motion_frames).
        self._motion_frame_cache: dict[str, Any] = {}
        # Convert a dictionary produced by tyro into MotionConfig.

        if isinstance(cfg.params["motion_config"], MotionConfig):
            self.motion_cfg = cfg.params["motion_config"]
        else:
            self.motion_cfg = MotionConfig(**cfg.params["motion_config"])
        self.init_pose_cfg: NoiseToInitialPoseConfig = self.motion_cfg.noise_to_initial_pose

    def setup(self) -> None:
        self.num_envs = self._env.num_envs
        self.device = self._env.device

        robot_body_names = self._env.simulator._body_list  # type: ignore[attr-defined]
        robot_body_names_alias = [FAKE_BODY_NAME_ALIASES.get(bn, bn) for bn in robot_body_names]

        robot_joint_names = self._env.simulator.dof_names  # type: ignore[attr-defined]

        # 1. load motion data
        assert self.motion_cfg.motion_file or self.motion_cfg.motion_dir, (
            "Either motion_file or motion_dir must be set in MotionConfig"
        )
        # MotionCommand advances exactly one reference frame per environment
        # control step, so the clip FPS must equal the control-loop rate.  Using
        # round here is diagnostic only; the strict comparison receives the full
        # floating-point rate below.
        self.control_fps = 1.0 / float(self._env.dt)
        self.motion: MotionLoader | MultiMotionLoader
        motion_storage_device = getattr(self.motion_cfg, "motion_storage_device", "") or None
        if self.motion_cfg.motion_dir:
            self.motion = MultiMotionLoader(
                self.motion_cfg.motion_dir,
                robot_body_names_alias,
                robot_joint_names,
                device=self.device,
                expected_fps=self.control_fps,
                storage_device=motion_storage_device,
            )
        else:
            self.motion = MotionLoader(
                self.motion_cfg.motion_file,
                robot_body_names_alias,
                robot_joint_names,
                device=self.device,
                expected_fps=self.control_fps,
                storage_device=motion_storage_device,
            )
        logger.info(
            f"Motion timebase verified: motion={self.motion.fps:g}Hz, control={self.control_fps:g}Hz "
            "(one reference frame per control step)"
        )

        # Store body and joint indexes for interpolation
        self._body_indexes_in_motion = self.motion._body_indexes
        self._joint_indexes_in_motion = self.motion._joint_indexes

        # Maybe prepend interpolated transition from default pose
        self._maybe_add_default_pose_transition(prepend=True)

        # Maybe append interpolated transition back to default pose
        self._maybe_add_default_pose_transition(prepend=False)

        # 2. get the indexes of the root link and the tracked links
        self.ref_body_index = robot_body_names.index(self.motion_cfg.body_name_ref[0])  # int
        self.tracked_body_indexes = self._get_index_of_a_in_b(
            self.motion_cfg.body_names_to_track, robot_body_names, self.device
        )
        self._setup_reference_root_velocity_frame()

        # Resolve object indices. Ignore explicitly empty object tracks when no object actor exists;
        # clips with real object tracks require an object actor.


        self.object_track_ignored = False
        if self.motion.has_object:
            self.object_name = "object"  # hardcoded object name
            indices, ignored = resolve_object_track_actor(self.motion, self._env.simulator, self.object_name)
            if ignored:
                self.object_track_ignored = True
            else:
                # cache the object_index_in_simulator
                self.object_indices_in_simulator = indices
                assert self._env.simulator.get_simulator_type() == SimulatorType.ISAACSIM, (
                    "Object is only supported in IsaacSim"
                )

        # 4. get the adaptive timesteps sampler
        if self.motion_cfg.use_adaptive_timesteps_sampler:
            phase_binning = getattr(self.motion_cfg, "adaptive_sampler_phase_binning", False)
            per_clip = getattr(self.motion_cfg, "adaptive_sampler_per_clip", False)
            # Longest single clip (frames) — sets the bin resolution in phase mode so even the
            # longest clip keeps ~1s bins. Cheap; computed from the per-clip boundaries.
            clip_lengths = (self.motion.motion_end_idx - self.motion.motion_start_idx)
            max_clip_time_step = int(clip_lengths.max().item()) if clip_lengths.numel() > 0 else self.motion.time_step_total
            self.adaptive_timesteps_sampler = AdaptiveTimestepsSampler(
                self.motion.time_step_total,
                self.device,
                int(round(self.control_fps)),
                phase_binning=phase_binning,
                max_clip_time_step=max_clip_time_step,
                per_clip=per_clip,
                num_clips=self.motion.num_motions,
                adaptive_uniform_ratio=getattr(self.motion_cfg, "adaptive_sampler_uniform_ratio", 0.1),
                adaptive_clip_temperature=getattr(
                    self.motion_cfg, "adaptive_sampler_clip_temperature", 1.0
                ),
                adaptive_clip_max_probability=getattr(
                    self.motion_cfg, "adaptive_sampler_clip_max_probability", 1.0
                ),
            )

        # 5. metrics
        self.metrics: dict[str, torch.Tensor] = {}

        self.init_buffers()

        # 6. visualization markers for isaacsim
        if self._env.viewer and self._env.simulator.get_simulator_type() == SimulatorType.ISAACSIM:
            self._setup_visualization_markers_for_isaacsim()

    def _setup_reference_root_velocity_frame(self) -> None:
        """Record the frame of the reference root velocity and the simulator row of the reference root body.

        ``reference_root_velocity_frame`` (:func:`resolve_reference_root_velocity_frames` of the
        motion source, decided PER CLIP through the loader's file list when it has one) tells the reset writers whether
        ``root_lin_vel_w`` is a COM velocity (legacy corpora: handed to the simulator unchanged) or a LINK-origin velocity
        (certified corpora: converted, see :meth:`_reset_root_lin_vel_for_simulator`).  A source mixing both is refused
        unless ``HERO_ALLOW_MIXED_VELOCITY_FRAMES=1``; then the frame is ``"mixed"`` and ``reference_root_velocity_is_link``
        (Bool[num_motions]) marks the clips whose rows are converted.  ``root_body_index`` is the SIMULATOR body row of the
        reference root: body 0 of the robot body order -- the column ``root_pos_w`` / ``root_lin_vel_w`` gather and the
        articulation root whose state ``simulator.robot_root_states`` holds (holosoma robot configs list the root link
        first).
        """
        source = self.motion_cfg.motion_dir if self.motion_cfg.motion_dir else self.motion_cfg.motion_file
        motion = getattr(self, "motion", None)
        clip_files = reference_clip_files(motion)
        resolved = resolve_reference_root_velocity_frames(source, clip_files)
        self.reference_root_velocity_frame = resolved.frame
        self.reference_root_velocity_frames_by_directory = dict(resolved.by_directory)
        self.reference_root_velocity_is_link: torch.Tensor | None = None
        self.root_body_index = 0
        if resolved.mixed:
            assert resolved.per_clip is not None  # resolve_reference_root_velocity_frames refuses mixed without per-clip frames
            is_link = torch.tensor([frame == "link" for frame in resolved.per_clip], dtype=torch.bool, device=getattr(self, "device", "cpu"))
            num_motions = getattr(motion, "num_motions", None)
            if num_motions is not None and is_link.numel() != int(num_motions):
                raise RuntimeError(
                    f"per-clip reference root velocity frames cover {is_link.numel()} clips but the loader holds {int(num_motions)}"
                )
            self.reference_root_velocity_is_link = is_link
            logger.warning(
                "Reference root velocity frame: MIXED ({}; {}={}); reset writes convert LINK-origin -> COM for the {} LINK clips "
                "and pass the {} COM clips through unchanged",
                resolved.describe(),
                ALLOW_MIXED_VELOCITY_FRAMES_ENV,
                os.environ.get(ALLOW_MIXED_VELOCITY_FRAMES_ENV, ""),
                int(is_link.sum().item()),
                int((~is_link).sum().item()),
            )
            return
        logger.info(
            "Reference root velocity frame: {} ({}; {}); reset writes {} the simulator's COM root velocity",
            self.reference_root_velocity_frame,
            source,
            "per clip" if clip_files is not None else "per motion source entry",
            "convert LINK-origin -> COM for" if self.reference_root_velocity_frame == "link" else "pass it through unchanged as",
        )

    def _reset_root_lin_vel_for_simulator(
        self,
        env_ids: torch.Tensor,
        root_quat_xyzw: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
    ) -> torch.Tensor:
        """The root linear velocity to write into ``simulator.robot_root_states[env_ids, 7:10]``.

        The generic root-state setter (IsaacLab ``write_root_velocity_to_sim``) expects the CENTRE-OF-MASS velocity.  A
        legacy corpus (``reference_root_velocity_frame == "com"``) already stores that: the (noised) reference is returned
        unchanged. A missing ``reference_root_velocity_frame`` (a ``setup()`` that never
        reached :meth:`_setup_reference_root_velocity_frame`) is an error: the writer cannot know the corpus convention, so
        it must not default to pass-through. A certified LINK-origin corpus stores the link-origin velocity,
        so ``v_com = v_link + omega x R(q) r_com`` with the FINAL (noised) orientation ``q`` / angular velocity ``omega``
        and the per-env LINK-frame COM offset of the root body (``simulator.body_com_offset_b[env_ids, root_body_index]``,
        the merged / randomized value the simulator actually uses).  Zero angular velocity leaves the value unchanged.
        ``"mixed"`` (opt-in) converts only the rows whose clip (``motion_ids[env_ids]``) is LINK-origin
        (``reference_root_velocity_is_link``).  A MuJoCo backend (free-joint ``qvel[0:3]`` = link-origin velocity) takes
        the reference unconverted; other backends require ``body_com_offset_b`` for conversion.
        """
        frame = getattr(self, "reference_root_velocity_frame", None)
        if frame is None:
            raise RuntimeError(
                f"{type(self).__name__}.reference_root_velocity_frame is not set: setup() never reached "
                "_setup_reference_root_velocity_frame(), so the reset writer cannot know whether the reference root velocity is "
                "a COM or a LINK-origin velocity; refusing to write it unconverted"
            )
        if frame == "com":
            return root_lin_vel
        if frame not in ("link", MIXED_REFERENCE_ROOT_VELOCITY_FRAME):
            raise RuntimeError(
                f"unknown reference_root_velocity_frame {frame!r}; expected one of "
                f"{REFERENCE_ROOT_VELOCITY_FRAMES + (MIXED_REFERENCE_ROOT_VELOCITY_FRAME,)}"
            )
        simulator = self._env.simulator
        offsets = getattr(simulator, "body_com_offset_b", None)
        if offsets is None:
            get_type = getattr(simulator, "get_simulator_type", None)
            if callable(get_type) and get_type() == SimulatorType.MUJOCO:
                # MuJoCo writes robot_root_states[7:10] into the free joint's qvel[0:3] = the BODY-FRAME (link-origin)
                # velocity, so a LINK-origin reference is already what that setter expects: no conversion (COM rows of a
                # mixed source keep the legacy pass-through).
                return root_lin_vel
            raise RuntimeError(
                "the simulator exposes no body_com_offset_b; a LINK-origin reference root velocity cannot be converted to the "
                "COM velocity its root-state setter expects (IsaacSim provides the accessor; legacy COM corpora need no conversion)"
            )
        r_com_b = offsets[env_ids, self.root_body_index].to(device=root_lin_vel.device, dtype=root_lin_vel.dtype)
        r_com_w = quat_apply(root_quat_xyzw, r_com_b, w_last=True)
        converted = root_lin_vel + torch.linalg.cross(root_ang_vel, r_com_w, dim=-1)
        if frame == "link":
            return converted
        is_link = getattr(self, "reference_root_velocity_is_link", None)
        if is_link is None:
            raise RuntimeError("reference_root_velocity_frame is 'mixed' but reference_root_velocity_is_link (per-clip frames) is missing")
        rows = is_link.to(device=env_ids.device)[self.motion_ids[env_ids]].to(device=root_lin_vel.device)
        return torch.where(rows.unsqueeze(-1), converted, root_lin_vel)

    def _apply_evaluation_phase_policy(
        self,
        phase: torch.Tensor,
        sampled_clip_ids: torch.Tensor | None,
    ) -> tuple[torch.Tensor, torch.Tensor | None]:
        """Apply the explicit phase-zero policy for evaluation resets.

        ``evaluation_random_phase`` changes only phase initialization.  The
        environment remains in evaluation mode, so training-only pose
        augmentation and other training branches remain disabled.
        """
        if self._env.is_evaluating and not self.evaluation_random_phase:
            return torch.zeros_like(phase), None
        return phase, sampled_clip_ids

    def _before_reset_pose_read(self, env_ids: torch.Tensor) -> None:
        """Hook run once per :meth:`reset`, after ``motion_ids`` / ``time_steps[env_ids]`` hold their FINAL values and
        immediately before the reference root pose is read (``root_pos_w[env_ids]``) and written to the simulator.

        Default: no-op (PATCHES #19).  A terrain provider that cannot rely on the lazy version-keyed refresh of
        ``reference_origins`` (e.g. ``time_steps`` is an inference tensor without ``_version``) refreshes its per-env
        origins here; ``env_ids`` is the index tensor of the envs being reset.  Nothing else in the stock reset path
        calls it, and a clip rollover (``_soft_reset_ended_clips`` -> ``reset``) reaches it the same way.
        """
        del env_ids

    def reset(self, env_ids: torch.Tensor | None) -> None:
        """called per reset_idx, reset timesteps and robot/object poses."""
        env_ids = self._ensure_index_tensor(env_ids)
        if env_ids.numel() == 0:
            return

        # WBT_POSE_AUG resamples per-environment XY offsets for training references.
        # WBT_POSE_AUG_CURR scales the radius with average episode length.
        # The shared object_pos_w property keeps reward, observations, and resets consistent.


        import os as _os
        _r = float(_os.environ.get("WBT_POSE_AUG", "0") or "0")
        if _r > 0.0 and self.motion.has_object and not self._env.is_evaluating:
            if getattr(self, "_pose_aug_offset", None) is None or self._pose_aug_offset.shape[0] != self._env.num_envs:
                self._pose_aug_offset = torch.zeros(self._env.num_envs, 3, device=self.device)
            xy = (torch.rand(env_ids.numel(), 2, device=self.device) * 2 - 1) * _r
            self._pose_aug_offset[env_ids, :2] = xy
            # WBT_POSE_AUG_Z samples elevated object starts in [0, zmax], with ground starts in half
            # the environments. The trainer places support using _pose_aug_offset.


            _zmax = float(_os.environ.get("WBT_POSE_AUG_Z", "0") or "0")
            if _zmax > 0.0:
                _z = torch.rand(env_ids.numel(), device=self.device) * _zmax
                _z = torch.where(torch.rand_like(_z) < 0.5, torch.zeros_like(_z), _z)
                self._pose_aug_offset[env_ids, 2] = _z
            else:
                self._pose_aug_offset[env_ids, 2] = 0.0

        # 0. Sample the time steps
        sampled_clip_ids = None
        if self.motion_cfg.use_adaptive_timesteps_sampler:
            # Match BeyondMimic behavior: update failed bins from environments
            # that terminated before this reset, then sample new phases.
            episode_failed = self._env.termination_manager.terminated[env_ids]
            if torch.any(episode_failed):
                failed_envs = env_ids[episode_failed]
                failed_at_time_step = self.time_steps[failed_envs]
                failed_at_phase = None
                failed_clip_ids = None
                if getattr(self.adaptive_timesteps_sampler, "phase_binning", False):
                    # Normalize the absolute failure timestep to per-clip phase [0,1] using the clip each
                    # failed env was assigned to (motion_ids was set at that env's previous reset).
                    f_mids = self.motion_ids[failed_envs]
                    f_start = self.motion.motion_start_idx[f_mids]
                    f_end = self.motion.motion_end_idx[f_mids]
                    f_len = (f_end - f_start).clamp(min=1).float()
                    failed_at_phase = ((failed_at_time_step - f_start).float() / f_len).clamp(0.0, 1.0)
                    failed_clip_ids = f_mids
                self.adaptive_timesteps_sampler.update_current_bin_failed_count(
                    failed_at_time_step, failed_at_phase=failed_at_phase, failed_clip_ids=failed_clip_ids
                )
            if getattr(self.adaptive_timesteps_sampler, "per_clip", False) and not self._env.is_evaluating:
                # Joint (clip, phase) draw from the per-(clip, bin) failure registry: a failure in
                # clip #80's lift only boosts clip #80's lift bin, and harder clips get sampled more.
                sampled_clip_ids, phase = self.adaptive_timesteps_sampler.sample_clip_phase(env_ids.numel())
            else:
                phase = self.adaptive_timesteps_sampler.sample(env_ids.numel())
        else:
            phase = torch.rand(env_ids.numel(), device=self.device)

        phase, sampled_clip_ids = self._apply_evaluation_phase_policy(phase, sampled_clip_ids)

        # For multi-motion: assign each env to a motion (jointly with phase in per_clip mode,
        # uniformly at random otherwise), then sample within that motion's range.
        n = env_ids.numel()
        num_motions = self.motion.num_motions
        if sampled_clip_ids is not None:
            self.motion_ids[env_ids] = sampled_clip_ids
        else:
            self.motion_ids[env_ids] = torch.randint(0, num_motions, (n,), device=self.device)
        start_idx = self.motion.motion_start_idx[self.motion_ids[env_ids]]
        end_idx = self.motion.motion_end_idx[self.motion_ids[env_ids]]
        motion_len = end_idx - start_idx

        self.time_steps[env_ids] = start_idx + (phase * (motion_len - 1).float()).long()

        # Handle start_at_timestep_zero_prob (reset to start of assigned motion)
        prob = self.motion_cfg.start_at_timestep_zero_prob
        if prob >= 1.0:
            self.time_steps[env_ids] = start_idx
        elif prob > 0.0:
            subset = self.time_steps[env_ids]
            rand_vals = torch.rand_like(subset, dtype=torch.float32)
            subset = torch.where(rand_vals < prob, start_idx, subset)
            self.time_steps[env_ids] = subset

        # If the motion is at the last timestep, set it to the second last timestep;
        # Otherwise, update_tasks_callback will advance the timestep to the next timestep -> out of bounds error.
        # end_idx-2 is floored at start_idx so a degenerate (<2 frame) clip can never index into the
        # previous clip of the concatenated multi-clip arrays.
        already_last_timestep_mask = self.time_steps[env_ids] >= end_idx - 1
        safe_second_last = torch.maximum(end_idx - 2, start_idx)
        self.time_steps[env_ids] = torch.where(already_last_timestep_mask, safe_second_last, self.time_steps[env_ids])

        # time_steps / motion_ids are final for these envs: let a reference-origin provider refresh (PATCHES #19)
        self._before_reset_pose_read(env_ids)

        # 1. Get the root/body poses from the motion data
        root_pos = self.root_pos_w[env_ids].clone()
        root_rot = self.root_quat_w[env_ids].clone()
        root_lin_vel = self.root_lin_vel_w[env_ids].clone()
        root_ang_vel = self.root_ang_vel_w[env_ids].clone()

        dof_pos = self.joint_pos[env_ids].clone()
        dof_vel = self.joint_vel[env_ids].clone()

        # 2. Adding noise
        # 2.1 prepare the noise scale
        dof_pos_noise = self.init_pose_cfg.dof_pos * self.init_pose_cfg.overall_noise_scale  # float
        root_pos_noise = (
            torch.tensor(
                self.init_pose_cfg.root_pos,
                device=self.device,
            )
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)
        root_rot_noise_rpy = (
            torch.tensor(
                self.init_pose_cfg.root_rot,
                device=self.device,
            )
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)
        root_vel_noise = (
            torch.tensor(
                self.init_pose_cfg.root_lin_vel,
                device=self.device,
            )
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)
        root_ang_vel_noise_rpy = (
            torch.tensor(
                self.init_pose_cfg.root_ang_vel,
                device=self.device,
            )
            * self.init_pose_cfg.overall_noise_scale
        )  # (3,)

        # 2.2 Adding noise to dof_pos, root_pos, root_vel, root_ang_vel, root_rot
        # 1.2.1 dof_pos
        target_dof_pos = (
            dof_pos + (torch.rand(dof_pos.shape, device=self.device) - 0.5) * 2 * dof_pos_noise
        )  # (num_envs, num_dofs)
        soft_joint_pos_limits = self._env.simulator.dof_pos_limits  # type: ignore[attr-defined]  # (num_dofs, 2)
        target_dof_pos = torch.clip(target_dof_pos, soft_joint_pos_limits[:, 0], soft_joint_pos_limits[:, 1])

        # 1.2.2 dof_vel no noise
        target_dof_vel = dof_vel

        # 1.2.3 root_pos
        target_root_pos = root_pos + (
            torch.rand(root_pos.shape, device=self.device) - 0.5
        ) * 2 * root_pos_noise.unsqueeze(0)  # (num_envs, 3)

        # 1.2.4 root_rot
        rand_sample_rpy = (torch.rand((len(env_ids), 3), device=self.device) - 0.5) * 2 * root_rot_noise_rpy
        orientations_delta = quat_from_euler_xyz(
            rand_sample_rpy[:, 0], rand_sample_rpy[:, 1], rand_sample_rpy[:, 2]
        )  # (num_envs, 4), xyzw
        target_root_rot = quat_mul(orientations_delta, root_rot, w_last=True)  # (num_envs, 4), xyzw

        # 1.2.5 root_lin_vel
        target_root_lin_vel = root_lin_vel + (
            torch.rand(root_lin_vel.shape, device=self.device) - 0.5
        ) * 2 * root_vel_noise.unsqueeze(0)  # (num_envs, 3)

        # 1.2.6 root_ang_vel
        target_root_ang_vel = root_ang_vel + (
            torch.rand(root_ang_vel.shape, device=self.device) - 0.5
        ) * 2 * root_ang_vel_noise_rpy.unsqueeze(0)  # (num_envs, 3)

        # 3. Set the robot states in simulator
        self._env.simulator.dof_pos[env_ids] = target_dof_pos
        self._env.simulator.dof_vel[env_ids] = target_dof_vel

        self._env.simulator.robot_root_states[env_ids, :3] = target_root_pos
        self._env.simulator.robot_root_states[env_ids, 3:7] = target_root_rot
        # the setter expects the COM velocity: a certified LINK-origin reference is converted with the FINAL (noised)
        # orientation / angular velocity; a COM reference is written unchanged.
        self._env.simulator.robot_root_states[env_ids, 7:10] = self._reset_root_lin_vel_for_simulator(
            env_ids, target_root_rot, target_root_lin_vel, target_root_ang_vel
        )
        self._env.simulator.robot_root_states[env_ids, 10:13] = target_root_ang_vel

        # 4. Set the object states in simulator
        if self.motion.has_object:
            obj_pos = self.object_pos_w[env_ids]
            obj_ori = self.object_quat_w[env_ids]
            obj_lin_vel = self.object_lin_vel_w[env_ids]

            # 4.2 add noise to the object states
            obj_pos_noise = torch.tensor(
                [self.init_pose_cfg.object_pos],
                device=self.device,
            )
            obj_pos_noise = obj_pos_noise * self.init_pose_cfg.overall_noise_scale  # (3,)
            target_obj_pos = obj_pos + (torch.rand(obj_pos.shape, device=self.device) - 0.5) * 2 * obj_pos_noise

            object_states = torch.cat(
                [target_obj_pos, obj_ori, obj_lin_vel, torch.zeros_like(obj_lin_vel)], dim=-1
            )  # (num_envs, 7)
            # 4.3 set the object states in simulator
            self._env.simulator.set_actor_states([self.object_name], env_ids, object_states)

    def _soft_reset_ended_clips(self, env_ids: torch.Tensor) -> None:
        """Teleport completed clips without carrying runtime history across the boundary.

        A clip rollover is deliberately *not* an environment reset: it must not
        end the episode, update curriculum statistics, or resample the randomized
        plant.  It is nevertheless a physical state discontinuity, so the local
        observation, controller, and contact-history state must obey the same
        boundary contract as a hard reset.
        """
        env = self._env

        # Clear history before mutating simulator state.  In particular, action
        # manager reset clears raw/previous actions and delegates to the joint
        # action term, which clears its control-delay queue and controller cache.
        if getattr(env, "observation_manager", None) is not None:
            env.observation_manager.reset(env_ids)
        if getattr(env, "action_manager", None) is not None:
            env.action_manager.reset(env_ids)

        # Resample the command and write its robot/object state into the simulator.
        # Do not call env.reset_envs_idx(): that would reset episode/curriculum/
        # randomization state and turn a soft clip boundary into a task reset.
        self.reset(env_ids)
        sim = env.simulator
        sim.set_actor_root_state_tensor_robots(env_ids, sim.robot_root_states)
        sim.set_dof_state_tensor_robots(env_ids, sim.dof_state)  # type: ignore[attr-defined]

        # Refresh before clearing contact history: IsaacSim/MuJoCo refresh copies
        # backend sensor history into the public buffer, so clearing first would
        # immediately repopulate it with pre-teleport impulses.
        sim.refresh_sim_tensors()
        sim.clear_contact_forces_history(env_ids)

        # Optional stateful reference rewards must forget anchors even when a
        # rollover resamples the SAME clip at a later phase.  Do not reset the
        # RewardManager here: its episode/curriculum statistics span rollovers.
        # Existing presets register no callbacks, so their behavior is unchanged.
        for reset_reference_state in getattr(env, "_reference_state_reset_callbacks", ()):
            reset_reference_state(env_ids)

        # Refresh the WBT pre-observation cache (notably env.base_quat) from the
        # newly refreshed simulator tensors.
        env._pre_compute_observations_callback()

    def _playback_advance_delta(self, advance_mask: torch.Tensor) -> torch.Tensor:
        """Frames to advance ``time_steps`` this step, per env.

        The base command plays back at exactly one reference frame per control
        step (``advance_mask`` only encodes the zero-phase freeze).  Subclasses
        override this to implement playback-speed augmentation; returning a
        clamped delta here — rather than correcting ``time_steps`` after the
        fact — keeps the clip-end check and the relative-target computation in
        ``step()`` consistent with the frame actually being tracked.
        """
        return advance_mask.long()

    def _handle_ended_clips(self) -> torch.Tensor:
        """Apply the configured clip-end policy and return affected environment IDs.

        Soft reset starts a new tracking sample. Hold mode retains the last valid
        reference frame without rewriting robot or object state, until the planner
        supplies new references."""

        per_motion_end = self.motion.motion_end_idx[self.motion_ids]
        ended_env_ids = torch.where(self.time_steps >= per_motion_end)[0]
        if ended_env_ids.numel() == 0:
            return ended_env_ids
        if self.motion_cfg.rollover_at_clip_end:
            self._soft_reset_ended_clips(ended_env_ids)
        else:
            # ``motion_end_idx`` is exclusive.  Clamping is required before any
            # reference property gathers from the concatenated motion tensor.
            self.time_steps[ended_env_ids] = per_motion_end[ended_env_ids] - 1
        return ended_env_ids

    def step(self) -> None:
        """called in _update_tasks_callback of the environment. (after compute_reward, before compute_observations)"""
        # 0. update time steps, all motion joint/body poses are updated automatically with the time steps.
        advance_mask = torch.ones_like(self.time_steps, dtype=torch.bool)

        # Handle freeze_at_timestep_zero_prob: for envs at their motion's start, randomly decide whether to advance
        freeze_prob = self.motion_cfg.freeze_at_timestep_zero_prob
        if freeze_prob > 0.0:
            zero_mask = self.time_steps == self.motion.motion_start_idx[self.motion_ids]
            if zero_mask.any():
                rand_vals = torch.rand(self.num_envs, device=self.device)
                freeze_mask = (rand_vals < freeze_prob) & zero_mask
                advance_mask = advance_mask & ~freeze_mask

        # Playback-speed subclasses (time warp / halt augmentation) adjust the
        # per-env advance HERE, before the clip-end check and before the
        # relative-target computation below, so every downstream invariant sees
        # the final frame index for this step.
        self.time_steps += self._playback_advance_delta(advance_mask)

        # Training defaults to BeyondMimic-style resampling.  Task-level
        # evaluators may instead hold the terminal frame to preserve physical
        # continuity across a longer externally managed horizon.
        self._handle_ended_clips()

        # 1. update body_pos_relative_w and body_quat_relative_w
        # definition of body_pos/quat_relative_w:
        # If I take this motion data and adapt it to where my robot currently is
        # (accounting for position(x, y) offset and yaw difference of a reference body),
        # what should each body part's target pose be?

        ## 1.0 get the reference body poses

        # Issue (This is a isaacgym only issue.):
        # ------------------------------------------------------------
        # In isaacgym, immediately after reset (self._env.episode_length_buf == 0), calling
        # simulator.set_actor_root_state_tensor and simulator.set_dof_state_tensor will reset
        # the robot_root_pos_w and robot_root_quat_w successfully.
        # However, the robot_body_pos_w and robot_body_quat_w are not updated successfully,
        # (since kinematic forward has not been applied yet).
        # Therefore, using robot_ref_pos_w and robot_ref_quat_w as reference body poses is not resetted correctly.

        # Solution:
        # ------------------------------------------------------------
        # if episode_length_buf == 0, use robot_root_pos_w and robot_root_quat_w as reference body.
        # else, use configured reference body as reference body.
        use_root = (self._env.episode_length_buf == 0).unsqueeze(1).float()

        ref_pos_w = self.root_pos_w * use_root + self.ref_pos_w * (1 - use_root)
        ref_quat_w = self.root_quat_w * use_root + self.ref_quat_w * (1 - use_root)
        robot_ref_pos_w = self.robot_root_pos_w * use_root + self.robot_ref_pos_w * (1 - use_root)
        robot_ref_quat_w = self.robot_root_quat_w * use_root + self.robot_ref_quat_w * (1 - use_root)

        ## 1.1 repeat to match the number of body parts
        ref_pos_w_repeat = ref_pos_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]
        ref_quat_w_repeat = ref_quat_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]
        robot_ref_pos_w_repeat = robot_ref_pos_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]
        robot_ref_quat_w_repeat = robot_ref_quat_w[:, None, :].repeat(1, len(self.motion_cfg.body_names_to_track), 1)  # type: ignore[arg-type]

        ## 1.2 compute the relative body poses
        delta_quat_w = yaw_quat(
            quat_mul(robot_ref_quat_w_repeat, quat_inverse(ref_quat_w_repeat, w_last=True), w_last=True), w_last=True
        )
        ### 1.2.1 body_quat_relative_w
        self.body_quat_relative_w = quat_mul(delta_quat_w, self.body_quat_w, w_last=True)
        ### 1.2.2 body_pos_relative_w
        delta_pos_w_height = ref_pos_w_repeat - robot_ref_pos_w_repeat
        delta_pos_w_height[..., :2] = 0.0  # adjusting for height differences
        self.body_pos_relative_w = (
            robot_ref_pos_w_repeat
            + delta_pos_w_height
            + quat_apply(delta_quat_w, self.body_pos_w - ref_pos_w_repeat, w_last=True)
        )

        ### 1.3 update the adaptive timesteps sampler
        if self.motion_cfg.use_adaptive_timesteps_sampler:
            if getattr(self, "_skip_adaptive_update_once", False):
                self._skip_adaptive_update_once = False
                self.adaptive_timesteps_sampler.current_bin_failed_count.zero_()
            else:
                self.adaptive_timesteps_sampler.update_bin_failed_count()

    @property
    def command(self) -> torch.Tensor:
        return torch.cat([self.joint_pos, self.joint_vel], dim=1)

    def future_command(self, num_future: int = 0) -> torch.Tensor:
        """Stacked reference command for offsets [0, 1, ..., num_future] frames ahead.

        For each future offset ``f`` the reference joint pos/vel is indexed at
        ``clamp(time_steps + f, max=motion_end_idx[motion_ids] - 1)`` so it never reads
        past the current clip's end (and never into the next concatenated clip in the
        multi-task loader). ``num_future=0`` reproduces :pyattr:`command` exactly.

        Returns:
            Tensor of shape ``[num_envs, ndof * 2 * (num_future + 1)]`` ordered as
            ``[pos_f0, vel_f0, pos_f1, vel_f1, ...]``.
        """
        # per-env last valid index within the current clip (end is exclusive)
        per_env_last = self.motion.motion_end_idx[self.motion_ids] - 1  # [N] long
        frames = []
        for f in range(num_future + 1):
            idx = torch.clamp(self.time_steps + f, max=per_env_last)
            frames.append(self.motion.frames("joint_pos", idx))
            frames.append(self.motion.frames("joint_vel", idx))
        return torch.cat(frames, dim=1)

    #########################################################################################
    ## Robot from motion data
    #########################################################################################
    def _motion_frames(self, name: str) -> torch.Tensor:
        """Per-step cache over motion frame gathers.

        The bulk-tensor gather ``motion.<name>[time_steps]`` is the hot path:
        reward, observation, and termination terms read these properties dozens
        of times per env step, while ``time_steps`` only changes once per step.
        The cache is keyed on ``time_steps``'s identity and in-place version
        counter, so any mutation (including by external eval harnesses that
        write ``time_steps`` directly) invalidates it.  Inference tensors do
        not track version counters — there the cache is skipped and every read
        gathers fresh (still cheap post-canonicalization: a [num_envs]-row
        gather rather than the historical full-timeline copy).  The gather runs
        through ``MotionLoader.frames`` so it also bridges a CPU
        ``storage_device`` with a single frames-only transfer.

        The cached tensors are read-only by contract: consumers that mutate
        (e.g. the reset path) must ``.clone()`` first — matching the historical
        behavior, where these gathers already returned fresh copies.
        """
        try:
            # init_buffers() rebinds time_steps to a fresh tensor whose version
            # restarts at 0, so identity is part of the key.
            key = (id(self.time_steps), self.time_steps._version)
        except RuntimeError:
            # Inference tensor (created/mutated under torch.inference_mode):
            # no version counter, no safe invalidation signal — do not cache.
            return self.motion.frames(name, self.time_steps)
        cache = self._motion_frame_cache
        if cache.get("_key") != key:
            cache.clear()
            cache["_key"] = key
        frames = cache.get(name)
        if frames is None:
            frames = self.motion.frames(name, self.time_steps)
            cache[name] = frames
        return frames

    @property
    def joint_pos(self) -> torch.Tensor:
        return self._motion_frames("joint_pos")

    @property
    def joint_vel(self) -> torch.Tensor:
        return self._motion_frames("joint_vel")

    @property
    def reference_origins(self) -> torch.Tensor:
        """Per-env world offset added to every REFERENCE position ``[num_envs, 3]`` (PATCHES #19).

        Every reference-side world position (``root_pos_w``, the anchor ``ref_pos_w``, the tracked ``body_pos_w`` incl. the
        end effectors, ``object_pos_w``, and the default-pose FK capture) is ``clip frame + reference_origins``; the
        robot-side ``robot_*`` accessors never go through it.  The default returns the scene's env origins -- the very
        tensor the simulator laid the envs out on -- so every existing preset is byte-identical.  A subclass (or an
        installed terrain provider) overrides it to put the reference on the terrain under each env (tile origin xy,
        ground height z).  It is also read during ``setup()`` (default-pose transition FK capture, env 0) BEFORE
        ``init_buffers`` creates ``time_steps`` / ``motion_ids``; an override must tolerate that (fall back to the
        scene origins until the buffers exist).
        """
        return self._env.simulator.scene.env_origins

    @property
    def body_pos_w(self) -> torch.Tensor:
        return (
            self._motion_frames("body_pos_w")[:, self.tracked_body_indexes]
            + self.reference_origins[:, None, :]
        )

    @property
    def body_quat_w(self) -> torch.Tensor:
        return self._motion_frames("body_quat_w")[:, self.tracked_body_indexes]

    @property
    def body_lin_vel_w(self) -> torch.Tensor:
        return self._motion_frames("body_lin_vel_w")[:, self.tracked_body_indexes]

    @property
    def body_ang_vel_w(self) -> torch.Tensor:
        return self._motion_frames("body_ang_vel_w")[:, self.tracked_body_indexes]

    @property
    def ref_pos_w(self) -> torch.Tensor:
        return self._motion_frames("body_pos_w")[:, self.ref_body_index] + self.reference_origins

    @property
    def ref_quat_w(self) -> torch.Tensor:
        return self._motion_frames("body_quat_w")[:, self.ref_body_index]

    @property
    def ref_lin_vel_w(self) -> torch.Tensor:
        return self._motion_frames("body_lin_vel_w")[:, self.ref_body_index]

    @property
    def ref_ang_vel_w(self) -> torch.Tensor:
        return self._motion_frames("body_ang_vel_w")[:, self.ref_body_index]

    @property
    def root_pos_w(self) -> torch.Tensor:
        return self._motion_frames("body_pos_w")[:, 0] + self.reference_origins

    @property
    def root_quat_w(self) -> torch.Tensor:
        return self._motion_frames("body_quat_w")[:, 0]

    @property
    def root_lin_vel_w(self) -> torch.Tensor:
        return self._motion_frames("body_lin_vel_w")[:, 0]

    @property
    def root_ang_vel_w(self) -> torch.Tensor:
        return self._motion_frames("body_ang_vel_w")[:, 0]

    #########################################################################################
    ## Robot from simulator
    #########################################################################################
    @property
    def robot_joint_pos(self) -> torch.Tensor:
        return self._env.simulator.dof_pos  # (num_envs, num_dofs)

    @property
    def robot_joint_vel(self) -> torch.Tensor:
        return self._env.simulator.dof_vel

    @property
    def robot_body_pos_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_pos[:, self.tracked_body_indexes, :]

    @property
    def robot_body_quat_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_rot[:, self.tracked_body_indexes, :]  # xyzw

    @property
    def robot_body_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_vel[:, self.tracked_body_indexes, :]

    @property
    def robot_body_link_lin_vel_w(self) -> torch.Tensor | None:
        """LINK-origin velocities of the tracked bodies (``simulator._rigid_body_link_vel``), the frame the clip velocities
        (finite differences of link-origin positions) live in; None when the simulator does not publish the buffer.  The
        ``robot_body_lin_vel_w`` property contains COM velocities."""
        buf = getattr(self._env.simulator, "_rigid_body_link_vel", None)
        return None if buf is None else buf[:, self.tracked_body_indexes, :]

    @property
    def robot_body_ang_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_ang_vel[:, self.tracked_body_indexes, :]

    @property
    def robot_root_pos_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, :3]  # type: ignore[attr-defined]

    @property
    def robot_root_quat_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, 3:7]  # type: ignore[attr-defined]

    @property
    def robot_root_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, 7:10]  # type: ignore[attr-defined]

    @property
    def robot_root_ang_vel_w(self) -> torch.Tensor:
        return self._env.simulator.robot_root_states[:, 10:13]  # type: ignore[attr-defined]

    @property
    def robot_ref_pos_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_pos[:, self.ref_body_index, :]

    @property
    def robot_ref_quat_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_rot[:, self.ref_body_index, :]  # xyzw

    @property
    def robot_ref_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_vel[:, self.ref_body_index, :]

    @property
    def robot_ref_link_lin_vel_w(self) -> torch.Tensor | None:
        """LINK-origin velocity of the reference body (see :attr:`robot_body_link_lin_vel_w`); None without the buffer."""
        buf = getattr(self._env.simulator, "_rigid_body_link_vel", None)
        return None if buf is None else buf[:, self.ref_body_index, :]

    @property
    def robot_ref_ang_vel_w(self) -> torch.Tensor:
        return self._env.simulator._rigid_body_ang_vel[:, self.ref_body_index, :]

    #########################################################################################
    ## Object from motion data
    #########################################################################################
    @property
    def object_pos_w(self) -> torch.Tensor:
        # Add environment origins and the per-environment WBT_POSE_AUG offset.
        # Reward, observations, termination, and reset all read this shared reference.


        base = self._motion_frames("object_pos_w") + self.reference_origins
        off = getattr(self, "_pose_aug_offset", None)
        if off is not None and off.shape[0] == base.shape[0]:
            return base + off
        return base

    @property
    def object_quat_w(self) -> torch.Tensor:
        return self._motion_frames("object_quat_w")

    @property
    def object_lin_vel_w(self) -> torch.Tensor:
        return self._motion_frames("object_lin_vel_w")

    def _object_retarget_delta(self):
        """Yaw+XY delta that maps the raw mocap frame to the robot's current (drifted) frame,
        identical to the body retarget in step() (lines ~836-859). Returns (delta_quat_w [E,4],
        delta_pos_w_height [E,3] with XY zeroed, ref_pos_w [E,3], robot_ref_pos_w [E,3]).
        Used by object_{pos,quat}_relative_w so the object reference lives in the SAME retargeted
        frame as the body reference. See object-reference-frame-bug."""
        use_root = (self._env.episode_length_buf == 0).unsqueeze(1).float()
        ref_pos_w = self.root_pos_w * use_root + self.ref_pos_w * (1 - use_root)
        ref_quat_w = self.root_quat_w * use_root + self.ref_quat_w * (1 - use_root)
        robot_ref_pos_w = self.robot_root_pos_w * use_root + self.robot_ref_pos_w * (1 - use_root)
        robot_ref_quat_w = self.robot_root_quat_w * use_root + self.robot_ref_quat_w * (1 - use_root)
        delta_quat_w = yaw_quat(
            quat_mul(robot_ref_quat_w, quat_inverse(ref_quat_w, w_last=True), w_last=True), w_last=True
        )
        delta_pos_w_height = (ref_pos_w - robot_ref_pos_w).clone()
        delta_pos_w_height[..., :2] = 0.0  # keep Z (height) delta only, hold XY at the robot's frame
        return delta_quat_w, delta_pos_w_height, ref_pos_w, robot_ref_pos_w

    @property
    def object_pos_relative_w(self) -> torch.Tensor:
        """Object reference retargeted into the robot's current frame, mirroring body_pos_relative_w.
        object_pos_w is raw mocap-frame; this re-anchors it so faithful body tracking carries the box
        to where the reward/termination expects it even after the robot drifts in XY/yaw."""
        delta_quat_w, delta_pos_w_height, ref_pos_w, robot_ref_pos_w = self._object_retarget_delta()
        return robot_ref_pos_w + delta_pos_w_height + quat_apply(
            delta_quat_w, self.object_pos_w - ref_pos_w, w_last=True
        )

    @property
    def object_quat_relative_w(self) -> torch.Tensor:
        """Object orientation reference retargeted by the same yaw delta as the body reference."""
        delta_quat_w, _, _, _ = self._object_retarget_delta()
        return quat_mul(delta_quat_w, self.object_quat_w, w_last=True)

    #########################################################################################
    ## Object from simulator
    #########################################################################################
    @property
    def simulator_object_pos_w(self) -> torch.Tensor:
        return self._env.simulator.all_root_states[self.object_indices_in_simulator][:, :3]

    @property
    def simulator_object_quat_w(self) -> torch.Tensor:
        return self._env.simulator.all_root_states[self.object_indices_in_simulator][:, 3:7]

    @property
    def simulator_object_lin_vel_w(self) -> torch.Tensor:
        return self._env.simulator.all_root_states[self.object_indices_in_simulator][:, 7:10]

    #########################################################################################
    ## Methods that does not fit into setup/step/reset pattern
    #########################################################################################

    def init_buffers(self, *, reset_adaptive_sampler: bool = True):
        # time_steps keys the per-step motion frame cache via its in-place
        # version counter (see _motion_frames).  Environment setup runs under
        # torch.inference_mode, where created tensors carry no version counter;
        # locally disabling inference mode makes time_steps an ordinary tensor
        # (ordinary tensors can still be read and mutated inside inference-mode
        # regions, so all existing call sites are unaffected).
        with torch.inference_mode(mode=False):
            self.time_steps = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.motion_ids = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self.body_pos_relative_w = torch.zeros(
            self.num_envs, len(self.motion_cfg.body_names_to_track), 3, device=self.device
        )  # type: ignore[arg-type]
        self.body_quat_relative_w = torch.zeros(
            self.num_envs, len(self.motion_cfg.body_names_to_track), 4, device=self.device
        )  # type: ignore[arg-type]
        self.body_quat_relative_w[:, :, 0] = 1.0

        if self.motion_cfg.use_adaptive_timesteps_sampler:
            if reset_adaptive_sampler:
                self.adaptive_timesteps_sampler.init_buffers()
            else:
                # A full simulator reset invalidates only failures pending from
                # the interrupted step.  The learned EMA is curriculum state and
                # must survive PPO learn()/export() resets and checkpoint resume.
                self.adaptive_timesteps_sampler.current_bin_failed_count.zero_()

    def _wrist_body_indices(self):
        """Indices of the tracked wrist/hand bodies, or None when none are tracked.

        Matched by substring rather than an exact name list so a change of hand
        geometry (paddle vs half-sphere, whose link names differ) keeps reporting
        this metric instead of silently dropping it.
        """
        cached = getattr(self, "_wrist_idx_cache", None)
        if cached is not None:
            return cached[0]
        names = list(getattr(self.motion_cfg, "body_names_to_track", []))
        idx = [i for i, name in enumerate(names) if "wrist" in name or "hand" in name]
        # Cached in a 1-tuple so "resolved to None" is distinguishable from
        # "not resolved yet" and the name scan runs once, not every step.
        resolved = torch.tensor(idx, dtype=torch.long, device=self.device) if idx else None
        self._wrist_idx_cache = (resolved,)
        return resolved

    def update_metrics(self):
        """Update the metrics. After action, before step() is called."""
        self.metrics["motion/error_ref_pos"] = torch.norm(self.ref_pos_w - self.robot_ref_pos_w, dim=-1)
        self.metrics["motion/error_ref_rot"] = quat_error_magnitude(self.ref_quat_w, self.robot_ref_quat_w)
        self.metrics["motion/error_ref_lin_vel"] = torch.norm(self.ref_lin_vel_w - self.robot_ref_lin_vel_w, dim=-1)
        self.metrics["motion/error_ref_ang_vel"] = torch.norm(self.ref_ang_vel_w - self.robot_ref_ang_vel_w, dim=-1)

        self.metrics["motion/error_body_pos"] = torch.norm(
            self.body_pos_relative_w - self.robot_body_pos_w, dim=-1
        ).mean(dim=-1)

        self.metrics["motion/error_body_rot"] = quat_error_magnitude(
            self.body_quat_relative_w, self.robot_body_quat_w
        ).mean(dim=-1)

        self.metrics["motion/error_body_lin_vel"] = torch.norm(
            self.body_lin_vel_w - self.robot_body_lin_vel_w, dim=-1
        ).mean(dim=-1)
        self.metrics["motion/error_body_ang_vel"] = torch.norm(
            self.body_ang_vel_w - self.robot_body_ang_vel_w, dim=-1
        ).mean(dim=-1)
        # Match the clip's link-origin velocity convention; the *_lin_vel metrics above use COM velocities.
        body_link_vel = self.robot_body_link_lin_vel_w
        if body_link_vel is not None:
            self.metrics["motion/error_body_lin_vel_link"] = torch.norm(self.body_lin_vel_w - body_link_vel, dim=-1).mean(dim=-1)
        ref_link_vel = self.robot_ref_link_lin_vel_w
        if ref_link_vel is not None:
            self.metrics["motion/error_ref_lin_vel_link"] = torch.norm(self.ref_lin_vel_w - ref_link_vel, dim=-1)

        self.metrics["motion/error_joint_pos"] = torch.norm(self.joint_pos - self.robot_joint_pos, dim=-1)
        self.metrics["motion/error_joint_vel"] = torch.norm(self.joint_vel - self.robot_joint_vel, dim=-1)

        # Root error split into horizontal and vertical parts. The fused 3-D
        # `error_ref_pos` above cannot distinguish "drifting away from the
        # reference path" from "standing at the wrong height", and for box carry
        # those call for opposite fixes: a crouch deep enough to reach a low box
        # and an outright fall both raise the fused norm.
        _root_err = self.ref_pos_w - self.robot_ref_pos_w
        self.metrics["motion/error_root_xy"] = torch.norm(_root_err[:, :2], dim=-1)
        self.metrics["motion/error_root_z"] = torch.abs(_root_err[:, 2])

        # Wrist-only body error. The all-body mean above is dominated by the legs,
        # so a hand that misses the box barely moves it — yet that is exactly what
        # decides whether a carry succeeds.
        _wrist_idx = self._wrist_body_indices()
        if _wrist_idx is not None:
            _wrist_err = self.body_pos_relative_w[:, _wrist_idx] - self.robot_body_pos_w[:, _wrist_idx]
            self.metrics["motion/error_wrist_pos"] = torch.norm(_wrist_err, dim=-1).mean(dim=-1)

        if self.motion_cfg.use_adaptive_timesteps_sampler:
            self.adaptive_timesteps_sampler.get_stats()
            self.metrics["motion/adaptive_timesteps_sampler_entropy"] = self.adaptive_timesteps_sampler.metrics[
                "sampling_entropy"
            ]
            self.metrics["motion/adaptive_timesteps_sampler_top1_prob"] = self.adaptive_timesteps_sampler.metrics[
                "sampling_top1_prob"
            ]
            self.metrics["motion/adaptive_timesteps_sampler_top1_bin"] = self.adaptive_timesteps_sampler.metrics[
                "sampling_top1_bin"
            ]
            if "clip_sampling_entropy" in self.adaptive_timesteps_sampler.metrics:
                self.metrics["motion/adaptive_sampler_clip_entropy"] = self.adaptive_timesteps_sampler.metrics[
                    "clip_sampling_entropy"
                ]
                self.metrics["motion/adaptive_sampler_clip_top1_prob"] = self.adaptive_timesteps_sampler.metrics[
                    "clip_top1_prob"
                ]
                self.metrics["motion/adaptive_sampler_clip_top1_id"] = self.adaptive_timesteps_sampler.metrics[
                    "clip_top1_id"
                ]
                self.metrics["motion/adaptive_sampler_clip_effective_count"] = (
                    self.adaptive_timesteps_sampler.metrics["clip_effective_count"]
                )

    #########################################################################################
    ## Internal helpers
    #########################################################################################
    def _maybe_add_default_pose_transition(self, *, prepend: bool) -> None:
        """Shared path for optionally inserting default-pose interpolation before/after the clip."""
        enabled = self.motion_cfg.enable_default_pose_prepend if prepend else self.motion_cfg.enable_default_pose_append
        if not enabled:
            return

        duration = (
            self.motion_cfg.default_pose_prepend_duration_s
            if prepend
            else self.motion_cfg.default_pose_append_duration_s
        )
        if duration <= 0.0:
            return

        num_steps = round(duration / self._env.dt)
        if num_steps <= 1:
            logger.warning(
                "Default pose {} duration {}s is too short for dt {}; skipping augmentation.",
                "prepend" if prepend else "append",
                duration,
                self._env.dt,
            )
            return

        action = "prepend" if prepend else "append"

        if isinstance(self.motion, MultiMotionLoader):
            # Insert default-pose transitions into each clip so transitions retain the clip identity.


            log_str = (f"per-clip {action} {num_steps} interpolated frames ({duration}s) "
                       f"x {self.motion.num_motions} clips")
            try:
                self._add_per_clip_transition_to_motion(num_steps, prepend=prepend)
                logger.info(log_str)
            except Exception as exc:
                logger.error(f"Failed to {action} per-clip default pose transition: {exc}")
                raise RuntimeError(
                    f"Critical error during per-clip motion interpolation setup: {exc}"
                ) from exc
            return

        default_state = self._build_default_pose_state(use_motion_end=not prepend)
        log_str = f"{action} {num_steps} interpolated frames ({duration}s) from default pose to motion"
        try:
            self._add_transition_to_motion(default_state, num_steps, prepend=prepend)
            logger.info(log_str)
        except Exception as exc:
            logger.error(f"Failed to {action} default pose transition: {exc}")
            raise RuntimeError(
                f"Critical error during motion interpolation setup: {exc}\n"
                "This indicates a mismatch in tensor dimensions during interpolation. "
                "Please check that the motion file and robot configuration are compatible."
            ) from exc

    def _add_per_clip_transition_to_motion(self, num_steps: int, prepend: bool) -> None:
        """Build one default-pose transition PER clip (anchored to that clip's own start/end frame)
        and splice each into its own clip's phase range via MultiMotionLoader.extend_each_clip_with_segments.
        Clip count is unchanged; transition frames belong to their clip (not a new sampled clip)."""
        assert isinstance(self.motion, MultiMotionLoader), (
            "per-clip transition path requires MultiMotionLoader"
        )
        assert self.motion.num_motions >= 1, "MultiMotionLoader must contain a clip"
        device = self.device
        dtype = self.motion._joint_pos.dtype
        starts = self.motion.motion_start_idx.tolist()
        ends = self.motion.motion_end_idx.tolist()

        alphas = torch.linspace(0.0, 1.0, steps=num_steps + 1, device=device, dtype=dtype)
        # prepend: default->clip0, drop the last alpha (=clip frame 0, already in the clip);
        # append: clip_end->default, drop the first alpha (=clip last frame, already in the clip).
        alphas = alphas[:-1] if prepend else alphas[1:]
        n = alphas.numel()
        alphas_joint = alphas.view(n, 1)
        alphas_body = alphas.view(n, 1, 1)

        per_clip_segments = []
        for c in range(self.motion.num_motions):
            anchor = int(starts[c]) if prepend else int(ends[c]) - 1  # clip c's own boundary frame
            default_state = self._build_default_pose_state_at(anchor)
            default_motion_state = self._default_motion_state(default_state, dtype=dtype, device=device)
            motion_state = self._motion_state(anchor, dtype=dtype, device=device)
            start_state = default_motion_state if prepend else motion_state
            target_state = motion_state if prepend else default_motion_state
            seg = self._build_transition_segments(start_state, target_state, alphas, alphas_joint, alphas_body)
            per_clip_segments.append(seg)

        self.motion = self.motion.extend_each_clip_with_segments(per_clip_segments, prepend=prepend)

    def _build_default_pose_state(self, use_motion_end: bool = False) -> dict[str, torch.Tensor]:
        """Build the default standing pose anchored to the WHOLE motion's start (or end).

        Single-clip / legacy path: motion start = frame 0, motion end = frame -1.
        """
        anchor = (self.motion.time_step_total - 1) if use_motion_end else 0
        return self._build_default_pose_state_at(int(anchor))

    def _build_default_pose_state_at(self, anchor_frame: int) -> dict[str, torch.Tensor]:
        """Build the robot's default standing pose, anchoring root x/y and yaw to a SPECIFIC motion
        frame index (so multi-clip per-clip transitions can each anchor to their own clip boundary)."""
        init_state = self._env.robot_config.init_state
        joint_pos = self._env.default_dof_pos_base.squeeze(0).to(self.device)
        joint_vel = torch.zeros_like(joint_pos)

        init_root_quat = torch.tensor(init_state.rot, dtype=torch.float32, device=self.device).unsqueeze(0)
        init_roll, init_pitch, _ = get_euler_xyz(init_root_quat, w_last=True)

        motion_idx = anchor_frame

        # Assume the pelvis is the first in robot_body_names
        motion_root_pos = self.motion.body_pos_w[motion_idx, 0].to(self.device)
        motion_root_quat = self.motion.body_quat_w[motion_idx, 0].to(self.device).unsqueeze(0)
        _, _, motion_yaw = get_euler_xyz(motion_root_quat, w_last=True)

        # Keep z from init config but adopt the clip's x,y at the chosen anchor frame.
        default_root_pos = torch.tensor(
            [motion_root_pos[0], motion_root_pos[1], init_state.pos[2]],
            dtype=torch.float32,
            device=self.device,
        ).unsqueeze(0)
        # Keep roll/pitch from init config but adopt the clip's yaw at the chosen anchor frame.
        default_root_quat = quat_from_euler_xyz(
            init_roll.squeeze(0),
            init_pitch.squeeze(0),
            motion_yaw.squeeze(0),
        )
        default_root_lin_vel = torch.tensor(init_state.lin_vel, dtype=torch.float32, device=self.device)
        default_root_ang_vel = torch.tensor(init_state.ang_vel, dtype=torch.float32, device=self.device)

        body_states = self._capture_body_states(
            joint_pos,
            joint_vel,
            default_root_pos,
            default_root_quat,
            default_root_lin_vel,
            default_root_ang_vel,
        )

        default_body_pos = self._map_robot_bodies_to_motion_order(body_states["pos"])
        default_body_quat = self._map_robot_bodies_to_motion_order(body_states["quat"])
        default_body_lin_vel = self._map_robot_bodies_to_motion_order(body_states["lin_vel"])
        default_body_ang_vel = self._map_robot_bodies_to_motion_order(body_states["ang_vel"])

        if self.motion.has_object:
            object_pos = self.motion._object_pos_w[motion_idx].to(self.device)
            object_quat = self.motion._object_quat_w[motion_idx].to(self.device)
            object_lin_vel = self.motion._object_lin_vel_w[motion_idx].to(self.device)
        else:
            object_pos = torch.zeros(0, 3, device=self.device, dtype=torch.float32)
            object_quat = torch.zeros(0, 4, device=self.device, dtype=torch.float32)
            object_lin_vel = torch.zeros(0, 3, device=self.device, dtype=torch.float32)

        return {
            "joint_pos": joint_pos.clone(),
            "joint_vel": joint_vel,
            "root_pos": default_root_pos,
            "root_quat": default_root_quat,
            "root_lin_vel": default_root_lin_vel,
            "root_ang_vel": default_root_ang_vel,
            "body_pos": default_body_pos,
            "body_quat": default_body_quat,
            "body_lin_vel": default_body_lin_vel,
            "body_ang_vel": default_body_ang_vel,
            "object_pos": object_pos,
            "object_quat": object_quat,
            "object_lin_vel": object_lin_vel,
        }

    def _add_transition_to_motion(self, default_state: dict[str, torch.Tensor], num_steps: int, prepend: bool) -> None:
        """Add interpolated frames either before or after the motion data."""
        assert self._body_indexes_in_motion is not None
        assert self._joint_indexes_in_motion is not None

        if num_steps <= 0:
            return

        device = self.device
        dtype = self.motion._joint_pos.dtype

        default_motion_state = self._default_motion_state(default_state, dtype=dtype, device=device)
        motion_state = self._motion_state(0 if prepend else -1, dtype=dtype, device=device)

        start_state = default_motion_state if prepend else motion_state
        target_state = motion_state if prepend else default_motion_state
        drop_first, drop_last = (False, True) if prepend else (True, False)

        self._build_and_apply_transition(
            start_state=start_state,
            target_state=target_state,
            num_steps=num_steps,
            prepend=prepend,
            drop_first=drop_first,
            drop_last=drop_last,
            dtype=dtype,
            device=device,
        )

    def _slerp_quat_sequence(self, start: torch.Tensor, end: torch.Tensor, alphas: torch.Tensor) -> torch.Tensor:
        """Spherically interpolate quaternions across multiple time steps."""
        if alphas.numel() == 0:
            return start.new_zeros((0,) + start.shape)

        num_steps = alphas.shape[0]
        start_expand = start.unsqueeze(0).expand(num_steps, -1, -1)
        end_expand = end.unsqueeze(0).expand(num_steps, -1, -1)
        alpha_flat = alphas.repeat_interleave(start.shape[0]).unsqueeze(-1)
        blended = slerp(
            start_expand.reshape(-1, 4),
            end_expand.reshape(-1, 4),
            alpha_flat,
        )
        return blended.view(num_steps, start.shape[0], 4)

    def _capture_body_states(
        self,
        joint_pos: torch.Tensor,
        joint_vel: torch.Tensor,
        root_pos: torch.Tensor,
        root_quat: torch.Tensor,
        root_lin_vel: torch.Tensor,
        root_ang_vel: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Capture body states by temporarily setting the robot state in the simulator."""
        simulator = self._env.simulator
        assert simulator.get_simulator_type() == SimulatorType.ISAACSIM, (
            "Default-pose interpolation only supports IsaacSim; IsaacGym write_state_updates does not run FK."
        )
        env_id = 0
        # reference-side origin (PATCHES #19): the FK result is expressed relative to the same origin the reference
        # positions are offset by; identical to the scene origin unless a terrain provider overrides reference_origins
        env_origin = self.reference_origins[env_id].to(self.device)

        root_backup = simulator.robot_root_states[env_id].clone()
        dof_pos_backup = simulator.dof_pos[env_id].clone()
        dof_vel_backup = simulator.dof_vel[env_id].clone()

        try:
            simulator.robot_root_states[env_id, :3] = root_pos + env_origin
            simulator.robot_root_states[env_id, 3:7] = root_quat
            simulator.robot_root_states[env_id, 7:10] = root_lin_vel
            simulator.robot_root_states[env_id, 10:13] = root_ang_vel
            simulator.dof_pos[env_id] = joint_pos
            simulator.dof_vel[env_id] = joint_vel

            simulator.set_actor_root_state_tensor_robots()
            simulator.set_dof_state_tensor_robots()
            simulator.write_state_updates()
            simulator.refresh_sim_tensors()

            body_pos = (simulator._rigid_body_pos[env_id] - env_origin).clone()
            body_quat = simulator._rigid_body_rot[env_id].clone()
            body_lin_vel = simulator._rigid_body_vel[env_id].clone()
            body_ang_vel = simulator._rigid_body_ang_vel[env_id].clone()
        finally:
            simulator.robot_root_states[env_id] = root_backup
            simulator.dof_pos[env_id] = dof_pos_backup
            simulator.dof_vel[env_id] = dof_vel_backup
            simulator.set_actor_root_state_tensor_robots()
            simulator.set_dof_state_tensor_robots()
            simulator.write_state_updates()
            simulator.refresh_sim_tensors()

        return {
            "pos": body_pos,
            "quat": body_quat,
            "lin_vel": body_lin_vel,
            "ang_vel": body_ang_vel,
        }

    def _map_robot_bodies_to_motion_order(self, robot_tensor: torch.Tensor) -> torch.Tensor:
        """Map robot body tensor to motion data order using body indexes."""
        assert self._body_indexes_in_motion is not None
        num_motion_bodies = self.motion._body_pos_w.shape[1]
        motion_shape = (num_motion_bodies,) + robot_tensor.shape[1:]
        motion_tensor = torch.zeros(motion_shape, device=robot_tensor.device, dtype=robot_tensor.dtype)
        motion_tensor[self._body_indexes_in_motion] = robot_tensor
        return motion_tensor

    def _map_robot_joints_to_motion_order(
        self, robot_tensor: torch.Tensor, num_motion_joints: int | None = None
    ) -> torch.Tensor:
        """Map robot joint tensor to motion data order using joint indexes."""
        assert self._joint_indexes_in_motion is not None
        if num_motion_joints is None:
            num_motion_joints = self.motion._joint_pos.shape[1]
        motion_shape = robot_tensor.shape[:-1] + (num_motion_joints,)
        motion_tensor = torch.zeros(motion_shape, device=robot_tensor.device, dtype=robot_tensor.dtype)
        motion_tensor[..., self._joint_indexes_in_motion] = robot_tensor
        return motion_tensor

    def _motion_state(self, idx: int, dtype: torch.dtype, device: torch.device) -> dict[str, torch.Tensor]:
        """Slice motion tensors at a given index into a state dict."""
        state = {
            "joint_pos": self.motion._joint_pos[idx].to(device=device, dtype=dtype),
            "joint_vel": self.motion._joint_vel[idx].to(device=device, dtype=dtype),
            "body_pos": self.motion._body_pos_w[idx].to(device=device, dtype=dtype),
            "body_quat": self.motion._body_quat_w[idx].to(device=device, dtype=dtype),
            "body_lin_vel": self.motion._body_lin_vel_w[idx].to(device=device, dtype=dtype),
            "body_ang_vel": self.motion._body_ang_vel_w[idx].to(device=device, dtype=dtype),
        }
        if self.motion.has_object:
            state["object_pos"] = self.motion._object_pos_w[idx].to(device=device, dtype=dtype)
            state["object_quat"] = self.motion._object_quat_w[idx].to(device=device, dtype=dtype)
            state["object_lin_vel"] = self.motion._object_lin_vel_w[idx].to(device=device, dtype=dtype)
        return state

    def _default_motion_state(
        self, default_state: dict[str, torch.Tensor], dtype: torch.dtype, device: torch.device
    ) -> dict[str, torch.Tensor]:
        """Map default robot-state tensors into motion order for interpolation."""
        state = {
            "joint_pos": self._map_robot_joints_to_motion_order(
                default_state["joint_pos"].to(device=device, dtype=dtype),
                num_motion_joints=self.motion._joint_pos.shape[1],
            ),
            "joint_vel": self._map_robot_joints_to_motion_order(
                default_state["joint_vel"].to(device=device, dtype=dtype),
                num_motion_joints=self.motion._joint_vel.shape[1],
            ),
            "body_pos": default_state["body_pos"].to(device=device, dtype=dtype),
            "body_quat": default_state["body_quat"].to(device=device, dtype=dtype),
            "body_lin_vel": default_state["body_lin_vel"].to(device=device, dtype=dtype),
            "body_ang_vel": default_state["body_ang_vel"].to(device=device, dtype=dtype),
        }
        if self.motion.has_object:
            state["object_pos"] = default_state["object_pos"].to(device=device, dtype=dtype)
            state["object_quat"] = default_state["object_quat"].to(device=device, dtype=dtype)
            state["object_lin_vel"] = default_state["object_lin_vel"].to(device=device, dtype=dtype)
        return state

    def _build_transition_segments(
        self,
        start: dict[str, torch.Tensor],
        target: dict[str, torch.Tensor],
        alphas: torch.Tensor,
        alphas_joint: torch.Tensor,
        alphas_body: torch.Tensor,
    ) -> dict[str, torch.Tensor]:
        """Build a kinematically and temporally consistent transition reference.

        Interpolate joint and root poses, then recompute body poses with simulator FK.
        Compute velocity channels by finite differences in chronological order."""

        def _lerp(a: torch.Tensor, b: torch.Tensor, view: torch.Tensor) -> torch.Tensor:
            return a.unsqueeze(0) + view * (b - a).unsqueeze(0)

        if alphas.ndim != 1 or alphas_joint.shape != (alphas.numel(), 1):
            raise ValueError(
                "transition alpha shapes must be [T] and [T,1], got "
                f"{tuple(alphas.shape)}/{tuple(alphas_joint.shape)}"
            )

        joint_pos = _lerp(start["joint_pos"], target["joint_pos"], alphas_joint)
        root_motion_index = int(self._body_indexes_in_motion[0].item())
        root_pos = _lerp(
            start["body_pos"][root_motion_index],
            target["body_pos"][root_motion_index],
            alphas_joint,
        )
        root_quat = self._slerp_quat_sequence(
            start["body_quat"][root_motion_index].unsqueeze(0),
            target["body_quat"][root_motion_index].unsqueeze(0),
            alphas,
        ).squeeze(1)

        body_pos_frames: list[torch.Tensor] = []
        body_quat_frames: list[torch.Tensor] = []
        zero_joint_vel = torch.zeros(
            len(self._joint_indexes_in_motion), device=joint_pos.device, dtype=joint_pos.dtype
        )
        zero_root_vel = torch.zeros(3, device=joint_pos.device, dtype=joint_pos.dtype)
        for index in range(alphas.numel()):
            robot_joint_pos = joint_pos[index, self._joint_indexes_in_motion]
            body_state = self._capture_body_states(
                robot_joint_pos,
                zero_joint_vel,
                root_pos[index],
                root_quat[index],
                zero_root_vel,
                zero_root_vel,
            )
            body_pos_frames.append(self._map_robot_bodies_to_motion_order(body_state["pos"]))
            body_quat_frames.append(self._map_robot_bodies_to_motion_order(body_state["quat"]))

        body_pos = torch.stack(body_pos_frames, dim=0)
        body_quat = torch.stack(body_quat_frames, dim=0)

        # The append path drops alpha=0 and therefore has a real previous frame
        # (the original clip endpoint).  The prepend path includes alpha=0, for
        # which a forward difference is the only non-duplicated derivative.
        has_previous = bool(alphas.numel() > 0 and float(alphas[0].item()) > 1.0e-8)
        previous_joint_pos = start["joint_pos"] if has_previous else None
        previous_body_pos = start["body_pos"] if has_previous else None
        previous_body_quat = start["body_quat"] if has_previous else None

        segments = {
            "joint_pos": joint_pos,
            "joint_vel": self._finite_difference_linear(joint_pos, previous_joint_pos),
            "body_pos": body_pos,
            "body_lin_vel": self._finite_difference_linear(body_pos, previous_body_pos),
            "body_ang_vel": self._finite_difference_quaternion(body_quat, previous_body_quat),
            "body_quat": body_quat,
        }

        if self.motion.has_object:
            object_pos = _lerp(start["object_pos"], target["object_pos"], alphas_joint)
            segments["object_pos"] = object_pos
            segments["object_lin_vel"] = self._finite_difference_linear(
                object_pos, start["object_pos"] if has_previous else None
            )
            segments["object_quat"] = self._slerp_quat_sequence(
                start["object_quat"].unsqueeze(0), target["object_quat"].unsqueeze(0), alphas
            ).squeeze(1)

        return segments

    def _finite_difference_linear(
        self, values: torch.Tensor, previous: torch.Tensor | None
    ) -> torch.Tensor:
        """Finite-difference a chronological pose sequence at the control rate."""
        if values.ndim < 2 or values.shape[0] == 0:
            raise ValueError(f"linear transition values must be nonempty [T,...], got {values.shape}")
        velocity = torch.zeros_like(values)
        inv_dt = 1.0 / float(self._env.dt)
        if previous is not None:
            if previous.shape != values.shape[1:]:
                raise ValueError(
                    f"previous linear state shape {previous.shape} != {values.shape[1:]}"
                )
            velocity[0] = (values[0] - previous) * inv_dt
        elif values.shape[0] > 1:
            velocity[0] = (values[1] - values[0]) * inv_dt
        if values.shape[0] > 1:
            velocity[1:] = (values[1:] - values[:-1]) * inv_dt
        return velocity

    def _finite_difference_quaternion(
        self, quaternions: torch.Tensor, previous: torch.Tensor | None
    ) -> torch.Tensor:
        """World-frame angular velocity from an ``xyzw`` quaternion sequence."""
        if quaternions.ndim < 3 or quaternions.shape[0] == 0 or quaternions.shape[-1] != 4:
            raise ValueError(
                "quaternion transition values must be nonempty [T,...,4], got "
                f"{quaternions.shape}"
            )
        angular_velocity = torch.zeros(quaternions.shape[:-1] + (3,), device=quaternions.device,
                                       dtype=quaternions.dtype)
        if previous is not None:
            if previous.shape != quaternions.shape[1:]:
                raise ValueError(
                    f"previous quaternion state shape {previous.shape} != {quaternions.shape[1:]}"
                )
            prior = torch.cat((previous.unsqueeze(0), quaternions[:-1]), dim=0)
        elif quaternions.shape[0] > 1:
            prior = torch.cat((quaternions[:1], quaternions[:-1]), dim=0)
        else:
            return angular_velocity

        delta = quat_mul(
            quaternions,
            quat_conjugate(prior, w_last=True),
            w_last=True,
        )
        # quat_to_angle_axis()[1] is the signed axis-angle vector despite the
        # historical tuple name; dividing by dt gives world angular velocity.
        axis_angle = quat_to_angle_axis(delta)[1]
        angular_velocity[:] = axis_angle / float(self._env.dt)
        if previous is None and quaternions.shape[0] > 1:
            first_delta = quat_mul(
                quaternions[1], quat_conjugate(quaternions[0], w_last=True), w_last=True
            )
            angular_velocity[0] = quat_to_angle_axis(first_delta)[1] / float(self._env.dt)
        return angular_velocity

    def _apply_transition_segments(self, segments: dict[str, torch.Tensor], prepend: bool) -> None:
        """Splice interpolated segments into motion data, either prepending or appending."""
        self.motion = self.motion.extend_with_segments(segments, prepend=prepend)

    def _build_and_apply_transition(
        self,
        start_state: dict[str, torch.Tensor],
        target_state: dict[str, torch.Tensor],
        num_steps: int,
        prepend: bool,
        drop_first: bool,
        drop_last: bool,
        dtype: torch.dtype,
        device: torch.device,
    ) -> None:
        """Shared interpolation path for prepend/append transitions."""
        if num_steps <= 0:
            return

        alphas = torch.linspace(0.0, 1.0, steps=num_steps + 1, device=device, dtype=dtype)
        if drop_first:
            alphas = alphas[1:]
        if drop_last:
            alphas = alphas[:-1]
        if alphas.numel() == 0:
            return

        alphas_joint = alphas.view(num_steps, 1)
        alphas_body = alphas.view(num_steps, 1, 1)

        segments = self._build_transition_segments(start_state, target_state, alphas, alphas_joint, alphas_body)
        self._apply_transition_segments(segments, prepend=prepend)

    def _setup_visualization_markers_for_isaacsim(self):
        from isaaclab.markers import VisualizationMarkers
        from isaaclab.markers.config import FRAME_MARKER_CFG, RAY_CASTER_MARKER_CFG

        visualization_markers_cfg = FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/Command/real_robot",
        )
        visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
        real_robot_visualizer = VisualizationMarkers(visualization_markers_cfg)

        visualization_markers_cfg = FRAME_MARKER_CFG.replace(
            prim_path="/Visuals/Command/motion_robot",
        )
        visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
        motion_robot_visualizer = VisualizationMarkers(visualization_markers_cfg)
        self.visualization_markers = {
            "real_robot": real_robot_visualizer,
            "motion_robot": motion_robot_visualizer,
        }

        for body_names in self.motion_cfg.body_names_to_track:
            visualization_markers_cfg = RAY_CASTER_MARKER_CFG.replace(
                prim_path=f"/Visuals/Command/motion_robot_body/motion_{body_names}",
            )
            visualization_markers_cfg.markers["hit"].radius = 0.03
            visualization_markers_cfg.markers["hit"].visual_material.diffuse_color = (0.0, 1.0, 0.0)
            self.visualization_markers[f"motion_{body_names}"] = VisualizationMarkers(visualization_markers_cfg)

        if self.motion.has_object:
            visualization_markers_cfg = FRAME_MARKER_CFG.replace(
                prim_path="/Visuals/Command/real_object",
            )
            visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
            real_object_visualizer = VisualizationMarkers(visualization_markers_cfg)

            visualization_markers_cfg = FRAME_MARKER_CFG.replace(
                prim_path="/Visuals/Command/motion_object",
            )
            visualization_markers_cfg.markers["frame"].scale = (0.2, 0.2, 0.2)
            motion_object_visualizer = VisualizationMarkers(visualization_markers_cfg)

            self.visualization_markers["real_object"] = real_object_visualizer
            self.visualization_markers["motion_object"] = motion_object_visualizer

    def _ensure_index_tensor(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        if env_ids is None:
            return torch.arange(self.num_envs, device=self.device, dtype=torch.long)
        if isinstance(env_ids, torch.Tensor):
            return env_ids.to(device=self.device, dtype=torch.long)
        return torch.as_tensor(env_ids, device=self.device, dtype=torch.long)

    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)
