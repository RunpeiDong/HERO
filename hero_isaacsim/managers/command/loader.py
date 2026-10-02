"""Load G1 motion clips and HERO palm-reference channels.

Stored quaternions are wxyz; runtime quaternions are xyzw. Clip metadata
and parent identifiers are retained for sampling and provenance."""

from __future__ import annotations

import dataclasses
import os
from pathlib import Path
from typing import Any, List, Mapping, Sequence

import numpy as np
import torch
from loguru import logger

from holosoma.managers.command.terms.wbt import (
    FAKE_BODY_NAME_ALIASES,
    MotionLoader,
    MotionTimebaseError,
    MultiMotionLoader,
)
from holosoma.utils.file_cache import cached_open

from hero_isaacsim.managers.command import shared_cache
from hero_isaacsim.managers.command.stock_object import DEFAULT_BOX_SIZE_M

try:  # Fall back to local defaults when shared constants are unavailable.
    from hero_isaacsim.constants import EE_BODY_NAMES, PALM_BODY_NAMES
except ImportError:  # pragma: no cover - compatibility fallback
    EE_BODY_NAMES = ["left_wrist_yaw_link", "right_wrist_yaw_link"]
    PALM_BODY_NAMES = ["left_hand_palm_link", "right_hand_palm_link"]


#: Fixed palm-link aliases accepted when loading compatible robot trajectories.
PADDLE_PALM_BODY_NAMES: tuple[str, str] = ("left_rubber_hand", "right_rubber_hand")


HERO_BODY_NAME_ALIASES: dict[str, str] = {
    **FAKE_BODY_NAME_ALIASES,
    PALM_BODY_NAMES[0]: EE_BODY_NAMES[0],
    PALM_BODY_NAMES[1]: EE_BODY_NAMES[1],
    PADDLE_PALM_BODY_NAMES[0]: EE_BODY_NAMES[0],
    PADDLE_PALM_BODY_NAMES[1]: EE_BODY_NAMES[1],
}

LICENSE_CLASSES: tuple[str, ...] = ("apache", "cc-by", "research-only", "nc", "unknown")
CLIP_END_POLICIES: tuple[str, ...] = ("rollover", "hold")

#: Extension timeline keys (name in npz == attribute name on the loaders).
EXTENSION_TIMELINE_KEYS: tuple[str, ...] = (
    "ee_pos_pelvis",
    "ee_quat_pelvis",
    "ee_pos_pelvis_zero_waist",
    "ee_quat_pelvis_zero_waist",
    "h_ref",
)

_IDENTITY_QUAT_XYZW = (0.0, 0.0, 0.0, 1.0)


def _read_scalar_str(data: Any, key: str, default: str) -> str:
    """Read a scalar string key from an open npz (0-d ``U``/``S`` arrays or 1-element arrays)."""
    if key not in data.files:
        return default
    try:
        value = np.asarray(data[key])
    except ValueError:  # object arrays need allow_pickle; the converter must not write them
        logger.warning(f"npz key '{key}' is an object array (needs pickle); using default {default!r}")
        return default
    if value.size != 1:
        logger.warning(f"npz key '{key}' has {value.size} elements, expected a scalar string; using default")
        return default
    item = value.reshape(()).item()
    if isinstance(item, bytes):
        item = item.decode("utf-8")
    return str(item)


def _read_scalar_bool(data: Any, key: str) -> bool | None:
    if key not in data.files:
        return None
    value = np.asarray(data[key])
    if value.size != 1:
        logger.warning(f"npz key '{key}' has {value.size} elements, expected a scalar bool; ignoring")
        return None
    item = value.reshape(()).item()
    if isinstance(item, (bytes, str)):
        return str(item).strip().lower() in ("1", "true", "yes")
    return bool(item)


def _wxyz_to_xyzw(q: torch.Tensor) -> torch.Tensor:
    """Reorder quaternion components from file order (wxyz) to runtime order (xyzw)."""
    return q[..., [1, 2, 3, 0]]


def _normalize_quat(q: torch.Tensor) -> torch.Tensor:
    norm = torch.linalg.norm(q, dim=-1, keepdim=True).clamp_min(1.0e-8)
    return q / norm


class HeroMotionLoader(MotionLoader):
    """Single-clip loader that also reads the HERO extension keys.

    Attributes added on top of :class:`MotionLoader` (all on ``storage_device``):

    - ``_ee_pos_pelvis`` (T,2,3) palm point position in the clip's own pelvis frame [left, right]
    - ``_ee_quat_pelvis`` (T,2,4) **xyzw**
    - ``_ee_pos_pelvis_zero_waist`` / ``_ee_quat_pelvis_zero_waist`` (same, waist joints zeroed)
    - ``_h_ref`` (T,) pelvis height above the floor
    - ``source_tag``, ``parent_id``, ``license_class`` (str), ``has_object`` (bool, from the base
      loader: presence of ``object_pos_w``), ``box_size`` (Tensor(3,) or None)
    - ``extension_missing`` (list[str]) keys that were absent and filled (``strict=False`` only)"""

    def __init__(
        self,
        motion_file: str,
        robot_body_names: list[str],
        robot_joint_names: list[str],
        device: str = "cpu",
        expected_fps: float | None = None,
        storage_device: str | None = None,
        *,
        strict: bool = True,
        body_name_aliases: Mapping[str, str] | None = None,
    ):
        self.strict = bool(strict)
        self._body_name_aliases: dict[str, str] = dict(
            HERO_BODY_NAME_ALIASES if body_name_aliases is None else body_name_aliases
        )
        self.motion_file = motion_file
        self.extension_missing: list[str] = []
        # Defaults so partially-constructed loaders are still introspectable.
        self.source_tag = "unknown"
        self.parent_id = Path(str(motion_file)).stem
        self.license_class = "unknown"
        self.box_size: torch.Tensor | None = None
        #: The clip's ``has_object`` flag as stored (None when absent).  ``has_object`` (base loader) = an object TRACK is
        #: present; a parked placeholder track may use ``has_object=False`` for object-free clips,
        #: so the multi loader's ``clip_has_object`` = track present AND flag is not False.
        self.clip_object_flag: bool | None = None
        super().__init__(
            motion_file,
            robot_body_names,
            robot_joint_names,
            device=device,
            expected_fps=expected_fps,
            storage_device=storage_device,
        )

    # ------------------------------------------------------------------ body aliasing
    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        """Index ``a_names`` in ``b_names``; robot names missing from the clip fall back to their alias.

        The alias is applied ONLY when the name itself is absent, so a clip that carries real
        palm bodies keeps them and a 32-body clip loaded by a 34-body preset maps palm ->
        wrist_yaw  and foot_contact_point -> ankle_roll (holosoma behaviour)."""
        indexes = []
        b_lookup = {name: i for i, name in enumerate(b_names)}
        for name in a_names:
            if name in b_lookup:
                indexes.append(b_lookup[name])
                continue
            alias = self._body_name_aliases.get(name)
            assert alias is not None and alias in b_lookup, (
                f"The specified name ({name}) doesn't exist in the clip and has no usable alias "
                f"(alias={alias!r}): {b_names}"
            )
            indexes.append(b_lookup[alias])
        return torch.tensor(indexes, dtype=torch.long, device=device)

    # ------------------------------------------------------------------ loading
    def _load_data_from_motion_npz(
        self, motion_file: str, device: str, *, expected_fps: float | None = None
    ) -> tuple[list[str], list[str]]:
        body_names, joint_names = super()._load_data_from_motion_npz(motion_file, device, expected_fps=expected_fps)
        # The base method owns the file handle; re-open (cached) to read the extension keys.
        with cached_open(motion_file, "rb") as f, np.load(f, allow_pickle=False) as data:
            self._load_hero_extension(data, motion_file, device)
        return body_names, joint_names

    def _load_hero_extension(self, data: Any, motion_file: str, device: str) -> None:
        keys = set(data.files)
        num_frames = int(self._joint_pos.shape[0])
        missing: list[str] = []

        def read_pair(pos_key: str, quat_key: str) -> tuple[torch.Tensor | None, torch.Tensor | None]:
            if pos_key in keys and quat_key in keys:
                pos_np = np.asarray(data[pos_key], dtype=np.float32)
                quat_np = np.asarray(data[quat_key], dtype=np.float32)
            elif all(f"{k}_{side}" in keys for k in (pos_key, quat_key) for side in ("l", "r")):

                pos_np = np.stack([data[f"{pos_key}_l"], data[f"{pos_key}_r"]], axis=1).astype(np.float32)
                quat_np = np.stack([data[f"{quat_key}_l"], data[f"{quat_key}_r"]], axis=1).astype(np.float32)
            else:
                return None, None
            if pos_np.shape != (num_frames, 2, 3):
                raise ValueError(f"'{pos_key}' has shape {pos_np.shape}, expected {(num_frames, 2, 3)} in {motion_file}")
            if quat_np.shape != (num_frames, 2, 4):
                raise ValueError(f"'{quat_key}' has shape {quat_np.shape}, expected {(num_frames, 2, 4)} in {motion_file}")
            pos_t = torch.tensor(pos_np, dtype=torch.float32, device=device)
            quat_t = _normalize_quat(_wxyz_to_xyzw(torch.tensor(quat_np, dtype=torch.float32, device=device)))
            return pos_t, quat_t

        ee_pos, ee_quat = read_pair("ee_pos_pelvis", "ee_quat_pelvis")
        if ee_pos is None:
            missing.append("ee_pos_pelvis/ee_quat_pelvis")
        ee_pos_zw, ee_quat_zw = read_pair("ee_pos_pelvis_zero_waist", "ee_quat_pelvis_zero_waist")
        if ee_pos_zw is None:
            missing.append("ee_pos_pelvis_zero_waist/ee_quat_pelvis_zero_waist")

        h_ref: torch.Tensor | None = None
        if "h_ref" in keys:
            h_np = np.asarray(data["h_ref"], dtype=np.float32).reshape(-1)
            if h_np.shape[0] != num_frames:
                raise ValueError(f"'h_ref' has {h_np.shape[0]} frames, expected {num_frames} in {motion_file}")
            h_ref = torch.tensor(h_np, dtype=torch.float32, device=device)
        else:
            missing.append("h_ref")

        if missing:
            if self.strict:
                raise ValueError(f"HERO extension keys missing in '{motion_file}': {missing}")
            logger.warning(
                f"HeroMotionLoader(strict=False): '{motion_file}' lacks {missing}; filling zeros / identity "
                "quaternions (EE and height references for this clip are NOT meaningful)."
            )
        self.extension_missing = missing

        if ee_pos is None:
            ee_pos = torch.zeros(num_frames, 2, 3, dtype=torch.float32, device=device)
            ee_quat = torch.tensor(_IDENTITY_QUAT_XYZW, dtype=torch.float32, device=device).expand(num_frames, 2, 4).clone()
        if ee_pos_zw is None:
            # Without a dedicated zero-waist FK the full-waist pose is the best available stand-in.
            ee_pos_zw, ee_quat_zw = ee_pos.clone(), ee_quat.clone()
        if h_ref is None:
            # joint_pos still carries the root xyz in its first three columns in the file.
            h_ref = torch.tensor(np.asarray(data["joint_pos"])[:, 2], dtype=torch.float32, device=device)

        self._ee_pos_pelvis = ee_pos.contiguous()
        self._ee_quat_pelvis = ee_quat.contiguous()
        self._ee_pos_pelvis_zero_waist = ee_pos_zw.contiguous()
        self._ee_quat_pelvis_zero_waist = ee_quat_zw.contiguous()
        self._h_ref = h_ref.contiguous()

        # ---- per-clip metadata
        self.source_tag = _read_scalar_str(data, "source_tag", "unknown") or "unknown"
        self.parent_id = _read_scalar_str(data, "parent_id", Path(str(motion_file)).stem)
        license_class = _read_scalar_str(data, "license_class", "unknown")
        if license_class not in LICENSE_CLASSES:
            logger.warning(f"'{motion_file}': unknown license_class {license_class!r}; recording 'unknown'")
            license_class = "unknown"
        self.license_class = license_class

        flag = _read_scalar_bool(data, "has_object")
        self.clip_object_flag = flag
        if flag is not None and flag != bool(self.has_object):
            if flag is False and self.has_object:
                # A placeholder track stays on the timeline, while the clip remains object-free.
                logger.debug(f"'{motion_file}': object track present with has_object=False -> object-free clip (dummy track)")
            else:
                logger.warning(
                    f"'{motion_file}': has_object flag={flag} but object track present={self.has_object}; "
                    "the object track presence wins."
                )
        if "box_size" in keys:
            box = np.asarray(data["box_size"], dtype=np.float32).reshape(-1)
            if box.shape[0] == 3:
                self.box_size = torch.tensor(box, dtype=torch.float32, device=device)
            else:
                logger.warning(f"'{motion_file}': box_size has shape {box.shape}, expected (3,); ignoring")

    # ------------------------------------------------------------------ properties
    @property
    def ee_pos_pelvis(self) -> torch.Tensor:
        return self._ee_pos_pelvis

    @property
    def ee_quat_pelvis(self) -> torch.Tensor:
        """(T,2,4) xyzw."""
        return self._ee_quat_pelvis

    @property
    def ee_pos_pelvis_zero_waist(self) -> torch.Tensor:
        return self._ee_pos_pelvis_zero_waist

    @property
    def ee_quat_pelvis_zero_waist(self) -> torch.Tensor:
        """(T,2,4) xyzw."""
        return self._ee_quat_pelvis_zero_waist

    @property
    def h_ref(self) -> torch.Tensor:
        return self._h_ref

    def extend_with_segments(self, segments: dict[str, torch.Tensor], prepend: bool) -> "HeroMotionLoader":
        """Extend base tensors (parent) and hold the boundary frame of the extension keys."""
        added = int(segments["joint_pos"].shape[0])
        _hold_extension_boundary(self, added, prepend)
        super().extend_with_segments(segments, prepend)
        return self


def _hold_extension_boundary(loader: Any, added: int, prepend: bool) -> None:
    """Pad every extension timeline by repeating the first (prepend) or last (append) frame."""
    if added <= 0:
        return
    for key in EXTENSION_TIMELINE_KEYS:
        attr = f"_{key}"
        existing = getattr(loader, attr)
        edge = existing[:1] if prepend else existing[-1:]
        pad = edge.expand(added, *existing.shape[1:]).clone()
        setattr(loader, attr, torch.cat((pad, existing) if prepend else (existing, pad), dim=0))


def _resolve_clip_end_policy(
    source_tag: str, policy_by_source: Mapping[str, str] | None, default_policy: str
) -> bool:
    """Return True for rollover, False for hold ."""
    policy = (policy_by_source or {}).get(source_tag, default_policy)
    if policy not in CLIP_END_POLICIES:
        raise ValueError(
            f"clip end policy for source '{source_tag}' is {policy!r}; expected one of {CLIP_END_POLICIES}"
        )
    return policy == "rollover"


#: holosoma base timelines (attribute ``_<key>``; canonicalized to robot order per clip) and the object track.
BASE_TIMELINE_KEYS: tuple[str, ...] = ("joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w")
OBJECT_TIMELINE_KEYS: tuple[str, ...] = ("object_pos_w", "object_quat_w", "object_lin_vel_w")


@dataclasses.dataclass
class ClipFacts:
    """What the multi loader keeps of a :class:`HeroMotionLoader` besides its timelines (one record per loaded clip).

    Both construction paths reduce a clip to this record -- the in-memory path from the live loader, the shared-cache path
    from ``meta.json`` (JSON round trip of exactly these fields) -- and ``_finalize_clip_metadata`` derives every
    ``clip_*`` tensor from it, so the two paths cannot drift apart.  ``box_size`` holds the float32 values as Python
    floats (exact), ``object_z0`` = ``object_pos_w[0, 2]`` as the in-memory path reads it (``None`` without a track)."""

    file: str
    num_frames: int
    fps: float
    source_tag: str
    parent_id: str
    license_class: str
    object_track: bool
    clip_object_flag: bool | None
    box_size: list[float] | None
    object_z0: float | None
    extension_missing: list[str]

    @classmethod
    def from_loader(cls, motion_file: str, ld: HeroMotionLoader) -> "ClipFacts":
        return cls(
            file=str(motion_file),
            num_frames=int(ld.time_step_total),
            fps=float(ld.fps),
            source_tag=str(ld.source_tag),
            parent_id=str(ld.parent_id),
            license_class=str(ld.license_class),
            object_track=bool(ld.has_object),
            clip_object_flag=ld.clip_object_flag,
            box_size=None if ld.box_size is None else [float(x) for x in ld.box_size.tolist()],
            object_z0=float(ld._object_pos_w[0, 2].item()) if ld.has_object and ld._object_pos_w.shape[0] > 0 else None,
            extension_missing=list(ld.extension_missing),
        )


class HeroMultiMotionLoader(MultiMotionLoader):
    """Directory loader for HERO corpora (mixed object / no-object, per-clip metadata).

    ``motion_dir`` may be a comma-separated list of directories and/or ``.npz`` files.

    Public per-clip attributes (small tensors live on the compute ``device``):

    - ``clip_files`` list[str], ``clip_lengths`` Long[num_clips]
    - ``clip_source_tag`` list[str], ``source_tags`` sorted vocabulary, ``clip_source_tag_id`` Long[num_clips]
    - ``clip_parent_id`` list[str], ``clip_license_class`` list[str]
    - ``clip_has_object`` Bool[num_clips] = object track present AND the clip's ``has_object`` flag is not False (a
      parked placeholder track with ``has_object=False`` is object-free); ``clip_object_track``
      Bool[num_clips] = track present; ``has_object`` = any clip has an object track (timeline tensors exist)
    - ``clip_rollover`` Bool[num_clips] (True = rollover / soft reset, False = hold last frame)
    - ``clip_box_size`` Float[num_clips,3] (NaN where absent), ``clip_has_box_size`` Bool[num_clips]
    - ``clip_box_size_effective`` Float[num_clips,3] (``DEFAULT_BOX_SIZE_M`` cube where absent), ``clip_object_bottom_z0`` Float[num_clips]
      (``object_pos_w[first frame, 2] - box_h / 2``, env-relative; NaN without an object track)

    Timeline attributes on ``storage_device``: the base ones plus ``ee_pos_pelvis``,
    ``ee_quat_pelvis`` (xyzw), ``ee_pos_pelvis_zero_waist``, ``ee_quat_pelvis_zero_waist``, ``h_ref``.

    ``shared_cache_dir`` (default ``None`` = every process concatenates its own private copy):
    a directory under which the concatenated timelines are stored ONCE per host as ``.npy`` files
    (``<shared_cache_dir>/<key>/``, see :mod:`hero_isaacsim.managers.command.shared_cache`) -- the first process to take
    the lock streams them clip by clip, every process then mmaps them copy-on-write, so the DDP ranks of a run (and later
    runs of the same corpus) share one page-cache / tmpfs copy and each rank's private memory is O(index arrays).  With
    ``storage_device="cpu"`` every public tensor and every ``clip_*`` attribute is identical to the in-memory path.
    The guarantee is CPU-only: the writer normalises the EE quaternions on the CPU, whereas the
    in-memory path normalises on the storage device, so a CUDA storage device can differ from its private path at the ULP
    level in ``ee_quat_*`` (the loader warns; a device storage also gets a private device copy -- the cache then only saves
    the npz parse, so it is not a supported production configuration)."""

    def __init__(
        self,
        motion_dir: str,
        robot_body_names: list[str],
        robot_joint_names: list[str],
        device: str = "cpu",
        expected_fps: float | None = None,
        storage_device: str | None = None,
        *,
        strict_extension_keys: bool = True,
        clip_end_policy_by_source: Mapping[str, str] | None = None,
        default_clip_end_policy: str = "rollover",
        body_name_aliases: Mapping[str, str] | None = None,
        max_skip_warnings: int = 3,
        shared_cache_dir: str | os.PathLike | None = None,
        shared_cache_timeout_s: float | None = None,
    ):
        # NOTE: MultiMotionLoader.__init__ is intentionally NOT called: it hard-codes the
        # MotionLoader class and raises on mixed object corpora (wbt.py).  The
        # remaining base logic (fps check, boundaries, concatenation) is mirrored below.
        self.device = device
        self.storage_device = storage_device if storage_device else device
        self.strict_extension_keys = bool(strict_extension_keys)
        self.clip_end_policy_by_source = dict(clip_end_policy_by_source or {})
        self.default_clip_end_policy = default_clip_end_policy
        if default_clip_end_policy not in CLIP_END_POLICIES:
            raise ValueError(f"default_clip_end_policy={default_clip_end_policy!r} not in {CLIP_END_POLICIES}")
        for tag, policy in self.clip_end_policy_by_source.items():
            if policy not in CLIP_END_POLICIES:
                raise ValueError(f"clip_end_policy_by_source[{tag!r}]={policy!r} not in {CLIP_END_POLICIES}")
        self.motion_dir = str(motion_dir)
        self._robot_body_names = list(robot_body_names)
        self._robot_joint_names = list(robot_joint_names)
        self._expected_fps = expected_fps
        self._body_name_aliases: dict[str, str] = dict(
            HERO_BODY_NAME_ALIASES if body_name_aliases is None else body_name_aliases
        )
        #: Set when the timelines are mmapped from a shared cache (``shared_cache_dir``): the entry, this process's role
        #: (``"built"`` = it wrote the cache, ``"hit"`` = it found it READY) and a one-shot flag for the re-materialisation warning.
        self.shared_cache_entry: shared_cache.CacheEntry | None = None
        self.shared_cache_role: str | None = None
        self._shared_cache_warned = False

        motion_files = self._discover_motion_files(motion_dir)
        assert len(motion_files) > 0, f"No .npz files found in {motion_dir}"
        logger.info(f"HeroMultiMotionLoader: loading {len(motion_files)} total motion files")

        if shared_cache_dir is None:
            facts = self._load_in_memory(motion_files, max_skip_warnings)
        else:
            facts = self._load_shared(motion_files, Path(shared_cache_dir), shared_cache_timeout_s, max_skip_warnings)
        self._finalize_clip_metadata(facts)
        self._log_census()

    # ------------------------------------------------------------------ per-clip loaders
    def _make_clip_loader(self, motion_file: str, *, device: str | None = None, storage_device: str | None = None) -> HeroMotionLoader:
        return HeroMotionLoader(
            motion_file,
            self._robot_body_names,
            self._robot_joint_names,
            device=self.device if device is None else device,
            expected_fps=self._expected_fps,
            storage_device=self.storage_device if storage_device is None else storage_device,
            strict=self.strict_extension_keys,
            body_name_aliases=self._body_name_aliases,
        )

    def _open_clips(
        self, motion_files: Sequence[str], max_skip_warnings: int, *, device: str | None = None, storage_device: str | None = None
    ):
        """Yield ``(motion_file, HeroMotionLoader)`` for the loadable clips; format problems are skipped and recorded in
        ``self.skipped_files`` / ``self.num_skipped`` (a wrong / mixed timebase raises)."""
        skipped = 0
        self.skipped_files = []
        for mf in motion_files:
            try:
                loader = self._make_clip_loader(mf, device=device, storage_device=storage_device)
            except MotionTimebaseError:
                raise
            except (KeyError, AssertionError, ValueError) as e:  # noqa: PERF203
                skipped += 1
                self.skipped_files.append((mf, str(e)))
                if skipped <= max_skip_warnings:
                    logger.warning(f"HeroMultiMotionLoader: skipping {mf}: {e}")
                self.num_skipped = skipped
                continue
            self.num_skipped = skipped
            yield mf, loader
        if skipped > max_skip_warnings:
            logger.warning(f"HeroMultiMotionLoader: skipped {skipped} files total due to format issues")
        self.num_skipped = skipped

    # ------------------------------------------------------------------ path 1: private in-memory concatenation (default)
    def _load_in_memory(self, motion_files: Sequence[str], max_skip_warnings: int) -> list[ClipFacts]:
        """Every clip's tensors on ``storage_device``, concatenated with ``torch.cat`` (one private copy per process)."""
        loaders: list[HeroMotionLoader] = []
        loaded_files: list[str] = []
        for mf, loader in self._open_clips(motion_files, max_skip_warnings):
            loaders.append(loader)
            loaded_files.append(mf)
        assert len(loaders) > 0, f"No compatible motion files found (skipped {self.num_skipped})"

        self._check_common_fps(loaders, loaded_files)

        # Base timelines (already canonicalized to robot order by each HeroMotionLoader).
        self._joint_pos = torch.cat([ld._joint_pos for ld in loaders], dim=0)
        self._joint_vel = torch.cat([ld._joint_vel for ld in loaders], dim=0)
        self._body_pos_w = torch.cat([ld._body_pos_w for ld in loaders], dim=0)
        self._body_quat_w = torch.cat([ld._body_quat_w for ld in loaders], dim=0)
        self._body_lin_vel_w = torch.cat([ld._body_lin_vel_w for ld in loaders], dim=0)
        self._body_ang_vel_w = torch.cat([ld._body_ang_vel_w for ld in loaders], dim=0)
        self._joint_indexes = loaders[0]._joint_indexes
        self._body_indexes = loaders[0]._body_indexes
        self.fps = loaders[0].fps

        # Extension timelines.
        self._ee_pos_pelvis = torch.cat([ld._ee_pos_pelvis for ld in loaders], dim=0)
        self._ee_quat_pelvis = torch.cat([ld._ee_quat_pelvis for ld in loaders], dim=0)
        self._ee_pos_pelvis_zero_waist = torch.cat([ld._ee_pos_pelvis_zero_waist for ld in loaders], dim=0)
        self._ee_quat_pelvis_zero_waist = torch.cat([ld._ee_quat_pelvis_zero_waist for ld in loaders], dim=0)
        self._h_ref = torch.cat([ld._h_ref for ld in loaders], dim=0)


        object_flags = [bool(ld.has_object) for ld in loaders]
        sdev = self.storage_device
        if any(object_flags):
            pos_parts, quat_parts, vel_parts = [], [], []
            for ld in loaders:
                if ld.has_object:
                    pos_parts.append(ld._object_pos_w)
                    quat_parts.append(ld._object_quat_w)
                    vel_parts.append(ld._object_lin_vel_w)
                else:
                    n = ld.time_step_total
                    pos_parts.append(torch.zeros(n, 3, dtype=torch.float32, device=sdev))
                    quat_parts.append(
                        torch.tensor(_IDENTITY_QUAT_XYZW, dtype=torch.float32, device=sdev).expand(n, 4).clone()
                    )
                    vel_parts.append(torch.zeros(n, 3, dtype=torch.float32, device=sdev))
            self._object_pos_w = torch.cat(pos_parts, dim=0)
            self._object_quat_w = torch.cat(quat_parts, dim=0)
            self._object_lin_vel_w = torch.cat(vel_parts, dim=0)
        else:
            self._object_pos_w = torch.zeros(0, 3, device=sdev)
            self._object_quat_w = torch.zeros(0, 4, device=sdev)
            self._object_lin_vel_w = torch.zeros(0, 3, device=sdev)
        return [ClipFacts.from_loader(mf, ld) for mf, ld in zip(loaded_files, loaders)]

    # ------------------------------------------------------------------ path 2: shared cache (one copy per host)
    @staticmethod
    def cache_params(
        *,
        robot_body_names: Sequence[str],
        robot_joint_names: Sequence[str],
        body_name_aliases: Mapping[str, str],
        strict_extension_keys: bool,
        expected_fps: float | None,
    ) -> dict[str, Any]:
        """Loader parameters that change the concatenated arrays (part of the cache key; the clip-end policy is not).

        ``timelines`` pins the set of arrays this code writes and reads (``BASE`` / ``EXTENSION`` / ``OBJECT_TIMELINE_KEYS``):
        adding an EE key changes the key, so an older READY entry is simply another key rather than a ``KeyError`` on every
        rank.  Static so the CLI (``--probe`` / ``--build``) computes the ranks' key without constructing a loader."""
        return {
            "robot_body_names": list(robot_body_names),
            "robot_joint_names": list(robot_joint_names),
            "body_name_aliases": sorted((str(k), str(v)) for k, v in dict(body_name_aliases).items()),
            "strict_extension_keys": bool(strict_extension_keys),
            "expected_fps": None if expected_fps is None else float(expected_fps),
            "timelines": sorted((*BASE_TIMELINE_KEYS, *EXTENSION_TIMELINE_KEYS, *OBJECT_TIMELINE_KEYS)),
        }

    def _cache_params(self) -> dict[str, Any]:
        return self.cache_params(
            robot_body_names=self._robot_body_names,
            robot_joint_names=self._robot_joint_names,
            body_name_aliases=self._body_name_aliases,
            strict_extension_keys=self.strict_extension_keys,
            expected_fps=self._expected_fps,
        )

    def _load_shared(
        self, motion_files: Sequence[str], base_dir: Path, timeout_s: float | None, max_skip_warnings: int
    ) -> list[ClipFacts]:
        """mmap the concatenated timelines from ``<base_dir>/<key>/`` (building them first when this process wins the lock)."""
        key = shared_cache.corpus_cache_key(motion_files, params=self._cache_params())
        # The entry validates that every timeline this code indexes is listed in meta (object ones when has_object): a READY
        # entry missing one is "stale" and rebuilt under the lock instead of raising KeyError below on every rank.
        entry = shared_cache.CacheEntry(
            base_dir,
            key,
            required_arrays=(*BASE_TIMELINE_KEYS, *EXTENSION_TIMELINE_KEYS),
            object_arrays=OBJECT_TIMELINE_KEYS,
        )

        def build(scratch: Path, progress) -> dict[str, Any]:
            return self._build_shared_cache(scratch, motion_files, max_skip_warnings, progress)

        ready_dir, role = shared_cache.acquire(entry, build, timeout_s=timeout_s)
        meta = entry.read_meta()
        arrays = shared_cache.load_arrays(ready_dir, meta)
        sdev = self.storage_device
        on_cpu = torch.device(sdev).type == "cpu"

        def as_tensor(name: str) -> torch.Tensor:
            t = torch.from_numpy(arrays[name])  # zero-copy view of the copy-on-write mapping (see shared_cache docstring)
            return t if on_cpu else t.to(sdev)

        for name in (*BASE_TIMELINE_KEYS, *EXTENSION_TIMELINE_KEYS):
            setattr(self, f"_{name}", as_tensor(name))
        if meta["has_object"]:
            for name in OBJECT_TIMELINE_KEYS:
                setattr(self, f"_{name}", as_tensor(name))
        else:
            self._object_pos_w = torch.zeros(0, 3, device=sdev)
            self._object_quat_w = torch.zeros(0, 4, device=sdev)
            self._object_lin_vel_w = torch.zeros(0, 3, device=sdev)
        # Post-canonicalization index maps are the identity (MotionLoader.__init__).
        self._joint_indexes = torch.arange(len(self._robot_joint_names), dtype=torch.long, device=sdev)
        self._body_indexes = torch.arange(len(self._robot_body_names), dtype=torch.long, device=sdev)
        self.fps = float(meta["fps"])
        self.skipped_files = [(str(f), str(why)) for f, why in meta.get("skipped_files", [])]
        self.num_skipped = len(self.skipped_files)
        self.shared_cache_entry = entry
        self.shared_cache_role = role
        writer = meta.get("writer", {})
        logger.info(
            f"HeroMultiMotionLoader: shared corpus cache {role.upper()} {entry.dir} "
            f"({shared_cache.entry_size_bytes(entry.dir) / 1e9:.2f} GB mmapped copy-on-write on {sdev}; written {meta.get('created_utc')} "
            f"by pid {writer.get('pid')} in {writer.get('build_s')} s)"
        )
        if not on_cpu:
            logger.warning(
                f"HeroMultiMotionLoader: storage_device={sdev!r} with a shared cache: the timelines are a PRIVATE device copy (the "
                f"cache only saved the npz parse) and the numerical identity with the private path is guaranteed for CPU storage "
                f"only (the writer normalised the EE quaternions on the CPU; ee_quat_* may differ at the ULP level). Use "
                f"motion_storage_device='cpu' or shared_cache_dir='off'."
            )
        return [ClipFacts(**c) for c in meta["clips"]]

    def _build_shared_cache(self, scratch: Path, motion_files: Sequence[str], max_skip_warnings: int, progress) -> dict[str, Any]:
        """Writer: stream every clip's tensors into ``scratch/arrays/<name>.npy`` (one clip resident at a time) + per-clip facts.

        Byte-for-byte the same arrays ``_load_in_memory`` gets from ``torch.cat`` (same per-clip loaders, same order, same
        zero / identity fill for object-less clips)."""
        arrays_dir = scratch / shared_cache.ARRAYS_DIR
        writers: dict[str, shared_cache.NpyStreamWriter] = {}
        facts: list[ClipFacts] = []
        first_fps: float | None = None
        first_file: str | None = None
        mixed: list[tuple[str, float]] = []
        any_object = False
        total = len(motion_files)
        identity = np.asarray(_IDENTITY_QUAT_XYZW, dtype=np.float32)
        try:
            for i, (mf, ld) in enumerate(self._open_clips(motion_files, max_skip_warnings, device="cpu", storage_device="cpu")):
                if first_fps is None:
                    first_fps, first_file = ld.fps, mf
                elif not np.isclose(ld.fps, first_fps, rtol=0.0, atol=1.0e-4):
                    mixed.append((mf, ld.fps))
                if not writers:
                    for name in (*BASE_TIMELINE_KEYS, *EXTENSION_TIMELINE_KEYS):
                        writers[name] = shared_cache.NpyStreamWriter(arrays_dir / f"{name}.npy", getattr(ld, f"_{name}").shape[1:])
                    for name, width in zip(OBJECT_TIMELINE_KEYS, (3, 4, 3)):
                        writers[name] = shared_cache.NpyStreamWriter(arrays_dir / f"{name}.npy", (width,))
                for name in (*BASE_TIMELINE_KEYS, *EXTENSION_TIMELINE_KEYS):
                    writers[name].append(getattr(ld, f"_{name}").numpy())
                n = int(ld.time_step_total)
                if ld.has_object:
                    any_object = True
                    for name in OBJECT_TIMELINE_KEYS:
                        writers[name].append(getattr(ld, f"_{name}").numpy())
                else:
                    writers["object_pos_w"].append(np.zeros((n, 3), dtype=np.float32))
                    writers["object_quat_w"].append(np.broadcast_to(identity, (n, 4)))
                    writers["object_lin_vel_w"].append(np.zeros((n, 3), dtype=np.float32))
                facts.append(ClipFacts.from_loader(mf, ld))
                del ld
                progress(i + 1, total)
            if not facts:
                raise AssertionError(f"No compatible motion files found (skipped {self.num_skipped})")
            if mixed:
                assert first_file is not None and first_fps is not None
                self._raise_mixed_fps(first_file, first_fps, mixed)
        except BaseException:
            for w in writers.values():
                w.abort()
            raise
        arrays = {name: w.close() for name, w in writers.items()}
        if not any_object:
            for name in OBJECT_TIMELINE_KEYS:
                (arrays_dir / f"{name}.npy").unlink()
                arrays.pop(name)
        return {
            "motion_dir": self.motion_dir,
            "num_files": total,
            "clips": [dataclasses.asdict(f) for f in facts],
            "skipped_files": list(self.skipped_files),
            "fps": float(first_fps),  # type: ignore[arg-type]
            "has_object": any_object,
            "time_step_total": int(arrays["joint_pos"]["shape"][0]),
            "arrays": arrays,
            "params": self._cache_params(),
        }

    # ------------------------------------------------------------------ per-clip metadata (both paths)
    def _finalize_clip_metadata(self, facts: Sequence[ClipFacts]) -> None:
        device = self.device
        lengths = [f.num_frames for f in facts]
        cumulative = torch.tensor(lengths, dtype=torch.long, device=device).cumsum(dim=0)
        self._motion_start_idx = torch.cat([torch.tensor([0], dtype=torch.long, device=device), cumulative[:-1]])
        self._motion_end_idx = cumulative
        self._num_motions = len(facts)
        self.time_step_total = self._joint_pos.shape[0]

        # ``has_object`` / the timeline follow the TRACK presence; ``clip_has_object`` carries the DATA meaning -- an explicit
        # ``has_object=False`` marks a placeholder track in an object-free clip.
        object_flags = [f.object_track for f in facts]
        self.clip_object_track = torch.tensor(object_flags, dtype=torch.bool, device=device)
        self.clip_has_object = torch.tensor(
            [t and f.clip_object_flag is not False for t, f in zip(object_flags, facts)], dtype=torch.bool, device=device
        )
        self.has_object = any(object_flags)

        self.clip_files = [f.file for f in facts]
        self.clip_source_tag = [f.source_tag for f in facts]
        self.clip_parent_id = [f.parent_id for f in facts]
        self.clip_license_class = [f.license_class for f in facts]
        self.source_tags = sorted(set(self.clip_source_tag))
        tag_to_id = {tag: i for i, tag in enumerate(self.source_tags)}
        self.clip_source_tag_id = torch.tensor(
            [tag_to_id[t] for t in self.clip_source_tag], dtype=torch.long, device=device
        )
        self.clip_rollover = torch.tensor(
            [
                _resolve_clip_end_policy(t, self.clip_end_policy_by_source, self.default_clip_end_policy)
                for t in self.clip_source_tag
            ],
            dtype=torch.bool,
            device=device,
        )
        box = torch.full((len(facts), 3), float("nan"), dtype=torch.float32, device=device)
        has_box = torch.zeros(len(facts), dtype=torch.bool, device=device)
        for i, f in enumerate(facts):
            if f.box_size is not None:
                box[i] = torch.tensor(f.box_size, dtype=torch.float32, device=device)
                has_box[i] = True
        self.clip_box_size = box
        self.clip_has_box_size = has_box


        box_eff = torch.where(torch.isnan(box), torch.full_like(box, float(DEFAULT_BOX_SIZE_M)), box)
        bottom = torch.full((len(facts),), float("nan"), dtype=torch.float32, device=device)
        for i, f in enumerate(facts):
            if f.object_z0 is not None:
                bottom[i] = f.object_z0 - 0.5 * float(box_eff[i, 2].item())
        self.clip_box_size_effective = box_eff
        self.clip_object_bottom_z0 = bottom
        self.clip_extension_missing = [list(f.extension_missing) for f in facts]

    # ------------------------------------------------------------------ helpers
    @staticmethod
    def _discover_motion_files(motion_dir: str) -> list[str]:
        files: list[str] = []
        for entry in [d.strip() for d in str(motion_dir).split(",") if d.strip()]:
            expanded = os.path.expanduser(entry)
            path = Path(expanded)
            if path.is_file() and path.suffix == ".npz":
                files.append(str(path))
                continue
            found = sorted(str(p) for p in path.glob("*.npz"))
            logger.info(f"HeroMultiMotionLoader: found {len(found)} .npz files in {expanded}")
            files.extend(found)
        return files

    @staticmethod
    def _check_common_fps(loaders: Sequence[MotionLoader], loaded_files: Sequence[str]) -> None:
        first_fps = loaders[0].fps
        mixed = [
            (path, ld.fps)
            for path, ld in zip(loaded_files, loaders)
            if not np.isclose(ld.fps, first_fps, rtol=0.0, atol=1.0e-4)
        ]
        if mixed:
            HeroMultiMotionLoader._raise_mixed_fps(loaded_files[0], first_fps, mixed)

    @staticmethod
    def _raise_mixed_fps(first_file: str, first_fps: float, mixed: Sequence[tuple[str, float]]) -> None:
        details = ", ".join([f"{first_file}={first_fps:g}Hz"] + [f"{p}={f:g}Hz" for p, f in mixed[:5]])
        if len(mixed) > 5:
            details += f", ... ({len(mixed)} mismatched clips total)"
        raise MotionTimebaseError(
            "HeroMultiMotionLoader cannot mix clip frame rates because references advance one frame per "
            f"control step: {details}. Resample all clips to one common FPS before loading."
        )

    def _note_private_rematerialisation(self, what: str) -> None:
        """Once per loader: ``torch.cat`` over mmapped timelines yields private tensors -- the shared cache stops saving RAM."""
        if self.shared_cache_entry is None or self._shared_cache_warned:
            return
        self._shared_cache_warned = True
        logger.warning(
            f"HeroMultiMotionLoader: {what} re-materialises the timelines in private memory (torch.cat); the shared corpus "
            f"cache {self.shared_cache_entry.dir} no longer saves RAM for this process (disable enable_default_pose_prepend / "
            "_append to keep the corpus shared)"
        )

    def _log_census(self) -> None:
        n_obj = int(self.clip_has_object.sum().item())
        n_dummy = int((self.clip_object_track & ~self.clip_has_object).sum().item())
        per_source = {
            tag: int((self.clip_source_tag_id == i).sum().item()) for i, tag in enumerate(self.source_tags)
        }
        n_roll = int(self.clip_rollover.sum().item())
        n_box = int(self.clip_has_box_size.sum().item())
        logger.info(
            f"HeroMultiMotionLoader: {self._num_motions} motions ({self.num_skipped} skipped), "
            f"{self.time_step_total} total frames, object clips {n_obj}/{self._num_motions} (+{n_dummy} dummy tracks "
            f"flagged has_object=False) -> has_object={self.has_object} (box_size on {n_box}, "
            f"assumed {DEFAULT_BOX_SIZE_M:g} m on {max(n_obj - n_box, 0)}); "
            f"clip end policy rollover={n_roll} hold={self._num_motions - n_roll}; "
            f"clips per source_tag={per_source}"
        )
        missing = sum(1 for m in self.clip_extension_missing if m)
        if missing:
            logger.warning(
                f"HeroMultiMotionLoader: {missing}/{self._num_motions} clips lacked HERO extension keys and were "
                "zero-filled (strict_extension_keys=False)."
            )

    @property
    def clip_lengths(self) -> torch.Tensor:
        return self._motion_end_idx - self._motion_start_idx

    def clips_per_source(self) -> dict[str, int]:
        return {tag: int((self.clip_source_tag_id == i).sum().item()) for i, tag in enumerate(self.source_tags)}

    # ------------------------------------------------------------------ extension properties
    @property
    def ee_pos_pelvis(self) -> torch.Tensor:
        return self._ee_pos_pelvis

    @property
    def ee_quat_pelvis(self) -> torch.Tensor:
        """(T_total,2,4) xyzw."""
        return self._ee_quat_pelvis

    @property
    def ee_pos_pelvis_zero_waist(self) -> torch.Tensor:
        return self._ee_pos_pelvis_zero_waist

    @property
    def ee_quat_pelvis_zero_waist(self) -> torch.Tensor:
        """(T_total,2,4) xyzw."""
        return self._ee_quat_pelvis_zero_waist

    @property
    def h_ref(self) -> torch.Tensor:
        return self._h_ref

    # ------------------------------------------------------------------ transitions
    def extend_with_segments(self, segments: dict[str, torch.Tensor], prepend: bool) -> "HeroMultiMotionLoader":
        """Whole-corpus prepend/append (registers a pseudo-clip, as in the parent).

        Extension keys hold the neighbouring boundary frame; the pseudo-clip inherits the
        metadata (source tag, end policy, object flag) of the clip it is attached to."""
        added = int(segments["joint_pos"].shape[0])
        self._note_private_rematerialisation("extend_with_segments")
        _hold_extension_boundary(self, added, prepend)
        super().extend_with_segments(segments, prepend)
        neighbour = 0 if prepend else -1

        def _insert(vec: torch.Tensor, value: torch.Tensor) -> torch.Tensor:
            value = value.reshape(1, *vec.shape[1:])
            return torch.cat((value, vec) if prepend else (vec, value), dim=0)

        self.clip_has_object = _insert(self.clip_has_object, self.clip_has_object[neighbour])
        self.clip_object_track = _insert(self.clip_object_track, self.clip_object_track[neighbour])
        self.clip_rollover = _insert(self.clip_rollover, self.clip_rollover[neighbour])
        self.clip_source_tag_id = _insert(self.clip_source_tag_id, self.clip_source_tag_id[neighbour])
        self.clip_box_size = _insert(self.clip_box_size, self.clip_box_size[neighbour])
        self.clip_has_box_size = _insert(self.clip_has_box_size, self.clip_has_box_size[neighbour])
        self.clip_box_size_effective = _insert(self.clip_box_size_effective, self.clip_box_size_effective[neighbour])
        self.clip_object_bottom_z0 = _insert(self.clip_object_bottom_z0, self.clip_object_bottom_z0[neighbour])
        meta_lists = (self.clip_files, self.clip_source_tag, self.clip_parent_id, self.clip_license_class)
        for lst in meta_lists:
            value = lst[neighbour]
            if prepend:
                lst.insert(0, value)
            else:
                lst.append(value)
        if prepend:
            self.clip_extension_missing.insert(0, [])
        else:
            self.clip_extension_missing.append([])
        return self

    def extend_each_clip_with_segments(
        self, per_clip_segments: list[dict[str, torch.Tensor]], prepend: bool
    ) -> "HeroMultiMotionLoader":
        """Per-clip transitions folded into each clip (clip count unchanged); extension keys hold the edge frame."""
        assert len(per_clip_segments) == self._num_motions
        self._note_private_rematerialisation("extend_each_clip_with_segments")
        starts = self._motion_start_idx.tolist()
        ends = self._motion_end_idx.tolist()
        new_ext: dict[str, list[torch.Tensor]] = {key: [] for key in EXTENSION_TIMELINE_KEYS}
        for c in range(self._num_motions):
            s, e = int(starts[c]), int(ends[c])
            add = int(per_clip_segments[c]["joint_pos"].shape[0])
            for key in EXTENSION_TIMELINE_KEYS:
                clip_slice = getattr(self, f"_{key}")[s:e]
                edge = clip_slice[:1] if prepend else clip_slice[-1:]
                pad = edge.expand(add, *clip_slice.shape[1:]).clone()
                new_ext[key].append(torch.cat((pad, clip_slice) if prepend else (clip_slice, pad), dim=0))
        super().extend_each_clip_with_segments(per_clip_segments, prepend)
        for key in EXTENSION_TIMELINE_KEYS:
            setattr(self, f"_{key}", torch.cat(new_ext[key], dim=0))
        return self
