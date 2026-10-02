"""HERO motion commands and reference sampling.

The command publishes end-effector references, upper-body joint targets,
height, velocity and stance commands for the tracking policy."""

from __future__ import annotations

import copy
import dataclasses
import json
import math
import os
from dataclasses import field
from pathlib import Path
from typing import Any, Mapping, Sequence

import torch
from loguru import logger
from pydantic.dataclasses import dataclass

from holosoma.config_types.command import MotionConfig
from holosoma.managers.command.terms.wbt import HELD_OUT_DIR_NAME, MotionCommand, manifest_reference_root_velocity_frame
from holosoma.utils.rotations import (
    quat_apply,
    quat_error_magnitude,
    quat_from_euler_xyz,
    quat_mul,
    quat_rotate_inverse,
    wrap_to_pi,
    yaw_quat,
)
from holosoma.utils.simulator_config import SimulatorType

from hero_isaacsim.managers.command import shared_cache
from hero_isaacsim.managers.command.loader import (
    HERO_BODY_NAME_ALIASES,
    PADDLE_PALM_BODY_NAMES,
    PALM_BODY_NAMES,
    HeroMultiMotionLoader,
)
from hero_isaacsim.managers.command.sampler import HeroAdaptiveTimestepsSampler, build_clip_prior
from hero_isaacsim.managers.command.stock_object import DEFAULT_BOX_SIZE_M, OBJECT_ACTOR_NAME

try:
    from hero_isaacsim.constants import DOF_NAMES, EE_BODY_NAMES, PALM_OFFSET, UPPER_DOF_IDX, WAIST_DOF_IDX
except ImportError:  # pragma: no cover - compatibility fallback
    EE_BODY_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
    PALM_OFFSET = {"left": (0.0415, 0.003, 0.0), "right": (0.0415, -0.003, 0.0)}
    DOF_NAMES = [
        "left_hip_pitch_joint", "left_hip_roll_joint", "left_hip_yaw_joint",
        "left_knee_joint", "left_ankle_pitch_joint", "left_ankle_roll_joint",
        "right_hip_pitch_joint", "right_hip_roll_joint", "right_hip_yaw_joint",
        "right_knee_joint", "right_ankle_pitch_joint", "right_ankle_roll_joint",
        "waist_yaw_joint", "waist_roll_joint", "waist_pitch_joint",
        "left_shoulder_pitch_joint", "left_shoulder_roll_joint", "left_shoulder_yaw_joint",
        "left_elbow_joint", "left_wrist_roll_joint", "left_wrist_pitch_joint", "left_wrist_yaw_joint",
        "right_shoulder_pitch_joint", "right_shoulder_roll_joint", "right_shoulder_yaw_joint",
        "right_elbow_joint", "right_wrist_roll_joint", "right_wrist_pitch_joint", "right_wrist_yaw_joint",
    ]
    WAIST_DOF_IDX = list(range(12, 15))
    UPPER_DOF_IDX = list(range(15, 29))

#: Joint names of the HERO upper-body reference (waist 3 + arms 14), holosoma order.
HERO_UPPER_REF_JOINT_NAMES: list[str] = [DOF_NAMES[i] for i in list(WAIST_DOF_IDX) + list(UPPER_DOF_IDX)]

STAND_FLAG_MODES: tuple[str, ...] = ("bernoulli_walk_0p6", "from_clip")
_IDENTITY_QUAT_XYZW = (0.0, 0.0, 0.0, 1.0)


@dataclass(frozen=True)
class HeroMotionConfig(MotionConfig):
    """Motion configuration with HERO command and reference fields."""

    source_weights: dict[str, float] = field(default_factory=dict)

    clip_end_policy_by_source: dict[str, str] = field(default_factory=dict)
    """``source_tag -> "rollover" | "hold"``.  Unlisted sources use ``default_clip_end_policy``."""

    default_clip_end_policy: str = "rollover"

    h_offset_range: tuple[float, float] = (-0.25, 0.0)
    """Uniform range of the per-episode height offset (HERO ``command_ranges.base_height``)."""

    h_offset_from_clip: bool = False
    """When enabled, ``h_cmd = clip h_ref + offset * curriculum * s(h_ref)`` (see
    ``h_offset_scale_*``), clamped to ``h_cmd_min``. Otherwise, use
    ``0.75 + offset * curriculum * (1 - stand_flag)`` without scaling or clamping."""

    h_cmd_default: float = 0.75
    """HERO ``rewards.desired_base_height``."""

    h_offset_scale_floor: float = 0.35
    h_offset_scale_ceiling: float = 0.75
    """With ``h_offset_from_clip=True``: the downward offset is multiplied by
    ``s(h_ref) = clamp((h_ref - floor) / (ceiling - floor), 0, 1)`` so a reference at or below ``floor`` (kneel /
    crawl / prone) gets no offset and a standing 0.75 m reference keeps the full HERO range (otherwise the command
    could drop below the terminations' 0.08 m floor).  ``ceiling <= floor`` disables the scaling (s = 1)."""

    h_cmd_min: float = 0.15
    """With ``h_offset_from_clip=True``: ``h_cmd = max(h_cmd, h_cmd_min)`` safety net above the 0.08 m termination floor."""

    h_curriculum_init: float = 0.1
    h_curriculum_up: float = 0.02
    h_curriculum_down: float = 0.01
    h_curriculum_up_threshold: int = 210
    h_curriculum_down_threshold: int = 200
    h_curriculum_range: tuple[float, float] = (0.0, 1.0)
    """Per-environment curriculum for the height-command offset scale."""

    fix_upper_body_prob: float = 0.3
    """Probability that an env freezes its upper-body reference at a random clip frame (HERO 0.3)."""

    stand_speed_thr: float = 0.15
    stand_yaw_rate_thr: float = 0.2
    stand_window_s: float = 0.5
    """Window-mean |v_xy| < thr and |w_z| < thr -> stance (``from_clip`` mode)."""

    stand_flag_mode: str = "bernoulli_walk_0p6"
    """Sample walking from Bernoulli(walk_prob) with ``bernoulli_walk_0p6``, or derive it from the motion with ``from_clip``."""

    walk_prob: float = 0.6
    """HERO ``stand_prob=0.4`` -> P(walking) = 0.6."""

    zero_waist_when_walking: bool = True
    """When enabled, walking environments use zero waist angles and the zero-waist EE reference."""

    vel_cmd_lin_range: tuple[float, float] = (-1.0, 1.0)
    vel_cmd_yaw_rate_range: tuple[float, float] = (-1.0, 1.0)
    vel_cmd_heading_gain: float = 0.5
    vel_cmd_dead_zone: float = 0.1
    """HERO random command sampler: v ~ U[-1,1]^2, yaw rate = clip(0.5 * wrap(heading - yaw)), dead zone 0.1."""

    command_resample_time_s: float = 5.0
    """HERO ``locomotion_command_resampling_time``: random commands / offsets / fix mask resample period."""

    ref_lookahead_frames: int = 1
    """HERO reads the reference one frame ahead of the current motion frame."""

    strict_extension_keys: bool = True
    """Skip clips missing the HERO extension keys (True) or zero-fill them with a warning (False)."""

    object_park_z: float = -5.0


    object_box_side: float | None = None
    """Edge length (m) of the box actor. If set, retain an object's track only when
    ``max(box_size)`` is within ``object_box_side +- object_box_tol``. None disables this size filter."""

    object_box_tol: float = 0.02
    """Half-width (m) of the accepted ``max(box_size)`` band around ``object_box_side``."""

    object_box_size_assume_sources: tuple[str, ...] = ("box_carry_bank19",)
    """Sources whose object clips may assume ``stock_object.DEFAULT_BOX_SIZE_M``
    when ``box_size`` is missing. Other clips with missing sizes are treated as object-free.
    An empty tuple disables size assumptions. Ignored when ``object_box_side`` is None."""

    max_start_bottom_z_m: float | None = None
    """An object clip keeps its box only when the box bottom at the clip's first frame
    (``clip_object_bottom_z0 = object_pos_w[0, 2] - box_h / 2``, above the floor) is <= this.  ``None`` disables the filter.  Without a support surface a box that starts on a table falls at reset."""

    adaptive_sampler_clip_max_relative: float | None = None
    """Bound on any clip's failure-weighted draw probability as a multiple of its prior share (source weight / clips
    of that source); ``None`` = unbounded. Applied after the source prior, so ``source_weights`` stay honoured and a
    zero-weight source stays at zero while no single clip can absorb the sampler."""

    source_weights_from_manifest: bool = False
    """Load source weights and clip-end policies from ``CORPUS_MANIFEST.json`` at setup.

    Look in ``motion_dir`` and its parent. Manifest weights apply when configuration
    weights are empty; manifest clip-end policies override configured policies.
    Resolved values are logged in ``HeroMotionCommand.provenance``. If no manifest
    is found, warn and use the configuration values."""

    terrain_relative_heights: bool = False
    """Metric only (``base/height_err_m``): measure the robot pelvis height above the TERRAIN under the base
    (``termination.hero.terrain_ground_z``) instead of the env origin z; identical on the flat plane."""

    shared_cache_dir: str | None = None

    def __post_init__(self) -> None:
        if self.stand_flag_mode not in STAND_FLAG_MODES:
            raise ValueError(f"stand_flag_mode={self.stand_flag_mode!r} not in {STAND_FLAG_MODES}")
        if not 0.0 <= self.fix_upper_body_prob <= 1.0:
            raise ValueError("fix_upper_body_prob must be in [0, 1]")
        if not 0.0 <= self.walk_prob <= 1.0:
            raise ValueError("walk_prob must be in [0, 1]")
        if self.h_offset_range[0] > self.h_offset_range[1]:
            raise ValueError("h_offset_range must be (low, high)")
        if self.h_cmd_min < 0.0:
            raise ValueError("h_cmd_min must be >= 0")
        if self.ref_lookahead_frames < 0:
            raise ValueError("ref_lookahead_frames must be >= 0")
        for tag, policy in self.clip_end_policy_by_source.items():
            if policy not in ("rollover", "hold"):
                raise ValueError(f"clip_end_policy_by_source[{tag!r}]={policy!r} must be 'rollover' or 'hold'")
        if self.object_box_side is not None and not self.object_box_side > 0.0:
            raise ValueError("object_box_side must be > 0 (m) or None")
        if self.object_box_tol < 0.0:
            raise ValueError("object_box_tol must be >= 0")
        if self.max_start_bottom_z_m is not None and not math.isfinite(self.max_start_bottom_z_m):
            raise ValueError("max_start_bottom_z_m must be finite or None")
        if self.adaptive_sampler_clip_max_relative is not None and not self.adaptive_sampler_clip_max_relative >= 1.0:
            raise ValueError("adaptive_sampler_clip_max_relative must be >= 1 or None")


HERO_MOTION_CONFIG_V4_FIELDS: tuple[str, ...] = (
    "object_box_side",
    "object_box_tol",
    "object_box_size_assume_sources",
    "max_start_bottom_z_m",
    "source_weights_from_manifest",
    "terrain_relative_heights",
    "shared_cache_dir",
)


def hero_motion_config_v4_defaults() -> dict[str, Any]:
    """``{field: default}`` of :data:`HERO_MOTION_CONFIG_V4_FIELDS` as declared on :class:`HeroMotionConfig`."""
    fields = {f.name: f for f in dataclasses.fields(HeroMotionConfig)}
    return {name: fields[name].default for name in HERO_MOTION_CONFIG_V4_FIELDS}


HERO_H1_MOTION_OVERRIDES: dict[str, Any] = {
    "h_offset_from_clip": False,
    "stand_flag_mode": "bernoulli_walk_0p6",
    "zero_waist_when_walking": True,
}
HERO_H2_MOTION_OVERRIDES: dict[str, Any] = {
    "h_offset_from_clip": True,
    "stand_flag_mode": "from_clip",
    "zero_waist_when_walking": False,
}


def coerce_hero_motion_config(motion_config: Any) -> HeroMotionConfig:
    """Build a ``HeroMotionConfig`` from a dict (tyro), a ``MotionConfig`` or a ``HeroMotionConfig``."""
    if isinstance(motion_config, HeroMotionConfig):
        return motion_config
    if isinstance(motion_config, MotionConfig):
        return HeroMotionConfig(**dataclasses.asdict(motion_config))
    if isinstance(motion_config, Mapping):
        return HeroMotionConfig(**dict(motion_config))
    raise TypeError(f"motion_config must be a mapping or MotionConfig, got {type(motion_config).__name__}")


def _with_hero_motion_config(cfg: Any) -> Any:
    """Return ``cfg`` with ``params["motion_config"]`` replaced by a ``HeroMotionConfig`` instance."""
    motion_config = cfg.params["motion_config"]
    hero_cfg = coerce_hero_motion_config(motion_config)
    if hero_cfg is motion_config:
        return cfg
    params = dict(cfg.params)
    params["motion_config"] = hero_cfg
    if dataclasses.is_dataclass(cfg) and not isinstance(cfg, type):
        return dataclasses.replace(cfg, params=params)
    new_cfg = copy.copy(cfg)
    new_cfg.params = params
    return new_cfg


def resolve_object_actor(simulator: Any, name: str = "object"):
    """Actor indices of ``name`` in the simulator, or ``None`` when the scene has no such actor."""
    try:
        return simulator.get_actor_indices(name, env_ids=None)
    except (KeyError, ValueError, AttributeError):
        return None


CORPUS_MANIFEST_NAME = "CORPUS_MANIFEST.json"


def corpus_manifest_candidates(motion_dir: str, first_file: str | None = None) -> list[Path]:
    """Corpus manifest candidates."""
    dirs: list[Path] = []
    for entry in str(motion_dir or "").split(","):
        entry = entry.strip()
        if not entry:
            continue
        path = Path(os.path.expanduser(entry))
        dirs.append(path.parent if path.suffix == ".npz" else path)
    if first_file:
        dirs.append(Path(os.path.realpath(first_file)).parent)
    out: list[Path] = []
    for d in dirs:
        for cand in (d / CORPUS_MANIFEST_NAME, d.parent / CORPUS_MANIFEST_NAME):
            if cand not in out:
                out.append(cand)
    return out


def load_corpus_manifest(motion_dir: str, first_file: str | None = None) -> tuple[dict[str, Any] | None, str | None]:
    """``(manifest, path)`` of the first readable ``CORPUS_MANIFEST.json`` (:func:`corpus_manifest_candidates`), else
    ``(None, None)``.  Unreadable / non-mapping candidates are skipped with a warning."""
    for cand in corpus_manifest_candidates(motion_dir, first_file):
        if not cand.is_file():
            continue
        try:
            with open(cand) as fh:
                data = json.load(fh)
        except (OSError, ValueError) as exc:
            logger.warning(f"load_corpus_manifest: cannot read {cand}: {exc}")
            continue
        if isinstance(data, dict):
            return data, str(cand)
        logger.warning(f"load_corpus_manifest: {cand} is not a mapping; ignored")
    return None, None


def load_corpus_manifests(motion_dir: str) -> tuple[list[tuple[dict[str, Any], str]], list[str]]:
    """``([(manifest, path), ...], entries_without_manifest)`` -- ONE manifest per entry of a comma-separated
    ``motion_dir`` (:func:`load_corpus_manifest` over the entry and the real directory of its first clip), de-duplicated
    by path.  A ``motion_dir`` that names several corpora (e.g. AMASS and generated IK clips) therefore contributes every
    corpus's manifest, not just the first readable one."""
    found: list[tuple[dict[str, Any], str]] = []
    missing: list[str] = []
    for entry in [e.strip() for e in str(motion_dir or "").split(",") if e.strip()]:
        files = HeroMultiMotionLoader._discover_motion_files(entry)
        manifest, path = load_corpus_manifest(entry, files[0] if files else None)
        if manifest is None or path is None:
            missing.append(entry)
            continue
        if path not in [q for _, q in found]:
            found.append((manifest, path))
    return found, missing


def manifest_source_weights(manifest: Mapping[str, Any]) -> dict[str, float]:
    """Validated ``manifest["source_weights"]`` (``{source_tag: episode share}``; finite, >= 0).  Missing / null -> ``{}``."""
    raw = manifest.get("source_weights") or {}
    if not isinstance(raw, Mapping):
        raise ValueError(f"CORPUS_MANIFEST.json source_weights must be a mapping, got {type(raw).__name__}")
    out: dict[str, float] = {}
    for tag, value in raw.items():
        w = float(value)
        if not math.isfinite(w) or w < 0.0:
            raise ValueError(f"CORPUS_MANIFEST.json source_weights[{tag!r}]={value!r} must be finite and >= 0")
        out[str(tag)] = w
    return out


def manifest_clip_end_policies(manifest: Mapping[str, Any]) -> dict[str, str]:
    """Validated ``manifest["clip_end_policy_by_source"]`` (values ``rollover`` | ``hold``).  Missing / null -> ``{}``."""
    raw = manifest.get("clip_end_policy_by_source") or {}
    if not isinstance(raw, Mapping):
        raise ValueError("CORPUS_MANIFEST.json clip_end_policy_by_source must be a mapping")
    out: dict[str, str] = {}
    for tag, policy in raw.items():
        policy = str(policy)
        if policy not in ("rollover", "hold"):
            raise ValueError(f"CORPUS_MANIFEST.json clip_end_policy_by_source[{tag!r}]={policy!r} must be rollover|hold")
        out[str(tag)] = policy
    return out


OBJECT_BUCKET_SIDES_M: tuple[float, ...] = (0.20, 0.30, 0.40, 0.50, 0.60)
#: The bucket served by holosoma's hard-coded ``object`` actor.
PRIMARY_OBJECT_BUCKET_SIDE_M: float = 0.30


def object_bucket_actor_name(side_m: float, primary_side_m: float = PRIMARY_OBJECT_BUCKET_SIDE_M) -> str:
    """Actor name of a box-size bucket: ``"object"`` for the primary bucket (the 0.30 m actor), else
    ``"object_<cm>cm"`` (``0.20 -> "object_20cm"``) -- the ``RigidObjectConfig.name`` / registry / ``set_actor_states``
    name used for that bucket."""
    if abs(float(side_m) - float(primary_side_m)) < 1.0e-6:
        return OBJECT_ACTOR_NAME
    return f"{OBJECT_ACTOR_NAME}_{int(round(float(side_m) * 100.0))}cm"


def nearest_object_bucket(box_size: Any, buckets: Sequence[float] = OBJECT_BUCKET_SIDES_M) -> torch.Tensor:
    """Nearest bucket edge for ``max(box_size)`` -- ``box_size`` ``[3]`` or ``[K, 3]`` (NaN rows -> the
    ``DEFAULT_BOX_SIZE_M`` cube) -> ``[]`` / ``[K]`` bucket sides (ties -> the smaller bucket)."""
    size = torch.as_tensor(box_size, dtype=torch.float32)
    squeeze = size.dim() == 1
    size = size.reshape(-1, 3)
    size = torch.where(torch.isnan(size), torch.full_like(size, float(DEFAULT_BOX_SIZE_M)), size)
    edge = size.max(dim=1).values  # [K]
    b = torch.as_tensor(sorted(float(v) for v in buckets), dtype=torch.float32, device=size.device)
    idx = (edge[:, None] - b[None, :]).abs().argmin(dim=1)  # argmin returns the first (smaller) bucket on ties
    out = b[idx]
    return out[0] if squeeze else out


def resolve_object_actor_for_box_size(
    box_size: Any,
    *,
    buckets: Sequence[float] = OBJECT_BUCKET_SIDES_M,
    primary_side_m: float = PRIMARY_OBJECT_BUCKET_SIDE_M,
    available: Sequence[str] | None = None,
) -> str | None:
    """Actor name serving ONE clip's ``box_size`` (``[3]``; NaN -> 0.30 cube): the nearest bucket's
    :func:`object_bucket_actor_name`, or ``None`` when ``available`` (the scene's actor names) is given and lacks it.
    For example, ``resolve_object_actor_for_box_size((0.3, 0.3, 0.3), available=["object"]) == "object"``."""
    side = float(nearest_object_bucket(box_size, buckets).item())
    name = object_bucket_actor_name(side, primary_side_m)
    if available is not None and name not in set(available):
        return None
    return name


class HeroMotionCommand(MotionCommand):
    """``MotionCommand`` with HERO commands, extended references and per-clip end policy."""

    def __init__(self, cfg: Any, env: Any):
        # Override of the dict->dataclass branch (wbt.py): the parent sees an
        # instance of a MotionConfig subclass and keeps it as is.
        super().__init__(_with_hero_motion_config(cfg), env)
        self.hero_cfg: HeroMotionConfig = self.motion_cfg  # type: ignore[assignment]
        self._in_soft_reset = False

    # ------------------------------------------------------------------ setup
    def setup(self) -> None:
        env = self._env
        self.num_envs = env.num_envs
        self.device = env.device
        cfg = self.hero_cfg

        robot_body_names = list(env.simulator._body_list)  # type: ignore[attr-defined]
        robot_joint_names = list(env.simulator.dof_names)  # type: ignore[attr-defined]

        assert cfg.motion_file or cfg.motion_dir, "Either motion_file or motion_dir must be set in HeroMotionConfig"


        for palm in (*PALM_BODY_NAMES, *PADDLE_PALM_BODY_NAMES):
            assert palm not in cfg.body_names_to_track, (
                f"{palm} must never be in body_names_to_track: the palm reference comes from the extension keys"
            )

        self.control_fps = 1.0 / float(env.dt)
        self._resolve_manifest_overrides(cfg.motion_dir if cfg.motion_dir else cfg.motion_file)
        self.motion: HeroMultiMotionLoader = self._build_motion_loader(robot_body_names, robot_joint_names)
        logger.info(
            f"Motion timebase verified: motion={self.motion.fps:g}Hz, control={self.control_fps:g}Hz "
            "(one reference frame per control step)"
        )
        self._body_indexes_in_motion = self.motion._body_indexes
        self._joint_indexes_in_motion = self.motion._joint_indexes
        self._check_manifest_weights_cover_corpus()

        self._maybe_add_default_pose_transition(prepend=True)
        self._maybe_add_default_pose_transition(prepend=False)

        self.ref_body_index = robot_body_names.index(cfg.body_name_ref[0])
        self.tracked_body_indexes = self._get_index_of_a_in_b(cfg.body_names_to_track, robot_body_names, self.device)

        self._setup_reference_root_velocity_frame()
        self._upper_dof_idx = self._get_index_of_a_in_b(HERO_UPPER_REF_JOINT_NAMES, robot_joint_names, self.device)


        self._ee_body_idx = self._get_index_of_a_in_b(list(EE_BODY_NAMES), robot_body_names, self.device)
        self._palm_offset = torch.tensor(
            [PALM_OFFSET["left"], PALM_OFFSET["right"]], dtype=torch.float32, device=self.device
        )  # (2,3) in the wrist_yaw frame

        self.scene_has_object = False
        if self.motion.has_object:
            self._setup_object_actor()


        self.clip_object_effective = self._compute_clip_object_effective()

        # Source prior is needed even without the adaptive sampler (clip choice at reset).
        self.clip_prior = build_clip_prior(
            self.motion.clip_source_tag_id, self.motion.source_tags, self.effective_source_weights, self.device
        )
        if cfg.use_adaptive_timesteps_sampler:
            self.adaptive_timesteps_sampler = self._build_sampler()
        logger.info(
            "HeroMotionCommand: source prior fractions "
            + str(
                {
                    tag: round(float(self.clip_prior[self.motion.clip_source_tag_id == i].sum().item()), 4)
                    for i, tag in enumerate(self.motion.source_tags)
                }
            )
        )

        self._prepare_clip_kinematic_stats()
        self.metrics: dict[str, torch.Tensor] = {}
        self.init_buffers()
        self.provenance: dict[str, Any] = self.command_provenance()

        if env.viewer and env.simulator.get_simulator_type() == SimulatorType.ISAACSIM:
            self._setup_visualization_markers_for_isaacsim()

    # ------------------------------------------------------------------ manifest-driven weights / policies
    def _resolve_manifest_overrides(self, motion_dir: str) -> None:
        """Decide the source weights / clip-end policies the loader and prior use ."""
        cfg = self.hero_cfg
        self.effective_source_weights: dict[str, float] = dict(cfg.source_weights)
        self.effective_clip_end_policy_by_source: dict[str, str] = dict(cfg.clip_end_policy_by_source)
        self.manifest_path: str | None = None
        self.manifest_paths: list[str] = []
        self.manifest_entries_without_manifest: list[str] = []
        self.manifest_source_weights: dict[str, float] = {}
        self.manifest_clip_end_policy_by_source: dict[str, str] = {}
        self.source_weights_source: str = "config" if cfg.source_weights else "uniform"
        if not cfg.source_weights_from_manifest:
            return
        found, missing = load_corpus_manifests(motion_dir)
        self.manifest_entries_without_manifest = list(missing)
        if not found:
            logger.error(
                "HeroMotionCommand: source_weights_from_manifest=True but no {} next to / above any entry of {!r} -> "
                "config source_weights ({}) and clip_end_policy_by_source apply",
                CORPUS_MANIFEST_NAME,
                motion_dir,
                "non-empty" if cfg.source_weights else "EMPTY = uniform over clips",
            )
            return
        if missing:
            logger.warning(
                "HeroMotionCommand: {} motion_dir entries have no {} ({}); their source tags must be covered by the "
                "manifests of the other entries or setup raises",
                len(missing),
                CORPUS_MANIFEST_NAME,
                missing,
            )
        weights: dict[str, float] = {}
        policies: dict[str, str] = {}
        weight_src: dict[str, str] = {}
        policy_src: dict[str, str] = {}
        for manifest, path in found:
            for tag, w in manifest_source_weights(manifest).items():
                if tag in weights and abs(weights[tag] - w) > 1.0e-9:
                    raise ValueError(
                        f"CORPUS_MANIFEST.json conflict: source_weights[{tag!r}] = {weights[tag]!r} in {weight_src[tag]} "
                        f"vs {w!r} in {path}"
                    )
                weights[tag] = w
                weight_src[tag] = path
            for tag, policy in manifest_clip_end_policies(manifest).items():
                if tag in policies and policies[tag] != policy:
                    raise ValueError(
                        f"CORPUS_MANIFEST.json conflict: clip_end_policy_by_source[{tag!r}] = {policies[tag]!r} in "
                        f"{policy_src[tag]} vs {policy!r} in {path}"
                    )
                policies[tag] = policy
                policy_src[tag] = path
        paths = [path for _, path in found]
        self.manifest_paths = paths
        self.manifest_path = paths[0]
        self.manifest_source_weights = weights
        self.manifest_clip_end_policy_by_source = policies
        joined = ",".join(paths)
        if cfg.source_weights:
            logger.info(
                "HeroMotionCommand: config source_weights are non-empty -> the manifest's source_weights ({} tags in {}) "
                "are ignored",
                len(weights),
                joined,
            )
        elif weights:
            self.effective_source_weights = dict(weights)
            self.source_weights_source = f"manifest:{joined}"
            logger.info(
                "HeroMotionCommand: source_weights from {}: {}",
                joined,
                {k: round(v, 4) for k, v in sorted(self.effective_source_weights.items())},
            )
        else:
            logger.error(
                "HeroMotionCommand: {} has no source_weights -> config source_weights are EMPTY = uniform over clips",
                joined,
            )
        if policies:
            changed = {t: p for t, p in policies.items() if self.effective_clip_end_policy_by_source.get(t) != p}
            self.effective_clip_end_policy_by_source.update(policies)
            logger.info(
                "HeroMotionCommand: clip_end_policy_by_source overlaid from {} ({} tags, {} differ from the config: {})",
                joined,
                len(policies),
                len(changed),
                changed,
            )

    def _check_manifest_weights_cover_corpus(self) -> None:
        """Manifest-sourced weights must list EVERY source tag of the loaded corpus .

        ``build_clip_prior`` gives an unlisted tag ``unlisted_source_weight=0`` with a warning only -- with a
        comma-separated ``motion_dir`` whose second corpus has no manifest that bank would contribute ZERO episodes for
        the whole run while ``motion/source_frac_<tag>`` silently reads 0.  Config-sourced weights keep the warning
        (an explicit choice to exclude a source).  Raises ``ValueError``."""
        if not self.source_weights_source.startswith("manifest"):
            return
        unlisted = sorted(set(self.motion.source_tags) - set(self.effective_source_weights))
        if unlisted:
            raise ValueError(
                "HeroMotionCommand: source_weights_from_manifest=True but the merged manifest source_weights "
                f"({self.source_weights_source}) do not list corpus source tags {unlisted} (clips per tag: "
                f"{ {t: self.motion.clips_per_source().get(t) for t in unlisted} }); entries without a manifest: "
                f"{self.manifest_entries_without_manifest}.  Add the tags to the manifest(s) or set source_weights."
            )


    def _compute_clip_object_effective(self) -> torch.Tensor:
        """Bool[num_clips]: the clip keeps its box = ``clip_has_object`` AND floor rule AND size rule AND start-height rule."""
        cfg, m = self.hero_cfg, self.motion
        dev = self.device
        has = m.clip_has_object.to(dev).bool()
        eff = has.clone()
        n = int(has.numel())
        tol = float(cfg.object_box_tol)
        assume_sources = tuple(str(t) for t in cfg.object_box_size_assume_sources)
        summary: dict[str, Any] = {
            "num_clips": n,
            "num_has_object": int(has.sum().item()),
            "object_box_side": cfg.object_box_side,
            "object_box_tol": tol,
            "object_box_size_assume_sources": list(assume_sources),
            "max_start_bottom_z_m": cfg.max_start_bottom_z_m,
            "num_below_floor_rejected": 0,
            "num_size_rejected": 0,
            "num_box_size_missing": 0,
            "num_box_size_assumed": 0,
            "num_height_rejected": 0,
        }
        z0 = getattr(m, "clip_object_bottom_z0", None)
        if z0 is not None:
            z0 = z0.to(dev)
            floor_ok = torch.isfinite(z0) & (z0 >= -tol)
            summary["num_below_floor_rejected"] = int((has & ~floor_ok).sum().item())
            eff = eff & floor_ok
        if cfg.object_box_side is not None:
            has_bs = getattr(m, "clip_has_box_size", None)
            box = getattr(m, "clip_box_size_effective", None)
            if box is None:
                box = m.clip_box_size
                box = torch.where(torch.isnan(box), torch.full_like(box, float(DEFAULT_BOX_SIZE_M)), box)
            has_bs = torch.ones(n, dtype=torch.bool, device=dev) if has_bs is None else has_bs.to(dev).bool()
            tags = list(getattr(m, "clip_source_tag", []))
            if len(tags) == n:
                assume = torch.tensor([t in assume_sources for t in tags], dtype=torch.bool, device=dev)
            else:  # a loader without per-clip tags: nothing can be assumed
                assume = torch.zeros(n, dtype=torch.bool, device=dev)
            edge = box.to(dev).max(dim=1).values
            band_ok = (edge - float(cfg.object_box_side)).abs() <= tol + 1.0e-6
            size_ok = band_ok & (has_bs | assume)
            summary["num_box_size_missing"] = int((has & ~has_bs).sum().item())
            summary["num_box_size_assumed"] = int((has & ~has_bs & assume).sum().item())
            summary["num_size_rejected"] = int((has & ~size_ok).sum().item())
            eff = eff & size_ok
        if cfg.max_start_bottom_z_m is not None and z0 is not None:
            height_ok = torch.isfinite(z0) & (z0 <= float(cfg.max_start_bottom_z_m))
            summary["num_height_rejected"] = int((has & ~height_ok).sum().item())
            eff = eff & height_ok
        summary["num_effective"] = int(eff.sum().item())
        self.object_rule_summary = summary
        if summary["num_has_object"]:
            logger.info("HeroMotionCommand: object rule {}", summary)
        return eff

    def command_provenance(self) -> dict[str, Any]:
        """What this command actually trained with: weights source, manifest path(s), effective weights / policies
        (corpus tags only), prior fractions, object rule counts, scene object flag.  Kept on ``self.provenance`` and
        published by ``HeroTrackingManager._sync_provenance_once`` as ``provenance["command"]`` (wandb config +
        provenance.json), so a uniform-fallback run can be told from a manifest-weighted one after the fact."""
        cfg = self.hero_cfg
        tags = list(self.motion.source_tags)
        prior_frac = {
            tag: round(float(self.clip_prior[self.motion.clip_source_tag_id == i].sum().item()), 6)
            for i, tag in enumerate(tags)
        }
        default_policy = cfg.default_clip_end_policy
        return {
            "source_weights_from_manifest": bool(cfg.source_weights_from_manifest),
            "source_weights_source": self.source_weights_source,
            "corpus_manifest": self.manifest_path,
            "corpus_manifests": list(self.manifest_paths),
            "motion_dir_entries_without_manifest": list(self.manifest_entries_without_manifest),
            "source_weights": dict(self.effective_source_weights),
            "source_prior_fractions": prior_frac,
            "clip_end_policy_by_source": {
                t: self.effective_clip_end_policy_by_source.get(t, default_policy) for t in tags
            },
            "object_rule": dict(getattr(self, "object_rule_summary", {})),
            "scene_has_object": bool(self.scene_has_object),
            "motion_storage_device": getattr(cfg, "motion_storage_device", "") or None,
            "shared_cache_dir": getattr(self, "shared_cache_dir", None),
            "shared_cache_source": getattr(self, "shared_cache_source", None),
            "shared_cache_key": getattr(self, "shared_cache_key", None),
            "shared_cache_role": getattr(self, "shared_cache_role", None),
        }

    def _build_motion_loader(self, robot_body_names: list[str], robot_joint_names: list[str]) -> HeroMultiMotionLoader:
        cfg = self.hero_cfg
        storage_device = getattr(cfg, "motion_storage_device", "") or None
        # A single motion_file is loaded through the multi loader too (num_motions == 1) so
        # that per-clip metadata / end policy / prior have one code path.
        motion_dir = cfg.motion_dir if cfg.motion_dir else cfg.motion_file


        spec = getattr(cfg, "shared_cache_dir", None)
        cache_dir, cache_source = shared_cache.resolve_cache_dir(
            spec, motion_files=lambda: HeroMultiMotionLoader._discover_motion_files(motion_dir)
        )
        self.shared_cache_dir: str | None = None if cache_dir is None else str(cache_dir)
        self.shared_cache_source = cache_source
        if cache_dir is not None:
            logger.info(f"HeroMotionCommand: shared corpus cache under {cache_dir} (source {cache_source}; storage device {storage_device or self.device})")
        elif cache_source == "env:off" or cache_source.startswith("config:off"):
            logger.info(f"HeroMotionCommand: shared corpus cache OFF ({cache_source}); every rank keeps a private copy of the corpus")
        motion = HeroMultiMotionLoader(
            motion_dir,
            robot_body_names,
            robot_joint_names,
            device=self.device,
            expected_fps=self.control_fps,
            storage_device=storage_device,
            strict_extension_keys=cfg.strict_extension_keys,
            clip_end_policy_by_source=self.effective_clip_end_policy_by_source,
            default_clip_end_policy=cfg.default_clip_end_policy,
            body_name_aliases=HERO_BODY_NAME_ALIASES,
            shared_cache_dir=cache_dir,
        )
        entry = motion.shared_cache_entry
        self.shared_cache_key = entry.key if entry is not None else None
        self.shared_cache_role = motion.shared_cache_role
        return motion

    def _build_sampler(self) -> HeroAdaptiveTimestepsSampler:
        cfg = self.hero_cfg
        clip_lengths = self.motion.motion_end_idx - self.motion.motion_start_idx
        max_clip_time_step = int(clip_lengths.max().item()) if clip_lengths.numel() > 0 else self.motion.time_step_total
        return HeroAdaptiveTimestepsSampler(
            self.motion.time_step_total,
            self.device,
            int(round(self.control_fps)),
            phase_binning=getattr(cfg, "adaptive_sampler_phase_binning", False),
            max_clip_time_step=max_clip_time_step,
            per_clip=getattr(cfg, "adaptive_sampler_per_clip", False),
            num_clips=self.motion.num_motions,
            adaptive_uniform_ratio=getattr(cfg, "adaptive_sampler_uniform_ratio", 0.1),
            adaptive_clip_temperature=getattr(cfg, "adaptive_sampler_clip_temperature", 1.0),
            adaptive_clip_max_probability=getattr(cfg, "adaptive_sampler_clip_max_probability", 1.0),
            clip_cap_relative=getattr(cfg, "adaptive_sampler_clip_max_relative", None),
            clip_prior=self.clip_prior,
            source_tag_ids=self.motion.clip_source_tag_id,
            source_tags=self.motion.source_tags,
        )

    def _setup_object_actor(self) -> None:
        """Resolve the single object actor (IsaacSim only)."""
        self.object_name = "object"  # hardcoded object name (holosoma contract)
        idx = resolve_object_actor(self._env.simulator, self.object_name)
        if idx is None:
            n_obj = int(self.motion.clip_has_object.sum().item()) if hasattr(self.motion, "clip_has_object") else -1
            logger.warning(
                "HeroMotionCommand: corpus has object tracks ({} clips) but the scene has no '{}' actor "
                "(robot preset without object) -> object references ignored, env_has_object forced False",
                n_obj,
                self.object_name,
            )
            self.scene_has_object = False
            return
        if getattr(self._env.simulator, "object_variant_ids", None) is not None:


            raise RuntimeError(
                "HeroMotionCommand: the scene has per-env object variants (robot.object.object_urdf_paths = "
                f"{list(getattr(self._env.simulator, 'object_variant_names', None) or [])}); "
                "HeroMotionCommand supports a single object actor."
            )
        self.object_indices_in_simulator = idx
        self.scene_has_object = True
        assert self._env.simulator.get_simulator_type() == SimulatorType.ISAACSIM, "Object is only supported in IsaacSim"

    def _prepare_clip_kinematic_stats(self) -> None:
        """Prepare clip kinematic stats."""
        lin = self.motion.body_lin_vel_w[:, 0, :2]
        speed = torch.linalg.norm(lin, dim=-1)
        yaw_rate = self.motion.body_ang_vel_w[:, 0, 2].abs()
        zero = torch.zeros(1, dtype=speed.dtype, device=speed.device)
        self._speed_csum = torch.cat([zero, speed.cumsum(0)])
        self._yaw_rate_csum = torch.cat([zero, yaw_rate.cumsum(0)])
        self._stand_half_window = max(int(round(0.5 * self.hero_cfg.stand_window_s * self.motion.fps)), 0)

    # ------------------------------------------------------------------ buffers
    def init_buffers(self, *, reset_adaptive_sampler: bool = True):
        super().init_buffers(reset_adaptive_sampler=reset_adaptive_sampler)
        n, dev = self.num_envs, self.device
        cfg = self.hero_cfg
        self.h_cmd = torch.full((n, 1), float(cfg.h_cmd_default), device=dev)
        self.stand_flag = torch.zeros(n, 1, device=dev)
        self.vel_cmd = torch.zeros(n, 3, device=dev)
        self.ref_upper_dof_pos = torch.zeros(n, len(HERO_UPPER_REF_JOINT_NAMES), device=dev)
        identity = torch.tensor(_IDENTITY_QUAT_XYZW, device=dev)
        self.ref_ee_pos_pelvis = torch.zeros(n, 2, 3, device=dev)
        self.ref_ee_quat_pelvis = identity.expand(n, 2, 4).clone()
        self.ref_ee_pos_pelvis_zero_waist = torch.zeros(n, 2, 3, device=dev)
        self.ref_ee_quat_pelvis_zero_waist = identity.expand(n, 2, 4).clone()
        self.ref_h = torch.zeros(n, device=dev)
        self.ref_time_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.ref_time_steps_upper = torch.zeros(n, dtype=torch.long, device=dev)
        self.env_has_object = torch.zeros(n, dtype=torch.bool, device=dev)
        self.fix_upper_body_mask = torch.zeros(n, dtype=torch.bool, device=dev)
        self.fix_upper_body_time_steps = torch.zeros(n, dtype=torch.long, device=dev)
        self.source_tag_id = torch.zeros(n, dtype=torch.long, device=dev)
        self.h_offset = torch.zeros(n, device=dev)
        self.clip_ends = torch.zeros(n, dtype=torch.bool, device=dev)
        self.clip_last_frame = torch.zeros(n, dtype=torch.bool, device=dev)
        self._vel_cmd_lin = torch.zeros(n, 2, device=dev)
        self._vel_cmd_heading = torch.zeros(n, device=dev)


        self._odom_robot_start_xy = torch.zeros(n, 2, device=dev)
        self._odom_robot_start_yaw = torch.zeros(n, device=dev)
        self._odom_ref_start_xy = torch.zeros(n, 2, device=dev)
        self._odom_ref_start_yaw = torch.zeros(n, device=dev)
        self._odom_start_step = torch.zeros(n, dtype=torch.long, device=dev)
        self.odom_xy_err = torch.zeros(n, device=dev)
        self.odom_yaw_err = torch.zeros(n, device=dev)
        self.odom_drift_rate = torch.zeros(n, device=dev)
        self.ee_pos_err = torch.zeros(n, 2, device=dev)
        self.ee_rot_err = torch.zeros(n, 2, device=dev)
        # Curriculum state: like the sampler EMA it survives reset_all(reset_adaptive_sampler=False).
        existing = getattr(self, "h_curriculum_scale", None)
        if reset_adaptive_sampler or existing is None or existing.shape[0] != n:
            self.h_curriculum_scale = torch.full((n,), float(cfg.h_curriculum_init), device=dev)

    # ------------------------------------------------------------------ reset
    def reset(self, env_ids: torch.Tensor | None) -> None:
        """Per reset_idx: curriculum, clip/phase sampling, robot/object state, HERO commands, references."""
        env_ids = self._ensure_index_tensor(env_ids)
        if env_ids.numel() == 0:
            return
        if not self._in_soft_reset:
            self._update_h_curriculum(env_ids)
        self._sample_clip_and_phase(env_ids)
        self._write_reset_states(env_ids)
        self._align_odometry(env_ids)
        self._resample_hero_commands(env_ids)
        self._refresh_hero_refs()
        self._update_clip_end_flags()

    def _soft_reset_ended_clips(self, env_ids: torch.Tensor) -> None:
        # A rollover is not an episode end: no curriculum update for these envs.
        self._in_soft_reset = True
        try:
            super()._soft_reset_ended_clips(env_ids)
        finally:
            self._in_soft_reset = False


        post = getattr(self._env, "_post_soft_reset", None)
        if callable(post):
            post(env_ids)

    def _episode_lengths_at_reset(self, env_ids: torch.Tensor) -> torch.Tensor:
        """Length of the episodes that just ended (BaseTask zeroes episode_length_buf before command reset)."""
        env = self._env
        pending = getattr(env, "_pending_episode_lengths", None)
        if isinstance(pending, torch.Tensor) and pending.shape[0] == self.num_envs:
            return pending[env_ids]
        return env.episode_length_buf[env_ids]

    def _update_h_curriculum(self, env_ids: torch.Tensor) -> None:
        cfg = self.hero_cfg
        if self._env.is_evaluating:
            self.h_curriculum_scale[env_ids] = 1.0
            return
        lengths = self._episode_lengths_at_reset(env_ids)
        scale = self.h_curriculum_scale[env_ids]
        scale = scale + (lengths > cfg.h_curriculum_up_threshold).float() * cfg.h_curriculum_up
        scale = scale - (lengths < cfg.h_curriculum_down_threshold).float() * cfg.h_curriculum_down
        self.h_curriculum_scale[env_ids] = scale.clamp(cfg.h_curriculum_range[0], cfg.h_curriculum_range[1])

    def _sample_clip_and_phase(self, env_ids: torch.Tensor) -> None:
        """Mirror of MotionCommand.reset step 0 (wbt.py) with the prior-weighted clip draw."""
        cfg = self.hero_cfg
        env = self._env
        n = env_ids.numel()
        sampled_clip_ids = None
        if cfg.use_adaptive_timesteps_sampler:
            episode_failed = env.termination_manager.terminated[env_ids]
            if torch.any(episode_failed):
                failed_envs = env_ids[episode_failed]
                failed_at_time_step = self.time_steps[failed_envs]
                failed_at_phase = None
                failed_clip_ids = None
                if getattr(self.adaptive_timesteps_sampler, "phase_binning", False):
                    f_mids = self.motion_ids[failed_envs]
                    f_start = self.motion.motion_start_idx[f_mids]
                    f_end = self.motion.motion_end_idx[f_mids]
                    f_len = (f_end - f_start).clamp(min=1).float()
                    failed_at_phase = ((failed_at_time_step - f_start).float() / f_len).clamp(0.0, 1.0)
                    failed_clip_ids = f_mids
                self.adaptive_timesteps_sampler.update_current_bin_failed_count(
                    failed_at_time_step, failed_at_phase=failed_at_phase, failed_clip_ids=failed_clip_ids
                )
            if getattr(self.adaptive_timesteps_sampler, "per_clip", False) and not env.is_evaluating:
                sampled_clip_ids, phase = self.adaptive_timesteps_sampler.sample_clip_phase(n)
            else:
                phase = self.adaptive_timesteps_sampler.sample(n)
        else:
            phase = torch.rand(n, device=self.device)

        phase, sampled_clip_ids = self._apply_evaluation_phase_policy(phase, sampled_clip_ids)

        if sampled_clip_ids is not None:
            self.motion_ids[env_ids] = sampled_clip_ids
        elif env.is_evaluating:
            # Evaluation keeps holosoma's uniform clip choice (benchmarks iterate clips).
            self.motion_ids[env_ids] = torch.randint(0, self.motion.num_motions, (n,), device=self.device)
        else:
            self.motion_ids[env_ids] = torch.multinomial(self.clip_prior, n, replacement=True)

        start_idx = self.motion.motion_start_idx[self.motion_ids[env_ids]]
        end_idx = self.motion.motion_end_idx[self.motion_ids[env_ids]]
        motion_len = end_idx - start_idx
        self.time_steps[env_ids] = start_idx + (phase * (motion_len - 1).float()).long()

        prob = cfg.start_at_timestep_zero_prob
        if prob >= 1.0:
            self.time_steps[env_ids] = start_idx
        elif prob > 0.0:
            subset = self.time_steps[env_ids]
            rand_vals = torch.rand_like(subset, dtype=torch.float32)
            self.time_steps[env_ids] = torch.where(rand_vals < prob, start_idx, subset)

        already_last = self.time_steps[env_ids] >= end_idx - 1
        safe_second_last = torch.maximum(end_idx - 2, start_idx)
        self.time_steps[env_ids] = torch.where(already_last, safe_second_last, self.time_steps[env_ids])

        mids = self.motion_ids[env_ids]
        self.source_tag_id[env_ids] = self.motion.clip_source_tag_id[mids]

        self.env_has_object[env_ids] = self.clip_object_effective[mids] if self.scene_has_object else False

    def _setup_reference_root_velocity_frame(self) -> None:
        """Setup reference root velocity frame."""
        super()._setup_reference_root_velocity_frame()
        conflicts: list[str] = []
        for path in getattr(self, "manifest_paths", None) or []:
            if manifest_reference_root_velocity_frame(Path(path)) == "link" and self.reference_root_velocity_frame != "link":
                conflicts.append(str(path))
        self.reference_root_velocity_frame_conflicts = conflicts
        if conflicts:
            by_dir = getattr(self, "reference_root_velocity_frames_by_directory", {})
            held_out_only = bool(by_dir) and all(Path(d).name == HELD_OUT_DIR_NAME for d, f in by_dir.items() if f != "link")
            logger.warning(
                "HeroMotionCommand: the source-weights manifest(s) {} carry a certified LINK-origin contract but the reference "
                "root velocity frame resolved to {!r} ({}); {}",
                conflicts,
                self.reference_root_velocity_frame,
                by_dir,
                "expected for held_out/ (byte-identical copy, legacy COM velocity convention retained)"
                if held_out_only
                else "check the corpus layout: these clips are outside the certified inventory and carry no LINK-origin kinematic "
                "stamp (legacy COM convention), so the reset writer passes their velocity through unchanged",
            )

    def _write_reset_states(self, env_ids: torch.Tensor) -> None:
        """Reset the robot to the sampled reference pose and joint state."""
        sim = self._env.simulator
        dev = self.device
        noise = self.init_pose_cfg
        scale = noise.overall_noise_scale

        # Refresh reference origins after clip sampling and before reading the root pose.
        # Terrain-aware commands can override this hook.
        self._before_reset_pose_read(env_ids)

        root_pos = self.root_pos_w[env_ids].clone()
        root_rot = self.root_quat_w[env_ids].clone()
        root_lin_vel = self.root_lin_vel_w[env_ids].clone()
        root_ang_vel = self.root_ang_vel_w[env_ids].clone()
        dof_pos = self.joint_pos[env_ids].clone()
        dof_vel = self.joint_vel[env_ids].clone()

        def _u(shape: tuple[int, ...]) -> torch.Tensor:
            return (torch.rand(shape, device=dev) - 0.5) * 2.0

        target_dof_pos = dof_pos + _u(dof_pos.shape) * (noise.dof_pos * scale)
        limits = sim.dof_pos_limits  # (num_dofs, 2)
        target_dof_pos = torch.clip(target_dof_pos, limits[:, 0], limits[:, 1])
        target_root_pos = root_pos + _u(root_pos.shape) * (torch.tensor(noise.root_pos, device=dev) * scale)[None]
        rpy = _u((env_ids.numel(), 3)) * (torch.tensor(noise.root_rot, device=dev) * scale)
        target_root_rot = quat_mul(quat_from_euler_xyz(rpy[:, 0], rpy[:, 1], rpy[:, 2]), root_rot, w_last=True)
        target_root_lin_vel = root_lin_vel + _u(root_lin_vel.shape) * (
            torch.tensor(noise.root_lin_vel, device=dev) * scale
        )[None]
        target_root_ang_vel = root_ang_vel + _u(root_ang_vel.shape) * (
            torch.tensor(noise.root_ang_vel, device=dev) * scale
        )[None]

        sim.dof_pos[env_ids] = target_dof_pos
        sim.dof_vel[env_ids] = dof_vel
        sim.robot_root_states[env_ids, :3] = target_root_pos
        sim.robot_root_states[env_ids, 3:7] = target_root_rot


        sim.robot_root_states[env_ids, 7:10] = self._reset_root_lin_vel_for_simulator(
            env_ids, target_root_rot, target_root_lin_vel, target_root_ang_vel
        )
        sim.robot_root_states[env_ids, 10:13] = target_root_ang_vel

        if self.motion.has_object and self.scene_has_object:
            obj_pos = self.object_pos_w[env_ids]
            obj_ori = self.object_quat_w[env_ids]
            obj_lin_vel = self.object_lin_vel_w[env_ids]
            obj_pos_noise = torch.tensor([noise.object_pos], device=dev) * scale
            target_obj_pos = obj_pos + _u(obj_pos.shape) * obj_pos_noise
            object_states = torch.cat([target_obj_pos, obj_ori, obj_lin_vel, torch.zeros_like(obj_lin_vel)], dim=-1)
            parked = ~self.env_has_object[env_ids]
            if parked.any():
                object_states[parked] = self._parked_object_states(env_ids[parked])
            sim.set_actor_states([self.object_name], env_ids, object_states)

    def _parked_object_states(self, env_ids: torch.Tensor) -> torch.Tensor:
        """13-D root state parking the box (to the side, on the floor) for envs whose clip has no object -- the shared
        ``stock_object.parked_object_states`` construction (``object_park_z`` is ignored)."""
        from hero_isaacsim.managers.command.stock_object import parked_object_states  # lazy: avoids an import cycle

        origins = self._env.simulator.scene.env_origins[env_ids]
        return parked_object_states(origins).to(self.device)

    # ------------------------------------------------------------------ HERO commands
    def _resample_hero_commands(self, env_ids: torch.Tensor) -> None:
        """Sample stand flag / random velocity command / height offset / fix-upper-body mask.

        Called on every reset and every ``command_resample_time_s`` (HERO ``_resample_commands``).
        Evaluation yields the deterministic HERO defaults (stance, zero velocity, zero offset)."""
        cfg = self.hero_cfg
        n = env_ids.numel()
        dev = self.device
        evaluating = bool(self._env.is_evaluating)

        if cfg.stand_flag_mode == "bernoulli_walk_0p6":
            walking = torch.zeros(n, device=dev) if evaluating else (torch.rand(n, device=dev) < cfg.walk_prob).float()
            self.stand_flag[env_ids, 0] = walking
            lo, hi = cfg.vel_cmd_lin_range
            lin = lo + (hi - lo) * torch.rand(n, 2, device=dev)
            lin = lin * (torch.linalg.norm(lin, dim=-1, keepdim=True) > cfg.vel_cmd_dead_zone).float()
            self._vel_cmd_lin[env_ids] = lin * walking[:, None]
            self._vel_cmd_heading[env_ids] = (torch.rand(n, device=dev) * 2.0 - 1.0) * math.pi

        lo, hi = cfg.h_offset_range
        offset = lo + (hi - lo) * torch.rand(n, device=dev)
        self.h_offset[env_ids] = torch.zeros_like(offset) if evaluating else offset

        fix = torch.zeros(n, dtype=torch.bool, device=dev) if evaluating else (torch.rand(n, device=dev) < cfg.fix_upper_body_prob)
        self.fix_upper_body_mask[env_ids] = fix
        start = self.motion.motion_start_idx[self.motion_ids[env_ids]]
        length = (self.motion.motion_end_idx[self.motion_ids[env_ids]] - start).clamp(min=1)
        self.fix_upper_body_time_steps[env_ids] = start + (torch.rand(n, device=dev) * (length - 1).float()).long()

    def periodic_resample_due_ids(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """PURE: the envs (all, or the subset ``env_ids``) whose periodic HERO command resample falls on THIS control step.

        Due <=> ``episode_length_buf > 0 and episode_length_buf % period == 0`` with ``period = round(command_resample_time_s
        / dt)`` (HERO ``locomotion_command_resampling_time``; 5 s / 0.02 s = 250 steps, so the 10 s pure timeout
        at step 500 is ALWAYS a due step).  Empty while evaluating or when the resample is disabled.  No side effects."""
        cfg = self.hero_cfg
        dev = self._env.episode_length_buf.device
        empty = torch.empty(0, dtype=torch.long, device=dev)
        if self._env.is_evaluating or cfg.command_resample_time_s <= 0.0:
            return empty
        period = int(round(cfg.command_resample_time_s / float(self._env.dt)))
        if period <= 0:
            return empty
        ep = self._env.episode_length_buf
        if env_ids is None:
            return torch.where((ep > 0) & (ep % period == 0))[0]
        env_ids = env_ids.to(device=dev, dtype=torch.long)
        ep = ep[env_ids]
        return env_ids[(ep > 0) & (ep % period == 0)]

    def advance_command_transition(self, env_ids: torch.Tensor | None = None) -> torch.Tensor:
        """The per-step COMMAND transition ``step()`` applies right after the stock reference advance: the periodic resample
        (stand flag / velocity command / height offset / fix-upper-body mask, ``_resample_hero_commands``) for the envs of
        ``env_ids`` (all when ``None``) whose period falls on this control step.  Returns the resampled env ids."""
        due = self.periodic_resample_due_ids(env_ids)
        if due.numel() > 0:
            self._resample_hero_commands(due)
        return due

    def _maybe_periodic_resample(self) -> None:
        self.advance_command_transition()

    def _gather_timeline(self, timeline: torch.Tensor, idx: torch.Tensor) -> torch.Tensor:
        if timeline.device == idx.device:
            return timeline[idx]
        return timeline[idx.to(timeline.device)].to(idx.device)

    def _clip_walking_flag(self, idx: torch.Tensor) -> torch.Tensor:
        """Clip walking flag."""
        cfg = self.hero_cfg
        start = self.motion.motion_start_idx[self.motion_ids]
        last = self.motion.motion_end_idx[self.motion_ids] - 1
        half = self._stand_half_window
        lo = torch.maximum(idx - half, start)
        hi = torch.minimum(idx + half, last)
        count = (hi - lo + 1).float()
        mean_speed = (self._gather_timeline(self._speed_csum, hi + 1) - self._gather_timeline(self._speed_csum, lo)) / count
        mean_yaw = (
            self._gather_timeline(self._yaw_rate_csum, hi + 1) - self._gather_timeline(self._yaw_rate_csum, lo)
        ) / count
        stance = (mean_speed < cfg.stand_speed_thr) & (mean_yaw < cfg.stand_yaw_rate_thr)
        return (~stance).float()

    def _h_offset_scale(self, ref_h: torch.Tensor) -> torch.Tensor:
        """``s(h_ref) = clamp((h_ref - floor) / (ceiling - floor), 0, 1)``; 1 everywhere when ``ceiling <= floor``."""
        cfg = self.hero_cfg
        lo, hi = float(cfg.h_offset_scale_floor), float(cfg.h_offset_scale_ceiling)
        if hi <= lo:
            return torch.ones_like(ref_h)
        return ((ref_h - lo) / (hi - lo)).clamp(0.0, 1.0)

    def _robot_heading(self) -> torch.Tensor:
        """Yaw of the simulated robot root (xyzw quaternion), used by the HERO heading controller."""
        q = self.robot_root_quat_w
        forward = quat_apply(q, torch.tensor([1.0, 0.0, 0.0], device=q.device).expand(q.shape[0], 3), w_last=True)
        return torch.atan2(forward[:, 1], forward[:, 0])

    def _refresh_hero_refs(self) -> None:
        """Compute every HERO command / reference tensor at the lookahead frame."""
        cfg = self.hero_cfg
        motion = self.motion
        start = motion.motion_start_idx[self.motion_ids]
        last = motion.motion_end_idx[self.motion_ids] - 1
        idx = torch.minimum(self.time_steps + cfg.ref_lookahead_frames, last)
        idx_upper = torch.where(self.fix_upper_body_mask, self.fix_upper_body_time_steps.clamp(min=0), idx)
        idx_upper = torch.minimum(torch.maximum(idx_upper, start), last)
        self.ref_time_steps = idx
        self.ref_time_steps_upper = idx_upper

        # Stand flag (from_clip mode is re-derived every frame; bernoulli mode is sampled at reset).
        if cfg.stand_flag_mode == "from_clip":
            self.stand_flag[:, 0] = self._clip_walking_flag(idx)
        walking = self.stand_flag[:, 0] > 0.5

        # Upper-body references, with optional freezing and zero-waist walking references.
        ref_upper = motion.frames("joint_pos", idx_upper)[:, self._upper_dof_idx].clone()
        ee_pos = motion.frames("ee_pos_pelvis", idx_upper).clone()
        ee_quat = motion.frames("ee_quat_pelvis", idx_upper).clone()
        ee_pos_zw = motion.frames("ee_pos_pelvis_zero_waist", idx_upper)
        ee_quat_zw = motion.frames("ee_quat_pelvis_zero_waist", idx_upper)
        if cfg.zero_waist_when_walking and walking.any():
            ref_upper[walking, : len(WAIST_DOF_IDX)] = 0.0
            ee_pos[walking] = ee_pos_zw[walking]
            ee_quat[walking] = ee_quat_zw[walking]
        self.ref_upper_dof_pos = ref_upper
        self.ref_ee_pos_pelvis = ee_pos
        self.ref_ee_quat_pelvis = ee_quat
        self.ref_ee_pos_pelvis_zero_waist = ee_pos_zw
        self.ref_ee_quat_pelvis_zero_waist = ee_quat_zw

        # Height command.
        self.ref_h = motion.frames("h_ref", idx)
        scaled_offset = self.h_offset * self.h_curriculum_scale
        if cfg.h_offset_from_clip:


            h_cmd = self.ref_h + scaled_offset * self._h_offset_scale(self.ref_h)
            h_cmd = h_cmd.clamp(min=float(cfg.h_cmd_min))
        else:
            h_cmd = cfg.h_cmd_default + scaled_offset * (1.0 - self.stand_flag[:, 0])
        self.h_cmd = h_cmd.unsqueeze(1)

        # Velocity command in the heading frame.
        walking_f = walking.float()
        if cfg.stand_flag_mode == "from_clip":
            root_quat = motion.frames("body_quat_w", idx)[:, 0]
            root_lin = motion.frames("body_lin_vel_w", idx)[:, 0]
            root_ang = motion.frames("body_ang_vel_w", idx)[:, 0]
            v_heading = quat_rotate_inverse(yaw_quat(root_quat, w_last=True), root_lin, w_last=True)
            vel = torch.stack([v_heading[:, 0], v_heading[:, 1], root_ang[:, 2]], dim=-1)
        else:
            yaw_err = wrap_to_pi(self._vel_cmd_heading - self._robot_heading())
            lo, hi = cfg.vel_cmd_yaw_rate_range
            yaw_rate = torch.clip(cfg.vel_cmd_heading_gain * yaw_err, lo, hi)
            vel = torch.cat([self._vel_cmd_lin, yaw_rate.unsqueeze(1)], dim=-1)
        self.vel_cmd = vel * walking_f[:, None]


    @property
    def ref_ee_pos_w(self) -> torch.Tensor:
        """[N,2,3] palm reference in world: ``robot_root_pos + R(robot_root_quat) @ ref_ee_pos_pelvis``.

        HERO tracks the hands RELATIVE to the pelvis, so the world-frame target is anchored to the
        robot's current root (not the clip's root); ``|robot_ee_pos_w - ref_ee_pos_w|`` therefore equals
        the norm of the HERO ``dif_local_rigid_body_pos`` term.  Includes env origins (simulator frame)."""
        root_pos = self.robot_root_pos_w
        root_quat = self.robot_root_quat_w
        return root_pos[:, None, :] + quat_apply(root_quat[:, None, :].expand(-1, 2, 4), self.ref_ee_pos_pelvis, w_last=True)

    @property
    def ref_ee_quat_w(self) -> torch.Tensor:
        """[N,2,4] xyzw palm orientation reference in world (robot root ⊗ pelvis-frame reference)."""
        root_quat = self.robot_root_quat_w[:, None, :].expand(-1, 2, 4)
        return quat_mul(root_quat, self.ref_ee_quat_pelvis, w_last=True)

    @property
    def robot_ee_pos_w(self) -> torch.Tensor:
        """[N,2,3] simulated palm point: wrist_yaw position + R(wrist_yaw) @ PALM_OFFSET."""
        sim = self._env.simulator
        wrist_pos = sim._rigid_body_pos[:, self._ee_body_idx, :]
        wrist_quat = sim._rigid_body_rot[:, self._ee_body_idx, :]
        offset = self._palm_offset[None].expand(wrist_pos.shape[0], 2, 3)
        return wrist_pos + quat_apply(wrist_quat, offset, w_last=True)

    @property
    def robot_ee_quat_w(self) -> torch.Tensor:
        """[N,2,4] xyzw simulated palm orientation (== wrist_yaw orientation, palm is a fixed child)."""
        return self._env.simulator._rigid_body_rot[:, self._ee_body_idx, :]

    def compute_ee_errors(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Per-hand palm error vs the pelvis-relative reference: (pos_err [N,2] m, rot_err [N,2] rad), [left, right]."""
        pos_err = torch.linalg.norm(self.robot_ee_pos_w - self.ref_ee_pos_w, dim=-1)
        rot_err = quat_error_magnitude(
            self.ref_ee_quat_w.reshape(-1, 4), self.robot_ee_quat_w.reshape(-1, 4), w_last=True
        ).view(-1, 2)
        return pos_err, rot_err


    @staticmethod
    def _yaw_of(quat_xyzw: torch.Tensor) -> torch.Tensor:
        forward = quat_apply(
            quat_xyzw, torch.tensor([1.0, 0.0, 0.0], device=quat_xyzw.device).expand(quat_xyzw.shape[0], 3), w_last=True
        )
        return torch.atan2(forward[:, 1], forward[:, 0])

    def _align_odometry(self, env_ids: torch.Tensor) -> None:
        """Capture the episode-start alignment between the robot root and the clip root.

        Called from ``reset`` (hard reset AND clip rollover -- both teleport the robot onto the clip),
        after the simulator root state has been written.  The reference root is read at the frame the
        robot was just placed on (``time_steps``), so the recorded pair differs only by the reset noise."""
        self._odom_robot_start_xy[env_ids] = self.robot_root_pos_w[env_ids, :2]
        self._odom_robot_start_yaw[env_ids] = self._yaw_of(self.robot_root_quat_w[env_ids])
        self._odom_ref_start_xy[env_ids] = self.root_pos_w[env_ids, :2]
        self._odom_ref_start_yaw[env_ids] = self._yaw_of(self.root_quat_w[env_ids])
        self._odom_start_step[env_ids] = self._env.episode_length_buf[env_ids]

    def expected_root_pose_from_odometry(self) -> tuple[torch.Tensor, torch.Tensor]:
        """Where the robot root SHOULD be now if it had followed the clip root exactly since alignment.

        Returns (xy [N,2] world incl. env origins, yaw [N] rad): the clip root displacement since the
        aligned start frame, rotated by the start yaw offset and applied to the robot's start pose."""
        dyaw = wrap_to_pi(self._odom_robot_start_yaw - self._odom_ref_start_yaw)
        ref_xy = self.root_pos_w[:, :2] - self._odom_ref_start_xy
        c, s_ = torch.cos(dyaw), torch.sin(dyaw)
        rot_xy = torch.stack([c * ref_xy[:, 0] - s_ * ref_xy[:, 1], s_ * ref_xy[:, 0] + c * ref_xy[:, 1]], dim=-1)
        expected_xy = self._odom_robot_start_xy + rot_xy
        expected_yaw = wrap_to_pi(self._yaw_of(self.root_quat_w) + dyaw)
        return expected_xy, expected_yaw

    def compute_odometry_error(self) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """GLOBAL base odometry error vs the reference root .

        Returns ``(xy_err [N] m, yaw_err [N] rad >= 0, drift_rate [N] m/s)`` where drift_rate is
        ``xy_err / elapsed_time_since_alignment`` (clamped to >= one control step).  Meaningful for
        ``from_clip`` where ``vel_cmd`` follows the clip root; otherwise this is diagnostic only."""
        expected_xy, expected_yaw = self.expected_root_pose_from_odometry()
        xy_err = torch.linalg.norm(self.robot_root_pos_w[:, :2] - expected_xy, dim=-1)
        yaw_err = wrap_to_pi(self._yaw_of(self.robot_root_quat_w) - expected_yaw).abs()
        elapsed = (self._env.episode_length_buf - self._odom_start_step).clamp(min=1).float() * float(self._env.dt)
        return xy_err, yaw_err, xy_err / elapsed

    def metrics_split_by_gait(self) -> dict[str, torch.Tensor]:
        """Not pushed into ``self.metrics`` (the episode accumulator averages per-env tensors); the
        tracking manager's ``_update_log_dict`` logs these directly."""
        walking = self.stand_flag[:, 0] > 0.5
        out: dict[str, torch.Tensor] = {}
        for gait, mask in (("walk", walking), ("stance", ~walking)):
            for side_i, side in enumerate(("left", "right")):
                out[f"ee/{side}_pos_err_m/{gait}"] = self.ee_pos_err[mask, side_i].mean()
                out[f"ee/{side}_rot_err_deg/{gait}"] = torch.rad2deg(self.ee_rot_err[mask, side_i]).mean()
            out[f"base/odom_xy_err_m/{gait}"] = self.odom_xy_err[mask].mean()
            out[f"base/odom_yaw_err_deg/{gait}"] = torch.rad2deg(self.odom_yaw_err[mask]).mean()
        return out

    # ------------------------------------------------------------------ clip end handling
    def _handle_ended_clips(self) -> torch.Tensor:
        """Handle ended clips."""
        per_motion_end = self.motion.motion_end_idx[self.motion_ids]
        ended = torch.where(self.time_steps >= per_motion_end)[0]
        if ended.numel() == 0:
            return ended
        rollover = self.motion.clip_rollover[self.motion_ids[ended]]
        if not self.hero_cfg.rollover_at_clip_end:
            rollover = torch.zeros_like(rollover)
        hold_ids = ended[~rollover]
        if hold_ids.numel() > 0:
            self.time_steps[hold_ids] = per_motion_end[hold_ids] - 1
        roll_ids = ended[rollover]
        if roll_ids.numel() > 0:
            self._soft_reset_ended_clips(roll_ids)
        return ended

    def _update_clip_end_flags(self) -> None:
        """``clip_last_frame``: reference reached the clip's last frame; ``clip_ends``: same, hold-policy clips only.

        ``hero:clip_ends`` (is_timeout=True) must fire only for clips that HOLD at their end; a
        rollover clip is teleported to a new clip inside ``step()`` and must not end the episode."""
        last = self.motion.motion_end_idx[self.motion_ids] - 1
        self.clip_last_frame = self.time_steps >= last
        self.clip_ends = self.clip_last_frame & ~self.motion.clip_rollover[self.motion_ids]

    # ------------------------------------------------------------------ step / metrics
    def step(self) -> None:
        super().step()
        self._maybe_periodic_resample()
        self._refresh_hero_refs()
        self._update_clip_end_flags()

    def update_metrics(self):
        super().update_metrics()

        self.ee_pos_err, self.ee_rot_err = self.compute_ee_errors()
        self.odom_xy_err, self.odom_yaw_err, self.odom_drift_rate = self.compute_odometry_error()
        for i, side in enumerate(("left", "right")):
            self.metrics[f"ee/{side}_pos_err_m"] = self.ee_pos_err[:, i]
            self.metrics[f"ee/{side}_rot_err_deg"] = torch.rad2deg(self.ee_rot_err[:, i])
        self.metrics["base/odom_xy_err_m"] = self.odom_xy_err
        self.metrics["base/odom_yaw_err_deg"] = torch.rad2deg(self.odom_yaw_err)
        self.metrics["base/drift_rate_m_per_s"] = self.odom_drift_rate
        if self.hero_cfg.terrain_relative_heights:
            from hero_isaacsim.managers.termination.hero import terrain_ground_z  # noqa: PLC0415 (light; no cycle)

            root_h = self.robot_root_pos_w[:, 2] - terrain_ground_z(self._env)
        else:
            root_h = self.robot_root_pos_w[:, 2] - self._env.simulator.scene.env_origins[:, 2]
        self.metrics["base/height_err_m"] = (root_h - self.h_cmd[:, 0]).abs()
        self.metrics["motion/object_clip_effective_frac"] = self.clip_object_effective[self.motion_ids].float()
        self.metrics["motion/stand_flag_walking_frac"] = self.stand_flag[:, 0]
        self.metrics["motion/h_cmd"] = self.h_cmd[:, 0]
        self.metrics["motion/h_curriculum_scale"] = self.h_curriculum_scale
        self.metrics["motion/fix_upper_body_frac"] = self.fix_upper_body_mask.float()
        self.metrics["motion/env_has_object_frac"] = self.env_has_object.float()
        self.metrics["motion/clip_hold_frac"] = (~self.motion.clip_rollover[self.motion_ids]).float()
        counts = torch.bincount(self.source_tag_id, minlength=len(self.motion.source_tags)).float()
        frac = counts / counts.sum().clamp_min(1.0)
        for i, tag in enumerate(self.motion.source_tags):
            self.metrics[f"motion/source_frac_{tag}"] = frac[i]
        sampler = getattr(self, "adaptive_timesteps_sampler", None)
        if isinstance(sampler, HeroAdaptiveTimestepsSampler):
            for key, value in sampler.expected_source_fractions().items():
                self.metrics[key.replace("motion/source_frac_", "motion/source_frac_expected_")] = value
