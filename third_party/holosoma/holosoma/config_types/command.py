"""Configuration types for the command & curriculum manager."""

from __future__ import annotations

from dataclasses import field
from typing import Any

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class CommandTermCfg:
    """Configuration for a single command or curriculum hook."""

    func: str
    """Import path for the command hook (function or callable class)."""

    params: dict[str, Any] = field(default_factory=dict)
    """Additional parameters forwarded to the hook."""


@dataclass(frozen=True)
class CommandManagerCfg:
    """Configuration for the command manager."""

    params: dict[str, Any] = field(default_factory=dict)
    """Global parameters shared across command hooks."""

    setup_terms: dict[str, CommandTermCfg] = field(default_factory=dict)
    """Hooks invoked during environment setup."""

    reset_terms: dict[str, CommandTermCfg] = field(default_factory=dict)
    """Hooks invoked on environment reset."""

    step_terms: dict[str, CommandTermCfg] = field(default_factory=dict)


########################################################################################################################
# Motion command configuration
########################################################################################################################
@dataclass(frozen=True)
class NoiseToInitialPoseConfig:
    """Initial pose of the robot and object to those in the motion file."""

    overall_noise_scale: float = 0.0
    """Overall noise scale for the initial pose."""

    dof_pos: float = 0.0
    """Noise scale for the initial dof position."""

    root_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root position x, y, z."""

    root_rot: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root rotation roll, pitch, yaw."""

    root_lin_vel: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root linear velocity vx, vy, vz."""

    root_ang_vel: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for root angular velocity wx, wy, wz."""

    object_pos: list[float] = field(default_factory=lambda: [0.0, 0.0, 0.0])
    """noise scale for object position x, y, z."""


@dataclass(frozen=True)
class MotionConfig:
    """Motion related configuration for Whole Body Tracking.

    NOTE:
    - Motion file is assumed to be in the format of:
      - joint_pos: (T, J)
      - joint_vel: (T, J)

      - body_pos_w: (T, B, 3)
      - body_quat_w: (T, B, 4) # wxyz -> xyzw
      - body_lin_vel_w: (T, B, 3)
      - body_ang_vel_w: (T, B, 3)

      If object is present in the motion file, it is assumed to be in the format of:
      - object_pos_w: (T, 3)
      - object_quat_w: (T, 4)
      - object_lin_vel_w: (T, 3)
      - object_ang_vel_w: (T, 3)

      If the motion clip assumes a terrain, the terrain has to be specified in holosoma/config/terrain/terrain_wbt.yaml
    """

    motion_file: str
    """Motion file (.npz) that contains motion_clips to track. """

    body_name_ref: list[str]
    """Body name of the reference frame (in general, torso_link). """
    body_names_to_track: list[str]
    """Key body names to track, used for reward/termination computation."""

    motion_dir: str = ""
    """Directory (or comma-separated directories) of .npz motion files.
    When non-empty, takes precedence over motion_file."""

    # motion sampling related
    use_adaptive_timesteps_sampler: bool = False
    """During training, whether to prioritize training on motion segments where the robot fails often."""

    adaptive_sampler_phase_binning: bool = False
    """Bin failures by normalized per-clip phase instead of the concatenated timeline.

    Bins have approximately one-second resolution based on the longest clip.
    All clips share one phase table and clip selection remains uniform.
    Enable adaptive_sampler_per_clip for separate failure tables and joint clip/phase sampling."""

    adaptive_sampler_per_clip: bool = False
    """Sample (clip, start-phase) jointly from per-clip failure counts plus a uniform floor.

    Each clip has its own normalized phase bins. Supersedes adaptive_sampler_phase_binning
    when enabled; a single clip reduces to phase-binning behavior. Evaluation uses uniform
    clip selection and starts at phase zero."""

    adaptive_sampler_uniform_ratio: float = 0.1
    """Fraction of the per-(clip, phase-bin) sampling distribution reserved for uniform exploration.
    The historical value is 0.1. This field only changes behavior when explicitly overridden."""

    adaptive_sampler_clip_temperature: float = 1.0
    """Temperature applied to the final clip marginal in per-clip mode. Values greater than 1 flatten
    the distribution across clips while preserving each clip's conditional phase curriculum. The
    backward-compatible default 1.0 disables tempering."""

    adaptive_sampler_clip_max_probability: float = 1.0
    """Hard upper bound on any final clip sampling probability in per-clip mode. It must be at least
    ``1 / num_clips`` for a multi-clip registry. The backward-compatible default 1.0 disables capping.
    This limit is applied after phase smoothing and temperature scaling, so the reported clip top-1
    probability obeys the configured bound."""

    start_at_timestep_zero_prob: float = 0.0
    """Probability of starting at timestep zero."""

    freeze_at_timestep_zero_prob: float = 0.0
    """When starting at timestep 0, probability of freezing motion counter at 0 (not advancing).
    This makes the robot practice holding the initial pose. Only applies when episode starts at timestep 0.
    Sampled independently each policy step; expected wait is roughly 1 / (1 - p) steps before unfreezing."""

    rollover_at_clip_end: bool = True
    """Whether a completed clip resamples and teleports robot/object to a new clip.

    The historical training/evaluation behavior is preserved by the ``True``
    default.  Long-horizon task evaluators that own their task clock must set
    this to ``False``; the command then holds the last valid reference frame
    instead of introducing an unreported physical state discontinuity.
    """

    enable_default_pose_prepend: bool = False
    """If True, pre-append interpolated frames from default pose to the motion's first pose.
    This provides a smooth transition trajectory that the policy can track."""

    default_pose_prepend_duration_s: float = 2.0
    """Duration in seconds of the pre-appended interpolation phase.
    Only used if enable_default_pose_prepend is True."""

    enable_default_pose_append: bool = False
    """If True, post-append interpolated frames from the motion's last pose back to default pose.
    This provides a smooth return trajectory that the policy can track."""

    default_pose_append_duration_s: float = 2.0
    """Duration in seconds of the post-appended interpolation phase.
    Only used if enable_default_pose_append is True."""

    motion_storage_device: str = ""
    """Device holding the bulk motion tensors (joint/body/object trajectories).
    Empty (default) stores them on the compute device.  Set to "cpu" for very
    large motion sets (tens of thousands of clips): bulk tensors then stay in
    host memory and only the frames indexed each step are transferred to the
    GPU.  Small per-clip index tensors always live on the compute device."""

    # noise related
    noise_to_initial_pose: NoiseToInitialPoseConfig = field(default_factory=NoiseToInitialPoseConfig)
