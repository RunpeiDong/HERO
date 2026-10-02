from __future__ import annotations

from dataclasses import field
from typing import Literal

from pydantic.dataclasses import dataclass


@dataclass(frozen=True)
class RobotBridgeConfig:
    """Bridge-specific configuration for robot SDK communication.

    Currently supports sim2sim (holosoma/run_sim.py) only.
    """

    sdk_type: str = "unitree"
    """SDK type for robot communication ('unitree', 'booster', 'ros2')."""

    motor_type: str = "serial"
    """Motor communication type ('serial', etc.)."""


@dataclass(frozen=True)
class RobotInitState:
    pos: list[float]
    rot: list[float]
    lin_vel: list[float]
    ang_vel: list[float]
    default_joint_angles: dict[str, float]


@dataclass(frozen=True)
class RobotControlConfig:
    control_type: str
    stiffness: dict[str, float]
    damping: dict[str, float]
    action_scale: float
    action_clip_value: float
    clip_actions: bool
    clip_torques: bool
    action_scales_by_effort_limit_over_p_gain: bool = False


@dataclass(frozen=True)
class RobotAssetConfig:
    asset_root: str
    collapse_fixed_joints: bool
    replace_cylinder_with_capsule: bool
    flip_visual_attachments: bool
    armature: float
    thickness: float
    max_angular_velocity: float
    max_linear_velocity: float
    angular_damping: float
    linear_damping: float
    urdf_file: str
    usd_file: str | None
    xml_file: str
    robot_type: str
    enable_self_collisions: bool
    default_dof_drive_mode: int
    fix_base_link: bool
    mesh_root: str | None = None
    density: float | None = None
    disable_gravity: bool | None = None
    foot_collision_profile: Literal["source", "sonic_train", "sonic_box"] = "source"
    """Isaac URDF collision preparation. ``source`` preserves the supplied asset.
    ``sonic_train`` uses the official training URDF's seven cylinders per foot
    and requires ``replace_cylinder_with_capsule=True`` like its official importer.
    ``sonic_box`` uses the separate MuJoCo deployment sole boxes. Both replace
    only G1 ankle-roll fixed-subtree collisions before USD conversion; visuals, inertials,
    marker frames and joint definitions stay unchanged. Native USD input is
    rejected for this profile because the URDF transform cannot certify it.
    The source URDF is never edited. Serialized in checkpoints so new training
    physics is explicit; older configurations retain the source default.
    """

    # Per-environment robot variants for the Isaac Sim backend.
    urdf_files: list[str] | None = None
    """Optional list of URDFs (relative to ``asset_root`` like ``urdf_file``), one per robot VARIANT; the list index is
    the variant id.  When set, the Isaac Sim backend converts every file to USD and spawns a per-env mix of them
    (Isaac Lab multi-asset spawning; ``scene.replicate_physics`` is forced to False).  All files must share the joint
    list / order and the set of rigid bodies that survive ``collapse_fixed_joints`` (Isaac Lab requirement).
    ``urdf_file`` stays the single-URDF fallback (MuJoCo backend, tools, ``usd_file``-less evaluators) and should be
    ``urdf_files[0]``.  ``None`` (default) = stock single-asset behaviour, byte-identical to upstream."""
    urdf_variant_weights: tuple[int, ...] | None = None
    """Per-variant repetition counts (ints, same length as ``urdf_files``), e.g. ``(2, 1, 1)`` = variant 0 on ~50 % of
    the envs, variants 1 and 2 on ~25 % each.  ``None`` = uniform.  Only read when ``urdf_files`` is set."""
    urdf_variant_names: tuple[str, ...] | None = None
    """Optional human-readable variant names (same length as ``urdf_files``), exposed as
    ``simulator.robot_variant_names`` for metrics / logging.  ``None`` = URDF file stems."""


@dataclass(frozen=True)
class RobotForceControlConfig:
    apply_force_link: list[str] | None = None
    left_hand_link: str | None = None
    right_hand_link: str | None = None


@dataclass(frozen=True)
class ObjectConfig:
    object_urdf_path: str | None = None

    # Per-environment object variants, using the same mechanism as RobotAssetConfig.urdf_files.
    object_urdf_paths: list[str] | None = None
    """Optional list of object URDFs (paths as ``object_urdf_path`` takes them: absolute, or ``resolve_data_file_path`` forms), one
    per object VARIANT (box shape); the list index is the variant id.  When set, the Isaac Sim backend converts every file to USD
    and spawns a per-env mix of them under ``/World/envs/env_.*/Object`` (``scene.replicate_physics`` is forced to False, the
    stock ``clone_environments()`` is skipped like for the robot variants) and publishes ``simulator.object_variant_ids``
    (LongTensor[num_envs]) / ``object_variant_names`` / ``object_variant_sizes``.  ``object_urdf_path`` is then IGNORED by the
    Isaac backend and stays the single-URDF fallback for other consumers (should be ``object_urdf_paths[0]``).  ``None``
    (default) = stock single-object behaviour, byte-identical to upstream."""
    object_variant_weights: tuple[int, ...] | None = None
    """Per-variant repetition counts (ints, same length as ``object_urdf_paths``), e.g. ``(6, 4, 2, ...)``; ``None`` = uniform.
    Only read when ``object_urdf_paths`` is set."""
    object_variant_names: tuple[str, ...] | None = None
    """Optional variant names (same length as ``object_urdf_paths``), exposed as ``simulator.object_variant_names`` and used by
    ``hero_isaacsim`` to map the clip meta ``shape_id`` strings to variant ids.  ``None`` = URDF file stems (the generated box
    assets are named ``<shape_id>.urdf``, so the stems ARE the shape ids)."""


@dataclass(frozen=True)
class RobotConfig:
    num_bodies: int
    dof_obs_size: int
    algo_obs_dim_dict: dict[str, int]
    actions_dim: int
    policy_obs_dim: int
    critic_obs_dim: int
    contact_pairs_multiplier: int
    key_bodies: list[str]
    num_feet: int
    foot_body_name: str
    """Name/pattern of the real foot link(s) used for contacts and kinematics."""
    foot_height_name: str
    """Name/pattern of auxiliary 'fake' foot link(s) used only to compute foot height/clearance"""
    knee_name: str
    torso_name: str
    dof_names: list[str]
    upper_dof_names: list[str]
    upper_left_arm_dof_names: list[str]
    upper_right_arm_dof_names: list[str]
    lower_dof_names: list[str]
    has_torso: bool
    has_upper_body_dof: bool
    left_ankle_dof_names: list[str]
    right_ankle_dof_names: list[str]
    knee_dof_names: list[str]
    hips_dof_names: list[str]
    dof_pos_lower_limit_list: list[float]
    dof_pos_upper_limit_list: list[float]
    dof_vel_limit_list: list[float]
    dof_effort_limit_list: list[float]
    dof_armature_list: list[float]
    dof_joint_friction_list: list[float]
    body_names: list[str]
    terminate_after_contacts_on: list[str]
    penalize_contacts_on: list[str]
    init_state: RobotInitState
    randomize_link_body_names: list[str]

    control: RobotControlConfig
    asset: RobotAssetConfig


    object: ObjectConfig = field(default_factory=ObjectConfig)

    bridge: RobotBridgeConfig = field(default_factory=RobotBridgeConfig)
    """Bridge SDK configuration for this robot."""

    waist_dof_names: list[str] | None = None
    waist_yaw_dof_name: str | None = None
    waist_roll_dof_name: str | None = None
    waist_pitch_dof_name: str | None = None

    arm_dof_names: list[str] | None = None
    left_arm_dof_names: list[str] | None = None
    right_arm_dof_names: list[str] | None = None

    symmetry_joint_names: dict[str, str] | None = None
    flip_sign_joint_names: list[str] | None = None

    apply_dof_armature_in_isaacgym: bool = True
    knee_joint_min_threshold: float = 0.2
    lidar_height_offset: float = 0.5

    soft_dof_pos_limit: float = 0.95
    termination_close_to_dof_pos_limit: float = 0.98
