"""G1/Dex3 robot, motor parameters, and packaged asset locations."""

from __future__ import annotations

import os
from dataclasses import replace
from pathlib import Path

from holosoma.config_types.robot import RobotConfig
from holosoma.config_values.robot import g1_29dof

from hero_isaacsim.constants import (
    ACTION_SCALE,
    DOF_NAMES,
    HERO_BODY_NAMES_34,
    HOLOSOMA_BODY_NAMES_32,
    INIT_POS_Z,
    MJCF_SCENE_FILE_NAME,
    MOTOR_7520_22,
    PALM_BODY_NAMES,
    ROBOT_PRESET_NAME,
    URDF_FILE_NAME,
)

# --------------------------------------------------------------------------------------------------
# Asset paths (absolute, resolved from the package location; layout mirrors holosoma's
# ``asset_root="@holosoma/data/robots"`` + ``urdf_file="g1/..."``). holosoma writes the converted USD to
# ``<asset_root>/converted_rank<slot>``; generated USD files stay out of Git.
# --------------------------------------------------------------------------------------------------
from hero_isaacsim.paths import RELEASE_ROOT as REPO_ROOT, ROBOT_ASSET_ROOT
ROBOT_ASSET_SUBDIR: str = "g1_modified"
URDF_FILE: str = f"{ROBOT_ASSET_SUBDIR}/{URDF_FILE_NAME}"
MJCF_FILE: str = f"{ROBOT_ASSET_SUBDIR}/{MJCF_SCENE_FILE_NAME}"
URDF_PATH: Path = ROBOT_ASSET_ROOT / URDF_FILE
MJCF_PATH: Path = ROBOT_ASSET_ROOT / MJCF_FILE


# --------------------------------------------------------------------------------------------------
# Helpers (pure python; mirror holosoma's per-joint matching so tests / export metadata can reuse them)
# --------------------------------------------------------------------------------------------------
def _override_per_joint(values: list[float], overrides: dict[str, float], dof_names=DOF_NAMES) -> list[float]:
    """Copy ``values`` (one per joint in ``dof_names``) replacing entries whose joint name contains an override key."""
    assert len(values) == len(dof_names), (len(values), len(dof_names))
    out = list(values)
    for i, name in enumerate(dof_names):
        for key, val in overrides.items():
            if key in name:
                out[i] = float(val)
    return out


def pd_gains_for(robot_cfg: RobotConfig) -> tuple[list[float], list[float]]:
    """Per-joint (kp, kd) exactly as ``JointPositionActionTerm._configure_pd_gains`` resolves them.

    Substring match of every ``control.stiffness`` key against the joint name; the *last* matching key wins
    (holosoma loops over all keys without ``break``). Raises if a joint has no match (holosoma raises too)."""
    kp: list[float] = []
    kd: list[float] = []
    for name in robot_cfg.dof_names:
        matched = None
        for key, stiffness in robot_cfg.control.stiffness.items():
            if key in name:
                matched = (float(stiffness), float(robot_cfg.control.damping[key]))
        if matched is None:
            raise ValueError(f"PD gains for joint '{name}' were not defined in the robot preset.")
        kp.append(matched[0])
        kd.append(matched[1])
    return kp, kd


def action_scales_for(robot_cfg: RobotConfig) -> list[float]:
    """Per-joint action scale exactly as ``JointPositionActionTerm._configure_action_scales`` computes it.

    ``action_scale * effort_limit / kp`` when ``action_scales_by_effort_limit_over_p_gain`` else the flat scale."""
    kp, _ = pd_gains_for(robot_cfg)
    ctrl = robot_cfg.control
    if not ctrl.action_scales_by_effort_limit_over_p_gain:
        return [float(ctrl.action_scale)] * len(kp)
    return [
        0.0 if k == 0.0 else float(ctrl.action_scale) * float(effort) / k
        for effort, k in zip(robot_cfg.dof_effort_limit_list, kp, strict=True)
    ]


# --------------------------------------------------------------------------------------------------
# The preset
# --------------------------------------------------------------------------------------------------
_HIP_PITCH_KEY = "hip_pitch"  # substring key used by holosoma's stiffness/damping dicts (matches left_ & right_)

g1_29dof_dex3_m12: RobotConfig = replace(
    g1_29dof,
    # --- bodies: 32 canonical + Dex3 palms (order enforced by holosoma find_bodies(preserve_order=True)) ---
    num_bodies=len(HERO_BODY_NAMES_34),
    body_names=list(HERO_BODY_NAMES_34),
    randomize_link_body_names=[*g1_29dof.randomize_link_body_names, *PALM_BODY_NAMES],
    # --- Model-12 plant: hip_pitch -> 7520_22 (stock g1_29dof has 7520_14 there) ---
    dof_effort_limit_list=_override_per_joint(
        g1_29dof.dof_effort_limit_list, {_HIP_PITCH_KEY: MOTOR_7520_22["effort_limit"]}
    ),
    dof_armature_list=_override_per_joint(g1_29dof.dof_armature_list, {_HIP_PITCH_KEY: MOTOR_7520_22["armature"]}),

    # Match the actuator velocity limit to the hip-pitch motor.
    dof_vel_limit_list=_override_per_joint(
        g1_29dof.dof_vel_limit_list, {_HIP_PITCH_KEY: MOTOR_7520_22["velocity_limit"]}
    ),
    init_state=replace(g1_29dof.init_state, pos=[0.0, 0.0, INIT_POS_Z]),
    control=replace(
        g1_29dof.control,
        stiffness={**g1_29dof.control.stiffness, _HIP_PITCH_KEY: MOTOR_7520_22["stiffness"]},
        damping={**g1_29dof.control.damping, _HIP_PITCH_KEY: MOTOR_7520_22["damping"]},
        action_scale=ACTION_SCALE,
        action_scales_by_effort_limit_over_p_gain=True,
    ),
    asset=replace(
        g1_29dof.asset,
        asset_root=str(ROBOT_ASSET_ROOT),
        urdf_file=URDF_FILE,
        usd_file=None,  # converted from the URDF at startup (collapse_fixed_joints=True, palms/foot points kept)

        # Use the configured training foot geometry during URDF conversion.
        foot_collision_profile="sonic_train",
        replace_cylinder_with_capsule=True,
        xml_file=MJCF_FILE,
        robot_type="g1_29dof",  # robot type used by the Unitree bridge and logging
        # Fixed-finger training geometry can overlap the hip during reference poses.
        # This matches the self-collision setting of the training plant.
        enable_self_collisions=False,
    ),
)

HERO_ROBOT_PRESETS: dict[str, RobotConfig] = {ROBOT_PRESET_NAME: g1_29dof_dex3_m12}

__all__ = [
    "HERO_ROBOT_PRESETS",
    "MJCF_FILE",
    "MJCF_PATH",
    "REPO_ROOT",
    "ROBOT_ASSET_ROOT",
    "URDF_FILE",
    "URDF_PATH",
    "action_scales_for",
    "g1_29dof_dex3_m12",
    "pd_gains_for",
]
