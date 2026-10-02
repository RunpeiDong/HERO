"""Whole Body Tracking command presets for the G1 robot."""

from dataclasses import replace

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg, MotionConfig, NoiseToInitialPoseConfig

init_pose_config = NoiseToInitialPoseConfig(
    overall_noise_scale=1.0,
    dof_pos=0.1,
    root_pos=[0.05, 0.05, 0.01],
    root_rot=[0.1, 0.1, 0.2],
    root_lin_vel=[0.5, 0.5, 0.2],
    root_ang_vel=[0.52, 0.52, 0.78],
    object_pos=[0.05, 0.05, 0.0],
)

motion_config = MotionConfig(
    motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj.npz",
    body_names_to_track=[
        "pelvis",
        "left_hip_roll_link",
        "left_knee_link",
        "left_ankle_roll_link",
        "right_hip_roll_link",
        "right_knee_link",
        "right_ankle_roll_link",
        "torso_link",
        "left_shoulder_roll_link",
        "left_elbow_link",
        "left_wrist_yaw_link",
        "right_shoulder_roll_link",
        "right_elbow_link",
        "right_wrist_yaw_link",
    ],
    body_name_ref=["torso_link"],
    use_adaptive_timesteps_sampler=True,
    noise_to_initial_pose=init_pose_config,
)

motion_config_w_object = replace(
    motion_config,
    motion_file="holosoma/data/motions/g1_29dof/whole_body_tracking/sub3_largebox_003_mj_w_obj.npz",
)

# Load every motion clip from motion_dir. Set this path through the CLI before training.


motion_config_w_object_multitask = replace(
    motion_config_w_object,
    motion_dir="",  # All *.npz files in this directory are loaded.
)

g1_29dof_wbt_command = CommandManagerCfg(
    params={},
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config,
            },
        ),
    },
    reset_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
        )
    },
    step_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
        )
    },
)

g1_29dof_wbt_command_w_object = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config_w_object,
            },
        )
    },
)

# Multi-clip object tracking; set motion_dir through the CLI.
g1_29dof_wbt_command_w_object_multitask = replace(
    g1_29dof_wbt_command,
    setup_terms={
        "motion_command": CommandTermCfg(
            func="holosoma.managers.command.terms.wbt:MotionCommand",
            params={
                "motion_config": motion_config_w_object_multitask,
            },
        )
    },
)

def _with_command_func(base: CommandManagerCfg, func: str) -> CommandManagerCfg:
    """Re-point every term of a command preset at another MotionCommand subclass.

    The three term groups must name the SAME class: they are three phases
    (setup/reset/step) of one command term, and the manager resolves each group
    independently.
    """
    return replace(
        base,
        setup_terms={
            name: replace(cfg, func=func) for name, cfg in base.setup_terms.items()
        },
        reset_terms={
            name: replace(cfg, func=func) for name, cfg in base.reset_terms.items()
        },
        step_terms={
            name: replace(cfg, func=func) for name, cfg in base.step_terms.items()
        },
    )


_TIME_WARP_FUNC = "holosoma.managers.command.terms.wbt_playback_aug:TimeWarpMotionCommand"
_HALT_AUG_FUNC = "holosoma.managers.command.terms.wbt_playback_aug:HaltAugMotionCommand"

# Playback-speed augmentation presets.  Both are gated by their own environment
# variables (WBT_MOTION_TIME_WARP / WBT_MOTION_HALT_PROB) and are identical to
# the base preset when those are unset, so they are safe drop-in replacements.
g1_29dof_wbt_command_timewarp = _with_command_func(g1_29dof_wbt_command, _TIME_WARP_FUNC)
g1_29dof_wbt_command_haltaug = _with_command_func(g1_29dof_wbt_command, _HALT_AUG_FUNC)
g1_29dof_wbt_command_w_object_haltaug = _with_command_func(g1_29dof_wbt_command_w_object, _HALT_AUG_FUNC)

__all__ = [
    "g1_29dof_wbt_command",
    "g1_29dof_wbt_command_w_object",
    "g1_29dof_wbt_command_w_object_multitask",
    "g1_29dof_wbt_command_timewarp",
    "g1_29dof_wbt_command_haltaug",
    "g1_29dof_wbt_command_w_object_haltaug",
]
