"""Whole Body Tracking termination presets for the G1 robot."""

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg

g1_29dof_wbt_termination = TerminationManagerCfg(
    terms={
        "timeout": TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded",
            is_timeout=True,
        ),
        "bad_tracking": TerminationTermCfg(
            func="holosoma.managers.termination.terms.wbt:BadTrackingZOnly",
            params={
                # robot tracking
                "bad_ref_pos_threshold": 0.5,
                "bad_ref_ori_threshold": 0.8,
                "bad_motion_body_pos_threshold": 0.25,
                # NOTE: body_names_to_track is shared with command_manager
                "body_names_to_track": [
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
                "bad_motion_body_pos_body_names": [
                    "left_ankle_roll_link",
                    "right_ankle_roll_link",
                    "left_wrist_yaw_link",
                    "right_wrist_yaw_link",
                ],
                # object tracking
                # only triggered when has_object=True
                "bad_object_pos_threshold": 0.25,
                "bad_object_ori_threshold": 0.8,
            },
        ),
    }
)

# Relax body and object tracking thresholds to allow longer episodes.


import dataclasses as _dc

g1_29dof_wbt_termination_relaxed = TerminationManagerCfg(
    terms={
        "timeout": g1_29dof_wbt_termination.terms["timeout"],
        "bad_tracking": _dc.replace(
            g1_29dof_wbt_termination.terms["bad_tracking"],
            params={
                **g1_29dof_wbt_termination.terms["bad_tracking"].params,
                "bad_ref_pos_threshold": 1.0,
                "bad_ref_ori_threshold": 1.2,
                "bad_motion_body_pos_threshold": 0.5,
                "bad_object_pos_threshold": 1.0,
                "bad_object_ori_threshold": 1.2,
            },
        ),
    }
)

# Evaluate object termination in the robot-relative reference frame.
# Pair with g1_29dof_wbt_reward_w_object_retgt.


g1_29dof_wbt_termination_retgtobj = TerminationManagerCfg(
    terms={
        "timeout": g1_29dof_wbt_termination.terms["timeout"],
        "bad_tracking": _dc.replace(
            g1_29dof_wbt_termination.terms["bad_tracking"],
            params={
                **g1_29dof_wbt_termination.terms["bad_tracking"].params,
                "retargeted_object": True,
            },
        ),
    }
)

# Latch object termination on after body tracking first meets its thresholds.
# Uses the relaxed termination distances and angles.


g1_29dof_wbt_termination_gated = TerminationManagerCfg(
    terms={
        "timeout": g1_29dof_wbt_termination.terms["timeout"],
        "bad_tracking": _dc.replace(
            g1_29dof_wbt_termination_relaxed.terms["bad_tracking"],
            params={
                **g1_29dof_wbt_termination_relaxed.terms["bad_tracking"].params,
                "object_term_requires_body_ok": True,
            },
        ),
    }
)

__all__ = [
    "g1_29dof_wbt_termination",
    "g1_29dof_wbt_termination_relaxed",
    "g1_29dof_wbt_termination_retgtobj",
    "g1_29dof_wbt_termination_gated",
]
