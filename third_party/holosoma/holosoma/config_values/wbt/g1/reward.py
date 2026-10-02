"""Whole Body Tracking reward presets for the G1 robot."""

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg

g1_29dof_wbt_reward = RewardManagerCfg(
    terms={
        # Motion tracking rewards - global reference frame
        "motion_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=0.5,
        ),
        "motion_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=0.5,
        ),
        # Motion tracking rewards - relative body frame
        "motion_relative_body_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "motion_relative_body_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
        # Motion tracking rewards - body velocities
        "motion_global_body_lin_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_lin_vel",
            params={"sigma": 1.0},
            weight=1.0,
        ),
        "motion_global_body_ang_vel": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_body_ang_vel",
            params={"sigma": 3.14},
            weight=1.0,
        ),
        # Regularization rewards
        "action_rate_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_action_rate",
            weight=-0.1,
        ),
        "limits_dof_pos": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:limits_dof_pos",
            params={"soft_dof_pos_limit": 0.9},
            weight=-10.0,
        ),
        "undesired_contacts": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:UndesiredContacts",
            params={
                "threshold": 1.0,
                "undesired_contacts_body_names": (
                    "^(?!left_foot_contact_point$)(?!right_foot_contact_point$)"
                    "(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$)"
                    "(?!left_ankle_roll_link$)(?!right_ankle_roll_link$).+$"
                ),
            },
            weight=-0.1,
        ),
    }
)

g1_29dof_wbt_fast_sac_reward = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward.terms,
        "action_rate_l2": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:penalty_action_rate",
            weight=-1.0,
        ),
        "motion_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "motion_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=0.5,
        ),
        # Motion tracking rewards - relative body frame
        "motion_relative_body_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_position_error_exp",
            params={"sigma": 0.3},
            weight=2.0,
        ),
        "motion_relative_body_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:motion_relative_body_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
    }
)

g1_29dof_wbt_reward_w_object = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward.terms,
        # Motion tracking rewards - global reference frame
        "object_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "object_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
    }
)

# Evaluate object tracking against the robot-relative reference.
# Pair with g1_29dof_wbt_termination_retgtobj.


g1_29dof_wbt_reward_w_object_retgt = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward.terms,
        "object_global_ref_position_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_relative_ref_position_error_exp",
            params={"sigma": 0.3},
            weight=1.0,
        ),
        "object_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_relative_ref_orientation_error_exp",
            params={"sigma": 0.4},
            weight=1.0,
        ),
    }
)

# General HOI contact-reward term, reusable across object-interaction tasks. Only the
# contactor_body_names regex changes per task (hands for carry, pelvis for sit, hand for wipe).
# Default: both hands (wrist+elbow) should contact the carried object.
hoi_hand_contact_term = RewardTermCfg(
    func="holosoma.managers.reward.terms.wbt:ObjectContactReward",
    params={
        "contactor_body_names": "^(left|right)_(wrist_yaw|elbow)_link$",
        "threshold": 5.0,
        "proximity": 0.35,
        "require_object_lifted": True,
        "lift_eps": 0.05,
    },
    weight=0.5,
)

# undesired_contacts override for the HOI-contact presets: the base term only exempts
# wrist/foot links, so the elbow — which hoi_hand_contact_term REWARDS (+0.5·frac) —
# was simultaneously PENALIZED (-0.1) by the blanket term, and torso/forearm-assisted
# hug-carry (the halfsphere hand carries via forearm squeeze) was punished too.  Exempt
# the full intended contact chain (wrist, elbow, torso) so the two terms cannot fight.
_hoi_undesired_contacts_term = RewardTermCfg(
    func="holosoma.managers.reward.terms.wbt:UndesiredContacts",
    params={
        "threshold": 1.0,
        "undesired_contacts_body_names": (
            "^(?!left_foot_contact_point$)(?!right_foot_contact_point$)"
            "(?!left_wrist_yaw_link$)(?!right_wrist_yaw_link$)"
            "(?!left_ankle_roll_link$)(?!right_ankle_roll_link$)"
            "(?!left_elbow_link$)(?!right_elbow_link$)"
            "(?!torso_link$).+$"
        ),
    },
    weight=-0.1,
)

# Reward preset that ADDS the general contact reward on top of the object tracking rewards.
g1_29dof_wbt_reward_w_object_contact = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward_w_object.terms,
        "object_hand_contact": hoi_hand_contact_term,
        "undesired_contacts": _hoi_undesired_contacts_term,
    }
)

# Increase object-orientation reward weight to 2.0, use sigma 0.3, and add hand-contact reward.


g1_29dof_wbt_reward_w_object_omnifix = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward_w_object.terms,
        "object_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_orientation_error_exp",
            params={"sigma": 0.3},
            weight=2.0,
        ),
        "object_hand_contact": hoi_hand_contact_term,
    }
)

# Anchor contact force and proximity rewards at the rubber-hand links.
# Requires a robot configuration that preserves the hand links as contact bodies.


hoi_rubberhand_contact_term = RewardTermCfg(
    func="holosoma.managers.reward.terms.wbt:ObjectContactReward",
    params={
        "contactor_body_names": "^(left|right)_rubber_hand$",
        "threshold": 5.0,
        "proximity": 0.22,
        "require_object_lifted": True,
        "lift_eps": 0.05,
    },
    weight=0.5,
)

# Hand-anchored contact reward preset (tracking rewards + rubber_hand contact). Pairs with the handbody robot.
g1_29dof_wbt_reward_w_object_handcontact = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward_w_object.terms,
        "object_hand_contact": hoi_rubberhand_contact_term,
    }
)

# Combine object-orientation and hand-anchored contact rewards.

g1_29dof_wbt_reward_w_object_omnifix_handcontact = RewardManagerCfg(
    terms={
        **g1_29dof_wbt_reward_w_object.terms,
        "object_global_ref_orientation_error_exp": RewardTermCfg(
            func="holosoma.managers.reward.terms.wbt:object_global_ref_orientation_error_exp",
            params={"sigma": 0.3},
            weight=2.0,
        ),
        "object_hand_contact": hoi_rubberhand_contact_term,
    }
)

__all__ = [
    "g1_29dof_wbt_fast_sac_reward",
    "g1_29dof_wbt_reward",
    "g1_29dof_wbt_reward_w_object",
    "g1_29dof_wbt_reward_w_object_retgt",
    "g1_29dof_wbt_reward_w_object_contact",
    "g1_29dof_wbt_reward_w_object_omnifix",
    "g1_29dof_wbt_reward_w_object_handcontact",
    "g1_29dof_wbt_reward_w_object_omnifix_handcontact",
    "hoi_hand_contact_term",
    "hoi_rubberhand_contact_term",
]
