"""HERO reward configuration."""
from __future__ import annotations

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg

HERO_REWARD_MODULE = "hero_isaacsim.managers.reward.hero"

LOWER_BODY = "lower_body"

UPPER_BODY = "upper_body"

PENALTY_CURRICULUM = "penalty_curriculum"

SIGMA = {
    "lin_vel": 0.25,
    "ang_vel": 0.25,
    "base_height": 0.05,
    "waist_dofs": 0.05,
    "upper_body_dofs": 0.1,
    "ee_rot": 0.10,
    "ee_pos": 0.05,
}

SOFT_DOF_POS_LIMIT = 0.95

SOFT_TORQUE_LIMIT = 0.95

CLOSE_FEET_THRESHOLD = 0.17

FEET_HEIGHT_TARGET = 0.11

FEET_HEIGHT_STAND = 0.025

STANCE_BASE_HEIGHT_PENALTY_SCALE = 5.0

PENALTY_CURRICULUM_NAMES: tuple[str, ...] = (
    "penalty_upper_body_torques",
    "penalty_upper_body_dof_acc",
    "penalty_upper_body_dof_vel",
    "penalty_upper_body_action_rate",
    "penalty_lower_body_torques",
    "penalty_lower_body_dof_acc",
    "penalty_lower_body_dof_vel",
    "penalty_lower_body_action_rate",
    "limits_upper_body_dof_pos",
    "limits_upper_body_dof_vel",
    "limits_upper_body_torque",
    "limits_lower_body_dof_pos",
)

_UPPER_BODY_TERMS: list[tuple[str, float, dict]] = [
    ("tracking_upper_body_dofs", 4.0, {"sigma": SIGMA["upper_body_dofs"]}),
    ("penalty_upper_body_torques", -1.0e-05, {}),
    ("penalty_upper_body_dof_vel", -0.001, {}),
    ("penalty_upper_body_dof_acc", -2.5e-07, {}),
    ("penalty_upper_body_action_rate", -0.1, {}),
    ("limits_upper_body_dof_pos", -5.0, {"soft_dof_pos_limit": SOFT_DOF_POS_LIMIT}),
    ("limits_upper_body_dof_vel", -5.0, {}),  # Use raw URDF velocity limits without a soft factor.
    ("limits_upper_body_torque", -0.1, {"soft_torque_limit": SOFT_TORQUE_LIMIT}),


    ("penalty_ee_lin_acc", -0.2, {}),
    ("penalty_ee_ang_acc", -0.02, {}),
    ("tracking_stance_ee_pos", 2.0, {"sigma": SIGMA["ee_pos"], "stance_only": True}),
    ("tracking_stance_ee_rot", 2.0, {"sigma": SIGMA["ee_rot"], "stance_only": True}),
]

_LOWER_BODY_TERMS: list[tuple[str, float, dict]] = [
    ("tracking_lin_vel_x", 2.0, {"sigma": SIGMA["lin_vel"]}),
    ("tracking_lin_vel_y", 1.5, {"sigma": SIGMA["lin_vel"]}),
    ("tracking_ang_vel", 4.0, {"sigma": SIGMA["ang_vel"]}),
    ("tracking_walk_base_height", 1.0, {"sigma": SIGMA["base_height"]}),
    ("tracking_stance_base_height", 4.0, {"sigma": SIGMA["base_height"]}),
    ("tracking_waist_dofs_tapping", 0.5, {"sigma": SIGMA["waist_dofs"]}),
    ("tracking_waist_dofs_stance", 3.0, {"sigma": SIGMA["waist_dofs"]}),
    ("penalty_lin_vel_z", -2.0, {"walk_only": True, "relative_to_reference": False}),
    ("penalty_ang_vel_xy", -0.05, {}),
    ("penalty_orientation", -1.5, {"relative_to_reference": False}),
    ("penalty_torso_orientation", -1.0, {"relative_to_reference": False}),
    ("penalty_lower_body_torques", -1.0e-05, {}),
    ("penalty_lower_body_dof_vel", -0.001, {}),
    ("penalty_lower_body_dof_acc", -2.5e-07, {}),
    ("penalty_lower_body_action_rate", -0.1, {}),
    ("penalty_contact_no_vel", -0.2, {}),
    ("penalty_feet_ori", -2.0, {}),
    ("limits_lower_body_dof_pos", -5.0, {"soft_dof_pos_limit": SOFT_DOF_POS_LIMIT}),
    ("feet_air_time", 4.0, {}),
    ("base_height", -10.0, {"stance_penalty_scale": STANCE_BASE_HEIGHT_PENALTY_SCALE}),
    ("termination", -250.0, {}),
    ("penalty_feet_height", -5.0, {"target": FEET_HEIGHT_TARGET}),
    (
        "penalty_feet_swing_height",
        -20.0,
        {"target_walk": FEET_HEIGHT_TARGET, "target_stance": FEET_HEIGHT_STAND},
    ),
    ("feet_heading_alignment", -0.25, {}),
    ("penalty_close_feet_xy", -10.0, {"close_feet_threshold": CLOSE_FEET_THRESHOLD}),
    ("penalty_hip_pos", -2.5, {}),
    ("penalty_ang_vel_xy_torso", -1.0, {}),
    ("penalty_stance_dof", -0.001, {}),
    ("penalty_stance_tap_feet", -5.0, {}),
    ("penalty_stance_root", -5.0, {}),
    ("penalty_stand_still", -0.15, {}),
    ("penalty_stance_symmetry", -0.5, {}),
    ("penalty_ankle_roll", -2.0, {}),
    ("penalty_contact", -4.0, {}),
    ("penalty_negative_knee_joint", -1.0, {}),
    ("penalty_diff_feet_air_time", -5.0, {}),
    ("penalty_shift_in_zero_command", -1.5, {}),
    ("penalty_ang_shift_in_zero_command", -1.5, {}),
]

def _term(name: str, weight: float, params: dict, group: str) -> RewardTermCfg:
    tags = [group]
    if name in PENALTY_CURRICULUM_NAMES:
        tags.append(PENALTY_CURRICULUM)
    return RewardTermCfg(func=f"{HERO_REWARD_MODULE}:{name}", params=dict(params), weight=weight, tags=tags)

def build_hero_terms() -> dict[str, RewardTermCfg]:
    terms: dict[str, RewardTermCfg] = {}
    for name, w, p in _LOWER_BODY_TERMS:
        terms[name] = _term(name, w, p, LOWER_BODY)
    for name, w, p in _UPPER_BODY_TERMS:
        terms[name] = _term(name, w, p, UPPER_BODY)
    return terms

hero_109_curr_reward = RewardManagerCfg(terms=build_hero_terms(), only_positive_rewards=False)
DEFAULTS = {"hero": hero_109_curr_reward}
