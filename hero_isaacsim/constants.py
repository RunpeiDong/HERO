"""Joint and body orders, end-effector geometry, and motor constants for HERO.

This module has no Torch or holosoma dependencies."""

from __future__ import annotations

# --------------------------------------------------------------------------------------------------
# Joints (29 DoF, holosoma / MuJoCo order)
# --------------------------------------------------------------------------------------------------
DOF_NAMES: tuple[str, ...] = (
    # left leg (6)
    "left_hip_pitch_joint",
    "left_hip_roll_joint",
    "left_hip_yaw_joint",
    "left_knee_joint",
    "left_ankle_pitch_joint",
    "left_ankle_roll_joint",
    # right leg (6)
    "right_hip_pitch_joint",
    "right_hip_roll_joint",
    "right_hip_yaw_joint",
    "right_knee_joint",
    "right_ankle_pitch_joint",
    "right_ankle_roll_joint",
    # waist (3)
    "waist_yaw_joint",
    "waist_roll_joint",
    "waist_pitch_joint",
    # left arm (7)
    "left_shoulder_pitch_joint",
    "left_shoulder_roll_joint",
    "left_shoulder_yaw_joint",
    "left_elbow_joint",
    "left_wrist_roll_joint",
    "left_wrist_pitch_joint",
    "left_wrist_yaw_joint",
    # right arm (7)
    "right_shoulder_pitch_joint",
    "right_shoulder_roll_joint",
    "right_shoulder_yaw_joint",
    "right_elbow_joint",
    "right_wrist_roll_joint",
    "right_wrist_pitch_joint",
    "right_wrist_yaw_joint",
)
NUM_DOF: int = len(DOF_NAMES)  # 29

# HERO dual-actor split: lower actor = legs + waist (15), upper actor = arms (14).
LEG_DOF_IDX: tuple[int, ...] = tuple(range(0, 12))
WAIST_DOF_IDX: tuple[int, ...] = (12, 13, 14)
LOWER_DOF_IDX: tuple[int, ...] = tuple(range(0, 15))
UPPER_DOF_IDX: tuple[int, ...] = tuple(range(15, 29))
LEFT_ARM_DOF_IDX: tuple[int, ...] = tuple(range(15, 22))
RIGHT_ARM_DOF_IDX: tuple[int, ...] = tuple(range(22, 29))
ARM_DOF_IDX: tuple[int, ...] = LEFT_ARM_DOF_IDX + RIGHT_ARM_DOF_IDX
# ``ref_upper_dof_pos`` (HeroMotionCommand): waist 3 + arms 14 = 17, holosoma order.
UPPER_REF_DOF_IDX: tuple[int, ...] = WAIST_DOF_IDX + ARM_DOF_IDX

NUM_LOWER_DOF: int = len(LOWER_DOF_IDX)  # 15
NUM_UPPER_DOF: int = len(UPPER_DOF_IDX)  # 14


HOLOSOMA_BODY_NAMES_32: tuple[str, ...] = (
    "pelvis",
    "left_hip_pitch_link",
    "left_hip_roll_link",
    "left_hip_yaw_link",
    "left_knee_link",
    "left_ankle_pitch_link",
    "left_ankle_roll_link",
    "left_foot_contact_point",
    "right_hip_pitch_link",
    "right_hip_roll_link",
    "right_hip_yaw_link",
    "right_knee_link",
    "right_ankle_pitch_link",
    "right_ankle_roll_link",
    "right_foot_contact_point",
    "waist_yaw_link",
    "waist_roll_link",
    "torso_link",
    "left_shoulder_pitch_link",
    "left_shoulder_roll_link",
    "left_shoulder_yaw_link",
    "left_elbow_link",
    "left_wrist_roll_link",
    "left_wrist_pitch_link",
    "left_wrist_yaw_link",
    "right_shoulder_pitch_link",
    "right_shoulder_roll_link",
    "right_shoulder_yaw_link",
    "right_elbow_link",
    "right_wrist_roll_link",
    "right_wrist_pitch_link",
    "right_wrist_yaw_link",
)

# Dex3 palm links (fixed to wrist_yaw with dont_collapse="true" in g1_29dof_dex3fixed_hero.urdf).
# They are extra rigid bodies in Isaac Sim (34-body preset) but are NEVER tracked from npz
# ``body_pos_w``; the palm EE reference comes from the extended npz keys ``ee_pos_pelvis*``.
PALM_BODY_NAMES: tuple[str, ...] = ("left_hand_palm_link", "right_hand_palm_link")

# 34-body list of the ``g1_29dof_dex3_m12`` preset (32 canonical + palms appended). holosoma's
# IsaacSim backend re-indexes bodies to this list (find_bodies(preserve_order=True)), preserving the order
# after fixed-joint merging.
HERO_BODY_NAMES_34: tuple[str, ...] = HOLOSOMA_BODY_NAMES_32 + PALM_BODY_NAMES

# HERO end-effector bodies (the wrist_yaw links, [left, right]); the tracked EE *point* is the
# Dex3 palm origin expressed in the wrist_yaw frame (``PALM_OFFSET``), i.e. the origin of
# ``*_hand_palm_joint`` in the robot URDF.
EE_BODY_NAMES: tuple[str, ...] = ("left_wrist_yaw_link", "right_wrist_yaw_link")
PALM_OFFSET: dict[str, tuple[float, float, float]] = {
    "left": (0.0415, 0.003, 0.0),
    "right": (0.0415, -0.003, 0.0),
}
EE_SIDES: tuple[str, ...] = ("left", "right")

# Foot contact points: fixed children of *_ankle_roll_link.
FOOT_CONTACT_POINT_BODY_NAMES: tuple[str, ...] = ("left_foot_contact_point", "right_foot_contact_point")
FOOT_CONTACT_POINT_PARENTS: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link")
FOOT_CONTACT_POINT_OFFSET: tuple[float, float, float] = (0.0, 0.0, -0.037)

# Other named bodies used by env / terminations.
PELVIS_BODY_NAME: str = "pelvis"
TORSO_BODY_NAME: str = "torso_link"
KNEE_BODY_NAMES: tuple[str, ...] = ("left_knee_link", "right_knee_link")
ANKLE_BODY_NAMES: tuple[str, ...] = ("left_ankle_roll_link", "right_ankle_roll_link")


ROBOT_PRESET_NAME: str = "g1_29dof_dex3_m12"
URDF_FILE_NAME: str = "g1_29dof_dex3fixed_hero.urdf"
MJCF_SCENE_FILE_NAME: str = "scene_g1_29dof_freebase_fixed_dex3.xml"

MOTOR_7520_22: dict[str, float] = {
    "stiffness": 99.098427777,
    "damping": 6.308801854,
    "armature": 0.025101925,
    "effort_limit": 139.0,
    "velocity_limit": 20.0,
}
MOTOR_7520_14: dict[str, float] = {
    "stiffness": 40.179238471,
    "damping": 2.557889765,
    "armature": 0.010177520,
    "effort_limit": 88.0,
    "velocity_limit": 32.0,
}

ACTION_SCALE: float = 0.25  # per-joint scale = 0.25 * effort_limit / stiffness (holosoma flag)
INIT_POS_Z: float = 0.76

# --------------------------------------------------------------------------------------------------
# Sanity checks (fail fast at import)
# --------------------------------------------------------------------------------------------------
assert NUM_DOF == 29
assert len(set(DOF_NAMES)) == NUM_DOF
assert LOWER_DOF_IDX + UPPER_DOF_IDX == tuple(range(NUM_DOF))
assert len(UPPER_REF_DOF_IDX) == 17
assert len(HOLOSOMA_BODY_NAMES_32) == 32 and len(set(HOLOSOMA_BODY_NAMES_32)) == 32
assert len(HERO_BODY_NAMES_34) == 34 and len(set(HERO_BODY_NAMES_34)) == 34
assert all(n in HOLOSOMA_BODY_NAMES_32 for n in EE_BODY_NAMES + FOOT_CONTACT_POINT_BODY_NAMES)
assert all(DOF_NAMES[i].startswith("waist_") for i in WAIST_DOF_IDX)
assert all("shoulder" in DOF_NAMES[i] or "elbow" in DOF_NAMES[i] or "wrist" in DOF_NAMES[i] for i in ARM_DOF_IDX)

__all__ = [
    "ACTION_SCALE",
    "ANKLE_BODY_NAMES",
    "ARM_DOF_IDX",
    "DOF_NAMES",
    "EE_BODY_NAMES",
    "EE_SIDES",
    "FOOT_CONTACT_POINT_BODY_NAMES",
    "FOOT_CONTACT_POINT_OFFSET",
    "FOOT_CONTACT_POINT_PARENTS",
    "HERO_BODY_NAMES_34",
    "HOLOSOMA_BODY_NAMES_32",
    "INIT_POS_Z",
    "KNEE_BODY_NAMES",
    "LEFT_ARM_DOF_IDX",
    "LEG_DOF_IDX",
    "LOWER_DOF_IDX",
    "MJCF_SCENE_FILE_NAME",
    "MOTOR_7520_14",
    "MOTOR_7520_22",
    "NUM_DOF",
    "NUM_LOWER_DOF",
    "NUM_UPPER_DOF",
    "PALM_BODY_NAMES",
    "PALM_OFFSET",
    "PELVIS_BODY_NAME",
    "RIGHT_ARM_DOF_IDX",
    "ROBOT_PRESET_NAME",
    "TORSO_BODY_NAME",
    "UPPER_DOF_IDX",
    "UPPER_REF_DOF_IDX",
    "URDF_FILE_NAME",
    "WAIST_DOF_IDX",
]
