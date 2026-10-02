"""HERO actor and critic observation configuration."""
from __future__ import annotations

import math

from dataclasses import replace

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg

HERO_OBS_MODULE = "hero_isaacsim.managers.observation.hero"

WBT_OBS_MODULE = "holosoma.managers.observation.terms.wbt"

ACTOR_GROUP = "actor_obs"

CRITIC_GROUP = "critic_obs"

ACTOR_HISTORY_LENGTH = 5

CRITIC_HISTORY_LENGTH = 1

HERO_TERM_DIMS: dict[str, int] = {
    # --- HERO actor (14 terms, 135 dims) ---
    "h00_actions": 29,
    "h01_base_ang_vel": 3,
    "h02_command_ang_vel": 1,
    "h03_command_base_height": 1,
    "h04_command_lin_vel": 2,
    "h05_command_stand": 1,
    "h06_command_waist_dofs": 3,
    "h07_dif_local_rigid_body_pos_ee": 6,
    "h08_dif_local_rigid_body_rot_ee": 12,
    "h09_dof_pos": 29,
    "h10_dof_vel": 29,
    "h11_projected_gravity": 3,
    "h12_ref_upper_dof_pos": 14,
    "h13_roll_and_pitch": 2,
    # Legacy term widths retained for reading exported model metadata.
    "h14_ref_lower_dof_pos": 12,
    "h15_ref_root_pitch_roll": 2,
    "h16_ref_body_pos_b": 12,  # ankles + knees relative to pelvis (4 bodies x 3)

    "h17_obj_pos_b": 3,
    "h18_obj_ori_b": 6,
    "h19_has_object_flag": 1,

    "h20_ref_root_pose_b": 20,


    "h21_ref_root_rot_b": 30,
    "h22_ref_root_height_b": 5,
    "h23_base_lin_vel_odom": 3,
    # --- critic-only HERO terms (sort right after h01_base_ang_vel: '_' < 'a' < 'b') ---
    "h01a_base_lin_vel": 3,
    "h01b_base_orientation": 4,
    # --- privileged critic terms reused verbatim from holosoma wbt (14 tracked bodies) ---
    "p00_robot_body_pos_b": 42,
    "p01_robot_body_ori_b": 84,
    "p02_motion_ref_ori_b": 6,
}

HERO_TERM_SCALES: dict[str, float] = {
    "h01_base_ang_vel": 0.25,
    "h03_command_base_height": 2.0,
    "h10_dof_vel": 0.05,
    "h01a_base_lin_vel": 2.0,
}

H1_ACTOR_TERM_NAMES: tuple[str, ...] = (
    "h00_actions",
    "h01_base_ang_vel",
    "h02_command_ang_vel",
    "h03_command_base_height",
    "h04_command_lin_vel",
    "h05_command_stand",
    "h06_command_waist_dofs",
    "h07_dif_local_rigid_body_pos_ee",
    "h08_dif_local_rigid_body_rot_ee",
    "h09_dof_pos",
    "h10_dof_vel",
    "h11_projected_gravity",
    "h12_ref_upper_dof_pos",
    "h13_roll_and_pitch",
)




ANCHOR_TERM_NAME = "h20_ref_root_pose_b"

H20_FUTURE_STEPS: tuple[int, ...] = (0, 5, 10, 15, 20)

H20_FRAME_DIM = 4

HERO_ANCHOR_ODOM_NOISE: dict[str, float] = {
    "noise_odom_bias_xy_m": 0.02,
    "noise_odom_bias_yaw_rad": math.radians(1.0),
    "noise_odom_walk_xy_m": 0.005,
    "noise_odom_walk_yaw_rad": math.radians(0.2),
}

ANCHOR2_TERM_NAMES: tuple[str, ...] = ("h21_ref_root_rot_b", "h22_ref_root_height_b")

ANCHOR2_VELOCITY_TERM_NAME = "h23_base_lin_vel_odom"

ANCHOR2V_TERM_NAMES: tuple[str, ...] = (*ANCHOR2_TERM_NAMES, ANCHOR2_VELOCITY_TERM_NAME)

H21_FRAME_DIM = 6  # column-major 6D rotation per future frame (torch-free mirror of managers.observation.hero.H21_FRAME_DIM)

H22_FRAME_DIM = 1  # z_ref - z_robot per future frame

ANCHOR2_FUTURE_TERM_FRAME_DIMS: dict[str, int] = {"h21_ref_root_rot_b": H21_FRAME_DIM, "h22_ref_root_height_b": H22_FRAME_DIM}

HERO_ANCHOR2_ODOM_NOISE: dict[str, dict[str, float]] = {
    "h21_ref_root_rot_b": {
        "noise_odom_bias_roll_pitch_rad": math.radians(0.5),
        "noise_odom_bias_yaw_rad": HERO_ANCHOR_ODOM_NOISE["noise_odom_bias_yaw_rad"],
        "noise_odom_walk_yaw_rad": HERO_ANCHOR_ODOM_NOISE["noise_odom_walk_yaw_rad"],
    },
    "h22_ref_root_height_b": {"noise_odom_bias_z_m": 0.01, "noise_odom_walk_z_m": 0.002},
    "h23_base_lin_vel_odom": {"noise_odom_vel_m_s": 0.05},
}

ANCHOR_SHARED_YAW_NOISE_KEYS: tuple[str, ...] = ("noise_odom_bias_yaw_rad", "noise_odom_walk_yaw_rad")

CRITIC_EXTRA_TERM_NAMES: tuple[str, ...] = ("h01a_base_lin_vel", "h01b_base_orientation")

PRIVILEGED_TERM_NAMES: tuple[str, ...] = ("p00_robot_body_pos_b", "p01_robot_body_ori_b")


_PRIVILEGED_FUNCS: dict[str, str] = {
    "p00_robot_body_pos_b": f"{WBT_OBS_MODULE}:robot_body_pos_b",
    "p01_robot_body_ori_b": f"{WBT_OBS_MODULE}:robot_body_ori_b",
    "p02_motion_ref_ori_b": f"{WBT_OBS_MODULE}:motion_ref_ori_b",
}

def hero_term(name: str, **overrides) -> ObsTermCfg:
    """Build an ``ObsTermCfg`` for a HERO term (func == term name inside the hero module)."""
    if name not in HERO_TERM_DIMS:
        raise KeyError(f"unknown HERO observation term {name!r}")
    func = _PRIVILEGED_FUNCS.get(name, f"{HERO_OBS_MODULE}:{name}")
    kwargs = {"func": func, "scale": HERO_TERM_SCALES.get(name, 1.0), "noise": 0.0}
    kwargs.update(overrides)
    return ObsTermCfg(**kwargs)

def _terms(names) -> dict[str, ObsTermCfg]:
    return {n: hero_term(n) for n in names}

def make_actor_group() -> ObsGroupCfg:
    """Paper actor inputs with five frames of history."""
    return ObsGroupCfg(concatenate=True, enable_noise=False,
                       history_length=ACTOR_HISTORY_LENGTH, terms=_terms(H1_ACTOR_TERM_NAMES))

def make_critic_group() -> ObsGroupCfg:
    """Paper critic inputs, including privileged simulator state."""
    names = (*H1_ACTOR_TERM_NAMES, *CRITIC_EXTRA_TERM_NAMES, *PRIVILEGED_TERM_NAMES)
    return ObsGroupCfg(concatenate=True, enable_noise=False,
                       history_length=CRITIC_HISTORY_LENGTH, terms=_terms(names))


def make_hero_observation() -> ObservationManagerCfg:
    """Build the paper observation layout before optional anchor augmentation."""
    return ObservationManagerCfg(groups={ACTOR_GROUP: make_actor_group(),
                                        CRITIC_GROUP: make_critic_group()}, clip_observations=100.0)

def sorted_layout(group: ObsGroupCfg) -> list[tuple[str, int]]:
    """``[(term_name, single_frame_dim)]`` in holosoma concatenation (alphabetical) order."""
    return [(name, HERO_TERM_DIMS[name]) for name in sorted(group.terms)]

def group_dim(group: ObsGroupCfg) -> int:
    """Flattened group width including history (== observation_manager.get_obs_dims())."""
    return sum(d for _, d in sorted_layout(group)) * group.history_length

def noise_params(cfg: ObservationManagerCfg, group: str = ACTOR_GROUP) -> dict[str, dict[str, float]]:
    """``{term: {noise param: value}}`` of the training-only noise carried by ``group`` (empty when disabled)."""
    out: dict[str, dict[str, float]] = {}
    for name, term in cfg.groups[group].terms.items():
        found = {k: v for k, v in term.params.items() if k.startswith("noise")}
        if found:
            out[name] = found
    return out

def with_anchor_term(
    cfg: ObservationManagerCfg,
    future_steps=H20_FUTURE_STEPS,
    odometry_noise: dict[str, float] | None = None,
    groups=(ACTOR_GROUP, CRITIC_GROUP),
    noise_groups=(ACTOR_GROUP,),
) -> ObservationManagerCfg:


    steps = [int(s) for s in future_steps]
    if len(steps) * H20_FRAME_DIM != HERO_TERM_DIMS[ANCHOR_TERM_NAME] or any(s < 0 for s in steps):
        raise ValueError(f"future_steps {steps} must be {HERO_TERM_DIMS[ANCHOR_TERM_NAME] // H20_FRAME_DIM} ints >= 0")
    noise = dict(HERO_ANCHOR_ODOM_NOISE if odometry_noise is None else odometry_noise)
    if any(not str(k).startswith("noise") for k in noise):
        raise ValueError(f"odometry noise params must start with 'noise' (provenance / noise_params contract): {sorted(noise)}")
    new_groups = dict(cfg.groups)
    for g in groups:
        group = cfg.groups[g]
        if ANCHOR_TERM_NAME in group.terms:
            raise ValueError(f"group {g!r} already carries {ANCHOR_TERM_NAME}")
        params: dict = {"future_steps": list(steps)}
        if g in noise_groups:
            params.update(noise)
        new_groups[g] = replace(group, terms={**group.terms, ANCHOR_TERM_NAME: hero_term(ANCHOR_TERM_NAME, params=params)})
    return replace(cfg, groups=new_groups)

def with_anchor2_terms(
    cfg: ObservationManagerCfg,
    future_steps=H20_FUTURE_STEPS,
    odometry_noise: dict[str, dict[str, float]] | None = None,
    groups=(ACTOR_GROUP, CRITIC_GROUP),
    noise_groups=(ACTOR_GROUP,),
    include_velocity: bool = False,
) -> ObservationManagerCfg:


    names = ANCHOR2V_TERM_NAMES if include_velocity else ANCHOR2_TERM_NAMES
    steps = [int(s) for s in future_steps]
    for name, fd in ANCHOR2_FUTURE_TERM_FRAME_DIMS.items():
        if len(steps) * fd != HERO_TERM_DIMS[name] or any(s < 0 for s in steps):
            raise ValueError(f"future_steps {steps} must be {HERO_TERM_DIMS[name] // fd} ints >= 0 ({name})")
    table = HERO_ANCHOR2_ODOM_NOISE if odometry_noise is None else odometry_noise
    noise = {str(t): {str(k): float(v) for k, v in dict(p).items()} for t, p in dict(table).items()}
    if any(t not in ANCHOR2V_TERM_NAMES for t in noise):
        raise ValueError(f"odometry noise names unknown anchor2 terms {sorted(set(noise) - set(ANCHOR2V_TERM_NAMES))}")
    if any(not k.startswith("noise") for p in noise.values() for k in p):
        raise ValueError(f"odometry noise params must start with 'noise' (provenance / noise_params contract): {noise}")
    new_groups = dict(cfg.groups)
    for g in groups:
        group = cfg.groups[g]
        h20 = group.terms.get(ANCHOR_TERM_NAME)
        if h20 is None:
            raise ValueError(f"group {g!r} has no {ANCHOR_TERM_NAME}: the anchor2 terms extend an anchor layout")
        if [int(s) for s in h20.params.get("future_steps", H20_FUTURE_STEPS)] != steps:
            raise ValueError(f"group {g!r}: h20 future_steps {h20.params.get('future_steps')} != anchor2 future_steps {steps}")
        present = [t for t in ANCHOR2V_TERM_NAMES if t in group.terms]
        if present:
            raise ValueError(f"group {g!r} already carries {present}")
        terms = dict(group.terms)
        for name in names:
            params: dict = {"future_steps": list(steps)} if name in ANCHOR2_FUTURE_TERM_FRAME_DIMS else {}
            if g in noise_groups:
                params.update(noise.get(name, {}))
            terms[name] = hero_term(name, params=params)
        h21 = terms[names[0]]
        for key in ANCHOR_SHARED_YAW_NOISE_KEYS:
            if abs(float(h20.params.get(key, 0.0)) - float(h21.params.get(key, 0.0))) > 1e-12:
                raise ValueError(
                    f"group {g!r}: h21 {key}={h21.params.get(key, 0.0)} != h20 {key}={h20.params.get(key, 0.0)} "
                    "(h20 and h21 share one odometry yaw channel; pass matching odometry_noise)"
                )
        new_groups[g] = replace(group, terms=terms)
    return replace(cfg, groups=new_groups)

DELTA_EE_TERM_NAMES: tuple[str, ...] = ("h07_dif_local_rigid_body_pos_ee", "h08_dif_local_rigid_body_rot_ee")

def without_delta_ee_terms(cfg: ObservationManagerCfg, groups=(ACTOR_GROUP,)) -> ObservationManagerCfg:
    """Drop HERO's residual end-effector feedback (h07 palm-position / h08 palm-rotation errors) from ``groups``.

    Everything else is kept: the remaining terms, their history length and the critic's privileged state. By default only
    the actor loses the feedback (asymmetric critic), so the value function still sees the tracking error it has to credit;
    pass ``groups=(ACTOR_GROUP, CRITIC_GROUP)`` to remove it from both."""
    new_groups = dict(cfg.groups)
    for g in groups:
        group = cfg.groups[g]
        missing = [t for t in DELTA_EE_TERM_NAMES if t not in group.terms]
        if missing:
            raise ValueError(f"group {g!r} has no {missing}: the delta-EE ablation removes the feedback of an existing layout")
        new_groups[g] = replace(group, terms={k: v for k, v in group.terms.items() if k not in DELTA_EE_TERM_NAMES})
    return replace(cfg, groups=new_groups)

def term_columns(group: ObsGroupCfg, name: str) -> list[int]:
    """Flat column indices of ``name`` in the group's TERM-MAJOR training layout (holosoma: every term's
    ``history_length`` frames are flattened first, terms concatenated in sorted order) -- one contiguous block
    ``[H * offset, H * (offset + dim))``.  The frame-major export layout (``agents/ppo_dual/layout.py``) is a permutation of them."""
    layout = sorted_layout(group)
    names = [n for n, _ in layout]
    if name not in names:
        raise KeyError(f"{name!r} not in group terms {names}")
    h = int(group.history_length)
    offset = sum(d for n, d in layout[: names.index(name)])
    dim = HERO_TERM_DIMS[name]
    return list(range(h * offset, h * (offset + dim)))

hero_h1_observation = make_hero_observation()
DEFAULTS = {"hero": hero_h1_observation}
