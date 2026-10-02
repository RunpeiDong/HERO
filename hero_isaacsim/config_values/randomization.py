"""HERO domain randomization configuration."""
from __future__ import annotations

from dataclasses import replace

from holosoma.config_types.randomization import RandomizationManagerCfg, RandomizationTermCfg

from holosoma.config_values.wbt.g1.randomization import (
    base_reset_terms,
    base_setup_terms,
    base_step_terms,
)

from hero_isaacsim.constants import PALM_BODY_NAMES

LOCO_RAND_MODULE = "holosoma.managers.randomization.terms.locomotion"

HERO_RAND_MODULE = "hero_isaacsim.managers.randomization.hero"

EE_MASS_FUNC = f"{HERO_RAND_MODULE}:randomize_ee_mass_startup"

EE_MASS_TERM_NAME = "ee_mass_randomizer"

LINK_MASS_RANGE = [0.9, 1.2]

ADDED_BASE_MASS_RANGE = [-1.0, 3.0]

FRICTION_RANGE = [0.25, 1.25]

KP_RANGE = [0.9, 1.1]

KD_RANGE = [0.9, 1.1]

CTRL_DELAY_STEP_RANGE = [0, 1]

EE_BODY_NAMES = list(PALM_BODY_NAMES)

EE_MASS_SCALE_RANGE = [0.5, 3.0]

EE_ADDED_MASS_RANGE = [0.0, 1.5]

def make_ee_mass_term(
    ee_body_names=None,
    ee_mass_scale_range=None,
    ee_added_mass_range=None,
) -> RandomizationTermCfg:

    return RandomizationTermCfg(
        func=EE_MASS_FUNC,
        params={
            "ee_body_names": list(ee_body_names or EE_BODY_NAMES),
            "ee_mass_scale_range": list(ee_mass_scale_range or EE_MASS_SCALE_RANGE),
            "ee_added_mass_range": list(ee_added_mass_range or EE_ADDED_MASS_RANGE),
            "enable_scale": True,
            "enable_added_mass": True,
            "enabled": True,
        },
    )

ee_mass_term = make_ee_mass_term()

def _with_params(term: RandomizationTermCfg, **params) -> RandomizationTermCfg:
    return replace(term, params={**term.params, **params})

hero_setup_terms: dict[str, RandomizationTermCfg] = {
    **base_setup_terms,
    # kp/kd x U(0.9,1.1) (HERO randomize_pd_gain)
    "actuator_randomizer_state": _with_params(
        base_setup_terms["actuator_randomizer_state"], kp_range=KP_RANGE, kd_range=KD_RANGE, enable_pd_gain=True
    ),
    # control delay 0-1 policy steps (HERO randomize_ctrl_delay)
    "setup_action_delay_buffers": _with_params(
        base_setup_terms["setup_action_delay_buffers"], ctrl_delay_step_range=CTRL_DELAY_STEP_RANGE, enabled=True
    ),
    # friction U(0.25,1.25): one range for static and dynamic (HERO uses a single coefficient)
    "randomize_robot_rigid_body_material_startup": _with_params(
        base_setup_terms["randomize_robot_rigid_body_material_startup"],
        static_friction_range=FRICTION_RANGE,
        dynamic_friction_range=FRICTION_RANGE,
    ),

    EE_MASS_TERM_NAME: ee_mass_term,
    # link mass scale + base added mass (HERO randomize_link_mass / randomize_base_mass)
    "mass_randomizer": RandomizationTermCfg(
        func=f"{LOCO_RAND_MODULE}:randomize_mass_startup",
        params={
            "enable_link_mass": True,
            "link_mass_range": LINK_MASS_RANGE,
            "enable_base_mass": True,
            "added_mass_range": ADDED_BASE_MASS_RANGE,
            "enabled": True,
        },
    ),
}

hero_randomization = RandomizationManagerCfg(
    setup_terms=dict(hero_setup_terms),
    reset_terms=dict(base_reset_terms),
    step_terms=dict(base_step_terms),
)
DEFAULTS = {"hero": hero_randomization}
