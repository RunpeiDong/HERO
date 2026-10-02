"""HERO training configurations: the paper dual-actor presets with and without delta anchor, the without-delta-EE ablation, and their single-actor (``*_single``) flavours."""
from hero_isaacsim.config_values.experiment import (
    DEFAULTS, DELTA_ANCHOR_TERMS, DELTA_EE_TERMS, DUAL_ACTOR_CONFIGS, SINGLE_ACTOR_CONFIGS, make_hero_recipe, observation_contract,
    with_delta_anchor, with_delta_anchor_single, without_delta_anchor, without_delta_anchor_single, without_delta_ee,
    without_delta_ee_single,
)
