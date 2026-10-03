"""HERO motion command configuration."""
from __future__ import annotations

from hero_isaacsim.managers.command.hero import HeroMotionConfig, coerce_hero_motion_config

from dataclasses import field, fields, replace
from typing import Mapping

from holosoma.config_types.command import CommandManagerCfg, CommandTermCfg, MotionConfig, NoiseToInitialPoseConfig

from hero_isaacsim.config_values.sampler import ADAPTIVE_CLIP_MAX_RELATIVE, ADAPTIVE_UNIFORM_RATIO

HERO_COMMAND_FUNC = "hero_isaacsim.managers.command.hero:HeroMotionCommand"

COMMAND_TERM_NAME = "motion_command"

BODY_NAMES_TO_TRACK: list[str] = [
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
]

BODY_NAME_REF: list[str] = ["pelvis"]  # HERO anchors everything at the pelvis (full orientation)

IK_REACH_SOURCE_TAG = "ik_reach_example"
"""Source tag of the generated reaching clips (``data_tools.npz_convert --source-tag``, see docs/data.md)."""

DEFAULT_CLIP_END_POLICY_BY_SOURCE = {"amass": "rollover", IK_REACH_SOURCE_TAG: "hold"}
"""Mocap clips roll over into a new clip when they end; generated reaching clips hold their last frame. A corpus
manifest's ``clip_end_policy_by_source`` is overlaid on these defaults by ``scripts/train.py``."""

init_pose_config = NoiseToInitialPoseConfig(
    overall_noise_scale=1.0,
    dof_pos=0.1,
    root_pos=[0.05, 0.05, 0.01],
    root_rot=[0.1, 0.1, 0.2],
    root_lin_vel=[0.5, 0.5, 0.2],
    root_ang_vel=[0.52, 0.52, 0.78],
    object_pos=[0.05, 0.05, 0.0],
)

def make_hero_motion_config(**overrides) -> HeroMotionConfig:
    """Build paper commands with explicit AMASS and generated IK source behavior."""
    arm_kwargs = dict(h_offset_from_clip=False, stand_flag_mode="bernoulli_walk_0p6", zero_waist_when_walking=True)
    kwargs = dict(
        motion_file="",  # placeholder; motion_dir (CLI) takes precedence when non-empty
        motion_dir="",
        body_names_to_track=list(BODY_NAMES_TO_TRACK),
        body_name_ref=list(BODY_NAME_REF),
        use_adaptive_timesteps_sampler=True,
        adaptive_sampler_per_clip=True,
        # Failure-weighted sampling needs these guards: with the backend defaults (no cap, uniform 0.1) one long clip
        # that fails at every phase absorbed 80 % of the draws within one iteration and episodes shrank to 0.2 s.
        # The bound is relative to each clip's prior share so that source_weights stay honoured (a flat absolute cap
        # starves sources with few clips and forces small corpora uniform).
        adaptive_sampler_clip_max_probability=1.0,
        adaptive_sampler_clip_max_relative=ADAPTIVE_CLIP_MAX_RELATIVE,
        adaptive_sampler_uniform_ratio=ADAPTIVE_UNIFORM_RATIO,
        rollover_at_clip_end=True,  # mocap default; per-source overrides via clip_end_policy_by_source
        # Keep bulk clip tensors in host RAM to limit GPU memory use.

        motion_storage_device="cpu",
        noise_to_initial_pose=init_pose_config,
        source_weights={},
        clip_end_policy_by_source=dict(DEFAULT_CLIP_END_POLICY_BY_SOURCE),
        h_offset_range=(-0.25, 0.0),
        fix_upper_body_prob=0.3,
        stand_speed_thr=0.15,
        stand_yaw_rate_thr=0.2,
        **arm_kwargs,
    )
    kwargs.update(overrides)
    return HeroMotionConfig(**kwargs)

def make_hero_command(motion_config: HeroMotionConfig, func: str = HERO_COMMAND_FUNC) -> CommandManagerCfg:
    """setup/reset/step all point at the same ``HeroMotionCommand`` class."""
    return CommandManagerCfg(
        params={},
        setup_terms={COMMAND_TERM_NAME: CommandTermCfg(func=func, params={"motion_config": motion_config})},
        reset_terms={COMMAND_TERM_NAME: CommandTermCfg(func=func)},
        step_terms={COMMAND_TERM_NAME: CommandTermCfg(func=func)},
    )

def get_motion_config(cfg: CommandManagerCfg) -> HeroMotionConfig:
    """Motion config of the command term; checkpoints deserialize it as a plain dict."""
    return coerce_hero_motion_config(cfg.setup_terms[COMMAND_TERM_NAME].params["motion_config"])

def with_motion_config(cfg: CommandManagerCfg, **overrides) -> CommandManagerCfg:
    """``cfg`` with the motion config's fields replaced by ``overrides``.

    A checkpoint's saved motion config is a plain dict; the overrides are applied to it *before* it is coerced into a
    ``HeroMotionConfig``, so values the dataclass refuses today (e.g. a pre-release absolute clip cap) can be
    replaced instead of failing validation first."""
    raw = cfg.setup_terms[COMMAND_TERM_NAME].params["motion_config"]
    if isinstance(raw, Mapping):
        mc = coerce_hero_motion_config({**raw, **overrides})
    else:
        mc = replace(coerce_hero_motion_config(raw), **overrides)
    return replace(
        cfg,
        setup_terms={
            COMMAND_TERM_NAME: replace(cfg.setup_terms[COMMAND_TERM_NAME], params={"motion_config": mc}),
        },
    )

INFERENCE_SAMPLER_FIELDS: tuple[str, ...] = ("use_adaptive_timesteps_sampler",) + tuple(
    f.name for f in fields(HeroMotionConfig) if f.name.startswith("adaptive_sampler_"))
"""Motion-config fields an inference-only environment (export, evaluation) takes from the live preset rather than
from the checkpoint: the sampler only matters for training draws, and a checkpoint's saved sampler settings may be
ones the current code refuses."""

def with_inference_motion_config(saved: CommandManagerCfg, preset: CommandManagerCfg, *, motion_dir: str) -> CommandManagerCfg:
    """The saved command config of a checkpoint, made loadable for inference-only use (ONNX export, evaluation).

    Keeps every reference / command setting the policy was trained with, points it at ``motion_dir``, takes the
    sampler settings (:data:`INFERENCE_SAMPLER_FIELDS`) from ``preset`` and sets ``reset_sampler_on_resume`` so a
    saved failure table the live sampler cannot load is dropped with a warning instead of an error: nothing is
    sampled for training, so neither the table nor its settings can change what the exported policy does."""
    preset_mc = get_motion_config(preset)
    overrides = {name: getattr(preset_mc, name) for name in INFERENCE_SAMPLER_FIELDS}
    return with_motion_config(saved, motion_dir=str(motion_dir), reset_sampler_on_resume=True, **overrides)

hero_h1_motion_config = make_hero_motion_config()

hero_h1_command = make_hero_command(hero_h1_motion_config)
DEFAULTS = {"hero": hero_h1_command}
