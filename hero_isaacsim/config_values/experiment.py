"""HERO training configurations: the paper recipe with and without delta anchor, and the delta-EE ablation.

All presets share the paper's action, reward, randomization and training settings and come in a dual-actor
(paper) and a single-actor flavour. Delta anchor adds the h20/h21/h22 observations; ``without_delta_ee`` removes the
residual end-effector feedback (h07/h08) from the actor and keeps everything else, including the critic."""
from __future__ import annotations

from dataclasses import replace
from typing import Any

from holosoma.config_types.action import ActionManagerCfg, ActionTermCfg
from holosoma.config_types.algo import (LayerConfig, ModuleConfig, OptimizerConfig, PPOAlgoConfig, PPOConfig, PPODualAlgoConfig,
                                        PPODualConfig, PPODualModuleDictConfig, PPOModuleDictConfig)
from holosoma.config_types.experiment import ExperimentConfig, TrainingConfig
from holosoma.config_types.logger import DisabledLoggerConfig
from holosoma.config_values import simulator, terrain
from hero_isaacsim.config_values import command, curriculum, observation, randomization, reward, termination
from hero_isaacsim.config_values.robot import g1_29dof_dex3_m12
from hero_isaacsim.constants import PALM_BODY_NAMES

WITH_DELTA_ANCHOR = "with_delta_anchor"
WITHOUT_DELTA_ANCHOR = "without_delta_anchor"
DELTA_ANCHOR_TERMS = (observation.ANCHOR_TERM_NAME, *observation.ANCHOR2_TERM_NAMES)
WITHOUT_DELTA_EE = "without_delta_ee"
DELTA_EE_TERMS = observation.DELTA_EE_TERM_NAMES
SAVE_INTERVAL = 500
NUM_LEARNING_ITERATIONS = 20000
HERO_PPO_KWARGS: dict[str, Any] = dict(
    num_learning_epochs=5,
    num_mini_batches=4,
    clip_param=0.2,
    gamma=0.99,
    lam=0.95,
    value_loss_coef=1.0,
    entropy_coef=0.01,
    actor_learning_rate=1e-4,
    critic_learning_rate=1e-4,
    actor_optimizer=OptimizerConfig(_target_="torch.optim.AdamW", weight_decay=1e-2),
    critic_optimizer=OptimizerConfig(_target_="torch.optim.AdamW", weight_decay=1e-2),
    max_grad_norm=1.0,
    schedule="adaptive",
    desired_kl=0.01,
    min_actor_learning_rate=1e-5,
    max_actor_learning_rate=1e-2,
    min_critic_learning_rate=1e-5,
    max_critic_learning_rate=1e-2,
    use_symmetry=False,
    num_steps_per_env=24,
    save_interval=SAVE_INTERVAL,
    load_optimizer=True,
    init_noise_std=0.8,  # lower body; upper body 0.6 (init_noise_std_upper on PPODualConfig)
    init_noise_std_upper=0.6,
    num_learning_iterations=NUM_LEARNING_ITERATIONS,
    init_at_random_ep_len=True,
    empirical_normalization=False,
    export_motion_in_onnx=False,  # Keep motion data separate from the policy.
)


def _module(output_dim: int, input_group: str) -> ModuleConfig:
    return ModuleConfig(type="MLP", input_dim=[input_group], output_dim=[output_dim],
                        layer_config=LayerConfig(hidden_dims=[512, 256, 128], activation="ELU"))


ACTOR_DUAL = "dual"
ACTOR_SINGLE = "single"
SINGLE_SUFFIX = "_single"
WITHOUT_DELTA_ANCHOR_SINGLE = WITHOUT_DELTA_ANCHOR + SINGLE_SUFFIX
WITH_DELTA_ANCHOR_SINGLE = WITH_DELTA_ANCHOR + SINGLE_SUFFIX
WITHOUT_DELTA_EE_SINGLE = WITHOUT_DELTA_EE + SINGLE_SUFFIX
PPO_SINGLE_TARGET = "hero_isaacsim.agents.ppo_single.ppo_single.PPOSingle"
PPO_DUAL_TARGET = "hero_isaacsim.agents.ppo_dual.ppo_dual.PPODual"


def make_hero_recipe(*, delta_anchor: bool = False, delta_ee: bool = True, actor: str = ACTOR_DUAL) -> ExperimentConfig:
    """Build a serializable HERO configuration with local logging by default.

    ``actor="dual"`` is the paper's two-actor PPO (lower-body and upper-body heads, 15 + 14 actions, their own
    critics); ``actor="single"`` trains one whole-body actor and one critic with holosoma's stock PPO on the same
    observations, rewards, action contract and randomization. Both export the same frame-major ONNX layout.
    ``delta_ee=False`` is the ablation without the residual end-effector feedback: the actor loses ``h07``/``h08``
    (675 -> 585 inputs), the critic and every other setting stay those of ``without_delta_anchor``; it has no
    delta-anchor flavour (that combination has no preset name, so it is rejected)."""
    if actor not in (ACTOR_DUAL, ACTOR_SINGLE):
        raise ValueError(f"actor must be {ACTOR_DUAL!r} or {ACTOR_SINGLE!r}, got {actor!r}")
    if delta_anchor and not delta_ee:
        raise ValueError("no preset combines delta anchor with the delta-EE ablation; use delta_ee=False with delta_anchor=False")
    name = WITH_DELTA_ANCHOR if delta_anchor else (WITHOUT_DELTA_ANCHOR if delta_ee else WITHOUT_DELTA_EE)
    if actor == ACTOR_SINGLE:
        name += SINGLE_SUFFIX
    obs = observation.hero_h1_observation
    if delta_anchor:
        obs = observation.with_anchor2_terms(observation.with_anchor_term(obs))
    if not delta_ee:
        obs = observation.without_delta_ee_terms(obs)
    robot = replace(g1_29dof_dex3_m12,
                    randomize_link_body_names=[b for b in g1_29dof_dex3_m12.randomize_link_body_names
                                               if b not in PALM_BODY_NAMES])
    sim = replace(simulator.isaacsim, config=replace(simulator.isaacsim.config,
        scene=replace(simulator.isaacsim.config.scene, env_spacing=3.0),
        sim=replace(simulator.isaacsim.config.sim, fps=500, control_decimation=10, max_episode_length_s=10.0)))
    if actor == ACTOR_SINGLE:
        single_kwargs = {k: v for k, v in HERO_PPO_KWARGS.items() if k != "init_noise_std_upper"}
        algo = PPOAlgoConfig(_target_=PPO_SINGLE_TARGET, _recursive_=False,
            config=PPOConfig(module_dict=PPOModuleDictConfig(
                actor=_module(29, observation.ACTOR_GROUP), critic=_module(1, observation.CRITIC_GROUP)), **single_kwargs))
    else:
        algo = PPODualAlgoConfig(_target_=PPO_DUAL_TARGET, _recursive_=False,
            config=PPODualConfig(module_dict=PPODualModuleDictConfig(
                actor_lower=_module(15, observation.ACTOR_GROUP), actor_upper=_module(14, observation.ACTOR_GROUP),
                critic_lower=_module(1, observation.CRITIC_GROUP), critic_upper=_module(1, observation.CRITIC_GROUP)),
                **HERO_PPO_KWARGS))
    return ExperimentConfig(
        env_class="hero_isaacsim.envs.hero_tracking_manager.HeroTrackingManager",
        training=TrainingConfig(project="hero", name=name, num_envs=4096, seed=0, headless=True),
        algo=algo, simulator=sim, terrain=terrain.terrain_locomotion_plane, robot=robot, observation=obs,
        action=ActionManagerCfg(terms={"joint_control": ActionTermCfg(
            func="hero_isaacsim.managers.action.hero_joint_control:HeroResidualJointPositionActionTerm",
            params={"residual_upper_body_action": True}, scale=1.0, clip=None)}),
        reward=reward.hero_109_curr_reward,
        termination=termination.hero_h1_termination,
        randomization=randomization.hero_randomization,
        command=command.hero_h1_command,
        curriculum=curriculum.hero_curriculum,
        logger=DisabledLoggerConfig(base_dir="runs"),
    )


without_delta_anchor = make_hero_recipe(delta_anchor=False)
with_delta_anchor = make_hero_recipe(delta_anchor=True)
without_delta_ee = make_hero_recipe(delta_ee=False)
without_delta_anchor_single = make_hero_recipe(delta_anchor=False, actor=ACTOR_SINGLE)
with_delta_anchor_single = make_hero_recipe(delta_anchor=True, actor=ACTOR_SINGLE)
without_delta_ee_single = make_hero_recipe(delta_ee=False, actor=ACTOR_SINGLE)
# Insertion order = DUAL_ACTOR_CONFIGS + SINGLE_ACTOR_CONFIGS (scripts/train.py --config choices follow the same order).
DEFAULTS = {WITHOUT_DELTA_ANCHOR: without_delta_anchor, WITH_DELTA_ANCHOR: with_delta_anchor, WITHOUT_DELTA_EE: without_delta_ee,
            WITHOUT_DELTA_ANCHOR_SINGLE: without_delta_anchor_single, WITH_DELTA_ANCHOR_SINGLE: with_delta_anchor_single,
            WITHOUT_DELTA_EE_SINGLE: without_delta_ee_single}
DUAL_ACTOR_CONFIGS = (WITHOUT_DELTA_ANCHOR, WITH_DELTA_ANCHOR, WITHOUT_DELTA_EE)
SINGLE_ACTOR_CONFIGS = (WITHOUT_DELTA_ANCHOR_SINGLE, WITH_DELTA_ANCHOR_SINGLE, WITHOUT_DELTA_EE_SINGLE)
ABLATION_PRESETS = {}  # compatibility for generic checkpoint/cache helpers
V4_PRESETS = {}


def observation_contract(config: ExperimentConfig) -> dict[str, Any]:
    return {name: {"terms": [t for t, _ in observation.sorted_layout(group)],
                   "term_dims": [d for _, d in observation.sorted_layout(group)],
                   "history_length": group.history_length, "dim": observation.group_dim(group)}
            for name, group in config.observation.groups.items()}


def validate_checkpoint_contract(checkpoint: dict[str, Any], config: ExperimentConfig) -> None:
    """Reject mismatched layouts before allocating an Isaac Sim environment.

    Comparing names and per-term parameters also catches equal-width but semantically
    different checkpoints. A bare legacy state dict must be converted explicitly."""
    saved = checkpoint.get("experiment_config")
    if not isinstance(saved, dict) or not isinstance(saved.get("observation"), dict):
        raise ValueError("Checkpoint has no saved observation config; convert it explicitly before resuming.")
    saved_algo = saved.get("algo") if isinstance(saved.get("algo"), dict) else {}
    if saved_algo.get("_target_") and saved_algo.get("_target_") != config.algo._target_:
        raise ValueError(f"Checkpoint was trained with {saved_algo.get('_target_')}, not {config.algo._target_}; "
                         f"select the matching configuration ({config.training.name} expects {config.algo._target_.rsplit('.', 1)[-1]}).")
    expected = config.observation
    actual = saved["observation"]
    if actual.get("clip_observations") != expected.clip_observations:
        raise ValueError("Checkpoint observation clipping differs from selected config.")
    for name, group in expected.groups.items():
        source = actual.get("groups", {}).get(name, {})
        if source.get("history_length") != group.history_length or set(source.get("terms", {})) != set(group.terms):
            raise ValueError(f"Checkpoint {name} layout differs from {config.training.name}; select matching weights or train from scratch.")
        for term_name, term in group.terms.items():
            old = source["terms"][term_name]
            for key in ("func", "scale", "params"):
                lhs, rhs = old.get(key), getattr(term, key)
                # JSON roundtrip normalizes tuple/list horizons in serialized configs.
                import json
                if json.dumps(lhs, sort_keys=True) != json.dumps(rhs, sort_keys=True):
                    raise ValueError(f"Checkpoint term {name}.{term_name}.{key} differs from selected config.")
