"""HERO training configurations: the paper recipe with and without delta anchor, and the delta-EE ablation.

All presets share the paper's action, reward, randomization and training settings and come in a dual-actor
(paper) and a single-actor flavour. Delta anchor adds the h20/h21/h22 observations; ``without_delta_ee`` removes the
residual end-effector feedback (h07/h08) from the actor and keeps everything else, including the critic."""
from __future__ import annotations

from dataclasses import dataclass, replace
import json
import math
from typing import Any, Mapping, Sequence

from loguru import logger

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


def _json_norm(value: Any) -> str:
    """JSON round trip normalises tuple/list horizons and key order in serialized configs."""
    return json.dumps(value, sort_keys=True, default=str)


def _collect_term_changes(changes: list[str], prefix: str, saved_terms: Any, live_terms: Any, fields: tuple[str, ...]) -> None:
    saved_terms = saved_terms if isinstance(saved_terms, Mapping) else {}
    live_terms = live_terms if isinstance(live_terms, Mapping) else {}
    for name in sorted(set(saved_terms) | set(live_terms)):
        if name not in saved_terms or name not in live_terms:
            changes.append(f"{prefix}.{name}: {'present' if name in saved_terms else 'absent'} in checkpoint, "
                           f"{'present' if name in live_terms else 'absent'} in the selected config")
            continue
        for field in fields:
            lhs, rhs = saved_terms[name].get(field), live_terms[name].get(field)
            if _json_norm(lhs) != _json_norm(rhs):
                changes.append(f"{prefix}.{name}.{field}: {_json_norm(lhs)} -> {_json_norm(rhs)}")


def _motion_config_dict(serialized: Mapping[str, Any]) -> dict[str, Any]:
    try:
        value = serialized["command"]["setup_terms"]["motion_command"]["params"]["motion_config"]
    except (KeyError, TypeError):
        return {}
    return dict(value) if isinstance(value, Mapping) else {}


RESUME_PER_RUN_MOTION_FIELDS: frozenset[str] = frozenset({
    "motion_dir", "motion_file",            # every resume has its own corpus path
    "source_weights",                       # the episode mix may legitimately change between runs (manifest-driven); a saved
                                            # sampler table built under another mix is a SAMPLER finding instead
                                            # (resume_sampler_state_problem), not a configuration change
    "source_weights_from_manifest",         # manifest bookkeeping
    "reset_sampler_on_resume",              # the resume opt-in itself
    "motion_storage_device", "shared_cache_dir",  # where the corpus lives in memory / on disk
})
"""Motion-config fields :func:`resume_config_changes` does NOT compare: per-run plumbing rather than training
settings. Everything else in the motion config is compared, in particular the sampler settings
(``adaptive_sampler_*``, ``use_adaptive_timesteps_sampler``), the clip-end policies (``clip_end_policy_by_source``,
``default_clip_end_policy``, ``rollover_at_clip_end``), the height-offset curriculum (``h_offset_*``,
``h_curriculum_*``), the command sampling (``walk_prob``, ``fix_upper_body_prob``, ``vel_cmd_*``,
``command_resample_time_s``) and the reference settings."""

RESUME_ALGO_CONFIG_FIELDS: tuple[str, ...] = ("hero_std_clamp_max",)
"""Algorithm-config fields :func:`resume_config_changes` compares (``algo.config.<field>``): the exploration-std clamp
(``scripts/train.py --std-clamp-max``) changes which trajectories the policy samples from the checkpoint on, so adding,
removing or changing it on resume is a finding. A checkpoint written before the field existed reads as ``null`` (no clamp)."""


def resume_config_changes(saved: Mapping[str, Any], config: ExperimentConfig) -> list[str]:
    """Key paths of the training-distribution settings that differ between a checkpoint's saved config and ``config``.

    Covers the whole motion config except :data:`RESUME_PER_RUN_MOTION_FIELDS`, the algorithm fields in
    :data:`RESUME_ALGO_CONFIG_FIELDS` (the exploration-std clamp), the curriculum parameters, reward term weights and
    parameters, termination parameters and randomization parameters -- the settings that change what the policy is
    trained on without changing the network. Observation layout and algorithm class are checked separately (hard
    errors). Each entry reads ``path: checkpoint -> selected``."""
    live = config.to_serializable_dict()
    changes: list[str] = []
    saved_mc, live_mc = _motion_config_dict(saved), _motion_config_dict(live)
    for key in sorted(k for k in set(saved_mc) | set(live_mc) if k not in RESUME_PER_RUN_MOTION_FIELDS):
        lhs, rhs = saved_mc.get(key), live_mc.get(key)
        if _json_norm(lhs) != _json_norm(rhs):
            changes.append(f"command.motion_command.motion_config.{key}: {_json_norm(lhs)} -> {_json_norm(rhs)}")
    saved_algo = saved.get("algo") if isinstance(saved.get("algo"), Mapping) else {}
    saved_algo_cfg = saved_algo.get("config") if isinstance(saved_algo.get("config"), Mapping) else {}
    live_algo_cfg = (live.get("algo") or {}).get("config") or {}
    for key in RESUME_ALGO_CONFIG_FIELDS:
        lhs, rhs = saved_algo_cfg.get(key), live_algo_cfg.get(key)
        if _json_norm(lhs) != _json_norm(rhs):
            changes.append(f"algo.config.{key}: {_json_norm(lhs)} -> {_json_norm(rhs)}")
    saved_cur = saved.get("curriculum") if isinstance(saved.get("curriculum"), Mapping) else {}
    live_cur = live.get("curriculum") or {}
    if _json_norm(saved_cur.get("params")) != _json_norm(live_cur.get("params")):
        changes.append(f"curriculum.params: {_json_norm(saved_cur.get('params'))} -> {_json_norm(live_cur.get('params'))}")
    for section in ("setup_terms", "reset_terms", "step_terms"):
        _collect_term_changes(changes, f"curriculum.{section}", saved_cur.get(section), live_cur.get(section), ("params",))
    saved_rew = saved.get("reward") if isinstance(saved.get("reward"), Mapping) else {}
    _collect_term_changes(changes, "reward.terms", saved_rew.get("terms"), (live.get("reward") or {}).get("terms"), ("weight", "params"))
    saved_term = saved.get("termination") if isinstance(saved.get("termination"), Mapping) else {}
    _collect_term_changes(changes, "termination.terms", saved_term.get("terms"), (live.get("termination") or {}).get("terms"), ("params",))
    saved_rand = saved.get("randomization") if isinstance(saved.get("randomization"), Mapping) else {}
    live_rand = live.get("randomization") or {}
    for section in ("setup_terms", "reset_terms", "step_terms"):
        _collect_term_changes(changes, f"randomization.{section}", saved_rand.get(section), live_rand.get(section), ("params",))
    return changes


def source_mix(weights: Any, corpus_tags: Sequence[str]) -> dict[str, float] | None:
    """Per-source episode shares the way ``build_clip_prior`` derives them for a corpus with ``corpus_tags``: tags the
    corpus lacks are dropped, corpus tags without an entry get 0, the rest is normalised to sum 1 (sorted by tag).
    ``None`` for an empty / absent mapping (the sampler then draws uniformly over clips). A zero total is returned
    unnormalised (all zeros): the sampler refuses such a corpus at construction, so it never matches a saved table."""
    if not isinstance(weights, Mapping) or not weights:
        return None
    mix: dict[str, float] = {}
    for tag in corpus_tags:
        value = weights.get(tag, 0.0)
        mix[str(tag)] = float(value) if isinstance(value, (int, float)) and not isinstance(value, bool) else float("nan")
    total = sum(mix.values())
    if not math.isfinite(total) or total <= 0.0:
        return dict(sorted(mix.items()))
    return {tag: value / total for tag, value in sorted(mix.items())}


def _mixes_differ(saved: dict[str, float] | None, live: dict[str, float] | None, *, atol: float = 1.0e-6) -> bool:
    if (saved is None) != (live is None):
        return True
    if saved is None or live is None:
        return False
    if set(saved) != set(live):
        return True
    return any(not (abs(saved[tag] - live[tag]) <= atol) for tag in saved)


def _format_mix(mix: dict[str, float] | None) -> str:
    if mix is None:
        return "uniform over clips"
    return "{" + ", ".join(f"{tag}: {share:.4g}" for tag, share in mix.items()) + "}"


def _sampler_registry_problem(checkpoint: Mapping[str, Any], sampler_state: Mapping[str, Any], motion_config: Any,
                              corpus_num_clips: int | None, corpus_source_tags: Sequence[str] | None) -> str | None:
    """Why the saved table's clip prior cannot be the live one (None if nothing is known to differ).

    The HERO sampler stores the source tags of its corpus (``hero_source_tags``) and the clip count
    (``hero_num_motions``) next to the table, and refuses a table whose ``clip_prior`` differs from the live one
    (``HeroAdaptiveTimestepsSampler.load_state_dict``). The live prior is ``source_weights`` restricted to the corpus
    tags and normalised (:func:`source_mix`) spread over each source's clips, so a changed mix, a changed clip count
    or another set of source tags means the table cannot be restored. The clip count is the number of ``*.npz`` clips
    in the motion directory (``scripts/train.py`` counts them; the loader registers exactly those unless it skips a
    malformed file). Without the saved tags (a table not written by the HERO sampler) nothing can be compared."""
    saved_tags = sampler_state.get("hero_source_tags")
    if not isinstance(saved_tags, (list, tuple)) or not saved_tags:
        return None
    saved_tags = sorted(str(tag) for tag in saved_tags)
    saved_count = sampler_state.get("hero_num_motions", sampler_state.get("num_clips"))
    saved_count = int(saved_count) if isinstance(saved_count, (int, float)) and not isinstance(saved_count, bool) else None
    live_tags = sorted({str(tag) for tag in corpus_source_tags}) if corpus_source_tags is not None else None
    live_count = int(corpus_num_clips) if corpus_num_clips is not None else None
    experiment_config = checkpoint.get("experiment_config")
    saved_weights = _motion_config_dict(experiment_config if isinstance(experiment_config, Mapping) else {}).get("source_weights")
    saved_mix = source_mix(saved_weights, saved_tags)
    live_mix = source_mix(getattr(motion_config, "source_weights", None), live_tags if live_tags is not None else saved_tags)
    tags_differ = live_tags is not None and live_tags != saved_tags
    count_differs = saved_count is not None and live_count is not None and saved_count != live_count
    if not (tags_differ or count_differs or _mixes_differ(saved_mix, live_mix)):
        return None

    def describe(mix: dict[str, float] | None, tags: list[str] | None, count: int | None) -> str:
        parts = [f"source mix {_format_mix(mix)}"]
        if tags_differ and tags is not None and mix is None:  # a mix already lists the tags
            parts.append(f"source tags {tags}")
        if count is not None:
            parts.append(f"{count} clips")
        return " / ".join(parts)

    return (f"the saved failure table was built for {describe(saved_mix, saved_tags, saved_count)}; this run uses "
            f"{describe(live_mix, live_tags, live_count)} -> it cannot be restored")


def resume_sampler_state_problem(checkpoint: Mapping[str, Any], config: ExperimentConfig, *,
                                 corpus_num_clips: int | None = None,
                                 corpus_source_tags: Sequence[str] | None = None) -> str | None:
    """Why the adaptive sampler table saved in ``checkpoint`` cannot be restored under ``config`` (None if it can).

    Compares, before any environment exists, (1) the saved ``sampling_policy`` (uniform ratio, temperature, absolute
    and relative caps, composition rule) with the policy the selected motion configuration builds and (2) the saved
    table's registry -- the source mix it was built for (the checkpoint's ``source_weights`` over its corpus' source
    tags, normalised like the live prior), its clip count and source tags -- with the live ``source_weights`` and the
    corpus census ``corpus_num_clips`` / ``corpus_source_tags`` (``scripts/train.py`` passes the count and tags of the
    ``*.npz`` clips it enumerates; without them only the mix is compared, over the saved tags). Every problem found is
    reported in one message; all of them need ``--reset-sampler-on-resume``."""
    from hero_isaacsim.config_values.command import get_motion_config  # noqa: PLC0415
    from hero_isaacsim.managers.command.sampler import HeroAdaptiveTimestepsSampler, expected_sampling_policy  # noqa: PLC0415

    env_state = checkpoint.get("env_state")
    sampler_state = env_state.get("adaptive_timesteps_sampler") if isinstance(env_state, Mapping) else None
    if not isinstance(sampler_state, Mapping):
        return None
    mc = get_motion_config(config.command)
    if not getattr(mc, "use_adaptive_timesteps_sampler", False):
        return "the checkpoint carries an adaptive sampler table but the selected config has the sampler disabled"
    expected = expected_sampling_policy(
        uniform_ratio=getattr(mc, "adaptive_sampler_uniform_ratio", 0.1),
        clip_temperature=getattr(mc, "adaptive_sampler_clip_temperature", 1.0),
        clip_max_probability=getattr(mc, "adaptive_sampler_clip_max_probability", 1.0),
        clip_cap_relative=getattr(mc, "adaptive_sampler_clip_max_relative", None),
    )
    saved_policy = sampler_state.get("sampling_policy")
    if not isinstance(saved_policy, Mapping):
        saved_policy = {}
    problems: list[str] = []
    mismatches = HeroAdaptiveTimestepsSampler.sampling_policy_mismatches(saved_policy, expected)
    if mismatches:
        problems.append("adaptive sampler checkpoint sampling policy differs from the live configuration "
                        f"(field: checkpoint -> selected): {mismatches}")
    registry = _sampler_registry_problem(checkpoint, sampler_state, mc, corpus_num_clips, corpus_source_tags)
    if registry:
        problems.append(registry)
    return "; ".join(problems) if problems else None


def stateful_curriculum_term_names(config: ExperimentConfig) -> list[str]:
    """Names of the curriculum terms whose progress a checkpoint of ``config`` carries under ``env_state["curriculum_terms"]``.

    Resolves the term classes from the configuration (no environment needed) with the rule the environment uses
    (``WholeBodyTrackingManager._stateful_curriculum_terms``): class-based terms with ``state_dict`` /
    ``load_state_dict``, except the episode-length tracker, which is persisted under its own key."""
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager  # noqa: PLC0415
    from holosoma.managers.curriculum.base import CurriculumTermBase  # noqa: PLC0415
    from holosoma.managers.utils import resolve_callable  # noqa: PLC0415

    names: list[str] = []
    curriculum = config.curriculum
    for section in (curriculum.setup_terms, curriculum.reset_terms, curriculum.step_terms):
        for name, term in (section or {}).items():
            if name in names or name == WholeBodyTrackingManager.AVERAGE_EPISODE_TRACKER_TERM:
                continue
            try:
                resolved = resolve_callable(term.func, context="curriculum term")
            except Exception:  # noqa: BLE001 - an unresolvable term is the manager's error to raise, not the pre-flight's
                continue
            if (isinstance(resolved, type) and issubclass(resolved, CurriculumTermBase)
                    and callable(getattr(resolved, "state_dict", None)) and callable(getattr(resolved, "load_state_dict", None))):
                names.append(name)
    return names


def curriculum_term_initial_values(config: ExperimentConfig, name: str) -> str:
    """Where curriculum term ``name`` restarts, from its live parameters (``'initial_scale 0.1'``); the parameters whose
    name starts with ``initial``. ``'its initial value'`` when the term declares none."""
    curriculum = config.curriculum
    for section in (curriculum.setup_terms, curriculum.reset_terms, curriculum.step_terms):
        term = (section or {}).get(name)
        params = getattr(term, "params", None)
        if isinstance(params, Mapping):
            initial = [(k, v) for k, v in params.items() if str(k).startswith("initial")]
            if initial:
                return ", ".join(f"{k} {v}" for k, v in initial)
    return "its initial value"


def resume_curriculum_state_problem(checkpoint: Mapping[str, Any], config: ExperimentConfig) -> str | None:
    """Why resuming ``checkpoint`` under ``config`` would restart a curriculum at its initial value (None if it would not).

    A checkpoint written before curriculum progress was persisted (``env_state`` without ``curriculum_terms``, or
    without an entry for a live term) restores the sampler table and the episode-length tracker but restarts e.g. the
    penalty scale at ``initial_scale`` (the logged ``penalty_scale`` shows the step); one written before the
    per-environment height-offset curriculum was persisted (``env_state`` without ``hero_h_curriculum_scale``) restarts
    every environment's scale at ``h_curriculum_init`` (logged as ``motion/h_curriculum_scale``). The environment logs
    a warning when it loads such a checkpoint; this reports both before Isaac Sim starts, with the restart values. A
    checkpoint with no ``env_state`` at all (not written by this code) is not reported: there is no partial state to
    be misled by."""
    from hero_isaacsim.config_values.command import get_motion_config  # noqa: PLC0415
    from hero_isaacsim.envs.hero_tracking_manager import HeroTrackingManager  # noqa: PLC0415

    env_state = checkpoint.get("env_state")
    if not isinstance(env_state, Mapping) or not env_state:
        return None
    problems: list[str] = []
    live = stateful_curriculum_term_names(config)
    if live:
        saved = env_state.get("curriculum_terms")
        saved_names = set(saved) if isinstance(saved, Mapping) else set()
        missing = [name for name in live if name not in saved_names]
        if missing:
            where = ("no 'curriculum_terms' (written before curriculum progress was checkpointed)" if saved is None
                     else f"no 'curriculum_terms' entry for {missing}")
            restarts = "; ".join(f"{name}: {curriculum_term_initial_values(config, name)}" for name in missing)
            problems.append(f"the checkpoint's env_state has {where}; the curriculum terms {missing} would restart at "
                            f"their initial values ({restarts})")
    h_init = getattr(get_motion_config(config.command), "h_curriculum_init", None)
    key = HeroTrackingManager.H_CURRICULUM_STATE_KEY
    if h_init is not None and key not in env_state:
        problems.append(f"the checkpoint's env_state has no '{key}' (written before the height-offset curriculum was "
                        f"checkpointed); the per-environment height-offset curriculum would restart at h_curriculum_init={h_init:g}")
    return "; ".join(problems) if problems else None


@dataclass(frozen=True)
class ResumePreflight:
    """Everything a resume needs the user to opt into, computed before Isaac Sim starts (``scripts/train.py``)."""

    config_changes: list[str]
    """:func:`resume_config_changes` -- gated by ``--allow-config-change``."""
    curriculum_state_problem: str | None
    """:func:`resume_curriculum_state_problem` (curriculum terms and the height-offset curriculum) -- gated by
    ``--allow-config-change``."""
    sampler_state_problem: str | None
    """:func:`resume_sampler_state_problem` (sampling policy, source mix, clip registry) -- gated by
    ``--reset-sampler-on-resume``."""

    CONFIG_FLAG = "--allow-config-change"
    SAMPLER_FLAG = "--reset-sampler-on-resume"

    def refusal(self, *, allow_config_change: bool, reset_sampler_on_resume: bool) -> str | None:
        """One message naming every finding the given flags do not cover and every flag still needed (None if accepted)."""
        findings: list[str] = []
        flags: list[str] = []
        if self.config_changes and not allow_config_change:
            listing = "\n    ".join(self.config_changes)
            findings.append("Checkpoint training settings differ from the selected configuration (checkpoint -> selected):\n    "
                            f"{listing}\n  Resuming would change the training distribution; {self.CONFIG_FLAG} continues anyway.")
            flags.append(self.CONFIG_FLAG)
        if self.curriculum_state_problem and not allow_config_change:
            findings.append(f"{self.curriculum_state_problem}; {self.CONFIG_FLAG} continues anyway.")
            if self.CONFIG_FLAG not in flags:
                flags.append(self.CONFIG_FLAG)
        if self.sampler_state_problem and not reset_sampler_on_resume:
            findings.append(f"{self.sampler_state_problem}. Resuming would silently train on a different clip distribution; "
                            f"{self.SAMPLER_FLAG} restarts the failure table from zeros instead.")
            flags.append(self.SAMPLER_FLAG)
        if not findings:
            return None
        return ("Resuming from this checkpoint needs explicit opt-in:\n" + "\n".join(f"- {f}" for f in findings)
                + f"\nPass {' and '.join(flags)} to continue.")

    def log_accepted(self) -> None:
        """Log what a resume that passed :meth:`refusal` continues with (the launcher prints the same in its JSON)."""
        if self.config_changes:
            logger.warning("Resuming with changed training settings ({}):\n  {}", self.CONFIG_FLAG, "\n  ".join(self.config_changes))
        if self.curriculum_state_problem:
            logger.warning("Resuming although {} ({})", self.curriculum_state_problem, self.CONFIG_FLAG)
        if self.sampler_state_problem:
            logger.warning("Resuming with the adaptive sampler table restarted from zeros ({}): {}", self.SAMPLER_FLAG,
                           self.sampler_state_problem)


def resume_preflight(checkpoint: Mapping[str, Any], config: ExperimentConfig, *, corpus_num_clips: int | None = None,
                     corpus_source_tags: Sequence[str] | None = None) -> ResumePreflight:
    """Hard contract checks (:func:`validate_checkpoint_contract`, layout / algorithm) then every opt-in finding at once,
    so a launcher can report all required flags in a single refusal instead of one per attempt. ``corpus_num_clips`` /
    ``corpus_source_tags`` describe the live corpus (``scripts/train.py`` enumerates its ``*.npz`` clips) for the
    sampler registry check; without them the saved table is compared on its source mix alone."""
    validate_checkpoint_contract(dict(checkpoint), config)
    saved = checkpoint["experiment_config"]
    return ResumePreflight(
        config_changes=resume_config_changes(saved, config),
        curriculum_state_problem=resume_curriculum_state_problem(checkpoint, config),
        sampler_state_problem=resume_sampler_state_problem(checkpoint, config, corpus_num_clips=corpus_num_clips,
                                                           corpus_source_tags=corpus_source_tags),
    )


def validate_checkpoint_contract(checkpoint: Mapping[str, Any], config: ExperimentConfig) -> None:
    """Reject mismatched layouts before allocating an Isaac Sim environment.

    Comparing names and per-term parameters also catches equal-width but semantically
    different checkpoints. A bare legacy state dict must be converted explicitly.

    The training-distribution settings are NOT compared here: that is the resume pre-flight (:func:`resume_preflight`
    -> :class:`ResumePreflight`), which reports every opt-in finding in one message; the export path only needs the
    layout check."""
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
                if _json_norm(lhs) != _json_norm(rhs):
                    raise ValueError(f"Checkpoint term {name}.{term_name}.{key} differs from selected config.")
