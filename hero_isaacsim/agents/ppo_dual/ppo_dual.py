"""PPO with separate upper- and lower-body rewards and policies."""

from __future__ import annotations

import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import torch
from loguru import logger
from torch import nn

from holosoma.agents.modules.data_utils import RolloutStorage
from holosoma.agents.ppo.ppo import PPO, EmpiricalNormalization
from holosoma.config_types.algo import PPODualConfig
from holosoma.utils.helpers import get_class, instantiate
from holosoma.utils.inference_helpers import (
    attach_onnx_metadata,
    publish_onnx_atomically,
    validate_onnx_deployment_metadata,
)

from .export import (
    ACTION_CONTRACT_HERO_RESIDUAL_UPPER,
    HERO_ANCHOR_TERM_NAME,
    HERO_ANCHOR_TERMS_KEY,
    HERO_FUTURE_STEPS_KEYS,
    HERO_H20_FUTURE_STEPS_KEY,
    HERO_H21_FUTURE_STEPS_KEY,
    HERO_H22_FUTURE_STEPS_KEY,
    HERO_PLUS_PER_FRAME_DIM,
    ONNX_INPUT_NAMES,
    ONNX_OUTPUT_NAME,
    DualActorOnnxWrapper,
    build_hero_sidecar,
    dual_export_metadata,
    export_dual_actor_as_onnx,
    hero_anchor_terms_block,
    hero_command_block,
    hero_sidecar_path,
    hero_term_future_steps,
    input_perm_for_layout,
)
from .layout import HISTORY_LAYOUT_FRAME_MAJOR, SUPPORTED_HISTORY_LAYOUTS, ActorObsLayout, layout_from_env
from .modules import GROUP_KEYS, GROUP_SUFFIX, LOWER_BODY, UPPER_BODY, PPODualActor, PPODualCritic

REWARDS_BY_GROUP_KEY = "rewards_by_group"
"""``extras`` key published by the env (== ``GroupedRewardManager.REWARDS_BY_GROUP_KEY``)."""

PER_GROUP_STORAGE_KEYS: tuple[str, ...] = ("rewards", "values", "returns", "advantages", "actions_log_prob")
"""Storage keys that exist once per body group (suffixed ``_lb`` / ``_ub``)."""

CHECKPOINT_MODULE_KEYS: tuple[str, ...] = ("actor_lower", "actor_upper", "critic_lower", "critic_upper")

HERO_EXPAND_KEY = "hero_expand"
"""Checkpoint key for source-model metadata, preserved across load and save."""
HERO_EXPAND_SUMMARY_KEYS: tuple[str, ...] = ("from", "from_path", "sha256", "exp", "src_exp", "iter", "new_terms", "timestamp", "lineage", "kind", "removed_terms")

STD_CLAMP_MAX_CONFIG_FIELD = "hero_std_clamp_max"
"""Optional algo-config attribute for the upper std clamp (``PPODualConfig`` itself has no such field -- vendored)."""
STD_CLAMP_MAX_ENV_VAR = "HERO_PPO_DUAL_STD_CLAMP_MAX"
"""Env-var override of the upper std clamp (float, e.g. ``0.8``; empty / unset = not set)."""
HERO_KNOB_NAMES: tuple[str, ...] = ("std_clamp_max",)
"""PPODual knobs without a ``PPODualConfig`` field; resolution kwarg > config attribute > env var > class default."""
STD_CLAMP_MAX_LOWER_CONFIG_FIELD = "hero_std_clamp_max_lower"
"""Optional lower-body std bound; ``None`` or absent inherits ``std_clamp_max``."""
STD_CLAMP_MAX_LOWER_ENV_VAR = "HERO_PPO_DUAL_STD_CLAMP_MAX_LOWER"
"""Env-var override of the LOWER-body upper std clamp (float, e.g. ``0.3``; empty / unset = not set)."""
HERO_HEAD_KNOB_NAMES: tuple[str, ...] = ("std_clamp_max_lower",)
ANCHOR_STD_CLAMP_MAX = 0.8
PPO_DUAL_TARGET = "hero_isaacsim.agents.ppo_dual.ppo_dual.PPODual"
PPO_DUAL_ANCHOR_TARGET = "hero_isaacsim.agents.ppo_dual.ppo_dual.PPODualAnchor"
PIN_STD_CLAMP_MAX = 0.5
PPO_DUAL_ANCHOR_PIN_TARGET = "hero_isaacsim.agents.ppo_dual.ppo_dual.PPODualAnchorPin"
PIN_LOWSTD_STD_CLAMP_MAX_LOWER = 0.3
PPO_DUAL_ANCHOR_PIN_LOWSTD_TARGET = "hero_isaacsim.agents.ppo_dual.ppo_dual.PPODualAnchorPinLowStd"


def storage_key(name: str, group: str) -> str:
    """``values`` + ``lower_body`` -> ``values_lb``."""
    return f"{name}_{GROUP_SUFFIX[group]}"


def _env_float(name: str) -> float | None:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return None
    try:
        return float(raw)
    except ValueError as exc:
        raise ValueError(f"{name}={raw!r} is not a float") from exc


def std_clamp_max_for_target(target: str) -> float | None:


    cls = get_class(target)
    if not (isinstance(cls, type) and issubclass(cls, PPODual)):
        raise TypeError(f"{target!r} is not a PPODual subclass ({cls!r})")
    value = cls.HERO_KNOB_DEFAULTS.get("std_clamp_max")
    return None if value is None else float(value)


def std_clamp_max_lower_for_target(target: str) -> float | None:
    """The ``std_clamp_max_lower`` class default carried by the algo class named ``target`` (dotted path), ``None`` when the class does
    not pin a separate LOWER-body clamp (inherits ``std_clamp_max``: every class but :class:`PPODualAnchorPinLowStd`)."""
    cls = get_class(target)
    if not (isinstance(cls, type) and issubclass(cls, PPODual)):
        raise TypeError(f"{target!r} is not a PPODual subclass ({cls!r})")
    value = cls.HERO_KNOB_DEFAULTS.get("std_clamp_max_lower")
    return None if value is None else float(value)


class PPODual(PPO):
    """PPO with separate upper- and lower-body actors, critics, and rewards."""

    config: PPODualConfig
    actor: PPODualActor
    critic: PPODualCritic

    _LOSS_LOG_NAMES = {
        **PPO._LOSS_LOG_NAMES,
        **{f"value_loss_{g}": f"Value_{g}" for g in GROUP_KEYS},
        **{f"surrogate_loss_{g}": f"Surrogate_{g}" for g in GROUP_KEYS},
        **{f"entropy_loss_{g}": f"Entropy_{g}" for g in GROUP_KEYS},
        **{f"kl_mean_{g}": f"KL_{g}" for g in GROUP_KEYS},
    }

    HERO_KNOB_DEFAULTS: dict[str, Any] = {"std_clamp_max": None}
    """Class defaults of the knobs without a ``PPODualConfig`` field (subclasses pin them; see :class:`PPODualAnchor`)."""

    def __init__(
        self,
        env,
        config: PPODualConfig,
        log_dir,
        device="cpu",
        multi_gpu_cfg: dict | None = None,
        *,
        std_clamp_max: float | None = None,
        std_clamp_max_lower: float | None = None,
    ):
        if not isinstance(config, PPODualConfig):
            raise TypeError(f"PPODual needs a PPODualConfig, got {type(config).__name__}")
        self.group_keys: tuple[str, str] = GROUP_KEYS
        self._std_clamp_max_override = std_clamp_max
        self._std_clamp_max_lower_override = std_clamp_max_lower
        self.hero_knob_sources: dict[str, str] = {}
        # ``hero_expand`` block of the checkpoint load() was given (expand_checkpoint lineage); save() writes it back
        self._hero_expand: dict[str, Any] | None = None
        super().__init__(env, config, log_dir, device, multi_gpu_cfg)
        self._rewards_by_group_missing_warned = False
        self._actor_obs_layout: ActorObsLayout | None = None
        # Per-group episode returns (HERO ``Train/mean_reward_<key>``): accumulated rank-locally during the rollout and
        # folded across ranks by ``_synchronize_group_returns`` (all_reduce SUM of sum / count, every rank enters it at the
        # end of ``_rollout_step``) so rank 0 logs the GLOBAL mean like ``LoggingHelper`` does for the total return

        n = self.env.num_envs
        self._group_return_running = {g: torch.zeros(n, dtype=torch.float, device=self.device) for g in GROUP_KEYS}
        self._group_return_interval_sum = {
            g: torch.zeros((), dtype=torch.float, device=self.device) for g in GROUP_KEYS
        }
        self._group_return_interval_count = {g: 0 for g in GROUP_KEYS}

    # ------------------------------------------------------------------ config
    def _init_config(self) -> None:
        super()._init_config()
        split = tuple(int(n) for n in self.config.action_split)
        if len(split) != 2 or any(n <= 0 for n in split):
            raise ValueError(f"action_split must be two positive ints, got {self.config.action_split}")
        if sum(split) != self.num_act:
            raise ValueError(f"action_split {split} does not sum to the robot action dim {self.num_act}")
        self.action_split: tuple[int, int] = (split[0], split[1])
        if self.config.use_symmetry:
            raise NotImplementedError("PPODual does not support symmetry augmentation (use_symmetry=True)")
        if self.config.history_layout_export not in SUPPORTED_HISTORY_LAYOUTS:
            raise ValueError(
                f"history_layout_export={self.config.history_layout_export!r}; "
                f"expected one of {SUPPORTED_HISTORY_LAYOUTS}"
            )
        if self.config.export_motion_in_onnx:
            logger.warning("PPODual ignores export_motion_in_onnx=True: the motion corpus is never baked into the ONNX")
        self.actor_learning_rates: dict[str, float] = {g: self.actor_learning_rate for g in GROUP_KEYS}
        self.critic_learning_rates: dict[str, float] = {g: self.critic_learning_rate for g in GROUP_KEYS}
        self.std_clamp_max: float | None = self._resolve_std_clamp_max()
        if self.std_clamp_max is not None:
            lo = float(self.config.module_dict.actor_lower.min_noise_std or 0.0)
            if not (lo < self.std_clamp_max <= 10.0):
                raise ValueError(f"std_clamp_max={self.std_clamp_max} must lie in ({lo}, 10]")
        self.std_clamp_max_lower: float | None = self._resolve_std_clamp_max_lower()
        if self.std_clamp_max_lower is not None:
            lo = float(self.config.module_dict.actor_lower.min_noise_std or 0.0)
            if not (lo < self.std_clamp_max_lower <= 10.0):
                raise ValueError(f"std_clamp_max_lower={self.std_clamp_max_lower} must lie in ({lo}, 10]")
        logger.info(
            "PPODual knobs: std_clamp_max={} ({}){}", self.std_clamp_max, self.hero_knob_sources["std_clamp_max"],
            (f"; std_clamp_max_lower={self.std_clamp_max_lower} ({self.hero_knob_sources['std_clamp_max_lower']}; the upper body keeps"
             f" std_clamp_max)") if self.std_clamp_max_lower is not None else "",
        )

    @property
    def algo_target(self) -> str:
        """Dotted path of the running algo class (== the preset's ``algo._target_``)."""
        return f"{type(self).__module__}.{type(self).__qualname__}"

    def _resolve_std_clamp_max(self) -> float | None:
        """kwarg > config attribute ``hero_std_clamp_max`` > env ``HERO_PPO_DUAL_STD_CLAMP_MAX`` > class default; records the source."""
        cfg_val = getattr(self.config, STD_CLAMP_MAX_CONFIG_FIELD, None)
        env_val = _env_float(STD_CLAMP_MAX_ENV_VAR)
        if self._std_clamp_max_override is not None:
            value, source = self._std_clamp_max_override, "kwarg"
        elif cfg_val is not None:
            value, source = cfg_val, f"config.{STD_CLAMP_MAX_CONFIG_FIELD}"
        elif env_val is not None:
            value, source = env_val, f"env.{STD_CLAMP_MAX_ENV_VAR}"
        else:
            value, source = self.HERO_KNOB_DEFAULTS.get("std_clamp_max"), f"class.{type(self).__name__}"
        self.hero_knob_sources["std_clamp_max"] = source
        return None if value is None else float(value)

    def _resolve_std_clamp_max_lower(self) -> float | None:
        """Lower-body std bound: kwarg > config attribute ``hero_std_clamp_max_lower`` > env ``HERO_PPO_DUAL_STD_CLAMP_MAX_LOWER`` >
        class default ``HERO_KNOB_DEFAULTS["std_clamp_max_lower"]`` (absent = ``None`` = inherit ``std_clamp_max``).  The source is recorded
        in ``hero_knob_sources`` ONLY when the knob resolves to a value, so classes without it record only ``std_clamp_max``."""
        cfg_val = getattr(self.config, STD_CLAMP_MAX_LOWER_CONFIG_FIELD, None)
        env_val = _env_float(STD_CLAMP_MAX_LOWER_ENV_VAR)
        if self._std_clamp_max_lower_override is not None:
            value, source = self._std_clamp_max_lower_override, "kwarg"
        elif cfg_val is not None:
            value, source = cfg_val, f"config.{STD_CLAMP_MAX_LOWER_CONFIG_FIELD}"
        elif env_val is not None:
            value, source = env_val, f"env.{STD_CLAMP_MAX_LOWER_ENV_VAR}"
        else:
            value, source = self.HERO_KNOB_DEFAULTS.get("std_clamp_max_lower"), f"class.{type(self).__name__}"
        if value is None:
            return None
        self.hero_knob_sources["std_clamp_max_lower"] = source
        return float(value)

    def _init_obs_keys(self) -> None:
        md = self.config.module_dict
        actor_keys = (list(md.actor_lower.input_dim), list(md.actor_upper.input_dim))
        critic_keys = (list(md.critic_lower.input_dim), list(md.critic_upper.input_dim))
        if actor_keys[0] != actor_keys[1]:
            raise ValueError(f"actor_lower / actor_upper must read the same obs groups: {actor_keys}")
        if critic_keys[0] != critic_keys[1]:
            raise ValueError(f"critic_lower / critic_upper must read the same obs groups: {critic_keys}")
        self.actor_obs_keys = actor_keys[0]
        self.critic_obs_keys = critic_keys[0]
        if hasattr(self.env, "require_final_observation_groups"):
            self.env.require_final_observation_groups(self.critic_obs_keys)

    # ------------------------------------------------------------------ models
    def _setup_models_and_optimizer(self) -> None:
        md = self.config.module_dict
        self.actor = PPODualActor(
            obs_dim_dict=self.algo_obs_dim_dict,
            actor_lower_cfg=md.actor_lower,
            actor_upper_cfg=md.actor_upper,
            action_split=self.action_split,
            init_noise_std=self._init_noise_std_by_group(),
            history_length=self.algo_history_length_dict,
            std_clamp_max=self.std_clamp_max,
            std_clamp_max_lower=self.std_clamp_max_lower,
        ).to(self.device)
        self.critic = PPODualCritic(
            obs_dim_dict=self.algo_obs_dim_dict,
            critic_lower_cfg=md.critic_lower,
            critic_upper_cfg=md.critic_upper,
            history_length=self.algo_history_length_dict,
        ).to(self.device)

        actor_obs_dim = self._get_obs_dim(self.actor_obs_keys)
        critic_obs_dim = self._get_obs_dim(self.critic_obs_keys)
        if self.empirical_normalization:
            self.actor_obs_normalizer: nn.Module = EmpiricalNormalization(shape=actor_obs_dim, device=self.device)
            self.critic_obs_normalizer: nn.Module = EmpiricalNormalization(shape=critic_obs_dim, device=self.device)
        else:
            self.actor_obs_normalizer = nn.Identity()
            self.critic_obs_normalizer = nn.Identity()

        # Same rank-0 broadcast as PPO (iterates self.actor / self.critic parameters).
        if self.is_multi_gpu:
            self._synchronize_model_weights()

        # One optimizer per role, one param group per body group (index == GROUP_KEYS order).
        # Each group keeps its own adaptive learning rate like HERO's per-key optimizers while
        # PPO._reduce_parameters keeps seeing exactly two containers.
        actor_groups = [
            {"params": list(self.actor.group_parameters(g)), "lr": self.actor_learning_rates[g]} for g in GROUP_KEYS
        ]
        critic_groups = [
            {"params": list(self.critic.group_parameters(g)), "lr": self.critic_learning_rates[g]} for g in GROUP_KEYS
        ]
        n_actor = sum(p.numel() for grp in actor_groups for p in grp["params"])
        n_critic = sum(p.numel() for grp in critic_groups for p in grp["params"])
        if n_actor != sum(p.numel() for p in self.actor.parameters()):
            raise RuntimeError("actor group parameters do not cover every actor parameter")
        if n_critic != sum(p.numel() for p in self.critic.parameters()):
            raise RuntimeError("critic group parameters do not cover every critic parameter")
        self.actor_optimizer = instantiate(
            self.config.actor_optimizer, params=actor_groups, lr=self.actor_learning_rate
        )
        self.critic_optimizer = instantiate(
            self.config.critic_optimizer, params=critic_groups, lr=self.critic_learning_rate
        )

    def _setup_storage(self) -> None:
        self.storage = RolloutStorage(self.env.num_envs, self.config.num_steps_per_env, device=self.device)
        actor_obs_dim = self._get_obs_dim(self.actor_obs_keys)
        critic_obs_dim = self._get_obs_dim(self.critic_obs_keys)
        self.storage.register("actor_obs", shape=(actor_obs_dim,), dtype=torch.float)
        self.storage.register("critic_obs", shape=(critic_obs_dim,), dtype=torch.float)
        shared_keys = [
            ("actions", (self.num_act,), torch.float),
            ("dones", (1,), torch.bool),
            ("action_mean", (self.num_act,), torch.float),
            ("action_sigma", (self.num_act,), torch.float),
        ]
        for key, shape, dtype in shared_keys:
            self.storage.register(key, shape=shape, dtype=dtype)
        for g in GROUP_KEYS:
            for name in PER_GROUP_STORAGE_KEYS:
                self.storage.register(storage_key(name, g), shape=(1,), dtype=torch.float)

    @property
    def inference_model(self) -> dict[str, nn.Module]:
        return {"actor": self.actor, "critic": self.critic}

    # ------------------------------------------------------------------- learn
    def learn(self) -> None:
        self._sync_provenance_best_effort()
        super().learn()

    def _group_rewards(self, rewards: torch.Tensor, infos: Mapping[str, Any]) -> dict[str, torch.Tensor]:
        """``{group: [N]}`` from ``infos["rewards_by_group"]``; falls back to the total reward for both groups."""
        by_group = infos.get(REWARDS_BY_GROUP_KEY) if isinstance(infos, Mapping) else None
        if isinstance(by_group, Mapping) and all(g in by_group for g in GROUP_KEYS):
            out: dict[str, torch.Tensor] = {}
            for g in GROUP_KEYS:
                r = torch.as_tensor(by_group[g], dtype=torch.float, device=self.device).reshape(-1)
                if r.shape != rewards.shape:
                    raise ValueError(
                        f"rewards_by_group[{g!r}] has shape {tuple(r.shape)}, expected {tuple(rewards.shape)}"
                    )
                out[g] = r
            return out
        if not self._rewards_by_group_missing_warned:
            logger.warning(
                f"PPODual: env extras lack {REWARDS_BY_GROUP_KEY!r} with groups {GROUP_KEYS}; "
                "both critics fall back to the TOTAL reward (HERO grouping disabled)"
            )
            self._rewards_by_group_missing_warned = True
        return {g: rewards for g in GROUP_KEYS}

    def _rollout_step(self, obs_dict):
        with torch.inference_mode():
            for _ in range(self.config.num_steps_per_env):
                actor_obs_raw = torch.cat([obs_dict[k] for k in self.actor_obs_keys], dim=1)
                critic_obs_raw = torch.cat([obs_dict[k] for k in self.critic_obs_keys], dim=1)
                actor_obs = self._normalize_actor_obs(actor_obs_raw)
                critic_obs = self._normalize_critic_obs(critic_obs_raw)

                actions = self.actor.act({"actor_obs": actor_obs})
                values = {g: v.detach() for g, v in self.critic.evaluate({"critic_obs": critic_obs}).items()}
                log_prob = self.actor.get_actions_log_prob_by_group(actions)
                action_mean = self.actor.action_mean.detach()
                action_sigma = self.actor.action_std.detach()

                obs_dict, rewards, dones, infos = self.env.step({"actions": actions})

                for obs_key in obs_dict:
                    obs_dict[obs_key] = obs_dict[obs_key].to(self.device)
                rewards, dones = rewards.to(self.device), dones.to(self.device)
                group_rewards = self._group_rewards(rewards, infos)

                # Time-out bootstrap with each group's own critic (HERO: r_key += gamma * V_key * time_outs).
                final_rewards = {g: torch.zeros_like(rewards) for g in GROUP_KEYS}
                if infos["time_outs"].any():
                    timeout_ids = infos["time_outs"].to(self.device).nonzero(as_tuple=False).flatten()
                    final_critic_obs = torch.cat(
                        [infos["final_observations"][k][timeout_ids] for k in self.critic_obs_keys], dim=1
                    )
                    final_critic_obs = self._normalize_critic_obs(final_critic_obs, update=False)
                    for g in GROUP_KEYS:
                        final_values = self.critic.evaluate_group({"critic_obs": final_critic_obs}, g).detach()
                        final_rewards[g].index_copy_(0, timeout_ids, self.config.gamma * final_values.squeeze(1))

                transition: dict[str, torch.Tensor] = {
                    "actor_obs": actor_obs,
                    "critic_obs": critic_obs,
                    "actions": actions,
                    "dones": dones.view(-1, 1),
                    "action_mean": action_mean,
                    "action_sigma": action_sigma,
                }
                for g in GROUP_KEYS:
                    transition[storage_key("values", g)] = values[g]
                    transition[storage_key("rewards", g)] = (group_rewards[g] + final_rewards[g]).view(-1, 1)
                    transition[storage_key("actions_log_prob", g)] = log_prob[g].detach().unsqueeze(1)
                self.storage.add(**transition)

                self.actor.reset(dones)
                self.critic.reset(dones)

                if self.log_dir is not None:
                    self.logging_helper.update_episode_stats(rewards, dones, infos)
                    self._update_group_episode_stats(group_rewards, dones, infos)

            # Two GAE passes, one per reward group with its own critic (values + last value).
            last_critic_obs = torch.cat([obs_dict[k] for k in self.critic_obs_keys], dim=1)
            last_critic_obs = self._normalize_critic_obs(last_critic_obs, update=False)
            dones_buf = self.storage["dones"].to(self.device)
            for g in GROUP_KEYS:
                last_values = self.critic.evaluate_group({"critic_obs": last_critic_obs}, g).detach().to(self.device)
                returns, advantages = self._compute_returns_and_advantages(
                    last_values,
                    self.storage[storage_key("values", g)].to(self.device),
                    dones_buf,
                    self.storage[storage_key("rewards", g)].to(self.device),
                )
                self.storage[storage_key("returns", g)] = returns
                self.storage[storage_key("advantages", g)] = advantages

        # Collective: every rank folds its per-group interval returns here (the base ``learn`` runs ``_rollout_step`` on
        # all ranks, ``_post_epoch_logging`` on rank 0 only) -- see ``_synchronize_group_returns``.
        self._synchronize_group_returns()
        return obs_dict

    def _update_group_episode_stats(
        self, group_rewards: Mapping[str, torch.Tensor], dones: torch.Tensor, infos: Mapping[str, Any]
    ) -> None:
        logging_dones = infos.get("logging_dones")
        logging_dones = dones if logging_dones is None else torch.as_tensor(logging_dones, device=self.device)
        done_ids = (logging_dones > 0).nonzero(as_tuple=True)[0]
        for g in GROUP_KEYS:
            running = self._group_return_running[g]
            running += group_rewards[g]
            if done_ids.numel() > 0:
                self._group_return_interval_sum[g] += running[done_ids].sum()
                self._group_return_interval_count[g] += int(done_ids.numel())
                running[done_ids] = 0.0

    def _group_return_stats(self) -> torch.Tensor:
        """``[sum_<g0>, sum_<g1>, count_<g0>, count_<g1>]`` (``GROUP_KEYS`` order) of this rank's completed-episode group
        returns in the current logging interval; float64 so the count all-reduce stays exact (same dtype choice as
        ``PPO._normalize_advantages``)."""
        sums = torch.stack([self._group_return_interval_sum[g] for g in GROUP_KEYS]).to(torch.float64)
        counts = torch.tensor(
            [float(self._group_return_interval_count[g]) for g in GROUP_KEYS], dtype=torch.float64, device=self.device
        )
        return torch.cat([sums, counts])

    def _synchronize_group_returns(self) -> None:


        if not self.is_multi_gpu:
            return
        stats = self._group_return_stats()
        torch.distributed.all_reduce(stats, op=torch.distributed.ReduceOp.SUM)
        n = len(GROUP_KEYS)
        if self.is_main_process:
            counts = stats[n:].cpu().tolist()  # one host transfer per interval (the sums stay on device)
            for i, g in enumerate(GROUP_KEYS):
                self._group_return_interval_sum[g].copy_(stats[i])
                self._group_return_interval_count[g] = int(round(counts[i]))
        else:
            for g in GROUP_KEYS:
                self._group_return_interval_sum[g].zero_()
                self._group_return_interval_count[g] = 0

    # -------------------------------------------------------------------- loss
    def _update_algo_step(self, minibatch, loss_accum):
        loss_dict = self._compute_ppo_loss(minibatch)

        self.actor_optimizer.zero_grad()
        self.critic_optimizer.zero_grad()
        (loss_dict["actor_loss"] + loss_dict["critic_loss"]).backward()

        if self.is_multi_gpu:
            self._reduce_parameters()

        # HERO clips each actor / critic separately; the per-group parameter sets are disjoint.
        for g in GROUP_KEYS:
            nn.utils.clip_grad_norm_(list(self.actor.group_parameters(g)), self.config.max_grad_norm)
            nn.utils.clip_grad_norm_(list(self.critic.group_parameters(g)), self.config.max_grad_norm)

        self.actor_optimizer.step()
        self.critic_optimizer.step()
        # Adam may step a raw std past the upper clamp; project it back so torch.clamp keeps a live gradient
        # (no-op without std_clamp_max; see PPODualActor.project_std_).
        self.actor.project_std_()

        for key, loss in loss_dict.items():
            if key in self._LOSS_KEYS_NOT_LOGGED:
                continue
            log_key = self._LOSS_LOG_NAMES.get(key, key)
            value = loss.detach() if torch.is_tensor(loss) else loss
            loss_accum[log_key] = value if log_key not in loss_accum else loss_accum[log_key] + value
        return loss_accum

    def _compute_ppo_loss(self, minibatch) -> dict[str, torch.Tensor]:
        """Per-group clipped PPO losses; ``actor_loss`` / ``critic_loss`` are the sums over groups."""
        actions_batch = minibatch["actions"]
        actor_obs = minibatch["actor_obs"]
        critic_obs = minibatch["critic_obs"]

        self.actor.act({"actor_obs": actor_obs})
        log_prob_by_group = self.actor.get_actions_log_prob_by_group(actions_batch)
        mu_batch = self.actor.action_mean
        sigma_batch = self.actor.action_std
        entropy_by_group = self.actor.entropy_by_group
        values_by_group = self.critic.evaluate({"critic_obs": critic_obs})

        adaptive = self.config.desired_kl is not None and self.config.schedule == "adaptive"
        clip = self.config.clip_param
        zero = torch.zeros((), device=actor_obs.device)
        out: dict[str, torch.Tensor] = {}
        totals = {"value_loss": zero, "surrogate_loss": zero, "entropy_loss": zero, "kl_mean": zero}
        actor_loss = zero
        critic_loss = zero
        for g in GROUP_KEYS:
            sl = self.actor.group_slices[g]
            old_log_prob = minibatch[storage_key("actions_log_prob", g)].squeeze(-1)
            advantages = minibatch[storage_key("advantages", g)].squeeze(-1)
            returns = minibatch[storage_key("returns", g)]
            target_values = minibatch[storage_key("values", g)]
            old_mu = minibatch["action_mean"][:, sl]
            old_sigma = minibatch["action_sigma"][:, sl]

            if adaptive:
                # Same all-reduce as PPO._compute_kl_div, once per group, fixed order on every rank.
                kl_mean = self._compute_kl_div(old_mu, old_sigma, mu_batch[:, sl], sigma_batch[:, sl])
                self._update_group_learning_rate(g, kl_mean)
            else:
                kl_mean = zero

            ratio = torch.exp(log_prob_by_group[g] - old_log_prob)
            surrogate = -advantages * ratio
            surrogate_clipped = -advantages * torch.clamp(ratio, 1.0 - clip, 1.0 + clip)
            surrogate_loss = torch.max(surrogate, surrogate_clipped).mean()

            value_batch = values_by_group[g]
            value_clipped = target_values + (value_batch - target_values).clamp(-clip, clip)
            value_losses = (value_batch - returns).pow(2)
            value_losses_clipped = (value_clipped - returns).pow(2)
            value_loss = torch.max(value_losses, value_losses_clipped).mean()

            entropy_loss = entropy_by_group[g].mean()

            actor_loss = actor_loss + surrogate_loss - self.config.entropy_coef * entropy_loss
            critic_loss = critic_loss + self.config.value_loss_coef * value_loss

            out[f"value_loss_{g}"] = value_loss
            out[f"surrogate_loss_{g}"] = surrogate_loss
            out[f"entropy_loss_{g}"] = entropy_loss
            out[f"kl_mean_{g}"] = kl_mean
            totals["value_loss"] = totals["value_loss"] + value_loss
            totals["surrogate_loss"] = totals["surrogate_loss"] + surrogate_loss
            totals["entropy_loss"] = totals["entropy_loss"] + entropy_loss
            totals["kl_mean"] = totals["kl_mean"] + kl_mean

        out["actor_loss"] = actor_loss
        out["critic_loss"] = critic_loss
        out.update(totals)  # joint entropy / KL of independent Gaussians = sum over groups
        return out

    def _update_group_learning_rate(self, group: str, kl_mean: torch.Tensor | float) -> None:
        """Apply the adaptive learning-rate schedule to one parameter group."""
        kl = float(kl_mean)
        idx = GROUP_KEYS.index(group)
        actor_lr = self.actor_learning_rates[group]
        critic_lr = self.critic_learning_rates[group]
        if kl > self.config.desired_kl * 2.0:
            actor_lr = max(self.min_actor_learning_rate, actor_lr / 1.5)
            critic_lr = max(self.min_critic_learning_rate, critic_lr / 1.5)
        elif 0.0 < kl < self.config.desired_kl / 2.0:
            actor_lr = min(self.max_actor_learning_rate, actor_lr * 1.5)
            critic_lr = min(self.max_critic_learning_rate, critic_lr * 1.5)
        self.actor_learning_rates[group] = actor_lr
        self.critic_learning_rates[group] = critic_lr
        self.actor_optimizer.param_groups[idx]["lr"] = actor_lr
        self.critic_optimizer.param_groups[idx]["lr"] = critic_lr
        self._sync_scalar_learning_rates()

    def _update_learning_rate(self, kl_mean: torch.Tensor) -> None:  # pragma: no cover - not used by PPODual
        raise RuntimeError("PPODual schedules learning rates per group; use _update_group_learning_rate")

    def _sync_scalar_learning_rates(self) -> None:
        """Keep the stock scalar attributes (logging / metadata) as the mean over groups."""
        self.actor_learning_rate = sum(self.actor_learning_rates.values()) / len(GROUP_KEYS)
        self.critic_learning_rate = sum(self.critic_learning_rates.values()) / len(GROUP_KEYS)

    def _set_learning_rates(self, actor_lrs: Mapping[str, float], critic_lrs: Mapping[str, float]) -> None:
        for idx, g in enumerate(GROUP_KEYS):
            self.actor_learning_rates[g] = float(actor_lrs[g])
            self.critic_learning_rates[g] = float(critic_lrs[g])
            self.actor_optimizer.param_groups[idx]["lr"] = self.actor_learning_rates[g]
            self.critic_optimizer.param_groups[idx]["lr"] = self.critic_learning_rates[g]
        self._sync_scalar_learning_rates()

    # ----------------------------------------------------------------- logging
    def _post_epoch_logging(self, it, loss_dict):
        std_by_group = self.actor.std_by_group
        policy: dict[str, float] = {"mean_noise_std": float(self.actor.std.mean().item())}
        for g in GROUP_KEYS:
            policy[f"mean_noise_std_{g}"] = float(std_by_group[g].mean().item())
        # effective (clamped) std per group + upper-bound occupancy / bound value when a clamp is active
        policy.update(self.actor.noise_std_stats())
        extra_log_dicts: dict[str, dict[str, float]] = {"Policy": policy}

        # Global (all-rank) means: ``_synchronize_group_returns`` folded every rank's interval sums / counts into these
        # accumulators at the end of the rollout (rank-local == global without DDP).
        train: dict[str, float] = {}
        for g in GROUP_KEYS:
            count = self._group_return_interval_count[g]
            if count > 0:
                train[f"mean_reward_{g}"] = float(self._group_return_interval_sum[g].item()) / count
            self._group_return_interval_sum[g].zero_()
            self._group_return_interval_count[g] = 0
        if train:
            extra_log_dicts["Train"] = train

        if self.is_multi_gpu:
            extra_log_dicts["Numerics"] = {
                "skipped_gradient_updates": float(self.nonfinite_gradient_updates),
                "nonfinite_gradient_elements": float(self.nonfinite_gradient_elements),
                "nonfinite_gradient_rank_events": float(self.nonfinite_gradient_rank_events),
                "postreduce_nonfinite_gradient_elements": float(self.postreduce_nonfinite_gradient_elements),
            }

        for g in GROUP_KEYS:
            loss_dict[f"actor_learning_rate_{g}"] = self.actor_learning_rates[g]
            loss_dict[f"critic_learning_rate_{g}"] = self.critic_learning_rates[g]
        loss_dict["actor_learning_rate"] = self.actor_learning_rate
        loss_dict["critic_learning_rate"] = self.critic_learning_rate
        self.logging_helper.post_epoch_logging(it=it, loss_dict=loss_dict, extra_log_dicts=extra_log_dicts)

    # ------------------------------------------------------------ checkpoints
    def save(self, path, infos=None):
        checkpoint: dict[str, Any] = {
            "algo": "PPODual",
            "body_keys": list(GROUP_KEYS),
            "action_split": list(self.action_split),
            "actor_lower": self.actor.group_state_dict(LOWER_BODY),
            "actor_upper": self.actor.group_state_dict(UPPER_BODY),
            "critic_lower": self.critic.critic_lower.state_dict(),
            "critic_upper": self.critic.critic_upper.state_dict(),
            "actor_optimizer_state_dict": self.actor_optimizer.state_dict(),
            "critic_optimizer_state_dict": self.critic_optimizer.state_dict(),
            "actor_learning_rates": dict(self.actor_learning_rates),
            "critic_learning_rates": dict(self.critic_learning_rates),
            "std_clamp_max": self.std_clamp_max,
            "std_clamp_max_lower": self.std_clamp_max_lower,
            "algo_target": self.algo_target,
            "actor_obs_normalizer_state_dict": (
                self.actor_obs_normalizer.state_dict() if self.empirical_normalization else None
            ),
            "critic_obs_normalizer_state_dict": (
                self.critic_obs_normalizer.state_dict() if self.empirical_normalization else None
            ),
            "iter": self.current_learning_iteration,
            "infos": infos,
        }
        if self._hero_expand:
            # fine-tune lineage: the expand_checkpoint block of the checkpoint this run was loaded from (see load())
            checkpoint[HERO_EXPAND_KEY] = dict(self._hero_expand)
        checkpoint.update(self._checkpoint_metadata(iteration=self.current_learning_iteration))
        env_state = self._collect_env_state()
        if env_state:
            checkpoint["env_state"] = env_state
        self.logging_helper.save_checkpoint_artifact(checkpoint, path)

    def load(self, ckpt_path: str | None) -> dict | None:
        if ckpt_path is None:
            return None
        logger.info(f"Loading checkpoint from {ckpt_path}")
        loaded = torch.load(ckpt_path, map_location=self.device)
        self.load_model_state(loaded)
        stored_clamp = loaded.get("std_clamp_max")
        if isinstance(stored_clamp, (int, float)) and self.std_clamp_max is not None and abs(float(stored_clamp) - self.std_clamp_max) > 1e-9:

            # class with a tighter one (PPODualAnchorPin: 0.5) -- the stored value is informational; project_std_() below re-clamps
            # the loaded raw std in place, so the run starts at the new bound instead of refusing the file.
            logger.info(
                "PPODual: checkpoint std_clamp_max={} differs from this run's {} ({}); the loaded std is re-projected under the new clamp",
                stored_clamp, self.std_clamp_max, self.hero_knob_sources.get("std_clamp_max"),
            )
        stored_lower = loaded.get("std_clamp_max_lower")
        if (stored_lower if isinstance(stored_lower, (int, float)) else None) != self.std_clamp_max_lower and (
            self.std_clamp_max_lower is not None or isinstance(stored_lower, (int, float))
        ):

            # project_std_() below re-clamps the lower-body std under the head's bound
            logger.info(
                "PPODual: checkpoint std_clamp_max_lower={} differs from this run's {} ({}); the loaded lower-body std is re-projected under {}",
                stored_lower, self.std_clamp_max_lower, self.hero_knob_sources.get("std_clamp_max_lower", "inherits std_clamp_max"),
                self.actor.std_clamp_by_group[LOWER_BODY],
            )
        if self.actor.project_std_():
            # .detach(): std_lower / std_upper are Parameters (requires_grad); a bare float() warns on every load
            logger.info(
                "PPODual: loaded std projected under std_clamp_max={} (lower mean {:.4f}, upper mean {:.4f})",
                self.actor.std_clamp_by_group if self.actor.per_head_clamp else self.std_clamp_max,
                float(self.actor.std_lower.detach().mean().item()),
                float(self.actor.std_upper.detach().mean().item()),
            )
        hx = loaded.get(HERO_EXPAND_KEY)
        if isinstance(hx, Mapping):
            # keep the block: save() writes it into every checkpoint of this run; sidecar / provenance carry its summary
            self._hero_expand = dict(hx)
            logger.info(
                "PPODual: checkpoint was expanded by expand_checkpoint from {} (sha256 {}..., new terms {}, columns {})",
                hx.get("from"),
                str(hx.get("sha256"))[:16],
                hx.get("new_terms"),
                hx.get("columns"),
            )
            chain = hx.get("lineage")
            if isinstance(chain, (list, tuple)) and len(chain) > 1:
                # Record prior training configurations in chronological order.
                root = str((chain[0] or {}).get("src_exp") or "?") if isinstance(chain[0], Mapping) else "?"
                hops = [f"{e.get('exp') or '?'} (iter {e.get('iter')})" for e in chain if isinstance(e, Mapping)]
                logger.info("PPODual: expansion lineage {}", " -> ".join([root, *hops]))
        else:
            self._hero_expand = None
        if self.empirical_normalization and loaded.get("actor_obs_normalizer_state_dict") is not None:
            self.actor_obs_normalizer.load_state_dict(loaded["actor_obs_normalizer_state_dict"])
        if self.empirical_normalization and loaded.get("critic_obs_normalizer_state_dict") is not None:
            self.critic_obs_normalizer.load_state_dict(loaded["critic_obs_normalizer_state_dict"])
        if self.config.load_optimizer and "actor_optimizer_state_dict" in loaded and "actor_lower" in loaded:
            self.actor_optimizer.load_state_dict(loaded["actor_optimizer_state_dict"])
            self.critic_optimizer.load_state_dict(loaded["critic_optimizer_state_dict"])
            actor_lrs = loaded.get("actor_learning_rates") or {
                g: self.actor_optimizer.param_groups[i]["lr"] for i, g in enumerate(GROUP_KEYS)
            }
            critic_lrs = loaded.get("critic_learning_rates") or {
                g: self.critic_optimizer.param_groups[i]["lr"] for i, g in enumerate(GROUP_KEYS)
            }
            self._set_learning_rates(actor_lrs, critic_lrs)
            logger.info(
                f"Optimizer loaded; actor lr {self.actor_learning_rates}, critic lr {self.critic_learning_rates}"
            )
        if "iter" in loaded:
            # Written AFTER iteration ``iter`` completed -> resume at ``iter + 1`` (same as PPO.load).
            self.current_learning_iteration = int(loaded["iter"]) + 1
        self._restore_env_state(loaded.get("env_state"))
        return loaded.get("infos")

    def load_model_state(self, loaded: Mapping[str, Any]) -> None:
        """Load the four modules from a PPODual checkpoint or a HERO ``PPOMultiActorCritic`` checkpoint."""
        if all(k in loaded for k in CHECKPOINT_MODULE_KEYS):
            self.actor.load_group_state_dict(LOWER_BODY, loaded["actor_lower"])
            self.actor.load_group_state_dict(UPPER_BODY, loaded["actor_upper"])
            self.critic.critic_lower.load_state_dict(loaded["critic_lower"])
            self.critic.critic_upper.load_state_dict(loaded["critic_upper"])
            return
        actors = loaded.get("actor_model_state_dict")
        if isinstance(actors, Mapping) and all(g in actors for g in GROUP_KEYS):
            # HERO format: {"lower_body": PPOActor sd, "upper_body": PPOActor sd}.
            self.actor.load_hero_actor_state_dicts(actors)
            critics = loaded.get("critic_model_state_dict") or {}
            for g, module in ((LOWER_BODY, self.critic.critic_lower), (UPPER_BODY, self.critic.critic_upper)):
                if g not in critics:
                    continue
                try:
                    module.load_state_dict(critics[g])
                except (RuntimeError, KeyError) as exc:  # Keep the current critics when the source checkpoint uses a different observation layout.
                    logger.warning(f"HERO critic {g} not loaded (shape/key mismatch): {exc}")
            return
        raise KeyError(
            f"checkpoint has neither {CHECKPOINT_MODULE_KEYS} nor HERO actor_model_state_dict[{GROUP_KEYS}] "
            f"(keys: {sorted(loaded.keys())})"
        )

    def hero_expand_summary(self) -> dict[str, Any] | None:
        """Lineage subset (:data:`HERO_EXPAND_SUMMARY_KEYS`) of the loaded checkpoint's ``hero_expand`` block for the sidecar /
        provenance; None when this run was not loaded from an expanded checkpoint."""
        if not self._hero_expand:
            return None
        return {k: self._hero_expand.get(k) for k in HERO_EXPAND_SUMMARY_KEYS if k in self._hero_expand}

    def _hero_future_steps(self, term: str) -> list[int] | None:


        if term not in HERO_PLUS_PER_FRAME_DIM:
            return None
        group = self.actor_obs_keys[0] if len(self.actor_obs_keys) == 1 else "actor_obs"
        key = HERO_FUTURE_STEPS_KEYS.get(term, f"{term[:3]}_future_steps")
        steps: list[int] | None = None
        sources = (getattr(getattr(self.env, "observation_manager", None), "cfg", None), getattr(self, "_experiment_config", None))
        for source in sources:
            try:
                steps = hero_term_future_steps(source, term, group=group)
            except Exception as exc:  # noqa: BLE001 - metadata must never break an export
                logger.warning(f"PPODual: {key} lookup skipped ({type(exc).__name__}: {exc})")
                steps = None
            if steps is not None:
                break
        try:
            layout = self._resolve_actor_obs_layout()
        except Exception:  # noqa: BLE001
            layout = None
        if layout is not None and term in layout.terms:
            dim = int(layout.term_dims[layout.terms.index(term)])
            per_frame = int(HERO_PLUS_PER_FRAME_DIM[term])
            if steps is None:
                logger.warning(
                    f"PPODual: actor layout carries {term} ({dim}) but its future_steps could not be resolved; "
                    f"sidecar {key} stays null (the sim2sim reader will warn and assume its default)"
                )
            elif len(steps) * per_frame != dim:
                logger.warning(
                    f"PPODual: {key} {steps} ({len(steps)} frames x {per_frame}) disagree with the exported {term} dim {dim}"
                )
        return steps

    def _h20_future_steps(self) -> list[int] | None:


        return self._hero_future_steps(HERO_ANCHOR_TERM_NAME)

    def _hero_anchor_terms(self) -> dict[str, dict[str, Any]] | None:


        try:
            layout = self._resolve_actor_obs_layout()
        except Exception:  # noqa: BLE001
            layout = None
        if layout is None:
            return None
        term_dims = dict(zip(layout.terms, (int(d) for d in layout.term_dims)))
        group = self.actor_obs_keys[0] if len(self.actor_obs_keys) == 1 else "actor_obs"
        block: dict[str, dict[str, Any]] | None = None
        sources = (getattr(getattr(self.env, "observation_manager", None), "cfg", None), getattr(self, "_experiment_config", None))
        for source in sources:
            try:
                block = hero_anchor_terms_block(source, term_dims, group=group)
            except Exception as exc:  # noqa: BLE001 - metadata must never break an export
                logger.warning(f"PPODual: {HERO_ANCHOR_TERMS_KEY} lookup skipped ({type(exc).__name__}: {exc})")
                block = None
            # a source that does not carry the terms yields None entries; prefer the first source that resolves every horizon
            if block is not None and all(e.get("noise") is not None for e in block.values()):
                break
        return block

    # ------------------------------------------------------------------ export
    def _resolve_actor_obs_layout(self) -> ActorObsLayout | None:
        """Per-term layout of the (single, concatenated) actor obs group; cached after the first call."""
        if self._actor_obs_layout is not None:
            return self._actor_obs_layout
        if len(self.actor_obs_keys) != 1:
            return None
        self._actor_obs_layout = layout_from_env(self.env, group_name=self.actor_obs_keys[0])
        return self._actor_obs_layout

    def _action_contract(self) -> str:
        contract = getattr(self.env, "action_contract", None)
        if not contract:
            action_manager = getattr(self.env, "action_manager", None)
            contract = getattr(action_manager, "action_contract", None)
        return str(contract) if contract else ACTION_CONTRACT_HERO_RESIDUAL_UPPER

    def _init_noise_std_by_group(self) -> dict[str, float]:
        return {LOWER_BODY: float(self.config.init_noise_std), UPPER_BODY: float(self.config.init_noise_std_upper)}

    def _onnx_deployment_metadata_dual(self, layout: ActorObsLayout | None, history_layout: str) -> dict[str, Any]:
        metadata = self._onnx_deployment_metadata()  # dof_names/kp/kd/action_scale/default_dof_pos/urdf/layout...
        contract = self._action_contract()
        if layout is not None:
            metadata.update(
                dual_export_metadata(
                    layout,
                    history_layout,
                    self.action_split,
                    base_actor_obs_layout=metadata.get("actor_obs_layout"),
                    action_contract=contract,
                )
            )
        else:
            metadata.update(
                {
                    "history_layout": history_layout,
                    "onnx_inputs": list(ONNX_INPUT_NAMES),
                    "onnx_output": ONNX_OUTPUT_NAME,
                    "body_keys": list(GROUP_KEYS),
                    "action_split": list(self.action_split),
                    "action_contract": contract,
                    "algo": "PPODual",
                }
            )
        dof_names = list(metadata.get("dof_names") or [])
        if len(dof_names) == self.num_act:
            n_lower = self.action_split[0]
            metadata["action_groups"] = {LOWER_BODY: dof_names[:n_lower], UPPER_BODY: dof_names[n_lower:]}
        metadata["init_noise_std"] = self._init_noise_std_by_group()
        metadata["std_clamp_max"] = self.std_clamp_max
        metadata["std_clamp_max_lower"] = self.std_clamp_max_lower  # Per-head clamp (None inherits std_clamp_max).

        metadata[HERO_H20_FUTURE_STEPS_KEY] = self._h20_future_steps()

        metadata[HERO_H21_FUTURE_STEPS_KEY] = self._hero_future_steps("h21_ref_root_rot_b")
        metadata[HERO_H22_FUTURE_STEPS_KEY] = self._hero_future_steps("h22_ref_root_height_b")
        metadata[HERO_ANCHOR_TERMS_KEY] = self._hero_anchor_terms()
        return metadata

    def _hero_sidecar(self, metadata: dict[str, Any], onnx_file_path: str) -> dict[str, Any]:
        """``model_XXXXX_hero.json`` content (``export.build_hero_sidecar``); every env lookup is best effort."""
        exp = getattr(self, "_experiment_config", None)
        preset = getattr(getattr(exp, "training", None), "name", None)
        robot_cfg = getattr(self.env, "robot_config", None)
        effort = getattr(robot_cfg, "dof_effort_limit_list", None)
        lower = getattr(robot_cfg, "dof_pos_lower_limit_list", None)
        upper = getattr(robot_cfg, "dof_pos_upper_limit_list", None)
        limits = (list(lower), list(upper)) if lower is not None and upper is not None else None
        object_urdf = getattr(getattr(robot_cfg, "object", None), "object_urdf_path", None)
        dt = getattr(self.env, "dt", None)
        detail = None
        meta_fn = getattr(getattr(self.env, "hero_joint_action_term", None), "action_contract_metadata", None)
        if callable(meta_fn):
            try:
                detail = dict(meta_fn())
            except Exception as exc:  # noqa: BLE001 - metadata must never break an export
                logger.warning(f"PPODual: action_contract_metadata skipped ({type(exc).__name__}: {exc})")
        clip = None
        obs_cfg = getattr(getattr(self.env, "observation_manager", None), "cfg", None)
        if obs_cfg is not None and hasattr(obs_cfg, "clip_observations"):
            try:
                clip = float(obs_cfg.clip_observations)
            except (TypeError, ValueError):
                clip = None
        return build_hero_sidecar(
            metadata,
            onnx_file_path,
            preset=str(preset) if preset else None,
            effort_limit=list(effort) if effort is not None else None,
            dof_pos_limits=limits,
            object_urdf_path=str(object_urdf) if object_urdf else None,
            policy_dt=float(dt) if isinstance(dt, (int, float)) else None,
            action_contract_detail=detail,
            clip_observations=clip,
            hero_command=hero_command_block(exp),
            wandb_run_path=getattr(self, "_wandb_run_path", None),
            h20_future_steps=metadata.get(HERO_H20_FUTURE_STEPS_KEY),  # resolved once in _onnx_deployment_metadata_dual
            hero_expand=self.hero_expand_summary(),  # fine-tune lineage (None from scratch)
            h21_future_steps=metadata.get(HERO_H21_FUTURE_STEPS_KEY),
            h22_future_steps=metadata.get(HERO_H22_FUTURE_STEPS_KEY),
            anchor_terms=metadata.get(HERO_ANCHOR_TERMS_KEY),
            extra={
                "std_clamp_max": self.std_clamp_max,
                "std_clamp_max_lower": self.std_clamp_max_lower,  # Lower-body clamp; None inherits std_clamp_max.
                "algo_target": self.algo_target,
            },
        )

    def export(self, onnx_file_path: str) -> None:
        """Dual-input ONNX (``actor_obs_lower_body`` / ``actor_obs_upper_body`` -> ``action``), HERO layout,
        plus the ``model_XXXXX_hero.json`` sidecar (:func:`hero_sidecar_path`).

        Never bakes the motion corpus into the graph (``PPO.export`` would when a
        ``motion_command`` exists).  Written, annotated and validated in a staging dir, then
        published with one rename each (same contract as ``PPO.export``); the ONNX is published
        BEFORE the sidecar, so a visible sidecar implies a complete ONNX."""
        was_training = self.actor.training
        self._eval_mode()

        history_layout = self.config.history_layout_export
        layout = self._resolve_actor_obs_layout()
        if layout is None and history_layout == HISTORY_LAYOUT_FRAME_MAJOR:
            raise ValueError(
                "frame_major_hero_v1 export needs exactly one concatenated actor obs group; "
                f"actor_obs_keys={self.actor_obs_keys}"
            )
        input_perm = input_perm_for_layout(layout, history_layout) if layout is not None else None
        wrapper = DualActorOnnxWrapper(
            self.actor,
            input_perm=input_perm,
            obs_normalizer=self.actor_obs_normalizer,
            empirical_normalization=self.empirical_normalization,
        ).to(self.device)

        sidecar_path = hero_sidecar_path(onnx_file_path)
        with tempfile.TemporaryDirectory(prefix="hero_ppo_dual_onnx_export_") as staging_dir:
            staged_path = os.path.join(staging_dir, os.path.basename(onnx_file_path))
            export_dual_actor_as_onnx(wrapper, staged_path, self._get_zero_input())
            metadata = self._onnx_deployment_metadata_dual(layout, history_layout)
            attach_onnx_metadata(onnx_path=staged_path, metadata=metadata)
            validate_onnx_deployment_metadata(staged_path)
            staged_sidecar = os.path.join(staging_dir, os.path.basename(sidecar_path))
            Path(staged_sidecar).write_text(json.dumps(self._hero_sidecar(metadata, onnx_file_path), indent=2, default=str))
            publish_onnx_atomically(staged_path, onnx_file_path)
            publish_onnx_atomically(staged_sidecar, sidecar_path)

        self.logging_helper.save_to_wandb(onnx_file_path)
        self.logging_helper.save_to_wandb(sidecar_path)
        if was_training:
            self._train_mode()

    # -------------------------------------------------------------- provenance
    def _find_corpus_manifest(self) -> str | None:
        env_path = os.environ.get("HERO_CORPUS_MANIFEST")
        if env_path and Path(env_path).is_file():
            return env_path
        command_manager = getattr(self.env, "command_manager", None)
        if command_manager is None:
            return None
        try:
            motion_command = command_manager.get_state("motion_command")
        except Exception:  # noqa: BLE001 - best effort
            return None
        cfg = getattr(motion_command, "config", None) or getattr(motion_command, "cfg", None)
        motion_cfg = getattr(cfg, "motion_config", None)
        motion_dir = getattr(cfg, "motion_dir", None) or getattr(motion_cfg, "motion_dir", None)
        if not motion_dir:
            return None
        for candidate in (Path(motion_dir) / "CORPUS_MANIFEST.json", Path(motion_dir).parent / "CORPUS_MANIFEST.json"):
            if candidate.is_file():
                return str(candidate)
        return None

    def _sync_provenance_best_effort(self) -> None:

        if not self.is_main_process:
            return
        try:
            from hero_isaacsim.logging import provenance as P

            exp = self._experiment_config
            training = getattr(exp, "training", None)
            arm = str(getattr(training, "name", None) or getattr(exp, "env_class", None) or "ppo_dual")
            seed = int(getattr(training, "seed", 0) or 0)
            md = self.config.module_dict
            network_sizes = {
                name: list(getattr(getattr(getattr(md, name), "layer_config", None), "hidden_dims", []) or [])
                for name in CHECKPOINT_MODULE_KEYS
            }
            layout: ActorObsLayout | None
            try:
                layout = self._resolve_actor_obs_layout()
            except Exception:  # noqa: BLE001
                layout = None
            obs_layout: dict[str, Any] = {
                "actor_obs_dim": self._get_obs_dim(self.actor_obs_keys),
                "critic_obs_dim": self._get_obs_dim(self.critic_obs_keys),
                "actor_history_length": self.algo_history_length_dict.get("actor_obs"),
                "critic_history_length": self.algo_history_length_dict.get("critic_obs"),
                "layout_train": P.OBS_HISTORY_LAYOUT_TRAIN,
                "layout_export": self.config.history_layout_export,
            }
            if layout is not None:
                obs_layout["actor_terms"] = list(layout.terms)
                obs_layout["actor_term_dims"] = list(layout.term_dims)
            envs_per_rank = int(self.env.num_envs)
            extra = {
                "algo": "PPODual",
                "body_keys": list(GROUP_KEYS),
                "action_split": list(self.action_split),
                "init_noise_std": self._init_noise_std_by_group(),
                "std_clamp_max": self.std_clamp_max,
                "std_clamp_max_lower": self.std_clamp_max_lower,
                "hero_knob_sources": dict(self.hero_knob_sources),
                "algo_target": self.algo_target,
                "num_steps_per_env": self.config.num_steps_per_env,
                "num_learning_iterations": self.config.num_learning_iterations,
                "num_mini_batches": self.config.num_mini_batches,
                "num_learning_epochs": self.config.num_learning_epochs,
                "actor_learning_rate": self.config.actor_learning_rate,
                "critic_learning_rate": self.config.critic_learning_rate,
                "desired_kl": self.config.desired_kl,
                "empirical_normalization": self.config.empirical_normalization,
                "log_dir": str(self.log_dir),
                "wandb_run_path": self._wandb_run_path,

                "hero_expand": self.hero_expand_summary(),
                HERO_H20_FUTURE_STEPS_KEY: self._h20_future_steps(),

                HERO_H21_FUTURE_STEPS_KEY: self._hero_future_steps("h21_ref_root_rot_b"),
                HERO_H22_FUTURE_STEPS_KEY: self._hero_future_steps("h22_ref_root_height_b"),
                HERO_ANCHOR_TERMS_KEY: self._hero_anchor_terms(),
            }
            extra.update(P.runtime_robot_asset_provenance(self.env))
            prov = P.collect_provenance(
                arm=arm,
                corpus_manifest=self._find_corpus_manifest(),
                seed=seed,
                global_num_envs=envs_per_rank * int(self.gpu_world_size),
                envs_per_rank=envs_per_rank,
                network_sizes=network_sizes,
                obs_layout=obs_layout,
                action_contract=self._action_contract(),
                extra=extra,
            )
            P.sync_once(prov, self.log_dir)
        except Exception as exc:  # noqa: BLE001 - provenance must never break training
            logger.warning(f"PPODual: provenance sync skipped ({type(exc).__name__}: {exc})")


class PPODualAnchor(PPODual):


    HERO_KNOB_DEFAULTS: dict[str, Any] = {"std_clamp_max": ANCHOR_STD_CLAMP_MAX}


class PPODualAnchorPin(PPODualAnchor):


    HERO_KNOB_DEFAULTS: dict[str, Any] = {"std_clamp_max": PIN_STD_CLAMP_MAX}


class PPODualAnchorPinLowStd(PPODualAnchorPin):


    HERO_KNOB_DEFAULTS: dict[str, Any] = {"std_clamp_max": PIN_STD_CLAMP_MAX, "std_clamp_max_lower": PIN_LOWSTD_STD_CLAMP_MAX_LOWER}


__all__ = [
    "ANCHOR_STD_CLAMP_MAX",
    "PIN_STD_CLAMP_MAX",
    "CHECKPOINT_MODULE_KEYS",
    "HERO_EXPAND_KEY",
    "HERO_EXPAND_SUMMARY_KEYS",
    "HERO_KNOB_NAMES",
    "HERO_HEAD_KNOB_NAMES",
    "PIN_LOWSTD_STD_CLAMP_MAX_LOWER",
    "PER_GROUP_STORAGE_KEYS",
    "PPO_DUAL_ANCHOR_PIN_LOWSTD_TARGET",
    "PPO_DUAL_ANCHOR_PIN_TARGET",
    "PPO_DUAL_ANCHOR_TARGET",
    "PPO_DUAL_TARGET",
    "REWARDS_BY_GROUP_KEY",
    "STD_CLAMP_MAX_CONFIG_FIELD",
    "STD_CLAMP_MAX_ENV_VAR",
    "STD_CLAMP_MAX_LOWER_CONFIG_FIELD",
    "STD_CLAMP_MAX_LOWER_ENV_VAR",
    "PPODual",
    "PPODualAnchor",
    "PPODualAnchorPin",
    "PPODualAnchorPinLowStd",
    "std_clamp_max_for_target",
    "std_clamp_max_lower_for_target",
    "storage_key",
]
