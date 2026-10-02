"""Upper- and lower-body actor and critic networks for HERO."""

from __future__ import annotations

import dataclasses
from collections.abc import Iterator, Mapping
from typing import Any

import torch
from torch import nn
from torch.distributions import Normal

from holosoma.agents.modules.modules import BaseModule
from holosoma.agents.modules.ppo_modules import PPOCritic
from holosoma.config_types.algo import ModuleConfig

LOWER_BODY = "lower_body"
UPPER_BODY = "upper_body"
GROUP_KEYS: tuple[str, str] = (LOWER_BODY, UPPER_BODY)
"""HERO ``body_keys`` order; also the concatenation order of the action vector."""

GROUP_SUFFIX: dict[str, str] = {LOWER_BODY: "lb", UPPER_BODY: "ub"}
"""Storage-key suffixes (``values_lb``, ``returns_ub`` ...)."""


def _resolve_output_dim(cfg: ModuleConfig, num_actions: int) -> ModuleConfig:
    """Return a copy of ``cfg`` with ``"robot_action_dim"`` replaced by ``num_actions``.

    Stock ``PPOActor`` mutates the config's ``output_dim`` list in place; presets are
    shared objects, so we replace instead."""
    output_dim = [num_actions if o == "robot_action_dim" else o for o in cfg.output_dim]
    return dataclasses.replace(cfg, output_dim=output_dim)


class PPODualActor(nn.Module):
    """Two MLP actors over the same observation; Gaussian policy with per-group std parameters.

    Parameters
    ----------
    obs_dim_dict : mapping of observation group -> total dim (history included), as produced by
        ``ObservationManager.get_obs_dims()``.
    actor_lower_cfg, actor_upper_cfg : ``ModuleConfig`` (type ``MLP``, ``input_dim=["actor_obs"]``).
    action_split : ``(num_lower, num_upper)``; contiguous slices of the dof order.
    init_noise_std : float (both groups) or ``{"lower_body": s_l, "upper_body": s_u}``.
    history_length : forwarded to ``BaseModule`` (unused for MLP, kept for API parity).
    std_clamp_max : upper clamp of the exploration std of BOTH groups (``None`` = no upper clamp, HERO original).
    std_clamp_max_lower : upper clamp of the LOWER-body std only (``None`` = same as ``std_clamp_max``); the upper body keeps ``std_clamp_max``."""

    def __init__(
        self,
        obs_dim_dict: Mapping[str, Any],
        actor_lower_cfg: ModuleConfig,
        actor_upper_cfg: ModuleConfig,
        action_split: tuple[int, int],
        init_noise_std: float | Mapping[str, float],
        history_length: Mapping[str, int] | None = None,
        std_clamp_max: float | None = None,
        std_clamp_max_lower: float | None = None,
    ):
        super().__init__()
        if len(action_split) != 2 or any(int(n) <= 0 for n in action_split):
            raise ValueError(f"action_split must be two positive ints, got {action_split}")
        n_lower, n_upper = (int(action_split[0]), int(action_split[1]))
        self.action_split: tuple[int, int] = (n_lower, n_upper)
        self.num_actions = n_lower + n_upper
        self.num_actions_by_group: dict[str, int] = {LOWER_BODY: n_lower, UPPER_BODY: n_upper}
        self.group_slices: dict[str, slice] = {
            LOWER_BODY: slice(0, n_lower),
            UPPER_BODY: slice(n_lower, n_lower + n_upper),
        }
        history_length = dict(history_length) if history_length is not None else {}

        if list(actor_lower_cfg.input_dim) != list(actor_upper_cfg.input_dim):
            raise ValueError(
                "both actors must read the same observation groups: "
                f"{list(actor_lower_cfg.input_dim)} != {list(actor_upper_cfg.input_dim)}"
            )
        self.input_keys: list[str] = list(actor_lower_cfg.input_dim)

        cfg_l = _resolve_output_dim(actor_lower_cfg, n_lower)
        cfg_u = _resolve_output_dim(actor_upper_cfg, n_upper)
        for name, cfg in (("actor_lower", cfg_l), ("actor_upper", cfg_u)):
            if cfg.type != "MLP":
                raise ValueError(f"{name}: PPODualActor supports type='MLP' only, got {cfg.type!r}")
        self.actor_lower = BaseModule(obs_dim_dict, cfg_l, history_length)
        self.actor_upper = BaseModule(obs_dim_dict, cfg_u, history_length)
        if self.actor_lower.output_dim != n_lower or self.actor_upper.output_dim != n_upper:
            raise ValueError(
                f"actor output dims ({self.actor_lower.output_dim}, {self.actor_upper.output_dim}) "
                f"!= action_split {self.action_split}"
            )

        if isinstance(init_noise_std, Mapping):
            missing = [g for g in GROUP_KEYS if g not in init_noise_std]
            if missing:
                raise ValueError(f"init_noise_std missing groups {missing}")
            std_l, std_u = float(init_noise_std[LOWER_BODY]), float(init_noise_std[UPPER_BODY])
        else:
            std_l = std_u = float(init_noise_std)
        # Separate parameters per group (HERO has one std per actor) so that the
        # optimizer param groups and the per-group KL/LR schedule stay disjoint.
        self.std_lower = nn.Parameter(std_l * torch.ones(n_lower))
        self.std_upper = nn.Parameter(std_u * torch.ones(n_upper))

        if (cfg_l.min_noise_std, cfg_l.min_mean_noise_std) != (cfg_u.min_noise_std, cfg_u.min_mean_noise_std):
            raise ValueError("min_noise_std / min_mean_noise_std must match between actor_lower and actor_upper")
        self.min_noise_std = cfg_l.min_noise_std
        self.min_mean_noise_std = cfg_l.min_mean_noise_std
        self.std_clamp_max: float | None = None if std_clamp_max is None else float(std_clamp_max)
        self.std_clamp_max_lower: float | None = None if std_clamp_max_lower is None else float(std_clamp_max_lower)
        lo = self.min_noise_std or 0.0
        for knob, value in (("std_clamp_max", self.std_clamp_max), ("std_clamp_max_lower", self.std_clamp_max_lower)):
            if value is None:
                continue
            if not (0.0 < value <= 10.0):
                raise ValueError(f"{knob}={value} must lie in (0, 10]")
            if value <= lo:
                raise ValueError(f"{knob}={value} must exceed the lower clamp min_noise_std={lo}")
        if self.std_clamp_max is not None or self.std_clamp_max_lower is not None:
            # the initial std is projected too, so that the first distribution is inside the band (per head)
            with torch.no_grad():
                for p, bound in ((self.std_lower, self.std_clamp_by_group[LOWER_BODY]), (self.std_upper, self.std_clamp_by_group[UPPER_BODY])):
                    if bound is not None and bool((p > bound).any()):
                        p.clamp_(max=bound)
        self.distribution: Normal | None = None
        Normal.set_default_validate_args(False)

    # ------------------------------------------------------------------ params
    @property
    def std(self) -> torch.Tensor:
        """Full ``[num_actions]`` std vector (lower then upper), read by ``PPO._post_epoch_logging``."""
        return torch.cat([self.std_lower, self.std_upper], dim=0)

    @property
    def std_by_group(self) -> dict[str, torch.Tensor]:
        return {LOWER_BODY: self.std_lower, UPPER_BODY: self.std_upper}

    @property
    def std_clamp_by_group(self) -> dict[str, float | None]:
        """Effective upper std bound per head: the lower body takes ``std_clamp_max_lower`` when set, else ``std_clamp_max``; the
        upper body always ``std_clamp_max`` (``None`` = unbounded)."""
        lower = self.std_clamp_max if self.std_clamp_max_lower is None else self.std_clamp_max_lower
        return {LOWER_BODY: lower, UPPER_BODY: self.std_clamp_max}

    @property
    def per_head_clamp(self) -> bool:
        """``True`` when the two heads carry different upper bounds (``std_clamp_max_lower`` set and != ``std_clamp_max``)."""
        return self.std_clamp_max_lower is not None and self.std_clamp_max_lower != self.std_clamp_max

    def group_module(self, group: str) -> BaseModule:
        return {LOWER_BODY: self.actor_lower, UPPER_BODY: self.actor_upper}[group]

    def group_parameters(self, group: str) -> Iterator[nn.Parameter]:
        """Parameters belonging to one group (its MLP + its std) - disjoint across groups."""
        yield from self.group_module(group).parameters()
        yield {LOWER_BODY: self.std_lower, UPPER_BODY: self.std_upper}[group]

    # ----------------------------------------------------------------- forward
    def forward(self, actor_obs: torch.Tensor) -> torch.Tensor:
        """Deterministic action mean ``[N, num_actions] = cat([lower(obs), upper(obs)])``."""
        return torch.cat([self.actor_lower(actor_obs), self.actor_upper(actor_obs)], dim=-1)

    def _current_std(self) -> torch.Tensor:
        """Effective std: stock lower handling (``min_noise_std`` / ``min_mean_noise_std``) then the optional upper clamp."""
        std = self.std
        if self.min_noise_std:
            std = torch.clamp(std, min=self.min_noise_std)
        elif self.min_mean_noise_std:
            current_mean = std.mean()
            if current_mean < self.min_mean_noise_std:
                std = std * (self.min_mean_noise_std / (current_mean + 1e-6))
        if self.per_head_clamp:
            bounds = self.std_clamp_by_group
            parts = []
            for g in GROUP_KEYS:
                part = std[self.group_slices[g]]
                parts.append(part if bounds[g] is None else torch.clamp(part, max=bounds[g]))
            return torch.cat(parts, dim=0)
        if self.std_clamp_max is not None:
            std = torch.clamp(std, max=self.std_clamp_max)
        return std

    def project_std_(self) -> bool:


        if self.std_clamp_max is None and self.std_clamp_max_lower is None:
            return False
        bounds = self.std_clamp_by_group
        moved = False
        with torch.no_grad():
            for p, bound in ((self.std_lower, bounds[LOWER_BODY]), (self.std_upper, bounds[UPPER_BODY])):
                if bound is not None and bool((p > bound).any()):
                    p.clamp_(max=bound)
                    moved = True
        return moved

    def noise_std_stats(self) -> dict[str, float]:
        """Effective (clamped) std statistics for ``Policy/*`` logging: overall and per-group means; with an upper clamp
        also the bound and the number of joints per group whose RAW std sits at (or above) it."""
        eff = self._current_std().detach()
        out = {"mean_noise_std_clamped": float(eff.mean().item())}
        for g, sl in self.group_slices.items():
            out[f"mean_noise_std_{g}_clamped"] = float(eff[sl].mean().item())
        bounds = self.std_clamp_by_group
        if self.std_clamp_max is not None:
            out["std_clamp_max"] = float(self.std_clamp_max)
        if self.per_head_clamp:
            # Log each head's bound alongside the shared bound when it is set.
            for g in GROUP_KEYS:
                if bounds[g] is not None:
                    out[f"std_clamp_max_{g}"] = float(bounds[g])
        if self.std_clamp_max is not None or self.std_clamp_max_lower is not None:
            raw = self.std_by_group
            for g in GROUP_KEYS:
                if bounds[g] is not None:
                    out[f"noise_std_at_upper_bound_{g}"] = float((raw[g].detach() >= bounds[g]).sum().item())
        return out

    def update_distribution(self, actor_obs: torch.Tensor) -> None:
        mean = self.forward(actor_obs)
        self.project_std_()  # never build the distribution from a raw std outside the band (dead gradient)
        self.distribution = Normal(mean, mean * 0.0 + self._current_std())

    def act(self, policy_state_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        self.update_distribution(policy_state_dict["actor_obs"])
        assert self.distribution is not None
        return self.distribution.sample()

    def act_inference(self, policy_state_dict: Mapping[str, torch.Tensor]) -> torch.Tensor:
        return self.forward(policy_state_dict["actor_obs"])

    def reset(self, dones: torch.Tensor | None = None) -> None:  # noqa: ARG002 - interface parity
        pass

    # ------------------------------------------------------------ distribution
    def _dist(self) -> Normal:
        if self.distribution is None:
            raise RuntimeError("call act()/update_distribution() before querying the distribution")
        return self.distribution

    @property
    def action_mean(self) -> torch.Tensor:
        return self._dist().mean

    @property
    def action_std(self) -> torch.Tensor:
        return self._dist().stddev

    @property
    def entropy(self) -> torch.Tensor:
        """Entropy summed over ALL action dims ``[N]``."""
        return self._dist().entropy().sum(dim=-1)

    @property
    def entropy_by_group(self) -> dict[str, torch.Tensor]:
        """Entropy summed over each group's action slice ``{group: [N]}``."""
        ent = self._dist().entropy()
        return {g: ent[..., sl].sum(dim=-1) for g, sl in self.group_slices.items()}

    def get_actions_log_prob(self, actions: torch.Tensor) -> torch.Tensor:
        """Joint log-prob over ALL action dims ``[N]`` (= sum of the per-group log-probs)."""
        return self._dist().log_prob(actions).sum(dim=-1)

    def get_actions_log_prob_by_group(self, actions: torch.Tensor) -> dict[str, torch.Tensor]:
        """Per-group log-prob over the group's own action slice ``{group: [N]}`` (HERO: each actor's log-prob)."""
        logp = self._dist().log_prob(actions)
        return {g: logp[..., sl].sum(dim=-1) for g, sl in self.group_slices.items()}

    # ------------------------------------------------------- HERO checkpoints
    def group_state_dict(self, group: str) -> dict[str, torch.Tensor]:
        """State dict of one actor in HERO ``PPOActor`` key format (``actor_module.module.*``, ``std``)."""
        module = self.group_module(group)
        out = {f"actor_module.{k}": v for k, v in module.state_dict().items()}
        out["std"] = {LOWER_BODY: self.std_lower, UPPER_BODY: self.std_upper}[group].detach().clone()
        return out

    def load_group_state_dict(self, group: str, state_dict: Mapping[str, torch.Tensor], strict: bool = True) -> None:
        """Load one actor from a HERO-format state dict (inverse of :meth:`group_state_dict`)."""
        module_sd = {k[len("actor_module.") :]: v for k, v in state_dict.items() if k.startswith("actor_module.")}
        self.group_module(group).load_state_dict(module_sd, strict=strict)
        if "std" in state_dict:
            target = {LOWER_BODY: self.std_lower, UPPER_BODY: self.std_upper}[group]
            std = torch.as_tensor(state_dict["std"], dtype=target.dtype, device=target.device).reshape(-1)
            if std.numel() != target.numel():
                raise ValueError(f"{group}: std has {std.numel()} entries, expected {target.numel()}")
            with torch.no_grad():
                target.copy_(std)
        elif strict:
            raise KeyError(f"{group}: state dict has no 'std'")

    def load_hero_actor_state_dicts(self, actor_state_dicts: Mapping[str, Mapping[str, torch.Tensor]]) -> None:
        """Load ``{"lower_body": sd, "upper_body": sd}`` as saved by HERO ``PPOMultiActorCritic.save``."""
        for g in GROUP_KEYS:
            self.load_group_state_dict(g, actor_state_dicts[g])


class PPODualCritic(nn.Module):
    """Two independent ``PPOCritic`` heads over the same ``critic_obs`` (one per reward group)."""

    def __init__(
        self,
        obs_dim_dict: Mapping[str, Any],
        critic_lower_cfg: ModuleConfig,
        critic_upper_cfg: ModuleConfig,
        history_length: Mapping[str, int] | None = None,
    ):
        super().__init__()
        if list(critic_lower_cfg.input_dim) != list(critic_upper_cfg.input_dim):
            raise ValueError(
                "both critics must read the same observation groups: "
                f"{list(critic_lower_cfg.input_dim)} != {list(critic_upper_cfg.input_dim)}"
            )
        self.input_keys: list[str] = list(critic_lower_cfg.input_dim)
        history_length = dict(history_length) if history_length is not None else {}
        self.critic_lower = PPOCritic(obs_dim_dict, critic_lower_cfg, history_length)
        self.critic_upper = PPOCritic(obs_dim_dict, critic_upper_cfg, history_length)

    def group_module(self, group: str) -> PPOCritic:
        return {LOWER_BODY: self.critic_lower, UPPER_BODY: self.critic_upper}[group]

    def group_parameters(self, group: str) -> Iterator[nn.Parameter]:
        yield from self.group_module(group).parameters()

    def evaluate_group(self, policy_state_dict: Mapping[str, torch.Tensor], group: str) -> torch.Tensor:
        """Value estimate ``[N, 1]`` of one reward group."""
        return self.group_module(group).evaluate(policy_state_dict)

    def evaluate(self, policy_state_dict: Mapping[str, torch.Tensor]) -> dict[str, torch.Tensor]:
        """Value estimates ``{group: [N, 1]}`` of both reward groups."""
        return {g: self.evaluate_group(policy_state_dict, g) for g in GROUP_KEYS}

    def reset(self, dones: torch.Tensor | None = None) -> None:
        self.critic_lower.reset(dones)
        self.critic_upper.reset(dones)


__all__ = [
    "GROUP_KEYS",
    "GROUP_SUFFIX",
    "LOWER_BODY",
    "PPODualActor",
    "PPODualCritic",
    "UPPER_BODY",
]
