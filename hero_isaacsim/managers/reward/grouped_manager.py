"""Reward manager with per-tag reward groups (HERO ``lower_body`` / ``upper_body``).

HERO keeps one reward buffer and trains one critic per reward group.  holosoma's ``RewardManager.compute`` returns a
single summed reward.  ``GroupedRewardManager`` keeps the exact same API (so
``PenaltyCurriculum`` and friends keep working through ``set_term_cfg``) and, in
addition, accumulates one buffer per group tag found in ``RewardTermCfg.tags``.

The env is expected to publish the groups as
``extras["rewards_by_group"] = {"lower_body": [N], "upper_body": [N]}``
(``PPODual`` reads exactly that key).

Importable without Isaac Sim."""

from __future__ import annotations

from collections.abc import Sequence
from typing import Any

import torch
from loguru import logger

from holosoma.config_types.reward import RewardManagerCfg, RewardTermCfg
from holosoma.managers.reward.manager import RewardManager

DEFAULT_GROUP_TAGS: tuple[str, str] = ("lower_body", "upper_body")
"""HERO reward groups (order == ``hero_isaacsim.agents.ppo_dual.modules.GROUP_KEYS``)."""

REWARDS_BY_GROUP_KEY = "rewards_by_group"
"""``extras`` key under which the env publishes ``GroupedRewardManager.group_rewards``."""


class GroupedRewardManager(RewardManager):
    """``RewardManager`` that additionally sums terms per group tag.

    Parameters
    ----------
    cfg, env, device
        As for :class:`holosoma.managers.reward.manager.RewardManager`.
    group_tags
        Tags that define reward groups (default ``("lower_body", "upper_body")``).  Every
        active term should carry exactly one of them; a term with none contributes to the
        total only (warned once, or raised when ``strict``), a term with several is
        added to each of them (warned).
    strict
        Raise instead of warn on ungrouped terms."""

    def __init__(
        self,
        cfg: RewardManagerCfg,
        env: Any,
        device: str,
        group_tags: Sequence[str] = DEFAULT_GROUP_TAGS,
        strict: bool = False,
    ):
        self.group_tags: tuple[str, ...] = tuple(group_tags)
        if len(set(self.group_tags)) != len(self.group_tags) or not self.group_tags:
            raise ValueError(f"group_tags must be a non-empty set of unique tags, got {group_tags}")
        self._strict = strict
        self._term_groups: dict[str, tuple[str, ...]] = {}
        super().__init__(cfg, env, device)
        self._group_reward_bufs: dict[str, torch.Tensor] = {
            tag: torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device) for tag in self.group_tags
        }
        self._episode_sums_group: dict[str, torch.Tensor] = {
            tag: torch.zeros(self.env.num_envs, dtype=torch.float, device=self.device) for tag in self.group_tags
        }
        self._refresh_term_groups(warn=True)

    # ------------------------------------------------------------------ groups
    def _refresh_term_groups(self, warn: bool = False) -> None:
        """Recompute term -> group tags from the current term configs."""
        groups: dict[str, tuple[str, ...]] = {}
        ungrouped: list[str] = []
        multi: list[str] = []
        for name, cfg in zip(self._term_names, self._term_cfgs):
            tags = tuple(t for t in self.group_tags if t in (cfg.tags or []))
            groups[name] = tags
            if not tags:
                ungrouped.append(name)
            elif len(tags) > 1:
                multi.append(name)
        self._term_groups = groups
        if ungrouped:
            msg = (
                f"GroupedRewardManager: {len(ungrouped)} active reward term(s) carry none of the group tags "
                f"{self.group_tags} and only contribute to the total reward: {ungrouped}"
            )
            if self._strict:
                raise ValueError(msg)
            if warn:
                logger.warning(msg)
        if multi and warn:
            logger.warning(
                f"GroupedRewardManager: term(s) {multi} carry several group tags and are added to each group"
            )

    @property
    def term_groups(self) -> dict[str, tuple[str, ...]]:
        """Active term name -> group tags it contributes to."""
        return dict(self._term_groups)

    @property
    def group_rewards(self) -> dict[str, torch.Tensor]:
        """Per-group reward of the last :meth:`compute` call ``{tag: [num_envs]}`` (weight x dt scaled)."""
        return self._group_reward_bufs

    @property
    def episode_sums_group(self) -> dict[str, torch.Tensor]:
        return self._episode_sums_group

    def group_terms(self, tag: str) -> list[str]:
        return [name for name, tags in self._term_groups.items() if tag in tags]

    # ----------------------------------------------------------------- compute
    def compute(self, dt: float) -> torch.Tensor:
        """Total reward ``[num_envs]``; also fills :attr:`group_rewards`.

        Same arithmetic as ``RewardManager.compute`` (``raw * weight * dt`` per term,
        episode sums, optional non-negative clipping - HERO clips per group, so the
        clip is applied to every group buffer as well as to the total)."""
        self._reward_buf[:] = 0.0
        for buf in self._group_reward_bufs.values():
            buf[:] = 0.0

        for term_name, term_cfg in zip(self._term_names, self._term_cfgs):
            if term_name in self._term_instances:
                rew_raw = self._term_instances[term_name](self.env, **term_cfg.params)
            else:
                rew_raw = self._term_funcs[term_name](self.env, **term_cfg.params)

            if rew_raw.shape[0] != self.env.num_envs:
                raise ValueError(
                    f"Reward term '{term_name}' returned wrong shape. "
                    f"Expected [{self.env.num_envs}], got {rew_raw.shape}"
                )

            rew_scaled = rew_raw * term_cfg.weight * dt
            self._reward_buf += rew_scaled
            for tag in self._term_groups.get(term_name, ()):
                self._group_reward_bufs[tag] += rew_scaled

            self._episode_sums[term_name] += rew_scaled
            self._episode_sums_raw[term_name] += rew_raw

        if self.cfg.only_positive_rewards:
            self._reward_buf[:] = torch.clip(self._reward_buf, min=0.0)
            for tag, buf in self._group_reward_bufs.items():
                buf[:] = torch.clip(buf, min=0.0)

        for tag, buf in self._group_reward_bufs.items():
            self._episode_sums_group[tag] += buf

        return self._reward_buf

    # ------------------------------------------------------------------- reset
    def reset(self, env_ids: torch.Tensor | None = None) -> dict[str, dict[str, torch.Tensor]]:
        """As ``RewardManager.reset`` plus ``rew_group_<tag>`` entries in ``episode`` / ``episode_all``."""
        extras = super().reset(env_ids)
        if env_ids is None:
            env_ids_slice: slice | torch.Tensor = slice(None)
        elif isinstance(env_ids, torch.Tensor):
            env_ids_slice = env_ids.to(device=self.device, dtype=torch.long)
        else:
            env_ids_slice = torch.as_tensor(env_ids, device=self.device, dtype=torch.long)
        for tag in self.group_tags:
            rew_all = self._episode_sums_group[tag] / self.env.max_episode_length_s
            extras["episode_all"][f"rew_group_{tag}"] = rew_all.detach().clone()
            extras["episode"][f"rew_group_{tag}"] = rew_all[env_ids_slice].detach().clone()
            self._episode_sums_group[tag][env_ids_slice] = 0.0
        return extras

    # ------------------------------------------------------------- term config
    def set_term_cfg(self, name: str, cfg: RewardTermCfg) -> None:
        """Preserved API (used by ``PenaltyCurriculum``); refreshes the group mapping if tags changed."""
        super().set_term_cfg(name, cfg)
        self._refresh_term_groups(warn=False)

    def __str__(self) -> str:
        msg = f"<GroupedRewardManager> contains {len(self._term_names)} active terms in groups {self.group_tags}.\n"
        for tag in self.group_tags:
            msg += f"Group {tag}:\n"
            for name in self.group_terms(tag):
                msg += f"  - {name}: weight={self.get_term_cfg(name).weight}\n"
        ungrouped = [n for n, t in self._term_groups.items() if not t]
        if ungrouped:
            msg += f"Ungrouped (total only): {ungrouped}\n"
        return msg


__all__ = ["DEFAULT_GROUP_TAGS", "REWARDS_BY_GROUP_KEY", "GroupedRewardManager"]
