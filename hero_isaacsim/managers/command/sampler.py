"""Source-aware motion sampling with explicit per-source weights."""

from __future__ import annotations

from typing import Any, Mapping, Sequence

import numpy as np
import torch
from loguru import logger

from holosoma.managers.command.terms.wbt import AdaptiveTimestepsSampler


def build_clip_prior(
    clip_source_tag_id: torch.Tensor,
    source_tags: Sequence[str],
    source_weights: Mapping[str, float] | None,
    device: str | torch.device,
    *,
    unlisted_source_weight: float = 0.0,
) -> torch.Tensor:
    """Per-clip sampling prior from per-source episode shares.

    ``source_weights`` are *episode* shares per ``source_tag`` , not clip counts:
    every clip of source ``s`` receives ``w_s / n_s`` so the source marginal equals ``w_s``
    irrespective of how many clips it contributes.  An empty mapping yields the uniform
    (clip-count proportional) prior.  Sources present in the corpus but absent from the
    mapping receive ``unlisted_source_weight`` (default 0 -> never sampled) with a loud
    warning; listed sources absent from the corpus are dropped (mass renormalised)."""


    clip_source_tag_id = torch.as_tensor(clip_source_tag_id, dtype=torch.long).detach().cpu()
    num_clips = int(clip_source_tag_id.numel())
    if num_clips == 0:
        raise ValueError("cannot build a clip prior for an empty corpus")
    if not source_weights:
        return torch.full((num_clips,), 1.0 / num_clips, dtype=torch.float32, device=device)

    counts = torch.bincount(clip_source_tag_id, minlength=len(source_tags)).to(torch.float32)
    per_source = torch.zeros(len(source_tags), dtype=torch.float32)
    unlisted: dict[str, int] = {}
    for i, tag in enumerate(source_tags):
        if counts[i] <= 0:
            continue
        if tag in source_weights:
            w = float(source_weights[tag])
            if not np.isfinite(w) or w < 0.0:
                raise ValueError(f"source_weights[{tag!r}]={w!r} must be finite and >= 0")
            per_source[i] = w
        else:
            per_source[i] = float(unlisted_source_weight)
            unlisted[tag] = int(counts[i].item())
    missing_in_corpus = [tag for tag in source_weights if tag not in set(source_tags)]
    if missing_in_corpus:
        logger.warning(f"build_clip_prior: source_weights list sources absent from the corpus: {missing_in_corpus}")
    if unlisted:
        logger.warning(
            f"build_clip_prior: sources in the corpus without a source_weights entry get weight "
            f"{unlisted_source_weight:g}: {unlisted} (tag -> clips)"
        )
    total = float(per_source.sum().item())
    if total <= 0.0:
        raise ValueError(
            f"clip prior is all zero: source_weights={dict(source_weights)} vs corpus sources={list(source_tags)}"
        )
    per_source = per_source / total
    per_clip_share = torch.where(counts > 0, per_source / counts.clamp_min(1.0), torch.zeros_like(per_source))
    prior = per_clip_share[clip_source_tag_id]
    prior = prior / prior.sum()
    return prior.to(device=device, dtype=torch.float32)


class HeroAdaptiveTimestepsSampler(AdaptiveTimestepsSampler):
    """``AdaptiveTimestepsSampler`` with a persistent per-clip source prior."""

    def __init__(
        self,
        motion_time_step_total: int,
        device: str,
        env_fps: int,
        *,
        clip_prior: torch.Tensor | None = None,
        source_tag_ids: torch.Tensor | None = None,
        source_tags: Sequence[str] | None = None,
        num_clips: int = 1,
        clip_cap_relative: float | None = None,
        **kwargs: Any,
    ):
        # The parent collapses ``num_clips`` to 1 in shared-row modes; keep the true count.
        self.num_motions = max(int(num_clips), 1)
        if clip_cap_relative is not None:
            clip_cap_relative = float(clip_cap_relative)
            if not clip_cap_relative >= 1.0:
                raise ValueError(f"clip_cap_relative must be >= 1 (1 == exactly the prior), got {clip_cap_relative!r}")
        #: Per-clip bound on the failure-weighted draw probability, as a multiple of the clip's prior share
        #: (``None`` = unbounded). Unlike a flat absolute cap it never starves small sources or zeroes exclusions.
        self.clip_cap_relative = clip_cap_relative
        super().__init__(motion_time_step_total, device, env_fps, num_clips=num_clips, **kwargs)
        if clip_prior is None:
            clip_prior = torch.full((self.num_motions,), 1.0 / self.num_motions, dtype=torch.float32)
        clip_prior = torch.as_tensor(clip_prior, dtype=torch.float32).reshape(-1)
        if clip_prior.numel() != self.num_motions:
            raise ValueError(f"clip_prior has {clip_prior.numel()} entries, expected num_clips={self.num_motions}")
        if not bool(torch.isfinite(clip_prior).all()) or bool((clip_prior < 0).any()) or float(clip_prior.sum()) <= 0:
            raise ValueError("clip_prior must be finite, non-negative and have positive mass")
        self.clip_prior = (clip_prior / clip_prior.sum()).to(self.device)
        self.source_tags = list(source_tags) if source_tags is not None else []
        if source_tag_ids is not None:
            ids = torch.as_tensor(source_tag_ids, dtype=torch.long).reshape(-1)
            if ids.numel() != self.num_motions:
                raise ValueError(f"source_tag_ids has {ids.numel()} entries, expected {self.num_motions}")
            self.source_tag_ids: torch.Tensor | None = ids.to(self.device)
            if not self.source_tags:
                self.source_tags = [str(i) for i in range(int(ids.max().item()) + 1)]
        else:
            self.source_tag_ids = None
        #: Transient per-clip eligibility mask (shape-aware draws); set only for the duration of a masked draw.
        self._active_clip_mask: torch.Tensor | None = None

    # ------------------------------------------------------------------ prior application
    @property
    def _prior_applies_to_table(self) -> bool:
        return bool(self.per_clip and self.num_clips > 1)

    @staticmethod
    def _project_with_vector_cap(prob: torch.Tensor, cap: torch.Tensor) -> torch.Tensor:
        """Water-filling projection of a categorical ``prob`` onto ``{q: 0 <= q_i <= cap_i, sum(q) = 1}``.

        The solution keeps the proportions of the uncapped entries: ``q_i = min(cap_i, s * prob_i)`` with the
        scale ``s`` found on the support sorted by ``prob_i / cap_i``. Entries with ``cap_i == 0`` get exactly 0.
        Feasibility requires ``sum(cap) >= 1`` (callers guarantee it: the caps are a multiple >= 1 of a prior)."""
        tiny = torch.finfo(prob.dtype).tiny
        support = cap > 0
        prob = torch.where(support, prob, torch.zeros_like(prob))
        total = prob.sum()
        if total <= 0:
            raise ValueError("clip cap projection: no probability mass on the capped support")
        prob = prob / total
        if bool((prob <= cap + 1.0e-12).all()):
            return prob
        if float(cap.sum()) < 1.0 - 1.0e-6:
            raise ValueError(f"clip cap projection infeasible: caps sum to {float(cap.sum()):.6f} < 1")
        ratio = torch.where(support, prob / cap.clamp_min(tiny), torch.zeros_like(prob))
        order = torch.argsort(ratio, descending=True)
        prob_s, cap_s = prob[order], cap[order]
        capped_mass = torch.cat([torch.zeros(1, dtype=prob.dtype, device=prob.device), torch.cumsum(cap_s, dim=0)[:-1]])
        tail_mass = torch.flip(torch.cumsum(torch.flip(prob_s, dims=(0,)), dim=0), dims=(0,))
        remaining = 1.0 - capped_mass
        scale = remaining / tail_mass.clamp_min(tiny)
        # k capped entries are consistent when the first free entry (index k) fits under its cap.
        valid = (remaining > 0.0) & (scale * prob_s <= cap_s + 1.0e-7)
        k = torch.argmax(valid.to(dtype=torch.int64))
        projected = torch.minimum(cap, scale[k] * prob)
        return projected / projected.sum()

    def _shape_clip_marginal(self, probabilities: torch.Tensor) -> torch.Tensor:
        """Failure-weighted clip marginal x source prior (x eligibility mask), bounded per clip relative to the prior.

        Order matters: the parent first shapes the prior-free failure/uniform table (temperature, optional absolute
        cap), THEN the prior multiplies the clip marginal, so the documented ``source_weights`` are reproduced exactly
        when failures are uniform and a zero-weight (or masked) clip gets exactly zero. ``clip_cap_relative`` finally
        bounds every clip at that multiple of its prior share, which stops one clip from absorbing the sampler
        without starving sources that have few clips."""
        if not self._prior_applies_to_table:
            return super()._shape_clip_marginal(probabilities)
        shaped = super()._shape_clip_marginal(probabilities)
        table = shaped.view(self.num_clips, self.num_bins)
        tiny = torch.finfo(table.dtype).tiny
        marginal = table.sum(dim=1)
        weight = self.clip_prior
        mask = self._active_clip_mask
        if mask is not None:
            weight = weight * mask.to(device=weight.device, dtype=weight.dtype)
        weighted = marginal * weight
        total = weighted.sum()
        if total <= 0:
            if mask is not None:
                raise ValueError("masked clip draw: no eligible clip carries probability mass (clip_mask all False or all masked clips have zero prior)")
            raise ValueError("clip prior leaves no clip with probability mass")
        clip_prob = weighted / total
        if self.clip_cap_relative is not None:
            share = weight / weight.sum()
            clip_prob = self._project_with_vector_cap(clip_prob, share * float(self.clip_cap_relative))
        uniform_phase = torch.full_like(table, 1.0 / float(self.num_bins))
        conditional = torch.where(marginal[:, None] > 0.0, table / marginal.clamp_min(tiny)[:, None], uniform_phase)
        return (conditional * clip_prob[:, None]).reshape(-1)

    # ------------------------------------------------------------------ masked draws (shape-aware sampling)
    def _check_clip_mask(self, clip_mask: torch.Tensor | None) -> torch.Tensor | None:
        if clip_mask is None:
            return None
        mask = torch.as_tensor(clip_mask, dtype=torch.bool).reshape(-1)
        if mask.numel() != self.num_motions:
            raise ValueError(f"clip_mask has {mask.numel()} entries, expected num_clips={self.num_motions}")
        if not bool(mask.any()):
            raise ValueError("clip_mask selects no clip")
        return mask.to(self.device)

    def masked_sampling_probabilities(self, clip_mask: torch.Tensor | None) -> torch.Tensor:
        """The flat (clip, bin) table the masked draw samples from (``sampling_probabilities`` with the mask applied inside
        :meth:`_shape_clip_marginal`); ``clip_mask=None`` -> the unmasked table."""
        mask = self._check_clip_mask(clip_mask)
        if mask is None:
            return self.sampling_probabilities
        self._active_clip_mask = mask
        try:
            return self.sampling_probabilities
        finally:
            self._active_clip_mask = None

    def sample_clip_phase(self, num_samples: int, clip_mask: torch.Tensor | None = None) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint (clip, phase) draw; with ``clip_mask`` Bool[num_clips] only the eligible clips can be drawn.
        ``clip_mask=None`` is the stock draw (byte-identical to the parent)."""
        if clip_mask is None:
            return super().sample_clip_phase(num_samples)
        assert self.per_clip, "sample_clip_phase requires per_clip mode"
        probs = self.masked_sampling_probabilities(clip_mask)
        flat = torch.multinomial(probs, num_samples, replacement=True)
        clip_ids = flat // self.num_bins
        bins = flat % self.num_bins
        phase = (bins.float() + torch.rand(num_samples, device=self.device)) / self.num_bins
        return clip_ids, phase

    def sample_clip_ids(self, num_samples: int, clip_mask: torch.Tensor | None = None) -> torch.Tensor:
        """Prior-weighted clip draw for the non-per-clip branch (replaces ``torch.randint``); ``clip_mask`` restricts the
        support (the prior is renormalised over the eligible clips)."""
        mask = self._check_clip_mask(clip_mask)
        if mask is None:
            return torch.multinomial(self.clip_prior, num_samples, replacement=True)
        weights = self.clip_prior * mask.to(dtype=self.clip_prior.dtype)
        if float(weights.sum()) <= 0.0:
            raise ValueError("masked clip draw: every eligible clip has zero prior mass")
        return torch.multinomial(weights, num_samples, replacement=True)

    def sample_clip_phase_grouped(self, group_ids: torch.Tensor, masks: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        """Joint (clip, phase) draw for ``n`` samples whose eligibility depends on their group: ``group_ids`` Long[n] indexes
        the rows of ``masks`` Bool[K, num_clips]; group ``k`` is drawn from the table masked with ``masks[k]`` (one masked
        draw per group present).  Returns ``(clip_ids Long[n], phase Float[n])`` aligned with ``group_ids``."""
        group_ids = torch.as_tensor(group_ids, dtype=torch.long).reshape(-1).to(self.device)
        masks = torch.as_tensor(masks, dtype=torch.bool)
        if masks.ndim != 2 or masks.shape[1] != self.num_motions:
            raise ValueError(f"masks must be [K, num_clips={self.num_motions}], got {tuple(masks.shape)}")
        n = group_ids.numel()
        clip_ids = torch.zeros(n, dtype=torch.long, device=self.device)
        phase = torch.zeros(n, dtype=torch.float32, device=self.device)
        if n == 0:
            return clip_ids, phase
        if int(group_ids.min()) < 0 or int(group_ids.max()) >= masks.shape[0]:
            raise ValueError(f"group_ids out of range [0, {masks.shape[0]}): {group_ids.min().item()}..{group_ids.max().item()}")
        for k in torch.unique(group_ids).tolist():
            sel = (group_ids == k).nonzero(as_tuple=False).flatten()
            ids_k, phase_k = self.sample_clip_phase(int(sel.numel()), clip_mask=masks[k])
            clip_ids[sel] = ids_k
            phase[sel] = phase_k
        return clip_ids, phase

    def masked_clip_marginal(self, clip_mask: torch.Tensor | None) -> torch.Tensor:
        """Clip marginal of the masked table (per-clip mode) or the masked, renormalised prior (``num_clips``,)."""
        if self._prior_applies_to_table:
            return self.masked_sampling_probabilities(clip_mask).view(self.num_clips, self.num_bins).sum(dim=1)
        mask = self._check_clip_mask(clip_mask)
        if mask is None:
            return self.clip_prior
        w = self.clip_prior * mask.to(dtype=self.clip_prior.dtype)
        return w / w.sum()

    def clip_marginal(self) -> torch.Tensor:
        """Effective clip marginal actually used for training draws (num_clips,)."""
        if self._prior_applies_to_table:
            return self.sampling_probabilities.view(self.num_clips, self.num_bins).sum(dim=1)
        return self.clip_prior

    # ------------------------------------------------------------------ logging helpers
    def _fractions_by_source(self, clip_mass: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.source_tag_ids is None:
            return {}
        frac = torch.zeros(len(self.source_tags), dtype=clip_mass.dtype, device=clip_mass.device)
        frac.index_add_(0, self.source_tag_ids.to(clip_mass.device), clip_mass)
        return {f"motion/source_frac_{tag}": frac[i] for i, tag in enumerate(self.source_tags)}

    def expected_source_fractions(self) -> dict[str, torch.Tensor]:
        """Expected episode share per source under the current sampling distribution."""
        return self._fractions_by_source(self.clip_marginal())

    def source_fractions_from_ids(self, motion_ids: torch.Tensor) -> dict[str, torch.Tensor]:
        """Empirical share per source among the clips currently assigned to envs."""
        motion_ids = torch.as_tensor(motion_ids, dtype=torch.long, device=self.device).reshape(-1)
        if motion_ids.numel() == 0 or self.source_tag_ids is None:
            return {}
        counts = torch.bincount(motion_ids, minlength=self.num_motions).to(torch.float32)
        return self._fractions_by_source(counts / counts.sum())

    def prior_source_fractions(self) -> dict[str, float]:
        """Static per-source mass of the prior itself (what the mix converges to with uniform failures)."""
        return {k: float(v.item()) for k, v in self._fractions_by_source(self.clip_prior).items()}

    # ------------------------------------------------------------------ checkpointing
    def state_dict(self) -> dict[str, Any]:
        state = super().state_dict()
        state["hero_clip_prior"] = self.clip_prior.detach().cpu().clone()
        state["hero_num_motions"] = int(self.num_motions)
        state["hero_source_tags"] = list(self.source_tags)
        return state

    def load_state_dict(self, state: Mapping[str, Any]) -> None:
        super().load_state_dict(state)
        if "hero_clip_prior" not in state:
            raise KeyError("adaptive sampler checkpoint is missing 'hero_clip_prior'")
        prior = torch.as_tensor(state["hero_clip_prior"], dtype=torch.float32, device=self.device).reshape(-1)
        if prior.numel() != self.clip_prior.numel():
            raise ValueError(
                f"checkpoint clip_prior has {prior.numel()} clips, live registry has {self.clip_prior.numel()}"
            )
        if not torch.allclose(prior, self.clip_prior, rtol=0.0, atol=1.0e-6):
            max_diff = float((prior - self.clip_prior).abs().max().item())
            raise ValueError(
                "checkpoint clip_prior differs from the live source_weights / corpus "
                f"(max abs diff {max_diff:.3e}); resuming would silently change the training mix"
            )
        ckpt_tags = list(state.get("hero_source_tags", []))
        if ckpt_tags and self.source_tags and ckpt_tags != self.source_tags:
            raise ValueError(f"checkpoint source tags {ckpt_tags} != live {self.source_tags}")
