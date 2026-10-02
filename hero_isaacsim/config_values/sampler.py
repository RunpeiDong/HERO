"""Adaptive motion sampling settings for HERO."""
from __future__ import annotations

import dataclasses

from pathlib import Path

from typing import Any, Mapping

ADAPTIVE_CLIP_MAX_PROBABILITY = 0.001
"""Per-clip cap on the adaptive sampler's draw probability (0.1 %), lifted to ``1 / num_clips`` for small corpora."""
ADAPTIVE_UNIFORM_RATIO = 0.3
"""Share of the sampling mass that stays uniform over every (clip, phase) cell regardless of failure counts."""
ADAPTIVE_CLIP_MAX_RELATIVE = 10.0
"""Per-clip bound on the failure-weighted draw probability as a multiple of the clip's prior share (source weight /
clips of that source). Bounded relative to the prior, the sampler cannot collapse onto one clip, yet a source with few
clips keeps its documented share and a zero-weight source stays at zero -- a flat absolute cap does neither."""

def count_corpus_clips(motion_dir: str | Path) -> int:
    """Number of ``*.npz`` clips the stock ``MultiMotionLoader`` will register from ``motion_dir`` (0 if unreadable)."""
    try:
        return sum(1 for p in Path(motion_dir).iterdir() if p.is_file() and p.suffix == ".npz")
    except OSError:
        return 0

def feasible_clip_max_probability(num_clips: int, cap: float = ADAPTIVE_CLIP_MAX_PROBABILITY) -> float:
    """``max(cap, 1 / num_clips)``: the per-clip cap ``AdaptiveTimestepsSampler`` accepts for ``num_clips`` clips.

    A per-clip probability cap below ``1 / num_clips`` cannot sum to 1 (the sampler raises ``ValueError`` in
    ``__init__``); for corpora with fewer than ``1 / cap`` clips the cap is lifted to ``1 / num_clips``, which
    admits only the uniform clip marginal. The HERO presets therefore leave the absolute cap at 1.0 (off) and bound
    clips relative to their prior share instead (``ADAPTIVE_CLIP_MAX_RELATIVE``). ``num_clips <= 1`` leaves ``cap``
    unchanged."""
    cap = float(cap)
    if num_clips is None or int(num_clips) <= 1:
        return cap
    return max(cap, 1.0 / float(int(num_clips)))

def with_feasible_sampler_cap(motion_config: Any, num_clips: int | None = None) -> Any:
    """``motion_config`` (a ``MotionConfig`` dataclass or its dict form) with a per-clip cap feasible for its corpus.

    Only touches configs with ``use_adaptive_timesteps_sampler`` and ``adaptive_sampler_per_clip`` set; the clip
    count defaults to ``count_corpus_clips(motion_dir)``.  Returns the input unchanged when nothing needs to move."""
    get = (lambda k, d=None: motion_config.get(k, d)) if isinstance(motion_config, dict) else (lambda k, d=None: getattr(motion_config, k, d))
    if not (get("use_adaptive_timesteps_sampler", False) and get("adaptive_sampler_per_clip", False)):
        return motion_config
    cap = float(get("adaptive_sampler_clip_max_probability", 1.0))
    if num_clips is None:
        motion_dir = get("motion_dir", "") or ""
        if not motion_dir:
            return motion_config
        num_clips = count_corpus_clips(motion_dir)
    new_cap = feasible_clip_max_probability(num_clips, cap)
    if new_cap == cap:
        return motion_config
    if isinstance(motion_config, dict):
        return {**motion_config, "adaptive_sampler_clip_max_probability": new_cap}
    return dataclasses.replace(motion_config, adaptive_sampler_clip_max_probability=new_cap)
