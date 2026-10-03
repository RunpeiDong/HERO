"""Adaptive motion sampling settings for HERO."""
from __future__ import annotations

ADAPTIVE_UNIFORM_RATIO = 0.3
"""Share of the sampling mass that stays uniform over every (clip, phase) cell regardless of failure counts."""
ADAPTIVE_CLIP_MAX_RELATIVE = 10.0
"""Per-clip bound on the failure-weighted draw probability as a multiple of the clip's prior share (source weight /
clips of that source). Bounded relative to the prior, the sampler cannot collapse onto one clip, yet a source with few
clips keeps its documented share and a zero-weight source stays at zero -- a flat absolute cap does neither.

The backend's absolute per-clip cap (``adaptive_sampler_clip_max_probability``) is refused below 1 for the per-clip HERO
sampler at configuration time (``HeroMotionConfig.__post_init__``), so there is no per-corpus "feasible cap" to lift."""
