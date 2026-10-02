"""HERO episode termination configuration."""
from __future__ import annotations

from holosoma.config_types.termination import TerminationManagerCfg, TerminationTermCfg

HERO_TERMINATION_MODULE = "hero_isaacsim.managers.termination.hero"

TIMEOUT_FUNC = "holosoma.managers.termination.terms.common:timeout_exceeded"

GRAVITY_TILT_THRESHOLD = 0.8

BASE_HEIGHT_MIN = 0.25

_hero_core_terms = {
    "timeout": TerminationTermCfg(func=TIMEOUT_FUNC, is_timeout=True),
    "clip_ends": TerminationTermCfg(func=f"{HERO_TERMINATION_MODULE}:clip_ends", is_timeout=True),
    "gravity_tilt": TerminationTermCfg(
        func=f"{HERO_TERMINATION_MODULE}:gravity_tilt", params={"threshold_x": GRAVITY_TILT_THRESHOLD, "threshold_y": GRAVITY_TILT_THRESHOLD}
    ),
    "base_height_below": TerminationTermCfg(
        func=f"{HERO_TERMINATION_MODULE}:base_height_below", params={"min_height": BASE_HEIGHT_MIN}
    ),
}

hero_h1_termination = TerminationManagerCfg(terms=dict(_hero_core_terms))
DEFAULTS = {"hero": hero_h1_termination}
