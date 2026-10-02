"""HERO penalty curriculum configuration."""
from __future__ import annotations

from holosoma.config_types.curriculum import CurriculumManagerCfg, CurriculumTermCfg

CURRICULUM_MODULE = "holosoma.managers.curriculum.terms.locomotion"

PENALTY_CURRICULUM_PARAMS = {
    "enabled": True,
    "tag": "penalty_curriculum",
    "initial_scale": 0.1,
    "min_scale": 0.0,
    "max_scale": 1.0,
    "level_down_threshold": 40.0,
    "level_up_threshold": 210.0,
    "degree": 0.00001,
}

hero_curriculum = CurriculumManagerCfg(
    params={"num_compute_average_epl": 10000},
    setup_terms={
        "average_episode_tracker": CurriculumTermCfg(
            func=f"{CURRICULUM_MODULE}:AverageEpisodeLengthTracker",
            params={},
        ),
        "penalty_curriculum": CurriculumTermCfg(
            func=f"{CURRICULUM_MODULE}:PenaltyCurriculum",
            params=dict(PENALTY_CURRICULUM_PARAMS),
        ),
    },
    reset_terms={},
    step_terms={},
)
DEFAULTS = {"hero": hero_curriculum}
