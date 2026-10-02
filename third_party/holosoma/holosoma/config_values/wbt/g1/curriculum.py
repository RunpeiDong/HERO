"""Whole Body Tracking curriculum presets for the G1 robot."""

from holosoma.config_types.curriculum import CurriculumManagerCfg, CurriculumTermCfg

g1_29dof_wbt_curriculum = CurriculumManagerCfg(
    params={
        "num_compute_average_epl": 1000,
    },
    setup_terms={
        "average_episode_tracker": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:AverageEpisodeLengthTracker",
            params={},
        ),
    },
    reset_terms={},
    step_terms={},
)

# Ramp object mass from 2–3 kg toward 2–8 kg as average episode length improves.
# The average_episode_tracker supplies the curriculum signal; mass is resampled on reset.


g1_29dof_wbt_curriculum_mass = CurriculumManagerCfg(
    params={
        "num_compute_average_epl": 1000,
    },
    setup_terms={
        "average_episode_tracker": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:AverageEpisodeLengthTracker",
            params={},
        ),
        "object_mass_curriculum": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:ObjectMassCurriculum",
            params={
                "mass_lo": 2.0,
                "mass_hi": 8.0,
                "mass_hi_start": 3.0,
                "level_up_threshold": 700.0,
                "level_down_threshold": 150.0,
                "step_kg": 0.5,
            },
        ),
    },
    reset_terms={
        "object_mass_curriculum": CurriculumTermCfg(
            func="holosoma.managers.curriculum.terms.locomotion:ObjectMassCurriculum",
            params={
                "mass_lo": 2.0,
                "mass_hi": 8.0,
                "mass_hi_start": 3.0,
                "level_up_threshold": 700.0,
                "level_down_threshold": 150.0,
                "step_kg": 0.5,
            },
        ),
    },
    step_terms={},
)

__all__ = ["g1_29dof_wbt_curriculum", "g1_29dof_wbt_curriculum_mass"]
