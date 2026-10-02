import tyro
from typing_extensions import Annotated

from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_values.loco.g1.experiment import g1_29dof, g1_29dof_fast_sac
from holosoma.config_values.loco.t1.experiment import t1_29dof, t1_29dof_fast_sac
from holosoma.config_values.wbt.g1.experiment import (
    g1_29dof_wbt,
    g1_29dof_wbt_fast_sac,
    g1_29dof_wbt_fast_sac_contact,
    g1_29dof_wbt_fast_sac_ee_residual,
    g1_29dof_wbt_fast_sac_obj_pose,
    g1_29dof_wbt_fast_sac_obj_pose_ee,
    g1_29dof_wbt_fast_sac_obj_pose_ee_contactforce,
    g1_29dof_wbt_flash_sac_obj_pose_ee,
    g1_29dof_wbt_fast_sac_obj_pose_ee_future1,
    g1_29dof_wbt_fast_sac_obj_pose_ee_future4,
    g1_29dof_wbt_fast_sac_obj_pose_ee_future8,
    g1_29dof_wbt_fast_sac_obj_pose_ee_future16,
    g1_29dof_wbt_fast_sac_obj_pose_ee_hist1,
    g1_29dof_wbt_fast_sac_obj_pose_ee_hist4,
    g1_29dof_wbt_fast_sac_obj_pose_ee_hist8,
    g1_29dof_wbt_fast_sac_obj_pose_ee_hist16,
    g1_29dof_wbt_fast_sac_omni,
    g1_29dof_wbt_fast_sac_omni_ee_residual,
    g1_29dof_wbt_fast_sac_omni_obj_pose,
    g1_29dof_wbt_fast_sac_omni_obj_pose_ee,
    g1_29dof_wbt_fast_sac_omni_obj_pose_ee_omnifix,
    g1_29dof_wbt_fast_sac_omni_obj_pose_ee_relaxterm,
    g1_29dof_wbt_fast_sac_obj_pose_ee_rubber,
    g1_29dof_wbt_fast_sac_obj_pose_ee_rubbercol,
    g1_29dof_wbt_fast_sac_obj_pose_ee_16convex,
    g1_29dof_wbt_fast_sac_obj_pose_ee_paddle3box,
    g1_29dof_wbt_fast_sac_obj_pose_ee_boxcol,
    g1_29dof_wbt_fast_sac_w_object_boxcol,
    g1_29dof_wbt_fast_sac_w_object_boxcol_projgrav,
    g1_29dof_wbt_fast_sac_obj_pose_ee_boxcol_projgrav,
    g1_29dof_wbt_fast_sac_w_object_boxcol_hist4,
    g1_29dof_wbt_fast_sac_obj_pose_ee_boxcol_hist4,
    g1_29dof_wbt_fast_sac_priv_proprio_boxcol,
    g1_29dof_wbt_fast_sac_chunk8_s0_boxcol,
    g1_29dof_wbt_fast_sac_obj_pose_ee_chunk8_s0_boxcol,
    g1_29dof_wbt_fast_sac_obj_pose_ee_priv_proprio_boxcol,
    g1_29dof_wbt_fast_sac_obj_pose_ee_priv_proprio_boxcol_mass212,
    g1_29dof_wbt_fast_sac_boxcol_mass_curriculum,
    g1_29dof_wbt_ppo_boxcol_mass_curriculum,
    g1_29dof_wbt_fast_sac_omni_obj_pose_ee_rubber,
    g1_29dof_wbt_fast_sac_omni_obj_pose_ee_rubber_omnifix,
    g1_29dof_wbt_fast_sac_omni_obj_pose_ee_rubbercol,
    g1_29dof_wbt_fast_sac_omni_obj_pose_ee_rubbercol_omnifix,
    g1_29dof_wbt_fast_sac_omni_priv_proprio_projgrav_rubbercol,
    g1_29dof_wbt_fast_sac_omni_obj_pose_ee_priv_proprio_projgrav_rubbercol,
    g1_29dof_wbt_fast_sac_w_object_rubbercol,
    g1_29dof_wbt_fast_sac_obj_pose_rubbercol,
    g1_29dof_wbt_fast_sac_ee_residual_rubbercol,
    g1_29dof_wbt_fast_sac_contact_rubbercol,
    g1_29dof_wbt_fast_sac_w_object_rubbercol_retgt,
    g1_29dof_wbt_fast_sac_obj_pose_ee_rubbercol_retgt,
    g1_29dof_wbt_fast_sac_w_object_halfsphere_retgt,
    g1_29dof_wbt_fast_sac_obj_pose_ee_halfsphere_retgt,
    g1_29dof_wbt_fast_sac_obj_pose_ee_handbody,
    g1_29dof_wbt_fast_sac_omni_obj_pose_ee_handbody,
    g1_29dof_wbt_fast_sac_omni_obj_pose_ee_handbody_omnifix,
    g1_29dof_wbt_fast_sac_w_object,
    g1_29dof_wbt_fast_sac_w_object_rubber,
    g1_29dof_wbt_ppo_blind,
    g1_29dof_wbt_ppo_obj_pose_ee,
    g1_29dof_wbt_ppo_obj_pose_ee_boxcol,
    g1_29dof_wbt_ppo_w_object_boxcol,
    g1_29dof_wbt_ppo_head_residual_boxcol,
    g1_29dof_wbt_ppo_ee_only_boxcol,
    g1_29dof_wbt_ppo_obj_pose_ee_head_boxcol,
    g1_29dof_wbt_fast_sac_head_residual_boxcol,
    g1_29dof_wbt_fast_sac_ee_only_boxcol,
    g1_29dof_wbt_fast_sac_obj_pose_ee_head_boxcol,
    g1_29dof_wbt_ppo_obj_pose_ee_boxcol_projgrav,
    g1_29dof_wbt_ppo_priv_proprio_boxcol,
    g1_29dof_wbt_ppo_obj_pose_ee_priv_proprio_boxcol,
    g1_29dof_wbt_ppo_obj_pose_ee_handbody,
    g1_29dof_wbt_ppo_omni,
    g1_29dof_wbt_ppo_omni_obj_pose_ee,
    g1_29dof_wbt_ppo_omni_rubbercol,
    g1_29dof_wbt_ppo_omni_obj_pose_ee_rubbercol,
    g1_29dof_wbt_ppo_omni_priv_proprio_rubbercol,
    g1_29dof_wbt_ppo_omni_obj_pose_ee_priv_proprio_rubbercol,
    g1_29dof_wbt_w_object,
)

DEFAULTS = {
    "g1_29dof": g1_29dof,
    "g1_29dof_fast_sac": g1_29dof_fast_sac,
    "t1_29dof": t1_29dof,
    "t1_29dof_fast_sac": t1_29dof_fast_sac,
    "g1_29dof_wbt": g1_29dof_wbt,
    "g1_29dof_wbt_w_object": g1_29dof_wbt_w_object,
    "g1_29dof_wbt_fast_sac": g1_29dof_wbt_fast_sac,
    "g1_29dof_wbt_fast_sac_w_object": g1_29dof_wbt_fast_sac_w_object,
    "g1_29dof_wbt_fast_sac_obj_pose": g1_29dof_wbt_fast_sac_obj_pose,
    "g1_29dof_wbt_fast_sac_ee_residual": g1_29dof_wbt_fast_sac_ee_residual,
    "g1_29dof_wbt_fast_sac_contact": g1_29dof_wbt_fast_sac_contact,
    "g1_29dof_wbt_fast_sac_obj_pose_ee": g1_29dof_wbt_fast_sac_obj_pose_ee,
    # FlashSAC with object-pose and end-effector-residual observations.
    "g1_29dof_wbt_flash_sac_obj_pose_ee": g1_29dof_wbt_flash_sac_obj_pose_ee,
    # Append hand contact-force observations (+6 dimensions).
    "g1_29dof_wbt_fast_sac_obj_pose_ee_contactforce": g1_29dof_wbt_fast_sac_obj_pose_ee_contactforce,
    # Add future reference frames to the actor.
    "g1_29dof_wbt_fast_sac_obj_pose_ee_future1": g1_29dof_wbt_fast_sac_obj_pose_ee_future1,
    "g1_29dof_wbt_fast_sac_obj_pose_ee_future4": g1_29dof_wbt_fast_sac_obj_pose_ee_future4,
    "g1_29dof_wbt_fast_sac_obj_pose_ee_future8": g1_29dof_wbt_fast_sac_obj_pose_ee_future8,
    "g1_29dof_wbt_fast_sac_obj_pose_ee_future16": g1_29dof_wbt_fast_sac_obj_pose_ee_future16,
    # Stack past actor-observation frames.
    "g1_29dof_wbt_fast_sac_obj_pose_ee_hist1": g1_29dof_wbt_fast_sac_obj_pose_ee_hist1,
    "g1_29dof_wbt_fast_sac_obj_pose_ee_hist4": g1_29dof_wbt_fast_sac_obj_pose_ee_hist4,
    "g1_29dof_wbt_fast_sac_obj_pose_ee_hist8": g1_29dof_wbt_fast_sac_obj_pose_ee_hist8,
    "g1_29dof_wbt_fast_sac_obj_pose_ee_hist16": g1_29dof_wbt_fast_sac_obj_pose_ee_hist16,
    # Multi-clip object tracking; set motion_dir through the CLI.
    "g1_29dof_wbt_fast_sac_omni": g1_29dof_wbt_fast_sac_omni,
    "g1_29dof_wbt_fast_sac_omni_obj_pose": g1_29dof_wbt_fast_sac_omni_obj_pose,
    "g1_29dof_wbt_fast_sac_omni_ee_residual": g1_29dof_wbt_fast_sac_omni_ee_residual,
    "g1_29dof_wbt_fast_sac_omni_obj_pose_ee": g1_29dof_wbt_fast_sac_omni_obj_pose_ee,

    "g1_29dof_wbt_fast_sac_omni_obj_pose_ee_omnifix": g1_29dof_wbt_fast_sac_omni_obj_pose_ee_omnifix,
    # PPO multi-clip object tracking.
    "g1_29dof_wbt_ppo_omni_obj_pose_ee": g1_29dof_wbt_ppo_omni_obj_pose_ee,
    "g1_29dof_wbt_ppo_omni": g1_29dof_wbt_ppo_omni,
    # PPO with motion-only actor observations and collision-mesh hands.
    "g1_29dof_wbt_ppo_omni_rubbercol": g1_29dof_wbt_ppo_omni_rubbercol,
    "g1_29dof_wbt_ppo_omni_obj_pose_ee_rubbercol": g1_29dof_wbt_ppo_omni_obj_pose_ee_rubbercol,
    "g1_29dof_wbt_ppo_omni_priv_proprio_rubbercol": g1_29dof_wbt_ppo_omni_priv_proprio_rubbercol,
    "g1_29dof_wbt_ppo_omni_obj_pose_ee_priv_proprio_rubbercol": g1_29dof_wbt_ppo_omni_obj_pose_ee_priv_proprio_rubbercol,
    # Single-clip PPO object tracking.
    "g1_29dof_wbt_ppo_obj_pose_ee": g1_29dof_wbt_ppo_obj_pose_ee,
    "g1_29dof_wbt_ppo_blind": g1_29dof_wbt_ppo_blind,
    # PPO with convex hand collisions.
    "g1_29dof_wbt_ppo_obj_pose_ee_boxcol": g1_29dof_wbt_ppo_obj_pose_ee_boxcol,
    "g1_29dof_wbt_ppo_w_object_boxcol": g1_29dof_wbt_ppo_w_object_boxcol,
    "g1_29dof_wbt_ppo_head_residual_boxcol": g1_29dof_wbt_ppo_head_residual_boxcol,
    "g1_29dof_wbt_ppo_ee_only_boxcol": g1_29dof_wbt_ppo_ee_only_boxcol,
    "g1_29dof_wbt_ppo_obj_pose_ee_head_boxcol": g1_29dof_wbt_ppo_obj_pose_ee_head_boxcol,
    "g1_29dof_wbt_fast_sac_head_residual_boxcol": g1_29dof_wbt_fast_sac_head_residual_boxcol,
    "g1_29dof_wbt_fast_sac_ee_only_boxcol": g1_29dof_wbt_fast_sac_ee_only_boxcol,
    "g1_29dof_wbt_fast_sac_obj_pose_ee_head_boxcol": g1_29dof_wbt_fast_sac_obj_pose_ee_head_boxcol,
    "g1_29dof_wbt_ppo_obj_pose_ee_boxcol_projgrav": g1_29dof_wbt_ppo_obj_pose_ee_boxcol_projgrav,
    # PRIVILEGED-PROPRIO actor (blind + critic's non-object privileged terms) — PPO to 60k
    "g1_29dof_wbt_ppo_priv_proprio_boxcol": g1_29dof_wbt_ppo_priv_proprio_boxcol,
    "g1_29dof_wbt_ppo_obj_pose_ee_priv_proprio_boxcol": g1_29dof_wbt_ppo_obj_pose_ee_priv_proprio_boxcol,
    "g1_29dof_wbt_ppo_obj_pose_ee_handbody": g1_29dof_wbt_ppo_obj_pose_ee_handbody,
    # RUBBER-HAND single-clip (match retargeting hand geometry vs half-sphere)
    "g1_29dof_wbt_fast_sac_obj_pose_ee_rubber": g1_29dof_wbt_fast_sac_obj_pose_ee_rubber,
    "g1_29dof_wbt_fast_sac_w_object_rubber": g1_29dof_wbt_fast_sac_w_object_rubber,
    # Multi-clip tracking with rubber hands.

    "g1_29dof_wbt_fast_sac_omni_obj_pose_ee_rubber": g1_29dof_wbt_fast_sac_omni_obj_pose_ee_rubber,
    "g1_29dof_wbt_fast_sac_omni_obj_pose_ee_rubber_omnifix": g1_29dof_wbt_fast_sac_omni_obj_pose_ee_rubber_omnifix,
    # RUBBER-HAND WITH HAND COLLISION MESH (real hand-box contact; new g1_29dof_rubberhand_collision.urdf)
    "g1_29dof_wbt_fast_sac_obj_pose_ee_rubbercol": g1_29dof_wbt_fast_sac_obj_pose_ee_rubbercol,
    "g1_29dof_wbt_fast_sac_obj_pose_ee_16convex": g1_29dof_wbt_fast_sac_obj_pose_ee_16convex,
    "g1_29dof_wbt_fast_sac_obj_pose_ee_paddle3box": g1_29dof_wbt_fast_sac_obj_pose_ee_paddle3box,
    "g1_29dof_wbt_fast_sac_obj_pose_ee_boxcol": g1_29dof_wbt_fast_sac_obj_pose_ee_boxcol,
    "g1_29dof_wbt_fast_sac_w_object_boxcol": g1_29dof_wbt_fast_sac_w_object_boxcol,
    # Add projected gravity to actor observations.
    "g1_29dof_wbt_fast_sac_w_object_boxcol_projgrav": g1_29dof_wbt_fast_sac_w_object_boxcol_projgrav,
    "g1_29dof_wbt_fast_sac_obj_pose_ee_boxcol_projgrav": g1_29dof_wbt_fast_sac_obj_pose_ee_boxcol_projgrav,
    # Stack four past actor-observation frames.
    "g1_29dof_wbt_fast_sac_w_object_boxcol_hist4": g1_29dof_wbt_fast_sac_w_object_boxcol_hist4,
    "g1_29dof_wbt_fast_sac_obj_pose_ee_boxcol_hist4": g1_29dof_wbt_fast_sac_obj_pose_ee_boxcol_hist4,
    # PRIVILEGED-PROPRIO actor (blind + critic's non-object privileged terms) — FastSAC to 300k
    "g1_29dof_wbt_fast_sac_priv_proprio_boxcol": g1_29dof_wbt_fast_sac_priv_proprio_boxcol,
    "g1_29dof_wbt_fast_sac_chunk8_s0_boxcol": g1_29dof_wbt_fast_sac_chunk8_s0_boxcol,
    "g1_29dof_wbt_fast_sac_obj_pose_ee_chunk8_s0_boxcol": g1_29dof_wbt_fast_sac_obj_pose_ee_chunk8_s0_boxcol,
    # Combine object observations with privileged proprioception.
    "g1_29dof_wbt_fast_sac_obj_pose_ee_priv_proprio_boxcol": g1_29dof_wbt_fast_sac_obj_pose_ee_priv_proprio_boxcol,
    "g1_29dof_wbt_fast_sac_obj_pose_ee_priv_proprio_boxcol_mass212": g1_29dof_wbt_fast_sac_obj_pose_ee_priv_proprio_boxcol_mass212,
    "g1_29dof_wbt_fast_sac_boxcol_mass_curriculum": g1_29dof_wbt_fast_sac_boxcol_mass_curriculum,
    "g1_29dof_wbt_ppo_boxcol_mass_curriculum": g1_29dof_wbt_ppo_boxcol_mass_curriculum,
    "g1_29dof_wbt_fast_sac_omni_obj_pose_ee_rubbercol": g1_29dof_wbt_fast_sac_omni_obj_pose_ee_rubbercol,
    "g1_29dof_wbt_fast_sac_omni_obj_pose_ee_rubbercol_omnifix": g1_29dof_wbt_fast_sac_omni_obj_pose_ee_rubbercol_omnifix,

    "g1_29dof_wbt_fast_sac_omni_priv_proprio_projgrav_rubbercol": g1_29dof_wbt_fast_sac_omni_priv_proprio_projgrav_rubbercol,
    "g1_29dof_wbt_fast_sac_omni_obj_pose_ee_priv_proprio_projgrav_rubbercol": g1_29dof_wbt_fast_sac_omni_obj_pose_ee_priv_proprio_projgrav_rubbercol,
    # Actor observation variants with collision-mesh hands.
    "g1_29dof_wbt_fast_sac_w_object_rubbercol": g1_29dof_wbt_fast_sac_w_object_rubbercol,
    "g1_29dof_wbt_fast_sac_obj_pose_rubbercol": g1_29dof_wbt_fast_sac_obj_pose_rubbercol,
    "g1_29dof_wbt_fast_sac_ee_residual_rubbercol": g1_29dof_wbt_fast_sac_ee_residual_rubbercol,
    "g1_29dof_wbt_fast_sac_contact_rubbercol": g1_29dof_wbt_fast_sac_contact_rubbercol,
    # Robot-relative object rewards and termination.
    "g1_29dof_wbt_fast_sac_w_object_rubbercol_retgt": g1_29dof_wbt_fast_sac_w_object_rubbercol_retgt,
    "g1_29dof_wbt_fast_sac_obj_pose_ee_rubbercol_retgt": g1_29dof_wbt_fast_sac_obj_pose_ee_rubbercol_retgt,
    "g1_29dof_wbt_fast_sac_w_object_halfsphere_retgt": g1_29dof_wbt_fast_sac_w_object_halfsphere_retgt,
    "g1_29dof_wbt_fast_sac_obj_pose_ee_halfsphere_retgt": g1_29dof_wbt_fast_sac_obj_pose_ee_halfsphere_retgt,
    # HANDBODY: rubber_hand as tracked contact body (dont_collapse) + hand-anchored contact reward
    "g1_29dof_wbt_fast_sac_obj_pose_ee_handbody": g1_29dof_wbt_fast_sac_obj_pose_ee_handbody,
    "g1_29dof_wbt_fast_sac_omni_obj_pose_ee_handbody": g1_29dof_wbt_fast_sac_omni_obj_pose_ee_handbody,
    "g1_29dof_wbt_fast_sac_omni_obj_pose_ee_handbody_omnifix": g1_29dof_wbt_fast_sac_omni_obj_pose_ee_handbody_omnifix,
    # RELAXED-termination FastSAC multitask (isolates whether early termination causes the 6%-reach-end)
    "g1_29dof_wbt_fast_sac_omni_obj_pose_ee_relaxterm": g1_29dof_wbt_fast_sac_omni_obj_pose_ee_relaxterm,
}

# Register HERO experiments before building the tyro subcommand type.
# The backend can also run without hero_isaacsim installed.


try:
    from hero_isaacsim.register import DEFAULTS as _HERO_DEFAULTS  # noqa: E402
except ImportError:
    _HERO_DEFAULTS = {}
except Exception as _hero_exc:  # noqa: BLE001 - a broken preset must not take the stock CLI down
    from loguru import logger as _hero_logger

    _hero_logger.warning(f"hero_isaacsim presets not registered: {_hero_exc!r}")
    _HERO_DEFAULTS = {}
DEFAULTS.update({k: v for k, v in _HERO_DEFAULTS.items() if k not in DEFAULTS})


AnnotatedExperimentConfig = Annotated[
    ExperimentConfig,
    tyro.conf.arg(
        constructor=tyro.extras.subcommand_type_from_defaults(
            {f"exp:{k.replace('_', '-')}": v for k, v in DEFAULTS.items()}
        )
    ),
]
