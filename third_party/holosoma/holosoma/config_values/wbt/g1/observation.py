"""Whole Body Tracking observation presets for the G1 robot."""

from holosoma.config_types.observation import ObservationManagerCfg, ObsGroupCfg, ObsTermCfg

actor_obs_shared = ObsGroupCfg(
    concatenate=True,
    enable_noise=True,
    history_length=1,
    terms={
        "motion_command": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:motion_command",
            scale=1.0,
            noise=0.0,
        ),
        "motion_ref_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b",
            scale=1.0,
            noise=0.05,
        ),
        "base_ang_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:base_ang_vel",
            scale=1.0,
            noise=0.2,
        ),
        "dof_pos": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:dof_pos",
            scale=1.0,
            noise=0.01,
        ),
        "dof_vel": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:dof_vel",
            scale=1.0,
            noise=0.5,
        ),
        "actions": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:actions",
            scale=1.0,
            noise=0.0,
        ),
    },
)

critic_obs_shared_terms = {
    "motion_command": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_command",
        scale=1.0,
        noise=0.0,
    ),
    "motion_ref_pos_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_pos_b",
        scale=1.0,
        noise=0.25,
    ),
    "motion_ref_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_ori_b",
        scale=1.0,
        noise=0.05,
    ),
    "robot_body_pos_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:robot_body_pos_b",
        scale=1.0,
        noise=0.0,
    ),
    "robot_body_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:robot_body_ori_b",
        scale=1.0,
        noise=0.0,
    ),
    "base_lin_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:base_lin_vel",
        scale=1.0,
        noise=0.0,
    ),
    "base_ang_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:base_ang_vel",
        scale=1.0,
        noise=0.2,
    ),
    "dof_pos": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_pos",
        scale=1.0,
        noise=0.01,
    ),
    "dof_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:dof_vel",
        scale=1.0,
        noise=0.5,
    ),
    "actions": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:actions",
        scale=1.0,
        noise=0.0,
    ),
}

critic_obs_w_object_terms = critic_obs_shared_terms.copy()
critic_obs_w_object_terms.update(
    {
        "obj_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_pos_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_ori_b",
            scale=1.0,
            noise=0.0,
        ),
        "obj_lin_vel_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:obj_lin_vel_b",
            scale=1.0,
            noise=0.0,
        ),
    }
)

g1_29dof_wbt_observation = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_shared,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_shared_terms,
        ),
    },
)

g1_29dof_wbt_observation_w_object = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_shared,
        "critic_obs": ObsGroupCfg(
            concatenate=True,
            enable_noise=False,
            history_length=1,
            terms=critic_obs_w_object_terms,
        ),
    },
)

# ---------------------------------------------------------------------------
# Object-aware ACTOR observation variants (the policy itself sees object/EE info,
# not just the critic). Baseline actor_obs is motion-only (154-d).
# ---------------------------------------------------------------------------

# Variant A: object 6DoF pose in robot frame -> actor (obj_pos_b 3 + obj_ori_b 6 = +9 dims)
_obj_pose_actor_terms = dict(actor_obs_shared.terms)
_obj_pose_actor_terms.update(
    {
        "obj_pos_b": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:obj_pos_b", scale=1.0, noise=0.0),
        "obj_ori_b": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:obj_ori_b", scale=1.0, noise=0.0),
    }
)
actor_obs_obj_pose = ObsGroupCfg(
    concatenate=True, enable_noise=True, history_length=1, terms=_obj_pose_actor_terms
)

# Variant B: HERO-style EE 6DoF residual -> actor (2 EE x (pos 3 + ori 6) = +18 dims)
_ee_residual_actor_terms = dict(actor_obs_shared.terms)
_ee_residual_actor_terms.update(
    {
        "ee_residual_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:ee_residual_pos_b", scale=1.0, noise=0.0
        ),
        "ee_residual_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:ee_residual_ori_b", scale=1.0, noise=0.0
        ),
    }
)
actor_obs_ee_residual = ObsGroupCfg(
    concatenate=True, enable_noise=True, history_length=1, terms=_ee_residual_actor_terms
)

# object-pose actor obs (critic keeps full object terms)
g1_29dof_wbt_observation_obj_pose_actor = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_obj_pose,
        "critic_obs": ObsGroupCfg(
            concatenate=True, enable_noise=False, history_length=1, terms=critic_obs_w_object_terms
        ),
    },
)

# EE-residual actor obs (critic keeps full object terms)
g1_29dof_wbt_observation_ee_residual_actor = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_ee_residual,
        "critic_obs": ObsGroupCfg(
            concatenate=True, enable_noise=False, history_length=1, terms=critic_obs_w_object_terms
        ),
    },
)

# Variant A+B: object 6DoF pose AND HERO-style EE 6DoF residual -> actor
# (motion 154 + obj_pos_b 3 + obj_ori_b 6 + ee_residual_pos_b 6 + ee_residual_ori_b 12 = 181 dims)
_obj_pose_ee_residual_actor_terms = dict(actor_obs_shared.terms)
_obj_pose_ee_residual_actor_terms.update(
    {
        "obj_pos_b": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:obj_pos_b", scale=1.0, noise=0.0),
        "obj_ori_b": ObsTermCfg(func="holosoma.managers.observation.terms.wbt:obj_ori_b", scale=1.0, noise=0.0),
        "ee_residual_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:ee_residual_pos_b", scale=1.0, noise=0.0
        ),
        "ee_residual_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:ee_residual_ori_b", scale=1.0, noise=0.0
        ),
    }
)
actor_obs_obj_pose_ee_residual = ObsGroupCfg(
    concatenate=True, enable_noise=True, history_length=1, terms=_obj_pose_ee_residual_actor_terms
)

# combined obj-pose + EE-residual actor obs (critic keeps full object terms)
g1_29dof_wbt_observation_obj_pose_ee_residual_actor = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_obj_pose_ee_residual,
        "critic_obs": ObsGroupCfg(
            concatenate=True, enable_noise=False, history_length=1, terms=critic_obs_w_object_terms
        ),
    },
)

# ---------------------------------------------------------------------------
# VR-3-point head_residual variants (head/torso + 2 hands, base-frame 6D residual).
# Like a VR tracker giving the policy its head + both hands relative to the reference.
# All keep critic at full w_object privilege (best critic setting).
# ---------------------------------------------------------------------------
_HEAD_RESIDUAL_TERMS = {
    "head_residual_pos_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:head_residual_pos_b", scale=1.0, noise=0.0
    ),
    "head_residual_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:head_residual_ori_b", scale=1.0, noise=0.0
    ),
}
# blind + head_residual (VR-3-point only, no object pose)
_head_residual_actor_terms = dict(actor_obs_shared.terms)
_head_residual_actor_terms.update(_HEAD_RESIDUAL_TERMS)
actor_obs_head_residual = ObsGroupCfg(
    concatenate=True, enable_noise=True, history_length=1, terms=_head_residual_actor_terms
)
g1_29dof_wbt_observation_head_residual_actor = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_head_residual,
        "critic_obs": ObsGroupCfg(
            concatenate=True, enable_noise=False, history_length=1, terms=critic_obs_w_object_terms
        ),
    },
)
# obj_pose + EE-residual + head_residual (full VR-3-point on top of object-aware)
_obj_pose_ee_head_actor_terms = dict(_obj_pose_ee_residual_actor_terms)
_obj_pose_ee_head_actor_terms.update(_HEAD_RESIDUAL_TERMS)
actor_obs_obj_pose_ee_head = ObsGroupCfg(
    concatenate=True, enable_noise=True, history_length=1, terms=_obj_pose_ee_head_actor_terms
)
g1_29dof_wbt_observation_obj_pose_ee_head_actor = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_obj_pose_ee_head,
        "critic_obs": ObsGroupCfg(
            concatenate=True, enable_noise=False, history_length=1, terms=critic_obs_w_object_terms
        ),
    },
)

_ee_only_actor_terms = dict(actor_obs_shared.terms)
_ee_only_actor_terms.update(
    {
        "ee_residual_pos_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:ee_residual_pos_b", scale=1.0, noise=0.0
        ),
        "ee_residual_ori_b": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:ee_residual_ori_b", scale=1.0, noise=0.0
        ),
    }
)
actor_obs_ee_only = ObsGroupCfg(
    concatenate=True, enable_noise=True, history_length=1, terms=_ee_only_actor_terms
)
g1_29dof_wbt_observation_ee_only_actor = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_ee_only,
        "critic_obs": ObsGroupCfg(
            concatenate=True, enable_noise=False, history_length=1, terms=critic_obs_w_object_terms
        ),
    },
)

# Replace the leading motion command with current and future reference frames.


def _make_obj_pose_ee_future_actor_obs(num_future: int) -> ObsGroupCfg:
    """obj_pose + EE-residual actor obs with the motion_command term replaced by
    a future-stacked motion_command_future(num_future) term (kept leading)."""
    terms = dict(_obj_pose_ee_residual_actor_terms)
    # replace the single-frame reference with the future-stacked reference, in place
    # (dict preserves insertion order, so 'motion_command' stays leading).
    terms["motion_command"] = ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_command_future",
        params={"num_future": num_future},
        scale=1.0,
        noise=0.0,
    )
    return ObsGroupCfg(concatenate=True, enable_noise=True, history_length=1, terms=terms)


def _make_obj_pose_ee_future_observation(num_future: int) -> ObservationManagerCfg:
    return ObservationManagerCfg(
        groups={
            "actor_obs": _make_obj_pose_ee_future_actor_obs(num_future),
            "critic_obs": ObsGroupCfg(
                concatenate=True, enable_noise=False, history_length=1, terms=critic_obs_w_object_terms
            ),
        },
    )


g1_29dof_wbt_observation_obj_pose_ee_future1 = _make_obj_pose_ee_future_observation(1)
g1_29dof_wbt_observation_obj_pose_ee_future4 = _make_obj_pose_ee_future_observation(4)
g1_29dof_wbt_observation_obj_pose_ee_future8 = _make_obj_pose_ee_future_observation(8)
g1_29dof_wbt_observation_obj_pose_ee_future16 = _make_obj_pose_ee_future_observation(16)

# Stack the 181-dimensional actor observation over history_length frames.
# History starts zero-padded and flattens to [N, 181 * history_length].


def _make_obj_pose_ee_history_observation(history_length: int) -> ObservationManagerCfg:
    """obj_pose+EE actor obs frame-stacked over `history_length` past frames (group-level)."""
    actor = ObsGroupCfg(
        concatenate=True, enable_noise=True, history_length=history_length,
        terms=_obj_pose_ee_residual_actor_terms,
    )
    return ObservationManagerCfg(
        groups={
            "actor_obs": actor,
            "critic_obs": ObsGroupCfg(
                concatenate=True, enable_noise=False, history_length=1, terms=critic_obs_w_object_terms
            ),
        },
    )


g1_29dof_wbt_observation_obj_pose_ee_hist1 = _make_obj_pose_ee_history_observation(1)
g1_29dof_wbt_observation_obj_pose_ee_hist4 = _make_obj_pose_ee_history_observation(4)
g1_29dof_wbt_observation_obj_pose_ee_hist8 = _make_obj_pose_ee_history_observation(8)
g1_29dof_wbt_observation_obj_pose_ee_hist16 = _make_obj_pose_ee_history_observation(16)


def _make_blind_history_observation(history_length: int) -> ObservationManagerCfg:
    """Frame-stack the motion-only actor over history_length past frames.

    Actor dimension is 154 * history_length; the critic retains object observations."""
    actor = ObsGroupCfg(
        concatenate=True, enable_noise=True, history_length=history_length,
        terms=actor_obs_shared.terms,
    )
    return ObservationManagerCfg(
        groups={
            "actor_obs": actor,
            "critic_obs": ObsGroupCfg(
                concatenate=True, enable_noise=False, history_length=1, terms=critic_obs_w_object_terms
            ),
        },
    )


g1_29dof_wbt_observation_blind_hist4 = _make_blind_history_observation(4)

# Add base-frame projected gravity to expose IMU-observable roll and pitch.


_PROJGRAV_TERM = {
    "projected_gravity": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:projected_gravity", scale=1.0, noise=0.05,
    ),
}

# blind actor + projected_gravity (154 + 3 = 157-d); critic keeps full object terms
_projgrav_actor_terms = dict(actor_obs_shared.terms)
_projgrav_actor_terms.update(_PROJGRAV_TERM)
actor_obs_projgrav = ObsGroupCfg(
    concatenate=True, enable_noise=True, history_length=1, terms=_projgrav_actor_terms
)
g1_29dof_wbt_observation_projgrav = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_projgrav,
        "critic_obs": ObsGroupCfg(
            concatenate=True, enable_noise=False, history_length=1, terms=critic_obs_w_object_terms
        ),
    },
)

# obj_pose+EE actor + projected_gravity (181 + 3 = 184-d); critic keeps full object terms
_obj_pose_ee_projgrav_actor_terms = dict(_obj_pose_ee_residual_actor_terms)
_obj_pose_ee_projgrav_actor_terms.update(_PROJGRAV_TERM)
actor_obs_obj_pose_ee_projgrav = ObsGroupCfg(
    concatenate=True, enable_noise=True, history_length=1, terms=_obj_pose_ee_projgrav_actor_terms
)
g1_29dof_wbt_observation_obj_pose_ee_projgrav = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_obj_pose_ee_projgrav,
        "critic_obs": ObsGroupCfg(
            concatenate=True, enable_noise=False, history_length=1, terms=critic_obs_w_object_terms
        ),
    },
)

# Add motion-reference position, body pose, and base linear velocity to the actor.
# The critic retains object observations.


_PRIV_PROPRIO_TERMS = {
    "motion_ref_pos_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:motion_ref_pos_b", scale=1.0, noise=0.25,
    ),
    "robot_body_pos_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:robot_body_pos_b", scale=1.0, noise=0.0,
    ),
    "robot_body_ori_b": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:robot_body_ori_b", scale=1.0, noise=0.0,
    ),
    "base_lin_vel": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:base_lin_vel", scale=1.0, noise=0.0,
    ),
}
_priv_proprio_actor_terms = dict(actor_obs_shared.terms)
_priv_proprio_actor_terms.update(_PRIV_PROPRIO_TERMS)
actor_obs_priv_proprio = ObsGroupCfg(
    concatenate=True, enable_noise=True, history_length=1, terms=_priv_proprio_actor_terms
)
g1_29dof_wbt_observation_priv_proprio = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_priv_proprio,
        "critic_obs": ObsGroupCfg(
            concatenate=True, enable_noise=False, history_length=1, terms=critic_obs_w_object_terms
        ),
    },
)

# Combine object pose and end-effector residuals with privileged proprioception.


_obj_pose_ee_priv_proprio_actor_terms = dict(_obj_pose_ee_residual_actor_terms)
_obj_pose_ee_priv_proprio_actor_terms.update(_PRIV_PROPRIO_TERMS)
actor_obs_obj_pose_ee_priv_proprio = ObsGroupCfg(
    concatenate=True, enable_noise=True, history_length=1, terms=_obj_pose_ee_priv_proprio_actor_terms
)
g1_29dof_wbt_observation_obj_pose_ee_priv_proprio = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_obj_pose_ee_priv_proprio,
        "critic_obs": ObsGroupCfg(
            concatenate=True, enable_noise=False, history_length=1, terms=critic_obs_w_object_terms
        ),
    },
)

# Add normalized object mass to the actor as a privileged simulation observation.


_OBJ_PHYSICS_TERM = {
    "obj_physics_priv": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:obj_physics_priv", scale=1.0, noise=0.0,
        params={"use_com": False, "mass_center": 3.0, "mass_scale": 3.0},
    ),
}
_obj_pose_ee_priv_proprio_physics_actor_terms = dict(_obj_pose_ee_priv_proprio_actor_terms)
_obj_pose_ee_priv_proprio_physics_actor_terms.update(_OBJ_PHYSICS_TERM)
actor_obs_obj_pose_ee_priv_proprio_physics = ObsGroupCfg(
    concatenate=True, enable_noise=True, history_length=1,
    terms=_obj_pose_ee_priv_proprio_physics_actor_terms,
)
g1_29dof_wbt_observation_obj_pose_ee_priv_proprio_physics = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_obj_pose_ee_priv_proprio_physics,
        "critic_obs": ObsGroupCfg(
            concatenate=True, enable_noise=False, history_length=1, terms=critic_obs_w_object_terms
        ),
    },
)

# Add scaled hand contact forces in the base frame (+6 dimensions),
# following the force-observation approach in GRAIL (arXiv:2606.05160).


_CONTACT_FORCE_TERM = {
    "hand_contact_force": ObsTermCfg(
        func="holosoma.managers.observation.terms.wbt:hand_contact_force",
        scale=1.0,
        noise=0.0,
        params={"force_scale": 50.0, "clamp": 3.0},
    ),
}
_obj_pose_ee_contactforce_actor_terms = dict(_obj_pose_ee_residual_actor_terms)
_obj_pose_ee_contactforce_actor_terms.update(_CONTACT_FORCE_TERM)
actor_obs_obj_pose_ee_contactforce = ObsGroupCfg(
    concatenate=True, enable_noise=True, history_length=1,
    terms=_obj_pose_ee_contactforce_actor_terms,
)
g1_29dof_wbt_observation_obj_pose_ee_contactforce = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_obj_pose_ee_contactforce,
        "critic_obs": ObsGroupCfg(
            concatenate=True, enable_noise=False, history_length=1, terms=critic_obs_w_object_terms
        ),
    },
)

# Combine privileged proprioception with projected gravity, with or without object observations.


_priv_proprio_projgrav_actor_terms = dict(actor_obs_shared.terms)
_priv_proprio_projgrav_actor_terms.update(_PRIV_PROPRIO_TERMS)
_priv_proprio_projgrav_actor_terms.update(_PROJGRAV_TERM)
actor_obs_priv_proprio_projgrav = ObsGroupCfg(
    concatenate=True, enable_noise=True, history_length=1, terms=_priv_proprio_projgrav_actor_terms
)
g1_29dof_wbt_observation_priv_proprio_projgrav = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_priv_proprio_projgrav,
        "critic_obs": ObsGroupCfg(
            concatenate=True, enable_noise=False, history_length=1, terms=critic_obs_w_object_terms
        ),
    },
)

_obj_pose_ee_priv_proprio_projgrav_actor_terms = dict(_obj_pose_ee_residual_actor_terms)
_obj_pose_ee_priv_proprio_projgrav_actor_terms.update(_PRIV_PROPRIO_TERMS)
_obj_pose_ee_priv_proprio_projgrav_actor_terms.update(_PROJGRAV_TERM)
actor_obs_obj_pose_ee_priv_proprio_projgrav = ObsGroupCfg(
    concatenate=True, enable_noise=True, history_length=1,
    terms=_obj_pose_ee_priv_proprio_projgrav_actor_terms,
)
g1_29dof_wbt_observation_obj_pose_ee_priv_proprio_projgrav = ObservationManagerCfg(
    groups={
        "actor_obs": actor_obs_obj_pose_ee_priv_proprio_projgrav,
        "critic_obs": ObsGroupCfg(
            concatenate=True, enable_noise=False, history_length=1, terms=critic_obs_w_object_terms
        ),
    },
)


# Capture a reference chunk every K steps, with timing, amplitude, and Gaussian perturbations.
# The actor executes the held chunk while the critic uses the live reference.


def _make_chunk_s0_observation(chunk_k: int, with_object_ee: bool) -> ObservationManagerCfg:
    terms = {
        "chunk_command": ObsTermCfg(
            func="holosoma.managers.observation.terms.wbt:chunk_command",
            params={"chunk_k": chunk_k},
            scale=1.0,
            noise=0.0,
        ),
    }
    base = _obj_pose_ee_residual_actor_terms if with_object_ee else actor_obs_shared.terms
    for k, v in base.items():
        if k == "motion_command":
            continue  # replaced by the frozen chunk
        terms[k] = v
    return ObservationManagerCfg(
        groups={
            "actor_obs": ObsGroupCfg(concatenate=True, enable_noise=True, history_length=1, terms=terms),
            "critic_obs": ObsGroupCfg(
                concatenate=True, enable_noise=False, history_length=1, terms=critic_obs_w_object_terms
            ),
        },
    )


g1_29dof_wbt_observation_chunk8_s0 = _make_chunk_s0_observation(8, False)
g1_29dof_wbt_observation_obj_pose_ee_chunk8_s0 = _make_chunk_s0_observation(8, True)

__all__ = [
    "g1_29dof_wbt_observation",
    "g1_29dof_wbt_observation_w_object",
    "g1_29dof_wbt_observation_obj_pose_actor",
    "g1_29dof_wbt_observation_ee_residual_actor",
    "g1_29dof_wbt_observation_obj_pose_ee_residual_actor",
    "g1_29dof_wbt_observation_obj_pose_ee_future1",
    "g1_29dof_wbt_observation_obj_pose_ee_future4",
    "g1_29dof_wbt_observation_obj_pose_ee_future8",
    "g1_29dof_wbt_observation_obj_pose_ee_future16",
    "g1_29dof_wbt_observation_obj_pose_ee_hist1",
    "g1_29dof_wbt_observation_obj_pose_ee_hist4",
    "g1_29dof_wbt_observation_obj_pose_ee_hist8",
    "g1_29dof_wbt_observation_obj_pose_ee_hist16",
    "g1_29dof_wbt_observation_blind_hist4",
    "g1_29dof_wbt_observation_projgrav",
    "g1_29dof_wbt_observation_obj_pose_ee_projgrav",
    "g1_29dof_wbt_observation_priv_proprio",
    "g1_29dof_wbt_observation_obj_pose_ee_priv_proprio",
    "g1_29dof_wbt_observation_priv_proprio_projgrav",
    "g1_29dof_wbt_observation_obj_pose_ee_priv_proprio_projgrav",
    "g1_29dof_wbt_observation_chunk8_s0",
    "g1_29dof_wbt_observation_obj_pose_ee_chunk8_s0",
    "g1_29dof_wbt_observation_head_residual_actor",
    "g1_29dof_wbt_observation_obj_pose_ee_head_actor",
    "g1_29dof_wbt_observation_ee_only_actor",
    "g1_29dof_wbt_observation_obj_pose_ee_contactforce",
]
