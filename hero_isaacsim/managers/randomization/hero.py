"""End-effector and link mass randomization for HERO."""
from __future__ import annotations

from typing import Any, Callable, Mapping, Sequence

import numpy as np

import torch

from loguru import logger

from holosoma.config_types.simulator import MujocoBackend

from holosoma.managers.randomization.exceptions import RandomizerNotSupportedError

from holosoma.managers.randomization.terms.locomotion import (
    _ensure_env_ids_tensor,
)

from holosoma.simulator import mujoco_required_field

from holosoma.simulator.shared.field_decorators import MUJOCO_FIELD_ATTR

DEFAULT_EE_BODY_NAMES = ("left_hand_palm_link", "right_hand_palm_link")

DEFAULT_EE_MASS_SCALE_RANGE = (0.5, 3.0)

DEFAULT_EE_ADDED_MASS_RANGE = (0.0, 1.5)

NOMINAL_MASS_ATTR = "_hero_ee_nominal_mass"  # env attr: {body_name: Tensor[num_envs]} kg (URDF nominal, CPU)

NOMINAL_INERTIA_ATTR = "_hero_ee_nominal_inertia"  # env attr: {body_name: Tensor[num_envs, 9]} (IsaacSim only)

PHYSX_CACHE_ATTR = "_hero_ee_physx_cache"  # env attr: {"masses"|"inertias"|"coms": full-size CPU tensor} (IsaacSim, per-episode path)

BODY_IDS_CACHE_ATTR = "_hero_ee_physx_body_ids"  # env attr: {tuple(body_names): Long[B]} PhysX-view body ids

def _check_range(
    name: str, rng: Sequence[float], *, allow_negative: bool, min_value: float | None = None
) -> tuple[float, float]:
    """``(lo, hi)`` of a 2-element range.  ``allow_negative=False`` demands a strictly positive lower bound (multiplicative
    scales); ``min_value`` (absolute masses in kg) replaces that rule with an explicit floor -- :data:`MIN_EE_MASS_KG`
    = 0.01 kg for the hand-configuration nominal masses (the 'none' configuration is 0.02 kg; PhysX rejects 0)."""
    if len(rng) != 2:
        raise ValueError(f"{name} must have exactly 2 elements, got {len(rng)}")
    lo, hi = float(rng[0]), float(rng[1])
    if hi < lo:
        raise ValueError(f"{name} must be (low, high) with low <= high, got {rng}")
    if min_value is not None:
        if lo < float(min_value):
            raise ValueError(f"{name} lower bound must be >= {float(min_value)} (kg floor), got {rng}")
    elif not allow_negative and lo <= 0.0:
        raise ValueError(f"{name} must be strictly positive (it is a multiplicative scale), got {rng}")
    return lo, hi

def _backend(simulator: Any) -> str:
    """``"isaacgym" | "isaacsim" | "mujoco_warp" | "unsupported"`` — same dispatch order as the startup term."""
    if hasattr(simulator, "gym"):
        return "isaacgym"
    if simulator.__class__.__name__ == "IsaacSim":
        return "isaacsim"
    if getattr(getattr(simulator, "simulator_config", None), "mujoco_backend", None) == MujocoBackend.WARP:
        return "mujoco_warp"
    return "unsupported"

def _isaacsim_robot(simulator: Any) -> Any:
    robot = getattr(simulator, "_robot", None)
    return robot if robot is not None else simulator.scene["robot"]

def _isaacsim_body_ids(robot: Any, body_names: Sequence[str]) -> torch.Tensor:
    """PhysX-view body indices of ``body_names`` in the REQUESTED order (``find_bodies`` matches full names)."""
    ids, found = robot.find_bodies(list(body_names), preserve_order=True)
    if len(ids) != len(body_names) or list(found) != list(body_names):
        raise ValueError(f"EE bodies {list(body_names)} not all found in the articulation (found {list(found)})")
    return torch.tensor(ids, dtype=torch.long, device="cpu")

def _isaacsim_body_ids_cached(env: Any, robot: Any, body_names: Sequence[str]) -> torch.Tensor:
    """:func:`_isaacsim_body_ids` memoised on the env per body-name tuple (``find_bodies`` is a regex resolution)."""
    cache = getattr(env, BODY_IDS_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(env, BODY_IDS_CACHE_ATTR, cache)
    key = tuple(body_names)
    ids = cache.get(key)
    if ids is None:
        ids = _isaacsim_body_ids(robot, body_names)
        cache[key] = ids
    return ids

def _physx_cached(env: Any, key: str, reader: Callable[[], torch.Tensor]) -> torch.Tensor:
    """Full-size CPU copy of a PhysX-view tensor (``masses [N,B_all]`` / ``inertias [N,B_all,9]`` / ``coms [N,B_all,7]``)."""
    cache = getattr(env, PHYSX_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        setattr(env, PHYSX_CACHE_ATTR, cache)
    t = cache.get(key)
    if t is None:
        t = reader()
        cache[key] = t
    return t

def _isaacgym_body_indices(simulator: Any, body_names: Sequence[str]) -> list[int]:
    missing = [n for n in body_names if n not in simulator._body_list]
    if missing:
        raise ValueError(f"EE bodies not in the IsaacGym articulation: {missing}")
    return [simulator._body_list.index(n) for n in body_names]

def _mujoco_body_ids(mj_model: Any, body_names: Sequence[str]) -> torch.Tensor:
    """Mirror of holosoma ``warp_randomization.resolve_entity_ids`` (name, then ``robot_<name>``) without importing warp."""
    import mujoco  # noqa: PLC0415 - optional backend dependency

    ids = []
    for name in body_names:
        idx = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, name)
        if idx == -1:
            idx = mujoco.mj_name2id(mj_model, mujoco.mjtObj.mjOBJ_BODY, "robot_" + name)
        if idx == -1:
            raise ValueError(f"EE body '{name}' not found in the MuJoCo model.")
        ids.append(idx)
    return torch.tensor(ids, dtype=torch.long)

def _warp_field(simulator: Any, field: str) -> Any:
    """Per-env expanded MuJoCo-warp model field (``[num_worlds, n_entities(, k)]``, torch-indexable)."""
    if simulator.num_envs > 1:
        expanded: set[str] = getattr(simulator.backend.mjw_model, "_expanded_fields", set())
        if field not in expanded:
            raise ValueError(
                f"Field '{field}' has not been expanded for per-environment randomization "
                f"(missing @mujoco_required_field('{field}')?). Expanded: {sorted(expanded) or 'none'}"
            )
    return getattr(simulator.backend.warp_model_bridge, field)

def _read_masses(env: Any, body_names: Sequence[str]) -> torch.Tensor:
    """Current simulator masses ``[num_envs, len(body_names)]`` (CPU float32) of the named bodies, all envs."""
    simulator = env.simulator
    backend = _backend(simulator)
    if backend == "isaacgym":
        bidx = _isaacgym_body_indices(simulator, body_names)
        rows = []
        for env_id in range(env.num_envs):
            props = simulator.gym.get_actor_rigid_body_properties(simulator.envs[env_id], simulator.robot_handles[env_id])
            rows.append([float(props[b].mass) for b in bidx])
        return torch.tensor(rows, dtype=torch.float32)
    if backend == "isaacsim":
        robot = _isaacsim_robot(simulator)
        ids = _isaacsim_body_ids(robot, body_names)
        # IsaacLab keeps the URDF masses in ``data.default_mass`` (what mdp.randomize_rigid_body_mass rescales from);
        # Fall back to the live PhysX view when default_mass is unavailable.
        default_mass = getattr(getattr(robot, "data", None), "default_mass", None)
        src = default_mass if default_mass is not None else robot.root_physx_view.get_masses()
        return torch.as_tensor(src)[:, ids].to(device="cpu", dtype=torch.float32).clone()
    if backend == "mujoco_warp":
        ids = _mujoco_body_ids(simulator.backend.model, body_names)
        field = _warp_field(simulator, "body_mass")
        return torch.as_tensor(field[:, ids.to(field.device)]).to(device="cpu", dtype=torch.float32).clone()
    raise RandomizerNotSupportedError(f"Unsupported simulator type '{type(simulator).__name__}' for EE mass randomization.")

def _read_inertias(env: Any, body_names: Sequence[str]) -> torch.Tensor | None:
    """Nominal inertia ``[num_envs, len(body_names), 9]`` (IsaacSim; ``data.default_inertia`` else live view) or None."""
    robot = _isaacsim_robot(env.simulator)
    ids = _isaacsim_body_ids(robot, body_names)
    default_inertia = getattr(getattr(robot, "data", None), "default_inertia", None)
    if default_inertia is None:
        get_inertias = getattr(robot.root_physx_view, "get_inertias", None)
        if not callable(get_inertias):
            return None
        default_inertia = get_inertias()
    return torch.as_tensor(default_inertia)[:, ids].to(device="cpu", dtype=torch.float32).clone()

def _cached_nominal(env: Any, attr: str, body_names: Sequence[str], reader) -> torch.Tensor | None:
    """Per-body-name cache on the env: read missing bodies once via ``reader(env, names) -> [N, B(, k)]``, stack."""
    cache = getattr(env, attr, None)
    if cache is None:
        cache = {}
        setattr(env, attr, cache)
    missing = [n for n in body_names if n not in cache]
    if missing:
        values = reader(env, missing)
        if values is None:
            return None
        for j, name in enumerate(missing):
            cache[name] = values[:, j].clone()
    return torch.stack([cache[n] for n in body_names], dim=1)

def nominal_ee_masses(env: Any, body_names: Sequence[str] = DEFAULT_EE_BODY_NAMES) -> torch.Tensor:
    """Cached nominal (URDF) masses ``[num_envs, B]`` (CPU); reads the simulator on first use per body."""
    return _cached_nominal(env, NOMINAL_MASS_ATTR, body_names, _read_masses)

def _write_masses(
    env: Any,
    env_ids: torch.Tensor,
    body_names: Sequence[str],
    new: torch.Tensor,
    nominal: torch.Tensor,
    *,
    cached: bool = True,
    inertia_ratio: torch.Tensor | None = None,
) -> None:
    """Write ABSOLUTE masses ``new [E, B]`` for ``env_ids`` (inertia follows the mass ratio where the backend allows).

    ``cached`` (IsaacSim): update the env-level full-size copies (:func:`_physx_cached`) instead of calling
    ``get_masses()/get_inertias()`` -- the per-episode path.  The startup term passes ``cached=False`` (fresh read; it
    runs before the cache may exist and must not seed it before later startup terms).  ``inertia_ratio [E, B]`` (hand-configuration mode) replaces the default ``new / nominal`` factor on the nominal inertia -- IsaacSim only
    (IsaacGym recomputes the inertia from the mass, MuJoCo-warp leaves ``body_inertia`` untouched)."""
    simulator = env.simulator
    backend = _backend(simulator)
    env_ids_cpu = env_ids.to(device="cpu", dtype=torch.long)
    if backend == "isaacgym":
        gym = simulator.gym
        bidx = _isaacgym_body_indices(simulator, body_names)
        for k, env_id in enumerate(env_ids_cpu.tolist()):
            env_ptr, actor = simulator.envs[env_id], simulator.robot_handles[env_id]
            props = gym.get_actor_rigid_body_properties(env_ptr, actor)
            for j, b in enumerate(bidx):
                props[b].mass = float(new[k, j])
            gym.set_actor_rigid_body_properties(env_ptr, actor, props, recomputeInertia=True)
    elif backend == "isaacsim":
        robot = _isaacsim_robot(simulator)
        view = robot.root_physx_view
        ids = _isaacsim_body_ids_cached(env, robot, body_names)
        masses = _physx_cached(env, "masses", view.get_masses) if cached else view.get_masses()  # [N, B_all] (CPU per IsaacLab)
        masses[env_ids_cpu[:, None], ids] = new.to(device=masses.device, dtype=masses.dtype)
        view.set_masses(masses, env_ids_cpu)
        # inertia ∝ mass about the unchanged CoM — IsaacLab mdp.randomize_rigid_body_mass(recompute_inertia=True) semantics
        nominal_inertia = _cached_nominal(env, NOMINAL_INERTIA_ATTR, body_names, _read_inertias)
        if nominal_inertia is not None and callable(getattr(view, "get_inertias", None)):
            inertias = _physx_cached(env, "inertias", view.get_inertias) if cached else view.get_inertias()  # [N, B_all, 9]
            ratio_src = inertia_ratio if inertia_ratio is not None else new / nominal[env_ids_cpu]
            ratio = ratio_src.to(device=inertias.device, dtype=inertias.dtype)[..., None]
            inertias[env_ids_cpu[:, None], ids] = nominal_inertia[env_ids_cpu].to(inertias.device, inertias.dtype) * ratio
            view.set_inertias(inertias, env_ids_cpu)
    elif backend == "mujoco_warp":
        ids = _mujoco_body_ids(simulator.backend.model, body_names)
        field = _warp_field(simulator, "body_mass")  # NOTE: like holosoma's stock warp mass term, body_inertia is untouched
        dev = field.device
        field[env_ids.to(dev, torch.long)[:, None], ids.to(dev)[None, :]] = new.to(device=dev, dtype=field.dtype)
    else:
        raise RandomizerNotSupportedError(
            f"Unsupported simulator type '{type(simulator).__name__}' for EE mass randomization."
        )

def randomize_ee_mass_startup(
    env: Any,
    env_ids: Sequence[int] | torch.Tensor | None = None,
    *,
    ee_body_names: Sequence[str] = DEFAULT_EE_BODY_NAMES,
    ee_mass_scale_range: Sequence[float] = DEFAULT_EE_MASS_SCALE_RANGE,
    ee_added_mass_range: Sequence[float] = DEFAULT_EE_ADDED_MASS_RANGE,
    enable_scale: bool = True,
    enable_added_mass: bool = True,
    enabled: bool = True,
    **_,
) -> None:
    """Randomize end-effector (palm) masses at startup: mass <- mass * U(scale) + U(added) [kg]."""
    if not enabled:
        return

    scale_lo, scale_hi = _check_range("ee_mass_scale_range", ee_mass_scale_range, allow_negative=False)
    add_lo, add_hi = _check_range("ee_added_mass_range", ee_added_mass_range, allow_negative=True)
    body_names = list(ee_body_names)
    if not body_names:
        raise ValueError("ee_body_names must not be empty")

    logger.info(
        "[Randomization] EE mass: bodies=%s scale=%s (enabled=%s) added_kg=%s (enabled=%s)",
        body_names,
        (scale_lo, scale_hi),
        enable_scale,
        (add_lo, add_hi),
        enable_added_mass,
    )

    idx = _ensure_env_ids_tensor(env, env_ids)
    if idx.numel() == 0:
        return

    env._randomize_ee_mass = True
    env._ee_mass_scale_range = (scale_lo, scale_hi)
    env._ee_added_mass_range = (add_lo, add_hi)
    _startup_mass_write(
        env, idx, body_names, (scale_lo, scale_hi), (add_lo, add_hi), enable_scale, enable_added_mass,
        term=randomize_ee_mass_startup, label="EE mass",
    )

def _startup_mass_write(
    env: Any,
    idx: torch.Tensor,
    body_names: Sequence[str],
    scale_range: tuple[float, float],
    added_range: tuple[float, float],
    enable_scale: bool,
    enable_added_mass: bool,
    *,
    term: Callable[..., None],
    label: str,
) -> None:
    """Backend shared by :func:`randomize_ee_mass_startup` and :func:`randomize_body_mass_startup`: one absolute write
    ``nominal * U(scale) + U(added)`` on ``body_names`` for the envs ``idx`` (IsaacGym / IsaacSim / MuJoCo-warp).
    ``term`` is the decorated public callable (MuJoCo field lookup); ``label`` names the term in log / error messages."""
    simulator = env.simulator
    scale_lo, scale_hi = scale_range
    add_lo, add_hi = added_range
    body_names = list(body_names)

    if hasattr(simulator, "gym"):
        # IsaacGym: edit rigid body properties per env (same pattern as randomize_mass_startup).
        gym = simulator.gym
        present = [n for n in body_names if n in simulator._body_list]
        missing = sorted(set(body_names) - set(present))
        if missing:
            logger.warning(f"[{term.__name__}] bodies not in articulation, skipped: {missing}")
        if present:
            nominal_ee_masses(env, present)  # cache URDF masses BEFORE scaling (reset term rescales from these)
        for env_id in idx.tolist():
            env_ptr = simulator.envs[env_id]
            actor = simulator.robot_handles[env_id]
            body_props = gym.get_actor_rigid_body_properties(env_ptr, actor)
            for body_name in present:
                body_index = simulator._body_list.index(body_name)
                if enable_scale:
                    body_props[body_index].mass *= np.random.uniform(scale_lo, scale_hi)
                if enable_added_mass:
                    body_props[body_index].mass += np.random.uniform(add_lo, add_hi)
            gym.set_actor_rigid_body_properties(env_ptr, actor, body_props, recomputeInertia=True)
    elif simulator.__class__.__name__ == "IsaacSim":


        env_ids_cpu = idx.to(device="cpu", dtype=torch.long)
        nominal = nominal_ee_masses(env, body_names)  # IsaacLab data.default_mass = URDF nominal (cached for the reset term)
        n_sel, n_body = env_ids_cpu.shape[0], len(body_names)
        scale = scale_lo + (scale_hi - scale_lo) * torch.rand(n_sel, n_body) if enable_scale else torch.ones(n_sel, n_body)
        added = add_lo + (add_hi - add_lo) * torch.rand(n_sel, n_body) if enable_added_mass else torch.zeros(n_sel, n_body)
        _write_masses(env, idx, body_names, nominal[env_ids_cpu] * scale + added, nominal, cached=False)
    elif getattr(getattr(simulator, "simulator_config", None), "mujoco_backend", None) == MujocoBackend.WARP:
        from holosoma.simulator.mujoco.backends.warp_randomization import randomize_field

        nominal_ee_masses(env, body_names)  # cache BEFORE scaling (reset term rescales from these)
        field = getattr(term, MUJOCO_FIELD_ATTR)
        if enable_scale:
            randomize_field(
                simulator, field=field, ranges=(scale_lo, scale_hi), env_ids=idx,
                entity_names=body_names, entity_type="body", operation="scale",
            )
        if enable_added_mass:
            randomize_field(
                simulator, field=field, ranges=(add_lo, add_hi), env_ids=idx,
                entity_names=body_names, entity_type="body", operation="add",
            )
    else:
        raise RandomizerNotSupportedError(
            f"Unsupported simulator type '{type(simulator).__name__}' for {label} randomization."
        )
