"""PhysX combine modes for trimesh and ``load_obj`` terrain materials.

A terrain term may define ``friction_combine_mode`` and ``restitution_combine_mode``
on itself or its ``_cfg`` / ``cfg``. These values are forwarded to the mesh material;
when absent, PhysX uses its default ``average`` mode. Using ``multiply`` matches
the plane material's friction randomization behavior.

This module uses only the standard library and can be imported without Isaac Sim.
"""

from __future__ import annotations

from typing import Any

#: The two ``RigidBodyMaterialCfg`` keywords a terrain term may publish.
COMBINE_MODE_KEYS = ("friction_combine_mode", "restitution_combine_mode")
#: Isaac Lab ``RigidBodyMaterialCfg`` accepted values (PhysX ``PxCombineMode``).
COMBINE_MODES = ("average", "min", "multiply", "max")
#: What PhysX uses when a material sets no combine mode.
PHYSX_DEFAULT_COMBINE_MODE = "average"


def _published(terrain_state: Any, key: str) -> Any:
    """The value a terrain term publishes for ``key``: the term first, then its cfg; None when neither has it."""
    value = getattr(terrain_state, key, None)
    if value is not None:
        return value
    for cfg_attr in ("_cfg", "cfg"):
        cfg = getattr(terrain_state, cfg_attr, None)
        if cfg is None:
            continue
        value = getattr(cfg, key, None)
        if value is not None:
            return value
    return None


def terrain_material_combine_modes(terrain_state: Any) -> dict[str, str]:
    """``RigidBodyMaterialCfg`` combine-mode keywords for the mesh material of ``terrain_state`` (the terrain term).

    Empty when the term publishes nothing (stock behaviour); an unknown mode name fails loudly instead of being silently
    dropped by the material.
    """
    modes: dict[str, str] = {}
    for key in COMBINE_MODE_KEYS:
        mode = _published(terrain_state, key)
        if mode is None:
            continue
        if not isinstance(mode, str) or mode not in COMBINE_MODES:
            raise ValueError(
                f"terrain term {type(terrain_state).__name__} publishes {key}={mode!r}; expected one of {COMBINE_MODES}"
            )
        modes[key] = mode
    return modes


def describe_combine_modes(modes: dict[str, str]) -> str:
    """Receipt fragment for the terrain mesh material, e.g. ``friction=multiply restitution=multiply`` or
    ``default(average)`` when nothing is published."""
    if not modes:
        return f"default({PHYSX_DEFAULT_COMBINE_MODE})"
    return " ".join(f"{key.removesuffix('_combine_mode')}={modes[key]}" for key in COMBINE_MODE_KEYS if key in modes)
