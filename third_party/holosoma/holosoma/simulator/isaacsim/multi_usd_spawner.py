"""Explicit per-environment multi-USD spawner for Isaac Lab.

``spawn_multi_usd_explicit`` extends Isaac Lab's ``spawn_multi_asset`` to accept
per-environment variant IDs without creating one prototype for every environment. It:

* spawns ONE prototype per distinct USD (``UsdFileCfg.func`` = the stock ``spawn_from_usd`` with the same rigid /
  articulation / contact-sensor properties),
* copies the prototype chosen by ``cfg.variant_ids[env_index]`` into every matched env prim with ``Sdf.CopySpec``
  (exactly what upstream does; the env index is parsed from the prim path, so the layout does not depend on the
  order in which ``find_matching_prim_paths`` returns the envs),
* removes the template scope and sets the ``/isaaclab/spawn/multi_assets`` carb flag like upstream.

Isaac Lab API (Isaac Lab 2.3 / Isaac Sim 5.1): ``isaaclab.sim.MultiUsdFileCfg`` (fields
``usd_path: str | list[str]``, ``random_choice``), ``isaaclab.sim.UsdFileCfg`` + its ``.replace(usd_path=...)``,
``isaaclab.sim.find_matching_prim_paths``, ``isaaclab.sim.utils.get_current_stage`` (falls back to
``omni.usd.get_context().get_stage()``), ``isaaclab.utils.configclass``; ``pxr.Sdf.{ChangeBlock, CreatePrimInLayer, CopySpec, Path}``;
``carb.settings``. ``isaacsim.py`` falls back to the stock
``MultiUsdFileCfg(random_choice=False)`` round-robin layout if this spawner raises.
"""

from __future__ import annotations

import re
import uuid
from collections.abc import Callable

import isaaclab.sim as sim_utils
from isaaclab.utils import configclass
from loguru import logger
from pxr import Sdf

from holosoma.simulator.isaacsim.multi_usd import env_index_from_prim_path

#: Fields of the cfg that are NOT copied into the per-prototype ``UsdFileCfg`` (mirrors ``spawn_multi_usd_file``).
_NON_TEMPLATE_FIELDS = frozenset({"func", "usd_path", "random_choice", "variant_ids", "template_root"})
#: Same regex as upstream: a literal prim path only has these characters, anything else is a regex expression.
_LITERAL_PATH_RE = re.compile(r"^[a-zA-Z0-9/_]+$")


def _get_stage():
    try:
        return sim_utils.utils.get_current_stage()
    except Exception:  # noqa: BLE001 - older Isaac Lab without the helper
        import omni.usd

        return omni.usd.get_context().get_stage()


def _set_multi_asset_flag() -> None:
    try:
        import carb

        carb.settings.get_settings().set_bool("/isaaclab/spawn/multi_assets", True)
    except Exception as e:  # noqa: BLE001 - informative flag only (InteractiveScene warns on replicate_physics)
        logger.warning(f"[MULTI-USD] could not set /isaaclab/spawn/multi_assets: {e}")


def spawn_multi_usd_explicit(prim_path: str, cfg: "ExplicitMultiUsdFileCfg", translation=None, orientation=None):
    """Spawn ``cfg.usd_path[cfg.variant_ids[i]]`` under every ``.../env_i/<asset>`` prim matched by ``prim_path``.

    Envs whose index cannot be parsed from the prim path, or that lie outside ``variant_ids``, fall back to the stock
    round-robin rule ``k % len(usd_path)`` (``k`` = match order) and are counted in the returned log line.
    Returns the prim of the first matched env (like upstream).
    """
    prim_path = str(prim_path)
    if not prim_path.startswith("/"):
        raise ValueError(f"Prim path '{prim_path}' is not global. It must start with '/'.")
    root_path, asset_name = prim_path.rsplit("/", 1)
    is_regex = _LITERAL_PATH_RE.match(root_path) is None
    if is_regex and root_path != "":
        source_prim_paths = list(sim_utils.find_matching_prim_paths(root_path))
        if not source_prim_paths:
            raise RuntimeError(f"Unable to find source prim path: '{root_path}'. Please create the prim before spawning.")
    else:
        source_prim_paths = [root_path]

    usd_paths = [cfg.usd_path] if isinstance(cfg.usd_path, str) else list(cfg.usd_path)
    if not usd_paths:
        raise ValueError("ExplicitMultiUsdFileCfg.usd_path is empty")
    variant_ids = None if cfg.variant_ids is None else [int(v) for v in cfg.variant_ids]
    if variant_ids is not None:
        bad = [v for v in variant_ids if not 0 <= v < len(usd_paths)]
        if bad:
            raise ValueError(f"variant_ids contain ids outside [0, {len(usd_paths)}): {sorted(set(bad))[:8]}")

    stage = _get_stage()
    template_prim_path = f"{cfg.template_root}_{uuid.uuid4().hex[:8]}"
    while stage.GetPrimAtPath(template_prim_path).IsValid():
        template_prim_path = f"{cfg.template_root}_{uuid.uuid4().hex[:8]}"
    stage.DefinePrim(template_prim_path, "Scope")

    try:
        # one UsdFileCfg per distinct USD, carrying every spawn-time property of the multi cfg (upstream recipe)
        usd_template_cfg = sim_utils.UsdFileCfg()
        for attr_name, attr_value in cfg.__dict__.items():
            if attr_name in _NON_TEMPLATE_FIELDS:
                continue
            setattr(usd_template_cfg, attr_name, attr_value)
        proto_prim_paths: list[str] = []
        for index, usd_path in enumerate(usd_paths):
            usd_cfg = usd_template_cfg.replace(usd_path=usd_path)
            proto_prim_path = f"{template_prim_path}/Asset_{index:04d}"
            usd_cfg.func(proto_prim_path, usd_cfg, translation=translation, orientation=orientation)
            proto_prim_paths.append(proto_prim_path)

        # copy the chosen prototype into every env prim (Sdf.CopySpec replaces any existing spec at the destination)
        n_fallback = 0
        chosen: list[int] = []
        with Sdf.ChangeBlock():
            for order, source_prim_path in enumerate(source_prim_paths):
                env_index = env_index_from_prim_path(source_prim_path)
                if variant_ids is not None and env_index is not None and env_index < len(variant_ids):
                    vid = variant_ids[env_index]
                else:
                    vid = order % len(proto_prim_paths)
                    n_fallback += 1
                dst = f"{source_prim_path}/{asset_name}"
                env_spec = Sdf.CreatePrimInLayer(stage.GetRootLayer(), dst)
                Sdf.CopySpec(env_spec.layer, Sdf.Path(proto_prim_paths[vid]), env_spec.layer, Sdf.Path(dst))
                chosen.append(vid)
    finally:
        # prototypes are scenery for the copy only; never leave them in the stage (PhysX would parse them)
        if stage.GetPrimAtPath(template_prim_path).IsValid():
            stage.RemovePrim(template_prim_path)

    _set_multi_asset_flag()
    counts = [chosen.count(k) for k in range(len(usd_paths))]
    logger.info(
        f"[MULTI-USD] explicit spawn of {len(source_prim_paths)} '{asset_name}' prims from {len(usd_paths)} USDs: "
        f"counts={counts}, round-robin fallbacks={n_fallback}"
    )
    return stage.GetPrimAtPath(f"{source_prim_paths[0]}/{asset_name}")


@configclass
class ExplicitMultiUsdFileCfg(sim_utils.MultiUsdFileCfg):
    """``MultiUsdFileCfg`` whose spawner takes an explicit per-env variant id (``variant_ids[env_index]`` indexes
    ``usd_path``).  ``random_choice`` is ignored (kept for type compatibility)."""

    func: Callable = spawn_multi_usd_explicit

    variant_ids: list[int] | None = None
    """Variant id per env index (``len == num_envs``).  ``None`` -> stock round-robin ``k % len(usd_path)``."""

    template_root: str = "/World/HolosomaMultiUsdTemplate"
    """Prefix of the transient scope that holds the prototypes while they are copied (removed afterwards)."""


__all__ = ["ExplicitMultiUsdFileCfg", "spawn_multi_usd_explicit"]
