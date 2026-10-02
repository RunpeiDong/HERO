"""Export dual-actor ONNX policies and their runtime metadata."""

from __future__ import annotations

import logging

from loguru import logger
import os
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import torch
from torch import nn

from holosoma.utils.robot_asset_metadata import ROBOT_COLLISION_EXPORT_KEYS

from .layout import HISTORY_LAYOUT_FRAME_MAJOR, HISTORY_LAYOUT_TERM_MAJOR, SUPPORTED_HISTORY_LAYOUTS, ActorObsLayout
from .modules import GROUP_KEYS, PPODualActor

ONNX_INPUT_NAMES: tuple[str, str] = ("actor_obs_lower_body", "actor_obs_upper_body")
ONNX_INPUT_NAMES_SINGLE: tuple[str] = ("actor_obs",)
"""Input of a single-actor (whole-body) export; the same frame-major vector as the dual inputs."""
WHOLE_BODY = "whole_body"
ONNX_OUTPUT_NAME = "action"
ONNX_OPSET = 13
ACTION_CONTRACT_HERO_RESIDUAL_UPPER = "hero_residual_upper_v1"
"""Network output ``a`` (29) -> target ``q = q_default + scale * a`` with the upper-body (arm) slice additionally
offset by ``q_ref_arm - q_default_arm`` (HERO residual upper-body action; applied by the env action term and by
the deploy stack, NOT inside the graph)."""

HERO_SIDECAR_SUFFIX = "_hero.json"
"""``model_00500.onnx`` -> ``model_00500_hero.json`` (same directory, same stem)."""
HERO_SIDECAR_SCHEMA = "hero_export_v1"
"""Sidecar ``schema`` value; must equal ``sim2sim.policy_hero_export.SIDECAR_SCHEMA`` (a different value only warns there)."""
HERO_OBJECT_TERM_NAMES: tuple[str, ...] = ("h17_obj_pos_b", "h18_obj_ori_b", "h19_has_object_flag")
HERO_ANCHOR_TERM_NAME = "h20_ref_root_pose_b"
"""Delta-anchor drift-feedback term (== ``config_values.observation.ANCHOR_TERM_NAME`` / ``sim2sim.policy_hero_export.ANCHOR_TERM``)."""
HERO_H20_FUTURE_STEPS_KEY = "h20_future_steps"
"""Sidecar / ONNX-metadata key of the actor h20 term's ``future_steps`` (:func:`hero_h20_future_steps`): the clip frames
(control steps ahead of ``t``) h20 is built from.  ``None`` for exports without h20.  The sim2sim reader prefers it over a
``hero_command.ref_root_pose_future_steps`` copy and raises when it disagrees with the exported h20 dim."""
HERO_ANCHOR2_TERM_NAMES: tuple[str, ...] = ("h21_ref_root_rot_b", "h22_ref_root_height_b", "h23_base_lin_vel_odom")
HERO_PLUS_TERM_NAMES: tuple[str, ...] = (HERO_ANCHOR_TERM_NAME, *HERO_ANCHOR2_TERM_NAMES)
"""Every Delta-anchor (drift / root feedback) observation term, in sorted order (h20 < h21 < h22 < h23)."""
HERO_PLUS_PER_FRAME_DIM: dict[str, int] = {"h20_ref_root_pose_b": 4, "h21_ref_root_rot_b": 6, "h22_ref_root_height_b": 1}
"""Per-future-frame width of the delta-anchor terms that carry ``future_steps`` (``dim == len(future_steps) x per_frame_dim``);
``h23_base_lin_vel_odom`` is absent on purpose (no horizon)."""
HERO_H21_FUTURE_STEPS_KEY = "h21_future_steps"
HERO_H22_FUTURE_STEPS_KEY = "h22_future_steps"
HERO_FUTURE_STEPS_KEYS: dict[str, str] = {
    HERO_ANCHOR_TERM_NAME: HERO_H20_FUTURE_STEPS_KEY,
    "h21_ref_root_rot_b": HERO_H21_FUTURE_STEPS_KEY,
    "h22_ref_root_height_b": HERO_H22_FUTURE_STEPS_KEY,
}
"""Sidecar / ONNX-metadata key per Delta-anchor term with a horizon (``h2N_future_steps``; None for exports without the term)."""
HERO_ANCHOR_TERMS_KEY = "anchor_terms"
"""Sidecar / ONNX-metadata key of the per-term Delta-anchor block (:func:`hero_anchor_terms_block`): ``{term: {"dim", "per_frame_dim",
"future_steps", "noise"}}`` for every Delta-anchor term of the actor layout (``noise`` = the actor copy's training-only ``noise*``
params, ``{}`` = exact; ``future_steps`` / ``per_frame_dim`` None for h23); None when the layout carries no Delta-anchor term."""
HERO_SIDECAR_ANCHOR2_KEYS: tuple[str, ...] = (HERO_H21_FUTURE_STEPS_KEY, HERO_H22_FUTURE_STEPS_KEY)
HERO_COMMAND_SIDECAR_KEYS: tuple[str, ...] = (
    "ref_lookahead_frames",
    "h_cmd_default",
    "h_cmd_min",
    "stand_speed_thr",
    "stand_yaw_rate_thr",
    "stand_window_s",
    "zero_waist_when_walking",
    "h_offset_from_clip",
    "stand_flag_mode",

    # Object-rule fields keep deployment behavior consistent with the motion configuration.
    # hero_command_block skips keys that the configuration does not define.
    "object_box_side",
    "object_box_tol",
    "max_start_bottom_z_m",
)
"""``HeroMotionConfig`` values the sim2sim controller rebuilds the command terms from (``hero_command`` block; the
reader's defaults equal ``config_values/command.py``, so the block is informational for the stock presets, and the
object-rule keys make ``--policy hero_export`` apply the training-time box rule without resolving the preset)."""
HERO_H2_MARKER_TERM = "h14_ref_lower_dof_pos"
HERO_HISTORY_PADDING = "zeros"
"""holosoma zero-fills the H-1 missing frames at episode start (observation/manager.py _apply_history)."""
HERO_SIDECAR_REQUIRED_KEYS: tuple[str, ...] = (
    "schema",
    "algo",
    "preset",
    "hero_arm",
    "iteration",
    "onnx_file",
    "onnx_inputs",
    "onnx_output",
    "history_layout",
    "history_padding",
    "actor_obs_layout",
    "actor_obs_terms",
    "actor_obs_term_dims",
    "actor_history_length",
    "actor_frame_dim",
    "actor_obs_dim",
    "action_contract",
    "action_split",
    "action_groups",
    "body_keys",
    "dof_names",
    "kp",
    "kd",
    "action_scale",
    "default_dof_pos",
    "effort_limit",
    "has_object",
    "object_terms",
    "object_urdf_path",
    "robot_urdf_path",
    "init_noise_std",
    "policy_dt",
    "clip_observations",
    "wandb_run_path",
)
HERO_SIDECAR_OPTIONAL_KEYS: tuple[str, ...] = (
    "hero_command",
    "command_ranges",
    "dof_pos_limits",
    "action_contract_detail",
    HERO_H20_FUTURE_STEPS_KEY,
    "hero_expand",
    HERO_ANCHOR_TERMS_KEY,
    *HERO_SIDECAR_ANCHOR2_KEYS,
)
"""Always written (None when unknown) but NOT required by the sim2sim reader: ``hero_command`` (:func:`hero_command_block`;
the reader's defaults equal ``config_values/command.py``), holosoma's ``command_ranges``, the plant's ``dof_pos_limits``,
the action term's ``action_contract_detail``, the delta-anchor ``h20_future_steps`` horizon (consumed by the reader when present),
the ``hero_expand`` fine-tune lineage summary (``PPODual.hero_expand_summary``; provenance only, may carry a ``lineage`` chain)
the per-term Delta-anchor ``anchor_terms`` block (:func:`hero_anchor_terms_block`) and the root-orientation and height horizons
:data:`HERO_SIDECAR_ANCHOR2_KEYS` (``h21_future_steps`` / ``h22_future_steps``; consumed by the reader when present, each must equal
``h20_future_steps``)."""


def hero_sidecar_path(onnx_file_path: str) -> str:
    """Sidecar path next to the ONNX: ``<dir>/<stem>_hero.json``."""
    p = Path(onnx_file_path)
    return str(p.with_name(p.stem + HERO_SIDECAR_SUFFIX))


def _jsonable_scalar(value: Any) -> Any:
    """Plain JSON scalar (bool / int / float / str / None) or ``str(value)``; numpy / torch scalars via ``.item()``."""
    if isinstance(value, (bool, int, float, str)) or value is None:
        return value
    item = getattr(value, "item", None)
    if callable(item):
        try:
            return _jsonable_scalar(item())
        except Exception:  # noqa: BLE001
            pass
    return str(value)


def hero_command_block(experiment_config: Any) -> dict[str, Any] | None:
    """``{key: value}`` of :data:`HERO_COMMAND_SIDECAR_KEYS` read from the experiment's ``motion_command`` motion config.

    Accepts an ``ExperimentConfig`` or its ``to_serializable_dict()`` mapping. Only
    configured keys are emitted; returns None when the motion configuration is unavailable."""

    def _get(node: Any, key: str) -> Any:
        if node is None:
            return None
        if isinstance(node, Mapping):
            return node.get(key)
        return getattr(node, key, None)

    cmd = _get(experiment_config, "command")
    setup = _get(cmd, "setup_terms")
    term = _get(setup, "motion_command") if isinstance(setup, Mapping) else None
    params = _get(term, "params")
    mc = _get(params, "motion_config") if isinstance(params, Mapping) else None
    if mc is None:
        return None
    out: dict[str, Any] = {}
    for key in HERO_COMMAND_SIDECAR_KEYS:
        if isinstance(mc, Mapping):
            if key not in mc:
                continue
            value = mc[key]
        else:
            if not hasattr(mc, key):
                continue
            value = getattr(mc, key)
        out[key] = _jsonable_scalar(value)
    return out or None


def _cfg_get(node: Any, key: str) -> Any:
    """``node[key]`` for mappings, ``getattr`` otherwise, None for None."""
    if node is None:
        return None
    if isinstance(node, Mapping):
        return node.get(key)
    return getattr(node, key, None)


def _group_terms(config: Any, group: str) -> Mapping[str, Any] | None:
    """The ``terms`` mapping of observation group ``group`` from an ``ExperimentConfig`` / its dict / an ``ObservationManagerCfg``
    / a mapping of that shape; None when it cannot be found."""
    groups = _cfg_get(config, "groups")
    if groups is None:
        groups = _cfg_get(_cfg_get(config, "observation"), "groups")
    terms = _cfg_get(_cfg_get(groups, group), "terms")
    return terms if isinstance(terms, Mapping) else None


def default_future_steps(term: str) -> list[int]:


    from hero_isaacsim.config_values import observation as O  # lazy: the config tables are not an export dependency

    for attr in (f"H{term[1:3]}_FUTURE_STEPS", "HERO_ANCHOR2_FUTURE_STEPS", "H20_FUTURE_STEPS"):
        steps = getattr(O, attr, None)
        if steps is not None:
            return [int(_jsonable_scalar(s)) for s in steps]
    raise AttributeError("config_values.observation defines no H20_FUTURE_STEPS")  # pragma: no cover


def hero_term_future_steps(config: Any, term: str, group: str = "actor_obs") -> list[int] | None:


    if term not in HERO_PLUS_PER_FRAME_DIM:
        return None
    terms = _group_terms(config, group)
    if terms is None or term not in terms:
        return None
    params = _cfg_get(terms[term], "params")
    steps = _cfg_get(params, "future_steps") if isinstance(params, Mapping) else None
    if steps is None:
        return default_future_steps(term)
    return [int(_jsonable_scalar(s)) for s in steps]


def hero_h20_future_steps(config: Any, group: str = "actor_obs") -> list[int] | None:


    return hero_term_future_steps(config, HERO_ANCHOR_TERM_NAME, group=group)


def hero_term_noise_params(config: Any, term: str, group: str = "actor_obs") -> dict[str, Any] | None:
    """The training-only ``noise*`` params the ``term`` of ``group`` is called with (``{}`` when the term is present but exact,
    e.g. every critic copy; None when ``group`` has no such term).  Same config shapes as :func:`hero_term_future_steps`."""
    terms = _group_terms(config, group)
    if terms is None or term not in terms:
        return None
    params = _cfg_get(terms[term], "params")
    if not isinstance(params, Mapping):
        return {}
    return {str(k): _jsonable_scalar(v) for k, v in params.items() if str(k).startswith("noise")}


def hero_anchor_terms_block(config: Any, term_dims: Mapping[str, int], group: str = "actor_obs") -> dict[str, dict[str, Any]] | None:


    out: dict[str, dict[str, Any]] = {}
    for term in HERO_PLUS_TERM_NAMES:
        if term not in term_dims:
            continue
        per_frame = HERO_PLUS_PER_FRAME_DIM.get(term)
        out[term] = {
            "dim": int(term_dims[term]),
            "per_frame_dim": per_frame,
            "future_steps": hero_term_future_steps(config, term, group=group) if per_frame is not None else None,
            "noise": hero_term_noise_params(config, term, group=group),
        }
    return out or None


def _h20_steps_list(steps: Any) -> list[int] | None:
    """``[int, ...]`` or None (empty / missing -> None)."""
    if steps is None:
        return None
    out = [int(_jsonable_scalar(s)) for s in steps]
    return out or None


def _anchor_terms_dict(block: Any) -> dict[str, dict[str, Any]] | None:
    """Plain-JSON copy of an ``anchor_terms`` block (``{term: {...}}``) or None (empty / missing / not a mapping -> None)."""
    if not isinstance(block, Mapping) or not block:
        return None
    out: dict[str, dict[str, Any]] = {}
    for term, entry in block.items():
        entry = dict(entry) if isinstance(entry, Mapping) else {}
        out[str(term)] = {
            "dim": None if entry.get("dim") is None else int(_jsonable_scalar(entry["dim"])),
            "per_frame_dim": None if entry.get("per_frame_dim") is None else int(_jsonable_scalar(entry["per_frame_dim"])),
            "future_steps": _h20_steps_list(entry.get("future_steps")),
            "noise": None if entry.get("noise") is None else {str(k): _jsonable_scalar(v) for k, v in dict(entry["noise"]).items()},
        }
    return out


def build_hero_sidecar(
    metadata: dict[str, Any],
    onnx_file_path: str,
    *,
    preset: str | None,
    effort_limit: Sequence[float] | None = None,
    dof_pos_limits: tuple[Sequence[float], Sequence[float]] | None = None,
    object_urdf_path: str | None = None,
    policy_dt: float | None = None,
    action_contract_detail: dict[str, Any] | None = None,
    clip_observations: float | None = None,
    hero_command: Mapping[str, Any] | None = None,
    wandb_run_path: str | None = None,
    h20_future_steps: Sequence[int] | None = None,
    hero_expand: Mapping[str, Any] | None = None,
    h21_future_steps: Sequence[int] | None = None,
    h22_future_steps: Sequence[int] | None = None,
    anchor_terms: Mapping[str, Mapping[str, Any]] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:


    layout = metadata.get("actor_obs_layout") or {}
    groups = list(layout.get("groups") or [])
    actor_group = next((g for g in groups if g.get("name") == "actor_obs"), groups[0] if groups else {})
    terms = [str(t) for t in (actor_group.get("terms") or [])]
    term_dims = [int(d) for d in (actor_group.get("term_dims") or [])]
    history = int(actor_group.get("history_length") or 1)
    frame_dim = int(actor_group.get("frame_dim") or sum(term_dims))
    total_dim = int(actor_group.get("dim") or frame_dim * history)
    object_terms = [t for t in terms if t in HERO_OBJECT_TERM_NAMES]
    sidecar: dict[str, Any] = {
        "schema": HERO_SIDECAR_SCHEMA,
        "algo": str(metadata.get("algo") or "PPODual"),
        "preset": preset,
        "hero_arm": "h2" if HERO_H2_MARKER_TERM in terms else "h1",
        "iteration": metadata.get("iteration"),
        "onnx_file": os.path.basename(onnx_file_path),
        "onnx_inputs": list(metadata.get("onnx_inputs") or ONNX_INPUT_NAMES),
        "onnx_output": str(metadata.get("onnx_output") or ONNX_OUTPUT_NAME),
        "onnx_inputs_note": metadata.get("onnx_inputs_note"),
        "history_layout": metadata.get("history_layout"),
        "history_layout_note": metadata.get("history_layout_note"),
        "history_padding": HERO_HISTORY_PADDING,
        "actor_obs_layout": layout,
        "actor_obs_terms": terms,
        "actor_obs_term_dims": term_dims,
        "actor_history_length": history,
        "actor_frame_dim": frame_dim,
        "actor_obs_dim": total_dim,
        "action_contract": str(metadata.get("action_contract") or ACTION_CONTRACT_HERO_RESIDUAL_UPPER),
        "action_contract_detail": dict(action_contract_detail) if action_contract_detail else None,
        "action_split": [int(n) for n in (metadata.get("action_split") or [])],
        "action_groups": metadata.get("action_groups"),
        "body_keys": list(metadata.get("body_keys") or GROUP_KEYS),
        "dof_names": list(metadata.get("dof_names") or []),
        "kp": metadata.get("kp"),
        "kd": metadata.get("kd"),
        "action_scale": metadata.get("action_scale"),
        "default_dof_pos": metadata.get("default_dof_pos"),
        "effort_limit": [float(x) for x in effort_limit] if effort_limit is not None else None,
        "dof_pos_limits": (
            {"lower": [float(x) for x in dof_pos_limits[0]], "upper": [float(x) for x in dof_pos_limits[1]]}
            if dof_pos_limits is not None
            else None
        ),
        "has_object": bool(object_terms),
        "object_terms": object_terms,
        "object_urdf_path": object_urdf_path,
        "robot_urdf_path": metadata.get("robot_urdf_path"),
        "init_noise_std": metadata.get("init_noise_std"),
        "policy_dt": float(policy_dt) if policy_dt is not None else None,
        "clip_observations": float(clip_observations) if clip_observations is not None else None,
        "hero_command": dict(hero_command) if hero_command else None,
        "wandb_run_path": str(wandb_run_path) if wandb_run_path else None,
        "command_ranges": metadata.get("command_ranges"),
        HERO_H20_FUTURE_STEPS_KEY: _h20_steps_list(h20_future_steps if h20_future_steps is not None else metadata.get(HERO_H20_FUTURE_STEPS_KEY)),
        "hero_expand": dict(hero_expand) if hero_expand else None,
        HERO_H21_FUTURE_STEPS_KEY: _h20_steps_list(h21_future_steps if h21_future_steps is not None else metadata.get(HERO_H21_FUTURE_STEPS_KEY)),
        HERO_H22_FUTURE_STEPS_KEY: _h20_steps_list(h22_future_steps if h22_future_steps is not None else metadata.get(HERO_H22_FUTURE_STEPS_KEY)),
        HERO_ANCHOR_TERMS_KEY: _anchor_terms_dict(anchor_terms if anchor_terms is not None else metadata.get(HERO_ANCHOR_TERMS_KEY)),
    }
    if metadata.get("history_layout") == HISTORY_LAYOUT_FRAME_MAJOR and "frame_major_to_term_major_perm" in metadata:
        sidecar["frame_major_to_term_major_perm"] = list(metadata["frame_major_to_term_major_perm"])
    # Runtime-only metadata is absent on exports or
    # environments that did not publish a prepared collision asset.
    for key in ROBOT_COLLISION_EXPORT_KEYS:
        if key in metadata:
            sidecar[key] = metadata[key]
    if extra:
        sidecar.update(dict(extra))
    missing = [k for k in HERO_SIDECAR_REQUIRED_KEYS if k not in sidecar]
    if missing:  # pragma: no cover - the literal above always carries them
        raise KeyError(f"hero sidecar missing keys {missing}")
    return sidecar


class DualActorOnnxWrapper(nn.Module):
    """``(actor_obs_lower_body, actor_obs_upper_body) -> action`` with optional input permutation + normalizer.

    Parameters
    ----------
    actor : PPODualActor
    input_perm : long tensor ``[H*D]`` or ``None``.  When given, each input is re-indexed with
        ``x.index_select(-1, input_perm)`` before the MLPs (frame-major -> term-major).
    obs_normalizer : module applied after the permutation when ``empirical_normalization`` (the
        normalizer was fitted on term-major training vectors, hence the order)."""

    def __init__(
        self,
        actor: PPODualActor,
        input_perm: torch.Tensor | None = None,
        obs_normalizer: nn.Module | None = None,
        empirical_normalization: bool = False,
    ):
        super().__init__()
        self.actor = actor
        self.empirical_normalization = bool(empirical_normalization)
        self.obs_normalizer = obs_normalizer if self.empirical_normalization else None
        if input_perm is not None:
            input_perm = torch.as_tensor(input_perm, dtype=torch.long)
            if input_perm.dim() != 1:
                raise ValueError(f"input_perm must be 1-D, got shape {tuple(input_perm.shape)}")
        self.input_perm: torch.Tensor | None
        if input_perm is None:
            self.input_perm = None
        else:
            self.register_buffer("input_perm", input_perm.clone())

    def _prepare(self, x: torch.Tensor) -> torch.Tensor:
        if self.input_perm is not None:
            x = x.index_select(-1, self.input_perm)
        if self.obs_normalizer is not None:
            x = self.obs_normalizer(x, update=False)
        return x

    def forward(self, actor_obs_lower_body: torch.Tensor, actor_obs_upper_body: torch.Tensor) -> torch.Tensor:
        lower = self.actor.actor_lower(self._prepare(actor_obs_lower_body))
        upper = self.actor.actor_upper(self._prepare(actor_obs_upper_body))
        return torch.cat([lower, upper], dim=-1)


class SingleActorOnnxWrapper(nn.Module):
    """``actor_obs -> action`` for a single whole-body actor (holosoma ``PPOActor``), with the same optional
    frame-major -> term-major input permutation and observation normalizer as :class:`DualActorOnnxWrapper`."""

    def __init__(
        self,
        actor: nn.Module,
        input_perm: torch.Tensor | None = None,
        obs_normalizer: nn.Module | None = None,
        empirical_normalization: bool = False,
    ):
        super().__init__()
        self.actor = actor
        self.empirical_normalization = bool(empirical_normalization)
        self.obs_normalizer = obs_normalizer if self.empirical_normalization else None
        self.input_perm: torch.Tensor | None
        if input_perm is None:
            self.input_perm = None
        else:
            input_perm = torch.as_tensor(input_perm, dtype=torch.long)
            if input_perm.dim() != 1:
                raise ValueError(f"input_perm must be 1-D, got shape {tuple(input_perm.shape)}")
            self.register_buffer("input_perm", input_perm.clone())

    def forward(self, actor_obs: torch.Tensor) -> torch.Tensor:
        x = actor_obs
        if self.input_perm is not None:
            x = x.index_select(-1, self.input_perm)
        if self.obs_normalizer is not None:
            x = self.obs_normalizer(x, update=False)
        return self.actor.act_inference({"actor_obs": x})


def hero_sidecar_from_algo(
    algo: Any,
    metadata: dict[str, Any],
    onnx_file_path: str,
    *,
    hero_expand: Mapping[str, Any] | None = None,
    extra: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """``model_XXXXX_hero.json`` content for any HERO agent: the robot / action / command fields are read from the
    agent's live env and attached experiment config (every lookup is best effort), the rest from ``metadata``."""
    env = algo.env
    exp = getattr(algo, "_experiment_config", None)
    preset = getattr(getattr(exp, "training", None), "name", None)
    robot_cfg = getattr(env, "robot_config", None)
    effort = getattr(robot_cfg, "dof_effort_limit_list", None)
    lower = getattr(robot_cfg, "dof_pos_lower_limit_list", None)
    upper = getattr(robot_cfg, "dof_pos_upper_limit_list", None)
    limits = (list(lower), list(upper)) if lower is not None and upper is not None else None
    object_urdf = getattr(getattr(robot_cfg, "object", None), "object_urdf_path", None)
    dt = getattr(env, "dt", None)
    detail = None
    meta_fn = getattr(getattr(env, "hero_joint_action_term", None), "action_contract_metadata", None)
    if callable(meta_fn):
        try:
            detail = dict(meta_fn())
        except Exception as exc:  # noqa: BLE001 - metadata must never break an export
            logger.warning(f"hero sidecar: action_contract_metadata skipped ({type(exc).__name__}: {exc})")
    clip = None
    obs_cfg = getattr(getattr(env, "observation_manager", None), "cfg", None)
    if obs_cfg is not None and hasattr(obs_cfg, "clip_observations"):
        try:
            clip = float(obs_cfg.clip_observations)
        except (TypeError, ValueError):
            clip = None
    return build_hero_sidecar(
        metadata,
        onnx_file_path,
        preset=str(preset) if preset else None,
        effort_limit=list(effort) if effort is not None else None,
        dof_pos_limits=limits,
        object_urdf_path=str(object_urdf) if object_urdf else None,
        policy_dt=float(dt) if isinstance(dt, (int, float)) else None,
        action_contract_detail=detail,
        clip_observations=clip,
        hero_command=hero_command_block(exp),
        wandb_run_path=getattr(algo, "_wandb_run_path", None),
        h20_future_steps=metadata.get(HERO_H20_FUTURE_STEPS_KEY),
        hero_expand=hero_expand,
        h21_future_steps=metadata.get(HERO_H21_FUTURE_STEPS_KEY),
        h22_future_steps=metadata.get(HERO_H22_FUTURE_STEPS_KEY),
        anchor_terms=metadata.get(HERO_ANCHOR_TERMS_KEY),
        extra=extra,
    )


def input_perm_for_layout(layout: ActorObsLayout, history_layout: str) -> torch.Tensor | None:
    """Permutation the graph must apply to its inputs for the requested export layout."""
    if history_layout not in SUPPORTED_HISTORY_LAYOUTS:
        raise ValueError(f"unknown history_layout {history_layout!r}; expected one of {SUPPORTED_HISTORY_LAYOUTS}")
    if history_layout == HISTORY_LAYOUT_TERM_MAJOR:
        return None
    return layout.frame_major_to_term_major_perm()


def export_dual_actor_as_onnx(
    wrapper: nn.Module,
    onnx_file_path: str,
    example_obs: torch.Tensor,
    input_names: Sequence[str] = ONNX_INPUT_NAMES,
    output_name: str = ONNX_OUTPUT_NAME,
) -> None:
    """Trace ``wrapper(example_obs, example_obs)`` to ONNX (opset 13, legacy exporter like holosoma)."""
    os.makedirs(Path(onnx_file_path).parent, exist_ok=True)
    if example_obs.dim() != 2:
        raise ValueError(f"example_obs must be [1, obs_dim], got {tuple(example_obs.shape)}")
    if len(input_names) not in (1, 2):
        raise ValueError(f"expected 1 or 2 input names, got {list(input_names)}")

    for logger_name in ("onnxscript", "onnx_ir", "torch.onnx"):
        logging.getLogger(logger_name).setLevel(logging.WARNING)

    was_training = wrapper.training
    wrapper.eval()
    try:
        with torch.no_grad():
            torch.onnx.export(
                wrapper,
                tuple(example_obs.clone() for _ in input_names),
                onnx_file_path,
                verbose=False,
                input_names=list(input_names),
                output_names=[output_name],
                opset_version=ONNX_OPSET,
                dynamo=False,
            )
    finally:
        if was_training:
            wrapper.train()


def dual_export_metadata(
    layout: ActorObsLayout,
    history_layout: str,
    action_split: Sequence[int] | None,
    base_actor_obs_layout: dict[str, Any] | None = None,
    action_contract: str = ACTION_CONTRACT_HERO_RESIDUAL_UPPER,
    *,
    input_names: Sequence[str] = ONNX_INPUT_NAMES,
    body_keys: Sequence[str] = GROUP_KEYS,
    algo: str = "PPODual",
    inputs_note: str = "Both actors receive the same actor_obs vector.",
) -> dict[str, Any]:
    """Metadata entries specific to the dual frame-major export (merged over ``PPO._onnx_deployment_metadata``).

    ``actor_obs_layout`` keeps holosoma's ``holosoma_actor_obs_layout_v1`` structure (validated by
    ``validate_onnx_deployment_metadata``) and gains per-term dims + the history layout."""
    if history_layout not in SUPPORTED_HISTORY_LAYOUTS:
        raise ValueError(f"unknown history_layout {history_layout!r}; expected one of {SUPPORTED_HISTORY_LAYOUTS}")
    layout_meta = layout.to_metadata(history_layout)

    actor_obs_layout: dict[str, Any] = dict(base_actor_obs_layout or {})
    actor_obs_layout.setdefault("schema", "holosoma_actor_obs_layout_v1")
    actor_obs_layout.setdefault("term_concat_order", "sorted_by_term_name_within_group")
    groups = [dict(g) for g in actor_obs_layout.get("groups", [])]
    found = False
    for g in groups:
        if g.get("name") == layout.group_name:
            g["terms"] = list(layout.terms)
            g["term_dims"] = list(layout_meta["term_dims"])
            g["history_length"] = layout.history_length
            g["dim"] = layout.total_dim
            g["frame_dim"] = layout.frame_dim
            found = True
    if not found:
        groups.append(
            {
                "name": layout.group_name,
                "terms": list(layout.terms),
                "term_dims": list(layout_meta["term_dims"]),
                "history_length": layout.history_length,
                "dim": layout.total_dim,
                "frame_dim": layout.frame_dim,
            }
        )
    actor_obs_layout["groups"] = groups
    actor_obs_layout["history_layout"] = history_layout
    actor_obs_layout["frame_order"] = "oldest_first"

    meta: dict[str, Any] = {
        "actor_obs_layout": actor_obs_layout,
        "history_layout": history_layout,
        "history_layout_note": (
            "frame_major_hero_v1: each ONNX input is [frame t-H+1 | ... | frame t], every frame = sorted terms; "
            "the graph permutes to holosoma term-major internally. Zero-fill the first H-1 frames at policy start."
            if history_layout == HISTORY_LAYOUT_FRAME_MAJOR
            else "term_major_holosoma_v1: each ONNX input is [term_0 x H | term_1 x H | ...] (holosoma native)."
        ),
        "onnx_inputs": list(input_names),
        "onnx_output": ONNX_OUTPUT_NAME,
        "onnx_inputs_note": inputs_note,
        "body_keys": list(body_keys),
        "action_split": [int(n) for n in action_split] if action_split is not None else None,
        "action_contract": action_contract,
        "algo": algo,
    }
    if history_layout == HISTORY_LAYOUT_FRAME_MAJOR:
        meta["frame_major_to_term_major_perm"] = layout_meta["frame_major_to_term_major_perm"]
    return meta


__all__ = [
    "ACTION_CONTRACT_HERO_RESIDUAL_UPPER",
    "ONNX_INPUT_NAMES_SINGLE",
    "WHOLE_BODY",
    "SingleActorOnnxWrapper",
    "hero_sidecar_from_algo",
    "HERO_ANCHOR2_TERM_NAMES",
    "HERO_ANCHOR_TERMS_KEY",
    "HERO_ANCHOR_TERM_NAME",
    "HERO_COMMAND_SIDECAR_KEYS",
    "HERO_FUTURE_STEPS_KEYS",
    "HERO_H20_FUTURE_STEPS_KEY",
    "HERO_H21_FUTURE_STEPS_KEY",
    "HERO_H22_FUTURE_STEPS_KEY",
    "HERO_PLUS_PER_FRAME_DIM",
    "HERO_PLUS_TERM_NAMES",
    "HERO_H2_MARKER_TERM",
    "HERO_HISTORY_PADDING",
    "HERO_OBJECT_TERM_NAMES",
    "HERO_SIDECAR_ANCHOR2_KEYS",
    "HERO_SIDECAR_OPTIONAL_KEYS",
    "HERO_SIDECAR_REQUIRED_KEYS",
    "HERO_SIDECAR_SCHEMA",
    "HERO_SIDECAR_SUFFIX",
    "ONNX_INPUT_NAMES",
    "ONNX_OPSET",
    "ONNX_OUTPUT_NAME",
    "DualActorOnnxWrapper",
    "build_hero_sidecar",
    "default_future_steps",
    "dual_export_metadata",
    "export_dual_actor_as_onnx",
    "hero_anchor_terms_block",
    "hero_command_block",
    "hero_h20_future_steps",
    "hero_sidecar_path",
    "hero_term_future_steps",
    "hero_term_noise_params",
    "input_perm_for_layout",
]
