"""HERO episode termination and reference-geometry helpers."""

from __future__ import annotations

import math
import re
from typing import Any

import torch
from loguru import logger

from holosoma.managers.observation.terms.wbt import get_projected_gravity, gravity_vector
from holosoma.managers.termination.base import TerminationTermBase

from hero_isaacsim.utils.ee_residual import (
    quat_angle_xyzw,
    quat_conjugate_xyzw,
    quat_mul_xyzw,
    quat_rotate_inverse_xyzw,
    quat_rotate_xyzw,
)

#: HERO terminates at |g_x| > 0.8 or |g_y| > 0.8 on the pelvis projected gravity g = Rᵀ(0,0,−1). For a pelvis tilted
#: by θ about a horizontal axis, |g_xy| = sin θ, so a roll-only or pitch-only fall trips HERO at
#: θ = asin(0.8) = 0.9273 rad (53.1°). With an upright reference the relative tilt equals θ, hence 0.93 rad
#: reproduces HERO on upright data (for oblique falls HERO's per-axis box is looser — |g_x| = |g_y| = 0.8 needs
#: |g_xy| = 1.13, unreachable — so the cone is marginally stricter there).
GRAVITY_TILT_RELATIVE_THRESHOLD_RAD: float = 0.93
assert abs(GRAVITY_TILT_RELATIVE_THRESHOLD_RAD - math.asin(0.8)) < 5e-3
#: robot pelvis may sit this far below min(reference pelvis height, h_cmd) before terminating (== HERO's
#: 0.75 − 0.5 nominal-to-lowest-command gap: with h_cmd = 0.5 the floor is HERO's absolute 0.25 m again).
BASE_HEIGHT_RELATIVE_MARGIN: float = 0.25
#: pelvis centre height below which the robot is on the floor whatever the reference (G1 prone pelvis ≈ 0.10–0.15 m).
BASE_HEIGHT_ABSOLUTE_FLOOR: float = 0.08


def _mc(env: Any):
    mc = env.command_manager.get_state("motion_command")
    assert mc is not None, "motion_command not found in command manager"
    return mc


def _mc_or_none(env: Any):
    """The motion command when the env has one, else None -- for OPTIONAL reads (``ResetGrace``'s command records) on envs without a
    ``motion_command`` state (``_mc`` asserts)."""
    cm = getattr(env, "command_manager", None)
    return cm.get_state("motion_command") if cm is not None and hasattr(cm, "get_state") else None


def _env_origin_z(env: Any):
    scene = getattr(env.simulator, "scene", None)
    origins = getattr(scene, "env_origins", None) if scene is not None else None
    return 0.0 if origins is None else origins[:, 2]


def command_reference_origins(mc: Any) -> torch.Tensor | None:
    """Command reference origins."""
    origins = getattr(mc, "reference_origins", None)
    if origins is None or not torch.is_tensor(origins) or origins.ndim != 2 or origins.shape[-1] != 3:
        return None
    return origins


def reference_ground_z(env: Any, mc: Any = None):
    """Reference ground z."""
    mc = _mc(env) if mc is None else mc
    origins = command_reference_origins(mc)
    if origins is not None:
        return origins[:, 2]
    return _env_origin_z(env)


# --------------------------------------------------------------------------------------------------
# Terrain-aware ground height (enabled per term).
# --------------------------------------------------------------------------------------------------

TERRAIN_STATE_NAME = "locomotion_terrain"
_TERRAIN_CACHE_ATTR = "_hero_terrain_ground_cache"
_TERRAIN_WARNED_ATTR = "_hero_terrain_fallback_warned"


def terrain_state(env: Any) -> Any | None:
    """holosoma's stateful terrain term (``TerrainLocomotion``: ``update_heights`` / ``base_heights`` / ``feet_heights``)
    or ``None`` when the env has no usable terrain manager (mocks, terrain-free fakes)."""
    tm = getattr(env, "terrain_manager", None)
    if tm is None or not hasattr(tm, "get_state"):
        return None
    try:
        st = tm.get_state(TERRAIN_STATE_NAME)
    except Exception:  # noqa: BLE001 -- a fake manager without the state
        return None
    if st is None or not hasattr(st, "base_heights") or not hasattr(st, "update_heights"):
        return None
    if not terrain_state_has_mesh(st):
        return None  # a PLANE term (terrain_locomotion_plane): its ray casts have no mesh to hit -> NaN heights; the plane IS the env origin
    return st


def terrain_state_has_mesh(st: Any) -> bool:
    """Terrain state has mesh."""
    if hasattr(st, "query_reference_ground"):
        return True
    cfg = getattr(st, "_cfg", None)
    mesh_type = getattr(cfg, "mesh_type", None)
    if mesh_type is not None and "plane" in str(getattr(mesh_type, "value", mesh_type)).lower():
        return False
    if hasattr(st, "warp_mesh") and getattr(st, "warp_mesh", None) is None:
        return False
    return True


def invalidate_terrain_cache(env: Any) -> None:
    """Drop the memoised ground heights (one ray-cast refresh per reward compute / termination check)."""
    cache = getattr(env, _TERRAIN_CACHE_ATTR, None)
    if cache:
        cache.clear()


def _origin_z_tensor(env: Any) -> torch.Tensor:
    origin = _env_origin_z(env)
    if torch.is_tensor(origin):
        return origin
    # robot_root_states is holosoma's RootStatesProxy on Isaac Sim (__getitem__ only: no .shape / .dtype) -- slice a column first
    root_z = env.simulator.robot_root_states[:, 2]
    return torch.full((root_z.shape[0],), float(origin), dtype=root_z.dtype, device=env.device)


def _refresh_terrain_cache(env: Any) -> dict[str, torch.Tensor]:
    cache = getattr(env, _TERRAIN_CACHE_ATTR, None)
    if cache is None:
        cache = {}
        try:
            setattr(env, _TERRAIN_CACHE_ATTR, cache)
        except AttributeError:  # read-only fakes: compute every call
            pass
    if "base" in cache and "feet" in cache:
        return cache
    root_z = env.simulator.robot_root_states[:, 2]
    n = root_z.shape[0]
    st = terrain_state(env)
    if st is None:
        if not getattr(env, _TERRAIN_WARNED_ATTR, False):
            try:
                setattr(env, _TERRAIN_WARNED_ATTR, True)
            except AttributeError:
                pass
            logger.warning(
                "terrain_ground_z: terrain_relative requested but the env has no '{}' terrain state -> env origin z",
                TERRAIN_STATE_NAME,
            )
        base = _origin_z_tensor(env)
        cache["base"] = base
        cache["feet"] = base[:, None].expand(n, 2)
        return cache
    st.update_heights()  # the WBT env does not drive the terrain term; refresh the ray casts here
    base = root_z - st.base_heights.to(root_z.device)
    if not bool(torch.isfinite(base).all()):  # a ray miss (off-mesh env) must never poison the critic / reward: env origin z instead
        base = torch.where(torch.isfinite(base), base, _origin_z_tensor(env).to(base.dtype))
    cache["base"] = base
    feet_h = getattr(st, "feet_heights", None)
    idx = getattr(env, "feet_height_indices", None)
    if (
        feet_h is not None
        and idx is not None
        and torch.is_tensor(feet_h)
        and feet_h.numel() > 0
        and tuple(feet_h.shape) == (n, len(idx))
    ):
        feet = env.simulator._rigid_body_pos[:, idx, 2] - feet_h.to(root_z.device)
        cache["feet"] = torch.where(torch.isfinite(feet), feet, base[:, None].expand_as(feet))
    else:
        cache["feet"] = base[:, None].expand(n, len(idx) if idx is not None else 2)
    return cache


def terrain_ground_z(env: Any, *, feet: bool = False) -> torch.Tensor:
    """Ground z (world) under the robot: ``[N]`` under the base (``root z - base_heights``), or with ``feet=True``
    ``[N, F]`` under each ``feet_height_indices`` body (``foot z - feet_heights``; falls back to the base value when the
    terrain term does not track feet).  Memoised until :func:`invalidate_terrain_cache`."""
    cache = _refresh_terrain_cache(env)
    return cache["feet"] if feet else cache["base"]


def ground_z(env: Any, terrain_relative: bool = False):
    """Height reference of the ROBOT height terms: env origin z (``[N]`` or 0.0) or the terrain under the base."""
    return terrain_ground_z(env) if terrain_relative else _env_origin_z(env)


# --------------------------------------------------------------------------------------------------
# stateless terms
# --------------------------------------------------------------------------------------------------


def gravity_tilt(env: Any, threshold_x: float = 0.8, threshold_y: float = 0.8) -> torch.Tensor:
    """|g_x| > threshold_x or |g_y| > threshold_y on the pelvis projected gravity (HERO termination_gravity 0.8)."""
    g = get_projected_gravity(env)
    return (torch.abs(g[:, 0]) > threshold_x) | (torch.abs(g[:, 1]) > threshold_y)


def base_height_below(env: Any, min_height: float = 0.25, terrain_relative: bool = False) -> torch.Tensor:
    """Pelvis height above its env origin (``terrain_relative``: above the terrain under the base) < min_height (HERO
    termination_min_base_height 0.25)."""
    return robot_root_height(env, terrain_relative) < min_height


def _up_vector_w(quat_xyzw: torch.Tensor) -> torch.Tensor:
    """World-frame z-axis of a body given its xyzw orientation: R·e_z, ``[N, 3]``."""
    e_z = torch.zeros_like(quat_xyzw[:, :3])
    e_z[:, 2] = 1.0
    return quat_rotate_xyzw(quat_xyzw, e_z)


def _angle_between_units(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """Angle between unit vectors via atan2(‖a×b‖, a·b) — well-conditioned near 0 and π (acos is not in float32)."""
    return torch.atan2(torch.linalg.norm(torch.cross(a, b, dim=-1), dim=-1), torch.sum(a * b, dim=-1))


def robot_root_height(env: Any, terrain_relative: bool = False) -> torch.Tensor:
    """Robot pelvis height above the env origin ``[N]`` (``terrain_relative``: above the terrain under the base)."""
    return env.simulator.robot_root_states[:, 2] - ground_z(env, terrain_relative)


def reference_root_height(env: Any) -> torch.Tensor:
    """REFERENCE pelvis height above the REFERENCE ground ``[N]`` ."""
    mc = _mc(env)
    return mc.root_pos_w[:, 2] - reference_ground_z(env, mc)


def reference_tilt_rad(env: Any) -> torch.Tensor:
    """Angle between the REFERENCE pelvis up-vector and the world vertical ``[N]`` (0 = upright, π/2 = prone)."""
    up_ref = _up_vector_w(_mc(env).root_quat_w)
    return torch.atan2(torch.linalg.norm(up_ref[:, :2], dim=-1), up_ref[:, 2])


def relative_tilt_rad(env: Any) -> torch.Tensor:
    """Angle between the robot pelvis up-vector and the REFERENCE pelvis up-vector ``[N]``."""
    up_rob = _up_vector_w(env.simulator.robot_root_states[:, 3:7])
    up_ref = _up_vector_w(_mc(env).root_quat_w)
    return _angle_between_units(up_rob, up_ref)


def upright_reference_mask(env: Any, min_ref_height: float = 0.55, max_ref_tilt_rad: float = 0.6) -> torch.Tensor:
    """bool ``[N]``: reference pelvis height ≥ ``min_ref_height`` AND reference tilt ≤ ``max_ref_tilt_rad``.

    G1 numbers: standing pelvis ≈ 0.75 m, deep squat ≈ 0.35–0.45, upright kneel ≈ 0.45 (tilt ≈ 0 but low →
    NOT upright), half-kneel ≈ 0.5, crawl ≈ 0.3–0.45 with 30–60° pitch, prone ≈ 0.1–0.15 with ≈ 90° pitch."""
    return (reference_root_height(env) >= min_ref_height) & (reference_tilt_rad(env) <= max_ref_tilt_rad)


def gravity_tilt_relative(
    env: Any, threshold_rad: float = GRAVITY_TILT_RELATIVE_THRESHOLD_RAD, upright_ref_only: bool = False
) -> torch.Tensor:
    """Reference-relative replacement of :func:`gravity_tilt`: a prone / crawling reference is no longer a
    "fall" — only tilting away from what the clip does is. Default threshold 0.93 rad = asin(0.8) so that with an
    upright reference the term fires exactly where HERO's |g_x|,|g_y| > 0.8 fires for a roll- or pitch-only fall
    (see :data:`GRAVITY_TILT_RELATIVE_THRESHOLD_RAD`). ``upright_ref_only=True`` additionally restricts the term
    to upright references (:func:`upright_reference_mask` defaults), leaving ground poses to BadTrackingRelative."""
    bad = relative_tilt_rad(env) > threshold_rad
    if upright_ref_only:
        bad = bad & upright_reference_mask(env)
    return bad


def base_height_below_relative(
    env: Any,
    margin: float = BASE_HEIGHT_RELATIVE_MARGIN,
    absolute_floor: float = BASE_HEIGHT_ABSOLUTE_FLOOR,
    include_h_cmd: bool = True,
    terrain_relative: bool = False,
) -> torch.Tensor:
    """Target height = REFERENCE pelvis height above the env origin; with ``include_h_cmd`` (default) it is
    ``min(reference height, h_cmd)`` for clip-relative height commands ``h_cmd = h_ref + U(−0.25, 0)·curriculum``  —
    a policy obeying the lowest command would otherwise sit exactly on the termination boundary. The absolute
    floor catches a collapsed robot when the reference itself is on the ground (crawl / prone).
    ``terrain_relative``: the ROBOT height is taken above the terrain under its base (the reference / h_cmd are
    clip-floor heights already)."""
    h_rob = robot_root_height(env, terrain_relative)
    target = reference_root_height(env)
    if include_h_cmd:
        h_cmd = getattr(_mc(env), "h_cmd", None)
        if h_cmd is not None and torch.is_tensor(h_cmd):
            target = torch.minimum(target, h_cmd.reshape(-1))
    return (h_rob < target - margin) | (h_rob < absolute_floor)


def clip_ends(env: Any) -> torch.Tensor:
    """Prefers the ``clip_ends`` bool tensor ``[N]`` published by ``HeroMotionCommand``
    (``clip_ended`` is accepted as an alias); falls
    back to the index arithmetic on the stock MotionCommand API. Configure with ``is_timeout=True`` so a finished
    clip counts as a timeout, not a failure."""
    mc = _mc(env)
    flag = getattr(mc, "clip_ends", None)
    if flag is None:
        flag = getattr(mc, "clip_ended", None)
    if flag is not None and torch.is_tensor(flag):
        return flag.reshape(-1).to(torch.bool)
    end_idx = mc.motion.motion_end_idx[mc.motion_ids]
    return mc.time_steps >= end_idx - 1


# --------------------------------------------------------------------------------------------------
# BadTrackingRelative
# --------------------------------------------------------------------------------------------------


class BadTrackingRelative(TerminationTermBase):
    """Drift-tolerant termination for poor reference tracking.

    Fires when ``bad_ref_ori | bad_body_pos`` (| ``bad_object`` if enabled), where

    params: ``body_names_to_track`` (must equal the command's list), ``bad_motion_body_pos_body_names``,
    ``bad_ref_ori_threshold`` (0.8), ``bad_motion_body_pos_threshold`` (0.5), ``body_ok_once_gate`` (True),
    ``check_object`` / ``object_termination_enabled`` (False; the config table spells it the second way),
    ``object_term_requires_body_ok`` (False), ``bad_object_pos_threshold`` (0.25), ``bad_object_ori_threshold`` (0.8).
    Any other key raises ``ValueError`` (an unknown key must not masquerade as a setting)."""

    KNOWN_PARAMS: frozenset[str] = frozenset(
        {
            "body_names_to_track",
            "bad_motion_body_pos_body_names",
            "bad_ref_ori_threshold",
            "bad_motion_body_pos_threshold",
            "body_ok_once_gate",
            "check_object",
            "object_termination_enabled",
            "object_term_requires_body_ok",
            "bad_object_pos_threshold",
            "bad_object_ori_threshold",
        }
    )

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        p = cfg.params
        unknown = sorted(set(p) - self.KNOWN_PARAMS)
        if unknown:
            raise ValueError(f"BadTrackingRelative: unknown params {unknown}; known: {sorted(self.KNOWN_PARAMS)}")
        if "check_object" in p and "object_termination_enabled" in p and bool(p["check_object"]) != bool(p["object_termination_enabled"]):
            raise ValueError(
                "BadTrackingRelative: check_object and object_termination_enabled are the same switch but disagree: "
                f"{p['check_object']!r} vs {p['object_termination_enabled']!r}"
            )
        self.bad_ref_ori_threshold = float(p.get("bad_ref_ori_threshold", 0.8))
        self.bad_motion_body_pos_threshold = float(p.get("bad_motion_body_pos_threshold", 0.5))
        self.body_names_to_track = list(p["body_names_to_track"])
        self.bad_motion_body_pos_body_names = list(p.get("bad_motion_body_pos_body_names", []))
        for name in self.bad_motion_body_pos_body_names:
            assert name in self.body_names_to_track, f"{name} not in body_names_to_track {self.body_names_to_track}"
        self.bad_motion_body_pos_body_indexes = torch.tensor(
            [self.body_names_to_track.index(n) for n in self.bad_motion_body_pos_body_names],
            dtype=torch.long,
            device=env.device,
        )
        self.body_ok_once_gate = bool(p.get("body_ok_once_gate", True))


        self.check_object = bool(p.get("check_object", p.get("object_termination_enabled", False)))
        self.object_term_requires_body_ok = bool(p.get("object_term_requires_body_ok", False))
        self.bad_object_pos_threshold = float(p.get("bad_object_pos_threshold", 0.25))
        self.bad_object_ori_threshold = float(p.get("bad_object_ori_threshold", 0.8))
        self._ever_ok: torch.Tensor | None = None
        # last per-component masks (for logging / tests)
        self.last_bad_ref_ori: torch.Tensor | None = None
        self.last_bad_body_pos: torch.Tensor | None = None
        self.last_bad_object: torch.Tensor | None = None

    # -- components --------------------------------------------------------------------------------
    def bad_ref_ori(self, mc: Any) -> torch.Tensor:
        g = gravity_vector(self.env)
        g_ref = quat_rotate_inverse_xyzw(mc.ref_quat_w, g)
        g_rob = quat_rotate_inverse_xyzw(mc.robot_ref_quat_w, g)
        return torch.abs(g_ref[:, 2] - g_rob[:, 2]) > self.bad_ref_ori_threshold

    def bad_body_pos(self, mc: Any) -> torch.Tensor:
        if self.bad_motion_body_pos_body_indexes.numel() == 0:
            return torch.zeros(self.env.num_envs, dtype=torch.bool, device=self.env.device)
        idx = self.bad_motion_body_pos_body_indexes
        err = torch.norm(mc.body_pos_relative_w[:, idx] - mc.robot_body_pos_w[:, idx], dim=-1)
        return torch.any(err > self.bad_motion_body_pos_threshold, dim=-1)

    def bad_object(self, mc: Any) -> torch.Tensor:
        n = self.env.num_envs
        if not self.check_object or not bool(getattr(getattr(mc, "motion", None), "has_object", False)):
            return torch.zeros(n, dtype=torch.bool, device=self.env.device)
        ref_pos = getattr(mc, "object_pos_relative_w", None)
        ref_quat = getattr(mc, "object_quat_relative_w", None)
        if ref_pos is None or ref_quat is None:
            ref_pos, ref_quat = mc.object_pos_w, mc.object_quat_w
        bad_pos = torch.norm(ref_pos - mc.simulator_object_pos_w, dim=-1) > self.bad_object_pos_threshold
        ang = quat_angle_xyzw(quat_mul_xyzw(quat_conjugate_xyzw(mc.simulator_object_quat_w), ref_quat))
        bad_ori = ang > self.bad_object_ori_threshold
        mask = getattr(mc, "env_has_object", None)
        mask = torch.ones(n, dtype=torch.bool, device=self.env.device) if mask is None else mask.reshape(-1).bool()
        return (bad_pos | bad_ori) & mask

    # -- protocol ----------------------------------------------------------------------------------
    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        mc = _mc(env)
        tracked = list(getattr(mc.motion_cfg, "body_names_to_track", self.body_names_to_track))
        assert tracked == self.body_names_to_track, (
            "body_names_to_track in motion_command and termination.params differ: "
            f"{tracked} vs {self.body_names_to_track}"
        )
        bad_ori = self.bad_ref_ori(mc)
        bad_body = self.bad_body_pos(mc)
        bad_obj = self.bad_object(mc)
        if self.object_term_requires_body_ok:
            bad_obj = bad_obj & ~bad_body
        self.last_bad_ref_ori, self.last_bad_body_pos, self.last_bad_object = bad_ori, bad_body, bad_obj
        bad = bad_ori | bad_body | bad_obj
        if not self.body_ok_once_gate:
            return bad
        if self._ever_ok is None or self._ever_ok.shape[0] != bad.shape[0]:
            self._ever_ok = torch.zeros_like(bad)
        gate = self._ever_ok.clone()  # latched BEFORE this step's ok-ness so frame-0 badness never fires
        self._ever_ok |= ~bad
        return bad & gate

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self._ever_ok is None:
            return
        if env_ids is None:
            self._ever_ok.zero_()
        else:
            self._ever_ok[env_ids] = False


# --------------------------------------------------------------------------------------------------
# Reset grace (disabled unless registered).
# --------------------------------------------------------------------------------------------------

#: Termination terms that keep firing during the reset grace: the fall family (gravity tilt, base height) and prohibited
#: contacts.  Matched against the term NAME (``re.search``); ``is_timeout`` terms are never masked either.
DEFAULT_GRACE_EXEMPT_REGEX: str = r"(gravity|base_height|fall|contact)"
#: 0.5 s at the 50 Hz control rate.
DEFAULT_GRACE_FRAMES: int = 25
#: Command attributes recording reset classification / lift at each env's last reset:
#: ``reset_nonstanding_start`` Bool[N], ``reset_lift_m`` Float[N] with 0 = no lift; a float tensor counts where ``> 0``, a bool tensor
#: where True.  ``reset_lift_mask`` is accepted as an alias.  Every available attribute is OR-ed into the grace mask.
RESET_LIFT_RECORD_ATTRS: tuple[str, ...] = ("reset_nonstanding_start", "reset_lift_m", "reset_lift_mask")
RESET_LIFT_MASK_ATTR: str = RESET_LIFT_RECORD_ATTRS[-1]
#: Optional per-env grace lengths, Long[N]. Reset environments in the grace mask
#: use max(grace_frames, record[env]); otherwise they use the configured scalar.
RESET_GRACE_FRAMES_ATTR: str = "reset_grace_frames"
#: ``env`` attribute :func:`check_with_causes` publishes the active grace mask on (``termination/reset_grace_active_frac``).
GRACE_ACTIVE_ATTR: str = "_reset_grace_active"


class ResetGrace(TerminationTermBase):
    """Mask selected termination terms for a fixed number of steps after reset.

    ``grace_left`` counts down once per :func:`check_with_causes` call. Timeout
    terms and terms matched by ``exempt_regex`` or ``exempt_terms`` stay active.
    This term never terminates an episode itself.

    With ``low_start_only=True``, grace applies to reference poses below
    ``min_ref_height`` or beyond ``max_ref_tilt_rad``. With
    ``use_command_lift_record=True``, command reset flags can also enable grace,
    and per-env ``reset_grace_frames`` can extend its duration. Clip rollovers
    do not restart the counter.

    Parameters: ``grace_frames`` (25), ``low_start_only`` (True),
    ``min_ref_height`` (0.55 m), ``max_ref_tilt_rad`` (0.6 rad),
    ``exempt_regex`` (:data:`DEFAULT_GRACE_EXEMPT_REGEX`; empty disables it),
    ``exempt_terms`` (None), ``use_command_lift_record`` (True).
    Unknown parameters raise ``ValueError``.
    """

    KNOWN_PARAMS: frozenset[str] = frozenset(
        {"grace_frames", "low_start_only", "min_ref_height", "max_ref_tilt_rad", "exempt_regex", "exempt_terms", "use_command_lift_record"}
    )

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        p = dict(getattr(cfg, "params", None) or {})
        unknown = sorted(set(p) - self.KNOWN_PARAMS)
        if unknown:
            raise ValueError(f"ResetGrace: unknown params {unknown}; known: {sorted(self.KNOWN_PARAMS)}")
        self.grace_frames = int(p.get("grace_frames", DEFAULT_GRACE_FRAMES))
        if self.grace_frames < 0:
            raise ValueError(f"ResetGrace: grace_frames must be >= 0, got {self.grace_frames}")
        self.low_start_only = bool(p.get("low_start_only", True))
        self.min_ref_height = float(p.get("min_ref_height", 0.55))
        self.max_ref_tilt_rad = float(p.get("max_ref_tilt_rad", 0.6))
        regex = p.get("exempt_regex", DEFAULT_GRACE_EXEMPT_REGEX)
        self._exempt_re = re.compile(str(regex)) if regex else None
        terms = p.get("exempt_terms", None)
        self.exempt_terms: frozenset[str] = frozenset(str(t) for t in terms) if terms else frozenset()
        self.use_command_lift_record = bool(p.get("use_command_lift_record", True))
        n = int(env.num_envs)
        self.grace_left = torch.zeros(n, dtype=torch.long, device=env.device)
        self.last_active = torch.zeros(n, dtype=torch.bool, device=env.device)

    # -- who is exempt ------------------------------------------------------------------------------
    def is_exempt(self, name: str, cfg: Any) -> bool:
        """True when term ``name`` keeps firing during the grace: timeouts, the explicit list, the regex family."""
        if bool(getattr(cfg, "is_timeout", False)) or name in self.exempt_terms:
            return True
        return bool(self._exempt_re is not None and self._exempt_re.search(name))

    # -- the reset-start classification -----------------------------------------------------------------
    def _grace_mask(self, env_ids: torch.Tensor | None) -> torch.Tensor:
        """Bool over ``env_ids`` (all envs when None): which of the reset envs get the grace."""
        n = int(self.env.num_envs)
        ids = torch.arange(n, device=self.env.device) if env_ids is None else torch.as_tensor(env_ids, dtype=torch.long, device=self.env.device).reshape(-1)
        if not self.low_start_only:
            return torch.ones(ids.numel(), dtype=torch.bool, device=self.env.device)
        low = ~upright_reference_mask(self.env, self.min_ref_height, self.max_ref_tilt_rad)
        mask = low.to(self.env.device)[ids]
        if self.use_command_lift_record:
            mc = _mc_or_none(self.env)  # optional record: an env without a motion_command keeps the reference-only mask
            for attr in RESET_LIFT_RECORD_ATTRS:
                rec = getattr(mc, attr, None) if mc is not None else None
                if rec is None or not torch.is_tensor(rec) or rec.numel() != n:
                    continue
                flagged = rec.reshape(-1).to(mask.device)
                flagged = flagged if flagged.dtype == torch.bool else flagged > 0
                mask = mask | flagged[ids]
        return mask

    # -- protocol -----------------------------------------------------------------------------------
    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        if self.grace_frames == 0:
            return
        mask = self._grace_mask(env_ids)
        value = torch.where(mask, torch.full_like(mask, self.grace_frames, dtype=torch.long), torch.zeros_like(mask, dtype=torch.long))
        if self.use_command_lift_record:  # Optional per-env grace length from the command reset.
            mc = _mc_or_none(self.env)  # Environments without a motion command keep the scalar grace.
            rec = getattr(mc, RESET_GRACE_FRAMES_ATTR, None) if mc is not None else None
            n = int(self.env.num_envs)
            if rec is not None and torch.is_tensor(rec) and rec.numel() == n:
                ids = torch.arange(n, device=self.env.device) if env_ids is None else torch.as_tensor(env_ids, dtype=torch.long, device=self.env.device).reshape(-1)
                per_env = rec.reshape(-1).to(device=value.device, dtype=torch.long)[ids.to(value.device)]
                value = torch.where(mask, torch.maximum(value, per_env), value)
        if env_ids is None:
            self.grace_left.copy_(value)
        else:
            self.grace_left[torch.as_tensor(env_ids, dtype=torch.long, device=self.grace_left.device).reshape(-1)] = value

    def active_mask(self) -> torch.Tensor:
        """Bool[N]: envs in grace at the CURRENT check (before this step's decrement)."""
        return self.grace_left > 0

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        self.last_active = self.active_mask()
        self.grace_left.sub_(1).clamp_(min=0)
        return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)


def grace_providers(termination_manager: Any) -> list[ResetGrace]:
    """The :class:`ResetGrace` instances registered on a holosoma ``TerminationManager`` (usually 0 or 1)."""
    return [t for t in getattr(termination_manager, "_term_instances", {}).values() if isinstance(t, ResetGrace)]


# --------------------------------------------------------------------------------------------------
# per-cause evaluation helper (used by HeroTrackingManager._check_termination)
# --------------------------------------------------------------------------------------------------


def split_done_flags(reset_flags: torch.Tensor, timeout_flags: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    """Split done flags."""
    failed = reset_flags.bool()
    timeout = timeout_flags.bool()
    return failed, timeout & ~failed, timeout & failed


def check_with_causes(termination_manager: Any, env: Any) -> tuple[torch.Tensor, torch.Tensor, dict[str, torch.Tensor]]:
    """Evaluate every term of a holosoma ``TerminationManager`` exactly once.

    Returns ``(reset_flags, timeout_flags, causes)`` with ``causes[name]`` the raw bool mask of each term, and
    updates ``termination_manager.terminated / time_outs`` like ``TerminationManager.check`` does.

    With a :class:`ResetGrace` term registered, the non-exempt, non-timeout masks are ANDed with ``~grace`` (the grace read at
    this check, before its decrement) BEFORE the flags are accumulated -- ``causes`` holds the masked masks, so the per-cause
    rates count what actually ended episodes; the active grace mask is published on ``env`` (:data:`GRACE_ACTIVE_ATTR`) for
    ``termination/reset_grace_active_frac``.  Without a grace term, all termination masks apply directly."""
    n = env.num_envs
    reset_flags = torch.zeros(n, dtype=torch.bool, device=env.device)
    timeout_flags = torch.zeros_like(reset_flags)
    causes: dict[str, torch.Tensor] = {}
    for name, cfg in zip(termination_manager._term_names, termination_manager._term_cfgs):
        if name in termination_manager._term_instances:
            result = termination_manager._term_instances[name](env, **cfg.params)
        else:
            result = termination_manager._term_funcs[name](env, **cfg.params)
        if result.dtype != torch.bool:
            raise TypeError(f"Termination term '{name}' returned dtype {result.dtype}, expected torch.bool tensor.")
        causes[name] = result
    providers = grace_providers(termination_manager)
    if providers:
        active = torch.zeros(n, dtype=torch.bool, device=env.device)
        for prov in providers:
            active |= prov.last_active.to(env.device)
        for name, cfg in zip(termination_manager._term_names, termination_manager._term_cfgs):
            if any(prov.is_exempt(name, cfg) for prov in providers):
                continue
            causes[name] = causes[name] & ~active
        try:
            setattr(env, GRACE_ACTIVE_ATTR, active)
        except AttributeError:  # read-only fakes
            pass
    for name, cfg in zip(termination_manager._term_names, termination_manager._term_cfgs):
        if cfg.is_timeout:
            timeout_flags |= causes[name]
        else:
            reset_flags |= causes[name]
    termination_manager.terminated = reset_flags.clone()
    termination_manager.time_outs = timeout_flags.clone()
    return reset_flags, timeout_flags, causes


def hero_termination_term_cfgs(
    body_names_to_track: list[str],
    bad_motion_body_pos_body_names: list[str] | None = None,
    *,
    gravity_threshold: float = 0.8,
    min_base_height: float = 0.25,
    bad_ref_ori_threshold: float = 0.8,
    bad_motion_body_pos_threshold: float = 0.5,
    include_timeout: bool = True,
    ground_safe: bool = False,
    gravity_threshold_rad: float = GRAVITY_TILT_RELATIVE_THRESHOLD_RAD,
    base_height_margin: float = BASE_HEIGHT_RELATIVE_MARGIN,
    base_height_floor: float = BASE_HEIGHT_ABSOLUTE_FLOOR,
    terrain_relative: bool = False,
):
    """``{name: TerminationTermCfg}`` for HERO (timeout, gravity, height, clip_ends, BadTrackingRelative)."""
    from holosoma.config_types.termination import TerminationTermCfg  # noqa: PLC0415

    mod = "hero_isaacsim.managers.termination.hero"
    terms = {}
    if include_timeout:
        terms["timeout"] = TerminationTermCfg(
            func="holosoma.managers.termination.terms.common:timeout_exceeded", is_timeout=True
        )
    if ground_safe:
        terms["gravity_tilt_relative"] = TerminationTermCfg(
            func=f"{mod}:gravity_tilt_relative", params={"threshold_rad": gravity_threshold_rad, "upright_ref_only": False}
        )
        height_params: dict[str, Any] = {
            "margin": base_height_margin, "absolute_floor": base_height_floor, "include_h_cmd": True
        }
        if terrain_relative:
            height_params["terrain_relative"] = True
        terms["base_height_below_relative"] = TerminationTermCfg(
            func=f"{mod}:base_height_below_relative", params=height_params
        )
    else:
        terms["gravity_tilt"] = TerminationTermCfg(
            func=f"{mod}:gravity_tilt", params={"threshold_x": gravity_threshold, "threshold_y": gravity_threshold}
        )
        height_params = {"min_height": min_base_height}
        if terrain_relative:
            height_params["terrain_relative"] = True
        terms["base_height_below"] = TerminationTermCfg(func=f"{mod}:base_height_below", params=height_params)
    terms["clip_ends"] = TerminationTermCfg(func=f"{mod}:clip_ends", is_timeout=True)
    terms["bad_tracking_relative"] = TerminationTermCfg(
        func=f"{mod}:BadTrackingRelative",
        params={
            "body_names_to_track": list(body_names_to_track),
            "bad_motion_body_pos_body_names": list(
                bad_motion_body_pos_body_names
                if bad_motion_body_pos_body_names is not None
                else [n for n in ("left_ankle_roll_link", "right_ankle_roll_link") if n in body_names_to_track]
            ),
            "bad_ref_ori_threshold": bad_ref_ori_threshold,
            "bad_motion_body_pos_threshold": bad_motion_body_pos_threshold,
            "body_ok_once_gate": True,
        },
    )
    return terms


__all__ = [
    "BASE_HEIGHT_ABSOLUTE_FLOOR",
    "BASE_HEIGHT_RELATIVE_MARGIN",
    "BadTrackingRelative",
    "DEFAULT_GRACE_EXEMPT_REGEX",
    "DEFAULT_GRACE_FRAMES",
    "GRACE_ACTIVE_ATTR",
    "GRAVITY_TILT_RELATIVE_THRESHOLD_RAD",
    "RESET_GRACE_FRAMES_ATTR",
    "RESET_LIFT_MASK_ATTR",
    "RESET_LIFT_RECORD_ATTRS",
    "ResetGrace",
    "TERRAIN_STATE_NAME",
    "base_height_below",
    "base_height_below_relative",
    "check_with_causes",
    "clip_ends",
    "command_reference_origins",
    "grace_providers",
    "gravity_tilt",
    "gravity_tilt_relative",
    "ground_z",
    "hero_termination_term_cfgs",
    "invalidate_terrain_cache",
    "reference_ground_z",
    "reference_root_height",
    "reference_tilt_rad",
    "relative_tilt_rad",
    "robot_root_height",
    "terrain_ground_z",
    "terrain_state",
    "upright_reference_mask",
]
