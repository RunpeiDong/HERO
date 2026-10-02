"""Advance motion references temporarily for timeout bootstrap observations.

The peek follows the normal reference and command transition for timed-out
environments, holds at clip end, and restores command state and RNG state on
exit. It does not change simulator state, observation histories, or sampling
statistics. Per-step caches are refreshed for the advanced rows and restored
without cloning immutable corpus tables.

Unsupported commands and failed advances retain the current reference and
log a warning once."""

from __future__ import annotations

import contextlib
import traceback
from typing import Any, Iterator

import torch
from loguru import logger

from holosoma.utils.rotations import quat_apply, quat_inverse, quat_mul, yaw_quat


from hero_isaacsim.managers.observation.hero import H20_FRAME_CACHE_ATTR, _clip_last_frame_index, _h20_cache_key

MOTION_COMMAND_TERM_NAME = "motion_command"
#: holosoma ``MotionCommand._motion_frames`` per-step cache: ``{"_key": (id(time_steps), time_steps._version), <property>: [N, ...]}``.
STOCK_FRAME_CACHE_ATTR = "_motion_frame_cache"
#: h20 cache layout (``observation/hero.py hero_ref_root_future_pose_w``): ``{"key": _h20_cache_key(mc, steps), "pos": [N, F, 3],
#: "quat": [N, F, 4]}``; ``key[H20_KEY_STEPS_INDEX]`` is the ``steps`` tuple (verified by recomputing the key before it is used).
H20_KEY_STEPS_INDEX = 3

#: Tensor attributes the advance needs on the command (``time_steps`` written in place, the relative buffers rebound).
REQUIRED_TENSOR_ATTRS: tuple[str, ...] = ("time_steps", "motion_ids", "body_pos_relative_w", "body_quat_relative_w")
#: Reference / robot properties the relative-target formula reads (class properties on the stock command, plain attributes on
#: the CPU contract double).
REQUIRED_REFERENCE_ATTRS: tuple[str, ...] = (
    "ref_pos_w",
    "ref_quat_w",
    "robot_ref_pos_w",
    "robot_ref_quat_w",
    "root_pos_w",
    "root_quat_w",
    "robot_root_pos_w",
    "robot_root_quat_w",
    "body_pos_w",
    "body_quat_w",
)
#: Delta-anchor per-step command transition (``HeroMotionCommand.advance_command_transition(env_ids)``: the periodic resample when due,
#: run by the peek for the timed-out envs under an isolated RNG state); absent on the stock command.
HERO_COMMAND_TRANSITION_HOOK: str = "advance_command_transition"
#: Delta-anchor hooks ``HeroMotionCommand.step`` runs after the command transition (deterministic given frame + command; called when present).
HERO_REFRESH_HOOKS: tuple[str, ...] = ("_refresh_hero_refs", "_update_clip_end_flags")
#: Env attributes a playback-augmentation command rebinds inside ``_playback_advance_delta`` (``HaltAugMotionCommand``).
ENV_ATTRS_REBOUND_BY_STEP: tuple[str, ...] = ("halt_frozen",)
#: The MUTABLE per-env tensor attributes the peek's advance writes in place or rebinds, per command class (unioned over the MRO of
#: the command).  Everything else on the command -- corpus tables, index maps, curriculum / sampler / metric state -- is neither
#: touched by the advance nor snapshotted.  An UNREGISTERED class in the MRO (a local subclass) adds every per-env tensor
#: (``shape[0] == num_envs``) as a safety net; corpus-sized tensors are never cloned in either case.
MUTABLE_STATE_BY_CLASS: dict[str, tuple[str, ...]] = {
    # holosoma wbt.py MotionCommand.step(): frame index written in place; relative targets rebound.
    "MotionCommand": ("time_steps", "body_pos_relative_w", "body_quat_relative_w"),
    # holosoma wbt_playback_aug.py: time-warp accumulator (in place); halt counter (in place) + accumulator rebound (+ env.halt_frozen).
    "TimeWarpMotionCommand": ("_tw_accum",),
    "HaltAugMotionCommand": ("_halt_step", "_tw_accum"),

    "SonicMotionCommand": (),
    "ContractSonicMotionCommand": (),
    # hero_isaacsim hero.py HeroMotionCommand.step(): the periodic resample (_resample_hero_commands) writes the command state in
    # place; _refresh_hero_refs rebinds the references (stand_flag written in place in from_clip mode); _update_clip_end_flags
    # rebinds the two flags.
    "HeroMotionCommand": (
        "stand_flag",
        "h_offset",
        "_vel_cmd_lin",
        "_vel_cmd_heading",
        "fix_upper_body_mask",
        "fix_upper_body_time_steps",
        "ref_time_steps",
        "ref_time_steps_upper",
        "ref_upper_dof_pos",
        "ref_ee_pos_pelvis",
        "ref_ee_quat_pelvis",
        "ref_ee_pos_pelvis_zero_waist",
        "ref_ee_quat_pelvis_zero_waist",
        "ref_h",
        "h_cmd",
        "vel_cmd",
        "clip_last_frame",
        "clip_ends",
    ),
    "ContractHeroMotionCommand": (),
}
#: Base classes carrying no per-env state of their own (never "unregistered").
_STATELESS_BASES: frozenset[str] = frozenset({"CommandTermBase", "ABC", "Generic", "object"})

_WARNED: set[str] = set()


def _warn_once(key: str, message: str) -> None:
    if key in _WARNED:
        return
    _WARNED.add(key)
    logger.warning(message)


def reference_peek_problem(mc: Any) -> str | None:
    """``None`` when ``mc`` exposes the stock ``MotionCommand`` surface the advance needs; otherwise the missing piece."""
    if mc is None:
        return "command term is None"
    for name in REQUIRED_TENSOR_ATTRS:
        if not isinstance(getattr(mc, name, None), torch.Tensor):
            return f"{name} is not a tensor attribute"
    motion = getattr(mc, "motion", None)
    for name in ("motion_start_idx", "motion_end_idx"):
        if not isinstance(getattr(motion, name, None), torch.Tensor):
            return f"motion.{name} is missing"
    cfg = getattr(mc, "motion_cfg", None)
    if cfg is None or getattr(cfg, "body_names_to_track", None) is None:
        return "motion_cfg.body_names_to_track is missing"
    env = getattr(mc, "_env", None)
    if not isinstance(getattr(env, "episode_length_buf", None), torch.Tensor):
        return "_env.episode_length_buf is missing"
    for name in REQUIRED_REFERENCE_ATTRS:
        if not (hasattr(type(mc), name) or name in getattr(mc, "__dict__", {})):
            return f"{name} is missing"
    return None


def stateful_command_terms(env: Any) -> list[tuple[str, Any]]:
    """``(name, term)`` of the env's stateful command terms (``CommandManager._state_terms``; ``get_state`` fallback for fakes)."""
    cm = getattr(env, "command_manager", None)
    if cm is None:
        return []
    terms = getattr(cm, "_state_terms", None)
    if isinstance(terms, dict):
        return [(str(name), term) for name, term in terms.items() if term is not None]
    get_state = getattr(cm, "get_state", None)
    if callable(get_state):
        mc = get_state(MOTION_COMMAND_TERM_NAME)
        return [(MOTION_COMMAND_TERM_NAME, mc)] if mc is not None else []
    return []


def _version_or_none(tensor: torch.Tensor) -> int | None:
    try:
        return int(tensor._version)
    except RuntimeError:  # inference tensor: no version counter
        return None


def mutable_state_names(mc: Any) -> tuple[tuple[str, ...], bool]:
    """``(names, explicit)``: the allow-listed mutable attribute names for ``mc``'s class (union over the MRO) and whether every
    class in the MRO is registered (``False`` -> the snapshot adds the per-env heuristic)."""
    names: list[str] = []
    explicit = True
    for klass in type(mc).__mro__:
        name = klass.__name__
        if name in _STATELESS_BASES:
            continue
        entry = MUTABLE_STATE_BY_CLASS.get(name)
        if entry is None:
            explicit = False
            continue
        names.extend(attr for attr in entry if attr not in names)
    return tuple(names), explicit


class _FrameCaches:
    """The per-step frame caches of a command that are WARM at capture time: the stock
    ``_motion_frame_cache`` entries and the h20 root gather, both keyed on ``time_steps``' identity + version counter.

    Capture BEFORE the in-place frame write.  :meth:`patch_rows` (right AFTER ``time_steps[ids] = target``) re-seeds every
    captured cache under the post-write key with a clone of the pre-peek tensor whose ``ids`` rows are re-gathered at ``target``
    -- the cached tensors stay untouched (read-only by contract), the corpus is read for ``len(ids)`` rows only.
    :meth:`reinstall` (AFTER the restore write put the pre-peek frame index back) re-seeds them under the new key with the
    pre-peek tensors themselves (no clone, no gather).  Nothing is captured for an inference-tensor ``time_steps`` (no version
    counter: both caches are disabled by their owners) or for a cache whose key is not the current one (cold / stale)."""

    def __init__(self, mc: Any):
        self.mc = mc
        self.stock: dict[str, torch.Tensor] | None = None
        self.h20: tuple[tuple[int, ...], torch.Tensor, torch.Tensor] | None = None
        ts = getattr(mc, "time_steps", None)
        if not isinstance(ts, torch.Tensor) or ts.dim() < 1:
            return
        version = _version_or_none(ts)
        if version is None:
            return
        n = int(ts.shape[0])
        cache = getattr(mc, STOCK_FRAME_CACHE_ATTR, None)
        if isinstance(cache, dict) and cache.get("_key") == (id(ts), version):
            entries = {
                name: value
                for name, value in cache.items()
                if name != "_key" and isinstance(value, torch.Tensor) and value.dim() >= 1 and value.shape[0] == n
            }
            if entries:
                self.stock = entries
        h20 = getattr(mc, H20_FRAME_CACHE_ATTR, None)
        if isinstance(h20, dict):
            key, pos, quat = h20.get("key"), h20.get("pos"), h20.get("quat")
            try:
                steps = tuple(int(s) for s in key[H20_KEY_STEPS_INDEX])
                current = key == _h20_cache_key(mc, steps)
            except Exception:  # noqa: BLE001 -- an unexpected layout only means "do not patch"
                current = False
            if (
                current
                and isinstance(pos, torch.Tensor)
                and isinstance(quat, torch.Tensor)
                and pos.shape[:2] == (n, len(steps))
                and quat.shape[:2] == (n, len(steps))
            ):
                self.h20 = (steps, pos, quat)

    @property
    def stock_names(self) -> tuple[str, ...]:
        return tuple(self.stock) if self.stock else ()

    @property
    def h20_steps(self) -> tuple[int, ...] | None:
        return self.h20[0] if self.h20 is not None else None

    def patch_rows(self, ids: torch.Tensor, target: torch.Tensor) -> None:
        """After ``time_steps[ids] = target``: re-seed the captured caches at the new key, re-gathering the ``ids`` rows only."""
        mc = self.mc
        ts = mc.time_steps
        version = _version_or_none(ts)
        if version is None:
            return
        if self.stock is not None:
            cache = getattr(mc, STOCK_FRAME_CACHE_ATTR, None)
            if isinstance(cache, dict):
                patched: dict[str, Any] = {"_key": (id(ts), version)}
                for name, frames in self.stock.items():
                    rows = mc.motion.frames(name, target)
                    copy = frames.clone()
                    copy[ids] = rows.to(device=copy.device, dtype=copy.dtype)
                    patched[name] = copy
                cache.clear()
                cache.update(patched)
        if self.h20 is not None:
            cache = getattr(mc, H20_FRAME_CACHE_ATTR, None)
            key = _h20_cache_key(mc, self.h20[0])
            if isinstance(cache, dict) and key is not None:
                steps, pos, quat = self.h20
                k, f = int(ids.numel()), len(steps)
                offs = torch.tensor(list(steps), dtype=ts.dtype, device=ts.device)
                last = _clip_last_frame_index(mc)[ids]
                idx = torch.minimum(target[:, None] + offs[None, :], last[:, None]).reshape(-1)  # [k * F]
                pos_new = pos.clone()
                pos_new[ids] = mc.motion.frames("body_pos_w", idx)[:, 0].reshape(k, f, 3).to(device=pos.device, dtype=pos.dtype)
                quat_new = quat.clone()
                quat_new[ids] = mc.motion.frames("body_quat_w", idx)[:, 0].reshape(k, f, 4).to(device=quat.device, dtype=quat.dtype)
                cache.clear()
                cache.update(key=key, pos=pos_new, quat=quat_new)

    def reinstall(self) -> None:
        """After the restore write: the pre-peek tensors are the current frames again -- re-seed them under the new key."""
        mc = self.mc
        ts = getattr(mc, "time_steps", None)
        if not isinstance(ts, torch.Tensor):
            return
        version = _version_or_none(ts)
        if version is None:
            return
        if self.stock is not None:
            cache = getattr(mc, STOCK_FRAME_CACHE_ATTR, None)
            if isinstance(cache, dict):
                cache.clear()
                cache["_key"] = (id(ts), version)
                cache.update(self.stock)
        if self.h20 is not None:
            cache = getattr(mc, H20_FRAME_CACHE_ATTR, None)
            steps, pos, quat = self.h20
            key = _h20_cache_key(mc, steps)
            if isinstance(cache, dict) and key is not None:
                cache.clear()
                cache.update(key=key, pos=pos, quat=quat)


class _StateSnapshot:
    """Identity + content of the MUTABLE per-env tensor attributes of a command (and the env attributes ``step()`` may rebind),
    plus the warm per-step frame caches (:class:`_FrameCaches`) re-seeded after the restore write."""

    def __init__(self, mc: Any, env: Any):
        self.mc = mc
        self.env = env
        ts = getattr(mc, "time_steps", None)
        self.num_envs = int(ts.shape[0]) if isinstance(ts, torch.Tensor) and ts.dim() >= 1 else None
        names, self.explicit = mutable_state_names(mc)
        attrs = vars(mc)
        selected: list[str] = [name for name in names if isinstance(attrs.get(name), torch.Tensor)]
        if not self.explicit and self.num_envs is not None:  # unregistered subclass: every per-env tensor as well
            for name, value in attrs.items():
                if name not in selected and isinstance(value, torch.Tensor) and value.dim() >= 1 and value.shape[0] == self.num_envs:
                    selected.append(name)
        self.tensors: list[tuple[str, torch.Tensor, int | None, torch.Tensor]] = []
        for name in selected:
            value = attrs[name]
            self.tensors.append((name, value, _version_or_none(value), value.clone()))
        self.env_attrs: list[tuple[str, Any]] = [(name, getattr(env, name)) for name in ENV_ATTRS_REBOUND_BY_STEP if hasattr(env, name)]
        self.frame_caches = _FrameCaches(mc)  # warm per-step frame caches at the pre-peek frame (re-seeded by restore())

    @property
    def cloned_names(self) -> tuple[str, ...]:
        return tuple(name for name, _, _, _ in self.tensors)

    def cloned_bytes(self) -> int:
        return sum(clone.numel() * clone.element_size() for _, _, _, clone in self.tensors)

    def restore(self) -> None:
        for name, tensor, version, clone in self.tensors:
            setattr(self.mc, name, tensor)  # identity: rebound attributes point at the pre-peek object again
            if version is not None and _version_or_none(tensor) == version:
                continue  # normal tensor never written in place (a free counter read, no device sync)
            if tensor.is_inference() and not torch.is_inference_mode_enabled():
                continue  # an in-place write to it would have raised inside the peek too: the content is the pre-peek content
            tensor.copy_(clone)  # written in place (time_steps, Delta-anchor command state / stand_flag, time-warp accumulators) or an
            # inference tensor (no counter): put the content back unconditionally -- no torch.equal device sync
        for name, value in self.env_attrs:
            setattr(self.env, name, value)
        self.frame_caches.reinstall()  # time_steps holds the pre-peek frames again: the pre-peek gathers are current under the new key


def _playback_delta(mc: Any, advance_mask: torch.Tensor) -> torch.Tensor:
    """Per-env frames ``step()`` advances this control step (the command's own hook: stock = 1 frame; playback-speed subclasses
    keep their accumulators consistent -- their side effects are undone by the snapshot)."""
    hook = getattr(mc, "_playback_advance_delta", None)
    if callable(hook):
        return hook(advance_mask)
    return advance_mask.long()


def _refresh_relative_targets(mc: Any, env: Any, ids: torch.Tensor) -> None:
    """``MotionCommand.step()`` 1.0 - 1.2 (holosoma wbt.py) for the rows ``ids`` only; the full buffers are REBOUND, not written."""
    num_bodies = len(mc.motion_cfg.body_names_to_track)
    # 1.0 reference body poses (the IsaacGym post-reset quirk: root instead of the configured reference body at episode step 0)
    use_root = (env.episode_length_buf[ids] == 0).unsqueeze(1).float()
    ref_pos_w = mc.root_pos_w[ids] * use_root + mc.ref_pos_w[ids] * (1 - use_root)
    ref_quat_w = mc.root_quat_w[ids] * use_root + mc.ref_quat_w[ids] * (1 - use_root)
    robot_ref_pos_w = mc.robot_root_pos_w[ids] * use_root + mc.robot_ref_pos_w[ids] * (1 - use_root)
    robot_ref_quat_w = mc.robot_root_quat_w[ids] * use_root + mc.robot_ref_quat_w[ids] * (1 - use_root)
    # 1.1 repeat to match the number of tracked bodies
    ref_pos_w_repeat = ref_pos_w[:, None, :].repeat(1, num_bodies, 1)
    ref_quat_w_repeat = ref_quat_w[:, None, :].repeat(1, num_bodies, 1)
    robot_ref_pos_w_repeat = robot_ref_pos_w[:, None, :].repeat(1, num_bodies, 1)
    robot_ref_quat_w_repeat = robot_ref_quat_w[:, None, :].repeat(1, num_bodies, 1)
    # 1.2 relative body poses
    delta_quat_w = yaw_quat(
        quat_mul(robot_ref_quat_w_repeat, quat_inverse(ref_quat_w_repeat, w_last=True), w_last=True), w_last=True
    )
    body_quat_relative = quat_mul(delta_quat_w, mc.body_quat_w[ids], w_last=True)
    delta_pos_w_height = ref_pos_w_repeat - robot_ref_pos_w_repeat
    delta_pos_w_height[..., :2] = 0.0
    body_pos_relative = (
        robot_ref_pos_w_repeat + delta_pos_w_height + quat_apply(delta_quat_w, mc.body_pos_w[ids] - ref_pos_w_repeat, w_last=True)
    )
    pos_full = mc.body_pos_relative_w.clone()
    pos_full[ids] = body_pos_relative
    quat_full = mc.body_quat_relative_w.clone()
    quat_full[ids] = body_quat_relative
    mc.body_pos_relative_w = pos_full
    mc.body_quat_relative_w = quat_full


def _rng_devices(mc: Any) -> list[int]:
    """CUDA devices whose RNG the command's draws consume (``fork_rng`` saves/restores them along with the CPU generator)."""
    try:
        device = torch.device(getattr(mc, "device", "cpu"))
    except (TypeError, RuntimeError):
        return []
    if device.type != "cuda":
        return []
    return [device.index if device.index is not None else torch.cuda.current_device()]


def advance_command_transition_isolated(mc: Any, ids: torch.Tensor) -> torch.Tensor | None:
    """Advance command transition isolated."""
    hook = getattr(mc, HERO_COMMAND_TRANSITION_HOOK, None)
    if not callable(hook):
        return None
    with torch.random.fork_rng(devices=_rng_devices(mc)):
        return hook(ids)


def advance_reference_of_command(mc: Any, env: Any, ids: torch.Tensor) -> torch.Tensor:
    """Advance the reference of ``ids`` on ``mc`` like ``step()`` would (no restore here; see the context manager).

    Returns the new frame indices ``[len(ids)]``.  ``time_steps`` is written IN PLACE so every cache keyed on its version counter
    (holosoma ``_motion_frames``, the delta-anchor h20 gather) invalidates; the caches that were warm are re-seeded right away with the
    ``ids`` rows re-gathered at the advanced frame and everything else untouched (:class:`_FrameCaches`; a cold cache re-gathers
    on its first read as before).  The clip end is HELD (``end - 1``).
    Order = ``HeroMotionCommand.step()``: stock advance -> command transition (due periodic resample) -> reference refresh."""
    ts = mc.time_steps
    advance_mask = torch.ones(ts.shape[0], dtype=torch.bool, device=ts.device)
    delta = _playback_delta(mc, advance_mask).to(device=ts.device, dtype=ts.dtype)
    clip = mc.motion_ids[ids]
    start = mc.motion.motion_start_idx.to(ts.device)[clip]
    last = mc.motion.motion_end_idx.to(ts.device)[clip] - 1
    target = torch.minimum(torch.maximum(ts[ids] + delta[ids], start), last)
    caches = _FrameCaches(mc)  # captured at the pre-peek key, BEFORE the write invalidates it
    ts[ids] = target
    caches.patch_rows(ids, target)  # BEFORE the first read below: len(ids) corpus rows instead of N per cached property
    _refresh_relative_targets(mc, env, ids)
    advance_command_transition_isolated(mc, ids)
    for name in HERO_REFRESH_HOOKS:
        hook = getattr(mc, name, None)
        if callable(hook):
            hook()
    return target


def _as_long_ids(env: Any, env_ids: Any, device: Any) -> torch.Tensor:
    if isinstance(env_ids, torch.Tensor):
        if env_ids.dtype == torch.bool:
            env_ids = env_ids.nonzero(as_tuple=False).flatten()
        return env_ids.to(device=device, dtype=torch.long)
    return torch.as_tensor(list(env_ids), dtype=torch.long, device=device)


def bootstrap_env_ids(env: Any, env_ids: Any) -> torch.Tensor:
    """The subset of the done ``env_ids`` whose final observation is bootstrapped from: ``time_out_buf`` True (PPO / PPODual read
    ``final_observations`` for the ``time_outs`` rows only; a failure-only env never needs the peek).  Without a ``time_out_buf``
    tensor every done env is returned (conservative)."""
    time_outs = getattr(env, "time_out_buf", None)
    device = time_outs.device if isinstance(time_outs, torch.Tensor) else getattr(env, "device", "cpu")
    ids = _as_long_ids(env, env_ids, device)
    if not isinstance(time_outs, torch.Tensor) or ids.numel() == 0:
        return ids
    return ids[time_outs[ids].bool()]


@contextlib.contextmanager
def advance_reference_for_final_observation(env: Any, env_ids: Any) -> Iterator[tuple[str, ...]]:
    """Context: the reference of ``env_ids`` reads like the next ``step()`` for every supported stateful command term of ``env``;
    yields the names of the advanced terms; restores every mutated command / env state exactly on exit (exceptions included).

    Use around ``BaseTask._compute_final_observations`` with the TIMED-OUT env ids (:func:`bootstrap_env_ids`) so the pre-reset
    observation of a timed-out env equals the next observation of a continuing env in the identical physical state (module
    docstring).  Unsupported terms fall back to the legacy behaviour with a single warning."""
    commands = stateful_command_terms(env)
    if not commands:
        yield ()
        return
    device = None
    for _, mc in commands:
        ts = getattr(mc, "time_steps", None)
        if isinstance(ts, torch.Tensor):
            device = ts.device
            break
    ids = _as_long_ids(env, env_ids, device if device is not None else getattr(env, "device", "cpu"))
    if ids.numel() == 0:
        yield ()
        return
    snapshots: list[_StateSnapshot] = []
    advanced: list[str] = []
    try:
        for name, mc in commands:
            problem = reference_peek_problem(mc)
            if problem is not None:
                if name == MOTION_COMMAND_TERM_NAME:
                    _warn_once(
                        f"unsupported:{name}:{type(mc).__name__}",
                        f"final observation: command term {name!r} ({type(mc).__name__}) has no stock MotionCommand surface ({problem}); "
                        "the pre-reset observation of timed-out envs keeps the legacy one-frame reference lag",
                    )
                continue
            snapshot = _StateSnapshot(mc, env)
            snapshots.append(snapshot)
            try:
                advance_reference_of_command(mc, env, ids)
            except Exception as exc:  # noqa: BLE001 -- never crash the rollout for the bootstrap observation
                snapshots.remove(snapshot)
                snapshot.restore()
                _warn_once(
                    f"failed:{name}:{type(mc).__name__}",
                    f"final observation: advancing the reference of command term {name!r} ({type(mc).__name__}) failed "
                    f"({exc!r}); keeping the legacy one-frame reference lag for it\n{traceback.format_exc()}",
                )
                continue
            advanced.append(name)
        if not advanced:
            _warn_once(
                "none-advanced",
                "final observation: no stateful command term could be advanced; the pre-reset observation of timed-out envs "
                "keeps the legacy one-frame reference lag",
            )
        yield tuple(advanced)
    finally:
        for snapshot in reversed(snapshots):
            snapshot.restore()


__all__ = [
    "ENV_ATTRS_REBOUND_BY_STEP",
    "H20_KEY_STEPS_INDEX",
    "HERO_COMMAND_TRANSITION_HOOK",
    "HERO_REFRESH_HOOKS",
    "MOTION_COMMAND_TERM_NAME",
    "MUTABLE_STATE_BY_CLASS",
    "REQUIRED_REFERENCE_ATTRS",
    "REQUIRED_TENSOR_ATTRS",
    "STOCK_FRAME_CACHE_ATTR",
    "advance_command_transition_isolated",
    "advance_reference_for_final_observation",
    "advance_reference_of_command",
    "bootstrap_env_ids",
    "mutable_state_names",
    "reference_peek_problem",
    "stateful_command_terms",
]
