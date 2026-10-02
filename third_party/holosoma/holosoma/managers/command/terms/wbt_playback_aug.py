"""Playback-speed augmentation for WBT motion references.

Two ``MotionCommand`` subclasses, both disabled by default so that with no
configuration they are numerically identical to the base command:

``TimeWarpMotionCommand``
    Plays each episode back at a per-env random speed in ``[1-w, 1+w]``.  A
    policy trained at exactly 1.0x can overfit to the reference's nominal
    speed; this teaches it to track faster and slower execution.

``HaltAugMotionCommand``
    Time warp plus a per-episode freeze window during which the reference pose
    is held while the robot keeps its momentum.  This trains the "operator
    stopped, arrest motion and hold" behavior that smooth-deceleration mocap
    clips never contain.

Playback is integer-indexed (one reference frame per control step), so a
continuous speed scale is realized by accumulating the fractional drift
``scale - 1`` per env and injecting its integer part into the advance:

* scale 1.2 (fast): accumulator gains +0.2/step, so roughly every fifth step
  advances two frames (one frame skipped).
* scale 0.8 (slow): accumulator loses -0.2/step, so roughly every fifth step
  advances zero frames (one frame repeated).

Both subclasses adjust the advance through ``_playback_advance_delta``, i.e.
BEFORE the parent's clip-end check and relative-target computation, so the
frame index is final for the whole step.  The delta is clamped so a fast-play
step can neither skip past a clip boundary (which would leak the next clip's
pose and bypass the observation-history flush) nor underflow below the clip
start on a slow-play step.
"""

from __future__ import annotations

import os

from loguru import logger

from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.utils.safe_torch_import import torch


_DEFAULT_HALT_MIN_FRAMES = 25
_DEFAULT_HALT_MAX_FRAMES = 75
# Freeze onset is sampled in [halt_min, halt_min + this) so the episode has
# built up motion (and momentum) before the reference stops.
_HALT_START_SPREAD_FRAMES = 60


def _env_float(name: str, default: float = 0.0) -> float:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return float(raw)
    except ValueError:
        return default


def _env_int(name: str, default: int) -> int:
    raw = os.environ.get(name)
    if raw is None or raw.strip() == "":
        return default
    try:
        return int(float(raw))
    except ValueError:
        return default


def _time_warp_width() -> float:
    """Half-width ``w`` of the playback-speed range ``[1-w, 1+w]``.

    Read from ``WBT_MOTION_TIME_WARP`` (``0.2`` means +/-20%).  Unset, empty, or
    non-positive disables the augmentation.  Capped below 1.0 so the slowest
    scale stays strictly positive.
    """
    width = _env_float("WBT_MOTION_TIME_WARP")
    if width <= 0.0:
        return 0.0
    return min(width, 0.95)


class TimeWarpMotionCommand(MotionCommand):
    """``MotionCommand`` with a per-episode random playback speed."""

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        # num_envs/device are not available yet (setup() sets them, init_buffers
        # allocates tensors), so only read scalar configuration here.
        self._tw_width = _time_warp_width()
        self._tw_enabled = self._tw_width > 0.0
        if self._tw_enabled:
            logger.info(
                f"[TimeWarp] playback-speed augmentation enabled: scale ~ U"
                f"[{1.0 - self._tw_width:.2f}, {1.0 + self._tw_width:.2f}] per episode"
            )

    def init_buffers(self, *, reset_adaptive_sampler: bool = True):
        super().init_buffers(reset_adaptive_sampler=reset_adaptive_sampler)
        self._tw_scale = torch.ones(self.num_envs, dtype=torch.float, device=self.device)
        self._tw_accum = torch.zeros(self.num_envs, dtype=torch.float, device=self.device)

    def reset(self, env_ids) -> None:
        super().reset(env_ids)
        if not self._tw_enabled:
            return
        env_ids = self._ensure_index_tensor(env_ids)
        if env_ids.numel() == 0:
            return
        self._tw_accum[env_ids] = 0.0
        if self._env.is_evaluating:
            # Benchmarks always play at nominal speed so numbers stay comparable.
            self._tw_scale[env_ids] = 1.0
        else:
            low = 1.0 - self._tw_width
            span = 2.0 * self._tw_width
            self._tw_scale[env_ids] = low + span * torch.rand(env_ids.numel(), device=self.device)

    def _playback_advance_delta(self, advance_mask: torch.Tensor) -> torch.Tensor:
        delta = super()._playback_advance_delta(advance_mask)
        if not self._tw_enabled:
            return delta

        # Only envs that advance this step accrue drift, so a zero-phase freeze
        # cannot silently bank a frame skip for later.
        advancing = advance_mask.float()
        self._tw_accum += (self._tw_scale - 1.0) * advancing
        extra = torch.floor(self._tw_accum).long()
        self._tw_accum -= extra.float()
        delta = delta + torch.where(advance_mask, extra, torch.zeros_like(extra))

        # Clamp so the resulting index stays inside the current clip.  ``motion_end_idx``
        # is the EXCLUSIVE upper bound (a cumulative sum of clip lengths), so
        # ``end_idx`` is the FIRST FRAME OF THE NEXT CLIP in the concatenated buffer.
        # Clamping to it would let a fast-warping env read one frame of a different
        # motion -- a silent cross-clip pose leak on the step where the rollover is
        # detected, and an out-of-bounds gather for the last clip.  The last valid
        # frame is ``end_idx - 1``.
        start_idx = self.motion.motion_start_idx[self.motion_ids]
        end_idx = self.motion.motion_end_idx[self.motion_ids]
        target = (self.time_steps + delta).clamp(min=start_idx, max=end_idx - 1)
        return target - self.time_steps


class HaltAugMotionCommand(TimeWarpMotionCommand):
    """Time warp plus a per-episode reference-freeze window.

    With probability ``WBT_MOTION_HALT_PROB`` an episode gets one freeze window
    of ``U[WBT_MOTION_HALT_MIN_FRAMES, WBT_MOTION_HALT_MAX_FRAMES]`` frames,
    during which the reference frame index is held constant.  Evaluation
    episodes never freeze, and with the probability unset the whole path is
    skipped.

    While frozen, ``env.halt_frozen`` is True for that env.  Reward terms that
    track the reference velocity must consult it: the reference POSE is held,
    but ``ref_lin_vel_w`` still reads the stored per-frame clip velocity, so
    without the mask the robot would be rewarded for continuing to move.
    """

    def __init__(self, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self._halt_prob = min(max(_env_float("WBT_MOTION_HALT_PROB"), 0.0), 1.0)
        self._halt_min = max(1, _env_int("WBT_MOTION_HALT_MIN_FRAMES", _DEFAULT_HALT_MIN_FRAMES))
        self._halt_max = max(self._halt_min, _env_int("WBT_MOTION_HALT_MAX_FRAMES", _DEFAULT_HALT_MAX_FRAMES))
        self._halt_enabled = self._halt_prob > 0.0
        if self._halt_enabled:
            logger.info(
                f"[HaltAug] halt augmentation enabled: p={self._halt_prob} per episode, "
                f"freeze duration U[{self._halt_min}, {self._halt_max}] frames"
            )

    def init_buffers(self, *, reset_adaptive_sampler: bool = True):
        super().init_buffers(reset_adaptive_sampler=reset_adaptive_sampler)
        self._halt_active = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)
        self._halt_start = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._halt_end = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        self._halt_step = torch.zeros(self.num_envs, dtype=torch.long, device=self.device)
        # Published unconditionally (all-False when disabled) so reward terms can
        # read it without branching on the command subclass.
        self._env.halt_frozen = torch.zeros(self.num_envs, dtype=torch.bool, device=self.device)

    def reset(self, env_ids) -> None:
        super().reset(env_ids)
        if not self._halt_enabled:
            return
        env_ids = self._ensure_index_tensor(env_ids)
        if env_ids.numel() == 0:
            return
        self._halt_step[env_ids] = 0
        self._env.halt_frozen[env_ids] = False
        if self._env.is_evaluating:
            self._halt_active[env_ids] = False
            return
        count = env_ids.numel()
        start = torch.randint(self._halt_min, self._halt_min + _HALT_START_SPREAD_FRAMES, (count,), device=self.device)
        duration = torch.randint(self._halt_min, self._halt_max + 1, (count,), device=self.device)
        self._halt_active[env_ids] = torch.rand(count, device=self.device) < self._halt_prob
        self._halt_start[env_ids] = start
        self._halt_end[env_ids] = start + duration

    def _playback_advance_delta(self, advance_mask: torch.Tensor) -> torch.Tensor:
        delta = super()._playback_advance_delta(advance_mask)
        if not self._halt_enabled:
            return delta

        frozen = self._halt_active & (self._halt_step >= self._halt_start) & (self._halt_step < self._halt_end)
        # The episode-step counter drives the window, so it must advance on every
        # step of the episode regardless of whether the reference advanced.
        self._halt_step += 1
        self._env.halt_frozen = frozen
        # Zero the drift accumulator for frozen envs: otherwise the freeze banks
        # fractional drift and releases it as a frame skip the instant it ends.
        if self._tw_enabled:
            self._tw_accum = torch.where(frozen, torch.zeros_like(self._tw_accum), self._tw_accum)
        return torch.where(frozen, torch.zeros_like(delta), delta)


__all__ = ["TimeWarpMotionCommand", "HaltAugMotionCommand"]
