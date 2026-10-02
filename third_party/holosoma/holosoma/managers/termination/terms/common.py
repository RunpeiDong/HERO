"""Common termination term helpers."""

from __future__ import annotations

from holosoma.utils.safe_torch_import import torch


def timeout_exceeded(env, **_) -> torch.Tensor:
    """Terminate after exactly max_episode_length steps.

    The episode counter increments before this check, matching Isaac Lab semantics."""
    return env.episode_length_buf >= env.max_episode_length
