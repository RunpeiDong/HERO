from typing import Any

import numpy as np
import torch
from torch import nn


class AverageMeter(nn.Module):
    def __init__(self, in_shape, max_size):
        super().__init__()
        self.max_size = max_size
        self.current_size = 0
        self.register_buffer("mean", torch.zeros(in_shape, dtype=torch.float32))

    def update(self, values):
        size = values.size()[0]
        if size == 0:
            return
        new_mean = torch.mean(values.float(), dim=0)
        size = np.clip(size, 0, self.max_size)
        old_size = min(self.max_size - size, self.current_size)
        size_sum = old_size + size
        self.current_size = size_sum
        self.mean = (self.mean * old_size + new_mean * size) / size_sum

    def clear(self):
        self.current_size = 0
        self.mean.fill_(0)

    def __len__(self):
        return self.current_size

    def get_mean(self):
        return self.mean.squeeze(0).cpu().numpy()


class TensorAverageMeter:
    def __init__(self):
        self.tensors = []

    def add(self, x):
        if len(x.shape) == 0:
            x = x.unsqueeze(0)
        self.tensors.append(x)

    def mean(self):
        if len(self.tensors) == 0:
            return 0
        cat = torch.cat(self.tensors, dim=0)
        if cat.numel() == 0:
            return 0
        return cat.mean()

    def clear(self):
        self.tensors = []

    def mean_and_clear(self):
        mean = self.mean()
        self.clear()
        return mean


class TensorAverageMeterDict:
    def __init__(self):
        self.data = {}

    def add(self, data_dict):
        for k, v in data_dict.items():
            # Originally used a defaultdict, this had lambda
            # pickling issues with DDP.
            if k not in self.data:
                self.data[k] = TensorAverageMeter()
            self.data[k].add(v)

    def mean(self):
        return {k: v.mean() for k, v in self.data.items()}

    def clear(self):
        self.data = {}

    def mean_and_clear(self):
        mean = self.mean()
        self.clear()
        return mean


class RatioScalar:
    """A ratio metric published as its (numerator, denominator) sufficient statistics.

    A conditional metric such as "mean sole drift of the
    ARMED envs" or "share of the FINISHED episodes whose failure coincided with the timeout" has a denominator that
    varies per step and per rank.  Publishing the per-step quotient as a plain scalar made
    ``holosoma.agents.modules.logging_utils.LoggingHelper`` average quotients with equal weight over steps and ranks:
    ranks with 1 vs 4 armed envs at 3 cm vs 1 cm logged 2 cm instead of the pooled 1.4 cm, and steps with an empty
    conditioning set contributed zeros.  A producer stores ``log_dict[name] = RatioScalar(numerator, denominator)``
    instead; the helper accumulates BOTH parts as sums over the logging interval (``update_episode_stats``) and across
    ranks (the distributed snapshot merge, a SUM like the episode moments) and reports
    ``sum(numerator) / sum(denominator)`` under ``name`` -- no value at all when the pooled denominator is zero.  Plain
    scalars are untouched.

    The class lives here, next to the other metric accumulators, so that env / reward code (the producers) depends on
    this dependency-free module only and never on the agents' logging module (which imports wandb, rich and the
    tensorboard writer at import time); ``logging_utils`` re-exports it.

    Both parts are 0-dim tensors on any device; building one costs no host sync.  ``float(x)`` / ``x.item()`` /
    ``x.value`` give THIS STEP's quotient (0 when its denominator is 0) for direct readers and debugging only -- the
    logger never averages that quotient.
    """

    __slots__ = ("numerator", "denominator")

    def __init__(self, numerator: Any, denominator: Any):
        num = torch.as_tensor(numerator).detach()
        den = torch.as_tensor(denominator).detach()
        if num.numel() != 1 or den.numel() != 1:
            raise ValueError(
                f"RatioScalar parts must be scalars, got shapes {tuple(num.shape)} and {tuple(den.shape)}"
            )
        self.numerator: torch.Tensor = num.reshape(())
        self.denominator: torch.Tensor = den.reshape(())

    def stats(self) -> torch.Tensor:
        """``[numerator, denominator]`` as float64 on the numerator's device (what the logger accumulates)."""
        return torch.stack(
            [self.numerator.to(dtype=torch.float64), self.denominator.to(dtype=torch.float64, device=self.numerator.device)]
        )

    @property
    def value(self) -> torch.Tensor:
        """This step's quotient (0-dim float64; 0 when the denominator is 0).  Debugging / direct readers only."""
        num, den = self.stats()
        return torch.where(den > 0, num / den, torch.zeros_like(num))

    def item(self) -> float:
        return float(self.value.item())

    def __float__(self) -> float:
        return self.item()

    def clone(self) -> "RatioScalar":
        return RatioScalar(self.numerator.clone(), self.denominator.clone())

    def detach(self) -> "RatioScalar":
        return self

    def to(self, *args: Any, **kwargs: Any) -> "RatioScalar":
        return RatioScalar(self.numerator.to(*args, **kwargs), self.denominator.to(*args, **kwargs))

    def __repr__(self) -> str:
        return f"RatioScalar(numerator={self.numerator!r}, denominator={self.denominator!r})"
