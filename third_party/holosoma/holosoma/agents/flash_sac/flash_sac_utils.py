"""FlashSAC support utilities ported from the official implementation.

Source: https://github.com/Holiday-Robot/FlashSAC,
``flash_rl/agents/utils/{reward_normalization.py,scheduler.py}`` and the zeta
noise-repeat sampling from ``flash_rl/agents/flashSAC/agent.py``. Math kept
identical to upstream (only the ``@torch.compile`` decorators are dropped here;
the agent compiles its hot paths itself). Torch-only, CPU-testable.
"""

from __future__ import annotations

import math
import os
from typing import Any, Callable

import torch
from torch import nn
from torch.amp import GradScaler


# ---------------------------------------------------------------------------
# Reward normalization (upstream reward_normalization.py)
# ---------------------------------------------------------------------------


# FlashSAC's categorical critic can tolerate ordinary task-scale variation, but
# a finite simulator explosion must not be treated as a very large legitimate
# return.  In VLR rewards are O(1) (the observed healthy maximum is < 4); 1e3 is
# deliberately loose while still separating the 1e11--1e13 failures that would
# otherwise permanently poison the monotone G_r_max statistic.
DEFAULT_MAX_ABS_REWARD = 1_000.0
DEFAULT_MAX_RETURN_GROWTH = 10.0


def finite_reward_outlier_mask(
    rewards: torch.Tensor,
    *,
    max_abs_reward: float = DEFAULT_MAX_ABS_REWARD,
) -> torch.Tensor:
    """Return a mask for finite rewards outside the executable reward scale.

    NaN/Inf values remain a separate fail-fast numerical-safety contract.  This
    helper only catches *finite* simulator explosions, which are especially
    dangerous because the upstream running maximum never decays.
    """

    if not isinstance(rewards, torch.Tensor):
        raise TypeError("rewards must be a torch.Tensor")
    if rewards.ndim != 1:
        raise RuntimeError(f"rewards must be rank-1, got shape {tuple(rewards.shape)}")
    if not math.isfinite(max_abs_reward) or max_abs_reward <= 0.0:
        raise ValueError("max_abs_reward must be finite and positive")
    return torch.isfinite(rewards) & (torch.abs(rewards) > max_abs_reward)


def _update_reward_stats(
    reward: torch.Tensor,
    terminated: torch.Tensor,
    truncated: torch.Tensor,
    G_r: torch.Tensor,
    G_r_max: torch.Tensor,
    gamma: float,
) -> tuple[torch.Tensor, torch.Tensor]:
    reward = reward.to(device=G_r.device, dtype=G_r.dtype)
    done = torch.logical_or(terminated, truncated).to(dtype=G_r.dtype)
    new_G_r = gamma * (1.0 - done) * G_r + reward
    new_G_r_max = torch.maximum(G_r_max, torch.max(torch.abs(new_G_r)))
    return new_G_r, new_G_r_max


def _scale_reward(
    rewards: torch.Tensor,
    G_var: torch.Tensor,
    G_r_max: torch.Tensor,
    G_max: float,
    eps: float,
) -> torch.Tensor:
    output_dtype = rewards.dtype
    rewards = rewards.to(device=G_var.device, dtype=G_var.dtype)
    var_denominator = torch.sqrt(G_var + eps)
    min_required_denominator = G_r_max / G_max
    denominator = torch.maximum(var_denominator, min_required_denominator)
    return (rewards / denominator).to(dtype=output_dtype)


def _update_mean_var_count_from_moments(
    samples: torch.Tensor,
    running_mean: torch.Tensor,
    running_var: torch.Tensor,
    running_count: torch.Tensor,
    epsilon: float,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
    samples = samples.to(device=running_mean.device, dtype=running_mean.dtype)
    sample_mean = torch.mean(samples, dim=0)
    sample_var = torch.var(samples, dim=0, unbiased=False)
    sample_count = float(samples.shape[0])

    delta = sample_mean - running_mean
    total_count = running_count + sample_count
    ratio = sample_count / total_count

    new_mean = running_mean + delta * ratio
    m_a = running_var * (running_count + epsilon)
    m_b = sample_var * sample_count
    M2 = m_a + m_b + torch.square(delta) * running_count * ratio
    new_var = M2 / total_count

    return new_mean, new_var, total_count


class RunningMeanStd:
    """Tracks the mean, variance and count of values (Chan's parallel algorithm)."""

    def __init__(
        self,
        device: torch.device,
        epsilon: float = 1e-4,
        shape: tuple[int, ...] = (),
        dtype: torch.dtype = torch.float32,
    ):
        self.mean = torch.zeros(shape, dtype=dtype, device=device)
        self.var = torch.ones(shape, dtype=dtype, device=device)
        self.count = torch.tensor(0.0, dtype=dtype, device=device)
        self.epsilon = epsilon
        self.device = device

    def update(self, x: torch.Tensor) -> None:
        self.mean, self.var, self.count = _update_mean_var_count_from_moments(
            samples=x,
            running_mean=self.mean,
            running_var=self.var,
            running_count=self.count,
            epsilon=self.epsilon,
        )


class RewardNormalizer:
    """Scale rewards by the std of a running estimate of discounted returns,
    floored so |normalized return| stays within G_max. The critic categorical
    support (+-G_max) is designed around this normalization — do not disable
    one without the other."""

    def __init__(
        self,
        gamma: float,
        G_max: float,
        device: torch.device,
        epsilon: float = 1e-8,
        max_abs_reward: float = DEFAULT_MAX_ABS_REWARD,
        max_return_growth: float = DEFAULT_MAX_RETURN_GROWTH,
    ):
        if not math.isfinite(max_abs_reward) or max_abs_reward <= 0.0:
            raise ValueError("max_abs_reward must be finite and positive")
        if not math.isfinite(max_return_growth) or max_return_growth <= 1.0:
            raise ValueError("max_return_growth must be finite and greater than one")
        self.gamma = gamma
        # Running-return moments are float64 even when policy/reward tensors are
        # float32.  One rare but finite simulator explosion can otherwise make
        # Chan's delta**2 * count overflow to inf and silently turn all future
        # normalized rewards into zero.
        self.G_r = torch.zeros(1, dtype=torch.float64, device=device)
        self.G_r_max = torch.zeros(1, dtype=torch.float64, device=device)
        self.G_rms = RunningMeanStd(shape=(1,), device=device, dtype=torch.float64)
        self.G_max = G_max
        self.epsilon = epsilon
        self.device = device
        self.max_abs_reward = float(max_abs_reward)
        self.max_return_growth = float(max_return_growth)

    def reward_outlier_mask(
        self,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> torch.Tensor:
        """Detect raw-reward or discounted-return jumps before state mutation.

        The absolute reward guard catches impossible task rewards.  The return
        guard catches a corrupted recurrence lane even when its current reward
        is small.  Normal growth remains byte-for-byte compatible with upstream
        FlashSAC; only a >10x jump beyond both the historical maximum and a
        generous bootstrap floor is rejected.
        """

        if not bool(torch.isfinite(reward).all()):
            raise FloatingPointError("FlashSAC reward normalizer received NaN/Inf reward")
        for name, value in {"reward": reward, "terminated": terminated, "truncated": truncated}.items():
            if not isinstance(value, torch.Tensor):
                raise TypeError(f"{name} must be a torch.Tensor")
            if value.ndim != 1:
                raise RuntimeError(f"{name} must be rank-1, got shape {tuple(value.shape)}")
        if reward.shape != terminated.shape or reward.shape != truncated.shape:
            raise RuntimeError("reward, terminated and truncated must have identical shapes")

        raw_outlier = finite_reward_outlier_mask(
            reward,
            max_abs_reward=self.max_abs_reward,
        )
        reward64 = reward.to(device=self.device, dtype=self.G_r.dtype)
        done64 = torch.logical_or(terminated, truncated).to(
            device=self.device,
            dtype=self.G_r.dtype,
        )
        if self.G_r.numel() not in (1, reward.numel()):
            raise RuntimeError(
                f"reward-return lanes {tuple(self.G_r.shape)} do not match "
                f"reward shape {tuple(reward.shape)}"
            )
        candidate_return = self.gamma * (1.0 - done64) * self.G_r + reward64
        allowed_return = torch.maximum(
            self.G_r_max * self.max_return_growth,
            torch.as_tensor(
                self.max_abs_reward,
                device=self.device,
                dtype=self.G_r.dtype,
            ),
        )
        return raw_outlier.to(device=self.device) | (torch.abs(candidate_return) > allowed_return)

    def update_reward_stats(
        self,
        reward: torch.Tensor,
        terminated: torch.Tensor,
        truncated: torch.Tensor,
    ) -> None:
        outlier_mask = self.reward_outlier_mask(reward, terminated, truncated)
        if bool(outlier_mask.any()):
            count = int(outlier_mask.sum().item())
            largest = float(torch.abs(reward[outlier_mask.to(device=reward.device)]).max().item())
            raise FloatingPointError(
                "FlashSAC finite reward/return outlier reached the normalizer "
                f"({count} lane(s), max |raw reward|={largest:.6g}); "
                "the synchronized collection must be dropped before updating statistics"
            )
        self.G_r, self.G_r_max = _update_reward_stats(
            reward=reward,
            terminated=terminated,
            truncated=truncated,
            G_r=self.G_r,
            G_r_max=self.G_r_max,
            gamma=self.gamma,
        )
        self.G_rms.update(self.G_r)
        self._assert_finite("update")

    def normalize_rewards(self, rewards: torch.Tensor) -> torch.Tensor:
        if not bool(torch.isfinite(rewards).all()):
            raise FloatingPointError("FlashSAC critic received NaN/Inf reward")
        self._assert_finite("normalize")
        normalized = _scale_reward(
            rewards=rewards,
            G_var=self.G_rms.var,
            G_r_max=self.G_r_max,
            G_max=self.G_max,
            eps=self.epsilon,
        )
        if not bool(torch.isfinite(normalized).all()):
            raise FloatingPointError("FlashSAC reward normalization produced NaN/Inf")
        return normalized

    def reset_return_lanes(self, boundary_mask: torch.Tensor) -> None:
        """Reset selected discounted-return recurrence lanes without moments.

        ``G_r`` is episode-local state, unlike the population moments and
        ``G_r_max``.  Callers must reset every lane whose trajectory has a
        replay or simulator boundary.  In particular, dropping one synchronized
        vectorized collection seals *all* still-open replay lanes, so all return
        lanes must be reset even when only one environment was numerically bad.
        The initial singleton state is expanded lazily without changing the
        running population statistics.
        """

        if not isinstance(boundary_mask, torch.Tensor):
            raise TypeError("boundary_mask must be a torch.Tensor")
        if boundary_mask.ndim != 1 or boundary_mask.dtype != torch.bool:
            raise RuntimeError("boundary_mask must be a rank-1 bool tensor")
        mask = boundary_mask.to(device=self.device)
        if self.G_r.numel() == 1 and mask.numel() != 1:
            self.G_r = self.G_r.expand(mask.numel()).clone()
        if self.G_r.shape != mask.shape:
            raise RuntimeError(
                f"reward-return lanes {tuple(self.G_r.shape)} do not match "
                f"invalid mask {tuple(mask.shape)}"
            )
        self.G_r = torch.where(mask, torch.zeros_like(self.G_r), self.G_r)
        self._assert_finite("return-lane reset")

    def _assert_finite(self, context: str) -> None:
        state = self.state_dict()
        bad = [name for name, value in state.items() if not bool(torch.isfinite(value).all())]
        if bad:
            raise FloatingPointError(
                f"FlashSAC reward-normalizer state is non-finite during {context}: {bad}"
            )

    def diagnostics(self) -> dict[str, torch.Tensor]:
        self._assert_finite("diagnostics")
        return {
            "return_abs_max": self.G_r_max.detach(),
            "return_mean": self.G_rms.mean.detach(),
            "return_std": torch.sqrt(self.G_rms.var + self.epsilon).detach(),
            "sample_count": self.G_rms.count.detach(),
        }

    def state_dict(self) -> dict[str, torch.Tensor]:
        return {
            "G_r": self.G_r,
            "G_r_max": self.G_r_max,
            "G_rms_mean": self.G_rms.mean,
            "G_rms_var": self.G_rms.var,
            "G_rms_count": self.G_rms.count,
        }

    def load_state_dict(self, state: dict[str, torch.Tensor]) -> None:
        dtype = torch.float64
        self.G_r = state["G_r"].to(device=self.device, dtype=dtype)
        self.G_r_max = state["G_r_max"].to(device=self.device, dtype=dtype)
        self.G_rms.mean = state["G_rms_mean"].to(device=self.device, dtype=dtype)
        self.G_rms.var = state["G_rms_var"].to(device=self.device, dtype=dtype)
        self.G_rms.count = state["G_rms_count"].to(device=self.device, dtype=dtype)
        self._assert_finite("load")


def validate_replay_collection_safety(
    *,
    invalid_transitions: torch.Tensor,
    rewards: torch.Tensor,
    dones: torch.Tensor,
    truncations: torch.Tensor,
    num_envs: int,
) -> bool:
    """Validate the numerical-safety handoff and decide replay disposition.

    FlashSAC's n-step replay is appended one synchronized environment column at
    a time.  If even one row is numerically invalid, the complete collection
    must be excluded from both replay *and* reward-normalizer statistics.  This
    function contains no behavioral/tracking criterion; it only validates the
    explicit mask emitted by the simulator safety termination.

    Returns ``True`` exactly when the collection is safe to store and use for
    reward-statistics updates.  The caller must separately reset the return
    recurrence for invalid terminal lanes; those resets are state-boundary
    bookkeeping, not a running-moment update.
    """

    values = {
        "invalid_transitions": invalid_transitions,
        "rewards": rewards,
        "dones": dones,
        "truncations": truncations,
    }
    for name, value in values.items():
        if not isinstance(value, torch.Tensor):
            raise TypeError(f"{name} must be a torch.Tensor")
        if value.shape != (num_envs,):
            raise RuntimeError(
                f"{name} must have shape [{num_envs}], got {tuple(value.shape)}"
            )
    if invalid_transitions.dtype != torch.bool:
        raise RuntimeError(
            "vls_numerical_safety_invalid_transition must have bool dtype, "
            f"got {invalid_transitions.dtype}"
        )

    invalid_count = int(invalid_transitions.sum().item())
    if invalid_count == 0:
        if not bool(torch.isfinite(rewards).all()):
            raise FloatingPointError(
                "non-finite reward reached a collection not classified by numerical safety"
            )
        return True

    if not bool(dones.bool()[invalid_transitions].all()):
        raise RuntimeError("numerical-safety invalid rows were not reset by the environment")
    if bool(truncations.bool()[invalid_transitions].any()):
        raise RuntimeError(
            "numerical-safety invalid rows must be true terminals, not timeouts"
        )
    if not bool((rewards[invalid_transitions] == 0.0).all()):
        raise RuntimeError(
            "numerical-safety invalid rows must carry sanitized zero reward"
        )
    if not bool(torch.isfinite(rewards).all()):
        raise FloatingPointError(
            "non-finite reward remained after numerical-safety sanitation"
        )
    return False


@torch.no_grad()
def seal_replay_before_dropped_collection(
    replay_buffer: Any,
    invalid_transitions: torch.Tensor,
) -> dict[str, int | bool]:
    """Prevent n-step sequences from crossing a dropped vectorized column.

    The last accepted row is a true terminal for lanes whose next transition
    numerically failed.  For the other lanes, dropping the synchronized column
    merely creates an observation gap, so the last accepted row is marked as a
    timeout/truncation and may bootstrap from its own valid next observation.
    Existing episode boundaries are preserved exactly.
    """

    if invalid_transitions.ndim != 1 or invalid_transitions.dtype != torch.bool:
        raise RuntimeError("invalid_transitions must be a rank-1 bool tensor")
    n_env = int(getattr(replay_buffer, "n_env"))
    if invalid_transitions.shape != (n_env,):
        raise RuntimeError(
            f"invalid mask {tuple(invalid_transitions.shape)} != replay lanes [{n_env}]"
        )
    n_steps = int(getattr(replay_buffer, "n_steps"))
    ptr = int(getattr(replay_buffer, "ptr"))
    if n_steps <= 1 or ptr <= 0:
        return {
            "sealed": False,
            "previous_index": -1,
            "true_terminal_lanes": 0,
            "truncated_gap_lanes": 0,
        }

    buffer_size = int(getattr(replay_buffer, "buffer_size"))
    previous_index = (ptr - 1) % buffer_size
    dones = replay_buffer.dones[:, previous_index]
    truncations = replay_buffer.truncations[:, previous_index]
    mask = invalid_transitions.to(device=dones.device)
    previously_open = ~dones.bool()
    invalid_open = previously_open & mask
    valid_gap_open = previously_open & ~mask
    dones[previously_open] = 1
    truncations[invalid_open] = 0
    truncations[valid_gap_open] = 1
    return {
        "sealed": bool(previously_open.any()),
        "previous_index": previous_index,
        "true_terminal_lanes": int(invalid_open.sum().item()),
        "truncated_gap_lanes": int(valid_gap_open.sum().item()),
    }


# ---------------------------------------------------------------------------
# LR schedule (upstream scheduler.py)
# ---------------------------------------------------------------------------


def warmup_cosine_decay_scheduler(
    init_value: float,
    peak_value: float,
    end_value: float,
    warmup_steps: int,
    decay_steps: int,
) -> Callable[[int], float]:
    def scheduler(step: int) -> float:
        if warmup_steps > 0 and step < warmup_steps:
            return init_value + (peak_value - init_value) * (step / warmup_steps)
        # optax convention: decay_steps is the total schedule length
        elif step < decay_steps:
            decay_step = step - warmup_steps
            progress = decay_step / (decay_steps - warmup_steps)
            return end_value + (peak_value - end_value) * 0.5 * (1 + math.cos(math.pi * progress))
        else:
            return end_value

    return scheduler


# ---------------------------------------------------------------------------
# Zeta-distributed noise repetition (upstream agent.py)
# ---------------------------------------------------------------------------


def build_truncated_zeta_cdf(mu: float, max_n: int, device: torch.device | str = "cpu") -> torch.Tensor:
    ns = torch.arange(1, max_n + 1, dtype=torch.float32, device=device)
    pmf = ns ** (-mu)
    pmf = pmf / torch.sum(pmf)
    return torch.cumsum(pmf, dim=0)


def sample_integer_from_cdf(cdf: torch.Tensor) -> torch.Tensor:
    """Sample an integer in [1, len(cdf)] from the CDF; returns 0-d int32."""
    u = torch.rand((), device=cdf.device)
    idx = torch.argmax((u < cdf).to(torch.int32))
    return (idx + 1).to(torch.int32)


def sample_actions_with_zeta_noise(
    actor: nn.Module,
    noise: torch.Tensor,
    observations: torch.Tensor,
    temperature: float,
    cur_count: torch.Tensor,
    cur_n: torch.Tensor,
    zeta_cdf: torch.Tensor,
    action_scale: torch.Tensor,
) -> tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    """Rollout action sampling with time-correlated (zeta-repeated) noise.

    The exploration noise epsilon is redrawn only every n steps where
    n ~ truncated Zeta(mu); between redraws the same epsilon is reused,
    giving temporally-correlated exploration. All envs share one repeat
    counter (upstream behavior). Scaled to env action space via action_scale.
    """
    mean, std = actor.get_mean_and_std(observations, training=False)
    if temperature == 0.0:
        return noise, torch.tanh(mean) * action_scale, cur_count, cur_n

    reinit = (cur_count == 0) | (cur_count >= cur_n)

    new_noise = torch.randn_like(mean)
    new_n = sample_integer_from_cdf(zeta_cdf)

    noise = torch.where(reinit, new_noise, noise)
    cur_n = torch.where(reinit, new_n, cur_n)
    cur_count = torch.where(reinit, torch.zeros_like(cur_count), cur_count)

    actions = torch.tanh(mean + std * noise * temperature) * action_scale

    return noise, actions, cur_count + 1, cur_n


# ---------------------------------------------------------------------------
# Weight-norm helper + checkpointing
# ---------------------------------------------------------------------------


def make_weight_normalizer(*modules: nn.Module) -> Callable[[], None]:
    """Collect all submodules exposing normalize_parameters() and return a
    single callable applying them (upstream Network.use_weight_normalization)."""
    norm_modules = [m for mod in modules for m in mod.modules() if hasattr(m, "normalize_parameters")]

    @torch.no_grad()
    def normalize() -> None:
        for m in norm_modules:
            m.normalize_parameters()

    return normalize


def cpu_state(sd: dict[str, Any]) -> dict[str, Any]:
    return {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in sd.items()}


def save_params(
    global_step: int,
    actor: nn.Module,
    critic: nn.Module,
    critic_target: nn.Module,
    temperature: nn.Module,
    reward_normalizer: RewardNormalizer | None,
    actor_optimizer: torch.optim.Optimizer,
    critic_optimizer: torch.optim.Optimizer,
    temp_optimizer: torch.optim.Optimizer,
    actor_scheduler: Any,
    critic_scheduler: Any,
    temp_scheduler: Any,
    scaler: GradScaler | None,
    args: Any,
    save_path: str,
    save_fn=torch.save,
    metadata: dict[str, Any] | None = None,
    env_state: dict[str, torch.Tensor | float] | None = None,
):
    """Save FlashSAC training state (mirrors fast_sac_utils.save_params layout)."""
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    save_dict: dict[str, Any] = {
        "actor_state_dict": cpu_state(actor.state_dict()),
        "critic_state_dict": cpu_state(critic.state_dict()),
        "critic_target_state_dict": cpu_state(critic_target.state_dict()),
        "temperature_state_dict": cpu_state(temperature.state_dict()),
        "reward_normalizer_state": (cpu_state(reward_normalizer.state_dict()) if reward_normalizer else None),
        "actor_optimizer_state_dict": actor_optimizer.state_dict(),
        "critic_optimizer_state_dict": critic_optimizer.state_dict(),
        "temp_optimizer_state_dict": temp_optimizer.state_dict(),
        "actor_scheduler_state_dict": actor_scheduler.state_dict() if actor_scheduler else None,
        "critic_scheduler_state_dict": critic_scheduler.state_dict() if critic_scheduler else None,
        "temp_scheduler_state_dict": temp_scheduler.state_dict() if temp_scheduler else None,
        "grad_scaler_state_dict": scaler.state_dict() if scaler is not None else None,
        "args": vars(args),
        "global_step": global_step,
    }
    if env_state:
        save_dict["env_state"] = env_state
    if metadata is None:
        raise ValueError("Checkpoint metadata is required when saving FlashSAC parameters.")
    save_dict.update(metadata)
    save_fn(save_dict, save_path)
    print(f"Saved parameters and configuration to {save_path}")
