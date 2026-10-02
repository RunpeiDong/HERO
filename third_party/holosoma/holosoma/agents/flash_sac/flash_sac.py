"""FlashSAC networks, ported from the official implementation.

Source: https://github.com/Holiday-Robot/FlashSAC (arXiv 2604.04539),
``flash_rl/agents/flashSAC/{layer.py,network.py}`` plus the compiled TD-target
helpers from ``update.py``. The trunk math is kept bit-identical to upstream so
it can be parity-tested against the reference implementation; the only
holosoma-specific additions are:

- ``FlashSACActor``/``FlashSACDoubleCritic`` slice the flattened obs tensor via
  ``obs_indices``/``obs_keys`` (same convention as ``agents.fast_sac``).
- The actor applies a per-joint ``action_scale`` after tanh, because holosoma
  environments expect actions already scaled to joint boundaries. The scale is
  a fixed bijection, so its constant log-Jacobian is deliberately NOT added to
  the log-prob: it would only shift entropy by a constant that the automatic
  temperature absorbs, and omitting it keeps the temperature/entropy dynamics
  identical to upstream FlashSAC.
- ``compute_categorical_td_target`` takes a per-sample ``discount`` tensor
  (gamma ** effective_n_steps from ``SimpleReplayBuffer``) instead of the
  upstream scalar ``gamma ** n_step``, which handles episode boundaries inside
  n-step windows correctly. With a constant discount it is bit-identical.

This module depends on torch only, so the parity tests can run on CPU without
holosoma/simulator imports.
"""

from __future__ import annotations

import math

import torch
import torch.nn as nn
import torch.nn.functional as F


def safe_tanh_log_det_jacobian(x: torch.Tensor) -> torch.Tensor:
    """Numerically safe log|d tanh(x)/dx| (upstream distribution.py)."""
    # log(1 - tanh^2(x)) = 2*(log(2) - x - softplus(-2x))
    return 2.0 * (math.log(2.0) - x - F.softplus(-2.0 * x))


# ---------------------------------------------------------------------------
# Hyperspherically-normalized layers (upstream layer.py, verbatim math)
# ---------------------------------------------------------------------------


class UnitLinear(nn.Module):
    def __init__(self, input_dim: int, output_dim: int):
        super().__init__()
        self.w = nn.Linear(input_dim, output_dim, bias=False)
        nn.init.orthogonal_(self.w.weight, gain=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.w(x)

    def normalize_parameters(self) -> None:
        """Renormalize each output row to the unit sphere (called after every
        optimizer step and once after init)."""
        self.w.weight.copy_(F.normalize(self.w.weight, dim=-1, eps=1e-8))


class UnitBatchNorm(nn.Module):
    running_mean: torch.Tensor
    running_var: torch.Tensor

    def __init__(self, input_dim: int, momentum: float = 0.01, eps: float = 1e-5):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(input_dim))
        self.bias = nn.Parameter(torch.zeros(input_dim))
        self.register_buffer("running_mean", torch.zeros(input_dim))
        self.register_buffer("running_var", torch.ones(input_dim))
        self.momentum = momentum
        self.eps = eps

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        return F.batch_norm(
            x,
            self.running_mean,
            self.running_var,
            self.weight,
            self.bias,
            training=training,
            momentum=self.momentum,
            eps=self.eps,
        )

    def normalize_parameters(self) -> None:
        """Normalize scale and bias jointly to norm sqrt(d)."""
        scale, bias = self.weight.data, self.bias.data
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale + bias * bias, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.data.copy_(scale * norm_factor)
        self.bias.data.copy_(bias * norm_factor)


class UnitRMSNorm(nn.Module):
    def __init__(self, input_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(input_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # Hand-written rms_norm (same math as F.rms_norm): aten::rms_norm has no
        # ONNX opset-13 export, which the deployment pipeline requires.
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight

    def normalize_parameters(self) -> None:
        scale = self.weight.data
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.data.copy_(scale * norm_factor)


class FlashSACEmbedder(nn.Module):
    """Input BatchNorm + UnitLinear. The BatchNorm doubles as the observation
    normalizer, so no external EmpiricalNormalization is used with FlashSAC."""

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.norm = UnitBatchNorm(input_dim)
        self.w = UnitLinear(input_dim, hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        x = self.norm(x, training=training)
        x = self.w(x)
        return x


class FlashSACBlock(nn.Module):
    def __init__(self, hidden_dim: int, expansion: int = 4):
        super().__init__()
        self.w1 = UnitLinear(hidden_dim, hidden_dim * expansion)
        self.w2 = UnitLinear(hidden_dim * expansion, hidden_dim)
        self.norm1 = UnitBatchNorm(hidden_dim * expansion)
        self.norm2 = UnitBatchNorm(hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        residual = x
        x = self.w1(x)
        x = self.norm1(x, training=training)
        x = F.relu(x)
        x = self.w2(x)
        x = self.norm2(x, training=training)
        x = F.relu(x)
        x = x + residual
        return x


class NormalTanhPolicy(nn.Module):
    """Upstream NormalTanhPolicy plus optional learnable per-dim head gains.

    Weight normalization renormalizes mean_w/std_w rows to the unit sphere after
    every step, so head outputs cannot be shrunk via weight init. With
    holosoma's per-joint full-range action scaling, unit-norm heads emit
    |tanh(mean)| ~ 0.6-0.9 at init — near joint limits, so a humanoid falls
    within a few control steps and SAC settles into a die-fast optimum
    (observed: episode length stuck at ~3.3 for 20K iters while FastSAC's
    zero-init starts at ~9 and climbs).

    head_init="safe": gains start at 0 → initial mean = 0 (tanh(0)·scale =
    default pose, FastSAC's zero-init convention) and initial std is a uniform
    init_std. Gains are plain parameters (no normalize_parameters), so the
    optimizer is free to grow them; everything else stays upstream math.
    head_init="upstream": gains fixed-init at 1, biases 0 — bit-identical to
    upstream (parity tests use this).
    """

    def __init__(
        self,
        hidden_dim: int,
        action_dim: int,
        log_std_min: float = -10.0,
        log_std_max: float = 2.0,
        head_init: str = "upstream",
        init_std: float = 0.15,
    ):
        super().__init__()
        if head_init not in ("upstream", "safe"):
            raise ValueError(f"head_init must be 'upstream' or 'safe', got {head_init!r}")
        self.mean_w = UnitLinear(hidden_dim, action_dim)
        self.mean_bias = nn.Parameter(torch.zeros(action_dim))

        self.std_w = UnitLinear(hidden_dim, action_dim)

        gain_init = 1.0 if head_init == "upstream" else 0.0
        self.mean_gain = nn.Parameter(torch.full((action_dim,), gain_init))
        self.std_gain = nn.Parameter(torch.full((action_dim,), gain_init))

        if head_init == "safe":
            # std_bias such that std(init) == init_std uniformly:
            # log_std = min + (max-min)*0.5*(1+tanh(bias)) == log(init_std)
            frac = (math.log(init_std) - log_std_min) / (log_std_max - log_std_min)
            if not 0.0 < frac < 1.0:
                raise ValueError(f"init_std {init_std} outside (log_std_min, log_std_max) range")
            self.std_bias = nn.Parameter(torch.full((action_dim,), math.atanh(2.0 * frac - 1.0)))
        else:
            self.std_bias = nn.Parameter(torch.zeros(action_dim))

        self.log_std_min = log_std_min
        self.log_std_max = log_std_max

    def get_mean_and_std(self, x: torch.Tensor, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
        # Use functional linear for AMP (upstream note)
        mean = self.mean_gain * F.linear(x, self.mean_w.w.weight) + self.mean_bias
        raw_log_std = self.std_gain * F.linear(x, self.std_w.w.weight) + self.std_bias

        # normalize log-stds for stability
        log_std = self.log_std_min + (self.log_std_max - self.log_std_min) * 0.5 * (1 + torch.tanh(raw_log_std))
        std = torch.exp(log_std)

        return mean, std

    def forward(self, x: torch.Tensor, training: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        mean, std = self.get_mean_and_std(x, training)

        dist = torch.distributions.Normal(mean, std)
        raw_action = dist.rsample()
        tanh_action = torch.tanh(raw_action)

        log_prob = dist.log_prob(raw_action)
        log_prob = log_prob - safe_tanh_log_det_jacobian(raw_action)
        log_prob = log_prob.sum(1)

        info: dict[str, torch.Tensor] = {"log_prob": log_prob}
        return tanh_action, info


# ---------------------------------------------------------------------------
# Ensembled layers for the fused double critic (upstream layer.py, verbatim)
# ---------------------------------------------------------------------------


class EnsembleUnitLinear(nn.Module):
    def __init__(self, num_ensemble: int, input_dim: int, output_dim: int):
        super().__init__()
        self.weight = nn.Parameter(torch.empty(num_ensemble, output_dim, input_dim))
        for i in range(num_ensemble):
            nn.init.orthogonal_(self.weight.data[i], gain=1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # [N, B, in] @ [N, in, out] -> [N, B, out]
        return torch.einsum("nbi,noi->nbo", x, self.weight)

    def normalize_parameters(self) -> None:
        self.weight.copy_(F.normalize(self.weight, dim=-1, eps=1e-8))


class EnsembleUnitBatchNorm(nn.Module):
    running_mean: torch.Tensor
    running_var: torch.Tensor

    def __init__(self, num_ensemble: int, input_dim: int, momentum: float = 0.01, eps: float = 1e-5):
        super().__init__()
        self.momentum = momentum
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(num_ensemble, input_dim))
        self.bias = nn.Parameter(torch.zeros(num_ensemble, input_dim))
        self.register_buffer("running_mean", torch.zeros(num_ensemble, input_dim))
        self.register_buffer("running_var", torch.ones(num_ensemble, input_dim))

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        if training:
            mean = x.mean(dim=1, keepdim=True)
            var = x.var(dim=1, correction=0, keepdim=True)
            with torch.no_grad():
                B = x.shape[1]
                # Cast to float32 for running stats (BatchNorm uses float32 even in AMP)
                self.running_mean.lerp_(mean.squeeze(1).float(), self.momentum)
                self.running_var.lerp_((var.squeeze(1) * (B / (B - 1))).float(), self.momentum)
            x = (x - mean) * torch.rsqrt(var + self.eps)
        else:
            x = (x - self.running_mean.unsqueeze(1)) * torch.rsqrt(self.running_var.unsqueeze(1) + self.eps)
        return x * self.weight.unsqueeze(1) + self.bias.unsqueeze(1)

    def normalize_parameters(self) -> None:
        scale, bias = self.weight.data, self.bias.data
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale + bias * bias, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.data.copy_(scale * norm_factor)
        self.bias.data.copy_(bias * norm_factor)


class EnsembleUnitRMSNorm(nn.Module):
    def __init__(self, num_ensemble: int, input_dim: int, eps: float = 1e-6):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(num_ensemble, input_dim))
        self.eps = eps

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        rms = torch.sqrt(torch.mean(x * x, dim=-1, keepdim=True) + self.eps)
        return (x / rms) * self.weight.unsqueeze(1)

    def normalize_parameters(self) -> None:
        scale = self.weight.data
        ndim = scale.shape[-1]
        sqsum = torch.sum(scale * scale, dim=-1, keepdim=True)
        norm_factor = math.sqrt(ndim) * torch.rsqrt(sqsum + 1e-8)
        self.weight.data.copy_(scale * norm_factor)


class EnsembleFlashSACEmbedder(nn.Module):
    def __init__(self, num_ensemble: int, input_dim: int, hidden_dim: int):
        super().__init__()
        self.norm = EnsembleUnitBatchNorm(num_ensemble, input_dim)
        self.w = EnsembleUnitLinear(num_ensemble, input_dim, hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        x = self.norm(x, training=training)
        x = self.w(x)
        return x


class EnsembleFlashSACBlock(nn.Module):
    def __init__(self, num_ensemble: int, hidden_dim: int, expansion: int = 4):
        super().__init__()
        self.w1 = EnsembleUnitLinear(num_ensemble, hidden_dim, hidden_dim * expansion)
        self.w2 = EnsembleUnitLinear(num_ensemble, hidden_dim * expansion, hidden_dim)
        self.norm1 = EnsembleUnitBatchNorm(num_ensemble, hidden_dim * expansion)
        self.norm2 = EnsembleUnitBatchNorm(num_ensemble, hidden_dim)

    def forward(self, x: torch.Tensor, training: bool) -> torch.Tensor:
        residual = x
        x = self.w1(x)
        x = self.norm1(x, training=training)
        x = F.relu(x)
        x = self.w2(x)
        x = self.norm2(x, training=training)
        x = F.relu(x)
        x = x + residual
        return x


class EnsembleCategoricalValue(nn.Module):
    bin_values: torch.Tensor

    def __init__(
        self,
        num_ensemble: int,
        hidden_dim: int,
        num_bins: int,
        min_v: float,
        max_v: float,
    ):
        super().__init__()
        self.w = EnsembleUnitLinear(num_ensemble, hidden_dim, num_bins)
        self.bias = nn.Parameter(torch.zeros(num_ensemble, num_bins))
        self.register_buffer(
            "bin_values",
            torch.linspace(start=min_v, end=max_v, steps=num_bins, dtype=torch.float32).reshape(1, 1, -1),
        )

    def forward(self, x: torch.Tensor, training: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        value = self.w(x) + self.bias.unsqueeze(1)
        log_prob = F.log_softmax(value, dim=-1)
        value = torch.sum(torch.exp(log_prob) * self.bin_values, dim=-1)
        info: dict[str, torch.Tensor] = {"log_prob": log_prob}
        return value, info


# ---------------------------------------------------------------------------
# holosoma-facing actor / critic (obs-dict slicing + action scaling)
# ---------------------------------------------------------------------------


class FlashSACActor(nn.Module):
    """FlashSAC actor trunk with holosoma obs slicing and per-joint action scale.

    Submodule names (embedder/encoder/post_norm/predictor) match upstream
    ``FlashSACActor`` so trunk state dicts transfer directly in parity tests.
    """

    def __init__(
        self,
        obs_indices: dict[str, dict[str, int]],
        obs_keys: list[str],
        num_blocks: int,
        hidden_dim: int,
        action_dim: int,
        action_scale: torch.Tensor | None = None,
        head_init: str = "upstream",
        init_std: float = 0.15,
    ):
        super().__init__()
        self.obs_indices = obs_indices
        self.obs_keys = obs_keys
        input_dim = sum(obs_indices[k]["size"] for k in obs_keys)

        self.embedder = FlashSACEmbedder(input_dim=input_dim, hidden_dim=hidden_dim)
        self.encoder = nn.ModuleList([FlashSACBlock(hidden_dim) for _ in range(num_blocks)])
        self.post_norm = UnitRMSNorm(hidden_dim)
        self.predictor = NormalTanhPolicy(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            head_init=head_init,
            init_std=init_std,
        )

        if action_scale is None:
            action_scale = torch.ones(action_dim)
        self.register_buffer("action_scale", action_scale)

    def process_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [obs[..., self.obs_indices[k]["start"] : self.obs_indices[k]["end"]] for k in self.obs_keys],
            -1,
        )

    def _trunk(self, obs: torch.Tensor, training: bool) -> torch.Tensor:
        x = self.process_obs(obs)
        x = self.embedder(x, training)
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        return x

    def get_mean_and_std(self, obs: torch.Tensor, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
        x = self._trunk(obs, training)
        return self.predictor.get_mean_and_std(x, training)

    def forward(self, obs: torch.Tensor, training: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        """Sample (reparameterized) scaled actions and their log-probs."""
        x = self._trunk(obs, training)
        tanh_action, info = self.predictor(x, training)
        return tanh_action * self.action_scale, info


class _Affine(nn.Module):
    """Per-channel learnable scale and shift (ResMLP's ``Aff``).

    Replaces LayerNorm: no batch or token statistics are computed, so unlike
    BatchNorm it behaves identically at any batch size, and unlike LayerNorm it
    cannot wash out the magnitude differences between observation groups.
    """

    def __init__(self, dim: int) -> None:
        super().__init__()
        self.alpha = nn.Parameter(torch.ones(1, 1, dim))
        self.beta = nn.Parameter(torch.zeros(1, 1, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.alpha + self.beta


class _ResMLPBlock(nn.Module):
    """ResMLP block (arXiv:2105.03404): affine, token mixing, affine, channel MLP.

    A linear token mixer avoids quadratic attention cost. LayerScale starts each
    residual branch near zero so the initial block approximates the identity."""

    def __init__(self, dim: int, num_tokens: int, expansion: int = 4, init_scale: float = 1e-4) -> None:
        super().__init__()
        self.pre_affine = _Affine(dim)
        self.token_mix = nn.Linear(num_tokens, num_tokens)
        self.post_affine = _Affine(dim)
        self.channel_mlp = nn.Sequential(
            nn.Linear(dim, dim * expansion), nn.GELU(), nn.Linear(dim * expansion, dim)
        )
        self.gamma_1 = nn.Parameter(init_scale * torch.ones(dim))
        self.gamma_2 = nn.Parameter(init_scale * torch.ones(dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        h = self.pre_affine(x)
        # transpose so the linear acts across TOKENS, then back
        x = x + self.gamma_1 * self.token_mix(h.transpose(-2, -1)).transpose(-2, -1)
        return x + self.gamma_2 * self.channel_mlp(self.post_affine(x))


class _ExportableAttentionBlock(nn.Module):
    """Pre-norm self-attention + MLP block written for ONNX opset 13.

    ``nn.TransformerEncoderLayer`` and ``nn.MultiheadAttention`` both dispatch to
    ``aten::scaled_dot_product_attention``, which has no opset-13 export (it landed
    in 14) -- the deployment pipeline pins 13, so a policy using them trains fine
    and then fails at export time. The attention is therefore written out as
    matmul/softmax, which exports cleanly and is mathematically the same.
    """

    def __init__(self, dim: int, num_heads: int) -> None:
        super().__init__()
        self.num_heads = int(num_heads)
        self.head_dim = dim // self.num_heads
        self.scale = self.head_dim ** -0.5
        self.norm1 = nn.LayerNorm(dim)
        self.qkv = nn.Linear(dim, dim * 3)
        self.proj = nn.Linear(dim, dim)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(nn.Linear(dim, dim * 4), nn.GELU(), nn.Linear(dim * 4, dim))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        batch, tokens, dim = x.shape
        h = self.norm1(x)
        qkv = self.qkv(h).reshape(batch, tokens, 3, self.num_heads, self.head_dim)
        # -> [batch, heads, tokens, head_dim]
        q, k, v = qkv.permute(2, 0, 3, 1, 4).unbind(0)
        attn = torch.softmax((q @ k.transpose(-2, -1)) * self.scale, dim=-1)
        out = (attn @ v).transpose(1, 2).reshape(batch, tokens, dim)
        x = x + self.proj(out)
        return x + self.mlp(self.norm2(x))


class FlashSACTransformerActor(nn.Module):
    """Transformer-trunk actor using FlashSAC normalization and policy output layers.

    Split observations into num_tokens equal chunks and apply attention between
    them. FlashSACEmbedder provides BatchNorm normalization; NormalTanhPolicy
    provides the action distribution. Attention cost grows quadratically with
    token count."""

    def __init__(
        self,
        obs_indices: dict[str, dict[str, int]],
        obs_keys: list[str],
        num_blocks: int,
        hidden_dim: int,
        action_dim: int,
        num_tokens: int = 4,
        num_heads: int = 4,
        token_mixer: str = "attention",
        action_scale: torch.Tensor | None = None,
        head_init: str = "upstream",
        init_std: float = 0.15,
    ):
        super().__init__()
        self.obs_indices = obs_indices
        self.obs_keys = obs_keys
        input_dim = sum(obs_indices[k]["size"] for k in obs_keys)
        if token_mixer == "attention" and hidden_dim % num_heads != 0:
            raise ValueError(f"hidden_dim {hidden_dim} must be divisible by num_heads {num_heads}")

        self.num_tokens = int(num_tokens)
        self.token_dim = (input_dim + self.num_tokens - 1) // self.num_tokens
        self.pad = self.num_tokens * self.token_dim - input_dim

        # Per-token projection after the shared observation normalizer.
        self.embedder = FlashSACEmbedder(input_dim=input_dim, hidden_dim=hidden_dim)
        self.token_proj = nn.Linear(self.token_dim, hidden_dim)
        self.pos = nn.Parameter(torch.zeros(1, self.num_tokens, hidden_dim))
        # Hand-rolled blocks rather than nn.TransformerEncoderLayer: that module
        # dispatches to aten::scaled_dot_product_attention, which has no ONNX
        # opset-13 export, and opset 13 is what the deployment pipeline requires.
        if token_mixer == "attention":
            blocks = [_ExportableAttentionBlock(hidden_dim, num_heads) for _ in range(num_blocks)]
        elif token_mixer == "resmlp":
            blocks = [_ResMLPBlock(hidden_dim, self.num_tokens) for _ in range(num_blocks)]
        else:
            raise ValueError(f"unknown token_mixer {token_mixer!r}; expected 'attention' or 'resmlp'")
        self.token_mixer = token_mixer
        self.encoder = nn.ModuleList(blocks)
        self.post_norm = UnitRMSNorm(hidden_dim)
        self.predictor = NormalTanhPolicy(
            hidden_dim=hidden_dim,
            action_dim=action_dim,
            head_init=head_init,
            init_std=init_std,
        )

        if action_scale is None:
            action_scale = torch.ones(action_dim)
        self.register_buffer("action_scale", action_scale)

    def process_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [obs[..., self.obs_indices[k]["start"] : self.obs_indices[k]["end"]] for k in self.obs_keys],
            -1,
        )

    def _trunk(self, obs: torch.Tensor, training: bool) -> torch.Tensor:
        x = self.process_obs(obs)
        # Normalize with the embedder's BatchNorm, then tokenize the NORMALIZED
        # observation: tokenizing raw obs would leave per-token scale differences
        # that attention reads as salience.
        x = self.embedder.norm(x, training=training)
        if self.pad:
            x = torch.cat([x, x.new_zeros(*x.shape[:-1], self.pad)], dim=-1)
        tokens = self.token_proj(x.view(*x.shape[:-1], self.num_tokens, self.token_dim)) + self.pos
        for block in self.encoder:
            tokens = block(tokens)
        return self.post_norm(tokens.mean(dim=-2))

    def get_mean_and_std(self, obs: torch.Tensor, training: bool) -> tuple[torch.Tensor, torch.Tensor]:
        return self.predictor.get_mean_and_std(self._trunk(obs, training), training)

    def forward(self, obs: torch.Tensor, training: bool) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        tanh_action, info = self.predictor(self._trunk(obs, training), training)
        return tanh_action * self.action_scale, info


class FlashSACDoubleCritic(nn.Module):
    """Fused double categorical critic with holosoma obs slicing.

    Actions are consumed in env scale (as stored in the replay buffer); the
    embedder's UnitBatchNorm absorbs per-joint scale differences.
    """

    def __init__(
        self,
        obs_indices: dict[str, dict[str, int]],
        obs_keys: list[str],
        num_blocks: int,
        hidden_dim: int,
        action_dim: int,
        num_bins: int,
        min_v: float,
        max_v: float,
        num_qs: int = 2,
    ):
        super().__init__()
        self.obs_indices = obs_indices
        self.obs_keys = obs_keys
        self.num_qs = num_qs
        input_dim = sum(obs_indices[k]["size"] for k in obs_keys) + action_dim

        self.embedder = EnsembleFlashSACEmbedder(num_qs, input_dim, hidden_dim)
        self.encoder = nn.ModuleList([EnsembleFlashSACBlock(num_qs, hidden_dim) for _ in range(num_blocks)])
        self.post_norm = EnsembleUnitRMSNorm(num_qs, hidden_dim)
        self.predictor = EnsembleCategoricalValue(
            num_ensemble=num_qs,
            hidden_dim=hidden_dim,
            num_bins=num_bins,
            min_v=min_v,
            max_v=max_v,
        )

    def process_obs(self, obs: torch.Tensor) -> torch.Tensor:
        return torch.cat(
            [obs[..., self.obs_indices[k]["start"] : self.obs_indices[k]["end"]] for k in self.obs_keys],
            -1,
        )

    def forward(
        self, obs: torch.Tensor, actions: torch.Tensor, training: bool
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        x = torch.cat((self.process_obs(obs), actions), dim=-1)  # [B, in_dim]
        x = x.unsqueeze(0).expand(self.num_qs, -1, -1)  # [num_qs, B, in_dim]
        x = self.embedder(x, training)
        for block in self.encoder:
            x = block(x, training)
        x = self.post_norm(x)
        qs, infos = self.predictor(x, training)
        return qs, infos


class FlashSACTemperature(nn.Module):
    def __init__(self, initial_value: float = 0.01):
        super().__init__()
        self.log_temp = nn.Parameter(torch.tensor([math.log(initial_value)], dtype=torch.float32))

    def forward(self) -> torch.Tensor:
        return torch.exp(self.log_temp)


# ---------------------------------------------------------------------------
# TD-target helpers (upstream update.py, verbatim except per-sample discount)
# ---------------------------------------------------------------------------


def select_min_q_log_probs(
    next_qs: torch.Tensor,  # (2, B)
    next_q_log_probs: torch.Tensor,  # (2, B, num_bins)
) -> torch.Tensor:
    """Select log-probs from the min-Q critic, returning (B, num_bins)."""
    num_bins = next_q_log_probs.shape[-1]
    min_indices = next_qs.argmin(dim=0)  # (B,)
    selected = torch.gather(
        next_q_log_probs,
        dim=0,
        index=min_indices[None, :, None].expand(1, -1, num_bins),
    )[0]  # (B, num_bins)
    return selected


def compute_categorical_td_target(
    target_log_probs: torch.Tensor,  # (B, num_bins)
    reward: torch.Tensor,  # (B,)
    done: torch.Tensor,  # (B,) terminated (NOT truncated)
    actor_entropy: torch.Tensor,  # (B,) temperature * next log-probs
    discount: torch.Tensor,  # (B,) per-sample gamma ** effective_n_steps
    num_bins: int,
    min_v: float,
    max_v: float,
) -> torch.Tensor:
    batch_size = reward.shape[0]

    reward = reward.reshape(-1, 1)
    done = done.reshape(-1, 1)
    discount = discount.reshape(-1, 1)
    actor_entropy = actor_entropy.reshape(-1, 1)

    # Compute target value buckets
    bin_width = (max_v - min_v) / (num_bins - 1)
    bin_values = torch.linspace(
        min_v, max_v, num_bins, device=target_log_probs.device, dtype=target_log_probs.dtype
    ).view(1, -1)

    target_bin_values = reward + discount * (bin_values - actor_entropy) * (1.0 - done)
    target_bin_values = torch.clamp(target_bin_values, min_v, max_v)

    b = (target_bin_values - min_v) / bin_width
    lower = torch.floor(b).long()
    upper = torch.clamp(lower + 1, 0, num_bins - 1)

    frac = b - lower.float()

    target_probs_exp = target_log_probs.exp()
    m_l = target_probs_exp * (1.0 - frac)
    m_u = target_probs_exp * frac

    target_probs = torch.zeros(batch_size, num_bins, dtype=target_probs_exp.dtype, device=target_probs_exp.device)
    target_probs.scatter_add_(1, lower, m_l)
    target_probs.scatter_add_(1, upper, m_u)

    return target_probs
