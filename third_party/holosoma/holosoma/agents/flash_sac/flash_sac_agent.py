"""FlashSAC agent for HoloSoma.

Source: https://github.com/Holiday-Robot/FlashSAC (arXiv:2604.04539).
Uses FastSACEnv, SimpleReplayBuffer, and LoggingHelper for environment access,
replay, checkpoints, and logging. Actions are scaled per joint after tanh;
replay uses per-sample discounts for episode boundaries. The embedder
provides observation normalization. BatchNorm statistics remain per rank."""

from __future__ import annotations

import copy
import itertools
import math
import os
import tempfile
from contextlib import contextmanager
from typing import Callable

import tqdm
from loguru import logger

from holosoma.agents.base_algo.base_algo import BaseAlgo
from holosoma.agents.callbacks.base_callback import RLEvalCallback
from holosoma.agents.fast_sac.fast_sac_agent import FastSACEnv
from holosoma.agents.fast_sac.fast_sac_utils import SimpleReplayBuffer
from holosoma.agents.flash_sac.flash_sac import (
    FlashSACActor,
    FlashSACDoubleCritic,
    FlashSACTemperature,
    compute_categorical_td_target,
    select_min_q_log_probs,
)
from holosoma.agents.flash_sac.flash_sac_utils import (
    DEFAULT_MAX_ABS_REWARD,
    RewardNormalizer,
    build_truncated_zeta_cdf,
    finite_reward_outlier_mask,
    make_weight_normalizer,
    sample_actions_with_zeta_noise,
    save_params,
    seal_replay_before_dropped_collection,
    validate_replay_collection_safety,
    warmup_cosine_decay_scheduler,
)
from holosoma.agents.modules.logging_utils import LoggingHelper
from holosoma.config_types.algo import FlashSACConfig
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.average_meters import TensorAverageMeterDict
from holosoma.utils.helpers import instantiate
from holosoma.utils.inference_helpers import (
    actor_obs_layout_from_env,
    attach_onnx_metadata,
    export_motion_and_policy_as_onnx,
    export_policy_as_onnx,
    get_command_ranges_from_env,
    get_control_gains_from_config,
    get_urdf_text_from_robot_config,
    publish_onnx_atomically,
    validate_onnx_deployment_metadata,
)
from holosoma.utils.safe_torch_import import (
    GradScaler,
    TensorboardSummaryWriter,
    TensorDict,
    autocast,
    nn,
    optim,
    torch,
)

torch.set_float32_matmul_precision("high")


class FlashSACAgent(BaseAlgo):
    config: FlashSACConfig
    env: FastSACEnv  # type: ignore[assignment]
    actor: FlashSACActor
    critic: FlashSACDoubleCritic

    def __init__(
        self, env: BaseTask, config: FlashSACConfig, device: str, log_dir: str, multi_gpu_cfg: dict | None = None
    ):
        wrapped_env = FastSACEnv(env, config.actor_obs_keys, config.critic_obs_keys)

        super().__init__(wrapped_env, config, device, multi_gpu_cfg)  # type: ignore[arg-type]
        self.unwrapped_env = env
        self.log_dir = log_dir
        self.global_step = 0
        self.writer = TensorboardSummaryWriter(log_dir=self.log_dir, flush_secs=10)
        self.logging_helper = LoggingHelper(
            self.writer,
            self.log_dir,
            device=self.device,
            num_envs=self.env.num_envs,
            num_steps_per_env=config.logging_interval,
            num_learning_iterations=config.num_learning_iterations,
            is_main_process=self.is_main_process,
            num_gpus=self.gpu_world_size,
        )

        self.training_metrics = TensorAverageMeterDict()
        self.eval_callbacks: list[RLEvalCallback] = []
        # Optional task-side differentiable penalty evaluated on the actor's
        # reparameterized actions.  Keeping this hook disabled by default makes
        # the native FlashSAC objective bit-for-bit unchanged, while adapters
        # such as VLR Dynamic-K can enforce a physical trust region without
        # waiting for an environment reward penalty to propagate through the
        # replay buffer and critic.
        self.actor_loss_regularizer: Callable[
            [torch.Tensor, TensorDict],
            tuple[torch.Tensor, dict[str, torch.Tensor]],
        ] | None = None
        self._actor_loss_regularizer_metrics: dict[str, torch.Tensor] = {}

    def setup(self) -> None:
        logger.info("Setting up FlashSAC")

        args = self.config
        device = self.device
        env = self.env

        algo_obs_dim_dict = self.env.observation_manager.get_obs_dims()
        algo_history_length_dict: dict[str, int] = {}
        for group_cfg in self.env.observation_manager.cfg.groups.values():
            history_len = getattr(group_cfg, "history_length", 1)
            for term_name in group_cfg.terms:
                algo_history_length_dict[term_name] = history_len

        n_act = self.env.robot_config.actions_dim

        actor_obs_dim = 0
        self.actor_obs_indices: dict[str, dict[str, int]] = {}
        for obs_key in args.actor_obs_keys:
            obs_size = algo_obs_dim_dict[obs_key] * algo_history_length_dict.get(obs_key, 1)
            self.actor_obs_indices[obs_key] = {
                "start": actor_obs_dim,
                "end": actor_obs_dim + obs_size,
                "size": obs_size,
            }
            actor_obs_dim += obs_size
        self.actor_obs_dim = actor_obs_dim

        critic_obs_dim = 0
        self.critic_obs_indices: dict[str, dict[str, int]] = {}
        for obs_key in args.critic_obs_keys:
            obs_size = algo_obs_dim_dict[obs_key] * algo_history_length_dict.get(obs_key, 1)
            self.critic_obs_indices[obs_key] = {
                "start": critic_obs_dim,
                "end": critic_obs_dim + obs_size,
                "size": obs_size,
            }
            critic_obs_dim += obs_size
        logger.info(f"actor_obs_dim: {actor_obs_dim}, critic_obs_dim: {critic_obs_dim}")

        self.scaler = GradScaler(enabled=args.amp)

        # Per-joint action scaling (holosoma env contract; FastSACEnv boundaries)
        action_scale = env._action_boundaries.to(device)

        actor_kwargs = dict(
            obs_indices=self.actor_obs_indices,
            obs_keys=list(args.actor_obs_keys),
            num_blocks=args.actor_num_blocks,
            hidden_dim=args.actor_hidden_dim,
            action_dim=n_act,
            action_scale=action_scale,
            head_init=args.head_init,
            init_std=args.init_std,
        )
        if args.actor_arch in ("transformer", "resmlp"):
            from holosoma.agents.flash_sac.flash_sac import FlashSACTransformerActor

            # Reuse tokenization, normalization, and the policy head with a different token mixer.

            self.actor = FlashSACTransformerActor(
                **actor_kwargs,
                num_tokens=args.actor_num_tokens,
                num_heads=args.actor_num_heads,
                token_mixer="attention" if args.actor_arch == "transformer" else "resmlp",
            ).to(device)
        elif args.actor_arch == "mlp":
            self.actor = FlashSACActor(**actor_kwargs).to(device)
        else:
            raise ValueError(
                f"unknown actor_arch {args.actor_arch!r}; expected 'mlp', 'transformer' or 'resmlp'"
            )

        v_max = args.g_max  # critic support is tied to the reward normalizer scale
        self.critic = FlashSACDoubleCritic(
            obs_indices=self.critic_obs_indices,
            obs_keys=list(args.critic_obs_keys),
            num_blocks=args.critic_num_blocks,
            hidden_dim=args.critic_hidden_dim,
            action_dim=n_act,
            num_bins=args.num_bins,
            min_v=-v_max,
            max_v=v_max,
        ).to(device)
        self.critic_target = FlashSACDoubleCritic(
            obs_indices=self.critic_obs_indices,
            obs_keys=list(args.critic_obs_keys),
            num_blocks=args.critic_num_blocks,
            hidden_dim=args.critic_hidden_dim,
            action_dim=n_act,
            num_bins=args.num_bins,
            min_v=-v_max,
            max_v=v_max,
        ).to(device)

        self.temperature = FlashSACTemperature(args.temp_initial_value).to(device)

        # sigma-derived entropy target (upstream agent.py)
        self.target_entropy = 0.5 * n_act * math.log(2 * math.pi * math.e * args.temp_target_sigma**2)

        # Optimizers: fused Adam, no weight decay (weight norm replaces it),
        # shared warmup+cosine schedule (upstream _init_flashsac_networks).
        total_update_steps = args.lr_decay_steps or (args.num_learning_iterations * args.num_updates)
        lr_schedule = warmup_cosine_decay_scheduler(
            init_value=args.learning_rate_init,
            peak_value=args.learning_rate_peak,
            end_value=args.learning_rate_end,
            warmup_steps=args.lr_warmup_steps,
            decay_steps=total_update_steps,
        )
        use_fused = str(device).startswith("cuda")

        self.actor_optimizer = optim.Adam(self.actor.parameters(), lr=args.learning_rate_peak, fused=use_fused)
        self.critic_optimizer = optim.Adam(self.critic.parameters(), lr=args.learning_rate_peak, fused=use_fused)
        self.temp_optimizer = optim.Adam(self.temperature.parameters(), lr=args.learning_rate_peak, fused=use_fused)

        def _lr_lambda(step: int) -> float:
            return lr_schedule(step) / args.learning_rate_peak

        self.actor_scheduler = optim.lr_scheduler.LambdaLR(self.actor_optimizer, lr_lambda=_lr_lambda)
        self.critic_scheduler = optim.lr_scheduler.LambdaLR(self.critic_optimizer, lr_lambda=_lr_lambda)
        self.temp_scheduler = optim.lr_scheduler.LambdaLR(self.temp_optimizer, lr_lambda=_lr_lambda)

        # Weight normalization: applied once at init and after every optimizer step
        self.normalize_actor_weights = make_weight_normalizer(self.actor)
        self.normalize_critic_weights = make_weight_normalizer(self.critic)
        self.normalize_actor_weights()
        self.normalize_critic_weights()
        make_weight_normalizer(self.critic_target)()

        # Synchronize before copying into the target so all ranks agree
        if self.is_multi_gpu:
            self._synchronize_model_parameters()
        self.critic_target.load_state_dict(self.critic.state_dict())

        # Reward normalizer (tied to the categorical support ±g_max)
        self.reward_normalizer: RewardNormalizer | None = None
        if args.normalize_reward:
            self.reward_normalizer = RewardNormalizer(gamma=args.gamma, G_max=args.g_max, device=torch.device(device))
        self._numerical_safety_invalid_since_log = 0
        self._numerical_safety_rows_since_log = 0
        self._numerical_safety_skipped_collections_since_log = 0
        self._finite_reward_outliers_since_log = 0
        self._finite_reward_outlier_collections_since_log = 0
        self._raw_reward_abs_max_since_log = 0.0

        # Zeta-distributed noise repetition state
        self.zeta_cdf = build_truncated_zeta_cdf(args.zeta_mu, args.zeta_max, device=device)
        self._noise = torch.randn(env.num_envs, n_act, device=device)
        self._noise_repeat_count = torch.tensor(0, dtype=torch.int32, device=device)
        self._noise_repeat_n = torch.tensor(1, dtype=torch.int32, device=device)

        self.rb = SimpleReplayBuffer(
            n_env=env.num_envs,
            buffer_size=args.buffer_size,
            n_obs=actor_obs_dim,
            n_act=n_act,
            n_critic_obs=critic_obs_dim,
            n_steps=args.num_steps,
            gamma=args.gamma,
            device=device,
        )

        self._update_step = 0

        print(self.actor)
        print(self.critic)

    @contextmanager
    def _maybe_amp(self):
        amp_dtype = torch.bfloat16 if self.config.amp_dtype == "bf16" else torch.float16
        with autocast(device_type="cuda", dtype=amp_dtype, enabled=self.config.amp):
            yield

    def _synchronize_model_parameters(self):
        for module in (self.actor, self.critic, self.temperature):
            for param in module.parameters():
                torch.distributed.broadcast(param.data, src=0)
        logger.info(f"Synchronized model parameters across {self.gpu_world_size} GPUs")

    def _all_reduce_model_grads(self, model: nn.Module) -> None:
        """Flatten, all-reduce and average grads across GPUs (one NCCL call)."""
        if not self.is_multi_gpu:
            return
        grads = [p.grad.view(-1) for p in model.parameters() if p.grad is not None]
        if not grads:
            return
        flat = torch.cat(grads)
        torch.distributed.all_reduce(flat, op=torch.distributed.ReduceOp.SUM)
        flat /= self.gpu_world_size
        offset = 0
        for p in model.parameters():
            if p.grad is not None:
                n = p.numel()
                p.grad.copy_(flat[offset : offset + n].view_as(p.grad))
                offset += n

    # ------------------------------------------------------------------
    # Updates (upstream update.py, on holosoma batches)
    # ------------------------------------------------------------------

    def _update_actor_and_temp(self, data: TensorDict) -> tuple[torch.Tensor, ...]:
        args = self.config

        with self._maybe_amp():
            # Concatenate obs and next_obs so the actor BatchNorm sees the same
            # 2B batch statistics as the critic update (upstream trick).
            actor_obs_all = torch.cat([data["observations"], data["next"]["observations"]], dim=0)
            actions_all, info = self.actor(actor_obs_all, training=True)
            log_probs_all = info["log_prob"]

            actions = torch.chunk(actions_all, 2, dim=0)[0]
            log_probs = torch.chunk(log_probs_all, 2, dim=0)[0]

            self.critic.requires_grad_(False)
            qs, _ = self.critic(data["critic_observations"], actions, training=False)
            q = torch.minimum(qs[0], qs[1])
            self.critic.requires_grad_(True)

            temp_value = self.temperature().detach()
            actor_base_loss = (log_probs * temp_value - q).mean()
            actor_regularizer_loss = actor_base_loss.new_zeros(())
            actor_regularizer_metrics: dict[str, torch.Tensor] = {}
            if self.actor_loss_regularizer is not None:
                actor_regularizer_loss, actor_regularizer_metrics = (
                    self.actor_loss_regularizer(actions, data)
                )
                if actor_regularizer_loss.ndim != 0:
                    raise RuntimeError("FlashSAC actor regularizer must return a scalar loss")
                torch._assert_async(
                    torch.isfinite(actor_regularizer_loss),
                    "FlashSAC actor regularizer loss is non-finite",
                )
            actor_loss = actor_base_loss + actor_regularizer_loss

            self._actor_loss_regularizer_metrics = {
                "actor_base_loss": actor_base_loss.detach(),
                "actor_regularizer_loss": actor_regularizer_loss.detach(),
                **{
                    str(name): torch.as_tensor(value, device=self.device).detach().mean()
                    for name, value in actor_regularizer_metrics.items()
                },
            }

            entropy = -log_probs.mean()
            action_std = torch.tensor(0.0, device=self.device)

        self.actor_optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(actor_loss).backward()
        self._all_reduce_model_grads(self.actor)
        self.scaler.unscale_(self.actor_optimizer)
        if args.max_grad_norm > 0:
            actor_grad_norm = torch.nn.utils.clip_grad_norm_(self.actor.parameters(), max_norm=args.max_grad_norm)
        else:
            actor_grad_norm = torch.tensor(0.0, device=self.device)
        self.scaler.step(self.actor_optimizer)
        self.scaler.update()
        self.actor_scheduler.step()
        self.normalize_actor_weights()

        # Temperature update (upstream update_temperature)
        temperature_value = self.temperature().clone()
        temperature_loss = temperature_value * (entropy.detach() - self.target_entropy).mean()
        self.temp_optimizer.zero_grad(set_to_none=True)
        temperature_loss.backward()
        if self.is_multi_gpu:
            for p in self.temperature.parameters():
                if p.grad is not None:
                    torch.distributed.all_reduce(p.grad.data, op=torch.distributed.ReduceOp.SUM)
                    p.grad.data /= self.gpu_world_size
        self.temp_optimizer.step()
        self.temp_scheduler.step()

        return (
            actor_grad_norm.detach(),
            actor_loss.detach(),
            entropy.detach(),
            action_std,
            temperature_loss.detach(),
        )

    def _update_critic(self, data: TensorDict) -> tuple[torch.Tensor, ...]:
        args = self.config

        with self._maybe_amp():
            rewards = data["next"]["rewards"]
            if self.reward_normalizer is not None:
                rewards = self.reward_normalizer.normalize_rewards(rewards)

            dones = data["next"]["dones"].bool()
            truncations = data["next"]["truncations"].bool()
            # holosoma dones include timeouts; the TD target must only cut the
            # bootstrap on true terminations.
            terminated = (dones & ~truncations).float()
            discount = args.gamma ** data["next"]["effective_n_steps"]

            with torch.no_grad():
                next_actions, info = self.actor(data["next"]["observations"], training=False)
                next_actions = next_actions.clone()
                next_actor_log_probs = info["log_prob"].clone()

                temp_value = self.temperature()
                next_actor_entropy = temp_value * next_actor_log_probs

                obs_all = torch.cat([data["critic_observations"], data["next"]["critic_observations"]], dim=0)
                act_all = torch.cat([data["actions"], next_actions], dim=0)

                qs_all, q_infos_all = self.critic_target(obs_all, act_all, training=True)
                next_qs = qs_all.chunk(2, dim=1)[1]
                next_q_log_probs = q_infos_all["log_prob"].chunk(2, dim=1)[1]
                next_q_log_probs = select_min_q_log_probs(next_qs, next_q_log_probs)

                target_probs = compute_categorical_td_target(
                    target_log_probs=next_q_log_probs,
                    reward=rewards,
                    done=terminated,
                    actor_entropy=next_actor_entropy,
                    discount=discount,
                    num_bins=args.num_bins,
                    min_v=-args.g_max,
                    max_v=args.g_max,
                )
                target_value_max = (next_qs.max()).detach()
                target_value_min = (next_qs.min()).detach()

            pred_qs_all, pred_q_infos = self.critic(obs_all, act_all, training=True)
            pred_log_probs = torch.chunk(pred_q_infos["log_prob"], 2, dim=1)[0]

            ce_loss = -(target_probs.unsqueeze(0) * pred_log_probs).sum(dim=-1)  # (2, B)
            critic_loss = ce_loss.mean()

        self.critic_optimizer.zero_grad(set_to_none=True)
        self.scaler.scale(critic_loss).backward()
        self._all_reduce_model_grads(self.critic)
        self.scaler.unscale_(self.critic_optimizer)
        if args.max_grad_norm > 0:
            critic_grad_norm = torch.nn.utils.clip_grad_norm_(self.critic.parameters(), max_norm=args.max_grad_norm)
        else:
            critic_grad_norm = torch.tensor(0.0, device=self.device)
        self.scaler.step(self.critic_optimizer)
        self.scaler.update()
        self.critic_scheduler.step()
        self.normalize_critic_weights()

        # Target critic EMA (upstream: target.lerp_(source, tau))
        with torch.no_grad():
            tgt_ps = [p.data for p in self.critic_target.parameters()]
            src_ps = [p.data for p in self.critic.parameters()]
            torch._foreach_lerp_(tgt_ps, src_ps, args.tau)

        return (
            rewards.mean().detach(),
            critic_grad_norm.detach(),
            critic_loss.detach(),
            target_value_max,
            target_value_min,
        )

    # ------------------------------------------------------------------
    # Training loop (mirrors FastSACAgent.learn)
    # ------------------------------------------------------------------

    def learn(self) -> None:
        args = self.config
        device = self.device
        env = self.env
        rb = self.rb

        if args.compile:
            update_critic = torch.compile(self._update_critic)
            update_actor_and_temp = torch.compile(self._update_actor_and_temp)
        else:
            update_critic = self._update_critic
            update_actor_and_temp = self._update_actor_and_temp

        obs, critic_obs = env.reset_with_critic_obs()
        critic_obs = torch.as_tensor(critic_obs, device=device, dtype=torch.float)
        if self.reward_normalizer is not None:
            # ``learn`` always starts with a simulator-wide reset.  On resume,
            # checkpointed population moments remain valid, but the restored
            # per-lane discounted-return recurrence belongs to the pre-reset
            # episodes and must not leak into the new ones.
            self.reward_normalizer.reset_return_lanes(
                torch.ones(int(env.num_envs), dtype=torch.bool, device=device)
            )

        actor_loss = torch.tensor(0.0, device=device)
        actor_grad_norm = torch.tensor(0.0, device=device)
        policy_entropy = torch.tensor(0.0, device=device)
        action_std = torch.tensor(0.0, device=device)
        temperature_loss = torch.tensor(0.0, device=device)
        pbar = tqdm.tqdm(total=args.num_learning_iterations, initial=self.global_step)

        while self.global_step <= args.num_learning_iterations:
            if self.is_multi_gpu:
                self._synchronize_curriculum_metrics()

            with self.logging_helper.record_collection_time():
                with torch.no_grad(), self._maybe_amp():
                    (
                        self._noise,
                        actions,
                        self._noise_repeat_count,
                        self._noise_repeat_n,
                    ) = sample_actions_with_zeta_noise(
                        actor=self.actor,
                        noise=self._noise,
                        observations=obs,
                        temperature=1.0,
                        cur_count=self._noise_repeat_count,
                        cur_n=self._noise_repeat_n,
                        zeta_cdf=self.zeta_cdf,
                        action_scale=self.actor.action_scale,
                    )

                next_obs, rewards, dones, infos = env.step(actions.float())
                truncations = infos["time_outs"]
                rewards_t = torch.as_tensor(rewards, device=device, dtype=torch.float)
                raw_reward_abs_max = float(torch.abs(rewards_t).max().item())
                self._raw_reward_abs_max_since_log = max(
                    self._raw_reward_abs_max_since_log,
                    raw_reward_abs_max,
                )

                invalid_transitions = infos.get("vls_numerical_safety_invalid_transition")
                if invalid_transitions is None:
                    invalid_transitions = torch.zeros(
                        env.num_envs, dtype=torch.bool, device=device
                    )
                else:
                    invalid_transitions = torch.as_tensor(invalid_transitions, device=device)
                store_collection = validate_replay_collection_safety(
                    invalid_transitions=invalid_transitions,
                    rewards=rewards,
                    dones=dones,
                    truncations=truncations,
                    num_envs=int(env.num_envs),
                )
                terminated = dones.bool() & ~truncations.bool()
                if self.reward_normalizer is not None:
                    reward_outliers = self.reward_normalizer.reward_outlier_mask(
                        rewards_t,
                        terminated=terminated,
                        truncated=truncations.bool(),
                    )
                else:
                    reward_outliers = finite_reward_outlier_mask(
                        rewards_t,
                        max_abs_reward=DEFAULT_MAX_ABS_REWARD,
                    )
                reported_reward_outliers = infos.get("vls_finite_reward_outlier")
                if reported_reward_outliers is not None:
                    reported_reward_outliers = torch.as_tensor(
                        reported_reward_outliers,
                        device=device,
                    )
                    if reported_reward_outliers.shape != (int(env.num_envs),):
                        raise RuntimeError(
                            "vls_finite_reward_outlier must have shape "
                            f"[{int(env.num_envs)}], got "
                            f"{tuple(reported_reward_outliers.shape)}"
                        )
                    if reported_reward_outliers.dtype != torch.bool:
                        raise RuntimeError(
                            "vls_finite_reward_outlier must have bool dtype"
                        )
                    reward_outliers = reward_outliers | reported_reward_outliers
                reward_outlier_count = int(reward_outliers.sum().item())
                local_drop_collection = (not store_collection) or reward_outlier_count > 0
                drop_collection = torch.tensor(
                    int(local_drop_collection),
                    dtype=torch.int32,
                    device=device,
                )
                if self.is_multi_gpu:
                    # Replay columns and return recurrences must stay aligned on
                    # every rank.  One bad lane on one GPU therefore drops the
                    # synchronized collection globally.
                    torch.distributed.all_reduce(
                        drop_collection,
                        op=torch.distributed.ReduceOp.MAX,
                    )
                store_collection = not bool(drop_collection.item())
                invalid_count = int(invalid_transitions.sum().item())
                self._numerical_safety_invalid_since_log += invalid_count
                self._numerical_safety_rows_since_log += int(env.num_envs)
                self._finite_reward_outliers_since_log += reward_outlier_count
                self._finite_reward_outlier_collections_since_log += int(
                    reward_outlier_count > 0
                )
                if not store_collection:
                    # The replay buffer is laid out as one synchronized column
                    # across all environments.  Dropping the complete rare
                    # collection is the only fail-closed option that cannot let
                    # an invalid row enter an n-step sequence.  The reward
                    # normalizer's moments are skipped under the same
                    # disposition.  Sealing the previous replay column creates
                    # a true-terminal or bootstrap-truncation boundary for
                    # every open lane, so no return recurrence may cross it.
                    seal_replay_before_dropped_collection(rb, invalid_transitions)
                    if self.reward_normalizer is not None:
                        self.reward_normalizer.reset_return_lanes(
                            torch.ones_like(invalid_transitions)
                        )
                    self._numerical_safety_skipped_collections_since_log += 1

                # A dropped finite outlier must not poison the episode logger
                # either.  The collection is absent from replay, so the zero is
                # diagnostic sanitation rather than a changed training reward.
                rewards_t = torch.where(
                    reward_outliers.to(device=device),
                    torch.zeros_like(rewards_t),
                    rewards_t,
                )
                self.logging_helper.update_episode_stats(rewards_t, dones, infos)

                next_critic_obs = infos["observations"]["critic"]
                true_next_obs = torch.where(
                    truncations[:, None] > 0, infos["observations"]["final"]["actor_obs"], next_obs
                )
                true_next_critic_obs = torch.where(
                    truncations[:, None] > 0, infos["observations"]["final"]["critic_obs"], next_critic_obs
                )

                if self.reward_normalizer is not None and store_collection:
                    self.reward_normalizer.update_reward_stats(
                        reward=rewards_t,
                        terminated=terminated,
                        truncated=truncations.bool(),
                    )

                transition = TensorDict(
                    {
                        "observations": obs,
                        "actions": torch.as_tensor(actions, device=device, dtype=torch.float),
                        "next": {
                            "observations": true_next_obs,
                            "rewards": rewards_t,
                            "truncations": truncations.long(),
                            "dones": dones.long(),
                        },
                    },
                    batch_size=(env.num_envs,),
                    device=device,
                )
                transition["critic_observations"] = critic_obs
                transition["next"]["critic_observations"] = true_next_critic_obs

                obs = next_obs
                critic_obs = next_critic_obs

                if store_collection:
                    rb.extend(transition)

            # args.batch_size is the global batch size
            batch_size = max(args.batch_size // env.num_envs // self.gpu_world_size, 1)
            if self.global_step > args.learning_starts:
                with self.logging_helper.record_learn_time():
                    for _ in range(args.num_updates):
                        data = rb.sample(batch_size)

                        if self._update_step % args.actor_update_period == 0:
                            (
                                actor_grad_norm,
                                actor_loss,
                                policy_entropy,
                                action_std,
                                temperature_loss,
                            ) = update_actor_and_temp(data)

                        (
                            buffer_rewards,
                            critic_grad_norm,
                            qf_loss,
                            qf_max,
                            qf_min,
                        ) = update_critic(data)
                        self._update_step += 1

                        actor_regularizer_metrics = getattr(
                            self, "_actor_loss_regularizer_metrics", {}
                        )
                        self.training_metrics.add(
                            {
                                "actor_loss": actor_loss,
                                "qf_loss": qf_loss,
                                "qf_max": qf_max,
                                "qf_min": qf_min,
                                "actor_grad_norm": actor_grad_norm,
                                "critic_grad_norm": critic_grad_norm,
                                "buffer_rewards": buffer_rewards,
                                "temperature_loss": temperature_loss,
                                "temperature_value": self.temperature().detach().mean(),
                                "policy_entropy": policy_entropy,
                                "action_std": action_std,
                                "actor_lr": torch.tensor(
                                    self.actor_scheduler.get_last_lr()[0], device=device
                                ),
                                **actor_regularizer_metrics,
                            }
                        )

                if self.global_step % args.logging_interval == 0:
                    with torch.no_grad():
                        accumulated_metrics = self.training_metrics.mean_and_clear()
                        loss_dict = {
                            k: (v.item() if isinstance(v, torch.Tensor) else float(v))
                            for k, v in accumulated_metrics.items()
                        }
                        loss_dict["env_rewards"] = rewards_t.mean().item()
                        rows = max(self._numerical_safety_rows_since_log, 1)
                        loss_dict["numerical_safety_invalid_transition_frac"] = (
                            self._numerical_safety_invalid_since_log / rows
                        )
                        loss_dict["numerical_safety_skipped_collections"] = float(
                            self._numerical_safety_skipped_collections_since_log
                        )
                        loss_dict["finite_reward_outlier_rows"] = float(
                            self._finite_reward_outliers_since_log
                        )
                        loss_dict["finite_reward_outlier_collections"] = float(
                            self._finite_reward_outlier_collections_since_log
                        )
                        loss_dict["raw_reward_abs_max_before_outlier_filter"] = float(
                            self._raw_reward_abs_max_since_log
                        )
                        loss_dict["raw_reward_abs_max_after_sanitize"] = float(
                            rewards_t.abs().max().item()
                        )
                        if self.reward_normalizer is not None:
                            for key, value in self.reward_normalizer.diagnostics().items():
                                loss_dict[f"reward_normalizer_{key}"] = float(value.item())
                        self._numerical_safety_invalid_since_log = 0
                        self._numerical_safety_rows_since_log = 0
                        self._numerical_safety_skipped_collections_since_log = 0
                        self._finite_reward_outliers_since_log = 0
                        self._finite_reward_outlier_collections_since_log = 0
                        self._raw_reward_abs_max_since_log = 0.0
                    self.logging_helper.post_epoch_logging(it=self.global_step, loss_dict=loss_dict, extra_log_dicts={})

                if args.save_interval > 0 and self.global_step > 0 and self.global_step % args.save_interval == 0:
                    if self.is_main_process:
                        logger.info(f"Saving model at global step {self.global_step}")
                        self.save(os.path.join(self.log_dir, f"model_{self.global_step:07d}.pt"))
                        self.export(onnx_file_path=os.path.join(self.log_dir, f"model_{self.global_step:07d}.onnx"))

            if self.global_step >= args.num_learning_iterations:
                break
            self.global_step += 1
            pbar.update(1)

        if self.is_main_process:
            self.save(os.path.join(self.log_dir, f"model_{self.global_step:07d}.pt"))
            self.export(onnx_file_path=os.path.join(self.log_dir, f"model_{self.global_step:07d}.onnx"))

    # ------------------------------------------------------------------
    # Checkpointing / inference / export (FastSAC conventions)
    # ------------------------------------------------------------------

    def save(self, path: str) -> None:  # type: ignore[override]
        env_state = self._collect_env_state()
        save_params(
            self.global_step,
            self.actor,
            self.critic,
            self.critic_target,
            self.temperature,
            self.reward_normalizer,
            self.actor_optimizer,
            self.critic_optimizer,
            self.temp_optimizer,
            self.actor_scheduler,
            self.critic_scheduler,
            self.temp_scheduler,
            self.scaler,
            self.config,
            path,
            save_fn=self.logging_helper.save_checkpoint_artifact,
            env_state=env_state or None,
            metadata=self._checkpoint_metadata(iteration=self.global_step),
        )

    def load(self, ckpt_path: str | None) -> None:
        if not ckpt_path:
            return
        ckpt = torch.load(ckpt_path, map_location=self.device, weights_only=False)
        self.actor.load_state_dict(ckpt["actor_state_dict"])
        self.critic.load_state_dict(ckpt["critic_state_dict"])
        self.critic_target.load_state_dict(ckpt["critic_target_state_dict"])
        self.temperature.load_state_dict(ckpt["temperature_state_dict"])
        if self.reward_normalizer is not None and ckpt.get("reward_normalizer_state") is not None:
            self.reward_normalizer.load_state_dict(ckpt["reward_normalizer_state"])
        self.actor_optimizer.load_state_dict(ckpt["actor_optimizer_state_dict"])
        self.critic_optimizer.load_state_dict(ckpt["critic_optimizer_state_dict"])
        self.temp_optimizer.load_state_dict(ckpt["temp_optimizer_state_dict"])
        for sched, key in (
            (self.actor_scheduler, "actor_scheduler_state_dict"),
            (self.critic_scheduler, "critic_scheduler_state_dict"),
            (self.temp_scheduler, "temp_scheduler_state_dict"),
        ):
            if ckpt.get(key) is not None:
                sched.load_state_dict(ckpt[key])
        if ckpt.get("grad_scaler_state_dict") is not None:
            self.scaler.load_state_dict(ckpt["grad_scaler_state_dict"])
        # The checkpoint records the step that had just COMPLETED (global_step is
        # incremented at the end of the loop body), so a resume starts at the
        # next one; restoring the stored value would re-run it.
        self.global_step = ckpt["global_step"] + 1
        self._restore_env_state(ckpt.get("env_state"))

    @torch.no_grad()
    def get_example_obs(self):
        obs_dict = self.unwrapped_env.reset_all()
        for k in obs_dict:
            obs_dict[k] = obs_dict[k].cpu()
        return {
            "actor_obs": torch.cat([obs_dict[k] for k in self.config.actor_obs_keys], dim=1),
            "critic_obs": torch.cat([obs_dict[k] for k in self.config.critic_obs_keys], dim=1),
        }

    def get_inference_policy(self, device: str | None = None) -> Callable[[dict[str, torch.Tensor]], torch.Tensor]:
        device = device or self.device
        actor = self.actor.to(device)
        actor.eval()

        def policy_fn(obs: dict[str, torch.Tensor]) -> torch.Tensor:
            mean, _ = actor.get_mean_and_std(obs["actor_obs"], training=False)
            return torch.tanh(mean) * actor.action_scale

        return policy_fn

    @property
    def actor_onnx_wrapper(self):
        actor = copy.deepcopy(self.actor).to("cpu")
        actor.eval()

        class ActorWrapper(nn.Module):
            def __init__(self, actor):
                super().__init__()
                self.actor = actor

            def forward(self, actor_obs):
                mean, _ = self.actor.get_mean_and_std(actor_obs, training=False)
                return torch.tanh(mean) * self.actor.action_scale

        return ActorWrapper(actor)

    def export(self, onnx_file_path: str) -> None:
        """Export the deterministic policy as ONNX (deployment, not resume)."""
        was_training = self.actor.training
        self.actor.eval()

        example_input_list = torch.zeros(1, self.actor_obs_dim, device="cpu")

        # Stage, annotate, and validate before publishing with a single rename:
        # log dirs are synced off-host, so an in-place write plus metadata
        # rewrite would expose partial or metadata-less artifacts.
        with tempfile.TemporaryDirectory(prefix="holosoma_onnx_export_") as staging_dir:
            staged_path = os.path.join(staging_dir, os.path.basename(onnx_file_path))

            motion_command = self.unwrapped_env.command_manager.get_state("motion_command")
            if motion_command is not None:
                export_motion_and_policy_as_onnx(
                    self.actor_onnx_wrapper,
                    motion_command,
                    staged_path,
                    self.device,
                )
            else:
                export_policy_as_onnx(
                    wrapper=self.actor_onnx_wrapper,
                    onnx_file_path=staged_path,
                    example_obs_dict={"actor_obs": example_input_list},
                )

            attach_onnx_metadata(onnx_path=staged_path, metadata=self._onnx_deployment_metadata())
            validate_onnx_deployment_metadata(staged_path)
            publish_onnx_atomically(staged_path, onnx_file_path)

        self.logging_helper.save_to_wandb(onnx_file_path)

        if was_training:
            self.actor.train()

    def _onnx_deployment_metadata(self) -> dict:
        """Everything a deployment needs that the ONNX graph cannot express."""
        kp_list, kd_list = get_control_gains_from_config(self.env.robot_config)
        cmd_ranges = get_command_ranges_from_env(self.unwrapped_env)
        action_scales = getattr(self.unwrapped_env, "action_scales", None)
        if action_scales is None:
            action_scale_metadata: float | list[float] = float(self.env.robot_config.control.action_scale)
        else:
            action_scale_metadata = action_scales.detach().cpu().tolist()
        urdf_file_path, urdf_str = get_urdf_text_from_robot_config(self.env.robot_config)

        default_dof_pos = getattr(self.unwrapped_env, "default_dof_pos_base", None)
        if default_dof_pos is not None:
            default_dof_pos = default_dof_pos.detach().cpu().reshape(-1).tolist()

        metadata = {
            "dof_names": self.env.robot_config.dof_names,
            "kp": kp_list,
            "kd": kd_list,
            "action_scale": action_scale_metadata,
            "command_ranges": cmd_ranges,
            "robot_urdf": urdf_str,
            "robot_urdf_path": urdf_file_path,
            # Nominal (pre-randomization) default pose: deployments offset both
            # the observed dof_pos and the commanded target by it.
            "default_dof_pos": default_dof_pos,
            # How the actor obs vector was concatenated at training time.
            "actor_obs_layout": actor_obs_layout_from_env(self.unwrapped_env, self.config.actor_obs_keys),
            # The network output is an absolute joint target (already scaled),
            # NOT a delta for the consumer to scale again.
            "action_contract": "absolute_target_v1",
        }
        metadata.update(self._checkpoint_metadata(iteration=self.global_step))
        return metadata

    @torch.no_grad()
    def evaluate_policy(self, max_eval_steps: int | None = None):
        self._create_eval_callbacks()
        self._pre_evaluate_policy()

        obs = self.env.reset()

        for step in itertools.islice(itertools.count(), max_eval_steps):
            mean, _ = self.actor.get_mean_and_std(obs, training=False)
            actions = torch.tanh(mean) * self.actor.action_scale

            actor_state = {"step": step, "actions": actions, "obs": obs}
            actor_state = self._pre_eval_env_step(actor_state)

            obs, _, _, _ = self.env.step(actor_state["actions"])
            actor_state["obs"] = obs
            actor_state = self._post_eval_env_step(actor_state)

        self._post_evaluate_policy()

    def _create_eval_callbacks(self):
        if self.config.eval_callbacks is not None:
            for cb_name in self.config.eval_callbacks:
                self.eval_callbacks.append(instantiate(self.config.eval_callbacks[cb_name], training_loop=self))

    def _pre_evaluate_policy(self):
        self.env.set_is_evaluating()
        for c in self.eval_callbacks:
            c.on_pre_evaluate_policy()

    def _post_evaluate_policy(self):
        for c in self.eval_callbacks:
            c.on_post_evaluate_policy()

    def _pre_eval_env_step(self, actor_state: dict) -> dict:
        for c in self.eval_callbacks:
            actor_state = c.on_pre_eval_env_step(actor_state)
        return actor_state

    def _post_eval_env_step(self, actor_state: dict) -> dict:
        for c in self.eval_callbacks:
            actor_state = c.on_post_eval_env_step(actor_state)
        return actor_state
