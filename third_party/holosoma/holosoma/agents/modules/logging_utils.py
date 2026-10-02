from __future__ import annotations

import os
import pathlib
import statistics
import time
from collections import deque
from contextlib import contextmanager
from typing import Any, Generator, TypedDict

import torch
import wandb
from loguru import logger
from rich.console import Console
from rich.live import Live
from rich.panel import Panel
from torch.utils.tensorboard import SummaryWriter

from holosoma.utils.average_meters import RatioScalar, TensorAverageMeterDict

# ``RatioScalar`` is defined in holosoma.utils.average_meters (dependency-free) so that the env / reward producers never
# import this module (which loads wandb, rich, and tensorboard); it is re-exported here for the logger
# side and for callers that already import it from this module (re-export, not an unused import).

console = Console()


class LogDict(TypedDict):
    """Dictionary containing iteration info, timing, and buffers for logging."""

    it: int
    """Current iteration number."""

    loss_dict: dict[str, float]
    """Dictionary of loss values."""


class TrainLogDict(TypedDict):
    """Dictionary containing training metrics."""

    fps: float
    """Frames per second (training speed)."""

    # Additional metrics can be added here


class _DistributedLoggingSnapshot(TypedDict):
    """Detached, compact sufficient statistics for one logging interval."""

    loss_dict: dict[str, float]
    episode_moments: dict[str, tuple[float, int]]
    raw_episode_moments: dict[str, tuple[float, int]]
    env_moments: dict[str, tuple[float, int]]
    interval_moments: dict[str, tuple[float, int]]
    ratio_moments: dict[str, tuple[float, float]]
    """``{name: (sum of numerators, sum of denominators)}`` of the :class:`RatioScalar` metrics (hero_isaacsim patch)."""
    has_logging_done_override: bool


def _split_ratio_metrics(to_log: dict[str, Any]) -> tuple[dict[str, Any], dict[str, RatioScalar]]:
    """Separate the :class:`RatioScalar` entries of an env log dict from the plain scalars (the dict is not mutated:
    ``extras["to_log"]`` IS ``env.log_dict``, which persists across steps)."""
    ratios = {key: value for key, value in to_log.items() if isinstance(value, RatioScalar)}
    if not ratios:
        return to_log, ratios
    plain = {key: value for key, value in to_log.items() if not isinstance(value, RatioScalar)}
    return plain, ratios


def _resolve_ratio_metrics(moments: dict[str, tuple[float, float]]) -> dict[str, float]:
    """``sum(numerator) / sum(denominator)`` per ratio metric; a metric whose pooled denominator is 0 is omitted (there
    is no conditional mean to report).  A negative pooled denominator is a producer bug and is reported, not divided."""
    out: dict[str, float] = {}
    for key, (numerator, denominator) in moments.items():
        if denominator > 0:
            out[key] = float(numerator) / float(denominator)
        elif denominator < 0:
            logger.warning(
                "RatioScalar metric {!r} accumulated a negative denominator ({}); dropped from this interval", key, denominator
            )
    return out


def _summarize_mapping_history(
    history: list[dict[str, Any]],
) -> dict[str, tuple[float, int]]:
    """Return scalar sum/count moments without retaining rollout tensors."""

    moments: dict[str, tuple[float, int]] = {}
    for values_by_key in history:
        for key, value in values_by_key.items():
            tensor = torch.as_tensor(value).detach()
            if tensor.numel() == 0:
                continue
            value_sum = float(tensor.to(dtype=torch.float64).sum().item())
            previous_sum, previous_count = moments.get(key, (0.0, 0))
            moments[key] = (
                previous_sum + value_sum,
                previous_count + int(tensor.numel()),
            )
    return moments


def _merge_distributed_logging_snapshots(
    snapshots: list[_DistributedLoggingSnapshot],
) -> _DistributedLoggingSnapshot:
    """Merge per-rank logging snapshots using their natural sample weights.

    Losses are already minibatch means on every PPO rank and are therefore
    averaged uniformly across ranks. Episode and environment metrics can have
    different numbers of observations, so those retain sum/count moments.
    """

    if not snapshots:
        raise ValueError("at least one distributed logging snapshot is required")

    expected_loss_keys = set(snapshots[0]["loss_dict"])
    for rank, snapshot in enumerate(snapshots[1:], start=1):
        actual_loss_keys = set(snapshot["loss_dict"])
        if actual_loss_keys != expected_loss_keys:
            missing = sorted(expected_loss_keys - actual_loss_keys)
            unexpected = sorted(actual_loss_keys - expected_loss_keys)
            raise RuntimeError(
                "distributed loss_dict keys differ across ranks: "
                f"rank={rank}, missing={missing}, unexpected={unexpected}"
            )

    merged_loss = {
        key: sum(snapshot["loss_dict"][key] for snapshot in snapshots)
        / len(snapshots)
        for key in sorted(expected_loss_keys)
    }

    def merge_moments(
        per_rank_moments: list[dict[str, tuple[float, int]]],
    ) -> dict[str, tuple[float, int]]:
        merged: dict[str, tuple[float, int]] = {}
        for rank_moments in per_rank_moments:
            for key, (value_sum, count) in rank_moments.items():
                previous_sum, previous_count = merged.get(key, (0.0, 0))
                merged[key] = (
                    previous_sum + float(value_sum),
                    previous_count + int(count),
                )
        return merged

    def merge_ratio_moments(
        per_rank_moments: list[dict[str, tuple[float, float]]],
    ) -> dict[str, tuple[float, float]]:
        merged: dict[str, tuple[float, float]] = {}
        for rank_moments in per_rank_moments:
            for key, (numerator, denominator) in rank_moments.items():
                previous_numerator, previous_denominator = merged.get(key, (0.0, 0.0))
                merged[key] = (
                    previous_numerator + float(numerator),
                    previous_denominator + float(denominator),
                )
        return merged

    return {
        "loss_dict": merged_loss,
        "episode_moments": merge_moments(
            [snapshot["episode_moments"] for snapshot in snapshots]
        ),
        "raw_episode_moments": merge_moments(
            [snapshot["raw_episode_moments"] for snapshot in snapshots]
        ),
        "env_moments": merge_moments(
            [snapshot["env_moments"] for snapshot in snapshots]
        ),
        "interval_moments": merge_moments(
            [snapshot["interval_moments"] for snapshot in snapshots]
        ),
        # RatioScalar metrics: both parts are SUMMED across ranks (like the moments above), the quotient is taken once
        # by rank 0 in post_epoch_logging.  ``get``: snapshots from a helper that predates the field carry none.
        "ratio_moments": merge_ratio_moments(
            [snapshot.get("ratio_moments", {}) for snapshot in snapshots]
        ),
        "has_logging_done_override": any(
            snapshot["has_logging_done_override"] for snapshot in snapshots
        ),
    }


class LoggingHelper:
    def __init__(
        self,
        writer: SummaryWriter,
        log_dir: str | pathlib.Path,
        num_envs: int,
        num_steps_per_env: int,
        num_learning_iterations: int,
        device: str = "cpu",
        prefix: str = "",
        title: str = "Training Log",
        is_main_process: bool = True,
        num_gpus: int = 1,
    ):
        """Initialize the logging helper.

        Parameters
        ----------
        writer : SummaryWriter
            TensorBoard writer for logging metrics
        log_dir : str
            Directory to store logs
        num_envs : int
            Number of environments to track
        num_steps_per_env : int
            Number of steps per environment between each call to `post_epoch_logging`.
        num_learning_iterations : int
            Number of total learning iterations.
        device : str, optional
            Device to use for tensors, by default "cpu"
        prefix : str, optional
            Prefix to add to all the logging keys.
        title : str, optional
            Title of the logging panel.
        is_main_process : bool, optional
            Whether this is the main process.
        num_gpus : int, optional
            Number of GPUs to use.
        """
        self.writer: SummaryWriter = writer
        self.log_dir: str = str(log_dir)
        self.device: str = device
        self.tot_timesteps: int = 0
        self.tot_time: float = 0.0
        self.collection_time: float = 0.0
        self.learn_time: float = 0.0
        self.num_envs: int = num_envs
        self.num_steps_per_env: int = num_steps_per_env
        self.num_learning_iterations: int = num_learning_iterations
        # The console ETA needs the global iteration
        # range of THIS run -- ``tot_time`` only spans this run, and ``it`` in ``post_epoch_logging`` is global while the
        # agents disagree on what ``num_learning_iterations`` (N) means.  Callers that know their loop bounds pin them with
        # ``set_run_iteration_range``; the fields stay None until then and the FIRST ``post_epoch_logging`` call infers
        # them from the logging CADENCE, i.e. from how many ``record_collection_time`` windows preceded it:
        #   * exactly one window (or none) -> a per-iteration logger: ``PPO.learn``
        #     loops ``range(start, start + N)`` (``start`` = 0 or the resumed checkpoint's ``iter + 1``), times one
        #     collection window per iteration and logs every iteration on rank 0, so the first logged ``it`` IS the
        #     loop's first iteration and N is the count this run APPENDS: range ``(it, it + N)``, header end ``it + N``;
        #   * two or more windows -> a periodic logger: FastSAC / FlashSAC time one window per env step of ``while
        #     global_step <= N`` (N = absolute INCLUSIVE end) and log only at multiples of ``logging_interval`` once
        #     ``global_step > learning_starts``, so the loop entry is ``it - (windows - 1)`` (0 fresh, checkpoint + 1
        #     resumed) and the range is ``(entry, N + 1)`` with header end N.
        # Explicit values always win over the inference.  See ``set_run_iteration_range`` / ``_create_console_output``.
        self._run_start_iteration: int | None = None
        self._run_end_iteration: int | None = None
        self._run_header_end: int | None = None
        # collection windows recorded since construction (never cleared per interval): the cadence signal above
        self._collection_windows: int = 0
        self.prefix: str = prefix
        self.title: str = title
        self.is_main_process: bool = is_main_process
        self.num_gpus: int = num_gpus

        # Book keeping
        self.ep_infos: list[dict[str, Any]] = []
        self.raw_ep_infos: list[dict[str, Any]] = []
        self.rewbuffer: deque[float] = deque(maxlen=100)
        self.native_rewbuffer: deque[float] = deque(maxlen=100)
        self.lenbuffer: deque[float] = deque(maxlen=100)
        # Some task adapters must terminate replay at a state discontinuity
        # (for example a MotionCommand clip teleport) without ending the
        # native HoloSoma hard-reset interval.  Keep the two statistics
        # separate and name each one by its actual event-boundary semantics.
        self.replay_segment_lenbuffer: deque[float] = deque(maxlen=100)
        # The console deques above only need host values once per logging
        # interval.  Completed-episode tensors are staged on-device here and
        # flushed with a single transfer in post_epoch_logging, instead of a
        # GPU->host sync on every step that contains a done.
        # perf_counter at the previous post_epoch_logging call; None on the first
        # one, where there is no interval to measure yet.
        self._last_log_wall_time: float | None = None
        self._pending_native_returns: list[torch.Tensor] = []
        self._pending_lengths: list[torch.Tensor] = []
        self._pending_replay_returns: list[torch.Tensor] = []
        self._pending_segments: list[torch.Tensor] = []
        self._has_logging_done_override = False
        # Canonical W&B/TensorBoard means use every completion in the current
        # logging interval.  The legacy maxlen=100 deques above remain only for
        # compact console/compatibility state; using them for metrics biases a
        # 4096-env vector step toward the final 100 environment indices.
        self._native_episode_length_interval_sum = torch.zeros(
            (), dtype=torch.float, device=self.device
        )
        self._native_episode_length_interval_count = 0
        self._replay_segment_length_interval_sum = torch.zeros(
            (), dtype=torch.float, device=self.device
        )
        self._replay_segment_interval_count = 0
        self._native_return_interval_sum = torch.zeros(
            (), dtype=torch.float, device=self.device
        )
        self._native_return_interval_count = 0
        self._replay_return_interval_sum = torch.zeros(
            (), dtype=torch.float, device=self.device
        )
        self._replay_return_interval_count = 0
        self.cur_reward_sum: torch.Tensor = torch.zeros(num_envs, dtype=torch.float, device=self.device)
        self.cur_native_reward_sum: torch.Tensor = torch.zeros(
            num_envs, dtype=torch.float, device=self.device
        )
        self.cur_episode_length: torch.Tensor = torch.zeros(num_envs, dtype=torch.float, device=self.device)
        self.cur_replay_segment_length: torch.Tensor = torch.zeros(
            num_envs, dtype=torch.float, device=self.device
        )
        self.episode_env_tensors: TensorAverageMeterDict = TensorAverageMeterDict()
        # Populated only at a distributed logging boundary. These compact
        # moments let rank 0 retain the existing post_epoch_logging path while
        # reporting all-rank data instead of rank-0-local data.
        self._synced_episode_moments: dict[str, tuple[float, int]] | None = None
        self._synced_raw_episode_moments: dict[str, tuple[float, int]] | None = None
        self._synced_env_moments: dict[str, tuple[float, int]] | None = None
        # RatioScalar metrics: per key a float64 ``[sum of
        # numerators, sum of denominators]`` on the training device, accumulated by update_episode_stats over the
        # interval; rank 0 receives the all-rank sums in ``_synced_ratio_moments`` at a distributed boundary.
        self._ratio_interval_sums: dict[str, torch.Tensor] = {}
        self._synced_ratio_moments: dict[str, tuple[float, float]] | None = None

    def set_run_iteration_range(
        self, first_iteration: int, end_iteration: int, *, header_end: int | None = None
    ) -> None:
        """Pin the GLOBAL iteration range this run's learn loop covers, for the console ETA.

        ``first_iteration`` is the loop's
        first iteration value (the one whose collection / learn time is the first to enter ``tot_time``: 0 on a fresh
        run, the resumed checkpoint's iteration + 1 otherwise) and ``end_iteration`` the EXCLUSIVE bound (one past the
        last iteration the loop runs): after iteration ``it`` completed the remaining work is ``end_iteration - (it + 1)``
        -- 0 on the run's last iteration.  The console header reads ``it/header_end``, ``header_end`` defaulting to
        ``end_iteration``; a loop with an INCLUSIVE bound passes its bound here to keep the stock ``it/N`` header.
        ``PPO.learn`` (``range(start, start + num_learning_iterations)``) would pass ``(start, start + N)``; the FastSAC /
        FlashSAC ``while global_step <= N`` loops would pass ``(global_step at loop entry, N + 1, header_end=N)``.  No
        agent is wired yet: without this call the first ``post_epoch_logging`` infers the range from the logging cadence
        (see ``__init__``), which reproduces both families' conventions.  Explicit values are the source of truth: the
        inference never overwrites them, and this call replaces an inferred range.
        """
        first_iteration = int(first_iteration)
        end_iteration = int(end_iteration)
        if first_iteration < 0 or end_iteration < first_iteration:
            raise ValueError(
                f"run iteration range must satisfy 0 <= first_iteration <= end_iteration, got "
                f"({first_iteration}, {end_iteration})"
            )
        self._run_start_iteration = first_iteration
        self._run_end_iteration = end_iteration
        self._run_header_end = end_iteration if header_end is None else int(header_end)

    def _infer_run_iteration_range(self, it: int) -> None:
        """First ``post_epoch_logging`` without an explicit range: pick the convention from the logging cadence.

        One collection window (or none: direct callers) per log call is the PPO family (N appended to the first logged
        iteration); several windows before the first log is the periodic FastSAC / FlashSAC family (N = absolute
        inclusive end, loop entry = ``it - (windows - 1)``).  See ``__init__``.  Only ever called once per helper.
        """
        it = int(it)
        windows = self._collection_windows
        if windows <= 1:
            self._run_start_iteration = it
            self._run_end_iteration = it + self.num_learning_iterations
            self._run_header_end = self._run_end_iteration
        else:
            self._run_start_iteration = max(it - (windows - 1), 0)
            self._run_end_iteration = self.num_learning_iterations + 1
            self._run_header_end = self.num_learning_iterations

    @contextmanager
    def record_collection_time(self) -> Generator[None, None, None]:
        """Record the time taken for collection."""
        start_time = time.perf_counter()
        yield
        self.collection_time += time.perf_counter() - start_time
        self._collection_windows += 1  # cadence signal for the run-range inference (see __init__)

    @contextmanager
    def record_learn_time(self) -> Generator[None, None, None]:
        """Record the time taken for learning."""
        start_time = time.perf_counter()
        yield
        self.learn_time += time.perf_counter() - start_time

    def update_episode_stats(self, rewards: torch.Tensor, dones: torch.Tensor, infos: dict[str, Any]) -> None:
        """Update episode statistics.

        Parameters
        ----------
        rewards : torch.Tensor
            Rewards from the environment
        dones : torch.Tensor
            Done flags from the environment
        infos : dict[str, Any]
            Additional info from the environment
        """
        replay_dones = torch.as_tensor(dones, device=self.device)
        if replay_dones.shape != (self.num_envs,):
            raise ValueError(
                "replay dones must have shape (num_envs,): "
                f"{tuple(replay_dones.shape)} != {(self.num_envs,)}"
            )
        logging_dones_raw = infos.get("logging_dones")
        if logging_dones_raw is None:
            logging_dones = replay_dones
        else:
            logging_dones = torch.as_tensor(logging_dones_raw, device=self.device)
            if logging_dones.shape != replay_dones.shape:
                raise ValueError(
                    "logging_dones must match replay dones: "
                    f"{tuple(logging_dones.shape)} != {tuple(replay_dones.shape)}"
                )
            if bool((logging_dones.bool() & ~replay_dones.bool()).any()):
                raise ValueError(
                    "every native logging done must also terminate replay"
                )
            self._has_logging_done_override = True

        rewards_tensor = torch.as_tensor(rewards, dtype=torch.float, device=self.device)
        if rewards_tensor.shape != (self.num_envs,):
            raise ValueError(
                "rewards must have shape (num_envs,): "
                f"{tuple(rewards_tensor.shape)} != {(self.num_envs,)}"
            )

        # Validate the boundary contract before mutating any logger state.  In
        # particular, this keeps a rejected logging_dones mask from appending
        # partial episode information or changing running returns.
        self.ep_infos.append(infos["episode"])
        if "raw_episode" in infos:
            self.raw_ep_infos.append(infos["raw_episode"])

        self.cur_reward_sum += rewards_tensor
        self.cur_native_reward_sum += rewards_tensor
        self.cur_episode_length += 1
        self.cur_replay_segment_length += 1

        new_ids = (logging_dones > 0).nonzero(as_tuple=True)[0]
        if new_ids.numel() > 0:
            completed_lengths = self.cur_episode_length[new_ids]
            completed_native_returns = self.cur_native_reward_sum[new_ids]
            if self.is_main_process:
                # These bounded deques feed only the rank-zero console.  Stage
                # on-device and flush once per logging interval to avoid a
                # device-to-host synchronization on every step with a done.
                self._pending_native_returns.append(completed_native_returns.detach())
                self._pending_lengths.append(completed_lengths.detach())
            self._native_episode_length_interval_sum += completed_lengths.sum()
            self._native_episode_length_interval_count += int(completed_lengths.numel())
            self._native_return_interval_sum += completed_native_returns.sum()
            self._native_return_interval_count += int(completed_native_returns.numel())
            self.cur_native_reward_sum[new_ids] = 0
            self.cur_episode_length[new_ids] = 0

        segment_ids = (replay_dones > 0).nonzero(as_tuple=True)[0]
        if segment_ids.numel() > 0:
            completed_segments = self.cur_replay_segment_length[segment_ids]
            completed_replay_returns = self.cur_reward_sum[segment_ids]
            if self.is_main_process:
                self._pending_replay_returns.append(completed_replay_returns.detach())
                self._pending_segments.append(completed_segments.detach())
            self._replay_segment_length_interval_sum += completed_segments.sum()
            self._replay_segment_interval_count += int(completed_segments.numel())
            self._replay_return_interval_sum += completed_replay_returns.sum()
            self._replay_return_interval_count += int(completed_replay_returns.numel())
            self.cur_reward_sum[segment_ids] = 0
            self.cur_replay_segment_length[segment_ids] = 0

        # Update episode environment tensors.  RatioScalar entries never enter the mean meter: their numerators and
        # denominators are summed separately for pooled conditional means.
        plain_to_log, ratio_to_log = _split_ratio_metrics(infos["to_log"])
        self.episode_env_tensors.add(plain_to_log)
        self._accumulate_ratio_metrics(ratio_to_log)

    def _accumulate_ratio_metrics(self, ratios: dict[str, RatioScalar]) -> None:
        for key, ratio in ratios.items():
            accumulator = self._ratio_interval_sums.get(key)
            if accumulator is None:
                accumulator = torch.zeros(2, dtype=torch.float64, device=self.device)
                self._ratio_interval_sums[key] = accumulator
            accumulator += ratio.stats().to(device=accumulator.device)

    def _local_ratio_moments(self) -> dict[str, tuple[float, float]]:
        """This rank's interval ``{name: (sum numerator, sum denominator)}`` as host floats (one sync per key)."""
        moments: dict[str, tuple[float, float]] = {}
        for key, accumulator in self._ratio_interval_sums.items():
            numerator, denominator = accumulator.tolist()
            moments[key] = (float(numerator), float(denominator))
        return moments

    def _distributed_logging_snapshot(
        self, loss_dict: dict[str, float]
    ) -> _DistributedLoggingSnapshot:
        """Detach this rank's interval state into compact sufficient statistics."""

        env_moments: dict[str, tuple[float, int]] = {}
        for key, meter in self.episode_env_tensors.data.items():
            summarized = _summarize_mapping_history(
                [{key: value} for value in meter.tensors]
            )
            if key in summarized:
                env_moments[key] = summarized[key]
        interval_moments = {
            "native_episode_length": (
                float(self._native_episode_length_interval_sum.item()),
                self._native_episode_length_interval_count,
            ),
            "replay_segment_length": (
                float(self._replay_segment_length_interval_sum.item()),
                self._replay_segment_interval_count,
            ),
            "native_return": (
                float(self._native_return_interval_sum.item()),
                self._native_return_interval_count,
            ),
            "replay_return": (
                float(self._replay_return_interval_sum.item()),
                self._replay_return_interval_count,
            ),
        }
        detached_losses: dict[str, float] = {}
        for key, value in loss_dict.items():
            if isinstance(value, torch.Tensor):
                if value.numel() != 1:
                    raise ValueError(
                        f"loss_dict[{key!r}] must be scalar, got shape {tuple(value.shape)}"
                    )
                detached_losses[key] = float(value.detach().item())
            else:
                detached_losses[key] = float(value)

        return {
            "loss_dict": detached_losses,
            "episode_moments": _summarize_mapping_history(self.ep_infos),
            "raw_episode_moments": _summarize_mapping_history(self.raw_ep_infos),
            "env_moments": env_moments,
            "interval_moments": interval_moments,
            "ratio_moments": self._local_ratio_moments(),
            "has_logging_done_override": self._has_logging_done_override,
        }

    def _clear_interval_payload(self, *, clear_timing: bool) -> None:
        """Clear completed interval data while preserving in-flight episodes."""

        self.ep_infos.clear()
        self.raw_ep_infos.clear()
        self.episode_env_tensors.clear()
        self._native_episode_length_interval_sum.zero_()
        self._native_episode_length_interval_count = 0
        self._replay_segment_length_interval_sum.zero_()
        self._replay_segment_interval_count = 0
        self._native_return_interval_sum.zero_()
        self._native_return_interval_count = 0
        self._replay_return_interval_sum.zero_()
        self._replay_return_interval_count = 0
        self._synced_episode_moments = None
        self._synced_raw_episode_moments = None
        self._synced_env_moments = None
        self._ratio_interval_sums.clear()
        self._synced_ratio_moments = None
        if clear_timing:
            self.learn_time = 0.0
            self.collection_time = 0.0

    @torch.no_grad()
    def synchronize_distributed_interval(
        self, loss_dict: dict[str, float]
    ) -> dict[str, float]:
        """Aggregate one PPO logging interval across the process group.

        Every rank must call this method exactly once per interval. Only
        detached Python sufficient statistics participate in the collective;
        no model tensors, optimizer state, or autograd graph are touched.
        Rank 0 receives globally weighted episode/environment moments for its
        normal ``post_epoch_logging`` call. Other ranks discard only completed
        logging payloads and retain their in-flight episode accumulators.
        """

        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            if self.num_gpus != 1:
                raise RuntimeError(
                    "distributed logging requested for multiple GPUs without an "
                    "initialized torch.distributed process group"
                )
            return self._distributed_logging_snapshot(loss_dict)["loss_dict"]

        world_size = torch.distributed.get_world_size()
        if world_size != self.num_gpus:
            raise RuntimeError(
                "LoggingHelper num_gpus disagrees with the process group: "
                f"{self.num_gpus} != {world_size}"
            )

        local_snapshot = self._distributed_logging_snapshot(loss_dict)
        gathered: list[_DistributedLoggingSnapshot | None] = [None] * world_size
        torch.distributed.all_gather_object(gathered, local_snapshot)
        if any(snapshot is None for snapshot in gathered):
            raise RuntimeError("distributed logging gather returned an empty rank payload")
        merged = _merge_distributed_logging_snapshots(
            [snapshot for snapshot in gathered if snapshot is not None]
        )

        # Non-main ranks do not call post_epoch_logging, so consume their
        # interval here. Rank 0 first replaces its local payload with the global
        # sufficient statistics and then lets the existing writer path consume it.
        self._clear_interval_payload(clear_timing=not self.is_main_process)
        if self.is_main_process:
            self._synced_episode_moments = merged["episode_moments"]
            self._synced_raw_episode_moments = merged["raw_episode_moments"]
            self._synced_env_moments = merged["env_moments"]
            self._synced_ratio_moments = merged["ratio_moments"]
            interval = merged["interval_moments"]
            self._native_episode_length_interval_sum.fill_(
                interval["native_episode_length"][0]
            )
            self._native_episode_length_interval_count = interval[
                "native_episode_length"
            ][1]
            self._replay_segment_length_interval_sum.fill_(
                interval["replay_segment_length"][0]
            )
            self._replay_segment_interval_count = interval[
                "replay_segment_length"
            ][1]
            self._native_return_interval_sum.fill_(interval["native_return"][0])
            self._native_return_interval_count = interval["native_return"][1]
            self._replay_return_interval_sum.fill_(interval["replay_return"][0])
            self._replay_return_interval_count = interval["replay_return"][1]
            self._has_logging_done_override = merged[
                "has_logging_done_override"
            ]

        return merged["loss_dict"]

    def _flush_pending_console_stats(self) -> None:
        """Move staged completed-episode tensors into the console deques.

        One host transfer per buffer per logging interval, instead of one per
        step that contained a done (see update_episode_stats).
        """
        for pending, buffer in (
            (self._pending_native_returns, self.native_rewbuffer),
            (self._pending_lengths, self.lenbuffer),
            (self._pending_replay_returns, self.rewbuffer),
            (self._pending_segments, self.replay_segment_lenbuffer),
        ):
            if pending:
                buffer.extend(torch.cat(pending).cpu().numpy().tolist())
                pending.clear()

    def post_epoch_logging(
        self,
        it: int,
        loss_dict: dict[str, float],
        extra_log_dicts: dict[str, dict[str, float]],
        width: int = 80,
        pad: int = 35,
    ) -> None:
        """Handle post-epoch logging for training metrics.

        This method handles all logging operations after each training epoch, including:
        - Updating total timesteps and time
        - Logging episode information
        - Writing metrics to TensorBoard
        - Creating and displaying console output
        - Clearing episode information after logging

        Parameters
        ----------
        it : int
            Current iteration number
        loss_dict : dict[str, float]
            Dictionary containing loss values
        extra_log_dicts : dict[str, dict[str, float]]
            Dictionary containing extra metrics to log: {section_name: {metric_name: metric_value}}
        width : int, optional
            Width of the console output, by default 80
        pad : int, optional
            Padding for aligned console output, by default 35
        """
        if self._run_start_iteration is None:
            # No explicit ``set_run_iteration_range``: infer the run range from the logging cadence (see ``__init__``).
            self._infer_run_iteration_range(it)
        self.tot_timesteps += self.num_steps_per_env * self.num_envs * self.num_gpus
        self.tot_time += self.collection_time + self.learn_time
        iteration_time = self.collection_time + self.learn_time

        self._flush_pending_console_stats()

        # Log episode info
        ep_string, ep_scalars_to_log = self._log_episode_info()

        if self._synced_env_moments is None:
            env_log_dict = self.episode_env_tensors.mean_and_clear()
            ratio_moments = self._local_ratio_moments()
        else:
            env_log_dict = {
                key: value_sum / count
                for key, (value_sum, count) in self._synced_env_moments.items()
                if count > 0
            }
            self._synced_env_moments = None
            ratio_moments = self._synced_ratio_moments or {}
        self._ratio_interval_sums.clear()
        self._synced_ratio_moments = None
        # RatioScalar metrics use the quotient of interval-and-rank sums.
        # If a key is also published as a plain scalar, the pooled ratio wins.
        ratio_log_dict = _resolve_ratio_metrics(ratio_moments)
        for key in ratio_log_dict.keys() & env_log_dict.keys():
            logger.warning("env metric {!r} was published both as a plain scalar and as a RatioScalar; the ratio is logged", key)
        env_log_dict.update(ratio_log_dict)
        # Keep A_MAJOR metrics in their own top-level wandb section.
        env_log_dict = {(k if k.startswith("A_MAJOR/") else f"Env/{k}"): v for k, v in env_log_dict.items()}

        frames = self.num_steps_per_env * self.num_envs * self.num_gpus
        fps = int(frames / (self.collection_time + self.learn_time + 1e-8))

        # Wall-clock throughput alongside the compute-only figure above. Everything
        # between two log calls that is neither collection nor learning —
        # checkpointing, video rendering, eval, S3 sync, straggling ranks — shows up
        # as a gap between the two. `Perf/total_fps` deliberately keeps its
        # compute-only meaning so it stays comparable with historical runs.
        _now = time.perf_counter()
        _previous = self._last_log_wall_time
        self._last_log_wall_time = _now
        wall_time = (self.collection_time + self.learn_time) if _previous is None else max(_now - _previous, 1e-6)
        # Merged into the caller's extras rather than threaded through every
        # writer/console signature; `Perf` is the section the writer already uses.
        extra_log_dicts = {
            **extra_log_dicts,
            "Perf": {
                **extra_log_dicts.get("Perf", {}),
                "wall_fps": int(frames / (wall_time + 1e-8)),
                "wall_time": wall_time,
            },
        }

        # Log to tensorboard
        self._logging_to_writer(
            it=it,
            loss_dict=loss_dict,
            extra_log_dicts=extra_log_dicts,
            env_log_dict=env_log_dict,
            fps=fps,
            ep_scalars_to_log=ep_scalars_to_log,
        )

        # Create console output
        log_string = self._create_console_output(
            it=it,
            loss_dict=loss_dict,
            env_log_dict=env_log_dict,
            extra_log_dicts=extra_log_dicts,
            ep_string=ep_string,
            width=width,
            pad=pad,
            iteration_time=iteration_time,
            fps=fps,
        )

        # Use rich Live to update console
        with Live(Panel(log_string, title=self.title), refresh_per_second=4, console=console):
            pass

        # Clear completed interval data, but keep per-environment accumulators
        # for episodes that span this logging boundary.
        self._clear_interval_payload(clear_timing=True)

    def _log_episode_info(self) -> tuple[str, dict[str, float]]:
        """Log episode information and return formatted string.

        Parameters
        ----------
        it : int
            Current iteration number

        Returns
        -------
        str
            Formatted string containing episode statistics
        """
        if not self.is_main_process:
            return "", {}
        ep_string = ""
        scalars_to_log: dict[str, float] = {}

        # Process regular episode info. Distributed PPO installs all-rank
        # sum/count moments; single-process and legacy callers retain the
        # original local tensor path.
        if self._synced_episode_moments is not None:
            for key, (value_sum, count) in self._synced_episode_moments.items():
                if count <= 0:
                    continue
                value = value_sum / count
                scalars_to_log[f"Episode/{key}"] = value
                ep_string += f"""{f"Mean episode {key}:":>35} {value:.4f}\n"""
            self._synced_episode_moments = None
        elif self.ep_infos:
            for key in self.ep_infos[0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in self.ep_infos:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                if len(infotensor) == 0:
                    continue
                value = torch.mean(infotensor).item()
                scalars_to_log[f"Episode/{key}"] = value
                ep_string += f"""{f"Mean episode {key}:":>35} {value:.4f}\n"""

        # Process raw episode info if it exists
        if self._synced_raw_episode_moments is not None:
            for key, (value_sum, count) in self._synced_raw_episode_moments.items():
                if count <= 0:
                    continue
                value = value_sum / count
                scalars_to_log[f"RawEpisode/{key}"] = value
                ep_string += f"""{f"Mean raw episode {key}:":>35} {value:.4f}\n"""
            self._synced_raw_episode_moments = None
        elif self.raw_ep_infos:
            for key in self.raw_ep_infos[0]:
                infotensor = torch.tensor([], device=self.device)
                for ep_info in self.raw_ep_infos:
                    if not isinstance(ep_info[key], torch.Tensor):
                        ep_info[key] = torch.Tensor([ep_info[key]])
                    if len(ep_info[key].shape) == 0:
                        ep_info[key] = ep_info[key].unsqueeze(0)
                    infotensor = torch.cat((infotensor, ep_info[key].to(self.device)))
                if len(infotensor) == 0:
                    continue
                value = torch.mean(infotensor).item()
                scalars_to_log[f"RawEpisode/{key}"] = value
                ep_string += f"""{f"Mean raw episode {key}:":>35} {value:.4f}\n"""

        return ep_string, scalars_to_log

    def _logging_to_writer(
        self,
        it: int,
        loss_dict: dict[str, float],
        env_log_dict: dict[str, float],
        extra_log_dicts: dict[str, dict[str, float]],
        fps: int,
        ep_scalars_to_log: dict[str, float],
    ) -> None:
        """Log metrics to tensorboard writer.

        Parameters
        ----------
        it : int
            Current iteration number
        loss_dict : dict[str, float]
            Dictionary containing loss metrics
        env_log_dict : dict[str, float]
            Dictionary containing environment metrics
        extra_log_dicts : dict[str, float]
            Dictionary containing extra metrics to log: {section_name: {metric_name: metric_value}}
        fps : int
            Frames per second (training speed).
        ep_scalars_to_log : dict[str, float]
            Dictionary containing episode metrics to log.
        """
        if not self.is_main_process:
            return
        # Log loss metrics
        scalars_to_log: dict[str, float] = {}
        for loss_key, loss_value in loss_dict.items():
            scalars_to_log[f"Loss/{loss_key}"] = loss_value

        scalars_to_log.update(env_log_dict)
        scalars_to_log.update(ep_scalars_to_log)

        # Log extra metrics
        for section_name, section_dict in extra_log_dicts.items():
            for key, value in section_dict.items():
                scalars_to_log[f"{section_name}/{key}"] = value

        # Log performance metrics
        scalars_to_log["Perf/total_fps"] = fps
        scalars_to_log["Perf/collection_time"] = self.collection_time
        scalars_to_log["Perf/learning_time"] = self.learn_time

        # ``Train/mean_*`` remains the upstream compatibility surface, but is
        # now an unbiased all-completions mean for this logging interval.  When
        # adapters provide distinct logging_dones, explicit metric names make
        # the native hard-reset interval and replay segment semantics visible.
        if self._replay_return_interval_count > 0:
            replay_return_mean = float(
                self._replay_return_interval_sum.item()
                / self._replay_return_interval_count
            )
            scalars_to_log["Train/mean_reward"] = replay_return_mean
            scalars_to_log["Train/mean_reward/time"] = replay_return_mean
            if self._has_logging_done_override:
                scalars_to_log["Train/replay_segment_return"] = replay_return_mean
        if self._has_logging_done_override and self._native_return_interval_count > 0:
            scalars_to_log["Train/native_episode_return"] = float(
                self._native_return_interval_sum.item()
                / self._native_return_interval_count
            )
        if self._native_episode_length_interval_count > 0:
            native_length_mean = float(
                self._native_episode_length_interval_sum.item()
                / self._native_episode_length_interval_count
            )
            scalars_to_log["Train/mean_episode_length"] = native_length_mean
            scalars_to_log["Train/mean_episode_length/time"] = native_length_mean
            scalars_to_log["Train/native_episode_length_steps"] = native_length_mean
        if self._has_logging_done_override and self._replay_segment_interval_count > 0:
            scalars_to_log["Train/replay_segment_length_steps"] = float(
                self._replay_segment_length_interval_sum.item()
                / self._replay_segment_interval_count
            )
        self._native_episode_length_interval_sum.zero_()
        self._native_episode_length_interval_count = 0
        self._replay_segment_length_interval_sum.zero_()
        self._replay_segment_interval_count = 0
        self._native_return_interval_sum.zero_()
        self._native_return_interval_count = 0
        self._replay_return_interval_sum.zero_()
        self._replay_return_interval_count = 0

        scalars_to_log["Train/num_samples"] = self.tot_timesteps

        # Add prefix to all keys
        scalars_to_log = {f"{self.prefix}{k}": v for k, v in scalars_to_log.items()}

        for k, v in scalars_to_log.items():
            self.writer.add_scalar(k, v, global_step=it)
        if wandb.run is not None:
            wandb.log(dict(scalars_to_log, global_step=it), step=it)

    def _create_console_output(
        self,
        it: int,
        loss_dict: dict[str, float],
        env_log_dict: dict[str, float],
        extra_log_dicts: dict[str, dict[str, float]],
        ep_string: str,
        width: int,
        pad: int,
        iteration_time: float,
        fps: int,
    ) -> str:
        """Create formatted console output string.

        Parameters
        ----------
        it : int
            Current iteration number
        loss_dict : dict[str, float]
            Dictionary containing loss metrics
        env_log_dict : dict[str, float]
            Dictionary containing environment metrics
        extra_log_dicts : dict[str, dict[str, float]]
            Dictionary containing extra metrics to log: {section_name: {metric_name: metric_value}}
        ep_string : str
            Formatted string containing episode statistics
        width : int
            Width of the console output
        pad : int
            Padding for aligned console output
        iteration_time : float
            Time taken for the current iteration
        fps : int
            Frames per second (training speed).

        Returns
        -------
        str
            Formatted string for console output
        """
        if not self.is_main_process:
            return ""
        # Estimate pace from time and iterations completed in this run. After
        # iteration ``it``, ``run_end - (it + 1)`` iterations remain.
        # The range comes from ``set_run_iteration_range`` when the caller pinned it (explicit values win); otherwise it
        # was inferred from the logging cadence at the first ``post_epoch_logging`` (PPO family: ``(first it, first it +
        # N)``; periodic FastSAC / FlashSAC: ``(loop entry, N + 1)`` with header end N, i.e. the stock header / ETA on a
        # fresh SAC run; see ``__init__``).  The header shows the GLOBAL iteration over the loop bound as the agent's
        # config expresses it (``29140/36000`` on that resumed PPO run, not ``29140/7000``; ``60000/100000`` on SAC).
        # getattr: duck-typed formatter fixtures predate the fields.
        run_start = getattr(self, "_run_start_iteration", None)
        if run_start is None:
            run_start = it
        run_end = getattr(self, "_run_end_iteration", None)
        if run_end is None:
            run_end = run_start + self.num_learning_iterations
        header_end = getattr(self, "_run_header_end", None)
        if header_end is None:
            header_end = run_end
        completed_this_run = max(it - run_start + 1, 1)
        remaining_iterations = max(run_end - (it + 1), 0)
        eta = self.tot_time / completed_this_run * remaining_iterations
        header = f" \033[1m Learning iteration {it}/{header_end} \033[0m "

        # Base log string with computation info
        log_string = (
            f"""{header.center(width, " ")}\n\n"""
            f"""{"Computation:":>{pad}} {fps:.0f} steps/s """
            f"""(Collection: {self.collection_time:.3f}s, Learning {self.learn_time:.3f}s)\n"""
        )

        # Add training metrics if available
        if len(self.rewbuffer) > 0:
            log_string += f"""{"Mean reward:":>{pad}} {statistics.mean(self.rewbuffer):.2f}\n"""
        if len(self.lenbuffer) > 0:
            log_string += f"""{"Mean episode length:":>{pad}} {statistics.mean(self.lenbuffer):.2f}\n"""

        # Add loss metrics
        for key, value in loss_dict.items():
            log_string += f"{f'{key}:':>{pad}} {value:.4f}\n"

        # Add environment metrics
        env_log_string = ""
        for k, v in env_log_dict.items():
            entry = f"{f'{k}:':>{pad}} {v:.4f}"
            env_log_string += f"{entry}\n"
        log_string += env_log_string

        # Add extra metrics
        for section_name, section_dict in extra_log_dicts.items():
            for key, value in section_dict.items():
                log_string += f"{f'{section_name}/{key}:':>{pad}} {value:.4f}\n"

        # Add episode info
        log_string += ep_string

        # Add timing info (``eta`` computed with the run range above: 0 once the last iteration of the run completed)
        log_string += (
            f"""{"-" * width}\n"""
            f"""{"Total timesteps:":>{pad}} {self.tot_timesteps}\n"""
            f"""{"Iteration time:":>{pad}} {iteration_time:.2f}s\n"""
            f"""{"Total time:":>{pad}} {self.tot_time:.2f}s\n"""
            f"""{"ETA:":>{pad}} {eta:.1f}s\n"""
        )
        log_string += f"Logging Directory: {self.log_dir}"

        return log_string

    def save_checkpoint_artifact(self, state_dict: dict[str, Any], path: str) -> None:
        """Serialize ``state_dict`` and publish it at ``path`` atomically.

        Write to ``<path>.<pid>.tmp`` in the same directory, fsync, then atomically
        replace ``path``. A failed write removes the temporary file and preserves
        the previous checkpoint. Uploads and ONNX export follow the completed write.
        """
        if not path.startswith(self.log_dir):
            raise ValueError(f"Path {path} is not in the logging directory {self.log_dir}")
        os.makedirs(os.path.dirname(path), exist_ok=True)
        logger.info(f"Saving checkpoint to {path}")
        tmp_path = f"{path}.{os.getpid()}.tmp"
        try:
            torch.save(state_dict, tmp_path)
            fd = os.open(tmp_path, os.O_RDONLY)
            try:
                os.fsync(fd)
            finally:
                os.close(fd)
            os.replace(tmp_path, path)
        except BaseException:
            try:
                os.unlink(tmp_path)
            except OSError:
                pass
            raise
        self.save_to_wandb(path)

    def save_to_wandb(self, file_path: str) -> None:
        """Saves file to wandb if run is initialized."""
        if wandb.run is None:
            return
        wandb.save(file_path, base_path=self.log_dir)
