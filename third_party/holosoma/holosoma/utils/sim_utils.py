"""Shared simulation utilities for holosoma.

This module provides common functionality for setting up and running simulations,
shared between eval_agent.py and run_sim.py.
"""

from __future__ import annotations

import argparse
import os
import sys
import threading
import time
import traceback
from typing import Any

from loguru import logger
from typing_extensions import Self

from holosoma.config_types.env import get_tyro_env_config
from holosoma.config_types.experiment import ExperimentConfig
from holosoma.config_types.full_sim import FullSimConfig
from holosoma.config_types.run_sim import RunSimConfig
from holosoma.managers.terrain.manager import TerrainManager
from holosoma.utils.common import seeding
from holosoma.utils.helpers import get_class
from holosoma.utils.rate import RateLimiter
from holosoma.utils.safe_torch_import import torch
from holosoma.utils.simulator_config import SimulatorType, get_simulator_type, set_simulator_type
from holosoma.utils.torch_utils import to_torch


LOCAL_RANK_GPU_OFFSET_ENV = "HOLOSOMA_LOCAL_RANK_GPU_OFFSET"
USD_CONVERSION_SLOT_ENV = "HOLOSOMA_USD_CONVERSION_SLOT"


def local_rank_gpu_offset() -> int:
    """Physical CUDA index of LOCAL_RANK 0 (``HOLOSOMA_LOCAL_RANK_GPU_OFFSET``, default 0).

    Isaac Sim / Kit does NOT honour ``CUDA_VISIBLE_DEVICES``: IsaacLab's ``AppLauncher``
    turns ``cuda:N`` into ``--/renderer/activeGpu=N`` and ``--/physics/cudaDevice=N``,
    and Kit resolves those against every GPU in the machine, while torch would have
    remapped ``N`` through ``CUDA_VISIBLE_DEVICES``.  A torchrun job restricted to
    GPUs 4-7 via ``CUDA_VISIBLE_DEVICES=4,5,6,7`` therefore pins torch to physical
    GPU 4+i but asks Kit for GPU i ("activeGpu i is not compatible with current
    foundation settings", PhysX "GPU solver pipeline failed, switching to software").

    The fix is to leave all GPUs visible and map ``LOCAL_RANK -> LOCAL_RANK + offset``
    everywhere (torch device, AppLauncher device, USD conversion slot).  A single
    (non-torchrun) process honours the offset as ``cuda:<offset>``.
    """
    raw = os.environ.get(LOCAL_RANK_GPU_OFFSET_ENV, "0").strip() or "0"
    try:
        offset = int(raw)
    except ValueError as exc:
        raise ValueError(f"{LOCAL_RANK_GPU_OFFSET_ENV} must be a non-negative integer, got {raw!r}.") from exc
    if offset < 0:
        raise ValueError(f"{LOCAL_RANK_GPU_OFFSET_ENV} must be a non-negative integer, got {offset}.")
    return offset


def local_rank_gpu_index(local_rank: int) -> int:
    """Physical CUDA device index of *local_rank* (``LOCAL_RANK + HOLOSOMA_LOCAL_RANK_GPU_OFFSET``)."""
    return int(local_rank) + local_rank_gpu_offset()


def _offset_note() -> str:
    offset = local_rank_gpu_offset()
    return f" (LOCAL_RANK + {LOCAL_RANK_GPU_OFFSET_ENV}={offset})" if offset else ""


def resolve_simulation_device(device: str | None = None, *, require_rank_cuda: bool = False) -> str:
    """Resolve the process-local simulation device.

    ``AppLauncher`` understands ``LOCAL_RANK``, but the Holosoma environment also
    receives a device string and uses it to construct managers and simulator
    buffers.  Letting those two paths resolve the device independently is unsafe:
    in a torchrun process rank 1 can launch PhysX on ``cuda:1`` while Holosoma
    still constructs tensors on the historical default ``cuda:0``.

    In a distributed process the local rank is therefore authoritative: the device
    is ``cuda:<LOCAL_RANK + HOLOSOMA_LOCAL_RANK_GPU_OFFSET>`` (see
    :func:`local_rank_gpu_offset`).  An explicit, conflicting CUDA device is
    rejected instead of producing a mixed-device articulation later in the first
    physics write.  A single process with a non-zero offset runs on
    ``cuda:<offset>`` and likewise rejects a conflicting explicit CUDA device.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        if "LOCAL_RANK" not in os.environ:
            raise ValueError(f"WORLD_SIZE={world_size} is set but LOCAL_RANK is missing.")
        local_rank = int(os.environ["LOCAL_RANK"])
        if local_rank < 0 or local_rank >= world_size:
            raise ValueError(f"LOCAL_RANK={local_rank} is outside WORLD_SIZE={world_size}.")
        rank_device = f"cuda:{local_rank_gpu_index(local_rank)}"
        if device is not None:
            requested = str(device)
            if requested == "cuda":
                requested = "cuda:0"
            if not requested.startswith("cuda"):
                if require_rank_cuda:
                    raise ValueError(
                        f"Distributed IsaacSim process LOCAL_RANK={local_rank} requires {rank_device}{_offset_note()}, "
                        f"but non-CUDA simulation device {device!r} was requested."
                    )
                return requested
            if requested.startswith("cuda") and requested != rank_device:
                raise ValueError(
                    f"Distributed process LOCAL_RANK={local_rank} requires {rank_device}{_offset_note()}, "
                    f"but simulation device {device!r} was requested."
                )
        return rank_device

    offset = local_rank_gpu_offset()
    if device is not None:
        requested = str(device)
        if offset and requested.startswith("cuda"):
            offset_device = f"cuda:{offset}"
            if requested == "cuda":
                return offset_device
            if requested != offset_device:
                raise ValueError(
                    f"Single process with {LOCAL_RANK_GPU_OFFSET_ENV}={offset} requires {offset_device}, "
                    f"but simulation device {device!r} was requested."
                )
        return requested
    return f"cuda:{offset}" if torch.cuda.is_available() else "cpu"


def activate_simulation_device(device: str) -> None:
    """Make *device* PyTorch's active CUDA device and verify the assignment.

    Isaac Sim/Kit initialization may change PyTorch's current device.  This must
    be called after ``AppLauncher`` and before constructing any environment or
    IsaacLab asset whose constructor creates implicit CUDA tensors.
    """
    if not device.startswith("cuda"):
        return
    target = torch.device(device)
    target_index = 0 if target.index is None else target.index
    torch.cuda.set_device(target_index)
    current_index = torch.cuda.current_device()
    if current_index != target_index:
        raise RuntimeError(
            f"Failed to activate simulation device {device}: current CUDA device is cuda:{current_index}."
        )


def setup_simulator_imports(config: ExperimentConfig | RunSimConfig) -> None:
    """Setup simulator-specific imports without side effects.

    Parameters
    ----------
    config : ExperimentConfig | RunSimConfig
        Configuration containing simulator settings.
    """
    set_simulator_type(config.simulator)
    simulator_type = get_simulator_type()

    if simulator_type == SimulatorType.MUJOCO:
        import mujoco

        assert mujoco is not None
    elif simulator_type == SimulatorType.ISAACGYM:
        import isaacgym

        assert isaacgym is not None

    # IsaacSim imports handled in setup_isaaclab_launcher


def _stagger_distributed_isaacsim_init() -> None:
    """Offset each rank's IsaacSim/Kit startup and pin its CUDA device first.

    Eight ranks entering Kit initialization simultaneously contend on shader
    compilation, asset conversion, and GPU memory; on PCIe-only hosts (no
    NVLink) that contention has shown up as init stalls.  Two mitigations:

    * Pin this process to its CUDA device BEFORE Kit starts.  Kit's background
      threads otherwise trigger the first CUDA initialization and can land on
      the wrong device.  The device is ``LOCAL_RANK + HOLOSOMA_LOCAL_RANK_GPU_OFFSET``
      (see :func:`local_rank_gpu_offset`); a single process with a non-zero
      offset is pinned to ``cuda:<offset>`` as well.
    * Sleep ``local_rank * HOLOSOMA_INIT_STAGGER_SEC`` seconds so ranks enter
      Kit in a staircase.  Set the variable to ``0`` to disable.

    This complements ``HOLOSOMA_USD_CONVERSION_SLOT`` (per-rank conversion
    directories), which prevents the USD cache write/read race.  With a non-zero
    offset the slot defaults to the physical GPU index (``converted_rank<gpu>``)
    so a job on GPUs 4-7 never shares ``converted_rank0..3`` with a job that is
    starting on GPUs 0-3 at the same time.
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    offset = local_rank_gpu_offset()
    if world_size <= 1:
        if offset and torch.cuda.is_available():
            torch.cuda.set_device(offset)
            logger.info(
                f"[INIT] single process: pinned CUDA device {offset} ({LOCAL_RANK_GPU_OFFSET_ENV}) before IsaacSim init"
            )
        return

    local_rank = int(os.environ.get("LOCAL_RANK", "0"))
    gpu_index = local_rank_gpu_index(local_rank)
    if offset and USD_CONVERSION_SLOT_ENV not in os.environ:
        os.environ[USD_CONVERSION_SLOT_ENV] = str(gpu_index)
        logger.info(
            f"[INIT] rank {local_rank}/{world_size}: {USD_CONVERSION_SLOT_ENV}={gpu_index} "
            "(physical GPU index; keeps converted_rank* dirs disjoint from a job on lower GPUs)"
        )
    if torch.cuda.is_available():
        torch.cuda.set_device(gpu_index)
        logger.info(
            f"[INIT] rank {local_rank}/{world_size}: pinned CUDA device {gpu_index}{_offset_note()} before IsaacSim init"
        )

    stagger_sec = float(os.environ.get("HOLOSOMA_INIT_STAGGER_SEC", "5"))
    delay = local_rank * stagger_sec
    if delay > 0:
        logger.info(f"[INIT] rank {local_rank}/{world_size}: staggering {delay:g}s before IsaacSim init")
        time.sleep(delay)


def _configure_launcher_device(args_cli: argparse.Namespace, device: str | None) -> str | None:
    """Set ``args_cli.device`` (and the distributed flags) for IsaacLab's ``AppLauncher``.

    Returns the CUDA device string the launcher is expected to end up on, or
    ``None`` when the launcher auto-detects (stock single-process behaviour).

    IsaacLab's ``--distributed`` path (``isaaclab/app/app_launcher.py``,
    ``_resolve_device_settings``) ignores the ``device`` argument and hard-wires
    ``device_id = LOCAL_RANK (+ JAX_LOCAL_RANK)``, i.e. physical GPU == LOCAL_RANK.
    With a non-zero ``HOLOSOMA_LOCAL_RANK_GPU_OFFSET`` we therefore pass the
    explicit physical ``cuda:<LOCAL_RANK + offset>`` WITHOUT ``distributed`` and
    re-create the two remaining side effects of that path ourselves: a
    single-GPU Kit (``multi_gpu=False``) and the per-process CPU thread limits
    (``PXR_WORK_THREAD_LIMIT`` / ``OPENBLAS_NUM_THREADS`` /
    ``--/plugins/carb.tasking.plugin/threadCount``, which ``AppLauncher`` strips
    from ``sys.argv`` after launch).
    """
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    if world_size > 1:
        # Distribute simulator across GPUs when using multi-gpu training
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        gpu_index = local_rank_gpu_index(local_rank)
        args_cli.device = f"cuda:{gpu_index}"
        if gpu_index == local_rank:
            args_cli.distributed = True
        else:
            args_cli.distributed = False
            args_cli.multi_gpu = False
            num_threads_per_process = max(1, (os.cpu_count() or 1) // world_size)
            os.environ["PXR_WORK_THREAD_LIMIT"] = str(num_threads_per_process)
            os.environ["OPENBLAS_NUM_THREADS"] = str(num_threads_per_process)
            sys.argv.append(f"--/plugins/carb.tasking.plugin/threadCount={num_threads_per_process}")
            logger.info(
                f"[INIT] rank {local_rank}/{world_size}: AppLauncher device {args_cli.device}{_offset_note()} "
                "passed explicitly (IsaacLab --distributed would force device_id = LOCAL_RANK); "
                f"multi_gpu=False, {num_threads_per_process} CPU threads/process"
            )
        return args_cli.device
    if device is not None:
        # Use the resolved device
        args_cli.device = device
        return device if str(device).startswith("cuda") else None
    offset = local_rank_gpu_offset()
    if offset:
        # single process on a non-zero physical GPU (all GPUs visible, Kit ignores CUDA_VISIBLE_DEVICES)
        args_cli.device = f"cuda:{offset}"
        args_cli.multi_gpu = False
        return args_cli.device
    # AppLauncher auto-detects
    return None


def _verify_launcher_device(app_launcher: Any, expected_device: str | None) -> None:
    """Fail loudly if ``AppLauncher`` resolved a different CUDA device than requested."""
    if expected_device is None or not str(expected_device).startswith("cuda:"):
        return
    expected_index = int(str(expected_device).split(":")[-1])
    actual_index = getattr(app_launcher, "device_id", None)
    if actual_index is not None and int(actual_index) != expected_index:
        raise RuntimeError(
            f"IsaacLab AppLauncher resolved device_id={actual_index} but {expected_device} was requested "
            f"(LOCAL_RANK={os.environ.get('LOCAL_RANK')}, {LOCAL_RANK_GPU_OFFSET_ENV}={local_rank_gpu_offset()}); "
            "Kit would run PhysX/rendering on the wrong physical GPU."
        )


def setup_isaaclab_launcher(
    config: ExperimentConfig | RunSimConfig,
    device: str | None = None,
    *,
    launcher_num_envs_global: int | None = None,
) -> Any | None:
    """Handle IsaacSim-specific launcher setup.

    Parameters
    ----------
    config : ExperimentConfig | RunSimConfig
        Configuration containing simulator and training settings.
    device : str
        Resolved device string (e.g., 'cuda:0', 'cpu').
    launcher_num_envs_global : int | None, optional
        Global environment count used only to size ``AppLauncher`` in a
        distributed process.  This is useful for callers that have already
        sharded ``config.training.num_envs`` before constructing the actual
        environment.  The default preserves the historical behavior.

    Returns
    -------
    Any | None
        IsaacSim simulation app instance, or None for other simulators.
    """
    from isaaclab.app import AppLauncher

    parser = argparse.ArgumentParser(description="Run simulation with IsaacSim.")
    parser.add_argument("--num_envs", type=int, default=1, help="Number of environments to simulate.")
    parser.add_argument("--seed", type=int, default=None, help="Seed used for the environment")
    parser.add_argument("--env_spacing", type=int, default=20, help="Distance between environments in simulator.")
    parser.add_argument("--output_dir", type=str, default="logs", help="Directory to store the output.")
    AppLauncher.add_app_launcher_args(parser)

    # Parse known arguments to get argparse params
    args_cli, unknown_args = parser.parse_known_args()

    # Set values from config — divide by world_size for multi-GPU so each rank's
    # AppLauncher only allocates resources for its share of environments.
    # HoloSoma's native train_agent initializes AppLauncher from the global
    # config and only then shards the environment config.  Custom callers that
    # use setup_simulation_environment() cannot split those two phases, so they
    # can provide the original global width explicitly while passing an already
    # sharded config to the environment constructor.
    world_size = int(os.environ.get("WORLD_SIZE", "1"))
    launcher_global_width = (
        int(config.training.num_envs)
        if launcher_num_envs_global is None
        else int(launcher_num_envs_global)
    )
    if launcher_global_width <= 0:
        raise ValueError(
            f"AppLauncher global num_envs must be positive, got {launcher_global_width}."
        )
    if world_size > 1 and launcher_global_width % world_size:
        raise ValueError(
            "AppLauncher global num_envs must be divisible by WORLD_SIZE: "
            f"global={launcher_global_width} world={world_size}."
        )
    args_cli.num_envs = (
        launcher_global_width // world_size if world_size > 1 else launcher_global_width
    )
    args_cli.seed = config.training.seed
    args_cli.env_spacing = config.simulator.config.scene.env_spacing
    args_cli.output_dir = config.logger.base_dir
    args_cli.headless = config.training.headless
    # Device: LOCAL_RANK + HOLOSOMA_LOCAL_RANK_GPU_OFFSET in a torchrun process, cuda:<offset> for a single
    # process with an offset, the resolved device or AppLauncher auto-detection otherwise.
    expected_launcher_device = _configure_launcher_device(args_cli, device)

    # Check if video recording is enabled and add --enable_cameras flag
    video_enabled = config.logger.video.enabled or config.logger.headless_recording
    # NV4: head depth camera also needs --enable_cameras. Recursively detect enable_head_depth.
    def _has_head_depth(o, seen=None):
        import dataclasses
        if seen is None: seen=set()
        if id(o) in seen or not dataclasses.is_dataclass(o): return False
        seen.add(id(o))
        for f in dataclasses.fields(o):
            if f.name=="enable_head_depth" and getattr(o,f.name,False): return True
            if _has_head_depth(getattr(o,f.name,None), seen): return True
        return False
    if video_enabled or _has_head_depth(config):
        args_cli.enable_cameras = True

    _stagger_distributed_isaacsim_init()

    app_launcher = AppLauncher(args_cli)
    _verify_launcher_device(app_launcher, expected_launcher_device)
    simulation_app = app_launcher.app

    logger.info(f"IsaacSim args_cli: {args_cli}")
    logger.info(f"IsaacSim unknown_args: {unknown_args}")
    sys.argv = [sys.argv[0]] + unknown_args

    return simulation_app


def setup_keyboard_listener(env) -> threading.Thread:
    """Setup keyboard listener thread for simulation control.

    Parameters
    ----------
    env
        Environment instance to control.

    Returns
    -------
    threading.Thread
        Keyboard listener thread (already started).
    """

    def on_press(key, env):
        """Handle keyboard input for simulation control."""
        try:
            if hasattr(key, "char") and key.char:
                if key.char == "n":
                    if hasattr(env, "next_task"):
                        env.next_task()
                        logger.info("Moved to the next task.")
                # Force Control
                elif key.char == "1":
                    if hasattr(env, "apply_force_scale"):
                        env.apply_force_scale /= 2.0
                        logger.info(f"apply_force_scale: {env.apply_force_scale}")
                elif key.char == "2":
                    if hasattr(env, "apply_force_scale"):
                        env.apply_force_scale *= 2.0
                        logger.info(f"apply_force_scale: {env.apply_force_scale}")
        except AttributeError:
            pass

    def listen_for_keypress(env):
        """Listen for keyboard input in a separate thread."""
        try:
            # Delay import so that one can run the rest of this script in headless mode.
            # Trying to import pynput in headless mode gives the following error:
            # ImportError: this platform is not supported:
            # ('failed to acquire X connection: Bad display name ""', DisplayNameError(''))
            from pynput import keyboard as pynput_keyboard

            logger.info("Keyboard controls:")
            logger.info("  n - Next task (if supported)")
            logger.info("  1/2 - Decrease/Increase force scale (if supported)")

            with pynput_keyboard.Listener(on_press=lambda key: on_press(key, env)) as listener:
                listener.join()
        except ImportError:
            logger.warning("pynput not available - keyboard controls disabled")
        except Exception as e:
            logger.warning(f"Keyboard listener failed: {e}")

    key_listener_thread = threading.Thread(target=listen_for_keypress, args=(env,))
    key_listener_thread.daemon = True
    key_listener_thread.start()
    return key_listener_thread


def setup_simulation_environment(
    config: ExperimentConfig | RunSimConfig,
    device: str | None = None,
    *,
    launcher_num_envs_global: int | None = None,
) -> tuple[Any, str, Any]:
    """Setup simulation environment with shared infrastructure.

    This function handles common setup for training, evaluation and direct simulation:
    - Simulator imports and initialization
    - Device selection and seeding
    - Environment creation
    - Keyboard listener setup (if not headless)

    Parameters
    ----------
    config : ExperimentConfig | RunSimConfig
        Configuration containing all simulation settings.
    device : str | None, optional
        Device to use for simulation. If None, auto-detects CUDA availability.
    launcher_num_envs_global : int | None, optional
        Original global environment count for distributed AppLauncher sizing
        when ``config.training.num_envs`` is already the per-rank width.

    Returns
    -------
    tuple[Any, str, Any]
        Tuple of (environment, device_string, simulation_app).
        simulation_app is None for simulators that don't need it (MuJoCo, IsaacGym).
    """
    logger.info("🚀 Setting up simulation environment...")

    # Setup simulator imports
    setup_simulator_imports(config)

    # Device selection - must happen before IsaacSim launcher setup.  LOCAL_RANK
    # is authoritative under torchrun so AppLauncher and the Holosoma env receive
    # exactly the same per-process device.
    device = resolve_simulation_device(
        device, require_rank_cuda=get_simulator_type() == SimulatorType.ISAACSIM
    )
    logger.info(f"Device: {device}")

    # Handle IsaacSim launcher if needed (for both ExperimentConfig and RunSimConfig)
    simulation_app = None
    if get_simulator_type() == SimulatorType.ISAACSIM:
        simulation_app = setup_isaaclab_launcher(
            config,
            device,
            launcher_num_envs_global=launcher_num_envs_global,
        )
        # Kit/AppLauncher may change torch's current device while loading plugins.
        # Restore it before seeding and, critically, before IsaacLab constructs
        # articulation actuator index tensors.
        activate_simulation_device(device)

    # Set random seed if specified (only for ExperimentConfig)
    if isinstance(config, ExperimentConfig) and config.training.seed is not None:
        seeding(config.training.seed, torch_deterministic=config.training.torch_deterministic)
        logger.info(f"Seed: {config.training.seed}")

    # For RunSimConfig, we need a different approach since it doesn't have env_class or training configs
    if isinstance(config, RunSimConfig):
        # For run_sim.py, we'll create the simulator directly instead of using environment wrapper
        logger.info("Direct simulation mode - creating simulator directly, without experiment config")

        # Create FullSimConfig from RunSimConfig
        # Extract SimulatorInitConfig from SimulatorConfig
        full_config = FullSimConfig(
            simulator=config.simulator.config,  # Extract .config from SimulatorConfig
            robot=config.robot,
            training=config.training,
            logger=config.logger,
            experiment_dir=None,
        )

        # For compatibility, minimal proxy for TerrainManager since it depends on env
        class EnvProxy:
            def __init__(self, device):
                self.num_envs = 1
                self.device = device

        # For compatibility, wrap in a minimal object that has .sim attribute
        class DirectSimWrapper:
            def __init__(self, simulator):
                self.sim = simulator

            def reset(self):
                # Basic reset - just initialize the simulator if needed
                if hasattr(self.sim, "reset"):
                    self.sim.reset()

            def close(self):
                if hasattr(self.sim, "close"):
                    self.sim.close()

        # Use terrain configuration from RunSimConfig
        terrain_manager = TerrainManager(config.terrain, env=EnvProxy(device), device=device)

        # Create simulator using get_class() to avoid circular imports
        simulator_class = get_class(config.simulator._target_)
        simulator = simulator_class(full_config, terrain_manager, device)

        # Now we have an "env" to return which is actually the direct simulator
        env = DirectSimWrapper(simulator)
        logger.debug("Direct simulator created successfully!")

    else:
        # Original ExperimentConfig path
        env_target = config.env_class
        tyro_env_config = get_tyro_env_config(config)

        logger.info(f"Creating environment: {env_target}")
        env_class = get_class(env_target)
        env = env_class(tyro_env_config, device=device)

        logger.debug("Environment created successfully!")

        # Setup keyboard listener if not headless
        if not config.training.headless:
            setup_keyboard_listener(env)

    return env, device, simulation_app


def close_simulation_app(simulation_app):
    """Close simulation app with workarounds for known issues.

    Parameters
    ----------
    simulation_app : Any
        The simulation app instance returned by init_sim_imports().
        Can be None for simulators that don't have an app (e.g., IsaacGym).
    """
    if simulation_app is not None and get_simulator_type() == SimulatorType.ISAACSIM:
        logger.info("Shutting down simulation app...")
        try:


            import omni.usd

            context = omni.usd.get_context()
            context_class = context.__class__

            # Replace with a no-op version
            def noop_close_stage(self, *args, **kwargs):
                logger.info("Skipping close_stage() to avoid hanging")
                return True

            # Apply the patch
            context_class.close_stage = noop_close_stage
            logger.info("Successfully patched close_stage method")
        except Exception as e:
            logger.warning(f"Could not patch close_stage method: {e}")

        try:
            # Work-around for IsaacLab SimulationContext._app_control_on_stop_handle_fn
            # hanging in an infinite render() loop on shutdown. When simulation_app.close()
            # triggers a timeline STOP event, the callback spins waiting for the timeline to
            # start playing again — which never happens. Disabling the callback prevents this.
            from isaaclab.sim import SimulationContext

            sim_context = SimulationContext.instance()
            if sim_context is not None:
                sim_context._disable_app_control_on_stop_handle = True
                logger.info("Disabled SimulationContext app_control_on_stop_handle to prevent shutdown hang")
        except Exception as e:
            logger.warning(f"Could not disable app_control_on_stop_handle: {e}")

        # Now close the app
        simulation_app.close(wait_for_replicator=False)
        logger.info("Simulation app closed.")
    else:
        logger.info("Simulation app closed.")


class DirectSimulation:
    """Encapsulates direct simulation logic for run_sim.py.

    This class provides a clean interface for running direct simulations without
    training or evaluation environments, handling all initialization,
    loop management, and cleanup logic.

    Can be used as a context manager for resource management.

    Examples
    --------
    >>> with DirectSimulation(config, env, device, simulation_app) as sim:
    ...     sim.run()
    """

    def __init__(self, config: RunSimConfig, env: Any, device: str, simulation_app: Any):
        """Initialize DirectSimulation instance.

        Parameters
        ----------
        config : RunSimConfig
            Configuration containing all simulation settings.
        env : Any
            Environment wrapper containing the simulator.
        device : str
            Device for tensor operations.
        simulation_app : Any
            Simulation app instance (if any).
        """
        self.config = config
        self.env = env
        self.device = device
        self.simulation_app = simulation_app
        self.simulator = env.sim

    def __enter__(self) -> Self:
        """Context manager entry - initialize the simulation.

        Returns
        -------
        Self
            Self for use in the with statement.
        """
        self.initialize()
        return self

    def __exit__(self, exc_type, exc_val, exc_tb) -> None:
        """Context manager exit - cleanup the simulation.

        Parameters
        ----------
        exc_type : type or None
            Exception type if an exception occurred.
        exc_val : Exception or None
            Exception instance if an exception occurred.
        exc_tb : traceback or None
            Traceback if an exception occurred.
        """
        self.cleanup()

    def initialize(self) -> None:
        """Handle the complete simulator initialization sequence.

        Performs the initialization process required for proper simulator
        lifecycle management. Ideally this is moved into the simulator interface and
        to simplify training, evaluation and direct usage.
        """
        logger.debug("Initializing simulator...")

        # Need to manually set headless since it's in training config currently
        self.simulator.set_headless(False)

        # Step 1: Basic setup
        self.simulator.setup()
        logger.debug("simulator.setup() completed")

        # Step 2: Setup terrain
        self.simulator.setup_terrain()
        logger.debug("simulator.setup_terrain() completed")

        # Step 3: Load assets (this initializes the bridge!)
        self.simulator.load_assets()
        logger.debug("simulator.load_assets() completed - bridge should now be initialized")

        # Step 4: Create environments (need to provide required parameters)
        # Create env_origins (single environment at origin)
        env_origins = torch.zeros(1, 3, device=self.device)

        # Create base_init_state from robot config
        base_init_state = self._create_base_init_state()

        self.simulator.create_envs(1, env_origins, base_init_state)
        logger.debug("simulator.create_envs() completed")

        # Step 5: Prepare simulation
        self.simulator.prepare_sim()
        logger.debug("simulator.prepare_sim() completed")

        # Step 5.5: Initialize episode (positions virtual gantry, etc.)
        self.simulator.on_episode_start(env_id=0)
        logger.debug("simulator.on_episode_start() completed")

        # Step 6: Setup viewer if not headless
        if not self.config.training.headless:
            self.simulator.setup_viewer()
            logger.debug("simulator.setup_viewer() completed")

        logger.info("Simulator initialized")

        # Step 7: Toggle start recording if enabled
        if self.simulator.video_recorder and self.simulator.video_recorder.enabled:
            # arbitrary episode ID given this is sim2sim, we may want to
            # actually support toggling recording and with better filenames too
            self.simulator.video_recorder.start_recording(episode_id=0)

    def run(self) -> None:
        """Run the direct simulation loop with viewer sync and FPS logging.

        Manages the complete simulation loop including rate limiting,
        viewer synchronization, FPS logging, and error handling.
        """
        # Setup rate limiting
        sim_frequency = self.config.simulator.config.sim.fps
        rate_limiter = RateLimiter(sim_frequency)

        # Calculate viewer sync frequency
        viewer_steps = self._calculate_viewer_steps()

        logger.info(f"Simulation rate: {sim_frequency} Hz ({1.0 / sim_frequency * 1000:.2f} ms)")
        logger.info(f"Viewer rate: {1 / self.config.viewer_dt:.1f} Hz (sync every {viewer_steps} steps)")
        logger.info("Starting direct simulation loop...")
        logger.info("Press Ctrl+C to stop simulation")

        # Determine refresh strategy based on simulator type
        # IsaacGym/IsaacSim: need pre-step to refresh tensors to sync simulator state
        # MuJoCo: no pre-step refresh needed because we are NOT running an envs/tasks requiring
        #         those tensors e.g, _rigid_body_rot, _rigid_body_vel, etc.
        simulator_type = get_simulator_type()
        if simulator_type in [SimulatorType.ISAACGYM, SimulatorType.ISAACSIM]:
            pre_step_refresh = self.simulator.refresh_sim_tensors
        else:
            pre_step_refresh = lambda: None  # noqa: E731  (No-op for MuJoCo)

        # Direct simulation loop (like holosoma_inference's simulation_thread)
        step_count = 0
        start_time = time.time()
        fps_start_time = start_time

        while True:
            try:
                # Refresh tensors if needed (no-op for MuJoCo)
                pre_step_refresh()

                # Direct simulator step - this triggers bridge.step() inside simulate_at_each_physics_step()
                self.simulator.simulate_at_each_physics_step()

                # Update viewer at display rate
                if step_count % viewer_steps == 0:
                    self.simulator.render()

                # Periodic FPS logging (every 1000 steps)
                if step_count > 0 and step_count % 1000 == 0:
                    fps_start_time = self._log_fps(step_count, fps_start_time)

                step_count += 1
                rate_limiter.sleep()

            except KeyboardInterrupt:  # noqa: PERF203
                logger.info("Simulation interrupted by user (Ctrl+C)")
                break
            except Exception as e:
                logger.error(f"Error during simulation step {step_count}: {e}")
                traceback.print_exc()
                break

        # Final statistics
        total_elapsed = time.time() - start_time
        avg_fps = step_count / total_elapsed if total_elapsed > 0 else 0
        logger.info(f"Simulation completed after {step_count} steps")
        logger.info(f"Average FPS: {avg_fps:.1f} (target: {sim_frequency})")

    def cleanup(self) -> None:
        """Handle simulation cleanup."""
        # Cleanup environment
        if hasattr(self.env, "close"):
            self.env.close()

        if self.simulator.video_recorder:
            self.simulator.video_recorder.cleanup()

        # Cleanup simulation app
        if self.simulation_app:
            close_simulation_app(self.simulation_app)

    def _create_base_init_state(self) -> torch.Tensor:
        """Create base initialization state tensor from robot configuration.

        Returns
        -------
        torch.Tensor
            Base initialization state tensor.
        """
        base_init_state_list = (
            self.config.robot.init_state.pos
            + self.config.robot.init_state.rot
            + self.config.robot.init_state.lin_vel
            + self.config.robot.init_state.ang_vel
        )
        return to_torch(base_init_state_list, device=self.device, requires_grad=False)

    def _calculate_viewer_steps(self) -> int:
        """Calculate viewer synchronization frequency.

        Returns
        -------
        int
            Number of simulation steps between viewer updates.
        """
        viewer_dt = self.config.viewer_dt
        sim_dt = 1.0 / self.config.simulator.config.sim.fps
        return max(1, int(viewer_dt / sim_dt))

    def _log_fps(self, step_count: int, fps_start_time: float) -> float:
        """Log FPS statistics for simulation performance monitoring.

        Parameters
        ----------
        step_count : int
            Current step count.
        fps_start_time : float
            Start time for FPS measurement.

        Returns
        -------
        float
            New start time for next FPS measurement.
        """
        elapsed = time.time() - fps_start_time
        fps = 1000 / elapsed
        logger.info(f"Simulation FPS: {fps:.1f}")
        return time.time()
