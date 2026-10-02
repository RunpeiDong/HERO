from __future__ import annotations

import builtins
import copy
import dataclasses
import os
import xml.etree.ElementTree as ET
from typing import Any

import pathlib
import trimesh

from holosoma.config_types.full_sim import FullSimConfig
import isaaclab.sim as sim_utils
from pxr import Gf
from isaaclab.assets import RigidObject, RigidObjectCfg
import isaaclab.terrains as terrain_gen
import omni.log
import torch
from isaaclab.actuators import IdealPDActuatorCfg
from isaaclab.assets import Articulation, ArticulationCfg
from isaaclab.envs import ViewerCfg, mdp
from isaaclab.managers import EventManager, SceneEntityCfg
from isaaclab.managers import EventTermCfg as EventTerm
from isaaclab.scene import InteractiveScene, InteractiveSceneCfg
from isaaclab.sensors import (ContactSensor, ContactSensorCfg, RayCaster, RayCasterCfg, patterns,
                              TiledCamera, TiledCameraCfg)
from isaaclab.sim import PhysxCfg, SimulationCfg, SimulationContext
from isaaclab.terrains import TerrainGeneratorCfg, TerrainImporterCfg
from isaaclab.terrains.utils import create_prim_from_mesh
from isaaclab.utils.timer import Timer
from loguru import logger
from omegaconf import DictConfig

from holosoma.utils.module_utils import get_holosoma_root
from holosoma.utils.path import resolve_data_file_path
from holosoma.config_types.simulator import SimulatorInitConfig, SceneConfig
from holosoma.managers.terrain import TerrainManager
from holosoma.simulator.base_simulator.base_simulator import BaseSimulator
from holosoma.simulator.isaacsim.event_cfg import EventCfg
from holosoma.simulator.isaacsim.events import randomize_body_com, randomize_rigid_body_inertia
from holosoma.simulator.isaacsim.isaaclab_viewpoint_camera_controller import ViewportCameraController
from holosoma.simulator.isaacsim.isaacsim_articulation_cfg import ARTICULATION_CFG
from holosoma.simulator.isaacsim import multi_usd as _multi_usd
from holosoma.simulator.isaacsim.usd_file_loader import USDFileLoader
from holosoma.simulator.isaacsim.registry_utils import register_objects
from holosoma.simulator.isaacsim.proxy_utils import AllRootStatesProxy, RootStatesProxy
from holosoma.simulator.isaacsim.state_adapter import IsaacSimStateAdapter
from holosoma.simulator.isaacsim.body_velocity import body_link_linear_velocity_w
from holosoma.simulator.isaacsim.foot_collision import prepare_foot_collision_urdf
from holosoma.simulator.isaacsim.terrain_material import describe_combine_modes, terrain_material_combine_modes
from holosoma.simulator.isaacsim.prim_utils import (
    log_robot_properties,
    print_prim_tree,
    UsdSceneLoaderCfg,
    create_usd_scene_loader,
)
from holosoma.simulator.isaacsim.video_recorder import IsaacSimVideoRecorder
from holosoma.simulator.shared.virtual_gantry import (
    VirtualGantry,
    create_virtual_gantry,
    GantryCommand,
    GantryCommandData,
)

from holosoma.simulator.types import ActorNames, ActorIndices, EnvIds, ActorStates, ActorPoses


class IsaacSim(BaseSimulator):
    def __init__(self, tyro_config: FullSimConfig, terrain_manager: TerrainManager, device: str):
        # This is intentionally before BaseSimulator/ObjectRegistry and before
        # SimulationContext/InteractiveScene construction.  IsaacLab creates
        # several implicit CUDA tensors (including actuator joint indices) while
        # initializing an articulation; they must use the same rank-local device
        # as the PhysX state tensors.  Keep this local guard even though the shared
        # simulation setup also restores the device after AppLauncher, since
        # IsaacSim can be constructed directly by external callers.
        if str(device).startswith("cuda"):
            requested = torch.device(device)
            requested_index = 0 if requested.index is None else requested.index
            torch.cuda.set_device(requested_index)
            if torch.cuda.current_device() != requested_index:
                raise RuntimeError(f"Could not activate rank-local IsaacSim device {device}.")

        super().__init__(tyro_config, terrain_manager, device)

        # Add device attribute for base simulator compatibility
        self.device = device

        sim_config: SimulationCfg = SimulationCfg(
            dt=1.0 / self.simulator_config.sim.fps,
            render_interval=self.simulator_config.sim.render_interval,
            device=self.sim_device,
            physx=PhysxCfg(
                bounce_threshold_velocity=self.simulator_config.sim.physx.bounce_threshold_velocity,
                solver_type=self.simulator_config.sim.physx.solver_type,
                max_position_iteration_count=self.simulator_config.sim.physx.num_position_iterations,
                max_velocity_iteration_count=self.simulator_config.sim.physx.num_velocity_iterations,
                gpu_max_rigid_patch_count=10 * 2**15,
            ),
            # Global physics material, can be overridden by the individual articulation
            # Can be inspected by:
            # materials = self._robot.root_physx_view.get_material_properties()
            physics_material=sim_utils.RigidBodyMaterialCfg(
                static_friction=1.0,  # default is 0.5
                dynamic_friction=1.0,  # default is 0.5
                restitution=0.0,
            ),
        )

        # create a simulation context to control the simulator
        if SimulationContext.instance() is None:
            self.sim: SimulationContext = SimulationContext(sim_config)
        else:
            raise RuntimeError("Simulation context already exists. Cannot create a new one.")

        self.sim.set_camera_view([2.0, 0.0, 2.5], [-0.5, 0.0, 0.5])

        logger.info("IsaacSim initialized.")
        # Log useful information
        logger.info("[INFO]: Base environment:")
        logger.info(f"\tEnvironment device    : {self.sim_device}")
        logger.info(f"\tPhysics step-size     : {1.0 / self.simulator_config.sim.fps}")
        logger.info(
            f"\tRendering step-size   : {1.0 / self.simulator_config.sim.fps * self.simulator_config.sim.substeps}"
        )

        if self.simulator_config.sim.render_interval < self.simulator_config.sim.control_decimation:
            msg = (
                f"The render interval ({self.simulator_config.sim.render_interval}) is smaller than the decimation "
                f"({self.simulator_config.sim.control_decimation}). Multiple render calls will happen for each "
                "environment step. If this is not intended, set the render interval to be equal to the decimation."
            )
            logger.warning(msg)

        # Per-environment robot variants; None identifies a scene with one asset.
        self.robot_variant_ids: torch.Tensor | None = None
        self.robot_variant_names: tuple[str, ...] | None = None
        self.robot_variant_usd_paths: list[str] | None = None
        self.robot_variant_weights: tuple[int, ...] | None = None
        self.robot_variant_layout: str | None = None
        self.robot_collision_assets: list[dict[str, Any]] = []
        self._multi_usd_planned_variant_ids: list[int] | None = None
        replicate_physics = self.simulator_config.scene.replicate_physics
        if getattr(self.robot_config.asset, "urdf_files", None):
            if replicate_physics:
                logger.warning(
                    "[MULTI-USD] robot.asset.urdf_files is set: forcing InteractiveSceneCfg.replicate_physics=False "
                    "(Isaac Lab requires it for heterogeneous per-env assets). Scene creation and PhysX parsing now "
                    "scale with num_envs (each env is parsed separately) -- expect a longer start-up and a higher host "
                    "RSS than the replicated scene."
                )
            replicate_physics = False
        # Per-environment object variants under /World/envs/env_.*/Object.
        # Same contract as the robot attributes above: present on every scene, None = single (or no) object asset.
        self.object_variant_ids: torch.Tensor | None = None
        self.object_variant_names: tuple[str, ...] | None = None
        self.object_variant_usd_paths: list[str] | None = None
        self.object_variant_urdf_paths: list[str] | None = None
        self.object_variant_weights: tuple[int, ...] | None = None
        self.object_variant_layout: str | None = None
        self.object_variant_sizes: torch.Tensor | None = None  # Float[K, 3] box edges parsed from the URDFs (NaN = no box primitive)
        self._multi_usd_object_planned_variant_ids: list[int] | None = None
        if getattr(self.robot_config.object, "object_urdf_paths", None):
            if replicate_physics:
                logger.warning(
                    "[MULTI-USD] robot.object.object_urdf_paths is set: forcing InteractiveSceneCfg.replicate_physics=False "
                    "(heterogeneous per-env object assets; same reason as urdf_files above)"
                )
            replicate_physics = False
        scene_config: InteractiveSceneCfg = InteractiveSceneCfg(
            num_envs=self.training_config.num_envs,
            env_spacing=self.simulator_config.scene.env_spacing,
            replicate_physics=replicate_physics,
        )
        # generate scene
        with Timer("[INFO]: Time taken for scene creation", "scene_creation"):
            self.scene = InteractiveScene(scene_config)
            self._setup_scene()
        print("[INFO]: Scene manager: ", self.scene)

        if self.simulator_config.viewer.enable_tracking:
            viewer_config: ViewerCfg = ViewerCfg(origin_type="asset_root", asset_name="robot", eye=(0.0, -1.5, 1.5))
        else:
            viewer_config: ViewerCfg = ViewerCfg()

        if self.sim.render_mode >= self.sim.RenderMode.PARTIAL_RENDERING:
            self.viewport_camera_controller: ViewportCameraController | None = ViewportCameraController(
                self, viewer_config
            )
        else:
            self.viewport_camera_controller = None

        # play the simulator to activate physics handles
        # note: this activates the physics simulation view that exposes TensorAPIs
        # note: when started in extension mode, first call sim.reset_async() and then initialize the managers
        if builtins.ISAAC_LAUNCHED_FROM_TERMINAL is False:  # type: ignore[attr-defined]
            logger.info("Starting the simulation. This may take a few seconds. Please wait...")
            with Timer("[INFO]: Time taken for simulation start", "simulation_start"):
                self.sim.reset()

        self._validate_articulation_tensor_devices()

        self.default_coms = self._robot.root_physx_view.get_coms().clone()
        # Link-frame COM offsets in simulator body order for the reset writers' link-to-COM root
        # velocity conversion; built lazily by the ``body_com_offset_b`` property (``body_ids`` exist only after
        # ``load_assets``) and refreshed by ``events.randomize_body_com``
        self._body_com_offset_b: torch.Tensor | None = None
        self.base_com_bias = torch.zeros((self.training_config.num_envs, 3), dtype=torch.float, device="cpu")

        self.events_cfg = EventCfg()

        self.event_manager = EventManager(self.events_cfg, self)
        print("[INFO] Event Manager: ", self.event_manager)

        if "startup" in self.event_manager.available_modes:
            self.event_manager.apply(mode="startup")

        # -- event manager used for randomization
        # if self.cfg.events:
        #     self.event_manager = EventManager(self.cfg.events, self)
        #     print("[INFO] Event Manager: ", self.event_manager)

        if "cuda" in self.sim_device:
            torch.cuda.set_device(self.sim_device)


        # perform events at the start of the simulation
        # if self.cfg.events:
        #     if "startup" in self.event_manager.available_modes:
        #         self.event_manager.apply(mode="startup")

        # # -- set the framerate of the gym video recorder wrapper so that the playback speed of
        # the produced video matches the simulation
        # self.metadata["render_fps"] = 1. / self.config.sim.fps * self.config.sim.control_decimation

        self._sim_step_counter = 0

        if self.video_config.enabled:
            self.video_recorder = IsaacSimVideoRecorder(self.video_config, self)

        # debug visualization
        # self.draw = _debug_draw.acquire_debug_draw_interface()

        # print the environment information
        logger.info("Completed setting up the environment...")

    # ------------------------------------------------------------------------------------------------
    # Per-environment robot variants from multiple USD assets.
    # ------------------------------------------------------------------------------------------------
    @staticmethod
    def _usd_conversion_slot() -> str:
        """Process-private conversion slot (same rule as the single-URDF branch: HOLOSOMA_USD_CONVERSION_SLOT or LOCAL_RANK)."""
        local_rank = int(os.environ.get("LOCAL_RANK", "0"))
        conversion_slot = os.environ.get("HOLOSOMA_USD_CONVERSION_SLOT", str(local_rank))
        if not conversion_slot.replace("_", "").replace("-", "").isalnum():
            raise ValueError(
                "HOLOSOMA_USD_CONVERSION_SLOT must contain only letters, "
                f"digits, '_' or '-', got {conversion_slot!r}"
            )
        return conversion_slot

    @property
    def contact_history_length(self) -> int:
        """Contact-force history length covering every physics substep of a control step."""
        return max(int(self.simulator_config.contact_sensor_history_length), int(self.simulator_config.sim.control_decimation))

    @property
    def body_com_offset_b(self) -> torch.Tensor:
        """``[num_envs, num_bodies, 3]`` LINK-frame centre-of-mass offset of every robot body in SIMULATOR body order
        (``self.body_names``), on the simulation device.

        IsaacLab's root-state setter (``write_root_velocity_to_sim``) takes the COM velocity while a certified link-origin
        corpus stores link-origin velocities; the command reset writers convert with this offset
        (``MotionCommand._reset_root_lin_vel_for_simulator``).  Source: ``root_physx_view.get_coms()`` (``[N, B, 7]`` pos + quat
        in the link frame, CPU, articulation link order) indexed with ``body_ids``; cached on first access and kept in step
        with the view by :meth:`refresh_body_com_offsets`, which every writer of ``set_coms`` on the robot calls after its
        write (``events.randomize_body_com`` with the full tensor; the per-episode EE COM randomization
        ``hero_isaacsim.managers.randomization.hero._write_coms`` with the rows it wrote).
        """
        cache = getattr(self, "_body_com_offset_b", None)
        if cache is None:
            cache = self.refresh_body_com_offsets()
            if cache is None:
                raise RuntimeError("body_com_offset_b is available only after load_assets() resolved the robot body order")
        return cache

    def refresh_body_com_offsets(self, coms: torch.Tensor | None = None, env_ids: torch.Tensor | None = None) -> torch.Tensor | None:
        """Rebuild / update the :attr:`body_com_offset_b` cache after a ``set_coms`` on the robot view.

        ``coms`` is the FULL ``get_coms()``-layout tensor ``[N, B_articulation, 7]`` that was handed to ``set_coms`` (None:
        read the live PhysX view).  ``env_ids`` (CPU / any device, long) restricts the update to those rows of an EXISTING
        cache -- the rows ``set_coms(coms, env_ids)`` actually wrote -- so a per-episode writer never overwrites rows it did not
        touch; without ``env_ids`` (or before the cache exists) the whole cache is rebuilt from ``coms``.  Before
        ``load_assets()`` (no ``body_ids`` yet, e.g. startup events) the cache is only invalidated and None is returned; the
        next property access rebuilds it from the live view."""
        body_ids = getattr(self, "body_ids", None)
        if body_ids is None:
            self._body_com_offset_b = None
            return None
        if coms is None:
            coms = self._robot.root_physx_view.get_coms()
        coms = torch.as_tensor(coms)
        cache = getattr(self, "_body_com_offset_b", None)
        if env_ids is None or cache is None:
            offsets = coms[:, list(body_ids), :3]
            self._body_com_offset_b = offsets.to(device=self.sim_device, dtype=torch.float32).clone()
            return self._body_com_offset_b
        rows = torch.as_tensor(env_ids).to(device=coms.device, dtype=torch.long)
        cache[rows.to(cache.device)] = coms[rows][:, list(body_ids), :3].to(device=cache.device, dtype=cache.dtype)
        return cache

    def _prepare_robot_collision_asset(self, source_path: str, conversion_dir: str, robot_asset_cfg):
        """Prepare an immutable URDF and expose the exact runtime asset contract."""
        profile = getattr(robot_asset_cfg, "foot_collision_profile", "source")
        if profile == "sonic_train" and not robot_asset_cfg.replace_cylinder_with_capsule:
            raise ValueError("sonic_train requires replace_cylinder_with_capsule=True, matching the official Isaac importer")
        prepared = prepare_foot_collision_urdf(
            source_path,
            os.path.join(conversion_dir, "foot_collision_assets"),
            mode=profile,
        )
        receipt = dataclasses.asdict(prepared)
        receipt["import_settings"] = {
            "replace_cylinders_with_capsules": robot_asset_cfg.replace_cylinder_with_capsule,
            "merge_fixed_joints": robot_asset_cfg.collapse_fixed_joints,
            "enabled_self_collisions": robot_asset_cfg.enable_self_collisions,
        }
        self.robot_collision_assets.append(receipt)
        logger.info(
            f"[FOOT-COLLISION] profile={prepared.profile} source={prepared.source_path} "
            f"source_sha256={prepared.source_sha256} effective={prepared.urdf_path} "
            f"effective_sha256={prepared.generated_sha256} cache_key={prepared.cache_key}"
        )
        return prepared

    def _build_multi_usd_spawn(self, robot_asset_cfg, asset_root: str, robot_rigid_props, robot_articulation_props):
        """Convert every ``asset.urdf_files`` entry to USD and build the per-env spawn cfg + the ghost template.

        * conversion: ``sim_utils.UrdfConverterCfg`` with exactly the single-URDF settings (fix_base,
          merge_fixed_joints=collapse_fixed_joints, replace_cylinders_with_capsules, force_usd_conversion=True,
          zero-gain "none" joint drive); cache dir ``<asset_root>/converted_rank<slot>/<stem>/<stem>.usd`` per file;
        * layout: ``HOLOSOMA_MULTI_USD_SPAWNER=explicit`` (default) -> ``multi_usd.plan_variant_assignment`` (weights ->
          tiling -> shuffle seeded with training seed + RANK) realised by ``ExplicitMultiUsdFileCfg``;
          ``=stock`` -> Isaac Lab ``MultiUsdFileCfg(usd_path=<weights-expanded list>, random_choice=False)`` whose
          round-robin ``k % len`` rule gives the tiled (unshuffled) layout;
        * ghosts (WBT_PLAN_GHOSTS) reuse variant 0's USD through a plain ``UsdFileCfg``.
        Returns ``(spawn_cfg, ghost_spawn_template)``.
        """
        urdf_files = [str(f) for f in robot_asset_cfg.urdf_files]
        weights = _multi_usd.validate_variant_weights(getattr(robot_asset_cfg, "urdf_variant_weights", None), len(urdf_files))
        names = _multi_usd.variant_names_for(urdf_files, getattr(robot_asset_cfg, "urdf_variant_names", None))
        conversion_dir = os.path.abspath(os.path.join(asset_root, f"converted_rank{self._usd_conversion_slot()}"))
        prepared_files = [
            self._prepare_robot_collision_asset(
                os.path.abspath(os.path.join(asset_root, f)), conversion_dir, robot_asset_cfg
            ).urdf_path
            for f in urdf_files
        ]
        # Prepared filenames include the source/dependency/profile hash. Thus
        # new foot geometry never overwrites a source-geometry USD cache slot.
        targets = _multi_usd.conversion_targets(asset_root, conversion_dir, prepared_files)

        usd_paths: list[str] = []
        for t in targets:
            if not os.path.isfile(t.urdf_path):
                raise FileNotFoundError(
                    f"[MULTI-USD] variant {t.variant_id} ({names[t.variant_id]}) URDF missing: {t.urdf_path}"
                )
            converter_cfg = sim_utils.UrdfConverterCfg(
                asset_path=t.urdf_path,
                usd_dir=t.usd_dir,
                usd_file_name=t.usd_file_name,
                fix_base=robot_asset_cfg.fix_base_link,
                merge_fixed_joints=robot_asset_cfg.collapse_fixed_joints,
                replace_cylinders_with_capsules=robot_asset_cfg.replace_cylinder_with_capsule,
                force_usd_conversion=True,
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0),
                    target_type="none",
                ),
            )
            with Timer(
                f"[MULTI-USD] URDF->USD variant {t.variant_id} ({names[t.variant_id]})",
                f"multi_usd_convert_{t.variant_id}",
            ):
                converter = sim_utils.UrdfConverter(converter_cfg)
            usd_path = os.path.abspath(converter.usd_path)
            if not os.path.isfile(usd_path):
                raise RuntimeError(f"[MULTI-USD] converter produced no file for variant {t.variant_id}: {usd_path}")
            usd_paths.append(usd_path)

        num_envs = int(self.training_config.num_envs)
        seed = _multi_usd.normalise_seed(
            getattr(self.training_config, "seed", 0), int(os.environ.get("RANK", "0") or 0)
        )
        spawner_mode = os.environ.get(_multi_usd.SPAWNER_ENV_VAR, _multi_usd.SPAWNER_EXPLICIT).strip().lower()
        if spawner_mode not in (_multi_usd.SPAWNER_EXPLICIT, _multi_usd.SPAWNER_STOCK):
            raise ValueError(
                f"{_multi_usd.SPAWNER_ENV_VAR} must be '{_multi_usd.SPAWNER_EXPLICIT}' or "
                f"'{_multi_usd.SPAWNER_STOCK}', got {spawner_mode!r}"
            )

        common = dict(
            activate_contact_sensors=True, rigid_props=robot_rigid_props, articulation_props=robot_articulation_props
        )
        if spawner_mode == _multi_usd.SPAWNER_STOCK:
            spawn, planned, layout = self._stock_multi_usd_spawn(usd_paths, weights, num_envs, common)
        else:
            from holosoma.simulator.isaacsim.multi_usd_spawner import ExplicitMultiUsdFileCfg

            planned = _multi_usd.plan_variant_assignment(num_envs, weights, seed, shuffle=True)
            env0_override = os.environ.get(_multi_usd.ENV0_VARIANT_ENV_VAR, "").strip()
            if env0_override:
                env0_vid = int(env0_override)
                if not 0 <= env0_vid < len(usd_paths):
                    raise ValueError(f"{_multi_usd.ENV0_VARIANT_ENV_VAR}={env0_override!r} is not a variant id of {list(names)}")
                planned = _multi_usd.force_env0_variant(planned, env0_vid)
                logger.warning(
                    f"[MULTI-USD] {_multi_usd.ENV0_VARIANT_ENV_VAR}={env0_vid}: env 0 forced to variant "
                    f"{names[env0_vid]!r} (smoke-test knob; startup helpers that read env 0's layout see the worst case)"
                )
            spawn = ExplicitMultiUsdFileCfg(
                usd_path=list(usd_paths), variant_ids=list(planned), random_choice=False, **common
            )
            layout = _multi_usd.LAYOUT_EXPLICIT_SHUFFLED
        ghost_spawn_template = sim_utils.UsdFileCfg(usd_path=usd_paths[0], **common)

        self.robot_variant_names = tuple(names)
        self.robot_variant_usd_paths = list(usd_paths)
        self.robot_variant_weights = tuple(weights)
        self.robot_variant_layout = layout
        self._multi_usd_planned_variant_ids = list(planned)
        self._multi_usd_common_spawn_kwargs = common
        logger.info(
            f"[MULTI-USD] {len(usd_paths)} robot variants {list(names)} weights={list(weights)} layout={layout} "
            f"seed={seed}; planned counts: "
            f"{_multi_usd.format_variant_summary(names, _multi_usd.variant_counts(planned, len(names)))}"
        )
        for vid, (name, path) in enumerate(zip(names, usd_paths, strict=True)):
            logger.info(f"[MULTI-USD]   variant {vid} {name}: {path}")
        return spawn, ghost_spawn_template

    @staticmethod
    def _stock_multi_usd_spawn(usd_paths, weights, num_envs, common):
        """Isaac Lab ``MultiUsdFileCfg(random_choice=False)``: prototype ``k % len(usd_path)`` for the k-th matched env
        (``spawn_multi_asset``), so the weights-expanded list [0,0,1,2] tiles the proportions without a shuffle."""
        pattern = _multi_usd.expand_variant_pattern(weights)
        planned = _multi_usd.tile_variant_pattern(pattern, num_envs)
        spawn = sim_utils.MultiUsdFileCfg(usd_path=[usd_paths[v] for v in pattern], random_choice=False, **common)
        return spawn, planned, _multi_usd.LAYOUT_STOCK_ROUND_ROBIN

    def _create_multi_usd_articulation(self, robot_articulation_config):
        """``Articulation(cfg)`` (which runs the spawner); if the explicit spawner raises and strict mode is off, retry
        once with the stock round-robin cfg so a node smoke still comes up (the layout change is logged as ERROR)."""
        strict = os.environ.get(_multi_usd.STRICT_ENV_VAR, "0") == "1"
        try:
            return Articulation(robot_articulation_config)
        except Exception as e:  # noqa: BLE001
            if strict or self.robot_variant_layout != _multi_usd.LAYOUT_EXPLICIT_SHUFFLED:
                raise
            logger.error(
                f"[MULTI-USD] explicit per-env spawner failed ({type(e).__name__}: {e}); retrying with the stock "
                f"Isaac Lab MultiUsdFileCfg round-robin layout (set {_multi_usd.STRICT_ENV_VAR}=1 to fail instead)"
            )
            spawn, planned, layout = self._stock_multi_usd_spawn(
                self.robot_variant_usd_paths,
                self.robot_variant_weights,
                int(self.training_config.num_envs),
                self._multi_usd_common_spawn_kwargs,
            )
            self.robot_variant_layout = layout
            self._multi_usd_planned_variant_ids = list(planned)
            return Articulation(robot_articulation_config.replace(spawn=spawn))

    def _finalize_multi_usd_variants(self) -> None:
        """Read the variant every env prim actually references back from the stage, reconcile with the plan and
        publish ``self.robot_variant_ids`` (LongTensor[num_envs] on the sim device)."""
        num_envs = int(self.training_config.num_envs)
        planned = list(self._multi_usd_planned_variant_ids)
        names = self.robot_variant_names
        strict = os.environ.get(_multi_usd.STRICT_ENV_VAR, "0") == "1"
        try:
            try:
                stage = sim_utils.utils.get_current_stage()
            except Exception:  # noqa: BLE001
                import omni.usd

                stage = omni.usd.get_context().get_stage()
            recovered = _multi_usd.recover_variant_ids_from_stage(stage, num_envs, self.robot_variant_usd_paths)
        except Exception as e:  # noqa: BLE001
            if strict:
                raise
            logger.warning(
                f"[MULTI-USD] per-env variant recovery from the stage failed ({e}); using the planned assignment"
            )
            recovered = [None] * num_envs
        res = _multi_usd.reconcile_variant_ids(planned, recovered)
        if res.n_unrecovered:
            msg = (
                f"[MULTI-USD] {res.n_unrecovered}/{num_envs} env Robot prims exposed no recognisable USD reference; "
                "their variant id falls back to the planned assignment (TODO(node-smoke): verify on the node)"
            )
            if strict:
                raise RuntimeError(msg)
            logger.warning(msg)
        if res.n_mismatch:
            msg = (
                f"[MULTI-USD] recovered variant differs from the plan on {res.n_mismatch}/{num_envs} envs "
                f"(first: {res.mismatched_envs[:8]}); the RECOVERED ids are used"
            )
            if strict:
                raise RuntimeError(msg)
            logger.warning(msg)
        else:
            logger.info(
                f"[MULTI-USD] per-env variant recovery agrees with the plan on all "
                f"{num_envs - res.n_unrecovered} recovered envs"
            )
        self.robot_variant_ids = torch.tensor(res.final, dtype=torch.long, device=self.sim_device)
        counts = _multi_usd.variant_counts(res.final, len(names))
        logger.info(
            f"[MULTI-USD] realised per-env hand variants ({self.robot_variant_layout}): "
            f"{_multi_usd.format_variant_summary(names, counts)}"
        )

    # ------------------------------------------------------------------------------------------------
    # Per-environment object variants under /World/envs/env_.*/Object.
    # Mirror of the three robot helpers above for holosoma's ``object`` RigidObject actor.
    # ------------------------------------------------------------------------------------------------
    def _build_multi_usd_object_spawn(self, object_cfg, conversion_dir: str, rigid_props, articulation_props):
        """Convert every ``object.object_urdf_paths`` entry to USD and build the per-env object spawn cfg.

        * conversion: ``sim_utils.UrdfConverterCfg`` with exactly the single-object settings of the ``UrdfFileCfg`` below
          (fix_base=False, replace_cylinders_with_capsules=True, force_usd_conversion=True, zero-gain joint drive); cache dir
          ``<conversion_dir>/<stem>/<stem>.usd`` per file (``conversion_dir`` = the per-slot ``converted_object/object_<slot>``);
        * layout: ``HOLOSOMA_MULTI_USD_SPAWNER`` explicit (default; ``plan_variant_assignment`` seeded with training seed + RANK,
          realised by ``ExplicitMultiUsdFileCfg``) or stock (``MultiUsdFileCfg(random_choice=False)`` round-robin);
          ``HOLOSOMA_OBJECT_ENV0_VARIANT=<id>`` forces env 0's variant (smoke knob, worst-case shape on env 0);
        * publishes ``object_variant_names / _usd_paths / _urdf_paths / _weights / _layout / _sizes`` (ids after the spawn).
        Returns the spawn cfg.
        """
        urdf_paths = [resolve_data_file_path(str(p)) for p in object_cfg.object_urdf_paths]
        weights = _multi_usd.validate_variant_weights(getattr(object_cfg, "object_variant_weights", None), len(urdf_paths))
        names = _multi_usd.variant_names_for(urdf_paths, getattr(object_cfg, "object_variant_names", None))
        targets = _multi_usd.object_conversion_targets(conversion_dir, urdf_paths)

        usd_paths: list[str] = []
        sizes: list[list[float]] = []
        for t in targets:
            if not os.path.isfile(t.urdf_path):
                raise FileNotFoundError(f"[MULTI-USD] object variant {t.variant_id} ({names[t.variant_id]}) URDF missing: {t.urdf_path}")
            box = _multi_usd.parse_urdf_box_size(t.urdf_path)
            sizes.append([float(v) for v in box] if box is not None else [float("nan")] * 3)
            converter_cfg = sim_utils.UrdfConverterCfg(
                asset_path=t.urdf_path,
                usd_dir=t.usd_dir,
                usd_file_name=t.usd_file_name,
                fix_base=False,
                replace_cylinders_with_capsules=True,
                force_usd_conversion=True,
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
                ),
            )
            with Timer(
                f"[MULTI-USD] object URDF->USD variant {t.variant_id} ({names[t.variant_id]})",
                f"multi_usd_object_convert_{t.variant_id}",
            ):
                converter = sim_utils.UrdfConverter(converter_cfg)
            usd_path = os.path.abspath(converter.usd_path)
            if not os.path.isfile(usd_path):
                raise RuntimeError(f"[MULTI-USD] converter produced no file for object variant {t.variant_id}: {usd_path}")
            usd_paths.append(usd_path)

        num_envs = int(self.training_config.num_envs)
        seed = _multi_usd.normalise_seed(getattr(self.training_config, "seed", 0), int(os.environ.get("RANK", "0") or 0))
        # a different stream from the robot variants (same seed + 1): the hand and the box shape must not be correlated per env
        seed = seed + 1
        spawner_mode = os.environ.get(_multi_usd.SPAWNER_ENV_VAR, _multi_usd.SPAWNER_EXPLICIT).strip().lower()
        if spawner_mode not in (_multi_usd.SPAWNER_EXPLICIT, _multi_usd.SPAWNER_STOCK):
            raise ValueError(
                f"{_multi_usd.SPAWNER_ENV_VAR} must be '{_multi_usd.SPAWNER_EXPLICIT}' or '{_multi_usd.SPAWNER_STOCK}', got {spawner_mode!r}"
            )
        common = dict(activate_contact_sensors=True, rigid_props=rigid_props, articulation_props=articulation_props)
        if spawner_mode == _multi_usd.SPAWNER_STOCK:
            spawn, planned, layout = self._stock_multi_usd_spawn(usd_paths, weights, num_envs, common)
        else:
            from holosoma.simulator.isaacsim.multi_usd_spawner import ExplicitMultiUsdFileCfg

            planned = _multi_usd.plan_variant_assignment(num_envs, weights, seed, shuffle=True)
            env0_override = os.environ.get(_multi_usd.OBJECT_ENV0_VARIANT_ENV_VAR, "").strip()
            if env0_override:
                env0_vid = int(env0_override)
                if not 0 <= env0_vid < len(usd_paths):
                    raise ValueError(
                        f"{_multi_usd.OBJECT_ENV0_VARIANT_ENV_VAR}={env0_override!r} is not an object variant id of {list(names)}"
                    )
                planned = _multi_usd.force_env0_variant(planned, env0_vid)
                logger.warning(
                    f"[MULTI-USD] {_multi_usd.OBJECT_ENV0_VARIANT_ENV_VAR}={env0_vid}: env 0's object forced to variant "
                    f"{names[env0_vid]!r} (smoke-test knob; startup helpers that read env 0's layout see this shape)"
                )
            spawn = ExplicitMultiUsdFileCfg(usd_path=list(usd_paths), variant_ids=list(planned), random_choice=False, **common)
            layout = _multi_usd.LAYOUT_EXPLICIT_SHUFFLED

        self.object_variant_names = tuple(names)
        self.object_variant_usd_paths = list(usd_paths)
        self.object_variant_urdf_paths = list(urdf_paths)
        self.object_variant_weights = tuple(weights)
        self.object_variant_layout = layout
        self.object_variant_sizes = torch.tensor(sizes, dtype=torch.float32, device=self.sim_device)
        self._multi_usd_object_planned_variant_ids = list(planned)
        self._multi_usd_object_common_spawn_kwargs = common
        logger.info(
            f"[MULTI-USD] {len(usd_paths)} object variants {list(names)} weights={list(weights)} layout={layout} seed={seed}; "
            f"planned counts: {_multi_usd.format_variant_summary(names, _multi_usd.variant_counts(planned, len(names)))}"
        )
        for vid, (name, path, size) in enumerate(zip(names, usd_paths, sizes, strict=True)):
            logger.info(f"[MULTI-USD]   object variant {vid} {name}: box {size} m  {path}")
        return spawn

    def _create_multi_usd_object(self, object_rigid_config):
        """``RigidObject(cfg)`` (runs the spawner); on a non-strict explicit-spawner failure retry once with the stock
        round-robin cfg (mirror of ``_create_multi_usd_articulation``)."""
        strict = os.environ.get(_multi_usd.STRICT_ENV_VAR, "0") == "1"
        try:
            return RigidObject(object_rigid_config)
        except Exception as e:  # noqa: BLE001
            if strict or self.object_variant_layout != _multi_usd.LAYOUT_EXPLICIT_SHUFFLED:
                raise
            logger.error(
                f"[MULTI-USD] explicit per-env OBJECT spawner failed ({type(e).__name__}: {e}); retrying with the stock "
                f"Isaac Lab MultiUsdFileCfg round-robin layout (set {_multi_usd.STRICT_ENV_VAR}=1 to fail instead)"
            )
            spawn, planned, layout = self._stock_multi_usd_spawn(
                self.object_variant_usd_paths,
                self.object_variant_weights,
                int(self.training_config.num_envs),
                self._multi_usd_object_common_spawn_kwargs,
            )
            self.object_variant_layout = layout
            self._multi_usd_object_planned_variant_ids = list(planned)
            return RigidObject(object_rigid_config.replace(spawn=spawn))

    def _finalize_multi_usd_object_variants(self) -> None:
        """Read object variants from the stage and publish object_variant_ids.

        HOLOSOMA_MULTI_USD_STRICT=1 raises if a variant cannot be recovered or differs
        from its assigned value."""
        num_envs = int(self.training_config.num_envs)
        planned = list(self._multi_usd_object_planned_variant_ids)
        names = self.object_variant_names
        strict = os.environ.get(_multi_usd.STRICT_ENV_VAR, "0") == "1"
        try:
            try:
                stage = sim_utils.utils.get_current_stage()
            except Exception:  # noqa: BLE001
                import omni.usd

                stage = omni.usd.get_context().get_stage()
            recovered = _multi_usd.recover_variant_ids_from_stage(
                stage, num_envs, self.object_variant_usd_paths, robot_prim_name=_multi_usd.OBJECT_PRIM_NAME
            )
        except Exception as e:  # noqa: BLE001
            if strict:
                raise
            logger.warning(f"[MULTI-USD] per-env OBJECT variant recovery from the stage failed ({e}); using the planned assignment")
            recovered = [None] * num_envs
        res = _multi_usd.reconcile_variant_ids(planned, recovered)
        if res.n_unrecovered:
            msg = (
                f"[MULTI-USD] {res.n_unrecovered}/{num_envs} env Object prims exposed no recognisable USD reference; "
                "their variant id falls back to the planned assignment (TODO(node-smoke): verify on the node)"
            )
            if strict:
                raise RuntimeError(msg)
            logger.warning(msg)
        if res.n_mismatch:
            msg = (
                f"[MULTI-USD] recovered OBJECT variant differs from the plan on {res.n_mismatch}/{num_envs} envs "
                f"(first: {res.mismatched_envs[:8]}); the RECOVERED ids are used"
            )
            if strict:
                raise RuntimeError(msg)
            logger.warning(msg)
        else:
            logger.info(
                f"[MULTI-USD] per-env OBJECT variant recovery agrees with the plan on all {num_envs - res.n_unrecovered} recovered envs"
            )
        self.object_variant_ids = torch.tensor(res.final, dtype=torch.long, device=self.sim_device)
        counts = _multi_usd.variant_counts(res.final, len(names))
        logger.info(
            f"[MULTI-USD] realised per-env object variants ({self.object_variant_layout}): "
            f"{_multi_usd.format_variant_summary(names, counts)}"
        )

    def _validate_articulation_tensor_devices(self) -> None:
        """Fail early if IsaacLab created actuator indices on another rank's GPU."""
        state_device = self._robot.data.joint_pos.device
        mismatches: list[str] = []
        for name, actuator in self._robot.actuators.items():
            indices = actuator.joint_indices
            # PyTorch permits CPU index tensors for CUDA data; reject only an
            # index tensor resident on a *different* CUDA rank.
            if (
                isinstance(indices, torch.Tensor)
                and indices.device.type != "cpu"
                and indices.device != state_device
            ):
                mismatches.append(f"{name}: indices={indices.device}, state={state_device}")
        if mismatches:
            details = "; ".join(mismatches)
            raise RuntimeError(
                "IsaacLab articulation has mixed-device actuator indices. "
                f"Requested Holosoma device={self.sim_device}; {details}. "
                "Ensure LOCAL_RANK is passed to setup_simulation_environment and restore the CUDA "
                "device after AppLauncher before constructing the environment."
            )

    def _setup_scene(self) -> None:
        self._load_scene_config()

        robot_asset_cfg = self.robot_config.asset
        foot_profile = getattr(robot_asset_cfg, "foot_collision_profile", "source")
        if foot_profile != "source" and robot_asset_cfg.usd_file is not None and not getattr(robot_asset_cfg, "urdf_files", None):
            raise ValueError(
                f"robot.asset.foot_collision_profile={foot_profile!r} requires URDF input; "
                "a native USD asset cannot be certified by the URDF foot transform. "
                "Select the URDF asset or explicitly preserve it with profile='source'."
            )

        asset_root = robot_asset_cfg.asset_root
        if asset_root.startswith("@holosoma/"):
            asset_root = asset_root.replace("@holosoma", get_holosoma_root())

        robot_rigid_props = sim_utils.RigidBodyPropertiesCfg(
            disable_gravity=False,
            retain_accelerations=False,
            linear_damping=robot_asset_cfg.linear_damping,
            angular_damping=robot_asset_cfg.angular_damping,
            max_linear_velocity=robot_asset_cfg.max_linear_velocity,
            max_angular_velocity=robot_asset_cfg.max_angular_velocity,
            max_depenetration_velocity=1.0,
        )

        robot_articulation_props = sim_utils.ArticulationRootPropertiesCfg(
            enabled_self_collisions=robot_asset_cfg.enable_self_collisions,
            # NOTE: (4, 0) -> (8, 4) necessary for reproducing FAR-tracking-implementation
            solver_position_iteration_count=8,
            solver_velocity_iteration_count=4,
        )

        multi_usd_files = getattr(robot_asset_cfg, "urdf_files", None)
        if multi_usd_files:
            # Convert each robot variant into its own USD, then assign variants deterministically per environment.

            spawn, ghost_spawn_template = self._build_multi_usd_spawn(
                robot_asset_cfg, asset_root, robot_rigid_props, robot_articulation_props
            )
        elif robot_asset_cfg.usd_file is None:
            # convert from urdf dynamically
            asset_path = robot_asset_cfg.urdf_file
            full_urdf_path = os.path.abspath(os.path.join(asset_root, asset_path))

            # Get a process-private conversion slot to avoid force-conversion
            # races.  LOCAL_RANK is correct under torchrun, while independent
            # single-GPU evaluators need an explicit slot without pretending to
            # be distributed (which would also change CUDA device selection).
            local_rank = int(os.environ.get("LOCAL_RANK", "0"))
            conversion_slot = os.environ.get(
                "HOLOSOMA_USD_CONVERSION_SLOT", str(local_rank)
            )
            if not conversion_slot.replace("_", "").replace("-", "").isalnum():
                raise ValueError(
                    "HOLOSOMA_USD_CONVERSION_SLOT must contain only letters, "
                    f"digits, '_' or '-', got {conversion_slot!r}"
                )
            usd_conversion_dir = os.path.abspath(
                os.path.join(asset_root, f"converted_rank{conversion_slot}")
            )
            prepared = self._prepare_robot_collision_asset(full_urdf_path, usd_conversion_dir, robot_asset_cfg)
            full_urdf_path = prepared.urdf_path
            if prepared.profile != "source":
                usd_conversion_dir = os.path.join(usd_conversion_dir, "foot_collision_usd", prepared.cache_key)

            spawn = sim_utils.UrdfFileCfg(
                usd_dir=usd_conversion_dir,
                asset_path=full_urdf_path,
                fix_base=robot_asset_cfg.fix_base_link,
                merge_fixed_joints=robot_asset_cfg.collapse_fixed_joints,
                replace_cylinders_with_capsules=robot_asset_cfg.replace_cylinder_with_capsule,
                force_usd_conversion=True,
                joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                    gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(
                        stiffness=0,
                        damping=0,
                    ),
                    target_type="none",
                ),
                activate_contact_sensors=True,
                rigid_props=robot_rigid_props,
                articulation_props=robot_articulation_props,
            )
            ghost_spawn_template = spawn
        else:
            asset_path = robot_asset_cfg.usd_file
            spawn = sim_utils.UsdFileCfg(
                usd_path=os.path.abspath(os.path.join(asset_root, asset_path)),
                activate_contact_sensors=True,
                rigid_props=robot_rigid_props,
                articulation_props=robot_articulation_props,
            )
            ghost_spawn_template = spawn

        # prepare to override the articulation configuration in
        # holosoma/holosoma/simulator/isaacsim_articulation_cfg.py
        default_joint_angles = copy.deepcopy(self.robot_config.init_state.default_joint_angles)
        # import ipdb; ipdb.set_trace()
        init_state = ArticulationCfg.InitialStateCfg(
            pos=tuple(self.robot_config.init_state.pos),
            joint_pos={joint_name: joint_angle for joint_name, joint_angle in default_joint_angles.items()},
            joint_vel={".*": 0.0},
        )

        dof_names_list = copy.deepcopy(self.robot_config.dof_names)
        # for i, name in enumerate(dof_names_list):
        #     dof_names_list[i] = name.replace("_joint", "")
        dof_effort_limit_list = self.robot_config.dof_effort_limit_list
        dof_vel_limit_list = self.robot_config.dof_vel_limit_list
        dof_armature_list = self.robot_config.dof_armature_list
        dof_joint_friction_list = self.robot_config.dof_joint_friction_list

        # get kp and kd from config
        kp_list = []
        kd_list = []
        stiffness_dict = self.robot_config.control.stiffness
        damping_dict = self.robot_config.control.damping

        for i in range(len(dof_names_list)):
            dof_names_i_without_joint = dof_names_list[i].replace("_joint", "")
            for key in stiffness_dict:
                if key in dof_names_i_without_joint:
                    kp_list.append(stiffness_dict[key])
                    kd_list.append(damping_dict[key])
                    print(f"key: {key}, kp: {stiffness_dict[key]}, kd: {damping_dict[key]}")

        # ImplicitActuatorCfg IdealPDActuatorCfg
        actuators = {
            dof_names_list[i]: IdealPDActuatorCfg(
                joint_names_expr=[dof_names_list[i]],
                effort_limit=dof_effort_limit_list[i],
                velocity_limit=dof_vel_limit_list[i],
                # effort_limit_sim=dof_effort_limit_list[i],
                # velocity_limit_sim=dof_vel_limit_list[i],
                stiffness=0,
                damping=0,
                armature=dof_armature_list[i],
                friction=dof_joint_friction_list[i],
            )
            for i in range(len(dof_names_list))
        }

        robot_articulation_config: ArticulationCfg = ARTICULATION_CFG.replace(
            prim_path="/World/envs/env_.*/Robot", spawn=spawn, init_state=init_state, actuators=actuators
        )

        # Rewards read the maximum contact force over one control step, so the
        # history must include every physics substep.
        contact_sensor_config: ContactSensorCfg = ContactSensorCfg(
            prim_path="/World/envs/env_.*/Robot/.*",
            history_length=self.contact_history_length,
            update_period=0.005,
            track_air_time=True,
            force_threshold=10.0,
            debug_vis=True,
        )

        terrain_prim_path = "/World/ground"
        height_scanner_config = None
        terrain_state = self.terrain_manager.get_state("locomotion_terrain")
        if terrain_state.mesh_type not in ["fake", None]:
            # Add a height scanner to the torso to detect the height of the terrain mesh
            # TODO: Scene USD files need ground mapping
            height_scanner_config = RayCasterCfg(
                prim_path=f"/World/envs/env_.*/Robot/{self.robot_config.body_names[0]}",
                offset=RayCasterCfg.OffsetCfg(pos=(0.0, 0.0, 0.0)),
                attach_yaw_only=True,
                # Apply a grid pattern that is smaller than the resolution to only return one height value.
                pattern_cfg=patterns.GridPatternCfg(resolution=0.1, size=[0.05, 0.05]),
                debug_vis=False,
                mesh_prim_paths=[terrain_prim_path],
            )

        global_collision_prims = []
        if terrain_state.mesh_type == "plane":
            terrain_config = TerrainImporterCfg(
                prim_path=terrain_prim_path,
                terrain_type="plane",
                collision_group=-1,
                physics_material=sim_utils.RigidBodyMaterialCfg(
                    friction_combine_mode="multiply",
                    restitution_combine_mode="multiply",
                    static_friction=terrain_state.static_friction,
                    dynamic_friction=terrain_state.dynamic_friction,
                    restitution=0.0,
                ),
                debug_vis=False,
            )
            terrain_config.num_envs = self.scene.cfg.num_envs
            terrain_config.env_spacing = self.scene.cfg.env_spacing
            terrain_config.class_type(terrain_config)
            global_collision_prims.append(terrain_config.prim_path)
        elif terrain_state.mesh_type in ["trimesh", "load_obj"]:
            self.terrain = self.terrain_manager.get_state("locomotion_terrain").terrain
            visual_material = sim_utils.PreviewSurfaceCfg(diffuse_color=(0.0, 0.0, 0.0))
            # Mesh terrain terms can set friction and restitution combine modes.
            # Without overrides, PhysX uses its default average mode.
            combine_modes = terrain_material_combine_modes(terrain_state)
            physics_material = sim_utils.RigidBodyMaterialCfg(
                static_friction=terrain_state.static_friction,
                dynamic_friction=terrain_state.dynamic_friction,
                restitution=terrain_state.restitution,
                **combine_modes,
            )

            create_prim_from_mesh(
                terrain_prim_path,
                self.terrain.mesh,
                visual_material=visual_material,
                physics_material=physics_material,
                translation=(0.0, 0.0, 0.0),
            )
            global_collision_prims.append(terrain_prim_path)
            print(
                "[INFO] Successfully created custom terrain mesh "
                f"(material combine modes: {describe_combine_modes(combine_modes)})"
            )
        else:
            raise ValueError(f"Unsupported terrain mesh type: {terrain_state.mesh_type}")

        if multi_usd_files:
            self._robot = self._create_multi_usd_articulation(robot_articulation_config)
            self._finalize_multi_usd_variants()
        else:
            self._robot = Articulation(robot_articulation_config)

        print_prim_tree("/World/envs/env_0/Robot")
        log_robot_properties("/World/envs/env_0/Robot", "*")

        self.scene.articulations["robot"] = self._robot

        self.contact_sensor = ContactSensor(contact_sensor_config)
        self.scene.sensors["contact_sensor"] = self.contact_sensor

        if height_scanner_config:
            self._height_scanner = RayCaster(height_scanner_config)
            self.scene.sensors["height_scanner"] = self._height_scanner

        # clone, filter, and replicate
        # NV4 egocentric head depth camera (gated on simulator.enable_head_depth; zero cost when off).
        if getattr(self.simulator_config.sim, "enable_head_depth", False):
            # Camera environment variables: HEAD_CAM_BODY selects the parent link; HEAD_CAM_POS
            # is a link-frame XYZ offset; HEAD_CAM_ROT is a ROS-convention wxyz quaternion;
            # HEAD_CAM_CLIP gives near/far distances in meters.


            import os as _os
            head_body = _os.environ.get("HEAD_CAM_BODY", "head_link")
            def _pf(env_key, default):
                v = _os.environ.get(env_key, "")
                try:
                    return tuple(float(x) for x in v.split(",")) if v else default
                except Exception:  # noqa: BLE001
                    return default
            # Place the camera 25 cm forward and 5 cm above head_link, tilted down about 30 degrees.
            # Camera resolution defaults to 96 × 96 pixels and can be overridden with environment variables.


            _pos = _pf("HEAD_CAM_POS", (0.25, 0.0, 0.05))
            _rot = _pf("HEAD_CAM_ROT", (0.430, -0.561, 0.561, -0.430))
            _clip = _pf("HEAD_CAM_CLIP", (0.02, 4.0))

            _wh = _pf("HEAD_CAM_WH", (96, 96))
            _W, _H = int(_wh[0]), int(_wh[1])
            # ★ optionally also render RGB from the SAME head cam (HEAD_CAM_RGB=1). Training stays depth-only
            # for speed; RGB is for visualization (what the egocentric view looks like in color).
            _data_types = ["distance_to_image_plane"]
            if _os.environ.get("HEAD_CAM_RGB", "") in ("1", "true", "True"):
                _data_types = ["distance_to_image_plane", "rgb"]
            _head_cam_cfg = TiledCameraCfg(
                prim_path=f"/World/envs/env_.*/Robot/{head_body}/head_depth_cam",
                offset=TiledCameraCfg.OffsetCfg(pos=tuple(_pos), rot=tuple(_rot), convention="ros"),
                data_types=_data_types,
                spawn=sim_utils.PinholeCameraCfg(focal_length=12.0, clipping_range=tuple(_clip)),
                width=_W, height=_H, update_period=0.0,
            )
            self._head_camera = TiledCamera(_head_cam_cfg)
            self.scene.sensors["head_depth_cam"] = self._head_camera
            # ★ optional THIRD-PERSON chase TiledCamera (CHASE_CAM=1), mounted on the same head_body but pulled
            # back/up to frame the whole robot+box. Read each step like the head cam -> guaranteed frame-synced
            # with the egocentric depth/RGB (no separate recorder cadence). RGB only; for visualization.
            if _os.environ.get("CHASE_CAM", "") in ("1", "true", "True"):
                # chase cam parented to a STABLE link. waist_yaw_link tracks the robot's horizontal yaw/position
                # but (unlike head_link) does NOT pitch with the torso lean -> far less tumbling than head/pelvis
                # during the bend-and-lift. Per-env (env_.*) so it follows EACH env's own robot (composite picks
                # a successful env). Look-at quaternion from the offline scan (3/4 side view).
                chase_body = _os.environ.get("CHASE_CAM_BODY", "waist_yaw_link")
                _cpos = _pf("CHASE_CAM_POS", (-1.8, -1.6, 1.0))
                _crot = _pf("CHASE_CAM_ROT", (0.227, 0.341, 0.759, 0.506))
                _chase_cfg = TiledCameraCfg(
                    prim_path=f"/World/envs/env_.*/Robot/{chase_body}/chase_cam",
                    offset=TiledCameraCfg.OffsetCfg(pos=tuple(_cpos), rot=tuple(_crot), convention="ros"),
                    data_types=["rgb"],
                    spawn=sim_utils.PinholeCameraCfg(focal_length=16.0, clipping_range=(0.05, 30.0)),
                    width=480, height=360, update_period=0.0,
                )
                self._chase_camera = TiledCamera(_chase_cfg)
                self.scene.sensors["chase_cam"] = self._chase_camera
            else:
                self._chase_camera = None
        else:
            self._head_camera = None
            self._chase_camera = None
        # Optional kinematic reference robots, one per keyframe listed in
        # WBT_PLAN_GHOSTS (e.g. "0,8,15"). Spawn before clone_environments so every
        # environment receives the reference prims. Ghost properties:
        #   * kinematic_enabled -> no solver cost, no gravity; they sit where the visualiser writes them;
        #   * NO contact sensors, NO actuators -- they are scenery, not agents;
        #   * NO collision with anything: same inverted-collision-group regime that isolates envs also
        #     isolates the ghosts (they are cloned per env and never added to any global collision group,
        #     and their colliders are disabled outright below for safety);
        #   * tinted semi-transparent via a per-ghost PreviewSurface so horizon position is readable
        #     (green = now -> red = far), and the LIVE robot keeps its normal appearance.
        _ghost_kfs = [k for k in os.environ.get("WBT_PLAN_GHOSTS", "").split(",") if k.strip() != ""]
        self._plan_ghosts = []
        _ghost_tints = []
        if _ghost_kfs:
            _n_g = len(_ghost_kfs)
            for _gi, _gk in enumerate(_ghost_kfs):
                _frac = _gi / max(_n_g - 1, 1)
                _tint = (min(1.0, 2.0 * _frac), min(1.0, 2.0 * (1.0 - _frac)), 0.15)
                _gspawn = copy.deepcopy(ghost_spawn_template)  # multi-USD scenes: variant 0's plain UsdFileCfg
                # ghosts reuse the SAME converted USD; a deepcopied UrdfFileCfg still carries
                # force_usd_conversion=True, which would re-run the converter once PER GHOST and rewrite
                # the very file the live robot's prim references. Reuse, never re-convert.
                if hasattr(_gspawn, "force_usd_conversion"):
                    _gspawn.force_usd_conversion = False
                _gspawn.activate_contact_sensors = False
                # ★NOT kinematic_enabled: PhysX turns each kinematic link into a STATIC body and refuses
                # to create joints between static bodies ("CreateJoint - cannot create a joint between
                # static bodies"), so the whole articulation fails to build -- hit on the second attempt.
                # kinematic is a RIGID-BODY concept (fine for the platform); an articulated ghost must stay
                # dynamic and be TAMED instead: no gravity, no colliders, fixed root, and the visualiser
                # rewrites root pose + joint state EVERY step (kinematic-by-write).
                _gspawn.rigid_props = sim_utils.RigidBodyPropertiesCfg(disable_gravity=True)
                # colliders OFF outright: filter_collisions only isolates env-from-env, so WITHIN an
                # env a ghost's colliders would shove the live robot around. Scenery must not collide.
                _gspawn.collision_props = sim_utils.CollisionPropertiesCfg(collision_enabled=False)
                _gspawn.articulation_props = sim_utils.ArticulationRootPropertiesCfg(
                    enabled_self_collisions=False, fix_root_link=False)
                _gspawn.visual_material = sim_utils.PreviewSurfaceCfg(
                    diffuse_color=_tint, opacity=0.45, roughness=0.9, metallic=0.0)
                _gcfg = ARTICULATION_CFG.replace(
                    prim_path=f"/World/envs/env_.*/PlanGhost{_gi}",
                    spawn=_gspawn,
                    init_state=ArticulationCfg.InitialStateCfg(
                        pos=(0.0, 0.0, -10.0 - 2.0 * _gi),   # parked underground until posed
                        joint_pos={jn: ja for jn, ja in default_joint_angles.items()},
                        joint_vel={".*": 0.0},
                    ),
                    actuators={},
                )
                _g = Articulation(_gcfg)
                _ghost_tints.append(_tint)
                self.scene.articulations[f"plan_ghost_{_gi}"] = _g
                self._plan_ghosts.append(_g)
            logger.info(f"[PLAN-GHOSTS] {_n_g} kinematic ghost robots created for keyframes {_ghost_kfs} "
                        f"(green->red over the horizon, opacity 0.35, no contacts, no actuators)")

        object_multi_usd = bool(getattr(self.robot_config.object, "object_urdf_paths", None))
        if multi_usd_files or object_multi_usd:
            # With heterogeneous assets, Isaac Lab clones environment Xforms before spawning.
            # Cloning again would compose env 0's asset references over each assigned variant.


            logger.info("[MULTI-USD] skipping InteractiveScene.clone_environments(): envs were cloned at scene init")
        else:
            self.scene.clone_environments(copy_from_source=False)

        # Apply ghost tint after cloning, which can make source prims instanceable.
        # Uninstance each ghost before editing and rebinding its materials.


        if _ghost_tints:
            from pxr import Usd as _Usd, UsdGeom as _UsdGeom, UsdShade as _UsdShade, Sdf as _Sdf
            _st_g = sim_utils.utils.get_current_stage()
            _n_env = self.scene.cfg.num_envs
            _tot = 0
            _nb = 0
            for _ei in range(_n_env):
                for _gi, _tint in enumerate(_ghost_tints):
                    _gp = f"/World/envs/env_{_ei}/PlanGhost{_gi}"
                    if not _st_g.GetPrimAtPath(_gp).IsValid():
                        continue
                    try:
                        sim_utils.make_uninstanceable(_gp)
                        # Bind each ghost material directly to its visual meshes with strongerThanDescendants.
                        # Ancestor bindings cannot override materials attached directly to imported meshes.


                        _mp = f"{_gp}/GhostLook"
                        _mat = _UsdShade.Material.Define(_st_g, _mp)
                        _sh_n = _UsdShade.Shader.Define(_st_g, f"{_mp}/Shader")
                        _sh_n.CreateIdAttr("UsdPreviewSurface")
                        _sh_n.CreateInput("diffuseColor", _Sdf.ValueTypeNames.Color3f).Set(Gf.Vec3f(*_tint))
                        _sh_n.CreateInput("emissiveColor", _Sdf.ValueTypeNames.Color3f).Set(
                            Gf.Vec3f(*[0.6 * _c for _c in _tint]))
                        _sh_n.CreateInput("opacity", _Sdf.ValueTypeNames.Float).Set(0.55)
                        _sh_n.CreateInput("roughness", _Sdf.ValueTypeNames.Float).Set(0.9)
                        _sh_n.CreateInput("metallic", _Sdf.ValueTypeNames.Float).Set(0.0)
                        _mat.CreateSurfaceOutput().ConnectToSource(_sh_n.ConnectableAPI(), "surface")
                        for _pr in _Usd.PrimRange(_st_g.GetPrimAtPath(_gp)):
                            if _pr.IsA(_UsdGeom.Gprim):
                                _bapi = _UsdShade.MaterialBindingAPI.Apply(_pr)
                                _bapi.Bind(_mat, _UsdShade.Tokens.strongerThanDescendants)
                                _UsdGeom.Gprim(_pr).CreateDisplayColorAttr([Gf.Vec3f(*_tint)])
                                _nb += 1
                            # belt-and-braces: also repaint any shader that is already there
                            _sh = _UsdShade.Shader(_pr)
                            if _sh:
                                for _inm, _val in (("diffuseColor", Gf.Vec3f(*_tint)),
                                                   ("diffuse_color_constant", Gf.Vec3f(*_tint)),
                                                   ("diffuse_tint", Gf.Vec3f(*_tint)),
                                                   ("emissive_color", Gf.Vec3f(*_tint)),
                                                   ("enable_emission", True),
                                                   ("emissive_intensity", 400.0)):
                                    _in = _sh.GetInput(_inm)
                                    if _in:
                                        _in.Set(_val)
                                        _tot += 1
                    except Exception as _e:  # noqa: BLE001 - visual only
                        logger.warning(f"[PLAN-GHOSTS] post-clone tint failed on {_gp}: {_e}")
            logger.info(f"[PLAN-GHOSTS] post-clone tint: {_nb} per-mesh material bindings + {_tot} shader "
                        f"inputs across {_n_env} envs x {len(_ghost_tints)} ghosts")

        if hasattr(self.simulator_config.scene, "usd_file"):
            # Activate collisions with the entire scene
            global_collision_prims.append("/World/scene")

        self.scene.filter_collisions(global_prim_paths=global_collision_prims)

        # Add objects from one URDF or the object_urdf_paths variants.
        if self.robot_config.object.object_urdf_path or object_multi_usd:
            # Resolve the object asset urdf path using importlib.resources (single-object path; the variants resolve their own)
            object_asset_urdf_path = (
                resolve_data_file_path(self.robot_config.object.object_urdf_path) if not object_multi_usd else None
            )
            # IsaacLab's default URDF conversion directory includes a process-local
            # pseudo-random suffix.  Independent evaluators launched in the same
            # second with the same seed can therefore collide even when they use
            # different GPUs.  Bind object conversion to the same validated
            # process slot used by the robot conversion, but keep its directory
            # distinct from both the robot and every other object task.
            explicit_object_conversion_slot = os.environ.get(
                "HOLOSOMA_USD_CONVERSION_SLOT"
            )
            if explicit_object_conversion_slot is None:
                object_conversion_slot = os.environ.get("LOCAL_RANK", "0")
                object_conversion_slot_source = "LOCAL_RANK_OR_ZERO_FALLBACK"
            else:
                object_conversion_slot = explicit_object_conversion_slot
                object_conversion_slot_source = "HOLOSOMA_USD_CONVERSION_SLOT"
            if not object_conversion_slot.replace("_", "").replace("-", "").isalnum():
                raise ValueError(
                    "HOLOSOMA_USD_CONVERSION_SLOT must contain only letters, "
                    f"digits, '_' or '-', got {object_conversion_slot!r}"
                )
            object_conversion_root_value = os.environ.get(
                "HOLOSOMA_USD_CONVERSION_ROOT"
            )
            if object_conversion_root_value is None:
                object_conversion_root = os.path.abspath(
                    os.path.join(asset_root, "converted_object")
                )
                object_conversion_root_source = "ASSET_ROOT_FALLBACK"
            else:
                if not os.path.isabs(object_conversion_root_value):
                    raise ValueError(
                        "HOLOSOMA_USD_CONVERSION_ROOT must be an absolute path, "
                        f"got {object_conversion_root_value!r}"
                    )
                object_conversion_root = os.path.realpath(
                    object_conversion_root_value
                )
                object_conversion_root_source = "HOLOSOMA_USD_CONVERSION_ROOT"
            object_usd_conversion_dir = os.path.realpath(
                os.path.join(
                    object_conversion_root,
                    f"object_{object_conversion_slot}",
                )
            )
            if os.path.commonpath(
                [object_conversion_root, object_usd_conversion_dir]
            ) != object_conversion_root:
                raise ValueError("object USD conversion directory escaped its root")

            def _run_object_usd_conversion_barrier(
                slot,
                barrier_root_value,
                expected_parties_value,
                participants_value,
                run_token_value,
                timeout_s_value,
            ):
                """Synchronize only an explicitly requested conversion preflight."""

                values = (
                    barrier_root_value,
                    expected_parties_value,
                    participants_value,
                    run_token_value,
                    timeout_s_value,
                )
                if all(value is None for value in values):
                    return {
                        "schema": "holosoma/object-usd-conversion-barrier/v1",
                        "enabled": False,
                    }
                if any(value is None for value in values):
                    raise ValueError(
                        "object USD conversion barrier requires root, expected "
                        "parties, participant IDs, run token, and timeout together"
                    )
                if object_conversion_slot_source != "HOLOSOMA_USD_CONVERSION_SLOT":
                    raise ValueError(
                        "object USD conversion barrier requires an explicit slot"
                    )
                if not os.path.isabs(barrier_root_value):
                    raise ValueError(
                        "HOLOSOMA_USD_CONVERSION_BARRIER_ROOT must be absolute"
                    )
                barrier_root_absolute = os.path.abspath(barrier_root_value)
                barrier_root = os.path.realpath(barrier_root_value)
                if (
                    barrier_root_absolute != barrier_root
                    or os.path.islink(barrier_root_value)
                    or not os.path.isdir(barrier_root)
                ):
                    raise ValueError(
                        "object USD conversion barrier root must be an existing "
                        "non-symlink directory"
                    )
                try:
                    expected_parties = int(expected_parties_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "object USD conversion barrier party count must be an integer"
                    ) from exc
                if expected_parties != 2:
                    raise ValueError(
                        "object USD conversion preflight requires exactly two parties"
                    )
                participants = participants_value.split(",")
                if (
                    len(participants) != expected_parties
                    or any(not participant for participant in participants)
                    or len(set(participants)) != expected_parties
                    or participants != sorted(participants)
                    or any(
                        not participant.replace("_", "").replace("-", "").isalnum()
                        for participant in participants
                    )
                    or slot not in participants
                ):
                    raise ValueError(
                        "object USD conversion barrier participants must be the "
                        "sorted exact two registered slots"
                    )
                participants_sha256 = __import__("hashlib").sha256(
                    ",".join(participants).encode("utf-8")
                ).hexdigest()
                if (
                    len(run_token_value) != 64
                    or any(character not in "0123456789abcdef" for character in run_token_value)
                ):
                    raise ValueError(
                        "object USD conversion barrier run token must be 64 lowercase hex"
                    )
                try:
                    timeout_s = float(timeout_s_value)
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        "object USD conversion barrier timeout must be numeric"
                    ) from exc
                if not 0.0 < timeout_s <= 60.0:
                    raise ValueError(
                        "object USD conversion barrier timeout must lie in (0, 60] s"
                    )

                def _barrier_marker_receipts():
                    names = []
                    expected_marker_names = {
                        f"party_{participant}.ready"
                        for participant in participants
                    }
                    allowed_temporary_prefixes = tuple(
                        f"party_{participant}." for participant in participants
                    )
                    with os.scandir(barrier_root) as entries:
                        for entry in entries:
                            if entry.is_symlink() or not entry.is_file(
                                follow_symlinks=False
                            ):
                                raise ValueError(
                                    "object USD conversion barrier contains a "
                                    "symlink or non-file entry"
                                )
                            if (
                                entry.name.endswith(".tmp")
                                and entry.name.startswith(allowed_temporary_prefixes)
                            ):
                                continue
                            if entry.name not in expected_marker_names:
                                raise ValueError(
                                    "object USD conversion barrier contains a stale "
                                    "or unregistered participant entry"
                                )
                            names.append(entry.name)
                    receipts = []
                    for name in sorted(names):
                        marker_path_to_read = os.path.join(barrier_root, name)
                        if os.path.islink(marker_path_to_read):
                            raise ValueError(
                                "object USD conversion barrier marker became a symlink"
                            )
                        with open(marker_path_to_read, "rb") as marker_stream:
                            marker_bytes = marker_stream.read()
                        try:
                            marker_fields = dict(
                                line.split("=", 1)
                                for line in marker_bytes.decode("utf-8").splitlines()
                            )
                        except (UnicodeDecodeError, ValueError) as exc:
                            raise ValueError(
                                "object USD conversion barrier marker payload is malformed"
                            ) from exc
                        participant = name[len("party_") : -len(".ready")]
                        if (
                            set(marker_fields)
                            != {
                                "slot",
                                "pid",
                                "arrival_ns",
                                "run_token",
                                "participants_sha256",
                            }
                            or marker_fields["slot"] != participant
                            or marker_fields["run_token"] != run_token_value
                            or marker_fields["participants_sha256"]
                            != participants_sha256
                            or int(marker_fields["pid"]) <= 0
                            or int(marker_fields["arrival_ns"]) <= 0
                        ):
                            raise ValueError(
                                "object USD conversion barrier marker identity/token drift"
                            )
                        receipts.append(
                            {
                                "name": name,
                                "slot": participant,
                                "pid": int(marker_fields["pid"]),
                                "arrival_monotonic_ns": int(
                                    marker_fields["arrival_ns"]
                                ),
                                "run_token": marker_fields["run_token"],
                                "participants_sha256": marker_fields[
                                    "participants_sha256"
                                ],
                                "sha256": __import__("hashlib").sha256(
                                    marker_bytes
                                ).hexdigest(),
                                "size_bytes": len(marker_bytes),
                            }
                        )
                    return receipts

                marker_name = f"party_{slot}.ready"
                marker_path = os.path.join(barrier_root, marker_name)
                initial_receipts = _barrier_marker_receipts()
                initial_names = [row["name"] for row in initial_receipts]
                if marker_name in initial_names or len(initial_names) >= expected_parties:
                    raise ValueError(
                        "object USD conversion barrier contains a stale party marker"
                    )
                arrival_ns = __import__("time").monotonic_ns()
                temporary_name = f"party_{slot}.{os.getpid()}.tmp"
                temporary_path = os.path.join(barrier_root, temporary_name)
                flags = os.O_WRONLY | os.O_CREAT | os.O_EXCL
                flags |= getattr(os, "O_NOFOLLOW", 0)
                marker_fd = os.open(temporary_path, flags, 0o600)
                try:
                    marker_payload = (
                        f"slot={slot}\npid={os.getpid()}\narrival_ns={arrival_ns}\n"
                        f"run_token={run_token_value}\n"
                        f"participants_sha256={participants_sha256}\n"
                    ).encode("utf-8")
                    os.write(marker_fd, marker_payload)
                    os.fsync(marker_fd)
                finally:
                    os.close(marker_fd)
                os.link(temporary_path, marker_path, follow_symlinks=False)
                os.unlink(temporary_path)

                deadline = __import__("time").monotonic() + timeout_s
                while True:
                    marker_receipts = _barrier_marker_receipts()
                    marker_names = [row["name"] for row in marker_receipts]
                    if len(marker_receipts) == expected_parties:
                        break
                    if len(marker_receipts) > expected_parties:
                        raise ValueError(
                            "object USD conversion barrier party denominator drift"
                        )
                    if __import__("time").monotonic() >= deadline:
                        raise TimeoutError(
                            "object USD conversion barrier timed out before two parties"
                        )
                    __import__("time").sleep(0.005)
                released_ns = __import__("time").monotonic_ns()
                return {
                    "schema": "holosoma/object-usd-conversion-barrier/v1",
                    "enabled": True,
                    "root": barrier_root,
                    "root_source": "HOLOSOMA_USD_CONVERSION_BARRIER_ROOT",
                    "expected_parties": expected_parties,
                    "participants": participants,
                    "participants_sha256": participants_sha256,
                    "run_token": run_token_value,
                    "timeout_s": timeout_s,
                    "slot": slot,
                    "marker_name": marker_name,
                    "arrival_monotonic_ns": arrival_ns,
                    "released_monotonic_ns": released_ns,
                    "observed_marker_names": marker_names,
                    "marker_receipts": marker_receipts,
                    "contract_passed": True,
                }

            object_name = "object"  # hardcoded object name
            # The (optional) two-party conversion preflight barrier runs BEFORE any object URDF->USD conversion on both paths:
            # the single path converts inside RigidObject(object_cfg) below (unchanged), the variant path converts the K files
            # while building its spawn cfg (_build_multi_usd_object_spawn), so the barrier is taken here, ahead of both.
            object_conversion_barrier_contract = (
                _run_object_usd_conversion_barrier(
                    object_conversion_slot,
                    os.environ.get("HOLOSOMA_USD_CONVERSION_BARRIER_ROOT"),
                    os.environ.get(
                        "HOLOSOMA_USD_CONVERSION_BARRIER_EXPECTED_PARTIES"
                    ),
                    os.environ.get(
                        "HOLOSOMA_USD_CONVERSION_BARRIER_PARTICIPANTS"
                    ),
                    os.environ.get("HOLOSOMA_USD_CONVERSION_BARRIER_RUN_TOKEN"),
                    os.environ.get("HOLOSOMA_USD_CONVERSION_BARRIER_TIMEOUT_S"),
                )
            )
            object_rigid_props = sim_utils.RigidBodyPropertiesCfg(
                disable_gravity=False,
                retain_accelerations=False,
                linear_damping=0.01,
                angular_damping=0.01,
                max_linear_velocity=1000.0,
                max_angular_velocity=1000.0,
                max_depenetration_velocity=1.0,
            )
            object_articulation_props = sim_utils.ArticulationRootPropertiesCfg(
                enabled_self_collisions=True,
                solver_position_iteration_count=8,
                solver_velocity_iteration_count=4,
            )
            if object_multi_usd:
                # Convert object variants into per-stem USD directories and assign them per environment.

                object_cfg = RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/Object",
                    spawn=self._build_multi_usd_object_spawn(
                        self.robot_config.object, object_usd_conversion_dir, object_rigid_props, object_articulation_props
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(0.0, 0.0, 0.5),
                    ),
                )
            else:
                object_cfg = RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/Object",
                    spawn=sim_utils.UrdfFileCfg(
                        usd_dir=object_usd_conversion_dir,
                        force_usd_conversion=True,
                        fix_base=False,
                        replace_cylinders_with_capsules=True,
                        asset_path=object_asset_urdf_path,
                        activate_contact_sensors=True,
                        rigid_props=object_rigid_props,
                        articulation_props=object_articulation_props,
                        joint_drive=sim_utils.UrdfConverterCfg.JointDriveCfg(
                            gains=sim_utils.UrdfConverterCfg.JointDriveCfg.PDGainsCfg(stiffness=0, damping=0)
                        ),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(
                        pos=(0.0, 0.0, 0.5),
                    ),
                )
            object_conversion_started_monotonic_ns = __import__(
                "time"
            ).monotonic_ns()
            if object_multi_usd:
                self._object = self._create_multi_usd_object(object_cfg)
                self._finalize_multi_usd_object_variants()
            else:
                self._object = RigidObject(object_cfg)
            object_conversion_completed_monotonic_ns = __import__(
                "time"
            ).monotonic_ns()
            self._object_usd_conversion_contract = {
                "schema": "holosoma/object-urdf-usd-conversion/v1",
                "slot": object_conversion_slot,
                "slot_source": object_conversion_slot_source,
                "root": object_conversion_root,
                "root_source": object_conversion_root_source,
                "usd_dir": object_usd_conversion_dir,
                "asset_path": os.path.realpath(object_asset_urdf_path) if object_asset_urdf_path is not None else None,
                # Per-environment object variants; None for a single object.
                "variants": (
                    {
                        "names": list(self.object_variant_names),
                        "urdf_paths": [os.path.realpath(p) for p in self.object_variant_urdf_paths],
                        "usd_paths": list(self.object_variant_usd_paths),
                        "weights": list(self.object_variant_weights),
                        "layout": self.object_variant_layout,
                        "box_sizes_m": self.object_variant_sizes.tolist(),
                        "env0_variant_override": os.environ.get(_multi_usd.OBJECT_ENV0_VARIANT_ENV_VAR) or None,
                        "strict": os.environ.get(_multi_usd.STRICT_ENV_VAR, "0") == "1",
                    }
                    if object_multi_usd
                    else None
                ),
                "force_usd_conversion": True,
                "conversion_clock": "time.monotonic_ns",
                "conversion_started_monotonic_ns": (
                    object_conversion_started_monotonic_ns
                ),
                "conversion_completed_monotonic_ns": (
                    object_conversion_completed_monotonic_ns
                ),
                "preflight_barrier": object_conversion_barrier_contract,
                "slot_is_validated": True,
                "formal_explicit_env_contract": (
                    object_conversion_slot_source
                    == "HOLOSOMA_USD_CONVERSION_SLOT"
                    and object_conversion_root_source
                    == "HOLOSOMA_USD_CONVERSION_ROOT"
                ),
            }
            logger.info(
                "[object USD conversion] "
                f"slot={object_conversion_slot} "
                f"usd_dir={object_usd_conversion_dir} "
                f"asset={object_asset_urdf_path if not object_multi_usd else f'{len(self.object_variant_names)} variants {list(self.object_variant_names)}'}"
            )
            self.scene.rigid_objects[object_name] = self._object

        # WBT_PLATFORM creates kinematic start/goal supports for elevated object references.
        # WBT_PLATFORM_EXTENSION adds a slab extending away from the robot. The training loop
        # hides start supports after pickup and enables goal supports near placement.


        import os as _os
        _plat = _os.environ.get("WBT_PLATFORM", "")
        if _plat:
            _sx, _sy, _sz = [float(v) for v in _plat.split(",")]
            _platform_defs = [
                ("platform", (_sx, _sy, _sz), (0.45, 0.35, 0.25)),
                ("platform_start", (_sx, _sy, _sz), (0.35, 0.42, 0.30)),
            ]
            _extension = _os.environ.get("WBT_PLATFORM_EXTENSION", "")
            if _extension:
                _ex, _ey, _ez = [float(v) for v in _extension.split(",")]
                _platform_defs.extend([
                    ("platform_extension", (_ex, _ey, _ez), (0.58, 0.43, 0.27)),
                    ("platform_start_extension", (_ex, _ey, _ez), (0.43, 0.55, 0.34)),
                ])
            _pnames = tuple(v[0] for v in _platform_defs)
            for _pname, _psize, _pcolor in _platform_defs:
                _pcfg = RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/{_pname.title().replace('_','')}",
                    spawn=sim_utils.CuboidCfg(
                        size=_psize,
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True,
                                                                     disable_gravity=True),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        visual_material=sim_utils.PreviewSurfaceCfg(diffuse_color=_pcolor),
                    ),
                    # spawn far below ground; the trainer teleports it up only when an episode needs it
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -5.0)),
                )
                self.scene.rigid_objects[_pname] = RigidObject(_pcfg)
            # Preserve the historical USD relation for compatibility.  Under replicate_physics it is
            # empirically inert, so runtime geometry MUST be safe for solid robot contact; that is why the
            # core is tight, the extension points away, and both pieces are phase-gated.
            self._filter_platform_robot_pairs(_pnames)

        # WBT_PLATFORM_PIECES="sx,sy,sz;..." creates independent kinematic cuboids.
        # Pieces are parked below ground until positioned to form a table or other support.


        _piece_spec = _os.environ.get("WBT_PLATFORM_PIECES", "")
        if _piece_spec:
            _piece_colors = [(0.45, 0.35, 0.25), (0.40, 0.31, 0.22), (0.40, 0.31, 0.22),
                             (0.35, 0.42, 0.30), (0.31, 0.38, 0.27), (0.31, 0.38, 0.27),
                             (0.85, 0.25, 0.20), (0.58, 0.43, 0.27)]
            _piece_names = []
            for _pi, _one in enumerate(_piece_spec.split(";")):
                _sx, _sy, _sz = [float(v) for v in _one.split(",")]
                _pname = f"platform_piece_{_pi}"
                _pcfg = RigidObjectCfg(
                    prim_path=f"/World/envs/env_.*/PlatformPiece{_pi}",
                    spawn=sim_utils.CuboidCfg(
                        size=(_sx, _sy, _sz),
                        rigid_props=sim_utils.RigidBodyPropertiesCfg(kinematic_enabled=True,
                                                                     disable_gravity=True),
                        collision_props=sim_utils.CollisionPropertiesCfg(),
                        visual_material=sim_utils.PreviewSurfaceCfg(
                            diffuse_color=_piece_colors[_pi % len(_piece_colors)]),
                    ),
                    init_state=RigidObjectCfg.InitialStateCfg(pos=(0.0, 0.0, -5.0 - _pi)),
                )
                self.scene.rigid_objects[_pname] = RigidObject(_pcfg)
                _piece_names.append(_pname)
            self._filter_platform_robot_pairs(tuple(_piece_names))

        # add lights
        # light_config = sim_utils.DomeLightCfg(intensity=2000.0, color=(0.98, 0.95, 0.88))
        # light_config.func("/World/Light", light_config)

        light_config1 = sim_utils.DomeLightCfg(
            intensity=1000.0,
            color=(0.98, 0.95, 0.88),
        )
        light_config1.func("/World/DomeLight", light_config1, translation=(1, 0, 10))

    def _filter_platform_robot_pairs(self, platform_names):
        """Write the platform-to-robot USD filtered-pairs relation.

        Replicated Isaac Lab scenes use collision groups, so this relation does not
        disable support collisions there. Supports must remain clear of the robot."""
        try:
            from pxr import UsdPhysics
            import omni.usd
        except Exception as e:  # noqa: BLE001
            logger.warning(f"[WBT_PLATFORM] pxr/omni.usd unavailable; platform<->robot NOT filtered ({e})")
            return
        stage = omni.usd.get_context().get_stage()
        n_env = self.scene.cfg.num_envs
        n_ok = 0
        for i in range(n_env):
            robot_path = f"/World/envs/env_{i}/Robot"
            if not stage.GetPrimAtPath(robot_path).IsValid():
                continue
            for _pname in platform_names:
                plat_path = f"/World/envs/env_{i}/{_pname.title().replace('_', '')}"
                prim = stage.GetPrimAtPath(plat_path)
                if not prim.IsValid():
                    continue
                # UsdPhysics.FilteredPairsAPI (verified on IsaacSim 5.1): apply on the platform prim, add
                # the env's Robot root as a filtered target -> PhysX skips that collision pair (symmetric).
                api = UsdPhysics.FilteredPairsAPI.Apply(prim)
                rel = api.GetFilteredPairsRel() or api.CreateFilteredPairsRel()
                rel.AddTarget(robot_path)
                n_ok += 1
        logger.info(f"[WBT_PLATFORM] wrote platform<->robot filteredPairs on {n_ok} prims across {n_env} "
                    "envs (do not assume this is active under replicate_physics; supports remain SOLID)")

    def get_depth_image(self, near=0.2, far=3.0):
        """Egocentric head depth normalized [0,1] (near->0,far->1), nan/inf->far. Returns [N,1,H,W]."""
        import torch as _t
        assert getattr(self, "_head_camera", None) is not None, "head depth cam not enabled (simulator.enable_head_depth=True)"
        d = self._head_camera.data.output["distance_to_image_plane"]
        if d.dim() == 4:
            d = d.squeeze(-1)
        d = _t.nan_to_num(d, nan=far, posinf=far, neginf=far).clamp(near, far)
        d = (d - near) / (far - near)
        return d.unsqueeze(1)

    def get_rgb_image(self):
        """Egocentric head RGB (requires HEAD_CAM_RGB=1 at scene build). Returns [N,H,W,3] uint8 or None."""
        cam = getattr(self, "_head_camera", None)
        if cam is None or "rgb" not in cam.data.output:
            return None
        return cam.data.output["rgb"][..., :3]    # drop alpha if present

    def update_chase_cam(self, env_id=0, dist=2.4, height=1.1, azim_deg=35.0):
        """Reposition the chase cam to a STABLE WORLD-oriented 3/4 view of robot `env_id` (tracks position,
        fixed world orientation via look-at) — avoids the tumbling you get when the cam is parented to a
        rotating link. Call each step BEFORE get_chase_image. Mirrors video_recorder._update_camera_position."""
        cam = getattr(self, "_chase_camera", None)
        if cam is None:
            return
        try:
            import torch as _t, math as _m
            from isaaclab.utils.math import create_rotation_matrix_from_view, quat_from_matrix
            root = self.robot_root_states[env_id, :3]                       # world pos of robot root
            a = _m.radians(azim_deg)
            off = _t.tensor([-dist * _m.cos(a), -dist * _m.sin(a), height], device=root.device, dtype=_t.float32)
            pos = (root + off).unsqueeze(0)
            tgt = (root + _t.tensor([0., 0., 0.3], device=root.device)).unsqueeze(0)   # aim at torso height
            ori = quat_from_matrix(create_rotation_matrix_from_view(pos, tgt, "Z", device=root.device))
            cam.set_world_poses(pos, ori, env_ids=_t.tensor([0], device=root.device))
        except Exception:  # noqa: BLE001
            pass  # if the API differs, fall back to the static parented pose

    def get_chase_image(self):
        """Third-person chase RGB (requires CHASE_CAM=1 at scene build). Returns [N,H,W,3] uint8 or None.
        Read per-step like the head cam -> frame-synced with egocentric depth/RGB (no recorder cadence).
        Call update_chase_cam() first for a stable world-oriented view."""
        cam = getattr(self, "_chase_camera", None)
        if cam is None or "rgb" not in cam.data.output:
            return None
        return cam.data.output["rgb"][..., :3]

    def _get_base_body_name(self, preference_order: list[str]) -> str:
        """Get the base body name with fallback logic.

        Args:
            preference_order: List of body names to try in order

        Returns:
            The first body name found in the robot's body list

        Raises:
            ValueError: If none of the preferred body names are found
        """
        _, body_names = self._robot.find_bodies(self.robot_config.body_names, preserve_order=True)

        for preferred_name in preference_order:
            if preferred_name in body_names:
                return preferred_name

        raise ValueError(
            f"None of the preferred base body names {preference_order} found in robot body names: {body_names}"
        )

    def get_supported_scene_formats(self) -> list[str]:
        """See base class.

        IsaacSim-specific notes:
        - Supports USD only currently

        Returns
        -------
        List[str]
            ["usd" ]
        """
        return ["usd"]

    def set_headless(self, headless):
        # call super
        super().set_headless(headless)
        if not self.headless:
            from isaacsim.util.debug_draw import _debug_draw

            self.draw = _debug_draw.acquire_debug_draw_interface()
        else:
            self.draw = None

    def _load_scene_config(self) -> None:
        """Load scene files and individual rigid objects."""
        if self.simulator_config.scene is None:
            return

        scene_config = self.simulator_config.scene

        # Load scene files (USD/URDF scene files as collections) - NEW APPROACH
        if scene_config.scene_files is not None:
            self._load_scene_files(scene_config)

        # Load individual rigid objects
        if scene_config.rigid_objects is not None:
            self._load_rigid_objects(scene_config)

    def _load_scene_files(self, scene_config: SceneConfig) -> None:
        """Load scene files (USD/URDF scene files as collections).

        Loads scene files as collections using the USDFileLoader. This is the new
        approach that replaces direct USD file loading with a more flexible system
        that supports multiple scene file formats.

        Parameters
        ----------
        scene_config : SceneConfig
            Scene configuration containing scene files and asset root path

        Raises
        ------
        ValueError
            If scene_files is an empty list
        """
        if not scene_config.scene_files:  # Empty list
            raise ValueError("scene.scene_files is empty list - remove field or provide scene files")

        usd_loader = USDFileLoader(self.sim, self.scene, self.sim_device)
        scene_collection = usd_loader.load_scene_files(scene_config.scene_files, scene_config.asset_root)

        if scene_collection is not None:
            self.scene.rigid_objects["usd_scene_objects"] = scene_collection

    def _load_rigid_objects(self, scene_config: SceneConfig) -> None:
        """Load individual rigid objects from configuration.

        Loads individual rigid objects using the USDFileLoader and adds them
        to the scene using their configuration names as keys.

        Parameters
        ----------
        scene_config : SceneConfig
            Scene configuration containing rigid objects and asset root path

        Raises
        ------
        ValueError
            If rigid_objects is an empty list
        """
        if not scene_config.rigid_objects:  # Empty list
            raise ValueError("scene.rigid_objects is empty list - remove field or provide objects")

        usd_loader = USDFileLoader(self.sim, self.scene, self.sim_device)
        individual_objects = usd_loader.load_rigid_objects(scene_config.rigid_objects, scene_config.asset_root)

        # Add individual objects to scene using direct config names
        for obj_name, rigid_object in individual_objects.items():
            self.scene.rigid_objects[obj_name] = rigid_object

    def setup(self):
        self.sim_dt = 1.0 / self.simulator_config.sim.fps

    def setup_terrain(self):
        pass

    def load_assets(self):
        """
        save self.num_dofs, self.num_bodies, self.dof_names, self.body_names in simulator class
        """

        dof_names_list = copy.deepcopy(self.robot_config.dof_names)
        # for i, name in enumerate(dof_names_list):
        #     dof_names_list[i] = name.replace("_joint", "")
        # isaacsim only support matching joint names without "joint" postfix

        # init_state=ArticulationCfg.InitialStateCfg(
        #     pos=(0.0, 0.0, 1.05),
        #     joint_pos={
        #         ".*_hip_yaw": 0.0,
        #         ".*_hip_roll": 0.0,
        #         ".*_hip_pitch": -0.28,  # -16 degrees
        #         ".*_knee": 0.79,  # 45 degrees
        #         ".*_ankle": -0.52,  # -30 degrees
        #         "torso": 0.0,
        #         ".*_shoulder_pitch": 0.28,
        #         ".*_shoulder_roll": 0.0,
        #         ".*_shoulder_yaw": 0.0,
        #         ".*_elbow": 0.52,
        #     },
        #     joint_vel={".*": 0.0},
        # ),

        # spawn=sim_utils.UsdFileCfg(
        #     usd_path=f"{ISAACLAB_NUCLEUS_DIR}/Robots/Unitree/G1/g1.usd",
        #     activate_contact_sensors=True,
        #     rigid_props=sim_utils.RigidBodyPropertiesCfg(
        #         disable_gravity=False,
        #         retain_accelerations=False,
        #         linear_damping=0.0,
        #         angular_damping=0.0,
        #         max_linear_velocity=1000.0,
        #         max_angular_velocity=1000.0,
        #         max_depenetration_velocity=1.0,
        #     ),
        #     articulation_props=sim_utils.ArticulationRootPropertiesCfg(
        #         enabled_self_collisions=False, solver_position_iteration_count=8, solver_velocity_iteration_count=4
        #     ),
        # ),

        self.dof_ids, self.dof_names = self._robot.find_joints(dof_names_list, preserve_order=True)
        self.body_ids, self.body_names = self._robot.find_bodies(self.robot_config.body_names, preserve_order=True)

        self._body_list = self.body_names.copy()
        # dof_ids and body_ids is convert dfs order (isaacsim) to dfs order (isaacgym, holosoma config)
        # i.e., bfs_order_tensor = dfs_order_tensor[dof_ids]

        # add joint names with "joint" postfix
        # for i, name in enumerate(self.dof_names):
        #     self.dof_names[i] = name + "_joint"
        """
        ipdb> self._robot.find_bodies(robot_config.body_names, preserve_order=True)
        ([0, 1, 4, 8, 12, 16, 2, 5, 9, 13, 17, 3, 6, 10, 14, 18, 7, 11, 15, 19],
        ['pelvis', 'left_hip_yaw_link', 'left_hip_roll_link', 'left_hip_pitch_link', 'left_knee_link',
        'left_ankle_link', 'right_hip_yaw_link', 'right_hip_roll_link', 'right_hip_pitch_link',
        'right_knee_link', 'right_ankle_link', 'torso_link', 'left_shoulder_pitch_link',
        'left_shoulder_roll_link', 'left_shoulder_yaw_link', 'left_elbow_link', 'right_shoulder_pitch_link',
        'right_shoulder_roll_link', 'right_shoulder_yaw_link', 'right_elbow_link'])
        ipdb> self._robot.find_bodies(robot_config.body_names, preserve_order=False)
        ([0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19],
        ['pelvis', 'left_hip_yaw_link', 'right_hip_yaw_link', 'torso_link', 'left_hip_roll_link',
        'right_hip_roll_link', 'left_shoulder_pitch_link', 'right_shoulder_pitch_link', 'left_hip_pitch_link',
        'right_hip_pitch_link', 'left_shoulder_roll_link', 'right_shoulder_roll_link', 'left_knee_link',
        'right_knee_link', 'left_shoulder_yaw_link', 'right_shoulder_yaw_link', 'left_ankle_link',
        'right_ankle_link', 'left_elbow_link', 'right_elbow_link'])
        """

        self.num_dof = len(self.dof_ids)
        self.num_bodies = len(self.body_ids)

        # warning if the dof_ids order does not match the joint_names order in robot_config
        if self.dof_ids != list(range(self.num_dof)):
            logger.warning(
                "The order of the joint_names in the robot_config does not match the "
                "order of the joint_ids in IsaacSim."
            )

        # assert if  aligns with config
        assert self.num_dof == len(self.robot_config.dof_names), "Number of DOFs must be equal to number of actions"
        assert self.num_bodies == len(self.robot_config.body_names), (
            "Number of bodies must be equal to number of body names"
        )
        # import ipdb; ipdb.set_trace()
        assert self.dof_names == self.robot_config.dof_names, "DOF names must match the config"
        assert self.body_names == self.robot_config.body_names, "Body names must match the config"

        # IsaacLab's SensorBase._initialize_callback SWALLOWS any init exception into
        # builtins.ISAACLAB_CALLBACK_EXCEPTION and still marks the sensor initialized, so a failed contact
        # sensor surfaces here as an unrelated "'NoneType' has no attribute 'prim_paths'".
        # Re-raise the initialization exception to retain its cause.
        if getattr(self.contact_sensor, "_body_physx_view", None) is None:
            _cbe = getattr(builtins, "ISAACLAB_CALLBACK_EXCEPTION", None)
            if _cbe is None:
                # nothing was stashed either -> the play callback never fired for this sensor. Run the init
                # in the open: it either raises the real cause, or it succeeds and we carry on.
                logger.warning("[CONTACT] sensor uninitialised and no stashed exception; re-running init")
                self.contact_sensor._is_initialized = False
                self.contact_sensor._initialize_impl()
                self.contact_sensor._is_initialized = True
            if getattr(self.contact_sensor, "_body_physx_view", None) is None:
                raise RuntimeError(
                    f"contact sensor never built its PhysX view; IsaacLab swallowed: {_cbe!r}") from _cbe
        self._contact_to_robot_body_ids = torch.tensor(
            [self.contact_sensor.body_names.index(body_name) for body_name in self.body_names],
            device=self.sim_device,
        )

        # return self.num_dof, self.num_bodies, self.dof_names, self.body_names

    def create_envs(self, num_envs, env_origins, base_init_state):
        self.num_envs = num_envs
        self.env_origins = env_origins
        self.base_init_state = base_init_state

        return self.scene, self._robot

    def get_dof_limits_properties(self):
        self.hard_dof_pos_limits = torch.zeros(
            self.num_dof, 2, dtype=torch.float, device=self.sim_device, requires_grad=False
        )
        self.dof_pos_limits = torch.zeros(
            self.num_dof, 2, dtype=torch.float, device=self.sim_device, requires_grad=False
        )
        self.dof_vel_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False)
        self.torque_limits = torch.zeros(self.num_dof, dtype=torch.float, device=self.sim_device, requires_grad=False)
        for i in range(self.num_dof):
            self.hard_dof_pos_limits[i, 0] = self.robot_config.dof_pos_lower_limit_list[i]
            self.hard_dof_pos_limits[i, 1] = self.robot_config.dof_pos_upper_limit_list[i]
            self.dof_pos_limits[i, 0] = self.robot_config.dof_pos_lower_limit_list[i]
            self.dof_pos_limits[i, 1] = self.robot_config.dof_pos_upper_limit_list[i]
            self.dof_vel_limits[i] = self.robot_config.dof_vel_limit_list[i]
            self.torque_limits[i] = self.robot_config.dof_effort_limit_list[i]
            # soft limits
            m = (self.dof_pos_limits[i, 0] + self.dof_pos_limits[i, 1]) / 2
            r = self.dof_pos_limits[i, 1] - self.dof_pos_limits[i, 0]
            self.dof_pos_limits[i, 0] = m - 0.5 * r * self.robot_config.soft_dof_pos_limit
            self.dof_pos_limits[i, 1] = m + 0.5 * r * self.robot_config.soft_dof_pos_limit
        return self.dof_pos_limits, self.dof_vel_limits, self.torque_limits

    def find_rigid_body_indice(self, body_name):
        """
        ipdb> self.simulator._robot.find_bodies("left_ankle_link")
        ([16], ['left_ankle_link'])
        ipdb> self.simulator.contact_sensor.find_bodies("left_ankle_link")
        ([4], ['left_ankle_link'])

        this function returns the indice of the body in BFS order
        """
        indices, names = self._robot.find_bodies(body_name)
        indices = [self.body_ids.index(i) for i in indices]
        if len(indices) == 0:
            logger.warning(f"Body {body_name} not found in the contact sensor.")
            return None
        if len(indices) == 1:
            return indices[0]
        # multiple bodies found
        logger.warning(f"Multiple bodies found for {body_name}.")
        return indices

    def prepare_sim(self):
        # Wait until play so rigid object collections are initialized
        register_objects(self)

        # Create before state adapter, needs a reference
        self.robot_root_states = RootStatesProxy(self._robot.data.root_state_w)  # (num_envs, 13)

        # Create state adapter after object registry and robot root states are set
        self._state_adapter = IsaacSimStateAdapter(
            device=self.device,
            object_registry=self.object_registry,
            scene=self.scene,
            robot=self._robot,
            robot_states=self.robot_root_states,
        )

        # Create unified access proxy using the state adapter
        self.all_root_states = AllRootStatesProxy(self._state_adapter)

        self.contact_forces_history = torch.zeros(
            self.num_envs, self.contact_history_length, self.num_bodies, 3, device=self.device
        )

        # Initialize virtual gantry system after object registry setup
        # Initialize virtual gantry using config
        gantry_cfg = self.simulator_config.virtual_gantry
        self.virtual_gantry = create_virtual_gantry(
            sim=self,
            enable=gantry_cfg.enabled,
            attachment_body_names=gantry_cfg.attachment_body_names,
            cfg=gantry_cfg,
        )

        # Initialize bridge system using base class helper
        self._init_bridge()

        # Setup video recording after scene is ready
        if self.video_recorder:
            self.video_recorder.setup_recording()

        # Initialize robot tensors
        self.refresh_sim_tensors()

        # Initialize acceleration tensors ONLY if bridge is enabled
        if self.simulator_config.bridge.enabled:
            logger.info("Bridge enabled: initializing acceleration computation tensors")
            self.dof_acc = torch.zeros(self.num_envs, self.num_dof, device=self.device)
            self.prev_dof_vel = torch.zeros(self.num_envs, self.num_dof, device=self.device)
            self.base_linear_acc = torch.zeros(self.num_envs, 3, device=self.device)
            self.prev_base_lin_vel = torch.zeros(self.num_envs, 3, device=self.device)
        else:
            logger.debug("Bridge disabled: skipping acceleration computation tensors")

    @property
    def dof_state(self):
        # This will always use the latest dof_pos and dof_vel
        return torch.cat([self.dof_pos[..., None], self.dof_vel[..., None]], dim=-1)

    def refresh_sim_tensors(self):
        # Apply reset to recache new wyxz -> xyzw tensor
        self.robot_root_states.reset(self._robot.data.root_state_w)  # (num_envs, 13)

        self.base_quat = self.robot_root_states[:, 3:7]  # (num_envs, 4), xyzw
        self.dof_pos = self._robot.data.joint_pos[:, self.dof_ids]  # (num_envs, num_dof)
        self.dof_vel = self._robot.data.joint_vel[:, self.dof_ids]

        # The body ordering of contact_sensor is different from the body ordering of the robot.
        self.contact_forces = self.contact_sensor.data.net_forces_w[
            :, self._contact_to_robot_body_ids
        ]  # (num_envs, num_bodies, 3)

        # Issue: data.net_forces_w_history is not cleared after a reset.
        # Solution: We only read the most recent decimation_factor steps.
        control_decimation = self.simulator_config.sim.control_decimation
        effective_history_length = min(control_decimation, int(self.contact_forces_history.shape[1]))  # buffer = contact_history_length frames
        self.contact_forces_history[:, :effective_history_length, :, :] = self.contact_sensor.data.net_forces_w_history[
            :, :effective_history_length, self._contact_to_robot_body_ids
        ]  # (num_envs, history_length, num_bodies, 3), the first index is the most recent

        self._rigid_body_pos = self._robot.data.body_pos_w[:, self.body_ids, :]
        self._rigid_body_rot = self._robot.data.body_quat_w[:, self.body_ids][
            :, :, [1, 2, 3, 0]
        ]  # (num_envs, 4) 3 isaacsim use wxyz, we keep xyzw for consistency
        self._rigid_body_vel = self._robot.data.body_lin_vel_w[:, self.body_ids, :]
        self._rigid_body_ang_vel = self._robot.data.body_ang_vel_w[:, self.body_ids, :]
        # Keep the legacy COM buffer for existing presets/checkpoints.  Position
        # derivatives and offset-point kinematics must use the explicit link
        # velocity, since _rigid_body_pos is a LINK-origin position in IsaacLab.
        self._rigid_body_com_vel = self._rigid_body_vel
        self._rigid_body_link_vel = body_link_linear_velocity_w(self._robot.data)[:, self.body_ids, :]

    def clear_contact_forces_history(self, env_id):
        if len(env_id) > 0:
            self.contact_forces_history[env_id, :, :, :] = 0.0

    def apply_torques_at_dof(self, torques):
        self._robot.set_joint_effort_target(torques, joint_ids=self.dof_ids)

    def draw_debug_viz(self):
        if self.virtual_gantry:
            self.virtual_gantry.draw_debug()

    def simulate_at_each_physics_step(self):
        self._sim_step_counter += 1
        # Only render if actively recording (not just if video recorder exists)
        has_video_recording = self.video_recorder is not None and self.video_recorder.is_recording
        is_rendering = self.sim.has_gui() or self.sim.has_rtx_sensors() or has_video_recording

        # Apply virtual gantry forces before physics step
        if self.virtual_gantry:
            self.virtual_gantry.step()

        # Step bridge for updated torques before physics step using base class helper
        self._step_bridge()

        self.scene.write_data_to_sim()

        # simulate
        self.sim.step(render=False)

        # Render between steps only IF the GUI or sensor need it
        # note: we assume the render interval to be the shortest accepted rendering interval.
        #    If a camera needs rendering at a faster frequency, this will lead to unexpected behavior.
        if self._sim_step_counter % self.simulator_config.sim.render_interval == 0 and is_rendering:
            self.render()

        # update buffers at sim
        self.scene.update(dt=1.0 / self.simulator_config.sim.fps)

        # Need to update these tensors after each step, since they are used in `_apply_force_in_physics_step`
        self.dof_pos = self._robot.data.joint_pos[:, self.dof_ids]  # (num_envs, num_dof)
        self.dof_vel = self._robot.data.joint_vel[:, self.dof_ids]

        # Update accelerations ONLY if bridge is enabled
        if self.simulator_config.bridge.enabled:
            # Update DOF acceleration using numerical differentiation
            self.dof_acc = (self.dof_vel - self.prev_dof_vel) / self.sim_dt
            self.prev_dof_vel = self.dof_vel.clone()

            # Update base linear acceleration using numerical differentiation
            current_base_vel = self.robot_root_states[:, 7:10]
            self.base_linear_acc = (current_base_vel - self.prev_base_lin_vel) / self.sim_dt
            self.prev_base_lin_vel = current_base_vel.clone()

        # Call video recorder capture frame if recording is active
        if self.video_recorder:
            self.capture_video_frame()

    def setup_viewer(self):
        self.viewer = self.viewport_camera_controller

        # Initialize commands tensor if not already done
        if not hasattr(self, "commands"):
            self.commands = torch.zeros((self.training_config.num_envs, 12), device=self.sim_device)

        # Set up keyboard handling
        if self.viewport_camera_controller is not None:
            self._setup_keyboard_controls()

    def _setup_keyboard_controls(self):
        """Set up keyboard controls for the simulator."""
        try:
            # Import necessary modules
            import carb.input
            import omni.appwindow

            # Get the input interface
            self.input_interface = carb.input.acquire_input_interface()
            self.appwindow = omni.appwindow.get_default_app_window()
            self.keyboard = self.appwindow.get_keyboard()

            # Define key mappings
            self.key_commands = {
                "W": "forward_command",
                "S": "backward_command",
                "A": "left_command",
                "D": "right_command",
                "Q": "heading_left_command",
                "E": "heading_right_command",
                "Z": "zero_command",
                "X": "walk_stand_toggle",
                "U": "height_up",
                "L": "height_down",
                "I": "waist_yaw_up",
                "K": "waist_yaw_down",
                "P": "push_robots",
                "Y": "toggle_camera_tracking",
                # Virtual gantry controls (using enum)
                "KEY_7": GantryCommand.LENGTH_ADJUST,  # decrease
                "KEY_8": GantryCommand.LENGTH_ADJUST,  # increase
                "KEY_9": GantryCommand.TOGGLE,
                "KEY_0": GantryCommand.FORCE_ADJUST,
                "MINUS": GantryCommand.FORCE_SIGN_TOGGLE,
            }

            # Initialize push_requested flag
            self.push_requested = False

            # Register keyboard callback
            def keyboard_callback(event, *args, **kwargs):
                # Only process key press events
                if event.type == carb.input.KeyboardEventType.KEY_PRESS:
                    if event.input.name in self.key_commands:
                        command = self.key_commands[event.input.name]
                        if command == "forward_command":
                            self.commands[:, 0] += 0.1
                            logger.info(f"Current Command: {self.commands[:,]}")
                        elif command == "backward_command":
                            self.commands[:, 0] -= 0.1
                            logger.info(f"Current Command: {self.commands[:,]}")
                        elif command == "left_command":
                            self.commands[:, 1] -= 0.1
                            logger.info(f"Current Command: {self.commands[:,]}")
                        elif command == "right_command":
                            self.commands[:, 1] += 0.1
                            logger.info(f"Current Command: {self.commands[:,]}")
                        elif command == "heading_left_command":
                            self.commands[:, 3] -= 0.1
                            logger.info(f"Current Command: {self.commands[:,]}")
                        elif command == "heading_right_command":
                            self.commands[:, 3] += 0.1
                            logger.info(f"Current Command: {self.commands[:,]}")
                        elif command == "zero_command":
                            self.commands[:, :4] = 0
                            logger.info(f"Current Command: {self.commands[:,]}")
                        elif command == "walk_stand_toggle":
                            self.commands[:, 4] = 1 - self.commands[:, 4]
                            logger.info(f"Current Command: {self.commands[:,]}")
                        elif command == "height_up":
                            self.commands[:, 8] += 0.1
                            logger.info(f"Current Command: {self.commands[:,]}")
                        elif command == "height_down":
                            self.commands[:, 8] -= 0.1
                            logger.info(f"Current Command: {self.commands[:,]}")
                        elif command == "waist_yaw_up":
                            self.commands[:, 5] += 0.1
                            logger.info(f"Current Command: {self.commands[:,]}")
                        elif command == "waist_yaw_down":
                            self.commands[:, 5] -= 0.1
                            logger.info(f"Current Command: {self.commands[:,]}")
                        elif command == "push_robots":
                            logger.info("Push Robots Requested")
                            self.push_requested = True
                        elif command == "toggle_camera_tracking":
                            was_enabled = self.simulator_config.viewer.enable_tracking
                            self.simulator_config = dataclasses.replace(
                                self.simulator_config,
                                viewer=dataclasses.replace(
                                    self.simulator_config.viewer, enable_tracking=not was_enabled
                                ),
                            )

                            if self.viewport_camera_controller is not None:
                                if self.simulator_config.viewer.enable_tracking and not was_enabled:
                                    # ENABLING tracking: capture current camera offset first
                                    self.viewport_camera_controller.capture_current_camera_offset()
                                    self.viewport_camera_controller.update_view_to_asset_root("robot")
                                elif not self.simulator_config.viewer.enable_tracking:
                                    # DISABLING tracking: freeze camera at current position
                                    # The callback only runs when origin_type == "asset_root", so setting it to
                                    # anything else will stop tracking while keeping the camera at its current position
                                    self.viewport_camera_controller.cfg.origin_type = "static"

                            status = "ON" if self.simulator_config.viewer.enable_tracking else "OFF"
                            logger.info(f"Camera tracking: {status}")
                        # Virtual gantry commands (using enum)
                        elif command == GantryCommand.LENGTH_ADJUST:
                            if self.virtual_gantry:
                                # Differentiate between KEY_7 (decrease) and KEY_8 (increase)
                                amount = -0.1 if event.input.name == "KEY_7" else 0.1
                                command_data = GantryCommandData(GantryCommand.LENGTH_ADJUST, {"amount": amount})
                                self.virtual_gantry.handle_command(command_data)
                        elif command == GantryCommand.TOGGLE:
                            if self.virtual_gantry:
                                command_data = GantryCommandData(GantryCommand.TOGGLE)
                                self.virtual_gantry.handle_command(command_data)
                        elif command == GantryCommand.FORCE_ADJUST:
                            if self.virtual_gantry:
                                command_data = GantryCommandData(GantryCommand.FORCE_ADJUST)
                                self.virtual_gantry.handle_command(command_data)
                        elif command == GantryCommand.FORCE_SIGN_TOGGLE:
                            if self.virtual_gantry:
                                command_data = GantryCommandData(GantryCommand.FORCE_SIGN_TOGGLE)
                                self.virtual_gantry.handle_command(command_data)
                        return True
                return False

            self.keyboard_sub = self.input_interface.subscribe_to_keyboard_events(
                self.keyboard,
                lambda event, *args: keyboard_callback(event, *args),
            )
            logger.info("Keyboard controls initialized")

        except Exception as e:
            logger.warning(f"Could not initialize keyboard controls: {e}")

    def render(self, sync_frame_time=True):
        self.sim.render()
        if self.debug_viz_enabled:
            self.clear_lines()
            self.draw_debug_viz()

    # debug visualization - delegate to draw adapter
    def clear_lines(self):
        """Delegate to draw adapter."""
        from holosoma.utils.draw import clear_lines

        clear_lines(self)

    def draw_sphere(self, pos, radius, color, env_id, pos_id):
        """Delegate to draw adapter."""
        from holosoma.utils.draw import draw_sphere

        draw_sphere(self, pos, radius, color, env_id, pos_id)

    def draw_line(self, start_point, end_point, color, env_id):
        """Delegate to draw adapter."""
        from holosoma.utils.draw import draw_line

        draw_line(self, start_point, end_point, color, env_id)

    def set_actor_root_state_tensor_robots(self, env_ids=None, root_states=None):
        """See base class.

        IsaacSim-specific notes:
        - Quaternions converted from (x,y,z,w) to (w,x,y,z) format for IsaacSim compatibility
        """
        if env_ids is None:
            env_ids = torch.arange(getattr(self, "num_envs", self.training_config.num_envs), device=self.sim_device)

        if root_states is None:
            robot_root_states = self.robot_root_states
        elif isinstance(root_states, AllRootStatesProxy):
            robot_root_states = self.robot_root_states
        elif isinstance(root_states, RootStatesProxy):
            # assumes the user passed in robot_root_states directly
            robot_root_states = root_states
        else:
            raise ValueError(f"Unexpected root states type: {type(root_states)}")

        self._robot.write_root_pose_to_sim(robot_root_states._get_wxyz(env_ids)[:, :7], env_ids)
        self._robot.write_root_velocity_to_sim(robot_root_states._get_wxyz(env_ids)[:, 7:], env_ids)

    def set_dof_state_tensor_robots(self, env_ids=None, dof_states=None):
        """See base class.

        IsaacSim-specific notes:
        - Tensor format: 3D [num_envs, num_dofs, 2] (differs from IsaacGym's flattened format)

        Examples
        --------
        >>> # IsaacSim format: 3D [num_envs, num_dofs, 2]
        >>> env_ids = torch.tensor([0, 1], device=device)
        >>> dof_states = torch.zeros(len(env_ids), sim.num_dof, 2, device=device)
        >>> dof_states[:, :, 0] = default_joint_positions  # 2D positions [envs, dofs]
        >>> dof_states[:, :, 1] = 0.0  # Zero velocities
        >>> sim.set_dof_state_tensor_robots(env_ids, dof_states)
        """
        if env_ids is None:
            env_ids = torch.arange(getattr(self, "num_envs", self.training_config.num_envs), device=self.sim_device)

        if dof_states is None:
            dof_states = self.dof_state

        dof_pos, dof_vel = dof_states[env_ids, :, 0], dof_states[env_ids, :, 1]
        self._robot.write_joint_state_to_sim(dof_pos, dof_vel, self.dof_ids, env_ids)

    def get_actor_indices(self, names: str | ActorNames, env_ids: EnvIds | None = None) -> ActorIndices:
        """See base class."""
        return self.object_registry.get_object_indices(names, env_ids)

    def get_actor_states(self, names: ActorNames, env_ids: EnvIds) -> ActorStates:
        """Read actor states through the IsaacSim adapter in public xyzw format.

        This intentionally avoids the BaseSimulator ``get_actor_states_by_index``
        placeholder.  Iterating names in caller order also avoids reordering by the
        interleaved registry when more than one actor is requested.
        """
        return self._state_adapter.get_named_object_states(names, env_ids)

    def set_actor_states(
        self,
        names: ActorNames,
        env_ids: EnvIds,
        states: ActorStates,
        write_updates: bool = True,
    ):
        """See base class.

        IsaacSim-specific notes:
        - Uses AllRootStatesProxy for unified tensor access
        - Automatically calls write_state_updates() for immediate sync
        """
        self._state_adapter.write_named_object_states(names, states, env_ids)
        if write_updates:
            self.write_state_updates()

    def get_actor_initial_poses(self, names: ActorNames, env_ids: EnvIds | None = None) -> ActorPoses:
        """See base class."""
        if not names:
            return torch.empty(0, 7, device=self.sim_device, dtype=torch.float32)

        # Determine which environments to use
        if env_ids is None:
            num_envs = getattr(self, "num_envs", self.scene.num_envs)
            env_ids = torch.arange(num_envs, device=self.sim_device)

        # Get base poses for each object (one per object)
        base_poses = []
        for obj_name in names:
            if obj_name == "robot":
                # Get robot base pose from configuration
                pos = torch.tensor(self.robot_config.init_state.pos, device=self.sim_device, dtype=torch.float32)
                rot = torch.tensor(self.robot_config.init_state.rot, device=self.sim_device, dtype=torch.float32)
                pose = torch.cat([pos, rot])  # [7] - [x,y,z,qx,qy,qz,qw]
                base_poses.append(pose)

            elif self._is_scene_object(obj_name):
                # Get scene object pose from scene collection
                scene_collection = self.scene.rigid_objects["usd_scene_objects"]
                default_state = self._get_scene_default_object_state(scene_collection, obj_name)
                pose = default_state[[0, 1, 2, 4, 5, 6, 3]]  # [x,y,z,qx,qy,qz,qw] reorder from wxyz to xyzw
                base_poses.append(pose)

            elif obj_name in self.scene.rigid_objects:
                # Get individual object pose from rigid object
                rigid_object = self.scene.rigid_objects[obj_name]
                default_state = rigid_object.data.default_root_state[0]  # [13]
                pose = default_state[[0, 1, 2, 4, 5, 6, 3]]  # [x,y,z,qx,qy,qz,qw] reorder from wxyz to xyzw
                base_poses.append(pose)

            else:
                available_objects = ["robot"] + list(self.scene.rigid_objects.keys())
                raise KeyError(f"Object '{obj_name}' not found. Available: {available_objects}")

        base_poses_tensor = torch.stack(base_poses)

        # Repeat to match ObjectRegistry index ordering: [obj0_env0, obj0_env1, obj1_env0, obj1_env1, ...]
        return base_poses_tensor.repeat_interleave(len(env_ids), dim=0)

    def _is_scene_object(self, object_name: str) -> bool:
        """Check if an object is part of the USD scene collection - IsaacSim implementation.

        Uses IsaacLab's native RigidObjectCollection methods to determine if an object
        belongs to a scene collection rather than being an individual rigid object.

        Parameters
        ----------
        object_name : str
            Name of the object to check, e.g., "obj0_0"

        Returns
        -------
        bool
            True if object is in a scene collection, False otherwise

        Notes
        -----
        - Currently assumes single scene collection named 'usd_scene_objects'
        - Uses full path name with "/world/" prefix for IsaacLab compatibility
        - Scene collections are loaded from USD/URDF scene files
        """
        collection_name = "usd_scene_objects"  # TODO fix assumption for one scene collection
        scene_collection = self.scene.rigid_objects.get(collection_name, None)
        full_path_name = f"/world/{object_name}"
        return scene_collection and full_path_name in scene_collection.object_names

    def _get_object_index_in_collection(self, object_name: str, scene_collection) -> int:
        """Get object index within scene collection - IsaacSim implementation.

        Uses IsaacLab's native RigidObjectCollection.find_objects() method to locate
        an object within a scene collection and return its internal index.

        Parameters
        ----------
        object_name : str
            Name of the object, e.g., "obj0_0"
        scene_collection : RigidObjectCollection
            The USD scene collection to search within

        Returns
        -------
        int
            Index of the object within the collection

        Raises
        ------
        KeyError
            If object not found in collection

        Notes
        -----
        - Uses full path name with "/world/" prefix for IsaacLab compatibility
        - Returns the first match if multiple objects found
        - Index is used for tensor access within the collection
        """
        # TODO: Fix remove /world prefix due to USD loader coupling
        full_path_name = f"/world/{object_name}"
        obj_indices, obj_names = scene_collection.find_objects(full_path_name)

        if len(obj_indices) == 0:
            available_names = scene_collection.object_names
            raise KeyError(f"Object '{object_name}' not found in collection. Available: {available_names}")

        return obj_indices[0].item()

    def _get_scene_default_object_state(self, scene_collection, object_name: str) -> torch.Tensor:
        """Get initial object state from scene collection - IsaacSim implementation.

        Retrieves the default/initial state for an object within a scene collection
        using IsaacLab's tensor data after simulation initialization.

        NOTE: Returns quat in IsaacSim wxyz format, not holosoma xyzw format (internal function)

        Parameters
        ----------
        scene_collection : RigidObjectCollection
            The scene collection containing the object
        object_name : str
            Name of the object to get state for

        Returns
        -------
        torch.Tensor
            Default object state [13] containing position, quaternion, and velocities

        Notes
        -----
        - Must be called after sim.play() when tensor data is available
        - Returns full 13-element state vector from IsaacLab's default_object_state
        - Used internally for initial pose extraction and reset operations
        """
        object_index = self._get_object_index_in_collection(object_name, scene_collection)
        return scene_collection.data.default_object_state[0, object_index]  # [13]

    def _get_object_states(self, object_name: str, env_ids: torch.Tensor) -> torch.Tensor:
        """Get object states for any object type - delegates to state adapter.

        Parameters
        ----------
        object_name : str
            Name of the object to query
        env_ids : torch.Tensor
            Environment IDs to query, shape [num_envs], dtype torch.long

        Returns
        -------
        torch.Tensor
            Object states [len(env_ids), 13] containing position, quaternion, and velocities
            in xyzw format (converted by state adapter)
        """
        return self._state_adapter.get_object_states(object_name, env_ids)

    def _write_object_state_unified(self, object_name: str, states: torch.Tensor, env_ids: torch.Tensor):
        """Write object states for any object type - delegates to state adapter."""
        self._state_adapter.write_object_states(object_name, states, env_ids)

    def time(self) -> float:
        """Get current simulation time.

        Returns:
            float: Current simulation time in seconds
        """
        return self.sim.current_time

    def get_dof_forces(self, env_id: int = 0):
        """Get DOF forces for a specific environment.

        This method provides access to measured joint forces. For IsaacSim,
        joint forces are computed from applied torques since direct force
        sensing is not available in the same way as IsaacGym.

        Args:
            env_id: Environment index (default: 0)

        Returns:
            torch.Tensor: Tensor of shape [num_dof] with computed joint forces

        Note:
            IsaacSim doesn't have the same DOF force sensor infrastructure as IsaacGym.
            This implementation returns the applied torques as an approximation.
            For actual force sensing, consider using contact sensors or force/torque sensors.
        """
        # IsaacSim doesn't have direct DOF force sensors like IsaacGym
        # Return the applied torques (which are the commanded forces)
        # This matches the bridge's usage pattern where forces are used for feedback
        if not hasattr(self._robot, "data") or not hasattr(self._robot.data, "applied_torque"):
            logger.warning(
                "DOF forces not directly available in IsaacSim. "
                "Returning zeros. For force feedback, the bridge will use commanded torques."
            )
            return torch.zeros(self.num_dof, device=self.device)

        # Get applied torques which represent the forces being applied to joints
        applied_torques = self._robot.data.applied_torque[env_id, self.dof_ids]
        return applied_torques

    def write_state_updates(self):
        """See base class.

        IsaacSim-specific notes:
        - Uses IsaacLab's scene.write_data_to_sim() for efficient batch synchronization
        - Only performs sync if state adapter indicates dirty state (performance optimization)
        """
        if not self._state_adapter.is_dirty():
            logger.debug("No object state changes to sync")
            return

        logger.debug("Syncing object state changes to simulation")

        # Single call to sync all object state changes
        self.scene.write_data_to_sim()

        # Clear dirty flag via state adapter
        self._state_adapter.clear_dirty()

        logger.debug("All object state changes synced to simulation")
