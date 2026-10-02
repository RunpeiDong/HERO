"""MuJoCo robot plant for the HERO tabletop scene.

The scene is compiled from the G1 URDF. Joint PD torques are clamped
to the exported policy effort limits at every physics substep. Geometry and
poses use named joints and bodies, with explicit quaternion conversion."""

from __future__ import annotations

import os
from collections import deque
from pathlib import Path
from typing import Callable, Mapping, Iterable, Sequence

import mujoco
import numpy as np

from hero_isaacsim import constants as HC
from sim2sim import plant_params as PP
from sim2sim.physics_profiles import (FOOT_COLLISIONS, PHYSICS_PROFILES, adapted_joint_tables,
                                     apply_adapted_solver, describe_physics, replace_urdf_foot_collisions)
from sim2sim.mathutil import quat_apply, wxyz_to_xyzw, xyzw_to_wxyz

URDF_FILE_NAME = HC.URDF_FILE_NAME  # g1_29dof_dex3fixed_hero.urdf
ROOT_BODY = HC.PELVIS_BODY_NAME
from hero_isaacsim.paths import RELEASE_ROOT, G1_ASSET_ROOT
#: ``dex3_urdf`` = the training URDF (Dex3 palms + fixed fingers; the paddle-hand copy goes through the same path via ``urdf_path``),
#: ``mjcf`` = a native holosoma G1 scene, ``nohand`` = the Dex3 URDF with the hands stripped (:func:`nohand_spec_edit`).
PLANT_KINDS: tuple[str, ...] = ("dex3_urdf", "mjcf", "nohand")
#: Plant kinds compiled from a URDF (``default_urdf_path`` resolution; the others take an MJCF scene).
URDF_PLANT_KINDS: tuple[str, ...] = ("dex3_urdf", "nohand")
#: Bare wrist-flange mass and isotropic inertia; moving bodies require nonzero inertia.
NOHAND_PALM_MASS_KG: float = 0.020
NOHAND_PALM_INERTIA: float = 1.0e-6
#: Nominal per-hand mass [kg], summed over the payload body subtree.
HAND_NOMINAL_MASS_KG: dict[str, float] = {"dex3": 0.6965, "rubber": 0.170, "none": NOHAND_PALM_MASS_KG}
MJCF_CONTACT_MODES: tuple[str, ...] = ("native", "robot_floor_only")
MJCF_JOINT_PARAM_MODES: tuple[str, ...] = ("sonic", "native")
NATIVE_G1_SCENE_FILE_NAME = "scene_g1_29dof_wbt_plane.xml"  # G1 scene filename under the robot asset root.
#: Palm bodies used by compatible paddle-hand URDFs.
PADDLE_PALM_BODY_NAMES: tuple[str, str] = ("left_rubber_hand", "right_rubber_hand")
#: Body families that receive ``payload_kg`` (first family fully present in the model wins): Dex3 palms, paddle rubber
#: hands, else the wrist_yaw links (MJCF scenes without separate hand bodies).
PAYLOAD_BODY_CANDIDATES: tuple[tuple[str, ...], ...] = (tuple(HC.PALM_BODY_NAMES), PADDLE_PALM_BODY_NAMES, tuple(HC.EE_BODY_NAMES))


def default_native_mjcf_path(hint: str | os.PathLike | None = None) -> Path:
    """Resolve a native G1 scene: explicit ``hint`` > ``$HERO_NATIVE_G1_MJCF`` > the vendored holosoma tree
    (``third_party/holosoma/holosoma/data/robots/g1/scenes/<scene>``) > ``$HERO_ROBOT_ASSET_ROOT/g1/scenes/<scene>``."""
    candidates: list[Path] = []
    if hint:
        candidates.append(Path(hint).expanduser())
    env = os.environ.get("HERO_NATIVE_G1_MJCF")
    if env:
        candidates.append(Path(env).expanduser())
    candidates.append(RELEASE_ROOT / "third_party" / "holosoma" / "holosoma" / "data" / "robots" / "g1" / "scenes" / NATIVE_G1_SCENE_FILE_NAME)
    candidates.append(Path(__file__).resolve().parents[1] / "third_party" / "holosoma" / "holosoma" / "data" / "robots" / "g1" / "scenes" / NATIVE_G1_SCENE_FILE_NAME)
    env_root = os.environ.get("HERO_ROBOT_ASSET_ROOT")
    if env_root:
        candidates.append(Path(env_root) / "g1" / "scenes" / NATIVE_G1_SCENE_FILE_NAME)
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"native G1 MJCF scene not found; tried {[str(c) for c in candidates]}")


def default_urdf_path(hint: str | os.PathLike | None = None) -> Path:
    """Resolve the training URDF: explicit ``hint`` (must exist -- an explicit file that is missing raises instead of
    silently falling back to the Dex3 default, e.g. a paddle-hand copy absent from the asset root) >
    ``$HERO_ROBOT_ASSET_ROOT/g1_modified/<urdf>`` > repo path > ``$HERO_SIM2SIM_URDF``.  Raises ``FileNotFoundError``
    listing the candidates."""
    candidates: list[Path] = []
    if hint:
        h = Path(hint).expanduser()
        if not h.is_file():
            raise FileNotFoundError(f"training URDF {h} (explicit hint) does not exist; default candidates would be {[str(c) for c in _default_urdf_candidates()]}")
        candidates.append(h)
    candidates.extend(_default_urdf_candidates())
    for c in candidates:
        if c.is_file():
            return c.resolve()
    raise FileNotFoundError(f"training URDF not found; tried {[str(c) for c in candidates]}")


def _default_urdf_candidates() -> list[Path]:
    """The hint-less search order of :func:`default_urdf_path` (``$HERO_ROBOT_ASSET_ROOT/g1_modified``, ``$HERO_ROBOT_ASSET_ROOT``,
    bundled robot assets, ``$HERO_SIM2SIM_URDF``)."""
    candidates: list[Path] = []
    env_root = os.environ.get("HERO_ROBOT_ASSET_ROOT")
    if env_root:
        candidates.append(Path(env_root) / "g1_modified" / URDF_FILE_NAME)
        candidates.append(Path(env_root) / URDF_FILE_NAME)
    candidates.append(G1_ASSET_ROOT / URDF_FILE_NAME)
    env_urdf = os.environ.get("HERO_SIM2SIM_URDF")
    if env_urdf:
        candidates.append(Path(env_urdf).expanduser())
    return candidates


def nohand_spec_edit(spec: "mujoco.MjSpec", *, palm_bodies: Sequence[str] = HC.PALM_BODY_NAMES, palm_mass_kg: float = NOHAND_PALM_MASS_KG,
                     palm_inertia: float = NOHAND_PALM_INERTIA) -> dict:
    """Turn a Dex3-URDF ``MjSpec`` into the no-hand variant in place: delete every body below
    the palm bodies (the fixed finger links), delete the palm bodies' visual + collision geoms, and give each palm body
    ``palm_mass_kg`` with an isotropic ``palm_inertia`` at its own origin (``explicitinertial``).  Nothing above the palm
    (wrist links, ``PALM_OFFSET`` joint origin) is touched.  Returns what was removed / set (``describe()["hand_variant"]``).
    Raises ``ValueError`` when a palm body is missing (e.g. the paddle URDF, whose hands are ``*_rubber_hand`` bodies)."""
    removed_bodies: list[str] = []
    removed_geoms = 0
    for name in palm_bodies:
        palm = spec.body(name)
        if palm is None:
            raise ValueError(f"nohand plant needs the Dex3 palm body {name!r} in the URDF (found no such body)")

        def _collect(b) -> list[str]:
            out: list[str] = []
            for c in list(b.bodies):
                out += _collect(c)
                out.append(c.name)
            return out

        descendants = _collect(palm)
        for c in list(palm.bodies):  # deleting a body removes its subtree
            spec.delete(c)
        removed_bodies += descendants
        for g in list(palm.geoms):
            spec.delete(g)
            removed_geoms += 1
        palm.mass = float(palm_mass_kg)
        palm.fullinertia = [float(palm_inertia)] * 3 + [0.0, 0.0, 0.0]
        palm.ipos = [0.0, 0.0, 0.0]
        palm.iquat = [1.0, 0.0, 0.0, 0.0]
        palm.explicitinertial = True
    return {
        "kind": "nohand",
        "palm_bodies": list(palm_bodies),
        "palm_mass_kg": float(palm_mass_kg),
        "palm_inertia_kg_m2": float(palm_inertia),
        "removed_bodies": removed_bodies,
        "removed_geoms": removed_geoms,
    }


def body_subtree_ids(model: mujoco.MjModel, body_id: int) -> list[int]:
    """``body_id`` followed by every descendant body (parent-before-child order) -- the bodies whose mass makes up a hand."""
    out = [int(body_id)]
    frontier = [int(body_id)]
    while frontier:
        parent = frontier.pop()
        kids = [b for b in range(model.nbody) if int(model.body_parentid[b]) == parent and b != parent]
        out += kids
        frontier += kids
    return out


def build_model(
    urdf_path: str | os.PathLike,
    *,
    physics_dt: float = 0.001,
    armature: Sequence[float] = PP.DOF_ARMATURE,
    damping: Sequence[float] = PP.DOF_DAMPING,
    frictionloss: Sequence[float] = PP.DOF_FRICTIONLOSS,
    self_collisions: bool = False,
    floor_friction: Sequence[float] = (1.0, 0.005, 0.0001),
    keep_visual: bool = True,
    foot_collision: str = "sonic_box",
    root_body: str = ROOT_BODY,
    spec_callback: Callable | None = None,
) -> mujoco.MjModel:
    """Compile the robot URDF with a floor and optional scene callback."""
    spec = mujoco.MjSpec.from_file(str(urdf_path))
    spec.meshdir = "."
    spec.compiler.fusestatic = False
    spec.compiler.discardvisual = not keep_visual
    spec.option.timestep = float(physics_dt)
    spec.option.gravity = [0.0, 0.0, -9.81]

    root = spec.body(root_body)
    if root is None:
        raise ValueError(f"{urdf_path}: no body named {root_body!r}")
    free = root.add_freejoint()
    free.name = "root"

    floor = spec.worldbody.add_geom()
    floor.name = "floor"
    floor.type = mujoco.mjtGeom.mjGEOM_PLANE
    floor.size = [0.0, 0.0, 1.0]
    floor.friction = list(floor_friction)
    floor.contype = 1
    floor.conaffinity = 1

    replace_urdf_foot_collisions(spec, foot_collision)

    for g in spec.geoms:
        if g.name == "floor":
            continue
        is_collision = (g.contype != 0) or (g.conaffinity != 0)
        if not is_collision:
            continue
        if self_collisions:
            g.contype, g.conaffinity = 1, 1
        else:
            g.contype, g.conaffinity = 1, 0  # collides with the floor (conaffinity 1) but not with other robot geoms

    dof_names = list(HC.DOF_NAMES)
    if not (len(armature) == len(damping) == len(frictionloss) == len(dof_names)):
        raise ValueError("armature / damping / frictionloss must have one entry per DOF")
    seen = set()
    for j in spec.joints:
        if j.name in dof_names:
            i = dof_names.index(j.name)
            j.armature = float(armature[i])
            # mujoco >= 3.10 stores per-axis joint damping (3-vector; a hinge uses component 0)
            j.damping = np.full(3, float(damping[i]), dtype=np.float64) if np.ndim(j.damping) else float(damping[i])
            j.frictionloss = float(frictionloss[i])
            seen.add(j.name)
    missing = [n for n in dof_names if n not in seen]
    if missing:
        raise ValueError(f"{urdf_path}: joints missing from the URDF: {missing}")

    if spec_callback is not None:
        spec_callback(spec)
    return spec.compile()


def _set_joint_params(j, armature: float, damping: float, frictionloss: float) -> None:
    j.armature = float(armature)
    # mujoco >= 3.10 stores per-axis joint damping (3-vector; a hinge uses component 0)
    j.damping = np.full(3, float(damping), dtype=np.float64) if np.ndim(j.damping) else float(damping)
    j.frictionloss = float(frictionloss)


def build_model_from_mjcf(
    mjcf_path: str | os.PathLike,
    *,
    physics_dt: float = 0.001,
    armature: Sequence[float] = PP.DOF_ARMATURE,
    damping: Sequence[float] = PP.DOF_DAMPING,
    frictionloss: Sequence[float] = PP.DOF_FRICTIONLOSS,
    joint_params: str = "sonic",
    contacts: str = "native",
    keep_visual: bool = True,
    root_body: str = ROOT_BODY,
    spec_callback: Callable | None = None,
) -> tuple[mujoco.MjModel, dict]:
    """Construct the MuJoCo robot model and exported actuator interface."""
    if joint_params not in MJCF_JOINT_PARAM_MODES:
        raise ValueError(f"joint_params must be one of {MJCF_JOINT_PARAM_MODES}, got {joint_params!r}")
    if contacts not in MJCF_CONTACT_MODES:
        raise ValueError(f"contacts must be one of {MJCF_CONTACT_MODES}, got {contacts!r}")
    spec = mujoco.MjSpec.from_file(str(mjcf_path))
    spec.compiler.discardvisual = not keep_visual
    info: dict = {
        "file_timestep": float(spec.option.timestep),
        "file_solver": int(spec.option.solver),
        "file_iterations": int(spec.option.iterations),
        "n_contact_pairs": len(spec.pairs),
        "n_contact_excludes": len(spec.excludes),
        "n_tendons": len(spec.tendons),
        "n_actuators": len(spec.actuators),
        "joint_params": joint_params,
        "contacts": contacts,
    }
    spec.option.timestep = float(physics_dt)
    spec.option.gravity = [0.0, 0.0, -9.81]

    if spec.body(root_body) is None:
        raise ValueError(f"{mjcf_path}: no body named {root_body!r}")
    free = [j for j in spec.joints if j.type == mujoco.mjtJoint.mjJNT_FREE]
    if len(free) != 1:
        raise ValueError(f"{mjcf_path}: expected exactly one free joint, found {[j.name for j in free]}")
    info["free_joint"] = free[0].name

    planes = [g for g in spec.geoms if g.type == mujoco.mjtGeom.mjGEOM_PLANE]
    if len(planes) > 1:
        raise ValueError(f"{mjcf_path}: more than one plane geom: {[g.name for g in planes]}")
    if planes:
        floor = planes[0]
        info["floor"] = floor.name or "<unnamed plane>"
        info["floor_added"] = False
    else:
        floor = spec.worldbody.add_geom()
        floor.name = "floor"
        floor.type = mujoco.mjtGeom.mjGEOM_PLANE
        floor.size = [0.0, 0.0, 1.0]
        floor.friction = [1.0, 0.005, 0.0001]
        info["floor"] = "floor"
        info["floor_added"] = True
    if contacts == "robot_floor_only":
        floor.contype, floor.conaffinity = 1, 1
        for g in spec.geoms:
            if g is floor or g.type == mujoco.mjtGeom.mjGEOM_PLANE:
                continue
            if (g.contype != 0) or (g.conaffinity != 0):
                g.contype, g.conaffinity = 1, 0

    dof_names = list(HC.DOF_NAMES)
    if not (len(armature) == len(damping) == len(frictionloss) == len(dof_names)):
        raise ValueError("armature / damping / frictionloss must have one entry per DOF")
    native_arm, native_damp, native_fric = [], [], []
    seen = set()
    by_name = {j.name: j for j in spec.joints}
    for i, name in enumerate(dof_names):
        j = by_name.get(name)
        if j is None or j.type != mujoco.mjtJoint.mjJNT_HINGE:
            continue
        native_arm.append(float(j.armature))
        native_damp.append(float(np.asarray(j.damping).reshape(-1)[0]))
        native_fric.append(float(j.frictionloss))
        if joint_params == "sonic":
            _set_joint_params(j, armature[i], damping[i], frictionloss[i])
        seen.add(name)
    missing = [n for n in dof_names if n not in seen]
    if missing:
        raise ValueError(f"{mjcf_path}: hinge joints missing from the MJCF: {missing}")
    info["file_armature"] = native_arm
    info["file_damping"] = native_damp
    info["file_frictionloss"] = native_fric

    if spec_callback is not None:
        spec_callback(spec)
    return spec.compile(), info


class MujocoPlant:
    """The compiled model + data with holosoma-shaped state accessors and the per-sub-step PD controller."""

    def __init__(
        self,
        urdf_path: str | os.PathLike | None = None,
        *,
        physics_dt: float | None = None,
        substeps: int | None = None,
        kp: Sequence[float] | None = None,
        kd: Sequence[float] | None = None,
        effort_limit: Sequence[float] | None = None,
        armature: Sequence[float] | None = None,
        damping: Sequence[float] | None = None,
        frictionloss: Sequence[float] | None = None,
        foot_collision: str = "sonic_box",
        physics_profile: str = "hero",
        self_collisions: bool = False,
        keep_visual: bool = True,
        plant_kind: str = "auto",
        mjcf_contacts: str = "native",
        mjcf_joint_params: str = "sonic",
        kp_scale: float = 1.0,
        kd_scale: float = 1.0,
        delay_steps: int = 0,
        payload_kg: float = 0.0,
        payload_bodies: Sequence[str] | None = None,
        friction: float | None = None,
        spec_callback: Callable | None = None,
        extra_free_joint_names: Sequence[str] = (),
        hand_mass_kg: float | None = None,
    ):
        """Construct the MuJoCo robot model and exported actuator interface."""
        if plant_kind == "auto":
            plant_kind = "mjcf" if urdf_path is not None and str(urdf_path).lower().endswith(".xml") else "dex3_urdf"
        if plant_kind not in PLANT_KINDS:
            raise ValueError(f"plant_kind must be one of {PLANT_KINDS} (or 'auto'), got {plant_kind!r}")
        self.plant_kind = plant_kind
        if foot_collision not in FOOT_COLLISIONS or physics_profile not in PHYSICS_PROFILES:
            raise ValueError(f"unknown foot_collision/physics_profile: {foot_collision!r}/{physics_profile!r}")
        if plant_kind == "mjcf" and physics_profile != "hero":
            raise ValueError("sonic_deploy_adapted applies to URDF plants only; native MJCF defines its own physics")
        self.physics_profile = physics_profile
        self.foot_collision_requested = foot_collision
        self.foot_collision = "native" if plant_kind == "mjcf" else foot_collision
        self.self_collisions_requested = bool(self_collisions)
        adapted = physics_profile == "sonic_deploy_adapted"
        defaults = adapted_joint_tables() if adapted else (PP.DOF_ARMATURE, PP.DOF_DAMPING, PP.DOF_FRICTIONLOSS)
        armature, damping, frictionloss = tuple(given if given is not None else default for given, default
                                               in zip((armature, damping, frictionloss), defaults))
        self.physics_dt = float((.005 if adapted else .001) if physics_dt is None else physics_dt)
        if not np.isfinite(self.physics_dt) or self.physics_dt <= 0:
            raise ValueError("physics_dt must be finite and positive")
        resolved_substeps = (4 if adapted else 20) if substeps is None else substeps
        if not np.isfinite(resolved_substeps) or int(resolved_substeps) != resolved_substeps or resolved_substeps <= 0:
            raise ValueError("substeps must be a positive integer")
        self.substeps = int(resolved_substeps)
        self.control_dt = self.physics_dt * self.substeps
        self.native_info: dict = {}
        self.hand_variant_edit: dict | None = None
        if plant_kind == "nohand":
            user_cb = spec_callback

            def spec_callback(spec, _cb=user_cb):  # noqa: F811 - the nohand edit runs first, then the caller's scene hook
                self.hand_variant_edit = nohand_spec_edit(spec)
                if _cb is not None:
                    _cb(spec)

        if plant_kind == "mjcf":
            self.urdf_path = default_native_mjcf_path(urdf_path)  # attribute name kept for callers; it is the MJCF scene
            self.model, self.native_info = build_model_from_mjcf(
                self.urdf_path,
                physics_dt=self.physics_dt,
                armature=armature,
                damping=damping,
                frictionloss=frictionloss,
                joint_params=mjcf_joint_params,
                contacts=("robot_floor_only" if (mjcf_contacts == "robot_floor_only" and not self_collisions) else mjcf_contacts),
                keep_visual=keep_visual,
                spec_callback=spec_callback,
            )
        else:  # dex3_urdf / nohand: the training URDF (nohand edits the spec through spec_callback above)
            self.urdf_path = default_urdf_path(urdf_path)
            self.model = build_model(
                self.urdf_path,
                physics_dt=self.physics_dt,
                armature=armature,
                damping=damping,
                frictionloss=frictionloss,
                self_collisions=self_collisions,
                keep_visual=keep_visual,
                foot_collision=foot_collision,
                spec_callback=spec_callback,
            )
        self.model_path = self.urdf_path
        if adapted:
            apply_adapted_solver(self.model)
        self.data = mujoco.MjData(self.model)
        m = self.model

        #: Body-frame gyro increment (XYZW), integrated over physics substeps
        #: with the trapezoidal rule. Substep integration avoids aliasing contact
        #: sway into yaw drift. Reset gives identity and clears imu_step_count.
        self.imu_delta_quat_b: np.ndarray = np.array([0.0, 0.0, 0.0, 1.0])
        self.imu_step_count: int = 0
        self.kp_scale, self.kd_scale = float(kp_scale), float(kd_scale)
        if self.kp_scale <= 0.0 or self.kd_scale < 0.0:
            raise ValueError(f"kp_scale must be > 0 and kd_scale >= 0, got {kp_scale} / {kd_scale}")
        self.kp_nominal = np.asarray(PP.KP if kp is None else kp, dtype=np.float64).reshape(29)
        self.kd_nominal = np.asarray(PP.KD if kd is None else kd, dtype=np.float64).reshape(29)
        self.kp = self.kp_nominal * self.kp_scale
        self.kd = self.kd_nominal * self.kd_scale
        self.effort_limit = np.asarray(PP.EFFORT_LIMIT if effort_limit is None else effort_limit, dtype=np.float64).reshape(29)
        self.delay_steps = int(delay_steps)
        if self.delay_steps < 0:
            raise ValueError(f"delay_steps must be >= 0, got {delay_steps}")
        self._delay_queue: deque[np.ndarray] = deque()
        self.last_q_target_applied: np.ndarray | None = None

        # --- index maps -------------------------------------------------------------------------------
        free_jids = [j for j in range(m.njnt) if m.jnt_type[j] == mujoco.mjtJoint.mjJNT_FREE]
        root_free = [j for j in free_jids if mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.jnt_bodyid[j])) == ROOT_BODY]
        extra_free = [j for j in free_jids if j not in root_free]
        allowed_free = set(extra_free_joint_names)
        if len(root_free) != 1 or m.jnt_qposadr[root_free[0]] != 0 or any(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, j) not in allowed_free for j in extra_free):
            raise RuntimeError(f"one free joint on {ROOT_BODY!r} owning qpos[0:7] (plus declared scene prop joints) is required (found joints {free_jids})")
        self.root_joint_name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_JOINT, root_free[0])
        qadr, vadr = [], []
        for name in HC.DOF_NAMES:
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            if jid < 0 or m.jnt_type[jid] != mujoco.mjtJoint.mjJNT_HINGE:
                raise RuntimeError(f"joint {name!r} missing or not a hinge")
            qadr.append(int(m.jnt_qposadr[jid]))
            vadr.append(int(m.jnt_dofadr[jid]))
        self.dof_qadr = np.asarray(qadr, dtype=np.int64)
        self.dof_vadr = np.asarray(vadr, dtype=np.int64)
        self.dof_in_mujoco_order = bool(np.all(np.diff(self.dof_qadr) == 1) and self.dof_qadr[0] == 7)

        self.body_id: dict[str, int] = {}
        for i in range(m.nbody):
            self.body_id[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, i)] = i
        # canonical bodies missing from the file: only the massless foot contact points may be synthesised
        # (``*_ankle_roll_link`` + FOOT_CONTACT_POINT_OFFSET, exactly how the holosoma URDF defines them)
        self.virtual_bodies: dict[str, tuple[int, np.ndarray]] = {}
        missing = [n for n in HC.HOLOSOMA_BODY_NAMES_32 if n not in self.body_id]
        for n in list(missing):
            if n in HC.FOOT_CONTACT_POINT_BODY_NAMES:
                parent = HC.FOOT_CONTACT_POINT_PARENTS[HC.FOOT_CONTACT_POINT_BODY_NAMES.index(n)]
                if parent in self.body_id:
                    self.virtual_bodies[n] = (self.body_id[parent], np.asarray(HC.FOOT_CONTACT_POINT_OFFSET, dtype=np.float64))
                    missing.remove(n)
        self.missing_bodies = tuple(missing)
        if missing:
            raise RuntimeError(f"canonical bodies missing from the compiled model: {missing}")
        self.canonical_body_ids = np.asarray([self.body_id.get(n, -1) for n in HC.HOLOSOMA_BODY_NAMES_32], dtype=np.int64)
        self.root_body_id = self.body_id[ROOT_BODY]
        self.ee_body_ids = np.asarray([self.body_id[n] for n in HC.EE_BODY_NAMES], dtype=np.int64)
        self.palm_body_ids = (
            np.asarray([self.body_id[n] for n in HC.PALM_BODY_NAMES], dtype=np.int64)
            if all(n in self.body_id for n in HC.PALM_BODY_NAMES)
            else None
        )
        self.palm_offset = np.asarray([HC.PALM_OFFSET[s] for s in HC.EE_SIDES], dtype=np.float64)  # (2, 3)
        self.dof_pos_limits = self._dof_limits()  # (29, 2) URDF ranges
        self._vel6 = np.zeros(6, dtype=np.float64)
        self.last_torque = np.zeros(29, dtype=np.float64)
        self.object = None

        # ---- perturbations: palm payload and contact friction ------------------
        self.payload_kg = float(payload_kg)
        if self.payload_kg < 0.0:
            raise ValueError(f"payload_kg must be >= 0, got {payload_kg}")
        self.payload_bodies: tuple[str, ...] = self._resolve_payload_bodies(payload_bodies)
        self.payload_body_ids = np.asarray([self.body_id[n] for n in self.payload_bodies], dtype=np.int64)
        # hand = payload body + everything below it (Dex3 fingers); the absolute override rescales that subtree BEFORE the payload
        self.hand_subtree_ids: tuple[tuple[int, ...], ...] = tuple(tuple(body_subtree_ids(m, int(b))) for b in self.payload_body_ids)
        self.hand_nominal_mass = np.asarray([float(m.body_mass[list(ids)].sum()) for ids in self.hand_subtree_ids], dtype=np.float64)
        self.hand_mass_kg = None if hand_mass_kg is None else float(hand_mass_kg)
        if self.hand_mass_kg is not None:
            if not (self.hand_mass_kg > 0.0):
                raise ValueError(f"hand_mass_kg must be > 0 (a massless moving body is not simulable), got {hand_mass_kg}")
            self._apply_hand_mass(self.hand_mass_kg)
        self.hand_mass_applied = np.asarray([float(m.body_mass[list(ids)].sum()) for ids in self.hand_subtree_ids], dtype=np.float64)
        self.payload_nominal_mass = np.asarray(m.body_mass[self.payload_body_ids], dtype=np.float64).copy()
        if self.payload_kg > 0.0:
            self._apply_payload(self.payload_kg)
        self.friction = None if friction is None else float(friction)
        self.friction_geoms: tuple[int, ...] = ()
        if self.friction is not None:
            if self.friction < 0.0:
                raise ValueError(f"friction must be >= 0, got {friction}")
            self.friction_geoms = self._apply_friction(self.friction)
        mujoco.mj_forward(m, self.data)

    # ------------------------------------------------------------------------------------------ perturbations
    def _resolve_payload_bodies(self, names: Sequence[str] | None) -> tuple[str, ...]:
        if names:
            missing = [n for n in names if n not in self.body_id]
            if missing:
                raise ValueError(f"payload bodies missing from the model: {missing}")
            return tuple(names)
        for family in PAYLOAD_BODY_CANDIDATES:
            if all(n in self.body_id for n in family):
                return tuple(family)
        raise RuntimeError(f"no payload body family present in the model (tried {PAYLOAD_BODY_CANDIDATES})")

    def _apply_payload(self, payload_kg: float) -> None:
        """Split ``payload_kg`` over the payload bodies: ``mass += share``, inertia scaled by the mass ratio, ``mj_setConst``."""
        m = self.model
        share = float(payload_kg) / len(self.payload_body_ids)
        for bid, nominal in zip(self.payload_body_ids, self.payload_nominal_mass):
            new_mass = float(nominal) + share
            ratio = new_mass / float(nominal) if nominal > 0.0 else 1.0
            m.body_mass[bid] = new_mass
            if nominal > 0.0:
                m.body_inertia[bid] = m.body_inertia[bid] * ratio
            else:  # massless placeholder body: give the point mass a tiny isotropic inertia so the solver stays regular
                m.body_inertia[bid] = np.full(3, 1e-5 * share, dtype=np.float64)
        mujoco.mj_setConst(m, self.data)

    def _apply_hand_mass(self, hand_mass_kg: float) -> None:
        """Set the mass of EACH hand subtree (payload body + descendants) to ``hand_mass_kg``: every body of the subtree has mass and
        inertia scaled by ``hand_mass_kg / subtree_mass`` (the hand's centre of mass is unchanged), then ``mj_setConst``."""
        m = self.model
        for ids, nominal in zip(self.hand_subtree_ids, self.hand_nominal_mass):
            if nominal <= 0.0:
                raise RuntimeError(f"hand subtree {[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ids]} has no mass to rescale")
            ratio = float(hand_mass_kg) / float(nominal)
            for bid in ids:
                m.body_mass[bid] = m.body_mass[bid] * ratio
                m.body_inertia[bid] = m.body_inertia[bid] * ratio
        mujoco.mj_setConst(m, self.data)

    def _apply_friction(self, mu: float) -> tuple[int, ...]:
        """Sliding friction of the floor and every robot collision geom (the object / pedestal geoms keep theirs)."""
        m = self.model
        skip = set()
        touched: list[int] = []
        for g in range(m.ngeom):
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g)
            if name in skip:
                continue
            if m.geom_type[g] != mujoco.mjtGeom.mjGEOM_PLANE and not (m.geom_contype[g] or m.geom_conaffinity[g]):
                continue  # visual-only geom
            m.geom_friction[g, 0] = float(mu)
            touched.append(int(g))
        return tuple(touched)

    def _delayed_target(self, q_target: np.ndarray) -> np.ndarray:
        """FIFO of the commanded targets; returns the one ``delay_steps`` control steps old (primed with the first target)."""
        if self.delay_steps <= 0:
            return q_target
        q = np.array(q_target, dtype=np.float64, copy=True)
        if not self._delay_queue:
            for _ in range(self.delay_steps):
                self._delay_queue.append(q.copy())
        self._delay_queue.append(q)
        return self._delay_queue.popleft()

    @property
    def perturbation(self) -> dict:
        """Realised perturbation knobs (series meta ``plant.perturbation``)."""
        m = self.model
        return {
            "kp_scale": self.kp_scale,
            "kd_scale": self.kd_scale,
            "delay_steps": self.delay_steps,
            "payload_kg": self.payload_kg,
            "payload_bodies": list(self.payload_bodies),
            "payload_body_mass_kg": [float(m.body_mass[b]) for b in self.payload_body_ids],
            "payload_body_mass_nominal_kg": self.payload_nominal_mass.tolist(),
            "friction": self.friction,
            "friction_geoms": len(self.friction_geoms),
            "hand_mass_kg": self.hand_mass_kg,
            "hand_mass_nominal_kg": self.hand_nominal_mass.tolist(),
            "hand_mass_applied_kg": self.hand_mass_applied.tolist(),
            "hand_subtree_bodies": [[mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, b) for b in ids] for ids in self.hand_subtree_ids],
            "active": bool(self.kp_scale != 1.0 or self.kd_scale != 1.0 or self.delay_steps > 0 or self.payload_kg > 0.0 or self.friction is not None
                           or self.hand_mass_kg is not None),
        }

    @property
    def hand_variant(self) -> dict:
        """Which hand family the plant carries (``describe()["hand_variant"]``): ``dex3`` (palm + fingers), ``rubber`` (paddle copy),
        ``none`` (no-hand plant) or ``mjcf_wrist`` (native scene without hand bodies), with the per-side hand mass and the no-hand edit."""
        if self.plant_kind == "nohand":
            kind = "none"
        elif tuple(self.payload_bodies) == tuple(HC.PALM_BODY_NAMES):
            kind = "dex3"
        elif tuple(self.payload_bodies) == PADDLE_PALM_BODY_NAMES:
            kind = "rubber"
        else:
            kind = "mjcf_wrist"
        return {
            "kind": kind,
            "nominal_mass_kg": HAND_NOMINAL_MASS_KG.get(kind),
            "hand_mass_nominal_kg": self.hand_nominal_mass.tolist(),
            "hand_mass_applied_kg": self.hand_mass_applied.tolist(),
            "edit": self.hand_variant_edit,
        }

    @property
    def has_object(self) -> bool:
        return self.object is not None

    # ------------------------------------------------------------------------------------------ helpers
    def _dof_limits(self) -> np.ndarray:
        m = self.model
        out = np.zeros((29, 2), dtype=np.float64)
        for i, name in enumerate(HC.DOF_NAMES):
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name)
            out[i] = m.jnt_range[jid]
        return out

    @property
    def total_mass(self) -> float:
        return float(self.model.body_mass.sum())

    # ------------------------------------------------------------------------------------------ state
    def reset(
        self,
        root_pos: np.ndarray,
        root_quat_xyzw: np.ndarray,
        dof_pos: np.ndarray,
        *,
        root_lin_vel_w: np.ndarray | None = None,
        root_ang_vel_w: np.ndarray | None = None,
        dof_vel: np.ndarray | None = None,
    ) -> None:
        """Set the full state (holosoma ``set_actor_root_state_tensor_robots`` + ``set_dof_state_tensor_robots``).

        Velocities default to zero.  ``root_ang_vel_w`` is the WORLD-frame angular velocity (holosoma
        ``robot_root_states[:, 10:13]``); MuJoCo's free joint stores the body-frame one.
        """
        d = self.data
        mujoco.mj_resetData(self.model, d)  # also clears the solver warm start -> rollouts are independent of the previous clip
        d.qfrc_applied[:] = 0.0
        d.qpos[0:3] = np.asarray(root_pos, dtype=np.float64)
        q = np.asarray(root_quat_xyzw, dtype=np.float64)
        q = q / max(np.linalg.norm(q), 1e-12)
        d.qpos[3:7] = xyzw_to_wxyz(q)
        d.qpos[self.dof_qadr] = np.asarray(dof_pos, dtype=np.float64)
        if root_lin_vel_w is not None:
            d.qvel[0:3] = np.asarray(root_lin_vel_w, dtype=np.float64)
        if root_ang_vel_w is not None:
            d.qvel[3:6] = quat_apply(q * np.array([-1.0, -1.0, -1.0, 1.0]), np.asarray(root_ang_vel_w, dtype=np.float64))
        if dof_vel is not None:
            d.qvel[self.dof_vadr] = np.asarray(dof_vel, dtype=np.float64)
        d.time = 0.0
        self.last_torque[:] = 0.0
        self._delay_queue.clear()
        self.last_q_target_applied = None
        self.imu_delta_quat_b = np.array([0.0, 0.0, 0.0, 1.0])
        self.imu_step_count = 0
        mujoco.mj_forward(self.model, d)

    @property
    def dof_pos(self) -> np.ndarray:
        return np.asarray(self.data.qpos[self.dof_qadr], dtype=np.float64)

    @property
    def dof_vel(self) -> np.ndarray:
        return np.asarray(self.data.qvel[self.dof_vadr], dtype=np.float64)

    @property
    def root_pos(self) -> np.ndarray:
        return np.asarray(self.data.qpos[0:3], dtype=np.float64)

    @property
    def root_quat(self) -> np.ndarray:
        """Pelvis orientation, xyzw (== the root body ``xquat``)."""
        return wxyz_to_xyzw(self.data.qpos[3:7])

    @property
    def root_lin_vel_w(self) -> np.ndarray:
        return np.asarray(self.data.qvel[0:3], dtype=np.float64)

    @property
    def root_ang_vel_b(self) -> np.ndarray:
        """Angular velocity in the pelvis frame (MuJoCo free-joint convention) == holosoma ``base_ang_vel`` obs."""
        return np.asarray(self.data.qvel[3:6], dtype=np.float64)

    @property
    def root_ang_vel_w(self) -> np.ndarray:
        return quat_apply(self.root_quat, self.root_ang_vel_b)

    def body_pos(self, ids: Iterable[int] | np.ndarray) -> np.ndarray:
        return np.asarray(self.data.xpos[np.asarray(list(ids) if not isinstance(ids, np.ndarray) else ids)], dtype=np.float64)

    def body_quat(self, ids: Iterable[int] | np.ndarray) -> np.ndarray:
        """Body orientations xyzw."""
        idx = np.asarray(list(ids) if not isinstance(ids, np.ndarray) else ids)
        return wxyz_to_xyzw(self.data.xquat[idx])

    def body_pose_by_name(self, names: Sequence[str]) -> tuple[np.ndarray, np.ndarray]:
        ids = np.asarray([self.body_id[n] for n in names], dtype=np.int64)
        return self.body_pos(ids), self.body_quat(ids)

    def canonical_body_poses(self) -> tuple[np.ndarray, np.ndarray]:
        """(32, 3) positions and (32, 4) xyzw orientations in ``HOLOSOMA_BODY_NAMES_32`` order (virtual foot contact
        points, when the file lacks them, are placed at ``ankle_roll + R(q_ankle) offset`` with the ankle orientation)."""
        ids = self.canonical_body_ids.copy()
        virtual = [(k, n) for k, n in enumerate(HC.HOLOSOMA_BODY_NAMES_32) if n in self.virtual_bodies]
        for k, n in virtual:
            ids[k] = self.virtual_bodies[n][0]
        pos, quat = self.body_pos(ids), self.body_quat(ids)
        for k, n in virtual:
            pos[k] = pos[k] + quat_apply(quat[k], self.virtual_bodies[n][1])
        return pos, quat

    def palm_pose(self) -> tuple[np.ndarray, np.ndarray]:
        """Palm poses [left, right]: wrist_yaw pose composed with ``PALM_OFFSET``.
        Returns positions ``(2, 3)`` and XYZW orientations ``(2, 4)``."""
        p = self.body_pos(self.ee_body_ids)
        q = self.body_quat(self.ee_body_ids)
        return p + quat_apply(q, self.palm_offset), q

    def palm_link_pose(self) -> tuple[np.ndarray, np.ndarray] | None:
        """Pose of the URDF ``*_hand_palm_link`` bodies (cross-check of ``PALM_OFFSET``); None if absent."""
        if self.palm_body_ids is None:
            return None
        return self.body_pos(self.palm_body_ids), self.body_quat(self.palm_body_ids)

    # ------------------------------------------------------------------------------------------ control
    def pd_torque(self, q_target: np.ndarray) -> np.ndarray:
        tau = self.kp * (np.asarray(q_target, dtype=np.float64) - self.dof_pos) - self.kd * self.dof_vel
        return np.clip(tau, -self.effort_limit, self.effort_limit)

    def step(self, q_target: np.ndarray, substeps: int | None = None) -> np.ndarray:
        """One 50 Hz control step: hold ``q_target`` and run ``substeps`` physics steps, recomputing the clipped PD
        torque from the current state before every sub-step.  Returns the torque of the last sub-step.  With
        ``delay_steps > 0`` the PD loop tracks the target commanded ``delay_steps`` steps earlier.
        Also integrates the body-frame gyro over the sub-steps into ``imu_delta_quat_b`` (trapezoid of the pre- / post-sub-step
        ``qvel[3:6]``, ``mju_quatIntegrate`` = the same exponential map MuJoCo's free joint uses) and bumps ``imu_step_count``."""
        n = self.substeps if substeps is None else int(substeps)
        d, m = self.data, self.model
        q_target = self._delayed_target(q_target)
        self.last_q_target_applied = np.asarray(q_target, dtype=np.float64)
        tau = self.last_torque
        dq = np.array([1.0, 0.0, 0.0, 0.0])  # wxyz working quaternion for mju_quatIntegrate
        w_prev = np.array(d.qvel[3:6], dtype=np.float64, copy=True)
        h = self.physics_dt
        for _ in range(n):
            tau = self.pd_torque(q_target)
            d.qfrc_applied[:] = 0.0
            d.qfrc_applied[self.dof_vadr] = tau
            mujoco.mj_step(m, d)
            w = np.array(d.qvel[3:6], dtype=np.float64, copy=True)
            mujoco.mju_quatIntegrate(dq, 0.5 * (w_prev + w), h)  # dq <- dq * exp(w_mid h): body-frame rate, right multiplication
            w_prev = w
        self.imu_delta_quat_b = wxyz_to_xyzw(dq)
        self.imu_step_count += 1
        self.last_torque = np.asarray(tau, dtype=np.float64)
        return self.last_torque

    def forward(self) -> None:
        mujoco.mj_forward(self.model, self.data)

    def describe(self) -> dict:
        m = self.model
        return {
            "urdf": str(self.urdf_path),
            "model_path": str(self.model_path),
            "plant_kind": self.plant_kind,
            "foot_collision": self.foot_collision,
            "foot_collision_requested": self.foot_collision_requested,
            "root_joint": self.root_joint_name,
            "virtual_bodies": sorted(self.virtual_bodies),
            "object": self.object.describe() if self.object is not None else None,
            "native": dict(self.native_info),
            "physics_dt": self.physics_dt,
            "substeps": self.substeps,
            "control_dt": self.control_dt,
            "nq": int(m.nq),
            "nv": int(m.nv),
            "nbody": int(m.nbody),
            "ngeom": int(m.ngeom),
            "n_collision_geoms": int((m.geom_contype != 0).sum()),
            "total_mass_kg": self.total_mass,
            "armature": [float(x) for x in m.dof_armature[self.dof_vadr]],
            "damping": [float(x) for x in m.dof_damping[self.dof_vadr]],
            "frictionloss": [float(x) for x in m.dof_frictionloss[self.dof_vadr]],
            "kp": self.kp.tolist(),
            "kd": self.kd.tolist(),
            "effort_limit": self.effort_limit.tolist(),
            "self_collisions": self.self_collisions_enabled,
            "mujoco_version": mujoco.__version__,
            "perturbation": self.perturbation,
            "hand_variant": self.hand_variant,
            "physics": describe_physics(self),
        }

    @property
    def self_collisions_enabled(self) -> bool:
        """True when two robot geoms pass bitmask filtering; explicit MJCF exclusions still apply.

        Separate objects/pedestals can collide with the robot even when robot self-collision is off;
        they must not make the self-collision metadata report True.
        """
        m = self.model
        bodies = set(body_subtree_ids(m, self.root_body_id))
        robot = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in bodies and (m.geom_contype[g] or m.geom_conaffinity[g])]
        ct, ca = m.geom_contype[robot], m.geom_conaffinity[robot]
        return bool(np.any((ct[:, None] & ca[None, :]) != 0))


__all__ = [
    "HAND_NOMINAL_MASS_KG",
    "MJCF_CONTACT_MODES",
    "MJCF_JOINT_PARAM_MODES",
    "MujocoPlant",
    "NATIVE_G1_SCENE_FILE_NAME",
    "NOHAND_PALM_INERTIA",
    "NOHAND_PALM_MASS_KG",
    "PADDLE_PALM_BODY_NAMES",
    "PAYLOAD_BODY_CANDIDATES",
    "PLANT_KINDS",
    "ROOT_BODY",
    "URDF_FILE_NAME",
    "URDF_PLANT_KINDS",
    "body_subtree_ids",
    "build_model",
    "build_model_from_mjcf",
    "default_native_mjcf_path",
    "default_urdf_path",
    "nohand_spec_edit",
]
