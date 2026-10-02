"""G1 collision geometry and MuJoCo integration parameters.

Helpers preserve visual meshes while selecting explicit collision shapes and
per-joint dynamics for scene construction."""
from __future__ import annotations

import math

from hero_isaacsim import constants as HC

FOOT_COLLISIONS = ("sonic_box", "sonic_train", "legacy_mesh")
PHYSICS_PROFILES = ("hero", "sonic_deploy_adapted")
SONIC_REVISION = "b042411fae38ee4d1af9aac82a37a1f8d14d6dd0"
SONIC_BOX_COMMIT = "98a376f2220167c36e58e24c7bf88f4e4a0f1b67"
SONIC_MODEL_PATH = "gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml"
SONIC_SOURCE = f"https://github.com/NVlabs/GR00T-WholeBodyControl/blob/{SONIC_REVISION}/{SONIC_MODEL_PATH}"
SOLE_BOX_HALF_SIZE = (0.085, 0.03, 0.005)
SOLE_BOX_POS = (0.035, 0.0, -0.03)
# Contact parameters of the deployment sole box; the training capsules take the very same values.
SOLE_CONTACT_CONDIM = 3
SOLE_CONTACT_FRICTION = (1.0, 0.005, 0.0001)

# Training foot geometry: seven URDF cylinders per ankle-roll link, identical on both feet.
# Matches holosoma.simulator.isaacsim.foot_collision.SONIC_TRAIN_FOOT_CYLINDERS:
# (ankle-local xyz, fixed-axis rpy, radius, cylinder LENGTH) -- the URDF's rounded +/-1.5708 kept verbatim.
SONIC_TRAIN_SOURCE_PATH = "gear_sonic/data/assets/robot_description/urdf/g1/main.urdf"
SONIC_TRAIN_SOURCE_SHA256 = "6e107391d015ab27438026bb953b21ef8fa6a02305e65393985908941d7fd98d"
SONIC_TRAIN_SOURCE = f"https://github.com/NVlabs/GR00T-WholeBodyControl/blob/{SONIC_REVISION}/{SONIC_TRAIN_SOURCE_PATH}"
SONIC_TRAIN_FOOT_CYLINDERS = (
    ((.075, -.026, -.025), (0., 1.5708, 0.), .010, .050),
    ((.0395, -.018, -.025), (0., -1.5708, 0.), .008, .167),
    ((.039, -.010, -.025), (0., -1.5708, 0.), .010, .182),
    ((.039, .000, -.025), (0., -1.5708, 0.), .010, .186),
    ((.039, .010, -.025), (0., -1.5708, 0.), .010, .182),
    ((.0395, .018, -.025), (0., -1.5708, 0.), .008, .167),
    ((.075, .026, -.025), (0., 1.5708, 0.), .010, .050),
)
SONIC_TRAIN_CAPSULES_PER_FOOT = len(SONIC_TRAIN_FOOT_CYLINDERS)


def urdf_rpy_to_quat_wxyz(rpy) -> list[float]:
    """URDF ``<origin rpy>`` (fixed-axis roll-pitch-yaw, ``R = Rz(yaw) Ry(pitch) Rx(roll)``) as a MuJoCo ``wxyz`` quaternion."""
    r, p, y = (float(a) * 0.5 for a in rpy)
    cr, sr, cp, sp, cy, sy = math.cos(r), math.sin(r), math.cos(p), math.sin(p), math.cos(y), math.sin(y)
    return [cr * cp * cy + sr * sp * sy,
            sr * cp * cy - cr * sp * sy,
            cr * sp * cy + sr * cp * sy,
            cr * cp * sy - sr * sp * cy]


def sonic_train_capsule_name(foot_body: str, index: int) -> str:
    return f"{foot_body}_sonic_train_capsule_{index}"


def _disable_collisions(body) -> None:
    """Turn every collision geom of ``body`` and its fixed descendants into a visual (the foot mesh keeps its appearance)."""
    for g in body.geoms:
        if g.contype or g.conaffinity:
            g.contype = g.conaffinity = 0
            g.group = 1  # retain the foot's mesh as a visual surface
    for child in body.bodies:
        _disable_collisions(child)


def _add_sole_contact_geom(body, name: str, geom_type):
    """A collision-only sole geom with the training sole contact parameters (shared by the box and the seven capsules)."""
    g = body.add_geom()
    g.name = name
    g.type = geom_type
    g.contype, g.conaffinity = 1, 1
    g.condim = SOLE_CONTACT_CONDIM
    g.friction = list(SOLE_CONTACT_FRICTION)
    g.group = 3  # collision-only visual group; keep the original foot appearance
    g.rgba = [.2, .2, .2, .4]
    return g


def replace_urdf_foot_collisions(spec, profile: str) -> None:
    """Disable prior foot collision surfaces and add the selected sole geometry per ankle.

    Called BEFORE build_model applies its robot/floor collision bitmask policy;
    the new geoms therefore inherit self-collision off/on exactly like the other
    robot geoms. Geometric mesh visuals and every body/inertia remain intact
    (the URDF bodies carry explicit inertials, so geom mass never enters).
    ``sonic_box``: one box; ``sonic_train``: seven capsules from
    ``SONIC_TRAIN_FOOT_CYLINDERS`` (full cylinder length as the spine, URDF pose).
    """
    if profile not in FOOT_COLLISIONS:
        raise ValueError(f"foot_collision must be one of {FOOT_COLLISIONS}")
    if profile == "legacy_mesh":
        return
    import mujoco
    for name in HC.ANKLE_BODY_NAMES:
        body = spec.body(name)
        if body is None:
            raise ValueError(f"{profile} requires foot body {name!r}")
        _disable_collisions(body)
        if profile == "sonic_box":
            g = _add_sole_contact_geom(body, name + "_sonic_sole_box", mujoco.mjtGeom.mjGEOM_BOX)
            g.pos = SOLE_BOX_POS
            g.size = SOLE_BOX_HALF_SIZE
            g.quat = [1., 0., 0., 0.]
            continue
        for i, (xyz, rpy, radius, length) in enumerate(SONIC_TRAIN_FOOT_CYLINDERS):
            g = _add_sole_contact_geom(body, sonic_train_capsule_name(name, i), mujoco.mjtGeom.mjGEOM_CAPSULE)
            g.pos = list(xyz)
            g.quat = urdf_rpy_to_quat_wxyz(rpy)
            # MuJoCo capsule size = (radius, HALF spine).  The official Isaac importer forwards the whole URDF
            # cylinder length as the capsule spine, so the half length is length / 2 .
            g.size = [float(radius), float(length) / 2.0, 0.0]
            g.mass = 0.0  # explicit: the ankle inertial is the URDF's; a massless collider cannot change it


def adapted_joint_tables():
    return ((.01,)*len(HC.DOF_NAMES), (.05,)*len(HC.DOF_NAMES),
            tuple(.1 if n.endswith(("_wrist_pitch_joint", "_wrist_yaw_joint")) else .2 for n in HC.DOF_NAMES))


def apply_adapted_solver(model) -> None:
    """Apply the robot XML's solver defaults for MuJoCo 3.10."""
    import mujoco
    o = model.opt
    o.integrator = mujoco.mjtIntegrator.mjINT_EULER
    o.solver = mujoco.mjtSolver.mjSOL_NEWTON
    o.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL
    o.iterations, o.tolerance, o.impratio = 100, 1e-8, 1.
    o.noslip_iterations, o.noslip_tolerance = 0, 1e-6
    o.ls_iterations, o.ls_tolerance = 50, .01
    model.geom_solref[:] = [.02, 1.]
    model.geom_solimp[:] = [.9, .95, .001, .5, 2.]
    model.geom_condim[:] = 3


def describe_physics(plant) -> dict:
    """Realized scalar settings and exact foot collision geoms, after overrides."""
    import mujoco
    m = plant.model
    feet = {}
    for name in HC.ANKLE_BODY_NAMES:
        bid = plant.body_id[name]
        descendants = {bid}
        for b in range(bid+1, m.nbody):
            if int(m.body_parentid[b]) in descendants:
                descendants.add(b)
        geoms = [g for g in range(m.ngeom) if int(m.geom_bodyid[g]) in descendants
                 and (m.geom_contype[g] or m.geom_conaffinity[g])]
        feet[name] = [{"name": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g),
                       "type": mujoco.mjtGeom(int(m.geom_type[g])).name,
                       "body": mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, int(m.geom_bodyid[g])),
                       "pos_local_m": m.geom_pos[g].tolist(), "quat_wxyz": m.geom_quat[g].tolist(),
                       "size": m.geom_size[g].tolist(), "contype": int(m.geom_contype[g]),
                       "conaffinity": int(m.geom_conaffinity[g]), "condim": int(m.geom_condim[g]),
                       "friction": m.geom_friction[g].tolist(), "solref": m.geom_solref[g].tolist(),
                       "solimp": m.geom_solimp[g].tolist()} for g in geoms]
    names = ("timestep", "integrator", "solver", "iterations", "tolerance", "impratio",
             "noslip_iterations", "noslip_tolerance", "cone", "ls_iterations", "ls_tolerance")
    fc = plant.foot_collision
    uses_sonic = fc in ("sonic_box", "sonic_train") or plant.physics_profile == "sonic_deploy_adapted"
    source_url = {"sonic_box": SONIC_SOURCE, "sonic_train": SONIC_TRAIN_SOURCE}.get(fc)
    source_path = {"sonic_box": SONIC_MODEL_PATH, "sonic_train": SONIC_TRAIN_SOURCE_PATH}.get(fc)
    geometry_note = {
        "sonic_box": "official MuJoCo deployment sole box (one per ankle)",
        "sonic_train": (f"official training URDF foot: {SONIC_TRAIN_CAPSULES_PER_FOOT} cylinders per ankle imported as capsules "
                        "(spine = full URDF cylinder length, MuJoCo size[1] = length / 2; URDF origin xyz + fixed-axis rpy); "
                        "contact parameters identical to the sole box -- pure collision-geometry substitution, not PhysX equivalence"),
    }.get(fc)
    return {"profile": plant.physics_profile, "foot_collision": fc,
            "foot_collision_requested": plant.foot_collision_requested,
            "source_revision": SONIC_REVISION if uses_sonic else None,
            "foot_box_introduced_commit": SONIC_BOX_COMMIT if fc == "sonic_box" else None,
            "source_url": source_url, "source_path": source_path,
            "source_sha256": SONIC_TRAIN_SOURCE_SHA256 if fc == "sonic_train" else None,
            "foot_geometry_note": geometry_note,
            "adaptation": "Project robot inertias, policy PD/effort and self-collision choice retained" if plant.physics_profile == "sonic_deploy_adapted" else None,
            "self_collisions_requested": plant.self_collisions_requested,
            "self_collisions_enabled": plant.self_collisions_enabled,
            "mujoco_version": mujoco.__version__,
            "source_defaults_compiled_with_mujoco": "3.10.0" if plant.physics_profile == "sonic_deploy_adapted" else None,
            "timing": {"physics_dt_s": plant.physics_dt, "substeps": plant.substeps,
                       "control_dt_s": plant.control_dt},
            "joint_dynamics": {"names": list(HC.DOF_NAMES),
                               "armature": m.dof_armature[plant.dof_vadr].tolist(),
                               "damping": m.dof_damping[plant.dof_vadr].tolist(),
                               "frictionloss": m.dof_frictionloss[plant.dof_vadr].tolist()},
            "solver": {n: float(getattr(m.opt, n)) for n in names}, "feet": feet,
            "explicit_exclusions": int(m.nexclude),
            "body_mass_total_kg": float(m.body_mass.sum())}
