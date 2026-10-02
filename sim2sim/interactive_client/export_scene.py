"""Export the native tabletop scene and assets for the local browser runtime.

Run from the repository root with ``python -m sim2sim.interactive_client.export_scene --hero PATH``.
"""
from __future__ import annotations

import argparse
import copy
from contextlib import contextmanager
import hashlib
import json
import shutil
import xml.etree.ElementTree as ET
from pathlib import Path

import mujoco
import numpy as np
from PIL import Image

from hero_isaacsim import constants as HC
from sim2sim.demo_policy import load_demo_policy
from sim2sim import interactive_scene as native_scene
from sim2sim.interactive_scene import DemoScene, HAND_OPEN, HAND_CLOSED, ROBOT_URDF, TABLES, TABLE_HEIGHTS

DEFAULT_OUT = Path(__file__).resolve().parents[2] / "build/demo/sceneassets"
BROWSER_ROUND_CENTER = [0.71, 0.0]
BROWSER_ROUND_COLLISION_SEGMENTS = 96


def _round_table_collision(spec) -> None:
    """Keep the textured cylinder visual and give its top a convex disk collider.

    MuJoCo 3.13's cylinder/box contact can return an invalid normal when a
    tilted object reaches the broad, thin cylinder top near its rim. The WASM
    regression reproduces objects penetrating the entire tabletop. A single
    convex prism avoids that contact path without changing gravity, contact
    stiffness, robot collisions, or the visible table. Its inscribed 96-sided
    edge is at most 0.231 mm inside the nominal 0.43 m circle.
    """
    visual = spec.geom("demo_table_0")
    if visual.type != mujoco.mjtGeom.mjGEOM_CYLINDER:
        raise ValueError("The round table must have a cylindrical visual top.")
    radius, half_height = np.asarray(visual.size)[:2]
    angles = np.arange(BROWSER_ROUND_COLLISION_SEGMENTS) * (2 * np.pi / BROWSER_ROUND_COLLISION_SEGMENTS)
    vertices = [[radius * np.cos(a), radius * np.sin(a), z]
                for z in (-half_height, half_height) for a in angles]
    mesh_name = "demo_round_table_collision_mesh"
    spec.add_mesh(name=mesh_name, uservert=np.asarray(vertices).reshape(-1).tolist())
    spec.worldbody.add_geom(name="demo_table_top_collision", type=mujoco.mjtGeom.mjGEOM_MESH,
        meshname=mesh_name, pos=np.asarray(visual.pos).copy(), quat=np.asarray(visual.quat).copy(),
        contype=int(visual.contype), conaffinity=int(visual.conaffinity), condim=int(visual.condim),
        friction=np.asarray(visual.friction).copy(), solref=np.asarray(visual.solref).copy(),
        solimp=np.asarray(visual.solimp).copy(), margin=float(visual.margin), gap=float(visual.gap),
        rgba=[0, 0, 0, 0])
    # Explicit pairs ignore collision masks. Retarget the material contact
    # pairs too, so the visual cylinder cannot regain physical contacts.
    for pair in spec.pairs:
        if pair.geomname1 == visual.name:
            pair.geomname1 = "demo_table_top_collision"
        if pair.geomname2 == visual.name:
            pair.geomname2 = "demo_table_top_collision"
    visual.contype = visual.conaffinity = 0


@contextmanager
def _browser_table_layout():
    """Compile round tables with the same 0.28 m near edge as the workbench.

    This is a physical layout change: collision geometry, catalog support tests,
    and native reference traces all use the moved tabletop and pedestal.
    The native demo's process-wide defaults are restored on exit.
    """
    original = native_scene.TABLES
    adjusted = copy.deepcopy(original)
    adjusted["round"]["center"] = BROWSER_ROUND_CENTER.copy()
    native_scene.TABLES = adjusted
    try:
        yield
    finally:
        native_scene.TABLES = original


def _json(path: Path, value) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, separators=(",", ":"), allow_nan=False))


def _capture_scene(mode: str, table: str, height: float, policy):
    original = mujoco.MjSpec.compile
    captured = []

    def compile_spec(spec, *args, **kwargs):
        if table == "round":
            _round_table_collision(spec)
        model = original(spec, *args, **kwargs)
        captured.append(spec)
        return model

    mujoco.MjSpec.compile = compile_spec
    try:
        scene = DemoScene(table_kind=table, table_height=height, policy=policy)
    finally:
        mujoco.MjSpec.compile = original
    # Explicit IK distance queries ignore contype/conaffinity. Exclude the
    # visual-only round cylinder from fixture lists so planning and dynamics
    # use the same convex collision surface.
    scene.table_geom_ids = np.asarray([g for g in scene.table_geom_ids
        if scene.model.geom_contype[g] or scene.model.geom_conaffinity[g]], dtype=int)
    return scene, captured[-1]


@_browser_table_layout()
def export_scenes(out: Path, modes=("hero_plus",), policy_paths=None) -> dict:
    out.mkdir(parents=True, exist_ok=True)
    asset_paths = set()
    entries = {}
    policies = {mode: load_demo_policy(mode, (policy_paths or {}).get(mode)) for mode in modes}
    for mode, policy in policies.items():
        for kind in TABLES:
            for height in TABLE_HEIGHTS:
                scene, spec = _capture_scene(mode, kind, height, policy)
                m = scene.model
                # MjSpec cannot serialize texture buffers. Externalize the exact
                # deterministic native pixels, then use portable relative paths.
                for texture in spec.textures:
                    # File-backed textures (e.g. the YCB carton print) keep their path and are copied below with the meshes.
                    if texture.file or not len(texture.data):
                        continue
                    relative = f"textures/{texture.name}.png"
                    dest = out / relative
                    dest.parent.mkdir(parents=True, exist_ok=True)
                    Image.fromarray(np.frombuffer(texture.data, np.uint8).reshape(
                        texture.height, texture.width, texture.nchannel)).save(dest)
                    texture.file, texture.data = str(dest.resolve()), b""
                    asset_paths.add(relative)
                xml = ET.fromstring(spec.to_xml())
                xml.find("compiler").set("meshdir", ".")
                for element in xml.findall("asset/*"):
                    source = element.get("file")
                    if source:
                        source_path = Path(source)
                        if not source_path.is_absolute():
                            source_path = ROBOT_URDF.parent / source_path
                        relative = (f"meshes/{source_path.name}" if element.tag == "mesh"
                                    else f"textures/{source_path.name}")
                        dest = out / relative
                        dest.parent.mkdir(parents=True, exist_ok=True)
                        if source_path.resolve() != dest.resolve():
                            shutil.copyfile(source_path, dest)
                        element.set("file", relative)
                        asset_paths.add(relative)
                key = f"{mode}_{kind}_{round(height * 100)}"
                xml_name, metadata_name = f"{key}.xml", f"{key}.json"
                (out / xml_name).write_text(ET.tostring(xml, encoding="unicode"))
                objects = {k: dict(bodyId=o.body_id, qadr=o.qadr, vadr=o.vadr,
                                  geomIds=o.geom_ids.tolist(), height=o.height, restZ=o.rest_z,
                                  active=False, size=list(o.size), mass=o.mass)
                           for k, o in scene.objects.items()}
                names = lambda objtype, count: [mujoco.mj_id2name(m, objtype, i) or "" for i in range(count)]
                virtual = {n: dict(bodyId=int(b), offset=np.asarray(offset).tolist())
                           for n, (b, offset) in scene.plant.virtual_bodies.items()}
                # Retain full precision for physical parameters instead of the
                # XML writer's abbreviated decimal strings.
                physical = ("body_pos", "body_quat", "body_ipos", "body_iquat", "body_mass", "body_inertia",
                            "jnt_pos", "jnt_axis", "jnt_range", "dof_armature", "dof_damping", "dof_dampingpoly", "dof_frictionloss",
                            "geom_pos", "geom_quat", "geom_size", "geom_friction", "geom_solref", "geom_solimp",
                            "geom_margin", "geom_gap", "geom_condim",
                            "pair_friction", "pair_solref", "pair_solreffriction", "pair_solimp",
                            "pair_margin", "pair_gap", "pair_dim")
                overrides = {name: np.asarray(getattr(m, name)).reshape(-1).tolist() for name in physical}
                metadata = dict(format="tabletop_scene_v1", mode=mode, tableKind=kind, tableTopZ=height,
                    nativeMujocoVersion=mujoco.__version__, targetMujocoVersion="3.13.0",
                    xml=xml_name, physicsDt=scene.plant.physics_dt, controlDt=scene.plant.control_dt,
                    initialQpos=scene.data.qpos.tolist(), initialQvel=scene.data.qvel.tolist(),
                    defaultBodyQ=scene.initial_dof_pos.tolist(), initialRootPos=scene.initial_root_pos.tolist(),
                    jointNames29=list(HC.DOF_NAMES), jointQposAddresses29=scene.plant.dof_qadr.tolist(),
                    jointDofAddresses29=scene.plant.dof_vadr.tolist(), jointLimited=m.jnt_limited.astype(bool).tolist(),
                    fingerJointNames14=list(scene.finger_joint_names), fingerQposAddresses14=scene.finger_qadr.tolist(),
                    fingerDofAddresses14=scene.finger_vadr.tolist(), fingerLimits=scene.finger_limits.tolist(),
                    fingerKp=scene.finger_kp.tolist(), fingerKd=scene.finger_kd.tolist(),
                    fingerEffortLimit=scene.finger_effort_limit.tolist(),
                    handOpen={k:v.tolist() for k,v in HAND_OPEN.items()}, handClosed={k:v.tolist() for k,v in HAND_CLOSED.items()},
                    handModel=native_scene.TABLETOP_HAND,
                    kp=scene.plant.kp.tolist(), kd=scene.plant.kd.tolist(), effortLimit=scene.plant.effort_limit.tolist(),
                    referenceBodyNames32=list(HC.HOLOSOMA_BODY_NAMES_32),
                    bodyReferenceIds32=scene.plant.canonical_body_ids.tolist(), virtualReferenceBodies=virtual,
                    palmBodyIds={side:m.body(f"{side}_hand_palm_link").id for side in ("left", "right")},
                    ankleBodyIds={side:m.body(f"{side}_ankle_roll_link").id for side in ("left", "right")},
                    bodyNames=names(mujoco.mjtObj.mjOBJ_BODY,m.nbody), geomNames=names(mujoco.mjtObj.mjOBJ_GEOM,m.ngeom),
                    tableGeomIds=scene.table_geom_ids.tolist(), trayGeomIds=scene.tray_geom_ids.tolist(),
                    trayBottomGeomId=scene.tray_bottom_geom_id, floorGeomId=scene.floor_geom_id,
                    egoCameraId=scene.ego_camera_id, egoCamera=scene.ego_camera_info,
                    objects=objects, catalog=scene.catalog(), table=scene.state()["table"], tray=scene.tray,
                    originalGeomRGBA=scene._object_rgba.reshape(-1).tolist(),
                    originalGeomContype=scene._object_contype.tolist(), originalGeomConaffinity=scene._object_conaffinity.tolist(),
                    modelOverrides=overrides,
                    contactMaterials=native_scene.contact_material_metadata(),
                    defaultPlacements={"uiuc_i": [.46 if kind == "round" else .41, -.23, 0]})
                _json(out / metadata_name, metadata)
                # A short reference trace detects adapter indexing/PD mistakes.
                scene.place_object("uiuc_i", *metadata["defaultPlacements"]["uiuc_i"])
                trace = [dict(t=float(scene.data.time), qpos=scene.data.qpos.tolist(), qvel=scene.data.qvel.tolist())]
                for _ in range(5):
                    scene.step(scene.initial_dof_pos)
                    trace.append(dict(t=float(scene.data.time), qpos=scene.data.qpos.tolist(), qvel=scene.data.qvel.tolist()))
                _json(out / f"{key}.native_trace.json", trace)
                entries[key] = dict(mode=mode, tableKind=kind, tableHeight=height, xml=xml_name, metadata=metadata_name)
                print(key, flush=True)
    files = {name:dict(bytes=(out/name).stat().st_size,sha256=hashlib.sha256((out/name).read_bytes()).hexdigest())
             for name in sorted(asset_paths)}
    manifest = dict(format="tabletop_scene_manifest_v1", nativeMujocoVersion=mujoco.__version__,
                    targetMujocoVersion="3.13.0", scenes=entries, assets=files,
                    policySha256={mode:hashlib.sha256(policy.onnx_path.read_bytes()).hexdigest()
                                  for mode,policy in policies.items()},
                    sceneLayoutOverrides={"round": {"center": BROWSER_ROUND_CENTER, "nearEdgeX": .28,
                        "reason": "Keep the standing robot clear of the tabletop at every offered height.",
                        "tabletopCollision": "convex_prism", "collisionSegments": BROWSER_ROUND_COLLISION_SEGMENTS,
                        "visual": "textured_cylinder", "maxRadialInsetM":
                            float(native_scene.TABLES["round"]["radius"] * (1 - np.cos(np.pi / BROWSER_ROUND_COLLISION_SEGMENTS))) }},
                    compatibilityEdits=["Texture buffers externalized losslessly to PNG with colorspace preserved.",
                                        "Full-precision native physical parameters, including polynomial damping, reapplied after loading."],
                    sourceSha256={name:hashlib.sha256((Path(__file__).parents[1]/name).read_bytes()).hexdigest()
                                  for name in ("interactive_scene.py", "plant.py", "interactive_client/export_scene.py")})
    _json(out / "manifest.json", manifest)
    return manifest


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", type=Path, default=DEFAULT_OUT)
    parser.add_argument("--modes", nargs="+", choices=("hero_plus",), default=("hero_plus",))
    parser.add_argument("--hero", type=Path, help="HERO ONNX or export directory, with or without delta anchor")
    args = parser.parse_args()
    export_scenes(args.out.resolve(), args.modes, {"hero_plus": args.hero})
