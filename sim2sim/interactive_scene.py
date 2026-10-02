"""Physical scene and independent Dex3 finger control for the local interactive demo.

The 29 body joints retain the policy's existing interface. Catalog objects are compiled once as free rigid bodies;
placing an object is an editor operation, while execution moves it exclusively through MuJoCo contact dynamics.
The mug has a bottom, twelve separate wall pieces and an open handle, so its interior is genuinely hollow.
"""

from __future__ import annotations

import copy
import math
import os
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import mujoco
import numpy as np

from sim2sim.mathutil import wxyz_to_xyzw
from sim2sim.plant import MujocoPlant
from sim2sim.scene_robot import HERO_DEFAULT_DOF_POS, HERO_EFFORT_LIMIT, HERO_KD, HERO_KP, hero_to_holo

from hero_isaacsim.paths import ASSETS_ROOT, G1_ASSET_ROOT
# The scene uses the articulated Dex3 hand.
TABLETOP_HAND = os.environ.get("TABLETOP_HAND", "dex3")
HAND_SPECS = {
    "dex3": dict(
        urdf=G1_ASSET_ROOT / "g1_29dof_with_hand_rev_1_0.urdf",
        finger_suffixes=("thumb_0", "thumb_1", "thumb_2", "index_0", "index_1", "middle_0", "middle_1"),
        hand_open={"left": [0.0, -0.7, 0.0, 0.0, 0.0, 0.0, 0.0], "right": [0.0, 0.7, 0.0, 0.0, 0.0, 0.0, 0.0]},
        # Clip finger targets by joint name to respect the URDF limits.
        hand_closed={"left": [0.0, 0.0, 1.2, -1.2, -1.2, -1.2, -1.2], "right": [0.0, 0.0, -1.2, 1.57, 1.2, 1.2, 1.2]},
        finger_effort=[2.45, 1.4, 1.4, 1.4, 1.4, 1.4, 1.4], finger_kp=1.2, finger_kd=0.04, finger_armature=0.001, finger_damping=0.01,
        rename_prefixes=None, virtual_palm=None),

}
if TABLETOP_HAND not in HAND_SPECS:
    raise ValueError(f"TABLETOP_HAND must be one of {sorted(HAND_SPECS)}")
HAND = HAND_SPECS[TABLETOP_HAND]
ROBOT_URDF = HAND["urdf"]
TABLE_HEIGHTS = (0.50, 0.74, 0.88)
# Clip the two far diagonal corners while retaining space along each hand's side.

PLACEMENT_REGION = dict(x_min=0.30, x_max=0.57, y_min=-0.40, y_max=0.40,
                        corner_cut=dict(x_start=0.45, y_start=0.24))
HAND_WORKSPACES = {
    "left": dict(copy.deepcopy(PLACEMENT_REGION), y_min=0.0),
    "right": dict(copy.deepcopy(PLACEMENT_REGION), y_max=0.0),
}


def contains_placement(region: dict, x: float, y: float, epsilon: float = 1e-9) -> bool:
    """Boundary-inclusive editor polygon, independent of tabletop and contact validation."""
    if not all(math.isfinite(value) for value in (x, y, epsilon)) or epsilon < 0:
        return False
    if not (region["x_min"] - epsilon <= x <= region["x_max"] + epsilon
            and region["y_min"] - epsilon <= y <= region["y_max"] + epsilon):
        return False
    cut = region.get("corner_cut")
    if cut is None:
        return True
    lateral = max(abs(region["y_min"]), abs(region["y_max"]))
    dx, dy = region["x_max"] - cut["x_start"], lateral - cut["y_start"]
    return dy * (x - cut["x_start"]) + dx * (abs(y) - lateral) <= epsilon * math.hypot(dx, dy)

# metres and the apple/hand sliding friction. Current defaults use 80 x 76 mm and hand friction 1.4375.
_APPLE_FRUIT_HEIGHT = float(os.environ.get("TABLETOP_APPLE_FRUIT_HEIGHT", "0.080"))
_APPLE_DIAMETER = float(os.environ.get("TABLETOP_APPLE_DIAMETER", "0.076"))
_APPLE_HAND_FRICTION = float(os.environ.get("TABLETOP_APPLE_HAND_FRICTION", "1.4375"))
# YCB object 003_cracker_box (the Cheez-It box; Calli et al., "The YCB Object and Model Set", 2015). The recentred
# 16k-face textured mesh and the 1024 px texture are prepared by scripts/fetch_ycb_cracker_box.py; the collision
# shape is the mesh's axis-aligned bounding box. The demo mass is 30% below the YCB catalogue's 411 g.
_CRACKER_BOX_DIR = ASSETS_ROOT / "objects/ycb/003_cracker_box"
_CRACKER_BOX_SIZE = [0.0718, 0.164, 0.2134]
OBJECTS = {
    "apple": dict(id="apple", label="Apple", size=[_APPLE_DIAMETER, _APPLE_DIAMETER, _APPLE_FRUIT_HEIGHT + 0.010],
                  height=_APPLE_FRUIT_HEIGHT + 0.010, mass=0.180,
                  color=[0.66, 0.045, 0.035, 1.0], footprint_radius=_APPLE_DIAMETER / 2, shape="apple",
                  fruit_body_height=_APPLE_FRUIT_HEIGHT, fruit_body_center=[0.0, 0.0, -0.005]),
    "can": dict(id="can", label="Can", size=[0.056, 0.056, 0.105], height=0.105, mass=0.085,
                color=[0.28, 0.63, 0.71, 1.0], footprint_radius=0.028),
    "bottle": dict(id="bottle", label="Bottle", size=[0.054, 0.054, 0.155], height=0.155, mass=0.100,
                   color=[0.42, 0.64, 0.38, 1.0], footprint_radius=0.027),
    "mug": dict(id="mug", label="Mug", size=[0.108, 0.074, 0.090], height=0.090, mass=0.120,
                color=[0.82, 0.66, 0.39, 1.0], footprint_radius=0.072),
    "uiuc_i": dict(id="uiuc_i", label="UIUC Block I", size=[0.120, 0.050, 0.240], height=0.240, mass=0.100,
                   color=[19 / 255, 41 / 255, 75 / 255, 1.0], accent_color=[1.0, 95 / 255, 5 / 255, 1.0],
                   footprint_radius=0.065, shape="block_i", grasp_hint="stem",
                   grasp_region=dict(center=[0, 0, 0], size=[0.052, 0.049, 0.180])),
    "cracker_box": dict(id="cracker_box", label="Cheez-It box", size=list(_CRACKER_BOX_SIZE), height=_CRACKER_BOX_SIZE[2], mass=0.2877,
                        color=[0.78, 0.15, 0.10, 1.0], footprint_radius=0.0895, shape="box", grasp_hint="end_face",
                        source="YCB 003_cracker_box"),
}
# Nominal dry-contact priors for the demo props, not measurements of the real
# robot or objects. Friction belongs to a contact pair, not an intrinsic
# material constant that can be combined by a universal mixing formula.
# Explicit pairs prevent MuJoCo's default maximum-per-geom rule from replacing
# these estimates. The fallback uses the Dex3 contact parameters.

DEX3_SLIDING_FRICTION = 1.5
OBJECT_CONTACT_MATERIALS = {
    "apple": dict(material="Fresh apple skin", hand=_APPLE_HAND_FRICTION, tabletop=0.35, tray=0.30,
                  handBasis="Optimistic dry rubber/apple-skin simulation estimate, estimated hand friction; not measured."),
    "can": dict(material="Coated aluminum", hand=1.0, tabletop=0.30, tray=0.25,
                handBasis="Adams dry rubber/aluminium reference static 0.80, dynamic 0.76; simulation hand coefficient, not calibrated."),
    "bottle": dict(material="PET plastic", hand=1.0625, tabletop=0.35, tray=0.30,
                   handBasis="Upper dry-grip hand contact estimate; analogy to rubber/acrylic and rubber/nylon, not measured PET."),
    "mug": dict(material="Glazed ceramic", hand=1.0625, tabletop=0.35, tray=0.30,
                handBasis="Upper dry rubber/glaze grip estimate; no matched measurement."),

    "uiuc_i": dict(material="Painted lightweight wood", hand=1.4625, tabletop=0.585, tray=0.52,
                   handBasis="Optimistic dry soft-rubber/wood hand contact estimate; paint and pad differ from published tests."),

    "cracker_box": dict(material="Printed paperboard carton", hand=1.1375, tabletop=0.455, tray=0.39,
                        handBasis="Dry rubber/coated-paperboard hand contact estimate; not measured."),
}
# MuJoCo's torsional friction coefficient is a length: torque / normal force.
# Use mu times a small effective contact moment arm.
CONTACT_TORSIONAL_ARM_M = dict(hand=0.004, tabletop=0.008, tray=0.006)
CONTACT_ROLLING_M = 0.0001  # Inactive at condim=4; retained for an explicit model definition.


def apple_skin_geometry(segments: int = 64) -> tuple[np.ndarray, np.ndarray]:
    """Rounded apple shoulders, a shallow stem well and restrained asymmetry.

    Smooth the visual meridian while retaining a sparse hull sample for native
    distance queries. Both use the same profile and 80 x 76 mm fruit envelope;
    the small support disk is at -45 mm and the decorative stem ends at +45 mm.
    """
    angles = np.arange(segments) * (2 * np.pi / segments)
    control = np.array([[-.055, .0105], [-.0535, .0145], [-.048, .0215],
                        [-.037, .0275], [-.021, .0315], [-.003, .0355],
                        [.014, .0376], [.026, .0370], [.036, .0335],
                        [.042, .027], [.0445, .0185], [.0430, .011],
                        [.0393, .004]])
    # Catmull-Rom interpolation gives smooth tangents between segments.
    # Sparse collision samples keep the same knots.
    subdivisions = 1 if segments <= 24 else 6
    knots = np.vstack([2 * control[0] - control[1], control,
                       2 * control[-1] - control[-2]])
    profile = []
    for k in range(len(control) - 1):
        p0, p1, p2, p3 = knots[k:k + 4]
        for t in np.arange(subdivisions) / subdivisions:
            profile.append(.5 * (2 * p1 + (-p0 + p2) * t
                + (2*p0 - 5*p1 + 4*p2 - p3) * t*t
                + (-p0 + 3*p1 - 3*p2 + p3) * t*t*t))
    profile.append(control[-1])
    rings = []
    for z, radius in profile:
        u = (z + .055) / .100
        crown = np.exp(-((u - .91) / .12) ** 2)
        base = np.exp(-((u - .05) / .07) ** 2)
        lobes = .004 * np.cos(5*angles + .3) + .003 * np.cos(3*angles - .7)
        asymmetry = .008 * np.cos(angles + .5) * np.sin(np.pi * u)
        radial = radius * (1 + asymmetry + crown * lobes + base * lobes * .45)
        # Sub-millimetre, unequal crown relief rather than five raised points.
        vertical = z + crown * (.00022*np.cos(3*angles - .4) + .00013*np.cos(5*angles + .7))
        rings.append(np.c_[radial * np.cos(angles), radial * np.sin(angles),
                           np.broadcast_to(vertical, angles.shape)])
    vertices = np.vstack(rings)
    vertices[:, :2] *= .038 / np.linalg.norm(vertices[:, :2], axis=1).max()
    zmin, zmax = vertices[:, 2].min(), vertices[:, 2].max()
    vertices[:, 2] = -.055 + (vertices[:, 2] - zmin) * (.100 / (zmax - zmin))
    bottom, top = len(vertices), len(vertices) + 1
    vertices = np.vstack([vertices, [0, 0, -.055], [0, 0, .0385]])
    # The reference profile above is 100 mm tall; map it to the requested
    # height to the catalog envelope so the fruit is only slightly taller
    # than it is wide. The collision sample follows this identical mapping.
    apple = OBJECTS["apple"]
    vertices[:, 2] = -apple["height"] / 2 + (vertices[:, 2] + .055) * (apple["fruit_body_height"] / .100)
    faces = []
    for ring in range(len(profile) - 1):
        for k in range(segments):
            a, b = ring * segments + k, ring * segments + (k + 1) % segments
            faces.extend(((a, b, b + segments), (a, b + segments, a + segments)))
    for k in range(segments):
        following = (k + 1) % segments
        faces.append((bottom, following, k))
        faces.append((top, (len(profile) - 1) * segments + k,
                      (len(profile) - 1) * segments + following))
    return vertices, np.asarray(faces, dtype=np.int32)


def apple_stem_geometry() -> tuple[np.ndarray, np.ndarray]:
    """A short, tapered curved stalk, kept as one decorative geom."""
    rings, sectors, vertices = 12, 12, []
    for t in np.linspace(0, 1, rings):
        center = np.array([-.001 + .006*t*t, -.0015*np.sin(np.pi*t), .0375 + .017*t])
        tangent = np.array([.012*t, -.0015*np.pi*np.cos(np.pi*t), .017])
        tangent /= np.linalg.norm(tangent)
        normal = np.cross(tangent, [0, 1, 0]); normal /= np.linalg.norm(normal)
        binormal = np.cross(tangent, normal)
        radius = .0019 - .00075*t
        for a in np.arange(sectors) * (2*np.pi/sectors):
            vertices.append(center + radius * (np.cos(a)*normal + np.sin(a)*binormal))
    vertices = np.asarray(vertices)
    vertices[:, 2] += OBJECTS["apple"]["height"] / 2 - vertices[:, 2].max()
    bottom, top = len(vertices), len(vertices) + 1
    vertices = np.vstack([vertices, vertices[:sectors].mean(axis=0), vertices[-sectors:].mean(axis=0)])
    faces = []
    for r in range(rings - 1):
        for k in range(sectors):
            a, b = r*sectors+k, r*sectors+(k+1)%sectors
            faces.extend(((a, b, b+sectors), (a, b+sectors, a+sectors)))
    for k in range(sectors):
        following = (k+1)%sectors
        faces.extend(((bottom, following, k), (top, (rings-1)*sectors+k, (rings-1)*sectors+following)))
    return vertices, np.asarray(faces, dtype=np.int32)


def apple_leaf_geometry() -> tuple[np.ndarray, np.ndarray]:
    """A thin curved blade with a raised midrib and gently twisted edges."""
    rows, columns, surface = 15, 7, []
    for t in np.linspace(0, 1, rows):
        width = .00012 + .0048 * np.sin(np.pi*t)**.85
        for v in np.linspace(-1, 1, columns):
            surface.append([.003 + .026*t,
                            .001 + .006*t + .002*np.sin(np.pi*t) + width*v,
                            .046 + .0055*np.sin(np.pi*t) + .0025*t
                            - .0014*v*v*np.sin(np.pi*t) + .0007*v*np.sin(2*np.pi*t)])
    surface = np.asarray(surface)
    vertices = np.vstack([surface + [0, 0, .00010], surface - [0, 0, .00010]])
    vertices[:, 2] += OBJECTS["apple"]["height"] / 2 - .055
    n, faces = len(surface), []
    for i in range(rows - 1):
        for j in range(columns - 1):
            a = i*columns+j; b, c, d = a+1, a+columns, a+columns+1
            faces.extend(((a, c, d), (a, d, b), (a+n, d+n, c+n), (a+n, b+n, d+n)))
    boundary = ([i*columns for i in range(rows)]
                + [(rows-1)*columns+j for j in range(1, columns)]
                + [i*columns+columns-1 for i in range(rows-2, -1, -1)]
                + list(range(columns-2, 0, -1)))
    for a, b in zip(boundary, boundary[1:] + boundary[:1]):
        faces.extend(((a, a+n, b+n), (a, b+n, b)))
    return vertices, np.asarray(faces, dtype=np.int32)


def contact_material_metadata() -> dict:
    hand_scale = 1.0
    objects = copy.deepcopy(OBJECT_CONTACT_MATERIALS)
    return dict(schema="tabletop_contact_materials_v2", calibrated=False,
                assumption="Nominal clean, dry surfaces; validate against physical props before hardware transfer.",
                surfaces=dict(hand="Dry compliant textured rubber grasp surfaces (assumed)", tabletop="Sealed wood",
                              tray="Rigid polypropylene"),
                handSlidingFriction=DEX3_SLIDING_FRICTION * hand_scale, handObjectCombination="explicit_pair",
                objects=objects,
                torsionalEffectiveMomentArmM=CONTACT_TORSIONAL_ARM_M.copy(),
                rollingFrictionM=CONTACT_ROLLING_M, condim=4,
                frictionModel="One Coulomb sliding coefficient; separate static and dynamic friction are not modeled.",
                references=[dict(source="Hexagon Adams View 2023.4 User Guide", pages="234-235",
                    url="https://help-be.hexagonmi.com/bundle/Adams_2023.4_Adams_View_User_Guide/raw/resource/enus/Adams_2023.4_Adams_View_User_Guide.pdf",
                    dryRubberAluminium=dict(static=0.80, dynamic=0.76),
                    note="Generalized literature-based table, not measurements of Dex3 or these props."),
                    dict(source="Luo et al., BioResources 9(4), 7372-7381 (2014)",
                         url="https://doi.org/10.15376/biores.9.4.7372-7381",
                         note="Rubber belt against wood panels; different surfaces and loading from a painted wood grasp prop.")],
                combination="Explicit object-hand, object-tabletop and object-tray contact-pair estimates. "
                            "Other contacts use MuJoCo geom defaults; the hand geom value is a fallback, not a universal material coefficient.")


TABLES = {
    "workbench": dict(id="workbench", label="Workbench", shape="rectangle", center=[0.61, 0.0], half_size=[0.33, 0.50]),
    "round": dict(id="round", label="Round table", shape="circle", center=[0.56, 0.0], radius=0.43),
    "pedestal": dict(id="pedestal", label="Pedestal", shape="rectangle", center=[0.46, 0.0], half_size=[0.18, 0.46]),
}
# Extend the far edge by 10 cm while keeping the robot-facing edge at x=0.32 m.
TRAY_CENTER = (0.52, 0.0)
TRAY_OUTER_SIZE = (0.40, 0.28)
TRAY_BOTTOM_THICKNESS = 0.006
TRAY_WALL_THICKNESS = 0.008
TRAY_WALL_HEIGHT = 0.036
EGO_CAMERA = dict(name="demo_ego", label="Robot first-person view", body="head_link",
                  local_position=[0.11, 0.0, 0.46], fovy=86.0, pitch_down_degrees=45.0)
# Scene-owned world-space lights and textured floor are enabled explicitly
# to avoid adding studio lights twice.
DEMO_RENDER_STYLE = dict(floor_rgba=(0.73, 0.72, 0.69, 1.0), floor_grid=False, add_sun=False,
                         headlight_ambient=(0.23, 0.24, 0.25), headlight_diffuse=(0.18, 0.18, 0.18))
FINGER_SUFFIXES = tuple(HAND["finger_suffixes"])
FINGERS_PER_HAND = len(FINGER_SUFFIXES)
HAND_OPEN = {side: np.array(HAND["hand_open"][side], dtype=float) for side in ("left", "right")}
HAND_CLOSED = {side: np.array(HAND["hand_closed"][side], dtype=float) for side in ("left", "right")}
FINGER_EFFORT = np.tile(np.array(HAND["finger_effort"], dtype=float), 2)
PARK_Z = -5.0


@dataclass
class DemoObject:
    kind: str
    body_id: int
    body_name: str
    joint_name: str
    qadr: int
    vadr: int
    geom_ids: np.ndarray
    geom_names: tuple[str, ...]
    size: tuple[float, float, float]
    height: float
    mass: float
    rest_z: float
    active: bool = False

    @property
    def id(self): return self.kind

    @property
    def bodyid(self): return self.body_id

    @property
    def jointqadr(self): return self.qadr

    @property
    def restz(self): return self.rest_z

    def __getitem__(self, key): return getattr(self, key)


class DemoScene:
    """A standing G1, physical table and one selected object from the catalog.

    ``yaw`` is in radians. ``reset`` restores the robot and the user's latest placements. All quaternions returned
    by the public API are xyzw. Use ``scene.step(q_body)`` rather than ``scene.plant.step`` so fingers are controlled.
    ``robot_setback`` moves the starting root along world -X while tables and object placements stay fixed.
    """

    def __init__(self, table_kind: str = "workbench", table_height: float = 0.74, policy=None, *, robot_setback: float = 0.03):
        if table_kind not in TABLES:
            raise ValueError(f"unknown table kind: {table_kind!r}")
        if not np.isfinite(table_height) or not any(abs(float(table_height) - h) < 1e-6 for h in TABLE_HEIGHTS):
            raise ValueError(f"table_height must be one of {TABLE_HEIGHTS}")
        if not np.isfinite(robot_setback) or robot_setback < 0:
            raise ValueError("robot_setback must be a finite non-negative distance in metres")
        self.robot_setback = float(robot_setback)
        self.table_kind, self.table_height = table_kind, float(table_height)
        self.table_top_z = self.table_height
        self.tray = self._tray_geometry(self.table_height)
        self.tray_info = self.tray
        self.policy = policy
        # Reset joint positions must match the loaded policy to avoid a control transient.
        kwargs = dict(kp=hero_to_holo(HERO_KP), kd=hero_to_holo(HERO_KD), effort_limit=hero_to_holo(HERO_EFFORT_LIMIT))
        kwargs.update({k: getattr(policy, k) for k in ("kp", "kd", "effort_limit") if policy is not None and hasattr(policy, k)})
        self.plant = MujocoPlant(ROBOT_URDF, spec_callback=self._add_scene,
                                 extra_free_joint_names=tuple(f"demo_{k}_free" for k in OBJECTS), **kwargs)
        self.model, self.data = self.plant.model, self.plant.data
        m = self.model
        self._placement_data = mujoco.MjData(m)
        self.objects: dict[str, DemoObject] = {}
        for kind, info in OBJECTS.items():
            name = f"demo_{kind}"
            bid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_BODY, name)
            jid = mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, name + "_free")
            gids = np.flatnonzero(m.geom_bodyid == bid)
            self.objects[kind] = DemoObject(kind, bid, name, name + "_free", int(m.jnt_qposadr[jid]),
                int(m.jnt_dofadr[jid]), gids, tuple(mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, int(g)) for g in gids),
                tuple(info["size"]), info["height"], info["mass"], self.table_height + info["height"] / 2 + 0.001)
        self.object_info = self.objects
        self._object_rgba = m.geom_rgba.copy()
        self._object_contype = m.geom_contype.copy()
        self._object_conaffinity = m.geom_conaffinity.copy()
        self.table_geom_ids = np.asarray([g for g in range(m.ngeom)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("demo_table")], dtype=int)
        self.tray_geom_ids = np.asarray([g for g in range(m.ngeom)
            if (mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_GEOM, g) or "").startswith("demo_tray_")], dtype=int)
        self.tray_bottom_geom_id = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "demo_tray_bottom"))
        self.tray["geom_ids"] = self.tray_geom_ids.tolist()
        self.tray["bottom_geom_id"] = self.tray_bottom_geom_id
        self.floor_geom_id = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_GEOM, "floor"))
        self.ego_camera_id = int(mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_CAMERA, EGO_CAMERA["name"]))
        self.ego_camera_info = dict(copy.deepcopy(EGO_CAMERA), id=self.ego_camera_id)
        self._placements: dict[str, tuple[float, float, float]] = {}
        self.finger_joint_names = tuple(f"{side}_hand_{finger}_joint" for side in ("left", "right") for finger in FINGER_SUFFIXES)
        jids = np.array([mujoco.mj_name2id(m, mujoco.mjtObj.mjOBJ_JOINT, n) for n in self.finger_joint_names], dtype=int)
        if np.any(jids < 0):
            raise RuntimeError(f"the interactive scene requires the movable {TABLETOP_HAND} finger joints {self.finger_joint_names}")
        self.finger_qadr, self.finger_vadr = m.jnt_qposadr[jids].copy(), m.jnt_dofadr[jids].copy()
        self.finger_limits = m.jnt_range[jids].copy()
        self.finger_kp = np.full(2 * FINGERS_PER_HAND, float(HAND["finger_kp"]))
        self.finger_kd = np.full(2 * FINGERS_PER_HAND, float(HAND["finger_kd"]))
        self.finger_effort_limit = FINGER_EFFORT.copy()
        self.hand_closure = {"left": 0.0, "right": 0.0}
        self._finger_goal = np.concatenate([HAND_OPEN[s] for s in ("left", "right")])
        self._finger_target = self._finger_goal.copy()
        self.last_finger_torque = np.zeros(2 * FINGERS_PER_HAND)
        default_pose = getattr(policy, "default_dof_pos", hero_to_holo(HERO_DEFAULT_DOF_POS))
        self.initial_dof_pos = np.asarray(default_pose, dtype=float).reshape(29).copy()
        if self.table_height >= 0.86:
            # HERO's straight fingers intersect an 88 cm tabletop. Fold the arms back while keeping its leg stance;
            # the online controller starts from this actual pose and plans the approach above the table edge.
            self.initial_dof_pos[15:] = [0.2, 0.2, 0, 0.6, 0, 0, 0, 0.2, -0.2, 0, 0.6, 0, 0, 0]
        self.initial_root_pos = np.array([-self.robot_setback, 0.0, 0.80])
        self.plant.reset(self.initial_root_pos, np.array([0, 0, 0, 1.0]), self.initial_dof_pos)
        self.initial_root_pos[2] -= self._sole_bottom()
        self.reset()

    def catalog(self) -> dict[str, Any]:
        return copy.deepcopy(dict(objects=list(OBJECTS.values()), tables=list(TABLES.values()), heights=list(TABLE_HEIGHTS),
                                  placement_region=PLACEMENT_REGION, hand_workspaces=HAND_WORKSPACES,
                                  max_objects=1, placement_mode="replace", yaw_unit="radians", tray=self.tray,
                                  ego_camera=self.ego_camera_info, robot_setback_m=self.robot_setback))

    @staticmethod
    def _tray_geometry(table_height: float) -> dict[str, Any]:
        x, y = TRAY_CENTER
        hx, hy = (float(v) / 2 for v in TRAY_OUTER_SIZE)
        wall = TRAY_WALL_THICKNESS
        bottom_top = table_height + TRAY_BOTTOM_THICKNESS
        return dict(id="drop_tray", label="Drop tray", center=[x, y], outer_size=list(TRAY_OUTER_SIZE),
                    inner_size=[2 * (hx - wall), 2 * (hy - wall)], wall_thickness=wall,
                    bottom_thickness=TRAY_BOTTOM_THICKNESS, wall_height=TRAY_WALL_HEIGHT,
                    outer_bounds=dict(x_min=x - hx, x_max=x + hx, y_min=y - hy, y_max=y + hy),
                    inner_bounds=dict(x_min=x - hx + wall, x_max=x + hx - wall, y_min=y - hy + wall, y_max=y + hy - wall),
                    bottom_z=float(table_height), bottom_top_z=float(bottom_top), wall_top_z=float(bottom_top + TRAY_WALL_HEIGHT),
                    drop_position=[x, y, float(bottom_top)], fixture=True,
                    color=[0.42, 0.58, 0.55, 1.0])

    def _add_scene(self, spec):
        """Add the interactive scene geometry before model compilation."""
        self._add_studio_materials(spec)
        # This URDF's head frame is below its mesh: the shell spans local z=.32..53 m.
        # Put the lens 34 mm ahead of the shell and fix it to the actual head body.
        # MuJoCo looks along camera -z; image right is robot -y, with a downward pitch.
        pitch = math.radians(EGO_CAMERA["pitch_down_degrees"])
        spec.body(EGO_CAMERA["body"]).add_camera(name=EGO_CAMERA["name"], pos=EGO_CAMERA["local_position"],
            xyaxes=[0, -1, 0, math.sin(pitch), 0, math.cos(pitch)], fovy=EGO_CAMERA["fovy"],
            mode=mujoco.mjtCamLight.mjCAMLIGHT_FIXED)
        # Explicit rotor inertia regularizes the small finger inertias at 1 kHz.
        for side in ("left", "right"):
            for finger in FINGER_SUFFIXES:
                j = spec.joint(f"{side}_hand_{finger}_joint")
                j.armature = float(HAND["finger_armature"])
                j.damping = np.full(3, float(HAND["finger_damping"])) if np.ndim(j.damping) else float(HAND["finger_damping"])
        self._add_table(spec)
        self._add_tray(spec)
        for kind, info in OBJECTS.items():
            contact = OBJECT_CONTACT_MATERIALS[kind]
            body = spec.worldbody.add_body(name=f"demo_{kind}", pos=[0, 0, PARK_Z], gravcomp=1.0)
            body.add_freejoint(name=f"demo_{kind}_free")
            parts = []

            def part(shape, size, pos=(0, 0, 0), *, color=None, quat=(1, 0, 0, 0), fromto=None,
                     collision=True, meshname=None, material=None):
                kwargs = dict(name=f"demo_{kind}_geom_{len(parts)}", type=shape, size=list(size), pos=list(pos),
                              quat=list(quat), rgba=list(info["color"] if color is None else color), mass=1.0 if collision else 0.0,
                              material=material or {"can": "demo_metal", "bottle": "demo_glazed", "mug": "demo_glazed"}.get(kind, "demo_satin"),
                              contype=int(collision), conaffinity=int(collision), condim=4,
                              friction=[contact["tabletop"], contact["tabletop"] * CONTACT_TORSIONAL_ARM_M["tabletop"], CONTACT_ROLLING_M],
                              solref=[0.008, 1.0])
                if fromto is not None:
                    kwargs["fromto"] = list(fromto)
                if meshname is not None:
                    kwargs["meshname"] = meshname
                parts.append(body.add_geom(**kwargs))

            if kind == "apple":
                skin_vertices, skin_faces = apple_skin_geometry()
                # Use an independent convex hull for dynamics. A 20-segment
                # sample matches the skin to sub-millimetre radial accuracy,
                # while the stem, leaf and shallow crown remain visual only.
                collision_vertices, _ = apple_skin_geometry(20)
                spec.add_mesh(name="demo_apple_collision_mesh", uservert=collision_vertices.reshape(-1).tolist(),
                              inertia=mujoco.mjtMeshInertia.mjMESH_INERTIA_CONVEX)
                spec.add_mesh(name="demo_apple_skin_mesh", uservert=skin_vertices.reshape(-1).tolist(),
                              userface=skin_faces.reshape(-1).tolist(), smoothnormal=True)
                part(mujoco.mjtGeom.mjGEOM_MESH, [1, 1, 1], meshname="demo_apple_collision_mesh", color=[0, 0, 0, 0])
                part(mujoco.mjtGeom.mjGEOM_MESH, [1, 1, 1], meshname="demo_apple_skin_mesh",
                     material="demo_apple_skin", collision=False)
                stem_vertices, stem_faces = apple_stem_geometry()
                spec.add_mesh(name="demo_apple_stem_mesh", uservert=stem_vertices.reshape(-1).tolist(),
                              userface=stem_faces.reshape(-1).tolist(), smoothnormal=True)
                part(mujoco.mjtGeom.mjGEOM_MESH, [1, 1, 1], meshname="demo_apple_stem_mesh",
                     color=[.22, .11, .045, 1], collision=False)
                leaf_vertices, leaf_faces = apple_leaf_geometry()
                spec.add_mesh(name="demo_apple_leaf_mesh", uservert=leaf_vertices.reshape(-1).tolist(),
                              userface=leaf_faces.reshape(-1).tolist(), smoothnormal=True)
                part(mujoco.mjtGeom.mjGEOM_MESH, [1, 1, 1], meshname="demo_apple_leaf_mesh",
                     color=[.13, .30, .055, 1], collision=False)
            elif kind == "can":
                part(mujoco.mjtGeom.mjGEOM_CYLINDER, [0.028, 0.0515, 0])
                for z in (-0.0515, 0.0515):
                    part(mujoco.mjtGeom.mjGEOM_CYLINDER, [0.0275, 0.001, 0], [0, 0, z], color=[0.65, 0.69, 0.72, 1])
            elif kind == "bottle":
                part(mujoco.mjtGeom.mjGEOM_CYLINDER, [0.027, 0.0525, 0], [0, 0, -0.025])
                part(mujoco.mjtGeom.mjGEOM_SPHERE, [0.024, 0, 0], [0, 0, 0.025])
                part(mujoco.mjtGeom.mjGEOM_CYLINDER, [0.012, 0.025, 0], [0, 0, 0.051])
                part(mujoco.mjtGeom.mjGEOM_CYLINDER, [0.014, 0.003, 0], [0, 0, 0.0745], color=[0.83, 0.83, 0.77, 1])
            elif kind == "uiuc_i":
                # One rigid I-shaped object: two 12 cm bars and an 18 cm clear central stem. The structural blue
                # boxes are 49 mm thick; thin orange faces bring the visible full thickness to exactly 50 mm.
                for z in (-0.105, 0.105):
                    part(mujoco.mjtGeom.mjGEOM_BOX, [0.060, 0.0245, 0.015], [0, 0, z])
                part(mujoco.mjtGeom.mjGEOM_BOX, [0.026, 0.0245, 0.090])
                for y in (-0.0248, 0.0248):
                    for z in (-0.105, 0.105):
                        part(mujoco.mjtGeom.mjGEOM_BOX, [0.055, 0.0002, 0.011], [0, y, z],
                             color=info["accent_color"], collision=False)
                    part(mujoco.mjtGeom.mjGEOM_BOX, [0.021, 0.0002, 0.094], [0, y, 0],
                         color=info["accent_color"], collision=False)
            elif kind == "cracker_box":
                # One rigid carton: the collision box is the textured mesh's bounding box (transparent so the
                # renderer shows only the print); the YCB mesh is visual-only and massless.
                mesh_file = _CRACKER_BOX_DIR / "cracker_box_visual.obj"
                if not mesh_file.exists():
                    raise FileNotFoundError(f"{mesh_file} missing; run scripts/fetch_ycb_cracker_box.py")
                part(mujoco.mjtGeom.mjGEOM_BOX, [v / 2 for v in info["size"]], color=[0, 0, 0, 0])
                spec.add_mesh(name="demo_cracker_box_visual_mesh", file=str(mesh_file))
                part(mujoco.mjtGeom.mjGEOM_MESH, [1, 1, 1], meshname="demo_cracker_box_visual_mesh",
                     material="demo_cracker_box_print", color=[1, 1, 1, 1], collision=False)
            elif kind == "mug":
                part(mujoco.mjtGeom.mjGEOM_CYLINDER, [0.037, 0.003, 0], [0, 0, -0.042])
                for k in range(12):
                    a = k * 2 * math.pi / 12
                    part(mujoco.mjtGeom.mjGEOM_BOX, [0.003, 0.034 * math.tan(math.pi / 12) * 1.03, 0.042],
                         [0.034 * math.cos(a), 0.034 * math.sin(a), 0.003], quat=[math.cos(a / 2), 0, 0, math.sin(a / 2)])
                # D-shaped open handle, capsules sharing endpoints; +x is the handle side.
                pts = [np.array([0.036 + 0.031 * math.sin(a), 0, 0.025 * math.cos(a)]) for a in np.linspace(0, math.pi, 9)]
                for a, b in zip(pts[:-1], pts[1:]):
                    part(mujoco.mjtGeom.mjGEOM_CAPSULE, [0.0045, 0, 0], fromto=np.r_[a, b])
            # Uniform density, so small decorative pieces do not carry the same mass as the main body.
            volumes = []
            for g in parts:
                if not (g.contype or g.conaffinity):
                    volumes.append(0.0)
                    continue
                a, b, c = np.asarray(g.size)
                if g.type == mujoco.mjtGeom.mjGEOM_BOX: v = 8 * a * b * c
                elif g.type == mujoco.mjtGeom.mjGEOM_CYLINDER: v = math.pi * a * a * 2 * b
                elif g.type == mujoco.mjtGeom.mjGEOM_SPHERE: v = 4 / 3 * math.pi * a ** 3
                elif g.type == mujoco.mjtGeom.mjGEOM_MESH: v = 1.0  # the apple has one physical hull, carrying its full mass
                else: v = math.pi * a * a * np.linalg.norm(np.asarray(g.fromto)[3:] - np.asarray(g.fromto)[:3]) + 4 / 3 * math.pi * a ** 3
                volumes.append(v)
            for g, volume in zip(parts, volumes):
                g.mass = info["mass"] * volume / sum(volumes)
        self._add_object_contact_pairs(spec)

    @staticmethod
    def _add_object_contact_pairs(spec):
        """Assign friction without changing shapes or implicit normal-contact settings.

        Explicit pairs bypass collision masks. Include only physical object and
        hand geoms, the tabletop and tray; inactive objects are parked far below
        these surfaces. Object-object pairs would collide at their shared park
        position, so those retain ordinary mask-filtered contacts.
        """
        physical = [g for g in spec.geoms if g.contype or g.conaffinity]
        hands = [g for g in physical if g.parent.name.startswith(("left_hand_", "right_hand_"))]
        hand_scale = 1.0
        for index, geom in enumerate(hands):
            if not geom.name:
                geom.name = f"demo_contact_{geom.parent.name}_{index}"
            friction = np.asarray(geom.friction).copy()
            friction[0] = DEX3_SLIDING_FRICTION * hand_scale
            geom.friction = friction
        surfaces = [(g, "hand") for g in hands]
        surfaces += [(g, "tabletop") for g in physical if g.name == "demo_table_0"]
        surfaces += [(g, "tray") for g in physical if g.name.startswith("demo_tray_")]
        for kind, profile in OBJECT_CONTACT_MATERIALS.items():
            objects = [g for g in physical if g.name.startswith(f"demo_{kind}_geom_")]
            for obj in objects:
                for surface, role in surfaces:
                    # All these current geoms have equal priority. Preserve
                    # their solmix-weighted normal solver and contact margins.
                    if obj.priority != surface.priority:
                        raise ValueError("Material contact pairs require equal-priority source geoms.")
                    total = obj.solmix + surface.solmix
                    weight = obj.solmix / total if total > 0 else 0.5
                    a, b = np.asarray(obj.solref), np.asarray(surface.solref)
                    solref = weight * a + (1 - weight) * b if np.all(a > 0) and np.all(b > 0) else np.minimum(a, b)
                    solimp = weight * np.asarray(obj.solimp) + (1 - weight) * np.asarray(surface.solimp)
                    mu = profile[role] * (hand_scale if role == "hand" else 1.0)
                    spec.add_pair(name=f"demo_material_{obj.name}__{surface.name}",
                                  geomname1=obj.name, geomname2=surface.name,
                                  condim=max(obj.condim, surface.condim),
                                  friction=[mu, mu, mu * CONTACT_TORSIONAL_ARM_M[role], CONTACT_ROLLING_M, CONTACT_ROLLING_M],
                                  solref=solref, solimp=solimp,
                                  margin=max(obj.margin, surface.margin), gap=max(obj.gap, surface.gap))

    @staticmethod
    def _add_studio_materials(spec):
        """Deterministic surface textures and studio lights; no bodies, collision shapes or mass edits."""
        # Texture pixels stay in the model, so startup works offline and exported pages need no assets.
        size = 256
        rng = np.random.default_rng(14)
        # Anisotropic filtered noise gives long, irregular fibres instead of repeating sine bands.
        spectrum = np.fft.rfft2(rng.normal(size=(size, size)))
        fx, fy = np.fft.rfftfreq(size)[None, :], np.fft.fftfreq(size)[:, None]
        fibres = np.fft.irfft2(spectrum * np.exp(-((fx / 0.11) ** 2 + (fy / 0.012) ** 2)), s=(size, size))
        grain = 0.011 * fibres / max(float(fibres.std()), 1e-9)
        wood = np.clip(0.955 + grain + rng.normal(0, 0.002, (size, size)), 0, 1)
        stone = np.clip(0.975 + rng.normal(0, 0.006, (size, size)), 0, 1)
        # Quiet metre-scale tile seams anchor the feet without a dense, flickering overlay grid.
        stone[:1, :] *= 0.92
        stone[:, :1] *= 0.92
        for name, pixels in (("demo_oak_texture", wood), ("demo_floor_texture", stone)):
            rgb = np.repeat(np.rint(pixels[..., None] * 255).astype(np.uint8), 3, axis=2)
            spec.add_texture(name=name, type=mujoco.mjtTexture.mjTEXTURE_2D, width=size, height=size,
                             nchannel=3, data=rgb.reshape(-1))
        spec.add_texture(name="demo_studio_sky", type=mujoco.mjtTexture.mjTEXTURE_SKYBOX,
                         builtin=mujoco.mjtBuiltin.mjBUILTIN_GRADIENT, rgb1=[0.43, 0.49, 0.56],
                         rgb2=[0.76, 0.75, 0.71], width=128, height=768)

        for name, specular, shininess in (("demo_satin", 0.22, 0.35), ("demo_metal", 0.62, 0.65),
                                         ("demo_glazed", 0.42, 0.55), ("demo_robot_shell", 0.5, 0.55),
                                         ("demo_robot_dark", 0.18, 0.3), ("demo_apple_skin", 0.32, 0.38)):
            spec.add_material(name=name, rgba=[1, 1, 1, 1], specular=specular, shininess=shininess)
        # Bind the RGB role explicitly; the leading slot is a user texture, not the surface colour.
        oak = spec.add_material(name="demo_oak", rgba=[1, 1, 1, 1], specular=0.18, shininess=0.28,
                               texuniform=True, texrepeat=[1.5, 1.5])
        floor = spec.add_material(name="demo_floor", rgba=[1, 1, 1, 1], specular=0.08, shininess=0.15,
                                 texuniform=True, texrepeat=[1, 1])
        texture_file = _CRACKER_BOX_DIR / "cracker_box_texture.png"
        if not texture_file.exists():
            raise FileNotFoundError(f"{texture_file} missing; run scripts/fetch_ycb_cracker_box.py")
        spec.add_texture(name="demo_cracker_box_texture", type=mujoco.mjtTexture.mjTEXTURE_2D, file=str(texture_file))
        carton = spec.add_material(name="demo_cracker_box_print", rgba=[1, 1, 1, 1], specular=0.12, shininess=0.22)
        for material, texture in ((oak, "demo_oak_texture"), (floor, "demo_floor_texture"), (carton, "demo_cracker_box_texture")):
            slots = list(material.textures)
            slots[int(mujoco.mjtTextureRole.mjTEXROLE_RGB)] = texture
            material.textures = slots
        for geom in spec.geoms:
            if geom.name == "floor":
                geom.material = "demo_floor"
                geom.rgba = DEMO_RENDER_STYLE["floor_rgba"]
            elif not (geom.contype or geom.conaffinity):
                geom.material = "demo_robot_dark" if np.mean(geom.rgba[:3]) < 0.4 else "demo_robot_shell"
                if geom.material == "demo_robot_shell":
                    geom.rgba = [0.76, 0.78, 0.8, float(geom.rgba[3])]

        # One shadow-casting key gives a readable contact shadow; the broad fill and rim do not
        # produce extra shadows. Fixed world directions keep both camera views visually consistent.
        for name, direction, diffuse, castshadow in (
            ("key", [-0.2, 0.5, -0.84], [0.24, 0.23, 0.21], True),
            ("fill", [0.6, 0.3, -0.75], [0.48, 0.49, 0.50], False),
            ("rim", [-0.8, -0.3, -0.55], [0.30, 0.32, 0.35], False),
        ):
            light = spec.worldbody.add_light(name="demo_studio_" + name, pos=[0.3, 0, 3], dir=direction,
                diffuse=diffuse, specular=[v * 0.6 for v in diffuse], ambient=[0, 0, 0], castshadow=castshadow)
            if hasattr(light, "type"):
                light.type = mujoco.mjtLightType.mjLIGHT_DIRECTIONAL
            else:
                light.directional = True

    def _add_table(self, spec):
        table = TABLES[self.table_kind]
        x, y = table["center"]
        top, thick = self.table_height, 0.035
        counter = 0

        def geom(shape, size, pos, color):
            nonlocal counter
            g = spec.worldbody.add_geom(name=f"demo_table_{counter}", type=shape, size=size, pos=pos,
                rgba=color, material="demo_oak" if counter == 0 else "demo_metal",
                contype=1, conaffinity=1, friction=[0.8, 0.01, 0.001])
            counter += 1
            return g

        wood = [0.67, 0.57, 0.43, 1.0]
        metal = [0.17, 0.20, 0.21, 1.0]
        if table["shape"] == "circle":
            geom(mujoco.mjtGeom.mjGEOM_CYLINDER, [table["radius"], thick / 2, 0], [x, y, top - thick / 2], wood)
            geom(mujoco.mjtGeom.mjGEOM_CYLINDER, [0.032, (top - thick) / 2, 0], [x + 0.05, y, (top - thick) / 2], metal)
            geom(mujoco.mjtGeom.mjGEOM_CYLINDER, [0.18, 0.015, 0], [x + 0.05, y, 0.015], metal)
        else:
            hx, hy = table["half_size"]
            geom(mujoco.mjtGeom.mjGEOM_BOX, [hx, hy, thick / 2], [x, y, top - thick / 2], wood)
            if self.table_kind == "pedestal":
                geom(mujoco.mjtGeom.mjGEOM_BOX, [0.07, 0.07, (top - thick) / 2], [x + 0.05, y, (top - thick) / 2], metal)
                geom(mujoco.mjtGeom.mjGEOM_BOX, [0.15, 0.16, 0.015], [x + 0.03, y, 0.015], metal)
            else:
                for dx in (-hx + 0.045, hx - 0.045):
                    for dy in (-hy + 0.045, hy - 0.045):
                        geom(mujoco.mjtGeom.mjGEOM_BOX, [0.025, 0.025, (top - thick) / 2], [x + dx, y + dy, (top - thick) / 2], metal)

    def _sole_bottom(self) -> float:
        """Lowest world z of the feet's COLLISION geometry in the current pose -- whatever shape it is.

        The plant's default foot collision is the sole box (physics_profiles.replace_urdf_foot_collisions; the foot mesh stays
        as a contype/conaffinity 0 visual), ``legacy_mesh`` keeps the URDF mesh: both, plus capsules / cylinders, go through
        geometry._geom_points_local (mesh vertices, box corners, sampled curved surfaces in the body frame).
        """
        from sim2sim.geometry import _geom_points_local

        m, d = self.model, self.data
        heights = []
        for gi in range(m.ngeom):
            bid = int(m.geom_bodyid[gi])
            name = mujoco.mj_id2name(m, mujoco.mjtObj.mjOBJ_BODY, bid) or ""
            if "ankle_roll_link" not in name or not (m.geom_contype[gi] or m.geom_conaffinity[gi]):
                continue
            pts = _geom_points_local(m, gi)  # geom points in the BODY frame (geom_pos / geom_quat applied)
            if pts.shape[0]:
                heights.append(float(np.min(pts @ d.xmat[bid].reshape(3, 3)[2] + d.xpos[bid, 2])))
        if not heights:
            raise RuntimeError("cannot determine the robot's sole height (no collision geometry on the ankle-roll bodies)")
        return float(min(heights))

    def _add_tray(self, spec) -> None:
        """A fixed open tray with a real floor and four walls; it is not one of the selectable free objects."""
        x, y = TRAY_CENTER
        hx, hy = np.asarray(TRAY_OUTER_SIZE) / 2
        wall, bottom = TRAY_WALL_THICKNESS, TRAY_BOTTOM_THICKNESS
        z = self.table_height

        def box(name, size, pos, color):
            spec.worldbody.add_geom(name="demo_tray_" + name, type=mujoco.mjtGeom.mjGEOM_BOX,
                size=list(size), pos=list(pos), rgba=color, material="demo_satin", contype=1, conaffinity=1,
                friction=[0.9, 0.01, 0.001], solref=[0.008, 1.0])

        box("bottom", [hx, hy, bottom / 2], [x, y, z + bottom / 2], [0.24, 0.31, 0.30, 1.0])
        wall_z = z + bottom + TRAY_WALL_HEIGHT / 2
        for sign, name in ((-1, "near"), (1, "far")):
            box(name, [wall / 2, hy, TRAY_WALL_HEIGHT / 2], [x + sign * (hx - wall / 2), y, wall_z], self.tray["color"])
        for sign, name in ((-1, "right"), (1, "left")):
            box(name, [hx - wall, wall / 2, TRAY_WALL_HEIGHT / 2], [x, y + sign * (hy - wall / 2), wall_z], self.tray["color"])

    @staticmethod
    def _convex_polygons_overlap(a: np.ndarray, b: np.ndarray, clearance: float = 0.003) -> bool:
        """Separating-axis test on the convex placement footprints; leave a small editor clearance at the tray."""
        ordered = []
        for polygon in (a, b):
            centered = polygon - polygon.mean(axis=0)
            ordered.append(polygon[np.argsort(np.arctan2(centered[:, 1], centered[:, 0]))])
        for polygon in ordered:
            for edge in np.roll(polygon, -1, axis=0) - polygon:
                axis = np.array([-edge[1], edge[0]])
                length = np.linalg.norm(axis)
                if length < 1e-12:
                    continue
                axis /= length
                ap, bp = a @ axis, b @ axis
                if ap.max() + clearance < bp.min() or bp.max() + clearance < ap.min():
                    return False
        return True

    def _footprint_points(self, kind: str, x: float, y: float, yaw: float) -> np.ndarray:
        if kind in ("apple", "can", "bottle"):
            radius = OBJECTS[kind]["footprint_radius"]
            a = np.linspace(0, 2 * math.pi, 32, endpoint=False)
            points = np.c_[radius * np.cos(a), radius * np.sin(a)]
        elif kind == "mug":
            # The handle points along local +x, not symmetrically about the body origin.
            points = np.array([[-0.038, -0.038], [-0.038, 0.038], [0.072, -0.038], [0.072, 0.038]])
        else:
            hx, hy = np.asarray(OBJECTS[kind]["size"][:2]) / 2
            points = np.array([[-hx, -hy], [-hx, hy], [hx, -hy], [hx, hy]])
        c, s = math.cos(yaw), math.sin(yaw)
        return points @ np.array([[c, s], [-s, c]]) + [x, y]

    def validate_placement(self, kind: str, x: float, y: float, yaw: float = 0.0) -> None:
        if kind not in OBJECTS:
            raise ValueError(f"unknown object: {kind!r}")
        if not np.isfinite([x, y, yaw]).all():
            raise ValueError("object pose must contain finite numbers")
        r = PLACEMENT_REGION
        if not contains_placement(r, x, y):
            raise ValueError("place the object inside the highlighted workspace")
        tb = TABLES[self.table_kind]
        world_points = self._footprint_points(kind, x, y, yaw)
        points = world_points - tb["center"]
        if tb["shape"] == "circle":
            supported = np.linalg.norm(points, axis=1).max() <= tb["radius"] - 0.006
        else:
            supported = np.all(np.abs(points) <= np.asarray(tb["half_size"]) - 0.006)
        if not supported:
            raise ValueError("the entire object, including its handle, must fit on the tabletop")
        bounds = self.tray["outer_bounds"]
        tray_points = np.array([[bounds["x_min"], bounds["y_min"]], [bounds["x_max"], bounds["y_min"]],
                                [bounds["x_max"], bounds["y_max"]], [bounds["x_min"], bounds["y_max"]]])
        if self._convex_polygons_overlap(world_points, tray_points):
            raise ValueError("keep the object outside the drop tray; the robot places it into the tray after grasping")

    def place_object(self, kind: str, x: float, y: float, yaw: float = 0.0) -> dict[str, Any]:
        """Select and place one object; reject invalid placements without changing the previous simulation state."""
        x, y, yaw = float(x), float(y), float(yaw)
        self.validate_placement(kind, x, y, yaw)
        if self._placement_intersects_robot(kind, x, y, yaw):
            raise ValueError("the object intersects the robot; move it farther forward or sideways")
        # Only commit after footprint and robot checks have passed. The trial ran on scratch MjData, so even a
        # rejected update to the currently held object preserves its simulated pose, velocity and solver state.
        for other in self.objects:
            if other != kind:
                self._place(other, None)
        self._place(kind, (x, y, yaw))
        self._placements = {kind: (x, y, yaw)}
        self.plant.forward()
        return self._object_state(kind)

    def _placement_intersects_robot(self, kind: str, x: float, y: float, yaw: float) -> bool:
        m, trial, obj = self.model, self._placement_data, self.objects[kind]
        mujoco.mj_copyData(trial, m, self.data)
        trial.qpos[obj.qadr:obj.qadr + 7] = [x, y, obj.rest_z, math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        trial.qvel[obj.vadr:obj.vadr + 6] = 0
        old_contype, old_conaffinity = m.geom_contype[obj.geom_ids].copy(), m.geom_conaffinity[obj.geom_ids].copy()
        try:
            m.geom_contype[obj.geom_ids] = self._object_contype[obj.geom_ids]
            m.geom_conaffinity[obj.geom_ids] = self._object_conaffinity[obj.geom_ids]
            mujoco.mj_forward(m, trial)
        finally:
            m.geom_contype[obj.geom_ids], m.geom_conaffinity[obj.geom_ids] = old_contype, old_conaffinity
        ids = set(obj.geom_ids)
        for c in trial.contact:
            if c.dist >= -0.0005:
                continue
            a, b = int(c.geom1), int(c.geom2)
            other = b if a in ids else a if b in ids else None
            if other is None:
                continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[other])) or ""
            if name != "world" and not name.startswith("demo_"):
                return True
        return False

    def _place(self, kind, pose):
        obj = self.objects[kind]
        m, d = self.model, self.data
        obj.active = pose is not None
        m.geom_contype[obj.geom_ids] = self._object_contype[obj.geom_ids] if obj.active else 0
        m.geom_conaffinity[obj.geom_ids] = self._object_conaffinity[obj.geom_ids] if obj.active else 0
        m.geom_rgba[obj.geom_ids] = self._object_rgba[obj.geom_ids]
        if not obj.active:
            m.geom_rgba[obj.geom_ids, 3] = 0
        m.body_gravcomp[obj.body_id] = 0 if obj.active else 1
        x, y, yaw = (0.0, 0.0, 0.0) if pose is None else pose
        d.qpos[obj.qadr:obj.qadr + 3] = [x, y, obj.rest_z if obj.active else PARK_Z]
        d.qpos[obj.qadr + 3:obj.qadr + 7] = [math.cos(yaw / 2), 0, 0, math.sin(yaw / 2)]
        d.qvel[obj.vadr:obj.vadr + 6] = 0

    def remove_object(self, kind: str) -> None:
        if kind not in OBJECTS:
            raise ValueError(f"unknown object: {kind!r}")
        self._placements.pop(kind, None)
        self._place(kind, None)
        self.plant.forward()

    def clear_objects(self) -> None:
        self._placements.clear()
        for kind in OBJECTS:
            self._place(kind, None)
        self.plant.forward()

    def placement_poses(self) -> dict[str, tuple[float, float, float]]:
        """Copy of the user's reset poses, independent of the objects' current simulated positions."""
        return dict(self._placements)

    def reset(self) -> dict[str, Any]:
        self.plant.reset(self.initial_root_pos, np.array([0, 0, 0, 1.0]), self.initial_dof_pos)
        self.hand_closure = {"left": 0.0, "right": 0.0}
        self._finger_goal = np.concatenate([HAND_OPEN[s] for s in ("left", "right")])
        self._finger_target = self._finger_goal.copy()
        self.data.qpos[self.finger_qadr] = self._finger_goal
        self.last_finger_torque[:] = 0
        for kind in OBJECTS:
            self._place(kind, self._placements.get(kind))
        self.plant.forward()
        if self.policy is not None and hasattr(self.policy, "reset"):
            self.policy.reset()
        return self.state()

    def set_hand_closure(self, hand: str, fraction: float) -> None:
        if hand not in HAND_OPEN:
            raise ValueError("hand must be 'left' or 'right'")
        fraction = float(fraction)
        if not np.isfinite(fraction) or not 0 <= fraction <= 1:
            raise ValueError("hand closure must be between 0 and 1")
        self.hand_closure[hand] = fraction
        ids = slice(0, FINGERS_PER_HAND) if hand == "left" else slice(FINGERS_PER_HAND, 2 * FINGERS_PER_HAND)
        target = HAND_OPEN[hand] * (1 - fraction) + HAND_CLOSED[hand] * fraction
        self._finger_goal[ids] = np.clip(target, self.finger_limits[ids, 0], self.finger_limits[ids, 1])

    def step(self, body_qtarget: Sequence[float]) -> np.ndarray:
        """Advance 20 ms using body and torque-limited finger PD at 1 kHz. No object state is prescribed here."""
        p, m, d = self.plant, self.model, self.data
        target = np.asarray(body_qtarget, dtype=float)
        if target.shape != (29,) or not np.isfinite(target).all():
            raise ValueError("body_qtarget must be 29 finite joint angles")
        target = p._delayed_target(target)
        p.last_q_target_applied = target.copy()
        dq = np.array([1.0, 0.0, 0.0, 0.0])
        previous_gyro = d.qvel[3:6].copy()
        for _ in range(p.substeps):
            self._finger_target += np.clip(self._finger_goal - self._finger_target, -2.0 * p.physics_dt, 2.0 * p.physics_dt)
            ft = self.finger_kp * (self._finger_target - d.qpos[self.finger_qadr]) - self.finger_kd * d.qvel[self.finger_vadr]
            self.last_finger_torque = np.clip(ft, -self.finger_effort_limit, self.finger_effort_limit)
            tau = p.pd_torque(target)
            d.qfrc_applied[:] = 0
            d.qfrc_applied[p.dof_vadr] = tau
            d.qfrc_applied[self.finger_vadr] = self.last_finger_torque
            mujoco.mj_step(m, d)
            gyro = d.qvel[3:6].copy()
            mujoco.mju_quatIntegrate(dq, (previous_gyro + gyro) / 2, p.physics_dt)
            previous_gyro = gyro
        p.imu_delta_quat_b = wxyz_to_xyzw(dq)
        p.imu_step_count += 1
        p.last_torque = np.asarray(tau).copy()
        return p.last_torque

    def object_pose(self, kind: str) -> tuple[np.ndarray, np.ndarray]:
        obj = self.objects[kind]
        return self.data.xpos[obj.body_id].copy(), wxyz_to_xyzw(self.data.xquat[obj.body_id]).copy()

    pose = object_pose

    def object_contacts(self, kind: str) -> dict[str, int]:
        ids = set(self.objects[kind].geom_ids)
        support = set(self.table_geom_ids) | {self.floor_geom_id}
        tray = set(self.tray_geom_ids)
        result = dict(left=0, right=0, robot_other=0, support=0, objects=0, tray=0, tray_bottom=0)
        for contact in self.data.contact:
            a, b = int(contact.geom1), int(contact.geom2)
            if a in ids: other = b
            elif b in ids: other = a
            else: continue
            name = mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_BODY, int(self.model.geom_bodyid[other])) or ""
            if other in tray:
                result["tray"] += 1
                result["support"] += 1
                if other == self.tray_bottom_geom_id:
                    result["tray_bottom"] += 1
            elif other in support: result["support"] += 1
            elif name.startswith("left_") and any(s in name for s in ("hand", "wrist", "elbow", "shoulder")): result["left"] += 1
            elif name.startswith("right_") and any(s in name for s in ("hand", "wrist", "elbow", "shoulder")): result["right"] += 1
            elif name.startswith("demo_"): result["objects"] += 1
            else: result["robot_other"] += 1
        return result

    def _object_state(self, kind: str) -> dict[str, Any]:
        obj = self.objects[kind]
        pos, q = self.object_pose(kind)
        x, y, z, w = q
        yaw = math.atan2(2 * (w * z + x * y), 1 - 2 * (y * y + z * z))
        return dict(copy.deepcopy(OBJECTS[kind]), active=obj.active, position=pos.tolist(), quaternion=q.tolist(),
                    yaw=yaw, rest_z=obj.rest_z, contacts=self.object_contacts(kind))

    def state(self) -> dict[str, Any]:
        table = copy.deepcopy(TABLES[self.table_kind])
        table.update(kind=self.table_kind, height=self.table_height)
        return dict(table=table, tray=copy.deepcopy(self.tray), objects=[self._object_state(k) for k, obj in self.objects.items() if obj.active],
                    robot=dict(position=self.plant.root_pos.tolist(), quaternion=self.plant.root_quat.tolist(),
                               joint_positions=self.plant.dof_pos.tolist(), setback_m=self.robot_setback), hand_closure=dict(self.hand_closure),
                    time=float(self.data.time), placement_region=copy.deepcopy(PLACEMENT_REGION), hand_workspaces=copy.deepcopy(HAND_WORKSPACES),
                    ego_camera=copy.deepcopy(self.ego_camera_info))
