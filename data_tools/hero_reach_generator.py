"""Generate whole-body G1 reaching motions with Mink inverse kinematics.

A specification supplies palm targets, base and foot support poses, timing, and
optional table obstacles. A quadratic-programming solver combines palm, foot,
pelvis, centre-of-mass, posture, velocity, and collision constraints. Solved
motions carry quality labels and reports; inspect them before adding the clips
to a training corpus. Quaternions in exported arrays use wxyz.

The example entry point is scripts/generate_ik_example.py. This module
also provides the model and solver primitives for custom generation workflows."""

from __future__ import annotations

from data_tools import _threads  # noqa: F401  -- single-threaded BLAS/OpenMP before numpy and MuJoCo load

import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from data_tools import reach_specs as rs
from data_tools.reach_specs import (
    BaseStrategy,
    ClipSpec,
    HandTarget,
    SegmentSpec,
    mat_to_quat_wxyz,
    quat_angle_wxyz,
    quat_slerp_wxyz,
    quat_wxyz_to_mat,
    rot_y,
    rot_z,
    smoothstep,
)
from hero_isaacsim.constants import DOF_NAMES, EE_BODY_NAMES, MJCF_SCENE_FILE_NAME, PALM_OFFSET

FPS: int = 50
DT: float = 1.0 / FPS
SCHEMA: str = "hero_reach_bank_v2"
GENERATOR_VERSION: str = "2.1.0"
from hero_isaacsim.paths import G1_ASSET_ROOT, RELEASE_ROOT as REPO_ROOT

DEFAULT_MJCF: Path = G1_ASSET_ROOT / MJCF_SCENE_FILE_NAME

SIDES: tuple[str, str] = ("left", "right")
TABLE_THICKNESS: float = 0.04
TABLE_WIDTH: float = 1.40
KNEE_FLOOR_Z: float = 0.06           # knee link origin height with the knee mesh resting on the floor
KNEEL_SHANK_X: float = 0.29          # horizontal knee -> toe distance of the kneeling leg
SETTLE_FRAMES: int = 15              # static frames at the start of every clip (0.3 s)
# Short-reach duration combines a fixed budget, translation and angular travel.
# Profiles may extend these timing limits for additional reaching examples.
REPLAN_T_BASE_S: float = 0.4
REPLAN_T_PER_M: float = 1.2
REPLAN_T_PER_PI: float = 0.5
REPLAN_REACH_MIN_S: float = 0.5
REPLAN_REACH_MAX_S: float = 3.0
RETRACT_HOLD_FRAMES: int = 25


def retract_rest_palm(side: str, pelvis_h: float = rs.STANDING_ROOT_Z) -> np.ndarray:
    """Nominal rest palm point (heading frame): 0.10 m ahead of the pelvis, 0.21 m to the hand's side, 2 cm below the standing pelvis
    height -- roughly the palm of the default arm posture.  Fallback for :func:`plan_segment` ``retract=True`` when no ``palm_rest`` is
    given; :func:`generate_clip` passes the clip's FRAME-0 palm poses (settle pose = default posture) as the exact rest pose."""
    return np.array([0.10, (0.21 if side == "left" else -0.21), pelvis_h - 0.02])


def _retract_rest(side: str, palm_start: dict[str, tuple[np.ndarray, np.ndarray]], palm_rest: dict[str, tuple[np.ndarray, np.ndarray]] | None,
                  pelvis_h: float) -> tuple[np.ndarray, np.ndarray]:
    """(rest position, rest quaternion wxyz) the retract phase returns ``side`` to."""
    if palm_rest is not None and side in palm_rest:
        pr, qr = palm_rest[side]
        return np.asarray(pr, dtype=np.float64), np.asarray(qr, dtype=np.float64)
    return retract_rest_palm(side, pelvis_h), np.asarray(palm_start[side][1], dtype=np.float64)


def _retime_factor(prev_tail: np.ndarray, qs: np.ndarray, resid: np.ndarray, plan: "SegmentPlan", bundle: "ModelBundle", cfg: "SolverConfig",
                   attempt: int) -> float | None:
    """Re-timing rule shared by the reach segments and the retract phase: None = accept this solve, else the factor to stretch the
    plan by (joint speed / acceleration over the solver caps; a hand held back by a collision limit or an over-long plan is accepted
    as is because slowing down cannot help)."""
    traj = np.concatenate([prev_tail, qs], axis=0)[:, bundle.dof_qpos_idx]
    qd = np.gradient(traj, DT, axis=0)
    v = float(np.abs(qd).max())
    a = float(np.abs(np.gradient(qd, DT, axis=0)).max()) if len(traj) > 2 else 0.0
    blocked = bool(resid[: plan.reach_frames, 0].max() > 0.03)   # hand held back by a collision limit: re-timing cannot help
    too_long = plan.frames >= int(cfg.max_segment_s * FPS)
    if (v <= cfg.max_qdot and a <= cfg.max_qddot) or attempt == cfg.max_retime or blocked or too_long:
        return None
    return min(cfg.retime_factor_cap, max(v / (0.8 * cfg.max_qdot), math.sqrt(a / (0.8 * cfg.max_qddot)), 1.15))


# --------------------------------------------------------------------------------------------------
# Configuration dataclasses
# --------------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class SolverConfig:
    palm_pos_cost: float = 20.0
    palm_ori_cost: float = 8.0
    foot_pos_cost: float = 200.0
    foot_ori_cost: float = 15.0
    kneel_task_cost: float = 30.0
    pelvis_pos_cost: tuple[float, float, float] = (0.5, 0.5, 20.0)
    pelvis_ori_cost: tuple[float, float, float] = (6.0, 6.0, 6.0)
    posture_leg_cost: float = 0.5
    posture_waist_cost: float = 0.8
    posture_arm_active_cost: float = 0.3
    posture_arm_inactive_cost: float = 3.0
    prev_cost: float = 0.6                    # arms
    prev_cost_lower: float = 2.0              # legs + waist: damps pelvis/waist pitch redundancy flips in bows
    frame_vel_cap: float = 4.0
    max_iters: int = 12
    first_frame_iters: int = 200
    vel_limit_rad_per_frame: float = 0.08
    limit_utilization: float = 0.98
    limit_min_margin: float = 0.012
    collision_min_dist: float = 0.02          # proxies are already ~1 cm larger than the meshes
    collision_min_dist_upperarm: float = 0.01
    collision_detect_dist: float = 0.12
    collision_gain: float = 0.5
    com_xy_cost: float = 20.0
    retime_factor_cap: float = 1.5
    max_segment_s: float = 12.0
    solver: str = "daqp"
    damping: float = 1e-6
    lm_damping: float = 1e-3
    exit_palm_pos: float = 2e-3
    exit_palm_ori: float = 0.01
    exit_foot: float = 1e-3
    exit_pelvis: float = 5e-3
    max_retime: int = 2
    max_qdot: float = 6.0
    max_qddot: float = 60.0

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


@dataclasses.dataclass(frozen=True)
class GateThresholds:
    strict_ee_pos_m: float = 0.015
    strict_ee_ori_rad: float = math.radians(5.0)
    relaxed_ee_pos_m: float = 0.06
    relaxed_ee_ori_rad: float = math.radians(15.0)
    strict_foot_drift_m: float = 0.001
    relaxed_foot_drift_m: float = 0.005
    strict_limit_margin_rad: float = 0.01
    relaxed_limit_margin_rad: float = 0.002
    strict_qdot: float = 6.0
    relaxed_qdot: float = 8.0
    strict_qddot: float = 60.0
    relaxed_qddot: float = 100.0
    strict_com_margin_m: float = 0.02
    relaxed_com_margin_m: float = -0.02
    ankle_limit_hit_margin_rad: float = 0.02

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


# --------------------------------------------------------------------------------------------------
# Model bundle
# --------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class TableBox:
    center: tuple[float, float, float]   # world, box centre (z = surface - thickness/2)
    half_size: tuple[float, float, float]
    surface_z: float

    def as_dict(self) -> dict[str, Any]:
        return {"center": list(self.center), "half_size": list(self.half_size), "surface_z": self.surface_z}

    @staticmethod
    def from_dict(d: dict[str, Any]) -> "TableBox":
        return TableBox(tuple(d["center"]), tuple(d["half_size"]), float(d["surface_z"]))


def table_for_segment(seg: SegmentSpec) -> TableBox | None:
    """Slab under a tabletop target: front edge ``table_edge_gap`` before the target, ``table_depth`` deep."""
    if not seg.has_table:
        return None
    pos = np.mean([t.pos for t in seg.targets], axis=0)
    if seg.layer == "behind":
        cx = float(pos[0]) + seg.table_edge_gap - seg.table_depth / 2.0
    else:
        cx = float(pos[0]) - seg.table_edge_gap + seg.table_depth / 2.0
    return TableBox(center=(cx, float(pos[1]) * 0.5, seg.surface_z - TABLE_THICKNESS / 2.0),
                    half_size=(seg.table_depth / 2.0, TABLE_WIDTH / 2.0, TABLE_THICKNESS / 2.0), surface_z=seg.surface_z)


@dataclasses.dataclass
class ModelBundle:
    model: Any
    data: Any
    mjcf_path: Path
    nq: int
    nv: int
    dof_qpos_idx: np.ndarray            # (29,) qpos index of every DOF (holosoma order)
    dof_vel_idx: np.ndarray             # (29,) dof index
    jnt_range: np.ndarray               # (29, 2)
    body_id: dict[str, int]
    site_id: dict[str, int]             # left_palm_site ... right_sole_site
    table_geoms: list[int | None]       # per segment
    foot_geoms: dict[str, set[int]]     # side -> sphere geoms
    leg_geoms: dict[str, set[int]]      # side -> knee/ankle/foot geoms (floor contact allowed while kneeling)
    robot_geoms: set[int]
    floor_geom: int
    collision_pairs: list[tuple[list[int], list[int]]]   # proxy self + floor pairs (table pairs per segment)
    collision_pairs_upperarm: list[tuple[list[int], list[int]]]  # close quarters (hands vs hips / own upper arm / shoulders)
    collision_pairs_upperarm_trunk: list[tuple[list[int], list[int]]]  # upper arm vs trunk (touching at rest: no-approach only)
    arm_geoms_all: list[int]                              # arm proxies
    table_partner_geoms: list[int]                        # trunk + leg proxies (kept off the table)
    proxy_geoms: set[int]
    baseline_self_pairs: set[tuple[int, int]]
    excluded_self_pairs: set[tuple[int, int]]   # geoms on links within kinematic distance <= 2 (mesh artefacts)
    r_flat: dict[str, np.ndarray]       # ankle_roll orientation at q=0 (identity for this model)


def _hash_file(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


PROXY_GROUP: int = 5
_PROXY_CACHE: dict[str, dict[str, Any]] = {}


def _measure_proxies(mjcf_path: Path) -> dict[str, Any]:
    """Zero-configuration link offsets and trunk mesh AABBs needed to size the collision proxies (cached)."""
    import mujoco

    key = str(mjcf_path)
    if key in _PROXY_CACHE:
        return _PROXY_CACHE[key]
    m = mujoco.MjModel.from_xml_path(key)
    d = mujoco.MjData(m)
    d.qpos[3] = 1.0
    mujoco.mj_kinematics(m, d)

    def pos(name: str) -> np.ndarray:
        return d.xpos[m.body(name).id].copy()

    def local_offset(parent: str, child: str) -> np.ndarray:
        """child origin expressed in the parent link frame (link frames may be rotated at q=0)."""
        R = d.xmat[m.body(parent).id].reshape(3, 3)
        return R.T @ (pos(child) - pos(parent))

    for name in ("torso_link", "pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"):
        if not np.allclose(d.xmat[m.body(name).id].reshape(3, 3), np.eye(3), atol=1e-3):
            raise RuntimeError(f"{name} frame is rotated at q=0; hand / trunk proxies assume aligned frames")

    def aabb(name: str) -> tuple[np.ndarray, np.ndarray]:
        """Link-frame AABB (centre, half sizes) of a link's collision geoms (meshes via vertices, primitives via geom_aabb)."""
        b = m.body(name).id
        pts = []
        for gidx in range(m.ngeom):
            if m.geom_bodyid[gidx] != b or m.geom_contype[gidx] == 0:
                continue
            R = d.geom_xmat[gidx].reshape(3, 3)
            if m.geom_type[gidx] == mujoco.mjtGeom.mjGEOM_MESH:
                mid = m.geom_dataid[gidx]
                loc = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
            else:
                c, h = m.geom_aabb[gidx][:3], m.geom_aabb[gidx][3:]
                loc = np.array([c + h * np.array([sx, sy, sz]) for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)])
            pts.append(loc @ R.T + d.geom_xpos[gidx] - d.xpos[b])
        if not pts:
            raise RuntimeError(f"{name} has no collision geoms")
        w = np.concatenate(pts, axis=0) @ d.xmat[b].reshape(3, 3)   # into the link frame (R^T applied on the right)
        lo, hi = w.min(axis=0), w.max(axis=0)
        return 0.5 * (lo + hi), 0.5 * (hi - lo)

    def enclosing_radius(bodies: Sequence[str], a: np.ndarray, b: np.ndarray, pad: float = 0.005) -> float:
        """Smallest capsule radius about world segment a->b that contains every collision-mesh vertex of ``bodies``."""
        ab = b - a
        L2 = max(1e-12, float(ab @ ab))
        worst = 0.0
        for bn in bodies:
            bid = m.body(bn).id
            for gidx in range(m.ngeom):
                if m.geom_bodyid[gidx] != bid or m.geom_contype[gidx] == 0 or m.geom_type[gidx] != mujoco.mjtGeom.mjGEOM_MESH:
                    continue
                mid = m.geom_dataid[gidx]
                v = m.mesh_vert[m.mesh_vertadr[mid]: m.mesh_vertadr[mid] + m.mesh_vertnum[mid]]
                w = v @ d.geom_xmat[gidx].reshape(3, 3).T + d.geom_xpos[gidx]
                t = np.clip(((w - a) @ ab) / L2, 0.0, 1.0)
                worst = max(worst, float(np.linalg.norm(w - (a + t[:, None] * ab), axis=1).max()))
        return worst + pad

    out: dict[str, Any] = {"torso_aabb": aabb("torso_link"), "pelvis_aabb": aabb("pelvis")}
    for s in SIDES:
        out[f"{s}_elbow_to_wrist"] = local_offset(f"{s}_elbow_link", f"{s}_wrist_yaw_link")
        out[f"{s}_shoulder_to_elbow"] = local_offset(f"{s}_shoulder_yaw_link", f"{s}_elbow_link")
        out[f"{s}_hip_to_knee"] = local_offset(f"{s}_hip_roll_link", f"{s}_knee_link")
        out[f"{s}_knee_to_ankle"] = local_offset(f"{s}_knee_link", f"{s}_ankle_pitch_link")
        out[f"{s}_forearm_r"] = enclosing_radius([f"{s}_elbow_link", f"{s}_wrist_roll_link", f"{s}_wrist_pitch_link"], pos(f"{s}_elbow_link"), pos(f"{s}_wrist_yaw_link"))
        out[f"{s}_upperarm_r"] = enclosing_radius([f"{s}_shoulder_yaw_link"], pos(f"{s}_shoulder_yaw_link"), pos(f"{s}_elbow_link"))
        out[f"{s}_hipyaw_to_knee"] = local_offset(f"{s}_hip_yaw_link", f"{s}_knee_link")
        out[f"{s}_thigh_r"] = enclosing_radius([f"{s}_hip_yaw_link"], pos(f"{s}_hip_yaw_link"), pos(f"{s}_knee_link"))
        out[f"{s}_hip_pitch_aabb"] = aabb(f"{s}_hip_pitch_link")
        out[f"{s}_hip_roll_aabb"] = aabb(f"{s}_hip_roll_link")
        out[f"{s}_shoulder_pitch_aabb"] = aabb(f"{s}_shoulder_pitch_link")
        out[f"{s}_shoulder_roll_aabb"] = aabb(f"{s}_shoulder_roll_link")
        out[f"{s}_shank_r"] = enclosing_radius([f"{s}_knee_link"], pos(f"{s}_knee_link"), pos(f"{s}_ankle_pitch_link"))
        wy = pos(f"{s}_wrist_yaw_link")
        hand_bodies = [f"{s}_wrist_yaw_link"] + [m.body(i).name for i in range(m.nbody) if m.body(i).name.startswith(f"{s}_hand_") and "thumb" not in m.body(i).name]
        out[f"{s}_hand_r"] = enclosing_radius(hand_bodies, wy + np.array([-0.02, 0.0, 0.0]), wy + np.array([0.19, 0.0, 0.0]))
    _PROXY_CACHE[key] = out
    return out


def _add_collision_proxies(spec: Any, mjcf_path: Path) -> list[str]:
    """Analytic collision proxies (group PROXY_GROUP, contype/conaffinity 0) for the mink QP limit.

    Mesh-mesh ``mj_geomDistance`` between the tiny Dex3 finger meshes and the thigh mesh is unreliable (measured:
    0.046 -> 0.000 -> 0.046 m on consecutive frames), so the QP sees capsules / boxes instead; ``mj_collision`` on
    the real meshes remains the gate.  Sizes from the zero-configuration mesh extents (hand envelope
    ``reach_specs.HAND_BBOX_PALM_FRAME``: fingertips 0.217 m along +x of wrist_yaw, thumb 0.115 m toward the midline).
    """
    import mujoco

    meas = _measure_proxies(mjcf_path)
    names: list[str] = []

    def capsule(body: str, name: str, a: np.ndarray, b: np.ndarray, r: float) -> None:
        spec.body(body).add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_CAPSULE, fromto=[*a, *b], size=[r, 0.0, 0.0],
                                 contype=0, conaffinity=0, group=PROXY_GROUP, rgba=[0.2, 0.8, 0.2, 0.25])
        names.append(name)

    def box(body: str, name: str, center: np.ndarray, half: np.ndarray) -> None:
        spec.body(body).add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, pos=list(center), size=list(half),
                                 contype=0, conaffinity=0, group=PROXY_GROUP, rgba=[0.2, 0.2, 0.8, 0.25])
        names.append(name)

    for s in SIDES:
        sgn = 1.0 if s == "right" else -1.0
        capsule(f"{s}_wrist_yaw_link", f"{s}_hand_proxy", np.array([-0.02, 0.0, 0.0]), np.array([0.19, 0.0, 0.0]), meas[f"{s}_hand_r"])
        capsule(f"{s}_wrist_yaw_link", f"{s}_thumb_proxy", np.array([0.10, sgn * 0.03, 0.0]), np.array([0.105, sgn * 0.105, 0.0]), 0.02)
        capsule(f"{s}_elbow_link", f"{s}_forearm_proxy", np.zeros(3), meas[f"{s}_elbow_to_wrist"], meas[f"{s}_forearm_r"])
        capsule(f"{s}_shoulder_yaw_link", f"{s}_upperarm_proxy", np.zeros(3), meas[f"{s}_shoulder_to_elbow"], meas[f"{s}_upperarm_r"])
        capsule(f"{s}_hip_yaw_link", f"{s}_thigh_proxy", np.zeros(3), meas[f"{s}_hipyaw_to_knee"], meas[f"{s}_thigh_r"])
        c, h = meas[f"{s}_hip_pitch_aabb"]
        box(f"{s}_hip_pitch_link", f"{s}_hip_pitch_proxy", c, h)
        c, h = meas[f"{s}_hip_roll_aabb"]
        box(f"{s}_hip_roll_link", f"{s}_hip_roll_proxy", c, h)
        c, h = meas[f"{s}_shoulder_pitch_aabb"]
        box(f"{s}_shoulder_pitch_link", f"{s}_shoulder_pitch_proxy", c, h)
        c, h = meas[f"{s}_shoulder_roll_aabb"]
        box(f"{s}_shoulder_roll_link", f"{s}_shoulder_roll_proxy", c, h)
        capsule(f"{s}_knee_link", f"{s}_shank_proxy", np.zeros(3), meas[f"{s}_knee_to_ankle"], meas[f"{s}_shank_r"])
    c, h = meas["torso_aabb"]
    box("torso_link", "torso_proxy", c, h - np.array([0.005, 0.01, 0.0]))   # the torso is rounded: trim the AABB a little
    c, h = meas["pelvis_aabb"]
    box("pelvis", "pelvis_proxy", c, h)
    return names


def build_model(tables: Sequence[TableBox | None] = (), mjcf_path: Path | str = DEFAULT_MJCF) -> ModelBundle:
    """Compile the Dex3 scene with palm / toe / heel / sole sites and per-segment table boxes."""
    import mujoco

    mjcf_path = Path(mjcf_path)
    spec = mujoco.MjSpec.from_file(str(mjcf_path))
    for side in SIDES:
        spec.body(f"{side}_wrist_yaw_link").add_site(name=f"{side}_palm_site", pos=list(PALM_OFFSET[side]))
        ankle = spec.body(f"{side}_ankle_roll_link")
        ankle.add_site(name=f"{side}_toe_site", pos=list(rs.TOE_LOCAL))
        ankle.add_site(name=f"{side}_heel_site", pos=list(rs.HEEL_LOCAL))
        ankle.add_site(name=f"{side}_sole_site", pos=list(rs.SOLE_LOCAL))
    table_names: list[str | None] = []
    for i, tb in enumerate(tables):
        if tb is None:
            table_names.append(None)
            continue
        name = f"table_{i}"
        spec.worldbody.add_geom(name=name, type=mujoco.mjtGeom.mjGEOM_BOX, size=list(tb.half_size), pos=list(tb.center),
                                contype=1, conaffinity=1, rgba=[0.6, 0.45, 0.3, 1.0])
        table_names.append(name)
    proxy_names = _add_collision_proxies(spec, mjcf_path)
    model = spec.compile()
    data = mujoco.MjData(model)

    def bid(name: str) -> int:
        i = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if i < 0:
            raise KeyError(f"body {name!r} not in {mjcf_path}")
        return int(i)

    joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, j) for j in range(model.njnt)]
    hinge = [j for j in range(model.njnt) if model.jnt_type[j] == mujoco.mjtJoint.mjJNT_HINGE]
    if tuple(joint_names[j] for j in hinge) != tuple(DOF_NAMES):
        raise RuntimeError("MuJoCo hinge order differs from hero_isaacsim.constants.DOF_NAMES")
    dof_qpos_idx = np.array([model.jnt_qposadr[j] for j in hinge], dtype=int)
    dof_vel_idx = np.array([model.jnt_dofadr[j] for j in hinge], dtype=int)
    jnt_range = np.array([model.jnt_range[j] for j in hinge], dtype=np.float64)

    body_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_BODY, b) for b in range(model.nbody)]
    body_id = {n: i for i, n in enumerate(body_names)}
    site_id = {f"{s}_{k}_site": int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_SITE, f"{s}_{k}_site")) for s in SIDES for k in ("palm", "toe", "heel", "sole")}
    floor_geom = int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, "floor"))
    table_geoms = [None if n is None else int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)) for n in table_names]
    pelvis_id = body_id["pelvis"]

    def subtree(b: int) -> set[int]:
        out = {b}
        for c in range(model.nbody):
            if model.body_parentid[c] == b and c != b:
                out |= subtree(c)
        return out

    robot_bodies = subtree(pelvis_id)
    col_geoms = {g for g in range(model.ngeom) if (model.geom_contype[g] != 0 or model.geom_conaffinity[g] != 0) and model.geom_group[g] != PROXY_GROUP}
    robot_geoms = {g for g in col_geoms if model.geom_bodyid[g] in robot_bodies}

    def geoms_of(*names: str, subtree_of: bool = False) -> list[int]:
        """Collision geoms of the named bodies (their own geoms, or their whole subtrees)."""
        out: set[int] = set()
        for n in names:
            bodies = subtree(body_id[n]) if subtree_of else {body_id[n]}
            out |= {g for g in col_geoms if model.geom_bodyid[g] in bodies}
        return sorted(out)

    foot_geoms = {s: set(geoms_of(f"{s}_ankle_roll_link", subtree_of=True)) for s in SIDES}
    leg_geoms = {s: set(geoms_of(f"{s}_knee_link", subtree_of=True)) for s in SIDES}   # knee + ankle links + foot spheres
    torso_geoms = geoms_of("pelvis", "torso_link", "waist_yaw_link", "waist_roll_link")
    thigh_geoms = {s: geoms_of(f"{s}_hip_pitch_link", f"{s}_hip_roll_link", f"{s}_hip_yaw_link", f"{s}_knee_link") for s in SIDES}
    arm_geoms = {s: geoms_of(f"{s}_elbow_link", subtree_of=True) for s in SIDES}      # elbow, wrists, hand, all finger links
    # QP collision-avoidance pairs use the analytic proxies (capsules / boxes, exact distances); the gate uses meshes.
    pid = {n: int(mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_GEOM, n)) for n in proxy_names}
    proxy_geoms = set(pid.values())
    hands = {s: [pid[f"{s}_hand_proxy"], pid[f"{s}_thumb_proxy"]] for s in SIDES}
    forearm = {s: [pid[f"{s}_forearm_proxy"]] for s in SIDES}
    upperarm = {s: [pid[f"{s}_upperarm_proxy"]] for s in SIDES}
    legs_p = {s: [pid[f"{s}_thigh_proxy"], pid[f"{s}_shank_proxy"]] for s in SIDES}
    hips_p = [pid[f"{s}_hip_pitch_proxy"] for s in SIDES] + [pid[f"{s}_hip_roll_proxy"] for s in SIDES]
    trunk_p = [pid["torso_proxy"], pid["pelvis_proxy"]]
    body_partners = trunk_p + legs_p["left"] + legs_p["right"]
    pairs: list[tuple[list[int], list[int]]] = []
    for s in SIDES:
        pairs.append((hands[s] + forearm[s], body_partners))
    pairs.append((hands["left"] + forearm["left"], hands["right"] + forearm["right"] + upperarm["right"]))
    pairs.append((hands["right"] + forearm["right"], upperarm["left"]))
    arm_geoms_all = hands["left"] + forearm["left"] + hands["right"] + forearm["right"]
    pairs.append((arm_geoms_all, [floor_geom]))
    shoulders_p = {s: [pid[f"{s}_shoulder_pitch_proxy"], pid[f"{s}_shoulder_roll_proxy"]] for s in SIDES}
    # close-quarters pairs (small minimum distance): upper arm vs trunk, hands / forearms vs the hip motor housings,
    # hands vs their own upper arm and both shoulder housings (full elbow flexion folds the Dex3 hand onto the shoulder)
    pairs_upperarm: list[tuple[list[int], list[int]]] = [
        (arm_geoms_all, hips_p),
        (hands["left"], upperarm["left"] + shoulders_p["left"] + shoulders_p["right"]),
        (hands["right"], upperarm["right"] + shoulders_p["right"] + shoulders_p["left"]),
    ]
    pairs_upperarm_trunk: list[tuple[list[int], list[int]]] = [(upperarm["left"] + upperarm["right"], trunk_p)]
    # table pairs are added per segment by the solver (MinkSolver.limits_for_segment)

    # links within kinematic distance <= 2 (parent-child is filtered by MuJoCo; grandparent-grandchild is not) touch
    # through their over-approximated collision meshes (wrist_roll|wrist_yaw at large wrist pitch, pelvis|hip_roll in
    # deep squats): exclude these pairs from the self-collision gate.
    def ancestors(bb: int, depth: int) -> set[int]:
        out = {bb}
        cur = bb
        for _ in range(depth):
            cur = int(model.body_parentid[cur])
            out.add(cur)
        return out

    excluded = set()
    for g1 in robot_geoms:
        for g2 in robot_geoms:
            if g1 >= g2:
                continue
            b1, b2 = int(model.geom_bodyid[g1]), int(model.geom_bodyid[g2])
            if b2 in ancestors(b1, 2) or b1 in ancestors(b2, 2) or (ancestors(b1, 1) & ancestors(b2, 1)):
                excluded.add((g1, g2))
    # baseline self-contact pairs at the holosoma default pose (model artefacts, excluded from the gate)
    q = np.zeros(model.nq)
    q[3] = 1.0
    q[2] = rs.STANDING_ROOT_Z
    q[dof_qpos_idx] = rs.DEFAULT_DOF_ARRAY
    data.qpos[:] = q
    mujoco.mj_kinematics(model, data)
    mujoco.mj_collision(model, data)
    baseline = set()
    for i in range(data.ncon):
        c = data.contact[i]
        g1, g2 = int(c.geom1), int(c.geom2)
        if g1 in robot_geoms and g2 in robot_geoms:
            baseline.add((min(g1, g2), max(g1, g2)))
    data.qpos[:] = 0.0
    data.qpos[3] = 1.0
    mujoco.mj_kinematics(model, data)
    r_flat = {s: data.xmat[body_id[f"{s}_ankle_roll_link"]].reshape(3, 3).copy() for s in SIDES}
    return ModelBundle(model=model, data=data, mjcf_path=mjcf_path, nq=model.nq, nv=model.nv, dof_qpos_idx=dof_qpos_idx,
                       dof_vel_idx=dof_vel_idx, jnt_range=jnt_range, body_id=body_id, site_id=site_id, table_geoms=table_geoms,
                       foot_geoms=foot_geoms, leg_geoms=leg_geoms, robot_geoms=robot_geoms, floor_geom=floor_geom,
                       collision_pairs=pairs, collision_pairs_upperarm=pairs_upperarm, collision_pairs_upperarm_trunk=pairs_upperarm_trunk, arm_geoms_all=arm_geoms_all,
                       table_partner_geoms=body_partners, proxy_geoms=proxy_geoms, baseline_self_pairs=baseline | excluded,
                       excluded_self_pairs=excluded, r_flat=r_flat)


# --------------------------------------------------------------------------------------------------
# Small geometry helpers
# --------------------------------------------------------------------------------------------------
def _ry_nose_down(theta: float) -> np.ndarray:
    """Foot rotation about the lateral axis; positive pitches the toes down (MuJoCo Ry convention)."""
    return rot_y(theta)


def polygon_margin(point_xy: np.ndarray, pts_xy: np.ndarray) -> float:
    """Signed distance of ``point_xy`` to the convex hull of ``pts_xy`` (positive inside)."""
    pts = np.asarray(pts_xy, dtype=np.float64).reshape(-1, 2)
    if len(pts) == 0:
        return -1.0
    if len(pts) == 1:
        return -float(np.linalg.norm(point_xy - pts[0]))
    if len(pts) == 2:
        a, b = pts
        ab = b - a
        t = float(np.clip(np.dot(point_xy - a, ab) / max(1e-12, np.dot(ab, ab)), 0.0, 1.0))
        return -float(np.linalg.norm(point_xy - (a + t * ab)))
    # monotone chain hull
    order = np.lexsort((pts[:, 1], pts[:, 0]))
    p = pts[order]

    def cross(o, a, b):
        return (a[0] - o[0]) * (b[1] - o[1]) - (a[1] - o[1]) * (b[0] - o[0])

    lower: list[np.ndarray] = []
    for q in p:
        while len(lower) >= 2 and cross(lower[-2], lower[-1], q) <= 1e-12:
            lower.pop()
        lower.append(q)
    upper: list[np.ndarray] = []
    for q in p[::-1]:
        while len(upper) >= 2 and cross(upper[-2], upper[-1], q) <= 1e-12:
            upper.pop()
        upper.append(q)
    hull = np.array(lower[:-1] + upper[:-1])
    if len(hull) < 3:
        return polygon_margin(point_xy, hull)
    margin = math.inf
    n = len(hull)
    for i in range(n):
        a, b = hull[i], hull[(i + 1) % n]
        e = b - a
        L = float(np.linalg.norm(e))
        if L < 1e-9:
            continue
        # hull is counter-clockwise: inside = left of every edge
        d = float((e[0] * (point_xy[1] - a[1]) - e[1] * (point_xy[0] - a[0])) / L)
        margin = min(margin, d)
    return float(margin)


def _quat_from_xmat(xmat9: np.ndarray) -> np.ndarray:
    return mat_to_quat_wxyz(np.asarray(xmat9).reshape(3, 3))


# --------------------------------------------------------------------------------------------------
# Segment planning (per-frame targets)
# --------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class FootPlan:
    mode: str                     # "flat" | "heel_lift" | "kneel" | "kneel_up"
    pivot_pos: np.ndarray         # (T,3) toe (heel_lift / kneel) or ankle_roll origin (flat) world target
    foot_rot: np.ndarray          # (T,3,3) foot orientation target
    pitch: np.ndarray             # (T,) nose-down pitch (heel lift) or kneel blend
    knee_pos: np.ndarray | None   # (T,3) knee link target for the kneeling leg
    contact: np.ndarray           # (T,) 1 = pivot is on the floor
    knee_cost: np.ndarray         # (T,) knee task position cost
    pivot_cost: np.ndarray        # (T,) toe/ankle position cost
    ori_cost: np.ndarray          # (T,) foot orientation cost


@dataclasses.dataclass
class SegmentPlan:
    frames: int
    reach_frames: int
    hold_frames: int
    palm_pos: np.ndarray          # (T,2,3)
    palm_quat: np.ndarray         # (T,2,4) wxyz
    palm_cost: np.ndarray         # (T,2) 0 -> task disabled
    pelvis_pos: np.ndarray        # (T,3)
    pelvis_rpy: np.ndarray        # (T,3) roll, pitch, yaw
    posture_q: np.ndarray         # (T,29)
    posture_cost: np.ndarray      # (T,29)
    feet: dict[str, FootPlan]
    target_pos_frame: np.ndarray  # (T,2,3) commanded target per frame (records the re-target switch)
    target_quat_frame: np.ndarray  # (T,2,4)
    target_final_pos: np.ndarray  # (2,3)
    target_final_quat: np.ndarray  # (2,4)
    retarget_frame: int           # -1 if none
    kind: str                     # "reach" | "retract"
    support_points: list[np.ndarray]  # per frame (K,2) planned support polygon (xy)


@dataclasses.dataclass
class ClipState:
    """Mutable state carried across segments."""

    yaw0: float
    foot_pos: dict[str, np.ndarray]       # ankle_roll world position when flat
    toe_pos: dict[str, np.ndarray]        # toe site world position (fixed pivot)
    stance_center: np.ndarray             # xy
    pelvis_height: float
    pelvis_pitch: float
    pelvis_yaw_delta: float
    pelvis_shift: np.ndarray              # xy
    waist_pitch: float
    waist_yaw: float
    leg_prior: dict[str, dict[str, float]]   # side -> hip/knee/ankle prior
    foot_mode: dict[str, str]
    heel_pitch: dict[str, float]
    kneel_side: str


def _waypoint_path(points: list[np.ndarray], n: int, u_of_t: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Piecewise smoothstep interpolation through ``points`` (time split proportional to leg length, min 15 %).

    Returns (positions (n,3), leg boundaries in phase units)."""
    pts = [np.asarray(p, dtype=np.float64) for p in points]
    legs = np.array([np.linalg.norm(pts[i + 1] - pts[i]) for i in range(len(pts) - 1)])
    if legs.sum() < 1e-9:
        return np.repeat(pts[-1][None], n, axis=0), np.linspace(0.0, 1.0, len(pts))
    frac = np.maximum(legs / legs.sum(), 0.15)
    frac = frac / frac.sum()
    bounds = np.concatenate([[0.0], np.cumsum(frac)])
    bounds[-1] = 1.0
    out = np.empty((n, 3))
    for t in range(n):
        u = float(np.clip(u_of_t[t], 0.0, 1.0))
        k = int(np.clip(np.searchsorted(bounds, u, side="right") - 1, 0, len(legs) - 1))
        a = (u - bounds[k]) / max(1e-9, bounds[k + 1] - bounds[k])
        out[t] = pts[k] + smoothstep(a) * (pts[k + 1] - pts[k])
    return out, bounds


def _clearance_height(q0: np.ndarray, qt: np.ndarray, side: str, surface_z: float) -> float:
    """Palm height that keeps the whole hand envelope >= 3 cm above ``surface_z`` for every orientation of the
    start->target slerp (the sweep the hand performs while it advances over the table / floor)."""
    worst = 0.0
    for a in np.linspace(0.0, 1.0, 9):
        R = quat_wxyz_to_mat(quat_slerp_wxyz(q0, qt, float(a)))
        worst = max(worst, -rs.hand_lowest_offset(R, side))
    return surface_z + worst + 0.03


def _palm_waypoints(p0: np.ndarray, q0: np.ndarray, target: HandTarget, shoulder_xy: np.ndarray, surface_z: float) -> list[np.ndarray]:
    """Lift -> advance (at a table/floor-clearing height) -> descend / (descend + ingress) -> target."""
    pt = np.asarray(target.pos, dtype=np.float64)
    z_clear = _clearance_height(q0, target.quat_wxyz(), target.hand, surface_z)
    if target.approach == "descend":
        a = np.array([p0[0], p0[1], max(p0[2] + target.lift, z_clear)])
        b = np.array([pt[0], pt[1], max(pt[2] + target.hover, z_clear, a[2] - 0.02)])
        return [p0, a, b, pt]
    a = np.array([p0[0], p0[1], max(p0[2] + 0.5 * target.lift, z_clear)])
    u = shoulder_xy - pt[:2]
    nrm = float(np.linalg.norm(u))
    u = u / nrm if nrm > 1e-6 else np.array([-1.0, 0.0])
    b = pt + target.ingress * np.array([u[0], u[1], 0.0])
    if z_clear > pt[2] + 0.01:
        b_high = np.array([b[0], b[1], max(z_clear, a[2] - 0.02)])
        return [p0, a, b_high, b, pt]
    return [p0, a, b, pt]


def _path_length(points: list[np.ndarray]) -> float:
    return float(sum(np.linalg.norm(points[i + 1] - points[i]) for i in range(len(points) - 1)))


def _leg_prior_for(base: BaseStrategy, side: str, state: ClipState) -> dict[str, float]:
    if base.foot_mode == "kneel" and base.kneel_side == side:
        return dict(rs.kneel_leg_prior(side))
    if base.leg_prior is not None:
        return dict(base.leg_prior)
    prior = rs.solve_leg_prior(base.pelvis_height, base.pelvis_pitch)
    return {"hip_pitch": prior.hip_pitch, "knee": prior.knee, "ankle_pitch": prior.ankle_pitch}


def _posture_from(leg: dict[str, dict[str, float]], waist_pitch: float, waist_yaw: float, pelvis_height: float = rs.STANDING_ROOT_Z,
                  pelvis_pitch: float = 0.0) -> np.ndarray:
    """Posture prior: leg boxes + waist targets; the arms' rest pose swings forward / outward as the squat deepens or the
    torso leans (bow) so that a passive (inactive) arm does not hang into the thighs."""
    q = rs.DEFAULT_DOF_ARRAY.copy()
    idx = {n: i for i, n in enumerate(DOF_NAMES)}
    for side in SIDES:
        q[idx[f"{side}_hip_pitch_joint"]] = leg[side]["hip_pitch"]
        q[idx[f"{side}_knee_joint"]] = leg[side]["knee"]
        q[idx[f"{side}_ankle_pitch_joint"]] = leg[side]["ankle_pitch"]
    q[idx["waist_pitch_joint"]] = waist_pitch
    q[idx["waist_yaw_joint"]] = waist_yaw
    depth = float(np.clip((0.70 - pelvis_height) / 0.30, 0.0, 1.0))
    lean = float(np.clip((pelvis_pitch + max(0.0, waist_pitch)) / 1.2, 0.0, 1.0))
    swing = max(depth, lean)
    for side, sgn in (("left", 1.0), ("right", -1.0)):
        q[idx[f"{side}_shoulder_pitch_joint"]] = 0.2 - 1.0 * swing
        q[idx[f"{side}_shoulder_roll_joint"]] = sgn * (0.2 + 0.35 * swing)
        q[idx[f"{side}_elbow_joint"]] = 0.6 + 0.25 * swing
    return q


def _posture_cost(cfg: SolverConfig, active: tuple[bool, bool], kneel_side: str, retract: bool = False) -> np.ndarray:
    c = np.empty(29)
    c[list(rs.LEFT_LEG_DOF_IDX) + list(rs.RIGHT_LEG_DOF_IDX)] = cfg.posture_leg_cost
    if kneel_side:
        c[list(rs.LEFT_LEG_DOF_IDX if kneel_side == "left" else rs.RIGHT_LEG_DOF_IDX)] = 1.0
    c[list(rs.WAIST_DOF_IDX)] = cfg.posture_waist_cost
    c[list(rs.LEFT_ARM_DOF_IDX)] = cfg.posture_arm_active_cost if (active[0] and not retract) else (1.0 if retract else cfg.posture_arm_inactive_cost)
    c[list(rs.RIGHT_ARM_DOF_IDX)] = cfg.posture_arm_active_cost if (active[1] and not retract) else (1.0 if retract else cfg.posture_arm_inactive_cost)
    return c


def _support_points(state_feet: dict[str, FootPlan], t: int, side_flat_pts: dict[str, np.ndarray]) -> np.ndarray:
    pts = []
    for side, fp in state_feet.items():
        if fp.mode == "flat":
            pts.append(side_flat_pts[side])
        elif fp.mode == "heel_lift":
            toe = fp.pivot_pos[t]
            pts.append(np.array([[toe[0] + 0.0, toe[1] + 0.03], [toe[0], toe[1] - 0.03]]))
        else:  # kneel / kneel_up: toe + knee once on the floor, toe only otherwise
            toe = fp.pivot_pos[t]
            pts.append(np.array([[toe[0], toe[1] + 0.03], [toe[0], toe[1] - 0.03]]))
            if fp.knee_pos is not None and fp.knee_pos[t][2] < 0.09:
                k = fp.knee_pos[t]
                pts.append(np.array([[k[0] + 0.04, k[1] + 0.05], [k[0] + 0.04, k[1] - 0.05], [k[0] - 0.04, k[1]]]))
    return np.concatenate(pts, axis=0)


def _flat_sole_points(foot_pos: np.ndarray, yaw: float) -> np.ndarray:
    R = rot_z(yaw)
    pts = np.array(rs.SOLE_POINTS_LOCAL)[:, :2]
    return (R[:2, :2] @ pts.T).T + foot_pos[:2]


def _foot_plans(state: ClipState, base: BaseStrategy | None, n_reach: int, n_total: int, s_base: np.ndarray,
                retract: bool) -> dict[str, FootPlan]:
    """Per-foot targets for one segment (flat / heel-lift ramp / kneel descent or stand-up)."""
    feet: dict[str, FootPlan] = {}
    for side in SIDES:
        Rz = rot_z(state.yaw0)
        mode_prev = state.foot_mode[side]
        if retract:
            mode_new, heel_new, kneel_here = "flat", 0.0, False
        else:
            assert base is not None
            kneel_here = base.foot_mode == "kneel" and base.kneel_side == side
            mode_new = "kneel" if kneel_here else ("heel_lift" if base.foot_mode == "heel_lift" else "flat")
            heel_new = base.heel_lift_pitch if base.foot_mode == "heel_lift" else 0.0
        heel_prev = state.heel_pitch[side]
        pitch = heel_prev + (heel_new - heel_prev) * s_base
        toe = state.toe_pos[side]
        pivot = np.empty((n_total, 3))
        frot = np.empty((n_total, 3, 3))
        contact = np.ones(n_total)
        knee_pos = None
        knee_cost = np.zeros(n_total)
        pivot_cost = np.full(n_total, 200.0)
        ori_cost = np.full(n_total, 15.0)
        if mode_new == "kneel" or (mode_prev == "kneel" and retract):
            # kneel descent (or stand-up in retract): toe slides to the kneel toe location, foot pitches nose-down
            # to 90 deg, knee comes down to KNEE_FLOOR_Z.  The support polygon during the transition is the other
            # foot + this toe; the kneeling leg's floor contacts are allowed by the contact gate.
            y = toe[1]
            knee_x_kneel = state.stance_center[0] - 0.05
            toe_kneel = np.array([knee_x_kneel - KNEEL_SHANK_X, y, 0.0])
            knee_stand = np.array([toe[0] - rs.TOE_LOCAL[0] * math.cos(state.yaw0), y, 0.40])
            knee_kneel = np.array([knee_x_kneel, y, KNEE_FLOOR_Z])
            if mode_new == "kneel":
                s = s_base
                toe_from, toe_to = toe, toe_kneel
                knee_from, knee_to = knee_stand, knee_kneel
                pitch = (math.pi / 2.0) * s
            else:
                s = s_base
                toe_from, toe_to = toe, state.foot_pos[side] + Rz @ np.array(rs.TOE_LOCAL) - np.array([0.0, 0.0, rs.TOE_LOCAL[2]]) * 0.0
                toe_to = np.array([state.foot_pos[side][0], state.foot_pos[side][1], 0.0]) + Rz @ np.array([rs.TOE_LOCAL[0], 0.0, 0.0])
                knee_from, knee_to = knee_kneel, knee_stand
                pitch = (math.pi / 2.0) * (1.0 - s)
            knee_pos = knee_from[None] + (knee_to - knee_from)[None] * s[:, None]
            pivot = toe_from[None] + (toe_to - toe_from)[None] * s[:, None]
            moving = (s > 0.02) & (s < 0.98)
            contact = np.where(moving, 0.0, 1.0)
            knee_cost = np.where(s > 0.5, 30.0, 0.0) if mode_new == "kneel" else np.where(s < 0.5, 30.0, 0.0)
            pivot_cost = np.where(moving, 20.0, 200.0)
            ori_cost = np.full(n_total, 2.0)
            for t in range(n_total):
                frot[t] = Rz @ _ry_nose_down(float(pitch[t]))
            mode_label = "kneel" if mode_new == "kneel" else "kneel_up"
        elif mode_new == "heel_lift" or heel_prev > 0.0:
            for t in range(n_total):
                frot[t] = Rz @ _ry_nose_down(float(pitch[t]))
            pivot[:] = toe[None]
            mode_label = "heel_lift" if heel_new > 0.0 else "heel_down"
        else:
            frot[:] = Rz[None]
            pivot[:] = state.foot_pos[side][None]
            mode_label = "flat"
        if mode_label == "flat":
            feet[side] = FootPlan("flat", pivot, frot, np.zeros(n_total), None, contact, knee_cost, pivot_cost, ori_cost)
        elif mode_label in ("heel_lift", "heel_down"):
            feet[side] = FootPlan("heel_lift", pivot, frot, pitch, None, contact, knee_cost, pivot_cost, ori_cost)
        else:
            feet[side] = FootPlan(mode_label, pivot, frot, pitch, knee_pos, contact, knee_cost, pivot_cost, ori_cost)
    return feet


def plan_segment(seg: SegmentSpec | None, spec: ClipSpec, state: ClipState, palm_start: dict[str, tuple[np.ndarray, np.ndarray]],
                 shoulder_xy: dict[str, np.ndarray], cfg: SolverConfig, *, time_scale_extra: float = 1.0,
                 retract: bool = False, palm_rest: dict[str, tuple[np.ndarray, np.ndarray]] | None = None) -> SegmentPlan:
    """Per-frame targets for one reach segment (``seg``) or for the retract-to-rest phase (``seg is None``, ``retract=True``).

    Retract (broad_v2 ``retract_share`` / bench ``bench_retract_share``): every active palm is TRACKED (palm task on) along the last
    segment's approach route run backwards from the last COMMANDED palm pose -- pull back along the ingress vector, rise to the table /
    floor clearance height, travel back over the edge, descend onto the rest pose ``palm_rest[side]`` (the clip's frame-0 palm;
    :func:`retract_rest_palm` fallback) -- on the piecewise-smoothstep profile (zero start / end velocity), the orientation slerped from
    the commanded to the rest orientation, while the active arms' posture cost ramps posture_arm_active_cost -> 1.0 (inactive arms keep
    theirs) and the base returns to standing. This avoids an abrupt posture-only
    jump at the start of retraction."""
    active = spec.active_hands
    base = None if retract else seg.base
    # ---- base (pelvis / torso / legs) end state --------------------------------------------------------
    if retract:
        pelvis_h_new = rs.STANDING_ROOT_Z
        pelvis_pitch_new, yaw_delta_new, shift_new = 0.0, 0.0, np.zeros(2)
        waist_pitch_new, waist_yaw_new = 0.0, 0.0
        prior = rs.solve_leg_prior(pelvis_h_new, 0.0)
        leg_new = {s: {"hip_pitch": prior.hip_pitch, "knee": prior.knee, "ankle_pitch": prior.ankle_pitch} for s in SIDES}
        time_scale = 1.0
    else:
        pelvis_h_new = base.pelvis_height
        pelvis_pitch_new, yaw_delta_new = base.pelvis_pitch, base.pelvis_yaw_delta
        shift_new = np.asarray(base.pelvis_shift)
        waist_pitch_new, waist_yaw_new = base.waist_pitch, base.waist_yaw
        leg_new = {s: _leg_prior_for(base, s, state) for s in SIDES}
        if base.foot_mode == "kneel":
            shift_new = shift_new + np.array([0.12, 0.0])
        elif base.foot_mode == "heel_lift":
            shift_new = shift_new + np.array([rs.HEEL_LIFT_PELVIS_X_PREF, 0.0])
        time_scale = seg.time_scale
    # ---- duration -------------------------------------------------------------------------------------
    paths: dict[str, list[np.ndarray]] = {}
    retract_q_start: dict[str, np.ndarray] = {}
    for i, side in enumerate(SIDES):
        if not active[i]:
            continue
        p0, _ = palm_start[side]
        if retract:
            rest_pos, rest_quat = _retract_rest(side, palm_start, palm_rest, pelvis_h_new)
            last = spec.segments[-1] if spec.segments else None
            tgt_last = next((t for t in last.targets if t.hand == side), None) if last is not None else None
            if tgt_last is not None:
                # egress = the reach's approach route (lift -> advance at clearance height -> ingress) run backwards, starting at the LAST
                # COMMANDED palm pose (target + mid-reach retarget offset), not at the achieved palm: a hand that was held back by a limit
                # retains its palm residual, avoiding an abrupt shoulder correction at the switch.
                if last.retarget is not None:
                    tgt_last = dataclasses.replace(tgt_last, pos=tuple(float(v) for v in (np.asarray(tgt_last.pos) + np.asarray(last.retarget.offset))))
                surface_for_clearance = (last.surface_z if last.has_table else 0.0) + last.sweep_clearance_extra
                wps = _palm_waypoints(rest_pos, rest_quat, tgt_last, shoulder_xy[side], surface_for_clearance)
                paths[side] = [np.asarray(w, dtype=np.float64) for w in reversed(wps)]
                retract_q_start[side] = tgt_last.quat_wxyz()
            else:
                paths[side] = [p0, rest_pos]
                retract_q_start[side] = np.asarray(palm_start[side][1], dtype=np.float64)
        else:
            tgt = next(t for t in seg.targets if t.hand == side)
            q0_side = palm_start[side][1]
            if getattr(seg, "replan", False):
                # Corrective re-reach: straight path from the current palm to the shifted target, starting from rest.
                paths[side] = [p0, np.asarray(tgt.pos, dtype=np.float64)]
            else:
                surface_for_clearance = (seg.surface_z if seg.has_table else 0.0) + seg.sweep_clearance_extra   # the floor plane is always there
                paths[side] = _palm_waypoints(p0, q0_side, tgt, shoulder_xy[side], surface_for_clearance)
    path_len = max([_path_length(p) for p in paths.values()] + [0.0])
    travel = 4.0 * abs(pelvis_h_new - state.pelvis_height) + abs(pelvis_pitch_new - state.pelvis_pitch) + abs(waist_pitch_new - state.waist_pitch) + abs(waist_yaw_new - state.waist_yaw)
    foot_change = any(state.foot_mode[s] != ("flat" if retract else ("kneel" if (base.foot_mode == "kneel" and base.kneel_side == s) else base.foot_mode)) for s in SIDES)
    travel += 1.6 if foot_change else 0.0
    replan_seg = (not retract) and bool(getattr(seg, "replan", False))
    if replan_seg:
        ang = max([quat_angle_wxyz(palm_start[s][1], next(t for t in seg.targets if t.hand == s).quat_wxyz()) for i, s in enumerate(SIDES) if active[i]] + [0.0])
        T_reach = float(np.clip(REPLAN_T_BASE_S + REPLAN_T_PER_M * path_len + REPLAN_T_PER_PI * ang / math.pi + 0.35 * travel,
                                REPLAN_REACH_MIN_S, REPLAN_REACH_MAX_S)) * time_scale * time_scale_extra
    else:
        T_reach = float(np.clip(0.8 + 1.6 * path_len + 0.35 * travel, 1.5, 6.0)) * time_scale * time_scale_extra
    n_reach = int(round(T_reach * FPS))
    retarget = None if (retract or seg.retarget is None) else seg.retarget
    n_ext = 0
    k_r = -1
    if retarget is not None:
        k_r = int(round(retarget.phase * n_reach))
        n_ext = int(round(FPS * (0.3 + 1.2 * float(np.linalg.norm(retarget.offset))) * time_scale))
    n_reach_total = n_reach + n_ext
    n_hold = RETRACT_HOLD_FRAMES if retract else int(round(seg.hold_s * FPS))
    n_total = n_reach_total + n_hold
    # ---- base timing ----------------------------------------------------------------------------------
    lead = (not retract) and base.base_timing == "lead"
    t_idx = np.arange(n_total, dtype=np.float64)
    if lead:
        s_base = np.array([smoothstep(t / max(1.0, 0.45 * n_reach_total)) for t in t_idx])
        u_hand = np.clip((t_idx - 0.25 * n_reach_total) / max(1.0, 0.75 * n_reach_total), 0.0, 1.0)
    else:
        s_base = np.array([smoothstep(t / max(1.0, n_reach_total)) for t in t_idx])
        u_hand = np.clip(t_idx / max(1.0, n_reach_total), 0.0, 1.0)
    # ---- pelvis + posture -----------------------------------------------------------------------------
    pelvis_pos = np.empty((n_total, 3))
    pelvis_rpy = np.empty((n_total, 3))
    posture_q = np.empty((n_total, 29))
    q_from = _posture_from(state.leg_prior, state.waist_pitch, state.waist_yaw, state.pelvis_height, state.pelvis_pitch)
    q_to = _posture_from(leg_new, waist_pitch_new, waist_yaw_new, pelvis_h_new, pelvis_pitch_new)
    for t in range(n_total):
        s = float(s_base[t])
        h = state.pelvis_height + (pelvis_h_new - state.pelvis_height) * s
        pitch = state.pelvis_pitch + (pelvis_pitch_new - state.pelvis_pitch) * s
        ydel = state.pelvis_yaw_delta + (yaw_delta_new - state.pelvis_yaw_delta) * s
        shift = state.pelvis_shift + (shift_new - state.pelvis_shift) * s
        xy = state.stance_center + rot_z(state.yaw0)[:2, :2] @ shift
        pelvis_pos[t] = [xy[0], xy[1], h]
        pelvis_rpy[t] = [0.0, pitch, state.yaw0 + ydel]
        posture_q[t] = q_from + (q_to - q_from) * s
    kneel_side = "" if retract else (base.kneel_side if base.foot_mode == "kneel" else "")
    posture_cost = np.repeat(_posture_cost(cfg, active, kneel_side, retract=retract)[None], n_total, axis=0)
    if retract:
        # no cost step at the switch: the active arms keep the reach's posture weight and ramp to the retract weight (1.0) with the base
        # profile; inactive arms retain posture_arm_inactive_cost to preserve their resting pose.
        for i, arm_idx in enumerate((rs.LEFT_ARM_DOF_IDX, rs.RIGHT_ARM_DOF_IDX)):
            if active[i]:
                posture_cost[:, list(arm_idx)] = (cfg.posture_arm_active_cost + (1.0 - cfg.posture_arm_active_cost) * s_base)[:, None]
            else:
                posture_cost[:, list(arm_idx)] = cfg.posture_arm_inactive_cost
    # ---- palms ----------------------------------------------------------------------------------------
    palm_pos = np.empty((n_total, 2, 3))
    palm_quat = np.empty((n_total, 2, 4))
    palm_cost = np.zeros((n_total, 2))
    target_pos_frame = np.empty((n_total, 2, 3))
    target_quat_frame = np.empty((n_total, 2, 4))
    target_final_pos = np.empty((2, 3))
    target_final_quat = np.empty((2, 4))
    for i, side in enumerate(SIDES):
        p0, q0 = palm_start[side]
        if not active[i]:
            palm_pos[:, i] = p0
            palm_quat[:, i] = q0
            target_pos_frame[:, i] = p0
            target_quat_frame[:, i] = q0
            target_final_pos[i], target_final_quat[i] = p0, q0
            continue
        if retract:
            rest_pos, rest_quat = _retract_rest(side, palm_start, palm_rest, pelvis_h_new)
            pos, _ = _waypoint_path(paths[side], n_total, u_hand)
            palm_pos[:, i] = pos
            q_start = retract_q_start.get(side, q0)
            palm_quat[:, i] = np.array([quat_slerp_wxyz(q_start, rest_quat, smoothstep(u)) for u in u_hand])
            palm_cost[:, i] = cfg.palm_pos_cost
            target_pos_frame[:, i] = rest_pos
            target_quat_frame[:, i] = rest_quat
            target_final_pos[i], target_final_quat[i] = rest_pos, rest_quat
            continue
        tgt = next(t for t in seg.targets if t.hand == side)
        qt = tgt.quat_wxyz()
        pos, bounds = _waypoint_path(paths[side], n_total, u_hand)
        # orientation reaches the target before the final descent / ingress legs start
        u_ori_end = float(bounds[-2]) if len(bounds) >= 3 else 0.85
        u_ori_end = min(max(u_ori_end, 0.35), 0.9)
        if replan_seg:   # the replanner slerps on the same scalar profile as the position
            quat = np.array([quat_slerp_wxyz(q0, qt, smoothstep(u)) for u in u_hand])
        else:
            quat = np.array([quat_slerp_wxyz(q0, qt, smoothstep((u - 0.05) / max(0.1, u_ori_end - 0.05))) for u in u_hand])
        pt_final = np.asarray(tgt.pos, dtype=np.float64)
        target_pos_frame[:, i] = pt_final
        target_quat_frame[:, i] = qt
        if retarget is not None:
            pt_new = pt_final + np.asarray(retarget.offset, dtype=np.float64)
            p_switch = pos[min(k_r, n_total - 1)].copy()
            q_switch = quat[min(k_r, n_total - 1)].copy()
            n_rest = n_reach_total - k_r
            for t in range(k_r, n_total):
                a = smoothstep((t - k_r) / max(1.0, n_rest))
                pos[t] = p_switch + (pt_new - p_switch) * a
                quat[t] = quat_slerp_wxyz(q_switch, qt, a)
            target_pos_frame[k_r:, i] = pt_new
            pt_final = pt_new
        palm_pos[:, i] = pos
        palm_quat[:, i] = quat
        palm_cost[:, i] = cfg.palm_pos_cost
        target_final_pos[i], target_final_quat[i] = pt_final, qt
    feet = _foot_plans(state, base, n_reach_total, n_total, s_base, retract)
    flat_pts = {s: _flat_sole_points(state.foot_pos[s], state.yaw0) for s in SIDES}
    support = [_support_points(feet, t, flat_pts) for t in range(n_total)]
    return SegmentPlan(frames=n_total, reach_frames=n_reach_total, hold_frames=n_hold, palm_pos=palm_pos, palm_quat=palm_quat,
                       palm_cost=palm_cost, pelvis_pos=pelvis_pos, pelvis_rpy=pelvis_rpy, posture_q=posture_q, posture_cost=posture_cost,
                       feet=feet, target_pos_frame=target_pos_frame, target_quat_frame=target_quat_frame, target_final_pos=target_final_pos,
                       target_final_quat=target_final_quat, retarget_frame=k_r, kind="retract" if retract else "reach", support_points=support)


def _blend_from_previous(qs: np.ndarray, q_prev: np.ndarray, n_blend: int) -> np.ndarray:
    """Crossfade the first ``n_blend`` frames of the solved configurations ``qs`` (root xyz + root quat wxyz + dofs) from ``q_prev``
    (weight ``1 - i / n_blend``) to the solution (``i / n_blend``); returns a copy."""
    out = np.array(qs, dtype=np.float64, copy=True)
    last = min(int(n_blend), len(out) - 1)
    for i in range(last + 1):
        w = i / float(n_blend)
        out[i, 0:3] = (1.0 - w) * q_prev[0:3] + w * qs[i, 0:3]
        out[i, 3:7] = quat_slerp_wxyz(q_prev[3:7], qs[i, 3:7], w)
        out[i, 7:] = (1.0 - w) * q_prev[7:] + w * qs[i, 7:]
    return out


def advance_state(state: ClipState, seg: SegmentSpec | None, plan: SegmentPlan, retract: bool) -> None:
    """Carry the segment end state (base, posture priors, foot modes) into the next segment."""
    if retract:
        state.pelvis_height, state.pelvis_pitch, state.pelvis_yaw_delta = rs.STANDING_ROOT_Z, 0.0, 0.0
        state.pelvis_shift = np.zeros(2)
        state.waist_pitch = state.waist_yaw = 0.0
        prior = rs.solve_leg_prior(rs.STANDING_ROOT_Z, 0.0)
        state.leg_prior = {s: {"hip_pitch": prior.hip_pitch, "knee": prior.knee, "ankle_pitch": prior.ankle_pitch} for s in SIDES}
        for s in SIDES:
            state.foot_mode[s] = "flat"
            state.heel_pitch[s] = 0.0
        state.kneel_side = ""
        return
    base = seg.base
    state.pelvis_height = base.pelvis_height
    state.pelvis_pitch = base.pelvis_pitch
    state.pelvis_yaw_delta = base.pelvis_yaw_delta
    state.pelvis_shift = np.asarray(base.pelvis_shift) + (np.array([0.12, 0.0]) if base.foot_mode == "kneel" else
                                                          (np.array([rs.HEEL_LIFT_PELVIS_X_PREF, 0.0]) if base.foot_mode == "heel_lift" else 0.0))
    state.waist_pitch, state.waist_yaw = base.waist_pitch, base.waist_yaw
    state.leg_prior = {s: _leg_prior_for(base, s, state) for s in SIDES}
    for s in SIDES:
        fp = plan.feet[s]
        if fp.mode == "kneel":
            state.foot_mode[s] = "kneel"
            state.toe_pos[s] = fp.pivot_pos[-1].copy()
            state.heel_pitch[s] = 0.0
        elif fp.mode == "heel_lift":
            state.foot_mode[s] = "heel_lift" if base.foot_mode == "heel_lift" else "flat"
            state.heel_pitch[s] = base.heel_lift_pitch if base.foot_mode == "heel_lift" else 0.0
        else:
            state.foot_mode[s] = "flat"
            state.heel_pitch[s] = 0.0
    state.kneel_side = base.kneel_side if base.foot_mode == "kneel" else ""


# --------------------------------------------------------------------------------------------------
# mink solver
# --------------------------------------------------------------------------------------------------
class MinkSolver:
    """Task / limit bundle for one model (built once per clip)."""

    def __init__(self, bundle: ModelBundle, cfg: SolverConfig):
        import mink

        self.mink = mink
        self.bundle = bundle
        self.cfg = cfg
        m = bundle.model
        lm = cfg.lm_damping
        self.q = mink.Configuration(m)
        self.palm = {s: mink.FrameTask(f"{s}_palm_site", "site", position_cost=cfg.palm_pos_cost, orientation_cost=cfg.palm_ori_cost, lm_damping=lm) for s in SIDES}
        self.ankle = {s: mink.FrameTask(f"{s}_ankle_roll_link", "body", position_cost=cfg.foot_pos_cost, orientation_cost=cfg.foot_ori_cost, lm_damping=lm) for s in SIDES}
        self.toe = {s: mink.FrameTask(f"{s}_toe_site", "site", position_cost=0.0, orientation_cost=0.0, lm_damping=lm) for s in SIDES}
        self.knee = {s: mink.FrameTask(f"{s}_knee_link", "body", position_cost=0.0, orientation_cost=0.0, lm_damping=lm) for s in SIDES}
        self.pelvis = mink.FrameTask("pelvis", "body", position_cost=list(cfg.pelvis_pos_cost), orientation_cost=list(cfg.pelvis_ori_cost), lm_damping=lm)
        self.posture = mink.PostureTask(m, cost=np.zeros(m.nv), lm_damping=lm)
        prev_cost = np.full(m.nv, cfg.prev_cost)
        prev_cost[:6] = 0.0
        prev_cost[bundle.dof_vel_idx[list(rs.LOWER_DOF_IDX)]] = cfg.prev_cost_lower
        self.prev = mink.PostureTask(m, cost=prev_cost, lm_damping=lm)
        self.com = mink.ComTask(cost=[cfg.com_xy_cost, cfg.com_xy_cost, 0.0], lm_damping=lm)
        self.tasks = [self.pelvis, *self.palm.values(), *self.ankle.values(), *self.toe.values(), *self.knee.values(), self.com, self.posture, self.prev]
        # ConfigurationLimit at 98 % of the range (>= limit_min_margin each side); snapshot semantics -> shrink & restore
        backup = m.jnt_range.copy()
        for j in range(m.njnt):
            if m.jnt_type[j] == 3 and m.jnt_limited[j]:
                lo, hi = backup[j]
                margin = max(cfg.limit_min_margin, 0.5 * (1.0 - cfg.limit_utilization) * (hi - lo))
                m.jnt_range[j] = (lo + margin, hi - margin)
        try:
            self.conf_limit = mink.ConfigurationLimit(m, gain=0.5)
        finally:
            m.jnt_range[:] = backup
        self.vel_limit = mink.VelocityLimit(m, {n: cfg.vel_limit_rad_per_frame / DT for n in DOF_NAMES})
        self._col_cache: dict[int | None, Any] = {}
        self.qp_failures = 0

    def _make_collision_limit(self, pairs, min_dist: float):
        """mink filters pairs by contype/conaffinity at construction: make the proxies collidable for that moment."""
        m = self.bundle.model
        ids = sorted(self.bundle.proxy_geoms)
        ct, ca = m.geom_contype[ids].copy(), m.geom_conaffinity[ids].copy()
        m.geom_contype[ids] = 1
        m.geom_conaffinity[ids] = 1
        try:
            return self.mink.CollisionAvoidanceLimit(m, pairs, gain=self.cfg.collision_gain, minimum_distance_from_collisions=min_dist,
                                                     collision_detection_distance=self.cfg.collision_detect_dist)
        finally:
            m.geom_contype[ids] = ct
            m.geom_conaffinity[ids] = ca

    def limits_for_segment(self, seg_index: int | None) -> list[Any]:
        table = None if seg_index is None or seg_index >= len(self.bundle.table_geoms) else self.bundle.table_geoms[seg_index]
        key = table
        if key not in self._col_cache:
            pairs = list(self.bundle.collision_pairs)
            if table is not None:
                pairs.append((self.bundle.arm_geoms_all, [table]))
                pairs.append((self.bundle.table_partner_geoms, [table]))
            self._col_cache[key] = [self._make_collision_limit(pairs, self.cfg.collision_min_dist),
                                    self._make_collision_limit(self.bundle.collision_pairs_upperarm, self.cfg.collision_min_dist_upperarm),
                                    self._make_collision_limit(self.bundle.collision_pairs_upperarm_trunk, 0.0)]
        return [self.conf_limit, self.vel_limit, *self._col_cache[key]]

    # ------------------------------------------------------------------------------------------
    def _se3(self, R: np.ndarray, p: np.ndarray):
        return self.mink.SE3.from_rotation_and_translation(self.mink.SO3.from_matrix(np.asarray(R, dtype=np.float64)), np.asarray(p, dtype=np.float64))

    def _se3_quat(self, q_wxyz: np.ndarray, p: np.ndarray):
        return self.mink.SE3.from_rotation_and_translation(self.mink.SO3(wxyz=np.asarray(q_wxyz, dtype=np.float64)), np.asarray(p, dtype=np.float64))

    def _step(self, limits: list[Any]) -> bool:
        try:
            v = self.mink.solve_ik(self.q, self.tasks, DT, self.cfg.solver, damping=self.cfg.damping, limits=limits)
        except self.mink.NoSolutionFound:
            self.qp_failures += 1
            try:
                v = self.mink.solve_ik(self.q, self.tasks, DT, self.cfg.solver, damping=self.cfg.damping, limits=limits[:2])
            except self.mink.NoSolutionFound:
                return False
        self.q.integrate_inplace(v, DT)
        return True

    def solve_initial_stance(self, state: ClipState, q_init: np.ndarray) -> np.ndarray:
        """Frame 0: default posture, feet at the stance targets, pelvis at standing height (many iterations)."""
        cfg = self.cfg
        b = self.bundle
        self.q.update(q_init.copy())
        Rz = rot_z(state.yaw0)
        for s in SIDES:
            self.ankle[s].set_position_cost(cfg.foot_pos_cost)
            self.ankle[s].set_orientation_cost(cfg.foot_ori_cost)
            self.ankle[s].set_target(self._se3(Rz @ b.r_flat[s], state.foot_pos[s]))
            self.toe[s].set_position_cost(0.0)
            self.toe[s].set_target(self._se3(Rz, state.toe_pos[s]))
            self.knee[s].set_position_cost(0.0)
            self.knee[s].set_target(self._se3(np.eye(3), np.zeros(3)))
            self.palm[s].set_position_cost(0.0)
            self.palm[s].set_orientation_cost(0.0)
            self.palm[s].set_target(self._se3(np.eye(3), np.zeros(3)))
        pel = np.array([*(state.stance_center + Rz[:2, :2] @ np.array([-rs.PELVIS_X_BEHIND_STANCE, 0.0])), state.pelvis_height])
        self.pelvis.set_target(self.mink.SE3.from_rotation_and_translation(self.mink.SO3.from_rpy_radians(0.0, 0.0, state.yaw0), pel))
        qpost = q_init.copy()
        cost = np.zeros(b.nv)
        cost[b.dof_vel_idx] = 1.0
        cost[b.dof_vel_idx[list(rs.LEFT_LEG_DOF_IDX) + list(rs.RIGHT_LEG_DOF_IDX)]] = 0.3
        self.posture.set_cost(cost)
        self.posture.set_target(qpost)
        self.prev.set_target(self.q.q.copy())
        sole_pts = np.concatenate([_flat_sole_points(state.foot_pos[s], state.yaw0) for s in SIDES], axis=0)
        self.com.set_target(np.array([sole_pts[:, 0].mean(), sole_pts[:, 1].mean(), 0.0]))
        limits = self.limits_for_segment(None)
        for _ in range(cfg.first_frame_iters):
            if not self._step(limits):
                break
            self.prev.set_target(self.q.q.copy())
            ef = max(np.linalg.norm(self.ankle[s].compute_error(self.q)[:3]) for s in SIDES)
            ep = abs(float(self.q.get_transform_frame_to_world("pelvis", "body").translation()[2]) - pel[2])
            if ef < 0.5 * cfg.exit_foot and ep < cfg.exit_pelvis:
                break
        return self.q.q.copy()

    def solve_plan(self, plan: SegmentPlan, seg_index: int | None, q_start: np.ndarray, active: tuple[bool, bool],
                   q_before_start: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray]:
        """Solve every frame of ``plan`` starting from ``q_start``; returns (qs (T,nq), residuals (T,4)).

        ``q_before_start`` (the frame before ``q_start``) seeds the per-frame velocity / acceleration limiter."""
        cfg = self.cfg
        b = self.bundle
        mink = self.mink
        self.q.update(q_start.copy())
        v_cap = cfg.frame_vel_cap * DT
        limits = self.limits_for_segment(seg_index)
        T = plan.frames
        qs = np.empty((T, b.nq))
        resid = np.zeros((T, 4))   # palm pos (max active), palm ori, foot pivot, pelvis z
        ratio = cfg.palm_ori_cost / cfg.palm_pos_cost
        qpost = np.zeros(b.nq)
        cost = np.zeros(b.nv)
        for t in range(T):
            for i, s in enumerate(SIDES):
                c = float(plan.palm_cost[t, i])
                self.palm[s].set_position_cost(c)
                self.palm[s].set_orientation_cost(c * ratio)
                if c > 0.0:
                    self.palm[s].set_target(self._se3_quat(plan.palm_quat[t, i], plan.palm_pos[t, i]))
                fp = plan.feet[s]
                R = fp.foot_rot[t]
                if fp.mode == "flat":
                    self.ankle[s].set_position_cost(cfg.foot_pos_cost)
                    self.ankle[s].set_orientation_cost(cfg.foot_ori_cost)
                    self.ankle[s].set_target(self._se3(R @ b.r_flat[s], fp.pivot_pos[t]))
                    self.toe[s].set_position_cost(0.0)
                    self.knee[s].set_position_cost(0.0)
                else:
                    self.ankle[s].set_position_cost(0.0)
                    self.ankle[s].set_orientation_cost(float(fp.ori_cost[t]))
                    self.ankle[s].set_target(self._se3(R @ b.r_flat[s], fp.pivot_pos[t]))
                    self.toe[s].set_position_cost(float(fp.pivot_cost[t]))
                    self.toe[s].set_target(self._se3(R, fp.pivot_pos[t]))
                    if fp.knee_pos is not None:
                        self.knee[s].set_position_cost(float(fp.knee_cost[t]))
                        self.knee[s].set_target(self._se3(np.eye(3), fp.knee_pos[t]))
                    else:
                        self.knee[s].set_position_cost(0.0)
            r, pch, yw = plan.pelvis_rpy[t]
            self.pelvis.set_target(mink.SE3.from_rotation_and_translation(mink.SO3.from_rpy_radians(float(r), float(pch), float(yw)), plan.pelvis_pos[t]))
            qpost[:7] = self.q.q[:7]
            qpost[b.dof_qpos_idx] = plan.posture_q[t]
            cost[:] = 0.0
            cost[b.dof_vel_idx] = plan.posture_cost[t]
            self.posture.set_cost(cost)
            self.posture.set_target(qpost)
            self.prev.set_target(self.q.q.copy())
            sp = plan.support_points[t]
            self.com.set_target(np.array([sp[:, 0].mean(), sp[:, 1].mean(), 0.0]))
            ep = ep_o = ef = ez = 0.0
            for _ in range(cfg.max_iters):
                if not self._step(limits):
                    break
                ep = ep_o = ef = 0.0
                for i, s in enumerate(SIDES):
                    if plan.palm_cost[t, i] > 0.0:
                        e = self.palm[s].compute_error(self.q)
                        ep = max(ep, float(np.linalg.norm(e[:3])))
                        ep_o = max(ep_o, float(np.linalg.norm(e[3:])))
                    fp = plan.feet[s]
                    task = self.ankle[s] if fp.mode == "flat" else self.toe[s]
                    if fp.contact[t] > 0.5:
                        ef = max(ef, float(np.linalg.norm(task.compute_error(self.q)[:3])))
                ez = abs(float(self.q.get_transform_frame_to_world("pelvis", "body").translation()[2]) - float(plan.pelvis_pos[t, 2]))
                if ep < cfg.exit_palm_pos and ep_o < cfg.exit_palm_ori and ef < cfg.exit_foot and ez < cfg.exit_pelvis:
                    break
            # hard per-frame joint-speed cap (uniform scaling of the whole step keeps the task directions consistent):
            # a hand held back by the collision limit must not snap forward.  Accelerations are gated, not clamped:
            # elementwise acceleration clamping can disrupt the coupled foot constraints.
            q_prev = q_start if t == 0 else qs[t - 1]
            dq = self.q.q[b.dof_qpos_idx] - q_prev[b.dof_qpos_idx]
            step_max = float(np.abs(dq).max())
            if step_max > v_cap:
                q_new = q_prev + (v_cap / step_max) * (self.q.q - q_prev)
                q_new[3:7] /= np.linalg.norm(q_new[3:7])
                self.q.update(q_new)
            qs[t] = self.q.q
            resid[t] = [ep, ep_o, ef, ez]
        return qs, resid


# --------------------------------------------------------------------------------------------------
# Kinematic audit and tiers
# --------------------------------------------------------------------------------------------------
def _root_pitch_from_quat(q_wxyz: np.ndarray) -> float:
    R = quat_wxyz_to_mat(q_wxyz)
    return float(math.atan2(-R[2, 0], math.hypot(R[0, 0], R[1, 0])))


def _ang_vel_from_quats(q: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity (T,3) from a wxyz quaternion trajectory (central differences)."""
    q = np.asarray(q, dtype=np.float64)
    dq = np.gradient(q, dt, axis=0)
    w, x, y, z = q.T
    conj = np.stack([w, -x, -y, -z], axis=1)
    # omega = 2 * (dq * q^-1) vector part
    a = dq
    b = conj
    prod = np.stack([
        a[:, 0] * b[:, 0] - a[:, 1] * b[:, 1] - a[:, 2] * b[:, 2] - a[:, 3] * b[:, 3],
        a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0] + a[:, 2] * b[:, 3] - a[:, 3] * b[:, 2],
        a[:, 0] * b[:, 2] - a[:, 1] * b[:, 3] + a[:, 2] * b[:, 0] + a[:, 3] * b[:, 1],
        a[:, 0] * b[:, 3] + a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1] + a[:, 3] * b[:, 0],
    ], axis=1)
    return 2.0 * prod[:, 1:]


@dataclasses.dataclass
class FrameRecord:
    """Per-frame bookkeeping needed by the audit (which plan governs which frame)."""

    plan: SegmentPlan | None      # None for settle frames
    local_t: int
    seg_index: int | None         # reach-segment index (table selection), None for settle; the retract carries the LAST segment's index
    phase: int                    # 0 settle, 1 reach, 2 hold, 3 retract


def fk_trajectory(bundle: ModelBundle, qs: np.ndarray) -> dict[str, np.ndarray]:
    """Palm / foot / knee / CoM kinematics for every frame (no contacts)."""
    import mujoco

    m, d = bundle.model, bundle.data
    T = len(qs)
    out = {
        "palm_pos": np.empty((T, 2, 3)), "palm_quat": np.empty((T, 2, 4)),
        "ankle_pos": np.empty((T, 2, 3)), "toe_pos": np.empty((T, 2, 3)), "knee_pos": np.empty((T, 2, 3)),
        "shoulder_pos": np.empty((T, 2, 3)), "com": np.empty((T, 3)),
    }
    pelvis = bundle.body_id["pelvis"]
    for t in range(T):
        d.qpos[:] = qs[t]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_comPos(m, d)
        for i, s in enumerate(SIDES):
            sid = bundle.site_id[f"{s}_palm_site"]
            out["palm_pos"][t, i] = d.site_xpos[sid]
            out["palm_quat"][t, i] = _quat_from_xmat(d.site_xmat[sid])
            out["ankle_pos"][t, i] = d.xpos[bundle.body_id[f"{s}_ankle_roll_link"]]
            out["toe_pos"][t, i] = d.site_xpos[bundle.site_id[f"{s}_toe_site"]]
            out["knee_pos"][t, i] = d.xpos[bundle.body_id[f"{s}_knee_link"]]
            out["shoulder_pos"][t, i] = d.xpos[bundle.body_id[f"{s}_shoulder_pitch_link"]]
        out["com"][t] = d.subtree_com[pelvis]
    return out


def contact_audit(bundle: ModelBundle, qs: np.ndarray, records: Sequence[FrameRecord], kneel_side_per_frame: Sequence[str]) -> dict[str, Any]:
    """``mj_collision`` per frame: self-collision (minus baseline pairs), robot-table, robot-floor (non-foot)."""
    import mujoco

    m, d = bundle.model, bundle.data
    foot = bundle.foot_geoms["left"] | bundle.foot_geoms["right"]
    tables = [g for g in bundle.table_geoms if g is not None]
    contype_backup = {g: (int(m.geom_contype[g]), int(m.geom_conaffinity[g])) for g in tables}
    self_frames = table_frames = floor_frames = 0
    self_pairs: dict[str, int] = {}
    floor_bodies: dict[str, int] = {}
    table_bodies: dict[str, int] = {}
    T = len(qs)
    for t in range(T):
        rec = records[t]
        active_table = None if rec.seg_index is None or rec.seg_index >= len(bundle.table_geoms) else bundle.table_geoms[rec.seg_index]
        for g in tables:
            on = g == active_table
            m.geom_contype[g] = 1 if on else 0
            m.geom_conaffinity[g] = 1 if on else 0
        d.qpos[:] = qs[t]
        mujoco.mj_kinematics(m, d)
        mujoco.mj_collision(m, d)
        kneel = kneel_side_per_frame[t]
        allowed_floor = set(foot) | (bundle.leg_geoms[kneel] if kneel else set())
        f_self = f_table = f_floor = False
        for i in range(d.ncon):
            c = d.contact[i]
            if c.dist > 0.0:
                continue
            g1, g2 = int(c.geom1), int(c.geom2)
            in1, in2 = g1 in bundle.robot_geoms, g2 in bundle.robot_geoms
            if in1 and in2:
                key = (min(g1, g2), max(g1, g2))
                if key in bundle.baseline_self_pairs:
                    continue
                f_self = True
                name = f"{m.body(m.geom_bodyid[g1]).name}|{m.body(m.geom_bodyid[g2]).name}"
                self_pairs[name] = self_pairs.get(name, 0) + 1
            elif in1 or in2:
                rg, og = (g1, g2) if in1 else (g2, g1)
                if og in tables:
                    f_table = True
                    bn = m.body(m.geom_bodyid[rg]).name
                    table_bodies[bn] = table_bodies.get(bn, 0) + 1
                elif og == bundle.floor_geom:
                    if rg in allowed_floor:
                        continue
                    f_floor = True
                    bn = m.body(m.geom_bodyid[rg]).name
                    floor_bodies[bn] = floor_bodies.get(bn, 0) + 1
        self_frames += f_self
        table_frames += f_table
        floor_frames += f_floor
    for g, (ct, ca) in contype_backup.items():
        m.geom_contype[g] = ct
        m.geom_conaffinity[g] = ca
    return {"self_collision_frames": int(self_frames), "robot_table_contact_frames": int(table_frames),
            "robot_floor_contact_frames": int(floor_frames), "self_collision_pairs": self_pairs, "floor_contact_bodies": floor_bodies,
            "table_contact_bodies": table_bodies}


def audit_trajectory(bundle: ModelBundle, qs: np.ndarray, records: Sequence[FrameRecord], spec: ClipSpec,
                     reach_plans: Sequence[SegmentPlan], seg_bounds: np.ndarray, thr: GateThresholds, fk: dict[str, np.ndarray] | None = None) -> dict[str, Any]:
    fk = fk or fk_trajectory(bundle, qs)
    T = len(qs)
    dofs = qs[:, bundle.dof_qpos_idx]
    qdot = np.gradient(dofs, DT, axis=0)
    qddot = np.gradient(qdot, DT, axis=0)
    lo, hi = bundle.jnt_range[:, 0], bundle.jnt_range[:, 1]
    margin = np.minimum(dofs - lo, hi - dofs)
    ankle_idx = [DOF_NAMES.index("left_ankle_pitch_joint"), DOF_NAMES.index("right_ankle_pitch_joint")]
    ankle_hits = np.any(margin[:, ankle_idx] < thr.ankle_limit_hit_margin_rad, axis=1)
    idx = {n: i for i, n in enumerate(DOF_NAMES)}
    # foot drift against the planned pivot (flat: ankle_roll origin; heel-lift / kneel: toe site while in contact)
    drift = np.zeros((T, 2))
    for t, rec in enumerate(records):
        for i, s in enumerate(SIDES):
            if rec.plan is None:
                continue
            fp = rec.plan.feet[s]
            if fp.contact[rec.local_t] < 0.5:
                continue
            actual = fk["ankle_pos"][t, i] if fp.mode == "flat" else fk["toe_pos"][t, i]
            drift[t, i] = np.linalg.norm(actual - fp.pivot_pos[rec.local_t])
    # CoM margin inside the planned support polygon
    com_margin = np.empty(T)
    stance_pts = np.concatenate([_flat_sole_points(fk["ankle_pos"][0, i], spec.stance.yaw0) for i in range(2)], axis=0)
    for t, rec in enumerate(records):
        pts = stance_pts if rec.plan is None else rec.plan.support_points[rec.local_t]
        com_margin[t] = polygon_margin(fk["com"][t, :2], pts)
    # terminal EE errors per reach segment / active hand
    terminal = []
    worst_pos = worst_ori = 0.0
    for k, plan in enumerate(reach_plans):
        end = int(seg_bounds[k, 1]) - 1
        row = {"segment": k, "hands": {}}
        for i, s in enumerate(SIDES):
            if not spec.active_hands[i]:
                continue
            e_pos = float(np.linalg.norm(fk["palm_pos"][end, i] - plan.target_final_pos[i]))
            e_ori = float(quat_angle_wxyz(fk["palm_quat"][end, i], plan.target_final_quat[i]))
            row["hands"][s] = {"pos_m": e_pos, "ori_rad": e_ori}
            worst_pos, worst_ori = max(worst_pos, e_pos), max(worst_ori, e_ori)
        terminal.append(row)
    kneel_per_frame = [("" if rec.plan is None else next((s for s in SIDES if rec.plan.feet[s].mode in ("kneel", "kneel_up")), "")) for rec in records]
    contacts = contact_audit(bundle, qs, records, kneel_per_frame)
    root_pitch = np.array([_root_pitch_from_quat(q) for q in qs[:, 3:7]])
    knee_z_kneel = None
    if any(kneel_per_frame):
        side = next(k for k in kneel_per_frame if k)
        i = SIDES.index(side)
        hold_frames = [t for t, rec in enumerate(records) if rec.phase == 2 and kneel_per_frame[t]]
        if hold_frames:
            knee_z_kneel = float(np.min(fk["knee_pos"][hold_frames, i, 2]))
    audit = {
        "frames": int(T),
        "duration_s": float(T / FPS),
        "finite": bool(np.isfinite(qs).all()),
        "terminal_ee_pos_error_m": worst_pos,
        "terminal_ee_ori_error_rad": worst_ori,
        "terminal_per_segment": terminal,
        "max_foot_drift_m": float(drift.max()),
        "max_foot_drift_per_foot_m": [float(drift[:, 0].max()), float(drift[:, 1].max())],
        "min_joint_limit_margin_rad": float(margin.min()),
        "limiting_joint": DOF_NAMES[int(np.argmin(margin.min(axis=0)))],
        "max_qdot_rad_s": float(np.abs(qdot).max()),
        "max_qddot_rad_s2": float(np.abs(qddot).max()),
        "min_com_margin_m": float(com_margin.min()),
        "ankle_limit_hit_frames": int(ankle_hits.sum()),
        "ankle_limit_hit_share": float(ankle_hits.mean()),
        "min_pelvis_height_m": float(qs[:, 2].min()),
        "max_pelvis_pitch_rad": float(root_pitch.max()),
        "max_abs_waist_pitch_rad": float(np.abs(dofs[:, idx["waist_pitch_joint"]]).max()),
        "max_abs_waist_yaw_rad": float(np.abs(dofs[:, idx["waist_yaw_joint"]]).max()),
        "max_abs_waist_roll_rad": float(np.abs(dofs[:, idx["waist_roll_joint"]]).max()),
        "max_knee_rad": float(max(dofs[:, idx["left_knee_joint"]].max(), dofs[:, idx["right_knee_joint"]].max())),
        "min_knee_z_kneel_m": knee_z_kneel,
        "palm_z_min_m": float(fk["palm_pos"][:, [i for i in range(2) if spec.active_hands[i]], 2].min()) if any(spec.active_hands) else None,
        **contacts,
    }
    return audit


def classify_tier(audit: dict[str, Any], thr: GateThresholds) -> tuple[str, list[str]]:
    """Four-tier label: core (all strict gates) / recoverable / contact_exploration / boundary."""
    flags: list[str] = []
    if not audit.get("finite", True):
        return "boundary", ["non_finite"]
    if audit["terminal_ee_pos_error_m"] > thr.strict_ee_pos_m:
        flags.append("coarse_ee_position")
    if audit["terminal_ee_ori_error_rad"] > thr.strict_ee_ori_rad:
        flags.append("coarse_ee_orientation")
    if audit["max_foot_drift_m"] > thr.strict_foot_drift_m:
        flags.append("foot_drift")
    if audit["min_joint_limit_margin_rad"] < thr.strict_limit_margin_rad:
        flags.append("joint_limit_boundary")
    if audit["max_qdot_rad_s"] > thr.strict_qdot * (1.0 + 1e-3):
        flags.append("fast_motion")
    if audit["max_qddot_rad_s2"] > thr.strict_qddot:
        flags.append("high_acceleration")
    if audit["min_com_margin_m"] < thr.strict_com_margin_m:
        flags.append("support_boundary")
    if audit["robot_table_contact_frames"] > 0:
        flags.append("table_contact")
    if audit["robot_floor_contact_frames"] > 0:
        flags.append("floor_contact")
    if audit["self_collision_frames"] > 0:
        flags.append("self_collision")
    if audit.get("qp_failures", 0) > 0:
        flags.append("qp_fallback")
    relaxed = (
        audit["terminal_ee_pos_error_m"] <= thr.relaxed_ee_pos_m
        and audit["terminal_ee_ori_error_rad"] <= thr.relaxed_ee_ori_rad
        and audit["max_foot_drift_m"] <= thr.relaxed_foot_drift_m
        and audit["min_joint_limit_margin_rad"] >= thr.relaxed_limit_margin_rad
        and audit["max_qdot_rad_s"] <= thr.relaxed_qdot
        and audit["max_qddot_rad_s2"] <= thr.relaxed_qddot
        and audit["min_com_margin_m"] >= thr.relaxed_com_margin_m
        and audit["self_collision_frames"] == 0
    )
    contact_flags = {"table_contact", "floor_contact"}
    hard = [f for f in flags if f not in contact_flags and f != "qp_fallback"]
    if not hard and not (set(flags) & contact_flags):
        tier = "core"
    elif relaxed and (set(flags) & contact_flags):
        tier = "contact_exploration"
    elif relaxed:
        tier = "recoverable"
    else:
        tier = "boundary"
    return tier, flags


# --------------------------------------------------------------------------------------------------
# Clip generation
# --------------------------------------------------------------------------------------------------
@dataclasses.dataclass
class ClipResult:
    spec: ClipSpec
    qs: np.ndarray
    records: list[FrameRecord]
    reach_plans: list[SegmentPlan]
    retract_plan: SegmentPlan | None
    seg_bounds: np.ndarray
    tables: list[TableBox | None]
    fk: dict[str, np.ndarray]
    audit: dict[str, Any]
    tier: str
    flags: list[str]
    solve_s: float
    retimes: int
    resid: np.ndarray | None = None   # (T,4) per-frame solver residuals: palm pos, palm ori, foot pivot, pelvis z


def _initial_q(bundle: ModelBundle, state: ClipState) -> np.ndarray:
    q = np.zeros(bundle.nq)
    Rz = rot_z(state.yaw0)
    q[0:2] = state.stance_center + Rz[:2, :2] @ np.array([-rs.PELVIS_X_BEHIND_STANCE, 0.0])
    q[2] = state.pelvis_height
    q[3:7] = mat_to_quat_wxyz(Rz)
    q[bundle.dof_qpos_idx] = rs.DEFAULT_DOF_ARRAY
    return q


def _initial_state(spec: ClipSpec) -> ClipState:
    st = spec.stance
    Rz = rot_z(st.yaw0)
    foot_pos = {}
    toe_pos = {}
    for s, x, y in (("left", st.left_x, st.half_width), ("right", st.right_x, -st.half_width)):
        p = Rz @ np.array([x, y, 0.0]) + np.array([0.0, 0.0, rs.SOLE_BELOW_ANKLE])
        foot_pos[s] = p
        toe_pos[s] = p + Rz @ np.array(rs.TOE_LOCAL)
    center = 0.5 * (foot_pos["left"][:2] + foot_pos["right"][:2])
    prior = rs.solve_leg_prior(rs.STANDING_ROOT_Z, 0.0)
    leg = {s: {"hip_pitch": prior.hip_pitch, "knee": prior.knee, "ankle_pitch": prior.ankle_pitch} for s in SIDES}
    return ClipState(yaw0=st.yaw0, foot_pos=foot_pos, toe_pos=toe_pos, stance_center=center, pelvis_height=rs.STANDING_ROOT_Z,
                     pelvis_pitch=0.0, pelvis_yaw_delta=0.0, pelvis_shift=np.zeros(2), waist_pitch=0.0, waist_yaw=0.0, leg_prior=leg,
                     foot_mode={s: "flat" for s in SIDES}, heel_pitch={s: 0.0 for s in SIDES}, kneel_side="")


def _palm_start(bundle: ModelBundle, q: np.ndarray) -> tuple[dict[str, tuple[np.ndarray, np.ndarray]], dict[str, np.ndarray]]:
    import mujoco

    d = bundle.data
    d.qpos[:] = q
    mujoco.mj_kinematics(bundle.model, d)
    palms = {}
    shoulders = {}
    for s in SIDES:
        sid = bundle.site_id[f"{s}_palm_site"]
        palms[s] = (d.site_xpos[sid].copy(), _quat_from_xmat(d.site_xmat[sid]))
        shoulders[s] = d.xpos[bundle.body_id[f"{s}_shoulder_pitch_link"]][:2].copy()
    return palms, shoulders


def generate_clip(spec: ClipSpec, cfg: SolverConfig = SolverConfig(), thr: GateThresholds = GateThresholds(),
                  mjcf_path: Path | str = DEFAULT_MJCF, bundle: ModelBundle | None = None) -> ClipResult:
    """Solve one clip spec end to end (model build, stance, segments, retract, audit, tier)."""
    t0 = time.perf_counter()
    tables = [table_for_segment(seg) for seg in spec.segments]
    if bundle is None:
        bundle = build_model(tables, mjcf_path)
    solver = MinkSolver(bundle, cfg)
    state = _initial_state(spec)
    q0 = solver.solve_initial_stance(state, _initial_q(bundle, state))
    palm_rest, _ = _palm_start(bundle, q0)          # frame-0 (settle) palms = the rest pose the retract phase returns to
    qs_list = [np.repeat(q0[None], SETTLE_FRAMES, axis=0)]
    resid_list = [np.zeros((SETTLE_FRAMES, 4))]
    records: list[FrameRecord] = [FrameRecord(None, t, None, 0) for t in range(SETTLE_FRAMES)]
    reach_plans: list[SegmentPlan] = []
    seg_bounds = []
    q_cur = q0
    frame = SETTLE_FRAMES
    retimes = 0
    for k, seg in enumerate(spec.segments):
        palms, shoulders = _palm_start(bundle, q_cur)
        extra = 1.0
        for attempt in range(cfg.max_retime + 1):
            plan = plan_segment(seg, spec, state, palms, shoulders, cfg, time_scale_extra=extra)
            q_prev_frame = qs_list[-1][-2] if len(qs_list[-1]) >= 2 else None
            qs, resid = solver.solve_plan(plan, k, q_cur, spec.active_hands, q_before_start=q_prev_frame)
            prev_tail = qs_list[-1][-3:] if len(qs_list[-1]) >= 3 else qs_list[-1]
            factor = _retime_factor(prev_tail, qs, resid, plan, bundle, cfg, attempt)
            if factor is None:
                break
            extra *= factor
            retimes += 1
        n_blend = int(round(float(getattr(seg, "blend_s", 0.0)) * FPS))
        if n_blend > 0 and k > 0:
            # Crossfade the preceding pose into the solved re-reach: linear positions
            # and joint angles, slerped root rotation. This smooths the QP
            # transition when a palm path restarts at the measured pose.
            qs = _blend_from_previous(qs, q_cur, n_blend)
        qs_list.append(qs)
        resid_list.append(resid)
        n_reach = plan.reach_frames
        records += [FrameRecord(plan, t, k, 1 if t < n_reach else 2) for t in range(plan.frames)]
        seg_bounds.append((frame, frame + plan.frames))
        frame += plan.frames
        reach_plans.append(plan)
        advance_state(state, seg, plan, retract=False)
        q_cur = qs[-1]
    retract_plan = None
    if any(seg.retract for seg in spec.segments):
        # retract-to-rest after the last segment: same re-timing rule as a reach, solved with the LAST table's collision limit and
        # audited against that table (the slab is still there while the hand withdraws)
        k_last = len(spec.segments) - 1
        palms, shoulders = _palm_start(bundle, q_cur)
        q_prev_frame = qs_list[-1][-2] if len(qs_list[-1]) >= 2 else None
        extra = 1.0
        for attempt in range(cfg.max_retime + 1):
            retract_plan = plan_segment(None, spec, state, palms, shoulders, cfg, time_scale_extra=extra, retract=True, palm_rest=palm_rest)
            qs, resid_r = solver.solve_plan(retract_plan, k_last, q_cur, spec.active_hands, q_before_start=q_prev_frame)
            prev_tail = qs_list[-1][-3:] if len(qs_list[-1]) >= 3 else qs_list[-1]
            factor = _retime_factor(prev_tail, qs, resid_r, retract_plan, bundle, cfg, attempt)
            if factor is None:
                break
            extra *= factor
            retimes += 1
        qs_list.append(qs)
        resid_list.append(resid_r)
        records += [FrameRecord(retract_plan, t, k_last, 3) for t in range(retract_plan.frames)]
        advance_state(state, None, retract_plan, retract=True)
    qs_all = np.concatenate(qs_list, axis=0)
    resid_all = np.concatenate(resid_list, axis=0)
    seg_bounds_arr = np.array(seg_bounds, dtype=np.int64).reshape(-1, 2)
    fk = fk_trajectory(bundle, qs_all)
    audit = audit_trajectory(bundle, qs_all, records, spec, reach_plans, seg_bounds_arr, thr, fk)
    audit["qp_failures"] = int(solver.qp_failures)
    audit["retimes"] = int(retimes)
    reach_mask = np.array([rec.phase == 1 for rec in records])
    audit["max_palm_path_error_m"] = float(resid_all[reach_mask, 0].max()) if reach_mask.any() else 0.0
    audit["mean_palm_path_error_m"] = float(resid_all[reach_mask, 0].mean()) if reach_mask.any() else 0.0
    audit["blocked_frames"] = int((resid_all[reach_mask, 0] > 0.03).sum()) if reach_mask.any() else 0
    tier, flags = classify_tier(audit, thr)
    return ClipResult(spec=spec, qs=qs_all, records=records, reach_plans=reach_plans, retract_plan=retract_plan, seg_bounds=seg_bounds_arr,
                      tables=tables, fk=fk, audit=audit, tier=tier, flags=flags, solve_s=time.perf_counter() - t0, retimes=retimes, resid=resid_all)


# --------------------------------------------------------------------------------------------------
# Bank payload (schema hero_reach_bank_v2), labels, manifest
# --------------------------------------------------------------------------------------------------
BANK_REQUIRED_KEYS: tuple[str, ...] = (
    "schema", "fps", "joint_names", "joint_pos", "joint_vel", "root_pos_w", "root_quat_w",
    "ee_pos_w_left", "ee_quat_w_left", "ee_pos_w_right", "ee_quat_w_right", "active_hands",
    "target_ee_pos_w", "target_ee_quat_w", "target_ee_pos_w_frame", "target_ee_quat_w_frame",
    "base_height_ref", "surface_z_m", "segment_bounds", "phase_id", "foot_contact",
    "body_joint_lower_limit_rad", "body_joint_upper_limit_rad", "labels_json",
)


def build_payload(res: ClipResult, bundle: ModelBundle, labels: dict[str, Any]) -> dict[str, Any]:
    qs = res.qs
    T = len(qs)
    dofs = qs[:, bundle.dof_qpos_idx]
    root_pos = qs[:, 0:3]
    root_quat = qs[:, 3:7]
    joint_vel = np.concatenate([np.gradient(root_pos, DT, axis=0), _ang_vel_from_quats(root_quat, DT), np.gradient(dofs, DT, axis=0)], axis=1)
    S = len(res.reach_plans)
    target_pos = np.stack([p.target_final_pos for p in res.reach_plans], axis=0) if S else np.zeros((0, 2, 3))
    target_quat = np.stack([p.target_final_quat for p in res.reach_plans], axis=0) if S else np.zeros((0, 2, 4))
    tp_frame = np.empty((T, 2, 3))
    tq_frame = np.empty((T, 2, 4))
    base_h = np.empty(T)
    phase = np.empty(T, dtype=np.int8)
    contact = np.ones((T, 2), dtype=np.float32)
    first = res.reach_plans[0] if S else None
    for t, rec in enumerate(res.records):
        phase[t] = rec.phase
        if rec.plan is None:
            base_h[t] = rs.STANDING_ROOT_Z if not res.reach_plans else qs[0, 2]
            if first is not None:
                tp_frame[t] = first.target_pos_frame[0]
                tq_frame[t] = first.target_quat_frame[0]
            else:
                tp_frame[t] = res.fk["palm_pos"][t]
                tq_frame[t] = res.fk["palm_quat"][t]
        elif rec.phase == 3:
            base_h[t] = rec.plan.pelvis_pos[rec.local_t, 2]
            tp_frame[t] = res.fk["palm_pos"][-1]
            tq_frame[t] = res.fk["palm_quat"][-1]
            for i, s in enumerate(SIDES):
                contact[t, i] = rec.plan.feet[s].contact[rec.local_t]
        else:
            base_h[t] = rec.plan.pelvis_pos[rec.local_t, 2]
            tp_frame[t] = rec.plan.target_pos_frame[rec.local_t]
            tq_frame[t] = rec.plan.target_quat_frame[rec.local_t]
            for i, s in enumerate(SIDES):
                contact[t, i] = rec.plan.feet[s].contact[rec.local_t]
    payload = {
        "schema": np.str_(SCHEMA),
        "generator_version": np.str_(GENERATOR_VERSION),
        "clip_id": np.str_(res.spec.clip_id),
        "fps": np.int64(FPS),
        "joint_names": np.array(DOF_NAMES),
        "joint_pos": qs.astype(np.float64),
        "joint_vel": joint_vel.astype(np.float64),
        "root_pos_w": root_pos.astype(np.float64),
        "root_quat_w": root_quat.astype(np.float64),
        "ee_pos_w_left": res.fk["palm_pos"][:, 0].astype(np.float64),
        "ee_quat_w_left": res.fk["palm_quat"][:, 0].astype(np.float64),
        "ee_pos_w_right": res.fk["palm_pos"][:, 1].astype(np.float64),
        "ee_quat_w_right": res.fk["palm_quat"][:, 1].astype(np.float64),
        "ee_body_names": np.array(EE_BODY_NAMES),
        "ee_palm_offset": np.array([PALM_OFFSET["left"], PALM_OFFSET["right"]], dtype=np.float64),
        "active_hands": np.array(res.spec.active_hands, dtype=bool),
        "target_ee_pos_w": target_pos.astype(np.float64),
        "target_ee_quat_w": target_quat.astype(np.float64),
        "target_ee_pos_w_frame": tp_frame.astype(np.float64),
        "target_ee_quat_w_frame": tq_frame.astype(np.float64),
        "base_height_ref": base_h.astype(np.float64),
        "surface_z_m": np.array([seg.surface_z for seg in res.spec.segments], dtype=np.float64),
        "segment_bounds": res.seg_bounds.astype(np.int64),
        "phase_id": phase,
        "foot_contact": contact,
        "com_w": res.fk["com"].astype(np.float64),
        "body_joint_lower_limit_rad": bundle.jnt_range[:, 0].astype(np.float64),
        "body_joint_upper_limit_rad": bundle.jnt_range[:, 1].astype(np.float64),
        "sole_below_ankle_m": np.float64(rs.SOLE_BELOW_ANKLE),
        "labels_json": np.str_(json.dumps(labels, sort_keys=True)),
    }
    return payload


def validate_bank_payload(payload: dict[str, Any]) -> dict[str, Any]:
    """Schema validator for ``hero_reach_bank_v2`` clips (raises ValueError on any violation)."""
    missing = [k for k in BANK_REQUIRED_KEYS if k not in payload]
    if missing:
        raise ValueError(f"bank clip misses keys {missing}")
    if str(payload["schema"]) != SCHEMA:
        raise ValueError(f"schema {payload['schema']!r} != {SCHEMA!r}")
    fps = int(np.asarray(payload["fps"]).reshape(-1)[0])
    if fps != FPS:
        raise ValueError(f"fps {fps} != {FPS}")
    names = tuple(str(v) for v in np.asarray(payload["joint_names"]))
    if names != tuple(DOF_NAMES):
        raise ValueError("joint_names differ from hero_isaacsim.constants.DOF_NAMES")
    jp = np.asarray(payload["joint_pos"], dtype=np.float64)
    if jp.ndim != 2 or jp.shape[1] != 36 or jp.shape[0] < 2:
        raise ValueError(f"joint_pos shape {jp.shape} != (T>=2, 36)")
    T = jp.shape[0]
    S = int(np.asarray(payload["surface_z_m"]).shape[0])
    shapes = {
        "joint_vel": (T, 35), "root_pos_w": (T, 3), "root_quat_w": (T, 4),
        "ee_pos_w_left": (T, 3), "ee_quat_w_left": (T, 4), "ee_pos_w_right": (T, 3), "ee_quat_w_right": (T, 4),
        "active_hands": (2,), "target_ee_pos_w": (S, 2, 3), "target_ee_quat_w": (S, 2, 4),
        "target_ee_pos_w_frame": (T, 2, 3), "target_ee_quat_w_frame": (T, 2, 4), "base_height_ref": (T,),
        "segment_bounds": (S, 2), "phase_id": (T,), "foot_contact": (T, 2),
        "body_joint_lower_limit_rad": (29,), "body_joint_upper_limit_rad": (29,),
    }
    for k, shape in shapes.items():
        v = np.asarray(payload[k])
        if v.shape != shape:
            raise ValueError(f"{k} has shape {v.shape}, expected {shape}")
        if v.dtype.kind in "fiu" and not np.isfinite(v.astype(np.float64)).all():
            raise ValueError(f"{k} contains NaN/Inf")
    if not np.isfinite(jp).all():
        raise ValueError("joint_pos contains NaN/Inf")
    if not np.allclose(jp[:, 0:3], payload["root_pos_w"], atol=1e-9) or not np.allclose(jp[:, 3:7], payload["root_quat_w"], atol=1e-9):
        raise ValueError("joint_pos root columns disagree with root_pos_w / root_quat_w")
    qn = np.linalg.norm(np.asarray(payload["root_quat_w"], dtype=np.float64), axis=1)
    if np.any(np.abs(qn - 1.0) > 1e-6):
        raise ValueError("root_quat_w is not unit")
    for k in ("ee_quat_w_left", "ee_quat_w_right"):
        if np.any(np.abs(np.linalg.norm(np.asarray(payload[k], dtype=np.float64), axis=1) - 1.0) > 1e-6):
            raise ValueError(f"{k} is not unit")
    lo = np.asarray(payload["body_joint_lower_limit_rad"], dtype=np.float64)
    hi = np.asarray(payload["body_joint_upper_limit_rad"], dtype=np.float64)
    if np.any(hi <= lo):
        raise ValueError("joint limits inverted")
    dofs = jp[:, 7:]
    if np.any(dofs < lo - 1e-6) or np.any(dofs > hi + 1e-6):
        raise ValueError("joint_pos exceeds the stored joint limits")
    sb = np.asarray(payload["segment_bounds"], dtype=np.int64)
    if S and (np.any(sb[:, 0] >= sb[:, 1]) or np.any(sb < 0) or np.any(sb > T) or np.any(np.diff(sb[:, 0]) <= 0)):
        raise ValueError("segment_bounds are not increasing frame ranges")
    ph = np.asarray(payload["phase_id"])
    if not set(np.unique(ph)).issubset({0, 1, 2, 3}):
        raise ValueError("phase_id contains unknown phases")
    if not bool(np.asarray(payload["active_hands"]).any()):
        raise ValueError("no active hand")
    labels = json.loads(str(payload["labels_json"]))
    if "tier" not in labels or "audit" not in labels:
        raise ValueError("labels_json lacks tier/audit")
    return {"frames": T, "segments": S, "duration_s": T / FPS, "tier": labels["tier"],
            "max_joint_speed_rad_s": float(np.abs(np.asarray(payload["joint_vel"])[:, 6:]).max())}


def bench_frame_labels(reach_start_frame: int, reach_frames: int, hold_frames: int, retract_start_frame: int | None = None) -> dict[str, Any]:
    """Global frame indices of the benchmark windows of a bench clip (50 Hz, settle frames included).

    ``reach_end_frame`` = first hold frame = ``reach_start_frame + reach_frames``; ``hold_end_frame`` = ``reach_end_frame +
    hold_frames`` (the hold window is ``[reach_end_frame, hold_end_frame)``); ``retract_start_frame`` = first frame of the
    retract-to-rest phase (== ``hold_end_frame`` for a single-segment clip), None when the clip does not retract."""
    start, reach, hold = int(reach_start_frame), int(reach_frames), int(hold_frames)
    if start < 0 or reach < 0 or hold < 0:
        raise ValueError("bench frame counts must be non-negative")
    end = start + reach
    return {"reach_start_frame": start, "reach_end_frame": end, "reach_frames": reach, "hold_frames": hold, "hold_end_frame": end + hold,
            "retract_start_frame": None if retract_start_frame is None else int(retract_start_frame)}


def make_labels(res: ClipResult, cfg: SolverConfig, thr: GateThresholds, bundle: ModelBundle) -> dict[str, Any]:
    spec = res.spec
    segs = []
    for seg, plan, tb in zip(spec.segments, res.reach_plans, res.tables):
        segs.append({
            "index": seg.index, "stratum": seg.stratum, "surface_z": seg.surface_z, "layer": seg.layer,
            "has_table": seg.has_table, "table": None if tb is None else tb.as_dict(),
            "family": seg.base.family, "drop_mode": seg.base.drop_mode, "pelvis_drop": seg.base.pelvis_drop,
            "pelvis_height": seg.base.pelvis_height, "pelvis_pitch": seg.base.pelvis_pitch, "waist_pitch": seg.base.waist_pitch,
            "waist_pitch_fraction": seg.base.waist_pitch_fraction, "waist_yaw": seg.base.waist_yaw,
            "foot_mode": seg.base.foot_mode, "heel_lift_pitch": seg.base.heel_lift_pitch, "kneel_side": seg.base.kneel_side,
            "forced_variant": seg.base.forced_variant, "base_timing": seg.base.base_timing,
            "hold_s": seg.hold_s, "retarget": None if seg.retarget is None else dataclasses.asdict(seg.retarget),
            "retract": seg.retract, "time_scale": seg.time_scale, "reach_frames": plan.reach_frames, "hold_frames": plan.hold_frames,
            "retarget_frame": plan.retarget_frame, "replan": bool(getattr(seg, "replan", False)),
            "blend_frames": int(round(float(getattr(seg, "blend_s", 0.0)) * FPS)),
            "targets": [{"hand": t.hand, "pos": list(t.pos), "yaw": t.yaw, "pitch": t.pitch, "roll": t.roll, "canonical_grasp": t.canonical_grasp,
                         "yaw_clipped": t.yaw_clipped, "approach": t.approach} for t in seg.targets],
        })
    labels = {
        "schema": SCHEMA,
        "generator_version": GENERATOR_VERSION,
        "clip_id": spec.clip_id,
        "profile": spec.profile,
        "seed": spec.seed,
        "hands_mode": spec.hands_mode,
        "active_hands": list(spec.active_hands),
        "stance": dataclasses.asdict(spec.stance),
        "segments": segs,
        "n_segments": len(segs),
        "has_retract": any(s.retract for s in spec.segments),
        "has_retarget": any(s.retarget is not None for s in spec.segments),
        "has_replan": any(getattr(s, "replan", False) for s in spec.segments),
        "n_replan_segments": int(sum(bool(getattr(s, "replan", False)) for s in spec.segments)),
        "hold_total_s": spec.hold_total_s,
        "max_hold_s": max(s.hold_s for s in spec.segments),
        "strata": sorted({s.stratum for s in spec.segments}),
        "foot_modes": sorted({s.base.foot_mode for s in spec.segments}),
        "min_pelvis_height_plan": min(s.base.pelvis_height for s in spec.segments),
        "tier": res.tier,
        "flags": res.flags,
        "training_eligible": res.tier in ("core", "recoverable"),
        "audit": res.audit,
        "solve_s": res.solve_s,
        "frames": int(len(res.qs)),
        "world_frame": "heading frame: pelvis xy at frame 0 = origin, x forward; quaternions wxyz",
        "sole_below_ankle_m": rs.SOLE_BELOW_ANKLE,
        "mjcf": str(bundle.mjcf_path.name),
        "solver_config": cfg.as_dict(),
        "gate_thresholds": thr.as_dict(),
    }
    if spec.bench is not None:
        # hero_bench_v1: one reach segment; reach_end_frame = first hold frame (global index, settle frames included)
        plan0 = res.reach_plans[0]
        hand_i = SIDES.index(spec.bench["hand"])
        end = int(res.seg_bounds[0, 1]) - 1
        labels["bench"] = {
            **spec.bench,
            "settle_frames": SETTLE_FRAMES,
            **bench_frame_labels(int(res.seg_bounds[0, 0]), int(plan0.reach_frames), int(plan0.hold_frames),
                                 retract_start_frame=int(res.seg_bounds[-1, 1]) if res.retract_plan is not None else None),
            "n_frames": int(len(res.qs)),
            "fps": FPS,
            "table": None if res.tables[0] is None else res.tables[0].as_dict(),
            "table_half_width": TABLE_WIDTH / 2.0,
            "terminal_ee_pos_error_m": float(np.linalg.norm(res.fk["palm_pos"][end, hand_i] - plan0.target_final_pos[hand_i])),
            "terminal_ee_ori_error_rad": float(quat_angle_wxyz(res.fk["palm_quat"][end, hand_i], plan0.target_final_quat[hand_i])),
            "ee_pos_error_at_reach_end_m": float(np.linalg.norm(res.fk["palm_pos"][int(res.seg_bounds[0, 0]) + int(plan0.reach_frames), hand_i] - plan0.target_final_pos[hand_i])),
            "ee_pos_error_hold_max_m": float(np.linalg.norm(res.fk["palm_pos"][int(res.seg_bounds[0, 0]) + int(plan0.reach_frames): end + 1, hand_i] - plan0.target_final_pos[hand_i], axis=1).max()),
            "pelvis_xy_frame0": [float(res.qs[0, 0]), float(res.qs[0, 1])],
            "min_pelvis_height_m": float(res.qs[:, 2].min()),
        }
        if labels["has_replan"]:
            # hero_reach_replan_v1: the corrective re-reaches (segment k >= 1) -- switch frame (global) = first frame of the segment, the
            # re-reach length in frames (the replanner timing), the protocol hold, the shifted target and the terminal palm error
            events = []
            for k, (seg, plan) in enumerate(zip(spec.segments, res.reach_plans)):
                if not getattr(seg, "replan", False):
                    continue
                b0, b1 = int(res.seg_bounds[k, 0]), int(res.seg_bounds[k, 1])
                events.append({
                    "segment": k, "switch_frame": b0, "reach_frames": int(plan.reach_frames), "hold_frames": int(plan.hold_frames),
                    "blend_frames": int(round(float(getattr(seg, "blend_s", 0.0)) * FPS)),
                    "reach_s": float(plan.reach_frames / FPS), "target_pos_w": [float(v) for v in plan.target_final_pos[hand_i]],
                    "switch_palm_jump_m": float(np.linalg.norm(res.fk["palm_pos"][b0, hand_i] - res.fk["palm_pos"][b0 - 1, hand_i])) if b0 > 0 else 0.0,
                    "terminal_ee_pos_error_m": float(np.linalg.norm(res.fk["palm_pos"][b1 - 1, hand_i] - plan.target_final_pos[hand_i])),
                })
            labels["bench"]["replan_events"] = events
    if res.retract_plan is not None:
        # bench_retract_share / broad_v2 retract: the retract-to-rest phase after the last segment (phase_id 3 frames)
        rp = res.retract_plan
        labels["retract"] = {
            "start_frame": int(res.seg_bounds[-1, 1]), "frames": int(rp.frames), "reach_frames": int(rp.reach_frames), "hold_frames": int(rp.hold_frames),
            "after_replan": bool(getattr(spec.segments[-1], "replan", False)),
            "final_palm_pos_w": {sd: [float(v) for v in res.fk["palm_pos"][-1, i]] for i, sd in enumerate(SIDES)},
            "rest_palm_pos_w": {sd: [float(v) for v in res.fk["palm_pos"][0, i]] for i, sd in enumerate(SIDES)},          # frame-0 (settle) palm
            "final_palm_to_rest_m": {sd: float(np.linalg.norm(res.fk["palm_pos"][-1, i] - res.fk["palm_pos"][0, i])) for i, sd in enumerate(SIDES) if spec.active_hands[i]},
            "final_palm_to_rest_rad": {sd: float(quat_angle_wxyz(res.fk["palm_quat"][-1, i], res.fk["palm_quat"][0, i])) for i, sd in enumerate(SIDES) if spec.active_hands[i]},
            "max_palm_path_error_m": float(res.resid[int(res.seg_bounds[-1, 1]):, 0].max()) if res.resid is not None else None,
            "final_pelvis_height_m": float(res.qs[-1, 2]),
        }
    return labels


def write_clip(res: ClipResult, bundle: ModelBundle, out_dir: Path, cfg: SolverConfig, thr: GateThresholds) -> dict[str, Any]:
    labels = make_labels(res, cfg, thr, bundle)
    payload = build_payload(res, bundle, labels)
    validate_bank_payload(payload)
    clips = out_dir / "clips"
    lab = out_dir / "labels"
    clips.mkdir(parents=True, exist_ok=True)
    lab.mkdir(parents=True, exist_ok=True)
    npz_path = clips / f"{res.spec.clip_id}.npz"
    np.savez_compressed(npz_path, **payload)
    (lab / f"{res.spec.clip_id}.json").write_text(json.dumps(labels, indent=1, sort_keys=True))
    a = res.audit
    return {
        "clip_id": res.spec.clip_id, "file": str(npz_path.relative_to(out_dir)), "labels": str((lab / f"{res.spec.clip_id}.json").relative_to(out_dir)),
        "sha256": _hash_file(npz_path), "bytes": npz_path.stat().st_size, "frames": int(len(res.qs)), "duration_s": float(len(res.qs) / FPS),
        "tier": res.tier, "flags": res.flags, "hands_mode": res.spec.hands_mode, "strata": labels["strata"], "foot_modes": labels["foot_modes"],
        "families": sorted({s.base.family for s in res.spec.segments}), "n_segments": len(res.spec.segments),
        "n_replan_segments": int(sum(bool(getattr(s, "replan", False)) for s in res.spec.segments)),
        "has_retract": bool(labels["has_retract"]),
        "min_pelvis_height_m": a["min_pelvis_height_m"], "terminal_ee_pos_error_m": a["terminal_ee_pos_error_m"],
        "terminal_ee_ori_error_rad": a["terminal_ee_ori_error_rad"], "max_foot_drift_m": a["max_foot_drift_m"],
        "min_joint_limit_margin_rad": a["min_joint_limit_margin_rad"], "max_qdot_rad_s": a["max_qdot_rad_s"], "max_qddot_rad_s2": a["max_qddot_rad_s2"],
        "min_com_margin_m": a["min_com_margin_m"], "ankle_limit_hit_share": a["ankle_limit_hit_share"], "self_collision_frames": a["self_collision_frames"],
        "robot_table_contact_frames": a["robot_table_contact_frames"], "robot_floor_contact_frames": a["robot_floor_contact_frames"],
        "qp_failures": a["qp_failures"], "retimes": a["retimes"], "solve_s": res.solve_s,
        "bench": labels.get("bench"),
    }


def _git_sha() -> str:
    """Git revision; HERO_GIT_SHA overrides when running without a Git checkout."""
    if os.environ.get("HERO_GIT_SHA"):
        return os.environ["HERO_GIT_SHA"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# --------------------------------------------------------------------------------------------------
# Worker / CLI
# --------------------------------------------------------------------------------------------------
def _worker(args: tuple[dict[str, Any], str, dict[str, Any], dict[str, Any], str]) -> dict[str, Any]:
    spec_d, out_dir, cfg_d, thr_d, mjcf = args
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    spec = rs.spec_from_dict(spec_d)
    cfg = SolverConfig(**cfg_d)
    thr = GateThresholds(**thr_d)
    try:
        tables = [table_for_segment(seg) for seg in spec.segments]
        bundle = build_model(tables, mjcf)
        res = generate_clip(spec, cfg, thr, mjcf, bundle=bundle)
        row = write_clip(res, bundle, Path(out_dir), cfg, thr)
        row["ok"] = True
        return row
    except Exception as e:  # noqa: BLE001
        import traceback

        return {"clip_id": spec.clip_id, "ok": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-2000:]}


_STRING_OVERRIDES: frozenset[str] = frozenset({"bench_edge_ref", "bench_approach", "sampler", "bench_y_side"})
_LIST_OVERRIDES: frozenset[str] = frozenset({"bench_heights", "bench_x_range", "bench_abs_y_range", "bench_z_above_range", "bench_yaw_deg", "bench_pitch_deg",
                                             "bench_roll_deg", "bench_edge_gap_range", "bench_stand_drop_range", "bench_squat_drop_range", "bench_stance_half_width_range",
                                             "bench_time_scale_range"})
_BOOL_OVERRIDES: frozenset[str] = frozenset({"bench_has_table"})
_BOOL_WORDS: dict[str, bool] = {"1": True, "true": True, "yes": True, "on": True, "0": False, "false": False, "no": False, "off": False}


def _parse_overrides(items: Sequence[str]) -> dict[str, Any]:
    """``k=v`` profile knobs: floats, ``segment_count_shares=1:0.4/2:0.35/3:0.25``, lists as ``bench_heights=0.5/0.74/0.88``,
    strings for ``bench_edge_ref`` / ``bench_approach`` / ``bench_y_side`` / ``sampler``, booleans (``true`` / ``false``) for
    ``bench_has_table``."""
    out: dict[str, Any] = {}
    for it in items:
        k, _, v = it.partition("=")
        k, v = k.strip(), v.strip()
        if k == "segment_count_shares":
            out[k] = {int(a): float(b) for a, b in (p.split(":") for p in v.split("/"))}
        elif k in _STRING_OVERRIDES:
            out[k] = v
        elif k in _BOOL_OVERRIDES:
            if v.lower() not in _BOOL_WORDS:
                raise ValueError(f"{k}: expected a boolean (true/false), got {v!r}")
            out[k] = _BOOL_WORDS[v.lower()]
        elif k in _LIST_OVERRIDES or "/" in v:
            out[k] = [float(x) for x in v.split("/") if x.strip()]
        else:
            out[k] = float(v)
    return out


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0])
    ap.add_argument("--n", type=int, required=True)
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--profile", default="broad_v2", choices=sorted(rs.PROFILES))
    ap.add_argument("--strata-weights", default=None, help='e.g. "floor=0.15,very_low=0.2,..."')
    ap.add_argument("--override", action="append", default=[], help="profile knob k=v (behind_share, retarget_share, ...)")
    ap.add_argument("--out-dir", required=True)
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--clip-prefix", default=None)
    ap.add_argument("--mjcf", default=str(DEFAULT_MJCF))
    ap.add_argument("--specs-only", action="store_true", help="write specs.jsonl and exit (no solving)")
    ap.add_argument("--max-iters", type=int, default=None)
    args = ap.parse_args(argv)

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    weights = rs.parse_strata_weights(args.strata_weights) if args.strata_weights else None
    overrides = _parse_overrides(args.override)
    specs = rs.sample_clip_specs(args.n, args.seed, profile=args.profile, strata_weights=weights, clip_prefix=args.clip_prefix, **overrides)
    (out_dir / "specs.jsonl").write_text(rs.specs_to_jsonl(specs))
    summary = rs.summarize_specs(specs)
    (out_dir / "specs_summary.json").write_text(json.dumps(summary, indent=1))
    if args.specs_only:
        print(json.dumps(summary, indent=1))
        return 0
    cfg = SolverConfig(**({"max_iters": args.max_iters} if args.max_iters else {}))
    thr = GateThresholds()
    jobs_args = [(dataclasses.asdict(s), str(out_dir), cfg.as_dict(), thr.as_dict(), args.mjcf) for s in specs]
    t0 = time.time()
    rows: list[dict[str, Any]] = []
    if args.jobs <= 1:
        for i, a in enumerate(jobs_args):
            rows.append(_worker(a))
            r = rows[-1]
            print(f"[{i + 1}/{len(jobs_args)}] {r['clip_id']} {'ok' if r['ok'] else 'FAIL'} {r.get('tier', r.get('error'))} {r.get('solve_s', 0):.1f}s", flush=True)
    else:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(args.jobs) as pool:
            for i, r in enumerate(pool.imap_unordered(_worker, jobs_args)):
                rows.append(r)
                print(f"[{i + 1}/{len(jobs_args)}] {r['clip_id']} {'ok' if r['ok'] else 'FAIL'} {r.get('tier', r.get('error'))} {r.get('solve_s', 0):.1f}s", flush=True)
    rows.sort(key=lambda r: r["clip_id"])
    ok = [r for r in rows if r["ok"]]
    tiers: dict[str, int] = {}
    flags: dict[str, int] = {}
    for r in ok:
        tiers[r["tier"]] = tiers.get(r["tier"], 0) + 1
        for f in r["flags"]:
            flags[f] = flags.get(f, 0) + 1
    solve = np.array([r["solve_s"] for r in ok]) if ok else np.zeros(1)
    frames = np.array([r["frames"] for r in ok]) if ok else np.zeros(1)
    manifest = {
        "schema": SCHEMA,
        "generator_version": GENERATOR_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": _git_sha(),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "mjcf": args.mjcf,
        "mjcf_sha256": _hash_file(Path(args.mjcf)) if Path(args.mjcf).is_file() else None,
        "profile": args.profile,
        "seed": args.seed,
        "n_requested": args.n,
        "n_written": len(ok),
        "n_failed": len(rows) - len(ok),
        "strata_weights": weights or rs.PROFILES[args.profile]["strata_weights"],
        "overrides": overrides,
        "spec_summary": summary,
        "solver_config": cfg.as_dict(),
        "gate_thresholds": thr.as_dict(),
        "tier_counts": tiers,
        "flag_counts": flags,
        "timing": {"wall_s": time.time() - t0, "jobs": args.jobs, "solve_s_p50": float(np.median(solve)), "solve_s_p95": float(np.quantile(solve, 0.95)),
                   "solve_s_per_frame_ms": float(1e3 * solve.sum() / max(1, frames.sum())), "frames_total": int(frames.sum())},
        "clips": rows,
    }
    (out_dir / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    print(json.dumps({k: manifest[k] for k in ("n_written", "n_failed", "tier_counts", "flag_counts", "timing")}, indent=1))
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
