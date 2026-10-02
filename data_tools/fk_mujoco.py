"""MuJoCo forward kinematics for the G1 Dex3 model and NumPy quaternion helpers.

The revision-1.0 MJCF matches the training URDF waist-frame geometry. Finger
joints are held at zero and the 29 robot joints are addressed by name. Virtual
foot-contact points and palm offsets are composed from their parent link poses.
HERO_FK_SCENE_XML can override the default model.

The IK data generator uses a separate fixed-finger MJCF with different
waist geometry. Do not substitute it for the conversion model without checking
pose consistency. All quaternion helpers here use wxyz."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Sequence

import numpy as np

from data_tools.schema import (
    DOF_NAMES,
    EE_BODY_NAMES,
    FOOT_CONTACT_POINT_OFFSET,
    FOOT_CONTACT_POINT_PARENT,
    HOLOSOMA_BODY_NAMES_32,
    PALM_OFFSET,
    WAIST_DOF_IDX,
)

from hero_isaacsim.paths import G1_ASSET_ROOT as _G1_DIR
DEFAULT_SCENE_XML: Path = Path(os.environ.get("HERO_FK_SCENE_XML", _G1_DIR / "g1_29dof_with_hand_rev_1_0.xml")).expanduser().resolve()
#: IK generation scene; its waist axes differ from the training URDF.
LEGACY_DEX3_SCENE_XML: Path = _G1_DIR / "scene_g1_29dof_freebase_fixed_dex3.xml"

NQ = 36  # holosoma joint_pos width: 7 root + 29 dof
NV = 35
ROOT_QPOS = 7

# --------------------------------------------------------------------------------------
# numpy quaternion helpers (wxyz, batched over leading dims)
# --------------------------------------------------------------------------------------


def quat_normalize(q: np.ndarray) -> np.ndarray:
    """Unit-normalise wxyz quaternions (leading dims arbitrary)."""
    q = np.asarray(q, dtype=np.float64)
    return q / np.maximum(np.linalg.norm(q, axis=-1, keepdims=True), 1.0e-12)


def quat_conjugate(q: np.ndarray) -> np.ndarray:
    """Conjugate (== inverse for unit quaternions) of wxyz quaternions."""
    q = np.asarray(q, dtype=np.float64)
    return q * np.array([1.0, -1.0, -1.0, -1.0])


def quat_mul(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Hamilton product ``a * b`` of wxyz quaternions (broadcasting)."""
    a = np.asarray(a, dtype=np.float64)
    b = np.asarray(b, dtype=np.float64)
    aw, ax, ay, az = a[..., 0], a[..., 1], a[..., 2], a[..., 3]
    bw, bx, by, bz = b[..., 0], b[..., 1], b[..., 2], b[..., 3]
    return np.stack(
        [
            aw * bw - ax * bx - ay * by - az * bz,
            aw * bx + ax * bw + ay * bz - az * by,
            aw * by - ax * bz + ay * bw + az * bx,
            aw * bz + ax * by - ay * bx + az * bw,
        ],
        axis=-1,
    )


def quat_to_matrix(q: np.ndarray) -> np.ndarray:
    """Rotation matrices ``(..., 3, 3)`` from unit wxyz quaternions."""
    q = quat_normalize(q)
    w, x, y, z = q[..., 0], q[..., 1], q[..., 2], q[..., 3]
    m = np.empty(q.shape[:-1] + (3, 3), dtype=np.float64)
    m[..., 0, 0] = 1 - 2 * (y * y + z * z)
    m[..., 0, 1] = 2 * (x * y - z * w)
    m[..., 0, 2] = 2 * (x * z + y * w)
    m[..., 1, 0] = 2 * (x * y + z * w)
    m[..., 1, 1] = 1 - 2 * (x * x + z * z)
    m[..., 1, 2] = 2 * (y * z - x * w)
    m[..., 2, 0] = 2 * (x * z - y * w)
    m[..., 2, 1] = 2 * (y * z + x * w)
    m[..., 2, 2] = 1 - 2 * (x * x + y * y)
    return m


def quat_rotate(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """Rotate vectors ``v`` (..., 3) by wxyz quaternions ``q`` (..., 4): ``R(q) v``."""
    return np.einsum("...ij,...j->...i", quat_to_matrix(q), np.asarray(v, dtype=np.float64))


def quat_rotate_inverse(q: np.ndarray, v: np.ndarray) -> np.ndarray:
    """``R(q)^T v`` for wxyz quaternions."""
    return np.einsum("...ji,...j->...i", quat_to_matrix(q), np.asarray(v, dtype=np.float64))


def quat_slerp(q0: np.ndarray, q1: np.ndarray, t: np.ndarray) -> np.ndarray:
    """Shortest-path slerp between wxyz quaternions ``q0``/``q1`` (..., 4) at ``t`` (...,)."""
    q0 = quat_normalize(q0)
    q1 = quat_normalize(q1)
    t = np.asarray(t, dtype=np.float64)[..., None]
    dot = np.sum(q0 * q1, axis=-1, keepdims=True)
    q1 = np.where(dot < 0.0, -q1, q1)
    dot = np.abs(dot)
    dot = np.clip(dot, -1.0, 1.0)
    theta = np.arccos(dot)
    sin_theta = np.sin(theta)
    small = sin_theta < 1.0e-6
    w0 = np.where(small, 1.0 - t, np.sin((1.0 - t) * theta) / np.where(small, 1.0, sin_theta))
    w1 = np.where(small, t, np.sin(t * theta) / np.where(small, 1.0, sin_theta))
    return quat_normalize(w0 * q0 + w1 * q1)


def express_in_frame(p_w: np.ndarray, q_w: np.ndarray, p_ref: np.ndarray, q_ref: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Express world poses in a reference frame (HERO convention).

    ``p_local = R(q_ref)^T (p_w - p_ref)``, ``q_local = q_ref^-1 * q_w`` (all wxyz).
    ``p_ref``/``q_ref`` broadcast against ``p_w``/``q_w`` (e.g. (T, 1, 3) vs (T, K, 3)).
    """
    p_local = quat_rotate_inverse(q_ref, np.asarray(p_w, dtype=np.float64) - np.asarray(p_ref, dtype=np.float64))
    q_local = quat_normalize(quat_mul(quat_conjugate(q_ref), q_w))
    return p_local, q_local


def angular_velocity_world(quat_wxyz: np.ndarray, fps: float) -> np.ndarray:
    """World angular velocity from wxyz quaternion differences, with time on axis 0."""
    q = quat_normalize(quat_wxyz)
    if q.shape[0] < 2:
        return np.zeros(q.shape[:-1] + (3,), dtype=np.float64)
    q_next = np.concatenate([q[1:], q[-1:]], axis=0)
    q_prev = np.concatenate([q[:1], q[:-1]], axis=0)
    dq = quat_mul(q_next, quat_conjugate(q_prev))  # world-frame increment
    dq = dq * np.where(dq[..., :1] < 0.0, -1.0, 1.0)
    vec = dq[..., 1:]
    norm = np.linalg.norm(vec, axis=-1, keepdims=True)
    angle = 2.0 * np.arctan2(norm, np.clip(dq[..., :1], -1.0, 1.0))
    axis = np.where(norm > 1.0e-12, vec / np.maximum(norm, 1.0e-12), 0.0)
    span = np.full(q.shape[:-1] + (1,), 2.0 / fps)
    span[0] = 1.0 / fps
    span[-1] = 1.0 / fps
    return axis * angle / span


def finite_difference(x: np.ndarray, fps: float) -> np.ndarray:
    """Central-difference time derivative along axis 0 (one-sided at the ends)."""
    x = np.asarray(x, dtype=np.float64)
    if x.shape[0] < 2:
        return np.zeros_like(x)
    return np.gradient(x, 1.0 / fps, axis=0)


# --------------------------------------------------------------------------------------
# FK
# --------------------------------------------------------------------------------------


class G1Dex3FK:
    """Batched forward kinematics on the Dex3 scene producing the holosoma 32-body layout.

    All returned quaternions are wxyz; positions in metres, world frame unless stated.
    """

    def __init__(self, scene_xml: str | Path = DEFAULT_SCENE_XML, *, floor_z: float = 0.0):
        import mujoco  # local import: keeps ``data_tools.schema`` importable without MuJoCo

        self._mujoco = mujoco
        self.scene_xml = Path(scene_xml)
        if not self.scene_xml.exists():
            raise FileNotFoundError(f"FK scene not found: {self.scene_xml}")
        self.model = mujoco.MjModel.from_xml_path(str(self.scene_xml))
        self.data = mujoco.MjData(self.model)
        self.floor_z = float(floor_z)
        model = self.model
        if model.nq < NQ or model.njnt < 30:
            raise RuntimeError(f"{self.scene_xml}: nq={model.nq} njnt={model.njnt}; need a free root + the 29 G1 joints")
        if int(model.jnt_type[0]) != int(mujoco.mjtJoint.mjJNT_FREE):
            raise RuntimeError(f"{self.scene_xml}: first joint must be the free root joint")
        if int(model.jnt_qposadr[0]) != 0:
            raise RuntimeError(f"{self.scene_xml}: free root joint must own qpos[0:7]")
        mj_joint_names = [mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_JOINT, i) for i in range(model.njnt)]
        qadr: list[int] = []
        for name in DOF_NAMES:
            if name not in mj_joint_names:
                raise RuntimeError(f"{self.scene_xml}: joint {name!r} missing (joint names must match DOF_NAMES)")
            jid = mj_joint_names.index(name)
            if int(model.jnt_type[jid]) != int(mujoco.mjtJoint.mjJNT_HINGE):
                raise RuntimeError(f"{self.scene_xml}: joint {name!r} is not a hinge")
            qadr.append(int(model.jnt_qposadr[jid]))
        self._dof_qadr = np.asarray(qadr, dtype=np.int64)
        #: names of extra (non-G1-dof) joints held at qpos0 (Dex3 finger joints on the default model)
        self.extra_joint_names: tuple[str, ...] = tuple(n for n in mj_joint_names[1:] if n not in DOF_NAMES)
        self._qpos_template = np.array(model.qpos0, dtype=np.float64)
        self._qpos_template[self._dof_qadr] = 0.0
        self.joint_names: tuple[str, ...] = tuple(DOF_NAMES)
        self.body_names: tuple[str, ...] = HOLOSOMA_BODY_NAMES_32

        # Direct bodies (present in the scene) and derived bodies (fixed offsets).
        direct_slots: list[int] = []
        direct_bids: list[int] = []
        derived: list[tuple[int, int, np.ndarray]] = []  # (slot, parent slot in the direct list, offset)
        parent_slot_of: dict[str, int] = {}
        for slot, name in enumerate(HOLOSOMA_BODY_NAMES_32):
            bid = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if bid >= 0:
                parent_slot_of[name] = len(direct_slots)
                direct_slots.append(slot)
                direct_bids.append(bid)
            elif name in FOOT_CONTACT_POINT_PARENT:
                derived.append((slot, -1, np.asarray(FOOT_CONTACT_POINT_OFFSET, dtype=np.float64)))
            else:
                raise RuntimeError(f"scene lacks body {name!r} and no derivation rule is known")
        self._derived: list[tuple[int, int, np.ndarray]] = []
        for slot, _, offset in derived:
            parent = FOOT_CONTACT_POINT_PARENT[HOLOSOMA_BODY_NAMES_32[slot]]
            self._derived.append((slot, parent_slot_of[parent], offset))
        self._direct_slots = np.asarray(direct_slots, dtype=np.int64)
        self._direct_bids = np.asarray(direct_bids, dtype=np.int64)
        self.derived_body_names: tuple[str, ...] = tuple(HOLOSOMA_BODY_NAMES_32[s] for s, _, _ in self._derived)
        #: slots of bodies that exist in the MJCF (everything except the derived foot contact points)
        self.shared_slots: np.ndarray = self._direct_slots.copy()
        self.derived_slots: np.ndarray = np.asarray([s for s, _, _ in self._derived], dtype=np.int64)

        self._ee_slots = np.asarray([HOLOSOMA_BODY_NAMES_32.index(n) for n in EE_BODY_NAMES], dtype=np.int64)
        self._palm_offset = np.asarray([PALM_OFFSET["left"], PALM_OFFSET["right"]], dtype=np.float64)  # (2, 3)
        self._pelvis_slot = HOLOSOMA_BODY_NAMES_32.index("pelvis")
        self._waist_cols = np.asarray([ROOT_QPOS + i for i in WAIST_DOF_IDX], dtype=np.int64)

    # ------------------------------------------------------------------------------
    @staticmethod
    def _check_joint_pos(joint_pos: np.ndarray) -> np.ndarray:
        jp = np.asarray(joint_pos, dtype=np.float64)
        if jp.ndim != 2 or jp.shape[1] != NQ:
            raise ValueError(f"joint_pos must be (T, {NQ}) [root xyz, root quat wxyz, 29 dof]; got {jp.shape}")
        if not np.isfinite(jp).all():
            raise ValueError("joint_pos contains NaN/Inf")
        return jp

    def body_poses(self, joint_pos: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """FK -> ``body_pos_w (T, 32, 3)``, ``body_quat_w (T, 32, 4) wxyz`` in HOLOSOMA_BODY_NAMES_32 order."""
        jp = self._check_joint_pos(joint_pos)
        jp = jp.copy()
        jp[:, 3:7] = quat_normalize(jp[:, 3:7])
        T = jp.shape[0]
        pos = np.empty((T, 32, 3), dtype=np.float64)
        quat = np.empty((T, 32, 4), dtype=np.float64)
        model, data, mj = self.model, self.data, self._mujoco
        xpos, xquat = data.xpos, data.xquat
        qpos = data.qpos
        qpos[:] = self._qpos_template
        for t in range(T):
            qpos[:ROOT_QPOS] = jp[t, :ROOT_QPOS]
            qpos[self._dof_qadr] = jp[t, ROOT_QPOS:]
            mj.mj_kinematics(model, data)
            pos[t, self._direct_slots] = xpos[self._direct_bids]
            quat[t, self._direct_slots] = xquat[self._direct_bids]
        self.apply_derived(pos, quat)
        return pos, quat

    def apply_derived(self, body_pos_w: np.ndarray, body_quat_w: np.ndarray) -> None:
        """Overwrite the derived slots (foot contact points) in place from their parent bodies."""
        for slot, parent_idx, offset in self._derived:
            pslot = self._direct_slots[parent_idx]
            body_quat_w[:, slot] = body_quat_w[:, pslot]
            body_pos_w[:, slot] = body_pos_w[:, pslot] + quat_rotate(body_quat_w[:, pslot], offset)

    def palm_poses_w(self, body_pos_w: np.ndarray, body_quat_w: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        """Dex3 palm point [left, right] in world from 32-body arrays: ``(T, 2, 3)``, ``(T, 2, 4) wxyz``."""
        q = np.asarray(body_quat_w, dtype=np.float64)[:, self._ee_slots]  # (T, 2, 4)
        p = np.asarray(body_pos_w, dtype=np.float64)[:, self._ee_slots] + quat_rotate(q, self._palm_offset[None])
        return p, quat_normalize(q)

    def fk_all(self, joint_pos: np.ndarray) -> dict[str, np.ndarray]:
        """One FK pass -> ``body_pos_w``, ``body_quat_w``, ``palm_pos_w``, ``palm_quat_w`` (float64, wxyz)."""
        pos, quat = self.body_poses(joint_pos)
        palm_p, palm_q = self.palm_poses_w(pos, quat)
        return {"body_pos_w": pos, "body_quat_w": quat, "palm_pos_w": palm_p, "palm_quat_w": palm_q}

    def zero_waist(self, joint_pos: np.ndarray) -> np.ndarray:
        """Copy of ``joint_pos`` with the three waist joints (yaw, roll, pitch) set to 0."""
        jp = self._check_joint_pos(joint_pos).copy()
        jp[:, self._waist_cols] = 0.0
        return jp

    def palm_in_pelvis(
        self, joint_pos: np.ndarray, *, zero_waist: bool = False, fk: dict[str, np.ndarray] | None = None
    ) -> tuple[np.ndarray, np.ndarray]:
        """Palm point [left, right] in the clip's pelvis frame -> ``(T, 2, 3)``, ``(T, 2, 4) wxyz``.

        ``p = R_pelvis^T (p_palm - p_pelvis)``, ``q = q_pelvis^-1 * q_palm`` with the *full*
        pelvis orientation (HERO ``ref_ee_pos2root_local`` convention, not heading-only).
        ``zero_waist=True`` re-runs FK with the waist joints zeroed (for references with a neutral waist).
        ``fk`` may pass a precomputed :meth:`fk_all` result for the non-zero-waist case.
        """
        if zero_waist:
            fk = self.fk_all(self.zero_waist(joint_pos))
        elif fk is None:
            fk = self.fk_all(joint_pos)
        p_pel = fk["body_pos_w"][:, self._pelvis_slot][:, None]  # (T, 1, 3)
        q_pel = fk["body_quat_w"][:, self._pelvis_slot][:, None]  # (T, 1, 4)
        return express_in_frame(fk["palm_pos_w"], fk["palm_quat_w"], p_pel, q_pel)

    def hero_extension_keys(self, joint_pos: np.ndarray, *, fk: dict[str, np.ndarray] | None = None) -> dict[str, np.ndarray]:
        """``ee_pos_pelvis``, ``ee_quat_pelvis``, ``*_zero_waist`` and ``h_ref`` (float64, wxyz)."""
        jp = self._check_joint_pos(joint_pos)
        if fk is None:
            fk = self.fk_all(jp)
        p, q = self.palm_in_pelvis(jp, fk=fk)
        p0, q0 = self.palm_in_pelvis(jp, zero_waist=True)
        return {
            "ee_pos_pelvis": p,
            "ee_quat_pelvis": q,
            "ee_pos_pelvis_zero_waist": p0,
            "ee_quat_pelvis_zero_waist": q0,
            "h_ref": fk["body_pos_w"][:, self._pelvis_slot, 2] - self.floor_z,
        }

    def body_index(self, names: Sequence[str]) -> np.ndarray:
        """Slots of ``names`` in the 32-body layout."""
        return np.asarray([HOLOSOMA_BODY_NAMES_32.index(n) for n in names], dtype=np.int64)


_FK_CACHE: dict[tuple[str, float], G1Dex3FK] = {}


def get_fk(scene_xml: str | Path = DEFAULT_SCENE_XML, floor_z: float = 0.0) -> G1Dex3FK:
    """Process-local cached :class:`G1Dex3FK` (MjModel construction is the expensive part)."""
    key = (str(Path(scene_xml).resolve()), float(floor_z))
    fk = _FK_CACHE.get(key)
    if fk is None:
        fk = _FK_CACHE[key] = G1Dex3FK(scene_xml, floor_z=floor_z)
    return fk
