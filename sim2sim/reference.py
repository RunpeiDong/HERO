"""Reference trajectory accessors for native HERO inference.

Joint and body arrays provide canonical G1 poses, velocities, and named palm
references. Quaternions use the runtime XYZW convention."""

from __future__ import annotations

import os
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from hero_isaacsim import constants as HC
from sim2sim.mathutil import express_in_frame, quat_apply, quat_normalize, wxyz_to_xyzw

REQUIRED_KEYS = ("fps", "joint_pos", "joint_vel", "body_pos_w", "body_quat_w", "body_lin_vel_w", "body_ang_vel_w", "body_names", "joint_names")
NUM_FUTURE_FRAMES = 10
FUTURE_STEP = 5
PELVIS_SLOT = HC.HOLOSOMA_BODY_NAMES_32.index(HC.PELVIS_BODY_NAME)  # 0


def source_tag_from_name(name: str) -> str:
    """``bank_a__AMASS_...npz`` -> ``bank_a``; no ``__`` -> ``unknown``."""
    base = os.path.basename(str(name))
    if base.endswith(".npz"):
        base = base[:-4]
    return base.split("__", 1)[0] if "__" in base else "unknown"


def _decode_names(arr) -> list[str]:
    return [x.decode() if isinstance(x, bytes) else str(x) for x in np.asarray(arr).reshape(-1).tolist()]


def _ang_vel_world_wxyz(q: np.ndarray, dt: float) -> np.ndarray:
    """World-frame angular velocity (T, 3) of a WXYZ quaternion trajectory: ``2 * vec(dq * q^-1)`` (central differences)."""
    q = np.asarray(q, dtype=np.float64)
    dq = np.gradient(q, dt, axis=0)
    w, x, y, z = q.T
    conj = np.stack([w, -x, -y, -z], axis=1)
    a, b = dq, conj
    prod = np.stack([
        a[:, 0] * b[:, 0] - a[:, 1] * b[:, 1] - a[:, 2] * b[:, 2] - a[:, 3] * b[:, 3],
        a[:, 0] * b[:, 1] + a[:, 1] * b[:, 0] + a[:, 2] * b[:, 3] - a[:, 3] * b[:, 2],
        a[:, 0] * b[:, 2] - a[:, 1] * b[:, 3] + a[:, 2] * b[:, 0] + a[:, 3] * b[:, 1],
        a[:, 0] * b[:, 3] + a[:, 1] * b[:, 2] - a[:, 2] * b[:, 1] + a[:, 3] * b[:, 0],
    ], axis=1)
    return 2.0 * prod[:, 1:]


class ClipReference:
    """One motion clip; every quaternion exposed is xyzw, positions in metres (floor z = 0)."""

    def __init__(self, path: str | os.PathLike | None = None, *, arrays: Mapping[str, Any] | None = None, name: str | None = None):
        if (path is None) == (arrays is None):
            raise ValueError("give exactly one of path / arrays")
        if path is not None:
            self.path = Path(path)
            self.name = name or self.path.name
            with np.load(self.path, allow_pickle=False) as d:
                self._init_from_mapping(d, set(d.files))
        else:
            self.path = None
            self.name = name or "<arrays>"
            self._init_from_mapping(arrays, set(arrays.keys()))
        self.source_tag = source_tag_from_name(self.name)

    @classmethod
    def from_arrays(
        cls,
        *,
        fps: int,
        joint_pos: np.ndarray,
        body_pos_w: np.ndarray,
        body_quat_w: np.ndarray,
        joint_vel: np.ndarray | None = None,
        body_lin_vel_w: np.ndarray | None = None,
        body_ang_vel_w: np.ndarray | None = None,
        joint_names: Sequence[str] = HC.DOF_NAMES,
        body_names: Sequence[str] = HC.HOLOSOMA_BODY_NAMES_32,
        name: str = "<arrays>",
    ) -> "ClipReference":
        """Build a reference from holosoma-layout arrays (``joint_pos (T, 7+29)`` root xyz + root quat WXYZ + dofs, body
        quaternions WXYZ like the npz contract).  Missing velocities are finite differences (``np.gradient``, like the
        corpus converter); ``joint_vel`` = ``[root lin vel (3) | root ang vel world (3) | dof vel (29)]``."""
        jp = np.asarray(joint_pos, dtype=np.float64)
        T = jp.shape[0]
        dt = 1.0 / float(fps)
        bp = np.asarray(body_pos_w, dtype=np.float64)
        bq = np.asarray(body_quat_w, dtype=np.float64)
        if jv_missing := joint_vel is None:
            root_lin = np.gradient(jp[:, 0:3], dt, axis=0) if T > 1 else np.zeros((T, 3))
            root_ang = _ang_vel_world_wxyz(jp[:, 3:7], dt) if T > 1 else np.zeros((T, 3))
            dof_vel = np.gradient(jp[:, 7:], dt, axis=0) if T > 1 else np.zeros((T, jp.shape[1] - 7))
            joint_vel = np.concatenate([root_lin, root_ang, dof_vel], axis=1)
        if body_lin_vel_w is None:
            body_lin_vel_w = np.gradient(bp, dt, axis=0) if T > 1 else np.zeros_like(bp)
        if body_ang_vel_w is None:
            body_ang_vel_w = np.stack([_ang_vel_world_wxyz(bq[:, b], dt) for b in range(bq.shape[1])], axis=1) if T > 1 else np.zeros_like(bp)
        arrays = {
            "fps": np.asarray(int(fps)),
            "joint_pos": jp,
            "joint_vel": np.asarray(joint_vel, dtype=np.float64),
            "body_pos_w": bp,
            "body_quat_w": bq,
            "body_lin_vel_w": np.asarray(body_lin_vel_w, dtype=np.float64),
            "body_ang_vel_w": np.asarray(body_ang_vel_w, dtype=np.float64),
            "body_names": np.asarray(list(body_names)),
            "joint_names": np.asarray(list(joint_names)),
        }
        del jv_missing
        return cls(arrays=arrays, name=name)

    def _init_from_mapping(self, d: Mapping[str, Any], keys: set[str]) -> None:
        missing = [k for k in REQUIRED_KEYS if k not in keys]
        if missing:
            raise ValueError(f"{self.name}: missing holosoma keys {missing}")
        self.fps = int(np.asarray(d["fps"]).reshape(-1)[0])
        joint_names = _decode_names(d["joint_names"])
        body_names = _decode_names(d["body_names"])
        jidx = [joint_names.index(n) for n in HC.DOF_NAMES]
        bidx = [body_names.index(n) for n in HC.HOLOSOMA_BODY_NAMES_32]
        jp = np.asarray(d["joint_pos"], dtype=np.float64)
        jv = np.asarray(d["joint_vel"], dtype=np.float64)
        if jp.shape[1] != len(joint_names) + 7 or jv.shape[1] != len(joint_names) + 6:
            raise ValueError(f"{self.name}: joint_pos/joint_vel widths {jp.shape[1]}/{jv.shape[1]} != {len(joint_names)}+7/+6")
        self.joint_pos = np.ascontiguousarray(jp[:, 7:][:, jidx])
        self.joint_vel = np.ascontiguousarray(jv[:, 6:][:, jidx])
        self.root_pos_from_joint_pos = np.ascontiguousarray(jp[:, 0:3])
        self.root_quat_from_joint_pos = quat_normalize(wxyz_to_xyzw(jp[:, 3:7]))
        self.body_pos_w = np.ascontiguousarray(np.asarray(d["body_pos_w"], dtype=np.float64)[:, bidx])
        self.body_quat_w = np.ascontiguousarray(quat_normalize(wxyz_to_xyzw(np.asarray(d["body_quat_w"], dtype=np.float64)[:, bidx])))
        self.body_lin_vel_w = np.ascontiguousarray(np.asarray(d["body_lin_vel_w"], dtype=np.float64)[:, bidx])
        self.body_ang_vel_w = np.ascontiguousarray(np.asarray(d["body_ang_vel_w"], dtype=np.float64)[:, bidx])
        self.ee_pos_pelvis = np.asarray(d["ee_pos_pelvis"], dtype=np.float64) if "ee_pos_pelvis" in keys else None
        self.ee_quat_pelvis = wxyz_to_xyzw(np.asarray(d["ee_quat_pelvis"], dtype=np.float64)) if "ee_quat_pelvis" in keys else None
        self.npz_source_tag = str(d["source_tag"]) if "source_tag" in keys else None
        # Optional object track: centre position, XYZW orientation, and box size.
        self.has_object = bool(np.asarray(d["has_object"]).reshape(-1)[0]) if "has_object" in keys else False
        self.object_pos_w = self.object_quat_w = self.box_size = None
        if self.has_object and "object_pos_w" in keys and "object_quat_w" in keys:
            op = np.asarray(d["object_pos_w"], dtype=np.float64)
            if op.ndim == 2 and op.shape[1] == 3:
                self.object_pos_w = np.ascontiguousarray(op)
                self.object_quat_w = np.ascontiguousarray(quat_normalize(wxyz_to_xyzw(np.asarray(d["object_quat_w"], dtype=np.float64))))
                self.box_size = np.asarray(d["box_size"], dtype=np.float64).reshape(3) if "box_size" in keys else None
            else:
                self.has_object = False
        else:
            self.has_object = False
        self.T = int(self.joint_pos.shape[0])
        if self.T < 2:
            raise ValueError(f"{self.name}: clip has {self.T} frame(s)")
        self.root_pos_w = self.body_pos_w[:, PELVIS_SLOT]
        self.root_quat_w = self.body_quat_w[:, PELVIS_SLOT]
        self.root_lin_vel_w = self.body_lin_vel_w[:, PELVIS_SLOT]
        self.root_ang_vel_w = self.body_ang_vel_w[:, PELVIS_SLOT]
        self._ee_slots = np.asarray([HC.HOLOSOMA_BODY_NAMES_32.index(n) for n in HC.EE_BODY_NAMES], dtype=np.int64)
        self._palm_offset = np.asarray([HC.PALM_OFFSET[s] for s in HC.EE_SIDES], dtype=np.float64)

    def padded(self, extra_steps: int) -> "ClipReference":
        """Copy with the last frame repeated ``extra_steps`` times and zero velocities on the pad.
        The clip name and source tag are preserved;
        ``T_original`` records the unpadded length."""
        n = int(extra_steps)
        if n <= 0:
            return self
        new = ClipReference.__new__(ClipReference)
        new.__dict__.update(self.__dict__)

        def pad(a: np.ndarray, zero: bool) -> np.ndarray:
            tail = np.zeros((n,) + a.shape[1:], dtype=a.dtype) if zero else np.repeat(a[-1:], n, axis=0)
            return np.ascontiguousarray(np.concatenate([a, tail], axis=0))

        new.joint_pos = pad(self.joint_pos, False)
        new.joint_vel = pad(self.joint_vel, True)
        new.root_pos_from_joint_pos = pad(self.root_pos_from_joint_pos, False)
        new.root_quat_from_joint_pos = pad(self.root_quat_from_joint_pos, False)
        new.body_pos_w = pad(self.body_pos_w, False)
        new.body_quat_w = pad(self.body_quat_w, False)
        new.body_lin_vel_w = pad(self.body_lin_vel_w, True)
        new.body_ang_vel_w = pad(self.body_ang_vel_w, True)
        if self.ee_pos_pelvis is not None:
            new.ee_pos_pelvis = pad(self.ee_pos_pelvis, False)
        if self.ee_quat_pelvis is not None:
            new.ee_quat_pelvis = pad(self.ee_quat_pelvis, False)
        if getattr(self, "object_pos_w", None) is not None:
            new.object_pos_w = pad(self.object_pos_w, False)
            new.object_quat_w = pad(self.object_quat_w, False)
        if getattr(self, "terrain_dz", None) is not None:
            new.terrain_dz = pad(self.terrain_dz, False)
        new.T = self.T + n
        new.T_original = getattr(self, "T_original", self.T)
        new.root_pos_w = new.body_pos_w[:, PELVIS_SLOT]
        new.root_quat_w = new.body_quat_w[:, PELVIS_SLOT]
        new.root_lin_vel_w = new.body_lin_vel_w[:, PELVIS_SLOT]
        new.root_ang_vel_w = new.body_ang_vel_w[:, PELVIS_SLOT]
        return new

    def shifted_z(self, dz: "float | np.ndarray") -> "ClipReference":
        """Copy with every world Z shifted by scalar ``dz`` or a ``(T,)`` per-frame array.  Shifted: ``body_pos_w`` (hence
        the root / palm / EE world positions), ``root_pos_from_joint_pos``, ``object_pos_w``.  NOT shifted: joints, velocities (the
        training shift moves positions only), the pelvis-local ``ee_pos_pelvis`` / ``ee_quat_pelvis``.  A zero shift returns ``self``
        (bit-identical, no copy); ``terrain_dz`` records the applied shift on the copy."""
        dz_arr = np.asarray(dz, dtype=np.float64)
        if dz_arr.ndim == 0:
            if float(dz_arr) == 0.0:
                return self
            col = np.full((self.T, 1), float(dz_arr), dtype=np.float64)
        elif dz_arr.shape == (self.T,):
            if not np.any(dz_arr):
                return self
            col = dz_arr.reshape(self.T, 1)
        else:
            raise ValueError(f"{self.name}: shifted_z expects a scalar or a ({self.T},) array, got shape {dz_arr.shape}")
        if not np.all(np.isfinite(col)):
            raise ValueError(f"{self.name}: shifted_z got a non-finite shift")
        new = ClipReference.__new__(ClipReference)
        new.__dict__.update(self.__dict__)
        bp = np.array(self.body_pos_w, dtype=np.float64, copy=True)
        bp[:, :, 2] += col
        new.body_pos_w = np.ascontiguousarray(bp)
        rp = np.array(self.root_pos_from_joint_pos, dtype=np.float64, copy=True)
        rp[:, 2] += col[:, 0]
        new.root_pos_from_joint_pos = np.ascontiguousarray(rp)
        if getattr(self, "object_pos_w", None) is not None:
            op = np.array(self.object_pos_w, dtype=np.float64, copy=True)
            op[:, 2] += col[:, 0]
            new.object_pos_w = np.ascontiguousarray(op)
        new.root_pos_w = new.body_pos_w[:, PELVIS_SLOT]
        new.root_quat_w = new.body_quat_w[:, PELVIS_SLOT]
        new.root_lin_vel_w = new.body_lin_vel_w[:, PELVIS_SLOT]
        new.root_ang_vel_w = new.body_ang_vel_w[:, PELVIS_SLOT]
        new.terrain_dz = col[:, 0].copy()
        return new

    # ------------------------------------------------------------------------------------------ frames
    def clamp(self, t: int) -> int:
        return int(min(max(int(t), 0), self.T - 1))

    def future_indices(self, t: int, num_frames: int = NUM_FUTURE_FRAMES, step: int = FUTURE_STEP) -> np.ndarray:
        idx = int(t) + np.arange(num_frames, dtype=np.int64) * int(step)
        return np.minimum(idx, self.T - 1)

    def future_joint_pos(self, t: int) -> np.ndarray:
        return self.joint_pos[self.future_indices(t)]  # (10, 29) RAW (no default subtraction)

    def future_joint_vel(self, t: int) -> np.ndarray:
        return self.joint_vel[self.future_indices(t)]

    def future_root_quat(self, t: int) -> np.ndarray:
        return self.root_quat_w[self.future_indices(t)]  # (10, 4) xyzw

    def future_object_pose(self, t: int, num_frames: int = NUM_FUTURE_FRAMES, step: int = FUTURE_STEP) -> tuple[np.ndarray, np.ndarray] | None:
        """Reference object centre / orientation (xyzw) at ``clamp(t + k*step, T-1)``, k < num_frames -> ``((F, 3), (F, 4))``;
        None when the clip has no object track (``has_object`` False).  Same indices as ``future_joint_pos``."""
        if self.object_pos_w is None:
            return None
        idx = self.future_indices(t, num_frames, step)
        return self.object_pos_w[idx], self.object_quat_w[idx]

    def body_slot(self, name: str) -> int:
        return HC.HOLOSOMA_BODY_NAMES_32.index(name)

    def body_pos_by_name(self, t: int, names: Sequence[str]) -> np.ndarray:
        return self.body_pos_w[self.clamp(t), [self.body_slot(n) for n in names]]

    def palm_pose_w(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Clip palm point [left, right] in world: ``((2, 3), (2, 4) xyzw)`` (wrist_yaw pose + PALM_OFFSET)."""
        t = self.clamp(t)
        p = self.body_pos_w[t, self._ee_slots]
        q = self.body_quat_w[t, self._ee_slots]
        return p + quat_apply(q, self._palm_offset), q

    def palm_pose_own_pelvis(self, t: int) -> tuple[np.ndarray, np.ndarray]:
        """Clip palm point in the clip's OWN pelvis frame (full orientation) -- HERO geometry."""
        t = self.clamp(t)
        p_w, q_w = self.palm_pose_w(t)
        return express_in_frame(p_w, q_w, self.root_pos_w[t][None], self.root_quat_w[t][None])

    def summary(self) -> dict:
        return {
            "name": self.name,
            "source_tag": self.source_tag,
            "fps": self.fps,
            "frames": self.T,
            "duration_s": self.T / self.fps,
            "root_z0": float(self.root_pos_w[0, 2]),
            "has_object": bool(self.has_object),
            "box_size": (self.box_size.tolist() if self.box_size is not None else None),
        }


def discover_clips(motion_dir: str | Sequence[str], pattern: str = "*.npz", max_clips: int | None = None) -> list[Path]:
    """Expand comma-separated directories or ``.npz`` paths in the given order.
    Each directory is expanded with a sorted glob of ``pattern``."""
    entries = [e.strip() for e in motion_dir.split(",")] if isinstance(motion_dir, str) else [str(e) for e in motion_dir]
    found: list[Path] = []
    for entry in entries:
        if not entry:
            continue
        p = Path(os.path.expanduser(entry))
        if p.is_file() and p.suffix == ".npz":
            found.append(p)
        elif p.is_dir():
            found.extend(sorted(p.glob(pattern)))
        else:
            raise FileNotFoundError(f"motion entry {entry!r} is neither an .npz file nor a directory")
    if max_clips is not None:
        found = found[: int(max_clips)]
    return found


__all__ = ["ClipReference", "FUTURE_STEP", "NUM_FUTURE_FRAMES", "PELVIS_SLOT", "REQUIRED_KEYS", "discover_clips", "source_tag_from_name"]
