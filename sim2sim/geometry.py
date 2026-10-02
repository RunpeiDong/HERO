"""Collision-geometry samples used to position the demo feet on the floor."""
from typing import Any
import numpy as np
from sim2sim.mathutil import quat_to_mat, wxyz_to_xyzw
_SPHERE_DIRS = np.asarray([[x,y,z] for x in (-1.,0.,1.) for y in (-1.,0.,1.) for z in (-1.,0.,1.) if x or y or z])
_SPHERE_DIRS /= np.linalg.norm(_SPHERE_DIRS, axis=1, keepdims=True)

def _geom_points_local(model: Any, g: int) -> np.ndarray:
    """Point cloud of geom ``g`` in its BODY frame: mesh vertices, box corners, sampled sphere / capsule / cylinder / ellipsoid surface."""
    import mujoco

    gt = int(model.geom_type[g])
    size = np.asarray(model.geom_size[g], dtype=np.float64)
    if gt == mujoco.mjtGeom.mjGEOM_MESH:
        mid = int(model.geom_dataid[g])
        va, vn = int(model.mesh_vertadr[mid]), int(model.mesh_vertnum[mid])
        pts = np.asarray(model.mesh_vert[va : va + vn], dtype=np.float64)
    elif gt == mujoco.mjtGeom.mjGEOM_BOX:
        pts = np.array([[sx * size[0], sy * size[1], sz * size[2]] for sx in (-1, 1) for sy in (-1, 1) for sz in (-1, 1)], dtype=np.float64)
    elif gt == mujoco.mjtGeom.mjGEOM_SPHERE:
        pts = _SPHERE_DIRS * float(size[0])
    elif gt == mujoco.mjtGeom.mjGEOM_CAPSULE:
        ends = np.array([[0.0, 0.0, -size[1]], [0.0, 0.0, size[1]]])
        pts = np.concatenate([e + _SPHERE_DIRS * float(size[0]) for e in ends], axis=0)
    elif gt == mujoco.mjtGeom.mjGEOM_CYLINDER:
        ang = np.linspace(0.0, 2.0 * np.pi, 16, endpoint=False)
        ring = np.stack([np.cos(ang) * size[0], np.sin(ang) * size[0], np.zeros_like(ang)], axis=1)
        pts = np.concatenate([ring + [0.0, 0.0, -size[1]], ring + [0.0, 0.0, size[1]]], axis=0)
    elif gt == mujoco.mjtGeom.mjGEOM_ELLIPSOID:
        pts = _SPHERE_DIRS * size[None, :]
    else:  # planes / hfields never sit on a foot
        return np.zeros((0, 3), dtype=np.float64)
    R = quat_to_mat(wxyz_to_xyzw(np.asarray(model.geom_quat[g], dtype=np.float64)))
    return np.asarray(model.geom_pos[g], dtype=np.float64) + pts @ R.T
