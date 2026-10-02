"""Shared numerical observation geometry for exported HERO policies."""
from __future__ import annotations
import numpy as np
from sim2sim.mathutil import quat_apply_inv, quat_conj, quat_mul, rot6d_from_quat
OBJECT_STATE_DIM = 10

def ee_residual_parts(
    root_pos: np.ndarray,
    root_quat: np.ndarray,
    palm_pos_w: np.ndarray,
    palm_quat_w: np.ndarray,
    ref_pos_local: np.ndarray,
    ref_quat_local: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """``hero_delta_ee_parts`` in numpy: ``(dp (K,3), rot6 (K,6), q_diff (K,4))`` for K palm points."""
    root_quat = np.asarray(root_quat, dtype=np.float64)
    p_cur_local = quat_apply_inv(root_quat[None, :], np.asarray(palm_pos_w) - np.asarray(root_pos)[None, :])
    q_cur_local = quat_mul(quat_conj(root_quat)[None, :], np.asarray(palm_quat_w))
    dp = p_cur_local - np.asarray(ref_pos_local)
    q_diff = quat_mul(quat_conj(q_cur_local), np.asarray(ref_quat_local))
    return dp, rot6d_from_quat(q_diff), q_diff


def ee_residual_18(root_pos, root_quat, palm_pos_w, palm_quat_w, ref_pos_local, ref_quat_local) -> np.ndarray:
    """``[dp_L(3), dp_R(3), rot6_L(6), rot6_R(6)]`` (``ee_residual_18`` / ``hero_delta_ee`` layout)."""
    dp, rot6, _ = ee_residual_parts(root_pos, root_quat, palm_pos_w, palm_quat_w, ref_pos_local, ref_quat_local)
    return np.concatenate([dp.reshape(-1), rot6.reshape(-1)]).astype(np.float32)


# ------------------------------------------------------------------------------------------------ object inputs
def object_state_10(root_pos, root_quat, obj_pos_w, obj_quat_w, has_object: bool) -> np.ndarray:
    """Object-state vector: ``[R_root^T (p_obj - p_root) (3) | rot6d(q_root^-1 * q_obj) (6)
    | has_object (1)]`` as float32.

    ``root`` = robot pelvis (Isaac ``robot_root_states[:, 0:3 / 3:7]``), ``obj`` = the SIMULATED box (Isaac
    ``simulator_object_pos_w / _quat_w``), quaternions xyzw.  ``has_object`` False -> all ten
    entries zero, flag included, whatever the poses are (they may be None).  Same geometry as HERO ``h17_obj_pos_b /
    h18_obj_ori_b / h19_has_object_flag``; identity pose -> ``[0 0 0 1 0 0 1 0 0 1]``.
    """
    out = np.zeros(OBJECT_STATE_DIM, dtype=np.float32)
    if not has_object:
        return out
    root_pos = np.asarray(root_pos, dtype=np.float64).reshape(3)
    root_quat = np.asarray(root_quat, dtype=np.float64).reshape(4)
    p_b = quat_apply_inv(root_quat, np.asarray(obj_pos_w, dtype=np.float64).reshape(3) - root_pos)
    rot6 = rot6d_from_quat(quat_mul(quat_conj(root_quat), np.asarray(obj_quat_w, dtype=np.float64).reshape(4)))
    out[0:3] = p_b
    out[3:9] = rot6
    out[9] = 1.0
    return out


