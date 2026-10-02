"""Per-step tracking metrics of the benchmark rollouts (the same definitions as the Isaac Sim fixed-horizon harness).

* ``ee_local_*_cm``   = ``|dp|`` of the own-pelvis palm residual (robot palm in the ROBOT pelvis frame minus clip palm
                       in the CLIP pelvis frame; :func:`sim2sim.observation_math.ee_residual_parts`), cm; mean over hands.
* ``ee_rot_*_deg``    = angle of the same residual quaternion ``q_cur_local^-1 q_ref_local``, deg.
* ``ee_global_*_cm``  = ``|p_palm_robot_w - p_palm_clip_w|``, cm.
* ``ee_rot_global_*_deg`` = angle of ``q_palm_robot_w^-1 q_palm_clip_w`` -- the palm orientation error against the
                       world-fixed goal orientation (frame-independent: what the HERO paper's "orientation error" measures;
                       ``ee_rot_deg`` above is taken between the two PELVIS frames and therefore also contains the
                       robot-vs-clip pelvis orientation mismatch).  NaN when the writer has no world reference quaternion.
* ``anchor_pos_cm`` / ``anchor_xy_cm`` = ``|p_ref_pelvis - p_robot_pelvis|`` (3D / xy), cm.
* ``base_height_cm``  = ``|z_robot_root - z_ref_root|``, cm.
* ``root_yaw_err_deg`` = |yaw(q_robot_root^-1 q_ref_root)| -- heading error against the reference root.
* ``joint_upper_rad``   = mean ``|q - q_ref|`` over the 14 arm joints; ``joint_upper17_rad`` over waist 3 + arms 14.

Odometry error columns (:data:`ODOM_METRIC_KEYS`, written ONLY when a :mod:`sim2sim.bench.odometry` estimator runs in the
rollout; the estimate the policy observed at control step ``k`` (pre-physics) is recorded in the step's column):

* ``odom_xy_err_cm`` / ``odom_pos_err_cm`` = ``|p_est - p_true|`` (xy / 3D) of the root position estimate;
* ``odom_yaw_err_deg`` = SIGNED wrapped ``yaw_est - yaw_true``;
* ``odom_z_err_cm``    = SIGNED ``z_est - z_true``;
* ``odom_vel_err_cm_s`` = ``|v_est_w - v_true_w|`` of the root linear velocity estimate;
* ``odom_stance_feet`` = number of feet the estimator held in stance (NaN for the LiDAR-inertial model, which has no stance notion).

Full-body keypoint columns (:data:`KEYPOINT_METRIC_KEYS`, the tail of :data:`METRIC_KEYS`; NaN when the caller passes no body
arrays; the reference is the clip placed where the episode started, NO per-step alignment, so drift accumulates in the
``global`` columns):

* ``kp_global_pos_cm``  = mean over the 32 tracked bodies (``HOLOSOMA_BODY_NAMES_32`` order) of ``|p_robot_w - p_ref_w|``, cm;
* ``kp_global_rot_deg`` = mean geodesic angle of ``q_robot_w^-1 q_ref_w`` (xyzw), deg;
* ``kp_local_pos_cm`` / ``kp_local_rot_deg`` = the same errors after expressing robot and reference bodies in their OWN heading-aligned
  pelvis frame (position relative to the pelvis rotated by the inverse pelvis YAW only, ``q = yaw(q_pelvis)^-1 q_body``): pelvis pitch /
  roll errors stay in, the root position / heading drift is removed;
* ``kp14_global_pos_mm`` / ``kp14_local_pos_mm`` = the two position errors over the 14 links of the SONIC evaluation protocol
  (:data:`SONIC14_BODY_NAMES`), mm -- ``kp14_local_pos_mm`` is the SONIC paper's MPJPE-L (root-relative mean per-joint position error);
* ``kp_group_<feet|hands|head|pelvis|legs|arms>_global_pos_cm`` = world-frame position error of the :data:`KEYPOINT_GROUPS` bodies, cm;
* ``sonic_fail`` = 0 / 1: the SONIC failure rule (``|z_root - z_root_ref| > 0.25 m`` OR any hand body ``|z - z_ref| > 0.25 m``,
  :data:`SONIC_FAIL_DZ_M`); the per-clip ``sonic_success`` (never failed inside the valid window) is aggregated by :mod:`sim2sim.bench.series`.
"""

from __future__ import annotations

import math

import numpy as np

from hero_isaacsim import constants as HC
from sim2sim.mathutil import quat_angle, quat_apply_inv, quat_conj, quat_mul, yaw_quat
from sim2sim.observation_math import ee_residual_parts

RAD2DEG = 180.0 / np.pi
#: The 14 evaluation links of the SONIC protocol (arXiv 2511.07820 reports MPJPE-L over "14 body links"): pelvis, hip_roll + knee +
#: ankle_roll (x2), torso, shoulder_roll + elbow + wrist_yaw (x2) -- the tracked bodies of the whole-body tracking recipe.
SONIC14_BODY_NAMES: tuple[str, ...] = (
    "pelvis",
    "left_hip_roll_link",
    "left_knee_link",
    "left_ankle_roll_link",
    "right_hip_roll_link",
    "right_knee_link",
    "right_ankle_roll_link",
    "torso_link",
    "left_shoulder_roll_link",
    "left_elbow_link",
    "left_wrist_yaw_link",
    "right_shoulder_roll_link",
    "right_elbow_link",
    "right_wrist_yaw_link",
)
#: Body groups of the ``kp_group_<name>_global_pos_cm`` columns.  The G1's 32 tracked bodies contain no head and no palm body: ``head`` =
#: the torso top body ``torso_link``; ``hands`` = the wrist_yaw links the Dex3 palms are fixed to (the palm POINT error is ``ee_global_*``);
#: ``feet`` = the ankle_roll links (the sole bodies); ``legs`` / ``arms`` = the real links of the chains (the massless
#: ``*_foot_contact_point`` bodies count in the 32-body means only).
KEYPOINT_GROUPS: dict[str, tuple[str, ...]] = {
    "feet": ("left_ankle_roll_link", "right_ankle_roll_link"),
    "hands": ("left_wrist_yaw_link", "right_wrist_yaw_link"),
    "head": ("torso_link",),
    "pelvis": ("pelvis",),
    "legs": (
        "left_hip_pitch_link", "left_hip_roll_link", "left_hip_yaw_link", "left_knee_link", "left_ankle_pitch_link", "left_ankle_roll_link",
        "right_hip_pitch_link", "right_hip_roll_link", "right_hip_yaw_link", "right_knee_link", "right_ankle_pitch_link", "right_ankle_roll_link",
    ),
    "arms": (
        "left_shoulder_pitch_link", "left_shoulder_roll_link", "left_shoulder_yaw_link", "left_elbow_link", "left_wrist_roll_link", "left_wrist_pitch_link", "left_wrist_yaw_link",
        "right_shoulder_pitch_link", "right_shoulder_roll_link", "right_shoulder_yaw_link", "right_elbow_link", "right_wrist_roll_link", "right_wrist_pitch_link", "right_wrist_yaw_link",
    ),
}
#: SONIC failure threshold (m): root height or a hand height off the reference by more than this fails the step (module docstring).
SONIC_FAIL_DZ_M: float = 0.25
#: Full-body keypoint columns (module docstring); the tail of :data:`METRIC_KEYS`, always returned by :func:`compute_metrics` (NaN without bodies).
KEYPOINT_METRIC_KEYS: tuple[str, ...] = (
    "kp_global_pos_cm",
    "kp_global_rot_deg",
    "kp_local_pos_cm",
    "kp_local_rot_deg",
    "kp14_global_pos_mm",
    "kp14_local_pos_mm",
    "kp_group_feet_global_pos_cm",
    "kp_group_hands_global_pos_cm",
    "kp_group_head_global_pos_cm",
    "kp_group_pelvis_global_pos_cm",
    "kp_group_legs_global_pos_cm",
    "kp_group_arms_global_pos_cm",
    "sonic_fail",
)
METRIC_KEYS: tuple[str, ...] = (
    "ee_local_cm",
    "ee_local_left_cm",
    "ee_local_right_cm",
    "ee_global_cm",
    "ee_global_left_cm",
    "ee_global_right_cm",
    "ee_rot_deg",
    "ee_rot_left_deg",
    "ee_rot_right_deg",
    "ee_rot_global_deg",
    "ee_rot_global_left_deg",
    "ee_rot_global_right_deg",
    "anchor_pos_cm",
    "anchor_xy_cm",
    "base_height_cm",
    "root_yaw_err_deg",
    "joint_upper_rad",
    "joint_upper17_rad",
) + KEYPOINT_METRIC_KEYS  # append-only contract: the keypoint columns follow the 18 palm / root / joint columns
#: Odometry error columns (module docstring); present in a series only when the estimator ran (``--odom so``).
ODOM_METRIC_KEYS: tuple[str, ...] = (
    "odom_xy_err_cm",
    "odom_pos_err_cm",
    "odom_yaw_err_deg",
    "odom_z_err_cm",
    "odom_vel_err_cm_s",
    "odom_stance_feet",
)
_ARM = np.asarray(HC.ARM_DOF_IDX, dtype=np.int64)
_UPPER17 = np.asarray(HC.UPPER_REF_DOF_IDX, dtype=np.int64)
_NUM_BODIES = len(HC.HOLOSOMA_BODY_NAMES_32)
_KP_PELVIS = HC.HOLOSOMA_BODY_NAMES_32.index(HC.PELVIS_BODY_NAME)
_KP_SLOTS14 = np.asarray([HC.HOLOSOMA_BODY_NAMES_32.index(n) for n in SONIC14_BODY_NAMES], dtype=np.int64)
_KP_GROUP_SLOTS = {g: np.asarray([HC.HOLOSOMA_BODY_NAMES_32.index(n) for n in names], dtype=np.int64) for g, names in KEYPOINT_GROUPS.items()}
_KP_HANDS = _KP_GROUP_SLOTS["hands"]
assert len(SONIC14_BODY_NAMES) == 14 and len(set(SONIC14_BODY_NAMES)) == 14
assert tuple(f"kp_group_{g}_global_pos_cm" for g in KEYPOINT_GROUPS) == KEYPOINT_METRIC_KEYS[6:12]


# ---- fused per-step kernel constants: the straightforward formulation (heading_frame_poses x2 via np.cross-based quat_apply + two separate
# quat_mul / quat_angle passes + eight fancy-indexed means) costs about 18 % of a 20-sub-step plant step; the kernel below computes the SAME
# numbers (pinned to 1e-9 by the tests) in about a quarter of the time ----------------
def _rel_quat_tensor() -> np.ndarray:
    """``(16, 4)``: ``conj(a) * b`` (xyzw) as a LINEAR map of the outer product ``a_i b_j`` (row ``4 i + j``), built from ``quat_mul`` on the
    basis so the sign pattern is the library's, not hand-typed: ``rel = (a[:, :, None] * b[:, None, :]).reshape(-1, 16) @ C``."""
    eye = np.eye(4)
    c = np.zeros((4, 4, 4))
    for i in range(4):
        for j in range(4):
            c[i, j] = quat_mul(quat_conj(eye[i]), eye[j])
    return c.reshape(16, 4)


def _mean_weights(slot_sets: list[np.ndarray]) -> np.ndarray:
    """``(len(slot_sets), 32)`` row-stochastic weights: ``W @ x`` == the mean of ``x`` over each body set (one matmul instead of eight fancy-indexed means)."""
    w = np.zeros((len(slot_sets), _NUM_BODIES))
    for r, slots in enumerate(slot_sets):
        w[r, slots] = 1.0 / len(slots)
    return w


_REL_QUAT_C16 = _rel_quat_tensor()
#: rows: kp14, then the KEYPOINT_GROUPS in order (feet, hands, head, pelvis, legs, arms)
_KP_MEAN_W = _mean_weights([_KP_SLOTS14, *_KP_GROUP_SLOTS.values()])
_KP_MEAN_ROWS: tuple[str, ...] = ("kp14", *KEYPOINT_GROUPS)


def _yaw_of(q: np.ndarray) -> float:
    """Heading (rad) of ONE xyzw quaternion -- the ``yaw_quat`` formula on Python scalars (a numpy round trip costs 10x on a 4-vector)."""
    x, y, z, w = (float(v) for v in q)
    return math.atan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z))


def heading_frame_poses(body_pos_w: np.ndarray, body_quat_w: np.ndarray, pelvis_slot: int = _KP_PELVIS) -> tuple[np.ndarray, np.ndarray]:
    """Bodies ``(N, 3) / (N, 4) xyzw`` in the heading-aligned pelvis frame of the SAME body set: ``p = R(yaw(q_pelvis))^T (p_w - p_pelvis)``,
    ``q = yaw(q_pelvis)^-1 q_w`` (yaw only: pelvis pitch / roll stay in the local orientations)."""
    p = np.asarray(body_pos_w, dtype=np.float64)
    q = np.asarray(body_quat_w, dtype=np.float64)
    q_yaw = yaw_quat(q[pelvis_slot])
    return quat_apply_inv(q_yaw, p - p[pelvis_slot]), quat_mul(quat_conj(q_yaw), q)


def keypoint_metrics(robot_body_pos_w: np.ndarray, robot_body_quat_w: np.ndarray, ref_body_pos_w: np.ndarray, ref_body_quat_w: np.ndarray) -> dict[str, float]:
    """:data:`KEYPOINT_METRIC_KEYS` of one step (module docstring) from the robot's and the reference's 32 body poses in ``HOLOSOMA_BODY_NAMES_32``
    order (world frame, unit xyzw quaternions; MuJoCo ``xquat`` is wxyz -> convert first, ``MujocoPlant.canonical_body_poses`` does)."""
    arrays = {"robot_body_pos_w": (robot_body_pos_w, 3), "robot_body_quat_w": (robot_body_quat_w, 4), "ref_body_pos_w": (ref_body_pos_w, 3), "ref_body_quat_w": (ref_body_quat_w, 4)}
    got = {}
    for name, (a, width) in arrays.items():
        a = np.asarray(a, dtype=np.float64)
        if a.shape != (_NUM_BODIES, width):
            raise ValueError(f"keypoint_metrics: {name} has shape {a.shape}, expected {(_NUM_BODIES, width)} (HOLOSOMA_BODY_NAMES_32 order)")
        got[name] = a
    rp, rq, fp, fq = got["robot_body_pos_w"], got["robot_body_quat_w"], got["ref_body_pos_w"], got["ref_body_quat_w"]
    # ---- fused kernel (see _rel_quat_tensor): the numbers of the straightforward formulation
    #        g_pos = |rp - fp|;  g_rot = quat_angle(conj(rq) fq);  (rp_l, rq_l), (fp_l, fq_l) = heading_frame_poses(...) x2;
    #        l_pos = |rp_l - fp_l|;  l_rot = quat_angle(conj(rq_l) fq_l)
    # at ~1/4 of its cost.  Norms and geodesic angles are invariant under a common rotation, so the two own-heading frames collapse into ONE
    # yaw difference dpsi = psi_robot - psi_ref applied to the reference set:
    #        |R(-psi_r)(p_r - c_r) - R(-psi_f)(p_f - c_f)| = |(p_r - c_r) - R(dpsi)(p_f - c_f)|
    #        angle(conj(yaw_r^-1 q_r) (yaw_f^-1 q_f))     = angle(conj(q_r) qz(dpsi) q_f)          (z rotations compose additively)
    # and the global + local relative quaternions come out of one outer-product matmul.
    g_pos = np.linalg.norm(rp - fp, axis=-1)  # (32,) m
    dpsi = _yaw_of(rq[_KP_PELVIS]) - _yaw_of(fq[_KP_PELVIS])
    c, s = math.cos(dpsi), math.sin(dpsi)
    c2, s2 = math.cos(0.5 * dpsi), math.sin(0.5 * dpsi)
    d_r = rp - rp[_KP_PELVIS]
    d_f = (fp - fp[_KP_PELVIS]) @ np.array([[c, s, 0.0], [-s, c, 0.0], [0.0, 0.0, 1.0]])  # rows d @ Rz(dpsi)^T == Rz(dpsi) d
    l_pos = np.linalg.norm(d_r - d_f, axis=-1)
    # qz(dpsi) * fq as a row-vector matmul (left multiplication by a fixed quaternion is linear in fq); stacked with fq itself -> [global; local]
    fq_l = fq @ np.array([[c2, s2, 0.0, 0.0], [-s2, c2, 0.0, 0.0], [0.0, 0.0, c2, -s2], [0.0, 0.0, s2, c2]])
    b = np.concatenate([fq, fq_l])  # (64, 4)
    a = np.concatenate([rq, rq])
    rel = (a[:, :, None] * b[:, None, :]).reshape(-1, 16) @ _REL_QUAT_C16  # (64, 4) = conj(rq) * b, xyzw
    ang = quat_angle(rel)  # (64,) rad: [global 32 | local 32]
    g_rot, l_rot = ang[:_NUM_BODIES], ang[_NUM_BODIES:]
    means_g = _KP_MEAN_W @ g_pos  # (7,) m: kp14, feet, hands, head, pelvis, legs, arms
    dz_root = abs(float(rp[_KP_PELVIS, 2] - fp[_KP_PELVIS, 2]))
    dz_hands = np.abs(rp[_KP_HANDS, 2] - fp[_KP_HANDS, 2])
    out = {
        "kp_global_pos_cm": float(g_pos.mean() * 100.0),
        "kp_global_rot_deg": float(g_rot.mean() * RAD2DEG),
        "kp_local_pos_cm": float(l_pos.mean() * 100.0),
        "kp_local_rot_deg": float(l_rot.mean() * RAD2DEG),
        "kp14_global_pos_mm": float(means_g[0] * 1000.0),
        "kp14_local_pos_mm": float((_KP_MEAN_W[0] @ l_pos) * 1000.0),
    }
    for g, mean_m in zip(_KP_MEAN_ROWS[1:], means_g[1:]):
        out[f"kp_group_{g}_global_pos_cm"] = float(mean_m * 100.0)
    out["sonic_fail"] = float(dz_root > SONIC_FAIL_DZ_M or bool(np.any(dz_hands > SONIC_FAIL_DZ_M)))
    return out


def keypoint_definitions() -> dict:
    """Provenance block of the keypoint columns (``summary.json["keypoints"]``): body order, the 14-link set, the groups, the failure rule."""
    return {
        "bodies": list(HC.HOLOSOMA_BODY_NAMES_32),
        "kp14_bodies": list(SONIC14_BODY_NAMES),
        "groups": {g: list(v) for g, v in KEYPOINT_GROUPS.items()},
        "local_frame": "own pelvis, heading-aligned (inverse pelvis yaw only)",
        "global_frame": "world; reference placed at the episode start, no per-step alignment",
        "sonic_fail_dz_m": SONIC_FAIL_DZ_M,
        "sonic_fail_bodies": ["pelvis", *KEYPOINT_GROUPS["hands"]],
        "units": {"kp_*_cm": "cm", "kp_*_deg": "deg", "kp14_*_mm": "mm", "sonic_fail": "0/1 per step"},
        # the same rule block the Isaac Sim harness writes, so a joiner of the two simulators' ``keypoints`` blocks reads one schema
        "sonic_fail_rule": {
            "height_m": SONIC_FAIL_DZ_M,
            "root": "|z_root - z_root_ref| > height_m",
            "hands": list(KEYPOINT_GROUPS["hands"]),
            "success": "no failing step inside the clip's valid window",
        },
    }


def compute_metrics(
    *,
    robot_root_pos: np.ndarray,
    robot_root_quat: np.ndarray,
    robot_palm_pos_w: np.ndarray,
    robot_palm_quat_w: np.ndarray,
    robot_dof_pos: np.ndarray,
    ref_root_pos: np.ndarray,
    ref_palm_pos_local: np.ndarray,
    ref_palm_quat_local: np.ndarray,
    ref_palm_pos_w: np.ndarray,
    ref_dof_pos: np.ndarray,
    ref_palm_quat_w: np.ndarray | None = None,
    ref_root_quat: np.ndarray | None = None,
    robot_body_pos_w: np.ndarray | None = None,
    robot_body_quat_w: np.ndarray | None = None,
    ref_body_pos_w: np.ndarray | None = None,
    ref_body_quat_w: np.ndarray | None = None,
) -> dict[str, float]:
    """One step's :data:`METRIC_KEYS` (module docstring).  The four ``*_body_*`` arrays (``(32, 3)`` / ``(32, 4)`` xyzw, ``HOLOSOMA_BODY_NAMES_32``
    order) feed :func:`keypoint_metrics`; give all four or none -- without them the keypoint columns are NaN."""
    kp_inputs = (robot_body_pos_w, robot_body_quat_w, ref_body_pos_w, ref_body_quat_w)
    if all(a is None for a in kp_inputs):
        kp = {k: float("nan") for k in KEYPOINT_METRIC_KEYS}
    elif any(a is None for a in kp_inputs):
        raise ValueError("compute_metrics: give all four keypoint arrays (robot_body_pos_w, robot_body_quat_w, ref_body_pos_w, ref_body_quat_w) or none")
    else:
        kp = keypoint_metrics(*kp_inputs)
    dp, _, q_diff = ee_residual_parts(robot_root_pos, robot_root_quat, robot_palm_pos_w, robot_palm_quat_w, ref_palm_pos_local, ref_palm_quat_local)
    pos_cm = np.linalg.norm(dp, axis=-1) * 100.0
    rot_deg = quat_angle(q_diff) * RAD2DEG
    if ref_palm_quat_w is not None:
        rot_g = quat_angle(quat_mul(quat_conj(np.asarray(robot_palm_quat_w, dtype=np.float64)), np.asarray(ref_palm_quat_w, dtype=np.float64))) * RAD2DEG
    else:
        rot_g = np.full(2, np.nan)
    glob_cm = np.linalg.norm(np.asarray(robot_palm_pos_w) - np.asarray(ref_palm_pos_w), axis=-1) * 100.0
    anchor = np.asarray(ref_root_pos, dtype=np.float64) - np.asarray(robot_root_pos, dtype=np.float64)
    if ref_root_quat is not None:  # heading error = yaw of ref relative to robot (navigation accuracy)
        q_rel = quat_mul(quat_conj(np.asarray(robot_root_quat, dtype=np.float64)), np.asarray(ref_root_quat, dtype=np.float64))
        yaw_err = abs(float(np.degrees(np.arctan2(2.0 * (q_rel[3] * q_rel[2] + q_rel[0] * q_rel[1]), 1.0 - 2.0 * (q_rel[1] ** 2 + q_rel[2] ** 2)))))
    else:
        yaw_err = float("nan")
    dq = np.abs(np.asarray(robot_dof_pos, dtype=np.float64) - np.asarray(ref_dof_pos, dtype=np.float64))
    return {
        "ee_local_cm": float(pos_cm.mean()),
        "ee_local_left_cm": float(pos_cm[0]),
        "ee_local_right_cm": float(pos_cm[1]),
        "ee_global_cm": float(glob_cm.mean()),
        "ee_global_left_cm": float(glob_cm[0]),
        "ee_global_right_cm": float(glob_cm[1]),
        "ee_rot_deg": float(rot_deg.mean()),
        "ee_rot_left_deg": float(rot_deg[0]),
        "ee_rot_right_deg": float(rot_deg[1]),
        "ee_rot_global_deg": float(rot_g.mean()),
        "ee_rot_global_left_deg": float(rot_g[0]),
        "ee_rot_global_right_deg": float(rot_g[1]),
        "anchor_pos_cm": float(np.linalg.norm(anchor) * 100.0),
        "anchor_xy_cm": float(np.linalg.norm(anchor[:2]) * 100.0),
        "base_height_cm": float(abs(anchor[2]) * 100.0),
        "root_yaw_err_deg": yaw_err,
        "joint_upper_rad": float(dq[_ARM].mean()),
        "joint_upper17_rad": float(dq[_UPPER17].mean()),
        **kp,
    }


def odometry_metrics(err, state=None) -> dict[str, float]:
    """:data:`ODOM_METRIC_KEYS` of one step from a :class:`sim2sim.bench.odometry.OdomError` (``xy_err_m`` / ``pos_err_m`` / ``yaw_err_rad`` /
    ``z_err_m`` / ``vel_err_m_s``) and the estimator state (``stance``; None -> NaN); duck-typed so the metrics module stays mujoco-free."""
    stance = getattr(state, "stance", None)
    return {
        "odom_xy_err_cm": float(err.xy_err_m) * 100.0,
        "odom_pos_err_cm": float(err.pos_err_m) * 100.0,
        "odom_yaw_err_deg": float(err.yaw_err_rad) * RAD2DEG,
        "odom_z_err_cm": float(err.z_err_m) * 100.0,
        "odom_vel_err_cm_s": float(err.vel_err_m_s) * 100.0,
        "odom_stance_feet": (float(np.count_nonzero(stance)) if stance is not None else float("nan")),
    }


__all__ = ["KEYPOINT_GROUPS", "KEYPOINT_METRIC_KEYS", "METRIC_KEYS", "ODOM_METRIC_KEYS", "RAD2DEG", "SONIC14_BODY_NAMES", "SONIC_FAIL_DZ_M",
           "compute_metrics", "heading_frame_poses", "keypoint_definitions", "keypoint_metrics", "odometry_metrics"]
