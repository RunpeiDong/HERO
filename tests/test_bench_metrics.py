"""Per-step metrics of the benchmark rollouts (sim2sim.bench.metrics): the append-only key contract, analytic keypoint cases (identity,
whole-robot yaw, translation, the 14-link subset, the failure rule), the fused keypoint kernel against the straightforward formulation and the
NaN pass-through for callers without body arrays."""
from __future__ import annotations

import json

import numpy as np
import pytest

from hero_isaacsim import constants as HC
from sim2sim.bench import metrics as M
from sim2sim.mathutil import quat_angle, quat_apply, quat_conj, quat_from_euler_xyz, quat_mul, quat_normalize

LEGACY_KEYS: tuple[str, ...] = (
    "ee_local_cm", "ee_local_left_cm", "ee_local_right_cm", "ee_global_cm", "ee_global_left_cm", "ee_global_right_cm",
    "ee_rot_deg", "ee_rot_left_deg", "ee_rot_right_deg", "ee_rot_global_deg", "ee_rot_global_left_deg", "ee_rot_global_right_deg",
    "anchor_pos_cm", "anchor_xy_cm", "base_height_cm", "root_yaw_err_deg", "joint_upper_rad", "joint_upper17_rad",
)
PELVIS = HC.HOLOSOMA_BODY_NAMES_32.index("pelvis")
SLOT = {n: i for i, n in enumerate(HC.HOLOSOMA_BODY_NAMES_32)}
SLOTS14 = np.asarray([SLOT[n] for n in M.SONIC14_BODY_NAMES])


def _pose_set(seed: int) -> tuple[np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    p = rng.normal(size=(32, 3)) * 0.3 + np.array([0.0, 0.0, 0.75])
    q = quat_normalize(rng.normal(size=(32, 4)))
    return p, q


def _legacy_inputs(seed: int) -> dict[str, np.ndarray]:
    rng = np.random.default_rng(seed)
    uq = lambda *shape: quat_normalize(rng.normal(size=(*shape, 4)))  # noqa: E731
    return dict(
        robot_root_pos=rng.normal(size=3), robot_root_quat=uq(), robot_palm_pos_w=rng.normal(size=(2, 3)), robot_palm_quat_w=uq(2),
        robot_dof_pos=rng.normal(size=29), ref_root_pos=rng.normal(size=3), ref_palm_pos_local=rng.normal(size=(2, 3)), ref_palm_quat_local=uq(2),
        ref_palm_pos_w=rng.normal(size=(2, 3)), ref_dof_pos=rng.normal(size=29), ref_palm_quat_w=uq(2), ref_root_quat=uq(),
    )


def test_metric_keys_are_append_only_and_the_body_sets_are_documented():
    assert M.METRIC_KEYS[: len(LEGACY_KEYS)] == LEGACY_KEYS and M.METRIC_KEYS[len(LEGACY_KEYS):] == M.KEYPOINT_METRIC_KEYS
    assert len(set(M.METRIC_KEYS)) == len(M.METRIC_KEYS) and set(M.KEYPOINT_METRIC_KEYS).isdisjoint(M.ODOM_METRIC_KEYS)
    assert len(M.SONIC14_BODY_NAMES) == 14 and all(n in SLOT for n in M.SONIC14_BODY_NAMES)
    assert tuple(M.KEYPOINT_GROUPS) == ("feet", "hands", "head", "pelvis", "legs", "arms")
    assert M.KEYPOINT_GROUPS["feet"] == HC.ANKLE_BODY_NAMES and M.KEYPOINT_GROUPS["hands"] == HC.EE_BODY_NAMES
    assert len(M.KEYPOINT_GROUPS["legs"]) == 12 and len(M.KEYPOINT_GROUPS["arms"]) == 14
    d = M.keypoint_definitions()
    assert d["bodies"] == list(HC.HOLOSOMA_BODY_NAMES_32) and d["sonic_fail_dz_m"] == 0.25 and d["sonic_fail_bodies"] == ["pelvis", "left_wrist_yaw_link", "right_wrist_yaw_link"]
    json.dumps(d)


def test_identity_gives_zero_everywhere_and_the_key_order():
    p, q = _pose_set(1)
    m = M.keypoint_metrics(p, q, p.copy(), q.copy())
    assert list(m) == list(M.KEYPOINT_METRIC_KEYS)
    for k, v in m.items():
        assert v == pytest.approx(0.0, abs=1e-9), k


def test_whole_robot_yaw_offset_is_global_rotation_only():
    p, q = _pose_set(2)
    psi = np.deg2rad(37.0)
    qz = quat_from_euler_xyz(0.0, 0.0, psi)
    p_r = p[PELVIS] + quat_apply(qz, p - p[PELVIS])
    q_r = quat_mul(qz, q)
    m = M.keypoint_metrics(p_r, q_r, p, q)
    assert m["kp_global_rot_deg"] == pytest.approx(37.0, abs=1e-6)
    assert m["kp_local_rot_deg"] == pytest.approx(0.0, abs=1e-6) and m["kp_local_pos_cm"] == pytest.approx(0.0, abs=1e-6) and m["kp14_local_pos_mm"] == pytest.approx(0.0, abs=1e-6)
    exp = np.linalg.norm(p_r - p, axis=-1)
    assert m["kp_group_pelvis_global_pos_cm"] == pytest.approx(0.0, abs=1e-9) and m["kp_global_pos_cm"] == pytest.approx(exp.mean() * 100.0) and m["kp_global_pos_cm"] > 1.0
    assert m["kp14_global_pos_mm"] == pytest.approx(exp[SLOTS14].mean() * 1000.0)
    for g, names in M.KEYPOINT_GROUPS.items():
        assert m[f"kp_group_{g}_global_pos_cm"] == pytest.approx(exp[[SLOT[n] for n in names]].mean() * 100.0), g
    assert m["sonic_fail"] == 0.0


def test_translated_robot_is_global_position_only_and_trips_the_fail_rule_by_height():
    p, q = _pose_set(3)
    d = np.array([0.12, -0.05, 0.03])
    m = M.keypoint_metrics(p + d, q, p, q)
    n_cm = float(np.linalg.norm(d)) * 100.0
    for k in ("kp_global_pos_cm", *(f"kp_group_{g}_global_pos_cm" for g in M.KEYPOINT_GROUPS)):
        assert m[k] == pytest.approx(n_cm, abs=1e-9), k
    assert m["kp14_global_pos_mm"] == pytest.approx(n_cm * 10.0, abs=1e-9)
    for k in ("kp_global_rot_deg", "kp_local_pos_cm", "kp_local_rot_deg", "kp14_local_pos_mm", "sonic_fail"):
        assert m[k] == pytest.approx(0.0, abs=1e-9), k
    m_up = M.keypoint_metrics(p + np.array([0.0, 0.0, 0.3]), q, p, q)
    assert m_up["sonic_fail"] == 1.0 and m_up["kp_local_pos_cm"] == pytest.approx(0.0, abs=1e-9)
    # a hand 30 cm too high fails too; a knee does not
    p_h = p.copy()
    p_h[SLOT["left_wrist_yaw_link"], 2] += 0.3
    assert M.keypoint_metrics(p_h, q, p, q)["sonic_fail"] == 1.0
    p_k = p.copy()
    p_k[SLOT["left_knee_link"], 2] += 0.3
    assert M.keypoint_metrics(p_k, q, p, q)["sonic_fail"] == 0.0


def _straightforward(rp, rq, fp, fq) -> dict[str, float]:
    g_pos = np.linalg.norm(rp - fp, axis=-1)
    g_rot = quat_angle(quat_mul(quat_conj(rq), fq))
    rp_l, rq_l = M.heading_frame_poses(rp, rq)
    fp_l, fq_l = M.heading_frame_poses(fp, fq)
    l_pos = np.linalg.norm(rp_l - fp_l, axis=-1)
    l_rot = quat_angle(quat_mul(quat_conj(rq_l), fq_l))
    return {"kp_global_pos_cm": g_pos.mean() * 100.0, "kp_global_rot_deg": g_rot.mean() * M.RAD2DEG, "kp_local_pos_cm": l_pos.mean() * 100.0,
            "kp_local_rot_deg": l_rot.mean() * M.RAD2DEG, "kp14_global_pos_mm": g_pos[SLOTS14].mean() * 1000.0, "kp14_local_pos_mm": l_pos[SLOTS14].mean() * 1000.0}


@pytest.mark.parametrize("seed", [10, 11, 12, 13])
def test_fused_kernel_matches_the_straightforward_formulation(seed):
    rp, rq = _pose_set(seed)
    fp, fq = _pose_set(seed + 100)
    m = M.keypoint_metrics(rp, rq, fp, fq)
    ref = _straightforward(rp, rq, fp, fq)
    for k, v in ref.items():
        assert m[k] == pytest.approx(v, abs=1e-9), k


def test_compute_metrics_nan_pass_through_for_old_callers_and_input_validation():
    kw = _legacy_inputs(5)
    m = M.compute_metrics(**kw)
    assert list(m) == list(M.METRIC_KEYS)
    assert all(np.isnan(m[k]) for k in M.KEYPOINT_METRIC_KEYS) and all(np.isfinite(m[k]) for k in LEGACY_KEYS)
    assert m["ee_global_cm"] == pytest.approx(np.linalg.norm(kw["robot_palm_pos_w"] - kw["ref_palm_pos_w"], axis=-1).mean() * 100.0)
    assert m["anchor_xy_cm"] == pytest.approx(np.linalg.norm((kw["ref_root_pos"] - kw["robot_root_pos"])[:2]) * 100.0)
    assert m["joint_upper_rad"] == pytest.approx(np.abs(kw["robot_dof_pos"] - kw["ref_dof_pos"])[list(HC.ARM_DOF_IDX)].mean())
    p, q = _pose_set(6)
    with pytest.raises(ValueError, match="all four"):
        M.compute_metrics(**kw, robot_body_pos_w=p, robot_body_quat_w=q, ref_body_pos_w=p)
    with pytest.raises(ValueError, match="shape"):
        M.keypoint_metrics(p[:31], q, p, q)
    full = M.compute_metrics(**kw, robot_body_pos_w=p, robot_body_quat_w=q, ref_body_pos_w=p, ref_body_quat_w=q)
    assert all(full[k] == pytest.approx(0.0, abs=1e-9) for k in M.KEYPOINT_METRIC_KEYS)
    # the odometry columns from a duck-typed error object; no stance -> NaN stance count
    class _Err:
        xy_err_m, pos_err_m, yaw_err_rad, z_err_m, vel_err_m_s = 0.01, 0.02, 0.1, -0.005, 0.3

    o = M.odometry_metrics(_Err(), None)
    assert o["odom_xy_err_cm"] == pytest.approx(1.0) and o["odom_yaw_err_deg"] == pytest.approx(np.degrees(0.1)) and o["odom_z_err_cm"] == pytest.approx(-0.5) and np.isnan(o["odom_stance_feet"])
