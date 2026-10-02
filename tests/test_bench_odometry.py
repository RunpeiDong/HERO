"""LiDAR-inertial odometry model (sim2sim.bench.odometry.LidarInertialOdometry): config / presets / continuous timing, zero-latency zero-delay
zero-noise == truth, the output age shifting the whole state by exactly its value, the latency lag under ``stale`` and its absence under ``imu``,
drift proportional to the distance travelled, the per-episode bias, seed determinism, the source dispatch helpers and the estimator on a frozen
MuJoCo plant (skipped without mujoco)."""
from __future__ import annotations

import numpy as np
import pytest

from sim2sim.bench.odometry import (
    DEG,
    LIO_PRESET_VALUES,
    LIO_PRESETS,
    LIO_PROPAGATIONS,
    ODOM_SOURCES,
    LidarInertialOdometry,
    LidarInertialOdometryConfig,
    LioState,
    OdomError,
    build_odometry,
    config_source,
    default_odometry_config,
    euler_zyx,
    wrap_angle,
)
from sim2sim.mathutil import quat_from_euler_xyz

from _bench_fixtures import needs_mujoco

DT = 0.02
NOISELESS = {"drift_pct": 0.0, "bias_xy_m": 0.0, "yaw_walk_deg_sqrt_s": 0.0, "bias_z_m": 0.0, "z_walk_m_sqrt_s": 0.0, "prop_vel_std_m_s": 0.0, "gyro_bias_deg_s": 0.0, "gyro_white_deg_s": 0.0}


def straight(T: float = 4.0, *, speed: float = 0.8, yaw: float = 0.3, z: float = 0.75):
    n = int(round(T / DT))
    v = speed * np.array([np.cos(yaw), np.sin(yaw), 0.0])
    q = quat_from_euler_xyz(0.0, 0.0, yaw)
    return [(np.array([0.0, 0.0, z]) + v * (k * DT), q.copy(), v.copy()) for k in range(n + 1)]


def curved(T: float = 6.0, *, speed: float = 0.6, yaw_rate: float = 0.5, z: float = 0.75):
    n = int(round(T / DT))
    out = []
    for k in range(n + 1):
        t = k * DT
        yaw = 0.2 + yaw_rate * t
        v = speed * np.array([np.cos(yaw), np.sin(yaw), 0.0]) + np.array([0.0, 0.0, 0.3 * np.cos(4.0 * t)])
        pos = np.array([0.2 + speed / yaw_rate * (np.sin(yaw) - np.sin(0.2)), -speed / yaw_rate * (np.cos(yaw) - np.cos(0.2)), z + 0.075 * np.sin(4.0 * t)])
        q = quat_from_euler_xyz(0.05 * np.sin(3.0 * t), 0.04 * np.cos(2.0 * t), yaw)
        out.append((pos, q, v))
    return out


def run_model(cfg, traj):
    lio = LidarInertialOdometry(cfg)
    p0, q0, v0 = traj[0]
    lio.reset(p0, q0, lin_vel_w=v0)
    states, errs = [], []
    for p, q, v in traj[1:]:
        states.append(lio.step(p, q, v))
        errs.append(lio.error_vs_truth(p, q, v))
    return states, errs, lio


def cfg_exact(**kw) -> LidarInertialOdometryConfig:
    return LidarInertialOdometryConfig(**{**NOISELESS, "output_delay_s": 0.0, **kw})


def test_config_presets_validation_and_continuous_timing():
    so = LidarInertialOdometryConfig.preset("so")
    assert LIO_PRESETS == ("so", "so_slow") and LIO_PROPAGATIONS == ("imu", "stale", "hold") and ODOM_SOURCES == ("truth", "leg", "so")
    assert (so.rate_hz, so.latency_s, so.output_delay_s, so.propagation, so.drift_pct, so.bias_xy_m) == (10.0, 0.03, 0.02, "imu", 0.3, 0.01) and so == LidarInertialOdometryConfig()
    slow = LidarInertialOdometryConfig.preset("so_slow")
    assert slow.latency_s == 0.1 and LIO_PRESET_VALUES["so_slow"]["latency_s"] == 0.1
    assert LidarInertialOdometryConfig.preset("so", latency_s=0.2, rate_hz=None).latency_s == 0.2  # None overrides are ignored
    assert (so.gyro_bias_deg_s, so.gyro_white_deg_s) == (0.01, 0.05) and so.noise_active and not cfg_exact().noise_active
    d = so.as_dict()
    assert d["scan_period_s"] == pytest.approx(0.1) and d["kind"] == "lidar_inertial" and d["noise_active"] is True
    assert LidarInertialOdometryConfig.coerce(d) == so and LidarInertialOdometryConfig.coerce(None) == so and LidarInertialOdometryConfig.coerce(so) is so
    for bad in ({"propagation": "x"}, {"rate_hz": 0.0}, {"latency_s": -0.1}, {"output_delay_s": -0.01}, {"drift_pct": -1.0}, {"dt": 0.0}):
        with pytest.raises(ValueError):
            LidarInertialOdometryConfig(**bad)
    with pytest.raises(ValueError):
        LidarInertialOdometryConfig.preset("fast_lio")
    with pytest.raises(ValueError, match="unknown LidarInertialOdometryConfig keys"):
        LidarInertialOdometryConfig.coerce({"latency_ms": 30})
    assert [so.scan_time(n) for n in range(4)] == pytest.approx([0.0, 0.1, 0.2, 0.3]) and so.arrival_time(1) == pytest.approx(0.13)
    assert [so.latest_available_scan(t) for t in (0.0, 0.029, 0.03, 0.1, 0.129, 0.13, 0.2, 0.23, 0.9999, 1.03)] == [-1, -1, 0, 0, 0, 1, 1, 2, 9, 10]
    c20 = LidarInertialOdometryConfig(rate_hz=20.0, latency_s=0.0)
    assert [c20.latest_available_scan(t) for t in (0.0, 0.04, 0.05, 0.07, 0.1)] == [0, 0, 1, 1, 2]


@pytest.mark.parametrize("propagation", ["imu", "stale"])
def test_zero_latency_zero_delay_zero_noise_equals_truth(propagation):
    for traj in (straight(), curved()):
        states, errs, _ = run_model(cfg_exact(latency_s=0.0, propagation=propagation), traj)
        for st, (p, q, v), e in zip(states, traj[1:], errs):
            assert np.allclose(st.pos_w, p, atol=1e-9) and abs(wrap_angle(st.yaw - euler_zyx(q)[2])) < 1e-9 and np.allclose(st.lin_vel_w, v, atol=1e-9)
            assert e.pos_err_m < 1e-9 and abs(e.yaw_err_rad) < 1e-9 and e.vel_err_m_s < 1e-9


def test_output_delay_shifts_the_whole_state_by_exactly_its_age():
    traj = straight(speed=0.8)
    v = traj[1][2]
    for prop in ("imu", "stale"):
        states, errs, _ = run_model(cfg_exact(latency_s=0.0, output_delay_s=0.05, propagation=prop), traj)
        for st, e in zip(states[20:], errs[20:]):   # after the clamp window
            assert st.output_age_s == pytest.approx(0.05)
            assert e.xy_err_m == pytest.approx(0.8 * 0.05, abs=1e-9)   # lag v x age on a constant-velocity line
            assert np.allclose(st.lin_vel_w, v, atol=1e-9)
    states, errs, _ = run_model(cfg_exact(latency_s=0.0, output_delay_s=0.05, propagation="hold"), traj)
    lags = np.array([e.xy_err_m for e in errs[20:]])   # hold: the age plus the scan age (staircase)
    assert lags.min() >= 0.8 * 0.05 - 1e-9 and lags.max() <= 0.8 * (0.05 + 0.1) + 1e-9


def test_latency_lag_is_v_times_latency_under_stale_and_absent_under_imu():
    traj = straight(speed=0.8)
    for lat in (0.03, 0.04, 0.1):
        _, errs_stale, _ = run_model(cfg_exact(latency_s=lat, propagation="stale"), traj)
        _, errs_imu, _ = run_model(cfg_exact(latency_s=lat, propagation="imu"), traj)
        # once a correction has arrived the stale pipeline lags by exactly v x latency; the imu pipeline has no lag at all
        assert all(e.xy_err_m == pytest.approx(0.8 * lat, abs=1e-9) for e in errs_stale[20:])
        assert all(e.xy_err_m < 1e-9 for e in errs_imu)
    # hold: staircase output, lag between v x latency and v x (latency + period)
    _, errs_hold, _ = run_model(cfg_exact(latency_s=0.03, propagation="hold"), traj)
    lags = np.array([e.xy_err_m for e in errs_hold[20:]])
    assert lags.min() >= 0.8 * 0.03 - 1e-9 and lags.max() <= 0.8 * (0.03 + 0.1) + 1e-9


def test_drift_scales_with_distance_and_bias_is_constant_per_episode():
    traj = straight(T=10.0, speed=0.8)
    cfg = LidarInertialOdometryConfig(**{**NOISELESS, "output_delay_s": 0.0, "latency_s": 0.0, "drift_pct": 1.0, "bias_xy_m": 0.0})
    states, _, lio = run_model(cfg, traj)
    s = 0.8 * np.array([st.step * DT for st in states])
    drift = np.array([np.linalg.norm(st.drift_xy) for st in states])
    rate = np.linalg.norm(lio._drift_rate)
    assert np.allclose(drift, rate * np.floor(s / 0.08 + 1e-9) * 0.08, atol=1e-9)  # the drift of the correction in force (scan every 0.1 s = 0.08 m)
    cfg_b = LidarInertialOdometryConfig(**{**NOISELESS, "output_delay_s": 0.0, "latency_s": 0.0, "bias_xy_m": 0.02})
    states_b, _, _ = run_model(cfg_b, traj)
    biases = np.array([st.drift_xy for st in states_b[10:]])
    assert np.allclose(biases, biases[0]) and np.linalg.norm(biases[0]) > 0.0


def test_seed_determinism_reset_reseeds_and_dispatch_helpers():
    traj = curved()
    a, _, _ = run_model(LidarInertialOdometryConfig(seed=3), traj)
    b, _, _ = run_model(LidarInertialOdometryConfig(seed=3), traj)
    c, _, _ = run_model(LidarInertialOdometryConfig(seed=4), traj)
    assert all(np.array_equal(x.pos_w, y.pos_w) for x, y in zip(a, b)) and any(not np.array_equal(x.pos_w, y.pos_w) for x, y in zip(a, c))
    lio = LidarInertialOdometry(LidarInertialOdometryConfig(seed=3))
    p0, q0, v0 = traj[0]
    lio.reset(p0, q0, lin_vel_w=v0)
    first = lio._bias_xy.copy()
    lio.reset(p0, q0, lin_vel_w=v0)
    assert not np.array_equal(first, lio._bias_xy) and lio.resets == 2
    st = lio.state
    assert isinstance(st, LioState) and st.stance is None and st.quat_wxyz.shape == (4,) and "pos_w" in st.as_dict()
    assert isinstance(lio.error_vs_truth(p0, q0, v0), OdomError)
    assert lio.describe()["kind"] == "lidar_inertial_odometry_v2" and lio.describe()["source"] == "so"
    assert config_source(LidarInertialOdometryConfig()) == "so" and default_odometry_config("truth") is None and default_odometry_config("so") == LidarInertialOdometryConfig.preset("so")
    with pytest.raises(TypeError):
        config_source({"not": "a config"})
    with pytest.raises(ValueError):
        default_odometry_config("leg")
    with pytest.raises(ValueError):
        default_odometry_config("gps")
    assert DEG == pytest.approx(np.pi / 180.0)


@needs_mujoco
def test_for_plant_frozen_plant_equals_truth_and_reads_the_control_step():
    from sim2sim import plant_params as PP
    from sim2sim.plant import MujocoPlant

    plant = MujocoPlant(kp=PP.KP, kd=PP.KD, effort_limit=PP.ACTION_SCALE * PP.KP / 0.25, keep_visual=False)
    plant.reset(np.array([0.3, -0.2, 0.76]), np.array([0.0, 0.0, 0.0, 1.0]), np.asarray(PP.DEFAULT_DOF_POS))
    lio = build_odometry(plant, cfg_exact(latency_s=0.0))
    assert lio.cfg.dt == pytest.approx(plant.control_dt)
    lio.reset_from_plant()
    for _ in range(5):  # a frozen plant (no step) keeps its pose: the estimate is the truth
        st = lio.step_from_plant()
        e = lio.error_vs_truth(plant.root_pos, plant.root_quat, plant.root_lin_vel_w)
    assert np.allclose(st.pos_w, plant.root_pos, atol=1e-9) and e.pos_err_m < 1e-9 and abs(e.yaw_err_rad) < 1e-9
