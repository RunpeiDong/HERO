"""Root-pose estimate behind the policy's root-feedback inputs: a LiDAR-inertial odometry error model.

The HERO root-feedback terms (``h20_ref_root_pose_b`` ... ``h23_base_lin_vel_odom``, :mod:`sim2sim.policy_hero_export`) are built from the
robot's root pose.  A real robot has no exact root pose; :class:`LidarInertialOdometry` is a TRUTH-BASED error model of a
LiDAR-visual-inertial estimator (IMU-centric, LiDAR corrections at the scan rate) -- no point clouds are rendered, the model describes the
estimate's error relative to the simulator truth.  ``HeroExportPolicy(odom_source="so")`` builds every root-feedback term from the estimate
(``RobotState.odom``, attached by :func:`sim2sim.bench.rollout.rollout_clip`), and the rollout records the estimate's error vs truth in the
:data:`sim2sim.bench.metrics.ODOM_METRIC_KEYS` columns.  ``odom_source="truth"`` (the default) bypasses the estimator.

Interface summary (quaternions xyzw everywhere; ``dt`` = the control step)::

    lio = LidarInertialOdometry.for_plant(plant, LidarInertialOdometryConfig.preset("so", latency_s=0.1))
    lio.reset_from_plant() | lio.reset(pos_w, quat_xyzw, lin_vel_w=)            # t0 = truth (alignment)
    st = lio.step_from_plant() | lio.step(pos_w, quat_xyzw, lin_vel_w)          # the TRUE pose of this control step -> LioState (estimate)
    err = lio.error_vs_truth(root_pos, root_quat, root_lin_vel_w)               # OdomError: xy_err_m pos_err_m yaw_err_rad z_err_m vel_err_m_s
    build_odometry(plant, cfg) / default_odometry_config(source) / config_source(cfg)
"""

from __future__ import annotations

from dataclasses import dataclass, replace
from typing import Any, Mapping

import numpy as np

from sim2sim.mathutil import quat_apply_inv, quat_from_euler_xyz, quat_normalize, xyzw_to_wxyz, yaw_quat

#: Root-pose sources a controller may name (``HeroExportPolicy.odom_source``): ``truth`` = the exact simulator state, ``so`` = the
#: LiDAR-inertial model below.  ``leg`` names the leg-odometry estimator of the browser demo / internal tooling; the benchmark runner does not
#: ship it (``--odom`` accepts ``truth`` / ``so``), the name is kept so the policy accepts every source the exports were tested with.
ODOM_SOURCES: tuple[str, ...] = ("truth", "leg", "so")
QUAT_CONVENTION = "xyzw"
DEG = np.pi / 180.0
#: IMU gyro noise of the G1-class sensor preset, acting over the IMU-only window of the model.
GYRO_NOISE: dict[str, float] = {"gyro_bias_deg_s": 0.01, "gyro_white_deg_s": 0.05}


# ================================================================================================ error vs truth
@dataclass
class OdomError:
    """Estimate minus truth of one step (``yaw_err_rad`` wrapped to [-pi, pi); ``z_err_m`` signed = est - true)."""

    xy_err_m: float
    pos_err_m: float
    yaw_err_rad: float
    z_err_m: float
    vel_err_m_s: float

    def as_dict(self) -> dict[str, float]:
        return {"xy_err_m": self.xy_err_m, "pos_err_m": self.pos_err_m, "yaw_err_rad": self.yaw_err_rad, "z_err_m": self.z_err_m, "vel_err_m_s": self.vel_err_m_s}


def estimate_error(state: Any, pos_w: np.ndarray, quat_xyzw: np.ndarray, lin_vel_w: np.ndarray | None = None) -> OdomError:
    """Estimate - truth of a state with ``pos_w`` / ``yaw`` / ``lin_vel_w``: planar / 3D position error, wrapped signed yaw error, signed z
    error, world-frame velocity error (NaN without a true velocity)."""
    p = np.asarray(pos_w, dtype=np.float64).reshape(3)
    d = np.asarray(state.pos_w, dtype=np.float64).reshape(3) - p
    _, _, yaw_true = euler_zyx(np.asarray(quat_xyzw, dtype=np.float64).reshape(4))
    v_err = float("nan") if lin_vel_w is None else float(np.linalg.norm(np.asarray(state.lin_vel_w, dtype=np.float64).reshape(3) - np.asarray(lin_vel_w, dtype=np.float64).reshape(3)))
    return OdomError(xy_err_m=float(np.linalg.norm(d[:2])), pos_err_m=float(np.linalg.norm(d)), yaw_err_rad=wrap_angle(float(state.yaw) - yaw_true), z_err_m=float(d[2]), vel_err_m_s=v_err)


# ================================================================================================ math helpers
def euler_zyx(q_xyzw: np.ndarray) -> tuple[float, float, float]:
    """(roll, pitch, yaw) of an xyzw quaternion (ZYX)."""
    x, y, z, w = (float(v) for v in np.asarray(q_xyzw, dtype=np.float64).reshape(4))
    roll = float(np.arctan2(2.0 * (w * x + y * z), 1.0 - 2.0 * (x * x + y * y)))
    pitch = float(np.arcsin(np.clip(2.0 * (w * y - z * x), -1.0, 1.0)))
    yaw = float(np.arctan2(2.0 * (w * z + x * y), 1.0 - 2.0 * (y * y + z * z)))
    return roll, pitch, yaw


def wrap_angle(a: float) -> float:
    return float((float(a) + np.pi) % (2.0 * np.pi) - np.pi)


def quat_rp_yaw(roll: float, pitch: float, yaw: float) -> np.ndarray:
    """``R_z(yaw) R_y(pitch) R_x(roll)`` as an xyzw quaternion."""
    return quat_from_euler_xyz(roll, pitch, yaw)


# ================================================================================================ LiDAR-inertial odometry
LIO_PRESETS: tuple[str, ...] = ("so", "so_slow")  # LidarInertialOdometryConfig.preset names
LIO_PROPAGATIONS: tuple[str, ...] = ("imu", "stale", "hold")  # LidarInertialOdometryConfig.propagation (class docstring); "imu" = the default
#: Preset values (:meth:`LidarInertialOdometryConfig.preset`): ``so`` = 10 Hz scans, 30 ms processing latency, IMU-rate output
#: (``propagation="imu"``: the latency only widens the IMU-only window), 20 ms output age at the policy tick (one control step of transport /
#: publish delay), 0.3 % of distance drift, 1 cm per-episode offset; ``so_slow`` = the same with a 100 ms processing latency.  Every time is
#: honoured EXACTLY (the truth is interpolated between control steps): 30 ms means 30 ms.
LIO_PRESET_VALUES: dict[str, dict[str, Any]] = {
    "so": {"rate_hz": 10.0, "latency_s": 0.03, "output_delay_s": 0.02, "propagation": "imu", "drift_pct": 0.3, "bias_xy_m": 0.01},
    "so_slow": {"rate_hz": 10.0, "latency_s": 0.10, "output_delay_s": 0.02, "propagation": "imu", "drift_pct": 0.3, "bias_xy_m": 0.01},
}
_LIO_EPS = 1.0e-9


@dataclass(frozen=True)
class LidarInertialOdometryConfig:
    """Every knob of :class:`LidarInertialOdometry` (defaults == the ``so`` preset; ``LidarInertialOdometryConfig.preset("so_slow")`` for
    the 100 ms variant).  Times in s, lengths in m, angles as named; ``dt`` = the control step the truth is sampled at.  None of the times is
    snapped to the control grid: the estimator interpolates the truth linearly between control steps, so ``latency_s`` / ``output_delay_s`` /
    ``1 / rate_hz`` act at their exact values."""

    rate_hz: float = 10.0  # LiDAR scan rate: a correction (pose of the scan time) becomes available every 1 / rate s
    latency_s: float = 0.03  # LiDAR processing latency: the correction of the scan taken at t_c is available at t_c + latency (exact, off-grid allowed)
    output_delay_s: float = 0.02  # age of the estimator output the policy reads at its tick (transport / publish delay; one control step)
    propagation: str = "imu"  # how the output is carried from the correction to now: "imu" (default) | "stale" | "hold" (class docstring)
    drift_pct: float = 0.3  # correction-frame planar drift, % of the distance travelled: RMS |drift_xy| = drift_pct / 100 x distance
    bias_xy_m: float = 0.01  # per-episode fixed offset of the correction frame vs the alignment pose, per axis (std)
    yaw_walk_deg_sqrt_s: float = 0.05  # correction-frame yaw drift: Wiener walk, std after T s = this x sqrt(T)
    bias_z_m: float = 0.005  # per-episode height offset of the correction frame (std)
    z_walk_m_sqrt_s: float = 0.002  # correction-frame height drift: Wiener walk, std after T s = this x sqrt(T)
    prop_vel_std_m_s: float = 0.02  # IMU-propagation velocity error per correction, per axis (std): the position error grows by this x the propagation window
    gyro_bias_deg_s: float = GYRO_NOISE["gyro_bias_deg_s"]  # gyro yaw bias per episode (std), acts over the IMU-only window
    gyro_white_deg_s: float = GYRO_NOISE["gyro_white_deg_s"]  # gyro white noise per control-step sample (std), acts over the IMU-only window
    dt: float = 0.02
    seed: int = 0

    def __post_init__(self) -> None:
        if self.propagation not in LIO_PROPAGATIONS:
            raise ValueError(f"propagation must be one of {LIO_PROPAGATIONS}, got {self.propagation!r}")
        if not float(self.rate_hz) > 0.0:
            raise ValueError(f"rate_hz must be > 0, got {self.rate_hz!r}")
        for k in ("latency_s", "output_delay_s", "drift_pct", "bias_xy_m", "yaw_walk_deg_sqrt_s", "bias_z_m", "z_walk_m_sqrt_s", "prop_vel_std_m_s", "gyro_bias_deg_s", "gyro_white_deg_s"):
            if float(getattr(self, k)) < 0.0:
                raise ValueError(f"LidarInertialOdometryConfig.{k} must be >= 0")
        if float(self.dt) <= 0.0:
            raise ValueError("dt must be > 0")

    # ---- timing (continuous; nothing is snapped to the control grid) ----
    @property
    def scan_period_s(self) -> float:
        return 1.0 / float(self.rate_hz)

    def scan_time(self, n: int) -> float:
        """Time of scan ``n`` (``n = 0`` = the reset instant = the alignment): ``n / rate``."""
        return float(n) / float(self.rate_hz)

    def arrival_time(self, n: int) -> float:
        """Time the correction of scan ``n`` becomes available: ``scan_time(n) + latency_s``."""
        return self.scan_time(n) + float(self.latency_s)

    def latest_available_scan(self, t: float) -> int:
        """Index of the newest scan whose correction has arrived by time ``t`` (``arrival_time(n) <= t``); ``0`` when only the reset scan is
        available (the reset itself is the alignment, no correction is applied for it), ``-1`` before even that (``t < latency``)."""
        m = float(t) - float(self.latency_s)
        if m < -_LIO_EPS:
            return -1
        return int(np.floor(max(m, 0.0) * float(self.rate_hz) + _LIO_EPS))

    @property
    def noise_active(self) -> bool:
        """Any random component on (drift / bias / walks / propagation error / gyro noise)."""
        return any(float(getattr(self, k)) > 0.0 for k in ("drift_pct", "bias_xy_m", "yaw_walk_deg_sqrt_s", "bias_z_m", "z_walk_m_sqrt_s", "prop_vel_std_m_s", "gyro_bias_deg_s", "gyro_white_deg_s"))

    @classmethod
    def preset(cls, name: str = "so", **overrides: Any) -> "LidarInertialOdometryConfig":
        """``preset("so")`` / ``preset("so_slow")`` (:data:`LIO_PRESET_VALUES`); ``overrides`` (None ignored) win -- e.g. ``latency_s=0.2`` or
        ``output_delay_s=0.1`` for a sweep, ``propagation="stale"`` for the pessimistic pipeline."""
        if name not in LIO_PRESETS:
            raise ValueError(f"unknown LiDAR-inertial preset {name!r} (known: {LIO_PRESETS})")
        kw: dict[str, Any] = dict(LIO_PRESET_VALUES[name])
        kw.update({k: v for k, v in overrides.items() if v is not None})
        return cls(**kw)

    @classmethod
    def coerce(cls, value: "LidarInertialOdometryConfig | Mapping[str, Any] | None") -> "LidarInertialOdometryConfig":
        """None -> defaults (== ``so``); a config -> itself; a mapping -> ``cls(**mapping)`` (the derived keys of :meth:`as_dict` are
        accepted for a round-trip; any other unknown key raises)."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        derived = ("scan_period_s", "noise_active", "kind")
        unknown = sorted(str(k) for k in value if k not in cls.__dataclass_fields__ and k not in derived)
        if unknown:
            raise ValueError(f"unknown LidarInertialOdometryConfig keys {unknown} (known: {sorted(cls.__dataclass_fields__)})")
        return cls(**{k: value[k] for k in cls.__dataclass_fields__ if k in value})

    def as_dict(self) -> dict[str, Any]:
        return {k: getattr(self, k) for k in self.__dataclass_fields__} | {"scan_period_s": self.scan_period_s, "noise_active": self.noise_active, "kind": "lidar_inertial"}


@dataclass
class LioState:
    """The LiDAR-inertial estimate after one :meth:`LidarInertialOdometry.step` (m, m/s, rad; quaternion xyzw): ``pos_w`` / ``quat_xyzw`` /
    ``quat_wxyz`` / ``yaw`` / ``roll`` / ``pitch`` / ``z`` / ``lin_vel_*`` for the consumers; ``stance`` is None (no foot-contact notion:
    ``odom_stance_feet`` = NaN).  The times are seconds since the reset (continuous, not steps)."""

    step: int
    pos_w: np.ndarray  # (3,) estimate in the odometry frame (== the world frame: aligned at the reset)
    quat_xyzw: np.ndarray  # (4,) R_z(yaw_est) R_y(pitch) R_x(roll); roll / pitch = the IMU's (true) ones at the output time
    yaw: float
    roll: float  # IMU-true (at the output time)
    pitch: float  # IMU-true (at the output time)
    lin_vel_b: np.ndarray  # (3,) R_est^T v_est_w
    lin_vel_w: np.ndarray  # (3,) velocity estimate (true velocity at the output time + the propagation velocity error; the scan-time velocity under "hold")
    lin_vel_heading: np.ndarray  # (3,) R_z(yaw_est)^T v_est_w (the h23 quantity)
    n_corrections: int  # index of the newest scan whose correction has been applied (0 = only the reset alignment)
    scan_time_s: float  # t_c: scan time of the correction in force (0 = the reset pose)
    arrival_time_s: float  # t_a = t_c + latency: when the correction in force became available (0 for the reset)
    output_time_s: float  # t_o = max(0, t_k - output_delay): the instant the output the policy reads was computed for
    output_age_s: float  # t_k - t_o (== output_delay_s once t_k >= output_delay_s)
    correction_age_s: float  # t_k - t_c: age of the pose the output is anchored to, at the policy tick
    propagation_s: float  # IMU-only window carried by the output: t_o - t_c under "imu", t_o - t_a under "stale", 0 under "hold"
    drift_xy: np.ndarray  # (2,) correction-frame planar error in force (bias + distance drift at t_c)
    drift_z: float  # correction-frame height error in force (bias + walk at t_c)
    drift_yaw: float  # correction-frame yaw error in force (walk at t_c)
    prop_vel_err: np.ndarray  # (3,) IMU-propagation velocity error in force (re-drawn at every correction)
    gyro_yaw_err: float  # gyro bias + white noise integrated over the IMU-only window (rad)
    stance: None = None

    @property
    def z(self) -> float:
        return float(self.pos_w[2])

    @property
    def quat_wxyz(self) -> np.ndarray:
        return xyzw_to_wxyz(self.quat_xyzw)

    def as_dict(self) -> dict[str, Any]:
        return {
            "step": self.step, "pos_w": self.pos_w.tolist(), "quat_xyzw": self.quat_xyzw.tolist(), "yaw": self.yaw, "roll": self.roll, "pitch": self.pitch,
            "lin_vel_b": self.lin_vel_b.tolist(), "lin_vel_w": self.lin_vel_w.tolist(), "lin_vel_heading": self.lin_vel_heading.tolist(),
            "n_corrections": self.n_corrections, "scan_time_s": self.scan_time_s, "arrival_time_s": self.arrival_time_s, "output_time_s": self.output_time_s,
            "output_age_s": self.output_age_s, "correction_age_s": self.correction_age_s, "propagation_s": self.propagation_s,
            "drift_xy": self.drift_xy.tolist(), "drift_z": self.drift_z, "drift_yaw": self.drift_yaw,
            "prop_vel_err": self.prop_vel_err.tolist(), "gyro_yaw_err": self.gyro_yaw_err, "stance": None,
        }


class LidarInertialOdometry:
    """Truth-based error model of a LiDAR-inertial estimator feeding the root-feedback terms (``odom_source="so"``).

    What is modelled (no point clouds -- the estimate is the simulator TRUTH corrupted the way such an estimator's output is):

    * **LiDAR corrections at the scan rate with a processing latency.**  Scan ``n`` is taken at ``t_n = n / rate_hz`` and its corrected pose
      becomes available ``latency_s`` later (``t_a = t_n + latency_s``).  Both instants are honoured EXACTLY: the truth is kept per control
      step and interpolated linearly in between (position / velocity / distance / the walks; roll / pitch / yaw wrap-aware).  The correction of
      scan ``n`` is the TRUE pose at ``t_n`` plus the slow drift of the correction frame (below).  Scan 0 is the reset instant: the estimator is
      initialised at the true reset pose (the reference clip is aligned to the robot there, so the odometry frame IS the world frame and no
      correction is applied for scan 0 -- the alignment absorbs it).
    * **Output age** (``output_delay_s``, default one control step): what the policy reads at its tick ``t_k`` is the estimator output computed
      for ``t_o = t_k - output_delay_s`` (clamped at the reset) -- the transport / publish delay between the estimator and the controller.  It is
      applied uniformly to the whole state (position, roll / pitch / yaw, velocity) whatever the propagation, so on a constant-velocity
      trajectory it costs a lag of ``v x output_delay_s`` under EVERY propagation mode.
    * **Propagation from the correction to the output time** (``propagation``):

      - ``"imu"`` (default): the IMU samples buffered since the scan time are re-applied on top of the arriving correction (the IMU-rate
        output of a LiDAR-inertial pipeline): NO positional lag from the LiDAR latency; the latency only lengthens the IMU-only window
        ``(t_c, t_o]`` the propagation error acts over.
      - ``"stale"`` (the pessimistic pipeline): the arriving correction REPLACES the state with the (latency-old) scan-time pose and the IMU
        dead-reckons forward from the arrival time ``t_a`` -- on a constant-velocity trajectory the output lags by exactly ``v x latency_s``
        (plus ``v x output_delay_s``) and is refreshed at the scan rate.
      - ``"hold"``: the output is the latest corrected pose held until the next one (a consumer of the odometry topic alone, no IMU):
        lag ``v x (latency + scan age + output_delay)``, staircase output.

      The IMU propagation itself is truth-based: the true delta pose over the window plus an **IMU-propagation error** = a velocity error
      ``e_v ~ N(0, prop_vel_std_m_s^2 I_3)`` re-drawn at every correction, growing the position error linearly over the window, and the gyro
      yaw noise (per-episode bias + white noise per control-step sample) integrated over the window.  ``t_ref = t_c`` under ``"imu"``, ``t_a``
      under ``"stale"``; the window is empty under ``"hold"``.
    * **Slow drift of the correction frame** (attached to every correction, evaluated at its scan time): planar position = a per-episode fixed
      offset ``b_xy ~ N(0, bias_xy_m^2)`` per axis + a per-episode drift-RATE vector ``u ~ N(0, I_2) x drift_pct / 100 / sqrt(2)`` times the
      planar distance travelled ``s(t_c)`` (RMS drift magnitude after distance ``s`` is exactly ``drift_pct / 100 x s``); yaw = a Wiener walk of
      ``yaw_walk_deg_sqrt_s``; height = a per-episode offset + a Wiener walk; roll / pitch IMU-true.
    * **Velocity output** (``lin_vel_w``): the true world velocity at ``t_o`` + ``e_v`` (``"imu"`` / ``"stale"``), the true velocity at the scan
      time + ``e_v`` (``"hold"``).

    Assumptions (documented limits of the model): a well-conditioned, feature-rich environment -- NO LiDAR degeneration, NO occlusion /
    dynamic-object corruption, NO map-loss or re-initialisation events, NO extrinsic calibration error, a constant processing latency / output
    age (no jitter) and a constant scan rate; the model is an OPTIMISTIC steady-state error budget, not a failure-mode simulator.  Zero latency
    + zero output delay + zero noise reproduces the truth exactly (tested).

    Streams: ``default_rng(seed + reset count)`` per :meth:`reset` (per-episode draws: ``b_xy``, ``b_z``, ``u``, gyro bias, the first ``e_v``);
    the rollout gives every clip its own ``seed`` (:func:`sim2sim.bench.rollout.clip_odometry_seed`)."""

    quat_convention = QUAT_CONVENTION
    source = "so"

    def __init__(self, cfg: LidarInertialOdometryConfig | Mapping[str, Any] | None = None):
        self.cfg = LidarInertialOdometryConfig.coerce(cfg)
        self.resets = 0
        self._rng = np.random.default_rng(int(self.cfg.seed))
        self._plant = None
        self.state: LioState | None = None
        self.last_error: OdomError | None = None
        self._bias_xy = np.zeros(2)
        self._bias_z = 0.0
        self._drift_rate = np.zeros(2)
        self._gyro_bias = np.zeros(3)
        self._reset_internal(np.zeros(3), np.array([0.0, 0.0, 0.0, 1.0]), None)

    # ------------------------------------------------------------------------------------------ construction
    @classmethod
    def for_plant(cls, plant: Any, cfg: LidarInertialOdometryConfig | Mapping[str, Any] | None = None) -> "LidarInertialOdometry":
        """Estimator over a ``MujocoPlant`` (``dt`` = its control step; the truth is read from its root pose / velocity)."""
        c = LidarInertialOdometryConfig.coerce(cfg)
        c = replace(c, dt=float(getattr(plant, "control_dt", c.dt)))
        self = cls(c)
        self._plant = plant
        return self

    # ------------------------------------------------------------------------------------------ reset
    def reset(self, pos_w: np.ndarray, quat_xyzw: np.ndarray, *, lin_vel_w: np.ndarray | None = None) -> LioState:
        """Initialise at the TRUE pose ``t0`` (the alignment instant = scan 0) and draw the per-episode quantities from ``seed + resets``."""
        cfg = self.cfg
        self.resets += 1
        self._rng = np.random.default_rng(int(cfg.seed) + self.resets)
        self._bias_xy = self._rng.normal(0.0, 1.0, size=2) * float(cfg.bias_xy_m)
        self._bias_z = float(self._rng.normal(0.0, 1.0)) * float(cfg.bias_z_m)
        self._drift_rate = self._rng.normal(0.0, 1.0, size=2) * (float(cfg.drift_pct) / 100.0 / np.sqrt(2.0))
        self._gyro_bias = self._rng.normal(0.0, 1.0, size=3) * float(cfg.gyro_bias_deg_s) * DEG
        return self._reset_internal(pos_w, quat_xyzw, lin_vel_w)

    def reset_from_plant(self, plant: Any | None = None) -> LioState:
        p = plant if plant is not None else self._plant
        if p is None:
            raise ValueError("no plant: build with LidarInertialOdometry.for_plant or pass one")
        return self.reset(p.root_pos, p.root_quat, lin_vel_w=p.root_lin_vel_w)

    def _draw_prop_vel(self) -> np.ndarray:
        std = float(self.cfg.prop_vel_std_m_s)
        return self._rng.normal(0.0, 1.0, size=3) * std if std > 0.0 else np.zeros(3)

    def _reset_internal(self, pos_w, quat_xyzw, lin_vel_w) -> LioState:
        q = quat_normalize(np.asarray(quat_xyzw, dtype=np.float64).reshape(4))
        roll, pitch, yaw = euler_zyx(q)
        p0 = np.array(pos_w, dtype=np.float64, copy=True).reshape(3)
        v0 = np.zeros(3) if lin_vel_w is None else np.asarray(lin_vel_w, dtype=np.float64).reshape(3).copy()
        # truth history indexed by control step (0 = reset, t = step x dt): pose (yaw UNWRAPPED for the interpolation), velocity, planar distance,
        # correction-frame walks, gyro noise integral
        self._hist_p: list[np.ndarray] = [p0.copy()]
        self._hist_roll: list[float] = [float(roll)]
        self._hist_pitch: list[float] = [float(pitch)]
        self._hist_yaw: list[float] = [float(yaw)]
        self._hist_v: list[np.ndarray] = [v0.copy()]
        self._hist_s: list[float] = [0.0]
        self._hist_wyaw: list[float] = [0.0]
        self._hist_wz: list[float] = [0.0]
        self._hist_g: list[float] = [0.0]
        self._n_applied = 0
        self._t_c = 0.0
        self._t_a = 0.0
        self._corr_p = p0.copy()
        self._corr_yaw = float(yaw)
        self._corr_v = v0.copy()
        self._corr_drift_xy = np.zeros(2)
        self._corr_drift_z = 0.0
        self._corr_drift_yaw = 0.0
        self._e_v = self._draw_prop_vel()
        self.step_count = 0
        q_est = quat_rp_yaw(roll, pitch, yaw)
        self.state = LioState(step=0, pos_w=p0.copy(), quat_xyzw=q_est, yaw=float(yaw), roll=float(roll), pitch=float(pitch), lin_vel_b=quat_apply_inv(q_est, v0),
                              lin_vel_w=v0.copy(), lin_vel_heading=quat_apply_inv(yaw_quat(q_est), v0), n_corrections=0, scan_time_s=0.0, arrival_time_s=0.0, output_time_s=0.0,
                              output_age_s=0.0, correction_age_s=0.0, propagation_s=0.0, drift_xy=np.zeros(2), drift_z=0.0, drift_yaw=0.0, prop_vel_err=self._e_v.copy(), gyro_yaw_err=0.0)
        self.last_error = None
        return self.state

    # ------------------------------------------------------------------------------------------ truth interpolation
    def _index(self, t: float) -> tuple[int, float]:
        """History index / fraction of time ``t`` (clamped to ``[0, t_now]``): ``t = (i + f) dt`` with ``0 <= f < 1``."""
        x = max(float(t), 0.0) / float(self.cfg.dt)
        i = int(np.floor(x + _LIO_EPS))
        f = x - i
        if f < _LIO_EPS:
            f = 0.0
        if i >= self.step_count:
            return self.step_count, 0.0
        return i, f

    def _at(self, hist: list, t: float, *, angle: bool = False):
        """Linear interpolation of a per-step history at time ``t`` (arrays or floats; ``angle`` = wrap-aware for a raw angle history)."""
        i, f = self._index(t)
        if f == 0.0:
            return hist[i]
        a, b = hist[i], hist[i + 1]
        if angle:
            return float(a) + f * wrap_angle(float(b) - float(a))
        return a + f * (b - a)

    # ------------------------------------------------------------------------------------------ step
    def step(self, pos_w: np.ndarray, quat_xyzw: np.ndarray, lin_vel_w: np.ndarray | None = None) -> LioState:
        """Advance one control step with the TRUE root pose / world velocity of this instant; returns the estimate the policy may observe."""
        cfg = self.cfg
        dt = float(cfg.dt)
        k = self.step_count + 1
        self.step_count = k
        t_k = k * dt
        p = np.asarray(pos_w, dtype=np.float64).reshape(3).copy()
        q = quat_normalize(np.asarray(quat_xyzw, dtype=np.float64).reshape(4))
        roll, pitch, yaw = euler_zyx(q)
        v = np.zeros(3) if lin_vel_w is None else np.asarray(lin_vel_w, dtype=np.float64).reshape(3).copy()
        # ---- truth history + the correction-frame / gyro processes (Wiener increments sqrt(dt); drawn only when their std is > 0) ----
        self._hist_p.append(p)
        self._hist_roll.append(float(roll))
        self._hist_pitch.append(float(pitch))
        self._hist_yaw.append(self._hist_yaw[-1] + wrap_angle(float(yaw) - self._hist_yaw[-1]))  # unwrapped
        self._hist_v.append(v)
        self._hist_s.append(self._hist_s[-1] + float(np.linalg.norm((p - self._hist_p[-2])[:2])))
        wy = float(cfg.yaw_walk_deg_sqrt_s) * DEG
        self._hist_wyaw.append(self._hist_wyaw[-1] + (float(self._rng.normal(0.0, 1.0)) * wy * np.sqrt(dt) if wy > 0.0 else 0.0))
        wz = float(cfg.z_walk_m_sqrt_s)
        self._hist_wz.append(self._hist_wz[-1] + (float(self._rng.normal(0.0, 1.0)) * wz * np.sqrt(dt) if wz > 0.0 else 0.0))
        g = float(self._gyro_bias[2]) * dt
        if float(cfg.gyro_white_deg_s) > 0.0:
            g += float(self._rng.normal(0.0, 1.0)) * float(cfg.gyro_white_deg_s) * DEG * dt
        self._hist_g.append(self._hist_g[-1] + g)
        # ---- the output the policy reads was computed for t_o = t_k - output_delay (clamped at the reset) ----
        t_o = max(t_k - float(cfg.output_delay_s), 0.0)
        # ---- a new correction has arrived by t_o: the true pose at its scan time + the correction-frame drift evaluated there ----
        n = cfg.latest_available_scan(t_o)
        if n > self._n_applied:
            self._n_applied = n
            t_c = cfg.scan_time(n)
            self._t_c, self._t_a = t_c, cfg.arrival_time(n)
            self._corr_drift_xy = self._bias_xy + self._drift_rate * float(self._at(self._hist_s, t_c))
            self._corr_drift_z = self._bias_z + float(self._at(self._hist_wz, t_c))
            self._corr_drift_yaw = float(self._at(self._hist_wyaw, t_c))
            self._corr_p = self._at(self._hist_p, t_c) + np.array([self._corr_drift_xy[0], self._corr_drift_xy[1], self._corr_drift_z])
            self._corr_yaw = float(self._at(self._hist_yaw, t_c)) + self._corr_drift_yaw
            self._corr_v = np.asarray(self._at(self._hist_v, t_c), dtype=np.float64).copy()
            self._e_v = self._draw_prop_vel()
        # ---- output: the correction carried to t_o ----
        mode = cfg.propagation
        roll_o = float(self._at(self._hist_roll, t_o, angle=True))
        pitch_o = float(self._at(self._hist_pitch, t_o, angle=True))
        if mode == "hold":
            p_est = self._corr_p.copy()
            yaw_est = self._corr_yaw
            v_est = self._corr_v + self._e_v
            g_err = 0.0
            window = 0.0
        else:
            t_ref = self._t_c if mode == "imu" else self._t_a
            window = t_o - t_ref
            if window < _LIO_EPS:  # an arrival exactly on the grid: k x dt and n / rate + latency agree only to float dust -> an empty window
                window = 0.0
            g_err = float(self._at(self._hist_g, t_o)) - float(self._at(self._hist_g, t_ref))
            p_est = self._corr_p + (self._at(self._hist_p, t_o) - self._at(self._hist_p, t_ref)) + self._e_v * window
            yaw_est = self._corr_yaw + (float(self._at(self._hist_yaw, t_o)) - float(self._at(self._hist_yaw, t_ref))) + g_err
            v_est = np.asarray(self._at(self._hist_v, t_o), dtype=np.float64) + self._e_v
        yaw_est = wrap_angle(yaw_est)
        q_est = quat_rp_yaw(roll_o, pitch_o, yaw_est)
        self.state = LioState(step=k, pos_w=np.asarray(p_est, dtype=np.float64).copy(), quat_xyzw=q_est, yaw=float(yaw_est), roll=roll_o, pitch=pitch_o, lin_vel_b=quat_apply_inv(q_est, v_est),
                              lin_vel_w=np.asarray(v_est, dtype=np.float64).copy(), lin_vel_heading=quat_apply_inv(yaw_quat(q_est), v_est), n_corrections=int(self._n_applied),
                              scan_time_s=float(self._t_c), arrival_time_s=float(self._t_a), output_time_s=float(t_o), output_age_s=float(t_k - t_o), correction_age_s=float(t_k - self._t_c),
                              propagation_s=float(window), drift_xy=self._corr_drift_xy.copy(), drift_z=float(self._corr_drift_z), drift_yaw=float(self._corr_drift_yaw),
                              prop_vel_err=self._e_v.copy(), gyro_yaw_err=float(g_err))
        return self.state

    def step_from_plant(self, plant: Any | None = None) -> LioState:
        p = plant if plant is not None else self._plant
        if p is None:
            raise ValueError("no plant: build with LidarInertialOdometry.for_plant or pass one")
        return self.step(p.root_pos, p.root_quat, getattr(p, "root_lin_vel_w", None))

    # ------------------------------------------------------------------------------------------ error / provenance
    def error_vs_truth(self, pos_w: np.ndarray, quat_xyzw: np.ndarray, lin_vel_w: np.ndarray | None = None) -> OdomError:
        """Estimate - truth of the CURRENT state (also stored in ``last_error``)."""
        if self.state is None:
            raise RuntimeError("no estimate yet")
        self.last_error = estimate_error(self.state, pos_w, quat_xyzw, lin_vel_w)
        return self.last_error

    def describe(self) -> dict[str, Any]:
        cfg = self.cfg
        return {"kind": "lidar_inertial_odometry_v2", "source": self.source, "quat_convention": self.quat_convention, "config": cfg.as_dict(), "propagation": cfg.propagation,
                "latency_s": float(cfg.latency_s), "output_delay_s": float(cfg.output_delay_s), "scan_period_s": cfg.scan_period_s, "timing": "continuous (truth interpolated between control steps; no grid snapping)",
                "model": "truth-based: true pose at the last LiDAR correction's scan time + correction-frame drift (per-episode offset + drift-rate x distance, yaw / z Wiener walks) + "
                         "IMU propagation to the output time (true delta pose + velocity-error x window + gyro yaw noise), read output_delay_s late; roll / pitch IMU-true; no degeneration / occlusion / extrinsic error",
                "n_corrections": (int(self.state.n_corrections) if self.state is not None else 0)}


# ================================================================================================ source dispatch
ODOMETRY_CONFIG_TYPES: tuple[type, ...] = (LidarInertialOdometryConfig,)


def config_source(cfg: Any) -> str:
    """The ``ODOM_SOURCES`` entry an estimator config implements: ``"so"`` for :class:`LidarInertialOdometryConfig`."""
    if isinstance(cfg, LidarInertialOdometryConfig):
        return "so"
    raise TypeError(f"not an odometry config: {type(cfg).__name__} (expected one of {[t.__name__ for t in ODOMETRY_CONFIG_TYPES]})")


def default_odometry_config(source: str) -> LidarInertialOdometryConfig | None:
    """The estimator config a controller with ``odom_source == source`` gets when the rollout carries none: None for ``"truth"``, the ``so``
    preset for ``"so"``; ``"leg"`` has no estimator in this package (raise)."""
    if source not in ODOM_SOURCES:
        raise ValueError(f"odom source must be one of {ODOM_SOURCES}, got {source!r}")
    if source == "truth":
        return None
    if source == "leg":
        raise ValueError("odom_source 'leg' (leg odometry) is not part of the benchmark runner; use 'truth' or 'so'")
    return LidarInertialOdometryConfig.preset("so")


def build_odometry(plant: Any, cfg: LidarInertialOdometryConfig) -> LidarInertialOdometry:
    """The estimator of a config on a plant (``dt`` from ``plant.control_dt``)."""
    config_source(cfg)
    return LidarInertialOdometry.for_plant(plant, cfg)


__all__ = [
    "DEG",
    "GYRO_NOISE",
    "LIO_PRESETS",
    "LIO_PRESET_VALUES",
    "LIO_PROPAGATIONS",
    "ODOMETRY_CONFIG_TYPES",
    "ODOM_SOURCES",
    "QUAT_CONVENTION",
    "LidarInertialOdometry",
    "LidarInertialOdometryConfig",
    "LioState",
    "OdomError",
    "build_odometry",
    "config_source",
    "default_odometry_config",
    "estimate_error",
    "euler_zyx",
    "quat_rp_yaw",
    "wrap_angle",
]
