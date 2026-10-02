"""Closed-loop replanning + goal adjustment (sim2sim.bench.replan): decision logic, in-memory references, handover smoothing, the hero_bench_v1
stop-frame / hand-back logic, the runner's flags and -- when mink and a local hero_bench_v1 corpus are present -- the online whole-body replanner."""
from __future__ import annotations

import json
import warnings

import numpy as np
import pytest

from hero_isaacsim import constants as HC
from sim2sim.bench.replan import (
    START_POSES,
    START_VELOCITIES,
    BenchGoal,
    BenchReplanner,
    GoalAdjuster,
    Handover,
    ReplanConfig,
    blend_handover_frames,
    clip_has_own_retract,
    clip_replan_config,
    hold_end_frame_of,
    load_bench_manifest,
    manifest_entry,
    palm_path,
    quintic_step,
    quintic_velocity_carry,
    ref_palm_velocity_w,
    resume_reference,
    slerp_wxyz,
)
from sim2sim.reference import ClipReference

from _bench_fixtures import BENCH_DIR, bench_clips, has_mink, has_mujoco, needs_bench, needs_mink


def _default_mjcf_exists() -> bool:
    try:
        from data_tools import hero_reach_generator as gen

        return gen.DEFAULT_MJCF.is_file()
    except Exception:  # noqa: BLE001
        return False


# ================================================================================================ GoalAdjuster
def test_config_validation():
    with pytest.raises(ValueError):
        ReplanConfig(base_mode="nope")
    with pytest.raises(ValueError):
        ReplanConfig(adjust_mode="paper")
    with pytest.raises(ValueError):
        ReplanConfig(period_s=0.0)
    assert ReplanConfig(period_s=0.2).period_frames == 10


def test_adjuster_stay_skip_replan_thresholds():
    cfg = ReplanConfig(goal_adjust=False)
    g = np.array([0.4, -0.2, 0.9])
    a = GoalAdjuster(g, cfg)
    assert a.decide(g + [0.01, 0, 0], 100) == "stay"
    assert a.stayed_at == 100
    assert a.decide(g + [0.10, 0, 0], 150) == "stay"
    b = GoalAdjuster(g, cfg)
    assert b.decide(g + [0.019, 0, 0], 10) == "skip"
    assert b.n_replans == 0
    assert b.decide(g + [0.05, 0, 0], 60) == "replan"
    assert b.n_replans == 1 and np.allclose(b.goal, g)
    assert [e["action"] for e in b.events] == ["skip", "replan"]


def test_adjuster_accumulate_clamp_and_gate():
    cfg = ReplanConfig(goal_adjust=True, adjust_gain=0.6, adjust_max_step_m=0.01, adjust_gate_m=0.15, adjust_mode="accumulate")
    g = np.zeros(3)
    a = GoalAdjuster(g, cfg)
    assert a.decide(np.array([0.30, 0, 0]), 1) == "replan"     # outside the 15 cm gate and fewer than 3 replans -> no adjustment
    assert np.allclose(a.goal, g)
    assert a.decide(np.array([0.05, 0, 0]), 2) == "replan"     # 0.6 * 5 = 3 cm -> clamped to 1 cm
    assert np.allclose(a.goal, [-0.01, 0, 0])
    assert a.decide(np.array([0.025, 0, 0]), 3) == "replan"
    assert np.allclose(a.goal, [-0.02, 0, 0])
    assert a.n_replans == 3
    assert a.decide(np.array([0.021, 0, 0]), 4) == "replan"
    assert np.allclose(a.goal, [-0.03, 0, 0])
    assert a.decide(np.array([0.40, 0, 0]), 5) == "replan"     # replan_times > 2 -> adjustment applies outside the gate
    assert np.allclose(a.goal, [-0.04, 0, 0])
    s = a.summary()
    assert s["n_replans"] == 5 and abs(s["goal_offset_cm"] - 4.0) < 1e-9


def test_adjuster_reset_mode_and_max_replans():
    a = GoalAdjuster(np.zeros(3), ReplanConfig(goal_adjust=True, adjust_mode="reset"))
    a.decide(np.array([0.05, 0, 0]), 1)
    a.decide(np.array([0.05, 0, 0]), 2)
    assert np.allclose(a.goal, [-0.01, 0, 0])
    b = GoalAdjuster(np.zeros(3), ReplanConfig(max_replans=2))
    assert [b.decide(np.array([0.1, 0, 0]), k) for k in range(4)] == ["replan", "replan", "skip", "skip"]


# ================================================================================================ handover smoothing (pure)
def test_handover_config_defaults_and_validation():
    cfg = ReplanConfig()
    assert (cfg.handover_blend_s, cfg.start_velocity, cfg.start_pose) == (0.0, "zero", "robot")
    assert cfg.blend_frames == 0 and not cfg.needs_live_reference
    assert ReplanConfig(handover_blend_s=0.3).blend_frames == 15 and ReplanConfig(handover_blend_s=0.3).needs_live_reference
    assert ReplanConfig(start_velocity="match").needs_live_reference and ReplanConfig(start_pose="reference").needs_live_reference
    assert START_VELOCITIES == ("zero", "match") and START_POSES == ("robot", "reference")
    for bad in (dict(start_velocity="carry"), dict(start_pose="old"), dict(handover_blend_s=-0.1)):
        with pytest.raises(ValueError):
            ReplanConfig(**bad)
    d = ReplanConfig(handover_blend_s=0.2, start_velocity="match", start_pose="reference").as_dict()
    assert (d["handover_blend_s"], d["start_velocity"], d["start_pose"], d["stop_frame"]) == (0.2, "match", "reference", None)


def test_quintic_profiles_are_the_hermite_polynomials():
    from numpy.polynomial import polynomial as P

    s5 = np.array([0, 0, 0, 10, -15, 6], dtype=float)
    h1 = np.array([0, 1, 0, -6, 8, -3], dtype=float)
    u = np.linspace(0, 1, 101)
    assert np.allclose([quintic_step(x) for x in u], P.polyval(u, s5), atol=1e-12)
    assert np.allclose([quintic_velocity_carry(x) for x in u], P.polyval(u, h1), atol=1e-12)
    assert np.allclose(P.polyval([0.0, 1.0], P.polyder(s5)), [0.0, 0.0]) and np.allclose(P.polyval([0.0, 1.0], P.polyder(h1)), [1.0, 0.0])
    assert np.allclose(P.polyval([0.0, 1.0], P.polyder(s5, 2)), [0.0, 0.0]) and np.allclose(P.polyval([0.0, 1.0], P.polyder(h1, 2)), [0.0, 0.0])
    assert quintic_step(-1.0) == 0.0 and quintic_step(2.0) == 1.0 and quintic_velocity_carry(2.0) == 0.0


def test_palm_path_zero_bit_identical_and_match_carries_velocity():
    from data_tools import reach_specs as rs

    p0, p1 = np.array([0.30, -0.20, 0.90]), np.array([0.52, -0.08, 0.80])
    q0, q1 = np.array([1.0, 0.0, 0.0, 0.0]), rs.mat_to_quat_wxyz(rs.rot_z(0.6) @ rs.rot_y(-0.3))
    n_reach, n_total = 30, 36
    pos, quat = palm_path(p0, q0, p1, q1, n_reach, n_total)
    exp = np.array([p0 + (p1 - p0) * rs.smoothstep(min(t / n_reach, 1.0)) for t in range(n_total)])
    assert np.array_equal(pos, exp)
    assert np.allclose(quat[0], q0) and abs(abs(float(np.dot(quat[-1], q1))) - 1.0) < 1e-12 and np.allclose(pos[n_reach:], p1)
    v0 = np.array([0.25, -0.10, 0.05])
    posm, quatm = palm_path(p0, q0, p1, q1, n_reach, n_total, start_velocity="match", v0_w=v0)
    assert np.allclose(posm[0], p0) and np.allclose(posm[n_reach:], p1) and np.allclose(quatm[0], q0)
    assert np.linalg.norm((posm[1] - posm[0]) * 50.0 - v0) < 0.02 * np.linalg.norm(v0) + 5e-3
    fine_pos, _ = palm_path(p0, q0, p1, q1, 3000, 3001, start_velocity="match", v0_w=v0, fps=5000)
    assert np.linalg.norm((fine_pos[1] - fine_pos[0]) * 5000.0 - v0) < 1e-4 and np.linalg.norm((fine_pos[3000] - fine_pos[2999]) * 5000.0) < 1e-4
    with pytest.raises(ValueError):
        palm_path(p0, q0, p1, q1, n_reach, n_total, start_velocity="carry")


def test_slerp_wxyz_endpoints_shortest_arc_and_reference_impl():
    from data_tools import reach_specs as rs

    rng = np.random.default_rng(0)
    q0 = rng.normal(size=(6, 4)); q0 /= np.linalg.norm(q0, axis=-1, keepdims=True)
    q1 = rng.normal(size=(6, 4)); q1 /= np.linalg.norm(q1, axis=-1, keepdims=True)
    q1[2] = -q1[2] if np.dot(q0[2], q1[2]) > 0 else q1[2]
    assert np.array_equal(slerp_wxyz(q0, q1, 0.0), q0) and np.array_equal(slerp_wxyz(q0, q1, 1.0), q1)
    mid = slerp_wxyz(q0, q1, 0.5)
    ang = lambda a, b: 2.0 * np.arccos(np.clip(np.abs(np.sum(a * b, axis=-1)), 0.0, 1.0))  # noqa: E731
    assert np.allclose(np.linalg.norm(mid, axis=-1), 1.0)
    assert np.allclose(ang(q0, mid), ang(mid, q1), atol=1e-9) and np.all(ang(q0, mid) <= 0.5 * ang(q0, q1) + 1e-9)
    for w in (0.3, 0.7):
        ref = rs.quat_slerp_wxyz(q0[2], q1[2], w)
        assert np.allclose(slerp_wxyz(q0[2], q1[2], w), ref) or np.allclose(slerp_wxyz(q0[2], q1[2], w), -ref)


def _synthetic_arrays(T: int, seed: int, vel: np.ndarray | None = None) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    rng = np.random.default_rng(seed)
    v = np.zeros(3) if vel is None else np.asarray(vel, dtype=np.float64)
    t = np.arange(T, dtype=np.float64)[:, None] / 50.0
    jp = np.zeros((T, 36))
    jp[:, 0:3] = np.array([0.0, 0.0, 0.75]) + t * v
    jp[:, 3:7] = np.array([1.0, 0.0, 0.0, 0.0])
    jp[:, 7:] = 0.4 * rng.normal(size=29) + t * 0.05 * rng.normal(size=29)
    base = rng.normal(size=(32, 3)) * 0.3
    bp = base[None] + t[:, :, None] * v[None, None, :]
    q = rng.normal(size=(32, 4)); q /= np.linalg.norm(q, axis=-1, keepdims=True)
    bq = np.repeat(q[None], T, axis=0)
    return jp, bp, bq


def _synthetic_reference(T: int, seed: int, vel: np.ndarray | None = None, name: str = "h050__synthetic.npz") -> ClipReference:
    jp, bp, bq = _synthetic_arrays(T, seed, vel)
    return ClipReference.from_arrays(fps=50, joint_pos=jp, body_pos_w=bp, body_quat_w=bq, name=name)


def test_ref_palm_velocity_and_handover_sampling():
    v = np.array([0.20, -0.10, 0.05])
    old = _synthetic_reference(40, seed=3, vel=v)
    assert np.allclose(ref_palm_velocity_w(old, 10), np.tile(v, (2, 1)), atol=1e-9)
    assert np.allclose(ref_palm_velocity_w(old, 0), np.tile(v, (2, 1)), atol=1e-9)
    assert np.allclose(ref_palm_velocity_w(old.padded(20), 50), 0.0)
    h = Handover.at(old, 10)
    p, q = old.palm_pose_w(10)
    assert h.t == 10 and np.allclose(h.palm_pos_w, p) and np.allclose(h.palm_quat_wxyz, np.roll(q, 1, axis=1)) and np.allclose(h.palm_vel_w, np.tile(v, (2, 1)))
    assert np.array_equal(h.dof_pos, old.joint_pos[10]) and h.summary(1)["palm_speed_cm_s"] == pytest.approx(np.linalg.norm(v) * 100.0)
    assert Handover.at(old, 99).t == 39


def test_velocity_matched_path_is_continuous_with_the_old_reference():
    v = np.array([0.18, -0.12, 0.04])
    old = _synthetic_reference(40, seed=5, vel=v)
    h = Handover.at(old, 10)
    goal = h.palm_pos_w[1] + np.array([0.25, 0.05, -0.10])
    goal_q = np.array([1.0, 0.0, 0.0, 0.0])
    v_before = (old.palm_pose_w(10)[0][1] - old.palm_pose_w(9)[0][1]) * 50.0
    pos_m, _ = palm_path(h.palm_pos_w[1], h.palm_quat_wxyz[1], goal, goal_q, 25, 30, start_velocity="match", v0_w=h.palm_vel_w[1])
    pos_z, _ = palm_path(h.palm_pos_w[1], h.palm_quat_wxyz[1], goal, goal_q, 25, 30, start_velocity="zero")
    jump_m = np.linalg.norm((pos_m[1] - pos_m[0]) * 50.0 - v_before)
    jump_z = np.linalg.norm((pos_z[1] - pos_z[0]) * 50.0 - v_before)
    assert jump_m < 0.015 and jump_z > 0.10 and jump_z > 5.0 * jump_m


def test_blend_handover_frames_endpoints_weights_and_untouched_tail():
    from sim2sim.mathutil import xyzw_to_wxyz

    old = _synthetic_reference(40, seed=1, vel=np.array([0.1, 0.0, 0.0]))
    new_jp, new_bp, new_bq = _synthetic_arrays(60, seed=2)
    jp, bp, bq = new_jp.copy(), new_bp.copy(), new_bq.copy()
    assert blend_handover_frames(old, 7, jp, bp, bq, 0) == 0 and np.array_equal(jp, new_jp)
    n = blend_handover_frames(old, 7, jp, bp, bq, 10)
    assert n == 11
    assert np.array_equal(jp[0, 7:], old.joint_pos[7]) and np.array_equal(jp[0, 0:3], old.root_pos_from_joint_pos[7])
    assert np.array_equal(bp[0], old.body_pos_w[7]) and np.array_equal(bq[0], xyzw_to_wxyz(old.body_quat_w[7]))
    assert np.array_equal(jp[10:], new_jp[10:]) and np.array_equal(bp[10:], new_bp[10:]) and np.array_equal(bq[10:], new_bq[10:])
    for i in range(1, 10):
        w = i / 10.0
        assert np.allclose(jp[i, 7:], (1 - w) * old.joint_pos[7 + i] + w * new_jp[i, 7:])
        assert np.allclose(bp[i], (1 - w) * old.body_pos_w[7 + i] + w * new_bp[i])
        assert np.allclose(np.linalg.norm(bq[i], axis=-1), 1.0)
    ref = ClipReference.from_arrays(fps=50, joint_pos=jp, body_pos_w=bp, body_quat_w=bq, name="h050__blend.npz")
    assert np.allclose(ref.palm_pose_w(0)[0], old.palm_pose_w(7)[0], atol=1e-9)
    jp2, bp2, bq2 = _synthetic_arrays(6, seed=8)
    orig2 = jp2.copy()
    assert blend_handover_frames(old, 38, jp2, bp2, bq2, 10) == 6
    assert np.array_equal(jp2[0, 7:], old.joint_pos[38]) and np.allclose(jp2[3, 7:], 0.7 * old.joint_pos[39] + 0.3 * orig2[3, 7:])


def test_replanner_requires_reference_for_handover_knobs():
    goal = BenchGoal(clip_name="x", hand="right", pos_w=np.zeros(3), quat_wxyz=np.array([1.0, 0, 0, 0]), table={"center": [0.6, 0, 0.7], "half_size": [0.3, 0.7, 0.02], "surface_z": 0.72},
                     reach_end_frame=10, pelvis_drop=0.0, pelvis_pitch=0.0, waist_pitch=0.0, base_family="stand", height_label="h074")
    for cfg in (ReplanConfig(handover_blend_s=0.3), ReplanConfig(start_velocity="match"), ReplanConfig(start_pose="reference")):
        with pytest.raises(ValueError, match="reference="):
            BenchReplanner(goal, cfg)  # refused before any model is built


def test_jerk_summary_windows():
    from sim2sim.bench.rollout import JERK_METRIC_KEYS, POST_REPLAN_WINDOW_S, jerk_summary

    assert JERK_METRIC_KEYS == ("arm_target_delta_rad", "arm_accel_rad_s2", "ref_palm_speed_cm_s") and POST_REPLAN_WINDOW_S == 0.3
    K = 100
    m = {k: np.full(200, np.nan, dtype=np.float32) for k in JERK_METRIC_KEYS}
    for k in m:
        m[k][:K] = 0.0
    m["arm_target_delta_rad"][20:35] = 1.0
    m["arm_accel_rad_s2"][94:100] = 2.0
    s = jerk_summary(m, K, [21, 95], 0.02)
    assert s["window_steps"] == 15 and s["n_replans"] == 2 and s["window_s"] == 0.3
    assert s["post_replan"]["arm_target_delta_rad"] == pytest.approx(15.0 / 21.0) and s["mean"]["arm_target_delta_rad"] == pytest.approx(0.15)
    assert s["post_replan"]["arm_accel_rad_s2"] == pytest.approx(6.0 * 2.0 / 21.0) and s["post_replan"]["ref_palm_speed_cm_s"] == 0.0
    s0 = jerk_summary(m, K, [], 0.02)
    assert all(v is None for v in s0["post_replan"].values()) and s0["mean"]["arm_target_delta_rad"] == pytest.approx(0.15)
    assert jerk_summary({}, K, [1], 0.02)["mean"] == {}


def test_cli_flags_reach_the_configs():
    from sim2sim.bench import run as R

    base = ["--motion-dir", "x", "--out", "y", "--replan"]
    d = R._replan_config(R.build_parser().parse_args(base))
    assert d == ReplanConfig() and not d.needs_live_reference
    c = R._replan_config(R.build_parser().parse_args(base + ["--replan-blend-s", "0.3", "--replan-start-velocity", "match", "--replan-start-pose", "reference",
                                                           "--replan-first", "reach_end", "--replan-period-s", "3.0", "--replan-base", "current", "--goal-adjust"]))
    assert (c.handover_blend_s, c.start_velocity, c.start_pose, c.blend_frames) == (0.3, "match", "reference", 15)
    assert (c.first_event, c.period_s, c.base_mode, c.goal_adjust, c.period_frames) == ("reach_end", 3.0, "current", True, 150)
    assert R._replan_config(R.build_parser().parse_args(["--motion-dir", "x", "--out", "y"])) is None
    a = R.build_parser().parse_args(["--motion-dir", "x", "--out", "y", "--pad-s", "4", "--horizon-s", "14", "--odom", "so", "--odom-seed", "7"])
    assert R.pad_steps_for(a, 0.02) == 200 and R.odometry_config(a).seed == 7 and R.odometry_config(a).latency_s == 0.03
    assert R.odometry_config(R.build_parser().parse_args(["--motion-dir", "x", "--out", "y"])) is None
    for bad in (["--replan-start-velocity", "carry"], ["--odom", "leg"], ["--policy", "sonic_v10"]):
        with pytest.raises(SystemExit):
            R.build_parser().parse_args(["--motion-dir", "x", "--out", "y"] + bad)


# ================================================================================================ hero_bench_v1 rows: table None / stop_frame / own retract
def _row(**kw) -> dict:
    e = {"file": "floor_pick__fp_000001.npz", "clip_id": "fp_000001", "hand": "left", "target_pos_w": [0.40, 0.20, 0.30], "target_yaw_deg": 10.0, "target_pitch_deg": -5.0,
         "target_roll_deg": 0.0, "table": None, "reach_end_frame": 120, "hold_frames": 150, "n_frames": 270, "height_label": "floor", "stratum": "floor_pick", "tier": "extended",
         "base_family": "squat", "pelvis_drop": 0.30, "pelvis_pitch_deg": 20.0, "waist_pitch_deg": 25.0}
    e.update(kw)
    return e


def test_bench_goal_accepts_table_none_and_legacy_edge_rows():
    g = BenchGoal.from_manifest_entry(_row())
    assert g.table is None and g.hand == "left" and g.reach_end_frame == 120 and g.height_label == "floor" and g.hand_index == 0 and g.active_hands == (True, False)
    g2 = BenchGoal.from_manifest_entry(_row(table=None, table_edge_x=0.32, table_top_z=0.74, file="h074__x.npz"))
    assert g2.table is not None and g2.table["surface_z"] == 0.74 and g2.table["center"][0] == pytest.approx(0.62)
    tb = {"center": [0.6, 0.1, 0.48], "half_size": [0.3, 0.7, 0.02], "surface_z": 0.5}
    assert BenchGoal.from_manifest_entry(_row(table=tb)).table == tb
    assert hold_end_frame_of(_row()) == 270 and hold_end_frame_of(_row(hold_end_frame=260)) == 260 and hold_end_frame_of({"reach_end_frame": 10}) is None
    with pytest.raises(ValueError):
        BenchGoal.from_manifest_entry(_row(hand="both"))


def test_clip_replan_config_guard_for_own_retract_clips():
    cfg = ReplanConfig()
    plain = _row()
    assert not clip_has_own_retract(plain) and clip_replan_config(cfg, plain) is cfg
    rt = _row(has_retract=True, hold_end_frame=270, retract_start_frame=270, n_frames=400, stratum="retract")
    assert clip_has_own_retract(rt) and clip_has_own_retract(_row(stratum="retract")) and clip_has_own_retract(_row(retract={"start_frame": 270}))
    c = clip_replan_config(cfg, rt)
    assert c.stop_frame == 270
    assert clip_replan_config(cfg, _row(has_retract=True, hold_end_frame=250)).stop_frame == 250
    assert clip_replan_config(ReplanConfig(stop_frame=100), rt).stop_frame == 100          # an earlier explicit stop wins
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert clip_replan_config(cfg, {"file": "retract__a.npz", "has_retract": True, "hand": "left"}) is cfg   # no frame to stop at: warn, keep going
        assert sum("replans are not stopped" in str(x.message) for x in w) == 1
    with pytest.raises(ValueError):
        ReplanConfig(stop_frame=0)
    assert ReplanConfig().as_dict()["stop_frame"] is None


def _synthetic_ref(T: int = 80, name: str = "retract__r_000001.npz") -> ClipReference:
    names = list(HC.HOLOSOMA_BODY_NAMES_32)
    bp = np.zeros((T, 32, 3))
    bp[:, :, 2] = 0.75
    bp[:, :, 0] = 0.01 * np.arange(T)[:, None]
    for n, dy in (("left_wrist_yaw_link", 0.2), ("right_wrist_yaw_link", -0.2)):
        bp[:, names.index(n), 0] += 0.4
        bp[:, names.index(n), 1] = dy
    bq = np.zeros((T, 32, 4))
    bq[:, :, 0] = 1.0
    jp = np.concatenate([bp[:, 0], bq[:, 0], np.zeros((T, 29))], axis=1)
    return ClipReference.from_arrays(fps=50, joint_pos=jp, body_pos_w=bp, body_quat_w=bq, name=name)


class _State:
    def __init__(self, palm_pos_w):
        self.palm_pos_w = np.asarray(palm_pos_w, dtype=np.float64)


def _stub_replanner(cfg: ReplanConfig, ref: ClipReference, monkeypatch) -> BenchReplanner:
    """A BenchReplanner without the IK bundle: ``plan_from_state`` is replaced by a synthetic 60-frame plan (the schedule / stop logic is real)."""
    rp = BenchReplanner.__new__(BenchReplanner)
    rp.goal = BenchGoal(clip_name=ref.name, hand="left", pos_w=np.array([0.5, 0.2, 0.75]), quat_wxyz=np.array([1.0, 0, 0, 0]), table=None, reach_end_frame=10,
                        pelvis_drop=0.0, pelvis_pitch=0.0, waist_pitch=0.0, base_family="stand", height_label="h074")
    rp.cfg, rp.reference, rp.arm_posture = cfg, ref, None
    rp.reset()
    calls: list[int] = []

    def fake_plan(st, plant, goal_pos_w, *, old_ref=None, old_t=0):
        calls.append(1)
        return _synthetic_ref(60, name=ref.name), {"terminal_palm_err_cm": 0.1, "max_palm_path_err_cm": 0.0, "qp_failures": 0, "frames": 60}

    monkeypatch.setattr(rp, "plan_from_state", fake_plan)
    rp._fake_calls = calls
    return rp


def test_stop_frame_stops_replans_and_hands_back_to_the_original_clip(monkeypatch):
    ref = _synthetic_ref(80)
    st = _State(np.array([[0.55, 0.2, 0.75], [0.4, -0.2, 0.75]]))   # 5 cm error on the left (active) hand -> every event replans
    base = dict(period_s=0.2, first_event="start")
    free = _stub_replanner(ReplanConfig(**base), ref, monkeypatch)
    stop = _stub_replanner(ReplanConfig(stop_frame=30, **base), ref, monkeypatch)
    swaps_free = [k for k in range(1, 50) if free.step(k, st, None) is not None]
    swaps_stop: dict[int, ClipReference] = {}
    for k in range(1, 50):
        r = stop.step(k, st, None)
        if r is not None:
            swaps_stop[k] = r
    assert swaps_free == [1, 11, 21, 31, 41]
    assert sorted(swaps_stop) == [1, 11, 21, 30] and len(stop._fake_calls) == 3
    hb = swaps_stop[30]
    assert stop.stopped_at == 30 and hb.T == ref.T - 30
    np.testing.assert_allclose(hb.root_pos_w[0], ref.root_pos_w[30])
    np.testing.assert_allclose(hb.palm_pose_w(0)[0], ref.palm_pose_w(30)[0])
    s = stop.summary()
    assert s["stop"]["stop_frame"] == 30 and s["stop"]["stopped_at_step"] == 30 and s["stop"]["handback"]["handed_back"] and s["n_replans"] == 3
    assert s["stop"]["handback"]["palm_jump_cm"] == pytest.approx(100.0 * np.linalg.norm(ref.palm_pose_w(30)[0][0] - _synthetic_ref(60).palm_pose_w(9)[0][0]))
    assert "stop" not in free.summary()
    assert json.dumps(s)
    quiet = _stub_replanner(ReplanConfig(stop_frame=5, **base), ref, monkeypatch)
    st_ok = _State(np.array([[0.5, 0.2, 0.75], [0.4, -0.2, 0.75]]))
    assert all(quiet.step(k, st_ok, None) is None for k in range(1, 10))
    assert quiet.summary()["stop"]["handback"] == {"step": 5, "handed_back": False, "reason": "controller already on the original clip"}
    blend = _stub_replanner(ReplanConfig(stop_frame=30, handover_blend_s=0.2, **base), ref, monkeypatch)
    out = {k: blend.step(k, st, None) for k in range(1, 31)}
    hb2 = out[30]
    old_plan, old_t = out[21], 9
    np.testing.assert_allclose(hb2.root_pos_w[0], old_plan.root_pos_w[old_t])
    np.testing.assert_allclose(hb2.root_pos_w[10], ref.root_pos_w[40])
    assert blend.summary()["stop"]["handback"]["blended_frames"] == 11
    rr = resume_reference(ref, 70)
    assert rr.T == 10 and rr.name == ref.name and np.allclose(rr.root_pos_w[0], ref.root_pos_w[70])


# ================================================================================================ references + the online replanner (mink + local corpus)
@needs_bench
def test_reference_from_arrays_roundtrip_and_padding():
    ref = ClipReference(bench_clips()[0])
    jp = np.concatenate([ref.root_pos_from_joint_pos, np.roll(ref.root_quat_from_joint_pos, 1, axis=1), ref.joint_pos], axis=1)
    bq = np.roll(ref.body_quat_w, 1, axis=2)
    a = ClipReference.from_arrays(fps=ref.fps, joint_pos=jp, body_pos_w=ref.body_pos_w, body_quat_w=bq, name="h050__x.npz")
    assert a.T == ref.T and a.source_tag == "h050"
    assert np.abs(a.joint_pos - ref.joint_pos).max() < 1e-9 and np.abs(a.body_quat_w - ref.body_quat_w).max() < 1e-9
    p_a, q_a = a.palm_pose_own_pelvis(ref.T - 1)
    p_r, q_r = ref.palm_pose_own_pelvis(ref.T - 1)
    assert np.abs(p_a - p_r).max() < 1e-9 and np.abs(q_a - q_r).max() < 1e-9
    pad = ref.padded(100)
    assert pad.T == ref.T + 100 and pad.T_original == ref.T and pad.name == ref.name
    assert np.array_equal(pad.body_pos_w[-1], ref.body_pos_w[-1]) and np.abs(pad.joint_vel[-1]).max() == 0.0
    assert ref.padded(0) is ref


@needs_bench
def test_bench_goal_from_manifest_matches_clip_end():
    man = load_bench_manifest(BENCH_DIR / "BENCH_MANIFEST.json")
    for path in bench_clips()[:3]:
        ref = ClipReference(path)
        goal = BenchGoal.from_manifest_entry(manifest_entry(man, ref.name))
        p_end, q_end = ref.palm_pose_w(ref.T - 1)
        assert np.linalg.norm(p_end[goal.hand_index] - goal.pos_w) < 0.012
        q_goal_xyzw = np.roll(goal.quat_wxyz, -1)
        assert abs(float(np.dot(q_end[goal.hand_index], q_goal_xyzw))) > np.cos(np.radians(5.0) / 2)
        assert 0 < goal.reach_end_frame < ref.T and goal.height_label == ref.source_tag


@pytest.mark.mujoco
@pytest.mark.skipif(not (has_mujoco() and has_mink() and _default_mjcf_exists()), reason="needs mink + the bundled Dex3 MJCF scene")
def test_replanner_builds_its_ik_model_without_a_table():
    goal = BenchGoal.from_manifest_entry(_row())
    rp = BenchReplanner(goal, ReplanConfig())
    assert rp.bundle.table_geoms == [None] and rp.goal.table is None
    rp_tab = BenchReplanner(BenchGoal.from_manifest_entry(_row(table={"center": [0.6, 0.1, 0.28], "half_size": [0.3, 0.7, 0.02], "surface_z": 0.3})), ReplanConfig())
    assert len(rp_tab.bundle.table_geoms) == 1 and rp_tab.bundle.table_geoms[0] is not None


def _mid_reach_state(back: int = 0):
    """Local bench clip, robot placed ``back`` frames behind the live reference frame t (mid-reach)."""
    from sim2sim import plant_params as PP
    from sim2sim.plant import MujocoPlant
    from sim2sim.state import read_state

    man = load_bench_manifest(BENCH_DIR / "BENCH_MANIFEST.json")
    ref = ClipReference(bench_clips()[0])
    goal = BenchGoal.from_manifest_entry(manifest_entry(man, ref.name))
    plant = MujocoPlant(kp=PP.KP, kd=PP.KD, effort_limit=PP.ACTION_SCALE * PP.KP / 0.25, keep_visual=False)
    t = goal.reach_end_frame - 30
    plant.reset(ref.root_pos_w[t - back], ref.root_quat_w[t - back], ref.joint_pos[t - back])
    return ref, goal, plant, read_state(plant), t


@needs_mink
@needs_bench
@pytest.mark.parametrize("base_mode", ["clip", "current"])
def test_replanner_from_mid_reach_state(base_mode):
    ref, goal, plant, st, t = _mid_reach_state()
    rp = BenchReplanner(goal, ReplanConfig(base_mode=base_mode, hold_s=2.0))
    new, info = rp.plan_from_state(st, plant, goal.pos_w)
    assert np.linalg.norm(new.root_pos_w[0] - st.root_pos) < 0.005
    assert np.abs(new.joint_pos[0] - st.dof_pos).max() < 0.03
    p0, _ = new.palm_pose_w(0)
    assert np.linalg.norm(p0[goal.hand_index] - st.palm_pos_w[goal.hand_index]) < 0.002
    p_end, _ = new.palm_pose_w(new.T - 1)
    assert np.linalg.norm(p_end[goal.hand_index] - goal.pos_w) < 0.015, info
    ankles, _ = plant.body_pose_by_name(["left_ankle_roll_link", "right_ankle_roll_link"])
    slots = [new.body_slot("left_ankle_roll_link"), new.body_slot("right_ankle_roll_link")]
    assert np.linalg.norm(new.body_pos_w[-1, slots] - ankles, axis=1).max() < 0.005
    inactive = list(range(22, 29)) if goal.hand == "left" else list(range(15, 22))
    assert np.abs(new.joint_pos[-1, inactive] - st.dof_pos[inactive]).max() < 0.05
    assert new.T == info["n_reach"] + int(round(2.0 * 50)) and info["qp_failures"] == 0
    if base_mode == "current":
        assert abs(new.root_pos_w[-1, 2] - st.root_pos[2]) < 0.02
    else:
        assert abs(new.root_pos_w[-1, 2] - ref.root_pos_w[-1, 2]) < 0.02


@needs_mink
@needs_bench
def test_replanner_schedule_and_summary():
    ref, goal, plant, st, t = _mid_reach_state()
    rp = BenchReplanner(goal, ReplanConfig(period_s=1.0, first_event="reach_end", hold_s=2.0, goal_adjust=True))
    re = goal.reach_end_frame
    assert rp.step(re - 1, st, plant) is None
    assert rp.step(re, st, plant) is not None
    assert rp.step(re + 10, st, plant) is None
    assert rp.step(re + 50, st, plant) is not None
    s = rp.summary()
    assert s["n_replans"] == 2 and s["n_plans"] == 2 and s["ik_time_s"] > 0
    assert [e["action"] for e in s["events"]] == ["replan", "replan"]
    assert json.dumps(s)


@needs_mink
@needs_bench
def test_default_config_ignores_the_live_reference_and_handover_knobs_act():
    ref, goal, plant, st, t = _mid_reach_state(back=10)
    hi = goal.hand_index
    cfg = ReplanConfig(base_mode="current", hold_s=2.0)
    a, info_a = BenchReplanner(goal, cfg).plan_from_state(st, plant, goal.pos_w)
    b, info_b = BenchReplanner(goal, cfg, reference=ref).plan_from_state(st, plant, goal.pos_w, old_ref=ref, old_t=t)
    assert np.array_equal(a.joint_pos, b.joint_pos) and np.array_equal(a.body_pos_w, b.body_pos_w)
    assert "handover" not in info_a and "handover" not in info_b
    # start_pose="reference": frame 0 == the OLD reference's palm, not the robot's
    p_old, _ = ref.palm_pose_w(t)
    assert np.linalg.norm(p_old[hi] - st.palm_pos_w[hi]) > 0.04
    new, info = BenchReplanner(goal, ReplanConfig(base_mode="current", hold_s=2.0, start_pose="reference"), reference=ref).plan_from_state(st, plant, goal.pos_w, old_ref=ref, old_t=t)
    p0, _ = new.palm_pose_w(0)
    assert np.linalg.norm(p0[hi] - p_old[hi]) < 0.015 and info["start_pose"] == "reference" and info["start_pose_offset_cm"] > 3.0 and info["settle_palm_err_cm"] < 1.5
    # handover blend: frame 0 IS the old reference's frame t, the goal is still reached
    cfgb = ReplanConfig(base_mode="current", hold_s=2.0, handover_blend_s=0.3)
    blended, infob = BenchReplanner(goal, cfgb, reference=ref).plan_from_state(st, plant, goal.pos_w, old_ref=ref, old_t=t)
    assert infob["blend_frames"] == 15 and infob["blended_frames"] == 16 and blended.T == a.T
    assert np.array_equal(blended.joint_pos[0], ref.joint_pos[t]) and np.allclose(blended.root_pos_w[0], ref.root_pos_w[t])
    p_end, _ = blended.palm_pose_w(blended.T - 1)
    assert np.linalg.norm(p_end[hi] - goal.pos_w) < 0.02
