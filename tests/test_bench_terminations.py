"""Would-terminate causes of the benchmark rollouts (sim2sim.bench.terminations): the adaptive height gate and the fall guard, the hero_bench_v1
``fall_low`` margin rule, the ``--fail-causes`` parser and the runner's termination config."""
from __future__ import annotations

import numpy as np
import pytest

from sim2sim.bench.terminations import CAUSE_ORDER, FALL_LOW_DEFAULT_MARGIN_M, TerminationConfig, check_terminations, parse_fail_causes, reanchor_body_pos, termination_config_from_causes


def _flags(cfg: TerminationConfig, ref_z: float, robot_z: float) -> dict[str, bool]:
    identity = np.array([0, 0, 0, 1.0])
    same = {n: np.array([0.1 * i, 0.0, 0.05]) for i, n in enumerate(cfg.tracked_bodies)}
    return check_terminations(cfg, ref_root_pos=np.array([0, 0, ref_z]), ref_root_quat=identity, robot_root_pos=np.array([0, 0, robot_z]), robot_root_quat=identity,
                              rel_body_pos=same, robot_body_pos=same)


def test_adaptive_threshold_and_fall_guard():
    cfg = TerminationConfig()
    identity = np.array([0, 0, 0, 1.0])
    names = cfg.tracked_bodies
    assert names == ("left_ankle_roll_link", "right_ankle_roll_link", "left_wrist_yaw_link", "right_wrist_yaw_link")
    same = {n: np.array([0.1 * i, 0.0, 0.05]) for i, n in enumerate(names)}
    # reference root low (0.4 m) -> 0.75 m height gate: a 0.5 m anchor height error does NOT terminate
    f = check_terminations(cfg, ref_root_pos=np.array([0, 0, 0.4]), ref_root_quat=identity, robot_root_pos=np.array([0, 0, 0.9]), robot_root_quat=identity, rel_body_pos=same, robot_body_pos=same)
    assert not f["anchor_pos"] and not f["fall"]
    # reference standing -> strict 0.15 m gate fires; pelvis at 0.2 with ref 0.75 -> fall
    f = check_terminations(cfg, ref_root_pos=np.array([0, 0, 0.75]), ref_root_quat=identity, robot_root_pos=np.array([0, 0, 0.2]), robot_root_quat=identity, rel_body_pos=same, robot_body_pos=same)
    assert f["anchor_pos"] and f["fall"] and not f["anchor_xy"]
    # 0.6 m xy drift -> anchor_xy only
    f = check_terminations(cfg, ref_root_pos=np.array([0, 0, 0.75]), ref_root_quat=identity, robot_root_pos=np.array([0.6, 0, 0.75]), robot_root_quat=identity, rel_body_pos=same, robot_body_pos=same)
    assert f["anchor_xy"] and not f["anchor_pos"] and not f["fall"] and list(f) == list(CAUSE_ORDER)
    # a 0.25 m ankle error -> foot_pos_xyz (the height gate stays quiet: the error is along x)
    feet = {n: (v + np.array([0.25, 0.0, 0.0]) if n == "left_ankle_roll_link" else v) for n, v in same.items()}
    f = check_terminations(cfg, ref_root_pos=np.array([0, 0, 0.75]), ref_root_quat=identity, robot_root_pos=np.array([0, 0, 0.75]), robot_root_quat=identity, rel_body_pos=same, robot_body_pos=feet)
    assert f["foot_pos_xyz"] and not f["ee_body_pos"]
    # orientation: ang^2 = 0.3 > 0.2 rad^2 fires anchor_ori_full only
    ang = np.sqrt(0.3)
    q_yaw = np.array([0.0, 0.0, np.sin(ang / 2), np.cos(ang / 2)])
    f = check_terminations(cfg, ref_root_pos=np.array([0, 0, 0.75]), ref_root_quat=identity, robot_root_pos=np.array([0, 0, 0.75]), robot_root_quat=q_yaw, rel_body_pos=same, robot_body_pos=same)
    assert f["anchor_ori_full"] and not any(f[c] for c in CAUSE_ORDER if c != "anchor_ori_full")
    with pytest.raises(ValueError):
        TerminationConfig(fail_causes=("fall", "nope"))


def test_reanchor_keeps_height_difference_and_heading_only():
    """The reference bodies are moved to the robot anchor xy + heading (yaw only), keeping the height difference."""
    ref_anchor = np.array([1.0, 2.0, 0.75])
    robot_anchor = np.array([1.3, 1.9, 0.70])
    q_ref = np.array([0.0, 0.0, np.sin(0.2), np.cos(0.2)])          # yaw 0.4
    q_rob = np.array([0.0, 0.0, np.sin(0.45), np.cos(0.45)])        # yaw 0.9
    body = ref_anchor + np.array([[0.5, 0.0, 0.1], [0.0, -0.3, -0.2]])
    rel = reanchor_body_pos(ref_anchor, q_ref, robot_anchor, q_rob, body)
    d = body - ref_anchor
    c, s = np.cos(0.5), np.sin(0.5)
    rot = np.stack([c * d[:, 0] - s * d[:, 1], s * d[:, 0] + c * d[:, 1], d[:, 2]], axis=1)
    exp = robot_anchor + np.array([0.0, 0.0, ref_anchor[2] - robot_anchor[2]]) + rot
    assert np.allclose(rel, exp, atol=1e-12)


def test_fall_low_default_off_margin_rule_and_continuity_at_050():
    plain = TerminationConfig()
    assert plain.fall_low_ref_margin_m is None
    low = TerminationConfig(fall_low_ref_margin_m=0.20)
    # default: a low reference (0.45 m) suppresses the fall guard whatever the robot does (the v1 rule)
    assert not _flags(plain, 0.45, 0.20)["fall"] and not _flags(plain, 0.45, 0.10)["fall"]
    # fall_low: robot pelvis more than 0.20 m below the (low) reference pelvis -> fall; within the margin -> no fall
    assert _flags(low, 0.45, 0.24)["fall"] and not _flags(low, 0.45, 0.26)["fall"]
    # continuity at the 0.50 m switch: just below (margin rule) and just above (plain rule) both fire at a 0.30 m pelvis
    assert _flags(low, 0.50, 0.29)["fall"] and not _flags(low, 0.50, 0.31)["fall"]
    assert _flags(low, 0.5001, 0.29)["fall"] and not _flags(low, 0.5001, 0.31)["fall"]
    assert _flags(plain, 0.5001, 0.29)["fall"] and not _flags(plain, 0.50, 0.29)["fall"]   # the plain rule alone is suppressed AT 0.50
    # a standing reference: the margin never changes the plain rule
    for z in (0.20, 0.29, 0.31, 0.60):
        assert _flags(low, 0.75, z)["fall"] == _flags(plain, 0.75, z)["fall"]
    # every other cause is untouched by the margin
    for ref_z, rob_z in ((0.45, 0.24), (0.50, 0.29), (0.75, 0.2)):
        a, b = _flags(low, ref_z, rob_z), _flags(plain, ref_z, rob_z)
        assert all(a[c] == b[c] for c in CAUSE_ORDER if c != "fall") and list(a) == list(CAUSE_ORDER)
    with pytest.raises(ValueError):
        TerminationConfig(fall_low_ref_margin_m=0.0)


def test_fail_causes_alias_fall_low_runner_and_parser():
    assert parse_fail_causes("fall,anchor_xy") == (("fall", "anchor_xy"), None)
    assert parse_fail_causes("fall_low,anchor_xy") == (("fall", "anchor_xy"), FALL_LOW_DEFAULT_MARGIN_M)
    assert parse_fail_causes("fall,fall_low:0.25") == (("fall",), 0.25)
    assert parse_fail_causes(None) == (tuple(CAUSE_ORDER), None) and parse_fail_causes("") == (tuple(CAUSE_ORDER), None)
    cfg = termination_config_from_causes("fall_low,anchor_xy")
    assert cfg.fail_causes == ("fall", "anchor_xy") and cfg.fall_low_ref_margin_m == 0.20
    assert termination_config_from_causes("fall,anchor_xy") == TerminationConfig(fail_causes=("fall", "anchor_xy"))   # no alias -> the exact old config
    with pytest.raises(ValueError):
        termination_config_from_causes("fall_lo")
    from sim2sim.bench import run as R

    base = ["--motion-dir", "x", "--out", "y"]
    assert R._termination_config(R.build_parser().parse_args(base)) == TerminationConfig()                       # default: unchanged
    t = R._termination_config(R.build_parser().parse_args(base + ["--fail-causes", "fall_low,anchor_xy"]))
    assert t.fail_causes == ("fall", "anchor_xy") and t.fall_low_ref_margin_m == 0.20
    assert R._termination_config(R.build_parser().parse_args(base + ["--fail-causes", "fall,anchor_xy"])) == TerminationConfig(fail_causes=("fall", "anchor_xy"))
