"""Series file + the three scoring layers (sim2sim.bench.series / summary / closed_loop_summary / report): the npz round trip and the
fixed-horizon aggregation, the keypoint columns' success aggregation, the bounded hold window / groups / Wilson success columns of the open-loop
summary, the manifest-derived groups and C3 of the closed-loop summary, the side-by-side report's ``n/a (harness)`` cell and "no lone global
number" rule -- plus regressions against reference results under HERO_BENCH_RESULTS_DIR (skipped when absent)."""
from __future__ import annotations

import csv
import json
import math
import warnings
from pathlib import Path

import numpy as np
import pytest

from sim2sim.bench import closed_loop_summary as CL
from sim2sim.bench import report as RP
from sim2sim.bench import summary as BS
from sim2sim.bench.metrics import KEYPOINT_METRIC_KEYS, METRIC_KEYS
from sim2sim.bench.rollout import JERK_METRIC_KEYS
from sim2sim.bench.series import SCHEMA, alive_until_steps, build_series, horizon_average, read_series, sonic_success_block, sonic_success_per_clip, summarize, summary_markdown, write_series

from _bench_fixtures import BENCH_DIR, BENCH_RESULTS_DIR

V1_MANIFEST = BENCH_DIR / "BENCH_MANIFEST.json"
OL_SERIES, OL_SUMMARY = BENCH_RESULTS_DIR / "open_loop" / "series.npz", BENCH_RESULTS_DIR / "open_loop" / "summary" / "bench_summary.json"
CL_SERIES, CL_SUMMARY = BENCH_RESULTS_DIR / "closed_loop" / "series.npz", BENCH_RESULTS_DIR / "closed_loop" / "summary" / "closed_loop_summary.json"
#: synthetic ``meta["termination"]`` blocks (what sim2sim.bench.run records): the open-loop run with the low-posture margin on, the closed-loop run without
OL_TERM = {"fail_causes": ["fall", "anchor_xy"], "fall_low_ref_margin_m": 0.2, "fall_pelvis_z": 0.3, "anchor_xy_threshold": 0.5}
CL_TERM = {"fail_causes": ["fall", "anchor_xy"], "fall_low_ref_margin_m": None, "fall_pelvis_z": 0.3, "anchor_xy_threshold": 0.5}


# ================================================================================================ series
def _series(tmp_path):
    T = 6
    metrics = {}
    base = np.array([[1, 2, 3, 4, 5, 6], [10, 10, 10, np.nan, np.nan, np.nan], [2, 2, 2, 2, 2, 2]], dtype=np.float32)
    for k in METRIC_KEYS:
        metrics[k] = base.copy()
    metrics["sonic_fail"] = np.zeros_like(base)   # the 0 / 1 failure flag: never failed
    s = build_series(
        sim="mujoco", dt=0.02, horizon_steps=T, meta={"policy": {"tag": "model_x", "onnx": "m.onnx"}, "plant": {"physics_dt": 0.001, "substeps": 20}},
        clip_name=["a__c1.npz", "b__c2.npz", "a__c3.npz"], source_tag=["a", "b", "a"], clip_len_steps=[100, 4, 50],
        valid_until=[6, 3, 6], first_fail_step=[4, -1, -1], first_fail_cause=["anchor_xy", "", ""], metrics=metrics,
    )
    return read_series(write_series(tmp_path / "series.npz", s))


def test_roundtrip_and_schema(tmp_path):
    r = _series(tmp_path)
    assert r["schema"] == SCHEMA and r["sim"] == "mujoco" and float(r["dt"]) == 0.02 and int(r["horizon_steps"]) == 6
    assert list(r["clip_name"]) == ["a__c1.npz", "b__c2.npz", "a__c3.npz"] and list(r["source_tag"]) == ["a", "b", "a"]
    assert r["first_fail_step"].tolist() == [4, -1, -1] and list(r["first_fail_cause"]) == ["anchor_xy", "", ""]
    assert set(r["metrics"]) == set(METRIC_KEYS) and r["metrics"]["ee_local_cm"].shape == (3, 6) and r["metrics"]["ee_local_cm"].dtype == np.float32
    assert json.loads(r["meta_json"])["policy"]["tag"] == "model_x" and r["meta"]["policy"]["tag"] == "model_x"
    with pytest.raises(ValueError, match="shape"):
        build_series(sim="mujoco", dt=0.02, horizon_steps=2, meta={}, clip_name=["a__x.npz"], source_tag=["a"], clip_len_steps=[3], valid_until=[2], first_fail_step=[-1], first_fail_cause=[""],
                     metrics={k: np.zeros((1, 3), np.float32) for k in METRIC_KEYS})


def test_horizon_average_alive_and_summary(tmp_path):
    r = _series(tmp_path)
    v = r["metrics"]["ee_local_cm"]
    assert np.allclose(horizon_average(v, r["valid_until"]), [3.5, 10.0, 2.0])
    au = alive_until_steps(r["valid_until"], r["first_fail_step"])
    assert au.tolist() == [3, 3, 6] and np.allclose(horizon_average(v, au), [2.0, 10.0, 2.0])
    s = summarize(r, times_s=(0.02, 0.06, 0.1))
    ov = s["overall"]
    assert ov["n_clips"] == 3 and ov["fail_free_fraction"] == 2 / 3
    assert ov["metrics"]["ee_local_cm"]["mean"] == np.mean([3.5, 10.0, 2.0]) and ov["metrics_alive"]["ee_local_cm"]["mean"] == np.mean([2.0, 10.0, 2.0])
    assert ov["first_fail_cause_counts"] == {"anchor_xy": 1, "none": 2} and ov["mean_valid_steps"] == 5.0 and ov["mean_alive_steps"] == 4.0
    assert s["by_source"]["a"]["n_clips"] == 2 and s["by_source"]["b"]["fail_free_fraction"] == 1.0 and s["by_source"]["a"]["metrics"]["ee_local_cm"]["mean"] == 2.75
    at = ov["at_time"]["0.06"]
    assert at["n_valid"] == 3 and at["n_alive"] == 3 and at["metrics"]["ee_local_cm"]["mean"] == 5.0
    at = ov["at_time"]["0.1"]
    assert at["n_valid"] == 2 and at["n_alive"] == 1 and at["metrics_alive"]["ee_local_cm"]["mean"] == 2.0
    assert s["clips"][1]["horizon_avg"]["ee_local_cm"] == 10.0
    # keypoint columns present in METRIC_KEYS -> the SONIC-protocol success aggregation + provenance block (all 1.0 here: never failed)
    assert s["keypoints"]["sonic_success_frac"] == 1.0 and s["keypoints"]["n"] == 3 and ov["sonic_success_fraction"] == 1.0 and s["clips"][0]["sonic_success"] is True
    md = summary_markdown(s)
    assert "fail-free fraction 0.667" in md and "| a | 2 |" in md and "a__c1.npz" in md and "## Full-body keypoints" in md


def test_sonic_success_per_clip_and_block():
    sf = np.array([[0, 0, 1, 0], [0, 0, 0, 0], [np.nan, np.nan, np.nan, np.nan]], dtype=np.float32)
    s, first = sonic_success_per_clip(sf, np.array([4, 4, 4]))
    assert s.tolist()[:2] == [0.0, 1.0] and np.isnan(s[2]) and first.tolist() == [3, -1, -1]
    s2, _ = sonic_success_per_clip(sf, np.array([2, 4, 4]))   # the failing step is outside the valid window
    assert s2[0] == 1.0
    b = sonic_success_block(s, ["x", "x", "y"])
    assert b["n"] == 2 and b["sonic_success"] == 1 and b["sonic_success_frac"] == 0.5 and b["by_source"] == {"x": {"n": 2, "sonic_success": 1, "sonic_success_frac": 0.5}}


def test_series_without_keypoint_columns_summarises_with_the_previous_keys_only(tmp_path):
    legacy = [k for k in METRIC_KEYS if k not in KEYPOINT_METRIC_KEYS]
    metrics = {k: np.ones((2, 5), dtype=np.float32) for k in legacy}
    s = build_series(sim="mujoco", dt=0.02, horizon_steps=5, meta={}, clip_name=["a__1.npz", "a__2.npz"], source_tag=["a", "a"], clip_len_steps=[9, 9], valid_until=[5, 5],
                     first_fail_step=[-1, -1], first_fail_cause=["", ""], metrics=metrics)
    r = read_series(write_series(tmp_path / "legacy.npz", s))
    assert set(r["metrics"]) == set(legacy)
    summ = summarize(r)
    assert "keypoints" not in summ and "sonic_success_fraction" not in summ["overall"] and "sonic_success" not in summ["clips"][0]
    assert "Full-body keypoints" not in summary_markdown(summ)


# ================================================================================================ open-loop summary: bounded hold window / groups / success
T_OL, DT_OL = 20, 0.5   # final window (1 s) = 2 columns


def _ol_series(names, metrics_fn, clen=None, vu=None, ff=None, sim="mujoco", meta=None):
    C = len(names)
    metrics = {k: np.full((C, T_OL), 1.0, dtype=np.float32) for k in METRIC_KEYS}
    metrics_fn(metrics)
    return build_series(sim=sim, dt=DT_OL, horizon_steps=T_OL, meta=(meta if meta is not None else {"tag": "t"}), clip_name=names, source_tag=[n.split("__")[0] for n in names],
                        clip_len_steps=clen or [20] * C, valid_until=vu or [19] * C, first_fail_step=ff or [-1] * C, first_fail_cause=["" if (ff or [-1] * C)[i] < 0 else "fall" for i in range(C)],
                        metrics=metrics)


def _read_ol(series_dict, tmp_path: Path, name: str = "s.npz"):
    return read_series(write_series(tmp_path / name, series_dict))


def test_hold_window_ends_at_hold_end_frame_and_rest_window(tmp_path):
    names = ["retract__a.npz", "h050__b.npz"]

    def fill(m):
        for k in ("ee_global_left_cm", "ee_global_right_cm", "ee_global_cm"):
            m[k][0, :10] = 1.0
            m[k][0, 10:] = 50.0      # the clip's own retract frames: a huge error vs the hold target
            m[k][1, :] = 2.0

    s = _read_ol(_ol_series(names, fill), tmp_path)
    strat = {"file": names[0], "hand": "left", "height_label": "h074", "stratum": "retract", "tier": "extended", "reach_end_frame": 4, "hold_frames": 6, "hold_end_frame": 10,
          "retract_start_frame": 10, "n_frames": 20, "fps": 2, "terminal_ee_pos_err_m": 0.004}
    legacy = {"file": names[1], "hand": "right", "height_label": "h050", "reach_end_frame": 4, "hold_frames": 16, "n_frames": 20, "fps": 2}
    summ = BS.summarize([s], ["mujoco"], BS.manifest_entries([strat, legacy]), final_s=1.0)
    rec = summ["sims"]["mujoco"]["clips"]["retract__a"]
    assert rec["hold_cols"] == [4, 10] and rec["hold_end_frame"] == 10 and rec["hold_mean"]["ee_global_active_cm"] == pytest.approx(1.0)
    assert rec["rest_cols"] == [17, 19] and rec["rest_mean"]["ee_global_active_cm"] == pytest.approx(50.0) and rec["ik_residual_cm"] == pytest.approx(0.4)
    assert rec["stratum"] == "retract" and rec["tier"] == "extended"
    rb = summ["sims"]["mujoco"]["clips"]["h050__b"]
    assert rb["hold_cols"] == [4, 19] and rb["stratum"] == "h050" and rb["tier"] == "core" and "rest_mean" not in rb
    old = BS.summarize([s], ["mujoco"], BS.manifest_entries([{"file": names[0], "hand": "left", "height_label": "h074", "reach_end_frame": 4, "fps": 2}, legacy]), final_s=1.0)
    ro = old["sims"]["mujoco"]["clips"]["retract__a"]
    assert ro["hold_cols"] == [4, 19] and ro["hold_mean"]["ee_global_active_cm"] == pytest.approx((6 * 1.0 + 9 * 50.0) / 15)
    assert BS.hold_columns(4, 20, 19, 20) == (4, 19) and BS.hold_columns(4, 20, 19, 20, 10) == (4, 10) and BS.hold_columns(12, 20, 19, 20, 10) == (12, 12)
    g = summ["sims"]["mujoco"]["bench_groups"]
    assert set(g) >= {"stratum:retract", "stratum:h050", "tier:extended", "tier:core", "cell:retract/h074/left", "cell:retract/h074/all", "cell:h050/h050/right"}
    assert g["stratum:retract"]["rest"]["metrics"]["ee_global_active_cm"]["mean"] == pytest.approx(50.0) and "rest" not in g["stratum:h050"]
    assert g["stratum:retract"]["success"]["ik_residual_cm"] == {"mean": pytest.approx(0.4), "n": 1, "p90": pytest.approx(0.4)}
    md = BS.render_markdown(summ)
    assert "| stratum:retract | mujoco | 1 |" in md and "IK residual (cm)" in md


def test_group_by_stratum_and_tier_with_legacy_fallbacks(tmp_path, capsys):
    names = ["h050__a.npz", "h050__b.npz", "far060__c.npz", "far060__d.npz"]

    def fill(m):
        for i, v in enumerate((2.0, 4.0, 6.0, 8.0)):
            for k in ("ee_global_left_cm", "ee_global_right_cm", "ee_global_cm"):
                m[k][i] = v

    s = _read_ol(_ol_series(names, fill), tmp_path)
    rows = [{"file": names[0], "hand": "left", "height_label": "h050", "reach_end_frame": 4, "hold_frames": 15, "n_frames": 19, "fps": 2},
            {"file": names[1], "hand": "right", "height_label": "h050", "reach_end_frame": 4, "hold_frames": 15, "n_frames": 19, "fps": 2, "tier": "core"},
            {"file": names[2], "hand": "left", "height_label": "h074", "stratum": "far060", "tier": "core", "reach_end_frame": 4, "hold_frames": 15, "n_frames": 19, "fps": 2},
            {"file": names[3], "hand": "right", "height_label": "h088", "stratum": "far060", "tier": "extended", "reach_end_frame": 4, "hold_frames": 15, "n_frames": 19, "fps": 2}]
    man = BS.manifest_entries(rows)
    assert BS.bench_group_value(rows[0], "stratum") == "h050" and BS.bench_group_value(rows[0], "tier") == "core" and BS.bench_group_value(rows[0], "orient_family") is None
    st = BS.summarize([s], ["mujoco"], man, final_s=1.0, group_by="stratum")
    assert st["group_by"] == "stratum" and st["group_values"] == ["far060", "h050"] and st["sims"]["mujoco"]["n_ungrouped"] == 0
    fg = st["sims"]["mujoco"]["family_groups"]
    assert fg["h050"]["all"]["n_clips"] == 2 and fg["h050"]["all"]["metrics"]["ee_global_active_cm"]["mean"] == pytest.approx(3.0)
    assert fg["far060"]["all"]["metrics"]["ee_global_active_cm"]["mean"] == pytest.approx(7.0) and fg["far060"]["left"]["n_clips"] == 1
    assert st["sims"]["mujoco"]["family_height_groups"]["far060"]["h088"]["right"]["metrics"]["ee_global_active_cm"]["mean"] == pytest.approx(8.0)
    tr = BS.summarize([s], ["mujoco"], man, final_s=1.0, group_by="tier")
    assert tr["group_values"] == ["core", "extended"] and tr["sims"]["mujoco"]["family_groups"]["core"]["all"]["n_clips"] == 3
    assert "### Per `tier`" in BS.render_markdown(tr) and any(r["orient_family"] == "extended" for r in BS.csv_rows(tr))
    plain = BS.summarize([s], ["mujoco"], man, final_s=1.0)
    g = plain["sims"]["mujoco"]["bench_groups"]
    assert g["stratum:far060"]["n_clips"] == 2 and g["tier:extended"]["n_clips"] == 1 and g["cell:h050/h050/left"]["n_clips"] == 1 and g["cell:far060/h074/all"]["n_clips"] == 1
    assert plain["strata"] == ["far060", "h050"] and plain["tiers"] == ["core", "extended"] and "group_by" not in plain
    sp = write_series(tmp_path / "cli.npz", _ol_series(names, fill))
    mp1 = tmp_path / "m1.json"
    mp1.write_text(json.dumps({"clips": [{k: v for k, v in r.items() if k not in ("stratum", "tier")} for r in rows]}))
    assert BS.main(["--series", str(sp), "--bench-manifest", str(mp1), "--out", str(tmp_path / "o1"), "--group-by", "stratum"]) == 0
    out = capsys.readouterr().out
    assert "legacy fallback in force" in out and "grouping skipped" not in out
    d = json.loads((tmp_path / "o1" / "bench_summary.json").read_text())
    assert d["group_by"] == "stratum" and sorted(d["group_values"]) == ["h050", "h074", "h088"] and d["success_protocol"]["source"] == "default"
    assert BS.main(["--series", str(sp), "--bench-manifest", str(mp1), "--out", str(tmp_path / "o2"), "--group-by", "orient_family"]) == 0
    assert "grouping skipped" in capsys.readouterr().out
    assert set(BS.GROUP_BY_CHOICES) == {"none", "orient_family", "stratum", "tier"}
    assert BS.default_labels([s, s]) == ["mujoco-t", "mujoco-t_2"] and BS.default_labels([s]) == ["mujoco"]


def test_wilson_ci_and_success_columns_known_values(tmp_path):
    lo, hi = BS.wilson_ci(0, 10)
    assert lo == 0.0 and hi == pytest.approx(0.27753, abs=1e-4)
    lo, hi = BS.wilson_ci(10, 10)
    assert lo == pytest.approx(0.72247, abs=1e-4) and hi == 1.0
    lo, hi = BS.wilson_ci(5, 10)
    assert lo == pytest.approx(0.23659, abs=1e-4) and hi == pytest.approx(0.76341, abs=1e-4)
    lo, hi = BS.wilson_ci(30, 60)
    assert lo == pytest.approx(0.3774, abs=1e-3) and hi == pytest.approx(0.6226, abs=1e-3)
    assert all(math.isnan(v) for v in BS.wilson_ci(0, 0))
    names = ["h074__a.npz", "h074__b.npz", "h074__c.npz", "h074__d.npz"]
    spec = [(2.0, 5.0, True), (6.0, 12.0, True), (4.0, 20.0, True), (1.0, 1.0, False)]

    def fill(m):
        for i, (p, r, _) in enumerate(spec):
            m["ee_global_right_cm"][i] = p
            m["ee_rot_global_right_deg"][i] = r
            m["ee_global_cm"][i] = p + 1.0

    s = _read_ol(_ol_series(names, fill, ff=[-1, -1, -1, 8]), tmp_path)
    rows = [{"file": n, "hand": "right", "height_label": "h074", "reach_end_frame": 4, "hold_frames": 15, "n_frames": 19, "fps": 2, "terminal_ee_pos_err_m": 0.01 * (i + 1)} for i, n in enumerate(names)]
    summ = BS.summarize([s], ["mujoco"], BS.manifest_entries(rows), final_s=1.0)
    g = summ["sims"]["mujoco"]["groups"]["all"]["all"]
    sc = g["success"]
    assert [summ["sims"]["mujoco"]["clips"][BS._stem(n)]["success"] for n in names] == [{"S7.5": True, "S5": True}, {"S7.5": True, "S5": False}, {"S7.5": False, "S5": False}, {"S7.5": False, "S5": False}]
    assert sc["levels"]["S7.5"]["k"] == 2 and sc["levels"]["S7.5"]["frac"] == 0.5 and sc["levels"]["S7.5"]["ci95"] == pytest.approx(list(BS.wilson_ci(2, 4)))
    assert sc["levels"]["S5"]["k"] == 1 and sc["levels"]["S5"]["pos_cm"] == 5.0 and sc["levels"]["S5"]["rot_deg"] == 10.0 and sc["levels"]["S7.5"]["rot_deg"] == 15.0
    assert sc["cdf_cm"] == {"2.5": pytest.approx(0.5), "5": pytest.approx(0.75), "10": pytest.approx(1.0)}
    assert sc["ik_residual_cm"]["mean"] == pytest.approx(2.5) and sc["ik_residual_cm"]["n"] == 4
    assert g["metrics"]["ee_rot_global_active_deg"]["mean"] == pytest.approx(9.5) and g["fail_free_frac"] == 0.75
    md = BS.render_markdown(summ)
    assert "| h074 | mujoco | 4 | 3.2 ± 1.9 | 9.5 ± 7.2 | 4.2 ± 1.9 | 1.0 ± 0.0 | 75.0% | 50.0% [15, 85] | 25.0% [5, 70] | 50.0% / 75.0% / 100.0% | 2.50 (n=4) |" in md
    pl = BS.flat_scalars(summ)
    assert pl["bench/mujoco/all/success_S7_5"] == 0.5 and pl["bench/mujoco/all/success_S5"] == 0.25 and pl["bench/mujoco/all/cdf_le_5cm"] == 0.75
    proto = {"success": {"open_loop": {"S7.5": [7.5, 15.0], "S5": {"max_pos_cm": 4.0, "rot_deg": 10.0}}, "closed_loop": {"C3": {"pos_m": 0.02, "rot_deg": 12.0}}, "cdf_points_cm": [2, 4]}}
    sp = BS.success_protocol(proto)
    assert sp["source"] == "manifest" and sp["open_loop"]["S5"] == {"pos_cm": 4.0, "rot_deg": 10.0} and sp["open_loop"]["S7.5"] == {"pos_cm": 7.5, "rot_deg": 15.0}
    assert sp["closed_loop"]["C3"]["pos_cm"] == pytest.approx(2.0) and sp["closed_loop"]["C3"]["rot_deg"] == 12.0 and sp["cdf_points_cm"] == [2.0, 4.0] and sp["stayed_threshold_cm"] == 1.75
    s2 = BS.summarize([s], ["mujoco"], BS.manifest_entries(rows), final_s=1.0, protocol=proto)
    sc2 = s2["sims"]["mujoco"]["groups"]["all"]["all"]["success"]
    assert sc2["levels"]["S5"]["k"] == 1 and sc2["levels"]["S5"]["pos_cm"] == 4.0 and list(sc2["cdf_cm"]) == ["2", "4"] and s2["success_protocol"]["source"] == "manifest"
    assert BS.success_protocol(None)["source"] == "default" and BS.success_protocol({"paper": "x"})["source"] == "default"
    assert BS.DEFAULT_SUCCESS["open_loop"]["S7.5"] == {"pos_cm": 7.5, "rot_deg": 15.0} and BS.DEFAULT_SUCCESS["closed_loop"]["C3"]["pos_cm"] == 3.0


# ================================================================================================ closed-loop summary: groups from the manifest
def _cl_series(tmp_path: Path, names, hands, final_err, hold3_err, *, T: int = 400, orig: int = 262, fail=None, name="cl") -> Path:
    C = len(names)
    metrics = {k: np.zeros((C, T), dtype=np.float32) for k in METRIC_KEYS}
    for i, h in enumerate(hands):
        metrics[f"ee_global_{h}_cm"][i, :] = hold3_err[i]
        metrics[f"ee_global_{h}_cm"][i, orig:] = final_err[i]
        metrics[f"ee_rot_global_{h}_deg"][i, :] = 5.0
    meta = {"clip_len_steps_original": {n: orig for n in names}, "termination": CL_TERM,
            "replan_log": {n: {"n_replans": 2, "replan_steps": [112, 262], "events": [], "ik_time_s": 0.1, "goal_offset_cm": 1.0, "stayed_at_step": (300 if i % 2 == 0 else None)} for i, n in enumerate(names)}}
    ff = fail or [-1] * C
    series = build_series(sim="mujoco", dt=0.02, horizon_steps=T, meta=meta, clip_name=names, source_tag=[n.split("__")[0] for n in names], clip_len_steps=[T] * C,
                          valid_until=[T] * C, first_fail_step=ff, first_fail_cause=["fall" if f > 0 else "" for f in ff], metrics=metrics)
    return write_series(tmp_path / name / "series.npz", series)


def test_closed_loop_groups_from_manifest_with_mixed_strata(tmp_path):
    names = ["h050__a.npz", "h050__b.npz", "floor_pick__c.npz", "floor_pick__d.npz"]
    hands = ["left", "right", "left", "right"]
    rows = [{"file": names[0], "hand": "left", "height_label": "h050", "reach_end_frame": 112, "hold_frames": 150, "n_frames": 262},
            {"file": names[1], "hand": "right", "height_label": "h050", "reach_end_frame": 112, "hold_frames": 150, "n_frames": 262, "tier": "core"},
            {"file": names[2], "hand": "left", "height_label": "floor", "stratum": "floor_pick", "tier": "extended", "table": None, "reach_end_frame": 100, "hold_end_frame": 150, "hold_frames": 50, "n_frames": 150},
            {"file": names[3], "hand": "right", "height_label": "floor", "stratum": "floor_pick", "tier": "extended", "table": None, "reach_end_frame": 100, "hold_frames": 50, "n_frames": 150}]
    mp = tmp_path / "BENCH_MANIFEST.json"
    mp.write_text(json.dumps({"clips": rows, "protocol": {"success": {"closed_loop": {"C3": {"pos_cm": 3.0, "rot_deg": 15.0}}}}}))
    p = _cl_series(tmp_path, names, hands, final_err=[2.0, 4.0, 1.0, 2.5], hold3_err=[6.0, 8.0, 10.0, 12.0], fail=[-1, -1, -1, 50])
    summ = CL.summarize([("replan", str(p))], str(mp))
    res = summ["results"]["replan"]
    g = res["groups"]
    assert res["heights"] == ["floor", "h050"] and summ["success_protocol"]["source"] == "manifest"
    assert set(g) >= {"all", "h050", "floor", "stratum:h050", "stratum:floor_pick", "tier:core", "tier:extended", "cell:floor_pick/floor/left", "cell:floor_pick/floor/all", "cell:h050/h050/right"}
    assert "h074" not in g and "h088" not in g
    per = {r["clip"]: r for r in summ["per_clip"]["replan"]}
    rows_full = CL.per_clip_rows(CL.read_series(str(p)), CL.load_manifest(str(mp)), hold_s=3.0, final_s=1.0)
    w = {r["clip"]: r["windows"] for r in rows_full}
    assert w[names[0]]["hold3"]["cols"] == [112, 262] and w[names[2]]["hold3"]["cols"] == [100, 150] and w[names[3]]["hold3"]["cols"] == [100, 150]
    assert per[names[2]]["hold_frames"] == 50 and per[names[0]]["hold_frames"] == 150 and per[names[2]]["stratum"] == "floor_pick" and per[names[0]]["stratum"] == "h050" and per[names[0]]["tier"] == "core"
    assert g["stratum:floor_pick"]["windows"]["hold3"]["ee_global_active_cm"]["mean"] == pytest.approx(11.0)
    sc = g["all"]["success"]
    assert sc["levels"]["C3"]["k"] == 2 and sc["levels"]["C3"]["n"] == 4 and sc["levels"]["C3"]["ci95"] == pytest.approx(list(BS.wilson_ci(2, 4))) and sc["levels"]["C3"]["window"] == "final"
    assert sc["stayed_frac"] == pytest.approx(0.5) and sc["cdf_cm"]["2.5"] == pytest.approx(0.75) and sc["cdf_cm"]["5"] == pytest.approx(1.0)
    assert g["stratum:floor_pick"]["success"]["levels"]["C3"]["k"] == 1 and g["tier:core"]["success"]["levels"]["C3"]["k"] == 1
    assert per[names[1]]["success"] == {"C3": False} and per[names[0]]["success"] == {"C3": True}
    md = CL.markdown(summ)
    assert "| run | floor | h050 | all | fail-free |" in md and "## closed-loop success" in md and "| replan | stratum:floor_pick | 2 | 0.500 | 50.0% [9, 91] | 0.50 |" in md
    row = CL.flat_rows(summ)[0]
    assert row["final_ee_global_active_cm_floor"] == pytest.approx(1.75) and row["final_ee_global_active_cm_h050"] == pytest.approx(3.0) and row["success_C3"] == 0.5 and "final_ee_global_active_cm_h074" not in row
    assert CL.main(["--series", f"replan|{p}", "--manifest", str(mp), "--out", str(tmp_path / "cli"), "--csv", str(tmp_path / "cli" / "t.csv")]) == 0
    got = list(csv.DictReader(open(tmp_path / "cli" / "t.csv")))
    assert got[0]["success_C3"] == "0.5" and float(got[0]["final_ee_global_active_cm_floor"]) == 1.75
    mp_old = tmp_path / "old.json"
    mp_old.write_text(json.dumps({"clips": [{"file": n, "hand": h, "reach_end_frame": 112, "n_frames": 262} for n, h in zip(names, hands)]}))
    w_old = {r["clip"]: r["windows"] for r in CL.per_clip_rows(CL.read_series(str(p)), CL.load_manifest(str(mp_old)), hold_s=2.0, final_s=1.0)}
    assert w_old[names[2]]["hold3"]["cols"] == [112, 212]


def test_entry_for_exact_stem_on_stratified_manifests_and_legacy_fallback_warns_once(tmp_path):
    # hero_bench_v1: the re-timed twins share their paper-protocol clip_id -- a clip missing from the manifest must never score against its twin's row
    strat = {"clips": [{"file": "h050__hero_bench_v1_000001.npz", "clip_id": "hero_bench_v1_000001", "stratum": "h050", "hand": "right", "reach_end_frame": 112},
                    {"file": "slow_x2__hero_bench_v1_000001.npz", "clip_id": "hero_bench_v1_000001", "stratum": "slow_x2", "hand": "right", "reach_end_frame": 224, "pair_of": "hero_bench_v1_000001"}]}
    mp = tmp_path / "stratified.json"
    mp.write_text(json.dumps(strat))
    man = CL.load_manifest(str(mp))
    assert isinstance(man, CL.ManifestRows) and man.path == str(mp) and not CL.manifest_is_legacy(man)
    assert CL.entry_for(man, "slow_x2__hero_bench_v1_000001.npz")["reach_end_frame"] == 224 and CL.entry_for(man, "/some/dir/h050__hero_bench_v1_000001.npz")["reach_end_frame"] == 112
    with pytest.raises(KeyError, match="hold6__hero_bench_v1_000001") as ei:
        CL.entry_for(man, "hold6__hero_bench_v1_000001.npz")
    assert str(mp) in str(ei.value) and "exact file name" in str(ei.value)
    # legacy manifest (no stratum anywhere): the old clip_id / name-tail fallback still works, with ONE warning per manifest
    legacy = {"clips": [{"file": "h050__hero_bench_v1_000001.npz", "clip_id": "hero_bench_v1_000001", "hand": "right", "reach_end_frame": 112}]}
    mpl = tmp_path / "legacy.json"
    mpl.write_text(json.dumps(legacy))
    manl = CL.load_manifest(str(mpl))
    assert CL.manifest_is_legacy(manl)
    with warnings.catch_warnings(record=True) as w:
        warnings.simplefilter("always")
        assert CL.entry_for(manl, "renamed__hero_bench_v1_000001.npz")["reach_end_frame"] == 112
        assert CL.entry_for(manl, "other__hero_bench_v1_000001.npz")["reach_end_frame"] == 112
        assert CL.entry_for(manl, "h050__hero_bench_v1_000001.npz")["reach_end_frame"] == 112   # exact: no fallback needed
    hits = [x for x in w if issubclass(x.category, RuntimeWarning) and "legacy bench manifest" in str(x.message)]
    assert len(hits) == 1 and str(mpl) in str(hits[0].message)
    with pytest.raises(KeyError, match="not in the bench manifest"):
        CL.entry_for(manl, "h050__nope.npz")


def _retract_series(tmp_path: Path, *, with_stop: bool, name: str) -> Path:
    """Two 50 Hz clips padded to T = 400: ``retract__r1`` (reach_end 100, hold [100, 150), its OWN retract [150, 250) at 30 cm from the goal, the
    rest pose held in the padding at 40 cm / 9 deg) and the plain ``h050__a`` (reach_end 100, hold to the clip end 250, 3 cm throughout)."""
    T = 400
    names = ["retract__r1.npz", "h050__a.npz"]
    metrics = {k: np.zeros((2, T), dtype=np.float32) for k in METRIC_KEYS}
    g = metrics["ee_global_left_cm"][0]
    g[:100], g[100:150], g[150:250], g[250:] = 20.0, 2.0, 30.0, 40.0
    r = metrics["ee_rot_global_left_deg"][0]
    r[:], r[250:] = 5.0, 9.0
    metrics["ee_global_right_cm"][1, :] = 3.0
    metrics["ee_rot_global_right_deg"][1, :] = 4.0
    rl = {"n_replans": 1, "replan_steps": [100], "events": [], "ik_time_s": 0.1, "goal_offset_cm": 0.5, "stayed_at_step": None}
    log = {names[0]: dict(rl, **({"stop": {"stop_frame": 150, "stopped_at_step": 150, "handback": {"handed_back": True}}} if with_stop else {})),
           names[1]: dict(rl, n_replans=3, replan_steps=[100, 250, 400], stayed_at_step=380)}
    meta = {"clip_len_steps_original": {n: 250 for n in names}, "replan_log": log, "termination": CL_TERM}
    series = build_series(sim="mujoco", dt=0.02, horizon_steps=T, meta=meta, clip_name=names, source_tag=["retract", "h050"], clip_len_steps=[T, T], valid_until=[T, T],
                          first_fail_step=[-1, -1], first_fail_cause=["", ""], metrics=metrics)
    return write_series(tmp_path / name / "series.npz", series)


def test_closed_loop_retract_rows_final_and_tail_stop_at_the_hand_back_and_rest_is_apart(tmp_path):
    rows = [{"file": "retract__r1.npz", "clip_id": "r1", "hand": "left", "height_label": "h074", "stratum": "retract", "tier": "extended", "reach_end_frame": 100, "hold_frames": 50,
             "hold_end_frame": 150, "retract_start_frame": 150, "n_frames": 250, "fps": 50},
            {"file": "h050__a.npz", "clip_id": "a", "hand": "right", "height_label": "h050", "stratum": "h050", "tier": "core", "reach_end_frame": 100, "hold_frames": 150, "hold_end_frame": 250,
             "n_frames": 250, "fps": 50}]
    mp = tmp_path / "BENCH_MANIFEST.json"
    mp.write_text(json.dumps({"clips": rows, "protocol": {"success": {"closed_loop": {"C3": {"pos_cm": 3.0, "rot_deg": 15.0}}}}}))
    p = _retract_series(tmp_path, with_stop=False, name="cl")
    summ = CL.summarize([("replan", str(p))], str(mp))
    per = {r["clip"]: r for r in CL.per_clip_rows(CL.read_series(str(p)), CL.load_manifest(str(mp)), hold_s=3.0, final_s=1.0)}
    rt, pl = per["retract__r1.npz"], per["h050__a.npz"]
    # retract row: hold3 unchanged, final = [150 - 50, 150) and tail = [100, 150) -- both stop at retract_start_frame; the rest pose is scored apart
    assert rt["windows"]["hold3"]["cols"] == [100, 150] and rt["windows"]["final"]["cols"] == [100, 150] and rt["windows"]["tail"]["cols"] == [100, 150]
    assert rt["windows"]["final"]["ee_global_active_cm"] == pytest.approx(2.0) and rt["windows"]["tail"]["ee_global_active_cm"] == pytest.approx(2.0) and rt["windows"]["final"]["ee_rot_global_active_deg"] == pytest.approx(5.0)
    assert rt["retract_start_frame"] == 150 and rt["rest"]["cols"] == [350, 400] and rt["rest"]["rest_pos_err_cm"] == pytest.approx(40.0) and rt["rest"]["rest_rot_err_deg"] == pytest.approx(9.0)
    assert rt["success"] == {"C3": True}   # C3 judges the reach (2 cm at the hand-back), not the rest pose
    # the plain row keeps today's windows exactly and has no rest block
    assert pl["windows"]["hold3"]["cols"] == [100, 250] and pl["windows"]["final"]["cols"] == [350, 400] and pl["windows"]["tail"]["cols"] == [100, 400] and "rest" not in pl and "retract_start_frame" not in pl
    g = summ["results"]["replan"]["groups"]
    assert g["stratum:retract"]["rest"]["n_clips"] == 1 and g["stratum:retract"]["rest"]["rest_pos_err_cm"]["mean"] == pytest.approx(40.0) and g["stratum:retract"]["rest"]["rest_rot_err_deg"]["mean"] == pytest.approx(9.0)
    assert g["stratum:retract"]["notes"] == [CL.RETRACT_NOTE] and g["all"]["notes"] == [CL.RETRACT_NOTE] and g["all"]["rest"]["n_clips"] == 1
    assert "rest" not in g["stratum:h050"] and "notes" not in g["stratum:h050"] and "rest" not in g["tier:core"]
    assert g["all"]["windows"]["final"]["ee_global_active_cm"]["mean"] == pytest.approx(2.5) and g["stratum:retract"]["success"]["levels"]["C3"]["k"] == 1   # (2 + 3) / 2: the rest pose never enters final
    pc = {r["clip"]: r for r in summ["per_clip"]["replan"]}
    assert pc["retract__r1.npz"]["rest"]["rest_pos_err_cm"] == pytest.approx(40.0) and "rest" not in pc["h050__a.npz"]
    md = CL.markdown(summ)
    assert "## `rest` window" in md and "| replan | stratum:retract | 1 | 40.0 ± 0.0 | 9.0 ± 0.0 |" in md and f"Note: {CL.RETRACT_NOTE}." in md
    row = CL.flat_rows(summ)[0]
    assert row["rest_n_clips"] == 1 and row["rest_pos_err_cm_mean"] == pytest.approx(40.0) and row["rest_rot_err_deg_mean"] == pytest.approx(9.0) and row["final_ee_global_active_cm_mean"] == pytest.approx(2.5)
    # --final-s shrinks the final window back from the hand-back (and the rest window back from the padded end)
    half = {r["clip"]: r for r in CL.per_clip_rows(CL.read_series(str(p)), CL.load_manifest(str(mp)), hold_s=3.0, final_s=0.5)}
    assert half["retract__r1.npz"]["windows"]["final"]["cols"] == [125, 150] and half["retract__r1.npz"]["rest"]["cols"] == [375, 400]
    # fallback: no retract_start_frame in the manifest, but the replanner recorded stop.stop_frame in the series meta
    mp2 = tmp_path / "no_rsf.json"
    mp2.write_text(json.dumps({"clips": [dict(rows[0], retract_start_frame=None), rows[1]]}))
    p2 = _retract_series(tmp_path, with_stop=True, name="cl2")
    w2 = {r["clip"]: r for r in CL.per_clip_rows(CL.read_series(str(p2)), CL.load_manifest(str(mp2)), hold_s=3.0, final_s=1.0)}
    assert w2["retract__r1.npz"]["windows"]["final"]["cols"] == [100, 150] and w2["retract__r1.npz"]["windows"]["tail"]["cols"] == [100, 150] and w2["retract__r1.npz"]["rest"]["cols"] == [350, 400]
    # neither -> the plain windows (the rest pose then dominates final: the previous behaviour, now only for rows without a retract segment)
    w3 = {r["clip"]: r for r in CL.per_clip_rows(CL.read_series(str(p)), CL.load_manifest(str(mp2)), hold_s=3.0, final_s=1.0)}
    assert w3["retract__r1.npz"]["windows"]["final"]["cols"] == [350, 400] and w3["retract__r1.npz"]["windows"]["final"]["ee_global_active_cm"] == pytest.approx(40.0) and "rest" not in w3["retract__r1.npz"]
    # the side-by-side report (closed loop only here): the optional trailing rest column + the note, in the json / md and the card
    clp = tmp_path / "closed_loop_summary.json"
    clp.write_text(json.dumps(summ, default=str))
    assert RP.main(["--closed-loop", str(clp), "--manifest", str(mp), "--out", str(tmp_path / "rep"), "--card"]) == 0
    rep = json.loads((tmp_path / "rep" / "hero_bench_report.json").read_text())
    assert rep["columns"][-1] == RP.REST_COL and rep["columns"][:-1] == [c for c, _ in RP.COLUMNS] and rep["notes"] == [CL.RETRACT_NOTE]
    cells = {r["key"]: r["cells"] for r in rep["rows"]}
    assert cells["stratum:retract"][RP.REST_COL] == "40.0 ± 0.0" and cells["all"][RP.REST_COL] == "40.0 ± 0.0" and cells["stratum:h050"][RP.REST_COL] == RP.EMPTY and cells["tier:core"][RP.REST_COL] == RP.EMPTY
    assert cells["stratum:retract"]["global closed-loop final-1s"] == "2.0 ± 0.0" and cells["stratum:retract"]["C3 (CI)"].startswith("100.0%")
    md = (tmp_path / "rep" / "hero_bench_report.md").read_text()
    assert f"| {RP.REST_COL} |" in md and f"Note: {CL.RETRACT_NOTE}." in md
    card = json.loads((tmp_path / "rep" / "results_card.json").read_text())
    assert card["notes"] == [CL.RETRACT_NOTE] and f"Note: {CL.RETRACT_NOTE}." in (tmp_path / "rep" / "RESULTS_CARD.md").read_text()
    # a summary without retract rows: no rest column at all (today's column set)
    p3 = _cl_series(tmp_path, ["h050__x.npz"], ["right"], final_err=[1.0], hold3_err=[2.0], name="plain")
    mp3 = tmp_path / "plain.json"
    mp3.write_text(json.dumps({"clips": [{"file": "h050__x.npz", "hand": "right", "height_label": "h050", "reach_end_frame": 112, "hold_frames": 150, "n_frames": 262}]}))
    plain = CL.summarize([("replan", str(p3))], str(mp3))
    assert "rest" not in plain["results"]["replan"]["groups"]["all"] and "## `rest` window" not in CL.markdown(plain) and CL.flat_rows(plain)[0]["rest_n_clips"] is None
    rep_plain = RP.build_report(None, None, plain, "replan")
    assert RP.REST_COL not in rep_plain["columns"] and rep_plain["notes"] == []


def test_closed_loop_summary_jerk_table(tmp_path):
    T, names = 60, ["h050__a.npz", "h074__b.npz"]
    metrics = {k: np.zeros((2, T), dtype=np.float32) for k in list(METRIC_KEYS) + list(JERK_METRIC_KEYS)}
    metrics["arm_target_delta_rad"][0, 10:25] = 0.02   # clip a: replan at step 11 -> 15-step window carries 0.02
    metrics["arm_target_delta_rad"][1, :] = 0.005      # clip b: no events
    meta = {"replan_log": {"h050__a.npz": {"n_replans": 1, "replan_steps": [11], "events": [], "ik_time_s": 0.1, "goal_offset_cm": 0.0}}}
    series = build_series(sim="mujoco", dt=0.02, horizon_steps=T, meta=meta, clip_name=names, source_tag=["h050", "h074"], clip_len_steps=[T, T], valid_until=[T, T],
                          first_fail_step=[-1, -1], first_fail_cause=["", ""], metrics=metrics)
    sp = write_series(tmp_path / "series.npz", series)
    mp = tmp_path / "BENCH_MANIFEST.json"
    mp.write_text(json.dumps({"clips": [{"file": n, "hand": "right", "reach_end_frame": 20, "n_frames": T, "hold_frames": 10} for n in names]}))
    summ = CL.summarize([("cl", str(sp))], str(mp), jerk_window_s=0.3)
    jk = summ["results"]["cl"]["groups"]["all"]["jerk"]
    assert jk["n_clips"] == 2 and jk["n_clips_with_events"] == 1
    assert jk["post_replan"]["arm_target_delta_rad"] == pytest.approx(0.02) and jk["mean"]["arm_target_delta_rad"] == pytest.approx(0.5 * (0.02 * 15 / T + 0.005))
    per = {r["clip"]: r["jerk"] for r in summ["per_clip"]["cl"]}
    assert per["h074__b.npz"]["post_replan"]["arm_target_delta_rad"] is None and per["h050__a.npz"]["n_replans"] == 1
    md = CL.markdown(summ)
    assert "## jerk" in md and "| cl | 5.00 | 20.00 |" in md


# ================================================================================================ side-by-side report
def _report_inputs(tmp_path: Path):
    """Open-loop summary over strata h050 and floor_pick; closed-loop summary with the floor picks MISSING (the harness could not run them)."""
    names = ["h050__a.npz", "h050__b.npz", "floor_pick__c.npz", "floor_pick__d.npz"]
    hands = ["left", "right", "left", "right"]
    rows = [{"file": names[0], "hand": "left", "height_label": "h050", "reach_end_frame": 4, "hold_frames": 15, "n_frames": 19, "fps": 2, "terminal_ee_pos_err_m": 0.003},
            {"file": names[1], "hand": "right", "height_label": "h050", "reach_end_frame": 4, "hold_frames": 15, "n_frames": 19, "fps": 2, "terminal_ee_pos_err_m": 0.005},
            {"file": names[2], "hand": "left", "height_label": "floor", "stratum": "floor_pick", "tier": "extended", "table": None, "reach_end_frame": 4, "hold_frames": 15, "n_frames": 19, "fps": 2},
            {"file": names[3], "hand": "right", "height_label": "floor", "stratum": "floor_pick", "tier": "extended", "table": None, "reach_end_frame": 4, "hold_frames": 15, "n_frames": 19, "fps": 2}]
    manifest = {"schema": "hero_bench_v1", "n_clips": 4, "clips": rows, "protocol": {"success": {"open_loop": {"S7.5": [7.5, 15], "S5": [5, 10]}, "closed_loop": {"C3": [3, 15]}}}}
    mp = tmp_path / "BENCH_MANIFEST.json"
    mp.write_text(json.dumps(manifest))

    def fill(m):
        for i, (h, v) in enumerate(zip(hands, (3.0, 4.0, 9.0, 11.0))):
            m[f"ee_global_{h}_cm"][i] = v
            m["ee_global_cm"][i] = v + 1.0
            m[f"ee_rot_global_{h}_deg"][i] = 8.0

    s = _read_ol(_ol_series(names, fill, meta={"tag": "t", "termination": OL_TERM}), tmp_path, "ol.npz")
    ol = BS.summarize([s], ["mujoco"], BS.manifest_entries(rows), final_s=1.0, protocol=manifest["protocol"], manifest_path=str(mp))
    olp = tmp_path / "bench_summary.json"
    olp.write_text(json.dumps(ol, default=str))
    mp_cl = tmp_path / "cl_manifest.json"
    mp_cl.write_text(json.dumps({"clips": rows[:2], "protocol": manifest["protocol"]}))
    p = _cl_series(tmp_path, names[:2], hands[:2], final_err=[2.0, 4.0], hold3_err=[5.0, 6.0], T=120, orig=60)
    cl = CL.summarize([("replan", str(p))], str(mp_cl))
    clp = tmp_path / "closed_loop_summary.json"
    clp.write_text(json.dumps(cl, default=str))
    return olp, clp, mp


def test_report_na_harness_cell_and_no_lone_global_number(tmp_path):
    olp, clp, mp = _report_inputs(tmp_path)
    assert RP.main(["--open-loop", str(olp), "--closed-loop", str(clp), "--manifest", str(mp), "--out", str(tmp_path / "rep"), "--card"]) == 0
    rep = json.loads((tmp_path / "rep" / "hero_bench_report.json").read_text())
    rows = {r["key"]: r for r in rep["rows"]}
    assert [r["kind"] for r in rep["rows"]][:3] == ["all", "tier", "tier"] and "cell:floor_pick/floor/left" in rows and "stratum:h050" in rows
    fp = rows["stratum:floor_pick"]["cells"]
    assert fp[RP.GLOBAL_OPEN_COL] == "10.0 ± 1.0" and fp[RP.GLOBAL_CLOSED_COL] == RP.NA_HARNESS and fp["global closed-loop final-1s"] == RP.NA_HARNESS
    assert fp["fail-free open / closed"] == f"100.0% / {RP.NA_HARNESS}" and fp["C3 (CI)"] == RP.NA_HARNESS and fp["stayed"] == RP.NA_HARNESS and rows["stratum:floor_pick"]["n_closed"] is None
    h = rows["stratum:h050"]["cells"]
    assert h[RP.GLOBAL_OPEN_COL] == "3.5 ± 0.5" and h[RP.GLOBAL_CLOSED_COL] == "5.5 ± 0.5" and h["global closed-loop final-1s"] == "3.0 ± 1.0"
    full = RP._ci({"n": 2, "frac": 1.0, "ci95": list(BS.wilson_ci(2, 2))})
    assert full.startswith("100.0% [") and h["S7.5"] == full and h["S5"] == full and h["C3 (CI)"] == "50.0% [9, 91]" and h["stayed"] == "50.0%" and h["IK residual (cm)"] == "0.40"
    assert rep["success_protocol"]["open_loop"]["S5"] == {"pos_cm": 5.0, "rot_deg": 10.0} and rep["success_protocol"]["source"] == {"open_loop": "manifest", "closed_loop": "manifest"}
    assert rep["tiers_run"] == {"open_loop": ["core", "extended"], "closed_loop": ["core"]} and rep["manifest"]["protocol_hash"] and len(rep["manifest"]["sha256"]) == 64
    md = (tmp_path / "rep" / "hero_bench_report.md").read_text()
    assert f"| stratum:floor_pick | 2 / {RP.NA_HARNESS} | 10.0 ± 1.0 | {RP.NA_HARNESS} | {RP.NA_HARNESS} |" in md and "| cell:h050/h050/left | 1 / 1 |" in md
    for r in rep["rows"]:
        if RP._is_number_cell(r["cells"][RP.GLOBAL_OPEN_COL]):
            nb = r["cells"][RP.GLOBAL_CLOSED_COL]
            assert RP._is_number_cell(nb) or nb == RP.NA_HARNESS, r["key"]
    with pytest.raises(ValueError, match="without its replan neighbour"):
        RP.validate_rows([{"key": "x", "cells": {RP.GLOBAL_OPEN_COL: "3.5 ± 0.5", RP.GLOBAL_CLOSED_COL: ""}}])
    with pytest.raises(ValueError):
        RP.validate_rows([{"key": "y", "cells": {RP.GLOBAL_OPEN_COL: "3.5 ± 0.5", RP.GLOBAL_CLOSED_COL: RP.EMPTY}}])
    RP.validate_rows([{"key": "z", "cells": {RP.GLOBAL_OPEN_COL: RP.EMPTY, RP.GLOBAL_CLOSED_COL: "1.0 ± 0.0"}}])
    assert RP.main(["--open-loop", str(olp), "--out", str(tmp_path / "rep2")]) == 0
    rep2 = json.loads((tmp_path / "rep2" / "hero_bench_report.json").read_text())
    assert rep2["closed_loop"] is None and all(r["cells"][RP.GLOBAL_CLOSED_COL] == RP.NA_HARNESS for r in rep2["rows"]) and rep2["tiers_run"]["closed_loop"] == []
    card = json.loads((tmp_path / "rep" / "results_card.json").read_text())
    assert card["protocol_hash"] == rep["manifest"]["protocol_hash"] and card["plant_sha256"].startswith("<plant") and card["policy_sha256"].startswith("<policy")
    assert {r["kind"] for r in card["rows"]} == {"all", "tier", "stratum"} and card["tiers_run"]["open_loop"] == ["core", "extended"]
    cmd = (tmp_path / "rep" / "RESULTS_CARD.md").read_text()
    assert "| protocol hash |" in cmd and "| tiers run |" in cmd and "External reference rows" in cmd and "| cell:" not in cmd and RP.NA_HARNESS in cmd
    with pytest.raises(SystemExit):
        RP.main(["--open-loop", str(olp), "--closed-loop", str(clp), "--out", str(tmp_path / "rep3"), "--card", "--external-labels", "mujoco,other"])
    with pytest.raises(SystemExit):
        RP.main(["--open-loop", str(olp), "--open-label", "nope", "--out", str(tmp_path / "rep4")])
    # the cli module reaches the same entry points
    from sim2sim.bench import cli

    assert cli.report(["--open-loop", str(olp), "--out", str(tmp_path / "rep5")]) == 0


def test_report_closed_loop_only_and_the_termination_block_in_the_card(tmp_path):
    olp, clp, mp = _report_inputs(tmp_path)
    # closed loop only: every open cell is --, the closed cells are numbers, the cell rule holds, the n column prints -- on the open side
    assert RP.main(["--closed-loop", str(clp), "--manifest", str(mp), "--out", str(tmp_path / "co"), "--card"]) == 0
    rep = json.loads((tmp_path / "co" / "hero_bench_report.json").read_text())
    assert rep["open_loop"] is None and rep["closed_loop"]["n_clips"] == 2 and rep["tiers_run"] == {"open_loop": [], "closed_loop": ["core"]}
    rows = {r["key"]: r for r in rep["rows"]}
    h = rows["stratum:h050"]["cells"]
    assert h[RP.GLOBAL_OPEN_COL] == RP.EMPTY and h["local open-loop"] == RP.EMPTY and h["S7.5"] == RP.EMPTY and h["S5"] == RP.EMPTY and h["IK residual (cm)"] == RP.EMPTY
    assert h[RP.GLOBAL_CLOSED_COL] == "5.5 ± 0.5" and h["global closed-loop final-1s"] == "3.0 ± 1.0" and h["fail-free open / closed"] == f"{RP.EMPTY} / 100.0%" and h["C3 (CI)"] == "50.0% [9, 91]"
    assert rows["stratum:h050"]["n_open"] == 0 and rows["stratum:h050"]["n_closed"] == 2 and "stratum:floor_pick" not in rows
    RP.validate_rows(rep["rows"])
    assert rep["success_protocol"]["closed_loop"]["C3"]["pos_cm"] == 3.0 and rep["success_protocol"]["open_loop"]["S5"]["pos_cm"] == 5.0   # thresholds from the closed summary's protocol block
    md = (tmp_path / "co" / "hero_bench_report.md").read_text()
    assert "Open loop **none**" in md and f"| stratum:h050 | {RP.EMPTY} / 2 |" in md and "fail-free rule: causes `fall, anchor_xy`; fall_low margin off" in md
    # the card records the fail-free rule the run used -- from the closed-loop series meta when there is no open-loop summary ...
    card = json.loads((tmp_path / "co" / "results_card.json").read_text())
    assert card["termination"] == {"fail_causes": ["fall", "anchor_xy"], "fall_low_ref_margin_m": None, "fall_pelvis_z": 0.3, "anchor_xy_threshold": 0.5}
    assert "| fail-free rule | causes `fall, anchor_xy`; fall_low margin off |" in (tmp_path / "co" / "RESULTS_CARD.md").read_text()
    # ... and from the open-loop series meta otherwise (here the low-posture margin was on)
    assert RP.main(["--open-loop", str(olp), "--closed-loop", str(clp), "--manifest", str(mp), "--out", str(tmp_path / "both"), "--card"]) == 0
    card2 = json.loads((tmp_path / "both" / "results_card.json").read_text())
    assert card2["termination"]["fail_causes"] == ["fall", "anchor_xy"] and card2["termination"]["fall_low_ref_margin_m"] == 0.2 and card2["termination"]["fall_pelvis_z"] == 0.3
    assert "| fail-free rule | causes `fall, anchor_xy`; fall_low margin 0.2 m |" in (tmp_path / "both" / "RESULTS_CARD.md").read_text()
    rep2 = json.loads((tmp_path / "both" / "hero_bench_report.json").read_text())
    assert rep2["termination"] == card2["termination"] and RP.termination_block({}) == {"fail_causes": None, "fall_low_ref_margin_m": None}
    assert RP._term_text({"termination": RP.termination_block({})}) == "causes `n/a (not recorded in the series)`; fall_low margin off"
    assert RP.termination_block({"fail_causes": ("fall",), "fall_low_ref_margin_m": 0.25}) == {"fail_causes": ["fall"], "fall_low_ref_margin_m": 0.25}   # top-level meta keys as the fallback
    # neither summary: an argparse error; the programmatic entry point refuses too
    with pytest.raises(SystemExit):
        RP.main(["--out", str(tmp_path / "none")])
    with pytest.raises(ValueError):
        RP.build_report(None, None, None, None)


# ================================================================================================ regressions on stored reference results
def _compare(old, new, path: str, diffs: list) -> None:
    """Every value of ``old`` must be in ``new`` (new keys allowed; string lists as subsets; floats to 1e-12)."""
    if isinstance(old, dict):
        if not isinstance(new, dict):
            diffs.append((path, "type"))
            return
        for k, v in old.items():
            if k not in new:
                diffs.append((f"{path}/{k}", "missing"))
            else:
                _compare(v, new[k], f"{path}/{k}", diffs)
    elif isinstance(old, list):
        if not isinstance(new, list):
            diffs.append((path, "type"))
        elif old and all(isinstance(x, str) for x in old):
            if not set(old) <= set(new):
                diffs.append((path, "strlist", old, new))
        elif len(old) != len(new):
            diffs.append((path, "len", len(old), len(new)))
        else:
            for i, (x, y) in enumerate(zip(old, new)):
                _compare(x, y, f"{path}[{i}]", diffs)
    elif isinstance(old, float):
        ok = isinstance(new, (int, float)) and ((math.isnan(old) and math.isnan(new)) or old == new or math.isclose(old, new, rel_tol=1e-12, abs_tol=1e-12))
        if not ok:
            diffs.append((path, old, new))
    elif old != new:
        diffs.append((path, old, new))


@pytest.mark.skipif(not (V1_MANIFEST.is_file() and OL_SERIES.is_file() and OL_SUMMARY.is_file()), reason="needs reference open-loop results under HERO_BENCH_RESULTS_DIR")
def test_regression_open_loop_numbers_match_the_stored_summary(tmp_path):
    out = tmp_path / "ol"
    assert BS.main(["--series", str(OL_SERIES), "--bench-manifest", str(V1_MANIFEST), "--out", str(out)]) == 0
    old = json.loads(OL_SUMMARY.read_text())
    new = json.loads((out / "bench_summary.json").read_text())
    diffs: list = []
    _compare(old, new, "", diffs)
    diffs = [d for d in diffs if not d[0].startswith("/eval_meta")]
    assert not diffs, diffs[:10]
    sim = new["sims"][list(new["sims"])[0]]
    assert sim["strata"] == ["h050", "h074", "h088"] and sim["tiers"] == ["core"] and "success" in sim["groups"]["all"]["all"] and "cell:h050/h050/left" in sim["bench_groups"]
    old_csv = OL_SUMMARY.with_suffix(".csv")
    if old_csv.is_file():
        o = list(csv.DictReader(open(old_csv)))
        n = {tuple(r[k] for k in o[0]) for r in csv.DictReader(open(out / "bench_summary.csv"))}
        assert all(tuple(r[k] for k in o[0]) in n for r in o)


@pytest.mark.skipif(not (V1_MANIFEST.is_file() and CL_SERIES.is_file() and CL_SUMMARY.is_file()), reason="needs reference closed-loop results under HERO_BENCH_RESULTS_DIR")
def test_regression_closed_loop_numbers_match_the_stored_summary(tmp_path):
    old = json.loads(CL_SUMMARY.read_text())
    label = old["labels"][0]
    out = tmp_path / "cl"
    argv = ["--series", f"{label}|{CL_SERIES}", "--manifest", str(V1_MANIFEST), "--out", str(out), "--hold-s", str(old["hold_s"]), "--final-s", str(old["final_s"])]
    if not old.get("per_clip"):
        argv.append("--no-per-clip")
    assert CL.main(argv) == 0
    new = json.loads((out / "closed_loop_summary.json").read_text())
    diffs: list = []
    _compare(old, new, "", diffs)
    diffs = [d for d in diffs if not d[0].startswith("/series/")]   # the stored run's absolute series path differs from the test's copy only by location
    assert not diffs, diffs[:10]
    g = new["results"][label]["groups"]
    assert new["results"][label]["heights"] == ["h050", "h074", "h088"] and "tier:core" in g and "stratum:h050" in g and "cell:h088/h088/right" in g
    assert g["all"]["success"]["levels"]["C3"]["n"] == g["all"]["n_clips"] and g["all"]["success"]["stayed_frac"] == pytest.approx(g["all"]["replan"]["stayed_frac"])
