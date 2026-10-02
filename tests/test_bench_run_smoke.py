"""End-to-end smoke of the benchmark scoring chain on two hero_bench_v1 clips with the example policy: open loop, closed loop (replanning +
goal adjustment, LiDAR-inertial odometry), the two summaries, the side-by-side report and the results card.  Skipped without mujoco /
onnxruntime / mink, the example checkpoint or a local hero_bench_v1 corpus (``HERO_BENCH_V1_DIR``)."""
from __future__ import annotations

import json

import numpy as np
import pytest

from _bench_fixtures import BENCH_DIR, EXAMPLE_POLICY, EXAMPLE_SIDECAR, SKIP_BENCH, SKIP_MINK, SKIP_POLICY

pytestmark = [pytest.mark.mujoco, pytest.mark.bench_data, SKIP_MINK, SKIP_POLICY, SKIP_BENCH]


@pytest.fixture(scope="module")
def chain(tmp_path_factory):
    from sim2sim.bench import cli

    out = tmp_path_factory.mktemp("bench_smoke")
    manifest = str(BENCH_DIR / "BENCH_MANIFEST.json")
    common = ["--onnx-dir", str(EXAMPLE_POLICY), "--sidecar", str(EXAMPLE_SIDECAR), "--motion-dir", str(BENCH_DIR), "--max-clips", "2", "--quiet",
              "--fail-causes", "fall,anchor_xy", "--odom", "so"]
    assert cli.run(common + ["--horizon-s", "4", "--out", str(out / "open")]) == 0
    assert cli.run(common + ["--pad-s", "1", "--horizon-s", "6", "--replan", "--bench-manifest", manifest, "--replan-first", "reach_end", "--replan-period-s", "3.0",
                             "--replan-base", "current", "--goal-adjust", "--replan-blend-s", "0.3", "--out", str(out / "closed")]) == 0
    assert cli.summary(["--series", str(out / "open" / "series.npz"), "--bench-manifest", manifest, "--out", str(out / "open" / "summary"), "--group-by", "stratum"]) == 0
    assert cli.closed_loop_summary(["--series", f"replan|{out / 'closed' / 'series.npz'}", "--manifest", manifest, "--out", str(out / "closed" / "summary"), "--csv", str(out / "closed" / "summary" / "flat.csv")]) == 0
    assert cli.report(["--open-loop", str(out / "open" / "summary" / "bench_summary.json"), "--closed-loop", str(out / "closed" / "summary" / "closed_loop_summary.json"),
                       "--manifest", manifest, "--out", str(out / "report"), "--card"]) == 0
    return out


def test_open_loop_series_and_summary(chain):
    from sim2sim.bench.metrics import METRIC_KEYS, ODOM_METRIC_KEYS
    from sim2sim.bench.series import read_series

    s = read_series(chain / "open" / "series.npz")
    assert s["sim"] == "mujoco" and len(s["clip_name"]) == 2 and int(s["horizon_steps"]) == 200 and float(s["dt"]) == pytest.approx(0.02)
    assert set(s["metrics"]) == set(METRIC_KEYS) | set(ODOM_METRIC_KEYS)
    for i in range(2):
        vu = int(s["valid_until"][i])
        assert vu == min(200, int(s["clip_len_steps"][i]) - 1)
        for k in ("ee_global_cm", "ee_local_cm", "anchor_xy_cm", "kp_global_pos_cm", "odom_xy_err_cm"):
            col = s["metrics"][k][i]
            assert np.isfinite(col[:vu]).all() and np.isnan(col[vu:]).all(), k
    meta = s["meta"]
    assert meta["policy_kind"] == "hero_export" and meta["odometry"]["source"] == "so" and meta["odometry"]["fed_to_policy"] is True and meta["replan"] is None
    assert meta["termination"]["fail_causes"] == ["fall", "anchor_xy"] and meta["plant"]["physics_dt"] == pytest.approx(0.001) and meta["plant"]["substeps"] == 20
    summ = json.loads((chain / "open" / "summary.json").read_text())
    assert summ["overall"]["n_clips"] == 2 and "odometry" in summ and "keypoints" in summ
    bs = json.loads((chain / "open" / "summary" / "bench_summary.json").read_text())
    sim = bs["sims"]["mujoco"]
    assert sim["n_matched"] == 2 and sim["tiers"] == ["core"] and "success" in sim["groups"]["all"]["all"] and "odometry" in sim["groups"]["all"]["all"]
    assert (chain / "open" / "summary" / "bench_summary.csv").is_file() and "Active (reaching) hand" in (chain / "open" / "summary" / "bench_summary.md").read_text()


def test_closed_loop_series_summary_and_report(chain):
    from sim2sim.bench.rollout import JERK_METRIC_KEYS
    from sim2sim.bench.series import read_series

    s = read_series(chain / "closed" / "series.npz")
    assert set(JERK_METRIC_KEYS) <= set(s["metrics"]) and int(s["horizon_steps"]) == 300
    meta = s["meta"]
    assert meta["replan"]["handover_blend_s"] == 0.3 and meta["replan"]["base_mode"] == "current" and meta["replan"]["goal_adjust"] is True
    log = meta["replan_log"]
    assert set(log) == set(str(n) for n in s["clip_name"])
    for name, entry in log.items():
        assert entry["n_replans"] >= 0 and all(p["qp_failures"] == 0 for p in entry["plans"])
        assert all(st >= 1 for st in entry["replan_steps"])
    cl = json.loads((chain / "closed" / "summary" / "closed_loop_summary.json").read_text())
    g = cl["results"]["replan"]["groups"]
    assert cl["labels"] == ["replan"] and g["all"]["n_clips"] == 2 and "C3" in g["all"]["success"]["levels"] and "jerk" in g["all"] and "replan" in g["all"]
    assert (chain / "closed" / "summary" / "flat.csv").is_file()
    rep = json.loads((chain / "report" / "hero_bench_report.json").read_text())
    rows = {r["key"]: r for r in rep["rows"]}
    assert rows["all"]["n_open"] == 2 and rows["all"]["n_closed"] == 2
    cells = rows["all"]["cells"]
    assert cells["global open-loop (hold)"][0].isdigit() and cells["global closed-loop replan+adjust (hold3)"][0].isdigit()
    card = json.loads((chain / "report" / "results_card.json").read_text())
    assert card["odometry"]["source"] == "so" and card["tiers_run"] == {"open_loop": ["core"], "closed_loop": ["core"]} and len(card["policy_sha256"]) == 64
    assert (chain / "report" / "RESULTS_CARD.md").is_file() and "n/a (harness)" not in (chain / "report" / "hero_bench_report.md").read_text().split("### Per layer x height x hand")[0].split("### All clips")[1]
