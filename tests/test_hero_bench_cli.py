"""scripts/hero_bench.py wiring: per-tier run / report orchestration and the closed-loop horizon rule (no MuJoCo needed)."""
from __future__ import annotations

import importlib.util
import hashlib
import io
import json
from pathlib import Path
import sys
import tarfile
from types import SimpleNamespace

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party/holosoma"), str(ROOT / "scripts")]


def load_hero_bench():
    spec = importlib.util.spec_from_file_location("hero_script_hero_bench", ROOT / "scripts" / "hero_bench.py")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_corpus(tmp_path: Path, tiers=("core", "extended"), longest_s=(9.0, 11.0)) -> Path:
    corpus = tmp_path / "corpus"
    rows_all = []
    for tier, seconds in zip(tiers, longest_s):
        rows = [{"file": f"{tier}__clip_{i}.npz", "clip_id": f"clip_{i}", "n_frames": int(seconds * 50) - 50 * i, "fps": 50, "tier": tier} for i in range(2)]
        (corpus / tier).mkdir(parents=True)
        (corpus / tier / "BENCH_MANIFEST.json").write_text(json.dumps({"schema": "hero_bench_v1", "clips": rows}))
        for row in rows:
            (corpus / tier / row["file"]).write_bytes(b"clip")
        rows_all += rows
    (corpus / "BENCH_MANIFEST.json").write_text(json.dumps({"schema": "hero_bench_v1", "clips": rows_all}))
    return corpus


def make_policy(tmp_path):
    policy = tmp_path / "policy"
    policy.mkdir()
    (policy / "model.onnx").write_bytes(b"policy")
    sidecar = policy / "model_hero.json"
    sidecar.write_text(json.dumps({"onnx_file": "model.onnx"}))
    return ["--onnx-dir", str(policy), "--sidecar", str(sidecar)]


def write_stub_series(path, names, errors=None):
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(path, schema=np.array("hero_eval_series_v1"), clip_name=np.array(names), meta_json=np.array(json.dumps({"errors": errors or {}})))


def test_closed_loop_horizon_follows_the_longest_clip_with_a_14_s_floor(tmp_path):
    bench = load_hero_bench()
    corpus = make_corpus(tmp_path, tiers=("core", "extended"), longest_s=(7.0, 11.0))
    assert bench.closed_loop_horizon_s(corpus / "core" / "BENCH_MANIFEST.json") == 14.0      # 7 s + 6 s < 14 s floor
    assert bench.closed_loop_horizon_s(corpus / "extended" / "BENCH_MANIFEST.json") == 17.0  # 11 s + 6 s


def test_run_and_report_dispatch_per_tier_with_the_protocol_flags(tmp_path, monkeypatch):
    bench = load_hero_bench()
    corpus = make_corpus(tmp_path)
    out = tmp_path / "results"
    policy_args = make_policy(tmp_path)
    calls: list[tuple[str, list[str]]] = []

    def fake_run(argv):
        calls.append(("run", list(argv)))
        o = Path(argv[argv.index("--out") + 1])
        o.mkdir(parents=True, exist_ok=True)
        manifest = Path(argv[argv.index("--motion-dir") + 1]) / "BENCH_MANIFEST.json"
        write_stub_series(o / "series.npz", [row["file"] for row in json.loads(manifest.read_text())["clips"]])
        return 0

    def fake_summary(argv):
        calls.append(("summary", list(argv)))
        o = Path(argv[argv.index("--out") + 1]); o.mkdir(parents=True, exist_ok=True); (o / "bench_summary.json").write_text("{}")
        return 0

    def fake_closed(argv):
        calls.append(("closed", list(argv)))
        o = Path(argv[argv.index("--out") + 1]); o.mkdir(parents=True, exist_ok=True); (o / "closed_loop_summary.json").write_text("{}")
        return 0

    def fake_report(argv):
        calls.append(("report", list(argv)))
        o = Path(argv[argv.index("--out") + 1]); o.mkdir(parents=True, exist_ok=True); (o / "hero_bench_report.md").write_text("| table |")
        return 0

    import sim2sim.bench.cli as cli
    monkeypatch.setattr(cli, "run", fake_run)
    monkeypatch.setattr(cli, "summary", fake_summary)
    monkeypatch.setattr(cli, "closed_loop_summary", fake_closed)
    monkeypatch.setattr(cli, "report", fake_report)

    assert bench.main(["run", "--corpus", str(corpus), *policy_args, "--out", str(out),
                       "--tiers", "core,extended", "--jobs", "3"]) == 0
    runs = [a for k, a in calls if k == "run"]
    assert len(runs) == 4  # 2 tiers x (open, closed)
    open_core, closed_core = runs[0], runs[1]
    assert "--replan" not in open_core and open_core[open_core.index("--horizon-s") + 1] == "14"
    assert "--replan" in closed_core and "--goal-adjust" in closed_core and closed_core[closed_core.index("--replan-first") + 1] == "reach_end"
    assert closed_core[closed_core.index("--bench-manifest") + 1] == str(corpus / "core" / "BENCH_MANIFEST.json")
    assert closed_core[closed_core.index("--motion-dir") + 1] == str(corpus / "core")
    assert all(a[a.index("--fail-causes") + 1] == "fall,anchor_xy" and a[a.index("--odom") + 1] == "so" and a[a.index("--jobs") + 1] == "3" for a in runs)
    assert runs[3][runs[3].index("--horizon-s") + 1] == "17"  # extended: longest 11 s clip + 6 s
    # re-running skips existing series
    n_before = len(calls)
    assert bench.main(["run", "--corpus", str(corpus), *policy_args, "--out", str(out), "--tiers", "core"]) == 0
    assert len(calls) == n_before

    assert bench.main(["report", "--corpus", str(corpus), "--results", str(out), "--card"]) == 0
    kinds = [k for k, _ in calls[n_before:]]
    assert kinds == ["summary", "closed", "report", "summary", "closed", "report"]
    report_core = next(a for k, a in calls[n_before:] if k == "report")
    assert "--card" in report_core and "--open-loop" in report_core and "--closed-loop" in report_core
    assert (out / "RESULTS.md").read_text().startswith("# core")


def test_unknown_tier_and_missing_manifest_are_rejected(tmp_path):
    bench = load_hero_bench()
    corpus = make_corpus(tmp_path, tiers=("core",), longest_s=(8.0,))
    with pytest.raises(SystemExit):
        bench.main(["run", "--corpus", str(corpus), "--onnx-dir", "p", "--sidecar", "s", "--out", str(tmp_path / "o"), "--tiers", "bogus"])
    with pytest.raises(SystemExit):
        bench.main(["run", "--corpus", str(corpus), "--onnx-dir", "p", "--sidecar", "s", "--out", str(tmp_path / "o"), "--tiers", "stress"])


def make_archive(path, entries):
    with tarfile.open(path, "w:gz") as tf:
        for entry, payload in entries:
            entry.size = len(payload)
            tf.addfile(entry, io.BytesIO(payload))
    return hashlib.sha256(path.read_bytes()).hexdigest()


@pytest.mark.parametrize("member", ["../escaped.txt", "/escaped.txt", "C:/escaped.txt", "dir\\escaped.txt"])
def test_fetch_rejects_unsafe_paths_without_publishing(tmp_path, member):
    bench = load_hero_bench()
    archive = tmp_path / "corpus.tar.gz"
    digest = make_archive(archive, [(tarfile.TarInfo(member), b"unsafe")])
    out = tmp_path / "out"
    assert bench.main(["fetch", "--url", str(archive), "--sha256", digest, "--out", str(out)]) == 1
    assert not out.exists()


@pytest.mark.parametrize("kind", [tarfile.SYMTYPE, tarfile.LNKTYPE, tarfile.FIFOTYPE])
def test_fetch_rejects_links_and_special_files(tmp_path, kind):
    bench = load_hero_bench()
    archive = tmp_path / "corpus.tar.gz"
    link = tarfile.TarInfo("escape")
    link.type, link.linkname = kind, str(tmp_path)
    make_archive(archive, [(link, b""), (tarfile.TarInfo("escape/escaped.txt"), b"unsafe"), (tarfile.TarInfo("BENCH_MANIFEST.json"), b"{}")])
    out = tmp_path / "out"
    assert bench.main(["fetch", "--url", str(archive), "--out", str(out)]) == 1
    assert not out.exists() and not (tmp_path / "escaped.txt").exists()


def test_fetch_hash_layout_and_existing_destination_are_validated_before_publication(tmp_path):
    bench = load_hero_bench()
    archive = tmp_path / "corpus.tar.gz"
    digest = make_archive(archive, [(tarfile.TarInfo("corpus/BENCH_MANIFEST.json"), b"{}"), (tarfile.TarInfo("corpus/core/clip.npz"), b"clip")])
    out = tmp_path / "out"
    assert bench.main(["fetch", "--url", str(archive), "--sha256", "0" * 64, "--out", str(out)]) == 1
    assert not out.exists()
    out.mkdir()  # an existing empty directory is also supported
    assert bench.main(["fetch", "--url", str(archive), "--sha256", digest.upper(), "--out", str(out)]) == 0
    assert (out / "BENCH_MANIFEST.json").read_text() == "{}" and (out / "core" / "clip.npz").read_bytes() == b"clip"
    assert not (out / "corpus").exists()
    assert bench.main(["fetch", "--url", str(archive), "--out", str(out)]) == 1
    assert (out / "core" / "clip.npz").read_bytes() == b"clip"
    make_archive(archive, [(tarfile.TarInfo("README.txt"), b"no manifest")])
    missing = tmp_path / "missing"
    assert bench.main(["fetch", "--url", str(archive), "--out", str(missing)]) == 1
    assert not missing.exists()


@pytest.mark.parametrize("change", ["policy", "sidecar", "manifest", "odom", "extra", "runtime", "series"])
def test_run_refuses_to_reuse_results_after_inputs_or_configuration_change(tmp_path, monkeypatch, change):
    bench = load_hero_bench()
    corpus = make_corpus(tmp_path, tiers=("core",), longest_s=(8.0,))
    policy_args = make_policy(tmp_path)
    out = tmp_path / "results"
    argv = ["run", "--corpus", str(corpus), *policy_args, "--out", str(out), "--loops", "open", "--jobs", "1"]
    calls = []
    monkeypatch.setattr(bench, "_runtime_fingerprint", lambda: {"implementation": "original"})

    def fake_run(args):
        calls.append(args)
        names = [r["file"] for r in json.loads((corpus / "core" / "BENCH_MANIFEST.json").read_text())["clips"]]
        write_stub_series(out / "core" / "open" / "series.npz", names)
        return 0

    import sim2sim.bench.cli as cli
    monkeypatch.setattr(cli, "run", fake_run)
    assert bench.main(argv) == 0
    assert bench.main([*argv, "--jobs", "3"]) == 0 and len(calls) == 1
    if change == "policy":
        (tmp_path / "policy" / "model.onnx").write_bytes(b"changed policy")
    elif change == "sidecar":
        (tmp_path / "policy" / "model_hero.json").write_text('{"onnx_file":"model.onnx", "changed":true}')
    elif change == "manifest":
        manifest = corpus / "core" / "BENCH_MANIFEST.json"
        data = json.loads(manifest.read_text()); data["clips"][0]["n_frames"] += 1; manifest.write_text(json.dumps(data))
    elif change == "odom":
        argv += ["--odom", "truth"]
    elif change == "extra":
        argv += ["--extra=--reset-vel ref"]
    elif change == "runtime":
        monkeypatch.setattr(bench, "_runtime_fingerprint", lambda: {"implementation": "updated"})
    else:
        write_stub_series(out / "core" / "open" / "series.npz", ["unexpected.npz"])
    assert bench.main(argv) == 1 and len(calls) == 1


@pytest.mark.parametrize("extra", ["--replan", "--pad-s 100", "--odom=truth", "--motion-dir elsewhere", "--replan-first reach_end"])
def test_run_extra_cannot_override_managed_protocol_flags(tmp_path, extra):
    bench = load_hero_bench()
    with pytest.raises(SystemExit, match="managed by the benchmark wrapper"):
        bench.main(["run", "--corpus", str(tmp_path), "--onnx-dir", "p", "--sidecar", "s", "--out", str(tmp_path / "o"), f"--extra={extra}"])


@pytest.mark.parametrize("problem", ["runner_error", "missing", "duplicate", "extra"])
def test_report_refuses_incomplete_or_mismatched_series(tmp_path, monkeypatch, problem):
    bench = load_hero_bench()
    corpus = make_corpus(tmp_path, tiers=("core",), longest_s=(8.0,))
    out = tmp_path / "results"
    names = [r["file"] for r in json.loads((corpus / "core" / "BENCH_MANIFEST.json").read_text())["clips"]]
    errors = {names[1]: "IK error"} if problem == "runner_error" else {}
    if problem in ("missing", "runner_error"):
        names = names[:1]
    elif problem == "duplicate":
        names = [names[0], names[0]]
    else:
        names += ["unknown.npz"]
    write_stub_series(out / "core" / "open" / "series.npz", names, errors)
    import sim2sim.bench.cli as cli
    monkeypatch.setattr(cli, "summary", lambda _: pytest.fail("must not summarize rejected series"))
    assert bench.main(["report", "--corpus", str(corpus), "--results", str(out), "--card"]) == 1
    assert not (out / "RESULTS.md").exists() and not (out / "core" / "report").exists()


def test_report_does_not_use_stale_summaries_after_summary_failure(tmp_path, monkeypatch):
    bench = load_hero_bench()
    corpus = make_corpus(tmp_path, tiers=("core",), longest_s=(8.0,))
    out = tmp_path / "results"
    names = [r["file"] for r in json.loads((corpus / "core" / "BENCH_MANIFEST.json").read_text())["clips"]]
    write_stub_series(out / "core" / "open" / "series.npz", names)
    import sim2sim.bench.cli as cli
    monkeypatch.setattr(cli, "summary", lambda _: 1)
    monkeypatch.setattr(cli, "report", lambda _: pytest.fail("must not report after summary failure"))
    assert bench.main(["report", "--corpus", str(corpus), "--results", str(out)]) == 1


def test_partial_runner_errors_return_failure_and_cannot_be_scored(tmp_path, monkeypatch, capsys):
    pytest.importorskip("mujoco")
    from sim2sim.bench import run as runner
    from sim2sim.bench import cli
    from sim2sim.bench.metrics import METRIC_KEYS
    from sim2sim.bench.rollout import ClipResult
    from sim2sim.bench.series import read_series

    plant = SimpleNamespace(control_dt=0.02, plant_kind="test", describe=lambda: {})
    policy = SimpleNamespace(tag="test", describe=lambda: {"tag": "test"})
    monkeypatch.setattr(runner, "_make_plant_policy", lambda _: (plant, policy))
    clips = [tmp_path / "good.npz", tmp_path / "broken.npz"]
    monkeypatch.setattr(runner, "discover_clips", lambda *_: clips)
    good = ClipResult(clip_name=clips[0].name, source_tag="h050", clip_len_steps=3, valid_until=2, first_fail_step=-1,
                      first_fail_cause="", first_fire={}, metrics={key: np.zeros(2, dtype=np.float32) for key in METRIC_KEYS})
    monkeypatch.setattr(runner, "_run_one", lambda task: (task[0], good, None, 0.0) if task[0] == 0 else (task[0], None, "ValueError: corrupt clip", 0.0))
    out = tmp_path / "partial"
    assert runner.main(["--motion-dir", str(tmp_path), "--out", str(out), "--horizon-s", "0.04", "--quiet", "--jobs", "1"]) == 3
    assert "ERROR broken.npz: ValueError: corrupt clip" in capsys.readouterr().err
    series = read_series(out / "series.npz")
    assert series["meta"]["num_clips"] == 2 and len(series["clip_name"]) == 1
    assert series["meta"]["errors"] == {"broken.npz": "ValueError: corrupt clip"}
    manifest = tmp_path / "BENCH_MANIFEST.json"
    manifest.write_text(json.dumps({"schema": "hero_bench_v1", "clips": [
        {"file": path.name, "stratum": "h050", "height_label": "h050", "tier": "core", "hand": "right", "reach_end_frame": 1, "hold_end_frame": 3,
         "n_frames": 3, "fps": 50} for path in clips]}))
    assert cli.summary(["--series", str(out / "series.npz"), "--bench-manifest", str(manifest), "--out", str(tmp_path / "open_summary")]) == 1
    assert cli.closed_loop_summary(["--series", f"replan|{out / 'series.npz'}", "--manifest", str(manifest), "--out", str(tmp_path / "closed_summary")]) == 1
    assert not (tmp_path / "open_summary").exists() and not (tmp_path / "closed_summary").exists()


def test_report_refuses_changed_manifest_and_mixed_loop_conditions(tmp_path, monkeypatch):
    bench = load_hero_bench()
    corpus = make_corpus(tmp_path, tiers=("core",), longest_s=(8.0,))
    out = tmp_path / "results"
    manifest = corpus / "core" / "BENCH_MANIFEST.json"
    names = [r["file"] for r in json.loads(manifest.read_text())["clips"]]
    config = {"manifest_sha256": bench._sha256(manifest), "condition": {"policy": "A"}}
    for loop in ("open", "closed"):
        write_stub_series(out / "core" / loop / "series.npz", names)
        (out / "core" / loop / "RUN_CONFIG.json").write_text(json.dumps({"configuration": config}))
    import sim2sim.bench.cli as cli
    monkeypatch.setattr(cli, "summary", lambda _: pytest.fail("must not summarize mixed conditions"))
    closed_record = out / "core" / "closed" / "RUN_CONFIG.json"
    closed_record.write_text(json.dumps({"configuration": {**config, "condition": {"policy": "B"}}}))
    argv = ["report", "--corpus", str(corpus), "--results", str(out), "--card"]
    assert bench.main(argv) == 1
    closed_record.write_text(json.dumps({"configuration": config}))
    data = json.loads(manifest.read_text()); data["clips"][0]["n_frames"] += 1; manifest.write_text(json.dumps(data))
    assert bench.main(argv) == 1


@pytest.mark.parametrize("selection", [[], ["--tiers", "core"]])
def test_report_without_series_returns_failure(tmp_path, selection):
    bench = load_hero_bench()
    corpus = make_corpus(tmp_path, tiers=("core",), longest_s=(8.0,))
    assert bench.main(["report", "--corpus", str(corpus), "--results", str(tmp_path / "empty"), *selection]) == 1


def test_report_failure_does_not_aggregate_existing_markdown(tmp_path, monkeypatch):
    bench = load_hero_bench()
    corpus = make_corpus(tmp_path, tiers=("core",), longest_s=(8.0,))
    out = tmp_path / "results"
    names = [r["file"] for r in json.loads((corpus / "core" / "BENCH_MANIFEST.json").read_text())["clips"]]
    write_stub_series(out / "core" / "open" / "series.npz", names)
    old = out / "core" / "report" / "hero_bench_report.md"
    old.parent.mkdir(); old.write_text("old report")
    import sim2sim.bench.cli as cli
    monkeypatch.setattr(cli, "summary", lambda _: 0)
    monkeypatch.setattr(cli, "report", lambda _: 1)
    assert bench.main(["report", "--corpus", str(corpus), "--results", str(out)]) == 1
    assert not (out / "RESULTS.md").exists()
