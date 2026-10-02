#!/usr/bin/env python3
"""hero_bench_v1 command line: ``plan`` (print the stratum table), ``build`` (build the corpus from generator banks + the paper-protocol
files) and ``verify`` (re-hash a built corpus against its BENCH_MANIFEST.json / SHA256SUMS).

``fetch`` downloads a frozen corpus archive (``.tar.gz``) and checks its sha256, ``run`` rolls a policy over the tiers (open loop and the
closed-loop replan protocol), ``report`` turns the series into the open-loop / closed-loop summaries, the side-by-side report and the
public results card (``sim2sim.bench``; mujoco, mink, onnxruntime: ``pip install -e '.[bench]'``).

    python scripts/hero_bench.py plan
    python scripts/hero_bench.py build --plan default --bank-root <root> --paper-dir <paper> --out <out> --generate --jobs 60 [--public]
    python scripts/hero_bench.py verify --dir <out>
    python scripts/hero_bench.py fetch --url https://.../hero_bench_v1_corpus_<sha8>.tar.gz --sha256 <hex> --out data/hero_bench_v1
    python scripts/hero_bench.py run --corpus data/hero_bench_v1 --onnx-dir <export> --sidecar <*_hero.json> --out results/<policy> [--tiers core]
    python scripts/hero_bench.py report --corpus data/hero_bench_v1 --results results/<policy> [--card]
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import shlex
import sys
from collections import Counter
from pathlib import Path, PurePosixPath, PureWindowsPath

from _bootstrap import ROOT  # noqa: F401  (puts the release root and third_party/holosoma on sys.path)

TIERS: tuple[str, ...] = ("core", "extended", "stress")
#: Common fail-free rule of the protocol (HERO-style exports have no pelvis reference; ``fall_low,anchor_xy`` is the low-posture diagnostic).
FAIL_CAUSES = "fall,anchor_xy"
OPEN_LOOP_HORIZON_S = 14.0
#: Closed-loop replan protocol (reach_end-anchored, 3 s period, re-plan from the current pose, goal adjustment, 0.3 s blend, 4 s padding).
REPLAN_ARGS: tuple[str, ...] = ("--pad-s", "4", "--replan", "--replan-first", "reach_end", "--replan-period-s", "3.0", "--replan-base", "current",
                                "--goal-adjust", "--replan-blend-s", "0.3")


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _series_issue(path: Path, manifest: Path) -> str | None:
    """A tier result must contain every manifest clip exactly once, with no runner errors."""
    import numpy as np

    try:
        with np.load(path, allow_pickle=False) as data:
            if str(data["schema"].item()) != "hero_eval_series_v1":
                return "unexpected series schema"
            names = [str(x) for x in data["clip_name"]]
            meta = json.loads(str(data["meta_json"].item()))
        if meta.get("errors"):
            return f"runner errors: {meta['errors']}"
        record = path.parent / "RUN_CONFIG.json"
        if record.is_file():
            configuration = json.loads(record.read_text()).get("configuration") or {}
            if configuration.get("manifest_sha256") != _sha256(manifest):
                return "manifest differs from the one recorded for this run"
        expected = [str(row["file"]) for row in json.loads(manifest.read_text())["clips"]]
        if Counter(names) != Counter(expected) or any(n != 1 for n in Counter(names).values()):
            return (f"clip coverage differs from manifest: {len(names)} results / {len(expected)} expected; "
                    f"missing {sorted((Counter(expected) - Counter(names)).elements())[:5]}, "
                    f"extra {sorted((Counter(names) - Counter(expected)).elements())[:5]}")
    except (OSError, ValueError, KeyError, TypeError) as exc:
        return f"cannot validate series: {exc}"
    return None


def _runtime_fingerprint() -> dict:
    import os
    import platform
    from importlib.metadata import PackageNotFoundError, version
    from hero_isaacsim.paths import G1_ASSET_ROOT

    versions = {}
    for package in ("numpy", "mujoco", "mink", "onnxruntime", "scipy", "daqp", "osqp", "qpsolvers", "quadprog"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = None
    sources = {Path(__file__).resolve()}
    for directory in (ROOT / "sim2sim", ROOT / "sim2sim" / "bench", ROOT / "data_tools", ROOT / "hero_isaacsim"):
        sources.update(directory.glob("*.py"))
    resources = {p for p in G1_ASSET_ROOT.rglob("*") if p.is_file()}
    override = os.environ.get("HERO_SIM2SIM_URDF")
    if override and Path(override).expanduser().is_file():
        resources.add(Path(override).expanduser())
    return {"python": sys.version, "platform": platform.platform(), "packages": versions,
            "environment": {name: os.environ.get(name) for name in ("HERO_ROOT", "HERO_ASSETS_ROOT", "HERO_ROBOT_ASSET_ROOT", "HERO_SIM2SIM_URDF")},
            "source_sha256": {str(p.resolve()): _sha256(p) for p in sorted(sources)},
            "resource_sha256": {str(p.resolve()): _sha256(p) for p in sorted(resources)}}


def _run_configuration(argv: list[str], manifest: Path, policy_files: dict[str, str], runtime: dict) -> dict:
    # Worker count, output path and console verbosity do not change a rollout.
    kept: list[str] = []
    i = 0
    while i < len(argv):
        flag = argv[i]
        if flag in ("--jobs", "--out"):
            i += 2
        elif flag == "--quiet":
            i += 1
        else:
            kept.append(flag)
            i += 1
    clips = {p.name: _sha256(p) for p in sorted(manifest.parent.glob("*.npz"))}
    expected = {str(row["file"]): row for row in json.loads(manifest.read_text())["clips"]}
    if set(clips) != set(expected):
        raise SystemExit(f"hero_bench run: clip files differ from {manifest}; run verify before evaluating")
    for name, row in expected.items():
        if row.get("sha256") and row["sha256"] != clips[name]:
            raise SystemExit(f"hero_bench run: {manifest.parent / name}: sha256 differs from manifest")
    files = dict(policy_files)
    for value in kept:
        p = Path(value).expanduser()
        if p.is_file():
            files[str(p.resolve())] = _sha256(p)
    return {"schema": "hero_bench_run_config_v1", "argv": kept, "manifest_sha256": _sha256(manifest), "clip_sha256": clips, "input_sha256": files,
            "runtime": runtime}


def _build_help() -> str:
    return "build the corpus; takes the flags of data_tools.build_hero_bench (--plan --bank-root --paper-dir --out --generate --jobs --nice --verifier --strata --public --no-clean --code-sha256)"


def _tier_manifest(corpus: Path, tier: str) -> Path:
    m = corpus / tier / "BENCH_MANIFEST.json"
    if not m.exists():
        available = sorted(p.name for p in corpus.iterdir() if (p / "BENCH_MANIFEST.json").exists()) if corpus.is_dir() else []
        raise SystemExit(f"hero_bench: {m} missing (tiers available: {available})")
    return m


def closed_loop_horizon_s(manifest_path: Path, floor_s: float = OPEN_LOOP_HORIZON_S) -> float:
    """Protocol rule: ``max(14 s, ceil(longest clip + 6 s))`` so slow / long-hold layers are not truncated."""
    rows = json.loads(manifest_path.read_text()).get("clips", [])
    longest = max((float(r.get("n_frames", 0)) / float(r.get("fps", 50) or 50) for r in rows), default=0.0)
    return max(floor_s, float(math.ceil(longest + 6.0)))


def _tiers(text: str | None, corpus: Path | None = None, results: Path | None = None) -> list[str]:
    if text in (None, "", "all"):
        if text == "all" or results is None:
            return list(TIERS)
        return [t for t in TIERS if (results / t).is_dir()]
    tiers = [t.strip() for t in text.split(",") if t.strip()]
    if not tiers:
        raise SystemExit("hero_bench: --tiers must name at least one tier")
    bad = [t for t in tiers if t not in TIERS]
    if bad:
        raise SystemExit(f"hero_bench: unknown tiers {bad}; choose from {list(TIERS)}")
    return tiers


def cmd_fetch(args: argparse.Namespace) -> int:
    import shutil
    import tarfile
    import tempfile
    import urllib.request

    out = Path(args.out).expanduser()
    if out.is_symlink() or (out.exists() and (not out.is_dir() or any(out.iterdir()))):
        print(f"hero_bench fetch: {out} must be an empty directory or a new path", file=sys.stderr)
        return 1
    out.parent.mkdir(parents=True, exist_ok=True)
    # Keep downloads and extraction away from the destination until all members and the layout pass validation.
    with tempfile.TemporaryDirectory(dir=out.parent, prefix=".hero_bench_fetch_") as tmp:
        archive = Path(tmp) / "corpus.tar.gz"
        staging = Path(tmp) / "extracted"
        staging.mkdir()
        src = args.url
        if "://" in src:
            print(f"hero_bench fetch: downloading {src}")
            urllib.request.urlretrieve(src, archive)
        else:
            shutil.copyfile(src, archive)
        digest = _sha256(archive)
        if args.sha256 and digest != args.sha256.lower():
            print(f"hero_bench fetch: sha256 mismatch: got {digest}, expected {args.sha256}", file=sys.stderr)
            return 1
        print(f"hero_bench fetch: sha256 {digest}{' (verified)' if args.sha256 else ' (unverified; pass --sha256)'}")
        with tarfile.open(archive, "r:*") as tf:
            members = tf.getmembers()
            for m in members:
                name = PurePosixPath(m.name)
                if name.is_absolute() or ".." in name.parts or "\\" in m.name or PureWindowsPath(m.name).drive or not (m.isfile() or m.isdir()):
                    print(f"hero_bench fetch: refusing unsafe archive member {m.name!r}", file=sys.stderr)
                    return 1
            tf.extractall(staging)
        # Archives may carry one top-level directory. Publish the corpus root itself without mixing trees.
        root = staging
        if not (root / "BENCH_MANIFEST.json").is_file():
            tops = list(staging.iterdir())
            if len(tops) == 1 and tops[0].is_dir() and (tops[0] / "BENCH_MANIFEST.json").is_file():
                root = tops[0]
        if not (root / "BENCH_MANIFEST.json").is_file():
            print(f"hero_bench fetch: no BENCH_MANIFEST.json under archive root", file=sys.stderr)
            return 1
        if out.exists():
            out.rmdir()
        shutil.move(str(root), str(out))
    print(f"hero_bench fetch: corpus at {out}; run `python scripts/hero_bench.py verify --dir {out}` to check every file")
    return 0


def cmd_run(args: argparse.Namespace) -> int:
    from sim2sim.bench import cli
    from sim2sim.policy_hero_export import resolve_hero_export

    corpus, out = Path(args.corpus), Path(args.out)
    loops = [x.strip() for x in args.loops.split(",") if x.strip()]
    if not loops or any(x not in ("open", "closed") for x in loops):
        raise SystemExit("hero_bench run: --loops takes open, closed or open,closed")
    extra = shlex.split(args.extra) if args.extra else []
    managed = ("--onnx-dir", "--sidecar", "--motion-dir", "--out", "--horizon-s", "--bench-manifest", "--jobs", "--fail-causes", "--odom", "--odom-seed",
               "--pad-s", "--replan", "--replan-first", "--replan-period-s", "--replan-base", "--goal-adjust", "--replan-blend-s")
    for token in extra:
        flag = token.split("=", 1)[0]
        if flag.startswith("--") and any(option == flag or option.startswith(flag) for option in managed):
            raise SystemExit(f"hero_bench run: {flag} is managed by the benchmark wrapper; use its named option or sim2sim.bench.run for a custom protocol")
    manifests = {tier: _tier_manifest(corpus, tier) for tier in _tiers(args.tiers)}
    onnx, sidecar = resolve_hero_export(args.onnx_dir, sidecar=args.sidecar)
    policy_files = {str(p): _sha256(p) for p in (onnx, sidecar) if p is not None}
    runtime = _runtime_fingerprint()
    rc_all = 0
    for tier, manifest in manifests.items():
        base = ["--onnx-dir", args.onnx_dir, "--sidecar", args.sidecar, "--motion-dir", str(corpus / tier), "--jobs", str(args.jobs), "--quiet",
                "--fail-causes", args.fail_causes, "--odom", args.odom, "--odom-seed", str(args.odom_seed), *extra]
        for loop in ("open", "closed"):
            if loop not in loops:
                continue
            d = out / tier / loop
            horizon = OPEN_LOOP_HORIZON_S if loop == "open" else closed_loop_horizon_s(manifest)
            argv = [*base, "--horizon-s", f"{horizon:g}"]
            if loop == "closed":
                argv += [*REPLAN_ARGS, "--bench-manifest", str(manifest)]
            argv += ["--out", str(d)]
            config = _run_configuration(argv, manifest, policy_files, runtime)
            config["condition"] = {"policy_sha256": policy_files, "fail_causes": args.fail_causes, "odom": args.odom, "odom_seed": args.odom_seed, "extra": extra}
            record, series = d / "RUN_CONFIG.json", d / "series.npz"
            if series.exists():
                try:
                    previous = json.loads(record.read_text())
                except (OSError, ValueError):
                    previous = {}
                if previous.get("configuration") != config or previous.get("series_sha256") != _sha256(series):
                    print(f"hero_bench run: {tier}/{loop} exists with a different or unrecorded configuration; choose a new --out", file=sys.stderr)
                    rc_all |= 1
                    continue
                issue = _series_issue(series, manifest)
                if issue:
                    print(f"hero_bench run: {tier}/{loop}: {issue}; choose a new --out", file=sys.stderr)
                    rc_all |= 1
                    continue
                print(f"hero_bench run: {tier}/{loop} exists with matching configuration, skipping")
                continue
            print(f"hero_bench run: {tier} {loop} loop (horizon {horizon:g} s)")
            rc = cli.run(argv)
            if rc == 0:
                issue = _series_issue(series, manifest)
                if issue:
                    print(f"hero_bench run: {tier}/{loop}: {issue}", file=sys.stderr)
                    rc = 1
                else:
                    record.write_text(json.dumps({"configuration": config, "series_sha256": _sha256(series)}, indent=2) + "\n")
            rc_all |= rc
    return rc_all


def cmd_report(args: argparse.Namespace) -> int:
    from sim2sim.bench import cli

    corpus, results = Path(args.corpus), Path(args.results)
    rc_all = 0
    written: list[str] = []
    for tier in _tiers(args.tiers, results=results):
        manifest = _tier_manifest(corpus, tier)
        open_series, closed_series = results / tier / "open" / "series.npz", results / tier / "closed" / "series.npz"
        if not open_series.exists() and not closed_series.exists():
            print(f"hero_bench report: {tier}: no series, skipping")
            continue
        issues = [(path, _series_issue(path, manifest)) for path in (open_series, closed_series) if path.exists()]
        if any(issue for _, issue in issues):
            for path, issue in issues:
                if issue:
                    print(f"hero_bench report: refusing {path}: {issue}", file=sys.stderr)
            rc_all |= 1
            continue
        open_record, closed_record = open_series.parent / "RUN_CONFIG.json", closed_series.parent / "RUN_CONFIG.json"
        if open_series.exists() and closed_series.exists() and open_record.is_file() and closed_record.is_file():
            open_config = json.loads(open_record.read_text())["configuration"]
            closed_config = json.loads(closed_record.read_text())["configuration"]
            if any(open_config.get(key) != closed_config.get(key) for key in ("condition", "runtime", "manifest_sha256", "clip_sha256")):
                print(f"hero_bench report: refusing {tier}: open and closed runs used different policies, inputs or evaluation conditions", file=sys.stderr)
                rc_all |= 1
                continue
        report_args = ["--manifest", str(manifest), "--out", str(results / tier / "report")]
        summary_rc = 0
        if open_series.exists():
            summary_rc |= cli.summary(["--series", str(open_series), "--bench-manifest", str(manifest), "--out", str(results / tier / "open" / "summary"), "--group-by", "stratum"])
            report_args += ["--open-loop", str(results / tier / "open" / "summary" / "bench_summary.json")]
        if closed_series.exists():
            summary_rc |= cli.closed_loop_summary(["--series", f"replan|{closed_series}", "--manifest", str(manifest), "--out", str(results / tier / "closed" / "summary")])
            report_args += ["--closed-loop", str(results / tier / "closed" / "summary" / "closed_loop_summary.json")]
        if summary_rc:
            rc_all |= summary_rc
            continue
        if args.card:
            report_args.append("--card")
            if args.policy_sha:
                report_args += ["--policy-sha", args.policy_sha]
            if args.plant_sha:
                report_args += ["--plant-sha", args.plant_sha]
        report_rc = cli.report(report_args)
        rc_all |= report_rc
        if report_rc == 0:
            written.append(tier)
    if written:
        parts = []
        for tier in written:
            md = results / tier / "report" / "hero_bench_report.md"
            if md.exists():
                parts.append(f"# {tier}\n\n" + md.read_text())
        (results / "RESULTS.md").write_text("\n\n".join(parts) + "\n")
        print(f"hero_bench report: {results / 'RESULTS.md'} ({', '.join(written)})")
    else:
        print(f"hero_bench report: no reports written under {results}", file=sys.stderr)
        rc_all |= 1
    return rc_all


def main(argv: list[str] | None = None) -> int:
    argv = list(sys.argv[1:] if argv is None else argv)
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = ap.add_subparsers(dest="cmd", required=True)
    p_plan = sub.add_parser("plan", help="print the stratum plan table (default plan or --plan <json>)")
    p_plan.add_argument("--plan", default="default", help="'default', 'reduced:<per_cell>' or a plan JSON file")
    p_plan.add_argument("--json", action="store_true", help="print the plan as JSON (bench_v1_plan_json form) instead of the table")
    sub.add_parser("build", help=_build_help(), add_help=False)
    p_verify = sub.add_parser("verify", help="re-hash a built corpus against BENCH_MANIFEST.json and SHA256SUMS (exit 1 on any mismatch)")
    p_verify.add_argument("--dir", required=True, help="corpus root (contains BENCH_MANIFEST.json, SHA256SUMS, core/ extended/ stress/)")
    p_verify.add_argument("--json", action="store_true", help="print the full verification result as JSON")
    p_fetch = sub.add_parser("fetch", help="download a frozen corpus archive (.tar.gz), check its sha256 and extract it")
    p_fetch.add_argument("--url", required=True, help="archive URL (https://... or a local path)")
    p_fetch.add_argument("--out", required=True, help="directory that will contain BENCH_MANIFEST.json, core/ extended/ stress/")
    p_fetch.add_argument("--sha256", default=None, help="expected sha256 of the archive (recommended)")
    p_run = sub.add_parser("run", help="roll a policy over the corpus tiers: open loop + closed-loop replan protocol")
    p_run.add_argument("--corpus", required=True, help="corpus root from fetch/build (per-tier BENCH_MANIFEST.json inside each tier dir)")
    p_run.add_argument("--onnx-dir", required=True, help="exported policy: directory with model_XXXXX.onnx + model_XXXXX_hero.json, or the .onnx file")
    p_run.add_argument("--sidecar", required=True, help="the policy's *_hero.json metadata")
    p_run.add_argument("--out", required=True, help="results root: <out>/<tier>/{open,closed}/series.npz")
    p_run.add_argument("--tiers", default="core", help="comma-separated tiers to run (default core; all = core,extended,stress)")
    p_run.add_argument("--loops", default="open,closed", help="open, closed or open,closed")
    p_run.add_argument("--jobs", type=int, default=8)
    p_run.add_argument("--fail-causes", default=FAIL_CAUSES, help=f"fail-free rule (default {FAIL_CAUSES}; low-posture diagnostic: fall_low,anchor_xy)")
    p_run.add_argument("--odom", default="so", help="root pose source fed to anchor terms / odometry metrics (so = LiDAR-inertial model, truth = diagnostic)")
    p_run.add_argument("--odom-seed", type=int, default=0)
    p_run.add_argument("--extra", default="", help="extra sim2sim.bench.run flags (one shell-quoted string; cannot override the wrapper's managed protocol flags)")
    p_report = sub.add_parser("report", help="summaries + side-by-side report (+ results card) from a run directory")
    p_report.add_argument("--corpus", required=True)
    p_report.add_argument("--results", required=True, help="the --out of `run`")
    p_report.add_argument("--tiers", default=None, help="tiers to report (default: those present under --results)")
    p_report.add_argument("--card", action="store_true", help="also write the public results card (RESULTS_CARD.md / results_card.json)")
    p_report.add_argument("--policy-sha", default=None)
    p_report.add_argument("--plant-sha", default=None)
    if argv and argv[0] == "build":
        from data_tools import build_hero_bench as B

        return B.main(argv[1:])
    args = ap.parse_args(argv)
    if args.cmd == "fetch":
        return cmd_fetch(args)
    if args.cmd == "run":
        return cmd_run(args)
    if args.cmd == "report":
        return cmd_report(args)
    from data_tools import build_hero_bench as B

    if args.cmd == "plan":
        try:
            plan = B.load_plan(args.plan)
        except B.BuildError as exc:
            ap.error(str(exc))
        print(json.dumps(B.plan_json(plan), indent=1) if args.json else B.plan_table(plan))
        return 0
    if args.cmd == "verify":
        try:
            res = B.verify_shipped(Path(args.dir))
        except FileNotFoundError as exc:
            print(f"hero_bench verify: {exc}", file=sys.stderr)
            return 1
        if args.json:
            print(json.dumps(res, indent=1))
        else:
            print(f"hero_bench verify {res['dir']}: {'OK' if res['ok'] else 'FAILED'} -- {res['n_identical']} / {res['n_rows']} manifest rows byte-identical, "
                  f"{len(res['mismatched'])} mismatched, {len(res['missing'])} missing, {len(res['extra_npz'])} extra; SHA256SUMS {res['sums_checked']} checked, "
                  f"{len(res['sums_mismatched'])} mismatched, {len(res['sums_missing'])} missing; protocol sha256 "
                  f"{'matches' if res['protocol_sha256'] == res['protocol_sha256_recomputed'] else 'DIFFERS'}")
            for key in ("mismatched", "missing", "extra_npz", "sums_mismatched", "sums_missing"):
                for item in res[key][:20]:
                    print(f"  {key}: {item}")
            for tier, check in res["tier_manifests"].items():
                if not check["consistent_with_root"] or not check["protocol_sha256_matches"]:
                    print(f"  tier_manifest: {tier}: rows / protocol differ from root" + (f" ({check['error']})" if check.get("error") else ""))
        return 0 if res["ok"] else 1
    ap.error(f"unknown command {args.cmd!r}")
    return 1


if __name__ == "__main__":
    raise SystemExit(main())
