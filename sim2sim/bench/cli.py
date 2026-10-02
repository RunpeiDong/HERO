"""Entry points of the benchmark tooling for a wrapper script (``scripts/hero_bench.py``) or ``python -m``.

Each function takes an ``argv`` list (the module's own command-line flags, see the module docstrings) and returns the process exit code:

* :func:`run` -- :mod:`sim2sim.bench.run` (roll the policy over the clips -> ``series.npz`` / ``summary.json`` / ``summary.md``);
* :func:`summary` -- :mod:`sim2sim.bench.summary` (open-loop per-height / per-layer tables + success columns -> ``bench_summary.json`` / ``.md`` / ``.csv``);
* :func:`closed_loop_summary` -- :mod:`sim2sim.bench.closed_loop_summary` (hold3 / final / tail windows, C3, replanner bookkeeping -> ``closed_loop_summary.json`` / ``.md``);
* :func:`report` -- :mod:`sim2sim.bench.report` (open-loop and closed-loop side by side, ``n/a (harness)`` cells, ``--card``).

Intended wiring of ``scripts/hero_bench.py`` (the script owns the per-tier defaults of the protocol; this module owns the flags)::

    from _bootstrap import ROOT               # puts the checkout on sys.path
    from sim2sim.bench import cli
    cli.run(["--onnx-dir", onnx_dir, "--sidecar", sidecar, "--motion-dir", bench_dir, "--out", f"{out}/open", "--jobs", "8", "--quiet",
             "--fail-causes", "fall,anchor_xy", "--odom", "so"])
    cli.run([... "--pad-s", "4", "--horizon-s", "14", "--replan", "--bench-manifest", manifest, "--replan-first", "reach_end",
             "--replan-period-s", "3.0", "--replan-base", "current", "--goal-adjust", "--replan-blend-s", "0.3", "--out", f"{out}/closed"])
    cli.summary(["--series", f"{out}/open/series.npz", "--bench-manifest", manifest, "--out", f"{out}/open/summary"])
    cli.closed_loop_summary(["--series", f"replan|{out}/closed/series.npz", "--manifest", manifest, "--out", f"{out}/closed/summary"])
    cli.report(["--open-loop", f"{out}/open/summary/bench_summary.json", "--closed-loop", f"{out}/closed/summary/closed_loop_summary.json",
                "--manifest", manifest, "--out", f"{out}/report", "--card"])

The heavy imports (mujoco, onnxruntime, mink) happen inside the functions, so importing this module is cheap.
"""

from __future__ import annotations

from typing import Sequence


def run(argv: Sequence[str] | None = None) -> int:
    """Roll a policy over benchmark clips (``python -m sim2sim.bench.run`` flags)."""
    from sim2sim.bench.run import main

    return int(main(list(argv) if argv is not None else None))


def summary(argv: Sequence[str] | None = None) -> int:
    """Open-loop benchmark tables (``python -m sim2sim.bench.summary`` flags)."""
    from sim2sim.bench.summary import main

    return int(main(list(argv) if argv is not None else None))


def closed_loop_summary(argv: Sequence[str] | None = None) -> int:
    """Closed-loop benchmark tables (``python -m sim2sim.bench.closed_loop_summary`` flags)."""
    from sim2sim.bench.closed_loop_summary import main

    return int(main(list(argv) if argv is not None else None))


def report(argv: Sequence[str] | None = None) -> int:
    """Side-by-side report + results card (``python -m sim2sim.bench.report`` flags)."""
    from sim2sim.bench.report import main

    return int(main(list(argv) if argv is not None else None))


__all__ = ["closed_loop_summary", "report", "run", "summary"]
