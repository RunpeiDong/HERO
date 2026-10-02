"""Shared helpers of the benchmark tests: the example policy, an optional local hero_bench_v1 corpus (and the paper-protocol source set), dependency probes."""
from __future__ import annotations

import os
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
EXAMPLE_POLICY = ROOT / "checkpoints" / "example"
EXAMPLE_SIDECAR = EXAMPLE_POLICY / "model_hero.json"


def _tier_dir(root: Path) -> Path:
    """``<root>/core`` when ``root`` is a fetched / built corpus (clips + per-tier BENCH_MANIFEST.json under core/), else ``root`` itself
    (a flat directory of clips next to their BENCH_MANIFEST.json)."""
    core = root / "core"
    return core if (core / "BENCH_MANIFEST.json").is_file() else root


#: a local hero_bench_v1 corpus (``scripts/hero_bench.py fetch --out data/hero_bench_v1`` or a build); ``HERO_BENCH_V1_DIR`` overrides the
#: location.  The smoke / replan tests score the paper-protocol clips (``h*__*.npz``) of its core tier.
BENCH_DIR = _tier_dir(Path(os.environ.get("HERO_BENCH_V1_DIR", ROOT / "data" / "hero_bench_v1")).expanduser())
#: the paper's 180-target protocol set as a flat dir (clips + BENCH_MANIFEST.json), the ``--paper-dir`` input of the corpus builder
PAPER_DIR = Path(os.environ.get("HERO_BENCH_PAPER_DIR", ROOT / "data" / "hero_bench_paper")).expanduser()
#: optional reference results under ``open_loop/`` and ``closed_loop/`` for the regression tests
BENCH_RESULTS_DIR = Path(os.environ.get("HERO_BENCH_RESULTS_DIR", ROOT / "results" / "benchmark")).expanduser()


def has_mujoco() -> bool:
    try:
        import mujoco  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def has_mink() -> bool:
    try:
        import mink  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def has_onnxruntime() -> bool:
    try:
        import onnxruntime  # noqa: F401

        return True
    except Exception:  # noqa: BLE001
        return False


def bench_clips() -> list[Path]:
    return sorted(BENCH_DIR.glob("h*__*.npz")) if (BENCH_DIR / "BENCH_MANIFEST.json").is_file() else []


def example_policy_available() -> bool:
    return (EXAMPLE_POLICY / "model.onnx").is_file() and EXAMPLE_SIDECAR.is_file()


def _stack(*marks):
    """Apply several pytest marks (the registered ``mujoco`` / ``bench_data`` markers + a skip condition) with one decorator."""
    def deco(fn):
        for m in reversed(marks):
            fn = m(fn)
        return fn
    return deco


SKIP_MUJOCO = pytest.mark.skipif(not has_mujoco(), reason="needs mujoco")
SKIP_MINK = pytest.mark.skipif(not (has_mujoco() and has_mink()), reason="needs mujoco + mink")
SKIP_BENCH = pytest.mark.skipif(not bench_clips(), reason=f"needs a hero_bench_v1 corpus at {BENCH_DIR} (HERO_BENCH_V1_DIR)")
SKIP_POLICY = pytest.mark.skipif(not (has_onnxruntime() and example_policy_available()), reason="needs onnxruntime + checkpoints/example")
needs_mujoco = _stack(pytest.mark.mujoco, SKIP_MUJOCO)
needs_mink = _stack(pytest.mark.mujoco, SKIP_MINK)
needs_bench = _stack(pytest.mark.bench_data, SKIP_BENCH)
needs_policy = SKIP_POLICY
