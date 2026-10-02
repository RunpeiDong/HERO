#!/usr/bin/env python3
"""build_hero_bench.py — plan-driven builder of the ``hero_bench_v1`` reaching benchmark (core / extended / stress tiers).

The plan is ``reach_specs.BENCH_V1_PLAN`` (20 ordered strata with their seeds, quotas, acceptance levels and health rules).  Every
stratum is one of four kinds and the builder treats each the same way on every run:

* ``verbatim`` (``paper``) — the paper's 180-target protocol set is copied byte for byte from ``--paper-dir`` (names unchanged, sha256
  checked against the source manifest row AND the copied file); rows keep their source fields and gain ``stratum = height_label``,
  ``tier = core``, ``verbatim_from``, ``hold_end_frame``.
* ``generate`` (15 strata) — one generator bank per stratum under ``--bank-root/<stratum>/batch<k>/`` (``clips/``, ``labels/``,
  ``manifest.json``; seed ``bench_v1_batch_seed(stratum, k)``; batch k >= 1 uses the clip prefix ``<prefix>_b<k>`` so clip ids stay
  unique).  Selection runs per quota cell ``(height_label, hand[, family])`` over the union of the batches in (batch, goal_index)
  order with the stratum's ``admitted_levels`` (``build_reach_bench.ACCEPT_LEVELS``; hard flags always reject); cells never borrow
  from another cell, family or stratum.  Every generated stratum has ``fill_until_quota``: while a cell is short the next top-up
  batch is generated (``--generate``) or read, up to ``fill_max_batches``; a core cell that is still short exits with code 3, an
  extended / stress stratum ships what it has and reports the shortfall (exit 0).  The strict+core yield over the batches used is
  recorded; below the plan's ``min_yield`` (extended 0.60 / stress 0.40) the stratum is flagged ``provisional`` (never dropped; stress
  strata also carry the plan's ``fallback`` note in the report).
* ``retime`` (``slow_x2`` / ``hold6`` / ``fast_x0p75``) — the ``paper_sub60`` subset (per paper-protocol table height the 10 right + 10 left clips
  with the smallest goal_index) re-timed by ``data_tools.retime_bench`` with the stratum's factor / hold; outputs are named
  ``<stratum>__<paper clip_id>.npz`` (``source_tag`` = stratum), rows carry ``pair_of`` = the paper-protocol twin, frame fields from the warp,
  and ``fast_x0p75`` drops clips over the speed / acceleration gate (``shortfall_ok``).
* ``pool`` (``recov_pool``) — recoverable-level candidates (terminal error <= 3 cm / 10 deg, soft flags allowed) of the source
  strata that were NOT selected into their own stratum, 12 per source in (batch, goal_index) order -- the source bank's GLOBAL goal_index,
  which interleaves the table heights of the source (the pool is deliberately not balanced per height; a per-height quota would be a new
  plan version) -- converted as ``recov_pool__<clip_id>.npz``
  with ``pool_source``; the closed-loop target stays the nominal ``target_pos_w`` (nothing extra is recorded).

Output tree::

    <out>/core/  <out>/extended/  <out>/stress/     flat npz dirs (<stratum>__<clip_id>.npz; paper-protocol files keep their names, in core/)
    <out>/tiers/{core,extended,stress,all,paper,paper_sub60}.txt
    <out>/BENCH_MANIFEST.json  (+ <out>/<tier>/BENCH_MANIFEST.json with that tier's rows)   <out>/BENCH_REPORT.md
    <out>/SHA256SUMS  <out>/DATA_LICENSE  <out>/core/LICENSE_paper_verbatim.txt

``--public`` scrubs host names, absolute paths, S3 URIs, node names and internal git-sha suffixes from the manifests (also available as
``scrub_public_manifest`` / ``--scrub-only``); ``verify_shipped`` re-hashes a built corpus against its manifest and ``SHA256SUMS``.

    python scripts/hero_bench.py build --plan default --bank-root <root> --paper-dir <paper> --out <out> --generate --jobs 60 [--nice 10] [--public]
    (equivalently ``python -m data_tools.build_hero_bench ...`` with the release root and third_party/holosoma on PYTHONPATH)

Without ``--generate`` the builder consumes the banks already under ``--bank-root`` (what runs after a separate generation pass; a
batch whose ``manifest.json`` exists is never regenerated, so an interrupted ``--generate`` run resumes).
"""

from __future__ import annotations

import argparse
import copy
import dataclasses
import hashlib
import json
import math
import os
import platform
import re
import shutil
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from data_tools import build_reach_bench as bb
from data_tools import reach_specs as R
from data_tools import retime_bench as RB
from data_tools.schema import read_npz

SCHEMA: str = R.BENCH_SCHEMA_V1
PLAN_VERSION: str = R.BENCH_SCHEMA_V1
BUILDER: str = "data_tools.build_hero_bench"
BUILDER_VERSION: str = "1.0.0"
FPS: int = bb.FPS
LICENSE_CLASS: str = bb.LICENSE_CLASS
TIERS: tuple[str, ...] = tuple(R.BENCH_V1_TIERS)
LEVELS = bb.ACCEPT_LEVELS
LEVEL_INDEX: dict[str, int] = {lv.name: i for i, lv in enumerate(LEVELS)}
RECOVERABLE_LEVEL: int = LEVEL_INDEX["recoverable"]
EXIT_OK, EXIT_ERROR, EXIT_CORE_GAP = 0, 1, 3
DEFAULT_PAPER_DIR: Path = Path(os.environ.get("HERO_BENCH_PAPER_DIR", "data/hero_bench_paper"))   # the paper's 180-target protocol set (clips + BENCH_MANIFEST.json)
DEFAULT_NICE: int = int(os.environ.get("HERO_BENCH_NICE", "10"))
MANIFEST_NAME: str = "BENCH_MANIFEST.json"
REPORT_NAME: str = "BENCH_REPORT.md"
SUMS_NAME: str = "SHA256SUMS"
DATA_LICENSE_NAME: str = "DATA_LICENSE"
PAPER_LICENSE_NAME: str = "LICENSE_paper_verbatim.txt"
TIERS_DIR: str = "tiers"
WORK_DIR: str = ".work"
VERIFIERS: tuple[str, ...] = ("auto", "hero", "holosoma", "schema", "none")
#: directory that holds ``data_tools/`` -- the generator subprocess runs ``python -m data_tools.hero_reach_generator`` from here
PACKAGE_ROOT: Path = Path(__file__).resolve().parents[1]
PAPER_BUILD: str = "paper_protocol"   # verbatim_from.build of the paper's 180-target protocol set, included byte for byte
RELEASE_DIST: str = "hero-isaacsim"   # the hero_release distribution name (pyproject [project] name)


def release_version() -> str | None:
    """Version of the hero_release package this builder ships with: ``importlib.metadata`` of :data:`RELEASE_DIST` when installed, else the
    ``[project] version`` of the ``pyproject.toml`` next to the package (``PACKAGE_ROOT``); None when neither exists (the internal checkout has
    no pyproject).  Recorded in the manifest as ``release_version`` -- provenance only, not part of the protocol block."""
    try:
        from importlib.metadata import version

        return str(version(RELEASE_DIST))
    except Exception:  # noqa: BLE001 - not installed
        pass
    pyproject = PACKAGE_ROOT / "pyproject.toml"
    if not pyproject.is_file():
        return None
    text = pyproject.read_text()
    try:
        import tomllib

        return str(tomllib.loads(text)["project"]["version"])
    except Exception:  # noqa: BLE001 - python < 3.11 or an unexpected layout: a regex over the [project] table
        pass
    table = re.search(r"^\[project\](.*?)(?=^\[|\Z)", text, re.S | re.M)
    hit = re.search(r'^version\s*=\s*"([^"]+)"', table.group(1), re.M) if table else None
    return hit.group(1) if hit else None


class BuildError(RuntimeError):
    """Configuration / data error; ``main`` exits with code 1."""

    exit_code = EXIT_ERROR


class CoreGapError(BuildError):
    """A core stratum cannot fill its quota even with every top-up batch (core strata never ship a gap); exit code 3."""

    exit_code = EXIT_CORE_GAP


# --------------------------------------------------------------------------------------------------
# Plan
# --------------------------------------------------------------------------------------------------
def plan_from_json(obj: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
    """Inverse of ``reach_specs.bench_v1_plan_json``: quota keys ``"h050/right"`` -> ``("h050", "right")``."""
    plan: dict[str, dict[str, Any]] = {}
    for name, entry in obj.items():
        e = dict(entry)
        e["quota"] = {tuple(str(k).split("/")): int(v) for k, v in dict(e["quota"]).items()}
        e.pop("quota_total", None)
        e.setdefault("stratum", name)
        plan[name] = e
    return plan


def load_plan(spec: str | None) -> dict[str, dict[str, Any]]:
    """``default`` -> ``BENCH_V1_PLAN``; ``reduced:<per_cell>`` -> :func:`reduced_plan`; otherwise a JSON file in ``bench_v1_plan_json`` form."""
    if spec in (None, "", "default"):
        return copy.deepcopy(R.BENCH_V1_PLAN)
    if str(spec).startswith("reduced:"):
        return reduced_plan(copy.deepcopy(R.BENCH_V1_PLAN), int(str(spec).split(":", 1)[1]))
    path = Path(spec)
    if not path.is_file():
        raise BuildError(f"--plan {spec!r}: not 'default', 'reduced:<n>' or an existing JSON file")
    return plan_from_json(json.loads(path.read_text()))


def reduced_plan(plan: Mapping[str, dict[str, Any]], per_cell: int, *, oversample: int = 2) -> dict[str, dict[str, Any]]:
    """Smoke-test copy of a plan: every quota cell capped at ``per_cell``; generated strata request ``oversample`` candidates per
    quota slot (a multiple of the profile's heights, as the generator requires).  Not for shipping."""
    if per_cell < 1:
        raise ValueError("per_cell must be >= 1")
    out: dict[str, dict[str, Any]] = {}
    for name, entry in plan.items():
        e = copy.deepcopy(entry)
        e["quota"] = {k: min(int(v), per_cell) for k, v in e["quota"].items()}
        if e["kind"] == "generate":
            cfg = R.PROFILES[e["profile"]]
            n_heights = len(cfg["bench_heights"])
            per_height = sum(e["quota"].values()) // n_heights
            e["n_candidates"] = n_heights * max(1, per_height) * oversample
        e["reduced"] = {"per_cell": per_cell, "oversample": oversample}
        out[name] = e
    return out


def plan_table(plan: Mapping[str, dict[str, Any]]) -> str:
    """Human-readable plan table (``hero_bench.py plan``)."""
    lines = ["| # | stratum | tier | kind | profile | seed | candidates | quota | cells | admitted | fill (max batches) | min yield |",
             "|---|---|---|---|---|---|---|---|---|---|---|---|"]
    for e in plan.values():
        lines.append(f"| {e.get('ordinal', '-')} | {e['stratum']} | {e['tier']} | {e['kind']} | {e.get('profile') or '-'} | {e['seed']} | {e['n_candidates'] or '-'} | "
                     f"{sum(e['quota'].values())} | {len(e['quota'])} | {'+'.join(e['admitted_levels'])} | "
                     f"{'yes (' + str(e['fill_max_batches']) + ')' if e.get('fill_until_quota') else 'no'} | {e.get('min_yield') if e.get('min_yield') is not None else '-'} |")
    totals = Counter()
    for e in plan.values():
        totals[e["tier"]] += sum(e["quota"].values())
    lines.append("")
    lines.append("Planned clips per tier: " + ", ".join(f"{t} {totals.get(t, 0)}" for t in TIERS) + f" (total {sum(totals.values())}).")
    return "\n".join(lines)


def plan_json(plan: Mapping[str, dict[str, Any]]) -> dict[str, Any]:
    out: dict[str, Any] = {}
    for name, e in plan.items():
        out[name] = {**e, "quota": {"/".join(k): v for k, v in e["quota"].items()}, "quota_total": int(sum(e["quota"].values()))}
    return out


# --------------------------------------------------------------------------------------------------
# Quota cells, bank batches, generator
# --------------------------------------------------------------------------------------------------
def quota_has_family(quota: Mapping[tuple[str, ...], int]) -> bool:
    return any(len(k) == 3 for k in quota)


def cell_of(bench: Mapping[str, Any], with_family: bool) -> tuple[str, ...]:
    key: tuple[str, ...] = (str(bench["height_label"]), str(bench["hand"]))
    return key + ((str(bench.get("orient_family") or ""),) if with_family else ())


def cell_key(cell: Sequence[str]) -> str:
    return "/".join(cell)


def batch_dir(bank_root: Path, stratum: str, batch: int) -> Path:
    return Path(bank_root) / stratum / f"batch{int(batch)}"


def batch_clip_prefix(entry: Mapping[str, Any], batch: int) -> str:
    """Batch 0 keeps the plan prefix; top-up batches append ``_b<k>`` so clip ids never collide across batches."""
    return str(entry["clip_prefix"]) if int(batch) == 0 else f"{entry['clip_prefix']}_b{int(batch)}"


def batch_seed(entry: Mapping[str, Any], batch: int) -> int:
    """``reach_specs.bench_v1_batch_seed`` for plan strata; the same rule (``seed + 1000 k``) for custom plans."""
    if entry["stratum"] in R.BENCH_V1_PLAN and R.BENCH_V1_PLAN[entry["stratum"]]["seed"] == entry["seed"]:
        return R.bench_v1_batch_seed(entry["stratum"], batch)
    if batch < 0 or batch > int(entry.get("fill_max_batches", 0)):
        raise ValueError(f"{entry['stratum']}: batch {batch} outside 0..{entry.get('fill_max_batches', 0)}")
    return int(entry["seed"]) + R.BENCH_V1_FILL_SEED_STRIDE * int(batch)


def generator_command(entry: Mapping[str, Any], batch: int, out_dir: Path, *, jobs: int, nice: int = 0, python: str | None = None) -> list[str]:
    """``[nice -n N] python -m data_tools.hero_reach_generator --profile ... --n ... --seed ... --out-dir ... --jobs ... --clip-prefix ...``."""
    cmd = [python or sys.executable, "-m", "data_tools.hero_reach_generator", "--profile", str(entry["profile"]), "--n", str(int(entry["n_candidates"])),
           "--seed", str(batch_seed(entry, batch)), "--out-dir", str(out_dir), "--jobs", str(max(1, int(jobs))), "--clip-prefix", batch_clip_prefix(entry, batch)]
    if nice and nice > 0 and shutil.which("nice"):
        cmd = ["nice", "-n", str(int(nice))] + cmd
    return cmd


def run_generator(entry: Mapping[str, Any], batch: int, out_dir: Path, *, jobs: int = 1, nice: int = DEFAULT_NICE,
                  log: Callable[[str], None] = print) -> Path:
    """Generate one candidate batch into ``out_dir`` (resumable: an existing ``manifest.json`` is kept).  Returns ``out_dir``."""
    out_dir = Path(out_dir)
    if (out_dir / "manifest.json").is_file():
        log(f"[hero_bench] {entry['stratum']} batch{batch}: manifest.json exists, skipping generation")
        return out_dir
    out_dir.mkdir(parents=True, exist_ok=True)
    cmd = generator_command(entry, batch, out_dir, jobs=jobs, nice=nice)
    env = dict(os.environ)
    env["PYTHONPATH"] = os.pathsep.join([p for p in sys.path if p] + ([env["PYTHONPATH"]] if env.get("PYTHONPATH") else []))
    log_path = out_dir / "generator.log"
    log(f"[hero_bench] {entry['stratum']} batch{batch}: {' '.join(cmd)}  (log {log_path})")
    t0 = time.time()
    with open(log_path, "w") as fh:
        proc = subprocess.run(cmd, cwd=str(PACKAGE_ROOT), env=env, stdout=fh, stderr=subprocess.STDOUT, text=True)
    if proc.returncode != 0 or not (out_dir / "manifest.json").is_file():
        tail = log_path.read_text()[-2000:] if log_path.is_file() else ""
        raise BuildError(f"{entry['stratum']} batch{batch}: generator exit {proc.returncode}, manifest.json {'present' if (out_dir / 'manifest.json').is_file() else 'missing'}\n{tail}")
    log(f"[hero_bench] {entry['stratum']} batch{batch}: generated in {time.time() - t0:.0f} s")
    return out_dir


def load_batch(bank: Path, batch: int, bank_root: Path | None = None) -> dict[str, Any]:
    """One bank batch: ``{"batch", "dir", "rel", "manifest", "rows"}``; rows carry ``bench`` (from ``labels/`` when the manifest row lacks it),
    ``_batch``, ``_bank_dir``, ``_bank_rel``, ``_seed``, ``_uid`` and, for retracting clips, ``_retract`` (the labels' ``retract`` block)."""
    bank = Path(bank)
    try:
        manifest, rows = bb.load_bank(bank)
    except ValueError:                      # a batch without a single usable clip (all generator failures): keep the rows for the statistics
        manifest = json.loads((bank / "manifest.json").read_text())
        rows = [dict(r) for r in manifest.get("clips", [])]
    rel = str(bank.relative_to(bank_root)) if bank_root is not None and bank_root in bank.parents else bank.name
    for r in rows:
        r["_batch"] = int(batch)
        r["_bank_dir"] = str(bank)
        r["_bank_rel"] = rel
        r["_seed"] = manifest.get("seed")
        r["_uid"] = f"b{int(batch)}:{r.get('clip_id')}"
        if r.get("ok", True) and r.get("has_retract") and r.get("labels") and "_retract" not in r:
            lab = bank / r["labels"]
            if lab.is_file():
                r["_retract"] = json.loads(lab.read_text()).get("retract")
    return {"batch": int(batch), "dir": bank, "rel": rel, "manifest": manifest, "rows": rows}


def load_batches(bank_root: Path, stratum: str, max_batches: int) -> list[dict[str, Any]]:
    """Existing batches ``batch0 .. batch<k>`` (contiguous from 0) of one stratum under ``bank_root``."""
    out: list[dict[str, Any]] = []
    for k in range(int(max_batches) + 1):
        d = batch_dir(bank_root, stratum, k)
        if not (d / "manifest.json").is_file():
            break
        out.append(load_batch(d, k, Path(bank_root)))
    return out


def batch_summary(b: Mapping[str, Any], *, generated_now: bool = False) -> dict[str, Any]:
    m = b["manifest"]
    lv = Counter(LEVELS[l].name if l is not None else "never" for l in (bb.level_of(r) for r in b["rows"]))
    return {"batch": b["batch"], "dir": b["rel"], "seed": m.get("seed"), "clip_prefix": (m.get("clips") or [{}])[0].get("clip_id", "").rsplit("_", 1)[0] if m.get("clips") else None,
            "generator_version": m.get("generator_version"), "n_requested": m.get("n_requested"), "n_written": m.get("n_written"), "n_failed": m.get("n_failed"),
            "n_rows": len(b["rows"]), "n_by_level": {n: int(lv.get(n, 0)) for n in [*LEVEL_INDEX, "never"]}, "tier_counts": m.get("tier_counts"),
            "flag_counts": m.get("flag_counts"), "mjcf_sha256": m.get("mjcf_sha256"), "git_sha": m.get("git_sha"), "created_utc": m.get("created_utc"),
            "overrides": m.get("overrides"), "solver_config": m.get("solver_config"), "gate_thresholds": m.get("gate_thresholds"), "host": m.get("host"),
            "generated_now": bool(generated_now)}


# --------------------------------------------------------------------------------------------------
# Selection (pure python)
# --------------------------------------------------------------------------------------------------
def _order_key(row: Mapping[str, Any]) -> tuple[int, int]:
    return (int(row.get("_batch", 0)), int(row["bench"]["goal_index"]))


def select_stratum(rows: Sequence[dict[str, Any]], quota: Mapping[tuple[str, ...], int], admitted_levels: Sequence[str]
                   ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], list[dict[str, Any]]]:
    """Per quota cell ``(height_label, hand[, family])``: the first ``quota`` candidates in (batch, goal_index) order at the best admitted
    level, then the next admitted level for a still-short cell (``accept_level`` / ``accept_level_name`` recorded).  Cells never borrow
    from another cell; candidates outside the quota cells are reported (``unplanned_cells``) and never selected.
    Returns ``(selected, per-cell stats, unselected candidates)``; unselected rows carry ``_level`` (None = never accepted)."""
    with_family = quota_has_family(quota)
    levels = [LEVEL_INDEX[n] for n in admitted_levels]
    if not levels:
        raise ValueError("admitted_levels must name at least one acceptance level")
    by_cell: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        b = r.get("bench")
        if not b or not r.get("ok", True):
            continue
        by_cell[cell_of(b, with_family)].append(r)
    selected: list[dict[str, Any]] = []
    unselected: list[dict[str, Any]] = []
    stats: dict[str, dict[str, Any]] = {}
    for cell, q in quota.items():
        cand = sorted(by_cell.get(tuple(cell), []), key=_order_key)
        lv = [bb.level_of(r) for r in cand]
        chosen: list[dict[str, Any]] = []
        for L in levels:
            for r, l in zip(cand, lv):
                if l == L and len(chosen) < q:
                    chosen.append({**r, "accept_level": L, "accept_level_name": LEVELS[L].name})
            if len(chosen) >= q:
                break
        chosen_ids = {c["_uid"] for c in chosen}
        unselected += [{**r, "_level": l} for r, l in zip(cand, lv) if r["_uid"] not in chosen_ids]
        reasons: Counter[str] = Counter()
        for r, l in zip(cand, lv):
            if l != 0:
                for reason in bb.strict_reject_reasons(r, LEVELS[0]) or ["unknown"]:
                    reasons[reason] += 1
        stats[cell_key(cell)] = {
            "height_label": cell[0], "hand": cell[1], "orient_family": cell[2] if with_family else None, "quota": int(q),
            "n_candidates": len(cand), "n_by_level": {LEVELS[i].name: int(sum(1 for l in lv if l == i)) for i in range(len(LEVELS))},
            "n_never_accepted": int(sum(1 for l in lv if l is None)), "n_selected": len(chosen),
            "n_selected_by_level": {LEVELS[i].name: int(sum(1 for c in chosen if c["accept_level"] == i)) for i in range(len(LEVELS))},
            "shortfall": max(0, int(q) - len(chosen)), "max_level_used": LEVELS[max(c["accept_level"] for c in chosen)].name if chosen else None,
            "fallback_used": bool(chosen) and any(c["accept_level"] > 0 for c in chosen),
            "selected": [[int(c["_batch"]), int(c["bench"]["goal_index"])] for c in chosen], "strict_reject_reasons": dict(reasons.most_common()),
        }
        selected += chosen
    planned = {tuple(c) for c in quota}
    unplanned = {cell_key(c): len(v) for c, v in by_cell.items() if c not in planned}
    if unplanned:
        stats["unplanned_cells"] = {"note": "candidates outside the quota cells (never selected)", "cells": unplanned}
    return selected, stats, unselected


def stratum_yield(rows: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Strict / core / recoverable / never counts over ALL candidates (generator failures included) and the strict+core yield."""
    lv = [bb.level_of(r) if r.get("ok", True) else None for r in rows]
    n = len(rows)
    by = {LEVELS[i].name: int(sum(1 for l in lv if l == i)) for i in range(len(LEVELS))}
    by["never"] = int(sum(1 for l in lv if l is None))
    sc = by["strict"] + by["core"]
    return {"n_candidates": n, "n_ok": int(sum(1 for r in rows if r.get("ok", True))), "n_failed": int(sum(1 for r in rows if not r.get("ok", True))),
            "admitted": by, "yield": (sc / n) if n else 0.0, "yield_strict": (by["strict"] / n) if n else 0.0}


# --------------------------------------------------------------------------------------------------
# Conversion of generated candidates (the production ik_reach_v2 path through build_reach_bench's worker)
# --------------------------------------------------------------------------------------------------
def bench_output_name(stratum: str, row: Mapping[str, Any]) -> str:
    return f"{stratum}__{row['clip_id']}.npz"


def bench_conversion_meta(row: Mapping[str, Any], stratum: str) -> dict[str, Any]:
    """``build_reach_bench.conversion_meta`` with the benchmark stratum as ``source_tag`` (== the file-name prefix)."""
    return {**bb.conversion_meta(row), "source_tag": str(stratum), "bench_schema": SCHEMA}


def convert_rows(selected: Sequence[dict[str, Any]], out_dir: Path, stratum: str, jobs: int = 1) -> list[dict[str, Any]]:
    """Convert the selected bank clips into ``out_dir/<stratum>__<clip_id>.npz`` (``build_reach_bench._convert_worker``)."""
    jobs_list = [(str(Path(r["_bank_dir"]) / r["file"]), str(Path(out_dir) / bench_output_name(stratum, r)), bench_conversion_meta(r, stratum)) for r in selected]
    if not jobs_list:
        return []
    if jobs <= 1 or len(jobs_list) <= 1:
        return [bb._convert_worker(j) for j in jobs_list]
    import multiprocessing as mp

    with mp.get_context("spawn").Pool(min(jobs, len(jobs_list))) as pool:
        return list(pool.imap(bb._convert_worker, jobs_list))


_TABLE_FLOAT_KEYS: tuple[str, ...] = ("table_top_z", "table_edge_x", "table_edge_gap_m")


def bench_clip_entry(row: Mapping[str, Any], out_path: Path, receipt: Mapping[str, Any], *, stratum: str, tier: str) -> dict[str, Any]:
    """``build_reach_bench.clip_entry`` + the stratified-plan fields (``stratum`` / ``tier`` / ``generator_tier`` / ``time_scale`` / ``hold_end_frame`` /
    ``retract_start_frame`` / ``retract`` / ``waist_soft_ok`` / ...).  Floor rows (``table`` None) keep ``None`` in the table fields."""
    b = row["bench"]
    patched = {**row, "bench": {**b, **{k: math.nan for k in _TABLE_FLOAT_KEYS if b.get(k) is None}}}
    e = bb.clip_entry(patched, Path(out_path), dict(receipt))
    for k in _TABLE_FLOAT_KEYS:
        if b.get(k) is None:
            e[k] = None
    reach_end, hold = int(e["reach_end_frame"]), int(e["hold_frames"])
    e.update({
        "source_tag": str(stratum), "stratum": str(stratum), "plan_stratum": str(stratum), "tier": str(tier), "generator_tier": row.get("tier"),
        "time_scale": float(b.get("time_scale", 1.0)), "hold_s": float(b.get("hold_s", hold / FPS)),
        "hold_end_frame": int(b.get("hold_end_frame") if b.get("hold_end_frame") is not None else reach_end + hold),
        "retract_start_frame": (int(b["retract_start_frame"]) if b.get("retract_start_frame") is not None else None),
        "retract": row.get("_retract"), "has_retract": bool(row.get("has_retract", False) or b.get("retract", False)),
        "waist_soft_ok": b.get("waist_soft_ok"), "has_table": bool(b.get("has_table", True)), "y_side": b.get("y_side"), "cross_side": b.get("cross_side"),
        "max_qddot_rad_s2": row.get("max_qddot_rad_s2"), "bank_batch": int(row.get("_batch", 0)), "bank_seed": row.get("_seed"), "bank_batch_dir": row.get("_bank_rel"),
        "pair_of": None, "verbatim_from": None, "pool_source": None, "license_class": LICENSE_CLASS,
    })
    if e["has_retract"] and e["retract_start_frame"] is None and isinstance(e["retract"], dict) and e["retract"].get("start_frame") is not None:
        e["retract_start_frame"] = int(e["retract"]["start_frame"])
    return e


# --------------------------------------------------------------------------------------------------
# Verbatim import of the paper-protocol set
# --------------------------------------------------------------------------------------------------
def load_paper(paper_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]], str]:
    """``(manifest, rows, manifest sha256)`` of a paper-protocol source dir (the 180 files next to their BENCH_MANIFEST.json)."""
    paper_dir = Path(paper_dir)
    man_path = paper_dir / MANIFEST_NAME
    if not man_path.is_file():
        raise BuildError(f"--paper-dir {paper_dir}: {MANIFEST_NAME} not found")
    manifest = json.loads(man_path.read_text())
    rows = [dict(r) for r in manifest.get("clips", [])]
    if not rows:
        raise BuildError(f"{man_path}: no clip rows")
    return manifest, rows, bb._sha256(man_path)


def import_verbatim(entry: Mapping[str, Any], paper_dir: Path, out_dir: Path, *, log: Callable[[str], None] = print
                    ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Copy every paper-protocol file byte for byte into ``out_dir`` (names unchanged); sha256 is checked against the source manifest row and the copy.
    The quota cells must match the source manifest exactly (the paper set is the authority).  Returns ``(rows, per-cell stats, info)``."""
    paper_dir, out_dir = Path(paper_dir), Path(out_dir)
    manifest, paper_rows, man_sha = load_paper(paper_dir)
    build = str(entry.get("verbatim_from", {}).get("build") or PAPER_BUILD)
    quota = entry["quota"]
    with_family = quota_has_family(quota)
    counts: Counter[tuple[str, ...]] = Counter()
    for r in paper_rows:
        counts[cell_of(r, with_family)] += 1
    problems = [f"{cell_key(c)}: source has {counts.get(tuple(c), 0)} rows, quota {q}" for c, q in quota.items() if counts.get(tuple(c), 0) != q]
    extra = {cell_key(c): n for c, n in counts.items() if c not in {tuple(k) for k in quota}}
    if problems or extra:
        raise BuildError(f"{entry['stratum']}: the {build} manifest does not match the plan quota: {problems} extra cells {extra}")
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    stats: dict[str, dict[str, Any]] = {}
    t0 = time.time()
    for r in sorted(paper_rows, key=lambda r: (str(r["height_label"]), str(r["hand"]), int(r["goal_index"]))):
        src = paper_dir / r["file"]
        if not src.is_file():
            raise BuildError(f"{entry['stratum']}: {src} missing")
        if r.get("sha256") and bb._sha256(src) != r["sha256"]:
            raise BuildError(f"{entry['stratum']}: {src.name}: sha256 differs from the {build} manifest row ({r['sha256'][:16]}...) -- the paper set is the authority, refusing")
        dst = out_dir / r["file"]
        shutil.copyfile(src, dst)
        got = bb._sha256(dst)
        if r.get("sha256") and got != r["sha256"]:
            raise BuildError(f"{entry['stratum']}: {dst.name}: copied file sha256 {got[:16]}... != manifest {r['sha256'][:16]}...")
        reach_end, hold = int(r["reach_end_frame"]), int(r["hold_frames"])
        rows.append({**r, "sha256": got, "bytes": dst.stat().st_size, "stratum": str(r["height_label"]), "plan_stratum": str(entry["stratum"]), "tier": str(entry["tier"]),
                     "generator_tier": r.get("tier"), "source_tag": str(r.get("source_tag") or r["height_label"]), "time_scale": 1.0,
                     "hold_s": hold / FPS, "hold_end_frame": reach_end + hold, "retract_start_frame": None, "retract": None, "has_retract": False,
                     "waist_soft_ok": None, "has_table": True, "pair_of": None, "pool_source": None, "license_class": str(r.get("license_class") or LICENSE_CLASS),
                     "verbatim_from": {"build": build, "file": r["file"], "sha256": r.get("sha256"), "manifest_sha256": man_sha, "byte_identical": True}})
    for c, q in quota.items():
        cs = [x for x in rows if cell_of(x, with_family) == tuple(c)]
        stats[cell_key(c)] = {"height_label": c[0], "hand": c[1], "orient_family": None, "quota": int(q), "n_candidates": len(cs), "n_selected": len(cs),
                              "shortfall": 0, "n_by_level": dict(Counter(str(x.get("accept_level_name")) for x in cs)), "selected": sorted(int(x["goal_index"]) for x in cs)}
    info = {"build": build, "paper_dir": str(paper_dir), "manifest_sha256": man_sha, "schema": manifest.get("schema"), "n_files": len(rows),
            "git_sha": manifest.get("git_sha"), "created_utc": manifest.get("created_utc"), "host": manifest.get("host"), "bank_dir": manifest.get("bank_dir"),
            "bank": manifest.get("bank"), "copy_wall_s": time.time() - t0}
    log(f"[hero_bench] {entry['stratum']}: {len(rows)} {build} files copied byte-identical from {paper_dir} ({info['copy_wall_s']:.1f} s)")
    return rows, stats, info


# --------------------------------------------------------------------------------------------------
# Re-timed strata
# --------------------------------------------------------------------------------------------------
def paper_subset(paper_rows: Sequence[dict[str, Any]], quota: Mapping[tuple[str, ...], int]) -> list[dict[str, Any]]:
    """Per quota cell ``(height_label, hand)`` the ``quota`` paper-protocol rows with the smallest goal_index (the ``paper_sub60`` rule for the default plan)."""
    out: list[dict[str, Any]] = []
    for cell, q in quota.items():
        cs = sorted((r for r in paper_rows if cell_of(r, False) == tuple(cell)), key=lambda r: int(r["goal_index"]))
        out += cs[:int(q)]
    return out


def build_retime_stratum(entry: Mapping[str, Any], paper_dir: Path, paper_top: Mapping[str, Any], subset: Sequence[dict[str, Any]], out_dir: Path,
                         work_dir: Path, *, verify: bool = True, log: Callable[[str], None] = print
                         ) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]], dict[str, Any]]:
    """Stage the paper subset (files + manifest subset), run ``retime_bench`` with the stratum's factor / hold / gate, move the renamed outputs
    into ``out_dir``.  Rows: the retimed manifest rows + ``stratum`` / ``tier`` / ``pair_of`` / ``hold_end_frame`` / ``time_scale``."""
    stratum, tier = str(entry["stratum"]), str(entry["tier"])
    rt = entry["retime"]
    factor, hold_s = float(rt["factor"]), rt.get("hold_s")
    gate = RB.normalize_gate(rt.get("gate"))
    stage = Path(work_dir) / stratum / "stage"
    rout = Path(work_dir) / stratum / "retimed"
    for d in (stage, rout):
        if d.exists():
            shutil.rmtree(d)
        d.mkdir(parents=True)
    by_file: dict[str, dict[str, Any]] = {}
    for r in subset:
        shutil.copyfile(Path(paper_dir) / r["file"], stage / r["file"])
        by_file[r["file"]] = r
    top = {k: v for k, v in paper_top.items() if k != "clips"}
    top["clips"] = [dict(r) for r in subset]
    (stage / MANIFEST_NAME).write_text(json.dumps(top, indent=1, sort_keys=True, default=str))

    def rename(name: str) -> str:
        return f"{stratum}__{by_file[name]['clip_id']}.npz"

    receipt = RB.retime_bench(stage, rout, factor, hold_s=(None if hold_s is None else float(hold_s)), quiet=True, verify=verify, gate=gate,
                              rename=rename, source_tag=stratum)
    retimed = json.loads((rout / RB.MANIFEST_NAME).read_text())["clips"]
    rec_by_file = {rec["file"]: rec for rec in receipt["clips"]}
    out_dir.mkdir(parents=True, exist_ok=True)
    rows: list[dict[str, Any]] = []
    for e in retimed:
        rec = rec_by_file[e["file"]]
        src_row = by_file[rec["src_file"]]
        os.replace(rout / e["file"], Path(out_dir) / e["file"])
        reach_end, hold = int(e["reach_end_frame"]), int(e["hold_frames"])
        rows.append({**e, "stratum": stratum, "plan_stratum": stratum, "tier": tier, "generator_tier": src_row.get("tier"), "source_tag": stratum,
                     "clip_id": str(src_row["clip_id"]), "parent_id": str(src_row["clip_id"]), "pair_of": str(src_row["clip_id"]), "pair_file": src_row["file"],
                     "time_scale": float(src_row.get("time_scale", 1.0)) * factor, "hold_s": (hold / FPS if hold_s is None else float(hold_s)),
                     "hold_end_frame": reach_end + hold, "retract_start_frame": None, "retract": None, "has_retract": False, "waist_soft_ok": None,
                     "has_table": True, "verbatim_from": None, "pool_source": None, "license_class": str(src_row.get("license_class") or LICENSE_CLASS),
                     "accept_level": src_row.get("accept_level"), "accept_level_name": src_row.get("accept_level_name"),
                     "max_qdot_rad_s": rec["max_qdot_rad_s"], "max_qddot_rad_s2": rec["max_qddot_rad_s2"], "retime_gate": gate})
    dropped = [{**d, "pair_of": by_file[d["src_file"]]["clip_id"]} for d in receipt.get("gate_dropped", [])]
    quota = entry["quota"]
    stats: dict[str, dict[str, Any]] = {}
    for c, q in quota.items():
        cs = [x for x in rows if cell_of(x, False) == tuple(c)]
        ds = [d for d in dropped if cell_of(by_file[d["src_file"]], False) == tuple(c)]
        stats[cell_key(c)] = {"height_label": c[0], "hand": c[1], "orient_family": None, "quota": int(q), "n_candidates": len(cs) + len(ds), "n_selected": len(cs),
                              "shortfall": max(0, int(q) - len(cs)), "n_gate_dropped": len(ds), "selected": sorted(int(x["goal_index"]) for x in cs)}
    short = sum(s["shortfall"] for s in stats.values())
    if short and not rt.get("shortfall_ok", False):
        raise BuildError(f"{stratum}: {short} clip(s) short of the quota (gate dropped {len(dropped)}) and shortfall_ok is False")
    info = {"factor": factor, "hold_s": hold_s, "gate": gate, "source_subset": rt.get("source_subset"), "n_source": len(subset), "n_shipped": len(rows),
            "n_gate_dropped": len(dropped), "gate_dropped": dropped, "shortfall_ok": bool(rt.get("shortfall_ok", False)), "retime_tool_version": receipt.get("tool_version"),
            "root_ang_frame_counts": receipt.get("root_ang_frame_counts"), "index_sides": receipt.get("index_sides"), "verification": receipt.get("verification")}
    shutil.rmtree(stage, ignore_errors=True)
    shutil.rmtree(rout, ignore_errors=True)
    log(f"[hero_bench] {stratum}: {len(rows)} / {len(subset)} re-timed (x{factor:g}, hold {hold_s}){'' if not dropped else f', {len(dropped)} dropped by the gate'}")
    return rows, stats, info


# --------------------------------------------------------------------------------------------------
# Protocol block (deterministic: hashed into the manifest)
# --------------------------------------------------------------------------------------------------
def canonical_json(obj: Any) -> str:
    return json.dumps(obj, sort_keys=True, separators=(",", ":"), ensure_ascii=True, allow_nan=False)


def protocol_sha256(protocol: Mapping[str, Any]) -> str:
    return hashlib.sha256(canonical_json(protocol).encode("utf-8")).hexdigest()


def protocol_block() -> dict[str, Any]:
    """Evaluation protocol of hero_bench_v1.  Consumers read ``success`` / ``fail_free`` / ``replan`` /
    ``plant`` / ``odometry``; nothing here depends on the build host, so ``protocol_sha256`` identifies the protocol across builds."""
    return {
        "schema": SCHEMA, "plan_version": PLAN_VERSION, "paper": bb.PAPER,
        "success": {
            "open_loop": {"S7.5": {"pos_cm": 7.5, "rot_deg": 15}, "S5": {"pos_cm": 5, "rot_deg": 10}},
            "closed_loop": {"C3": {"pos_cm": 3, "rot_deg": 15, "window": "final"}},
            "cdf_points_cm": [2.5, 5, 10], "stayed_threshold_cm": 1.75, "headline": "ee_global_active_cm",
            "fail_free_required": True, "denominator": "shipped clips of the stratum (a shortfall shrinks n, never the denominator of a core stratum)",
            "ci": "Wilson 95 %; policies are compared with paired per-clip tests, not by overlapping intervals",
            "open_loop_window": "hold = [reach_end_frame, hold_end_frame) mean of the reaching hand's world-frame palm error",
            "closed_loop_window": "final = last 1.0 s before the padded end of the replanned rollout",
        },
        "fail_free": {"common": ["fall", "anchor_xy"], "anchor_xy_m": 0.5, "fall_rule": "robot pelvis z < 0.30 m while the reference pelvis > 0.50 m",
                      "fall_low_margin_m": 0.20, "fall_low_default": False, "native": "the policy's own training termination set (n/a for HERO-style exports)",
                      "failed_clips": "enter the per-stratum mean with their valid window; alive-only tables repeat every metric"},
        "replan": {"first": "reach_end", "period_s": 3.0, "base": "current", "goal_adjust": True, "blend_s": 0.3, "pad_s": 4, "horizon_rule": "max(14, clip_s + 6)",
                   "gain": 0.6, "max_step_cm": 1.0, "gate_cm": 15.0, "stay_cm": 1.75, "skip_cm": 2.0, "max_replans": 20,
                   "stop_frame": "hold_end_frame for rows with their own retract segment (retract_start_frame); no extra retract phase is added",
                   "goal_source": "manifest row: hand, target_pos_w, target_yaw/pitch/roll_deg, table (None for floor rows), reach_end_frame, pelvis_drop, pelvis_pitch_deg, waist_pitch_deg",
                   "recov_pool_target": "the nominal target_pos_w / orientation in the world frame (same goal as the open loop; the 1.5-3 cm IK residual is reported alongside)",
                   "not_available": "a stratum the harness cannot replan yet prints 'n/a (harness)'; a world-frame number never appears without its replan column"},
        "plant": {"urdf": "g1_29dof_dex3fixed_hero.urdf", "foot_collision": "sonic_box", "physics_profile": "hero", "physics_hz": 1000, "policy_hz": 50,
                  "self_collision": False, "reset": "frame 0, zero velocity", "no_table_geometry": True, "h_cmd": "auto"},
        "odometry": {"mode": "so", "seed_rule": "0 + crc32(filename) % 2^31", "truth_column": "diagnostic only",
                     "file_names_are_seeds": "renaming a clip changes its odometry noise realisation; the paper-protocol files therefore keep their names"},
        "windows": {"fps": FPS, "settle_frames": 15, "reach_start_frame": 15, "reach_end_frame": "first hold frame (global index)", "hold": "[reach_end_frame, hold_end_frame)",
                    "final_open_loop": "last 1.0 s of the hold window",
                    "retract_rest": "retract rows: open loop = the last 1.0 s (50 frames) of the clip, i.e. the generator's 25 rest frames preceded by the last 0.5 s of the "
                                    "retract motion; closed loop = the last 1.0 s of the padded rollout (the clip's final rest frame held), reported apart from final -- "
                                    "final / tail of a retract row end at retract_start_frame, where the replanner hands the controller back to the clip",
                    "retimed_rows": "frame indices mapped through the retime warp (reach_end ceil, settle / reach_start floor)"},
        "acceptance_levels": [lv.as_dict() for lv in LEVELS], "hard_reject_flags": list(bb.HARD_REJECT_FLAGS), "soft_flags": list(bb.SOFT_FLAGS),
        "tiers": {"core": "run on every checkpoint", "extended": "decision points", "stress": "release / paper"},
        "file_naming": "<stratum>__<clip_id>.npz under <tier>/ (group by the prefix before '__'); paper-protocol rows keep their original names",
        "orientation_convention": "palm R = Rz(yaw) Ry(-pitch) Rx(roll) (reach_specs.palm_rotation) in the heading frame; yaw 0 = fingers forward, palm normal toward the body midline; quaternions wxyz",
        "target_frame": "heading frame == clip world frame: stance centre at the origin, x forward, y left, z up (per clip: pelvis_xy_frame0)",
        "license_class": LICENSE_CLASS, "data_license": "Apache-2.0 (DATA_LICENSE)",
        "columns": ["global open-loop (hold)", "global closed-loop replan+adjust (hold3)", "global closed-loop final-1s", "local open-loop", "fail-free open / closed"],
        "ik_residual": "every result table prints the stratum's mean terminal IK residual (terminal_ee_pos_err_m) next to the policy error",
    }


# --------------------------------------------------------------------------------------------------
# Manifest / report / sums / licences
# --------------------------------------------------------------------------------------------------
def _ik_residual_mm(rows: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    vals = [float(r["terminal_ee_pos_err_m"]) * 1e3 for r in rows if r.get("terminal_ee_pos_err_m") is not None and np.isfinite(float(r["terminal_ee_pos_err_m"]))]
    if not vals:
        return {"n": 0}
    a = np.asarray(vals)
    return {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.median(a)), "max": float(a.max())}


@dataclasses.dataclass
class BuildContext:
    plan: dict[str, dict[str, Any]]
    bank_root: Path | None
    out_dir: Path
    paper_dir: Path | None
    generate: bool
    jobs: int
    nice: int
    verifier: str
    log: Callable[[str], None]
    rows: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    strata: dict[str, dict[str, Any]] = dataclasses.field(default_factory=dict)
    pool_candidates: dict[str, list[dict[str, Any]]] = dataclasses.field(default_factory=dict)
    paper: dict[str, Any] = dataclasses.field(default_factory=dict)
    paper_rows: list[dict[str, Any]] = dataclasses.field(default_factory=list)
    paper_top: dict[str, Any] = dataclasses.field(default_factory=dict)
    paper_sub60_files: list[str] = dataclasses.field(default_factory=list)
    generator_versions: dict[str, list[str]] = dataclasses.field(default_factory=dict)
    notes: list[str] = dataclasses.field(default_factory=list)
    code_sha256: str | None = None   # --code-sha256: sha256 of the code archive the build ran from (provenance; None when not given)

    def tier_dir(self, tier: str) -> Path:
        return self.out_dir / tier


def build_manifest(ctx: BuildContext, verification: Mapping[str, Any]) -> dict[str, Any]:
    rows = sorted(ctx.rows, key=lambda r: (TIERS.index(r["tier"]), int(ctx.plan[r["plan_stratum"]].get("ordinal", 0)), str(r.get("height_label")), str(r.get("hand")), int(r.get("goal_index", 0)), r["file"]))
    for r in rows:
        r["path"] = f"{r['tier']}/{r['file']}"
    protocol = protocol_block()
    per_tier = {t: [r for r in rows if r["tier"] == t] for t in TIERS}
    planned = Counter()
    for e in ctx.plan.values():
        planned[e["tier"]] += sum(e["quota"].values())
    gv = sorted({v for vs in ctx.generator_versions.values() for v in vs if v})
    return {
        "schema": SCHEMA, "plan_version": PLAN_VERSION, "builder": BUILDER, "builder_version": BUILDER_VERSION,
        "generator_version": (gv[0] if len(gv) == 1 else gv) if gv else None, "generator_versions_by_stratum": ctx.generator_versions,
        "retime_tool_version": RB.TOOL_VERSION,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()), "git_sha": bb._git_sha(), "host": platform.node(), "python": sys.version.split()[0],
        "release_version": release_version(), "code_sha256": ctx.code_sha256,
        "bank_root": (str(ctx.bank_root) if ctx.bank_root else None), "paper_dir": (str(ctx.paper_dir) if ctx.paper_dir else None), "paper_protocol": ctx.paper or None,
        "n_clips": len(rows), "tiers": {t: {"n_clips": len(per_tier[t]), "n_planned": int(planned.get(t, 0)), "dir": f"{t}/", "files_list": f"{TIERS_DIR}/{t}.txt",
                                           "strata": sorted({r["plan_stratum"] for r in per_tier[t]}, key=lambda s: int(ctx.plan[s].get("ordinal", 0))),
                                           "manifest": f"{t}/{MANIFEST_NAME}"} for t in TIERS},
        "tier_totals_planned": dict(planned), "tier_totals_shipped": {t: len(per_tier[t]) for t in TIERS},
        "strata": ctx.strata, "plan": plan_json(ctx.plan),
        "paper_sub60": {"rule": R.PAPER_SUB60_RULE, "files": list(ctx.paper_sub60_files), "files_list": f"{TIERS_DIR}/paper_sub60.txt"},
        "excluded_ranges": copy.deepcopy(R.BENCH_V1_EXCLUDED_RANGES),
        "protocol": protocol, "protocol_sha256": protocol_sha256(protocol),
        "row_fields": {"stratum": "benchmark stratum id (paper-protocol rows: h050 / h074 / h088; plan_stratum = paper)", "tier": "core | extended | stress",
                       "generator_tier": "the IK generator's own tier (core | recoverable) -- not the benchmark tier",
                       "accept_level": "0 strict / 1 core / 2 recoverable (protocol.acceptance_levels)", "time_scale": "reach time scale (high shelves 1.3 / 1.4; re-timed rows x factor)",
                       "hold_end_frame": "reach_end_frame + hold_frames (the hold window is [reach_end_frame, hold_end_frame))",
                       "retract_start_frame": "first frame of the retract-to-rest phase (retract rows; None otherwise)", "retract": "generator retract block (retract rows)",
                       "pair_of": "clip_id of the paper-protocol twin (re-timed rows)", "verbatim_from": "paper-protocol provenance (verbatim rows: build, file, sha256, manifest_sha256)", "pool_source": "source stratum (recov_pool rows)",
                       "table": "slab geometry (None for floor rows)", "path": "<tier>/<file>", "sha256": "of the shipped file"},
        "sha256sums": {"file": SUMS_NAME, "covers": "every shipped *.npz, tiers/*.txt, DATA_LICENSE and core/LICENSE_paper_verbatim.txt (manifests / report excluded)"},
        "verification": dict(verification), "notes": list(ctx.notes),
        "clips": rows,
    }


def tier_manifest(manifest: Mapping[str, Any], tier: str) -> dict[str, Any]:
    out = {k: v for k, v in manifest.items() if k != "clips"}
    out["tier"] = tier
    out["clips"] = [r for r in manifest["clips"] if r["tier"] == tier]
    out["n_clips"] = len(out["clips"])
    return out


def write_tier_lists(out_dir: Path, manifest: Mapping[str, Any]) -> dict[str, list[str]]:
    d = Path(out_dir) / TIERS_DIR
    d.mkdir(parents=True, exist_ok=True)
    lists: dict[str, list[str]] = {t: [r["path"] for r in manifest["clips"] if r["tier"] == t] for t in TIERS}
    lists["all"] = [r["path"] for r in manifest["clips"]]
    lists["paper"] = [r["path"] for r in manifest["clips"] if r.get("verbatim_from")]
    lists["paper_sub60"] = [f"core/{f}" for f in manifest.get("paper_sub60", {}).get("files", [])]
    for name, files in lists.items():
        (d / f"{name}.txt").write_text("".join(f + "\n" for f in files))
    return lists


def sums_targets(out_dir: Path) -> list[Path]:
    out_dir = Path(out_dir)
    targets: list[Path] = []
    for t in TIERS:
        targets += sorted((out_dir / t).glob("*.npz")) if (out_dir / t).is_dir() else []
    targets += sorted((out_dir / TIERS_DIR).glob("*.txt")) if (out_dir / TIERS_DIR).is_dir() else []
    for extra in (out_dir / DATA_LICENSE_NAME, out_dir / "core" / PAPER_LICENSE_NAME):
        if extra.is_file():
            targets.append(extra)
    return targets


def write_sha256sums(out_dir: Path) -> list[str]:
    out_dir = Path(out_dir)
    lines = [f"{bb._sha256(p)}  {p.relative_to(out_dir).as_posix()}" for p in sums_targets(out_dir)]
    (out_dir / SUMS_NAME).write_text("".join(l + "\n" for l in lines))
    return lines


DATA_LICENSE_TEXT = """hero_bench_v1 -- benchmark reference motions (DATA LICENSE)

The motion clips (*.npz) of this corpus, the per-clip manifest rows and the derived lists are released under the
Apache License, Version 2.0 (http://www.apache.org/licenses/LICENSE-2.0).  Every clip carries license_class = "apache".

They are synthetic whole-body reaching references for the Unitree G1 + Dex3 model, produced by the HERO IK reach generator
(data_tools.hero_reach_generator) and its deterministic re-timing tool (data_tools.retime_bench); no human motion-capture
data is included.  The paper's 180-target protocol set, reproduced byte for byte under core/, is covered by core/LICENSE_paper_verbatim.txt.

Unless required by applicable law or agreed to in writing, the data is distributed on an "AS IS" BASIS, WITHOUT WARRANTIES
OR CONDITIONS OF ANY KIND, either express or implied.  The accompanying source code is licensed separately (see LICENSE).
"""

PAPER_LICENSE_TEXT = """Paper-protocol files included verbatim in hero_bench_v1/core

The 180 files named h050__hero_bench_v1_*.npz / h074__hero_bench_v1_*.npz / h088__hero_bench_v1_*.npz in this directory are the
paper's 180-target reaching protocol (three table heights x 60 targets), included byte for byte (file names unchanged; sha256 per
file in BENCH_MANIFEST.json, rows with verbatim_from).  They are licensed under the same terms as the rest of the corpus: Apache
License, Version 2.0 (see ../DATA_LICENSE); every clip carries license_class = "apache".  Keeping the bytes and names identical
preserves every number published for this protocol (the odometry noise seed of the evaluator is derived from the file name).
"""


def write_licenses(out_dir: Path) -> None:
    out_dir = Path(out_dir)
    (out_dir / DATA_LICENSE_NAME).write_text(DATA_LICENSE_TEXT)
    (out_dir / "core").mkdir(parents=True, exist_ok=True)
    (out_dir / "core" / PAPER_LICENSE_NAME).write_text(PAPER_LICENSE_TEXT)


def _fmt(v: Any, fmt: str = "{:.2f}") -> str:
    try:
        return fmt.format(float(v)) if v is not None and np.isfinite(float(v)) else "-"
    except (TypeError, ValueError):
        return "-"


def write_report(manifest: Mapping[str, Any], path: Path) -> None:
    clips = manifest["clips"]
    strata = manifest["strata"]
    plan = manifest["plan"]
    L: list[str] = []
    L.append(f"# BENCH_REPORT — `{manifest['schema']}` (plan `{manifest['plan_version']}`)")
    L.append("")
    L.append(f"Built {manifest['created_utc']} by `{manifest['builder']}` {manifest['builder_version']} (git `{str(manifest.get('git_sha'))[:12]}`, "
             f"release version `{manifest.get('release_version') or 'n/a'}`, code sha256 `{manifest.get('code_sha256') or 'n/a'}`"
             + (f", host `{manifest['host']}`" if manifest.get("host") else "") + f"); generator {manifest.get('generator_version')}, retime tool {manifest.get('retime_tool_version')}; "
             f"protocol sha256 `{manifest['protocol_sha256'][:16]}…`.")
    L.append("")
    tt = manifest["tier_totals_shipped"]
    tp = manifest["tier_totals_planned"]
    L.append("**Shipped**: " + ", ".join(f"{t} {tt.get(t, 0)} / {tp.get(t, 0)} planned" for t in TIERS) + f"; total {manifest['n_clips']} clips.")
    L.append("")
    # stratum table
    L.append("## Strata")
    L.append("")
    L.append("| stratum | tier | kind | quota | shipped | yield strict+core | candidates (strict / core / recoverable / never) | IK residual mm mean / p50 / max | batches | health | shortfall |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for name, s in strata.items():
        ad = s.get("admitted") or {}
        ik = s.get("ik_residual_mm") or {}
        L.append(f"| {name} | {s['tier']} | {s['kind']} | {s.get('quota_total', '-')} | "
                 f"{s.get('shipped', 0)} | {_fmt(s.get('yield'), '{:.2f}') if s.get('yield') is not None else '-'} | "
                 f"{s.get('n_candidates', '-')} ({ad.get('strict', '-')} / {ad.get('core', '-')} / {ad.get('recoverable', '-')} / {ad.get('never', '-')}) | "
                 f"{_fmt(ik.get('mean'))} / {_fmt(ik.get('p50'))} / {_fmt(ik.get('max'))} | {s.get('fill_batches_used', '-')} | {s.get('health', '-')}"
                 f"{' PROVISIONAL' if s.get('provisional') else ''} | {s.get('shortfall_total', 0)} |")
    L.append("")
    # stratum x height x hand
    labels = sorted({str(c.get("height_label")) for c in clips}, key=lambda s: (s != "floor", s))
    L.append("## Stratum × height × hand (shipped clips, right / left)")
    L.append("")
    L.append("| stratum | tier | " + " | ".join(labels) + " | total |")
    L.append("|---|---|" + "---|" * (len(labels) + 1))
    order = sorted({(c["stratum"], c["plan_stratum"]) for c in clips}, key=lambda k: (int(plan[k[1]].get("ordinal", 0)), k[0]))
    for stratum, plan_stratum in order:
        cs = [c for c in clips if c["stratum"] == stratum and c["plan_stratum"] == plan_stratum]
        cells = []
        for lab in labels:
            cl = [c for c in cs if str(c.get("height_label")) == lab]
            cells.append(f"{sum(1 for c in cl if c['hand'] == 'right')} / {sum(1 for c in cl if c['hand'] == 'left')}" if cl else "")
        L.append(f"| {stratum}{'' if stratum == plan_stratum else ' (' + plan_stratum + ')'} | {cs[0]['tier']} | " + " | ".join(cells) + f" | {len(cs)} |")
    L.append("")
    # per-stratum cell details
    L.append("## Per-stratum yields and cells")
    L.append("")
    for name, s in strata.items():
        L.append(f"### {name} ({s['tier']}, {s['kind']})")
        L.append("")
        if s["kind"] == "generate":
            L.append(f"- profile `{s.get('profile')}`, seed {s.get('seed')}, admitted levels {s.get('admitted_levels')}, fill {s.get('fill_until_quota')} (max {s.get('fill_max_batches')} top-ups), "
                     f"min yield {s.get('min_yield')}; batches used {s.get('fill_batches_used')}: "
                     + "; ".join(f"batch{b['batch']} seed {b['seed']} {b['n_written']}/{b['n_requested']} written, levels {b['n_by_level']}" for b in s.get("batches", [])))
            L.append(f"- yield strict+core {_fmt(s.get('yield'), '{:.3f}')} (strict {_fmt(s.get('yield_strict'), '{:.3f}')}); health `{s.get('health')}`"
                     + (f"; plan fallback to consider: {s.get('fallback')}" if s.get("provisional") and s.get("fallback") else ""))
        elif s["kind"] == "retime":
            info = s.get("retime") or {}
            L.append(f"- factor {info.get('factor')}, hold_s {info.get('hold_s')}, gate {info.get('gate')}, source {info.get('source_subset')} ({info.get('n_source')} clips), "
                     f"shipped {info.get('n_shipped')}, dropped by the gate {info.get('n_gate_dropped')}"
                     + (": " + ", ".join(f"{d['file']} ({', '.join(f'{k} {d[k]:.1f}' for k in d['exceeded'])})" for d in info.get("gate_dropped", [])) if info.get("gate_dropped") else ""))
        elif s["kind"] == "pool":
            L.append(f"- per source: " + ", ".join(f"{k} {v.get('n_selected', 0)}/{v.get('quota')} (recoverable unselected {v.get('n_candidates', 0)})" for k, v in (s.get("cells") or {}).items() if isinstance(v, dict) and "quota" in v))
        elif s["kind"] == "verbatim":
            v = s.get("verbatim") or {}
            L.append(f"- {v.get('n_files')} files of the `{v.get('build')}` set (manifest sha256 `{str(v.get('manifest_sha256'))[:16]}…`), byte-identical, names unchanged")
        cells = {k: v for k, v in (s.get("cells") or {}).items() if isinstance(v, dict) and "quota" in v}
        if cells and s["kind"] != "verbatim":
            L.append("")
            L.append("| cell | quota | candidates | strict | core | recoverable | never | selected | shortfall |")
            L.append("|---|---|---|---|---|---|---|---|---|")
            for k, c in cells.items():
                nb = c.get("n_by_level") or {}
                L.append(f"| {k} | {c['quota']} | {c.get('n_candidates', '-')} | {nb.get('strict', '-')} | {nb.get('core', '-')} | {nb.get('recoverable', '-')} | {c.get('n_never_accepted', '-')} | {c.get('n_selected', 0)} | {c.get('shortfall', 0)} |")
        if s.get("notes"):
            for n in s["notes"]:
                L.append(f"- note: {n}")
        L.append("")
    # excluded ranges
    L.append("## Excluded ranges (measured feasibility)")
    L.append("")
    L.append("| range | measured | disposition |")
    L.append("|---|---|---|")
    for r in manifest["excluded_ranges"]:
        L.append(f"| {r['range']} | {r['measured']} | {r['disposition']} |")
    L.append("")
    flagged = [(n, s) for n, s in strata.items() if s.get("provisional") or s.get("shortfall_total")]
    L.append("## Provisional / shortfall notes")
    L.append("")
    if not flagged:
        L.append("- none: every stratum shipped its full quota at the planned acceptance levels.")
    for n, s in flagged:
        L.append(f"- `{n}`: {'PROVISIONAL ' if s.get('provisional') else ''}health `{s.get('health')}`, yield {_fmt(s.get('yield'), '{:.3f}')}, batches {s.get('fill_batches_used', '-')}, "
                 f"shortfall {s.get('shortfall') or {}}" + (f"; plan fallback to consider: {s.get('fallback')}" if s.get("provisional") and s.get("fallback") else ""))
    L.append("")
    L.append("## Protocol")
    L.append("")
    su = manifest["protocol"]["success"]
    L.append(f"- success: open-loop " + ", ".join(f"{k} <= {v['pos_cm']} cm & {v['rot_deg']} deg" for k, v in su["open_loop"].items())
             + "; closed-loop " + ", ".join(f"{k} <= {v['pos_cm']} cm & {v['rot_deg']} deg ({v['window']})" for k, v in su["closed_loop"].items())
             + f"; stayed {su['stayed_threshold_cm']} cm; CDF points {su['cdf_points_cm']} cm; headline `{su['headline']}`.")
    ff = manifest["protocol"]["fail_free"]
    L.append(f"- fail-free: common {ff['common']} (fall_low margin {ff['fall_low_margin_m']} m, default {'on' if ff['fall_low_default'] else 'off'}).")
    rp = manifest["protocol"]["replan"]
    L.append(f"- replan: first {rp['first']}, period {rp['period_s']} s, base {rp['base']}, goal adjust {rp['goal_adjust']}, blend {rp['blend_s']} s, pad {rp['pad_s']} s, horizon {rp['horizon_rule']}.")
    pl = manifest["protocol"]["plant"]
    L.append(f"- plant: {pl['urdf']}, foot collision {pl['foot_collision']}, physics profile {pl['physics_profile']}, no table geometry; odometry {manifest['protocol']['odometry']['mode']} "
             f"(seed {manifest['protocol']['odometry']['seed_rule']}).")
    L.append(f"- protocol sha256 `{manifest['protocol_sha256']}`.")
    L.append("")
    v = manifest.get("verification") or {}
    L.append("## Verification")
    L.append("")
    for t in TIERS:
        vt = v.get(t) or {}
        L.append(f"- `{t}`: verifier `{vt.get('verifier')}`: {vt.get('n_loaded')} / {vt.get('n_files')} files loaded, {vt.get('n_skipped')} skipped" + (f"; notes {vt['notes']}" if vt.get("notes") else ""))
    L.append("")
    Path(path).write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------------------------------------
# Public scrub
# --------------------------------------------------------------------------------------------------
#: keys removed from a public manifest (host / machine specific)
PUBLIC_DROP_KEYS: frozenset[str] = frozenset({"host", "bank_root", "paper_dir", "out_dir", "bank_dir", "src_dir", "work_dir", "traceback", "mjcf", "bank_manifest_sha256_path"})
_ABS_PATH = re.compile(r"^(/|[A-Za-z]:[\\/])")
_NODE_NAME = re.compile(r"(^|[^A-Za-z0-9])ws-[A-Za-z0-9]")
_GIT_SHA_KEY = re.compile(r"git_sha$")


def scrub_public_value(key: str, value: Any) -> Any:
    """Recursive scrub of one manifest value: drop ``PUBLIC_DROP_KEYS``, absolute paths -> basename, ``s3://`` / node names -> ``<redacted>``,
    ``*git_sha`` -> the bare hex (no ``+suffix``)."""
    if isinstance(value, dict):
        return {k: scrub_public_value(str(k), v) for k, v in value.items() if str(k) not in PUBLIC_DROP_KEYS}
    if isinstance(value, list):
        return [scrub_public_value(key, v) for v in value]
    if isinstance(value, str):
        if "s3://" in value or _NODE_NAME.search(value):
            return "<redacted>"
        if _GIT_SHA_KEY.search(key):
            return value.split("+", 1)[0]
        if _ABS_PATH.match(value) or "/root/" in value or "/Users/" in value or "/home/" in value:
            return Path(value).name
        return value
    return value


def scrub_public_manifest(manifest: Mapping[str, Any]) -> dict[str, Any]:
    out = scrub_public_value("", dict(manifest))
    out["public"] = True
    out["scrubbed"] = sorted(PUBLIC_DROP_KEYS)
    return out


def scrub_public_dir(out_dir: Path) -> dict[str, Any]:
    """Scrub the root and per-tier manifests of a built corpus in place and regenerate the report from the scrubbed root manifest."""
    out_dir = Path(out_dir)
    root = json.loads((out_dir / MANIFEST_NAME).read_text())
    pub = scrub_public_manifest(root)
    (out_dir / MANIFEST_NAME).write_text(json.dumps(pub, indent=1, sort_keys=True))
    for t in TIERS:
        tp = out_dir / t / MANIFEST_NAME
        if tp.is_file():
            tp.write_text(json.dumps(tier_manifest(pub, t), indent=1, sort_keys=True))
    write_report(pub, out_dir / REPORT_NAME)
    return pub


# --------------------------------------------------------------------------------------------------
# Verification of a shipped corpus (hero_bench.py verify)
# --------------------------------------------------------------------------------------------------
def verify_shipped(out_dir: Path) -> dict[str, Any]:
    """Re-hash every manifest row's file and every ``SHA256SUMS`` line; ``ok`` iff everything is byte-identical and nothing is missing / extra."""
    out_dir = Path(out_dir)
    man_path = out_dir / MANIFEST_NAME
    if not man_path.is_file():
        raise FileNotFoundError(f"{man_path} not found")
    manifest = json.loads(man_path.read_text())
    result: dict[str, Any] = {"dir": str(out_dir), "schema": manifest.get("schema"), "protocol_sha256": manifest.get("protocol_sha256"),
                              "protocol_sha256_recomputed": protocol_sha256(manifest["protocol"]) if manifest.get("protocol") else None,
                              "n_rows": len(manifest.get("clips", [])), "n_identical": 0, "mismatched": [], "missing": [], "extra_npz": [],
                              "sums_checked": 0, "sums_mismatched": [], "sums_missing": [], "tier_manifests": {}}
    listed: set[str] = set()
    for r in manifest.get("clips", []):
        rel = r.get("path") or f"{r['tier']}/{r['file']}"
        listed.add(rel)
        p = out_dir / rel
        if not p.is_file():
            result["missing"].append(rel)
        elif bb._sha256(p) != r.get("sha256"):
            result["mismatched"].append(rel)
        else:
            result["n_identical"] += 1
    on_disk = {p.relative_to(out_dir).as_posix() for t in TIERS if (out_dir / t).is_dir() for p in (out_dir / t).glob("*.npz")}
    result["extra_npz"] = sorted(on_disk - listed)
    sums = out_dir / SUMS_NAME
    sums_listed: set[str] = set()
    if sums.is_file():
        for line in sums.read_text().splitlines():
            if not line.strip():
                continue
            digest, _, rel = line.partition("  ")
            sums_listed.add(rel)
            p = out_dir / rel
            result["sums_checked"] += 1
            if not p.is_file():
                result["sums_missing"].append(rel)
            elif bb._sha256(p) != digest:
                result["sums_mismatched"].append(rel)
    else:
        result["sums_missing"].append(SUMS_NAME)
    required_sums = listed | {p.relative_to(out_dir).as_posix() for p in sums_targets(out_dir)}
    result["sums_missing"] += sorted(required_sums - sums_listed - set(result["sums_missing"]))
    for t in TIERS:
        tp = out_dir / t / MANIFEST_NAME
        if not tp.is_file():
            result["missing"].append(f"{t}/{MANIFEST_NAME}")
            result["tier_manifests"][t] = {"n_rows": 0, "consistent_with_root": False, "protocol_sha256_matches": False, "error": "missing tier manifest"}
            continue
        try:
            tm = json.loads(tp.read_text())
            expected = tier_manifest(manifest, t)
            # Serialization order may differ; every scoring field of each row must still equal the root-derived row.
            rows = sorted(tm.get("clips", []), key=lambda r: r.get("path") or f"{r['tier']}/{r['file']}")
            root_rows = sorted(expected["clips"], key=lambda r: r.get("path") or f"{r['tier']}/{r['file']}")
            result["tier_manifests"][t] = {
                "n_rows": len(rows), "consistent_with_root": rows == root_rows and tm.get("tier") == t and tm.get("n_clips") == len(root_rows),
                "protocol_sha256_matches": (tm.get("protocol") == manifest.get("protocol") and tm.get("protocol_sha256") == manifest.get("protocol_sha256")
                                             and protocol_sha256(tm.get("protocol") or {}) == manifest.get("protocol_sha256")),
            }
        except (ValueError, TypeError, KeyError, AttributeError) as exc:
            result["tier_manifests"][t] = {"n_rows": 0, "consistent_with_root": False, "protocol_sha256_matches": False, "error": str(exc)}
    result["ok"] = (not result["mismatched"] and not result["missing"] and not result["extra_npz"] and not result["sums_mismatched"] and not result["sums_missing"]
                    and result["protocol_sha256"] == result["protocol_sha256_recomputed"] and all(v["consistent_with_root"] and v["protocol_sha256_matches"] for v in result["tier_manifests"].values()))
    return result


# --------------------------------------------------------------------------------------------------
# Per-kind builders
# --------------------------------------------------------------------------------------------------
def _stratum_base(entry: Mapping[str, Any]) -> dict[str, Any]:
    return {"stratum": entry["stratum"], "tier": entry["tier"], "kind": entry["kind"], "ordinal": entry.get("ordinal"), "profile": entry.get("profile"), "seed": entry.get("seed"),
            "clip_prefix": entry.get("clip_prefix"), "quota": {cell_key(k): v for k, v in entry["quota"].items()}, "quota_total": int(sum(entry["quota"].values())),
            "admitted_levels": list(entry["admitted_levels"]), "fill_until_quota": bool(entry.get("fill_until_quota")),
            "fill_max_batches": int(entry.get("fill_max_batches", 0)), "min_yield": entry.get("min_yield"), "fallback": entry.get("fallback"),
            "provisional": False, "health": "ok", "shipped": 0, "shortfall": {}, "shortfall_total": 0, "notes": []}


def _finish_stats(st: dict[str, Any], rows: Sequence[Mapping[str, Any]], cells: Mapping[str, Any]) -> None:
    st["shipped"] = len(rows)
    st["shipped_by_level"] = dict(Counter(str(r.get("accept_level_name")) for r in rows))
    st["cells"] = dict(cells)
    st["shortfall"] = {k: int(c["shortfall"]) for k, c in cells.items() if isinstance(c, dict) and c.get("shortfall")}
    st["shortfall_total"] = int(sum(st["shortfall"].values()))
    st["ik_residual_mm"] = _ik_residual_mm(rows)
    st["files"] = [r["file"] for r in rows]
    if st["shortfall_total"]:
        tag = "short_gate_dropped" if (st.get("retime") or {}).get("n_gate_dropped") else "short"
        st["health"] = f"provisional_{tag}" if st.get("provisional") else tag
    else:
        st["health"] = "provisional" if st.get("provisional") else "ok"


def build_generate_stratum(entry: Mapping[str, Any], ctx: BuildContext) -> None:
    stratum, tier = str(entry["stratum"]), str(entry["tier"])
    if ctx.bank_root is None:
        raise BuildError(f"{stratum}: --bank-root is required for generated strata")
    st = _stratum_base(entry)
    quota = dict(entry["quota"])
    fill, max_b = bool(entry.get("fill_until_quota")), int(entry.get("fill_max_batches", 0))
    batches = load_batches(ctx.bank_root, stratum, max_b)
    generated: set[int] = set()
    if not batches:
        d0 = batch_dir(ctx.bank_root, stratum, 0)
        if not ctx.generate:
            msg = f"{stratum}: no candidate bank at {d0} (run with --generate, or generate it first)"
            raise CoreGapError(msg) if tier == "core" else BuildError(msg)
        run_generator(entry, 0, d0, jobs=ctx.jobs, nice=ctx.nice, log=ctx.log)
        batches.append(load_batch(d0, 0, ctx.bank_root))
        generated.add(0)
    k = 0
    while True:
        rows = [r for b in batches[: k + 1] for r in b["rows"]]
        selected, cells, unselected = select_stratum(rows, quota, entry["admitted_levels"])
        short = {c: s["shortfall"] for c, s in cells.items() if isinstance(s, dict) and s.get("shortfall")}
        if not short or not fill or k >= max_b:
            break
        if k + 1 < len(batches):
            k += 1
            continue
        if not ctx.generate:
            st["notes"].append(f"top-up batch{k + 1} missing on disk and --generate not set")
            break
        d = batch_dir(ctx.bank_root, stratum, k + 1)
        run_generator(entry, k + 1, d, jobs=ctx.jobs, nice=ctx.nice, log=ctx.log)
        batches.append(load_batch(d, k + 1, ctx.bank_root))
        generated.add(k + 1)
        k += 1
    used = batches[: k + 1]
    rows = [r for b in used for r in b["rows"]]
    st["fill_batches_used"] = len(used)
    st["batches"] = [batch_summary(b, generated_now=b["batch"] in generated) for b in used]
    if len(batches) > len(used):
        st["notes"].append(f"{len(batches) - len(used)} further batch(es) on disk not needed (quota filled)")
    st.update(stratum_yield(rows))
    ctx.generator_versions[stratum] = sorted({str(b["manifest"].get("generator_version")) for b in used})
    if short:
        detail = ", ".join(f"{c} short {n}" for c, n in short.items())
        if tier == "core":
            raise CoreGapError(f"{stratum} ({tier}): quota cells still short after {len(used)} batch(es) (plan allows {max_b + 1}): {detail} -- core strata never ship a gap; "
                               + ("raise fill_max_batches / n_candidates or widen the profile" if len(used) > max_b else "run with --generate to produce the next top-up batch"))
        st["notes"].append(f"{sum(short.values())} slot(s) short after {len(used)} batch(es) (plan allows {max_b + 1}): {detail}; shipped as is, shortfall reported")
    # health flag: the strict+core yield over the batches used, never a reason to drop or shrink a stratum
    min_yield = entry.get("min_yield")
    if min_yield is not None and float(st["yield"]) < float(min_yield):
        st["provisional"] = True
        st["notes"].append(f"strict+core yield {st['yield']:.3f} < {min_yield}: flagged provisional" + (f"; plan fallback: {entry['fallback']}" if entry.get("fallback") else ""))
    ctx.pool_candidates[stratum] = unselected
    out_dir = ctx.tier_dir(tier)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = convert_rows(selected, out_dir, stratum, jobs=ctx.jobs) if selected else []
    errors = [r for r in results if not r["ok"]]
    if errors:
        for e in errors:
            ctx.log(f"[hero_bench] CONVERT FAIL {stratum} {Path(e['src']).name}: {e.get('error')}")
        raise BuildError(f"{stratum}: {len(errors)} / {len(results)} conversions failed")
    entries = [bench_clip_entry(r, Path(res["dst"]), res["receipt"], stratum=stratum, tier=tier) for r, res in zip(selected, results)]
    _finish_stats(st, entries, cells)
    ctx.rows += entries
    ctx.strata[stratum] = st
    ctx.log(f"[hero_bench] {stratum} ({tier}): {len(entries)} / {st['quota_total']} shipped, yield {st['yield']:.3f}, batches {st['fill_batches_used']}, health {st['health']}")


def build_verbatim_stratum(entry: Mapping[str, Any], ctx: BuildContext) -> None:
    if ctx.paper_dir is None:
        raise BuildError(f"{entry['stratum']}: --paper-dir is required (the paper-protocol set: 180 files next to their {MANIFEST_NAME})")
    st = _stratum_base(entry)
    rows, cells, info = import_verbatim(entry, ctx.paper_dir, ctx.tier_dir(entry["tier"]), log=ctx.log)
    st["verbatim"] = info
    ctx.paper = info
    _finish_stats(st, rows, cells)
    ctx.rows += rows
    ctx.strata[entry["stratum"]] = st


def build_retime_stratum_ctx(entry: Mapping[str, Any], ctx: BuildContext) -> None:
    if ctx.paper_dir is None:
        raise BuildError(f"{entry['stratum']}: --paper-dir is required for re-timed strata")
    if not ctx.paper_rows:
        ctx.paper_top, ctx.paper_rows, _ = load_paper(ctx.paper_dir)
    st = _stratum_base(entry)
    subset = paper_subset(ctx.paper_rows, entry["quota"])
    if not ctx.paper_sub60_files:
        ctx.paper_sub60_files = [r["file"] for r in subset]
    rows, cells, info = build_retime_stratum(entry, ctx.paper_dir, ctx.paper_top, subset, ctx.tier_dir(entry["tier"]), ctx.out_dir / WORK_DIR,
                                              verify=ctx.verifier != "none", log=ctx.log)
    st["retime"] = info
    if info["n_gate_dropped"]:
        st["notes"].append(f"{info['n_gate_dropped']} clip(s) dropped by the speed / acceleration gate {info['gate']} (shortfall_ok {info['shortfall_ok']})")
    _finish_stats(st, rows, cells)
    ctx.rows += rows
    ctx.strata[entry["stratum"]] = st


def build_pool_stratum(entry: Mapping[str, Any], ctx: BuildContext) -> None:
    stratum, tier = str(entry["stratum"]), str(entry["tier"])
    pool = entry["pool"]
    level = LEVEL_INDEX[str(pool.get("level", "recoverable"))]
    st = _stratum_base(entry)
    st["pool"] = dict(pool)
    selected: list[dict[str, Any]] = []
    cells: dict[str, dict[str, Any]] = {}
    for cell, q in entry["quota"].items():
        source = cell[0]
        cands = sorted((r for r in ctx.pool_candidates.get(source, []) if r.get("_level") == level), key=_order_key)
        chosen = [{**r, "accept_level": level, "accept_level_name": LEVELS[level].name, "_pool_source": source} for r in cands[:int(q)]]
        selected += chosen
        cells[cell_key(cell)] = {"source": source, "quota": int(q), "n_candidates": len(cands), "n_selected": len(chosen), "shortfall": max(0, int(q) - len(chosen)),
                                 "source_built": source in ctx.strata, "selected": [[int(c["_batch"]), int(c["bench"]["goal_index"])] for c in chosen],
                                 "n_by_level": {LEVELS[level].name: len(cands)}}
        if source not in ctx.pool_candidates:
            st["notes"].append(f"source stratum {source} not built in this run (0 candidates)")
    out_dir = ctx.tier_dir(tier)
    out_dir.mkdir(parents=True, exist_ok=True)
    results = convert_rows(selected, out_dir, stratum, jobs=ctx.jobs) if selected else []
    errors = [r for r in results if not r["ok"]]
    if errors:
        raise BuildError(f"{stratum}: {len(errors)} / {len(results)} conversions failed: " + "; ".join(f"{Path(e['src']).name}: {e.get('error')}" for e in errors[:3]))
    entries = []
    for r, res in zip(selected, results):
        e = bench_clip_entry(r, Path(res["dst"]), res["receipt"], stratum=stratum, tier=tier)
        e["pool_source"] = r["_pool_source"]
        entries.append(e)
    _finish_stats(st, entries, cells)
    ctx.rows += entries
    ctx.strata[stratum] = st
    ctx.log(f"[hero_bench] {stratum} ({tier}): {len(entries)} / {st['quota_total']} recoverable-level clips pooled from {sorted({e['pool_source'] for e in entries})}")


BUILDERS: dict[str, Callable[[Mapping[str, Any], BuildContext], None]] = {
    "generate": build_generate_stratum, "verbatim": build_verbatim_stratum, "retime": build_retime_stratum_ctx, "pool": build_pool_stratum,
}


# --------------------------------------------------------------------------------------------------
# Build driver
# --------------------------------------------------------------------------------------------------
def _verify_tier(ctx: BuildContext, tier: str, files: Sequence[str]) -> dict[str, Any]:
    d = ctx.tier_dir(tier)
    if not files:
        return {"verifier": "none (empty tier)", "n_files": 0, "n_loaded": 0, "n_skipped": 0, "skipped": [], "notes": []}
    if ctx.verifier == "none":
        return {"verifier": "none (skipped by --verifier none)", "n_files": len(files), "n_loaded": 0, "n_skipped": 0, "skipped": [], "notes": ["loader pass skipped"]}
    return bb.verify_corpus(d, list(files), ctx.verifier)


def build(plan: Mapping[str, dict[str, Any]] | None = None, *, bank_root: Path | str | None, out_dir: Path | str, paper_dir: Path | str | None = None,
          generate: bool = False, jobs: int = 1, nice: int = DEFAULT_NICE, verifier: str = "auto", strata: Sequence[str] | None = None,
          public: bool = False, clean: bool = True, code_sha256: str | None = None, log: Callable[[str], None] = print) -> dict[str, Any]:
    """Build the corpus; returns the root manifest (written to ``<out>/BENCH_MANIFEST.json``).  Raises :class:`BuildError` /
    :class:`CoreGapError` (``exit_code`` 1 / 3).  ``code_sha256`` is recorded verbatim (e.g. the sha256 of the release tarball the node ran)."""
    plan = copy.deepcopy(dict(plan if plan is not None else R.BENCH_V1_PLAN))
    if verifier not in VERIFIERS:
        raise BuildError(f"verifier must be one of {VERIFIERS}, got {verifier!r}")
    out_dir = Path(out_dir)
    if paper_dir is None and DEFAULT_PAPER_DIR.is_dir():
        paper_dir = DEFAULT_PAPER_DIR
    ctx = BuildContext(plan=plan, bank_root=(Path(bank_root) if bank_root else None), out_dir=out_dir, paper_dir=(Path(paper_dir) if paper_dir else None),
                       generate=bool(generate), jobs=int(jobs), nice=int(nice), verifier=verifier, log=log, code_sha256=(str(code_sha256) if code_sha256 else None))
    if strata:
        unknown = sorted(set(strata) - set(plan))
        if unknown:
            raise BuildError(f"--strata: unknown strata {unknown}; plan has {list(plan)}")
    wanted = [e for e in plan.values() if not strata or e["stratum"] in strata]
    t0 = time.time()
    out_dir.mkdir(parents=True, exist_ok=True)
    for t in TIERS:
        ctx.tier_dir(t).mkdir(parents=True, exist_ok=True)
    (out_dir / WORK_DIR).mkdir(exist_ok=True)
    for entry in wanted:
        kind = entry["kind"]
        if kind not in BUILDERS:
            raise BuildError(f"{entry['stratum']}: unknown stratum kind {kind!r}")
        BUILDERS[kind](entry, ctx)
    # file-name uniqueness across the corpus
    dup = [f for f, n in Counter(r["file"] for r in ctx.rows).items() if n > 1]
    if dup:
        raise BuildError(f"duplicate output file names across strata: {dup[:5]}")
    # stale files
    removed: list[str] = []
    wanted_files = {(r["tier"], r["file"]) for r in ctx.rows}
    if clean:
        for t in TIERS:
            for p in ctx.tier_dir(t).glob("*.npz"):
                if (t, p.name) not in wanted_files:
                    p.unlink()
                    removed.append(f"{t}/{p.name}")
    verification: dict[str, Any] = {}
    for t in TIERS:
        verification[t] = _verify_tier(ctx, t, [r["file"] for r in ctx.rows if r["tier"] == t])
    verification["removed_stale_files"] = removed
    manifest = build_manifest(ctx, verification)
    manifest["build_wall_s"] = time.time() - t0
    manifest["strata_built"] = [e["stratum"] for e in wanted]
    if public:
        manifest = scrub_public_manifest(manifest)
    write_licenses(out_dir)
    write_tier_lists(out_dir, manifest)
    (out_dir / MANIFEST_NAME).write_text(json.dumps(manifest, indent=1, sort_keys=True))
    for t in TIERS:
        (ctx.tier_dir(t) / MANIFEST_NAME).write_text(json.dumps(tier_manifest(manifest, t), indent=1, sort_keys=True))
    write_report(manifest, out_dir / REPORT_NAME)
    write_sha256sums(out_dir)
    shutil.rmtree(out_dir / WORK_DIR, ignore_errors=True)
    shipped_txt = ", ".join(f"{t} {manifest['tier_totals_shipped'][t]}" for t in TIERS)
    log(f"[hero_bench] wrote {manifest['n_clips']} clips to {out_dir} ({shipped_txt}) in {manifest['build_wall_s']:.0f} s")
    return manifest


def summary_of(manifest: Mapping[str, Any]) -> dict[str, Any]:
    return {"schema": manifest["schema"], "n_clips": manifest["n_clips"], "tier_totals_shipped": manifest["tier_totals_shipped"], "tier_totals_planned": manifest["tier_totals_planned"],
            "protocol_sha256": manifest["protocol_sha256"], "strata": {n: {k: s.get(k) for k in ("tier", "kind", "shipped", "quota_total", "yield", "fill_batches_used", "health", "provisional", "shortfall_total")}
                                                                       for n, s in manifest["strata"].items()},
            "public": bool(manifest.get("public", False)), "build_wall_s": manifest.get("build_wall_s")}


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--plan", default="default", help="'default' (reach_specs.BENCH_V1_PLAN), 'reduced:<per_cell>' (smoke) or a JSON file in bench_v1_plan_json form")
    ap.add_argument("--bank-root", default=None, help="generator banks: <root>/<stratum>/batch<k>/{clips,labels,manifest.json}")
    ap.add_argument("--paper-dir", "--v1-dir", dest="paper_dir", default=None,
                    help=f"paper-protocol source dir: the 180 h050 / h074 / h088 files next to their {MANIFEST_NAME} (default {DEFAULT_PAPER_DIR} when it exists, env HERO_BENCH_PAPER_DIR; required for the paper / re-timed strata; --v1-dir is an alias)")
    ap.add_argument("--out", required=False, help="output corpus root (core/ extended/ stress/ tiers/ BENCH_MANIFEST.json ...)")
    ap.add_argument("--generate", action="store_true", help="run the generator for missing candidate batches (resumable: batches with manifest.json are kept)")
    ap.add_argument("--jobs", type=int, default=1, help="generator / converter worker processes")
    ap.add_argument("--nice", type=int, default=DEFAULT_NICE, help=f"nice level of the generator subprocess (0 = none; env HERO_BENCH_NICE, default {DEFAULT_NICE})")
    ap.add_argument("--verifier", default="auto", choices=VERIFIERS, help="loader-contract check of every shipped file (auto = HeroMultiMotionLoader, then holosoma, then schema)")
    ap.add_argument("--strata", default=None, help="comma-separated subset of strata to build (default: the whole plan)")
    ap.add_argument("--public", action="store_true", help="scrub host / paths / S3 / node names / internal git-sha suffixes from the manifests")
    ap.add_argument("--no-clean", action="store_true", help="keep stale *.npz in the tier dirs")
    ap.add_argument("--code-sha256", default=None, help="provenance: sha256 of the code archive this build ran from (e.g. the release tarball on the node); recorded as code_sha256")
    ap.add_argument("--scrub-only", action="store_true", help="only scrub the manifests of an existing --out corpus (no build)")
    ap.add_argument("--print-plan", action="store_true", help="print the plan table and exit")
    args = ap.parse_args(argv)
    try:
        plan = load_plan(args.plan)
        if args.print_plan:
            print(plan_table(plan))
            return EXIT_OK
        if not args.out:
            ap.error("--out is required")
        if args.scrub_only:
            pub = scrub_public_dir(Path(args.out))
            print(json.dumps({"scrubbed": str(args.out), "n_clips": pub["n_clips"], "protocol_sha256": pub["protocol_sha256"]}, indent=1))
            return EXIT_OK
        strata = [s.strip() for s in args.strata.split(",") if s.strip()] if args.strata else None
        manifest = build(plan, bank_root=args.bank_root, out_dir=args.out, paper_dir=args.paper_dir, generate=args.generate, jobs=args.jobs, nice=args.nice,
                         verifier=args.verifier, strata=strata, public=args.public, clean=not args.no_clean, code_sha256=args.code_sha256)
    except BuildError as exc:
        print(f"[hero_bench] ERROR: {exc}", file=sys.stderr)
        return exc.exit_code
    print(json.dumps(summary_of(manifest), indent=1))
    return EXIT_OK


__all__ = ["BUILDER_VERSION", "BuildContext", "BuildError", "CoreGapError", "DATA_LICENSE_NAME", "EXIT_CORE_GAP", "EXIT_ERROR", "EXIT_OK", "MANIFEST_NAME", "PLAN_VERSION",
           "PUBLIC_DROP_KEYS", "REPORT_NAME", "SCHEMA", "SUMS_NAME", "TIERS", "TIERS_DIR", "PAPER_LICENSE_NAME", "batch_clip_prefix", "batch_dir",
           "batch_seed", "build", "build_manifest", "build_retime_stratum", "canonical_json", "cell_of", "convert_rows", "generator_command", "import_verbatim",
           "load_batch", "load_batches", "load_plan", "load_paper", "main", "plan_from_json", "plan_json", "plan_table", "protocol_block", "protocol_sha256", "reduced_plan", "release_version",
           "run_generator", "scrub_public_dir", "scrub_public_manifest", "scrub_public_value", "select_stratum", "stratum_yield", "summary_of", "tier_manifest", "paper_subset",
           "bench_clip_entry", "bench_conversion_meta", "bench_output_name", "verify_shipped", "write_report", "write_sha256sums", "write_tier_lists"]


if __name__ == "__main__":
    raise SystemExit(main())
