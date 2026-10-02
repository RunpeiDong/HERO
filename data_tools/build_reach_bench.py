#!/usr/bin/env python3
"""Select the accepted reaching goals of a generator bank and convert them into a flat benchmark corpus (``hero_bench_v1`` /
``hero_bench_v2`` layout: ``<height_label>__<clip_id>.npz`` + ``BENCH_MANIFEST.json`` + ``BENCH_REPORT.md``).

Protocol (HERO, arXiv 2602.16705): 3 table heights (0.50 / 0.74 / 0.88 m) x 60 goals (30 right + 30 left hand), goal 5-15 cm above
the table top, side grasp with |yaw| <= 45 deg, robot 10-20 cm from the table edge.  Candidates come from
``python -m data_tools.hero_reach_generator --profile hero_bench_v1`` (or ``hero_bench_v2`` with its grasp-orientation families);
every bank row / labels file carries a ``bench`` dict.

Steps
-----
1. Load ``<bank>/manifest.json`` (+ ``labels/`` when a row lacks ``bench``).
2. Acceptance (``ACCEPT_LEVELS``): strict = generator tier ``core`` and terminal EE position error <= 1 cm (orientation <= 5 deg)
   with no collision / limit flag; fallback levels ``core`` (<= 1.5 cm) and ``recoverable`` (<= 3 cm / 10 deg, soft flags allowed).
   Per (height[, orientation family], hand) the first ``--per-hand`` accepted goals by ``goal_index`` are kept; a short group is
   filled by the next level in goal-index order (``accept_level`` recorded per clip; ``--no-fallback`` reports the shortfall instead).
   hero_bench_v2 banks select per family with ``--family-quota`` and may borrow strict clips across families (``--no-family-borrow``).
3. Conversion through ``data_tools.npz_convert.convert_clip`` (FK on the Dex3 scene, finite-difference velocities, HERO extension
   keys, no ``legs_default_standing``), ``source_tag`` = height label, ``license_class`` apache, ``parent_id`` = clip_id.
4. Loader-contract check of every output (``hero_isaacsim.managers.command.loader.HeroMultiMotionLoader``; fallbacks
   ``--verifier holosoma`` / ``schema``); a skipped or rejected file fails the build.
5. ``BENCH_MANIFEST.json`` (protocol, selection statistics, verification, one row per clip) and ``BENCH_REPORT.md``.

    python -m data_tools.build_reach_bench --bank-dir <bank> --out-dir <corpus> --per-hand 30 --jobs 16

``data_tools.build_hero_bench`` (the stratified hero_bench_v1 builder) imports the acceptance levels, selection helpers,
conversion worker, loader check and manifest row writer from this module.
"""

from __future__ import annotations

import argparse
import dataclasses
import hashlib
import json
import math
import os
import platform
import subprocess
import sys
import time
from collections import Counter, defaultdict
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from data_tools import reach_specs as rs
from data_tools.schema import DOF_NAMES, EXTENSION_KEYS, HOLOSOMA_BODY_NAMES_32, decode_names, read_npz, validate_holosoma_npz

BENCH_SCHEMA: str = rs.BENCH_SCHEMA
FPS: int = 50
LICENSE_CLASS: str = "apache"        # data_tools/sources.yaml row ik_reach_v2 (generated in-house)
CLIP_END_POLICY: str = "hold"        # hero_isaacsim.config_values.command.DEFAULT_CLIP_END_POLICY_BY_SOURCE["ik_reach_v2"]
DEFAULT_PER_HAND: int = 30
REPO_ROOT: Path = Path(__file__).resolve().parents[1]
PAPER: str = "HERO, arXiv 2602.16705 (sim2real/simulation_benchmark/sim_benchmark_scene.xml, sim2real/data/multi_table_data_distribution/test_1106.csv)"


# --------------------------------------------------------------------------------------------------
# Acceptance
# --------------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class AcceptLevel:
    name: str
    tiers: tuple[str, ...]
    max_ee_pos_m: float
    max_ee_ori_rad: float
    allowed_flags: tuple[str, ...]

    def accepts(self, tier: str, flags: set[str], e_pos: float, e_ori: float) -> bool:
        return tier in self.tiers and e_pos <= self.max_ee_pos_m and e_ori <= self.max_ee_ori_rad and flags <= set(self.allowed_flags)

    def as_dict(self) -> dict[str, Any]:
        return {**dataclasses.asdict(self), "max_ee_ori_deg": math.degrees(self.max_ee_ori_rad)}


#: Flags that always reject (collision / limit / numerical): a clip carrying one never enters the benchmark.
HARD_REJECT_FLAGS: tuple[str, ...] = ("self_collision", "table_contact", "floor_contact", "joint_limit_boundary", "non_finite")
#: Soft flags tolerated only by the last fallback level (they make a clip ``recoverable`` in the generator's tiering).
SOFT_FLAGS: tuple[str, ...] = ("coarse_ee_position", "coarse_ee_orientation", "foot_drift", "fast_motion", "high_acceleration", "support_boundary")
ACCEPT_LEVELS: tuple[AcceptLevel, ...] = (
    AcceptLevel("strict", ("core",), 0.010, math.radians(5.0), ("qp_fallback",)),
    AcceptLevel("core", ("core",), 0.015, math.radians(5.0), ("qp_fallback",)),
    AcceptLevel("recoverable", ("core", "recoverable"), 0.030, math.radians(10.0), ("qp_fallback", *SOFT_FLAGS)),
)


def level_of(row: dict[str, Any], levels: Sequence[AcceptLevel] = ACCEPT_LEVELS) -> int | None:
    """Index of the first acceptance level the manifest row satisfies (None = never accepted)."""
    if not row.get("ok", True) or row.get("bench") is None:
        return None
    tier = str(row.get("tier"))
    flags = set(row.get("flags") or [])
    e_pos = float(row.get("terminal_ee_pos_error_m", math.inf))
    e_ori = float(row.get("terminal_ee_ori_error_rad", math.inf))
    if not (math.isfinite(e_pos) and math.isfinite(e_ori)):
        return None
    for i, lv in enumerate(levels):
        if lv.accepts(tier, flags, e_pos, e_ori):
            return i
    return None


def strict_reject_reasons(row: dict[str, Any], level: AcceptLevel = ACCEPT_LEVELS[0]) -> list[str]:
    """Why a row misses the strict level (report bookkeeping; empty when it passes)."""
    if not row.get("ok", True):
        return ["generator_error"]
    if row.get("bench") is None:
        return ["no_bench_labels"]
    out: list[str] = []
    tier = str(row.get("tier"))
    if tier not in level.tiers:
        out.append(f"tier:{tier}")
    if float(row.get("terminal_ee_pos_error_m", math.inf)) > level.max_ee_pos_m:
        out.append(f"ee_pos>{level.max_ee_pos_m * 1e3:.0f}mm")
    if float(row.get("terminal_ee_ori_error_rad", math.inf)) > level.max_ee_ori_rad:
        out.append(f"ee_ori>{math.degrees(level.max_ee_ori_rad):.0f}deg")
    bad = sorted(set(row.get("flags") or []) - set(level.allowed_flags))
    if bad:
        out.append("flags:" + "+".join(bad))
    return out


# --------------------------------------------------------------------------------------------------
# Bank loading and selection (pure python: unit-tested on synthetic manifests)
# --------------------------------------------------------------------------------------------------
def load_bank(bank_dir: Path) -> tuple[dict[str, Any], list[dict[str, Any]]]:
    """``manifest.json`` rows with their ``bench`` dict (read from ``labels/`` when the row lacks it)."""
    bank_dir = Path(bank_dir)
    man_path = bank_dir / "manifest.json"
    if not man_path.is_file():
        raise FileNotFoundError(f"{man_path} not found (run hero_reach_generator --profile hero_bench_v1 first)")
    manifest = json.loads(man_path.read_text())
    rows: list[dict[str, Any]] = []
    for row in manifest.get("clips", []):
        row = dict(row)
        if row.get("ok", True) and row.get("bench") is None and row.get("labels"):
            lab_path = bank_dir / row["labels"]
            if lab_path.is_file():
                row["bench"] = json.loads(lab_path.read_text()).get("bench")
        rows.append(row)
    n_bench = sum(1 for r in rows if r.get("bench"))
    if n_bench == 0:
        raise ValueError(f"{man_path}: no clip carries hero_bench_v1 labels (profile {manifest.get('profile')!r}); not a benchmark bank")
    return manifest, rows


def bank_schema(rows: Sequence[dict[str, Any]]) -> str:
    """Bench schema of the bank (``bench.schema`` of the rows; ``hero_bench_v1`` when the rows do not say)."""
    for r in rows:
        b = r.get("bench")
        if b and b.get("schema"):
            return str(b["schema"])
    return rs.BENCH_SCHEMA


def bank_has_families(rows: Sequence[dict[str, Any]]) -> bool:
    return any((r.get("bench") or {}).get("orient_family") for r in rows)


def default_family_quota(rows: Sequence[dict[str, Any]], per_hand: int) -> dict[str, int] | None:
    """Per-(height, hand) family quota from the profile shares of the bank's schema (None for banks without families).
    Families found in the bank but not in the profile get quota 0 (reported, never selected unless --family-quota says so)."""
    if not bank_has_families(rows):
        return None
    schema = bank_schema(rows)
    fams = (rs.PROFILES.get(schema) or {}).get("bench_orient_families") or rs.BENCH_V2_ORIENT_FAMILIES
    quota = rs.bench_family_quota(per_hand, fams)
    for r in rows:
        f = (r.get("bench") or {}).get("orient_family")
        if f and f not in quota:
            quota[str(f)] = 0
    return quota


def parse_family_quota(text: str | None) -> dict[str, int] | None:
    """``"fan_side=50,top_down=25,tilted=25"`` -> dict (None / empty -> None = profile shares)."""
    if not text:
        return None
    out: dict[str, int] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, value = item.partition("=")
        out[name.strip()] = int(value)
    return out


def _group_key(b: dict[str, Any], with_family: bool) -> tuple[str, ...]:
    if with_family:
        return (str(b["height_label"]), str(b.get("orient_family") or ""), str(b["hand"]))
    return (str(b["height_label"]), str(b["hand"]))


def select_goals(rows: Sequence[dict[str, Any]], per_hand: int = DEFAULT_PER_HAND, *, levels: Sequence[AcceptLevel] = ACCEPT_LEVELS,
                 allow_fallback: bool = True, heights: Sequence[str] | None = None, family_quota: dict[str, int] | None = None,
                 allow_family_borrow: bool = True) -> tuple[list[dict[str, Any]], dict[str, dict[str, Any]]]:
    """Per (height label[, orient_family], hand): the first ``quota`` accepted goals by ``goal_index``; strict level first,
    then (if the group is short and ``allow_fallback``) the next levels in goal-index order.

    v1 banks (no ``orient_family``): quota = ``per_hand``, stats keyed ``"<height_label>/<hand>"``.
    v2 banks: quota per family = ``family_quota[family]`` (default :func:`default_family_quota`: profile share x ``per_hand``),
    stats keyed ``"<height_label>/<family>/<hand>"``.  A group still short after the level fallback is filled, when
    ``allow_family_borrow``, with unused *strict*-level candidates of the other families of the same (height, hand) in
    goal-index order (``accept_note = "borrowed_from:<family>"``, ``borrowed = True``; the row keeps its own ``orient_family``),
    so the (height, hand) total stays ``per_hand`` whenever the strict pool allows; remaining shortfall is reported.
    Returns (selected rows with ``accept_level`` / ``accept_level_name`` [/ ``accept_note``], per-group statistics)."""
    with_family = bank_has_families(rows)
    if with_family and family_quota is None:
        family_quota = default_family_quota(rows, per_hand)
    groups: dict[tuple[str, ...], list[dict[str, Any]]] = defaultdict(list)
    for r in rows:
        b = r.get("bench")
        if not b:
            continue
        if heights is not None and b["height_label"] not in heights:
            continue
        groups[_group_key(b, with_family)].append(r)
    selected: list[dict[str, Any]] = []
    stats: dict[str, dict[str, Any]] = {}
    chosen_by_key: dict[tuple[str, ...], list[dict[str, Any]]] = {}
    strict_unused: dict[tuple[str, ...], list[dict[str, Any]]] = {}   # per group: strict-level rows not selected (borrow pool)
    n_levels = len(levels) if allow_fallback else 1
    for key in sorted(groups):
        cand = sorted(groups[key], key=lambda r: int(r["bench"]["goal_index"]))
        lv = [level_of(r, levels) for r in cand]
        quota = per_hand if not with_family else int((family_quota or {}).get(key[1], 0))
        chosen: list[dict[str, Any]] = []
        for L in range(n_levels):
            for r, l in zip(cand, lv):
                if l == L and len(chosen) < quota:
                    chosen.append({**r, "accept_level": L, "accept_level_name": levels[L].name})
            if len(chosen) >= quota:
                break
        chosen_ids = {c["clip_id"] for c in chosen}
        strict_unused[key] = [r for r, l in zip(cand, lv) if l == 0 and r["clip_id"] not in chosen_ids]
        reasons: Counter[str] = Counter()
        for r, l in zip(cand, lv):
            if l != 0:
                for reason in strict_reject_reasons(r, levels[0]) or ["unknown"]:
                    reasons[reason] += 1
        st: dict[str, Any] = {
            "height_label": key[0], "hand": key[-1], "height_m": float(cand[0]["bench"]["height_m"]),
            "n_candidates": len(cand), "n_by_level": {levels[i].name: int(sum(1 for l in lv if l == i)) for i in range(len(levels))},
            "n_never_accepted": int(sum(1 for l in lv if l is None)), "n_selected": len(chosen), "per_hand_target": per_hand, "quota": quota,
            "shortfall": max(0, quota - len(chosen)),
            "max_level_used": levels[max(c["accept_level"] for c in chosen)].name if chosen else None,
            "fallback_used": bool(chosen) and any(c["accept_level"] > 0 for c in chosen),
            "n_selected_by_level": {levels[i].name: int(sum(1 for c in chosen if c["accept_level"] == i)) for i in range(len(levels))},
            "strict_reject_reasons": dict(reasons.most_common()),
            "selected_goal_indices": [int(c["bench"]["goal_index"]) for c in chosen],
            "n_borrowed": 0, "borrowed_from": {},
        }
        if with_family:
            st["orient_family"] = key[1]
        stats["/".join(key)] = st
        chosen_by_key[key] = chosen
    if with_family and allow_family_borrow:
        # family borrow: fill short groups from the strict pools of the other families of the same (height, hand), goal order
        for key in sorted(chosen_by_key):
            st = stats["/".join(key)]
            if st["shortfall"] <= 0:
                continue
            donors = sorted(k for k in strict_unused if k != key and k[0] == key[0] and k[-1] == key[-1])
            pool = sorted((r for k in donors for r in strict_unused[k]), key=lambda r: int(r["bench"]["goal_index"]))
            taken: list[dict[str, Any]] = []
            for r in pool:
                if len(taken) >= st["shortfall"]:
                    break
                taken.append({**r, "accept_level": 0, "accept_level_name": levels[0].name, "borrowed": True,
                              "accept_note": f"borrowed_from:{r['bench'].get('orient_family')}"})
            for t in taken:
                for k in donors:
                    strict_unused[k] = [r for r in strict_unused[k] if r["clip_id"] != t["clip_id"]]
            if taken:
                chosen_by_key[key] += taken
                st["n_selected"] = len(chosen_by_key[key])
                st["shortfall"] = max(0, st["quota"] - st["n_selected"])
                st["n_borrowed"] = len(taken)
                st["borrowed_from"] = dict(Counter(str(t["bench"].get("orient_family")) for t in taken))
                st["fallback_used"] = True
                st["max_level_used"] = levels[max(c["accept_level"] for c in chosen_by_key[key])].name
                st["selected_goal_indices"] = sorted(int(c["bench"]["goal_index"]) for c in chosen_by_key[key])
    for key in sorted(chosen_by_key):
        chosen = sorted(chosen_by_key[key], key=lambda r: int(r["bench"]["goal_index"]))
        selected += chosen
    return selected, stats


# --------------------------------------------------------------------------------------------------
# Conversion (same code path as the production ik_reach_v2 source) — module-level worker for the spawn pool
# --------------------------------------------------------------------------------------------------
def conversion_meta(row: dict[str, Any]) -> dict[str, Any]:
    """The plain-dict metadata ``normalize_corpus`` hands to ``convert_clip`` for the ``ik_reach_v2`` source, with the
    benchmark's ``source_tag`` (height label) and ``parent_id`` (clip id).  Keys outside ``ClipMeta`` land in ``ClipMeta.extra``."""
    b = row["bench"]
    return {
        "source_tag": str(b["height_label"]), "kind": "ik_bank", "family": "ee_reach", "needs_convert": True,
        "parent_id": str(row["clip_id"]), "license_class": LICENSE_CLASS, "clip_end_policy": CLIP_END_POLICY,
        "object_policy": "strip", "legs_default_standing": False, "target_fps": FPS, "name": str(row["clip_id"]),
        "tier": row.get("tier"), "core": False, "bench_schema": str(b.get("schema") or BENCH_SCHEMA),
    }


def output_name(row: dict[str, Any]) -> str:
    return f"{row['bench']['height_label']}__{row['clip_id']}.npz"


def _convert_worker(job: tuple[str, str, dict[str, Any]]) -> dict[str, Any]:
    src, dst, meta = job
    os.environ.setdefault("OMP_NUM_THREADS", "1")
    try:
        from data_tools import fk_mujoco as fkm
        from data_tools.npz_convert import convert_clip

        receipt = convert_clip(src, dst, meta, fk=fkm.get_fk(), overwrite=True)
        return {"src": src, "dst": dst, "ok": receipt.get("status") == "ok", "receipt": receipt}
    except Exception as e:  # noqa: BLE001 - per-clip report, the build decides
        import traceback

        return {"src": src, "dst": dst, "ok": False, "error": f"{type(e).__name__}: {e}", "traceback": traceback.format_exc()[-1500:]}


def convert_selected(selected: Sequence[dict[str, Any]], bank_dir: Path, out_dir: Path, jobs: int = 1) -> list[dict[str, Any]]:
    jobs_list = [(str(Path(bank_dir) / r["file"]), str(Path(out_dir) / output_name(r)), conversion_meta(r)) for r in selected]
    results: list[dict[str, Any]] = []
    if jobs <= 1 or len(jobs_list) <= 1:
        for j in jobs_list:
            results.append(_convert_worker(j))
    else:
        import multiprocessing as mp

        with mp.get_context("spawn").Pool(min(jobs, len(jobs_list))) as pool:
            results = list(pool.imap(_convert_worker, jobs_list))
    return results


def _sha256(path: Path) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def _git_sha() -> str:
    """Repository HEAD; ``HERO_GIT_SHA`` overrides when running without a Git checkout."""
    if os.environ.get("HERO_GIT_SHA"):
        return os.environ["HERO_GIT_SHA"]
    try:
        return subprocess.check_output(["git", "rev-parse", "HEAD"], cwd=str(REPO_ROOT), stderr=subprocess.DEVNULL, text=True).strip()
    except Exception:  # noqa: BLE001
        return "unknown"


# --------------------------------------------------------------------------------------------------
# Loader-contract verification
# --------------------------------------------------------------------------------------------------
def verify_corpus(out_dir: Path, files: Sequence[str], mode: str = "auto") -> dict[str, Any]:
    """Load every benchmark file with the training-side loader; raises RuntimeError when any file is skipped / rejected."""
    out_dir = Path(out_dir)
    notes: list[str] = []
    result: dict[str, Any] = {"mode": mode, "n_files": len(files)}
    if mode in ("auto", "hero"):
        try:
            from hero_isaacsim.managers.command.loader import HeroMultiMotionLoader
        except Exception as e:  # noqa: BLE001
            notes.append(f"HeroMultiMotionLoader import failed: {type(e).__name__}: {e}")
            if mode == "hero":
                raise
        else:
            ld = HeroMultiMotionLoader(str(out_dir), list(HOLOSOMA_BODY_NAMES_32), list(DOF_NAMES), device="cpu", expected_fps=float(FPS))
            loaded = [Path(f).name for f in ld.clip_files]
            result.update({
                "verifier": "hero_isaacsim.managers.command.loader.HeroMultiMotionLoader", "n_loaded": int(ld._num_motions),
                "n_skipped": int(ld.num_skipped), "skipped": [[Path(f).name, err] for f, err in ld.skipped_files],
                "source_tags": sorted(set(ld.clip_source_tag)), "total_frames": int(sum(int(x) for x in ld.clip_lengths.tolist())),
                "clip_lengths": {n: int(l) for n, l in zip(loaded, ld.clip_lengths.tolist())},
                "unexpected_files": sorted(set(loaded) - set(files)), "missing_files": sorted(set(files) - set(loaded)), "notes": notes,
            })
            _raise_on_skips(result)
            return result
    if mode in ("auto", "holosoma"):
        try:
            from holosoma.managers.command.terms.wbt import MotionLoader
        except Exception as e:  # noqa: BLE001
            notes.append(f"holosoma MotionLoader import failed: {type(e).__name__}: {e}")
            if mode == "holosoma":
                raise
        else:
            skipped: list[list[str]] = []
            lengths: dict[str, int] = {}
            for name in files:
                try:
                    ld = MotionLoader(str(out_dir / name), list(HOLOSOMA_BODY_NAMES_32), list(DOF_NAMES), device="cpu", expected_fps=float(FPS))
                    lengths[name] = int(ld.joint_pos.shape[0])
                except Exception as e:  # noqa: BLE001
                    skipped.append([name, f"{type(e).__name__}: {e}"])
            result.update({"verifier": "holosoma.managers.command.terms.wbt.MotionLoader (per file)", "n_loaded": len(lengths),
                           "n_skipped": len(skipped), "skipped": skipped, "clip_lengths": lengths, "unexpected_files": [], "missing_files": [], "notes": notes})
            _raise_on_skips(result)
            return result
    # schema-level fallback (no torch): flatten_corpus --body-check logic + the full validator
    skipped = []
    lengths = {}
    tags: set[str] = set()
    for name in files:
        try:
            z = read_npz(out_dir / name)
            problems = validate_holosoma_npz(z, require_extension=True, expected_fps=FPS)
            if decode_names(z["body_names"]) != list(HOLOSOMA_BODY_NAMES_32):
                problems.append("body_names != canonical 32 list")
            if z["body_pos_w"].shape[1:] != (32, 3):
                problems.append(f"body_pos_w shape {z['body_pos_w'].shape}")
            if problems:
                raise ValueError("; ".join(problems))
            lengths[name] = int(z["joint_pos"].shape[0])
            tags.add(str(np.asarray(z["source_tag"]).reshape(()).item()))
        except Exception as e:  # noqa: BLE001
            skipped.append([name, f"{type(e).__name__}: {e}"])
    result.update({"verifier": "data_tools.schema.validate_holosoma_npz (+ 32-body check)", "n_loaded": len(lengths), "n_skipped": len(skipped),
                   "skipped": skipped, "source_tags": sorted(tags), "clip_lengths": lengths, "unexpected_files": [], "missing_files": [], "notes": notes})
    _raise_on_skips(result)
    return result


def _raise_on_skips(result: dict[str, Any]) -> None:
    if result["n_skipped"] or result["n_loaded"] != result["n_files"] or result.get("unexpected_files") or result.get("missing_files"):
        raise RuntimeError(f"loader contract violated: {json.dumps({k: result.get(k) for k in ('verifier', 'n_files', 'n_loaded', 'n_skipped', 'skipped', 'unexpected_files', 'missing_files')}, indent=1)}")


# --------------------------------------------------------------------------------------------------
# Manifest / report
# --------------------------------------------------------------------------------------------------
def clip_entry(row: dict[str, Any], out_path: Path, receipt: dict[str, Any]) -> dict[str, Any]:
    b = row["bench"]
    z = read_npz(out_path)
    n_frames = int(z["joint_pos"].shape[0])
    if n_frames != int(b["n_frames"]):
        raise ValueError(f"{out_path.name}: converted frames {n_frames} != bank frames {b['n_frames']} (fps mismatch?)")
    return {
        "file": out_path.name, "clip_id": row["clip_id"], "source_tag": b["height_label"], "height_m": float(b["height_m"]),
        "height_label": b["height_label"], "hand": b["hand"], "goal_index": int(b["goal_index"]), "candidate_index": int(b.get("candidate_index", -1)),
        "target_pos_w": [float(v) for v in b["target_pos_w"]],
        "target_z_raw": float(b["target_z_raw"]), "clearance_raised": bool(b.get("clearance_raised", False)),
        "target_x_raw": float(b.get("target_x_raw", b["target_pos_w"][0])), "target_x_shifted": bool(b.get("target_x_shifted", False)),
        "target_yaw_deg": float(b["target_yaw_deg"]), "target_pitch_deg": float(b["target_pitch_deg"]), "target_roll_deg": float(b["target_roll_deg"]),
        "table_top_z": float(b["table_top_z"]), "table_edge_x": float(b["table_edge_x"]), "table_edge_gap_m": float(b.get("table_edge_gap_m", math.nan)),
        "table_edge_ref": b.get("table_edge_ref"), "table": b.get("table"),
        "base_family": b.get("base_family"), "pelvis_drop": float(b["pelvis_drop"]), "pelvis_pitch_deg": float(b.get("pelvis_pitch_deg", 0.0)),
        "waist_pitch_deg": float(b.get("waist_pitch_deg", 0.0)), "reach_prefilter_ok": bool(b.get("reach_prefilter_ok", True)),
        "settle_frames": int(b["settle_frames"]), "reach_start_frame": int(b.get("reach_start_frame", b["settle_frames"])),
        "reach_end_frame": int(b["reach_end_frame"]), "reach_frames": int(b["reach_frames"]), "hold_frames": int(b["hold_frames"]),
        "n_frames": n_frames, "fps": FPS, "duration_s": n_frames / FPS,
        "terminal_ee_pos_err_m": float(row["terminal_ee_pos_error_m"]), "terminal_ee_ori_err_rad": float(row["terminal_ee_ori_error_rad"]),
        "ee_pos_err_at_reach_end_m": float(b.get("ee_pos_error_at_reach_end_m", math.nan)), "ee_pos_err_hold_max_m": float(b.get("ee_pos_error_hold_max_m", math.nan)),
        "pelvis_height_min_m": float(row["min_pelvis_height_m"]), "h_ref_min_m": float(z["h_ref"].min()), "h_ref_max_m": float(z["h_ref"].max()),
        "pelvis_xy_frame0": [float(z["joint_pos"][0, 0]), float(z["joint_pos"][0, 1])],
        "tier": row["tier"], "flags": list(row.get("flags") or []), "accept_level": int(row["accept_level"]), "accept_level_name": row["accept_level_name"],
        "accept_note": row.get("accept_note"), "borrowed": bool(row.get("borrowed", False)),
        # hero_bench_v2 orientation family (None / NaN for v1 banks)
        "orient_family": b.get("orient_family"), "family_index": b.get("family_index"),
        "yaw_deg": float(b.get("yaw_deg", b["target_yaw_deg"])), "pitch_deg": float(b.get("pitch_deg", b["target_pitch_deg"])),
        "roll_deg": float(b.get("roll_deg", b["target_roll_deg"])), "roll_sign": b.get("roll_sign"),
        "rot_from_canonical_deg": float(b.get("rot_from_canonical_deg", math.nan)), "rot_from_side_grasp_deg": float(b.get("rot_from_side_grasp_deg", math.nan)),
        "canonical_ypr_deg": b.get("canonical_ypr_deg"), "orient_resamples": b.get("orient_resamples"),
        "clearance_margin_m": float(b.get("clearance_margin_m", math.nan)), "target_z_min_clearance": float(b.get("target_z_min_clearance", math.nan)),
        "max_qdot_rad_s": float(row.get("max_qdot_rad_s", math.nan)), "min_com_margin_m": float(row.get("min_com_margin_m", math.nan)),
        "bank_file": row["file"], "bank_sha256": row.get("sha256"), "sha256": _sha256(out_path), "bytes": out_path.stat().st_size,
        "convert_family": receipt.get("family"), "license_class": LICENSE_CLASS, "parent_id": row["clip_id"], "has_object": False,
    }


def _stats(values: Sequence[float]) -> dict[str, float]:
    a = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if a.size == 0:
        return {"n": 0}
    return {"n": int(a.size), "mean": float(a.mean()), "p50": float(np.median(a)), "p95": float(np.quantile(a, 0.95)), "min": float(a.min()), "max": float(a.max())}


def _clip_group_stats(cs: Sequence[dict[str, Any]]) -> dict[str, Any]:
    """Selected-clip statistics of one group (height / family / family x height)."""
    return {"n_clips": len(cs),
            "right": sum(1 for c in cs if c["hand"] == "right"), "left": sum(1 for c in cs if c["hand"] == "left"),
            "n_fallback": sum(1 for c in cs if c["accept_level"] > 0), "n_borrowed": sum(1 for c in cs if c.get("borrowed")),
            "terminal_ee_pos_err_m": _stats([c["terminal_ee_pos_err_m"] for c in cs]),
            "terminal_ee_ori_err_deg": _stats([math.degrees(c["terminal_ee_ori_err_rad"]) for c in cs if c.get("terminal_ee_ori_err_rad") is not None]),
            "reach_duration_s": _stats([c["reach_frames"] / FPS for c in cs]),
            "duration_s": _stats([c["duration_s"] for c in cs]),
            "pelvis_height_min_m": _stats([c["pelvis_height_min_m"] for c in cs]),
            "target_z_above_table_m": _stats([c["target_pos_w"][2] - c["table_top_z"] for c in cs]),
            "rot_from_canonical_deg": _stats([c.get("rot_from_canonical_deg", math.nan) for c in cs]),
            "rot_from_side_grasp_deg": _stats([c.get("rot_from_side_grasp_deg", math.nan) for c in cs]),
            "yaw_deg": _stats([c.get("yaw_deg", c.get("target_yaw_deg", math.nan)) for c in cs]),
            "pitch_deg": _stats([c.get("pitch_deg", c.get("target_pitch_deg", math.nan)) for c in cs]),
            "abs_roll_deg": _stats([abs(c.get("roll_deg", c.get("target_roll_deg", math.nan))) for c in cs])}


def family_tables(clips: Sequence[dict[str, Any]], stats: dict[str, dict[str, Any]], labels: Sequence[str]) -> tuple[dict[str, Any], dict[str, Any]]:
    """``per_family`` (all heights) and ``per_family_height`` (``"<family>/<height_label>"``) blocks: accepted / candidates from
    the selection stats plus the selected-clip statistics.  Empty dicts for v1 banks."""
    fams = sorted({str(c.get("orient_family")) for c in clips if c.get("orient_family")} | {str(s["orient_family"]) for s in stats.values() if s.get("orient_family")})
    if not fams:
        return {}, {}
    per_family: dict[str, Any] = {}
    per_family_height: dict[str, Any] = {}
    for fam in fams:
        ss = [s for s in stats.values() if s.get("orient_family") == fam]
        cs = [c for c in clips if c.get("orient_family") == fam]
        per_family[fam] = {**_clip_group_stats(cs), "n_candidates": sum(s["n_candidates"] for s in ss), "quota_total": sum(s["quota"] for s in ss),
                           "n_strict_candidates": sum(s["n_by_level"].get("strict", 0) for s in ss),
                           "n_slots_filled": sum(s["n_selected"] for s in ss), "n_lent_out": sum(1 for c in cs if c.get("borrowed")),
                           "shortfall": sum(s["shortfall"] for s in ss)}
        for lab in labels:
            ssh = [s for s in ss if s["height_label"] == lab]
            csh = [c for c in cs if c["height_label"] == lab]
            per_family_height[f"{fam}/{lab}"] = {**_clip_group_stats(csh), "orient_family": fam, "height_label": lab,
                                                 "n_candidates": sum(s["n_candidates"] for s in ssh), "quota_total": sum(s["quota"] for s in ssh),
                                                 "n_strict_candidates": sum(s["n_by_level"].get("strict", 0) for s in ssh),
                                                 "n_slots_filled": sum(s["n_selected"] for s in ssh), "shortfall": sum(s["shortfall"] for s in ssh)}
    return per_family, per_family_height


def build_manifest(*, bank_manifest: dict[str, Any], bank_dir: Path, out_dir: Path, clips: Sequence[dict[str, Any]], stats: dict[str, dict[str, Any]],
                   verification: dict[str, Any], per_hand: int, allow_fallback: bool, profile_cfg: dict[str, Any] | None = None,
                   schema: str | None = None, family_quota: dict[str, int] | None = None, allow_family_borrow: bool = True) -> dict[str, Any]:
    profile = str(bank_manifest.get("profile") or "")
    if schema is None:
        schema = profile if profile in rs.PROFILES and rs.PROFILES[profile].get("sampler") == "bench" else BENCH_SCHEMA
    cfg = dict(rs.PROFILES.get(schema) or rs.PROFILES["hero_bench_v1"])
    cfg.update(bank_manifest.get("overrides") or {})
    if profile_cfg:
        cfg.update(profile_cfg)
    heights = sorted({float(c["height_m"]) for c in clips}) or [float(h) for h in cfg["bench_heights"]]
    labels = [rs.height_label(h) for h in heights]
    per_height = {lab: _clip_group_stats([c for c in clips if c["height_label"] == lab]) for lab in labels}
    per_family, per_family_height = family_tables(clips, stats, labels)
    families_cfg = cfg.get("bench_orient_families") if per_family else None
    if per_family and family_quota is None:
        family_quota = {f: int(next((s["quota"] for s in stats.values() if s.get("orient_family") == f), 0)) for f in per_family}
    bank_man_path = Path(bank_dir) / "manifest.json"
    return {
        "schema": schema,
        "created_utc": time.strftime("%Y-%m-%dT%H:%M:%SZ", time.gmtime()),
        "git_sha": _git_sha(), "host": platform.node(), "python": sys.version.split()[0],
        "bank_dir": str(bank_dir), "bank_manifest_sha256": _sha256(bank_man_path) if bank_man_path.is_file() else None,
        "bank": {k: bank_manifest.get(k) for k in ("schema", "generator_version", "created_utc", "git_sha", "profile", "seed", "n_requested", "n_written", "n_failed", "tier_counts", "flag_counts", "mjcf", "mjcf_sha256", "overrides")},
        "heights_m": heights, "height_labels": labels, "n_clips": len(clips), "per_hand_target": per_hand, "goals_per_height_target": 2 * per_hand,
        "protocol": {
            "paper": PAPER,
            "table_heights_m": [float(h) for h in cfg["bench_heights"]], "goals_per_height": 2 * per_hand,
            "hands": f"{per_hand} right + {per_hand} left per height; candidates alternate right (even goal_index) / left (odd)",
            "candidates_per_height": int(bank_manifest.get("n_requested", 0) // max(1, len(cfg["bench_heights"]))) if bank_manifest.get("n_requested") else cfg["bench_candidates_per_height"],
            "target_x_range_m": list(cfg["bench_x_range"]), "target_abs_y_range_m": list(cfg["bench_abs_y_range"]),
            "target_frame": "heading frame == clip world frame: stance centre at the origin, x forward, y left, z up; the solved pelvis xy at frame 0 "
                            "lies within ~5 mm of the origin (per clip: pelvis_xy_frame0); target_pos_w is in that frame",
            "target_z_above_table_m": list(cfg["bench_z_above_range"]),
            "z_clearance_rule": ("z raised to table + (-reach_specs.hand_lowest_offset(palm rotation, hand)) + clearance_margin_m when the sampled z is lower "
                                 "(Dex3 hand envelope; margin 2 cm, hero_bench_v2 top_down 3.5 cm = 2 cm + 5 deg orientation tolerance x 0.175 m fingertip lever); "
                                 "target_z_raw keeps the sample, target_z_min_clearance the bound"),
            "grasp_yaw_deg": list(cfg["bench_yaw_deg"]) if "bench_yaw_deg" in cfg else None,
            "pitch_deg": list(cfg["bench_pitch_deg"]) if "bench_pitch_deg" in cfg else None,
            "roll_deg": list(cfg["bench_roll_deg"]) if "bench_roll_deg" in cfg else None,
            "orientation_families": None if not families_cfg else {
                "families": {k: {kk: vv for kk, vv in v.items()} for k, v in families_cfg.items()},
                "hard_rejects": {"max_abs_roll_deg": cfg.get("bench_max_abs_roll_deg"), "max_rot_from_canonical_deg": cfg.get("bench_max_rot_from_canonical_deg"),
                                 "rule": "resample the (yaw, pitch, roll) triple (orient_resamples counts the draws)"},
                "candidate_layout": "goal indices blocked by family (reach_specs.bench_family_layout: share x candidates per height, largest remainder), hands alternating right / left inside each block",
                "quota_per_height_hand": family_quota, "family_borrow_allowed": allow_family_borrow,
                "family_borrow_rule": "a (height, family, hand) group still short after the level fallback takes unused strict-level candidates of the other families of the same (height, hand) in goal-index order; the clip keeps its own orient_family (accept_note = borrowed_from:<family>)",
                "rot_from_canonical_deg": "geodesic angle between the goal palm rotation and the family's canonical grasp (canonical_ypr_deg; tilted mirrors the canonical roll with roll_sign); rot_from_side_grasp_deg is the angle to the canonical inward side grasp (0, 0, 0)",
            },
            "orientation_convention": "palm R = Rz(yaw) Ry(-pitch) Rx(roll) (reach_specs.palm_rotation) in the heading frame; yaw 0 = fingers forward, palm normal toward the body midline (canonical inward-facing side grasp); quaternions wxyz",
            "table": {"edge_gap_m": list(cfg["bench_edge_gap_range"]), "edge_ref": cfg["bench_edge_ref"],
                      "edge_ref_note": "HERO sim_benchmark_scene.xml puts every table front edge 0.30 m ahead of the origin (= 0.18 m from the toes). The gap is measured from the foot front (toe, 0.12 m ahead of the ankle line); the paper's 10-20 cm is restricted to 16-20 cm because with the G1's resting hand (palm ~0.2 m ahead, fingertips ~0.28 m, 0.72 m high) any edge closer than 0.28 m starts the IK inside / under the slab (measured: 100 % of such standing clips blocked). A goal sampled closer than 3 cm behind the edge is shifted to edge + 0.03 m (target_x_raw / target_x_shifted).",
                      "depth_m": float(cfg["bench_table_depth"]), "half_width_m": 0.70, "thickness_m": 0.04, "in_ik_collision_avoidance": True},
            "base": {"stand_heights_m": [h for h in cfg["bench_heights"] if h >= cfg["bench_squat_below_m"]], "stand_pelvis_drop_m": list(cfg["bench_stand_drop_range"]),
                     "squat_heights_m": [h for h in cfg["bench_heights"] if h < cfg["bench_squat_below_m"]], "squat_pelvis_drop_m": list(cfg["bench_squat_drop_range"]),
                     "feet": "flat only (no heel lift / kneel / deep squat / bow)",
                     "torso_lean": f"pelvis 40 % / waist 60 % ramp (8 steps) up to {cfg['bench_stand_lean_deg_max']} deg (stand) / {cfg['bench_squat_lean_deg_max']} deg (squat), stopped at the first lean passing the geometric reach prefilter"},
            "timing": {"fps": FPS, "settle_frames": 15, "hold_s": float(cfg["bench_hold_s"]), "time_scale": 1.0, "approach": rs.BENCH_APPROACH,
                       "reach_end_frame": "first hold frame (global index, settle frames included); frames [reach_end_frame, n_frames) hold the goal"},
            "acceptance_levels": [lv.as_dict() for lv in ACCEPT_LEVELS], "fallback_allowed": allow_fallback,
            "conversion": "data_tools.npz_convert.convert_clip (production ik_reach_v2 path: S4 family, FK on the Dex3 scene, no legs_default_standing)",
            "source_tag": "height label (h050 / h074 / h088); recommended clip_end_policy 'hold' (as ik_reach_v2)",
            "license_class": LICENSE_CLASS, "clip_end_policy": CLIP_END_POLICY, "has_object": False,
            "file_naming": "<height_label>__<clip_id>.npz (group by the prefix before '__')",
        },
        "selection": stats,
        "fallback_used": any(s["fallback_used"] for s in stats.values()),
        "fallback_groups": sorted(k for k, s in stats.items() if s["fallback_used"]),
        "shortfall_groups": {k: s["shortfall"] for k, s in stats.items() if s["shortfall"]},
        "borrow_groups": {k: s["borrowed_from"] for k, s in stats.items() if s.get("n_borrowed")},
        "per_height": per_height,
        "orient_families": sorted(per_family) if per_family else None,
        "family_quota_per_height_hand": family_quota,
        "per_family": per_family,
        "per_family_height": per_family_height,
        "verification": verification,
        "clips": list(clips),
    }


def write_report(manifest: dict[str, Any], path: Path, *, conversion_errors: Sequence[dict[str, Any]] = ()) -> None:
    clips = manifest["clips"]
    stats = manifest["selection"]
    L: list[str] = []
    fam_mode = bool(manifest.get("per_family"))
    of = manifest["protocol"].get("orientation_families") or {}
    L.append(f"# BENCH_REPORT — `{manifest.get('schema', BENCH_SCHEMA)}`")
    L.append("")
    L.append(f"Generated {manifest['created_utc']} on `{manifest['host']}` (git `{str(manifest['git_sha'])[:12]}`) from bank `{manifest['bank_dir']}` "
             f"(generator {manifest['bank'].get('generator_version')}, profile `{manifest['bank'].get('profile')}`, seed {manifest['bank'].get('seed')}, "
             f"{manifest['bank'].get('n_written')} / {manifest['bank'].get('n_requested')} candidates written, tiers {manifest['bank'].get('tier_counts')}).")
    L.append("")
    if fam_mode:
        fams = of.get("families", {})
        orient_txt = ("grasp-orientation families " + "; ".join(
            f"`{k}` {100 * float(v.get('share', 0)):.0f} % (yaw {v.get('yaw_deg')}, pitch {v.get('pitch_deg')}, roll {'±' if v.get('roll_sign_random') else ''}{v.get('roll_deg')} deg, "
            f"canonical {v.get('canonical_ypr_deg')}, clearance margin {v.get('clearance_margin_m')} m)" for k, v in fams.items())
            + f"; hard rejects |roll| > {of.get('hard_rejects', {}).get('max_abs_roll_deg')} deg or rotation from the family canonical > "
            f"{of.get('hard_rejects', {}).get('max_rot_from_canonical_deg')} deg (resampled); quota per (height, hand) {manifest.get('family_quota_per_height_hand')}, "
            f"family borrow {'allowed' if of.get('family_borrow_allowed') else 'off'}")
    else:
        orient_txt = f"grasp yaw in {manifest['protocol']['grasp_yaw_deg']} deg"
    L.append(f"**Protocol**: {manifest['protocol']['paper']}. Table heights {manifest['heights_m']} m x {manifest['goals_per_height_target']} goals "
             f"({manifest['per_hand_target']} right + {manifest['per_hand_target']} left); goal 5-15 cm above the table top (hand-envelope clearance raise recorded), "
             f"{orient_txt}, table edge {manifest['protocol']['table']['edge_gap_m']} m ahead of the {manifest['protocol']['table']['edge_ref']}, "
             f"hold {manifest['protocol']['timing']['hold_s']} s, fps {FPS}.")
    L.append("")
    L.append(f"**Result**: {manifest['n_clips']} clips; fallback used: **{manifest['fallback_used']}**"
             + (f" in {manifest['fallback_groups']}" if manifest["fallback_used"] else "")
             + (f"; family borrow {manifest['borrow_groups']}" if manifest.get("borrow_groups") else "")
             + (f"; SHORTFALL {manifest['shortfall_groups']}" if manifest["shortfall_groups"] else "; no shortfall") + ".")
    L.append("")
    L.append("## Selection per table height / " + ("orientation family / " if fam_mode else "") + "hand")
    L.append("")
    fam_col = " family |" if fam_mode else ""
    L.append(f"| height |{fam_col} hand | quota | candidates | strict (core, <=1 cm) | core (<=1.5 cm) | recoverable (<=3 cm) | never | selected | of which fallback | borrowed | max level | shortfall |")
    L.append("|---|" + ("---|" if fam_mode else "") + "---|---|---|---|---|---|---|---|---|---|---|---|")
    for key, s in sorted(stats.items()):
        nb = s["n_by_level"]
        fam_cell = f" {s.get('orient_family')} |" if fam_mode else ""
        L.append(f"| {s['height_m']:.2f} m ({s['height_label']}) |{fam_cell} {s['hand']} | {s.get('quota', s['per_hand_target'])} | {s['n_candidates']} | {nb.get('strict', 0)} | {nb.get('core', 0)} | {nb.get('recoverable', 0)} | "
                 f"{s['n_never_accepted']} | {s['n_selected']} | {s['n_selected'] - s['n_selected_by_level'].get('strict', 0) - s.get('n_borrowed', 0)} | "
                 f"{s.get('n_borrowed', 0)}{(' (' + ', '.join(f'{k} {v}' for k, v in s['borrowed_from'].items()) + ')') if s.get('borrowed_from') else ''} | {s['max_level_used']} | {s['shortfall']} |")
    L.append("")
    L.append("### Why candidates miss the strict level (per group; a clip may count in several rows)")
    L.append("")
    for key, s in sorted(stats.items()):
        L.append(f"- `{key}`: " + (", ".join(f"{r} x{n}" for r, n in s["strict_reject_reasons"].items()) or "none"))
    L.append("")
    L.append("## Per height (selected clips)")
    L.append("")
    L.append("| height | clips (R/L) | terminal EE pos err mm mean / p50 / p95 / max | EE err at reach end mm mean / max | reach duration s mean / min / max | clip duration s mean / max | pelvis height min m mean / min | target z above table cm min / mean / max | clearance-raised | table edge x m min / max | torso lean deg (pelvis+waist) mean / max |")
    L.append("|---|---|---|---|---|---|---|---|---|---|---|")
    for lab in manifest["height_labels"]:
        cs = [c for c in clips if c["height_label"] == lab]
        if not cs:
            L.append(f"| {lab} | 0 | | | | | | | | | |")
            continue
        e = _stats([c["terminal_ee_pos_err_m"] * 1e3 for c in cs])
        er = _stats([c["ee_pos_err_at_reach_end_m"] * 1e3 for c in cs])
        rd = _stats([c["reach_frames"] / FPS for c in cs])
        du = _stats([c["duration_s"] for c in cs])
        ph = _stats([c["pelvis_height_min_m"] for c in cs])
        tz = _stats([(c["target_pos_w"][2] - c["table_top_z"]) * 1e2 for c in cs])
        ex = _stats([c["table_edge_x"] for c in cs])
        ln = _stats([c["pelvis_pitch_deg"] + c["waist_pitch_deg"] for c in cs])
        nr = sum(1 for c in cs if c["hand"] == "right")
        L.append(f"| {cs[0]['height_m']:.2f} m ({lab}) | {len(cs)} ({nr}/{len(cs) - nr}) | {e['mean']:.2f} / {e['p50']:.2f} / {e['p95']:.2f} / {e['max']:.2f} | "
                 f"{er.get('mean', float('nan')):.2f} / {er.get('max', float('nan')):.2f} | {rd['mean']:.2f} / {rd['min']:.2f} / {rd['max']:.2f} | {du['mean']:.2f} / {du['max']:.2f} | "
                 f"{ph['mean']:.3f} / {ph['min']:.3f} | {tz['min']:.1f} / {tz['mean']:.1f} / {tz['max']:.1f} | {sum(1 for c in cs if c['clearance_raised'])} | "
                 f"{ex['min']:.3f} / {ex['max']:.3f} | {ln['mean']:.1f} / {ln['max']:.1f} |")
    L.append("")
    all_e = _stats([c["terminal_ee_pos_err_m"] * 1e3 for c in clips])
    if all_e["n"]:
        L.append(f"All clips: terminal EE position error mean {all_e['mean']:.2f} mm, p95 {all_e['p95']:.2f} mm, max {all_e['max']:.2f} mm; "
                 f"total {sum(c['n_frames'] for c in clips)} frames = {sum(c['n_frames'] for c in clips) / FPS / 60:.1f} min of reference motion.")
        L.append("")
    if fam_mode:
        def fam_row(name: str, g: dict[str, Any]) -> str:
            e = g["terminal_ee_pos_err_m"]; eo = g["terminal_ee_ori_err_deg"]; rd = g["reach_duration_s"]; rc = g["rot_from_canonical_deg"]
            rsg = g["rot_from_side_grasp_deg"]; tz = g["target_z_above_table_m"]; ar = g["abs_roll_deg"]; pt = g["pitch_deg"]
            def f3(st, keys, scale=1.0, fmt="{:.2f}"):
                return " / ".join(fmt.format(st[k] * scale) if st.get("n") else "-" for k in keys)
            return (f"| {name} | {g.get('quota_total', '-')} | {g.get('n_candidates', '-')} | {g.get('n_strict_candidates', '-')} | {g.get('n_slots_filled', '-')} | {g.get('shortfall', '-')} | "
                    f"{g['n_clips']} ({g['right']}/{g['left']}) | {g['n_fallback']} | {g['n_borrowed']} | {f3(e, ('mean', 'p50', 'p95', 'max'), 1e3)} | {f3(eo, ('mean', 'max'))} | "
                    f"{f3(rd, ('mean', 'min', 'max'))} | {f3(rc, ('mean', 'p95', 'max'), 1.0, '{:.1f}')} | {f3(rsg, ('mean', 'max'), 1.0, '{:.1f}')} | "
                    f"{f3(pt, ('min', 'mean', 'max'), 1.0, '{:.1f}')} | {f3(ar, ('mean', 'max'), 1.0, '{:.1f}')} | {f3(tz, ('min', 'mean', 'max'), 1e2, '{:.1f}')} |")

        head = ("| {} | quota | candidates | strict cand. | slots filled | shortfall | clips (R/L) | fallback | borrowed (this family's clips lent) | terminal EE pos err mm mean / p50 / p95 / max | "
                "EE ori err deg mean / max | reach duration s mean / min / max | rot from canonical deg mean / p95 / max | rot from side grasp deg mean / max | "
                "pitch deg min / mean / max | |roll| deg mean / max | target z above table cm min / mean / max |")
        sep = "|---|" + "---|" * 17
        L.append("## Per orientation family (selected clips, all heights)")
        L.append("")
        L.append(head.format("family"))
        L.append(sep)
        for fam, g in manifest["per_family"].items():
            L.append(fam_row(fam, g))
        L.append("")
        L.append("Slots = (height, family, hand) selection slots of the family (borrowed clips fill another family's slots but are counted under their own family in `clips`).")
        L.append("")
        L.append("## Family x height (selected clips)")
        L.append("")
        L.append(head.format("family / height"))
        L.append(sep)
        for key, g in manifest["per_family_height"].items():
            L.append(fam_row(key, g))
        L.append("")
    v = manifest["verification"]
    L.append("## Loader contract")
    L.append("")
    L.append(f"- verifier `{v.get('verifier')}`: {v.get('n_loaded')} / {v.get('n_files')} files loaded, {v.get('n_skipped')} skipped; source tags {v.get('source_tags')}."
             + (f" Notes: {v['notes']}" if v.get("notes") else ""))
    if conversion_errors:
        L.append(f"- conversion errors ({len(conversion_errors)}): " + "; ".join(f"{Path(e['src']).name}: {e.get('error')}" for e in conversion_errors))
    L.append("")
    L.append("## Notes / deviations from the paper protocol")
    L.append("")
    L.append(f"- Table-edge distance measured from the foot front (see protocol.table.edge_ref_note): HERO's own scene has the edge 0.30 m ahead of the origin; measuring 10-20 cm from the pelvis would put the 0.50 m slab into the squatting thighs.")
    L.append(f"- Goal x in {manifest['protocol']['target_x_range_m']} m ahead of the pelvis (HERO's CSV also lists x up to 0.90 m, unreachable without stepping; the tracker reference keeps a fixed stance).")
    L.append(f"- Torso lean is chosen per goal as the smallest lean passing the geometric reach prefilter ({manifest['protocol']['base']['torso_lean']}).")
    L.append(f"- Acceptance strict = tier core & terminal EE position error <= 1 cm; fallback levels: {[lv.name for lv in ACCEPT_LEVELS[1:]]} (used: {manifest['fallback_used']}).")
    if fam_mode:
        L.append(f"- Orientation families are selected per (height, family, hand) with the quota {manifest.get('family_quota_per_height_hand')}; a short group first relaxes the level "
                 f"inside the family, then ({'allowed' if of.get('family_borrow_allowed') else 'OFF'}) borrows strict-level clips of the other families of the same height / hand "
                 f"(borrow used: {bool(manifest.get('borrow_groups'))}). `rot_from_canonical_deg` = angle to the family's canonical grasp (top_down: fingers straight down; "
                 f"tilted: side grasp rolled by the canonical 35 deg with the goal's roll sign); `rot_from_side_grasp_deg` = angle to the inward side grasp.")
        L.append(f"- top_down goals sit on the fingertip envelope: palm point >= table + 0.175 m x sin(|pitch|) (+ hand thickness) + 3.5 cm margin, i.e. ~0.21 m above the table top "
                 f"(the 5-15 cm sample is always raised; `target_z_raw` keeps the sample).")
    path.write_text("\n".join(L) + "\n")


# --------------------------------------------------------------------------------------------------
# CLI
# --------------------------------------------------------------------------------------------------
def build(bank_dir: Path, out_dir: Path, *, per_hand: int = DEFAULT_PER_HAND, jobs: int = 1, verifier: str = "auto", allow_fallback: bool = True,
          heights: Sequence[str] | None = None, clean: bool = True, family_quota: dict[str, int] | None = None, allow_family_borrow: bool = True) -> dict[str, Any]:
    bank_dir, out_dir = Path(bank_dir), Path(out_dir)
    t0 = time.time()
    bank_manifest, rows = load_bank(bank_dir)
    if family_quota is None:
        family_quota = default_family_quota(rows, per_hand)
    selected, stats = select_goals(rows, per_hand, allow_fallback=allow_fallback, heights=heights, family_quota=family_quota, allow_family_borrow=allow_family_borrow)
    if not selected:
        raise RuntimeError("no candidate passed the acceptance levels; nothing to build")
    out_dir.mkdir(parents=True, exist_ok=True)
    wanted = {output_name(r) for r in selected}
    removed: list[str] = []
    if clean:
        for p in out_dir.glob("*.npz"):
            if p.name not in wanted:
                p.unlink()
                removed.append(p.name)
    results = convert_selected(selected, bank_dir, out_dir, jobs=jobs)
    errors = [r for r in results if not r["ok"]]
    if errors:
        for e in errors:
            print(f"[build_reach_bench] CONVERT FAIL {Path(e['src']).name}: {e.get('error')}", file=sys.stderr)
        raise RuntimeError(f"{len(errors)} / {len(results)} conversions failed")
    clips = [clip_entry(r, Path(res["dst"]), res["receipt"]) for r, res in zip(selected, results)]
    clips.sort(key=lambda c: (c["height_m"], c["hand"], c["goal_index"]))
    files = [c["file"] for c in clips]
    verification = verify_corpus(out_dir, files, verifier)
    verification["removed_stale_files"] = removed
    manifest = build_manifest(bank_manifest=bank_manifest, bank_dir=bank_dir, out_dir=out_dir, clips=clips, stats=stats, verification=verification,
                              per_hand=per_hand, allow_fallback=allow_fallback, schema=bank_schema(rows), family_quota=family_quota,
                              allow_family_borrow=allow_family_borrow)
    manifest["build_wall_s"] = time.time() - t0
    (out_dir / "BENCH_MANIFEST.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    write_report(manifest, out_dir / "BENCH_REPORT.md", conversion_errors=errors)
    return manifest


def main(argv: Sequence[str] | None = None) -> int:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--bank-dir", required=True, help="generator output (clips/, labels/, manifest.json) of --profile hero_bench_v1")
    ap.add_argument("--out-dir", required=True, help="flat corpus dir: <height_label>__<clip_id>.npz + BENCH_MANIFEST.json + BENCH_REPORT.md")
    ap.add_argument("--per-hand", type=int, default=DEFAULT_PER_HAND, help="goals per (height, hand) (default 30 -> 60 per height)")
    ap.add_argument("--jobs", type=int, default=1)
    ap.add_argument("--verifier", default="auto", choices=["auto", "hero", "holosoma", "schema"])
    ap.add_argument("--no-fallback", action="store_true", help="strict acceptance only (report the shortfall instead of relaxing)")
    ap.add_argument("--heights", default=None, help="comma-separated height labels to build (default: all in the bank), e.g. h050,h074")
    ap.add_argument("--no-clean", action="store_true", help="keep stale *.npz in --out-dir (default: remove files not produced by this build)")
    ap.add_argument("--family-quota", default=None, help="hero_bench_v2: goals per (height, family, hand), e.g. fan_side=50,top_down=25,tilted=25 (default: profile share x --per-hand)")
    ap.add_argument("--no-family-borrow", action="store_true", help="hero_bench_v2: never fill a short family group with strict clips of the other families (report the shortfall)")
    args = ap.parse_args(argv)
    heights = [h.strip() for h in args.heights.split(",") if h.strip()] if args.heights else None
    manifest = build(Path(args.bank_dir), Path(args.out_dir), per_hand=args.per_hand, jobs=args.jobs, verifier=args.verifier,
                     allow_fallback=not args.no_fallback, heights=heights, clean=not args.no_clean, family_quota=parse_family_quota(args.family_quota),
                     allow_family_borrow=not args.no_family_borrow)
    summary = {k: manifest[k] for k in ("schema", "n_clips", "heights_m", "fallback_used", "fallback_groups", "shortfall_groups", "borrow_groups")}
    summary["per_height"] = {k: {kk: v[kk] for kk in ("n_clips", "right", "left", "n_fallback")} for k, v in manifest["per_height"].items()}
    if manifest.get("per_family"):
        summary["per_family"] = {k: {kk: v[kk] for kk in ("n_clips", "right", "left", "n_candidates", "n_fallback", "n_borrowed", "shortfall")} for k, v in manifest["per_family"].items()}
        summary["per_family_height"] = {k: {kk: v[kk] for kk in ("n_clips", "right", "left", "n_candidates", "shortfall")} for k, v in manifest["per_family_height"].items()}
    summary["verification"] = {k: manifest["verification"].get(k) for k in ("verifier", "n_files", "n_loaded", "n_skipped")}
    summary["out_dir"] = str(Path(args.out_dir).resolve())
    print(json.dumps(summary, indent=1))
    return 0 if not manifest["shortfall_groups"] else 3


if __name__ == "__main__":
    raise SystemExit(main())
