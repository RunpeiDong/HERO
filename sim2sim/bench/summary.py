"""Per-table-height / per-layer open-loop benchmark summary (hero_bench_v1) from ``hero_eval_series_v1`` files -- the paper's numbers.

    python -m sim2sim.bench.summary --series <run>/series.npz [...] --bench-manifest <bench>/BENCH_MANIFEST.json --out <dir> \\
        [--labels a,b] [--final-s 1.0] [--group-by none|orient_family|stratum|tier] [--title ...]

Inputs
  --series S [S ...]         one or more series npz (the MuJoCo runner and / or the Isaac Sim fixed-horizon evaluator); each is
                             labelled by its ``sim`` field (``--labels`` overrides)
  --bench-manifest M.json    BENCH_MANIFEST.json of the corpus: per clip ``file, clip_id, height_m, height_label, hand, goal_index,
                             target_pos_w, reach_end_frame, hold_frames, n_frames, fps`` (a list, a dict with a ``clips`` list, or a dict
                             keyed by stem)
  --out DIR                  writes ``bench_summary.json`` / ``.md`` / ``.csv``
  --group-by F               optional extra grouping: ``orient_family`` (hero_bench_v2 rows), ``stratum`` (benchmark layer) or ``tier`` (benchmark tier);
                             the same statistics are emitted per group (all heights, per hand) and per group x height

Matching.  Series rows and manifest clips are joined on the file stem ``<source_tag>__<clip_id>`` (``Path(file).stem``;
fallback ``<height_label>__<clip_id>``).  Height group = manifest ``height_label`` (else the series ``source_tag``);
hand = manifest ``hand`` (``left`` / ``right``; anything else -> ``both``).

Hold window.  Series column ``j`` is the measurement of control step ``j + 1``, i.e. one step after frame ``j``.  Hold columns of a clip =
``[reach_end_frame, min(clip_len_steps, T)) ∩ [0, valid_until)`` -- the reach is complete and the reference is held -- bounded by the row's
``hold_end_frame`` when the manifest has it (fallback ``reach_end_frame + hold_frames``; rows with neither keep the unbounded end), so a
retract clip's own retract frames never enter the hold metrics; rows with ``retract_start_frame`` also get a ``rest`` window (the last
``--final-s`` of the clip = the rest pose) in ``rest_mean`` / group ``rest``.  "final" = the last ``--final-s`` seconds (default 1.0 s) of the
hold window (the whole window when shorter).

Statistics (per sim, per height in {labels..., all}, per hand in {left, right, all}).  For every metric in :data:`BENCH_METRICS` the
per-clip hold-window mean is reduced across clips to mean, std (population, ddof 0), p50 / p80 / p90 (the CDF points of the paper appendix),
and the mean / std of the per-clip "final" means.  Also ``fail_free_frac`` (no training-termination failure inside the clip's valid window:
``first_fail_step < 0`` or ``> valid_until``), ``reach_fail_frac`` (failure at or before ``reach_end_frame``) and ``n_clips``.  The ``alive``
block repeats the metric statistics over the fail-free clips only.  Active-hand metrics (``*_active_*``) use the per-hand column of the
manifest hand (``ee_global_left_cm`` for a left-hand clip; the two-hand mean for ``both``).

Layer / tier groups and success columns (new keys only; every legacy number unchanged): independent of ``--group-by`` every sim summary carries
``bench_groups`` = ``cell:<stratum>/<height>/<hand>``, ``stratum:<s>``, ``tier:<t>`` group statistics (rows without a stratum field -- legacy manifests -- fall back to ``stratum =
height_label``, ``tier = "core"``; the input of :mod:`sim2sim.bench.report`); every group carries ``success``: the open-loop success fractions
(``S7.5`` = fail-free and hold-mean ACTIVE-hand world-frame palm error <= 7.5 cm and world-frame rotation error <= 15 deg; ``S5`` = <= 5 cm and
<= 10 deg) with Wilson 95 % intervals (denominator = every clip of the group), the CDF points of the active-hand hold mean at 2.5 / 5 / 10 cm and
the layer's IK residual mean (manifest ``terminal_ee_pos_err_m``).  Thresholds / CDF points come from the manifest ``protocol.success`` block
when present (:func:`success_protocol`), else :data:`DEFAULT_SUCCESS`; the headline column is the ACTIVE hand's world-frame error
``ee_global_active_cm`` (``ee_rot_global_active_deg`` = its world-frame rotation), the both-hands mean ``ee_global_cm`` is the secondary column.

Odometry (series with ``odom_xy_err_cm`` / ``odom_yaw_err_deg`` / ``odom_z_err_cm``, :data:`ODOM_METRICS`): every clip record gets an
``odometry`` block (hold-window mean |error| and the value at the clip end), every group an ``odometry`` statistics block and the markdown an
extra table.  Series without the columns produce exactly the previous outputs.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import sys
from collections.abc import Mapping, Sequence
from pathlib import Path
from typing import Any

import numpy as np

from sim2sim.bench.series import read_series

SCHEMA = "hero_bench_summary_v1"
BENCH_METRICS: tuple[str, ...] = ("ee_global_cm", "ee_local_cm", "ee_rot_deg", "joint_upper_rad", "joint_upper17_rad", "anchor_xy_cm")
ACTIVE_METRICS: dict[str, tuple[str, str, str]] = {  # active-hand metric -> (left column, right column, two-hand column)
    "ee_global_active_cm": ("ee_global_left_cm", "ee_global_right_cm", "ee_global_cm"),
    "ee_local_active_cm": ("ee_local_left_cm", "ee_local_right_cm", "ee_local_cm"),
    "ee_rot_active_deg": ("ee_rot_left_deg", "ee_rot_right_deg", "ee_rot_deg"),
    "ee_rot_global_active_deg": ("ee_rot_global_left_deg", "ee_rot_global_right_deg", "ee_rot_global_deg"),   # world-frame rotation (success gate)
}
ALL_METRICS: tuple[str, ...] = BENCH_METRICS + tuple(ACTIVE_METRICS)
#: hero_bench_v1 success protocol defaults; a manifest ``protocol.success`` block overrides
DEFAULT_SUCCESS: dict[str, Any] = {
    "open_loop": {"S7.5": {"pos_cm": 7.5, "rot_deg": 15.0}, "S5": {"pos_cm": 5.0, "rot_deg": 10.0}},
    "closed_loop": {"C3": {"pos_cm": 3.0, "rot_deg": 15.0, "window": "final"}},
    "cdf_points_cm": [2.5, 5.0, 10.0],
    "stayed_threshold_cm": 1.75,
    "ci": "wilson95",
}
WILSON_Z95: float = 1.959963984540054
#: grouping fields with their legacy fallbacks (``stratum`` -> height_label, ``tier`` -> core)
BENCH_GROUP_FIELDS: dict[str, str] = {"stratum": "height_label", "tier": "core"}
#: Odometry error columns (a subset of ``sim2sim.bench.metrics.ODOM_METRIC_KEYS``; ``--odom so`` runs only) -- optional extra block
ODOM_METRICS: tuple[str, ...] = ("odom_xy_err_cm", "odom_yaw_err_deg", "odom_z_err_cm")
PERCENTILES: tuple[int, ...] = (50, 80, 90)
HANDS: tuple[str, ...] = ("left", "right", "all")
ALL = "all"
GROUP_BY_CHOICES: tuple[str, ...] = ("none", "orient_family", "stratum", "tier")
PAPER_COLUMNS = (("ee_global_cm", "Trans (cm)"), ("ee_rot_deg", "Orient (deg)"), ("joint_upper_rad", "Joint (rad)"))


# ==================================================================================================
# manifest
# ==================================================================================================
def _stem(name: str) -> str:
    base = os.path.basename(str(name))
    return base[:-4] if base.endswith(".npz") else base


def manifest_key(entry: Mapping[str, Any]) -> str:
    """Join key of a manifest clip: ``Path(file).stem``, else ``<height_label>__<clip_id>``."""
    if entry.get("file"):
        return _stem(str(entry["file"]))
    return f"{entry.get('height_label', 'unknown')}__{entry.get('clip_id', '')}"


def load_manifest(path: str | Path) -> dict[str, dict[str, Any]]:
    """``{stem: entry}`` from BENCH_MANIFEST.json (list / ``{"clips": [...]}`` / ``{stem: entry}``)."""
    obj = json.loads(Path(path).read_text())
    return manifest_entries(obj)


def load_manifest_protocol(path: str | Path) -> dict[str, Any]:
    """The manifest's top-level ``protocol`` block (``{}`` for a bare list / a manifest without one)."""
    obj = json.loads(Path(path).read_text())
    proto = obj.get("protocol") if isinstance(obj, dict) else None
    return dict(proto) if isinstance(proto, dict) else {}


def _threshold_pair(block: Any, pos_default: float, rot_default: float) -> dict[str, float]:
    """``{"pos_cm", "rot_deg"}`` from a manifest success entry (accepts ``pos_cm | max_pos_cm | pos_m`` and ``rot_deg | max_rot_deg``)."""
    out = {"pos_cm": float(pos_default), "rot_deg": float(rot_default)}
    if isinstance(block, Mapping):
        if block.get("pos_cm") is not None:
            out["pos_cm"] = float(block["pos_cm"])
        elif block.get("max_pos_cm") is not None:
            out["pos_cm"] = float(block["max_pos_cm"])
        elif block.get("pos_m") is not None:
            out["pos_cm"] = 100.0 * float(block["pos_m"])
        if block.get("rot_deg") is not None:
            out["rot_deg"] = float(block["rot_deg"])
        elif block.get("max_rot_deg") is not None:
            out["rot_deg"] = float(block["max_rot_deg"])
        for k, v in block.items():
            if k not in ("pos_cm", "max_pos_cm", "pos_m", "rot_deg", "max_rot_deg"):
                out[k] = v
    elif isinstance(block, (list, tuple)) and len(block) == 2:
        out["pos_cm"], out["rot_deg"] = float(block[0]), float(block[1])
    return out


def success_protocol(protocol: Mapping[str, Any] | None) -> dict[str, Any]:
    """The success thresholds in force: manifest ``protocol.success`` (when present; levels may be given as ``{"S7.5": {"pos_cm", "rot_deg"}}``,
    ``[pos_cm, rot_deg]`` or the alternative key names of :func:`_threshold_pair`) layered over :data:`DEFAULT_SUCCESS`; ``source`` records which."""
    base = json.loads(json.dumps(DEFAULT_SUCCESS))
    succ = (protocol or {}).get("success") if isinstance(protocol, Mapping) else None
    out = dict(base)
    out["source"] = "default"
    if isinstance(succ, Mapping) and succ:
        out["source"] = "manifest"
        for side in ("open_loop", "closed_loop"):
            levels = succ.get(side)
            if isinstance(levels, Mapping) and levels:
                out[side] = {str(name): _threshold_pair(blk, *(base[side].get(str(name), {"pos_cm": float("nan"), "rot_deg": float("nan")}).get(k) for k in ("pos_cm", "rot_deg")))
                             for name, blk in levels.items()}
        if succ.get("cdf_points_cm") is not None:
            out["cdf_points_cm"] = [float(v) for v in succ["cdf_points_cm"]]
        if succ.get("stayed_threshold_cm") is not None:
            out["stayed_threshold_cm"] = float(succ["stayed_threshold_cm"])
        for k, v in succ.items():
            if k not in out:
                out[k] = v
    return out


def wilson_ci(k: int, n: int, z: float = WILSON_Z95) -> tuple[float, float]:
    """Wilson score interval of ``k / n`` (NaN pair for ``n == 0``)."""
    if n <= 0:
        return float("nan"), float("nan")
    p = float(k) / float(n)
    z2 = z * z
    denom = 1.0 + z2 / n
    centre = (p + z2 / (2.0 * n)) / denom
    half = z * math.sqrt(p * (1.0 - p) / n + z2 / (4.0 * n * n)) / denom
    lo, hi = max(0.0, centre - half), min(1.0, centre + half)
    return (0.0 if k <= 0 else lo), (1.0 if k >= n else hi)   # exact ends (no 0.9999999999999999 from the float arithmetic)


def bench_group_value(entry: Mapping[str, Any], field: str) -> str | None:
    """Value of a grouping field of a manifest row with the legacy fallbacks of :data:`BENCH_GROUP_FIELDS` (``stratum`` -> ``height_label``,
    ``tier`` -> ``"core"``); other fields (``orient_family``) have no fallback -> None when absent."""
    v = entry.get(field)
    if v not in (None, ""):
        return str(v)
    fb = BENCH_GROUP_FIELDS.get(field)
    if fb is None:
        return None
    if field == "stratum":
        v = entry.get(fb) or entry.get("source_tag")
        return str(v) if v not in (None, "") else None
    return str(fb)


def manifest_entries(obj: Any) -> dict[str, dict[str, Any]]:
    if isinstance(obj, dict) and isinstance(obj.get("clips"), list):
        entries = obj["clips"]
    elif isinstance(obj, list):
        entries = obj
    elif isinstance(obj, dict):
        entries = [dict(v, **({"file": k} if "file" not in v and "clip_id" not in v else {})) for k, v in obj.items() if isinstance(v, dict)]
    else:
        raise ValueError("unrecognised manifest layout (expected a list, {'clips': [...]} or {stem: entry})")
    out: dict[str, dict[str, Any]] = {}
    for e in entries:
        out[manifest_key(e)] = dict(e)
    return out


def hand_of(entry: Mapping[str, Any]) -> str:
    h = str(entry.get("hand", "")).lower()
    return h if h in ("left", "right") else "both"


def reach_end_frame_of(entry: Mapping[str, Any]) -> int:
    if entry.get("reach_end_frame") is not None:
        return int(entry["reach_end_frame"])
    if entry.get("n_frames") is not None and entry.get("hold_frames") is not None:
        return max(int(entry["n_frames"]) - int(entry["hold_frames"]), 0)
    return 0


def hold_end_frame_of(entry: Mapping[str, Any]) -> int | None:
    """End (exclusive) of the hold window: the row's ``hold_end_frame`` (hero_bench_v1), else ``reach_end_frame + hold_frames``; None for
    rows that carry neither (old manifests: the window runs to the clip end as before)."""
    if entry.get("hold_end_frame") is not None:
        return int(entry["hold_end_frame"])
    if entry.get("hold_frames") is not None and (entry.get("reach_end_frame") is not None or entry.get("n_frames") is not None):
        return reach_end_frame_of(entry) + int(entry["hold_frames"])
    return None


# ==================================================================================================
# per-clip hold-window reduction
# ==================================================================================================
def hold_columns(reach_end_frame: int, clip_len_steps: int, valid_until: int, horizon_steps: int, hold_end_frame: int | None = None) -> tuple[int, int]:
    """``[start, stop)`` column range of the hold window (empty when ``start >= stop``); ``hold_end_frame`` (hero_bench_v1) bounds the end
    so a clip's own retract frames stay out of the hold metrics (None = the old unbounded end)."""
    start = max(int(reach_end_frame), 0)
    stop = min(int(clip_len_steps), int(horizon_steps), int(valid_until))
    if hold_end_frame is not None:
        stop = min(stop, int(hold_end_frame))
    return start, max(stop, start)


def final_columns(start: int, stop: int, dt: float, final_s: float) -> tuple[int, int]:
    """Last ``final_s`` seconds of ``[start, stop)`` (the whole window when shorter)."""
    n = max(int(round(final_s / dt)), 1)
    return max(stop - n, start), stop


def _nanmean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


def clip_metric_series(metrics: Mapping[str, np.ndarray], row: int, hand: str) -> dict[str, np.ndarray]:
    """``{metric: (T,)}`` for one series row incl. the active-hand metrics."""
    out: dict[str, np.ndarray] = {}
    for m in BENCH_METRICS:
        arr = metrics.get(m)
        out[m] = arr[row] if arr is not None else None
    for m, (left, right, both) in ACTIVE_METRICS.items():
        src = left if hand == "left" else right if hand == "right" else both
        arr = metrics.get(src)
        out[m] = arr[row] if arr is not None else None
    return out


def clip_success(rec: Mapping[str, Any], success: Mapping[str, Any]) -> dict[str, bool]:
    """Open-loop success flags of one clip record: fail-free AND hold-mean active-hand world-frame palm error <= pos AND world-frame rotation
    error <= rot for every level of ``success["open_loop"]`` (a NaN metric never passes)."""
    pos = rec["hold_mean"].get("ee_global_active_cm", float("nan"))
    rot = rec["hold_mean"].get("ee_rot_global_active_deg", float("nan"))
    out: dict[str, bool] = {}
    for name, thr in success.get("open_loop", {}).items():
        ok = bool(rec["fail_free"]) and math.isfinite(pos) and math.isfinite(rot) and pos <= float(thr["pos_cm"]) and rot <= float(thr["rot_deg"])
        out[str(name)] = ok
    return out


def per_clip_table(series: Mapping[str, Any], manifest: Mapping[str, Mapping[str, Any]], final_s: float, group_by: str | None = None,
                   success: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Join one series with the manifest -> ``{"clips": {stem: record}, "unmatched_rows": [...], "manifest_missing": [...]}``.
    ``group_by`` (e.g. ``orient_family`` / ``stratum`` / ``tier``) copies that manifest field (:func:`bench_group_value`) into every record
    (``None`` when the row lacks it and no fallback exists).  ``success`` = :func:`success_protocol` block (default thresholds when None)."""
    T = int(series["horizon_steps"])
    dt = float(series["dt"])
    metrics = series["metrics"]
    success = success if success is not None else success_protocol(None)
    clips: dict[str, Any] = {}
    unmatched: list[str] = []
    for i, name in enumerate(series["clip_name"]):
        stem = _stem(name)
        entry = manifest.get(stem)
        if entry is None:
            unmatched.append(stem)
            continue
        hand = hand_of(entry)
        reach_end = reach_end_frame_of(entry)
        hold_end = hold_end_frame_of(entry)
        vu = int(series["valid_until"][i])
        clen = int(series["clip_len_steps"][i])
        start, stop = hold_columns(reach_end, clen, vu, T, hold_end)
        fstart, fstop = final_columns(start, stop, dt, final_s)
        ff = int(series["first_fail_step"][i])
        failed_in_window = 1 <= ff <= vu
        ik_res = entry.get("terminal_ee_pos_err_m")
        rec: dict[str, Any] = {
            "height": str(entry.get("height_label") or series["source_tag"][i]),
            "height_m": entry.get("height_m"),
            "hand": hand,
            "goal_index": entry.get("goal_index"),
            "group": (bench_group_value(entry, group_by) if group_by else None),
            "stratum": bench_group_value(entry, "stratum"),
            "tier": bench_group_value(entry, "tier"),
            "reach_end_frame": reach_end,
            "hold_end_frame": hold_end,
            "clip_len_steps": clen,
            "valid_until": vu,
            "hold_cols": [start, stop],
            "final_cols": [fstart, fstop],
            "first_fail_step": ff,
            "first_fail_cause": str(series["first_fail_cause"][i]),
            "fail_free": not failed_in_window,
            "reach_fail": bool(1 <= ff <= reach_end),
            "ik_residual_cm": (100.0 * float(ik_res) if ik_res is not None and math.isfinite(float(ik_res)) else None),
            "hold_mean": {},
            "final_mean": {},
        }
        rest_cols = None
        if entry.get("retract_start_frame") is not None:   # retract layer: the rest window = the last final_s of the clip (hand back at rest)
            rs_, re_ = int(entry["retract_start_frame"]), min(clen, T, vu)
            if re_ > rs_:
                rest_cols = final_columns(rs_, re_, dt, final_s)
                rec["rest_cols"] = [int(rest_cols[0]), int(rest_cols[1])]
                rec["rest_mean"] = {}
        for m, arr in clip_metric_series(metrics, i, hand).items():
            if arr is None or stop <= start:
                rec["hold_mean"][m] = float("nan")
                rec["final_mean"][m] = float("nan")
            else:
                rec["hold_mean"][m] = _nanmean(arr[start:stop])
                rec["final_mean"][m] = _nanmean(arr[fstart:fstop])
            if rest_cols is not None:
                rec["rest_mean"][m] = _nanmean(arr[rest_cols[0]:rest_cols[1]]) if arr is not None else float("nan")
        rec["success"] = clip_success(rec, success)
        odom_keys = [m for m in ODOM_METRICS if metrics.get(m) is not None]
        if odom_keys:  # --odom so series: |error| over the hold window and the value at the clip end (last valid column)
            end_col = min(clen, T, vu) - 1
            rec["odometry"] = {"hold_mean_abs": {}, "end": {}}
            for m in odom_keys:
                arr = np.asarray(metrics[m][i], dtype=np.float64)
                rec["odometry"]["hold_mean_abs"][m] = _nanmean(np.abs(arr[start:stop])) if stop > start else float("nan")
                rec["odometry"]["end"][m] = float(arr[end_col]) if 0 <= end_col < arr.shape[0] and math.isfinite(float(arr[end_col])) else float("nan")
        clips[stem] = rec
    missing = sorted(set(manifest) - set(clips))
    return {"clips": clips, "unmatched_rows": unmatched, "manifest_missing": missing}


# ==================================================================================================
# group statistics
# ==================================================================================================
def _stats(values: Sequence[float], finals: Sequence[float]) -> dict[str, float | int]:
    v = np.asarray([x for x in values if math.isfinite(x)], dtype=np.float64)
    f = np.asarray([x for x in finals if math.isfinite(x)], dtype=np.float64)
    out: dict[str, float | int] = {"n": int(v.size)}
    if v.size == 0:
        out.update({"mean": float("nan"), "std": float("nan"), **{f"p{p}": float("nan") for p in PERCENTILES}})
    else:
        out.update({"mean": float(v.mean()), "std": float(v.std(ddof=0)), **{f"p{p}": float(np.percentile(v, p)) for p in PERCENTILES}})
    out["final_mean"] = float(f.mean()) if f.size else float("nan")
    out["final_std"] = float(f.std(ddof=0)) if f.size else float("nan")
    return out


def success_stats(recs: Sequence[Mapping[str, Any]], success: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Group success block: per open-loop level ``k / n`` (denominator = every clip of the group), the fraction and its Wilson 95 % interval;
    ``cdf_cm`` = fraction of clips whose hold-mean ACTIVE-hand world error is <= each CDF point (failed / NaN clips count in the denominator,
    never as passes); ``ik_residual_cm`` = mean of the rows' ``terminal_ee_pos_err_m`` (manifest IK residual)."""
    success = success if success is not None else success_protocol(None)
    n = len(recs)
    out: dict[str, Any] = {"n": n, "levels": {}, "cdf_cm": {}, "ik_residual_cm": {"mean": float("nan"), "n": 0}}
    levels = list(success.get("open_loop", {}).keys())
    if not levels and recs:
        levels = list(recs[0].get("success", {}).keys())
    for name in levels:
        k = sum(1 for r in recs if r.get("success", {}).get(name))
        lo, hi = wilson_ci(k, n)
        thr = success.get("open_loop", {}).get(name, {})
        out["levels"][name] = {"k": int(k), "n": n, "frac": (k / n if n else float("nan")), "ci95": [lo, hi],
                               "pos_cm": thr.get("pos_cm"), "rot_deg": thr.get("rot_deg")}
    vals = [r["hold_mean"].get("ee_global_active_cm", float("nan")) for r in recs]
    for x in success.get("cdf_points_cm", []):
        key = f"{float(x):g}"
        out["cdf_cm"][key] = (sum(1 for v in vals if math.isfinite(v) and v <= float(x)) / n) if n else float("nan")
    ik = [r["ik_residual_cm"] for r in recs if r.get("ik_residual_cm") is not None]
    if ik:
        out["ik_residual_cm"] = {"mean": float(np.mean(ik)), "n": len(ik), "p90": float(np.percentile(ik, 90))}
    return out


def group_stats(clips: Mapping[str, Mapping[str, Any]], success: Mapping[str, Any] | None = None) -> dict[str, Any]:
    recs = list(clips.values())
    n = len(recs)
    fail_free = [r for r in recs if r["fail_free"]]
    out: dict[str, Any] = {
        "n_clips": n,
        "n_hold_valid": sum(1 for r in recs if r["hold_cols"][1] > r["hold_cols"][0]),
        "fail_free_frac": (len(fail_free) / n) if n else float("nan"),
        "reach_fail_frac": (sum(1 for r in recs if r["reach_fail"]) / n) if n else float("nan"),
        "metrics": {m: _stats([r["hold_mean"][m] for r in recs], [r["final_mean"][m] for r in recs]) for m in ALL_METRICS},
        "alive": {
            "n_clips": len(fail_free),
            "metrics": {m: _stats([r["hold_mean"][m] for r in fail_free], [r["final_mean"][m] for r in fail_free]) for m in ALL_METRICS},
        },
        "success": success_stats(recs, success),
    }
    rest = [r for r in recs if r.get("rest_mean")]
    if rest:   # retract layer: rest-pose window statistics (new key only)
        out["rest"] = {"n_clips": len(rest), "metrics": {m: _stats([r["rest_mean"][m] for r in rest], []) for m in ALL_METRICS}}
    causes: dict[str, int] = {}
    for r in recs:
        if not r["fail_free"]:
            causes[r["first_fail_cause"] or "?"] = causes.get(r["first_fail_cause"] or "?", 0) + 1
    out["fail_causes"] = dict(sorted(causes.items(), key=lambda kv: -kv[1]))
    odo = [r["odometry"] for r in recs if r.get("odometry")]
    if odo:  # odometry block (only when the series carried the columns)
        out["odometry"] = {}
        for m in ODOM_METRICS:
            hold = [o["hold_mean_abs"][m] for o in odo if m in o["hold_mean_abs"]]
            end = [o["end"][m] for o in odo if m in o["end"]]
            if not hold and not end:
                continue
            fin_end = [e for e in end if math.isfinite(e)]
            out["odometry"][m] = {"hold_mean_abs": _stats(hold, []), "end_abs": _stats([abs(e) for e in end], []), "end_signed_mean": (float(np.mean(fin_end)) if fin_end else float("nan"))}
    return out


def height_labels(clips: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return sorted({r["height"] for r in clips.values()})


def group_values(clips: Mapping[str, Mapping[str, Any]]) -> list[str]:
    return sorted({r["group"] for r in clips.values() if r.get("group") is not None})


def bench_groups(clips: Mapping[str, Mapping[str, Any]], success: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """hero_bench_v1 group blocks keyed like the closed-loop summary's: ``cell:<stratum>/<height>/<hand>`` (hand in left / right / all),
    ``stratum:<s>`` and ``tier:<t>`` (all hands) -> :func:`group_stats`.  The side-by-side report joins the two summaries on these keys."""
    out: dict[str, Any] = {}
    strata = sorted({r["stratum"] for r in clips.values() if r.get("stratum")})
    tiers = sorted({r["tier"] for r in clips.values() if r.get("tier")})
    for st in strata:
        rs_ = {k: r for k, r in clips.items() if r.get("stratum") == st}
        out[f"stratum:{st}"] = group_stats(rs_, success)
        for h in sorted({r["height"] for r in rs_.values()}):
            for hand in HANDS:
                sel = {k: r for k, r in rs_.items() if r["height"] == h and (hand == ALL or r["hand"] == hand)}
                if sel:
                    out[f"cell:{st}/{h}/{hand}"] = group_stats(sel, success)
    for t in tiers:
        out[f"tier:{t}"] = group_stats({k: r for k, r in clips.items() if r.get("tier") == t}, success)
    return out


def summarize_series(series: Mapping[str, Any], manifest: Mapping[str, Mapping[str, Any]], final_s: float, path: str = "",
                     group_by: str | None = None, success: Mapping[str, Any] | None = None) -> dict[str, Any]:
    meta = series.get("meta") or json.loads(str(series.get("meta_json") or "{}"))
    if meta.get("errors"):
        raise ValueError(f"refusing to score an incomplete run with runner errors: {meta['errors']}")
    group_by = None if group_by in (None, "", "none") else str(group_by)
    success = success if success is not None else success_protocol(None)
    joined = per_clip_table(series, manifest, final_s, group_by, success)
    clips = joined["clips"]
    heights = height_labels(clips)
    groups: dict[str, Any] = {}
    for h in heights + [ALL]:
        groups[h] = {}
        for hand in HANDS:
            sel = {k: r for k, r in clips.items() if (h == ALL or r["height"] == h) and (hand == ALL or r["hand"] == hand)}
            groups[h][hand] = group_stats(sel, success)
    out = {
        "series_path": str(path),
        "sim": str(series["sim"]),
        "dt": float(series["dt"]),
        "horizon_steps": int(series["horizon_steps"]),
        "meta": dict(series.get("meta", {})),
        "n_rows": len(series["clip_name"]),
        "n_matched": len(clips),
        "n_unmatched_rows": len(joined["unmatched_rows"]),
        "unmatched_rows": joined["unmatched_rows"][:50],
        "n_manifest_missing": len(joined["manifest_missing"]),
        "manifest_missing": joined["manifest_missing"][:50],
        "heights": heights,
        "groups": groups,
        "bench_groups": bench_groups(clips, success),
        "strata": sorted({r["stratum"] for r in clips.values() if r.get("stratum")}),
        "tiers": sorted({r["tier"] for r in clips.values() if r.get("tier")}),
        "clips": clips,
    }
    if group_by:
        # extra grouping: per group value x hand over all heights, and per group value x height (hands all + per hand) -- same statistics as the
        # height groups; clips whose manifest row lacks the field are left out of these tables
        values = group_values(clips)
        fam_groups: dict[str, Any] = {}
        fam_height_groups: dict[str, Any] = {}
        for g in values:
            fam_groups[g] = {}
            for hand in HANDS:
                sel = {k: r for k, r in clips.items() if r.get("group") == g and (hand == ALL or r["hand"] == hand)}
                fam_groups[g][hand] = group_stats(sel, success)
            fam_height_groups[g] = {}
            for h in heights:
                fam_height_groups[g][h] = {}
                for hand in HANDS:
                    sel = {k: r for k, r in clips.items() if r.get("group") == g and r["height"] == h and (hand == ALL or r["hand"] == hand)}
                    fam_height_groups[g][h][hand] = group_stats(sel, success)
        out.update({"group_by": group_by, "group_values": values, "n_ungrouped": sum(1 for r in clips.values() if r.get("group") is None),
                    "family_groups": fam_groups, "family_height_groups": fam_height_groups})
    return out


def summarize(series_list: Sequence[Mapping[str, Any]], labels: Sequence[str], manifest: Mapping[str, Mapping[str, Any]],
              final_s: float = 1.0, paths: Sequence[str] = (), eval_meta: Mapping[str, Any] | None = None,
              manifest_path: str = "", group_by: str | None = None, protocol: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """``protocol`` = the manifest's top-level ``protocol`` block (:func:`load_manifest_protocol`): its ``success`` thresholds drive the success
    columns (defaults otherwise; recorded under ``success_protocol``)."""
    success = success_protocol(protocol)
    sims: dict[str, Any] = {}
    for i, (s, lab) in enumerate(zip(series_list, labels)):
        sims[lab] = summarize_series(s, manifest, final_s, paths[i] if i < len(paths) else "", group_by, success)
    heights = sorted({h for s in sims.values() for h in s["heights"]})
    out = {
        "schema": SCHEMA,
        "manifest": str(manifest_path),
        "n_manifest_clips": len(manifest),
        "hold_window": {"start": "reach_end_frame (series column index)", "end": "hold_end_frame | reach_end_frame + hold_frames | clip end (old manifests)",
                        "final_s": float(final_s), "percentiles": list(PERCENTILES)},
        "metrics": list(ALL_METRICS),
        "heights": heights,
        "strata": sorted({g for s in sims.values() for g in s.get("strata", [])}),
        "tiers": sorted({g for s in sims.values() for g in s.get("tiers", [])}),
        "success_protocol": success,
        "eval_meta": dict(eval_meta or {}),
        "sims": sims,
    }
    if group_by and group_by != "none":
        out["group_by"] = str(group_by)
        out["group_values"] = sorted({g for s in sims.values() for g in s.get("group_values", [])})
    return out


def default_labels(series_list: Sequence[Mapping[str, Any]]) -> list[str]:
    """Label = ``sim`` when unique; series sharing a ``sim`` get ``<sim>-<meta.tag>`` when they carry a tag (the runner's ``--tag``), else
    ``<sim>_2``, ..."""
    sims = [str(s["sim"]) for s in series_list]
    tags = [str((s.get("meta") or {}).get("tag") or "") for s in series_list]
    seen: dict[str, int] = {}
    out: list[str] = []
    for sim, tag in zip(sims, tags):
        lab = f"{sim}-{tag}" if (sims.count(sim) > 1 and tag) else sim
        seen[lab] = seen.get(lab, 0) + 1
        out.append(lab if seen[lab] == 1 else f"{lab}_{seen[lab]}")
    return out


# ==================================================================================================
# outputs
# ==================================================================================================
def _digits(metric: str) -> int:
    return 3 if metric.endswith("_rad") else 1


def _pm(st: Mapping[str, Any], metric: str, key: str = "mean", sd: str = "std") -> str:
    if not st or st.get("n", 0) == 0 or not math.isfinite(st.get(key, float("nan"))):
        return "-"
    d = _digits(metric)
    return f"{st[key]:.{d}f} ± {st[sd]:.{d}f}"


def _val(x: float | None, metric: str = "") -> str:
    if x is None or not (isinstance(x, (int, float)) and math.isfinite(x)):
        return "-"
    return f"{x:.{_digits(metric)}f}"


def _pct(x: float | None) -> str:
    return "-" if x is None or not math.isfinite(x) else f"{100.0 * x:.1f}%"


def _ci(level: Mapping[str, Any] | None) -> str:
    if not level or not level.get("n") or not math.isfinite(level.get("frac", float("nan"))):
        return "-"
    lo, hi = level.get("ci95", [float("nan"), float("nan")])
    return f"{100.0 * level['frac']:.1f}% [{100.0 * lo:.0f}, {100.0 * hi:.0f}]"


def success_row_cells(g: Mapping[str, Any]) -> str:
    """Cells of one row of the active-hand / success table (``n | global active | rot world active | global both | local active | fail-free |
    <levels> | CDF | IK residual``) of a group-statistics block."""
    gm = g.get("metrics", {})
    sc = g.get("success", {})
    cells = [str(g.get("n_clips", 0)), _pm(gm.get("ee_global_active_cm"), "ee_global_active_cm"), _pm(gm.get("ee_rot_global_active_deg"), "ee_rot_global_active_deg"),
             _pm(gm.get("ee_global_cm"), "ee_global_cm"), _pm(gm.get("ee_local_active_cm"), "ee_local_active_cm"), _pct(g.get("fail_free_frac", float("nan")))]
    cells += [_ci(sc.get("levels", {}).get(lv)) for lv in sc.get("levels", {})]
    cdf = sc.get("cdf_cm", {})
    cells.append(" / ".join(_pct(cdf[k]) for k in cdf) if cdf else "-")
    ik = sc.get("ik_residual_cm", {})
    cells.append(f"{ik['mean']:.2f} (n={ik['n']})" if ik.get("n") else "-")
    return " | ".join(cells)


def render_markdown(summary: Mapping[str, Any], title: str = "hero_bench per-height benchmark") -> str:
    sims = summary["sims"]
    labels = list(sims)
    heights = list(summary["heights"]) + [ALL]
    em = summary.get("eval_meta", {})
    lines = [f"## {title}", ""]
    if em:
        lines.append(f"ckpt `{em.get('ckpt', '?')}` (iter {em.get('iter', '?')}), preset `{em.get('exp', '?')}`  ")
    for lab in labels:
        s = sims[lab]
        m = s.get("meta", {})
        lines.append(
            f"`{lab}`: `{s['series_path']}` (sim {s['sim']}, dt {s['dt']:g} s, horizon {s['horizon_steps']} steps; "
            f"{s['n_matched']} / {s['n_rows']} rows matched to the manifest, {s['n_manifest_missing']} manifest clips missing"
            + (f"; ckpt `{m.get('ckpt')}`" if m.get("ckpt") else "") + ")  "
        )
    lines.append(f"Manifest `{summary['manifest']}` ({summary['n_manifest_clips']} clips). Hold window = frames >= reach_end_frame within the "
                 f"valid window (ending at the row's hold_end_frame when the manifest has it); \"final\" = last {summary['hold_window']['final_s']:g} s of the hold. "
                 "Values: per-clip hold mean -> across-clip mean ± std.")
    lines.append("")

    def paper_table(trans_metric: str, heading: str) -> None:
        cols = ((trans_metric, "Trans (cm)"), ("ee_rot_deg", "Orient (deg)"), ("joint_upper_rad", "Joint (rad)"))
        lines.append(f"### {heading} -- all clips, both hands")
        lines.append("")
        head = "| height | n |" + "".join(f" {lab} {c} |" for lab in labels for _, c in cols) + "".join(f" {lab} fail-free |" for lab in labels)
        lines.append(head)
        lines.append("|---|---|" + "---|" * (len(labels) * (len(cols) + 1)))
        for h in heights:
            cells = []
            n = "/".join(str(sims[lab]["groups"].get(h, {}).get(ALL, {}).get("n_clips", 0)) for lab in labels)
            for lab in labels:
                g = sims[lab]["groups"].get(h, {}).get(ALL, {})
                cells += [_pm(g.get("metrics", {}).get(m), m) for m, _ in cols]
            for lab in labels:
                g = sims[lab]["groups"].get(h, {}).get(ALL, {})
                cells.append(_pct(g.get("fail_free_frac", float("nan"))))
            lines.append(f"| {h} | {n} | " + " | ".join(cells) + " |")
        lines.append("")

    paper_table("ee_global_cm", "Global EE (world frame; includes root drift)")
    paper_table("ee_local_cm", "Local EE (own-pelvis HERO geometry)")

    sp = summary.get("success_protocol") or {}
    levels = list((sp.get("open_loop") or {}).keys())
    if levels:
        thr = ", ".join(f"{lv} = <= {sp['open_loop'][lv]['pos_cm']:g} cm & <= {sp['open_loop'][lv]['rot_deg']:g} deg" for lv in levels)
        cdf_pts = [f"{float(x):g}" for x in sp.get("cdf_points_cm", [])]
        lines.append(f"### Active (reaching) hand -- world-frame error (HEADLINE), success rates and CDF points (hero_bench_v1 protocol; thresholds from the "
                     f"{sp.get('source', 'default')}: {thr}; fail-free required; Wilson 95 % CI; denominator = every clip of the group)")
        lines.append("")
        lines.append("| group | sim | n | global active (cm) | rot world active (deg) | global both (cm) | local active (cm) | fail-free | " + " | ".join(levels)
                     + " | CDF <= " + " / ".join(cdf_pts) + " cm | IK residual (cm) |")
        lines.append("|---|---|---|---|---|---|---|---|" + "---|" * len(levels) + "---|---|")
        for h in heights:
            for lab in labels:
                g = sims[lab]["groups"].get(h, {}).get(ALL, {})
                if not g or g.get("n_clips", 0) == 0:
                    continue
                lines.append(f"| {h} | {lab} | " + success_row_cells(g) + " |")
        extra = [k for lab in labels for k in sims[lab].get("bench_groups", {}) if k.startswith("stratum:") or k.startswith("tier:")]
        seen: list[str] = []
        for k in extra:
            if k not in seen:
                seen.append(k)
        for k in sorted(seen):
            if k.startswith("stratum:") and k[len("stratum:"):] in heights:
                continue   # paper-protocol strata coincide with the height rows above
            for lab in labels:
                g = sims[lab].get("bench_groups", {}).get(k)
                if g and g.get("n_clips", 0):
                    lines.append(f"| {k} | {lab} | " + success_row_cells(g) + " |")
        lines.append("")

    lines.append("### Alive-only (clips with no training-termination failure inside their valid window)")
    lines.append("")
    lines.append("| height |" + "".join(f" {lab} n alive | {lab} global (cm) | {lab} local (cm) | {lab} orient (deg) | {lab} joint (rad) |" for lab in labels))
    lines.append("|---|" + "---|" * (5 * len(labels)))
    for h in heights:
        cells = []
        for lab in labels:
            a = sims[lab]["groups"].get(h, {}).get(ALL, {}).get("alive", {})
            am = a.get("metrics", {})
            cells += [str(a.get("n_clips", 0))] + [_pm(am.get(m), m) for m in ("ee_global_cm", "ee_local_cm", "ee_rot_deg", "joint_upper_rad")]
        lines.append(f"| {h} | " + " | ".join(cells) + " |")
    lines.append("")

    if any("odometry" in sims[lab]["groups"].get(h, {}).get(ALL, {}) for lab in labels for h in heights):
        lines.append("### Odometry error (`--odom so`; per-clip hold-window mean |estimate - truth| -> mean ± std over clips; `end` = |error| at the clip end)")
        lines.append("")
        lines.append("| height | sim | n | xy hold (cm) | xy end (cm) | yaw hold (deg) | yaw end (deg) | z hold (cm) | z end (cm) | z end signed (cm) |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")

        def _o(st: Mapping[str, Any] | None) -> str:
            if not st or st.get("n", 0) == 0 or not math.isfinite(st.get("mean", float("nan"))):
                return "-"
            return f"{st['mean']:.2f} ± {st['std']:.2f}"

        for h in heights:
            for lab in labels:
                g = sims[lab]["groups"].get(h, {}).get(ALL, {})
                od = g.get("odometry")
                if not od:
                    continue
                cells = [str(g.get("n_clips", 0))]
                for m in ODOM_METRICS:
                    cells += [_o(od.get(m, {}).get("hold_mean_abs")), _o(od.get(m, {}).get("end_abs"))]
                zs = od.get("odom_z_err_cm", {}).get("end_signed_mean", float("nan"))
                cells.append("-" if not math.isfinite(zs) else f"{zs:.2f}")
                lines.append(f"| {h} | {lab} | " + " | ".join(cells) + " |")
        lines.append("")

    lines.append("### CDF points (per-clip hold means; p50 / p80 / p90) and final-second means")
    lines.append("")
    cdf_metrics = ("ee_global_cm", "ee_local_cm", "ee_rot_deg", "joint_upper_rad", "anchor_xy_cm")
    lines.append("| height | sim | " + " | ".join(f"{m} p50/p80/p90" for m in cdf_metrics) + " | " + " | ".join(f"{m} final" for m in cdf_metrics) + " |")
    lines.append("|---|---|" + "---|" * (2 * len(cdf_metrics)))
    for h in heights:
        for lab in labels:
            g = sims[lab]["groups"].get(h, {}).get(ALL, {}).get("metrics", {})
            cdf = ["/".join(_val(g.get(m, {}).get(f"p{p}"), m) for p in PERCENTILES) if g.get(m, {}).get("n", 0) else "-" for m in cdf_metrics]
            fin = [_val(g.get(m, {}).get("final_mean"), m) if g.get(m, {}).get("n", 0) else "-" for m in cdf_metrics]
            lines.append(f"| {h} | {lab} | " + " | ".join(cdf) + " | " + " | ".join(fin) + " |")
    lines.append("")

    lines.append("### Per hand (mean ± std of per-clip hold means; `active` = the reaching hand's own column)")
    lines.append("")
    lines.append("| height | hand | sim | n | global (cm) | global active (cm) | local (cm) | local active (cm) | orient (deg) | joint (rad) | anchor xy (cm) | fail-free | reach-fail |")
    lines.append("|---|---|---|---|---|---|---|---|---|---|---|---|---|")
    for h in heights:
        for hand in HANDS:
            for lab in labels:
                g = sims[lab]["groups"].get(h, {}).get(hand, {})
                if not g or g.get("n_clips", 0) == 0:
                    continue
                gm = g["metrics"]
                lines.append(
                    f"| {h} | {hand} | {lab} | {g['n_clips']} | {_pm(gm.get('ee_global_cm'), 'ee_global_cm')} | "
                    f"{_pm(gm.get('ee_global_active_cm'), 'ee_global_active_cm')} | {_pm(gm.get('ee_local_cm'), 'ee_local_cm')} | "
                    f"{_pm(gm.get('ee_local_active_cm'), 'ee_local_active_cm')} | {_pm(gm.get('ee_rot_deg'), 'ee_rot_deg')} | "
                    f"{_pm(gm.get('joint_upper_rad'), 'joint_upper_rad')} | {_pm(gm.get('anchor_xy_cm'), 'anchor_xy_cm')} | "
                    f"{_pct(g['fail_free_frac'])} | {_pct(g['reach_fail_frac'])} |"
                )
    lines.append("")

    fam_values = list(summary.get("group_values") or [])
    if summary.get("group_by") and fam_values:
        gb = summary["group_by"]
        cols = ("ee_global_cm", "ee_global_active_cm", "ee_local_cm", "ee_rot_deg", "joint_upper_rad", "anchor_xy_cm")

        def fam_line(name: str, hand: str, getter) -> str | None:
            cells = []
            any_n = False
            for lab in labels:
                g = getter(lab) or {}
                n = g.get("n_clips", 0)
                any_n = any_n or n > 0
                gm = g.get("metrics", {})
                cells += [str(n)] + [_pm(gm.get(m), m) for m in cols] + [_pct(g.get("fail_free_frac", float("nan"))), _pct(g.get("reach_fail_frac", float("nan")))]
            if not any_n:
                return None
            return f"| {name} | {hand} | " + " | ".join(cells) + " |"

        head_cols = "".join(f" {lab} n | {lab} global (cm) | {lab} global active (cm) | {lab} local (cm) | {lab} orient (deg) | {lab} joint (rad) | {lab} anchor xy (cm) | {lab} fail-free | {lab} reach-fail |" for lab in labels)
        lines.append(f"### Per `{gb}` (all heights; per-clip hold means -> mean ± std)")
        lines.append("")
        lines.append(f"| {gb} | hand |" + head_cols)
        lines.append("|---|---|" + "---|" * (9 * len(labels)))
        for g in fam_values:
            for hand in HANDS:
                row = fam_line(g, hand, lambda lab, g=g, hand=hand: sims[lab].get("family_groups", {}).get(g, {}).get(hand))
                if row:
                    lines.append(row)
        lines.append("")
        lines.append(f"### `{gb}` x height (both hands)")
        lines.append("")
        lines.append(f"| {gb} | height |" + head_cols)
        lines.append("|---|---|" + "---|" * (9 * len(labels)))
        for g in fam_values:
            for h in summary["heights"]:
                row = fam_line(g, h, lambda lab, g=g, h=h: sims[lab].get("family_height_groups", {}).get(g, {}).get(h, {}).get(ALL))
                if row:
                    lines.append(row)
        lines.append("")
        for lab in labels:
            if sims[lab].get("n_ungrouped"):
                lines.append(f"`{lab}`: {sims[lab]['n_ungrouped']} matched clips without `{gb}` in the manifest (not in the tables above).  ")
    for lab in labels:
        s = sims[lab]
        fc = s["groups"].get(ALL, {}).get(ALL, {}).get("fail_causes", {})
        if fc:
            lines.append(f"`{lab}` first-failure causes over all clips: " + ", ".join(f"{k} {v}" for k, v in fc.items()) + ".  ")
        if s["n_unmatched_rows"]:
            lines.append(f"`{lab}`: {s['n_unmatched_rows']} series rows not in the manifest (e.g. {', '.join(s['unmatched_rows'][:3])}).  ")
    return "\n".join(lines) + "\n"


CSV_FIELDS = ("sim", "height", "hand", "orient_family", "subset", "metric", "n", "mean", "std", "p50", "p80", "p90", "final_mean", "final_std",
              "n_clips", "fail_free_frac", "reach_fail_frac")


def _csv_block(rows: list[dict[str, Any]], lab: str, h: str, hand: str, fam: str, g: Mapping[str, Any]) -> None:
    for subset, block in (("all", g), ("alive", g["alive"])):
        for m, st in block["metrics"].items():
            rows.append({
                "sim": lab, "height": h, "hand": hand, "orient_family": fam, "subset": subset, "metric": m,
                **{k: st[k] for k in ("n", "mean", "std", "p50", "p80", "p90", "final_mean", "final_std")},
                "n_clips": block["n_clips"], "fail_free_frac": g["fail_free_frac"], "reach_fail_frac": g["reach_fail_frac"],
            })


def csv_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    """Height rows (``orient_family`` empty) + (with --group-by) group rows: ``height`` = ``all`` for the all-heights group blocks, the height
    label for group x height."""
    rows: list[dict[str, Any]] = []
    for lab, s in summary["sims"].items():
        for h, hands in s["groups"].items():
            for hand, g in hands.items():
                _csv_block(rows, lab, h, hand, "", g)
        for fam, hands in (s.get("family_groups") or {}).items():
            for hand, g in hands.items():
                _csv_block(rows, lab, ALL, hand, fam, g)
        for fam, heights in (s.get("family_height_groups") or {}).items():
            for h, hands in heights.items():
                for hand, g in hands.items():
                    _csv_block(rows, lab, h, hand, fam, g)
    return rows


def write_outputs(summary: Mapping[str, Any], out_dir: str | Path, title: str = "") -> dict[str, Path]:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    paths = {"json": out / "bench_summary.json", "md": out / "bench_summary.md", "csv": out / "bench_summary.csv"}
    paths["json"].write_text(json.dumps(summary, indent=1, default=str))
    paths["md"].write_text(render_markdown(summary, title or "hero_bench per-height benchmark"))
    with open(paths["csv"], "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(CSV_FIELDS))
        w.writeheader()
        for r in csv_rows(summary):
            w.writerow(r)
    return paths


def _payload_group(payload: dict[str, float], base: str, g: Mapping[str, Any]) -> None:
    for m, st in g["metrics"].items():
        if st["n"]:
            payload[f"{base}/{m}"] = st["mean"]
            payload[f"{base}/{m}_p90"] = st["p90"]
            payload[f"{base}/{m}_final"] = st["final_mean"]
    for m, st in g["alive"]["metrics"].items():
        if st["n"]:
            payload[f"{base}/alive/{m}"] = st["mean"]
    payload[f"{base}/fail_free_frac"] = g["fail_free_frac"]
    payload[f"{base}/reach_fail_frac"] = g["reach_fail_frac"]
    payload[f"{base}/n_clips"] = g["n_clips"]
    payload[f"{base}/n_alive"] = g["alive"]["n_clips"]
    for lv, st in (g.get("success") or {}).get("levels", {}).items():   # hero_bench_v1 success fractions
        if st.get("n"):
            payload[f"{base}/success_{lv.replace('.', '_')}"] = st["frac"]
    for pt, frac in (g.get("success") or {}).get("cdf_cm", {}).items():
        payload[f"{base}/cdf_le_{pt.replace('.', '_')}cm"] = frac
    for m, st in (g.get("odometry") or {}).items():  # odometry error (--odom so series only)
        if st["hold_mean_abs"]["n"]:
            payload[f"{base}/{m}_hold"] = st["hold_mean_abs"]["mean"]
        if st["end_abs"]["n"]:
            payload[f"{base}/{m}_end"] = st["end_abs"]["mean"]


def flat_scalars(summary: Mapping[str, Any]) -> dict[str, float]:
    """Flat ``bench/<sim>/<height>/<metric>[_p90|_final]`` + ``fail_free_frac`` / ``reach_fail_frac`` / ``n_clips`` / ``alive/`` / ``success_*`` /
    ``cdf_le_*`` scalars (one experiment-tracker payload); with --group-by also ``bench/<sim>/fam-<group>/...`` (all heights) and
    ``bench/<sim>/fam-<group>/<height>/...``.  Non-finite values are dropped."""
    payload: dict[str, float] = {}
    for lab, s in summary["sims"].items():
        for h, hands in s["groups"].items():
            _payload_group(payload, f"bench/{lab}/{h}", hands[ALL])
        for fam, hands in (s.get("family_groups") or {}).items():
            _payload_group(payload, f"bench/{lab}/fam-{fam}", hands[ALL])
        for fam, heights in (s.get("family_height_groups") or {}).items():
            for h, hands in heights.items():
                _payload_group(payload, f"bench/{lab}/fam-{fam}/{h}", hands[ALL])
    return {k: float(v) for k, v in payload.items() if isinstance(v, (int, float)) and math.isfinite(float(v))}


# ==================================================================================================
# CLI
# ==================================================================================================
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", nargs="+", required=True, help="hero_eval_series_v1 npz files (mujoco / isaacsim)")
    ap.add_argument("--labels", default="", help="comma-separated labels for --series (default: the sim field, de-duplicated)")
    ap.add_argument("--bench-manifest", required=True)
    ap.add_argument("--out", required=True, help="output directory")
    ap.add_argument("--final-s", type=float, default=1.0, help="length of the 'final' window at the end of the hold (s)")
    ap.add_argument("--title", default="")
    ap.add_argument("--group-by", default="none", choices=list(GROUP_BY_CHOICES),
                    help="extra grouping by a manifest field (orient_family; stratum = layer, tier = core/extended/stress; rows without those fields fall back to "
                         "stratum = height_label / tier = core) -> per-group and group x height tables (default off; the bench_groups JSON block is always written)")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    a = build_parser().parse_args(argv)
    manifest = load_manifest(a.bench_manifest)
    series_list = [read_series(p) for p in a.series]
    labels = [x.strip() for x in a.labels.split(",") if x.strip()] if a.labels else default_labels(series_list)
    if len(labels) != len(series_list):
        raise SystemExit(f"--labels has {len(labels)} entries for {len(series_list)} series")
    eval_meta: dict[str, Any] = {}
    for s in series_list:
        if s.get("meta", {}).get("ckpt"):
            eval_meta = dict(s["meta"])
            break
    for s in series_list:
        fps = {int(e["fps"]) for e in manifest.values() if e.get("fps")}
        if fps and any(abs(1.0 / f - float(s["dt"])) > 1e-6 for f in fps):
            print(f"[bench] WARNING: manifest fps {sorted(fps)} vs series dt {s['dt']} (sim {s['sim']}) -- frame/step mismatch", flush=True)
    group_by = None if a.group_by == "none" else a.group_by
    if group_by and group_by not in BENCH_GROUP_FIELDS and not any(e.get(group_by) for e in manifest.values()):
        print(f"[bench] WARNING: --group-by {group_by} requested but no manifest row carries that field (legacy manifest?) -- grouping skipped", flush=True)
        group_by = None
    elif group_by in BENCH_GROUP_FIELDS and not any(e.get(group_by) for e in manifest.values()):
        print(f"[bench] note: no manifest row carries `{group_by}` -- legacy fallback in force ({group_by} = {BENCH_GROUP_FIELDS[group_by]})", flush=True)
    protocol = load_manifest_protocol(a.bench_manifest)
    try:
        summary = summarize(series_list, labels, manifest, a.final_s, a.series, eval_meta, a.bench_manifest, group_by, protocol)
    except ValueError as exc:
        print(f"[bench] {exc}", file=sys.stderr)
        return 1
    sp = summary["success_protocol"]
    print(f"[bench] success thresholds from the {sp['source']}: " + ", ".join(f"{k} <= {v['pos_cm']:g} cm / {v['rot_deg']:g} deg" for k, v in sp["open_loop"].items())
          + f"; CDF points {sp['cdf_points_cm']} cm", flush=True)
    paths = write_outputs(summary, a.out, a.title)
    print(paths["md"].read_text(), flush=True)
    for lab, s in summary["sims"].items():
        print(f"[bench] {lab}: matched {s['n_matched']}/{s['n_rows']} rows; manifest missing {s['n_manifest_missing']}", flush=True)
    print(f"[bench] wrote {paths['json']} {paths['md']} {paths['csv']}", flush=True)
    return 0


if __name__ == "__main__":
    sys.exit(main())
