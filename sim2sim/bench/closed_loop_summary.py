"""Closed-loop (replanning / goal-adjustment) benchmark summary: several ``hero_eval_series_v1`` runs of the SAME bench corpus,
scored per table height / layer in three windows, plus the replanner bookkeeping stored in the series meta.

    python -m sim2sim.bench.closed_loop_summary --manifest <corpus>/BENCH_MANIFEST.json --out <dir> \\
        --series "open|<ol>/series.npz" --series "replan|<cl>/series.npz" ... [--csv table.csv]

Windows (columns of the (C, T) metric arrays; column j = policy step j+1 = reference frame j+1 of the ORIGINAL clip):

* ``hold3``  = ``[reach_end, hold_end_frame)`` of the ROW (``hold_end_frame``, else ``reach_end + hold_frames``; ``--hold-s`` is only the
  fallback for rows that carry neither) -- the original protocol's hold window (the open-loop plan says the hand has arrived; the same window
  :mod:`sim2sim.bench.summary` uses);
* ``final``  = the last ``final_s`` seconds before the (padded) clip end -- where a closed loop has converged;
* ``tail``   = ``[reach_end, clip_len_steps)`` -- everything after the planned arrival.

Rows with their OWN retract segment (the hero_bench_v1 ``retract`` layer: manifest ``retract_start_frame``, else the replanner's recorded
``stop.stop_frame`` in ``replan_log[clip]``): the replanner hands the controller back to the clip at that frame, so the end of the padded
rollout is the REST pose, not the goal.  For them ``final`` = ``[retract_start_frame - final_s * fps, retract_start_frame)`` and ``tail`` =
``[reach_end, retract_start_frame)`` (``hold3`` unchanged), and the rest pose is scored apart in a ``rest`` window = the last ``final_s``
seconds of the padded clip (per clip / group ``rest_pos_err_cm`` / ``rest_rot_err_deg`` of the active hand; never mixed into ``final`` /
``C3``).  Such groups carry a ``notes`` entry: the replan / stayed counts of retract rows allow one replan decision before the hand-back.
Rows without a retract segment keep the windows above exactly.

All windows are intersected with ``[0, valid_until)``.  Metrics: the reaching hand's world-frame palm error (``ee_global_<hand>_cm``, the
paper's number), its own-pelvis error and rotation error, the both-hand means, the 14 arm joints, root xy drift and root height error --
per-clip window means reduced to mean / std / p50 / p90 per group; fail-free fraction from ``first_fail_step``; replan stats (count,
converged / "stayed" fraction, goal offset, IK time) from ``meta_json["replan_log"]``.  Writes ``<out>/closed_loop_summary.json`` and ``.md``.

Groups come from the MANIFEST: every height label seen (``height_label``, else the series ``source_tag``) plus ``stratum:<s>`` / ``tier:<t>`` /
``cell:<stratum>/<height>/<hand>`` (legacy rows without a stratum field: stratum = height_label, tier = core; the keys :mod:`sim2sim.bench.report` joins on) and
``fam:<orient_family>`` when the rows carry one; ``results[label]["heights"]`` lists the height columns of the tables.  Every group carries
``success``: ``C3`` = fail-free and ``final``-window ACTIVE-hand world-frame palm error <= 3 cm and world-frame rotation error <= 15 deg
(thresholds from the manifest ``protocol.success.closed_loop`` when present, else the :mod:`sim2sim.bench.summary` defaults) with the Wilson
95 % CI, the ``stayed`` fraction (HERO's own 1.75 cm stop rule, from the replan log) and the CDF points of the final-window active error.

Jerk table (series with the :data:`sim2sim.bench.rollout.JERK_METRIC_KEYS` columns): per clip the mean over the valid steps and the mean over
the ``--jerk-window-s`` (0.3 s) after every replan step of ``replan_log[clip]["replan_steps"]`` (open-loop runs have no events: ``--``), reduced
over clips; series without the columns skip the table.  ``--csv PATH`` writes one flat row per series label.

Manifest rows are matched to clips by the EXACT file stem (:func:`entry_for`); only a legacy manifest without any ``stratum`` field falls
back to the ``clip_id`` / name tail (one warning) -- in hero_bench_v1 the re-timed twins share their paper-protocol ``clip_id``.
"""

from __future__ import annotations

import argparse
import json
import os
import sys
import warnings
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sim2sim.bench.series import read_series
from sim2sim.bench.summary import load_manifest_protocol, success_protocol, bench_group_value, wilson_ci

HEIGHTS: tuple[str, ...] = ("h050", "h074", "h088")   # fallback table columns for summaries without ``results[label]["heights"]``
WINDOWS: tuple[str, ...] = ("hold3", "final", "tail")
REST_WINDOW: str = "rest"   # own-retract rows only: the last final_s of the padded clip, reported apart from the reach windows
#: group / per-clip keys of the rest window -> the metric they summarise (active hand, world frame)
REST_METRICS: dict[str, str] = {"rest_pos_err_cm": "ee_global_active_cm", "rest_rot_err_deg": "ee_rot_global_active_deg"}
RETRACT_NOTE: str = "stayed/replan counts: retract rows allow one replan decision before hand-back"
JERK_KEYS: tuple[str, ...] = ("arm_target_delta_rad", "arm_accel_rad_s2", "ref_palm_speed_cm_s")  # == sim2sim.bench.rollout.JERK_METRIC_KEYS
METRICS: tuple[str, ...] = (
    "ee_global_active_cm", "ee_local_active_cm", "ee_rot_active_deg", "ee_rot_global_active_deg",
    "ee_global_cm", "ee_local_cm", "ee_rot_deg", "ee_rot_global_deg", "joint_upper_rad", "anchor_xy_cm", "base_height_cm",
)


# ================================================================================================ manifest
class ManifestRows(dict):
    """``{file stem: manifest row}`` plus the manifest ``path`` it was read from (error messages) and the once-per-manifest legacy-fallback flag."""

    path: str | None = None
    legacy_fallback_warned: bool = False


def load_manifest(path: str | os.PathLike) -> ManifestRows:
    obj = json.loads(Path(path).read_text())
    rows = obj.get("clips") if isinstance(obj, dict) else obj
    out = ManifestRows()
    out.path = str(path)
    for e in rows:
        key = str(e.get("file") or e.get("clip_id"))
        out[key[:-4] if key.endswith(".npz") else key] = dict(e)
    return out


def manifest_is_legacy(manifest: Mapping[str, Mapping[str, Any]]) -> bool:
    """True for a legacy manifest: no row carries a ``stratum`` field (every hero_bench_v1 row does)."""
    return not any(isinstance(v, Mapping) and v.get("stratum") is not None for v in manifest.values())


def entry_for(manifest: Mapping[str, dict[str, Any]], clip_name: str) -> dict[str, Any]:
    """The manifest row of a clip.  hero_bench_v1 manifests (rows with ``stratum``): the EXACT file stem, else a KeyError naming the clip and
    the manifest -- the re-timed twins ``slow_x2__<id>`` / ``hold6__<id>`` / ``fast_x0p75__<id>`` share their paper-protocol ``clip_id``, so a partial match
    would silently score a clip against its twin's frames.  Legacy manifests (no ``stratum`` anywhere): the old fallback -- the part after
    ``__`` or the ``clip_id`` -- with ONE RuntimeWarning per manifest."""
    stem = os.path.basename(str(clip_name))
    if stem.endswith(".npz"):
        stem = stem[:-4]
    if stem in manifest:
        return manifest[stem]
    where = f" ({manifest.path})" if getattr(manifest, "path", None) else ""
    if not manifest_is_legacy(manifest):
        raise KeyError(f"clip {clip_name!r}: no manifest row with the file stem {stem!r} in the bench manifest{where} -- hero_bench_v1 rows are matched by "
                       "exact file name (re-timed twins share a clip_id, so no partial match is attempted); is this the manifest of the clip's tier?")
    tail = stem.split("__", 1)[-1]
    for k, v in manifest.items():
        if k.split("__", 1)[-1] == tail or str(v.get("clip_id")) == tail:
            if not getattr(manifest, "legacy_fallback_warned", False):
                warnings.warn(f"legacy bench manifest{where} (no stratum field): clips without an exact file-stem row are matched by their name tail / clip_id",
                              RuntimeWarning, stacklevel=2)
                try:
                    manifest.legacy_fallback_warned = True   # type: ignore[attr-defined]
                except AttributeError:   # a plain dict: the warnings module de-duplicates the (constant) message instead
                    pass
            return v
    raise KeyError(f"clip {clip_name!r} not in the bench manifest{where}")


def reach_end_of(e: Mapping[str, Any]) -> int:
    if e.get("reach_end_frame") is not None:
        return int(e["reach_end_frame"])
    return max(int(e["n_frames"]) - int(e["hold_frames"]), 0)


def hold_end_of(e: Mapping[str, Any]) -> int | None:
    """End (exclusive) of the row's hold window: ``hold_end_frame`` (hero_bench_v1), else ``reach_end + hold_frames``; None when neither is known."""
    if e.get("hold_end_frame") is not None:
        return int(e["hold_end_frame"])
    if e.get("hold_frames") is not None and (e.get("reach_end_frame") is not None or e.get("n_frames") is not None):
        return reach_end_of(e) + int(e["hold_frames"])
    return None


def retract_start_of(e: Mapping[str, Any], replan_entry: Mapping[str, Any] | None) -> int | None:
    """First frame of the row's OWN retract segment: the manifest ``retract_start_frame``, else the replanner's recorded ``stop.stop_frame``
    (series meta ``replan_log[clip]["stop"]``, written when the run stopped replanning for the clip); None for every other row."""
    if e.get("retract_start_frame") is not None:
        return int(e["retract_start_frame"])
    stop = (replan_entry or {}).get("stop") or {}
    if isinstance(stop, Mapping) and stop.get("stop_frame") is not None:
        return int(stop["stop_frame"])
    return None


def heights_of(rows: Sequence[Mapping[str, Any]]) -> list[str]:
    """Sorted height labels of the per-clip rows (the manifest-derived table columns)."""
    return sorted({str(r["height"]) for r in rows})


# ================================================================================================ per clip
def _nanmean(a: np.ndarray) -> float:
    a = np.asarray(a, dtype=np.float64)
    return float(np.nanmean(a)) if a.size and np.isfinite(a).any() else float("nan")


def jerk_for_clip(m: Mapping[str, np.ndarray], i: int, stop: int, replan_steps: Sequence[int], n_win: int) -> dict[str, Any] | None:
    """Per-clip jerk block: ``mean`` over columns ``[0, stop)`` and ``post_replan`` over the union of ``[s-1, s-1+n_win)`` for every
    1-based replan step ``s`` (None without events); None when the series has no jerk columns."""
    if not all(k in m for k in JERK_KEYS):
        return None
    mask = np.zeros(int(stop), dtype=bool)
    for s_ in replan_steps or ():
        a = max(int(s_) - 1, 0)
        mask[a: min(a + n_win, int(stop))] = True
    out: dict[str, Any] = {"n_replans": int(len(replan_steps or ())), "mean": {}, "post_replan": {}}
    for k in JERK_KEYS:
        col = np.asarray(m[k][i, :stop], dtype=np.float64)
        out["mean"][k] = _nanmean(col)
        out["post_replan"][k] = _nanmean(col[mask]) if (mask.any() and np.isfinite(col[mask]).any()) else None
    return out


def per_clip_rows(series: Mapping[str, Any], manifest: Mapping[str, dict[str, Any]], *, hold_s: float, final_s: float, jerk_window_s: float = 0.3,
                  success: Mapping[str, Any] | None = None) -> list[dict[str, Any]]:
    """``hold_s`` is only the fallback hold3 length for rows without ``hold_end_frame`` / ``hold_frames``; ``success`` = the protocol block
    (:func:`sim2sim.bench.summary.success_protocol`) whose ``closed_loop`` levels are evaluated per clip on the ``final`` window."""
    success = success if success is not None else success_protocol(None)
    m = series["metrics"]
    dt = float(series["dt"])
    fps = int(round(1.0 / dt))
    T = int(series["horizon_steps"])
    meta = json.loads(str(series["meta_json"])) if "meta_json" in series else {}
    if meta.get("errors"):
        raise ValueError(f"refusing to score an incomplete run with runner errors: {meta['errors']}")
    orig_len = meta.get("clip_len_steps_original") or {}
    replan_log = meta.get("replan_log") or {}
    names = [str(x) for x in series["clip_name"]]
    tags = [str(x) for x in series["source_tag"]]
    vu = np.asarray(series["valid_until"], dtype=np.int64)
    ff = np.asarray(series["first_fail_step"], dtype=np.int64)
    clen = np.asarray(series["clip_len_steps"], dtype=np.int64)
    rows = []
    for i, name in enumerate(names):
        e = entry_for(manifest, name)
        hand = str(e.get("hand", "right")).lower()
        re_ = reach_end_of(e)
        he_ = hold_end_of(e)
        hold_frames = int(he_ - re_) if he_ is not None else int(round(hold_s * fps))   # the row's own hold length; --hold-s only for old rows
        t_orig = int(orig_len.get(name, clen[i]))
        stop_all = int(min(clen[i], T, vu[i]))
        n_final = int(round(final_s * fps))
        rl = replan_log.get(name)
        rs_ = retract_start_of(e, rl)
        if rs_ is None:   # the plain closed loop: final = the last final_s before the padded end, tail = everything after the planned arrival
            windows = {
                "hold3": (re_, min(re_ + hold_frames, t_orig, stop_all)),
                "final": (max(stop_all - n_final, re_, 0), stop_all),
                "tail": (re_, stop_all),
            }
        else:             # own-retract row: the controller is handed back to the clip at rs_, so the reach windows end there (module docstring)
            reach_stop = min(int(rs_), stop_all)
            a_final = max(int(rs_) - n_final, 0)
            windows = {
                "hold3": (re_, min(re_ + hold_frames, t_orig, stop_all)),
                "final": (a_final, max(a_final, reach_stop)),   # [rs - final_s, rs) intersected with [0, valid_until): empty when the clip ended before it
                "tail": (re_, reach_stop),
            }
        cols = {
            "ee_global_active_cm": f"ee_global_{hand}_cm", "ee_local_active_cm": f"ee_local_{hand}_cm", "ee_rot_active_deg": f"ee_rot_{hand}_deg",
            "ee_rot_global_active_deg": f"ee_rot_global_{hand}_deg",
            "ee_global_cm": "ee_global_cm", "ee_local_cm": "ee_local_cm", "ee_rot_deg": "ee_rot_deg", "ee_rot_global_deg": "ee_rot_global_deg",
            "joint_upper_rad": "joint_upper_rad", "anchor_xy_cm": "anchor_xy_cm", "base_height_cm": "base_height_cm",
        }
        row: dict[str, Any] = {
            "clip": name, "height": str(e.get("height_label") or tags[i]), "hand": hand, "reach_end": re_, "hold_end": he_, "hold_frames": hold_frames,
            "valid_until": int(vu[i]), "clip_len_steps": int(clen[i]),
            "clip_len_original": t_orig, "first_fail_step": int(ff[i]), "fail_free": bool(ff[i] < 0 or ff[i] > vu[i]),
            "orient_family": e.get("orient_family"), "stratum": bench_group_value(e, "stratum"), "tier": bench_group_value(e, "tier"),
            "table": e.get("table"), "windows": {},
        }
        for w, (a, b) in windows.items():
            row["windows"][w] = {"cols": [int(a), int(b)]}
            for k, c in cols.items():
                row["windows"][w][k] = _nanmean(m[c][i, a:b]) if (b > a and c in m) else float("nan")
        if rs_ is not None:   # the rest pose, scored apart: the last final_s of the padded clip
            a, b = max(stop_all - n_final, 0), stop_all
            row["retract_start_frame"] = int(rs_)
            row[REST_WINDOW] = {"cols": [int(a), int(b)], "retract_start_frame": int(rs_)}
            for k, c in REST_METRICS.items():
                row[REST_WINDOW][k] = _nanmean(m[cols[c]][i, a:b]) if (b > a and cols[c] in m) else float("nan")
        row["success"] = clip_success(row, success)
        row["jerk"] = jerk_for_clip(m, i, stop_all, (rl or {}).get("replan_steps") or [], max(int(round(jerk_window_s * fps)), 1))
        if rl:
            row["replan"] = {
                "n_replans": int(rl.get("n_replans") or 0),
                "stayed": rl.get("stayed_at_step") is not None,
                "stayed_at_s": (float(rl["stayed_at_step"]) * dt) if rl.get("stayed_at_step") is not None else None,
                "goal_offset_cm": float(rl.get("goal_offset_cm") or 0.0),
                "ik_time_s": float(rl.get("ik_time_s") or 0.0),
                "mean_terminal_ik_err_cm": rl.get("mean_terminal_ik_err_cm"),
                "final_event_err_cm": (rl.get("events") or [{}])[-1].get("err_cm") if rl.get("events") else None,
            }
        rows.append(row)
    return rows


# ================================================================================================ curves
def aligned_curves(series: Mapping[str, Any], manifest: Mapping[str, dict[str, Any]], *, metrics: Sequence[str] = ("ee_global_active_cm", "ee_rot_global_active_deg", "anchor_xy_cm"),
                   t_from_s: float = -1.0, t_to_s: float = 8.0, stride: int = 5) -> dict[str, Any]:
    """Mean over clips of per-step metrics aligned at each clip's reach_end (t = 0), sampled every ``stride`` steps; a clip
    contributes while the sample is inside its valid window.  For error-vs-time charts."""
    m = series["metrics"]
    dt = float(series["dt"])
    names = [str(x) for x in series["clip_name"]]
    vu = np.asarray(series["valid_until"], dtype=np.int64)
    clen = np.asarray(series["clip_len_steps"], dtype=np.int64)
    offs = np.arange(int(round(t_from_s / dt)), int(round(t_to_s / dt)) + 1, int(stride))
    out: dict[str, Any] = {"t_s": [float(o * dt) for o in offs]}
    for key in metrics:
        acc = np.zeros(len(offs)); cnt = np.zeros(len(offs))
        for i, name in enumerate(names):
            e = entry_for(manifest, name)
            hand = str(e.get("hand", "right")).lower()
            col = {"ee_global_active_cm": f"ee_global_{hand}_cm", "ee_rot_global_active_deg": f"ee_rot_global_{hand}_deg"}.get(key, key)
            if col not in m:
                continue
            re_ = reach_end_of(e)
            stop = int(min(clen[i], vu[i], m[col].shape[1]))
            for j, o in enumerate(offs):
                c = re_ + int(o)
                if 0 <= c < stop and np.isfinite(m[col][i, c]):
                    acc[j] += float(m[col][i, c]); cnt[j] += 1
        out[key] = [float(a / n) if n > 0 else None for a, n in zip(acc, cnt)]
        out[key + "_n"] = [int(n) for n in cnt]
    return out


def clip_success(row: Mapping[str, Any], success: Mapping[str, Any]) -> dict[str, bool]:
    """Closed-loop success flags of one row: fail-free AND the level's window (``final`` unless the protocol says otherwise) ACTIVE-hand world-frame
    palm error <= pos AND world-frame rotation error <= rot (NaN never passes)."""
    out: dict[str, bool] = {}
    for name, thr in (success.get("closed_loop") or {}).items():
        w = str(thr.get("window") or "final")
        win = row["windows"].get(w) or {}
        pos, rot = win.get("ee_global_active_cm", float("nan")), win.get("ee_rot_global_active_deg", float("nan"))
        out[str(name)] = bool(row["fail_free"]) and np.isfinite(pos) and np.isfinite(rot) and pos <= float(thr["pos_cm"]) and rot <= float(thr["rot_deg"])
    return out


def success_block(rs: Sequence[Mapping[str, Any]], success: Mapping[str, Any]) -> dict[str, Any]:
    """Group success block: per closed-loop level ``k / n`` + fraction + Wilson 95 % CI (denominator = every clip of the group), the ``stayed``
    fraction (replan log; None without one) and the CDF points of the ``final``-window active-hand world error."""
    n = len(rs)
    out: dict[str, Any] = {"n": n, "levels": {}, "cdf_cm": {}, "stayed_frac": None, "stayed_threshold_cm": success.get("stayed_threshold_cm")}
    for name, thr in (success.get("closed_loop") or {}).items():
        k = sum(1 for r in rs if r.get("success", {}).get(name))
        lo, hi = wilson_ci(k, n)
        out["levels"][str(name)] = {"k": int(k), "n": n, "frac": (k / n if n else float("nan")), "ci95": [lo, hi], "pos_cm": thr.get("pos_cm"), "rot_deg": thr.get("rot_deg"),
                                    "window": str(thr.get("window") or "final")}
    vals = [r["windows"]["final"]["ee_global_active_cm"] for r in rs]
    for x in success.get("cdf_points_cm", []):
        out["cdf_cm"][f"{float(x):g}"] = (sum(1 for v in vals if np.isfinite(v) and v <= float(x)) / n) if n else float("nan")
    rp = [r["replan"] for r in rs if r.get("replan")]
    if rp:
        out["stayed_frac"] = float(np.mean([bool(x["stayed"]) for x in rp]))
    return out


# ================================================================================================ aggregation
def _stats(values: Sequence[float]) -> dict[str, float | int]:
    x = np.asarray([v for v in values if v is not None and np.isfinite(v)], dtype=np.float64)
    if x.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "p50": float("nan"), "p90": float("nan"), "n": 0}
    return {"mean": float(x.mean()), "std": float(x.std()), "p50": float(np.quantile(x, 0.5)), "p90": float(np.quantile(x, 0.9)), "n": int(x.size)}


def aggregate(rows: Sequence[dict[str, Any]], success: Mapping[str, Any] | None = None) -> dict[str, Any]:
    """Groups: ``all``, every height label of the rows, ``fam:<orient_family>``, and the hero_bench_v1 keys ``stratum:<s>`` / ``tier:<t>`` /
    ``cell:<stratum>/<height>/<hand>`` (hand = left / right / all).  ``heights`` lists the height labels (table columns)."""
    success = success if success is not None else success_protocol(None)
    groups: dict[str, list[dict[str, Any]]] = {"all": list(rows)}
    for r in rows:
        groups.setdefault(r["height"], []).append(r)
    for r in rows:
        if r.get("orient_family"):
            groups.setdefault(f"fam:{r['orient_family']}", []).append(r)
    for r in rows:   # hero_bench_v1 grouping keys (shared with sim2sim.bench.summary.bench_groups)
        st, tier = r.get("stratum"), r.get("tier")
        if st:
            groups.setdefault(f"stratum:{st}", []).append(r)
            groups.setdefault(f"cell:{st}/{r['height']}/{r['hand']}", []).append(r)
            groups.setdefault(f"cell:{st}/{r['height']}/all", []).append(r)
        if tier:
            groups.setdefault(f"tier:{tier}", []).append(r)
    out: dict[str, Any] = {"n_clips": len(rows), "heights": heights_of(rows), "groups": {}}
    for g, rs in groups.items():
        entry: dict[str, Any] = {"n_clips": len(rs), "fail_free_frac": float(np.mean([r["fail_free"] for r in rs])) if rs else float("nan"), "windows": {},
                                 "success": success_block(rs, success)}
        for w in WINDOWS:
            entry["windows"][w] = {k: _stats([r["windows"][w][k] for r in rs]) for k in METRICS}
            entry["windows"][w]["n_with_window"] = int(sum(1 for r in rs if r["windows"][w]["cols"][1] > r["windows"][w]["cols"][0]))
        rest = [r[REST_WINDOW] for r in rs if r.get(REST_WINDOW)]
        if rest:   # own-retract rows (module docstring): the rest pose apart from the reach windows + the bookkeeping caveat
            entry[REST_WINDOW] = {"n_clips": len(rest), "n_with_window": int(sum(1 for x in rest if x["cols"][1] > x["cols"][0])),
                                  **{k: _stats([x[k] for x in rest]) for k in REST_METRICS}}
            entry["notes"] = [RETRACT_NOTE]
        jk = [r["jerk"] for r in rs if r.get("jerk")]
        if jk:
            entry["jerk"] = {
                "n_clips": len(jk),
                "mean": {k: _nanmean([x["mean"][k] for x in jk]) for k in JERK_KEYS},
                "post_replan": {k: (_nanmean([x["post_replan"][k] for x in jk if x["post_replan"][k] is not None]) if any(x["post_replan"][k] is not None for x in jk) else None) for k in JERK_KEYS},
                "n_clips_with_events": int(sum(1 for x in jk if x["n_replans"] > 0)),
            }
        rp = [r["replan"] for r in rs if r.get("replan")]
        if rp:
            entry["replan"] = {
                "n_clips": len(rp),
                "mean_n_replans": float(np.mean([x["n_replans"] for x in rp])),
                "stayed_frac": float(np.mean([x["stayed"] for x in rp])),
                "mean_stayed_at_s": _nanmean([x["stayed_at_s"] for x in rp if x["stayed_at_s"] is not None]) if any(x["stayed_at_s"] is not None for x in rp) else None,
                "mean_goal_offset_cm": float(np.mean([x["goal_offset_cm"] for x in rp])),
                "mean_ik_time_s": float(np.mean([x["ik_time_s"] for x in rp])),
                "mean_terminal_ik_err_cm": _nanmean([x["mean_terminal_ik_err_cm"] for x in rp if x["mean_terminal_ik_err_cm"] is not None]),
            }
        out["groups"][g] = entry
    return out


# ================================================================================================ report
def _fmt(s: Mapping[str, Any], unit_dec: int = 1, std: bool = True) -> str:
    if not s or s.get("n", 0) == 0 or not np.isfinite(s["mean"]):
        return "--"
    return f"{s['mean']:.{unit_dec}f} ± {s['std']:.{unit_dec}f}" if std else f"{s['mean']:.{unit_dec}f}"


def table_heights(heights: Sequence[str] | None) -> list[str]:
    """Height columns of a table from the manifest-derived labels: the labels themselves, except that a run whose labels are all paper-protocol labels keeps the
    full legacy triple :data:`HEIGHTS` (a partial paper-protocol run prints the same columns as before); None / empty -> the triple."""
    hs = sorted(set(heights or ()))
    if not hs or set(hs) <= set(HEIGHTS):
        return list(HEIGHTS)
    return hs


def summary_heights(summary: Mapping[str, Any]) -> list[str]:
    """Height columns of the tables: :func:`table_heights` of the union of ``results[label]["heights"]`` (manifest-derived)."""
    return table_heights([h for lab in summary["labels"] for h in (summary["results"][lab].get("heights") or [])])


def _ci(level: Mapping[str, Any] | None) -> str:
    if not level or not level.get("n") or not np.isfinite(level.get("frac", float("nan"))):
        return "--"
    lo, hi = level.get("ci95", [float("nan"), float("nan")])
    return f"{100.0 * level['frac']:.1f}% [{100.0 * lo:.0f}, {100.0 * hi:.0f}]"


def markdown(summary: dict[str, Any]) -> str:
    labels = summary["labels"]
    heights = summary_heights(summary)
    L = ["# Closed-loop benchmark summary", "",
         f"Manifest `{summary['manifest']}`; windows: hold3 = the row's own hold window ([reach_end, hold_end_frame); --hold-s {summary['hold_s']:g} s only for rows "
         f"without one), final = last {summary['final_s']:g} s before the padded clip end, tail = [reach_end, clip end). Rows with their own retract segment "
         f"(retract_start_frame): final = the last {summary['final_s']:g} s BEFORE retract_start_frame, tail = [reach_end, retract_start_frame), and the rest pose is scored "
         "apart in the `rest` table. Per-clip window means -> across-clip mean ± std.", ""]
    for w in WINDOWS:
        for metric, title, dec in (("ee_global_active_cm", "world-frame palm error of the reaching hand (cm)", 1),
                                   ("ee_rot_global_active_deg", "world-frame palm rotation error of the reaching hand (deg)", 1),
                                   ("ee_local_active_cm", "own-pelvis palm error vs the ORIGINAL clip, reaching hand (cm; not meaningful under replanning)", 1)):
            L += [f"## `{w}` -- {title}", "", "| run | " + " | ".join(heights) + " | all | fail-free |", "|---|" + "---|" * (len(heights) + 2)]
            for lab in labels:
                g = summary["results"][lab]["groups"]
                cells = [_fmt(g[h]["windows"][w][metric], dec) if h in g else "--" for h in heights]
                L.append(f"| {lab} | " + " | ".join(cells) + f" | {_fmt(g['all']['windows'][w][metric], dec)} | {g['all']['fail_free_frac']:.3f} |")
            L.append("")
    sp = summary.get("success_protocol") or {}
    levels = list((sp.get("closed_loop") or {}).keys())
    if levels:
        thr = ", ".join(f"{lv} = {sp['closed_loop'][lv].get('window', 'final')} <= {sp['closed_loop'][lv]['pos_cm']:g} cm & <= {sp['closed_loop'][lv]['rot_deg']:g} deg" for lv in levels)
        cdf_pts = [f"{float(x):g}" for x in sp.get("cdf_points_cm", [])]
        L += [f"## closed-loop success (hero_bench_v1; thresholds from the {sp.get('source', 'default')}: {thr}; fail-free required; Wilson 95 % CI; "
              f"stayed = HERO stop rule <= {sp.get('stayed_threshold_cm', 1.75):g} cm; CDF = final-window active error <= " + " / ".join(cdf_pts) + " cm)", "",
              "| run | group | n | fail-free | " + " | ".join(levels) + " | stayed | CDF |", "|---|---|---|---|" + "---|" * len(levels) + "---|---|"]
        for lab in labels:
            g = summary["results"][lab]["groups"]
            keys = ["all"] + [h for h in heights if h in g] + sorted(k for k in g if (k.startswith("stratum:") and k[len("stratum:"):] not in heights) or k.startswith("tier:"))
            for k in keys:
                sc = g[k].get("success") or {}
                cells = [str(g[k]["n_clips"]), f"{g[k]['fail_free_frac']:.3f}"] + [_ci(sc.get("levels", {}).get(lv)) for lv in levels]
                cells.append("--" if sc.get("stayed_frac") is None else f"{sc['stayed_frac']:.2f}")
                cdf = sc.get("cdf_cm", {})
                cells.append(" / ".join(f"{100.0 * cdf[c]:.0f}%" for c in cdf) if cdf else "--")
                L.append(f"| {lab} | {k} | " + " | ".join(cells) + " |")
        L.append("")
    L += ["## `final` window -- all metrics (all heights)", "", "| run | glob act | rot(world) act | loc act (vs orig) | rot(pelvis) act | glob both | rot(world) both | arm rad (vs orig) | root xy | root dz |", "|---|---|---|---|---|---|---|---|---|---|"]
    for lab in labels:
        a = summary["results"][lab]["groups"]["all"]["windows"]["final"]
        L.append(f"| {lab} | {_fmt(a['ee_global_active_cm'], 1, False)} | {_fmt(a['ee_rot_global_active_deg'], 1, False)} | {_fmt(a['ee_local_active_cm'], 1, False)} | {_fmt(a['ee_rot_active_deg'], 1, False)} | "
                 f"{_fmt(a['ee_global_cm'], 1, False)} | {_fmt(a['ee_rot_global_deg'], 1, False)} | {_fmt(a['joint_upper_rad'], 3, False)} | "
                 f"{_fmt(a['anchor_xy_cm'], 1, False)} | {_fmt(a['base_height_cm'], 1, False)} |")
    if any(summary["results"][lab]["groups"]["all"].get(REST_WINDOW) for lab in labels):
        L += ["", f"## `rest` window -- own-retract rows only: world-frame error of the reaching hand over the last {summary['final_s']:g} s of the padded clip "
                  "(the rest pose after the hand-back; not part of `final` / C3)", "",
              "| run | group | n retract rows | rest pos err cm | rest rot err deg |", "|---|---|---|---|---|"]
        for lab in labels:
            g = summary["results"][lab]["groups"]
            for k in ["all"] + sorted(k for k in g if k != "all" and g[k].get(REST_WINDOW) and (k in heights or k.startswith("stratum:") or k.startswith("tier:"))):
                rt = g[k].get(REST_WINDOW)
                if rt:
                    L.append(f"| {lab} | {k} | {rt['n_clips']} | {_fmt(rt['rest_pos_err_cm'], 1)} | {_fmt(rt['rest_rot_err_deg'], 1)} |")
        notes = sorted({n for lab in labels for grp in summary["results"][lab]["groups"].values() for n in (grp.get("notes") or [])})
        L += [""] + [f"Note: {n}." for n in notes]
    L += ["", "## replanner bookkeeping (all heights)", "", "| run | replans / clip | converged (stay) | mean stay time s | goal offset cm | IK s / clip | IK terminal err cm |", "|---|---|---|---|---|---|---|"]
    for lab in labels:
        rp = summary["results"][lab]["groups"]["all"].get("replan")
        if not rp:
            L.append(f"| {lab} | -- | -- | -- | -- | -- | -- |")
            continue
        st = f"{rp['mean_stayed_at_s']:.1f}" if rp.get("mean_stayed_at_s") is not None else "--"
        L.append(f"| {lab} | {rp['mean_n_replans']:.1f} | {rp['stayed_frac']:.2f} | {st} | {rp['mean_goal_offset_cm']:.1f} | {rp['mean_ik_time_s']:.1f} | {rp['mean_terminal_ik_err_cm']:.2f} |")
    if any(summary["results"][lab]["groups"]["all"].get("jerk") for lab in labels):
        L += ["", f"## jerk -- arm PD-target discontinuity / measured arm joint acceleration / live-reference palm speed (all heights; all valid steps vs the {summary.get('jerk_window_s', 0.3):g} s after each replan)", "",
              "| run | target delta mrad/step (all) | (post-replan) | arm accel rad/s2 (all) | (post-replan) | ref palm speed cm/s (all) | (post-replan) | clips w/ events |", "|---|---|---|---|---|---|---|---|"]
        for lab in labels:
            jk = summary["results"][lab]["groups"]["all"].get("jerk")
            if not jk:
                L.append(f"| {lab} | -- | -- | -- | -- | -- | -- | -- |")
                continue

            def _v(block, k, scale=1.0, dec=2):
                v = jk[block].get(k)
                return "--" if v is None or not np.isfinite(v) else f"{scale * v:.{dec}f}"

            L.append(f"| {lab} | {_v('mean', 'arm_target_delta_rad', 1000.0)} | {_v('post_replan', 'arm_target_delta_rad', 1000.0)} | {_v('mean', 'arm_accel_rad_s2')} | {_v('post_replan', 'arm_accel_rad_s2')} | "
                     f"{_v('mean', 'ref_palm_speed_cm_s', 1.0, 1)} | {_v('post_replan', 'ref_palm_speed_cm_s', 1.0, 1)} | {jk['n_clips_with_events']}/{jk['n_clips']} |")
    fams = sorted({g for lab in labels for g in summary["results"][lab]["groups"] if g.startswith("fam:")})
    if fams:
        L += ["", "## `final` window -- world-frame error of the reaching hand by orientation family (cm)", "", "| run | " + " | ".join(f[4:] for f in fams) + " |", "|---|" + "---|" * len(fams)]
        for lab in labels:
            g = summary["results"][lab]["groups"]
            L.append(f"| {lab} | " + " | ".join(_fmt(g[f]["windows"]["final"]["ee_global_active_cm"], 1) if f in g else "--" for f in fams) + " |")
    return "\n".join(L) + "\n"


# ================================================================================================ flat table (CSV)
def flat_row(label: str, result: Mapping[str, Any]) -> dict[str, Any]:
    """One flat row of scalars for a series label (``summary["results"][label]``): reach windows of the reaching hand (mean / p90 of the
    world-frame error, ``tail`` / ``final`` / ``hold3`` overall and ``final`` per height), fail-free fraction, replans / clip, the success
    fractions and the jerk means (all -> post replan).  Missing blocks give None (empty CSV cells)."""
    g = result["groups"]
    a = g["all"]
    row: dict[str, Any] = {"label": label, "n_clips": a["n_clips"], "fail_free": a["fail_free_frac"]}
    for w in ("tail", "final", "hold3"):
        st = a["windows"][w]["ee_global_active_cm"]
        row[f"{w}_ee_global_active_cm_mean"] = st["mean"]
        row[f"{w}_ee_global_active_cm_p90"] = st["p90"]
    row["final_ee_rot_global_active_deg_mean"] = a["windows"]["final"]["ee_rot_global_active_deg"]["mean"]
    row["final_anchor_xy_cm_mean"] = a["windows"]["final"]["anchor_xy_cm"]["mean"]
    heights = table_heights(result.get("heights"))
    for h in heights:
        row[f"final_ee_global_active_cm_{h}"] = g[h]["windows"]["final"]["ee_global_active_cm"]["mean"] if h in g else None
    rp = a.get("replan")
    row["replans_per_clip"] = rp["mean_n_replans"] if rp else None
    row["stayed_frac"] = rp["stayed_frac"] if rp else None
    for lv, st in ((a.get("success") or {}).get("levels") or {}).items():   # hero_bench_v1 success fractions
        row[f"success_{lv}"] = st["frac"]
    jk = a.get("jerk")
    for k in JERK_KEYS:
        row[f"jerk_{k}_all"] = jk["mean"].get(k) if jk else None
        row[f"jerk_{k}_post_replan"] = jk["post_replan"].get(k) if jk else None
    rt = a.get(REST_WINDOW)   # own-retract rows: the rest pose apart (None without such rows)
    row["rest_n_clips"] = rt["n_clips"] if rt else None
    for k in REST_METRICS:
        row[f"{k}_mean"] = rt[k]["mean"] if rt else None
    return row


def flat_rows(summary: Mapping[str, Any]) -> list[dict[str, Any]]:
    return [flat_row(lab, summary["results"][lab]) for lab in summary["labels"]]


def write_csv(summary: Mapping[str, Any], path: str | os.PathLike, *, decimals: int = 4) -> Path:
    """``flat_rows`` -> CSV (one row per series label, NaN / None -> empty cell, floats rounded to ``decimals``)."""
    import csv

    rows = flat_rows(summary)
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    cols = list(rows[0].keys()) if rows else ["label"]
    for r in rows[1:]:   # labels whose manifest rows span other heights add their columns at the end
        cols += [k for k in r if k not in cols]
    with open(path, "w", newline="") as fh:
        w = csv.DictWriter(fh, fieldnames=cols)
        w.writeheader()
        for r in rows:
            out = {}
            for k, v in r.items():
                if v is None or (isinstance(v, float) and not np.isfinite(v)):
                    out[k] = ""
                elif isinstance(v, str):
                    out[k] = v
                elif isinstance(v, (bool, np.bool_)):
                    out[k] = int(v)
                elif isinstance(v, (int, np.integer)):
                    out[k] = int(v)
                else:
                    out[k] = round(float(v), decimals)
            w.writerow(out)
    return path


def summarize(series_specs: Sequence[tuple[str, str]], manifest_path: str, *, hold_s: float = 3.0, final_s: float = 1.0, keep_per_clip: bool = True, jerk_window_s: float = 0.3) -> dict[str, Any]:
    manifest = load_manifest(manifest_path)
    success = success_protocol(load_manifest_protocol(manifest_path))
    out: dict[str, Any] = {"schema": "hero_closed_loop_summary_v1", "manifest": str(manifest_path), "hold_s": hold_s, "final_s": final_s, "jerk_window_s": jerk_window_s,
                           "success_protocol": success, "labels": [], "results": {}, "series": {}, "per_clip": {}, "curves": {}}
    for label, path in series_specs:
        s = read_series(path)
        meta = json.loads(str(s["meta_json"])) if "meta_json" in s else {}
        rows = per_clip_rows(s, manifest, hold_s=hold_s, final_s=final_s, jerk_window_s=jerk_window_s, success=success)
        out["labels"].append(label)
        out["results"][label] = aggregate(rows, success)
        out["curves"][label] = aligned_curves(s, manifest)
        out["series"][label] = {"path": str(path), "sim": str(s.get("sim", "")), "policy_kind": meta.get("policy_kind"), "tag": meta.get("tag"),
                                "replan": meta.get("replan"), "pad_s": meta.get("pad_s"), "horizon_steps": int(s["horizon_steps"]), "n_clips": len(rows),
                                # provenance the side-by-side report / results card reads when no open-loop summary is given
                                "meta": {k: meta.get(k) for k in ("policy", "odometry", "termination", "fail_causes", "fall_low_ref_margin_m") if meta.get(k) is not None}}
        if keep_per_clip:
            out["per_clip"][label] = [{"clip": r["clip"], "height": r["height"], "hand": r["hand"], "fail_free": r["fail_free"], "orient_family": r.get("orient_family"),
                                       "stratum": r.get("stratum"), "tier": r.get("tier"), "hold_frames": r.get("hold_frames"), "success": r.get("success"),
                                       **{f"{w}_{k}": r["windows"][w][k] for w in WINDOWS for k in ("ee_global_active_cm", "ee_local_active_cm", "ee_rot_active_deg", "ee_rot_global_active_deg")},
                                       "replan": r.get("replan"), "jerk": r.get("jerk"), **({REST_WINDOW: r[REST_WINDOW]} if r.get(REST_WINDOW) else {})} for r in rows]
    return out


def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--series", action="append", required=True, help='"label|path/to/series.npz" (repeatable, order = table order)')
    ap.add_argument("--manifest", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--hold-s", type=float, default=3.0, help="hold3 window length ONLY for manifest rows without hold_end_frame / hold_frames (hero_bench_v1 rows carry their own)")
    ap.add_argument("--final-s", type=float, default=1.0)
    ap.add_argument("--no-per-clip", action="store_true")
    ap.add_argument("--jerk-window-s", type=float, default=0.3, help="window after every replan step for the post-replan jerk means (sim2sim.bench.rollout.POST_REPLAN_WINDOW_S)")
    ap.add_argument("--csv", default=None, help="also write one flat CSV row per series label (reach windows, fail-free, replans, success, jerk columns; flat_row)")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    specs = []
    for s in a.series:
        if "|" not in s:
            ap.error(f"--series needs label|path, got {s!r}")
        lab, path = s.split("|", 1)
        specs.append((lab.strip(), path.strip()))
    try:
        summary = summarize(specs, a.manifest, hold_s=a.hold_s, final_s=a.final_s, keep_per_clip=not a.no_per_clip, jerk_window_s=a.jerk_window_s)
    except ValueError as exc:
        print(f"[bench] {exc}", file=sys.stderr)
        return 1
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "closed_loop_summary.json").write_text(json.dumps(summary, indent=1, default=str))
    md = markdown(summary)
    (out / "closed_loop_summary.md").write_text(md)
    if a.csv:
        write_csv(summary, a.csv)
    print(md)
    return 0


if __name__ == "__main__":
    sys.exit(main())
