"""``hero_eval_series_v1`` series file + the fixed-horizon aggregation.

npz layout::

    schema="hero_eval_series_v1"  sim="mujoco"|"isaacsim"  dt (float)  horizon_steps (int)  meta_json (str)
    clip_name (C,) str   source_tag (C,) str   clip_len_steps (C,) int64   valid_until (C,) int64
    first_fail_step (C,) int64 (1-based, -1 never)   first_fail_cause (C,) str ("" = none)
    m/<key> (C, T) float32, NaN where invalid, keys = sim2sim.bench.metrics.METRIC_KEYS (+ optional odom_* / jerk columns)

Step ``k`` (1-based) of a clip is valid iff ``k <= valid_until`` (``valid_until = min(T, clip_len - 1)``); ``alive_until``
= last step before the first training failure.  Aggregates report per-clip HORIZON AVERAGES (mean over valid steps)
reduced over clips as mean / p50 / p90 (``stats``), the fail-free fraction (``survived_valid_window``: no failure inside
the valid window), the same statistics over pre-failure steps only (``metrics_alive``), survival per second and the
first-failure cause histogram; overall and per ``source_tag``.  Both simulators write this layout; :func:`read_series` accepts
either producer.

Odometry columns (``m/odom_*``, :data:`sim2sim.bench.metrics.ODOM_METRIC_KEYS`; present only when the runner ran an estimator, ``--odom so``):
:func:`summarize` adds an ``odometry`` block (per clip ``mean_abs`` over the valid steps and the value at the clip END = last valid column;
overall + per source mean / p90 over clips) and :func:`summary_markdown` a section + an ``odometry`` bullet in the header -- new keys / lines
only, emitted solely when the estimator ran: series without the columns summarise exactly as before.

Full-body keypoint columns (``m/kp_*`` + ``m/sonic_fail``, :data:`sim2sim.bench.metrics.KEYPOINT_METRIC_KEYS`; OPTIONAL in :func:`build_series`):
the ``kp_*`` columns aggregate like every other metric; the ``sonic_fail`` column additionally yields the per-clip success of the SONIC protocol
(:func:`sonic_success_per_clip`: never failed inside the valid window) -> ``sonic_success_fraction`` / ``sonic_success_count`` /
``sonic_success_alive_fraction`` per group and ``sonic_success`` / ``sonic_first_fail_step`` per clip, a ``keypoints`` provenance block
(:func:`sim2sim.bench.metrics.keypoint_definitions` + :func:`sonic_success_block`: ``n`` evaluated clips, ``sonic_success`` COUNT,
``sonic_success_frac``, ``by_source``, ``sonic_fail_rule`` -- the same aggregate names the Isaac Sim harness writes) and the markdown section
"Full-body keypoints".
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from sim2sim.bench.metrics import KEYPOINT_METRIC_KEYS, METRIC_KEYS, ODOM_METRIC_KEYS, keypoint_definitions

SCHEMA = "hero_eval_series_v1"
NO_FAILURE_CAUSE = ""
DEFAULT_TIMES_S: tuple[float, ...] = (1.0, 2.0, 3.0, 5.0, 10.0)


# ------------------------------------------------------------------------------------------------ build / io
def build_series(
    *,
    sim: str,
    dt: float,
    horizon_steps: int,
    meta: Mapping[str, Any],
    clip_name: Sequence[str],
    source_tag: Sequence[str],
    clip_len_steps: Sequence[int],
    valid_until: Sequence[int],
    first_fail_step: Sequence[int],
    first_fail_cause: Sequence[str],
    metrics: Mapping[str, np.ndarray],
) -> dict[str, Any]:
    """The npz dict (module docstring)."""
    n = len(clip_name)
    T = int(horizon_steps)
    out: dict[str, Any] = {
        "schema": np.array(SCHEMA),
        "sim": np.array(str(sim)),
        "dt": np.array(float(dt)),
        "horizon_steps": np.array(T, dtype=np.int64),
        "meta_json": np.array(json.dumps(dict(meta), default=str)),
        "clip_name": np.asarray(list(clip_name), dtype=str),
        "source_tag": np.asarray(list(source_tag), dtype=str),
        "clip_len_steps": np.asarray(list(clip_len_steps), dtype=np.int64),
        "valid_until": np.asarray(list(valid_until), dtype=np.int64),
        "first_fail_step": np.asarray(list(first_fail_step), dtype=np.int64),
        "first_fail_cause": np.asarray(list(first_fail_cause), dtype=str),
    }
    # the keypoint columns are optional: a writer without them produces the previous file byte for byte
    required = [k for k in METRIC_KEYS if k not in KEYPOINT_METRIC_KEYS or k in metrics]
    for k in required + [k for k in metrics if k not in METRIC_KEYS]:  # extra m/* keys (odometry / jerk columns) are allowed
        v = np.asarray(metrics[k], dtype=np.float32)
        if v.shape != (n, T):
            raise ValueError(f"metric {k}: shape {v.shape} != {(n, T)}")
        out[f"m/{k}"] = v
    return out


def write_series(path: str | Path, series: Mapping[str, Any]) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(path, **series)
    return path


def read_series(path: str | Path) -> dict[str, Any]:
    """Load a series file: scalars as Python values, per-clip arrays as 1-d numpy arrays, the metrics as ``(C, T)`` float32 arrays under
    ``metrics[name]`` (prefix stripped) and the decoded ``meta_json`` under ``meta``."""
    with np.load(path, allow_pickle=False) as d:
        out: dict[str, Any] = {}
        for k in d.files:
            v = d[k]
            out[k] = v.item() if v.ndim == 0 else v
    if str(out.get("schema")) != SCHEMA:
        raise ValueError(f"{path}: schema {out.get('schema')!r} != {SCHEMA!r}")
    out["metrics"] = {k[2:]: out[k] for k in list(out) if k.startswith("m/")}
    try:
        out["meta"] = json.loads(str(out.get("meta_json", "{}") or "{}"))
    except json.JSONDecodeError:
        out["meta"] = {}
    return out


# ------------------------------------------------------------------------------------------------ aggregation
def stats(x: np.ndarray) -> dict[str, float | int]:
    x = np.asarray(x, dtype=np.float64).reshape(-1)
    x = x[np.isfinite(x)]
    if x.size == 0:
        return {"mean": float("nan"), "p50": float("nan"), "p90": float("nan"), "n": 0}
    return {"mean": float(x.mean()), "p50": float(np.quantile(x, 0.5)), "p90": float(np.quantile(x, 0.9)), "n": int(x.size)}


def horizon_average(values: np.ndarray, until: np.ndarray) -> np.ndarray:
    """Per-clip mean over steps ``1..until`` (NaN when none).  ``values (C, T)``."""
    C, T = values.shape
    steps = np.arange(1, T + 1)[None, :]
    valid = (steps <= np.asarray(until).reshape(C, 1)) & np.isfinite(values)
    num = np.where(valid, values, 0.0).sum(axis=1)
    cnt = valid.sum(axis=1)
    with np.errstate(invalid="ignore", divide="ignore"):
        out = num / np.maximum(cnt, 1)
    return np.where(cnt > 0, out, np.nan)


def alive_until_steps(valid_until: np.ndarray, first_fail_step: np.ndarray) -> np.ndarray:
    ff = np.asarray(first_fail_step)
    vu = np.asarray(valid_until)
    return np.where(ff < 0, vu, np.minimum(vu, np.maximum(ff - 1, 0)))


def sonic_success_per_clip(sonic_fail: np.ndarray, until: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """Per-clip success from the 0 / 1 ``sonic_fail`` column ``(C, T)``: ``success`` = 1.0 when no valid step ``1..until`` failed, 0.0 when
    one did, NaN without a valid finite step; ``first_fail_step`` = the first failing step (1-based, -1 = never inside the window)."""
    v = np.asarray(sonic_fail, dtype=np.float64)
    C, T = v.shape
    steps = np.arange(1, T + 1)[None, :]
    valid = (steps <= np.asarray(until).reshape(C, 1)) & np.isfinite(v)
    fail = valid & (v > 0.5)
    any_fail = fail.any(axis=1)
    first = np.where(any_fail, fail.argmax(axis=1) + 1, -1).astype(np.int64)
    success = np.where(valid.any(axis=1), np.where(any_fail, 0.0, 1.0), np.nan)
    return success, first


def sonic_success_block(success: np.ndarray, source_tag: Sequence[str] | np.ndarray | None = None) -> dict[str, Any]:
    """The success aggregate of the SONIC protocol from the per-clip ``success`` of :func:`sonic_success_per_clip` (1 / 0 / NaN = not
    evaluated): ``n`` = evaluated clips (a finite success), ``sonic_success`` = their COUNT of successes, ``sonic_success_frac`` = count / n (NaN
    when nothing was evaluated), ``by_source`` = the same three per source tag (a tag with no evaluated clip is absent).  Lives in
    ``summarize()["keypoints"]`` next to the provenance, with the same aggregate names the Isaac Sim harness writes."""
    s = np.asarray(success, dtype=np.float64).reshape(-1)
    evaluated = np.isfinite(s)

    def _block(mask: np.ndarray) -> dict[str, Any]:
        m = mask & evaluated
        n = int(m.sum())
        k = int(np.sum(s[m] > 0.5))
        return {"n": n, "sonic_success": k, "sonic_success_frac": (k / n) if n else float("nan")}

    out: dict[str, Any] = _block(np.ones(s.shape[0], dtype=bool))
    out["by_source"] = {}
    if source_tag is not None:
        tags = np.asarray([str(t) for t in source_tag], dtype=str)
        if tags.shape != s.shape:
            raise ValueError(f"sonic_success_block: {tags.shape[0]} source tags for {s.shape[0]} clips")
        for tag in sorted(set(tags.tolist())):
            b = _block(tags == tag)
            if b["n"]:
                out["by_source"][tag] = b
    return out


def _finite_mean(x: np.ndarray) -> float:
    x = np.asarray(x, dtype=np.float64)
    x = x[np.isfinite(x)]
    return float(x.mean()) if x.size else float("nan")


def _group(metrics: Mapping[str, np.ndarray], valid_until, first_fail_step, first_fail_cause, mask, dt: float, times_s: Sequence[float]) -> dict[str, Any]:
    vu = np.asarray(valid_until)[mask]
    ff = np.asarray(first_fail_step)[mask]
    causes = np.asarray(first_fail_cause)[mask]
    T = next(iter(metrics.values())).shape[1]
    au = alive_until_steps(vu, ff)
    has_any = vu >= 1
    survived = (ff < 0) | (ff > vu)
    entry: dict[str, Any] = {
        "n_clips": int(mask.sum()),
        "n_clips_valid": int(has_any.sum()),
        "mean_valid_steps": float(vu[has_any].mean()) if has_any.any() else 0.0,
        "mean_alive_steps": float(au[has_any].mean()) if has_any.any() else 0.0,
        "fail_free_fraction": float(survived[has_any].mean()) if has_any.any() else float("nan"),
        "metrics": {},
        "metrics_alive": {},
        "first_fail_cause_counts": {},
        "survival_at_s": {},
        "at_time": {},
    }
    for k, v in metrics.items():
        sub = np.asarray(v)[mask]
        entry["metrics"][k] = stats(horizon_average(sub, vu)[has_any])
        entry["metrics_alive"][k] = stats(horizon_average(sub, au)[au >= 1])
    uniq, cnt = np.unique(causes[has_any], return_counts=True)
    entry["first_fail_cause_counts"] = {(str(u) if str(u) else "none"): int(c) for u, c in zip(uniq, cnt)}
    for s in range(1, int(math.floor(T * dt + 1e-9)) + 1):
        k = int(round(s / dt))
        env_mask = vu >= k
        if env_mask.sum() == 0:
            continue
        alive = ((ff < 0) | (ff > k)) & env_mask
        entry["survival_at_s"][f"{s}"] = float(alive.sum() / env_mask.sum())
    for t in times_s:
        k = int(round(t / dt))
        if k < 1 or k > T:
            continue
        env_mask = vu >= k
        alive_mask = au >= k
        at: dict[str, Any] = {"step": k, "n_valid": int(env_mask.sum()), "n_alive": int(alive_mask.sum()), "metrics": {}, "metrics_alive": {}}
        for kk, v in metrics.items():
            col = np.asarray(v)[mask][:, k - 1]
            at["metrics"][kk] = stats(col[env_mask])
            at["metrics_alive"][kk] = stats(col[alive_mask])
        entry["at_time"][f"{t:g}"] = at
    if "ee_global_cm" in entry["metrics"] and "ee_local_cm" in entry["metrics"]:
        entry["gap_global_minus_local_cm"] = entry["metrics"]["ee_global_cm"]["mean"] - entry["metrics"]["ee_local_cm"]["mean"]
    if "sonic_fail" in metrics:  # success of the SONIC protocol = never failed inside the window; new keys only when the column exists
        sf = np.asarray(metrics["sonic_fail"])[mask]
        s_valid, _ = sonic_success_per_clip(sf, vu)
        s_alive, _ = sonic_success_per_clip(sf, au)
        entry["sonic_success_fraction"] = _finite_mean(s_valid)
        entry["sonic_success_count"] = int(np.sum(s_valid[np.isfinite(s_valid)] > 0.5))
        entry["sonic_success_alive_fraction"] = _finite_mean(s_alive)
    return entry


def end_values(values: np.ndarray, until: np.ndarray) -> np.ndarray:
    """Per-clip value at the clip END = the last finite column among steps ``1..until`` (NaN when none).  ``values (C, T)``."""
    C, T = values.shape
    steps = np.arange(1, T + 1)[None, :]
    valid = (steps <= np.asarray(until).reshape(C, 1)) & np.isfinite(values)
    out = np.full(C, np.nan)
    for i in range(C):
        idx = np.flatnonzero(valid[i])
        if idx.size:
            out[i] = values[i, idx[-1]]
    return out


def odometry_summary(metrics: Mapping[str, np.ndarray], valid_until, source_tag) -> dict[str, Any] | None:
    """The ``odometry`` block of :func:`summarize` (None when the series carries no ``odom_*`` column): per key the across-clip mean / p50 /
    p90 of the per-clip MEAN |error| over the valid steps and of the |error| at the clip END (signed end mean too, for the yaw / z bias);
    overall and per ``source_tag``; per clip ``mean_abs`` / ``end``.  Keys in cm / deg as the columns."""
    keys = [k for k in ODOM_METRIC_KEYS if k in metrics]
    if not keys:
        return None
    vu = np.asarray(valid_until)
    tags = np.asarray(source_tag, dtype=str)
    n = tags.shape[0]
    per_clip_mean = {k: horizon_average(np.abs(np.asarray(metrics[k], dtype=np.float64)), vu) for k in keys}
    per_clip_end = {k: end_values(np.asarray(metrics[k], dtype=np.float64), vu) for k in keys}

    def group(mask: np.ndarray) -> dict[str, Any]:
        g: dict[str, Any] = {"n_clips": int(mask.sum()), "keys": {}}
        for k in keys:
            e = per_clip_end[k][mask]
            g["keys"][k] = {"mean_abs": stats(per_clip_mean[k][mask]), "end_abs": stats(np.abs(e)), "end_signed_mean": (float(np.nanmean(e)) if np.isfinite(e).any() else float("nan"))}
        if "odom_stance_feet" in metrics:
            st = np.asarray(metrics["odom_stance_feet"], dtype=np.float64)[mask]
            fin = np.isfinite(st)
            g["stance_fraction"] = float(np.mean(st[fin] > 0)) if fin.any() else float("nan")
        return g

    return {
        "keys": keys,
        "overall": group(np.ones(n, dtype=bool)),
        "by_source": {tag: group(tags == tag) for tag in sorted(set(tags.tolist()))},
        "clips": [{k: {"mean_abs": (None if not np.isfinite(per_clip_mean[k][i]) else float(per_clip_mean[k][i])), "end": (None if not np.isfinite(per_clip_end[k][i]) else float(per_clip_end[k][i]))} for k in keys} for i in range(n)],
    }


def stats_std(x: Sequence[float] | np.ndarray) -> dict[str, float | int]:
    """:func:`stats` plus the standard deviation (the closed-loop tables print ``mean ± std``); None / non-finite values are dropped."""
    a = np.asarray([v for v in np.asarray(x, dtype=object).reshape(-1).tolist() if v is not None], dtype=np.float64).reshape(-1)
    a = a[np.isfinite(a)]
    if a.size == 0:
        return {"mean": float("nan"), "std": float("nan"), "p50": float("nan"), "p90": float("nan"), "n": 0}
    return {"mean": float(a.mean()), "std": float(a.std()), "p50": float(np.quantile(a, 0.5)), "p90": float(np.quantile(a, 0.9)), "n": int(a.size)}


def summarize(series: Mapping[str, Any], times_s: Sequence[float] = DEFAULT_TIMES_S) -> dict[str, Any]:
    """Overall + per-source aggregation of a (read) series dict (+ the ``odometry`` block when the series has ``odom_*`` columns, + the
    ``keypoints`` block when it carries the keypoint columns -- module docstring)."""
    metrics = series["metrics"] if "metrics" in series else {k[2:]: series[k] for k in series if k.startswith("m/")}
    dt = float(series["dt"])
    tags = np.asarray(series["source_tag"], dtype=str)
    n = tags.shape[0]
    vu, ff, fc = series["valid_until"], series["first_fail_step"], series["first_fail_cause"]
    allmask = np.ones(n, dtype=bool)
    out: dict[str, Any] = {
        "schema": "hero_mujoco_eval_summary_v1",
        "sim": str(series.get("sim", "")),
        "dt": dt,
        "horizon_steps": int(series["horizon_steps"]),
        "meta": json.loads(str(series["meta_json"])) if series.get("meta_json") is not None else {},
        "overall": _group(metrics, vu, ff, fc, allmask, dt, times_s),
        "by_source": {},
        "clips": [],
    }
    for tag in sorted(set(tags.tolist())):
        out["by_source"][tag] = _group(metrics, vu, ff, fc, tags == tag, dt, times_s)
    odo = odometry_summary(metrics, vu, tags)
    if odo is not None:
        out["odometry"] = odo
    sonic = sonic_success_per_clip(np.asarray(metrics["sonic_fail"]), np.asarray(vu)) if "sonic_fail" in metrics else None
    if any(k in metrics for k in KEYPOINT_METRIC_KEYS):  # keypoint columns present: provenance of the body sets / rule (new key only)
        out["keypoints"] = keypoint_definitions()
        if sonic is not None:
            out["keypoints"].update(sonic_success_block(sonic[0], tags))
    havg = {k: horizon_average(np.asarray(v), np.asarray(vu)) for k, v in metrics.items()}
    for i in range(n):
        out["clips"].append(
            {
                "clip_name": str(series["clip_name"][i]),
                "source_tag": str(tags[i]),
                "clip_len_steps": int(series["clip_len_steps"][i]),
                "valid_until": int(vu[i]),
                "first_fail_step": int(ff[i]),
                "first_fail_cause": str(fc[i]),
                "horizon_avg": {k: (None if not np.isfinite(havg[k][i]) else float(havg[k][i])) for k in metrics},
                **({"odometry": odo["clips"][i]} if odo is not None else {}),
                **({"sonic_success": (None if not np.isfinite(sonic[0][i]) else bool(sonic[0][i] > 0.5)), "sonic_first_fail_step": int(sonic[1][i])} if sonic is not None else {}),
            }
        )
    return out


# ------------------------------------------------------------------------------------------------ markdown
_MD_METRICS = (
    ("ee_local_cm", "local EE cm", 2),
    ("ee_global_cm", "global EE cm", 2),
    ("ee_rot_deg", "EE rot deg", 1),
    ("joint_upper_rad", "arm joint rad", 3),
    ("joint_upper17_rad", "upper17 joint rad", 3),
    ("anchor_xy_cm", "anchor xy cm", 2),
    ("anchor_pos_cm", "anchor pos cm", 2),
    ("base_height_cm", "base h cm", 2),
)


#: "Full-body keypoints" table (column, label, digits); emitted only when the series carries the keypoint columns.
_MD_KP = (
    ("kp_global_pos_cm", "kp global pos cm", 2),
    ("kp_global_rot_deg", "kp global rot deg", 1),
    ("kp_local_pos_cm", "kp local pos cm", 2),
    ("kp_local_rot_deg", "kp local rot deg", 1),
    ("kp14_global_pos_mm", "kp14 global mm", 1),
    ("kp14_local_pos_mm", "kp14 local mm (MPJPE-L)", 1),
    ("kp_group_feet_global_pos_cm", "feet global cm", 2),
    ("kp_group_hands_global_pos_cm", "hands global cm", 2),
)


_MD_ODOM = (
    ("odom_xy_err_cm", "odom xy cm", 2),
    ("odom_pos_err_cm", "odom pos cm", 2),
    ("odom_yaw_err_deg", "odom yaw deg", 2),
    ("odom_z_err_cm", "odom z cm", 2),
    ("odom_vel_err_cm_s", "odom vel cm/s", 2),
)


def _fmt(s: Mapping[str, Any] | None, d: int) -> str:
    if not s or s.get("n", 0) == 0 or not np.isfinite(s.get("mean", float("nan"))):
        return "-"
    return f"{s['mean']:.{d}f} / {s['p50']:.{d}f} / {s['p90']:.{d}f}"


def summary_markdown(summary: Mapping[str, Any]) -> str:
    meta = summary.get("meta", {})
    ov = summary["overall"]
    lines = [
        f"# MuJoCo sim2sim eval - {meta.get('policy', {}).get('tag', '?')}",
        "",
        f"- sim: {summary.get('sim')}  dt={summary.get('dt')}  horizon={summary.get('horizon_steps')} steps  "
        f"physics_dt={meta.get('plant', {}).get('physics_dt')}  substeps={meta.get('plant', {}).get('substeps')}",
        f"- onnx: `{meta.get('policy', {}).get('encoder') or meta.get('policy', {}).get('onnx')}`"
        + (f" / `{meta.get('policy', {}).get('decoder')}`" if meta.get('policy', {}).get('decoder') else ""),
        f"- clips: {ov['n_clips']}  mean valid steps {ov['mean_valid_steps']:.0f}  mean pre-failure steps {ov['mean_alive_steps']:.0f}  "
        f"fail-free fraction {ov['fail_free_fraction']:.3f}",
        f"- first-failure causes: {ov['first_fail_cause_counts']}",
    ]
    odom_meta = meta.get("odometry")
    odom_is_lio = isinstance(odom_meta, Mapping) and str(odom_meta.get("source")) == "so"
    if isinstance(odom_meta, Mapping) and odom_meta.get("config") is not None:  # only when the estimator ran: a truth run's summary.md is unchanged
        oc = odom_meta.get("config") or {}
        if odom_is_lio:
            lines.append(f"- odometry: source=so  preset={odom_meta.get('preset')}  rate_hz={oc.get('rate_hz')}  latency_s={oc.get('latency_s')}  output_delay_s={oc.get('output_delay_s')}  propagation={oc.get('propagation')}  "
                         f"drift_pct={oc.get('drift_pct')}  bias_xy_m={oc.get('bias_xy_m')}  fed_to_policy={odom_meta.get('fed_to_policy')}  policy_terms={odom_meta.get('policy_terms')}")
        else:
            lines.append(f"- odometry: source={odom_meta.get('source')}  noise={odom_meta.get('noise')}  fed_to_policy={odom_meta.get('fed_to_policy')}  "
                         f"policy_terms={odom_meta.get('policy_terms')}  contact_threshold_n={oc.get('contact_threshold_n')}")
    lines += [
        "",
        "## Overall (per-clip horizon averages; mean / p50 / p90 over clips)",
        "",
        "| metric | all valid steps | pre-failure steps only |",
        "|---|---|---|",
    ]
    for key, label, d in _MD_METRICS:
        lines.append(f"| {label} | {_fmt(ov['metrics'].get(key), d)} | {_fmt(ov['metrics_alive'].get(key), d)} |")
    if ov.get("survival_at_s"):
        lines += ["", "survival (no training failure) per second: " + ", ".join(f"{s}s {v:.2f}" for s, v in ov["survival_at_s"].items())]
    odo = summary.get("odometry")
    if odo:
        title = "LiDAR-inertial odometry model" if odom_is_lio else "Odometry"
        lines += ["", f"## {title} (estimate vs truth; per-clip |error| mean over valid steps and at the clip end; mean / p50 / p90 over clips)", "",
                  "| error | mean over steps | at clip end | end signed mean |", "|---|---|---|---|"]
        for key, label, d in _MD_ODOM:
            e = odo["overall"]["keys"].get(key)
            if not e:
                continue
            sg = e.get("end_signed_mean", float("nan"))
            lines.append(f"| {label} | {_fmt(e['mean_abs'], d)} | {_fmt(e['end_abs'], d)} | {'-' if not np.isfinite(sg) else f'{sg:.{d}f}'} |")
        sf = odo["overall"].get("stance_fraction")
        if sf is not None and np.isfinite(sf):
            lines.append(f"\nsteps with >= 1 stance foot: {sf:.3f}")
        if len(odo.get("by_source", {})) > 1:
            lines += ["", "| source | n | xy end cm | yaw end deg | z end cm | xy mean cm | yaw mean deg | z mean cm |", "|---|---|---|---|---|---|---|---|"]
            for tag, g in odo["by_source"].items():
                kk = g["keys"]

                def ge(k, field="end_abs"):
                    v = kk.get(k, {}).get(field, {}).get("mean", float("nan"))
                    return "-" if not np.isfinite(v) else f"{v:.2f}"

                lines.append(f"| {tag} | {g['n_clips']} | {ge('odom_xy_err_cm')} | {ge('odom_yaw_err_deg')} | {ge('odom_z_err_cm')} | "
                             f"{ge('odom_xy_err_cm', 'mean_abs')} | {ge('odom_yaw_err_deg', 'mean_abs')} | {ge('odom_z_err_cm', 'mean_abs')} |")
    bys = summary.get("by_source", {})
    if bys:  # always reported per source_tag (bench: h050 / h074 / h088 table heights)
        lines += ["", "## By source tag (horizon-avg means; `alive` = pre-failure steps only)", ""]
        lines.append("| source | n | local EE cm | global EE cm | global EE cm (alive) | EE rot deg | arm joint rad | anchor xy cm | base h cm | fail-free |")
        lines.append("|---|---|---|---|---|---|---|---|---|---|")
        for tag, e in bys.items():
            m, ma = e["metrics"], e["metrics_alive"]

            def g(d, k):
                v = d.get(k, {}).get("mean", float("nan"))
                return "-" if not np.isfinite(v) else f"{v:.2f}"

            lines.append(
                f"| {tag} | {e['n_clips']} | {g(m, 'ee_local_cm')} | {g(m, 'ee_global_cm')} | {g(ma, 'ee_global_cm')} | {g(m, 'ee_rot_deg')} | "
                f"{g(m, 'joint_upper_rad')} | {g(m, 'anchor_xy_cm')} | {g(m, 'base_height_cm')} | {e['fail_free_fraction']:.2f} |"
            )
    if "kp_global_pos_cm" in ov["metrics"]:  # keypoint columns present; an older series prints nothing here
        kp = summary.get("keypoints") or {}
        lines += ["", "## Full-body keypoints (32 holosoma bodies vs the reference placed at the episode start; global = world frame, no alignment; "
                      "local = own heading-aligned pelvis frame; kp14 = the SONIC protocol's 14 tracked links, MPJPE-L = kp14 local; "
                      f"sonic_success = never |dz root| or |dz hand| > {kp.get('sonic_fail_dz_m', 0.25)} m inside the valid window; horizon-avg means over clips)", ""]
        kp_head = "| source | n | sonic_success | fail-free | " + " | ".join(label for _, label, _ in _MD_KP) + " |"
        kp_sep = "|" + "---|" * (4 + len(_MD_KP))

        def kp_row(tag: str, e: Mapping[str, Any], block: str, succ_key: str) -> str:
            m = e[block]
            sv = e.get(succ_key, float("nan"))
            cells = []
            for k, _, d in _MD_KP:
                v = m.get(k, {}).get("mean", float("nan"))
                cells.append("-" if not np.isfinite(v) else f"{v:.{d}f}")
            return f"| {tag} | {e['n_clips']} | {'-' if not np.isfinite(sv) else f'{sv:.2f}'} | {e['fail_free_fraction']:.2f} | " + " | ".join(cells) + " |"

        lines += [kp_head, kp_sep, kp_row("all", ov, "metrics", "sonic_success_fraction")]
        for tag, e in bys.items():
            lines.append(kp_row(tag, e, "metrics", "sonic_success_fraction"))
        lines += ["", "pre-failure steps only (`alive`; sonic_success over the pre-failure window):", "", kp_head, kp_sep, kp_row("all", ov, "metrics_alive", "sonic_success_alive_fraction")]
        for tag, e in bys.items():
            lines.append(kp_row(tag, e, "metrics_alive", "sonic_success_alive_fraction"))
    clips = summary.get("clips", [])
    if clips and len(clips) <= 60:
        lines += ["", "## Per clip", "", "| clip | valid | fail step | cause | local EE cm | global EE cm | rot deg | arm rad |", "|---|---|---|---|---|---|---|---|"]
        for c in clips:
            h = c["horizon_avg"]

            def f(k, d=2):
                v = h.get(k)
                return "-" if v is None else f"{v:.{d}f}"

            lines.append(f"| {c['clip_name']} | {c['valid_until']} | {c['first_fail_step']} | {c['first_fail_cause'] or '-'} | {f('ee_local_cm')} | {f('ee_global_cm')} | {f('ee_rot_deg', 1)} | {f('joint_upper_rad', 3)} |")
    return "\n".join(lines) + "\n"


__all__ = ["DEFAULT_TIMES_S", "NO_FAILURE_CAUSE", "SCHEMA", "alive_until_steps", "build_series", "end_values", "horizon_average", "odometry_summary", "read_series",
           "sonic_success_block", "sonic_success_per_clip", "stats", "stats_std", "summarize", "summary_markdown", "write_series"]
