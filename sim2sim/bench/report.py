"""hero_bench_v1 side-by-side report: merges ONE open-loop ``bench_summary.json`` (:mod:`sim2sim.bench.summary`) and ONE closed-loop
``closed_loop_summary.json`` (:mod:`sim2sim.bench.closed_loop_summary`) of the same corpus into one table per group -- ``all``, every tier
(``tier:<t>``), every layer (``stratum:<s>``) and every layer x height x hand cell (``cell:<s>/<h>/<hand>``) -- with the fixed adjacent columns

    global open-loop (hold) | global closed-loop replan+adjust (hold3) | global closed-loop final-1s | local open-loop | fail-free open / closed |
    S7.5 | S5 | C3 (Wilson 95 % CI) | stayed | IK residual

* a group whose closed-loop block is missing (a layer the closed-loop harness cannot run) prints ``n/a (harness)`` in every closed-loop
  cell -- a world-frame (global) number is NEVER printed without its replan neighbour (:func:`validate_rows` enforces it and the writer refuses
  to emit a table that violates it);
* thresholds / CDF points / stayed rule come from the summaries' ``success_protocol`` blocks (= the manifest ``protocol.success`` when the
  manifest had one, else the :mod:`sim2sim.bench.summary` defaults) -- nothing is hard-coded here;
* ``--card`` writes the results-card skeleton (``RESULTS_CARD.md`` + ``results_card.json``): protocol hash (sha256 of the manifest's
  ``protocol`` block), manifest sha256, plant / policy sha placeholders (``--plant-sha`` / ``--policy-sha`` fill them), odometry mode + seed,
  which tiers were run, the tier / layer tables -- and NO external reference rows (a label listed in ``--external-labels`` is refused in card
  mode: the card reports the evaluated policy only).

    python -m sim2sim.bench.report --open-loop <ol>/bench_summary.json --closed-loop <cl>/closed_loop_summary.json \\
        --manifest <bench>/BENCH_MANIFEST.json --out <dir> [--open-label mujoco] [--closed-label replan] [--card]

Writes ``<out>/hero_bench_report.md`` + ``hero_bench_report.json`` (and the card files with ``--card``).  Pure stdlib.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import sys
from pathlib import Path
from typing import Any, Mapping, Sequence

SCHEMA = "hero_bench_report_v1"
NA_HARNESS = "n/a (harness)"
EMPTY = "--"
HANDS: tuple[str, ...] = ("left", "right", "all")
#: report columns in order (name, kind): kind ``open`` / ``closed`` / ``both`` / ``meta`` decides what a missing block prints
COLUMNS: tuple[tuple[str, str], ...] = (
    ("global open-loop (hold)", "open"),
    ("global closed-loop replan+adjust (hold3)", "closed"),
    ("global closed-loop final-1s", "closed"),
    ("local open-loop", "open"),
    ("fail-free open / closed", "both"),
    ("S7.5", "open"),
    ("S5", "open"),
    ("C3 (CI)", "closed"),
    ("stayed", "closed"),
    ("IK residual (cm)", "open"),
)
GLOBAL_OPEN_COL, GLOBAL_CLOSED_COL = COLUMNS[0][0], COLUMNS[1][0]
#: optional trailing column, present when a closed-loop group has own-retract rows: the rest pose after the hand-back, scored apart from ``final``
REST_COL = "global closed-loop rest-1s (retract rows)"


# ================================================================================================ inputs
def _load(path: str | os.PathLike) -> dict[str, Any]:
    return json.loads(Path(path).read_text())


def sha256_file(path: str | os.PathLike | None) -> str | None:
    if not path or not Path(path).is_file():
        return None
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def protocol_hash(protocol: Mapping[str, Any] | None) -> str | None:
    """sha256 (first 16 hex) of the canonical JSON of the manifest ``protocol`` block (None without one)."""
    if not protocol:
        return None
    return hashlib.sha256(json.dumps(protocol, sort_keys=True, separators=(",", ":"), default=str).encode()).hexdigest()[:16]


def pick_label(summary: Mapping[str, Any], labels_key: str, wanted: str | None, what: str) -> str:
    labels = list(summary.get(labels_key) or [])
    if not labels:
        raise SystemExit(f"{what}: no labels in the summary")
    if wanted is None:
        return labels[0]
    if wanted not in labels:
        raise SystemExit(f"{what}: label {wanted!r} not in {labels}")
    return wanted


def open_groups(summary: Mapping[str, Any], label: str) -> dict[str, Any]:
    """``{group key: group statistics}`` of one open-loop sim: ``all`` + the ``bench_groups`` keys (``stratum:`` / ``tier:`` / ``cell:``)."""
    sim = summary["sims"][label]
    out: dict[str, Any] = {"all": sim["groups"]["all"]["all"]}
    out.update(sim.get("bench_groups") or {})
    return out


def closed_groups(summary: Mapping[str, Any], label: str) -> dict[str, Any]:
    return dict(summary["results"][label]["groups"])


def group_keys(open_g: Mapping[str, Any], closed_g: Mapping[str, Any]) -> list[str]:
    """Row order: all, tiers, strata, cells (sorted inside each kind); the union of both summaries."""
    keys = set(open_g) | set(closed_g)
    order = [k for k in ("all",) if k in keys]
    for prefix in ("tier:", "stratum:", "cell:"):
        order += sorted(k for k in keys if k.startswith(prefix))
    return order


def kind_of(key: str) -> str:
    return "all" if key == "all" else key.split(":", 1)[0]


# ================================================================================================ cells
def _pm(st: Mapping[str, Any] | None, dec: int = 1) -> str:
    if not st or not st.get("n") or not _finite(st.get("mean")):
        return EMPTY
    return f"{st['mean']:.{dec}f} ± {st['std']:.{dec}f}"


def _finite(x: Any) -> bool:
    return isinstance(x, (int, float)) and math.isfinite(float(x))


def _pct(x: Any) -> str:
    return EMPTY if not _finite(x) else f"{100.0 * float(x):.1f}%"


def _ci(level: Mapping[str, Any] | None) -> str:
    if not level or not level.get("n") or not _finite(level.get("frac")):
        return EMPTY
    lo, hi = level.get("ci95", [float("nan"), float("nan")])
    return f"{100.0 * level['frac']:.1f}% [{100.0 * lo:.0f}, {100.0 * hi:.0f}]"


def row_cells(key: str, og: Mapping[str, Any] | None, cg: Mapping[str, Any] | None, *, rest_col: bool = False) -> dict[str, str]:
    """The formatted cells of one group row.  ``cg`` None -> every closed-loop cell = ``n/a (harness)``; ``og`` None -> open cells ``--``;
    ``rest_col`` adds :data:`REST_COL` (the closed-loop group's ``rest`` block, ``--`` for groups without own-retract rows)."""
    cells: dict[str, str] = {}
    o_ok = bool(og) and og.get("n_clips", 0) > 0
    c_ok = bool(cg) and cg.get("n_clips", 0) > 0
    om = (og or {}).get("metrics", {}) if o_ok else {}
    osc = (og or {}).get("success", {}) if o_ok else {}
    cw = (cg or {}).get("windows", {}) if c_ok else {}
    csc = (cg or {}).get("success", {}) if c_ok else {}
    cells[GLOBAL_OPEN_COL] = _pm(om.get("ee_global_active_cm")) if o_ok else EMPTY
    cells[GLOBAL_CLOSED_COL] = _pm(cw.get("hold3", {}).get("ee_global_active_cm")) if c_ok else NA_HARNESS
    cells["global closed-loop final-1s"] = _pm(cw.get("final", {}).get("ee_global_active_cm")) if c_ok else NA_HARNESS
    cells["local open-loop"] = _pm(om.get("ee_local_active_cm")) if o_ok else EMPTY
    cells["fail-free open / closed"] = (_pct(og.get("fail_free_frac")) if o_ok else EMPTY) + " / " + (_pct(cg.get("fail_free_frac")) if c_ok else NA_HARNESS)
    levels = osc.get("levels", {}) if o_ok else {}
    cells["S7.5"] = _ci(levels.get("S7.5")) if o_ok else EMPTY
    cells["S5"] = _ci(levels.get("S5")) if o_ok else EMPTY
    if o_ok and (set(levels) - {"S7.5", "S5"}):   # a manifest with other level names: list them in the S7.5 column
        cells["S7.5"] = "; ".join(f"{lv} {_ci(st)}" for lv, st in levels.items())
    if c_ok:
        cl = csc.get("levels", {})
        c3 = cl.get("C3")
        cells["C3 (CI)"] = _ci(c3) if c3 is not None else ("; ".join(f"{lv} {_ci(st)}" for lv, st in cl.items()) or EMPTY)
        sf = csc.get("stayed_frac")
        if sf is None:
            rp = (cg or {}).get("replan") or {}
            sf = rp.get("stayed_frac")
        cells["stayed"] = _pct(sf)
    else:
        cells["C3 (CI)"] = NA_HARNESS
        cells["stayed"] = NA_HARNESS
    ik = osc.get("ik_residual_cm", {}) if o_ok else {}
    cells["IK residual (cm)"] = f"{ik['mean']:.2f}" if ik.get("n") else EMPTY
    if rest_col:
        rt = (cg or {}).get("rest") if c_ok else None
        cells[REST_COL] = _pm(rt.get("rest_pos_err_cm")) if rt else EMPTY
    return cells


def _is_number_cell(cell: str) -> bool:
    return cell not in (EMPTY, NA_HARNESS) and bool(cell) and cell[0].isdigit()


def validate_rows(rows: Sequence[Mapping[str, Any]]) -> None:
    """The cell rule: a row that prints a world-frame open-loop number must print its replan neighbour (a number or ``n/a (harness)``) --
    never an empty cell; raises ValueError otherwise."""
    for r in rows:
        cells = r["cells"]
        if _is_number_cell(cells.get(GLOBAL_OPEN_COL, "")):
            nb = cells.get(GLOBAL_CLOSED_COL, "")
            if not (_is_number_cell(nb) or nb == NA_HARNESS):
                raise ValueError(f"row {r['key']!r}: global open-loop number {cells[GLOBAL_OPEN_COL]!r} without its replan neighbour ({nb!r})")
        if _is_number_cell(cells.get(GLOBAL_CLOSED_COL, "")):
            nb = cells.get(GLOBAL_OPEN_COL, "")
            if not (_is_number_cell(nb) or nb == EMPTY or nb == NA_HARNESS):
                raise ValueError(f"row {r['key']!r}: closed-loop global number without an open-loop column ({nb!r})")


# ================================================================================================ report
def series_meta(open_summary: Mapping[str, Any] | None, open_label: str | None, closed_summary: Mapping[str, Any] | None, closed_label: str | None) -> dict[str, Any]:
    """Provenance of the run for the report / card: the open-loop sim's ``meta`` (the series ``meta_json``), else the closed-loop summary's
    ``series[label]["meta"]`` block (policy / odometry / termination), else empty."""
    if open_summary is not None and open_label is not None:
        m = open_summary["sims"][open_label].get("meta") or {}
        if m:
            return dict(m)
    if closed_summary is not None and closed_label is not None:
        return dict(((closed_summary.get("series") or {}).get(closed_label) or {}).get("meta") or {})
    return {}


def termination_block(meta: Mapping[str, Any]) -> dict[str, Any]:
    """The fail-free rule the run used, for the results card: ``fail_causes`` and ``fall_low_ref_margin_m`` (from ``meta["termination"]``, else the
    top-level meta keys; None when the series does not record them) plus everything else under ``meta["termination"]``."""
    term = meta.get("termination") if isinstance(meta.get("termination"), Mapping) else {}
    causes = term.get("fail_causes", meta.get("fail_causes"))
    out: dict[str, Any] = {"fail_causes": (list(causes) if isinstance(causes, (list, tuple)) else causes),
                           "fall_low_ref_margin_m": term.get("fall_low_ref_margin_m", meta.get("fall_low_ref_margin_m"))}
    out.update({k: v for k, v in term.items() if k not in out})
    return out


def build_report(open_summary: Mapping[str, Any] | None, open_label: str | None, closed_summary: Mapping[str, Any] | None, closed_label: str | None, *,
                 manifest: Mapping[str, Any] | None = None, manifest_path: str | None = None, open_path: str = "", closed_path: str = "") -> dict[str, Any]:
    """Either summary may be missing (never both): without the open loop every open cell is ``--``; without the closed loop every closed cell is
    ``n/a (harness)``.  Groups with own-retract rows add the trailing :data:`REST_COL` and their ``notes``."""
    og_all = open_groups(open_summary, open_label) if (open_summary is not None and open_label is not None) else {}
    cg_all = closed_groups(closed_summary, closed_label) if (closed_summary is not None and closed_label is not None) else {}
    if not og_all and not cg_all:
        raise ValueError("build_report needs an open-loop and / or a closed-loop summary")
    rest_col = any(bool(cg.get("rest")) for cg in cg_all.values())
    rows: list[dict[str, Any]] = []
    for key in group_keys(og_all, cg_all):
        og, cg = og_all.get(key), cg_all.get(key)
        rows.append({
            "key": key, "kind": kind_of(key),
            "n_open": int(og.get("n_clips", 0)) if og else 0, "n_closed": int(cg.get("n_clips", 0)) if cg else None,
            "open": og, "closed": cg, "cells": row_cells(key, og, cg, rest_col=rest_col),
            "notes": list(cg.get("notes") or []) if cg else [],
        })
    validate_rows(rows)
    protocol = dict(manifest.get("protocol") or {}) if isinstance(manifest, Mapping) else None
    sp_open = (open_summary or {}).get("success_protocol") or {}
    sp_closed = (closed_summary or {}).get("success_protocol") or {}
    tiers_open = sorted({k[len("tier:"):] for k in og_all if k.startswith("tier:")})
    tiers_closed = sorted({k[len("tier:"):] for k in cg_all if k.startswith("tier:")})
    meta = series_meta(open_summary, open_label, closed_summary, closed_label)
    pol = meta.get("policy") or {}
    odom = meta.get("odometry") or {}
    return {
        "schema": SCHEMA,
        "open_loop": ({"path": open_path, "label": open_label, "n_clips": og_all.get("all", {}).get("n_clips")} if og_all else None),
        "closed_loop": ({"path": closed_path, "label": closed_label, "n_clips": cg_all.get("all", {}).get("n_clips")} if cg_all else None),
        "manifest": {"path": manifest_path, "sha256": sha256_file(manifest_path), "protocol_hash": protocol_hash(protocol), "n_clips": (manifest or {}).get("n_clips") if isinstance(manifest, Mapping) else None},
        "success_protocol": {"open_loop": sp_open.get("open_loop") or sp_closed.get("open_loop"), "closed_loop": sp_closed.get("closed_loop") or sp_open.get("closed_loop"),
                             "cdf_points_cm": sp_open.get("cdf_points_cm") or sp_closed.get("cdf_points_cm"), "stayed_threshold_cm": sp_closed.get("stayed_threshold_cm") or sp_open.get("stayed_threshold_cm"),
                             "source": {"open_loop": sp_open.get("source"), "closed_loop": sp_closed.get("source")}},
        "tiers_run": {"open_loop": tiers_open, "closed_loop": tiers_closed},
        "strata_run": {"open_loop": sorted({k[len("stratum:"):] for k in og_all if k.startswith("stratum:")}), "closed_loop": sorted({k[len("stratum:"):] for k in cg_all if k.startswith("stratum:")})},
        "policy": {"tag": pol.get("tag") or meta.get("tag"), "onnx": pol.get("onnx"), "onnx_bytes": pol.get("onnx_bytes"), "iteration": pol.get("iteration"), "preset": pol.get("preset"),
                   "odom_source": pol.get("odom_source") or odom.get("source"), "odom_seed": (odom.get("config") or {}).get("seed", (pol.get("odom_noise") or {}).get("seed"))},
        "termination": termination_block(meta),
        "notes": sorted({n for r in rows for n in r["notes"]}),
        "columns": [c for c, _ in COLUMNS] + ([REST_COL] if rest_col else []),
        "rows": rows,
    }


def _term_text(report: Mapping[str, Any]) -> str:
    t = report.get("termination") or {}
    causes = t.get("fail_causes")
    c = ", ".join(str(x) for x in causes) if isinstance(causes, (list, tuple)) else (str(causes) if causes else "n/a (not recorded in the series)")
    m = t.get("fall_low_ref_margin_m")
    return f"causes `{c}`; fall_low margin " + (f"{float(m):g} m" if isinstance(m, (int, float)) and not isinstance(m, bool) else "off")


def _thr_text(report: Mapping[str, Any]) -> str:
    sp = report.get("success_protocol") or {}
    parts = []
    for lv, thr in (sp.get("open_loop") or {}).items():
        parts.append(f"{lv} = open-loop hold mean <= {thr['pos_cm']:g} cm & <= {thr['rot_deg']:g} deg")
    for lv, thr in (sp.get("closed_loop") or {}).items():
        parts.append(f"{lv} = closed-loop {thr.get('window', 'final')}-1 s <= {thr['pos_cm']:g} cm & <= {thr['rot_deg']:g} deg")
    st = sp.get("stayed_threshold_cm")
    if st is not None:
        parts.append(f"stayed = HERO stop rule <= {st:g} cm")
    return "; ".join(parts) + f" (thresholds from the {(sp.get('source') or {}).get('open_loop') or 'default'})"


def table_markdown(report: Mapping[str, Any], kinds: Sequence[str], title: str) -> list[str]:
    cols = list(report["columns"])
    L = [f"### {title}", "", "| group | n open / closed | " + " | ".join(cols) + " |", "|---|---|" + "---|" * len(cols)]
    for r in report["rows"]:
        if r["kind"] not in kinds:
            continue
        n = f"{r['n_open'] if report.get('open_loop') else EMPTY} / {r['n_closed'] if r['n_closed'] is not None else NA_HARNESS}"
        L.append(f"| {r['key']} | {n} | " + " | ".join(r["cells"][c] for c in cols) + " |")
    L.append("")
    return L


def render_markdown(report: Mapping[str, Any], title: str = "hero_bench_v1 side-by-side report") -> str:
    ol, cl, mf = report.get("open_loop"), report.get("closed_loop"), report["manifest"]
    L = [f"# {title}", "",
         "Open loop " + (f"`{ol['label']}` (`{ol['path']}`, {ol.get('n_clips')} clips)" if ol else f"**none** (every open-loop cell = `{EMPTY}`)") + "; closed loop "
         + (f"`{cl['label']}` (`{cl['path']}`, {cl.get('n_clips')} clips)" if cl else f"**none** (every closed-loop cell = `{NA_HARNESS}`)")
         + f"; manifest `{mf.get('path')}` protocol hash `{mf.get('protocol_hash') or 'n/a'}`; fail-free rule: {_term_text(report)}.", "",
         "Cell rule: the world-frame columns always appear as the pair `open-loop (hold)` | `closed-loop replan+adjust (hold3)`; a group the "
         f"closed-loop harness cannot run prints `{NA_HARNESS}`.  Headline = the ACTIVE (reaching) hand's world-frame palm error; `local` = own-pelvis error.  "
         + _thr_text(report) + ".  Success denominators = every clip of the group (Wilson 95 % CI in brackets).", ""]
    L += table_markdown(report, ("all", "tier"), "All clips and per tier")
    L += table_markdown(report, ("stratum",), "Per layer (stratum)")
    L += table_markdown(report, ("cell",), "Per layer x height x hand")
    L += _notes_lines(report)
    tr = report["tiers_run"]
    L.append(f"Tiers run: open loop {tr['open_loop'] or '-'}; closed loop {tr['closed_loop'] or '-'}.")
    return "\n".join(L) + "\n"


def _notes_lines(report: Mapping[str, Any]) -> list[str]:
    """The optional rest column's legend + the closed-loop groups' notes (own-retract rows), as markdown lines (empty when neither applies)."""
    L: list[str] = []
    if REST_COL in report.get("columns", []):
        L.append(f"`{REST_COL}` = groups with own-retract rows: the active hand's world-frame error over the last second of the padded clip (the rest pose after "
                 f"the hand-back), scored apart from `final` / C3; `{EMPTY}` for groups without such rows.")
    for n in report.get("notes") or []:
        L.append(f"Note: {n}.")
    if L:
        L.append("")
    return L


# ================================================================================================ results card
def render_card(report: Mapping[str, Any], *, plant_sha: str | None, policy_sha: str | None, title: str) -> tuple[str, dict[str, Any]]:
    """The results-card skeleton: provenance header + tier / layer tables; NO external reference rows."""
    mf, pol = report["manifest"], report["policy"]
    card = {
        "schema": "hero_bench_results_card_v1",
        "title": title,
        "protocol_hash": mf.get("protocol_hash"),
        "manifest_sha256": mf.get("sha256"),
        "plant_sha256": plant_sha or "<plant URDF sha256: TODO>",
        "policy_sha256": policy_sha or "<policy ONNX sha256: TODO>",
        "policy": {k: pol.get(k) for k in ("tag", "iteration", "preset", "onnx_bytes")},
        "odometry": {"source": pol.get("odom_source"), "seed": pol.get("odom_seed")},
        "termination": report.get("termination"),
        "tiers_run": report["tiers_run"],
        "success_protocol": report["success_protocol"],
        "notes": list(report.get("notes") or []),
        "external_reference_rows": "none (the card reports the evaluated policy only)",
        "rows": [{"key": r["key"], "kind": r["kind"], "n_open": r["n_open"], "n_closed": r["n_closed"], "cells": r["cells"]} for r in report["rows"] if r["kind"] in ("all", "tier", "stratum")],
    }
    L = [f"# {title}", "",
         "| field | value |", "|---|---|",
         f"| protocol hash | `{card['protocol_hash'] or 'n/a (no manifest given)'}` |",
         f"| manifest sha256 | `{card['manifest_sha256'] or 'n/a'}` |",
         f"| plant (URDF) sha256 | `{card['plant_sha256']}` |",
         f"| policy (ONNX) sha256 | `{card['policy_sha256']}` |",
         f"| policy | `{pol.get('tag')}` iter {pol.get('iteration')} preset `{pol.get('preset')}` |",
         f"| odometry | `{pol.get('odom_source')}` seed {pol.get('odom_seed')} |",
         f"| fail-free rule | {_term_text(report)} |",
         f"| tiers run | open loop {report['tiers_run']['open_loop'] or '-'}; closed loop {report['tiers_run']['closed_loop'] or '-'} |",
         "", _thr_text(report) + ".", ""]
    L += table_markdown(report, ("all", "tier"), "All clips and per tier")
    L += table_markdown(report, ("stratum",), "Per layer (stratum)")
    L += _notes_lines(report)
    L.append("External reference rows (other policies / baselines): not part of this card -- it reports the evaluated policy only.")
    return "\n".join(L) + "\n", card


# ================================================================================================ CLI
def build_parser() -> argparse.ArgumentParser:
    ap = argparse.ArgumentParser(description=__doc__.split("\n\n")[0], formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--open-loop", default=None, help="bench_summary.json of the open-loop run (omit -> every open-loop cell is --; at least one of --open-loop / --closed-loop)")
    ap.add_argument("--open-label", default=None, help="sim label inside the open-loop summary (default: the first)")
    ap.add_argument("--closed-loop", default=None, help="closed_loop_summary.json of the replan + goal-adjust run (omit -> every closed-loop cell is n/a (harness))")
    ap.add_argument("--closed-label", default=None, help="series label inside the closed-loop summary (default: the first)")
    ap.add_argument("--manifest", default=None, help="BENCH_MANIFEST.json (protocol hash / sha256 for the card)")
    ap.add_argument("--out", required=True)
    ap.add_argument("--title", default="hero_bench_v1 side-by-side report")
    ap.add_argument("--card", action="store_true", help="also write the results-card skeleton (RESULTS_CARD.md + results_card.json)")
    ap.add_argument("--plant-sha", default=None, help="[--card] plant URDF sha256 (placeholder when omitted)")
    ap.add_argument("--policy-sha", default=None, help="[--card] policy ONNX sha256 (placeholder when omitted; computed from the open-loop meta onnx path when it exists)")
    ap.add_argument("--external-labels", default="", help="comma list of labels that are EXTERNAL reference rows (other policies); refused in --card mode")
    return ap


def main(argv: Sequence[str] | None = None) -> int:
    ap = build_parser()
    a = ap.parse_args(argv)
    if not a.open_loop and not a.closed_loop:
        ap.error("give --open-loop and / or --closed-loop")
    open_summary = _load(a.open_loop) if a.open_loop else None
    open_label = pick_label(open_summary, "sims", a.open_label, "--open-loop") if open_summary is not None else None
    closed_summary = _load(a.closed_loop) if a.closed_loop else None
    closed_label = pick_label(closed_summary, "labels", a.closed_label, "--closed-loop") if closed_summary is not None else None
    manifest = _load(a.manifest) if a.manifest else None
    report = build_report(open_summary, open_label, closed_summary, closed_label, manifest=manifest, manifest_path=a.manifest, open_path=a.open_loop or "", closed_path=a.closed_loop or "")
    out = Path(a.out)
    out.mkdir(parents=True, exist_ok=True)
    (out / "hero_bench_report.json").write_text(json.dumps(report, indent=1, default=str))
    md = render_markdown(report, a.title)
    (out / "hero_bench_report.md").write_text(md)
    print(md)
    if a.card:
        external = {x.strip() for x in a.external_labels.split(",") if x.strip()}
        if (open_label in external if open_label else False) or (closed_label in external if closed_label else False):
            raise SystemExit(f"--card: label {open_label!r} / {closed_label!r} is an external reference row -- not part of the results card")
        policy_sha = a.policy_sha or sha256_file(report["policy"].get("onnx"))
        card_md, card = render_card(report, plant_sha=a.plant_sha, policy_sha=policy_sha, title=a.title.replace("side-by-side report", "results card"))
        (out / "RESULTS_CARD.md").write_text(card_md)
        (out / "results_card.json").write_text(json.dumps(card, indent=1, default=str))
        print(f"[hero_bench_report] card -> {out / 'RESULTS_CARD.md'}")
    print(f"[hero_bench_report] wrote {out / 'hero_bench_report.md'} {out / 'hero_bench_report.json'}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
