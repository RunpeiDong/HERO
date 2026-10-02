"""Tests for data_tools.build_hero_bench (the plan-driven hero_bench_v1 corpus builder) and scripts/hero_bench.py.

Run: ``cd hero_release && python -m pytest tests/test_hero_bench_build.py -q``

Synthetic 32-body clips, a synthetic paper-protocol dir with a manifest, synthetic generator banks per acceptance level; the generator
call and the IK conversion are mocked.  Covers: per-cell selection (quota / admitted levels / no borrowing), fill_until_quota with batch
seeds and the exit-3 core gap, verbatim import (sha verified, mismatch fails, names unchanged), re-timed strata (naming, pair_of, frame
fields, velocity gate, shortfall_ok), the recoverable pool, the manifest protocol block + deterministic sha, per-tier manifests / tiers
lists / SHA256SUMS, --public scrub, verify_shipped, health rules, the hero_bench.py CLI (plan / build / verify) and, when
data/hero_bench_paper (or HERO_BENCH_PAPER_DIR) holds the paper-protocol set, the real 180-file verbatim import.
"""

from __future__ import annotations

import copy
import hashlib
import json
import shutil
import sys
from pathlib import Path

import numpy as np
import pytest

from data_tools import build_reach_bench as bb
from data_tools import build_hero_bench as B
from data_tools import reach_specs as R
from data_tools import retime_bench as RB
from hero_isaacsim import constants as HC
from _bench_fixtures import PAPER_DIR

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
import hero_bench  # noqa: E402  (scripts/hero_bench.py)

FPS = 50
REAL_PAPER = PAPER_DIR
HEIGHTS = {"h025": 0.25, "h030": 0.30, "h040": 0.40, "h050": 0.5, "h062": 0.62, "h074": 0.74, "h080": 0.80, "h088": 0.88, "h100": 1.0, "h110": 1.1, "h115": 1.15, "h120": 1.2, "floor": 0.0}


# ------------------------------------------------------------------------------------------------ synthetic clips
def _yaw_quat(yaw: np.ndarray) -> np.ndarray:
    yaw = np.asarray(yaw, dtype=np.float64)
    return np.stack([np.cos(yaw / 2.0), np.zeros_like(yaw), np.zeros_like(yaw), np.sin(yaw / 2.0)], axis=-1)


def synthetic_clip(T: int = 101, *, reach_end: int = 60, settle: int = 15, dof_rate: float = 0.3, source_tag: str = "h050", seed: int = 0) -> dict[str, np.ndarray]:
    """32-body bench-like clip: settle [0, settle], linear reach, static hold from ``reach_end``; velocities from the retime tool's estimators."""
    t = np.arange(T, dtype=np.float64) / FPS
    t = np.clip(t, settle / FPS, reach_end / FPS)
    v = np.array([0.02, -0.01, 0.0])
    root_pos = np.array([0.0, 0.0, 0.75]) + v[None, :] * t[:, None]
    root_quat = _yaw_quat(0.1 * t)
    pattern = np.linspace(-1.0, 1.0, HC.NUM_DOF)
    dofs = 0.1 * pattern[None, :] + dof_rate * t[:, None] * pattern[None, :]
    joint_pos = np.concatenate([root_pos, root_quat, dofs], axis=1)
    rng = np.random.default_rng(seed)
    offsets = rng.uniform(-0.4, 0.4, size=(len(HC.HOLOSOMA_BODY_NAMES_32), 3))
    offsets[0] = 0.0
    body_pos_w = root_pos[:, None, :] + offsets[None, :, :]
    body_quat_w = np.repeat(root_quat[:, None, :], len(HC.HOLOSOMA_BODY_NAMES_32), axis=1)
    ee_pos = np.stack([np.array([0.3, 0.2, 0.1]) + 0.05 * t[:, None], np.array([0.3, -0.2, 0.1]) - 0.05 * t[:, None]], axis=1)
    ee_quat = np.repeat(_yaw_quat(0.05 * t)[:, None, :], 2, axis=1)
    arrays: dict[str, np.ndarray] = {
        "fps": np.asarray(FPS, dtype=np.int64), "joint_pos": joint_pos.astype(np.float32), "body_pos_w": body_pos_w.astype(np.float32),
        "body_quat_w": body_quat_w.astype(np.float32), "body_names": np.asarray(list(HC.HOLOSOMA_BODY_NAMES_32)), "joint_names": np.asarray(list(HC.DOF_NAMES)),
        "ee_pos_pelvis": ee_pos.astype(np.float32), "ee_quat_pelvis": ee_quat.astype(np.float32),
        "ee_pos_pelvis_zero_waist": (ee_pos + 0.01).astype(np.float32), "ee_quat_pelvis_zero_waist": ee_quat.astype(np.float32),
        "h_ref": root_pos[:, 2].astype(np.float32), "source_tag": np.asarray(source_tag), "parent_id": np.asarray("synthetic"), "license_class": np.asarray("apache"),
        "has_object": np.asarray(False, dtype=np.bool_),
    }
    for k, a in RB.recompute_velocities(arrays, FPS, "world").items():
        arrays[k] = a.astype(np.float32)
    return arrays


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


# ------------------------------------------------------------------------------------------------ synthetic hero_bench_v1 dir
def make_paper(root: Path, per_cell: int = 1, *, T: int = 101, reach_end: int = 60, hot: set[tuple[str, str, int]] = frozenset()) -> Path:
    """``per_cell`` right + ``per_cell`` left clips per paper-protocol table height (goal_index even = right, odd = left), a paper-protocol-style manifest.
    ``hot`` = {(label, hand, goal_index)} clips with a 7 rad/s dof ramp (over the fast_x0p75 gate at x0.75, under it at x1 / x2)."""
    root.mkdir(parents=True, exist_ok=True)
    rows = []
    for hi, lab in enumerate(("h050", "h074", "h088")):
        hm = HEIGHTS[lab]
        for gi in range(2 * per_cell):
            hand = "right" if gi % 2 == 0 else "left"
            cid = f"hero_bench_v1_{hi * 100 + gi:06d}"
            name = f"{lab}__{cid}.npz"
            arrays = synthetic_clip(T, reach_end=reach_end, dof_rate=7.0 if (lab, hand, gi) in hot else 0.3, source_tag=lab, seed=hi * 100 + gi)
            np.savez_compressed(root / name, **arrays)
            rows.append({
                "file": name, "clip_id": cid, "source_tag": lab, "height_label": lab, "height_m": hm, "hand": hand, "goal_index": gi, "candidate_index": gi,
                "target_pos_w": [0.35, -0.2 if hand == "right" else 0.2, hm + 0.1], "target_yaw_deg": 10.0, "target_pitch_deg": -5.0, "target_roll_deg": 0.0,
                "table": {"center": [0.62, 0.14, hm - 0.02], "half_size": [0.3, 0.7, 0.02], "surface_z": hm}, "table_top_z": hm, "table_edge_x": 0.3, "table_edge_gap_m": 0.18,
                "table_edge_ref": "toe", "base_family": "stand", "pelvis_drop": 0.02, "pelvis_pitch_deg": 4.0, "waist_pitch_deg": 6.0, "reach_prefilter_ok": True,
                "settle_frames": 15, "reach_start_frame": 15, "reach_end_frame": reach_end, "reach_frames": reach_end - 15, "hold_frames": T - reach_end, "n_frames": T, "fps": FPS,
                "duration_s": T / FPS, "terminal_ee_pos_err_m": 0.003, "terminal_ee_ori_err_rad": 0.01, "ee_pos_err_at_reach_end_m": 0.003, "ee_pos_err_hold_max_m": 0.003,
                "pelvis_height_min_m": 0.72, "h_ref_min_m": float(arrays["h_ref"].min()), "h_ref_max_m": float(arrays["h_ref"].max()), "pelvis_xy_frame0": [0.0, 0.0],
                "tier": "core", "flags": [], "accept_level": 0, "accept_level_name": "strict", "max_qdot_rad_s": float(np.abs(arrays["joint_vel"][:, 6:]).max()),
                "min_com_margin_m": 0.05, "sha256": _sha(root / name), "bytes": (root / name).stat().st_size, "license_class": "apache", "parent_id": cid, "has_object": False,
                "bank_file": f"clips/{cid}.npz", "bank_sha256": "0" * 64,
            })
    manifest = {"schema": "hero_bench_v1", "heights_m": [0.5, 0.74, 0.88], "height_labels": ["h050", "h074", "h088"], "n_clips": len(rows), "per_hand_target": per_cell,
                "git_sha": "123abcd+bench-wip", "host": "ws-example-node", "bank_dir": "/root/example/project/data/hero_bench_v1",
                "bank": {"generator_version": "2.0.0", "seed": 20260906, "mjcf": "/root/example/project/scene.xml", "git_sha": "123abcd+bench-wip", "profile": "hero_bench_v1"},
                "protocol": {"timing": {"fps": FPS, "settle_frames": 15, "hold_s": 3.0, "time_scale": 1.0}, "paper": "arXiv 2602.16705"},
                "verification": {"clip_lengths": {r["file"]: r["n_frames"] for r in rows}, "n_files": len(rows), "n_loaded": len(rows), "n_skipped": 0}, "clips": rows}
    (root / "BENCH_MANIFEST.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return root


# ------------------------------------------------------------------------------------------------ synthetic generator banks
LEVEL_ROWS = {"strict": ("core", [], 0.004), "core": ("core", [], 0.012), "recoverable": ("recoverable", ["high_acceleration"], 0.025), "reject": ("boundary", ["self_collision"], 0.002)}


def bank_row_specs(entry: dict, levels_per_cell: dict[str, int] | list[str]) -> list[dict]:
    """Candidate specs for every quota cell of ``entry``: ``levels_per_cell`` = level -> count (or an ordered list of levels)."""
    seq = levels_per_cell if isinstance(levels_per_cell, list) else [lv for lv, n in levels_per_cell.items() for _ in range(n)]
    specs = []
    for cell in entry["quota"]:
        for level in seq:
            specs.append({"cell": tuple(cell), "level": level})
    return specs


def make_bank(root: Path, entry: dict, batch: int, specs: list[dict], *, T: int = 101, reach_end: int = 60, retract: bool = False) -> Path:
    """Generator-bank layout (clips/, labels/, manifest.json) for one stratum batch; goal_index counts up per height like the generator."""
    root.mkdir(parents=True, exist_ok=True)
    (root / "clips").mkdir(exist_ok=True)
    (root / "labels").mkdir(exist_ok=True)
    prefix = B.batch_clip_prefix(entry, batch)
    with_family = B.quota_has_family(entry["quota"])
    per_height: dict[str, int] = {}
    rows = []
    for i, sp in enumerate(specs):
        cell = sp["cell"]
        lab, hand = cell[0], cell[1]
        fam = cell[2] if with_family else None
        gi = per_height.get(lab, 0)
        per_height[lab] = gi + 1
        cid = f"{prefix}_{i:06d}"
        tier, flags, e_pos = LEVEL_ROWS[sp["level"]]
        hold = T - reach_end
        n_frames = T + (40 if retract else 0)
        arrays = synthetic_clip(n_frames, reach_end=reach_end, source_tag=lab if lab != "floor" else "floor", seed=1000 + i)
        if retract:   # the synthetic clip just holds; the frame bookkeeping is what the builder copies
            pass
        np.savez_compressed(root / "clips" / f"{cid}.npz", **arrays)
        hm = HEIGHTS[lab]
        bench = {"schema": "hero_bench_v1", "height_m": hm, "height_label": lab, "goal_index": gi, "candidate_index": i, "hand": hand,
                 "target_pos_w": [0.4, -0.2 if hand == "right" else 0.2, hm + 0.1], "target_x_raw": 0.4, "target_x_shifted": False, "target_z_raw": hm + 0.08, "clearance_raised": True,
                 "target_yaw_deg": 10.0, "target_pitch_deg": -5.0, "target_roll_deg": 0.0, "table_top_z": hm, "table_edge_x": 0.3, "table_edge_gap_m": 0.18, "table_edge_ref": "toe",
                 "table": {"center": [0.62, 0.14, hm - 0.02], "half_size": [0.3, 0.7, 0.02], "surface_z": hm}, "base_family": "stand", "pelvis_drop": 0.02, "pelvis_pitch_deg": 4.0,
                 "waist_pitch_deg": 6.0, "reach_prefilter_ok": True, "settle_frames": 15, "reach_start_frame": 15, "reach_end_frame": reach_end, "reach_frames": reach_end - 15,
                 "hold_frames": hold, "hold_end_frame": reach_end + hold, "retract_start_frame": reach_end + hold if retract else None, "n_frames": n_frames, "fps": FPS,
                 "ee_pos_error_at_reach_end_m": e_pos, "ee_pos_error_hold_max_m": e_pos, "stratum": entry["stratum"], "tier": entry["tier"], "plan_version": "hero_bench_v1",
                 "has_table": lab != "floor", "y_side": "same", "cross_side": False, "time_scale": 1.3 if "high" in entry["stratum"] else 1.0, "hold_s": 3.0, "retract": retract,
                 "pelvis_xy_frame0": [0.0, 0.0], "min_pelvis_height_m": 0.72}
        if lab == "floor":
            bench.update({"table": None, "table_top_z": None, "table_edge_x": None, "table_edge_gap_m": None, "table_edge_ref": None})
        if fam:
            bench.update({"orient_family": fam, "family_index": gi // 2, "yaw_deg": 10.0, "pitch_deg": -5.0, "roll_deg": 0.0, "roll_sign": 1.0, "rot_from_canonical_deg": 12.0,
                          "rot_from_side_grasp_deg": 12.0, "canonical_ypr_deg": [0.0, 0.0, 0.0], "orient_resamples": 0, "clearance_margin_m": 0.02})
        labels = {"schema": "hero_reach_bank_v2", "clip_id": cid, "tier": tier, "flags": flags, "bench": bench, "has_retract": retract}
        if retract:
            labels["retract"] = {"start_frame": reach_end + hold, "frames": 40, "reach_frames": 15, "hold_frames": 25, "after_replan": False,
                                 "final_palm_pos_w": {"left": [0.2, 0.2, 0.9], "right": [0.2, -0.2, 0.9]}, "rest_palm_pos_w": {"left": [0.2, 0.2, 0.9], "right": [0.2, -0.2, 0.9]}}
        (root / "labels" / f"{cid}.json").write_text(json.dumps(labels, sort_keys=True))
        rows.append({"clip_id": cid, "file": f"clips/{cid}.npz", "labels": f"labels/{cid}.json", "ok": True, "tier": tier, "flags": flags, "hands_mode": hand,
                     "terminal_ee_pos_error_m": e_pos, "terminal_ee_ori_error_rad": 0.01, "min_pelvis_height_m": 0.72, "sha256": _sha(root / "clips" / f"{cid}.npz"),
                     "bytes": 1000, "frames": n_frames, "max_qdot_rad_s": 1.2, "max_qddot_rad_s2": 20.0, "min_com_margin_m": 0.05, "has_retract": retract, "bench": bench})
    manifest = {"schema": "hero_reach_bank_v2", "generator_version": "2.1.0", "profile": entry["profile"], "seed": B.batch_seed(entry, batch), "n_requested": len(rows),
                "n_written": len(rows), "n_failed": 0, "tier_counts": dict(), "flag_counts": {}, "overrides": {}, "host": "ws-fixture-node", "git_sha": "abc1234+wip",
                "mjcf": "/root/example/project/scene.xml", "mjcf_sha256": "a18d3588" + "0" * 56, "solver_config": {"max_iters": 12}, "gate_thresholds": {"strict_qdot": 6.0}, "clips": rows}
    (root / "manifest.json").write_text(json.dumps(manifest, indent=1, sort_keys=True))
    return root


def fake_convert_rows(selected, out_dir, stratum, jobs=1):
    """Stand-in for the IK conversion: the synthetic bank clip already has the corpus layout; copy it with source_tag = stratum."""
    out = []
    for r in selected:
        src = Path(r["_bank_dir"]) / r["file"]
        dst = Path(out_dir) / B.bench_output_name(stratum, r)
        with np.load(src, allow_pickle=False) as z:
            arrays = {k: z[k] for k in z.files}
        arrays["source_tag"] = np.asarray(stratum)
        np.savez_compressed(dst, **arrays)
        out.append({"src": str(src), "dst": str(dst), "ok": True, "receipt": {"status": "ok", "family": "S4"}})
    return out


def tiny_plan(*strata: str, per_cell: int = 1) -> dict:
    plan = B.reduced_plan(R.BENCH_V1_PLAN, per_cell)
    return {k: plan[k] for k in strata}


@pytest.fixture(autouse=True)
def _mock_conversion(monkeypatch):
    monkeypatch.setattr(B, "convert_rows", fake_convert_rows)


def _quiet(_msg: str) -> None:
    pass


# ================================================================================================ 1. selection
def test_select_stratum_quota_levels_and_no_borrow(tmp_path):
    entry = tiny_plan("far060", per_cell=2)["far060"]
    bank = make_bank(tmp_path / "b0", entry, 0, bank_row_specs(entry, ["strict", "core", "reject", "strict", "recoverable"]))
    b = B.load_batch(bank, 0, tmp_path)
    assert len(b["rows"]) == 30 and all("_uid" in r and r["_batch"] == 0 for r in b["rows"])
    sel, stats, unsel = B.select_stratum(b["rows"], entry["quota"], ["strict"])
    assert len(sel) == 12 and all(c["accept_level"] == 0 for c in sel) and len(unsel) == 18
    for key, st in stats.items():
        assert st["quota"] == 2 and st["n_selected"] == 2 and st["shortfall"] == 0 and st["n_by_level"] == {"strict": 2, "core": 1, "recoverable": 1}
        assert st["n_never_accepted"] == 1 and len(st["selected"]) == 2 and st["selected"][0][1] < st["selected"][1][1] and not st["fallback_used"]
        assert st["strict_reject_reasons"] == {"ee_pos>10mm": 2, "tier:recoverable": 1, "flags:high_acceleration": 1, "tier:boundary": 1, "flags:self_collision": 1}
    assert stats["h050/right"]["selected"] == [[0, 0], [0, 3]] and stats["h050/left"]["selected"] == [[0, 5], [0, 8]]     # goal_index counts per height over both hands
    # strict-only with a short cell: no fallback, no borrowing from the neighbouring cell
    rows = [r for r in b["rows"] if not (r["bench"]["height_label"] == "h050" and r["bench"]["hand"] == "left" and bb.level_of(r) == 0)]
    sel2, stats2, _ = B.select_stratum(rows, entry["quota"], ["strict"])
    assert stats2["h050/left"]["n_selected"] == 0 and stats2["h050/left"]["shortfall"] == 2 and stats2["h050/right"]["n_selected"] == 2 and len(sel2) == 10
    # strict -> core fallback fills the cell with the core-level row only (recoverable not admitted)
    sel3, stats3, _ = B.select_stratum(rows, entry["quota"], ["strict", "core"])
    assert stats3["h050/left"]["n_selected"] == 1 and stats3["h050/left"]["shortfall"] == 1 and stats3["h050/left"]["max_level_used"] == "core" and stats3["h050/left"]["fallback_used"]
    assert [c["accept_level_name"] for c in sel3 if c["bench"]["height_label"] == "h050" and c["bench"]["hand"] == "left"] == ["core"]
    # families are their own cells: a short family never takes another family's strict rows
    oc = tiny_plan("orient_core", per_cell=1)["orient_core"]
    specs = [s for s in bank_row_specs(oc, ["strict", "strict"]) if not (s["cell"] == ("h050", "right", "tilted"))]
    bank2 = make_bank(tmp_path / "oc", oc, 0, specs)
    sel4, stats4, _ = B.select_stratum(B.load_batch(bank2, 0)["rows"], oc["quota"], ["strict"])
    assert stats4["h050/right/tilted"]["n_selected"] == 0 and stats4["h050/right/tilted"]["shortfall"] == 1 and stats4["h050/right/top_down"]["n_selected"] == 1 and len(sel4) == 11
    # candidates outside the quota cells are reported, never selected
    extra = B.load_batch(bank, 0)["rows"]
    for r in extra[:3]:
        r["bench"] = {**r["bench"], "height_label": "h062"}
    _, stats5, _ = B.select_stratum(extra, entry["quota"], ["strict"])
    assert stats5["unplanned_cells"]["cells"] == {"h062/right": 3}
    y = B.stratum_yield(b["rows"])
    assert y["n_candidates"] == 30 and y["admitted"] == {"strict": 12, "core": 6, "recoverable": 6, "never": 6} and abs(y["yield"] - 18 / 30) < 1e-12


# ================================================================================================ 2. fill_until_quota
def test_fill_until_quota_with_mocked_generator(tmp_path, monkeypatch):
    plan = tiny_plan("far060", per_cell=2)
    entry = plan["far060"]
    calls: list[tuple[int, int]] = []

    def fake_generator(e, batch, out_dir, *, jobs=1, nice=0, log=print):
        calls.append((batch, B.batch_seed(e, batch)))
        if (Path(out_dir) / "manifest.json").is_file():
            return Path(out_dir)
        levels = ["strict", "core", "reject"] if batch == 0 else ["strict"]          # batch 0: one strict per cell (quota 2) -> top-up batch 1
        return make_bank(Path(out_dir), e, batch, bank_row_specs(e, levels))

    monkeypatch.setattr(B, "run_generator", fake_generator)
    out = tmp_path / "out"
    man = B.build(plan, bank_root=tmp_path / "banks", out_dir=out, paper_dir=None, generate=True, jobs=1, verifier="schema", log=_quiet)
    st = man["strata"]["far060"]
    assert calls == [(0, R.bench_v1_batch_seed("far060", 0)), (1, R.bench_v1_batch_seed("far060", 1))] and calls[1][1] == 20261002 + 1000
    assert st["fill_batches_used"] == 2 and st["shipped"] == 12 and st["shortfall_total"] == 0 and st["health"] == "ok" and not st["provisional"]
    assert [b["seed"] for b in st["batches"]] == [20261002, 20262002] and st["batches"][1]["dir"] == "far060/batch1" and all(b["generated_now"] for b in st["batches"])
    assert st["n_candidates"] == 24 and st["admitted"] == {"strict": 12, "core": 6, "recoverable": 0, "never": 6}
    assert not (tmp_path / "banks" / "far060" / "batch2").exists()
    rows = man["clips"]
    assert len(rows) == 12 and all(r["tier"] == "core" and r["stratum"] == "far060" and r["accept_level"] == 0 and r["generator_tier"] == "core" for r in rows)
    names = sorted(r["file"] for r in rows)
    assert all(n.startswith("far060__hero_bench_v1_far060") and n.endswith(".npz") for n in names)
    assert sum(1 for r in rows if r["bank_batch"] == 1) == 6 and all(r["bank_batch_dir"] == f"far060/batch{r['bank_batch']}" for r in rows)
    assert all((out / "core" / r["file"]).is_file() for r in rows) and len(set(names)) == 12
    # resumable: a second build without --generate consumes the two batches on disk and does not call the generator
    calls.clear()
    man2 = B.build(plan, bank_root=tmp_path / "banks", out_dir=tmp_path / "out2", paper_dir=None, generate=False, verifier="schema", log=_quiet)
    assert calls == [] and man2["strata"]["far060"]["fill_batches_used"] == 2 and man2["n_clips"] == 12 and not any(b["generated_now"] for b in man2["strata"]["far060"]["batches"])
    # a stratum whose cell never fills: every top-up batch is generated, then exit 3 with the cell named
    calls.clear()

    def hopeless(e, batch, out_dir, *, jobs=1, nice=0, log=print):
        calls.append(batch)
        if (Path(out_dir) / "manifest.json").is_file():
            return Path(out_dir)
        specs = [s for s in bank_row_specs(e, ["strict", "strict"]) if s["cell"] != ("h050", "left")] + [{"cell": ("h050", "left"), "level": "core"}]
        return make_bank(Path(out_dir), e, batch, specs)

    monkeypatch.setattr(B, "run_generator", hopeless)
    with pytest.raises(B.CoreGapError, match="h050/left short 2"):
        B.build(plan, bank_root=tmp_path / "banks_gap", out_dir=tmp_path / "out3", paper_dir=None, generate=True, verifier="schema", log=_quiet)
    assert calls == list(range(entry["fill_max_batches"] + 1))
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(B.plan_json(plan)))
    assert B.main(["--plan", str(plan_file), "--bank-root", str(tmp_path / "banks_gap"), "--out", str(tmp_path / "out4"), "--generate", "--verifier", "schema"]) == 3
    # no bank at all and no --generate: a core stratum is a gap (3), the message says to generate
    with pytest.raises(B.CoreGapError, match="--generate"):
        B.build(plan, bank_root=tmp_path / "nowhere", out_dir=tmp_path / "out5", paper_dir=None, generate=False, verifier="schema", log=_quiet)
    assert B.main(["--plan", str(plan_file), "--bank-root", str(tmp_path / "nowhere"), "--out", str(tmp_path / "out6"), "--verifier", "schema"]) == 3


def test_generator_command_nice_and_resume(tmp_path):
    entry = R.BENCH_V1_PLAN["far060"]
    cmd = B.generator_command(entry, 0, tmp_path / "b0", jobs=4, nice=0)
    assert cmd[1:3] == ["-m", "data_tools.hero_reach_generator"] and "--profile" in cmd and cmd[cmd.index("--profile") + 1] == "hero_bench_v1_far060"
    assert cmd[cmd.index("--seed") + 1] == str(R.bench_v1_batch_seed("far060", 0)) and cmd[cmd.index("--n") + 1] == "180" and cmd[cmd.index("--clip-prefix") + 1] == "hero_bench_v1_far060"
    cmd1 = B.generator_command(entry, 1, tmp_path / "b1", jobs=60, nice=10)
    if shutil.which("nice"):
        assert cmd1[:3] == ["nice", "-n", "10"]
    assert cmd1[cmd1.index("--seed") + 1] == str(R.bench_v1_batch_seed("far060", 1)) and cmd1[cmd1.index("--clip-prefix") + 1] == "hero_bench_v1_far060_b1" and cmd1[cmd1.index("--jobs") + 1] == "60"
    (tmp_path / "b0").mkdir()
    (tmp_path / "b0" / "manifest.json").write_text("{}")
    assert B.run_generator(entry, 0, tmp_path / "b0", jobs=1, nice=0, log=_quiet) == tmp_path / "b0"      # resumable: no subprocess
    with pytest.raises(ValueError):
        B.batch_seed(entry, 7)


# ================================================================================================ 3. verbatim
def test_verbatim_import_sha_names_and_mismatch(tmp_path):
    paper = make_paper(tmp_path / "paper", per_cell=1)
    plan = tiny_plan("paper", per_cell=1)
    out = tmp_path / "out"
    man = B.build(plan, bank_root=None, out_dir=out, paper_dir=paper, verifier="schema", log=_quiet)
    rows = man["clips"]
    assert man["n_clips"] == 6 and all(r["tier"] == "core" and r["plan_stratum"] == "paper" and r["stratum"] == r["height_label"] for r in rows)
    src_rows = {r["file"]: r for r in json.loads((paper / "BENCH_MANIFEST.json").read_text())["clips"]}
    for r in rows:
        assert r["file"] in src_rows and (out / "core" / r["file"]).read_bytes() == (paper / r["file"]).read_bytes()
        assert r["sha256"] == src_rows[r["file"]]["sha256"] == _sha(out / "core" / r["file"])
        assert r["verbatim_from"] == {"build": "paper_protocol", "file": r["file"], "sha256": r["sha256"], "manifest_sha256": _sha(paper / "BENCH_MANIFEST.json"), "byte_identical": True}
        assert r["hold_end_frame"] == r["reach_end_frame"] + r["hold_frames"] == 101 and r["time_scale"] == 1.0 and r["license_class"] == "apache" and r["generator_tier"] == "core"
        assert r["path"] == f"core/{r['file']}" and r["retract_start_frame"] is None and r["pair_of"] is None
    assert (out / "tiers" / "paper.txt").read_text().split() == [r["path"] for r in rows] and (out / "core" / B.PAPER_LICENSE_NAME).is_file()
    assert man["strata"]["paper"]["verbatim"]["n_files"] == 6 and man["paper_protocol"]["manifest_sha256"] == _sha(paper / "BENCH_MANIFEST.json")
    # a changed byte in the paper corpus is refused (the shipped paper files are the authority)
    bad = tmp_path / "paper_bad"
    shutil.copytree(paper, bad)
    f = bad / rows[0]["file"]
    data = bytearray(f.read_bytes())
    data[-1] ^= 0xFF
    f.write_bytes(bytes(data))
    with pytest.raises(B.BuildError, match="sha256 differs"):
        B.build(plan, bank_root=None, out_dir=tmp_path / "out_bad", paper_dir=bad, verifier="schema", log=_quiet)
    # the quota must match the corpus exactly (no silent subset / gap)
    with pytest.raises(B.BuildError, match="does not match the plan quota"):
        B.build(tiny_plan("paper", per_cell=2), bank_root=None, out_dir=tmp_path / "out_q", paper_dir=paper, verifier="schema", log=_quiet)
    with pytest.raises(B.BuildError, match="--paper-dir"):
        B.build(plan, bank_root=None, out_dir=tmp_path / "out_nopaper", paper_dir=tmp_path / "missing", verifier="schema", log=_quiet)


# ================================================================================================ 4. re-timed strata
def test_retime_strata_naming_pair_of_frames_and_gate(tmp_path):
    paper = make_paper(tmp_path / "paper", per_cell=2, hot={("h074", "right", 0)})     # goal 0 is in the subset (smallest goal_index); goal 2 is not
    plan = tiny_plan("paper", "slow_x2", "hold6", "fast_x0p75", per_cell=1)
    plan["paper"] = tiny_plan("paper", per_cell=2)["paper"]        # the verbatim quota must match the 12-clip paper dir; the re-timed strata take 1 per cell
    out = tmp_path / "out"
    man = B.build(plan, bank_root=None, out_dir=out, paper_dir=paper, verifier="schema", log=_quiet)
    by = {name: [r for r in man["clips"] if r["plan_stratum"] == name] for name in ("paper", "slow_x2", "hold6", "fast_x0p75")}
    assert len(by["paper"]) == 12 and len(by["slow_x2"]) == 6 and len(by["hold6"]) == 6 and len(by["fast_x0p75"]) == 5
    sub = man["paper_sub60"]["files"]
    assert len(sub) == 6 and all(f[-6:-4] in ("00", "01") for f in sub)     # smallest goal_index per (height, hand): goals 0 (right) and 1 (left) of every height
    assert (out / "tiers" / "paper_sub60.txt").read_text().split() == [f"core/{f}" for f in sub]
    paper_rows = {r["clip_id"]: r for r in by["paper"]}
    for name, factor, hold_frames, tier in (("slow_x2", 2.0, 150, "extended"), ("hold6", 1.0, 300, "extended"), ("fast_x0p75", 0.75, 150, "stress")):
        for r in by[name]:
            assert r["file"] == f"{name}__{r['clip_id']}.npz" and r["stratum"] == name and r["tier"] == tier and r["path"] == f"{tier}/{r['file']}"
            twin = paper_rows[r["pair_of"]]
            assert r["clip_id"] == twin["clip_id"] and r["pair_file"] == twin["file"] and r["target_pos_w"] == twin["target_pos_w"] and r["hand"] == twin["hand"]
            assert r["time_scale"] == factor and r["hold_frames"] == hold_frames and r["hold_end_frame"] == r["reach_end_frame"] + r["hold_frames"] == r["n_frames"]
            assert r["reach_end_frame"] == round(factor * 60) and r["settle_frames"] == int(np.floor(factor * 15 + 1e-9)) and r["reach_frames"] == r["reach_end_frame"] - r["reach_start_frame"]
            assert r["retime"]["src_sha256"] == twin["sha256"] and r["retime"]["src_file"] == twin["file"] and r["accept_level_name"] == "strict" and r["generator_tier"] == "core"
            assert r["sha256"] == _sha(out / tier / r["file"]) and r["max_qdot_rad_s"] > 0 and r["max_qddot_rad_s2"] > 0 and r["retime_gate"] == (None if name != "fast_x0p75" else {"max_qdot_rad_s": 8.0, "max_qddot_rad_s2": 100.0})
            if name == "fast_x0p75":
                assert r["max_qdot_rad_s"] <= 8.0 and r["max_qddot_rad_s2"] <= 100.0
            with np.load(out / tier / r["file"], allow_pickle=False) as z:
                assert str(z["source_tag"]) == name and z["joint_pos"].shape[0] == r["n_frames"]
                np.testing.assert_allclose(z["joint_pos"][r["reach_end_frame"]:], np.broadcast_to(z["joint_pos"][-1], (r["hold_frames"], 36)), atol=2e-6)   # the hold IS the goal pose
    fast = man["strata"]["fast_x0p75"]
    assert fast["shipped"] == 5 and fast["shortfall"] == {"h074/right": 1} and fast["health"] == "short_gate_dropped" and fast["retime"]["n_gate_dropped"] == 1
    d = fast["retime"]["gate_dropped"][0]
    assert d["file"] == "fast_x0p75__hero_bench_v1_000100.npz" and d["pair_of"] == "hero_bench_v1_000100" and d["exceeded"] == ["max_qdot_rad_s", "max_qddot_rad_s2"] and d["max_qdot_rad_s"] > 8.0
    assert fast["retime"]["gate"] == {"max_qdot_rad_s": 8.0, "max_qddot_rad_s2": 100.0} and not (out / "stress" / "fast_x0p75__hero_bench_v1_000100.npz").exists()
    assert man["strata"]["slow_x2"]["retime"]["n_gate_dropped"] == 0 and man["strata"]["slow_x2"]["health"] == "ok" and man["strata"]["hold6"]["shortfall_total"] == 0
    assert man["tier_totals_shipped"] == {"core": 12, "extended": 12, "stress": 5}
    # shortfall_ok False + a gate that drops: the build refuses
    strict_plan = {"paper": tiny_plan("paper", per_cell=2)["paper"], "slow_x2": copy.deepcopy(plan["slow_x2"])}
    strict_plan["slow_x2"]["retime"]["gate"] = {"max_qdot_rad_s": 0.01}
    with pytest.raises(B.BuildError, match="shortfall_ok is False"):
        B.build(strict_plan, bank_root=None, out_dir=tmp_path / "out_strict", paper_dir=paper, verifier="schema", log=_quiet)


def test_retime_gate_helpers_and_default_behaviour_unchanged(tmp_path):
    assert RB.normalize_gate(None) is None and RB.normalize_gate({"qdot_max": 8, "qddot_max": 100}) == {"max_qdot_rad_s": 8.0, "max_qddot_rad_s2": 100.0}
    assert RB.normalize_gate({"max_qdot_rad_s": 8.0}) == {"max_qdot_rad_s": 8.0}
    with pytest.raises(ValueError):
        RB.normalize_gate({"bogus": 1.0})
    with pytest.raises(ValueError):
        RB.normalize_gate({"max_qdot_rad_s": -1.0})
    clip = synthetic_clip(101, dof_rate=3.0)
    lim = RB.dof_speed_limits(clip["joint_pos"], FPS)
    assert abs(lim["max_qdot_rad_s"] - 3.0) < 0.05 and 60.0 < lim["max_qddot_rad_s2"] < 90.0       # kink 0 -> 3 rad/s over one frame: 3 / 0.04 = 75
    assert RB.gate_check(lim, {"max_qdot_rad_s": 8.0, "max_qddot_rad_s2": 100.0}) == [] and RB.gate_check(lim, {"max_qdot_rad_s": 2.0}) == ["max_qdot_rad_s"]
    # no gate / no rename / no source_tag -> the receipt still carries the measured limits and nothing is dropped; names unchanged
    src = tmp_path / "src"
    src.mkdir()
    np.savez_compressed(src / "h050__a_000000.npz", **clip)
    receipt = RB.retime_bench(src, tmp_path / "out", 2.0, quiet=True)
    assert receipt["n_written"] == 1 and receipt["gate"] is None and receipt["gate_dropped"] == [] and receipt["clips"][0]["file"] == "h050__a_000000.npz" == receipt["clips"][0]["src_file"]
    assert receipt["clips"][0]["written"] is True and abs(receipt["clips"][0]["max_qdot_rad_s"] - 1.5) < 0.05
    with np.load(tmp_path / "out" / "h050__a_000000.npz") as z:
        assert str(z["source_tag"]) == "h050"
    # rename + source_tag + gate dropping everything
    receipt2 = RB.retime_bench(src, tmp_path / "out2", 0.75, quiet=True, gate={"max_qdot_rad_s": 1.0}, rename=lambda n: "fast__" + n.split("__", 1)[1], source_tag="fast")
    assert receipt2["n_written"] == 0 and receipt2["n_gate_dropped"] == 1 and receipt2["gate_dropped"][0]["file"] == "fast__a_000000.npz" and not list((tmp_path / "out2").glob("*.npz"))
    receipt3 = RB.retime_bench(src, tmp_path / "out3", 0.75, quiet=True, gate={"max_qdot_rad_s": 8.0}, rename=lambda n: "fast__" + n.split("__", 1)[1], source_tag="fast")
    assert receipt3["n_written"] == 1 and (tmp_path / "out3" / "fast__a_000000.npz").is_file()
    with np.load(tmp_path / "out3" / "fast__a_000000.npz") as z:
        assert str(z["source_tag"]) == "fast"


# ================================================================================================ 5. pool
def test_pool_takes_unselected_recoverable_rows_only(tmp_path):
    plan = tiny_plan("far060", "recov_pool", per_cell=1)
    plan["recov_pool"]["quota"] = {("far060",): 2}
    plan["recov_pool"]["pool"]["sources"] = ["far060"]
    plan["recov_pool"]["pool"]["per_source"] = 2
    entry = plan["far060"]
    # per cell: strict (selected), strict (unselected extra), recoverable x3, reject -> pool = recoverable rows only, smallest (batch, goal_index) first
    make_bank(tmp_path / "banks" / "far060" / "batch0", entry, 0, bank_row_specs(entry, ["strict", "strict", "recoverable", "recoverable", "recoverable", "reject"]))
    man = B.build(plan, bank_root=tmp_path / "banks", out_dir=tmp_path / "out", paper_dir=None, verifier="schema", log=_quiet)
    pool = [r for r in man["clips"] if r["plan_stratum"] == "recov_pool"]
    far = [r for r in man["clips"] if r["plan_stratum"] == "far060"]
    assert len(far) == 6 and len(pool) == 2
    shipped_ids = {r["clip_id"] for r in far}
    for r in pool:
        assert r["file"] == f"recov_pool__{r['clip_id']}.npz" and r["stratum"] == "recov_pool" and r["tier"] == "stress" and r["pool_source"] == "far060"
        assert r["accept_level"] == 2 and r["accept_level_name"] == "recoverable" and r["generator_tier"] == "recoverable" and r["flags"] == ["high_acceleration"]
        assert r["clip_id"] not in shipped_ids and r["terminal_ee_pos_err_m"] == 0.025 and (tmp_path / "out" / "stress" / r["file"]).is_file()
        assert r["target_pos_w"] and "ref_palm_pos_reach_end_w" not in r          # closed-loop target = the nominal goal, nothing extra (ruling #12)
    # the two pooled rows are the first recoverable candidates of the whole source in (batch, goal_index) order: goal 2 of h050/right, then goal 2 of h074/right
    # (goal_index restarts per height in the generator, so the smallest-goal_index rule interleaves the heights)
    assert sorted((r["height_label"], r["hand"], r["goal_index"]) for r in pool) == [("h050", "right", 2), ("h074", "right", 2)]
    st = man["strata"]["recov_pool"]
    assert st["shipped"] == 2 and st["cells"]["far060"] == {"source": "far060", "quota": 2, "n_candidates": 18, "n_selected": 2, "shortfall": 0, "source_built": True,
                                                            "selected": [[0, 2], [0, 2]], "n_by_level": {"recoverable": 18}}
    assert st["ik_residual_mm"]["mean"] == pytest.approx(25.0)
    # a source that was not built contributes nothing and is noted
    plan2 = copy.deepcopy(plan)
    plan2["recov_pool"]["quota"] = {("far060",): 1, ("floor_pick",): 1}
    man2 = B.build(plan2, bank_root=tmp_path / "banks", out_dir=tmp_path / "out2", paper_dir=None, verifier="schema", log=_quiet)
    assert man2["strata"]["recov_pool"]["shipped"] == 1 and man2["strata"]["recov_pool"]["shortfall"] == {"floor_pick": 1} and any("floor_pick" in n for n in man2["strata"]["recov_pool"]["notes"])


# ================================================================================================ 6. health rules (fill for every stratum, provisional flag, non-core shortfall ships)
def test_health_rules_fill_provisional_and_shortfall(tmp_path, monkeypatch):
    plan = tiny_plan("hover_above", "far070_bow", "high_h115_120", per_cell=2)
    assert all(plan[n]["fill_until_quota"] and plan[n]["fill_max_batches"] == 6 for n in plan)
    # banks on disk, no --generate: a short extended / stress stratum ships what exists and reports the shortfall (exit 0); nothing is dropped or halved
    make_bank(tmp_path / "banks" / "hover_above" / "batch0", plan["hover_above"], 0, bank_row_specs(plan["hover_above"], ["strict", "reject", "reject", "reject"]))        # yield 0.25 < 0.60
    make_bank(tmp_path / "banks" / "far070_bow" / "batch0", plan["far070_bow"], 0, bank_row_specs(plan["far070_bow"], ["strict", "strict", "reject", "reject"]))         # yield 0.50 >= 0.40
    make_bank(tmp_path / "banks" / "high_h115_120" / "batch0", plan["high_h115_120"], 0, bank_row_specs(plan["high_h115_120"], ["core", "reject", "reject", "reject"]))  # yield 0.25 < 0.40
    man = B.build(plan, bank_root=tmp_path / "banks", out_dir=tmp_path / "out", paper_dir=None, verifier="schema", log=_quiet)
    h = man["strata"]["hover_above"]
    assert h["provisional"] and h["health"] == "provisional_short" and h["yield"] == pytest.approx(0.25) and h["shipped"] == 6 and h["shortfall_total"] == 6 and h["quota_total"] == 12
    assert h["fill_until_quota"] and h["fill_max_batches"] == 6 and h["fill_batches_used"] == 1 and any("--generate not set" in n for n in h["notes"]) and any("shipped as is" in n for n in h["notes"])
    f = man["strata"]["far070_bow"]
    assert not f["provisional"] and f["health"] == "ok" and f["shipped"] == 12 and f["shortfall_total"] == 0 and f["yield"] == pytest.approx(0.5) and f["quota_total"] == 12
    hi = man["strata"]["high_h115_120"]
    assert hi["provisional"] and hi["health"] == "provisional_short" and hi["shipped"] == 4 and hi["shortfall_total"] == 4 and hi["yield"] == pytest.approx(0.25)   # 4 cells (h115 / h120 x hand)
    assert any("keep the 1.15 m shelf only" in n for n in hi["notes"]) and all(r["accept_level_name"] == "core" for r in man["clips"] if r["plan_stratum"] == "high_h115_120")
    assert "dropped" not in hi and man["tier_totals_shipped"] == {"core": 0, "extended": 6, "stress": 16}
    rep = (tmp_path / "out" / "BENCH_REPORT.md").read_text()
    assert "PROVISIONAL" in rep and "DROPPED" not in rep and "keep the 1.15 m shelf only" in rep and "hover_above" in rep and "Excluded ranges" in rep
    plan_file = tmp_path / "plan.json"
    plan_file.write_text(json.dumps(B.plan_json(plan)))
    assert B.main(["--plan", str(plan_file), "--bank-root", str(tmp_path / "banks"), "--out", str(tmp_path / "out_cli"), "--verifier", "schema"]) == 0   # non-core shortfall: exit 0
    # with --generate an extended stratum tops up across batches until full; the yield over the batches USED decides the provisional flag
    calls: list[tuple[str, int]] = []

    def gen(e, batch, out_dir, *, jobs=1, nice=0, log=print):
        calls.append((e["stratum"], batch))
        if (Path(out_dir) / "manifest.json").is_file():
            return Path(out_dir)
        return make_bank(Path(out_dir), e, batch, bank_row_specs(e, ["strict", "reject", "reject", "reject"] if batch == 0 else ["strict", "strict"]))

    monkeypatch.setattr(B, "run_generator", gen)
    man2 = B.build(tiny_plan("hover_above", per_cell=2), bank_root=tmp_path / "banks2", out_dir=tmp_path / "out2", paper_dir=None, generate=True, verifier="schema", log=_quiet)
    h2 = man2["strata"]["hover_above"]
    assert calls == [("hover_above", 0), ("hover_above", 1)] and h2["fill_batches_used"] == 2 and h2["shipped"] == 12 and h2["shortfall_total"] == 0
    assert h2["yield"] == pytest.approx(18 / 36) and h2["provisional"] and h2["health"] == "provisional" and [b["seed"] for b in h2["batches"]] == [R.bench_v1_batch_seed("hover_above", 0), R.bench_v1_batch_seed("hover_above", 1)]
    # a stress stratum whose cell never fills: every top-up batch is tried, then it ships the shortfall with exit 0 (only core exits 3)
    calls.clear()

    def hopeless(e, batch, out_dir, *, jobs=1, nice=0, log=print):
        calls.append(batch)
        if (Path(out_dir) / "manifest.json").is_file():
            return Path(out_dir)
        return make_bank(Path(out_dir), e, batch, [s for s in bank_row_specs(e, ["strict"]) if s["cell"] != ("h050", "left")] + [{"cell": ("h050", "left"), "level": "reject"}])

    monkeypatch.setattr(B, "run_generator", hopeless)
    sp = tiny_plan("far070_bow", per_cell=1)
    man3 = B.build(sp, bank_root=tmp_path / "banks3", out_dir=tmp_path / "out3", paper_dir=None, generate=True, verifier="schema", log=_quiet)
    s3 = man3["strata"]["far070_bow"]
    assert calls == list(range(7)) and s3["fill_batches_used"] == 7 and s3["shipped"] == 5 and s3["shortfall"] == {"h050/left": 1} and s3["health"] == "short" and not s3["provisional"]
    plan_file3 = tmp_path / "plan3.json"
    plan_file3.write_text(json.dumps(B.plan_json(sp)))
    assert B.main(["--plan", str(plan_file3), "--bank-root", str(tmp_path / "banks3"), "--out", str(tmp_path / "out3_cli"), "--generate", "--verifier", "schema"]) == 0


# ================================================================================================ 7. full build: manifest, tiers, sums, verify, scrub
@pytest.fixture
def full_build(tmp_path):
    paper = make_paper(tmp_path / "paper", per_cell=1)
    plan = tiny_plan("paper", "far060", "floor_pick", "slow_x2", "retract", "far070_bow", "fast_x0p75", "recov_pool", per_cell=1)
    plan["recov_pool"]["quota"] = {("far060",): 1, ("floor_pick",): 1, ("far070_bow",): 1}
    plan["recov_pool"]["pool"]["sources"] = ["far060", "floor_pick", "far070_bow"]
    banks = tmp_path / "banks"
    for name in ("far060", "floor_pick", "far070_bow"):
        make_bank(banks / name / "batch0", plan[name], 0, bank_row_specs(plan[name], ["strict", "recoverable"]))
    make_bank(banks / "retract" / "batch0", plan["retract"], 0, bank_row_specs(plan["retract"], ["strict"]), retract=True)
    out = tmp_path / "out"
    man = B.build(plan, bank_root=banks, out_dir=out, paper_dir=paper, verifier="schema", log=_quiet)
    return {"plan": plan, "paper": paper, "banks": banks, "out": out, "man": man, "tmp": tmp_path}


def test_full_build_manifest_rows_protocol_tiers_and_sums(full_build):
    man, out, plan = full_build["man"], full_build["out"], full_build["plan"]
    assert man["schema"] == "hero_bench_v1" and man["plan_version"] == "hero_bench_v1" and man["generator_version"] == "2.1.0" and man["builder"] == "data_tools.build_hero_bench"
    assert man["tier_totals_shipped"] == {"core": 12, "extended": 14, "stress": 15} and man["n_clips"] == 41 == len(man["clips"])
    # protocol block: the consumer-facing shape and a deterministic sha over the canonical JSON
    su = man["protocol"]["success"]
    assert su["open_loop"] == {"S7.5": {"pos_cm": 7.5, "rot_deg": 15}, "S5": {"pos_cm": 5, "rot_deg": 10}}
    assert su["closed_loop"] == {"C3": {"pos_cm": 3, "rot_deg": 15, "window": "final"}} and su["cdf_points_cm"] == [2.5, 5, 10] and su["stayed_threshold_cm"] == 1.75 and su["headline"] == "ee_global_active_cm"
    assert man["protocol"]["fail_free"]["common"] == ["fall", "anchor_xy"] and man["protocol"]["fail_free"]["fall_low_margin_m"] == 0.20 and man["protocol"]["fail_free"]["fall_low_default"] is False
    assert man["protocol"]["replan"]["first"] == "reach_end" and man["protocol"]["replan"]["period_s"] == 3.0 and man["protocol"]["replan"]["base"] == "current" and man["protocol"]["replan"]["goal_adjust"] is True
    assert man["protocol"]["replan"]["blend_s"] == 0.3 and man["protocol"]["replan"]["pad_s"] == 4 and man["protocol"]["replan"]["horizon_rule"] == "max(14, clip_s + 6)"
    assert man["protocol"]["plant"] == {"urdf": "g1_29dof_dex3fixed_hero.urdf", "foot_collision": "sonic_box", "physics_profile": "hero", "physics_hz": 1000, "policy_hz": 50,
                                        "self_collision": False, "reset": "frame 0, zero velocity", "no_table_geometry": True, "h_cmd": "auto"}
    assert man["protocol"]["odometry"]["mode"] == "so" and man["protocol"]["odometry"]["seed_rule"] == "0 + crc32(filename) % 2^31"
    assert man["protocol_sha256"] == hashlib.sha256(json.dumps(man["protocol"], sort_keys=True, separators=(",", ":")).encode()).hexdigest() == B.protocol_sha256(B.protocol_block())
    assert man["excluded_ranges"] == R.BENCH_V1_EXCLUDED_RANGES and len(man["excluded_ranges"]) == 7
    # every row: the fields the evaluators key on
    for r in man["clips"]:
        for k in ("file", "path", "clip_id", "stratum", "plan_stratum", "tier", "generator_tier", "height_label", "hand", "goal_index", "accept_level", "accept_level_name", "time_scale",
                  "hold_frames", "hold_end_frame", "retract_start_frame", "retract", "pair_of", "verbatim_from", "pool_source", "table", "license_class", "sha256", "bytes", "n_frames",
                  "reach_end_frame", "target_pos_w", "target_yaw_deg", "target_pitch_deg", "target_roll_deg", "source_tag", "terminal_ee_pos_err_m"):
            assert k in r, (k, r["file"])
        assert r["license_class"] == "apache" and r["path"] == f"{r['tier']}/{r['file']}" and (out / r["path"]).is_file() and r["sha256"] == _sha(out / r["path"])
        assert r["hold_end_frame"] == r["reach_end_frame"] + r["hold_frames"] and r["file"].split("__")[0] == r["stratum"] == r["source_tag"]
        with np.load(out / r["path"], allow_pickle=False) as z:
            assert z["joint_pos"].shape[0] == r["n_frames"] and str(z["source_tag"]) == r["source_tag"]
    floor = [r for r in man["clips"] if r["plan_stratum"] == "floor_pick"]
    assert len(floor) == 2 and all(r["height_label"] == "floor" and r["table"] is None and r["table_top_z"] is None and r["table_edge_x"] is None and r["has_table"] is False for r in floor)
    ret = [r for r in man["clips"] if r["plan_stratum"] == "retract"]
    assert len(ret) == 6 and all(r["retract_start_frame"] == r["hold_end_frame"] == 101 and r["has_retract"] and r["retract"]["start_frame"] == 101 and r["retract"]["hold_frames"] == 25 for r in ret)
    assert all(r["n_frames"] == 141 for r in ret) and all(r["retract_start_frame"] is None for r in man["clips"] if r["plan_stratum"] != "retract")
    pool = [r for r in man["clips"] if r["plan_stratum"] == "recov_pool"]
    assert sorted(r["pool_source"] for r in pool) == ["far060", "far070_bow", "floor_pick"] and {r["tier"] for r in pool} == {"stress"}
    assert sum(1 for r in man["clips"] if r["time_scale"] == 2.0) == 6 and sum(1 for r in man["clips"] if r["time_scale"] == 0.75) == 6
    # per-tier manifests and tiers/*.txt agree with the root manifest and the files on disk
    for t in B.TIERS:
        tm = json.loads((out / t / "BENCH_MANIFEST.json").read_text())
        assert tm["tier"] == t and tm["protocol_sha256"] == man["protocol_sha256"] and [r["file"] for r in tm["clips"]] == [r["file"] for r in man["clips"] if r["tier"] == t]
        assert tm["n_clips"] == man["tier_totals_shipped"][t] and tm["strata"] == man["strata"] and tm["excluded_ranges"] == man["excluded_ranges"]
        listed = (out / "tiers" / f"{t}.txt").read_text().split()
        assert listed == [r["path"] for r in man["clips"] if r["tier"] == t] and sorted(p.name for p in (out / t).glob("*.npz")) == sorted(Path(x).name for x in listed)
    assert (out / "tiers" / "all.txt").read_text().split() == [r["path"] for r in man["clips"]]
    assert len((out / "tiers" / "paper.txt").read_text().split()) == 6 and len((out / "tiers" / "paper_sub60.txt").read_text().split()) == 6
    # SHA256SUMS covers every npz + lists + licences, verify_shipped passes, the per-stratum stats are complete
    sums = dict(line.split("  ", 1)[::-1] for line in (out / "SHA256SUMS").read_text().splitlines())
    assert set(sums) >= {r["path"] for r in man["clips"]} and "DATA_LICENSE" in sums and "core/LICENSE_paper_verbatim.txt" in sums and "tiers/all.txt" in sums
    assert all(sums[r["path"]] == r["sha256"] for r in man["clips"]) and (out / "DATA_LICENSE").read_text().startswith("hero_bench_v1") and "Apache License" in (out / "DATA_LICENSE").read_text()
    v = B.verify_shipped(out)
    assert v["ok"] and v["n_identical"] == 41 and v["mismatched"] == [] and v["missing"] == [] and v["extra_npz"] == [] and v["sums_checked"] == len(sums) and v["protocol_sha256_recomputed"] == man["protocol_sha256"]
    assert set(man["strata"]) == set(plan) and all({"quota", "shipped", "yield", "health", "provisional", "shortfall", "cells", "ik_residual_mm"} <= set(s) for s in man["strata"].values() if s["kind"] == "generate")
    assert man["strata"]["far060"]["fill_batches_used"] == 1 and man["strata"]["far060"]["batches"][0]["seed"] == R.bench_v1_batch_seed("far060", 0)
    assert man["verification"]["core"]["n_loaded"] == 12 and man["verification"]["core"]["n_skipped"] == 0 and "validate_holosoma_npz" in man["verification"]["core"]["verifier"]
    rep = (out / "BENCH_REPORT.md").read_text()
    for s in ("## Strata", "## Stratum × height × hand", "## Per-stratum yields and cells", "## Excluded ranges", "## Provisional / shortfall notes", "## Protocol", "floor", "recov_pool", "S7.5"):
        assert s in rep


def test_protocol_sha_is_deterministic_across_builds(full_build):
    man = full_build["man"]
    second = B.build(full_build["plan"], bank_root=full_build["banks"], out_dir=full_build["tmp"] / "out_again", paper_dir=full_build["paper"], verifier="schema", log=_quiet)
    assert second["protocol_sha256"] == man["protocol_sha256"] and second["protocol"] == man["protocol"]
    assert [r["sha256"] for r in second["clips"] if r["verbatim_from"]] == [r["sha256"] for r in man["clips"] if r["verbatim_from"]]
    assert second["tier_totals_shipped"] == man["tier_totals_shipped"] and [r["file"] for r in second["clips"]] == [r["file"] for r in man["clips"]]


def test_verify_shipped_detects_a_flipped_byte_and_extra_files(full_build):
    out = full_build["out"]
    target = out / full_build["man"]["clips"][5]["path"]
    data = bytearray(target.read_bytes())
    data[100] ^= 0x01
    target.write_bytes(bytes(data))
    v = B.verify_shipped(out)
    assert not v["ok"] and v["mismatched"] == [full_build["man"]["clips"][5]["path"]] and v["sums_mismatched"] == [full_build["man"]["clips"][5]["path"]]
    target.write_bytes(bytes(data[:100] + bytes([data[100] ^ 0x01]) + data[101:]))
    (out / "stress" / "zz__extra.npz").write_bytes(b"x")
    v2 = B.verify_shipped(out)
    assert not v2["ok"] and v2["extra_npz"] == ["stress/zz__extra.npz"] and v2["mismatched"] == []


@pytest.mark.parametrize("change", ["missing", "goal", "frames", "hand", "protocol", "protocol_hash", "duplicate", "malformed"])
def test_verify_shipped_rejects_tier_scoring_metadata_different_from_root(full_build, change):
    out = full_build["out"]
    tier_path = out / "core" / B.MANIFEST_NAME
    if change == "missing":
        tier_path.unlink()
    elif change == "malformed":
        tier_path.write_text("[]")
    else:
        tier = json.loads(tier_path.read_text())
        if change == "goal":
            tier["clips"][0]["target_pos_w"][0] += 1.0
        elif change == "frames":
            tier["clips"][0]["reach_end_frame"] += 1
        elif change == "hand":
            tier["clips"][0]["hand"] = "right" if tier["clips"][0]["hand"] == "left" else "left"
        elif change == "protocol":
            tier["protocol"]["success"]["closed_loop"]["C3"]["pos_cm"] = 99.0
        elif change == "protocol_hash":
            tier["protocol_sha256"] = "0" * 64
        else:
            tier["clips"].append(tier["clips"][0])
        tier_path.write_text(json.dumps(tier))
    result = B.verify_shipped(out)
    assert not result["ok"]
    assert not result["tier_manifests"]["core"]["consistent_with_root"] or not result["tier_manifests"]["core"]["protocol_sha256_matches"]


def test_verify_shipped_rejects_truncated_checksum_coverage(full_build):
    out = full_build["out"]
    (out / B.SUMS_NAME).write_text("")
    result = B.verify_shipped(out)
    assert not result["ok"] and result["sums_checked"] == 0
    assert full_build["man"]["clips"][0]["path"] in result["sums_missing"]


def test_public_scrub_removes_hosts_paths_and_internal_suffixes(full_build):
    man, out = full_build["man"], full_build["out"]
    raw = json.dumps(man)
    assert "ws-" in raw and "/root/" in raw and "+bench-wip" in raw and str(full_build["banks"]) in raw and str(full_build["paper"]) in raw   # the private manifest does carry them
    pub = B.scrub_public_manifest({**man, "s3_mirror": "s3://bucket/hero_bench_v1", "nodes": ["ws-example-node", "ok"], "git_sha": "abcdef0+bench-wip"})
    txt = json.dumps(pub)
    for bad in ("/root/", "/Users/", "s3://", "ws-", "+bench-wip", str(full_build["banks"]), str(full_build["paper"]), "/private/", "/tmp/"):
        assert bad not in txt, bad
    assert pub["public"] is True and "host" not in pub and "bank_root" not in pub and "paper_dir" not in pub and pub["git_sha"] == "abcdef0"
    assert pub["protocol_sha256"] == man["protocol_sha256"] and pub["protocol"] == man["protocol"] and pub["excluded_ranges"] == man["excluded_ranges"]
    assert [r["sha256"] for r in pub["clips"]] == [r["sha256"] for r in man["clips"]] and pub["strata"]["far060"]["batches"][0]["seed"] == man["strata"]["far060"]["batches"][0]["seed"]
    assert pub["strata"]["far060"]["batches"][0]["solver_config"] == {"max_iters": 12} and pub["strata"]["far060"]["batches"][0]["gate_thresholds"] == {"strict_qdot": 6.0}
    assert pub["strata"]["far060"]["batches"][0]["mjcf_sha256"] == man["strata"]["far060"]["batches"][0]["mjcf_sha256"] and pub["strata"]["far060"]["batches"][0]["git_sha"] == "abc1234"
    assert "host" not in pub["paper_protocol"] and "bank_dir" not in pub["paper_protocol"] and pub["paper_protocol"]["bank"]["git_sha"] == "123abcd" and "mjcf" not in pub["paper_protocol"]["bank"] and pub["paper_protocol"]["manifest_sha256"] == man["paper_protocol"]["manifest_sha256"]
    assert all(r["hold_end_frame"] == r["reach_end_frame"] + r["hold_frames"] for r in pub["clips"]) and pub["generator_version"] == "2.1.0" and pub["plan_version"] == "hero_bench_v1"
    # in-place scrub of a built dir keeps it verifiable (the sums do not cover the manifests) and rewrites the per-tier copies + report
    B.scrub_public_dir(out)
    root = json.loads((out / "BENCH_MANIFEST.json").read_text())
    assert root["public"] and "host" not in root and B.verify_shipped(out)["ok"]
    for t in B.TIERS:
        tm = json.loads((out / t / "BENCH_MANIFEST.json").read_text())
        assert tm["public"] and "host" not in tm and tm["tier"] == t
    rep = (out / "BENCH_REPORT.md").read_text()
    assert "ws-" not in rep and "/root/" not in rep
    # --public at build time and the --scrub-only CLI path
    plan_file = full_build["tmp"] / "plan.json"
    plan_file.write_text(json.dumps(B.plan_json(full_build["plan"])))
    assert B.main(["--plan", str(plan_file), "--bank-root", str(full_build["banks"]), "--v1-dir", str(full_build["paper"]), "--out", str(full_build["tmp"] / "out_pub"), "--verifier", "schema", "--public"]) == 0   # --v1-dir = alias of --paper-dir
    pub2 = json.loads((full_build["tmp"] / "out_pub" / "BENCH_MANIFEST.json").read_text())
    assert pub2["public"] and "ws-" not in json.dumps(pub2) and pub2["n_clips"] == 41
    assert B.main(["--out", str(out), "--scrub-only"]) == 0


def test_plan_helpers_roundtrip_and_cli_plan_print(tmp_path, capsys):
    plan = B.load_plan("default")
    assert list(plan) == list(R.BENCH_V1_PLAN) and plan["recov_pool"]["quota"] == R.BENCH_V1_PLAN["recov_pool"]["quota"]
    back = B.plan_from_json(json.loads(json.dumps(B.plan_json(plan))))
    assert back["orient_core"]["quota"] == plan["orient_core"]["quota"] and back["recov_pool"]["quota"] == plan["recov_pool"]["quota"] and back["fast_x0p75"]["retime"] == plan["fast_x0p75"]["retime"]
    red = B.load_plan("reduced:1")
    assert all(max(e["quota"].values()) == 1 for e in red.values()) and red["far060"]["n_candidates"] == 12 and red["orient_core"]["n_candidates"] == 24 and red["floor_pick"]["n_candidates"] == 4
    assert red["far060"]["seed"] == R.BENCH_V1_PLAN["far060"]["seed"] and B.batch_seed(red["far060"], 1) == R.bench_v1_batch_seed("far060", 1)
    assert B.main(["--print-plan"]) == 0
    out = capsys.readouterr().out
    assert "| far060 | core | generate |" in out and "core 420, extended 618, stress 260" in out
    with pytest.raises(B.BuildError):
        B.load_plan(str(tmp_path / "missing.json"))
    assert B.paper_subset([{"height_label": "h050", "hand": "right", "goal_index": g} for g in (4, 0, 2)], {("h050", "right"): 2}) == [{"height_label": "h050", "hand": "right", "goal_index": 0},
                                                                                                                                          {"height_label": "h050", "hand": "right", "goal_index": 2}]


# ================================================================================================ 8. the real hero_bench_v1 corpus (skipped when absent)
@pytest.mark.skipif(not (REAL_PAPER / "BENCH_MANIFEST.json").is_file() or len(list(REAL_PAPER.glob("*.npz"))) != 180, reason=f"paper-protocol set (180 clips) not present at {REAL_PAPER} (HERO_BENCH_PAPER_DIR)")
def test_real_paper_verbatim_import_all_180_sha_match(tmp_path):
    plan = {"paper": copy.deepcopy(R.BENCH_V1_PLAN["paper"])}
    out = tmp_path / "out"
    man = B.build(plan, bank_root=None, out_dir=out, paper_dir=REAL_PAPER, verifier="schema", log=_quiet)
    src = {r["file"]: r for r in json.loads((REAL_PAPER / "BENCH_MANIFEST.json").read_text())["clips"]}
    assert man["n_clips"] == 180 and len(src) == 180 and man["tier_totals_shipped"] == {"core": 180, "extended": 0, "stress": 0}
    for r in man["clips"]:
        assert r["sha256"] == src[r["file"]]["sha256"] == _sha(out / "core" / r["file"]) and r["verbatim_from"]["sha256"] == r["sha256"] and r["stratum"] == r["height_label"]
        assert r["hold_end_frame"] == r["reach_end_frame"] + r["hold_frames"] == r["n_frames"] and r["tier"] == "core"
    sub = man["paper_sub60"]["files"] if man["paper_sub60"]["files"] else B.paper_subset(list(src.values()), R.BENCH_V1_PLAN["slow_x2"]["quota"])
    assert len(sub) == 60
    assert B.verify_shipped(out)["ok"] and man["verification"]["core"]["n_loaded"] == 180


# ================================================================================================ 9. scripts/hero_bench.py
def test_hero_bench_cli_plan_build_and_verify(full_build, capsys):
    assert hero_bench.main(["plan"]) == 0
    out_txt = capsys.readouterr().out
    assert "| paper | core | verbatim |" in out_txt and "core 420, extended 618, stress 260" in out_txt
    assert hero_bench.main(["plan", "--json", "--plan", "reduced:1"]) == 0
    pj = json.loads(capsys.readouterr().out)
    assert pj["far060"]["quota"]["h050/right"] == 1 and pj["far060"]["n_candidates"] == 12
    # build through the CLI wrapper with the same flags as the module CLI (+ the provenance flag --code-sha256)
    plan_file = full_build["tmp"] / "plan_cli.json"
    plan_file.write_text(json.dumps(B.plan_json(full_build["plan"])))
    out = full_build["tmp"] / "out_cli"
    assert hero_bench.main(["build", "--plan", str(plan_file), "--bank-root", str(full_build["banks"]), "--paper-dir", str(full_build["paper"]), "--out", str(out), "--verifier", "schema",
                          "--code-sha256", "deadbeef" * 8]) == 0
    capsys.readouterr()
    man = json.loads((out / "BENCH_MANIFEST.json").read_text())
    assert man["n_clips"] == 41 and man["protocol_sha256"] == full_build["man"]["protocol_sha256"]
    assert man["code_sha256"] == "deadbeef" * 8 and man["release_version"] == B.release_version() == full_build["man"]["release_version"]
    # verify passes on the built dir, fails (exit 1) after flipping a byte, and reports the file
    assert hero_bench.main(["verify", "--dir", str(out)]) == 0
    assert "OK" in capsys.readouterr().out
    target = out / man["clips"][3]["path"]
    data = bytearray(target.read_bytes())
    data[50] ^= 0x10
    target.write_bytes(bytes(data))
    assert hero_bench.main(["verify", "--dir", str(out)]) == 1
    txt = capsys.readouterr().out
    assert "FAILED" in txt and man["clips"][3]["path"] in txt
    assert hero_bench.main(["verify", "--dir", str(out), "--json"]) == 1
    res = json.loads(capsys.readouterr().out)
    assert res["mismatched"] == [man["clips"][3]["path"]] and not res["ok"]
    assert hero_bench.main(["verify", "--dir", str(full_build["tmp"] / "nowhere")]) == 1
