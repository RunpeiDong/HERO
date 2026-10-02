"""hero_bench_v1 plan and sampler checks (specs only, no MuJoCo solve).

Covers: the legacy hero_bench_v1 / hero_bench_v2 banks stay byte-identical, the plan table (strata, seeds, prefixes, quotas,
admitted levels, top-up batches), every generated stratum's profile samples and covers its quota cells, the new sampler knobs
(bench_has_table, bench_y_side, bench_time_scale_range, bench_lean_pelvis_share), the new orientation families with per-family
caps and hand mirroring, the CLI override parsing and the bench frame labels of the generator."""
from __future__ import annotations

import hashlib
import json
import math
from collections import Counter, defaultdict

import numpy as np
import pytest

from data_tools import reach_specs as rs

LEGACY_HASHES = {
    # sha256 of the canonical (rounded, key-sorted) JSON of every spec, recorded before the stratified-plan change (2026-10-01); the bank seeds of the shipped sets
    ("hero_bench_v1", 20260906, 30): "b430ee68de1b8356de42a7d11a4d3390268ba85cce283f0356540d2ec8418ff0",
    ("hero_bench_v1", 20260906, 300): "65a5c5aaf136d333eae1c431c84ad2f28befa5d803bc300cf69c4e90db251880",
    ("hero_bench_v2", 20260907, 30): "4fdcbf5b630308450a8170c7a018a0723202ad61bc0bfdb2841de6960dc099b0",
    ("hero_bench_v2", 20260907, 900): "a3cdc6d1fa23795e33f10775a73a3218049423dc7e04c38a055dd29eb5a8f11b",
    ("hero_reach_replan_v1", 20260923, 60): "c2daa6e2bb722a0a547f53de61553eee17d35e3666bd445cc4d8bdedf25d7e04",
    ("hero_reach_far_v2", 20260923, 60): "7a0e41562776fd1928258c7d6cfdfb9c279cff25a36b1247658b486032fe4948",
    ("hero_reach_replan_v2", 20260923, 60): "cd939735d3184e93ba7d7d672d28908f77020e2be556f3ebebb81beb238fb2c4",
}
FAR_V1_LIKE_HASH = "fa4521112e5658515cd8cfb0557d916225d2b90c9f640e4c7c4da1398e3231c7"   # hero_bench_v1 sampler, bench_x_range 0.50-0.70, seed 20260923
GENERATE = [name for name, e in rs.BENCH_V1_PLAN.items() if e["kind"] == "generate"]


def _canon(o):
    """Platform-stable view of a spec: floats rounded to 10 significant digits, keys sorted (the raw ``to_json`` text differs between
    numpy 1.x and 2.x scalar reprs although the sampled values are identical)."""
    if isinstance(o, float):
        return float(f"{o:.10g}")
    if isinstance(o, dict):
        return {k: _canon(v) for k, v in sorted(o.items())}
    if isinstance(o, (list, tuple)):
        return [_canon(v) for v in o]
    return o


def _hash(specs) -> str:
    return hashlib.sha256("\n".join(json.dumps(_canon(json.loads(s.to_json())), sort_keys=True, separators=(",", ":")) for s in specs).encode()).hexdigest()


def _cell(b: dict) -> tuple[str, ...]:
    return (b["height_label"], b["hand"]) + ((b["orient_family"],) if "orient_family" in b else ())


# --------------------------------------------------------------------------------------------------
# legacy banks
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("profile,seed,n", sorted(LEGACY_HASHES))
def test_legacy_bench_profiles_are_byte_identical(profile, seed, n):
    specs = rs.sample_clip_specs(n, seed, profile=profile)
    assert _hash(specs) == LEGACY_HASHES[(profile, seed, n)]
    b = specs[0].bench
    assert "stratum" not in b and "has_table" not in b and "y_side" not in b and b["time_scale"] == 1.0   # no stratified-plan bookkeeping on legacy banks


def test_far_v1_style_override_bank_is_byte_identical():
    assert _hash(rs.sample_clip_specs(60, 20260923, profile="hero_bench_v1", bench_x_range=[0.50, 0.70])) == FAR_V1_LIKE_HASH
    assert list(rs.BENCH_V2_ORIENT_FAMILIES) == ["fan_side", "top_down", "tilted"]
    assert not any(k in fam for fam in rs.BENCH_V2_ORIENT_FAMILIES.values() for k in ("yaw_sign_random", "roll_mirror_by_hand", "max_abs_roll_deg"))


# --------------------------------------------------------------------------------------------------
# plan table
# --------------------------------------------------------------------------------------------------
def test_plan_strata_tiers_kinds_and_totals():
    assert rs.BENCH_SCHEMA_V1 == "hero_bench_v1"
    assert list(rs.BENCH_V1_PLAN) == ["paper", "far060", "cross_mid", "mid_h062_h080", "orient_core",
                                      "low_h030_h040", "floor_pick", "high_h100_h110", "wide_lat", "orient_ext", "hover_above", "slow_x2", "hold6", "retract", "close",
                                      "far070_bow", "low_h025", "high_h115_120", "fast_x0p75", "recov_pool"]
    assert "off_axis" not in rs.BENCH_V1_PLAN and "mid_same" not in rs.BENCH_V1_PLAN   # reserved for a later revision
    assert rs.bench_v1_tier_totals() == {"core": 420, "extended": 618, "stress": 260} == rs.BENCH_V1_TIER_TOTALS
    assert sum(rs.bench_v1_tier_totals().values()) == 1298
    kinds = {name: e["kind"] for name, e in rs.BENCH_V1_PLAN.items()}
    assert kinds["paper"] == "verbatim" and kinds["recov_pool"] == "pool" and {kinds[k] for k in ("slow_x2", "hold6", "fast_x0p75")} == {"retime"}
    assert len(GENERATE) == 15 and all(e["profile"] is None for n, e in rs.BENCH_V1_PLAN.items() if n not in GENERATE)
    quota = {n: rs.bench_v1_quota_total(n) for n in rs.BENCH_V1_PLAN}
    assert quota == {"paper": 180, "far060": 60, "cross_mid": 60, "mid_h062_h080": 60, "orient_core": 60, "low_h030_h040": 80, "floor_pick": 60,
                     "high_h100_h110": 60, "wide_lat": 60, "orient_ext": 90, "hover_above": 48, "slow_x2": 60, "hold6": 60, "retract": 60, "close": 40,
                     "far070_bow": 60, "low_h025": 40, "high_h115_120": 40, "fast_x0p75": 60, "recov_pool": 60}
    cand = {n: rs.BENCH_V1_PLAN[n]["n_candidates"] for n in GENERATE}
    assert cand == {"far060": 180, "cross_mid": 96, "mid_h062_h080": 84, "orient_core": 120, "low_h030_h040": 160, "floor_pick": 150, "high_h100_h110": 120,
                    "wide_lat": 108, "orient_ext": 135, "hover_above": 72, "retract": 84, "close": 56, "far070_bow": 180, "low_h025": 100, "high_h115_120": 120}
    assert sum(cand.values()) == 1765
    for name, e in rs.BENCH_V1_PLAN.items():
        assert e["tier"] in rs.BENCH_V1_TIERS and e["stratum"] == name and set(e["admitted_levels"]) <= {"strict", "core", "recoverable"}
        assert all(isinstance(k, tuple) and v > 0 for k, v in e["quota"].items())
        if e["kind"] == "generate":
            assert e["fill_until_quota"] is True and e["fill_max_batches"] == 6      # every generated stratum tops up until its quota is full
        else:
            assert e["fill_until_quota"] is False and e["fill_max_batches"] == 0
        if e["kind"] == "generate" and e["tier"] == "core":
            assert e["admitted_levels"] == ["strict"] and e["min_yield"] is None
        if e["kind"] == "generate" and e["tier"] != "core":
            assert e["min_yield"] == (0.60 if e["tier"] == "extended" else 0.40)
            assert e["admitted_levels"] == (["strict"] if name == "hover_above" else ["strict", "core"])
    assert rs.BENCH_V1_PLAN["far070_bow"]["fallback"] and rs.BENCH_V1_PLAN["high_h115_120"]["fallback"]
    # verbatim / retime / pool blocks
    assert rs.BENCH_V1_PLAN["paper"]["verbatim_from"]["build"] == "paper_protocol" and rs.BENCH_V1_PLAN["paper"]["verbatim_from"]["rule"] == "all 180"
    assert rs.BENCH_V1_PLAN["slow_x2"]["retime"] == {"factor": 2.0, "hold_s": 3.0, "source_subset": "paper_sub60", "gate": None, "shortfall_ok": False}
    assert rs.BENCH_V1_PLAN["hold6"]["retime"]["factor"] == 1.0 and rs.BENCH_V1_PLAN["hold6"]["retime"]["hold_s"] == 6.0
    fast = rs.BENCH_V1_PLAN["fast_x0p75"]["retime"]
    assert fast["factor"] == 0.75 and fast["hold_s"] == 3.0 and fast["gate"] == {"max_qdot_rad_s": 8.0, "max_qddot_rad_s2": 100.0} and fast["shortfall_ok"]
    pool = rs.BENCH_V1_PLAN["recov_pool"]["pool"]
    assert pool == {"sources": ["far060", "floor_pick", "high_h100_h110", "far070_bow", "high_h115_120"], "per_source": 12, "level": "recoverable",
                    "order": "smallest goal_index first"}
    assert rs.BENCH_V1_PLAN["recov_pool"]["quota"] == {(s,): 12 for s in pool["sources"]} and rs.BENCH_V1_PLAN["recov_pool"]["admitted_levels"] == ["recoverable"]
    assert rs.PAPER_SUB60_RULE["per_height_per_hand"] == 10 and rs.PAPER_SUB60_RULE["total"] == 60 and rs.PAPER_SUB60_RULE["layer"] == "paper"
    for name in ("slow_x2", "hold6", "fast_x0p75"):
        assert rs.BENCH_V1_PLAN[name]["quota"] == {(h, hand): 10 for h in ("h050", "h074", "h088") for hand in ("right", "left")}
    assert len(rs.BENCH_V1_EXCLUDED_RANGES) == 7 and all({"range", "measured", "disposition"} <= set(r) for r in rs.BENCH_V1_EXCLUDED_RANGES)
    json.dumps(rs.bench_v1_plan_json())   # serialisable (tuple cells joined)
    assert rs.bench_v1_plan_json()["orient_core"]["quota"]["h050/right/top_down"] == 5 and rs.bench_v1_plan_json()["far060"]["quota_total"] == 60


def test_plan_seeds_and_prefixes_are_unique_and_disjoint_from_legacy_banks():
    legacy_seeds = {20260906, 20260907, 20260923}
    legacy_prefixes = {"hero_reach_hero_bench_v1", "hero_reach_hero_bench_v2", "hero_reach_far_v1", "hero_bench_v1", "hero_bench_v2"}
    seeds = [e["seed"] for e in rs.BENCH_V1_PLAN.values()]
    prefixes = [e["clip_prefix"] for e in rs.BENCH_V1_PLAN.values()]
    assert seeds == [20261001 + k for k in range(20)] and len(set(seeds)) == 20 and not set(seeds) & legacy_seeds
    assert len(set(prefixes)) == 20 and not set(prefixes) & legacy_prefixes and all(p == f"hero_bench_v1_{n}" for n, p in zip(rs.BENCH_V1_PLAN, prefixes))
    # top-up batches: seed + 1000 k never collides with another stratum's batch or a legacy seed
    batch_seeds = Counter()
    for name, e in rs.BENCH_V1_PLAN.items():
        for k in range(e["fill_max_batches"] + 1):
            batch_seeds[rs.bench_v1_batch_seed(name, k)] += 1
    assert max(batch_seeds.values()) == 1 and not set(batch_seeds) & legacy_seeds
    assert rs.bench_v1_batch_seed("far060", 0) == 20261002 and rs.bench_v1_batch_seed("far060", 6) == 20267002
    with pytest.raises(ValueError):
        rs.bench_v1_batch_seed("far060", 7)
    with pytest.raises(ValueError):
        rs.bench_v1_batch_seed("slow_x2", 1)   # not a generated stratum: no top-up batches
    # sampler default clip prefix is unique per stratum too (differs from the v1 / v2 defaults)
    for name in GENERATE:
        s = rs.sample_clip_specs(None, 1, profile=rs.BENCH_V1_PLAN[name]["profile"])[0]
        assert s.clip_id.startswith(f"hero_reach_hero_bench_v1_{name}_") and s.profile == "hero_bench_v1" and s.bench["schema"] == "hero_bench_v1"


# --------------------------------------------------------------------------------------------------
# profiles
# --------------------------------------------------------------------------------------------------
@pytest.mark.parametrize("name", GENERATE)
def test_generate_profiles_sample_deterministically_and_cover_their_quota_cells(name):
    e = rs.BENCH_V1_PLAN[name]
    assert e["profile"] == f"hero_bench_v1_{name}" and e["profile"] in rs.PROFILES
    cfg = rs.PROFILES[e["profile"]]
    assert cfg["sampler"] == "bench" and cfg["bench_schema"] == "hero_bench_v1" and cfg["bench_stratum"] == name and cfg["bench_tier"] == e["tier"]
    assert cfg["bench_candidates_per_height"] * len(cfg["bench_heights"]) == e["n_candidates"]
    specs = rs.sample_clip_specs(None, e["seed"], profile=e["profile"])
    again = rs.sample_clip_specs(None, e["seed"], profile=e["profile"])
    assert len(specs) == e["n_candidates"] and [s.to_json() for s in specs] == [s.to_json() for s in again]
    assert _hash(rs.sample_clip_specs(None, e["seed"] + 1, profile=e["profile"])) != _hash(specs)
    cells = Counter(_cell(s.bench) for s in specs)
    assert set(cells) == set(e["quota"]), (set(cells) ^ set(e["quota"]))
    assert all(cells[c] >= q for c, q in e["quota"].items())
    x_lo, x_hi = cfg["bench_x_range"]
    ay_lo, ay_hi = cfg["bench_abs_y_range"]
    z_lo, z_hi = cfg["bench_z_above_range"]
    for i, s in enumerate(specs):
        b = s.bench
        seg = s.segments[0]
        assert b["stratum"] == name and b["tier"] == e["tier"] and b["plan_version"] == "hero_bench_v1" and b["candidate_index"] == i
        assert b["hold_s"] == 3.0 == seg.hold_s and seg.retarget is None and len(s.segments) == 1
        assert x_lo - 1e-9 <= b["target_x_raw"] <= x_hi + 1e-9 and ay_lo - 1e-9 <= abs(seg.targets[0].pos[1]) <= ay_hi + 1e-9
        assert b["height_m"] + z_lo - 1e-9 <= b["target_z_raw"] <= b["height_m"] + z_hi + 1e-9 and seg.targets[0].pos[2] >= b["target_z_raw"] - 1e-12
        assert seg.base.foot_mode == "flat" and seg.base.family == ("squat" if b["height_m"] < 0.60 else "stand")
        assert b["time_scale"] == seg.time_scale and b["time_scale_range"][0] <= seg.time_scale <= b["time_scale_range"][1]
        assert seg.retract is (name == "retract") and b["retract"] is (name == "retract")
        assert b["has_table"] is seg.has_table is (name != "floor_pick")
        assert rs.spec_from_dict(json.loads(s.to_json())).to_json() == s.to_json()
    json.loads(rs.specs_to_jsonl(specs).splitlines()[0])
    rs.summarize_specs(specs)


def test_profile_knobs_follow_the_plan_table():
    P = rs.PROFILES
    v1 = P["hero_bench_v1"]
    assert P["hero_bench_v1_far060"]["bench_x_range"] == [0.50, 0.60] and (P["hero_bench_v1_far060"]["bench_stand_lean_deg_max"], P["hero_bench_v1_far060"]["bench_squat_lean_deg_max"]) == (60.0, 65.0)
    assert P["hero_bench_v1_cross_mid"]["bench_heights"] == [0.50, 0.62, 0.74] and P["hero_bench_v1_cross_mid"]["bench_abs_y_range"] == [0.02, 0.20] and P["hero_bench_v1_cross_mid"]["bench_y_side"] == "cross"
    assert P["hero_bench_v1_mid_h062_h080"]["bench_heights"] == [0.62, 0.80] and P["hero_bench_v1_mid_h062_h080"]["bench_x_range"] == v1["bench_x_range"]
    oc = P["hero_bench_v1_orient_core"]["bench_orient_families"]
    assert list(oc) == ["top_down", "tilted"] and oc["top_down"]["yaw_deg"] == [-75.0, 75.0] and oc["top_down"]["pitch_deg"] == [-85.0, -45.0]
    assert oc["tilted"]["yaw_deg"] == [-55.0, 55.0] and oc["tilted"]["pitch_deg"] == [-35.0, 15.0] and oc["tilted"]["roll_deg"] == [25.0, 45.0]
    assert "bench_yaw_deg" not in P["hero_bench_v1_orient_core"] and P["hero_bench_v1_orient_core"]["bench_max_abs_roll_deg"] == 45.0
    low = P["hero_bench_v1_low_h030_h040"]
    assert low["bench_heights"] == [0.30, 0.40] and low["bench_x_range"] == [0.40, 0.55] and low["bench_squat_drop_range"] == [0.24, 0.34] and low["bench_squat_lean_deg_max"] == 65.0 and low["bench_edge_gap_range"] == [0.22, 0.30]
    fl = P["hero_bench_v1_floor_pick"]
    assert fl["bench_heights"] == [0.0] and fl["bench_has_table"] is False and fl["bench_x_range"] == [0.30, 0.45] and fl["bench_abs_y_range"] == [0.12, 0.32]
    assert fl["bench_z_above_range"] == [0.25, 0.38] and fl["bench_pitch_deg"] == [-20.0, 10.0] and fl["bench_squat_drop_range"] == [0.29, 0.36] and fl["bench_squat_lean_deg_max"] == 70.0
    hi = P["hero_bench_v1_high_h100_h110"]
    assert hi["bench_heights"] == [1.00, 1.10] and hi["bench_table_depth"] == 0.30 and hi["bench_x_range"] == [0.26, 0.40] and hi["bench_abs_y_range"] == [0.15, 0.35]
    assert hi["bench_pitch_deg"] == [-10.0, 15.0] and hi["bench_time_scale_range"] == [1.3, 1.3] and hi["bench_stand_lean_deg_max"] == 45.0
    assert P["hero_bench_v1_wide_lat"]["bench_x_range"] == [0.30, 0.45] and P["hero_bench_v1_wide_lat"]["bench_abs_y_range"] == [0.35, 0.45]
    assert list(P["hero_bench_v1_orient_ext"]["bench_orient_families"]) == ["palm_down", "palm_up", "fan_wide"]
    assert P["hero_bench_v1_hover_above"]["bench_z_above_range"] == [0.15, 0.30]
    assert P["hero_bench_v1_retract"]["bench_retract_share"] == 1.0 and P["hero_bench_v1_retract"]["bench_x_range"] == v1["bench_x_range"]
    cl = P["hero_bench_v1_close"]
    assert cl["bench_heights"] == [0.50, 0.74] and cl["bench_x_range"] == [0.31, 0.36] and cl["bench_abs_y_range"] == [0.05, 0.20] and (cl["bench_stand_lean_deg_max"], cl["bench_squat_lean_deg_max"]) == (40.0, 40.0)
    bow = P["hero_bench_v1_far070_bow"]
    assert bow["bench_x_range"] == [0.60, 0.70] and bow["bench_lean_pelvis_share"] == 0.7 and (bow["bench_stand_lean_deg_max"], bow["bench_squat_lean_deg_max"]) == (80.0, 85.0) and bow["bench_stand_drop_range"] == [0.00, 0.08]
    l25 = P["hero_bench_v1_low_h025"]
    assert l25["bench_heights"] == [0.25] and l25["bench_squat_drop_range"] == [0.30, 0.36] and l25["bench_edge_gap_range"] == [0.22, 0.30]
    h12 = P["hero_bench_v1_high_h115_120"]
    assert h12["bench_heights"] == [1.15, 1.20] and h12["bench_table_depth"] == 0.30 and h12["bench_x_range"] == [0.20, 0.32] and h12["bench_abs_y_range"] == [0.12, 0.30] and h12["bench_time_scale_range"] == [1.4, 1.4]
    # untouched v1 knobs are inherited everywhere
    for name in GENERATE:
        cfg = P[f"hero_bench_v1_{name}"]
        assert cfg["bench_edge_ref"] == "toe" and cfg["bench_approach"] == "ingress" and cfg["bench_squat_below_m"] == 0.60 and cfg["bench_stance_half_width_range"] == [0.11, 0.14]
    # the legacy profiles did not pick up stratified-plan keys
    for name in ("hero_bench_v1", "hero_bench_v2", "hero_reach_replan_v1", "hero_reach_far_v2", "hero_reach_replan_v2"):
        assert not any(k in P[name] for k in ("bench_stratum", "bench_tier", "bench_plan_version", "bench_has_table", "bench_y_side", "bench_time_scale_range", "bench_lean_pelvis_share"))


# --------------------------------------------------------------------------------------------------
# new sampler knobs
# --------------------------------------------------------------------------------------------------
def test_floor_profile_has_no_table():
    specs = rs.sample_clip_specs(None, rs.BENCH_V1_PLAN["floor_pick"]["seed"], profile="hero_bench_v1_floor_pick")
    assert len(specs) == 150 and Counter(s.hands_mode for s in specs) == {"right": 75, "left": 75}
    for s in specs:
        seg, b = s.segments[0], s.bench
        assert seg.has_table is False and seg.table_depth == 0.0 and seg.table_edge_gap == 0.0 and seg.stratum == "floor" and seg.surface_z == 0.0
        assert b["table"] is None and b["height_label"] == "floor" and b["height_m"] == 0.0 and b["has_table"] is False
        assert all(b[k] is None for k in ("table_top_z", "table_edge_x", "table_edge_gap_m", "table_edge_ref", "table_edge_ref_x", "table_depth"))
        assert b["target_x_shifted"] is False and abs(b["target_x_raw"] - seg.targets[0].pos[0]) < 1e-12   # no slab -> no edge shift
        assert 0.25 - 1e-9 <= seg.targets[0].pos[2] <= 0.38 + 1e-9 and seg.targets[0].pos[2] >= rs.ee_min_height_above_surface(seg.targets[0].rotation(), seg.targets[0].hand) - 1e-9
        assert seg.base.family == "squat" and 0.29 - 1e-9 <= seg.base.pelvis_drop <= 0.36 + 1e-9 and seg.base.foot_mode == "flat"   # pelvis 0.397-0.467 m
        assert -20.0 - 1e-9 <= b["target_pitch_deg"] <= 10.0 + 1e-9
    with pytest.raises(ValueError, match="bench_heights == \\[0.0\\]"):
        rs.sample_clip_specs(6, 1, profile="hero_bench_v1", bench_heights=[0.74], bench_has_table=False)
    # the knob also works on the v1 profile when the heights are the floor
    fv1 = rs.sample_clip_specs(4, 1, profile="hero_bench_v1", bench_heights=[0.0], bench_has_table=False, bench_z_above_range=[0.25, 0.38])
    assert all(not s.segments[0].has_table and s.bench["table"] is None and "stratum" not in s.bench for s in fv1)


def test_y_side_cross_and_mixed():
    cross = rs.sample_clip_specs(None, rs.BENCH_V1_PLAN["cross_mid"]["seed"], profile="hero_bench_v1_cross_mid")
    assert len(cross) == 96 and {s.bench["height_label"] for s in cross} == {"h050", "h062", "h074"}
    for s in cross:
        y = s.segments[0].targets[0].pos[1]
        assert (y > 0) == (s.hands_mode == "right") and 0.02 - 1e-9 <= abs(y) <= 0.20 + 1e-9   # opposite side of the reaching hand
        assert s.bench["y_side"] == "cross" and s.bench["cross_side"] is True
        assert s.segments[0].base.family == ("squat" if s.bench["height_m"] < 0.60 else "stand")
    same = rs.sample_clip_specs(None, rs.BENCH_V1_PLAN["cross_mid"]["seed"], profile="hero_bench_v1_cross_mid", bench_y_side="same")
    for a, b in zip(cross, same):
        pa, pb = a.segments[0].targets[0].pos, b.segments[0].targets[0].pos
        assert pa[0] == pb[0] and pa[2] == pb[2] and abs(pa[1] + pb[1]) < 1e-12   # same draws, mirrored y only
    mixed = rs.sample_clip_specs(60, 3, profile="hero_bench_v1", bench_heights=[0.74], bench_y_side="mixed", bench_stratum="probe")
    per_hand = defaultdict(list)
    for s in mixed:
        per_hand[s.hands_mode].append(s.bench["cross_side"])
    assert per_hand["right"] == [False, True] * 15 and per_hand["left"] == [False, True] * 15
    assert all(((s.segments[0].targets[0].pos[1] < 0) == (s.hands_mode == "right")) != s.bench["cross_side"] for s in mixed)
    with pytest.raises(ValueError, match="bench_y_side"):
        rs.sample_clip_specs(6, 1, profile="hero_bench_v1", bench_heights=[0.74], bench_y_side="left")
    with pytest.raises(ValueError, match="non-negative"):
        rs.sample_clip_specs(6, 1, profile="hero_bench_v1", bench_heights=[0.74], bench_abs_y_range=[-0.2, -0.05])


def test_time_scale_range_pinned_and_drawn():
    for name, ts in (("high_h100_h110", 1.3), ("high_h115_120", 1.4)):
        specs = rs.sample_clip_specs(None, rs.BENCH_V1_PLAN[name]["seed"], profile=f"hero_bench_v1_{name}")
        assert all(s.segments[0].time_scale == ts and s.bench["time_scale"] == ts and s.bench["time_scale_range"] == [ts, ts] for s in specs)
        assert all(s.segments[0].table_depth == 0.30 and s.segments[0].has_table for s in specs)
    pinned = rs.sample_clip_specs(30, 4, profile="hero_bench_v1", bench_heights=[0.74], bench_stratum="probe")
    drawn = rs.sample_clip_specs(30, 4, profile="hero_bench_v1", bench_heights=[0.74], bench_stratum="probe", bench_time_scale_range=[1.1, 1.5])
    ts = [s.segments[0].time_scale for s in drawn]
    assert all(1.1 <= t <= 1.5 for t in ts) and len(set(ts)) > 20 and all(s.bench["time_scale"] == s.segments[0].time_scale for s in drawn)
    assert all(a.segments[0].targets[0].pos == b.segments[0].targets[0].pos for a, b in zip(pinned, drawn))   # the goal geometry is untouched
    assert all(s.segments[0].time_scale == 1.0 for s in pinned)
    with pytest.raises(ValueError, match="positive"):
        rs.sample_clip_specs(6, 1, profile="hero_bench_v1", bench_heights=[0.74], bench_time_scale_range=[0.0, 1.0])


def test_lean_pelvis_share():
    bow = rs.sample_clip_specs(None, rs.BENCH_V1_PLAN["far070_bow"]["seed"], profile="hero_bench_v1_far070_bow")
    leaned = 0
    for s in bow:
        base = s.segments[0].base
        assert s.bench["lean_pelvis_share"] == 0.7 and abs(base.waist_pitch_fraction - 0.3) < 1e-12
        lean = base.pelvis_pitch / 0.7
        assert abs(base.waist_pitch - min(0.3 * lean, rs.WAIST_PITCH_LIMIT)) < 1e-9
        assert lean <= math.radians(85.0 if base.family == "squat" else 80.0) + 1e-9
        leaned += base.pelvis_pitch > math.radians(30.0)
    assert leaned > len(bow) // 2   # the bow strategy really leans the pelvis
    with pytest.raises(ValueError, match="bench_lean_pelvis_share"):
        rs.sample_clip_specs(6, 1, profile="hero_bench_v1", bench_heights=[0.74], bench_lean_pelvis_share=0.0)
    v1 = rs.sample_clip_specs(6, 1, profile="hero_bench_v1", bench_heights=[0.74])
    assert all(abs(s.segments[0].base.waist_pitch_fraction - 0.6) < 1e-12 for s in v1)


# --------------------------------------------------------------------------------------------------
# orientation families
# --------------------------------------------------------------------------------------------------
def test_orient_ext_families_palm_down_up_fan_wide():
    specs = rs.sample_clip_specs(None, rs.BENCH_V1_PLAN["orient_ext"]["seed"], profile="hero_bench_v1_orient_ext")
    by = defaultdict(list)
    for s in specs:
        by[(s.bench["orient_family"], s.hands_mode)].append(s)
    assert set(by) == {(f, h) for f in ("palm_down", "palm_up", "fan_wide") for h in ("right", "left")}
    for (fam, hand), ss in by.items():
        for s in ss:
            b, t = s.bench, s.segments[0].targets[0]
            R = t.rotation()
            palm_normal = R @ np.array([0.0, 1.0 if hand == "right" else -1.0, 0.0])   # Dex3 palm normal: toward the midline at roll 0
            assert b["orient_limits"]["max_abs_roll_deg"] == (100.0 if fam != "fan_wide" else 45.0)
            assert b["orient_limits"]["max_rot_from_canonical_deg"] == (90.0 if fam == "fan_wide" else 80.0)
            assert b["rot_from_canonical_deg"] <= b["orient_limits"]["max_rot_from_canonical_deg"] + 1e-9
            if fam == "fan_wide":
                assert 50.0 - 1e-9 <= abs(b["yaw_deg"]) <= 80.0 + 1e-9 and b["yaw_sign"] == math.copysign(1.0, b["yaw_deg"]) and b["roll_mirrored_by_hand"] is False
                assert -15.0 - 1e-9 <= b["pitch_deg"] <= 10.0 + 1e-9 and abs(b["roll_deg"]) <= 10.0 + 1e-9 and b["canonical_ypr_deg"] == [0.0, 0.0, 0.0]
            else:
                assert 70.0 - 1e-9 <= abs(b["roll_deg"]) <= 100.0 + 1e-9 and abs(b["yaw_deg"]) <= 30.0 + 1e-9 and abs(b["pitch_deg"]) <= 10.0 + 1e-9
                assert b["roll_mirrored_by_hand"] is (hand == "right") and b["yaw_sign"] == 1.0
                # the band is written for the left hand; the right hand is mirrored so that the family name holds for both hands
                left_sign = 1.0 if fam == "palm_down" else -1.0
                assert math.copysign(1.0, b["roll_deg"]) == (left_sign if hand == "left" else -left_sign)
                assert b["roll_sign"] == (-1.0 if hand == "right" else 1.0)   # the sign applied to the band (mirror only; no random roll sign)
                assert b["canonical_ypr_deg"] == [0.0, 0.0, 90.0 * math.copysign(1.0, b["roll_deg"])]
                assert (palm_normal[2] < -0.5) if fam == "palm_down" else (palm_normal[2] > 0.5)
        assert {math.copysign(1.0, s.bench["yaw_deg"]) for s in ss} == {1.0, -1.0} if fam == "fan_wide" else True
    # direct sampler: per-family cap admits palm_down under the default 45 deg profile cap; without the key it is rejected
    fam = rs.BENCH_V1_ORIENT_FAMILIES["palm_down"]
    o = rs.sample_bench_orientation(np.random.default_rng(0), fam, [0.5, 0.5, 0.5], max_abs_roll_deg=45.0, max_rot_deg=80.0, hand="left")
    assert o["orient_resamples"] == 0 and abs(o["roll_deg"] - 85.0) < 1e-9 and o["max_abs_roll_deg"] == 100.0
    o_r = rs.sample_bench_orientation(np.random.default_rng(0), fam, [0.5, 0.5, 0.5], max_abs_roll_deg=45.0, max_rot_deg=80.0, hand="right")
    assert abs(o_r["roll_deg"] + 85.0) < 1e-9 and o_r["roll_mirrored_by_hand"] and o_r["canonical_ypr_deg"] == [0.0, 0.0, -90.0]
    with pytest.raises(RuntimeError, match="consecutive hard rejects"):
        rs.sample_bench_orientation(np.random.default_rng(0), {k: v for k, v in fam.items() if k != "max_abs_roll_deg"}, [0.5, 0.5, 0.5],
                                    max_abs_roll_deg=45.0, max_rot_deg=80.0, hand="left", max_tries=20)
    with pytest.raises(ValueError, match="hand"):
        rs.sample_bench_orientation(np.random.default_rng(0), fam, [0.5, 0.5, 0.5], max_abs_roll_deg=45.0, max_rot_deg=80.0)
    # legacy families: no extra draws, same output as before for the same rng state
    o_v2 = rs.sample_bench_orientation(np.random.default_rng(0), rs.BENCH_V2_ORIENT_FAMILIES["fan_side"], [0.25, 0.5, 1.0], max_abs_roll_deg=45.0, max_rot_deg=80.0)
    assert o_v2["orient_resamples"] == 0 and abs(o_v2["yaw_deg"] + 30.0) < 1e-9 and o_v2["yaw_sign"] == 1.0 and o_v2["roll_sign"] == 1.0 and not o_v2["roll_mirrored_by_hand"]


def test_orient_core_widened_bands():
    specs = rs.sample_clip_specs(None, rs.BENCH_V1_PLAN["orient_core"]["seed"], profile="hero_bench_v1_orient_core")
    cells = Counter((s.bench["height_label"], s.bench["orient_family"], s.hands_mode) for s in specs)
    assert set(cells.values()) == {10}
    by = defaultdict(list)
    for s in specs:
        by[s.bench["orient_family"]].append(s.bench)
    assert all(-85.0 - 1e-9 <= b["pitch_deg"] <= -45.0 + 1e-9 and abs(b["yaw_deg"]) <= 75.0 + 1e-9 and abs(b["roll_deg"]) <= 20.0 + 1e-9 for b in by["top_down"])
    assert all(-35.0 - 1e-9 <= b["pitch_deg"] <= 15.0 + 1e-9 and abs(b["yaw_deg"]) <= 55.0 + 1e-9 and 25.0 - 1e-9 <= abs(b["roll_deg"]) <= 45.0 + 1e-9 for b in by["tilted"])
    assert max(abs(b["yaw_deg"]) for b in by["top_down"]) > 60.0 and min(b["pitch_deg"] for b in by["top_down"]) < -80.0   # really wider than v2
    assert all(b["clearance_raised"] and b["clearance_margin_m"] == 0.035 for b in by["top_down"])
    assert all(b["rot_from_canonical_deg"] <= 80.0 + 1e-9 and abs(b["roll_deg"]) <= 45.0 for b in by["top_down"] + by["tilted"])
    assert {b["roll_sign"] for b in by["tilted"]} == {1.0, -1.0}


# --------------------------------------------------------------------------------------------------
# generator-side helpers (import needs mujoco / mink, both hard dependencies of the package)
# --------------------------------------------------------------------------------------------------
def test_generator_override_parsing_and_bench_frame_labels():
    from data_tools import hero_reach_generator as g

    out = g._parse_overrides(["bench_has_table=false", "bench_y_side=cross", "bench_time_scale_range=1.2/1.4", "bench_lean_pelvis_share=0.7", "bench_heights=0.5/0.74"])
    assert out == {"bench_has_table": False, "bench_y_side": "cross", "bench_time_scale_range": [1.2, 1.4], "bench_lean_pelvis_share": 0.7, "bench_heights": [0.5, 0.74]}
    assert g._parse_overrides(["bench_has_table=True"]) == {"bench_has_table": True} and g._parse_overrides(["bench_has_table=0"]) == {"bench_has_table": False}
    with pytest.raises(ValueError, match="boolean"):
        g._parse_overrides(["bench_has_table=maybe"])
    # the parsed overrides drive the sampler
    specs = rs.sample_clip_specs(4, 1, profile="hero_bench_v1", bench_heights=[0.0], **g._parse_overrides(["bench_has_table=false", "bench_z_above_range=0.25/0.38"]))
    assert all(not s.segments[0].has_table for s in specs)
    assert g.bench_frame_labels(15, 100, 150) == {"reach_start_frame": 15, "reach_end_frame": 115, "reach_frames": 100, "hold_frames": 150,
                                                  "hold_end_frame": 265, "retract_start_frame": None}
    assert g.bench_frame_labels(15, 100, 150, retract_start_frame=265)["retract_start_frame"] == 265
    with pytest.raises(ValueError):
        g.bench_frame_labels(15, -1, 150)
