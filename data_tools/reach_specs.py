"""Sample target specifications for HERO IK reaching-data generation.

Coordinates use a heading-aligned frame: x forward, y left, z up. Samples vary
palm goals, root height, torso strategy, foot support, reach duration, and target
changes. Analytic leg priors reject infeasible flat-foot configurations before
the whole-body IK solve. Profile overrides allow additional motion coverage.

Robot constants are shared with hero_isaacsim.constants. Angles are radians
unless a field explicitly names degrees; positions and distances are metres."""

from __future__ import annotations

import dataclasses
import json
import math
from typing import Any, Iterable, Sequence

import numpy as np

from hero_isaacsim.constants import (  # noqa: E402
    DOF_NAMES,
    EE_BODY_NAMES,
    FOOT_CONTACT_POINT_OFFSET,
    HOLOSOMA_BODY_NAMES_32,
    LEFT_ARM_DOF_IDX,
    LOWER_DOF_IDX,
    PALM_BODY_NAMES,
    PALM_OFFSET,
    RIGHT_ARM_DOF_IDX,
    UPPER_DOF_IDX,
    WAIST_DOF_IDX,
)

LEFT_LEG_DOF_IDX: tuple[int, ...] = tuple(range(0, 6))
RIGHT_LEG_DOF_IDX: tuple[int, ...] = tuple(range(6, 12))

# Default G1 joint angles in canonical joint order.
DEFAULT_JOINT_POS: dict[str, float] = {name: 0.0 for name in DOF_NAMES}
DEFAULT_JOINT_POS.update({
    "left_hip_pitch_joint": -0.312, "left_knee_joint": 0.669, "left_ankle_pitch_joint": -0.363,
    "right_hip_pitch_joint": -0.312, "right_knee_joint": 0.669, "right_ankle_pitch_joint": -0.363,
    "left_shoulder_pitch_joint": 0.2, "left_shoulder_roll_joint": 0.2, "left_elbow_joint": 0.6,
    "right_shoulder_pitch_joint": 0.2, "right_shoulder_roll_joint": -0.2, "right_elbow_joint": 0.6,
})
DEFAULT_DOF_ARRAY: np.ndarray = np.array([DEFAULT_JOINT_POS[n] for n in DOF_NAMES], dtype=np.float64)

# Joint ranges (rad) of the joints the analytic leg model reasons about (MuJoCo model).
HIP_PITCH_RANGE: tuple[float, float] = (-2.5307, 2.8798)
KNEE_RANGE: tuple[float, float] = (-0.0873, 2.8798)
ANKLE_PITCH_RANGE: tuple[float, float] = (-0.8727, 0.5236)
WAIST_PITCH_LIMIT: float = 0.52
WAIST_YAW_LIMIT: float = 2.618
FLAT_FOOT_CLOSURE_MAX: float = 0.87
# Thigh-torso clearance requires an opening angle of at least about 0.95 rad.
# The opening angle is pi - |hip_pitch| - waist_pitch.
THIGH_TORSO_MAX_FOLD: float = 2.05    # |hip_pitch| + waist_pitch <= 2.05 rad (opening >= ~63 deg; shoulder/knee clearance)
KNEE_SOFT_MAX: float = 2.4            # above this the plan routes to heel-lift / kneel variants
FLAT_FOOT_MIN_HEIGHT: float = 0.35    # below this the plan routes to heel-lift / kneel variants


@dataclasses.dataclass(frozen=True)
class LegGeometry:
    """Sagittal-plane chain pelvis -> hip_pitch -> knee -> ankle_pitch -> sole (metres)."""

    hip_drop: float = 0.1027        # pelvis origin to hip pitch axis (vertical)
    hip_lateral: float = 0.1186     # pelvis midline to knee/ankle (lateral)
    thigh: float = 0.3366           # hip pitch axis to knee axis
    shank: float = 0.30001          # knee axis to ankle pitch axis
    ankle_to_sole: float = 0.05256  # ankle pitch axis to sole plane (0.01756 + SOLE_BELOW_ANKLE)
    toe_x: float = 0.12             # sole contact spheres (ankle_roll frame): toes (+-0.03 y), heels (+-0.025 y)
    heel_x: float = -0.05
    foot_half_width: float = 0.03


LEG = LegGeometry()
# Sole plane below the ankle_roll origin in the Dex3 MuJoCo scene: the four r=5 mm contact spheres are
# centred at z=-0.030 so their bottoms (and the visual foot mesh, z_min=-0.0354) touch the floor at -0.035.
# holosoma's ``*_foot_contact_point`` sits at -0.037 (FOOT_CONTACT_POINT_OFFSET); the 2 mm difference is
# below every gate in this module and is documented in the bank labels (``sole_below_ankle_m``).
SOLE_BELOW_ANKLE: float = 0.035
assert abs(abs(FOOT_CONTACT_POINT_OFFSET[2]) - SOLE_BELOW_ANKLE) < 0.005
TOE_LOCAL: tuple[float, float, float] = (0.12, 0.0, -SOLE_BELOW_ANKLE)
HEEL_LOCAL: tuple[float, float, float] = (-0.05, 0.0, -SOLE_BELOW_ANKLE)
SOLE_LOCAL: tuple[float, float, float] = (0.0, 0.0, -SOLE_BELOW_ANKLE)
# Sole contact points in the ankle_roll frame (toe L/R, heel L/R) — the support polygon of a flat foot.
SOLE_POINTS_LOCAL: tuple[tuple[float, float, float], ...] = (
    (0.12, 0.03, -SOLE_BELOW_ANKLE), (0.12, -0.03, -SOLE_BELOW_ANKLE),
    (-0.05, 0.025, -SOLE_BELOW_ANKLE), (-0.05, -0.025, -SOLE_BELOW_ANKLE),
)
STANDING_ROOT_Z: float = 0.7567  # holosoma default pose, both soles on the floor (MuJoCo FK: ankle_roll z 0.7217 + 0.035)
# Soft pelvis XY target behind the stance centre. The centre-of-mass task
# keeps the solved pelvis near the origin; labels record ``pelvis_xy_frame0``.
PELVIS_X_BEHIND_STANCE: float = 0.025


# ----------------------------------------------------------------------------------
# Strata / profiles
# ----------------------------------------------------------------------------------
# name -> (surface z low, surface z high, default episode weight)
STRATA: dict[str, tuple[float, float, float]] = {
    "floor": (0.00, 0.15, 0.15),
    "very_low": (0.15, 0.45, 0.20),
    "low": (0.45, 0.65, 0.20),
    "standard": (0.65, 0.85, 0.20),
    "high": (0.85, 1.15, 0.15),
    "overhead": (1.15, 1.35, 0.10),
}
STRATUM_NAMES: tuple[str, ...] = tuple(STRATA)
DEFAULT_STRATA_WEIGHTS: dict[str, float] = {k: v[2] for k, v in STRATA.items()}

# finger-elevation sub-range preferred per stratum (70 % of samples; 30 % use the full range)
PITCH_FULL_RANGE_DEG: tuple[float, float] = (-75.0, 45.0)
PITCH_PREFERRED_DEG: dict[str, tuple[float, float]] = {
    "floor": (-75.0, -15.0),
    "very_low": (-70.0, -5.0),
    "low": (-50.0, 15.0),
    "standard": (-40.0, 20.0),
    "high": (-20.0, 30.0),
    "overhead": (-10.0, 45.0),
}
ROLL_RANGE_DEG: tuple[float, float] = (-40.0, 40.0)
YAW_SAMPLE_DEG: tuple[float, float] = (-90.0, 90.0)
YAW_CLIP_DEG: float = 70.0
X_FRONT_RANGE: tuple[float, float] = (-0.10, 0.70)
X_BEHIND_RANGE: tuple[float, float] = (-0.45, -0.10)
Y_RANGE: tuple[float, float] = (-0.60, 0.60)
Y_BEHIND_ABS_RANGE: tuple[float, float] = (0.20, 0.60)
EE_ABOVE_SURFACE_RANGE: tuple[float, float] = (0.05, 0.20)
BIMANUAL_DISTANCE_RANGE: tuple[float, float] = (0.30, 0.60)
HAND_MODE_SHARES: dict[str, float] = {"right": 0.40, "left": 0.40, "bimanual": 0.20}
SEGMENT_COUNT_SHARES: dict[int, float] = {1: 0.40, 2: 0.35, 3: 0.25}

PROFILES: dict[str, dict[str, Any]] = {
    "broad_v2": {
        "strata_weights": dict(DEFAULT_STRATA_WEIGHTS),
        "behind_share": 0.05,
        "retarget_share": 0.20,
        "retract_share": 0.20,
        "canonical_share": 0.30,
        "hold_long_share": 0.35,
        "forced_variant_share": 0.10,
    },
    "floor_feasibility_200": {
        "strata_weights": {"floor": 15.0 / 35.0, "very_low": 20.0 / 35.0},
        "behind_share": 0.0,
        "retarget_share": 0.0,
        "retract_share": 0.0,
        "canonical_share": 0.30,
        "hold_long_share": 0.35,
        "forced_variant_share": 0.10,
        "segment_count_shares": {1: 1.0},
    },
    # HERO paper reaching setup: table heights 0.50 / 0.74 / 0.88 m, front edge
    # 0.30 m ahead of the origin, and side grasps with |yaw| <= 45 deg.
    # ``sample_bench_specs`` alternates hands, with one reach and a 3 s hold per
    # candidate. Override parameters with ``--override k=v``; lists use ``a/b/c``.
    "hero_bench_v1": {
        "sampler": "bench",
        "strata_weights": {},
        "bench_heights": [0.50, 0.74, 0.88],          # table top z (m)
        "bench_candidates_per_height": 100,           # default when n is not given (n = heights x candidates)
        "bench_x_range": [0.30, 0.50],                # heading frame (pelvis xy at frame 0 = origin), m
        "bench_abs_y_range": [0.10, 0.35],            # |y|; sign by hand (right hand -> y < 0)
        "bench_z_above_range": [0.05, 0.15],          # palm point above the table top before the hand-envelope clearance raise
        "bench_clearance_margin_m": 0.02,             # z >= table + hand-envelope drop + margin (== HAND_SURFACE_CLEARANCE_MARGIN, the generator's rule)
        "bench_yaw_deg": [-45.0, 45.0],               # grasp yaw about the canonical inward-facing side grasp (yaw 0)
        "bench_pitch_deg": [-15.0, 10.0],
        "bench_roll_deg": [-10.0, 10.0],
        "bench_edge_gap_range": [0.16, 0.20],
        "bench_edge_ref": "toe",                      # "toe" (foot front, 0.12 m ahead of the ankle line) | "pelvis"
        "bench_table_depth": 0.60,
        "bench_hold_s": 3.0,
        "bench_approach": "ingress",
        "bench_sweep_clearance_extra_m": 0.0,         # extra height of the approach sweep over the table (planner only; the goal is unchanged)
        "bench_squat_below_m": 0.60,                  # table tops below this use the squat family
        "bench_stand_drop_range": [0.00, 0.04],       # pelvis drop below STANDING_ROOT_Z (m)
        "bench_squat_drop_range": [0.08, 0.14],       # pelvis ~0.62-0.68 m
        "bench_stand_lean_deg_max": 45.0,             # pelvis <= 18 deg, waist <= 27 deg
        "bench_squat_lean_deg_max": 55.0,             # pelvis <= 22 deg, waist <= 30 deg (WAIST_PITCH_LIMIT clamp)
        "bench_stance_half_width_range": [0.11, 0.14],
        "bench_retract_share": 0.0,
    },
}
BENCH_SCHEMA: str = "hero_bench_v1"
BENCH_RETRACT_STREAM: int = 0x5E7AC7   # second word of the retract-flag rng seed (np.random.default_rng([seed, BENCH_RETRACT_STREAM]))
BENCH_SCHEMA_V2: str = "hero_bench_v2"

BENCH_V2_ORIENT_FAMILIES: dict[str, dict[str, Any]] = {
    "fan_side": {"share": 0.50, "yaw_deg": [-60.0, 60.0], "pitch_deg": [-20.0, 15.0], "roll_deg": [-15.0, 15.0], "roll_sign_random": False,
                 "canonical_ypr_deg": [0.0, 0.0, 0.0], "clearance_margin_m": 0.02,   # == HAND_SURFACE_CLEARANCE_MARGIN (asserted below)
                 "note": "canonical inward side grasp fanned in yaw (the fan)"},
    "top_down": {"share": 0.25, "yaw_deg": [-60.0, 60.0], "pitch_deg": [-80.0, -50.0], "roll_deg": [-20.0, 20.0], "roll_sign_random": False,
                 "canonical_ypr_deg": [0.0, -90.0, 0.0], "clearance_margin_m": 0.035,
                 "note": "fingers pointing down; palm point raised to the fingertip envelope (~0.21 m above the table)"},
    "tilted": {"share": 0.25, "yaw_deg": [-45.0, 45.0], "pitch_deg": [-30.0, 10.0], "roll_deg": [25.0, 45.0], "roll_sign_random": True,
               "canonical_ypr_deg": [0.0, 0.0, 35.0], "clearance_margin_m": 0.02,
               "note": "side grasp rolled about the finger axis by 25-45 deg (sign random; canonical roll follows the sign)"},
}
BENCH_V2_MAX_ORIENT_TRIES: int = 500   # consecutive hard rejects before the sampler gives up (only reachable with bad overrides)

PROFILES["hero_bench_v2"] = {
    **{k: v for k, v in PROFILES["hero_bench_v1"].items() if k not in ("bench_yaw_deg", "bench_pitch_deg", "bench_roll_deg")},
    "bench_schema": BENCH_SCHEMA_V2,
    # Allocate candidates by orientation-family share using largest remainders.
    # Goal indices follow BENCH_V2_ORIENT_FAMILIES order; hands alternate within
    # each family block, with one extra right-hand candidate for odd block sizes.
    "bench_candidates_per_height": 300,
    "bench_orient_families": {k: dict(v) for k, v in BENCH_V2_ORIENT_FAMILIES.items()},
    "bench_max_abs_roll_deg": 45.0,
    "bench_max_rot_from_canonical_deg": 80.0,
}

PROFILES["hero_reach_replan_v1"] = {
    **PROFILES["hero_bench_v2"],
    "bench_x_range": [0.30, 0.70],
    "bench_heights": [0.45, 0.55, 0.65, 0.74, 0.82, 0.90],
    "bench_stand_lean_deg_max": 60.0,
    "bench_squat_lean_deg_max": 65.0,
    "bench_candidates_per_height": 500,
    "bench_replan_events": [1, 2],                # corrective re-reaches per clip (uniform over the list)
    "bench_replan_period_s": 3.0,                 # hold after every corrective re-reach
    "bench_replan_first_hold_s": [0.0, 0.5],      # hold between the initial reach and the first corrective re-reach
    "bench_replan_offset_cm": [1.0, 6.0],         # |target shift| per re-reach (goal adjustment / residual error scale)
    "bench_replan_orient_jitter_deg": 3.0,        # +- yaw / pitch / roll jitter of the shifted target
    "bench_replan_midreach_share": 0.30,          # clips whose ORIGINAL reach also switches target mid-way (RetargetSpec)
    "bench_replan_midreach_offset_cm": [2.0, 8.0],
    "bench_replan_blend_s": 0.3,                  # handover crossfade duration for each corrective re-reach
}

PROFILES["hero_reach_far_v2"] = {
    **PROFILES["hero_reach_replan_v1"],
    "bench_replan_events": [0],                   # no re-reach; the other bench_replan_* knobs are inert
    "bench_replan_midreach_share": 0.0,
    "bench_retract_share": 0.5,
}
PROFILES["hero_reach_replan_v2"] = {
    **PROFILES["hero_reach_replan_v1"],
    "bench_retract_share": 0.5,
}


def rotation_angle_deg(R1: np.ndarray, R2: np.ndarray) -> float:
    """Geodesic angle (deg) between two rotation matrices."""
    c = (float(np.trace(np.asarray(R1, dtype=np.float64).T @ np.asarray(R2, dtype=np.float64))) - 1.0) / 2.0
    return math.degrees(math.acos(max(-1.0, min(1.0, c))))


def bench_family_layout(per_height: int, families: dict[str, dict[str, Any]]) -> list[tuple[str, str, int]]:
    """hero_bench_v2 candidate slots of one table height: ``(family, hand, family_index)`` per goal index, families blocked in
    dict order with largest-remainder counts from their ``share``, hands alternating right (even family index) / left (odd)."""
    names = list(families)
    w = np.array([max(0.0, float(families[k]["share"])) for k in names], dtype=np.float64)
    if w.sum() <= 0.0:
        raise ValueError("bench_orient_families shares must sum to a positive number")
    raw = w / w.sum() * per_height
    counts = np.floor(raw).astype(int)
    for i in np.argsort(-(raw - counts), kind="stable")[: per_height - int(counts.sum())]:   # ties -> earlier family (dict order)
        counts[i] += 1
    slots: list[tuple[str, str, int]] = []
    for name, n in zip(names, counts):
        slots += [(name, "right" if j % 2 == 0 else "left", j) for j in range(int(n))]
    return slots


def bench_family_quota(per_hand: int, families: dict[str, dict[str, Any]]) -> dict[str, int]:
    """Accepted goals per (height, family, hand) for ``per_hand`` goals per (height, hand): share x per_hand, largest remainder."""
    layout = bench_family_layout(per_hand, families)
    return {name: sum(1 for f, _, _ in layout if f == name) for name in families}


def sample_bench_orientation(rng: np.random.Generator, fam: dict[str, Any], u: Sequence[float], *, max_abs_roll_deg: float,
                             max_rot_deg: float, max_tries: int = BENCH_V2_MAX_ORIENT_TRIES, hand: str | None = None) -> dict[str, Any]:
    """One grasp orientation of family ``fam`` (degrees in, radians + bookkeeping out).

    The first draw uses the LHS coordinates ``u`` (yaw, pitch, roll in [0, 1)); a hard reject (|roll| above the roll cap or
    rotation from the family's canonical grasp above the rotation cap) re-draws uniformly from ``rng``.  The caps are the
    family's own ``max_abs_roll_deg`` / ``max_rot_from_canonical_deg`` when present, else ``max_abs_roll_deg`` / ``max_rot_deg``.
    ``roll_sign_random`` families pick the roll sign once per goal (``roll_sign``) and mirror the canonical roll with it;
    ``yaw_sign_random`` families do the same for yaw (``yaw_sign``); ``roll_mirror_by_hand`` families negate the roll band and the
    canonical roll for the right hand (``hand`` required), so a band written for the left hand names the same palm orientation
    for both hands (the Dex3 palm normal points toward the midline, so one heading-frame roll turns the two palms opposite ways).
    """
    y_lo, y_hi = (float(v) for v in fam["yaw_deg"])
    p_lo, p_hi = (float(v) for v in fam["pitch_deg"])
    r_lo, r_hi = (float(v) for v in fam["roll_deg"])
    roll_cap = float(fam.get("max_abs_roll_deg", max_abs_roll_deg))
    rot_cap = float(fam.get("max_rot_from_canonical_deg", max_rot_deg))
    sign = 1.0
    if fam.get("roll_sign_random", False):
        sign = 1.0 if rng.random() < 0.5 else -1.0
    mirrored = False
    if fam.get("roll_mirror_by_hand", False):
        if hand not in ("left", "right"):
            raise ValueError("roll_mirror_by_hand families need hand='left' | 'right'")
        mirrored = hand == "right"
        if mirrored:
            sign = -sign
    yaw_sign = 1.0
    if fam.get("yaw_sign_random", False):
        yaw_sign = 1.0 if rng.random() < 0.5 else -1.0
    cy, cp, cr = (float(v) for v in fam["canonical_ypr_deg"])
    canonical = (cy * yaw_sign, cp, cr * sign)
    R_c = palm_rotation(*(math.radians(v) for v in canonical))
    uu = [float(u[0]), float(u[1]), float(u[2])]
    for k in range(int(max_tries)):
        yaw_d = yaw_sign * (y_lo + (y_hi - y_lo) * uu[0])
        pitch_d = p_lo + (p_hi - p_lo) * uu[1]
        roll_d = sign * (r_lo + (r_hi - r_lo) * uu[2])
        R = palm_rotation(math.radians(yaw_d), math.radians(pitch_d), math.radians(roll_d))
        rot_c = rotation_angle_deg(R_c, R)
        if abs(roll_d) <= roll_cap + 1e-9 and rot_c <= rot_cap + 1e-9:
            return {"yaw": math.radians(yaw_d), "pitch": math.radians(pitch_d), "roll": math.radians(roll_d),
                    "yaw_deg": yaw_d, "pitch_deg": pitch_d, "roll_deg": roll_d, "roll_sign": sign, "yaw_sign": yaw_sign,
                    "roll_mirrored_by_hand": mirrored, "max_abs_roll_deg": roll_cap, "max_rot_from_canonical_deg": rot_cap,
                    "rot_from_canonical_deg": rot_c, "rot_from_side_grasp_deg": rotation_angle_deg(np.eye(3), R),
                    "canonical_ypr_deg": list(canonical), "orient_resamples": k}
        uu = [float(v) for v in rng.random(3)]
    raise RuntimeError(f"bench orientation sampler: {max_tries} consecutive hard rejects for family {fam} "
                       f"(|roll| <= {roll_cap} deg, rot from canonical <= {rot_cap} deg)")


def height_label(height_m: float) -> str:
    """0.50 -> ``h050``, 0.74 -> ``h074``, 0.88 -> ``h088`` (the evaluator groups clips by this prefix)."""
    return f"h{int(round(float(height_m) * 100.0)):03d}"


def stratum_for_surface(surface_z: float) -> str:
    """Name of the stratum whose surface range contains ``surface_z`` (upper bound inclusive; clamped at the ends)."""
    z = float(surface_z)
    for name, (lo, hi, _) in STRATA.items():
        if lo <= z <= hi:
            return name
    return STRATUM_NAMES[0] if z < STRATA[STRATUM_NAMES[0]][0] else STRATUM_NAMES[-1]


# ----------------------------------------------------------------------------------
# hero_bench_v1: stratified benchmark plan (strata, quotas, seeds, admitted levels) and the per-stratum sampler profiles
# ----------------------------------------------------------------------------------
BENCH_SCHEMA_V1: str = "hero_bench_v1"
BENCH_V1_SEED_BASE: int = 20261001                      # stratum seed = BENCH_V1_SEED_BASE + plan ordinal (unique per stratum)
BENCH_V1_FILL_MAX_BATCHES: int = 6                      # every generated stratum: top-up candidate batches k = 1..6, seed + 1000 k
BENCH_V1_FILL_SEED_STRIDE: int = 1000
BENCH_V1_LEGACY_SEEDS: frozenset[int] = frozenset({20260906, 20260907, 20260923})     # generator banks of the legacy profiles (hero_bench_v1 / hero_bench_v2 / far_v1)
BENCH_V1_LEGACY_PREFIXES: frozenset[str] = frozenset({"hero_reach_hero_bench_v1", "hero_reach_hero_bench_v2", "hero_reach_far_v1",
                                                      "hero_bench_v1", "hero_bench_v2"})
BENCH_V1_TIERS: tuple[str, str, str] = ("core", "extended", "stress")
BENCH_V1_TIER_TOTALS: dict[str, int] = {"core": 420, "extended": 618, "stress": 260}   # shipped clips per tier (1,298 in total)
BENCH_V1_HANDS: tuple[str, str] = ("right", "left")
BENCH_V1_HEIGHTS: list[float] = [0.50, 0.74, 0.88]      # the paper-protocol table tops
BENCH_V1_HIGH_BAND: dict[str, list[float]] = {"bench_yaw_deg": [-45.0, 45.0], "bench_pitch_deg": [-10.0, 15.0], "bench_roll_deg": [-10.0, 10.0]}
# paper_sub60: per paper-protocol table height the 10 right-hand + 10 left-hand clips with the smallest goal_index (60 clips) -- the
# source of the re-timed strata (slow_x2 / hold6 / fast_x0p75); their manifest rows carry pair_of = the paper twin's clip_id.
PAPER_SUB60_RULE: dict[str, Any] = {"layer": "paper", "per_height_per_hand": 10, "hands": list(BENCH_V1_HANDS),
                                 "heights": list(BENCH_V1_HEIGHTS), "order": "smallest goal_index first", "total": 60}
# Ranges deliberately outside hero_bench_v1 (recorded in the benchmark manifest with the measured feasibility that excluded them).
BENCH_V1_EXCLUDED_RANGES: list[dict[str, str]] = [
    {"range": "forward reach x 0.70-0.85 m", "measured": "0/9 feasible (support boundary)", "disposition": "needs a step; reserved for a later revision"},
    {"range": "shelf >= 1.22 m (palm >= 1.27 m)", "measured": "4/16 and 0/8; 0/200 in the geometric prefilter at 1.30 m", "disposition": "dropped"},
    {"range": "contralateral goals at the 0.88 m table", "measured": "1/4 and 1/8", "disposition": "cross_mid excludes 0.88 m"},
    {"range": "close goals at the 0.88 m table", "measured": "2/6", "disposition": "close excludes 0.88 m"},
    {"range": "|y| 0.45-0.55 m", "measured": "11/24 (2/8 at 0.50 m)", "disposition": "wide_lat capped at 0.45 m"},
    {"range": "kneeling / heel lift", "measured": "boundary tier throughout", "disposition": "not sampled"},
    {"range": "palm < 0.25 m above the floor", "measured": "5/16", "disposition": "floor_pick lower bound 0.25 m"},
]

# Orientation families of the generated strata.  Angles in degrees, ``palm_rotation(yaw, pitch, roll)`` convention (yaw 0 = fingers
# forward, palm toward the midline; pitch < 0 = fingers down; roll about the finger axis).  Optional per-family keys read by
# :func:`sample_bench_orientation`: ``max_abs_roll_deg`` / ``max_rot_from_canonical_deg`` (replace the profile-level caps),
# ``yaw_sign_random`` (yaw sign drawn per goal, canonical yaw mirrored with it) and ``roll_mirror_by_hand`` (roll band and
# canonical roll negated for the right hand: the Dex3 palm normal points toward the body midline, so one heading-frame roll turns
# the left palm down but the right palm up -- the bands below are written for the left hand).
BENCH_V1_ORIENT_FAMILIES: dict[str, dict[str, Any]] = {
    "top_down": {"yaw_deg": [-75.0, 75.0], "pitch_deg": [-85.0, -45.0], "roll_deg": [-20.0, 20.0], "roll_sign_random": False,
                 "canonical_ypr_deg": [0.0, -90.0, 0.0], "clearance_margin_m": 0.035,
                 "note": "fingers pointing down; the hero_bench_v2 band widened by 15 deg in yaw and 5 deg in pitch on each side"},
    "tilted": {"yaw_deg": [-55.0, 55.0], "pitch_deg": [-35.0, 15.0], "roll_deg": [25.0, 45.0], "roll_sign_random": True,
               "canonical_ypr_deg": [0.0, 0.0, 35.0], "clearance_margin_m": 0.02,
               "note": "side grasp rolled 25-45 deg about the finger axis; the hero_bench_v2 band widened by 10 deg in yaw and 5 deg in pitch on each side"},
    "fan_wide": {"yaw_deg": [50.0, 80.0], "yaw_sign_random": True, "pitch_deg": [-15.0, 10.0], "roll_deg": [-10.0, 10.0], "roll_sign_random": False,
                 "canonical_ypr_deg": [0.0, 0.0, 0.0], "clearance_margin_m": 0.02, "max_rot_from_canonical_deg": 90.0,
                 "note": "side grasp fanned 50-80 deg in yaw, sign random per goal"},
    "palm_down": {"yaw_deg": [-30.0, 30.0], "pitch_deg": [-10.0, 10.0], "roll_deg": [70.0, 100.0], "roll_sign_random": False, "roll_mirror_by_hand": True,
                  "canonical_ypr_deg": [0.0, 0.0, 90.0], "clearance_margin_m": 0.02, "max_abs_roll_deg": 100.0,
                  "note": "palm facing the surface (band written for the left hand, mirrored for the right)"},
    "palm_up": {"yaw_deg": [-30.0, 30.0], "pitch_deg": [-10.0, 10.0], "roll_deg": [-100.0, -70.0], "roll_sign_random": False, "roll_mirror_by_hand": True,
                "canonical_ypr_deg": [0.0, 0.0, -90.0], "clearance_margin_m": 0.02, "max_abs_roll_deg": 100.0,
                "note": "palm facing up (band written for the left hand, mirrored for the right)"},
}


def _bench_v1_profile(stratum: str, tier: str, n_candidates: int, *, families: Sequence[str] | None = None, **knobs: Any) -> dict[str, Any]:
    """Sampler profile of one generated stratum: the paper-protocol sampler (``PROFILES["hero_bench_v1"]``, single orientation band) or, with
    ``families``, the orientation-family machinery of ``PROFILES["hero_bench_v2"]`` over :data:`BENCH_V1_ORIENT_FAMILIES` (equal shares), plus the
    stratum's overrides."""
    if families is None:
        cfg = dict(PROFILES["hero_bench_v1"])
    else:
        cfg = dict(PROFILES["hero_bench_v2"])
        cfg["bench_orient_families"] = {name: {**BENCH_V1_ORIENT_FAMILIES[name], "share": 1.0} for name in families}
    cfg.update(knobs)
    n_heights = len(cfg["bench_heights"])
    if n_candidates <= 0 or n_candidates % n_heights:
        raise ValueError(f"{stratum}: n_candidates {n_candidates} must be a positive multiple of the number of heights ({n_heights})")
    cfg.update({"bench_schema": BENCH_SCHEMA_V1, "bench_stratum": stratum, "bench_tier": tier, "bench_plan_version": BENCH_SCHEMA_V1,
                "bench_candidates_per_height": n_candidates // n_heights})
    return cfg


def bench_v1_quota(per_cell: int, height_labels: Sequence[str], families: Sequence[str] | None = None) -> dict[tuple[str, ...], int]:
    """Accepted-clip quota per cell: ``(height_label, hand)`` -> count, or ``(height_label, hand, family)`` with ``families``."""
    cells: dict[tuple[str, ...], int] = {}
    for lab in height_labels:
        for hand in BENCH_V1_HANDS:
            if families is None:
                cells[(str(lab), hand)] = int(per_cell)
            else:
                for fam in families:
                    cells[(str(lab), hand, str(fam))] = int(per_cell)
    return cells


def _build_bench_v1_plan() -> dict[str, dict[str, Any]]:
    """Ordered stratum -> plan entry; registers ``PROFILES["hero_bench_v1_<stratum>"]`` for every generated stratum.

    Entry keys: ``tier`` (core | extended | stress), ``kind`` (generate | verbatim | retime | pool), ``profile`` (generate only),
    ``seed`` (BENCH_V1_SEED_BASE + ordinal), ``n_candidates``, ``quota`` (cell -> count, see :func:`bench_v1_quota`; pool strata are
    keyed by ``(source_stratum,)``), ``admitted_levels`` (builder acceptance levels, best first), ``fill_until_quota`` /
    ``fill_max_batches`` (every generated stratum: top-up batches with :func:`bench_v1_batch_seed` until every cell is full; a core
    cell still short after the last batch fails the build, an extended / stress stratum ships what it has and reports the shortfall),
    ``min_yield`` (extended 0.60 / stress 0.40 of strict + core over the candidates of the batches used; below it the stratum ships
    flagged provisional), ``fallback`` (what to try when a stress stratum misses its yield; noted in the report),
    ``clip_prefix`` (unique per stratum; verbatim files keep their source names), and per kind ``verbatim_from`` / ``retime`` / ``pool``.
    """
    strict, strict_core = ["strict"], ["strict", "core"]
    h3 = [height_label(h) for h in BENCH_V1_HEIGHTS]

    def gen(stratum: str, tier: str, n_candidates: int, quota: dict[tuple[str, ...], int], admitted: list[str], *,
            families: Sequence[str] | None = None, min_yield: float | None = None, fallback: str | None = None, **knobs: Any) -> dict[str, Any]:
        name = f"hero_bench_v1_{stratum}"
        PROFILES[name] = _bench_v1_profile(stratum, tier, n_candidates, families=families, **knobs)
        fill = True                                          # ruling: keep producing, never ship a gap (top-up batches for every generated stratum)
        return {"tier": tier, "kind": "generate", "profile": name, "n_candidates": int(n_candidates), "quota": quota, "admitted_levels": list(admitted),
                "fill_until_quota": fill, "fill_max_batches": BENCH_V1_FILL_MAX_BATCHES if fill else 0, "min_yield": min_yield, "fallback": fallback}

    def retime(tier: str, factor: float, hold_s: float, *, gate: dict[str, float] | None = None, shortfall_ok: bool = False) -> dict[str, Any]:
        return {"tier": tier, "kind": "retime", "profile": None, "n_candidates": 0, "quota": bench_v1_quota(10, h3), "admitted_levels": list(strict),
                "fill_until_quota": False, "fill_max_batches": 0, "min_yield": None, "fallback": None,
                "retime": {"factor": float(factor), "hold_s": float(hold_s), "source_subset": "paper_sub60", "gate": gate, "shortfall_ok": bool(shortfall_ok)}}

    pool_sources = ["far060", "floor_pick", "high_h100_h110", "far070_bow", "high_h115_120"]
    strata: list[tuple[str, dict[str, Any]]] = [
        # ---- core (420): the paper protocol and its nearest extensions; strict goals only, every cell filled ----
        ("paper", {"tier": "core", "kind": "verbatim", "profile": None, "n_candidates": 0, "quota": bench_v1_quota(30, h3), "admitted_levels": list(strict),
                      "fill_until_quota": False, "fill_max_batches": 0, "min_yield": None, "fallback": None,
                      "verbatim_from": {"build": "paper_protocol", "rule": "all 180", "file_names": "unchanged", "check": "sha256 per file"}}),
        ("far060", gen("far060", "core", 180, bench_v1_quota(10, h3), strict,
                       bench_x_range=[0.50, 0.60], bench_stand_lean_deg_max=60.0, bench_squat_lean_deg_max=65.0)),
        ("cross_mid", gen("cross_mid", "core", 96, bench_v1_quota(10, [height_label(h) for h in (0.50, 0.62, 0.74)]), strict,
                          bench_heights=[0.50, 0.62, 0.74], bench_x_range=[0.35, 0.50], bench_abs_y_range=[0.02, 0.20], bench_y_side="cross")),
        ("mid_h062_h080", gen("mid_h062_h080", "core", 84, bench_v1_quota(15, ["h062", "h080"]), strict, bench_heights=[0.62, 0.80])),
        ("orient_core", gen("orient_core", "core", 120, bench_v1_quota(5, h3, ["top_down", "tilted"]), strict, families=["top_down", "tilted"])),
        # ---- extended (618) ----
        ("low_h030_h040", gen("low_h030_h040", "extended", 160, bench_v1_quota(20, ["h030", "h040"]), strict_core, min_yield=0.60,
                              bench_heights=[0.30, 0.40], bench_x_range=[0.40, 0.55], bench_squat_drop_range=[0.24, 0.34], bench_squat_lean_deg_max=65.0,
                              bench_edge_gap_range=[0.22, 0.30])),
        ("floor_pick", gen("floor_pick", "extended", 150, bench_v1_quota(30, ["floor"]), strict_core, min_yield=0.60,
                           bench_heights=[0.0], bench_has_table=False, bench_x_range=[0.30, 0.45], bench_abs_y_range=[0.12, 0.32],
                           bench_z_above_range=[0.25, 0.38], bench_pitch_deg=[-20.0, 10.0], bench_squat_drop_range=[0.29, 0.36], bench_squat_lean_deg_max=70.0)),
        ("high_h100_h110", gen("high_h100_h110", "extended", 120, bench_v1_quota(15, ["h100", "h110"]), strict_core, min_yield=0.60,
                               bench_heights=[1.00, 1.10], bench_table_depth=0.30, bench_x_range=[0.26, 0.40], bench_abs_y_range=[0.15, 0.35],
                               bench_time_scale_range=[1.3, 1.3], **BENCH_V1_HIGH_BAND)),
        ("wide_lat", gen("wide_lat", "extended", 108, bench_v1_quota(10, h3), strict_core, min_yield=0.60,
                         bench_x_range=[0.30, 0.45], bench_abs_y_range=[0.35, 0.45])),
        ("orient_ext", gen("orient_ext", "extended", 135, bench_v1_quota(5, h3, ["palm_down", "palm_up", "fan_wide"]), strict_core, min_yield=0.60,
                           families=["palm_down", "palm_up", "fan_wide"])),
        ("hover_above", gen("hover_above", "extended", 72, bench_v1_quota(8, h3), strict, min_yield=0.60, bench_z_above_range=[0.15, 0.30])),
        ("slow_x2", retime("extended", 2.0, 3.0)),
        ("hold6", retime("extended", 1.0, 6.0)),
        ("retract", gen("retract", "extended", 84, bench_v1_quota(10, h3), strict_core, min_yield=0.60, bench_retract_share=1.0)),
        ("close", gen("close", "extended", 56, bench_v1_quota(10, ["h050", "h074"]), strict_core, min_yield=0.60,
                      bench_heights=[0.50, 0.74], bench_x_range=[0.31, 0.36], bench_abs_y_range=[0.05, 0.20],
                      bench_stand_lean_deg_max=40.0, bench_squat_lean_deg_max=40.0)),
        # Reserved for a later revision (not in this plan): off_axis (stance yaw / stagger knobs) and mid_same (|y| 0.00-0.10 m on the same side).
        # ---- stress (260) ----
        ("far070_bow", gen("far070_bow", "stress", 180, bench_v1_quota(10, h3), strict_core, min_yield=0.40, fallback="shrink bench_x_range to [0.60, 0.65]",
                           bench_x_range=[0.60, 0.70], bench_lean_pelvis_share=0.7, bench_stand_lean_deg_max=80.0, bench_squat_lean_deg_max=85.0,
                           bench_stand_drop_range=[0.00, 0.08])),
        ("low_h025", gen("low_h025", "stress", 100, bench_v1_quota(20, ["h025"]), strict_core, min_yield=0.40,
                         bench_heights=[0.25], bench_x_range=[0.40, 0.55], bench_squat_drop_range=[0.30, 0.36], bench_squat_lean_deg_max=65.0,
                         bench_edge_gap_range=[0.22, 0.30])),
        ("high_h115_120", gen("high_h115_120", "stress", 120, bench_v1_quota(10, ["h115", "h120"]), strict_core, min_yield=0.40, fallback="keep the 1.15 m shelf only",
                              bench_heights=[1.15, 1.20], bench_table_depth=0.30, bench_x_range=[0.20, 0.32], bench_abs_y_range=[0.12, 0.30],
                              bench_time_scale_range=[1.4, 1.4], **BENCH_V1_HIGH_BAND)),
        ("fast_x0p75", retime("stress", 0.75, 3.0, gate={"max_qdot_rad_s": 8.0, "max_qddot_rad_s2": 100.0}, shortfall_ok=True)),
        ("recov_pool", {"tier": "stress", "kind": "pool", "profile": None, "n_candidates": 0, "quota": {(s,): 12 for s in pool_sources},
                        "admitted_levels": ["recoverable"], "fill_until_quota": False, "fill_max_batches": 0, "min_yield": None, "fallback": None,
                        "pool": {"sources": list(pool_sources), "per_source": 12, "level": "recoverable", "order": "smallest goal_index first"}}),
    ]
    plan: dict[str, dict[str, Any]] = {}
    for ordinal, (name, entry) in enumerate(strata):
        plan[name] = {"stratum": name, "ordinal": ordinal, "seed": BENCH_V1_SEED_BASE + ordinal, "clip_prefix": f"hero_bench_v1_{name}", **entry}
    return plan


BENCH_V1_PLAN: dict[str, dict[str, Any]] = _build_bench_v1_plan()


def bench_v1_quota_total(stratum: str) -> int:
    """Shipped clips of one stratum (sum of its quota cells)."""
    return int(sum(BENCH_V1_PLAN[stratum]["quota"].values()))


def bench_v1_tier_totals() -> dict[str, int]:
    """Shipped clips per tier."""
    out = {tier: 0 for tier in BENCH_V1_TIERS}
    for name, entry in BENCH_V1_PLAN.items():
        out[entry["tier"]] += bench_v1_quota_total(name)
    return out


def bench_v1_batch_seed(stratum: str, batch: int = 0) -> int:
    """Seed of candidate batch ``batch`` of a generated stratum: 0 = the plan seed, 1..``fill_max_batches`` = the top-up batches."""
    entry = BENCH_V1_PLAN[stratum]
    if batch < 0 or batch > entry["fill_max_batches"]:
        raise ValueError(f"{stratum}: batch {batch} outside 0..{entry['fill_max_batches']}")
    return int(entry["seed"]) + BENCH_V1_FILL_SEED_STRIDE * int(batch)


def bench_v1_plan_json() -> dict[str, Any]:
    """JSON-serialisable copy of the plan (quota cells joined with ``/``)."""
    out: dict[str, Any] = {}
    for name, entry in BENCH_V1_PLAN.items():
        out[name] = {**entry, "quota": {"/".join(k): v for k, v in entry["quota"].items()}, "quota_total": bench_v1_quota_total(name)}
    return out


def _check_bench_v1_plan() -> None:
    if bench_v1_tier_totals() != BENCH_V1_TIER_TOTALS:
        raise ValueError(f"hero_bench_v1 plan totals {bench_v1_tier_totals()} != {BENCH_V1_TIER_TOTALS}")
    seeds = [e["seed"] for e in BENCH_V1_PLAN.values()]
    prefixes = [e["clip_prefix"] for e in BENCH_V1_PLAN.values()]
    if len(set(seeds)) != len(seeds) or set(seeds) & BENCH_V1_LEGACY_SEEDS:
        raise ValueError("hero_bench_v1 plan seeds must be unique and disjoint from the legacy bank seeds")
    if len(set(prefixes)) != len(prefixes) or set(prefixes) & BENCH_V1_LEGACY_PREFIXES:
        raise ValueError("hero_bench_v1 plan clip prefixes must be unique and disjoint from the legacy bank prefixes")
    if len(BENCH_V1_PLAN) >= BENCH_V1_FILL_SEED_STRIDE:
        raise ValueError("top-up batch seeds would collide with other strata")
    for name, entry in BENCH_V1_PLAN.items():
        if entry["kind"] == "generate" and entry["profile"] not in PROFILES:
            raise ValueError(f"{name}: profile {entry['profile']} missing")


_check_bench_v1_plan()


# ----------------------------------------------------------------------------------
# Small math helpers (wxyz quaternions)
# ----------------------------------------------------------------------------------
def smoothstep(x: float) -> float:
    x = min(1.0, max(0.0, float(x)))
    return x * x * (3.0 - 2.0 * x)


def rot_x(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[1, 0, 0], [0, c, -s], [0, s, c]], dtype=np.float64)


def rot_y(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]], dtype=np.float64)


def rot_z(a: float) -> np.ndarray:
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1]], dtype=np.float64)


def palm_rotation(yaw: float, pitch: float, roll: float) -> np.ndarray:
    """Palm target rotation in the heading frame (see module docstring)."""
    return rot_z(yaw) @ rot_y(-pitch) @ rot_x(roll)


def mat_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    """Rotation matrix -> unit quaternion (w, x, y, z)."""
    R = np.asarray(R, dtype=np.float64)
    t = np.trace(R)
    if t > 0.0:
        s = math.sqrt(t + 1.0) * 2.0
        w = 0.25 * s
        x = (R[2, 1] - R[1, 2]) / s
        y = (R[0, 2] - R[2, 0]) / s
        z = (R[1, 0] - R[0, 1]) / s
    elif R[0, 0] > R[1, 1] and R[0, 0] > R[2, 2]:
        s = math.sqrt(1.0 + R[0, 0] - R[1, 1] - R[2, 2]) * 2.0
        w = (R[2, 1] - R[1, 2]) / s
        x = 0.25 * s
        y = (R[0, 1] + R[1, 0]) / s
        z = (R[0, 2] + R[2, 0]) / s
    elif R[1, 1] > R[2, 2]:
        s = math.sqrt(1.0 + R[1, 1] - R[0, 0] - R[2, 2]) * 2.0
        w = (R[0, 2] - R[2, 0]) / s
        x = (R[0, 1] + R[1, 0]) / s
        y = 0.25 * s
        z = (R[1, 2] + R[2, 1]) / s
    else:
        s = math.sqrt(1.0 + R[2, 2] - R[0, 0] - R[1, 1]) * 2.0
        w = (R[1, 0] - R[0, 1]) / s
        x = (R[0, 2] + R[2, 0]) / s
        y = (R[1, 2] + R[2, 1]) / s
        z = 0.25 * s
    q = np.array([w, x, y, z], dtype=np.float64)
    if q[0] < 0.0:
        q = -q
    return q / np.linalg.norm(q)


def quat_wxyz_to_mat(q: np.ndarray) -> np.ndarray:
    w, x, y, z = (float(v) for v in np.asarray(q, dtype=np.float64) / np.linalg.norm(q))
    return np.array([
        [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
        [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
        [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
    ], dtype=np.float64)


def quat_angle_wxyz(q1: np.ndarray, q2: np.ndarray) -> float:
    """Geodesic angle (rad) between two wxyz quaternions."""
    d = abs(float(np.dot(np.asarray(q1, dtype=np.float64), np.asarray(q2, dtype=np.float64))))
    d = min(1.0, d / max(1e-12, float(np.linalg.norm(q1)) * float(np.linalg.norm(q2))))
    return 2.0 * math.acos(d)


def quat_slerp_wxyz(q0: np.ndarray, q1: np.ndarray, alpha: float) -> np.ndarray:
    q0 = np.asarray(q0, dtype=np.float64) / np.linalg.norm(q0)
    q1 = np.asarray(q1, dtype=np.float64) / np.linalg.norm(q1)
    d = float(np.dot(q0, q1))
    if d < 0.0:
        q1 = -q1
        d = -d
    if d > 0.9995:
        out = (1.0 - alpha) * q0 + alpha * q1
        return out / np.linalg.norm(out)
    th = math.acos(min(1.0, d))
    s = math.sin(th)
    return (math.sin((1.0 - alpha) * th) / s) * q0 + (math.sin(alpha * th) / s) * q1


# ----------------------------------------------------------------------------------
# Dex3 hand envelope (measured on the MuJoCo scene at q=0, all right-hand collision geoms expressed in the
# wrist_yaw frame): fingertips reach x=0.217 (palm site at x=0.0415 -> 0.175 m beyond the palm
# point), the thumb extends 0.115 m toward the body midline (+y for the right hand), thickness +-0.045.
# ----------------------------------------------------------------------------------
HAND_BBOX_PALM_FRAME: tuple[tuple[float, float], tuple[float, float], tuple[float, float]] = ((-0.06, 0.175), (-0.03, 0.12), (-0.045, 0.045))
HAND_SURFACE_CLEARANCE_MARGIN: float = 0.02
assert BENCH_V2_ORIENT_FAMILIES["fan_side"]["clearance_margin_m"] == BENCH_V2_ORIENT_FAMILIES["tilted"]["clearance_margin_m"] == HAND_SURFACE_CLEARANCE_MARGIN


def hand_bbox_corners(side: str) -> np.ndarray:
    """(8,3) corners of the hand envelope relative to the palm point (thumb side mirrored for the left hand)."""
    (x0, x1), (y0, y1), (z0, z1) = HAND_BBOX_PALM_FRAME
    sgn = 1.0 if side == "right" else -1.0
    return np.array([[x, sgn * y, z] for x in (x0, x1) for y in (y0, y1) for z in (z0, z1)], dtype=np.float64)


def hand_lowest_offset(R: np.ndarray, side: str) -> float:
    """Lowest hand-envelope point (m, <= 0) relative to the palm point for palm rotation ``R`` (heading frame)."""
    corners = hand_bbox_corners(side) @ np.asarray(R, dtype=np.float64).T
    return float(min(0.0, corners[:, 2].min()))


def ee_min_height_above_surface(R: np.ndarray, side: str) -> float:
    """Minimum palm-point height above a horizontal surface so that no part of the hand touches it."""
    return -hand_lowest_offset(R, side) + HAND_SURFACE_CLEARANCE_MARGIN


# ----------------------------------------------------------------------------------
# Latin hypercube sampling
# ----------------------------------------------------------------------------------
def stratified_lhs(count: int, dims: int, seed: int) -> np.ndarray:
    """Deterministic Latin-hypercube samples in [0,1)^dims (one stratum per row per dim)."""
    if count < 1 or dims < 1:
        raise ValueError("stratified_lhs needs positive count and dims")
    rng = np.random.default_rng(seed)
    out = np.empty((count, dims), dtype=np.float64)
    for d in range(dims):
        perm = rng.permutation(count)
        out[:, d] = (perm.astype(np.float64) + rng.random(count)) / float(count)
    return out


def allocate_categories(count: int, shares: dict[Any, float], rng: np.random.Generator) -> list[Any]:
    """Proportional (largest-remainder) allocation of ``count`` items to categories, shuffled."""
    keys = list(shares)
    w = np.array([max(0.0, float(shares[k])) for k in keys], dtype=np.float64)
    if w.sum() <= 0.0:
        raise ValueError("category shares must sum to a positive number")
    w = w / w.sum()
    raw = w * count
    base = np.floor(raw).astype(int)
    rem = count - int(base.sum())
    order = np.argsort(-(raw - base))
    for i in order[:rem]:
        base[i] += 1
    out: list[Any] = []
    for k, n in zip(keys, base):
        out.extend([k] * int(n))
    rng.shuffle(out)
    return out


def parse_strata_weights(text: str | None) -> dict[str, float]:
    """Parse ``"floor=0.15,very_low=0.2,..."``; unknown names raise."""
    if not text:
        return dict(DEFAULT_STRATA_WEIGHTS)
    out: dict[str, float] = {}
    for item in text.split(","):
        item = item.strip()
        if not item:
            continue
        name, _, value = item.partition("=")
        name = name.strip()
        if name not in STRATA:
            raise ValueError(f"unknown stratum {name!r}; known: {STRATUM_NAMES}")
        out[name] = float(value)
    return out


def ankle_pitch_for_flat_foot(hip_pitch: float, knee: float, pelvis_pitch: float) -> float:
    """Closed-chain ankle pitch that keeps the sole flat on the floor."""
    return -(float(pelvis_pitch) + float(hip_pitch) + float(knee))


def flat_foot_feasible(hip_pitch: float, knee: float, pelvis_pitch: float, *, margin: float = 0.0) -> bool:
    """Flat-foot feasibility: ankle_pitch = -(pelvis_pitch + hip + knee) in [-0.873, 0.524] and
    knee - |hip_pitch| + pelvis_pitch <= 0.87 rad (plus knee within its range)."""
    ank = ankle_pitch_for_flat_foot(hip_pitch, knee, pelvis_pitch)
    lo, hi = ANKLE_PITCH_RANGE
    if not (lo + margin <= ank <= hi - margin):
        return False
    if float(knee) - abs(float(hip_pitch)) + float(pelvis_pitch) > FLAT_FOOT_CLOSURE_MAX + 1e-9:
        return False
    return KNEE_RANGE[0] + margin <= float(knee) <= KNEE_RANGE[1] - margin


def thigh_torso_clear(hip_pitch: float, waist_pitch: float) -> bool:
    """Deep-squat self-collision rule: the thigh must not fold into the torso."""
    return abs(float(hip_pitch)) + max(0.0, float(waist_pitch)) <= THIGH_TORSO_MAX_FOLD + 1e-9


def sagittal_sole_offset(hip_pitch, knee, ankle_pitch, pelvis_pitch, geom: LegGeometry = LEG):
    """Sole-centre (ankle projection) position relative to the pelvis origin in the sagittal plane.

    Returns (x_forward, z_up) — vectorised over numpy inputs.  Positive hip_pitch rotates
    the thigh backwards (MuJoCo +y axis), so hip flexion is negative; knee flexion positive.
    """
    a_pel = np.asarray(pelvis_pitch, dtype=np.float64)
    a_th = a_pel + np.asarray(hip_pitch, dtype=np.float64)
    a_sh = a_th + np.asarray(knee, dtype=np.float64)
    a_ft = a_sh + np.asarray(ankle_pitch, dtype=np.float64)
    # Ry(a) applied to (0,0,-L) = (-L sin a, 0, -L cos a)
    x = -geom.hip_drop * np.sin(a_pel) - geom.thigh * np.sin(a_th) - geom.shank * np.sin(a_sh) - geom.ankle_to_sole * np.sin(a_ft)
    z = -geom.hip_drop * np.cos(a_pel) - geom.thigh * np.cos(a_th) - geom.shank * np.cos(a_sh) - geom.ankle_to_sole * np.cos(a_ft)
    return x, z


def pelvis_height_flat_foot(hip_pitch, knee, pelvis_pitch, geom: LegGeometry = LEG):
    """Pelvis height above the floor for a flat-foot closed chain (vectorised)."""
    ank = -(np.asarray(pelvis_pitch, dtype=np.float64) + np.asarray(hip_pitch, dtype=np.float64) + np.asarray(knee, dtype=np.float64))
    _, z = sagittal_sole_offset(hip_pitch, knee, ank, pelvis_pitch, geom)
    return -z


@dataclasses.dataclass(frozen=True)
class LegPrior:
    hip_pitch: float
    knee: float
    ankle_pitch: float
    pelvis_pitch: float
    height: float
    pelvis_x_from_ankle: float   # pelvis origin forward of the ankle (m); negative = pelvis behind the ankle
    feasible_flat: bool

    def as_dict(self) -> dict[str, Any]:
        return dataclasses.asdict(self)


def solve_leg_prior(height: float, pelvis_pitch: float, *, foot_pitch: float = 0.0, pelvis_x_pref: float = -0.02,
                    knee_max: float = KNEE_SOFT_MAX, geom: LegGeometry = LEG) -> LegPrior:
    """Closed-form sagittal (hip, knee, ankle) triple giving pelvis ``height`` with the sole (or, for a
    heel-lift ``foot_pitch`` > 0, the toe pivot) on the floor.

    The foot pitch is the nose-down rotation of the foot (0 = flat).  The pelvis-x offset from the
    ankle is scanned over a 1-D grid; among the candidates that satisfy the ankle range (with a
    0.01 rad margin), knee <= ``knee_max`` and the ankle-closure constraint, the one closest to
    ``pelvis_x_pref`` (slightly behind the ankle keeps the CoM over the sole when the torso leans
    forward) is returned.  If no candidate is feasible, ``feasible_flat`` is False and the candidate
    with the smallest ankle-range violation is returned for reference.
    """
    pp = float(pelvis_pitch)
    fp = float(foot_pitch)
    # pivot (sole centre or toe) below the ankle pitch axis after rotating the foot nose-down by fp
    pivot_drop = geom.ankle_to_sole * math.cos(fp) + (geom.toe_x * math.sin(fp) if fp > 0.0 else 0.0)
    d = np.arange(-0.30, 0.15 + 1e-9, 0.0025)      # pelvis x relative to the ankle axis
    ankle_x = -d                                     # ankle relative to pelvis
    ankle_z = -(float(height) - pivot_drop)
    hip_x, hip_z = -geom.hip_drop * math.sin(pp), -geom.hip_drop * math.cos(pp)
    dx, dz = ankle_x - hip_x, ankle_z - hip_z
    L = np.hypot(dx, dz)
    reach_ok = (L <= geom.thigh + geom.shank - 1e-6) & (L >= abs(geom.thigh - geom.shank) + 1e-6)
    L = np.clip(L, abs(geom.thigh - geom.shank) + 1e-6, geom.thigh + geom.shank - 1e-6)
    cos_k = (L ** 2 - geom.thigh ** 2 - geom.shank ** 2) / (2.0 * geom.thigh * geom.shank)
    knee = np.arccos(np.clip(cos_k, -1.0, 1.0))
    phi = np.arctan2(-dx, -dz)                      # angle of the hip->ankle vector (Ry convention)
    beta = np.arctan2(geom.shank * np.sin(knee), geom.thigh + geom.shank * np.cos(knee))
    a_th = phi - beta
    hip = a_th - pp
    ank = fp - (pp + hip + knee)
    lo, hi = ANKLE_PITCH_RANGE
    viol = np.maximum(0.0, lo + 0.01 - ank) + np.maximum(0.0, ank - (hi - 0.01))
    ok = reach_ok & (viol <= 0.0) & (knee <= knee_max) & (knee >= KNEE_RANGE[0] + 0.01)
    ok &= (hip >= HIP_PITCH_RANGE[0] + 0.01) & (hip <= HIP_PITCH_RANGE[1] - 0.01)
    if fp <= 0.0:
        ok &= (knee - np.abs(hip) + pp) <= FLAT_FOOT_CLOSURE_MAX + 1e-9
    if np.any(ok):
        score = np.where(ok, (d - pelvis_x_pref) ** 2 + 1e-4 * knee, np.inf)
        i = int(np.argmin(score))
        return LegPrior(float(hip[i]), float(knee[i]), float(ank[i]), pp, float(height), float(d[i]), True)
    score = np.where(reach_ok, viol + 0.05 * np.abs(d - pelvis_x_pref), np.inf)
    if not np.isfinite(score).any():
        i = int(np.argmin(np.abs(d - pelvis_x_pref)))
    else:
        i = int(np.argmin(score))
    return LegPrior(float(hip[i]), float(knee[i]), float(ank[i]), pp, float(height), float(d[i]), False)


def kneel_leg_prior(side: str) -> dict[str, float]:
    """Joint prior for a single-knee kneel with the toes tucked."""
    return {"hip_pitch": -0.35, "knee": 1.95, "ankle_pitch": 0.45}


# ----------------------------------------------------------------------------------
# Spec dataclasses
# ----------------------------------------------------------------------------------
@dataclasses.dataclass
class HandTarget:
    hand: str                       # "left" | "right"
    pos: tuple[float, float, float]  # heading frame, metres
    yaw: float                      # rad, heading frame (see module docstring)
    pitch: float                    # rad, finger elevation (negative = down)
    roll: float                     # rad
    canonical_grasp: bool
    yaw_clipped: bool
    approach: str                   # "descend" | "ingress"
    hover: float                    # m above target for the descend approach
    ingress: float                  # m pre-target offset for the ingress approach
    lift: float                     # m initial lift clearance

    def rotation(self) -> np.ndarray:
        return palm_rotation(self.yaw, self.pitch, self.roll)

    def quat_wxyz(self) -> np.ndarray:
        return mat_to_quat_wxyz(self.rotation())


@dataclasses.dataclass
class BaseStrategy:
    family: str                     # "stand" | "squat" | "stand_bend" | "bow" | "kneel"
    drop_mode: str                  # "stand" | "shallow" | "medium" | "deep"
    pelvis_drop: float              # m below STANDING_ROOT_Z
    pelvis_pitch: float             # rad (forward lean of the pelvis)
    waist_pitch: float              # rad (forward bend of the waist joint)
    waist_pitch_fraction: float     # share of the torso lean taken by the waist
    waist_yaw: float                # rad prior (behind-body layer)
    pelvis_shift: tuple[float, float]  # m, heading frame
    pelvis_yaw_delta: float         # rad, pelvis yaw change during the reach
    foot_mode: str                  # "flat" | "heel_lift" | "kneel"
    heel_lift_pitch: float          # rad foot pitch target (nose down) for heel_lift
    kneel_side: str                 # "left" | "right" | ""
    leg_prior: dict[str, float] | None  # hip_pitch/knee/ankle_pitch for the flat-foot family
    base_timing: str                # "sync" | "lead"
    forced_variant: bool

    @property
    def pelvis_height(self) -> float:
        return STANDING_ROOT_Z - self.pelvis_drop


@dataclasses.dataclass
class RetargetSpec:
    phase: float                    # reach phase in [0.3, 0.7] at which the target switches
    offset: tuple[float, float, float]  # m, heading frame, |offset| in [0.10, 0.40]


@dataclasses.dataclass
class SegmentSpec:
    index: int
    stratum: str
    surface_z: float
    layer: str                      # "front" | "behind"
    has_table: bool
    table_depth: float              # m (x extent) of the slab under tabletop targets
    table_edge_gap: float           # m from the table front edge to the target
    targets: list[HandTarget]
    base: BaseStrategy
    hold_s: float
    retarget: RetargetSpec | None
    retract: bool
    time_scale: float               # U(0.75, 1.3) multiplier on T_reach
    reach_filtered: bool = True     # False: no (target, base) pair passed the reachability filter (kept for coverage labels)
    sweep_clearance_extra: float = 0.0   # m added to the table/floor clearance of the palm *approach sweep* (goal unchanged; hero_bench_v1)
    replan: bool = False            # Corrective re-reach along a straight palm path, starting from rest.
    blend_s: float = 0.0            # Crossfade duration from the preceding solved pose; 0 gives a hard switch.


@dataclasses.dataclass
class StanceSpec:
    half_width: float               # m, feet at y = +-half_width (heading frame)
    stagger: float                  # m, left foot x minus right foot x
    yaw0: float                     # rad, root yaw residual relative to the heading frame
    left_x: float
    right_x: float


@dataclasses.dataclass
class ClipSpec:
    clip_id: str
    seed: int
    profile: str
    hands_mode: str                 # "left" | "right" | "bimanual"
    active_hands: tuple[bool, bool]  # (left, right)
    stance: StanceSpec
    segments: list[SegmentSpec]
    bench: dict[str, Any] | None = None   # hero_bench_v1 goal metadata (copied into the clip labels), None for the mixture profiles

    def to_json(self) -> str:
        return json.dumps(dataclasses.asdict(self), sort_keys=True)

    @property
    def hold_total_s(self) -> float:
        return float(sum(s.hold_s for s in self.segments))


def spec_from_dict(d: dict[str, Any]) -> ClipSpec:
    segs = []
    for s in d["segments"]:
        s = dict(s)
        s["targets"] = [HandTarget(**{**t, "pos": tuple(t["pos"])}) for t in s["targets"]]
        b = dict(s["base"]); b["pelvis_shift"] = tuple(b["pelvis_shift"])
        s["base"] = BaseStrategy(**b)
        s["retarget"] = None if s["retarget"] is None else RetargetSpec(phase=s["retarget"]["phase"], offset=tuple(s["retarget"]["offset"]))
        segs.append(SegmentSpec(**s))
    return ClipSpec(clip_id=d["clip_id"], seed=int(d["seed"]), profile=d["profile"], hands_mode=d["hands_mode"],
                    active_hands=tuple(bool(v) for v in d["active_hands"]), stance=StanceSpec(**d["stance"]), segments=segs,
                    bench=d.get("bench"))


# ----------------------------------------------------------------------------------
# Geometric pre-checks (cheap reachability filter before the QP solver)
# ----------------------------------------------------------------------------------
SHOULDER_OFFSET_PELVIS = np.array([0.0, 0.1002, 0.2918])   # pelvis frame, left shoulder pitch link
WAIST_TO_TORSO_Z = 0.054
# G1 arm (MuJoCo FK): shoulder_pitch -> elbow 0.193, elbow -> wrist_yaw 0.184, palm offset 0.0415 => 0.419 m fully
# extended; 0.38 (91 %) leaves room for the wrist to realise pitched palm orientations.
ARM_REACH_MAX = 0.38
ARM_REACH_MIN = 0.12


def shoulder_position_for_base(base: BaseStrategy, hand: str) -> np.ndarray:
    """Approximate shoulder position (heading frame) after applying the base strategy."""
    side = 1.0 if hand == "left" else -1.0
    pelvis = np.array([base.pelvis_shift[0], base.pelvis_shift[1], base.pelvis_height])
    Rp = rot_z(base.pelvis_yaw_delta) @ rot_y(base.pelvis_pitch)
    waist_origin = pelvis + Rp @ np.array([0.0, 0.0, WAIST_TO_TORSO_Z])
    Rt = Rp @ rot_z(base.waist_yaw) @ rot_y(base.waist_pitch)
    shoulder_local = SHOULDER_OFFSET_PELVIS * np.array([1.0, side, 1.0]) - np.array([0.0, 0.0, WAIST_TO_TORSO_Z])
    return waist_origin + Rt @ shoulder_local


BODY_EXCLUSION_RADIUS: float = 0.24   # palm point must stay this far (xy) from the torso / thigh axis
TABLE_MIN_EDGE_X: dict[str, float] = {"high_surface": 0.25, "low_surface": 0.35}   # table front edge ahead of the pelvis


LEG_CLEARANCE_THIGH: float = 0.15   # palm point to the hip->knee segment (thigh radius 0.07 + hand 0.05 + margin)
LEG_CLEARANCE_SHANK: float = 0.14


def _segment_distance(p: np.ndarray, a: np.ndarray, b: np.ndarray) -> float:
    ab = b - a
    t = float(np.clip(np.dot(p - a, ab) / max(1e-12, np.dot(ab, ab)), 0.0, 1.0))
    return float(np.linalg.norm(p - (a + t * ab)))


def leg_segments_for_base(base: BaseStrategy) -> list[tuple[np.ndarray, np.ndarray]]:
    """(hip, knee) and (knee, ankle) segments per leg (heading frame) from the base's leg prior."""
    out = []
    pelvis = np.array([base.pelvis_shift[0], base.pelvis_shift[1], base.pelvis_height])
    Rp = rot_z(base.pelvis_yaw_delta) @ rot_y(base.pelvis_pitch)
    for side, sgn in (("left", 1.0), ("right", -1.0)):
        if base.foot_mode == "kneel" and base.kneel_side == side:
            prior = kneel_leg_prior(side)
        elif base.leg_prior is not None:
            prior = base.leg_prior
        else:
            lp = solve_leg_prior(base.pelvis_height, base.pelvis_pitch)
            prior = {"hip_pitch": lp.hip_pitch, "knee": lp.knee, "ankle_pitch": lp.ankle_pitch}
        hip = pelvis + Rp @ np.array([0.0, sgn * LEG.hip_lateral, -LEG.hip_drop])
        R_th = Rp @ rot_y(prior["hip_pitch"])
        knee = hip + R_th @ np.array([0.0, 0.0, -LEG.thigh])
        R_sh = R_th @ rot_y(prior["knee"])
        ankle = knee + R_sh @ np.array([0.0, 0.0, -LEG.shank])
        out.append((hip, knee))
        out.append((knee, ankle))
    return out


def target_clear_of_legs(base: BaseStrategy, target: HandTarget) -> bool:
    """Palm targets on top of / inside the thighs and shanks are unreachable (the hand + leg envelopes collide)."""
    p = np.asarray(target.pos, dtype=np.float64)
    segs = leg_segments_for_base(base)
    for i, (a, b) in enumerate(segs):
        if _segment_distance(p, a, b) < (LEG_CLEARANCE_THIGH if i % 2 == 0 else LEG_CLEARANCE_SHANK):
            return False
    return True


def target_clear_of_body(base: BaseStrategy, target: HandTarget) -> bool:
    """Reject palm targets inside the body column (torso between pelvis and shoulders, thighs below the pelvis)."""
    p = np.asarray(target.pos, dtype=np.float64)
    pelvis = np.array([base.pelvis_shift[0], base.pelvis_shift[1], base.pelvis_height])
    sh = 0.5 * (shoulder_position_for_base(base, "left") + shoulder_position_for_base(base, "right"))
    if p[2] > sh[2] + 0.12 or p[2] < pelvis[2] - 0.55:
        return True
    if p[2] >= pelvis[2]:
        a = float(np.clip((p[2] - pelvis[2]) / max(1e-6, sh[2] - pelvis[2]), 0.0, 1.0))
        axis = pelvis + a * (sh - pelvis)
    else:
        axis = pelvis  # thighs: roughly below the pelvis (front-leaning thighs handled by the QP collision limit)
    return float(np.hypot(p[0] - axis[0], p[1] - axis[1])) >= BODY_EXCLUSION_RADIUS


def target_reachable(base: BaseStrategy, target: HandTarget) -> bool:
    d = float(np.linalg.norm(np.asarray(target.pos) - shoulder_position_for_base(base, target.hand)))
    return ARM_REACH_MIN <= d <= ARM_REACH_MAX and target_clear_of_body(base, target) and target_clear_of_legs(base, target)


def table_allowed(layer: str, stratum: str, surface_z: float, targets: Sequence[HandTarget], edge_gap: float) -> bool:
    """A slab is only placed in front of the robot, with its front edge clear of the pelvis / knees."""
    if stratum == "floor" or layer != "front":
        return False
    edge_x = min(t.pos[0] for t in targets) - edge_gap
    return edge_x >= (TABLE_MIN_EDGE_X["high_surface"] if surface_z >= 0.60 else TABLE_MIN_EDGE_X["low_surface"])


# ----------------------------------------------------------------------------------
# Sampler
# ----------------------------------------------------------------------------------
def _drop_mode(drop: float) -> str:
    if drop < 0.05:
        return "stand"
    if drop < 0.12:
        return "shallow"
    if drop < 0.20:
        return "medium"
    if drop < 0.32:
        return "deep"
    return "extra_deep"


def _sample_hand_target(rng: np.random.Generator, hand: str, stratum: str, surface_z: float, layer: str,
                        canonical: bool, u: np.ndarray) -> HandTarget:
    """u: 6 LHS coordinates (x, y, z_above, yaw, pitch, roll) in [0,1)."""
    z = surface_z + EE_ABOVE_SURFACE_RANGE[0] + (EE_ABOVE_SURFACE_RANGE[1] - EE_ABOVE_SURFACE_RANGE[0]) * u[2]
    side = 1.0 if hand == "left" else -1.0
    if layer == "behind":
        x = X_BEHIND_RANGE[0] + (X_BEHIND_RANGE[1] - X_BEHIND_RANGE[0]) * u[0]
        ay = Y_BEHIND_ABS_RANGE[0] + (Y_BEHIND_ABS_RANGE[1] - Y_BEHIND_ABS_RANGE[0]) * u[1]
        y = ay * (side if rng.random() < 0.8 else -side)
    else:
        x = X_FRONT_RANGE[0] + (X_FRONT_RANGE[1] - X_FRONT_RANGE[0]) * u[0]
        y = Y_RANGE[0] + (Y_RANGE[1] - Y_RANGE[0]) * u[1]
    yaw_clipped = False
    if canonical:
        # Palm faces the object from the hand's side, with fingers rotated inward.
        yaw = -side * math.radians(25.0 + 35.0 * u[3])
        pitch = math.radians(-12.0 + 24.0 * u[4])
        roll = math.radians(-10.0 + 20.0 * u[5])
    else:
        yaw = math.radians(YAW_SAMPLE_DEG[0] + (YAW_SAMPLE_DEG[1] - YAW_SAMPLE_DEG[0]) * u[3])
        if abs(yaw) > math.radians(YAW_CLIP_DEG):
            yaw = math.copysign(math.radians(YAW_CLIP_DEG), yaw)
            yaw_clipped = True
        if rng.random() < 0.70:
            lo, hi = PITCH_PREFERRED_DEG[stratum]
        else:
            lo, hi = PITCH_FULL_RANGE_DEG
        pitch = math.radians(lo + (hi - lo) * u[4])
        roll = math.radians(ROLL_RANGE_DEG[0] + (ROLL_RANGE_DEG[1] - ROLL_RANGE_DEG[0]) * u[5])
    if layer == "behind":
        # a behind-body target is approached sideways at target height
        approach = "ingress"
    elif stratum in ("high", "overhead"):
        approach = "ingress"
    elif stratum == "standard":
        approach = "descend" if rng.random() < 0.5 else "ingress"
    else:
        approach = "descend"
    # the hand envelope must clear the surface for the sampled orientation (fingers pointing down need the
    # palm point >= 0.19 m above the surface; fingers level need ~0.065 m)
    z_min = surface_z + ee_min_height_above_surface(palm_rotation(yaw, pitch, roll), hand)
    z = max(z, z_min)
    return HandTarget(hand=hand, pos=(float(x), float(y), float(z)), yaw=float(yaw), pitch=float(pitch), roll=float(roll),
                      canonical_grasp=bool(canonical), yaw_clipped=yaw_clipped, approach=approach,
                      hover=float(rng.uniform(0.10, 0.25)), ingress=float(rng.uniform(0.10, 0.20)),
                      lift=float(rng.uniform(0.10, 0.25)))


DEEP_FAMILY_SHARE_LOW_STRATA: float = 0.35   # floor / very_low segments that sample the extra-low family
HEEL_LIFT_PELVIS_X_PREF: float = 0.10        # heel lift balances on the toes (+0.12 ahead of the ankle): pelvis goes forward
DEEP_FAMILY_HEIGHT_RANGE: tuple[float, float] = (0.28, 0.45)
KNEEL_PELVIS_HEIGHT_RANGE: tuple[float, float] = (0.47, 0.52)   # thigh mesh touches the floor below ~0.46


def _sample_base(rng: np.random.Generator, stratum: str, surface_z: float, layer: str, hands_mode: str,
                 forced_variant_share: float) -> BaseStrategy:
    """Joint base-height / torso strategy for one segment.

    Families: ``stand`` (drop 0-0.06), ``squat`` (nominal drop = clip(0.65*(0.70-surface_z), 0, 0.32) +-3 cm),
    ``stand_bend`` (stand + waist/pelvis lean), ``bow`` (hip-flexion bow, pelvis 0.52-0.64, torso 75-120 deg),
    and for the floor / very-low strata the extra-low ``deep_squat`` family (pelvis 0.28-0.45).  Every
    flat-foot family is checked with the analytic closure rule; a pelvis height < FLAT_FOOT_MIN_HEIGHT,
    an infeasible closure or knee > KNEE_SOFT_MAX routes the segment to a heel-lift (60 %) or a
    single-knee kneel (40 %) variant.
    """
    nominal = float(np.clip(0.65 * (0.70 - surface_z), 0.0, 0.32))
    family_u = rng.random()
    waist_yaw = 0.0
    low_strata = surface_z < 0.45
    if low_strata and family_u < DEEP_FAMILY_SHARE_LOW_STRATA:
        family = "deep_squat"
        height = float(rng.uniform(*DEEP_FAMILY_HEIGHT_RANGE))
        drop = float(STANDING_ROOT_Z - height)
        lean = math.radians(rng.uniform(15.0, 40.0))
        pelvis_pitch = lean
        waist_pitch = math.radians(rng.uniform(0.0, 30.0))
        f = pelvis_pitch / max(1e-6, pelvis_pitch + waist_pitch)
    elif surface_z < 0.65:
        u2 = (family_u - (DEEP_FAMILY_SHARE_LOW_STRATA if low_strata else 0.0)) / (1.0 - (DEEP_FAMILY_SHARE_LOW_STRATA if low_strata else 0.0))
        if u2 < 0.60:
            family = "squat"
            drop = float(np.clip(nominal + rng.uniform(-0.03, 0.03), 0.0, 0.32))
            lean = math.radians(rng.uniform(0.0, 25.0))
            f = rng.uniform(0.0, 1.0)
            pelvis_pitch = f * lean
            waist_pitch = min((1.0 - f) * lean, math.radians(30.0))
        elif u2 < 0.80:
            family = "stand_bend"
            drop = float(rng.uniform(0.0, 0.06))
            lean = math.radians(rng.uniform(15.0, 25.0))
            f = rng.uniform(0.0, 0.6)
            pelvis_pitch = f * lean
            waist_pitch = min((1.0 - f) * lean + math.radians(rng.uniform(0.0, 10.0)), math.radians(30.0))
        else:
            family = "bow"   # hip-flexion bow: pelvis 0.52-0.64 m, torso 60-95 deg from vertical
            drop = float(STANDING_ROOT_Z - rng.uniform(0.52, 0.64))
            torso = math.radians(rng.uniform(60.0, 95.0))
            pelvis_pitch = math.radians(rng.uniform(45.0, 70.0))
            waist_pitch = float(np.clip(torso - pelvis_pitch, 0.0, math.radians(30.0)))
            f = 1.0 - (waist_pitch / max(1e-6, pelvis_pitch + waist_pitch))
    else:
        if family_u < 0.85:
            family = "stand"
            drop = float(np.clip(nominal + rng.uniform(-0.03, 0.03), 0.0, 0.06))
        else:
            family = "squat"
            drop = float(rng.uniform(0.06, 0.12))
        lean = math.radians(rng.uniform(0.0, 25.0)) if stratum in ("standard",) else math.radians(rng.uniform(0.0, 12.0))
        f = rng.uniform(0.0, 1.0)
        pelvis_pitch = f * lean
        waist_pitch = min((1.0 - f) * lean, math.radians(30.0))
    if layer == "behind":
        waist_yaw = math.radians(rng.uniform(35.0, 80.0))  # sign fixed later from the target side
    shift = (float(rng.uniform(-0.05, 0.05)), float(rng.uniform(-0.05, 0.05)))
    yaw_delta = math.radians(rng.uniform(-15.0, 15.0))
    timing = "sync" if rng.random() < 0.5 else "lead"

    height = STANDING_ROOT_Z - drop
    prior = solve_leg_prior(height, pelvis_pitch)
    # thigh-torso clearance: first give up waist pitch, then raise the pelvis (deep families only)
    if prior.feasible_flat and not thigh_torso_clear(prior.hip_pitch, waist_pitch):
        waist_pitch = float(max(0.0, THIGH_TORSO_MAX_FOLD - abs(prior.hip_pitch)))
        f = pelvis_pitch / max(1e-6, pelvis_pitch + waist_pitch)
        while prior.feasible_flat and not thigh_torso_clear(prior.hip_pitch, waist_pitch) and height < 0.60:
            height += 0.01
            drop = float(STANDING_ROOT_Z - height)
            prior = solve_leg_prior(height, pelvis_pitch)
    foot_mode = "flat"
    heel = 0.0
    kneel_side = ""
    forced = False
    leg_prior: dict[str, float] | None = {"hip_pitch": prior.hip_pitch, "knee": prior.knee, "ankle_pitch": prior.ankle_pitch}
    needs_variant = (not prior.feasible_flat) or height < FLAT_FOOT_MIN_HEIGHT or prior.knee > KNEE_SOFT_MAX
    if not needs_variant and stratum in ("floor", "very_low") and rng.random() < forced_variant_share:
        needs_variant = True
        forced = True
    if needs_variant and family != "bow":
        if rng.random() < 0.6:
            foot_mode = "heel_lift"
            heel = float(rng.uniform(0.30, 0.45))
            # toe-pivot heel lift: the closure is relaxed by the foot pitch; knee may go to 2.75 rad
            prior2 = solve_leg_prior(height, pelvis_pitch, foot_pitch=heel, knee_max=2.75, pelvis_x_pref=HEEL_LIFT_PELVIS_X_PREF)
            if not (prior2.feasible_flat and thigh_torso_clear(prior2.hip_pitch, waist_pitch)):
                # even the heel lift cannot reach this height cleanly: raise the pelvis to the lowest feasible one
                for h_try in np.arange(height + 0.01, 0.60, 0.01):
                    prior2 = solve_leg_prior(float(h_try), pelvis_pitch, foot_pitch=heel, knee_max=2.75, pelvis_x_pref=HEEL_LIFT_PELVIS_X_PREF)
                    if prior2.feasible_flat and thigh_torso_clear(prior2.hip_pitch, waist_pitch):
                        height = float(h_try)
                        drop = float(STANDING_ROOT_Z - height)
                        break
            leg_prior = {"hip_pitch": prior2.hip_pitch, "knee": prior2.knee, "ankle_pitch": prior2.ankle_pitch}
        else:
            foot_mode = "kneel"
            family = "kneel"
            kneel_side = "right" if rng.random() < 0.5 else "left"
            drop = float(STANDING_ROOT_Z - rng.uniform(*KNEEL_PELVIS_HEIGHT_RANGE))
            pelvis_pitch = math.radians(rng.uniform(0.0, 20.0))
            waist_pitch = math.radians(rng.uniform(0.0, 25.0))
            f = pelvis_pitch / max(1e-6, pelvis_pitch + waist_pitch)
            leg_prior = None
    elif needs_variant and family == "bow":
        # a bow with the pelvis at 0.52-0.64 m is always flat-foot feasible; keep flat
        foot_mode = "flat"
    return BaseStrategy(family=family, drop_mode=_drop_mode(drop), pelvis_drop=drop, pelvis_pitch=float(pelvis_pitch),
                        waist_pitch=float(waist_pitch), waist_pitch_fraction=float(f), waist_yaw=float(waist_yaw),
                        pelvis_shift=shift, pelvis_yaw_delta=float(yaw_delta), foot_mode=foot_mode, heel_lift_pitch=heel,
                        kneel_side=kneel_side, leg_prior=leg_prior, base_timing=timing, forced_variant=forced)


def _sample_hold(rng: np.random.Generator, long_share: float) -> float:
    if rng.random() < long_share:
        return float(rng.uniform(2.0, 3.0))
    return float(rng.uniform(0.5, 2.0))


# ----------------------------------------------------------------------------------
# HERO benchmark sampler (profile hero_bench_v1)
# ----------------------------------------------------------------------------------
BENCH_TARGET_MIN_BEHIND_EDGE: float = 0.03   # the goal must lie above the slab: target x >= table edge + 3 cm (x is shifted, the edge is kept)
BENCH_LEAN_PELVIS_SHARE: float = 0.4         # torso lean split: 40 % pelvis pitch / 60 % waist pitch
BENCH_LEAN_STEPS: int = 8
BENCH_APPROACH: str = "ingress"
BENCH_HOVER_M: float = 0.15
BENCH_INGRESS_M: float = 0.12
BENCH_LIFT_M: float = 0.10


def _pair(cfg: dict[str, Any], key: str) -> tuple[float, float]:
    v = cfg[key]
    if isinstance(v, (int, float)):
        v = [float(v), float(v)]   # a scalar override pins the range
    if not isinstance(v, (list, tuple)) or len(v) != 2:
        raise ValueError(f"{key} must be a [lo, hi] pair, got {v!r}")
    lo, hi = float(v[0]), float(v[1])
    if hi < lo:
        raise ValueError(f"{key}: hi {hi} < lo {lo}")
    return lo, hi


def _bench_base(table_z: float, drop: float, target: HandTarget, cfg: dict[str, Any]) -> tuple[BaseStrategy, bool]:
    """Flat-foot stand / squat base for one benchmark goal.

    The torso lean (pelvis share ``bench_lean_pelvis_share``, default 40 % pelvis / 60 % waist) is ramped from 0 to the family's
    maximum in ``BENCH_LEAN_STEPS`` steps and the first lean that passes the geometric reach prefilter (:func:`target_reachable`)
    is kept; if none passes the maximum lean is kept and the second return value is False (the IK still runs; the selector
    decides on the audited result).
    """
    squat = table_z < float(cfg["bench_squat_below_m"])
    family = "squat" if squat else "stand"
    lean_max = math.radians(float(cfg["bench_squat_lean_deg_max"] if squat else cfg["bench_stand_lean_deg_max"]))
    pelvis_share = float(cfg.get("bench_lean_pelvis_share", BENCH_LEAN_PELVIS_SHARE))
    if not 0.0 < pelvis_share <= 1.0:
        raise ValueError(f"bench_lean_pelvis_share must be in (0, 1], got {pelvis_share}")
    height = STANDING_ROOT_Z - drop
    last: BaseStrategy | None = None
    for k in range(BENCH_LEAN_STEPS):
        lean = lean_max * k / (BENCH_LEAN_STEPS - 1)
        pelvis_pitch = pelvis_share * lean
        waist_pitch = min((1.0 - pelvis_share) * lean, WAIST_PITCH_LIMIT)
        prior = solve_leg_prior(height, pelvis_pitch)
        if not prior.feasible_flat or not thigh_torso_clear(prior.hip_pitch, waist_pitch):
            continue
        base = BaseStrategy(family=family, drop_mode=_drop_mode(drop), pelvis_drop=float(drop), pelvis_pitch=float(pelvis_pitch),
                            waist_pitch=float(waist_pitch), waist_pitch_fraction=float(1.0 - pelvis_share), waist_yaw=0.0,
                            pelvis_shift=(0.0, 0.0), pelvis_yaw_delta=0.0, foot_mode="flat", heel_lift_pitch=0.0, kneel_side="",
                            leg_prior={"hip_pitch": prior.hip_pitch, "knee": prior.knee, "ankle_pitch": prior.ankle_pitch},
                            base_timing="sync", forced_variant=False)
        last = base
        if target_reachable(base, target):
            return base, True
    if last is None:  # pragma: no cover - stand / squat drops in the profile ranges are always flat-foot feasible
        raise RuntimeError(f"no flat-foot base for table {table_z} drop {drop}")
    return last, False


def sample_bench_specs(n: int | None, seed: int, cfg: dict[str, Any], clip_prefix: str | None = None) -> list[ClipSpec]:
    """HERO reaching-benchmark candidates (profiles ``hero_bench_v1`` / ``hero_bench_v2``), deterministic in ``seed``.

    ``n`` candidates in total, ``n / len(bench_heights)`` per table height (``n`` None -> ``bench_candidates_per_height``
    per height).  Candidate ``i`` -> height index ``i // per_height``, goal index ``i % per_height``.  v1: the hand alternates
    right (even goal index) / left (odd), so every height gets an exact 50/50 split when ``per_height`` is even, one yaw /
    pitch / roll band.  v2 (``bench_orient_families`` set): goal indices are blocked by orientation family
    (:func:`bench_family_layout`, hands alternating inside each block), the orientation is drawn per family with the hard
    rejects of :func:`sample_bench_orientation`, the clearance margin is per family, and ``bench`` carries ``orient_family`` /
    ``yaw_deg`` / ``pitch_deg`` / ``roll_deg`` / ``rot_from_canonical_deg``.  Targets
    are given in the heading frame (pelvis xy at frame 0 = origin, x forward), which is the bank's world frame (the solved
    frame-0 pelvis sits within ~5 mm of the stance-centre origin, see ``PELVIS_X_BEHIND_STANCE``; the labels record
    ``pelvis_xy_frame0``).  ``ClipSpec.bench`` carries the goal metadata.

    Stratified-plan knobs (all optional; the defaults reproduce the ``hero_bench_v1`` / ``hero_bench_v2`` profile banks byte for byte): ``bench_has_table`` (False: floor reaches
    with ``bench_heights == [0.0]``, no slab, ``height_label`` "floor", the table fields of ``bench`` None), ``bench_y_side``
    ("same" | "cross" | "mixed": goal on the hand's own side, the opposite side, or alternating by goal pair; ``bench_abs_y_range``
    must be non-negative), ``bench_time_scale_range`` ([lo, hi] multiplier on the reach duration, drawn per goal; a pinned pair
    draws nothing), ``bench_lean_pelvis_share`` (pelvis share of the torso lean, see :func:`_bench_base`); profiles that set
    ``bench_stratum`` also label ``bench`` with ``stratum`` / ``tier`` / ``plan_version`` / ``has_table`` / ``y_side`` /
    ``time_scale`` and default the clip prefix to ``hero_reach_<schema>_<stratum>``.
    """
    raw_h = cfg["bench_heights"]
    heights = [float(h) for h in (raw_h if isinstance(raw_h, (list, tuple, np.ndarray)) else [raw_h])]
    if not heights:
        raise ValueError("bench_heights is empty")
    if n is None or int(n) <= 0:
        per_height = int(cfg["bench_candidates_per_height"])
    else:
        if int(n) % len(heights):
            raise ValueError(f"n={n} must be a multiple of the number of table heights ({len(heights)})")
        per_height = int(n) // len(heights)
    if per_height <= 0:
        raise ValueError("no candidates per height")
    x_lo, x_hi = _pair(cfg, "bench_x_range")
    ay_lo, ay_hi = _pair(cfg, "bench_abs_y_range")
    z_lo, z_hi = _pair(cfg, "bench_z_above_range")
    if ay_lo < 0.0:
        raise ValueError(f"bench_abs_y_range must be non-negative (got lo {ay_lo}); use bench_y_side='cross' for goals on the opposite side")
    has_table = bool(cfg.get("bench_has_table", True))
    if not has_table and any(abs(h) > 1e-9 for h in heights):
        raise ValueError(f"bench_has_table=False (floor reaches) requires bench_heights == [0.0], got {heights}")
    y_side = str(cfg.get("bench_y_side", "same"))
    if y_side not in ("same", "cross", "mixed"):
        raise ValueError(f"bench_y_side must be 'same', 'cross' or 'mixed', got {y_side!r}")
    ts_lo, ts_hi = _pair(cfg, "bench_time_scale_range") if cfg.get("bench_time_scale_range") is not None else (1.0, 1.0)
    if ts_lo <= 0.0:
        raise ValueError(f"bench_time_scale_range must be positive, got [{ts_lo}, {ts_hi}]")
    pelvis_share = float(cfg.get("bench_lean_pelvis_share", BENCH_LEAN_PELVIS_SHARE))
    stratum_tag = cfg.get("bench_stratum")
    families: dict[str, dict[str, Any]] | None = cfg.get("bench_orient_families")   # hero_bench_v2; None -> v1 single band
    if families:
        max_abs_roll = float(cfg.get("bench_max_abs_roll_deg", 45.0))
        max_rot = float(cfg.get("bench_max_rot_from_canonical_deg", 80.0))
        for name, fam in families.items():
            for key in ("yaw_deg", "pitch_deg", "roll_deg"):
                _pair(fam, key)
            if len(fam["canonical_ypr_deg"]) != 3:
                raise ValueError(f"family {name}: canonical_ypr_deg must have 3 entries")
    else:
        yaw_lo, yaw_hi = _pair(cfg, "bench_yaw_deg")
        pit_lo, pit_hi = _pair(cfg, "bench_pitch_deg")
        rol_lo, rol_hi = _pair(cfg, "bench_roll_deg")
    gap_lo, gap_hi = _pair(cfg, "bench_edge_gap_range")
    hw_lo, hw_hi = _pair(cfg, "bench_stance_half_width_range")
    edge_ref = str(cfg["bench_edge_ref"])
    if edge_ref not in ("toe", "pelvis"):
        raise ValueError(f"bench_edge_ref must be 'toe' or 'pelvis', got {edge_ref!r}")
    approach = str(cfg.get("bench_approach", BENCH_APPROACH))
    if approach not in ("ingress", "descend"):
        raise ValueError(f"bench_approach must be 'ingress' or 'descend', got {approach!r}")
    sweep_extra = float(cfg.get("bench_sweep_clearance_extra_m", 0.0))
    table_depth = float(cfg["bench_table_depth"])
    hold_s = float(cfg["bench_hold_s"])
    clearance_margin_default = float(cfg.get("bench_clearance_margin_m", HAND_SURFACE_CLEARANCE_MARGIN))
    schema = str(cfg.get("bench_schema", BENCH_SCHEMA))
    prefix = clip_prefix or (f"hero_reach_{schema}_{stratum_tag}" if stratum_tag else f"hero_reach_{schema}")
    rng = np.random.default_rng(seed)
    n_total = per_height * len(heights)
    retract_share = float(cfg.get("bench_retract_share", 0.0))
    if not 0.0 <= retract_share <= 1.0:
        raise ValueError(f"bench_retract_share must be in [0, 1], got {retract_share}")
    if retract_share > 0.0:
        # exact largest-remainder share over the whole bank, shuffled on its own stream: the geometry draws below never see it
        retract_flags = allocate_categories(n_total, {True: retract_share, False: 1.0 - retract_share}, np.random.default_rng([seed, BENCH_RETRACT_STREAM]))
    else:
        retract_flags = [False] * n_total
    replan_events_cfg = cfg.get("bench_replan_events")
    replan_on = bool(replan_events_cfg) and any(int(v) > 0 for v in (replan_events_cfg if isinstance(replan_events_cfg, (list, tuple, np.ndarray)) else [replan_events_cfg]))
    pelvis_x0 = 0.0                                        # heading-frame origin == bank world origin (PELVIS_X_BEHIND_STANCE note)
    toe_x0 = TOE_LOCAL[0]                                  # foot front in the world frame (ankles at x = 0, no stagger)
    specs: list[ClipSpec] = []
    for hi_, table_z in enumerate(heights):
        label = height_label(table_z) if has_table else "floor"
        stratum = stratum_for_surface(table_z)
        squat = table_z < float(cfg["bench_squat_below_m"])
        d_lo, d_hi = _pair(cfg, "bench_squat_drop_range" if squat else "bench_stand_drop_range")
        if families:
            # v2: one LHS per (height, family) so every family's angles / positions are spread on their own; hands alternate
            # inside the family block (bench_family_layout)
            slots = bench_family_layout(per_height, families)
            fam_lhs = {name: stratified_lhs(max(1, sum(1 for f, _, _ in slots if f == name)), 8, seed + 7919 + 101 * hi_ + fi)
                       for fi, name in enumerate(families)}
        else:
            slots = [(None, "right" if gi % 2 == 0 else "left", gi) for gi in range(per_height)]
            lhs = stratified_lhs(per_height, 8, seed + 7919 + hi_)   # x, |y|, z_above, yaw, pitch, roll, edge gap, pelvis drop
        for gi in range(per_height):
            i = hi_ * per_height + gi
            fam_name, hand, fam_index = slots[gi]
            u = fam_lhs[fam_name][fam_index] if families else lhs[gi]
            side = 1.0 if hand == "left" else -1.0
            cross = y_side == "cross" or (y_side == "mixed" and ((fam_index if families else gi) // 2) % 2 == 1)
            y_sign = -side if cross else side
            x_h = x_lo + (x_hi - x_lo) * u[0]
            y_h = y_sign * (ay_lo + (ay_hi - ay_lo) * u[1])
            z_raw = table_z + z_lo + (z_hi - z_lo) * u[2]
            orient: dict[str, Any] | None = None
            if families:
                fam = families[fam_name]
                orient = sample_bench_orientation(rng, fam, u[3:6], max_abs_roll_deg=max_abs_roll, max_rot_deg=max_rot, hand=hand)
                yaw, pitch, roll = orient["yaw"], orient["pitch"], orient["roll"]
                clearance_margin = float(fam.get("clearance_margin_m", clearance_margin_default))
            else:
                yaw = math.radians(yaw_lo + (yaw_hi - yaw_lo) * u[3])
                pitch = math.radians(pit_lo + (pit_hi - pit_lo) * u[4])
                roll = math.radians(rol_lo + (rol_hi - rol_lo) * u[5])
                clearance_margin = clearance_margin_default
            z_min = table_z - hand_lowest_offset(palm_rotation(yaw, pitch, roll), hand) + clearance_margin
            z = max(z_raw, z_min)
            pos_w = (float(x_h + pelvis_x0), float(y_h), float(z))
            target = HandTarget(hand=hand, pos=pos_w, yaw=float(yaw), pitch=float(pitch), roll=float(roll), canonical_grasp=True,
                                yaw_clipped=False, approach=approach, hover=BENCH_HOVER_M, ingress=BENCH_INGRESS_M, lift=BENCH_LIFT_M)
            gap = gap_lo + (gap_hi - gap_lo) * u[6]
            edge_ref_x = toe_x0 if edge_ref == "toe" else pelvis_x0
            edge_x = edge_ref_x + gap
            x_raw = pos_w[0]
            x_shifted = False
            if has_table and edge_x + BENCH_TARGET_MIN_BEHIND_EDGE > pos_w[0]:
                # keep the robot-table distance (it decides IK feasibility) and move the goal onto the slab
                pos_w = (float(edge_x + BENCH_TARGET_MIN_BEHIND_EDGE), pos_w[1], pos_w[2])
                target.pos = pos_w
                x_shifted = True
            drop = d_lo + (d_hi - d_lo) * u[7]
            base, reach_ok = _bench_base(table_z, drop, target, cfg)
            time_scale = float(rng.uniform(ts_lo, ts_hi)) if ts_hi > ts_lo else float(ts_lo)
            seg = SegmentSpec(index=0, stratum=stratum, surface_z=float(table_z), layer="front", has_table=has_table,
                              table_depth=table_depth if has_table else 0.0, table_edge_gap=float(pos_w[0] - edge_x) if has_table else 0.0,
                              targets=[target], base=base, hold_s=hold_s, retarget=None,
                              retract=False, time_scale=time_scale, reach_filtered=reach_ok, sweep_clearance_extra=sweep_extra)
            stance = StanceSpec(half_width=float(rng.uniform(hw_lo, hw_hi)), stagger=0.0, yaw0=0.0, left_x=0.0, right_x=0.0)
            segments = [seg]
            replan_meta = None
            if replan_on:
                segments, replan_meta = _bench_replan_segments(rng, seg, cfg, table_z=float(table_z), edge_x=float(edge_x), clearance_margin=clearance_margin)
            if retract_flags[i]:
                segments[-1].retract = True                # Retract to rest after the final hold.
            bench = {
                "schema": schema, "height_m": float(table_z), "height_label": label, "height_index": hi_, "goal_index": gi,
                "candidate_index": i, "hand": hand,
                "target_pos_w": list(pos_w),
                "target_x_raw": float(x_raw), "target_x_shifted": x_shifted,
                "target_z_raw": float(z_raw), "target_z_min_clearance": float(z_min), "clearance_raised": bool(z > z_raw + 1e-12),
                "clearance_margin_m": clearance_margin,
                "target_yaw_deg": math.degrees(yaw), "target_pitch_deg": math.degrees(pitch), "target_roll_deg": math.degrees(roll),
                "orientation_convention": "palm R = Rz(yaw) Ry(-pitch) Rx(roll) in the heading frame; yaw 0 = fingers forward, palm "
                                          "normal toward the body midline (canonical inward-facing side grasp); quaternions wxyz",
                "table_top_z": float(table_z), "table_edge_x": float(edge_x), "table_edge_gap_m": float(gap), "table_edge_ref": edge_ref,
                "table_edge_ref_x": float(edge_ref_x), "table_depth": table_depth,
                "base_family": base.family, "pelvis_drop": float(base.pelvis_drop), "pelvis_height_plan": float(base.pelvis_height),
                "pelvis_pitch_deg": math.degrees(base.pelvis_pitch), "waist_pitch_deg": math.degrees(base.waist_pitch),
                "reach_prefilter_ok": bool(reach_ok), "hold_s": hold_s, "time_scale": time_scale, "approach": approach,
                "sweep_clearance_extra_m": sweep_extra, "retract": bool(retract_flags[i]),
                "frame": "world == heading frame: stance centre at the origin, x forward, y left, z up; the solved pelvis xy at frame 0 "
                         "lies within ~5 mm of the origin (labels: bench.pelvis_xy_frame0)",
            }
            if orient is not None:
                # hero_bench_v2 orientation family bookkeeping (duplicates yaw/pitch/roll under the requested short names)
                bench.update({
                    "orient_family": fam_name, "family_index": int(fam_index),
                    "yaw_deg": orient["yaw_deg"], "pitch_deg": orient["pitch_deg"], "roll_deg": orient["roll_deg"], "roll_sign": orient["roll_sign"],
                    # exact sampled degrees (the v1 keys above hold the radians -> degrees round trip; keep both names identical)
                    "target_yaw_deg": orient["yaw_deg"], "target_pitch_deg": orient["pitch_deg"], "target_roll_deg": orient["roll_deg"],
                    "rot_from_canonical_deg": orient["rot_from_canonical_deg"], "rot_from_side_grasp_deg": orient["rot_from_side_grasp_deg"],
                    "canonical_ypr_deg": orient["canonical_ypr_deg"], "orient_resamples": int(orient["orient_resamples"]),
                    "orient_limits": {"max_abs_roll_deg": orient["max_abs_roll_deg"], "max_rot_from_canonical_deg": orient["max_rot_from_canonical_deg"]},
                })
            if stratum_tag is not None:
                # stratified-plan bookkeeping (only profiles that declare a stratum; the legacy profile banks are byte-identical without it)
                bench.update({
                    "stratum": str(stratum_tag), "tier": cfg.get("bench_tier"), "plan_version": str(cfg.get("bench_plan_version", BENCH_SCHEMA_V1)),
                    "has_table": has_table, "y_side": y_side, "cross_side": bool(cross), "time_scale_range": [ts_lo, ts_hi],
                    "lean_pelvis_share": pelvis_share,
                })
                if orient is not None:
                    bench.update({"yaw_sign": orient["yaw_sign"], "roll_mirrored_by_hand": bool(orient["roll_mirrored_by_hand"])})
            if not has_table:
                bench.update({"table": None, "table_top_z": None, "table_edge_x": None, "table_edge_gap_m": None, "table_edge_ref": None,
                              "table_edge_ref_x": None, "table_depth": None, "target_x_shifted": False})
            if replan_meta is not None:
                bench["replan"] = replan_meta
            specs.append(ClipSpec(clip_id=f"{prefix}_{i:06d}", seed=int(rng.integers(0, 2**31 - 1)), profile=schema, hands_mode=hand,
                                  active_hands=(hand == "left", hand == "right"), stance=stance, segments=segments, bench=bench))
    return specs


def _bench_replan_segments(rng: np.random.Generator, seg0: SegmentSpec, cfg: dict[str, Any], *, table_z: float, edge_x: float,
                           clearance_margin: float) -> tuple[list[SegmentSpec], dict[str, Any]]:
    """hero_reach_replan_v1: turn the single bench segment ``seg0`` into ``[seg0 (short first hold, optional mid-reach retarget),
    re-reach_1, ..., re-reach_n]`` with corrective re-reaches.  Every re-reach keeps ``seg0``'s base /
    table / stratum, targets ``previous target + delta`` (|delta| ~ U(bench_replan_offset_cm), random direction, z kept above the
    hand-envelope clearance of the jittered orientation, x kept on the slab), orientation jittered by +-``bench_replan_orient_jitter_deg``,
    ``hold_s = bench_replan_period_s`` and ``replan=True`` (the generator plans it as a straight re-reach with the replanner timing)."""
    events_cfg = cfg["bench_replan_events"]
    events = [int(v) for v in (events_cfg if isinstance(events_cfg, (list, tuple, np.ndarray)) else [events_cfg])]
    n_events = int(events[int(rng.integers(0, len(events)))])
    period = float(cfg["bench_replan_period_s"])
    fh_lo, fh_hi = _pair(cfg, "bench_replan_first_hold_s")
    off_lo, off_hi = _pair(cfg, "bench_replan_offset_cm")
    jitter = math.radians(float(cfg.get("bench_replan_orient_jitter_deg", 0.0)))
    mid_share = float(cfg.get("bench_replan_midreach_share", 0.0))
    blend_s = float(cfg.get("bench_replan_blend_s", 0.0))
    if blend_s < 0.0:
        raise ValueError("bench_replan_blend_s must be >= 0")
    mid_lo, mid_hi = _pair(cfg, "bench_replan_midreach_offset_cm") if mid_share > 0.0 else (0.0, 0.0)
    if n_events < 0 or period <= 0.0 or off_lo <= 0.0 or off_hi < off_lo:
        raise ValueError("bench_replan_events >= 0, bench_replan_period_s > 0 and 0 < bench_replan_offset_cm lo <= hi required")
    t0 = seg0.targets[0]
    hand = t0.hand
    first_hold = float(rng.uniform(fh_lo, fh_hi))
    seg0.hold_s = first_hold
    midreach = None
    if mid_share > 0.0 and rng.random() < mid_share:
        d = _random_unit(rng) * (mid_lo + (mid_hi - mid_lo) * rng.random()) / 100.0
        seg0.retarget = RetargetSpec(phase=float(rng.uniform(0.30, 0.70)), offset=(float(d[0]), float(d[1]), float(d[2])))
        midreach = {"phase": seg0.retarget.phase, "offset_cm": [float(v) * 100.0 for v in d]}
    segments = [seg0]
    prev_pos = np.asarray(t0.pos, dtype=np.float64) + (np.asarray(seg0.retarget.offset) if seg0.retarget is not None else 0.0)
    prev_ypr = np.array([t0.yaw, t0.pitch, t0.roll], dtype=np.float64)
    offsets_cm: list[list[float]] = []
    for k in range(1, n_events + 1):
        mag = (off_lo + (off_hi - off_lo) * rng.random()) / 100.0
        delta = _random_unit(rng) * mag
        ypr = prev_ypr + rng.uniform(-jitter, jitter, size=3) if jitter > 0.0 else prev_ypr.copy()
        pos = prev_pos + delta
        z_min = table_z - hand_lowest_offset(palm_rotation(float(ypr[0]), float(ypr[1]), float(ypr[2])), hand) + clearance_margin
        x_min = edge_x + BENCH_TARGET_MIN_BEHIND_EDGE
        for _ in range(2):
            pos[2] = max(pos[2], z_min)                                        # above the table clearance of the new orientation
            pos[0] = max(pos[0], x_min)                                        # on the slab
            dvec = pos - prev_pos
            length = float(np.linalg.norm(dvec))
            if length > mag + 1e-12:                                           # a clamp lengthened the shift: pull it back onto the sphere ...
                pos = prev_pos + dvec * (mag / length)                        # ... then re-clamp (residual excess <= the clamp itself)
        pos[2] = max(pos[2], z_min)
        pos[0] = max(pos[0], x_min)
        target = HandTarget(hand=hand, pos=(float(pos[0]), float(pos[1]), float(pos[2])), yaw=float(ypr[0]), pitch=float(ypr[1]), roll=float(ypr[2]),
                            canonical_grasp=t0.canonical_grasp, yaw_clipped=False, approach=t0.approach, hover=t0.hover, ingress=t0.ingress, lift=t0.lift)
        segments.append(SegmentSpec(index=k, stratum=seg0.stratum, surface_z=seg0.surface_z, layer=seg0.layer, has_table=seg0.has_table,
                                    table_depth=seg0.table_depth, table_edge_gap=float(pos[0] - edge_x), targets=[target], base=seg0.base,
                                    hold_s=period, retarget=None, retract=False, time_scale=1.0, reach_filtered=seg0.reach_filtered,
                                    sweep_clearance_extra=seg0.sweep_clearance_extra, replan=True, blend_s=blend_s))
        offsets_cm.append([float(v) * 100.0 for v in (pos - prev_pos)])
        prev_pos, prev_ypr = pos, ypr
    meta = {"events": n_events, "period_s": period, "first_hold_s": first_hold, "offsets_cm": offsets_cm,
            "orient_jitter_deg": math.degrees(jitter), "midreach": midreach, "blend_s": blend_s}
    return segments, meta


def _random_unit(rng: np.random.Generator) -> np.ndarray:
    v = rng.normal(size=3)
    n = float(np.linalg.norm(v))
    return v / n if n > 1e-9 else np.array([1.0, 0.0, 0.0])


def sample_clip_specs(n: int, seed: int, *, profile: str = "broad_v2", strata_weights: dict[str, float] | None = None,
                      clip_prefix: str | None = None, **overrides: Any) -> list[ClipSpec]:
    """Sample ``n`` clip specifications (deterministic in ``seed``).

    ``strata_weights`` overrides the profile's stratum mixture; ``overrides`` may set any
    profile knob (behind_share, retarget_share, retract_share, canonical_share,
    hold_long_share, forced_variant_share, segment_count_shares).  Profiles with
    ``sampler: bench`` (``hero_bench_v1`` / ``hero_bench_v2``) are routed to :func:`sample_bench_specs`.
    """
    if profile not in PROFILES:
        raise ValueError(f"unknown profile {profile!r}; known: {tuple(PROFILES)}")
    cfg = dict(PROFILES[profile])
    cfg.update({k: v for k, v in overrides.items() if v is not None})
    if cfg.get("sampler") == "bench":
        return sample_bench_specs(n, seed, cfg, clip_prefix)
    weights = dict(cfg["strata_weights"]) if strata_weights is None else dict(strata_weights)
    for k in weights:
        if k not in STRATA:
            raise ValueError(f"unknown stratum {k!r}")
    seg_shares = cfg.get("segment_count_shares", SEGMENT_COUNT_SHARES)
    prefix = clip_prefix or f"hero_reach_{profile}"
    rng = np.random.default_rng(seed)
    hands = allocate_categories(n, HAND_MODE_SHARES, rng)
    n_segments = allocate_categories(n, seg_shares, rng)
    total_segments = int(sum(n_segments))
    strata = allocate_categories(total_segments, weights, rng)
    lhs = stratified_lhs(total_segments, 14, seed + 7919)   # per-segment continuous coordinates
    behind_flags = allocate_categories(total_segments, {True: cfg["behind_share"], False: 1.0 - cfg["behind_share"]}, rng)
    retarget_flags = allocate_categories(total_segments, {True: cfg["retarget_share"], False: 1.0 - cfg["retarget_share"]}, rng)
    canonical_flags = allocate_categories(total_segments, {True: cfg["canonical_share"], False: 1.0 - cfg["canonical_share"]}, rng)
    retract_flags = allocate_categories(n, {True: cfg["retract_share"], False: 1.0 - cfg["retract_share"]}, rng)
    specs: list[ClipSpec] = []
    seg_cursor = 0
    for i in range(n):
        hands_mode = hands[i]
        active = (hands_mode in ("left", "bimanual"), hands_mode in ("right", "bimanual"))
        clip_seed = int(rng.integers(0, 2**31 - 1))
        segments: list[SegmentSpec] = []
        kneel_stance: str | None = None
        for k in range(int(n_segments[i])):
            u = lhs[seg_cursor]
            stratum = strata[seg_cursor]
            layer = "behind" if behind_flags[seg_cursor] else "front"
            canonical = bool(canonical_flags[seg_cursor]) and layer == "front"
            do_retarget = bool(retarget_flags[seg_cursor])
            seg_cursor += 1
            lo, hi, _ = STRATA[stratum]
            surface_z = lo + (hi - lo) * u[6]
            want_table = stratum != "floor" and layer == "front" and rng.random() < 0.8
            table_depth = float(rng.uniform(0.30, 0.60))
            table_edge_gap = float(rng.uniform(0.05, 0.20))
            # joint (target, base) sampling with a cheap reachability filter
            targets: list[HandTarget] = []
            base: BaseStrategy | None = None
            hand_list = [h for h, a in zip(("left", "right"), active) if a]
            for attempt in range(40):
                uu = u[:6] if attempt == 0 else rng.random(6)
                cand = [_sample_hand_target(rng, hand_list[0], stratum, surface_z, layer, canonical, uu)]
                if len(hand_list) == 2:
                    for _ in range(30):
                        if rng.random() < 0.6:
                            mirrored = HandTarget(**{**dataclasses.asdict(cand[0]), "hand": hand_list[1]})
                            p0 = np.asarray(cand[0].pos)
                            p1 = p0 * np.array([1.0, -1.0, 1.0]) + rng.normal(0.0, 0.04, size=3)
                            if layer == "behind":
                                p1[0] = np.clip(p1[0], *X_BEHIND_RANGE)
                                p1[1] = math.copysign(np.clip(abs(p1[1]), *Y_BEHIND_ABS_RANGE), p1[1])
                            else:
                                p1[0] = np.clip(p1[0], *X_FRONT_RANGE)
                                p1[1] = np.clip(p1[1], *Y_RANGE)
                            mirrored.yaw = -cand[0].yaw
                            mirrored.roll = -cand[0].roll
                            z_min = surface_z + ee_min_height_above_surface(mirrored.rotation(), mirrored.hand)
                            mirrored.pos = (float(p1[0]), float(p1[1]), float(max(p1[2], z_min, surface_z + EE_ABOVE_SURFACE_RANGE[0])))
                            second = mirrored
                        else:
                            second = _sample_hand_target(rng, hand_list[1], stratum, surface_z, layer, canonical, rng.random(6))
                        dist = float(np.linalg.norm(np.asarray(second.pos) - np.asarray(cand[0].pos)))
                        if BIMANUAL_DISTANCE_RANGE[0] <= dist <= BIMANUAL_DISTANCE_RANGE[1]:
                            cand.append(second)
                            break
                    if len(cand) < 2:
                        continue
                for _ in range(6):
                    b = _sample_base(rng, stratum, surface_z, layer, hands_mode, cfg["forced_variant_share"])
                    if kneel_stance is not None and b.foot_mode != "kneel":
                        continue  # a kneel clip keeps its staggered stance: all segments kneel
                    if k > 0 and kneel_stance is None and b.foot_mode == "kneel":
                        continue  # never introduce a kneel after a flat/heel-lift segment
                    if layer == "behind":
                        b.waist_yaw = math.copysign(b.waist_yaw, cand[0].pos[1])
                    if all(target_reachable(b, t) for t in cand):
                        base = b
                        break
                if base is not None:
                    targets = cand
                    break
            reach_filtered = base is not None
            if base is None:
                # fall back: last candidates; flagged in the spec and caught by the solver gates
                targets = list(cand)
                base = b
                if len(targets) < len(hand_list):
                    # bimanual segment whose second target never satisfied the inter-hand distance: mirror the first
                    # one and push it out laterally to the minimum distance
                    first = targets[0]
                    other = HandTarget(**{**dataclasses.asdict(first), "hand": hand_list[1]})
                    sgn = 1.0 if other.hand == "left" else -1.0
                    other.yaw, other.roll = -first.yaw, -first.roll
                    if layer == "behind":
                        # one hand per side behind the body, |y| in [0.20, 0.30] -> 0.40-0.60 m apart
                        ay = float(np.clip(abs(first.pos[1]), Y_BEHIND_ABS_RANGE[0], 0.30))
                        first.pos = (first.pos[0], float(math.copysign(ay, first.pos[1])), first.pos[2])
                        other.pos = (first.pos[0], float(-math.copysign(ay, first.pos[1])), first.pos[2])
                    else:
                        gap = 0.36   # side by side, inside BIMANUAL_DISTANCE_RANGE
                        y_cands = [first.pos[1] + sgn * gap, first.pos[1] - sgn * gap]
                        y_new = next((y for y in y_cands if Y_RANGE[0] <= y <= Y_RANGE[1]), float(np.clip(y_cands[0], *Y_RANGE)))
                        other.pos = (first.pos[0], float(y_new), first.pos[2])
                    targets.append(other)
                if kneel_stance is not None and base.foot_mode != "kneel":
                    # a kneel clip keeps kneeling: convert the fallback base to a kneel on the same knee
                    base = dataclasses.replace(base, family="kneel", foot_mode="kneel", kneel_side=kneel_stance,
                                               pelvis_drop=float(STANDING_ROOT_Z - rng.uniform(*KNEEL_PELVIS_HEIGHT_RANGE)),
                                               heel_lift_pitch=0.0, leg_prior=None, drop_mode="deep")
                elif kneel_stance is None and k > 0 and base.foot_mode == "kneel":
                    # never introduce a kneel after a flat / heel-lift segment: fall back to a heel lift at the same height
                    h_k = max(0.40, base.pelvis_height)
                    pr = solve_leg_prior(h_k, base.pelvis_pitch, foot_pitch=0.35, knee_max=2.75, pelvis_x_pref=HEEL_LIFT_PELVIS_X_PREF)
                    base = dataclasses.replace(base, family="deep_squat", foot_mode="heel_lift", kneel_side="", heel_lift_pitch=0.35,
                                               pelvis_drop=float(STANDING_ROOT_Z - h_k),
                                               leg_prior={"hip_pitch": pr.hip_pitch, "knee": pr.knee, "ankle_pitch": pr.ankle_pitch})
            has_table = want_table and table_allowed(layer, stratum, surface_z, targets, table_edge_gap)
            if base.foot_mode == "kneel" and kneel_stance is None:
                kneel_stance = base.kneel_side
            elif kneel_stance is not None and base.foot_mode == "kneel":
                base.kneel_side = kneel_stance
            retarget = None
            if do_retarget:
                # the re-targeted goal must itself be reachable / clear of the body / above the surface envelope
                for _ in range(30):
                    direction = rng.normal(size=3)
                    direction[2] *= 0.5
                    direction /= max(1e-9, np.linalg.norm(direction))
                    mag = float(rng.uniform(0.10, 0.40))
                    offset = direction * mag
                    ok = True
                    for t in targets:
                        moved = HandTarget(**{**dataclasses.asdict(t), "pos": tuple(float(v) for v in np.asarray(t.pos) + offset)})
                        z_min = surface_z + ee_min_height_above_surface(moved.rotation(), moved.hand)
                        if not target_reachable(base, moved) or moved.pos[2] < z_min:
                            ok = False
                            break
                    if ok:
                        retarget = RetargetSpec(phase=float(rng.uniform(0.30, 0.70)), offset=tuple(float(v) for v in offset))
                        break
            segments.append(SegmentSpec(index=k, stratum=stratum, surface_z=float(surface_z), layer=layer, has_table=has_table,
                                        table_depth=table_depth, table_edge_gap=table_edge_gap, targets=targets, base=base,
                                        hold_s=_sample_hold(rng, cfg["hold_long_share"]), retarget=retarget,
                                        retract=False, time_scale=float(rng.uniform(0.75, 1.30)), reach_filtered=reach_filtered))
        if retract_flags[i]:
            segments[-1].retract = True
        if kneel_stance is not None:
            back = float(rng.uniform(0.30, 0.36))
            front = float(rng.uniform(0.15, 0.22))
            lx, rx = (-back, front) if kneel_stance == "left" else (front, -back)
        else:
            stag = float(rng.uniform(-0.05, 0.05))
            lx, rx = stag / 2.0, -stag / 2.0
        stance = StanceSpec(half_width=float(rng.uniform(0.10, 0.16)), stagger=float(lx - rx),
                            yaw0=math.radians(rng.uniform(-15.0, 15.0)), left_x=lx, right_x=rx)
        specs.append(ClipSpec(clip_id=f"{prefix}_{i:06d}", seed=clip_seed, profile=profile, hands_mode=hands_mode,
                              active_hands=active, stance=stance, segments=segments))
    return specs


def summarize_specs(specs: Sequence[ClipSpec]) -> dict[str, Any]:
    """Coverage shares for the sampled clip specifications."""
    from collections import Counter

    strata = Counter(); hands = Counter(); foot = Counter(); fam = Counter(); layer = Counter(); approach = Counter()
    holds = []; nseg = Counter(); retarget = 0; retract = 0; canonical = 0; targets = 0; long_hold_clips = 0
    replan_clips = 0; replan_segments = 0; retract_after_replan = 0
    for s in specs:
        hands[s.hands_mode] += 1
        nseg[len(s.segments)] += 1
        replan_clips += any(getattr(seg, "replan", False) for seg in s.segments)
        replan_segments += sum(bool(getattr(seg, "replan", False)) for seg in s.segments)
        if any(seg.retract for seg in s.segments):
            retract += 1
            retract_after_replan += bool(getattr(s.segments[-1], "replan", False) and s.segments[-1].retract)
        if any(seg.hold_s >= 2.0 for seg in s.segments):
            long_hold_clips += 1
        for seg in s.segments:
            strata[seg.stratum] += 1
            foot[seg.base.foot_mode] += 1
            fam[seg.base.family] += 1
            layer[seg.layer] += 1
            holds.append(seg.hold_s)
            retarget += seg.retarget is not None
            for t in seg.targets:
                targets += 1
                canonical += t.canonical_grasp
                approach[t.approach] += 1
    n_seg = max(1, sum(strata.values()))
    return {
        "clips": len(specs),
        "segments": n_seg,
        "strata_share": {k: v / n_seg for k, v in sorted(strata.items())},
        "hands_share": {k: v / max(1, len(specs)) for k, v in sorted(hands.items())},
        "foot_mode_share": {k: v / n_seg for k, v in sorted(foot.items())},
        "family_share": {k: v / n_seg for k, v in sorted(fam.items())},
        "layer_share": {k: v / n_seg for k, v in sorted(layer.items())},
        "approach_share": {k: v / max(1, targets) for k, v in sorted(approach.items())},
        "segments_per_clip_share": {int(k): v / max(1, len(specs)) for k, v in sorted(nseg.items())},
        "retarget_share": retarget / n_seg,
        "retract_share": retract / max(1, len(specs)),
        "retract_after_replan_share": retract_after_replan / max(1, len(specs)),   # replan_v2: retract sits on the last re-reach
        "canonical_share": canonical / max(1, targets),
        "hold_ge_2s_share_segments": float(np.mean([h >= 2.0 for h in holds])) if holds else 0.0,
        "hold_ge_2s_share_clips": long_hold_clips / max(1, len(specs)),
        "hold_mean_s": float(np.mean(holds)) if holds else 0.0,
        "replan_share_clips": replan_clips / max(1, len(specs)),
        "replan_segments_per_clip": replan_segments / max(1, len(specs)),
    }


def specs_to_jsonl(specs: Iterable[ClipSpec]) -> str:
    return "\n".join(s.to_json() for s in specs) + "\n"
