"""CLI: MuJoCo evaluation of an exported HERO policy over benchmark clips (hero_bench_v1).

    # open loop (the reference is the clip; the robot starts at its frame 0)
    python -m sim2sim.bench.run --onnx-dir checkpoints/example --sidecar checkpoints/example/model_hero.json \\
        --motion-dir <bench dir[,dir2,...] or file.npz> [--clips '*.npz'] [--max-clips N] --out <dir> \\
        [--horizon-s 10] [--pad-s 0] [--fail-causes fall,anchor_xy] [--odom truth|so] [--odom-seed 0] [--jobs N] [--quiet]

    # closed loop: HERO replanning + goal adjustment from the live robot state (needs the bench manifest for the goals)
    python -m sim2sim.bench.run ... --pad-s 4 --horizon-s 14 --replan --bench-manifest <bench>/BENCH_MANIFEST.json \\
        --replan-first reach_end --replan-period-s 3.0 --replan-base current --goal-adjust --replan-blend-s 0.3

The policy is a ``model_XXXXX.onnx`` + ``model_XXXXX_hero.json`` export (:mod:`sim2sim.policy_hero_export`; the sidecar carries the
observation layout, the actuator tables and the command semantics).  ``--odom so`` feeds the root-feedback inputs (when the export has
them) from the LiDAR-inertial odometry model (:mod:`sim2sim.bench.odometry`) instead of the exact root pose; exports without those inputs
ignore it (the estimator only records its error).  ``--fail-causes`` lists the termination causes that count as failure (all are evaluated);
the alias ``fall_low`` enables the low-posture fall margin of the hero_bench_v1 low layers.

Writes ``<out>/series.npz`` (``hero_eval_series_v1``, ``sim="mujoco"``; policy / plant / odometry / replan provenance in ``meta_json``),
``<out>/summary.json`` and ``<out>/summary.md``.  Score the series with :mod:`sim2sim.bench.summary` (open loop) /
:mod:`sim2sim.bench.closed_loop_summary` (closed loop) and merge both with :mod:`sim2sim.bench.report`.
"""

from __future__ import annotations

import argparse
import datetime as _dt
import json
import os
import platform
import sys
import time
from dataclasses import asdict
from pathlib import Path
from typing import Any, Mapping

import numpy as np

from sim2sim.bench import __version__
from sim2sim.bench.metrics import METRIC_KEYS, ODOM_METRIC_KEYS
from sim2sim.bench.odometry import LidarInertialOdometryConfig
from sim2sim.bench.replan import ADJUST_MODES, BASE_MODES, FIRST_EVENTS, START_POSES, START_VELOCITIES, BenchGoal, BenchReplanner, ReplanConfig, clip_replan_config, load_bench_manifest, manifest_entry
from sim2sim.bench.rollout import JERK_METRIC_KEYS, ClipResult, RolloutConfig, rollout_clip
from sim2sim.bench.series import build_series, read_series, summarize, summary_markdown, write_series
from sim2sim.bench.terminations import CAUSE_ORDER, TerminationConfig, parse_fail_causes
from sim2sim.physics_profiles import FOOT_COLLISIONS, PHYSICS_PROFILES
from sim2sim.plant import URDF_FILE_NAME, MujocoPlant, default_urdf_path
from sim2sim.policy_hero_export import H_CMD_MODES as HX_H_CMD_MODES
from sim2sim.policy_hero_export import HOLD_MODES as HX_HOLD_MODES
from sim2sim.policy_hero_export import HeroExportPolicy
from sim2sim.reference import ClipReference, discover_clips

#: ``--urdf <alias>``: ``paddle`` = the paddle-hand copy of the training URDF (exports whose sidecar names a paddle-hand robot), ``dex3`` = the
#: Dex3 training URDF itself.
PADDLE_URDF_FILE_NAME = "g1_29dof_paddle3box_hero.urdf"
URDF_ALIASES: dict[str, str] = {"paddle": PADDLE_URDF_FILE_NAME, "dex3": URDF_FILE_NAME}
POLICIES: tuple[str, ...] = ("hero_export",)
#: ``--odom`` choices of the benchmark runner (``truth`` = exact simulator root pose; ``so`` = the LiDAR-inertial odometry model)
ODOM_CHOICES: tuple[str, ...] = ("truth", "so")
_WORKER: dict[str, Any] = {}


def _termination_config(args) -> TerminationConfig:
    """``--fail-causes`` -> :class:`TerminationConfig` (:func:`sim2sim.bench.terminations.parse_fail_causes`: the alias ``fall_low`` counts ``fall``
    and enables the hero_bench_v1 low-posture margin; without it ``fall_low_ref_margin_m`` stays None = every legacy number unchanged)."""
    causes, margin = parse_fail_causes(getattr(args, "fail_causes", "") or ",".join(CAUSE_ORDER))
    return TerminationConfig(fail_causes=causes, fall_low_ref_margin_m=margin)


def _replan_config(args) -> ReplanConfig | None:
    """``--replan`` -> the :class:`ReplanConfig` of the run (None without the flag)."""
    if not getattr(args, "replan", False):
        return None
    return ReplanConfig(
        period_s=args.replan_period_s,
        first_event=args.replan_first,
        first_delay_s=args.replan_first_delay_s,
        base_mode=args.replan_base,
        hold_s=args.replan_hold_s,
        reach_min_s=args.replan_reach_min_s,
        reach_max_s=args.replan_reach_max_s,
        stay_threshold_m=args.replan_stay_m,
        skip_below_m=args.replan_skip_m,
        max_replans=args.replan_max,
        goal_adjust=bool(args.goal_adjust),
        adjust_gain=args.goal_adjust_gain,
        adjust_max_step_m=args.goal_adjust_max_step_m,
        adjust_gate_m=args.goal_adjust_gate_m,
        adjust_mode=args.goal_adjust_mode,
        handover_blend_s=float(getattr(args, "replan_blend_s", 0.0) or 0.0),
        start_velocity=str(getattr(args, "replan_start_velocity", "zero")),
        start_pose=str(getattr(args, "replan_start_pose", "robot")),
    )


def pad_steps_for(args, control_dt: float) -> int:
    """Padding of every clip in control steps (``--pad-s``: the last reference frame is repeated so the rollout runs past the clip end)."""
    return int(round(args.pad_s / control_dt)) if (args.pad_s and args.pad_s > 0) else 0


def _bench_manifest_path(args) -> Path | None:
    if args.bench_manifest:
        return Path(args.bench_manifest)
    first = str(args.motion_dir).split(",")[0].strip()
    cand = Path(os.path.expanduser(first))
    cand = cand if cand.is_dir() else cand.parent
    m = cand / "BENCH_MANIFEST.json"
    return m if m.is_file() else None


def _load_manifest(args) -> dict[str, dict[str, Any]] | None:
    if _replan_config(args) is None:
        return None
    path = _bench_manifest_path(args)
    if path is None:
        raise SystemExit("--replan needs --bench-manifest (or a BENCH_MANIFEST.json next to the clips)")
    return load_bench_manifest(path)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("--policy", choices=POLICIES, default="hero_export", help="policy kind (hero_export = model_XXXXX.onnx + model_XXXXX_hero.json)")
    p.add_argument("--onnx-dir", default=None, help="dir with model_XXXXX.onnx (+ model_XXXXX_hero.json) or the .onnx file")
    p.add_argument("--sidecar", default=None, help="explicit *_hero.json (default: highest iteration in --onnx-dir; falls back to the ONNX metadata when no sidecar exists)")
    p.add_argument("--hero-export-h-cmd", choices=HX_H_CMD_MODES, default="auto", help="command_base_height: auto (clip-driven exports: clip h_ref clamped to h_cmd_min; fixed-command exports: h_cmd_default 0.75), clip (h_ref for both), fixed (--hero-export-h-fixed for both)")
    p.add_argument("--hero-export-h-fixed", type=float, default=None, help="h_cmd_default override (m)")
    p.add_argument("--hero-export-stand-flag", type=float, default=None, help="force command_stand (1 = walking gait); default = the export's evaluation semantics")
    p.add_argument("--hero-export-lookahead", type=int, default=None, help="ref_lookahead_frames override (default: sidecar hero_command, else 1)")
    p.add_argument("--hero-export-hold-mode", choices=HX_HOLD_MODES, default="zero_action", help="target of the one hold step after reset: zero_action (q_default + arm residual of the lookahead frame, default) or clip (frame-0 pose)")
    p.add_argument("--odom", choices=ODOM_CHOICES, default="truth", help="source of the robot root pose / velocity behind the root-feedback inputs: truth (exact simulator state, default) or so (LiDAR-inertial odometry model: 10 Hz corrections, 30 ms processing latency, 20 ms output age, 0.3 %% of distance drift). With so the estimator runs for every export and its error is recorded (odom_* columns); only exports with root-feedback inputs consume it")
    p.add_argument("--odom-seed", type=int, default=0, help="[--odom so] BASE seed of the noise streams; each clip's estimator runs with base + crc32(clip name) %% 2**31 (reproducible per clip across runs / conditions)")
    p.add_argument("--motion-dir", required=True, help="comma-separated clip directories and/or .npz files")
    p.add_argument("--clips", default="*.npz", help="glob inside each directory (default *.npz)")
    p.add_argument("--max-clips", type=int, default=None)
    p.add_argument("--out", required=True)
    p.add_argument("--horizon-s", type=float, default=10.0)
    p.add_argument("--physics-dt", type=float, default=None, help="physics step override (hero profile default .001 s)")
    p.add_argument("--substeps", type=int, default=None, help="physics steps per policy tick (hero profile default 20 -> 50 Hz control)")
    p.add_argument("--foot-collision", choices=FOOT_COLLISIONS, default="sonic_box", help="URDF foot collision: the audited deployment sole box (default), sonic_train = the training foot (seven capsules per ankle) or legacy_mesh for historical reproduction")
    p.add_argument("--physics-profile", choices=PHYSICS_PROFILES, default="hero", help="hero = training joint tables / 1 kHz physics (default); sonic_deploy_adapted = deployment timing / joint / contact defaults")
    p.add_argument("--urdf", default=None, help="training URDF path (default: the bundled g1_29dof_dex3fixed_hero.urdf, or the family named by the export's robot_urdf_path); 'paddle' / 'dex3' select the bundled copies")
    p.add_argument("--reset-vel", choices=("zero", "ref"), default="zero", help="reset velocities: zero (default) or the clip's frame-0 velocities")
    p.add_argument("--fail-causes", default=",".join(CAUSE_ORDER), help="comma list of termination causes that count as failure for first_fail_step (all are still evaluated); e.g. fall,anchor_xy for a policy without a pelvis-pose reference. Alias fall_low (hero_bench_v1 low-posture layers) = fall + the low-reference margin rule (robot pelvis < reference pelvis - 0.20 m while the reference pelvis <= 0.50 m; fall_low:<m> sets the margin); default off = legacy numbers unchanged")
    p.add_argument("--self-collisions", action="store_true", help="enable robot self-collisions (training had them OFF)")
    p.add_argument("--pad-s", type=float, default=0.0, help="repeat the last reference frame for this many seconds (rollout runs past the clip end; hold window grows)")
    # ---- closed loop (HERO replanning + goal adjustment; sim2sim.bench.replan) ---------------------------------------------
    p.add_argument("--replan", action="store_true", help="closed-loop replanning from the live robot state with the benchmark mink IK (needs --bench-manifest)")
    p.add_argument("--bench-manifest", default=None, help="BENCH_MANIFEST.json of the bench corpus (default: <first motion dir>/BENCH_MANIFEST.json)")
    p.add_argument("--replan-period-s", type=float, default=1.0, help="replan cadence (s) after the first event (HERO paper: 6 s; benchmark script update_rate 5 Hz)")
    p.add_argument("--replan-first", choices=FIRST_EVENTS, default="reach_end", help="first replan event: the original plan's reach_end frame (default) or the first step")
    p.add_argument("--replan-first-delay-s", type=float, default=0.0)
    p.add_argument("--replan-base", choices=BASE_MODES, default="clip", help="base end state of the replanned reference: the original plan (full-body policies) or the current pose (HERO: arms + waist only)")
    p.add_argument("--replan-hold-s", type=float, default=12.0)
    p.add_argument("--replan-reach-min-s", type=float, default=0.5)
    p.add_argument("--replan-reach-max-s", type=float, default=3.0)
    p.add_argument("--replan-stay-m", type=float, default=0.0175, help="HERO STAY_THRESHOLD: freeze the reference below this world EE error")
    p.add_argument("--replan-skip-m", type=float, default=0.02, help="HERO STOP_SHIFT_GAIN_THRESHOLD: no replan below this error")
    p.add_argument("--replan-max", type=int, default=20, help="HERO MAX_REPLAN_NUMBER")
    p.add_argument("--goal-adjust", action="store_true", help="HERO goal adjustment g <- g - beta e at every replan update")
    p.add_argument("--goal-adjust-gain", type=float, default=0.6, help="beta / SHIFT_GAIN")
    p.add_argument("--goal-adjust-max-step-m", type=float, default=0.01, help="per-update clip (code 1 cm; paper text 5 mm)")
    p.add_argument("--goal-adjust-gate-m", type=float, default=0.15, help="REPLAN_DISTANCE: adjust only when |e| < gate (or after 3 replans)")
    p.add_argument("--goal-adjust-mode", choices=ADJUST_MODES, default="accumulate", help="accumulate (paper formula) or reset (deployed code re-derives g from the world goal each update)")
    p.add_argument("--replan-mjcf", default=None, help="Dex3 MJCF scene for the IK model (default: data_tools.hero_reach_generator.DEFAULT_MJCF)")
    p.add_argument("--replan-blend-s", type=float, default=0.0, help="[--replan] crossfade the reference the policy sees from the OLD reference to the new plan over this many seconds after each swap (positions / dofs linear, quaternions slerp, baked into the new reference; 0 = hard swap, default)")
    p.add_argument("--replan-start-velocity", choices=START_VELOCITIES, default="zero", help="[--replan] zero (default: the palm path restarts from rest on a cubic smoothstep) | match (quintic minimum-jerk path leaving with the OLD reference's palm velocity at the swap)")
    p.add_argument("--replan-start-pose", choices=START_POSES, default="robot", help="[--replan] robot (default: the palm path starts at the robot's palm) | reference (starts at the OLD reference's current palm pose with the IK seeded by its arm joints)")
    p.add_argument("--jerk-metrics", action="store_true", help="record the jerk columns (arm PD-target discontinuity, measured arm joint acceleration, live-reference palm speed) for every rollout -- always on under --replan")
    p.add_argument("--jobs", type=int, default=1, help="worker processes (clips are independent)")
    p.add_argument("--tag", default=None, help="label stored in meta (default: policy tag)")
    p.add_argument("--quiet", action="store_true")
    return p


def _make_policy(args) -> HeroExportPolicy:
    if not args.onnx_dir and not args.sidecar:
        raise SystemExit("--policy hero_export needs --onnx-dir (model_XXXXX.onnx [+ model_XXXXX_hero.json]) or --sidecar")
    return HeroExportPolicy(
        args.onnx_dir,
        sidecar=args.sidecar,
        h_cmd=args.hero_export_h_cmd,
        h_fixed=args.hero_export_h_fixed,
        stand_flag=args.hero_export_stand_flag,
        ref_lookahead=args.hero_export_lookahead,
        hold_mode=args.hero_export_hold_mode,
        tag=args.tag,
        odom_source=getattr(args, "odom", "truth"),
    )


def resolve_urdf_alias(alias: str) -> str:
    """``"paddle"`` / ``"dex3"`` -> the file next to the default Dex3 URDF.  The file MUST exist: ``default_urdf_path`` would otherwise fall
    through to the Dex3 URDF and the whole run would be reported on the wrong plant."""
    if alias not in URDF_ALIASES:
        raise SystemExit(f"unknown URDF alias {alias!r} (known: {sorted(URDF_ALIASES)})")
    base = default_urdf_path(None).parent
    path = base / URDF_ALIASES[alias]
    if not path.is_file():
        raise SystemExit(f"URDF alias {alias!r} -> {URDF_ALIASES[alias]} is missing next to the Dex3 URDF ({base}). Set HERO_ROBOT_ASSET_ROOT=<root>/robots or pass --urdf <file>.")
    return str(path)


def _urdf_hint(args, policy=None) -> str | None:
    """``--urdf`` (alias or path), else the family the export prefers (``preferred_urdf_alias``: ``paddle`` for paddle-hand exports), else None
    (= the bundled Dex3 training URDF)."""
    hint = args.urdf
    if hint is None and policy is not None:
        hint = getattr(policy, "preferred_urdf_alias", None)
    if hint in URDF_ALIASES:
        hint = resolve_urdf_alias(hint)
    return hint


def odometry_config(args) -> LidarInertialOdometryConfig | None:
    """``--odom so`` -> the LiDAR-inertial config (``so`` preset, ``--odom-seed``); None under ``--odom truth`` (no estimator, no odom columns)."""
    src = str(getattr(args, "odom", "truth"))
    if src == "truth":
        return None
    if src != "so":
        raise SystemExit(f"--odom must be one of {ODOM_CHOICES}, got {src!r}")
    return LidarInertialOdometryConfig.preset("so", seed=int(args.odom_seed))


def odometry_meta(args, policy) -> dict[str, Any]:
    """Series meta ``odometry``: the requested source, the estimator config, whether the policy consumes the estimate and which terms it feeds
    (``fed_to_policy`` False = the estimator only records its error: an export without root-feedback terms)."""
    cfg = odometry_config(args)
    feeds = list(getattr(policy, "odom_feeds_terms", ()) or ())
    src = str(getattr(args, "odom", "truth"))
    meta = {
        "source": src,
        "noise": "off",
        "config": (cfg.as_dict() if cfg is not None else None),
        "fed_to_policy": bool(cfg is not None and feeds),
        "policy_terms": feeds,
        "policy_odom_source": getattr(policy, "odom_source", None),
        "metric_keys": (list(ODOM_METRIC_KEYS) if cfg is not None else []),
    }
    if src == "so":
        meta["preset"] = "so"
        meta["kind"] = "lidar_inertial"
    return meta


def default_tag(args, policy) -> str:
    """``--tag`` when given, else the policy tag with an ``_odom-so`` suffix under ``--odom so``."""
    if args.tag:
        return str(args.tag)
    tag = str(policy.tag)
    if str(getattr(args, "odom", "truth")) == "so":
        tag += "_odom-so"
    return tag


def _make_plant_policy(args) -> tuple[MujocoPlant, HeroExportPolicy]:
    policy = _make_policy(args)
    for w in list(getattr(getattr(policy, "contract", None), "warnings", None) or []):
        # sidecar / metadata disagreements are worth a line even under --quiet (the series meta carries policy.warnings as well)
        print(f"[bench] WARNING {getattr(policy, 'tag', args.policy)}: {w}", file=sys.stderr, flush=True)
    hint = _urdf_hint(args, policy)
    if args.urdf is None and hint is not None and not getattr(args, "quiet", False):
        print(f"[bench] plant URDF selected by the export ({getattr(policy, 'tag', args.policy)} robot_urdf_path family '{getattr(policy, 'preferred_urdf_alias', None)}'): {hint}", flush=True)
    plant = MujocoPlant(
        default_urdf_path(hint),
        plant_kind="dex3_urdf",
        physics_dt=args.physics_dt,
        substeps=args.substeps,
        foot_collision=getattr(args, "foot_collision", "sonic_box"),
        physics_profile=getattr(args, "physics_profile", "hero"),
        kp=policy.kp,
        kd=policy.kd,
        effort_limit=policy.effort_limit,
        self_collisions=bool(args.self_collisions),
        keep_visual=False,
    )
    return plant, policy


def _init_worker(args) -> None:
    """Pool initializer (spawn context)."""
    _WORKER["plant"], _WORKER["policy"] = _make_plant_policy(args)
    _WORKER["args"] = args
    _WORKER["manifest"] = _load_manifest(args)
    _WORKER["replan_cfg"] = _replan_config(args)


def _run_one(task: tuple[int, str]) -> tuple[int, ClipResult | None, str | None, float]:
    idx, path = task
    args = _WORKER["args"]
    plant, policy = _WORKER["plant"], _WORKER["policy"]
    t0 = time.time()
    try:
        ref = ClipReference(path)
        pad_steps = pad_steps_for(args, plant.control_dt)
        if pad_steps > 0:
            ref = ref.padded(pad_steps)
        cfg = RolloutConfig(
            horizon_steps=int(round(args.horizon_s / plant.control_dt)),
            reset_velocities=args.reset_vel,
            termination=_termination_config(args),
            odometry=odometry_config(args),
            record_jerk=bool(getattr(args, "jerk_metrics", False)),
        )
        replanner = None
        rp_cfg = _WORKER.get("replan_cfg")
        if rp_cfg is not None:
            entry = manifest_entry(_WORKER["manifest"], ref.name)
            # hero_bench_v1: a clip with its OWN retract segment stops replanning at hold_end_frame (ReplanConfig.stop_frame)
            rp_cfg = clip_replan_config(rp_cfg, entry)
            goal = BenchGoal.from_manifest_entry(entry)
            arm_posture = ref.joint_pos[ref.clamp(goal.reach_end_frame)]
            replanner = BenchReplanner(goal, rp_cfg, mjcf_path=args.replan_mjcf, arm_posture=arm_posture, reference=ref)
        res = rollout_clip(plant, policy, ref, cfg, replanner=replanner)
        return idx, res, None, time.time() - t0
    except Exception as exc:  # noqa: BLE001 - one bad clip must not kill the run
        return idx, None, f"{type(exc).__name__}: {exc}", time.time() - t0


def _policy_line(policy: HeroExportPolicy) -> str:
    d = policy.describe()
    return (f"{d['tag']} ({d['kind']}, {d['arm']} obs {d['obs_dim']} = {d['frame_dim']} x {d['history_length']} {d['history_layout']}, "
            f"contract={d['action_contract']} clip={d['action_clip']}, source={d['source']}, preset={d['preset']}, iter={d['iteration']}, "
            f"h_cmd={d['h_cmd_mode']}, lookahead={d['ref_lookahead']}, hold={d['hold_mode']}, urdf={d['preferred_urdf_alias'] or 'dex3'}, effort={d['effort_limit_source']}, "
            f"odom={d['odom_source']}" + (f" -> {','.join(t[:3] for t in d['odom_feeds_terms'])}" if d.get("odom_feeds_terms") else "") + ")")


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    clips = discover_clips(args.motion_dir, args.clips, args.max_clips)
    if not clips:
        print(f"[bench] no clips found in {args.motion_dir} ({args.clips})", file=sys.stderr)
        return 2
    plant, policy = _make_plant_policy(args)
    odom_meta = odometry_meta(args, policy)
    if odom_meta["config"] is not None and not odom_meta["fed_to_policy"] and not args.quiet:
        print(f"[bench] NOTE --odom {args.odom}: policy {getattr(policy, 'tag', args.policy)} consumes no odometry estimate (no root-feedback term in the export); the estimator only records its error", flush=True)
    horizon_steps = int(round(args.horizon_s / plant.control_dt))
    rp_cfg_main = _replan_config(args)
    meta = {
        "sim2sim_version": __version__,
        "timestamp": _dt.datetime.now(_dt.timezone.utc).isoformat(timespec="seconds"),
        "host": platform.node(),
        "python": sys.version.split()[0],
        "numpy": np.__version__,
        "policy_kind": args.policy,
        "policy": policy.describe(),
        "plant_kind": plant.plant_kind,
        "plant": plant.describe(),
        "rollout": {"horizon_s": args.horizon_s, "horizon_steps": horizon_steps, "control_dt": plant.control_dt, "reset_velocities": args.reset_vel},
        "termination": asdict(_termination_config(args)),
        "pad_s": float(args.pad_s),
        "odometry": odom_meta,
        "replan": (rp_cfg_main.as_dict() if rp_cfg_main is not None else None),
        "bench_manifest": (str(_bench_manifest_path(args)) if rp_cfg_main is not None else None),
        "motion_dir": args.motion_dir,
        "clips_glob": args.clips,
        "num_clips": len(clips),
        "tag": default_tag(args, policy),
        "argv": sys.argv[1:] if argv is None else list(argv),
    }
    if not args.quiet:
        odom_s = ""
        if odom_meta["config"] is not None:
            oc = odom_meta["config"]
            odom_s = f" | odometry so rate={oc['rate_hz']:g} Hz latency={1000.0 * oc['latency_s']:g} ms output_age={1000.0 * oc['output_delay_s']:g} ms propagation={oc['propagation']} drift={oc['drift_pct']:g} % fed={odom_meta['fed_to_policy']}"
        print(f"[bench] policy {_policy_line(policy)} | plant {plant.plant_kind} {plant.model_path.name} "
              f"mass {plant.total_mass:.2f} kg dt={plant.physics_dt} x{plant.substeps} self_coll={plant.self_collisions_enabled} | {len(clips)} clips, horizon {horizon_steps} steps"
              f" | replan={rp_cfg_main is not None}{odom_s}", flush=True)
    tasks = [(i, str(p)) for i, p in enumerate(clips)]
    results: list[ClipResult | None] = [None] * len(clips)
    errors: dict[str, str] = {}
    t_start = time.time()
    if args.jobs and args.jobs > 1:
        import multiprocessing as mp

        ctx = mp.get_context("spawn")
        with ctx.Pool(processes=int(args.jobs), initializer=_init_worker, initargs=(args,)) as pool:
            for idx, res, err, dt_ in pool.imap_unordered(_run_one, tasks):
                _report(idx, clips, res, err, dt_, args.quiet)
                results[idx] = res
                if err:
                    errors[clips[idx].name] = err
    else:
        _WORKER["plant"], _WORKER["policy"], _WORKER["args"] = plant, policy, args
        _WORKER["manifest"], _WORKER["replan_cfg"] = _load_manifest(args), rp_cfg_main
        for task in tasks:
            idx, res, err, dt_ = _run_one(task)
            _report(idx, clips, res, err, dt_, args.quiet)
            results[idx] = res
            if err:
                errors[clips[idx].name] = err
    done = [r for r in results if r is not None]
    for name, error in errors.items():
        print(f"[bench] ERROR {name}: {error}", file=sys.stderr, flush=True)
    if not done:
        print(f"[bench] every clip failed: {errors}", file=sys.stderr)
        return 3
    replan_log = {r.clip_name: r.extra.get("replan") for r in done if r.extra.get("replan") is not None}
    metric_keys = (list(METRIC_KEYS) + (list(ODOM_METRIC_KEYS) if odom_meta["config"] is not None else [])
                   + (list(JERK_METRIC_KEYS) if (rp_cfg_main is not None or args.jerk_metrics) else []))
    clip_len_original = {r.clip_name: int(r.extra.get("clip_len_steps_original", r.clip_len_steps)) for r in done}
    series = build_series(
        sim="mujoco",
        dt=plant.control_dt,
        horizon_steps=horizon_steps,
        meta={**meta, "errors": errors, "wall_s": time.time() - t_start, "clip_len_steps_original": clip_len_original,
              "replan_log": ({k: _compact_replan(v) for k, v in replan_log.items()} if replan_log else None)},
        clip_name=[r.clip_name for r in done],
        source_tag=[r.source_tag for r in done],
        clip_len_steps=[r.clip_len_steps for r in done],
        valid_until=[r.valid_until for r in done],
        first_fail_step=[r.first_fail_step for r in done],
        first_fail_cause=[r.first_fail_cause for r in done],
        metrics={k: np.stack([r.metrics[k] for r in done]) for k in metric_keys},
    )
    series_path = write_series(out / "series.npz", series)
    summary = summarize(read_series(series_path))
    (out / "summary.json").write_text(json.dumps(summary, indent=2, default=str))
    (out / "summary.md").write_text(summary_markdown(summary))
    ov = summary["overall"]
    if not args.quiet:
        odo = summary.get("odometry")
        odo_s = ""
        if odo:
            kk = odo["overall"]["keys"]

            def _e(k):
                return kk.get(k, {}).get("end_abs", {}).get("mean", float("nan"))

            oc = odom_meta["config"]
            odo_s = (f"\n  odometry (so, latency {1000.0 * oc['latency_s']:g} ms, output age {1000.0 * oc['output_delay_s']:g} ms, {oc['propagation']}): "
                     f"end |xy| {_e('odom_xy_err_cm'):.2f} cm | end |yaw| {_e('odom_yaw_err_deg'):.2f} deg | end |z| {_e('odom_z_err_cm'):.2f} cm | fed_to_policy {odom_meta['fed_to_policy']}")
        print(
            f"[bench] done {len(done)}/{len(clips)} clips in {time.time() - t_start:.1f}s -> {out}\n"
            f"  local EE {ov['metrics']['ee_local_cm']['mean']:.2f} cm | global EE {ov['metrics']['ee_global_cm']['mean']:.2f} cm | "
            f"rot {ov['metrics']['ee_rot_deg']['mean']:.1f} deg | arm joint {ov['metrics']['joint_upper_rad']['mean']:.3f} rad | "
            f"fail-free {ov['fail_free_fraction']:.3f} | causes {ov['first_fail_cause_counts']}{odo_s}",
            flush=True,
        )
    # Keep partial series for diagnosis, but never mark a run with omitted clips as successful.
    return 3 if errors else 0


def _compact_replan(v: Mapping[str, Any]) -> dict[str, Any]:
    """Per-clip replan summary kept in the series meta (events + per-plan IK stats, no arrays)."""
    return {
        "n_replans": v.get("n_replans"),
        "stayed_at_step": v.get("stayed_at_step"),
        "goal_offset_cm": v.get("goal_offset_cm"),
        "goal_offset_m": v.get("goal_offset_m"),
        "ik_time_s": v.get("ik_time_s"),
        "mean_terminal_ik_err_cm": v.get("mean_terminal_ik_err_cm"),
        "replan_steps": v.get("replan_steps"),
        "events": v.get("events"),
        "plans": [{k: p.get(k) for k in ("step", "dist_cm", "ang_deg", "n_reach", "ik_s", "terminal_palm_err_cm", "max_palm_path_err_cm", "qp_failures", "goal_offset_cm", "pelvis_z_start", "pelvis_z_end",
                                         "start_pose", "start_velocity", "start_pose_offset_cm", "start_speed_cm_s", "settle_palm_err_cm", "blend_frames", "blended_frames",
                                         "handover") if k in p} for p in (v.get("plans") or [])],
        **({"stop": v.get("stop")} if v.get("stop") is not None else {}),
    }


def _report(idx: int, clips, res: ClipResult | None, err: str | None, dt_: float, quiet: bool) -> None:
    if quiet:
        return
    name = Path(clips[idx]).name
    if err:
        print(f"[bench] {idx + 1}/{len(clips)} {name}: ERROR {err}", flush=True)
        return
    m = res.metrics
    vu = res.valid_until

    def avg(k):
        v = m[k][:vu]
        return float(np.nanmean(v)) if vu > 0 else float("nan")

    fail = f"fail@{res.first_fail_step} {res.first_fail_cause}" if res.first_fail_step > 0 else "no-fail"
    rp = res.extra.get("replan") if isinstance(res.extra, dict) else None
    rp_s = f" | replans {rp.get('n_replans')} stay@{rp.get('stayed_at_step')} goal_off {rp.get('goal_offset_cm', 0.0):.1f} cm ik {rp.get('ik_time_s', 0.0):.1f}s" if rp else ""
    jk = res.extra.get("jerk") if isinstance(res.extra, dict) else None
    if rp and jk and jk.get("post_replan", {}).get("arm_target_delta_rad") is not None:
        rp_s += (f" | jerk tgt {1000.0 * (jk['mean'].get('arm_target_delta_rad') or 0.0):.1f}->{1000.0 * jk['post_replan']['arm_target_delta_rad']:.1f} mrad/step"
                 f" acc {jk['mean'].get('arm_accel_rad_s2') or 0.0:.1f}->{jk['post_replan'].get('arm_accel_rad_s2') or 0.0:.1f} rad/s2 (all->post {jk['window_s']:g} s)")
    odo = res.extra.get("odometry") if isinstance(res.extra, dict) else None
    if odo and odo.get("end"):
        e = odo["end"]

        def _ev(k):
            v = e.get(k)
            return float("nan") if v is None else float(v)

        rp_s += f" | odom end xy {_ev('odom_xy_err_cm'):.1f} cm yaw {_ev('odom_yaw_err_deg'):.2f} deg z {_ev('odom_z_err_cm'):.1f} cm"
    print(
        f"[bench] {idx + 1}/{len(clips)} {name}: steps {vu} | local {avg('ee_local_cm'):.2f} cm | global {avg('ee_global_cm'):.2f} cm | "
        f"rot {avg('ee_rot_deg'):.1f} deg | arm {avg('joint_upper_rad'):.3f} rad | xy {avg('anchor_xy_cm'):.1f} cm | {fail}{rp_s} | {dt_:.1f}s",
        flush=True,
    )


if __name__ == "__main__":
    sys.exit(main())
