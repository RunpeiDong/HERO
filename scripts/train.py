#!/usr/bin/env python3
"""Train HERO in Isaac Sim; local TensorBoard/checkpoint logging is the default."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
import math
from pathlib import Path
import os
from typing import NamedTuple

from _bootstrap import ROOT


def parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--config", choices=("without_delta_anchor", "with_delta_anchor", "without_delta_ee",
                                        "without_delta_anchor_single", "with_delta_anchor_single", "without_delta_ee_single"),
                   default="without_delta_anchor", help="dual-actor PPO (paper) with/without delta anchor, the without_delta_ee ablation "
                   "(no residual EE feedback in the actor, no delta anchor), or the single whole-body actor variants (*_single)")
    p.add_argument("--motion-dir", type=Path, required=True, help="Converted AMASS directory from scripts/prepare_amass.py")
    p.add_argument("--allow-mixed-data", action="store_true", help="Explicitly use the source weights in CORPUS_MANIFEST.json; never adds data automatically")
    p.add_argument("--checkpoint", type=Path, help="Resume a matching training checkpoint; example ONNX weights have a different layout")
    p.add_argument("--allow-config-change", action="store_true",
                   help="With --checkpoint: continue although motion / curriculum / reward / termination / randomization settings "
                        "or the exploration-std clamp (--std-clamp-max) differ from the checkpoint, or the checkpoint carries no "
                        "progress for a curriculum term or for the per-environment height-offset curriculum (they restart at their "
                        "initial values). The findings are printed; without this flag they are an error")
    p.add_argument("--reset-sampler-on-resume", action="store_true",
                   help="With --checkpoint: restart the adaptive sampler's failure table from zeros if the checkpoint's table cannot "
                        "be restored: another sampling rule or sampler settings, or a table built for another source mix "
                        "(CORPUS_MANIFEST.json source_weights), clip count or set of source tags than this corpus. Without this "
                        "flag that is an error. A changed source mix is legitimate; it only forces the table to restart")
    p.add_argument("--std-clamp-max", type=float, default=None, metavar="FLOAT",
                   help="Upper clamp of the exploration std of both actor heads (dual-actor configs); default: none, the paper "
                        "recipe. See configs/README.md")
    p.add_argument("--output", type=Path, default=ROOT / "runs")
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--iterations", type=int, default=20000, help="Learning iterations to run; when resuming, they follow the checkpoint's iteration")
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logger", choices=("local", "wandb-offline"), default="local")
    p.add_argument("--dry-run", action="store_true", help="Validate config/checkpoint and print resolved contract without starting Isaac Sim")
    return p


class CorpusCensus(NamedTuple):
    """What ``build_config`` learns about the motion directory while validating it: the ``*.npz`` clip count the loader
    will register and the sorted source tags. The resume pre-flight compares both with a checkpoint's sampler table."""

    num_clips: int
    source_tags: tuple[str, ...]


def build_config(args):
    """The resolved :class:`ExperimentConfig` for ``args`` (see :func:`build_config_and_census`)."""
    return build_config_and_census(args)[0]


def build_config_and_census(args):
    """Validate the launcher arguments, the motion directory and its manifest; return the configuration and the corpus census."""
    from configs import DEFAULTS, SINGLE_ACTOR_CONFIGS
    from hero_isaacsim.config_values.command import get_motion_config, with_motion_config
    from hero_isaacsim.managers.command.hero import manifest_clip_end_policies
    from holosoma.config_types.logger import DisabledLoggerConfig, WandbLoggerConfig

    for name in ("num_envs", "iterations", "save_interval"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
    std_clamp_max = getattr(args, "std_clamp_max", None)
    if std_clamp_max is not None:
        if not (math.isfinite(std_clamp_max) and std_clamp_max > 0.0):
            raise ValueError("--std-clamp-max must be a positive float")
        if args.config in SINGLE_ACTOR_CONFIGS:
            raise ValueError("--std-clamp-max applies to the dual-actor configurations only; the single-actor PPO has no exploration-std clamp")
    motion_dir = args.motion_dir.expanduser().resolve()
    if not motion_dir.is_dir():
        raise ValueError(f"Motion directory does not exist: {motion_dir}")
    if not any(motion_dir.glob("*.npz")):
        raise ValueError(f"Motion directory contains no NPZ clips: {motion_dir}")
    manifest_path = motion_dir / "CORPUS_MANIFEST.json"
    if not manifest_path.is_file():
        raise ValueError("CORPUS_MANIFEST.json is required; prepare AMASS with scripts/prepare_amass.py")
    manifest = json.loads(manifest_path.read_text())
    weights = manifest.get("source_weights", {})
    if not isinstance(weights, dict) or not weights or any(
        not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in weights.values()
    ) or sum(weights.values()) <= 0:
        raise ValueError("Manifest must declare finite non-negative source_weights with a positive total")
    if not args.allow_mixed_data and (set(weights) != {"amass"} or set(manifest.get("source_tags", weights)) != {"amass"}):
        raise ValueError("Quick Start accepts AMASS only; mixed data requires --allow-mixed-data and explicit manifest weights")
    import numpy as np
    source_tags = set()
    num_clips = 0
    for clip in sorted(motion_dir.glob("*.npz")):
        num_clips += 1
        with np.load(clip, allow_pickle=False) as data:
            tag = str(np.asarray(data.get("source_tag", "")).item())
            if not tag or tag not in weights:
                raise ValueError(f"{clip.name}: source_tag {tag!r} has no explicit manifest weight")
            if bool(np.asarray(data.get("has_object", False)).item()):
                raise ValueError(f"{clip.name}: object clips are not supported in HERO training")
            source_tags.add(tag)
    if not args.allow_mixed_data and source_tags != {"amass"}:
        raise ValueError("Default training requires source_tag='amass' on every clip")
    cfg = DEFAULTS[args.config]
    checkpoint = str(args.checkpoint.expanduser().resolve()) if args.checkpoint else None
    if checkpoint and not Path(checkpoint).is_file():
        raise ValueError(f"Checkpoint does not exist: {checkpoint}")
    resume_flags = [flag for flag, given in (("--allow-config-change", getattr(args, "allow_config_change", False)),
                                             ("--reset-sampler-on-resume", getattr(args, "reset_sampler_on_resume", False))) if given]
    if resume_flags and not checkpoint:
        verb = "requires" if len(resume_flags) == 1 else "require"
        raise ValueError(f"{' and '.join(resume_flags)} {verb} --checkpoint: the resume opt-ins have no effect on a fresh run")
    log_dir = str(args.output.expanduser().resolve())
    logger = (DisabledLoggerConfig(base_dir=log_dir) if args.logger == "local" else
              WandbLoggerConfig(mode="offline", entity=None, project="hero", name=args.config, base_dir=log_dir))
    # The manifest's clip-end policies overlay the preset defaults (mocap rolls over, generated reaching clips hold).
    clip_end_policies = {**get_motion_config(cfg.command).clip_end_policy_by_source, **manifest_clip_end_policies(manifest)}
    algo_overrides = dict(num_learning_iterations=args.iterations, save_interval=args.save_interval)
    if std_clamp_max is not None:
        algo_overrides["hero_std_clamp_max"] = float(std_clamp_max)
    cfg = replace(cfg,
        training=replace(cfg.training, num_envs=args.num_envs, seed=args.seed, checkpoint=checkpoint),
        command=with_motion_config(cfg.command, motion_dir=str(motion_dir), source_weights=weights,
                                   clip_end_policy_by_source=clip_end_policies,
                                   reset_sampler_on_resume=bool(getattr(args, "reset_sampler_on_resume", False))),
        algo=replace(cfg.algo, config=replace(cfg.algo.config, **algo_overrides)),
        logger=logger)
    return cfg, CorpusCensus(num_clips=num_clips, source_tags=tuple(sorted(source_tags)))


def main(argv=None) -> int:
    p = parser()
    args = p.parse_args(argv)
    config_changes: list[str] = []
    curriculum_state_reset: str | None = None
    sampler_reset: str | None = None
    try:
        cfg, census = build_config_and_census(args)
        from hero_isaacsim.config_values.experiment import observation_contract, resume_preflight
        if cfg.training.checkpoint:
            import torch
            checkpoint = torch.load(cfg.training.checkpoint, map_location="cpu", weights_only=False)
            # Every opt-in finding is computed first and refused in ONE message, so a checkpoint that needs both flags
            # (e.g. one written before the current sampler rule and curriculum persistence, or one whose sampler table
            # was built for another source mix / clip registry) costs one launch, not two.
            preflight = resume_preflight(checkpoint, cfg, corpus_num_clips=census.num_clips, corpus_source_tags=census.source_tags)
            refusal = preflight.refusal(allow_config_change=args.allow_config_change,
                                        reset_sampler_on_resume=args.reset_sampler_on_resume)
            if refusal:
                raise ValueError(refusal)
            preflight.log_accepted()
            config_changes = preflight.config_changes
            curriculum_state_reset = preflight.curriculum_state_problem
            sampler_reset = preflight.sampler_state_problem
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    print(json.dumps({"config": args.config, "motion_dir": str(args.motion_dir.resolve()),
                      "checkpoint": cfg.training.checkpoint, "num_envs": cfg.training.num_envs,
                      "corpus": {"num_clips": census.num_clips, "source_tags": list(census.source_tags)},
                      "iterations": cfg.algo.config.num_learning_iterations, "logger": args.logger,
                      "std_clamp_max": getattr(cfg.algo.config, "hero_std_clamp_max", None),
                      "config_changes": config_changes, "curriculum_state_reset": curriculum_state_reset,
                      "sampler_reset": sampler_reset,
                      "observation": observation_contract(cfg)}, indent=2))
    if args.dry_run:
        return 0
    if args.logger == "wandb-offline":
        os.environ["WANDB_MODE"] = "offline"
    from holosoma.train_agent import train
    train(cfg)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
