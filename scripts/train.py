#!/usr/bin/env python3
"""Train HERO in Isaac Sim; local TensorBoard/checkpoint logging is the default."""
from __future__ import annotations

import argparse
from dataclasses import replace
import json
from pathlib import Path
import os

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
    p.add_argument("--output", type=Path, default=ROOT / "runs")
    p.add_argument("--num-envs", type=int, default=4096)
    p.add_argument("--iterations", type=int, default=20000, help="Learning iterations to run; when resuming, they follow the checkpoint's iteration")
    p.add_argument("--save-interval", type=int, default=500)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--logger", choices=("local", "wandb-offline"), default="local")
    p.add_argument("--dry-run", action="store_true", help="Validate config/checkpoint and print resolved contract without starting Isaac Sim")
    return p


def build_config(args):
    from configs import DEFAULTS
    from hero_isaacsim.config_values.command import with_motion_config
    from holosoma.config_types.logger import DisabledLoggerConfig, WandbLoggerConfig

    for name in ("num_envs", "iterations", "save_interval"):
        if getattr(args, name) <= 0:
            raise ValueError(f"--{name.replace('_', '-')} must be positive")
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
    import math
    if not isinstance(weights, dict) or not weights or any(
        not isinstance(v, (int, float)) or not math.isfinite(v) or v < 0 for v in weights.values()
    ) or sum(weights.values()) <= 0:
        raise ValueError("Manifest must declare finite non-negative source_weights with a positive total")
    if not args.allow_mixed_data and (set(weights) != {"amass"} or set(manifest.get("source_tags", weights)) != {"amass"}):
        raise ValueError("Quick Start accepts AMASS only; mixed data requires --allow-mixed-data and explicit manifest weights")
    import numpy as np
    source_tags = set()
    for clip in sorted(motion_dir.glob("*.npz")):
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
    log_dir = str(args.output.expanduser().resolve())
    logger = (DisabledLoggerConfig(base_dir=log_dir) if args.logger == "local" else
              WandbLoggerConfig(mode="offline", entity=None, project="hero", name=args.config, base_dir=log_dir))
    return replace(cfg,
        training=replace(cfg.training, num_envs=args.num_envs, seed=args.seed, checkpoint=checkpoint),
        command=with_motion_config(cfg.command, motion_dir=str(motion_dir), source_weights=weights),
        algo=replace(cfg.algo, config=replace(cfg.algo.config,
            num_learning_iterations=args.iterations, save_interval=args.save_interval)),
        logger=logger)


def main(argv=None) -> int:
    p = parser()
    args = p.parse_args(argv)
    try:
        cfg = build_config(args)
        from hero_isaacsim.config_values.experiment import observation_contract, validate_checkpoint_contract
        if cfg.training.checkpoint:
            import torch
            validate_checkpoint_contract(torch.load(cfg.training.checkpoint, map_location="cpu", weights_only=False), cfg)
    except (ValueError, OSError) as exc:
        p.error(str(exc))
    print(json.dumps({"config": args.config, "motion_dir": str(args.motion_dir.resolve()),
                      "checkpoint": cfg.training.checkpoint, "num_envs": cfg.training.num_envs,
                      "iterations": cfg.algo.config.num_learning_iterations, "logger": args.logger,
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
