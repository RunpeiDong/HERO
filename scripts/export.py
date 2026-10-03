#!/usr/bin/env python3
"""Export a HERO training checkpoint to an ONNX + JSON bundle using Isaac Sim."""
from __future__ import annotations
import argparse
from dataclasses import replace
import json
from pathlib import Path
import sys

from _bootstrap import ROOT


def validate_bundle(onnx_path: Path) -> tuple[Path, Path]:
    """Check an exported ONNX and its ``_hero.json`` sidecar; raise ValueError on any mismatch."""
    import onnx
    from hero_isaacsim.agents.ppo_dual.export import hero_sidecar_path
    sidecar = Path(hero_sidecar_path(str(onnx_path)))
    if not onnx_path.is_file() or not sidecar.is_file():
        raise ValueError(f"Export must produce both {onnx_path.name} and {sidecar.name}")
    data = json.loads(sidecar.read_text())
    model = onnx.load(str(onnx_path))
    onnx.checker.check_model(model)
    expected = int(data["actor_obs_dim"])
    if len(model.graph.input) not in (1, 2):
        raise ValueError("HERO export must have one (single actor) or two (dual actor) inputs")
    for value in model.graph.input:
        if value.type.tensor_type.shape.dim[-1].dim_value != expected:
            raise ValueError("ONNX input width disagrees with sidecar actor_obs_dim")
    if len(model.graph.output) != 1 or model.graph.output[0].type.tensor_type.shape.dim[-1].dim_value != 29:
        raise ValueError("HERO export must produce a 29-DoF action")
    return onnx_path, sidecar


def export_config(raw: dict, motion_dir: Path, output: Path):
    """``(saved, eval_cfg)`` for exporting the checkpoint ``raw``: the saved training configuration made loadable for an
    inference-only environment, and the single-env evaluation config built from it.

    Export never depends on the checkpoint's training-distribution state: the robot assets come from this checkout,
    the sampler settings from the current preset and ``reset_sampler_on_resume`` is set
    (``with_inference_motion_config``), so a checkpoint written under earlier sampler settings, or exported against
    a different ``--motion-dir`` than it was trained on, builds its environment and loads without the resume-only
    errors ``scripts/train.py`` guards with ``--reset-sampler-on-resume``. Raises ``ValueError`` when the checkpoint
    was not trained with a HERO configuration or its observation layout does not match that configuration."""
    from holosoma.config_types.experiment import ExperimentConfig
    from holosoma.config_types.logger import DisabledLoggerConfig
    from hero_isaacsim.config_values.command import with_inference_motion_config
    from configs import DEFAULTS
    from hero_isaacsim.config_values.experiment import validate_checkpoint_contract
    saved = ExperimentConfig(**raw["experiment_config"])
    if saved.training.name not in DEFAULTS:
        raise ValueError("Export requires a checkpoint trained with a HERO configuration.")
    current = DEFAULTS[saved.training.name]
    validate_checkpoint_contract(raw, current)
    # Resolve robot assets from this checkout, never from an old saved absolute path.
    saved = replace(saved, robot=current.robot,
                    command=with_inference_motion_config(saved.command, current.command, motion_dir=str(motion_dir)),
                    logger=DisabledLoggerConfig(base_dir=str(output / "eval_logs")))
    cfg = saved.get_eval_config()
    cfg = replace(cfg, training=replace(cfg.training, headless=True, num_envs=1, export_onnx=True, max_eval_steps=1))
    return saved, cfg


def main(argv=None) -> int:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--checkpoint", required=True, type=Path)
    p.add_argument("--motion-dir", required=True, type=Path, help="Evaluation clips needed to initialize the training environment")
    p.add_argument("--output", required=True, type=Path)
    p.add_argument("--dry-run", action="store_true")
    args = p.parse_args(argv)
    checkpoint = args.checkpoint.expanduser().resolve()
    motion_dir = args.motion_dir.expanduser().resolve()
    output = args.output.expanduser().resolve()
    if not checkpoint.is_file() or not motion_dir.is_dir():
        p.error("--checkpoint must be a file and --motion-dir must be a directory")
    from hero_isaacsim.agents.ppo_dual.export import hero_sidecar_path
    from hero_isaacsim.config_values.experiment import observation_contract
    import torch
    raw = torch.load(checkpoint, map_location="cpu", weights_only=False)
    try:
        saved, cfg = export_config(raw, motion_dir, output)
    except ValueError as exc:
        p.error(str(exc))
    target = output / (checkpoint.stem + ".onnx")
    if args.dry_run:
        print(json.dumps({"checkpoint": str(checkpoint), "output": str(output), "onnx": str(target),
                          "sidecar": hero_sidecar_path(str(target)), "requires": "Isaac Sim",
                          "observation": observation_contract(cfg)}, indent=2))
        return 0
    # Closing the Isaac Sim app ends the Python process, so the bundle is written, checked, and
    # reported before close_simulation_app(); a visible sidecar means the ONNX next to it is complete.
    from holosoma.utils.config_utils import CONFIG_NAME
    from holosoma.utils.experiment_paths import get_experiment_dir, get_timestamp
    from holosoma.utils.helpers import get_class
    from holosoma.utils.sim_utils import close_simulation_app, setup_simulation_environment
    env, device, simulation_app = setup_simulation_environment(cfg)
    code = 0
    try:
        log_dir = get_experiment_dir(cfg.logger, cfg.training, get_timestamp(), task_name="export")
        log_dir.mkdir(parents=True, exist_ok=True)
        cfg.save_config(str(log_dir / CONFIG_NAME))
        algo = get_class(cfg.algo._target_)(device=device, env=env, config=cfg.algo.config, log_dir=str(log_dir), multi_gpu_cfg=None)
        algo.setup()
        algo.attach_checkpoint_metadata(saved, None)
        algo.load(str(checkpoint))
        output.mkdir(parents=True, exist_ok=True)
        algo.export(onnx_file_path=str(target))
        for path in validate_bundle(target):
            print(path)
    except Exception as exc:  # noqa: BLE001 - report, discard a partial bundle, then still shut the app down
        for path in (target, Path(hero_sidecar_path(str(target)))):
            path.unlink(missing_ok=True)
        print(f"export failed: {exc}", file=sys.stderr)
        code = 1
    finally:
        sys.stdout.flush()
        sys.stderr.flush()
        close_simulation_app(simulation_app)
    return code


if __name__ == "__main__":
    raise SystemExit(main())
