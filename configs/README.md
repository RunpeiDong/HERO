# Training configurations

The default `without_delta_anchor` preset implements the paper's controller formulation in Isaac Sim. Presets are exposed in [presets.py](presets.py) and defined in [experiment.py](../hero_isaacsim/config_values/experiment.py).

| Configuration | Actor | Actor / critic observation dimensions | ΔEE feedback (actor input) | Delta anchor |
| --- | --- | --- | --- | --- |
| `without_delta_anchor` (default) | dual (lower / upper body heads, paper) | 675 / 268 | Yes | No |
| `with_delta_anchor` | dual | 950 / 323 | Yes | Yes |
| `without_delta_ee` (ablation) | dual | 585 / 268 | No | No |
| `without_delta_anchor_single` | single whole-body actor | 675 / 268 | Yes | No |
| `with_delta_anchor_single` | single whole-body actor | 950 / 323 | Yes | Yes |
| `without_delta_ee_single` (ablation) | single whole-body actor | 585 / 268 | No | No |

The `*_delta_anchor*` configurations retain HERO's residual end-effector feedback (ΔEE: the palm position and rotation errors `h07`/`h08`, 18 inputs per frame); the `without_delta_ee*` ablations remove it from the actor only, so the policy follows the reference arm joints without seeing its own end-effector error, while the critic, rewards, and all other settings stay those of `without_delta_anchor`. All configurations share the robot, rewards, action contract, and domain randomization. The dual-actor configurations are the paper's architecture: two PPO actors (15 lower-body and 14 upper-body actions) with their own critics. The `_single` configurations train one 29-DoF actor and one critic with the stock PPO on identical observations; they export the same frame-major ONNX layout with a single `actor_obs` input, so the demo and the deployment reader accept both. Delta anchor adds reference-root planar pose, orientation, and height errors across future reference frames (`h20`, `h21`, `h22`), with odometry noise during training.

Shared defaults: 4,096 environments, 20,000 iterations, 500 Hz physics, and a 50 Hz policy.

Run from the repository root:

```bash
python scripts/train.py --motion-dir data/amass
python scripts/train.py --config with_delta_anchor --motion-dir data/amass
python scripts/train.py --config without_delta_anchor_single --motion-dir data/amass
python scripts/train.py --config without_delta_ee --motion-dir data/amass
```

Use `--dry-run` to inspect the resolved configuration, `--checkpoint` to resume matching weights, and `--num-envs`, `--iterations`, or `--seed` to override training settings. When resuming, `--iterations` counts the additional iterations after the checkpoint's iteration. Logs and checkpoints are saved under `runs/`; `--output` changes this directory. TensorBoard logging is enabled by default; `--logger wandb-offline` enables offline W&B logging.

See [data preparation](../docs/data.md) for combining AMASS and generated IK trajectories.

Export a trained policy and its matching `_hero.json` metadata:

```bash
python scripts/export.py \
  --checkpoint /path/to/model_20000.pt \
  --motion-dir data/amass --output checkpoints/my_policy
```

The example ONNX model uses a separate observation layout. See [checkpoint compatibility](../checkpoints/README.md) before deploying custom policies.
