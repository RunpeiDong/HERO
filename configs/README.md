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

### Resuming

A training checkpoint stores the network, optimizer and the state that shapes the training distribution: the adaptive clip sampler's failure table, the average-episode-length tracker, every curriculum term's progress (the penalty scale) and the per-environment height-offset curriculum. All of it is restored on `--checkpoint`, so a resumed run continues the curriculum where the checkpoint left it (with a different `--num-envs` the per-environment height-offset scale restarts at the checkpoint's mean, logged by the environment). Before Isaac Sim starts, `train.py` compares the checkpoint's saved configuration with the selected one. Observation layout and algorithm class must match. The whole motion configuration except per-run plumbing (`motion_dir` / `motion_file`, `source_weights`, `source_weights_from_manifest`, `reset_sampler_on_resume`, `motion_storage_device`, `shared_cache_dir`) is compared -- so the sampler settings (`adaptive_sampler_*`), the clip-end policies (`clip_end_policy_by_source`, `default_clip_end_policy`, `rollover_at_clip_end`), the height-offset curriculum (`h_offset_*`, `h_curriculum_*`), the command sampling (`walk_prob`, `fix_upper_body_prob`, `vel_cmd_*`, `command_resample_time_s`) and the reference settings -- together with the exploration-std clamp (`--std-clamp-max`, saved as `algo.config.hero_std_clamp_max`; the clamp is not inherited from the checkpoint, so adding, removing or changing it is a finding), the curriculum parameters, reward weights and parameters, termination and randomization parameters; every difference is listed and stops the launch. A checkpoint that carries no saved progress for a curriculum term or for the height-offset curriculum (written before they were checkpointed) is a finding too: the term would restart at its initial value (the penalty scale at `initial_scale` 0.1, every environment's height-offset scale at `h_curriculum_init` 0.1), and the finding names that value. `--allow-config-change` continues anyway. A different source mix (`source_weights`) is not a configuration-change finding -- the mix may legitimately change between runs -- but the saved failure table was built for the checkpoint's mix over its clip registry: a changed mix (compared after normalisation over the corpus' source tags, the way the sampler builds its prior), a changed clip count or another set of source tags is reported as a sampler finding, as is a table written under other sampler settings or another sampling rule. Such a table cannot be restored; `--reset-sampler-on-resume` restarts it from zeros, otherwise the launch stops. All findings of one checkpoint are reported in a single message naming every flag it needs. With the flags, the launcher prints the findings (also in its `--dry-run` JSON under `config_changes`, `curriculum_state_reset` and `sampler_reset`; the clip count and source tags it compared under `corpus`) before Isaac Sim starts; the sampler reset and the curriculum restarts are logged again by the environment when it loads the checkpoint. The findings are not written into the run's checkpoints. The random episode-length offsets that desynchronise a fresh start are not applied on resume: they would inflate the restored episode-length tracker and the curricula it drives. The forced all-environment resets at agent construction and at the start of `learn()` are infrastructure, not episode ends: they step neither the episode-length tracker, nor the sampler table, nor the height-offset curriculum. A fresh run therefore starts every environment's height-offset scale at `h_curriculum_init` = 0.10; before this was enforced the two forced resets each applied the short-episode step, so earlier runs (including the ones that produced the release checkpoints) effectively started at 0.08 -- their early height-command statistics differ from a new run's by that offset until an environment's scale reaches a clamp bound.

### Exploration std clamp

`--std-clamp-max FLOAT` bounds the exploration standard deviation of both actor heads from above (the lower-body / upper-body actors start at 0.8 / 0.6 and the clamp is applied after every update and when a checkpoint is loaded). The default is no clamp, the paper recipe: under it the per-joint std grows to about 1.3 over training, so consecutive checkpoints (500 iterations apart) sample noticeably different trajectories and differ on the benchmark while the training reward looks flat. Choose release checkpoints on `hero_bench_v1` (open-loop and closed-loop replan), not on training reward or iteration count. The option applies to the dual-actor configurations; the single-actor PPO has no clamp. The chosen value is saved in the checkpoint's configuration and in the training log (`PPODual knobs: std_clamp_max=...`). It is not inherited on resume: pass it again with `--checkpoint`, otherwise the pre-flight reports the removed clamp as a configuration change (see Resuming) and, once accepted, the loader logs that the std continues unclamped.

See [data preparation](../docs/data.md) for combining AMASS and generated IK trajectories.

Export a trained policy and its matching `_hero.json` metadata:

```bash
python scripts/export.py \
  --checkpoint /path/to/model_20000.pt \
  --motion-dir data/amass --output checkpoints/my_policy
```

The example ONNX model uses a separate observation layout. See [checkpoint compatibility](../checkpoints/README.md) before deploying custom policies.
