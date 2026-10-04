# HERO checkpoints

The browser demo uses `example/model.onnx` and `example/model_hero.json`. Keep them together. The model predicts 29 body-joint actions; the demo controls the fingers separately.

The example model uses delta anchor and a five-frame, 1,000-dimensional actor input. Its metadata defines the observation layout.

| Use | Actor input | Matching artifact |
| --- | --- | --- |
| `without_delta_anchor` (default) | 675 | Train and export this configuration |
| `with_delta_anchor` | 950 | Train and export this configuration |
| `without_delta_anchor_single` / `with_delta_anchor_single` | 675 / 950 | Train and export this configuration |
| `without_delta_ee` / `without_delta_ee_single` (ablation: actor without ΔEE feedback) | 585 | Train and export this configuration |
| Example ONNX model | 1,000 | `model.onnx` + `model_hero.json` |

Dual-actor exports have two ONNX inputs (`actor_obs_lower_body`, `actor_obs_upper_body`) that are fed the same vector; `_single` exports have one `actor_obs` input. The metadata lists the inputs.

The example model is for inference. To resume training, use a `.pt` checkpoint matching your configuration. The dual-actor configurations clamp the exploration std at 0.5 by default; a `.pt` checkpoint trained without the clamp (the paper-faithful recipe, or one written before the setting existed) resumes with a pre-flight finding on `algo.config.hero_std_clamp_max` unless you pass the same setting (`--std-clamp-max none`) or `--allow-config-change`. See [Exploration std clamp](../configs/README.md#exploration-std-clamp).

Export your checkpoint with `scripts/export.py` in the Isaac Lab environment. It writes the ONNX model and matching `_hero.json` metadata. See [training configurations](../configs/README.md).

The browser exporter verifies the example model's hash. Use `--allow-custom-policy` when building a demo with your own export.
