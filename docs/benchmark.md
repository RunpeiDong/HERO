# HERO reaching benchmark (hero_bench_v1)

`hero_bench_v1` is the public end-effector reaching benchmark of HERO: 1,298 synthetic whole-body reaching references for the
Unitree G1 + Dex3 model, organised in three tiers, scored in MuJoCo with the tooling in `sim2sim/bench/` (`pip install -e ".[bench]"`).
The paper's 180-target protocol (three table heights x 60 targets, a 3 s hold) is contained byte for byte as the core tier's `paper`
layer (files `h050__hero_bench_v1_*.npz` / `h074__…` / `h088__…`), so every published number is reproduced by scoring the core tier and
filtering on the `h050` / `h074` / `h088` layers.  This file covers the corpus (layout, tiers, how to obtain, verify or rebuild it, licence) and scoring a policy on it.

## The corpus

### Layout

```
hero_bench_v1/
  BENCH_MANIFEST.json      # every clip: file, sha256, stratum, tier, hand, height_label, target_pos_w, target_{yaw,pitch,roll}_deg,
                           # reach_end_frame, hold_frames, hold_end_frame, retract_start_frame, table (null for floor clips), accept_level,
                           # terminal IK residuals, pair_of (re-timed twins), verbatim_from (paper-protocol files); top level: per-layer selection
                           # statistics and yields, excluded ranges, the protocol block and its sha256 (protocol_sha256)
  BENCH_REPORT.md          # layer x height x hand table, yields, IK residuals per layer
  SHA256SUMS  DATA_LICENSE  tiers/{core,extended,stress,all,paper,paper_sub60}.txt
  core/      420 clips + BENCH_MANIFEST.json (core rows only) + LICENSE_paper_verbatim.txt
  extended/  618 clips + BENCH_MANIFEST.json
  stress/    260 clips + BENCH_MANIFEST.json
```

Clip files are `<stratum>__<clip_id>.npz` (generated layers: `<stratum>__hero_bench_v1_<stratum>_NNNNNN.npz`); the paper-protocol files
keep their original names (`h050__hero_bench_v1_000001.npz`, ...).  The
`source_tag` of a clip is its layer.  Every clip is a 50 Hz reference: 0.3 s settle at the rest pose, a 1.5-6 s reach, a 3 s hold
(6 s in `hold6`), and for the `retract` layer a return along the approach path to the rest pose.

### Tiers and layers

| tier | clips | layers | what it adds over the paper protocol |
|---|---|---|---|
| core | 420 | `h050`/`h074`/`h088` (the paper protocol, 180), `far060`, `cross_mid`, `mid_h062_h080`, `orient_core` | reach to 0.60 m, across the body mid-line, two intermediate table heights (0.62 / 0.80 m), top-down and tilted grasps |
| extended | 618 | `low_h030_h040`, `floor_pick`, `high_h100_h110`, `wide_lat`, `orient_ext`, `hover_above`, `slow_x2`, `hold6`, `retract`, `close` | deep squats (0.30 / 0.40 m tables), floor pick-up without a table, shelves at 1.00 / 1.10 m, wide lateral targets, palm-down / palm-up / fanned grasps, hovering 15-30 cm above the table, half-speed reaches, 6 s holds, retracting to the rest pose, targets at the table edge |
| stress | 260 | `far070_bow`, `low_h025`, `high_h115_120`, `fast_x0p75`, `recov_pool` | bowing reaches to 0.70 m (provisional: 33 % IK yield), a 0.25 m table, shelves at 1.15 / 1.20 m, 1.33x speed, and a pool of clips whose IK terminal residual is 1.5-3 cm |

The full per-layer table (quota, candidates, IK yield, residuals) is `BENCH_REPORT.md`; the design parameters of every layer are
the `hero_bench_v1_*` profiles in `data_tools/reach_specs.py` (`python scripts/hero_bench.py plan` prints the plan).  Re-timed layers
are deterministic re-samplings of the 60-clip `paper_sub60` subset of the paper protocol (`pair_of` names the twin), all velocities recomputed.

### Obtaining the corpus

The frozen corpus is distributed as a versioned archive (hero_bench_v1_corpus_33542601.tar.gz, 516 MB, sha256 33542601ed456ac606f8177ab75892d2e616585ec562f01976cb6af238c9c5d0; hosted at https://huggingface.co/datasets/RunpeiDong/hero_bench).  Download, check the
archive hash, extract and verify every file:

```sh
python scripts/hero_bench.py fetch --url https://huggingface.co/datasets/RunpeiDong/hero_bench/resolve/main/hero_bench_v1_corpus_33542601.tar.gz --sha256 33542601ed456ac606f8177ab75892d2e616585ec562f01976cb6af238c9c5d0 --out data/hero_bench_v1
python scripts/hero_bench.py verify --dir data/hero_bench_v1        # re-hashes every clip against BENCH_MANIFEST.json and SHA256SUMS
```

`verify` prints how many files are byte-identical to the frozen manifest and whether the protocol block hash matches; a scored result
is only comparable to published numbers when it reports all files identical and the same `protocol_sha256`.

`fetch` requires a new or empty output directory. For an existing download, use `verify` instead.

### Rebuilding the corpus

The corpus is a deterministic function of the plan in `data_tools/reach_specs.py` (seeds, quotas, acceptance levels), the IK generator
(`data_tools/hero_reach_generator.py`) and the paper-protocol files (the 180 `h050` / `h074` / `h088` clips next to their own
`BENCH_MANIFEST.json`, as written by `data_tools/build_reach_bench.py`; `--paper-dir`, env `HERO_BENCH_PAPER_DIR`):

```sh
python scripts/hero_bench.py build --plan default --bank-root banks/ --paper-dir data/hero_bench_paper --out data/hero_bench_v1_rebuilt \
    --generate --jobs 32 --public
python scripts/hero_bench.py verify --dir data/hero_bench_v1_rebuilt
```

Sampling is deterministic, but the IK solutions depend on the MuJoCo / mink / daqp / BLAS builds, so a rebuild is
*protocol-equivalent* (same targets, frames and acceptance levels) and usually not byte-identical; `verify` reports which.  Layers
whose quota is not met by the planned candidates are topped up with further seed batches (`fill_until_quota`, up to six batches);
the manifest records the yield and batches of every layer.  The `--public` flag scrubs host names and local paths from the manifest.

### Protocol block

`BENCH_MANIFEST.json["protocol"]` fixes everything a score depends on: the acceptance levels of the IK candidates, the plant
(training URDF `g1_29dof_dex3fixed_hero.urdf`, Dex3 hand, audited sole box, 1 kHz physics / 50 Hz policy, no table geometry: the palm
may pass through the table unpenalised, as in the paper), the odometry model and its seed rule, the common fail-free rule
(`fall`, `anchor_xy`; `fall_low` as the low-posture diagnostic), the closed-loop replan constants, and the success thresholds:

| criterion | loop | window | condition |
|---|---|---|---|
| S7.5 | open | hold | fail-free, active-hand world palm error <= 7.5 cm, world rotation error <= 15 deg |
| S5 | open | hold | fail-free, <= 5 cm, <= 10 deg |
| C3 | closed (replan + adjust) | last 1 s | fail-free, <= 3 cm, <= 15 deg |

All rates are reported with Wilson 95 % intervals over every clip of the group; the headline metric is the active (reaching) hand's
world-frame palm error, with the both-hands mean as a secondary column.  `protocol_sha256` is the hash of this block.

The odometry noise seed of a clip is derived from its file name (`seed_rule`: `--odom-seed` + crc32 of the file name), so the re-timed
twins (`slow_x2__…`, `hold6__…`, `fast_x0p75__…`) see a different noise realisation from their `pair_of` v1 clip, and renaming a clip
changes its score.  `recov_pool` rows (IK terminal residual 1.5-3 cm) are scored against the reference palm in the open loop and against
the nominal goal (`target_pos_w` / orientation) in the closed loop; the IK residual column shows the gap between the two targets.

### Licence

The clips, manifest rows and lists are released under the Apache License 2.0 (`DATA_LICENSE`; every row carries
`license_class = "apache"`); no human motion-capture data is included.  The paper-protocol files reproduced under `core/` are covered by
`core/LICENSE_paper_verbatim.txt`.  The code is licensed separately (`LICENSE`).

## Scoring a policy

Everything below runs from the repository root with the core dependencies installed (`pip install -e ".[bench]"` names the
scoring stack explicitly: numpy, mujoco, mink, daqp, onnxruntime).  The commands take the example export
`checkpoints/example/model.onnx` + `model_hero.json`; substitute your own `model_XXXXX.onnx` + `model_XXXXX_hero.json` pair
(the sidecar written by `scripts/export.py`).  `$BENCH` is a built corpus directory (clips + `BENCH_MANIFEST.json`), `$OUT` an output
directory.

### 0. Get the corpus

`python scripts/hero_bench.py fetch ... && python scripts/hero_bench.py verify --dir data/hero_bench_v1` (see above), or build it.  The
one-command path over the tiers is

```sh
python scripts/hero_bench.py run --corpus data/hero_bench_v1 --onnx-dir checkpoints/example --sidecar checkpoints/example/model_hero.json \
    --out results/example --tiers core            # all = core,extended,stress; open loop + closed loop per tier
python scripts/hero_bench.py report --corpus data/hero_bench_v1 --results results/example --card
```

which runs steps 1-4 below per tier (closed-loop horizon = max(14 s, longest clip + 6 s)) and concatenates the tier reports into
`results/example/RESULTS.md`.  The steps are spelled out for a single directory of clips.

Completed runs are reused only when the policy, corpus, parameters and runtime match the saved `RUN_CONFIG.json`.
Use a new output directory when any of these change. Runs with clip errors return a nonzero exit code; tier reports refuse
errored, missing or duplicate clip results.

The scorer only needs the clip files `<stratum>__<clip_id>.npz` and the `BENCH_MANIFEST.json` next to them (per clip: `file`,
`hand`, `target_pos_w`, `target_{yaw,pitch,roll}_deg`, `reach_end_frame`, `hold_frames` / `hold_end_frame`, `n_frames`, `fps`, `table`
and, for hero_bench_v1 rows, `stratum` / `tier`; a top-level `protocol.success` block overrides the success thresholds).  hero_bench_v1
rows are matched to clips by the exact file stem (the re-timed twins share a `clip_id`, so no partial match is attempted; a clip without a
row is an error); only legacy manifests without a `stratum` field fall back to matching the `clip_id`.  To score only the core tier by
hand, point every step below at that tier directory and its own manifest:

```sh
python -m sim2sim.bench.run --onnx-dir checkpoints/example --sidecar checkpoints/example/model_hero.json \
    --motion-dir data/hero_bench_v1/core --bench-manifest data/hero_bench_v1/core/BENCH_MANIFEST.json --out $OUT/open --jobs 8 --quiet
```

### 1. Open loop

The robot starts at frame 0 of each clip and tracks the clip as its reference; the world-frame palm error accumulates every drift
of the base.

```sh
python -m sim2sim.bench.run --onnx-dir checkpoints/example --sidecar checkpoints/example/model_hero.json \
    --motion-dir $BENCH --out $OUT/open --jobs 8 --quiet \
    --fail-causes fall,anchor_xy --odom so
```

* `--fail-causes fall,anchor_xy` -- the causes that count as failure for a policy without a pelvis-pose reference (every cause is still
  evaluated and reported).  Add the alias `fall_low` for the hero_bench_v1 low-posture layers (robot pelvis more than 0.20 m below a
  reference pelvis that sits at or below 0.50 m counts as a fall).
* `--odom so` -- the root-feedback inputs of the policy (h20-h22) are built from the LiDAR-inertial odometry model
  (10 Hz corrections, 30 ms processing latency, 20 ms output age, 0.3 % of distance drift; `sim2sim/bench/odometry.py`) instead of the
  exact simulator root pose; `--odom-seed` sets the base seed (each clip derives its own).  Exports without those inputs ignore the flag
  (the estimator only records its error in the `odom_*` columns).
* Plant: the training URDF `assets/robots/g1_modified/g1_29dof_dex3fixed_hero.urdf` (Dex3 hand, self-collisions off), the audited sole
  box (`--foot-collision sonic_box`), the `hero` physics profile (1 kHz physics, 20 sub-steps = 50 Hz policy), frame-0 reset with zero
  velocities.  All are the defaults.

Output: `$OUT/open/series.npz` (schema `hero_eval_series_v1`: per clip and per control step every metric column, plus the provenance in
`meta_json`), `summary.json` / `summary.md` (horizon averages per source tag).

### 2. Closed loop (replanning + goal adjustment)

HERO's deployment protocol: at the original plan's `reach_end` frame and every 3 s afterwards the remaining reach is re-planned from the
live robot state with the benchmark IK (`data_tools/hero_reach_generator.py`), the commanded goal is nudged against the measured
world-frame palm error (`g <- g - 0.6 e`, 1 cm per update, frozen below 1.75 cm), the base is held where it is (`--replan-base current`)
and the reference the policy sees is cross-faded over 0.3 s at every swap.  The clips are padded so the loop has time to converge.

```sh
python -m sim2sim.bench.run --onnx-dir checkpoints/example --sidecar checkpoints/example/model_hero.json \
    --motion-dir $BENCH --out $OUT/closed --jobs 8 --quiet \
    --fail-causes fall,anchor_xy --odom so \
    --pad-s 4 --horizon-s 14 --replan --bench-manifest $BENCH/BENCH_MANIFEST.json \
    --replan-first reach_end --replan-period-s 3.0 --replan-base current --goal-adjust --replan-blend-s 0.3
```

Metrics are still scored against the ORIGINAL clip on its own timeline (the goal is fixed in the world); the replanner's bookkeeping
(events, goal offsets, IK residuals, per-plan timings) is stored under `meta_json["replan_log"]`.  Clips of the hero_bench_v1 retract
layer carry their own retract segment: replanning stops at their `hold_end_frame` and the controller is handed back to the clip.

### 3. Summaries

```sh
python -m sim2sim.bench.summary --series $OUT/open/series.npz --bench-manifest $BENCH/BENCH_MANIFEST.json \
    --out $OUT/open/summary --group-by stratum
python -m sim2sim.bench.closed_loop_summary --series "replan|$OUT/closed/series.npz" --manifest $BENCH/BENCH_MANIFEST.json \
    --out $OUT/closed/summary --csv $OUT/closed/summary/flat.csv
```

* `summary` scores the HOLD window of every clip (`[reach_end_frame, hold_end_frame)`; the paper's numbers): per height / layer / tier /
  hand the mean +- std and p50 / p80 / p90 of the per-clip hold means, the fail-free fraction, the success rates `S7.5` (fail-free and
  hold-mean active-hand world error <= 7.5 cm and world rotation error <= 15 deg) and `S5` (<= 5 cm, <= 10 deg) with Wilson 95 %
  intervals (denominator = every clip of the group), the CDF points at 2.5 / 5 / 10 cm and the layer's IK residual.  Writes
  `bench_summary.json` / `.md` / `.csv`.
* `closed_loop_summary` scores three windows -- `hold3` (the row's own hold window), `final` (the last second before the padded clip
  end, where the loop has converged) and `tail` -- plus `C3` (fail-free and final-window active-hand error <= 3 cm, <= 15 deg), the
  `stayed` fraction (HERO's own stop rule) and the replanner / jerk tables.  For rows with their own retract segment (the `retract`
  layer, `retract_start_frame`) the replanner hands the controller back to the clip at `retract_start_frame`, so `final` is the last
  second BEFORE that frame and `tail` ends there; the rest pose is scored separately as the `rest` window (the last second of the
  padded clip, `rest_pos_err_cm` / `rest_rot_err_deg`), never mixed into `final` / `C3`.  Writes `closed_loop_summary.json` / `.md`.

The thresholds come from the manifest's `protocol.success` block when present, else the defaults above (recorded under
`success_protocol` in both files).

### 4. Report and results card

```sh
python -m sim2sim.bench.report --open-loop $OUT/open/summary/bench_summary.json \
    --closed-loop $OUT/closed/summary/closed_loop_summary.json \
    --manifest $BENCH/BENCH_MANIFEST.json --out $OUT/report --card
```

`hero_bench_report.md` / `.json` put both runs side by side per group (`all`, every tier, every layer, every layer x height x hand cell)
with the fixed columns (plus, when a group has retract rows, a trailing `global closed-loop rest-1s (retract rows)` column)

```
global open-loop (hold) | global closed-loop replan+adjust (hold3) | global closed-loop final-1s | local open-loop |
fail-free open / closed | S7.5 | S5 | C3 (CI) | stayed | IK residual (cm)
```

**Cell rule.**  A world-frame (global) number is never printed alone: every open-loop `global` cell sits next to its closed-loop
replan + adjust neighbour.  A group the closed-loop harness cannot run (no closed-loop series, or a layer the replanner does not
support) prints `n/a (harness)` in every closed-loop cell -- never an empty cell; the writer refuses to emit a table that violates
the rule.  `--card` additionally writes `RESULTS_CARD.md` + `results_card.json`: the protocol hash (sha256 of the manifest's
`protocol` block), the manifest sha256, the plant / policy sha256 (`--plant-sha` / `--policy-sha`; the policy hash is computed from
the ONNX path when it exists), the odometry mode and seed, the fail-free rule the run used (`termination`: `fail_causes`,
`fall_low_ref_margin_m` and the rest of the runner's termination config, from the series meta), the tiers run, and the tier / layer
tables.  Either summary may be omitted (`--open-loop` / `--closed-loop` are each optional; the missing side prints `--` /
`n/a (harness)`).  The card reports the evaluated policy only (`--external-labels` names comparison rows that must not enter a card).

### Programmatic use

`sim2sim/bench/cli.py` exposes `run(argv)`, `summary(argv)`, `closed_loop_summary(argv)` and `report(argv)` (each takes the module's
flags as a list and returns the exit code) for a wrapper script such as `scripts/hero_bench.py` that owns the per-tier protocol defaults.

### Checking the scorer

`python -m pytest tests -q` runs the unit tests of every scoring layer; with a fetched or built corpus at `data/hero_bench_v1` (or
`HERO_BENCH_V1_DIR=...`; its core tier is used) it also runs an end-to-end smoke (two paper-protocol clips open loop + closed loop, both
summaries, report and card), and with the paper-protocol set at `data/hero_bench_paper` (or `HERO_BENCH_PAPER_DIR=...`) the real
180-file verbatim import of the builder.
Optional result regressions use `results/benchmark` (or `HERO_BENCH_RESULTS_DIR=...`) with
`open_loop/{series.npz,summary/bench_summary.json}` and `closed_loop/{series.npz,summary/closed_loop_summary.json}`.
