# AMASS and generated IK data

The paper trains on **AMASS motions and generated IK reaching trajectories**. The Quick Start accepts AMASS alone so you can run the training pipeline before generating the reaching corpus. This shortcut is not the paper's full data recipe.

The training entry point checks both the corpus manifest and each clip's `source_tag`. A combined AMASS/IK corpus must declare its source weights explicitly.

## Prepare AMASS

1. Obtain AMASS from <https://amass.is.tue.mpg.de/> and follow its dataset terms.
2. Retarget the human motions to the G1 using your retargeting pipeline. This repository does not include the original AMASS data, body-model files, or a human-to-robot retargeter.
3. Save the robot trajectories as NPZ files and run `scripts/prepare_amass.py`.

The retargeted input format is:

| Field | Shape | Convention |
| --- | --- | --- |
| `joint_pos` or `qpos` | `(T, 36)` | Root XYZ, root quaternion WXYZ, 29 G1 joint angles in radians |
| `fps` | Scalar | Positive source sampling frequency |
| `joint_names` | `(29,)`, optional | Joint names used to reorder the 29 actuated joints |

At least two frames are required. Without `joint_names`, joint order must match `DOF_NAMES` in [`hero_isaacsim/constants.py`](../hero_isaacsim/constants.py). Root position is in meters. Raw SMPL/SMPL-H/SMPL-X arrays are not robot joint trajectories and are rejected.

```bash
python scripts/prepare_amass.py \
  --input /path/to/retargeted_amass_g1 \
  --output data/amass \
  --jobs 8
```

Use a new output directory. Preparation resamples to 50 Hz, computes G1 forward kinematics and velocities, and writes the palm-reference channels used by HERO. A completed output includes the converted clips and `CORPUS_MANIFEST.json`, with `source_weights` set to `{"amass": 1.0}`. A failed conversion retains an incomplete-build marker rather than declaring a partial corpus ready.

The loader consumes the standard 32-body motion representation and HERO's end-effector references:

| Field | Shape | Meaning |
| --- | --- | --- |
| `joint_pos` | `(T, 36)` | Root pose and actuated joint positions |
| `joint_vel` | `(T, 35)` | Root linear/angular velocity and joint velocities |
| `body_pos_w` | `(T, 32, 3)` | Body link origins in world coordinates |
| `body_quat_w` | `(T, 32, 4)` | World body orientations, WXYZ |
| `body_lin_vel_w`, `body_ang_vel_w` | `(T, 32, 3)` | World body velocities |
| `ee_pos_pelvis`, `ee_quat_pelvis` | `(T, 2, 3)` / `(T, 2, 4)` | Left/right palm poses in the reference pelvis frame |
| `ee_pos_pelvis_zero_waist`, `ee_quat_pelvis_zero_waist` | `(T, 2, 3)` / `(T, 2, 4)` | Palm references evaluated with zero waist angles |
| `h_ref` | `(T,)` | Reference pelvis height |
| `source_tag`, `parent_id`, `license_class` | Scalars | Source, sequence identity, and data-use metadata |
| `has_object` | Scalar | Must be `False` |

See [`schema.py`](../data_tools/schema.py) for body names and complete validation rules. Stored quaternions use **WXYZ**; runtime Torch tensors use **XYZW**. Do not interchange them.

## Generate the reaching motions

The reaching data used for training must be generated; no IK dataset is shipped. The script below is a small, extensible example of sampling targets and solving whole-body reaching trajectories:

```bash
python scripts/generate_ik_example.py \
  --output data/ik_examples --num-clips 32 --seed 0
```

`--jobs` runs that many IK workers (at most 64); set it to the CPU cores available to the process (inside a container that is the CPU request, not the host core count). The workers keep BLAS and OpenMP single-threaded unless `OMP_NUM_THREADS` or `OPENBLAS_NUM_THREADS` is already set in the shell. Use `--specs-only` with a separate output directory to inspect sampled targets without running the solver. `--profile`, `--seed`, `--num-clips`, and repeatable `--override KEY=VALUE` arguments expose the sampler. Consult [`reach_specs.py`](../data_tools/reach_specs.py) for supported fields before extending positions, orientations, timing, or posture ranges.

This is an **example generator**. It is reasonable to generate more motions or add task-specific profiles; retain the solver's feasibility checks and inspect the resulting motions. Generating more samples does not guarantee that every requested target is reachable.

## Combine AMASS and generated reaching data

The generator writes reach-bank clips (`clips/*.npz`), not training clips: they carry no `source_tag`, body channels, or palm references. Convert them with the lower-level converter, which runs the same forward kinematics as the AMASS preparation and stamps the source tag you choose, then place them next to the converted AMASS clips in a new directory:

```bash
python -m data_tools.npz_convert \
  --in-dir data/ik_examples/clips --out-dir data/combined \
  --source-tag ik_reach_example --license-class research-only --jobs 8
cp data/amass/*.npz data/combined/
```

`--license-class` records a provenance label for the generated clips; the converter's default is `unknown`, so pass `research-only` (the label `prepare_amass.py` stamps on AMASS) or the class that applies to your data. Then write `data/combined/CORPUS_MANIFEST.json` with one non-negative weight per source tag, for example:

```json
{"source_weights": {"amass": 0.5, "ik_reach_example": 0.5}, "source_tags": ["amass", "ik_reach_example"],
 "clip_end_policy_by_source": {"amass": "rollover", "ik_reach_example": "hold"}}
```

`clip_end_policy_by_source` decides what happens when a reference clip ends: `rollover` teleports the robot onto a new clip and continues the episode (mocap default); `hold` ends the episode at the clip's last frame (counted as a timeout, not a failure; the reference is pinned to that frame), so a held clip contributes at most its own length per episode (from the sampled start phase to its last frame), never more (generated reaching clips). The training configuration already applies `hold` to the `ik_reach_example` tag; the manifest entry overrides the defaults for any tag, so keep the tag you pass to `--source-tag` identical in the clips, the manifest, and this map. The training log reports the resulting census as `clip end policy rollover=N hold=M`.

Keep the clips' source and parent identifiers, and preserve parent-sequence separation between training and evaluation. IK data is never automatically added to `data/amass`.

The weights are per-source episode shares, independent of how many clips each source has. The failure-weighted sampler can raise a clip to at most ten times its share and never changes a zero weight, so a source with a handful of generated clips keeps exactly the share you give it.

Launch the mixture intentionally:

```bash
python scripts/train.py \
  --motion-dir /path/to/combined_corpus \
  --allow-mixed-data
```

The flag permits the supplied manifest; it does not retarget, convert, copy, or generate data. The default invocation remains an AMASS-only Quick Start.
