# HoloSoma integration changes

This directory contains the HoloSoma backend used by HERO. Keep its upstream license, notice, and third-party attributions when redistributing it. The source snapshot is recorded in [VENDOR.md](VENDOR.md).

HERO-specific algorithms and task terms live in `hero_isaacsim/`. Backend changes support the following integration points:

- **Configuration registration.** The experiment registry imports HERO configurations. Algorithm configuration types support separate upper/lower actors and critics.
- **Observation history and export.** HERO exports frame-major observation history with an explicit permutation from the backend's training layout. Exported policies do not contain the motion corpus.
- **Simulation and device initialization.** Per-process USD conversion directories and explicit GPU/rank mapping avoid conversion collisions. Asset routing supports the robot geometry selected by the task configuration.
- **Simulator state conventions.** Root reset velocities distinguish link-origin and center-of-mass velocities. Rigid-body state access exposes the frame needed by the task; center-of-mass randomization refreshes the relevant offsets.
- **Episode boundaries.** Final observations can be computed for selected terminating environments before reset, with the motion reference advanced consistently for value bootstrapping. Motion sources retain their file identities through clip loading.
- **Logging and checkpoints.** Grouped reward accounting, pooled numerator/denominator metrics, run-relative progress, and atomic checkpoint writes support training and resume.
- **Physics compatibility.** Collision-profile routing, solver settings, fixed-joint handling, and material application preserve the selected robot simulation contract.

HERO uses flat terrain and the G1/Dex3 robot.

Record the backend revision and rerun configuration, export-parity, and Isaac Sim integration checks when updating this dependency.
