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

- **Resume state.** `envs/wbt/wbt_manager.py` persists every stateful curriculum term (the penalty scale) under `env_state["curriculum_terms"]` next to the episode-length tracker and the sampler table, restores it on load, refuses state for terms the live task does not run, and warns when a checkpoint carries no state for a live term (it restarts at its initial value). `managers/command/terms/wbt.py` factors the sampler's `sampling_policy()` / `sampling_policy_mismatches()` so subclasses can add entries (HERO adds its relative clip cap and the name of its composition rule) and a table written under another rule is refused on load. `agents/ppo/ppo.py` applies the random initial episode-length offsets (`init_at_random_ep_len`) to fresh starts only (`_randomize_initial_episode_lengths`): on a resume they would inflate the restored episode-length tracker and the curricula it drives. `config_types/algo.py` adds `PPODualConfig.hero_std_clamp_max` (upper exploration-std clamp, `None` = unclamped) so the launcher option is part of the saved configuration. The forced all-environment resets at agent construction (`PPO.__init__`) and at `learn()` entry are not episode ends: `envs/wbt/wbt_manager.py` already kept the sampler table and the episode-length tracker out of them, and HERO's `HeroTrackingManager.reset_all` (`hero_isaacsim/envs/hero_tracking_manager.py`) now flags the motion command so the per-environment height-offset curriculum is not stepped either -- a fresh run starts at `h_curriculum_init` = 0.10, where the two forced resets previously stepped it to 0.08 (documented in `configs/README.md`, Resuming).

HERO uses flat terrain and the G1/Dex3 robot.

Record the backend revision and rerun configuration, export-parity, and Isaac Sim integration checks when updating this dependency.
