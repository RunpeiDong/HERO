"""MuJoCo benchmark scoring for exported HERO policies (hero_bench_v1).

The package rolls an exported policy (``model.onnx`` + ``model_hero.json``, :mod:`sim2sim.policy_hero_export`) over
the benchmark clips on the training URDF plant (:mod:`sim2sim.plant`), records the ``hero_eval_series_v1`` time series
(:mod:`sim2sim.bench.series`) and reduces it to the benchmark tables:

* :mod:`sim2sim.bench.run` -- the runner (``python -m sim2sim.bench.run``): open loop, or closed loop with HERO's
  replanning + goal adjustment (:mod:`sim2sim.bench.replan`) and the LiDAR-inertial odometry model
  (:mod:`sim2sim.bench.odometry`) behind the root-feedback inputs;
* :mod:`sim2sim.bench.summary` -- per-height / per-layer open-loop tables with the success columns;
* :mod:`sim2sim.bench.closed_loop_summary` -- the closed-loop windows (hold3 / final / tail), C3 and the replanner bookkeeping;
* :mod:`sim2sim.bench.report` -- the side-by-side table (every world-frame number next to its closed-loop neighbour) and the
  results card;
* :mod:`sim2sim.bench.cli` -- ``run(argv)`` / ``summary(argv)`` / ``closed_loop_summary(argv)`` / ``report(argv)`` entry points.

Runtime dependencies: numpy, mujoco, onnxruntime; closed loop adds mink (+ daqp) through ``data_tools.hero_reach_generator``.
"""

__all__ = ["__version__"]
__version__ = "0.3.0"
