"""One clip = one deterministic rollout (no reset on failure); mirrors the Isaac Sim fixed-horizon harness timing.

Step order per policy step ``k`` (1-based), mirroring the training environment's ``step`` + the fixed-horizon evaluator:

1. observation at reference frame ``t_k`` from the current robot state (post-physics of step k-1) and the last action;
2. policy -> raw action ``a_k`` (stored as ``last_actions`` for the next observation);
3. ``q_target`` (residual contract: ``q_default + scale * clip(a_k) + (q_ref(t_k) - q_default)`` on the residual joints) held for one
   control step (20 x 1 ms sub-steps, PD recomputed per sub-step);
4. metrics + would-terminate causes: robot after the physics step vs reference frame ``t_k`` (the frame the policy saw);
   re-anchored reference bodies use the robot anchor pose from BEFORE the physics step (the training command's timing);
5. advance ``t_{k+1} = min(t_k + 1, T - 1)`` (hold the last frame at the clip end).

Reset: root pose + joints of frame 0, velocities zero (``reset_velocities="ref"`` uses the clip's frame-0 velocities like the training
environment), histories cleared.  Then, like the training ``reset_all``, one control step with the controller's *reset target* is executed
(``HeroExportPolicy.reset_target_for``: zero action -> ``q_default`` + the arm residual of the lookahead frame) and the reference advances to
frame 1 before the first policy call -- the first recorded step is therefore frame 1 (``valid_until = min(horizon, T - 1)``).

Controllers.  The loop only needs an object with ``reset()``, ``reset_target(q_ref0) -> q_target`` and ``control(ref, t, state) ->
q_target`` (:class:`sim2sim.state.RobotState` = root pose / velocities / joints / palm points read from the plant before the step).
:class:`sim2sim.policy_hero_export.HeroExportPolicy` implements the protocol.  Optional protocol extensions the loop honours when present:
``reset_target_for(ref)`` (the controller picks the reference frame of the reset step itself), ``clip_object_effective(ref)`` /
``object_rule_decision(ref)`` / ``object_obs_summary()`` (bookkeeping of exports with object inputs; the benchmark plant has no object, so
those inputs are always zero).

Odometry (``RolloutConfig.odometry`` = a :class:`sim2sim.bench.odometry.LidarInertialOdometryConfig`, or a controller with ``odom_source ==
"so"``): the estimator is built on the clip's plant, initialised at the TRUE reset pose (frame 0, before the hold step) and driven ONCE per
control step right after :func:`sim2sim.state.read_state` and BEFORE the controller observes; its estimate travels in ``RobotState.odom`` so
``HeroExportPolicy(odom_source="so")`` builds the root-feedback terms from it instead of the exact root pose.  The estimate's error vs truth of
every step is recorded in the ``odom_*`` metric columns (:data:`sim2sim.bench.metrics.ODOM_METRIC_KEYS`; the step's column = the estimate the
policy saw) and summarised in ``ClipResult.extra["odometry"]``.  One estimator per clip: its noise seed is ``clip_odometry_seed(cfg.seed,
ref.name)`` = ``base + crc32(clip name) % 2**31`` so every clip of a bench draws its own noise realisation, reproducibly per clip.  Without an
estimator nothing changes (no odom columns).

Jerk columns (:data:`JERK_METRIC_KEYS`; recorded when ``RolloutConfig.record_jerk`` is set or a ``replanner`` is given -- the closed loop is what
they diagnose; NEW keys only, a plain rollout's key set / every pre-existing column is byte-identical): ``arm_target_delta_rad`` = mean over the
14 arm dofs of ``|q_target_k - q_target_{k-1}|`` (the PD target the controller sent this step vs the previous one -- the reset hold target for
k = 1), ``arm_accel_rad_s2`` = mean over the arm dofs of ``|dof_vel_k - dof_vel_{k-1}| / dt`` (finite-difference acceleration of the MEASURED arm
joint velocities read before the step) and ``ref_palm_speed_cm_s`` = mean over both hands of ``|palm_ref(k) - palm_ref(k-1)| / dt`` of the
reference the controller actually followed (the live one under replanning).  ``ClipResult.extra["jerk"]`` (:func:`jerk_summary`) holds their
means over the valid steps and, under replanning, over the :data:`POST_REPLAN_WINDOW_S` after every replan step.
"""

from __future__ import annotations

import zlib
from dataclasses import dataclass, field, replace
from typing import Any, Mapping, Protocol, Sequence, runtime_checkable

import numpy as np

from hero_isaacsim import constants as HC
from sim2sim.bench.metrics import METRIC_KEYS, ODOM_METRIC_KEYS, compute_metrics, odometry_metrics
from sim2sim.bench.odometry import LidarInertialOdometry, LidarInertialOdometryConfig, build_odometry, config_source, default_odometry_config
from sim2sim.bench.terminations import CAUSE_ORDER, TerminationConfig, check_terminations, reanchor_body_pos
from sim2sim.plant import MujocoPlant
from sim2sim.reference import ClipReference
from sim2sim.state import RobotState, read_state

#: per-step arm command discontinuity / measured arm acceleration / live-reference palm speed (module docstring); ``RolloutConfig.record_jerk`` / replanner
JERK_METRIC_KEYS: tuple[str, ...] = ("arm_target_delta_rad", "arm_accel_rad_s2", "ref_palm_speed_cm_s")
#: window after every replan step over which ``extra["jerk"]`` / the closed-loop summary report the post-event means
POST_REPLAN_WINDOW_S: float = 0.3
_ARM_IDX = np.asarray(HC.ARM_DOF_IDX, dtype=np.int64)


@dataclass
class RolloutConfig:
    horizon_steps: int = 500
    reset_velocities: str = "zero"  # "zero" | "ref"
    termination: TerminationConfig = field(default_factory=TerminationConfig)
    #: odometry estimator config driven once per control step before the controller observes (:class:`LidarInertialOdometryConfig`); None = no
    #: estimator UNLESS the controller reports ``odom_source == "so"`` (then the ``so`` preset is used).  Adds the ``odom_*`` metric columns and
    #: ``extra["odometry"]``; a controller with ``odom_source == "truth"`` ignores the estimate (metrics only).  ``seed`` is the BASE seed: the
    #: clip's estimator runs with :func:`clip_odometry_seed`.
    odometry: LidarInertialOdometryConfig | None = None
    #: record the :data:`JERK_METRIC_KEYS` columns + ``extra["jerk"]`` (module docstring); forced on when ``rollout_clip`` gets a ``replanner``.
    record_jerk: bool = False


@dataclass
class ClipResult:
    clip_name: str
    source_tag: str
    clip_len_steps: int
    valid_until: int
    first_fail_step: int
    first_fail_cause: str
    first_fire: dict[str, int]
    metrics: dict[str, np.ndarray]  # (horizon_steps,) float32, NaN beyond valid_until
    extra: dict[str, Any] = field(default_factory=dict)


@runtime_checkable
class Controller(Protocol):
    def reset(self) -> None: ...

    def reset_target(self, q_ref0: np.ndarray) -> np.ndarray: ...

    def control(self, ref: ClipReference, t: int, st: RobotState) -> np.ndarray: ...


def as_controller(policy: Any) -> Controller:
    """Type-check the controller protocol (``reset`` / ``reset_target`` / ``control``)."""
    if isinstance(policy, Controller):
        return policy
    raise TypeError(f"{type(policy).__name__} is not a rollout Controller (reset / reset_target / control)")


def rollout_clip(
    plant: MujocoPlant,
    policy: Any,
    ref: ClipReference,
    cfg: RolloutConfig,
    *,
    replanner: Any | None = None,
) -> ClipResult:
    """``replanner`` (:class:`sim2sim.bench.replan.BenchReplanner` or anything with ``reset()`` / ``step(k, state, plant) -> ref | None``
    / ``summary()``): closed-loop replanning.  When ``step`` returns a new :class:`ClipReference` the CONTROLLER switches to
    it at its frame 0 (the reference then advances one frame per control step as usual) while metrics, re-anchoring and
    would-terminate checks keep using the ORIGINAL ``ref`` on the original timeline (world-fixed goal)."""
    T_h = int(cfg.horizon_steps)
    K = int(min(T_h, ref.T - 1))
    ctrl = as_controller(policy)
    ctrl.reset()
    if replanner is not None:
        replanner.reset()
    # ---- odometry (module docstring): estimator on THIS plant; a "so" controller without a config gets the preset; the noise seed is derived
    # per clip (one estimator per clip -> the bare base seed would repeat one noise realisation on every clip) ----
    odom_source = str(getattr(ctrl, "odom_source", "truth"))
    odom_cfg_base = cfg.odometry if cfg.odometry is not None else default_odometry_config(odom_source)
    if odom_cfg_base is not None and odom_source != "truth" and config_source(odom_cfg_base) != odom_source:
        raise ValueError(f"controller odom_source={odom_source!r} but RolloutConfig.odometry is a {type(odom_cfg_base).__name__} ({config_source(odom_cfg_base)!r} estimator)")
    odom_cfg = replace(odom_cfg_base, seed=clip_odometry_seed(odom_cfg_base.seed, ref.name)) if odom_cfg_base is not None else None
    odom = build_odometry(plant, odom_cfg) if odom_cfg is not None else None
    term = cfg.termination
    tracked = term.tracked_bodies
    tracked_ids = np.asarray([plant.body_id[n] for n in tracked], dtype=np.int64)

    # ---- reset at frame 0 ------------------------------------------------------------------------
    if cfg.reset_velocities not in ("zero", "ref"):
        raise ValueError(f"reset_velocities must be 'zero' | 'ref', got {cfg.reset_velocities!r}")
    if cfg.reset_velocities == "ref":
        plant.reset(ref.root_pos_w[0], ref.root_quat_w[0], ref.joint_pos[0], root_lin_vel_w=ref.root_lin_vel_w[0], root_ang_vel_w=ref.root_ang_vel_w[0], dof_vel=ref.joint_vel[0])
    else:
        plant.reset(ref.root_pos_w[0], ref.root_quat_w[0], ref.joint_pos[0])
    if odom is not None:
        odom.reset_from_plant(plant)  # t0 = the reset pose (the clip is aligned to the robot here); the hold step is the first estimated interval
    # training ``reset_all``: one control step with the reset target, then the command advances to frame 1 (controllers with
    # ``reset_target_for`` pick the reference frame of that step themselves: HERO's lookahead frame)
    record_jerk = bool(cfg.record_jerk) or replanner is not None
    hold_target = ctrl.reset_target_for(ref) if callable(getattr(ctrl, "reset_target_for", None)) else ctrl.reset_target(ref.joint_pos[0])
    prev_dof_vel = plant.dof_vel.copy() if record_jerk else None  # jerk columns: the reset velocities, the hold target and the frame-0 palm are step 0
    plant.step(hold_target)
    prev_target = np.array(hold_target, dtype=np.float64, copy=True) if record_jerk else None
    prev_ref_palm = ref.palm_pose_w(0)[0] if record_jerk else None
    dt_ctrl = float(plant.control_dt)
    t = ref.clamp(1)
    ref_live, t_live = ref, t
    replan_steps: list[int] = []

    metrics = {k: np.full(T_h, np.nan, dtype=np.float32) for k in METRIC_KEYS}
    if record_jerk:
        metrics.update({k: np.full(T_h, np.nan, dtype=np.float32) for k in JERK_METRIC_KEYS})
    if odom is not None:
        metrics.update({k: np.full(T_h, np.nan, dtype=np.float32) for k in ODOM_METRIC_KEYS})
    first_fire = {c: -1 for c in CAUSE_ORDER}
    for k in range(1, K + 1):
        # ---- observation at frame t -> action -> target -------------------------------------------
        st = read_state(plant)
        odom_vals: dict[str, float] | None = None
        if odom is not None:  # the SAME instant as ``st``; the estimate is what the controller may observe
            st.odom = odom.step_from_plant(plant)
            odom_vals = odometry_metrics(odom.error_vs_truth(st.root_pos, st.root_quat, st.root_lin_vel_w), st.odom)
        if replanner is not None:
            new_ref = replanner.step(k, st, plant)
            if new_ref is not None:
                ref_live, t_live = new_ref, 0
                replan_steps.append(k)
        q_target = ctrl.control(ref_live, t_live, st)
        # ---- jerk columns (module docstring): command discontinuity, measured arm acceleration, live-reference palm speed ----
        jerk_vals: dict[str, float] | None = None
        if record_jerk:
            ref_palm_live = ref_live.palm_pose_w(t_live)[0]
            q_target_arr = np.asarray(q_target, dtype=np.float64)
            jerk_vals = {
                "arm_target_delta_rad": float(np.abs(q_target_arr[_ARM_IDX] - prev_target[_ARM_IDX]).mean()),
                "arm_accel_rad_s2": float(np.abs(st.dof_vel[_ARM_IDX] - prev_dof_vel[_ARM_IDX]).mean() / dt_ctrl),
                "ref_palm_speed_cm_s": float(np.linalg.norm(ref_palm_live - prev_ref_palm, axis=-1).mean() * 100.0 / dt_ctrl),
            }
            prev_target, prev_dof_vel, prev_ref_palm = q_target_arr.copy(), st.dof_vel.copy(), ref_palm_live
        # ---- re-anchored reference bodies with the PRE-physics robot anchor (the training command's timing) -----
        rel = reanchor_body_pos(ref.root_pos_w[t], ref.root_quat_w[t], st.root_pos, st.root_quat, ref.body_pos_by_name(t, tracked))
        rel_by_name = {n: rel[i] for i, n in enumerate(tracked)}
        # ---- physics ----------------------------------------------------------------------------
        plant.step(q_target)
        # ---- metrics + would-terminate vs frame t ----------------------------------------------------
        root_pos, root_quat = plant.root_pos, plant.root_quat
        palm_p, palm_q = plant.palm_pose()
        # full-body keypoint columns: the robot's 32 bodies in HOLOSOMA_BODY_NAMES_32 order, xyzw (the plant converts MuJoCo's wxyz xquat;
        # virtual foot contact points synthesised) vs the ORIGINAL clip's bodies at frame t (world frame, the clip as loaded = placed where the
        # episode started, no re-anchoring).  Read exactly like the palm columns above: MuJoCo's derived body poses, one physics sub-step behind
        # the integrated qpos the anchor columns read (MujocoPlant.step refreshes no kinematics)
        kp_pos_w, kp_quat_w = plant.canonical_body_poses()
        ref_p_local, ref_q_local = ref.palm_pose_own_pelvis(t)
        ref_palm_w, ref_palm_q_w = ref.palm_pose_w(t)
        m = compute_metrics(
            robot_root_pos=root_pos,
            robot_root_quat=root_quat,
            robot_palm_pos_w=palm_p,
            robot_palm_quat_w=palm_q,
            robot_dof_pos=plant.dof_pos,
            ref_root_pos=ref.root_pos_w[t],
            ref_palm_pos_local=ref_p_local,
            ref_palm_quat_local=ref_q_local,
            ref_palm_pos_w=ref_palm_w,
            ref_dof_pos=ref.joint_pos[t],
            ref_palm_quat_w=ref_palm_q_w,
            ref_root_quat=ref.root_quat_w[t],
            robot_body_pos_w=kp_pos_w,
            robot_body_quat_w=kp_quat_w,
            ref_body_pos_w=ref.body_pos_w[t],
            ref_body_quat_w=ref.body_quat_w[t],
        )
        if odom_vals is not None:
            m.update(odom_vals)
        if jerk_vals is not None:
            m.update(jerk_vals)
        for key, val in m.items():
            metrics[key][k - 1] = val
        robot_body = plant.body_pos(tracked_ids)
        causes = check_terminations(
            term,
            ref_root_pos=ref.root_pos_w[t],
            ref_root_quat=ref.root_quat_w[t],
            robot_root_pos=root_pos,
            robot_root_quat=root_quat,
            rel_body_pos=rel_by_name,
            robot_body_pos={n: robot_body[i] for i, n in enumerate(tracked)},
        )
        for c, fired in causes.items():
            if fired and first_fire[c] < 0:
                first_fire[c] = k
        t = ref.clamp(t + 1)
        t_live = ref_live.clamp(t_live + 1)

    counted = tuple(getattr(term, "fail_causes", CAUSE_ORDER))
    training = [first_fire[c] for c in counted if first_fire[c] >= 0]
    if training:
        ff = min(training)
        cause = next(c for c in counted if first_fire[c] == ff)
    else:
        ff, cause = -1, ""
    rp_summary = replanner.summary() if replanner is not None else None
    return ClipResult(
        clip_name=ref.name,
        source_tag=ref.source_tag,
        clip_len_steps=int(ref.T),
        valid_until=K,
        first_fail_step=int(ff),
        first_fail_cause=cause,
        first_fire=dict(first_fire),
        metrics=metrics,
        extra={
            "final_root_z": float(plant.root_pos[2]),
            "clip_len_steps_original": int(getattr(ref, "T_original", ref.T)),
            "replan": ({**rp_summary, "replan_steps": replan_steps} if rp_summary is not None else None),
            # exports with object inputs report how those inputs were fed (always zeros on the object-free benchmark plant); others None
            "object_obs": (ctrl.object_obs_summary() if callable(getattr(ctrl, "object_obs_summary", None)) else None),
            "action_delay_steps": int(getattr(ctrl, "delay_steps", 0) or 0),
            # odometry: what ran, whether the controller consumed it, the estimate error at the end / averaged over the valid steps
            "odometry": (odometry_extra(odom, odom_source, ctrl, metrics, K, base_seed=int(odom_cfg_base.seed)) if odom is not None else None),
            # jerk columns over the valid steps (+ the POST_REPLAN_WINDOW_S after every replan step under replanning); None when not recorded
            "jerk": (jerk_summary(metrics, K, replan_steps, dt_ctrl) if record_jerk else None),
        },
    )


def jerk_summary(metrics: Mapping[str, np.ndarray], K: int, replan_steps: Sequence[int] = (), dt: float = 0.02, *, window_s: float = POST_REPLAN_WINDOW_S) -> dict[str, Any]:
    """``ClipResult.extra["jerk"]``: per :data:`JERK_METRIC_KEYS` column the mean over the valid steps ``1..K`` (``mean``) and -- when
    ``replan_steps`` is non-empty -- the mean over the union of the windows ``[s, s + window_s)`` after every replan step ``s``
    (``post_replan``; 1-based steps -> columns ``s-1 ..``), plus the window length and the event count.  NaN-safe; None when a column
    has no finite value."""
    K = int(K)
    n_win = max(int(round(float(window_s) / float(dt))), 1)
    mask = np.zeros(K, dtype=bool)
    for s_ in replan_steps:
        a = max(int(s_) - 1, 0)
        mask[a: min(a + n_win, K)] = True
    out: dict[str, Any] = {"window_s": float(window_s), "window_steps": int(n_win), "n_replans": int(len(replan_steps)), "mean": {}, "post_replan": {}}
    for key in JERK_METRIC_KEYS:
        if key not in metrics:
            continue
        col = np.asarray(metrics[key][:K], dtype=np.float64)
        fin = np.isfinite(col)
        out["mean"][key] = float(col[fin].mean()) if fin.any() else None
        sel = fin & mask
        out["post_replan"][key] = float(col[sel].mean()) if (len(replan_steps) and sel.any()) else None
    return out


def clip_odometry_seed(base_seed: int, clip_name: str) -> int:
    """The odometry noise seed of one clip: ``base + crc32(clip name) % 2**31`` -- different clips of a run draw different noise streams, the
    same clip is reproducible across runs and conditions whatever the clip ordering / subset."""
    return int(base_seed) + (zlib.crc32(str(clip_name).encode("utf-8")) % (2**31))


def odometry_extra(odom: LidarInertialOdometry, odom_source: str, ctrl: Any, metrics: Mapping[str, np.ndarray], K: int, *, base_seed: int | None = None) -> dict[str, Any]:
    """``ClipResult.extra["odometry"]``: ``requested_source`` (the estimator that ran -- this block exists only then), ``policy_source`` (what
    the CONTROLLER builds its terms from: ``"so"`` for a ``HeroExportPolicy(odom_source="so")``, ``"truth"`` otherwise, which then only has its
    error recorded), ``fed_to_policy``, ``policy_terms``, the estimator description, ``seed`` (base / effective per clip), ``end`` / ``mean_abs``
    of the odom error columns (cm / deg) and the fraction of steps with a stance foot (None for the LiDAR-inertial model)."""
    feeds = list(getattr(ctrl, "odom_feeds_terms", ()) or ())
    est_source = str(getattr(odom, "source", "so"))
    end: dict[str, float | None] = {}
    mean: dict[str, float | None] = {}
    for key in ODOM_METRIC_KEYS:
        col = np.asarray(metrics[key][:K], dtype=np.float64)
        fin = col[np.isfinite(col)]
        end[key] = float(fin[-1]) if fin.size else None
        mean[key] = float(np.abs(fin).mean()) if fin.size else None
    stance = np.asarray(metrics["odom_stance_feet"][:K], dtype=np.float64)
    return {
        "requested_source": est_source,
        "policy_source": odom_source,
        "fed_to_policy": bool(odom_source == est_source and odom_source != "truth" and feeds),
        "policy_terms": feeds,
        "estimator": odom.describe(),
        "seed": {"base": base_seed, "effective": int(odom.cfg.seed), "rule": "base + crc32(clip_name) % 2**31"},
        "steps": int(odom.step_count),
        "end": end,
        "mean_abs": mean,
        "stance_fraction": (float(np.mean(stance[np.isfinite(stance)] > 0)) if np.isfinite(stance).any() else None),
    }


__all__ = ["JERK_METRIC_KEYS", "POST_REPLAN_WINDOW_S", "ClipResult", "Controller", "RobotState", "RolloutConfig", "as_controller", "clip_odometry_seed", "jerk_summary",
           "odometry_extra", "read_state", "rollout_clip"]
