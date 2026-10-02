"""Whole body tracking observation terms."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING

import torch

from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.utils.rotations import (
    quat_inverse,
    quat_mul,
    quat_rotate_inverse,
    quaternion_to_matrix,
    subtract_frame_transforms,
)
from holosoma.utils.torch_utils import get_axis_params, to_torch

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


#########################################################################################################
## terms same to managers/observation/terms/locomotion.py
#########################################################################################################
def _base_quat(env: WholeBodyTrackingManager) -> torch.Tensor:
    return env.base_quat


def gravity_vector(env: WholeBodyTrackingManager, up_axis_idx: int = 2) -> torch.Tensor:
    axis = to_torch(get_axis_params(-1.0, up_axis_idx), device=env.device)
    return axis.unsqueeze(0).expand(env.num_envs, -1)


def base_forward_vector(env: WholeBodyTrackingManager) -> torch.Tensor:
    axis = to_torch([1.0, 0.0, 0.0], device=env.device)
    return axis.unsqueeze(0).expand(env.num_envs, -1)


def get_base_lin_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    root_states = env.simulator.robot_root_states
    lin_vel_world = root_states[:, 7:10]
    return quat_rotate_inverse(_base_quat(env), lin_vel_world, w_last=True)


def get_base_ang_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    ang_vel_world = env.simulator.robot_root_states[:, 10:13]
    return quat_rotate_inverse(_base_quat(env), ang_vel_world, w_last=True)


def get_projected_gravity(env: WholeBodyTrackingManager) -> torch.Tensor:
    return quat_rotate_inverse(_base_quat(env), gravity_vector(env), w_last=True)


def base_lin_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Base linear velocity in base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_base_lin_vel()
    """
    return get_base_lin_vel(env)


def base_ang_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Base angular velocity in base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_base_ang_vel()
    """
    return get_base_ang_vel(env)


def projected_gravity(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Gravity vector projected into base frame.

    Returns:
        Tensor of shape [num_envs, 3]

    Equivalent to:
        env._get_obs_projected_gravity()
    """
    return get_projected_gravity(env)


def dof_pos(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Joint positions relative to default positions.

    Returns:
        Tensor of shape [num_envs, num_dof]

    Equivalent to:
        env._get_obs_dof_pos()
    """
    return env.simulator.dof_pos - env.default_dof_pos


def dof_vel(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Joint velocities.

    Returns:
        Tensor of shape [num_envs, num_dof]

    Equivalent to:
        env._get_obs_dof_vel()
    """
    return env.simulator.dof_vel


def actions(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Last actions taken by the policy.

    Returns:
        Tensor of shape [num_envs, num_actions]

    Equivalent to:
        env._get_obs_actions()
    """
    return env.action_manager.action


#########################################################################################################
## terms specific to Whole Body Tracking
#########################################################################################################


def _get_motion_command_and_assert_type(env: WholeBodyTrackingManager) -> MotionCommand:
    motion_command = env.command_manager.get_state("motion_command")
    assert motion_command is not None, "motion_command not found in command manager"
    assert isinstance(motion_command, MotionCommand), f"Expected MotionCommand, got {type(motion_command)}"
    return motion_command


def motion_command(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    return motion_command.command


def motion_command_future(env: WholeBodyTrackingManager, num_future: int = 0) -> torch.Tensor:
    """Reference command stacked over ``[0, 1, ..., num_future]`` future frames.

    With ``num_future=0`` this is byte-identical to :func:`motion_command` (the current
    single frame). Output shape is ``[num_envs, 58 * (num_future + 1)]`` for 29-dof G1.
    Indices are clamped to the current clip end so it never reads past the clip boundary.
    """
    motion_command = _get_motion_command_and_assert_type(env)
    return motion_command.future_command(num_future)


def motion_ref_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    pos, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.ref_pos_w,
        motion_command.ref_quat_w,
    )
    return pos.view(env.num_envs, -1)


def motion_ref_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    _, ori = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.ref_pos_w,
        motion_command.ref_quat_w,
    )
    mat = quaternion_to_matrix(ori, w_last=True)
    return mat[..., :2].reshape(mat.shape[0], -1)


def robot_body_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)

    num_bodies = len(motion_command.motion_cfg.body_names_to_track)
    pos_b, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_ref_quat_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_body_pos_w,
        motion_command.robot_body_quat_w,
    )

    return pos_b.view(env.num_envs, -1)


def robot_body_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)

    num_bodies = len(motion_command.motion_cfg.body_names_to_track)
    _, ori_b = subtract_frame_transforms(
        motion_command.robot_ref_pos_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_ref_quat_w[:, None, :].repeat(1, num_bodies, 1),
        motion_command.robot_body_pos_w,
        motion_command.robot_body_quat_w,
    )
    mat = quaternion_to_matrix(ori_b, w_last=True)
    return mat[..., :2].reshape(mat.shape[0], -1)


def obj_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    pos, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.simulator_object_pos_w,
        motion_command.simulator_object_quat_w,
    )
    return pos.view(env.num_envs, -1)


def obj_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    _, ori = subtract_frame_transforms(
        motion_command.robot_ref_pos_w,
        motion_command.robot_ref_quat_w,
        motion_command.simulator_object_pos_w,
        motion_command.simulator_object_quat_w,
    )
    mat = quaternion_to_matrix(ori, w_last=True)
    return mat[..., :2].reshape(mat.shape[0], -1)


def obj_lin_vel_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    unit_quat = torch.tensor([0.0, 0.0, 0.0, 1.0], device=env.device).unsqueeze(0).repeat(env.num_envs, 1)
    vel_b, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w.clone(),
        motion_command.robot_ref_quat_w.clone(),
        motion_command.simulator_object_lin_vel_w,
        unit_quat,
    )
    return vel_b.view(env.num_envs, -1)


# Privileged object mass and center-of-mass observations, read from the simulator.
# Cache constant episode values and rebuild when the environment count or tensor changes.


def obj_physics_priv(env: WholeBodyTrackingManager, use_com: bool = False,
                     mass_center: float = 3.0, mass_scale: float = 3.0,
                     com_scale: float = 0.1) -> torch.Tensor:
    """Privileged object physics: normalized mass [N,1] (+ CoM xyz [N,3] if use_com).

    mass_n = (mass - mass_center) / mass_scale ; com_n = com_xyz / com_scale (CoM is the object-frame
    center-of-mass offset). Returns [N, 1] or [N, 4]. Read from the object's root_physx_view; cached
    because mass/CoM are per-episode constants. If the view is unavailable, returns zeros (safe no-op).
    """
    try:
        view = env.simulator.scene.rigid_objects["object"].root_physx_view
    except Exception:  # noqa: BLE001
        return torch.zeros(env.num_envs, 4 if use_com else 1, device=env.device)
    cache = getattr(env, "_obj_physics_priv_cache", None)
    if cache is None or cache["use_com"] != use_com or cache["n"] != env.num_envs:
        masses = view.get_masses().to(env.device).view(env.num_envs, -1)[:, :1]  # [N,1]
        mass_n = (masses - mass_center) / mass_scale
        if use_com:
            coms = view.get_coms().to(env.device).view(env.num_envs, -1)[:, :3]   # [N,3] xyz offset
            feat = torch.cat([mass_n, coms / com_scale], dim=1)                    # [N,4]
        else:
            feat = mass_n                                                          # [N,1]
        env._obj_physics_priv_cache = {"feat": feat, "use_com": use_com, "n": env.num_envs}
    # Clone so a consumer that writes into the returned tensor (noise injection,
    # masking) cannot corrupt the cache for every subsequent step.
    return env._obj_physics_priv_cache["feat"].clone()


# Net hand contact forces in the robot base frame, inspired by GRAIL (arXiv:2606.05160).
# Resolve wrist contact-body indices once; scale and clamp forces to limit impact spikes.
# Output shape: [N, 6] with left and right XYZ forces.


_HAND_CONTACT_BODY_REGEX = r"^(left|right)_wrist_yaw_link$"


def _hand_contact_indices(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Indices (into simulator.body_names) of the hand-region contact bodies; cached on the env.

    Resolved by regex against ``env.simulator.body_names`` exactly like ObjectContactReward
    (reward/terms/wbt.py). Ordered left-then-right to match the regex alternation; asserts both
    hands matched so a silent partial-match never collapses the obs dim.
    """
    cached = getattr(env, "_hand_contact_indices", None)
    if cached is not None:
        return cached
    names = env.simulator.body_names
    matched = [b for b in names if re.match(_HAND_CONTACT_BODY_REGEX, b)]
    assert len(matched) == 2, (
        f"hand_contact_force: expected 2 bodies matching '{_HAND_CONTACT_BODY_REGEX}', "
        f"got {matched} from body_names={names}"
    )
    # keep a stable left-then-right order regardless of body_names ordering
    matched_sorted = sorted(matched, key=lambda b: (0 if b.startswith("left") else 1))
    idx = torch.tensor([names.index(b) for b in matched_sorted], device=env.device, dtype=torch.long)
    env._hand_contact_indices = idx
    return idx


def hand_contact_force(
    env: WholeBodyTrackingManager, force_scale: float = 50.0, clamp: float = 3.0
) -> torch.Tensor:
    """Net contact force on the two hand bodies, in the robot base frame, scaled. [N, 6].

    force_b = R_base^{-1} @ contact_force_world for each hand; then / force_scale and clamped to
    [-clamp, clamp]. Returns left(3)+right(3). If contact_forces is unavailable returns zeros
    (safe no-op so the term never crashes a config-build / mujoco path).
    """
    cf = getattr(env.simulator, "contact_forces", None)
    if cf is None or (hasattr(cf, "numel") and cf.numel() == 0):
        return torch.zeros(env.num_envs, 6, device=env.device)
    idx = _hand_contact_indices(env)
    forces_w = cf[:, idx, :]  # [N, 2, 3] world-frame net contact force
    # rotate each hand's force into the robot base frame (yaw/heading-invariant)
    q = _base_quat(env)  # [N, 4] xyzw
    n_hands = forces_w.shape[1]
    q_exp = q[:, None, :].expand(env.num_envs, n_hands, 4).reshape(-1, 4)
    forces_b = quat_rotate_inverse(q_exp, forces_w.reshape(-1, 3), w_last=True).reshape(env.num_envs, n_hands, 3)
    forces_b = (forces_b / force_scale).clamp(-clamp, clamp)
    return forces_b.reshape(env.num_envs, -1)


# ---------------------------------------------------------------------------
# End-effector (hand) 6DoF residual, HERO-style (arXiv:2602.16705).
# residual = reference EE pose - current EE pose, expressed in the robot base frame.
#   - translation: 3D  (ref_EE_pos - actual_EE_pos in base frame)
#   - rotation:    6D continuous representation (first 2 columns of the relative rotation
#                  matrix); the geodesic rotation distance is implicit in this 6D delta.
# The EE bodies are the wrist links among body_names_to_track. We resolve their indices
# once and cache them on the env.
# ---------------------------------------------------------------------------
_EE_BODY_NAMES = ("left_wrist_yaw_link", "right_wrist_yaw_link")
# VR-3-point style: head (torso_link is the closest upper-body/neck proxy on G1) + two hands.
# Same base-frame 6D residual representation as ee_residual, just one extra tracked body.
_3POINT_BODY_NAMES = ("torso_link", "left_wrist_yaw_link", "right_wrist_yaw_link")


def _ee_local_indices(motion_command) -> list[int]:
    """Indices (into the tracked-body axis) of the EE bodies; cached on the command term."""
    cached = getattr(motion_command, "_ee_local_indices", None)
    if cached is not None:
        return cached
    tracked = list(motion_command.motion_cfg.body_names_to_track)
    idx = [tracked.index(n) for n in _EE_BODY_NAMES if n in tracked]
    motion_command._ee_local_indices = idx
    return idx


def _3point_local_indices(motion_command) -> list[int]:
    """Indices of the VR-3-point bodies (head/torso + 2 hands); cached on the command term."""
    cached = getattr(motion_command, "_3point_local_indices", None)
    if cached is not None:
        return cached
    tracked = list(motion_command.motion_cfg.body_names_to_track)
    idx = [tracked.index(n) for n in _3POINT_BODY_NAMES if n in tracked]
    motion_command._3point_local_indices = idx
    return idx


def ee_residual_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Per-EE translation residual (ref - actual) in base frame. [num_envs, n_ee*3]."""
    motion_command = _get_motion_command_and_assert_type(env)
    ee = _ee_local_indices(motion_command)
    n = len(motion_command.motion_cfg.body_names_to_track)
    # reference EE pose in base frame
    ref_pos_b, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w[:, None, :].repeat(1, n, 1),
        motion_command.robot_ref_quat_w[:, None, :].repeat(1, n, 1),
        motion_command.body_pos_w,
        motion_command.body_quat_w,
    )
    # actual EE pose in base frame
    act_pos_b, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w[:, None, :].repeat(1, n, 1),
        motion_command.robot_ref_quat_w[:, None, :].repeat(1, n, 1),
        motion_command.robot_body_pos_w,
        motion_command.robot_body_quat_w,
    )
    res = (ref_pos_b - act_pos_b)[:, ee, :]  # [num_envs, n_ee, 3]
    return res.reshape(env.num_envs, -1)


def ee_residual_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Per-EE rotation residual as 6D (first 2 cols of ref*actual^-1 rotation). [num_envs, n_ee*6]."""
    motion_command = _get_motion_command_and_assert_type(env)
    ee = _ee_local_indices(motion_command)
    n = len(motion_command.motion_cfg.body_names_to_track)
    ref_q = motion_command.body_quat_w[:, ee, :]          # [num_envs, n_ee, 4]
    act_q = motion_command.robot_body_quat_w[:, ee, :]
    # relative rotation ref * actual^-1 (the geodesic delta), as 6D rotation rep
    rel_q = quat_mul(ref_q, quat_inverse(act_q, w_last=True), w_last=True)
    mat = quaternion_to_matrix(rel_q, w_last=True)        # [num_envs, n_ee, 3, 3]
    return mat[..., :2].reshape(env.num_envs, -1)


def head_residual_pos_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    """VR-3-point translation residual (ref - actual) in base frame for [head/torso, L-hand, R-hand].
    [num_envs, 3*3]. Same construction as ee_residual_pos_b but over the 3-point body set."""
    motion_command = _get_motion_command_and_assert_type(env)
    pts = _3point_local_indices(motion_command)
    n = len(motion_command.motion_cfg.body_names_to_track)
    ref_pos_b, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w[:, None, :].repeat(1, n, 1),
        motion_command.robot_ref_quat_w[:, None, :].repeat(1, n, 1),
        motion_command.body_pos_w,
        motion_command.body_quat_w,
    )
    act_pos_b, _ = subtract_frame_transforms(
        motion_command.robot_ref_pos_w[:, None, :].repeat(1, n, 1),
        motion_command.robot_ref_quat_w[:, None, :].repeat(1, n, 1),
        motion_command.robot_body_pos_w,
        motion_command.robot_body_quat_w,
    )
    res = (ref_pos_b - act_pos_b)[:, pts, :]              # [num_envs, 3, 3]
    return res.reshape(env.num_envs, -1)


def head_residual_ori_b(env: WholeBodyTrackingManager) -> torch.Tensor:
    """VR-3-point rotation residual as 6D for [head/torso, L-hand, R-hand]. [num_envs, 3*6]."""
    motion_command = _get_motion_command_and_assert_type(env)
    pts = _3point_local_indices(motion_command)
    ref_q = motion_command.body_quat_w[:, pts, :]
    act_q = motion_command.robot_body_quat_w[:, pts, :]
    rel_q = quat_mul(ref_q, quat_inverse(act_q, w_last=True), w_last=True)
    mat = quaternion_to_matrix(rel_q, w_last=True)
    return mat[..., :2].reshape(env.num_envs, -1)


# Chunked reference observations.


def chunk_command(
    env: WholeBodyTrackingManager,
    chunk_k: int = 8,
    noise_jp: float = 0.03,
    noise_jv: float = 0.3,
    time_shift: int = 2,
    amp_lo: float = 0.9,
    amp_hi: float = 1.1,
) -> torch.Tensor:
    """Return a held reference chunk and its normalized in-chunk phase.

    Capture chunk_k future frames and perturb their timing, joint positions, joint
    velocities, and amplitude once per chunk. Replan when the chunk is exhausted
    or a motion reset changes its clip or timestep. Rewards retain the true
    reference. Output shape: [num_envs, 58 * chunk_k + 1]."""
    mc = _get_motion_command_and_assert_type(env)
    n = env.num_envs
    dev = mc.time_steps.device
    ndof = mc.motion.joint_pos.shape[1]
    width = 2 * ndof * chunk_k
    st = getattr(env, "_chunk_cmd_state", None)
    if st is None or st["chunk"].shape[0] != n or st["chunk"].shape[1] != width:
        st = {
            "chunk": torch.zeros(n, width, device=dev),
            "start_ts": torch.full((n,), -(10**9), dtype=torch.long, device=dev),
            "motion_ids": torch.full((n,), -1, dtype=torch.long, device=dev),
        }
        env._chunk_cmd_state = st

    ts = mc.time_steps
    mids = mc.motion_ids if hasattr(mc, "motion_ids") else torch.zeros_like(ts)
    since = ts - st["start_ts"]
    replan = (since < 0) | (since >= chunk_k) | (mids != st["motion_ids"])
    if replan.any():
        ids = torch.where(replan)[0]
        m = ids.numel()
        start = mc.motion.motion_start_idx[mids[ids]]
        last = mc.motion.motion_end_idx[mids[ids]] - 1
        shift = torch.randint(-time_shift, time_shift + 1, (m,), device=dev) if time_shift > 0 \
            else torch.zeros(m, dtype=torch.long, device=dev)
        offs = torch.arange(chunk_k, device=dev)
        idx = (ts[ids, None] + shift[:, None] + offs[None, :]).clamp(min=0)
        idx = torch.minimum(torch.maximum(idx, start[:, None]), last[:, None])  # [m, K]
        # frames() bridges a CPU motion storage device (transfers only the
        # gathered frames); with same-device storage it is a plain gather.
        jp = mc.motion.frames("joint_pos", idx)   # [m, K, ndof]
        jv = mc.motion.frames("joint_vel", idx)
        amp = amp_lo + (amp_hi - amp_lo) * torch.rand(m, 1, 1, device=dev)
        jp = jp[:, :1, :] + amp * (jp - jp[:, :1, :])
        jv = amp * jv
        jp = jp + noise_jp * torch.randn_like(jp)
        jv = jv + noise_jv * torch.randn_like(jv)
        st["chunk"][ids] = torch.cat([jp, jv], dim=2).reshape(m, width)
        st["start_ts"][ids] = ts[ids]
        st["motion_ids"][ids] = mids[ids]

    phase = ((ts - st["start_ts"]).float() / float(chunk_k)).clamp(0.0, 1.0)[:, None]
    return torch.cat([st["chunk"], phase], dim=1)
