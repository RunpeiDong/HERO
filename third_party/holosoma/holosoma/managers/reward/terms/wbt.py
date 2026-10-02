"""Reward terms for Whole Body Tracking tasks."""

from __future__ import annotations

import re
from typing import TYPE_CHECKING, List

import torch

from holosoma.config_types.reward import RewardTermCfg
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.reward.base import RewardTermBase
from holosoma.utils.rotations import quat_error_magnitude

if TYPE_CHECKING:
    from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager


def _get_motion_command_and_assert_type(env: WholeBodyTrackingManager) -> MotionCommand:
    motion_command = env.command_manager.get_state("motion_command")
    assert motion_command is not None, "motion_command not found in command manager"
    assert isinstance(motion_command, MotionCommand), f"Expected MotionCommand, got {type(motion_command)}"
    return motion_command


#########################################################################################################
## terms same to managers/reward/terms/locomotion.py
#########################################################################################################


def penalty_action_rate(env: WholeBodyTrackingManager) -> torch.Tensor:
    """Penalize changes in actions between steps.

    Args:
        env: The environment instance

    Returns:
        Reward tensor [num_envs]
    """
    actions = env.action_manager.action
    prev_actions = env.action_manager.prev_action
    return torch.sum(torch.square(prev_actions - actions), dim=1)


def limits_dof_pos(env: WholeBodyTrackingManager, soft_dof_pos_limit: float = 0.95) -> torch.Tensor:
    """Penalize joint positions too close to limits.

    Args:
        env: The environment instance
        soft_dof_pos_limit: Soft limit as fraction of hard limit

    Returns:
        Reward tensor [num_envs]
    """
    # Use soft limits as fraction of hard limits
    m = (env.simulator.hard_dof_pos_limits[:, 0] + env.simulator.hard_dof_pos_limits[:, 1]) / 2  # type: ignore[attr-defined]
    r = env.simulator.hard_dof_pos_limits[:, 1] - env.simulator.hard_dof_pos_limits[:, 0]  # type: ignore[attr-defined]
    lower_soft_limit = m - 0.5 * r * soft_dof_pos_limit
    upper_soft_limit = m + 0.5 * r * soft_dof_pos_limit

    out_of_limits = -(env.simulator.dof_pos - lower_soft_limit).clip(max=0.0)  # lower limit
    out_of_limits += (env.simulator.dof_pos - upper_soft_limit).clip(min=0.0)
    return torch.sum(out_of_limits, dim=1)


#########################################################################################################
## terms specific to Whole Body Tracking
#########################################################################################################

# ================================================================================================
# Robot Tracking Rewards
# ================================================================================================


def motion_global_ref_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.ref_pos_w - motion_command.robot_ref_pos_w), dim=-1)
    return torch.exp(-error / sigma**2)


def motion_global_ref_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.ref_quat_w, motion_command.robot_ref_quat_w) ** 2
    return torch.exp(-error / sigma**2)


def motion_relative_body_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_pos_relative_w - motion_command.robot_body_pos_w), dim=-1)
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_relative_body_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.body_quat_relative_w, motion_command.robot_body_quat_w) ** 2
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_global_body_lin_vel(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_lin_vel_w - motion_command.robot_body_lin_vel_w), dim=-1)
    return torch.exp(-error.mean(-1) / sigma**2)


def motion_global_body_ang_vel(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.body_ang_vel_w - motion_command.robot_body_ang_vel_w), dim=-1)
    return torch.exp(-error.mean(-1) / sigma**2)


# ================================================================================================
# Object Tracking Rewards
# ================================================================================================


def object_global_ref_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(torch.square(motion_command.object_pos_w - motion_command.simulator_object_pos_w), dim=-1)
    return torch.exp(-error / sigma**2)


def object_global_ref_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(motion_command.object_quat_w, motion_command.simulator_object_quat_w) ** 2
    return torch.exp(-error / sigma**2)


def object_relative_ref_position_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    """Like object_global_ref_position_error_exp but scores against the RETARGETED object reference
    (object_pos_relative_w), so the box target follows the robot's drifted frame just as the body
    target does. Fixes the object-reference-frame asymmetry that falsely penalizes drift-induced error."""
    motion_command = _get_motion_command_and_assert_type(env)
    error = torch.sum(
        torch.square(motion_command.object_pos_relative_w - motion_command.simulator_object_pos_w), dim=-1
    )
    return torch.exp(-error / sigma**2)


def object_relative_ref_orientation_error_exp(env: WholeBodyTrackingManager, sigma: float) -> torch.Tensor:
    """Retargeted-frame counterpart of object_global_ref_orientation_error_exp."""
    motion_command = _get_motion_command_and_assert_type(env)
    error = quat_error_magnitude(
        motion_command.object_quat_relative_w, motion_command.simulator_object_quat_w
    ) ** 2
    return torch.exp(-error / sigma**2)


# ================================================================================================
# Undesired Contacts Rewards
# ================================================================================================


class UndesiredContacts(RewardTermBase):
    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        undesired_contacts_body_names = [
            body_name
            for body_name in self.env.simulator.body_names  # type: ignore[attr-defined]
            if re.match(cfg.params.get("undesired_contacts_body_names", ""), body_name)
        ]
        self.undesired_contacts_body_indexes = self._get_index_of_a_in_b(
            undesired_contacts_body_names,
            self.env.simulator.body_names,  # type: ignore[attr-defined]
            self.env.device,
        )
        self.threshold = cfg.params.get("threshold", 1.0)

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        # (num_envs, history_length, num_bodies, 3)
        net_contact_forces = self.env.simulator.contact_forces_history
        is_contact = (
            torch.max(torch.norm(net_contact_forces[:, :, self.undesired_contacts_body_indexes], dim=-1), dim=1)[0]
            > self.threshold
        )
        return torch.sum(is_contact, dim=1)

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        pass

    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)


# ================================================================================================
# General HOI contact reward (ResMimic-style, task-agnostic)
#
# Encourages DESIRED contact between a configurable set of "contactor" robot bodies and the
# manipulated object. Designed to generalize across humanoid-object-interaction tasks by only
# changing the body-name regex + a per-contactor proximity gate:
#   - box carrying   : contactor_body_names = "(left|right)_(wrist|elbow)..."  (hands hold box)
#   - sitting a chair: contactor_body_names = "pelvis|torso_link"              (seat contacts chair)
#   - wiping a board : contactor_body_names = "right_wrist_yaw_link"           (hand on board)
#
# A contact counts as "with the object" only when the contactor body is within `proximity` of
# the object COM (gates out floor/self contacts that also register a force). Reward is the
# fraction of intended contactors that are simultaneously (force > threshold) AND (near object).
# Optionally compared against an "intended-contact" schedule derived from the reference object
# trajectory (object being lifted => contact intended); when `require_object_moving` is set, the
# reward is only active while the reference object is off its start height (i.e. being manipulated).
# ================================================================================================


class ObjectContactReward(RewardTermBase):
    """Task-agnostic body<->object contact reward for HOI tasks.

    params:
      contactor_body_names: regex selecting robot bodies that SHOULD contact the object.
      threshold:            net contact-force norm (N) above which a body counts as in contact.
      proximity:            max distance (m) from contactor body to object COM for the contact
                            to be attributed to the object (gates out floor/self contact).
      require_object_lifted: if True, reward is gated to timesteps where the *reference* object
                            is lifted above its start height by `lift_eps` (contact is intended).
      lift_eps:             height (m) above start to consider the object "being manipulated".
    """

    def __init__(self, cfg: RewardTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)
        self.env = env
        pattern = cfg.params.get("contactor_body_names", "")
        contactor_names = [b for b in env.simulator.body_names if re.match(pattern, b)]
        assert len(contactor_names) > 0, f"ObjectContactReward: no body matched regex '{pattern}'"
        self.contactor_idx = self._get_index_of_a_in_b(
            contactor_names, env.simulator.body_names, env.device
        )
        self.threshold = float(cfg.params.get("threshold", 5.0))
        self.proximity = float(cfg.params.get("proximity", 0.35))
        self.require_object_lifted = bool(cfg.params.get("require_object_lifted", False))
        self.lift_eps = float(cfg.params.get("lift_eps", 0.05))
        self._start_obj_z = None

    def __call__(self, env: WholeBodyTrackingManager, **kwargs) -> torch.Tensor:
        mc = _get_motion_command_and_assert_type(env)
        # contactor body positions in world: [E, n_contactor, 3]
        body_pos_w = env.simulator._rigid_body_pos[:, self.contactor_idx, :]
        obj_pos_w = mc.simulator_object_pos_w[:, None, :]  # [E,1,3]
        dist = torch.norm(body_pos_w - obj_pos_w, dim=-1)  # [E, n_contactor]
        near = dist < self.proximity

        # contact force on contactor bodies (latest frame): [E, n_contactor]
        forces = env.simulator.contact_forces[:, self.contactor_idx, :]
        in_contact = torch.norm(forces, dim=-1) > self.threshold

        # a contactor "holds the object" iff it is both near the object and registering force
        holding = (near & in_contact).float()  # [E, n_contactor]
        reward = holding.mean(dim=1)  # fraction of intended contactors holding -> [E]

        if self.require_object_lifted:
            # Gate the reward on reference-object lift relative to that clip's start height.
            # Index by motion ID so random-phase starts and clip resampling use the correct baseline.


            ref_z = mc.object_pos_w[:, 2]
            start_z = None
            try:
                motion = getattr(mc, "motion", None)
                starts = getattr(motion, "motion_start_idx", None)
                if motion is not None and starts is not None and motion.object_pos_w.shape[0] > 0:
                    if not torch.is_tensor(starts):
                        starts = torch.as_tensor(starts, dtype=torch.long, device=ref_z.device)
                    mids = getattr(mc, "motion_ids", None)
                    idx = starts[mids] if (mids is not None and starts.numel() > 1) \
                        else starts.reshape(-1)[0].expand(ref_z.shape[0])
                    # mc.object_pos_w includes env origins; raw loader track does not.
                    # frames() bridges a CPU motion storage device.
                    start_z = motion.frames("object_pos_w", idx)[:, 2] \
                        + env.simulator.scene.env_origins[:, 2]
            except Exception:
                start_z = None
            if start_z is None:
                # legacy fallback (loaders without the start-idx API): first-call cache
                if self._start_obj_z is None or self._start_obj_z.shape[0] != ref_z.shape[0]:
                    self._start_obj_z = ref_z.clone()
                start_z = self._start_obj_z
            intended = (ref_z - start_z) > self.lift_eps
            reward = reward * intended.float()
        return reward

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        # The lift baseline is indexed by clip, so reset has no state to refresh.
        # Reward reset precedes command reset and must not cache the outgoing reference.


        return

    #########################################################################################################
    ## Internal Helper functions
    #########################################################################################################
    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)
