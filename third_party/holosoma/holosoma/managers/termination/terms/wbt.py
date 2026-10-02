"""Whole Body Tracking-specific termination terms."""

from __future__ import annotations

from typing import Any, List

from holosoma.config_types.termination import TerminationTermCfg
from holosoma.envs.wbt.wbt_manager import WholeBodyTrackingManager
from holosoma.managers.command.terms.wbt import MotionCommand
from holosoma.managers.observation.terms.wbt import gravity_vector
from holosoma.managers.termination.base import TerminationTermBase
from holosoma.utils.rotations import (
    quat_error_magnitude,
    quat_rotate_inverse,
)
from holosoma.utils.safe_torch_import import torch


#########################################################################################################
## Termination terms
#########################################################################################################
def motion_ends(env, **_) -> torch.Tensor:
    """Terminate if the motion ends."""
    motion_command = env.command_manager.get_state("motion_command")
    return motion_command.time_steps >= motion_command.motion.time_step_total - 2


class BadTracking(TerminationTermBase):
    """Terminate if the tracking is bad.

    - bad ref pos
    - bad ref ori
    - bad motion body pos
    if has object:
        - bad object pos
        - bad object ori

    When bad tracking is detected, the motion_commmand.AdaptiveTimestepsSampler will be updated.
    """

    def __init__(self, cfg: TerminationTermCfg, env: WholeBodyTrackingManager):
        super().__init__(cfg, env)

        self.bad_ref_pos_threshold = cfg.params["bad_ref_pos_threshold"]
        self.bad_ref_ori_threshold = cfg.params["bad_ref_ori_threshold"]

        self.bad_motion_body_pos_body_names = cfg.params["bad_motion_body_pos_body_names"]

        # NOTE: body_names_to_track is shared with command_manager
        self.body_names_to_track = cfg.params["body_names_to_track"]
        self.bad_motion_body_pos_threshold = cfg.params["bad_motion_body_pos_threshold"]
        self.bad_motion_body_pos_body_indexes = self._get_index_of_a_in_b(
            self.bad_motion_body_pos_body_names, self.body_names_to_track, self.env.device
        )

        self.bad_object_pos_threshold = cfg.params["bad_object_pos_threshold"]
        self.bad_object_ori_threshold = cfg.params["bad_object_ori_threshold"]
        # When True, score the object-drop gate against the RETARGETED object reference
        # (object_pos_relative_w), mirroring the body termination's retargeted frame. Default False
        # preserves the legacy raw-mocap-frame behavior. See object-reference-frame-bug.
        self.retargeted_object = bool(cfg.params.get("retargeted_object", False))

        # Gate object termination on first achieving the body-tracking thresholds.
        # Once enabled, the gate stays latched for the episode, following OmniRetarget
        # (arXiv:2509.26633, Sec. IV). When disabled, object termination applies immediately.


        self.object_term_requires_body_ok = bool(cfg.params.get("object_term_requires_body_ok", False))
        # Per-env latch: True once body tracking has been OK at least once this episode. Lazily
        # allocated on first __call__ (num_envs/device are known then) so __init__ stays cheap and
        # mirrors the lazy style used elsewhere. Cleared per-env by reset().
        self._body_ever_ok: torch.Tensor | None = None

    def __call__(self, env: Any, **kwargs) -> torch.Tensor:
        motion_command = self.env.command_manager.get_state("motion_command")
        assert motion_command.motion_cfg.body_names_to_track == self.body_names_to_track, (
            "body_names_to_track in motion_command and termination.params are not the same"
            f"motion_command.motion_cfg.body_names_to_track: {motion_command.motion_cfg.body_names_to_track}"
            f"termination.params['body_names_to_track']: {self.body_names_to_track}"
        )

        # return torch.zeros(env.num_envs, dtype=torch.bool, device=env.device)
        bad_ref_pos = self.bad_ref_pos(motion_command)
        bad_ref_ori = self.bad_ref_ori(motion_command)
        bad_motion_body_pos = self.bad_motion_body_pos(motion_command)
        bad_body = bad_ref_pos | bad_ref_ori | bad_motion_body_pos
        bad_tracking = bad_body

        if motion_command.motion.has_object:
            bad_object = self.bad_object_pos(motion_command) | self.bad_object_ori(motion_command)
            if self.object_term_requires_body_ok:
                bad_tracking = bad_tracking | (bad_object & self._object_gate(bad_body))
            else:
                bad_tracking = bad_tracking | bad_object

        return bad_tracking

    def _object_gate(self, bad_body: torch.Tensor) -> torch.Tensor:
        """Return the per-environment latch that enables object termination.

        An environment qualifies when all body-tracking errors are below threshold.
        The latch then stays enabled until reset, even if body tracking later degrades."""
        body_ok = ~bad_body
        if self._body_ever_ok is None or self._body_ever_ok.shape[0] != body_ok.shape[0]:
            # Lazy alloc (or resize on env-count change) on the same device as the inputs.
            self._body_ever_ok = torch.zeros_like(body_ok)
        self._body_ever_ok |= body_ok
        return self._body_ever_ok

    def bad_ref_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the reference position is too far from the robot's position."""
        return torch.norm(motion_command.ref_pos_w - motion_command.robot_ref_pos_w, dim=1) > self.bad_ref_pos_threshold

    def bad_ref_ori(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the reference orientation is too far from the robot's orientation."""
        motion_projected_gravity_b = quat_rotate_inverse(
            motion_command.ref_quat_w, gravity_vector(self.env), w_last=True
        )
        robot_projected_gravity_b = quat_rotate_inverse(
            motion_command.robot_ref_quat_w, gravity_vector(self.env), w_last=True
        )
        return (
            torch.abs(motion_projected_gravity_b[:, 2] - robot_projected_gravity_b[:, 2]) > self.bad_ref_ori_threshold
        )

    def bad_motion_body_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the motion body position is too far from the robot's body position."""
        body_idx = self.bad_motion_body_pos_body_indexes
        error = torch.norm(
            motion_command.body_pos_relative_w[:, body_idx] - motion_command.robot_body_pos_w[:, body_idx], dim=-1
        )
        return torch.any(error > self.bad_motion_body_pos_threshold, dim=-1)

    def bad_object_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the object position is too far from the simulator's object position."""
        ref = motion_command.object_pos_relative_w if self.retargeted_object else motion_command.object_pos_w
        return (
            torch.norm(ref - motion_command.simulator_object_pos_w, dim=-1)
            > self.bad_object_pos_threshold
        )

    def bad_object_ori(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the object orientation is too far from the simulator's object orientation."""
        ref = motion_command.object_quat_relative_w if self.retargeted_object else motion_command.object_quat_w
        return (
            quat_error_magnitude(ref, motion_command.simulator_object_quat_w)
            > self.bad_object_ori_threshold
        )

    def reset(self, env_ids: torch.Tensor | None = None) -> None:
        """Clear the object-termination latch for the specified environments.

        None resets all environments. No-op before the latch is allocated."""
        if self._body_ever_ok is None:
            return
        if env_ids is None:
            self._body_ever_ok.zero_()
        else:
            self._body_ever_ok[env_ids] = False

    #########################################################################################################
    ## Internal Helper functions
    #########################################################################################################
    def _get_index_of_a_in_b(self, a_names: List[str], b_names: List[str], device: str = "cpu") -> torch.Tensor:
        indexes = []
        for name in a_names:
            assert name in b_names, f"The specified name ({name}) doesn't exist: {b_names}"
            indexes.append(b_names.index(name))
        return torch.tensor(indexes, dtype=torch.long, device=device)


class BadTrackingZOnly(BadTracking):
    """BadTracking variant using z-axis-only position checks for parity with BM Wo-State-Estimation."""

    def bad_ref_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if the reference z position is too far from the robot's z position."""
        z_err = torch.abs(motion_command.ref_pos_w[:, -1] - motion_command.robot_ref_pos_w[:, -1])
        return z_err > self.bad_ref_pos_threshold

    def bad_motion_body_pos(self, motion_command: MotionCommand) -> torch.Tensor:
        """Terminate if tracked bodies have too much z-axis position error."""
        body_idx = self.bad_motion_body_pos_body_indexes
        error = torch.abs(
            motion_command.body_pos_relative_w[:, body_idx, -1] - motion_command.robot_body_pos_w[:, body_idx, -1]
        )
        return torch.any(error > self.bad_motion_body_pos_threshold, dim=-1)
