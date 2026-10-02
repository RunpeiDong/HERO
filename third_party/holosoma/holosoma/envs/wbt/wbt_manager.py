from __future__ import annotations

import time
from typing import Any

import torch

from holosoma.envs.base_task.base_task import BaseTask
from holosoma.utils.rotations import quat_error_magnitude

# from holosoma.envs.legged_base_task.legged_robot_base import LeggedRobotBase
from holosoma.utils.simulator_config import SimulatorType


class WholeBodyTrackingManager(BaseTask):
    def __init__(self, tyro_config, *, device):
        super().__init__(tyro_config, device=device)
        assert not hasattr(self.simulator, "gym"), "WBT requires IsaacSim — IsaacGym is not supported."

    def _init_buffers(self):
        """Initialize torch tensors which will contain simulation states and processed quantities"""
        super()._init_buffers()

        # -------------------------------- terms same with locomotion_manager.py [start]--------------------------------
        self.base_quat = self.simulator.base_quat
        self.need_to_refresh_envs = torch.ones(self.num_envs, dtype=torch.bool, device=self.device, requires_grad=False)
        self._configure_default_dof_pos()
        self._init_domain_rand_buffers()

    def _configure_default_dof_pos(self):
        self.default_dof_pos_base = torch.zeros(
            self.num_dof, dtype=torch.float, device=self.device, requires_grad=False
        )
        for i in range(self.num_dofs):
            name = self.dof_names[i]
            if name not in self.robot_config.init_state.default_joint_angles:
                raise ValueError(f"Missing default joint angle for DOF '{name}' in robot configuration.")
            angle = self.robot_config.init_state.default_joint_angles[name]
            self.default_dof_pos_base[i] = angle

        self.default_dof_pos_base = self.default_dof_pos_base.unsqueeze(0)  # (1, num_dof)
        self.default_dof_pos = self.default_dof_pos_base.repeat(self.num_envs, 1).clone()  # (num_envs, num_dof)

    def _pre_compute_observations_callback(self):
        self.base_quat[:] = self.simulator.base_quat[:]

    def _reset_buffers_callback(self, env_ids, target_buf=None):
        # Clear the carried flag so success requires a lift in the current episode.


        if getattr(self, "_box_lift_max", None) is not None:
            self._box_lift_max[env_ids] = 0.0
        self.need_to_refresh_envs[env_ids] = True
        self.episode_length_buf[env_ids] = 0
        self.reset_buf[env_ids] = 1
        # pending_episode_update_mask is only used in curriculum_term::AverageEpisodeLengthTracker.
        self._pending_episode_update_mask[env_ids] = True

    def _get_envs_to_refresh(self):
        return self.need_to_refresh_envs.nonzero(as_tuple=False).flatten()

    def _refresh_envs_after_reset(self, env_ids):
        self.simulator.set_actor_root_state_tensor(env_ids, self.simulator.all_root_states)
        self.simulator.set_dof_state_tensor(env_ids, self.simulator.dof_state)
        self.simulator.clear_contact_forces_history(env_ids)
        self.need_to_refresh_envs[env_ids] = False
        self.simulator.refresh_sim_tensors()
        self._pre_compute_observations_callback()

    def _get_average_episode_tracker(self):
        tracker = self.curriculum_manager.get_term("average_episode_tracker")
        if tracker is None:
            raise RuntimeError("AverageEpisodeLengthTracker is not registered with the curriculum manager.")
        return tracker

    # -------------------------------- terms same with locomotion_manager.py [end]--------------------------------

    def _update_log_dict(self):
        # _update_log_dict happens before reset_envs_idx
        # -------------------------------- terms same with locomotion_manager.py [start]--------------------------------
        # Keep the value on the training device: this runs every env step, and a
        # `.cpu()` here forces a device sync per step.  The logging helper only
        # reduces these tensors at the end of each logging interval.
        avg = self._get_average_episode_tracker().get_average()
        self.log_dict["average_episode_length"] = avg.detach()
        # -------------------------------- terms same with locomotion_manager.py [end]--------------------------------
        # Episode endings split by cause. A single `done` rate cannot tell a policy
        # that survives to the clip's end from one that keeps falling, and those
        # move in opposite directions: as tracking improves, terminations should
        # fall and timeouts should rise. Kept on the training device; the logging
        # helper reduces at interval boundaries.
        _timeout = self.time_out_buf
        _done = self.reset_buf
        self.log_dict["termination/done_rate"] = _done.float().mean()
        self.log_dict["termination/timeout_rate"] = _timeout.float().mean()
        # A done that is not a timeout is a real failure (fall, box dropped, drift).
        self.log_dict["termination/failure_rate"] = (_done & ~_timeout).float().mean()
        self.log_dict["termination/timeout_frac_of_done"] = (
            _timeout.sum() / _done.sum().clamp(min=1)
        )

        # Add tracking metrics to log_dict
        motion_command = self.command_manager.get_state("motion_command")
        motion_command.update_metrics()
        self.log_dict.update(motion_command.metrics)
        self._update_object_task_success(motion_command)

    # Evaluate placement success at several position and orientation tolerances.


    _SR_TIERS = (
        ("loose", 0.30, 1.047),   # 30 cm / 60 deg -- co-train parity
        ("mid", 0.15, 0.524),     # 15 cm / 30 deg
        ("tight", 0.07, 0.262),   # 7 cm / 15 deg
        ("strict", 0.03, 0.131),  # 3 cm / 7.5 deg -- placement-grade
    )
    # The tier the unqualified SR_goal / SR_clean keys report, so those names keep
    # meaning what they meant before the tiers existed.
    _SR_DEFAULT_TIER = "loose"
    _SR_PELVIS_Z_M = 0.5
    _SR_LIFT_M = 0.1

    def _update_object_task_success(self, motion_command):
        """Log per-step task-success gates alongside dense tracking rewards."""
        # has_object lives on the LOADER, not the command term. Reading it off the
        # command would be silently False and drop these metrics on every run.
        if not getattr(getattr(motion_command, "motion", None), "has_object", False):
            return

        box_pos = motion_command.simulator_object_pos_w
        ref_pos = motion_command.object_pos_w
        # Horizontal distance only: vertical offset is governed by whether the box is
        # held or resting, which the lift gate below judges separately.
        pos_err = torch.norm(box_pos[:, :2] - ref_pos[:, :2], dim=-1)
        rot_err = quat_error_magnitude(motion_command.object_quat_w, motion_command.simulator_object_quat_w)

        self.log_dict["task/box_pos_err_m"] = pos_err.mean()
        self.log_dict["task/box_rot_err_rad"] = rot_err.mean()

        # Upright, and the box genuinely carried: together these rule out
        # "succeeding" by collapsing onto the box and shoving it along the ground,
        # which satisfies the position gate while being useless on hardware.
        upright = self.simulator.robot_root_states[:, 2] > self._SR_PELVIS_Z_M
        # Carried is a per-EPISODE property (was the box ever off the floor?), not a
        # per-step one: at the moment of a successful placement the box is back down,
        # so an instantaneous height test would reject every real success. The
        # running max is reset per episode in reset_envs_idx.
        if getattr(self, "_box_lift_max", None) is None or self._box_lift_max.shape[0] != box_pos.shape[0]:
            self._box_lift_max = torch.zeros_like(box_pos[:, 2])
            # Floor height of a resting box, taken from the REFERENCE track's own
            # minimum rather than a hardcoded box size: the corpora use different
            # box scales, and a wrong constant here would silently make every
            # episode look either always-carried or never-carried.
            self._box_floor_z = ref_pos[:, 2].detach().clone()
        self._box_floor_z = torch.minimum(self._box_floor_z, ref_pos[:, 2])
        lift = (box_pos[:, 2] - self._box_floor_z).clamp(min=0.0)
        self._box_lift_max = torch.maximum(self._box_lift_max, lift)
        carried = self._box_lift_max > self._SR_LIFT_M

        self.log_dict["task/upright_frac"] = upright.float().mean()
        self.log_dict["task/box_lift_max_m"] = self._box_lift_max.mean()

        # Report upright and carried success at multiple placement tolerances.


        properly_done = upright & carried
        for name, pos_tol, rot_tol in self._SR_TIERS:
            at_goal = (pos_err <= pos_tol) & (rot_err <= rot_tol)
            self.log_dict[f"task/SR_goal_{name}"] = at_goal.float().mean()
            self.log_dict[f"task/SR_clean_{name}"] = (at_goal & properly_done).float().mean()
            if name == self._SR_DEFAULT_TIER:
                self.log_dict["task/SR_goal"] = self.log_dict[f"task/SR_goal_{name}"]
                self.log_dict["task/SR_clean"] = self.log_dict[f"task/SR_clean_{name}"]

    def reset_all(self):
        # Reset per-environment motion buffers without erasing the learned
        # adaptive failure table.  PPO calls reset_all() once in __init__ and
        # again at learn() entry, including after load(); clearing the table here
        # would make adaptive-sampler checkpoint restoration ineffective.
        motion_command = self.command_manager.get_state("motion_command")
        motion_command.init_buffers(reset_adaptive_sampler=False)
        if getattr(motion_command.motion_cfg, "use_adaptive_timesteps_sampler", False):
            # BaseTask.reset_all() performs one zero-action step solely to
            # materialize observations.  It contains no curriculum evidence and
            # must not apply an artificial zero-failure EMA decay after resume.
            motion_command._skip_adaptive_update_once = True
        # The forced all-env reset also is not an episode sample for the scalar
        # episode-length EMA (notably rank 0 calls this during ONNX export).
        self._get_average_episode_tracker().suppress_next_update()
        # CommandManager.reset runs before TerminationManager.reset in the
        # ordinary per-env reset path so MotionCommand can attribute genuine
        # failures.  A full infrastructure reset has no terminal transition;
        # clear stale flags before super().reset_all() or they would be recorded
        # as artificial phase-zero failures after the buffers above are reset.
        self.termination_manager.reset(None)
        return super().reset_all()

    def has_curricula_enabled(self) -> bool:
        """Expose WBT's adaptive sampler to the algorithm sync hook."""

        motion_command = self.command_manager.get_state("motion_command")
        adaptive = bool(
            motion_command is not None
            and getattr(motion_command.motion_cfg, "use_adaptive_timesteps_sampler", False)
        )
        return adaptive or bool(getattr(self, "use_reward_penalty_curriculum", False)) or bool(
            getattr(self, "use_domain_rand_scale_curriculum", False)
        )

    def get_checkpoint_state(self) -> dict[str, Any]:
        """Persist state that changes the WBT training distribution."""

        state: dict[str, Any] = {
            "average_episode_tracker": self._get_average_episode_tracker().state_dict(),
        }
        motion_command = self.command_manager.get_state("motion_command")
        if (
            motion_command is not None
            and getattr(motion_command.motion_cfg, "use_adaptive_timesteps_sampler", False)
        ):
            state["adaptive_timesteps_sampler"] = motion_command.adaptive_timesteps_sampler.state_dict()
        return state

    def load_checkpoint_state(self, state: dict[str, Any] | None) -> None:
        """Restore WBT curriculum state after task construction."""

        if not state:
            return

        tracker_state = state.get("average_episode_tracker")
        if tracker_state is not None:
            tracker = self._get_average_episode_tracker()
            tracker.load_state_dict(tracker_state)
            # The full reset at PPO learn() entry is infrastructure, not a real
            # episode end, and must not immediately overwrite the restored EMA.
            tracker.suppress_next_update()

        sampler_state = state.get("adaptive_timesteps_sampler")
        if sampler_state is not None:
            motion_command = self.command_manager.get_state("motion_command")
            if motion_command is None or not getattr(
                motion_command.motion_cfg, "use_adaptive_timesteps_sampler", False
            ):
                raise ValueError(
                    "checkpoint contains adaptive sampler state but the live WBT task has it disabled"
                )
            motion_command.adaptive_timesteps_sampler.load_state_dict(sampler_state)

    def synchronize_curriculum_state(self, *, device: str, world_size: int) -> None:
        """Synchronize WBT curriculum state at a native PPO rollout boundary."""

        if world_size <= 1:
            return
        if not torch.distributed.is_available() or not torch.distributed.is_initialized():
            raise RuntimeError(
                "WBT requested multi-rank curriculum synchronization without a process group"
            )

        motion_command = self.command_manager.get_state("motion_command")
        if (
            motion_command is not None
            and getattr(motion_command.motion_cfg, "use_adaptive_timesteps_sampler", False)
        ):
            motion_command.adaptive_timesteps_sampler.synchronize_distributed(
                world_size=world_size
            )

        # Preserve HoloSoma locomotion's established rank-0 contract for the
        # scalar episode-length curriculum while the failure table above uses a
        # rank mean (its EMA is linear and data-distribution state).
        tracker = self._get_average_episode_tracker()
        avg_tensor = tracker.get_average().clone().detach().to(device)
        torch.distributed.broadcast(avg_tensor, src=0)
        tracker.set_average(avg_tensor.to(self.device), suppress_update=False)

    def _reset_robot_states_callback(self, env_ids, target_states=None):
        # MotionCommand.reset restores robot and object states.

        pass

    ########################################################### Push robots #########################################
    # TODO: This should be moved to the randomization manager.
    def _init_domain_rand_buffers(self):
        ######################################### DR related tensors #########################################
        # Action delay buffers are now initialized by randomization manager's setup_action_delay_buffers term

        self.push_robot_vel_buf = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
        )
        self.record_push_robot_vel_buf = torch.zeros(
            self.num_envs, 6, dtype=torch.float, device=self.device, requires_grad=False
        )
        self._randomize_push_robots = False
        self._max_push_vel = torch.zeros(6, dtype=torch.float32, device=self.device)

    def _push_robots(self, env_ids):
        """Random pushes the robots. Emulates an impulse by setting a randomized base velocity."""
        if len(env_ids) == 0:
            return
        self.need_to_refresh_envs[env_ids] = True
        max_vel_tensor = self._max_push_vel
        if self.randomization_manager is not None:
            state = self.randomization_manager.get_state("push_randomizer_state")
            if state is not None:
                max_vel_tensor = state.max_push_vel.clone().to(self.device)

        if not isinstance(max_vel_tensor, torch.Tensor) or max_vel_tensor.numel() != 6:
            raise ValueError("WholeBodyTracking push velocity vector must have exactly 6 components.")

        rand = torch.rand(len(env_ids), 6, device=self.device) * 2 - 1
        self.push_robot_vel_buf[env_ids] = rand * max_vel_tensor.unsqueeze(0)
        self.record_push_robot_vel_buf[env_ids] = self.push_robot_vel_buf[env_ids].clone()
        # Additive push to match BeyondMimic/IsaacLab's push_by_setting_velocity.
        self.simulator.robot_root_states[env_ids, 7:13] += self.push_robot_vel_buf[env_ids]
        # Push impulses only take effect in the simulator once we write the mutated root state tensor back.
        self.simulator.set_actor_root_state_tensor_robots(env_ids, self.simulator.robot_root_states)
        self._max_push_vel = max_vel_tensor.clone()

    #########################################################################################################
    ## Debug visualization
    #########################################################################################################

    def _draw_debug_vis_isaacsim(self):
        motion_command = self.command_manager.get_state("motion_command")
        # torso link
        real_robot_pos_xyz = motion_command.robot_ref_pos_w.clone()
        real_robot_quat_xyzw = motion_command.robot_ref_quat_w.clone()
        real_robot_quat_wxyz = real_robot_quat_xyzw[:, [3, 0, 1, 2]]
        motion_command.visualization_markers["real_robot"].visualize(real_robot_pos_xyz, real_robot_quat_wxyz)

        motion_robot_pos_xyz = motion_command.ref_pos_w.clone()
        motion_robot_quat_xyzw = motion_command.ref_quat_w.clone()
        motion_robot_quat_wxyz = motion_robot_quat_xyzw[:, [3, 0, 1, 2]]
        motion_command.visualization_markers["motion_robot"].visualize(motion_robot_pos_xyz, motion_robot_quat_wxyz)

        for body_idx, body_names in enumerate(motion_command.motion_cfg.body_names_to_track):
            motion_robot_body_pos_xyz = motion_command.body_pos_w[0, body_idx].clone()
            motion_command.visualization_markers[f"motion_{body_names}"].visualize(
                motion_robot_body_pos_xyz.unsqueeze(0)
            )

        # object
        if motion_command.motion.has_object:
            real_object_pos_xyz = motion_command.simulator_object_pos_w.clone()
            real_object_quat_xyzw = motion_command.simulator_object_quat_w.clone()
            real_object_quat_wxyz = real_object_quat_xyzw[:, [3, 0, 1, 2]]
            motion_command.visualization_markers["real_object"].visualize(real_object_pos_xyz, real_object_quat_wxyz)

            motion_object_pos_xyz = motion_command.object_pos_w.clone()
            motion_object_quat_xyzw = motion_command.object_quat_w.clone()
            motion_object_quat_wxyz = motion_object_quat_xyzw[:, [3, 0, 1, 2]]
            motion_command.visualization_markers["motion_object"].visualize(
                motion_object_pos_xyz, motion_object_quat_wxyz
            )

    def _draw_debug_vis_isaacgym(self):
        self.simulator.clear_lines()
        n_bodies = len(self.motion_command.motion_cfg.body_names_to_track)
        for env_id in range(self.num_envs):
            for body_idx in range(n_bodies):
                color = (0.0, 1.0, 0.0)
                self.simulator.draw_sphere(
                    self.motion_command.body_pos_relative_w[env_id, body_idx], 0.03, color, env_id, body_idx
                )

                color = (0.0, 0.0, 1.0)
                self.simulator.draw_sphere(
                    self.motion_command.robot_body_pos_w[env_id, body_idx], 0.03, color, env_id, n_bodies + body_idx
                )

            color = (0.0, 1.0, 0.0)
            self.simulator.draw_sphere(self.motion_command.ref_pos_w[env_id], 0.05, color, env_id, n_bodies * 2 + 0)
            color = (0.0, 0.0, 1.0)
            self.simulator.draw_sphere(
                self.motion_command.robot_ref_pos_w[env_id], 0.05, color, env_id, n_bodies * 2 + 1
            )

    def _draw_debug_vis(self):
        if self.simulator.get_simulator_type() == SimulatorType.ISAACSIM:
            self._draw_debug_vis_isaacsim()
        elif self.simulator.get_simulator_type() == SimulatorType.ISAACGYM:
            self._draw_debug_vis_isaacgym()

    def step_visualize_motion(self, actions):
        motion_command = self.command_manager.get_state("motion_command")
        dt = 1.0 / float(motion_command.motion.fps)
        motion_command.step()
        print("time_steps: ", motion_command.time_steps[0].item())
        self._draw_debug_vis()

        # set root_states_from_motion_command
        root_pos = motion_command.root_pos_w.clone()
        root_ori = motion_command.root_quat_w.clone()  # wxyz
        root_lin_vel = motion_command.body_lin_vel_w[:, 0].clone()
        root_ang_vel = motion_command.body_ang_vel_w[:, 0].clone()

        joint_pos = motion_command.joint_pos.clone()
        joint_vel = motion_command.joint_vel.clone()

        env_ids = torch.arange(self.num_envs, device=self.device)
        self.simulator.dof_pos[env_ids] = joint_pos
        self.simulator.dof_vel[env_ids] = joint_vel

        self.simulator.robot_root_states[env_ids, :3] = root_pos
        self.simulator.robot_root_states[env_ids, 3:7] = root_ori
        self.simulator.robot_root_states[env_ids, 7:10] = root_lin_vel
        self.simulator.robot_root_states[env_ids, 10:13] = root_ang_vel

        self.simulator.set_actor_root_state_tensor(env_ids, self.simulator.all_root_states)
        self.simulator.set_dof_state_tensor(env_ids, self.simulator.dof_state)

        if motion_command.motion.has_object:
            # set object root_states from motion command
            object_pos = motion_command.object_pos_w.clone()
            object_ori = motion_command.object_quat_w.clone()
            object_lin_vel = motion_command.object_lin_vel_w.clone()

            object_states = torch.zeros(len(env_ids), 13, device=self.device)
            object_states[:, :3] = object_pos[:]
            object_states[:, 3:7] = object_ori[:]
            object_states[:, 7:10] = object_lin_vel[:]
            object_states[:, 10:13] = torch.zeros_like(object_lin_vel[:])
            self.simulator.set_actor_states(["object"], env_ids, object_states)

        self.simulator.scene.write_data_to_sim()
        self.simulator.sim.forward()
        self.simulator.sim.render()
        self.simulator.refresh_sim_tensors()

        time.sleep(dt)

        return motion_command.time_steps[0].item() >= motion_command.motion.time_step_total - 2
