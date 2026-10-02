"""HERO residual joint-position control.

The upper-body reference is added to the policy action before PD control.
The exported action contract records joint order, gains and scales."""

from __future__ import annotations

from typing import Any

import torch

from holosoma.managers.action.terms.joint_control import JointPositionActionTerm

from hero_isaacsim.utils import hero_constants

_C = hero_constants()

ACTION_CONTRACT_NAME = "hero_residual_upper_v1"
"""Contract prefix when a subset of joints carries the reference residual."""
ACTION_CONTRACT_NAME_ALL = "hero_residual_all29_v1"
"""Contract prefix for ``q_target = q_ref(t) + scale * a`` on all 29 joints."""

KNOWN_PARAMS = frozenset({"residual_upper_body_action", "residual_dof_names", "ref_attr"})


class HeroResidualJointPositionActionTerm(JointPositionActionTerm):
    """JointPositionActionTerm with HERO's residual around the reference joint angles.

    cfg.params (unknown keys raise -- a mistyped key would otherwise silently keep the default):
        residual_upper_body_action (bool, default True): add ``q_ref - q_default`` to the scaled actions of the
            residual joints.
        residual_dof_names (list[str], optional): joints receiving the residual (default: the 14 arm joints).
        ref_attr (str, default "ref_upper_dof_pos"): attribute on the motion command holding the absolute
            reference angles: ``[N, 29]`` (``joint_pos``, env DOF order), ``[N, 17]`` (waist 3 + arms 14,
            ``UPPER_REF_DOF_IDX`` order) or ``[N, 14]`` (arms, ``ARM_DOF_IDX`` order).  Columns are matched to
            the residual joints by identity, never by position."""

    def __init__(self, cfg: Any, env: Any):
        super().__init__(cfg, env)
        params = dict(cfg.params or {})
        unknown = set(params) - KNOWN_PARAMS
        if unknown:
            raise ValueError(
                f"HeroResidualJointPositionActionTerm: unknown params {sorted(unknown)}; known: {sorted(KNOWN_PARAMS)}"
            )
        self.residual_upper_body_action = bool(params.get("residual_upper_body_action", True))
        self.ref_attr = str(params.get("ref_attr", "ref_upper_dof_pos"))
        names = list(params.get("residual_dof_names") or [_C.DOF_NAMES[i] for i in _C.ARM_DOF_IDX])
        dof_names = list(env.dof_names)
        missing = [n for n in names if n not in dof_names]
        if missing:
            raise ValueError(f"residual_dof_names not in env.dof_names: {missing}")
        self.residual_dof_names = names
        self.residual_dof_indices = torch.tensor([dof_names.index(n) for n in names], dtype=torch.long, device=env.device)
        self.residual_covers_all_dofs = len(set(names)) == len(dof_names)
        if self.ref_attr != "joint_pos":
            # ref_upper_dof_pos (17 / 14 wide) only carries the waist + arm joints
            upper_names = [_C.DOF_NAMES[i] for i in _C.UPPER_REF_DOF_IDX]
            uncovered = [n for n in names if n not in upper_names]
            if uncovered:
                raise ValueError(
                    f"residual_dof_names {uncovered} are not part of {self.ref_attr!r} (waist + arm joints only); "
                    "use ref_attr='joint_pos' for a full-body reference"
                )
        self._ref_cols_by_width: dict[int, torch.Tensor] = {}
        self._residual = torch.zeros(env.num_envs, len(names), device=env.device)
        env.hero_joint_action_term = self  # lets reward terms find `torques` without knowing the term name

    # ------------------------------------------------------------------------------------------
    def _reference_columns(self, width: int) -> torch.Tensor:
        """Column of each residual joint inside a ``width``-wide reference tensor (cached per width).

        ``len(DOF_NAMES)`` -> env DOF order (``joint_pos``); ``len(UPPER_REF_DOF_IDX)`` -> waist + arms
        (``ref_upper_dof_pos``); ``len(ARM_DOF_IDX)`` -> arms only.  Raises if the layout is unknown or does not
        contain one of the residual joints (never falls back to a positional slice)."""
        cols = self._ref_cols_by_width.get(width)
        if cols is not None:
            return cols
        dof_idx = self.residual_dof_indices.tolist()
        layouts = {
            len(_C.DOF_NAMES): list(range(len(_C.DOF_NAMES))),
            len(_C.UPPER_REF_DOF_IDX): list(_C.UPPER_REF_DOF_IDX),
            len(_C.ARM_DOF_IDX): list(_C.ARM_DOF_IDX),
        }
        layout = layouts.get(width)
        if layout is None:
            raise ValueError(
                f"{self.ref_attr!r} has {width} columns; known reference layouts: {sorted(layouts)} "
                "(env DOFs / waist+arms / arms)"
            )
        uncovered = [self.residual_dof_names[i] for i, d in enumerate(dof_idx) if d not in layout]
        if uncovered:
            raise ValueError(f"reference {self.ref_attr!r} ({width} wide) has no column for residual joints {uncovered}")
        cols = torch.tensor([layout.index(d) for d in dof_idx], dtype=torch.long, device=self.residual_dof_indices.device)
        self._ref_cols_by_width[width] = cols
        return cols

    def _reference_arm_pos(self) -> torch.Tensor | None:
        """``[N, len(residual_dof_indices)]`` reference angles of the residual joints (by identity), or None."""
        cm = getattr(self.env, "command_manager", None)
        mc = cm.get_state("motion_command") if cm is not None else None
        ref = getattr(mc, self.ref_attr, None) if mc is not None else None
        if ref is None:
            return None
        return ref[:, self._reference_columns(int(ref.shape[1]))]

    def current_residual(self) -> torch.Tensor:
        """``[N, K]`` residual ``q_ref - q_default`` for the residual joints (zeros if no reference / disabled)."""
        if not self.residual_upper_body_action:
            return torch.zeros_like(self._residual)
        ref = self._reference_arm_pos()
        if ref is None:
            return torch.zeros_like(self._residual)
        return ref - self.env.default_dof_pos[:, self.residual_dof_indices]

    def scaled_actions_with_residual(self, actions: torch.Tensor) -> torch.Tensor:
        """``a * action_scales`` with the residual added on the residual joints."""
        actions_scaled = actions * self.action_scales
        if self.residual_upper_body_action:
            res = self.current_residual()
            actions_scaled = actions_scaled.clone()
            actions_scaled[:, self.residual_dof_indices] += res
        return actions_scaled

    def _compute_torques(self, actions: torch.Tensor) -> torch.Tensor:
        """Mirror of ``JointPositionActionTerm._compute_torques`` (joint_control.py) with the residual."""
        actions_scaled = self.scaled_actions_with_residual(actions)
        control_type = self.env.robot_config.control.control_type
        if control_type == "P":
            torques = (
                self._kp_scale * self.p_gains * (actions_scaled + self.env.default_dof_pos - self.env.simulator.dof_pos)
                - self._kd_scale * self.d_gains * self.env.simulator.dof_vel
            )
        elif control_type == "V":
            torques = (
                self._kp_scale * self.p_gains * (actions_scaled - self.env.simulator.dof_vel)
                - self._kd_scale * self.d_gains * (self.env.simulator.dof_vel - self._prev_dof_vel) / self.env.sim_dt
            )
        elif control_type == "T":
            torques = actions_scaled
        else:
            raise ValueError(f"Unknown controller type: {control_type}")

        if self._randomize_torque_rfi:
            torques = (
                torques
                + (torch.rand_like(torques) * 2.0 - 1.0) * self._rfi_lim * self._rfi_lim_scale * self.env.torque_limits
            )
        if self.env.robot_config.control.clip_torques:
            torques = torch.clip(torques, -self.env.torque_limits, self.env.torque_limits)
        return torques

    # ------------------------------------------------------------------------------------------
    @property
    def contract_name(self) -> str:
        """``hero_residual_all29_v1`` when every env DOF carries the residual, else ``hero_residual_upper_v1``."""
        return ACTION_CONTRACT_NAME_ALL if (self.residual_upper_body_action and self.residual_covers_all_dofs) else ACTION_CONTRACT_NAME

    @property
    def action_contract(self) -> str:
        """Human/ONNX-metadata description of the action mapping (stable prefix = :pyattr:`contract_name`)."""
        control = self.env.robot_config.control
        scales = [round(float(s), 6) for s in self.action_scales.tolist()]
        idx = self.residual_dof_indices.tolist()
        residual = "on" if self.residual_upper_body_action else "off"
        return (
            f"{self.contract_name}: q_target = q_default + action_scales * a"
            f"{' + (q_ref - q_default) on dofs ' + str(idx) if self.residual_upper_body_action else ''}; "
            f"residual={residual}; ref_attr={self.ref_attr}; control_type={control.control_type}; "
            f"action_scale={control.action_scale} by_effort_limit_over_p_gain={control.action_scales_by_effort_limit_over_p_gain}; "
            f"action_scales={scales}; action_clip={control.action_clip_value if control.clip_actions else None}; "
            f"residual_dof_names={self.residual_dof_names}"
        )

    def action_contract_metadata(self) -> dict[str, Any]:
        """Structured version of :pyattr:`action_contract` for ONNX metadata writers (sidecar ``action_contract_detail``)."""
        control = self.env.robot_config.control
        return {
            "action_contract": self.contract_name,
            "residual_upper_body_action": self.residual_upper_body_action,
            "residual_dof_indices": self.residual_dof_indices.tolist(),
            "residual_dof_names": list(self.residual_dof_names),
            "ref_attr": self.ref_attr,  # q_ref source on the motion command: joint_pos (29) / ref_upper_dof_pos (17)
            "action_scales": [float(s) for s in self.action_scales.tolist()],
            "action_scale": float(control.action_scale),
            "action_scales_by_effort_limit_over_p_gain": bool(control.action_scales_by_effort_limit_over_p_gain),
            "action_clip_value": float(control.action_clip_value) if control.clip_actions else None,
            "control_type": control.control_type,
        }


__all__ = ["ACTION_CONTRACT_NAME", "ACTION_CONTRACT_NAME_ALL", "KNOWN_PARAMS", "HeroResidualJointPositionActionTerm"]
