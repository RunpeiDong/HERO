"""Single-actor PPO for HERO.

One MLP actor over the whole 29-DoF action and one critic, trained by the stock holosoma PPO. Only the export
differs from holosoma: the ONNX takes one frame-major ``actor_obs`` input (same layout as the dual export, so
the same deployment reader works), never embeds the motion corpus, and is published with a ``_hero.json``
sidecar. The paper's dual-actor variant lives in ``hero_isaacsim.agents.ppo_dual``."""
from __future__ import annotations

import json
import os
import tempfile
from pathlib import Path
from typing import Any

import torch
from loguru import logger

from holosoma.agents.ppo.ppo import PPO
from holosoma.config_types.algo import PPOConfig
from holosoma.utils.inference_helpers import attach_onnx_metadata, publish_onnx_atomically, validate_onnx_deployment_metadata

from hero_isaacsim.agents.ppo_dual.export import (
    ACTION_CONTRACT_HERO_RESIDUAL_UPPER,
    HERO_ANCHOR_TERMS_KEY,
    HERO_H20_FUTURE_STEPS_KEY,
    HERO_H21_FUTURE_STEPS_KEY,
    HERO_H22_FUTURE_STEPS_KEY,
    ONNX_INPUT_NAMES_SINGLE,
    ONNX_OUTPUT_NAME,
    WHOLE_BODY,
    SingleActorOnnxWrapper,
    dual_export_metadata,
    export_dual_actor_as_onnx,
    hero_anchor_terms_block,
    hero_h20_future_steps,
    hero_sidecar_from_algo,
    hero_sidecar_path,
    hero_term_future_steps,
    input_perm_for_layout,
)
from hero_isaacsim.agents.ppo_dual.layout import HISTORY_LAYOUT_FRAME_MAJOR, ActorObsLayout, layout_from_env

ALGO_NAME = "PPOSingle"


class PPOSingle(PPO):
    """holosoma PPO (single actor, single critic) with the HERO frame-major ONNX export and sidecar."""

    def __init__(self, env, config: PPOConfig, log_dir, device="cpu", multi_gpu_cfg: dict | None = None):
        super().__init__(env, config, log_dir, device=device, multi_gpu_cfg=multi_gpu_cfg)
        self._actor_obs_layout: ActorObsLayout | None = None
        self.history_layout_export = str(getattr(config, "history_layout_export", HISTORY_LAYOUT_FRAME_MAJOR))

    # ------------------------------------------------------------------ layout / contract helpers
    def _resolve_actor_obs_layout(self) -> ActorObsLayout | None:
        if self._actor_obs_layout is not None:
            return self._actor_obs_layout
        if len(self.actor_obs_keys) != 1:
            return None
        self._actor_obs_layout = layout_from_env(self.env, group_name=self.actor_obs_keys[0])
        return self._actor_obs_layout

    def _action_contract(self) -> str:
        contract = getattr(self.env, "action_contract", None)
        if not contract:
            contract = getattr(getattr(self.env, "action_manager", None), "action_contract", None)
        return str(contract) if contract else ACTION_CONTRACT_HERO_RESIDUAL_UPPER

    def _export_metadata(self, layout: ActorObsLayout | None, history_layout: str) -> dict[str, Any]:
        metadata = self._onnx_deployment_metadata()  # dof_names / kp / kd / action_scale / default_dof_pos / urdf ...
        contract = self._action_contract()
        if layout is not None:
            metadata.update(dual_export_metadata(layout, history_layout, None, base_actor_obs_layout=metadata.get("actor_obs_layout"),
                                                 action_contract=contract, input_names=ONNX_INPUT_NAMES_SINGLE, body_keys=(WHOLE_BODY,),
                                                 algo=ALGO_NAME, inputs_note="One whole-body actor; the single input is the frame-major actor_obs vector."))
        else:
            metadata.update({"history_layout": history_layout, "onnx_inputs": list(ONNX_INPUT_NAMES_SINGLE), "onnx_output": ONNX_OUTPUT_NAME,
                             "body_keys": [WHOLE_BODY], "action_split": None, "action_contract": contract, "algo": ALGO_NAME})
        dof_names = list(metadata.get("dof_names") or [])
        if len(dof_names) == self.num_act:
            metadata["action_groups"] = {WHOLE_BODY: dof_names}
        metadata["init_noise_std"] = {WHOLE_BODY: float(self.config.init_noise_std)}
        exp = getattr(self, "_experiment_config", None)
        metadata[HERO_H20_FUTURE_STEPS_KEY] = hero_h20_future_steps(exp) if exp is not None else None
        metadata[HERO_H21_FUTURE_STEPS_KEY] = hero_term_future_steps(exp, "h21_ref_root_rot_b") if exp is not None else None
        metadata[HERO_H22_FUTURE_STEPS_KEY] = hero_term_future_steps(exp, "h22_ref_root_height_b") if exp is not None else None
        term_dims = dict(zip(layout.terms, layout.to_metadata(history_layout)["term_dims"])) if layout is not None else {}
        metadata[HERO_ANCHOR_TERMS_KEY] = hero_anchor_terms_block(exp, term_dims) if (exp is not None and term_dims) else None
        return metadata

    # ------------------------------------------------------------------ export
    def export(self, onnx_file_path: str) -> None:
        """Single-input ONNX (``actor_obs`` -> ``action``, HERO frame-major layout) plus the ``_hero.json`` sidecar.

        Written, annotated and validated in a staging directory, then published with one rename each; the ONNX is
        published before the sidecar, so a visible sidecar implies a complete ONNX. The motion corpus is never
        embedded (holosoma's ``PPO.export`` would embed it when a motion command exists)."""
        was_training = self.actor.training
        self._eval_mode()
        history_layout = self.history_layout_export
        layout = self._resolve_actor_obs_layout()
        if layout is None and history_layout == HISTORY_LAYOUT_FRAME_MAJOR:
            raise ValueError(f"frame_major_hero_v1 export needs exactly one concatenated actor obs group; actor_obs_keys={self.actor_obs_keys}")
        input_perm = input_perm_for_layout(layout, history_layout) if layout is not None else None
        wrapper = SingleActorOnnxWrapper(self.actor, input_perm=input_perm, obs_normalizer=self.actor_obs_normalizer,
                                         empirical_normalization=self.empirical_normalization).to(self.device)
        sidecar_path = hero_sidecar_path(onnx_file_path)
        with tempfile.TemporaryDirectory(prefix="hero_ppo_single_onnx_export_") as staging_dir:
            staged_path = os.path.join(staging_dir, os.path.basename(onnx_file_path))
            export_dual_actor_as_onnx(wrapper, staged_path, self._get_zero_input(), input_names=ONNX_INPUT_NAMES_SINGLE)
            metadata = self._export_metadata(layout, history_layout)
            attach_onnx_metadata(onnx_path=staged_path, metadata=metadata)
            validate_onnx_deployment_metadata(staged_path)
            staged_sidecar = os.path.join(staging_dir, os.path.basename(sidecar_path))
            sidecar = hero_sidecar_from_algo(self, metadata, onnx_file_path, extra={"algo_target": f"{type(self).__module__}.{type(self).__name__}"})
            Path(staged_sidecar).write_text(json.dumps(sidecar, indent=2, default=str))
            publish_onnx_atomically(staged_path, onnx_file_path)
            publish_onnx_atomically(staged_sidecar, sidecar_path)
        self.logging_helper.save_to_wandb(onnx_file_path)
        self.logging_helper.save_to_wandb(sidecar_path)
        logger.info(f"PPOSingle: exported {onnx_file_path} (+ {os.path.basename(sidecar_path)})")
        if was_training:
            self._train_mode()
