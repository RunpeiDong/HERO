"""Export is inference-only: a checkpoint's saved sampler settings and failure table must never stop it.

Before this guard, ``scripts/export.py`` built the environment from the checkpoint's saved motion config (an absolute clip
cap of 0.001 in checkpoints written before this change set, refused by today's sampler) and ``algo.load()`` restored the env state (a table
written under the previous rule, or for another corpus than ``--motion-dir``, refused without an opt-in export does
not have). Export now takes the sampler settings from the preset and drops a non-loadable table with a warning."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party/holosoma"), str(ROOT / "scripts"), str(ROOT / "tests")]
from configs import DEFAULTS
from hero_isaacsim.config_values.command import INFERENCE_SAMPLER_FIELDS, get_motion_config, with_inference_motion_config
from hero_isaacsim.managers.command.hero import coerce_hero_motion_config
from hero_isaacsim.managers.command.sampler import HeroAdaptiveTimestepsSampler, build_clip_prior
import test_checkpoint_state_roundtrip as roundtrip
from test_resume_preflight import LEGACY_POLICY, REAL_PRE_FIX_CHECKPOINT, motion_config, pre_fix_saved_config


def load_script(name):
    spec = importlib.util.spec_from_file_location("hero_script_" + name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def make_corpus(tmp_path, tag="amass"):
    tmp_path.mkdir(parents=True, exist_ok=True)
    np.savez(tmp_path / "clip.npz", source_tag=np.asarray(tag), has_object=np.asarray(False))
    (tmp_path / "CORPUS_MANIFEST.json").write_text(json.dumps({"source_weights": {tag: 1.0}, "source_tags": [tag]}))
    return tmp_path


def pre_fix_checkpoint(cfg, num_clips=2000):
    """Legacy-style: pre-fix saved config, tracker, a sampler table for ``num_clips`` clips under the previous rule."""
    bins = 3
    return {"experiment_config": pre_fix_saved_config(cfg), "iter": 7500, "env_state": {
        "average_episode_tracker": {"average_episode_length": torch.tensor(109.94), "suppress_next_update": False},
        "adaptive_timesteps_sampler": {
            "schema": "holosoma_adaptive_timesteps_sampler_v1", "motion_time_step_total": num_clips * 100, "env_fps": 50,
            "num_bins": bins, "num_clips": num_clips, "phase_binning": True, "per_clip": True, "sampling_policy": dict(LEGACY_POLICY),
            "bin_failed_count": torch.zeros(num_clips, bins), "current_bin_failed_count": torch.zeros(num_clips, bins),
            "hero_clip_prior": torch.full((num_clips,), 1.0 / num_clips), "hero_num_motions": num_clips, "hero_source_tags": ["amass"]}}}


def sampler_from(mc, num_clips=2000):
    """``HeroMotionCommand._build_sampler`` on a ``num_clips``-clip single-source registry."""
    ids = torch.zeros(num_clips, dtype=torch.long)
    return HeroAdaptiveTimestepsSampler(
        num_clips * 100, "cpu", 50, phase_binning=mc.adaptive_sampler_phase_binning, max_clip_time_step=100,
        per_clip=mc.adaptive_sampler_per_clip, num_clips=num_clips, adaptive_uniform_ratio=mc.adaptive_sampler_uniform_ratio,
        adaptive_clip_temperature=mc.adaptive_sampler_clip_temperature, adaptive_clip_max_probability=mc.adaptive_sampler_clip_max_probability,
        clip_cap_relative=mc.adaptive_sampler_clip_max_relative, clip_prior=build_clip_prior(ids, ["amass"], {"amass": 1.0}, "cpu"),
        source_tag_ids=ids, source_tags=["amass"])


def test_export_builds_an_inference_config_a_pre_fix_checkpoint_can_load(tmp_path):
    cfg = DEFAULTS["without_delta_anchor"]
    raw = pre_fix_checkpoint(cfg)
    export = load_script("export")
    corpus = make_corpus(tmp_path / "corpus")
    # the checkpoint's own motion config is refused by today's dataclass (absolute cap) ...
    with pytest.raises(ValueError, match="adaptive_sampler_clip_max_relative"):
        coerce_hero_motion_config(motion_config(raw["experiment_config"]))
    # ... export's config path replaces the sampler settings before coercion and points at --motion-dir
    saved, eval_cfg = export.export_config(raw, corpus, tmp_path / "bundle")
    mc, preset = get_motion_config(eval_cfg.command), get_motion_config(cfg.command)
    assert mc.motion_dir == str(corpus) and mc.reset_sampler_on_resume is True
    assert set(INFERENCE_SAMPLER_FIELDS) >= {"use_adaptive_timesteps_sampler", "adaptive_sampler_clip_max_probability",
                                              "adaptive_sampler_clip_max_relative", "adaptive_sampler_uniform_ratio"}
    for name in INFERENCE_SAMPLER_FIELDS:
        assert getattr(mc, name) == getattr(preset, name), name
    # everything the policy was trained with stays the checkpoint's
    assert mc.clip_end_policy_by_source == {"amass": "rollover", "reach_example": "hold"}
    assert mc.source_weights == {"amass": 0.8, "ik_reach_example": 0.2}
    assert eval_cfg.training.num_envs == 1 and eval_cfg.training.export_onnx and eval_cfg.training.max_eval_steps == 1
    assert eval_cfg.robot == cfg.robot and get_motion_config(saved.command) == mc
    # the environment builds its sampler from it (this is where the pre-fix config raised) ...
    sampler = sampler_from(mc)
    assert sampler.clip_cap_relative == 10.0 and sampler.adaptive_clip_max_probability == 1.0
    # ... and algo.load()'s env-state restore drops the non-loadable table with a warning instead of raising
    env = roundtrip.make_env(reset_sampler_on_resume=mc.reset_sampler_on_resume)
    with roundtrip.captured_warnings() as warnings:
        env.load_checkpoint_state(raw["env_state"])
    assert [w for w in warnings if "Restarting the adaptive sampler from zeros" in w]
    assert float(roundtrip.tracker(env).get_average()) == pytest.approx(109.94, abs=0.01)
    assert not env.command_manager.command.adaptive_timesteps_sampler.bin_failed_count.any()
    # without export's opt-in the same state is the resume error
    with pytest.raises(ValueError, match="--reset-sampler-on-resume"):
        roundtrip.make_env().load_checkpoint_state(raw["env_state"])
    # the script's dry run goes through the same path
    checkpoint = tmp_path / "model_legacy.pt"
    torch.save(raw, checkpoint)
    assert export.main(["--checkpoint", str(checkpoint), "--motion-dir", str(corpus), "--output", str(tmp_path / "bundle"), "--dry-run"]) == 0
    # a checkpoint of a non-HERO configuration or another layout is still refused
    foreign = {**raw, "experiment_config": {**raw["experiment_config"], "training": {**raw["experiment_config"]["training"], "name": "other"}}}
    with pytest.raises(ValueError, match="HERO configuration"):
        export.export_config(foreign, corpus, tmp_path / "bundle")
    with pytest.raises(ValueError, match="layout differs"):
        export.export_config({**raw, "experiment_config": {**raw["experiment_config"], "observation": DEFAULTS["with_delta_anchor"].to_serializable_dict()["observation"]}},
                             corpus, tmp_path / "bundle")


def test_with_inference_motion_config_overrides_before_coercion():
    cfg = DEFAULTS["without_delta_anchor"]
    saved_cmd = DEFAULTS["without_delta_anchor"].command
    # a dict-form saved command (as checkpoints store it) with a refused value
    from dataclasses import replace
    term = saved_cmd.setup_terms["motion_command"]
    stored = replace(saved_cmd, setup_terms={"motion_command": replace(term, params={"motion_config": motion_config(pre_fix_saved_config(cfg))})})
    mc = get_motion_config(with_inference_motion_config(stored, cfg.command, motion_dir="/corpus"))
    assert mc.adaptive_sampler_clip_max_probability == 1.0 and mc.adaptive_sampler_clip_max_relative == 10.0
    assert mc.motion_dir == "/corpus" and mc.reset_sampler_on_resume and mc.clip_end_policy_by_source["reach_example"] == "hold"
    # a dataclass-form command works the same
    mc2 = get_motion_config(with_inference_motion_config(cfg.command, cfg.command, motion_dir="/corpus"))
    assert mc2.motion_dir == "/corpus" and mc2.reset_sampler_on_resume


@pytest.mark.skipif(not REAL_PRE_FIX_CHECKPOINT.is_file(), reason="the real pre-fix checkpoint is not on this machine")
def test_the_real_pre_fix_checkpoint_exports(tmp_path):
    raw = torch.load(REAL_PRE_FIX_CHECKPOINT, map_location="cpu", weights_only=False)
    export = load_script("export")
    corpus = make_corpus(tmp_path / "corpus")
    saved, eval_cfg = export.export_config(raw, corpus, tmp_path / "bundle")
    mc = get_motion_config(eval_cfg.command)
    assert mc.adaptive_sampler_clip_max_probability == 1.0 and mc.adaptive_sampler_clip_max_relative == 10.0 and mc.reset_sampler_on_resume
    sampler_from(mc)
    env = roundtrip.make_env(reset_sampler_on_resume=True)
    env.load_checkpoint_state(raw["env_state"])  # the large table written under the previous rule: dropped, not raised
    assert export.main(["--checkpoint", str(REAL_PRE_FIX_CHECKPOINT), "--motion-dir", str(corpus), "--output", str(tmp_path / "bundle"), "--dry-run"]) == 0
