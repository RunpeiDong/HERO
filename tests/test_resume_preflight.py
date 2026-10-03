"""Resume pre-flight and launcher options: configuration diff, sampler probe (policy, source mix, clip registry), curriculum
restarts (terms and the height-offset curriculum), episode-length offsets, std clamp."""
from __future__ import annotations

import os
from dataclasses import replace
import importlib.util
import inspect
import json
from pathlib import Path
import sys
from types import SimpleNamespace

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party/holosoma"), str(ROOT / "scripts")]
from configs import DEFAULTS
from hero_isaacsim.config_values.command import with_motion_config
from hero_isaacsim.config_values.experiment import (RESUME_ALGO_CONFIG_FIELDS, RESUME_PER_RUN_MOTION_FIELDS, ResumePreflight,
                                                    curriculum_term_initial_values, resume_config_changes,
                                                    resume_curriculum_state_problem, resume_preflight, resume_sampler_state_problem,
                                                    source_mix, stateful_curriculum_term_names, validate_checkpoint_contract)
from hero_isaacsim.envs.hero_tracking_manager import HeroTrackingManager
from hero_isaacsim.managers.command.hero import HeroMotionConfig
from hero_isaacsim.managers.command.sampler import expected_sampling_policy

MOTION_CONFIG_PATH = ("command", "setup_terms", "motion_command", "params", "motion_config")
REAL_PRE_FIX_CHECKPOINT = Path(os.environ.get("HERO_LEGACY_CHECKPOINT") or "/nonexistent/legacy_checkpoint.pt")
"""Optional: a real checkpoint written before the current sampler rule, curriculum persistence and the IK tag rename (HERO_LEGACY_CHECKPOINT=...)."""
LEGACY_POLICY = {"adaptive_uniform_ratio": 0.3, "adaptive_clip_temperature": 1.0, "adaptive_clip_max_probability": 0.001}
CURRENT_POLICY = expected_sampling_policy(uniform_ratio=0.3, clip_cap_relative=10.0)
H_KEY = HeroTrackingManager.H_CURRICULUM_STATE_KEY
H_CURRICULUM_RESTART = "the per-environment height-offset curriculum would restart at h_curriculum_init=0.1"
PENALTY_RESTART = "would restart at their initial values (penalty_curriculum: initial_scale 0.1)"
PRE_FIX_CONFIG_CHANGES = [
    "command.motion_command.motion_config.adaptive_sampler_clip_max_probability: 0.001 -> 1.0",
    "command.motion_command.motion_config.adaptive_sampler_clip_max_relative: null -> 10.0",
    'command.motion_command.motion_config.clip_end_policy_by_source: {"amass": "rollover", "reach_example": "hold"} -> '
    '{"amass": "rollover", "ik_reach_example": "hold"}',
]


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


def saved_config(cfg):
    return cfg.to_serializable_dict()


def motion_config(serialized):
    node = serialized
    for key in MOTION_CONFIG_PATH:
        node = node[key]
    return node


def pre_fix_saved_config(cfg):
    """The saved configuration of a legacy-style checkpoint: absolute clip cap 0.001, no relative cap, no resume
    opt-in field, the old IK tag in the clip-end policies, another corpus path and source mix."""
    saved = saved_config(cfg)
    mc = motion_config(saved)
    mc["adaptive_sampler_clip_max_probability"] = 0.001
    mc.pop("adaptive_sampler_clip_max_relative")
    mc.pop("reset_sampler_on_resume")
    mc["motion_dir"] = "/data/motions/combined"
    mc["source_weights"] = {"amass": 0.8, "ik_reach_example": 0.2}
    mc["clip_end_policy_by_source"] = {"amass": "rollover", "reach_example": "hold"}
    return saved


def pre_fix_env_state():
    """Tracker + a sampler table written under the previous rule, no curriculum progress, no height-offset curriculum."""
    return {"average_episode_tracker": {"average_episode_length": torch.tensor(109.94), "suppress_next_update": False},
            "adaptive_timesteps_sampler": {"schema": "holosoma_adaptive_timesteps_sampler_v1", "sampling_policy": dict(LEGACY_POLICY)}}


def current_sampler_state(tags=("amass",), num_clips=1, policy=None):
    """The sampler block a current-code checkpoint writes for a corpus of ``num_clips`` clips over source ``tags``."""
    return {"schema": "holosoma_adaptive_timesteps_sampler_v1", "num_clips": num_clips, "sampling_policy": dict(policy or CURRENT_POLICY),
            "hero_num_motions": num_clips, "hero_source_tags": list(tags)}


def current_env_state(*, tags=("amass",), num_clips=1, num_envs=8):
    """Everything a current-code checkpoint persists: tracker, curriculum progress, sampler table, height-offset curriculum."""
    return {"average_episode_tracker": {"average_episode_length": torch.tensor(180.0), "suppress_next_update": False},
            "curriculum_terms": {"penalty_curriculum": {"current_scale": 0.05}},
            "adaptive_timesteps_sampler": current_sampler_state(tags, num_clips),
            H_KEY: torch.full((num_envs,), 0.3)}


def with_weights(cfg, weights):
    return replace(cfg, command=with_motion_config(cfg.command, source_weights=weights))


def refusal_for(checkpoint, cfg, **flags):
    flags = {"allow_config_change": False, "reset_sampler_on_resume": False, **flags}
    return resume_preflight(checkpoint, cfg).refusal(**flags)


@pytest.mark.parametrize("name", tuple(DEFAULTS))
def test_an_unchanged_configuration_resumes_without_findings(name):
    cfg = DEFAULTS[name]
    assert resume_config_changes(saved_config(cfg), cfg) == []
    assert resume_sampler_state_problem({"experiment_config": saved_config(cfg)}, cfg) is None
    assert resume_curriculum_state_problem({"experiment_config": saved_config(cfg)}, cfg) is None
    preflight = resume_preflight({"experiment_config": saved_config(cfg)}, cfg)
    assert preflight == ResumePreflight([], None, None)
    assert preflight.refusal(allow_config_change=False, reset_sampler_on_resume=False) is None
    # a current-code checkpoint with its complete env_state (terms, sampler table, height-offset curriculum) is clean too
    checkpoint = {"experiment_config": saved_config(cfg), "env_state": current_env_state()}
    assert resume_preflight(checkpoint, cfg, corpus_num_clips=1, corpus_source_tags=("amass",)) == ResumePreflight([], None, None)


def test_a_sampler_cap_change_is_named_and_gated_by_the_flag():
    cfg = DEFAULTS["without_delta_anchor"]
    saved = saved_config(cfg)
    motion_config(saved)["adaptive_sampler_clip_max_relative"] = 3.0
    checkpoint = {"experiment_config": saved}
    change = "command.motion_command.motion_config.adaptive_sampler_clip_max_relative: 3.0 -> 10.0"
    assert resume_config_changes(saved, cfg) == [change]
    message = refusal_for(checkpoint, cfg)
    assert change in message and message.endswith("Pass --allow-config-change to continue.")
    assert refusal_for(checkpoint, cfg, allow_config_change=True) is None
    # the layout-only check (export) ignores training settings and is the only entry point without the resume findings
    assert validate_checkpoint_contract(checkpoint, cfg) is None
    assert "resume" not in inspect.signature(validate_checkpoint_contract).parameters
    # observation and algorithm mismatches stay hard errors whatever the flag
    with pytest.raises(ValueError, match="layout differs"):
        resume_preflight(checkpoint, DEFAULTS["with_delta_anchor"])
    with pytest.raises(ValueError, match="trained with"):
        resume_preflight(checkpoint, DEFAULTS["without_delta_anchor_single"])


def test_reward_curriculum_termination_and_randomization_changes_are_named():
    cfg = DEFAULTS["without_delta_anchor"]
    saved = saved_config(cfg)
    reward_name = next(n for n, t in saved["reward"]["terms"].items() if t["weight"])
    saved["reward"]["terms"][reward_name]["weight"] *= 2.0
    saved["reward"]["terms"].pop(next(n for n in saved["reward"]["terms"] if n != reward_name))
    saved["curriculum"]["setup_terms"]["penalty_curriculum"]["params"]["degree"] = 2e-5
    saved["curriculum"]["params"]["num_compute_average_epl"] = 500
    saved["termination"]["terms"]["gravity_tilt"]["params"]["threshold_x"] = 0.9
    saved["randomization"]["setup_terms"]["push_randomizer_state"]["params"]["enabled"] = False
    saved["randomization"]["setup_terms"]["extra_term"] = {"func": "x:y", "params": {}}
    changes = resume_config_changes(saved, cfg)
    prefixes = [c.split(":")[0] for c in changes]
    assert f"reward.terms.{reward_name}.weight" in prefixes
    assert any(p.startswith("reward.terms.") and "absent in the selected config" not in p for p in prefixes)
    assert "curriculum.setup_terms.penalty_curriculum.params" in prefixes
    assert "curriculum.params" in prefixes
    assert "termination.terms.gravity_tilt.params" in prefixes
    assert "randomization.setup_terms.push_randomizer_state.params" in prefixes
    assert any(c.startswith("randomization.setup_terms.extra_term: present in checkpoint, absent") for c in changes)
    assert any("absent in checkpoint, present in the selected config" in c and c.startswith("reward.terms.") for c in changes)
    assert f"reward.terms.{reward_name}.weight" in refusal_for({"experiment_config": saved}, cfg)
    # per-run plumbing is not a training-setting change: a different source mix or motion directory (every resume has
    # its own corpus path), the resume opt-in itself, where the corpus lives
    saved = saved_config(cfg)
    mc = motion_config(saved)
    mc["motion_dir"], mc["motion_file"] = "/elsewhere", "/elsewhere/clip.npz"
    mc["source_weights"] = {"amass": 0.6, "ik_reach_example": 0.4}
    mc["source_weights_from_manifest"] = True
    mc["reset_sampler_on_resume"] = True
    mc["motion_storage_device"], mc["shared_cache_dir"] = "cuda", "/cache"
    assert set(mc) >= RESUME_PER_RUN_MOTION_FIELDS
    assert resume_config_changes(saved, cfg) == []


HERO_MOTION_FIELDS = {f.name for f in HeroMotionConfig.__dataclass_fields__.values()}


@pytest.mark.parametrize("field, value", [
    ("adaptive_sampler_uniform_ratio", 0.1), ("use_adaptive_timesteps_sampler", False),
    ("clip_end_policy_by_source", {"amass": "hold"}), ("default_clip_end_policy", "hold"), ("rollover_at_clip_end", False),
    ("h_curriculum_up", 0.05), ("h_curriculum_up_threshold", 999), ("h_curriculum_init", 0.9), ("h_curriculum_range", [0.0, 0.5]),
    ("h_offset_range", [-0.5, 0.0]), ("h_offset_from_clip", True),
    ("walk_prob", 0.2), ("fix_upper_body_prob", 0.0), ("command_resample_time_s", 2.0), ("vel_cmd_lin_range", [-0.5, 0.5]),
    ("noise_to_initial_pose", None), ("body_names_to_track", ["pelvis"]), ("ref_lookahead_frames", 0),
])
def test_every_motion_setting_except_per_run_plumbing_is_compared(field, value):
    """The P2 clip-end fix, the height-offset curriculum and the command sampling all change what the policy is trained
    on; each of them must be a finding (previously only ``adaptive_sampler_*`` was compared)."""
    cfg = DEFAULTS["without_delta_anchor"]
    assert field in HERO_MOTION_FIELDS and field not in RESUME_PER_RUN_MOTION_FIELDS
    assert RESUME_PER_RUN_MOTION_FIELDS <= HERO_MOTION_FIELDS
    saved = saved_config(cfg)
    assert motion_config(saved)[field] != value
    motion_config(saved)[field] = value
    changes = resume_config_changes(saved, cfg)
    assert [c.split(":")[0] for c in changes] == [f"command.motion_command.motion_config.{field}"], changes
    assert field in refusal_for({"experiment_config": saved}, cfg)


def test_a_pre_fix_checkpoint_reports_the_clip_end_policy_change_next_to_the_sampler_caps():
    """Legacy-style saved config vs the live default: the two sampler-cap changes AND the clip-end-policy change
    (the IK tag rename switches ~27 % of the clips of that corpus from rollover to hold); the different corpus path,
    source mix and missing opt-in field are not findings."""
    cfg = DEFAULTS["without_delta_anchor"]
    assert resume_config_changes(pre_fix_saved_config(cfg), cfg) == PRE_FIX_CONFIG_CHANGES
    if REAL_PRE_FIX_CHECKPOINT.is_file():
        real = torch.load(REAL_PRE_FIX_CHECKPOINT, map_location="cpu", weights_only=False)
        assert resume_config_changes(real["experiment_config"], cfg) == PRE_FIX_CONFIG_CHANGES


def test_missing_curriculum_progress_is_a_finding_with_the_restart_value():
    cfg = DEFAULTS["without_delta_anchor"]
    assert stateful_curriculum_term_names(cfg) == ["penalty_curriculum"]
    assert curriculum_term_initial_values(cfg, "penalty_curriculum") == "initial_scale 0.1"
    assert curriculum_term_initial_values(cfg, "no_such_term") == "its initial value"
    saved = saved_config(cfg)
    problem = resume_curriculum_state_problem({"experiment_config": saved, "env_state": pre_fix_env_state()}, cfg)
    assert problem and "no 'curriculum_terms' (written before curriculum progress was checkpointed)" in problem
    assert f"the curriculum terms ['penalty_curriculum'] {PENALTY_RESTART}" in problem
    assert "look continuous" not in problem  # the logged penalty_scale shows the step; say where it restarts instead
    partial = {**pre_fix_env_state(), "curriculum_terms": {}}
    problem = resume_curriculum_state_problem({"experiment_config": saved, "env_state": partial}, cfg)
    assert problem and "entry for ['penalty_curriculum']" in problem and PENALTY_RESTART in problem
    complete = {**pre_fix_env_state(), "curriculum_terms": {"penalty_curriculum": {"current_scale": 0.05}}, H_KEY: torch.full((8,), 0.4)}
    assert resume_curriculum_state_problem({"experiment_config": saved, "env_state": complete}, cfg) is None
    # a checkpoint without any env_state carries no partial state to be misled by
    assert resume_curriculum_state_problem({"experiment_config": saved}, cfg) is None
    if REAL_PRE_FIX_CHECKPOINT.is_file():
        real = torch.load(REAL_PRE_FIX_CHECKPOINT, map_location="cpu", weights_only=False)
        assert PENALTY_RESTART in (resume_curriculum_state_problem(real, cfg) or "")


def test_a_missing_height_offset_curriculum_is_a_finding():
    """A pre-fix checkpoint (tracker + sampler table, no hero_h_curriculum_scale) restarts every environment's height-offset
    scale at h_curriculum_init = 0.1 while the h_offset_range stays (-0.25, 0): the pre-flight names it next to the curriculum
    terms, under the same flag; a current-code checkpoint is clean."""
    cfg = DEFAULTS["without_delta_anchor"]
    saved = saved_config(cfg)
    problem = resume_curriculum_state_problem({"experiment_config": saved, "env_state": pre_fix_env_state()}, cfg)
    assert f"the checkpoint's env_state has no '{H_KEY}' (written before the height-offset curriculum was checkpointed); {H_CURRICULUM_RESTART}" in problem
    assert problem.index("penalty_curriculum") < problem.index(H_KEY)
    # curriculum progress present, height-offset curriculum absent: only the latter is reported
    terms_only = {**pre_fix_env_state(), "curriculum_terms": {"penalty_curriculum": {"current_scale": 0.05}}}
    problem = resume_curriculum_state_problem({"experiment_config": saved, "env_state": terms_only}, cfg)
    assert problem.startswith(f"the checkpoint's env_state has no '{H_KEY}'") and "penalty_curriculum" not in problem
    # the reverse: the height-offset curriculum present, no curriculum progress
    h_only = {**pre_fix_env_state(), H_KEY: torch.full((8,), 0.4)}
    problem = resume_curriculum_state_problem({"experiment_config": saved, "env_state": h_only}, cfg)
    assert PENALTY_RESTART in problem and H_KEY not in problem
    assert resume_curriculum_state_problem({"experiment_config": saved, "env_state": current_env_state()}, cfg) is None
    # gated by --allow-config-change like the curriculum terms
    checkpoint = {"experiment_config": saved, "env_state": {**current_env_state()}}
    checkpoint["env_state"].pop(H_KEY)
    message = refusal_for(checkpoint, cfg)
    assert H_CURRICULUM_RESTART in message and message.endswith("Pass --allow-config-change to continue.")
    assert refusal_for(checkpoint, cfg, allow_config_change=True) is None
    if REAL_PRE_FIX_CHECKPOINT.is_file():
        real = torch.load(REAL_PRE_FIX_CHECKPOINT, map_location="cpu", weights_only=False)
        assert H_KEY not in real["env_state"]
        problem = resume_curriculum_state_problem(real, cfg)
        assert PENALTY_RESTART in problem and H_CURRICULUM_RESTART in problem


def test_the_combined_refusal_names_every_flag_once():
    cfg = DEFAULTS["without_delta_anchor"]
    checkpoint = {"experiment_config": pre_fix_saved_config(cfg), "env_state": pre_fix_env_state()}
    preflight = resume_preflight(checkpoint, cfg)
    assert preflight.config_changes == PRE_FIX_CONFIG_CHANGES
    assert PENALTY_RESTART in preflight.curriculum_state_problem and H_CURRICULUM_RESTART in preflight.curriculum_state_problem
    assert "semantics" in preflight.sampler_state_problem
    message = preflight.refusal(allow_config_change=False, reset_sampler_on_resume=False)
    assert message.startswith("Resuming from this checkpoint needs explicit opt-in:\n- Checkpoint training settings differ")
    assert all(change in message for change in PRE_FIX_CONFIG_CHANGES)
    assert H_CURRICULUM_RESTART in message and PENALTY_RESTART in message
    assert message.count("--allow-config-change") == 3 and message.count("--reset-sampler-on-resume") == 2
    assert message.endswith("Pass --allow-config-change and --reset-sampler-on-resume to continue.")
    only_sampler = preflight.refusal(allow_config_change=True, reset_sampler_on_resume=False)
    assert "training settings" not in only_sampler and "penalty_curriculum" not in only_sampler and H_KEY not in only_sampler
    assert only_sampler.endswith("Pass --reset-sampler-on-resume to continue.")
    only_config = preflight.refusal(allow_config_change=False, reset_sampler_on_resume=True)
    assert "semantics" not in only_config and only_config.endswith("Pass --allow-config-change to continue.")
    assert preflight.refusal(allow_config_change=True, reset_sampler_on_resume=True) is None
    # layout / algorithm mismatches stay hard errors ahead of the opt-in findings
    with pytest.raises(ValueError, match="layout differs"):
        resume_preflight(checkpoint, DEFAULTS["with_delta_anchor"])


def test_sampler_probe_compares_the_saved_policy_with_the_selected_config():
    cfg = DEFAULTS["without_delta_anchor"]
    live = expected_sampling_policy(uniform_ratio=0.3, clip_cap_relative=10.0)
    assert resume_sampler_state_problem({"experiment_config": saved_config(cfg), "env_state": {"adaptive_timesteps_sampler": {"sampling_policy": live}}}, cfg) is None
    legacy = {"adaptive_uniform_ratio": 0.3, "adaptive_clip_temperature": 1.0, "adaptive_clip_max_probability": 0.001}
    problem = resume_sampler_state_problem({"experiment_config": saved_config(cfg), "env_state": {"adaptive_timesteps_sampler": {"sampling_policy": legacy}}}, cfg)
    assert problem and all(k in problem for k in ("adaptive_clip_max_probability", "clip_cap_relative", "semantics"))
    assert resume_sampler_state_problem({"experiment_config": saved_config(cfg), "env_state": {"average_episode_tracker": {}}}, cfg) is None


def test_source_mix_normalises_like_the_sampler_prior():
    tags = ("amass", "ik_reach_example")
    assert source_mix({}, tags) is None and source_mix(None, tags) is None  # uniform over clips
    assert source_mix({"amass": 0.7, "ik_reach_example": 0.3}, tags) == pytest.approx({"amass": 0.7, "ik_reach_example": 0.3})
    # unnormalised weights, another key order and a tag the corpus lacks spell the same mix; a corpus tag without an entry gets 0
    assert source_mix({"ik_reach_example": 3, "amass": 7, "boxes": 5}, tags) == pytest.approx({"amass": 0.7, "ik_reach_example": 0.3})
    assert source_mix({"amass": 2.0}, tags) == {"amass": 1.0, "ik_reach_example": 0.0}
    assert list(source_mix({"ik_reach_example": 1, "amass": 1}, tags)) == ["amass", "ik_reach_example"]
    assert source_mix({"boxes": 1.0}, tags) == {"amass": 0.0, "ik_reach_example": 0.0}  # zero mass: never matches a table


def test_a_changed_source_mix_or_clip_registry_is_a_sampler_finding():
    """The saved table's clip prior is the checkpoint's source_weights over its corpus; the live prior is the manifest's over the
    live corpus. A changed mix is not a configuration change (RESUME_PER_RUN_MOTION_FIELDS) but the table cannot be restored
    under it -- reported before Isaac Sim starts, under --reset-sampler-on-resume, instead of failing in load_checkpoint_state."""
    cfg = DEFAULTS["without_delta_anchor"]
    tags = ("amass", "ik_reach_example")
    saved_cfg = with_weights(cfg, {"amass": 0.7, "ik_reach_example": 0.3})
    checkpoint = {"experiment_config": saved_config(saved_cfg), "env_state": current_env_state(tags=tags, num_clips=36)}
    # the same mix over the same registry: nothing to report, with or without the corpus census
    same = with_weights(cfg, {"amass": 0.7, "ik_reach_example": 0.3})
    assert resume_sampler_state_problem(checkpoint, same) is None
    assert resume_sampler_state_problem(checkpoint, same, corpus_num_clips=36, corpus_source_tags=tags) is None
    assert resume_preflight(checkpoint, same, corpus_num_clips=36, corpus_source_tags=tags) == ResumePreflight([], None, None)
    # identical weights spelled differently (unnormalised, other order, a tag the corpus lacks) normalise equal: no finding
    spelled = with_weights(cfg, {"ik_reach_example": 3, "amass": 7, "boxes": 5})
    assert resume_sampler_state_problem(checkpoint, spelled, corpus_num_clips=36, corpus_source_tags=tags) is None
    assert resume_config_changes(saved_config(saved_cfg), spelled) == []
    # a changed mix: a sampler finding naming both mixes and the clip counts, not a configuration change
    changed = with_weights(cfg, {"amass": 0.5, "ik_reach_example": 0.5})
    assert resume_config_changes(saved_config(saved_cfg), changed) == []
    expected = ("the saved failure table was built for source mix {amass: 0.7, ik_reach_example: 0.3} / 36 clips; this run uses "
                "source mix {amass: 0.5, ik_reach_example: 0.5} / 36 clips -> it cannot be restored")
    assert resume_sampler_state_problem(checkpoint, changed, corpus_num_clips=36, corpus_source_tags=tags) == expected
    preflight = resume_preflight(checkpoint, changed, corpus_num_clips=36, corpus_source_tags=tags)
    assert preflight == ResumePreflight([], None, expected)
    message = preflight.refusal(allow_config_change=False, reset_sampler_on_resume=False)
    assert expected in message and "--allow-config-change" not in message and message.endswith("Pass --reset-sampler-on-resume to continue.")
    assert preflight.refusal(allow_config_change=False, reset_sampler_on_resume=True) is None
    # without a census the mix is still compared, over the saved tags (the saved clip count is known, the live one is not)
    assert resume_sampler_state_problem(checkpoint, changed) == expected.replace("0.5} / 36 clips", "0.5}")
    # a changed clip count (clips added or removed under the same mix)
    problem = resume_sampler_state_problem(checkpoint, same, corpus_num_clips=37, corpus_source_tags=tags)
    assert problem == expected.replace("source mix {amass: 0.5, ik_reach_example: 0.5} / 36 clips", "source mix {amass: 0.7, ik_reach_example: 0.3} / 37 clips")
    # another set of source tags: the live mix is the live weights over the live corpus
    problem = resume_sampler_state_problem(checkpoint, same, corpus_num_clips=36, corpus_source_tags=("amass", "lafan"))
    assert problem and "this run uses source mix {amass: 1, lafan: 0} / 36 clips" in problem
    # uniform mixes (no source_weights) show their tags explicitly when the tag sets differ
    uniform = {"experiment_config": saved_config(cfg), "env_state": current_env_state(tags=tags, num_clips=36)}
    assert resume_sampler_state_problem(uniform, cfg, corpus_num_clips=36, corpus_source_tags=tags) is None
    problem = resume_sampler_state_problem(uniform, cfg, corpus_num_clips=36, corpus_source_tags=("amass",))
    assert problem == ("the saved failure table was built for source mix uniform over clips / source tags ['amass', 'ik_reach_example'] / 36 clips; "
                       "this run uses source mix uniform over clips / source tags ['amass'] / 36 clips -> it cannot be restored")
    # a uniform table resumed under a weighted mix (and vice versa) differs
    assert "uniform over clips" in resume_sampler_state_problem(uniform, same, corpus_num_clips=36, corpus_source_tags=tags)
    assert "uniform over clips" in resume_sampler_state_problem(checkpoint, cfg, corpus_num_clips=36, corpus_source_tags=tags)
    # a table without the HERO registry entries (not written by the HERO sampler) cannot be compared on its registry
    stock = {**checkpoint, "env_state": {**current_env_state(), "adaptive_timesteps_sampler": {"sampling_policy": dict(CURRENT_POLICY)}}}
    assert resume_sampler_state_problem(stock, changed, corpus_num_clips=99, corpus_source_tags=("amass",)) is None
    # both sampler problems of one checkpoint are reported together under the one flag
    legacy = {**checkpoint, "env_state": {**checkpoint["env_state"],
                                          "adaptive_timesteps_sampler": current_sampler_state(tags, 36, policy=LEGACY_POLICY)}}
    problem = resume_sampler_state_problem(legacy, changed, corpus_num_clips=36, corpus_source_tags=tags)
    assert problem.startswith("adaptive sampler checkpoint sampling policy differs") and problem.endswith(expected)
    message = resume_preflight(legacy, changed, corpus_num_clips=36, corpus_source_tags=tags).refusal(allow_config_change=False, reset_sampler_on_resume=False)
    assert message.count("--reset-sampler-on-resume") == 2 and message.endswith("Pass --reset-sampler-on-resume to continue.")


def make_mixed_corpus(tmp_path, weights):
    tmp_path.mkdir(parents=True, exist_ok=True)
    for tag in weights:
        np.savez(tmp_path / f"{tag}.npz", source_tag=np.asarray(tag), has_object=np.asarray(False))
    (tmp_path / "CORPUS_MANIFEST.json").write_text(json.dumps({"source_weights": weights, "source_tags": list(weights)}))
    return tmp_path


def test_train_launcher_reports_a_changed_source_mix_before_isaac_sim(tmp_path, capsys):
    """A current-code checkpoint resumed on a corpus with another manifest mix / clip count: ONE refusal naming
    --reset-sampler-on-resume (previously zero findings, then the sampler ValueError inside Isaac Sim)."""
    train = load_script("train")
    corpus_a = make_corpus(tmp_path / "a")  # {amass: 1.0}, 1 clip
    corpus_b = make_mixed_corpus(tmp_path / "b", {"amass": 0.5, "ik_reach_example": 0.5})  # 2 clips
    saved = saved_config(train.build_config(train.parser().parse_args(["--motion-dir", str(corpus_a)])))
    assert motion_config(saved)["source_weights"] == {"amass": 1.0}
    checkpoint = tmp_path / "current.pt"
    torch.save({"experiment_config": saved, "iter": 10, "env_state": current_env_state(tags=("amass",), num_clips=1, num_envs=4096)}, checkpoint)
    # the same corpus: no finding, the census is printed
    assert train.main(["--motion-dir", str(corpus_a), "--checkpoint", str(checkpoint), "--dry-run"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["config_changes"] == [] and report["curriculum_state_reset"] is None and report["sampler_reset"] is None
    assert report["corpus"] == {"num_clips": 1, "source_tags": ["amass"]}
    # another mix and clip count: refused once, with the sampler flag only (the mix is not a configuration change)
    base_b = ["--motion-dir", str(corpus_b), "--allow-mixed-data", "--checkpoint", str(checkpoint), "--dry-run"]
    with pytest.raises(SystemExit):
        train.main(base_b)
    err = capsys.readouterr().err
    assert ("the saved failure table was built for source mix {amass: 1} / 1 clips; this run uses "
            "source mix {amass: 0.5, ik_reach_example: 0.5} / 2 clips -> it cannot be restored") in err
    body = err.split("error: ", 1)[1]  # argparse's usage header lists every flag; the refusal must not
    assert "--allow-config-change" not in body and body.rstrip().endswith("Pass --reset-sampler-on-resume to continue.")
    assert train.main([*base_b, "--reset-sampler-on-resume"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert "cannot be restored" in report["sampler_reset"] and report["config_changes"] == [] and report["curriculum_state_reset"] is None
    assert report["corpus"] == {"num_clips": 2, "source_tags": ["amass", "ik_reach_example"]}


def test_train_launcher_gates_resume_on_both_flags(tmp_path, capsys):
    train = load_script("train")
    corpus = make_corpus(tmp_path / "corpus")
    cfg = DEFAULTS["without_delta_anchor"]
    base = ["--motion-dir", str(corpus), "--dry-run"]
    progress = {"penalty_curriculum": {"current_scale": 0.05}}

    changed = saved_config(cfg)
    motion_config(changed)["adaptive_sampler_clip_max_relative"] = 3.0
    ck_changed = tmp_path / "changed.pt"
    torch.save({"experiment_config": changed, "iter": 10}, ck_changed)
    with pytest.raises(SystemExit):
        train.main([*base, "--checkpoint", str(ck_changed)])
    err = capsys.readouterr().err
    assert "adaptive_sampler_clip_max_relative" in err and err.rstrip().endswith("Pass --allow-config-change to continue.")
    assert train.main([*base, "--checkpoint", str(ck_changed), "--allow-config-change"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["config_changes"] == ["command.motion_command.motion_config.adaptive_sampler_clip_max_relative: 3.0 -> 10.0"]
    assert report["curriculum_state_reset"] is None and report["sampler_reset"] is None

    legacy_policy = {"adaptive_uniform_ratio": 0.3, "adaptive_clip_temperature": 1.0, "adaptive_clip_max_probability": 1.0}
    ck_sampler = tmp_path / "sampler.pt"
    torch.save({"experiment_config": saved_config(cfg), "iter": 10,
                "env_state": {"adaptive_timesteps_sampler": {"sampling_policy": legacy_policy}, "curriculum_terms": progress,
                              H_KEY: torch.full((4096,), 0.3)}}, ck_sampler)
    with pytest.raises(SystemExit):
        train.main([*base, "--checkpoint", str(ck_sampler)])
    err = capsys.readouterr().err
    assert "semantics" in err and err.rstrip().endswith("Pass --reset-sampler-on-resume to continue.")
    assert train.main([*base, "--checkpoint", str(ck_sampler), "--reset-sampler-on-resume"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert "semantics" in report["sampler_reset"] and report["config_changes"] == [] and report["curriculum_state_reset"] is None

    # a pre-fix checkpoint needs both flags: ONE refusal names every finding and both flags
    ck_legacy = tmp_path / "legacy.pt"
    torch.save({"experiment_config": pre_fix_saved_config(cfg), "iter": 7500, "env_state": pre_fix_env_state()}, ck_legacy)
    with pytest.raises(SystemExit):
        train.main([*base, "--checkpoint", str(ck_legacy)])
    err = capsys.readouterr().err
    assert all(change in err for change in PRE_FIX_CONFIG_CHANGES) and PENALTY_RESTART in err and H_CURRICULUM_RESTART in err and "semantics" in err
    assert err.rstrip().endswith("Pass --allow-config-change and --reset-sampler-on-resume to continue.")
    with pytest.raises(SystemExit):
        train.main([*base, "--checkpoint", str(ck_legacy), "--allow-config-change"])
    err = capsys.readouterr().err
    assert "training settings" not in err and H_KEY not in err and err.rstrip().endswith("Pass --reset-sampler-on-resume to continue.")
    assert train.main([*base, "--checkpoint", str(ck_legacy), "--allow-config-change", "--reset-sampler-on-resume"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["config_changes"] == PRE_FIX_CONFIG_CHANGES
    assert PENALTY_RESTART in report["curriculum_state_reset"] and H_CURRICULUM_RESTART in report["curriculum_state_reset"]
    assert "semantics" in report["sampler_reset"]

    # the opt-ins are meaningless without a checkpoint and are refused (nothing is baked into a fresh run's config)
    for flags in (["--allow-config-change"], ["--reset-sampler-on-resume"], ["--allow-config-change", "--reset-sampler-on-resume"]):
        with pytest.raises(SystemExit):
            train.main([*base, *flags])
        assert "--checkpoint" in capsys.readouterr().err
    args = train.parser().parse_args(["--motion-dir", str(corpus), "--checkpoint", str(ck_sampler), "--reset-sampler-on-resume"])
    built = train.build_config(args)
    assert built.command.setup_terms["motion_command"].params["motion_config"].reset_sampler_on_resume is True
    assert train.build_config(train.parser().parse_args(["--motion-dir", str(corpus)])).command.setup_terms["motion_command"].params["motion_config"].reset_sampler_on_resume is False


@pytest.mark.skipif(not REAL_PRE_FIX_CHECKPOINT.is_file(), reason="the real pre-fix checkpoint is not on this machine")
def test_the_real_pre_fix_checkpoint_is_refused_once_with_both_flags_named(tmp_path, capsys):
    train = load_script("train")
    corpus = make_corpus(tmp_path / "corpus")
    base = ["--motion-dir", str(corpus), "--dry-run", "--checkpoint", str(REAL_PRE_FIX_CHECKPOINT)]
    with pytest.raises(SystemExit):
        train.main(base)
    err = capsys.readouterr().err
    assert all(change in err for change in PRE_FIX_CONFIG_CHANGES)
    assert PENALTY_RESTART in err and H_CURRICULUM_RESTART in err and "'semantics': (None, 'source_conditional_v1')" in err
    # the checkpoint's large multi-source table vs this one-clip corpus is the registry half of the sampler finding
    assert "/ 29548 clips; this run uses source mix {amass: 1} / 1 clips -> it cannot be restored" in err
    assert err.rstrip().endswith("Pass --allow-config-change and --reset-sampler-on-resume to continue.")
    assert train.main([*base, "--allow-config-change", "--reset-sampler-on-resume"]) == 0
    report = json.loads(capsys.readouterr().out)
    assert report["config_changes"] == PRE_FIX_CONFIG_CHANGES
    assert PENALTY_RESTART in report["curriculum_state_reset"] and H_CURRICULUM_RESTART in report["curriculum_state_reset"]
    assert "clip_cap_relative" in report["sampler_reset"] and "29548 clips" in report["sampler_reset"]


def test_random_episode_length_offsets_apply_to_fresh_starts_only():
    from holosoma.agents.ppo.ppo import PPO
    from hero_isaacsim.agents.ppo_dual.ppo_dual import PPODual
    from hero_isaacsim.agents.ppo_single.ppo_single import PPOSingle

    def agent(cls, iteration, randomize=True):
        a = cls.__new__(cls)
        a.config = SimpleNamespace(init_at_random_ep_len=randomize)
        a.current_learning_iteration = iteration
        a.env = SimpleNamespace(episode_length_buf=torch.zeros(64, dtype=torch.long), max_episode_length=500)
        return a

    torch.manual_seed(0)
    expected = torch.randint_like(torch.zeros(64, dtype=torch.long), high=500)
    assert int(expected.max()) > 0
    for cls in (PPO, PPODual, PPOSingle):
        fresh = agent(cls, 0)
        torch.manual_seed(0)
        fresh._randomize_initial_episode_lengths()
        assert torch.equal(fresh.env.episode_length_buf, expected), cls  # fresh start: unchanged draw
        resumed = agent(cls, 7501)
        resumed._randomize_initial_episode_lengths()
        assert not resumed.env.episode_length_buf.any(), cls  # resume: counters stay at zero
        disabled = agent(cls, 0, randomize=False)
        disabled._randomize_initial_episode_lengths()
        assert not disabled.env.episode_length_buf.any(), cls
    source = inspect.getsource(PPO.learn)
    assert "_randomize_initial_episode_lengths()" in source and "randint_like" not in source


def test_std_clamp_max_flag_reaches_the_dual_actor_knob(tmp_path, monkeypatch, capsys):
    from hero_isaacsim.agents.ppo_dual.ppo_dual import PPODual, STD_CLAMP_MAX_CONFIG_FIELD

    monkeypatch.delenv("HERO_PPO_DUAL_STD_CLAMP_MAX", raising=False)
    train = load_script("train")
    corpus = make_corpus(tmp_path / "corpus")
    parse = lambda *extra: train.parser().parse_args(["--motion-dir", str(corpus), *extra])
    clamped = train.build_config(parse("--std-clamp-max", "0.5"))
    default = train.build_config(parse())
    assert STD_CLAMP_MAX_CONFIG_FIELD == "hero_std_clamp_max"
    assert clamped.algo.config.hero_std_clamp_max == 0.5 and default.algo.config.hero_std_clamp_max is None

    def resolve(config):
        algo = PPODual.__new__(PPODual)
        algo.config, algo._std_clamp_max_override, algo.hero_knob_sources = config, None, {}
        return algo._resolve_std_clamp_max(), algo.hero_knob_sources["std_clamp_max"]

    assert resolve(clamped.algo.config) == (0.5, "config.hero_std_clamp_max")
    assert resolve(default.algo.config) == (None, "class.PPODual")  # the paper recipe: no clamp
    with pytest.raises(ValueError, match="dual-actor"):
        train.build_config(parse("--config", "without_delta_anchor_single", "--std-clamp-max", "0.5"))
    with pytest.raises(ValueError, match="positive"):
        train.build_config(parse("--std-clamp-max", "-0.1"))
    # the clamp is compared on resume: adding, removing or changing it is a finding (it is not inherited from the checkpoint)
    assert RESUME_ALGO_CONFIG_FIELDS == (STD_CLAMP_MAX_CONFIG_FIELD,)
    assert resume_config_changes(saved_config(default), clamped) == ["algo.config.hero_std_clamp_max: null -> 0.5"]
    assert resume_config_changes(saved_config(clamped), default) == ["algo.config.hero_std_clamp_max: 0.5 -> null"]
    assert resume_config_changes(saved_config(clamped), clamped) == [] and resume_config_changes(saved_config(default), default) == []
    before_the_field = saved_config(default)
    before_the_field["algo"]["config"].pop("hero_std_clamp_max")  # a checkpoint written before the knob existed: no clamp
    assert resume_config_changes(before_the_field, default) == []
    assert resume_config_changes(before_the_field, clamped) == ["algo.config.hero_std_clamp_max: null -> 0.5"]
    unclamped = tmp_path / "unclamped.pt"
    torch.save({"experiment_config": saved_config(default), "iter": 10}, unclamped)
    with pytest.raises(SystemExit):
        train.main(["--motion-dir", str(corpus), "--checkpoint", str(unclamped), "--std-clamp-max", "0.5", "--dry-run"])
    err = capsys.readouterr().err
    assert "algo.config.hero_std_clamp_max: null -> 0.5" in err and err.rstrip().endswith("Pass --allow-config-change to continue.")
    assert train.main(["--motion-dir", str(corpus), "--checkpoint", str(unclamped), "--std-clamp-max", "0.5", "--dry-run", "--allow-config-change"]) == 0
    assert json.loads(capsys.readouterr().out)["config_changes"] == ["algo.config.hero_std_clamp_max: null -> 0.5"]
    assert train.main(["--motion-dir", str(corpus), "--std-clamp-max", "0.5", "--dry-run"]) == 0
    assert json.loads(capsys.readouterr().out)["std_clamp_max"] == 0.5
    readme = (ROOT / "configs/README.md").read_text()
    assert "--std-clamp-max" in readme and "--allow-config-change" in readme and "--reset-sampler-on-resume" in readme
    assert "hero_std_clamp_max" in readme and "not inherited" in readme


def test_ppo_dual_load_logs_a_removed_or_changed_std_clamp(tmp_path):
    """The in-process trace of the clamp change: a clamped checkpoint loaded without the clamp says the std is unclamped from
    here on (previously silent), a different clamp says it re-projects; the same clamp says nothing."""
    from loguru import logger
    from hero_isaacsim.agents.ppo_dual.ppo_dual import PPODual

    def agent(clamp):
        a = PPODual.__new__(PPODual)
        a.device, a.std_clamp_max, a.std_clamp_max_lower, a.empirical_normalization = "cpu", clamp, None, False
        a.hero_knob_sources = {"std_clamp_max": "class.PPODual" if clamp is None else "config.hero_std_clamp_max"}
        a.config = SimpleNamespace(load_optimizer=False)
        a.actor = SimpleNamespace(project_std_=lambda: False, per_head_clamp=False, std_clamp_by_group={})
        a.load_model_state = lambda loaded: None
        a._restore_env_state = lambda state: None
        return a

    path = tmp_path / "clamped.pt"
    torch.save({"std_clamp_max": 1.0, "std_clamp_max_lower": None}, path)
    lines: list[str] = []
    sink = logger.add(lines.append, level="INFO", format="{message}")
    try:
        agent(None).load(str(path))
        assert [l for l in lines if "checkpoint std_clamp_max=1.0 but this run has no clamp (class.PPODual)" in l and "unclamped from here on" in l]
        lines.clear()
        agent(1.0).load(str(path))
        assert not [l for l in lines if "std_clamp_max=1.0" in l]
        lines.clear()
        agent(0.5).load(str(path))
        assert [l for l in lines if "checkpoint std_clamp_max=1.0 differs from this run's 0.5" in l and "re-projected" in l]
        lines.clear()
        torch.save({"std_clamp_max": None}, path)
        agent(0.5).load(str(path))  # an unclamped checkpoint says nothing about the clamp (project_std_ reports the projection)
        assert not [l for l in lines if "std_clamp_max=" in l]
    finally:
        logger.remove(sink)
