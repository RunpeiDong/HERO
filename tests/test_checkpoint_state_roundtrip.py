"""Round trip of the training-distribution state through the real environment checkpoint hooks (CPU, mock managers).

A checkpoint must carry the curriculum terms' progress, the per-environment height-offset curriculum and the adaptive
sampler's table, and a resume must restore them -- or refuse loudly when the saved sampler table was written under a
different sampling rule."""
from __future__ import annotations

import os
from contextlib import contextmanager
from dataclasses import dataclass
import functools
import io
from pathlib import Path
import sys

from loguru import logger
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party/holosoma")]
from holosoma.envs.base_task.base_task import BaseTask
from holosoma.managers.curriculum.manager import CurriculumManager
from hero_isaacsim.config_values.command import make_hero_motion_config
from hero_isaacsim.config_values.curriculum import PENALTY_CURRICULUM_PARAMS, hero_curriculum
from hero_isaacsim.envs.hero_tracking_manager import HeroTrackingManager
from hero_isaacsim.managers.command.hero import HeroMotionCommand
from hero_isaacsim.managers.command.sampler import (SAMPLING_SEMANTICS, HeroAdaptiveTimestepsSampler, build_clip_prior,
                                                    expected_sampling_policy)

PENALTY_TERM = "penalty_action_rate"
DECAY_CALLS = 24460  # 0.1 * (1 - 1e-5) ** 24460 = 0.0783
REAL_PRE_FIX_CHECKPOINT = Path(os.environ.get("HERO_LEGACY_CHECKPOINT") or "/nonexistent/legacy_checkpoint.pt")
"""Optional: a real training checkpoint written before curriculum persistence / the current sampler rule (HERO_LEGACY_CHECKPOINT=...)."""


@functools.lru_cache(maxsize=1)
def real_pre_fix_env_state():
    return torch.load(REAL_PRE_FIX_CHECKPOINT, map_location="cpu", weights_only=False)["env_state"]


@contextmanager
def captured_warnings():
    lines: list[str] = []
    sink = logger.add(lines.append, level="WARNING", format="{message}")
    try:
        yield lines
    finally:
        logger.remove(sink)


@dataclass(frozen=True)
class TermCfg:
    weight: float
    tags: tuple = ()


class RewardManager:
    def __init__(self):
        self.cfgs = {PENALTY_TERM: TermCfg(-0.1, ("penalty_curriculum", "lower_body")), "tracking": TermCfg(1.0, ("lower_body",))}

    @property
    def active_terms(self):
        return list(self.cfgs)

    def get_term_cfg(self, name):
        return self.cfgs[name]

    def set_term_cfg(self, name, cfg):
        self.cfgs[name] = cfg


class MotionCfg:
    def __init__(self, reset_sampler_on_resume):
        self.use_adaptive_timesteps_sampler = True
        self.reset_sampler_on_resume = reset_sampler_on_resume


class MotionCommand:
    def __init__(self, sampler, num_envs, reset_sampler_on_resume):
        self.motion_cfg = MotionCfg(reset_sampler_on_resume)
        self.adaptive_timesteps_sampler = sampler
        self.h_curriculum_scale = torch.full((num_envs,), 0.1)


class CommandManager:
    def __init__(self, command):
        self.command = command

    def get_state(self, name):
        return self.command if name == "motion_command" else None

    def reset(self, env_ids):
        self.command.reset(env_ids)


class TerminationManager:
    def __init__(self):
        self.resets = []

    def reset(self, env_ids):
        self.resets.append(env_ids)


def make_sampler(relative=10.0, uniform=0.3):
    ids = torch.tensor([0] * 30 + [1] * 6)
    tags = ["amass", "ik_reach_example"]
    prior = build_clip_prior(ids, tags, {"amass": 0.7, "ik_reach_example": 0.3}, "cpu")
    return HeroAdaptiveTimestepsSampler(3600, "cpu", 50, per_clip=True, num_clips=36, max_clip_time_step=100,
                                        adaptive_uniform_ratio=uniform, adaptive_clip_max_probability=1.0, clip_cap_relative=relative,
                                        clip_prior=prior, source_tag_ids=ids, source_tags=tags)


def make_env(num_envs=8, *, sampler=None, reset_sampler_on_resume=False):
    """A HeroTrackingManager without Isaac Sim: only the managers the checkpoint hooks touch."""
    env = HeroTrackingManager.__new__(HeroTrackingManager)
    env.num_envs, env.device, env.log_dict = num_envs, "cpu", {}
    env.reward_manager = RewardManager()
    env._pending_episode_lengths = torch.zeros(num_envs, dtype=torch.long)
    env._pending_episode_update_mask = torch.zeros(num_envs, dtype=torch.bool)
    env.curriculum_manager = CurriculumManager(hero_curriculum, env, "cpu")
    env.curriculum_manager.setup()
    env.command_manager = CommandManager(MotionCommand(sampler or make_sampler(), num_envs, reset_sampler_on_resume))
    return env


def penalty(env):
    return env.curriculum_manager.get_term("penalty_curriculum")


def tracker(env):
    return env.curriculum_manager.get_term("average_episode_tracker")


def pickled(state):
    buffer = io.BytesIO()
    torch.save(state, buffer)
    buffer.seek(0)
    return torch.load(buffer, map_location="cpu", weights_only=False)


def trained_env():
    env = make_env()
    tracker(env).set_average(20.0, suppress_update=False)  # below level_down_threshold -> the penalty scale decays
    for _ in range(DECAY_CALLS):
        penalty(env).reset(None)
    env.command_manager.command.h_curriculum_scale.copy_(torch.linspace(0.0, 0.7, env.num_envs))
    sampler = env.command_manager.command.adaptive_timesteps_sampler
    sampler.bin_failed_count[3, :] = 2.0
    sampler.bin_failed_count[33, 1] = 0.5
    return env


def test_curriculum_terms_h_curriculum_and_sampler_round_trip():
    env1 = trained_env()
    expected_scale = PENALTY_CURRICULUM_PARAMS["initial_scale"] * (1.0 - PENALTY_CURRICULUM_PARAMS["degree"]) ** DECAY_CALLS
    assert penalty(env1).current_scale == pytest.approx(expected_scale, rel=1e-9)
    assert penalty(env1).current_scale == pytest.approx(0.0783, abs=5e-4)

    state = pickled(env1.get_checkpoint_state())
    assert set(state) == {"average_episode_tracker", "curriculum_terms", "adaptive_timesteps_sampler", "hero_h_curriculum_scale"}
    assert state["curriculum_terms"] == {"penalty_curriculum": {"current_scale": pytest.approx(expected_scale)}}
    assert state["adaptive_timesteps_sampler"]["sampling_policy"]["semantics"] == SAMPLING_SEMANTICS

    env2 = make_env()  # a fresh process: everything starts at its initial value
    assert penalty(env2).current_scale == 0.1 and env2.reward_manager.cfgs[PENALTY_TERM].weight == pytest.approx(-0.01)
    env2.load_checkpoint_state(state)
    env2.curriculum_manager.step()  # the per-step hook re-derives the weights from the restored scale
    assert penalty(env2).current_scale == pytest.approx(expected_scale, rel=1e-9)
    assert env2.reward_penalty_scale == pytest.approx(expected_scale, rel=1e-9)
    assert env2.reward_manager.cfgs[PENALTY_TERM].weight == pytest.approx(-0.1 * expected_scale, rel=1e-9)
    assert env2.reward_manager.cfgs["tracking"].weight == 1.0
    assert float(tracker(env2).get_average()) == pytest.approx(20.0) and tracker(env2)._suppress_next_update
    assert torch.equal(env2.command_manager.command.h_curriculum_scale, torch.linspace(0.0, 0.7, 8))
    assert torch.equal(env2.command_manager.command.adaptive_timesteps_sampler.bin_failed_count,
                       env1.command_manager.command.adaptive_timesteps_sampler.bin_failed_count)

    # a different environment count gets the saved mean everywhere
    env3 = make_env(num_envs=5)
    env3.load_checkpoint_state(state)
    assert torch.allclose(env3.command_manager.command.h_curriculum_scale, torch.full((5,), 0.35))

    # checkpoints written before these keys existed still load (tracker + sampler only) -- and say which curriculum
    # terms restart at their initial values instead of doing so silently
    env4 = make_env()
    with captured_warnings() as warnings:
        env4.load_checkpoint_state({k: v for k, v in state.items() if k in ("average_episode_tracker", "adaptive_timesteps_sampler")})
    assert penalty(env4).current_scale == 0.1 and float(tracker(env4).get_average()) == pytest.approx(20.0)
    assert [w for w in warnings if "no 'curriculum_terms'" in w and "['penalty_curriculum']" in w and "initial values" in w]
    # ... and that the per-environment height-offset curriculum restarts (the scale stays at its initial value)
    assert torch.equal(env4.command_manager.command.h_curriculum_scale, torch.full((8,), 0.1))
    assert [w for w in warnings if "no 'hero_h_curriculum_scale'" in w and "restarts at h_curriculum_init=0.1" in w]
    # the height-offset warning needs other env_state next to the missing key (an empty state is not a resume)
    with captured_warnings() as warnings:
        make_env().load_checkpoint_state({})
    assert not warnings
    # a curriculum_terms block without an entry for a live term is reported the same way
    env5 = make_env()
    with captured_warnings() as warnings:
        env5.load_checkpoint_state({**state, "curriculum_terms": {}})
    assert penalty(env5).current_scale == 0.1
    assert [w for w in warnings if "carries no state" in w and "['penalty_curriculum']" in w]
    with captured_warnings() as warnings:
        make_env().load_checkpoint_state(state)
    assert not [w for w in warnings if "initial values" in w or "hero_h_curriculum_scale" in w]

    # curriculum state for a term the live task does not run is an error, not a silent drop
    with pytest.raises(ValueError, match="does not run.*object_mass"):
        make_env().load_checkpoint_state({**state, "curriculum_terms": {"object_mass_curriculum": {"cur_hi": 2.0}}})


def legacy_sampler_state(state):
    """The sampler block as written under the previous composition rule (no relative cap / semantics entries)."""
    sampler_state = dict(state["adaptive_timesteps_sampler"])
    sampler_state["sampling_policy"] = {"adaptive_uniform_ratio": 0.3, "adaptive_clip_temperature": 1.0, "adaptive_clip_max_probability": 0.001}
    return sampler_state


def test_a_sampler_table_from_another_rule_is_refused_unless_opted_in():
    state = pickled(trained_env().get_checkpoint_state())
    legacy = {**state, "adaptive_timesteps_sampler": legacy_sampler_state(state)}

    with pytest.raises(ValueError) as excinfo:
        make_env().load_checkpoint_state(legacy)
    message = str(excinfo.value)
    for field in ("semantics", "clip_cap_relative", "adaptive_clip_max_probability", "--reset-sampler-on-resume"):
        assert field in message, message

    # a different relative cap is a different rule too
    with pytest.raises(ValueError, match="clip_cap_relative.*10.0, 5.0"):
        make_env(sampler=make_sampler(relative=5.0)).load_checkpoint_state(state)

    # the opt-in restarts the table from zeros and still restores everything else
    env = make_env(reset_sampler_on_resume=True)
    env.load_checkpoint_state(legacy)
    assert not env.command_manager.command.adaptive_timesteps_sampler.bin_failed_count.any()
    assert penalty(env).current_scale == pytest.approx(0.0783, abs=5e-4)
    assert torch.equal(env.command_manager.command.h_curriculum_scale, torch.linspace(0.0, 0.7, 8))

    # a matching table loads without the opt-in
    env = make_env()
    env.load_checkpoint_state(state)
    assert float(env.command_manager.command.adaptive_timesteps_sampler.bin_failed_count[3].sum()) == pytest.approx(2.0 * 3)


def test_sampler_state_carries_the_policy_and_the_rule_name():
    s = make_sampler()
    policy = s.state_dict()["sampling_policy"]
    assert policy == expected_sampling_policy(uniform_ratio=0.3, clip_cap_relative=10.0)
    assert policy["semantics"] == SAMPLING_SEMANTICS == "source_conditional_v1"
    assert HeroAdaptiveTimestepsSampler.sampling_policy_mismatches(policy, policy) == {}
    assert HeroAdaptiveTimestepsSampler.sampling_policy_mismatches({**policy, "adaptive_uniform_ratio": 0.3 + 1e-14}, policy) == {}
    assert set(HeroAdaptiveTimestepsSampler.sampling_policy_mismatches({}, policy)) == set(policy)
    with pytest.raises(ValueError, match="semantics"):
        make_sampler().load_state_dict({**s.state_dict(), "sampling_policy": {k: v for k, v in policy.items() if k != "semantics"}})
    with pytest.raises(ValueError, match="clip_cap_relative"):
        make_sampler(relative=3.0).load_state_dict(s.state_dict())
    # the backend's clip-marginal shaping is not used by this rule: refuse it instead of silently ignoring it
    ids = torch.tensor([0] * 4)
    with pytest.raises(ValueError, match="clip_cap_relative"):
        HeroAdaptiveTimestepsSampler(400, "cpu", 50, per_clip=True, num_clips=4, max_clip_time_step=100, adaptive_clip_max_probability=0.5,
                                     clip_prior=torch.full((4,), 0.25), source_tag_ids=ids, source_tags=["a"])


def hero_motion_command(env, num_envs):
    """The real ``HeroMotionCommand`` reduced to what the height-offset curriculum touches (no loader, no simulator)."""
    cmd = HeroMotionCommand.__new__(HeroMotionCommand)
    cmd.hero_cfg = cmd.motion_cfg = make_hero_motion_config()
    cmd._env, cmd.num_envs, cmd.device = env, num_envs, "cpu"
    cmd._in_soft_reset = False
    cmd._skip_h_curriculum_update_once = False
    cmd.adaptive_timesteps_sampler = make_sampler()
    cmd.h_curriculum_scale = torch.full((num_envs,), float(cmd.hero_cfg.h_curriculum_init))
    return cmd


def test_the_forced_reset_all_leaves_a_restored_h_curriculum_scale_alone(monkeypatch):
    """PPO calls ``env.reset_all()`` at construction and at ``learn()`` entry, after the checkpoint was loaded. That reset is
    infrastructure, not an episode end: without the skip it applied the short-episode step (-h_curriculum_down) to every
    environment, so a restored per-env scale was never the saved one. Driven through the real dispatch chain --
    ``HeroTrackingManager.reset_all`` -> ``WholeBodyTrackingManager.reset_all`` (``init_buffers(reset_adaptive_sampler=False)``,
    tracker suppression, termination reset) -> ``BaseTask.reset_all`` -> ``reset_envs_idx`` records the pending episode lengths
    and calls ``command_manager.reset`` -> ``HeroMotionCommand.reset`` -> ``_update_h_curriculum``; only ``BaseTask.reset_all``
    (the simulator writes and the zero-action step) is stubbed, and the command's simulator-facing reset steps are no-ops."""
    state = pickled(trained_env().get_checkpoint_state())
    saved = state[HeroTrackingManager.H_CURRICULUM_STATE_KEY]
    assert torch.equal(saved, torch.linspace(0.0, 0.7, 8))

    env = make_env()
    env.is_evaluating = False
    env.termination_manager = TerminationManager()
    cmd = hero_motion_command(env, env.num_envs)
    env.command_manager = CommandManager(cmd)
    for name in ("_sample_clip_and_phase", "_write_reset_states", "_align_odometry", "_resample_hero_commands",
                 "_refresh_hero_refs", "_update_clip_end_flags"):
        monkeypatch.setattr(cmd, name, lambda *args, **kwargs: None)
    env.load_checkpoint_state(state)
    assert torch.equal(cmd.h_curriculum_scale, saved)

    def base_reset_all(self):
        # BaseTask.reset_all -> reset_envs_idx: every environment's (1-step) episode length becomes pending, then the
        # command manager's reset runs; the simulator writes and the zero-action step that follow are not needed here.
        env_ids = torch.arange(self.num_envs)
        self._pending_episode_lengths[env_ids] = self.episode_length_buf[env_ids]
        self.command_manager.reset(env_ids)
        return {}

    env.episode_length_buf = torch.ones(env.num_envs, dtype=torch.long)
    monkeypatch.setattr(BaseTask, "reset_all", base_reset_all)
    env.reset_all()
    assert torch.equal(env.get_checkpoint_state()[HeroTrackingManager.H_CURRICULUM_STATE_KEY], saved)
    assert cmd._skip_h_curriculum_update_once is False  # consumed by the command reset the forced reset dispatched
    assert cmd._skip_adaptive_update_once is True and tracker(env)._suppress_next_update  # the WBT base's own suppressions
    assert env.termination_manager.resets == [None]
    # init_buffers(reset_adaptive_sampler=False) -- the first thing the forced reset does -- keeps a pending skip flag and
    # the restored scale (a fresh buffer set would restart the curriculum at h_curriculum_init)
    cmd._skip_h_curriculum_update_once = True
    cmd.init_buffers(reset_adaptive_sampler=False)
    assert cmd._skip_h_curriculum_update_once is True and torch.equal(cmd.h_curriculum_scale, saved)
    cmd._skip_h_curriculum_update_once = False
    # the next short episode is a real one
    env._pending_episode_lengths[:] = 1
    env.command_manager.reset(torch.arange(env.num_envs))
    assert torch.allclose(cmd.h_curriculum_scale, (saved - 0.01).clamp(0.0, 1.0))
    # a second forced reset is skipped again, long episodes still move the scale up afterwards
    env.reset_all()
    assert torch.allclose(cmd.h_curriculum_scale, (saved - 0.01).clamp(0.0, 1.0))
    assert cmd._skip_h_curriculum_update_once is False
    env._pending_episode_lengths[:] = 300
    env.command_manager.reset(torch.arange(env.num_envs))
    assert torch.allclose(cmd.h_curriculum_scale, ((saved - 0.01).clamp(0.0, 1.0) + 0.02).clamp(0.0, 1.0))
    # a fresh run: the forced reset leaves every environment at h_curriculum_init (previously stepped to 0.09 / 0.08)
    fresh = hero_motion_command(env, env.num_envs)
    for name in ("_sample_clip_and_phase", "_write_reset_states", "_align_odometry", "_resample_hero_commands",
                 "_refresh_hero_refs", "_update_clip_end_flags"):
        monkeypatch.setattr(fresh, name, lambda *args, **kwargs: None)
    env.command_manager = CommandManager(fresh)
    env.reset_all()
    env.reset_all()
    assert torch.equal(fresh.h_curriculum_scale, torch.full((env.num_envs,), float(fresh.hero_cfg.h_curriculum_init)))
    assert fresh.hero_cfg.h_curriculum_init == 0.1
    # evaluation pins the scale to 1 and consumes the flag too
    env.is_evaluating = True
    fresh._skip_h_curriculum_update_once = True
    fresh._update_h_curriculum(torch.arange(env.num_envs))
    assert bool((fresh.h_curriculum_scale == 1.0).all()) and fresh._skip_h_curriculum_update_once is False


@pytest.mark.skipif(not REAL_PRE_FIX_CHECKPOINT.is_file(), reason="the real pre-fix checkpoint is not on this machine")
def test_the_real_pre_fix_env_state_restores_the_tracker_and_warns_about_the_curriculum():
    """A real checkpoint written before this change set: tracker + a sampler table under the previous rule,
    no curriculum progress. Without the opt-in it is refused for the sampler; with it the tracker is restored, the
    table restarts and the penalty curriculum's silent restart is named."""
    env_state = real_pre_fix_env_state()
    assert set(env_state) == {"average_episode_tracker", "adaptive_timesteps_sampler"}
    with pytest.raises(ValueError, match="--reset-sampler-on-resume"):
        make_env().load_checkpoint_state(env_state)
    env = make_env(reset_sampler_on_resume=True)
    with captured_warnings() as warnings:
        env.load_checkpoint_state(env_state)
    assert float(tracker(env).get_average()) == pytest.approx(109.94, abs=0.01) and tracker(env)._suppress_next_update
    assert penalty(env).current_scale == PENALTY_CURRICULUM_PARAMS["initial_scale"]
    assert not env.command_manager.command.adaptive_timesteps_sampler.bin_failed_count.any()
    assert [w for w in warnings if "Restarting the adaptive sampler from zeros" in w]
    assert [w for w in warnings if "no 'curriculum_terms'" in w and "['penalty_curriculum']" in w]
    # ... and the per-environment height-offset curriculum restarts at h_curriculum_init, said out loud
    assert torch.equal(env.command_manager.command.h_curriculum_scale, torch.full((8,), 0.1))
    assert [w for w in warnings if "no 'hero_h_curriculum_scale'" in w and "restarts at h_curriculum_init=0.1" in w]
