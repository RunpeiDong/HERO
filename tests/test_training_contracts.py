"""CPU checks of HERO training and export contracts."""
from dataclasses import asdict, replace
import importlib.util
import json
from pathlib import Path
import sys

import numpy as np
import pytest
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party/holosoma"), str(ROOT / "scripts")]
from configs import DEFAULTS
from hero_isaacsim.config_values.experiment import DELTA_ANCHOR_TERMS, DELTA_EE_TERMS, observation_contract, validate_checkpoint_contract
from hero_isaacsim.agents.ppo_dual.modules import PPODualActor
from holosoma.agents.modules.ppo_modules import PPOActor

DUAL = ("without_delta_anchor", "with_delta_anchor", "without_delta_ee")
SINGLE = ("without_delta_anchor_single", "with_delta_anchor_single", "without_delta_ee_single")


def make_actor(cfg, obs_dim):
    """Random actor of the preset's architecture; returns (module, act_inference(flat term-major tensor))."""
    md = cfg.algo.config.module_dict
    assert (cfg.training.name in SINGLE) == (getattr(md, "actor", None) is not None), cfg.training.name
    if cfg.training.name in SINGLE:
        actor = PPOActor(obs_dim_dict={"actor_obs": obs_dim}, module_config_dict=md.actor, num_actions=29,
                         init_noise_std=cfg.algo.config.init_noise_std, history_length={"actor_obs": 5}).eval()
    else:
        actor = PPODualActor(obs_dim_dict={"actor_obs": obs_dim}, actor_lower_cfg=md.actor_lower, actor_upper_cfg=md.actor_upper,
                             history_length={"actor_obs": 5}, action_split=(15, 14), init_noise_std={"lower_body": 0.8, "upper_body": 0.6}).eval()
    return actor, (lambda x: actor.act_inference({"actor_obs": x}))


def load_script(name):
    spec = importlib.util.spec_from_file_location("hero_script_" + name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_delta_anchor_only_extends_observations():
    base, anchor = DEFAULTS["without_delta_anchor"], DEFAULTS["with_delta_anchor"]
    assert tuple(DEFAULTS) == DUAL + SINGLE
    assert [observation_contract(c)["actor_obs"]["dim"] for c in (base, anchor)] == [675, 950]
    assert [observation_contract(c)["critic_obs"]["dim"] for c in (base, anchor)] == [268, 323]
    for field in ("algo", "reward", "termination", "robot", "randomization", "action", "command", "curriculum", "simulator"):
        assert getattr(base, field) == getattr(anchor, field), field
    for name, group in base.observation.groups.items():
        extended = anchor.observation.groups[name]
        assert set(extended.terms) - set(group.terms) == set(DELTA_ANCHOR_TERMS)
        assert all(extended.terms[n] == term for n, term in group.terms.items())
        assert not any(n.startswith(("h14_", "h15_", "h16_", "h17_", "h18_", "h19_")) for n in extended.terms)
    assert base.simulator.config.sim.fps == 500
    assert base.simulator.config.sim.control_decimation == 10
    assert base.algo.config.num_learning_iterations == 20000
    assert base.training.num_envs == 4096
    assert base.logger.type == "disabled"


def test_delta_ee_ablation_removes_only_the_actor_feedback():
    base, ablation = DEFAULTS["without_delta_anchor"], DEFAULTS["without_delta_ee"]
    assert observation_contract(ablation)["actor_obs"]["dim"] == 585
    assert observation_contract(ablation)["critic_obs"]["dim"] == 268
    for field in ("algo", "reward", "termination", "robot", "randomization", "action", "command", "curriculum", "simulator"):
        assert getattr(base, field) == getattr(ablation, field), field
    actor_base, actor_ablation = base.observation.groups["actor_obs"], ablation.observation.groups["actor_obs"]
    assert set(actor_base.terms) - set(actor_ablation.terms) == set(DELTA_EE_TERMS) == {"h07_dif_local_rigid_body_pos_ee", "h08_dif_local_rigid_body_rot_ee"}
    assert actor_ablation.history_length == actor_base.history_length == 5
    assert all(actor_ablation.terms[n] == term for n, term in actor_base.terms.items() if n not in DELTA_EE_TERMS)
    assert ablation.observation.groups["critic_obs"] == base.observation.groups["critic_obs"]
    assert not any(n.startswith(("h20_", "h21_", "h22_", "h23_")) for n in actor_ablation.terms)
    assert observation_contract(DEFAULTS["without_delta_ee_single"]) == observation_contract(ablation)
    with pytest.raises(ValueError, match="layout differs"):
        validate_checkpoint_contract({"experiment_config": asdict(base)}, ablation)
    with pytest.raises(ValueError, match="layout differs"):
        validate_checkpoint_contract({"experiment_config": ablation.to_serializable_dict()}, base)
    from hero_isaacsim.config_values.experiment import make_hero_recipe
    with pytest.raises(ValueError, match="anchor"):
        make_hero_recipe(delta_anchor=True, delta_ee=False)


def test_export_reader_accepts_the_delta_ee_ablation_layout_but_not_a_partial_pair():
    from sim2sim.policy_hero_export import DELTA_EE_TERMS as READER_DELTA_EE_TERMS, HERO_TERM_DIMS, HeroExportLayout
    assert tuple(READER_DELTA_EE_TERMS) == tuple(DELTA_EE_TERMS)
    for name, expect in (("without_delta_anchor", True), ("without_delta_ee", False), ("with_delta_anchor", True)):
        group = observation_contract(DEFAULTS[name])["actor_obs"]
        layout = HeroExportLayout(terms=tuple(group["terms"]), term_dims=tuple(group["term_dims"]), history_length=group["history_length"])
        assert layout.has_delta_ee_terms is expect and layout.total_dim == group["dim"], name
    base = observation_contract(DEFAULTS["without_delta_anchor"])["actor_obs"]
    partial = tuple(t for t in base["terms"] if t != "h08_dif_local_rigid_body_rot_ee")
    with pytest.raises(ValueError, match="partial"):
        HeroExportLayout(terms=partial, term_dims=tuple(HERO_TERM_DIMS[t] for t in partial), history_length=5)
    without_proprio = tuple(t for t in base["terms"] if t != "h09_dof_pos")
    with pytest.raises(ValueError, match="required"):
        HeroExportLayout(terms=without_proprio, term_dims=tuple(HERO_TERM_DIMS[t] for t in without_proprio), history_length=5)


def test_export_reader_frames_the_delta_ee_ablation_layout_end_to_end():
    """Sidecar metadata -> deployment contract -> policy tables -> one observation, for the 585 and the 675 layouts."""
    from sim2sim.policy_hero_export import HERO_TERM_DIMS, HeroExportPolicy, contract_from_metadata
    for name, dim in (("without_delta_ee", 585), ("without_delta_anchor", 675)):
        group = observation_contract(DEFAULTS[name])["actor_obs"]
        meta = {"actor_obs_layout": {"groups": [{"name": "actor_obs", "terms": list(group["terms"]), "history_length": group["history_length"]}],
                                     "history_layout": "frame_major_hero_v1"}, "preset": name}
        policy = HeroExportPolicy.tables_only(contract_from_metadata(meta, None))
        assert policy.layout.total_dim == dim and policy.layout.has_delta_ee_terms == (name == "without_delta_anchor")
        # the runner always computes every HERO term; the exported layout decides which ones enter the frame
        observation = policy.observe({t: np.zeros(HERO_TERM_DIMS[t]) for t in HERO_TERM_DIMS})
        assert observation.shape == (dim,) and observation.dtype == np.float32 and np.isfinite(observation).all()


@pytest.mark.parametrize("name", tuple(DEFAULTS))
def test_serialized_checkpoint_contract_and_network_forward(name):
    cfg = DEFAULTS[name]
    validate_checkpoint_contract({"experiment_config": asdict(cfg)}, cfg)
    contract = observation_contract(cfg)
    actor, act = make_actor(cfg, contract["actor_obs"]["dim"])
    inputs = torch.randn(2, contract["actor_obs"]["dim"])
    actions = act(inputs)
    assert actions.shape == (2, 29)
    assert torch.isfinite(actions).all()


def test_checkpoint_wrong_layout_and_same_width_wrong_semantics_are_rejected():
    base, anchor = DEFAULTS["without_delta_anchor"], DEFAULTS["with_delta_anchor"]
    with pytest.raises(ValueError, match="layout differs"):
        validate_checkpoint_contract({"experiment_config": asdict(base)}, anchor)
    bad = asdict(base)
    bad["observation"]["groups"]["actor_obs"]["terms"]["h01_base_ang_vel"]["scale"] = 0.125
    with pytest.raises(ValueError, match="scale differs"):
        validate_checkpoint_contract({"experiment_config": bad}, base)


def make_corpus(tmp_path, tag="amass", has_object=False):
    tmp_path.mkdir(parents=True, exist_ok=True)
    np.savez(tmp_path / "clip.npz", source_tag=np.asarray(tag), has_object=np.asarray(has_object))
    (tmp_path / "CORPUS_MANIFEST.json").write_text(json.dumps({"source_weights": {tag: 1.0}, "source_tags": [tag]}))
    return tmp_path


def test_launcher_defaults_are_local_amass_and_paper_recipe(tmp_path):
    script = load_script("train")
    args = script.parser().parse_args(["--motion-dir", str(make_corpus(tmp_path))])
    cfg = script.build_config(args)
    assert cfg.training.name == "without_delta_anchor"
    assert cfg.logger.type == "disabled"
    assert cfg.training.checkpoint is None
    assert cfg.command.setup_terms["motion_command"].params["motion_config"].source_weights == {"amass": 1.0}
    assert script.main(["--motion-dir", str(tmp_path), "--dry-run"]) == 0


def test_mixed_data_requires_opt_in_and_objects_are_rejected(tmp_path):
    script = load_script("train")
    corpus = make_corpus(tmp_path, "ik_reach_example")
    with pytest.raises(ValueError, match="AMASS only"):
        script.build_config(script.parser().parse_args(["--motion-dir", str(corpus)]))
    cfg = script.build_config(script.parser().parse_args(["--motion-dir", str(corpus), "--allow-mixed-data"]))
    assert cfg.command.setup_terms["motion_command"].params["motion_config"].source_weights == {"ik_reach_example": 1.0}
    make_corpus(tmp_path, has_object=True)
    with pytest.raises(ValueError, match="object clips"):
        script.build_config(script.parser().parse_args(["--motion-dir", str(corpus)]))


def test_export_bundle_validation_requires_shape_matched_sidecar(tmp_path):
    import onnx
    from onnx import TensorProto, helper
    script = load_script("export")
    src = tmp_path / "model.onnx"
    inputs = [helper.make_tensor_value_info(name, TensorProto.FLOAT, [1, 675])
              for name in ("actor_obs_lower_body", "actor_obs_upper_body")]
    output = helper.make_tensor_value_info("action", TensorProto.FLOAT, [1, 29])
    graph = helper.make_graph([helper.make_node("Constant", [], [output.name],
        value=helper.make_tensor("zero", TensorProto.FLOAT, [1, 29], [0.0] * 29))], "bundle", inputs, [output])
    onnx.save(helper.make_model(graph), src)
    sidecar = tmp_path / "model_hero.json"
    with pytest.raises(ValueError, match="both"):
        script.validate_bundle(src)
    sidecar.write_text(json.dumps({"actor_obs_dim": 675}))
    assert script.validate_bundle(src) == (src, sidecar)
    sidecar.write_text(json.dumps({"actor_obs_dim": 950}))
    with pytest.raises(ValueError, match="width"):
        script.validate_bundle(src)
    single = tmp_path / "single.onnx"
    graph1 = helper.make_graph([helper.make_node("Constant", [], [output.name],
        value=helper.make_tensor("zero", TensorProto.FLOAT, [1, 29], [0.0] * 29))], "single",
        [helper.make_tensor_value_info("actor_obs", TensorProto.FLOAT, [1, 675])], [output])
    onnx.save(helper.make_model(graph1), single)
    (tmp_path / "single_hero.json").write_text(json.dumps({"actor_obs_dim": 675}))
    assert script.validate_bundle(single)[0] == single


@pytest.mark.parametrize("name", tuple(DEFAULTS))
def test_actual_onnx_export_preserves_both_actor_heads_and_history(name, tmp_path):
    import onnxruntime as ort
    from hero_isaacsim.agents.ppo_dual.export import DualActorOnnxWrapper, export_dual_actor_as_onnx
    from hero_isaacsim.agents.ppo_dual.layout import layout_from_term_dims
    from hero_isaacsim.agents.ppo_dual.export import ONNX_INPUT_NAMES_SINGLE, SingleActorOnnxWrapper
    cfg = DEFAULTS[name]
    group = observation_contract(cfg)["actor_obs"]
    actor, act = make_actor(cfg, group["dim"])
    layout = layout_from_term_dims(dict(zip(group["terms"], group["term_dims"])), group["history_length"])
    permutation = layout.frame_major_to_term_major_perm()
    single = name in SINGLE
    wrapper = (SingleActorOnnxWrapper(actor, input_perm=permutation) if single else DualActorOnnxWrapper(actor, input_perm=permutation)).eval()
    frame_major = torch.randn(1, group["dim"])
    path = tmp_path / (name + ".onnx")
    export_dual_actor_as_onnx(wrapper, str(path), frame_major, input_names=ONNX_INPUT_NAMES_SINGLE if single else ("actor_obs_lower_body", "actor_obs_upper_body"))
    session = ort.InferenceSession(str(path), providers=["CPUExecutionProvider"])
    assert [i.name for i in session.get_inputs()] == (["actor_obs"] if single else ["actor_obs_lower_body", "actor_obs_upper_body"])
    result = session.run(None, {i.name: frame_major.numpy() for i in session.get_inputs()})[0]
    with torch.no_grad():
        expected = act(frame_major.index_select(-1, permutation)).numpy()
    assert result.shape == (1, 29)
    np.testing.assert_allclose(result, expected, rtol=1e-5, atol=2e-6)


@pytest.mark.parametrize("name", tuple(DEFAULTS))
def test_export_and_resume_accept_a_serialized_training_checkpoint(name, tmp_path):
    """Checkpoints store the experiment config as plain dicts; both launchers must accept them."""
    cfg = DEFAULTS[name]
    checkpoint = tmp_path / "model_00010.pt"
    torch.save({"experiment_config": cfg.to_serializable_dict(), "model_state_dict": {}, "iter": 10}, checkpoint)
    corpus = make_corpus(tmp_path / "corpus")
    export = load_script("export")
    assert export.main(["--checkpoint", str(checkpoint), "--motion-dir", str(corpus),
                        "--output", str(tmp_path / "bundle"), "--dry-run"]) == 0
    train = load_script("train")
    assert train.main(["--config", name, "--motion-dir", str(corpus), "--checkpoint", str(checkpoint), "--dry-run"]) == 0
    other = next(n for n in DEFAULTS if n != name)
    with pytest.raises(SystemExit):
        train.main(["--config", other, "--motion-dir", str(corpus), "--checkpoint", str(checkpoint), "--dry-run"])


def test_single_actor_presets_share_everything_but_the_algorithm():
    for base_name, single_name in zip(DUAL, SINGLE):
        base, single = DEFAULTS[base_name], DEFAULTS[single_name]
        assert single.algo._target_.endswith("ppo_single.PPOSingle") and base.algo._target_.endswith("ppo_dual.PPODual")
        for field in ("reward", "termination", "robot", "randomization", "action", "command", "curriculum", "simulator", "observation"):
            assert getattr(base, field) == getattr(single, field), field
        md = single.algo.config.module_dict
        assert md.actor.output_dim == [29] and md.actor.input_dim == ["actor_obs"] and md.critic.output_dim == [1]
        assert single.algo.config.init_noise_std == base.algo.config.init_noise_std == 0.8
        assert single.algo.config.export_motion_in_onnx is False
        assert observation_contract(single) == observation_contract(base)
    with pytest.raises(ValueError, match="dual|single"):
        from hero_isaacsim.config_values.experiment import make_hero_recipe
        make_hero_recipe(actor="triple")


def test_checkpoint_from_the_other_actor_family_is_rejected():
    dual, single = DEFAULTS["without_delta_anchor"], DEFAULTS["without_delta_anchor_single"]
    with pytest.raises(ValueError, match="trained with"):
        validate_checkpoint_contract({"experiment_config": asdict(dual)}, single)
    with pytest.raises(ValueError, match="trained with"):
        validate_checkpoint_contract({"experiment_config": single.to_serializable_dict()}, dual)
    validate_checkpoint_contract({"experiment_config": single.to_serializable_dict()}, single)


def test_default_motion_sampler_cannot_collapse_onto_one_clip():
    """The adaptive sampler must keep a per-clip cap and a uniform floor (the backend defaults let one clip take over)."""
    from hero_isaacsim.config_values.command import get_motion_config, make_hero_motion_config
    from hero_isaacsim.config_values import sampler as sampler_settings
    for cfg in DEFAULTS.values():
        mc = get_motion_config(cfg.command)
        assert mc.use_adaptive_timesteps_sampler and mc.adaptive_sampler_per_clip
        assert mc.adaptive_sampler_clip_max_probability == 1.0  # the absolute cap stays off: it would starve small sources
        assert mc.adaptive_sampler_clip_max_relative == 10.0
        assert mc.adaptive_sampler_uniform_ratio == 0.3
    # the backend's absolute cap / temperature are refused at configuration time for the per-clip HERO sampler, naming
    # the relative cap as the knob -- not lifted to 1 / num_clips first and refused by the sampler afterwards
    with pytest.raises(ValueError, match="adaptive_sampler_clip_max_relative"):
        make_hero_motion_config(adaptive_sampler_clip_max_probability=0.001)
    with pytest.raises(ValueError, match="adaptive_sampler_clip_max_relative"):
        make_hero_motion_config(adaptive_sampler_clip_temperature=0.5)
    make_hero_motion_config(adaptive_sampler_per_clip=False, adaptive_sampler_clip_max_probability=0.001)  # backend mode keeps the backend knobs
    # so there is no per-corpus "feasible cap" to lift for a HERO config: the helpers that did so are gone (nothing used
    # them, and their dict-form HERO detection missed pre-fix saved configs); the settings module keeps the two constants
    assert mc.adaptive_sampler_uniform_ratio == sampler_settings.ADAPTIVE_UNIFORM_RATIO
    assert mc.adaptive_sampler_clip_max_relative == sampler_settings.ADAPTIVE_CLIP_MAX_RELATIVE
    assert not any(name.startswith(("with_feasible", "feasible_", "count_corpus")) for name in dir(sampler_settings))


def _sampler(n_a, n_b, w_a, w_b, relative=10.0, uniform=0.3):
    from hero_isaacsim.managers.command.sampler import HeroAdaptiveTimestepsSampler, build_clip_prior
    ids = torch.tensor([0] * n_a + [1] * n_b)
    prior = build_clip_prior(ids, ["amass", "ik"], {"amass": w_a, "ik": w_b}, "cpu")
    n = n_a + n_b
    return HeroAdaptiveTimestepsSampler(n * 100, "cpu", 50, per_clip=True, num_clips=n, max_clip_time_step=100,
        adaptive_uniform_ratio=uniform, adaptive_clip_max_probability=1.0, clip_prior=prior,
        source_tag_ids=ids, source_tags=["amass", "ik"], clip_cap_relative=relative)


def test_source_weights_are_honoured_and_no_clip_can_absorb_the_sampler():
    ik = lambda s: float(s.expected_source_fractions()["motion/source_frac_ik"])
    # few generated IK clips next to a large corpus keep their documented share (the flat 0.1 % cap gave 3.2 %)
    s = _sampler(10000, 32, 0.7, 0.3)
    assert abs(ik(s) - 0.3) < 1e-5
    # an explicitly excluded source stays at exactly zero, also for a small corpus
    assert ik(_sampler(300, 200, 1.0, 0.0)) == 0.0
    assert abs(ik(_sampler(300, 200, 1.0, 0.1)) - 0.1 / 1.1) < 1e-5
    # a clip that fails at every phase is pulled up, but never beyond 10x its prior share
    s = _sampler(300, 200, 1.0, 0.1)
    prior5 = float(s.clip_prior[5]); s.bin_failed_count[5, :] = 1000.0
    m5 = float(s.clip_marginal()[5])
    assert prior5 * 1.5 < m5 <= prior5 * 10.0 + 1e-6
    assert abs(float(s.clip_marginal().sum()) - 1.0) < 1e-5
    # the pathological case from the 20K run: one clip out of ~30k failing everywhere is bounded at 10x uniform
    s = _sampler(29000, 548, 0.8, 0.2); s.bin_failed_count[7, :] = 1e6
    assert float(s.clip_marginal()[7]) <= 10.0 * float(s.clip_prior[7]) + 1e-6
    with pytest.raises(ValueError, match=">= 1"):
        _sampler(10, 10, 0.5, 0.5, relative=0.5)
