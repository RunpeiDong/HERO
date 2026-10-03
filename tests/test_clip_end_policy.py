"""The generated reaching clips hold their last frame: one source tag in the preset, the docs and the converter."""
from __future__ import annotations

import importlib.util
import json
from pathlib import Path
import re
import sys

import numpy as np
import pytest
import torch
from loguru import logger

ROOT = Path(__file__).resolve().parents[1]
sys.path[:0] = [str(ROOT), str(ROOT / "third_party/holosoma"), str(ROOT / "scripts")]
from hero_isaacsim.config_values.command import (DEFAULT_CLIP_END_POLICY_BY_SOURCE, IK_REACH_SOURCE_TAG, get_motion_config,
                                                 hero_h1_command, with_motion_config)
from hero_isaacsim.managers.command.loader import ClipFacts, HeroMultiMotionLoader


def load_script(name):
    spec = importlib.util.spec_from_file_location("hero_script_" + name, ROOT / "scripts" / (name + ".py"))
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _facts(tag, n, frames=100):
    return [ClipFacts(file=f"{tag}_{i}.npz", num_frames=frames, fps=50.0, source_tag=tag, parent_id=f"{tag}_p{i}",
                      license_class="research-only", object_track=False, clip_object_flag=None, box_size=None, object_z0=None,
                      extension_missing=[]) for i in range(n)]


def _loader(policies, facts):
    """A loader built from clip facts only (no timelines): exactly the metadata path the census reads."""
    ld = HeroMultiMotionLoader.__new__(HeroMultiMotionLoader)
    ld.device = "cpu"
    ld.clip_end_policy_by_source = dict(policies)
    ld.default_clip_end_policy = "rollover"
    ld.num_skipped, ld.skipped_files = 0, []
    ld._joint_pos = torch.zeros(sum(f.num_frames for f in facts), 1)
    ld._finalize_clip_metadata(facts)
    return ld


def _census_line(ld):
    lines: list[str] = []
    sink = logger.add(lines.append, format="{message}")
    try:
        ld._log_census()
    finally:
        logger.remove(sink)
    return "\n".join(lines)


def make_corpus(tmp_path, tags=("amass", IK_REACH_SOURCE_TAG), policies=None):
    tmp_path.mkdir(parents=True, exist_ok=True)
    for tag in tags:
        np.savez(tmp_path / f"{tag}.npz", source_tag=np.asarray(tag), has_object=np.asarray(False))
    manifest = {"source_weights": {tag: 1.0 for tag in tags}, "source_tags": list(tags)}
    if policies is not None:
        manifest["clip_end_policy_by_source"] = policies
    (tmp_path / "CORPUS_MANIFEST.json").write_text(json.dumps(manifest))
    return tmp_path


def test_the_ik_source_tag_is_spelled_the_same_everywhere():
    assert IK_REACH_SOURCE_TAG == "ik_reach_example"
    assert DEFAULT_CLIP_END_POLICY_BY_SOURCE == {"amass": "rollover", IK_REACH_SOURCE_TAG: "hold"}
    assert get_motion_config(hero_h1_command).clip_end_policy_by_source[IK_REACH_SOURCE_TAG] == "hold"
    data_md = (ROOT / "docs/data.md").read_text()
    assert re.findall(r"--source-tag (\S+)", data_md) == [IK_REACH_SOURCE_TAG]
    assert f'"{IK_REACH_SOURCE_TAG}": "hold"' in data_md


def test_loader_census_reports_hold_for_every_ik_clip():
    facts = _facts("amass", 5) + _facts(IK_REACH_SOURCE_TAG, 7)
    ld = _loader(DEFAULT_CLIP_END_POLICY_BY_SOURCE, facts)
    assert int((~ld.clip_rollover).sum()) == 7 and int(ld.clip_rollover.sum()) == 5
    assert ld.clip_rollover.tolist() == [True] * 5 + [False] * 7
    assert "clip end policy rollover=5 hold=7" in _census_line(ld)
    # a policy keyed by another spelling of the tag silently leaves every reaching clip rolling over
    misspelt = _loader({"amass": "rollover", "reach_example": "hold"}, facts)
    assert "clip end policy rollover=12 hold=0" in _census_line(misspelt)


def test_manifest_end_policies_reach_the_motion_config(tmp_path):
    script = load_script("train")
    parse = lambda corpus, *extra: script.parser().parse_args(["--motion-dir", str(corpus), "--allow-mixed-data", *extra])
    mc = lambda cfg: get_motion_config(cfg.command)
    # no manifest entry: the preset defaults apply, so the IK clips hold
    cfg = script.build_config(parse(make_corpus(tmp_path / "a")))
    assert mc(cfg).clip_end_policy_by_source == DEFAULT_CLIP_END_POLICY_BY_SOURCE
    # the manifest overlays the defaults tag by tag
    cfg = script.build_config(parse(make_corpus(tmp_path / "b", policies={IK_REACH_SOURCE_TAG: "rollover", "custom": "hold"})))
    assert mc(cfg).clip_end_policy_by_source == {"amass": "rollover", IK_REACH_SOURCE_TAG: "rollover", "custom": "hold"}
    with pytest.raises(ValueError, match="rollover\\|hold"):
        script.build_config(parse(make_corpus(tmp_path / "c", policies={IK_REACH_SOURCE_TAG: "freeze"})))
    assert script.main(["--motion-dir", str(tmp_path / "a"), "--allow-mixed-data", "--dry-run"]) == 0


def test_with_motion_config_passes_the_policy_map_through():
    cfg = with_motion_config(hero_h1_command, clip_end_policy_by_source={"x": "hold", "amass": "rollover"})
    assert get_motion_config(cfg).clip_end_policy_by_source == {"x": "hold", "amass": "rollover"}
    assert get_motion_config(hero_h1_command).clip_end_policy_by_source == DEFAULT_CLIP_END_POLICY_BY_SOURCE  # untouched
    with pytest.raises(ValueError, match="rollover"):
        with_motion_config(hero_h1_command, clip_end_policy_by_source={"x": "stop"})
