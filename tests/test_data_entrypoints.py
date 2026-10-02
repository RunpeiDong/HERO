"""CPU checks for AMASS preparation and input validation."""
from pathlib import Path

import numpy as np
import pytest

ROOT = Path(__file__).resolve().parents[1]

from data_tools.prepare_amass import input_payload, prepare
from hero_isaacsim.constants import DOF_NAMES


def _retargeted_clip(path: Path) -> None:
    pose = np.zeros((5, 36), dtype=np.float64)
    pose[:, 2] = 0.76
    pose[:, 3] = 1.0
    pose[:, 7:] = np.linspace(-0.02, 0.02, 29)
    np.savez(path, joint_pos=pose, fps=np.asarray(50), joint_names=np.asarray(DOF_NAMES))


def test_raw_amass_requires_external_retargeting():
    with pytest.raises(ValueError, match="external retargeting"):
        input_payload({"poses": np.zeros((5, 156)), "trans": np.zeros((5, 3))})


def test_frame_rate_and_object_track_are_explicit():
    pose = np.zeros((5, 36))
    with pytest.raises(ValueError, match="declare fps"):
        input_payload({"joint_pos": pose})
    with pytest.raises(ValueError, match="object track"):
        input_payload({"joint_pos": pose, "fps": np.asarray(50), "object_pos_w": np.zeros((5, 3))})


def test_amass_conversion_is_source_only_and_preserves_input(tmp_path):
    source = tmp_path / "source"
    source.mkdir()
    clip = source / "walk.npz"
    _retargeted_clip(clip)
    original = clip.read_bytes()
    output = tmp_path / "converted"
    result = prepare(source, output)
    assert result["complete"] is True
    assert result["source_weights"] == {"amass": 1.0}
    assert clip.read_bytes() == original
    assert (output / "CORPUS_MANIFEST.json").exists()
    assert not (output / "BUILD_INCOMPLETE").exists()
    with np.load(next(output.glob("amass__*.npz")), allow_pickle=False) as data:
        assert data["source_tag"].item() == "amass"
        assert not data["has_object"].item()
        assert data["joint_pos"].shape == (5, 36)
        assert data["ee_pos_pelvis"].shape == (5, 2, 3)
        assert np.isfinite(data["body_lin_vel_w"]).all()


def test_failed_conversion_never_publishes_training_manifest(tmp_path):
    source = tmp_path / "raw"
    source.mkdir()
    np.savez(source / "raw_amass.npz", poses=np.zeros((5, 156)))
    output = tmp_path / "failed"
    result = prepare(source, output)
    assert result["complete"] is False
    assert (output / "BUILD_INCOMPLETE").exists()
    assert not (output / "CORPUS_MANIFEST.json").exists()
    assert "external retargeting" in result["files"][0]["error"]


@pytest.mark.parametrize("module", ["data_tools.hero_reach_generator", "data_tools.npz_convert", "data_tools.prepare_amass"])
def test_data_tool_entry_points_keep_math_single_threaded(module):
    """Worker pools must not start one BLAS thread per host core; the defaults have to land before numpy loads."""
    import os
    import subprocess
    import sys
    env = {k: v for k, v in os.environ.items() if not k.endswith("_NUM_THREADS") and k != "VECLIB_MAXIMUM_THREADS"}
    env["PYTHONPATH"] = os.pathsep.join(p for p in (str(ROOT), str(ROOT / "third_party/holosoma"), env.get("PYTHONPATH", "")) if p)
    code = (f"import {module}, os, sys, numpy; mods = list(sys.modules); "
            "print(os.environ['OPENBLAS_NUM_THREADS'], os.environ['OMP_NUM_THREADS'], mods.index('data_tools._threads') < mods.index('numpy'))")
    out = subprocess.run([sys.executable, "-c", code], env=env, capture_output=True, text=True, timeout=300)
    assert out.returncode == 0, out.stderr[-800:]
    assert out.stdout.split() == ["1", "1", "True"], out.stdout  # the defaults must land before numpy is imported
