"""Prepare AMASS motions already retargeted to the 29-DoF G1 for HERO training.

This is a format converter, not SMPL/SMPL-X retargeting. Input NPZ files must have
joint_pos or qpos with shape (T, 36): world xyz, world wxyz quaternion, and the
29 G1 joint angles. fps is required; optional joint_names specifies joint order.
Obtain AMASS under its terms and perform robot retargeting before this step.
"""
from __future__ import annotations

from data_tools import _threads  # noqa: F401  -- single-threaded BLAS/OpenMP before numpy and MuJoCo load

import argparse
from concurrent.futures import ProcessPoolExecutor
import hashlib
import io
import json
import os
from pathlib import Path
import re
from typing import Any

import numpy as np

from data_tools.fk_mujoco import DEFAULT_SCENE_XML, G1Dex3FK
from data_tools.npz_convert import convert_payload
from data_tools.schema import ClipMeta, LICENSE_CLASSES, scalar_fps

_FK: G1Dex3FK | None = None


def _init_worker(scene: str) -> None:
    global _FK
    _FK = G1Dex3FK(scene)


def input_payload(data: dict[str, np.ndarray]) -> dict[str, np.ndarray]:
    """Accept explicit G1 poses and recompute all velocities/body channels by FK."""
    key = "joint_pos" if "joint_pos" in data else "qpos" if "qpos" in data else None
    if key is None:
        raise ValueError("Expected G1 joint_pos or qpos (T, 36). Raw AMASS SMPL/SMPL-X poses require external retargeting first.")
    qpos = np.asarray(data[key])
    if qpos.ndim != 2 or qpos.shape[1] != 36 or len(qpos) < 2:
        raise ValueError(f"Expected at least two frames of G1 poses (T, 36), got {qpos.shape}")
    if "fps" not in data:
        raise ValueError("Retargeted input must declare fps; no source frame rate is assumed")
    fps = scalar_fps(data["fps"])
    if not np.isfinite(fps) or fps <= 0:
        raise ValueError("fps must be finite and positive")
    if ("has_object" in data and bool(np.asarray(data["has_object"]).item())) or "object_pos_w" in data:
        raise ValueError("The AMASS-only training input must not contain an object track")
    payload = {"joint_pos": qpos, "fps": np.asarray(fps)}
    if "joint_names" in data:
        payload["joint_names"] = data["joint_names"]
    return payload


def _convert(job: tuple[str, str, str, str]) -> dict[str, Any]:
    source_name, relative_name, output_name, license_class = job
    source, output = Path(source_name), Path(output_name)
    row: dict[str, Any] = {"file": output.name, "source": relative_name, "status": "error"}
    try:
        raw = source.read_bytes()
        row["input_sha256"] = hashlib.sha256(raw).hexdigest()
        with np.load(io.BytesIO(raw), allow_pickle=False) as archive:
            data = input_payload(dict(archive))
        payload, receipt = convert_payload(data, ClipMeta(source_tag="amass", license_class=license_class,
            keep_object=False, target_fps=50), parent_id=f"amass/{Path(relative_name).with_suffix('').as_posix()}", fk=_FK)
        temporary = output.with_name(f".{output.name}.{os.getpid()}.tmp")
        try:
            with temporary.open("xb") as stream:
                np.savez_compressed(stream, **payload)
            os.replace(temporary, output)
        finally:
            temporary.unlink(missing_ok=True)
        row.update(status="ok", sha256=hashlib.sha256(output.read_bytes()).hexdigest(),
            frames=int(receipt["frames_out"]), fps=50, has_object=False)
    except Exception as exc:
        row["error"] = f"{type(exc).__name__}: {exc}"
    return row


def prepare(input_dir: Path, output_dir: Path, *, jobs: int = 1,
            scene_xml: Path = DEFAULT_SCENE_XML, license_class: str = "research-only") -> dict[str, Any]:
    source, output = input_dir.expanduser().resolve(), output_dir.expanduser().resolve()
    scene_xml = scene_xml.expanduser().resolve()
    if not source.is_dir() or not scene_xml.is_file():
        raise ValueError("Input directory and G1 FK model must exist")
    if output.exists() or source == output or source in output.parents or output in source.parents:
        raise ValueError("Output must be a new directory outside the input tree")
    if not 1 <= jobs <= 64:
        raise ValueError("jobs must be between 1 and 64")
    if license_class not in LICENSE_CLASSES:
        raise ValueError(f"Unknown license class: {license_class}")
    clips = sorted(source.rglob("*.npz"))
    if not clips:
        raise ValueError("No retargeted NPZ clips found under input")
    work = []
    for clip in clips:
        relative = clip.relative_to(source).as_posix()
        safe = re.sub(r"[^A-Za-z0-9_-]+", "_", clip.stem)[:80]
        suffix = hashlib.sha256(relative.encode()).hexdigest()[:12]
        work.append((str(clip), relative, str(output / f"amass__{safe}__{suffix}.npz"), license_class))
    output.mkdir(parents=True)
    marker = output / "BUILD_INCOMPLETE"
    marker.write_text("Do not train from this directory until conversion completes.\n")
    if jobs == 1:
        _init_worker(str(scene_xml))
        rows = list(map(_convert, work))
    else:
        with ProcessPoolExecutor(jobs, initializer=_init_worker, initargs=(str(scene_xml),)) as pool:
            rows = list(pool.map(_convert, work))
    complete = all(row["status"] == "ok" for row in rows)
    report = {"schema": "hero_amass_corpus/v1", "complete": complete,
        "source_tags": ["amass"], "source_weights": {"amass": 1.0}, "clips": len(rows),
        "fps": 50, "robot": "g1_29dof", "has_object": False,
        "input_contract": "externally retargeted G1 poses; world xyz+wxyz, 29 joint angles",
        "license_class": license_class, "files": rows}
    (output / "BUILD_REPORT.json").write_text(json.dumps(report, indent=2) + "\n")
    if complete:
        (output / "CORPUS_MANIFEST.json").write_text(json.dumps(report, indent=2) + "\n")
        marker.unlink()
    return report


def main(argv: list[str] | None = None) -> int:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--input", type=Path, required=True, help="Directory of AMASS motions already retargeted to G1")
    parser.add_argument("--output", type=Path, required=True, help="New output directory; no bundled AMASS data is provided")
    parser.add_argument("--jobs", type=int, default=1)
    parser.add_argument("--scene-xml", type=Path, default=DEFAULT_SCENE_XML)
    parser.add_argument("--license-class", choices=LICENSE_CLASSES, default="research-only", help="Provenance label; does not grant dataset rights")
    args = parser.parse_args(argv)
    try:
        report = prepare(args.input, args.output, jobs=args.jobs, scene_xml=args.scene_xml, license_class=args.license_class)
    except (ValueError, OSError) as exc:
        parser.error(str(exc))
    print(json.dumps({"complete": report["complete"], "clips": report["clips"], "source_weights": report["source_weights"],
        "report": str(args.output / "BUILD_REPORT.json")}, indent=2))
    return 0 if report["complete"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
