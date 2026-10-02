"""Record prepared runtime robot assets without mistaking a preset for runtime evidence."""

from __future__ import annotations

import copy
import hashlib
from pathlib import Path
from typing import Any

# Conditional export fields: old simulators/checkpoints without preparation
# receipts retain their historical metadata and source URDF behavior.
ROBOT_COLLISION_EXPORT_KEYS = (
    "robot_collision_assets",
    "robot_foot_collision_profile",
    "robot_urdf_effective_path",
    "robot_urdf_source_sha256",
    "robot_urdf_effective_sha256",
    "robot_urdf_embedded_variant_index",
)


def runtime_robot_asset_metadata(simulator: Any) -> dict[str, Any]:
    """Copy receipts actually published by the simulator; never synthesize one from config."""
    records = getattr(simulator, "robot_collision_assets", None)
    if records is None or records == [] or records == ():
        return {}
    if not isinstance(records, (list, tuple)):
        raise ValueError("simulator.robot_collision_assets must be a list of preparation receipts")
    records = copy.deepcopy(list(records))
    for record in records:
        if not isinstance(record, dict):
            raise ValueError("robot collision preparation receipt must be a dictionary")
        for key in ("profile", "source_path", "urdf_path", "source_sha256", "generated_sha256"):
            if not isinstance(record.get(key), str) or not record[key]:
                raise ValueError(f"robot collision preparation receipt lacks {key}")
        for key in ("source_sha256", "generated_sha256"):
            digest = record[key]
            if len(digest) != 64 or any(c not in "0123456789abcdef" for c in digest):
                raise ValueError(f"robot collision preparation receipt has an invalid {key}")
    profiles = {record["profile"] for record in records}
    return {
        "robot_collision_assets": records,
        "robot_foot_collision_profile": next(iter(profiles)) if len(profiles) == 1 else "mixed",
    }


def effective_robot_urdf_metadata(source_path: str, simulator: Any) -> dict[str, Any]:
    """Embed the URDF actually sent to the importer, after verifying its preparation hash.

    ``robot_urdf_path`` remains the configured source-family path for existing
    deployment selectors.  The explicit effective path/hash and full variant
    receipts disambiguate the embedded text.  For multi-asset training, select
    the same source family as the configuration's single-URDF fallback; a
    missing/ambiguous match is an error, never an arbitrary hand substitution.
    """
    metadata = runtime_robot_asset_metadata(simulator)
    if not metadata:
        return {}
    records = metadata["robot_collision_assets"]
    configured = Path(source_path).resolve()
    matches = [i for i, record in enumerate(records) if Path(record["source_path"]).resolve() == configured]
    if len(matches) != 1:
        raise ValueError(f"Expected one runtime collision asset for configured URDF {configured}; got {len(matches)}")
    texts: list[str] = []
    for record in records:
        path = Path(record["urdf_path"])
        payload = path.read_bytes()
        actual = hashlib.sha256(payload).hexdigest()
        if actual != record["generated_sha256"]:
            raise ValueError(f"Runtime robot URDF changed after preparation: {path}")
        texts.append(payload.decode("utf-8"))
    index = matches[0]
    record = records[index]
    metadata.update(
        robot_urdf=texts[index],
        robot_urdf_effective_path=record["urdf_path"],
        robot_urdf_source_sha256=record["source_sha256"],
        robot_urdf_effective_sha256=record["generated_sha256"],
        robot_urdf_embedded_variant_index=index,
    )
    return metadata
