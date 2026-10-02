"""Prepare an immutable G1 URDF with explicit SONIC foot geometry before USD import.

This module uses only the standard library; it does not require an Isaac Sim process.
The source asset is never edited.  ``source`` (also ``legacy_mesh``) returns its
original path. ``sonic_train`` uses the official training URDF's seven cylinders
(the Isaac caller MUST enable cylinder-to-capsule conversion). ``sonic_box`` uses
the separate MuJoCo deployment XML's one box. Both replace the ankle-roll fixed
subtree collision geometry. Visuals, inertials,
joints, and collision geometry on other articulated links keep their semantics.

Geometry sources: NVlabs/GR00T-WholeBodyControl commit
``b042411fae38ee4d1af9aac82a37a1f8d14d6dd0``,
training ``gear_sonic/data/assets/robot_description/urdf/g1/main.urdf`` and
deployment ``gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml``.
This matches its geometry, not its entire simulator/robot configuration.
"""

from __future__ import annotations

import hashlib
import json
import os
import tempfile
import xml.etree.ElementTree as ET
from dataclasses import dataclass
from pathlib import Path
from urllib.parse import unquote, urlparse

G1_FOOT_LINKS = ("left_ankle_roll_link", "right_ankle_roll_link")
SONIC_FOOT_BOX_CENTER_M = (0.035, 0.0, -0.030)
SONIC_FOOT_BOX_SIZE_M = (0.170, 0.060, 0.010)  # URDF full extents, not MJCF half extents.
SONIC_SOURCE_COMMIT = "b042411fae38ee4d1af9aac82a37a1f8d14d6dd0"
_TRANSFORM_VERSION = "g1_sonic_foot_collision_v1"
SONIC_TRAIN_SOURCE_PATH = "gear_sonic/data/assets/robot_description/urdf/g1/main.urdf"
SONIC_TRAIN_SOURCE_SHA256 = "6e107391d015ab27438026bb953b21ef8fa6a02305e65393985908941d7fd98d"
# Exact URDF values, including its rounded +/-pi/2 rotation. Same on both feet.
# (ankle-local xyz, rpy, radius, cylinder length); official g1.py imports as capsules.
SONIC_TRAIN_FOOT_CYLINDERS = (
    ((.075, -.026, -.025), (0., 1.5708, 0.), .010, .050),
    ((.0395, -.018, -.025), (0., -1.5708, 0.), .008, .167),
    ((.039, -.010, -.025), (0., -1.5708, 0.), .010, .182),
    ((.039, .000, -.025), (0., -1.5708, 0.), .010, .186),
    ((.039, .010, -.025), (0., -1.5708, 0.), .010, .182),
    ((.0395, .018, -.025), (0., -1.5708, 0.), .008, .167),
    ((.075, .026, -.025), (0., 1.5708, 0.), .010, .050),
)


@dataclass(frozen=True)
class AssetDependency:
    path: str
    sha256: str


@dataclass(frozen=True)
class PreparedFootCollisionAsset:
    profile: str
    source_path: str
    urdf_path: str
    source_sha256: str
    generated_sha256: str
    cache_key: str
    foot_links: tuple[str, ...]
    replaced_collision_counts: tuple[int, ...]
    dependencies: tuple[AssetDependency, ...]
    geometry_source_commit: str = ""
    geometry_source_path: str = ""
    geometry_source_sha256: str = ""


def _sha256(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def _fixed_foot_subtrees(root: ET.Element) -> tuple[dict[str, ET.Element], tuple[tuple[str, ...], ...]]:
    """Validate the link graph and find fixed descendants without crossing actuated joints."""
    links: dict[str, ET.Element] = {}
    for link in root.findall("link"):
        name = link.get("name")
        if not name or name in links:
            raise ValueError(f"URDF has missing or duplicate link name: {name!r}")
        links[name] = link
    missing = set(G1_FOOT_LINKS) - links.keys()
    if missing:
        raise ValueError(f"SONIC foot preparation requires both G1 ankle-roll links; missing {sorted(missing)}")

    fixed_children: dict[str, list[str]] = {}
    parents: dict[str, str] = {}
    for joint in root.findall("joint"):
        parent, child = joint.find("parent"), joint.find("child")
        p = None if parent is None else parent.get("link")
        c = None if child is None else child.get("link")
        if p not in links or c not in links:
            raise ValueError(f"URDF joint {joint.get('name')!r} refers to an undeclared link")
        if c in parents:
            raise ValueError(f"URDF link {c!r} has more than one parent joint")
        parents[c] = p
        if joint.get("type") == "fixed":
            fixed_children.setdefault(p, []).append(c)

    subtrees: list[tuple[str, ...]] = []
    for foot in G1_FOOT_LINKS:
        visited: set[str] = set()
        pending = [foot]
        names: list[str] = []
        while pending:
            name = pending.pop()
            if name in visited:
                raise ValueError(f"URDF has a fixed-joint cycle under {foot!r}")
            visited.add(name)
            names.append(name)
            pending.extend(reversed(fixed_children.get(name, ())))
        subtrees.append(tuple(names))
    if set(subtrees[0]) & set(subtrees[1]):
        raise ValueError("G1 ankle-roll fixed subtrees overlap")
    return links, tuple(subtrees)


def _resolve_resource_filename(filename: str, source_dir: Path) -> Path:
    """Local asset references must remain resolvable when the URDF moves into a cache."""
    if filename.startswith("file://"):
        parsed = urlparse(filename)
        if parsed.netloc not in ("", "localhost") or parsed.query or parsed.fragment:
            raise ValueError(f"Unsupported file URI in robot URDF: {filename!r}")
        path = Path(unquote(parsed.path))
    elif "://" in filename:
        raise ValueError(
            f"SONIC foot profiles need resolved local mesh/texture filenames, got {filename!r}; "
            "resolve package/network resources before preparing the collision asset"
        )
    else:
        path = Path(filename)
    path = (source_dir / path).resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Robot URDF mesh/texture asset does not exist: {path}")
    return path


def _absolutize_resources(root: ET.Element, source_dir: Path) -> tuple[AssetDependency, ...]:
    paths: set[Path] = set()
    for tag in ("mesh", "texture"):
        for resource in root.iter(tag):
            filename = resource.get("filename")
            if not filename:
                raise ValueError(f"Robot URDF {tag} is missing filename")
            path = _resolve_resource_filename(filename, source_dir)
            resource.set("filename", str(path))
            paths.add(path)
    # Include local resource content in the cache key.  Editing an STL must not
    # accidentally reuse an old USD just because the URDF text did not change.
    return tuple(AssetDependency(str(path), _sha256(path.read_bytes())) for path in sorted(paths))


def _publish_immutable(path: Path, payload: bytes) -> None:
    """Publish a complete file atomically across ranks; never repair/overwrite a corrupt cache."""
    path.parent.mkdir(parents=True, exist_ok=True)
    if path.exists():
        if path.read_bytes() != payload:
            raise RuntimeError(f"Generated foot-collision cache content was modified: {path}")
        return
    tmp_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(dir=path.parent, prefix=".foot_collision_", delete=False) as tmp:
            tmp_name = tmp.name
            tmp.write(payload)
            tmp.flush()
            os.fsync(tmp.fileno())
        try:
            # Hard-link publication is atomic and fails if another rank won.
            os.link(tmp_name, path)
        except FileExistsError:
            if path.read_bytes() != payload:
                raise RuntimeError(f"Generated foot-collision cache content was modified: {path}") from None
    finally:
        if tmp_name is not None:
            os.unlink(tmp_name)


def prepare_foot_collision_urdf(
    source_path: str | os.PathLike[str],
    cache_dir: str | os.PathLike[str],
    mode: str = "sonic_box",
) -> PreparedFootCollisionAsset:
    """Return the source URDF or a content-addressed, generated SONIC-profile URDF.

    The caller supplies its process-private conversion directory as ``cache_dir``;
    each generated basename contains a hash of source/transform/resource content
    so single- and multi-USD converters cannot confuse collision profiles.  This
    helper does not modify self-collision flags or any other physics setting.
    ``source`` reproduces the original URDF verbatim and does not create a cache.
    """
    if mode not in ("source", "legacy_mesh", "sonic_box", "sonic_train"):
        raise ValueError(f"Unknown foot collision profile {mode!r}; expected 'source', 'sonic_box' or 'sonic_train'")
    source = Path(source_path).resolve(strict=True)
    source_bytes = source.read_bytes()
    source_sha = _sha256(source_bytes)
    if mode in ("source", "legacy_mesh"):
        return PreparedFootCollisionAsset(
            "source", str(source), str(source), source_sha, source_sha, source_sha, (), (), ()
        )

    parser = ET.XMLParser(target=ET.TreeBuilder(insert_comments=True))
    root = ET.fromstring(source_bytes, parser=parser)
    if root.tag != "robot":
        raise ValueError(f"Expected URDF <robot> root, got {root.tag!r}")
    links, subtrees = _fixed_foot_subtrees(root)
    removed: list[int] = []
    for foot, subtree in zip(G1_FOOT_LINKS, subtrees, strict=True):
        count = 0
        for name in subtree:
            link = links[name]
            for collision in link.findall("collision"):
                link.remove(collision)
                count += 1
        if mode == "sonic_box":
            collision = ET.SubElement(links[foot], "collision", name=f"{foot}_sonic_box")
            ET.SubElement(collision, "origin", xyz="0.035 0 -0.03", rpy="0 0 0")
            geometry = ET.SubElement(collision, "geometry")
            ET.SubElement(geometry, "box", size="0.17 0.06 0.01")
        else:
            for index, (xyz, rpy, radius, length) in enumerate(SONIC_TRAIN_FOOT_CYLINDERS):
                collision = ET.SubElement(links[foot], "collision", name=f"{foot}_sonic_train_{index}")
                ET.SubElement(collision, "origin", xyz=" ".join(map(str, xyz)), rpy=" ".join(map(str, rpy)))
                geometry = ET.SubElement(collision, "geometry")
                ET.SubElement(geometry, "cylinder", radius=str(radius), length=str(length))
        removed.append(count)

    dependencies = _absolutize_resources(root, source.parent)
    generated = ET.tostring(root, encoding="utf-8", xml_declaration=True)
    generated_sha = _sha256(generated)
    key_payload = json.dumps(
        {
            "transform": _TRANSFORM_VERSION,
            "profile": mode,
            "source_path": str(source),
            "source_sha256": source_sha,
            "generated_sha256": generated_sha,
            "dependencies": [(dep.path, dep.sha256) for dep in dependencies],
        },
        sort_keys=True,
        separators=(",", ":"),
    ).encode("utf-8")
    cache_key = _sha256(key_payload)
    generated_path = Path(cache_dir).resolve() / f"{source.stem}__{mode}_{cache_key}.urdf"
    if generated_path == source:
        raise ValueError("Generated foot-collision URDF cannot overwrite its source")
    _publish_immutable(generated_path, generated)
    return PreparedFootCollisionAsset(
        mode, str(source), str(generated_path), source_sha, generated_sha, cache_key,
        G1_FOOT_LINKS, tuple(removed), dependencies,
        SONIC_SOURCE_COMMIT,
        SONIC_TRAIN_SOURCE_PATH if mode == "sonic_train" else "gear_sonic/data/robot_model/model_data/g1/g1_29dof_with_hand.xml",
        SONIC_TRAIN_SOURCE_SHA256 if mode == "sonic_train" else "",
    )
