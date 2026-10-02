"""Helpers for per-environment robot variants from several URDF/USD files.

One Isaac Lab articulation view spans environments using different converted USDs
with the same joints and rigid bodies and ``replicate_physics=False``.
``isaacsim.py`` owns the simulator calls; this module provides:

* the per-env variant ASSIGNMENT (``plan_variant_assignment``): repeat variant ids by their integer weights
  ((2, 1, 1) -> [0, 0, 1, 2]), tile the pattern cyclically over ``num_envs`` (so the realised proportions are exact up
  to the tiling remainder), then shuffle with a ``torch.Generator`` seeded from the training seed (deterministic;
  decorrelates the hand type from the env index, which holosoma also uses for env origins / terrain columns);
* the stock Isaac Lab layout (``tile_variant_pattern`` without shuffle) = what ``MultiUsdFileCfg(random_choice=False)``
  realises (``proto_prim_paths[index % len]``), used by the ``HOLOSOMA_MULTI_USD_SPAWNER=stock`` fallback;
* RECOVERY of the variant a spawned env prim actually carries (``match_usd_variant``): the env prim's USD reference
  identifiers are matched against the converted USD paths (full path first, then unique basename);
* reconciliation of planned vs recovered ids (``reconcile_variant_ids``; recovered wins, unrecovered fall back to the
  plan) and the per-file conversion cache layout (``conversion_targets``: one sub-directory per URDF so the
  ``.asset_hash`` / ``Props/instanceable_meshes.usd`` side files of the converter never collide).

The two helpers at the bottom touch ``pxr`` and import it lazily; everything else needs only torch.
"""

from __future__ import annotations

import dataclasses
import os
import re
from collections.abc import Iterable, Sequence
from pathlib import PurePath

import torch

#: ``/World/envs/env_17`` or ``/World/envs/env_17/Robot`` -> 17 (first ``env_<int>`` path component).
ENV_INDEX_RE = re.compile(r"(?:^|/)env_(\d+)(?=/|$)")

#: Env var read by ``isaacsim.py``: ``explicit`` (default; seeded shuffle realised by
#: ``multi_usd_spawner.spawn_multi_usd_explicit``) or ``stock`` (Isaac Lab ``MultiUsdFileCfg(random_choice=False)``,
#: round-robin layout).
SPAWNER_ENV_VAR = "HOLOSOMA_MULTI_USD_SPAWNER"
ENV0_VARIANT_ENV_VAR = "HOLOSOMA_MULTI_USD_ENV0_VARIANT"  # smoke knob: force env 0 to this variant id (see force_env0_variant)
SPAWNER_EXPLICIT = "explicit"
SPAWNER_STOCK = "stock"
#: Env var: ``1`` = raise instead of warn when the per-env recovery is incomplete or disagrees with the plan.
STRICT_ENV_VAR = "HOLOSOMA_MULTI_USD_STRICT"
#: Force environment 0's object to this
#: variant id (e.g. the tallest shape) so startup helpers that read env 0's prim layout / shape see the worst case.  Read by
#: ``isaacsim.py::_build_multi_usd_object_spawn``; ``STRICT_ENV_VAR`` also governs the object recovery.
OBJECT_ENV0_VARIANT_ENV_VAR = "HOLOSOMA_OBJECT_ENV0_VARIANT"
#: Leaf prim name of holosoma's per-env object actor (``/World/envs/env_i/Object``; the registry name is ``"object"``).
OBJECT_PRIM_NAME = "Object"

LAYOUT_EXPLICIT_SHUFFLED = "explicit_shuffled"
LAYOUT_STOCK_ROUND_ROBIN = "stock_round_robin"


# --------------------------------------------------------------------------------------------------
# Weights / assignment
# --------------------------------------------------------------------------------------------------
def validate_variant_weights(weights: Sequence[int] | None, num_variants: int) -> tuple[int, ...]:
    """Normalise ``urdf_variant_weights``: ``None`` -> uniform; else non-negative ints, one per variant, sum > 0."""
    if num_variants <= 0:
        raise ValueError("at least one variant (URDF file) is required")
    if weights is None:
        return (1,) * num_variants
    weights = tuple(weights)
    if len(weights) != num_variants:
        raise ValueError(f"urdf_variant_weights has {len(weights)} entries but there are {num_variants} urdf_files")
    out: list[int] = []
    for w in weights:
        if isinstance(w, bool) or not isinstance(w, int):
            if isinstance(w, float) and w.is_integer():
                w = int(w)
            else:
                raise ValueError(f"urdf_variant_weights must be integers (repetition counts), got {w!r}")
        if w < 0:
            raise ValueError(f"urdf_variant_weights must be >= 0, got {w}")
        out.append(int(w))
    if sum(out) <= 0:
        raise ValueError("urdf_variant_weights must have a positive sum")
    return tuple(out)


def expand_variant_pattern(weights: Sequence[int]) -> list[int]:
    """Repeat each variant id by its weight: ``(2, 1, 1)`` -> ``[0, 0, 1, 2]`` (== the ``usd_path`` list order that
    reproduces the proportions with Isaac Lab's ``MultiUsdFileCfg``)."""
    pattern: list[int] = []
    for vid, w in enumerate(weights):
        pattern.extend([vid] * int(w))
    if not pattern:
        raise ValueError("empty variant pattern (all weights zero)")
    return pattern


def tile_variant_pattern(pattern: Sequence[int], num_envs: int) -> list[int]:
    """Cyclic tiling ``pattern[i % len(pattern)]`` for ``i < num_envs`` -- the stock Isaac Lab round-robin layout."""
    if num_envs < 0:
        raise ValueError("num_envs must be >= 0")
    if not pattern:
        raise ValueError("empty variant pattern")
    n = len(pattern)
    return [int(pattern[i % n]) for i in range(num_envs)]


def plan_variant_assignment(
    num_envs: int, weights: Sequence[int] | None, seed: int, *, num_variants: int | None = None, shuffle: bool = True
) -> list[int]:
    """Per-env variant ids: weights -> pattern -> cyclic tiling -> (optional) seeded shuffle.

    Deterministic for a given ``(num_envs, weights, seed)``; the shuffle is a ``torch.randperm`` drawn from a private
    ``torch.Generator`` (never touches the global RNG).  Counts per variant equal those of the tiled pattern, i.e.
    ``floor``/``ceil`` of ``num_envs * w / sum(w)`` (the remainder goes to the first pattern entries).
    """
    if num_variants is None:
        if weights is None:
            raise ValueError("num_variants is required when weights is None")
        num_variants = len(weights)
    w = validate_variant_weights(weights, num_variants)
    tiled = tile_variant_pattern(expand_variant_pattern(w), num_envs)
    if not shuffle or num_envs <= 1:
        return tiled
    gen = torch.Generator(device="cpu")
    gen.manual_seed(int(seed) % (2**63 - 1))
    perm = torch.randperm(num_envs, generator=gen)
    return [tiled[int(p)] for p in perm.tolist()]


def force_env0_variant(assignment: Sequence[int], variant_id: int) -> list[int]:
    """Swap env 0 with an environment holding variant_id, preserving variant counts.

    HOLOSOMA_MULTI_USD_ENV0_VARIANT selects which collision layout startup helpers
    inspect. Return unchanged when the assignment is empty or the variant is absent."""
    out = list(assignment)
    if not out or out[0] == variant_id:
        return out
    for i, vid in enumerate(out):
        if vid == variant_id:
            out[0], out[i] = out[i], out[0]
            break
    return out


def variant_counts(assignment: Iterable[int], num_variants: int) -> list[int]:
    """Number of envs per variant id (ids outside ``[0, num_variants)`` raise)."""
    counts = [0] * num_variants
    for vid in assignment:
        if not 0 <= int(vid) < num_variants:
            raise ValueError(f"variant id {vid} outside [0, {num_variants})")
        counts[int(vid)] += 1
    return counts


def normalise_seed(seed: int | None, rank: int = 0) -> int:
    """Training seed + global rank as a non-negative int (holosoma seeds ranks with ``seed + global_rank`` too;
    ``-1`` / ``None`` = "random" in holosoma -> 0 here so the layout is still reproducible from the log)."""
    s = 0 if seed is None else int(seed)
    if s < 0:
        s = 0
    return s + max(int(rank), 0)


# --------------------------------------------------------------------------------------------------
# Recovery from the stage (pure part)
# --------------------------------------------------------------------------------------------------
def env_index_from_prim_path(prim_path: str) -> int | None:
    """``/World/envs/env_17/Robot`` -> 17; ``None`` when no ``env_<int>`` component exists."""
    m = ENV_INDEX_RE.search(str(prim_path))
    return int(m.group(1)) if m else None


def _norm(path: str) -> str:
    p = str(path)
    # USD identifiers may carry a ``file:`` scheme or anchored-layer args (``foo.usd:SDF_FORMAT_ARGS:...``).
    if p.startswith("file://"):
        p = p[len("file://"):]
    elif p.startswith("file:"):
        p = p[len("file:"):]
    p = p.split(":SDF_FORMAT_ARGS:", 1)[0]
    return os.path.normpath(p)


def match_usd_variant(identifiers: Iterable[str], usd_paths: Sequence[str]) -> int | None:
    """Identify the USD variant referenced by an environment prim.

    Match normalized full paths first, then unique basenames. Return None if no
    variant matches or multiple variants are referenced."""
    if not usd_paths:
        return None
    norm_paths = [_norm(p) for p in usd_paths]
    norm_bases = [os.path.basename(p) for p in norm_paths]
    base_unique = len(set(norm_bases)) == len(norm_bases)
    found: set[int] = set()
    for ident in identifiers:
        if not ident:
            continue
        n = _norm(ident)
        hit = [i for i, p in enumerate(norm_paths) if p == n]
        if not hit and base_unique:
            b = os.path.basename(n)
            hit = [i for i, nb in enumerate(norm_bases) if nb == b]
        found.update(hit)
    if len(found) != 1:
        return None
    return found.pop()


@dataclasses.dataclass(frozen=True)
class ReconcileResult:
    """Outcome of :func:`reconcile_variant_ids`."""

    final: list[int]
    n_unrecovered: int
    n_mismatch: int
    mismatched_envs: list[int]

    @property
    def ok(self) -> bool:
        return self.n_unrecovered == 0 and self.n_mismatch == 0


def reconcile_variant_ids(planned: Sequence[int], recovered: Sequence[int | None]) -> ReconcileResult:
    """Merge the planned assignment with what was read back from the stage: recovered wins where available,
    unrecovered envs keep the plan.  Lengths must match."""
    if len(planned) != len(recovered):
        raise ValueError(f"planned ({len(planned)}) and recovered ({len(recovered)}) lengths differ")
    final: list[int] = []
    mism: list[int] = []
    n_unrec = 0
    for i, (p, r) in enumerate(zip(planned, recovered, strict=True)):
        if r is None:
            n_unrec += 1
            final.append(int(p))
        else:
            if int(r) != int(p):
                mism.append(i)
            final.append(int(r))
    return ReconcileResult(final=final, n_unrecovered=n_unrec, n_mismatch=len(mism), mismatched_envs=mism)


# --------------------------------------------------------------------------------------------------
# Conversion cache layout / names
# --------------------------------------------------------------------------------------------------
@dataclasses.dataclass(frozen=True)
class ConversionTarget:
    """One URDF -> USD conversion: absolute URDF path, its private cache dir and the USD file name inside it."""

    variant_id: int
    stem: str
    urdf_path: str
    usd_dir: str
    usd_file_name: str

    @property
    def usd_path(self) -> str:
        return os.path.join(self.usd_dir, self.usd_file_name)


def conversion_targets(asset_root: str, conversion_dir: str, urdf_files: Sequence[str]) -> list[ConversionTarget]:
    """Per-file cache layout ``<conversion_dir>/<stem>/<stem>.usd`` (stems must be unique; the single-URDF path keeps
    writing ``<conversion_dir>/<stem>.usd`` so the two never overlap)."""
    if not urdf_files:
        raise ValueError("urdf_files is empty")
    stems = [PurePath(f).stem for f in urdf_files]
    if len(set(stems)) != len(stems):
        raise ValueError(f"urdf_files must have unique file stems (they key the USD cache): {stems}")
    out: list[ConversionTarget] = []
    for vid, (f, stem) in enumerate(zip(urdf_files, stems, strict=True)):
        urdf_path = os.path.abspath(os.path.join(asset_root, f))
        usd_dir = os.path.abspath(os.path.join(conversion_dir, stem))
        out.append(ConversionTarget(variant_id=vid, stem=stem, urdf_path=urdf_path, usd_dir=usd_dir, usd_file_name=f"{stem}.usd"))
    return out


def object_conversion_targets(conversion_dir: str, urdf_paths: Sequence[str]) -> list[ConversionTarget]:
    """Return object-variant conversion paths as <conversion_dir>/<stem>/<stem>.usd.

    Use a subdirectory per variant to keep single-object and multi-object caches separate."""
    return conversion_targets("", conversion_dir, [os.path.abspath(str(p)) for p in urdf_paths])


def parse_urdf_box_size(urdf_path: str) -> tuple[float, float, float] | None:
    """``<collision><geometry><box size="x y z"/>`` of the first link (visual box as fallback) or ``None`` when the URDF has no
    box primitive.  Publishes ``simulator.object_variant_sizes`` without any catalogue dependency (the S-2 box assets are
    primitive boxes; a mesh-based object yields ``None`` -> NaN row)."""
    import xml.etree.ElementTree as ET  # noqa: PLC0415 - stdlib, lazy

    root = ET.parse(str(urdf_path)).getroot()
    for tag in ("collision", "visual"):
        for link in root.iter("link"):
            for geom in link.iter(tag):
                box = geom.find("geometry/box")
                if box is not None and box.get("size"):
                    vals = tuple(float(v) for v in box.get("size").split())
                    if len(vals) == 3:
                        return vals  # type: ignore[return-value]
    return None


def variant_names_for(urdf_files: Sequence[str], names: Sequence[str] | None = None) -> tuple[str, ...]:
    """``urdf_variant_names`` validated against ``urdf_files`` (same length, unique); ``None`` -> file stems."""
    if names is None:
        out = tuple(PurePath(f).stem for f in urdf_files)
    else:
        out = tuple(str(n) for n in names)
        if len(out) != len(urdf_files):
            raise ValueError(f"urdf_variant_names has {len(out)} entries but there are {len(urdf_files)} urdf_files")
    if len(set(out)) != len(out):
        raise ValueError(f"variant names must be unique: {out}")
    return out


def format_variant_summary(names: Sequence[str], counts: Sequence[int]) -> str:
    """``dex3=2048 (50.0%), rubber=1024 (25.0%), none=1024 (25.0%)`` for the start-up log."""
    total = max(int(sum(counts)), 1)
    return ", ".join(f"{n}={c} ({100.0 * c / total:.1f}%)" for n, c in zip(names, counts, strict=True))


# --------------------------------------------------------------------------------------------------
# Recovery from the stage (pxr part; lazy imports so the module stays importable without Isaac)
# --------------------------------------------------------------------------------------------------
def collect_prim_reference_identifiers(prim) -> list[str]:
    """Every USD identifier that can tell which file a prim references: authored ``references`` list-op items
    (all list positions), the layers of its prim stack, and the reference arcs of a composition query.

    Best-effort: each source is wrapped in its own ``try`` because the exact ``Sdf`` list-op accessors differ across
    USD releases; an empty list means "unknown" (the caller falls back to the planned assignment).
    """
    out: list[str] = []

    def _add(v) -> None:
        if v:
            s = str(v)
            if s and s not in out:
                out.append(s)

    # 1) authored references metadata (Sdf.ReferenceListOp) -- what prim_utils.create_prim(usd_path=...) writes
    try:
        lo = prim.GetMetadata("references")
        if lo is not None:
            for attr in ("GetAddedOrExplicitItems", "prependedItems", "appendedItems", "explicitItems", "addedItems", "orderedItems"):
                try:
                    items = getattr(lo, attr)
                    items = items() if callable(items) else items
                    for ref in items or []:
                        _add(getattr(ref, "assetPath", None))
                except Exception:  # noqa: BLE001 - accessor may not exist in this USD build
                    continue
    except Exception:  # noqa: BLE001
        pass
    # 2) prim stack: the spec authored in the referenced layer shows up with that layer's identifier
    try:
        for spec in prim.GetPrimStack():
            try:
                _add(spec.layer.identifier)
            except Exception:  # noqa: BLE001
                pass
            try:
                for ref in spec.referenceList.GetAddedOrExplicitItems():
                    _add(getattr(ref, "assetPath", None))
            except Exception:  # noqa: BLE001
                pass
    except Exception:  # noqa: BLE001
        pass
    # 3) composition query (reference arcs -> target layer identifier)
    try:
        from pxr import Pcp, Usd  # noqa: WPS433 - lazy on purpose

        q = Usd.PrimCompositionQuery(prim)
        for arc in q.GetCompositionArcs():
            try:
                if arc.GetArcType() != Pcp.ArcTypeReference:
                    continue
                node = arc.GetTargetNode()
                _add(node.layerStack.identifier.rootLayer.identifier)
            except Exception:  # noqa: BLE001
                continue
    except Exception:  # noqa: BLE001
        pass
    return out


def recover_variant_ids_from_stage(
    stage, num_envs: int, usd_paths: Sequence[str], *, env_ns: str = "/World/envs", robot_prim_name: str = "Robot"
) -> list[int | None]:
    """``[match_usd_variant(identifiers of /World/envs/env_i/Robot) for i in range(num_envs)]`` (``None`` = unknown)."""
    out: list[int | None] = []
    for i in range(int(num_envs)):
        path = f"{env_ns}/env_{i}/{robot_prim_name}"
        try:
            prim = stage.GetPrimAtPath(path)
            if not prim or not prim.IsValid():
                out.append(None)
                continue
            out.append(match_usd_variant(collect_prim_reference_identifiers(prim), usd_paths))
        except Exception:  # noqa: BLE001
            out.append(None)
    return out


__all__ = [
    "ConversionTarget",
    "ENV_INDEX_RE",
    "LAYOUT_EXPLICIT_SHUFFLED",
    "LAYOUT_STOCK_ROUND_ROBIN",
    "OBJECT_ENV0_VARIANT_ENV_VAR",
    "OBJECT_PRIM_NAME",
    "ReconcileResult",
    "SPAWNER_ENV_VAR",
    "SPAWNER_EXPLICIT",
    "SPAWNER_STOCK",
    "STRICT_ENV_VAR",
    "collect_prim_reference_identifiers",
    "conversion_targets",
    "env_index_from_prim_path",
    "expand_variant_pattern",
    "format_variant_summary",
    "match_usd_variant",
    "normalise_seed",
    "object_conversion_targets",
    "parse_urdf_box_size",
    "plan_variant_assignment",
    "reconcile_variant_ids",
    "recover_variant_ids_from_stage",
    "tile_variant_pattern",
    "validate_variant_weights",
    "variant_counts",
    "variant_names_for",
]
