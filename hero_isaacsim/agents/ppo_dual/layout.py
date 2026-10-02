"""Convert observation histories between term-major and frame-major layouts."""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any

import torch

HISTORY_LAYOUT_FRAME_MAJOR = "frame_major_hero_v1"
"""HERO inference contract: ``[frame t-H+1 | ... | frame t]``, each frame sorted terms."""

HISTORY_LAYOUT_TERM_MAJOR = "term_major_holosoma_v1"
"""holosoma native contract: ``[term_0 x H | term_1 x H | ...]``, terms sorted, oldest frame first."""

SUPPORTED_HISTORY_LAYOUTS = (HISTORY_LAYOUT_FRAME_MAJOR, HISTORY_LAYOUT_TERM_MAJOR)


# Term names carry the ``hNN_`` prefix so that sorted order == HERO order.
HERO_H1_ACTOR_TERM_DIMS: dict[str, int] = {
    "h00_actions": 29,
    "h01_base_ang_vel": 3,
    "h02_command_ang_vel": 1,
    "h03_command_base_height": 1,
    "h04_command_lin_vel": 2,
    "h05_command_stand": 1,
    "h06_command_waist_dofs": 3,
    "h07_dif_local_rigid_body_pos_ee": 6,
    "h08_dif_local_rigid_body_rot_ee": 12,
    "h09_dof_pos": 29,
    "h10_dof_vel": 29,
    "h11_projected_gravity": 3,
    "h12_ref_upper_dof_pos": 14,
    "h13_roll_and_pitch": 2,
}
HERO_H1_ACTOR_HISTORY_LENGTH = 5


@dataclass(frozen=True)
class ActorObsLayout:
    """Resolved layout of one concatenated observation group with history.

    ``terms`` are in the exact concatenation order (sorted by name), ``term_dims``
    are single-frame dims, ``history_length`` the number of stacked frames
    (oldest first, current frame last)."""

    terms: tuple[str, ...]
    term_dims: tuple[int, ...]
    history_length: int
    group_name: str = "actor_obs"

    def __post_init__(self) -> None:
        if len(self.terms) != len(self.term_dims):
            raise ValueError(f"terms ({len(self.terms)}) and term_dims ({len(self.term_dims)}) length mismatch")
        if list(self.terms) != sorted(self.terms):
            raise ValueError(f"terms must be in sorted order (holosoma/HERO contract), got {list(self.terms)}")
        if len(set(self.terms)) != len(self.terms):
            raise ValueError(f"duplicate term names in layout: {list(self.terms)}")
        if self.history_length < 1:
            raise ValueError(f"history_length must be >= 1, got {self.history_length}")
        if any(int(d) <= 0 for d in self.term_dims):
            raise ValueError(f"every term dim must be > 0, got {list(self.term_dims)}")

    # ------------------------------------------------------------------ dims
    @property
    def num_terms(self) -> int:
        return len(self.terms)

    @property
    def frame_dim(self) -> int:
        """``D``: width of one frame (sum of single-frame term dims)."""
        return int(sum(self.term_dims))

    @property
    def total_dim(self) -> int:
        """``H * D``: width of the concatenated group vector."""
        return self.frame_dim * self.history_length

    @property
    def term_offsets(self) -> tuple[int, ...]:
        """``o_i``: offset of term ``i`` inside one frame."""
        offsets = []
        acc = 0
        for d in self.term_dims:
            offsets.append(acc)
            acc += int(d)
        return tuple(offsets)

    def term_index(self, name: str) -> int:
        return self.terms.index(name)

    # --------------------------------------------------------------- indices
    def term_major_index(self, i: int, f: int, k: int) -> int:
        """Flat index of (term i, frame f, element k) in the holosoma term-major vector."""
        self._check_ifk(i, f, k)
        return self.history_length * self.term_offsets[i] + f * int(self.term_dims[i]) + k

    def frame_major_index(self, i: int, f: int, k: int) -> int:
        """Flat index of (term i, frame f, element k) in the HERO frame-major vector."""
        self._check_ifk(i, f, k)
        return f * self.frame_dim + self.term_offsets[i] + k

    def _check_ifk(self, i: int, f: int, k: int) -> None:
        if not 0 <= i < self.num_terms:
            raise IndexError(f"term index {i} out of range [0, {self.num_terms})")
        if not 0 <= f < self.history_length:
            raise IndexError(f"frame index {f} out of range [0, {self.history_length})")
        if not 0 <= k < int(self.term_dims[i]):
            raise IndexError(f"element index {k} out of range [0, {self.term_dims[i]}) for term {self.terms[i]}")

    # ----------------------------------------------------------- permutation
    def frame_major_to_term_major_perm(self, device: torch.device | str | None = None) -> torch.Tensor:
        """Index vector ``P`` (long, ``[H*D]``) such that ``term_major = frame_major[..., P]``.

        ``P[tm(i,f,k)] = fm(i,f,k)``.  Apply with ``x.index_select(-1, P)`` (exports as ONNX ``Gather``)."""
        perm = torch.empty(self.total_dim, dtype=torch.long)
        h = self.history_length
        d_frame = self.frame_dim
        for i, (d, o) in enumerate(zip(self.term_dims, self.term_offsets)):
            d = int(d)
            for f in range(h):
                tm0 = h * o + f * d
                fm0 = f * d_frame + o
                perm[tm0 : tm0 + d] = torch.arange(fm0, fm0 + d, dtype=torch.long)
        return perm.to(device) if device is not None else perm

    def term_major_to_frame_major_perm(self, device: torch.device | str | None = None) -> torch.Tensor:
        """Inverse permutation ``Q`` such that ``frame_major = term_major[..., Q]``."""
        p = self.frame_major_to_term_major_perm()
        q = torch.empty_like(p)
        q[p] = torch.arange(self.total_dim, dtype=torch.long)
        return q.to(device) if device is not None else q

    def frame_major_to_term_major(self, x: torch.Tensor) -> torch.Tensor:
        """Convert a frame-major vector/batch ``[..., H*D]`` to term-major."""
        self._check_width(x)
        return x.index_select(-1, self.frame_major_to_term_major_perm(x.device))

    def term_major_to_frame_major(self, x: torch.Tensor) -> torch.Tensor:
        """Convert a term-major vector/batch ``[..., H*D]`` to frame-major."""
        self._check_width(x)
        return x.index_select(-1, self.term_major_to_frame_major_perm(x.device))

    def _check_width(self, x: torch.Tensor) -> None:
        if x.shape[-1] != self.total_dim:
            raise ValueError(
                f"expected last dim {self.total_dim} (H={self.history_length} x D={self.frame_dim}), got {x.shape}"
            )

    # ------------------------------------------------- per-term <-> flat vectors
    def flatten_term_major(self, per_term: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Build the holosoma term-major vector from ``{term: [N, H, d_i]}`` (frame axis oldest-first)."""
        parts = [
            self._per_term_tensor(per_term, i).reshape(per_term[self.terms[i]].shape[0], -1)
            for i in range(self.num_terms)
        ]
        return torch.cat(parts, dim=-1)

    def flatten_frame_major(self, per_term: Mapping[str, torch.Tensor]) -> torch.Tensor:
        """Build the HERO frame-major vector from ``{term: [N, H, d_i]}`` (frame axis oldest-first)."""
        frames = torch.cat([self._per_term_tensor(per_term, i) for i in range(self.num_terms)], dim=-1)  # [N, H, D]
        return frames.reshape(frames.shape[0], -1)

    def split_term_major(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Inverse of :meth:`flatten_term_major`: ``[N, H*D] -> {term: [N, H, d_i]}``."""
        self._check_width(x)
        out: dict[str, torch.Tensor] = {}
        h = self.history_length
        for name, d, o in zip(self.terms, self.term_dims, self.term_offsets):
            d = int(d)
            out[name] = x[..., h * o : h * o + h * d].reshape(*x.shape[:-1], h, d)
        return out

    def split_frame_major(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        """Inverse of :meth:`flatten_frame_major`: ``[N, H*D] -> {term: [N, H, d_i]}``."""
        self._check_width(x)
        frames = x.reshape(*x.shape[:-1], self.history_length, self.frame_dim)
        return {
            name: frames[..., o : o + int(d)] for name, d, o in zip(self.terms, self.term_dims, self.term_offsets)
        }

    def _per_term_tensor(self, per_term: Mapping[str, torch.Tensor], i: int) -> torch.Tensor:
        name = self.terms[i]
        if name not in per_term:
            raise KeyError(f"missing term {name!r} in per-term data")
        t = per_term[name]
        if t.dim() != 3 or t.shape[1] != self.history_length or t.shape[2] != int(self.term_dims[i]):
            raise ValueError(
                f"term {name!r}: expected shape [N, {self.history_length}, {self.term_dims[i]}], got {tuple(t.shape)}"
            )
        return t

    # -------------------------------------------------------------- metadata
    def to_metadata(self, history_layout: str = HISTORY_LAYOUT_FRAME_MAJOR) -> dict[str, Any]:
        """JSON-serialisable description written into the ONNX metadata."""
        if history_layout not in SUPPORTED_HISTORY_LAYOUTS:
            raise ValueError(f"unknown history_layout {history_layout!r}; expected one of {SUPPORTED_HISTORY_LAYOUTS}")
        meta: dict[str, Any] = {
            "group": self.group_name,
            "terms": list(self.terms),
            "term_dims": [int(d) for d in self.term_dims],
            "term_offsets_in_frame": list(self.term_offsets),
            "history_length": int(self.history_length),
            "frame_dim": self.frame_dim,
            "total_dim": self.total_dim,
            "frame_order": "oldest_first",
            "history_layout": history_layout,
        }
        if history_layout == HISTORY_LAYOUT_FRAME_MAJOR:
            # The permutation the graph applies to its (frame-major) input before the MLPs.
            meta["frame_major_to_term_major_perm"] = self.frame_major_to_term_major_perm().tolist()
        return meta


# ----------------------------------------------------------------------------- builders
def layout_from_term_dims(
    term_dims: Mapping[str, int], history_length: int, group_name: str = "actor_obs"
) -> ActorObsLayout:
    """Build a layout from ``{term_name: single_frame_dim}`` (any order; sorted internally)."""
    names = sorted(term_dims.keys())
    return ActorObsLayout(
        terms=tuple(names),
        term_dims=tuple(int(term_dims[n]) for n in names),
        history_length=int(history_length),
        group_name=group_name,
    )


def hero_h1_actor_layout() -> ActorObsLayout:
    """The HERO actor layout: 14 terms, 135 x 5 = 675."""
    return layout_from_term_dims(HERO_H1_ACTOR_TERM_DIMS, HERO_H1_ACTOR_HISTORY_LENGTH)


def term_dims_from_observation_manager(manager: Any, group_name: str) -> dict[str, int]:
    """Single-frame dim of every term of ``group_name``.

    Uses ``manager.get_term_dims(group_name)`` when the manager offers it, else the
    same ``_compute_term`` path ``ObservationManager.get_obs_dims`` uses."""
    getter = getattr(manager, "get_term_dims", None)
    if callable(getter):
        dims = getter(group_name)
        return {str(k): int(v) for k, v in dims.items()}
    group_cfg = manager.cfg.groups[group_name]
    dims = {}
    for term_name, term_cfg in group_cfg.terms.items():
        obs = manager._compute_term(group_name, term_name, term_cfg)
        dims[term_name] = int(obs.shape[-1])
    return dims


def layout_from_env(
    env: Any,
    group_name: str = "actor_obs",
    term_dims: Mapping[str, int] | None = None,
) -> ActorObsLayout:
    """Resolve the layout of one concatenated observation group from a live env.

    Mirrors ``holosoma.utils.inference_helpers.actor_obs_layout_from_env`` (sorted
    term names + group history) and additionally resolves the per-term dims that
    the frame-major permutation needs.  ``term_dims`` may be passed to skip the
    term evaluation (e.g. in tests)."""
    manager = getattr(env, "observation_manager", None)
    if manager is None:
        raise ValueError("env has no observation_manager; cannot resolve the actor obs layout")
    group_cfg = manager.cfg.groups[group_name]
    if not getattr(group_cfg, "concatenate", True):
        raise ValueError(f"group {group_name!r} is not concatenated; frame-major layout undefined")
    history_length = int(getattr(group_cfg, "history_length", 1))
    dims = dict(term_dims) if term_dims is not None else term_dims_from_observation_manager(manager, group_name)
    expected = set(group_cfg.terms.keys())
    if set(dims.keys()) != expected:
        raise ValueError(f"term_dims keys {sorted(dims)} do not match group terms {sorted(expected)}")
    layout = layout_from_term_dims(dims, history_length, group_name)
    obs_buf = getattr(env, "obs_buf_dict", None) or {}
    group_tensor = obs_buf.get(group_name) if isinstance(obs_buf, dict) else None
    if group_tensor is not None and int(group_tensor.shape[-1]) != layout.total_dim:
        raise ValueError(
            f"resolved layout width {layout.total_dim} != env obs_buf_dict[{group_name!r}] "
            f"width {group_tensor.shape[-1]}"
        )
    return layout


def permutation_is_bijection(perm: torch.Tensor) -> bool:
    """True iff ``perm`` is a permutation of ``0..len(perm)-1``."""
    if perm.dim() != 1:
        return False
    return bool(torch.equal(torch.sort(perm).values, torch.arange(perm.numel(), dtype=perm.dtype, device=perm.device)))


__all__: Sequence[str] = [
    "ActorObsLayout",
    "HERO_H1_ACTOR_HISTORY_LENGTH",
    "HERO_H1_ACTOR_TERM_DIMS",
    "HISTORY_LAYOUT_FRAME_MAJOR",
    "HISTORY_LAYOUT_TERM_MAJOR",
    "SUPPORTED_HISTORY_LAYOUTS",
    "hero_h1_actor_layout",
    "layout_from_env",
    "layout_from_term_dims",
    "permutation_is_bijection",
    "term_dims_from_observation_manager",
]
