"""Object state helpers for holosoma motion-command compatibility."""
from __future__ import annotations

from typing import Any, Mapping, Sequence

import torch

OBJECT_ACTOR_NAME = "object"  # holosoma's hard-coded actor name (wbt.py)

DEFAULT_BOX_SIZE_M = 0.30

OBJECT_PARK_OFFSET_M: tuple[float, float, float] = (0.0, 1000.0, round(0.5 * DEFAULT_BOX_SIZE_M + 0.01, 6))

TERRAIN_PARK_GROUND_Z_M = 0.0

_IDENTITY_QUAT_XYZW = (0.0, 0.0, 0.0, 1.0)

def parked_object_states(
    origins: torch.Tensor,
    park_offset: Sequence[float] = OBJECT_PARK_OFFSET_M,
    *,
    park_z: Any = None,
    park_xy: torch.Tensor | None = None,
    park_ground_z: float = TERRAIN_PARK_GROUND_Z_M,
) -> torch.Tensor:
    """``[K, 13]`` root state ``[origin + park_offset | identity xyzw | zero lin/ang vel]`` (the
    offset :data:`OBJECT_PARK_OFFSET_M` parks the object to the side, on the floor).  ``park_z`` (float or
    ``[K]`` tensor) replaces the z component per env.

    ``park_xy`` (``[K, 2]``) replaces the whole
    lateral rule: the box goes to that world xy with NO offset, standing on ``park_ground_z`` (the flat border band, 0)
    instead of the env origin's z -- the height component stays ``park_offset[2]`` / ``park_z``.  ``None`` = unchanged."""
    k = origins.shape[0]
    off = [float(v) for v in park_offset]
    if len(off) != 3:
        raise ValueError(f"park_offset must have 3 components, got {park_offset!r}")
    pos = origins + torch.tensor(off, device=origins.device, dtype=origins.dtype)
    base_z = origins[:, 2]
    if park_xy is not None:
        xy = torch.as_tensor(park_xy, dtype=origins.dtype, device=origins.device).reshape(-1, 2)
        if xy.shape[0] != k:
            raise ValueError(f"park_xy must have {k} rows, got {tuple(xy.shape)}")
        base_z = torch.full_like(origins[:, 2], float(park_ground_z))
        pos = torch.cat([xy, (base_z + off[2])[:, None]], dim=1)
    if park_z is not None:
        z = torch.as_tensor(park_z, dtype=origins.dtype, device=origins.device).reshape(-1)
        if z.numel() not in (1, k):
            raise ValueError(f"park_z must be a scalar or have {k} entries, got {z.numel()}")
        pos = pos.clone()
        pos[:, 2] = base_z + z
    quat = torch.tensor(_IDENTITY_QUAT_XYZW, device=origins.device, dtype=origins.dtype).expand(k, 4)
    vel = torch.zeros(k, 6, device=origins.device, dtype=origins.dtype)
    return torch.cat([pos, quat, vel], dim=-1)
