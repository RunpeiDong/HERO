"""Shared tensor mathematics and constants for HERO managers."""

from __future__ import annotations

from types import ModuleType

from hero_isaacsim import constants as _constants


def hero_constants() -> ModuleType:
    """Return ``hero_isaacsim.constants`` (joint/body orders, EE definition, PALM_OFFSET, ...)."""
    return _constants


__all__ = ["hero_constants"]
