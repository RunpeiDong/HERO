"""Locations shared by training, data preparation, and the HERO demo.

The source checkout and an installed wheel use the same packaged assets.
Environment overrides are resolved once when this module is imported."""
from __future__ import annotations

import os
from pathlib import Path


def _path(variable: str, default: Path) -> Path:
    return Path(os.environ.get(variable, str(default))).expanduser().resolve()


RELEASE_ROOT = _path("HERO_ROOT", Path(__file__).resolve().parents[1])


def _default_assets() -> Path:
    checkout_assets = RELEASE_ROOT / "assets"
    if (checkout_assets / "robots" / "g1_modified").is_dir():
        return checkout_assets
    # setup.py maps the assets/ source directory to this resource-only package.
    import hero_assets

    return Path(hero_assets.__file__).resolve().parent


ASSETS_ROOT = (_path("HERO_ASSETS_ROOT", Path(".")) if os.environ.get("HERO_ASSETS_ROOT") else _default_assets())
ROBOT_ASSET_ROOT = _path("HERO_ROBOT_ASSET_ROOT", ASSETS_ROOT / "robots")
G1_ASSET_ROOT = ROBOT_ASSET_ROOT / "g1_modified"


def _default_checkpoints() -> Path:
    checkout_checkpoints = RELEASE_ROOT / "checkpoints"
    if checkout_checkpoints.is_dir():
        return checkout_checkpoints
    import hero_checkpoints

    return Path(hero_checkpoints.__file__).resolve().parent


CHECKPOINTS_ROOT = (_path("HERO_CHECKPOINTS_ROOT", Path("."))
    if os.environ.get("HERO_CHECKPOINTS_ROOT") else _default_checkpoints())
