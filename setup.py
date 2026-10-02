"""Discover Python packages and map the asset directories to package names."""
from setuptools import find_packages, setup


setup(
    packages=find_packages(include=("hero_isaacsim*", "sim2sim*", "data_tools*", "configs*"))
    + ["hero_assets", "hero_checkpoints"],
    package_dir={"hero_assets": "assets", "hero_checkpoints": "checkpoints"},
)
