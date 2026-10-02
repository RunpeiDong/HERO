"""Load the HERO export selected for the browser scene."""
from pathlib import Path
from hero_isaacsim.paths import CHECKPOINTS_ROOT
from sim2sim.policy_hero_export import HeroExportPolicy

DEFAULT_HERO_POLICY = CHECKPOINTS_ROOT / "example"

def load_demo_policy(mode="hero_plus", policy_path=None):
    if mode != "hero_plus":
        raise ValueError("The demo requires a HERO policy.")
    # The exported term layout determines which anchor inputs to build.
    return HeroExportPolicy(policy_path or DEFAULT_HERO_POLICY, stand_flag=0.)
