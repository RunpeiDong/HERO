"""Shared training/evaluation gates without a policy-family dependency."""
from typing import Any


def training_noise_active(env: Any) -> bool:
    return not bool(getattr(env, "is_evaluating", False))
