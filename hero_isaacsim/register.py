"""Register HERO training configurations with holosoma."""
from hero_isaacsim.config_values.experiment import DEFAULTS

ABLATION_DEFAULTS = {}
P0_EXPERIMENT_NAMES = tuple(DEFAULTS)
ABLATIONS_FROM_ENV = False


def cli_name(preset_key: str) -> str:
    return "exp:" + preset_key.replace("_", "-")


def register(include_ablations=False, subconfigs=True, overwrite=False):
    from holosoma.config_values import experiment
    experiment.DEFAULTS.update(DEFAULTS)
    return {"experiment": list(DEFAULTS)}


def registered_experiment_names():
    return list(DEFAULTS)
