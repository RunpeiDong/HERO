"""Default termination manager configurations."""

from holosoma.config_values.loco.g1.termination import g1_29dof_termination
from holosoma.config_values.loco.t1.termination import t1_29dof_termination
from holosoma.config_values.wbt.g1.termination import (
    g1_29dof_wbt_termination,
    g1_29dof_wbt_termination_gated,
    g1_29dof_wbt_termination_relaxed,
    g1_29dof_wbt_termination_retgtobj,
)

none = None

DEFAULTS = {
    "none": none,
    "t1_29dof": t1_29dof_termination,
    "g1_29dof": g1_29dof_termination,
    "g1_29dof_wbt": g1_29dof_wbt_termination,
    "g1_29dof_wbt_relaxed": g1_29dof_wbt_termination_relaxed,
    "g1_29dof_wbt_retgtobj": g1_29dof_wbt_termination_retgtobj,
    "g1_29dof_wbt_gated": g1_29dof_wbt_termination_gated,
}
