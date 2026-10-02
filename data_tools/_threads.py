"""Keep the numerical libraries single-threaded inside data-tool processes.

Import this module before numpy, MuJoCo, or Mink in every data-tools entry point. BLAS and OpenMP pools size
themselves at import time from these variables (defaulting to the host core count, even inside a container
with a smaller CPU request), so N process workers would otherwise each start one thread per host core: a
64-worker reaching-data generator ran at about 20 clips per minute that way and at about 560 once each
worker was single-threaded. Spawned worker processes re-import their module, so the defaults reach them too.
Values already present in the environment are respected."""
from __future__ import annotations

import os

THREAD_VARS = (
    "OMP_NUM_THREADS",
    "OPENBLAS_NUM_THREADS",
    "MKL_NUM_THREADS",
    "NUMEXPR_NUM_THREADS",
    "VECLIB_MAXIMUM_THREADS",
)


def single_threaded_math() -> dict[str, str]:
    """Set every thread-count variable that is not already defined to ``"1"``; return the resulting values."""
    for name in THREAD_VARS:
        os.environ.setdefault(name, "1")
    return {name: os.environ[name] for name in THREAD_VARS}


single_threaded_math()
