#!/usr/bin/env python3
"""Convert user-provided, G1-retargeted AMASS motions into the HERO format."""
import _bootstrap  # noqa: F401
from data_tools.prepare_amass import main

if __name__ == "__main__":
    raise SystemExit(main())
