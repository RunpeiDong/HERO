#!/usr/bin/env python3
"""Generate IK reaching data separately from the AMASS Quick Start corpus.

Extend profiles or pass repeated --override KEY=VALUE options to build additional
motions.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import _bootstrap  # noqa: F401


def main(argv=None) -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--num-clips", type=int, default=4)
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--profile", default="broad_v2")
    parser.add_argument("--jobs", type=int, default=1, help="Parallel single-threaded IK workers; use the CPU cores available to this process (inside a container that is the CPU request, not the host core count)")
    parser.add_argument("--override", action="append", default=[], metavar="KEY=VALUE")
    parser.add_argument("--specs-only", action="store_true", help="Inspect sampled targets without solving IK")
    args = parser.parse_args(argv)
    if args.num_clips < 1 or not 1 <= args.jobs <= 64:
        parser.error("num-clips must be positive and jobs must be 1..64")
    if args.output.exists():
        parser.error("output must be a new directory")
    from data_tools.hero_reach_generator import main as generate
    command = ["--out-dir", str(args.output), "--n", str(args.num_clips), "--seed", str(args.seed),
        "--profile", args.profile, "--jobs", str(args.jobs)]
    for value in args.override:
        command += ["--override", value]
    if args.specs_only:
        command.append("--specs-only")
    return generate(command)


if __name__ == "__main__":
    raise SystemExit(main())
