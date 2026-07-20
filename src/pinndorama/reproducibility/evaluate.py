"""Dispatch explicit checkpoint evaluation to one publication solver."""

from __future__ import annotations

import argparse
from pathlib import Path
import subprocess
import sys

from .._paths import REPO_ROOT

SOLVER_MODULES = {
    "toy-problem-1d": "pinndorama.solvers.toy_problem_1d.evaluate",
    "punctures-2d": "pinndorama.solvers.punctures_2d.evaluate",
    "punctures-2d-parametric": "pinndorama.solvers.punctures_2d_parametric.evaluate",
    "punctures-3d": "pinndorama.solvers.punctures_3d.evaluate",
}


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="solver", required=True)
    for name in SOLVER_MODULES:
        subparser = subparsers.add_parser(name)
        subparser.add_argument("--config", type=Path, required=True)
        subparser.add_argument("--checkpoint", type=Path, required=True)
        subparser.add_argument("--coords", type=Path, required=True)
        subparser.add_argument("--output", type=Path, required=True)
        if name == "punctures-2d-parametric":
            subparser.add_argument(
                "--equal-spin-sz",
                type=float,
                required=True,
                help="Raw Bowen-York S_z with S0_z=S1_z.",
            )
    return parser


def main(argv: list[str] | None = None) -> int:
    args = build_parser().parse_args(argv)
    command = [
        sys.executable,
        "-m",
        SOLVER_MODULES[args.solver],
        "--config",
        str(args.config),
        "--checkpoint",
        str(args.checkpoint),
        "--coords",
        str(args.coords),
        "--output",
        str(args.output),
    ]
    if args.solver == "punctures-2d-parametric":
        command.extend(("--equal-spin-sz", repr(args.equal_spin_sz)))
    subprocess.run(command, cwd=REPO_ROOT, check=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
