"""Run the canonical reader and compute the publication relative-L2 metric."""

from __future__ import annotations

import argparse
import json
import os
from pathlib import Path
import subprocess
import tempfile

import numpy as np

from .metrics import volume_weighted_relative_l2
from .reference import (
    assert_same_problem,
    load_reference_metadata,
    load_solver_problem,
    sha256_file,
)
from .._paths import NRPYELLIPTIC_READER

DEFAULT_READER = NRPYELLIPTIC_READER


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--nrpy-binary", type=Path, required=True)
    parser.add_argument("--reference-config", type=Path, required=True)
    parser.add_argument("--solver-config", type=Path, required=True)
    parser.add_argument("--coords", type=Path, required=True)
    parser.add_argument(
        "--nn-values",
        type=Path,
        required=True,
        help="Text table whose final column contains checkpoint predictions.",
    )
    parser.add_argument(
        "--volume-weights",
        type=Path,
        required=True,
        help="Text file containing one positive quadrature weight per point.",
    )
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--reader", type=Path, default=DEFAULT_READER)
    parser.add_argument(
        "--equal-spin-sz",
        type=float,
        help="Raw Bowen-York S_z for a parametric checkpoint member.",
    )
    return parser.parse_args()


def last_column(path: Path, *, skip_header: bool = False) -> np.ndarray:
    values = np.loadtxt(path, comments="#", ndmin=2)
    if values.size == 0:
        raise ValueError(f"{path} contains no numeric rows")
    return np.asarray(values[:, -1], dtype=np.float64)


def atomic_json(path: Path, document: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            json.dump(document, stream, sort_keys=True, indent=2)
            stream.write("\n")
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    args = parse_args()
    binary = args.nrpy_binary.expanduser().resolve()
    reference = load_reference_metadata(args.reference_config, binary)
    geometry, physics = load_solver_problem(
        args.solver_config, equal_spin_sz=args.equal_spin_sz
    )
    assert_same_problem(reference, geometry, physics)
    reader = args.reader.expanduser().resolve()
    if not reader.is_file():
        raise FileNotFoundError(
            f"canonical reader not found at {reader}; build it with "
            "`make -C reference_solvers/nrpyelliptic/"
            "READER_nrpyelliptic_conformally_flat`"
        )

    output_path = args.output.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="pinndorama-reader-", dir=output_path.parent
    ) as temporary:
        nrpy_values_path = Path(temporary) / "nrpy_values.txt"
        subprocess.run(
            [
                str(reader),
                "--binary",
                str(binary),
                "--coords",
                str(args.coords),
                "--output",
                str(nrpy_values_path),
            ],
            check=True,
        )
        reference_values = last_column(nrpy_values_path)

    prediction = last_column(args.nn_values)
    weights = last_column(args.volume_weights)
    metric = volume_weighted_relative_l2(prediction, reference_values, weights)
    document: dict[str, object] = {
        "schema_version": 1,
        "metric": "volume_weighted_relative_l2",
        "value": metric,
        "sample_count": int(prediction.size),
        "nrpy_binary": str(binary),
        "nrpy_binary_sha256": reference.binary_sha256,
        "reference_config": str(reference.path),
        "solver_config": str(args.solver_config.resolve()),
        "checkpoint_values": str(args.nn_values.resolve()),
        "checkpoint_values_sha256": sha256_file(args.nn_values.resolve()),
        "coords": str(args.coords.resolve()),
        "volume_weights": str(args.volume_weights.resolve()),
        "reader": str(reader),
    }
    if args.equal_spin_sz is not None:
        document["equal_spin_sz"] = float(args.equal_spin_sz)
    atomic_json(output_path, document)
    print(f"volume-weighted relative L2 = {metric:.17e}")
    print(f"Wrote {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
