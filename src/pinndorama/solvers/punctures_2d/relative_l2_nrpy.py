"""Compare 2D puncture checkpoints with one batched NRPy interpolation."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import math
import os
from pathlib import Path
import subprocess
import tempfile
from typing import Iterable, Mapping, Sequence

import numpy as np

from ..._paths import NRPYELLIPTIC_READER, NRPYELLIPTIC_ROOT
from ...reproducibility.reference import (
    assert_same_problem,
    load_reference_metadata,
)
from ...reproducibility.sampling import (
    cell_centers,
    sinhsymtp_radius,
    sinhsymtp_to_cartesian,
    sinhsymtp_volume_element,
)
from . import checkpoint, config, model


DEFAULT_BINARY = (
    NRPYELLIPTIC_ROOT
    / "nrpyelliptic_conformally_flat_symmetricID"
    / "NRPYELL_solution.bin"
)
DEFAULT_READER = NRPYELLIPTIC_READER
DEFAULT_CHECKPOINT_NAME = "checkpoint_final.npz"
DEFAULT_EVALUATION_POINTS = 760
DEFAULT_RADIUS = 7.5


@dataclass(frozen=True, order=True)
class GridKey:
    """Hashable description of one cell-centered native grid."""

    nxx0: int
    nxx1: int
    xx0_min: float
    xx0_max: float
    xx1_min: float
    xx1_max: float


@dataclass(frozen=True)
class EvaluationGrid:
    """Masked native, Cartesian, and quadrature data for one grid."""

    key: GridKey
    native: np.ndarray
    cartesian: np.ndarray
    weights: np.ndarray


@dataclass(frozen=True)
class CheckpointRecord:
    """A checkpoint and the validated metadata needed by this analysis."""

    path: Path
    params: list[dict[str, np.ndarray]]
    metadata: dict[str, object]
    native_grid_key: GridKey


def _positive_integer(value: str) -> int:
    parsed = int(value)
    if parsed <= 0:
        raise argparse.ArgumentTypeError("value must be a positive integer")
    return parsed


def _positive_float(value: str) -> float:
    parsed = float(value)
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise argparse.ArgumentTypeError("value must be finite and positive")
    return parsed


def default_reference_config(binary: str | Path) -> Path:
    """Return the conventional authenticated-reference path for ``binary``."""

    return Path(binary).with_suffix(".reference.toml")


def discover_checkpoints(
    inputs: Iterable[str | Path],
    *,
    checkpoint_name: str = DEFAULT_CHECKPOINT_NAME,
) -> list[Path]:
    """Discover named checkpoints recursively, accepting explicit files too."""

    discovered: set[Path] = set()
    for supplied in inputs:
        candidate = Path(supplied).expanduser().resolve()
        if candidate.is_file():
            if candidate.suffix != ".npz":
                raise ValueError(f"checkpoint file must end in .npz: {candidate}")
            discovered.add(candidate)
        elif candidate.is_dir():
            discovered.update(
                path.resolve() for path in candidate.rglob(checkpoint_name)
            )
        else:
            raise FileNotFoundError(f"checkpoint input does not exist: {candidate}")
    if not discovered:
        raise ValueError(
            f"no {checkpoint_name!r} files found beneath the supplied inputs"
        )
    return sorted(discovered, key=lambda path: str(path))


def _grid_key(collocation: Mapping[str, object]) -> GridKey:
    required = (
        "Nxx0",
        "Nxx1",
        "xx0_min",
        "xx0_max",
        "xx1_min",
        "xx1_max",
    )
    missing = [name for name in required if name not in collocation]
    if missing:
        raise ValueError(f"checkpoint collocation is missing {missing}")
    if collocation.get("sampling") != "cell_centered":
        raise ValueError("only cell-centered checkpoint collocation is supported")
    nxx0 = int(collocation["Nxx0"])
    nxx1 = int(collocation["Nxx1"])
    bounds = tuple(float(collocation[name]) for name in required[2:])
    if nxx0 <= 0 or nxx1 <= 0:
        raise ValueError("checkpoint collocation counts must be positive")
    if not all(math.isfinite(value) for value in bounds):
        raise ValueError("checkpoint collocation bounds must be finite")
    if bounds[0] >= bounds[1] or bounds[2] >= bounds[3]:
        raise ValueError("checkpoint collocation bounds must be increasing")
    return GridKey(nxx0, nxx1, *bounds)


def load_checkpoint_records(
    paths: Sequence[Path], solver_config: config.SolverConfig
) -> list[CheckpointRecord]:
    """Load checkpoints and require immutable compatibility with the config."""

    records: list[CheckpointRecord] = []
    for path in paths:
        params, metadata = checkpoint.load_checkpoint(path)
        checkpoint.validate_resume(metadata, solver_config)
        collocation = metadata.get("collocation")
        if not isinstance(collocation, Mapping):
            raise ValueError(f"checkpoint is missing collocation metadata: {path}")
        records.append(
            CheckpointRecord(
                path=path,
                params=params,
                metadata=metadata,
                native_grid_key=_grid_key(collocation),
            )
        )
    return records


def common_grid_key(
    solver_config: config.SolverConfig, nxx0: int, nxx1: int
) -> GridKey:
    """Build an independent grid key using the configured native domain."""

    collocation = solver_config.collocation
    return GridKey(
        int(nxx0),
        int(nxx1),
        float(collocation["xx0_min"]),
        float(collocation["xx0_max"]),
        float(collocation["xx1_min"]),
        float(collocation["xx1_max"]),
    )


def build_evaluation_grid(
    key: GridKey,
    geometry: Mapping[str, float],
    radius_limit: float,
) -> EvaluationGrid:
    """Build and radially mask one weighted, cell-centered native grid."""

    xx0_line = cell_centers(key.xx0_min, key.xx0_max, key.nxx0)
    xx1_line = cell_centers(key.xx1_min, key.xx1_max, key.nxx1)
    xx0, xx1 = np.meshgrid(xx0_line, xx1_line, indexing="ij")
    xx0 = xx0.reshape(-1)
    xx1 = xx1.reshape(-1)
    geometry_kwargs = {
        "amax": float(geometry["AMAX"]),
        "bscale": float(geometry["bScale"]),
        "sinhwaa": float(geometry["SINHWAA"]),
    }
    radius = sinhsymtp_radius(xx0, xx1, **geometry_kwargs)
    mask = radius < float(radius_limit)
    if not np.any(mask):
        raise ValueError(
            f"grid {key.nxx0}x{key.nxx1} has no points inside r<{radius_limit:g}"
        )
    xx0 = xx0[mask]
    xx1 = xx1[mask]
    spacing0 = (key.xx0_max - key.xx0_min) / key.nxx0
    spacing1 = (key.xx1_max - key.xx1_min) / key.nxx1
    weights = (
        sinhsymtp_volume_element(xx0, xx1, **geometry_kwargs)
        * spacing0
        * spacing1
    )
    x, y, z = sinhsymtp_to_cartesian(
        xx0, xx1, np.zeros_like(xx0), **geometry_kwargs
    )
    native = np.column_stack((xx0, xx1))
    cartesian = np.column_stack((x, y, z))
    if not (
        np.all(np.isfinite(native))
        and np.all(np.isfinite(cartesian))
        and np.all(np.isfinite(weights))
        and np.all(weights > 0.0)
    ):
        raise ValueError("evaluation grid contains invalid coordinates or weights")
    return EvaluationGrid(key, native, cartesian, weights)


def interpolate_reference_once(
    reader: Path,
    binary: Path,
    grids: Mapping[GridKey, EvaluationGrid],
    *,
    temporary_parent: Path,
) -> dict[GridKey, np.ndarray]:
    """Interpolate every unique grid with one reader subprocess invocation."""

    ordered = [(key, grids[key]) for key in sorted(grids)]
    all_cartesian = np.concatenate([grid.cartesian for _key, grid in ordered], axis=0)
    temporary_parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(
        prefix="pinndorama-nrpy-batch-", dir=temporary_parent
    ) as temporary:
        temporary_path = Path(temporary)
        coordinates_path = temporary_path / "cartesian_points.txt"
        output_path = temporary_path / "interpolated_uu.txt"
        np.savetxt(coordinates_path, all_cartesian, fmt="%.17e")
        subprocess.run(
            [
                str(reader),
                "--binary",
                str(binary),
                "--coords",
                str(coordinates_path),
                "--output",
                str(output_path),
            ],
            check=True,
        )
        table = np.loadtxt(output_path, comments="#", ndmin=2)
    if table.shape != (all_cartesian.shape[0], 4):
        raise ValueError(
            "reader output shape differs from the submitted coordinate batch: "
            f"expected {(all_cartesian.shape[0], 4)}, got {table.shape}"
        )
    if not np.all(np.isfinite(table)):
        raise ValueError("reader output contains non-finite values")
    if not np.allclose(
        table[:, :3], all_cartesian, rtol=0.0, atol=1.0e-12
    ):
        raise ValueError("reader output coordinates do not preserve input order")

    values: dict[GridKey, np.ndarray] = {}
    start = 0
    for key, grid in ordered:
        stop = start + grid.native.shape[0]
        values[key] = np.asarray(table[start:stop, 3], dtype=np.float64)
        start = stop
    return values


def _metric_components(
    prediction: np.ndarray, reference: np.ndarray, weights: np.ndarray
) -> tuple[float, float, float]:
    prediction = np.asarray(prediction, dtype=np.float64).reshape(-1)
    reference = np.asarray(reference, dtype=np.float64).reshape(-1)
    weights = np.asarray(weights, dtype=np.float64).reshape(-1)
    if not (prediction.shape == reference.shape == weights.shape):
        raise ValueError("prediction, reference, and weights must have equal sizes")
    if prediction.size == 0 or not all(
        np.all(np.isfinite(values)) for values in (prediction, reference, weights)
    ):
        raise ValueError("metric arrays must be nonempty and finite")
    if np.any(weights < 0.0) or not np.any(weights > 0.0):
        raise ValueError("metric weights must be nonnegative and not all zero")
    numerator = float(
        np.sum(weights * np.square(prediction - reference), dtype=np.float64)
    )
    denominator = float(
        np.sum(weights * np.square(reference), dtype=np.float64)
    )
    if denominator <= 0.0:
        raise ValueError("weighted reference norm is zero")
    return numerator, denominator, float(math.sqrt(numerator / denominator))


def write_tsv(path: Path, rows: Sequence[Mapping[str, object]]) -> None:
    """Atomically write the comparison summary."""

    if not rows:
        raise ValueError("cannot write an empty comparison table")
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            writer = csv.DictWriter(
                stream, fieldnames=list(rows[0]), delimiter="\t", lineterminator="\n"
            )
            writer.writeheader()
            writer.writerows(rows)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def analyze(
    records: Sequence[CheckpointRecord],
    solver_config: config.SolverConfig,
    *,
    grid_mode: str,
    common_nxx0: int,
    common_nxx1: int,
    radius_limit: float,
    reader: Path,
    binary: Path,
    reference_sha256: str,
    output: Path,
) -> list[dict[str, object]]:
    """Interpolate once, evaluate every checkpoint, and write comparison rows."""

    if grid_mode == "common":
        selected_key = common_grid_key(solver_config, common_nxx0, common_nxx1)
        keys = {selected_key}
    elif grid_mode == "native":
        selected_key = None
        keys = {record.native_grid_key for record in records}
    else:
        raise ValueError(f"unsupported grid mode: {grid_mode!r}")

    grids: dict[GridKey, EvaluationGrid] = {}
    for key in sorted(keys):
        grid = build_evaluation_grid(key, solver_config.geometry, radius_limit)
        grids[key] = grid
        print(
            f"Grid {key.nxx0}x{key.nxx1}: retained {grid.native.shape[0]} "
            f"of {key.nxx0 * key.nxx1} points inside r<{radius_limit:g}."
        )
    total_points = sum(grid.native.shape[0] for grid in grids.values())
    print(
        f"Running the 9-point NRPy reader once for {total_points} points "
        f"across {len(grids)} unique grid(s) ..."
    )
    reference_values = interpolate_reference_once(
        reader, binary, grids, temporary_parent=output.parent.resolve()
    )
    print("Reference interpolation complete; evaluating neural networks.")

    geometry = solver_config.geometry

    rows: list[dict[str, object]] = []
    total = len(records)
    for index, record in enumerate(records, start=1):
        key = selected_key if selected_key is not None else record.native_grid_key
        grid = grids[key]
        print(f"[{index}/{total}] {record.path}")
        prediction = np.asarray(
            model.forward_numpy(
                record.params,
                grid.native,
                amax=geometry["AMAX"],
                bscale=geometry["bScale"],
                sinhwaa=geometry["SINHWAA"],
            )
        ).reshape(-1)
        numerator, denominator, relative_l2 = _metric_components(
            prediction, reference_values[key], grid.weights
        )
        metadata = record.metadata
        training = metadata.get("training")
        if not isinstance(training, Mapping):
            raise ValueError(f"checkpoint is missing training metadata: {record.path}")
        row: dict[str, object] = {
            "checkpoint": str(record.path),
            "checkpoint_sha256": checkpoint.sha256_file(record.path),
            "config_sha256": str(metadata.get("config_sha256", "")),
            "nrpy_binary_sha256": reference_sha256,
            "seed": int(training["seed"]),
            "train_Nxx0": record.native_grid_key.nxx0,
            "train_Nxx1": record.native_grid_key.nxx1,
            "stage": str(metadata.get("stage", "")),
            "step": int(metadata.get("step", 0)),
            "grid_mode": grid_mode,
            "eval_Nxx0": key.nxx0,
            "eval_Nxx1": key.nxx1,
            "sample_count": int(grid.native.shape[0]),
            "radius_limit": f"{radius_limit:.17e}",
            "weighted_error_squared": f"{numerator:.17e}",
            "weighted_reference_squared": f"{denominator:.17e}",
            "relative_l2": f"{relative_l2:.17e}",
        }
        rows.append(row)
        print(f"    relative L2 = {relative_l2:.8e}")
    write_tsv(output, rows)
    return rows


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Checkpoint file or directory searched recursively for checkpoints.",
    )
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--nrpy-binary", type=Path, default=DEFAULT_BINARY)
    parser.add_argument(
        "--reference-config",
        type=Path,
        help="Authenticated reference TOML; defaults beside --nrpy-binary.",
    )
    parser.add_argument("--reader", type=Path, default=DEFAULT_READER)
    parser.add_argument(
        "--grid-mode", choices=("common", "native"), default="common"
    )
    parser.add_argument(
        "--Nxx0", type=_positive_integer, default=DEFAULT_EVALUATION_POINTS
    )
    parser.add_argument(
        "--Nxx1", type=_positive_integer, default=DEFAULT_EVALUATION_POINTS
    )
    parser.add_argument("--radius", type=_positive_float, default=DEFAULT_RADIUS)
    parser.add_argument("--checkpoint-name", default=DEFAULT_CHECKPOINT_NAME)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        solver_config = config.load_config(args.config)
        paths = discover_checkpoints(args.inputs, checkpoint_name=args.checkpoint_name)
        print(f"Found {len(paths)} checkpoint(s).")
        records = load_checkpoint_records(paths, solver_config)

        binary = args.nrpy_binary.expanduser().resolve()
        reference_path = (
            args.reference_config.expanduser().resolve()
            if args.reference_config is not None
            else default_reference_config(binary).resolve()
        )
        reference = load_reference_metadata(reference_path, binary)
        assert_same_problem(
            reference,
            {
                name: solver_config.geometry[name]
                for name in reference.geometry
            },
            solver_config.physics,
        )
        reader = args.reader.expanduser().resolve()
        if not reader.is_file():
            raise FileNotFoundError(
                f"canonical reader not found at {reader}; build it with "
                "`make -C reference_solvers/nrpyelliptic/"
                "READER_nrpyelliptic_conformally_flat`"
            )
        if not os.access(reader, os.X_OK):
            raise PermissionError(f"reader is not executable: {reader}")
        output = args.output.expanduser().resolve()
        analyze(
            records,
            solver_config,
            grid_mode=args.grid_mode,
            common_nxx0=args.Nxx0,
            common_nxx1=args.Nxx1,
            radius_limit=args.radius,
            reader=reader,
            binary=binary,
            reference_sha256=reference.binary_sha256,
            output=output,
        )
    except (
        config.ConfigError,
        checkpoint.CheckpointError,
        FileNotFoundError,
        OSError,
        subprocess.CalledProcessError,
        ValueError,
    ) as error:
        parser.error(str(error))
    print(f"Done: wrote {len(records)} comparison row(s) to {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
