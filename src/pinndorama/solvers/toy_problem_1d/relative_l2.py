"""Measure checkpoint errors on an independent cell-centered evaluation grid."""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from . import checkpoint, config, coordinates, equation, model

SUMMARY_FIELDS = (
    "checkpoint",
    "checkpoint_sha256",
    "config_name",
    "training_Nxx0",
    "evaluation_Nxx0",
    "xx0_min",
    "xx0_max",
    "seed",
    "stage",
    "step",
    "architecture",
    "initialization",
    "relative_l2_w1",
    "rms_absolute_error",
    "rms_exact",
    "max_absolute_error",
    "max_pointwise_relative_error",
)


def discover_checkpoints(
    inputs: Sequence[str | Path], *, checkpoint_name: str = "checkpoint_final.npz"
) -> list[Path]:
    """Resolve explicit checkpoint files and recursive directory searches."""

    if not checkpoint_name or Path(checkpoint_name).name != checkpoint_name:
        raise ValueError("checkpoint_name must be a file name, not a path")
    discovered: dict[Path, None] = {}
    for raw_path in inputs:
        path = Path(raw_path).expanduser()
        if path.is_file():
            if path.suffix != ".npz":
                raise ValueError(f"checkpoint file must end in .npz: {path}")
            discovered[path.resolve()] = None
        elif path.is_dir():
            for candidate in sorted(path.rglob(checkpoint_name)):
                if candidate.is_file():
                    discovered[candidate.resolve()] = None
        else:
            raise ValueError(f"checkpoint input does not exist: {path}")
    if not discovered:
        raise ValueError(
            f"no checkpoint files named {checkpoint_name!r} were discovered"
        )
    return sorted(discovered)


def _mapping(metadata: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = metadata.get(name)
    if not isinstance(value, Mapping):
        raise checkpoint.CheckpointError(f"checkpoint metadata is missing {name}")
    return value


def _evaluation_context(metadata: Mapping[str, Any]) -> dict[str, Any]:
    geometry = _mapping(metadata, "geometry")
    physics = _mapping(metadata, "physics")
    ansatz = _mapping(metadata, "ansatz")
    collocation = _mapping(metadata, "collocation")
    try:
        context = {
            "coordinate_system": str(geometry["coordinate_system"]),
            "AMPL": float(geometry["AMPL"]),
            "SINHW": float(geometry["SINHW"]),
            "m": float(physics["m"]),
            "output_transform": str(ansatz["output_transform"]),
            "xx0_min": float(collocation["xx0_min"]),
            "xx0_max": float(collocation["xx0_max"]),
        }
    except (KeyError, TypeError, ValueError) as error:
        raise checkpoint.CheckpointError(
            f"checkpoint metadata has an invalid evaluation context: {error}"
        ) from error
    numeric = np.asarray(
        [
            context["AMPL"],
            context["SINHW"],
            context["m"],
            context["xx0_min"],
            context["xx0_max"],
        ]
    )
    if not np.all(np.isfinite(numeric)):
        raise checkpoint.CheckpointError(
            "checkpoint evaluation context must contain finite numbers"
        )
    if context["coordinate_system"] != "SinhSpherical":
        raise checkpoint.CheckpointError(
            "checkpoint coordinate system must be 'SinhSpherical'"
        )
    if context["output_transform"] != config.OUTPUT_TRANSFORM:
        raise checkpoint.CheckpointError(
            f"checkpoint output transform must be {config.OUTPUT_TRANSFORM!r}"
        )
    if not context["xx0_min"] < context["xx0_max"]:
        raise checkpoint.CheckpointError(
            "checkpoint metadata requires xx0_min < xx0_max"
        )
    return context


def _metadata_integer(metadata: Mapping[str, Any], section: str, key: str) -> int:
    values = _mapping(metadata, section)
    value = values.get(key)
    if isinstance(value, bool) or not isinstance(value, int):
        raise checkpoint.CheckpointError(
            f"checkpoint metadata is missing integer {section}.{key}"
        )
    return value


def analyze_checkpoints(
    checkpoint_paths: Sequence[str | Path],
    *,
    evaluation_Nxx0: int | None = None,
    use_training_points: bool = False,
    show_progress: bool = False,
) -> list[dict[str, Any]]:
    """Return unweighted relative-L2 diagnostics for each checkpoint."""

    if use_training_points and evaluation_Nxx0 is not None:
        raise ValueError(
            "evaluation_Nxx0 and use_training_points are mutually exclusive"
        )
    if not use_training_points and evaluation_Nxx0 is None:
        raise ValueError("evaluation_Nxx0 is required for an independent grid")
    if evaluation_Nxx0 is not None and (
        isinstance(evaluation_Nxx0, bool) or evaluation_Nxx0 < 1
    ):
        raise ValueError("evaluation_Nxx0 must be a positive integer")
    records: list[dict[str, Any]] = []
    common_context: dict[str, Any] | None = None
    evaluation_grids: dict[int, tuple[np.ndarray, np.ndarray]] = {}
    total = len(checkpoint_paths)
    for index, raw_path in enumerate(checkpoint_paths, start=1):
        path = Path(raw_path).expanduser().resolve()
        if show_progress:
            print(
                f"[{index:>{len(str(total))}}/{total}] Analyzing {path} ... ",
                end="",
                flush=True,
            )
        params, metadata = checkpoint.load_checkpoint(path)
        context = _evaluation_context(metadata)
        if common_context is None:
            common_context = context
        elif context != common_context:
            raise checkpoint.CheckpointError(
                f"checkpoint {path} does not share the common evaluation context"
            )

        training_Nxx0 = _metadata_integer(metadata, "collocation", "Nxx0")
        current_evaluation_Nxx0 = (
            training_Nxx0 if use_training_points else evaluation_Nxx0
        )
        assert current_evaluation_Nxx0 is not None
        if current_evaluation_Nxx0 not in evaluation_grids:
            xx0 = np.asarray(
                coordinates.cell_centered_xx0(
                    Nxx0=current_evaluation_Nxx0,
                    xx0_min=context["xx0_min"],
                    xx0_max=context["xx0_max"],
                    xp=np,
                ),
                dtype=np.float64,
            )
            radius = coordinates.radius_from_xx0(
                xx0, ampl=context["AMPL"], sinhw=context["SINHW"], xp=np
            )
            exact = np.asarray(equation.exact_solution(radius, context["m"], xp=np))
            evaluation_grids[current_evaluation_Nxx0] = (xx0, exact)

        xx0, exact = evaluation_grids[current_evaluation_Nxx0]
        values = np.asarray(
            model.forward_numpy(
                params,
                xx0[:, None],
                ampl=context["AMPL"],
                sinhw=context["SINHW"],
            )
        ).reshape(-1)
        error = values - exact
        if not all(np.all(np.isfinite(array)) for array in (values, exact, error)):
            raise ValueError(f"nonfinite evaluation encountered for checkpoint {path}")
        exact_norm_sq = float(np.sum(exact * exact))
        if exact_norm_sq <= 0.0:
            raise ValueError("exact solution has zero discrete L2 norm")
        relative_l2 = float(np.sqrt(np.sum(error * error) / exact_norm_sq))
        absolute_error = np.abs(error)
        pointwise_relative = np.divide(
            absolute_error,
            np.abs(exact),
            out=np.full_like(absolute_error, np.inf),
            where=exact != 0.0,
        )

        architecture = _mapping(metadata, "architecture")
        layers = architecture.get("layers")
        if not isinstance(layers, list):
            raise checkpoint.CheckpointError(
                "checkpoint metadata is missing architecture.layers"
            )
        seed = _metadata_integer(metadata, "training", "seed")
        record = {
            "checkpoint": str(path),
            "checkpoint_sha256": checkpoint.sha256_file(path),
            "config_name": str(metadata.get("config_name", "")),
            "training_Nxx0": training_Nxx0,
            "evaluation_Nxx0": current_evaluation_Nxx0,
            "xx0_min": context["xx0_min"],
            "xx0_max": context["xx0_max"],
            "seed": seed,
            "stage": str(metadata.get("stage", "")),
            "step": int(metadata.get("step", -1)),
            "architecture": "x".join(str(value) for value in layers),
            "initialization": str(architecture.get("initialization", "")),
            "relative_l2_w1": relative_l2,
            "rms_absolute_error": float(np.sqrt(np.mean(error * error))),
            "rms_exact": float(np.sqrt(np.mean(exact * exact))),
            "max_absolute_error": float(np.max(absolute_error)),
            "max_pointwise_relative_error": float(np.max(pointwise_relative)),
        }
        records.append(record)
        if show_progress:
            print(
                f"Nxx0(train/eval)={training_Nxx0}/{current_evaluation_Nxx0}, "
                f"seed={seed}, "
                f"relative_L2={relative_l2:.6e}",
                flush=True,
            )
    return records


def write_summary(
    records: Sequence[Mapping[str, Any]], output_path: str | Path
) -> Path:
    """Write one tab-separated summary row per evaluated checkpoint."""

    if not records:
        raise ValueError("at least one checkpoint record is required")
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    float_fields = {
        "xx0_min",
        "xx0_max",
        "relative_l2_w1",
        "rms_absolute_error",
        "rms_exact",
        "max_absolute_error",
        "max_pointwise_relative_error",
    }
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=SUMMARY_FIELDS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for record in records:
            row = dict(record)
            for field in float_fields:
                row[field] = f"{float(row[field]):.16e}"
            writer.writerow(row)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs",
        nargs="+",
        help="Checkpoint files or directories searched recursively",
    )
    evaluation_grid = parser.add_mutually_exclusive_group(required=True)
    evaluation_grid.add_argument(
        "--Nxx0",
        dest="evaluation_Nxx0",
        type=int,
        help="Number of independent cell-centered evaluation points",
    )
    evaluation_grid.add_argument(
        "--use-training-points",
        action="store_true",
        help="Evaluate each checkpoint on its native cell-centered training grid",
    )
    parser.add_argument("--output", required=True, help="Output summary TSV")
    parser.add_argument(
        "--checkpoint-name",
        default="checkpoint_final.npz",
        help="File name discovered recursively inside directory inputs",
    )
    parser.add_argument(
        "--quiet",
        action="store_true",
        help="Suppress progress messages; errors are still reported",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        if not args.quiet:
            print(
                f"Searching {len(args.inputs)} input path(s) for "
                f"{args.checkpoint_name!r} ...",
                flush=True,
            )
        paths = discover_checkpoints(args.inputs, checkpoint_name=args.checkpoint_name)
        if not args.quiet:
            print(f"Found {len(paths)} checkpoint(s).", flush=True)
            if args.use_training_points:
                print(
                    "Evaluation grids: native cell-centered training points from "
                    "each checkpoint",
                    flush=True,
                )
            else:
                print(
                    "Evaluation grid: "
                    f"Nxx0={args.evaluation_Nxx0}, cell-centered, independent of "
                    "training",
                    flush=True,
                )
        records = analyze_checkpoints(
            paths,
            evaluation_Nxx0=args.evaluation_Nxx0,
            use_training_points=args.use_training_points,
            show_progress=not args.quiet,
        )
        if not args.quiet:
            print(
                f"Writing summary to {Path(args.output).expanduser()} ...", flush=True
            )
        output = write_summary(records, args.output)
    except (checkpoint.CheckpointError, OSError, ValueError) as error:
        parser.error(str(error))
    if not args.quiet:
        if args.use_training_points:
            completion = (
                f"Done: evaluated {len(records)} checkpoints on their native "
                "cell-centered training grids."
            )
        else:
            completion = (
                f"Done: evaluated {len(records)} checkpoints on an independent "
                f"Nxx0={args.evaluation_Nxx0} cell-centered grid."
            )
        print(completion, flush=True)
        print(f"Output: {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
