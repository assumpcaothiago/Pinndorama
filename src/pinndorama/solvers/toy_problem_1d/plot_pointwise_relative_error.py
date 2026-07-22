"""Plot pointwise toy-solution errors for selected training resolutions."""

from __future__ import annotations

import argparse
import csv
from dataclasses import dataclass
import json
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

from . import checkpoint, coordinates, equation, model, relative_l2


@dataclass(frozen=True)
class SelectedCheckpoint:
    """A final checkpoint selected by training resolution and seed."""

    path: Path
    params: list[dict[str, np.ndarray]]
    metadata: dict[str, Any]
    training_Nxx0: int
    seed: int
    context: dict[str, Any]


def parse_resolutions(value: str) -> list[int]:
    """Parse a comma-separated list of distinct positive resolutions."""

    pieces = [piece.strip() for piece in value.split(",")]
    if not pieces or any(not piece for piece in pieces):
        raise ValueError("resolutions must be a comma-separated list of integers")
    try:
        resolutions = [int(piece) for piece in pieces]
    except ValueError as error:
        raise ValueError("resolutions must be a comma-separated list of integers") from error
    if any(value < 1 for value in resolutions):
        raise ValueError("resolutions must be positive")
    if len(set(resolutions)) != len(resolutions):
        raise ValueError("resolutions must not contain duplicates")
    return sorted(resolutions)


def select_checkpoints(
    checkpoint_paths: Sequence[str | Path],
    *,
    resolutions: Sequence[int],
    seed: int,
) -> list[SelectedCheckpoint]:
    """Select exactly one checkpoint for every requested resolution."""

    requested = set(resolutions)
    selected: dict[int, SelectedCheckpoint] = {}
    for raw_path in checkpoint_paths:
        path = Path(raw_path).expanduser().resolve()
        params, metadata = checkpoint.load_checkpoint(path)
        training_Nxx0 = relative_l2._metadata_integer(
            metadata, "collocation", "Nxx0"
        )
        checkpoint_seed = relative_l2._metadata_integer(metadata, "training", "seed")
        if training_Nxx0 not in requested or checkpoint_seed != seed:
            continue
        if training_Nxx0 in selected:
            raise ValueError(
                f"multiple checkpoints match Nxx0={training_Nxx0}, seed={seed}: "
                f"{selected[training_Nxx0].path} and {path}"
            )
        selected[training_Nxx0] = SelectedCheckpoint(
            path=path,
            params=params,
            metadata=metadata,
            training_Nxx0=training_Nxx0,
            seed=checkpoint_seed,
            context=relative_l2._evaluation_context(metadata),
        )

    missing = sorted(requested.difference(selected))
    if missing:
        values = ", ".join(str(value) for value in missing)
        raise ValueError(f"no checkpoint with seed={seed} was found for Nxx0={values}")

    ordered = [selected[value] for value in sorted(requested)]
    common_context = ordered[0].context
    for item in ordered[1:]:
        if item.context != common_context:
            raise checkpoint.CheckpointError(
                f"checkpoint {item.path} does not share the common evaluation context"
            )
    return ordered


def evaluate_selected(
    selected: Sequence[SelectedCheckpoint], *, evaluation_Nxx0: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray, dict[int, np.ndarray], dict[int, np.ndarray]]:
    """Evaluate selected networks and pointwise relative errors on one grid."""

    if not selected:
        raise ValueError("at least one selected checkpoint is required")
    if isinstance(evaluation_Nxx0, bool) or evaluation_Nxx0 < 1:
        raise ValueError("evaluation_Nxx0 must be a positive integer")
    context = selected[0].context
    xx0 = np.asarray(
        coordinates.cell_centered_xx0(
            Nxx0=evaluation_Nxx0,
            xx0_min=context["xx0_min"],
            xx0_max=context["xx0_max"],
            xp=np,
        ),
        dtype=np.float64,
    )
    radius = np.asarray(
        coordinates.radius_from_xx0(
            xx0, ampl=context["AMPL"], sinhw=context["SINHW"], xp=np
        ),
        dtype=np.float64,
    )
    exact = np.asarray(equation.exact_solution(radius, context["m"], xp=np))
    if np.any(exact == 0.0) or not np.all(np.isfinite(exact)):
        raise ValueError("exact solution must be finite and nonzero on the evaluation grid")

    values_by_resolution: dict[int, np.ndarray] = {}
    errors_by_resolution: dict[int, np.ndarray] = {}
    for item in selected:
        values = np.asarray(
            model.forward_numpy(
                item.params,
                xx0[:, None],
                ampl=context["AMPL"],
                sinhw=context["SINHW"],
            )
        ).reshape(-1)
        relative_error = np.abs(values - exact) / np.abs(exact)
        if not np.all(np.isfinite(values)) or not np.all(np.isfinite(relative_error)):
            raise ValueError(f"nonfinite evaluation encountered for {item.path}")
        values_by_resolution[item.training_Nxx0] = values
        errors_by_resolution[item.training_Nxx0] = relative_error
    return xx0, radius, exact, values_by_resolution, errors_by_resolution


def write_table(
    output_path: str | Path,
    *,
    xx0: np.ndarray,
    radius: np.ndarray,
    exact: np.ndarray,
    values_by_resolution: Mapping[int, np.ndarray],
    errors_by_resolution: Mapping[int, np.ndarray],
) -> Path:
    """Write plot-ready values and errors in a wide TSV table."""

    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    resolutions = sorted(values_by_resolution)
    fields = ["xx0", "r", "u_exact"]
    for resolution in resolutions:
        fields.extend((f"u_nn_N{resolution:03d}", f"relative_error_N{resolution:03d}"))
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.writer(stream, delimiter="\t", lineterminator="\n")
        writer.writerow(fields)
        for index in range(xx0.size):
            row = [xx0[index], radius[index], exact[index]]
            for resolution in resolutions:
                row.extend(
                    (
                        values_by_resolution[resolution][index],
                        errors_by_resolution[resolution][index],
                    )
                )
            writer.writerow(f"{float(value):.16e}" for value in row)
    return output


def plot_errors(
    output_prefix: str | Path,
    *,
    radius: np.ndarray,
    errors_by_resolution: Mapping[int, np.ndarray],
    r_min: float | None = None,
    r_max: float | None = None,
    dpi: int = 240,
) -> tuple[Path, Path]:
    """Create publication-style PNG and PDF log-log error plots."""

    if dpi < 1:
        raise ValueError("dpi must be positive")
    if r_min is not None and (not np.isfinite(r_min) or r_min <= 0.0):
        raise ValueError("r_min must be positive and finite")
    if r_max is not None and (not np.isfinite(r_max) or r_max <= 0.0):
        raise ValueError("r_max must be positive and finite")
    if r_min is not None and r_max is not None and r_min >= r_max:
        raise ValueError("r_min must be smaller than r_max")

    figure, axis = plt.subplots(figsize=(6.6, 4.8))
    plotted = 0
    for resolution in sorted(errors_by_resolution):
        relative_error = errors_by_resolution[resolution]
        mask = np.isfinite(radius) & np.isfinite(relative_error)
        mask &= (radius > 0.0) & (relative_error > 0.0)
        if r_min is not None:
            mask &= radius >= r_min
        if r_max is not None:
            mask &= radius <= r_max
        if not np.any(mask):
            raise ValueError(f"N={resolution} has no positive points in the plot range")
        axis.loglog(
            radius[mask],
            relative_error[mask],
            linewidth=1.35,
            label=rf"$N={resolution}$",
        )
        plotted += 1
    if plotted == 0:
        raise ValueError("no curves were available to plot")

    axis.set_xlabel(r"$r$")
    axis.set_ylabel(
        r"Pointwise relative error "
        r"$\left|u_{\mathrm{NN}}-u_{\mathrm{exact}}\right|/\left|u_{\mathrm{exact}}\right|$"
    )
    axis.grid(True, which="both", alpha=0.25, linewidth=0.6)
    axis.legend(frameon=False, ncol=2)
    figure.tight_layout()

    prefix = Path(output_prefix).expanduser()
    if prefix.suffix:
        raise ValueError("output_prefix must not have a file extension")
    prefix.parent.mkdir(parents=True, exist_ok=True)
    png = prefix.with_suffix(".png")
    pdf = prefix.with_suffix(".pdf")
    figure.savefig(png, dpi=dpi)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def write_metadata(
    output_path: str | Path,
    *,
    selected: Sequence[SelectedCheckpoint],
    evaluation_Nxx0: int,
    table_path: Path,
    png_path: Path,
    pdf_path: Path,
    r_min: float | None,
    r_max: float | None,
) -> Path:
    """Write checkpoint and evaluation provenance as JSON."""

    payload = {
        "metric": "pointwise_absolute_relative_error",
        "evaluation_grid": {
            "sampling": "cell_centered",
            "Nxx0": evaluation_Nxx0,
            "xx0_min": selected[0].context["xx0_min"],
            "xx0_max": selected[0].context["xx0_max"],
        },
        "plot_range": {"r_min": r_min, "r_max": r_max},
        "seed": selected[0].seed,
        "training_resolutions": [item.training_Nxx0 for item in selected],
        "checkpoints": [
            {
                "path": str(item.path),
                "sha256": checkpoint.sha256_file(item.path),
                "training_Nxx0": item.training_Nxx0,
                "seed": item.seed,
                "stage": str(item.metadata.get("stage", "")),
                "step": int(item.metadata.get("step", -1)),
            }
            for item in selected
        ],
        "outputs": {
            "table": str(table_path),
            "png": str(png_path),
            "pdf": str(pdf_path),
        },
    }
    output = Path(output_path).expanduser()
    output.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "inputs", nargs="+", help="Checkpoint files or directories searched recursively"
    )
    parser.add_argument(
        "--resolutions",
        required=True,
        help="Comma-separated training resolutions, for example 80,152,224,296,380",
    )
    parser.add_argument("--seed", type=int, default=503, help="Initialization seed")
    parser.add_argument(
        "--evaluation-Nxx0",
        type=int,
        default=760,
        help="Independent cell-centered evaluation resolution",
    )
    parser.add_argument("--output-prefix", required=True, help="Suffix-free output prefix")
    parser.add_argument("--r-min", type=float, help="Optional lower plotting radius")
    parser.add_argument("--r-max", type=float, help="Optional upper plotting radius")
    parser.add_argument("--dpi", type=int, default=240, help="PNG resolution")
    parser.add_argument(
        "--checkpoint-name",
        default="checkpoint_final.npz",
        help="File name discovered recursively inside directory inputs",
    )
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        resolutions = parse_resolutions(args.resolutions)
        print(
            f"Searching {len(args.inputs)} input path(s) for "
            f"{args.checkpoint_name!r} ...",
            flush=True,
        )
        paths = relative_l2.discover_checkpoints(
            args.inputs, checkpoint_name=args.checkpoint_name
        )
        print(f"Found {len(paths)} checkpoint(s).", flush=True)
        print(
            f"Selecting seed={args.seed} at Nxx0={','.join(map(str, resolutions))} ...",
            flush=True,
        )
        selected = select_checkpoints(paths, resolutions=resolutions, seed=args.seed)
        for item in selected:
            print(f"  Nxx0={item.training_Nxx0}: {item.path}", flush=True)
        print(
            f"Evaluating {len(selected)} checkpoint(s) on one independent "
            f"Nxx0={args.evaluation_Nxx0} cell-centered grid ...",
            flush=True,
        )
        xx0, radius, exact, values, errors = evaluate_selected(
            selected, evaluation_Nxx0=args.evaluation_Nxx0
        )
        prefix = Path(args.output_prefix).expanduser()
        table = write_table(
            Path(f"{prefix}.tsv"),
            xx0=xx0,
            radius=radius,
            exact=exact,
            values_by_resolution=values,
            errors_by_resolution=errors,
        )
        png, pdf = plot_errors(
            prefix,
            radius=radius,
            errors_by_resolution=errors,
            r_min=args.r_min,
            r_max=args.r_max,
            dpi=args.dpi,
        )
        metadata = write_metadata(
            Path(f"{prefix}.json"),
            selected=selected,
            evaluation_Nxx0=args.evaluation_Nxx0,
            table_path=table,
            png_path=png,
            pdf_path=pdf,
            r_min=args.r_min,
            r_max=args.r_max,
        )
    except (checkpoint.CheckpointError, OSError, ValueError) as error:
        parser.error(str(error))

    print("Done. Outputs:", flush=True)
    for output in (png, pdf, table, metadata):
        print(f"  {output}", flush=True)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
