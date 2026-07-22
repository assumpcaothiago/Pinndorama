"""Plot relative-L2 convergence across training resolutions and seeds."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

REQUIRED_FIELDS = {
    "training_Nxx0",
    "evaluation_Nxx0",
    "seed",
    "relative_l2_w1",
}
SUMMARY_FIELDS = (
    "training_Nxx0",
    "evaluation_Nxx0",
    "n_runs",
    "n_seeds",
    "mean",
    "median",
    "std",
    "min",
    "max",
    "q25",
    "q75",
)


def load_records(input_path: str | Path) -> list[dict[str, Any]]:
    """Load finite, positive relative-L2 records from an analyzer TSV."""

    path = Path(input_path).expanduser()
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("relative-L2 table must have a named header")
        missing = REQUIRED_FIELDS.difference(reader.fieldnames)
        if missing:
            names = ", ".join(sorted(missing))
            raise ValueError(f"relative-L2 table is missing required columns: {names}")

        records: list[dict[str, Any]] = []
        for row_number, row in enumerate(reader, start=2):
            try:
                record = {
                    "training_Nxx0": int(row["training_Nxx0"]),
                    "evaluation_Nxx0": int(row["evaluation_Nxx0"]),
                    "seed": int(row["seed"]),
                    "relative_l2_w1": float(row["relative_l2_w1"]),
                }
            except (TypeError, ValueError) as error:
                raise ValueError(
                    f"invalid convergence value on row {row_number}: {error}"
                ) from error
            if record["training_Nxx0"] < 1 or record["evaluation_Nxx0"] < 1:
                raise ValueError(f"resolutions must be positive on row {row_number}")
            error_value = record["relative_l2_w1"]
            if not np.isfinite(error_value) or error_value <= 0.0:
                raise ValueError(
                    f"relative_l2_w1 must be positive and finite on row {row_number}"
                )
            records.append(record)

    if not records:
        raise ValueError("relative-L2 table contains no data rows")
    evaluation_resolutions = {record["evaluation_Nxx0"] for record in records}
    uses_training_grids = all(
        record["evaluation_Nxx0"] == record["training_Nxx0"] for record in records
    )
    if len(evaluation_resolutions) != 1 and not uses_training_grids:
        values = ", ".join(str(value) for value in sorted(evaluation_resolutions))
        raise ValueError(
            "records must use one independent evaluation resolution or each "
            f"checkpoint's native training grid; found evaluation values {values}"
        )
    return records


def summarize_records(records: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    """Aggregate relative-L2 statistics at each training resolution."""

    if not records:
        raise ValueError("at least one relative-L2 record is required")
    grouped: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        grouped[int(record["training_Nxx0"])].append(record)

    summary: list[dict[str, Any]] = []
    for training_Nxx0 in sorted(grouped):
        group = grouped[training_Nxx0]
        values = np.asarray(
            [float(record["relative_l2_w1"]) for record in group],
            dtype=np.float64,
        )
        evaluation_resolutions = {int(record["evaluation_Nxx0"]) for record in group}
        if len(evaluation_resolutions) != 1:
            raise ValueError("each resolution group must use one evaluation grid")
        summary.append(
            {
                "training_Nxx0": training_Nxx0,
                "evaluation_Nxx0": evaluation_resolutions.pop(),
                "n_runs": len(group),
                "n_seeds": len({int(record["seed"]) for record in group}),
                "mean": float(np.mean(values)),
                "median": float(np.median(values)),
                "std": float(np.std(values)),
                "min": float(np.min(values)),
                "max": float(np.max(values)),
                "q25": float(np.quantile(values, 0.25)),
                "q75": float(np.quantile(values, 0.75)),
            }
        )
    return summary


def write_summary(
    summary: Sequence[Mapping[str, Any]], output_path: str | Path
) -> Path:
    """Write one tab-separated statistics row per training resolution."""

    if not summary:
        raise ValueError("at least one summary row is required")
    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    statistic_fields = {"mean", "median", "std", "min", "max", "q25", "q75"}
    with output.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=SUMMARY_FIELDS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for record in summary:
            row = dict(record)
            for field in statistic_fields:
                row[field] = f"{float(row[field]):.16e}"
            writer.writerow(row)
    return output


def plot_convergence(
    records: Sequence[Mapping[str, Any]],
    summary: Sequence[Mapping[str, Any]],
    output_path: str | Path,
    *,
    dpi: int = 300,
) -> Path:
    """Plot raw run values, seed curves, and aggregate convergence statistics."""

    if dpi <= 0:
        raise ValueError("dpi must be positive")
    if not records or not summary:
        raise ValueError("records and summary must not be empty")

    output = Path(output_path).expanduser()
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.6, 4.8))

    by_seed: dict[int, list[Mapping[str, Any]]] = defaultdict(list)
    for record in records:
        by_seed[int(record["seed"])].append(record)
    colors = plt.get_cmap("tab20")(np.linspace(0.0, 1.0, max(len(by_seed), 2)))
    individual_label_used = False
    for color, seed in zip(colors, sorted(by_seed)):
        seed_records = by_seed[seed]
        axis.scatter(
            [int(record["training_Nxx0"]) for record in seed_records],
            [float(record["relative_l2_w1"]) for record in seed_records],
            s=15,
            color=color,
            alpha=0.48,
            linewidths=0.0,
            zorder=2,
            label="Individual seeds" if not individual_label_used else None,
        )
        individual_label_used = True

        seed_by_resolution: dict[int, list[float]] = defaultdict(list)
        for record in seed_records:
            seed_by_resolution[int(record["training_Nxx0"])].append(
                float(record["relative_l2_w1"])
            )
        seed_x = np.asarray(sorted(seed_by_resolution), dtype=np.int64)
        seed_y = np.asarray(
            [np.mean(seed_by_resolution[value]) for value in seed_x],
            dtype=np.float64,
        )
        axis.plot(seed_x, seed_y, color=color, alpha=0.26, linewidth=0.7, zorder=1)

    training_Nxx0 = np.asarray(
        [int(record["training_Nxx0"]) for record in summary], dtype=np.int64
    )
    mean = np.asarray([float(record["mean"]) for record in summary])
    median = np.asarray([float(record["median"]) for record in summary])
    q25 = np.asarray([float(record["q25"]) for record in summary])
    q75 = np.asarray([float(record["q75"]) for record in summary])
    axis.fill_between(
        training_Nxx0,
        q25,
        q75,
        color="0.45",
        alpha=0.16,
        linewidth=0.0,
        label="Interquartile range",
        zorder=0,
    )
    axis.plot(
        training_Nxx0,
        mean,
        color="black",
        linewidth=2.0,
        marker="o",
        markersize=3.5,
        label="Mean",
        zorder=4,
    )
    axis.plot(
        training_Nxx0,
        median,
        color="tab:red",
        linewidth=1.6,
        linestyle="--",
        label="Median",
        zorder=4,
    )

    axis.set_yscale("log")
    axis.set_xlabel(r"Training resolution $N_{\mathrm{xx0}}$")
    axis.set_ylabel(r"Relative $L^2$-norm, $w_i=1$")
    axis.margins(x=0.02)
    axis.grid(True, which="major", alpha=0.28)
    axis.grid(True, which="minor", axis="y", alpha=0.12)
    axis.legend(frameon=False, fontsize=9)
    figure.tight_layout()
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="TSV produced by toy_problem_1d.relative_l2"
    )
    parser.add_argument(
        "--output-prefix",
        required=True,
        help="Output prefix for .png, .pdf, and _summary.tsv files",
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        records = load_records(args.input)
        summary = summarize_records(records)
        prefix = Path(args.output_prefix).expanduser()
        png = plot_convergence(
            records, summary, prefix.with_suffix(".png"), dpi=args.dpi
        )
        pdf = plot_convergence(
            records, summary, prefix.with_suffix(".pdf"), dpi=args.dpi
        )
        table = write_summary(summary, prefix.parent / f"{prefix.name}_summary.tsv")
    except (OSError, ValueError) as error:
        parser.error(str(error))

    seeds = {int(record["seed"]) for record in records}
    print(
        f"Loaded {len(records)} runs at {len(summary)} training resolutions "
        f"and {len(seeds)} seeds."
    )
    if all(
        record["evaluation_Nxx0"] == record["training_Nxx0"] for record in records
    ):
        print("Evaluation grids: each checkpoint's native training points")
    else:
        evaluation_Nxx0 = int(records[0]["evaluation_Nxx0"])
        print(f"Independent evaluation grid: Nxx0={evaluation_Nxx0}")
    print(f"PNG: {png}")
    print(f"PDF: {pdf}")
    print(f"Summary: {table}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
