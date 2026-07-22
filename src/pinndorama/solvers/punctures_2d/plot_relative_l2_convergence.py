"""Plot fixed-grid relative-L2 convergence across 2D training resolutions."""

from __future__ import annotations

import argparse
import csv
from collections import defaultdict
from dataclasses import dataclass
import hashlib
import json
import math
import os
from pathlib import Path
import tempfile
from typing import Mapping, Sequence

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402


REQUIRED_FIELDS = {
    "checkpoint",
    "config_sha256",
    "nrpy_binary_sha256",
    "seed",
    "train_Nxx0",
    "train_Nxx1",
    "grid_mode",
    "eval_Nxx0",
    "eval_Nxx1",
    "sample_count",
    "radius_limit",
    "relative_l2",
}
SUMMARY_FIELDS = (
    "N",
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


@dataclass(frozen=True)
class ConvergenceRecord:
    """One validated checkpoint error from a common evaluation grid."""

    checkpoint: str
    config_sha256: str
    nrpy_binary_sha256: str
    seed: int
    training_n: int
    evaluation_nxx0: int
    evaluation_nxx1: int
    sample_count: int
    radius_limit: float
    relative_l2: float


@dataclass(frozen=True)
class DatasetMetadata:
    """Provenance shared by every record in one convergence dataset."""

    config_sha256: str
    nrpy_binary_sha256: str
    evaluation_nxx0: int
    evaluation_nxx1: int
    sample_count: int
    radius_limit: float
    seeds: tuple[int, ...]
    resolutions: tuple[int, ...]


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _integer(row: Mapping[str, str], field: str, row_number: int) -> int:
    try:
        value = int(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"row {row_number}: {field} must be an integer") from error
    if value <= 0:
        raise ValueError(f"row {row_number}: {field} must be positive")
    return value


def _seed(row: Mapping[str, str], row_number: int) -> int:
    try:
        value = int(row["seed"])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"row {row_number}: seed must be an integer") from error
    if value < 0:
        raise ValueError(f"row {row_number}: seed must be nonnegative")
    return value


def _positive_float(row: Mapping[str, str], field: str, row_number: int) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as error:
        raise ValueError(f"row {row_number}: {field} must be numeric") from error
    if not math.isfinite(value) or value <= 0.0:
        raise ValueError(f"row {row_number}: {field} must be positive and finite")
    return value


def _digest(row: Mapping[str, str], field: str, row_number: int) -> str:
    value = row.get(field, "").lower()
    if len(value) != 64 or any(
        character not in "0123456789abcdef" for character in value
    ):
        raise ValueError(f"row {row_number}: {field} must be a SHA-256 digest")
    return value


def _parse_record(row: Mapping[str, str], row_number: int) -> ConvergenceRecord:
    if row.get("grid_mode") != "common":
        raise ValueError(
            f"row {row_number}: convergence plots require grid_mode=common"
        )
    train_nxx0 = _integer(row, "train_Nxx0", row_number)
    train_nxx1 = _integer(row, "train_Nxx1", row_number)
    if train_nxx0 != train_nxx1:
        raise ValueError(
            f"row {row_number}: training grid must be square, got "
            f"{train_nxx0}x{train_nxx1}"
        )
    checkpoint = row.get("checkpoint", "")
    if not checkpoint:
        raise ValueError(f"row {row_number}: checkpoint must not be empty")
    return ConvergenceRecord(
        checkpoint=checkpoint,
        config_sha256=_digest(row, "config_sha256", row_number),
        nrpy_binary_sha256=_digest(row, "nrpy_binary_sha256", row_number),
        seed=_seed(row, row_number),
        training_n=train_nxx0,
        evaluation_nxx0=_integer(row, "eval_Nxx0", row_number),
        evaluation_nxx1=_integer(row, "eval_Nxx1", row_number),
        sample_count=_integer(row, "sample_count", row_number),
        radius_limit=_positive_float(row, "radius_limit", row_number),
        relative_l2=_positive_float(row, "relative_l2", row_number),
    )


def _shared_metadata(records: Sequence[ConvergenceRecord]) -> DatasetMetadata:
    if not records:
        raise ValueError("relative-L2 table contains no data rows")
    shared_fields = (
        "config_sha256",
        "nrpy_binary_sha256",
        "evaluation_nxx0",
        "evaluation_nxx1",
        "sample_count",
        "radius_limit",
    )
    for field in shared_fields:
        values = {getattr(record, field) for record in records}
        if len(values) != 1:
            raise ValueError(f"all rows must share one {field}")

    grouped_seeds: dict[int, set[int]] = defaultdict(set)
    pairs: set[tuple[int, int]] = set()
    for record in records:
        pair = (record.training_n, record.seed)
        if pair in pairs:
            raise ValueError(
                f"duplicate convergence row for N={record.training_n}, "
                f"seed={record.seed}"
            )
        pairs.add(pair)
        grouped_seeds[record.training_n].add(record.seed)
    expected_seeds = set(next(iter(grouped_seeds.values())))
    for training_n, seeds in sorted(grouped_seeds.items()):
        if seeds != expected_seeds:
            raise ValueError(
                f"N={training_n} has seeds {sorted(seeds)}, expected "
                f"{sorted(expected_seeds)}"
            )

    first = records[0]
    return DatasetMetadata(
        config_sha256=first.config_sha256,
        nrpy_binary_sha256=first.nrpy_binary_sha256,
        evaluation_nxx0=first.evaluation_nxx0,
        evaluation_nxx1=first.evaluation_nxx1,
        sample_count=first.sample_count,
        radius_limit=first.radius_limit,
        seeds=tuple(sorted(expected_seeds)),
        resolutions=tuple(sorted(grouped_seeds)),
    )


def load_records(
    input_path: str | Path,
) -> tuple[list[ConvergenceRecord], DatasetMetadata]:
    """Load and validate one coherent common-grid analyzer TSV."""

    path = Path(input_path).expanduser()
    with path.open("r", encoding="utf-8", newline="") as stream:
        reader = csv.DictReader(stream, delimiter="\t")
        if reader.fieldnames is None:
            raise ValueError("relative-L2 table must have a named header")
        missing = REQUIRED_FIELDS.difference(reader.fieldnames)
        if missing:
            raise ValueError(
                "relative-L2 table is missing required columns: "
                + ", ".join(sorted(missing))
            )
        records = [
            _parse_record(row, row_number)
            for row_number, row in enumerate(reader, start=2)
        ]
    records.sort(key=lambda record: (record.training_n, record.seed))
    return records, _shared_metadata(records)


def summarize_records(
    records: Sequence[ConvergenceRecord],
) -> list[dict[str, int | float]]:
    """Compute seed-distribution statistics at every training resolution."""

    if not records:
        raise ValueError("at least one convergence record is required")
    grouped: dict[int, list[ConvergenceRecord]] = defaultdict(list)
    for record in records:
        grouped[record.training_n].append(record)
    summary: list[dict[str, int | float]] = []
    for training_n in sorted(grouped):
        group = grouped[training_n]
        values = np.asarray(
            [record.relative_l2 for record in group], dtype=np.float64
        )
        summary.append(
            {
                "N": training_n,
                "n_runs": len(group),
                "n_seeds": len({record.seed for record in group}),
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


def _atomic_text(path: Path, write) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="") as stream:
            write(stream)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, path)
    except BaseException:
        try:
            os.unlink(temporary)
        except FileNotFoundError:
            pass
        raise


def write_summary(
    summary: Sequence[Mapping[str, int | float]], output_path: str | Path
) -> Path:
    """Write one statistics row per training resolution."""

    if not summary:
        raise ValueError("summary must not be empty")
    output = Path(output_path).expanduser()

    def write(stream) -> None:
        writer = csv.DictWriter(
            stream, fieldnames=SUMMARY_FIELDS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        for source in summary:
            row = dict(source)
            for field in ("mean", "median", "std", "min", "max", "q25", "q75"):
                row[field] = f"{float(row[field]):.17e}"
            writer.writerow(row)

    _atomic_text(output, write)
    return output


def write_metadata(
    input_path: str | Path,
    metadata: DatasetMetadata,
    row_count: int,
    output_path: str | Path,
) -> Path:
    """Write analysis provenance for the generated plot set."""

    input_file = Path(input_path).expanduser().resolve()
    output = Path(output_path).expanduser()
    document = {
        "schema_version": 1,
        "input_tsv": str(input_file),
        "input_tsv_sha256": sha256_file(input_file),
        "config_sha256": metadata.config_sha256,
        "nrpy_binary_sha256": metadata.nrpy_binary_sha256,
        "grid_mode": "common",
        "evaluation_grid": [metadata.evaluation_nxx0, metadata.evaluation_nxx1],
        "sample_count": metadata.sample_count,
        "radius_limit": metadata.radius_limit,
        "seeds": list(metadata.seeds),
        "resolutions": list(metadata.resolutions),
        "row_count": int(row_count),
    }

    def write(stream) -> None:
        json.dump(document, stream, indent=2, sort_keys=True, allow_nan=False)
        stream.write("\n")

    _atomic_text(output, write)
    return output


def _style_axis(axis, radius_limit: float) -> None:
    axis.set_yscale("log")
    axis.set_xlabel(r"Training resolution $N$")
    axis.set_ylabel(
        rf"Volume-weighted relative $L^2$ error ($r<{radius_limit:g}M$)"
    )
    axis.margins(x=0.02)
    axis.grid(True, which="major", alpha=0.28)
    axis.grid(True, which="minor", axis="y", alpha=0.12)


def plot_seed_curves(
    records: Sequence[ConvergenceRecord],
    metadata: DatasetMetadata,
    png_path: str | Path,
    pdf_path: str | Path,
    *,
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Plot one explicitly labeled curve per initialization seed."""

    if dpi <= 0 or not records:
        raise ValueError("seed plot requires records and a positive DPI")
    png = Path(png_path).expanduser()
    pdf = Path(pdf_path).expanduser()
    png.parent.mkdir(parents=True, exist_ok=True)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.6, 4.8))
    colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, len(metadata.seeds)))
    for color, seed in zip(colors, metadata.seeds):
        selected = [record for record in records if record.seed == seed]
        axis.plot(
            [record.training_n for record in selected],
            [record.relative_l2 for record in selected],
            color=color,
            linewidth=1.25,
            marker="o",
            markersize=3.5,
            label=f"Seed {seed}",
        )
    _style_axis(axis, metadata.radius_limit)
    axis.legend(frameon=False, fontsize=9, ncol=2)
    figure.tight_layout()
    figure.savefig(png, dpi=dpi)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def plot_aggregate(
    records: Sequence[ConvergenceRecord],
    summary: Sequence[Mapping[str, int | float]],
    metadata: DatasetMetadata,
    png_path: str | Path,
    pdf_path: str | Path,
    *,
    dpi: int = 300,
) -> tuple[Path, Path]:
    """Plot individual runs together with median and interquartile range."""

    if dpi <= 0 or not records or not summary:
        raise ValueError("aggregate plot requires data and a positive DPI")
    png = Path(png_path).expanduser()
    pdf = Path(pdf_path).expanduser()
    png.parent.mkdir(parents=True, exist_ok=True)
    pdf.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.6, 4.8))
    axis.scatter(
        [record.training_n for record in records],
        [record.relative_l2 for record in records],
        color="0.45",
        s=18,
        alpha=0.5,
        linewidths=0.0,
        label="Individual runs",
        zorder=2,
    )
    resolutions = np.asarray([int(row["N"]) for row in summary])
    median = np.asarray([float(row["median"]) for row in summary])
    q25 = np.asarray([float(row["q25"]) for row in summary])
    q75 = np.asarray([float(row["q75"]) for row in summary])
    axis.fill_between(
        resolutions,
        q25,
        q75,
        color="tab:blue",
        alpha=0.18,
        linewidth=0.0,
        label="Interquartile range",
        zorder=1,
    )
    axis.plot(
        resolutions,
        median,
        color="black",
        linewidth=1.8,
        marker="o",
        markersize=3.5,
        label="Median",
        zorder=3,
    )
    _style_axis(axis, metadata.radius_limit)
    axis.legend(frameon=False, fontsize=9)
    figure.tight_layout()
    figure.savefig(png, dpi=dpi)
    figure.savefig(pdf)
    plt.close(figure)
    return png, pdf


def output_paths(prefix: str | Path) -> dict[str, Path]:
    """Return stable output names derived from one user-supplied prefix."""

    base = Path(prefix).expanduser()
    return {
        "seed_png": base.parent / f"{base.name}_seed_curves.png",
        "seed_pdf": base.parent / f"{base.name}_seed_curves.pdf",
        "aggregate_png": base.parent / f"{base.name}_aggregate.png",
        "aggregate_pdf": base.parent / f"{base.name}_aggregate.pdf",
        "summary": base.parent / f"{base.name}_summary.tsv",
        "metadata": base.parent / f"{base.name}_metadata.json",
    }


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="Common-grid TSV from relative_l2_nrpy"
    )
    parser.add_argument(
        "--output-prefix", required=True, help="Prefix for plots and data tables"
    )
    parser.add_argument("--dpi", type=int, default=300, help="PNG resolution")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    if args.dpi <= 0:
        parser.error("--dpi must be positive")
    try:
        records, metadata = load_records(args.input)
        summary = summarize_records(records)
        paths = output_paths(args.output_prefix)
        plot_seed_curves(
            records,
            metadata,
            paths["seed_png"],
            paths["seed_pdf"],
            dpi=args.dpi,
        )
        plot_aggregate(
            records,
            summary,
            metadata,
            paths["aggregate_png"],
            paths["aggregate_pdf"],
            dpi=args.dpi,
        )
        write_summary(summary, paths["summary"])
        write_metadata(args.input, metadata, len(records), paths["metadata"])
    except (OSError, ValueError) as error:
        parser.error(str(error))

    print(
        f"Loaded {len(records)} runs at {len(metadata.resolutions)} resolutions "
        f"and {len(metadata.seeds)} seeds."
    )
    print(
        f"Fixed evaluation grid: {metadata.evaluation_nxx0}x"
        f"{metadata.evaluation_nxx1}, r<{metadata.radius_limit:g}M"
    )
    for name in (
        "seed_png",
        "seed_pdf",
        "aggregate_png",
        "aggregate_pdf",
        "summary",
        "metadata",
    ):
        print(f"{name}: {paths[name]}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
