"""Plot the toy-problem relative error from an evaluation table."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
from matplotlib import pyplot as plt  # noqa: E402

REQUIRED_COLUMNS = {"r", "relative_error"}


def load_relative_error(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load positive, finite radius and relative-error values for log plotting."""
    table = np.genfromtxt(path, names=True, dtype=np.float64, encoding="utf-8")
    if table.dtype.names is None:
        raise ValueError("evaluation table must have a named header")
    missing = REQUIRED_COLUMNS.difference(table.dtype.names)
    if missing:
        names = ", ".join(sorted(missing))
        raise ValueError(f"evaluation table is missing required columns: {names}")

    radius = np.atleast_1d(np.asarray(table["r"], dtype=np.float64))
    relative_error = np.atleast_1d(
        np.asarray(table["relative_error"], dtype=np.float64)
    )
    valid = (
        np.isfinite(radius)
        & np.isfinite(relative_error)
        & (radius > 0.0)
        & (relative_error > 0.0)
    )
    if not np.any(valid):
        raise ValueError(
            "evaluation table has no positive, finite radius/error pairs to plot"
        )
    order = np.argsort(radius[valid])
    return radius[valid][order], relative_error[valid][order]


def plot_relative_error(
    input_path: str | Path,
    output_path: str | Path,
    *,
    dpi: int = 200,
) -> Path:
    """Create a log-log relative-error plot and return its output path."""
    if dpi <= 0:
        raise ValueError("dpi must be positive")
    radius, relative_error = load_relative_error(input_path)

    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    figure, axis = plt.subplots(figsize=(6.4, 4.8))
    axis.loglog(radius, relative_error, color="tab:blue", linewidth=1.5)
    axis.set_xlabel(r"$r$")
    axis.set_ylabel(r"$|u_{\mathrm{NN}}-u_{\mathrm{exact}}|/|u_{\mathrm{exact}}|$")
    axis.grid(True, which="both", alpha=0.3)
    figure.tight_layout()
    figure.savefig(output, dpi=dpi)
    plt.close(figure)
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--input", required=True, help="Toy evaluator's six-column text table"
    )
    parser.add_argument("--output", required=True, help="Output plot path")
    parser.add_argument("--dpi", type=int, default=200, help="Output resolution")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        output = plot_relative_error(args.input, args.output, dpi=args.dpi)
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
