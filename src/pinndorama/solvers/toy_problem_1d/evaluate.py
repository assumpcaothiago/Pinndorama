"""Evaluate a 1D toy-problem checkpoint at explicit native coordinates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import checkpoint, config, coordinates, equation, model


def load_coordinates(path: str | Path) -> np.ndarray:
    coordinate_path = Path(path)
    if coordinate_path.suffix == ".npz":
        with np.load(coordinate_path, allow_pickle=False) as archive:
            if set(archive.files) != {"xx0"}:
                raise ValueError("coordinate NPZ must contain exactly xx0")
            xx0 = np.asarray(archive["xx0"], dtype=np.float64).reshape(-1)
    else:
        data = np.loadtxt(coordinate_path, dtype=np.float64, ndmin=2)
        if data.shape[1] != 1:
            raise ValueError("coordinate text file must contain exactly one column")
        xx0 = data[:, 0]
    if not np.all(np.isfinite(xx0)):
        raise ValueError("xx0 coordinates must be finite")
    if np.any(xx0 < 0.0) or np.any(xx0 > 1.0):
        raise ValueError("xx0 coordinates must lie in [0, 1]")
    return xx0


def evaluate_checkpoint(
    solver_config: config.SolverConfig,
    checkpoint_path: str | Path,
    coordinates_path: str | Path,
    output_path: str | Path,
) -> Path:
    config.enable_jax_x64()
    import jax.numpy as jnp

    params, metadata = checkpoint.load_checkpoint(checkpoint_path)
    checkpoint.validate_resume(metadata, solver_config)
    params = [
        {
            "W": jnp.asarray(layer["W"], dtype=jnp.float64),
            "b": jnp.asarray(layer["b"], dtype=jnp.float64),
        }
        for layer in params
    ]
    xx0 = load_coordinates(coordinates_path)
    radius = coordinates.radius_from_xx0(
        xx0,
        ampl=solver_config.geometry["AMPL"],
        sinhw=solver_config.geometry["SINHW"],
        xp=np,
    )
    values = np.asarray(
        model.forward(
            params,
            jnp.asarray(xx0[:, None], dtype=jnp.float64),
            ampl=solver_config.geometry["AMPL"],
            sinhw=solver_config.geometry["SINHW"],
        )
    ).reshape(-1)
    exact = equation.exact_solution(radius, solver_config.physics["m"], xp=np)
    absolute_error = np.abs(values - exact)
    relative_error = absolute_error / np.abs(exact)
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output,
        np.column_stack((xx0, radius, values, exact, absolute_error, relative_error)),
        header="xx0 r u_nn u_exact absolute_error relative_error",
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Reference TOML configuration")
    parser.add_argument("--checkpoint", required=True, help="Versioned NPZ checkpoint")
    parser.add_argument("--coords", required=True, help="One-column text or xx0 NPZ")
    parser.add_argument("--output", required=True, help="Output text table")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        solver_config = config.load_config(args.config)
        output = evaluate_checkpoint(
            solver_config, args.checkpoint, args.coords, args.output
        )
    except (
        config.ConfigError,
        checkpoint.CheckpointError,
        OSError,
        ValueError,
    ) as error:
        parser.error(str(error))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
