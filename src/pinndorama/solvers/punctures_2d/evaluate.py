"""Evaluate a fixed 2D publication checkpoint at explicit native coordinates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import checkpoint, config, coordinates, model


def load_coordinates(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load ``xx0, xx1`` from text or an NPZ with matching named arrays."""

    coordinate_path = Path(path)
    if coordinate_path.suffix == ".npz":
        with np.load(coordinate_path, allow_pickle=False) as archive:
            if set(archive.files) != {"xx0", "xx1"}:
                raise ValueError("coordinate NPZ must contain exactly xx0 and xx1")
            xx0 = np.asarray(archive["xx0"], dtype=np.float64).reshape(-1)
            xx1 = np.asarray(archive["xx1"], dtype=np.float64).reshape(-1)
    else:
        data = np.loadtxt(coordinate_path, dtype=np.float64, ndmin=2)
        if data.shape[1] != 2:
            raise ValueError("coordinate text file must contain exactly two columns")
        xx0, xx1 = data[:, 0], data[:, 1]
    if (
        xx0.shape != xx1.shape
        or not np.all(np.isfinite(xx0))
        or not np.all(np.isfinite(xx1))
    ):
        raise ValueError("coordinates must be finite arrays with matching shapes")
    return xx0, xx1


def evaluate_checkpoint(
    solver_config: config.SolverConfig,
    checkpoint_path: str | Path,
    coordinates_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Write ``xx0 xx1 r theta u`` for an explicit checkpoint and point set."""

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
    xx0, xx1 = load_coordinates(coordinates_path)
    native = jnp.asarray(np.column_stack((xx0, xx1)), dtype=jnp.float64)
    values = np.asarray(
        model.forward(
            params,
            native,
            amax=solver_config.geometry["AMAX"],
            bscale=solver_config.geometry["bScale"],
            sinhwaa=solver_config.geometry["SINHWAA"],
        )
    ).reshape(-1)
    radius, theta = coordinates.xx_to_spherical(
        xx0,
        xx1,
        amax=solver_config.geometry["AMAX"],
        bscale=solver_config.geometry["bScale"],
        sinhwaa=solver_config.geometry["SINHWAA"],
        xp=np,
    )
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output,
        np.column_stack((xx0, xx1, radius, theta, values)),
        header="xx0 xx1 r_spherical theta_spherical u_nn",
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Reference TOML configuration")
    parser.add_argument(
        "--checkpoint", required=True, help="Publication NPZ checkpoint"
    )
    parser.add_argument(
        "--coords", required=True, help="Two-column text or xx0/xx1 NPZ"
    )
    parser.add_argument("--output", required=True, help="Output text table")
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        solver_config = config.load_config(args.config)
        output = evaluate_checkpoint(
            solver_config,
            args.checkpoint,
            args.coords,
            args.output,
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
