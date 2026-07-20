"""Evaluate a 3D publication checkpoint at explicit native coordinates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import checkpoint, config


def load_coordinates(path: str | Path) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Load ``xx0, xx1, xx2`` from a three-column text file or named NPZ."""

    coordinate_path = Path(path)
    if coordinate_path.suffix.lower() == ".npz":
        with np.load(coordinate_path, allow_pickle=False) as archive:
            if set(archive.files) != {"xx0", "xx1", "xx2"}:
                raise ValueError(
                    "coordinate NPZ must contain exactly xx0, xx1, and xx2"
                )
            arrays = tuple(
                np.asarray(archive[name], dtype=np.float64).reshape(-1)
                for name in ("xx0", "xx1", "xx2")
            )
    else:
        values = np.loadtxt(coordinate_path, dtype=np.float64, ndmin=2)
        if values.shape[1] != 3:
            raise ValueError("coordinate text file must contain exactly three columns")
        arrays = values[:, 0], values[:, 1], values[:, 2]
    xx0, xx1, xx2 = arrays
    if xx0.size == 0 or not (xx0.shape == xx1.shape == xx2.shape):
        raise ValueError("coordinate arrays must be nonempty and have matching shapes")
    if not all(np.all(np.isfinite(array)) for array in arrays):
        raise ValueError("coordinates must be finite")
    return xx0, xx1, xx2


def _validate_domain(
    xx0: np.ndarray,
    xx1: np.ndarray,
    xx2: np.ndarray,
    collocation: config.Collocation,
) -> None:
    bounds = (
        (xx0, collocation.xx0_min, collocation.xx0_max, "xx0"),
        (xx1, collocation.xx1_min, collocation.xx1_max, "xx1"),
        (xx2, collocation.xx2_min, collocation.xx2_max, "xx2"),
    )
    for values, lower, upper, name in bounds:
        if np.any((values < lower) | (values > upper)):
            raise ValueError(
                f"native {name} coordinates must lie in [{lower}, {upper}]"
            )


def evaluate_checkpoint(
    run: config.RunConfig,
    checkpoint_path: str | Path,
    coordinates_path: str | Path,
    output_path: str | Path,
    *,
    batch_size: int = 65536,
) -> Path:
    """Write ``xx0 xx1 xx2 x y z u_nn`` for explicit native points."""

    if batch_size <= 0:
        raise ValueError("--batch-size must be positive")
    config.enable_jax_x64()
    import jax
    import jax.numpy as jnp
    from . import coordinates, model

    leaves, metadata, checkpoint_hash = checkpoint.load(checkpoint_path)
    checkpoint.validate_resume(run, metadata)
    params = model.params_from_leaves(leaves, run.architecture.layers)
    xx0, xx1, xx2 = load_coordinates(coordinates_path)
    _validate_domain(xx0, xx1, xx2, run.collocation)

    @jax.jit
    def batch_forward(native_points):
        return model.forward(params, native_points, run.geometry)[:, 0]

    batches = []
    for start in range(0, xx0.size, batch_size):
        stop = min(start + batch_size, xx0.size)
        native_points = jnp.asarray(
            np.column_stack((xx0[start:stop], xx1[start:stop], xx2[start:stop])),
            dtype=jnp.float64,
        )
        batches.append(np.asarray(jax.device_get(batch_forward(native_points))))
    solution = np.concatenate(batches).astype(np.float64, copy=False)
    x_cart, y_cart, z_cart = coordinates.xx_to_cart(xx0, xx1, xx2, run.geometry, xp=np)
    table = np.column_stack((xx0, xx1, xx2, x_cart, y_cart, z_cart, solution))
    output = Path(output_path).expanduser().resolve()
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output,
        table,
        header="xx0 xx1 xx2 x_cartesian y_cartesian z_cartesian u_nn",
        fmt="%.17e",
    )
    print(
        f"Wrote {table.shape[0]} values to {output} "
        f"(checkpoint sha256={checkpoint_hash})"
    )
    return output


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", required=True, help="Reference C04/C05 TOML")
    parser.add_argument(
        "--checkpoint", required=True, help="Publication NPZ checkpoint"
    )
    parser.add_argument(
        "--coords",
        required=True,
        help="Three-column native-coordinate text file or xx0/xx1/xx2 NPZ",
    )
    parser.add_argument("--output", required=True, help="Output text table")
    parser.add_argument("--batch-size", type=int, default=65536)
    return parser


def main(argv: list[str] | None = None) -> int:
    parser = build_parser()
    args = parser.parse_args(argv)
    try:
        run = config.load(args.config)
        output = evaluate_checkpoint(
            run,
            args.checkpoint,
            args.coords,
            args.output,
            batch_size=args.batch_size,
        )
    except (OSError, ValueError) as error:
        parser.error(str(error))
    print(output)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
