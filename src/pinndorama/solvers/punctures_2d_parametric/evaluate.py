"""Evaluate a parametric checkpoint at explicit native coordinates."""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

from . import checkpoint, config, coordinates


def load_coordinates(path: str | Path) -> tuple[np.ndarray, np.ndarray]:
    """Load native ``xx0, xx1`` from text or a named-array NPZ."""

    coordinate_path = Path(path)
    if coordinate_path.suffix == ".npz":
        with np.load(coordinate_path, allow_pickle=False) as archive:
            if set(archive.files) != {"xx0", "xx1"}:
                raise ValueError("coordinate NPZ must contain exactly xx0 and xx1")
            xx0 = np.asarray(archive["xx0"], dtype=np.float64).reshape(-1)
            xx1 = np.asarray(archive["xx1"], dtype=np.float64).reshape(-1)
    else:
        values = np.loadtxt(coordinate_path, dtype=np.float64, ndmin=2)
        if values.shape[1] != 2:
            raise ValueError("coordinate text file must contain exactly two columns")
        xx0, xx1 = values[:, 0], values[:, 1]
    if (
        xx0.shape != xx1.shape
        or not np.all(np.isfinite(xx0))
        or not np.all(np.isfinite(xx1))
    ):
        raise ValueError("coordinates must be finite arrays with matching shapes")
    return xx0, xx1


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoint", type=Path, required=True)
    parser.add_argument(
        "--equal-spin-sz",
        type=float,
        required=True,
        help="Raw Bowen-York S_z used for both S0_z and S1_z.",
    )
    parser.add_argument(
        "--coords",
        type=Path,
        required=True,
        help="Two-column native-coordinate text file or an xx0/xx1 NPZ.",
    )
    parser.add_argument("--output", type=Path, required=True)
    return parser.parse_args(argv)


def evaluate_checkpoint(
    cfg,
    checkpoint_path: str | Path,
    equal_spin_sz: float,
    coordinates_path: str | Path,
    output_path: str | Path,
) -> Path:
    """Write native, spherical, parameter, and model values for explicit points."""

    spin = cfg["parameter"]["equal_spin_sz"]
    if not spin["minimum"] <= equal_spin_sz <= spin["maximum"]:
        raise ValueError(
            f"equal_spin_sz={equal_spin_sz} is outside the checkpoint "
            f"range [{spin['minimum']}, {spin['maximum']}]"
        )

    params, metadata = checkpoint.load_checkpoint(checkpoint_path)
    checkpoint.validate_immutable_metadata(cfg, metadata)
    xx0, xx1 = load_coordinates(coordinates_path)
    if np.any((xx0 < 0.0) | (xx0 > 1.0)):
        raise ValueError("native xx0 coordinates must lie in [0, 1]")
    if np.any((xx1 < 0.0) | (xx1 > np.pi)):
        raise ValueError("native xx1 coordinates must lie in [0, pi]")

    config.apply_runtime_config(cfg)
    config.enable_jax_x64()
    import jax
    import jax.numpy as jnp
    from . import model

    jax_params = jax.tree_util.tree_map(
        lambda value: jnp.asarray(value, dtype=jnp.float64), params
    )
    inputs = np.column_stack(
        (xx0, xx1, np.full(xx0.shape[0], equal_spin_sz, dtype=np.float64))
    )
    geometry = config.geometry_parameter_dict(cfg)
    values = np.asarray(
        model.forward(
            jax_params,
            jnp.asarray(inputs, dtype=jnp.float64),
            amax=geometry["AMAX"],
            bscale=geometry["bScale"],
            sinhwaa=geometry["SINHWAA"],
        )
    ).reshape(-1)
    radius, theta = coordinates.xx_to_spherical(
        xx0,
        xx1,
        amax=geometry["AMAX"],
        bscale=geometry["bScale"],
        sinhwaa=geometry["SINHWAA"],
        xp=np,
    )
    table = np.column_stack((xx0, xx1, radius, theta, inputs[:, 2], values))
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(
        output,
        table,
        header="xx0 xx1 r_spherical theta_spherical equal_spin_sz u_nn",
        fmt="%.17e",
    )
    print(
        f"Wrote {table.shape[0]} values to {output} "
        f"(checkpoint sha256={checkpoint.file_sha256(checkpoint_path)})"
    )
    return output


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    cfg = config.load_config(args.config)
    evaluate_checkpoint(
        cfg,
        args.checkpoint,
        args.equal_spin_sz,
        args.coords,
        args.output,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
