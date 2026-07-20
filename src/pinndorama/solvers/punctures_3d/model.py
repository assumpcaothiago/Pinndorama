"""Float64 native-coordinate MLP with the publication falloff ansatz."""

from __future__ import annotations

import math
from typing import Any, Sequence

from . import config, coordinates

Params = list[dict[str, Any]]


def init_mlp_params(key, layers: Sequence[int]) -> Params:
    """He-initialize dense layers in Equinox ``(out, in)`` storage order."""

    config.enable_jax_x64()
    import jax
    import jax.numpy as jnp

    layer_sizes = tuple(int(value) for value in layers)
    keys = jax.random.split(key, len(layer_sizes) - 1)
    params: Params = []
    for layer_key, in_dim, out_dim in zip(keys, layer_sizes[:-1], layer_sizes[1:]):
        std = math.sqrt(2.0 / in_dim)
        params.append(
            {
                "W": std
                * jax.random.normal(
                    layer_key,
                    (in_dim, out_dim),
                    dtype=jnp.float64,
                ).T,
                "b": jnp.zeros((out_dim,), dtype=jnp.float64),
            }
        )
    return params


def forward(params: Params, native_x, geometry: config.Geometry):
    """Evaluate ``raw_MLP(xx) / sqrt(1 + r(xx)^2)``."""

    config.enable_jax_x64()
    import jax.numpy as jnp

    native_x = jnp.asarray(native_x, dtype=jnp.float64)
    if native_x.ndim != 2 or native_x.shape[1] != 3:
        raise ValueError("native_x must have shape (N, 3)")
    hidden = native_x
    for layer in params[:-1]:
        hidden = jnp.tanh(hidden @ layer["W"].T + layer["b"])
    raw = hidden @ params[-1]["W"].T + params[-1]["b"]
    radius = coordinates.spherical_radius_from_xx(
        native_x[:, 0:1],
        native_x[:, 1:2],
        geometry,
        xp=jnp,
    )
    return raw / jnp.sqrt(1.0 + radius * radius)


def parameter_leaves(params: Params) -> list[Any]:
    """Return the stable checkpoint order ``W0,b0,W1,b1,...``."""

    return [leaf for layer in params for leaf in (layer["W"], layer["b"])]


def params_from_leaves(leaves: Sequence[Any], layers: Sequence[int]) -> Params:
    """Validate and rebuild model parameters from alternating checkpoint leaves."""

    config.enable_jax_x64()
    import jax.numpy as jnp
    import numpy as np

    layer_sizes = tuple(int(value) for value in layers)
    expected_count = 2 * (len(layer_sizes) - 1)
    if len(leaves) != expected_count:
        raise ValueError(
            f"checkpoint has {len(leaves)} leaves; expected {expected_count}"
        )
    params: Params = []
    for index, (in_dim, out_dim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        weights_np = np.asarray(leaves[2 * index])
        bias_np = np.asarray(leaves[2 * index + 1])
        if weights_np.dtype != np.float64 or bias_np.dtype != np.float64:
            raise ValueError("all checkpoint parameter leaves must have dtype float64")
        if weights_np.shape != (out_dim, in_dim):
            raise ValueError(
                f"leaf_{2 * index:03d} has shape {weights_np.shape}; "
                f"expected {(out_dim, in_dim)}"
            )
        if bias_np.shape != (out_dim,):
            raise ValueError(
                f"leaf_{2 * index + 1:03d} has shape {bias_np.shape}; "
                f"expected {(out_dim,)}"
            )
        params.append(
            {
                "W": jnp.asarray(weights_np, dtype=jnp.float64),
                "b": jnp.asarray(bias_np, dtype=jnp.float64),
            }
        )
    return params
