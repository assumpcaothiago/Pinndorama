"""Publication parametric MLP in native SinhSymTP coordinates."""

from __future__ import annotations

import math

from . import config, coordinates


def _require_jax():
    config.enable_jax_x64()
    import jax
    import jax.numpy as jnp

    return jax, jnp


def _init_layer_params(key, in_dim: int, out_dim: int):
    """Initialize one dense layer."""

    jax, jnp = _require_jax()
    dtype = config.jax_dtype()
    init_type = config.initialization_type.lower()
    if init_type == "xavier":
        limit = math.sqrt(6.0 / (in_dim + out_dim))
        weights = jax.random.uniform(
            key,
            (out_dim, in_dim),
            minval=-limit,
            maxval=limit,
            dtype=dtype,
        )
    elif init_type == "he":
        std = math.sqrt(2.0 / in_dim)
        weights = std * jax.random.normal(key, (out_dim, in_dim), dtype=dtype)
    else:
        weights = jax.random.uniform(
            key,
            (out_dim, in_dim),
            minval=-0.1,
            maxval=0.1,
            dtype=dtype,
        )
    return {"W": weights, "b": jnp.zeros((out_dim,), dtype=dtype)}


def init_mlp_params(key, layers: list[int] | None = None):
    """Initialize MLP parameters for ``[xx0, xx1, equal_spin_sz]`` inputs.

    Dense weights use Equinox Linear's ``(out_features, in_features)`` layout.
    """

    jax, _jnp = _require_jax()
    if layers is None:
        layers = config.net_layers
    keys = jax.random.split(key, len(layers) - 1)
    return [
        _init_layer_params(layer_key, in_dim, out_dim)
        for layer_key, in_dim, out_dim in zip(keys, layers[:-1], layers[1:])
    ]


def forward(
    params,
    x,
    *,
    amax: float = config.AMAX,
    bscale: float = config.bScale,
    sinhwaa: float = config.SINHWAA,
):
    """Evaluate the MLP on native-coordinate inputs of shape ``(N, 3)``."""

    _jax, jnp = _require_jax()
    native_x = jnp.asarray(x, dtype=config.jax_dtype())
    xx0 = native_x[:, 0:1]
    xx1 = native_x[:, 1:2]

    h = native_x
    for layer in params[:-1]:
        h = jnp.tanh(h @ layer["W"].T + layer["b"])
    raw_out = h @ params[-1]["W"].T + params[-1]["b"]

    r_sph = coordinates.spherical_radius_from_xx(
        xx0,
        xx1,
        amax=amax,
        bscale=bscale,
        sinhwaa=sinhwaa,
        xp=jnp,
    )
    return raw_out / jnp.sqrt(1.0 + r_sph * r_sph)
