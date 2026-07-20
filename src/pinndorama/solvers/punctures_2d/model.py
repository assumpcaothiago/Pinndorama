"""Native-coordinate JAX MLP for the SinhSymTP experiment."""

from __future__ import annotations

import math

from . import config, coordinates


def _require_jax():
    config.enable_jax_x64()
    import jax
    import jax.numpy as jnp

    return jax, jnp


def _init_layer_params(key, in_dim: int, out_dim: int, initialization: str):
    """Initialize one dense layer in Equinox ``(out, in)`` storage order."""

    jax, jnp = _require_jax()
    dtype = config.jax_dtype()
    init_type = initialization.lower()
    if init_type == "xavier":
        limit = math.sqrt(6.0 / (in_dim + out_dim))
        weights = jax.random.uniform(
            key,
            (in_dim, out_dim),
            minval=-limit,
            maxval=limit,
            dtype=dtype,
        ).T
    elif init_type == "he":
        std = math.sqrt(2.0 / in_dim)
        # Generate in the historical (in, out) shape, then store in Equinox
        # order. This preserves the seeded initial function across the cleanup.
        weights = std * jax.random.normal(key, (in_dim, out_dim), dtype=dtype).T
    else:
        raise ValueError(f"unsupported initialization: {initialization!r}")
    return {"W": weights, "b": jnp.zeros((out_dim,), dtype=dtype)}


def init_mlp_params(
    key,
    layers: list[int] | tuple[int, ...],
    *,
    initialization: str = "he",
):
    """Initialize MLP parameters for native ``[xx0, xx1]`` inputs."""

    jax, _jnp = _require_jax()
    keys = jax.random.split(key, len(layers) - 1)
    return [
        _init_layer_params(layer_key, in_dim, out_dim, initialization)
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
    """Evaluate the MLP on native-coordinate inputs of shape ``(N, 2)``."""

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
