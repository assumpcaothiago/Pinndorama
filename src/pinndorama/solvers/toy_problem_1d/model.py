"""Float64 one-input MLP with physical inverse-radius falloff."""

from __future__ import annotations

import math

import numpy as np

from . import config, coordinates


def _init_layer_params(key, in_dim: int, out_dim: int, initialization: str):
    config.enable_jax_x64()
    import jax
    import jax.numpy as jnp

    if initialization == "xavier":
        limit = math.sqrt(6.0 / (in_dim + out_dim))
        weights = jax.random.uniform(
            key,
            (in_dim, out_dim),
            minval=-limit,
            maxval=limit,
            dtype=jnp.float64,
        ).T
    elif initialization == "he":
        weights = (
            math.sqrt(2.0 / in_dim)
            * jax.random.normal(key, (in_dim, out_dim), dtype=jnp.float64).T
        )
    else:
        raise ValueError(f"unsupported initialization: {initialization!r}")
    return {"W": weights, "b": jnp.zeros((out_dim,), dtype=jnp.float64)}


def init_mlp_params(key, layers, *, initialization: str = "he"):
    config.enable_jax_x64()
    import jax

    keys = jax.random.split(key, len(layers) - 1)
    return [
        _init_layer_params(layer_key, in_dim, out_dim, initialization)
        for layer_key, in_dim, out_dim in zip(keys, layers[:-1], layers[1:])
    ]


def forward(params, x, *, ampl: float, sinhw: float):
    """Evaluate ``u = raw / sqrt(1+r^2)`` for shape ``(N, 1)`` inputs."""

    config.enable_jax_x64()
    import jax.numpy as jnp

    native = jnp.asarray(x, dtype=jnp.float64)
    if native.ndim != 2 or native.shape[1] != 1:
        raise ValueError("model input must have shape (N, 1)")
    hidden = native
    for layer in params[:-1]:
        hidden = jnp.tanh(hidden @ layer["W"].T + layer["b"])
    raw = hidden @ params[-1]["W"].T + params[-1]["b"]
    radius = coordinates.radius_from_xx0(native[:, 0:1], ampl=ampl, sinhw=sinhw, xp=jnp)
    return raw / jnp.sqrt(1.0 + radius * radius)


def forward_numpy(params, x, *, ampl: float, sinhw: float) -> np.ndarray:
    """Evaluate the trained model with NumPy for lightweight postprocessing."""

    native = np.asarray(x, dtype=np.float64)
    if native.ndim != 2 or native.shape[1] != 1:
        raise ValueError("model input must have shape (N, 1)")
    hidden = native
    for layer in params[:-1]:
        weight = np.asarray(layer["W"], dtype=np.float64)
        bias = np.asarray(layer["b"], dtype=np.float64)
        hidden = np.tanh(hidden @ weight.T + bias)
    final_weight = np.asarray(params[-1]["W"], dtype=np.float64)
    final_bias = np.asarray(params[-1]["b"], dtype=np.float64)
    raw = hidden @ final_weight.T + final_bias
    radius = coordinates.radius_from_xx0(native[:, 0:1], ampl=ampl, sinhw=sinhw, xp=np)
    return raw / np.sqrt(1.0 + radius * radius)
