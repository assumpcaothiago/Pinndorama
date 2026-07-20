"""Lazy numerical wrappers around NRPy-generated SinhSymTP expressions."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import numpy as np
import sympy as sp

from . import config
from . import nrpy_expression_builder as builder


def _with_overrides(
    defaults: dict[str, float], overrides: dict[str, Any]
) -> dict[str, Any]:
    unknown = sorted(set(overrides) - set(defaults))
    if unknown:
        raise TypeError(f"unknown source parameters: {', '.join(unknown)}")
    values = dict(defaults)
    for name, value in overrides.items():
        if value is not None:
            values[name] = value
    return values


@lru_cache(maxsize=1)
def _numpy_residual_callable():
    expressions = builder.build_symbolic_expressions()
    return sp.lambdify(expressions.residual_symbols, expressions.residual_h, "numpy")


@lru_cache(maxsize=1)
def _numpy_regularized_residual_callable():
    expressions = builder.build_symbolic_expressions()
    return sp.lambdify(
        expressions.residual_symbols,
        expressions.regularized_residual_h,
        "numpy",
    )


@lru_cache(maxsize=1)
def _numpy_volume_element_callable():
    expressions = builder.build_symbolic_expressions()
    return sp.lambdify(
        expressions.geometry_symbols,
        expressions.volume_element,
        "numpy",
    )


@lru_cache(maxsize=1)
def _numpy_source_callable():
    expressions = builder.build_symbolic_expressions()
    return sp.lambdify(
        expressions.source_symbols,
        (expressions.psi_background, expressions.add_times_auu),
        "numpy",
    )


def _jax_modules():
    import jax.numpy as jnp

    return [
        {
            "sin": jnp.sin,
            "cos": jnp.cos,
            "tan": jnp.tan,
            "sinh": jnp.sinh,
            "cosh": jnp.cosh,
            "exp": jnp.exp,
            "sqrt": jnp.sqrt,
            "Abs": jnp.abs,
        }
    ]


@lru_cache(maxsize=1)
def _jax_residual_callable():
    expressions = builder.build_symbolic_expressions()
    return sp.lambdify(
        expressions.residual_symbols,
        expressions.residual_h,
        _jax_modules(),
        cse=True,
    )


@lru_cache(maxsize=1)
def _jax_regularized_residual_callable():
    expressions = builder.build_symbolic_expressions()
    return sp.lambdify(
        expressions.residual_symbols,
        expressions.regularized_residual_h,
        _jax_modules(),
        cse=True,
    )


@lru_cache(maxsize=1)
def _jax_volume_element_callable():
    expressions = builder.build_symbolic_expressions()
    return sp.lambdify(
        expressions.geometry_symbols,
        expressions.volume_element,
        _jax_modules(),
        cse=True,
    )


@lru_cache(maxsize=1)
def _jax_source_callable():
    expressions = builder.build_symbolic_expressions()
    return sp.lambdify(
        expressions.source_symbols,
        (expressions.psi_background, expressions.add_times_auu),
        _jax_modules(),
        cse=True,
    )


def source_terms_numpy(
    xx0: Any,
    xx1: Any,
    xx2: Any = 0.0,
    **parameter_overrides: Any,
) -> tuple[Any, Any]:
    """Evaluate NRPy-generated ``psi_background`` and ``ADD_times_AUU`` with NumPy."""

    parameters = _with_overrides(config.source_parameter_dict(), parameter_overrides)
    args = []
    for name in builder.SOURCE_INPUT_NAMES:
        if name == "xx0":
            args.append(xx0)
        elif name == "xx1":
            args.append(xx1)
        elif name == "xx2":
            args.append(xx2)
        else:
            args.append(parameters[name])
    return _numpy_source_callable()(*args)


def volume_element_numpy(
    xx0: Any,
    xx1: Any,
    *,
    AMAX: float = config.AMAX,
    bScale: float = config.bScale,
    SINHWAA: float = config.SINHWAA,
) -> Any:
    """Evaluate ``sqrt(det(gammahat))`` for SinhSymTP with NumPy."""

    return _numpy_volume_element_callable()(xx0, xx1, AMAX, bScale, SINHWAA)


def volume_element_jax(
    xx0: Any,
    xx1: Any,
    *,
    AMAX: float = config.AMAX,
    bScale: float = config.bScale,
    SINHWAA: float = config.SINHWAA,
) -> Any:
    """Evaluate ``sqrt(det(gammahat))`` for SinhSymTP with JAX operations."""

    return _jax_volume_element_callable()(xx0, xx1, AMAX, bScale, SINHWAA)


def source_terms_jax(
    xx0: Any,
    xx1: Any,
    xx2: Any = 0.0,
    **parameter_overrides: Any,
) -> tuple[Any, Any]:
    """Evaluate NRPy-generated source terms with JAX operations."""

    parameters = _with_overrides(config.source_parameter_dict(), parameter_overrides)
    args = []
    for name in builder.SOURCE_INPUT_NAMES:
        if name == "xx0":
            args.append(xx0)
        elif name == "xx1":
            args.append(xx1)
        elif name == "xx2":
            args.append(xx2)
        else:
            args.append(parameters[name])
    return _jax_source_callable()(*args)


def residual_h_numpy(
    xx0: Any,
    xx1: Any,
    uu: Any,
    uu_dD0: Any,
    uu_dD1: Any,
    uu_dDD00: Any,
    uu_dDD11: Any,
    psi_background: Any,
    ADD_times_AUU: Any,
    *,
    AMAX: float = config.AMAX,
    bScale: float = config.bScale,
    SINHWAA: float = config.SINHWAA,
) -> Any:
    """Evaluate the NRPy-generated Hamiltonian residual with NumPy."""

    return _numpy_residual_callable()(
        xx0,
        xx1,
        AMAX,
        bScale,
        SINHWAA,
        uu,
        uu_dD0,
        uu_dD1,
        uu_dDD00,
        uu_dDD11,
        psi_background,
        ADD_times_AUU,
    )


def regularized_residual_h_numpy(
    xx0: Any,
    xx1: Any,
    uu: Any,
    uu_dD0: Any,
    uu_dD1: Any,
    uu_dDD00: Any,
    uu_dDD11: Any,
    psi_background: Any,
    ADD_times_AUU: Any,
    *,
    AMAX: float = config.AMAX,
    bScale: float = config.bScale,
    SINHWAA: float = config.SINHWAA,
) -> Any:
    """Evaluate the regularized ``J * residual_H`` with NumPy."""

    return _numpy_regularized_residual_callable()(
        xx0,
        xx1,
        AMAX,
        bScale,
        SINHWAA,
        uu,
        uu_dD0,
        uu_dD1,
        uu_dDD00,
        uu_dDD11,
        psi_background,
        ADD_times_AUU,
    )


def residual_h_jax(
    xx0: Any,
    xx1: Any,
    uu: Any,
    uu_dD0: Any,
    uu_dD1: Any,
    uu_dDD00: Any,
    uu_dDD11: Any,
    psi_background: Any,
    ADD_times_AUU: Any,
    *,
    AMAX: float = config.AMAX,
    bScale: float = config.bScale,
    SINHWAA: float = config.SINHWAA,
) -> Any:
    """Evaluate the NRPy-generated Hamiltonian residual with JAX operations."""

    return _jax_residual_callable()(
        xx0,
        xx1,
        AMAX,
        bScale,
        SINHWAA,
        uu,
        uu_dD0,
        uu_dD1,
        uu_dDD00,
        uu_dDD11,
        psi_background,
        ADD_times_AUU,
    )


def regularized_residual_h_jax(
    xx0: Any,
    xx1: Any,
    uu: Any,
    uu_dD0: Any,
    uu_dD1: Any,
    uu_dDD00: Any,
    uu_dDD11: Any,
    psi_background: Any,
    ADD_times_AUU: Any,
    *,
    AMAX: float = config.AMAX,
    bScale: float = config.bScale,
    SINHWAA: float = config.SINHWAA,
) -> Any:
    """Evaluate the regularized ``J * residual_H`` with JAX operations."""

    return _jax_regularized_residual_callable()(
        xx0,
        xx1,
        AMAX,
        bScale,
        SINHWAA,
        uu,
        uu_dD0,
        uu_dD1,
        uu_dDD00,
        uu_dDD11,
        psi_background,
        ADD_times_AUU,
    )


def finite_stats(values: Any) -> dict[str, float | int | None]:
    """Return compact finite-value statistics for reports and tests."""

    array = np.asarray(values, dtype=np.float64)
    finite = array[np.isfinite(array)]
    if finite.size == 0:
        return {"count": 0, "max_abs": None, "median_abs": None, "rms": None}
    return {
        "count": int(finite.size),
        "max_abs": float(np.max(np.abs(finite))),
        "median_abs": float(np.median(np.abs(finite))),
        "rms": float(np.sqrt(np.mean(finite * finite))),
    }
