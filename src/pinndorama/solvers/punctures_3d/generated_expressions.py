"""Cached JAX callables for the NRPy-derived publication expressions."""

from __future__ import annotations

from functools import lru_cache
from typing import Any

import sympy as sp

from . import config
from . import nrpy_expression_builder as builder


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
def _regularized_residual_callable():
    expressions = builder.build_symbolic_expressions()
    return sp.lambdify(
        expressions.residual_symbols,
        expressions.regularized_residual_h,
        _jax_modules(),
        cse=True,
    )


@lru_cache(maxsize=1)
def _source_callable():
    expressions = builder.build_symbolic_expressions()
    return sp.lambdify(
        expressions.source_symbols,
        (expressions.psi_background, expressions.add_times_auu),
        _jax_modules(),
        cse=True,
    )


def source_terms_jax(
    xx0: Any,
    xx1: Any,
    xx2: Any,
    geometry: config.Geometry,
    physics: config.Physics,
) -> tuple[Any, Any]:
    """Evaluate the upstream two-puncture source with separation only on z."""

    parameters = {
        **config.source_parameter_dict(physics),
        "AMAX": geometry.AMAX,
        "bScale": geometry.bScale,
        "SINHWAA": geometry.SINHWAA,
    }
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
    return _source_callable()(*args)


def regularized_residual_h_jax(
    xx0: Any,
    xx1: Any,
    xx2: Any,
    uu: Any,
    uu_dD0: Any,
    uu_dD1: Any,
    uu_dD2: Any,
    uu_dDD00: Any,
    uu_dDD11: Any,
    uu_dDD22: Any,
    psi_background: Any,
    add_times_auu: Any,
    geometry: config.Geometry,
) -> Any:
    """Evaluate the sole supported residual, ``J * R_H``."""

    return _regularized_residual_callable()(
        xx0,
        xx1,
        xx2,
        geometry.AMAX,
        geometry.bScale,
        geometry.SINHWAA,
        uu,
        uu_dD0,
        uu_dD1,
        uu_dD2,
        uu_dDD00,
        uu_dDD11,
        uu_dDD22,
        psi_background,
        add_times_auu,
    )
