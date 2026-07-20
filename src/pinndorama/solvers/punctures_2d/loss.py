"""Native ``xx``-coordinate loss for the SinhSymTP 2D experiment."""

from __future__ import annotations

from typing import Mapping

from . import config, coordinates, generated_expressions, model

SourceTerms = tuple[object, object]


def _require_jax():
    config.enable_jax_x64()
    import jax
    import jax.numpy as jnp

    return jax, jnp


def _model_scalar(
    params,
    xx0_scalar,
    xx1_scalar,
    AMAX,
    bScale,
    SINHWAA,
):
    """Evaluate the model at a single native-coordinate point."""

    _jax, jnp = _require_jax()
    x = jnp.array([[xx0_scalar, xx1_scalar]], dtype=config.jax_dtype())
    return model.forward(
        params,
        x,
        amax=AMAX,
        bscale=bScale,
        sinhwaa=SINHWAA,
    )[0, 0]


def _regularized_residual_scalar_with_source(
    params,
    xx0_scalar,
    xx1_scalar,
    psi_background,
    add_times_auu,
    AMAX,
    bScale,
    SINHWAA,
):
    """Evaluate the divergence-form regularized residual ``J R_H``."""

    jax, jnp = _require_jax()
    uu = _model_scalar(params, xx0_scalar, xx1_scalar, AMAX, bScale, SINHWAA)
    uu_dD0 = jax.grad(_model_scalar, argnums=1)(
        params, xx0_scalar, xx1_scalar, AMAX, bScale, SINHWAA
    )
    uu_dD1 = jax.grad(_model_scalar, argnums=2)(
        params, xx0_scalar, xx1_scalar, AMAX, bScale, SINHWAA
    )
    uu_dDD00 = jax.grad(jax.grad(_model_scalar, argnums=1), argnums=1)(
        params, xx0_scalar, xx1_scalar, AMAX, bScale, SINHWAA
    )
    uu_dDD11 = jax.grad(jax.grad(_model_scalar, argnums=2), argnums=2)(
        params, xx0_scalar, xx1_scalar, AMAX, bScale, SINHWAA
    )
    return generated_expressions.regularized_residual_h_jax(
        xx0_scalar,
        xx1_scalar,
        uu,
        uu_dD0,
        uu_dD1,
        uu_dDD00,
        uu_dDD11,
        psi_background,
        add_times_auu,
        AMAX=AMAX,
        bScale=bScale,
        SINHWAA=SINHWAA,
    )


def precompute_source_terms(
    xx0,
    xx1,
    *,
    AMAX: float = config.AMAX,
    bScale: float = config.bScale,
    SINHWAA: float = config.SINHWAA,
    physics: Mapping[str, float] | None = None,
) -> SourceTerms:
    """Return fixed-grid NRPy source terms for residual collocation points."""

    _jax, jnp = _require_jax()
    xx0_flat = jnp.asarray(xx0, dtype=config.jax_dtype()).reshape(-1)
    xx1_flat = jnp.asarray(xx1, dtype=config.jax_dtype()).reshape(-1)
    source_parameters = config.DEFAULT_PHYSICS if physics is None else dict(physics)
    psi_background, add_times_auu = generated_expressions.source_terms_jax(
        xx0_flat,
        xx1_flat,
        0.0,
        AMAX=AMAX,
        bScale=bScale,
        SINHWAA=SINHWAA,
        **source_parameters,
    )
    return (
        jnp.asarray(psi_background, dtype=config.jax_dtype()).reshape(-1),
        jnp.asarray(add_times_auu, dtype=config.jax_dtype()).reshape(-1),
    )


def residual_loss(
    params,
    xx0,
    xx1,
    *,
    source_terms: SourceTerms | None = None,
    AMAX: float = config.AMAX,
    bScale: float = config.bScale,
    SINHWAA: float = config.SINHWAA,
    physics: Mapping[str, float] | None = None,
):
    """Mean-square divergence-form ``J R_H`` on native collocation points."""

    jax, jnp = _require_jax()
    xx0_flat = jnp.asarray(xx0, dtype=config.jax_dtype()).reshape(-1)
    xx1_flat = jnp.asarray(xx1, dtype=config.jax_dtype()).reshape(-1)
    if source_terms is None:
        source_terms = precompute_source_terms(
            xx0_flat,
            xx1_flat,
            AMAX=AMAX,
            bScale=bScale,
            SINHWAA=SINHWAA,
            physics=physics,
        )
    psi_background, add_times_auu = source_terms
    residual = jax.vmap(
        _regularized_residual_scalar_with_source,
        in_axes=(None, 0, 0, 0, 0, None, None, None),
    )(
        params,
        xx0_flat,
        xx1_flat,
        jnp.asarray(psi_background, dtype=config.jax_dtype()).reshape(-1),
        jnp.asarray(add_times_auu, dtype=config.jax_dtype()).reshape(-1),
        AMAX,
        bScale,
        SINHWAA,
    )
    return jnp.mean(residual * residual)


def _outer_robin_boundary_residual_scalar(
    params,
    xx0_scalar,
    xx1_scalar,
    AMAX,
    bScale,
    SINHWAA,
):
    """Return scaled ``r du/dr + u`` at one outer-boundary sample."""

    jax, jnp = _require_jax()
    uu = _model_scalar(params, xx0_scalar, xx1_scalar, AMAX, bScale, SINHWAA)
    uu_dD0 = jax.grad(_model_scalar, argnums=1)(
        params, xx0_scalar, xx1_scalar, AMAX, bScale, SINHWAA
    )
    uu_dD1 = jax.grad(_model_scalar, argnums=2)(
        params, xx0_scalar, xx1_scalar, AMAX, bScale, SINHWAA
    )
    r_sph = coordinates.spherical_radius_from_xx(
        xx0_scalar,
        xx1_scalar,
        amax=AMAX,
        bscale=bScale,
        sinhwaa=SINHWAA,
        xp=jnp,
    )
    uu_dr = coordinates.spherical_radial_derivative_from_xx_gradient(
        xx0_scalar,
        xx1_scalar,
        uu_dD0,
        uu_dD1,
        amax=AMAX,
        bscale=bScale,
        sinhwaa=SINHWAA,
        xp=jnp,
    )
    raw_residual = r_sph * uu_dr + uu
    return (1.0 + r_sph) * raw_residual


def outer_robin_boundary_residual(
    params,
    xx0,
    xx1,
    *,
    AMAX: float = config.AMAX,
    bScale: float = config.bScale,
    SINHWAA: float = config.SINHWAA,
):
    """Vectorized scaled outer Robin residual over boundary samples."""

    jax, jnp = _require_jax()
    xx0_flat = jnp.asarray(xx0, dtype=config.jax_dtype()).reshape(-1)
    xx1_flat = jnp.asarray(xx1, dtype=config.jax_dtype()).reshape(-1)
    return jax.vmap(
        _outer_robin_boundary_residual_scalar,
        in_axes=(None, 0, 0, None, None, None),
    )(
        params,
        xx0_flat,
        xx1_flat,
        AMAX,
        bScale,
        SINHWAA,
    )


def outer_robin_boundary_loss(
    params,
    xx0_samples=None,
    xx1_samples=None,
    *,
    AMAX: float = config.AMAX,
    bScale: float = config.bScale,
    SINHWAA: float = config.SINHWAA,
):
    """Mean-square scaled outer Robin falloff condition."""

    _jax, jnp = _require_jax()
    if xx0_samples is None or xx1_samples is None:
        xx0_samples, xx1_samples = coordinates.outer_robin_boundary_samples(xp=jnp)
    residual = outer_robin_boundary_residual(
        params,
        xx0_samples,
        xx1_samples,
        AMAX=AMAX,
        bScale=bScale,
        SINHWAA=SINHWAA,
    )
    return jnp.mean(residual * residual)


def theta_inner_parity_loss(
    params,
    xx0_samples=None,
    delta_samples=None,
    *,
    AMAX: float = config.AMAX,
    bScale: float = config.bScale,
    SINHWAA: float = config.SINHWAA,
):
    """Scalar parity loss at the lower and upper theta inner boundaries."""

    _jax, jnp = _require_jax()
    if xx0_samples is None or delta_samples is None:
        xx0_samples, delta_samples = coordinates.theta_inner_parity_samples(xp=jnp)
    xx0 = jnp.asarray(xx0_samples, dtype=config.jax_dtype()).reshape(-1)
    delta = jnp.asarray(delta_samples, dtype=config.jax_dtype()).reshape(-1)

    lower_inside = jax_vmap_model(
        params, xx0, delta, AMAX=AMAX, bScale=bScale, SINHWAA=SINHWAA
    )
    lower_outside = jax_vmap_model(
        params, xx0, -delta, AMAX=AMAX, bScale=bScale, SINHWAA=SINHWAA
    )
    upper_inside = jax_vmap_model(
        params, xx0, jnp.pi - delta, AMAX=AMAX, bScale=bScale, SINHWAA=SINHWAA
    )
    upper_outside = jax_vmap_model(
        params, xx0, jnp.pi + delta, AMAX=AMAX, bScale=bScale, SINHWAA=SINHWAA
    )
    return 0.5 * (
        jnp.mean((lower_outside - lower_inside) ** 2)
        + jnp.mean((upper_outside - upper_inside) ** 2)
    )


def jax_vmap_model(
    params,
    xx0,
    xx1,
    *,
    AMAX: float = config.AMAX,
    bScale: float = config.bScale,
    SINHWAA: float = config.SINHWAA,
):
    """Vectorized scalar model helper."""

    jax, _jnp = _require_jax()
    return jax.vmap(
        _model_scalar,
        in_axes=(None, 0, 0, None, None, None),
    )(params, xx0, xx1, AMAX, bScale, SINHWAA)


def compute_loss(
    params,
    xx0,
    xx1,
    *,
    parity_xx0=None,
    parity_delta=None,
    outer_xx0=None,
    outer_xx1=None,
    source_terms: SourceTerms | None = None,
    AMAX: float = config.AMAX,
    bScale: float = config.bScale,
    SINHWAA: float = config.SINHWAA,
    physics: Mapping[str, float] | None = None,
    interior_weight: float = 1.0,
    theta_parity_weight: float = 1.0,
    outer_robin_weight: float = 1.0,
):
    """Return total native loss and named components."""

    residual = residual_loss(
        params,
        xx0,
        xx1,
        source_terms=source_terms,
        AMAX=AMAX,
        bScale=bScale,
        SINHWAA=SINHWAA,
        physics=physics,
    )
    parity = theta_inner_parity_loss(
        params,
        parity_xx0,
        parity_delta,
        AMAX=AMAX,
        bScale=bScale,
        SINHWAA=SINHWAA,
    )
    outer = outer_robin_boundary_loss(
        params,
        outer_xx0,
        outer_xx1,
        AMAX=AMAX,
        bScale=bScale,
        SINHWAA=SINHWAA,
    )
    total = (
        interior_weight * residual
        + theta_parity_weight * parity
        + outer_robin_weight * outer
    )
    return total, {
        "residual_loss": residual,
        "theta_inner_parity_loss": parity,
        "outer_robin_boundary_loss": outer,
    }
