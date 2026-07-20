"""Parametric native ``xx``-coordinate loss for the SinhSymTP 2D experiment."""

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
    equal_spin_sz_scalar,
    AMAX,
    bScale,
    SINHWAA,
):
    """Evaluate the model at a single native-coordinate point."""

    _jax, jnp = _require_jax()
    x = jnp.array(
        [[xx0_scalar, xx1_scalar, equal_spin_sz_scalar]], dtype=config.jax_dtype()
    )
    return model.forward(
        params,
        x,
        amax=AMAX,
        bscale=bScale,
        sinhwaa=SINHWAA,
    )[0, 0]


def _residual_scalar(
    params,
    xx0_scalar,
    xx1_scalar,
    equal_spin_sz_scalar,
    AMAX,
    bScale,
    SINHWAA,
):
    """Evaluate the scaled regularized native residual for one model point."""

    _jax, _jnp = _require_jax()
    psi_background, add_times_auu = generated_expressions.equal_spin_source_terms_jax(
        xx0_scalar,
        xx1_scalar,
        equal_spin_sz_scalar,
        xx2=0.0,
        AMAX=AMAX,
        bScale=bScale,
        SINHWAA=SINHWAA,
    )
    return _residual_scalar_with_source(
        params,
        xx0_scalar,
        xx1_scalar,
        equal_spin_sz_scalar,
        psi_background,
        add_times_auu,
        AMAX,
        bScale,
        SINHWAA,
    )


def _residual_scalar_with_source(
    params,
    xx0_scalar,
    xx1_scalar,
    equal_spin_sz_scalar,
    psi_background,
    add_times_auu,
    AMAX,
    bScale,
    SINHWAA,
):
    """Evaluate the scaled regularized residual using precomputed source terms."""

    jax, jnp = _require_jax()
    uu = _model_scalar(
        params,
        xx0_scalar,
        xx1_scalar,
        equal_spin_sz_scalar,
        AMAX,
        bScale,
        SINHWAA,
    )
    uu_dD0 = jax.grad(_model_scalar, argnums=1)(
        params, xx0_scalar, xx1_scalar, equal_spin_sz_scalar, AMAX, bScale, SINHWAA
    )
    uu_dD1 = jax.grad(_model_scalar, argnums=2)(
        params, xx0_scalar, xx1_scalar, equal_spin_sz_scalar, AMAX, bScale, SINHWAA
    )
    uu_dDD00 = jax.grad(jax.grad(_model_scalar, argnums=1), argnums=1)(
        params, xx0_scalar, xx1_scalar, equal_spin_sz_scalar, AMAX, bScale, SINHWAA
    )
    uu_dDD11 = jax.grad(jax.grad(_model_scalar, argnums=2), argnums=2)(
        params, xx0_scalar, xx1_scalar, equal_spin_sz_scalar, AMAX, bScale, SINHWAA
    )
    regularized_residual = generated_expressions.regularized_residual_h_jax(
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
    scale = jnp.asarray(config.regularized_residual_scale, dtype=config.jax_dtype())
    return regularized_residual / scale


def precompute_source_terms(
    xx0,
    xx1,
    equal_spin_sz,
    *,
    AMAX: float = config.AMAX,
    bScale: float = config.bScale,
    SINHWAA: float = config.SINHWAA,
    physics: Mapping[str, float] | None = None,
) -> SourceTerms:
    """Return sources with ``S0_z=S1_z=equal_spin_sz``."""

    _jax, jnp = _require_jax()
    xx0_flat = jnp.asarray(xx0, dtype=config.jax_dtype()).reshape(-1)
    xx1_flat = jnp.asarray(xx1, dtype=config.jax_dtype()).reshape(-1)
    equal_spin_sz_flat = jnp.asarray(equal_spin_sz, dtype=config.jax_dtype()).reshape(
        -1
    )
    psi_background, add_times_auu = generated_expressions.equal_spin_source_terms_jax(
        xx0_flat,
        xx1_flat,
        equal_spin_sz_flat,
        xx2=0.0,
        physics=physics,
        AMAX=AMAX,
        bScale=bScale,
        SINHWAA=SINHWAA,
    )
    return (
        jnp.asarray(psi_background, dtype=config.jax_dtype()).reshape(-1),
        jnp.asarray(add_times_auu, dtype=config.jax_dtype()).reshape(-1),
    )


def residual_loss(
    params,
    xx0,
    xx1,
    equal_spin_sz,
    *,
    source_terms: SourceTerms | None = None,
    AMAX: float = config.AMAX,
    bScale: float = config.bScale,
    SINHWAA: float = config.SINHWAA,
):
    """Mean-square scaled regularized residual over native collocation points."""

    jax, jnp = _require_jax()
    xx0_flat = jnp.asarray(xx0, dtype=config.jax_dtype()).reshape(-1)
    xx1_flat = jnp.asarray(xx1, dtype=config.jax_dtype()).reshape(-1)
    equal_spin_sz_flat = jnp.asarray(equal_spin_sz, dtype=config.jax_dtype()).reshape(
        -1
    )
    if source_terms is None:
        residual = jax.vmap(
            _residual_scalar,
            in_axes=(None, 0, 0, 0, None, None, None),
        )(
            params,
            xx0_flat,
            xx1_flat,
            equal_spin_sz_flat,
            AMAX,
            bScale,
            SINHWAA,
        )
    else:
        psi_background, add_times_auu = source_terms
        residual = jax.vmap(
            _residual_scalar_with_source,
            in_axes=(None, 0, 0, 0, 0, 0, None, None, None),
        )(
            params,
            xx0_flat,
            xx1_flat,
            equal_spin_sz_flat,
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
    equal_spin_sz_scalar,
    AMAX,
    bScale,
    SINHWAA,
):
    """Return scaled ``r du/dr + u`` at one outer-boundary sample."""

    jax, jnp = _require_jax()
    uu = _model_scalar(
        params,
        xx0_scalar,
        xx1_scalar,
        equal_spin_sz_scalar,
        AMAX,
        bScale,
        SINHWAA,
    )
    uu_dD0 = jax.grad(_model_scalar, argnums=1)(
        params, xx0_scalar, xx1_scalar, equal_spin_sz_scalar, AMAX, bScale, SINHWAA
    )
    uu_dD1 = jax.grad(_model_scalar, argnums=2)(
        params, xx0_scalar, xx1_scalar, equal_spin_sz_scalar, AMAX, bScale, SINHWAA
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
    if config.outer_robin_boundary_scale == "one_plus_r":
        return (1.0 + r_sph) * raw_residual
    if config.outer_robin_boundary_scale == "none":
        return raw_residual
    raise ValueError(
        f"Unsupported outer_robin_boundary_scale: {config.outer_robin_boundary_scale}"
    )


def outer_robin_boundary_residual(
    params,
    xx0,
    xx1,
    equal_spin_sz,
    *,
    AMAX: float = config.AMAX,
    bScale: float = config.bScale,
    SINHWAA: float = config.SINHWAA,
):
    """Vectorized scaled outer Robin residual over boundary samples."""

    jax, jnp = _require_jax()
    xx0_flat = jnp.asarray(xx0, dtype=config.jax_dtype()).reshape(-1)
    xx1_flat = jnp.asarray(xx1, dtype=config.jax_dtype()).reshape(-1)
    equal_spin_sz_flat = jnp.asarray(equal_spin_sz, dtype=config.jax_dtype()).reshape(
        -1
    )
    return jax.vmap(
        _outer_robin_boundary_residual_scalar,
        in_axes=(None, 0, 0, 0, None, None, None),
    )(
        params,
        xx0_flat,
        xx1_flat,
        equal_spin_sz_flat,
        AMAX,
        bScale,
        SINHWAA,
    )


def outer_robin_boundary_loss(
    params,
    xx0_samples=None,
    xx1_samples=None,
    equal_spin_sz_samples=None,
    *,
    AMAX: float = config.AMAX,
    bScale: float = config.bScale,
    SINHWAA: float = config.SINHWAA,
):
    """Mean-square scaled outer Robin falloff condition."""

    _jax, jnp = _require_jax()
    if xx0_samples is None or xx1_samples is None:
        xx0_samples, xx1_samples = coordinates.outer_robin_boundary_samples(xp=jnp)
    if equal_spin_sz_samples is None:
        equal_spin_sz_samples = jnp.zeros_like(
            jnp.asarray(xx0_samples, dtype=config.jax_dtype())
        )
    residual = outer_robin_boundary_residual(
        params,
        xx0_samples,
        xx1_samples,
        equal_spin_sz_samples,
        AMAX=AMAX,
        bScale=bScale,
        SINHWAA=SINHWAA,
    )
    return jnp.mean(residual * residual)


def theta_inner_parity_loss(
    params,
    xx0_samples=None,
    delta_samples=None,
    equal_spin_sz_samples=None,
    *,
    AMAX: float = config.AMAX,
    bScale: float = config.bScale,
    SINHWAA: float = config.SINHWAA,
):
    """Scalar parity loss at the lower and upper theta inner boundaries."""

    _jax, jnp = _require_jax()
    if xx0_samples is None or delta_samples is None:
        xx0_samples, delta_samples = coordinates.theta_inner_parity_samples(xp=jnp)
    if equal_spin_sz_samples is None:
        equal_spin_sz_samples = jnp.zeros_like(
            jnp.asarray(xx0_samples, dtype=config.jax_dtype())
        )
    xx0 = jnp.asarray(xx0_samples, dtype=config.jax_dtype()).reshape(-1)
    delta = jnp.asarray(delta_samples, dtype=config.jax_dtype()).reshape(-1)
    equal_spin_sz = jnp.asarray(
        equal_spin_sz_samples, dtype=config.jax_dtype()
    ).reshape(-1)

    lower_inside = jax_vmap_model(
        params, xx0, delta, equal_spin_sz, AMAX=AMAX, bScale=bScale, SINHWAA=SINHWAA
    )
    lower_outside = jax_vmap_model(
        params, xx0, -delta, equal_spin_sz, AMAX=AMAX, bScale=bScale, SINHWAA=SINHWAA
    )
    upper_inside = jax_vmap_model(
        params,
        xx0,
        jnp.pi - delta,
        equal_spin_sz,
        AMAX=AMAX,
        bScale=bScale,
        SINHWAA=SINHWAA,
    )
    upper_outside = jax_vmap_model(
        params,
        xx0,
        jnp.pi + delta,
        equal_spin_sz,
        AMAX=AMAX,
        bScale=bScale,
        SINHWAA=SINHWAA,
    )
    return 0.5 * (
        jnp.mean((lower_outside - lower_inside) ** 2)
        + jnp.mean((upper_outside - upper_inside) ** 2)
    )


def jax_vmap_model(
    params,
    xx0,
    xx1,
    equal_spin_sz,
    *,
    AMAX: float = config.AMAX,
    bScale: float = config.bScale,
    SINHWAA: float = config.SINHWAA,
):
    """Vectorized scalar model helper."""

    jax, _jnp = _require_jax()
    return jax.vmap(
        _model_scalar,
        in_axes=(None, 0, 0, 0, None, None, None),
    )(params, xx0, xx1, equal_spin_sz, AMAX, bScale, SINHWAA)


def compute_loss(
    params,
    xx0,
    xx1,
    equal_spin_sz,
    *,
    parity_xx0=None,
    parity_delta=None,
    parity_equal_spin_sz=None,
    outer_xx0=None,
    outer_xx1=None,
    outer_equal_spin_sz=None,
    source_terms: SourceTerms | None = None,
    AMAX: float = config.AMAX,
    bScale: float = config.bScale,
    SINHWAA: float = config.SINHWAA,
):
    """Return total native loss and named components."""

    residual = residual_loss(
        params,
        xx0,
        xx1,
        equal_spin_sz,
        source_terms=source_terms,
        AMAX=AMAX,
        bScale=bScale,
        SINHWAA=SINHWAA,
    )
    parity = theta_inner_parity_loss(
        params,
        parity_xx0,
        parity_delta,
        parity_equal_spin_sz,
        AMAX=AMAX,
        bScale=bScale,
        SINHWAA=SINHWAA,
    )
    outer = outer_robin_boundary_loss(
        params,
        outer_xx0,
        outer_xx1,
        outer_equal_spin_sz,
        AMAX=AMAX,
        bScale=bScale,
        SINHWAA=SINHWAA,
    )
    total = (
        config.int_weight * residual
        + config.theta_inner_parity_weight * parity
        + config.outer_robin_boundary_weight * outer
    )
    return total, {
        "residual_loss": residual,
        "theta_inner_parity_loss": parity,
        "outer_robin_boundary_loss": outer,
    }
