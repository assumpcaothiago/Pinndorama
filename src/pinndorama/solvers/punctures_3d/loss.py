"""Publication loss: ``J R_H`` plus the three native-coordinate constraints."""

from __future__ import annotations

from typing import Any

from . import config, coordinates, generated_expressions, model

SourceTerms = tuple[Any, Any]


def _model_scalar(params, xx0, xx1, xx2, geometry: config.Geometry):
    import jax.numpy as jnp

    point = jnp.asarray([[xx0, xx1, xx2]], dtype=jnp.float64)
    return model.forward(params, point, geometry)[0, 0]


def _regularized_residual_scalar(
    params,
    xx0,
    xx1,
    xx2,
    psi_background,
    add_times_auu,
    geometry: config.Geometry,
):
    import jax

    scalar_model = lambda p, x0, x1, x2: _model_scalar(p, x0, x1, x2, geometry)
    uu = scalar_model(params, xx0, xx1, xx2)
    uu_dD0 = jax.grad(scalar_model, argnums=1)(params, xx0, xx1, xx2)
    uu_dD1 = jax.grad(scalar_model, argnums=2)(params, xx0, xx1, xx2)
    uu_dD2 = jax.grad(scalar_model, argnums=3)(params, xx0, xx1, xx2)
    uu_dDD00 = jax.grad(jax.grad(scalar_model, argnums=1), argnums=1)(
        params, xx0, xx1, xx2
    )
    uu_dDD11 = jax.grad(jax.grad(scalar_model, argnums=2), argnums=2)(
        params, xx0, xx1, xx2
    )
    uu_dDD22 = jax.grad(jax.grad(scalar_model, argnums=3), argnums=3)(
        params, xx0, xx1, xx2
    )
    return generated_expressions.regularized_residual_h_jax(
        xx0,
        xx1,
        xx2,
        uu,
        uu_dD0,
        uu_dD1,
        uu_dD2,
        uu_dDD00,
        uu_dDD11,
        uu_dDD22,
        psi_background,
        add_times_auu,
        geometry,
    )


def precompute_source_terms(
    xx0,
    xx1,
    xx2,
    geometry: config.Geometry,
    physics: config.Physics,
) -> SourceTerms:
    import jax.numpy as jnp

    psi_background, add_times_auu = generated_expressions.source_terms_jax(
        jnp.asarray(xx0, dtype=jnp.float64).reshape(-1),
        jnp.asarray(xx1, dtype=jnp.float64).reshape(-1),
        jnp.asarray(xx2, dtype=jnp.float64).reshape(-1),
        geometry,
        physics,
    )
    return (
        jnp.asarray(psi_background, dtype=jnp.float64).reshape(-1),
        jnp.asarray(add_times_auu, dtype=jnp.float64).reshape(-1),
    )


def residual_loss(
    params,
    xx0,
    xx1,
    xx2,
    source_terms: SourceTerms,
    geometry: config.Geometry,
):
    import jax
    import jax.numpy as jnp

    psi_background, add_times_auu = source_terms
    residual = jax.vmap(
        lambda x0, x1, x2, psi, add: _regularized_residual_scalar(
            params, x0, x1, x2, psi, add, geometry
        )
    )(
        jnp.asarray(xx0, dtype=jnp.float64).reshape(-1),
        jnp.asarray(xx1, dtype=jnp.float64).reshape(-1),
        jnp.asarray(xx2, dtype=jnp.float64).reshape(-1),
        jnp.asarray(psi_background, dtype=jnp.float64).reshape(-1),
        jnp.asarray(add_times_auu, dtype=jnp.float64).reshape(-1),
    )
    return jnp.mean(residual * residual)


def _outer_robin_residual_scalar(params, xx0, xx1, xx2, geometry: config.Geometry):
    import jax
    import jax.numpy as jnp

    scalar_model = lambda p, x0, x1, x2: _model_scalar(p, x0, x1, x2, geometry)
    uu = scalar_model(params, xx0, xx1, xx2)
    uu_dD0 = jax.grad(scalar_model, argnums=1)(params, xx0, xx1, xx2)
    uu_dD1 = jax.grad(scalar_model, argnums=2)(params, xx0, xx1, xx2)
    radius = coordinates.spherical_radius_from_xx(xx0, xx1, geometry, xp=jnp)
    uu_dr = coordinates.spherical_radial_derivative_from_xx_gradient(
        xx0,
        xx1,
        uu_dD0,
        uu_dD1,
        geometry,
        xp=jnp,
    )
    return (1.0 + radius) * (radius * uu_dr + uu)


def outer_robin_boundary_loss(
    params,
    xx0,
    xx1,
    xx2,
    geometry: config.Geometry,
):
    import jax
    import jax.numpy as jnp

    residual = jax.vmap(
        lambda x0, x1, x2: _outer_robin_residual_scalar(params, x0, x1, x2, geometry)
    )(
        jnp.asarray(xx0, dtype=jnp.float64).reshape(-1),
        jnp.asarray(xx1, dtype=jnp.float64).reshape(-1),
        jnp.asarray(xx2, dtype=jnp.float64).reshape(-1),
    )
    return jnp.mean(residual * residual)


def _vmap_model(params, xx0, xx1, xx2, geometry: config.Geometry):
    import jax

    return jax.vmap(lambda x0, x1, x2: _model_scalar(params, x0, x1, x2, geometry))(
        xx0, xx1, xx2
    )


def theta_axis_regularity_loss(
    params,
    xx0,
    delta,
    phi,
    geometry: config.Geometry,
):
    import jax.numpy as jnp

    xx0 = jnp.asarray(xx0, dtype=jnp.float64).reshape(-1)
    delta = jnp.asarray(delta, dtype=jnp.float64).reshape(-1)
    phi = jnp.asarray(phi, dtype=jnp.float64).reshape(-1)
    antipodal_phi = jnp.mod(phi + 2.0 * jnp.pi, 2.0 * jnp.pi) - jnp.pi
    lower_inside = _vmap_model(params, xx0, delta, phi, geometry)
    lower_outside = _vmap_model(params, xx0, -delta, antipodal_phi, geometry)
    upper_inside = _vmap_model(params, xx0, jnp.pi - delta, phi, geometry)
    upper_outside = _vmap_model(params, xx0, jnp.pi + delta, antipodal_phi, geometry)
    return 0.5 * (
        jnp.mean((lower_outside - lower_inside) ** 2)
        + jnp.mean((upper_outside - upper_inside) ** 2)
    )


def phi_periodicity_loss(
    params,
    xx0,
    xx1,
    geometry: config.Geometry,
    collocation: config.Collocation,
):
    import jax.numpy as jnp

    xx0 = jnp.asarray(xx0, dtype=jnp.float64).reshape(-1)
    xx1 = jnp.asarray(xx1, dtype=jnp.float64).reshape(-1)
    xx2_min = jnp.zeros_like(xx0) + collocation.xx2_min
    xx2_max = jnp.zeros_like(xx0) + collocation.xx2_max
    u_min = _vmap_model(params, xx0, xx1, xx2_min, geometry)
    u_max = _vmap_model(params, xx0, xx1, xx2_max, geometry)
    return jnp.mean((u_max - u_min) ** 2)


def compute_loss(
    params,
    xx0,
    xx1,
    xx2,
    *,
    axis_xx0,
    axis_delta,
    axis_phi,
    periodic_xx0,
    periodic_xx1,
    outer_xx0,
    outer_xx1,
    outer_xx2,
    source_terms: SourceTerms,
    geometry: config.Geometry,
    physics: config.Physics,
    collocation: config.Collocation,
):
    """Return total loss and the four publication components."""

    residual = residual_loss(params, xx0, xx1, xx2, source_terms, geometry)
    theta_axis = theta_axis_regularity_loss(
        params, axis_xx0, axis_delta, axis_phi, geometry
    )
    phi_periodic = phi_periodicity_loss(
        params, periodic_xx0, periodic_xx1, geometry, collocation
    )
    outer = outer_robin_boundary_loss(params, outer_xx0, outer_xx1, outer_xx2, geometry)
    total = (
        residual
        + physics.theta_axis_regularity_weight * theta_axis
        + physics.phi_periodicity_weight * phi_periodic
        + physics.outer_robin_weight * outer
    )
    return total, {
        "residual_loss": residual,
        "theta_axis_regularity_loss": theta_axis,
        "phi_periodicity_loss": phi_periodic,
        "outer_robin_boundary_loss": outer,
    }
