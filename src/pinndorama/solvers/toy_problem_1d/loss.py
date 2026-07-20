"""Regularized PDE, origin, and outer-boundary losses for the 1D toy problem."""

from __future__ import annotations

from . import config, coordinates, equation, model


def _model_scalar(params, xx0_scalar, ampl, sinhw):
    config.enable_jax_x64()
    import jax.numpy as jnp

    point = jnp.asarray([[xx0_scalar]], dtype=jnp.float64)
    return model.forward(params, point, ampl=ampl, sinhw=sinhw)[0, 0]


def _radial_values(params, xx0_scalar, ampl, sinhw):
    config.enable_jax_x64()
    import jax
    import jax.numpy as jnp

    u = _model_scalar(params, xx0_scalar, ampl, sinhw)
    u_xx0 = jax.grad(_model_scalar, argnums=1)(params, xx0_scalar, ampl, sinhw)
    u_xx0xx0 = jax.grad(jax.grad(_model_scalar, argnums=1), argnums=1)(
        params, xx0_scalar, ampl, sinhw
    )
    radius = coordinates.radius_from_xx0(xx0_scalar, ampl=ampl, sinhw=sinhw, xp=jnp)
    u_r, u_rr = coordinates.radial_derivatives_from_xx0(
        xx0_scalar,
        u_xx0,
        u_xx0xx0,
        ampl=ampl,
        sinhw=sinhw,
        xp=jnp,
    )
    return radius, u, u_r, u_rr


def _regularized_residual_scalar(params, xx0_scalar, ampl, sinhw, m):
    config.enable_jax_x64()
    import jax.numpy as jnp

    radius, u, u_r, u_rr = _radial_values(params, xx0_scalar, ampl, sinhw)
    return equation.regularized_residual_from_radial_derivatives(
        radius, u, u_r, u_rr, m, xp=jnp
    )


def regularized_residual(params, xx0, *, ampl: float, sinhw: float, m: float):
    config.enable_jax_x64()
    import jax
    import jax.numpy as jnp

    points = jnp.asarray(xx0, dtype=jnp.float64).reshape(-1)
    return jax.vmap(
        _regularized_residual_scalar,
        in_axes=(None, 0, None, None, None),
    )(params, points, ampl, sinhw, m)


def residual_loss(params, xx0, *, ampl: float, sinhw: float, m: float):
    config.enable_jax_x64()
    import jax.numpy as jnp

    residual = regularized_residual(params, xx0, ampl=ampl, sinhw=sinhw, m=m)
    return jnp.mean(residual * residual)


def origin_regularity_residual(params, *, ampl: float, sinhw: float):
    config.enable_jax_x64()
    import jax.numpy as jnp

    _radius, _u, u_r, _u_rr = _radial_values(
        params, jnp.asarray(0.0, dtype=jnp.float64), ampl, sinhw
    )
    return u_r


def origin_regularity_loss(params, *, ampl: float, sinhw: float):
    residual = origin_regularity_residual(params, ampl=ampl, sinhw=sinhw)
    return residual * residual


def first_order_robin_residual_from_radial_values(radius, u, u_r):
    """Return the scaled first-order Robin residual."""

    return (1.0 + radius) * (radius * u_r + u)


def second_order_robin_residual_from_radial_values(radius, u, u_r, m):
    """Return the second-order Robin residual."""

    return radius**3 * (u_r + u / radius) - 2.0 * m


def outer_boundary_residual_from_radial_values(radius, u, u_r, m, *, condition: str):
    if condition == "first_order_robin":
        return first_order_robin_residual_from_radial_values(radius, u, u_r)
    if condition == "second_order_robin":
        return second_order_robin_residual_from_radial_values(radius, u, u_r, m)
    raise ValueError(f"unknown outer boundary condition: {condition!r}")


def _first_order_robin_residual_scalar(params, xx0_scalar, ampl, sinhw, _m):
    radius, u, u_r, _u_rr = _radial_values(params, xx0_scalar, ampl, sinhw)
    return first_order_robin_residual_from_radial_values(radius, u, u_r)


def _second_order_robin_residual_scalar(params, xx0_scalar, ampl, sinhw, m):
    radius, u, u_r, _u_rr = _radial_values(params, xx0_scalar, ampl, sinhw)
    return second_order_robin_residual_from_radial_values(radius, u, u_r, m)


def outer_boundary_residual(
    params,
    outer_xx0,
    *,
    ampl: float,
    sinhw: float,
    m: float,
    outer_boundary_condition: str,
):
    config.enable_jax_x64()
    import jax
    import jax.numpy as jnp

    points = jnp.asarray(outer_xx0, dtype=jnp.float64).reshape(-1)
    if outer_boundary_condition == "first_order_robin":
        scalar_residual = _first_order_robin_residual_scalar
    elif outer_boundary_condition == "second_order_robin":
        scalar_residual = _second_order_robin_residual_scalar
    else:
        raise ValueError(
            f"unknown outer boundary condition: {outer_boundary_condition!r}"
        )
    return jax.vmap(
        scalar_residual,
        in_axes=(None, 0, None, None, None),
    )(params, points, ampl, sinhw, m)


def outer_boundary_loss(
    params,
    outer_xx0,
    *,
    ampl: float,
    sinhw: float,
    m: float,
    outer_boundary_condition: str,
):
    config.enable_jax_x64()
    import jax.numpy as jnp

    residual = outer_boundary_residual(
        params,
        outer_xx0,
        ampl=ampl,
        sinhw=sinhw,
        m=m,
        outer_boundary_condition=outer_boundary_condition,
    )
    return jnp.mean(residual * residual)


def compute_loss(
    params,
    xx0,
    outer_xx0,
    *,
    ampl: float,
    sinhw: float,
    m: float,
    interior_weight: float = 1.0,
    origin_regularity_weight: float = 1.0,
    outer_boundary_condition: str,
    outer_boundary_weight: float = 1.0,
):
    residual = residual_loss(params, xx0, ampl=ampl, sinhw=sinhw, m=m)
    origin = origin_regularity_loss(params, ampl=ampl, sinhw=sinhw)
    outer = outer_boundary_loss(
        params,
        outer_xx0,
        ampl=ampl,
        sinhw=sinhw,
        m=m,
        outer_boundary_condition=outer_boundary_condition,
    )
    total = (
        interior_weight * residual
        + origin_regularity_weight * origin
        + outer_boundary_weight * outer
    )
    return total, {
        "residual_loss": residual,
        "origin_regularity_loss": origin,
        "outer_boundary_loss": outer,
    }
