"""Native SinhSymTP coordinate map and publication collocation rules."""

from __future__ import annotations

from typing import Any

import numpy as np

from .config import Collocation, Geometry


def aa_from_xx0(xx0: Any, geometry: Geometry, *, xp=np):
    return (
        geometry.AMAX
        * xp.sinh(xx0 / geometry.SINHWAA)
        / xp.sinh(1.0 / geometry.SINHWAA)
    )


def daa_dxx0(xx0: Any, geometry: Geometry, *, xp=np):
    return (
        geometry.AMAX
        * xp.cosh(xx0 / geometry.SINHWAA)
        / (geometry.SINHWAA * xp.sinh(1.0 / geometry.SINHWAA))
    )


def reference_metric_scale_factors(
    xx0: Any,
    xx1: Any,
    geometry: Geometry,
    *,
    xp=np,
) -> tuple[Any, Any, Any]:
    """Return the three orthogonal SinhSymTP reference-metric scale factors."""

    aa = aa_from_xx0(xx0, geometry, xp=xp)
    aa_d0 = daa_dxx0(xx0, geometry, xp=xp)
    var1 = xp.sqrt(aa * aa + geometry.bScale**2 * xp.sin(xx1) ** 2)
    var2 = xp.sqrt(aa * aa + geometry.bScale**2)
    return aa_d0 * var1 / var2, var1, aa * xp.sin(xx1)


def volume_element_from_xx(xx0: Any, xx1: Any, geometry: Geometry, *, xp=np):
    h0, h1, h2 = reference_metric_scale_factors(xx0, xx1, geometry, xp=xp)
    return h0 * h1 * h2


def spherical_radius_from_xx(xx0: Any, xx1: Any, geometry: Geometry, *, xp=np):
    aa = aa_from_xx0(xx0, geometry, xp=xp)
    return xp.sqrt(aa * aa + geometry.bScale**2 * xp.cos(xx1) ** 2)


def spherical_jacobian_from_xx(
    xx0: Any,
    xx1: Any,
    geometry: Geometry,
    *,
    xp=np,
) -> tuple[Any, Any, Any, Any]:
    """Return derivatives of spherical ``(r, theta)`` with respect to ``xx``."""

    aa = aa_from_xx0(xx0, geometry, xp=xp)
    aa_d0 = daa_dxx0(xx0, geometry, xp=xp)
    cos1 = xp.cos(xx1)
    sin1 = xp.sin(xx1)
    radius = xp.sqrt(aa * aa + geometry.bScale**2 * cos1 * cos1)
    sph_radius_of_aa = xp.sqrt(aa * aa + geometry.bScale**2)

    dr_dxx0 = aa * aa_d0 / radius
    dr_dxx1 = -(geometry.bScale**2) * cos1 * sin1 / radius
    d_sph_radius_dxx0 = aa * aa_d0 / sph_radius_of_aa
    theta_arg = sph_radius_of_aa * cos1 / radius
    dtheta_arg_dxx0 = (
        cos1
        * (d_sph_radius_dxx0 * radius - sph_radius_of_aa * dr_dxx0)
        / (radius * radius)
    )
    dtheta_arg_dxx1 = (
        -sph_radius_of_aa * sin1 * radius - sph_radius_of_aa * cos1 * dr_dxx1
    ) / (radius * radius)
    sin_theta = xp.sqrt(xp.maximum(1.0 - theta_arg * theta_arg, 1.0e-300))
    return (
        dr_dxx0,
        dr_dxx1,
        -dtheta_arg_dxx0 / sin_theta,
        -dtheta_arg_dxx1 / sin_theta,
    )


def spherical_radial_derivative_from_xx_gradient(
    xx0: Any,
    xx1: Any,
    grad_xx0: Any,
    grad_xx1: Any,
    geometry: Geometry,
    *,
    xp=np,
):
    dr_dxx0, dr_dxx1, dtheta_dxx0, dtheta_dxx1 = spherical_jacobian_from_xx(
        xx0, xx1, geometry, xp=xp
    )
    determinant = dr_dxx0 * dtheta_dxx1 - dtheta_dxx0 * dr_dxx1
    return (grad_xx0 * dtheta_dxx1 - dtheta_dxx0 * grad_xx1) / determinant


def xx_to_cart(
    xx0: Any,
    xx1: Any,
    xx2: Any,
    geometry: Geometry,
    *,
    xp=np,
) -> tuple[Any, Any, Any]:
    aa = aa_from_xx0(xx0, geometry, xp=xp)
    return (
        aa * xp.sin(xx1) * xp.cos(xx2),
        aa * xp.sin(xx1) * xp.sin(xx2),
        xp.sqrt(aa * aa + geometry.bScale**2) * xp.cos(xx1),
    )


def _cell_centers(xp, start: float, stop: float, num: int):
    if num <= 0:
        raise ValueError("number of cell centers must be positive")
    spacing = (stop - start) / num
    return xp.linspace(start, stop, num, endpoint=False) + 0.5 * spacing


def xx0_collocation_points(collocation: Collocation, *, xp=np):
    """Return uniform or explicit two-zone cell-centered xx0 points."""

    if collocation.xx0_sampling == "uniform":
        return _cell_centers(
            xp,
            collocation.xx0_min,
            collocation.xx0_max,
            collocation.Nxx0,
        )
    if collocation.xx0_sampling != "two-zone":  # protected by TOML validation
        raise ValueError(f"unsupported xx0 sampling: {collocation.xx0_sampling}")
    assert collocation.xx0_cut is not None
    assert collocation.Nxx0_inner is not None
    inner = _cell_centers(
        xp,
        collocation.xx0_min,
        collocation.xx0_cut,
        collocation.Nxx0_inner,
    )
    outer = _cell_centers(
        xp,
        collocation.xx0_cut,
        collocation.xx0_max,
        collocation.Nxx0 - collocation.Nxx0_inner,
    )
    return xp.concatenate((inner, outer))


def native_collocation_grid(
    collocation: Collocation,
    *,
    xp=np,
) -> tuple[Any, Any, Any]:
    """Return the NRPy-style cell-centered tensor-product residual grid."""

    xx0 = xx0_collocation_points(collocation, xp=xp)
    xx1 = _cell_centers(
        xp,
        collocation.xx1_min,
        collocation.xx1_max,
        collocation.Nxx1,
    )
    xx2 = _cell_centers(
        xp,
        collocation.xx2_min,
        collocation.xx2_max,
        collocation.Nxx2,
    )
    grid = xp.meshgrid(xx0, xx1, xx2, indexing="ij")
    return tuple(value.reshape(-1) for value in grid)


def theta_axis_regularization_samples(
    collocation: Collocation,
    *,
    xp=np,
) -> tuple[Any, Any, Any]:
    xx0 = xp.linspace(
        collocation.boundary_xx0_min,
        collocation.xx0_max,
        collocation.Nxx0,
    )
    delta = xp.linspace(
        collocation.xx1_axis_delta_min,
        collocation.xx1_axis_delta_max,
        collocation.Nxx1,
    )
    phi = xp.linspace(
        collocation.xx2_min,
        collocation.xx2_max,
        collocation.Nxx2,
        endpoint=False,
    )
    grid = xp.meshgrid(xx0, delta, phi, indexing="ij")
    return tuple(value.reshape(-1) for value in grid)


def phi_periodicity_samples(
    collocation: Collocation,
    *,
    xp=np,
) -> tuple[Any, Any]:
    xx0 = xp.linspace(
        collocation.boundary_xx0_min,
        collocation.xx0_max,
        collocation.Nxx0,
    )
    xx1 = xp.linspace(
        collocation.boundary_xx1_epsilon,
        np.pi - collocation.boundary_xx1_epsilon,
        collocation.Nxx1,
    )
    grid = xp.meshgrid(xx0, xx1, indexing="ij")
    return tuple(value.reshape(-1) for value in grid)


def outer_robin_boundary_samples(
    collocation: Collocation,
    *,
    xp=np,
) -> tuple[Any, Any, Any]:
    xx1 = xp.linspace(
        collocation.boundary_xx1_epsilon,
        np.pi - collocation.boundary_xx1_epsilon,
        collocation.Nxx1,
    )
    xx2 = xp.linspace(
        collocation.xx2_min,
        collocation.xx2_max,
        collocation.Nxx2,
        endpoint=False,
    )
    xx1_grid, xx2_grid = xp.meshgrid(xx1, xx2, indexing="ij")
    xx0_grid = xp.zeros_like(xx1_grid) + collocation.xx0_max
    return xx0_grid.reshape(-1), xx1_grid.reshape(-1), xx2_grid.reshape(-1)
