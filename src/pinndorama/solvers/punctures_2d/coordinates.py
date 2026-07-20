"""SinhSymTP coordinate helpers in native numerical coordinates."""

from __future__ import annotations

from typing import Any

import numpy as np

from . import config


def aa_from_xx0(
    xx0: Any,
    *,
    amax: float = config.AMAX,
    sinhwaa: float = config.SINHWAA,
    xp=np,
):
    """Return the SinhSymTP radial-like coordinate ``AA(xx0)``."""

    return amax * xp.sinh(xx0 / sinhwaa) / xp.sinh(1.0 / sinhwaa)


def daa_dxx0(
    xx0: Any,
    *,
    amax: float = config.AMAX,
    sinhwaa: float = config.SINHWAA,
    xp=np,
):
    """Return ``dAA/dxx0`` for the SinhSymTP radial-like coordinate."""

    return amax * xp.cosh(xx0 / sinhwaa) / (sinhwaa * xp.sinh(1.0 / sinhwaa))


def reference_metric_scale_factors(
    xx0: Any,
    xx1: Any,
    *,
    amax: float = config.AMAX,
    bscale: float = config.bScale,
    sinhwaa: float = config.SINHWAA,
    xp=np,
) -> tuple[Any, Any, Any]:
    """Return the SinhSymTP orthogonal reference-metric scale factors."""

    aa = aa_from_xx0(xx0, amax=amax, sinhwaa=sinhwaa, xp=xp)
    aa_d0 = daa_dxx0(xx0, amax=amax, sinhwaa=sinhwaa, xp=xp)
    var1 = xp.sqrt(aa * aa + bscale * bscale * xp.sin(xx1) ** 2)
    var2 = xp.sqrt(aa * aa + bscale * bscale)
    h0 = aa_d0 * var1 / var2
    h1 = var1
    h2 = aa * xp.sin(xx1)
    return h0, h1, h2


def volume_element_from_xx(
    xx0: Any,
    xx1: Any,
    *,
    amax: float = config.AMAX,
    bscale: float = config.bScale,
    sinhwaa: float = config.SINHWAA,
    xp=np,
):
    """Return ``sqrt(det(gammahat)) = h0 h1 h2`` for SinhSymTP."""

    h0, h1, h2 = reference_metric_scale_factors(
        xx0,
        xx1,
        amax=amax,
        bscale=bscale,
        sinhwaa=sinhwaa,
        xp=xp,
    )
    return h0 * h1 * h2


def spherical_radius_from_xx(
    xx0: Any,
    xx1: Any,
    *,
    amax: float = config.AMAX,
    bscale: float = config.bScale,
    sinhwaa: float = config.SINHWAA,
    xp=np,
):
    """Return spherical radius corresponding to SinhSymTP ``(xx0, xx1)``."""

    aa = aa_from_xx0(xx0, amax=amax, sinhwaa=sinhwaa, xp=xp)
    return xp.sqrt(aa * aa + bscale * bscale * xp.cos(xx1) ** 2)


def spherical_theta_from_xx(
    xx0: Any,
    xx1: Any,
    *,
    amax: float = config.AMAX,
    bscale: float = config.bScale,
    sinhwaa: float = config.SINHWAA,
    xp=np,
):
    """Return spherical polar angle corresponding to SinhSymTP ``(xx0, xx1)``."""

    aa = aa_from_xx0(xx0, amax=amax, sinhwaa=sinhwaa, xp=xp)
    r_sph = spherical_radius_from_xx(
        xx0,
        xx1,
        amax=amax,
        bscale=bscale,
        sinhwaa=sinhwaa,
        xp=xp,
    )
    theta_arg = xp.sqrt(aa * aa + bscale * bscale) * xp.cos(xx1) / r_sph
    return xp.arccos(xp.clip(theta_arg, -1.0, 1.0))


def xx_to_spherical(
    xx0: Any,
    xx1: Any,
    *,
    amax: float = config.AMAX,
    bscale: float = config.bScale,
    sinhwaa: float = config.SINHWAA,
    xp=np,
) -> tuple[Any, Any]:
    """Map SinhSymTP ``(xx0, xx1)`` to spherical ``(r_sph, theta_sph)``."""

    return (
        spherical_radius_from_xx(
            xx0,
            xx1,
            amax=amax,
            bscale=bscale,
            sinhwaa=sinhwaa,
            xp=xp,
        ),
        spherical_theta_from_xx(
            xx0,
            xx1,
            amax=amax,
            bscale=bscale,
            sinhwaa=sinhwaa,
            xp=xp,
        ),
    )


def spherical_jacobian_from_xx(
    xx0: Any,
    xx1: Any,
    *,
    amax: float = config.AMAX,
    bscale: float = config.bScale,
    sinhwaa: float = config.SINHWAA,
    xp=np,
) -> tuple[Any, Any, Any, Any]:
    """Return first derivatives of ``(r_sph, theta_sph)`` with respect to ``xx``."""

    aa = aa_from_xx0(xx0, amax=amax, sinhwaa=sinhwaa, xp=xp)
    aa_d0 = daa_dxx0(xx0, amax=amax, sinhwaa=sinhwaa, xp=xp)
    cos1 = xp.cos(xx1)
    sin1 = xp.sin(xx1)
    r_sph = xp.sqrt(aa * aa + bscale * bscale * cos1 * cos1)
    sph_radius_of_aa = xp.sqrt(aa * aa + bscale * bscale)

    dr_dxx0 = aa * aa_d0 / r_sph
    dr_dxx1 = -(bscale * bscale) * cos1 * sin1 / r_sph

    d_sph_radius_of_aa_dxx0 = aa * aa_d0 / sph_radius_of_aa
    theta_arg = sph_radius_of_aa * cos1 / r_sph
    dtheta_arg_dxx0 = (
        cos1
        * (d_sph_radius_of_aa_dxx0 * r_sph - sph_radius_of_aa * dr_dxx0)
        / (r_sph * r_sph)
    )
    dtheta_arg_dxx1 = (
        -sph_radius_of_aa * sin1 * r_sph - sph_radius_of_aa * cos1 * dr_dxx1
    ) / (r_sph * r_sph)

    sin_theta = xp.sqrt(xp.maximum(1.0 - theta_arg * theta_arg, 1.0e-300))
    dtheta_dxx0 = -dtheta_arg_dxx0 / sin_theta
    dtheta_dxx1 = -dtheta_arg_dxx1 / sin_theta
    return dr_dxx0, dr_dxx1, dtheta_dxx0, dtheta_dxx1


def spherical_radial_derivative_from_xx_gradient(
    xx0: Any,
    xx1: Any,
    grad_xx0: Any,
    grad_xx1: Any,
    *,
    amax: float = config.AMAX,
    bscale: float = config.bScale,
    sinhwaa: float = config.SINHWAA,
    xp=np,
):
    """Transform ``xx`` first derivatives to ``du/dr_sph`` at fixed spherical angle."""

    dr_dxx0, dr_dxx1, dtheta_dxx0, dtheta_dxx1 = spherical_jacobian_from_xx(
        xx0,
        xx1,
        amax=amax,
        bscale=bscale,
        sinhwaa=sinhwaa,
        xp=xp,
    )
    determinant = dr_dxx0 * dtheta_dxx1 - dtheta_dxx0 * dr_dxx1
    return (grad_xx0 * dtheta_dxx1 - dtheta_dxx0 * grad_xx1) / determinant


def xx_to_cart(
    xx0: Any,
    xx1: Any,
    xx2: Any = 0.0,
    *,
    amax: float = config.AMAX,
    bscale: float = config.bScale,
    sinhwaa: float = config.SINHWAA,
    xp=np,
) -> tuple[Any, Any, Any]:
    """Map SinhSymTP ``(xx0, xx1, xx2)`` to Cartesian coordinates."""

    aa = aa_from_xx0(xx0, amax=amax, sinhwaa=sinhwaa, xp=xp)
    x_cart = aa * xp.sin(xx1) * xp.cos(xx2)
    y_cart = aa * xp.sin(xx1) * xp.sin(xx2)
    z_cart = xp.sqrt(aa * aa + bscale * bscale) * xp.cos(xx1)
    return x_cart, y_cart, z_cart


def cart_to_xx(
    x_cart: Any,
    y_cart: Any,
    z_cart: Any,
    *,
    amax: float = config.AMAX,
    bscale: float = config.bScale,
    sinhwaa: float = config.SINHWAA,
    xp=np,
) -> tuple[Any, Any, Any]:
    """Map Cartesian coordinates to SinhSymTP ``(xx0, xx1, xx2)``."""

    cylindrical_radius_squared = x_cart * x_cart + y_cart * y_cart
    tmp3 = (
        z_cart * z_cart
        + cylindrical_radius_squared
        + xp.sqrt(
            (cylindrical_radius_squared + (-z_cart + bscale) ** 2)
            * (cylindrical_radius_squared + (z_cart + bscale) ** 2)
        )
    )
    aa_argument = xp.sqrt(xp.maximum(tmp3 - bscale * bscale, 0.0))
    xx0 = sinhwaa * xp.arcsinh(
        aa_argument * xp.sinh(1.0 / sinhwaa) / (xp.sqrt(2.0) * amax)
    )
    xx1_arg = xp.sqrt(2.0) * z_cart / xp.sqrt(bscale * bscale + tmp3)
    xx1 = xp.arccos(xp.clip(xx1_arg, -1.0, 1.0))
    xx2 = xp.arctan2(y_cart, x_cart)
    return xx0, xx1, xx2


def native_collocation_grid(
    *,
    Nxx0: int = config.Nxx0,
    Nxx1: int = config.Nxx1,
    xx0_min: float = config.xx0_min,
    xx0_max: float = config.xx0_max,
    xx1_min: float = config.xx1_min,
    xx1_max: float = config.xx1_max,
    xp=np,
) -> tuple[Any, Any]:
    """Return a flattened cell-centered native-coordinate grid."""

    if Nxx0 < 1 or Nxx1 < 1:
        raise ValueError("collocation dimensions must be positive")
    xx0 = xx0_min + (xp.arange(Nxx0) + 0.5) * ((xx0_max - xx0_min) / Nxx0)
    xx1 = xx1_min + (xp.arange(Nxx1) + 0.5) * ((xx1_max - xx1_min) / Nxx1)
    xx0_grid, xx1_grid = xp.meshgrid(xx0, xx1, indexing="ij")
    return xx0_grid.reshape(-1), xx1_grid.reshape(-1)


def theta_inner_parity_samples(
    *,
    Nxx0: int = config.Nparity,
    Ndelta: int = config.Nparity,
    xx0_min: float = config.xx0_min,
    xx0_max: float = config.xx0_max,
    delta_min: float = config.xx1_parity_delta_min,
    delta_max: float = config.xx1_parity_delta_max,
    xp=np,
) -> tuple[Any, Any]:
    """Return flattened ``(xx0, delta)`` samples near theta boundaries."""

    if Nxx0 < 1 or Ndelta < 1:
        raise ValueError("parity sample dimensions must be positive")
    xx0 = xx0_min + (xp.arange(Nxx0) + 0.5) * ((xx0_max - xx0_min) / Nxx0)
    delta = delta_min + (xp.arange(Ndelta) + 0.5) * ((delta_max - delta_min) / Ndelta)
    xx0_grid, delta_grid = xp.meshgrid(xx0, delta, indexing="ij")
    return xx0_grid.reshape(-1), delta_grid.reshape(-1)


def outer_robin_boundary_samples(
    *,
    Nxx1: int = config.Nxx1_outer,
    xx0_value: float = config.xx0_max,
    xx1_min: float = config.xx1_min,
    xx1_max: float = config.xx1_max,
    xp=np,
) -> tuple[Any, Any]:
    """Return flattened samples on the outer ``xx0`` boundary."""

    if Nxx1 < 1:
        raise ValueError("outer-boundary sample count must be positive")
    xx1 = xx1_min + (xp.arange(Nxx1) + 0.5) * ((xx1_max - xx1_min) / Nxx1)
    xx0 = xp.zeros_like(xx1) + xx0_value
    return xx0.reshape(-1), xx1.reshape(-1)
