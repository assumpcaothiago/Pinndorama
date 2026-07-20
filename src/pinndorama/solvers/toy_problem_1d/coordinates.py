"""NRPy SinhSpherical radial map and cell-centered sampling."""

from __future__ import annotations

from typing import Any

import numpy as np


def radius_from_xx0(xx0: Any, *, ampl: float, sinhw: float, xp=np):
    """Map native ``xx0`` to the physical spherical radius."""

    return ampl * xp.sinh(xx0 / sinhw) / xp.sinh(1.0 / sinhw)


def dr_dxx0(xx0: Any, *, ampl: float, sinhw: float, xp=np):
    return ampl * xp.cosh(xx0 / sinhw) / (sinhw * xp.sinh(1.0 / sinhw))


def d2r_dxx02(xx0: Any, *, ampl: float, sinhw: float, xp=np):
    return ampl * xp.sinh(xx0 / sinhw) / (sinhw * sinhw * xp.sinh(1.0 / sinhw))


def radial_derivatives_from_xx0(
    xx0: Any,
    u_xx0: Any,
    u_xx0xx0: Any,
    *,
    ampl: float,
    sinhw: float,
    xp=np,
) -> tuple[Any, Any]:
    """Transform first and second native derivatives to physical radius."""

    first_map = dr_dxx0(xx0, ampl=ampl, sinhw=sinhw, xp=xp)
    second_map = d2r_dxx02(xx0, ampl=ampl, sinhw=sinhw, xp=xp)
    u_r = u_xx0 / first_map
    u_rr = u_xx0xx0 / (first_map * first_map) - (
        u_xx0 * second_map / (first_map * first_map * first_map)
    )
    return u_r, u_rr


def cell_centered_xx0(*, Nxx0: int, xx0_min: float = 0.0, xx0_max: float = 1.0, xp=np):
    """Return ``Nxx0`` native-coordinate cell centers."""

    if Nxx0 < 1:
        raise ValueError("Nxx0 must be positive")
    if not xx0_min < xx0_max:
        raise ValueError("xx0_min must be smaller than xx0_max")
    spacing = (xx0_max - xx0_min) / Nxx0
    return xx0_min + (xp.arange(Nxx0) + 0.5) * spacing


def outer_boundary_xx0(
    interior_xx0: Any,
    *,
    region: str,
    ampl: float,
    sinhw: float,
    r_min: float | None = None,
    xp=np,
):
    """Return the native-coordinate samples for the selected boundary region."""

    points = xp.asarray(interior_xx0).reshape(-1)
    endpoint = xp.asarray([1.0], dtype=points.dtype)
    if region == "endpoint":
        if r_min is not None:
            raise ValueError("r_min is not used for the endpoint boundary region")
        return endpoint
    if region != "radial_band":
        raise ValueError(f"unknown outer boundary region: {region!r}")
    if r_min is None:
        raise ValueError("r_min is required for the radial_band boundary region")
    radius = radius_from_xx0(points, ampl=ampl, sinhw=sinhw, xp=xp)
    selected = points[radius > r_min]
    return xp.concatenate((selected, endpoint))
