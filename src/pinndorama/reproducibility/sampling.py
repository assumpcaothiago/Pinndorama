"""Pure-NumPy sampling and SinhSymTP coordinate conventions."""

from __future__ import annotations

import numpy as np


def cell_centers(start: float, stop: float, count: int) -> np.ndarray:
    """Return ``count`` equal-width cell centers over ``[start, stop]``."""

    start = float(start)
    stop = float(stop)
    count = int(count)
    if count <= 0:
        raise ValueError("count must be positive")
    if not np.isfinite(start) or not np.isfinite(stop) or start >= stop:
        raise ValueError("cell-center interval must be finite and increasing")
    spacing = (stop - start) / count
    return start + (np.arange(count, dtype=np.float64) + 0.5) * spacing


def two_zone_cell_centers(
    start: float,
    cut: float,
    stop: float,
    total_count: int,
    inner_count: int,
) -> np.ndarray:
    """Return contiguous cell centers for two independently uniform zones."""

    total_count = int(total_count)
    inner_count = int(inner_count)
    outer_count = total_count - inner_count
    if not float(start) < float(cut) < float(stop):
        raise ValueError("two-zone bounds must satisfy start < cut < stop")
    if total_count < 2 or inner_count <= 0 or outer_count <= 0:
        raise ValueError("both radial zones must contain at least one point")
    return np.concatenate(
        (
            cell_centers(start, cut, inner_count),
            cell_centers(cut, stop, outer_count),
        )
    )


def sinhsymtp_radius(
    xx0: np.ndarray | float,
    xx1: np.ndarray | float,
    *,
    amax: float,
    bscale: float,
    sinhwaa: float,
) -> np.ndarray:
    """Return the Cartesian radius associated with native SinhSymTP points."""

    xx0 = np.asarray(xx0, dtype=np.float64)
    xx1 = np.asarray(xx1, dtype=np.float64)
    radial = float(amax) * np.sinh(xx0 / float(sinhwaa)) / np.sinh(1.0 / float(sinhwaa))
    return np.sqrt(
        radial * radial + float(bscale) * float(bscale) * np.square(np.cos(xx1))
    )


def sinhsymtp_to_cartesian(
    xx0: np.ndarray | float,
    xx1: np.ndarray | float,
    xx2: np.ndarray | float,
    *,
    amax: float,
    bscale: float,
    sinhwaa: float,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """Map native SinhSymTP coordinates to Cartesian coordinates."""

    xx0, xx1, xx2 = np.broadcast_arrays(
        np.asarray(xx0, dtype=np.float64),
        np.asarray(xx1, dtype=np.float64),
        np.asarray(xx2, dtype=np.float64),
    )
    radial = float(amax) * np.sinh(xx0 / float(sinhwaa)) / np.sinh(1.0 / float(sinhwaa))
    x = radial * np.sin(xx1) * np.cos(xx2)
    y = radial * np.sin(xx1) * np.sin(xx2)
    z = np.sqrt(radial * radial + float(bscale) ** 2) * np.cos(xx1)
    return x, y, z


def sinhsymtp_volume_element(
    xx0: np.ndarray | float,
    xx1: np.ndarray | float,
    *,
    amax: float,
    bscale: float,
    sinhwaa: float,
) -> np.ndarray:
    """Return ``sqrt(det(gamma_hat))`` for the native SinhSymTP map."""

    xx0, xx1 = np.broadcast_arrays(
        np.asarray(xx0, dtype=np.float64),
        np.asarray(xx1, dtype=np.float64),
    )
    denominator = np.sinh(1.0 / float(sinhwaa))
    radial = float(amax) * np.sinh(xx0 / float(sinhwaa)) / denominator
    radial_derivative = (
        float(amax) * np.cosh(xx0 / float(sinhwaa)) / (float(sinhwaa) * denominator)
    )
    sin_theta = np.sin(xx1)
    bscale_squared = float(bscale) ** 2
    return (
        radial_derivative
        * radial
        * sin_theta
        * (radial * radial + bscale_squared * sin_theta * sin_theta)
        / np.sqrt(radial * radial + bscale_squared)
    )
