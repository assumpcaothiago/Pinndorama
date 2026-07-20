"""Manufactured nonlinear spherical elliptic equation."""

from __future__ import annotations

from typing import Any

import numpy as np


def exact_solution(r: Any, m: float, *, xp=np):
    r = xp.asarray(r)
    return 2.0 * m * (1.0 + r) / (r * r + 2.0 * r + 2.0)


def exact_solution_dr(r: Any, m: float, *, xp=np):
    r = xp.asarray(r)
    denominator = r * r + 2.0 * r + 2.0
    return -2.0 * m * r * (r + 2.0) / (denominator * denominator)


def exact_solution_drr(r: Any, m: float, *, xp=np):
    r = xp.asarray(r)
    denominator = r * r + 2.0 * r + 2.0
    return 4.0 * m * (r + 1.0) * (r * r + 2.0 * r - 2.0) / denominator**3


def exact_laplacian(r: Any, m: float, *, xp=np):
    r = xp.asarray(r)
    denominator = r * r + 2.0 * r + 2.0
    return -4.0 * m * (r * r + 6.0 * r + 6.0) / denominator**3


def psi_background(r: Any, m: float, *, xp=np):
    """Return ``1 + m/(2r)``; callers must exclude the origin."""

    r = xp.asarray(r)
    return 1.0 + m / (2.0 * r)


def source_A(r: Any, m: float, *, xp=np):
    """Return the explicit singular source ``A(r)`` away from the origin."""

    r = xp.asarray(r)
    d = r * r + 2.0 * r + 2.0
    q = r * r + 6.0 * r + 6.0
    b = 2.0 * r * d + m * (5.0 * r * r + 6.0 * r + 2.0)
    return m * q * b**7 / (4.0 * r**7 * d**10)


def nonlinear_term(r: Any, u: Any, m: float, *, xp=np):
    """Evaluate ``A (psi + u)^-7 / 8`` in a finite, equivalent form."""

    r = xp.asarray(r)
    u = xp.asarray(u)
    d = r * r + 2.0 * r + 2.0
    q = r * r + 6.0 * r + 6.0
    b = 2.0 * r * d + m * (5.0 * r * r + 6.0 * r + 2.0)
    denominator = d**10 * (m + 2.0 * r * (1.0 + u)) ** 7
    return 4.0 * m * q * b**7 / denominator


def regularized_residual_from_radial_derivatives(
    r: Any, u: Any, u_r: Any, u_rr: Any, m: float, *, xp=np
):
    """Return ``r R = r u_rr + 2 u_r + r N(r,u,m)``."""

    return r * u_rr + 2.0 * u_r + r * nonlinear_term(r, u, m, xp=xp)
