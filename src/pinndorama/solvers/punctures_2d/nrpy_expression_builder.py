"""Build native SinhSymTP expressions from vendored NRPy."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import sys
from typing import Iterable

from ..._paths import NRPY_ROOT


def _ensure_vendored_nrpy_on_path() -> None:
    if NRPY_ROOT.is_dir():
        nrpy_path = str(NRPY_ROOT)
        if nrpy_path in sys.path:
            sys.path.remove(nrpy_path)
        sys.path.insert(0, nrpy_path)


_ensure_vendored_nrpy_on_path()

import nrpy  # noqa: E402
import sympy as sp  # noqa: E402

from pathlib import Path

if not Path(nrpy.__file__).resolve().is_relative_to(NRPY_ROOT.resolve()):
    raise ImportError(
        "the publication solver must use the repository's pinned nrpy snapshot; "
        f"imported {nrpy.__file__}"
    )


RESIDUAL_INPUT_NAMES = (
    "xx0",
    "xx1",
    "AMAX",
    "bScale",
    "SINHWAA",
    "uu",
    "uu_dD0",
    "uu_dD1",
    "uu_dDD00",
    "uu_dDD11",
    "psi_background",
    "ADD_times_AUU",
)

GEOMETRY_INPUT_NAMES = (
    "xx0",
    "xx1",
    "AMAX",
    "bScale",
    "SINHWAA",
)

SOURCE_INPUT_NAMES = (
    "xx0",
    "xx1",
    "xx2",
    "AMAX",
    "bScale",
    "SINHWAA",
    "bare_mass_0",
    "bare_mass_1",
    "zPunc",
    "P0_x",
    "P0_y",
    "P0_z",
    "P1_x",
    "P1_y",
    "P1_z",
    "S0_x",
    "S0_y",
    "S0_z",
    "S1_x",
    "S1_y",
    "S1_z",
)

DIRECTION2_DERIVATIVE_SYMBOLS = (
    "uu_dD2",
    "uu_dDD02",
    "uu_dDD12",
    "uu_dDD20",
    "uu_dDD21",
    "uu_dDD22",
)


@dataclass(frozen=True)
class SymbolicExpressions:
    """NRPy-generated symbolic expressions and stable argument ordering."""

    residual_h: sp.Expr
    regularized_residual_h: sp.Expr
    volume_element: sp.Expr
    psi_background: sp.Expr
    add_times_auu: sp.Expr
    residual_symbols: tuple[sp.Symbol, ...]
    geometry_symbols: tuple[sp.Symbol, ...]
    source_symbols: tuple[sp.Symbol, ...]

    @property
    def residual_free_symbol_names(self) -> tuple[str, ...]:
        return tuple(sorted(str(symbol) for symbol in self.residual_h.free_symbols))

    @property
    def regularized_residual_free_symbol_names(self) -> tuple[str, ...]:
        return tuple(
            sorted(str(symbol) for symbol in self.regularized_residual_h.free_symbols)
        )


def _symbols_by_name(names: Iterable[str]) -> tuple[sp.Symbol, ...]:
    return tuple(sp.Symbol(name, real=True) for name in names)


@lru_cache(maxsize=1)
def build_symbolic_expressions() -> SymbolicExpressions:
    """Return NRPy expressions for the native 2D SinhSymTP problem.

    The important detail is ``indexedexp::symmetry_axes = "2"``. NRPy then
    declares derivatives across direction 2 as zero, so the residual expression
    is genuinely 2D without manual post-hoc substitutions.
    """

    import nrpy.indexedexp  # noqa: F401
    import nrpy.params as par
    import nrpy.reference_metric as refmetric
    from nrpy.equations.nrpyelliptic.ConformallyFlat_RHSs import (
        HyperbolicRelaxationCurvilinearRHSs,
    )
    from nrpy.equations.nrpyelliptic.ConformallyFlat_SourceTerms import (
        compute_psi_background_and_ADD_times_AUU,
    )

    par.set_parval_from_str("Infrastructure", "BHaH")
    par.set_parval_from_str("indexedexp::symmetry_axes", "2")

    rhs = HyperbolicRelaxationCurvilinearRHSs("SinhSymTP", False)
    rfm = refmetric.reference_metric["SinhSymTP"]
    psi_background, add_times_auu = compute_psi_background_and_ADD_times_AUU(
        "SinhSymTP"
    )

    residual_symbols = _symbols_by_name(RESIDUAL_INPUT_NAMES)
    geometry_symbols = _symbols_by_name(GEOMETRY_INPUT_NAMES)
    source_symbols = _symbols_by_name(SOURCE_INPUT_NAMES)
    residual_symbol_by_name = {str(symbol): symbol for symbol in residual_symbols}

    xx0 = rfm.xx[0]
    xx1 = rfm.xx[1]
    uu = residual_symbol_by_name["uu"]
    uu_dD0 = residual_symbol_by_name["uu_dD0"]
    uu_dD1 = residual_symbol_by_name["uu_dD1"]
    uu_dDD00 = residual_symbol_by_name["uu_dDD00"]
    uu_dDD11 = residual_symbol_by_name["uu_dDD11"]
    psi_background_symbol = residual_symbol_by_name["psi_background"]
    add_times_auu_symbol = residual_symbol_by_name["ADD_times_AUU"]

    # For SinhSymTP, the reference metric is orthogonal and the axisymmetric
    # reduction removes all direction-2 derivatives. Use divergence form for
    # J * residual so coordinate-singular denominators are avoided algebraically.
    volume_element = sp.prod(rfm.scalefactor_orthog)
    source_term = (
        sp.Rational(1, 8)
        * add_times_auu_symbol
        * sp.Pow(psi_background_symbol + uu, -7)
    )
    regularized_residual_h = (
        sp.diff(volume_element * rfm.ghatUU[0][0], xx0) * uu_dD0
        + volume_element * rfm.ghatUU[0][0] * uu_dDD00
        + sp.diff(volume_element * rfm.ghatUU[1][1], xx1) * uu_dD1
        + volume_element * rfm.ghatUU[1][1] * uu_dDD11
        + volume_element * source_term
    )

    return SymbolicExpressions(
        residual_h=rhs.residual,
        regularized_residual_h=regularized_residual_h,
        volume_element=volume_element,
        psi_background=psi_background,
        add_times_auu=add_times_auu,
        residual_symbols=residual_symbols,
        geometry_symbols=geometry_symbols,
        source_symbols=source_symbols,
    )


def residual_uses_no_direction2_derivatives() -> bool:
    """Return True if NRPy symmetry axes removed direction-2 derivatives."""

    residual_text = str(build_symbolic_expressions().residual_h)
    return not any(name in residual_text for name in DIRECTION2_DERIVATIVE_SYMBOLS)


def regularized_residual_uses_no_direction2_derivatives() -> bool:
    """Return True if the regularized residual has no direction-2 derivatives."""

    residual_text = str(build_symbolic_expressions().regularized_residual_h)
    return not any(name in residual_text for name in DIRECTION2_DERIVATIVE_SYMBOLS)
