"""Build the publication divergence-form ``J R_H`` expression from NRPy."""

from __future__ import annotations

from dataclasses import dataclass
from functools import lru_cache
import sys
from typing import Iterable

from ..._paths import NRPY_ROOT


def _ensure_vendored_nrpy_on_path() -> None:
    if NRPY_ROOT.is_dir() and str(NRPY_ROOT) not in sys.path:
        sys.path.insert(0, str(NRPY_ROOT))


_ensure_vendored_nrpy_on_path()

import sympy as sp  # noqa: E402

RESIDUAL_INPUT_NAMES = (
    "xx0",
    "xx1",
    "xx2",
    "AMAX",
    "bScale",
    "SINHWAA",
    "uu",
    "uu_dD0",
    "uu_dD1",
    "uu_dD2",
    "uu_dDD00",
    "uu_dDD11",
    "uu_dDD22",
    "psi_background",
    "ADD_times_AUU",
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


@dataclass(frozen=True)
class SymbolicExpressions:
    regularized_residual_h: sp.Expr
    volume_element: sp.Expr
    psi_background: sp.Expr
    add_times_auu: sp.Expr
    residual_symbols: tuple[sp.Symbol, ...]
    source_symbols: tuple[sp.Symbol, ...]


def _symbols(names: Iterable[str]) -> tuple[sp.Symbol, ...]:
    return tuple(sp.Symbol(name, real=True) for name in names)


@lru_cache(maxsize=1)
def build_symbolic_expressions() -> SymbolicExpressions:
    """Return the fixed native-coordinate residual and z-separated source model."""

    import nrpy.indexedexp  # noqa: F401
    import nrpy.params as par
    import nrpy.reference_metric as refmetric
    from nrpy.equations.nrpyelliptic.ConformallyFlat_SourceTerms import (
        compute_psi_background_and_ADD_times_AUU,
    )

    par.set_parval_from_str("Infrastructure", "BHaH")
    par.set_parval_from_str("indexedexp::symmetry_axes", "")
    rfm = refmetric.reference_metric["SinhSymTP"]
    psi_background, add_times_auu = compute_psi_background_and_ADD_times_AUU(
        "SinhSymTP"
    )

    residual_symbols = _symbols(RESIDUAL_INPUT_NAMES)
    source_symbols = _symbols(SOURCE_INPUT_NAMES)
    by_name = {str(symbol): symbol for symbol in residual_symbols}
    xx0, xx1, xx2 = rfm.xx
    uu = by_name["uu"]
    uu_dD = (by_name["uu_dD0"], by_name["uu_dD1"], by_name["uu_dD2"])
    uu_dDD = (
        by_name["uu_dDD00"],
        by_name["uu_dDD11"],
        by_name["uu_dDD22"],
    )

    # For the orthogonal reference metric,
    #   J R_H = sum_i [d_i(J g^ii) d_i u + J g^ii d_ii u]
    #           + J (ADD_times_AUU / 8) (psi_background + u)^(-7).
    # Evaluating this divergence form avoids the coordinate-singular denominators
    # present in the raw Hamiltonian residual.
    volume_element = sp.prod(rfm.scalefactor_orthog)
    coordinates = (xx0, xx1, xx2)
    flux_factors = tuple(
        volume_element * rfm.ghatUU[index][index] for index in range(3)
    )
    divergence_terms = []
    for coordinate, factor, first, second in zip(
        coordinates, flux_factors, uu_dD, uu_dDD
    ):
        divergence_terms.extend((sp.diff(factor, coordinate) * first, factor * second))
    source_term = (
        volume_element
        * sp.Rational(1, 8)
        * by_name["ADD_times_AUU"]
        * sp.Pow(by_name["psi_background"] + uu, -7)
    )

    return SymbolicExpressions(
        regularized_residual_h=sum(divergence_terms) + source_term,
        volume_element=volume_element,
        psi_background=psi_background,
        add_times_auu=add_times_auu,
        residual_symbols=residual_symbols,
        source_symbols=source_symbols,
    )
