"""Canonical paths to repository resources and pinned dependencies."""

from __future__ import annotations

from pathlib import Path

PACKAGE_ROOT = Path(__file__).resolve().parent
REPO_ROOT = PACKAGE_ROOT.parents[1]
CONFIG_ROOT = REPO_ROOT / "configs"
VENDOR_ROOT = REPO_ROOT / "vendor"
NRPY_ROOT = VENDOR_ROOT / "nrpy"
SSBROYDEN_ROOT = VENDOR_ROOT / "ssbroyden"
REFERENCE_SOLVERS_ROOT = REPO_ROOT / "reference_solvers"
NRPYELLIPTIC_ROOT = REFERENCE_SOLVERS_ROOT / "nrpyelliptic"
NRPYELLIPTIC_AXISYMMETRIC_ROOT = NRPYELLIPTIC_ROOT / "axisymmetric"
NRPYELLIPTIC_THREE_DIMENSIONAL_ROOT = NRPYELLIPTIC_ROOT / "three_dimensional"
NRPYELLIPTIC_READER = (
    NRPYELLIPTIC_THREE_DIMENSIONAL_ROOT
    / "READER_nrpyelliptic_conformally_flat"
    / "nrpyell_reader"
)


def solver_config_dir(solver: str) -> Path:
    """Return the explicit repository configuration directory for a solver."""

    return CONFIG_ROOT / solver
