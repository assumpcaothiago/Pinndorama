from __future__ import annotations

from pathlib import Path
import shutil
import subprocess

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SOLVER_DIRS = (
    REPO_ROOT
    / "reference_solvers"
    / "nrpyelliptic"
    / "axisymmetric"
    / "nrpyelliptic_conformally_flat",
    REPO_ROOT
    / "reference_solvers"
    / "nrpyelliptic"
    / "three_dimensional"
    / "nrpyelliptic_conformally_flat",
)


@pytest.mark.slow
@pytest.mark.parametrize("solver_dir", SOLVER_DIRS, ids=("2d", "3d"))
def test_standard_nrpy_solver_builds(solver_dir: Path) -> None:
    if shutil.which("make") is None or shutil.which("cc") is None:
        pytest.skip("C build tools are unavailable")
    try:
        subprocess.run(["make", "-j2"], cwd=solver_dir, check=True)
        result = subprocess.run(
            [str(solver_dir / "nrpyelliptic_conformally_flat"), "--help"],
            cwd=solver_dir,
            text=True,
            capture_output=True,
        )
        assert result.returncode == 0
        assert "Usage option" in result.stderr
    finally:
        subprocess.run(["make", "clean"], cwd=solver_dir, check=False)
