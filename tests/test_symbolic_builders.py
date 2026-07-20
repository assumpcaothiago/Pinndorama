from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SOLVERS = (
    "pinndorama.solvers.punctures_2d.nrpy_expression_builder",
    "pinndorama.solvers.punctures_2d_parametric.nrpy_expression_builder",
    "pinndorama.solvers.punctures_3d.nrpy_expression_builder",
)
EXPECTED_SOURCE_NAMES = {
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
}
RETIRED_SOURCE_NAME = "x" + "Punc"


@pytest.mark.parametrize("solver", SOLVERS)
def test_symbolic_builder_matches_pinned_upstream_nrpy(solver: str) -> None:
    script = r"""
import importlib
import json
import sys

module = importlib.import_module(sys.argv[1])
expressions = module.build_symbolic_expressions()
source_symbols = {str(symbol) for symbol in expressions.source_symbols}
free_symbols = {
    str(symbol)
    for expression in (expressions.psi_background, expressions.add_times_auu)
    for symbol in expression.free_symbols
}
print(json.dumps({"source": sorted(source_symbols), "free": sorted(free_symbols)}))
"""
    environment = os.environ.copy()
    environment["PYTHONPYCACHEPREFIX"] = "/tmp/pinndorama_pycache"
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", script, solver],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    document = json.loads(result.stdout.splitlines()[-1])
    assert set(document["source"]) == EXPECTED_SOURCE_NAMES
    assert "zPunc" in document["free"]
    assert RETIRED_SOURCE_NAME not in document["source"]
    assert RETIRED_SOURCE_NAME not in document["free"]


def test_parametric_source_golden_value_symmetry_and_regularization_identity() -> None:
    builder = "pinndorama.solvers.punctures_2d_parametric.nrpy_expression_builder"
    script = r"""
import importlib
import json
import pathlib
import sympy as sp
import sys

module = importlib.import_module(sys.argv[1])
expressions = module.build_symbolic_expressions()

source = {
    "xx0": 0.37, "xx1": 1.1, "xx2": 0.4,
    "AMAX": 1.0e6, "bScale": 2.5, "SINHWAA": 0.07,
    "bare_mass_0": 0.5, "bare_mass_1": 0.5, "zPunc": 2.5,
    "P0_x": 0.0, "P0_y": 0.0, "P0_z": 0.0,
    "P1_x": 0.0, "P1_y": 0.0, "P1_z": 0.0,
    "S0_x": 0.0, "S0_y": 0.0, "S0_z": 0.1,
    "S1_x": 0.0, "S1_y": 0.0, "S1_z": 0.1,
}

def evaluate(expression, values):
    substitutions = {symbol: values[str(symbol)] for symbol in expression.free_symbols}
    return float(sp.N(expression.subs(substitutions), 18))

psi = evaluate(expressions.psi_background, source)
add_positive = evaluate(expressions.add_times_auu, source)
source["S0_z"] = source["S1_z"] = -0.1
add_negative = evaluate(expressions.add_times_auu, source)

residual = {
    "xx0": 0.37, "xx1": 1.1, "xx2": 0.4,
    "AMAX": 1.0e6, "bScale": 2.5, "SINHWAA": 0.07,
    "uu": 0.2, "uu_dD0": -0.3, "uu_dD1": 0.4, "uu_dD2": 0.0,
    "uu_dDD00": 0.7, "uu_dDD01": 0.0, "uu_dDD02": 0.0,
    "uu_dDD10": 0.0, "uu_dDD11": -0.2, "uu_dDD12": 0.0,
    "uu_dDD20": 0.0, "uu_dDD21": 0.0, "uu_dDD22": 0.0,
    "psi_background": 1.1, "ADD_times_AUU": 0.3,
}
j_times_raw = evaluate(
    expressions.volume_element * expressions.residual_h, residual
)
regularized = evaluate(expressions.regularized_residual_h, residual)
print(json.dumps({
    "psi": psi,
    "add_positive": add_positive,
    "add_negative": add_negative,
    "j_times_raw": j_times_raw,
    "regularized": regularized,
}))
"""
    environment = os.environ.copy()
    environment["PYTHONPYCACHEPREFIX"] = "/tmp/pinndorama_pycache"
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", script, str(builder)],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    document = json.loads(result.stdout.splitlines()[-1])
    assert document["psi"] == pytest.approx(1.004051156641175, rel=1.0e-14)
    assert document["add_positive"] == pytest.approx(1.618593366367429e-13, rel=1.0e-13)
    assert document["add_negative"] == pytest.approx(
        document["add_positive"], rel=1.0e-14
    )
    assert document["regularized"] == pytest.approx(
        document["j_times_raw"], rel=2.0e-15
    )
