from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
READER_DIR = (
    REPO_ROOT
    / "reference_solvers"
    / "nrpyelliptic"
    / "READER_nrpyelliptic_conformally_flat"
)
PHYSICS = {
    "bare_mass_0": 0.5,
    "bare_mass_1": 0.5,
    "zPunc": 2.5,
    "P0_x": 0.0,
    "P0_y": 0.0,
    "P0_z": 0.0,
    "P1_x": 0.0,
    "P1_y": 0.0,
    "P1_z": 0.0,
    "S0_x": 0.0,
    "S0_y": 0.0,
    "S0_z": 0.1,
    "S1_x": 0.0,
    "S1_y": 0.0,
    "S1_z": 0.1,
}


def _write_fixture(binary: Path, coords: Path) -> None:
    count = 11
    xx0 = np.linspace(0.0, 1.0, count, dtype=np.float64)
    xx1 = np.linspace(0.5, 1.5, count, dtype=np.float64)
    xx2 = np.linspace(-0.5, 0.5, count, dtype=np.float64)
    amax, bscale, sinhwaa = 10.0, 2.5, 0.7
    fields = np.zeros((4, count, count, count), dtype=np.float64)
    fields[0].fill(3.25)
    fields[1].fill(1.0)
    fields[2].fill(2.0)

    with binary.open("wb") as stream:
        stream.write(b"NRPYELL3")
        stream.write(struct.pack("=7i", 1, count, count, count, 5, count**3, 4))
        stream.write(struct.pack("=6d", amax, bscale, sinhwaa, 0.1, 0.1, 0.1))
        xx0.tofile(stream)
        xx1.tofile(stream)
        xx2.tofile(stream)
        fields.ravel().tofile(stream)

    native0, native1, native2 = xx0[5], xx1[5], xx2[5]
    radial = amax * math.sinh(native0 / sinhwaa) / math.sinh(1.0 / sinhwaa)
    x = radial * math.sin(native1) * math.cos(native2)
    y = radial * math.sin(native1) * math.sin(native2)
    z = math.sqrt(radial * radial + bscale * bscale) * math.cos(native1)
    coords.write_text(f"{x:.17e} {y:.17e} {z:.17e}\n", encoding="utf-8")


@pytest.mark.slow
def test_canonical_reader_uses_fixed_nine_point_stencil(tmp_path: Path) -> None:
    binary = tmp_path / "fixture.bin"
    coords = tmp_path / "coords.txt"
    output = tmp_path / "output.txt"
    _write_fixture(binary, coords)
    try:
        subprocess.run(["make"], cwd=READER_DIR, check=True)
        help_result = subprocess.run(
            [str(READER_DIR / "nrpyell_reader"), "--help"],
            text=True,
            capture_output=True,
            check=True,
        )
        assert "--binary FILE --coords FILE --output FILE" in help_result.stdout
        subprocess.run(
            [
                str(READER_DIR / "nrpyell_reader"),
                "--binary",
                str(binary),
                "--coords",
                str(coords),
                "--output",
                str(output),
            ],
            check=True,
        )
        table = np.loadtxt(output, comments="#", ndmin=2)
        assert table.shape == (1, 4)
        assert table[0, 3] == pytest.approx(3.25, rel=0.0, abs=1.0e-12)
        source = (READER_DIR / "main.c").read_text(encoding="utf-8")
        assert "NINTERP_GHOSTS = 4" in source

        geometry = {"AMAX": 10.0, "bScale": 2.5, "SINHWAA": 0.7}
        solver_config = tmp_path / "solver.toml"
        reference_config = tmp_path / "fixture.reference.toml"
        nn_values = tmp_path / "nn_values.txt"
        weights = tmp_path / "weights.txt"
        comparison = tmp_path / "comparison.json"
        solver_lines = ["[geometry]\n"]
        solver_lines.extend(f"{key} = {value!r}\n" for key, value in geometry.items())
        solver_lines.append("\n[physics]\n")
        solver_lines.extend(f"{key} = {value!r}\n" for key, value in PHYSICS.items())
        solver_config.write_text("".join(solver_lines), encoding="utf-8")
        reference_lines = [
            "schema_version = 1\n",
            f'binary_file = "{binary.name}"\n',
            f'binary_sha256 = "{hashlib.sha256(binary.read_bytes()).hexdigest()}"\n',
            "\n[geometry]\n",
        ]
        reference_lines.extend(
            f"{key} = {value!r}\n" for key, value in geometry.items()
        )
        reference_lines.append("\n[physics]\n")
        reference_lines.extend(f"{key} = {value!r}\n" for key, value in PHYSICS.items())
        reference_config.write_text("".join(reference_lines), encoding="utf-8")
        nn_values.write_text("# u_nn\n3.0\n", encoding="utf-8")
        weights.write_text("2.0\n", encoding="utf-8")
        subprocess.run(
            [
                sys.executable,
                "-m",
                "pinndorama.reproducibility.compare",
                "--nrpy-binary",
                str(binary),
                "--reference-config",
                str(reference_config),
                "--solver-config",
                str(solver_config),
                "--coords",
                str(coords),
                "--nn-values",
                str(nn_values),
                "--volume-weights",
                str(weights),
                "--reader",
                str(READER_DIR / "nrpyell_reader"),
                "--output",
                str(comparison),
            ],
            cwd=REPO_ROOT,
            check=True,
        )
        report = json.loads(comparison.read_text(encoding="utf-8"))
        assert report["metric"] == "volume_weighted_relative_l2"
        assert report["value"] == pytest.approx(1.0 / 13.0)
        assert report["sample_count"] == 1
    finally:
        subprocess.run(["make", "clean"], cwd=READER_DIR, check=False)
