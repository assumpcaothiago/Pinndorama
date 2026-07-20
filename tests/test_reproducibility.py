from __future__ import annotations

import hashlib
from pathlib import Path
import struct
import subprocess
import sys

import numpy as np
import pytest

from pinndorama.reproducibility.metrics import volume_weighted_relative_l2
from pinndorama.reproducibility.reference import (
    assert_same_problem,
    load_reference_metadata,
)
from pinndorama.reproducibility.sampling import (
    cell_centers,
    sinhsymtp_to_cartesian,
    sinhsymtp_volume_element,
    two_zone_cell_centers,
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
GEOMETRY = {"AMAX": 1.0e6, "bScale": 2.5, "SINHWAA": 0.07}


def _write_binary_header(path: Path, geometry: dict[str, float] = GEOMETRY) -> None:
    counts = (1, 11, 11, 11, 5, 11**3, 4)
    spacings = (0.1, 0.1, 0.1)
    path.write_bytes(
        b"NRPYELL3"
        + struct.pack("=7i", *counts)
        + struct.pack(
            "=6d",
            geometry["AMAX"],
            geometry["bScale"],
            geometry["SINHWAA"],
            *spacings,
        )
    )


def test_cell_centered_and_two_zone_sampling() -> None:
    np.testing.assert_allclose(cell_centers(-0.2, 0.2, 4), [-0.15, -0.05, 0.05, 0.15])
    points = two_zone_cell_centers(0.0, 0.5, 1.0, 8, 6)
    np.testing.assert_allclose(points[:6], cell_centers(0.0, 0.5, 6))
    np.testing.assert_allclose(points[6:], cell_centers(0.5, 1.0, 2))
    assert np.all(np.diff(points) > 0.0)
    with pytest.raises(ValueError, match="both radial zones"):
        two_zone_cell_centers(0.0, 0.5, 1.0, 8, 8)


def test_sinhsymtp_coordinate_map_axis_and_equator() -> None:
    x, y, z = sinhsymtp_to_cartesian(
        np.array([0.4, 0.4]),
        np.array([0.0, np.pi / 2.0]),
        np.array([1.2, 0.0]),
        amax=10.0,
        bscale=2.5,
        sinhwaa=0.7,
    )
    assert x[0] == pytest.approx(0.0, abs=1.0e-14)
    assert y[0] == pytest.approx(0.0, abs=1.0e-14)
    assert z[0] > 2.5
    assert y[1] == pytest.approx(0.0, abs=1.0e-14)
    assert z[1] == pytest.approx(0.0, abs=1.0e-14)
    assert x[1] > 0.0


def test_sinhsymtp_volume_element_golden_value() -> None:
    value = sinhsymtp_volume_element(0.37, 1.1, amax=1.0e6, bscale=2.5, sinhwaa=0.07)
    assert float(value) == pytest.approx(23931560.49595826, rel=5.0e-15)


def test_volume_weighted_relative_l2_definition() -> None:
    prediction = np.array([1.0, 3.0])
    reference = np.array([1.0, 2.0])
    weights = np.array([2.0, 4.0])
    expected = np.sqrt(4.0 / (2.0 + 16.0))
    assert volume_weighted_relative_l2(prediction, reference, weights) == pytest.approx(
        expected
    )
    with pytest.raises(ValueError, match="zero"):
        volume_weighted_relative_l2([1.0], [0.0], [1.0])


def _write_reference(path: Path, binary: Path, *, digest: str) -> None:
    lines = [
        "schema_version = 1\n",
        f'binary_file = "{binary.name}"\n',
        f'binary_sha256 = "{digest}"\n',
        "\n[geometry]\n",
    ]
    lines.extend(f"{key} = {value!r}\n" for key, value in GEOMETRY.items())
    lines.append("\n[physics]\n")
    lines.extend(f"{key} = {value!r}\n" for key, value in PHYSICS.items())
    path.write_text("".join(lines), encoding="utf-8")


def test_reference_toml_is_beside_and_authenticates_binary(tmp_path: Path) -> None:
    binary = tmp_path / "solution.bin"
    _write_binary_header(binary)
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    reference_path = tmp_path / "solution.reference.toml"
    _write_reference(reference_path, binary, digest=digest)

    reference = load_reference_metadata(reference_path, binary)
    assert reference.binary_sha256 == digest
    assert_same_problem(reference, GEOMETRY, PHYSICS)

    binary.write_bytes(b"changed")
    with pytest.raises(ValueError, match="SHA-256 mismatch"):
        load_reference_metadata(reference_path, binary)


def test_reference_toml_cannot_be_detached(tmp_path: Path) -> None:
    binary = tmp_path / "solution.bin"
    _write_binary_header(binary)
    elsewhere = tmp_path / "metadata"
    elsewhere.mkdir()
    reference_path = elsewhere / "solution.reference.toml"
    _write_reference(
        reference_path, binary, digest=hashlib.sha256(binary.read_bytes()).hexdigest()
    )
    with pytest.raises(ValueError, match="stored beside"):
        load_reference_metadata(reference_path, binary)


def test_create_reference_cli_round_trip(tmp_path: Path) -> None:
    binary = tmp_path / "solution.bin"
    _write_binary_header(binary)
    solver_config = tmp_path / "solver.toml"
    lines = ["[geometry]\n"]
    lines.extend(f"{key} = {value!r}\n" for key, value in GEOMETRY.items())
    lines.append("\n[physics]\n")
    lines.extend(f"{key} = {value!r}\n" for key, value in PHYSICS.items())
    solver_config.write_text("".join(lines), encoding="utf-8")
    output = tmp_path / "solution.reference.toml"
    subprocess.run(
        [
            sys.executable,
            "-m",
            "pinndorama.reproducibility.create_reference",
            "--binary",
            str(binary),
            "--solver-config",
            str(solver_config),
            "--output",
            str(output),
        ],
        cwd=Path(__file__).resolve().parents[1],
        check=True,
    )
    reference = load_reference_metadata(output, binary)
    assert_same_problem(reference, GEOMETRY, PHYSICS)


def test_reference_geometry_must_match_binary_header(tmp_path: Path) -> None:
    binary = tmp_path / "solution.bin"
    _write_binary_header(binary)
    digest = hashlib.sha256(binary.read_bytes()).hexdigest()
    reference_path = tmp_path / "solution.reference.toml"
    mismatched = dict(GEOMETRY)
    mismatched["bScale"] = 5.0
    lines = [
        "schema_version = 1\n",
        f'binary_file = "{binary.name}"\n',
        f'binary_sha256 = "{digest}"\n',
        "\n[geometry]\n",
    ]
    lines.extend(f"{key} = {value!r}\n" for key, value in mismatched.items())
    lines.append("\n[physics]\n")
    lines.extend(f"{key} = {value!r}\n" for key, value in PHYSICS.items())
    reference_path.write_text("".join(lines), encoding="utf-8")
    with pytest.raises(ValueError, match="does not match the geometry encoded"):
        load_reference_metadata(reference_path, binary)
