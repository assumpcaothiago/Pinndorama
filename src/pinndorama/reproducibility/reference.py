"""Validation for NRPy binaries and their required physical metadata."""

from __future__ import annotations

from dataclasses import dataclass
import hashlib
import math
from pathlib import Path
import struct
import tomllib
from typing import Any, Mapping

GEOMETRY_KEYS = ("AMAX", "bScale", "SINHWAA")
PHYSICS_KEYS = (
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
RETIRED_TRANSVERSE_SEPARATION_NAME = "x" + "Punc"
NRPYELL_BINARY_HEADER = struct.Struct("=8s7i6d")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_binary_geometry(binary: str | Path) -> dict[str, float]:
    """Read the geometry encoded by a version-1 ``NRPYELL3`` binary."""

    binary = Path(binary)
    with binary.open("rb") as stream:
        payload = stream.read(NRPYELL_BINARY_HEADER.size)
    if len(payload) != NRPYELL_BINARY_HEADER.size:
        raise ValueError(f"{binary} is too short to contain an NRPYELL3 header")
    (
        magic,
        version,
        nxx0,
        nxx1,
        nxx2,
        nghosts,
        total_points,
        number_of_fields,
        amax,
        bscale,
        sinhwaa,
        _dxx0,
        _dxx1,
        _dxx2,
    ) = NRPYELL_BINARY_HEADER.unpack(payload)
    if magic != b"NRPYELL3":
        raise ValueError(f"{binary} does not use the NRPYELL3 binary format")
    if version != 1:
        raise ValueError(f"unsupported NRPYELL3 binary version {version}")
    if (
        min(nxx0, nxx1, nxx2, nghosts, total_points, number_of_fields) <= 0
        or total_points != nxx0 * nxx1 * nxx2
    ):
        raise ValueError(f"{binary} has an invalid NRPYELL3 grid header")
    geometry = {
        "AMAX": float(amax),
        "bScale": float(bscale),
        "SINHWAA": float(sinhwaa),
    }
    if not all(value > 0.0 for value in geometry.values()):
        raise ValueError(f"{binary} has non-positive SinhSymTP geometry")
    return geometry


def _numeric_table(
    document: Mapping[str, Any], section: str, keys: tuple[str, ...]
) -> dict[str, float]:
    table = document.get(section)
    if not isinstance(table, Mapping):
        raise ValueError(f"reference TOML requires a [{section}] table")
    unknown = set(table) - set(keys)
    missing = set(keys) - set(table)
    if unknown or missing:
        raise ValueError(
            f"[{section}] keys differ: missing={sorted(missing)}, "
            f"unknown={sorted(unknown)}"
        )
    result: dict[str, float] = {}
    for key in keys:
        value = table[key]
        if isinstance(value, bool) or not isinstance(value, (int, float)):
            raise ValueError(f"{section}.{key} must be numeric")
        result[key] = float(value)
        if not math.isfinite(result[key]):
            raise ValueError(f"{section}.{key} must be finite")
    return result


@dataclass(frozen=True)
class ReferenceMetadata:
    path: Path
    binary_path: Path
    binary_sha256: str
    geometry: dict[str, float]
    physics: dict[str, float]


def load_reference_metadata(
    reference_toml: str | Path, binary: str | Path
) -> ReferenceMetadata:
    """Load and authenticate a reference TOML stored beside its binary."""

    reference_toml = Path(reference_toml).expanduser().resolve()
    binary = Path(binary).expanduser().resolve()
    if reference_toml.parent != binary.parent:
        raise ValueError("reference TOML must be stored beside the NRPy binary")
    with reference_toml.open("rb") as stream:
        document = tomllib.load(stream)
    if document.get("schema_version") != 1:
        raise ValueError("reference TOML schema_version must equal 1")
    if RETIRED_TRANSVERSE_SEPARATION_NAME in str(document):
        raise ValueError(
            "a retired transverse puncture-separation parameter is not part "
            "of the publication source model"
        )
    expected_name = document.get("binary_file")
    expected_hash = document.get("binary_sha256")
    if expected_name != binary.name:
        raise ValueError(
            f"reference binary_file={expected_name!r} does not name {binary.name!r}"
        )
    if not isinstance(expected_hash, str) or len(expected_hash) != 64:
        raise ValueError("reference binary_sha256 must be a 64-character digest")
    actual_hash = sha256_file(binary)
    if actual_hash != expected_hash.lower():
        raise ValueError(
            f"NRPy binary SHA-256 mismatch: expected {expected_hash}, got {actual_hash}"
        )
    geometry = _numeric_table(document, "geometry", GEOMETRY_KEYS)
    encoded_geometry = load_binary_geometry(binary)
    if geometry != encoded_geometry:
        raise ValueError(
            "reference [geometry] does not match the geometry encoded by "
            f"{binary.name}: reference={geometry}, binary={encoded_geometry}"
        )
    return ReferenceMetadata(
        path=reference_toml,
        binary_path=binary,
        binary_sha256=actual_hash,
        geometry=geometry,
        physics=_numeric_table(document, "physics", PHYSICS_KEYS),
    )


def assert_same_problem(
    reference: ReferenceMetadata,
    geometry: Mapping[str, float],
    physics: Mapping[str, float],
) -> None:
    """Require exact publication metadata compatibility before comparison."""

    for section_name, expected, supplied in (
        ("geometry", reference.geometry, geometry),
        ("physics", reference.physics, physics),
    ):
        if set(supplied) != set(expected):
            raise ValueError(f"{section_name} parameter names do not match")
        for key, expected_value in expected.items():
            if float(supplied[key]) != expected_value:
                raise ValueError(
                    f"{section_name}.{key} differs: "
                    f"reference={expected_value!r}, supplied={supplied[key]!r}"
                )


def load_solver_problem(
    solver_toml: str | Path, *, equal_spin_sz: float | None = None
) -> tuple[dict[str, float], dict[str, float]]:
    """Extract publication geometry/physics from a checked-in solver TOML."""

    solver_toml = Path(solver_toml)
    with solver_toml.open("rb") as stream:
        document = tomllib.load(stream)
    geometry_document = document.get("geometry")
    if not isinstance(geometry_document, Mapping):
        raise ValueError("solver TOML requires a [geometry] table")
    coordinate_system = geometry_document.get("coordinate_system", "SinhSymTP")
    if coordinate_system != "SinhSymTP":
        raise ValueError("solver geometry.coordinate_system must be SinhSymTP")
    numeric_geometry = {
        key: value
        for key, value in geometry_document.items()
        if key != "coordinate_system"
    }
    geometry = _numeric_table({"geometry": numeric_geometry}, "geometry", GEOMETRY_KEYS)
    physics_table = document.get("physics")
    if not isinstance(physics_table, Mapping):
        raise ValueError("solver TOML requires a [physics] table")
    parameter_table = document.get("parameter")
    if parameter_table is not None:
        if not isinstance(parameter_table, Mapping):
            raise ValueError("solver [parameter] must be a table")
        spin_table = parameter_table.get("equal_spin_sz")
        if not isinstance(spin_table, Mapping):
            raise ValueError(
                "parametric solver TOML requires [parameter.equal_spin_sz]"
            )
        minimum = spin_table.get("minimum")
        maximum = spin_table.get("maximum")
        if (
            isinstance(minimum, bool)
            or isinstance(maximum, bool)
            or not isinstance(minimum, (int, float))
            or not isinstance(maximum, (int, float))
            or not math.isfinite(float(minimum))
            or not math.isfinite(float(maximum))
            or float(minimum) >= float(maximum)
        ):
            raise ValueError(
                "equal_spin_sz parameter range must be finite and increasing"
            )
        if equal_spin_sz is not None and not (
            float(minimum) <= float(equal_spin_sz) <= float(maximum)
        ):
            raise ValueError(
                f"equal_spin_sz={equal_spin_sz} is outside the configured range "
                f"[{minimum}, {maximum}]"
            )
    numeric_physics = {
        key: value
        for key, value in physics_table.items()
        if key in PHYSICS_KEYS and not isinstance(value, bool)
    }
    if equal_spin_sz is not None:
        numeric_physics["S0_z"] = equal_spin_sz
        numeric_physics["S1_z"] = equal_spin_sz
    physics = _numeric_table({"physics": numeric_physics}, "physics", PHYSICS_KEYS)
    return geometry, physics
