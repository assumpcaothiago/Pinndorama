"""Validated TOML configuration for the publication 2D SinhSymTP solver."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
import tomllib
from typing import Any, Mapping

SCHEMA_VERSION = 2
DTYPE = "float64"
OUTPUT_TRANSFORM = "smooth_inverse_radius"
ACTIVATION = "tanh"
RESIDUAL = "divergence_form_J_R_H"

# Defaults exist only to make the numerical helper functions convenient to use.
# Publication training always obtains these values from a checked-in TOML file.
DEFAULT_GEOMETRY = {"AMAX": 1.0e6, "bScale": 2.5, "SINHWAA": 0.07}
DEFAULT_PHYSICS = {
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
    "S0_z": 0.2,
    "S1_x": 0.0,
    "S1_y": 0.0,
    "S1_z": -0.15,
}

# Numerical helper defaults. They are deliberately not a second training
# configuration: ``train.py`` always loads a TOML and passes its values.
AMAX = DEFAULT_GEOMETRY["AMAX"]
bScale = DEFAULT_GEOMETRY["bScale"]
SINHWAA = DEFAULT_GEOMETRY["SINHWAA"]
Nxx0 = 128
Nxx1 = 128
Nparity = 64
Nxx1_outer = 128
xx0_min = 0.0
xx0_max = 1.0
xx1_min = 0.0
xx1_max = math.pi
xx1_parity_delta_min = 0.0
xx1_parity_delta_max = 0.1


def source_parameter_dict() -> dict[str, float]:
    """Return the source defaults used only by standalone expression helpers."""

    return {**DEFAULT_GEOMETRY, **DEFAULT_PHYSICS}


class ConfigError(ValueError):
    """Raised when a publication configuration is incomplete or inconsistent."""


@dataclass(frozen=True)
class SolverConfig:
    """Deeply validated solver configuration."""

    path: Path
    name: str
    dtype: str
    geometry: dict[str, float]
    physics: dict[str, float]
    architecture: dict[str, Any]
    ansatz: dict[str, Any]
    loss: dict[str, Any]
    collocation: dict[str, Any]
    training: dict[str, Any]

    def with_overrides(
        self,
        *,
        Nxx0: int | None = None,
        Nxx1: int | None = None,
        seed: int | None = None,
        adam_steps: int | None = None,
        ssbroyden_steps: int | None = None,
    ) -> "SolverConfig":
        """Return a validated copy with the allowed command-line overrides."""

        raw = self.as_dict()
        if Nxx0 is not None:
            raw["collocation"]["Nxx0"] = Nxx0
        if Nxx1 is not None:
            raw["collocation"]["Nxx1"] = Nxx1
        if seed is not None:
            raw["training"]["seed"] = seed
        if adam_steps is not None:
            raw["training"]["adam_steps"] = adam_steps
        if ssbroyden_steps is not None:
            raw["training"]["ssbroyden_steps"] = ssbroyden_steps
        return _validate(raw, self.path)

    def as_dict(self) -> dict[str, Any]:
        return {
            "schema_version": SCHEMA_VERSION,
            "name": self.name,
            "dtype": self.dtype,
            "geometry": deepcopy(self.geometry),
            "physics": deepcopy(self.physics),
            "architecture": deepcopy(self.architecture),
            "ansatz": deepcopy(self.ansatz),
            "loss": deepcopy(self.loss),
            "collocation": deepcopy(self.collocation),
            "training": deepcopy(self.training),
        }


_TOP_LEVEL = {
    "schema_version",
    "name",
    "dtype",
    "geometry",
    "physics",
    "architecture",
    "ansatz",
    "loss",
    "collocation",
    "training",
}
_SECTION_KEYS = {
    "geometry": {"coordinate_system", "AMAX", "bScale", "SINHWAA"},
    "physics": set(DEFAULT_PHYSICS),
    "architecture": {"layers", "activation", "initialization"},
    "ansatz": {"output_transform"},
    "loss": {
        "interior_residual",
        "interior_weight",
        "theta_parity_weight",
        "outer_robin_weight",
        "outer_robin_scale",
    },
    "collocation": {
        "sampling",
        "Nxx0",
        "Nxx1",
        "xx0_min",
        "xx0_max",
        "xx1_min",
        "xx1_max",
        "xx1_parity_delta_max",
    },
    "training": {
        "seed",
        "adam_steps",
        "adam_learning_rate",
        "adam_min_learning_rate",
        "gradient_clip",
        "ssbroyden_steps",
        "ssbroyden_rtol",
        "ssbroyden_atol",
        "log_every",
        "checkpoint_every",
    },
}


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"[{name}] must be a TOML table")
    return dict(value)


def _require_exact_keys(raw: Mapping[str, Any], expected: set[str], name: str) -> None:
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing:
        raise ConfigError(f"{name} is missing keys: {', '.join(missing)}")
    if unknown:
        raise ConfigError(f"{name} has unknown keys: {', '.join(unknown)}")


def _real(value: Any, name: str, *, positive: bool = False) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ConfigError(f"{name} must be a real number")
    result = float(value)
    if not math.isfinite(result):
        raise ConfigError(f"{name} must be finite")
    if positive and result <= 0.0:
        raise ConfigError(f"{name} must be positive")
    return result


def _integer(value: Any, name: str, *, minimum: int = 0) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value < minimum:
        raise ConfigError(f"{name} must be an integer >= {minimum}")
    return int(value)


def _validate(raw_input: Mapping[str, Any], path: Path) -> SolverConfig:
    raw = deepcopy(dict(raw_input))
    _require_exact_keys(raw, _TOP_LEVEL, "configuration")
    if (
        isinstance(raw["schema_version"], bool)
        or raw["schema_version"] != SCHEMA_VERSION
    ):
        raise ConfigError(
            f"schema_version must be {SCHEMA_VERSION}, got {raw['schema_version']!r}"
        )
    if not isinstance(raw["name"], str) or not raw["name"].strip():
        raise ConfigError("name must be a non-empty string")
    if raw["dtype"] != DTYPE:
        raise ConfigError(f"dtype must be {DTYPE!r}")

    sections: dict[str, dict[str, Any]] = {}
    for section_name, expected_keys in _SECTION_KEYS.items():
        section = _mapping(raw[section_name], section_name)
        _require_exact_keys(section, expected_keys, f"[{section_name}]")
        sections[section_name] = section

    geometry = sections["geometry"]
    if geometry["coordinate_system"] != "SinhSymTP":
        raise ConfigError("geometry.coordinate_system must be 'SinhSymTP'")
    for key in ("AMAX", "bScale", "SINHWAA"):
        geometry[key] = _real(geometry[key], f"geometry.{key}", positive=True)

    physics = sections["physics"]
    for key in _SECTION_KEYS["physics"]:
        physics[key] = _real(
            physics[key],
            f"physics.{key}",
            positive=key.startswith("bare_mass"),
        )
    if physics["zPunc"] <= 0.0:
        raise ConfigError("physics.zPunc must be positive")

    architecture = sections["architecture"]
    layers = architecture["layers"]
    if (
        not isinstance(layers, list)
        or len(layers) < 2
        or any(
            isinstance(value, bool) or not isinstance(value, int) or value <= 0
            for value in layers
        )
    ):
        raise ConfigError("architecture.layers must be a list of positive integers")
    if layers[0] != 2 or layers[-1] != 1:
        raise ConfigError("architecture.layers must start with 2 and end with 1")
    if architecture["activation"] != ACTIVATION:
        raise ConfigError(f"architecture.activation must be {ACTIVATION!r}")
    if architecture["initialization"] not in {"he", "xavier"}:
        raise ConfigError("architecture.initialization must be 'he' or 'xavier'")
    architecture["layers"] = [int(value) for value in layers]

    ansatz = sections["ansatz"]
    if ansatz["output_transform"] != OUTPUT_TRANSFORM:
        raise ConfigError(f"ansatz.output_transform must be {OUTPUT_TRANSFORM!r}")

    loss = sections["loss"]
    if loss["interior_residual"] != RESIDUAL:
        raise ConfigError(f"loss.interior_residual must be {RESIDUAL!r}")
    if loss["outer_robin_scale"] != "one_plus_r":
        raise ConfigError("loss.outer_robin_scale must be 'one_plus_r'")
    for key in ("interior_weight", "theta_parity_weight", "outer_robin_weight"):
        loss[key] = _real(loss[key], f"loss.{key}", positive=True)

    collocation = sections["collocation"]
    if collocation["sampling"] != "cell_centered":
        raise ConfigError("collocation.sampling must be 'cell_centered'")
    for key in ("Nxx0", "Nxx1"):
        collocation[key] = _integer(collocation[key], f"collocation.{key}", minimum=2)
    for key in (
        "xx0_min",
        "xx0_max",
        "xx1_min",
        "xx1_max",
        "xx1_parity_delta_max",
    ):
        collocation[key] = _real(collocation[key], f"collocation.{key}")
    if not collocation["xx0_min"] < collocation["xx0_max"]:
        raise ConfigError("collocation xx0_min must be smaller than xx0_max")
    if not collocation["xx1_min"] < collocation["xx1_max"]:
        raise ConfigError("collocation xx1_min must be smaller than xx1_max")
    if collocation["xx1_parity_delta_max"] <= 0.0:
        raise ConfigError("collocation.xx1_parity_delta_max must be positive")

    training = sections["training"]
    training["seed"] = _integer(training["seed"], "training.seed")
    for key in ("adam_steps", "ssbroyden_steps"):
        training[key] = _integer(training[key], f"training.{key}")
    for key in ("log_every", "checkpoint_every"):
        training[key] = _integer(training[key], f"training.{key}", minimum=1)
    for key in (
        "adam_learning_rate",
        "adam_min_learning_rate",
        "gradient_clip",
        "ssbroyden_rtol",
        "ssbroyden_atol",
    ):
        training[key] = _real(training[key], f"training.{key}", positive=True)
    if training["adam_min_learning_rate"] > training["adam_learning_rate"]:
        raise ConfigError("Adam minimum learning rate cannot exceed its initial rate")

    return SolverConfig(
        path=path,
        name=raw["name"],
        dtype=raw["dtype"],
        geometry=geometry,
        physics=physics,
        architecture=architecture,
        ansatz=ansatz,
        loss=loss,
        collocation=collocation,
        training=training,
    )


def load_config(path: str | Path) -> SolverConfig:
    """Read and validate one publication TOML configuration."""

    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open("rb") as file:
            raw = tomllib.load(file)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"could not read {config_path}: {error}") from error
    return _validate(raw, config_path)


def immutable_metadata(solver_config: SolverConfig) -> dict[str, Any]:
    """Return checkpoint fields that must match on continuation."""

    return {
        "dtype": solver_config.dtype,
        "architecture": deepcopy(solver_config.architecture),
        "geometry": deepcopy(solver_config.geometry),
        "physics": deepcopy(solver_config.physics),
        "ansatz": deepcopy(solver_config.ansatz),
    }


def enable_jax_x64() -> None:
    """Enable JAX float64 before any arrays are constructed."""

    from jax import config as jax_config

    jax_config.update("jax_enable_x64", True)


def jax_dtype():
    enable_jax_x64()
    import jax.numpy as jnp

    return jnp.float64
