"""Validated configuration for the 1D SinhSpherical toy solver."""

from __future__ import annotations

from copy import deepcopy
from dataclasses import dataclass
import math
from pathlib import Path
import tomllib
from typing import Any, Mapping

SCHEMA_VERSION = 4
DTYPE = "float64"
ACTIVATION = "tanh"
OUTPUT_TRANSFORM = "smooth_inverse_radius"
RESIDUAL = "r_times_spherical_pde"
OUTER_BOUNDARY_CONDITIONS = {"first_order_robin", "second_order_robin"}
OUTER_BOUNDARY_REGIONS = {"endpoint", "radial_band"}


class ConfigError(ValueError):
    """Raised when a toy-problem configuration violates its schema."""


@dataclass(frozen=True)
class SolverConfig:
    """Fully validated configuration for one non-parametric run."""

    path: Path
    name: str
    dtype: str
    geometry: dict[str, Any]
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
        seed: int | None = None,
        adam_steps: int | None = None,
        ssbroyden_steps: int | None = None,
    ) -> "SolverConfig":
        raw = self.as_dict()
        if Nxx0 is not None:
            raw["collocation"]["Nxx0"] = Nxx0
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
    "geometry": {"coordinate_system", "AMPL", "SINHW"},
    "physics": {"m"},
    "architecture": {"layers", "activation", "initialization"},
    "ansatz": {"output_transform"},
    "collocation": {"sampling", "Nxx0", "xx0_min", "xx0_max"},
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


def _require_exact_keys(raw: Mapping[str, Any], expected: set[str], name: str) -> None:
    missing = sorted(expected - set(raw))
    unknown = sorted(set(raw) - expected)
    if missing:
        raise ConfigError(f"{name} is missing keys: {', '.join(missing)}")
    if unknown:
        raise ConfigError(f"{name} has unknown keys: {', '.join(unknown)}")


def _mapping(value: Any, name: str) -> dict[str, Any]:
    if not isinstance(value, Mapping):
        raise ConfigError(f"[{name}] must be a TOML table")
    return dict(value)


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
    if raw["schema_version"] != SCHEMA_VERSION or isinstance(
        raw["schema_version"], bool
    ):
        raise ConfigError(f"schema_version must be {SCHEMA_VERSION}")
    if not isinstance(raw["name"], str) or not raw["name"].strip():
        raise ConfigError("name must be a non-empty string")
    if raw["dtype"] != DTYPE:
        raise ConfigError(f"dtype must be {DTYPE!r}")

    sections: dict[str, dict[str, Any]] = {}
    for name, expected in _SECTION_KEYS.items():
        section = _mapping(raw[name], name)
        _require_exact_keys(section, expected, f"[{name}]")
        sections[name] = section

    loss = _mapping(raw["loss"], "loss")
    loss_common_keys = {
        "interior_residual",
        "interior_weight",
        "origin_regularity_weight",
        "outer_boundary_condition",
        "outer_boundary_region",
        "outer_boundary_weight",
    }
    missing_common = sorted(loss_common_keys - set(loss))
    if missing_common:
        raise ConfigError(f"[loss] is missing keys: {', '.join(missing_common)}")
    if loss["outer_boundary_region"] not in OUTER_BOUNDARY_REGIONS:
        choices = ", ".join(sorted(OUTER_BOUNDARY_REGIONS))
        raise ConfigError(f"loss.outer_boundary_region must be one of: {choices}")
    expected_loss_keys = set(loss_common_keys)
    if loss["outer_boundary_region"] == "radial_band":
        expected_loss_keys.add("outer_boundary_r_min")
    _require_exact_keys(loss, expected_loss_keys, "[loss]")
    sections["loss"] = loss

    geometry = sections["geometry"]
    if geometry["coordinate_system"] != "SinhSpherical":
        raise ConfigError("geometry.coordinate_system must be 'SinhSpherical'")
    for key in ("AMPL", "SINHW"):
        geometry[key] = _real(geometry[key], f"geometry.{key}", positive=True)

    physics = sections["physics"]
    physics["m"] = _real(physics["m"], "physics.m", positive=True)

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
    if layers[0] != 1 or layers[-1] != 1:
        raise ConfigError("architecture.layers must start with 1 and end with 1")
    if architecture["activation"] != ACTIVATION:
        raise ConfigError(f"architecture.activation must be {ACTIVATION!r}")
    if architecture["initialization"] not in {"he", "xavier"}:
        raise ConfigError("architecture.initialization must be 'he' or 'xavier'")
    architecture["layers"] = [int(value) for value in layers]

    ansatz = sections["ansatz"]
    if ansatz["output_transform"] != OUTPUT_TRANSFORM:
        raise ConfigError(f"ansatz.output_transform must be {OUTPUT_TRANSFORM!r}")

    if loss["interior_residual"] != RESIDUAL:
        raise ConfigError(f"loss.interior_residual must be {RESIDUAL!r}")
    if loss["outer_boundary_condition"] not in OUTER_BOUNDARY_CONDITIONS:
        choices = ", ".join(sorted(OUTER_BOUNDARY_CONDITIONS))
        raise ConfigError(f"loss.outer_boundary_condition must be one of: {choices}")
    for key in (
        "interior_weight",
        "origin_regularity_weight",
        "outer_boundary_weight",
    ):
        loss[key] = _real(loss[key], f"loss.{key}", positive=True)
    if loss["outer_boundary_region"] == "radial_band":
        loss["outer_boundary_r_min"] = _real(
            loss["outer_boundary_r_min"],
            "loss.outer_boundary_r_min",
            positive=True,
        )
        if loss["outer_boundary_r_min"] >= geometry["AMPL"]:
            raise ConfigError(
                "loss.outer_boundary_r_min must be smaller than geometry.AMPL"
            )

    collocation = sections["collocation"]
    if collocation["sampling"] != "cell_centered":
        raise ConfigError("collocation.sampling must be 'cell_centered'")
    collocation["Nxx0"] = _integer(collocation["Nxx0"], "collocation.Nxx0", minimum=2)
    for key in ("xx0_min", "xx0_max"):
        collocation[key] = _real(collocation[key], f"collocation.{key}")
    if collocation["xx0_min"] != 0.0 or collocation["xx0_max"] != 1.0:
        raise ConfigError("SinhSpherical collocation must span 0 <= xx0 <= 1")

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
    config_path = Path(path).expanduser().resolve()
    try:
        with config_path.open("rb") as stream:
            raw = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as error:
        raise ConfigError(f"could not read {config_path}: {error}") from error
    return _validate(raw, config_path)


def immutable_metadata(solver_config: SolverConfig) -> dict[str, Any]:
    return {
        "dtype": solver_config.dtype,
        "architecture": deepcopy(solver_config.architecture),
        "geometry": deepcopy(solver_config.geometry),
        "physics": deepcopy(solver_config.physics),
        "ansatz": deepcopy(solver_config.ansatz),
    }


def enable_jax_x64() -> None:
    from jax import config as jax_config

    jax_config.update("jax_enable_x64", True)


def jax_dtype():
    enable_jax_x64()
    import jax.numpy as jnp

    return jnp.float64
