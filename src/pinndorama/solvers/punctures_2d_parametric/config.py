"""Validated publication configuration for the parametric 2D solver."""

from __future__ import annotations

import copy
import math
from pathlib import Path
import tomllib
from typing import Any, Mapping

SCHEMA_VERSION = 2
EXPECTED_LAYERS = [3, 40, 40, 40, 40, 1]
FIXED_PHYSICS_NAMES = (
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
    "S1_x",
    "S1_y",
)


class ConfigError(ValueError):
    """Raised when a publication TOML file violates the supported schema."""


# P001 defaults also support direct imports by the numerical helper modules.
AMAX = 1.0e6
bScale = 2.5
SINHWAA = 0.07
bare_mass_0 = 0.5
bare_mass_1 = 0.5
zPunc = 2.5
P0_x = P0_y = P0_z = 0.0
P1_x = P1_y = P1_z = 0.0
S0_x = S0_y = S1_x = S1_y = 0.0
Nxx0 = 128
Nxx1 = 64
equal_spin_sz_min = -0.2
equal_spin_sz_max = 0.2
xx0_min = 9.0e-4
xx0_max = 1.0
xx1_min = 1.0e-5
xx1_max = math.pi - 1.0e-5
Nparity = 64
xx1_parity_delta_min = xx1_min
xx1_parity_delta_max = 0.1
Nxx1_outer = Nxx1
int_weight = 1.0
theta_inner_parity_weight = 1.0
outer_robin_boundary_weight = 1.0
regularized_residual_scale = 1.0
outer_robin_boundary_scale = "one_plus_r"
net_layers = list(EXPECTED_LAYERS)
initialization_type = "he"
dtype_name = "float64"
ssbroyden_step = 100
ssbroyden_max_steps = 50000
ssbroyden_rtol = 1.0e-12
ssbroyden_atol = 1.0e-12


def _table(data: Mapping[str, Any], name: str) -> dict[str, Any]:
    value = data.get(name)
    if not isinstance(value, dict):
        raise ConfigError(f"[{name}] must be present and must be a TOML table")
    return value


def _finite_number(value: Any, name: str) -> float:
    if isinstance(value, bool):
        raise ConfigError(f"{name} must be a finite number")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ConfigError(f"{name} must be a finite number") from exc
    if not math.isfinite(result):
        raise ConfigError(f"{name} must be finite")
    return result


def _positive_int(value: Any, name: str, *, allow_zero: bool = False) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ConfigError(f"{name} must be an integer")
    invalid = value < 0 if allow_zero else value <= 0
    if invalid:
        qualifier = "non-negative" if allow_zero else "positive"
        raise ConfigError(f"{name} must be {qualifier}")
    return int(value)


def _require_exact_keys(
    table: Mapping[str, Any], expected: set[str], table_name: str
) -> None:
    missing = expected - set(table)
    extra = set(table) - expected
    if missing:
        raise ConfigError(f"{table_name} is missing: {', '.join(sorted(missing))}")
    if extra:
        raise ConfigError(
            f"{table_name} contains unsupported fields: {', '.join(sorted(extra))}"
        )


def validate_config(data: Mapping[str, Any]) -> dict[str, Any]:
    """Return a deep-copied, validated configuration mapping."""

    cfg = copy.deepcopy(dict(data))
    expected_top = {
        "schema_version",
        "name",
        "dtype",
        "seed",
        "geometry",
        "physics",
        "parameter",
        "domain",
        "collocation",
        "architecture",
        "ansatz",
        "loss",
        "training",
    }
    _require_exact_keys(cfg, expected_top, "configuration")
    if cfg["schema_version"] != SCHEMA_VERSION:
        raise ConfigError(f"schema_version must be {SCHEMA_VERSION}")
    if not isinstance(cfg["name"], str) or not cfg["name"].strip():
        raise ConfigError("name must be a non-empty string")
    if cfg["dtype"] != "float64":
        raise ConfigError("dtype must be 'float64'")
    cfg["seed"] = _positive_int(cfg["seed"], "seed", allow_zero=True)

    geometry = _table(cfg, "geometry")
    _require_exact_keys(
        geometry, {"coordinate_system", "AMAX", "bScale", "SINHWAA"}, "geometry"
    )
    if geometry["coordinate_system"] != "SinhSymTP":
        raise ConfigError("geometry.coordinate_system must be 'SinhSymTP'")
    for key in ("AMAX", "bScale", "SINHWAA"):
        geometry[key] = _finite_number(geometry[key], f"geometry.{key}")
        if geometry[key] <= 0.0:
            raise ConfigError(f"geometry.{key} must be positive")

    physics = _table(cfg, "physics")
    _require_exact_keys(physics, set(FIXED_PHYSICS_NAMES), "physics")
    for key in FIXED_PHYSICS_NAMES:
        physics[key] = _finite_number(physics[key], f"physics.{key}")
    if physics["bare_mass_0"] <= 0.0 or physics["bare_mass_1"] <= 0.0:
        raise ConfigError("bare puncture masses must be positive")
    if physics["zPunc"] <= 0.0:
        raise ConfigError("physics.zPunc must be positive")

    parameter = _table(cfg, "parameter")
    _require_exact_keys(parameter, {"equal_spin_sz"}, "parameter")
    equal_spin = parameter["equal_spin_sz"]
    if not isinstance(equal_spin, dict):
        raise ConfigError("[parameter.equal_spin_sz] must be a TOML table")
    _require_exact_keys(
        equal_spin, {"minimum", "maximum", "sampling"}, "parameter.equal_spin_sz"
    )
    equal_spin["minimum"] = _finite_number(
        equal_spin["minimum"], "parameter.equal_spin_sz.minimum"
    )
    equal_spin["maximum"] = _finite_number(
        equal_spin["maximum"], "parameter.equal_spin_sz.maximum"
    )
    if equal_spin["minimum"] >= equal_spin["maximum"]:
        raise ConfigError("equal_spin_sz minimum must be smaller than maximum")
    if equal_spin["sampling"] != "cell_centered":
        raise ConfigError("equal_spin_sz sampling must be 'cell_centered'")

    domain = _table(cfg, "domain")
    _require_exact_keys(
        domain,
        {"xx0_min", "xx0_max", "xx1_min", "xx1_max", "xx1_parity_delta_max"},
        "domain",
    )
    for key in domain:
        domain[key] = _finite_number(domain[key], f"domain.{key}")
    if not 0.0 <= domain["xx0_min"] < domain["xx0_max"] <= 1.0:
        raise ConfigError("domain must satisfy 0 <= xx0_min < xx0_max <= 1")
    if not 0.0 < domain["xx1_min"] < domain["xx1_max"] < math.pi:
        raise ConfigError("domain must satisfy 0 < xx1_min < xx1_max < pi")
    if not domain["xx1_min"] <= domain["xx1_parity_delta_max"] < math.pi / 2:
        raise ConfigError("xx1_parity_delta_max is outside the supported range")

    collocation = _table(cfg, "collocation")
    _require_exact_keys(
        collocation,
        {"Nxx0", "Nxx1", "equal_spin_sz_points"},
        "collocation",
    )
    for key in collocation:
        collocation[key] = _positive_int(collocation[key], f"collocation.{key}")

    architecture = _table(cfg, "architecture")
    _require_exact_keys(
        architecture, {"layers", "activation", "initialization"}, "architecture"
    )
    if architecture["layers"] != EXPECTED_LAYERS:
        raise ConfigError(
            f"architecture.layers must be the publication w40d4 {EXPECTED_LAYERS}"
        )
    if architecture["activation"] != "tanh":
        raise ConfigError("architecture.activation must be 'tanh'")
    if architecture["initialization"] != "he":
        raise ConfigError("architecture.initialization must be 'he'")

    ansatz = _table(cfg, "ansatz")
    _require_exact_keys(ansatz, {"output_transform"}, "ansatz")
    if ansatz["output_transform"] != "smooth_inverse_radius":
        raise ConfigError("ansatz.output_transform must be 'smooth_inverse_radius'")

    loss = _table(cfg, "loss")
    _require_exact_keys(
        loss,
        {
            "residual",
            "residual_weight",
            "regularized_residual_scale",
            "theta_parity_weight",
            "outer_robin_weight",
            "outer_robin_scale",
        },
        "loss",
    )
    if loss["residual"] != "J_R_H":
        raise ConfigError("loss.residual must be 'J_R_H'")
    if loss["outer_robin_scale"] != "one_plus_r":
        raise ConfigError("loss.outer_robin_scale must be 'one_plus_r'")
    for key in (
        "residual_weight",
        "regularized_residual_scale",
        "theta_parity_weight",
        "outer_robin_weight",
    ):
        loss[key] = _finite_number(loss[key], f"loss.{key}")
        if loss[key] <= 0.0:
            raise ConfigError(f"loss.{key} must be positive")

    training = _table(cfg, "training")
    _require_exact_keys(training, {"adam", "ssbroyden"}, "training")
    adam = training["adam"]
    ssb = training["ssbroyden"]
    if not isinstance(adam, dict) or not isinstance(ssb, dict):
        raise ConfigError("training.adam and training.ssbroyden must be tables")
    _require_exact_keys(
        adam,
        {
            "steps",
            "learning_rate",
            "min_learning_rate",
            "gradient_clip",
            "log_every",
            "checkpoint_every",
        },
        "training.adam",
    )
    _require_exact_keys(
        ssb,
        {"steps", "rtol", "atol", "log_every", "checkpoint_every"},
        "training.ssbroyden",
    )
    for table_name, table in (("adam", adam), ("ssbroyden", ssb)):
        table["steps"] = _positive_int(
            table["steps"], f"training.{table_name}.steps", allow_zero=True
        )
        table["log_every"] = _positive_int(
            table["log_every"], f"training.{table_name}.log_every"
        )
        table["checkpoint_every"] = _positive_int(
            table["checkpoint_every"],
            f"training.{table_name}.checkpoint_every",
            allow_zero=True,
        )
    for key in ("learning_rate", "min_learning_rate", "gradient_clip"):
        adam[key] = _finite_number(adam[key], f"training.adam.{key}")
        if adam[key] <= 0.0:
            raise ConfigError(f"training.adam.{key} must be positive")
    if adam["min_learning_rate"] > adam["learning_rate"]:
        raise ConfigError("Adam min_learning_rate cannot exceed learning_rate")
    for key in ("rtol", "atol"):
        ssb[key] = _finite_number(ssb[key], f"training.ssbroyden.{key}")
        if ssb[key] <= 0.0:
            raise ConfigError(f"training.ssbroyden.{key} must be positive")
    return cfg


def load_config(path: str | Path) -> dict[str, Any]:
    """Load and validate a publication TOML configuration."""

    config_path = Path(path)
    try:
        with config_path.open("rb") as stream:
            data = tomllib.load(stream)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise ConfigError(f"Could not load {config_path}: {exc}") from exc
    return validate_config(data)


def apply_runtime_config(cfg: Mapping[str, Any]) -> None:
    """Apply validated values used by the numerical helper modules."""

    global AMAX, bScale, SINHWAA
    global Nxx0, Nxx1
    global equal_spin_sz_min, equal_spin_sz_max
    global xx0_min, xx0_max, xx1_min, xx1_max
    global xx1_parity_delta_min, xx1_parity_delta_max
    global Nparity, Nxx1_outer
    global int_weight, theta_inner_parity_weight, outer_robin_boundary_weight
    global regularized_residual_scale, outer_robin_boundary_scale
    global net_layers, initialization_type, dtype_name
    global ssbroyden_step, ssbroyden_max_steps, ssbroyden_rtol, ssbroyden_atol

    validated = validate_config(cfg)
    geometry = validated["geometry"]
    AMAX, bScale, SINHWAA = geometry["AMAX"], geometry["bScale"], geometry["SINHWAA"]
    for name in FIXED_PHYSICS_NAMES:
        globals()[name] = validated["physics"][name]
    collocation = validated["collocation"]
    Nxx0 = collocation["Nxx0"]
    Nxx1 = collocation["Nxx1"]
    spin = validated["parameter"]["equal_spin_sz"]
    equal_spin_sz_min, equal_spin_sz_max = spin["minimum"], spin["maximum"]
    domain = validated["domain"]
    xx0_min, xx0_max = domain["xx0_min"], domain["xx0_max"]
    xx1_min, xx1_max = domain["xx1_min"], domain["xx1_max"]
    xx1_parity_delta_min = xx1_min
    xx1_parity_delta_max = domain["xx1_parity_delta_max"]
    Nparity = Nxx1
    Nxx1_outer = Nxx1
    loss = validated["loss"]
    int_weight = loss["residual_weight"]
    theta_inner_parity_weight = loss["theta_parity_weight"]
    outer_robin_boundary_weight = loss["outer_robin_weight"]
    regularized_residual_scale = loss["regularized_residual_scale"]
    outer_robin_boundary_scale = loss["outer_robin_scale"]
    net_layers = list(validated["architecture"]["layers"])
    initialization_type = validated["architecture"]["initialization"]
    dtype_name = validated["dtype"]
    ssbroyden_step = validated["training"]["ssbroyden"]["log_every"]
    ssbroyden_max_steps = validated["training"]["ssbroyden"]["steps"]
    ssbroyden_rtol = validated["training"]["ssbroyden"]["rtol"]
    ssbroyden_atol = validated["training"]["ssbroyden"]["atol"]


def geometry_parameter_dict(cfg: Mapping[str, Any] | None = None) -> dict[str, float]:
    if cfg is None:
        return {"AMAX": float(AMAX), "bScale": float(bScale), "SINHWAA": float(SINHWAA)}
    return {key: float(cfg["geometry"][key]) for key in ("AMAX", "bScale", "SINHWAA")}


def source_parameter_dict(cfg: Mapping[str, Any] | None = None) -> dict[str, float]:
    if cfg is None:
        return {
            "AMAX": float(AMAX),
            "bScale": float(bScale),
            "SINHWAA": float(SINHWAA),
            **{name: float(globals()[name]) for name in FIXED_PHYSICS_NAMES},
            "S0_z": 0.0,
            "S1_z": 0.0,
        }
    return {
        **geometry_parameter_dict(cfg),
        **{name: float(cfg["physics"][name]) for name in FIXED_PHYSICS_NAMES},
        "S0_z": 0.0,
        "S1_z": 0.0,
    }


def immutable_metadata(cfg: Mapping[str, Any]) -> dict[str, Any]:
    """Return fields that must agree when loading a checkpoint."""

    return {
        "dtype": cfg["dtype"],
        "architecture": {
            "layers": list(cfg["architecture"]["layers"]),
            "activation": cfg["architecture"]["activation"],
        },
        "geometry": copy.deepcopy(cfg["geometry"]),
        "physics": copy.deepcopy(cfg["physics"]),
        "parameter": copy.deepcopy(cfg["parameter"]),
        "ansatz": copy.deepcopy(cfg["ansatz"]),
    }


def enable_jax_x64() -> None:
    from jax import config as jax_config

    jax_config.update("jax_enable_x64", True)


def jax_dtype():
    enable_jax_x64()
    import jax.numpy as jnp

    return jnp.float64
