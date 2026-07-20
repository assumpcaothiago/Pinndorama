"""Validated TOML configuration for the publication 3D SinhSymTP solver."""

from __future__ import annotations

from dataclasses import asdict, dataclass, replace
from pathlib import Path
import math
import tomllib
from typing import Any, Mapping

SCHEMA_VERSION = 2
DTYPE = "float64"
ACTIVATION = "tanh"
INITIALIZATION = "he"
OUTPUT_TRANSFORM = "smooth_inverse_radius"
PHYSICS_MODEL = "bowen_york_two_punctures_z_axis"
RESIDUAL_FORM = "divergence_form_regularized_j_times_r_h"
XX0_SAMPLING_MODES = ("uniform", "two-zone")


@dataclass(frozen=True)
class Geometry:
    AMAX: float
    bScale: float
    SINHWAA: float


@dataclass(frozen=True)
class Physics:
    model: str
    residual: str
    bare_mass_0: float
    bare_mass_1: float
    zPunc: float
    P0_x: float
    P0_y: float
    P0_z: float
    P1_x: float
    P1_y: float
    P1_z: float
    S0_x: float
    S0_y: float
    S0_z: float
    S1_x: float
    S1_y: float
    S1_z: float
    theta_axis_regularity_weight: float
    phi_periodicity_weight: float
    outer_robin_weight: float
    outer_robin_scale: str


@dataclass(frozen=True)
class Architecture:
    layers: tuple[int, ...]
    activation: str
    initialization: str


@dataclass(frozen=True)
class Ansatz:
    output_transform: str


@dataclass(frozen=True)
class Collocation:
    Nxx0: int
    Nxx1: int
    Nxx2: int
    xx0_sampling: str
    xx0_cut: float | None
    Nxx0_inner: int | None
    xx0_min: float
    xx0_max: float
    xx1_min: float
    xx1_max: float
    xx2_min: float
    xx2_max: float
    boundary_xx0_min: float
    boundary_xx1_epsilon: float
    xx1_axis_delta_min: float
    xx1_axis_delta_max: float


@dataclass(frozen=True)
class Training:
    seed: int
    adam_steps: int
    learning_rate: float
    min_learning_rate: float
    gradient_clip_value: float
    ssbroyden_steps: int
    ssbroyden_rtol: float
    ssbroyden_atol: float
    log_every: int
    checkpoint_every: int


@dataclass(frozen=True)
class Continuation:
    require_resume: bool


@dataclass(frozen=True)
class RunConfig:
    schema_version: int
    name: str
    dtype: str
    geometry: Geometry
    physics: Physics
    architecture: Architecture
    ansatz: Ansatz
    collocation: Collocation
    training: Training
    continuation: Continuation
    source_path: Path

    def metadata_sections(self) -> dict[str, Any]:
        """Return the immutable/scientific configuration in JSON-ready form."""

        return {
            "dtype": self.dtype,
            "architecture": asdict(self.architecture),
            "geometry": asdict(self.geometry),
            "physics": asdict(self.physics),
            "ansatz": asdict(self.ansatz),
            "collocation": asdict(self.collocation),
            "training": asdict(self.training),
        }


def enable_jax_x64() -> None:
    """Enable the only supported numerical precision before arrays are created."""

    from jax import config as jax_config

    jax_config.update("jax_enable_x64", True)


def jax_dtype():
    enable_jax_x64()
    import jax.numpy as jnp

    return jnp.float64


def _table(raw: Mapping[str, Any], name: str) -> Mapping[str, Any]:
    value = raw.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"TOML section [{name}] is required")
    return value


def _exact_keys(
    table: Mapping[str, Any],
    *,
    section: str,
    required: set[str],
    optional: set[str] | None = None,
) -> None:
    if optional is None:
        optional = set()
    missing = sorted(required - set(table))
    unknown = sorted(set(table) - required - optional)
    if missing:
        raise ValueError(f"[{section}] is missing required keys: {', '.join(missing)}")
    if unknown:
        raise ValueError(f"[{section}] has unsupported keys: {', '.join(unknown)}")


def _positive(value: Any, name: str) -> float:
    result = float(value)
    if not math.isfinite(result) or result <= 0.0:
        raise ValueError(f"{name} must be a positive finite number")
    return result


def _nonnegative_int(value: Any, name: str) -> int:
    if isinstance(value, bool) or int(value) != value or int(value) < 0:
        raise ValueError(f"{name} must be a non-negative integer")
    return int(value)


def _positive_int(value: Any, name: str) -> int:
    result = _nonnegative_int(value, name)
    if result == 0:
        raise ValueError(f"{name} must be positive")
    return result


def load(path: str | Path) -> RunConfig:
    """Load and strictly validate a publication run configuration."""

    source_path = Path(path).expanduser().resolve()
    with source_path.open("rb") as stream:
        raw = tomllib.load(stream)

    top_required = {
        "schema_version",
        "name",
        "dtype",
        "geometry",
        "physics",
        "architecture",
        "ansatz",
        "collocation",
        "training",
        "continuation",
    }
    _exact_keys(raw, section="top level", required=top_required)
    if int(raw["schema_version"]) != SCHEMA_VERSION:
        raise ValueError(f"schema_version must be {SCHEMA_VERSION}")
    if raw["dtype"] != DTYPE:
        raise ValueError(f"dtype must be {DTYPE!r}")

    geometry_raw = _table(raw, "geometry")
    _exact_keys(
        geometry_raw, section="geometry", required={"AMAX", "bScale", "SINHWAA"}
    )
    geometry = Geometry(
        AMAX=_positive(geometry_raw["AMAX"], "geometry.AMAX"),
        bScale=_positive(geometry_raw["bScale"], "geometry.bScale"),
        SINHWAA=_positive(geometry_raw["SINHWAA"], "geometry.SINHWAA"),
    )

    physics_keys = {
        "model",
        "residual",
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
        "theta_axis_regularity_weight",
        "phi_periodicity_weight",
        "outer_robin_weight",
        "outer_robin_scale",
    }
    physics_raw = _table(raw, "physics")
    _exact_keys(physics_raw, section="physics", required=physics_keys)
    if physics_raw["model"] != PHYSICS_MODEL:
        raise ValueError(f"physics.model must be {PHYSICS_MODEL!r}")
    if physics_raw["residual"] != RESIDUAL_FORM:
        raise ValueError(f"physics.residual must be {RESIDUAL_FORM!r}")
    if physics_raw["outer_robin_scale"] != "one_plus_r":
        raise ValueError("physics.outer_robin_scale must be 'one_plus_r'")
    numeric_physics = {
        key: float(physics_raw[key])
        for key in physics_keys
        if key not in {"model", "residual", "outer_robin_scale"}
    }
    if not all(math.isfinite(value) for value in numeric_physics.values()):
        raise ValueError("all numeric [physics] values must be finite")
    for key in (
        "bare_mass_0",
        "bare_mass_1",
        "zPunc",
        "theta_axis_regularity_weight",
        "phi_periodicity_weight",
        "outer_robin_weight",
    ):
        if numeric_physics[key] <= 0.0:
            raise ValueError(f"physics.{key} must be positive")
    physics = Physics(
        model=str(physics_raw["model"]),
        residual=str(physics_raw["residual"]),
        outer_robin_scale=str(physics_raw["outer_robin_scale"]),
        **numeric_physics,
    )

    architecture_raw = _table(raw, "architecture")
    _exact_keys(
        architecture_raw,
        section="architecture",
        required={"layers", "activation", "initialization"},
    )
    layers = tuple(int(value) for value in architecture_raw["layers"])
    if (
        len(layers) < 3
        or layers[0] != 3
        or layers[-1] != 1
        or any(value <= 0 for value in layers)
    ):
        raise ValueError(
            "architecture.layers must be [3, hidden..., 1] with positive widths"
        )
    if architecture_raw["activation"] != ACTIVATION:
        raise ValueError(f"architecture.activation must be {ACTIVATION!r}")
    if architecture_raw["initialization"] != INITIALIZATION:
        raise ValueError(f"architecture.initialization must be {INITIALIZATION!r}")
    architecture = Architecture(
        layers=layers,
        activation=str(architecture_raw["activation"]),
        initialization=str(architecture_raw["initialization"]),
    )

    ansatz_raw = _table(raw, "ansatz")
    _exact_keys(ansatz_raw, section="ansatz", required={"output_transform"})
    if ansatz_raw["output_transform"] != OUTPUT_TRANSFORM:
        raise ValueError(f"ansatz.output_transform must be {OUTPUT_TRANSFORM!r}")
    ansatz = Ansatz(output_transform=str(ansatz_raw["output_transform"]))

    collocation_raw = _table(raw, "collocation")
    collocation_required = {
        "Nxx0",
        "Nxx1",
        "Nxx2",
        "xx0_sampling",
        "xx0_min",
        "xx0_max",
        "xx1_min",
        "xx1_max",
        "xx2_min",
        "xx2_max",
        "boundary_xx0_min",
        "boundary_xx1_epsilon",
        "xx1_axis_delta_min",
        "xx1_axis_delta_max",
    }
    _exact_keys(
        collocation_raw,
        section="collocation",
        required=collocation_required,
        optional={"xx0_cut", "Nxx0_inner"},
    )
    sampling = str(collocation_raw["xx0_sampling"])
    if sampling not in XX0_SAMPLING_MODES:
        raise ValueError(
            f"collocation.xx0_sampling must be one of {XX0_SAMPLING_MODES}"
        )
    Nxx0 = _positive_int(collocation_raw["Nxx0"], "collocation.Nxx0")
    xx0_cut = (
        None if "xx0_cut" not in collocation_raw else float(collocation_raw["xx0_cut"])
    )
    Nxx0_inner = (
        None
        if "Nxx0_inner" not in collocation_raw
        else _positive_int(
            collocation_raw["Nxx0_inner"],
            "collocation.Nxx0_inner",
        )
    )
    xx0_min = float(collocation_raw["xx0_min"])
    xx0_max = float(collocation_raw["xx0_max"])
    if not 0.0 <= xx0_min < xx0_max <= 1.0:
        raise ValueError("collocation xx0 bounds must satisfy 0 <= min < max <= 1")
    if sampling == "uniform":
        if xx0_cut is not None or Nxx0_inner is not None:
            raise ValueError("uniform sampling must not set xx0_cut or Nxx0_inner")
    else:
        if xx0_cut is None or not xx0_min < xx0_cut < xx0_max:
            raise ValueError("two-zone sampling requires xx0_min < xx0_cut < xx0_max")
        if Nxx0_inner is None or not 0 < Nxx0_inner < Nxx0:
            raise ValueError("two-zone sampling requires 0 < Nxx0_inner < Nxx0")
    xx1_min = float(collocation_raw["xx1_min"])
    xx1_max = float(collocation_raw["xx1_max"])
    xx2_min = float(collocation_raw["xx2_min"])
    xx2_max = float(collocation_raw["xx2_max"])
    if not 0.0 <= xx1_min < xx1_max <= math.pi:
        raise ValueError("collocation xx1 bounds must lie in [0, pi]")
    if not -math.pi <= xx2_min < xx2_max <= math.pi:
        raise ValueError("collocation xx2 bounds must lie in [-pi, pi]")
    boundary_xx0_min = float(collocation_raw["boundary_xx0_min"])
    boundary_xx1_epsilon = _positive(
        collocation_raw["boundary_xx1_epsilon"],
        "collocation.boundary_xx1_epsilon",
    )
    delta_min = _positive(
        collocation_raw["xx1_axis_delta_min"],
        "collocation.xx1_axis_delta_min",
    )
    delta_max = _positive(
        collocation_raw["xx1_axis_delta_max"],
        "collocation.xx1_axis_delta_max",
    )
    if not xx0_min < boundary_xx0_min <= xx0_max:
        raise ValueError("collocation.boundary_xx0_min must lie inside the xx0 domain")
    if not 0.0 < boundary_xx1_epsilon < 0.5 * math.pi:
        raise ValueError("collocation.boundary_xx1_epsilon must lie in (0, pi/2)")
    if not 0.0 < delta_min < delta_max < 0.5 * math.pi:
        raise ValueError("xx1-axis deltas must satisfy 0 < min < max < pi/2")
    collocation = Collocation(
        Nxx0=Nxx0,
        Nxx1=_positive_int(collocation_raw["Nxx1"], "collocation.Nxx1"),
        Nxx2=_positive_int(collocation_raw["Nxx2"], "collocation.Nxx2"),
        xx0_sampling=sampling,
        xx0_cut=xx0_cut,
        Nxx0_inner=Nxx0_inner,
        xx0_min=xx0_min,
        xx0_max=xx0_max,
        xx1_min=xx1_min,
        xx1_max=xx1_max,
        xx2_min=xx2_min,
        xx2_max=xx2_max,
        boundary_xx0_min=boundary_xx0_min,
        boundary_xx1_epsilon=boundary_xx1_epsilon,
        xx1_axis_delta_min=delta_min,
        xx1_axis_delta_max=delta_max,
    )

    training_raw = _table(raw, "training")
    training_keys = {
        "seed",
        "adam_steps",
        "learning_rate",
        "min_learning_rate",
        "gradient_clip_value",
        "ssbroyden_steps",
        "ssbroyden_rtol",
        "ssbroyden_atol",
        "log_every",
        "checkpoint_every",
    }
    _exact_keys(training_raw, section="training", required=training_keys)
    learning_rate = _positive(training_raw["learning_rate"], "training.learning_rate")
    min_learning_rate = _positive(
        training_raw["min_learning_rate"], "training.min_learning_rate"
    )
    if min_learning_rate > learning_rate:
        raise ValueError("training.min_learning_rate must not exceed learning_rate")
    training = Training(
        seed=_nonnegative_int(training_raw["seed"], "training.seed"),
        adam_steps=_nonnegative_int(training_raw["adam_steps"], "training.adam_steps"),
        learning_rate=learning_rate,
        min_learning_rate=min_learning_rate,
        gradient_clip_value=_positive(
            training_raw["gradient_clip_value"], "training.gradient_clip_value"
        ),
        ssbroyden_steps=_nonnegative_int(
            training_raw["ssbroyden_steps"], "training.ssbroyden_steps"
        ),
        ssbroyden_rtol=_positive(
            training_raw["ssbroyden_rtol"], "training.ssbroyden_rtol"
        ),
        ssbroyden_atol=_positive(
            training_raw["ssbroyden_atol"], "training.ssbroyden_atol"
        ),
        log_every=_positive_int(training_raw["log_every"], "training.log_every"),
        checkpoint_every=_positive_int(
            training_raw["checkpoint_every"], "training.checkpoint_every"
        ),
    )

    continuation_raw = _table(raw, "continuation")
    _exact_keys(continuation_raw, section="continuation", required={"require_resume"})
    if not isinstance(continuation_raw["require_resume"], bool):
        raise ValueError("continuation.require_resume must be a boolean")

    return RunConfig(
        schema_version=SCHEMA_VERSION,
        name=str(raw["name"]),
        dtype=DTYPE,
        geometry=geometry,
        physics=physics,
        architecture=architecture,
        ansatz=ansatz,
        collocation=collocation,
        training=training,
        continuation=Continuation(require_resume=continuation_raw["require_resume"]),
        source_path=source_path,
    )


def with_overrides(
    run: RunConfig,
    *,
    Nxx0: int | None = None,
    Nxx1: int | None = None,
    Nxx2: int | None = None,
    seed: int | None = None,
    adam_steps: int | None = None,
    ssbroyden_steps: int | None = None,
) -> RunConfig:
    """Apply the deliberately small command-line override surface."""

    collocation = replace(
        run.collocation,
        Nxx0=(run.collocation.Nxx0 if Nxx0 is None else _positive_int(Nxx0, "--Nxx0")),
        Nxx1=(run.collocation.Nxx1 if Nxx1 is None else _positive_int(Nxx1, "--Nxx1")),
        Nxx2=(run.collocation.Nxx2 if Nxx2 is None else _positive_int(Nxx2, "--Nxx2")),
    )
    if collocation.xx0_sampling == "two-zone":
        assert collocation.Nxx0_inner is not None
        if collocation.Nxx0_inner >= collocation.Nxx0:
            raise ValueError(
                "--Nxx0 must remain greater than the configured " "two-zone Nxx0_inner"
            )
    training = replace(
        run.training,
        seed=(run.training.seed if seed is None else _nonnegative_int(seed, "--seed")),
        adam_steps=(
            run.training.adam_steps
            if adam_steps is None
            else _nonnegative_int(adam_steps, "--adam-steps")
        ),
        ssbroyden_steps=(
            run.training.ssbroyden_steps
            if ssbroyden_steps is None
            else _nonnegative_int(ssbroyden_steps, "--ssbroyden-steps")
        ),
    )
    return replace(run, collocation=collocation, training=training)


def source_parameter_dict(physics: Physics) -> dict[str, float]:
    """Return parameters in the stable order expected by NRPy source expressions."""

    values = asdict(physics)
    excluded = {
        "model",
        "residual",
        "theta_axis_regularity_weight",
        "phi_periodicity_weight",
        "outer_robin_weight",
        "outer_robin_scale",
    }
    return {key: float(value) for key, value in values.items() if key not in excluded}
