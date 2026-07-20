"""Portable, atomic NPZ checkpoints for the 1D toy solver."""

from __future__ import annotations

from copy import deepcopy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from . import config

SCHEMA_VERSION = 1
IMMUTABLE_FIELDS = ("dtype", "architecture", "geometry", "physics", "ansatz")


class CheckpointError(ValueError):
    """Raised for malformed or incompatible checkpoints."""


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_metadata(
    solver_config: config.SolverConfig,
    *,
    stage: str,
    step: int,
    parent_checkpoint_sha256: str | None = None,
    diagnostics: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    """Construct versioned checkpoint metadata from the effective config."""

    metadata = {
        "schema_version": SCHEMA_VERSION,
        "config_name": solver_config.name,
        "config_sha256": sha256_file(solver_config.path),
        **config.immutable_metadata(solver_config),
        "loss": deepcopy(solver_config.loss),
        "collocation": deepcopy(solver_config.collocation),
        "training": deepcopy(solver_config.training),
        "stage": str(stage),
        "step": int(step),
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
    }
    if diagnostics is not None:
        metadata["diagnostics"] = deepcopy(dict(diagnostics))
    return metadata


def _parameter_arrays(params: Any) -> list[np.ndarray]:
    """Return Equinox-order dense arrays in deterministic weight/bias order."""

    arrays: list[np.ndarray] = []
    if not isinstance(params, (list, tuple)):
        raise CheckpointError("parameters must be a sequence of dense layers")
    for index, layer in enumerate(params):
        if not isinstance(layer, Mapping) or set(layer) != {"W", "b"}:
            raise CheckpointError(f"layer {index} must contain exactly W and b")
        weight = np.asarray(layer["W"])
        bias = np.asarray(layer["b"])
        if weight.dtype != np.float64 or bias.dtype != np.float64:
            raise CheckpointError("all checkpoint arrays must have dtype float64")
        if not np.all(np.isfinite(weight)) or not np.all(np.isfinite(bias)):
            raise CheckpointError("checkpoint arrays must contain only finite values")
        if weight.ndim != 2 or bias.ndim != 1 or weight.shape[0] != bias.shape[0]:
            raise CheckpointError(f"invalid dense layer shapes at layer {index}")
        arrays.extend((weight, bias))
    return arrays


def _validate_array_shapes(arrays: list[np.ndarray], layers: list[int]) -> None:
    if len(layers) < 2 or any(
        isinstance(value, bool) or not isinstance(value, int) or value <= 0
        for value in layers
    ):
        raise CheckpointError("architecture.layers must contain positive integers")
    expected_count = 2 * (len(layers) - 1)
    if len(arrays) != expected_count:
        raise CheckpointError(
            f"checkpoint has {len(arrays)} leaves; expected {expected_count}"
        )
    for layer_index, (in_size, out_size) in enumerate(zip(layers[:-1], layers[1:])):
        weight = arrays[2 * layer_index]
        bias = arrays[2 * layer_index + 1]
        if weight.dtype != np.float64 or bias.dtype != np.float64:
            raise CheckpointError("checkpoint leaves must have dtype float64")
        if not np.all(np.isfinite(weight)) or not np.all(np.isfinite(bias)):
            raise CheckpointError("checkpoint leaves must contain only finite values")
        # Match Equinox ``Linear`` storage: (out_features, in_features).
        if weight.shape != (out_size, in_size) or bias.shape != (out_size,):
            raise CheckpointError(
                f"layer {layer_index} has shapes {weight.shape}/{bias.shape}; "
                f"expected {(out_size, in_size)}/{(out_size,)}"
            )


def save_checkpoint(
    path: str | Path,
    params: Any,
    metadata: Mapping[str, Any],
) -> Path:
    """Atomically save dense parameters and UTF-8 JSON metadata to ``.npz``."""

    destination = Path(path)
    if destination.suffix != ".npz":
        raise CheckpointError("checkpoint paths must end in .npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    metadata_dict = deepcopy(dict(metadata))
    if metadata_dict.get("schema_version") != SCHEMA_VERSION:
        raise CheckpointError(f"metadata schema_version must be {SCHEMA_VERSION}")
    arrays = _parameter_arrays(params)
    layers = metadata_dict.get("architecture", {}).get("layers")
    if not isinstance(layers, list):
        raise CheckpointError("metadata must define architecture.layers")
    _validate_array_shapes(arrays, layers)

    payload: dict[str, np.ndarray] = {
        f"leaf_{index:03d}": array for index, array in enumerate(arrays)
    }
    encoded_metadata = json.dumps(
        metadata_dict,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")
    payload["metadata_json"] = np.frombuffer(encoded_metadata, dtype=np.uint8)

    temporary_name: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="wb",
            prefix=f".{destination.name}.",
            suffix=".tmp",
            dir=destination.parent,
            delete=False,
        ) as temporary:
            temporary_name = temporary.name
            np.savez(temporary, **payload)
            temporary.flush()
            os.fsync(temporary.fileno())
        os.replace(temporary_name, destination)
    finally:
        if temporary_name is not None and os.path.exists(temporary_name):
            os.unlink(temporary_name)
    return destination


def load_checkpoint(
    path: str | Path,
) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    """Load and structurally validate a toy-solver checkpoint."""

    checkpoint_path = Path(path)
    if checkpoint_path.suffix != ".npz":
        raise CheckpointError("only versioned .npz checkpoints are supported")
    try:
        with np.load(checkpoint_path, allow_pickle=False) as archive:
            names = set(archive.files)
            if "metadata_json" not in names:
                raise CheckpointError("checkpoint is missing metadata_json")
            metadata_array = archive["metadata_json"]
            if metadata_array.dtype != np.uint8 or metadata_array.ndim != 1:
                raise CheckpointError(
                    "metadata_json must be a one-dimensional uint8 array"
                )
            try:
                metadata = json.loads(metadata_array.tobytes().decode("utf-8"))
            except (UnicodeDecodeError, json.JSONDecodeError) as error:
                raise CheckpointError(f"invalid metadata_json: {error}") from error
            leaf_names = sorted(name for name in names if name.startswith("leaf_"))
            expected_names = [f"leaf_{index:03d}" for index in range(len(leaf_names))]
            if leaf_names != expected_names or names != set(leaf_names) | {
                "metadata_json"
            }:
                raise CheckpointError(
                    "checkpoint leaf names are not contiguous or are unknown"
                )
            arrays = [np.array(archive[name], copy=True) for name in leaf_names]
    except (OSError, ValueError) as error:
        if isinstance(error, CheckpointError):
            raise
        raise CheckpointError(f"could not load {checkpoint_path}: {error}") from error

    if (
        not isinstance(metadata, dict)
        or metadata.get("schema_version") != SCHEMA_VERSION
    ):
        raise CheckpointError(f"checkpoint schema_version must be {SCHEMA_VERSION}")
    architecture = metadata.get("architecture")
    layers = architecture.get("layers") if isinstance(architecture, dict) else None
    if not isinstance(layers, list):
        raise CheckpointError("checkpoint metadata is missing architecture.layers")
    _validate_array_shapes(arrays, layers)
    params = [
        {"W": arrays[index], "b": arrays[index + 1]}
        for index in range(0, len(arrays), 2)
    ]
    return params, metadata


def validate_resume(
    metadata: Mapping[str, Any], solver_config: config.SolverConfig
) -> None:
    """Reject changes to immutable numerical/model state on continuation."""

    expected = config.immutable_metadata(solver_config)
    mismatches = [
        field for field in IMMUTABLE_FIELDS if metadata.get(field) != expected[field]
    ]
    if mismatches:
        raise CheckpointError(
            "checkpoint is incompatible with the supplied config; immutable fields differ: "
            + ", ".join(mismatches)
        )
