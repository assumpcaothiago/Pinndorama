"""Versioned, atomic NPZ checkpoints for the publication parametric solver."""

from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping

import numpy as np

from . import config

SCHEMA_VERSION = 1
IMMUTABLE_KEYS = ("dtype", "architecture", "geometry", "physics", "parameter", "ansatz")


def file_sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def _metadata_bytes(metadata: Mapping[str, Any]) -> bytes:
    return json.dumps(
        metadata, sort_keys=True, separators=(",", ":"), allow_nan=False
    ).encode("utf-8")


def decode_metadata(value: np.ndarray) -> dict[str, Any]:
    array = np.asarray(value)
    if array.dtype != np.uint8:
        raise ValueError("metadata_json must be stored as a uint8 UTF-8 byte array")
    try:
        result = json.loads(array.reshape(-1).tobytes().decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid metadata_json: {exc}") from exc
    if not isinstance(result, dict):
        raise ValueError("checkpoint metadata_json must contain a JSON object")
    if result.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"checkpoint schema_version must be {SCHEMA_VERSION}")
    return result


def model_leaves(params: Any) -> list[np.ndarray]:
    """Flatten dense layers as weight, bias pairs in layer order."""

    leaves: list[np.ndarray] = []
    for index, layer in enumerate(params):
        if not isinstance(layer, Mapping) or set(layer) != {"W", "b"}:
            raise ValueError(f"model layer {index} must contain exactly W and b")
        leaves.extend((np.asarray(layer["W"]), np.asarray(layer["b"])))
    return leaves


def _validate_leaves(leaves: list[np.ndarray], layers: list[int]) -> list[np.ndarray]:
    if len(leaves) != 2 * (len(layers) - 1):
        raise ValueError(
            f"expected {2 * (len(layers) - 1)} model arrays, got {len(leaves)}"
        )
    validated: list[np.ndarray] = []
    for index, (in_dim, out_dim) in enumerate(zip(layers[:-1], layers[1:])):
        weight = np.asarray(leaves[2 * index])
        bias = np.asarray(leaves[2 * index + 1])
        if weight.dtype != np.dtype(np.float64) or bias.dtype != np.dtype(np.float64):
            raise ValueError("all checkpoint model arrays must have dtype float64")
        # Equinox Linear's native storage order is (out_features, in_features).
        if weight.shape != (out_dim, in_dim):
            raise ValueError(
                f"leaf_{2 * index:03d} must have Equinox weight shape "
                f"{(out_dim, in_dim)}, got {weight.shape}"
            )
        if bias.shape != (out_dim,):
            raise ValueError(
                f"leaf_{2 * index + 1:03d} must have bias shape {(out_dim,)}, "
                f"got {bias.shape}"
            )
        if not np.all(np.isfinite(weight)) or not np.all(np.isfinite(bias)):
            raise ValueError("checkpoint model arrays must be finite")
        validated.extend((weight, bias))
    return validated


def params_from_leaves(
    leaves: list[np.ndarray], layers: list[int]
) -> list[dict[str, np.ndarray]]:
    validated = _validate_leaves(leaves, layers)
    return [
        {"W": validated[index], "b": validated[index + 1]}
        for index in range(0, len(validated), 2)
    ]


def save_checkpoint(
    path: str | Path,
    params: Any,
    metadata: Mapping[str, Any],
) -> Path:
    """Atomically write model arrays and canonical embedded JSON metadata."""

    output = Path(path)
    if output.suffix != ".npz":
        raise ValueError("publication checkpoints must use the .npz extension")
    output.parent.mkdir(parents=True, exist_ok=True)
    metadata_copy = copy.deepcopy(dict(metadata))
    if metadata_copy.get("schema_version") != SCHEMA_VERSION:
        raise ValueError(f"metadata schema_version must be {SCHEMA_VERSION}")
    layers = [int(value) for value in metadata_copy["architecture"]["layers"]]
    leaves = _validate_leaves(model_leaves(params), layers)
    payload: dict[str, np.ndarray] = {
        f"leaf_{index:03d}": leaf for index, leaf in enumerate(leaves)
    }
    payload["metadata_json"] = np.frombuffer(
        _metadata_bytes(metadata_copy), dtype=np.uint8
    ).copy()

    temporary: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            mode="w+b",
            prefix=f".{output.name}.",
            suffix=".tmp",
            dir=output.parent,
            delete=False,
        ) as stream:
            temporary = Path(stream.name)
            np.savez(stream, **payload)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary, output)
    finally:
        if temporary is not None and temporary.exists():
            temporary.unlink()
    return output


def load_checkpoint(
    path: str | Path,
) -> tuple[list[dict[str, np.ndarray]], dict[str, Any]]:
    """Load a clean-break checkpoint without enabling pickle."""

    checkpoint_path = Path(path)
    with np.load(checkpoint_path, allow_pickle=False) as archive:
        if "metadata_json" not in archive.files:
            raise ValueError("checkpoint is missing metadata_json")
        metadata = decode_metadata(archive["metadata_json"])
        leaf_names = sorted(
            (name for name in archive.files if name.startswith("leaf_")),
            key=lambda name: int(name.removeprefix("leaf_")),
        )
        expected_count = 2 * (len(metadata["architecture"]["layers"]) - 1)
        expected_names = [f"leaf_{index:03d}" for index in range(expected_count)]
        if leaf_names != expected_names:
            raise ValueError(
                f"checkpoint model arrays must be {expected_names}; got {leaf_names}"
            )
        unexpected = set(archive.files) - set(expected_names) - {"metadata_json"}
        if unexpected:
            raise ValueError(
                "checkpoint contains unsupported arrays: "
                + ", ".join(sorted(unexpected))
            )
        leaves = [np.array(archive[name], copy=True) for name in leaf_names]
    params = params_from_leaves(leaves, list(metadata["architecture"]["layers"]))
    return params, metadata


def validate_immutable_metadata(
    cfg: Mapping[str, Any], checkpoint_metadata: Mapping[str, Any]
) -> None:
    """Reject a resume/evaluation checkpoint with incompatible invariants."""

    expected = config.immutable_metadata(cfg)
    differences: list[str] = []
    for key in IMMUTABLE_KEYS:
        if checkpoint_metadata.get(key) != expected[key]:
            differences.append(key)
    if differences:
        raise ValueError(
            "checkpoint is incompatible with the supplied configuration: "
            + ", ".join(differences)
        )


def build_metadata(
    cfg: Mapping[str, Any],
    *,
    stage: str,
    step: int,
    parent_checkpoint_sha256: str | None,
    loss_value: float | None = None,
    loss_components: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Build the complete version-1 publication checkpoint metadata."""

    immutable = config.immutable_metadata(cfg)
    metadata: dict[str, Any] = {
        "schema_version": SCHEMA_VERSION,
        "config_name": cfg["name"],
        **immutable,
        "domain": copy.deepcopy(cfg["domain"]),
        "loss": copy.deepcopy(cfg["loss"]),
        "collocation": copy.deepcopy(cfg["collocation"]),
        "training": copy.deepcopy(cfg["training"]),
        "seed": int(cfg["seed"]),
        "stage": str(stage),
        "step": int(step),
        "parent_checkpoint_sha256": parent_checkpoint_sha256,
    }
    if loss_value is not None:
        metadata["loss_value"] = float(loss_value)
    if loss_components is not None:
        metadata["loss_components"] = {
            str(name): float(value) for name, value in loss_components.items()
        }
    return metadata
