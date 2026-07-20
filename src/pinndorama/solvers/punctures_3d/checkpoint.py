"""Portable, atomic, non-pickle checkpoints for publication runs."""

from __future__ import annotations

from datetime import datetime, timezone
import hashlib
import json
import os
from pathlib import Path
import tempfile
from typing import Any, Mapping, Sequence

import numpy as np

from . import config

CHECKPOINT_SCHEMA_VERSION = 1

IMMUTABLE_METADATA_FIELDS = (
    "dtype",
    "architecture",
    "geometry",
    "physics",
    "ansatz",
)


def sha256(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def build_metadata(
    run: config.RunConfig,
    *,
    stage: str,
    step: int,
    parent_sha256: str | None,
    metrics: Mapping[str, float] | None = None,
) -> dict[str, Any]:
    """Build schema-version-1 checkpoint metadata."""

    metadata = {
        "schema_version": CHECKPOINT_SCHEMA_VERSION,
        "name": run.name,
        **run.metadata_sections(),
        "stage": str(stage),
        "step": int(step),
        "parent_checkpoint_sha256": parent_sha256,
        "created_utc": datetime.now(timezone.utc).isoformat(),
    }
    if metrics is not None:
        metadata["metrics"] = {key: float(value) for key, value in metrics.items()}
    return metadata


def _json_bytes(metadata: Mapping[str, Any]) -> bytes:
    return json.dumps(
        metadata,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    ).encode("utf-8")


def _validate_leaves(leaves: Sequence[np.ndarray], metadata: Mapping[str, Any]) -> None:
    architecture = metadata.get("architecture")
    layers = architecture.get("layers") if isinstance(architecture, Mapping) else None
    if not isinstance(layers, (list, tuple)) or len(layers) < 2:
        raise ValueError("checkpoint metadata must define architecture.layers")
    layer_sizes = tuple(int(value) for value in layers)
    if len(leaves) != 2 * (len(layer_sizes) - 1):
        raise ValueError("checkpoint leaf count does not match architecture.layers")
    for index, (in_dim, out_dim) in enumerate(zip(layer_sizes[:-1], layer_sizes[1:])):
        weights = np.asarray(leaves[2 * index])
        bias = np.asarray(leaves[2 * index + 1])
        if weights.dtype != np.float64 or bias.dtype != np.float64:
            raise ValueError("all checkpoint leaves must have dtype float64")
        if weights.shape != (out_dim, in_dim) or bias.shape != (out_dim,):
            raise ValueError(
                f"layer {index} has shapes {weights.shape}/{bias.shape}; "
                f"expected {(out_dim, in_dim)}/{(out_dim,)}"
            )
        if not np.all(np.isfinite(weights)) or not np.all(np.isfinite(bias)):
            raise ValueError("checkpoint parameter leaves must be finite")


def save(
    path: str | Path,
    parameter_leaves: Sequence[Any],
    metadata: Mapping[str, Any],
) -> str:
    """Atomically write ``leaf_000...`` and uint8 ``metadata_json`` to NPZ."""

    destination = Path(path).expanduser().resolve()
    if destination.suffix != ".npz":
        raise ValueError("checkpoint path must end in .npz")
    destination.parent.mkdir(parents=True, exist_ok=True)
    if metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(f"metadata schema_version must be {CHECKPOINT_SCHEMA_VERSION}")
    leaves = [np.asarray(leaf) for leaf in parameter_leaves]
    _validate_leaves(leaves, metadata)
    arrays: dict[str, np.ndarray] = {}
    for index, array in enumerate(leaves):
        if array.dtype != np.float64:
            raise ValueError(
                f"leaf_{index:03d} has dtype {array.dtype}; checkpoints require float64"
            )
        arrays[f"leaf_{index:03d}"] = array
    arrays["metadata_json"] = np.frombuffer(_json_bytes(metadata), dtype=np.uint8)

    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{destination.name}.",
        suffix=".tmp",
        dir=destination.parent,
    )
    try:
        with os.fdopen(descriptor, "w+b") as stream:
            np.savez_compressed(stream, **arrays)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, destination)
        try:
            directory_fd = os.open(destination.parent, os.O_RDONLY)
        except OSError:
            directory_fd = None
        if directory_fd is not None:
            try:
                os.fsync(directory_fd)
            finally:
                os.close(directory_fd)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise
    return sha256(destination)


def load(path: str | Path) -> tuple[list[np.ndarray], dict[str, Any], str]:
    """Load and structurally validate a schema-version-1 checkpoint."""

    source = Path(path).expanduser().resolve()
    with np.load(source, allow_pickle=False) as archive:
        if "metadata_json" not in archive.files:
            raise ValueError(f"{source} is missing metadata_json")
        metadata_array = np.asarray(archive["metadata_json"])
        if metadata_array.dtype != np.uint8 or metadata_array.ndim != 1:
            raise ValueError("metadata_json must be a one-dimensional uint8 array")
        try:
            metadata = json.loads(metadata_array.tobytes().decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError) as error:
            raise ValueError(f"invalid checkpoint metadata_json: {error}") from error
        leaf_names = sorted(name for name in archive.files if name.startswith("leaf_"))
        expected_names = [f"leaf_{index:03d}" for index in range(len(leaf_names))]
        if leaf_names != expected_names:
            raise ValueError(
                "checkpoint leaves must be contiguous leaf_000, leaf_001, ..."
            )
        unexpected = sorted(set(archive.files) - set(leaf_names) - {"metadata_json"})
        if unexpected:
            raise ValueError(
                f"checkpoint contains unsupported arrays: {', '.join(unexpected)}"
            )
        leaves = [np.asarray(archive[name]) for name in leaf_names]

    if not isinstance(metadata, dict):
        raise ValueError("checkpoint metadata_json must decode to an object")
    if metadata.get("schema_version") != CHECKPOINT_SCHEMA_VERSION:
        raise ValueError(
            f"checkpoint schema_version must be {CHECKPOINT_SCHEMA_VERSION}"
        )
    required = {
        "dtype",
        "architecture",
        "geometry",
        "physics",
        "ansatz",
        "collocation",
        "training",
        "stage",
        "step",
    }
    missing = sorted(required - set(metadata))
    if missing:
        raise ValueError(f"checkpoint metadata is missing: {', '.join(missing)}")
    _validate_leaves(leaves, metadata)
    return leaves, metadata, sha256(source)


def _normalized(value: Any) -> Any:
    """Normalize tuples and mapping order through canonical JSON."""

    return json.loads(json.dumps(value, sort_keys=True))


def validate_resume(run: config.RunConfig, metadata: Mapping[str, Any]) -> None:
    """Reject changes to the scientific/model identity of a continuation run."""

    expected = run.metadata_sections()
    mismatches = []
    for field in IMMUTABLE_METADATA_FIELDS:
        if _normalized(metadata.get(field)) != _normalized(expected[field]):
            mismatches.append(field)
    if mismatches:
        raise ValueError(
            "resume checkpoint is incompatible in immutable fields: "
            + ", ".join(mismatches)
        )
