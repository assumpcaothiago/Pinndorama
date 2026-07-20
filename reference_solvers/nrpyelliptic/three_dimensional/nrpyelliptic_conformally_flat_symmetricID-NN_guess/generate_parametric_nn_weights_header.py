#!/usr/bin/env python3
"""Export a publication checkpoint's fixed w40d4 network as a C header.

The input is the version-1 clean-break Pinndorama ``.npz`` checkpoint format:
model leaves are named ``leaf_000``, ``leaf_001``, ... in pytree order and
canonical UTF-8 JSON metadata is embedded as a one-dimensional ``uint8`` array
named ``metadata_json``.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import tempfile
from collections.abc import Mapping
from pathlib import Path
from typing import Any

import numpy as np

EXPECTED_LAYERS = [3, 40, 40, 40, 40, 1]
EXPECTED_LEAF_COUNT = 2 * (len(EXPECTED_LAYERS) - 1)
FIXED_SOURCE_NAMES = (
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
EXPECTED_GEOMETRY = {
    "coordinate_system": "SinhSymTP",
    "AMAX": 1.0e6,
    "bScale": 2.5,
    "SINHWAA": 0.07,
}
EXPECTED_PHYSICS = {
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
    "S1_x": 0.0,
    "S1_y": 0.0,
}
EXPECTED_EQUAL_SPIN_SZ = {
    "minimum": -0.2,
    "maximum": 0.2,
    "sampling": "cell_centered",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--checkpoint",
        type=Path,
        required=True,
        help="Publication parametric SinhSymTP .npz checkpoint.",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("neural_net_weights.h"),
        help="Generated C header path.",
    )
    return parser.parse_args()


def sha256_bytes(payload: bytes) -> str:
    return hashlib.sha256(payload).hexdigest()


def decode_metadata_json(value: np.ndarray) -> bytes:
    """Decode the sole metadata representation in checkpoint schema version 1."""

    array = np.asarray(value)
    if array.dtype != np.dtype(np.uint8) or array.ndim != 1:
        raise ValueError("metadata_json must be a one-dimensional uint8 array")
    return array.tobytes()


def load_metadata(
    checkpoint: Mapping[str, np.ndarray],
) -> tuple[dict[str, Any], bytes, str]:
    if "metadata_json" not in checkpoint:
        raise ValueError("checkpoint is missing embedded metadata_json")
    raw = decode_metadata_json(checkpoint["metadata_json"])
    origin = "embedded metadata_json"
    try:
        metadata = json.loads(raw.decode("utf-8"))
    except (UnicodeDecodeError, json.JSONDecodeError) as exc:
        raise ValueError(f"Invalid UTF-8 JSON metadata from {origin}: {exc}") from exc
    if not isinstance(metadata, dict):
        raise ValueError("Checkpoint metadata must be a JSON object")
    return metadata, raw, origin


def require_canonical_table(
    metadata: Mapping[str, Any], name: str, expected_keys: set[str]
) -> Mapping[str, Any]:
    value = metadata.get(name)
    if not isinstance(value, Mapping):
        raise ValueError(f"checkpoint metadata.{name} must be a canonical JSON object")
    if set(value) != expected_keys:
        raise ValueError(
            f"checkpoint metadata.{name} must contain exactly "
            f"{sorted(expected_keys)}; got {sorted(value)}"
        )
    return value


def require_exact_number(value: Any, expected: float, path: str) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"checkpoint {path} must be the number {expected!r}")
    result = float(value)
    if not np.isfinite(result) or result != expected:
        raise ValueError(f"checkpoint {path} must be {expected!r}; got {value!r}")
    return result


def validate_metadata(metadata: Mapping[str, Any]) -> dict[str, Any]:
    if (
        type(metadata.get("schema_version")) is not int
        or metadata["schema_version"] != 1
    ):
        raise ValueError("checkpoint metadata.schema_version must be the integer 1")
    if metadata.get("dtype") != "float64":
        raise ValueError("checkpoint metadata.dtype must be exactly 'float64'")

    architecture = require_canonical_table(
        metadata, "architecture", {"layers", "activation"}
    )
    layers_value = architecture["layers"]
    if not isinstance(layers_value, list) or any(
        type(value) is not int for value in layers_value
    ):
        raise ValueError("checkpoint architecture.layers must be an integer list")
    layers = list(layers_value)
    if layers != EXPECTED_LAYERS:
        raise ValueError(
            f"NN warm start is fixed to w40d4 {EXPECTED_LAYERS}; got {layers}"
        )
    if architecture["activation"] != "tanh":
        raise ValueError("checkpoint architecture.activation must be exactly 'tanh'")

    ansatz = require_canonical_table(metadata, "ansatz", {"output_transform"})
    if ansatz["output_transform"] != "smooth_inverse_radius":
        raise ValueError(
            "checkpoint ansatz.output_transform must be exactly "
            "'smooth_inverse_radius'"
        )

    geometry_metadata = require_canonical_table(
        metadata, "geometry", set(EXPECTED_GEOMETRY)
    )
    if geometry_metadata["coordinate_system"] != "SinhSymTP":
        raise ValueError(
            "checkpoint geometry.coordinate_system must be exactly 'SinhSymTP'"
        )
    geometry = {
        name: require_exact_number(
            geometry_metadata[name], expected, f"geometry.{name}"
        )
        for name, expected in EXPECTED_GEOMETRY.items()
        if name != "coordinate_system"
    }

    physics_metadata = require_canonical_table(
        metadata, "physics", set(EXPECTED_PHYSICS)
    )
    source = {
        name: require_exact_number(physics_metadata[name], expected, f"physics.{name}")
        for name, expected in EXPECTED_PHYSICS.items()
    }

    parameter = require_canonical_table(metadata, "parameter", {"equal_spin_sz"})
    equal_spin = parameter["equal_spin_sz"]
    if not isinstance(equal_spin, Mapping) or set(equal_spin) != set(
        EXPECTED_EQUAL_SPIN_SZ
    ):
        raise ValueError(
            "checkpoint parameter.equal_spin_sz must contain exactly "
            "minimum, maximum, and sampling"
        )
    spin_min = require_exact_number(
        equal_spin["minimum"],
        EXPECTED_EQUAL_SPIN_SZ["minimum"],
        "parameter.equal_spin_sz.minimum",
    )
    spin_max = require_exact_number(
        equal_spin["maximum"],
        EXPECTED_EQUAL_SPIN_SZ["maximum"],
        "parameter.equal_spin_sz.maximum",
    )
    if equal_spin["sampling"] != "cell_centered":
        raise ValueError(
            "checkpoint parameter.equal_spin_sz.sampling must be exactly "
            "'cell_centered'"
        )

    return {
        "layers": layers,
        "dtype": "float64",
        "activation": "tanh",
        "output_transform": "smooth_inverse_radius",
        "geometry": geometry,
        "source": source,
        "spin_min": spin_min,
        "spin_max": spin_max,
        "sampling": equal_spin["sampling"],
    }


def ordered_model_leaves(checkpoint: Mapping[str, np.ndarray]) -> list[np.ndarray]:
    leaf_names = sorted(
        (name for name in checkpoint if name.startswith("leaf_")),
        key=lambda name: int(name.removeprefix("leaf_")),
    )
    expected_names = [f"leaf_{index:03d}" for index in range(EXPECTED_LEAF_COUNT)]
    if leaf_names != expected_names:
        raise ValueError(
            "Expected exactly the ten w40d4 model arrays "
            f"{expected_names[0]}...{expected_names[-1]}; got {leaf_names}"
        )
    unexpected = set(checkpoint) - set(expected_names) - {"metadata_json"}
    if unexpected:
        raise ValueError(
            "checkpoint contains unsupported arrays: " + ", ".join(sorted(unexpected))
        )
    leaves: list[np.ndarray] = []
    for name in leaf_names:
        value = np.asarray(checkpoint[name])
        if value.dtype != np.dtype(np.float64):
            raise ValueError(f"{name} must be stored as float64; got {value.dtype}")
        if not np.all(np.isfinite(value)):
            raise ValueError(f"{name} contains non-finite values")
        leaves.append(value)
    return leaves


def dense_layers(leaves: list[np.ndarray]) -> list[tuple[np.ndarray, np.ndarray]]:
    result: list[tuple[np.ndarray, np.ndarray]] = []
    for layer_index, (in_dim, out_dim) in enumerate(
        zip(EXPECTED_LAYERS[:-1], EXPECTED_LAYERS[1:])
    ):
        weights = leaves[2 * layer_index]
        bias = leaves[2 * layer_index + 1]
        if bias.shape != (out_dim,):
            raise ValueError(
                f"Layer {layer_index} bias must have shape {(out_dim,)}, got {bias.shape}"
            )
        expected_weight_shape = (out_dim, in_dim)
        if weights.shape != expected_weight_shape:
            raise ValueError(
                f"Layer {layer_index} weights must use Equinox (out, in) shape "
                f"{expected_weight_shape}; got {weights.shape}"
            )
        # The C evaluator uses row-vector x @ W, so transpose the required
        # Equinox Linear (out_features, in_features) checkpoint representation.
        result.append((weights.T, bias))
    return result


def c_number(value: float) -> str:
    return f"{float(value):.17e}"


def format_vector(name: str, values: np.ndarray) -> str:
    entries = ", ".join(c_number(value) for value in values.reshape(-1))
    return f"static const REAL {name}[{values.size}] = {{{entries}}};\n"


def format_matrix(name: str, values: np.ndarray) -> str:
    rows = ["  {" + ", ".join(c_number(value) for value in row) + "}" for row in values]
    return (
        f"static const REAL {name}[{values.shape[0]}][{values.shape[1]}] = {{\n"
        + ",\n".join(rows)
        + "\n};\n"
    )


def generate_header(
    layers: list[tuple[np.ndarray, np.ndarray]],
    validated: Mapping[str, Any],
    checkpoint_hash: str,
    metadata_hash: str,
    checkpoint_name: str,
    metadata_origin: str,
) -> str:
    geometry = validated["geometry"]
    source = validated["source"]
    dimensions = validated["layers"]
    lines = [
        "#ifndef NEURAL_NET_WEIGHTS_H\n",
        "#define NEURAL_NET_WEIGHTS_H\n\n",
        "/* Generated by generate_parametric_nn_weights_header.py. */\n",
        "/* Do not edit numeric constants by hand. */\n",
        f"/* Checkpoint: {checkpoint_name} */\n",
        f"/* Metadata source: {metadata_origin} */\n\n",
        "#define PINNDORAMA_GENERATED_PARAMETRIC_NN_HEADER 1\n",
        "#define NN_HEADER_SCHEMA_VERSION 1\n",
        f"#define NN_NUM_LAYERS {len(layers)}\n",
        f"#define NN_INPUT_DIM {dimensions[0]}\n",
        "#define NN_WIDTH 40\n",
        f"#define NN_OUTPUT_DIM {dimensions[-1]}\n\n",
        f'static const char NN_CHECKPOINT_SHA256[] = "{checkpoint_hash}";\n',
        f'static const char NN_METADATA_SHA256[] = "{metadata_hash}";\n',
        'static const char NN_DTYPE[] = "float64";\n',
        'static const char NN_ACTIVATION[] = "tanh";\n',
        'static const char NN_OUTPUT_TRANSFORM[] = "smooth_inverse_radius";\n',
        'static const char NN_EQUAL_SPIN_SZ_SAMPLING[] = "cell_centered";\n',
        'static const char NN_COORDINATE_SYSTEM[] = "SinhSymTP";\n\n',
        f"static const REAL NN_CHECKPOINT_AMAX = {c_number(geometry['AMAX'])};\n",
        f"static const REAL NN_CHECKPOINT_BSCALE = {c_number(geometry['bScale'])};\n",
        f"static const REAL NN_CHECKPOINT_SINHWAA = {c_number(geometry['SINHWAA'])};\n",
        f"static const REAL NN_EQUAL_SPIN_SZ_MIN = {c_number(validated['spin_min'])};\n",
        f"static const REAL NN_EQUAL_SPIN_SZ_MAX = {c_number(validated['spin_max'])};\n\n",
    ]
    for name in FIXED_SOURCE_NAMES:
        lines.append(
            f"static const REAL NN_SOURCE_{name.upper()} = {c_number(source[name])};\n"
        )
    lines.extend(
        [
            "\n",
            f"static const int NN_LAYER_DIMS[{len(dimensions)}] = {{",
            ", ".join(str(dimension) for dimension in dimensions),
            "};\n\n",
        ]
    )
    for index, (weights, bias) in enumerate(layers):
        lines.append(format_matrix(f"NN_W{index}", weights))
        lines.append(format_vector(f"NN_B{index}", bias))
        lines.append("\n")
    lines.append("#endif /* NEURAL_NET_WEIGHTS_H */\n")
    return "".join(lines)


def atomic_write(path: Path, content: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    descriptor, temporary_name = tempfile.mkstemp(
        prefix=f".{path.name}.", suffix=".tmp", dir=path.parent, text=True
    )
    try:
        with os.fdopen(descriptor, "w", encoding="utf-8", newline="\n") as stream:
            stream.write(content)
            stream.flush()
            os.fsync(stream.fileno())
        os.replace(temporary_name, path)
    except BaseException:
        try:
            os.unlink(temporary_name)
        except FileNotFoundError:
            pass
        raise


def main() -> int:
    args = parse_args()
    checkpoint_bytes = args.checkpoint.read_bytes()
    with np.load(args.checkpoint, allow_pickle=False) as checkpoint:
        metadata, metadata_bytes, metadata_origin = load_metadata(checkpoint)
        validated = validate_metadata(metadata)
        layers = dense_layers(ordered_model_leaves(checkpoint))

    header = generate_header(
        layers,
        validated,
        checkpoint_hash=sha256_bytes(checkpoint_bytes),
        metadata_hash=sha256_bytes(metadata_bytes),
        checkpoint_name=args.checkpoint.name,
        metadata_origin=metadata_origin,
    )
    atomic_write(args.output, header)
    print(
        f"Wrote {args.output} from {args.checkpoint} "
        f"(sha256={sha256_bytes(checkpoint_bytes)})"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
