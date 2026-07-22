from __future__ import annotations

import copy
import hashlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
WARM_DIR = (
    REPO_ROOT
    / "reference_solvers"
    / "nrpyelliptic"
    / "nrpyelliptic_conformally_flat_symmetricID-NN_guess"
)
EXPORTER = WARM_DIR / "generate_parametric_nn_weights_header.py"
LAYERS = [3, 40, 40, 40, 40, 1]


def _metadata() -> dict[str, object]:
    return {
        "schema_version": 1,
        "dtype": "float64",
        "architecture": {"layers": LAYERS, "activation": "tanh"},
        "geometry": {
            "coordinate_system": "SinhSymTP",
            "AMAX": 1.0e6,
            "bScale": 2.5,
            "SINHWAA": 0.07,
        },
        "physics": {
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
        },
        "parameter": {
            "equal_spin_sz": {
                "minimum": -0.2,
                "maximum": 0.2,
                "sampling": "cell_centered",
            }
        },
        "ansatz": {"output_transform": "smooth_inverse_radius"},
        # Current clean-break checkpoints carry additional unrelated top-level
        # run state. The exporter deliberately ignores these fields.
        "stage": "adam",
        "step": 1,
        "collocation": {
            "Nxx0": 128,
            "Nxx1": 64,
            "equal_spin_sz_points": 80,
        },
    }


def _write_checkpoint(path: Path) -> list[np.ndarray]:
    rng = np.random.default_rng(1701)
    leaves: list[np.ndarray] = []
    arrays: dict[str, np.ndarray] = {}
    for index, (input_dim, output_dim) in enumerate(zip(LAYERS[:-1], LAYERS[1:])):
        weights = rng.normal(scale=0.03, size=(output_dim, input_dim)).astype(
            np.float64
        )
        bias = rng.normal(scale=0.02, size=(output_dim,)).astype(np.float64)
        leaves.extend((weights, bias))
        arrays[f"leaf_{2 * index:03d}"] = weights
        arrays[f"leaf_{2 * index + 1:03d}"] = bias
    metadata_bytes = json.dumps(_metadata(), sort_keys=True).encode("utf-8")
    arrays["metadata_json"] = np.frombuffer(metadata_bytes, dtype=np.uint8)
    np.savez(path, **arrays)
    return leaves


def _numpy_prediction(
    leaves: list[np.ndarray], xx0: float, xx1: float, spin: float
) -> float:
    value = np.array([xx0, xx1, spin], dtype=np.float64)
    for layer in range(4):
        value = np.tanh(leaves[2 * layer] @ value + leaves[2 * layer + 1])
    raw = float((leaves[8] @ value + leaves[9])[0])
    radial = 1.0e6 * np.sinh(xx0 / 0.07) / np.sinh(1.0 / 0.07)
    radius = np.sqrt(radial * radial + 2.5**2 * np.cos(xx1) ** 2)
    return raw / np.sqrt(1.0 + radius * radius)


def _export(checkpoint: Path, header: Path) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(EXPORTER),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(header),
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )


def _attempt_export(
    checkpoint: Path, header: Path, *extra_arguments: str
) -> subprocess.CompletedProcess[str]:
    return subprocess.run(
        [
            sys.executable,
            str(EXPORTER),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(header),
            *extra_arguments,
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
    )


def _checkpoint_arrays(path: Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as archive:
        return {name: np.array(archive[name], copy=True) for name in archive.files}


def _replace_metadata(
    path: Path,
    metadata: dict[str, object],
    *,
    encoding: str = "uint8",
) -> None:
    arrays = _checkpoint_arrays(path)
    raw = json.dumps(metadata, sort_keys=True).encode("utf-8")
    if encoding == "uint8":
        metadata_array = np.frombuffer(raw, dtype=np.uint8).copy()
    elif encoding == "uint8_matrix":
        metadata_array = np.frombuffer(raw, dtype=np.uint8).reshape(1, -1).copy()
    elif encoding == "unicode_scalar":
        metadata_array = np.asarray(raw.decode("utf-8"))
    else:  # pragma: no cover - test helper misuse
        raise AssertionError(f"unsupported test encoding: {encoding}")
    arrays["metadata_json"] = metadata_array
    np.savez(path, **arrays)


def test_exporter_accepts_actual_p001_checkpoint_schema(tmp_path: Path) -> None:
    checkpoint_path = tmp_path / "actual-p001.npz"
    script = r"""
import importlib
import pathlib
import sys
import numpy as np

config_path = pathlib.Path(sys.argv[1])
output = pathlib.Path(sys.argv[2])
checkpoint = importlib.import_module(
    "pinndorama.solvers.punctures_2d_parametric.checkpoint"
)
config = importlib.import_module(
    "pinndorama.solvers.punctures_2d_parametric.config"
)

cfg = config.load_config(config_path)
rng = np.random.default_rng(1702)
layers = cfg["architecture"]["layers"]
params = [
    {
        "W": rng.normal(size=(out_dim, in_dim)).astype(np.float64),
        "b": rng.normal(size=(out_dim,)).astype(np.float64),
    }
    for in_dim, out_dim in zip(layers[:-1], layers[1:])
]
metadata = checkpoint.build_metadata(
    cfg,
    stage="adam",
    step=1,
    parent_checkpoint_sha256=None,
    loss_value=1.0,
)
checkpoint.save_checkpoint(output, params, metadata)
"""
    environment = os.environ.copy()
    environment["PYTHONPYCACHEPREFIX"] = str(tmp_path / "pycache")
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(REPO_ROOT / "configs/punctures_2d_parametric/P001_w40d4.toml"),
            str(checkpoint_path),
        ],
        cwd=REPO_ROOT,
        env=environment,
        check=True,
    )
    header_path = tmp_path / "neural_net_weights.h"
    _export(checkpoint_path, header_path)
    header = header_path.read_text(encoding="utf-8")
    assert hashlib.sha256(checkpoint_path.read_bytes()).hexdigest() in header
    assert 'NN_COORDINATE_SYSTEM[] = "SinhSymTP"' in header


@pytest.mark.slow
def test_parametric_checkpoint_export_and_c_numpy_parity(tmp_path: Path) -> None:
    compiler = shutil.which("cc")
    if compiler is None:
        pytest.skip("C compiler is unavailable")
    checkpoint = tmp_path / "parametric.npz"
    header = tmp_path / "neural_net_weights.h"
    leaves = _write_checkpoint(checkpoint)
    _export(checkpoint, header)
    header_text = header.read_text(encoding="utf-8")
    assert hashlib.sha256(checkpoint.read_bytes()).hexdigest() in header_text
    assert "PINNDORAMA_GENERATED_PARAMETRIC_NN_HEADER" in header_text
    assert 'NN_DTYPE[] = "float64"' in header_text

    harness = tmp_path / "evaluate.c"
    harness.write_text(
        """
#include "BHaH_defines.h"
#include "BHaH_function_prototypes.h"
int main(int argc, const char **argv) {
  if (argc != 4) return 2;
  printf("%.17e\\n", eval_uNN(strtod(argv[1], NULL), strtod(argv[2], NULL), strtod(argv[3], NULL)));
  return 0;
}
""",
        encoding="utf-8",
    )
    executable = tmp_path / "evaluate"
    subprocess.run(
        [
            compiler,
            "-std=gnu99",
            "-O2",
            "-I",
            str(WARM_DIR),
            "-I",
            str(tmp_path),
            str(WARM_DIR / "neural_net_guess_single_point.c"),
            str(harness),
            "-lm",
            "-o",
            str(executable),
        ],
        check=True,
    )
    points = (
        (0.0009, 1.0e-5, -0.1975),
        (0.2, 0.7, -0.1),
        (0.5, np.pi / 2.0, 0.0),
        (0.8, 2.4, 0.1),
        (1.0, np.pi - 1.0e-5, 0.1975),
    )
    for xx0, xx1, spin in points:
        result = subprocess.run(
            [str(executable), repr(xx0), repr(xx1), repr(spin)],
            text=True,
            capture_output=True,
            check=True,
        )
        c_value = float(result.stdout)
        numpy_value = _numpy_prediction(leaves, xx0, xx1, spin)
        assert c_value == pytest.approx(numpy_value, rel=2.0e-14, abs=1.0e-16)


def test_exporter_rejects_non_float64_checkpoint(tmp_path: Path) -> None:
    checkpoint = tmp_path / "invalid.npz"
    _write_checkpoint(checkpoint)
    with np.load(checkpoint, allow_pickle=False) as archive:
        arrays = {name: archive[name] for name in archive.files}
    arrays["leaf_000"] = arrays["leaf_000"].astype(np.float32)
    np.savez(checkpoint, **arrays)
    result = subprocess.run(
        [
            sys.executable,
            str(EXPORTER),
            "--checkpoint",
            str(checkpoint),
            "--output",
            str(tmp_path / "header.h"),
        ],
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "must be stored as float64" in result.stderr


@pytest.mark.parametrize("encoding", ("unicode_scalar", "uint8_matrix"))
def test_exporter_rejects_legacy_metadata_encodings(
    tmp_path: Path, encoding: str
) -> None:
    checkpoint = tmp_path / f"legacy-{encoding}.npz"
    _write_checkpoint(checkpoint)
    _replace_metadata(checkpoint, _metadata(), encoding=encoding)
    result = _attempt_export(checkpoint, tmp_path / "header.h")
    assert result.returncode != 0
    assert "metadata_json must be a one-dimensional uint8 array" in result.stderr


@pytest.mark.parametrize(
    ("variant", "error_fragment"),
    (
        ("schema_string", "schema_version"),
        ("dtype_alias", "metadata.dtype"),
        ("architecture_alias", "metadata.architecture"),
        ("geometry_alias", "metadata.geometry"),
        ("physics_value", "physics.bare_mass_0"),
        ("transform_alias", "ansatz.output_transform"),
        ("sampling_spelling", "equal_spin_sz.sampling"),
    ),
)
def test_exporter_rejects_legacy_or_non_p001_metadata(
    tmp_path: Path, variant: str, error_fragment: str
) -> None:
    checkpoint = tmp_path / f"legacy-{variant}.npz"
    _write_checkpoint(checkpoint)
    metadata = copy.deepcopy(_metadata())
    if variant == "schema_string":
        metadata["schema_version"] = "1"
    elif variant == "dtype_alias":
        metadata["dtype"] = "double"
    elif variant == "architecture_alias":
        architecture = metadata.pop("architecture")
        metadata["model"] = architecture
    elif variant == "geometry_alias":
        geometry = metadata["geometry"]
        assert isinstance(geometry, dict)
        geometry["bscale"] = geometry.pop("bScale")
    elif variant == "physics_value":
        physics = metadata["physics"]
        assert isinstance(physics, dict)
        physics["bare_mass_0"] = 0.6
    elif variant == "transform_alias":
        ansatz = metadata["ansatz"]
        assert isinstance(ansatz, dict)
        ansatz["output_transform"] = "raw_over_sqrt_1_plus_r2"
    elif variant == "sampling_spelling":
        parameter = metadata["parameter"]
        assert isinstance(parameter, dict)
        equal_spin = parameter["equal_spin_sz"]
        assert isinstance(equal_spin, dict)
        equal_spin["sampling"] = "cell_centred"
    else:  # pragma: no cover - parametrization misuse
        raise AssertionError(variant)
    _replace_metadata(checkpoint, metadata)
    result = _attempt_export(checkpoint, tmp_path / "header.h")
    assert result.returncode != 0
    assert error_fragment in result.stderr


def test_exporter_rejects_legacy_in_out_weight_layout(tmp_path: Path) -> None:
    checkpoint = tmp_path / "legacy-in-out.npz"
    _write_checkpoint(checkpoint)
    arrays = _checkpoint_arrays(checkpoint)
    for index in range(0, 10, 2):
        arrays[f"leaf_{index:03d}"] = arrays[f"leaf_{index:03d}"].T
    np.savez(checkpoint, **arrays)
    result = _attempt_export(checkpoint, tmp_path / "header.h")
    assert result.returncode != 0
    assert "must use Equinox (out, in) shape" in result.stderr


def test_exporter_has_no_metadata_sidecar_interface(tmp_path: Path) -> None:
    checkpoint = tmp_path / "parametric.npz"
    _write_checkpoint(checkpoint)
    sidecar = tmp_path / "metadata.json"
    sidecar.write_text(json.dumps(_metadata()), encoding="utf-8")
    result = _attempt_export(
        checkpoint,
        tmp_path / "header.h",
        "--metadata",
        str(sidecar),
    )
    assert result.returncode == 2
    assert "unrecognized arguments: --metadata" in result.stderr


@pytest.mark.slow
def test_zero_nn_selection_and_invalid_physics_rejection(tmp_path: Path) -> None:
    if shutil.which("make") is None or shutil.which("cc") is None:
        pytest.skip("C build tools are unavailable")
    checkpoint = tmp_path / "parametric.npz"
    generated_header = tmp_path / "neural_net_weights.h"
    _write_checkpoint(checkpoint)
    _export(checkpoint, generated_header)
    active_header = WARM_DIR / "neural_net_weights.h"
    if active_header.exists():
        raise AssertionError(
            "generated warm-start header must not be present initially"
        )
    try:
        shutil.copyfile(generated_header, active_header)
        subprocess.run(["make", "-j2"], cwd=WARM_DIR, check=True)
        executable = WARM_DIR / "nrpyelliptic_conformally_flat"

        zero_text = (
            (WARM_DIR / "nrpyelliptic_conformally_flat_zero.par")
            .read_text(encoding="utf-8")
            .replace("t_final = 1000000.0", "t_final = 0.0")
        )
        nn_text = (
            (WARM_DIR / "nrpyelliptic_conformally_flat_equal_spin_nn.par")
            .read_text(encoding="utf-8")
            .replace("t_final = 1000000.0", "t_final = 0.0")
        )
        zero_dir = tmp_path / "zero"
        nn_dir = tmp_path / "nn"
        invalid_dir = tmp_path / "invalid"
        zero_dir.mkdir()
        nn_dir.mkdir()
        invalid_dir.mkdir()
        zero_par = zero_dir / "run.par"
        nn_par = nn_dir / "run.par"
        invalid_par = invalid_dir / "run.par"
        zero_par.write_text(zero_text, encoding="utf-8")
        nn_par.write_text(nn_text, encoding="utf-8")
        invalid_par.write_text(
            nn_text.replace("bare_mass_0 = 0.5", "bare_mass_0 = 0.6"),
            encoding="utf-8",
        )

        zero = subprocess.run(
            [str(executable), str(zero_par)],
            cwd=zero_dir,
            text=True,
            capture_output=True,
            check=True,
        )
        assert "Using zero initial guess" in zero.stderr
        nn = subprocess.run(
            [str(executable), str(nn_par)],
            cwd=nn_dir,
            text=True,
            capture_output=True,
            check=True,
        )
        assert "Using parametric NN warm start" in nn.stderr
        invalid = subprocess.run(
            [str(executable), str(invalid_par)],
            cwd=invalid_dir,
            text=True,
            capture_output=True,
        )
        assert invalid.returncode != 0
        assert "differs from NN checkpoint" in invalid.stderr

        initial_data = (WARM_DIR / "initial_data.c").read_text(encoding="utf-8")
        assert initial_data.index("read_checkpoint") < initial_data.index(
            "strcmp(commondata->initial_guess"
        )
    finally:
        subprocess.run(["make", "clean"], cwd=WARM_DIR, check=False)
        active_header.unlink(missing_ok=True)
