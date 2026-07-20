"""Black-box CPU smokes for all neural training entrypoints."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
from typing import Any

import numpy as np
import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


CASES = (
    {
        "name": "toy_problem_1d",
        "evaluator": "toy-problem-1d",
        "evaluation_coords": "0.5\n",
        "module": "pinndorama.solvers.toy_problem_1d.train",
        "config": "configs/toy_problem_1d/T001_w40d4.toml",
        "overrides": (
            "--Nxx0",
            "2",
            "--adam-steps",
            "1",
            "--ssbroyden-steps",
            "1",
        ),
        "adam_checkpoint": "checkpoint_adam_final.npz",
        "final_stage": "complete",
        "loss_section": "diagnostics",
    },
    {
        "name": "fixed_2d",
        "evaluator": "punctures-2d",
        "evaluation_coords": "0.5 1.0\n",
        "module": "pinndorama.solvers.punctures_2d.train",
        "config": "configs/punctures_2d/C002_w40d4.toml",
        "overrides": (
            "--Nxx0",
            "2",
            "--Nxx1",
            "2",
            "--adam-steps",
            "1",
            "--ssbroyden-steps",
            "1",
        ),
        "adam_checkpoint": "checkpoint_adam_final.npz",
        "final_stage": "complete",
        "loss_section": "diagnostics",
    },
    {
        "name": "parametric_2d",
        "evaluator": "punctures-2d-parametric",
        "evaluation_coords": "0.5 1.0\n",
        "module": "pinndorama.solvers.punctures_2d_parametric.train",
        "config": "configs/punctures_2d_parametric/P001_w40d4.toml",
        "overrides": (
            "--Nxx0",
            "2",
            "--Nxx1",
            "2",
            "--equal-spin-sz-points",
            "2",
            "--adam-steps",
            "1",
            "--ssbroyden-steps",
            "1",
        ),
        "adam_checkpoint": "checkpoint_adam_final.npz",
        "final_stage": "ssbroyden",
        "loss_section": "loss_value",
    },
    {
        "name": "three_dimensional",
        "evaluator": "punctures-3d",
        "evaluation_coords": "0.5 1.0 0.0\n",
        "module": "pinndorama.solvers.punctures_3d.train",
        "config": "configs/punctures_3d/C04.toml",
        "overrides": (
            "--Nxx0",
            "2",
            "--Nxx1",
            "2",
            "--Nxx2",
            "2",
            "--adam-steps",
            "1",
            "--ssbroyden-steps",
            "1",
        ),
        "adam_checkpoint": "adam_step_00000001.npz",
        "final_stage": "ssbroyden",
        "loss_section": "metrics",
    },
)


def _subprocess_environment(cache_root: Path) -> dict[str, str]:
    environment = os.environ.copy()
    environment.update(
        {
            "CUDA_VISIBLE_DEVICES": "",
            "JAX_ENABLE_X64": "1",
            "JAX_PLATFORM_NAME": "cpu",
            "MPLCONFIGDIR": str(cache_root / "matplotlib"),
            "OMP_NUM_THREADS": "1",
            "OPENBLAS_NUM_THREADS": "1",
            "PYTHONPYCACHEPREFIX": str(cache_root / "pycache"),
            "PYTHONPATH": str(REPO_ROOT / "src"),
            "XDG_CACHE_HOME": str(cache_root / "xdg"),
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
        }
    )
    return environment


def _load_publication_checkpoint(
    path: Path,
) -> tuple[list[np.ndarray], dict[str, Any]]:
    with np.load(path, allow_pickle=False) as archive:
        assert "metadata_json" in archive.files
        metadata_array = np.asarray(archive["metadata_json"])
        assert metadata_array.dtype == np.uint8
        assert metadata_array.ndim == 1
        metadata = json.loads(metadata_array.tobytes().decode("utf-8"))
        leaf_names = sorted(name for name in archive.files if name.startswith("leaf_"))
        assert leaf_names == [f"leaf_{index:03d}" for index in range(len(leaf_names))]
        assert set(archive.files) == set(leaf_names) | {"metadata_json"}
        leaves = [np.array(archive[name], copy=True) for name in leaf_names]
    return leaves, metadata


def _assert_dense_float64_checkpoint(
    leaves: list[np.ndarray], metadata: dict[str, Any]
) -> None:
    layers = metadata["architecture"]["layers"]
    assert len(leaves) == 2 * (len(layers) - 1)
    for index, (input_size, output_size) in enumerate(zip(layers[:-1], layers[1:])):
        weight = leaves[2 * index]
        bias = leaves[2 * index + 1]
        assert weight.dtype == np.float64
        assert bias.dtype == np.float64
        assert weight.shape == (output_size, input_size)
        assert bias.shape == (output_size,)
        assert np.all(np.isfinite(weight))
        assert np.all(np.isfinite(bias))


def _assert_effective_overrides(case: dict[str, Any], metadata: dict[str, Any]) -> None:
    collocation = metadata["collocation"]
    assert not {
        "radial_points",
        "theta_points",
        "phi_points",
        "radial_min",
        "theta_min",
        "phi_min",
    } & set(collocation)
    assert collocation["Nxx0"] == 2
    if case["name"] != "toy_problem_1d":
        assert collocation["Nxx1"] == 2
    if case["name"] == "parametric_2d":
        assert collocation["equal_spin_sz_points"] == 2
        assert metadata["training"]["adam"]["steps"] == 1
        assert metadata["training"]["ssbroyden"]["steps"] == 1
    else:
        assert metadata["training"]["adam_steps"] == 1
        assert metadata["training"]["ssbroyden_steps"] == 1
    if case["name"] == "three_dimensional":
        assert collocation["Nxx2"] == 2
    if case["name"] == "toy_problem_1d":
        assert metadata["loss"]["outer_boundary_condition"] == "second_order_robin"
        assert metadata["loss"]["outer_boundary_region"] == "radial_band"
        assert metadata["loss"]["outer_boundary_r_min"] == 1.0e3
        assert "outer_robin_weight" not in metadata["loss"]
        assert "outer_boundary_loss" in metadata["diagnostics"]["components"]


def _final_loss(case: dict[str, Any], metadata: dict[str, Any]) -> float:
    section = case["loss_section"]
    if section == "diagnostics":
        return float(metadata[section]["total_loss"])
    if section == "metrics":
        return float(metadata[section]["total_loss"])
    return float(metadata[section])


@pytest.mark.jax
@pytest.mark.slow
@pytest.mark.parametrize("case", CASES, ids=lambda case: case["name"])
def test_public_training_cli_adam_and_ssbroyden_smoke(
    case: dict[str, Any], tmp_path: Path
) -> None:
    output_dir = tmp_path / case["name"]
    command = [
        sys.executable,
        "-m",
        case["module"],
        "--config",
        str(REPO_ROOT / case["config"]),
        "--output-dir",
        str(output_dir),
        *case["overrides"],
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=_subprocess_environment(tmp_path / "cache"),
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, (
        f"command failed: {' '.join(command)}\n"
        f"stdout:\n{completed.stdout}\n"
        f"stderr:\n{completed.stderr}"
    )

    assert (output_dir / case["adam_checkpoint"]).is_file()
    final_path = output_dir / "checkpoint_final.npz"
    assert final_path.is_file()
    leaves, metadata = _load_publication_checkpoint(final_path)

    assert metadata["schema_version"] == 1
    assert metadata["dtype"] == "float64"
    assert metadata["architecture"]["activation"] == "tanh"
    assert metadata["ansatz"]["output_transform"] == "smooth_inverse_radius"
    assert metadata["stage"] == case["final_stage"]
    assert metadata["step"] == 1
    assert metadata["parent_checkpoint_sha256"] is None
    assert ("x" + "Punc") not in json.dumps(metadata, sort_keys=True)
    assert math.isfinite(_final_loss(case, metadata))
    _assert_effective_overrides(case, metadata)
    _assert_dense_float64_checkpoint(leaves, metadata)

    coords_path = tmp_path / f"{case['name']}_native_coords.txt"
    values_path = tmp_path / f"{case['name']}_values.txt"
    coords_path.write_text(case["evaluation_coords"], encoding="utf-8")
    evaluate_command = [
        sys.executable,
        "-m",
        "pinndorama.reproducibility.evaluate",
        case["evaluator"],
        "--config",
        str(REPO_ROOT / case["config"]),
        "--checkpoint",
        str(final_path),
        "--coords",
        str(coords_path),
        "--output",
        str(values_path),
    ]
    if case["evaluator"] == "punctures-2d-parametric":
        evaluate_command.extend(("--equal-spin-sz", "0.1"))
    evaluated = subprocess.run(
        evaluate_command,
        cwd=REPO_ROOT,
        env=_subprocess_environment(tmp_path / "evaluation_cache"),
        text=True,
        capture_output=True,
        timeout=60,
        check=False,
    )
    assert evaluated.returncode == 0, evaluated.stderr
    table = np.loadtxt(values_path, comments="#", ndmin=2)
    assert table.shape[0] == 1
    assert math.isfinite(float(table[0, -1]))


@pytest.mark.jax
@pytest.mark.slow
def test_toy_first_order_robin_endpoint_training_smoke(tmp_path: Path) -> None:
    source_config = REPO_ROOT / "configs/toy_problem_1d/T001_w40d4.toml"
    endpoint_config = tmp_path / "T001_first_order_endpoint.toml"
    endpoint_config.write_text(
        source_config.read_text(encoding="utf-8")
        .replace(
            'outer_boundary_condition = "second_order_robin"',
            'outer_boundary_condition = "first_order_robin"',
        )
        .replace(
            'outer_boundary_region = "radial_band"\n' "outer_boundary_r_min = 1.0e3",
            'outer_boundary_region = "endpoint"',
        ),
        encoding="utf-8",
    )
    output_dir = tmp_path / "first_order_endpoint"
    command = [
        sys.executable,
        "-m",
        "pinndorama.solvers.toy_problem_1d.train",
        "--config",
        str(endpoint_config),
        "--output-dir",
        str(output_dir),
        "--Nxx0",
        "2",
        "--adam-steps",
        "1",
        "--ssbroyden-steps",
        "1",
    ]
    completed = subprocess.run(
        command,
        cwd=REPO_ROOT,
        env=_subprocess_environment(tmp_path / "endpoint_cache"),
        text=True,
        capture_output=True,
        timeout=180,
        check=False,
    )
    assert completed.returncode == 0, completed.stderr
    _leaves, metadata = _load_publication_checkpoint(
        output_dir / "checkpoint_final.npz"
    )
    assert metadata["loss"]["outer_boundary_condition"] == "first_order_robin"
    assert metadata["loss"]["outer_boundary_region"] == "endpoint"
    assert "outer_boundary_r_min" not in metadata["loss"]
    assert "outer_boundary_loss" in metadata["diagnostics"]["components"]
