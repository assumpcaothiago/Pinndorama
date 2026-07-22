from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pinndorama.solvers.punctures_2d import checkpoint, config, coordinates, model
from pinndorama.solvers.punctures_2d.relative_l2_nrpy import (
    _metric_components,
    build_evaluation_grid,
    common_grid_key,
    discover_checkpoints,
    interpolate_reference_once,
    load_checkpoint_records,
)


REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG = REPO_ROOT / "configs" / "punctures_2d" / "C004_square_w40d4.toml"


def _zero_params(layers: list[int]) -> list[dict[str, np.ndarray]]:
    return [
        {
            "W": np.zeros((out_size, in_size), dtype=np.float64),
            "b": np.zeros((out_size,), dtype=np.float64),
        }
        for in_size, out_size in zip(layers[:-1], layers[1:])
    ]


def _write_checkpoint(path: Path, nxx0: int, nxx1: int) -> None:
    solver_config = config.load_config(CONFIG)
    metadata = checkpoint.build_metadata(solver_config, stage="complete", step=50000)
    metadata["collocation"]["Nxx0"] = nxx0
    metadata["collocation"]["Nxx1"] = nxx1
    checkpoint.save_checkpoint(
        path, _zero_params(solver_config.architecture["layers"]), metadata
    )


def test_discovers_checkpoints_and_loads_native_grid_metadata(tmp_path: Path) -> None:
    first = tmp_path / "N80" / "seed503" / "checkpoint_final.npz"
    second = tmp_path / "N86" / "seed504" / "checkpoint_final.npz"
    first.parent.mkdir(parents=True)
    second.parent.mkdir(parents=True)
    _write_checkpoint(first, 80, 80)
    _write_checkpoint(second, 86, 86)

    paths = discover_checkpoints([tmp_path])
    assert paths == sorted([first.resolve(), second.resolve()], key=str)
    records = load_checkpoint_records(paths, config.load_config(CONFIG))
    assert [record.native_grid_key.nxx0 for record in records] == [80, 86]
    assert [record.metadata["training"]["seed"] for record in records] == [503, 503]


def test_evaluation_grid_is_cell_centered_masked_and_weighted() -> None:
    solver_config = config.load_config(CONFIG)
    key = common_grid_key(solver_config, 32, 24)
    grid = build_evaluation_grid(key, solver_config.geometry, 7.5)

    assert grid.native.shape[1] == 2
    assert grid.cartesian.shape == (grid.native.shape[0], 3)
    assert grid.weights.shape == (grid.native.shape[0],)
    assert 0 < grid.native.shape[0] < 32 * 24
    assert np.all(grid.native[:, 0] > key.xx0_min)
    assert np.all(grid.native[:, 0] < key.xx0_max)
    assert np.all(grid.weights > 0.0)


def test_all_unique_grids_use_one_reader_invocation(tmp_path: Path) -> None:
    solver_config = config.load_config(CONFIG)
    keys = (
        common_grid_key(solver_config, 12, 10),
        common_grid_key(solver_config, 14, 12),
    )
    grids = {
        key: build_evaluation_grid(key, solver_config.geometry, 7.5) for key in keys
    }
    reader = tmp_path / "mock_reader.py"
    reader.write_text(
        """#!/usr/bin/env python3
import argparse
from pathlib import Path
import numpy as np
p = argparse.ArgumentParser()
p.add_argument('--binary')
p.add_argument('--coords')
p.add_argument('--output')
a = p.parse_args()
count = Path(__file__).with_name('reader_count.txt')
count.write_text(str(int(count.read_text()) + 1) if count.exists() else '1')
coords = np.loadtxt(a.coords, ndmin=2)
values = coords[:, 0] - 2.0 * coords[:, 1] + 3.0 * coords[:, 2]
np.savetxt(a.output, np.column_stack((coords, values)), header='x y z uu')
""",
        encoding="utf-8",
    )
    reader.chmod(0o700)
    binary = tmp_path / "unused.bin"
    binary.write_bytes(b"unused by mock")

    values = interpolate_reference_once(
        reader, binary, grids, temporary_parent=tmp_path
    )

    assert (tmp_path / "reader_count.txt").read_text(encoding="utf-8") == "1"
    assert set(values) == set(keys)
    for key in keys:
        cartesian = grids[key].cartesian
        expected = cartesian[:, 0] - 2.0 * cartesian[:, 1] + 3.0 * cartesian[:, 2]
        np.testing.assert_allclose(values[key], expected, rtol=0.0, atol=1.0e-14)


def test_reader_coordinate_reordering_is_rejected(tmp_path: Path) -> None:
    solver_config = config.load_config(CONFIG)
    key = common_grid_key(solver_config, 12, 10)
    grids = {key: build_evaluation_grid(key, solver_config.geometry, 7.5)}
    reader = tmp_path / "reordering_reader.py"
    reader.write_text(
        """#!/usr/bin/env python3
import argparse
import numpy as np
p = argparse.ArgumentParser()
p.add_argument('--binary')
p.add_argument('--coords')
p.add_argument('--output')
a = p.parse_args()
coords = np.loadtxt(a.coords, ndmin=2)[::-1]
np.savetxt(a.output, np.column_stack((coords, np.ones(coords.shape[0]))))
""",
        encoding="utf-8",
    )
    reader.chmod(0o700)
    binary = tmp_path / "unused.bin"
    binary.write_bytes(b"unused by mock")

    with pytest.raises(ValueError, match="preserve input order"):
        interpolate_reference_once(reader, binary, grids, temporary_parent=tmp_path)


def test_metric_components_match_manual_weighted_relative_l2() -> None:
    prediction = np.array([1.0, 3.0])
    reference = np.array([1.0, 2.0])
    weights = np.array([2.0, 4.0])
    numerator, denominator, relative_l2 = _metric_components(
        prediction, reference, weights
    )
    assert numerator == pytest.approx(4.0)
    assert denominator == pytest.approx(18.0)
    assert relative_l2 == pytest.approx(np.sqrt(4.0 / 18.0))


def test_numpy_checkpoint_evaluation_applies_radial_ansatz() -> None:
    native = np.array([[0.2, 0.4], [0.3, 1.1]], dtype=np.float64)
    params = [
        {
            "W": np.array([[1.0, 2.0]], dtype=np.float64),
            "b": np.array([0.5], dtype=np.float64),
        }
    ]
    values = model.forward_numpy(
        params, native, amax=10.0, bscale=2.5, sinhwaa=0.7
    ).reshape(-1)
    radius = coordinates.spherical_radius_from_xx(
        native[:, 0],
        native[:, 1],
        amax=10.0,
        bscale=2.5,
        sinhwaa=0.7,
        xp=np,
    )
    raw = native[:, 0] + 2.0 * native[:, 1] + 0.5
    np.testing.assert_allclose(values, raw / np.sqrt(1.0 + radius * radius))


def test_invalid_native_sampling_is_rejected(tmp_path: Path) -> None:
    path = tmp_path / "checkpoint_final.npz"
    solver_config = config.load_config(CONFIG)
    metadata = checkpoint.build_metadata(solver_config, stage="complete", step=50000)
    metadata["collocation"]["sampling"] = "two_zone"
    checkpoint.save_checkpoint(
        path, _zero_params(solver_config.architecture["layers"]), metadata
    )
    with pytest.raises(ValueError, match="cell-centered"):
        load_checkpoint_records([path], solver_config)
