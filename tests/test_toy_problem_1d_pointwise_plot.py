from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest

from pinndorama.solvers.toy_problem_1d import checkpoint
from pinndorama.solvers.toy_problem_1d import plot_pointwise_relative_error as pointwise


def _write_checkpoint(path: Path, *, resolution: int, seed: int) -> Path:
    params = [{"W": np.zeros((1, 1)), "b": np.zeros((1,))}]
    metadata = {
        "schema_version": 1,
        "config_name": "test",
        "geometry": {
            "coordinate_system": "SinhSpherical",
            "AMPL": 1.0e6,
            "SINHW": 0.07,
        },
        "physics": {"m": 1.0},
        "ansatz": {"output_transform": "smooth_inverse_radius"},
        "collocation": {"Nxx0": resolution, "xx0_min": 0.0, "xx0_max": 1.0},
        "training": {"seed": seed},
        "architecture": {"layers": [1, 1], "initialization": "xavier"},
        "stage": "ssbroyden",
        "step": 10,
    }
    return checkpoint.save_checkpoint(path, params, metadata)


def test_parse_resolutions() -> None:
    assert pointwise.parse_resolutions("380, 80,152") == [80, 152, 380]
    with pytest.raises(ValueError, match="duplicates"):
        pointwise.parse_resolutions("80,80")


def test_select_evaluate_and_plot_pointwise_errors(tmp_path: Path) -> None:
    paths = [
        _write_checkpoint(tmp_path / "n80s503.npz", resolution=80, seed=503),
        _write_checkpoint(tmp_path / "n92s503.npz", resolution=92, seed=503),
        _write_checkpoint(tmp_path / "n80s504.npz", resolution=80, seed=504),
    ]
    selected = pointwise.select_checkpoints(paths, resolutions=[80, 92], seed=503)
    assert [item.training_Nxx0 for item in selected] == [80, 92]

    xx0, radius, exact, values, errors = pointwise.evaluate_selected(
        selected, evaluation_Nxx0=16
    )
    assert xx0.shape == radius.shape == exact.shape == (16,)
    assert set(values) == set(errors) == {80, 92}
    assert np.allclose(errors[80], 1.0)

    prefix = tmp_path / "pointwise"
    table = pointwise.write_table(
        prefix.with_suffix(".tsv"),
        xx0=xx0,
        radius=radius,
        exact=exact,
        values_by_resolution=values,
        errors_by_resolution=errors,
    )
    png, pdf = pointwise.plot_errors(
        prefix, radius=radius, errors_by_resolution=errors
    )
    for output in (table, png, pdf):
        assert output.is_file()
        assert output.stat().st_size > 0
