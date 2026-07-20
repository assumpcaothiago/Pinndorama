from __future__ import annotations

from copy import deepcopy
from pathlib import Path
import subprocess
import sys

import numpy as np
import pytest

from pinndorama.solvers.toy_problem_1d import (
    checkpoint,
    config,
    coordinates,
    equation,
    evaluate,
    loss,
    model,
    plot_error,
)

REPO_ROOT = Path(__file__).resolve().parents[1]
CONFIG_PATH = REPO_ROOT / "configs" / "toy_problem_1d" / "T001_w40d4.toml"


def test_checked_toy_configuration_is_fixed_mass_and_one_input() -> None:
    run = config.load_config(CONFIG_PATH)
    assert run.as_dict()["schema_version"] == 4
    assert run.physics == {"m": 1.0}
    assert run.geometry == {
        "coordinate_system": "SinhSpherical",
        "AMPL": 1.0e6,
        "SINHW": 0.07,
    }
    assert run.collocation == {
        "sampling": "cell_centered",
        "Nxx0": 256,
        "xx0_min": 0.0,
        "xx0_max": 1.0,
    }
    assert run.architecture["layers"] == [1, 40, 40, 40, 40, 1]
    assert run.loss == {
        "interior_residual": "r_times_spherical_pde",
        "interior_weight": 1.0,
        "origin_regularity_weight": 1.0,
        "outer_boundary_condition": "second_order_robin",
        "outer_boundary_region": "radial_band",
        "outer_boundary_r_min": 1.0e3,
        "outer_boundary_weight": 1.0,
    }
    assert run.training["adam_steps"] == 1500
    assert run.training["ssbroyden_steps"] == 10000


@pytest.mark.parametrize(
    ("old", "new"),
    (
        ("m = 1.0", "m = 0.0"),
        (
            "layers = [1, 40, 40, 40, 40, 1]",
            "layers = [2, 40, 40, 40, 40, 1]",
        ),
        ("Nxx0 = 256", "radial_points = 256"),
        ("schema_version = 4", "schema_version = 3"),
        ("outer_boundary_r_min = 1.0e3", "outer_boundary_r_min = 0.0"),
        ("outer_boundary_r_min = 1.0e3", "outer_boundary_r_min = 1.0e6"),
        (
            'outer_boundary_condition = "second_order_robin"',
            'outer_boundary_condition = "unknown"',
        ),
        (
            'outer_boundary_condition = "second_order_robin"',
            'outer_boundary_condition = "homogeneous_robin"',
        ),
        (
            'outer_boundary_condition = "second_order_robin"',
            'outer_boundary_condition = "subleading_asymptotic"',
        ),
        ('outer_boundary_region = "radial_band"', 'outer_boundary_region = "shell"'),
    ),
)
def test_toy_configuration_rejects_invalid_physics_architecture_and_keys(
    old: str, new: str, tmp_path: Path
) -> None:
    invalid = tmp_path / "invalid.toml"
    invalid.write_text(
        CONFIG_PATH.read_text(encoding="utf-8").replace(old, new),
        encoding="utf-8",
    )
    with pytest.raises(config.ConfigError):
        config.load_config(invalid)


@pytest.mark.parametrize(
    ("condition", "region"),
    (
        ("first_order_robin", "endpoint"),
        ("first_order_robin", "radial_band"),
        ("second_order_robin", "endpoint"),
        ("second_order_robin", "radial_band"),
    ),
)
def test_toy_configuration_accepts_each_boundary_combination(
    condition: str, region: str, tmp_path: Path
) -> None:
    contents = CONFIG_PATH.read_text(encoding="utf-8").replace(
        'outer_boundary_condition = "second_order_robin"',
        f'outer_boundary_condition = "{condition}"',
    )
    if region == "endpoint":
        contents = contents.replace(
            'outer_boundary_region = "radial_band"\nouter_boundary_r_min = 1.0e3',
            'outer_boundary_region = "endpoint"',
        )
    candidate = tmp_path / f"{condition}_{region}.toml"
    candidate.write_text(contents, encoding="utf-8")
    run = config.load_config(candidate)
    assert run.loss["outer_boundary_condition"] == condition
    assert run.loss["outer_boundary_region"] == region
    assert ("outer_boundary_r_min" in run.loss) == (region == "radial_band")


@pytest.mark.parametrize(
    "replacement",
    (
        'outer_robin_weight = 1.0\nouter_robin_scale = "one_plus_r"',
        'outer_asymptotic_condition = "subleading_inverse_radius"\n'
        "outer_asymptotic_weight = 1.0\nouter_asymptotic_r_min = 1.0e3",
    ),
)
def test_toy_configuration_rejects_retired_boundary_keys(
    replacement: str, tmp_path: Path
) -> None:
    retired = tmp_path / "retired.toml"
    retired.write_text(
        CONFIG_PATH.read_text(encoding="utf-8").replace(
            'outer_boundary_condition = "second_order_robin"\n'
            'outer_boundary_region = "radial_band"\n'
            "outer_boundary_r_min = 1.0e3\n"
            "outer_boundary_weight = 1.0",
            replacement,
        ),
        encoding="utf-8",
    )
    with pytest.raises(config.ConfigError):
        config.load_config(retired)


def test_endpoint_region_rejects_irrelevant_radius_cutoff(tmp_path: Path) -> None:
    invalid = tmp_path / "endpoint_with_cutoff.toml"
    invalid.write_text(
        CONFIG_PATH.read_text(encoding="utf-8").replace(
            'outer_boundary_region = "radial_band"',
            'outer_boundary_region = "endpoint"',
        ),
        encoding="utf-8",
    )
    with pytest.raises(config.ConfigError, match="unknown keys: outer_boundary_r_min"):
        config.load_config(invalid)


def test_toy_cli_exposes_only_Nxx0_resolution_override(tmp_path: Path) -> None:
    help_result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pinndorama.solvers.toy_problem_1d.train",
            "--help",
        ],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--Nxx0" in help_result.stdout
    for forbidden in ("--Nxx1", "--radial-points", "--m"):
        assert forbidden not in help_result.stdout
        rejected = subprocess.run(
            [
                sys.executable,
                "-m",
                "pinndorama.solvers.toy_problem_1d.train",
                "--config",
                str(CONFIG_PATH),
                "--output-dir",
                str(tmp_path),
                forbidden,
                "2",
            ],
            cwd=REPO_ROOT,
            text=True,
            capture_output=True,
        )
        assert rejected.returncode == 2
        assert "unrecognized arguments" in rejected.stderr


def test_sinh_spherical_map_derivatives_and_cell_centers() -> None:
    ampl = 1.0e6
    sinhw = 0.07
    xx0 = np.asarray([0.0, 0.2, 0.7, 1.0])
    radius = coordinates.radius_from_xx0(xx0, ampl=ampl, sinhw=sinhw)
    expected = ampl * np.sinh(xx0 / sinhw) / np.sinh(1.0 / sinhw)
    assert np.array_equal(radius, expected)
    assert radius[0] == 0.0
    assert radius[-1] == pytest.approx(ampl)
    assert np.allclose(
        coordinates.d2r_dxx02(xx0, ampl=ampl, sinhw=sinhw),
        radius / sinhw**2,
        rtol=2.0e-15,
        atol=0.0,
    )

    point = 0.43
    step = 1.0e-6
    finite_difference = (
        coordinates.radius_from_xx0(point + step, ampl=ampl, sinhw=sinhw)
        - coordinates.radius_from_xx0(point - step, ampl=ampl, sinhw=sinhw)
    ) / (2.0 * step)
    assert coordinates.dr_dxx0(point, ampl=ampl, sinhw=sinhw) == pytest.approx(
        finite_difference, rel=2.0e-10
    )
    assert np.allclose(
        coordinates.cell_centered_xx0(Nxx0=4),
        np.asarray([0.125, 0.375, 0.625, 0.875]),
    )

    interior = coordinates.cell_centered_xx0(Nxx0=256)
    endpoint = coordinates.outer_boundary_xx0(
        interior, region="endpoint", ampl=ampl, sinhw=sinhw
    )
    assert np.array_equal(endpoint, np.asarray([1.0]))

    outer = coordinates.outer_boundary_xx0(
        interior,
        region="radial_band",
        ampl=ampl,
        sinhw=sinhw,
        r_min=1.0e3,
    )
    outer_radius = coordinates.radius_from_xx0(outer, ampl=ampl, sinhw=sinhw)
    interior_radius = coordinates.radius_from_xx0(interior, ampl=ampl, sinhw=sinhw)
    assert outer[-1] == 1.0
    assert np.all(outer_radius > 1.0e3)
    assert np.array_equal(outer[:-1], interior[interior_radius > 1.0e3])

    minimal = coordinates.cell_centered_xx0(Nxx0=2)
    minimal_outer = coordinates.outer_boundary_xx0(
        minimal,
        region="radial_band",
        ampl=ampl,
        sinhw=sinhw,
        r_min=1.0e3,
    )
    assert minimal_outer.size >= 1
    assert minimal_outer[-1] == 1.0


def test_stable_nonlinear_term_matches_original_and_has_finite_origin() -> None:
    m = 1.3
    radius = np.asarray([0.1, 0.7, 3.0, 20.0])
    values = np.asarray([0.4, -0.1, 0.02, 0.001])
    direct = (
        equation.source_A(radius, m)
        / 8.0
        * (equation.psi_background(radius, m) + values) ** -7
    )
    stable = equation.nonlinear_term(radius, values, m)
    assert np.allclose(stable, direct, rtol=2.0e-14, atol=0.0)
    assert equation.nonlinear_term(0.0, 42.0, m) == pytest.approx(3.0 * m)


def test_manufactured_solution_satisfies_equation_and_origin_regularity() -> None:
    m = 0.8
    radius = np.asarray([0.0, 0.01, 0.2, 1.0, 10.0, 1.0e4])
    exact = equation.exact_solution(radius, m)
    pde = equation.exact_laplacian(radius, m) + equation.nonlinear_term(
        radius, exact, m
    )
    assert np.allclose(pde, 0.0, rtol=0.0, atol=2.0e-14)
    regularized = equation.regularized_residual_from_radial_derivatives(
        radius,
        exact,
        equation.exact_solution_dr(radius, m),
        equation.exact_solution_drr(radius, m),
        m,
    )
    assert np.allclose(regularized, 0.0, rtol=0.0, atol=2.0e-13)
    assert equation.exact_solution_dr(0.0, m) == 0.0

    outer_radius = np.asarray([1.0e3, 1.0e4, 1.0e5, 1.0e6])
    outer_value = equation.exact_solution(outer_radius, m)
    outer_derivative = equation.exact_solution_dr(outer_radius, m)
    mismatch = loss.second_order_robin_residual_from_radial_values(
        outer_radius, outer_value, outer_derivative, m
    )
    assert np.all(np.abs(mismatch[1:]) < np.abs(mismatch[:-1]))
    assert abs(mismatch[-1]) < 1.0e-9


def test_outer_boundary_residuals_select_first_or_second_order_behavior() -> None:
    radius = np.asarray([1.0e3, 2.0e3, 1.0e4])
    m = 1.3
    leading = 7.0

    second_order_coefficient = -2.0 * m
    correct_u = leading / radius + second_order_coefficient / radius**2
    correct_u_r = -leading / radius**2 - 2.0 * second_order_coefficient / radius**3
    correct = loss.second_order_robin_residual_from_radial_values(
        radius, correct_u, correct_u_r, m
    )
    assert np.allclose(correct, 0.0, rtol=0.0, atol=1.0e-11)

    first_order_u = leading / radius
    first_order_u_r = -leading / radius**2
    second_order_mismatch = loss.second_order_robin_residual_from_radial_values(
        radius, first_order_u, first_order_u_r, m
    )
    assert np.allclose(second_order_mismatch, -2.0 * m, rtol=0.0, atol=1.0e-11)
    robin = loss.first_order_robin_residual_from_radial_values(
        radius, first_order_u, first_order_u_r
    )
    assert np.allclose(robin, 0.0, rtol=0.0, atol=1.0e-11)

    assert np.array_equal(
        loss.outer_boundary_residual_from_radial_values(
            radius,
            first_order_u,
            first_order_u_r,
            m,
            condition="first_order_robin",
        ),
        robin,
    )
    with pytest.raises(ValueError, match="unknown outer boundary condition"):
        loss.outer_boundary_residual_from_radial_values(
            radius, first_order_u, first_order_u_r, m, condition="unknown"
        )


@pytest.mark.jax
def test_model_falloff_and_boundary_losses_use_physical_radius() -> None:
    import jax.numpy as jnp

    config.enable_jax_x64()
    ampl = 1.0e6
    sinhw = 0.07
    params = [
        {
            "W": jnp.zeros((1, 1), dtype=jnp.float64),
            "b": jnp.asarray([2.0], dtype=jnp.float64),
        }
    ]
    xx0 = jnp.asarray([[0.0], [0.5], [1.0]], dtype=jnp.float64)
    radius = coordinates.radius_from_xx0(
        np.asarray(xx0).reshape(-1), ampl=ampl, sinhw=sinhw
    )
    values = np.asarray(model.forward(params, xx0, ampl=ampl, sinhw=sinhw))[:, 0]
    assert np.allclose(values, 2.0 / np.sqrt(1.0 + radius * radius))
    assert float(loss.origin_regularity_loss(params, ampl=ampl, sinhw=sinhw)) == (
        pytest.approx(0.0, abs=1.0e-28)
    )
    outer_xx0 = jnp.asarray([0.8, 0.9, 1.0], dtype=jnp.float64)
    for condition in ("first_order_robin", "second_order_robin"):
        outer = float(
            loss.outer_boundary_loss(
                params,
                outer_xx0,
                ampl=ampl,
                sinhw=sinhw,
                m=1.0,
                outer_boundary_condition=condition,
            )
        )
        assert np.isfinite(outer)
        assert outer > 0.0

        total, components = loss.compute_loss(
            params,
            jnp.asarray([0.25, 0.75], dtype=jnp.float64),
            outer_xx0,
            ampl=ampl,
            sinhw=sinhw,
            m=1.0,
            outer_boundary_condition=condition,
        )
        assert set(components) == {
            "residual_loss",
            "origin_regularity_loss",
            "outer_boundary_loss",
        }
        assert float(total) == pytest.approx(
            sum(float(value) for value in components.values())
        )


def test_toy_checkpoint_resume_and_evaluation(tmp_path: Path) -> None:
    run = config.load_config(CONFIG_PATH)
    params = [
        {
            "W": np.zeros((out_dim, in_dim), dtype=np.float64),
            "b": np.zeros((out_dim,), dtype=np.float64),
        }
        for in_dim, out_dim in zip(
            run.architecture["layers"][:-1], run.architecture["layers"][1:]
        )
    ]
    metadata = checkpoint.build_metadata(
        run,
        stage="adam",
        step=1,
        parent_checkpoint_sha256=None,
    )
    parent = checkpoint.save_checkpoint(tmp_path / "parent.npz", params, metadata)
    loaded, loaded_metadata = checkpoint.load_checkpoint(parent)
    checkpoint.validate_resume(loaded_metadata, run)
    assert loaded_metadata["loss"] == run.loss
    assert "outer_robin_weight" not in loaded_metadata["loss"]
    assert loaded_metadata["loss"]["outer_boundary_condition"] == ("second_order_robin")

    mutable = deepcopy(loaded_metadata)
    mutable["collocation"]["Nxx0"] = 2
    checkpoint.validate_resume(mutable, run)
    legacy_loss = deepcopy(loaded_metadata)
    legacy_loss["loss"] = {
        "interior_residual": "r_times_spherical_pde",
        "interior_weight": 1.0,
        "origin_regularity_weight": 1.0,
        "outer_robin_weight": 1.0,
        "outer_robin_scale": "one_plus_r",
    }
    legacy_parent = checkpoint.save_checkpoint(
        tmp_path / "legacy_parent.npz", loaded, legacy_loss
    )
    legacy_params, legacy_metadata = checkpoint.load_checkpoint(legacy_parent)
    checkpoint.validate_resume(legacy_metadata, run)
    incompatible = deepcopy(loaded_metadata)
    incompatible["physics"]["m"] = 2.0
    with pytest.raises(checkpoint.CheckpointError):
        checkpoint.validate_resume(incompatible, run)

    digest = checkpoint.sha256_file(legacy_parent)
    child_metadata = checkpoint.build_metadata(
        run,
        stage="complete",
        step=1,
        parent_checkpoint_sha256=digest,
    )
    child = checkpoint.save_checkpoint(
        tmp_path / "child.npz", legacy_params, child_metadata
    )
    _, child_loaded = checkpoint.load_checkpoint(child)
    assert child_loaded["parent_checkpoint_sha256"] == digest
    assert not list(tmp_path.glob(".*.tmp"))

    coordinate_path = tmp_path / "xx0.txt"
    coordinate_path.write_text("0.0\n0.5\n1.0\n", encoding="utf-8")
    output = evaluate.evaluate_checkpoint(
        run, legacy_parent, coordinate_path, tmp_path / "values.txt"
    )
    table = np.loadtxt(output, comments="#", ndmin=2)
    assert table.shape == (3, 6)
    assert np.all(table[:, 2] == 0.0)
    assert np.all(table[:, 3] > 0.0)
    assert np.all(table[:, 4] == table[:, 3])
    assert np.all(table[:, 5] == 1.0)


def test_toy_relative_error_loglog_plot(tmp_path: Path) -> None:
    values = tmp_path / "values.txt"
    np.savetxt(
        values,
        np.asarray(
            [
                [0.0, 0.0, 1.0, 1.0, 0.0, 0.0],
                [0.2, 10.0, 0.9, 1.0, 0.1, 1.0e-1],
                [0.1, 1.0, 0.99, 1.0, 0.01, 1.0e-2],
            ]
        ),
        header="xx0 r u_nn u_exact absolute_error relative_error",
    )

    radius, relative_error = plot_error.load_relative_error(values)
    assert np.array_equal(radius, np.asarray([1.0, 10.0]))
    assert np.array_equal(relative_error, np.asarray([1.0e-2, 1.0e-1]))

    output = plot_error.plot_relative_error(values, tmp_path / "plots" / "error.png")
    assert output.is_file()
    assert output.stat().st_size > 0
