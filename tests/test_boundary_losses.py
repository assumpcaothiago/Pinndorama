"""Small float64 CPU checks for publication boundary and regularity losses."""

from __future__ import annotations

import json
import math
import os
from pathlib import Path
import subprocess
import sys
import textwrap

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _run_solver_script(solver: str, body: str) -> dict[str, object]:
    """Run one solver in an isolated process to avoid local-module collisions."""

    script = (
        "import importlib\n"
        "import json\n"
        "import pathlib\n"
        "import sys\n\n"
        "module_root = sys.argv[1]\n"
        "config_root = pathlib.Path(sys.argv[2]).resolve()\n"
        "config = importlib.import_module(f'{module_root}.config')\n"
        "loss = importlib.import_module(f'{module_root}.loss')\n"
        "model = importlib.import_module(f'{module_root}.model')\n"
        "generated_expressions = importlib.import_module(\n"
        "    f'{module_root}.generated_expressions'\n"
        ")\n\n" + textwrap.dedent(body)
    )
    environment = os.environ.copy()
    environment.update(
        {
            "JAX_PLATFORMS": "cpu",
            "JAX_ENABLE_X64": "1",
            "XLA_PYTHON_CLIENT_PREALLOCATE": "false",
            "PYTHONPYCACHEPREFIX": "/tmp/pinndorama_boundary_test_pycache",
            "PYTHONPATH": str(REPO_ROOT / "src"),
        }
    )
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            f"pinndorama.solvers.{solver}",
            str(REPO_ROOT / "configs" / solver),
        ],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    return json.loads(result.stdout.splitlines()[-1])


def _assert_boundary_document(
    document: dict[str, object], names: tuple[str, ...]
) -> None:
    zero = document["zero"]
    nontrivial = document["nontrivial"]
    assert isinstance(zero, dict)
    assert isinstance(nontrivial, dict)
    assert set(zero) == set(names)
    assert set(nontrivial) == set(names)
    for name in names:
        assert zero[name] == pytest.approx(0.0, abs=1.0e-28)
        value = float(nontrivial[name])
        assert math.isfinite(value)
        assert value >= 0.0
    assert any(float(nontrivial[name]) > 0.0 for name in names)


def test_fixed_2d_theta_parity_and_outer_robin_losses() -> None:
    document = _run_solver_script(
        "punctures_2d",
        r"""
        import jax
        import jax.numpy as jnp
        config.enable_jax_x64()
        run = config.load_config(config_root / "C002_w40d4.toml")
        layers = run.architecture["layers"]
        zero_params = [
            {
                "W": jnp.zeros((out_dim, in_dim), dtype=jnp.float64),
                "b": jnp.zeros((out_dim,), dtype=jnp.float64),
            }
            for in_dim, out_dim in zip(layers[:-1], layers[1:])
        ]
        params = model.init_mlp_params(
            jax.random.PRNGKey(17),
            layers,
            initialization=run.architecture["initialization"],
        )
        geometry = {
            name: run.geometry[name] for name in ("AMAX", "bScale", "SINHWAA")
        }
        parity_xx0 = jnp.asarray([0.2, 0.7], dtype=jnp.float64)
        parity_delta = jnp.asarray([0.02, 0.07], dtype=jnp.float64)
        outer_xx0 = jnp.ones(2, dtype=jnp.float64)
        outer_xx1 = jnp.asarray([0.4, 2.1], dtype=jnp.float64)

        def terms(current):
            return {
                "theta_parity": float(
                    loss.theta_inner_parity_loss(
                        current, parity_xx0, parity_delta, **geometry
                    )
                ),
                "outer_robin": float(
                    loss.outer_robin_boundary_loss(
                        current, outer_xx0, outer_xx1, **geometry
                    )
                ),
            }

        print(json.dumps({"zero": terms(zero_params), "nontrivial": terms(params)}))
        """,
    )
    _assert_boundary_document(document, ("theta_parity", "outer_robin"))


def test_parametric_2d_boundaries_and_equal_spin_source() -> None:
    document = _run_solver_script(
        "punctures_2d_parametric",
        r"""
        import jax
        import jax.numpy as jnp
        import numpy as np
        run = config.load_config(config_root / "P001_w40d4.toml")
        config.apply_runtime_config(run)
        config.enable_jax_x64()
        layers = run["architecture"]["layers"]
        zero_params = [
            {
                "W": jnp.zeros((out_dim, in_dim), dtype=jnp.float64),
                "b": jnp.zeros((out_dim,), dtype=jnp.float64),
            }
            for in_dim, out_dim in zip(layers[:-1], layers[1:])
        ]
        params = model.init_mlp_params(jax.random.PRNGKey(19), layers=layers)
        geometry = config.geometry_parameter_dict(run)
        parity_xx0 = jnp.asarray([0.2, 0.7], dtype=jnp.float64)
        parity_delta = jnp.asarray([0.02, 0.07], dtype=jnp.float64)
        equal_spin_sz = jnp.asarray([0.1, -0.1], dtype=jnp.float64)
        outer_xx0 = jnp.ones(2, dtype=jnp.float64)
        outer_xx1 = jnp.asarray([0.4, 2.1], dtype=jnp.float64)

        def terms(current):
            return {
                "theta_parity": float(
                    loss.theta_inner_parity_loss(
                        current,
                        parity_xx0,
                        parity_delta,
                        equal_spin_sz,
                        **geometry,
                    )
                ),
                "outer_robin": float(
                    loss.outer_robin_boundary_loss(
                        current,
                        outer_xx0,
                        outer_xx1,
                        equal_spin_sz,
                        **geometry,
                    )
                ),
            }

        source_xx0 = jnp.asarray([0.31, 0.63], dtype=jnp.float64)
        source_xx1 = jnp.asarray([0.8, 2.0], dtype=jnp.float64)
        source_spin = jnp.asarray([0.13, -0.09], dtype=jnp.float64)
        through_public_api = generated_expressions.equal_spin_source_terms_jax(
            source_xx0,
            source_xx1,
            source_spin,
            physics=run["physics"],
            **geometry,
        )
        explicit_equal = generated_expressions.source_terms_jax(
            source_xx0,
            source_xx1,
            0.0,
            **run["physics"],
            **geometry,
            S0_z=source_spin,
            S1_z=source_spin,
        )
        source_matches = all(
            np.allclose(np.asarray(public), np.asarray(explicit), rtol=0.0, atol=0.0)
            for public, explicit in zip(through_public_api, explicit_equal)
        )
        plus = generated_expressions.equal_spin_source_terms_jax(
            source_xx0,
            source_xx1,
            jnp.abs(source_spin),
            physics=run["physics"],
            **geometry,
        )
        minus = generated_expressions.equal_spin_source_terms_jax(
            source_xx0,
            source_xx1,
            -jnp.abs(source_spin),
            physics=run["physics"],
            **geometry,
        )
        sign_symmetric = all(
            np.allclose(np.asarray(positive), np.asarray(negative), rtol=1.0e-14, atol=1.0e-14)
            for positive, negative in zip(plus, minus)
        )
        print(
            json.dumps(
                {
                    "zero": terms(zero_params),
                    "nontrivial": terms(params),
                    "source_matches_explicit_equal_spins": source_matches,
                    "source_is_sign_symmetric": sign_symmetric,
                }
            )
        )
        """,
    )
    _assert_boundary_document(document, ("theta_parity", "outer_robin"))
    assert document["source_matches_explicit_equal_spins"] is True
    assert document["source_is_sign_symmetric"] is True


def test_3d_axis_periodicity_and_outer_robin_losses() -> None:
    document = _run_solver_script(
        "punctures_3d",
        r"""
        import jax
        import jax.numpy as jnp
        config.enable_jax_x64()
        run = config.load(config_root / "C04.toml")
        layers = run.architecture.layers
        zero_params = [
            {
                "W": jnp.zeros((out_dim, in_dim), dtype=jnp.float64),
                "b": jnp.zeros((out_dim,), dtype=jnp.float64),
            }
            for in_dim, out_dim in zip(layers[:-1], layers[1:])
        ]
        params = model.init_mlp_params(jax.random.PRNGKey(23), layers)
        axis_xx0 = jnp.asarray([0.2, 0.7], dtype=jnp.float64)
        axis_delta = jnp.asarray([0.02, 0.07], dtype=jnp.float64)
        axis_phi = jnp.asarray([-1.0, 1.1], dtype=jnp.float64)
        periodic_xx0 = jnp.asarray([0.3, 0.8], dtype=jnp.float64)
        periodic_xx1 = jnp.asarray([0.5, 2.0], dtype=jnp.float64)
        outer_xx0 = jnp.ones(2, dtype=jnp.float64)
        outer_xx1 = jnp.asarray([0.6, 2.1], dtype=jnp.float64)
        outer_xx2 = jnp.asarray([-0.7, 1.2], dtype=jnp.float64)

        def terms(current):
            return {
                "theta_axis": float(
                    loss.theta_axis_regularity_loss(
                        current, axis_xx0, axis_delta, axis_phi, run.geometry
                    )
                ),
                "phi_periodicity": float(
                    loss.phi_periodicity_loss(
                        current,
                        periodic_xx0,
                        periodic_xx1,
                        run.geometry,
                        run.collocation,
                    )
                ),
                "outer_robin": float(
                    loss.outer_robin_boundary_loss(
                        current,
                        outer_xx0,
                        outer_xx1,
                        outer_xx2,
                        run.geometry,
                    )
                ),
            }

        print(json.dumps({"zero": terms(zero_params), "nontrivial": terms(params)}))
        """,
    )
    _assert_boundary_document(
        document, ("theta_axis", "phi_periodicity", "outer_robin")
    )
