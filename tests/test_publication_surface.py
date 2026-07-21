from __future__ import annotations

import ast
import importlib
from pathlib import Path
import subprocess
import tomllib

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SOLVER_ROOTS = (
    REPO_ROOT / "src" / "pinndorama" / "solvers" / "toy_problem_1d",
    REPO_ROOT / "src" / "pinndorama" / "solvers" / "punctures_2d",
    REPO_ROOT / "src" / "pinndorama" / "solvers" / "punctures_2d_parametric",
    REPO_ROOT / "src" / "pinndorama" / "solvers" / "punctures_3d",
)
ACTIVE_CODE_ROOTS = (
    *SOLVER_ROOTS,
    REPO_ROOT / "src" / "pinndorama" / "reproducibility",
    REPO_ROOT / "reference_solvers",
)
RETIRED_SOURCE_NAME = "x" + "Punc"
RETIRED_CLUSTER_NAME = "Tur" + "sa"
RETIRED_BATCH_DIRECTIVE = "#" + "SBATCH"
RETIRED_HOME_PREFIXES = (
    "/" + "home/dp" + "325",
    "/" + "home/dp" + "310",
)


def test_active_top_level_contains_only_repository_resources() -> None:
    allowed = {
        ".agents",
        ".codex",
        ".git",
        ".gitattributes",
        ".gitignore",
        ".pytest_cache",
        "AGENTS.md",
        "README.md",
        "archive",
        "configs",
        "docs",
        "pyproject.toml",
        "reference_solvers",
        "src",
        "tests",
        "vendor",
    }
    tracked = subprocess.run(
        ["git", "ls-files"],
        cwd=REPO_ROOT,
        check=True,
        capture_output=True,
        text=True,
    ).stdout.splitlines()
    top_level = {Path(path).parts[0] for path in tracked}
    unexpected = top_level - allowed
    assert not unexpected, f"unexpected active top-level paths: {sorted(unexpected)}"
    for retired in (
        "2d_jax_sinhsymtp",
        "2d_jax_sinhsymtp_parametric",
        "3d_jax_sinhsymtp",
        "reproducibility",
        "nrpy",
        "NRPyElliptic_2d_C_codes",
        "NRPyElliptic_3d_C_codes",
    ):
        assert not (REPO_ROOT / retired).exists()


def test_active_solver_source_is_cluster_neutral_and_z_separation_only() -> None:
    forbidden = (
        RETIRED_SOURCE_NAME,
        RETIRED_CLUSTER_NAME,
        RETIRED_BATCH_DIRECTIVE,
        *RETIRED_HOME_PREFIXES,
    )
    suffixes = {".py", ".md", ".toml", ".c", ".h", ".sh", ".slurm"}
    violations: list[str] = []
    for root in ACTIVE_CODE_ROOTS:
        for path in root.rglob("*"):
            if not path.is_file() or path.suffix not in suffixes:
                continue
            text = path.read_text(encoding="utf-8", errors="replace")
            for pattern in forbidden:
                if pattern in text:
                    violations.append(f"{path.relative_to(REPO_ROOT)}: {pattern}")
    assert not violations, "\n".join(violations)


def test_active_training_does_not_use_pickle_checkpoints() -> None:
    violations: list[str] = []
    for root in SOLVER_ROOTS:
        for name in ("checkpoint.py", "train.py", "evaluate.py"):
            path = root / name
            if not path.exists():
                continue
            tree = ast.parse(path.read_text(encoding="utf-8"), filename=str(path))
            for node in ast.walk(tree):
                if isinstance(node, ast.Import) and any(
                    alias.name == "pickle" for alias in node.names
                ):
                    violations.append(f"{path.relative_to(REPO_ROOT)}: pickle import")
                if isinstance(node, ast.ImportFrom) and node.module == "pickle":
                    violations.append(f"{path.relative_to(REPO_ROOT)}: pickle import")
                if isinstance(node, ast.Call):
                    for keyword in node.keywords:
                        if keyword.arg == "allow_pickle" and not (
                            isinstance(keyword.value, ast.Constant)
                            and keyword.value.value is False
                        ):
                            violations.append(
                                f"{path.relative_to(REPO_ROOT)}: unsafe allow_pickle"
                            )
    assert not violations, f"pickle references remain in {violations}"


def test_no_active_batch_scripts_remain() -> None:
    paths = [path for root in SOLVER_ROOTS for path in root.rglob("*.slurm")]
    assert not paths


def test_root_metadata_has_supported_python_bounds() -> None:
    with (REPO_ROOT / "pyproject.toml").open("rb") as stream:
        document = tomllib.load(stream)
    project = document["project"]
    assert project["name"] == "pinndorama"
    assert project["requires-python"] == ">=3.11,<3.13"
    assert project["optional-dependencies"]["test"] == ["pytest>=8,<9"]
    dependencies = project["dependencies"]
    assert "black==26.5.1" in dependencies
    assert (
        "jax[cuda12]>=0.4.38,<0.7; platform_system == 'Linux' and "
        "platform_machine == 'x86_64'"
    ) in dependencies
    assert (
        "jax>=0.4.38,<0.7; platform_system != 'Linux' or "
        "platform_machine != 'x86_64'"
    ) in dependencies
    for package in (
        "optax",
        "equinox",
        "lineax",
        "jaxtyping",
        "numpy",
        "sympy",
        "matplotlib",
    ):
        requirement = next(item for item in dependencies if item.startswith(package))
        assert ">=" in requirement and "<" in requirement
    assert document["tool"]["setuptools"]["package-dir"] == {"": "src"}
    assert document["tool"]["setuptools"]["packages"]["find"] == {
        "where": ["src"],
        "include": ["pinndorama*"],
    }
    assert document["tool"]["pytest"]["ini_options"]["pythonpath"] == ["src"]


def test_all_public_python_modules_import_through_pinndorama() -> None:
    module_suffixes = (
        "checkpoint",
        "config",
        "coordinates",
        "evaluate",
        "generated_expressions",
        "loss",
        "model",
        "nrpy_expression_builder",
        "ssbroyden_trainer",
        "train",
    )
    for solver in ("punctures_2d", "punctures_2d_parametric", "punctures_3d"):
        for suffix in module_suffixes:
            importlib.import_module(f"pinndorama.solvers.{solver}.{suffix}")
    for suffix in (
        "checkpoint",
        "config",
        "coordinates",
        "equation",
        "evaluate",
        "loss",
        "model",
        "plot_error",
        "ssbroyden_trainer",
        "train",
    ):
        importlib.import_module(f"pinndorama.solvers.toy_problem_1d.{suffix}")
    for suffix in (
        "compare",
        "create_reference",
        "evaluate",
        "metrics",
        "reference",
        "sampling",
    ):
        importlib.import_module(f"pinndorama.reproducibility.{suffix}")


def test_evaluator_uses_only_new_solver_names() -> None:
    evaluate = importlib.import_module("pinndorama.reproducibility.evaluate")
    assert set(evaluate.SOLVER_MODULES) == {
        "toy-problem-1d",
        "punctures-2d",
        "punctures-2d-parametric",
        "punctures-3d",
    }
    for retired in ("fixed-2d", "parametric-2d", "three-d"):
        with pytest.raises(SystemExit):
            evaluate.build_parser().parse_args([retired])


def test_comparison_reader_uses_reference_solver_hierarchy() -> None:
    compare = importlib.import_module("pinndorama.reproducibility.compare")
    assert compare.DEFAULT_READER == (
        REPO_ROOT
        / "reference_solvers"
        / "nrpyelliptic"
        / "three_dimensional"
        / "READER_nrpyelliptic_conformally_flat"
        / "nrpyell_reader"
    )


def test_all_solvers_use_the_shared_ssbroyden_loader() -> None:
    shared = importlib.import_module("pinndorama._ssbroyden")
    expected = REPO_ROOT / "vendor" / "ssbroyden"
    assert shared.SSBROYDEN_ROOT == expected
    for solver in (
        "toy_problem_1d",
        "punctures_2d",
        "punctures_2d_parametric",
        "punctures_3d",
    ):
        module = importlib.import_module(
            f"pinndorama.solvers.{solver}.ssbroyden_trainer"
        )
        assert module.load_ssbroyden_api is shared.load_ssbroyden_api
    assert (expected / "ssbroyden_family.py").is_file()
    assert not (expected / "ssbrodyen_family.py").exists()
    for retired in ("punctures_2d", "punctures_2d_parametric", "punctures_3d"):
        assert not (expected / retired).exists()
