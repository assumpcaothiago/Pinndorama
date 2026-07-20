from __future__ import annotations

import os
import json
from pathlib import Path
import subprocess
import sys
import tomllib

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


def _toml(path: str) -> dict:
    with (REPO_ROOT / path).open("rb") as stream:
        return tomllib.load(stream)


def test_checked_in_publication_anchors() -> None:
    c002_40 = _toml("configs/punctures_2d/C002_w40d4.toml")
    c002_light = _toml("configs/punctures_2d/C002_70x70_w40d4.toml")
    c002_60 = _toml("configs/punctures_2d/C002_w60d10.toml")
    c004 = _toml("configs/punctures_2d/C004_square_w40d4.toml")
    p001 = _toml("configs/punctures_2d_parametric/P001_w40d4.toml")
    c04 = _toml("configs/punctures_3d/C04.toml")
    c05 = _toml("configs/punctures_3d/C05.toml")

    assert all(
        document["schema_version"] == 2
        for document in (c002_40, c002_light, c002_60, c004, p001, c04, c05)
    )

    assert (
        c002_40["collocation"]["Nxx0"],
        c002_40["collocation"]["Nxx1"],
    ) == (384, 192)
    assert c002_40["architecture"]["layers"] == [2, 40, 40, 40, 40, 1]
    assert (
        c002_light["collocation"]["Nxx0"],
        c002_light["collocation"]["Nxx1"],
    ) == (70, 70)
    assert c002_60["architecture"]["layers"] == [2] + [60] * 10 + [1]
    assert (
        c004["collocation"]["Nxx0"],
        c004["collocation"]["Nxx1"],
    ) == (256, 256)

    assert p001["collocation"] == {
        "Nxx0": 128,
        "Nxx1": 64,
        "equal_spin_sz_points": 80,
    }
    assert p001["parameter"]["equal_spin_sz"] == {
        "minimum": -0.2,
        "maximum": 0.2,
        "sampling": "cell_centered",
    }
    assert p001["architecture"]["layers"] == [3, 40, 40, 40, 40, 1]

    assert (
        c04["collocation"]["Nxx0"],
        c04["collocation"]["Nxx1"],
        c04["collocation"]["Nxx2"],
    ) == (256, 256, 16)
    assert c04["training"]["adam_steps"] == 10000
    assert c04["training"]["learning_rate"] == 1.0e-3
    assert c04["training"]["ssbroyden_steps"] == 50000
    assert c05["training"]["adam_steps"] == 0
    assert c05["training"]["ssbroyden_steps"] == 50000
    assert c05["collocation"]["xx0_sampling"] == "two-zone"
    assert c05["collocation"]["xx0_cut"] == 0.5
    assert c05["collocation"]["Nxx0_inner"] == 236
    assert c05["continuation"]["require_resume"] is True


@pytest.mark.parametrize(
    ("module_name", "configuration", "legacy_key"),
    (
        (
            "pinndorama.solvers.punctures_2d.config",
            "configs/punctures_2d/C002_w40d4.toml",
            "radial_points",
        ),
        (
            "pinndorama.solvers.punctures_2d_parametric.config",
            "configs/punctures_2d_parametric/P001_w40d4.toml",
            "radial_points",
        ),
        (
            "pinndorama.solvers.punctures_3d.config",
            "configs/punctures_3d/C04.toml",
            "radial_points",
        ),
    ),
)
def test_configuration_schema_rejects_legacy_collocation_names(
    module_name: str,
    configuration: str,
    legacy_key: str,
    tmp_path: Path,
) -> None:
    import importlib

    source = REPO_ROOT / configuration
    migrated = source.read_text(encoding="utf-8").replace("Nxx0", legacy_key, 1)
    legacy = tmp_path / source.name
    legacy.write_text(migrated, encoding="utf-8")
    module = importlib.import_module(module_name)
    loader = (
        module.load
        if module_name.endswith("punctures_3d.config")
        else module.load_config
    )
    with pytest.raises(ValueError):
        loader(legacy)


@pytest.mark.parametrize(
    "module",
    (
        "pinndorama.solvers.punctures_2d.train",
        "pinndorama.solvers.punctures_2d_parametric.train",
        "pinndorama.solvers.punctures_3d.train",
    ),
)
def test_training_cli_exposes_only_Nxx_resolution_flags(module: str) -> None:
    result = subprocess.run(
        [sys.executable, "-m", module, "--help"],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        text=True,
        capture_output=True,
        check=True,
    )
    assert "--Nxx0" in result.stdout
    assert "--Nxx1" in result.stdout
    if module.endswith("punctures_3d.train"):
        assert "--Nxx2" in result.stdout
    assert "--radial-points" not in result.stdout
    assert "--theta-points" not in result.stdout
    assert "--phi-points" not in result.stdout


@pytest.mark.parametrize(
    ("module", "configuration", "legacy_flag"),
    (
        (
            "pinndorama.solvers.punctures_2d.train",
            "configs/punctures_2d/C002_w40d4.toml",
            "--radial-points",
        ),
        (
            "pinndorama.solvers.punctures_2d_parametric.train",
            "configs/punctures_2d_parametric/P001_w40d4.toml",
            "--theta-points",
        ),
        (
            "pinndorama.solvers.punctures_3d.train",
            "configs/punctures_3d/C04.toml",
            "--phi-points",
        ),
    ),
)
def test_training_cli_rejects_legacy_resolution_flags(
    module: str, configuration: str, legacy_flag: str, tmp_path: Path
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            module,
            "--config",
            configuration,
            "--output-dir",
            str(tmp_path),
            legacy_flag,
            "2",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr


@pytest.mark.parametrize(
    ("module", "configuration"),
    (
        (
            "pinndorama.solvers.punctures_2d.train",
            "configs/punctures_2d/C002_w40d4.toml",
        ),
        (
            "pinndorama.solvers.punctures_2d_parametric.train",
            "configs/punctures_2d_parametric/P001_w40d4.toml",
        ),
        ("pinndorama.solvers.punctures_3d.train", "configs/punctures_3d/C04.toml"),
    ),
)
def test_training_cli_rejects_physics_overrides(
    module: str, configuration: str, tmp_path: Path
) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            module,
            "--config",
            configuration,
            "--output-dir",
            str(tmp_path),
            "--bscale",
            "7",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        text=True,
        capture_output=True,
    )
    assert result.returncode == 2
    assert "unrecognized arguments" in result.stderr


def test_c05_requires_explicit_parent_checkpoint(tmp_path: Path) -> None:
    result = subprocess.run(
        [
            sys.executable,
            "-m",
            "pinndorama.solvers.punctures_3d.train",
            "--config",
            "configs/punctures_3d/C05.toml",
            "--output-dir",
            str(tmp_path),
            "--adam-steps",
            "0",
            "--ssbroyden-steps",
            "0",
        ],
        cwd=REPO_ROOT,
        env={**os.environ, "PYTHONPATH": str(REPO_ROOT / "src")},
        text=True,
        capture_output=True,
    )
    assert result.returncode != 0
    assert "requires --resume-checkpoint" in result.stderr


def test_two_zone_override_cannot_remove_outer_zone() -> None:
    script = r"""
import importlib
import pathlib
import sys
config = importlib.import_module("pinndorama.solvers.punctures_3d.config")
run = config.load(pathlib.Path(sys.argv[1]))
try:
    config.with_overrides(run, Nxx0=236)
except ValueError as error:
    print(error)
    raise SystemExit(0)
raise SystemExit(1)
"""
    environment = os.environ.copy()
    environment["PYTHONPYCACHEPREFIX"] = "/tmp/pinndorama_pycache"
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(REPO_ROOT / "configs/punctures_3d/C05.toml"),
        ],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
    )
    assert result.returncode == 0
    assert "greater than" in result.stdout


def test_3d_publication_xx0_sampling_is_cell_centered() -> None:
    script = r"""
import importlib
import json
import pathlib
import sys
config = importlib.import_module("pinndorama.solvers.punctures_3d.config")
coordinates = importlib.import_module("pinndorama.solvers.punctures_3d.coordinates")
config_root = pathlib.Path(sys.argv[1])
c04 = config.load(config_root / "C04.toml")
c05 = config.load(config_root / "C05.toml")
uniform = coordinates.xx0_collocation_points(c04.collocation)
two_zone = coordinates.xx0_collocation_points(c05.collocation)
print(json.dumps({
    "uniform": [len(uniform), uniform[0], uniform[-1]],
    "two_zone": [
        len(two_zone),
        two_zone[0],
        two_zone[235],
        two_zone[236],
        two_zone[-1],
    ],
}))
"""
    environment = os.environ.copy()
    environment["PYTHONPYCACHEPREFIX"] = "/tmp/pinndorama_pycache"
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(
        [sys.executable, "-c", script, str(REPO_ROOT / "configs/punctures_3d")],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    document = json.loads(result.stdout)
    assert document["uniform"] == pytest.approx([256, 0.5 / 256.0, 1.0 - 0.5 / 256.0])
    inner_spacing = 0.5 / 236.0
    outer_spacing = 0.5 / 20.0
    assert document["two_zone"] == pytest.approx(
        [
            256,
            0.5 * inner_spacing,
            0.5 - 0.5 * inner_spacing,
            0.5 + 0.5 * outer_spacing,
            1.0 - 0.5 * outer_spacing,
        ]
    )
