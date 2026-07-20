from __future__ import annotations

import json
import os
from pathlib import Path
import subprocess
import sys

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]


@pytest.mark.parametrize(
    ("solver", "configuration"),
    (
        ("pinndorama.solvers.punctures_2d", "configs/punctures_2d/C002_w40d4.toml"),
        (
            "pinndorama.solvers.punctures_2d_parametric",
            "configs/punctures_2d_parametric/P001_w40d4.toml",
        ),
    ),
)
def test_npz_checkpoint_atomic_round_trip_and_parent_provenance(
    solver: str, configuration: str, tmp_path: Path
) -> None:
    script = r"""
import copy
import importlib
import json
import pathlib
import sys
import numpy as np

module_name = sys.argv[1]
config_path = pathlib.Path(sys.argv[2])
output = pathlib.Path(sys.argv[3])
config = importlib.import_module(f"{module_name}.config")
checkpoint = importlib.import_module(f"{module_name}.checkpoint")

cfg = config.load_config(config_path)
if hasattr(cfg, "architecture"):
    layers = cfg.architecture["layers"]
else:
    layers = cfg["architecture"]["layers"]
rng = np.random.default_rng(41)
params = [
    {
        "W": rng.normal(size=(out_dim, in_dim)).astype(np.float64),
        "b": rng.normal(size=(out_dim,)).astype(np.float64),
    }
    for in_dim, out_dim in zip(layers[:-1], layers[1:])
]

if module_name.endswith("punctures_2d"):
    metadata = checkpoint.build_metadata(cfg, stage="adam", step=1)
else:
    metadata = checkpoint.build_metadata(
        cfg, stage="adam", step=1, parent_checkpoint_sha256=None
    )
parent = output / "parent.npz"
checkpoint.save_checkpoint(parent, params, metadata)
loaded, loaded_metadata = checkpoint.load_checkpoint(parent)

if module_name.endswith("punctures_2d"):
    checkpoint.validate_resume(loaded_metadata, cfg)
else:
    checkpoint.validate_immutable_metadata(cfg, loaded_metadata)

legacy_metadata = copy.deepcopy(loaded_metadata)
collocation = legacy_metadata["collocation"]
if module_name.endswith("punctures_2d"):
    legacy_metadata["collocation"] = {
        "sampling": collocation["sampling"],
        "radial_points": collocation["Nxx0"],
        "theta_points": collocation["Nxx1"],
        "radial_min": collocation["xx0_min"],
        "radial_max": collocation["xx0_max"],
        "theta_min": collocation["xx1_min"],
        "theta_max": collocation["xx1_max"],
        "parity_delta_max": collocation["xx1_parity_delta_max"],
    }
else:
    legacy_metadata["collocation"] = {
        "radial_points": collocation["Nxx0"],
        "theta_points": collocation["Nxx1"],
        "equal_spin_sz_points": collocation["equal_spin_sz_points"],
    }
legacy = output / "legacy_collocation.npz"
checkpoint.save_checkpoint(legacy, loaded, legacy_metadata)
_, loaded_legacy_metadata = checkpoint.load_checkpoint(legacy)
if module_name.endswith("punctures_2d"):
    checkpoint.validate_resume(loaded_legacy_metadata, cfg)
else:
    checkpoint.validate_immutable_metadata(cfg, loaded_legacy_metadata)

parent_hash = (
    checkpoint.sha256_file(parent)
    if hasattr(checkpoint, "sha256_file")
    else checkpoint.file_sha256(parent)
)
if module_name.endswith("punctures_2d"):
    child_metadata = checkpoint.build_metadata(
        cfg,
        stage="ssbroyden",
        step=1,
        parent_checkpoint_sha256=parent_hash,
    )
else:
    child_metadata = checkpoint.build_metadata(
        cfg,
        stage="ssbroyden",
        step=1,
        parent_checkpoint_sha256=parent_hash,
    )
child = output / "child.npz"
checkpoint.save_checkpoint(child, loaded, child_metadata)
_, child_loaded_metadata = checkpoint.load_checkpoint(child)
if "provenance" in child_loaded_metadata:
    recorded_parent = child_loaded_metadata["provenance"]["parent_checkpoint_sha256"]
else:
    recorded_parent = child_loaded_metadata["parent_checkpoint_sha256"]

bad_metadata = copy.deepcopy(loaded_metadata)
bad_metadata["geometry"]["AMAX"] += 1.0
rejected = False
try:
    if module_name.endswith("punctures_2d"):
        checkpoint.validate_resume(bad_metadata, cfg)
    else:
        checkpoint.validate_immutable_metadata(cfg, bad_metadata)
except ValueError:
    rejected = True

with np.load(child, allow_pickle=False) as archive:
    metadata_dtype = str(archive["metadata_json"].dtype)
    names = sorted(archive.files)
print(json.dumps({
    "parent_hash": parent_hash,
    "recorded_parent": recorded_parent,
    "rejected": rejected,
    "metadata_dtype": metadata_dtype,
    "names": names,
    "legacy_collocation_accepted": True,
    "temporary_files": sorted(path.name for path in output.glob(".*.tmp")),
}))
"""
    environment = os.environ.copy()
    environment["PYTHONPYCACHEPREFIX"] = "/tmp/pinndorama_pycache"
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            solver,
            str(REPO_ROOT / configuration),
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    document = json.loads(result.stdout.splitlines()[-1])
    assert document["recorded_parent"] == document["parent_hash"]
    assert document["rejected"] is True
    assert document["metadata_dtype"] == "uint8"
    assert document["legacy_collocation_accepted"] is True
    assert document["names"][0] == "leaf_000"
    assert document["names"][-1] == "metadata_json"
    assert document["temporary_files"] == []


def test_3d_c04_to_c05_checkpoint_continuation(tmp_path: Path) -> None:
    script = r"""
import copy
import importlib
import json
import pathlib
import sys
import numpy as np

config_root = pathlib.Path(sys.argv[1]).resolve()
output = pathlib.Path(sys.argv[2])
config = importlib.import_module("pinndorama.solvers.punctures_3d.config")
checkpoint = importlib.import_module("pinndorama.solvers.punctures_3d.checkpoint")

c04 = config.load(config_root / "C04.toml")
c05 = config.load(config_root / "C05.toml")
rng = np.random.default_rng(42)
leaves = []
for in_dim, out_dim in zip(c04.architecture.layers[:-1], c04.architecture.layers[1:]):
    leaves.extend((
        rng.normal(size=(out_dim, in_dim)).astype(np.float64),
        rng.normal(size=(out_dim,)).astype(np.float64),
    ))
parent_metadata = checkpoint.build_metadata(
    c04, stage="ssbroyden", step=50000, parent_sha256=None
)
parent = output / "c04.npz"
parent_hash = checkpoint.save(parent, leaves, parent_metadata)
loaded_leaves, loaded_metadata, loaded_hash = checkpoint.load(parent)
checkpoint.validate_resume(c04, loaded_metadata)
# Optimizer and collocation changes are deliberately permitted.
checkpoint.validate_resume(c05, loaded_metadata)

legacy_metadata = copy.deepcopy(loaded_metadata)
collocation = legacy_metadata["collocation"]
legacy_metadata["collocation"] = {
    "radial_points": collocation["Nxx0"],
    "theta_points": collocation["Nxx1"],
    "phi_points": collocation["Nxx2"],
    "radial_sampling": collocation["xx0_sampling"],
    "radial_min": collocation["xx0_min"],
    "radial_max": collocation["xx0_max"],
    "theta_min": collocation["xx1_min"],
    "theta_max": collocation["xx1_max"],
    "phi_min": collocation["xx2_min"],
    "phi_max": collocation["xx2_max"],
    "boundary_radial_min": collocation["boundary_xx0_min"],
    "boundary_theta_epsilon": collocation["boundary_xx1_epsilon"],
    "theta_axis_delta_min": collocation["xx1_axis_delta_min"],
    "theta_axis_delta_max": collocation["xx1_axis_delta_max"],
}
legacy = output / "legacy_collocation.npz"
checkpoint.save(legacy, loaded_leaves, legacy_metadata)
_, loaded_legacy_metadata, _ = checkpoint.load(legacy)
checkpoint.validate_resume(c05, loaded_legacy_metadata)

child_metadata = checkpoint.build_metadata(
    c05, stage="ssbroyden", step=1, parent_sha256=parent_hash
)
child = output / "c05.npz"
checkpoint.save(child, loaded_leaves, child_metadata)
_, loaded_child_metadata, _ = checkpoint.load(child)

bad = copy.deepcopy(loaded_metadata)
bad["physics"]["bare_mass_0"] += 0.01
rejected = False
try:
    checkpoint.validate_resume(c05, bad)
except ValueError:
    rejected = True

with np.load(child, allow_pickle=False) as archive:
    metadata_dtype = str(archive["metadata_json"].dtype)
print(json.dumps({
    "hash_matches": parent_hash == loaded_hash,
    "parent": loaded_child_metadata["parent_checkpoint_sha256"],
    "rejected": rejected,
    "metadata_dtype": metadata_dtype,
    "c05_requires_resume": c05.continuation.require_resume,
    "c05_sampling": c05.collocation.xx0_sampling,
    "c05_inner": c05.collocation.Nxx0_inner,
    "legacy_collocation_accepted": True,
    "temporary_files": sorted(path.name for path in output.glob(".*.tmp")),
}))
"""
    environment = os.environ.copy()
    environment["PYTHONPYCACHEPREFIX"] = "/tmp/pinndorama_pycache"
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    result = subprocess.run(
        [
            sys.executable,
            "-c",
            script,
            str(REPO_ROOT / "configs/punctures_3d"),
            str(tmp_path),
        ],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    document = json.loads(result.stdout.splitlines()[-1])
    assert document == {
        "hash_matches": True,
        "parent": document["parent"],
        "rejected": True,
        "metadata_dtype": "uint8",
        "c05_requires_resume": True,
        "c05_sampling": "two-zone",
        "c05_inner": 236,
        "legacy_collocation_accepted": True,
        "temporary_files": [],
    }
    assert len(document["parent"]) == 64
