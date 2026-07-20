from __future__ import annotations

import hashlib
import json
import os
from pathlib import Path
import subprocess
import sys

REPO_ROOT = Path(__file__).resolve().parents[1]
VENDOR_ROOT = REPO_ROOT / "vendor" / "ssbroyden"


def _checksummed_runtime_files() -> set[Path]:
    files = {
        VENDOR_ROOT / "optimistix_wrapper.py",
        VENDOR_ROOT / "ssbroyden_family.py",
    }
    files.update(
        path
        for path in (VENDOR_ROOT / "optimistix").rglob("*")
        if path.is_file() and path.name != "README.md"
    )
    return files


def test_shared_runtime_checksums_are_complete_and_current() -> None:
    manifest = VENDOR_ROOT / "SHA256SUMS"
    recorded: dict[Path, str] = {}
    for line in manifest.read_text(encoding="utf-8").splitlines():
        digest, relative = line.split("  ", maxsplit=1)
        recorded[VENDOR_ROOT / relative] = digest

    expected = _checksummed_runtime_files()
    assert set(recorded) == expected
    for path, digest in recorded.items():
        assert hashlib.sha256(path.read_bytes()).hexdigest() == digest


def test_shared_runtime_has_no_gitlinks() -> None:
    result = subprocess.run(
        ["git", "ls-files", "--stage", "vendor/ssbroyden"],
        cwd=REPO_ROOT,
        text=True,
        capture_output=True,
        check=True,
    )
    assert all(not line.startswith("160000 ") for line in result.stdout.splitlines())


def test_optimistix_import_does_not_query_distribution_metadata(tmp_path: Path) -> None:
    script = r"""
import importlib.metadata
import json
from pathlib import Path

import equinox
import jax
import jaxtyping
import lineax
import optax

def forbidden(*args, **kwargs):
    raise AssertionError("vendored Optimistix queried distribution metadata")

importlib.metadata.version = forbidden

from pinndorama._ssbroyden import (
    EXPECTED_OPTIMISTIX_VERSION,
    OPTIMISTIX_ROOT,
    SSBROYDEN_ROOT,
    load_ssbroyden_api,
)

solver, wrapper = load_ssbroyden_api()
import optimistix
import optimistix_wrapper
import ssbroyden_family

print(json.dumps({
    "version": optimistix.__version__,
    "expected_version": EXPECTED_OPTIMISTIX_VERSION,
    "optimistix": str(Path(optimistix.__file__).resolve()),
    "wrapper": str(Path(optimistix_wrapper.__file__).resolve()),
    "family": str(Path(ssbroyden_family.__file__).resolve()),
    "optimistix_root": str(OPTIMISTIX_ROOT.resolve()),
    "vendor_root": str(SSBROYDEN_ROOT.resolve()),
    "solver_name": solver.__name__,
    "wrapper_name": wrapper.__name__,
}))
"""
    environment = os.environ.copy()
    environment["PYTHONPATH"] = str(REPO_ROOT / "src")
    environment["PYTHONPYCACHEPREFIX"] = str(tmp_path / "pycache")
    result = subprocess.run(
        [sys.executable, "-c", script],
        cwd=REPO_ROOT,
        env=environment,
        text=True,
        capture_output=True,
        check=True,
    )
    document = json.loads(result.stdout)
    assert document["version"] == document["expected_version"] == "0.1.0"
    assert Path(document["optimistix"]).is_relative_to(
        Path(document["optimistix_root"])
    )
    for key in ("wrapper", "family"):
        assert Path(document[key]).is_relative_to(Path(document["vendor_root"]))
    assert document["solver_name"] == "SSBroyden"
    assert document["wrapper_name"] == "run_optimization"
