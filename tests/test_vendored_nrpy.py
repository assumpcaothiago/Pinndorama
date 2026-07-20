from __future__ import annotations

from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
RETIRED_SOURCE_NAME = "x" + "Punc"


def test_nrpy_provenance_is_pinned() -> None:
    provenance = (REPO_ROOT / "vendor" / "nrpy" / "VENDORED_FROM.md").read_text(
        encoding="utf-8"
    )
    assert "4d16f0e35ce90a50b87508088d2dfcf744aa65b9" in provenance
    assert 'version = "2.2026.6"' in (
        REPO_ROOT / "vendor" / "nrpy" / "nrpy" / "release.txt"
    ).read_text(encoding="utf-8")


def test_upstream_source_model_has_zpunc_only() -> None:
    source = (
        REPO_ROOT
        / "vendor"
        / "nrpy"
        / "nrpy"
        / "equations"
        / "nrpyelliptic"
        / "ConformallyFlat_SourceTerms.py"
    ).read_text(encoding="utf-8")
    assert "zPunc" in source
    assert RETIRED_SOURCE_NAME not in source
