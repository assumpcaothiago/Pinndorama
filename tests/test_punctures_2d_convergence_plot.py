from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np
import pytest

from pinndorama.solvers.punctures_2d.plot_relative_l2_convergence import (
    load_records,
    output_paths,
    plot_aggregate,
    plot_seed_curves,
    summarize_records,
    write_metadata,
    write_summary,
)


FIELDS = (
    "checkpoint",
    "config_sha256",
    "nrpy_binary_sha256",
    "seed",
    "train_Nxx0",
    "train_Nxx1",
    "grid_mode",
    "eval_Nxx0",
    "eval_Nxx1",
    "sample_count",
    "radius_limit",
    "relative_l2",
)
CONFIG_HASH = "a" * 64
BINARY_HASH = "b" * 64


def _valid_rows() -> list[dict[str, str]]:
    rows: list[dict[str, str]] = []
    values = {
        (80, 503): 4.0e-4,
        (80, 504): 8.0e-4,
        (86, 503): 2.0e-4,
        (86, 504): 6.0e-4,
    }
    for (training_n, seed), relative_l2 in values.items():
        rows.append(
            {
                "checkpoint": f"/runs/N{training_n}/seed{seed}/checkpoint_final.npz",
                "config_sha256": CONFIG_HASH,
                "nrpy_binary_sha256": BINARY_HASH,
                "seed": str(seed),
                "train_Nxx0": str(training_n),
                "train_Nxx1": str(training_n),
                "grid_mode": "common",
                "eval_Nxx0": "760",
                "eval_Nxx1": "760",
                "sample_count": "99628",
                "radius_limit": "7.5",
                "relative_l2": f"{relative_l2:.17e}",
            }
        )
    return rows


def _write_table(path: Path, rows: list[dict[str, str]]) -> None:
    with path.open("w", encoding="utf-8", newline="") as stream:
        writer = csv.DictWriter(
            stream, fieldnames=FIELDS, delimiter="\t", lineterminator="\n"
        )
        writer.writeheader()
        writer.writerows(rows)


def test_loads_orders_and_summarizes_common_grid_data(tmp_path: Path) -> None:
    input_path = tmp_path / "relative_l2.tsv"
    _write_table(input_path, list(reversed(_valid_rows())))

    records, metadata = load_records(input_path)
    assert [(record.training_n, record.seed) for record in records] == [
        (80, 503),
        (80, 504),
        (86, 503),
        (86, 504),
    ]
    assert metadata.evaluation_nxx0 == 760
    assert metadata.evaluation_nxx1 == 760
    assert metadata.sample_count == 99628
    assert metadata.radius_limit == pytest.approx(7.5)
    assert metadata.seeds == (503, 504)
    assert metadata.resolutions == (80, 86)

    summary = summarize_records(records)
    assert [row["N"] for row in summary] == [80, 86]
    assert summary[0]["mean"] == pytest.approx(6.0e-4)
    assert summary[0]["median"] == pytest.approx(6.0e-4)
    assert summary[0]["std"] == pytest.approx(2.0e-4)
    assert summary[0]["q25"] == pytest.approx(5.0e-4)
    assert summary[0]["q75"] == pytest.approx(7.0e-4)


@pytest.mark.parametrize(
    ("field", "replacement", "message"),
    [
        ("grid_mode", "native", "grid_mode=common"),
        ("train_Nxx1", "81", "training grid must be square"),
        ("eval_Nxx0", "512", "one evaluation_nxx0"),
        ("eval_Nxx1", "512", "one evaluation_nxx1"),
        ("sample_count", "123", "one sample_count"),
        ("radius_limit", "10.0", "one radius_limit"),
        ("config_sha256", "c" * 64, "one config_sha256"),
        ("nrpy_binary_sha256", "d" * 64, "one nrpy_binary_sha256"),
        ("relative_l2", "nan", "positive and finite"),
    ],
)
def test_rejects_inconsistent_or_invalid_rows(
    tmp_path: Path, field: str, replacement: str, message: str
) -> None:
    rows = _valid_rows()
    rows[-1][field] = replacement
    input_path = tmp_path / "invalid.tsv"
    _write_table(input_path, rows)
    with pytest.raises(ValueError, match=message):
        load_records(input_path)


def test_rejects_duplicate_resolution_seed_pair(tmp_path: Path) -> None:
    rows = _valid_rows()
    rows.append(dict(rows[0]))
    input_path = tmp_path / "duplicate.tsv"
    _write_table(input_path, rows)
    with pytest.raises(ValueError, match="duplicate convergence row"):
        load_records(input_path)


def test_rejects_incomplete_seed_set(tmp_path: Path) -> None:
    rows = _valid_rows()
    rows.pop()
    input_path = tmp_path / "incomplete.tsv"
    _write_table(input_path, rows)
    with pytest.raises(ValueError, match="expected"):
        load_records(input_path)


def test_writes_figures_summary_and_metadata(tmp_path: Path) -> None:
    input_path = tmp_path / "relative_l2.tsv"
    _write_table(input_path, _valid_rows())
    records, metadata = load_records(input_path)
    summary = summarize_records(records)
    paths = output_paths(tmp_path / "c004_convergence")

    plot_seed_curves(
        records, metadata, paths["seed_png"], paths["seed_pdf"], dpi=100
    )
    plot_aggregate(
        records,
        summary,
        metadata,
        paths["aggregate_png"],
        paths["aggregate_pdf"],
        dpi=100,
    )
    write_summary(summary, paths["summary"])
    write_metadata(input_path, metadata, len(records), paths["metadata"])

    for path in paths.values():
        assert path.is_file()
        assert path.stat().st_size > 0
    with paths["summary"].open("r", encoding="utf-8", newline="") as stream:
        table = list(csv.DictReader(stream, delimiter="\t"))
    assert len(table) == 2
    assert table[0]["N"] == "80"
    report = json.loads(paths["metadata"].read_text(encoding="utf-8"))
    assert report["grid_mode"] == "common"
    assert report["evaluation_grid"] == [760, 760]
    assert report["seeds"] == [503, 504]
    assert report["resolutions"] == [80, 86]
    assert report["row_count"] == 4
    assert len(report["input_tsv_sha256"]) == 64
