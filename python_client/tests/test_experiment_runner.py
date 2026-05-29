"""Tests for :mod:`python_client.experiment_runner`."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from python_client.experiment_runner import (
    _batch_suffix,
    _build_result_column_map,
    _ensure_result_columns,
    ensure_dir,
    load_progress,
)


def test_ensure_dir_creates_path(tmp_path) -> None:
    created = ensure_dir(tmp_path / "a" / "b")
    assert created.exists()
    assert created.is_dir()


def test_batch_suffix_uses_expected_extension() -> None:
    assert _batch_suffix("csv") == ".csv"
    assert _batch_suffix("xlsx") == ".xlsx"


def test_load_progress_reads_highest_batch_number(tmp_path) -> None:
    output_dir = Path(tmp_path)
    (output_dir / "batch_0.xlsx").write_text("x", encoding="utf-8")
    (output_dir / "batch_2.xlsx").write_text("x", encoding="utf-8")
    assert load_progress(output_dir, "xlsx") == 3


def test_ensure_result_columns_adds_missing_fields() -> None:
    frame = pd.DataFrame({"patient_id": ["P1"]})
    result = _ensure_result_columns(
        frame,
        {
            "json_response": "runner_json",
            "full_response": "runner_full",
            "payload": "runner_payload",
            "truncated": "runner_truncated",
        },
    )
    assert list(result.columns) == [
        "patient_id",
        "runner_json",
        "runner_full",
        "runner_payload",
        "runner_truncated",
    ]
    assert bool(result.iloc[0]["runner_truncated"]) is False


def test_build_result_column_map_names_columns_consistently() -> None:
    columns = _build_result_column_map("mixtral_runner", "zero_shot_hist")
    assert columns["json_response"] == "mixtral_runner_Json_Response_zero_shot_hist"
    assert columns["truncated"] == "Truncated_zero_shot_hist"
