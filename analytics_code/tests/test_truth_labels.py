"""Tests for document-level truth synthesis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analytics_code.config import AnalysisConfig
from analytics_code.truth_labels import (
    DOCUMENT_TRUTH_COLUMN,
    build_document_truth_series,
    prepare_truth_frame,
    truth_mode,
)


def _config(tmp_path: Path, *, mode: str = "patient") -> AnalysisConfig:
    return AnalysisConfig(
        raw={
            "paths": {"output_root": str(tmp_path / "outputs")},
            "analysis": {"truth_mode": mode},
        },
        config_path=tmp_path / "config.json",
    )


def test_build_document_truth_series_uses_only_active_documents() -> None:
    frame = pd.DataFrame(
        {
            "report_sequence_name": ["endo", "clinic_preceding", "endo_hist"],
            "endoscopy_ibd_flag": [1, 0, 0],
            "preceding_clinic_ibd_flag": [0, 1, 0],
            "histology_ibd_flag": [0, 0, 1],
            "ground_truth": [0, 0, 0],
        }
    )

    result = build_document_truth_series(frame)

    assert list(result) == [1.0, 1.0, 1.0]


def test_prepare_truth_frame_materializes_document_truth_column(tmp_path: Path) -> None:
    frame = pd.DataFrame(
        {
            "report_sequence_name": ["endo_hist", "hist"],
            "endoscopy_ibd_flag": [0, 0],
            "histology_ibd_flag": [1, 0],
        }
    )

    prepared, truth_col = prepare_truth_frame(
        frame, mode=truth_mode(_config(tmp_path, mode="document"))
    )

    assert truth_col == DOCUMENT_TRUTH_COLUMN
    assert list(prepared[DOCUMENT_TRUTH_COLUMN]) == [1.0, 0.0]


def test_prepare_truth_frame_preserves_patient_truth_mode() -> None:
    frame = pd.DataFrame({"ground_truth": [0, 1]})
    prepared, truth_col = prepare_truth_frame(frame, mode="patient")
    assert truth_col == "ground_truth"
    assert DOCUMENT_TRUTH_COLUMN not in prepared.columns


def test_build_document_truth_series_ignores_metadata_columns() -> None:
    frame = pd.DataFrame(
        {
            "report_sequence_name": ["clinic_preceding", "clinic_preceding"],
            "preceding_clinic_time_diff": [1, 1],
            "preceding_clinic_ibd_flag": [0, 1],
        }
    )

    result = build_document_truth_series(frame)

    assert list(result) == [0.0, 1.0]


def test_build_document_truth_series_returns_nan_without_explicit_marker_columns() -> None:
    frame = pd.DataFrame(
        {
            "report_sequence_name": ["clinic_preceding", "clinic_preceding"],
            "preceding_clinic_time_diff": [1, 1],
        }
    )

    result = build_document_truth_series(frame)

    assert result.isna().all()
