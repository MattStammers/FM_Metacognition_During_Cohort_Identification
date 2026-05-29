"""Tests for episode-id derivation in the chronology builder."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from python_client.data_processing import build_chronology_dataframe


def _write_csv(path: Path, frame: pd.DataFrame) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    frame.to_csv(path, index=False)


def _config(primary: Path, secondary: Path, notes: Path) -> dict:
    return {
        "general": {"max_cases": None},
        "matching": {"max_days_between_primary_and_secondary": 2},
        "data_sources": {
            "primary_reports": {
                "path": str(primary),
                "patient_id_column": "patient_id",
                "date_column": "sample_received_date",
                "text_column": "result_report",
                "date_format": "%Y-%m-%d",
            },
            "secondary_reports": {
                "path": str(secondary),
                "patient_id_column": "patient_id",
                "date_column": "procedure_date",
                "text_column": "Combined_Content",
                "date_format": "%Y-%m-%d",
            },
            "context_notes": {
                "path": str(notes),
                "patient_id_column": "patient_id",
                "date_column": "date_creation",
                "text_column": "clean_content",
                "date_format": "%Y-%m-%d",
            },
        },
    }


def test_episode_id_is_deterministic_per_patient_procedure(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "primary.csv",
        pd.DataFrame(
            {
                "patient_id": ["P1", "P1"],
                "sample_received_date": ["2024-01-01", "2024-01-02"],
                "result_report": ["a", "b"],
            }
        ),
    )
    _write_csv(
        tmp_path / "secondary.csv",
        pd.DataFrame(
            {
                "patient_id": ["P1"],
                "procedure_date": ["2024-01-01"],
                "Combined_Content": ["endo"],
            }
        ),
    )
    _write_csv(
        tmp_path / "notes.csv",
        pd.DataFrame(
            {
                "patient_id": ["P1"],
                "date_creation": ["2023-12-30"],
                "clean_content": ["clinic"],
            }
        ),
    )
    frame = build_chronology_dataframe(
        _config(
            tmp_path / "primary.csv", tmp_path / "secondary.csv", tmp_path / "notes.csv"
        )
    )
    # Nearest primary (date_diff=0) is kept; one episode per (patient,
    # procedure_date) and the id is a stable composite of both.
    assert len(frame) == 1
    assert frame.loc[0, "episode_id"] == "P1::2024-01-01"
    assert frame.loc[0, "result_report"] == "a"


def test_clinic_letter_window_excludes_distant_letters(tmp_path: Path) -> None:
    _write_csv(
        tmp_path / "primary.csv",
        pd.DataFrame(
            {
                "patient_id": ["P1"],
                "sample_received_date": ["2024-06-01"],
                "result_report": ["histo"],
            }
        ),
    )
    _write_csv(
        tmp_path / "secondary.csv",
        pd.DataFrame(
            {
                "patient_id": ["P1"],
                "procedure_date": ["2024-06-01"],
                "Combined_Content": ["endo"],
            }
        ),
    )
    _write_csv(
        tmp_path / "notes.csv",
        pd.DataFrame(
            {
                "patient_id": ["P1", "P1"],
                "date_creation": ["2020-01-01", "2024-05-25"],
                "clean_content": ["old", "recent"],
            }
        ),
    )
    config = _config(
        tmp_path / "primary.csv", tmp_path / "secondary.csv", tmp_path / "notes.csv"
    )
    config["matching"]["max_days_preceding_clinic_letter"] = 60
    frame = build_chronology_dataframe(config)
    assert frame.loc[0, "preceding_clinic_letter"] == "recent"
