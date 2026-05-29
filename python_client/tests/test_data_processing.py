"""Tests for :mod:`python_client.data_processing`."""

from __future__ import annotations

import pandas as pd

from python_client.data_processing import build_chronology_dataframe


def test_build_chronology_dataframe_assigns_preceding_and_following_notes(
    tmp_path,
) -> None:
    primary = pd.DataFrame(
        [
            {
                "patient_id": "p1",
                "sample_received_date": "2024-01-11",
                "result_report": "Histology text",
            }
        ]
    )
    secondary = pd.DataFrame(
        [
            {
                "patient_id": "p1",
                "procedure_date": "2024-01-10",
                "Combined_Content": "Endoscopy text",
            }
        ]
    )
    notes = pd.DataFrame(
        [
            {
                "patient_id": "p1",
                "date_creation": "09-Jan-2024 09:00:00",
                "clean_content": "Before clinic",
            },
            {
                "patient_id": "p1",
                "date_creation": "12-Jan-2024 10:00:00",
                "clean_content": "After clinic",
            },
        ]
    )

    primary_path = tmp_path / "primary.csv"
    secondary_path = tmp_path / "secondary.csv"
    notes_path = tmp_path / "notes.csv"
    primary.to_csv(primary_path, index=False)
    secondary.to_csv(secondary_path, index=False)
    notes.to_csv(notes_path, index=False)

    config = {
        "general": {"max_cases": None},
        "matching": {"max_days_between_primary_and_secondary": 2},
        "data_sources": {
            "primary_reports": {
                "path": str(primary_path),
                "patient_id_column": "patient_id",
                "date_column": "sample_received_date",
                "text_column": "result_report",
                "date_format": "%Y-%m-%d",
            },
            "secondary_reports": {
                "path": str(secondary_path),
                "patient_id_column": "patient_id",
                "date_column": "procedure_date",
                "text_column": "Combined_Content",
                "date_format": "%Y-%m-%d",
            },
            "context_notes": {
                "path": str(notes_path),
                "patient_id_column": "patient_id",
                "date_column": "date_creation",
                "text_column": "clean_content",
                "date_format": "%d-%b-%Y %H:%M:%S",
            },
        },
    }

    result = build_chronology_dataframe(config)

    assert len(result) == 1
    row = result.iloc[0]
    assert row["patient_id"] == "P1"
    assert row["preceding_clinic_letter"] == "Before clinic"
    assert row["following_clinic_letter"] == "After clinic"
    assert row["date_diff"] == 1
