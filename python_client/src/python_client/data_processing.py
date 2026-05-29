"""Chronology dataframe assembly for the experiment runner.

Reads the three configured CSV sources (primary reports, secondary
reports, and longitudinal context notes), aligns them by patient id
with a configurable date tolerance, and produces the long-form
dataframe consumed by the experiment runner.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any

import pandas as pd


@dataclass(frozen=True)
class SourceFrame:
    """Aliases used when renaming columns on a per-source basis.

    Attributes
    ----------
    patient_id:
        Column holding the patient identifier in the source CSV.
    event_date:
        Column holding the event date used for chronological alignment.
    text:
        Column holding the free-text report content.
    """

    patient_id: str
    event_date: str
    text: str


def _read_source_csv(
    source_config: dict[str, Any], *, text_alias: str, date_alias: str
) -> pd.DataFrame:
    """Load a single source CSV and rename its columns to canonical aliases.

    Patient identifiers are upper-cased and stripped; the date column is
    parsed with the optional ``date_format`` from the source config.
    Rows missing either the patient id or the date are dropped.
    """
    frame = pd.read_csv(source_config["path"], low_memory=False)
    frame = frame.rename(
        columns={
            source_config["patient_id_column"]: "patient_id",
            source_config["date_column"]: date_alias,
            source_config["text_column"]: text_alias,
        }
    )
    frame["patient_id"] = frame["patient_id"].astype(str).str.strip().str.upper()
    frame[date_alias] = pd.to_datetime(
        frame[date_alias],
        format=source_config.get("date_format"),
        errors="coerce",
    )
    frame[text_alias] = frame[text_alias].fillna("").astype(str)
    frame = frame.dropna(subset=["patient_id", date_alias])
    return frame[["patient_id", date_alias, text_alias]].copy()


def build_chronology_dataframe(config: dict[str, Any]) -> pd.DataFrame:
    """Build the chronologically aligned dataframe used to drive experiments.

    The function joins primary and secondary reports on ``patient_id``
    within ``matching.max_days_between_primary_and_secondary`` days,
    then attaches the most recent preceding clinic letter and the
    earliest following clinic letter for each (patient, procedure) pair.
    Missing letters are replaced with ``"NOT_AVAILABLE"``.

    The returned frame contains the canonical columns consumed by
    :func:`python_client.prompts.prepare_reports_section`. When
    ``general.max_cases`` is set the frame is truncated to that number
    of rows.
    """
    sources = config["data_sources"]
    matching = config["matching"]
    max_cases = config["general"].get("max_cases")

    primary_df = _read_source_csv(
        sources["primary_reports"],
        text_alias="result_report",
        date_alias="sample_received_date",
    )
    secondary_df = _read_source_csv(
        sources["secondary_reports"],
        text_alias="Combined_Content",
        date_alias="procedure_date",
    )
    notes_df = _read_source_csv(
        sources["context_notes"],
        text_alias="clean_content",
        date_alias="date_creation",
    )

    merged_df = pd.merge(primary_df, secondary_df, on="patient_id", how="inner")
    merged_df["date_diff"] = (
        (merged_df["sample_received_date"] - merged_df["procedure_date"]).abs().dt.days
    )
    merged_df = merged_df[
        merged_df["date_diff"]
        <= int(matching["max_days_between_primary_and_secondary"])
    ].copy()

    # Deterministic primary/secondary pairing: for each (patient,
    # procedure_date) keep the histopathology report with the smallest
    # absolute date difference (and the earliest sample_received_date
    # as a stable tie-break) so the same chronology is reproduced on
    # repeated runs even when several primary reports fall within the
    # matching window.
    merged_df = merged_df.sort_values(
        ["patient_id", "procedure_date", "date_diff", "sample_received_date"],
        ascending=[True, True, True, True],
    )
    merged_df = merged_df.drop_duplicates(
        subset=["patient_id", "procedure_date"], keep="first"
    ).reset_index(drop=True)
    merged_df["episode_id"] = (
        merged_df["patient_id"].astype(str)
        + "::"
        + merged_df["procedure_date"].dt.strftime("%Y-%m-%d")
    )

    note_merge_df = pd.merge(
        merged_df,
        notes_df,
        on="patient_id",
        how="left",
    )
    note_merge_df["clinic_time_diff"] = (
        note_merge_df["date_creation"] - note_merge_df["procedure_date"]
    ).dt.days

    max_preceding = matching.get("max_days_preceding_clinic_letter")
    max_following = matching.get("max_days_following_clinic_letter")

    preceding_notes = note_merge_df[note_merge_df["clinic_time_diff"] < 0].copy()
    if max_preceding is not None:
        preceding_notes = preceding_notes[
            preceding_notes["clinic_time_diff"].abs() <= int(max_preceding)
        ]
    preceding_notes = preceding_notes.sort_values(
        ["patient_id", "procedure_date", "date_creation"],
        ascending=[True, True, False],
    )
    preceding_notes = (
        preceding_notes.groupby(["patient_id", "procedure_date"]).first().reset_index()
    )
    preceding_notes = preceding_notes.rename(
        columns={
            "clean_content": "preceding_clinic_letter",
            "date_creation": "preceding_clinic_date",
        }
    )

    following_notes = note_merge_df[note_merge_df["clinic_time_diff"] > 0].copy()
    if max_following is not None:
        following_notes = following_notes[
            following_notes["clinic_time_diff"] <= int(max_following)
        ]
    following_notes = following_notes.sort_values(
        ["patient_id", "procedure_date", "date_creation"],
        ascending=[True, True, True],
    )
    following_notes = (
        following_notes.groupby(["patient_id", "procedure_date"]).first().reset_index()
    )
    following_notes = following_notes.rename(
        columns={
            "clean_content": "following_clinic_letter",
            "date_creation": "following_clinic_date",
        }
    )

    final_df = pd.merge(
        merged_df,
        preceding_notes[
            [
                "patient_id",
                "procedure_date",
                "preceding_clinic_letter",
                "preceding_clinic_date",
            ]
        ],
        on=["patient_id", "procedure_date"],
        how="left",
    )
    final_df = pd.merge(
        final_df,
        following_notes[
            [
                "patient_id",
                "procedure_date",
                "following_clinic_letter",
                "following_clinic_date",
            ]
        ],
        on=["patient_id", "procedure_date"],
        how="left",
    )

    final_df["preceding_clinic_letter"] = final_df["preceding_clinic_letter"].fillna(
        "NOT_AVAILABLE"
    )
    final_df["following_clinic_letter"] = final_df["following_clinic_letter"].fillna(
        "NOT_AVAILABLE"
    )
    final_df["preceding_clinic_time_diff"] = (
        final_df["procedure_date"] - final_df["preceding_clinic_date"]
    ).dt.days
    final_df["following_clinic_time_diff"] = (
        final_df["following_clinic_date"] - final_df["procedure_date"]
    ).dt.days

    ordered_columns = [
        "patient_id",
        "episode_id",
        "procedure_date",
        "sample_received_date",
        "result_report",
        "Combined_Content",
        "preceding_clinic_letter",
        "following_clinic_letter",
        "preceding_clinic_date",
        "following_clinic_date",
        "preceding_clinic_time_diff",
        "following_clinic_time_diff",
        "date_diff",
    ]
    final_df = final_df[ordered_columns].copy()
    if max_cases is not None:
        final_df = final_df.head(int(max_cases)).copy()
    return final_df
