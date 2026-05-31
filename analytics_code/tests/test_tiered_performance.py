"""Tests for the Document / Cumulative / Final / Doc2Patient tier stage."""

from __future__ import annotations

from pathlib import Path

import pandas as pd
import pytest

from analytics_code.common import tier_of
from analytics_code.config import AnalysisConfig
from analytics_code.data_prep import (
    PATIENT_DOCUMENT_TRUTH,
    _attach_document_sequence_truth_columns,
)
from analytics_code.tiered_performance import (
    ALL_ATTEMPTS,
    COMPLETE_CASE,
    _cumulative_tier,
    _doc2patient_tier,
    _document_tier,
    _final_tier,
    _prepare_base,
    _resolve_flag_columns,
    run_tiered_performance,
)


def _build_frame() -> pd.DataFrame:
    """Tiny hand-checkable frame: 2 patients x 1 model x 1 shot x 1 temp.

    Patient P1 -- truly IBD positive (histology positive). Patient P2
    -- truly IBD negative.
    """
    rows = []
    # P1: positive histology flag, negative endoscopy, missing clinic.
    for data_type, like in [
        ("hist", 8),
        ("endo", 3),
        ("clinic_preceding", 4),
        ("clinic_following", 2),
        ("endo_hist", 7),
        ("clinic_both", 1),
        ("all_docs_in_sequence", 9),
    ]:
        rows.append(
            {
                "patient_id": "P1",
                "model_canon": "mixtral7b",
                "shot_type": "zero",
                "temperature": 0.75,
                "report_sequence_name": data_type,
                "likelihood_score": like,
                "histology_ibd_flag": 1,
                "endoscopy_ibd_flag": 0,
                "preceding_clinic_ibd_flag": None,
                "following_clinic_ibd_flag": None,
                "Patient_Has_IBD": 1,
            }
        )
    # P2: all flags negative or missing; chart label says 0.
    for data_type, like in [
        ("hist", 2),
        ("endo", 1),
        ("clinic_preceding", 0),
        ("clinic_following", 1),
        ("endo_hist", 2),
        ("clinic_both", 0),
        ("all_docs_in_sequence", 3),
    ]:
        rows.append(
            {
                "patient_id": "P2",
                "model_canon": "mixtral7b",
                "shot_type": "zero",
                "temperature": 0.75,
                "report_sequence_name": data_type,
                "likelihood_score": like,
                "histology_ibd_flag": 0,
                "endoscopy_ibd_flag": 0,
                "preceding_clinic_ibd_flag": 0,
                "following_clinic_ibd_flag": 0,
                "Patient_Has_IBD": 0,
            }
        )
    # One unparseable likelihood row for P2 (tests all_attempts policy).
    rows.append(
        {
            "patient_id": "P2",
            "model_canon": "mixtral7b",
            "shot_type": "zero",
            "temperature": 0.75,
            "report_sequence_name": "hist_endo",
            "likelihood_score": None,
            "histology_ibd_flag": 0,
            "endoscopy_ibd_flag": 0,
            "preceding_clinic_ibd_flag": 0,
            "following_clinic_ibd_flag": 0,
            "Patient_Has_IBD": 0,
        }
    )
    return pd.DataFrame(rows)


def test_tier_of_classifies_data_types() -> None:
    assert tier_of("hist") == "document"
    assert tier_of("endo") == "document"
    assert tier_of("clinic_preceding") == "document"
    assert tier_of("clinic_following") == "document"
    assert tier_of("all_docs_in_sequence") == "cumulative"
    assert tier_of("jumbled") == "cumulative"
    assert tier_of("nonsense") == "unknown"


def test_attach_patient_document_truth_is_or_across_flags() -> None:
    frame = _build_frame()
    out = _attach_document_sequence_truth_columns(frame)
    assert PATIENT_DOCUMENT_TRUTH in out.columns
    p1 = out.loc[out["patient_id"] == "P1", PATIENT_DOCUMENT_TRUTH].iloc[0]
    p2 = out.loc[out["patient_id"] == "P2", PATIENT_DOCUMENT_TRUTH].iloc[0]
    assert p1 == 1.0
    assert p2 == 0.0


def test_document_tier_scores_against_single_marker() -> None:
    frame = _attach_document_sequence_truth_columns(_build_frame())
    base = _prepare_base(frame, flag_columns=_resolve_flag_columns(_dummy_config()))
    overall, per_dt = _document_tier(
        base, policy=ALL_ATTEMPTS, n_boot=20, seed=1, patient_col="patient_id"
    )
    # Single (model, shot, temp) cell -> one row.
    assert len(overall) == 1
    # Four single-doc data types observed -> four per_data_type rows.
    assert set(per_dt["report_sequence_name"]) == {
        "hist",
        "endo",
        "clinic_preceding",
        "clinic_following",
    }
    # Per-doc-type sanity: P1 hist is TP (pred=1 vs truth=1) -> accuracy 1
    # across the two patients (P2 hist also correct: pred=0, truth=0).
    hist_row = per_dt[per_dt["report_sequence_name"] == "hist"].iloc[0]
    assert hist_row["Accuracy"] == pytest.approx(1.0)


def test_cumulative_tier_uses_or_over_shown_docs() -> None:
    frame = _attach_document_sequence_truth_columns(_build_frame())
    flag_columns = _resolve_flag_columns(_dummy_config())
    base = _prepare_base(frame, flag_columns=flag_columns)
    overall, per_dt = _cumulative_tier(
        base,
        policy=ALL_ATTEMPTS,
        n_boot=20,
        seed=1,
        patient_col="patient_id",
        flag_columns=flag_columns,
    )
    assert len(overall) == 1
    # endo_hist truth for P1 = OR(endo=0, hist=1) = 1; pred=1 -> correct.
    edhist = per_dt[per_dt["report_sequence_name"] == "endo_hist"]
    assert not edhist.empty


def test_cumulative_tier_uses_document_rows_not_direct_multi_doc_scores() -> None:
    frame = pd.DataFrame(
        {
            "patient_id": ["P1", "P1", "P1", "P2", "P2", "P2"],
            "model_canon": ["mixtral7b"] * 6,
            "shot_type": ["zero"] * 6,
            "temperature": [0.75] * 6,
            "report_sequence_name": [
                "hist",
                "endo",
                "endo_hist",
                "hist",
                "endo",
                "endo_hist",
            ],
            "likelihood_score": [8, 3, 0, 2, 1, 10],
            "histology_ibd_flag": [1, 1, 1, 0, 0, 0],
            "endoscopy_ibd_flag": [0, 0, 0, 0, 0, 0],
            "preceding_clinic_ibd_flag": [0, 0, 0, 0, 0, 0],
            "following_clinic_ibd_flag": [0, 0, 0, 0, 0, 0],
            "Patient_Has_IBD": [1, 1, 1, 0, 0, 0],
        }
    )
    frame = _attach_document_sequence_truth_columns(frame)
    flag_columns = _resolve_flag_columns(_dummy_config())
    base = _prepare_base(frame, flag_columns=flag_columns)

    _, per_dt = _cumulative_tier(
        base,
        policy=ALL_ATTEMPTS,
        n_boot=20,
        seed=1,
        patient_col="patient_id",
        flag_columns=flag_columns,
    )

    edhist = per_dt[per_dt["report_sequence_name"] == "endo_hist"].iloc[0]
    assert edhist["Accuracy"] == pytest.approx(1.0)


def test_final_tier_aggregates_one_row_per_patient_group() -> None:
    frame = _attach_document_sequence_truth_columns(_build_frame())
    base = _prepare_base(frame, flag_columns=_resolve_flag_columns(_dummy_config()))
    overall, per_model = _final_tier(
        base, policy=ALL_ATTEMPTS, n_boot=20, seed=1, patient_col="patient_id"
    )
    assert len(overall) == 1
    # Two patients -> n=2 in the per-group metrics row.
    assert int(overall["n"].iloc[0]) == 2
    # P1 OR of preds includes hist (8>=5) -> 1; truth=1.
    # P2 OR of preds: all rows <5 -> 0; truth=0.
    assert overall["Accuracy"].iloc[0] == pytest.approx(1.0)


def test_final_complete_case_drops_unparseable_rows() -> None:
    frame = _attach_document_sequence_truth_columns(_build_frame())
    base = _prepare_base(frame, flag_columns=_resolve_flag_columns(_dummy_config()))
    overall_attempt, _ = _final_tier(
        base, policy=ALL_ATTEMPTS, n_boot=20, seed=1, patient_col="patient_id"
    )
    overall_complete, _ = _final_tier(
        base, policy=COMPLETE_CASE, n_boot=20, seed=1, patient_col="patient_id"
    )
    # Both policies still see both patients (P2 has other rows besides
    # the unparseable one) so n=2 in both, and accuracy stays at 1.
    assert int(overall_complete["n"].iloc[0]) == 2
    assert overall_complete["Accuracy"].iloc[0] == pytest.approx(1.0)
    assert overall_attempt["Accuracy"].iloc[0] == pytest.approx(1.0)


def test_doc2patient_tier_uses_chart_truth() -> None:
    frame = _attach_document_sequence_truth_columns(_build_frame())
    base = _prepare_base(frame, flag_columns=_resolve_flag_columns(_dummy_config()))
    overall, per_model = _doc2patient_tier(
        base,
        policy=ALL_ATTEMPTS,
        n_boot=20,
        seed=1,
        patient_col="patient_id",
        chart_truth_col="Patient_Has_IBD",
    )
    assert len(overall) == 1
    assert int(overall["n"].iloc[0]) == 2
    assert overall["Accuracy"].iloc[0] == pytest.approx(1.0)


def test_run_tiered_performance_smoke(tmp_path: Path) -> None:
    frame = _attach_document_sequence_truth_columns(_build_frame())
    data_prep_dir = tmp_path / "data_prep"
    data_prep_dir.mkdir()
    frame.to_csv(data_prep_dir / "merged_outputs.csv", index=False)
    config = AnalysisConfig(
        raw={
            "paths": {"output_root": str(tmp_path)},
            "analysis": {"bootstrap_iterations": 20, "random_seed": 1},
        },
        config_path=tmp_path / "config.json",
    )
    outputs = run_tiered_performance(config)
    stage_root = tmp_path / "tiered_performance"
    for tier in ("Document", "Cumulative", "Final", "Doc2Patient"):
        for policy in (ALL_ATTEMPTS, COMPLETE_CASE):
            overall = stage_root / tier / policy / "overall.csv"
            assert overall.exists(), f"missing {overall}"
    # Final per_model.csv must have exactly one row per model_canon.
    final_per_model = pd.read_csv(stage_root / "Final" / ALL_ATTEMPTS / "per_model.csv")
    assert list(final_per_model["model_canon"]) == ["mixtral7b"]


def _dummy_config() -> AnalysisConfig:
    return AnalysisConfig(
        raw={"paths": {"output_root": "."}, "analysis": {}},
        config_path=Path("config.json"),
    )
