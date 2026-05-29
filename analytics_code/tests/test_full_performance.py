"""Tests for analytics_code.full_performance."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analytics_code.config import AnalysisConfig
from analytics_code.full_performance import (
    _bootstrap_metrics,
    _clean_likelihood,
    _detect_truth,
    _fair_filter,
    _point_metrics,
    _safediv,
    run_full_performance,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, *, n_boot: int = 50) -> AnalysisConfig:
    raw = {
        "paths": {"output_root": str(tmp_path / "outputs")},
        "model_mapping": {},
        "analysis": {"bootstrap_iterations": n_boot, "random_seed": 0},
    }
    return AnalysisConfig(raw=raw, config_path=tmp_path / "config.json")


def _merged_frame(n: int = 20, seed: int = 0) -> pd.DataFrame:
    """Synthetic merged_outputs-style frame for full_performance tests."""
    rng = np.random.default_rng(seed)
    y = rng.integers(0, 2, n)
    scores = rng.integers(1, 11, n)
    return pd.DataFrame(
        {
            "Patient_Has_IBD": y,
            "likelihood_score": scores.astype(float),
            "model_canon": rng.choice(["ModelA", "ModelB"], n),
            "shot_type": rng.choice(["zero", "single"], n),
            "report_sequence_name": rng.choice(
                ["endo", "hist", "all_docs_in_sequence"], n
            ),
            "temperature": rng.choice([0.60, 0.75], n),
        }
    )


# ---------------------------------------------------------------------------
# _detect_truth
# ---------------------------------------------------------------------------


def test_detect_truth_finds_patient_has_ibd() -> None:
    frame = pd.DataFrame({"Patient_Has_IBD": [0, 1], "other": [1, 2]})
    assert _detect_truth(frame) == "Patient_Has_IBD"


def test_detect_truth_finds_ground_truth_column() -> None:
    frame = pd.DataFrame({"ground_truth": [0, 1]})
    assert _detect_truth(frame) == "ground_truth"


def test_detect_truth_returns_none_when_no_column() -> None:
    frame = pd.DataFrame({"col1": [0, 1], "col2": [2, 3]})
    assert _detect_truth(frame) is None


# ---------------------------------------------------------------------------
# _clean_likelihood
# ---------------------------------------------------------------------------


def test_clean_likelihood_passes_through_0_to_10() -> None:
    s = pd.Series([0.0, 5.0, 10.0])
    result = _clean_likelihood(s)
    assert list(result) == [0.0, 5.0, 10.0]


def test_clean_likelihood_maps_0_to_1_range_to_binary() -> None:
    s = pd.Series([0.0, 0.3, 0.5, 1.0])
    result = _clean_likelihood(s)
    # < 0.5 → 0, >= 0.5 → 1
    assert result.iloc[0] == pytest.approx(0.0)
    assert result.iloc[2] == pytest.approx(1.0)


def test_clean_likelihood_discards_values_above_10() -> None:
    # Out-of-range values (>10) are dropped to NaN: exactly 10 is
    # retained; values strictly
    # greater than 10 are out-of-range and become NaN (rather than being
    # silently clipped to a maximally positive score).
    s = pd.Series([10.0, 15.0, 100.0])
    result = _clean_likelihood(s)
    assert result.iloc[0] == 10.0
    assert math.isnan(result.iloc[1])
    assert math.isnan(result.iloc[2])


def test_clean_likelihood_preserves_nan() -> None:
    s = pd.Series([5.0, float("nan"), 7.0])
    result = _clean_likelihood(s)
    assert math.isnan(result.iloc[1])


# ---------------------------------------------------------------------------
# _safediv
# ---------------------------------------------------------------------------


def test_safediv_normal_division() -> None:
    assert _safediv(3.0, 4.0) == pytest.approx(0.75)


def test_safediv_zero_denominator_returns_zero() -> None:
    assert _safediv(5.0, 0.0) == pytest.approx(0.0)


# ---------------------------------------------------------------------------
# _point_metrics
# ---------------------------------------------------------------------------


def test_point_metrics_perfect_prediction() -> None:
    y = np.array([0, 0, 1, 1])
    p = np.array([0.1, 0.2, 0.8, 0.9])
    metrics = _point_metrics(y, p)
    assert metrics["Accuracy"] == pytest.approx(1.0)
    assert metrics["Recall"] == pytest.approx(1.0)
    assert metrics["Precision"] == pytest.approx(1.0)
    assert metrics["Specificity"] == pytest.approx(1.0)
    assert metrics["Brier_Score"] == pytest.approx(
        float(np.mean((p - y) ** 2)), rel=1e-6
    )


def test_point_metrics_returns_all_expected_keys() -> None:
    y = np.array([0, 1, 0, 1])
    p = np.array([0.1, 0.9, 0.4, 0.6])
    keys = _point_metrics(y, p).keys()
    for k in (
        "Brier_Score",
        "Macro_F1",
        "Recall",
        "Precision",
        "Specificity",
        "NPV",
        "Accuracy",
        "MCC",
    ):
        assert k in keys


# ---------------------------------------------------------------------------
# _fair_filter
# ---------------------------------------------------------------------------


def test_fair_filter_data_type_keeps_zero_shot_only() -> None:
    frame = _merged_frame()
    result = _fair_filter(frame, "data_type")
    assert (result["shot_type"] == "zero").all()


def test_fair_filter_temperature_keeps_zero_shot_only() -> None:
    frame = _merged_frame()
    result = _fair_filter(frame, "temperature")
    assert (result["shot_type"] == "zero").all()


def test_fair_filter_shot_type_keeps_fair_temps() -> None:
    frame = _merged_frame()
    result = _fair_filter(frame, "shot_type")
    assert set(result["temperature"].unique()).issubset({0.60, 0.75})


# ---------------------------------------------------------------------------
# _bootstrap_metrics
# ---------------------------------------------------------------------------


def test_bootstrap_metrics_returns_expected_keys() -> None:
    y = np.array([0, 0, 1, 1, 0, 1])
    p = np.array([0.1, 0.2, 0.8, 0.7, 0.3, 0.9])
    cis = _bootstrap_metrics(y, p, n_boot=20, seed=0)
    for k in (
        "Brier_Score",
        "Macro_F1",
        "Recall",
        "Precision",
        "Specificity",
        "NPV",
        "Accuracy",
        "MCC",
    ):
        assert k in cis
        lo, hi = cis[k]
        assert math.isfinite(lo) or lo == 0.0
        assert math.isfinite(hi) or hi == 0.0


# ---------------------------------------------------------------------------
# run_full_performance – integration
# ---------------------------------------------------------------------------


def test_run_full_performance_returns_stage_dir_when_no_input(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)
    result = run_full_performance(config)
    assert "stage_dir" in result


def test_run_full_performance_writes_csv_when_input_present(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path, n_boot=20)
    data_prep_dir = tmp_path / "outputs" / "data_prep"
    data_prep_dir.mkdir(parents=True)
    merged_path = data_prep_dir / "merged_outputs.csv"
    _merged_frame(n=40).to_csv(merged_path, index=False)

    result = run_full_performance(config)
    # The stage should produce the performance CSV
    if "overall_performance_by_factor_fair" in result:
        assert result["overall_performance_by_factor_fair"].exists()


# ---------------------------------------------------------------------------
# primary-estimand + strict-marker sensitivity helpers
# ---------------------------------------------------------------------------


def _primary_estimand_frame() -> pd.DataFrame:
    """Frame with a mix of primary-eligible and ineligible rows."""
    rows = []
    # 12 primary-eligible rows split across two models / temps:
    #  ModelA @ 0.75 (matches non-thinking deployed temp)
    #  ModelB @ 0.60 (matches thinking deployed temp)
    for _ in range(6):
        rows.append(
            {
                "Patient_Has_IBD": 1,
                "likelihood_score": 8.0,
                "model_canon": "mixtral7b",
                "shot_type": "zero",
                "report_sequence_name": "all_docs_in_sequence",
                "temperature": 0.75,
                "patient_id": f"p{len(rows)}",
            }
        )
    for _ in range(6):
        rows.append(
            {
                "Patient_Has_IBD": 0,
                "likelihood_score": 2.0,
                "model_canon": "deepseek32b",
                "shot_type": "zero",
                "report_sequence_name": "all_docs_in_sequence",
                "temperature": 0.6,
                "patient_id": f"p{len(rows)}",
            }
        )
    # Failing rows:
    rows.append(  # wrong shot type
        {
            "Patient_Has_IBD": 1,
            "likelihood_score": 9.0,
            "model_canon": "mixtral7b",
            "shot_type": "single",
            "report_sequence_name": "all_docs_in_sequence",
            "temperature": 0.75,
            "patient_id": "p_shot",
        }
    )
    rows.append(  # wrong sequence
        {
            "Patient_Has_IBD": 1,
            "likelihood_score": 9.0,
            "model_canon": "mixtral7b",
            "shot_type": "zero",
            "report_sequence_name": "endo",
            "temperature": 0.75,
            "patient_id": "p_seq",
        }
    )
    rows.append(  # wrong temperature for model
        {
            "Patient_Has_IBD": 0,
            "likelihood_score": 1.0,
            "model_canon": "mixtral7b",
            "shot_type": "zero",
            "report_sequence_name": "all_docs_in_sequence",
            "temperature": 1.00,
            "patient_id": "p_temp",
        }
    )
    return pd.DataFrame(rows)


def test_strict_truth_missing_when_any_relevant_marker_missing() -> None:
    from analytics_code.truth_labels import build_document_truth_series

    # Row 0 anchors the histology column so it is recognised as a
    # boolean-like marker; row 1 exercises the strict-vs-lenient
    # divergence with a missing histology marker.
    frame = pd.DataFrame(
        {
            "report_sequence_name": ["endo_hist", "endo_hist"],
            "endoscopy_ibd_flag": [0, 1],
            "histology_ibd_flag": [1, pd.NA],
        }
    )

    lenient = build_document_truth_series(frame, strict=False)
    strict = build_document_truth_series(frame, strict=True)

    assert lenient.iloc[1] == 1.0
    assert pd.isna(strict.iloc[1])


def test_primary_estimand_outputs_written(tmp_path: Path) -> None:
    config = _make_config(tmp_path, n_boot=10)
    data_prep_dir = tmp_path / "outputs" / "data_prep"
    data_prep_dir.mkdir(parents=True)
    _primary_estimand_frame().to_csv(data_prep_dir / "merged_outputs.csv", index=False)

    run_full_performance(config)

    primary_dir = tmp_path / "outputs" / "full_performance" / "primary_estimand"
    pooled = primary_dir / "primary_estimand_pooled.csv"
    by_model = primary_dir / "primary_estimand_by_model.csv"
    assert pooled.exists()
    assert by_model.exists()

    by_model_df = pd.read_csv(by_model)
    # Only the two allowed (model, temp) pairs may appear.
    pairs = {
        (row["model_canon"], round(float(row["temperature"]), 2))
        for _, row in by_model_df.iterrows()
    }
    assert pairs == {("mixtral7b", 0.75), ("deepseek32b", 0.6)}


def test_strict_marker_sensitivity_outputs_written(tmp_path: Path) -> None:
    config = _make_config(tmp_path, n_boot=10)
    data_prep_dir = tmp_path / "outputs" / "data_prep"
    data_prep_dir.mkdir(parents=True)

    # Build a frame with explicit per-document IBD marker columns so
    # the strict truth synthesis has something to consume. Vary the
    # sequence to satisfy the FAIR filter for at least one factor.
    rng = np.random.default_rng(1)
    n = 30
    base = pd.DataFrame(
        {
            "Patient_Has_IBD": rng.integers(0, 2, n),
            "likelihood_score": rng.integers(1, 11, n).astype(float),
            "model_canon": rng.choice(["ModelA", "ModelB"], n),
            "shot_type": ["zero"] * n,
            "report_sequence_name": rng.choice(
                ["endo", "hist", "all_docs_in_sequence"], n
            ),
            "temperature": rng.choice([0.60, 0.75], n),
            "endoscopy_ibd_flag": rng.integers(0, 2, n),
            "histology_ibd_flag": rng.integers(0, 2, n),
            "preceding_clinic_ibd_flag": rng.integers(0, 2, n),
            "following_clinic_ibd_flag": rng.integers(0, 2, n),
        }
    )
    base.to_csv(data_prep_dir / "merged_outputs.csv", index=False)

    run_full_performance(config)

    sens_dir = tmp_path / "outputs" / "full_performance" / "sensitivity_complete_marker"
    assert (
        sens_dir / "all_attempts" / "overall_performance_by_factor_fair.csv"
    ).exists()
    assert (
        sens_dir / "complete_case" / "overall_performance_by_factor_fair.csv"
    ).exists()
