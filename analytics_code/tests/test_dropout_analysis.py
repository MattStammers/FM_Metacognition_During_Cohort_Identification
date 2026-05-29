"""Tests for analytics_code.dropout_analysis."""

from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

from analytics_code.config import AnalysisConfig
from analytics_code.dropout_analysis import (
    _compute_stats,
    _filtered,
    _parse_folder,
    _parse_percent_column,
    run_dropout_analysis,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path) -> AnalysisConfig:
    raw = {
        "paths": {"output_root": str(tmp_path / "outputs")},
        "model_mapping": {},
    }
    return AnalysisConfig(raw=raw, config_path=tmp_path / "config.json")


def _minimal_summary_df() -> pd.DataFrame:
    """Minimal summary frame that run_dropout_analysis can consume."""
    return pd.DataFrame(
        {
            "folder": [
                "deepseek_14b_t0_75_zero_shot_endo",
                "deepseek_14b_t0_75_zero_shot_hist",
                "qwen_32b_t0_60_zero_shot_endo",
                "deepseek_14b_t0_75_single_shot_endo",
            ],
            "total_success_percent": [90.0, 85.0, 80.0, 95.0],
            "percent_ge_5": [55.0, 60.0, 50.0, 70.0],
            "no_repair_percent": [70.0, 65.0, 60.0, 75.0],
        }
    )


# ---------------------------------------------------------------------------
# _parse_folder
# ---------------------------------------------------------------------------


def test_parse_folder_extracts_all_parts() -> None:
    result = _parse_folder("deepseek_14b_t0_75_zero_shot_endo")
    assert result["model"] is not None
    assert result["temperature"] is not None
    assert result["shot"] == "zero"
    assert result["data_type"] == "endo"


def test_parse_folder_extracts_single_shot() -> None:
    result = _parse_folder("model_t0_60_single_shot_hist")
    assert result["shot"] == "single"
    assert result["data_type"] == "hist"


def test_parse_folder_returns_none_for_missing_parts() -> None:
    result = _parse_folder("no_patterns_here")
    assert result["shot"] is None
    assert result["data_type"] is None


# ---------------------------------------------------------------------------
# _parse_percent_column
# ---------------------------------------------------------------------------


def test_parse_percent_column_handles_float_below_one() -> None:
    assert _parse_percent_column(0.85) == pytest.approx(85.0)


def test_parse_percent_column_handles_float_above_one() -> None:
    assert _parse_percent_column(75.0) == pytest.approx(75.0)


def test_parse_percent_column_handles_parenthetical_percent() -> None:
    assert _parse_percent_column("something (42.5%)") == pytest.approx(42.5)


def test_parse_percent_column_handles_boolean_true() -> None:
    assert _parse_percent_column(True) == pytest.approx(100.0)


def test_parse_percent_column_handles_boolean_false() -> None:
    assert _parse_percent_column(False) == pytest.approx(0.0)


def test_parse_percent_column_returns_nan_for_none() -> None:
    assert math.isnan(_parse_percent_column(None))


# ---------------------------------------------------------------------------
# _compute_stats
# ---------------------------------------------------------------------------


def test_compute_stats_returns_correct_mean() -> None:
    s = pd.Series([10.0, 20.0, 30.0])
    stats = _compute_stats(s)
    assert stats["mean"] == pytest.approx(20.0)


def test_compute_stats_returns_nan_keys_for_empty_series() -> None:
    stats = _compute_stats(pd.Series([], dtype=float))
    assert math.isnan(stats["mean"])
    assert math.isnan(stats["std_dev"])


def test_compute_stats_contains_all_expected_keys() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = _compute_stats(s)
    for key in (
        "mean",
        "std_dev",
        "n",
        "ci_lower",
        "ci_upper",
        "median",
        "iqr",
        "skewness",
        "kurtosis",
    ):
        assert key in stats


def test_compute_stats_ci_bounds_around_mean() -> None:
    s = pd.Series([10.0, 20.0, 30.0, 40.0, 50.0])
    stats = _compute_stats(s)
    assert stats["ci_lower"] <= stats["mean"] <= stats["ci_upper"]


def test_compute_stats_skewness_zero_for_symmetric_data() -> None:
    s = pd.Series([1.0, 2.0, 3.0, 4.0, 5.0])
    stats = _compute_stats(s)
    assert stats["skewness"] == pytest.approx(0.0, abs=1e-10)


# ---------------------------------------------------------------------------
# _filtered
# ---------------------------------------------------------------------------


def test_filtered_temperature_keeps_zero_shot_only() -> None:
    frame = pd.DataFrame(
        {
            "shot": ["zero", "single", "zero", "zero"],
            "temperature": ["0_75", "0_75", "0_60", "0_75"],
            "model": ["m1", "m1", "m1", "m1"],
            "data_type": ["endo", "endo", "hist", "off_matrix"],
        }
    )
    result = _filtered(frame, "temperature")
    # zero-shot only, FAIR data types only, both FAIR temperatures kept
    assert (result["shot"] == "zero").all()
    assert set(result["temperature"].unique()).issubset({"0_60", "0_75"})
    assert "off_matrix" not in result["data_type"].values


def test_filtered_shot_uses_fair_temperatures() -> None:
    frame = pd.DataFrame(
        {
            "shot": ["zero", "single", "dual"],
            "temperature": ["0_60", "0_75", "0_75"],
            "model": ["m1", "m2", "m1"],
            "data_type": ["all_docs_in_sequence", "endo", "hist"],
        }
    )
    result = _filtered(frame, "shot")
    assert set(result["temperature"].unique()).issubset({"0_60", "0_75"})
    assert set(result["data_type"].unique()).issubset(
        {"all_docs_in_sequence", "endo", "hist", "clinic_preceding", "clinic_following"}
    )


# ---------------------------------------------------------------------------
# run_dropout_analysis – integration (no figures generated)
# ---------------------------------------------------------------------------


def test_run_dropout_analysis_returns_empty_when_source_missing(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    result = run_dropout_analysis(config, source_path=tmp_path / "no_such.csv")
    assert result == {}


def test_run_dropout_analysis_returns_empty_when_folder_column_missing(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)
    csv_path = tmp_path / "summary.csv"
    pd.DataFrame({"not_folder": [1, 2]}).to_csv(csv_path, index=False)
    result = run_dropout_analysis(config, source_path=csv_path)
    assert result == {}


def test_run_dropout_analysis_produces_outputs(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    csv_path = tmp_path / "summary.csv"
    _minimal_summary_df().to_csv(csv_path, index=False)
    result = run_dropout_analysis(config, source_path=csv_path)
    # At least one output path should be returned
    assert len(result) > 0
    for path in result.values():
        assert path.exists()
