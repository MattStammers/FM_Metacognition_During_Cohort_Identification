"""Tests for analytics_code.metrics."""

from __future__ import annotations

import math

import numpy as np
import pandas as pd
import pytest

from analytics_code.metrics import (
    BinaryMetrics,
    bootstrap_metric,
    coerce_binary,
    compute_binary_metrics,
)

# ---------------------------------------------------------------------------
# coerce_binary
# ---------------------------------------------------------------------------


def test_coerce_binary_already_binary_values_preserved() -> None:
    s = pd.Series([0, 1, 0, 1])
    result = coerce_binary(s, threshold=5)
    assert list(result.dropna()) == [0, 1, 0, 1]


def test_coerce_binary_applies_threshold() -> None:
    s = pd.Series([3, 5, 7, 9])
    result = coerce_binary(s, threshold=5)
    assert list(result.dropna()) == [0, 1, 1, 1]


def test_coerce_binary_nan_input_becomes_na() -> None:
    # NaN inputs are preserved as pandas.NA rather than being silently
    # treated as 0; this avoids inflating the negative class with
    # missing values.
    s = pd.Series([1, None, 8])
    result = coerce_binary(s, threshold=5)
    assert result.iloc[0] == 0
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == 1


def test_coerce_binary_non_numeric_becomes_na() -> None:
    # non-numeric strings coerce to NaN and are preserved as pandas.NA.
    s = pd.Series(["high", "low", 7])
    result = coerce_binary(s, threshold=5)
    assert pd.isna(result.iloc[0])
    assert pd.isna(result.iloc[1])
    assert result.iloc[2] == 1


# ---------------------------------------------------------------------------
# compute_binary_metrics – perfect classifier
# ---------------------------------------------------------------------------


def test_compute_binary_metrics_perfect_classifier() -> None:
    y = pd.Series([0, 0, 1, 1])
    p = pd.Series([0, 0, 1, 1])
    m = compute_binary_metrics(y, p)
    assert m.accuracy == pytest.approx(1.0)
    assert m.precision == pytest.approx(1.0)
    assert m.recall == pytest.approx(1.0)
    assert m.f1 == pytest.approx(1.0)
    assert m.specificity == pytest.approx(1.0)
    assert m.npv == pytest.approx(1.0)
    assert m.mcc == pytest.approx(1.0)


def test_compute_binary_metrics_all_wrong() -> None:
    y = pd.Series([0, 0, 1, 1])
    p = pd.Series([1, 1, 0, 0])
    m = compute_binary_metrics(y, p)
    assert m.accuracy == pytest.approx(0.0)
    assert m.recall == pytest.approx(0.0)
    assert m.specificity == pytest.approx(0.0)


def test_compute_binary_metrics_returns_nan_for_empty() -> None:
    m = compute_binary_metrics(pd.Series([], dtype=float), pd.Series([], dtype=float))
    assert math.isnan(m.accuracy)


def test_compute_binary_metrics_drops_na_pairs() -> None:
    y = pd.Series([0, None, 1, 1])
    p = pd.Series([0, 1, None, 1])
    m = compute_binary_metrics(y, p)
    # Only rows (0,0) and (1,1) remain; perfect on those two rows
    assert m.accuracy == pytest.approx(1.0)


def test_compute_binary_metrics_prevalence() -> None:
    y = pd.Series([1, 1, 1, 0])
    p = pd.Series([1, 1, 0, 0])
    m = compute_binary_metrics(y, p)
    assert m.prevalence == pytest.approx(0.75)


# ---------------------------------------------------------------------------
# compute_binary_metrics – with score (AUC / Brier)
# ---------------------------------------------------------------------------


def test_compute_binary_metrics_with_score_computes_auc() -> None:
    y = pd.Series([0, 0, 1, 1])
    p = pd.Series([0, 0, 1, 1])
    score = pd.Series([1.0, 2.0, 8.0, 9.0])  # well separated
    m = compute_binary_metrics(y, p, y_score=score)
    assert m.roc_auc == pytest.approx(1.0)


def test_compute_binary_metrics_with_score_computes_brier() -> None:
    y = pd.Series([0, 1])
    p = pd.Series([0, 1])
    # scores of 0 and 10 → probabilities 0.0 and 1.0 → perfect Brier = 0
    score = pd.Series([0.0, 10.0])
    m = compute_binary_metrics(y, p, y_score=score)
    assert m.brier == pytest.approx(0.0)


def test_compute_binary_metrics_auc_none_when_single_class_score() -> None:
    y = pd.Series([1, 1, 1])
    p = pd.Series([1, 1, 1])
    score = pd.Series([7.0, 8.0, 9.0])
    m = compute_binary_metrics(y, p, y_score=score)
    assert m.roc_auc is None


# ---------------------------------------------------------------------------
# bootstrap_metric
# ---------------------------------------------------------------------------


def test_bootstrap_metric_returns_nan_for_empty_df() -> None:
    df = pd.DataFrame()
    mean, lo, hi = bootstrap_metric(df, lambda x: 1.0)
    assert math.isnan(mean)
    assert math.isnan(lo)
    assert math.isnan(hi)


def test_bootstrap_metric_mean_within_ci() -> None:
    rng = np.random.default_rng(0)
    df = pd.DataFrame({"v": rng.random(100)})
    mean, lo, hi = bootstrap_metric(
        df, lambda x: float(x["v"].mean()), iterations=200, random_seed=42
    )
    assert lo <= mean <= hi


def test_bootstrap_metric_constant_function_has_zero_spread() -> None:
    df = pd.DataFrame({"v": [1, 2, 3, 4, 5]})
    mean, lo, hi = bootstrap_metric(df, lambda _: 0.5, iterations=50, random_seed=0)
    assert mean == pytest.approx(0.5)
    assert lo == pytest.approx(0.5)
    assert hi == pytest.approx(0.5)
