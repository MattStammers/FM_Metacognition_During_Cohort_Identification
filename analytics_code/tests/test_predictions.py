"""Tests for :mod:`analytics_code.predictions`."""

from __future__ import annotations

import pandas as pd

from analytics_code.predictions import (
    DECISION_THRESHOLD,
    clean_likelihood,
    combine_document_truths,
    derive_prediction,
    likelihood_to_probability,
)


def test_clean_likelihood_handles_mixed_scales() -> None:
    # Out-of-range values (>10) are
    # discarded (NaN) rather than clipped, so an erroneous 15 does
    # not become a maximally positive score.
    out = clean_likelihood(pd.Series([0.0, 0.3, 0.7, 3, 7.4, 9.6, 15, "bad", None]))
    assert list(out.fillna(-1).round(2)) == [
        0.0,
        0.0,
        1.0,
        3.0,
        7.0,
        10.0,
        -1.0,
        -1.0,
        -1.0,
    ]


def test_likelihood_to_probability_clips_and_scales() -> None:
    out = likelihood_to_probability(pd.Series([0, 5, 10, 15]))
    assert list(out) == [0.0, 0.5, 1.0, 1.0]


def test_derive_prediction_negative_policy_treats_nan_as_zero() -> None:
    out = derive_prediction(
        pd.Series([0, 4, 5, None, "junk"]), missing_policy="negative"
    )
    assert out.dtype.kind == "i"
    assert list(out) == [0, 0, 1, 0, 0]


def test_derive_prediction_drop_policy_keeps_na() -> None:
    out = derive_prediction(pd.Series([0, 4, 5, None, "junk"]), missing_policy="drop")
    assert str(out.dtype) == "Int64"
    # NaN entries preserved
    assert out.isna().sum() == 2
    assert list(out.dropna().astype(int)) == [0, 0, 1]


def test_decision_threshold_constant() -> None:
    assert DECISION_THRESHOLD == 5


def test_combine_document_truths_partial_missing_stays_unknown() -> None:
    scores = pd.DataFrame(
        {
            "endo": [1, 0, None, None],
            "hist": [0, None, 1, None],
        }
    )
    out = combine_document_truths(scores)
    assert str(out.dtype) == "Int64"
    # row 0: 1 or 0 -> 1; row 1: 0 (hist missing) -> 0; row 2: 1 -> 1;
    # row 3: all missing -> NA
    assert list(out[:3]) == [1, 0, 1]
    assert pd.isna(out.iloc[3])
