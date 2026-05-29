"""Canonical prediction / score coercion for the analytics pipeline.

This module is the *single* place where the pipeline decides:

* how a raw likelihood score is normalised onto the ``[0, 10]`` grid;
* how that bucketed score is turned into a probability in ``[0, 1]``;
* how a binary prediction is derived from the bucketed score;
* how missing predictions are handled.

Every stage that scores model output must route through these helpers
so the same row is classified identically in `full_performance`,
`narrative_analysis`, `missingness_threshold`, and any future stage.

Design choices (see project decision log):

* The likelihood scale published in the prompts is ``0..10`` (integer
  rating). Some early models emit ``0..1`` probabilities; those are
  dichotomised at ``0.5`` for backwards compatibility.
* The decision threshold is ``>= 5``, matching the published bronze
  tier and the dummy-data behaviour validated in tests.
* The default missing-output policy is ``"negative"``: any row whose
  likelihood cannot be coerced to a number is treated as a negative
  prediction (``0``). Missing, null, or NaN responses are treated as
  negative for the relevant output field; this removes the silent
  dropout bias that previously caused metrics stages to overstate
  per-model accuracy.
"""

from __future__ import annotations

from typing import Literal

import numpy as np
import pandas as pd

#: Threshold (inclusive lower bound) on the ``0..10`` likelihood scale
#: above which a row is classified as a positive prediction. Matches
#: the "Bronze" tier.
DECISION_THRESHOLD: int = 5

MissingPolicy = Literal["negative", "drop"]
DEFAULT_MISSING_POLICY: MissingPolicy = "negative"


def clean_likelihood(series: pd.Series) -> pd.Series:
    """Snap a heterogeneous likelihood column to integers in ``[0, 10]``.

    * Values in ``[0, 1]`` are dichotomised at ``0.5`` (legacy 0..1
      probability scale).
    * Values in ``(1, 10)`` are rounded to the nearest integer.
    * Values equal to ``10`` are kept as ``10``.
    * Negative values and values strictly greater than ``10`` become
      ``NaN``: out-of-range likelihood values are discarded rather
      than clipped, so e.g. an erroneous ``100`` is not silently
      treated as a maximally positive score.
    * Non-numeric entries become ``NaN``.

    Returns a ``float64`` series so that NaN can be preserved
    upstream of any missing-output policy decision.
    """
    x = pd.to_numeric(series, errors="coerce")
    y = pd.Series(np.nan, index=x.index, dtype="float64")
    mask_01 = (x >= 0) & (x <= 1)
    y.loc[mask_01] = (x.loc[mask_01] >= 0.5).astype(int)
    mask_1_10 = (x > 1) & (x < 10)
    y.loc[mask_1_10] = np.floor(x.loc[mask_1_10] + 0.5)
    y.loc[x == 10] = 10
    # Values <0 or >10 already left as NaN by the masks above.
    return y


def likelihood_to_probability(bucket: pd.Series) -> pd.Series:
    """Map a cleaned ``0..10`` bucket to a probability in ``[0, 1]``."""
    return (pd.to_numeric(bucket, errors="coerce").astype(float) / 10.0).clip(0.0, 1.0)


def derive_prediction(
    likelihood: pd.Series,
    *,
    threshold: int = DECISION_THRESHOLD,
    missing_policy: MissingPolicy = DEFAULT_MISSING_POLICY,
) -> pd.Series:
    """Return a per-row binary prediction series.

    Parameters
    ----------
    likelihood:
        Raw likelihood column (any of the scales accepted by
        :func:`clean_likelihood`).
    threshold:
        Inclusive lower bound on the ``0..10`` bucket above which the
        prediction is positive.
    missing_policy:
        ``"negative"`` (default) — failed / missing rows become ``0``
        and the returned dtype is ``int64``.
        ``"drop"`` — failed / missing rows stay ``<NA>`` and the
        returned dtype is ``Int64`` (nullable). Callers that want to
        exclude failures from a metric should drop these rows
        themselves and report the dropout count separately.
    """
    bucket = clean_likelihood(likelihood)
    if missing_policy == "drop":
        result = pd.Series(pd.NA, index=bucket.index, dtype="Int64")
        valid = bucket.notna()
        result.loc[valid] = (bucket.loc[valid] >= threshold).astype("Int64")
        return result
    if missing_policy != "negative":
        raise ValueError(f"unknown missing_policy: {missing_policy!r}")
    bucket_filled = bucket.fillna(0.0)
    return (bucket_filled >= threshold).astype(int)


def combine_document_truths(scores: pd.DataFrame) -> pd.Series:
    """Aggregate per-document truth columns into a sequence-level truth.

    Rule: if *every* component column is missing the row stays
    ``<NA>``; otherwise the sequence truth is the boolean OR of the
    available components and missing components are treated as ``0``.
    This is the same conservative rule documented for
    :mod:`analytics_code.truth_labels` and removes the previous silent
    bias toward negative when only some component flags were missing.
    """
    if scores.empty:
        return pd.Series(dtype="Int64", index=scores.index)
    numeric = scores.apply(pd.to_numeric, errors="coerce")
    all_missing = numeric.isna().all(axis=1)
    any_positive = numeric.fillna(0).ge(1).any(axis=1).astype("Int64")
    any_positive[all_missing] = pd.NA
    return any_positive
