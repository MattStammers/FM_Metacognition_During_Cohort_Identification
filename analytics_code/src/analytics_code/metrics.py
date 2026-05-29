"""Binary classification metrics with bootstrap confidence intervals.

This module provides reusable building blocks for the performance
stages of the pipeline: it computes the standard binary metrics from
confusion-matrix counts, optionally adds Brier score and ROC AUC when
probabilistic scores are available, and exposes a generic non-parametric
bootstrap helper used throughout the analysis stages.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd
from sklearn.metrics import brier_score_loss, roc_auc_score


@dataclass(slots=True)
class BinaryMetrics:
    """Container of point estimates for a binary classifier.

    Each attribute is the corresponding metric on a single sample.
    Confidence intervals are produced separately by
    :func:`bootstrap_metric`.
    """

    accuracy: float
    precision: float
    recall: float
    f1: float
    specificity: float
    npv: float
    mcc: float
    brier: float | None
    prevalence: float
    roc_auc: float | None


def coerce_binary(series: pd.Series, threshold: float = 5) -> pd.Series:
    """Coerce a numeric series to a 0/1 label series.

    If every non-null entry is already in ``{0, 1}`` the series is
    returned unchanged (with nullable integer dtype). Otherwise values
    are dichotomised using ``value >= threshold``.

    Parameters
    ----------
    series:
        Input series, typically a likelihood score.
    threshold:
        Decision threshold for the dichotomisation; defaults to ``5``.

    Returns
    -------
    pandas.Series
        Series of dtype ``Int64`` with values in ``{0, 1}``. Entries
        that could not be coerced to a number (originally ``NaN`` or
        non-numeric) are preserved as ``pandas.NA`` rather than being
        silently treated as ``0``; callers that prefer to count
        missing values as negatives should call
        :meth:`pandas.Series.fillna` upstream.
    """
    numeric = pd.to_numeric(series, errors="coerce")
    unique_values = set(numeric.dropna().unique().tolist())
    if unique_values and unique_values.issubset({0, 1, 0.0, 1.0}):
        return numeric.astype("Int64")
    result = pd.Series(pd.NA, index=numeric.index, dtype="Int64")
    valid = numeric.notna()
    result.loc[valid] = (numeric.loc[valid] >= threshold).astype("Int64")
    return result


def compute_binary_metrics(
    y_true: pd.Series, y_pred: pd.Series, y_score: pd.Series | None = None
) -> BinaryMetrics:
    """Compute point-estimate metrics for a binary classifier.

    Parameters
    ----------
    y_true:
        Ground-truth labels (0/1). Rows where either ``y_true`` or
        ``y_pred`` is missing are dropped.
    y_pred:
        Predicted labels (0/1).
    y_score:
        Optional probabilistic scores in ``[0, 10]`` (the pipeline's
        ``Likelihood of IBD`` scale). When provided, ROC AUC and Brier
        score are computed as well; the score is rescaled to ``[0, 1]``
        before computing the Brier loss.

    Returns
    -------
    BinaryMetrics
        Populated dataclass; metrics that cannot be computed (because
        denominators are zero or only one class is present) are returned
        as ``NaN`` or ``None``.
    """
    pair = pd.DataFrame({"y_true": y_true, "y_pred": y_pred})
    pair = pair.dropna()
    if pair.empty:
        return BinaryMetrics(
            np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, np.nan, None, np.nan, None
        )

    yt = pair["y_true"].astype(int)
    yp = pair["y_pred"].astype(int)
    tn = int(((yt == 0) & (yp == 0)).sum())
    fp = int(((yt == 0) & (yp == 1)).sum())
    fn = int(((yt == 1) & (yp == 0)).sum())
    tp = int(((yt == 1) & (yp == 1)).sum())
    total = len(pair)
    accuracy = float((tp + tn) / total) if total else np.nan
    precision = float(tp / (tp + fp)) if (tp + fp) else 0.0
    recall = float(tp / (tp + fn)) if (tp + fn) else 0.0
    f1 = (
        float((2 * precision * recall) / (precision + recall))
        if (precision + recall)
        else 0.0
    )
    specificity = float(tn / (tn + fp)) if (tn + fp) else np.nan
    npv = float(tn / (tn + fn)) if (tn + fn) else np.nan
    mcc_denom = float((tp + fp) * (tp + fn) * (tn + fp) * (tn + fn))
    mcc = (
        float(((tp * tn) - (fp * fn)) / np.sqrt(mcc_denom)) if mcc_denom > 0 else np.nan
    )

    auc = None
    brier = None
    if y_score is not None:
        score_df = pd.DataFrame({"y_true": y_true, "y_score": y_score}).dropna()
        if score_df["y_true"].nunique() > 1:
            auc = float(
                roc_auc_score(
                    score_df["y_true"].astype(int), score_df["y_score"].astype(float)
                )
            )
        if not score_df.empty:
            probabilities = (score_df["y_score"].astype(float) / 10.0).clip(0.0, 1.0)
            brier = float(
                brier_score_loss(score_df["y_true"].astype(int), probabilities)
            )

    return BinaryMetrics(
        accuracy=accuracy,
        precision=precision,
        recall=recall,
        f1=f1,
        specificity=specificity,
        npv=npv,
        mcc=mcc,
        brier=brier,
        prevalence=float(yt.mean()),
        roc_auc=auc,
    )


def bootstrap_metric(
    dataframe: pd.DataFrame,
    metric_fn,
    iterations: int = 250,
    random_seed: int = 42,
) -> tuple[float, float, float]:
    """Bootstrap a scalar metric over rows of a dataframe.

    Parameters
    ----------
    dataframe:
        Source dataframe; sampling is done with replacement at the row
        level.
    metric_fn:
        Callable that accepts a dataframe (a bootstrap sample) and
        returns a scalar metric.
    iterations:
        Number of bootstrap resamples. Defaults to ``250``.
    random_seed:
        Seed for the NumPy random generator that drives sampling.

    Returns
    -------
    tuple[float, float, float]
        ``(mean, lower_2.5%, upper_97.5%)`` of the bootstrap
        distribution. Returns three NaNs when ``dataframe`` is empty.
    """
    if dataframe.empty:
        return (np.nan, np.nan, np.nan)

    rng = np.random.default_rng(random_seed)
    samples = []
    for _ in range(iterations):
        sampled_idx = rng.integers(0, len(dataframe), len(dataframe))
        sample = dataframe.iloc[sampled_idx]
        samples.append(metric_fn(sample))

    values = np.asarray(samples, dtype=float)
    return (
        float(np.nanmean(values)),
        float(np.nanpercentile(values, 2.5)),
        float(np.nanpercentile(values, 97.5)),
    )
