"""Inferential sensitivity analyses for the full-performance stage.

Implements two analyses used as experimental controls:

* :func:`mcnemar_pairwise_models` — exact McNemar test for paired
  binary correctness between every pair of models, conditioning on
  rows where both models produced an evaluable prediction for the
  same ``(patient, experiment)``.
* :func:`mixed_effects_logistic_correctness` — mixed-effects logistic
  regression of row-level correctness with fixed effects for model,
  prompt condition, document-context condition and temperature group,
  and a random intercept for patient.

These analyses are descriptive sensitivity checks; the primary
performance interpretation remains based on the patient-level
clustered bootstrap estimates of macro-F1 and MCC produced by
:mod:`analytics_code.full_performance`.
"""

from __future__ import annotations

import logging
from itertools import combinations

import numpy as np
import pandas as pd

from analytics_code.predictions import (
    DECISION_THRESHOLD,
    clean_likelihood,
    derive_prediction,
)

LOGGER = logging.getLogger("analytics_code.inference")


def _exact_mcnemar_p(n10: int, n01: int) -> float:
    """Return the two-sided exact McNemar p-value for discordant counts.

    Conditional on ``N_D = n10 + n01`` and ``H_0``, the number of
    discordant pairs falling in one cell follows ``Binomial(N_D, 0.5)``.
    When both discordant counts are zero the p-value is 1.0.
    """
    n_d = n10 + n01
    if n_d == 0:
        return 1.0
    m = min(n10, n01)
    # P(X <= m) for X ~ Binomial(N_D, 0.5).
    cum = sum(_binomial_coeff(n_d, k) for k in range(m + 1)) / (2.0**n_d)
    return float(min(1.0, 2.0 * cum))


def _binomial_coeff(n: int, k: int) -> int:
    """Return C(n, k); small-k optimised, no scipy dependency."""
    if k < 0 or k > n:
        return 0
    k = min(k, n - k)
    numerator = 1
    denominator = 1
    for i in range(1, k + 1):
        numerator *= n - (k - i)
        denominator *= i
    return numerator // denominator


def _row_correctness(frame: pd.DataFrame) -> pd.Series:
    """Return a 0/1 correctness flag using the prespecified threshold.

    Prefers the precomputed ``_y`` (truth) and ``_bucket`` (cleaned
    likelihood) columns that ``full_performance`` materialises, but
    falls back to computing them from the raw columns when called in
    isolation (e.g. from tests).
    """
    if "_y" in frame.columns and "_bucket" in frame.columns:
        truth = pd.to_numeric(frame["_y"], errors="coerce")
        likelihood = pd.to_numeric(frame["_bucket"], errors="coerce")
        prediction = (likelihood >= DECISION_THRESHOLD).where(likelihood.notna(), 0)
        return (prediction.astype("Int64") == truth.astype("Int64")).astype("Int64")
    likelihood = clean_likelihood(frame["likelihood_score"])
    prediction = derive_prediction(
        likelihood,
        threshold=DECISION_THRESHOLD,
        missing_policy="negative",
    )
    truth_col = next(
        (
            col
            for col in (
                "_document_level_truth",
                "ground_truth",
                "document_truth_label",
                "patient_truth_label",
                "truth_label",
            )
            if col in frame.columns
        ),
        None,
    )
    if truth_col is None:
        return pd.Series(np.nan, index=frame.index)
    truth = pd.to_numeric(frame[truth_col], errors="coerce")
    return (prediction.astype("Int64") == truth.astype("Int64")).astype("Int64")


def mcnemar_pairwise_models(frame: pd.DataFrame) -> pd.DataFrame:
    """Run pairwise exact McNemar tests across models.

    Only the rows where the two compared models share a
    ``(patient_id, experiment_name)`` key are used, preserving the
    paired structure required by the test. Returns an empty frame
    when fewer than two distinct models or no shared rows are present.
    """
    if "model_canon" not in frame.columns or "experiment_name" not in frame.columns:
        return pd.DataFrame()
    patient_col = next(
        (
            col
            for col in ("patient_id", "study_id", "PatientID")
            if col in frame.columns
        ),
        None,
    )
    if patient_col is None:
        return pd.DataFrame()

    working = frame.copy()
    working["_correct"] = _row_correctness(working)
    working = working.dropna(subset=["_correct"])
    if working.empty:
        return pd.DataFrame()
    working["_correct"] = working["_correct"].astype(int)

    pivot_index = [patient_col, "experiment_name"]
    pivot = working.pivot_table(
        index=pivot_index,
        columns="model_canon",
        values="_correct",
        aggfunc="first",
    ).dropna(how="all")
    if pivot.shape[1] < 2:
        return pd.DataFrame()

    rows: list[dict] = []
    for model_a, model_b in combinations(sorted(pivot.columns), 2):
        paired = pivot[[model_a, model_b]].dropna().astype(int)
        if paired.empty:
            continue
        a = paired[model_a].to_numpy()
        b = paired[model_b].to_numpy()
        n10 = int(((a == 1) & (b == 0)).sum())
        n01 = int(((a == 0) & (b == 1)).sum())
        n11 = int(((a == 1) & (b == 1)).sum())
        n00 = int(((a == 0) & (b == 0)).sum())
        rows.append(
            {
                "model_a": model_a,
                "model_b": model_b,
                "n_pairs": int(len(paired)),
                "n11": n11,
                "n10": n10,
                "n01": n01,
                "n00": n00,
                "p_exact_mcnemar": _exact_mcnemar_p(n10, n01),
                "accuracy_a": float(a.mean()),
                "accuracy_b": float(b.mean()),
                "accuracy_diff": float(a.mean() - b.mean()),
            }
        )
    return pd.DataFrame(rows)


def mixed_effects_logistic_correctness(
    frame: pd.DataFrame, patient_id_col: str | None
) -> pd.DataFrame | None:
    """Fit a mixed-effects logistic regression of row-level correctness.

    Generalised linear mixed model with binomial error and logit link:

        logit(P(correct_ij)) = beta0 + beta1*model + beta2*shot
                              + beta3*context + beta4*temperature_group
                              + u_i,   u_i ~ N(0, sigma_u^2)

    Fixed effects: ``model_canon``, ``shot_type``, ``report_sequence_name``
    and ``temperature_group``. Random intercept: patient.

    Primary fit: ``statsmodels.BinomialBayesMixedGLM`` (variational
    Bayes). Fallback: logistic GLM with patient-clustered robust
    standard errors. Returns a tidy coefficient frame, or ``None``
    when the frame is too small or no fixed effect varies.
    """
    if patient_id_col is None or patient_id_col not in frame.columns:
        LOGGER.info("Mixed-effects logistic skipped: no patient identifier column")
        return None
    try:
        import statsmodels.formula.api as smf  # type: ignore
    except Exception:  # pragma: no cover - optional dependency
        LOGGER.info(
            "Mixed-effects logistic skipped: statsmodels not installed in this env"
        )
        return None

    working = frame.copy()
    working["_correct"] = _row_correctness(working)
    working = working.dropna(subset=["_correct", patient_id_col]).reset_index(drop=True)
    if len(working) < 50:
        LOGGER.info("Mixed-effects logistic skipped: fewer than 50 evaluable rows")
        return None
    working["_correct"] = working["_correct"].astype(int)
    working[patient_id_col] = working[patient_id_col].astype(str)

    if "temperature" in working.columns:
        temp = pd.to_numeric(working["temperature"], errors="coerce").round(2)
        working["temperature_group"] = pd.cut(
            temp,
            bins=[-0.01, 0.55, 0.70, 0.85, 1.10],
            labels=["low", "midlow", "midhigh", "high"],
        ).astype(str)
    else:
        working["temperature_group"] = "unknown"

    # Cast all candidate categorical predictors to plain strings to
    # avoid statsmodels indexing errors with empty pandas Categorical
    # levels that can survive the dropna above.
    for col in ("model_canon", "shot_type", "report_sequence_name"):
        if col in working.columns:
            working[col] = working[col].astype(str)

    fixed_terms: list[str] = []
    has_model = (
        "model_canon" in working.columns
        and working["model_canon"].nunique(dropna=True) > 1
    )
    for term, column in (
        ("C(model_canon)", "model_canon"),
        ("C(shot_type)", "shot_type"),
        ("C(report_sequence_name)", "report_sequence_name"),
        ("C(temperature_group)", "temperature_group"),
    ):
        # Drop temperature_group when model is already in the design
        # matrix: temperature is fully confounded with model in the
        # released matrix, so including both creates a singular
        # design.
        if column == "temperature_group" and has_model:
            continue
        if column in working.columns and working[column].nunique(dropna=True) > 1:
            fixed_terms.append(term)
    if not fixed_terms:
        LOGGER.info("Mixed-effects logistic skipped: no varying fixed effects")
        return None

    formula = "_correct ~ " + " + ".join(fixed_terms)
    # Binomial GLMM with logit link and a patient-level random
    # intercept; variational-Bayes fit via BinomialBayesMixedGLM.
    # Falls back to a logistic GLM with patient-clustered robust SEs
    # if the GLMM cannot converge.
    try:
        import statsmodels.api as sm  # type: ignore
        from statsmodels.genmod.bayes_mixed_glm import (  # type: ignore
            BinomialBayesMixedGLM,
        )

        # Patsy-style random-intercept specification: 1 | patient
        random_formula = {"patient": "0 + C(" + patient_id_col + ")"}
        bayes_model = BinomialBayesMixedGLM.from_formula(
            formula,
            random_formula,
            data=working,
        )
        bayes_result = bayes_model.fit_vb()
    except Exception as exc:
        LOGGER.info(
            "GLMM (BinomialBayesMixedGLM) fit failed: %s; "
            "falling back to logistic GLM with cluster-robust SE",
            exc,
        )
        try:
            import statsmodels.api as sm  # type: ignore
            import statsmodels.formula.api as smf  # type: ignore

            glm_model = smf.glm(
                formula,
                data=working,
                family=sm.families.Binomial(),
            )
            result = glm_model.fit(
                cov_type="cluster",
                cov_kwds={"groups": working[patient_id_col].to_numpy()},
                disp=False,
            )
        except Exception as glm_exc:
            LOGGER.warning("Logistic GLM fallback also failed: %s", glm_exc)
            return None
        params = result.params
        bse = result.bse
        pvalues = result.pvalues
        tidy = pd.DataFrame(
            {
                "term": params.index,
                "estimate_log_odds": params.values,
                "std_error": bse.reindex(params.index).values,
                "p_value": pvalues.reindex(params.index).values,
            }
        )
        tidy["odds_ratio"] = np.exp(tidy["estimate_log_odds"])
        tidy["model_type"] = "logistic_glm_cluster_robust_fallback"
        tidy["formula"] = formula
        tidy["n_rows"] = int(len(working))
        return tidy

    # Tidy table from the variational-Bayes GLMM fit:
    # ``fe_mean`` / ``fe_sd`` are fixed-effect posterior means and SDs;
    # ``vcp_mean`` is the patient random-intercept posterior SD.
    fe_names = list(bayes_result.model.exog_names)
    fe_mean = np.asarray(bayes_result.fe_mean)
    fe_sd = np.asarray(bayes_result.fe_sd)
    z = fe_mean / np.where(fe_sd > 0, fe_sd, np.nan)
    from math import erf, sqrt

    pvals = 2.0 * (1.0 - np.array([0.5 * (1.0 + erf(abs(zi) / sqrt(2.0))) for zi in z]))
    tidy = pd.DataFrame(
        {
            "term": fe_names,
            "estimate_log_odds": fe_mean,
            "posterior_sd": fe_sd,
            "approx_p_value": pvals,
        }
    )
    tidy["odds_ratio"] = np.exp(tidy["estimate_log_odds"])

    # Append the patient random-intercept SD as a self-contained row.
    try:
        vcp_sd = float(np.asarray(bayes_result.vcp_mean)[0])
    except Exception:
        vcp_sd = float("nan")
    tidy = pd.concat(
        [
            tidy,
            pd.DataFrame(
                [
                    {
                        "term": "patient_random_intercept_sd",
                        "estimate_log_odds": vcp_sd,
                        "posterior_sd": float("nan"),
                        "approx_p_value": float("nan"),
                        "odds_ratio": float("nan"),
                    }
                ]
            ),
        ],
        ignore_index=True,
    )
    tidy["model_type"] = "binomial_bayes_mixed_glm_logit"
    tidy["formula"] = formula
    tidy["n_rows"] = int(len(working))
    return tidy
