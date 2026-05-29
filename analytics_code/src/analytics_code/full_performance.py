"""Full performance analysis stage.

Computes the FAIR-filtered overall performance table (Brier, Macro F1, Recall,
Precision, Specificity, NPV, Accuracy, MCC) with bootstrap CIs and writes one
horizontal bar plot per metric, coloured by factor (temperature=red,
shot_type=green, model=yellow, data_type=blue).
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score

from analytics_code.common import (
    FAIR_DATA_TYPES,
    FAIR_TEMPERATURES_NUMERIC,
    ZERO_SHOT_LABEL,
    ensure_dir,
    format_factor_value,
    nice_numeric_ticks,
    remove_tree,
    save_figure,
    sentence_case,
    set_publication_axes,
    write_dataframe,
)
from analytics_code.config import AnalysisConfig
from analytics_code.predictions import clean_likelihood, likelihood_to_probability
from analytics_code.truth_labels import (
    build_document_truth_series,
    prepare_truth_frame,
    truth_mode,
)

#: Sensitivity-only strict document-sequence truth column. Populated
#: by :func:`_attach_strict_document_truth` when relevant document
#: marker columns are present. A row is missing whenever *any*
#: relevant marker for its ``report_sequence_name`` is missing,
#: otherwise it is the OR of the relevant markers. This column is
#: never used as the default truth -- it is consumed only by the
#: strict-marker sensitivity outputs.
STRICT_DOC_TRUTH_COLUMN = "_document_sequence_truth_strict"

#: Deployed pool temperature per canonical model. Mirrors the mapping
#: applied by :mod:`analytics_code.data_prep` when computing
#: ``is_primary_configuration``. Kept here so the primary-estimand
#: subset can still be derived when an input frame predates the
#: column.
PRIMARY_TEMPERATURE_BY_CANON: dict[str, float] = {
    "mixtral7b": 0.75,
    "m42_8b": 0.75,
    "deepseek14b": 0.75,
    "deepseek32b": 0.6,
    "deepseek70b": 0.6,
    "qwen32b": 0.6,
    "gemma4_31b": 0.6,
}

LOGGER = logging.getLogger("analytics_code.full_performance")


def _clear_stage_outputs(stage_dir: Path) -> None:
    """Remove stale full-performance artefacts when the stage is skipped."""
    if stage_dir.exists():
        remove_tree(stage_dir)
    ensure_dir(stage_dir)


# Re-exported for backwards compatibility with notebooks and tests
# that imported the constants from this module before they were
# centralised in :mod:`analytics_code.common`.
FAIR_TEMP_SET = set(FAIR_TEMPERATURES_NUMERIC)
ZERO_SHOT_DTYPES = set(FAIR_DATA_TYPES)

FACTOR_COLUMN = {
    "shot_type": "shot_type",
    "model": "model_canon",
    "temperature": "temperature",
    "data_type": "report_sequence_name",
}
FACTOR_COLOR = {
    "shot_type": "#2E7D32",
    "model": "#E0B400",
    "temperature": "#D32F2F",
    "data_type": "#1E88E5",
}
FACTOR_ORDER = ["shot_type", "model", "temperature", "data_type"]

# Factors that are intrinsically confounded with another dimension in
# the released production matrix. The production servers crossed each
# model with exactly one serving temperature (mixtral / m42 /
# deepseek14 at 0.75; deepseek32 / qwen32 at 0.60), so "temperature"
# is collinear with "model" and the per-temperature row in the
# overall-performance table is therefore not an independent
# experimental contrast. The factor is suppressed by default and can
# be re-enabled by setting ``analysis.enable_temperature_analysis``
# to ``true`` in the config (for example when a real model x
# temperature matrix has been run separately).
CONFOUNDED_FACTORS: frozenset[str] = frozenset({"temperature"})
METRICS = (
    "Brier_Score",
    "Macro_F1",
    "Recall",
    "Precision",
    "Specificity",
    "NPV",
    "Accuracy",
    "MCC",
)
TRUTH_CANDIDATES = ("Patient_Has_IBD", "ground_truth", "label", "gold_label")


def _detect_truth(frame: pd.DataFrame) -> str | None:
    """Return the first ground-truth column name present in ``frame``."""
    _, truth_col = prepare_truth_frame(
        frame, mode="patient", candidates=TRUTH_CANDIDATES
    )
    return truth_col


def _clean_likelihood(series: pd.Series) -> pd.Series:
    """Deprecated alias for :func:`analytics_code.predictions.clean_likelihood`.

    Retained so external notebooks / tests that imported the private
    helper keep working. New code should call ``clean_likelihood``
    from :mod:`analytics_code.predictions` directly.
    """
    return clean_likelihood(series)


def _fair_filter(frame: pd.DataFrame, factor: str) -> pd.DataFrame:
    """Apply the FAIR filter for the requested ``factor``.

    Each per-factor comparison is restricted to a sub-cube of the full
    experimental matrix in which every dimension *other than the
    factor under study* is held to a fixed, balanced subset. The rules
    are:

    * ``shot_type``: temperature in :data:`FAIR_TEMPERATURES_NUMERIC`
      and ``report_sequence_name`` in :data:`FAIR_DATA_TYPES`. Models
      remain pooled because each model is run with every shot type.
    * ``model``: zero-shot prompts only, temperature in
      :data:`FAIR_TEMPERATURES_NUMERIC` and ``report_sequence_name`` in
      :data:`FAIR_DATA_TYPES`.
    * ``temperature``: zero-shot prompts only and
      ``report_sequence_name`` in :data:`FAIR_DATA_TYPES`. The
      production matrix only runs each model at a single temperature,
      so this comparison is intrinsically confounded with model and
      should be interpreted with that caveat (the per-row ``model``
      mix is reported alongside).
    * ``data_type``: zero-shot prompts only and temperature in
      :data:`FAIR_TEMPERATURES_NUMERIC`.
    """
    fair_temps = {round(t, 2) for t in FAIR_TEMPERATURES_NUMERIC}
    fair_dtypes = set(FAIR_DATA_TYPES)
    df = frame
    if factor == "shot_type":
        df = df[df["temperature"].isin(fair_temps)]
        df = df[df["report_sequence_name"].isin(fair_dtypes)]
    elif factor == "model":
        df = df[df["shot_type"] == ZERO_SHOT_LABEL]
        df = df[df["temperature"].isin(fair_temps)]
        df = df[df["report_sequence_name"].isin(fair_dtypes)]
    elif factor == "temperature":
        df = df[df["shot_type"] == ZERO_SHOT_LABEL]
        df = df[df["report_sequence_name"].isin(fair_dtypes)]
    elif factor == "data_type":
        df = df[df["shot_type"] == ZERO_SHOT_LABEL]
        df = df[df["temperature"].isin(fair_temps)]
    return df


def _conf_counts(y: np.ndarray, pred: np.ndarray) -> tuple[int, int, int, int]:
    """Return ``(tp, fp, tn, fn)`` for two 0/1 arrays of equal length."""
    tp = int(((y == 1) & (pred == 1)).sum())
    tn = int(((y == 0) & (pred == 0)).sum())
    fp = int(((y == 0) & (pred == 1)).sum())
    fn = int(((y == 1) & (pred == 0)).sum())
    return tp, fp, tn, fn


def _safediv(num: float, den: float) -> float:
    """Return ``num / den``, or ``0.0`` when ``den <= 0``."""
    return float(num) / float(den) if den > 0 else 0.0


def _point_metrics(y: np.ndarray, p: np.ndarray) -> dict[str, float]:
    """Compute the point estimate of every metric in :data:`METRICS`.

    ``p`` is the probabilistic score in ``[0, 1]``; the binary
    prediction is taken as ``p >= 0.5``.
    """
    pred = (p >= 0.5).astype(int)
    tp, fp, tn, fn = _conf_counts(y, pred)
    return {
        "Brier_Score": float(np.mean((p - y) ** 2)),
        "Macro_F1": float(f1_score(y, pred, average="macro", zero_division=0)),
        "Recall": float(recall_score(y, pred, zero_division=0)),
        "MCC": float(matthews_corrcoef(y, pred))
        if pred.std() > 0 and y.std() > 0
        else 0.0,
        "Precision": float(precision_score(y, pred, zero_division=0)),
        "Accuracy": _safediv(tp + tn, len(y)),
        "Specificity": _safediv(tn, tn + fp),
        "NPV": _safediv(tn, tn + fn),
    }


def _bootstrap_metrics(
    y: np.ndarray,
    p: np.ndarray,
    n_boot: int,
    seed: int,
    patient_ids: np.ndarray | None = None,
) -> dict[str, tuple[float, float]]:
    """Bootstrap each metric in :data:`METRICS` and return ``(lo, hi)`` arms.

    When ``patient_ids`` is supplied, sampling is performed at the
    patient level (clustered bootstrap): patients are sampled with
    replacement and all rows belonging to each sampled patient are
    retained, in line with the methodology specification ("patient-
    level clustered bootstrap"). Otherwise sampling falls back to the
    row-level scheme.

    Returns a mapping from metric name to ``(mean - p2.5, p97.5 - mean)``
    suitable for direct use as ``xerr`` in a matplotlib bar chart.
    """
    rng = np.random.default_rng(seed)
    n = len(y)
    arrs = {m: np.empty(n_boot, dtype=float) for m in METRICS}

    if patient_ids is not None and len(patient_ids) == n:
        unique_patients, inverse = np.unique(patient_ids, return_inverse=True)
        # Pre-index rows by patient for fast lookup.
        patient_to_rows: list[np.ndarray] = [
            np.flatnonzero(inverse == idx) for idx in range(len(unique_patients))
        ]
        n_patients = len(unique_patients)
        for b in range(n_boot):
            sampled = rng.integers(0, n_patients, n_patients)
            row_idx = np.concatenate([patient_to_rows[i] for i in sampled])
            met = _point_metrics(y[row_idx], p[row_idx])
            for m in METRICS:
                arrs[m][b] = met[m]
    else:
        for b in range(n_boot):
            idx = rng.integers(0, n, n)
            met = _point_metrics(y[idx], p[idx])
            for m in METRICS:
                arrs[m][b] = met[m]

    out: dict[str, tuple[float, float]] = {}
    for m in METRICS:
        lo = float(np.percentile(arrs[m], 2.5))
        hi = float(np.percentile(arrs[m], 97.5))
        mean = float(arrs[m].mean())
        out[m] = (mean - lo, hi - mean)
    return out


def _plot_metric(perf: pd.DataFrame, metric: str, out_dir: Path) -> None:
    """Render the FAIR-filtered horizontal bar plot for a single metric."""
    ascending = metric == "Brier_Score"
    labels: list[str] = []
    means: list[float] = []
    errors: list[tuple[float, float]] = []
    colors: list[str] = []
    positions: list[float] = []
    cur = 0.0
    gap = 0.7
    for factor in FACTOR_ORDER:
        sub = perf[perf["factor"] == factor].sort_values(metric, ascending=ascending)
        if sub.empty:
            continue
        for _, row in sub.iterrows():
            display_value = format_factor_value(factor, row["value"])
            labels.append(f"{sentence_case(factor)}: {display_value}")
            means.append(float(row[metric]))
            errors.append(
                (
                    float(row.get(f"{metric}_lo", 0.0)),
                    float(row.get(f"{metric}_hi", 0.0)),
                )
            )
            colors.append(FACTOR_COLOR[factor])
            positions.append(cur)
            cur += 1.0
        cur += gap
    if not positions:
        return
    fig, ax = plt.subplots(figsize=(11.6, max(4.2, 0.38 * len(positions) + 2.0)))
    ax.barh(positions, means, color=colors, linewidth=1.0, edgecolor="black", zorder=3)
    for pos, m, (lo, hi) in zip(positions, means, errors):
        if not np.isfinite(m) or (lo == 0.0 and hi == 0.0):
            continue
        ax.errorbar(
            x=m,
            y=pos,
            xerr=np.array([[max(lo, 0.0)], [max(hi, 0.0)]]),
            fmt="none",
            ecolor="black",
            elinewidth=1.3,
            capsize=3.2,
            capthick=1.3,
            zorder=4,
        )
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(sentence_case(metric))
    ax.set_title(f"{sentence_case(metric)} by factor")
    ax.set_xlim(0.0, 1.02)
    ax.set_xticks(nice_numeric_ticks(0.0, 1.0, 0.1))
    set_publication_axes(ax, show_grid_y=False, show_grid_x=True)
    fig.tight_layout()
    save_figure(out_dir / f"{metric.lower()}_by_factors_fair_bootstrap.png")


def _run_factor_loop(
    working: pd.DataFrame,
    *,
    denominator: str,
    n_boot: int,
    seed: int,
    patient_id_col: str | None,
    skip_factors: set[str],
) -> list[dict]:
    """Per-factor metric loop shared by the standard and sensitivity passes.

    ``working`` is expected to expose ``_y`` (0/1 truth) and ``_p``
    (probability in ``[0, 1]``) columns alongside the usual factor
    columns. Returns one row per (factor, value) pair, tagged with
    the supplied ``denominator`` label.
    """
    rows: list[dict] = []
    for factor, col in FACTOR_COLUMN.items():
        if factor in skip_factors:
            LOGGER.info(
                "Skipping per-%s comparison: confounded with model in the released "
                "matrix. Set analysis.enable_temperature_analysis=true to override.",
                factor,
            )
            continue
        base = _fair_filter(working, factor)
        if base.empty or col not in base.columns:
            continue
        for value, sub in base.groupby(col, dropna=False):
            y = sub["_y"].to_numpy()
            p = sub["_p"].to_numpy()
            if len(y) == 0:
                continue
            point = _point_metrics(y, p)
            patient_ids = sub[patient_id_col].to_numpy() if patient_id_col else None
            cis = _bootstrap_metrics(
                y, p, n_boot=n_boot, seed=seed, patient_ids=patient_ids
            )
            row = {
                "factor": factor,
                "value": str(value),
                "n": int(len(y)),
                "denominator": denominator,
            }
            for metric in METRICS:
                row[metric] = point[metric]
                row[f"{metric}_lo"] = cis[metric][0]
                row[f"{metric}_hi"] = cis[metric][1]
            rows.append(row)
    return rows


def _metric_row(
    working: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
    patient_id_col: str | None,
    extras: dict,
) -> dict | None:
    """Compute one metric row (point estimate + clustered bootstrap CI)."""
    y = working["_y"].to_numpy()
    p = working["_p"].to_numpy()
    if len(y) == 0:
        return None
    point = _point_metrics(y, p)
    patient_ids = (
        working[patient_id_col].to_numpy()
        if patient_id_col and patient_id_col in working.columns
        else None
    )
    cis = _bootstrap_metrics(y, p, n_boot=n_boot, seed=seed, patient_ids=patient_ids)
    row: dict = {"n": int(len(y)), **extras}
    for metric in METRICS:
        row[metric] = point[metric]
        row[f"{metric}_lo"] = cis[metric][0]
        row[f"{metric}_hi"] = cis[metric][1]
    return row


def _is_primary_row(row: pd.Series) -> bool:
    """Apply the primary-configuration rule (used as a fallback)."""
    if str(row.get("shot_type")) != ZERO_SHOT_LABEL:
        return False
    if str(row.get("report_sequence_name")) != "all_docs_in_sequence":
        return False
    canon = str(row.get("model_canon"))
    expected = PRIMARY_TEMPERATURE_BY_CANON.get(canon)
    if expected is None:
        return False
    try:
        temp_f = float(row.get("temperature"))
    except (TypeError, ValueError):
        return False
    return abs(temp_f - expected) < 1e-6


def _primary_estimand_subset(attempts_frame: pd.DataFrame) -> pd.DataFrame:
    """Return the primary-estimand rows from an all-attempts frame."""
    if "is_primary_configuration" in attempts_frame.columns:
        mask = attempts_frame["is_primary_configuration"].fillna(False).astype(bool)
        return attempts_frame.loc[mask].copy()
    return attempts_frame[attempts_frame.apply(_is_primary_row, axis=1)].copy()


def _write_primary_estimand_outputs(
    attempts_frame: pd.DataFrame,
    *,
    stage_dir: Path,
    n_boot: int,
    seed: int,
    patient_id_col: str | None,
) -> dict[str, Path]:
    """Emit pooled and per-model primary-estimand CSVs."""
    subset = _primary_estimand_subset(attempts_frame)
    if subset.empty:
        LOGGER.info("Primary-estimand subset is empty; skipping outputs")
        return {}

    out_dir = ensure_dir(stage_dir / "primary_estimand")
    outputs: dict[str, Path] = {}

    pooled = _metric_row(
        subset,
        n_boot=n_boot,
        seed=seed,
        patient_id_col=patient_id_col,
        extras={"scope": "pooled"},
    )
    if pooled is not None:
        outputs["primary_estimand_pooled"] = write_dataframe(
            pd.DataFrame([pooled]), out_dir / "primary_estimand_pooled.csv"
        )

    by_model_rows: list[dict] = []
    group_cols = [c for c in ("model_canon", "temperature") if c in subset.columns]
    if group_cols:
        for keys, sub in subset.groupby(group_cols, dropna=False):
            if not isinstance(keys, tuple):
                keys = (keys,)
            extras = {col: keys[i] for i, col in enumerate(group_cols)}
            if "temperature" in extras:
                try:
                    extras["temperature"] = float(extras["temperature"])
                except (TypeError, ValueError):
                    extras["temperature"] = None
            if "model_canon" in extras:
                extras["model_canon"] = str(extras["model_canon"])
            row = _metric_row(
                sub,
                n_boot=n_boot,
                seed=seed,
                patient_id_col=patient_id_col,
                extras=extras,
            )
            if row is not None:
                by_model_rows.append(row)
    if by_model_rows:
        outputs["primary_estimand_by_model"] = write_dataframe(
            pd.DataFrame(by_model_rows), out_dir / "primary_estimand_by_model.csv"
        )

    return outputs


def _attach_strict_document_truth(frame: pd.DataFrame) -> pd.Series | None:
    """Compute the strict document-sequence truth column.

    Returns the populated series, or ``None`` when no usable document
    marker columns are present in ``frame``.
    """
    if "report_sequence_name" not in frame.columns:
        return None
    strict = build_document_truth_series(frame, strict=True)
    if strict.notna().sum() == 0:
        return None
    return strict


def _write_strict_marker_sensitivity_outputs(
    frame: pd.DataFrame,
    *,
    stage_dir: Path,
    n_boot: int,
    seed: int,
    patient_id_col: str | None,
    skip_factors: set[str],
) -> dict[str, Path]:
    """Rerun the per-factor performance loop against the strict truth label."""
    strict_truth = (
        frame[STRICT_DOC_TRUTH_COLUMN]
        if STRICT_DOC_TRUTH_COLUMN in frame.columns
        else _attach_strict_document_truth(frame)
    )
    if strict_truth is None or strict_truth.notna().sum() == 0:
        LOGGER.info(
            "Strict document-sequence truth unavailable; skipping strict-marker "
            "sensitivity outputs"
        )
        return {}

    base = frame.copy()
    base[STRICT_DOC_TRUTH_COLUMN] = strict_truth
    base["_bucket"] = clean_likelihood(base["likelihood_score"])
    base["temperature"] = pd.to_numeric(base["temperature"], errors="coerce").round(2)

    attempts = base.dropna(subset=[STRICT_DOC_TRUTH_COLUMN]).copy()
    attempts["_y"] = attempts[STRICT_DOC_TRUTH_COLUMN].astype(int)
    attempts["_p"] = (
        likelihood_to_probability(attempts["_bucket"]).fillna(0.0).to_numpy()
    )

    complete = base.dropna(subset=[STRICT_DOC_TRUTH_COLUMN, "_bucket"]).copy()
    complete["_y"] = complete[STRICT_DOC_TRUTH_COLUMN].astype(int)
    complete["_p"] = likelihood_to_probability(complete["_bucket"]).to_numpy()

    outputs: dict[str, Path] = {}
    for label, working in (("all_attempts", attempts), ("complete_case", complete)):
        rows = _run_factor_loop(
            working,
            denominator=label,
            n_boot=n_boot,
            seed=seed,
            patient_id_col=patient_id_col,
            skip_factors=skip_factors,
        )
        if not rows:
            continue
        out_dir = ensure_dir(stage_dir / "sensitivity_complete_marker" / label)
        outputs[f"sensitivity_complete_marker_{label}"] = write_dataframe(
            pd.DataFrame(rows), out_dir / "overall_performance_by_factor_fair.csv"
        )
    return outputs


def run_full_performance(config: AnalysisConfig) -> dict[str, Path]:
    """Run the full-performance stage.

    Reads the merged per-row outputs from ``data_prep``, applies the
    FAIR filter for each factor, and emits the overall performance CSV
    (one row per factor / value combination with point estimates and
    bootstrap CI half-widths) plus one bar plot per metric in
    :data:`METRICS`.

    Parameters
    ----------
    config:
        The active :class:`AnalysisConfig`. Bootstrap iterations and
        random seed are read from ``config.analysis``.

    Returns
    -------
    dict[str, pathlib.Path]
        Mapping with ``stage_dir`` and, when results are produced, the
        path to ``overall_performance_by_factor_fair.csv``.
    """
    stage_dir = ensure_dir(config.paths["output_root"] / "full_performance")
    out_dir = ensure_dir(stage_dir / "overall_perf_plots_clean_eval_fair")

    merged = config.paths["output_root"] / "data_prep" / "merged_outputs.csv"
    if not merged.exists():
        merged = config.paths["output_root"] / "data_prep" / "parsed_outputs.csv"
    if not merged.exists():
        LOGGER.warning("No input frame at %s", merged)
        return {"stage_dir": stage_dir}
    frame = pd.read_csv(merged)
    frame, truth_col = prepare_truth_frame(
        frame, mode=truth_mode(config), candidates=TRUTH_CANDIDATES
    )
    if truth_col is None or "likelihood_score" not in frame.columns:
        LOGGER.warning("Missing truth or likelihood columns; skipping")
        _clear_stage_outputs(stage_dir)
        return {"stage_dir": stage_dir}

    frame[truth_col] = pd.to_numeric(frame[truth_col], errors="coerce")
    frame["_bucket"] = clean_likelihood(frame["likelihood_score"])
    # all-attempts frame: retain every row with a valid reference
    # label, including those with an unparseable / missing likelihood.
    # Failed likelihoods are treated as a negative prediction: a model
    # that fails to return a valid positive classification does not
    # identify the patient as IBD-positive.
    attempts_frame = frame.dropna(subset=[truth_col]).copy()
    attempts_frame["_y"] = attempts_frame[truth_col].astype(int)
    # ``_bucket`` is NaN for failed/unparseable likelihoods; the
    # negative-prediction policy means the implied probability for
    # those rows is 0, which the rescaled bucket-to-probability
    # mapping handles via NaN -> 0 fillna below.
    attempts_p = likelihood_to_probability(attempts_frame["_bucket"]).fillna(0.0)
    attempts_frame["_p"] = attempts_p.to_numpy()
    attempts_frame["temperature"] = pd.to_numeric(
        attempts_frame["temperature"], errors="coerce"
    ).round(2)

    # complete-case frame: drop rows where the likelihood was not
    # parseable; this is the discriminatory-performance view.
    frame = frame.dropna(subset=[truth_col, "_bucket"])
    frame["_y"] = frame[truth_col].astype(int)
    frame["_p"] = likelihood_to_probability(frame["_bucket"]).to_numpy()
    frame["temperature"] = pd.to_numeric(frame["temperature"], errors="coerce").round(2)

    n_boot = int(config.analysis.get("bootstrap_iterations", 1000))
    seed = int(config.analysis.get("random_seed", 42))
    enable_temperature = bool(config.analysis.get("enable_temperature_analysis", False))
    skip_factors = set() if enable_temperature else set(CONFOUNDED_FACTORS)

    patient_id_col = next(
        (
            col
            for col in ("patient_id", "study_id", "PatientID")
            if col in frame.columns
        ),
        None,
    )

    def _run_loop(working: pd.DataFrame, *, denominator: str) -> list[dict]:
        return _run_factor_loop(
            working,
            denominator=denominator,
            n_boot=n_boot,
            seed=seed,
            patient_id_col=patient_id_col,
            skip_factors=skip_factors,
        )

    # Primary view: "all-attempts" denominator -- retains every
    # attempted inference call with a valid reference label and
    # treats failed/unevaluable outputs as negative predictions.
    # This is the deployment-relevant view.
    rows = _run_loop(attempts_frame, denominator="all_attempts")

    # Reported side-by-side with the complete-case denominator, which
    # restricts to rows where a valid likelihood was parsed. This
    # estimates classification performance conditional on the model
    # returning an evaluable answer.
    rows.extend(_run_loop(frame, denominator="complete_case"))

    # Token-truncation sensitivity analysis: re-run the per-factor
    # loop on the subset of rows where no API or section truncation
    # was recorded. Two-shot calls consume more of the available
    # context window and are therefore more likely to be truncated.
    truncation_cols = [
        col
        for col in ("api_truncated", "section_truncated", "truncated")
        if col in frame.columns
    ]
    if truncation_cols:
        truncated_mask = frame[truncation_cols].fillna(False).astype(bool).any(axis=1)
        non_truncated = frame.loc[~truncated_mask]
        if not non_truncated.empty:
            rows.extend(_run_loop(non_truncated, denominator="non_truncated_only"))

    if not rows:
        return {"stage_dir": stage_dir}

    perf_df = pd.DataFrame(rows)
    csv_path = out_dir / "overall_performance_by_factor_fair.csv"
    write_dataframe(perf_df, csv_path)
    for metric in METRICS:
        # Plot the primary (all-attempts) view; complete-case and
        # truncation-sensitivity rows live in the same CSV for
        # transparency.
        _plot_metric(perf_df[perf_df["denominator"] == "all_attempts"], metric, out_dir)

    outputs: dict[str, Path] = {
        "stage_dir": stage_dir,
        "overall_performance_by_factor_fair": csv_path,
    }

    # Optional inferential sensitivity analyses (paired McNemar +
    # mixed-effects logistic regression). These are descriptive and
    # add no claims about temperature; failures are logged and do not
    # abort the stage.
    try:
        from analytics_code.inference import (
            mcnemar_pairwise_models,
            mixed_effects_logistic_correctness,
        )
    except Exception as exc:  # pragma: no cover - defensive import
        LOGGER.info("Inference helpers unavailable: %s", exc)
    else:
        try:
            mcnemar_df = mcnemar_pairwise_models(frame)
        except Exception as exc:
            LOGGER.warning("McNemar pairwise comparison failed: %s", exc)
        else:
            if not mcnemar_df.empty:
                outputs["mcnemar_model_pairs"] = write_dataframe(
                    mcnemar_df, stage_dir / "mcnemar_model_pairs.csv"
                )
        try:
            mixed_df = mixed_effects_logistic_correctness(frame, patient_id_col)
        except Exception as exc:
            LOGGER.warning("Mixed-effects logistic regression failed: %s", exc)
        else:
            if mixed_df is not None and not mixed_df.empty:
                outputs["mixed_effects_correctness"] = write_dataframe(
                    mixed_df, stage_dir / "mixed_effects_correctness.csv"
                )

    # Additive production outputs: primary-estimand pass + strict-marker
    # sensitivity pass. Both reuse the existing metric / bootstrap
    # machinery and are skipped cleanly when prerequisites are not
    # satisfied.
    try:
        outputs.update(
            _write_primary_estimand_outputs(
                attempts_frame,
                stage_dir=stage_dir,
                n_boot=n_boot,
                seed=seed,
                patient_id_col=patient_id_col,
            )
        )
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Primary-estimand outputs failed: %s", exc)

    try:
        # Read the raw frame again so document marker columns survive
        # the truth_mode-driven coercions above.
        raw_frame = pd.read_csv(merged)
        outputs.update(
            _write_strict_marker_sensitivity_outputs(
                raw_frame,
                stage_dir=stage_dir,
                n_boot=n_boot,
                seed=seed,
                patient_id_col=patient_id_col,
                skip_factors=skip_factors,
            )
        )
    except Exception as exc:  # pragma: no cover - defensive
        LOGGER.warning("Strict-marker sensitivity outputs failed: %s", exc)

    return outputs
