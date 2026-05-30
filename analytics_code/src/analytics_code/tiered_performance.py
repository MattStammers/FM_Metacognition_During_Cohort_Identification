"""Tiered performance stage (Document / Cumulative / Final / Doc2Patient).

Implements the published "Table 3.2.4-1" analysis:

* ``Document_*`` -- one row per single-doc data type
  (``hist``, ``endo``, ``clinic_preceding``, ``clinic_following``);
  prediction is the model row likelihood ``>= 5``, reference is the
  corresponding physician marker.
* ``Cumulative_*`` -- one row per multi-doc data type (the linked
  document bundle); prediction is the model row likelihood ``>= 5``,
  reference is the logical OR of the physician markers for the docs
  shown in that data_type (see :data:`DATA_TYPE_DOCUMENT_SET`).
* ``Final_*`` -- one row per ``(study_id, model_canon, shot_type,
  temperature)``; prediction is the logical OR of every per-row
  prediction for that group, reference is the logical OR over all
  four physician markers for that patient
  (:data:`PATIENT_DOCUMENT_TRUTH`).
* ``Doc2Patient`` -- same unit and prediction as ``Final_*`` but
  reference is the **chart-verified** patient label
  (``Patient_Has_IBD`` / ``ground_truth``). Secondary endpoint.

Outputs land under::

    output_root/
      Document/{all_attempts,complete_case}/overall.csv, per_data_type.csv
      Cumulative/{all_attempts,complete_case}/overall.csv, per_data_type.csv
      Final/{all_attempts,complete_case}/overall.csv, per_model.csv
      Doc2Patient/{all_attempts,complete_case}/overall.csv, per_model.csv

Each ``overall.csv`` / ``per_*.csv`` is accompanied by per-metric bar
plots using the existing publication helpers.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analytics_code.common import (
    DATA_TYPE_DOCUMENT_SET,
    DOC_FLAG_COLUMNS,
    SINGLE_DOC_DATA_TYPES,
    SINGLE_DOC_TYPE_MAP,
    ensure_dir,
    format_model_display_name,
    nice_numeric_ticks,
    remove_tree,
    save_figure,
    sentence_case,
    set_publication_axes,
    tier_of,
    write_dataframe,
)
from analytics_code.config import AnalysisConfig
from analytics_code.data_prep import PATIENT_DOCUMENT_TRUTH
from analytics_code.full_performance import METRICS, _bootstrap_metrics, _point_metrics
from analytics_code.predictions import (
    DECISION_THRESHOLD,
    clean_likelihood,
    likelihood_to_probability,
)
from analytics_code.truth_labels import _coerce_binary_series

LOGGER = logging.getLogger("analytics_code.tiered_performance")

#: Reference candidates for the chart-verified (Doc2Patient) label.
CHART_TRUTH_CANDIDATES: tuple[str, ...] = (
    "Patient_Has_IBD",
    "ground_truth",
    "patient_has_ibd",
    "label",
    "gold_label",
)

#: Missing-policy labels emitted in every CSV.
ALL_ATTEMPTS = "all_attempts"
COMPLETE_CASE = "complete_case"


def _detect_chart_truth(frame: pd.DataFrame) -> str | None:
    for candidate in CHART_TRUTH_CANDIDATES:
        if candidate in frame.columns:
            return candidate
    return None


def _patient_id_col(frame: pd.DataFrame) -> str | None:
    for col in ("patient_id", "study_id", "PatientID"):
        if col in frame.columns:
            return col
    return None


# ---------------------------------------------------------------------------
# Row-level prediction helpers (shared by every tier).
# ---------------------------------------------------------------------------


def _row_bucket(frame: pd.DataFrame) -> pd.Series:
    """Cleaned ``0..10`` likelihood bucket for every row (NaN if unparseable)."""
    return clean_likelihood(frame["likelihood_score"])


def _row_prediction(bucket: pd.Series, *, policy: str) -> pd.Series:
    """Binary 0/1 prediction with NaN handling per ``policy``.

    ``all_attempts``: unparseable likelihoods are treated as 0.
    ``complete_case``: unparseable likelihoods stay NaN (caller drops).
    """
    if policy == ALL_ATTEMPTS:
        return (bucket.fillna(-1) >= DECISION_THRESHOLD).astype(int)
    return bucket.where(bucket.notna()).map(
        lambda v: int(v >= DECISION_THRESHOLD) if pd.notna(v) else pd.NA
    )


def _row_probability(bucket: pd.Series, *, policy: str) -> pd.Series:
    """Probability score in ``[0, 1]`` with NaN handling per ``policy``."""
    prob = likelihood_to_probability(bucket)
    if policy == ALL_ATTEMPTS:
        return prob.fillna(0.0)
    return prob


# ---------------------------------------------------------------------------
# Per-row truth builders.
# ---------------------------------------------------------------------------


def _document_row_truth(
    frame: pd.DataFrame, *, flag_columns: dict[str, str]
) -> pd.Series:
    """Single-doc tier: the physician marker matching the row's data_type."""
    truth = pd.Series(np.nan, index=frame.index, dtype="float64")
    if "report_sequence_name" not in frame.columns:
        return truth
    seq = frame["report_sequence_name"].astype(str)
    for data_type, doc_type in SINGLE_DOC_TYPE_MAP.items():
        column = flag_columns.get(doc_type)
        if column is None or column not in frame.columns:
            continue
        mask = seq == data_type
        truth.loc[mask] = _coerce_binary_series(frame.loc[mask, column]).astype(
            "float64"
        )
    return truth


def _cumulative_row_truth(
    frame: pd.DataFrame, *, flag_columns: dict[str, str]
) -> pd.Series:
    """Multi-doc tier: OR over physician markers for the docs shown in the row."""
    truth = pd.Series(np.nan, index=frame.index, dtype="float64")
    if "report_sequence_name" not in frame.columns:
        return truth
    seq = frame["report_sequence_name"].astype(str)
    binary = {
        doc_type: _coerce_binary_series(frame[column])
        for doc_type, column in flag_columns.items()
        if column in frame.columns
    }
    for data_type, doc_types in DATA_TYPE_DOCUMENT_SET.items():
        if data_type in SINGLE_DOC_DATA_TYPES:
            continue
        mask = seq == data_type
        if not mask.any():
            continue
        available = [binary[d] for d in doc_types if d in binary]
        if not available:
            continue
        values = pd.concat(available, axis=1).loc[mask]
        any_true = values.eq(1).any(axis=1)
        any_known = values.notna().any(axis=1)
        truth.loc[mask] = np.where(any_true, 1.0, np.where(any_known, 0.0, np.nan))
    return truth


# ---------------------------------------------------------------------------
# Tier scaffolding.
# ---------------------------------------------------------------------------


GROUP_KEYS = ("model_canon", "shot_type", "temperature")


def _resolve_flag_columns(config: AnalysisConfig) -> dict[str, str]:
    """Return the doc-type -> reference-column map (config overrides default)."""
    override = config.analysis.get("document_flag_columns") if config else None
    if isinstance(override, dict) and override:
        return {str(k): str(v) for k, v in override.items()}
    return dict(DOC_FLAG_COLUMNS)


def _prepare_base(frame: pd.DataFrame, *, flag_columns: dict[str, str]) -> pd.DataFrame:
    """Attach ``_bucket``, ``_tier``, document/cumulative truth columns."""
    out = frame.copy()
    if "temperature" in out.columns:
        out["temperature"] = pd.to_numeric(out["temperature"], errors="coerce").round(2)
    out["_bucket"] = _row_bucket(out)
    out["_tier"] = (
        out["report_sequence_name"].map(tier_of)
        if "report_sequence_name" in out.columns
        else "unknown"
    )
    out["_doc_truth"] = _document_row_truth(out, flag_columns=flag_columns)
    out["_cum_truth"] = _cumulative_row_truth(out, flag_columns=flag_columns)
    return out


def _metric_block(
    y: np.ndarray,
    p: np.ndarray,
    *,
    patient_ids: np.ndarray | None,
    n_boot: int,
    seed: int,
) -> dict:
    """Return a flat metric dict (point + lo/hi CI half-widths) for one slice."""
    if len(y) == 0:
        return {}
    point = _point_metrics(y, p)
    cis = _bootstrap_metrics(y, p, n_boot=n_boot, seed=seed, patient_ids=patient_ids)
    row: dict = {"n": int(len(y))}
    for metric in METRICS:
        row[metric] = point[metric]
        row[f"{metric}_lo"] = cis[metric][0]
        row[f"{metric}_hi"] = cis[metric][1]
    return row


# ---------------------------------------------------------------------------
# Document tier.
# ---------------------------------------------------------------------------


def _document_tier(
    base: pd.DataFrame,
    *,
    policy: str,
    n_boot: int,
    seed: int,
    patient_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Build the Document overall + per-data-type tables for one policy."""
    rows = base[base["_tier"] == "document"].copy()
    rows["_pred"] = _row_prediction(rows["_bucket"], policy=policy)
    rows["_p"] = _row_probability(rows["_bucket"], policy=policy)
    rows = rows.dropna(subset=["_doc_truth"])
    if policy == COMPLETE_CASE:
        rows = rows.dropna(subset=["_pred"])
    if rows.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows["_y"] = rows["_doc_truth"].astype(int)
    rows["_pred"] = rows["_pred"].astype(int)
    rows["_p"] = rows["_p"].astype(float)

    overall = _per_group_table(
        rows, n_boot=n_boot, seed=seed, patient_col=patient_col, denominator=policy
    )
    per_dt = _per_group_table(
        rows,
        n_boot=n_boot,
        seed=seed,
        patient_col=patient_col,
        denominator=policy,
        extra_group=("report_sequence_name",),
    )
    return overall, per_dt


# ---------------------------------------------------------------------------
# Cumulative tier.
# ---------------------------------------------------------------------------


def _cumulative_tier(
    base: pd.DataFrame,
    *,
    policy: str,
    n_boot: int,
    seed: int,
    patient_col: str | None,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    rows = base[base["_tier"] == "cumulative"].copy()
    rows["_pred"] = _row_prediction(rows["_bucket"], policy=policy)
    rows["_p"] = _row_probability(rows["_bucket"], policy=policy)
    rows = rows.dropna(subset=["_cum_truth"])
    if policy == COMPLETE_CASE:
        rows = rows.dropna(subset=["_pred"])
    if rows.empty:
        return pd.DataFrame(), pd.DataFrame()
    rows["_y"] = rows["_cum_truth"].astype(int)
    rows["_pred"] = rows["_pred"].astype(int)
    rows["_p"] = rows["_p"].astype(float)

    overall = _per_group_table(
        rows, n_boot=n_boot, seed=seed, patient_col=patient_col, denominator=policy
    )
    per_dt = _per_group_table(
        rows,
        n_boot=n_boot,
        seed=seed,
        patient_col=patient_col,
        denominator=policy,
        extra_group=("report_sequence_name",),
    )
    return overall, per_dt


# ---------------------------------------------------------------------------
# Final / Doc2Patient (patient-aggregated) tiers.
# ---------------------------------------------------------------------------


def _aggregate_patient_predictions(
    base: pd.DataFrame, *, policy: str, patient_col: str
) -> pd.DataFrame:
    """Collapse rows to one prediction per ``(patient, model, shot, temp)``.

    For every group, the Final prediction is the logical OR of the
    per-row binary predictions. Under ``all_attempts`` a missing
    likelihood counts as a 0 vote (and still participates in the OR).
    Under ``complete_case`` rows with missing likelihoods are dropped
    before the OR; if every row for a patient is dropped, that patient
    is omitted from the group.
    """
    rows = base.copy()
    rows["_pred"] = _row_prediction(rows["_bucket"], policy=policy)
    if policy == COMPLETE_CASE:
        rows = rows.dropna(subset=["_pred"])
    if rows.empty:
        return pd.DataFrame()
    rows["_pred"] = rows["_pred"].astype(int)
    group_cols = [patient_col] + [c for c in GROUP_KEYS if c in rows.columns]
    agg = rows.groupby(group_cols, dropna=False)["_pred"].max().reset_index()
    agg = agg.rename(columns={"_pred": "_final_pred"})
    agg["_p"] = agg["_final_pred"].astype(float)
    return agg


def _final_tier(
    base: pd.DataFrame,
    *,
    policy: str,
    n_boot: int,
    seed: int,
    patient_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    if PATIENT_DOCUMENT_TRUTH not in base.columns:
        LOGGER.warning(
            "Final tier requested but %s missing from frame; skipping",
            PATIENT_DOCUMENT_TRUTH,
        )
        return pd.DataFrame(), pd.DataFrame()
    truth_map = (
        base.dropna(subset=[PATIENT_DOCUMENT_TRUTH])
        .groupby(patient_col)[PATIENT_DOCUMENT_TRUTH]
        .first()
    )
    if truth_map.empty:
        return pd.DataFrame(), pd.DataFrame()

    agg = _aggregate_patient_predictions(base, policy=policy, patient_col=patient_col)
    if agg.empty:
        return pd.DataFrame(), pd.DataFrame()
    agg["_y"] = agg[patient_col].map(truth_map)
    agg = agg.dropna(subset=["_y"])
    if agg.empty:
        return pd.DataFrame(), pd.DataFrame()
    agg["_y"] = agg["_y"].astype(int)
    agg["_pred"] = agg["_final_pred"].astype(int)
    agg["_p"] = agg["_p"].astype(float)

    overall = _per_group_table(
        agg, n_boot=n_boot, seed=seed, patient_col=None, denominator=policy
    )
    per_model = _per_group_table(
        agg,
        n_boot=n_boot,
        seed=seed,
        patient_col=None,
        denominator=policy,
        extra_group=(),
        # Final per-model table is already keyed by model_canon via the
        # per-group loop below.
        per_model_only=True,
    )
    return overall, per_model


def _doc2patient_tier(
    base: pd.DataFrame,
    *,
    policy: str,
    n_boot: int,
    seed: int,
    patient_col: str,
    chart_truth_col: str,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    truth_map = (
        base.dropna(subset=[chart_truth_col])
        .groupby(patient_col)[chart_truth_col]
        .first()
    )
    truth_map = pd.to_numeric(truth_map, errors="coerce").dropna()
    if truth_map.empty:
        return pd.DataFrame(), pd.DataFrame()

    agg = _aggregate_patient_predictions(base, policy=policy, patient_col=patient_col)
    if agg.empty:
        return pd.DataFrame(), pd.DataFrame()
    agg["_y"] = agg[patient_col].map(truth_map)
    agg = agg.dropna(subset=["_y"])
    if agg.empty:
        return pd.DataFrame(), pd.DataFrame()
    agg["_y"] = agg["_y"].astype(int)
    agg["_pred"] = agg["_final_pred"].astype(int)
    agg["_p"] = agg["_p"].astype(float)

    overall = _per_group_table(
        agg, n_boot=n_boot, seed=seed, patient_col=None, denominator=policy
    )
    per_model = _per_group_table(
        agg,
        n_boot=n_boot,
        seed=seed,
        patient_col=None,
        denominator=policy,
        extra_group=(),
        per_model_only=True,
    )
    return overall, per_model


# ---------------------------------------------------------------------------
# Per-group metric table (shared across tiers).
# ---------------------------------------------------------------------------


def _per_group_table(
    rows: pd.DataFrame,
    *,
    n_boot: int,
    seed: int,
    patient_col: str | None,
    denominator: str,
    extra_group: tuple[str, ...] = (),
    per_model_only: bool = False,
) -> pd.DataFrame:
    """Per-(model, shot, temp [, extra]) metric loop.

    When ``per_model_only`` is ``True`` the group is reduced to
    ``model_canon`` only (used for Final per-model tables that pool
    across shot / temperature so the patient is only counted once per
    model).
    """
    if rows.empty:
        return pd.DataFrame()
    base_keys = (
        ["model_canon"]
        if per_model_only
        else [c for c in GROUP_KEYS if c in rows.columns]
    )
    keys = base_keys + [c for c in extra_group if c in rows.columns]
    out_rows: list[dict] = []
    grouper = rows.groupby(keys, dropna=False) if keys else [((None,), rows)]
    for key_values, sub in grouper:
        if not isinstance(key_values, tuple):
            key_values = (key_values,)
        y = sub["_y"].to_numpy()
        p = sub["_p"].to_numpy()
        patient_ids = (
            sub[patient_col].to_numpy()
            if patient_col and patient_col in sub.columns
            else None
        )
        metrics = _metric_block(y, p, patient_ids=patient_ids, n_boot=n_boot, seed=seed)
        if not metrics:
            continue
        row: dict = {"denominator": denominator}
        for col_name, value in zip(keys, key_values):
            row[col_name] = value
        if "model_canon" in row:
            row["model_display"] = format_model_display_name(row["model_canon"])
        row.update(metrics)
        out_rows.append(row)
    return pd.DataFrame(out_rows)


# ---------------------------------------------------------------------------
# Plotting.
# ---------------------------------------------------------------------------


def _plot_tier_metric(
    df: pd.DataFrame,
    *,
    metric: str,
    label_cols: list[str],
    out_path: Path,
    title: str,
) -> None:
    """Render a horizontal bar plot for one metric across one tier table."""
    if df.empty or metric not in df.columns:
        return
    ascending = metric == "Brier_Score"
    plot_df = df.sort_values(metric, ascending=ascending).reset_index(drop=True)
    labels = [
        " / ".join(str(row[c]) for c in label_cols if c in plot_df.columns)
        for _, row in plot_df.iterrows()
    ]
    means = plot_df[metric].to_numpy()
    lo = plot_df.get(f"{metric}_lo", pd.Series(0, index=plot_df.index)).to_numpy()
    hi = plot_df.get(f"{metric}_hi", pd.Series(0, index=plot_df.index)).to_numpy()
    positions = np.arange(len(plot_df))
    fig, ax = plt.subplots(figsize=(11.6, max(4.0, 0.36 * len(positions) + 1.8)))
    ax.barh(
        positions, means, color="#1E88E5", linewidth=1.0, edgecolor="black", zorder=3
    )
    for pos, m, lov, hiv in zip(positions, means, lo, hi):
        if not np.isfinite(m) or (lov == 0.0 and hiv == 0.0):
            continue
        ax.errorbar(
            x=m,
            y=pos,
            xerr=np.array([[max(lov, 0.0)], [max(hiv, 0.0)]]),
            fmt="none",
            ecolor="black",
            elinewidth=1.2,
            capsize=3.0,
            capthick=1.2,
            zorder=4,
        )
    ax.set_yticks(positions)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(sentence_case(metric))
    ax.set_title(title)
    ax.set_xlim(0.0, 1.02)
    ax.set_xticks(nice_numeric_ticks(0.0, 1.0, 0.1))
    set_publication_axes(ax, show_grid_y=False, show_grid_x=True)
    fig.tight_layout()
    save_figure(out_path)


def _plot_tier_outputs(
    df: pd.DataFrame,
    *,
    out_dir: Path,
    tier_label: str,
    table_label: str,
    label_cols: list[str],
) -> None:
    if df.empty:
        return
    plots_dir = ensure_dir(out_dir / f"{table_label}_plots")
    for metric in METRICS:
        _plot_tier_metric(
            df,
            metric=metric,
            label_cols=label_cols,
            out_path=plots_dir / f"{metric.lower()}.png",
            title=f"{tier_label} ({table_label}) - {sentence_case(metric)}",
        )


# ---------------------------------------------------------------------------
# Top-level entry point.
# ---------------------------------------------------------------------------


def run_tiered_performance(config: AnalysisConfig) -> dict[str, Path]:
    """Run the Document / Cumulative / Final / Doc2Patient tier stage.

    Reads the merged outputs produced by ``data_prep`` and writes one
    folder per tier into ``output_root``. Each tier folder contains
    one sub-folder per missing policy (``all_attempts`` /
    ``complete_case``), and each sub-folder contains an ``overall.csv``
    plus a per-data-type or per-model breakdown table with bar plots.
    """
    output_root = config.paths["output_root"]
    stage_root = ensure_dir(output_root / "tiered_performance")
    tier_dirs = {
        "Document": ensure_dir(stage_root / "Document"),
        "Cumulative": ensure_dir(stage_root / "Cumulative"),
        "Final": ensure_dir(stage_root / "Final"),
        "Doc2Patient": ensure_dir(stage_root / "Doc2Patient"),
    }
    # Wipe stale outputs so reruns reflect the current frame only.
    for d in tier_dirs.values():
        remove_tree(d)
        ensure_dir(d)

    merged_path = output_root / "data_prep" / "merged_outputs.csv"
    if not merged_path.exists():
        LOGGER.warning("Tiered stage skipped: %s not found", merged_path)
        return {"stage_root": stage_root}
    frame = pd.read_csv(merged_path)
    if frame.empty or "likelihood_score" not in frame.columns:
        LOGGER.warning("Tiered stage skipped: empty frame or missing likelihood column")
        return {"stage_root": stage_root}

    patient_col = _patient_id_col(frame)
    if patient_col is None:
        LOGGER.warning("Tiered stage skipped: no patient identifier column present")
        return {"stage_root": stage_root}

    flag_columns = _resolve_flag_columns(config)
    base = _prepare_base(frame, flag_columns=flag_columns)

    n_boot = int(config.analysis.get("bootstrap_iterations", 1000))
    seed = int(config.analysis.get("random_seed", 42))

    outputs: dict[str, Path] = {"stage_root": stage_root}
    for policy in (ALL_ATTEMPTS, COMPLETE_CASE):
        outputs.update(
            _emit_document(
                base,
                tier_dir=tier_dirs["Document"],
                policy=policy,
                n_boot=n_boot,
                seed=seed,
                patient_col=patient_col,
            )
        )
        outputs.update(
            _emit_cumulative(
                base,
                tier_dir=tier_dirs["Cumulative"],
                policy=policy,
                n_boot=n_boot,
                seed=seed,
                patient_col=patient_col,
            )
        )
        outputs.update(
            _emit_final(
                base,
                tier_dir=tier_dirs["Final"],
                policy=policy,
                n_boot=n_boot,
                seed=seed,
                patient_col=patient_col,
            )
        )
        chart_truth_col = _detect_chart_truth(base)
        if chart_truth_col is not None:
            outputs.update(
                _emit_doc2patient(
                    base,
                    tier_dir=tier_dirs["Doc2Patient"],
                    policy=policy,
                    n_boot=n_boot,
                    seed=seed,
                    patient_col=patient_col,
                    chart_truth_col=chart_truth_col,
                )
            )
        else:
            LOGGER.info(
                "Doc2Patient tier skipped (%s policy): no chart-verified truth column",
                policy,
            )
    return outputs


def _emit_document(
    base: pd.DataFrame,
    *,
    tier_dir: Path,
    policy: str,
    n_boot: int,
    seed: int,
    patient_col: str,
) -> dict[str, Path]:
    overall, per_dt = _document_tier(
        base, policy=policy, n_boot=n_boot, seed=seed, patient_col=patient_col
    )
    out_dir = ensure_dir(tier_dir / policy)
    written: dict[str, Path] = {}
    if not overall.empty:
        written[f"Document_{policy}_overall"] = write_dataframe(
            overall, out_dir / "overall.csv"
        )
        _plot_tier_outputs(
            overall,
            out_dir=out_dir,
            tier_label="Document",
            table_label="overall",
            label_cols=["model_display", "shot_type", "temperature"],
        )
    if not per_dt.empty:
        written[f"Document_{policy}_per_data_type"] = write_dataframe(
            per_dt, out_dir / "per_data_type.csv"
        )
        _plot_tier_outputs(
            per_dt,
            out_dir=out_dir,
            tier_label="Document",
            table_label="per_data_type",
            label_cols=["model_display", "report_sequence_name"],
        )
    return written


def _emit_cumulative(
    base: pd.DataFrame,
    *,
    tier_dir: Path,
    policy: str,
    n_boot: int,
    seed: int,
    patient_col: str,
) -> dict[str, Path]:
    overall, per_dt = _cumulative_tier(
        base, policy=policy, n_boot=n_boot, seed=seed, patient_col=patient_col
    )
    out_dir = ensure_dir(tier_dir / policy)
    written: dict[str, Path] = {}
    if not overall.empty:
        written[f"Cumulative_{policy}_overall"] = write_dataframe(
            overall, out_dir / "overall.csv"
        )
        _plot_tier_outputs(
            overall,
            out_dir=out_dir,
            tier_label="Cumulative",
            table_label="overall",
            label_cols=["model_display", "shot_type", "temperature"],
        )
    if not per_dt.empty:
        written[f"Cumulative_{policy}_per_data_type"] = write_dataframe(
            per_dt, out_dir / "per_data_type.csv"
        )
        _plot_tier_outputs(
            per_dt,
            out_dir=out_dir,
            tier_label="Cumulative",
            table_label="per_data_type",
            label_cols=["model_display", "report_sequence_name"],
        )
    return written


def _emit_final(
    base: pd.DataFrame,
    *,
    tier_dir: Path,
    policy: str,
    n_boot: int,
    seed: int,
    patient_col: str,
) -> dict[str, Path]:
    overall, per_model = _final_tier(
        base, policy=policy, n_boot=n_boot, seed=seed, patient_col=patient_col
    )
    out_dir = ensure_dir(tier_dir / policy)
    written: dict[str, Path] = {}
    if not overall.empty:
        written[f"Final_{policy}_overall"] = write_dataframe(
            overall, out_dir / "overall.csv"
        )
        _plot_tier_outputs(
            overall,
            out_dir=out_dir,
            tier_label="Final",
            table_label="overall",
            label_cols=["model_display", "shot_type", "temperature"],
        )
    if not per_model.empty:
        written[f"Final_{policy}_per_model"] = write_dataframe(
            per_model, out_dir / "per_model.csv"
        )
        _plot_tier_outputs(
            per_model,
            out_dir=out_dir,
            tier_label="Final",
            table_label="per_model",
            label_cols=["model_display"],
        )
    return written


def _emit_doc2patient(
    base: pd.DataFrame,
    *,
    tier_dir: Path,
    policy: str,
    n_boot: int,
    seed: int,
    patient_col: str,
    chart_truth_col: str,
) -> dict[str, Path]:
    overall, per_model = _doc2patient_tier(
        base,
        policy=policy,
        n_boot=n_boot,
        seed=seed,
        patient_col=patient_col,
        chart_truth_col=chart_truth_col,
    )
    out_dir = ensure_dir(tier_dir / policy)
    written: dict[str, Path] = {}
    if not overall.empty:
        written[f"Doc2Patient_{policy}_overall"] = write_dataframe(
            overall, out_dir / "overall.csv"
        )
        _plot_tier_outputs(
            overall,
            out_dir=out_dir,
            tier_label="Doc2Patient",
            table_label="overall",
            label_cols=["model_display", "shot_type", "temperature"],
        )
    if not per_model.empty:
        written[f"Doc2Patient_{policy}_per_model"] = write_dataframe(
            per_model, out_dir / "per_model.csv"
        )
        _plot_tier_outputs(
            per_model,
            out_dir=out_dir,
            tier_label="Doc2Patient",
            table_label="per_model",
            label_cols=["model_display"],
        )
    return written
