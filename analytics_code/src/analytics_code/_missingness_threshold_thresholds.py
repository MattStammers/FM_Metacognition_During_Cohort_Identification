"""Likelihood-based threshold sweeps for the missingness-threshold stage.

All decision thresholds emitted by this module are univariable cut-offs
on the ``0..10`` *likelihood of IBD* scale. The Bronze (>=5),
Silver (>=6) and Gold (>=7) tiers are simple likelihood thresholds,
not a multivariable rule combining likelihood with certainty and
case complexity. Certainty and complexity remain available in
:mod:`analytics_code._missingness_threshold_descriptive` for
secondary analyses but they do not enter the decision threshold
implemented here.
"""

from __future__ import annotations

import logging
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score, matthews_corrcoef, precision_score, recall_score

from analytics_code._missingness_threshold_shared import (
    THRESHOLD_TIERS,
    _model_pool_label,
    _model_pool_line_style,
    _model_pool_sort_key,
    _save_multi,
    _threshold_metrics,
)
from analytics_code.common import (
    THINKING_POOL_COLORS,
    ensure_dir,
    format_factor_value,
    format_model_display_name,
    model_color,
    nice_numeric_ticks,
    sentence_case,
    set_publication_axes,
    write_dataframe,
)

LOGGER = logging.getLogger("analytics_code.missingness_threshold")


def _bucket_axis_limits(
    values: pd.Series | np.ndarray,
) -> tuple[tuple[float, float], list[int]]:
    """Return a tight integer x-axis range around observed likelihood buckets."""
    numeric = pd.to_numeric(pd.Series(values), errors="coerce").dropna()
    if numeric.empty:
        return (-0.1, 10.1), list(range(0, 11))
    lower = max(0, int(np.floor(numeric.min())) - 1)
    upper = min(10, int(np.ceil(numeric.max())) + 1)
    return (lower - 0.1, upper + 0.1), list(range(lower, upper + 1))


def _threshold_sweep(y: np.ndarray, bucket: np.ndarray) -> pd.DataFrame:
    """Sweep integer thresholds 0..10 and return per-threshold metrics."""
    rows = []
    for threshold in range(0, 11):
        pred = (bucket >= threshold).astype(int)
        rows.append(
            {
                "threshold": threshold,
                "macro_f1": f1_score(y, pred, average="macro", zero_division=0),
                "precision": precision_score(y, pred, zero_division=0),
                "recall": recall_score(y, pred, zero_division=0),
                "mcc": matthews_corrcoef(y, pred)
                if pred.std() > 0 and y.std() > 0
                else 0.0,
                "positives_predicted": int((pred == 1).sum()),
                "n_eval": int(len(y)),
            }
        )
    return pd.DataFrame(rows)


def _macro_f1_calibration(frame: pd.DataFrame, truth_col: str, stage_dir: Path) -> None:
    """Per-model macro-F1 threshold sweep with paired CSV/PNG outputs."""
    out_dir = ensure_dir(stage_dir / "macro_f1_calibration")
    if "Likelihood_of_IBD_1" not in frame.columns:
        return

    work = frame[[truth_col, "Likelihood_of_IBD_1", "model_canon"]].copy()
    work[truth_col] = pd.to_numeric(work[truth_col], errors="coerce")
    work["bucket"] = pd.to_numeric(work["Likelihood_of_IBD_1"], errors="coerce")
    work = work.dropna(subset=[truth_col, "bucket"])
    if work.empty:
        LOGGER.info("Macro-F1 calibration skipped (no valid rows)")
        return
    x_limits, x_ticks = _bucket_axis_limits(work["bucket"])

    all_curves: list[pd.DataFrame] = []
    best_rows: list[dict[str, object]] = []
    for model, sub in work.groupby("model_canon", dropna=False):
        y = sub[truth_col].astype(int).to_numpy()
        bucket = sub["bucket"].astype(int).to_numpy()
        if len(y) == 0:
            continue
        curve = _threshold_sweep(y, bucket)
        curve["model"] = model
        curve["model_display"] = format_model_display_name(model)
        all_curves.append(curve)
        best = curve.sort_values(
            ["macro_f1", "mcc", "precision"], ascending=False
        ).iloc[0]
        best_rows.append(
            {
                "model": model,
                "model_display": format_model_display_name(model),
                "best_threshold": int(best["threshold"]),
                "best_macro_f1": float(best["macro_f1"]),
                "mcc_at_best": float(best["mcc"]),
                "precision_at_best": float(best["precision"]),
                "recall_at_best": float(best["recall"]),
                "n_eval": int(best["n_eval"]),
            }
        )

    curves_df = (
        pd.concat(all_curves, ignore_index=True) if all_curves else pd.DataFrame()
    )
    best_df = pd.DataFrame(best_rows).sort_values("best_macro_f1", ascending=False)
    if not curves_df.empty:
        write_dataframe(curves_df, out_dir / "macro_f1_threshold_curves_by_model.csv")
    if not best_df.empty:
        write_dataframe(best_df, out_dir / "best_thresholds_by_model_macroF1.csv")
    if curves_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for model, grp in curves_df.groupby("model"):
        grp_sorted = grp.sort_values("threshold")
        ax.plot(
            grp_sorted["threshold"],
            grp_sorted["macro_f1"],
            marker="o",
            linewidth=2.0,
            markersize=4,
            color=model_color(model),
            label=format_model_display_name(model),
        )
    ax.set_xlabel("Likelihood bucket threshold")
    ax.set_ylabel("Macro F1")
    ax.set_title("Macro F1 by threshold and model")
    ax.set_xlim(*x_limits)
    ax.set_xticks(x_ticks)
    ax.set_ylim(0.3, 0.8)
    ax.set_yticks(nice_numeric_ticks(0.3, 0.8, 0.1))
    set_publication_axes(ax, show_grid_y=True, show_grid_x=False)
    ax.legend(fontsize=8, frameon=False, ncol=2)
    _save_multi(fig, out_dir / "macro_f1_threshold_curves_by_model")

    fig2, ax2 = plt.subplots(figsize=(7.0, 4.6))
    for model, grp in curves_df.groupby("model"):
        grp_sorted = grp.sort_values("threshold")
        ax2.plot(
            grp_sorted["threshold"],
            grp_sorted["macro_f1"],
            marker="o",
            linewidth=2.1,
            markersize=3.5,
            markerfacecolor="white",
            markeredgewidth=0.8,
            color=model_color(model),
            label=format_model_display_name(model),
        )
    ax2.set_xlim(*x_limits)
    ax2.set_xticks(x_ticks)
    ax2.set_xlabel("Likelihood bucket threshold")
    ax2.set_ylabel("Macro F1")
    ax2.set_title("Macro F1 by threshold and model")
    ax2.set_ylim(0.3, 0.8)
    ax2.set_yticks(nice_numeric_ticks(0.3, 0.8, 0.1))
    set_publication_axes(ax2, show_grid_y=True, show_grid_x=False)
    ax2.legend(title="Model", frameon=False, fontsize=8)
    _save_multi(fig2, out_dir / "macro_f1_threshold_curves_by_model_pub")

    if "temperature" not in frame.columns:
        return

    work_temp = frame[
        [truth_col, "Likelihood_of_IBD_1", "model_canon", "temperature"]
    ].copy()
    work_temp[truth_col] = pd.to_numeric(work_temp[truth_col], errors="coerce")
    work_temp["bucket"] = pd.to_numeric(
        work_temp["Likelihood_of_IBD_1"], errors="coerce"
    )
    work_temp["temperature"] = pd.to_numeric(
        work_temp["temperature"], errors="coerce"
    ).round(2)
    work_temp = work_temp.dropna(subset=[truth_col, "bucket", "temperature"])
    if work_temp.empty:
        return
    x_temp_limits, x_temp_ticks = _bucket_axis_limits(work_temp["bucket"])

    curves_by_temp: list[pd.DataFrame] = []
    best_by_temp: list[dict[str, object]] = []
    for (model, temperature), sub in work_temp.groupby(
        ["model_canon", "temperature"], dropna=False
    ):
        y = sub[truth_col].astype(int).to_numpy()
        bucket = sub["bucket"].astype(int).to_numpy()
        if len(y) == 0:
            continue
        curve = _threshold_sweep(y, bucket)
        curve["model"] = model
        curve["temperature"] = float(temperature)
        curve[
            "model_temperature"
        ] = f"{format_model_display_name(model)} @ {format_factor_value('temperature', float(temperature))}"
        curves_by_temp.append(curve)
        best = curve.sort_values(
            ["macro_f1", "mcc", "precision"], ascending=False
        ).iloc[0]
        best_by_temp.append(
            {
                "model": model,
                "model_display": format_model_display_name(model),
                "temperature": float(temperature),
                "best_threshold": int(best["threshold"]),
                "best_macro_f1": float(best["macro_f1"]),
                "mcc_at_best": float(best["mcc"]),
                "precision_at_best": float(best["precision"]),
                "recall_at_best": float(best["recall"]),
                "n_eval": int(best["n_eval"]),
            }
        )

    curves_temp_df = (
        pd.concat(curves_by_temp, ignore_index=True)
        if curves_by_temp
        else pd.DataFrame()
    )
    best_temp_df = pd.DataFrame(best_by_temp)
    if not curves_temp_df.empty:
        write_dataframe(
            curves_temp_df,
            out_dir / "macro_f1_threshold_curves_by_model_temperature.csv",
        )
    if not best_temp_df.empty:
        write_dataframe(
            best_temp_df.sort_values(
                ["best_macro_f1", "temperature"], ascending=[False, True]
            ),
            out_dir / "best_thresholds_by_model_temperature_macroF1.csv",
        )
    if curves_temp_df.empty:
        return

    fig3, ax3 = plt.subplots(figsize=(7.2, 4.8))
    for (model, temperature), grp in curves_temp_df.groupby(["model", "temperature"]):
        grp_sorted = grp.sort_values("threshold")
        ax3.plot(
            grp_sorted["threshold"],
            grp_sorted["macro_f1"],
            marker="o",
            linewidth=1.7,
            markersize=3.5,
            color=model_color(model),
            alpha=0.55 if float(temperature) not in {0.5, 1.0} else 0.9,
            linestyle="-" if float(temperature) in {0.5, 0.75} else "--",
            label=f"{format_model_display_name(model)} @ {format_factor_value('temperature', temperature)}",
        )
    ax3.set_xlim(*x_temp_limits)
    ax3.set_xticks(x_temp_ticks)
    ax3.set_xlabel("Likelihood bucket threshold")
    ax3.set_ylabel("Macro F1")
    ax3.set_title("Macro F1 by threshold, model, and temperature")
    ax3.set_ylim(0.3, 0.8)
    ax3.set_yticks(nice_numeric_ticks(0.3, 0.8, 0.1))
    set_publication_axes(ax3, show_grid_y=True, show_grid_x=False)
    ax3.legend(frameon=False, fontsize=7, ncol=2)
    _save_multi(fig3, out_dir / "macro_f1_threshold_curves_by_model_temperature")

    pooled = work.assign(model_pool=work["model_canon"].map(_model_pool_label))
    pooled = pooled.dropna(subset=["model_pool"])
    if pooled.empty:
        return

    pooled_curves: list[pd.DataFrame] = []
    pooled_best_rows: list[dict[str, object]] = []
    for pool, sub in pooled.groupby("model_pool", dropna=False):
        y = sub[truth_col].astype(int).to_numpy()
        bucket = sub["bucket"].astype(int).to_numpy()
        if len(y) == 0:
            continue
        curve = _threshold_sweep(y, bucket)
        curve["model_pool"] = pool
        pooled_curves.append(curve)
        best = curve.sort_values(
            ["macro_f1", "mcc", "precision"], ascending=False
        ).iloc[0]
        pooled_best_rows.append(
            {
                "model_pool": pool,
                "best_threshold": int(best["threshold"]),
                "best_macro_f1": float(best["macro_f1"]),
                "mcc_at_best": float(best["mcc"]),
                "precision_at_best": float(best["precision"]),
                "recall_at_best": float(best["recall"]),
                "n_eval": int(best["n_eval"]),
            }
        )

    pooled_curves_df = (
        pd.concat(pooled_curves, ignore_index=True) if pooled_curves else pd.DataFrame()
    )
    if pooled_curves_df.empty:
        return
    pooled_curves_df["pool_order"] = pooled_curves_df["model_pool"].map(
        _model_pool_sort_key
    )
    pooled_curves_df = pooled_curves_df.sort_values(
        ["pool_order", "model_pool", "threshold"]
    )
    write_dataframe(
        pooled_curves_df.drop(columns=["pool_order"]),
        out_dir / "macro_f1_threshold_curves_by_model_pool.csv",
    )
    if pooled_best_rows:
        write_dataframe(
            pd.DataFrame(pooled_best_rows)
            .assign(pool_order=lambda df: df["model_pool"].map(_model_pool_sort_key))
            .sort_values(["pool_order", "best_macro_f1"], ascending=[True, False])
            .drop(columns=["pool_order"]),
            out_dir / "best_thresholds_by_model_pool_macroF1.csv",
        )

    fig_pool, ax_pool = plt.subplots(figsize=(7.0, 4.6))
    for pool, grp in pooled_curves_df.groupby("model_pool"):
        style = _model_pool_line_style(pool)
        grp_sorted = grp.sort_values("threshold")
        ax_pool.plot(
            grp_sorted["threshold"],
            grp_sorted["macro_f1"],
            marker=style["marker"],
            linewidth=float(style["linewidth"]),
            markersize=4,
            color=str(style["color"]),
            linestyle=str(style["linestyle"]),
            label=str(pool),
        )
    ax_pool.set_xlabel("Likelihood bucket threshold")
    ax_pool.set_ylabel("Macro F1")
    ax_pool.set_title("Macro F1 by threshold for pooled model families")
    ax_pool.set_xlim(*x_limits)
    ax_pool.set_xticks(x_ticks)
    ax_pool.set_ylim(0.3, 0.8)
    ax_pool.set_yticks(nice_numeric_ticks(0.3, 0.8, 0.1))
    set_publication_axes(ax_pool, show_grid_y=True, show_grid_x=False)
    ax_pool.legend(frameon=False)
    _save_multi(fig_pool, out_dir / "macro_f1_threshold_curves_by_model_pool")


def _basic_threshold_tables(
    frame: pd.DataFrame, truth_col: str, stage_dir: Path
) -> None:
    """Write fixed-threshold comparison tables for the bronze/silver/gold tiers."""
    if "Likelihood_of_IBD_1" not in frame.columns or "model_canon" not in frame.columns:
        return
    out_dir = ensure_dir(stage_dir / "basic_thresholds")
    group_cols = ["model_canon"]
    if "temperature" in frame.columns:
        group_cols.append("temperature")
    if "report_sequence_name" in frame.columns:
        group_cols.append("report_sequence_name")

    work = frame[[truth_col, "Likelihood_of_IBD_1", *group_cols]].copy()
    work[truth_col] = pd.to_numeric(work[truth_col], errors="coerce")
    work["bucket"] = pd.to_numeric(work["Likelihood_of_IBD_1"], errors="coerce")
    if "temperature" in work.columns:
        work["temperature"] = pd.to_numeric(work["temperature"], errors="coerce").round(
            2
        )
    work = work.dropna(subset=[truth_col, "bucket"])
    if work.empty:
        return

    rows: list[dict[str, object]] = []
    for keys, sub in work.groupby(group_cols, dropna=False):
        if not isinstance(keys, tuple):
            keys = (keys,)
        base = dict(zip(group_cols, keys))
        y_true = sub[truth_col].astype(int).to_numpy()
        bucket = sub["bucket"].astype(int).to_numpy()
        for tier_name, threshold in THRESHOLD_TIERS:
            metrics = _threshold_metrics(y_true, bucket, threshold)
            rows.append(
                {
                    "model": base.get("model_canon"),
                    "temperature": base.get("temperature"),
                    "dataset": base.get("report_sequence_name", "overall"),
                    "threshold_label": f"{tier_name} ({threshold}+)",
                    "threshold": threshold,
                    **metrics,
                }
            )
    if not rows:
        return

    result = pd.DataFrame(rows).sort_values(
        ["model", "temperature", "dataset", "threshold"],
        kind="stable",
    )
    write_dataframe(result, out_dir / "basic_threshold_performance.csv")

    summary = (
        result.groupby(["model", "threshold_label"], dropna=False)["macro_f1"]
        .mean()
        .reset_index()
    )
    if summary.empty:
        return
    pivot = summary.pivot(index="model", columns="threshold_label", values="macro_f1")
    tier_order = [f"{name} ({threshold}+)" for name, threshold in THRESHOLD_TIERS]
    pivot = pivot.reindex(columns=tier_order)
    pivot.index = [format_model_display_name(value) for value in pivot.index]
    fig4, ax4 = plt.subplots(figsize=(7.2, 4.6))
    x = np.arange(len(pivot.index))
    width = 0.24
    tier_colors = {
        tier_order[0]: "#CD7F32",
        tier_order[1]: "#9CA3AF",
        tier_order[2]: "#D4AF37",
    }
    for index, column in enumerate(pivot.columns):
        ax4.bar(
            x + (index - 1) * width,
            pivot[column].to_numpy(dtype=float),
            width=width,
            color=tier_colors.get(column, "#4C78A8"),
            edgecolor="black",
            linewidth=1.0,
            label=sentence_case(column),
            zorder=3,
        )
    ax4.set_xticks(x)
    ax4.set_xticklabels(pivot.index, rotation=20, ha="right")
    ax4.set_ylabel("Macro F1")
    ax4.set_xlabel("Model")
    ax4.set_title("Bronze, silver, and gold threshold performance")
    ax4.set_yticks(nice_numeric_ticks(0.0, 1.0, 0.1))
    set_publication_axes(ax4, show_grid_y=True, show_grid_x=False)
    ax4.legend(frameon=False)
    fig4.tight_layout()
    _save_multi(fig4, out_dir / "threshold_tier_performance_summary")
