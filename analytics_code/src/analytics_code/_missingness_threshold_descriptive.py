"""Descriptive and confidence-complexity outputs for missingness threshold analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import f1_score

from analytics_code._missingness_threshold_shared import (
    _bar_plot,
    _clean_likelihood,
    _dropout_summary,
    _is_small_model,
    _model_pool_label,
    _model_pool_line_style,
    _model_pool_sort_key,
    _sanitize,
    _save_multi,
)
from analytics_code.common import (
    THINKING_POOL_COLORS,
    ensure_dir,
    format_factor_value,
    model_color,
    nice_numeric_ticks,
    sentence_case,
    set_publication_axes,
    write_dataframe,
)


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


def _descriptive_stats(frame: pd.DataFrame, stage_dir: Path) -> None:
    """Write missingness, numeric and dropout summary tables/plots."""
    desc_dir = ensure_dir(stage_dir / "descriptive_stats")
    figs_dir = ensure_dir(desc_dir / "figures")

    if "likelihood_score" in frame.columns:
        frame["Likelihood_of_IBD_1_raw"] = frame["likelihood_score"]
        frame["Likelihood_of_IBD_1"] = _clean_likelihood(frame["likelihood_score"])
    for col in ("certainty_score", "complexity_score"):
        if col in frame.columns:
            frame[col] = pd.to_numeric(frame[col], errors="coerce")

    missingness = (
        (frame.isna().mean().sort_values(ascending=False) * 100)
        .round(2)
        .rename("pct_missing")
        .reset_index()
        .rename(columns={"index": "column"})
    )
    write_dataframe(missingness, desc_dir / "missingness_summary.csv")

    numeric_summary = (
        frame.describe(include="number")
        .T.reset_index()
        .rename(columns={"index": "column"})
    )
    write_dataframe(numeric_summary, desc_dir / "overall_numeric_summary.csv")

    if "model_canon" in frame.columns:
        model_summary = (
            frame.groupby("model_canon", dropna=False)
            .agg(
                n_rows=("model_canon", "size"),
                likelihood_mean=("Likelihood_of_IBD_1", "mean")
                if "Likelihood_of_IBD_1" in frame.columns
                else ("model_canon", "size"),
                certainty_mean=("certainty_score", "mean")
                if "certainty_score" in frame.columns
                else ("model_canon", "size"),
                complexity_mean=("complexity_score", "mean")
                if "complexity_score" in frame.columns
                else ("model_canon", "size"),
            )
            .reset_index()
        )
        write_dataframe(model_summary, desc_dir / "model_level_summary.csv")
        for metric, title in (
            ("likelihood_mean", "Mean Likelihood by Canonical Model"),
            ("certainty_mean", "Mean Certainty by Canonical Model"),
            ("complexity_mean", "Mean Complexity by Canonical Model"),
        ):
            if metric in model_summary.columns:
                _bar_plot(
                    model_summary,
                    "model_canon",
                    metric,
                    title,
                    figs_dir / f"model_{metric}.png",
                )

        group_cols = [
            c
            for c in ("model_canon", "report_sequence_name", "shot_type", "temperature")
            if c in frame.columns
        ]
        if group_cols:
            grouped = (
                frame.groupby(group_cols, dropna=False)
                .agg(
                    n_rows=("model_canon", "size"),
                    likelihood_mean=("Likelihood_of_IBD_1", "mean")
                    if "Likelihood_of_IBD_1" in frame.columns
                    else ("model_canon", "size"),
                    certainty_mean=("certainty_score", "mean")
                    if "certainty_score" in frame.columns
                    else ("model_canon", "size"),
                    complexity_mean=("complexity_score", "mean")
                    if "complexity_score" in frame.columns
                    else ("model_canon", "size"),
                )
                .reset_index()
            )
            write_dataframe(grouped, desc_dir / "grouped_summary.csv")

    output_cols = [
        c
        for c in ("Likelihood_of_IBD_1", "certainty_score", "complexity_score")
        if c in frame.columns
    ]
    if not output_cols:
        return

    frame["_is_dropout"] = frame[output_cols].isna().any(axis=1)
    title_lookup = {
        "temperature": "Dropout by temperature",
        "shot_type": "Dropout by shot type",
        "report_sequence_name": "Dropout by narrative ordering",
        "model_canon": "Dropout by model",
    }
    for by in ("temperature", "shot_type", "report_sequence_name", "model_canon"):
        if by not in frame.columns:
            continue
        dsum = _dropout_summary(frame, [by])
        write_dataframe(dsum, desc_dir / f"dropout_by_{by}.csv")
        _bar_plot(
            dsum,
            by,
            "dropout_pct",
            title_lookup.get(by, f"Dropout by {by}"),
            figs_dir / f"dropout_by_{by}.png",
            annotate_as_percent=True,
        )

    if not {"temperature", "report_sequence_name"}.issubset(frame.columns):
        return

    for context, sub in frame.groupby("report_sequence_name", dropna=False):
        if pd.isna(context) or sub["temperature"].dropna().nunique() <= 1:
            continue
        context_slug = _sanitize(str(context))
        dsum = _dropout_summary(sub, ["temperature"])
        write_dataframe(
            dsum,
            desc_dir / f"dropout_by_temperature_context_{context_slug}.csv",
        )
        _bar_plot(
            dsum,
            "temperature",
            "dropout_pct",
            f"Dropout by temperature - {format_factor_value('report_sequence_name', context)}",
            figs_dir / f"dropout_by_temperature_context_{context_slug}.png",
            annotate_as_percent=True,
        )

        if "model_canon" not in sub.columns:
            continue
        smaller = sub[sub["model_canon"].map(_is_small_model)]
        if smaller.empty or smaller["temperature"].dropna().nunique() <= 1:
            continue
        smaller_dsum = _dropout_summary(smaller, ["temperature"])
        write_dataframe(
            smaller_dsum,
            desc_dir
            / f"dropout_by_temperature_context_{context_slug}_small_models.csv",
        )
        _bar_plot(
            smaller_dsum,
            "temperature",
            "dropout_pct",
            f"Dropout by temperature - {format_factor_value('report_sequence_name', context)} - smaller models",
            figs_dir
            / f"dropout_by_temperature_context_{context_slug}_small_models.png",
            annotate_as_percent=True,
        )

    if {"temperature", "model_canon"}.issubset(frame.columns):
        pooled_temp = _dropout_summary(frame, ["model_canon", "temperature"])
        if not pooled_temp.empty and pooled_temp["temperature"].nunique() > 1:
            write_dataframe(
                pooled_temp, desc_dir / "dropout_by_model_and_temperature.csv"
            )
            pooled_temp["model_label"] = pooled_temp["model_canon"].map(
                lambda value: format_factor_value("model_canon", value)
            )
            pooled_temp["temperature_label"] = pooled_temp["temperature"].map(
                lambda value: format_factor_value("temperature", value)
            )
            fig, ax = plt.subplots(figsize=(8.4, 4.8))
            for model, sub in pooled_temp.groupby("model_canon", sort=False):
                ordered = sub.sort_values("temperature")
                ax.plot(
                    ordered["temperature"],
                    ordered["dropout_pct"],
                    marker="o",
                    linewidth=2.0,
                    markersize=5,
                    color=model_color(model),
                    label=format_factor_value("model_canon", model),
                )
            ax.set_xlabel("Temperature")
            ax.set_ylabel("Dropout percentage")
            ax.set_title("Dropout by temperature and model")
            ax.set_xticks(sorted(pooled_temp["temperature"].dropna().unique()))
            ax.set_xticklabels(
                [
                    format_factor_value("temperature", value)
                    for value in sorted(pooled_temp["temperature"].dropna().unique())
                ]
            )
            ymax = (
                float(pooled_temp["dropout_pct"].max())
                if not pooled_temp.empty
                else 0.0
            )
            ax.set_yticks(nice_numeric_ticks(0.0, max(5.0, ymax * 1.05), 1.0))
            set_publication_axes(ax, show_grid_y=True, show_grid_x=False)
            ax.legend(frameon=False, ncol=2)
            fig.tight_layout()
            _save_multi(fig, figs_dir / "dropout_by_model_temperature")


def _confidence_complexity(
    frame: pd.DataFrame, truth_col: str, stage_dir: Path
) -> None:
    """Plot accuracy vs certainty/complexity per likelihood bucket."""
    needed = ("Likelihood_of_IBD_1", "certainty_score", "complexity_score")
    if not all(col in frame.columns for col in needed):
        return
    out_dir = ensure_dir(stage_dir / "confidence_complexity_plots")
    work = frame[
        [truth_col, "Likelihood_of_IBD_1", "certainty_score", "complexity_score"]
    ].copy()
    work[truth_col] = pd.to_numeric(work[truth_col], errors="coerce")
    work["bucket"] = pd.to_numeric(work["Likelihood_of_IBD_1"], errors="coerce")
    work = work.dropna(subset=[truth_col, "bucket"])
    if work.empty:
        return
    work["pred_binary"] = (work["bucket"] >= 5).astype(int)
    work["correct"] = (work["pred_binary"] == work[truth_col].astype(int)).astype(int)

    rows = []
    for bucket in range(0, 11):
        sub = work[work["bucket"] == bucket]
        if sub.empty:
            continue
        rows.append(
            {
                "bucket": bucket,
                "accuracy": float(sub["correct"].mean()),
                "mean_certainty": float(sub["certainty_score"].mean()),
                "mean_complexity": float(sub["complexity_score"].mean()),
                "n": int(len(sub)),
            }
        )
    agg = pd.DataFrame(rows).sort_values("bucket")
    if agg.empty:
        return
    write_dataframe(agg, out_dir / "bucket_summary_confidence_complexity.csv")
    grey = agg[(agg["bucket"] >= 3) & (agg["bucket"] <= 6)]
    write_dataframe(grey, out_dir / "grey_zone_3_6_summary.csv")

    for y_col, color, label, stem in (
        (
            "mean_certainty",
            "#0072B2",
            "Mean certainty",
            "accuracy_vs_certainty_by_bucket",
        ),
        (
            "mean_complexity",
            "#D55E00",
            "Mean complexity",
            "accuracy_vs_complexity_by_bucket",
        ),
    ):
        fig, ax = plt.subplots(figsize=(6.8, 4.2))
        ax.plot(
            agg["bucket"],
            agg["accuracy"],
            marker="o",
            color="black",
            label="Accuracy",
            linewidth=1.6,
        )
        ax.plot(
            agg["bucket"],
            agg[y_col],
            marker="s",
            color=color,
            label=label,
            linewidth=1.6,
        )
        ax.axvspan(3, 6, color="grey", alpha=0.12, label="Grey zone (3–6)")
        ax.set_xlabel("Likelihood bucket")
        ax.set_ylabel("Value")
        ax.set_xticks(range(int(agg["bucket"].min()), int(agg["bucket"].max()) + 1))
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
        _save_multi(fig, out_dir / stem)

    if not grey.empty:
        fig, ax = plt.subplots(figsize=(6.0, 4.0))
        ax.bar(
            grey["bucket"],
            grey["accuracy"],
            width=0.6,
            color="black",
            alpha=0.7,
            label="Accuracy",
        )
        ax.plot(
            grey["bucket"],
            grey["mean_certainty"],
            marker="o",
            color="#0072B2",
            label="Mean certainty",
        )
        ax.plot(
            grey["bucket"],
            grey["mean_complexity"],
            marker="^",
            color="#D55E00",
            label="Mean complexity",
        )
        ax.set_xlabel("Likelihood bucket (grey zone 3–6)")
        ax.set_ylabel("Value")
        ax.set_xticks([3, 4, 5, 6])
        ax.grid(True, axis="y", linestyle="--", alpha=0.4)
        for spine in ("top", "right"):
            ax.spines[spine].set_visible(False)
        ax.legend(frameon=False, loc="best")
        fig.tight_layout()
        _save_multi(fig, out_dir / "grey_zone_3_6_accuracy_certainty_complexity")

    if "model_canon" not in frame.columns:
        return

    pool_work = frame[
        [truth_col, "Likelihood_of_IBD_1", "certainty_score", "model_canon"]
    ].copy()
    pool_work[truth_col] = pd.to_numeric(pool_work[truth_col], errors="coerce")
    pool_work["bucket"] = pd.to_numeric(
        pool_work["Likelihood_of_IBD_1"], errors="coerce"
    )
    pool_work["certainty_score"] = pd.to_numeric(
        pool_work["certainty_score"], errors="coerce"
    )
    pool_work["model_pool"] = pool_work["model_canon"].map(_model_pool_label)
    pool_work = pool_work.dropna(
        subset=[truth_col, "bucket", "certainty_score", "model_pool"]
    )
    if pool_work.empty:
        return
    pool_work["truth"] = pool_work[truth_col].astype(int)

    pool_rows = []
    for (pool, bucket), sub in pool_work.groupby(
        ["model_pool", "bucket"], dropna=False
    ):
        pool_rows.append(
            {
                "model_pool": pool,
                "bucket": int(bucket),
                "mean_certainty": float(sub["certainty_score"].mean()),
                "observed_ibd_rate": float(sub["truth"].mean()),
                "n": int(len(sub)),
            }
        )
    pool_df = pd.DataFrame(pool_rows)
    pool_df["pool_order"] = pool_df["model_pool"].map(_model_pool_sort_key)
    pool_df = pool_df.sort_values(["pool_order", "model_pool", "bucket"])
    write_dataframe(
        pool_df.drop(columns=["pool_order"]),
        out_dir / "bucket_summary_by_model_pool.csv",
    )
    x_limits, x_ticks = _bucket_axis_limits(pool_df["bucket"])

    fig_pool, ax_pool = plt.subplots(figsize=(6.8, 4.2))
    for pool, sub in pool_df.groupby("model_pool"):
        style = _model_pool_line_style(pool)
        ax_pool.plot(
            sub["bucket"],
            sub["mean_certainty"],
            marker=style["marker"],
            linewidth=float(style["linewidth"]),
            markersize=4,
            linestyle=str(style["linestyle"]),
            color=str(style["color"]),
            label=str(pool),
        )
    ax_pool.axvspan(3, 6, color="grey", alpha=0.12, label="Grey zone (3–6)")
    ax_pool.set_xlabel("Likelihood bucket")
    ax_pool.set_ylabel("Mean certainty")
    ax_pool.set_title("Mean certainty by likelihood bucket for pooled model families")
    ax_pool.set_xlim(*x_limits)
    ax_pool.set_xticks(x_ticks)
    ax_pool.set_ylim(4.0, 10.0)
    ax_pool.set_yticks(nice_numeric_ticks(4.0, 10.0, 0.5))
    set_publication_axes(ax_pool, show_grid_y=True, show_grid_x=False)
    ax_pool.legend(frameon=False)
    fig_pool.tight_layout()
    _save_multi(fig_pool, out_dir / "certainty_by_likelihood_bucket_model_pool")

    fig_rate, ax_rate = plt.subplots(figsize=(6.8, 4.2))
    for pool, sub in pool_df.groupby("model_pool"):
        style = _model_pool_line_style(pool)
        ax_rate.plot(
            sub["bucket"],
            sub["observed_ibd_rate"],
            marker=style["marker"],
            linewidth=float(style["linewidth"]),
            markersize=4,
            linestyle=str(style["linestyle"]),
            color=str(style["color"]),
            label=str(pool),
        )
    ax_rate.plot(
        [x_ticks[0], x_ticks[-1]],
        [0, 1],
        linestyle="--",
        color="black",
        linewidth=1.0,
        alpha=0.7,
    )
    ax_rate.axvspan(3, 6, color="grey", alpha=0.12, label="Grey zone (3–6)")
    ax_rate.set_xlabel("Likelihood bucket")
    ax_rate.set_ylabel("Observed IBD rate")
    ax_rate.set_title(
        "Observed IBD rate by likelihood bucket for pooled model families"
    )
    ax_rate.set_xlim(*x_limits)
    ax_rate.set_xticks(x_ticks)
    ax_rate.set_yticks(nice_numeric_ticks(0.0, 1.0, 0.1))
    set_publication_axes(ax_rate, show_grid_y=True, show_grid_x=False)
    ax_rate.legend(frameon=False)
    fig_rate.tight_layout()
    _save_multi(fig_rate, out_dir / "observed_ibd_rate_by_likelihood_bucket_model_pool")

    certainty_work = frame[
        [truth_col, "Likelihood_of_IBD_1", "certainty_score", "model_canon"]
    ].copy()
    certainty_work[truth_col] = pd.to_numeric(
        certainty_work[truth_col], errors="coerce"
    )
    certainty_work["bucket"] = pd.to_numeric(
        certainty_work["Likelihood_of_IBD_1"], errors="coerce"
    )
    certainty_work["certainty_score"] = pd.to_numeric(
        certainty_work["certainty_score"], errors="coerce"
    )
    certainty_work["model_pool"] = certainty_work["model_canon"].map(_model_pool_label)
    certainty_work = certainty_work.dropna(
        subset=[truth_col, "bucket", "certainty_score", "model_pool"]
    )
    if certainty_work.empty:
        return
    certainty_work["truth"] = certainty_work[truth_col].astype(int)
    certainty_work["pred"] = (certainty_work["bucket"] >= 5).astype(int)
    certainty_work["certainty_bucket"] = (
        certainty_work["certainty_score"].round().clip(1, 10).astype(int)
    )

    certainty_rows = []
    for (pool, certainty_bucket), sub in certainty_work.groupby(
        ["model_pool", "certainty_bucket"], dropna=False
    ):
        certainty_rows.append(
            {
                "model_pool": pool,
                "certainty_bucket": int(certainty_bucket),
                "macro_f1": float(
                    f1_score(
                        sub["truth"].to_numpy(dtype=int),
                        sub["pred"].to_numpy(dtype=int),
                        average="macro",
                        zero_division=0,
                    )
                ),
                "n": int(len(sub)),
            }
        )
    certainty_df = pd.DataFrame(certainty_rows)
    certainty_df["pool_order"] = certainty_df["model_pool"].map(_model_pool_sort_key)
    certainty_df = certainty_df.sort_values(
        ["pool_order", "model_pool", "certainty_bucket"]
    )
    write_dataframe(
        certainty_df.drop(columns=["pool_order"]),
        out_dir / "macro_f1_by_certainty_bucket_model_pool.csv",
    )

    fig_cert, ax_cert = plt.subplots(figsize=(6.8, 4.2))
    for pool, sub in certainty_df.groupby("model_pool"):
        style = _model_pool_line_style(pool)
        ax_cert.plot(
            sub["certainty_bucket"],
            sub["macro_f1"],
            marker=style["marker"],
            linewidth=float(style["linewidth"]),
            markersize=4,
            linestyle=str(style["linestyle"]),
            color=str(style["color"]),
            label=str(pool),
        )
    certainty_ticks = list(
        range(
            int(certainty_df["certainty_bucket"].min()),
            int(certainty_df["certainty_bucket"].max()) + 1,
        )
    )
    ax_cert.set_xlabel("Certainty bucket")
    ax_cert.set_ylabel("Macro F1")
    ax_cert.set_title("Macro F1 by certainty bucket for pooled model families")
    ax_cert.set_xlim(certainty_ticks[0] - 0.1, certainty_ticks[-1] + 0.1)
    ax_cert.set_xticks(certainty_ticks)
    ax_cert.set_ylim(0.0, 1.0)
    ax_cert.set_yticks(nice_numeric_ticks(0.0, 1.0, 0.1))
    set_publication_axes(ax_cert, show_grid_y=True, show_grid_x=False)
    ax_cert.legend(frameon=False)
    fig_cert.tight_layout()
    _save_multi(fig_cert, out_dir / "macro_f1_by_certainty_bucket_model_pool")
