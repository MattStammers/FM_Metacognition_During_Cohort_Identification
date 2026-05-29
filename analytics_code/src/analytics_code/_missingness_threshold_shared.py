"""Shared helpers for the missingness, threshold and calibration stage."""

from __future__ import annotations

import logging
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.metrics import (
    f1_score,
    matthews_corrcoef,
    precision_score,
    recall_score,
    roc_auc_score,
)

from analytics_code.common import (
    PUBLICATION_DPI,
    apply_publication_style,
    ensure_dir,
    format_factor_value,
    model_color,
    nice_numeric_ticks,
    sentence_case,
    set_publication_axes,
)
from analytics_code.config import AnalysisConfig
from analytics_code.predictions import clean_likelihood as _shared_clean_likelihood
from analytics_code.truth_labels import (
    DEFAULT_TRUTH_CANDIDATES,
    prepare_truth_frame,
    truth_mode,
)

LOGGER = logging.getLogger("analytics_code.missingness_threshold")

TRUTH_CANDIDATES = DEFAULT_TRUTH_CANDIDATES
THINKING_PATTERNS = ("qwen", "deepseek")
NON_THINKING_PATTERNS = ("m42", "mixtral")
GEMMA_PATTERNS = ("gemma",)
SMALL_MODEL_EXCLUDE_PATTERNS = ("32b",)
THRESHOLD_TIERS: tuple[tuple[str, int], ...] = (
    ("Bronze", 5),
    ("Silver", 6),
    ("Gold", 7),
)


def _sanitize(name: str) -> str:
    """Return a filesystem-safe slug derived from ``name``."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(name)).strip("_")


def _detect_truth_column(
    frame: pd.DataFrame,
    prepared_truth_col: str | None = None,
    *,
    require_prepared_truth: bool = False,
) -> str | None:
    """Return the active truth column for the loaded analysis frame."""
    if require_prepared_truth:
        return prepared_truth_col
    if prepared_truth_col is not None:
        return prepared_truth_col
    for candidate in TRUTH_CANDIDATES:
        if candidate in frame.columns:
            return candidate
    return None


def _clean_likelihood(series: pd.Series) -> pd.Series:
    """Deprecated alias for :func:`analytics_code.predictions.clean_likelihood`."""
    return _shared_clean_likelihood(series)


def _load_frame(config: AnalysisConfig) -> tuple[pd.DataFrame, str | None, bool]:
    """Load the merged-or-parsed per-row dataframe written by ``data_prep``."""
    merged = config.paths["output_root"] / "data_prep" / "merged_outputs.csv"
    if not merged.exists():
        merged = config.paths["output_root"] / "data_prep" / "parsed_outputs.csv"
    if not merged.exists():
        LOGGER.warning("No parsed/merged input available at %s", merged)
        return pd.DataFrame(), None, False
    frame = pd.read_csv(merged)
    mode = truth_mode(config)
    frame, truth_col = prepare_truth_frame(frame, mode=mode)
    return frame, truth_col, mode in {"document", "document_complete"}


def _save_multi(fig, stem: Path) -> None:
    """Save ``fig`` as PNG, PDF and SVG sharing the same path stem."""
    ensure_dir(stem.parent)
    apply_publication_style()
    for ext in ("png", "pdf", "svg"):
        fig.savefig(
            stem.with_suffix(f".{ext}"), dpi=PUBLICATION_DPI, bbox_inches="tight"
        )
    plt.close(fig)


def _model_pool_label(value: object) -> str | None:
    """Collapse models into pooled thinking / non-thinking families."""
    low = str(value).lower()
    if any(pattern in low for pattern in GEMMA_PATTERNS):
        return "Gemma (2026)"
    if any(pattern in low for pattern in THINKING_PATTERNS):
        return "Thinking (2025)"
    if any(pattern in low for pattern in NON_THINKING_PATTERNS):
        return "Non-thinking (2024)"
    return None


def _model_pool_sort_key(value: object) -> int:
    """Return a stable display order for pooled model-family charts."""
    order = {
        "Non-thinking (2024)": 0,
        "Thinking (2025)": 1,
        "Gemma (2026)": 2,
    }
    return order.get(str(value), len(order))


def _model_pool_line_style(value: object) -> dict[str, object]:
    """Return publication styling for pooled model-family curves."""
    pool = str(value)
    styles = {
        "Non-thinking (2024)": {
            "color": "#D55E00",
            "linestyle": "-",
            "linewidth": 1.9,
            "marker": "o",
        },
        "Thinking (2025)": {
            "color": "#0072B2",
            "linestyle": "-",
            "linewidth": 1.9,
            "marker": "o",
        },
        "Gemma (2026)": {
            "color": "#2CA02C",
            "linestyle": "-",
            "linewidth": 2.4,
            "marker": "o",
        },
    }
    return styles.get(
        pool,
        {
            "color": "#4C78A8",
            "linestyle": "-",
            "linewidth": 1.9,
            "marker": "o",
        },
    )


def _is_small_model(value: object) -> bool:
    """Return ``True`` for the smaller-model comparison subset."""
    low = str(value).lower()
    return not any(pattern in low for pattern in SMALL_MODEL_EXCLUDE_PATTERNS)


def _annotate_bar_values(ax, bars, *, as_percent: bool = False) -> None:
    """Draw value labels just above each bar."""
    heights = [float(bar.get_height()) for bar in bars if np.isfinite(bar.get_height())]
    max_height = max(heights, default=0.0)
    offset = max(0.6, max_height * 0.015)
    for bar in bars:
        height = float(bar.get_height())
        if not np.isfinite(height):
            continue
        label = f"{height:.1f}%" if as_percent else f"{height:.2f}"
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            height + offset,
            label,
            ha="center",
            va="bottom",
            fontsize=8,
        )


def _bar_plot(
    df: pd.DataFrame,
    x: str,
    y: str,
    title: str,
    outfile: Path,
    *,
    sort_desc: bool = True,
    annotate_as_percent: bool = False,
) -> None:
    """Render a categorical bar plot sorted by ``y`` with optional labels."""
    plot_df = df[[x, y]].copy()
    plot_df[y] = pd.to_numeric(plot_df[y], errors="coerce")
    plot_df = plot_df.dropna(subset=[y])
    if plot_df.empty:
        return
    if sort_desc:
        plot_df = plot_df.sort_values(y, ascending=False, kind="stable")
    labels = [format_factor_value(x, value) for value in plot_df[x]]
    values = plot_df[y].astype(float).to_numpy()
    horizontal = (
        x in {"report_sequence_name", "data_type", "dataset"}
        or max((len(label) for label in labels), default=0) > 18
    )
    if x == "model_canon":
        colors = [model_color(value) for value in plot_df[x]]
    elif x == "shot_type":
        colors = ["#0072B2", "#56B4E9", "#009E73"][: len(plot_df)]
        if len(colors) < len(plot_df):
            colors = ["#0072B2"] * len(plot_df)
    elif x == "temperature":
        colors = ["#D32F2F"] * len(plot_df)
    else:
        colors = ["#0F766E"] * len(plot_df)
    if horizontal:
        fig_h = max(4.2, 0.42 * len(plot_df) + 1.8)
        fig_w = 11.5 if x in {"report_sequence_name", "data_type", "dataset"} else 8.2
        fig, ax = plt.subplots(figsize=(fig_w, fig_h))
        positions = np.arange(len(plot_df))
        bars = ax.barh(
            positions,
            values,
            color=colors,
            edgecolor="black",
            linewidth=1.15,
            zorder=3,
        )
        ax.set_yticks(positions)
        ax.set_yticklabels(labels)
        ax.invert_yaxis()
        ax.set_xlabel(sentence_case(y))
        ax.set_ylabel(sentence_case(x))
    else:
        fig_w = max(6.8, 1.25 * len(plot_df) + 1.8)
        fig, ax = plt.subplots(figsize=(fig_w, 4.8))
        bars = ax.bar(
            labels,
            values,
            color=colors,
            edgecolor="black",
            linewidth=1.15,
            zorder=3,
        )
        ax.set_xlabel(sentence_case(x))
        ax.set_ylabel(sentence_case(y))
        ax.tick_params(axis="x", rotation=20, labelsize=9)
        for label in ax.get_xticklabels():
            label.set_ha("right")
    ax.set_title(sentence_case(title))
    set_publication_axes(ax, show_grid_y=not horizontal, show_grid_x=False)
    if annotate_as_percent:
        max_y = float(plot_df[y].max()) if not plot_df.empty else 0.0
        if horizontal:
            for bar, value in zip(bars, values):
                ax.text(
                    value + max(0.35, max_y * 0.012),
                    bar.get_y() + bar.get_height() / 2.0,
                    f"{value:.2f}%",
                    va="center",
                    ha="left",
                    fontsize=9,
                )
            axis_max = max(5.0, max_y * 1.18)
            tick_step = 5.0 if axis_max >= 15.0 else 2.0 if axis_max >= 6.0 else 1.0
            ax.set_xlim(0, axis_max)
            ax.set_xticks(nice_numeric_ticks(0.0, max(5.0, max_y * 1.05), tick_step))
        else:
            _annotate_bar_values(ax, bars, as_percent=True)
            axis_max = max(5.0, max_y * 1.14)
            tick_step = 5.0 if axis_max >= 15.0 else 2.0 if axis_max >= 6.0 else 1.0
            ax.set_ylim(0, axis_max)
            ax.set_yticks(nice_numeric_ticks(0.0, max(5.0, max_y * 1.05), tick_step))
    _save_multi(fig, outfile.with_suffix(""))


def _dropout_summary(frame: pd.DataFrame, by: list[str]) -> pd.DataFrame:
    """Summarise dropout percentage for the requested grouping columns."""
    return (
        frame.groupby(by, dropna=False)["_is_dropout"]
        .mean()
        .mul(100.0)
        .rename("dropout_pct")
        .reset_index()
    )


def _threshold_metrics(
    y_true: np.ndarray, bucket: np.ndarray, threshold: int
) -> dict[str, float]:
    """Return classification metrics at a single integer likelihood threshold."""
    pred = (bucket >= threshold).astype(int)
    try:
        auroc = float(roc_auc_score(y_true, bucket / 10.0))
    except ValueError:
        auroc = float("nan")
    return {
        "accuracy": float((pred == y_true).mean()),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "micro_f1": float(f1_score(y_true, pred, average="micro", zero_division=0)),
        "macro_f1": float(f1_score(y_true, pred, average="macro", zero_division=0)),
        "mcc": float(matthews_corrcoef(y_true, pred))
        if pred.std() > 0 and y_true.std() > 0
        else 0.0,
        "auroc": auroc,
        "n_eval": int(len(y_true)),
    }
