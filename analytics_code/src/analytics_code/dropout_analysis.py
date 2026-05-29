"""Dropout analysis stage.

This produces the three
signature breakdown workbooks (``total_success_breakdown.xlsx``,
``combined_percent_scoring_more_than_5_breakdown.xlsx``,
``no_repair_breakdown.xlsx``) and three plot families per metric
(``<metric>_mean_ci.png``, ``<metric>_skew_kurtosis.png``,
``<metric>_forest_improved.png``).
"""

from __future__ import annotations

import logging
import re
import warnings
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats

from analytics_code.common import (
    FAIR_DATA_TYPES,
    FAIR_TEMPERATURE_LABELS,
    ZERO_SHOT_LABEL,
    ensure_dir,
    format_factor_value,
    model_color,
    nice_numeric_ticks,
    save_figure,
    sentence_case,
    set_publication_axes,
    write_dataframe,
)
from analytics_code.config import AnalysisConfig

LOGGER = logging.getLogger("analytics_code.dropout")

TEMP_RE = re.compile(r"_t(\d+_\d{2})")
MODEL_RE = re.compile(r"^(.+?)_t")
SHOT_RE = re.compile(r"_(zero|single|dual)_shot")
DATA_TYPE_RE = re.compile(r"_(?:zero|single|dual)_shot_(.+)$")

METRIC_CONFIG = {
    "total_success_percent": {
        "workbook": "total_success_breakdown.xlsx",
        "xlabel": "Total successful responses (%)",
        "prefix": "total_success_percent",
    },
    "percent_ge_5": {
        "workbook": "combined_percent_scoring_more_than_5_breakdown.xlsx",
        "xlabel": "Average likelihood of IBD responses \u22655 (%)",
        "prefix": "percent_ge_5",
    },
    "no_repair_percent": {
        "workbook": "no_repair_breakdown.xlsx",
        "xlabel": "No-repair-needed responses (%)",
        "prefix": "no_repair_percent",
    },
}

ATTRIBUTE_ORDER = ["temperature", "data_type", "model", "shot"]
ATTRIBUTE_COLORS = {
    "temperature": "#D32F2F",
    "data_type": "#1E88E5",
    "model": "#F57C00",
    "shot": "#2E7D32",
}


def _parse_folder(folder: str) -> dict[str, str | None]:
    """Decode a folder name into its (model, temperature, shot, data_type) parts.

    Folder names follow the convention
    ``<model>_t<temp>_<shot>_shot_<data_type>``. Any component that
    cannot be matched is returned as ``None``.
    """
    m_model = MODEL_RE.search(folder)
    m_temp = TEMP_RE.search(folder)
    m_shot = SHOT_RE.search(folder)
    m_data = DATA_TYPE_RE.search(folder)
    return {
        "model": m_model.group(1) if m_model else None,
        "temperature": m_temp.group(1) if m_temp else None,
        "shot": m_shot.group(1) if m_shot else None,
        "data_type": m_data.group(1) if m_data else None,
    }


def _parse_percent_column(value: object) -> float:
    """Coerce a heterogeneous percentage cell into a numeric percentage.

    Handles boolean flags, fractional values in ``[0, 1]`` (rescaled to
    ``[0, 100]``), bare numerics and the formatted ``"x/y (z%)"``
    strings produced by the data-prep summary tables.
    """
    if isinstance(value, bool):
        return 100.0 if value else 0.0
    if isinstance(value, (int, float)) and not pd.isna(value):
        num = float(value)
        return num * 100.0 if 0.0 <= num <= 1.0 else num
    if value is None or (isinstance(value, float) and pd.isna(value)):
        return np.nan
    match = re.search(r"\(([-+]?\d+(?:\.\d+)?)%\)", str(value))
    if match:
        return float(match.group(1))
    try:
        return float(value)
    except (TypeError, ValueError):
        return np.nan


def _compute_stats(values: pd.Series) -> dict[str, float]:
    """Return descriptive statistics (mean, CI, IQR, skew, kurtosis) for ``values``.

    Confidence intervals use the Student t distribution; skew and
    kurtosis use the unbiased estimators from :mod:`scipy.stats`.
    Returns ``NaN`` for every metric when the series contains no
    finite values.
    """
    v = pd.to_numeric(values, errors="coerce").dropna()
    n = len(v)
    keys = (
        "mean",
        "std_dev",
        "n",
        "ci_lower",
        "ci_upper",
        "median",
        "iqr",
        "skewness",
        "kurtosis",
    )
    if n == 0:
        return {k: np.nan for k in keys}
    mean = float(v.mean())
    std = float(v.std(ddof=1)) if n > 1 else 0.0
    median = float(v.median())
    q75, q25 = np.percentile(v, [75, 25])
    iqr = float(q75 - q25)
    if n > 1 and std > 0:
        half = float(stats.t.ppf(0.975, df=n - 1)) * std / np.sqrt(n)
        ci_lower, ci_upper = mean - half, mean + half
    else:
        ci_lower, ci_upper = mean, mean
    skew = 0.0
    kurt = 0.0
    if n > 2 and std > 0:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", category=RuntimeWarning)
            skew = float(stats.skew(v, bias=False))
            if n > 3:
                kurt = float(stats.kurtosis(v, fisher=True, bias=False))
    return {
        "mean": mean,
        "std_dev": std,
        "n": int(n),
        "ci_lower": ci_lower,
        "ci_upper": ci_upper,
        "median": median,
        "iqr": iqr,
        "skewness": skew,
        "kurtosis": kurt,
    }


def _filtered(frame: pd.DataFrame, attribute: str) -> pd.DataFrame:
    """Apply the per-attribute FAIR filter used for breakdown tables.

    Mirrors :func:`analytics_code.full_performance._fair_filter` so
    that the dropout breakdowns and the FAIR performance tables are
    computed on the same sub-cube of the experimental matrix. The
    rules are:

    * ``temperature``: zero-shot prompts only and ``data_type`` in
      :data:`analytics_code.common.FAIR_DATA_TYPES`. The temperature
      column is *not* subset because it is the variable under study;
      the comparison is intrinsically confounded with model because
      the production matrix runs each model at a single temperature.
    * ``data_type``: zero-shot prompts only and temperature in
      :data:`analytics_code.common.FAIR_TEMPERATURE_LABELS`.
    * ``model``: zero-shot prompts only, temperature in
      :data:`analytics_code.common.FAIR_TEMPERATURE_LABELS` and
      ``data_type`` in :data:`analytics_code.common.FAIR_DATA_TYPES`.
    * ``shot``: temperature in
      :data:`analytics_code.common.FAIR_TEMPERATURE_LABELS` and
      ``data_type`` in :data:`analytics_code.common.FAIR_DATA_TYPES`.

    Parameters
    ----------
    frame:
        Per-folder summary dataframe with the columns produced by
        :func:`_parse_folder` (``shot``, ``temperature``, ``model``,
        ``data_type``).
    attribute:
        Name of the attribute under study (one of ``"temperature"``,
        ``"data_type"``, ``"model"``, ``"shot"``).

    Returns
    -------
    pandas.DataFrame
        The filtered view of ``frame``. Returns the input unchanged
        when ``attribute`` is not recognised.
    """
    fair_temps = set(FAIR_TEMPERATURE_LABELS)
    fair_dtypes = set(FAIR_DATA_TYPES)
    df = frame.copy()
    if attribute == "temperature":
        df = df[df["shot"] == ZERO_SHOT_LABEL]
        df = df[df["data_type"].isin(fair_dtypes)]
    elif attribute == "data_type":
        df = df[df["shot"] == ZERO_SHOT_LABEL]
        df = df[df["temperature"].isin(fair_temps)]
    elif attribute == "model":
        df = df[df["shot"] == ZERO_SHOT_LABEL]
        df = df[df["temperature"].isin(fair_temps)]
        df = df[df["data_type"].isin(fair_dtypes)]
    elif attribute == "shot":
        df = df[df["temperature"].isin(fair_temps)]
        df = df[df["data_type"].isin(fair_dtypes)]
    return df


def _format_ci(row: pd.Series) -> str:
    """Format a row's mean and CI as ``mean% (lo%-hi%)``."""
    return f"{row['mean']:.1f}% ({row['ci_lower']:.1f}%\u2013{row['ci_upper']:.1f}%)"


def _format_median_iqr(row: pd.Series) -> str:
    """Format a row's median and IQR as ``median% [iqr]``."""
    return f"{row['median']:.1f}% [{row['iqr']:.1f}]"


def _build_breakdown(frame: pd.DataFrame, metric: str) -> dict[str, pd.DataFrame]:
    """Build the per-attribute breakdown tables for a metric.

    Returns a mapping ``{attribute: table}`` where each table has one
    row per attribute level and the descriptive statistics returned by
    :func:`_compute_stats` plus formatted ``percent_ci`` and
    ``median_iqr`` strings.
    """
    tables: dict[str, pd.DataFrame] = {}
    for attr in ATTRIBUTE_ORDER:
        sub = _filtered(frame, attr)
        if sub.empty:
            continue
        rows = []
        for name, group in sub.groupby(attr, dropna=False):
            rows.append({attr: name, **_compute_stats(group[metric])})
        table = pd.DataFrame(rows)
        if table.empty:
            continue
        table["percent_ci"] = table.apply(_format_ci, axis=1)
        table["median_iqr"] = table.apply(_format_median_iqr, axis=1)
        table["display_label"] = table[attr].map(
            lambda value: format_factor_value(attr, value)
        )
        if attr == "temperature":
            table = table.sort_values(attr)
        tables[attr] = table
    return tables


def _plot_mean_ci(
    breakdown: dict[str, pd.DataFrame], xlabel: str, out_path: Path
) -> None:
    """Plot a side-by-side mean +/- CI panel, one column per attribute."""
    attrs = [a for a in ATTRIBUTE_ORDER if a in breakdown]
    if not attrs:
        return
    fig, axes = plt.subplots(1, len(attrs), figsize=(4.4 * len(attrs), 4.8))
    if len(attrs) == 1:
        axes = [axes]
    for ax, attr in zip(axes, attrs):
        tab = breakdown[attr]
        y = np.arange(len(tab))
        lo = (tab["mean"] - tab["ci_lower"]).clip(lower=0).values
        hi = (tab["ci_upper"] - tab["mean"]).clip(lower=0).values
        color = ATTRIBUTE_COLORS.get(attr, "#444")
        ax.errorbar(
            tab["mean"].values,
            y,
            xerr=[lo, hi],
            fmt="o",
            color=color,
            ecolor=color,
            capsize=3,
            linewidth=1.3,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(tab["display_label"].astype(str).values, fontsize=8)
        ax.set_xlabel(xlabel, fontsize=9)
        ax.set_title(sentence_case(attr), fontsize=11)
        set_publication_axes(ax, show_grid_y=False, show_grid_x=True)
    fig.tight_layout()
    save_figure(out_path)


def _plot_skew_kurtosis(
    breakdown: dict[str, pd.DataFrame], metric: str, out_path: Path
) -> None:
    """Scatter the skewness vs excess kurtosis of each attribute level."""
    rows = []
    for attr, tab in breakdown.items():
        for _, r in tab.iterrows():
            rows.append(
                {
                    "attribute_type": attr,
                    "label": str(r[attr]),
                    "skewness": r["skewness"],
                    "kurtosis": r["kurtosis"],
                }
            )
    if not rows:
        return
    scatter_df = pd.DataFrame(rows)
    fig, ax = plt.subplots(figsize=(7.4, 4.8))
    for attr, grp in scatter_df.groupby("attribute_type"):
        ax.scatter(
            grp["skewness"],
            grp["kurtosis"],
            label=attr,
            s=40,
            color=ATTRIBUTE_COLORS.get(attr, "#444"),
            alpha=0.85,
            edgecolor="white",
            linewidth=0.6,
        )
        for _, r in grp.iterrows():
            ax.annotate(
                r["label"],
                (r["skewness"], r["kurtosis"]),
                fontsize=8,
                alpha=0.7,
                xytext=(4, 2),
                textcoords="offset points",
            )
    ax.axhline(0, linestyle="--", color="black", alpha=0.3, linewidth=0.8)
    ax.axvline(0, linestyle="--", color="black", alpha=0.3, linewidth=0.8)
    ax.set_xlabel(f"Skewness of {sentence_case(metric)}")
    ax.set_ylabel(f"Excess kurtosis of {sentence_case(metric)}")
    ax.set_title(f"Distribution shape for {sentence_case(metric)}")
    ax.legend(frameon=False, fontsize=8)
    set_publication_axes(ax, show_grid_y=True, show_grid_x=True)
    fig.tight_layout()
    save_figure(out_path)


def _plot_forest(
    breakdown: dict[str, pd.DataFrame], metric: str, xlabel: str, out_path: Path
) -> None:
    """Render the multi-panel forest plot used in the dropout analysis report."""
    attrs = [a for a in ATTRIBUTE_ORDER if a in breakdown]
    if not attrs:
        return
    heights = [max(1, len(breakdown[a])) for a in attrs]
    fig, axes = plt.subplots(
        nrows=len(attrs),
        ncols=1,
        figsize=(10.8, max(3.2, sum(heights) * 0.46 + 1.6)),
        gridspec_kw={"height_ratios": heights},
        sharex=True,
    )
    if len(attrs) == 1:
        axes = [axes]
    effective_xlabel = (
        xlabel
        if metric == "percent_ge_5"
        else f"{metric.replace('_', ' ').title()} (%)"
    )
    for ax, attr in zip(axes, attrs):
        tab = breakdown[attr].copy()
        if attr == "temperature":
            tab["_label"] = tab["display_label"]
            tab = tab.sort_values("_label")
        else:
            tab["_label"] = tab["display_label"]
        y = np.arange(len(tab))
        lo = (tab["mean"] - tab["ci_lower"]).clip(lower=0).values
        hi = (tab["ci_upper"] - tab["mean"]).clip(lower=0).values
        if attr == "model":
            bar_colors = ["#E0B400"] * len(tab)
        else:
            bar_colors = [ATTRIBUTE_COLORS.get(attr, "#555")] * len(tab)
        ax.barh(
            y,
            tab["mean"].values,
            color=bar_colors,
            edgecolor="black",
            linewidth=0.9,
            alpha=0.92,
            zorder=2,
        )
        ax.errorbar(
            tab["mean"].values,
            y,
            xerr=[lo, hi],
            fmt="none",
            color="black",
            ecolor="black",
            capsize=3,
            linewidth=1.4,
            markersize=5,
            zorder=4,
        )
        ax.set_yticks(y)
        ax.set_yticklabels(tab["_label"].values, fontsize=8)
        ax.set_ylabel(sentence_case(attr), fontsize=10)
        set_publication_axes(ax, show_grid_y=False, show_grid_x=True)
        xmax = max(100.0, float(np.nanmax(tab["ci_upper"].values)) * 1.05)
        ax.set_xlim(0.0, xmax)
        tick_step = 10.0 if xmax > 40.0 else 5.0
        ax.set_xticks(nice_numeric_ticks(0.0, xmax, tick_step))
    axes[-1].set_xlabel(effective_xlabel, fontsize=9)
    fig.suptitle(f"Forest plot for {sentence_case(metric)}", fontsize=12)
    fig.tight_layout(rect=[0, 0, 1, 0.97])
    save_figure(out_path)


def _resolve_metric(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    """Ensure ``frame`` contains a numeric column for ``metric``.

    When the canonical column is missing, fall back to parsing the
    legacy formatted column (e.g. ``total_successful``) via
    :func:`_parse_percent_column`.
    """
    if metric in frame.columns:
        frame[metric] = pd.to_numeric(frame[metric], errors="coerce")
        return frame
    fallback = {
        "total_success_percent": "total_successful",
        "percent_ge_5": "IBD_predicted_ge_5",
        "no_repair_percent": "no_repair_needed",
    }.get(metric)
    if fallback and fallback in frame.columns:
        frame[metric] = frame[fallback].map(_parse_percent_column)
    return frame


def run_dropout_analysis(
    config: AnalysisConfig, source_path: Path | None = None
) -> dict[str, Path]:
    """Run the dropout-analysis stage.

    Reads the per-folder LLM JSON-processing summary produced by
    :func:`analytics_code.data_prep.run_data_prep`, decomposes the
    folder names into experimental factors, and emits Excel breakdown
    workbooks plus three plot families (mean+CI, skew/kurtosis,
    forest) per metric in :data:`METRIC_CONFIG`. CSV companions of
    every workbook sheet are written next to the Excel file.

    Parameters
    ----------
    config:
        The active :class:`AnalysisConfig`.
    source_path:
        Optional override for the input summary CSV. Defaults to
        ``<output_root>/data_prep/llm_json_processing_summary.csv``.

    Returns
    -------
    dict[str, pathlib.Path]
        Map of artefact name to file path. Empty when the input is
        missing or has the wrong schema.
    """
    output_dir = ensure_dir(config.paths["output_root"] / "dropout_analysis")
    figures_dir = ensure_dir(output_dir / "figures")

    summary_path = source_path or (
        config.paths["output_root"] / "data_prep" / "llm_json_processing_summary.csv"
    )
    if not summary_path.exists():
        LOGGER.warning("Dropout summary input missing: %s", summary_path)
        return {}

    summary = pd.read_csv(summary_path)
    if "folder" not in summary.columns:
        LOGGER.warning("Summary CSV lacks 'folder' column: %s", summary_path)
        return {}

    parsed = summary["folder"].apply(_parse_folder).apply(pd.Series)
    frame = pd.concat([summary, parsed], axis=1)
    for metric in METRIC_CONFIG:
        frame = _resolve_metric(frame, metric)

    outputs: dict[str, Path] = {}
    for metric, cfg in METRIC_CONFIG.items():
        if metric not in frame.columns or frame[metric].dropna().empty:
            LOGGER.info("Skipping metric %s (no data)", metric)
            continue
        breakdown = _build_breakdown(frame, metric)
        if not breakdown:
            continue
        workbook = output_dir / cfg["workbook"]
        ensure_dir(workbook.parent)
        with pd.ExcelWriter(workbook, engine="openpyxl") as writer:
            for attr, tab in breakdown.items():
                tab.to_excel(writer, sheet_name=attr, index=False)
        LOGGER.info("Wrote %s", workbook)
        outputs[cfg["workbook"]] = workbook
        # CSV companions (one per sheet) so the workbook is also inspectable
        # in plain-text editors / VS Code preview.
        csv_dir = ensure_dir(output_dir / Path(cfg["workbook"]).stem)
        for attr, tab in breakdown.items():
            tab.to_csv(csv_dir / f"{attr}.csv", index=False)

        _plot_mean_ci(
            breakdown, cfg["xlabel"], figures_dir / f"{cfg['prefix']}_mean_ci.png"
        )
        outputs[f"{cfg['prefix']}_mean_ci"] = (
            figures_dir / f"{cfg['prefix']}_mean_ci.png"
        )

        _plot_skew_kurtosis(
            breakdown, metric, figures_dir / f"{cfg['prefix']}_skew_kurtosis.png"
        )
        outputs[f"{cfg['prefix']}_skew_kurtosis"] = (
            figures_dir / f"{cfg['prefix']}_skew_kurtosis.png"
        )

        _plot_forest(
            breakdown,
            metric,
            cfg["xlabel"],
            figures_dir / f"{cfg['prefix']}_forest_improved.png",
        )
        outputs[f"{cfg['prefix']}_forest_improved"] = (
            figures_dir / f"{cfg['prefix']}_forest_improved.png"
        )

    records = []
    for attr in ATTRIBUTE_ORDER:
        sub = _filtered(frame, attr)
        if sub.empty:
            continue
        for metric in METRIC_CONFIG:
            if metric not in sub.columns:
                continue
            for name, group in sub.groupby(attr, dropna=False):
                records.append(
                    {
                        "attribute_type": attr,
                        "attribute_value": name,
                        "metric": metric,
                        **_compute_stats(group[metric]),
                    }
                )
    if records:
        outputs["dropout_summary"] = write_dataframe(
            pd.DataFrame(records), output_dir / "dropout_summary.csv"
        )

    return outputs
