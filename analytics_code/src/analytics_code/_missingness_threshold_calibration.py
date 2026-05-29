"""Calibration outputs for missingness threshold analysis."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analytics_code._missingness_threshold_shared import (
    _model_pool_label,
    _model_pool_line_style,
    _model_pool_sort_key,
    _sanitize,
    _save_multi,
)
from analytics_code.common import ensure_dir, write_dataframe


def _calibration_table(y: np.ndarray, p: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Return the per-bin reliability table for a probabilistic classifier."""
    bins = np.linspace(0.0, 1.0, n_bins + 1)
    idx = np.clip(np.digitize(p, bins) - 1, 0, n_bins - 1)
    rows = []
    for index in range(n_bins):
        mask = idx == index
        if not np.any(mask):
            continue
        rows.append(
            {
                "bin": index,
                "bin_left": float(bins[index]),
                "bin_right": float(bins[index + 1]),
                "mean_pred": float(p[mask].mean()),
                "obs_rate": float(y[mask].mean()),
                "count": int(mask.sum()),
            }
        )
    return pd.DataFrame(rows)


def _plot_reliability(
    table: pd.DataFrame,
    model: str,
    brier: float,
    ece: float,
    mce: float,
    out_dir: Path,
) -> None:
    """Render a reliability diagram annotated with Brier/ECE/MCE."""
    fig, ax = plt.subplots(figsize=(6.2, 4.0))
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="black",
        linewidth=1.0,
        alpha=0.7,
        label="Perfect",
    )
    ax.plot(
        table["mean_pred"],
        table["obs_rate"],
        marker="o",
        linewidth=1.6,
        color="#0072B2",
        label="Empirical",
    )
    ax.axvline(0.5, color="#D55E00", linestyle=":", linewidth=1.2)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability (Likelihood/10)")
    ax.set_ylabel("Observed IBD rate")
    ax.grid(True, linestyle="--", alpha=0.45)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(
        frameon=False,
        loc="upper left",
        title=f"Brier={brier:.3f} | ECE={ece:.3f} | MCE={mce:.3f} | n={int(table['count'].sum())}",
    )
    fig.tight_layout()
    _save_multi(fig, out_dir / f"calibration_{_sanitize(model)}")


def _per_model_calibration(
    frame: pd.DataFrame, truth_col: str, stage_dir: Path
) -> None:
    """Compute reliability tables, Brier/ECE/MCE and pooled comparisons."""
    if "Likelihood_of_IBD_1" not in frame.columns or "model_canon" not in frame.columns:
        return
    out_dir = ensure_dir(stage_dir / "calibration_per_model")
    for stale_name in (
        "calibration_table_Thinking.csv",
        "calibration_table_Non-thinking.csv",
    ):
        stale_path = out_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    work = frame[[truth_col, "Likelihood_of_IBD_1", "model_canon"]].copy()
    work[truth_col] = pd.to_numeric(work[truth_col], errors="coerce")
    work["p_hat"] = np.clip(
        pd.to_numeric(work["Likelihood_of_IBD_1"], errors="coerce") / 10.0,
        0.0,
        1.0,
    )
    work = work.dropna(subset=[truth_col, "p_hat"])
    if work.empty:
        return

    summary_rows = []
    for model, sub in work.groupby("model_canon", dropna=False):
        y = sub[truth_col].astype(int).to_numpy()
        p = sub["p_hat"].to_numpy()
        if len(y) == 0:
            continue
        table = _calibration_table(y, p)
        if table.empty:
            continue
        brier = float(np.mean((p - y) ** 2))
        weights = table["count"] / float(table["count"].sum())
        diffs = (table["obs_rate"] - table["mean_pred"]).abs()
        ece = float((diffs * weights).sum())
        mce = float(diffs.max())
        write_dataframe(table, out_dir / f"calibration_table_{_sanitize(model)}.csv")
        _plot_reliability(table, str(model), brier, ece, mce, out_dir)
        summary_rows.append(
            {
                "model": model,
                "n": len(y),
                "brier": brier,
                "ece": ece,
                "mce": mce,
                "positive_rate": float(y.mean()),
            }
        )
    if summary_rows:
        write_dataframe(
            pd.DataFrame(summary_rows).sort_values("brier"),
            out_dir / "calibration_summary_by_model.csv",
        )

    work["model_type"] = work["model_canon"].map(_model_pool_label)
    work = work.dropna(subset=["model_type"])
    if work.empty:
        return
    fig, ax = plt.subplots(figsize=(6.4, 4.0))
    ax.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="black",
        linewidth=1.0,
        alpha=0.7,
        label="Perfect",
    )
    minimal_labels: list[str] = []
    for model_type, sub in work.groupby("model_type"):
        y = sub[truth_col].astype(int).to_numpy()
        p = sub["p_hat"].to_numpy()
        table = _calibration_table(y, p)
        if table.empty:
            continue
        brier = float(np.mean((p - y) ** 2))
        short_label = str(model_type)
        style = _model_pool_line_style(model_type)
        minimal_labels.append(f"{short_label} (Brier={brier:.3f})")
        ax.plot(
            table["mean_pred"],
            table["obs_rate"],
            marker=style["marker"],
            linewidth=float(style["linewidth"]),
            color=str(style["color"]),
            linestyle=str(style["linestyle"]),
            label=f"{short_label} (Brier={brier:.3f})",
        )
    ax.axvline(0.5, color="#7F7F7F", linestyle=":", linewidth=1.0, alpha=0.9)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Observed IBD rate")
    ax.grid(True, linestyle="--", alpha=0.45)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.legend(frameon=False)
    fig.tight_layout()
    _save_multi(fig, out_dir / "calibration_thinking_vs_non")

    fig_min, ax_min = plt.subplots(figsize=(6.0, 3.8))
    ax_min.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="black",
        linewidth=1.0,
        alpha=0.7,
    )
    label_index = 0
    for model_type, sub in work.groupby("model_type"):
        y = sub[truth_col].astype(int).to_numpy()
        p = sub["p_hat"].to_numpy()
        table = _calibration_table(y, p)
        if table.empty:
            continue
        style = _model_pool_line_style(model_type)
        ax_min.plot(
            table["mean_pred"],
            table["obs_rate"],
            marker=style["marker"],
            linewidth=float(style["linewidth"]),
            color=str(style["color"]),
            linestyle=str(style["linestyle"]),
            label=minimal_labels[label_index],
        )
        label_index += 1
    ax_min.axvline(0.5, color="#7F7F7F", linestyle=":", linewidth=1.0, alpha=0.9)
    ax_min.set_xlim(0, 1)
    ax_min.set_ylim(0, 1)
    ax_min.set_xlabel("Predicted probability")
    ax_min.set_ylabel("Observed IBD rate")
    ax_min.grid(True, linestyle="--", alpha=0.45)
    for spine in ("top", "right"):
        ax_min.spines[spine].set_visible(False)
    ax_min.legend(frameon=False)
    fig_min.tight_layout()
    _save_multi(fig_min, out_dir / "calibration_thinking_vs_non_minimal")

    pooled = work[work["model_canon"].map(_model_pool_label).notna()].copy()
    if pooled.empty:
        return
    pooled["model_pool"] = pooled["model_canon"].map(_model_pool_label)
    pooled_rows = []
    fig2, ax2 = plt.subplots(figsize=(6.4, 4.0))
    ax2.plot(
        [0, 1],
        [0, 1],
        linestyle="--",
        color="black",
        linewidth=1.0,
        alpha=0.7,
        label="Perfect",
    )
    for pool, sub in pooled.groupby("model_pool"):
        y = sub[truth_col].astype(int).to_numpy()
        p = sub["p_hat"].to_numpy()
        table = _calibration_table(y, p)
        if table.empty:
            continue
        style = _model_pool_line_style(pool)
        brier = float(np.mean((p - y) ** 2))
        diffs = (table["obs_rate"] - table["mean_pred"]).abs()
        weights = table["count"] / float(table["count"].sum())
        ece = float((diffs * weights).sum())
        mce = float(diffs.max())
        pooled_rows.append(
            {
                "model_pool": pool,
                "n": int(len(y)),
                "brier": brier,
                "ece": ece,
                "mce": mce,
            }
        )
        write_dataframe(
            table, out_dir / f"calibration_table_{_sanitize(str(pool))}.csv"
        )
        ax2.plot(
            table["mean_pred"],
            table["obs_rate"],
            marker=style["marker"],
            linewidth=float(style["linewidth"]),
            color=str(style["color"]),
            linestyle=str(style["linestyle"]),
            label=str(pool),
        )
    if pooled_rows:
        write_dataframe(
            pd.DataFrame(pooled_rows)
            .assign(pool_order=lambda df: df["model_pool"].map(_model_pool_sort_key))
            .sort_values(["pool_order", "model_pool"])
            .drop(columns=["pool_order"]),
            out_dir / "calibration_summary_by_model_pool.csv",
        )
        ax2.axvline(0.5, color="#7F7F7F", linestyle=":", linewidth=1.0, alpha=0.9)
        ax2.set_xlim(0, 1)
        ax2.set_ylim(0, 1)
        ax2.set_xlabel("Predicted probability")
        ax2.set_ylabel("Observed IBD rate")
        ax2.grid(True, linestyle="--", alpha=0.45)
        for spine in ("top", "right"):
            ax2.spines[spine].set_visible(False)
        ax2.legend(frameon=False)
        fig2.tight_layout()
        _save_multi(fig2, out_dir / "calibration_model_pool")
    else:
        plt.close(fig2)
