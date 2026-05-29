"""Section builders for the narrative analysis stage."""

from __future__ import annotations

from pathlib import Path
from typing import Callable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from analytics_code._narrative_shared import (
    CUE_FAMILY_ORDER,
    HEURISTIC_THEME_ORDER,
    MIN_TOPIC_DOCUMENTS,
    N_TOPICS,
    SHOT_ORDER,
    TOP_EXAMPLES_PER_SLICE,
    TOPIC_MAX_DOCUMENTS_PER_OUTCOME,
    _comparison_filter,
    _comparison_levels,
    _cue_density_frame,
    _cue_heatmap_comparison,
    _diverging_bar,
    _fit_lda,
    _fit_nmf,
    _heatmap,
    _heuristic_theme_density,
    _ngram_counts,
    _plot_signed_bar,
    _safe_slug,
    _topic_shot_skew,
    _topic_table,
)
from analytics_code.common import ensure_dir, save_figure, write_dataframe


def _ngram_section(
    frame: pd.DataFrame,
    root: Path,
    title_col: str | None,
    feat_col: str | None,
) -> None:
    """Emit per-overall and per-model diverging n-gram plots for FP vs FN."""
    for text_col, tag in ((title_col, ""), (feat_col, "_features")):
        if text_col is None:
            continue
        text_label = "title text" if tag == "" else "feature summary"
        fp_texts = frame.loc[frame["_is_fp"], text_col].dropna().astype(str).tolist()
        fn_texts = frame.loc[frame["_is_fn"], text_col].dropna().astype(str).tolist()
        for ngram, label in ((1, "words"), (3, "trigrams"), (5, "5grams")):
            _diverging_bar(
                _ngram_counts(fp_texts, ngram),
                _ngram_counts(fn_texts, ngram),
                f"False-positive vs false-negative {label} - overall {text_label}",
                root / f"diverging_{label}_overall{tag}.png",
            )
        if "model_canon" not in frame.columns:
            continue
        for model, sub in frame.dropna(subset=["model_canon"]).groupby("model_canon"):
            slug = _safe_slug(model)
            for ngram, label in ((1, "words"), (3, "trigrams"), (5, "5grams")):
                _diverging_bar(
                    _ngram_counts(
                        sub.loc[sub["_is_fp"], text_col].dropna().astype(str), ngram
                    ),
                    _ngram_counts(
                        sub.loc[sub["_is_fn"], text_col].dropna().astype(str), ngram
                    ),
                    f"False-positive vs false-negative {label} - {model} {text_label}",
                    root / f"diverging_{label}_{slug}{tag}.png",
                )


def _plot_topic_skew(topic_df: pd.DataFrame, title: str, out_path: Path) -> None:
    """Render a horizontal bar of ``skew_score`` per topic."""
    if topic_df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.4, 3.6))
    ordered = topic_df.sort_values("skew_score", ascending=True)
    colours = [
        "#D32F2F" if score > 0.1 else "#1E88E5" if score < -0.1 else "#888"
        for score in ordered["skew_score"]
    ]
    ax.barh(ordered["topic"].astype(str), ordered["skew_score"], color=colours)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Relative skew toward false positives or false negatives")
    ax.set_title(title)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    save_figure(out_path)


def _topic_section(
    frame: pd.DataFrame,
    root: Path,
    text_col: str,
    truth_col: str,
    *,
    label: str,
) -> None:
    """Fit NMF and LDA topic models and write skew tables/plots under ``root/label``."""
    del truth_col
    out_dir = ensure_dir(root / label)
    fp_idx = frame.index[frame["_is_fp"]]
    fn_idx = frame.index[frame["_is_fn"]]
    if min(len(fp_idx), len(fn_idx)) < MIN_TOPIC_DOCUMENTS:
        return
    if len(fp_idx) == 0 and len(fn_idx) == 0:
        return
    if len(fp_idx) > TOPIC_MAX_DOCUMENTS_PER_OUTCOME:
        fp_idx = pd.Index(
            pd.Series(fp_idx, copy=False).sample(
                n=TOPIC_MAX_DOCUMENTS_PER_OUTCOME,
                random_state=42,
            )
        )
    if len(fn_idx) > TOPIC_MAX_DOCUMENTS_PER_OUTCOME:
        fn_idx = pd.Index(
            pd.Series(fn_idx, copy=False).sample(
                n=TOPIC_MAX_DOCUMENTS_PER_OUTCOME,
                random_state=42,
            )
        )
    combined_idx = list(fp_idx) + list(fn_idx)
    texts = frame.loc[combined_idx, text_col].fillna("").astype(str).tolist()
    fp_mask = np.array([index in set(fp_idx) for index in combined_idx])
    fn_mask = ~fp_mask

    for name, fitter in (("nmf", _fit_nmf), ("lda", _fit_lda)):
        result = fitter(texts, N_TOPICS)
        if result is None:
            continue
        weights, components, vocab = result
        topic_df = _topic_table(weights, components, vocab, fp_mask, fn_mask)
        topic_df["false_positive_topic_share"] = topic_df["fp_weight"]
        topic_df["false_negative_topic_share"] = topic_df["fn_weight"]
        topic_df["false_positive_minus_false_negative_skew"] = topic_df["skew_score"]
        topic_df["top_topic_terms"] = topic_df["top_terms"]
        write_dataframe(topic_df, out_dir / f"topic_skew_{name}.csv")
        _plot_topic_skew(
            topic_df,
            f"Topic skew: false positive vs false negative - {name.upper()} - {label.replace('_', ' ')}",
            out_dir / f"topic_skew_{name}.png",
        )


def _heuristic_theme_comparison(
    frame: pd.DataFrame,
    theme_df: pd.DataFrame,
    root: Path,
    *,
    group_col: str,
    out_dir_name: str,
    prefix: str,
    display_name: str,
    preferred_order: list[object] | None = None,
    formatter: Callable[[object], str] | None = None,
) -> None:
    """Emit heuristic-theme delta heatmaps grouped by one comparison factor."""
    if group_col not in frame.columns:
        return
    filtered = _comparison_filter(frame, group_col)
    if filtered.empty or filtered[group_col].dropna().nunique() <= 1:
        return
    filtered_theme = theme_df.loc[filtered.index]
    groups = _comparison_levels(
        filtered[group_col], preferred_order=preferred_order, formatter=formatter
    )
    if len(groups) <= 1:
        return

    out_dir = ensure_dir(root / out_dir_name)
    case_specs = {
        "fp": {"case": filtered["_is_fp"], "ref": filtered["_is_tn"]},
        "fn": {"case": filtered["_is_fn"], "ref": filtered["_is_tp"]},
        "catastrophic": {
            "case": filtered["_catastrophic"],
            "ref": filtered["_is_error"] & ~filtered["_catastrophic"],
        },
    }
    for case_label, masks in case_specs.items():
        rows: dict[str, pd.Series] = {}
        for raw_value, label_value in groups:
            group_mask = filtered[group_col] == raw_value
            case_mask = masks["case"] & group_mask
            ref_mask = masks["ref"] & group_mask
            if case_mask.sum() == 0 or ref_mask.sum() == 0:
                continue
            rows[label_value] = (
                filtered_theme.loc[case_mask].mean()
                - filtered_theme.loc[ref_mask].mean()
            )
        if not rows:
            continue
        matrix = pd.DataFrame(rows).reindex(HEURISTIC_THEME_ORDER)
        write_dataframe(
            matrix.reset_index().rename(columns={"index": "theme"}),
            out_dir / f"{prefix}_heuristic_theme_deltas_{case_label}.csv",
        )
        _heatmap(
            matrix,
            title=f"Heuristic theme deltas by {display_name} - {case_label.replace('_', ' ')}",
            out_path=out_dir / f"{prefix}_heuristic_theme_heatmap_{case_label}.png",
            cbar_label="Theme density delta (per 100 tokens)",
            fmt="{:+.2f}",
        )


def _heuristic_theme_section(
    frame: pd.DataFrame,
    root: Path,
    text_col: str,
    *,
    label: str,
) -> None:
    """Write heuristic theme tables and heatmaps for the supplied narrative text."""
    out_dir = ensure_dir(root / label)
    theme_df = pd.DataFrame(
        [
            _heuristic_theme_density(text)
            for text in frame[text_col].fillna("").astype(str)
        ],
        index=frame.index,
        columns=HEURISTIC_THEME_ORDER,
    )
    if theme_df.empty:
        return

    outcome_masks = {
        "False positive": frame["_is_fp"],
        "False negative": frame["_is_fn"],
        "True positive": frame["_is_tp"],
        "True negative": frame["_is_tn"],
        "Catastrophic": frame["_catastrophic"],
    }
    prevalence_rows: list[dict[str, object]] = []
    prevalence_matrix: dict[str, pd.Series] = {}
    for outcome, mask in outcome_masks.items():
        if mask.sum() == 0:
            continue
        mean_density = theme_df.loc[mask].mean()
        prevalence_matrix[outcome] = mean_density
        for theme, value in mean_density.items():
            prevalence_rows.append(
                {
                    "outcome_group": outcome,
                    "theme": theme,
                    "mean_density_per_100_tokens": float(value),
                    "n_documents": int(mask.sum()),
                }
            )
    if prevalence_rows:
        write_dataframe(
            pd.DataFrame(prevalence_rows), out_dir / "theme_prevalence_by_outcome.csv"
        )
    if prevalence_matrix:
        prevalence_heatmap = pd.DataFrame(prevalence_matrix).reindex(
            HEURISTIC_THEME_ORDER
        )
        write_dataframe(
            prevalence_heatmap.reset_index().rename(columns={"index": "theme"}),
            out_dir / "theme_prevalence_heatmap.csv",
        )
        _heatmap(
            prevalence_heatmap,
            title=f"Heuristic theme prevalence - {label.replace('_', ' ')}",
            out_path=out_dir / "theme_prevalence_heatmap.png",
            cbar_label="Mean mentions per 100 tokens",
            fmt="{:.2f}",
        )

    if frame["_is_fp"].sum() and frame["_is_fn"].sum():
        fp_mean = theme_df.loc[frame["_is_fp"]].mean()
        fn_mean = theme_df.loc[frame["_is_fn"]].mean()
        skew = pd.DataFrame(
            {
                "theme": HEURISTIC_THEME_ORDER,
                "fp_weight": fp_mean.reindex(HEURISTIC_THEME_ORDER).to_numpy(
                    dtype=float
                ),
                "fn_weight": fn_mean.reindex(HEURISTIC_THEME_ORDER).to_numpy(
                    dtype=float
                ),
            }
        )
        total = skew["fp_weight"] + skew["fn_weight"]
        with np.errstate(divide="ignore", invalid="ignore"):
            skew["skew_score"] = np.where(
                total > 0,
                (skew["fp_weight"] - skew["fn_weight"]) / total,
                0.0,
            )
        skew["false_positive_theme_share"] = skew["fp_weight"]
        skew["false_negative_theme_share"] = skew["fn_weight"]
        skew["false_positive_minus_false_negative_skew"] = skew["skew_score"]
        write_dataframe(skew, out_dir / "heuristic_theme_skew.csv")
        _plot_signed_bar(
            skew,
            label_col="theme",
            value_col="skew_score",
            title=f"Heuristic theme skew: false positive vs false negative - {label.replace('_', ' ')}",
            xlabel="Relative skew toward false positives or false negatives",
            out_path=out_dir / "heuristic_theme_skew.png",
        )

    if "model_canon" in frame.columns:
        _heuristic_theme_comparison(
            frame,
            theme_df,
            root,
            group_col="model_canon",
            out_dir_name=f"{label}_model_comparison",
            prefix=label,
            display_name="model",
        )
    if "shot_type" in frame.columns and frame["shot_type"].dropna().nunique() > 1:
        _heuristic_theme_comparison(
            frame,
            theme_df,
            root,
            group_col="shot_type",
            out_dir_name=f"{label}_shot_comparison",
            prefix=label,
            display_name="shot type",
            preferred_order=SHOT_ORDER,
            formatter=str,
        )


def _plot_certainty_table(df: pd.DataFrame, out_path: Path) -> None:
    """Plot per-model high vs low certainty error rates as paired bars."""
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, max(2.8, 0.45 * len(df) + 1.2)))
    y = np.arange(len(df))
    ax.barh(
        y - 0.2,
        df["high_certainty_error_rate"],
        height=0.4,
        color="#D32F2F",
        label="High certainty: error rate",
    )
    ax.barh(
        y + 0.2,
        df["low_certainty_error_rate"],
        height=0.4,
        color="#1E88E5",
        label="Lower certainty: error rate",
    )
    ax.set_yticks(y)
    ax.set_yticklabels(df["model"].astype(str), fontsize=9)
    ax.set_xlabel("Error rate")
    ax.set_title("Per-model error rate by certainty band")
    ax.legend(frameon=False, fontsize=8)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    save_figure(out_path)


def _plot_catastrophic_rates(df: pd.DataFrame, out_path: Path) -> None:
    """Plot the catastrophic failure rate per model."""
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(6.4, max(2.8, 0.4 * len(df) + 1.2)))
    ax.barh(df["model"].astype(str), df["catastrophic_failure_rate"], color="#7B1FA2")
    ax.set_xlabel("Catastrophic failure rate")
    ax.set_title("Catastrophic failure rate by model")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    save_figure(out_path)


def _model_error_profiles(
    frame: pd.DataFrame,
    root: Path,
    feat_col: str,
    truth_col: str,
) -> None:
    """Write per-model example tables and certainty / catastrophic summaries."""
    out_dir = ensure_dir(root / "model_error_profiles")
    cert_rows: list[dict[str, object]] = []
    cat_rows: list[dict[str, object]] = []

    for model, sub in frame.dropna(subset=["model_canon"]).groupby("model_canon"):
        slug = _safe_slug(model)
        mdir = ensure_dir(out_dir / slug)
        for label, mask_col in (
            ("fp", "_is_fp"),
            ("fn", "_is_fn"),
            ("catastrophic", "_catastrophic"),
        ):
            cases = sub.loc[sub[mask_col]]
            if cases.empty:
                continue
            cols = [
                col
                for col in (
                    feat_col,
                    truth_col,
                    "_pred",
                    "likelihood_score",
                    "certainty_score",
                    "shot_type",
                )
                if col in cases.columns
            ]
            cases[cols].head(TOP_EXAMPLES_PER_SLICE).to_csv(
                mdir / f"top_{label}_examples.csv", index=False
            )

        n_high = int(sub["_high_cert"].sum())
        n_low = int(sub["_low_cert"].sum())
        high_err = (
            float(sub.loc[sub["_high_cert"], "_is_error"].mean())
            if n_high
            else float("nan")
        )
        low_err = (
            float(sub.loc[sub["_low_cert"], "_is_error"].mean())
            if n_low
            else float("nan")
        )
        cert_rows.append(
            {
                "model": model,
                "n": int(len(sub)),
                "n_high_certainty": n_high,
                "n_low_certainty": n_low,
                "high_certainty_error_rate": high_err,
                "low_certainty_error_rate": low_err,
                "high_minus_low_error_rate": (high_err - low_err)
                if (n_high and n_low)
                else float("nan"),
                "high_certainty_minus_low_certainty_error_rate": (high_err - low_err)
                if (n_high and n_low)
                else float("nan"),
            }
        )
        cat_rows.append(
            {
                "model": model,
                "n": int(len(sub)),
                "n_catastrophic": int(sub["_catastrophic"].sum()),
                "catastrophic_failure_rate": float(sub["_catastrophic"].mean()),
            }
        )

    if cert_rows:
        cert_df = pd.DataFrame(cert_rows).sort_values("model")
        write_dataframe(cert_df, out_dir / "certainty_error_rates.csv")
        _plot_certainty_table(cert_df, out_dir / "certainty_error_rates.png")
    if cat_rows:
        cat_df = pd.DataFrame(cat_rows).sort_values("model")
        write_dataframe(cat_df, out_dir / "catastrophic_failure_rates.csv")
        _plot_catastrophic_rates(cat_df, out_dir / "catastrophic_failure_rates.png")


def _mechanism_summary(
    per_case: dict[str, pd.Series], n_top: int = 2
) -> dict[str, str]:
    """Top overweighted and underweighted cue families per case label."""
    summary: dict[str, str] = {}
    for case_label, deltas in per_case.items():
        positive = deltas.sort_values(ascending=False)
        negative = deltas.sort_values(ascending=True)
        over = [name for name, value in positive.items() if value > 0][:n_top]
        under = [name for name, value in negative.items() if value < 0][:n_top]
        over_text = "; ".join(over) if over else "none"
        under_text = "; ".join(under) if under else "none"
        summary[case_label] = f"Overweights: {over_text} | Underweights: {under_text}"
    return summary


def _reasoning_audit(frame: pd.DataFrame, root: Path, feat_col: str) -> None:
    """Write per-model reasoning cue deltas plus heatmaps and mechanism summaries."""
    audit_dir = ensure_dir(root / "reasoning_audit")
    cues = _cue_density_frame(frame[feat_col])
    case_specs = {
        "fp": {"case": frame["_is_fp"], "ref": frame["_is_tn"]},
        "fn": {"case": frame["_is_fn"], "ref": frame["_is_tp"]},
        "catastrophic": {
            "case": frame["_catastrophic"],
            "ref": frame["_is_error"] & ~frame["_catastrophic"],
        },
    }

    delta_rows: list[dict[str, object]] = []
    mechanism_rows: dict[str, dict[str, str]] = {}

    for model, sub in frame.dropna(subset=["model_canon"]).groupby("model_canon"):
        slug = _safe_slug(model)
        mdir = ensure_dir(audit_dir / slug)
        sub_cues = cues.loc[sub.index]
        per_case: dict[str, pd.Series] = {}
        for case_label, masks in case_specs.items():
            case_mask = masks["case"].loc[sub.index]
            ref_mask = masks["ref"].loc[sub.index]
            if case_mask.sum() == 0 or ref_mask.sum() == 0:
                continue
            case_mean = sub_cues.loc[case_mask].mean()
            ref_mean = sub_cues.loc[ref_mask].mean()
            delta = case_mean - ref_mean
            per_case[case_label] = delta
            for family, value in delta.items():
                delta_rows.append(
                    {
                        "model": model,
                        "case": case_label,
                        "cue_family": family,
                        "case_mean": float(case_mean[family]),
                        "reference_mean": float(ref_mean[family]),
                        "delta": float(value),
                    }
                )
        if per_case:
            pd.DataFrame(per_case).rename_axis("cue_family").to_csv(
                mdir / "cue_deltas.csv"
            )
        mechanism_rows[model] = _mechanism_summary(per_case)

    if not delta_rows:
        return

    delta_df = pd.DataFrame(delta_rows)
    write_dataframe(delta_df, audit_dir / "reasoning_cue_deltas_all_models.csv")
    for case_label, fig_title in (
        ("fp", "False positive vs true negative"),
        ("fn", "False negative vs true positive"),
        ("catastrophic", "Catastrophic failure vs other errors"),
    ):
        case_df = delta_df[delta_df["case"] == case_label]
        if case_df.empty:
            continue
        matrix = (
            case_df.pivot(index="cue_family", columns="model", values="delta")
            .reindex(CUE_FAMILY_ORDER)
            .dropna(axis=1, how="all")
        )
        if matrix.empty:
            continue
        write_dataframe(
            matrix.reset_index(), audit_dir / f"{case_label}_cue_heatmap_matrix.csv"
        )
        _heatmap(
            matrix,
            title=f"Reasoning cue deltas - {fig_title}",
            out_path=audit_dir / f"{case_label}_cue_heatmap.png",
            cbar_label="Cue density delta (per 100 tokens)",
        )

    if mechanism_rows:
        mech_df = pd.DataFrame(
            [
                {
                    "model": model,
                    "dominant_fp": entries.get("fp", ""),
                    "dominant_fn": entries.get("fn", ""),
                    "dominant_catastrophic": entries.get("catastrophic", ""),
                }
                for model, entries in mechanism_rows.items()
            ]
        ).sort_values("model")
        write_dataframe(mech_df, audit_dir / "narrative_error_mechanisms.csv")


def _shot_comparison(frame: pd.DataFrame, root: Path, feat_col: str) -> None:
    """Emit shot-type comparison heatmaps for cue families and topic skews."""
    out_dir = ensure_dir(root / "shot_comparison")
    _cue_heatmap_comparison(
        frame,
        root,
        feat_col,
        group_col="shot_type",
        out_dir_name="shot_comparison",
        prefix="shot",
        display_name="shot type",
        preferred_order=SHOT_ORDER,
        formatter=str,
    )

    filtered = _comparison_filter(frame, "shot_type")
    if filtered.empty:
        return
    shot = filtered["shot_type"].astype(str)
    fp_idx = filtered.index[filtered["_is_fp"]]
    fn_idx = filtered.index[filtered["_is_fn"]]
    combined_idx = list(fp_idx) + list(fn_idx)
    if not combined_idx:
        return
    texts = filtered.loc[combined_idx, feat_col].fillna("").astype(str).tolist()
    shot_combined = shot.loc[combined_idx]
    fp_mask = np.array([index in set(fp_idx) for index in combined_idx])
    fn_mask = ~fp_mask

    for name, fitter in (("nmf", _fit_nmf), ("lda", _fit_lda)):
        result = fitter(texts, N_TOPICS)
        if result is None:
            continue
        weights, _, _ = result
        skew_df = _topic_shot_skew(
            weights, fp_mask, fn_mask, shot_combined.reset_index(drop=True)
        )
        skew_df = skew_df.loc[:, [col for col in SHOT_ORDER if col in skew_df.columns]]
        if skew_df.empty:
            continue
        write_dataframe(skew_df.reset_index(), out_dir / f"shot_topic_skew_{name}.csv")
        _heatmap(
            skew_df,
            title=f"Topic skew: false positive vs false negative by shot type - {name.upper()}",
            out_path=out_dir / f"shot_topic_skew_heatmap_{name}.png",
            cbar_label="Relative skew toward false positives or false negatives",
        )
