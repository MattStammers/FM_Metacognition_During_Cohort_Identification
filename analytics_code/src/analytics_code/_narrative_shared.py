"""Shared helpers for the narrative analysis stage."""

from __future__ import annotations

import re
from collections import Counter
from pathlib import Path
from typing import Callable, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.colors import TwoSlopeNorm
from sklearn.decomposition import NMF, LatentDirichletAllocation
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

from analytics_code.common import (
    FAIR_DATA_TYPES,
    FAIR_TEMPERATURES_NUMERIC,
    ZERO_SHOT_LABEL,
    ensure_dir,
    format_factor_value,
    format_model_display_name,
    format_shot_label,
    format_temperature_label,
    save_figure,
    sentence_case,
    set_publication_axes,
    write_dataframe,
)

TRUTH_CANDIDATES = ("Patient_Has_IBD", "ground_truth", "label")
WORD_PATTERN = re.compile(r"[^a-zA-Z\s]")

# Decision threshold lives in :mod:`analytics_code.predictions` so every
# stage (full_performance, narrative_analysis, missingness_threshold)
# agrees on the cut-off used to derive a binary prediction from the
# ``0..10`` likelihood scale. Re-exported here for compatibility with
# modules that already imported it from ``_narrative_shared``.
from analytics_code.predictions import DECISION_THRESHOLD  # noqa: E402

HIGH_CERTAINTY_THRESHOLD = 7
EXTREME_LIKELIHOOD_LOW = 2
EXTREME_LIKELIHOOD_HIGH = 8
N_TOPICS = 8
TOP_TERMS_PER_TOPIC = 10
# Methodology: |skew| > 0.10 is used as a pragmatic exploratory signal
# (~10 percentage point difference in cue-family prevalence between
# erroneous and matched correct outputs). It is NOT a statistical
# significance threshold.
SKEW_EXPLORATORY_THRESHOLD = 0.10
MIN_TOPIC_DOCUMENTS = 25
TOP_EXAMPLES_PER_SLICE = 8
TOPIC_MAX_DOCUMENTS_PER_OUTCOME = 3000
TOPIC_MAX_FEATURES = 1000
TOPIC_MAX_ITER_NMF = 100
TOPIC_MAX_ITER_LDA = 10

CUE_FAMILIES: dict[str, list[str]] = {
    "chronic_history": [
        "history",
        "known crohn",
        "known uc",
        "known ulcerative colitis",
        "known ibd",
        "previous",
        "prior",
        "established",
        "longstanding",
        "chronic",
        "ileocolonic",
        "crohn",
        "ulcerative colitis",
    ],
    "active_inflammation": [
        "flare",
        "active",
        "inflammation",
        "inflammatory",
        "colitis",
        "ulceration",
        "erosions",
        "raised crp",
        "faecal calprotectin",
        "fecal calprotectin",
        "diarrhoea",
        "diarrhea",
        "abdominal pain",
        "bleeding",
        "mucosal",
    ],
    "reassuring_negative": [
        "normal",
        "no evidence",
        "negative",
        "unremarkable",
        "stable",
        "well",
        "resolved",
        "benign",
        "no active",
        "no inflammation",
        "quiescent",
        "in remission",
    ],
    "procedure_surveillance": [
        "endoscopy",
        "colonoscopy",
        "scope",
        "biopsy",
        "histology",
        "ileoscopy",
        "terminal ileum",
        "surveillance",
        "follow up",
        "follow-up",
        "clinic",
        "review",
        "repeat",
    ],
    "treatment_escalation": [
        "steroids",
        "prednisolone",
        "biologic",
        "infliximab",
        "adalimumab",
        "vedolizumab",
        "ustekinumab",
        "mesalazine",
        "azathioprine",
        "treatment",
        "therapy",
        "response",
    ],
    "alternative_non_ibd": [
        "infection",
        "infective",
        "diverticular",
        "diverticulitis",
        "ischaemic",
        "ischemic",
        "ibs",
        "functional",
        "haemorrhoids",
        "hemorrhoids",
        "bile acid",
        "post operative",
        "post-op",
        "adhesions",
    ],
    "uncertainty_hedging": [
        "possible",
        "suggest",
        "may",
        "could",
        "unclear",
        "equivocal",
        "likely",
        "unlikely",
        "consider",
        "query",
        "?",
        "perhaps",
    ],
}
CUE_FAMILY_ORDER = list(CUE_FAMILIES)
SHOT_ORDER = ["zero", "single", "dual"]
TEMPERATURE_ORDER = [f"{temp:.2f}" for temp in FAIR_TEMPERATURES_NUMERIC]
CONTEXT_ORDER = list(FAIR_DATA_TYPES)
HEURISTIC_THEMES: dict[str, list[str]] = {
    "Procedure anchoring": [
        "endoscopy",
        "colonoscopy",
        "scope",
        "biopsy",
        "histology",
        "terminal ileum",
        "ileoscopy",
        "mucosa",
    ],
    "Symptom/inflammation anchoring": [
        "flare",
        "abdominal pain",
        "diarrhoea",
        "diarrhea",
        "crp",
        "inflammatory",
        "colitis",
        "active disease",
        "raised",
    ],
    "Reassurance/negation overweighting": [
        "normal",
        "no evidence",
        "unremarkable",
        "negative",
        "stable",
        "well",
        "resolved",
        "benign",
    ],
    "History underweighting": [
        "history",
        "known crohn",
        "known uc",
        "ibd history",
        "previous",
        "prior",
        "chronic",
        "established",
    ],
    "Follow-up/surveillance framing": [
        "follow up",
        "review",
        "monitoring",
        "surveillance",
        "clinic",
        "outpatient",
        "plan",
        "repeat",
    ],
    "Treatment-response anchoring": [
        "steroids",
        "biologic",
        "infliximab",
        "adalimumab",
        "mesalazine",
        "prednisolone",
        "response",
        "improved",
    ],
    "Post-operative or alternative-cause confusion": [
        "post operative",
        "post-op",
        "surgery",
        "adhesions",
        "infection",
        "diverticular",
        "ischaemic",
        "functional",
    ],
}
HEURISTIC_THEME_ORDER = list(HEURISTIC_THEMES)


def _detect_truth(frame: pd.DataFrame) -> str | None:
    """Return the first ground-truth column present in ``frame``."""
    for col in TRUTH_CANDIDATES:
        if col in frame.columns:
            return col
    return None


def _safe_slug(value: object) -> str:
    """Return a filesystem-safe slug derived from ``value``."""
    return re.sub(r"[^A-Za-z0-9._-]+", "_", str(value)).strip("_")


def _tokenize(text: str) -> list[str]:
    """Lowercase ``text`` and return its alphabetic word tokens."""
    if not isinstance(text, str):
        return []
    return WORD_PATTERN.sub(" ", text.lower()).split()


def _ngram_counts(texts: Iterable[str], n: int, top_k: int = 30) -> pd.DataFrame:
    """Return the ``top_k`` most frequent ``n``-grams across ``texts``."""
    counter: Counter[str] = Counter()
    for text in texts:
        tokens = _tokenize(text)
        if len(tokens) < n:
            continue
        for index in range(len(tokens) - n + 1):
            counter[" ".join(tokens[index : index + n])] += 1
    if not counter:
        return pd.DataFrame(columns=["term", "count"])
    return pd.DataFrame(counter.most_common(top_k), columns=["term", "count"])


def _regex_for_keyword(keyword: str) -> re.Pattern[str]:
    """Compile a whole-word regex for a multi-token cue keyword."""
    parts = [re.escape(part) for part in str(keyword).lower().split()]
    return (
        re.compile(r"\b" + r"\s+".join(parts) + r"\b") if parts else re.compile(r"$^")
    )


_CUE_PATTERNS = {
    family: [_regex_for_keyword(kw) for kw in keywords]
    for family, keywords in CUE_FAMILIES.items()
}
_HEURISTIC_THEME_PATTERNS = {
    theme: [_regex_for_keyword(kw) for kw in keywords]
    for theme, keywords in HEURISTIC_THEMES.items()
}


def _cue_density(text: str) -> dict[str, float]:
    """Return mentions per 100 word-tokens for each cue family."""
    if not isinstance(text, str) or not text.strip():
        return {family: 0.0 for family in CUE_FAMILY_ORDER}
    low = text.lower()
    n_tokens = max(len(re.findall(r"\b[a-z]{2,}\b", low)), 1)
    out: dict[str, float] = {}
    for family in CUE_FAMILY_ORDER:
        hits = sum(len(pattern.findall(low)) for pattern in _CUE_PATTERNS[family])
        out[family] = 100.0 * hits / n_tokens
    return out


def _heuristic_theme_density(text: str) -> dict[str, float]:
    """Return mentions per 100 word-tokens for each heuristic theme."""
    if not isinstance(text, str) or not text.strip():
        return {theme: 0.0 for theme in HEURISTIC_THEME_ORDER}
    low = text.lower()
    n_tokens = max(len(re.findall(r"\b[a-z]{2,}\b", low)), 1)
    out: dict[str, float] = {}
    for theme in HEURISTIC_THEME_ORDER:
        hits = sum(
            len(pattern.findall(low)) for pattern in _HEURISTIC_THEME_PATTERNS[theme]
        )
        out[theme] = 100.0 * hits / n_tokens
    return out


def _cue_density_frame(text_series: pd.Series) -> pd.DataFrame:
    """Apply :func:`_cue_density` to every entry of ``text_series``."""
    return pd.DataFrame(
        [_cue_density(text) for text in text_series.fillna("").astype(str)],
        index=text_series.index,
        columns=CUE_FAMILY_ORDER,
    )


def _diverging_bar(
    fp_counts: pd.DataFrame,
    fn_counts: pd.DataFrame,
    title: str,
    out_path: Path,
    top_k: int = 20,
) -> None:
    """Plot a left/right diverging bar chart of FP vs FN n-gram counts."""
    if fp_counts.empty and fn_counts.empty:
        return
    top_fp = fp_counts.head(top_k).set_index("term")["count"]
    top_fn = fn_counts.head(top_k).set_index("term")["count"]
    terms = list(dict.fromkeys(list(top_fp.index) + list(top_fn.index)))
    fp_vals = [-int(top_fp.get(term, 0)) for term in terms]
    fn_vals = [int(top_fn.get(term, 0)) for term in terms]
    fig, ax = plt.subplots(figsize=(7.5, max(3.5, 0.25 * len(terms) + 1.5)))
    y = np.arange(len(terms))
    ax.barh(y, fp_vals, color="#D32F2F", label="False positive")
    ax.barh(y, fn_vals, color="#1E88E5", label="False negative")
    ax.set_yticks(y)
    ax.set_yticklabels(terms, fontsize=8)
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel("Mentions (FP vs FN)")
    ax.set_title(title)
    ax.legend(frameon=False, fontsize=8, loc="lower right")
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    fig.tight_layout()
    save_figure(out_path)


def _heatmap(
    matrix: pd.DataFrame,
    *,
    title: str,
    out_path: Path,
    cbar_label: str,
    cmap: str = "RdBu_r",
    annotate: bool = True,
    fmt: str = "{:+.2f}",
    save_svg: bool = True,
) -> None:
    """Render a diverging heatmap (PNG + optional SVG) of ``matrix``."""
    if matrix.empty:
        return
    display_matrix = matrix.copy()
    display_matrix.index = [
        _display_heatmap_label(value) for value in display_matrix.index
    ]
    display_matrix.columns = [
        _display_heatmap_label(value) for value in display_matrix.columns
    ]
    data = display_matrix.to_numpy(dtype=float)
    abs_max = float(np.nanmax(np.abs(data))) if np.isfinite(data).any() else 0.0
    if abs_max == 0.0:
        abs_max = 1e-6
    norm = TwoSlopeNorm(vmin=-abs_max, vcenter=0.0, vmax=abs_max)
    fig_w = max(6.6, 1.15 * len(display_matrix.columns) + 2.4)
    fig_h = max(3.8, 0.58 * len(display_matrix.index) + 1.8)
    fig, ax = plt.subplots(figsize=(fig_w, fig_h))
    im = ax.imshow(data, aspect="auto", cmap=cmap, norm=norm)
    ax.set_xticks(np.arange(len(display_matrix.columns)))
    ax.set_xticklabels(display_matrix.columns, rotation=25, ha="right", fontsize=10)
    ax.set_yticks(np.arange(len(display_matrix.index)))
    ax.set_yticklabels(display_matrix.index, fontsize=10)
    if annotate:
        for row in range(data.shape[0]):
            for col in range(data.shape[1]):
                value = data[row, col]
                if not np.isfinite(value):
                    continue
                colour = "black" if abs(value) < 0.55 * abs_max else "white"
                ax.text(
                    col,
                    row,
                    fmt.format(value),
                    ha="center",
                    va="center",
                    fontsize=8.5,
                    color=colour,
                )
    ax.set_title(sentence_case(title))
    set_publication_axes(ax, show_grid_y=False, show_grid_x=False)
    fig.colorbar(im, ax=ax).set_label(sentence_case(cbar_label))
    save_figure(out_path)
    if not save_svg:
        return
    svg_path = out_path.with_suffix(".svg")
    fig2, ax2 = plt.subplots(figsize=(fig_w, fig_h))
    im2 = ax2.imshow(data, aspect="auto", cmap=cmap, norm=norm)
    ax2.set_xticks(np.arange(len(display_matrix.columns)))
    ax2.set_xticklabels(display_matrix.columns, rotation=25, ha="right", fontsize=10)
    ax2.set_yticks(np.arange(len(display_matrix.index)))
    ax2.set_yticklabels(display_matrix.index, fontsize=10)
    ax2.set_title(sentence_case(title))
    set_publication_axes(ax2, show_grid_y=False, show_grid_x=False)
    fig2.colorbar(im2, ax=ax2).set_label(sentence_case(cbar_label))
    save_figure(svg_path)


def _display_heatmap_label(value: object) -> str:
    """Return a readable label for heatmap rows and columns."""
    raw = str(value)
    low = raw.lower()
    if low in {"zero", "single", "dual"}:
        return format_shot_label(raw)
    if re.fullmatch(r"\d+(?:\.\d+)?", raw):
        return format_temperature_label(raw)
    if any(token in low for token in ("mixtral", "m42", "deepseek", "qwen")):
        return format_model_display_name(raw)
    if any(token in low for token in ("clinic", "endo", "hist", "sequence", "jumbled")):
        return format_factor_value("report_sequence_name", raw)
    return sentence_case(raw)


def _plot_signed_bar(
    df: pd.DataFrame,
    *,
    label_col: str,
    value_col: str,
    title: str,
    xlabel: str,
    out_path: Path,
) -> None:
    """Render a horizontal signed bar chart with symmetric colouring."""
    if df.empty:
        return
    fig, ax = plt.subplots(figsize=(7.0, max(3.4, 0.45 * len(df) + 1.4)))
    ordered = df.sort_values(value_col, ascending=True)
    colours = [
        "#D32F2F" if value > 0.1 else "#1E88E5" if value < -0.1 else "#888888"
        for value in ordered[value_col]
    ]
    ax.barh(
        ordered[label_col].map(sentence_case),
        ordered[value_col],
        color=colours,
        edgecolor="black",
        linewidth=0.9,
    )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_xlabel(sentence_case(xlabel))
    ax.set_title(sentence_case(title))
    set_publication_axes(ax, show_grid_y=False, show_grid_x=True)
    fig.tight_layout()
    save_figure(out_path)


def _fit_nmf(
    texts: list[str], n_topics: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Fit a TF-IDF NMF topic model; return ``(W, H, vocab)`` or ``None``."""
    if len(texts) < max(2, n_topics):
        return None
    min_df = 2 if len(texts) >= 100 else 1
    for max_df in (0.95, 1.0):
        try:
            vec = TfidfVectorizer(
                max_df=max_df,
                min_df=min_df,
                ngram_range=(1, 1),
                stop_words="english",
                max_features=TOPIC_MAX_FEATURES,
            )
            matrix = vec.fit_transform(texts)
            break
        except ValueError:
            continue
    else:
        return None
    n_topics_eff = min(n_topics, matrix.shape[0], matrix.shape[1])
    if n_topics_eff < 2:
        return None
    model = NMF(
        n_components=n_topics_eff,
        init="nndsvd",
        random_state=42,
        max_iter=TOPIC_MAX_ITER_NMF,
    )
    weights = model.fit_transform(matrix)
    return weights, model.components_, np.array(vec.get_feature_names_out())


def _fit_lda(
    texts: list[str], n_topics: int
) -> tuple[np.ndarray, np.ndarray, np.ndarray] | None:
    """Fit a count-vector LDA topic model; return ``(W, H, vocab)`` or ``None``."""
    if len(texts) < max(2, n_topics):
        return None
    min_df = 2 if len(texts) >= 100 else 1
    for max_df in (0.95, 1.0):
        try:
            vec = CountVectorizer(
                max_df=max_df,
                min_df=min_df,
                ngram_range=(1, 1),
                stop_words="english",
                max_features=TOPIC_MAX_FEATURES,
            )
            matrix = vec.fit_transform(texts)
            break
        except ValueError:
            continue
    else:
        return None
    n_topics_eff = min(n_topics, matrix.shape[0], matrix.shape[1])
    if n_topics_eff < 2:
        return None
    model = LatentDirichletAllocation(
        n_components=n_topics_eff,
        random_state=42,
        learning_method="batch",
        max_iter=TOPIC_MAX_ITER_LDA,
    )
    weights = model.fit_transform(matrix)
    return weights, model.components_, np.array(vec.get_feature_names_out())


_TOPIC_NAME = [
    "Zero",
    "One",
    "Two",
    "Three",
    "Four",
    "Five",
    "Six",
    "Seven",
    "Eight",
    "Nine",
    "Ten",
    "Eleven",
]


def _topic_table(
    weights: np.ndarray,
    components: np.ndarray,
    vocab: np.ndarray,
    fp_mask: np.ndarray,
    fn_mask: np.ndarray,
    *,
    top_terms: int = TOP_TERMS_PER_TOPIC,
) -> pd.DataFrame:
    """Build the per-topic skew table: FP/FN weight share and top vocabulary terms."""
    fp_w = weights[fp_mask].sum(axis=0)
    fn_w = weights[fn_mask].sum(axis=0)
    fp_share = fp_w / fp_w.sum() if fp_w.sum() > 0 else np.zeros_like(fp_w)
    fn_share = fn_w / fn_w.sum() if fn_w.sum() > 0 else np.zeros_like(fn_w)
    total = fp_share + fn_share
    with np.errstate(divide="ignore", invalid="ignore"):
        skew = np.where(total > 0, (fp_share - fn_share) / total, 0.0)
    rows = []
    for index in range(components.shape[0]):
        term_idx = np.argsort(components[index])[::-1][:top_terms]
        rows.append(
            {
                "topic": _TOPIC_NAME[index] if index < len(_TOPIC_NAME) else str(index),
                "topic_index": index,
                "fp_weight": float(fp_share[index]),
                "fn_weight": float(fn_share[index]),
                "skew_score": float(skew[index]),
                "abs_skew": float(abs(skew[index])),
                "exploratory_signal": bool(
                    abs(float(skew[index])) > SKEW_EXPLORATORY_THRESHOLD
                ),
                "top_terms": ", ".join(vocab[term_idx]),
            }
        )
    return (
        pd.DataFrame(rows)
        .sort_values("skew_score", ascending=False)
        .reset_index(drop=True)
    )


def _topic_shot_skew(
    weights: np.ndarray,
    fp_mask: np.ndarray,
    fn_mask: np.ndarray,
    shot: pd.Series,
) -> pd.DataFrame:
    """Per-shot skew (FP - FN) / (FP + FN) per topic, indexed by topic."""
    rows = {}
    for shot_label in SHOT_ORDER:
        shot_mask = (shot == shot_label).to_numpy()
        fp = weights[fp_mask & shot_mask].sum(axis=0)
        fn = weights[fn_mask & shot_mask].sum(axis=0)
        total = fp + fn
        with np.errstate(divide="ignore", invalid="ignore"):
            rows[shot_label] = np.where(total > 0, (fp - fn) / total, 0.0)
    out = pd.DataFrame(rows)
    out.index = [
        _TOPIC_NAME[index] if index < len(_TOPIC_NAME) else str(index)
        for index in range(weights.shape[1])
    ]
    out.index.name = "topic"
    return out


def _comparison_filter(frame: pd.DataFrame, factor: str) -> pd.DataFrame:
    """Apply the FAIR comparison filter for ``factor``."""
    fair_temps = {round(temp, 2) for temp in FAIR_TEMPERATURES_NUMERIC}
    fair_contexts = set(FAIR_DATA_TYPES)
    df = frame.copy()
    if factor == "shot_type":
        df = df[df["temperature"].isin(fair_temps)]
        df = df[df["report_sequence_name"].isin(fair_contexts)]
    elif factor == "temperature":
        df = df[df["shot_type"] == ZERO_SHOT_LABEL]
        df = df[df["report_sequence_name"].isin(fair_contexts)]
    elif factor == "report_sequence_name":
        df = df[df["shot_type"] == ZERO_SHOT_LABEL]
        df = df[df["temperature"].isin(fair_temps)]
    elif factor == "model_canon":
        df = df[df["shot_type"] == ZERO_SHOT_LABEL]
        df = df[df["temperature"].isin(fair_temps)]
        df = df[df["report_sequence_name"].isin(fair_contexts)]
    return df


def _comparison_levels(
    series: pd.Series,
    *,
    preferred_order: list[object] | None = None,
    formatter: Callable[[object], str] | None = None,
) -> list[tuple[object, str]]:
    """Return ordered ``(raw_value, label)`` pairs for a grouping column."""
    formatter = formatter or str
    values = [value for value in series.dropna().unique().tolist()]
    ordered: list[object] = []
    if preferred_order is not None:
        for value in preferred_order:
            if value in values:
                ordered.append(value)
    remaining = [value for value in values if value not in ordered]
    ordered.extend(sorted(remaining, key=lambda value: str(value)))
    return [(value, formatter(value)) for value in ordered]


def _cue_heatmap_comparison(
    frame: pd.DataFrame,
    root: Path,
    feat_col: str,
    *,
    group_col: str,
    out_dir_name: str,
    prefix: str,
    display_name: str,
    preferred_order: list[object] | None = None,
    formatter: Callable[[object], str] | None = None,
) -> None:
    """Emit cue-family failure-mode heatmaps grouped by one comparison factor."""
    if group_col not in frame.columns:
        return
    filtered = _comparison_filter(frame, group_col)
    if filtered.empty or filtered[group_col].dropna().nunique() <= 1:
        return

    out_dir = ensure_dir(root / out_dir_name)
    cues = _cue_density_frame(filtered[feat_col])
    groups = _comparison_levels(
        filtered[group_col], preferred_order=preferred_order, formatter=formatter
    )
    if len(groups) <= 1:
        return

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
        for raw_value, label in groups:
            group_mask = filtered[group_col] == raw_value
            case_mask = masks["case"] & group_mask
            ref_mask = masks["ref"] & group_mask
            if case_mask.sum() == 0 or ref_mask.sum() == 0:
                continue
            rows[label] = cues.loc[case_mask].mean() - cues.loc[ref_mask].mean()
        if not rows:
            continue
        matrix = pd.DataFrame(rows).reindex(CUE_FAMILY_ORDER)
        write_dataframe(
            matrix.reset_index().rename(columns={"index": "cue_family"}),
            out_dir / f"{prefix}_cue_deltas_{case_label}.csv",
        )
        _heatmap(
            matrix,
            title=f"Cue deltas by {display_name} - {case_label.upper()}",
            out_path=out_dir / f"{prefix}_cue_heatmap_{case_label}.png",
            cbar_label="Cue density delta (per 100 tokens)",
        )


def _full_context_column(frame: pd.DataFrame) -> str | None:
    """Pick the first available full-text context column, or ``None``."""
    for candidate in (
        "Combined_Content",
        "full_response",
        "reasoning_text",
        "clues_text",
        "full_context",
        "context_text",
    ):
        if candidate in frame.columns:
            return candidate
    return None
