"""Narrative analysis stage.

This code produces:

- ``narrative_exploration/diverging_{words,trigrams,5grams}_{model|overall}{_features}.png``
  N-gram contrasts of false-positive vs false-negative ``Title`` and ``Features``.
- ``narrative_exploration/themes/topic_skew_{nmf,lda}.csv`` and ``.png``
  NMF and LDA topic models fit on the shared FP+FN corpus, with skew
  scores ``(FP - FN) / (FP + FN)`` and the top terms per topic.
- ``narrative_exploration/heuristic_themes/`` and
    ``narrative_exploration/heuristic_themes_full_context/``
    Mechanistic keyword-theme summaries, skew tables and comparison
    heatmaps for the narrative error analysis.
- ``narrative_exploration/themes_full_context/`` — same models on the full
  clinical context.
- ``narrative_exploration/model_error_profiles/``
  - ``certainty_error_rates.csv``: per-model high vs low certainty error
    rates and their differences.
  - ``catastrophic_failure_rates.csv``: per-model catastrophic failure
    rate.
  - ``narrative_error_mechanisms.csv``: per-model dominant FP, FN, and
    catastrophic failure cue families.
  - per-model ``top_{fp,fn,catastrophic}_examples.csv``.
- ``narrative_exploration/reasoning_audit/``
  - per-model ``cue_deltas.csv`` for FP/FN/catastrophic vs reference.
  - cross-model heatmaps ``{fp,fn,catastrophic}_cue_heatmap.{png,svg}``
- ``narrative_exploration/shot_comparison/``
    - cue heatmaps (cue family x shot type) for FP, FN, catastrophic.
    - topic skew heatmaps (topic x shot type) for NMF and LDA.
- ``narrative_exploration/{temperature,clinical_context,model}_comparison/``
    - cue heatmaps (cue family x comparison factor) for FP, FN,
        catastrophic, using the same FAIR-style controls as the other
        comparison stages.

The pipeline depends only on columns produced by ``data_prep``:
``model_canon``, ``shot_type``, ``likelihood_score``, ``certainty_score``,
``Title``, ``Features``, and a ground-truth column.
"""

from __future__ import annotations

import logging
import shutil
from pathlib import Path

import pandas as pd

from analytics_code._narrative_sections import (
    _heuristic_theme_section,
    _model_error_profiles,
    _ngram_section,
    _reasoning_audit,
    _shot_comparison,
    _topic_section,
)
from analytics_code._narrative_shared import (
    CONTEXT_ORDER,
    DECISION_THRESHOLD,
    EXTREME_LIKELIHOOD_HIGH,
    EXTREME_LIKELIHOOD_LOW,
    HIGH_CERTAINTY_THRESHOLD,
    _comparison_filter,
    _cue_heatmap_comparison,
    _detect_truth,
    _full_context_column,
)
from analytics_code.common import (
    FAIR_TEMPERATURES_NUMERIC,
    ensure_dir,
    remove_tree,
    write_dataframe,
)
from analytics_code.config import AnalysisConfig
from analytics_code.predictions import derive_prediction
from analytics_code.truth_labels import prepare_truth_frame, truth_mode

LOGGER = logging.getLogger("analytics_code.narrative_analysis")


def _clear_stage_outputs(stage_dir: Path) -> None:
    """Remove stale narrative artefacts when the stage is skipped."""
    if stage_dir.exists():
        remove_tree(stage_dir)
    ensure_dir(stage_dir)


def _copy_tree_contents(source: Path, destination: Path) -> None:
    """Copy every file under ``source`` into ``destination`` preserving structure."""
    if not source.exists():
        return
    for path in source.rglob("*"):
        if path.is_dir():
            continue
        target = destination / path.relative_to(source)
        ensure_dir(target.parent)
        shutil.copy2(path, target)


def _copy_with_csv_header_aliases(
    source: Path,
    destination: Path,
    *,
    rename_map: dict[str, str] | None = None,
) -> None:
    """Copy ``source`` to ``destination``, renaming CSV headers when requested."""
    ensure_dir(destination.parent)
    if source.suffix.lower() == ".csv" and rename_map:
        dataframe = pd.read_csv(source)
        write_dataframe(dataframe.rename(columns=rename_map), destination)
        return
    shutil.copy2(source, destination)


def _mirror_narrative_outputs(stage_dir: Path) -> None:
    """Create a cleaner top-level narrative layout beside the legacy tree."""
    legacy_root = stage_dir / "narrative_exploration"
    clean_root = ensure_dir(stage_dir / "narrative_outputs")
    if not legacy_root.exists():
        return

    directory_map = {
        "model_error_profiles": "error_profiles",
        "reasoning_audit": "reasoning_audit",
        "themes": "topic_models/feature_summaries",
        "themes_full_context": "topic_models/full_context",
        "heuristic_themes": "heuristic_themes/feature_summaries",
        "heuristic_themes_full_context": "heuristic_themes/full_context",
        "temperature_comparison": "comparisons/by_temperature",
        "clinical_context_comparison": "comparisons/by_clinical_context",
        "model_comparison": "comparisons/by_model",
        "shot_comparison": "comparisons/by_shot_type",
    }
    for source_name, target_name in directory_map.items():
        _copy_tree_contents(legacy_root / source_name, clean_root / target_name)

    header_aliases = {
        "fp_weight": "false_positive_topic_share",
        "fn_weight": "false_negative_topic_share",
        "skew_score": "false_positive_minus_false_negative_skew",
        "top_terms": "top_topic_terms",
        "zero": "zero_shot",
        "single": "single_shot",
        "dual": "dual_shot",
        "high_minus_low_error_rate": "high_certainty_minus_low_certainty_error_rate",
    }
    file_aliases = {
        legacy_root
        / "diverging_words_overall.png": clean_root
        / "ngram_contrasts/title_text/overall_false-positive_vs_false-negative_unigrams.png",
        legacy_root
        / "diverging_trigrams_overall.png": clean_root
        / "ngram_contrasts/title_text/overall_false-positive_vs_false-negative_trigrams.png",
        legacy_root
        / "diverging_5grams_overall.png": clean_root
        / "ngram_contrasts/title_text/overall_false-positive_vs_false-negative_5grams.png",
        legacy_root
        / "diverging_words_overall_features.png": clean_root
        / "ngram_contrasts/feature_summaries/overall_false-positive_vs_false-negative_unigrams.png",
        legacy_root
        / "diverging_trigrams_overall_features.png": clean_root
        / "ngram_contrasts/feature_summaries/overall_false-positive_vs_false-negative_trigrams.png",
        legacy_root
        / "diverging_5grams_overall_features.png": clean_root
        / "ngram_contrasts/feature_summaries/overall_false-positive_vs_false-negative_5grams.png",
        legacy_root
        / "model_error_profiles/certainty_error_rates.csv": clean_root
        / "error_profiles/error_rates_by_certainty_band.csv",
        legacy_root
        / "model_error_profiles/catastrophic_failure_rates.csv": clean_root
        / "error_profiles/catastrophic_failure_rates_by_model.csv",
        legacy_root
        / "reasoning_audit/reasoning_cue_deltas_all_models.csv": clean_root
        / "reasoning_audit/reasoning_cue_deltas_by_model.csv",
        legacy_root
        / "reasoning_audit/narrative_error_mechanisms.csv": clean_root
        / "reasoning_audit/dominant_narrative_error_mechanisms.csv",
        legacy_root
        / "themes/topic_skew_nmf.csv": clean_root
        / "topic_models/feature_summaries/topic_skew_false-positive_vs_false-negative_nmf.csv",
        legacy_root
        / "themes/topic_skew_lda.csv": clean_root
        / "topic_models/feature_summaries/topic_skew_false-positive_vs_false-negative_lda.csv",
        legacy_root
        / "themes_full_context/topic_skew_nmf.csv": clean_root
        / "topic_models/full_context/topic_skew_false-positive_vs_false-negative_nmf.csv",
        legacy_root
        / "themes_full_context/topic_skew_lda.csv": clean_root
        / "topic_models/full_context/topic_skew_false-positive_vs_false-negative_lda.csv",
        legacy_root
        / "heuristic_themes/heuristic_theme_skew.csv": clean_root
        / "heuristic_themes/feature_summaries/heuristic_theme_false-positive_vs_false-negative_skew.csv",
        legacy_root
        / "heuristic_themes/theme_prevalence_by_outcome.csv": clean_root
        / "heuristic_themes/feature_summaries/heuristic_theme_prevalence_by_outcome.csv",
        legacy_root
        / "heuristic_themes_full_context/heuristic_theme_skew.csv": clean_root
        / "heuristic_themes/full_context/heuristic_theme_false-positive_vs_false-negative_skew.csv",
        legacy_root
        / "heuristic_themes_full_context/theme_prevalence_by_outcome.csv": clean_root
        / "heuristic_themes/full_context/heuristic_theme_prevalence_by_outcome.csv",
    }
    for source, destination in file_aliases.items():
        if source.exists():
            _copy_with_csv_header_aliases(
                source, destination, rename_map=header_aliases
            )

    write_dataframe(
        pd.DataFrame(
            [
                {
                    "section": "N-gram contrasts",
                    "path": "narrative_outputs/ngram_contrasts",
                },
                {"section": "Topic models", "path": "narrative_outputs/topic_models"},
                {
                    "section": "Heuristic themes",
                    "path": "narrative_outputs/heuristic_themes",
                },
                {
                    "section": "Error profiles",
                    "path": "narrative_outputs/error_profiles",
                },
                {
                    "section": "Reasoning audit",
                    "path": "narrative_outputs/reasoning_audit",
                },
                {"section": "Comparisons", "path": "narrative_outputs/comparisons"},
                {"section": "Legacy layout", "path": "narrative_exploration"},
            ]
        ),
        clean_root / "output_index.csv",
    )


def run_narrative_analysis(config: AnalysisConfig) -> dict[str, Path]:
    """Run the narrative-analysis stage.

    Loads the per-row dataframe written by ``data_prep`` and produces
    n-gram diverging plots, NMF/LDA topic skew tables, per-model error
    profiles (high vs low certainty error rates, catastrophic failure
    rates, top FP/FN/catastrophic example tables), reasoning-cue
    deltas with cross-model heatmaps, and a shot-type comparison.

    Parameters
    ----------
    config:
        The active :class:`AnalysisConfig`.

    Returns
    -------
    dict[str, pathlib.Path]
        Mapping with at least ``stage_dir``.
    """
    stage_dir = ensure_dir(config.paths["output_root"] / "narrative_analysis")
    root = ensure_dir(stage_dir / "narrative_exploration")

    merged_path = config.paths["output_root"] / "data_prep" / "merged_outputs.csv"
    if not merged_path.exists():
        merged_path = config.paths["output_root"] / "data_prep" / "parsed_outputs.csv"
    if not merged_path.exists():
        LOGGER.warning(
            "Narrative analysis skipped: no merged or parsed inputs at %s", merged_path
        )
        return {"stage_dir": stage_dir}

    frame = pd.read_csv(merged_path)
    frame, truth_col = prepare_truth_frame(frame, mode=truth_mode(config))
    if truth_col is None or "likelihood_score" not in frame.columns:
        LOGGER.warning(
            "Narrative analysis skipped: truth or likelihood_score columns missing"
        )
        _clear_stage_outputs(stage_dir)
        return {"stage_dir": stage_dir}

    frame[truth_col] = pd.to_numeric(frame[truth_col], errors="coerce")
    frame = frame.dropna(subset=[truth_col]).copy()
    frame[truth_col] = frame[truth_col].astype(int)
    likelihood = pd.to_numeric(frame["likelihood_score"], errors="coerce")
    certainty = pd.to_numeric(
        frame.get("certainty_score", pd.Series(dtype=float)), errors="coerce"
    )
    # Use the shared "drop" policy here: rows whose likelihood cannot
    # be coerced stay as <NA> rather than being silently classified
    # as negative. Downstream filtering below removes them.
    frame["_pred"] = derive_prediction(likelihood, missing_policy="drop")

    valid = frame["_pred"].notna()
    frame = frame.loc[valid].copy()
    frame["_pred"] = frame["_pred"].astype(int)
    likelihood = likelihood.loc[frame.index]
    certainty = certainty.loc[frame.index]

    frame["_is_fp"] = (frame["_pred"] == 1) & (frame[truth_col] == 0)
    frame["_is_fn"] = (frame["_pred"] == 0) & (frame[truth_col] == 1)
    frame["_is_tp"] = (frame["_pred"] == 1) & (frame[truth_col] == 1)
    frame["_is_tn"] = (frame["_pred"] == 0) & (frame[truth_col] == 0)
    frame["_is_error"] = frame["_is_fp"] | frame["_is_fn"]
    frame["_high_cert"] = (certainty >= HIGH_CERTAINTY_THRESHOLD).fillna(False)
    frame["_low_cert"] = (certainty < HIGH_CERTAINTY_THRESHOLD).fillna(False)
    frame["_extreme_like"] = (
        (likelihood <= EXTREME_LIKELIHOOD_LOW) | (likelihood >= EXTREME_LIKELIHOOD_HIGH)
    ).fillna(False)
    frame["_catastrophic"] = (
        frame["_is_error"] & frame["_high_cert"] & frame["_extreme_like"]
    )

    title_col = "Title" if "Title" in frame.columns else None
    feat_col = "Features" if "Features" in frame.columns else None

    _ngram_section(frame, root, title_col, feat_col)

    if feat_col is not None:
        _topic_section(frame, root, feat_col, truth_col, label="themes")
        _heuristic_theme_section(frame, root, feat_col, label="heuristic_themes")
    full_text_col = _full_context_column(frame)
    if full_text_col is not None:
        _topic_section(
            frame, root, full_text_col, truth_col, label="themes_full_context"
        )
        _heuristic_theme_section(
            frame,
            root,
            full_text_col,
            label="heuristic_themes_full_context",
        )

    # Mirror the thematic outputs as soon as they exist so long-running
    # narrative jobs expose the clean layout before the final stages finish.
    _mirror_narrative_outputs(stage_dir)

    if "model_canon" in frame.columns and feat_col is not None:
        _model_error_profiles(frame, root, feat_col, truth_col)
        _reasoning_audit(frame, root, feat_col)

    if feat_col is not None:
        _cue_heatmap_comparison(
            frame,
            root,
            feat_col,
            group_col="temperature",
            out_dir_name="temperature_comparison",
            prefix="temperature",
            display_name="temperature",
            preferred_order=[round(temp, 2) for temp in FAIR_TEMPERATURES_NUMERIC],
            formatter=lambda value: f"{float(value):.2f}",
        )
        _cue_heatmap_comparison(
            frame,
            root,
            feat_col,
            group_col="report_sequence_name",
            out_dir_name="clinical_context_comparison",
            prefix="clinical_context",
            display_name="clinical contextual variation",
            preferred_order=CONTEXT_ORDER,
            formatter=str,
        )
        _cue_heatmap_comparison(
            frame,
            root,
            feat_col,
            group_col="model_canon",
            out_dir_name="model_comparison",
            prefix="model",
            display_name="model",
            formatter=str,
        )

    if (
        "shot_type" in frame.columns
        and feat_col is not None
        and frame["shot_type"].dropna().nunique() > 1
    ):
        _shot_comparison(frame, root, feat_col)

    _mirror_narrative_outputs(stage_dir)

    return {"stage_dir": stage_dir}
