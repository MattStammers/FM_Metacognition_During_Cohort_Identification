"""Tests for analytics_code.narrative_analysis."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analytics_code.config import AnalysisConfig
from analytics_code.narrative_analysis import _comparison_filter, run_narrative_analysis


def _make_config(tmp_path: Path) -> AnalysisConfig:
    raw = {
        "paths": {"output_root": str(tmp_path / "outputs")},
        "model_mapping": {},
        "analysis": {},
    }
    return AnalysisConfig(raw=raw, config_path=tmp_path / "config.json")


def _build_row(
    *,
    model: str,
    temperature: float,
    shot: str,
    context: str,
    truth: int,
    likelihood: int,
    certainty: int,
    features: str,
) -> dict[str, object]:
    return {
        "model_canon": model,
        "temperature": temperature,
        "shot_type": shot,
        "report_sequence_name": context,
        "Patient_Has_IBD": truth,
        "likelihood_score": likelihood,
        "certainty_score": certainty,
        "Features": features,
        "Title": features.split()[0].title(),
        "Combined_Content": features,
    }


def _merged_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    cases = [
        {
            "truth": 0,
            "likelihood": 2,
            "certainty": 4,
            "features": "normal well remission",
        },
        {
            "truth": 0,
            "likelihood": 8,
            "certainty": 4,
            "features": "crohn active inflammation",
        },
        {
            "truth": 1,
            "likelihood": 8,
            "certainty": 4,
            "features": "crohn flare bleeding",
        },
        {
            "truth": 1,
            "likelihood": 2,
            "certainty": 4,
            "features": "normal benign resolved",
        },
        {
            "truth": 0,
            "likelihood": 9,
            "certainty": 9,
            "features": "ulcerative colitis severe flare",
        },
        {
            "truth": 1,
            "likelihood": 3,
            "certainty": 4,
            "features": "unclear possible inflammatory activity",
        },
    ]
    for model in ("ModelA", "ModelB"):
        for temperature in (0.60, 0.75):
            for shot in ("zero", "single"):
                for context in ("endo", "hist"):
                    for case in cases:
                        rows.append(
                            _build_row(
                                model=model,
                                temperature=temperature,
                                shot=shot,
                                context=context,
                                truth=int(case["truth"]),
                                likelihood=int(case["likelihood"]),
                                certainty=int(case["certainty"]),
                                features=str(case["features"]),
                            )
                        )
    return pd.DataFrame(rows)


def test_comparison_filter_temperature_keeps_zero_shot_and_fair_contexts() -> None:
    frame = pd.DataFrame(
        {
            "shot_type": ["zero", "single", "zero", "zero"],
            "temperature": [0.60, 0.75, 0.90, 0.75],
            "report_sequence_name": ["endo", "endo", "hist", "off_matrix"],
            "model_canon": ["ModelA", "ModelA", "ModelB", "ModelB"],
        }
    )

    result = _comparison_filter(frame, "temperature")

    assert (result["shot_type"] == "zero").all()
    assert set(result["report_sequence_name"].unique()).issubset({"endo", "hist"})


def test_comparison_filter_model_keeps_fair_slice() -> None:
    frame = pd.DataFrame(
        {
            "shot_type": ["zero", "single", "zero", "zero"],
            "temperature": [0.60, 0.75, 0.90, 0.75],
            "report_sequence_name": ["endo", "hist", "endo", "off_matrix"],
            "model_canon": ["ModelA", "ModelA", "ModelB", "ModelB"],
        }
    )

    result = _comparison_filter(frame, "model_canon")

    assert (result["shot_type"] == "zero").all()
    assert set(result["temperature"].unique()).issubset({0.60, 0.75})
    assert set(result["report_sequence_name"].unique()).issubset({"endo", "hist"})


def test_run_narrative_analysis_writes_new_comparison_outputs(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    data_prep_dir = tmp_path / "outputs" / "data_prep"
    data_prep_dir.mkdir(parents=True)
    _merged_frame().to_csv(data_prep_dir / "merged_outputs.csv", index=False)

    result = run_narrative_analysis(config)

    root = result["stage_dir"] / "narrative_exploration"
    assert (root / "temperature_comparison" / "temperature_cue_heatmap_fp.png").exists()
    assert (
        root / "clinical_context_comparison" / "clinical_context_cue_heatmap_fn.png"
    ).exists()
    assert (root / "model_comparison" / "model_cue_heatmap_catastrophic.png").exists()
    assert (root / "shot_comparison" / "shot_cue_heatmap_fp.png").exists()
    assert (root / "heuristic_themes" / "heuristic_theme_skew.csv").exists()
    assert (
        root
        / "heuristic_themes_model_comparison"
        / "heuristic_themes_heuristic_theme_heatmap_fp.png"
    ).exists()
    assert (
        root / "heuristic_themes_full_context" / "theme_prevalence_by_outcome.csv"
    ).exists()

    clean_root = result["stage_dir"] / "narrative_outputs"
    assert (clean_root / "output_index.csv").exists()
    assert (
        clean_root
        / "topic_models"
        / "feature_summaries"
        / "topic_skew_false-positive_vs_false-negative_nmf.csv"
    ).exists()


def test_run_narrative_analysis_uses_full_response_for_full_context_outputs(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)
    data_prep_dir = tmp_path / "outputs" / "data_prep"
    data_prep_dir.mkdir(parents=True)
    frame = _merged_frame().drop(columns=["Combined_Content"])
    frame["full_response"] = frame["Features"] + " extra full context"
    frame.to_csv(data_prep_dir / "merged_outputs.csv", index=False)

    result = run_narrative_analysis(config)

    root = result["stage_dir"] / "narrative_exploration"
    assert (root / "themes_full_context" / "topic_skew_nmf.csv").exists()
    assert (
        root / "heuristic_themes_full_context" / "theme_prevalence_by_outcome.csv"
    ).exists()
