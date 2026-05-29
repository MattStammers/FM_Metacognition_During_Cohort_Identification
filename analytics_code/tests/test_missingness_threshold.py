"""Tests for analytics_code.missingness_threshold."""

from __future__ import annotations

from pathlib import Path

import pandas as pd

from analytics_code.config import AnalysisConfig
from analytics_code.missingness_threshold import run_missingness_threshold


def _make_config(tmp_path: Path) -> AnalysisConfig:
    raw = {
        "paths": {"output_root": str(tmp_path / "outputs")},
        "model_mapping": {},
        "analysis": {},
    }
    return AnalysisConfig(raw=raw, config_path=tmp_path / "config.json")


def _make_document_config(tmp_path: Path, mode: str) -> AnalysisConfig:
    raw = {
        "paths": {"output_root": str(tmp_path / "outputs")},
        "model_mapping": {},
        "analysis": {"truth_mode": mode},
    }
    return AnalysisConfig(raw=raw, config_path=tmp_path / "config.json")


def _merged_frame() -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    specs = [
        ("mixtral_7b", 0.75, "zero", "all_docs_in_sequence"),
        ("mixtral_7b", 1.00, "zero", "all_docs_in_sequence"),
        ("m42_8b", 0.75, "zero", "all_docs_in_sequence"),
        ("m42_8b", 1.00, "zero", "all_docs_in_sequence"),
        ("deepseek_14b", 0.50, "zero", "all_docs_in_sequence"),
        ("deepseek_14b", 0.75, "zero", "all_docs_in_sequence"),
        ("qwen_32b", 0.60, "zero", "all_docs_in_sequence"),
        ("qwen_32b", 0.75, "zero", "all_docs_in_sequence"),
    ]
    for model, temperature, shot, context in specs:
        for idx in range(10):
            truth = idx % 2
            likelihood = (7 + (idx % 3)) if truth else (2 + (idx % 3))
            certainty = 5 + (idx % 4)
            complexity = 3 + (idx % 4)
            if idx == 0 and temperature >= 0.75:
                likelihood = None
            rows.append(
                {
                    "model_canon": model,
                    "temperature": temperature,
                    "shot_type": shot,
                    "report_sequence_name": context,
                    "Patient_Has_IBD": truth,
                    "likelihood_score": likelihood,
                    "certainty_score": certainty,
                    "complexity_score": complexity,
                }
            )
    return pd.DataFrame(rows)


def test_run_missingness_threshold_writes_extended_outputs(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    data_prep_dir = tmp_path / "outputs" / "data_prep"
    data_prep_dir.mkdir(parents=True)
    _merged_frame().to_csv(data_prep_dir / "merged_outputs.csv", index=False)

    result = run_missingness_threshold(config)

    root = result["stage_dir"]
    assert (
        root
        / "descriptive_stats"
        / "dropout_by_temperature_context_all_docs_in_sequence.csv"
    ).exists()
    assert (
        root
        / "descriptive_stats"
        / "dropout_by_temperature_context_all_docs_in_sequence_small_models.csv"
    ).exists()
    assert (root / "basic_thresholds" / "basic_threshold_performance.csv").exists()
    assert (
        root
        / "macro_f1_calibration"
        / "best_thresholds_by_model_temperature_macroF1.csv"
    ).exists()
    assert (
        root / "confidence_complexity_plots" / "bucket_summary_by_model_pool.csv"
    ).exists()
    assert (
        root / "calibration_per_model" / "calibration_summary_by_model_pool.csv"
    ).exists()


def test_run_missingness_threshold_clears_stale_truth_outputs_when_truth_missing(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)
    data_prep_dir = tmp_path / "outputs" / "data_prep"
    data_prep_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "model_canon": ["ModelA"],
            "report_sequence_name": ["clinic_preceding"],
            "likelihood_score": [5],
            "certainty_score": [5],
            "complexity_score": [5],
            "preceding_clinic_time_diff": [1],
        }
    ).to_csv(data_prep_dir / "merged_outputs.csv", index=False)

    stale = (
        tmp_path
        / "outputs"
        / "missingness_threshold"
        / "macro_f1_calibration"
        / "best_thresholds_by_model_macroF1.csv"
    )
    stale.parent.mkdir(parents=True)
    stale.write_text("stale\n", encoding="ascii")

    result = run_missingness_threshold(config)

    root = result["stage_dir"]
    assert (root / "descriptive_stats" / "missingness_summary.csv").exists()
    assert not (root / "macro_f1_calibration").exists()


def test_run_missingness_threshold_does_not_fall_back_to_patient_truth_in_document_mode(
    tmp_path: Path,
) -> None:
    config = _make_document_config(tmp_path, "document_complete")
    data_prep_dir = tmp_path / "outputs" / "data_prep"
    data_prep_dir.mkdir(parents=True)
    pd.DataFrame(
        {
            "model_canon": ["ModelA"],
            "report_sequence_name": ["clinic_preceding"],
            "likelihood_score": [7],
            "certainty_score": [5],
            "complexity_score": [5],
            "ground_truth": [1],
            "preceding_clinic_time_diff": [1],
        }
    ).to_csv(data_prep_dir / "merged_outputs.csv", index=False)

    stale = (
        tmp_path
        / "outputs"
        / "missingness_threshold"
        / "macro_f1_calibration"
        / "best_thresholds_by_model_macroF1.csv"
    )
    stale.parent.mkdir(parents=True)
    stale.write_text("stale\n", encoding="ascii")

    result = run_missingness_threshold(config)

    root = result["stage_dir"]
    assert (root / "descriptive_stats" / "missingness_summary.csv").exists()
    assert not (root / "macro_f1_calibration").exists()
