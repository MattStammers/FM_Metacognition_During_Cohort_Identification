"""Tests for analytics_code.data_prep."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from analytics_code.config import AnalysisConfig
from analytics_code.data_prep import (
    DOCUMENT_SEQUENCE_TRUTH_LENIENT,
    DOCUMENT_SEQUENCE_TRUTH_STRICT,
    _attach_document_sequence_truth_columns,
    _extract_prompt_sections,
    _infer_temperature,
    _response_column_map,
    combine_batch_exports,
    extract_json_payload,
    find_batch_files,
    normalize_chronology_outputs,
)

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _make_config(tmp_path: Path, extra_raw: dict | None = None) -> AnalysisConfig:
    raw = {
        "paths": {"output_root": str(tmp_path / "outputs")},
        "model_mapping": {},
        "runner_metadata": {},
    }
    if extra_raw:
        raw.update(extra_raw)
    return AnalysisConfig(raw=raw, config_path=tmp_path / "config.json")


# ---------------------------------------------------------------------------
# find_batch_files
# ---------------------------------------------------------------------------


def test_find_batch_files_returns_sorted_matches(tmp_path: Path) -> None:
    (tmp_path / "batch_b.csv").write_text("a,b", encoding="utf-8")
    (tmp_path / "batch_a.csv").write_text("a,b", encoding="utf-8")
    (tmp_path / "other.txt").write_text("x", encoding="utf-8")
    result = find_batch_files(tmp_path, "batch_*.csv")
    names = [p.name for p in result]
    assert names == ["batch_a.csv", "batch_b.csv"]


def test_find_batch_files_empty_when_no_match(tmp_path: Path) -> None:
    result = find_batch_files(tmp_path, "*.xlsx")
    assert result == []


# ---------------------------------------------------------------------------
# combine_batch_exports
# ---------------------------------------------------------------------------


def test_combine_batch_exports_adds_source_columns(tmp_path: Path) -> None:
    runner_dir = tmp_path / "runner1" / "exp1"
    runner_dir.mkdir(parents=True)
    f = runner_dir / "batch.csv"
    pd.DataFrame({"col": [1, 2]}).to_csv(f, index=False)

    result = combine_batch_exports([f])
    assert "source_file" in result.columns
    assert "runner_name" in result.columns
    assert result["source_file"].iloc[0] == "batch.csv"
    assert result["runner_name"].iloc[0] == "runner1"


def test_combine_batch_exports_concatenates_multiple_files(tmp_path: Path) -> None:
    dir_a = tmp_path / "runner" / "exp_a"
    dir_b = tmp_path / "runner" / "exp_b"
    dir_a.mkdir(parents=True)
    dir_b.mkdir(parents=True)
    pd.DataFrame({"v": [1]}).to_csv(dir_a / "f.csv", index=False)
    pd.DataFrame({"v": [2]}).to_csv(dir_b / "f.csv", index=False)
    result = combine_batch_exports([dir_a / "f.csv", dir_b / "f.csv"])
    assert len(result) == 2


def test_combine_batch_exports_returns_empty_for_no_files() -> None:
    result = combine_batch_exports([])
    assert result.empty


# ---------------------------------------------------------------------------
# extract_json_payload
# ---------------------------------------------------------------------------


def test_extract_json_payload_parses_clean_json() -> None:
    payload = {"Likelihood of IBD": 7, "Certainty Level": 8}
    result = extract_json_payload(json.dumps(payload))
    assert result == payload


def test_extract_json_payload_extracts_embedded_json() -> None:
    text = 'Some text before {"Likelihood of IBD": 5} and after'
    result = extract_json_payload(text)
    assert result is not None
    assert result["Likelihood of IBD"] == 5


def test_extract_json_payload_returns_none_for_invalid() -> None:
    assert extract_json_payload("not json at all") is None


def test_extract_json_payload_returns_none_for_nan() -> None:
    assert extract_json_payload(float("nan")) is None


def test_extract_json_payload_returns_none_for_empty_string() -> None:
    assert extract_json_payload("") is None


def test_extract_json_payload_returns_dict_unchanged() -> None:
    d = {"key": "val"}
    assert extract_json_payload(d) is d


# ---------------------------------------------------------------------------
# _infer_temperature
# ---------------------------------------------------------------------------


def test_infer_temperature_from_metadata() -> None:
    metadata = {"display_model_x": {"temperature": 0.75}}
    result = _infer_temperature("display_model_x", metadata)
    assert result == pytest.approx(0.75)


def test_infer_temperature_from_two_digit_suffix() -> None:
    result = _infer_temperature("model_t075", {})
    assert result == pytest.approx(0.75)


def test_infer_temperature_from_three_digit_suffix() -> None:
    result = _infer_temperature("model_t060", {})
    assert result == pytest.approx(0.60)


def test_infer_temperature_returns_none_when_no_match() -> None:
    assert _infer_temperature("no_temp_here", {}) is None


# ---------------------------------------------------------------------------
# _extract_prompt_sections
# ---------------------------------------------------------------------------


def test_extract_prompt_sections_parses_structured_response() -> None:
    text = (
        "**CLUES:** The patient shows signs of IBD. "
        "**REASONING:** Based on the clues above. "
        "**OUTCOME:** positive"
    )
    clues, reasoning = _extract_prompt_sections(text)
    assert "IBD" in clues
    assert "clues above" in reasoning


def test_extract_prompt_sections_returns_empty_strings_for_no_match() -> None:
    clues, reasoning = _extract_prompt_sections("Random text with no sections")
    assert clues == ""
    assert reasoning == ""


def test_extract_prompt_sections_returns_empty_strings_for_nan() -> None:
    clues, reasoning = _extract_prompt_sections(float("nan"))
    assert clues == ""
    assert reasoning == ""


# ---------------------------------------------------------------------------
# _response_column_map
# ---------------------------------------------------------------------------


def test_response_column_map_categorizes_known_columns() -> None:
    cols = [
        "model_Json_Response_1",
        "model_Full_Response_1",
        "model_Payload_1",
        "Truncated_response",
        "unrelated",
    ]
    mapping = _response_column_map(cols)
    assert "model_Json_Response_1" in mapping["json"]
    assert "model_Full_Response_1" in mapping["full"]
    assert "model_Payload_1" in mapping["payload"]
    assert "Truncated_response" in mapping["truncated"]
    # 'unrelated' should not appear in any bucket
    for bucket in mapping.values():
        assert "unrelated" not in bucket


# ---------------------------------------------------------------------------
# normalize_chronology_outputs
# ---------------------------------------------------------------------------


def _minimal_batch_frame() -> pd.DataFrame:
    """Minimal combined batch frame for normalize_chronology_outputs."""
    return pd.DataFrame(
        {
            "patient_id": ["p1", "p2"],
            "experiment_name": ["zero_shot_endo", "single_shot_hist"],
            "runner_name": ["model_t075", "model_t060"],
            "model_Zero_shot_Full_Response_1": ["resp1", "resp2"],
            "model_Zero_shot_Json_Response_1": [
                '{"Likelihood of IBD": 7, "Certainty Level": 6, "Complexity of Case": 5}',
                '{"Likelihood of IBD": 3, "Certainty Level": 4, "Complexity of Case": 2}',
            ],
        }
    )


def test_normalize_chronology_outputs_adds_shot_type(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    frame = _minimal_batch_frame()
    result = normalize_chronology_outputs(frame, config)
    assert "shot_type" in result.columns
    assert result["shot_type"].iloc[0] == "zero"


def test_normalize_chronology_outputs_adds_model_canon(tmp_path: Path) -> None:
    config = _make_config(
        tmp_path, extra_raw={"model_mapping": {"model_t075": "MyModel"}}
    )
    frame = _minimal_batch_frame()
    result = normalize_chronology_outputs(frame, config)
    assert result["model_canon"].iloc[0] == "MyModel"


def test_normalize_chronology_outputs_parses_json_response(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    frame = _minimal_batch_frame()
    result = normalize_chronology_outputs(frame, config)
    assert result["json_parse_success"].iloc[0] == 1
    assert result["likelihood_score"].iloc[0] == pytest.approx(7.0)


def test_normalize_chronology_outputs_returns_empty_for_empty_input(
    tmp_path: Path,
) -> None:
    config = _make_config(tmp_path)
    result = normalize_chronology_outputs(pd.DataFrame(), config)
    assert result.empty


def test_normalize_chronology_outputs_infers_icl_shots(tmp_path: Path) -> None:
    config = _make_config(tmp_path)
    frame = _minimal_batch_frame()
    result = normalize_chronology_outputs(frame, config)
    assert result["icl_shots"].iloc[0] == 0  # zero_shot
    assert result["icl_shots"].iloc[1] == 1  # single_shot


def test_attach_document_sequence_truth_columns_emits_lenient_and_strict() -> None:
    frame = pd.DataFrame(
        {
            "report_sequence_name": [
                "all_docs_in_sequence",
                "all_docs_in_sequence",
                "all_docs_in_sequence",
            ],
            "preceding_clinic_IBD": [1, 0, 0],
            "endoscopy_IBD": [0, 0, 1],
            "histology_IBD": [0, float("nan"), 0],
        }
    )
    out = _attach_document_sequence_truth_columns(frame)
    assert DOCUMENT_SEQUENCE_TRUTH_LENIENT in out.columns
    assert DOCUMENT_SEQUENCE_TRUTH_STRICT in out.columns
    # Lenient: row 0 positive (any marker), row 1 negative (some markers observed,
    # none positive), row 2 positive.
    assert out[DOCUMENT_SEQUENCE_TRUTH_LENIENT].tolist() == [1.0, 0.0, 1.0]
    # Strict: row 1 must be NaN because histology marker is missing.
    strict = out[DOCUMENT_SEQUENCE_TRUTH_STRICT]
    assert strict.iloc[0] == 1.0
    assert pd.isna(strict.iloc[1])
    assert strict.iloc[2] == 1.0
