"""Tests for :mod:`python_client.config`."""

from __future__ import annotations

import json

import pytest

from python_client.config import ConfigError, load_config


def _write_template_config(tmp_path, *, overrides=None):
    prompt_dir = tmp_path / "prompt_templates"
    prompt_dir.mkdir()
    for name in ("zero", "single", "dual"):
        (prompt_dir / f"{name}.txt").write_text(f"{name} prompt", encoding="utf-8")

    config = {
        "general": {
            "batch_size": 2,
            "version": "v1",
            "output_dir": "outputs",
            "max_parallel_runners": 2,
            "api_endpoints": {"runner_a": "http://127.0.0.1:9001"},
        },
        "data_sources": {
            "primary_reports": {
                "path": "primary.csv",
                "patient_id_column": "patient_id",
                "date_column": "sample_received_date",
                "text_column": "result_report",
                "label": "HISTOPATHOLOGY REPORT",
            },
            "secondary_reports": {
                "path": "secondary.csv",
                "patient_id_column": "patient_id",
                "date_column": "procedure_date",
                "text_column": "Combined_Content",
                "label": "ENDOSCOPY REPORT",
            },
            "clinic_letters": {
                "path": "notes.csv",
                "patient_id_column": "patient_id",
                "date_column": "date_creation",
                "text_column": "clean_content",
                "preceding_label": "PRECEDING CLINIC LETTER",
                "following_label": "FOLLOWING CLINIC LETTER",
            },
        },
        "matching": {"max_days_between_primary_and_secondary": 2},
        "prompt_templates": {
            "zero": "prompt_templates/zero.txt",
            "single": "prompt_templates/single.txt",
            "dual": "prompt_templates/dual.txt",
        },
        "report_sequences": {"hist": ["HISTOPATHOLOGY REPORT"]},
    }
    if overrides:
        config.update(overrides)

    config_path = tmp_path / "config.json"
    config_path.write_text(json.dumps(config), encoding="utf-8")
    return config_path


def test_load_config_normalizes_clinic_letters_and_runners(tmp_path) -> None:
    config_path = _write_template_config(tmp_path)
    loaded = load_config(config_path)

    assert "context_notes" in loaded["data_sources"]
    assert loaded["runner_endpoints"]["runner_a"]["enabled"] is True
    assert loaded["general"]["tokenizer_encoding"] == "cl100k_base"
    assert loaded["general"]["output_dir"].endswith("outputs")
    assert loaded["prompt_templates"]["zero"] == "zero prompt"


def test_load_config_rejects_too_many_parallel_runners(tmp_path) -> None:
    config_path = _write_template_config(
        tmp_path,
        overrides={
            "general": {
                "batch_size": 2,
                "version": "v1",
                "output_dir": "outputs",
                "max_parallel_runners": 5,
                "api_endpoints": {"runner_a": "http://127.0.0.1:9001"},
            }
        },
    )

    with pytest.raises(ConfigError, match="cannot exceed 4"):
        load_config(config_path)
