"""Tests for :mod:`python_client.prompts`."""

from __future__ import annotations

import pandas as pd

from python_client.prompts import (
    construct_combined_message,
    determine_context_note_timing,
    generate_experiments,
    prepare_reports_section,
)


def test_determine_context_note_timing() -> None:
    assert determine_context_note_timing("clinic_preceding") == "preceding"
    assert determine_context_note_timing("clinic_following") == "following"
    assert determine_context_note_timing("clinic_preceding_following") == "both"


def test_generate_experiments_builds_cross_product() -> None:
    experiments = generate_experiments(
        {
            "prompt_templates": {"zero": "z", "single": "s"},
            "report_sequences": {
                "hist": ["HISTOPATHOLOGY REPORT"],
                "endo": ["ENDOSCOPY REPORT"],
            },
        }
    )
    assert len(experiments) == 4
    assert experiments[0]["experiment_name"].startswith("zero_shot_")


def test_prepare_reports_section_handles_clinic_letter_aliases() -> None:
    row = pd.Series(
        {
            "result_report": "Hist text",
            "Combined_Content": "Endo text",
            "preceding_clinic_letter": "Pre letter",
            "following_clinic_letter": "Post letter",
        }
    )
    section = prepare_reports_section(
        row,
        ["CLINIC LETTER", "ENDOSCOPY REPORT", "HISTOPATHOLOGY REPORT"],
        "both",
    )
    assert "PRECEDING CLINIC LETTER" in section
    assert "FOLLOWING CLINIC LETTER" in section
    assert "ENDOSCOPY REPORT" in section
    assert "HISTOPATHOLOGY REPORT" in section


def test_construct_combined_message_embeds_template_and_reports() -> None:
    message = construct_combined_message("Prompt body", "Report section")
    assert "Prompt body" in message
    assert "Report section" in message
    assert "Provide only a SINGLE JSON response" in message
