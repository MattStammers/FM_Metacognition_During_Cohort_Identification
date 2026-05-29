"""Tests for section-aware message truncation."""

from __future__ import annotations

import pandas as pd
import tiktoken

from python_client.prompts import (
    TRAILING_REMINDER,
    build_message_with_budget,
    prepare_report_sections,
)

TOKENIZER = tiktoken.get_encoding("cl100k_base")


def _row(**overrides) -> pd.Series:
    base = {
        "result_report": "histology body",
        "Combined_Content": "endoscopy body",
        "preceding_clinic_letter": "preceding clinic body",
        "following_clinic_letter": "following clinic body",
    }
    base.update(overrides)
    return pd.Series(base)


def test_short_message_is_not_truncated() -> None:
    sections = prepare_report_sections(
        _row(),
        ["HISTOPATHOLOGY REPORT", "ENDOSCOPY REPORT"],
        context_note_timing="preceding",
    )
    message, truncated, sections_dropped = build_message_with_budget(
        "TEMPLATE", sections, TOKENIZER, token_limit=4000
    )
    assert truncated is False
    assert sections_dropped == []
    assert "histology body" in message
    assert "endoscopy body" in message
    assert TRAILING_REMINDER in message


def test_long_later_section_is_truncated_not_dropped() -> None:
    long_body = ("alpha beta gamma " * 500).strip()
    sections = prepare_report_sections(
        _row(Combined_Content=long_body),
        ["HISTOPATHOLOGY REPORT", "ENDOSCOPY REPORT"],
        context_note_timing="preceding",
    )
    message, truncated, sections_dropped = build_message_with_budget(
        "TEMPLATE", sections, TOKENIZER, token_limit=200
    )
    assert truncated is True
    # The endoscopy report is the longer one; it should be the one
    # whose body is reduced, but its header must still appear so the
    # model sees the section exists.
    assert "ENDOSCOPY REPORT" in message
    assert "HISTOPATHOLOGY REPORT" in message
    assert "histology body" in message
    assert "ENDOSCOPY REPORT" in sections_dropped


def test_message_token_count_respects_limit() -> None:
    sections = prepare_report_sections(
        _row(result_report="x " * 1000, Combined_Content="y " * 1000),
        ["HISTOPATHOLOGY REPORT", "ENDOSCOPY REPORT"],
        context_note_timing="preceding",
    )
    limit = 300
    message, truncated, _ = build_message_with_budget(
        "TEMPLATE", sections, TOKENIZER, token_limit=limit
    )
    assert truncated is True
    assert len(TOKENIZER.encode(message)) <= limit
