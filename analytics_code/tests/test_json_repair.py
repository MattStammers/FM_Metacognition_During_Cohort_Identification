"""Tests for :mod:`analytics_code.json_repair`."""

from __future__ import annotations

import math

from analytics_code.json_repair import REPAIR_TYPES, parse_response, validate_schema


def _outcome(likelihood: int = 6, certainty: int = 5, complexity: int = 4) -> dict:
    return {
        "CLUES": ["one", "two"],
        "REASONING": "because",
        "OUTCOME": {
            "Title": "x",
            "Features": [],
            "Likelihood of IBD": likelihood,
            "Certainty Level": certainty,
            "Complexity of Case": complexity,
        },
    }


def test_raw_valid_json_round_trip() -> None:
    text = '{"OUTCOME": {"Likelihood of IBD": 7, "Certainty Level": 5, "Complexity of Case": 4}}'
    result = parse_response(text)
    assert result.repair_type == "raw_valid_json"
    assert result.schema_valid is True


def test_extracted_embedded_json_is_labelled_separately() -> None:
    text = "Here is the analysis: " + (
        '{"OUTCOME": {"Likelihood of IBD": 1, "Certainty Level": 1, "Complexity of Case": 1}}'
        + " thanks."
    )
    result = parse_response(text)
    assert result.repair_type == "extracted_embedded_json"
    assert result.schema_valid is True


def test_repaired_json_for_unclosed_braces() -> None:
    text = (
        '{"OUTCOME": {"Likelihood of IBD": 7, "Certainty Level": 5, '
        '"Complexity of Case": 4}'
    )
    result = parse_response(text)
    assert result.repair_type == "repaired_json"
    assert result.payload is not None
    assert result.schema_valid is True


def test_curly_quote_normalisation_enables_parse() -> None:
    text = "{\u201cOUTCOME\u201d: {\u201cLikelihood of IBD\u201d: 7, \u201cCertainty Level\u201d: 5, \u201cComplexity of Case\u201d: 4}}"
    result = parse_response(text)
    # Curly quotes are normalised to ASCII; the body parses straight.
    assert result.payload is not None
    assert result.schema_valid is True


def test_unparseable_returns_none() -> None:
    result = parse_response("no json here, just words")
    assert result.repair_type == "unparseable"
    assert result.payload is None


def test_empty_input_is_empty_tag() -> None:
    assert parse_response("").repair_type == "empty"
    assert parse_response(None).repair_type == "empty"
    assert parse_response(math.nan).repair_type == "empty"


def test_schema_invalid_when_likelihood_out_of_range() -> None:
    text = '{"OUTCOME": {"Likelihood of IBD": 99, "Certainty Level": 5, "Complexity of Case": 4}}'
    result = parse_response(text)
    assert result.repair_type == "schema_invalid"
    assert result.schema_valid is False
    assert any("out_of_range" in err for err in result.schema_errors)


def test_validate_schema_direct_payload() -> None:
    ok, errors = validate_schema(_outcome())
    assert ok and errors == ()


def test_repair_types_constant_is_complete() -> None:
    assert set(REPAIR_TYPES) >= {
        "raw_valid_json",
        "extracted_embedded_json",
        "repaired_json",
        "schema_invalid",
        "unparseable",
        "empty",
    }
