"""Real JSON repair and schema validation for LLM responses.

The historical pipeline did two things and called them both "repair":

1. ``json.loads`` the whole string.
2. Regex out the first ``{...}`` block and ``json.loads`` *that*.

Step 2 is not repair, it is extraction. This module restores the
methods-text contract by implementing genuine repair:

* trim leading / trailing prose and code fences;
* normalise typographic quotes (``\u201c \u201d \u2018 \u2019``) and
  stray non-breaking spaces to ASCII;
* close obviously-truncated objects by appending the missing closing
  braces / brackets;
* drop trailing commas before a closing brace.

It also validates the parsed payload against the published prompt
schema (``CLUES`` / ``REASONING`` / ``OUTCOME`` with three
``0..10`` numeric fields), so downstream stages can distinguish
"valid JSON" from "valid JSON that actually answers the question".

Every parse attempt is tagged with one of the discrete labels in
:data:`REPAIR_TYPES`. Stages that summarise dropout should aggregate
these tags rather than re-deriving "repaired" from the raw success
flag.
"""

from __future__ import annotations

import json
import re
from dataclasses import dataclass
from typing import Any, Literal

import pandas as pd

#: Discrete repair / extraction outcomes. ``schema_invalid`` covers
#: payloads that parsed as JSON but failed schema validation;
#: ``unparseable`` covers payloads that could not be coerced even
#: after repair.
RepairType = Literal[
    "raw_valid_json",
    "extracted_embedded_json",
    "repaired_json",
    "schema_invalid",
    "unparseable",
    "empty",
]

REPAIR_TYPES: tuple[RepairType, ...] = (
    "raw_valid_json",
    "extracted_embedded_json",
    "repaired_json",
    "schema_invalid",
    "unparseable",
    "empty",
)

_FENCE_PATTERN = re.compile(r"```(?:json)?\s*(.*?)\s*```", re.DOTALL | re.IGNORECASE)
_OBJECT_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
_TRAILING_COMMA_PATTERN = re.compile(r",\s*([}\]])")
_QUOTE_TRANSLATION = str.maketrans(
    {
        "\u201c": '"',
        "\u201d": '"',
        "\u2018": "'",
        "\u2019": "'",
        "\u00a0": " ",
    }
)

# Schema constants
_NUMERIC_FIELDS: tuple[str, ...] = (
    "Likelihood of IBD",
    "Certainty Level",
    "Complexity of Case",
)


@dataclass(frozen=True)
class ParseResult:
    """Outcome of a single response parse attempt."""

    payload: dict[str, Any] | None
    repair_type: RepairType
    schema_valid: bool
    schema_errors: tuple[str, ...] = ()


def _normalise(text: str) -> str:
    """Return ``text`` with curly quotes / NBSP normalised to ASCII."""
    return text.translate(_QUOTE_TRANSLATION)


def _strip_fences(text: str) -> str:
    """If ``text`` contains a fenced code block, return its body; otherwise return ``text``."""
    match = _FENCE_PATTERN.search(text)
    if match:
        return match.group(1).strip()
    return text


def _balance_braces(text: str) -> str:
    """Append missing closing ``}`` / ``]`` characters to ``text``.

    Counts opening and closing braces / brackets *outside* string
    literals and appends whatever is missing. Returns the original
    text unchanged when the counts already match.
    """
    open_curly = open_square = 0
    in_string = False
    escape = False
    for ch in text:
        if escape:
            escape = False
            continue
        if ch == "\\" and in_string:
            escape = True
            continue
        if ch == '"':
            in_string = not in_string
            continue
        if in_string:
            continue
        if ch == "{":
            open_curly += 1
        elif ch == "}":
            open_curly -= 1
        elif ch == "[":
            open_square += 1
        elif ch == "]":
            open_square -= 1
    suffix = ""
    if open_square > 0:
        suffix += "]" * open_square
    if open_curly > 0:
        suffix += "}" * open_curly
    return text + suffix


def _try_load(text: str) -> dict[str, Any] | None:
    try:
        result = json.loads(text)
    except (json.JSONDecodeError, ValueError):
        return None
    return result if isinstance(result, dict) else None


def _attempt_repair(text: str) -> dict[str, Any] | None:
    """Try a sequence of repairs and return the first parse that succeeds."""
    candidate = _TRAILING_COMMA_PATTERN.sub(r"\1", text)
    result = _try_load(candidate)
    if result is not None:
        return result
    balanced = _balance_braces(candidate)
    if balanced != candidate:
        result = _try_load(balanced)
        if result is not None:
            return result
    return None


def validate_schema(payload: dict[str, Any]) -> tuple[bool, tuple[str, ...]]:
    """Validate ``payload`` against the published prompt schema.

    Returns ``(is_valid, errors)``. The schema requires the three
    numeric ``OUTCOME`` fields to be present and within ``0..10``; the
    ``CLUES`` / ``REASONING`` / ``Title`` / ``Features`` fields are
    treated as soft (their absence is not by itself a schema failure)
    because the early prompt revisions used different casings.
    """
    errors: list[str] = []
    flat: dict[str, Any] = {}
    flat.update(payload)
    outcome = payload.get("OUTCOME")
    if isinstance(outcome, dict):
        flat.update(outcome)
    for field in _NUMERIC_FIELDS:
        if field not in flat:
            errors.append(f"missing:{field}")
            continue
        try:
            value = float(flat[field])
        except (TypeError, ValueError):
            errors.append(f"non_numeric:{field}")
            continue
        if not 0.0 <= value <= 10.0:
            errors.append(f"out_of_range:{field}")
    return (not errors, tuple(errors))


def parse_response(text: Any) -> ParseResult:
    """Parse a single model response with repair + schema validation.

    The function preserves the historical "extract embedded JSON"
    behaviour as a labelled step so downstream summaries can
    distinguish raw-valid responses from extracted ones from
    genuinely-repaired ones from unparseable ones.
    """
    if pd.isna(text):
        return ParseResult(None, "empty", False)
    if isinstance(text, dict):
        valid, errors = validate_schema(text)
        return ParseResult(
            text, "raw_valid_json" if valid else "schema_invalid", valid, errors
        )
    raw = str(text).strip()
    if not raw:
        return ParseResult(None, "empty", False)

    raw = _normalise(raw)
    raw = _strip_fences(raw)

    payload = _try_load(raw)
    repair: RepairType
    if payload is not None:
        repair = "raw_valid_json"
    else:
        match = _OBJECT_PATTERN.search(raw)
        if match is not None:
            payload = _try_load(match.group(0))
            if payload is not None:
                repair = "extracted_embedded_json"
            else:
                payload = _attempt_repair(match.group(0))
                repair = "repaired_json" if payload is not None else "unparseable"
        else:
            payload = _attempt_repair(raw)
            repair = "repaired_json" if payload is not None else "unparseable"

    if payload is None:
        return ParseResult(None, "unparseable", False)
    valid, errors = validate_schema(payload)
    if not valid:
        return ParseResult(payload, "schema_invalid", False, errors)
    return ParseResult(payload, repair, True, ())
