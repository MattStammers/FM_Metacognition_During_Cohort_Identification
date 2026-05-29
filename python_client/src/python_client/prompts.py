"""Prompt assembly for chronological experiments.

Given the prompt templates and report sequences declared in the
configuration, this module enumerates the experiments to run and
renders the per-row input message that is sent to the language model.
"""

from __future__ import annotations

from typing import Any

import pandas as pd
import tiktoken

#: Trailing instruction appended to every prompt so the parser has a
#: stable contract. Kept as a module constant so the section-aware
#: truncation helper can reserve its token budget.
TRAILING_REMINDER: str = (
    "**Important:** Provide only a SINGLE JSON response regarding the "
    "likelihood of IBD. Do not repeat or include multiple JSON blocks. "
    "Please ensure the final response is in the correct JSON format."
)


def determine_context_note_timing(report_sequence_name: str) -> str:
    """Infer whether a sequence uses preceding, following, or both context notes.

    The decision is purely lexical: it inspects ``report_sequence_name``
    for the substrings ``preceding`` and ``following`` and returns
    ``"both"``, ``"following"`` or (default) ``"preceding"``. Sequences
    declared with the explicit ``PRECEDING CLINIC LETTER`` /
    ``FOLLOWING CLINIC LETTER`` entries are preferred over the
    ambiguous ``CLINIC LETTER`` token; the explicit forms bypass this
    heuristic entirely.
    """
    if "preceding" in report_sequence_name and "following" in report_sequence_name:
        return "both"
    if "following" in report_sequence_name:
        return "following"
    return "preceding"


def generate_experiments(config: dict[str, Any]) -> list[dict[str, Any]]:
    """Cross-product the prompt templates with the report sequences.

    Returns one experiment dict per (template, sequence) combination,
    populated with the keys consumed by
    :func:`python_client.experiment_runner.run_chronological_experiments`.
    """
    experiments: list[dict[str, Any]] = []
    for prompt_name, prompt_content in config["prompt_templates"].items():
        for sequence_name, report_sequence in config["report_sequences"].items():
            experiments.append(
                {
                    "experiment_name": f"{prompt_name}_shot_{sequence_name}",
                    "prompt_name": prompt_name,
                    "prompt_content": prompt_content,
                    "report_sequence_name": sequence_name,
                    "report_sequence": report_sequence,
                    "context_note_timing": determine_context_note_timing(sequence_name),
                }
            )
    return experiments


def _append_section(parts: list[str], label: str, text: str) -> None:
    """Append ``**LABEL**: text`` to ``parts``, substituting ``NOT_AVAILABLE`` if blank."""
    normalized = text if isinstance(text, str) and text.strip() else "NOT_AVAILABLE"
    parts.append(f"**{label}**:\n\n{normalized}\n")


def prepare_report_sections(
    row: pd.Series, report_sequence: list[str], context_note_timing: str
) -> list[tuple[str, str]]:
    """Render the per-row report payload as a structured ``(label, text)`` list.

    Each entry is ``(LABEL, text)`` where ``text`` is either the
    section body or ``"NOT_AVAILABLE"``. Used by the section-aware
    truncation helper so token budgets can be allocated per section
    rather than dropping later documents wholesale.
    """
    sections: list[tuple[str, str]] = []

    def _push(label: str, text: object) -> None:
        normalized = (
            str(text) if isinstance(text, str) and text.strip() else "NOT_AVAILABLE"
        )
        sections.append((label, normalized))

    for report_type in report_sequence:
        if report_type in {"HISTOPATHOLOGY REPORT", "PRIMARY REPORT"}:
            _push(
                "HISTOPATHOLOGY REPORT",
                row.get("result_report", row.get("primary_report_text", "")),
            )
        elif report_type in {"ENDOSCOPY REPORT", "SECONDARY REPORT"}:
            _push(
                "ENDOSCOPY REPORT",
                row.get("Combined_Content", row.get("secondary_report_text", "")),
            )
        elif report_type in {"PRECEDING CLINIC LETTER", "PRECEDING CONTEXT NOTE"}:
            _push(
                "PRECEDING CLINIC LETTER",
                row.get(
                    "preceding_clinic_letter", row.get("preceding_context_note", "")
                ),
            )
        elif report_type in {"FOLLOWING CLINIC LETTER", "FOLLOWING CONTEXT NOTE"}:
            _push(
                "FOLLOWING CLINIC LETTER",
                row.get(
                    "following_clinic_letter", row.get("following_context_note", "")
                ),
            )
        elif report_type in {"CLINIC LETTER", "CONTEXT NOTE"}:
            # The bare ``CLINIC LETTER`` token is ambiguous and
            # retained only for backwards compatibility with older
            # configs. Prefer the explicit ``PRECEDING CLINIC LETTER``
            # / ``FOLLOWING CLINIC LETTER`` / ``BOTH CLINIC LETTERS``
            # alternatives below in new configurations.
            if context_note_timing == "both":
                _push(
                    "PRECEDING CLINIC LETTER",
                    row.get(
                        "preceding_clinic_letter",
                        row.get("preceding_context_note", ""),
                    ),
                )
                _push(
                    "FOLLOWING CLINIC LETTER",
                    row.get(
                        "following_clinic_letter",
                        row.get("following_context_note", ""),
                    ),
                )
            elif context_note_timing == "following":
                _push(
                    "CLINIC LETTER",
                    row.get(
                        "following_clinic_letter",
                        row.get("following_context_note", ""),
                    ),
                )
            else:
                _push(
                    "CLINIC LETTER",
                    row.get(
                        "preceding_clinic_letter",
                        row.get("preceding_context_note", ""),
                    ),
                )
        elif report_type in {"BOTH CLINIC LETTERS", "CLINIC LETTERS"}:
            _push(
                "PRECEDING CLINIC LETTER",
                row.get(
                    "preceding_clinic_letter", row.get("preceding_context_note", "")
                ),
            )
            _push(
                "FOLLOWING CLINIC LETTER",
                row.get(
                    "following_clinic_letter", row.get("following_context_note", "")
                ),
            )
    return sections


def prepare_reports_section(
    row: pd.Series, report_sequence: list[str], context_note_timing: str
) -> str:
    """Render the per-row report payload for a single experiment.

    Kept for backwards compatibility with existing tests. New callers
    should consume :func:`prepare_report_sections` instead so that
    :func:`build_message_with_budget` can perform section-aware token
    accounting.
    """
    sections = prepare_report_sections(row, report_sequence, context_note_timing)
    parts: list[str] = []
    for label, text in sections:
        _append_section(parts, label, text)
    return "\n".join(parts).strip()


def construct_combined_message(template_content: str, reports_section: str) -> str:
    """Concatenate the prompt template with the rendered reports section.

    The trailing reminder asking for a single JSON response is appended
    so that downstream parsing has a stable contract.
    """
    return f"""
{template_content}

**INPUT:**

{reports_section}

{TRAILING_REMINDER}
"""


def _encode_len(tokenizer: tiktoken.Encoding, text: str) -> int:
    """Return the token length of ``text`` under ``tokenizer``."""
    return len(tokenizer.encode(text))


def _truncate_text(tokenizer: tiktoken.Encoding, text: str, budget: int) -> str:
    """Hard-truncate ``text`` to ``budget`` tokens (returns ``""`` when ``budget <= 0``)."""
    if budget <= 0:
        return ""
    tokens = tokenizer.encode(text)
    if len(tokens) <= budget:
        return text
    return tokenizer.decode(tokens[:budget])


def build_message_with_budget(
    template_content: str,
    sections: list[tuple[str, str]],
    tokenizer: tiktoken.Encoding,
    token_limit: int,
) -> tuple[str, bool, list[str]]:
    """Section-aware truncation of the per-row request message.

    Reserves the prompt template, the ``**INPUT:**`` marker, and the
    trailing JSON reminder; allocates the remaining budget across the
    report sections so that later documents in the requested sequence
    are not silently dropped when the combined prompt exceeds
    ``token_limit``.

    The allocation is fair-share with slack redistribution: every
    section that fits within ``budget / N`` consumes only its
    actual length, and the surplus is added to the per-section budget
    of the remaining sections in declared order. Truncated sections
    are tagged in the returned list so the runner can persist a
    per-row audit trail.

    Returns ``(message, was_truncated, truncated_section_labels)``.
    """
    preamble = f"{template_content}\n\n**INPUT:**\n\n"
    suffix = f"\n\n{TRAILING_REMINDER}\n"

    preamble_tokens = _encode_len(tokenizer, preamble)
    suffix_tokens = _encode_len(tokenizer, suffix)
    reserved = preamble_tokens + suffix_tokens
    if reserved >= token_limit:
        # The non-negotiable framing alone exceeds the budget. Fall
        # back to the legacy whole-message truncation so the request
        # at least reaches the model.
        full = (
            preamble
            + "\n".join(f"**{label}**:\n\n{text}\n" for label, text in sections)
            + suffix
        )
        tokens = tokenizer.encode(full)
        if len(tokens) <= token_limit:
            return full, False, []
        return (
            tokenizer.decode(tokens[:token_limit]),
            True,
            [label for label, _ in sections],
        )

    section_budget = token_limit - reserved

    rendered_sections: list[tuple[str, str, int]] = []
    for label, text in sections:
        header = f"**{label}**:\n\n"
        trailer = "\n"
        body = text if text and text.strip() else "NOT_AVAILABLE"
        rendered_sections.append(
            (
                header,
                trailer,
                _encode_len(tokenizer, header) + _encode_len(tokenizer, trailer),
            )
        )

    truncated_labels: list[str] = []

    remaining = section_budget
    rendered: list[str] = []
    sections_left = list(range(len(sections)))
    while sections_left:
        share = remaining // len(sections_left) if remaining > 0 else 0
        index = sections_left.pop(0)
        label, text = sections[index]
        header, trailer, frame_tokens = rendered_sections[index]
        body_budget = max(0, share - frame_tokens)
        body_tokens = _encode_len(tokenizer, text)
        if body_tokens <= body_budget:
            piece = f"{header}{text}{trailer}"
            remaining -= _encode_len(tokenizer, piece)
            rendered.append(piece)
            continue
        truncated_body = _truncate_text(tokenizer, text, body_budget)
        if not truncated_body:
            truncated_body = "[TRUNCATED]"
        piece = f"{header}{truncated_body}{trailer}"
        rendered.append(piece)
        remaining -= _encode_len(tokenizer, piece)
        truncated_labels.append(label)

    message = preamble + "".join(rendered) + suffix
    was_truncated = bool(truncated_labels)
    # Safety net: if section-aware allocation still drifted over the
    # cap (e.g. due to tokenizer boundary effects), hard-trim.
    if _encode_len(tokenizer, message) > token_limit:
        tokens = tokenizer.encode(message)
        message = tokenizer.decode(tokens[:token_limit])
        was_truncated = True
    return message, was_truncated, truncated_labels
