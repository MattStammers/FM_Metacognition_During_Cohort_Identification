"""Truth-label helpers for patient-level and document-level analytics runs."""

from __future__ import annotations

import logging
import re
from typing import Iterable

import numpy as np
import pandas as pd

LOGGER = logging.getLogger("analytics_code.truth_labels")

DEFAULT_TRUTH_CANDIDATES = (
    "_document_level_truth",
    "Patient_Has_IBD",
    "ground_truth",
    "patient_has_ibd",
    "label",
    "gold_label",
)
DOCUMENT_TRUTH_COLUMN = "_document_level_truth"

_DOC_TYPE_ALIASES: dict[str, tuple[tuple[str, ...], ...]] = {
    "preceding_clinic": (
        ("preceding", "clinic"),
        ("clinic", "preceding"),
        ("clinic", "pre"),
        ("preceding", "letter"),
        ("clinic_pre",),
        ("preceding_clinic",),
    ),
    "following_clinic": (
        ("following", "clinic"),
        ("clinic", "following"),
        ("clinic", "follow"),
        ("following", "letter"),
        ("clinic_follow",),
        ("following_clinic",),
    ),
    "endoscopy": (
        ("endoscopy",),
        ("endo",),
    ),
    "histology": (
        ("histology",),
        ("histopathology",),
        ("histo",),
        ("hist",),
    ),
}
_LABEL_HINTS = {
    "ibd",
    "truth",
    "label",
    "flag",
    "gold",
    "ground",
    "target",
    "positive",
    "has",
    "present",
}
_SEQUENCE_DOCS: dict[str, tuple[str, ...]] = {
    "all_docs_in_sequence": (
        "preceding_clinic",
        "endoscopy",
        "histology",
        "following_clinic",
    ),
    "all_docs_in_sequence_with_clinic_following": (
        "preceding_clinic",
        "endoscopy",
        "histology",
        "following_clinic",
    ),
    "all_docs_in_reverse_sequence": (
        "following_clinic",
        "histology",
        "endoscopy",
        "preceding_clinic",
    ),
    "endo_hist": ("endoscopy", "histology"),
    "hist_endo": ("histology", "endoscopy"),
    "hist": ("histology",),
    "endo": ("endoscopy",),
    "clinic_preceding": ("preceding_clinic",),
    "clinic_following": ("following_clinic",),
    "clinic_both": ("preceding_clinic", "following_clinic"),
    "clinic_pre_clinic_follow": ("preceding_clinic", "following_clinic"),
    "clinic_endo_hist": ("preceding_clinic", "endoscopy", "histology"),
    "endo_clinic_pre_hist": ("endoscopy", "preceding_clinic", "histology"),
    "endo_hist_clinic": ("endoscopy", "histology", "preceding_clinic"),
    "endo_hist_clinic_pre": ("endoscopy", "histology", "preceding_clinic"),
    "hist_clinic_endo": ("histology", "preceding_clinic", "endoscopy"),
    "hist_endo_clinic_pre": ("histology", "endoscopy", "preceding_clinic"),
    "jumbled": (
        "following_clinic",
        "endoscopy",
        "preceding_clinic",
        "histology",
    ),
    "clinic_follow_endo_clinic_pre_hist": (
        "following_clinic",
        "endoscopy",
        "preceding_clinic",
        "histology",
    ),
}


def truth_mode(config) -> str:
    """Return the configured truth mode.

    Recognised values:

    * ``"patient"`` (default) -- patient-level gold label.
    * ``"document"`` -- per-row OR-of-active-documents truth label;
      a row is positive if any relevant document marker is positive,
      negative if at least one relevant marker is known and none are
      positive, missing only if all relevant markers are unavailable.
    * ``"document_complete"`` -- sensitivity variant of ``document``
      that restricts the analysis to rows whose relevant document
      markers are *all* present (no missing markers in the sequence).
    """
    mode = str(config.analysis.get("truth_mode", "patient")).strip().lower()
    return mode if mode in {"patient", "document", "document_complete"} else "patient"


def detect_truth_column(
    frame: pd.DataFrame, candidates: Iterable[str] | None = None
) -> str | None:
    """Return the first truth column name present in ``frame``."""
    search = tuple(candidates or DEFAULT_TRUTH_CANDIDATES)
    for candidate in search:
        if candidate in frame.columns:
            return candidate
    return None


def prepare_truth_frame(
    frame: pd.DataFrame,
    *,
    mode: str = "patient",
    candidates: Iterable[str] | None = None,
) -> tuple[pd.DataFrame, str | None]:
    """Return a frame ready for analysis and the chosen truth column name."""
    if frame.empty:
        return frame, None
    if mode not in {"document", "document_complete"}:
        return frame, detect_truth_column(frame, candidates)

    out = frame.copy()
    strict = mode == "document_complete"
    document_truth = build_document_truth_series(out, strict=strict)
    if document_truth.notna().any():
        out[DOCUMENT_TRUTH_COLUMN] = document_truth
        return out, DOCUMENT_TRUTH_COLUMN

    LOGGER.warning(
        "Document-level truth mode selected but no usable document marker columns were resolved"
    )
    return out, None


def build_document_truth_series(
    frame: pd.DataFrame, *, strict: bool = False
) -> pd.Series:
    """Build a per-row OR-of-active-documents truth label from marker columns.

    When ``strict`` is ``True`` (the ``document_complete`` sensitivity
    arm), rows whose relevant document markers are not *all* present
    are emitted as ``NaN`` so downstream analyses restrict to rows
    with complete document coverage.
    """
    if "report_sequence_name" not in frame.columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")

    doc_columns = _resolve_document_marker_columns(frame)
    if not doc_columns:
        return pd.Series(np.nan, index=frame.index, dtype="float64")

    binary_columns = {
        doc_type: _coerce_binary_series(frame[column])
        for doc_type, column in doc_columns.items()
    }
    truth = pd.Series(np.nan, index=frame.index, dtype="float64")
    for sequence, idx in frame.groupby(
        "report_sequence_name", dropna=False
    ).groups.items():
        active_docs = _document_types_for_sequence(sequence)
        available = [
            binary_columns[doc] for doc in active_docs if doc in binary_columns
        ]
        if not available:
            continue
        values = pd.concat(available, axis=1)
        values = values.loc[idx]
        any_true = values.eq(1).any(axis=1)
        any_known = values.notna().any(axis=1)
        if strict:
            # Sensitivity variant: restrict to rows whose relevant
            # markers are ALL present (no missing markers).
            all_known = values.notna().all(axis=1)
            row_truth = np.where(all_known, np.where(any_true, 1.0, 0.0), np.nan)
        else:
            row_truth = np.where(any_true, 1.0, np.where(any_known, 0.0, np.nan))
        truth.loc[idx] = row_truth
    return truth


def _resolve_document_marker_columns(frame: pd.DataFrame) -> dict[str, str]:
    """Pick the most plausible boolean-like marker column for each document type."""
    resolved: dict[str, str] = {}
    for doc_type, aliases in _DOC_TYPE_ALIASES.items():
        best_score = -1
        best_column: str | None = None
        for column in frame.columns:
            if not _is_boolean_like(frame[column]):
                continue
            if not _has_label_hint(column):
                continue
            score = _column_match_score(column, aliases)
            if score > best_score:
                best_score = score
                best_column = column
        if best_column is not None and best_score >= 1:
            resolved[doc_type] = best_column
    return resolved


def _has_label_hint(column: object) -> bool:
    """Return ``True`` when ``column`` looks like an explicit truth/flag field."""
    return bool(_LABEL_HINTS.intersection(_tokens(column)))


def _column_match_score(column: object, aliases: tuple[tuple[str, ...], ...]) -> int:
    """Return a heuristic match score for a document-marker column."""
    tokens = _tokens(column)
    if not tokens:
        return -1
    label_bonus = 10 if _LABEL_HINTS.intersection(tokens) else 0
    best = -1
    compact = str(column).strip().lower()
    for alias in aliases:
        if len(alias) == 1 and alias[0] in compact:
            best = max(best, 1 + label_bonus)
        elif all(part in tokens for part in alias):
            best = max(best, len(alias) + label_bonus)
    return best


def _document_types_for_sequence(sequence: object) -> tuple[str, ...]:
    """Infer which document markers are relevant for a report sequence slug."""
    raw = str(sequence or "").strip().lower()
    if not raw:
        return ()
    if raw in _SEQUENCE_DOCS:
        return _SEQUENCE_DOCS[raw]

    tokens = _tokens(raw)
    docs: list[str] = []
    if {"all", "docs", "sequence"}.issubset(tokens):
        docs.extend(("preceding_clinic", "endoscopy", "histology"))
        if {"follow", "following"}.intersection(tokens):
            docs.append("following_clinic")
        elif raw in {"all_docs_in_sequence", "all_docs_in_reverse_sequence"}:
            docs.append("following_clinic")
    if {"endo", "endoscopy"}.intersection(tokens):
        docs.append("endoscopy")
    if {"hist", "histo", "histology", "histopathology"}.intersection(tokens):
        docs.append("histology")
    if "clinic" in tokens:
        if {"follow", "following"}.intersection(tokens):
            docs.append("following_clinic")
        elif {"pre", "preceding"}.intersection(tokens):
            docs.append("preceding_clinic")
        else:
            docs.append("preceding_clinic")
    seen: list[str] = []
    for item in docs:
        if item not in seen:
            seen.append(item)
    return tuple(seen)


def _tokens(value: object) -> set[str]:
    """Tokenize a column name or sequence slug into lowercase alphanumerics."""
    text = str(value).strip().lower().replace("-", "_")
    parts = re.findall(r"[a-z0-9]+", text)
    return set(parts)


def _coerce_binary_series(series: pd.Series) -> pd.Series:
    """Coerce a heterogeneous marker series into 0/1/NaN floats."""
    normalized = series.replace(
        {
            True: 1,
            False: 0,
            "true": 1,
            "false": 0,
            "TRUE": 1,
            "FALSE": 0,
            "yes": 1,
            "no": 0,
            "Y": 1,
            "N": 0,
        }
    )
    numeric = pd.to_numeric(normalized, errors="coerce")
    numeric = numeric.where(numeric.isin([0, 1]))
    return numeric.astype("float64")


def _is_boolean_like(series: pd.Series) -> bool:
    """Return ``True`` when the non-null values can be read as a 0/1 flag."""
    numeric = _coerce_binary_series(series)
    if numeric.notna().sum() == 0:
        return False
    return bool(numeric.dropna().isin([0, 1]).all())
