"""Shared helpers for the analytics pipeline.

This module centralises low-level utilities used by every pipeline
stage: package logging, file-system helpers, dataframe / JSON / figure
sinks, and small column-handling utilities such as
:func:`canonicalize_model`. It also defines the experimental-matrix
constants (FAIR temperatures, zero-shot data types, the zero-shot
label) that the comparison stages share so that every per-factor
analysis is reported on a consistent sub-cube of the experiment.
"""

from __future__ import annotations

import json
import logging
import os
import re
import shutil
import stat
import time
from pathlib import Path
from typing import Any, Iterable

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

LOGGER = logging.getLogger("analytics_code")

# ---------------------------------------------------------------------------
# Experimental-matrix constants
# ---------------------------------------------------------------------------
#
# Every per-factor comparison published by the pipeline (in
# :mod:`analytics_code.full_performance` and
# :mod:`analytics_code.dropout_analysis`) is reported on the sub-cube
# defined by these constants. Both stages must use the same constants
# so that, e.g., the model ranking in the breakdown workbook can be
# cross-referenced against the model ranking in the FAIR performance
# CSV without having to reconcile silently different filter rules.
#
# The production matrix only crosses each model with a single
# temperature (mixtral / m42 / deepseek14 at 0.75; deepseek32 / qwen32
# at 0.60), so a perfectly balanced sub-cube does not exist and the
# ``temperature`` comparison is intrinsically confounded with model.
# This is documented in the per-factor docstrings below.

#: Temperatures used by the production model servers. Per-factor
#: comparisons that are *not* studying temperature are restricted to
#: this set so that off-matrix experimental temperatures cannot bias
#: the result.
FAIR_TEMPERATURES_NUMERIC: tuple[float, ...] = (0.50, 0.60, 0.75, 1.00)

#: String form of :data:`FAIR_TEMPERATURES_NUMERIC` matching the
#: ``_t<temp>`` slug embedded in batch folder names (e.g. ``"0_60"``,
#: ``"0_75"``).
FAIR_TEMPERATURE_LABELS: tuple[str, ...] = ("0_50", "0_60", "0_75", "1_00")

#: Canonical label used for the zero-shot prompt template throughout
#: the pipeline.
ZERO_SHOT_LABEL: str = "zero"

#: Data types (``report_sequence_name`` values) that are run for every
#: shot type in the production matrix. Per-factor comparisons that are
#: not studying ``data_type`` restrict to this list to keep the source
#: documents balanced across the other factors.
FAIR_DATA_TYPES: tuple[str, ...] = (
    "all_docs_in_sequence",
    "endo",
    "hist",
    "clinic_preceding",
    "clinic_following",
)

# ---------------------------------------------------------------------------
# Tier definitions for the published "Document / Cumulative / Final /
# Doc2Patient" analysis (mirrors Table 3.2.4-1 of the paper).
# ---------------------------------------------------------------------------
#
# Document tier: single-physician-marker rows. Each row corresponds to
# one report type and is scored against the matching physician flag.
SINGLE_DOC_DATA_TYPES: tuple[str, ...] = (
    "hist",
    "endo",
    "clinic_preceding",
    "clinic_following",
)

#: Mapping from canonical document type to the physician-marker column
#: in the human reference workbook. The dummy reference file
#: (``pseudonymised_dummy_reference_document_flags.csv``) provides the
#: canonical column names; real-data configs can override via
#: ``analysis.document_flag_columns``.
DOC_FLAG_COLUMNS: dict[str, str] = {
    "histology": "histology_ibd_flag",
    "endoscopy": "endoscopy_ibd_flag",
    "preceding_clinic": "preceding_clinic_ibd_flag",
    "following_clinic": "following_clinic_ibd_flag",
}

#: Mapping from each single-doc ``report_sequence_name`` value to the
#: canonical doc type that row represents. Used by the Document tier
#: to pick the right physician marker per row.
SINGLE_DOC_TYPE_MAP: dict[str, str] = {
    "hist": "histology",
    "endo": "endoscopy",
    "clinic_preceding": "preceding_clinic",
    "clinic_following": "following_clinic",
}

#: Explicit data_type -> set of physician-marker doc types shown in
#: that prompt. Drives the Cumulative-tier reference (logical OR over
#: the documents actually shown to the model in that data_type). The
#: three-document ``*_clinic*`` sequences in the dummy prompt configs
#: include a single "CLINIC LETTER" slot but the methodology treats
#: that as evidence from either clinic letter, so both clinic markers
#: are OR'd (matches the "any IBD evidence in any linked doc" intent).
DATA_TYPE_DOCUMENT_SET: dict[str, tuple[str, ...]] = {
    # Single-doc data types -- one physician marker each.
    "hist": ("histology",),
    "endo": ("endoscopy",),
    "clinic_preceding": ("preceding_clinic",),
    "clinic_following": ("following_clinic",),
    # Two-doc data types.
    "endo_hist": ("endoscopy", "histology"),
    "hist_endo": ("histology", "endoscopy"),
    "clinic_both": ("preceding_clinic", "following_clinic"),
    # Three-doc data types -- single "CLINIC LETTER" slot maps to
    # OR(preceding, following) per the methodology.
    "clinic_endo_hist": (
        "preceding_clinic",
        "following_clinic",
        "endoscopy",
        "histology",
    ),
    "endo_hist_clinic": (
        "endoscopy",
        "histology",
        "preceding_clinic",
        "following_clinic",
    ),
    "hist_endo_clinic": (
        "histology",
        "endoscopy",
        "preceding_clinic",
        "following_clinic",
    ),
    "hist_clinic_endo": (
        "histology",
        "preceding_clinic",
        "following_clinic",
        "endoscopy",
    ),
    # Four-doc / full bundle data types.
    "all_docs_in_sequence": (
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
    "jumbled": (
        "following_clinic",
        "endoscopy",
        "preceding_clinic",
        "histology",
    ),
}

#: Cumulative-tier data types -- everything in
#: :data:`DATA_TYPE_DOCUMENT_SET` that is not a single-doc row.
MULTI_DOC_DATA_TYPES: tuple[str, ...] = tuple(
    dt for dt in DATA_TYPE_DOCUMENT_SET if dt not in SINGLE_DOC_DATA_TYPES
)


def tier_of(data_type: object) -> str:
    """Classify ``data_type`` into ``"document"``, ``"cumulative"`` or ``"unknown"``.

    Used by :mod:`analytics_code.tiered_performance` to route each row
    of the merged outputs to the right tier.
    """
    key = str(data_type).strip().lower() if data_type is not None else ""
    if key in SINGLE_DOC_DATA_TYPES:
        return "document"
    if key in DATA_TYPE_DOCUMENT_SET:
        return "cumulative"
    return "unknown"


PUBLICATION_DPI: int = 600

MODEL_DISPLAY_NAMES: dict[str, str] = {
    "mixtral7b": "Mixtral-7B (2024)",
    "mixtral_7b": "Mixtral-7B (2024)",
    "mixtral-7b": "Mixtral-7B (2024)",
    "m42_8b": "M42-8B (2024)",
    "m42-8b": "M42-8B (2024)",
    "deepseek14b": "DeepSeek-14B (2025)",
    "deepseek_14b": "DeepSeek-14B (2025)",
    "deepseek-14b": "DeepSeek-14B (2025)",
    "deepseek32b": "DeepSeek-32B (2025)",
    "deepseek_32b": "DeepSeek-32B (2025)",
    "deepseek-32b": "DeepSeek-32B (2025)",
    "deepseek70b": "DeepSeek-70B (2025)",
    "gpt-oss-20b": "GPT-OSS-20B",
    "gemma4_e2b": "Gemma-4-E2B",
    "gemma-4-e2b": "Gemma-4-E2B",
    "gemma4_e4b": "Gemma-4-E4B",
    "gemma-4-e4b": "Gemma-4-E4B",
    "gemma4_26b_a4b": "Gemma-4-26B-A4B",
    "gemma-4-26b-a4b": "Gemma-4-26B-A4B",
    "gemma4_31b": "Gemma-4-31B (2026)",
    "gemma-4-31b": "Gemma-4-31B (2026)",
    "qwen32b": "Qwen-32B (2025)",
    "qwen_32b": "Qwen-32B (2025)",
    "qwen-32b": "Qwen-32B (2025)",
}

MODEL_COLOR_MAP: dict[str, str] = {
    # Non-thinking (2024) family -- warm/red-purple tones.
    "Mixtral-7B (2024)": "#117733",
    "M42-8B (2024)": "#882255",
    # Thinking (2025) family -- cool blues/teal tones.
    "DeepSeek-14B (2025)": "#332288",
    "DeepSeek-32B (2025)": "#44AA99",
    "DeepSeek-70B (2025)": "#88CCEE",
    "Qwen-32B (2025)": "#999933",
    # Gemma 4 (2026) -- distinctive green to mark the 2026 generation.
    "Gemma-4-31B (2026)": "#2CA02C",
    # Legacy / exploratory variants (not in the methodology matrix).
    "GPT-OSS-20B": "#AA4499",
    "Gemma-4-E2B": "#CC6677",
    "Gemma-4-E4B": "#DDCC77",
    "Gemma-4-26B-A4B": "#661100",
}

THINKING_POOL_COLORS: dict[str, str] = {
    "Thinking (2025)": "#0072B2",
    "Non-thinking (2024)": "#D55E00",
    "Gemma (2026)": "#2CA02C",
}

SEQUENCE_DISPLAY_NAMES: dict[str, str] = {
    "all_docs_in_sequence": "Preceding clinic letter / Endoscopy / Histology narrative order",
    "all_docs_in_sequence_with_clinic_following": "Preceding clinic letter / Endoscopy / Histology / Following clinic letter narrative order",
    "all_docs_in_reverse_sequence": "Following clinic letter / Histology / Endoscopy / Preceding clinic letter narrative order",
    "endo_hist": "Endoscopy / Histology narrative order",
    "hist": "Histology narrative order",
    "clinic_endo_hist": "Preceding clinic letter / Endoscopy / Histology narrative order",
    "endo_clinic_pre_hist": "Endoscopy / Preceding clinic letter / Histology narrative order",
    "hist_endo": "Histology / Endoscopy narrative order",
    "endo": "Endoscopy narrative order",
    "endo_hist_clinic": "Endoscopy / Histology / Preceding clinic letter narrative order",
    "endo_hist_clinic_pre": "Endoscopy / Histology / Preceding clinic letter narrative order",
    "clinic_both": "Preceding clinic letter / Following clinic letter narrative order",
    "clinic_pre_clinic_follow": "Preceding clinic letter / Following clinic letter narrative order",
    "clinic_following": "Following clinic letter narrative order",
    "jumbled": "Following clinic letter / Endoscopy / Preceding clinic letter / Histology narrative order",
    "clinic_follow_endo_clinic_pre_hist": "Following clinic letter / Endoscopy / Preceding clinic letter / Histology narrative order",
    "clinic_preceding": "Preceding clinic letter narrative order",
    "hist_clinic_endo": "Histology / Endoscopy / Preceding clinic letter narrative order",
    "hist_endo_clinic_pre": "Histology / Endoscopy / Preceding clinic letter narrative order",
}

SHOT_DISPLAY_NAMES: dict[str, str] = {
    "zero": "Zero shot",
    "single": "Single shot",
    "dual": "Dual shot",
}


def sentence_case(text: object) -> str:
    """Return a readable sentence-case label derived from ``text``."""
    value = str(text).strip()
    if not value:
        return ""
    normalized = value.replace("_", " ").replace("-", " ")
    normalized = re.sub(r"\s+", " ", normalized).strip()
    lower = normalized.lower()
    return lower[:1].upper() + lower[1:]


def format_model_display_name(model_name: object) -> str:
    """Return the publication label for a model name."""
    key = str(model_name).strip()
    if not key:
        return ""
    low = key.lower()
    if low in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[low]
    compact = low.replace(" ", "").replace("_", "-")
    if compact in MODEL_DISPLAY_NAMES:
        return MODEL_DISPLAY_NAMES[compact]
    match = re.match(r"([a-z]+)[_-]?(\d+)([bm])", low)
    if match:
        family, size, suffix = match.groups()
        family_title = family[:1].upper() + family[1:]
        return f"{family_title}-{size}{suffix.upper()}"
    return sentence_case(key)


def format_temperature_label(value: object) -> str:
    """Return a normalised temperature label."""
    if pd.isna(value):
        return "Missing"
    try:
        return f"{float(value):.2f}".rstrip("0").rstrip(".")
    except (TypeError, ValueError):
        raw = str(value).strip()
        if re.fullmatch(r"\d+_\d+", raw):
            return raw.replace("_", ".")
        return raw


def format_shot_label(value: object) -> str:
    """Return a readable shot-type label."""
    raw = str(value).strip().lower()
    return SHOT_DISPLAY_NAMES.get(raw, sentence_case(raw))


def format_data_type_label(value: object) -> str:
    """Return a short sentence-case label for data-type charts."""
    raw = str(value).strip()
    if not raw:
        return ""
    return sentence_case(raw.replace("_", " "))


def format_sequence_label(value: object) -> str:
    """Return a readable label for a report-sequence / context slug."""
    raw = str(value).strip()
    low = raw.lower()
    if low in SEQUENCE_DISPLAY_NAMES:
        return SEQUENCE_DISPLAY_NAMES[low]
    parts = [part for part in re.split(r"[_/]+", low) if part]
    token_map = {
        "all": "All",
        "docs": "documents",
        "in": "in",
        "sequence": "sequence",
        "endo": "Endoscopy",
        "endoscopy": "Endoscopy",
        "hist": "Histology",
        "histology": "Histology",
        "clinic": "Clinic letter",
        "preceding": "Preceding",
        "pre": "Preceding",
        "following": "Following",
        "follow": "Following",
        "report": "report",
        "letter": "letter",
    }
    if not parts:
        return ""
    translated = [token_map.get(part, sentence_case(part)) for part in parts]
    joined = " / ".join(translated)
    joined = re.sub(r"\bPreceding / Clinic letter\b", "Preceding clinic letter", joined)
    joined = re.sub(r"\bFollowing / Clinic letter\b", "Following clinic letter", joined)
    joined = re.sub(
        r"\bAll / documents / in / sequence\b", "All documents in sequence", joined
    )
    if re.search(
        r"\b(Endoscopy|Histology|clinic letter)\b", joined, flags=re.IGNORECASE
    ):
        joined = f"{joined} narrative order"
    joined = re.sub(r"\s+", " ", joined).strip()
    return joined[:1].upper() + joined[1:] if joined else joined


def format_factor_value(factor: str, value: object) -> str:
    """Format a factor value according to its semantic type."""
    if factor in {"model", "model_canon"}:
        return format_model_display_name(value)
    if factor in {"temperature"}:
        return format_temperature_label(value)
    if factor in {"shot", "shot_type"}:
        return format_shot_label(value)
    if factor in {"data_type"}:
        return format_data_type_label(value)
    if factor in {"report_sequence_name", "dataset"}:
        return format_sequence_label(value)
    return sentence_case(value)


def model_color(model_name: object, fallback: str = "#4C78A8") -> str:
    """Return the canonical colour for a model label."""
    display = format_model_display_name(model_name)
    return MODEL_COLOR_MAP.get(display, fallback)


def apply_publication_style() -> None:
    """Apply the shared publication-ready matplotlib style."""
    plt.rcParams.update(
        {
            "figure.dpi": PUBLICATION_DPI,
            "savefig.dpi": PUBLICATION_DPI,
            "font.family": "DejaVu Sans",
            "font.size": 10,
            "axes.titlesize": 13,
            "axes.titleweight": "semibold",
            "axes.labelsize": 11,
            "axes.edgecolor": "#111111",
            "axes.linewidth": 1.2,
            "xtick.labelsize": 9,
            "ytick.labelsize": 9,
            "legend.fontsize": 9,
            "grid.alpha": 0.35,
        }
    )


def set_publication_axes(
    ax, *, show_grid_y: bool = True, show_grid_x: bool = False
) -> None:
    """Apply consistent axis styling to ``ax``."""
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_linewidth(1.2)
        ax.spines[side].set_color("#111111")
    if show_grid_y:
        ax.grid(True, axis="y", linestyle="--", linewidth=0.6, alpha=0.18)
    if show_grid_x:
        ax.grid(True, axis="x", linestyle="--", linewidth=0.6, alpha=0.18)
    ax.tick_params(axis="both", which="major", length=4, width=1.0, color="#111111")


def nice_numeric_ticks(min_value: float, max_value: float, step: float) -> np.ndarray:
    """Return evenly spaced ticks covering ``[min_value, max_value]``."""
    if not np.isfinite(min_value) or not np.isfinite(max_value):
        return np.array([])
    start = step * np.floor(min_value / step)
    stop = step * np.ceil(max_value / step)
    if np.isclose(start, stop):
        stop = start + step
    count = int(round((stop - start) / step)) + 1
    return np.round(np.linspace(start, stop, count), 10)


def setup_logging() -> None:
    """Configure root logging with the pipeline's standard format.

    Sets the root logger to ``INFO`` and installs a single stream
    handler with the ``timestamp | level | logger | message`` layout
    used by every stage.
    """
    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s | %(levelname)s | %(name)s | %(message)s",
    )
    apply_publication_style()


def ensure_dir(path: Path) -> Path:
    """Create ``path`` (including parents) and return it.

    The call is idempotent; existing directories are left untouched.
    """
    path.mkdir(parents=True, exist_ok=True)
    return path


def remove_tree(path: Path, *, retries: int = 5, delay_seconds: float = 0.2) -> None:
    """Remove ``path`` recursively, retrying transient Windows lock failures."""
    if not path.exists():
        return

    def _onerror(func, failed_path, exc_info) -> None:
        try:
            os.chmod(failed_path, stat.S_IWRITE)
        except OSError:
            pass
        func(failed_path)

    last_error: Exception | None = None
    for attempt in range(retries):
        try:
            shutil.rmtree(path, onerror=_onerror)
            return
        except PermissionError as exc:
            last_error = exc
            if attempt == retries - 1:
                raise
            time.sleep(delay_seconds)
    if last_error is not None:
        raise last_error


# Prefer .xlsx for combined exports, switch to .parquet once the
# serialised payload exceeds this many bytes.
PARQUET_SIZE_THRESHOLD_BYTES = 10 * 1024 * 1024


def write_dataframe(dataframe: pd.DataFrame, path: Path, index: bool = False) -> Path:
    """Write ``dataframe`` to ``path`` as CSV, XLSX or Parquet.

    The output format is determined by the file suffix: ``.xlsx`` is
    written via :meth:`pandas.DataFrame.to_excel`, ``.parquet`` via
    :meth:`pandas.DataFrame.to_parquet`; everything else uses
    :meth:`pandas.DataFrame.to_csv`. Parent directories are created on
    demand.

    When the requested suffix is ``.xlsx`` and the estimated payload
    exceeds :data:`PARQUET_SIZE_THRESHOLD_BYTES` (10 MB), the file is
    silently promoted to ``.parquet`` in line with the methodology
    specification, and the rewritten path is returned.

    Parameters
    ----------
    dataframe:
        Frame to serialise.
    path:
        Destination path. The parent directory is created if missing.
    index:
        Whether to write the dataframe index. Defaults to ``False``.

    Returns
    -------
    pathlib.Path
        The path that was actually written (may differ from ``path`` if
        an XLSX target was promoted to Parquet for size reasons).
    """
    ensure_dir(path.parent)
    suffix = path.suffix.lower()

    def _coerce_for_parquet(frame: pd.DataFrame) -> pd.DataFrame:
        # Parquet requires homogeneous types per column. Coerce object
        # columns that contain mixed types (e.g. mingled ints and
        # strings in patient_id) to plain strings.
        out = frame.copy()
        for col in out.columns:
            if out[col].dtype == object:
                out[col] = out[col].where(out[col].isna(), out[col].astype(str))
        return out

    if suffix == ".xlsx":
        estimated_bytes = int(dataframe.memory_usage(index=index, deep=True).sum())
        if estimated_bytes > PARQUET_SIZE_THRESHOLD_BYTES:
            promoted = path.with_suffix(".parquet")
            LOGGER.info(
                "Promoting %s to Parquet (estimated %.1f MB > 10 MB threshold)",
                path.name,
                estimated_bytes / (1024 * 1024),
            )
            _coerce_for_parquet(dataframe).to_parquet(promoted, index=index)
            LOGGER.info("Wrote %s rows to %s", len(dataframe), promoted)
            return promoted
        dataframe.to_excel(path, index=index)
    elif suffix == ".parquet":
        _coerce_for_parquet(dataframe).to_parquet(path, index=index)
    else:
        dataframe.to_csv(path, index=index)
    LOGGER.info("Wrote %s rows to %s", len(dataframe), path)
    return path


def write_json(data: dict[str, Any] | list[dict[str, Any]], path: Path) -> Path:
    """Write ``data`` as pretty-printed UTF-8 JSON to ``path``.

    Parent directories are created on demand. Returns ``path`` for
    chaining.
    """
    ensure_dir(path.parent)
    path.write_text(json.dumps(data, indent=2), encoding="utf-8")
    LOGGER.info("Wrote JSON to %s", path)
    return path


def save_figure(path: Path) -> Path:
    """Persist the *current* matplotlib figure to ``path``.

    Applies :func:`matplotlib.pyplot.tight_layout`, saves at 200 DPI
    with a tight bounding box and then closes the active figure to free
    memory. Parent directories are created on demand.
    """
    ensure_dir(path.parent)
    plt.tight_layout()
    plt.savefig(path, dpi=PUBLICATION_DPI, bbox_inches="tight", pad_inches=0.2)
    plt.close()
    LOGGER.info("Saved figure to %s", path)
    return path


def first_existing(items: Iterable[str], candidates: Iterable[str]) -> str | None:
    """Return the first entry in ``candidates`` that also appears in ``items``.

    Useful for picking the first available column from a list of
    aliases. Returns ``None`` when no candidate matches.
    """
    item_set = {str(item) for item in items}
    for candidate in candidates:
        if candidate in item_set:
            return candidate
    return None


def canonicalize_model(model_name: str, mapping: dict[str, str]) -> str:
    """Map a raw model name onto its canonical label.

    The ``mapping`` argument is taken from ``config.model_mapping``;
    unknown names are returned unchanged after stripping whitespace.
    """
    stripped = str(model_name).strip()
    if stripped in mapping:
        return mapping[stripped]

    lowered = stripped.lower()
    lowered_mapping = {
        str(key).strip().lower(): value for key, value in mapping.items()
    }
    if lowered in lowered_mapping:
        return lowered_mapping[lowered]

    family_patterns: tuple[tuple[str, str], ...] = (
        (r"deepseek.*14", "deepseek14b"),
        (r"deepseek.*32", "deepseek32b"),
        (r"deepseek.*70", "deepseek70b"),
        (r"mixtral", "mixtral7b"),
        (r"m42", "m42_8b"),
        (r"qwen.*32", "qwen32b"),
        (r"gemma.*31", "gemma4_31b"),
        (r"gemma.*26", "gemma4_26b_a4b"),
        (r"gemma.*e2b", "gemma4_e2b"),
        (r"gemma.*e4b", "gemma4_e4b"),
    )
    for pattern, canonical in family_patterns:
        if re.search(pattern, lowered):
            return canonical
    return stripped
