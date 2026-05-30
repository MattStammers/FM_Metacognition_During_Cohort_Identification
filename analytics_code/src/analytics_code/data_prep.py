"""Data preparation stage.

This produces the base data model and prepares
the total cohort and LLM JSON processing summaries for downstream stages.
If a human reference dataset is provided, the merge with reference and
corresponding summary are also produced.
"""

from __future__ import annotations

import re
from pathlib import Path
from typing import Any

import numpy as np
import pandas as pd

from analytics_code.common import (
    MULTI_DOC_DATA_TYPES,
    SINGLE_DOC_DATA_TYPES,
    canonicalize_model,
    ensure_dir,
    write_dataframe,
    write_json,
)
from analytics_code.config import AnalysisConfig
from analytics_code.json_repair import REPAIR_TYPES, parse_response
from analytics_code.predictions import clean_likelihood
from analytics_code.truth_labels import build_document_truth_series

#: Lenient document-sequence truth column materialised into
#: ``merged_outputs.csv``. Positive if any relevant document marker
#: for the row's sequence is positive; negative if at least one
#: relevant marker is observed and none are positive; missing only
#: when every relevant marker is missing.
DOCUMENT_SEQUENCE_TRUTH_LENIENT = "_document_sequence_truth_lenient"

#: Strict (complete-marker) document-sequence truth column. Same
#: positive rule as the lenient variant, but missing whenever *any*
#: relevant marker for the sequence is missing. Sensitivity label
#: only -- never used as the default truth.
DOCUMENT_SEQUENCE_TRUTH_STRICT = "_document_sequence_truth_strict"

#: Patient-level document-derived truth: logical OR over the four
#: physician document markers for each ``patient_id``. Constant within
#: a patient and used as the reference standard for the Final_* tier
#: in :mod:`analytics_code.tiered_performance`.
PATIENT_DOCUMENT_TRUTH = "_patient_document_truth"

JSON_PATTERN = re.compile(r"\{.*\}", re.DOTALL)
PROMPT_PATTERN = re.compile(r"^(zero|single|dual)_shot_(.+)$")
TEMP_PATTERN = re.compile(r"(?:_t0?_?|_t)(\d{2,3})$")
SECTION_PATTERN = re.compile(
    r"\*\*CLUES:\*\*\s*(.*?)\s*\*\*REASONING:\*\*\s*(.*?)\s*\*\*OUTCOME:\*\*",
    re.DOTALL,
)


def find_batch_files(batch_root: Path, batch_glob: str) -> list[Path]:
    """Return a sorted list of LLM batch files under ``batch_root``.

    Parameters
    ----------
    batch_root:
        Root directory to scan.
    batch_glob:
        Glob pattern, evaluated relative to ``batch_root`` (for example
        ``"**/batch_*.xlsx"``).
    """
    return sorted(path for path in batch_root.glob(batch_glob) if path.is_file())


def combine_batch_exports(batch_files: list[Path]) -> pd.DataFrame:
    """Concatenate per-batch CSV/XLSX exports into a single dataframe.

    Each input frame is annotated with three provenance columns derived
    from the file path: ``source_file``, ``source_parent`` /
    ``experiment_name`` (the parent directory name) and ``runner_name``
    (the grand-parent directory name).

    Returns an empty :class:`~pandas.DataFrame` when no files are
    supplied.
    """
    frames: list[pd.DataFrame] = []
    for path in batch_files:
        suffix = path.suffix.lower()
        if suffix in {".xlsx", ".xls"}:
            frame = pd.read_excel(path)
        elif suffix == ".parquet":
            frame = pd.read_parquet(path)
        else:
            frame = pd.read_csv(path)
        frame = frame.copy()
        frame["source_file"] = path.name
        frame["source_parent"] = path.parent.name
        frame["experiment_name"] = path.parent.name
        frame["runner_name"] = path.parent.parent.name
        frames.append(frame)
    return pd.concat(frames, ignore_index=True) if frames else pd.DataFrame()


def _infer_temperature(value: str, runner_metadata: dict[str, Any]) -> float | None:
    """Infer a sampling temperature from a model name or runner metadata.

    Looks the value up in ``runner_metadata`` first; if no temperature
    is recorded there, falls back to parsing a ``_t075`` / ``_t0_75``
    style suffix from ``value``.
    """
    metadata_value = (
        runner_metadata.get(value, {}).get("temperature")
        if isinstance(runner_metadata.get(value, {}), dict)
        else None
    )
    if metadata_value is not None:
        return float(metadata_value)
    match = TEMP_PATTERN.search(value)
    if not match:
        return None
    digits = match.group(1)
    if len(digits) == 2:
        return float(f"0.{digits}")
    if len(digits) == 3:
        return (
            float(f"{digits[0]}.{digits[1:]}")
            if digits[0] != "0"
            else float(f"0.{digits[1:]}")
        )
    return None


def _extract_prompt_sections(text: Any) -> tuple[str, str]:
    """Extract the ``CLUES`` and ``REASONING`` sections from a model response.

    Returns ``(clues, reasoning)`` as stripped strings, or two empty
    strings when the expected markers are not present.
    """
    if pd.isna(text):
        return "", ""
    match = SECTION_PATTERN.search(str(text))
    if not match:
        return "", ""
    return match.group(1).strip(), match.group(2).strip()


def _response_column_map(columns: list[str]) -> dict[str, list[str]]:
    """Group the per-runner response columns by their semantic role.

    Returns a mapping with keys ``json``, ``full``, ``payload`` and
    ``truncated``; each value is a list of matching column names.
    """
    mapping: dict[str, list[str]] = {
        "json": [],
        "full": [],
        "payload": [],
        "truncated": [],
    }
    for column in columns:
        if "_Json_Response_" in column:
            mapping["json"].append(column)
        elif "_Full_Response_" in column:
            mapping["full"].append(column)
        elif "_Payload_" in column:
            mapping["payload"].append(column)
        elif column.startswith("Truncated_"):
            mapping["truncated"].append(column)
    return mapping


def _coalesce_row_values(
    frame: pd.DataFrame, columns: list[str], default: Any
) -> pd.Series:
    """Return the first non-null value across ``columns`` for each row.

    Useful for collapsing the per-runner ``*_Json_Response_<exp>`` style
    columns into a single ``json_response`` column. ``default`` is used
    when every column is missing for a row.
    """
    if not columns:
        return pd.Series(default, index=frame.index)

    values = frame[columns].infer_objects(copy=False).bfill(axis=1).iloc[:, 0]
    if isinstance(default, bool):
        return values.fillna(default).astype(bool)
    return values.fillna(default)


def normalize_chronology_outputs(
    dataframe: pd.DataFrame, config: AnalysisConfig
) -> pd.DataFrame:
    """Normalise raw batch exports into the long-form analysis schema.

    Adds the canonical ``shot_type``, ``report_sequence_name``,
    ``icl_shots``, ``model``, ``display_model``, ``temperature`` and
    ``model_canon`` columns; collapses the per-runner response columns
    into ``full_response`` / ``json_response`` / ``payload`` /
    ``truncated``; extracts the ``CLUES`` and ``REASONING`` text blocks;
    and parses the JSON payload to surface ``Likelihood of IBD``,
    ``Certainty Level`` and ``Complexity of Case`` as numeric columns.

    Parameters
    ----------
    dataframe:
        Output of :func:`combine_batch_exports`.
    config:
        The active :class:`AnalysisConfig`. ``runner_metadata`` and
        ``model_mapping`` are consulted for display names and canonical
        labels.

    Returns
    -------
    pandas.DataFrame
        A frame restricted to the canonical analysis columns. Columns
        absent from the input are silently dropped.
    """
    if dataframe.empty:
        return pd.DataFrame()

    frame = dataframe.copy()
    runner_metadata = config.raw.get("runner_metadata", {})
    prompt_info = frame["experiment_name"].astype(str).str.extract(PROMPT_PATTERN)
    frame["shot_type"] = prompt_info[0].fillna("unknown")
    frame["report_sequence_name"] = prompt_info[1].fillna(
        frame["experiment_name"].astype(str)
    )
    frame["icl_shots"] = (
        frame["shot_type"]
        .map({"zero": 0, "single": 1, "dual": 2})
        .fillna(-1)
        .astype(int)
    )
    frame["model"] = frame["runner_name"]
    frame["display_model"] = frame["runner_name"].map(
        lambda value: runner_metadata.get(value, {}).get("display_name", value)
    )
    frame["temperature"] = frame["display_model"].map(
        lambda value: _infer_temperature(str(value), runner_metadata)
    )
    frame["model_canon"] = frame["display_model"].map(
        lambda value: canonicalize_model(str(value), config.model_mapping)
    )

    # Primary-configuration tag (pre-specified hierarchy). The
    # primary deployment configuration is: zero-shot prompt,
    # ``all_docs_in_sequence`` context, and the family-default
    # deployment temperature listed below. Downstream analyses filter
    # on ``is_primary_configuration`` to surface the primary row
    # alongside sensitivity comparisons.
    primary_temperature_by_canon = {
        "mixtral7b": 0.75,
        "m42_8b": 0.75,
        "deepseek14b": 0.75,
        "deepseek32b": 0.6,
        "deepseek70b": 0.6,
        "qwen32b": 0.6,
        "gemma4_31b": 0.6,
    }

    def _is_primary(row: pd.Series) -> bool:
        if str(row.get("shot_type")) != "zero":
            return False
        if str(row.get("report_sequence_name")) != "all_docs_in_sequence":
            return False
        canon = str(row.get("model_canon"))
        expected = primary_temperature_by_canon.get(canon)
        if expected is None:
            return False
        temp = row.get("temperature")
        try:
            temp_f = float(temp)
        except (TypeError, ValueError):
            return False
        return abs(temp_f - expected) < 1e-6

    frame["is_primary_configuration"] = frame.apply(_is_primary, axis=1)

    response_map = _response_column_map(frame.columns.tolist())
    frame["full_response"] = _coalesce_row_values(frame, response_map["full"], "")
    frame["json_response"] = _coalesce_row_values(frame, response_map["json"], "")
    frame["payload"] = _coalesce_row_values(frame, response_map["payload"], "")
    frame["truncated"] = _coalesce_row_values(frame, response_map["truncated"], False)

    clues_reasoning = frame["full_response"].apply(_extract_prompt_sections)
    frame["clues_text"] = clues_reasoning.map(lambda item: item[0])
    frame["reasoning_text"] = clues_reasoning.map(lambda item: item[1])

    # Primary parse target is ``json_response``. The upstream client
    # substitutes the literal ``"No processable JSON response"`` when
    # its own ``json.loads`` failed, hiding the original payload from
    # the repair pipeline. Re-run :func:`parse_response` against
    # ``full_response`` for rows whose ``json_response`` parse failed.
    payloads_results = frame["json_response"].apply(parse_response)
    failed_mask = payloads_results.map(lambda r: r.payload is None)
    if bool(failed_mask.any()):
        fallback = frame.loc[failed_mask, "full_response"].apply(parse_response)
        # Only overwrite where the fallback produced a usable payload.
        recovered = fallback.map(lambda r: r.payload is not None)
        if bool(recovered.any()):
            payloads_results.loc[fallback.index[recovered]] = fallback[recovered]
    payloads = payloads_results.map(lambda r: r.payload)
    parsed = (
        pd.json_normalize([payload or {} for payload in payloads])
        if len(payloads)
        else pd.DataFrame(index=frame.index)
    )
    parsed.index = frame.index
    normalized = pd.concat([frame, parsed], axis=1)
    normalized["json_parse_success"] = payloads.notna().astype(int)
    normalized["json_repair_type"] = payloads_results.map(lambda r: r.repair_type)
    normalized["json_schema_valid"] = payloads_results.map(
        lambda r: int(r.schema_valid)
    )
    normalized["likelihood_score"] = pd.to_numeric(
        normalized.get("Likelihood of IBD"), errors="coerce"
    )
    normalized["certainty_score"] = pd.to_numeric(
        normalized.get("Certainty Level"), errors="coerce"
    )
    normalized["complexity_score"] = pd.to_numeric(
        normalized.get("Complexity of Case"), errors="coerce"
    )
    normalized["response_tokens"] = (
        normalized["full_response"].fillna("").astype(str).str.split().map(len)
    )

    preferred_columns = [
        "patient_id",
        "procedure_date",
        "sample_received_date",
        "result_report",
        "Combined_Content",
        "preceding_clinic_letter",
        "following_clinic_letter",
        "preceding_clinic_date",
        "following_clinic_date",
        "preceding_clinic_time_diff",
        "following_clinic_time_diff",
        "date_diff",
        "source_file",
        "source_parent",
        "experiment_name",
        "runner_name",
        "shot_type",
        "report_sequence_name",
        "icl_shots",
        "model",
        "display_model",
        "temperature",
        "model_canon",
        "is_primary_configuration",
        "full_response",
        "json_response",
        "payload",
        "truncated",
        "clues_text",
        "reasoning_text",
        "Title",
        "Features",
        "Likelihood of IBD",
        "Certainty Level",
        "Complexity of Case",
        "json_parse_success",
        "json_repair_type",
        "json_schema_valid",
        "likelihood_score",
        "certainty_score",
        "complexity_score",
        "response_tokens",
    ]
    available_columns = [
        column for column in preferred_columns if column in normalized.columns
    ]
    return normalized[available_columns].copy()


def extract_json_payload(text: Any) -> dict[str, Any] | None:
    """Parse a JSON payload from a model response.

    Thin compatibility wrapper around
    :func:`analytics_code.json_repair.parse_response` that returns
    only the parsed dict (or ``None``). Stages that need the
    ``repair_type`` / ``schema_valid`` flags should call
    ``parse_response`` directly.
    """
    return parse_response(text).payload


def parse_llm_outputs(
    dataframe: pd.DataFrame, response_column: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Parse a single response column and return the parsed frame plus a summary.

    Each row of ``dataframe`` has its ``response_column`` value parsed
    by :func:`extract_json_payload`. A ``json_parse_success`` flag is
    appended to the returned frame, and a per-source-folder summary of
    parse outcomes is produced as a second dataframe.
    """
    working = dataframe.copy()
    payloads = (
        working[response_column].apply(extract_json_payload)
        if response_column in working
        else pd.Series(dtype=object)
    )
    parsed = (
        pd.json_normalize([payload or {} for payload in payloads])
        if len(payloads)
        else pd.DataFrame(index=working.index)
    )
    parsed.index = working.index
    result = pd.concat([working, parsed], axis=1)
    result["json_parse_success"] = payloads.notna().astype(int)

    summary = (
        result.assign(
            parsed_flag=result["json_parse_success"].map({1: "parsed", 0: "failed"})
        )
        .groupby(["source_parent", "parsed_flag"], dropna=False)
        .size()
        .reset_index(name="n_rows")
    )
    return result, summary


def merge_with_reference(
    parsed_outputs: pd.DataFrame, reference_path: Path, id_column: str
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Left-merge LLM outputs onto the human reference cohort.

    The reference is read from CSV or XLSX based on ``reference_path``
    suffix. Returns ``(merged_frame, summary_frame)``; the summary
    reports row counts before, during and after the merge.
    """
    reference = (
        pd.read_excel(reference_path)
        if reference_path.suffix.lower() in {".xlsx", ".xls"}
        else pd.read_csv(reference_path)
    )
    merged = reference.merge(
        parsed_outputs, on=id_column, how="left", suffixes=("", "_llm")
    )
    summary = pd.DataFrame(
        {
            "metric": ["reference_rows", "parsed_rows", "merged_rows", "matched_rows"],
            "value": [
                len(reference),
                len(parsed_outputs),
                len(merged),
                int(
                    merged.filter(regex=r"json_parse_success").notna().any(axis=1).sum()
                ),
            ],
        }
    )
    return merged, summary


def build_total_cohort_summary(dataframe: pd.DataFrame) -> pd.DataFrame:
    """Build the ``4_3_1_total_study_cohort.csv`` summary table.

    Reports the number of rows and unique patients/models/experiments/
    report sequences. Returns an empty frame when the input is empty.
    """
    if dataframe.empty:
        return pd.DataFrame(columns=["metric", "value"])
    return pd.DataFrame(
        {
            "metric": [
                "n_rows",
                "n_unique_patients",
                "n_unique_models",
                "n_unique_experiments",
                "n_unique_report_sequences",
            ],
            "value": [
                len(dataframe),
                dataframe.get("patient_id", pd.Series(dtype=object)).nunique(),
                dataframe.get("model_canon", pd.Series(dtype=object)).nunique(),
                dataframe.get("experiment_name", pd.Series(dtype=object)).nunique(),
                dataframe.get(
                    "report_sequence_name", pd.Series(dtype=object)
                ).nunique(),
            ],
        }
    )


def _format_count_percent(numerator: int, total: int) -> str:
    """Render a ``count/total (pct%)`` string used in summary workbooks."""
    if total <= 0:
        return "0/0 (0.0%)"
    return f"{int(numerator)}/{int(total)} ({numerator / total * 100:.1f}%)"


def _attach_document_sequence_truth_columns(frame: pd.DataFrame) -> pd.DataFrame:
    """Materialise lenient and strict document-sequence truth columns.

    Adds :data:`DOCUMENT_SEQUENCE_TRUTH_LENIENT` and
    :data:`DOCUMENT_SEQUENCE_TRUTH_STRICT` to ``frame`` so downstream
    stages can select either truth definition without recomputing it.
    Also attaches :data:`PATIENT_DOCUMENT_TRUTH`, the per-``patient_id``
    OR over the four physician document markers, used as the Final_*
    tier reference standard.

    Returns ``frame`` unchanged when no document markers can be
    resolved (e.g. reference-only chronology configs).
    """
    if frame is None or frame.empty:
        return frame
    try:
        lenient = build_document_truth_series(frame, strict=False)
        strict = build_document_truth_series(frame, strict=True)
    except Exception:
        return frame
    out = frame.copy()
    out[DOCUMENT_SEQUENCE_TRUTH_LENIENT] = lenient
    out[DOCUMENT_SEQUENCE_TRUTH_STRICT] = strict
    out = _attach_patient_document_truth(out)
    return out


def _attach_patient_document_truth(frame: pd.DataFrame) -> pd.DataFrame:
    """Attach :data:`PATIENT_DOCUMENT_TRUTH` (OR over four doc flags per patient).

    Looks up the four physician marker columns named in
    :data:`analytics_code.common.DOC_FLAG_COLUMNS`. For each
    ``patient_id`` the value is ``1`` if any available marker is ``1``,
    ``0`` if at least one marker is observed and none are positive,
    and ``NaN`` if no marker is available for that patient. The column
    is broadcast back across every row for that patient so each row
    carries the patient-level label.
    """
    from analytics_code.common import DOC_FLAG_COLUMNS
    from analytics_code.truth_labels import _coerce_binary_series

    id_col = next(
        (c for c in ("patient_id", "study_id", "PatientID") if c in frame.columns),
        None,
    )
    if id_col is None:
        return frame
    marker_cols = [c for c in DOC_FLAG_COLUMNS.values() if c in frame.columns]
    if not marker_cols:
        return frame
    binary = pd.concat(
        {c: _coerce_binary_series(frame[c]) for c in marker_cols}, axis=1
    )
    # Reduce to one row per patient by taking the max (any positive)
    # and tracking availability separately.
    grouped = binary.groupby(frame[id_col].astype(object), dropna=False)
    any_positive = grouped.max()  # 1 if any positive, 0 if all zero, NaN if all missing
    any_known = grouped.apply(lambda g: g.notna().any().any())
    patient_truth = pd.Series(np.nan, index=any_positive.index, dtype="float64")
    pos_mask = (any_positive == 1).any(axis=1)
    patient_truth.loc[pos_mask] = 1.0
    neg_mask = (~pos_mask) & any_known
    patient_truth.loc[neg_mask] = 0.0
    mapped = frame[id_col].astype(object).map(patient_truth)
    out = frame.copy()
    out[PATIENT_DOCUMENT_TRUTH] = pd.to_numeric(mapped, errors="coerce")
    return out


def _validation_level(config: AnalysisConfig) -> str:
    """Return the configured validation view slug.

    Supported values are ``document``, ``cumulative``, ``final``, and
    ``doc2patient``. Any other value disables view transformation.
    """
    level = str(config.analysis.get("validation_level", "")).strip().lower()
    if level in {"document", "cumulative", "final", "doc2patient"}:
        return level
    return ""


def _concat_unique_text(values: pd.Series) -> str:
    """Concatenate non-empty unique strings preserving first-seen order."""
    seen: list[str] = []
    for value in values.fillna("").astype(str):
        text = value.strip()
        if not text or text in seen:
            continue
        seen.append(text)
    return "\n\n".join(seen)


def _aggregate_patient_level_view(
    frame: pd.DataFrame, *, final_truth: bool = False
) -> pd.DataFrame:
    """Collapse rows to one patient-level record per model/shot/temperature.

    This operationalises the methodology's Final / Doc2Patient unit of
    analysis: one row per ``(patient_id, model_canon, shot_type,
    temperature)`` where the prediction is the logical OR over every
    linked-row prediction for that patient.

    ``final_truth=True`` rewrites ``Patient_Has_IBD`` / ``ground_truth``
    to the document-derived patient endpoint so downstream stages score
    against the Final_* reference standard. ``False`` keeps the chart-
    verified patient label for the Doc2Patient secondary endpoint.
    """
    if frame.empty or "likelihood_score" not in frame.columns:
        return frame
    patient_col = next(
        (c for c in ("patient_id", "study_id", "PatientID") if c in frame.columns),
        None,
    )
    if patient_col is None:
        return frame

    group_cols = [
        patient_col,
        *[
            c
            for c in (
                "runner_name",
                "model",
                "display_model",
                "model_canon",
                "shot_type",
                "temperature",
            )
            if c in frame.columns
        ],
    ]
    working = frame.copy()
    bucket = clean_likelihood(working["likelihood_score"])
    working["__row_pred"] = bucket.fillna(-1).ge(5).astype(int)
    grouped = working.groupby(group_cols, dropna=False)
    out = grouped.first().reset_index()
    out["likelihood_score"] = grouped["__row_pred"].max().to_numpy(dtype=float) * 10.0
    if "Likelihood of IBD" in out.columns:
        out["Likelihood of IBD"] = out["likelihood_score"]
    if "certainty_score" in working.columns:
        out["certainty_score"] = grouped["certainty_score"].max().to_numpy(dtype=float)
    if "Certainty Level" in out.columns and "certainty_score" in out.columns:
        out["Certainty Level"] = out["certainty_score"]
    if "complexity_score" in working.columns:
        out["complexity_score"] = (
            grouped["complexity_score"].max().to_numpy(dtype=float)
        )
    if "Complexity of Case" in out.columns and "complexity_score" in out.columns:
        out["Complexity of Case"] = out["complexity_score"]
    if "json_parse_success" in working.columns:
        out["json_parse_success"] = (
            grouped["json_parse_success"].max().to_numpy(dtype=int)
        )
    if "json_schema_valid" in working.columns:
        out["json_schema_valid"] = (
            grouped["json_schema_valid"].max().to_numpy(dtype=int)
        )
    if "truncated" in working.columns:
        out["truncated"] = grouped["truncated"].any().to_numpy(dtype=bool)
    if "response_tokens" in working.columns:
        out["response_tokens"] = grouped["response_tokens"].sum().to_numpy(dtype=int)
    for text_col in (
        "full_response",
        "json_response",
        "payload",
        "clues_text",
        "reasoning_text",
        "Title",
        "Features",
        "Combined_Content",
        "result_report",
    ):
        if text_col in working.columns:
            out[text_col] = grouped[text_col].apply(_concat_unique_text).to_numpy()

    # Reuse the production FAIR filters by assigning a single canonical
    # context slug to the patient-level endpoint.
    out["report_sequence_name"] = "all_docs_in_sequence"
    if "experiment_name" in out.columns:
        out["experiment_name"] = "final_patient"
    if PATIENT_DOCUMENT_TRUTH in working.columns:
        out[PATIENT_DOCUMENT_TRUTH] = (
            grouped[PATIENT_DOCUMENT_TRUTH].first().to_numpy(dtype=float)
        )
        if final_truth:
            out["Patient_Has_IBD"] = out[PATIENT_DOCUMENT_TRUTH]
            out["ground_truth"] = out[PATIENT_DOCUMENT_TRUTH]
    return out.drop(columns=[c for c in ("__row_pred",) if c in out.columns])


def _apply_validation_level_view(
    frame: pd.DataFrame, config: AnalysisConfig
) -> pd.DataFrame:
    """Return ``frame`` filtered/aggregated for the configured validation level."""
    level = _validation_level(config)
    if not level or frame.empty or "report_sequence_name" not in frame.columns:
        return frame
    if level == "document":
        return frame[frame["report_sequence_name"].isin(SINGLE_DOC_DATA_TYPES)].copy()
    if level == "cumulative":
        return frame[frame["report_sequence_name"].isin(MULTI_DOC_DATA_TYPES)].copy()
    if level == "final":
        return _aggregate_patient_level_view(frame, final_truth=True)
    if level == "doc2patient":
        return _aggregate_patient_level_view(frame, final_truth=False)
    return frame


def _folder_label(row: pd.Series) -> str:
    """Build folder strings, e.g. mixtral7b_t0_75_zero_shot_all_docs_in_sequence."""
    display = str(row.get("display_model") or row.get("model") or "unknown_model")
    shot = str(row.get("shot_type") or "zero")
    seq = str(row.get("report_sequence_name") or "unknown_sequence")
    return f"{display}_{shot}_shot_{seq}"


def build_llm_json_processing_summary(normalized: pd.DataFrame) -> pd.DataFrame:
    """Per-folder ``llm_json_processing_summary.csv``.

    Reports, per source folder, the breakdown of JSON-parse outcomes
    using the explicit tags emitted by
    :func:`analytics_code.json_repair.parse_response`:

    * ``raw_valid_json`` — the response string was valid JSON as-is.
    * ``extracted_embedded_json`` — a ``{...}`` block was lifted out
      of surrounding free text and parsed unchanged.
    * ``repaired_json`` — quote / brace / trailing-comma repair was
      required before the payload would parse.
    * ``schema_invalid`` — parsed as JSON but failed schema
      validation against the published prompt contract.
    * ``unparseable`` — could not be coerced even after repair.
    * ``no_response`` — empty / missing ``full_response``.

    The historical column names (``no_repair_needed``, ``JSON_repaired``,
    ``total_successful``, ``IBD_predicted_ge_5``) are retained as
    ``count/total (pct%)`` strings for downstream compatibility; the
    new explicit tag columns are added alongside.
    """
    if normalized.empty:
        return pd.DataFrame()

    frame = normalized.copy()
    frame["__folder"] = frame.apply(_folder_label, axis=1)
    repair_type = (
        frame.get("json_repair_type")
        if "json_repair_type" in frame.columns
        else pd.Series(index=frame.index, dtype=object)
    ).astype(object)
    schema_valid = (
        pd.to_numeric(frame.get("json_schema_valid"), errors="coerce")
        .fillna(0)
        .astype(int)
    )
    likelihood = pd.to_numeric(frame.get("likelihood_score"), errors="coerce")
    full_response = (
        frame.get("full_response").fillna("")
        if "full_response" in frame
        else pd.Series("", index=frame.index)
    )
    no_response = full_response.astype(str).str.strip().eq("")

    rows = []
    for folder, idx in frame.groupby("__folder").groups.items():
        sub = frame.loc[idx]
        total = len(sub)
        sub_repair = repair_type.loc[idx]
        counts = {tag: int((sub_repair == tag).sum()) for tag in REPAIR_TYPES}
        raw_valid = counts["raw_valid_json"]
        extracted = counts["extracted_embedded_json"]
        repaired = counts["repaired_json"]
        schema_invalid = counts["schema_invalid"]
        unparseable = counts["unparseable"]
        empty = counts["empty"]
        total_valid = raw_valid + extracted + repaired
        no_resp = int(no_response.loc[idx].sum())
        unique_patients = int(sub.get("patient_id", pd.Series(dtype=object)).nunique())
        high_lik = int((likelihood.loc[idx] >= 5).sum())
        rows.append(
            {
                "folder": folder,
                "status": "ok" if total > 0 else "empty",
                "unique_patients": _format_count_percent(unique_patients, total),
                "raw_valid_json": _format_count_percent(raw_valid, total),
                "extracted_embedded_json": _format_count_percent(extracted, total),
                "repaired_json": _format_count_percent(repaired, total),
                "schema_invalid": _format_count_percent(schema_invalid, total),
                "unparseable": _format_count_percent(unparseable, total),
                # Legacy columns retained for downstream compatibility.
                # ``no_repair_needed`` now means raw-valid-JSON only.
                "no_repair_needed": _format_count_percent(raw_valid, total),
                "JSON_repaired": _format_count_percent(repaired, total),
                "no_response": _format_count_percent(no_resp, total),
                "total_successful": _format_count_percent(total_valid, total),
                "schema_valid_total": _format_count_percent(
                    int(schema_valid.loc[idx].sum()), total
                ),
                "IBD_predicted_ge_5": _format_count_percent(high_lik, total),
                # numeric helpers for downstream consumers
                "total_success_percent": total_valid / total * 100 if total else 0.0,
                "no_repair_percent": raw_valid / total * 100 if total else 0.0,
                "schema_valid_percent": int(schema_valid.loc[idx].sum()) / total * 100
                if total
                else 0.0,
                "percent_ge_5": high_lik / total * 100 if total else 0.0,
                "n_rows": total,
                "n_empty_responses": empty,
            }
        )
    return pd.DataFrame(rows).sort_values("folder").reset_index(drop=True)


def build_merge_processing_summary(
    normalized: pd.DataFrame, merged: pd.DataFrame | None, id_column: str
) -> pd.DataFrame:
    """Per-folder ``merge_processing_summary.csv``."""
    if normalized.empty:
        return pd.DataFrame()

    frame = normalized.copy()
    frame["__folder"] = frame.apply(_folder_label, axis=1)
    full_response = (
        frame.get("full_response").fillna("")
        if "full_response" in frame
        else pd.Series("", index=frame.index)
    )
    api_failed = full_response.astype(str).str.strip().isin({"", "{}"}) | (
        pd.to_numeric(frame.get("json_parse_success"), errors="coerce")
        .fillna(0)
        .astype(int)
        == 0
    )

    merged_by_folder: dict[str, int] = {}
    if merged is not None and not merged.empty and "__folder" not in merged.columns:
        merged = merged.copy()
        if all(
            col in merged.columns
            for col in ("display_model", "shot_type", "report_sequence_name")
        ):
            merged["__folder"] = merged.apply(_folder_label, axis=1)
            merged_by_folder = merged.groupby("__folder").size().to_dict()

    rows = []
    for folder, idx in frame.groupby("__folder").groups.items():
        sub = frame.loc[idx]
        input_rows = len(sub)
        unique_patients = int(sub.get("patient_id", pd.Series(dtype=object)).nunique())
        duplicates_dropped = max(input_rows - unique_patients, 0)
        post_dedup = unique_patients
        merged_rows = int(merged_by_folder.get(folder, post_dedup))
        api_failures = int(api_failed.loc[idx].sum())
        rows.append(
            {
                "folder": folder,
                "output": f"{folder}_merged.xlsx",
                "input_rows": input_rows,
                "duplicates_dropped": duplicates_dropped,
                "merged_rows": merged_rows,
                "post_dedup_rows": post_dedup,
                "api_failures": api_failures,
                "status": "ok",
                "error": "",
            }
        )
    return pd.DataFrame(rows).sort_values("folder").reset_index(drop=True)


def run_data_prep(config: AnalysisConfig) -> dict[str, Path]:
    """Run the data-preparation stage of the pipeline.

    Reads the per-runner batch exports from ``paths.llm_batch_root``,
    normalises and parses them, optionally merges against the human
    reference cohort at ``paths.human_reference``, and writes every
    artefact under ``<output_root>/data_prep``.

    Parameters
    ----------
    config:
        The active :class:`AnalysisConfig`.

    Returns
    -------
    dict[str, pathlib.Path]
        Mapping from a short artefact name to its output path. The
        mapping is empty when no batch root is configured.
    """
    paths = config.paths
    stage_dir = ensure_dir(paths["output_root"] / "data_prep")
    batch_root = paths.get("llm_batch_root")
    id_column = config.data_prep.get("id_column", "study_id")

    outputs: dict[str, Path] = {}
    normalized = pd.DataFrame()

    if batch_root and batch_root.exists():
        # Accept any combination of batch_*.{xlsx,parquet,csv} produced by
        # the runner; methodology defaults to xlsx, promoted to parquet
        # for batches above the 10 MB size threshold.
        batch_glob_patterns = config.data_prep.get(
            "batch_glob",
            ("**/batch_*.xlsx", "**/batch_*.parquet", "**/batch_*.csv"),
        )
        if isinstance(batch_glob_patterns, str):
            batch_glob_patterns = (batch_glob_patterns,)
        batch_files: list[Path] = []
        for pattern in batch_glob_patterns:
            batch_files.extend(find_batch_files(batch_root, pattern))
        batch_files = sorted(set(batch_files))
        combined = combine_batch_exports(batch_files)
        outputs["combined_batch_outputs"] = write_dataframe(
            combined, stage_dir / "combined_batch_outputs.xlsx"
        )

        normalized = normalize_chronology_outputs(combined, config)
        normalized = _apply_validation_level_view(normalized, config)
        outputs["parsed_outputs"] = write_dataframe(
            normalized, stage_dir / "parsed_outputs.csv"
        )
        outputs["llm_json_processing_summary"] = write_dataframe(
            build_llm_json_processing_summary(normalized),
            stage_dir / "llm_json_processing_summary.csv",
        )
        outputs["4_3_1_total_study_cohort"] = write_dataframe(
            build_total_cohort_summary(normalized),
            stage_dir / "4_3_1_total_study_cohort.csv",
        )

    merged: pd.DataFrame | None = None
    reference_path = paths.get("human_reference")
    if (
        reference_path
        and reference_path.exists()
        and not normalized.empty
        and id_column in normalized.columns
    ):
        merged, _ = merge_with_reference(
            normalized, reference_path, id_column=id_column
        )
        merged = _attach_document_sequence_truth_columns(merged)
        merged = _apply_validation_level_view(merged, config)
        outputs["merged_outputs"] = write_dataframe(
            merged, stage_dir / "merged_outputs.csv"
        )

    if not normalized.empty:
        outputs["merge_processing_summary"] = write_dataframe(
            build_merge_processing_summary(normalized, merged, id_column),
            stage_dir / "merge_processing_summary.csv",
        )
    else:
        write_json(
            {"note": "Reference merge skipped"},
            stage_dir / "merge_processing_summary.json",
        )
        outputs["merge_processing_summary"] = (
            stage_dir / "merge_processing_summary.json"
        )

    return outputs
