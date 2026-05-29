"""Orchestrates a chronological experiment run across multiple Gradio runners.

For each (prompt template x report sequence) experiment, the runner
dispatches the same per-row message to every enabled runner endpoint
in parallel, accumulates their responses on the in-memory dataframe,
and persists the per-batch slice as soon as the batch completes. The
batch counter is restored from disk when an interrupted run is
resumed.
"""

from __future__ import annotations

import gc
import logging
import tempfile
from concurrent.futures import ThreadPoolExecutor, as_completed
from pathlib import Path
from typing import Any

import pandas as pd

from python_client.api import call_gradio_chat, get_tokenizer
from python_client.data_processing import build_chronology_dataframe
from python_client.prompts import (
    build_message_with_budget,
    generate_experiments,
    prepare_report_sections,
)
from python_client.run_manifest import (
    build_run_manifest_path,
    hash_text,
    write_manifest_entry,
)

BASE_OUTPUT_COLUMNS = [
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
    "episode_id",
]


def ensure_dir(directory: str | Path) -> Path:
    """Create ``directory`` (and any parents) and return it as a :class:`Path`."""
    path = Path(directory)
    path.mkdir(parents=True, exist_ok=True)
    return path


# Methodology threshold: batches above this serialised size are
# promoted from .xlsx to .parquet (see "Experimental templating
# versioning system").
PARQUET_PROMOTION_BYTES = 10 * 1024 * 1024


def _batch_suffix(save_format: str) -> str:
    """Map a save format (``"csv"``/``"xlsx"``/``"parquet"``) to its file suffix."""
    fmt = save_format.lower()
    if fmt == "csv":
        return ".csv"
    if fmt == "parquet":
        return ".parquet"
    return ".xlsx"


def load_progress(output_dir: str | Path, save_format: str) -> int:
    """Return the next batch number to process given the on-disk batches.

    Scans ``output_dir`` for ``batch_<N>`` files. The primary suffix is
    determined by ``save_format``, but any of ``.xlsx``/``.parquet``/
    ``.csv`` are also recognised so that a batch that was promoted to a
    larger format during the previous run is still counted.
    """
    output_dir = Path(output_dir)
    if not output_dir.exists():
        return 0
    suffixes = {_batch_suffix(save_format), ".xlsx", ".parquet", ".csv"}
    batch_numbers: list[int] = []
    for suffix in suffixes:
        for file_path in output_dir.glob(f"batch_*{suffix}"):
            stem = file_path.stem
            try:
                batch_numbers.append(int(stem.split("_")[1]))
            except (IndexError, ValueError):
                continue
    return max(batch_numbers) + 1 if batch_numbers else 0


def save_batch(
    data_frame: pd.DataFrame,
    runner_output_dir: str | Path,
    batch_number: int,
    runner_name: str,
    result_columns: dict[str, str],
    *,
    batch_size: int,
    save_format: str,
) -> None:
    """Persist a single batch slice atomically.

    Writes to a temporary file in the same directory and only renames
    it into ``batch_<batch_number>.<suffix>`` once the write completes,
    so partially-written batches are never observed by ``load_progress``.
    """
    runner_output_dir = ensure_dir(runner_output_dir)
    suffix = _batch_suffix(save_format)
    batch_file = runner_output_dir / f"batch_{batch_number}{suffix}"

    columns = BASE_OUTPUT_COLUMNS + [
        result_columns["full_response"],
        result_columns["json_response"],
        result_columns["payload"],
        result_columns["truncated"],
        result_columns["truncated_sections"],
    ]
    columns = [column for column in columns if column in data_frame.columns]

    start_row = batch_number * batch_size
    end_row = start_row + batch_size
    batch_df = data_frame.iloc[start_row:end_row][columns].copy()

    with tempfile.NamedTemporaryFile(delete=False, suffix=suffix) as temp_file:
        temp_path = Path(temp_file.name)
    try:
        fmt = save_format.lower()
        if fmt == "csv":
            batch_df.to_csv(temp_path, index=False)
        elif fmt == "parquet":
            batch_df.to_parquet(temp_path, index=False)
        else:
            estimated_bytes = int(batch_df.memory_usage(deep=True).sum())
            if estimated_bytes > PARQUET_PROMOTION_BYTES:
                # Methodology: promote large batches to parquet.
                promoted_path = temp_path.with_suffix(".parquet")
                batch_df.to_parquet(promoted_path, index=False)
                temp_path.unlink(missing_ok=True)
                temp_path = promoted_path
                batch_file = batch_file.with_suffix(".parquet")
                logging.info(
                    "Promoting batch %s for runner %s to Parquet (estimated %.1f MB)",
                    batch_number,
                    runner_name,
                    estimated_bytes / (1024 * 1024),
                )
            else:
                batch_df.to_excel(temp_path, index=False, engine="openpyxl")
        temp_path.replace(batch_file)
    finally:
        if temp_path.exists():
            temp_path.unlink(missing_ok=True)

    logging.info(
        "Saved batch %s for runner %s to %s", batch_number, runner_name, batch_file
    )


def _ensure_result_columns(
    data_frame: pd.DataFrame, result_columns: dict[str, str]
) -> pd.DataFrame:
    """Return ``data_frame`` augmented with any missing per-runner result columns."""
    new_columns: dict[str, Any] = {}
    for logical_name, column_name in result_columns.items():
        if column_name in data_frame.columns:
            continue
        new_columns[column_name] = False if logical_name == "truncated" else ""
    if not new_columns:
        return data_frame
    return data_frame.assign(**new_columns)


def _build_result_column_map(runner_name: str, experiment_name: str) -> dict[str, str]:
    """Return the per-runner output column names for a given experiment."""
    return {
        "json_response": f"{runner_name}_Json_Response_{experiment_name}",
        "full_response": f"{runner_name}_Full_Response_{experiment_name}",
        "payload": f"{runner_name}_Payload_{experiment_name}",
        "truncated": f"Truncated_{experiment_name}",
        "truncated_sections": f"Truncated_Sections_{experiment_name}",
    }


def _active_runners(config: dict[str, Any]) -> list[tuple[str, dict[str, Any]]]:
    """Return the enabled runners, capped at ``general.max_parallel_runners``."""
    active = [
        (runner_name, runner_config)
        for runner_name, runner_config in config["runner_endpoints"].items()
        if runner_config.get("enabled", False)
    ]
    return active[: int(config["general"]["max_parallel_runners"])]


def run_chronological_experiments(config: dict[str, Any]) -> None:
    """Run every (prompt template x report sequence) experiment for every active runner.

    The function builds the chronology dataframe once, derives the
    experiment list, then iterates over experiments and over batches.
    For each row inside a batch the prompt is dispatched in parallel
    to every active runner (one thread per runner via
    :class:`concurrent.futures.ThreadPoolExecutor`). Already-completed
    batches are skipped on resume.

    Parameters
    ----------
    config:
        The configuration dictionary returned by
        :func:`python_client.config.load_config`.
    """
    output_dir = ensure_dir(
        Path(config["general"]["output_dir"]) / config["general"]["version"]
    )
    manifest_path = build_run_manifest_path(output_dir)
    save_format = config["general"].get("save_format", "xlsx")
    batch_size = int(config["general"]["batch_size"])
    retry_delay_seconds = float(config["general"]["retry_delay_seconds"])
    max_retries = int(config["general"]["max_retries"])
    token_limit = int(config["general"]["token_limit"])
    predict_timeout_seconds = float(config["general"]["predict_timeout_seconds"])
    tokenizer = get_tokenizer(config["general"]["tokenizer_encoding"])

    chronology_df = build_chronology_dataframe(config)
    experiments = generate_experiments(config)
    runners = _active_runners(config)

    logging.info("Loaded %s chronology rows.", len(chronology_df))
    logging.info("Generated %s experiments.", len(experiments))
    logging.info("Using %s active runners.", len(runners))

    for experiment in experiments:
        experiment_name = experiment["experiment_name"]
        logging.info("Starting experiment %s", experiment_name)

        runner_progress: dict[str, dict[str, Any]] = {}
        all_completed = True
        for runner_name, runner_config in runners:
            runner_output_dir = output_dir / runner_name / experiment_name
            ensure_dir(runner_output_dir)
            batch_number = load_progress(runner_output_dir, save_format)
            start_row = batch_number * batch_size
            if start_row < len(chronology_df):
                all_completed = False
            runner_progress[runner_name] = {
                "runner_config": runner_config,
                "runner_output_dir": runner_output_dir,
                "batch_number": batch_number,
                "start_row": start_row,
                "processed_rows": start_row,
            }

        if all_completed:
            logging.info(
                "Experiment %s already complete for all runners.", experiment_name
            )
            continue

        number_of_batches = (len(chronology_df) + batch_size - 1) // batch_size
        for batch_number in range(number_of_batches):
            batch_start = batch_number * batch_size
            batch_end = min(batch_start + batch_size, len(chronology_df))
            batch_df = chronology_df.iloc[batch_start:batch_end].copy()

            runners_to_process = [
                (runner_name, progress)
                for runner_name, progress in runner_progress.items()
                if progress["batch_number"] <= batch_number
            ]
            if not runners_to_process:
                continue

            logging.info(
                "Processing experiment %s batch %s covering rows %s-%s with runners %s",
                experiment_name,
                batch_number,
                batch_start,
                batch_end - 1,
                [runner_name for runner_name, _ in runners_to_process],
            )

            for row_index, row in batch_df.iterrows():
                sections = prepare_report_sections(
                    row,
                    experiment["report_sequence"],
                    experiment["context_note_timing"],
                )
                reports_section = "\n".join(
                    f"**{label}**:\n\n{text}\n" for label, text in sections
                ).strip()
                if not reports_section:
                    logging.warning(
                        "Skipping row %s because no report content was available.",
                        row_index,
                    )
                    continue

                (
                    combined_message,
                    pre_truncated,
                    truncated_sections,
                ) = build_message_with_budget(
                    experiment["prompt_content"],
                    sections,
                    tokenizer,
                    token_limit,
                )
                if pre_truncated:
                    logging.info(
                        "Section-truncated row %s for experiment %s: %s",
                        row_index,
                        experiment_name,
                        truncated_sections,
                    )
                future_to_runner: dict[
                    Any, tuple[str, dict[str, Any], dict[str, str]]
                ] = {}

                with ThreadPoolExecutor(
                    max_workers=len(runners_to_process)
                ) as executor:
                    for runner_name, progress in runners_to_process:
                        runner_config = progress["runner_config"]
                        result_columns = _build_result_column_map(
                            runner_name, experiment_name
                        )
                        chronology_df = _ensure_result_columns(
                            chronology_df, result_columns
                        )
                        chronology_df.at[
                            row_index, result_columns["payload"]
                        ] = combined_message

                        future = executor.submit(
                            call_gradio_chat,
                            runner_config["url"],
                            combined_message,
                            retry_delay_seconds=retry_delay_seconds,
                            max_retries=max_retries,
                            tokenizer=tokenizer,
                            token_limit=token_limit,
                            predict_timeout_seconds=predict_timeout_seconds,
                        )
                        future_to_runner[future] = (
                            runner_name,
                            progress,
                            result_columns,
                        )

                    for future in as_completed(future_to_runner):
                        runner_name, progress, result_columns = future_to_runner[future]
                        try:
                            (
                                full_response,
                                json_response,
                                was_truncated,
                            ) = future.result()
                        except Exception as exc:
                            logging.error(
                                "Runner %s failed for row %s in experiment %s: %s",
                                runner_name,
                                row_index,
                                experiment_name,
                                exc,
                            )
                            full_response = "API call failed"
                            json_response = "API call failed"
                            was_truncated = False

                        manifest_entry = {
                            "experiment": experiment_name,
                            "prompt_template": experiment["prompt_name"],
                            "report_sequence_name": experiment["report_sequence_name"],
                            "report_sequence": experiment["report_sequence"],
                            "context_note_timing": experiment["context_note_timing"],
                            "runner_name": runner_name,
                            "endpoint": progress["runner_config"].get("url"),
                            "row_index": int(row_index),
                            "patient_id": str(row.get("patient_id", "")),
                            "episode_id": str(row.get("episode_id", "")),
                            "batch_number": batch_number,
                            "token_limit": token_limit,
                            "prompt_template_sha256": hash_text(
                                experiment["prompt_content"]
                            ),
                            "message_sha256": hash_text(combined_message),
                            "section_truncated": bool(pre_truncated),
                            "truncated_sections": list(truncated_sections),
                            "api_truncated": bool(was_truncated),
                            "full_response_chars": len(str(full_response or "")),
                            "json_response_chars": len(str(json_response or "")),
                        }
                        write_manifest_entry(manifest_path, manifest_entry)

                        chronology_df.at[
                            row_index, result_columns["full_response"]
                        ] = full_response
                        chronology_df.at[
                            row_index, result_columns["json_response"]
                        ] = json_response
                        chronology_df.at[row_index, result_columns["truncated"]] = (
                            was_truncated or pre_truncated
                        )
                        chronology_df.at[
                            row_index, result_columns["truncated_sections"]
                        ] = ",".join(truncated_sections)
                        progress["processed_rows"] += 1
                        logging.info(
                            "%s | %s: %.2f%% complete (%s/%s rows)",
                            runner_name,
                            experiment_name,
                            (progress["processed_rows"] / len(chronology_df)) * 100,
                            progress["processed_rows"],
                            len(chronology_df),
                        )

            for runner_name, progress in runners_to_process:
                result_columns = _build_result_column_map(runner_name, experiment_name)
                save_batch(
                    chronology_df,
                    progress["runner_output_dir"],
                    batch_number,
                    runner_name,
                    result_columns,
                    batch_size=batch_size,
                    save_format=save_format,
                )
                progress["batch_number"] = batch_number + 1
                progress["start_row"] = (batch_number + 1) * batch_size

            gc.collect()

        logging.info("Finished experiment %s", experiment_name)
