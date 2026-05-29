"""Configuration loading for the chronological experiment client.

A configuration document (JSON) declares the data sources, prompt
templates, report sequences and the runner endpoints to dispatch
requests to. :func:`load_config` validates the document, normalises
legacy fields, resolves relative paths against the configuration file
directory and inlines every prompt template so downstream stages have
a ready-to-use mapping.
"""

from __future__ import annotations

import json
from pathlib import Path
from typing import Any

REQUIRED_TOP_LEVEL_KEYS = {
    "general",
    "data_sources",
    "matching",
    "prompt_templates",
    "report_sequences",
}


class ConfigError(ValueError):
    """Raised when the client configuration is missing required fields or invalid."""


def _resolve_path(base_dir: Path, raw_path: str | None) -> str | None:
    """Resolve ``raw_path`` to an absolute path against ``base_dir``.

    Returns ``None`` when ``raw_path`` is ``None`` (used so optional
    fields stay optional after normalisation).
    """
    if raw_path is None:
        return None
    path = Path(raw_path)
    if not path.is_absolute():
        path = (base_dir / path).resolve()
    return str(path)


def _validate_general(config: dict[str, Any]) -> None:
    """Check the ``general`` block for the mandatory fields and runner cap."""
    required = {"batch_size", "version", "output_dir", "max_parallel_runners"}
    general = config["general"]
    missing = sorted(required - set(general))
    if missing:
        raise ConfigError(
            f"Missing required general config fields: {', '.join(missing)}"
        )
    if int(general["max_parallel_runners"]) > 4:
        raise ConfigError("max_parallel_runners cannot exceed 4.")


def _normalize_data_sources(config: dict[str, Any]) -> None:
    """Alias the legacy ``clinic_letters`` data source to ``context_notes``."""
    data_sources = config["data_sources"]
    if "clinic_letters" in data_sources and "context_notes" not in data_sources:
        data_sources["context_notes"] = data_sources["clinic_letters"]


def _normalize_runners(config: dict[str, Any]) -> None:
    """Build ``runner_endpoints`` from a legacy ``general.api_endpoints`` mapping.

    No-op when ``runner_endpoints`` is already provided. Raises
    :class:`ConfigError` when neither field is set.
    """
    if "runner_endpoints" in config:
        return

    api_endpoints = config["general"].get("api_endpoints", {})
    if not api_endpoints:
        raise ConfigError(
            "Either runner_endpoints or general.api_endpoints must be provided."
        )

    config["runner_endpoints"] = {
        runner_name: {
            "enabled": True,
            "display_name": runner_name,
            "url": runner_url,
        }
        for runner_name, runner_url in api_endpoints.items()
    }


def _validate_data_sources(config: dict[str, Any]) -> None:
    """Check that every data source declares its required column labels."""
    _normalize_data_sources(config)
    required_sections = {"primary_reports", "secondary_reports", "context_notes"}
    data_sources = config["data_sources"]
    missing_sections = sorted(required_sections - set(data_sources))
    if missing_sections:
        raise ConfigError(
            f"Missing data_sources sections: {', '.join(missing_sections)}"
        )

    required_report_fields = {
        "path",
        "patient_id_column",
        "date_column",
        "text_column",
        "label",
    }
    for key in ("primary_reports", "secondary_reports"):
        missing = sorted(required_report_fields - set(data_sources[key]))
        if missing:
            raise ConfigError(f"Missing fields in {key}: {', '.join(missing)}")

    note_fields = {
        "path",
        "patient_id_column",
        "date_column",
        "text_column",
        "preceding_label",
        "following_label",
    }
    missing = sorted(note_fields - set(data_sources["context_notes"]))
    if missing:
        raise ConfigError(
            f"Missing fields in clinic_letters/context_notes: {', '.join(missing)}"
        )


def _validate_runners(config: dict[str, Any]) -> None:
    """Ensure at least one enabled runner exists and respects the parallel cap."""
    _normalize_runners(config)
    runners = config["runner_endpoints"]
    enabled_runners = [
        name for name, entry in runners.items() if entry.get("enabled", False)
    ]
    if not enabled_runners:
        raise ConfigError("At least one enabled runner endpoint is required.")
    if len(enabled_runners) > int(config["general"]["max_parallel_runners"]):
        raise ConfigError("Enabled runners exceed max_parallel_runners.")
    for runner_name in enabled_runners:
        entry = runners[runner_name]
        for field in ("display_name", "url"):
            if field not in entry:
                raise ConfigError(f"Missing '{field}' in runner '{runner_name}'.")


def load_config(config_path: str | Path) -> dict[str, Any]:
    """Load, validate and normalise a client configuration document.

    The returned dictionary is the parsed JSON augmented with:

    * ``__config_path__`` / ``__base_dir__`` -- absolute paths used to
      resolve relative entries.
    * ``general.output_dir`` resolved to an absolute path and several
      sensible defaults applied (``max_retries``, ``token_limit``,
      ``save_format`` etc.).
    * ``data_sources.<name>.path`` resolved to an absolute path.
    * ``prompt_templates`` replaced by an in-memory mapping of template
      name to template text. The original paths are preserved under
      ``prompt_template_paths``.

    Raises
    ------
    ConfigError
        On any validation failure (missing file, missing fields,
        prompt template not found, etc.).
    """
    config_path = Path(config_path).resolve()
    if not config_path.exists():
        raise ConfigError(f"Config file not found: {config_path}")

    with config_path.open("r", encoding="utf-8") as handle:
        config = json.load(handle)

    missing_top_level = sorted(REQUIRED_TOP_LEVEL_KEYS - set(config))
    if missing_top_level:
        raise ConfigError(
            f"Missing top-level config sections: {', '.join(missing_top_level)}"
        )

    _validate_general(config)
    _validate_data_sources(config)
    _validate_runners(config)

    base_dir = config_path.parent
    config["__config_path__"] = str(config_path)
    config["__base_dir__"] = str(base_dir)

    config["general"]["output_dir"] = _resolve_path(
        base_dir, config["general"]["output_dir"]
    )
    config["general"].setdefault("max_retries", 3)
    config["general"].setdefault("retry_delay_seconds", 0.25)
    config["general"].setdefault("token_limit", 4000)
    config["general"].setdefault("predict_timeout_seconds", 300)
    config["general"].setdefault("tokenizer_encoding", "cl100k_base")
    config["general"].setdefault("save_format", "xlsx")
    save_format = str(config["general"]["save_format"]).lower()
    if save_format not in {"csv", "xlsx", "parquet"}:
        raise ValueError(
            f"general.save_format must be one of 'csv', 'xlsx' or 'parquet'; got {save_format!r}"
        )
    config["general"]["save_format"] = save_format
    config["general"].setdefault("max_cases", None)

    for source_config in config["data_sources"].values():
        source_config["path"] = _resolve_path(base_dir, source_config["path"])

    loaded_templates: dict[str, str] = {}
    resolved_template_paths: dict[str, str] = {}
    for template_name, template_path in config["prompt_templates"].items():
        resolved_path = _resolve_path(base_dir, template_path)
        if resolved_path is None or not Path(resolved_path).exists():
            raise ConfigError(
                f"Prompt template not found for {template_name}: {template_path}"
            )
        resolved_template_paths[template_name] = resolved_path
        loaded_templates[template_name] = Path(resolved_path).read_text(
            encoding="utf-8"
        )

    config["prompt_template_paths"] = resolved_template_paths
    config["prompt_templates"] = loaded_templates
    return config
