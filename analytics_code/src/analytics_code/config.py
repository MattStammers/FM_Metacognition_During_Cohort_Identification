"""Configuration loading and validation for the analytics pipeline.

The pipeline is driven by a JSON document that follows the schema
documented in ``analytics_code/configs/analysis_config.example.json``.
This module exposes :class:`AnalysisConfig`, a thin typed wrapper around
the parsed JSON, and :func:`load_config`, the canonical entry point
used by the CLI and individual stages.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any


class ConfigError(ValueError):
    """Raised when the analysis configuration is missing or malformed."""


@dataclass(slots=True)
class AnalysisConfig:
    """Typed wrapper around a parsed analytics configuration document.

    Attributes
    ----------
    raw:
        The decoded JSON document as a plain ``dict``. All keys are
        preserved so individual stages can read additional sections
        (for example ``runner_metadata``) without going through this
        class.
    config_path:
        Absolute path to the JSON file that was loaded; used to resolve
        relative paths in the ``paths`` block.
    """

    raw: dict[str, Any]
    config_path: Path

    @property
    def paths(self) -> dict[str, Path]:
        """Return the ``paths`` block with every entry resolved to an absolute path.

        Relative paths are resolved against the directory containing the
        configuration file.
        """
        values = self.raw.get("paths", {})
        return {
            key: _resolve_path(self.config_path.parent, value)
            for key, value in values.items()
        }

    @property
    def data_prep(self) -> dict[str, Any]:
        """Return the ``data_prep`` sub-section, or an empty mapping if absent."""
        return self.raw.get("data_prep", {})

    @property
    def analysis(self) -> dict[str, Any]:
        """Return the ``analysis`` sub-section, or an empty mapping if absent."""
        return self.raw.get("analysis", {})

    @property
    def model_mapping(self) -> dict[str, str]:
        """Return ``model_mapping`` coerced to ``dict[str, str]``.

        This is consumed by :func:`analytics_code.common.canonicalize_model`
        to collapse runner-specific model identifiers to a canonical set
        of display labels.
        """
        return {str(k): str(v) for k, v in self.raw.get("model_mapping", {}).items()}


def _resolve_path(base_dir: Path, value: str | Path) -> Path:
    """Return ``value`` as an absolute :class:`Path`, resolving against ``base_dir``."""
    path = Path(value)
    return path if path.is_absolute() else (base_dir / path).resolve()


def load_config(config_path: str | Path) -> AnalysisConfig:
    """Read and validate an analytics configuration document.

    Parameters
    ----------
    config_path:
        Path to the JSON configuration file.

    Returns
    -------
    AnalysisConfig
        The validated configuration.

    Raises
    ------
    ConfigError
        If the file is missing, contains invalid JSON, or fails the
        structural checks performed by :func:`_validate_config`.
    """
    path = Path(config_path).resolve()
    if not path.exists():
        raise ConfigError(f"Config file not found: {path}")

    try:
        raw = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ConfigError(f"Invalid JSON in config file: {path}") from exc

    _validate_config(raw, path)
    return AnalysisConfig(raw=raw, config_path=path)


def _validate_config(raw: dict[str, Any], path: Path) -> None:
    """Validate the minimal structural requirements of a config document."""
    if not isinstance(raw, dict):
        raise ConfigError(f"Config must be a JSON object: {path}")

    required_paths = {"output_root"}
    paths = raw.get("paths")
    if not isinstance(paths, dict):
        raise ConfigError("Config must include a 'paths' object")

    missing = required_paths - set(paths)
    if missing:
        raise ConfigError(f"Missing required path entries: {sorted(missing)}")

    thresholds = raw.get("analysis", {}).get("thresholds", [])
    if thresholds and not isinstance(thresholds, list):
        raise ConfigError("analysis.thresholds must be a list")
