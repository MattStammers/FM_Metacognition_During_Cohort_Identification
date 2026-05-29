"""Tests for analytics_code.config."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analytics_code.config import AnalysisConfig, ConfigError, load_config


def _write_config(tmp_path: Path, payload: dict) -> Path:
    p = tmp_path / "config.json"
    p.write_text(json.dumps(payload), encoding="utf-8")
    return p


def _minimal_config(tmp_path: Path, *, extra: dict | None = None) -> Path:
    data: dict = {
        "paths": {"output_root": "outputs"},
        "data_prep": {},
        "analysis": {},
        "model_mapping": {},
    }
    if extra:
        data.update(extra)
    return _write_config(tmp_path, data)


# ---------------------------------------------------------------------------
# load_config – happy path
# ---------------------------------------------------------------------------


def test_load_config_returns_analysis_config(tmp_path: Path) -> None:
    path = _minimal_config(tmp_path)
    config = load_config(path)
    assert isinstance(config, AnalysisConfig)


def test_load_config_resolves_relative_output_root(tmp_path: Path) -> None:
    path = _minimal_config(tmp_path)
    config = load_config(path)
    resolved = config.paths["output_root"]
    assert resolved.is_absolute()
    assert resolved == (tmp_path / "outputs").resolve()


def test_load_config_absolute_path_kept(tmp_path: Path) -> None:
    abs_out = tmp_path / "abs_output"
    data = {"paths": {"output_root": str(abs_out)}}
    path = _write_config(tmp_path, data)
    config = load_config(path)
    assert config.paths["output_root"] == abs_out.resolve()


def test_load_config_model_mapping_coerces_to_strings(tmp_path: Path) -> None:
    data = {
        "paths": {"output_root": "out"},
        "model_mapping": {1: "model_a", "key": 99},
    }
    path = _write_config(tmp_path, data)
    config = load_config(path)
    assert config.model_mapping["1"] == "model_a"
    assert config.model_mapping["key"] == "99"


def test_load_config_data_prep_and_analysis_properties(tmp_path: Path) -> None:
    data = {
        "paths": {"output_root": "out"},
        "data_prep": {"batch_glob": "*.csv"},
        "analysis": {"thresholds": [5, 7]},
    }
    path = _write_config(tmp_path, data)
    config = load_config(path)
    assert config.data_prep["batch_glob"] == "*.csv"
    assert config.analysis["thresholds"] == [5, 7]


# ---------------------------------------------------------------------------
# load_config – error cases
# ---------------------------------------------------------------------------


def test_load_config_raises_for_missing_file(tmp_path: Path) -> None:
    with pytest.raises(ConfigError, match="not found"):
        load_config(tmp_path / "no_such_file.json")


def test_load_config_raises_for_invalid_json(tmp_path: Path) -> None:
    p = tmp_path / "bad.json"
    p.write_text("{not valid json}", encoding="utf-8")
    with pytest.raises(ConfigError, match="Invalid JSON"):
        load_config(p)


def test_load_config_raises_when_paths_missing(tmp_path: Path) -> None:
    p = _write_config(tmp_path, {"analysis": {}})
    with pytest.raises(ConfigError, match="paths"):
        load_config(p)


def test_load_config_raises_when_output_root_missing(tmp_path: Path) -> None:
    p = _write_config(tmp_path, {"paths": {"batch_root": "batches"}})
    with pytest.raises(ConfigError, match="output_root"):
        load_config(p)


def test_load_config_raises_for_non_object_root(tmp_path: Path) -> None:
    p = tmp_path / "list.json"
    p.write_text("[1, 2, 3]", encoding="utf-8")
    with pytest.raises(ConfigError, match="JSON object"):
        load_config(p)


def test_load_config_raises_when_thresholds_not_list(tmp_path: Path) -> None:
    data = {
        "paths": {"output_root": "out"},
        "analysis": {"thresholds": "bad"},
    }
    p = _write_config(tmp_path, data)
    with pytest.raises(ConfigError, match="thresholds"):
        load_config(p)


# ---------------------------------------------------------------------------
# AnalysisConfig – empty / default properties
# ---------------------------------------------------------------------------


def test_analysis_config_defaults_on_empty_raw(tmp_path: Path) -> None:
    path = _minimal_config(tmp_path)
    config = load_config(path)
    assert config.data_prep == {}
    assert config.model_mapping == {}
