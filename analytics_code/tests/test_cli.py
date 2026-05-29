"""Tests for analytics_code.cli."""

from __future__ import annotations

import json
from pathlib import Path

import pytest

from analytics_code.cli import build_parser, main

# ---------------------------------------------------------------------------
# helpers
# ---------------------------------------------------------------------------


def _write_minimal_config(tmp_path: Path) -> Path:
    data = {
        "paths": {"output_root": str(tmp_path / "outputs")},
        "data_prep": {},
        "analysis": {},
        "model_mapping": {},
    }
    p = tmp_path / "config.json"
    p.write_text(json.dumps(data), encoding="utf-8")
    return p


# ---------------------------------------------------------------------------
# build_parser
# ---------------------------------------------------------------------------


def test_build_parser_validate_config_command(tmp_path: Path) -> None:
    config = _write_minimal_config(tmp_path)
    parser = build_parser()
    args = parser.parse_args(["validate-config", "--config", str(config)])
    assert args.command == "validate-config"
    assert args.config == str(config)


def test_build_parser_run_all_command(tmp_path: Path) -> None:
    config = _write_minimal_config(tmp_path)
    parser = build_parser()
    args = parser.parse_args(["run-all", "--config", str(config)])
    assert args.command == "run-all"


def test_build_parser_run_document_level_all_command(tmp_path: Path) -> None:
    config = _write_minimal_config(tmp_path)
    parser = build_parser()
    args = parser.parse_args(["run-document-level-all", "--config", str(config)])
    assert args.command == "run-document-level-all"


def test_build_parser_run_stage_command(tmp_path: Path) -> None:
    config = _write_minimal_config(tmp_path)
    parser = build_parser()
    args = parser.parse_args(
        ["run-stage", "--config", str(config), "--stage", "data_prep"]
    )
    assert args.command == "run-stage"
    assert args.stage == "data_prep"


def test_build_parser_rejects_unknown_stage(tmp_path: Path) -> None:
    config = _write_minimal_config(tmp_path)
    parser = build_parser()
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["run-stage", "--config", str(config), "--stage", "nonexistent"]
        )


# ---------------------------------------------------------------------------
# main – validate-config
# ---------------------------------------------------------------------------


def test_main_validate_config_returns_zero(tmp_path: Path) -> None:
    config = _write_minimal_config(tmp_path)
    result = main(["validate-config", "--config", str(config)])
    assert result == 0


def test_main_validate_config_raises_on_invalid_config(tmp_path: Path) -> None:
    bad_config = tmp_path / "bad.json"
    bad_config.write_text("{invalid json}", encoding="utf-8")
    with pytest.raises(Exception):
        main(["validate-config", "--config", str(bad_config)])
