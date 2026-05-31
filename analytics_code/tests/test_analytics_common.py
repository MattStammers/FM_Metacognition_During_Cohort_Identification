"""Tests for analytics_code.common."""

from __future__ import annotations

import json
from pathlib import Path

import pandas as pd
import pytest

from analytics_code.common import (
    canonicalize_model,
    ensure_dir,
    first_existing,
    format_sequence_label,
    write_dataframe,
    write_json,
)

# ---------------------------------------------------------------------------
# ensure_dir
# ---------------------------------------------------------------------------


def test_ensure_dir_creates_nested_directories(tmp_path: Path) -> None:
    target = tmp_path / "a" / "b" / "c"
    result = ensure_dir(target)
    assert result.is_dir()
    assert result == target


def test_ensure_dir_is_idempotent(tmp_path: Path) -> None:
    target = tmp_path / "existing"
    target.mkdir()
    result = ensure_dir(target)
    assert result.is_dir()


# ---------------------------------------------------------------------------
# write_dataframe
# ---------------------------------------------------------------------------


def test_write_dataframe_writes_csv(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1, 2], "b": [3, 4]})
    out = tmp_path / "out.csv"
    write_dataframe(df, out)
    loaded = pd.read_csv(out)
    assert list(loaded.columns) == ["a", "b"]
    assert len(loaded) == 2


def test_write_dataframe_writes_xlsx(tmp_path: Path) -> None:
    df = pd.DataFrame({"x": [10, 20]})
    out = tmp_path / "out.xlsx"
    write_dataframe(df, out)
    loaded = pd.read_excel(out)
    assert list(loaded["x"]) == [10, 20]


def test_write_dataframe_creates_parent_dirs(tmp_path: Path) -> None:
    df = pd.DataFrame({"v": [1]})
    out = tmp_path / "deep" / "dir" / "out.csv"
    write_dataframe(df, out)
    assert out.exists()


def test_write_dataframe_with_index(tmp_path: Path) -> None:
    df = pd.DataFrame({"a": [1]}, index=["row0"])
    out = tmp_path / "indexed.csv"
    write_dataframe(df, out, index=True)
    loaded = pd.read_csv(out, index_col=0)
    assert loaded.index[0] == "row0"


# ---------------------------------------------------------------------------
# write_json
# ---------------------------------------------------------------------------


def test_write_json_writes_dict(tmp_path: Path) -> None:
    data = {"key": "value", "num": 42}
    out = tmp_path / "out.json"
    write_json(data, out)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == data


def test_write_json_writes_list(tmp_path: Path) -> None:
    data = [{"a": 1}, {"b": 2}]
    out = tmp_path / "out.json"
    write_json(data, out)
    loaded = json.loads(out.read_text(encoding="utf-8"))
    assert loaded == data


def test_write_json_creates_parent_dirs(tmp_path: Path) -> None:
    data = {"x": 1}
    out = tmp_path / "sub" / "out.json"
    write_json(data, out)
    assert out.exists()


# ---------------------------------------------------------------------------
# first_existing
# ---------------------------------------------------------------------------


def test_first_existing_returns_first_match() -> None:
    result = first_existing(["a", "b", "c"], ["b", "a"])
    assert result == "b"


def test_first_existing_returns_none_when_no_match() -> None:
    result = first_existing(["x", "y"], ["a", "b"])
    assert result is None


def test_first_existing_returns_none_for_empty_candidates() -> None:
    assert first_existing(["a", "b"], []) is None


def test_first_existing_returns_none_for_empty_items() -> None:
    assert first_existing([], ["a"]) is None


# ---------------------------------------------------------------------------
# canonicalize_model
# ---------------------------------------------------------------------------


def test_canonicalize_model_applies_mapping() -> None:
    mapping = {"gpt-4o": "GPT-4o", "deepseek_14b": "DeepSeek-14B"}
    assert canonicalize_model("gpt-4o", mapping) == "GPT-4o"


def test_canonicalize_model_returns_stripped_original_when_no_mapping() -> None:
    assert canonicalize_model("  unknown_model  ", {}) == "unknown_model"


def test_canonicalize_model_strips_whitespace_before_lookup() -> None:
    mapping = {"model_a": "Model A"}
    assert canonicalize_model("  model_a  ", mapping) == "Model A"


def test_canonicalize_model_falls_back_to_known_family_patterns() -> None:
    assert canonicalize_model("mixtral_t1_runner", {}) == "mixtral7b"
    assert canonicalize_model("deepseek14_t05_runner", {}) == "deepseek14b"
    assert canonicalize_model("m42_t1_runner", {}) == "m42_8b"


def test_format_sequence_label_shortens_clinical_context_labels() -> None:
    assert (
        format_sequence_label("all_docs_in_sequence")
        == "Preceding clinic letter / Endoscopy / Histology narrative order"
    )


def test_format_sequence_label_keeps_clinic_letter_timing_explicit() -> None:
    assert (
        format_sequence_label("clinic_following")
        == "Following clinic letter narrative order"
    )
