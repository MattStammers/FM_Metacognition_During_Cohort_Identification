"""Tests for :mod:`server_common`."""

from __future__ import annotations

import os

from server_common import (
    build_logger,
    build_max_memory,
    configure_cuda_allocator,
    configure_gradio_debug_env,
    env_flag,
)


def test_env_flag_accepts_truthy_values(monkeypatch) -> None:
    monkeypatch.setenv("TEST_FLAG", "YeS")
    assert env_flag("TEST_FLAG") is True


def test_build_max_memory_uses_at_least_one_slot() -> None:
    assert build_max_memory("40GB", 0) == {0: "40GB"}
    assert build_max_memory("40GB", 2) == {0: "40GB", 1: "40GB"}


def test_configure_gradio_debug_env_sets_numeric_value() -> None:
    configure_gradio_debug_env(True)
    assert os.environ["GRADIO_DEBUG"] == "1"
    configure_gradio_debug_env(False)
    assert os.environ["GRADIO_DEBUG"] == "0"


def test_configure_cuda_allocator_sets_expected_value() -> None:
    configure_cuda_allocator()
    assert os.environ["PYTORCH_CUDA_ALLOC_CONF"] == "max_split_size_mb:512"


def test_build_logger_reuses_existing_handlers(tmp_path) -> None:
    logger = build_logger(
        logger_name="test.server.logger",
        logs_dir=str(tmp_path),
        log_filename="server.log",
    )
    handler_count = len(logger.handlers)
    same_logger = build_logger(
        logger_name="test.server.logger",
        logs_dir=str(tmp_path),
        log_filename="server.log",
    )
    assert logger is same_logger
    assert len(same_logger.handlers) == handler_count
