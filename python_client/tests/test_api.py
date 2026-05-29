"""Tests for :mod:`python_client.api`."""

from __future__ import annotations

import time

from python_client.api import call_gradio_chat, extract_json, truncate_to_token_limit


class FakeEncoding:
    def encode(self, message: str) -> list[int]:
        return [ord(char) for char in message]

    def decode(self, tokens: list[int]) -> str:
        return "".join(chr(token) for token in tokens)


class FakeClient:
    raise_stream = False
    hang_chat = False

    def __init__(self, endpoint: str):
        self.endpoint = endpoint
        self.calls: list[tuple[str, tuple[object, ...]]] = []

    def predict(self, *args, api_name: str):
        self.calls.append((api_name, args))
        if api_name == "/user":
            if self.raise_stream:
                raise RuntimeError("stream unavailable")
            return ["", [[args[0], ""]]]
        if api_name == "/bot":
            return [[args[0][0][0], '{"status": "ok"}']]
        if api_name == "/chat":
            if self.hang_chat:
                time.sleep(1)
            return '{"status": "fallback"}'
        raise AssertionError(f"Unexpected api_name: {api_name}")

    def close(self) -> None:
        return None


def test_truncate_to_token_limit_returns_flag() -> None:
    message, was_truncated = truncate_to_token_limit(
        "abcd", FakeEncoding(), token_limit=3
    )
    assert message == "abc"
    assert was_truncated is True


def test_extract_json_finds_code_block_payload() -> None:
    response = 'before```json\n{"value": 1}\n```after'
    assert extract_json(response) == '{"value": 1}'


def test_call_gradio_chat_uses_streaming_when_available(monkeypatch) -> None:
    FakeClient.raise_stream = False
    FakeClient.hang_chat = False
    monkeypatch.setattr("python_client.api.Client", FakeClient)
    full_response, json_response, was_truncated = call_gradio_chat(
        "http://example",
        "hello",
        retry_delay_seconds=0,
        max_retries=1,
        tokenizer=FakeEncoding(),
        token_limit=10,
    )
    assert full_response == '{"status": "ok"}'
    assert json_response == '{"status": "ok"}'
    assert was_truncated is False


def test_call_gradio_chat_falls_back_to_chat(monkeypatch) -> None:
    FakeClient.raise_stream = True
    FakeClient.hang_chat = False
    monkeypatch.setattr("python_client.api.Client", FakeClient)
    full_response, json_response, _ = call_gradio_chat(
        "http://example",
        "hello",
        retry_delay_seconds=0,
        max_retries=1,
        tokenizer=FakeEncoding(),
        token_limit=10,
    )
    assert full_response == '{"status": "fallback"}'
    assert json_response == '{"status": "fallback"}'


def test_call_gradio_chat_times_out_hung_predict(monkeypatch) -> None:
    FakeClient.raise_stream = True
    FakeClient.hang_chat = True
    monkeypatch.setattr("python_client.api.Client", FakeClient)
    full_response, json_response, _ = call_gradio_chat(
        "http://example",
        "hello",
        retry_delay_seconds=0,
        max_retries=1,
        tokenizer=FakeEncoding(),
        token_limit=10,
        predict_timeout_seconds=0.01,
    )
    assert full_response == "API call failed"
    assert json_response == "API call failed"
