"""Wrappers around the Gradio HTTP API used by the experiment runner.

Provides token-aware truncation, JSON extraction from free-text
responses, and a retrying wrapper around the Gradio chat endpoints.

Requests are submitted via the Gradio ``/user`` and ``/bot`` API
routes (represented in ``gradio_client`` calls as the doubled-slash
``//user`` / ``//bot`` strings that the client library expects),
with a fallback to the single-shot ``/chat`` route when the
streaming pair fails.
"""

from __future__ import annotations

import json
import logging
import random
import time
from concurrent.futures import ThreadPoolExecutor, TimeoutError

import tiktoken
from gradio_client import Client

DEFAULT_PREDICT_TIMEOUT_SECONDS = 300.0


def get_tokenizer(encoding_name: str) -> tiktoken.Encoding:
    """Return a :mod:`tiktoken` encoder by name (e.g. ``"cl100k_base"``)."""
    return tiktoken.get_encoding(encoding_name)


def truncate_to_token_limit(
    message: str, tokenizer: tiktoken.Encoding, token_limit: int
) -> tuple[str, bool]:
    """Truncate ``message`` to at most ``token_limit`` tokens.

    Returns ``(possibly_truncated_message, was_truncated)``.
    """
    tokens = tokenizer.encode(message)
    if len(tokens) <= token_limit:
        return message, False
    truncated_tokens = tokens[:token_limit]
    return tokenizer.decode(truncated_tokens), True


def extract_json(response: str) -> str | None:
    """Extract a JSON object from a model response.

    Recognises both fenced ```` ```json ... ``` ```` blocks and bare
    JSON objects embedded in free text. Returns the JSON text on
    success, or ``None`` if no parseable JSON object can be located.
    """
    if not response:
        return None
    if "```json" in response:
        json_start = response.find("```json") + len("```json")
        json_end = response.find("```", json_start)
        if json_end == -1:
            json_end = len(response)
        payload = response[json_start:json_end].strip()
    else:
        start = response.find("{")
        end = response.rfind("}")
        payload = (
            response[start : end + 1].strip()
            if start != -1 and end != -1 and end > start
            else ""
        )

    if not payload:
        return None

    try:
        json.loads(payload)
        return payload
    except json.JSONDecodeError:
        return None


def _predict_with_timeout(
    client: Client,
    *args: object,
    api_name: str,
    timeout_seconds: float,
) -> object:
    """Run one Gradio predict call with a hard timeout."""
    executor = ThreadPoolExecutor(max_workers=1)
    future = executor.submit(client.predict, *args, api_name=api_name)
    try:
        return future.result(timeout=timeout_seconds)
    except TimeoutError as exc:
        future.cancel()
        raise TimeoutError(
            f"Gradio predict call {api_name} exceeded "
            f"{timeout_seconds:.1f}s timeout"
        ) from exc
    finally:
        executor.shutdown(wait=False, cancel_futures=True)


def call_gradio_chat(
    api_endpoint: str,
    message: str,
    *,
    retry_delay_seconds: float,
    max_retries: int,
    tokenizer: tiktoken.Encoding,
    token_limit: int,
    predict_timeout_seconds: float = DEFAULT_PREDICT_TIMEOUT_SECONDS,
) -> tuple[str, str, bool]:
    """Call a Gradio chat endpoint with retry, token-truncation and JSON extraction.

    Tries the streaming ``/user`` + ``/bot`` API first; falls back to
    the single-shot ``/chat`` API when the streaming pair fails.
    Failures are retried with exponential backoff up to ``max_retries``
    times.

    Parameters
    ----------
    api_endpoint:
        URL of the Gradio server.
    message:
        User message to send. Truncated to ``token_limit`` tokens.
    retry_delay_seconds:
        Base delay (in seconds) for the exponential-backoff schedule.
    max_retries:
        Total attempts before propagating the final failure.
    tokenizer:
        :mod:`tiktoken` encoder used for the truncation step.
    token_limit:
        Maximum number of tokens accepted by the model.
    predict_timeout_seconds:
        Maximum wall-clock seconds allowed for each individual
        ``client.predict`` call before it is treated as failed and the
        retry/fallback path is used.

    Returns
    -------
    tuple[str, str, bool]
        ``(full_response, json_response, was_truncated)``. On failure
        the response strings are set to ``"API call failed"``.
    """
    truncated_message, was_truncated = truncate_to_token_limit(
        message, tokenizer, token_limit
    )
    client: Client | None = None
    full_response = "API call failed"
    json_response = "API call failed"

    try:
        client = Client(api_endpoint)
        for attempt in range(1, max_retries + 1):
            try:
                history: list[list[str]] = []
                try:
                    result_user = _predict_with_timeout(
                        client,
                        truncated_message,
                        history,
                        api_name="/user",
                        timeout_seconds=predict_timeout_seconds,
                    )
                    if (
                        not isinstance(result_user, list | tuple)
                        or len(result_user) < 2
                    ):
                        raise ValueError("Unexpected response shape from /user")

                    updated_history = result_user[1]
                    result_bot = _predict_with_timeout(
                        client,
                        updated_history,
                        api_name="/bot",
                        timeout_seconds=predict_timeout_seconds,
                    )
                    if (
                        isinstance(result_bot, list)
                        and result_bot
                        and isinstance(result_bot[-1], list | tuple)
                        and len(result_bot[-1]) >= 2
                    ):
                        full_response = str(result_bot[-1][1])
                    else:
                        raise ValueError("Unexpected response shape from /bot")
                except Exception as stream_exc:
                    logging.info(
                        "Falling back to /chat for %s after /user,/bot failed: %s",
                        api_endpoint,
                        stream_exc,
                    )
                    full_response = _predict_with_timeout(
                        client,
                        truncated_message,
                        api_name="/chat",
                        timeout_seconds=predict_timeout_seconds,
                    )

                if isinstance(full_response, str) and full_response.strip():
                    json_response = (
                        extract_json(full_response) or "No processable JSON response"
                    )
                    return full_response, json_response, was_truncated
                raise ValueError("Empty response from API")
            except Exception as exc:
                if attempt >= max_retries:
                    raise exc
                sleep_time = retry_delay_seconds * (
                    2 ** (attempt - 1)
                ) + random.uniform(0, 0.1)
                logging.warning(
                    "API call to %s failed on attempt %s/%s: %s. Retrying in %.2fs",
                    api_endpoint,
                    attempt,
                    max_retries,
                    exc,
                    sleep_time,
                )
                time.sleep(sleep_time)
    except Exception as exc:
        logging.error("API call failed for %s: %s", api_endpoint, exc)
        json_response = "API call failed"
    finally:
        if client is not None:
            try:
                client.close()
            except Exception as exc:
                logging.warning(
                    "Failed to close API client for %s: %s", api_endpoint, exc
                )

    return full_response, json_response, was_truncated
