"""Shared helpers for the Gradio model server scripts.

Each top-level model server script (``deepseek_14b_t075.py``,
``mixtral_7b_t075.py`` etc.) loads its tokenizer / model and then
composes a small Gradio :class:`gr.Blocks` UI from the helpers in this
module, so the wire-format exposed by every server is consistent.

Most servers can simply hand a small :class:`ModelServerSpec` to
:func:`serve_model`, which performs the entire boot sequence
(environment parsing, Hugging Face login, model loading, worker pool,
Gradio launch) using the shared implementation in this file.
"""

from __future__ import annotations

import gc
import logging
import os
import queue
import threading
import traceback
from collections.abc import Callable, Generator, Sequence
from contextlib import contextmanager
from dataclasses import dataclass, field
from logging.handlers import RotatingFileHandler
from pathlib import Path
from threading import Semaphore, Thread

import gradio as gr
from huggingface_hub import login

LOG_FORMAT = "%(asctime)s - %(levelname)s - %(filename)s:%(lineno)d - %(message)s"
LOG_DATE_FORMAT = "%Y-%m-%d %H:%M:%S"


def env_flag(name: str, default: bool = False) -> bool:
    """Return the boolean value of environment variable ``name``.

    Recognises ``1``, ``true``, ``yes`` and ``on`` (case-insensitive)
    as truthy. Returns ``default`` when the variable is unset.
    """
    value = os.getenv(name)
    if value is None:
        return default
    return value.strip().lower() in {"1", "true", "yes", "on"}


def configure_gradio_debug_env(enabled: bool) -> None:
    """Toggle the ``GRADIO_DEBUG`` environment flag."""
    os.environ["GRADIO_DEBUG"] = "1" if enabled else "0"


def build_logger(
    *, logger_name: str, logs_dir: str, log_filename: str
) -> logging.Logger:
    """Create or return a logger writing to both stderr and a rotating file.

    Subsequent calls with the same ``logger_name`` re-use the existing
    logger so that handlers are not duplicated when this helper is
    imported repeatedly by reloaded modules.
    """
    Path(logs_dir).mkdir(parents=True, exist_ok=True)

    logger = logging.getLogger(logger_name)
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    logger.propagate = False
    formatter = logging.Formatter(fmt=LOG_FORMAT, datefmt=LOG_DATE_FORMAT)

    console_handler = logging.StreamHandler()
    console_handler.setLevel(logging.INFO)
    console_handler.setFormatter(formatter)

    file_handler = RotatingFileHandler(
        os.path.join(logs_dir, log_filename),
        mode="a",
        maxBytes=5 * 1024 * 1024,
        backupCount=5,
        encoding="utf-8",
    )
    file_handler.setLevel(logging.INFO)
    file_handler.setFormatter(formatter)

    logger.addHandler(console_handler)
    logger.addHandler(file_handler)
    return logger


def login_to_hugging_face(
    *,
    logger: logging.Logger,
    access_token: str | None,
    token_path: str = "",
    warn_if_missing: bool = True,
) -> str | None:
    """Authenticate against Hugging Face Hub using a token.

    Prefers ``access_token`` when set, otherwise reads it from
    ``token_path``. Returns the token on success or ``None`` when no
    token is available; when ``warn_if_missing`` is true the absence
    is logged at WARNING level, otherwise at ERROR level.
    """
    token = (access_token or "").strip()
    if not token and token_path and os.path.exists(token_path):
        token = Path(token_path).read_text(encoding="utf-8").strip()

    if token:
        login(token=token)
        return token

    if warn_if_missing:
        logger.warning(
            "No Hugging Face token available; private model access may fail."
        )
    else:
        logger.error("Access token is missing.")
    return None


def configure_cuda_allocator() -> None:
    """Set ``PYTORCH_CUDA_ALLOC_CONF`` to a conservative ``max_split_size_mb``."""
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "max_split_size_mb:512"


def build_max_memory(max_memory_per_gpu: str, device_count: int) -> dict[int, str]:
    """Build the ``max_memory`` mapping passed to ``from_pretrained``.

    Returns ``{gpu_index: max_memory_per_gpu}`` for ``device_count``
    GPUs (always at least one entry).
    """
    return {gpu_index: max_memory_per_gpu for gpu_index in range(max(device_count, 1))}


def build_standard_demo(
    *,
    title: str,
    prompt_placeholder: str,
    response_label: str,
    predict_fn: Callable[[str, list[list[str]]], Generator[str, None, None]],
    queue_max_size: int = 50,
) -> gr.Blocks:
    """Compose the standard Gradio :class:`gr.Blocks` UI used by every model server.

    Wires up the ``/user``, ``/bot``, ``/lambda`` and ``/chat`` API
    endpoints expected by :func:`python_client.api.call_gradio_chat`,
    along with a minimal interactive web UI that simply calls
    ``predict_fn`` once per click.

    Parameters
    ----------
    title:
        Heading displayed in the web UI and used as the page title.
    prompt_placeholder:
        Placeholder shown in the prompt textbox.
    response_label:
        Label of the response textbox.
    predict_fn:
        Generator-returning callable that produces incremental response
        chunks given a message and a chat history.
    queue_max_size:
        Maximum size of the Gradio request queue.
    """

    def one_shot_chat(message: str) -> str:
        return "".join(predict_fn(message, []))

    def user_fn(message: str, history: list[list[str]] | None):
        normalized_history = history or []
        return "", normalized_history + [[message, ""]]

    def bot_fn(history: list[list[str]] | None):
        normalized_history = history or []
        if not normalized_history:
            return []
        message = normalized_history[-1][0]
        final_response = ""
        for chunk in predict_fn(message, normalized_history[:-1]):
            final_response = chunk
        normalized_history[-1][1] = final_response
        return normalized_history

    with gr.Blocks(title=title) as demo:
        gr.Markdown(title)
        message_box = gr.Textbox(
            lines=8,
            placeholder=prompt_placeholder,
            label="Prompt",
        )
        response_box = gr.Textbox(lines=16, label=response_label)
        history_state = gr.JSON(value=[], visible=False, label="History")
        run_button = gr.Button("Generate")
        clear_button = gr.Button("Clear")

        run_button.click(one_shot_chat, inputs=[message_box], outputs=[response_box])
        clear_button.click(
            lambda: ("", "", []),
            inputs=None,
            outputs=[message_box, response_box, history_state],
            queue=False,
        )

        user_api_button = gr.Button(visible=False)
        bot_api_button = gr.Button(visible=False)
        lambda_api_button = gr.Button(visible=False)

        user_api_button.click(
            user_fn,
            inputs=[message_box, history_state],
            outputs=[message_box, history_state],
            api_name="/user",
            queue=False,
        )
        bot_api_button.click(
            bot_fn,
            inputs=[history_state],
            outputs=[history_state],
            api_name="/bot",
        )
        lambda_api_button.click(
            lambda: [],
            inputs=None,
            outputs=[history_state],
            api_name="/lambda",
            queue=False,
        )

        gr.Interface(
            fn=one_shot_chat,
            inputs="text",
            outputs="text",
            api_name="/chat",
        )

    demo.queue(max_size=queue_max_size)
    return demo


# ---------------------------------------------------------------------------
# High-level model-server orchestration
# ---------------------------------------------------------------------------


@dataclass(frozen=True)
class ModelServerSpec:
    """Per-model configuration consumed by :func:`serve_model`.

    Centralises everything that varies between the otherwise identical
    Gradio model-server scripts (model id, decoding temperature, stop
    tokens, port, demo labels) so each script reduces to a single
    declarative spec plus a call to :func:`serve_model`.

    Parameters
    ----------
    model_id:
        Hugging Face model identifier passed to
        ``AutoTokenizer.from_pretrained`` and
        ``AutoModelForCausalLM.from_pretrained``.
    demo_title:
        Heading shown in the Gradio UI and used as the page title.
    prompt_placeholder:
        Placeholder text shown in the prompt input box.
    default_port:
        TCP port used when ``GRADIO_SERVER_PORT`` is unset.
    log_filename:
        File name (under ``LOGS_DIR``) for the rotating server log.
    stop_token_ids:
        Token ids that terminate generation when produced as the most
        recent token.
    temperature:
        Sampling temperature passed to ``model.generate``.
    streamer_timeout:
        Per-token timeout (seconds) for the
        :class:`~transformers.TextIteratorStreamer`.
    max_new_tokens, top_p, top_k:
        Standard ``model.generate`` decoding controls. ``max_new_tokens``
        is treated as an upper cap: at request time the server
        computes ``min(max_new_tokens, model_context_window - input_tokens)``
        so the combined input + output never exceeds the model's
        context window.
    model_context_window:
        Hard upper bound on input + output tokens for this model. The
        default of 4096 matches the smaller instruction-tuned models
        served here; override for larger-context models (e.g. Qwen3
        32B). The actual window read at boot time from the loaded
        tokenizer / model takes precedence when available.
    min_new_tokens:
        Floor on the dynamically-computed generation budget so that
        heavily truncated prompts still produce a usable response.
    use_chat_template:
        When ``True`` (default) the request prompt is wrapped with
        ``tokenizer.apply_chat_template`` so each instruction-tuned
        model sees its native ``[INST]`` / ``<|im_start|>`` / etc.
        framing. Set to ``False`` only for base-completion models.
    max_concurrent_threads, max_queue_size, num_workers:
        Concurrency limits for the in-process request pipeline.
    predict_get_timeout:
        Poll interval used when forwarding worker output back to the
        Gradio request thread.
    """

    model_id: str
    demo_title: str
    prompt_placeholder: str
    default_port: int
    log_filename: str
    stop_token_ids: Sequence[int]
    temperature: float
    streamer_timeout: float = 60.0
    max_new_tokens: int = 4096
    top_p: float = 0.95
    top_k: int = 1000
    model_context_window: int = 4096
    min_new_tokens: int = 128
    use_chat_template: bool = True
    max_concurrent_threads: int = 5
    max_queue_size: int = 50
    num_workers: int = 5
    predict_get_timeout: float = 0.5
    response_label: str = "Response"
    extra_from_pretrained_kwargs: dict[str, object] = field(default_factory=dict)


def _build_stop_criteria(stop_token_ids: Sequence[int]):
    """Return a ``StoppingCriteriaList`` that stops on any of ``stop_token_ids``.

    Imported lazily so :mod:`server_common` can be imported in
    environments that do not have :mod:`torch` / :mod:`transformers`
    installed (e.g. unit tests).
    """
    import torch
    from transformers import StoppingCriteria, StoppingCriteriaList

    stop_set = set(int(tok) for tok in stop_token_ids)

    class StopOnTokens(StoppingCriteria):
        def __call__(
            self,
            input_ids: "torch.LongTensor",
            scores: "torch.FloatTensor",
            **kwargs: object,
        ) -> bool:
            last = int(input_ids[0][-1])
            return last in stop_set

    return StoppingCriteriaList([StopOnTokens()])


def serve_model(spec: ModelServerSpec) -> None:
    """Boot, configure and launch a Gradio model server from ``spec``.

    Performs the full sequence shared by every model server script:

    1. Read environment overrides (``GRADIO_SERVER_NAME``,
       ``GRADIO_SERVER_PORT``, ``GRADIO_DEBUG``,
       ``MAX_MEMORY_PER_GPU``, ``LOGS_DIR``, ``HF_ACCESS_TOKEN``).
    2. Build the rotating-file logger, log in to Hugging Face, and
       configure the CUDA allocator.
    3. Load the tokenizer and model with ``device_map="auto"`` and the
       per-GPU memory cap, offloading state to ``offload/`` on disk.
    4. Spawn a fixed-size worker thread pool that reads prompts from a
       bounded request queue and streams tokens back to callers via a
       per-request response queue.
    5. Wire the ``predict`` generator into :func:`build_standard_demo`
       and launch the resulting Gradio app.

    The function blocks until the Gradio server is shut down.
    """
    import torch
    from transformers import AutoModelForCausalLM, AutoTokenizer, TextIteratorStreamer

    access_token = os.getenv("HF_ACCESS_TOKEN")
    server_name = os.getenv("GRADIO_SERVER_NAME", "0.0.0.0")
    server_port = int(os.getenv("GRADIO_SERVER_PORT", str(spec.default_port)))
    gradio_debug = env_flag("GRADIO_DEBUG")
    max_memory_per_gpu = os.getenv("MAX_MEMORY_PER_GPU", "40GB")
    logs_dir = os.getenv("LOGS_DIR", "logs")

    logger = build_logger(
        logger_name=f"model_server.{server_port}",
        logs_dir=logs_dir,
        log_filename=spec.log_filename,
    )
    configure_gradio_debug_env(gradio_debug)
    login_to_hugging_face(
        logger=logger,
        access_token=access_token,
        warn_if_missing=False,
    )
    configure_cuda_allocator()

    torch.cuda.empty_cache()
    gc.collect()

    max_memory = build_max_memory(max_memory_per_gpu, torch.cuda.device_count())

    logger.info("Loading tokenizer and model %s ...", spec.model_id)
    try:
        tokenizer = AutoTokenizer.from_pretrained(spec.model_id, use_fast=True)
        model = AutoModelForCausalLM.from_pretrained(
            spec.model_id,
            torch_dtype=torch.float16,
            device_map="auto",
            offload_folder="offload",
            offload_state_dict=True,
            max_memory=max_memory,
            **spec.extra_from_pretrained_kwargs,
        )
        logger.info("Tokenizer and model loaded successfully.")
    except Exception:
        logger.error("Failed to load tokenizer and model:")
        logger.error(traceback.format_exc())
        raise

    semaphore = Semaphore(spec.max_concurrent_threads)
    request_queue: "queue.Queue[tuple[str, list, queue.Queue] | None]" = queue.Queue(
        spec.max_queue_size
    )

    @contextmanager
    def acquire_semaphore(sem: Semaphore):
        sem.acquire()
        try:
            yield
        finally:
            sem.release()

    def _generate_text(
        generate_kwargs: dict,
        exception_event: threading.Event,
        exception_container: list,
    ) -> None:
        try:
            model.generate(**generate_kwargs)
        except Exception as exc:  # noqa: BLE001 - propagated via container
            exception_container.append(exc)
            exception_event.set()

    def _resolve_context_window() -> int:
        """Pick the smallest sensible context window for this model.

        Reads, in order: the model config ``max_position_embeddings``
        attribute, the tokenizer ``model_max_length`` (capped to a
        sane upper bound when the tokenizer reports an oversized
        sentinel), and finally falls back to ``spec.model_context_window``.
        """
        from_model = getattr(
            getattr(model, "config", None), "max_position_embeddings", None
        )
        from_tokenizer = getattr(tokenizer, "model_max_length", None)
        candidates: list[int] = [spec.model_context_window]
        if isinstance(from_model, int) and from_model > 0:
            candidates.append(int(from_model))
        if isinstance(from_tokenizer, int) and 0 < from_tokenizer < 10_000_000:
            candidates.append(int(from_tokenizer))
        return min(candidates)

    context_window = _resolve_context_window()

    def _build_input_ids(message: str, history: list):
        """Format the request as a chat-template tokenisation when supported.

        Falls back to the original ``<lang>en</lang>\n<human>:.../<bot>:...``
        framing when the tokenizer does not expose ``apply_chat_template``
        or when the spec opts out via ``use_chat_template=False``.
        Returns the encoded :class:`~transformers.BatchEncoding`.
        """
        if spec.use_chat_template and hasattr(tokenizer, "apply_chat_template"):
            messages = []
            for item in history or []:
                if len(item) >= 1 and item[0]:
                    messages.append({"role": "user", "content": str(item[0])})
                if len(item) >= 2 and item[1]:
                    messages.append({"role": "assistant", "content": str(item[1])})
            messages.append({"role": "user", "content": message})
            try:
                input_ids = tokenizer.apply_chat_template(
                    messages,
                    add_generation_prompt=True,
                    return_tensors="pt",
                )
                return {"input_ids": input_ids.to(model.device)}
            except Exception as exc:  # noqa: BLE001 - fall back to legacy framing
                logger.warning(
                    "apply_chat_template failed (%s); falling back to legacy framing",
                    exc,
                )

        history_format = (history or []) + [[message, ""]]
        prompt = "<lang>en</lang>" + "".join(
            "\n<human>:" + item[0] + "\n<bot>:" + (item[1] or "")
            for item in history_format
        )
        encoded = tokenizer([prompt], return_tensors="pt")
        return {key: value.to(model.device) for key, value in encoded.items()}

    def generate_response(message: str, history: list) -> Generator[str, None, None]:
        try:
            torch.cuda.empty_cache()
            gc.collect()

            inputs = _build_input_ids(message, history)
            input_ids = inputs.get("input_ids")
            input_tokens = int(input_ids.shape[-1]) if input_ids is not None else 0
            available = max(
                spec.min_new_tokens,
                min(spec.max_new_tokens, context_window - input_tokens),
            )
            streamer = TextIteratorStreamer(
                tokenizer,
                timeout=spec.streamer_timeout,
                skip_prompt=True,
                skip_special_tokens=True,
            )
            generate_kwargs = dict(
                **inputs,
                streamer=streamer,
                max_new_tokens=available,
                do_sample=True,
                top_p=spec.top_p,
                top_k=spec.top_k,
                temperature=spec.temperature,
                num_beams=1,
                stopping_criteria=_build_stop_criteria(spec.stop_token_ids),
            )
            logger.info(
                "generate input_tokens=%s context_window=%s max_new_tokens=%s",
                input_tokens,
                context_window,
                available,
            )

            exception_event = threading.Event()
            exception_container: list = []

            with acquire_semaphore(semaphore):
                worker_thread = Thread(
                    target=_generate_text,
                    args=(generate_kwargs, exception_event, exception_container),
                )
                worker_thread.start()

                partial = ""
                try:
                    for new_token in streamer:
                        if exception_event.is_set():
                            raise exception_container[0]
                        if new_token.strip() and new_token != "<":
                            partial += new_token
                            yield partial
                except GeneratorExit:
                    logger.info(
                        "Generator was closed by the client. Cleaning up resources."
                    )
                    raise
                except Exception as exc:  # noqa: BLE001 - reported to caller
                    logger.error(
                        "Exception in generate_response for message: %s", message
                    )
                    logger.error(traceback.format_exc())
                    yield f"An error occurred during text generation: {exc}"
                finally:
                    worker_thread.join()
                    torch.cuda.empty_cache()
                    gc.collect()
        except Exception as exc:  # noqa: BLE001 - reported to caller
            logger.error("Exception in generate_response outer: %s", exc)
            logger.error(traceback.format_exc())
            yield f"An error occurred in generate_response: {exc}"

    def predict(message: str, history: list) -> Generator[str, None, None]:
        if not message.strip():
            yield "Please enter a message."
            return

        try:
            response_queue: "queue.Queue[str | None]" = queue.Queue()
            request_queue.put_nowait((message, history, response_queue))
        except queue.Full:
            yield "Server is busy. Please try again later."
            return

        while True:
            try:
                response = response_queue.get(timeout=spec.predict_get_timeout)
            except queue.Empty:
                continue
            if response is None:
                break
            yield response

    def worker() -> None:
        while True:
            try:
                item = request_queue.get()
                if item is None:
                    break
                msg, hist, resp_q = item
                generator = generate_response(msg, hist)
                try:
                    for response in generator:
                        resp_q.put(response)
                except GeneratorExit:
                    logger.info("Worker detected client disconnection.")
                    resp_q.put("Client disconnected.")
                except Exception as exc:  # noqa: BLE001 - reported to client
                    logger.error(
                        "Exception in worker during generate_response: %s", exc
                    )
                    logger.error(traceback.format_exc())
                    resp_q.put(f"An error occurred: {exc}")
                finally:
                    resp_q.put(None)
                    request_queue.task_done()
            except Exception:  # noqa: BLE001 - never let the worker die silently
                logger.error("Exception in worker:")
                logger.error(traceback.format_exc())

    workers: list[Thread] = []
    for _ in range(spec.num_workers):
        thread = Thread(target=worker, daemon=True)
        thread.start()
        workers.append(thread)

    demo = build_standard_demo(
        title=spec.demo_title,
        prompt_placeholder=spec.prompt_placeholder,
        response_label=spec.response_label,
        predict_fn=predict,
        queue_max_size=spec.max_queue_size,
    )

    demo.launch(
        server_name=server_name,
        server_port=server_port,
        show_error=True,
        debug=gradio_debug,
    )
