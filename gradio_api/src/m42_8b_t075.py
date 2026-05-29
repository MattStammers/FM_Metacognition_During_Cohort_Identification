"""Gradio model server for ``m42-health/Llama3-Med42-8B`` at temperature 0.75.

Loads the model with the memory limits configured via ``MAX_MEMORY_PER_GPU``
and exposes the ``/user``, ``/bot`` and ``/chat`` endpoints expected by
:func:`python_client.api.call_gradio_chat`. All boot, model-loading and
worker-pool plumbing is delegated to :func:`server_common.serve_model`.
"""

from __future__ import annotations

from server_common import ModelServerSpec, serve_model

SPEC = ModelServerSpec(
    model_id="m42-health/Llama3-Med42-8B",
    demo_title="Med42 8B Gradio API",
    prompt_placeholder="Type a prompt for m42-health/Llama3-Med42-8B...",
    default_port=9002,
    log_filename="server9002.log",
    stop_token_ids=(6465, 128001),
    temperature=0.75,
    streamer_timeout=10.0,
    predict_get_timeout=0.1,
)


if __name__ == "__main__":
    serve_model(SPEC)
