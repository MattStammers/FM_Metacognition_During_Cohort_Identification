"""Gradio model server for ``mistralai/Mistral-7B-Instruct-v0.3`` at temperature 0.75.

Loads the model with the memory limits configured via ``MAX_MEMORY_PER_GPU``
and exposes the ``/user``, ``/bot`` and ``/chat`` endpoints expected by
:func:`python_client.api.call_gradio_chat`. All boot, model-loading and
worker-pool plumbing is delegated to :func:`server_common.serve_model`.
"""

from __future__ import annotations

from server_common import ModelServerSpec, serve_model

SPEC = ModelServerSpec(
    model_id="mistralai/Mistral-7B-Instruct-v0.3",
    demo_title="Mixtral 7B Gradio API",
    prompt_placeholder="Type a prompt for mistralai/Mistral-7B-Instruct-v0.3...",
    default_port=9001,
    log_filename="server9001.log",
    stop_token_ids=(29557,),
    temperature=0.75,
    streamer_timeout=10.0,
    predict_get_timeout=0.1,
)


if __name__ == "__main__":
    serve_model(SPEC)
