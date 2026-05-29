"""Gradio model server for ``deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`` at temperature 0.75.

Loads the model with the memory limits configured via ``MAX_MEMORY_PER_GPU``
and exposes the ``/user``, ``/bot`` and ``/chat`` endpoints expected by
:func:`python_client.api.call_gradio_chat`. All boot, model-loading and
worker-pool plumbing is delegated to :func:`server_common.serve_model`.
"""

from __future__ import annotations

from server_common import ModelServerSpec, serve_model

SPEC = ModelServerSpec(
    model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    demo_title="DeepSeek 14B Gradio API",
    prompt_placeholder=(
        "Type a prompt for deepseek-ai/DeepSeek-R1-Distill-Qwen-14B..."
    ),
    default_port=9003,
    log_filename="server9003.log",
    stop_token_ids=(30113, 5267),
    temperature=0.75,
)


if __name__ == "__main__":
    serve_model(SPEC)
