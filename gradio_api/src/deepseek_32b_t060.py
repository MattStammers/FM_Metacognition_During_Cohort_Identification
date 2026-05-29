"""Gradio model server for ``deepseek-ai/DeepSeek-R1-Distill-Qwen-32B`` at temperature 0.60.

Loads the model sharded across the visible GPUs and exposes the
``/user``, ``/bot`` and ``/chat`` endpoints expected by
:func:`python_client.api.call_gradio_chat`. All boot, model-loading and
worker-pool plumbing is delegated to :func:`server_common.serve_model`.
"""

from __future__ import annotations

from server_common import ModelServerSpec, serve_model

SPEC = ModelServerSpec(
    model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-32B",
    demo_title="DeepSeek 32B Gradio API",
    prompt_placeholder=(
        "Type a prompt for deepseek-ai/DeepSeek-R1-Distill-Qwen-32B..."
    ),
    default_port=9004,
    log_filename="server9004.log",
    stop_token_ids=(30113, 5267),
    temperature=0.60,
)


if __name__ == "__main__":
    serve_model(SPEC)
