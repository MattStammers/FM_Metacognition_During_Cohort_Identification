"""Gradio model server for ``m42-health/Llama3-Med42-8B`` at temperature 1.0.

Mirror of :mod:`m42_8b_t075` exposed on port 9012 with the upper
non-thinking-family temperature setting.
"""

from __future__ import annotations

from server_common import ModelServerSpec, serve_model

SPEC = ModelServerSpec(
    model_id="m42-health/Llama3-Med42-8B",
    demo_title="Med42 8B Gradio API (T=1.0)",
    prompt_placeholder="Type a prompt for m42-health/Llama3-Med42-8B...",
    default_port=9012,
    log_filename="server9012.log",
    stop_token_ids=(6465, 128001),
    temperature=1.0,
    streamer_timeout=10.0,
    predict_get_timeout=0.1,
)


if __name__ == "__main__":
    serve_model(SPEC)
