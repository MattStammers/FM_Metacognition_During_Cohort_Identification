"""Gradio model server for ``mistralai/Mistral-7B-Instruct-v0.3`` at temperature 1.0.

Mirror of :mod:`mixtral_7b_t075` exposed on port 9011 with the upper
non-thinking-family temperature setting.
"""

from __future__ import annotations

from server_common import ModelServerSpec, serve_model

SPEC = ModelServerSpec(
    model_id="mistralai/Mistral-7B-Instruct-v0.3",
    demo_title="Mixtral 7B Gradio API (T=1.0)",
    prompt_placeholder="Type a prompt for mistralai/Mistral-7B-Instruct-v0.3...",
    default_port=9011,
    log_filename="server9011.log",
    stop_token_ids=(29557,),
    temperature=1.0,
    streamer_timeout=10.0,
    predict_get_timeout=0.1,
)


if __name__ == "__main__":
    serve_model(SPEC)
