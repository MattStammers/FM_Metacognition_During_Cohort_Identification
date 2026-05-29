"""Gradio model server for ``deepseek-ai/DeepSeek-R1-Distill-Qwen-14B`` at temperature 0.5.

Lower-temperature configuration for the 14B reasoning model used in
the temperature-testing arm, where the smaller thinking models are
evaluated at both 0.5 and 0.75 to characterise configuration
sensitivity within a feasible deployment range.
"""

from __future__ import annotations

from server_common import ModelServerSpec, serve_model

SPEC = ModelServerSpec(
    model_id="deepseek-ai/DeepSeek-R1-Distill-Qwen-14B",
    demo_title="DeepSeek 14B Gradio API (T=0.5)",
    prompt_placeholder=(
        "Type a prompt for deepseek-ai/DeepSeek-R1-Distill-Qwen-14B..."
    ),
    default_port=9013,
    log_filename="server9013.log",
    stop_token_ids=(30113, 5267),
    temperature=0.5,
)


if __name__ == "__main__":
    serve_model(SPEC)
