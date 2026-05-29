"""Gradio model server for ``google/gemma-4-31b-it`` at temperature 0.6.

This server hosts the dense 31 B variant of the Gemma 4 family that was
released on 2 April 2026. It is included as a late-evaluation,
newer-generation open-weight model to test whether observed model
behaviours (bias, metacognition, dropout, reasoning, classification)
persist into a more recent model generation.

No quantisation is applied: the model is loaded in FP16 via
:func:`server_common.serve_model` (``torch.float16``,
``device_map="auto"``), matching the loader configuration used for
every model in the matrix.

The 31/32 B models are evaluated at a single temperature of 0.6,
following the DeepSeek-R1 manufacturer guidance and reducing
computational/energy cost.
"""

from __future__ import annotations

from server_common import ModelServerSpec, serve_model

SPEC = ModelServerSpec(
    # Hugging Face repository identifier for the dense 31B Gemma 4
    # checkpoint. Adjust at deployment time if Google publishes the
    # weights under a different repo name.
    model_id="google/gemma-4-31b-it",
    demo_title="Gemma 4 31B Gradio API (T=0.6)",
    prompt_placeholder="Type a prompt for google/gemma-4-31b-it...",
    default_port=9006,
    log_filename="server9006.log",
    stop_token_ids=(),
    temperature=0.6,
)


if __name__ == "__main__":
    serve_model(SPEC)
