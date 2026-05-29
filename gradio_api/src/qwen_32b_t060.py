"""Gradio model server for ``Qwen/Qwen3-32B`` at temperature 0.60.

Loads the model sharded across the visible GPUs and exposes the
``/user``, ``/bot`` and ``/chat`` endpoints expected by
:func:`python_client.api.call_gradio_chat`. All boot, model-loading and
worker-pool plumbing is delegated to :func:`server_common.serve_model`.

Note
----
This script was previously pointed at ``Qwen/Qwen2.5-32B-Instruct``.
The experiment matrix references ``Qwen/Qwen3-32B``; the identifier
below is the authoritative value and any previously-recorded runs
against Qwen 2.5 must be regenerated before being reported.
"""

from __future__ import annotations

from server_common import ModelServerSpec, serve_model

SPEC = ModelServerSpec(
    model_id="Qwen/Qwen3-32B",
    demo_title="Qwen 32B Gradio API",
    prompt_placeholder="Type a prompt for Qwen/Qwen3-32B...",
    default_port=9005,
    log_filename="server9005.log",
    stop_token_ids=(151645,),
    temperature=0.60,
)


if __name__ == "__main__":
    serve_model(SPEC)
