"""Chronological experiment client package.

This package wraps the orchestration code that pulls reports from the
local CSV dumps, builds chronologically-ordered prompts and dispatches
them to one or more Gradio-hosted model servers in parallel. The
command-line entry point lives in :mod:`python_client.cli`.
"""

__all__ = ["__version__"]

__version__ = "0.1.0"
