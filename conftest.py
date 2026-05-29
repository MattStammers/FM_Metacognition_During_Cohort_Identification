"""Workspace-wide pytest configuration.

Makes the ``src`` directories of every package importable so tests can
refer to ``analytics_code``, ``python_client`` and ``gradio_api``
directly without an editable install.
"""

from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parent
for relative_path in ("python_client/src", "analytics_code/src", "gradio_api/src"):
    candidate = ROOT / relative_path
    candidate_str = str(candidate)
    if candidate.exists() and candidate_str not in sys.path:
        sys.path.insert(0, candidate_str)
