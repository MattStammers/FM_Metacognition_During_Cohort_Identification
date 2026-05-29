"""Run manifest helpers for the python client.

Emits a machine-readable record of every (model, prompt, sequence,
batch) request issued by :func:`run_chronological_experiments` so the
production matrix can be reconstructed from the on-disk outputs alone.
"""

from __future__ import annotations

import hashlib
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


def hash_text(text: str) -> str:
    """Return the SHA-256 hex digest of ``text``."""
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def hash_file(path: Path) -> str | None:
    """Return the SHA-256 hex digest of ``path``'s contents, or ``None`` if missing."""
    try:
        return hash_text(Path(path).read_text(encoding="utf-8"))
    except OSError:
        return None


def write_manifest_entry(manifest_path: Path, entry: dict[str, Any]) -> None:
    """Append a single jsonl entry to ``manifest_path`` (creating parents)."""
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    record = {
        "timestamp": datetime.now(timezone.utc).isoformat(),
        "hostname": socket.gethostname(),
        "pid": os.getpid(),
        "python_version": sys.version.split()[0],
        **entry,
    }
    with manifest_path.open("a", encoding="utf-8") as handle:
        handle.write(json.dumps(record, sort_keys=True, default=str))
        handle.write("\n")


def build_run_manifest_path(output_dir: Path) -> Path:
    """Return the canonical manifest path under ``output_dir``."""
    return Path(output_dir) / "run_manifest.jsonl"


def now_unix() -> float:
    """Return the current wall-clock time as a Unix timestamp."""
    return time.time()
