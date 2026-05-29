"""Helper for ``run_analytics_pipeline.bat``.

Rewrites the analysis config so all relative ``paths`` entries are
rebased onto a short-drive substitute of the repo root. Run as::

    python _materialize_short_config.py <user_config> <root_dir> <short_root> <out_path>
"""

from __future__ import annotations

import json
import pathlib
import sys


def _rebase(
    value: str,
    repo_root: pathlib.Path,
    short_root: pathlib.Path,
    config_dir: pathlib.Path,
) -> str:
    path = pathlib.Path(value)
    abs_path = path if path.is_absolute() else (config_dir / path).resolve()
    try:
        rel = abs_path.relative_to(repo_root)
    except ValueError:
        # Path lives outside the repo; leave it untouched.
        return str(abs_path)
    return str(short_root / rel)


def main(argv: list[str]) -> int:
    if len(argv) != 5:
        print(
            "Usage: _materialize_short_config.py <user_config> <root_dir> "
            "<short_root> <out_path>",
            file=sys.stderr,
        )
        return 2
    _, user_config, root_dir, short_root, out_path = argv
    config_path = pathlib.Path(user_config)
    repo_root = pathlib.Path(root_dir).resolve()
    # Normalise bare drive letters like "X:" to absolute roots ("X:\\"),
    # otherwise ``Path("X:") / "sub"`` yields the drive-relative path
    # ``X:sub`` which downstream resolution will mis-interpret.
    if len(short_root) == 2 and short_root.endswith(":"):
        short_root = short_root + "\\"
    short_root_path = pathlib.Path(short_root)
    out_file = pathlib.Path(out_path)

    raw = json.loads(config_path.read_text(encoding="utf-8"))
    paths = raw.setdefault("paths", {})
    config_dir = config_path.parent.resolve()
    for key in ("llm_batch_root", "human_reference", "output_root"):
        if key in paths:
            paths[key] = _rebase(paths[key], repo_root, short_root_path, config_dir)
    out_file.write_text(json.dumps(raw, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main(sys.argv))
