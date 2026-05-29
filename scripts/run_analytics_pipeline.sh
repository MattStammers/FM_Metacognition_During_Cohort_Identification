#!/usr/bin/env bash
# ---------------------------------------------------------------------------
# Run the analytics_code pipeline against the bundled all-models dummy
# analysis config. The analytics suite is
# evaluated at three truth-label resolutions:
#
#   * patient-level (default)             -- ``run-all``
#   * document-level (sibling)             -- ``run-document-level-all``
#       writes to ``<output_root>_document_level``
#   * document-level complete-marker arm   -- ``run-document-level-complete-all``
#       writes to ``<output_root>_document_level_complete``
#       (sensitivity variant: restricts to rows whose relevant
#       document markers are all present)
#
# All three passes share the same source data and are executed
# sequentially here so the published outputs always include the
# primary plus both sensitivity views.
# ---------------------------------------------------------------------------
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

CONFIG_PATH="${1:-$ROOT_DIR/analytics_code/configs/all_models_dummy_analysis.json}"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"

if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "Python venv not found at $PYTHON_BIN. Create it with: python -m venv $ROOT_DIR/.venv" >&2
  exit 1
fi

if [[ ! -f "$CONFIG_PATH" ]]; then
  echo "Analytics config not found: $CONFIG_PATH" >&2
  exit 1
fi

cd "$ROOT_DIR"
# shellcheck source=/dev/null
source .venv/bin/activate
python -m pip install --upgrade pip >/dev/null
python -m pip install -r analytics_code/requirements.txt
python -m pip install -e analytics_code

cd "$ROOT_DIR/analytics_code"
"$PYTHON_BIN" -m analytics_code validate-config --config "$CONFIG_PATH"
"$PYTHON_BIN" -m analytics_code run-all --config "$CONFIG_PATH"
"$PYTHON_BIN" -m analytics_code run-document-level-all --config "$CONFIG_PATH"
"$PYTHON_BIN" -m analytics_code run-document-level-complete-all --config "$CONFIG_PATH"

echo "Analytics pipeline completed using config: $CONFIG_PATH"
echo "Patient-level outputs:                    <output_root>"
echo "Document-level outputs:                   <output_root>_document_level"
echo "Document-level (complete-marker) outputs: <output_root>_document_level_complete"
