#!/usr/bin/env bash
# Entrypoint for the analytics Docker image. Runs the bundled dummy
# config through the same default passes as the native wrapper:
# patient-level, document-level, complete-marker sensitivity, and
# validation-view endpoint runs. Any positional arguments are
# forwarded to the analytics CLI and replace that default sequence.
set -euo pipefail

CONFIG="${ANALYTICS_CONFIG:-/workspace/analytics_code/configs/all_models_dummy_analysis.json}"

if [[ ! -f "$CONFIG" ]]; then
  echo "[analytics] Config not found inside container at $CONFIG" >&2
  echo "[analytics] Make sure the repo is mounted, e.g.:" >&2
  echo "  docker run --rm -v \"\$PWD:/workspace\" llm-metacognition-analytics" >&2
  exit 1
fi

# Always re-install the package against the *mounted* workspace so any
# in-place edits to analytics_code/src take effect.
pip install --quiet -e /workspace/analytics_code

if [[ $# -gt 0 ]]; then
  exec python -m analytics_code "$@"
fi

echo "[analytics] validate-config..."
python -m analytics_code validate-config --config "$CONFIG"

echo "[analytics] run-all (patient-level)..."
python -m analytics_code run-all --config "$CONFIG"

echo "[analytics] run-document-level-all (document-level sibling)..."
python -m analytics_code run-document-level-all --config "$CONFIG"

echo "[analytics] run-document-level-complete-all (complete-marker sensitivity)..."
python -m analytics_code run-document-level-complete-all --config "$CONFIG"

echo "[analytics] run-validation-views-all (Document/Cumulative/Final/Doc2Patient)..."
python -m analytics_code run-validation-views-all --config "$CONFIG"

echo "[analytics] Done. Outputs under <output_root>, <output_root>_document_level, <output_root>_document_level_complete, and <output_root>/{Document,Cumulative,Final,Doc2Patient}."
