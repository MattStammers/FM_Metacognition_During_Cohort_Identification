#!/usr/bin/env bash
# Foolproof analytics quickstart. Tries Docker first (fully isolated,
# zero local Python setup required). Falls back to the native venv
# runner if Docker is unavailable.
#
# Usage:
#   scripts/quickstart_analytics.sh
#   scripts/quickstart_analytics.sh path/to/analysis_config.json
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
USER_CONFIG="${1:-}"

if command -v docker >/dev/null 2>&1 && docker info >/dev/null 2>&1; then
  echo "[quickstart] Docker detected; running analytics in container."
  if [[ -z "$USER_CONFIG" ]]; then
    exec docker compose -f "$SCRIPT_DIR/docker-compose.analytics.yml" run --rm analytics
  fi
  # Translate the host path into the in-container /workspace mount.
  abs_config="$(cd "$(dirname "$USER_CONFIG")" && pwd)/$(basename "$USER_CONFIG")"
  rel_config="${abs_config#$ROOT_DIR/}"
  exec docker compose -f "$SCRIPT_DIR/docker-compose.analytics.yml" run --rm \
    -e ANALYTICS_CONFIG="/workspace/$rel_config" analytics
fi

echo "[quickstart] Docker not available; falling back to native venv runner."
if [[ -z "$USER_CONFIG" ]]; then
  exec "$SCRIPT_DIR/run_analytics_pipeline.sh"
fi
exec "$SCRIPT_DIR/run_analytics_pipeline.sh" "$USER_CONFIG"
