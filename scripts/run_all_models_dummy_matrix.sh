#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
CLIENT_DIR="$ROOT_DIR/python_client"
COMPOSE_DIR="$ROOT_DIR/gradio_api"

ensure_env() {
  cd "$ROOT_DIR"
  if [[ ! -d ".venv" ]]; then
    /usr/bin/python -m venv .venv
  fi
  source .venv/bin/activate
  python -m pip install --upgrade pip
  python -m pip install -r python_client/requirements.txt
  python -m pip install -e python_client
}

wait_for_endpoint() {
  local url="$1"
  local timeout_seconds="${RUN_READY_TIMEOUT_SECONDS:-7200}"
  local attempts=$((timeout_seconds / 5))
  for _ in $(seq 1 "$attempts"); do
    if "$PYTHON_BIN" - <<PY >/dev/null 2>&1
from gradio_client import Client
client = None
try:
    client = Client("$url")
finally:
    if client is not None:
        client.close()
PY
    then
      return 0
    fi
    sleep 5
  done
  return 1
}

run_phase() {
  local phase_name="$1"
  local profile_name="$2"
  local config_path="$3"
  local readiness_url="$4"

  echo "==> Phase: $phase_name"
  cd "$COMPOSE_DIR"
  local compose_args=(docker compose --env-file "$ROOT_DIR/.env" --profile "$profile_name")

  "${compose_args[@]}" up --build -d

  echo "Waiting for $readiness_url"
  if ! wait_for_endpoint "$readiness_url"; then
    echo "Service at $readiness_url did not become ready for phase $phase_name" >&2
    exit 1
  fi

  cd "$CLIENT_DIR"
  "$PYTHON_BIN" -m python_client validate-config --config "$config_path"
  "$PYTHON_BIN" -m python_client run --config "$config_path"

  cd "$COMPOSE_DIR"
  "${compose_args[@]}" down
}

if [[ ! -f "$ROOT_DIR/.env" ]]; then
  echo "Missing $ROOT_DIR/.env with HF_ACCESS_TOKEN set." >&2
  exit 1
fi

ensure_env

run_phase "mixtral" "mixtral" "$CLIENT_DIR/configs/all_models_dummy_mixtral.json" "http://127.0.0.1:9001"
run_phase "mixtral_t1" "mixtral_t1" "$CLIENT_DIR/configs/all_models_dummy_mixtral_t1.json" "http://127.0.0.1:9011"
run_phase "m42" "m42" "$CLIENT_DIR/configs/all_models_dummy_m42.json" "http://127.0.0.1:9002"
run_phase "m42_t1" "m42_t1" "$CLIENT_DIR/configs/all_models_dummy_m42_t1.json" "http://127.0.0.1:9012"
run_phase "deepseek14" "deepseek14" "$CLIENT_DIR/configs/all_models_dummy_deepseek14.json" "http://127.0.0.1:9003"
run_phase "deepseek14_t05" "deepseek14_t05" "$CLIENT_DIR/configs/all_models_dummy_deepseek14_t050.json" "http://127.0.0.1:9013"
run_phase "deepseek32" "deepseek32" "$CLIENT_DIR/configs/all_models_dummy_deepseek32.json" "http://127.0.0.1:9004"
run_phase "qwen32" "qwen32" "$CLIENT_DIR/configs/all_models_dummy_qwen32.json" "http://127.0.0.1:9005"
run_phase "gemma4_31b" "gemma4_31b" "$CLIENT_DIR/configs/all_models_dummy_gemma4_31b.json" "http://127.0.0.1:9006"

echo "Completed multi-phase all-model dummy matrix run. Outputs are in python_client/outputs/chronology_runs/all_models_realistic_dummy_v1"
