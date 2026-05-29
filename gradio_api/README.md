# gradio_api

Containerised Gradio servers exposing the six open-weight model families.
Two of the non-thinking families (Mistral-7B and
Med42-8B) are deployed at two temperatures each, and DeepSeek-14B is
deployed at two temperatures, giving nine services in total. Each model
runs as its own service so that ports, GPU allocation, and host
bindings can be controlled independently.

## Services

| Profile | Script | Port | Notes |
| ------- | ------ | ---- | ----- |
| `mixtral` | `src/mixtral_7b_t075.py` | 9001 | Mistral 7B Instruct v0.3, T=0.75 |
| `mixtral_t1` | `src/mixtral_7b_t100.py` | 9011 | Mistral 7B Instruct v0.3, T=1.0 |
| `m42` | `src/m42_8b_t075.py` | 9002 | Med42 8B, T=0.75 |
| `m42_t1` | `src/m42_8b_t100.py` | 9012 | Med42 8B, T=1.0 |
| `deepseek14` | `src/deepseek_14b_t075.py` | 9003 | DeepSeek-R1-Distill-Qwen-14B, T=0.75 |
| `deepseek14_t05` | `src/deepseek_14b_t050.py` | 9013 | DeepSeek-R1-Distill-Qwen-14B, T=0.5 |
| `deepseek32` | `src/deepseek_32b_t060.py` | 9004 | DeepSeek-R1-Distill-Qwen-32B, multi-GPU sharded |
| `qwen32` | `src/qwen_32b_t060.py` | 9005 | Qwen3 32B, multi-GPU sharded |
| `gemma4_31b` | `src/gemma_4_31b_t060.py` | 9006 | Gemma 4 31B Instruct, T=0.6 |

All services share the same container image and runtime helpers in
`src/server_common.py`. Models are loaded in FP16 (`torch.float16`,
`device_map="auto"`) with no quantisation applied. Only the model
identifier and generation parameters differ between scripts.

## Layout

```text
gradio_api/
├── Dockerfile
├── docker-compose.yml
├── .env.example
├── .dockerignore
├── logging.yaml
├── requirements.txt
├── README.md
├── hf_cache/        # mounted Hugging Face cache (kept out of git)
├── logs/            # rotating log files written by the servers
├── offload/         # disk offload directory for large models
├── src/
│   ├── server_common.py
│   ├── mixtral_7b_t075.py
│   ├── mixtral_7b_t100.py
│   ├── m42_8b_t075.py
│   ├── m42_8b_t100.py
│   ├── deepseek_14b_t075.py
│   ├── deepseek_14b_t050.py
│   ├── deepseek_32b_t060.py
│   ├── qwen_32b_t060.py
│   └── gemma_4_31b_t060.py
└── tests/
    └── test_server_common.py
```

## Requirements

- Docker 24+
- Docker Compose v2
- NVIDIA GPU drivers and the NVIDIA Container Toolkit on the host
- A Hugging Face access token with permission for any gated weights

GPU access inside the container is provided through the `gpus: all`
directive in `docker-compose.yml` together with the
`NVIDIA_VISIBLE_DEVICES` environment variable for each service.

## Configuration

Copy the example environment file and edit it to suit your host:

```bash
cp .env.example .env
```

Variables read by `docker-compose.yml`:

| Variable | Purpose |
| -------- | ------- |
| `HF_ACCESS_TOKEN` | Hugging Face token used at model-load time |
| `GRADIO_DEBUG` | `1` to enable Gradio debug output, `0` otherwise |
| `LOGS_DIR` | Log directory inside the container (default `/app/logs`) |
| `MAX_MEMORY_PER_GPU` | Per-device memory cap for the multi-GPU services |
| `<SERVICE>_HOST_BIND_IP` | Host interface that the published port binds to |
| `<SERVICE>_VISIBLE_GPUS` | Comma-separated list of GPU indices to expose |

`<SERVICE>` is one of `MIXTRAL`, `MIXTRAL_T1`, `M42`, `M42_T1`,
`DEEPSEEK14`, `DEEPSEEK14_T05`, `DEEPSEEK32`, `QWEN32`, or
`GEMMA4_31B`. Bind addresses default to `127.0.0.1`, so the services
are not reachable from outside the host unless this is changed
deliberately.

## Building and running with Docker Compose

Build the image once and start a single service:

```bash
docker compose --profile mixtral up --build -d
```

Subsequent services reuse the cached image:

```bash
docker compose --profile deepseek32 up -d
docker compose --profile qwen32 up -d
```

Stop a running service:

```bash
docker compose --profile mixtral down
```

The same commands work on Windows under Docker Desktop, provided that
WSL2 GPU support and the NVIDIA Container Toolkit have been enabled.

## Running a single container without Compose

```bash
docker build -t gradio-api .

docker run --rm -it \
  --gpus all \
  -e HF_ACCESS_TOKEN=hf_your_token \
  -e GRADIO_SERVER_NAME=0.0.0.0 \
  -e GRADIO_SERVER_PORT=9001 \
  -p 127.0.0.1:9001:9001 \
  gradio-api python src/mixtral_7b_t075.py
```

## Verifying that a server is up

A live server publishes a Gradio interface at the configured port. The
recommended readiness check is a `gradio_client` handshake, which is what
the orchestration scripts in `scripts/` use:

```python
from gradio_client import Client
Client("http://127.0.0.1:9001").close()
```

A successful return indicates that the server has finished loading the
model and is accepting requests.

## Tests

Unit tests for the shared helpers live in `tests/`. They do not load any
model weights and run quickly:

```bash
python -m pytest gradio_api/tests
```

## Notes on disk usage

Model weights are downloaded into `hf_cache/` on first use and shared
across containers via a bind mount. Expect tens to hundreds of gigabytes
depending on which services are started. The `offload/` directory is
used by `accelerate` when sharded weights spill from GPU to CPU/disk.
