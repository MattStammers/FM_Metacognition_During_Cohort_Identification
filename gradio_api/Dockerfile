FROM python:3.12-slim

ENV PYTHONDONTWRITEBYTECODE=1 \
    PYTHONUNBUFFERED=1 \
    PIP_NO_CACHE_DIR=1 \
    HF_HOME=/app/.cache/huggingface

WORKDIR /app

RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt ./requirements.txt
RUN python -m pip install --upgrade pip && \
    python -m pip install -r requirements.txt

COPY src ./src
COPY logging.yaml ./logging.yaml
COPY README.md ./README.md
COPY offload ./offload

RUN mkdir -p /app/logs /app/.cache/huggingface

EXPOSE 9001 9002 9003 9004 9005 9006 9011 9012 9013

CMD ["python", "src/mixtral_7b_t075.py"]
