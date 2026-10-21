# selfjev server on one NVIDIA GPU (>= 16 GB): Jev's decisions API on selfjev-4b.
#   docker build -f deploy/Dockerfile -t selfjev .
#   docker run --gpus all -p 8000:8000 -e SELFJEV_API_KEYS=my-key selfjev
FROM nvidia/cuda:12.8.1-cudnn-runtime-ubuntu24.04

ENV DEBIAN_FRONTEND=noninteractive PYTHONUNBUFFERED=1 UV_COMPILE_BYTECODE=1 UV_LINK_MODE=copy HF_HOME=/models
RUN apt-get update && apt-get install -y --no-install-recommends ca-certificates curl && rm -rf /var/lib/apt/lists/*
COPY --from=ghcr.io/astral-sh/uv:0.11 /uv /uvx /usr/local/bin/

WORKDIR /app
COPY pyproject.toml uv.lock README.md ./
RUN uv sync --frozen --no-dev --no-install-project --extra serve --extra gpu
COPY src ./src
RUN uv sync --frozen --no-dev --extra serve --extra gpu

# the adapter and the pinned base model (about 9 GB), baked in so a container serves without downloading; --build-arg BAKE_MODEL=0 to skip
ARG BAKE_MODEL=1
RUN if [ "$BAKE_MODEL" = 1 ]; then uv run --no-sync python -c "from huggingface_hub import snapshot_download; \
    snapshot_download('Qwen/Qwen3.5-4B', revision='851bf6e806efd8d0a36b00ddf55e13ccb7b8cd0a'); \
    snapshot_download('Jwuthrich/selfjev-4b-vision', allow_patterns=['adapter_model.safetensors', 'adapter_config.json', 'model.json'])"; fi

EXPOSE 8000
HEALTHCHECK --interval=30s --timeout=5s --start-period=300s CMD curl -fs localhost:8000/health || exit 1
CMD ["uv", "run", "--no-sync", "selfjev", "serve", "--host", "0.0.0.0", "--port", "8000"]
