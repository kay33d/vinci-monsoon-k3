# =============================================================================
# SUBMISSION IMAGE — CPU-ONLY, linux/amd64, HYBRID MODE.
#
# Three lanes, cheapest first (see src/router/dispatch.py):
#   1. deterministic rule lane (clear sentiment / pure arithmetic) — 0 tokens
#   2. bundled quantized Qwen GGUF (sentiment/summarization)       — 0 tokens
#   3. Fireworks for everything else (the accuracy-critical categories)
# Classification is ALWAYS the keyword heuristic — the GGUF never classifies
# (the small-GGUF classifier experiment was measurably inaccurate).
#
# Image size is the binding constraint, confirmed by THREE platform TIMEOUTs:
# 2.0 GB timed out (twice), then a 1.2 GB Qwen2.5-1.5B build ALSO timed out
# (vinci-monsoon-k5 run), while a 527 MB image completed cleanly. The pull
# cliff sits between 527 MB and 1.2 GB, so this build deliberately uses the
# smallest Qwen GGUF (0.5B) to land under it — accepting weaker local-answer
# quality in exchange for the model lane existing at all. The answer
# VALIDATION gate in dispatch.py (label/format checks) is the safety net for
# that weakness: a bad 0.5B answer escalates to Fireworks exactly as if the
# lane were absent, same as it did for the 1.5B. Override at build time:
#   docker build --build-arg MODEL_URL=<other-gguf-url> ...
# The grading VM is 4 GB RAM / 2 vCPU with NO GPU — never add CUDA/ROCm.
# =============================================================================

# ---------- Stage 1: build llama-cpp-python wheel + fetch the GGUF ----------
FROM python:3.11-slim AS builder

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

# Portable amd64 build: no -march=native (the grading CPU is unknown).
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF" FORCE_CMAKE=1

COPY requirements.txt .
RUN pip install --no-cache-dir --prefix=/install -r requirements.txt

ARG MODEL_URL="https://huggingface.co/Qwen/Qwen2.5-0.5B-Instruct-GGUF/resolve/main/qwen2.5-0.5b-instruct-q4_k_m.gguf"
RUN mkdir -p /models && curl -L --fail --retry 3 -o /models/model.gguf "$MODEL_URL"

# ---------- Stage 2: slim runtime -------------------------------------------
FROM python:3.11-slim

# libgomp is the only extra runtime lib llama.cpp needs on CPU
RUN apt-get update && apt-get install -y --no-install-recommends libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /install /usr/local
COPY --from=builder /models /models

WORKDIR /app
COPY src ./src
COPY config ./config
COPY entrypoint.py .

ENV LOCAL_MODEL_PATH=/models/model.gguf \
    PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS are injected by the
# grading harness at runtime — never baked into the image.
ENTRYPOINT ["python", "entrypoint.py"]
