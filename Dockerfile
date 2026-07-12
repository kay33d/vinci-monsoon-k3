# =============================================================================
# SUBMISSION IMAGE — CPU-ONLY, linux/amd64, ALL-REMOTE MODE.
#
# No local GGUF and no llama-cpp-python: classification uses the deterministic
# keyword heuristic in src/local_models/loader.py and EVERY task is answered
# by an allowed Fireworks model (force_all_remote in the router). Rationale:
#   * three consecutive platform TIMEOUTs while our process finished in ~150s
#     pointed at image PULL time — this image is ~150 MB vs 2.0 GB before;
#   * the 0.5B GGUF classifier was unusably inaccurate, and tokens only
#     matter after the 80% accuracy gate is passed.
# The grading VM is 4 GB RAM / 2 vCPU with NO GPU — keep this image lean;
# never add CUDA/ROCm layers.
#
# Build:  docker build --platform linux/amd64 -t hybrid-router:local .
# =============================================================================

FROM python:3.11-slim

COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

WORKDIR /app
COPY src ./src
COPY config ./config
COPY entrypoint.py .

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1

# FIREWORKS_API_KEY, FIREWORKS_BASE_URL, ALLOWED_MODELS are injected by the
# grading harness at runtime — never baked into the image.
ENTRYPOINT ["python", "entrypoint.py"]
