# Build llama-cpp-python against a CPU backend. The compiler/toolchain stays
# in this stage and is not copied into the production image.
FROM python:3.10-slim AS builder

WORKDIR /build
ENV PIP_NO_CACHE_DIR=1
ENV CMAKE_ARGS="-DGGML_NATIVE=OFF -DGGML_OPENMP=ON -DGGML_BLAS=ON -DGGML_BLAS_VENDOR=OpenBLAS"

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    pkg-config \
    libopenblas-dev \
    && rm -rf /var/lib/apt/lists/*

COPY requirements.txt .
RUN pip install --no-cache-dir --upgrade pip && \
    pip install --no-cache-dir -r requirements.txt

# Download the exact Q4_K_M text model used by this test branch.
RUN mkdir -p /app/models && \
    python -c "from huggingface_hub import hf_hub_download; hf_hub_download(repo_id='SoAIHQ/Qwen3.5-4B-GGUF', filename='Qwen3.5-4B-Q4_K_M.gguf', local_dir='/app/models')" && \
    echo '2ebe6eb67cb92adeffc7073c48dd4e3baf79465ac650793dc751cad2b29cf219  /app/models/Qwen3.5-4B-Q4_K_M.gguf' | sha256sum -c - && \
    rm -rf /app/models/.cache

FROM python:3.10-slim AS runtime
WORKDIR /app

ENV PYTHONUNBUFFERED=1 \
    PYTHONDONTWRITEBYTECODE=1 \
    LLAMA_MODEL_PATH=/app/models/Qwen3.5-4B-Q4_K_M.gguf \
    LLM_N_CTX=4096 \
    LLM_N_THREADS=4 \
    LLM_N_BATCH=128

RUN apt-get update && apt-get install -y --no-install-recommends \
    libopenblas0-pthread \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local /usr/local
COPY --from=builder /app/models /app/models
COPY app.py .

EXPOSE 8000
CMD ["uvicorn", "app:app", "--host", "0.0.0.0", "--port", "8000", "--workers", "1"]
