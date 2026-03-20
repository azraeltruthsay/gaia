# gaia-core: The Brain — Cognitive loop + embedded Core inference engine
# Inherits unified GAIA engine base with all ML capabilities.
# Adds: FastAPI application, llama-server (GGUF fallback), service code.

ARG LLAMA_CPP_TAG=b8250
ARG CUDA_VERSION=12.8.1

# ── Build llama-server (CUDA) — for GGUF CPU fallback ───────────────────
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS llama-builder

ARG LLAMA_CPP_TAG
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl ca-certificates libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
    && ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so \
    && ldconfig

WORKDIR /build
RUN git clone --depth 1 --branch ${LLAMA_CPP_TAG} https://github.com/ggml-org/llama.cpp.git \
    && cd llama.cpp \
    && cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="86;89;120" \
        -DLLAMA_CURL=ON \
        -DGGML_CPU_AARCH64=OFF \
    && cmake --build build --target llama-server llama-quantize -j$(nproc)

# ── Main image (from unified engine base) ────────────────────────────────
FROM localhost:5000/gaia-engine-base:latest

# System deps for llama-server
RUN apt-get update && apt-get install -y --no-install-recommends \
    git curl build-essential cmake \
    libcurl4 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CUDA runtime libs + llama binaries (for GGUF CPU/GPU fallback)
COPY --from=llama-builder /usr/local/cuda/lib64/libcudart.so* /usr/local/lib/
COPY --from=llama-builder /usr/local/cuda/lib64/libcublas.so* /usr/local/lib/
COPY --from=llama-builder /usr/local/cuda/lib64/libcublasLt.so* /usr/local/lib/
COPY --from=llama-builder /build/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=llama-builder /build/llama.cpp/build/bin/llama-quantize /usr/local/bin/llama-quantize
COPY --from=llama-builder /build/llama.cpp/build/bin/lib*.so* /usr/local/lib/
RUN ldconfig

# flash-linear-attention for Qwen3.5 hybrid attention (DeltaNet fast path)
RUN pip install --no-cache-dir flash-linear-attention

# Create non-root user
RUN groupadd -g 1000 gaia && useradd -u 1000 -g gaia -m -s /bin/bash gaia
ENV HOME=/home/gaia
ENV USER=gaia

WORKDIR /app

# Application deps (FastAPI, Discord, httpx, etc.)
COPY candidates/gaia-core/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Dev dependencies
COPY candidates/gaia-core/pyproject.toml .
COPY candidates/gaia-core/README.md .
RUN pip install -e ".[dev]"

# Service code
COPY candidates/gaia-core/gaia_core/ ./gaia_core/
COPY candidates/gaia-core/gaia_rescue.py ./gaia_rescue.py
COPY candidates/gaia-core/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh
COPY candidates/gaia-common/gaia_common/constants/gaia_constants.json ./gaia_common/constants/gaia_constants.json

RUN mkdir -p /app/.pytest_cache /logs \
    && chown -R gaia:gaia /app /logs

EXPOSE 6415

HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:6415/health || exit 1

USER gaia

ENTRYPOINT ["/entrypoint.sh"]
