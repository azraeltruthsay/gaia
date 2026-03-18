# gaia-nano: Dual-mode Reflex tier — GGUF (llama-server) + Safetensors (inference_server.py)
# GGUF mode: fast inference via llama.cpp with CUDA
# Safetensors mode: full activation monitoring (polygraph), SAE analysis, device migration
ARG LLAMA_CPP_TAG=b8250
ARG CUDA_VERSION=12.8.1

# ── Build llama-server (CUDA) ──────────────────────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS builder

ARG LLAMA_CPP_TAG
ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl ca-certificates libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

RUN ln -s /usr/local/cuda/lib64/stubs/libcuda.so /usr/lib/x86_64-linux-gnu/libcuda.so.1 \
    && ln -s /usr/lib/x86_64-linux-gnu/libcuda.so.1 /usr/lib/x86_64-linux-gnu/libcuda.so \
    && ldconfig

RUN git clone --depth 1 --branch ${LLAMA_CPP_TAG} https://github.com/ggml-org/llama.cpp.git \
    && cd llama.cpp \
    && cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_CUDA=ON \
        -DCMAKE_CUDA_ARCHITECTURES="86;89;120" \
        -DLLAMA_CURL=ON \
        -DGGML_CPU_AARCH64=OFF \
    && cmake --build build --target llama-server -j$(nproc)

# ── Runtime stage ──────────────────────────────────────────────────────────
FROM python:3.11-slim

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl ca-certificates libcurl4 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# CUDA runtime libs for llama-server GPU mode
COPY --from=builder /usr/local/cuda/lib64/libcudart.so* /usr/local/lib/
COPY --from=builder /usr/local/cuda/lib64/libcublas.so* /usr/local/lib/
COPY --from=builder /usr/local/cuda/lib64/libcublasLt.so* /usr/local/lib/

# llama-server binary and shared libs
COPY --from=builder /build/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=builder /build/llama.cpp/build/bin/lib*.so* /usr/local/lib/
RUN ldconfig

# PyTorch + transformers for safetensors mode
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128
RUN pip install --no-cache-dir transformers accelerate
# Note: flash-linear-attention omitted — Qwen3.5 falls back to torch implementation
# which works correctly. FLA's custom_device_ctx has a torch.cpu.device bug.

WORKDIR /app

# Copy the inference server (shared with gaia-core — same code, different model)
COPY gaia-core/gaia_core/inference_server.py /app/inference_server.py

# Copy entrypoint
COPY gaia-llama/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080

ENTRYPOINT ["/entrypoint.sh"]
