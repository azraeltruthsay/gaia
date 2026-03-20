# gaia-nano: Dual-mode Reflex tier — GGUF (llama-server) + Safetensors (GAIAEngine)
# GGUF mode: fast inference via llama.cpp with CUDA
# Safetensors mode: full activation monitoring (polygraph), SAE analysis, adapter loading
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

# ── Runtime stage (from unified engine base) ──────────────────────────────
FROM localhost:5000/gaia-engine-base:latest

# GGUF support: llama-server binary + CUDA runtime libs
RUN apt-get update && apt-get install -y --no-install-recommends \
    libcurl4 libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /usr/local/cuda/lib64/libcudart.so* /usr/local/lib/
COPY --from=builder /usr/local/cuda/lib64/libcublas.so* /usr/local/lib/
COPY --from=builder /usr/local/cuda/lib64/libcublasLt.so* /usr/local/lib/
COPY --from=builder /build/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=builder /build/llama.cpp/build/bin/lib*.so* /usr/local/lib/
RUN ldconfig

# Copy entrypoint
COPY gaia-llama/entrypoint.sh /entrypoint.sh
RUN chmod +x /entrypoint.sh

HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8080/health || exit 1

EXPOSE 8080

ENTRYPOINT ["/entrypoint.sh"]
