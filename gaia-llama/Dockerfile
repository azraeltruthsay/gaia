# gaia-nano: llama.cpp server with CUDA support
# Builds llama-server from source with GPU acceleration for Nano triage model
ARG LLAMA_CPP_TAG=b8250
ARG CUDA_VERSION=12.8.1

# ── Build stage ──────────────────────────────────────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-devel-ubuntu22.04 AS builder

ARG LLAMA_CPP_TAG

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    libcurl4-openssl-dev \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Make the stub libcuda.so visible to the standard linker path
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

# ── Runtime stage ────────────────────────────────────────────────────────────
FROM nvidia/cuda:${CUDA_VERSION}-runtime-ubuntu22.04

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    ca-certificates \
    libcurl4 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

COPY --from=builder /build/llama.cpp/build/bin/llama-server /usr/local/bin/llama-server
COPY --from=builder /build/llama.cpp/build/bin/lib*.so* /usr/local/lib/
RUN ldconfig

ENTRYPOINT ["llama-server"]
