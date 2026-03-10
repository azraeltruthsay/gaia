# gaia-llama: CPU inference server for Nano/Core models
# Builds llama.cpp from source with qwen35 architecture support
# Provides OpenAI-compatible API via llama-server

FROM debian:bookworm-slim AS builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential \
    cmake \
    git \
    curl \
    ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build

# Clone llama.cpp at a known-good tag with qwen35 support
# b8250 has full qwen35 support (dense + MoE, shape fixes, model detection)
ARG LLAMA_CPP_TAG=b8250
RUN git clone --depth 1 --branch ${LLAMA_CPP_TAG} https://github.com/ggml-org/llama.cpp.git \
    && cd llama.cpp \
    && cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=ON \
        -DLLAMA_CURL=ON \
        -DGGML_CPU_AARCH64=OFF \
    && cmake --build build --target llama-server -j$(nproc)

# Runtime stage
FROM debian:bookworm-slim

RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libcurl4 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user
RUN groupadd -g 1000 gaia && useradd -u 1000 -g gaia -m -s /bin/bash gaia

# Copy the built binary and all shared libraries (all in bin/ dir)
COPY --from=builder /build/llama.cpp/build/bin/ /usr/local/bin/
RUN cp /usr/local/bin/lib*.so* /usr/local/lib/ && ldconfig

# Health check
HEALTHCHECK --interval=15s --timeout=5s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:${PORT:-8080}/health || exit 1

USER gaia

# Default: serve a model on port 8080 with OpenAI-compatible API
# Override via docker-compose command
ENTRYPOINT ["llama-server"]
CMD ["--host", "0.0.0.0", "--port", "8080"]
