# gaia-core: The Brain - Cognitive loop and reasoning engine
# CPU-only container — GPU inference is offloaded to gaia-prime via PRIME_ENDPOINT
# Includes embedded llama-server for Core/Lite CPU inference (Qwen3.5-4B)

# ── Stage 1: Build llama-server from source ─────────────────────────────────
FROM debian:bookworm-slim AS llama-builder

ENV DEBIAN_FRONTEND=noninteractive

RUN apt-get update && apt-get install -y --no-install-recommends \
    build-essential cmake git curl ca-certificates \
    && rm -rf /var/lib/apt/lists/*

WORKDIR /build
ARG LLAMA_CPP_TAG=b8250
RUN git clone --depth 1 --branch ${LLAMA_CPP_TAG} https://github.com/ggml-org/llama.cpp.git \
    && cd llama.cpp \
    && cmake -B build \
        -DCMAKE_BUILD_TYPE=Release \
        -DGGML_NATIVE=ON \
        -DLLAMA_CURL=ON \
        -DGGML_CPU_AARCH64=OFF \
    && cmake --build build --target llama-server -j$(nproc)

# ── Stage 2: Python runtime ─────────────────────────────────────────────────
FROM python:3.11-slim AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
# libcurl4 + libgomp1 required by llama-server
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    cmake \
    libcurl4 \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# Copy llama-server binary and shared libraries from builder
COPY --from=llama-builder /build/llama.cpp/build/bin/ /usr/local/bin/
RUN cp /usr/local/bin/lib*.so* /usr/local/lib/ 2>/dev/null; ldconfig

# Create non-root user for container execution (host UID mapping)
RUN groupadd -g 1000 gaia && useradd -u 1000 -g gaia -m -s /bin/bash gaia
ENV HOME=/home/gaia
ENV USER=gaia

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# ═══════════════════════════════════════════════════════════════════════════
# Install dependencies from requirements.txt (better layer caching)
# ═══════════════════════════════════════════════════════════════════════════

# Copy requirements first for better caching
COPY gaia-core/requirements.txt .

# Copy gaia-common source code so it's available to be installed by requirements.txt
COPY gaia-common /gaia-common

# Install Python dependencies
# GPU inference is handled by gaia-prime — no GPU packages needed here
RUN pip install --no-cache-dir -r requirements.txt

# ═══════════════════════════════════════════════════════════════════════════
# Install development dependencies for validation (pytest, ruff, mypy)
# ═══════════════════════════════════════════════════════════════════════════
COPY gaia-core/pyproject.toml .
COPY gaia-core/README.md .
RUN pip install -e ".[dev]"

# ═══════════════════════════════════════════════════════════════════════════
# Copy service code
# ═══════════════════════════════════════════════════════════════════════════

COPY gaia-core/gaia_core/ ./gaia_core/
COPY gaia-core/gaia_rescue.py ./gaia_rescue.py
# Copy canonical constants file from gaia-common (single source of truth)
COPY gaia-common/gaia_common/constants/gaia_constants.json ./gaia_common/constants/gaia_constants.json

# Create writable directories and set ownership
RUN mkdir -p /app/.pytest_cache /logs \
    && chown -R gaia:gaia /app /logs

# Expose service port
EXPOSE 6415

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:6415/health || exit 1

# Run as non-root user
USER gaia

# Copy entrypoint script (starts llama-server + uvicorn)
COPY gaia-core/entrypoint.sh /entrypoint.sh

# Run the service (entrypoint starts llama-server in background, then uvicorn)
CMD ["/bin/bash", "/entrypoint.sh"]
