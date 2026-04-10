# gaia-study: The Subconscious - Background processing and learning
# GPU-enabled container for embedding and training operations
# SOLE WRITER to vector store and LoRA adapters

FROM nvidia/cuda:12.8.1-devel-ubuntu22.04 AS base

ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install Python 3.11 and system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    python3.11 \
    python3.11-dev \
    python3.11-venv \
    python3-pip \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Set Python 3.11 as default
RUN update-alternatives --install /usr/bin/python3 python3 /usr/bin/python3.11 1 \
    && update-alternatives --install /usr/bin/python python /usr/bin/python3.11 1

# Create non-root user for container execution (host UID mapping)
RUN groupadd -g 1000 gaia && useradd -u 1000 -g gaia -m -s /bin/bash gaia
ENV HOME=/home/gaia
ENV USER=gaia

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# ═══════════════════════════════════════════════════════════════════════════
# Install dependencies from requirements.txt (better layer caching)
# ═══════════════════════════════════════════════════════════════════════════

# Copy requirements first for better caching
COPY gaia-study/requirements.txt .

# Install CUDA-enabled PyTorch FIRST to prevent CPU-only torch from
# being pulled in by transitive deps (sentence-transformers, etc.)
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cu128

# Install Python dependencies (torch already satisfied from CUDA index above)
RUN pip install --no-cache-dir -r requirements.txt

# gptqmodel needs torch at build time for CUDA kernel compilation.
# Must be installed after torch, with --no-build-isolation so it sees torch.
# CUDA_ARCH_LIST set manually since nvidia-smi isn't available during build.
# 12.0 = Blackwell (RTX 5080)
# gptqmodel 5.7.0's pyproject.toml has a dual-format license field that
# setuptools >=78 rejects. Downgrade setuptools before build, restore after.
RUN pip install --no-cache-dir "setuptools==77.0.3" \
    && CUDA_ARCH_LIST="12.0" TORCH_CUDA_ARCH_LIST="12.0" \
       pip install --no-cache-dir --no-build-isolation gptqmodel==5.7.0 \
    && pip install --no-cache-dir --upgrade setuptools

# optimum is required by PEFT when gptqmodel is present
RUN pip install --no-cache-dir optimum>=1.24.0

# quanto: CPU-first int4 quantization for large models that don't fit GPU in bf16
RUN pip install --no-cache-dir quanto>=0.2.0

# AutoAWQ needs torch at build time for CUDA kernel compilation.
RUN CUDA_ARCH_LIST="12.0" TORCH_CUDA_ARCH_LIST="12.0" \
    pip install --no-cache-dir autoawq>=0.2.0

# ═══════════════════════════════════════════════════════════════════════════
# llama.cpp quantization toolchain (bf16 → GGUF)
# Builds llama-quantize binary and bundles convert_hf_to_gguf.py
# ═══════════════════════════════════════════════════════════════════════════
RUN apt-get update && apt-get install -y --no-install-recommends \
    cmake build-essential \
    && rm -rf /var/lib/apt/lists/*

RUN git clone --depth 1 https://github.com/ggerganov/llama.cpp /opt/llama.cpp \
    && cd /opt/llama.cpp \
    && cmake -B build \
    && cmake --build build --target llama-quantize -j$(nproc) \
    && cp build/bin/llama-quantize /usr/local/bin/ \
    && cp build/bin/lib*.so* /usr/local/lib/ \
    && ldconfig \
    && pip install --no-cache-dir sentencepiece \
    && rm -rf /opt/llama.cpp/build  # keep source/scripts, remove build artifacts

# llama.cpp convert script needs its local gguf-py package
ENV PYTHONPATH="${PYTHONPATH}:/opt/llama.cpp"

# Use canonical versions for SOA-decoupled architecture
COPY gaia-common/ /gaia-common/
COPY gaia-study/ /app/gaia-study/
COPY gaia-study/pyproject.toml /app/gaia-study/pyproject.toml

ENV PYTHONPATH "${PYTHONPATH}:/gaia-common:/app"

RUN pip install -e "/gaia-common[vector]"
# --no-deps on editable install to prevent pip from pulling CPU torch
# (all deps already satisfied from requirements.txt + CUDA torch above)
RUN pip install --no-deps -e /app/gaia-study/.[dev]

# Set ownership for non-root execution
RUN chown -R gaia:gaia /app




# Expose service port
EXPOSE 8766

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8766/health || exit 1

# Run as non-root user
USER gaia

# Run the service
CMD ["python", "-m", "uvicorn", "gaia_study.main:app", "--host", "0.0.0.0", "--port", "8766"]
