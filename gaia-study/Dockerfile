# gaia-study: The Subconscious - Background processing and learning
# GPU-enabled container for embedding and training operations
# SOLE WRITER to vector store and LoRA adapters

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS base

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

# Upgrade pip
RUN python3 -m pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# ═══════════════════════════════════════════════════════════════════════════
# Install dependencies from requirements.txt (better layer caching)
# ═══════════════════════════════════════════════════════════════════════════

# Copy requirements first for better caching
COPY gaia-study/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Install gaia-common with vector dependencies from mounted volume
RUN --mount=type=bind,source=./gaia-common,target=/gaia-common \
    pip install --no-cache-dir "/gaia-common[vector]"

# Optional: Install training dependencies (uncomment for LoRA fine-tuning)
# COPY requirements-training.txt .
# RUN pip install --no-cache-dir -r requirements-training.txt

# ═══════════════════════════════════════════════════════════════════════════
# Copy service code
# ═══════════════════════════════════════════════════════════════════════════

COPY gaia-study/gaia_study/ ./gaia_study/

# Expose service port
EXPOSE 8766

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8766/health || exit 1

# Run the service
CMD ["python", "-m", "uvicorn", "gaia_study.main:app", "--host", "0.0.0.0", "--port", "8766"]
