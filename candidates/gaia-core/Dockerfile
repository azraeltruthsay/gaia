# gaia-core: The Brain - Cognitive loop and reasoning engine
# GPU-enabled container for model inference

FROM nvidia/cuda:12.4.0-devel-ubuntu22.04 AS base

# Prevent interactive prompts during package installation
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
    libxcb1 \
    libx11-6 \
    libgl1-mesa-glx \
    build-essential \
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
COPY gaia-core/requirements.txt .
COPY gaia-core/requirements-gpu.txt .

# Copy gaia-common source code so it's available to be installed by requirements.txt
COPY gaia-common /gaia-common

# Install Python dependencies, including GPU dependencies
# This now also installs gaia-common from the path specified in requirements.txt
RUN pip install --no-cache-dir -r requirements.txt -r requirements-gpu.txt

# ═══════════════════════════════════════════════════════════════════════════
# Copy service code
# ═══════════════════════════════════════════════════════════════════════════

COPY gaia-core/gaia_core/ ./gaia_core/
COPY gaia-core/gaia_rescue.py ./gaia_rescue.py
# Copy constants file for model configuration
COPY gaia-core/gaia_core/gaia_constants.json ./gaia_core/gaia_constants.json

# Expose service port
EXPOSE 6415

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:6415/health || exit 1

# Run the service
CMD ["python", "-m", "uvicorn", "gaia_core.main:app", "--host", "0.0.0.0", "--port", "6415"]
