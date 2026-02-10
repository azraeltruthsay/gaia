# gaia-core: The Brain - Cognitive loop and reasoning engine
# CPU-only container — GPU inference is offloaded to gaia-prime via PRIME_ENDPOINT

FROM python:3.11-slim AS base

# Prevent interactive prompts during package installation
ENV DEBIAN_FRONTEND=noninteractive
ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    git \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

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
# Copy constants file for model configuration
COPY gaia-core/gaia_core/gaia_constants.json ./gaia_core/gaia_constants.json

# Expose service port
EXPOSE 6415

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:6415/health || exit 1

# Run the service
CMD ["python", "-m", "uvicorn", "gaia_core.main:app", "--host", "0.0.0.0", "--port", "6415"]
