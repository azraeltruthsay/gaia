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
COPY candidates/gaia-core/requirements.txt .

# Copy gaia-common source code so it's available to be installed by requirements.txt
COPY candidates/gaia-common /gaia-common

# Install Python dependencies
# GPU inference is handled by gaia-prime — no GPU packages needed here
RUN pip install --no-cache-dir -r requirements.txt

# ═══════════════════════════════════════════════════════════════════════════
# Install development dependencies for validation (pytest, ruff, mypy)
# ═══════════════════════════════════════════════════════════════════════════
COPY candidates/gaia-core/pyproject.toml .
COPY candidates/gaia-core/README.md .
RUN pip install -e ".[dev]"

# ═══════════════════════════════════════════════════════════════════════════
# Copy service code
# ═══════════════════════════════════════════════════════════════════════════

COPY candidates/gaia-core/gaia_core/ ./gaia_core/
COPY candidates/gaia-core/gaia_rescue.py ./gaia_rescue.py
# Copy canonical constants file from gaia-common (single source of truth)
COPY candidates/gaia-common/gaia_common/constants/gaia_constants.json ./gaia_common/constants/gaia_constants.json

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

# Run the service
CMD ["python", "-m", "uvicorn", "gaia_core.main:app", "--host", "0.0.0.0", "--port", "6415"]
