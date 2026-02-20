# gaia-web: The Face - UI and API gateway
# Lightweight container for web serving

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    libopus0 \
    libopus-dev \
    ffmpeg \
    libffi-dev \
    build-essential \
    python3-dev \
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
COPY candidates/gaia-web/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy and install gaia-common and gaia-web in editable mode
COPY candidates/gaia-common/ /app/gaia-common/
COPY candidates/gaia-web/ /app/gaia-web/
COPY candidates/gaia-web/pyproject.toml /app/gaia-web/pyproject.toml
RUN pip install -e /app/gaia-common/
RUN pip install -e /app/gaia-web/.[dev]

# Set ownership for non-root execution
RUN chown -R gaia:gaia /app

# Expose service port
EXPOSE 6414

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:6414/health || exit 1

ENV PYTHONPATH "${PYTHONPATH}:/app"

# Run as non-root user
USER gaia

# Run the service
CMD ["python", "-m", "uvicorn", "gaia_web.main:app", "--host", "0.0.0.0", "--port", "6414"]
