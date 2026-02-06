# gaia-web: The Face - UI and API gateway
# Lightweight container for web serving

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# ═══════════════════════════════════════════════════════════════════════════
# Install dependencies from requirements.txt (better layer caching)
# ═══════════════════════════════════════════════════════════════════════════

# Copy requirements first for better caching
COPY gaia-web/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy and install gaia-common and gaia-web in editable mode
COPY gaia-common/ /app/gaia-common/
COPY gaia-web/ /app/gaia-web/
RUN pip install -e /app/gaia-common/
RUN pip install -e /app/gaia-web/

# Expose service port
EXPOSE 6414

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:6414/health || exit 1

ENV PYTHONPATH "${PYTHONPATH}:/app:/gaia-common"

# Run the service
CMD ["python", "-m", "uvicorn", "gaia_web.main:app", "--host", "0.0.0.0", "--port", "6414"]
