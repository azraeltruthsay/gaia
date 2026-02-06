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
COPY candidates/gaia-study/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt



# Use candidate versions for SOA-decoupled architecture
COPY candidates/gaia-common/ /gaia-common/
COPY candidates/gaia-study/ /app/gaia-study/

ENV PYTHONPATH "${PYTHONPATH}:/gaia-common:/app"

RUN pip install -e "/gaia-common[vector]"
RUN pip install -e /app/gaia-study/

RUN pip install -e "/gaia-common[vector]"



# Expose service port
EXPOSE 8766

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8766/health || exit 1

# Run the service
CMD ["python", "-m", "uvicorn", "gaia_study.main:app", "--host", "0.0.0.0", "--port", "8766"]
