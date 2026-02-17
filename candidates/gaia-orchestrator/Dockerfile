FROM python:3.11-slim

# Create non-root user for container execution (host UID mapping)
RUN groupadd -g 1000 gaia && useradd -u 1000 -g gaia -m -s /bin/bash gaia
ENV HOME=/home/gaia
ENV USER=gaia

WORKDIR /app

# Install system dependencies for Docker SDK and health checks
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY candidates/gaia-orchestrator/requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Install shared library (candidate version)
COPY candidates/gaia-common /gaia-common
RUN pip install --no-cache-dir -e /gaia-common/

# Copy application code
COPY candidates/gaia-orchestrator/gaia_orchestrator/ ./gaia_orchestrator/
COPY candidates/gaia-orchestrator/config/ ./config/
COPY candidates/gaia-orchestrator/pyproject.toml .

# Install the package
RUN pip install --no-cache-dir -e .

# Create directories for state persistence and set ownership
RUN mkdir -p /shared/orchestrator \
    && chown -R gaia:gaia /app /shared/orchestrator

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:6410/health || exit 1

# Expose port
EXPOSE 6410

# Run as non-root user
USER gaia

# Run the orchestrator
CMD ["uvicorn", "gaia_orchestrator.main:app", "--host", "0.0.0.0", "--port", "6410"]
