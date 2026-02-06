FROM python:3.11-slim

WORKDIR /app

# Install system dependencies for Docker SDK and GPU monitoring
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Copy requirements and install Python dependencies
COPY requirements.txt .
RUN pip install --no-cache-dir -r requirements.txt

# Copy application code
COPY gaia_orchestrator/ ./gaia_orchestrator/
COPY pyproject.toml .

# Install the package
RUN pip install --no-cache-dir -e .

# Create directories for state persistence
RUN mkdir -p /shared/orchestrator

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=5s --retries=3 \
    CMD curl -f http://localhost:6410/health || exit 1

# Expose port
EXPOSE 6410

# Run the orchestrator
CMD ["uvicorn", "gaia_orchestrator.main:app", "--host", "0.0.0.0", "--port", "6410"]
