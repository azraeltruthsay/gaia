# gaia-mcp: The Hands - Sandboxed tool execution environment
# Security-hardened container for safe tool execution

FROM python:3.11-slim AS base

ENV PYTHONUNBUFFERED=1
ENV PYTHONDONTWRITEBYTECODE=1

# Install minimal system dependencies
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    build-essential \
    && rm -rf /var/lib/apt/lists/*

# Create non-root user for sandboxed execution
RUN useradd --create-home --shell /bin/bash gaia \
    && mkdir -p /sandbox /var/log/gaia /knowledge/system_reference/blueprints /knowledge/system_reference/cheatsheets /knowledge/system_reference/thought_seeds /knowledge/seeds \
    && chown -R gaia:gaia /sandbox /var/log/gaia /knowledge

# Upgrade pip
RUN pip install --no-cache-dir --upgrade pip setuptools wheel

WORKDIR /app

# ═══════════════════════════════════════════════════════════════════════════
# Install dependencies from requirements.txt (better layer caching)
# ═══════════════════════════════════════════════════════════════════════════

# Copy requirements first for better caching
COPY gaia-mcp/requirements.txt .

# Install Python dependencies
RUN pip install --no-cache-dir -r requirements.txt

# Copy and install gaia-common and gaia-mcp in editable mode
# Install gaia-common and gaia-mcp in editable mode
COPY gaia-common/ /gaia-common/
COPY gaia-mcp/ /app/gaia-mcp/
COPY gaia-mcp/pyproject.toml /app/gaia-mcp/pyproject.toml
ENV PYTHONPATH "${PYTHONPATH}:/gaia-common"

RUN pip install -e /gaia-common/
RUN pip install -e /app/gaia-mcp/.[dev]


# Set ownership
RUN chown -R gaia:gaia /app

# Expose service port
EXPOSE 8765

# Health check
HEALTHCHECK --interval=30s --timeout=10s --start-period=30s --retries=3 \
    CMD curl -f http://localhost:8765/health || exit 1

# Run as non-root user (can be overridden for specific tools)
USER gaia

# Run the service
CMD ["python", "-m", "uvicorn", "gaia_mcp.main:app", "--host", "0.0.0.0", "--port", "8765"]
