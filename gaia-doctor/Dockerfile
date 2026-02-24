FROM python:3.11-slim

# Create non-root user for container execution (host UID mapping)
RUN groupadd -g 1000 gaia && useradd -u 1000 -g gaia -m -s /bin/bash gaia
ENV HOME=/home/gaia
ENV USER=gaia

# Install curl (for healthcheck). Docker CLI + compose plugin are bind-mounted
# from the host via docker-compose.yml to avoid pulling the full docker.io daemon.
RUN apt-get update && apt-get install -y --no-install-recommends \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Create docker CLI plugin directory for bind-mounted compose plugin
RUN mkdir -p /usr/lib/docker/cli-plugins

WORKDIR /app

# Copy watchdog script (stdlib only — no pip install needed)
COPY gaia-doctor/doctor.py .

# Create shared state directory
RUN mkdir -p /shared/doctor && chown -R gaia:gaia /app /shared/doctor

# Health check
HEALTHCHECK --interval=30s --timeout=5s --start-period=10s --retries=3 \
    CMD curl -f http://localhost:6419/health || exit 1

EXPOSE 6419

USER gaia

CMD ["python", "doctor.py"]
