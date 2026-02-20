# Adding a New Service

Follow this checklist when adding a new service to the GAIA stack.

## 1. Create the Service Directory

```
gaia-<name>/
├── Dockerfile
├── pyproject.toml
├── requirements.txt
├── gaia_<name>/
│   ├── __init__.py
│   └── main.py
└── tests/
    └── __init__.py
```

## 2. Create the Candidate Copy

Mirror the directory under `candidates/`:

```
candidates/gaia-<name>/
├── Dockerfile
├── ...
```

## 3. Add to docker-compose.yml

Follow the existing service pattern:

```yaml
gaia-<name>:
  build:
    context: .
    dockerfile: ./gaia-<name>/Dockerfile
  image: localhost:5000/gaia-<name>:local
  container_name: gaia-<name>
  hostname: gaia-<name>
  restart: unless-stopped

  volumes:
    - ./gaia-<name>:/app:rw
    - ./gaia-common:/gaia-common:ro

  environment:
    - PYTHONPATH=/app:/gaia-common
    - GAIA_SERVICE=<name>
    - GAIA_ENV=${GAIA_ENV:-development}
    - LOG_LEVEL=${LOG_LEVEL:-INFO}

  networks:
    - gaia-net

  healthcheck:
    test: ["CMD", "curl", "-f", "http://localhost:<port>/health"]
    interval: 30s
    timeout: 10s
    retries: 3
    start_period: 30s
```

## 4. Add to docker-compose.candidate.yml

Same pattern but with candidate-specific settings (separate ports, volumes, profiles).

## 5. Create a Blueprint

Add `knowledge/blueprints/gaia-<name>.yaml` following the schema in existing blueprints.

## 6. Add to gaia.sh

Add a `cmd_<name>` function and register it in the dispatch case statement.

## 7. Add Health Check Filter

If using FastAPI, add the health check log filter:

```python
try:
    from gaia_common.utils import install_health_check_filter
    install_health_check_filter()
except ImportError:
    pass
```

## 8. Update This Wiki

Add an architecture page and update the service map on the index page.
