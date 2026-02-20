# Code Quality

## Linting

All services use ruff for linting and formatting:

```bash
# Check
ruff check gaia-core/

# Fix auto-fixable issues
ruff check --fix gaia-core/

# Format
ruff format gaia-core/
```

## Type Checking

mypy is used for type checking:

```bash
docker compose exec -T gaia-core python -m mypy gaia_core/ --ignore-missing-imports
```

## Testing

Tests run inside Docker containers — never on the host:

```bash
# Unit tests
docker compose exec -T gaia-core python -m pytest /app/tests/ -v --tb=short

# Specific test file
docker compose exec -T gaia-core python -m pytest /app/tests/test_checkpoint_endpoint.py -v

# With coverage
docker compose exec -T gaia-core python -m pytest /app/tests/ --cov=gaia_core --cov-report=term
```

## Conventions

- **Logging:** Use `logging.getLogger("GAIA.<Service>.<Component>")` naming
- **Health checks:** Every service exposes `GET /health` returning `{"status": "healthy", "service": "<name>"}`
- **Environment:** Configuration via environment variables, never hardcoded paths
- **Shared code:** Common protocols and utilities go in `gaia-common/`
- **Imports:** Health check filter suppression at module top level
