---
paths:
  - tests/**
  - "**/tests/**"
---

# Testing Rules

**CRITICAL**: Never run pytest on the host — Python 3.14 lacks project dependencies. Always use Docker:

```bash
# gaia-core tests
docker compose exec -T gaia-core python -m pytest <path> -v --tb=short

# gaia-common tests (via gaia-core container)
docker compose exec -T gaia-core python -m pytest /gaia-common/tests/<file> -v --tb=short

# gaia-web tests
docker compose exec -T gaia-web python -m pytest <path> -v --tb=short
```

Production containers mount production dirs — to test candidate code there, copy it to production paths first. Alternatively, run tests in the candidate containers (e.g. `gaia-core-candidate`), which mount `candidates/` directly (see `docker-compose.candidate.yml`).
