Run tests for: $ARGUMENTS

Execute pytest inside the appropriate Docker container.
NEVER run tests on the host — Python 3.14 lacks project dependencies.

Container mapping:
- gaia-core: `docker compose exec -T gaia-core python -m pytest tests/ -v --tb=short`
- gaia-web: `docker compose exec -T gaia-web python -m pytest tests/ -v --tb=short`
- gaia-common: `docker compose exec -T gaia-core python -m pytest /gaia-common/tests/ -v --tb=short`
- all: run each service sequentially

If a specific test file or path is provided (e.g., `gaia-core tests/test_foo.py`), pass it through.
Show full output to the user.
