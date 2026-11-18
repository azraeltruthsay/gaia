Validate candidate code: $ARGUMENTS

Execute: `./scripts/validate.sh $ARGUMENTS`

Runs ruff lint, mypy type check, and pytest unit tests inside Docker containers.
If no service specified, validates all candidates.

Show full output so the user can see any lint errors, type issues, or test failures.
