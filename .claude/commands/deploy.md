Run the promotion pipeline: $ARGUMENTS

Execute: `./scripts/promote_pipeline.sh $ARGUMENTS`

Common options:
- `--dry-run` — show what would happen without making changes
- `--skip-validate` — skip ruff/mypy/pytest validation
- `--skip-smoke` — skip cognitive smoke battery
- `--services LIST` — only promote specific services

Always show the user the pipeline output in real-time.
If no arguments provided, run a full promotion with all checks.
