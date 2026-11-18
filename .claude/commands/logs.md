Show recent logs for: $ARGUMENTS

Run: `docker logs <service> --tail 50`

If "candidate" is mentioned, use the candidate container name (e.g., gaia-core-candidate).
If a number is specified (e.g., `gaia-core 100`), use that as the tail count instead of 50.
If "follow" or "-f" is specified, use `docker logs -f --tail 20` as a background task.
