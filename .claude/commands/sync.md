Sync candidate and production for service: $ARGUMENTS

1. Determine direction (candidate→production or production→candidate)
2. Copy files using cp -r preserving structure
3. Restart affected containers
4. Verify health after restart

Services: gaia-core, gaia-web, gaia-mcp, gaia-common, gaia-study, gaia-audio, gaia-orchestrator
(NOT gaia-prime — `candidates/gaia-prime/` is a divergent experimental build, never blind-synced)
Default direction: candidate → production (promotion)

If direction is "reverse" or "prod→candidate", copy production to candidate instead.

Example usage:
- `/project:sync gaia-core` — promote candidate to production
- `/project:sync gaia-web reverse` — copy production to candidate
- `/project:sync all` — sync all services candidate → production
