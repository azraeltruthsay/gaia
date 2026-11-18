---
paths:
  - gaia-mcp/**
  - "**/security/**"
---

# Safety & Security Rules

## Sovereign Shield
- `py_compile` gate on all `.py` writes (ai_write, write_file, replace) in `gaia-mcp/gaia_mcp/tools.py`
- Prevents GAIA from introducing syntax errors during self-repair

## Blast Shield
Deterministic pre-flight safety (independent of LLM reasoning):
- **run_shell**: Blocks `rm -rf`, `sudo`, `mkfs`, `dd`, `> /dev/sd`
- **write_file**: Blocks `/etc`, `/boot`, `.ssh` paths

## Circuit Breaker
- `HEALING_REQUIRED.lock` in `/shared/` — created on fatal loop threshold
- AgentCore checks at turn start; refuses to process if lock exists
- Manual intervention required to clear

## Zero-Trust Identity
- Docker secrets at `/run/secrets/` for API keys
- No secrets in env vars or config files
- Never commit secrets, .env files, or credentials
