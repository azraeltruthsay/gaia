# GAIA Security Patterns

> Distilled from the Feb 2026 comprehensive audit (commit c9f0581). These are the patterns that exist — a reviewer should verify they are correctly applied in new code.

## Input Validation

### File Access Allowlist (gaia-mcp)
```python
allow_roots = [
    Path("/knowledge").resolve(),
    Path("/gaia-common").resolve(),
    Path("/sandbox").resolve(),
]
```
- String prefix match + post-resolve symlink check (double validation)
- `write_file` re-resolves after `resolve()` to catch symlink attacks
- `read_file`: 512 KB max (DoS prevention)
- `list_files`: 5000 entries max
- `list_tree`: 1000 entries, depth limit 6

### Shell Execution Safety (gaia-common)
```python
def run_shell_safe(command: str, safe_cmds: Set[str]) -> str:
    parts = shlex.split(command)       # No shell=True
    if parts[0] not in safe_cmds:      # First token whitelist
        raise ValueError(...)
    subprocess.run(parts, shell=False, timeout=10)
```
- Whitelist: `ls, find, grep, cat, head, tail, wc` (from gaia_constants.json)
- `shell=False` prevents `; || &&` escapes
- 10-second timeout prevents hangs

### Grep Injection Prevention (gaia-mcp)
```python
keywords = query.split()
safe_keywords = [k for k in keywords if k.isalnum()]
pattern = "|".join(safe_keywords)
cmd = ["grep", "-r", "-i", "-l", "-E", pattern, doc_dir]
subprocess.run(cmd, capture_output=True, shell=False)
```
- Only alphanumeric tokens pass through
- `shell=False` + list args prevents interpretation

### Output Sanitization (gaia-core)
- Detects + removes fabricated user messages: `User: | User Message | You >`
- Limits backtick count (prevents excessive code blocks)
- Enforces single-response format (removes multi-turn artifacts)
- File verification gate: requires `ai.read()` EXECUTE if file-related response

## Cryptographic Patterns

- **Hashing**: SHA-256 (migrated from MD5 in Feb 2026 audit)
- **Timestamps**: `datetime.now(UTC)` (migrated from deprecated `datetime.utcnow()`)
- **Approval codes**: 5-char alphabetic, challenge-response reversal

## Audit Trail

- Approval flow logged: action_id, method, challenge, TTL
- `write_file` logged: target path, byte count
- `run_shell` logged: command token, safe_cmds check
- CognitionPacket.Response.audit: vote count, signatures
- CognitionPacket.Response.governance: approval chain tracking

## Known Fixes Applied (Feb 2026)

| Category | What was fixed |
|----------|---------------|
| Shell injection | mcp_client, safe_execution, gaia_rescue_helper param escaping |
| Path traversal | code_read, code_span, code_symbol, ai_write symlink checks |
| GPU race conditions | Wake-state concurrency + owner bug on async wake |
| Blocking I/O | Docker SDK calls → run_in_executor |
| Hardcoded URLs | Service endpoints → environment variables |
| Cryptography | MD5 → SHA256 across embedding/indexing |
| Timestamp safety | utcnow() → now(UTC) across 53 files |
