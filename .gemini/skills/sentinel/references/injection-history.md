# GAIA Injection History

> Known injection vectors discovered and fixed. A reviewer should verify new code doesn't reintroduce these patterns.

## Shell Injection Vectors (Fixed Feb 2026)

### mcp_client parameter passing
**Before**: Tool parameters interpolated into shell command strings
**After**: Parameters passed as list elements to subprocess with shell=False
**Files affected**: gaia-core/utils/mcp_client.py

### safe_execution command parsing
**Before**: Command string passed directly
**After**: shlex.split() + whitelist check on first token + shell=False
**Files affected**: gaia-common/utils/safe_execution.py

### gaia_rescue_helper
**Before**: Parameters concatenated into command strings
**After**: Proper parameter escaping via shlex
**Files affected**: gaia-core/utils/gaia_rescue_helper.py

## Path Traversal Vectors (Fixed Feb 2026)

### Symlink breakout in file tools
**Before**: Single resolve() check — symlink could point outside allowlist
**After**: Double validation — string prefix match + post-resolve re-check
**Tools affected**: code_read, code_span, code_symbol, ai_write
**Pattern to verify**: Any new file access tool MUST use the double-validation pattern

### Parent directory creation
**Before**: `os.makedirs(parent, exist_ok=True)` without path validation
**After**: Path validation happens BEFORE makedirs
**Risk**: Creating directories outside sandbox via `../../` in filename

## Prompt Injection Surface

### User messages → LLM context
**Path**: Discord message → CognitionPacket.content.prompt → system prompt assembly → Prime/Lite
**Mitigation**: Output sanitizer strips fabricated user/assistant turns from responses
**Residual risk**: User content is included verbatim in prompts — no escaping possible without losing fidelity. Defense is at the output layer.

### Tool results → LLM context
**Path**: MCP tool result → CognitionPacket.tool_routing.execution_result → reinjection into prompt
**Mitigation**: Reinjection capped at 3x; output sanitizer applies to final response
**Residual risk**: Malicious content in fetched web pages or Kanka entities could influence LLM reasoning. No content sanitization on tool results currently.

## Lite Model Pattern-Matching Exploit (Fixed Feb 2026)

**Not a security vulnerability per se, but security-adjacent:**
When Lite model (3B) received tool execution results alongside EXECUTE: examples in the system prompt, it pattern-matched and re-emitted tool execution commands instead of synthesizing a response.

**Fix**: Suppress TOOL CALLING CONVENTION block when tool_routing.execution_status == EXECUTED. Inject assistant prefill to steer toward prose output.

**Lesson for review**: Small models don't understand instructions — they pattern-match. System prompts that contain executable syntax are an implicit injection surface.

## No Active Unresolved Vectors

As of Feb 2026 audit, no known unresolved injection vectors. All identified issues were fixed in commit c9f0581.
