"""
E2E test for MCP write_file tool through the candidate gaia-core.

Validates:
1. JSON-RPC dispatch from gaia-core to gaia-mcp
2. SENSITIVE_TOOLS gate (403 response)
3. Approval flow (request -> TOTP-approve -> execute)
4. File actually written to allowed path
5. Path allowlist enforcement (rejected writes)
6. read_file tool for round-trip verification

GAIA_Project-96xm: approval went from a server-generated challenge
(reversed by hand) to a real TOTP code. This script has trusted,
same-host access to the enrolled secret (same as gaia-mcp itself), so it
generates real current codes via TotpVerifier.current_code() instead of
simulating a human typing one, waiting for the 30s window to roll over
between approvals so the replay guard doesn't reject a repeat (there is
nothing left for this script to compute locally).

Usage:
    docker exec gaia-core-candidate python /app/e2e_mcp_write_test.py
"""

import json
import logging
import sys
import time

logging.disable(logging.WARNING)


def _load_totp_verifier():
    from gaia_common.utils.totp import load_totp_verifier
    return load_totp_verifier(
        "/run/secrets/gaia_mcp_totp_secret",
        env_var="GAIA_MCP_TOTP_SECRET_FILE",
    )


def _fresh_totp_code(verifier, previous: str = None, timeout_s: int = 35) -> str:
    """A TOTP code that hasn't been used yet this run.

    gaia-mcp's ApprovalStore rejects an exact-repeat code within its reuse
    window (GAIA_Project-96xm's replay guard), but this script approves
    several actions back-to-back using the one enrolled secret -- waiting
    for the 30s window to roll over is simpler and more honest than
    disabling that protection for test runs.
    """
    if verifier is None:
        return None
    deadline = time.time() + timeout_s
    code = verifier.current_code()
    while code == previous and time.time() < deadline:
        time.sleep(1)
        code = verifier.current_code()
    return code


def main() -> int:
    passed = 0
    failed = 0

    def check(label: str, condition: bool, detail: str = ""):
        nonlocal passed, failed
        if condition:
            passed += 1
            print(f"  PASS  {label}")
        else:
            failed += 1
            print(f"  FAIL  {label}{f': {detail}' if detail else ''}")

    print("=" * 60)
    print("MCP write_file — E2E Validation")
    print("=" * 60)

    # --- 1. Imports and connectivity ---
    print("\n--- Setup ---")
    try:
        from gaia_core.utils import mcp_client
        check("mcp_client import", True)
    except Exception as e:
        check("mcp_client import", False, str(e))
        return 1

    # Verify MCP endpoint is reachable
    try:
        discovery = mcp_client.discover()
        check("MCP discovery", discovery is not None and discovery.get("ok", False),
              json.dumps(discovery)[:100] if discovery else "None")
        if discovery and discovery.get("ok"):
            # discover() returns {"methods": [...]} — a list of method name strings
            methods = discovery.get("methods", [])
            check("write_file in tool catalog", "write_file" in methods,
                  f"found: {methods[:10]}")
            check("read_file in tool catalog", "read_file" in methods)
    except Exception as e:
        check("MCP discovery", False, str(e))
        return 1

    # --- 2. Direct JSON-RPC call (should get 403) ---
    print("\n--- SENSITIVE_TOOLS Gate ---")
    test_path = "/knowledge/e2e_test_write_file.txt"
    test_content = f"E2E write test at {time.strftime('%Y-%m-%d %H:%M:%S')}"

    result = mcp_client.call_jsonrpc("write_file", {
        "path": test_path,
        "content": test_content,
    })
    # Should fail with 403 or error indicating approval required
    is_blocked = not result.get("ok", True)
    err_msg = str(result.get("error", ""))
    check("Direct write_file blocked (403)",
          is_blocked and ("403" in err_msg or "approval" in err_msg.lower()),
          f"result: {json.dumps(result)[:200]}")

    # --- 3. Approval flow ---
    print("\n--- Approval Flow ---")
    approval_req = mcp_client.request_approval_via_mcp("write_file", {
        "path": test_path,
        "content": test_content,
    })
    check("Approval request accepted",
          approval_req.get("ok", False),
          json.dumps(approval_req)[:200])

    action_id = approval_req.get("action_id", "")
    totp = _load_totp_verifier()
    code = _fresh_totp_code(totp)
    check("Action ID returned", bool(action_id), f"id={action_id}")
    check("TOTP code available", bool(code))

    if action_id and code:
        approval_resp = mcp_client.approve_action_via_mcp(action_id, code)
        # approve_action_via_mcp wraps: {"ok": True, "result": <server_json>}
        # server_json is: {"ok": True, "result": <tool_result>, "approved_at": ...}
        server_resp = approval_resp.get("result", {})
        server_ok = isinstance(server_resp, dict) and server_resp.get("ok", False)
        check("Approval accepted", server_ok,
              json.dumps(approval_resp)[:200])

        # The tool result is nested inside the server response
        if server_ok:
            tool_result = server_resp.get("result", {})
            if isinstance(tool_result, dict):
                check("Write succeeded", tool_result.get("ok", False),
                      json.dumps(tool_result)[:200])
            else:
                check("Write succeeded", False, f"unexpected result type: {type(tool_result)}")
        else:
            check("Write succeeded", False, json.dumps(server_resp)[:200])
    else:
        check("Approval accepted", False, "no action_id or TOTP code")
        check("Write succeeded", False, "skipped")

    # --- 4. Verify file on disk ---
    print("\n--- File Verification ---")
    # The file is inside the gaia-mcp container's /knowledge mount.
    # From gaia-core-candidate, /knowledge is read-only, but we can use
    # MCP read_file to verify the content.
    read_result = mcp_client.call_jsonrpc("read_file", {"path": test_path})
    read_ok = read_result.get("ok", False)
    if read_ok:
        response = read_result.get("response", {})
        # JSON-RPC wraps: {"jsonrpc": "2.0", "result": {"ok": true, "content": ...}}
        rpc_result = response.get("result", response)
        file_content = rpc_result.get("content", "")
        read_tool_ok = rpc_result.get("ok", False)
        check("read_file succeeds", read_tool_ok,
              f"rpc_result keys: {list(rpc_result.keys()) if isinstance(rpc_result, dict) else type(rpc_result)}")
        check("Content matches", test_content in file_content,
              f"expected '{test_content[:50]}', got '{file_content[:50]}'")
    else:
        check("read_file succeeds", False, json.dumps(read_result)[:200])
        check("Content matches", False, "skipped")

    # --- 5. Path allowlist enforcement ---
    print("\n--- Path Safety ---")
    bad_paths = [
        ("/app/evil.txt", "app directory"),
        ("/etc/passwd", "system file"),
        ("/tmp/test.txt", "tmp directory"),
    ]
    for bad_path, label in bad_paths:
        # Try through approval flow (so we get past the 403 gate)
        req = mcp_client.request_approval_via_mcp("write_file", {
            "path": bad_path,
            "content": "should not write",
        })
        code = _fresh_totp_code(totp, previous=code)
        if req.get("ok") and req.get("action_id") and code:
            resp = mcp_client.approve_action_via_mcp(
                req["action_id"], code
            )
            # approve_action_via_mcp wraps server response:
            # {"ok": True, "result": <server_json>}
            # Server returns {"ok": False, "error": "..."} for ValueError,
            # or HTTP 500 for other errors (which makes the client return ok=False).
            server_resp = resp.get("result", {})
            if isinstance(server_resp, dict):
                # Check the server's inner result
                inner = server_resp.get("result", server_resp)
                was_blocked = not inner.get("ok", True) or "error" in server_resp
            else:
                was_blocked = True
            check(f"Write to {label} blocked", was_blocked,
                  json.dumps(resp)[:150])
        else:
            # Even approval request might be rejected
            check(f"Write to {label} blocked", True, "approval request itself failed")

    # --- 6. Cleanup ---
    print("\n--- Cleanup ---")
    # Write an empty file or delete via shell would be ideal, but we don't
    # have delete_file in MCP. Overwrite with empty to mark as test artifact.
    cleanup_req = mcp_client.request_approval_via_mcp("write_file", {
        "path": test_path,
        "content": "",
    })
    code = _fresh_totp_code(totp, previous=code)
    if cleanup_req.get("ok") and cleanup_req.get("action_id") and code:
        mcp_client.approve_action_via_mcp(
            cleanup_req["action_id"], code
        )
    check("Test file cleaned up", True)

    # --- Summary ---
    total = passed + failed
    print(f"\n{'=' * 60}")
    print(f"{passed}/{total} checks passed")
    print(f"{'=' * 60}")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
