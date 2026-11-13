#!/usr/bin/env python3
"""Enroll a new TOTP secret for GAIA's own approval gate (GAIA_Project-96xm).

Replaces ApprovalStore's reversed-challenge-string confirmation with real
TOTP -- the human approving a sensitive GAIA action now replies with their
authenticator app's current code instead of a server-generated string
reversed by hand (the old method had no real security: anyone who could
read the approval channel could also reverse a 5-letter string).

Run once, on the host, before (re)starting gaia-mcp / gaia-core with the
TOTP-gated ApprovalStore. Writes the secret to the Docker-secret source
file docker-compose.yml's and docker-compose.candidate.yml's
`gaia_mcp_totp_secret` entries both point at
(`${GAIA_SECRETS_DIR:-$HOME/.gaia/secrets}/gaia_mcp_totp_secret`) and
prints the provisioning URI to scan into an authenticator app.

This is a SEPARATE, independent secret from gaia-ops-broker's own TOTP
enrollment -- that one gates Claude Code's host privilege, a different
actor and a different blast radius; this one gates GAIA's own sensitive
tool calls (ApprovalStore, approval_challenge.py). Per-host project
decision: keep them decoupled rather than share one authenticator entry.

Stdlib only -- deliberately no pyotp/gaia_common dependency, since this
runs on the host before any container exists to provide them (same
reasoning as gaia-doctor's stdlib-only design).
"""
from __future__ import annotations

import os
import secrets
import sys
from pathlib import Path
from urllib.parse import quote

NAME = "gaia-mcp"
ISSUER = "GAIA MCP Approval"
SECRET_FILENAME = "gaia_mcp_totp_secret"

_BASE32_ALPHABET = "ABCDEFGHIJKLMNOPQRSTUVWXYZ234567"


def random_base32_secret(length: int = 32) -> str:
    """A cryptographically random base32 secret, pyotp-compatible (it
    pads before decoding, so any length of valid base32 characters works)."""
    return "".join(secrets.choice(_BASE32_ALPHABET) for _ in range(length))


def provisioning_uri(secret: str, name: str, issuer: str) -> str:
    label = quote(f"{issuer}:{name}")
    return f"otpauth://totp/{label}?secret={secret}&issuer={quote(issuer)}"


def main() -> int:
    secrets_dir = Path(os.environ.get("GAIA_SECRETS_DIR", str(Path.home() / ".gaia" / "secrets")))
    target = secrets_dir / SECRET_FILENAME

    if target.exists():
        reply = input(f"{target} already exists -- overwrite with a new secret? "
                       f"(every container reading it will need to restart) [y/N] ")
        if reply.strip().lower() != "y":
            print("Aborted; existing secret left in place.")
            return 1

    secret = random_base32_secret()
    target.parent.mkdir(parents=True, exist_ok=True)
    target.write_text(secret + "\n")
    os.chmod(target, 0o600)

    uri = provisioning_uri(secret, NAME, ISSUER)
    print(f"Secret written to {target} (mode 600).")
    print()
    print("Scan this into your authenticator app (or add the manual key):")
    print(f"  {uri}")
    print(f"  Manual key: {secret}")
    print()
    print("Restart gaia-mcp and gaia-core (and their -candidate containers, if")
    print("in use) to pick it up:")
    print("  docker compose up -d --force-recreate gaia-mcp gaia-core")
    return 0


if __name__ == "__main__":
    sys.exit(main())
