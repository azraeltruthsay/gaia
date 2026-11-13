"""TOTP verification for human-approval gates (GAIA_Project-96xm).

Replaces a reversed-challenge-string confirmation (ApprovalStore handed out
a server-generated code and asked the human to type it backwards) that had
zero cryptographic value -- anyone who can read the approval channel can
also reverse a short string, so it was one factor (channel access) dressed
up as two.

Ported from gaia-ops-broker/broker.py's live-verified pattern (pyotp, same
algorithm as Google Authenticator/Authy) rather than re-derived. That
service stays on its own inline implementation -- it is deliberately
standalone/dependency-free (see its own docstring) and gates a different
actor (Claude Code's host privilege) than this module (GAIA's own sensitive
tool calls), so it is not refactored to import gaia_common. This module
exists so gaia-mcp, which already depends on gaia_common, doesn't have to
re-derive the same logic from scratch.
"""

from __future__ import annotations

import logging
import os
import threading
import time
from pathlib import Path
from typing import Optional

import pyotp

logger = logging.getLogger("GAIA.Totp")


class TotpVerifier:
    """Verifies a 6-digit TOTP code against one secret, with replay protection.

    A TOTP code stays valid for ~30-60s (valid_window=1) and isn't tied to
    any one pending action the way the old per-action challenge string was --
    without replay protection, a single observed code could approve two
    different pending actions inside its validity window. Mirrors
    gaia-ops-broker/broker.py's 90s reuse-rejection.
    """

    def __init__(self, secret: str, reuse_window_s: int = 90):
        self._totp = pyotp.TOTP(secret)
        self._reuse_window_s = reuse_window_s
        self._last_code: Optional[str] = None
        self._last_code_at: float = 0.0
        self._lock = threading.Lock()

    def verify(self, code: str) -> bool:
        code = (code or "").strip()
        if not code:
            return False
        with self._lock:
            now = time.time()
            if code == self._last_code and (now - self._last_code_at) < self._reuse_window_s:
                logger.warning(
                    "TOTP code reuse rejected (replay within %ds window)",
                    self._reuse_window_s,
                )
                return False
            if not self._totp.verify(code, valid_window=1):
                return False
            self._last_code = code
            self._last_code_at = now
            return True


def load_totp_verifier(*paths: str, env_var: str = "") -> Optional[TotpVerifier]:
    """Load a TotpVerifier from the first existing secret file among `paths`,
    falling back to the file named by `env_var` if none exist. Returns None
    (and logs a warning) when no secret is found -- callers must fail
    closed on `None`, not treat it as "approval disabled".

    Mirrors gaia_common.utils.service_auth._load_secret()'s shape.
    """
    candidates = list(paths)
    if env_var:
        env_path = os.environ.get(env_var, "")
        if env_path:
            candidates.append(env_path)

    for path in candidates:
        if path and os.path.exists(path):
            try:
                secret = Path(path).read_text().strip()
                if secret:
                    logger.info("TOTP secret loaded from %s", path)
                    return TotpVerifier(secret)
            except Exception:
                continue

    logger.warning("No TOTP secret found (checked %s) -- approval gate will fail closed", candidates)
    return None


def enroll(path: Path, name: str, issuer: str) -> str:
    """Generate a new TOTP secret, write it to `path` (mode 0600), and
    return the provisioning URI for an authenticator app. Intended to be
    called once, by hand, before the service that reads `path` is started.
    """
    secret = pyotp.random_base32()
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(secret)
    os.chmod(path, 0o600)
    return pyotp.TOTP(secret).provisioning_uri(name=name, issuer_name=issuer)
