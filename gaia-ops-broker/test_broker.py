"""
Direct unit tests for OpsBroker's approval logic — no real Discord connection.
Run: .venv/bin/python test_broker.py
"""
import asyncio
import sys
import time
from pathlib import Path
from unittest.mock import AsyncMock, MagicMock

import pyotp

sys.path.insert(0, str(Path(__file__).parent))
import broker as broker_mod  # noqa: E402

APPROVER_ID = 111222333
OTHER_USER_ID = 999888777


def make_message(author_id: int, content: str, is_dm: bool = True):
    msg = MagicMock()
    msg.author.id = author_id
    msg.guild = None if is_dm else MagicMock()
    msg.content = content
    msg.channel.send = AsyncMock()
    return msg


def make_ops_broker(secret: str) -> "broker_mod.OpsBroker":
    ob = broker_mod.OpsBroker.__new__(broker_mod.OpsBroker)
    ob.approver_id = APPROVER_ID
    ob.pending = None
    ob.totp = pyotp.TOTP(secret)
    ob._last_used_code = None
    ob._last_used_at = 0.0
    ob.client = MagicMock()
    ob.client.user.id = 42  # bot's own id, distinct from approver/others
    return ob


async def run() -> int:
    failures = 0

    def check(name: str, cond: bool):
        nonlocal failures
        status = "PASS" if cond else "FAIL"
        if not cond:
            failures += 1
        print(f"  [{status}] {name}")

    secret = pyotp.random_base32()
    totp = pyotp.TOTP(secret)

    # 1. Correct code approves and writes a session
    ob = make_ops_broker(secret)
    broker_mod.SESSION_FILE = Path("/tmp/gaia_ops_broker_test_session.json")
    broker_mod.SESSION_FILE.unlink(missing_ok=True)
    pending = broker_mod.PendingApproval("echo test", "T1", "unit test")
    ob.pending = pending
    msg = make_message(APPROVER_ID, totp.now())
    await ob.on_message(msg)
    check("correct TOTP code approves", pending.result is True)
    check("session file written", broker_mod.SESSION_FILE.exists())

    # 2. Wrong code is rejected
    pending2 = broker_mod.PendingApproval("echo test2", "T1", "")
    ob.pending = pending2
    msg2 = make_message(APPROVER_ID, "000000")
    await ob.on_message(msg2)
    check("wrong code does not resolve pending", not pending2.event.is_set())

    # 3. Replay of the same code within 90s is rejected on a NEW request
    code = totp.now()
    pending3 = broker_mod.PendingApproval("echo test3", "T1", "")
    ob.pending = pending3
    ob._last_used_code = code
    ob._last_used_at = time.time()
    msg3 = make_message(APPROVER_ID, code)
    await ob.on_message(msg3)
    check("replayed code within window is rejected", not pending3.event.is_set())

    # 4. "cancel" denies regardless of code validity
    pending4 = broker_mod.PendingApproval("echo test4", "T1", "")
    ob.pending = pending4
    msg4 = make_message(APPROVER_ID, "cancel")
    await ob.on_message(msg4)
    check("cancel denies", pending4.result is False and pending4.event.is_set())

    # 5. Expired pending request is refused even with a valid code
    pending5 = broker_mod.PendingApproval("echo test5", "T1", "")
    pending5.expiry = time.time() - 1  # force expiry
    ob.pending = pending5
    ob._last_used_code = None  # clear replay guard from test 3
    msg5 = make_message(APPROVER_ID, totp.now())
    await ob.on_message(msg5)
    check("expired request is refused", pending5.result is False)

    # 6. A DM from someone other than the approver is ignored entirely
    pending6 = broker_mod.PendingApproval("echo test6", "T1", "")
    ob.pending = pending6
    msg6 = make_message(OTHER_USER_ID, totp.now())
    await ob.on_message(msg6)
    check("non-approver DM is ignored", not pending6.event.is_set())

    # 7. A message in a guild (not a DM) is ignored
    pending7 = broker_mod.PendingApproval("echo test7", "T1", "")
    ob.pending = pending7
    msg7 = make_message(APPROVER_ID, totp.now(), is_dm=False)
    await ob.on_message(msg7)
    check("guild message is ignored", not pending7.event.is_set())

    broker_mod.SESSION_FILE.unlink(missing_ok=True)
    return failures


if __name__ == "__main__":
    failures = asyncio.run(run())
    print(f"\n{'ALL PASSED' if failures == 0 else f'{failures} FAILURE(S)'}")
    sys.exit(1 if failures else 0)
