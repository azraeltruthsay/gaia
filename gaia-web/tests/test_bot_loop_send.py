"""8xr4.21: sends scheduled from a route handler must not deadlock the bot loop.

main.py runs the Discord bot on uvicorn's own loop, so /push handlers are ON
the bot loop. The old sync helper blocked that loop on future.result() and
every cross-tier follow-up timed out.
"""
import asyncio
import threading
import time

import pytest

from gaia_web import discord_interface as di


async def _work():
    await asyncio.sleep(0.01)
    return "sent"


def test_same_loop_does_not_deadlock(monkeypatch):
    async def main():
        monkeypatch.setattr(di, "_bot_loop", asyncio.get_running_loop())
        t = time.monotonic()
        result = await di._await_on_bot_loop(_work(), timeout=2.0)
        return result, time.monotonic() - t

    result, elapsed = asyncio.run(main())
    assert result == "sent"
    assert elapsed < 1.0


def test_cross_thread_loop():
    bot_loop = asyncio.new_event_loop()
    thread = threading.Thread(target=bot_loop.run_forever, daemon=True)
    thread.start()
    try:
        di._bot_loop = bot_loop
        assert asyncio.run(di._await_on_bot_loop(_work(), timeout=2.0)) == "sent"
    finally:
        di._bot_loop = None
        bot_loop.call_soon_threadsafe(bot_loop.stop)
        thread.join(timeout=2)


def test_no_loop_raises(monkeypatch):
    monkeypatch.setattr(di, "_bot_loop", None)
    with pytest.raises(RuntimeError):
        asyncio.run(di._await_on_bot_loop(_work()))


def test_send_to_user_on_bot_loop_completes(monkeypatch):
    """End to end: send_to_user awaited on the bot loop, as /push does."""
    sent = []

    class _User:
        async def send(self, msg):
            sent.append(msg)

    class _Bot:
        def is_ready(self):
            return True

        async def fetch_user(self, uid):
            return _User()

    async def main():
        monkeypatch.setattr(di, "_bot_loop", asyncio.get_running_loop())
        monkeypatch.setattr(di, "_bot", _Bot())
        return await asyncio.wait_for(di.send_to_user("123", "hello"), timeout=3.0)

    assert asyncio.run(main()) is True
    assert sent == ["hello"]
