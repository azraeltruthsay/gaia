"""6nw3: the time-check canary waits out a request already in flight."""

import json

from gaia_common.utils import heartbeat_time_check as H


def test_canary_uses_check_timeout(monkeypatch):
    seen = {}

    class Resp:
        def __enter__(self):
            return self

        def __exit__(self, *a):
            return False

        def read(self):
            return json.dumps({"choices": [{"message": {"content": "x"}}]}).encode()

    def fake_urlopen(req, timeout):
        seen["timeout"] = timeout
        return Resp()

    monkeypatch.setattr(H, "urlopen", fake_urlopen)
    H.run_time_check("http://core:8092")
    assert seen["timeout"] == H.CHECK_TIMEOUT >= 45
