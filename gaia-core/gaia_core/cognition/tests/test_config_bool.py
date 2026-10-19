"""Unit tests for _config_bool (md2o): SafeConfigProxy's __getattr__ never
raises AttributeError for unmapped names -- it returns None instead, which
silently defeats getattr()'s own default argument for boolean flags like
_slim_prompt. _config_bool substitutes the default only when the proxy's
answer is None, while still respecting an explicit, genuine False (a plain
`or default` guard would wrongly flip that back to True)."""

from gaia_core.cognition.agent_core import _config_bool


class _StubRaw:
    """Stands in for the real Config object SafeConfigProxy wraps."""


class _StubProxy:
    """Mimics SafeConfigProxy's real behavior: returns the real attribute
    when present, otherwise None (never raises AttributeError)."""

    def __init__(self, raw: _StubRaw):
        self._raw = raw

    def __getattr__(self, name):
        return getattr(self._raw, name, None)


def test_missing_attr_returns_default_true():
    proxy = _StubProxy(_StubRaw())
    assert _config_bool(proxy, "_slim_prompt", True) is True


def test_missing_attr_returns_default_false():
    proxy = _StubProxy(_StubRaw())
    assert _config_bool(proxy, "_slim_prompt", False) is False


def test_explicit_true_is_preserved():
    raw = _StubRaw()
    raw._slim_prompt = True
    proxy = _StubProxy(raw)
    assert _config_bool(proxy, "_slim_prompt", False) is True


def test_explicit_false_is_preserved_not_flipped_to_default():
    raw = _StubRaw()
    raw._slim_prompt = False
    proxy = _StubProxy(raw)
    # This is the exact case a plain `getattr(...) or default` guard gets
    # wrong: False or True == True, silently overriding a real, intentional
    # False. _config_bool must not do that.
    assert _config_bool(proxy, "_slim_prompt", True) is False
