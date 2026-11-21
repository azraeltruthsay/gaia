"""Links aren't secrets; keys still are (2026-11-20)."""
from gaia_web.security.scanner import SecretsScanner


def test_google_doc_link_is_not_a_secret():
    text = "GAIA, try to access this link please. https://docs.google.com/document/d/1Kx9QpZ3vR7tYb2LmN8wH4sJ6dF0aE5cU1gT9iO2pV3q/edit?usp=sharing"
    assert SecretsScanner().scan(text) == []


def test_keys_still_caught_inside_and_outside_links():
    s = SecretsScanner()
    assert s.scan("my key is ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8")
    assert s.scan("see https://example.com/?k=ghp_" + "a1B2c3D4e5F6g7H8i9J0k1L2m3N4o5P6q7R8")
    assert s.scan("token Zx9qL2vB7nM4kP8wR3tY6uI1oA5sD0fG")       # high-entropy, no link
