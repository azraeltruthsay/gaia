"""Tests for gaia_common.exceptions — exception hierarchy."""

import pytest

from gaia_common.exceptions import (
    GaiaError,
    GaiaConfigError,
    GaiaModelError,
    GaiaSafetyError,
    GaiaToolError,
    GaiaNetworkError,
)


class TestGaiaError:
    """Test the GaiaError base class."""

    def test_construction_with_known_code(self):
        err = GaiaError("GAIA-CORE-001", detail="test detail")
        assert err.error_code == "GAIA-CORE-001"
        assert err.detail == "test detail"
        assert err.hint != ""  # Should be populated from registry
        assert "GAIA-CORE-001" in str(err)

    def test_construction_with_unknown_code(self):
        err = GaiaError("GAIA-FAKE-999", detail="unknown code")
        assert err.error_code == "GAIA-FAKE-999"
        assert err.hint == ""
        assert "GAIA-FAKE-999" in str(err)

    def test_to_dict(self):
        err = GaiaError("GAIA-CORE-001", detail="test", context={"key": "val"})
        d = err.to_dict()
        assert d["error_code"] == "GAIA-CORE-001"
        assert d["detail"] == "test"
        assert "hint" in d
        assert d["context"] == {"key": "val"}

    def test_is_exception(self):
        err = GaiaError("GAIA-CORE-001")
        assert isinstance(err, Exception)
        with pytest.raises(GaiaError):
            raise err


class TestSubclasses:
    """Test that all subclasses inherit properly."""

    def test_config_error(self):
        err = GaiaConfigError("GAIA-CORE-003")
        assert isinstance(err, GaiaError)
        assert isinstance(err, GaiaConfigError)

    def test_model_error(self):
        err = GaiaModelError("GAIA-CORE-050")
        assert isinstance(err, GaiaError)

    def test_safety_error(self):
        err = GaiaSafetyError("GAIA-CORE-010")
        assert isinstance(err, GaiaError)

    def test_tool_error(self):
        err = GaiaToolError("GAIA-MCP-001")
        assert isinstance(err, GaiaError)

    def test_network_error(self):
        err = GaiaNetworkError("GAIA-WEB-001")
        assert isinstance(err, GaiaError)
