"""Tests for gaia_common.errors — error registry."""

import logging
import pytest

from gaia_common.errors import (
    ErrorCategory,
    register,
    lookup,
    all_errors,
    _REGISTRY,
)


class TestErrorRegistry:
    """Test the error registry CRUD operations."""

    def test_lookup_existing_code(self):
        defn = lookup("GAIA-CORE-001")
        assert defn is not None
        assert defn.code == "GAIA-CORE-001"
        assert defn.level == logging.CRITICAL
        assert defn.category == ErrorCategory.LOOP
        assert "HEALING_REQUIRED" in defn.hint

    def test_lookup_nonexistent_returns_none(self):
        assert lookup("GAIA-FAKE-999") is None

    def test_register_duplicate_raises(self):
        with pytest.raises(ValueError, match="Duplicate error code"):
            register("GAIA-CORE-001", "duplicate", "should fail")

    def test_all_errors_returns_copy(self):
        errors = all_errors()
        assert isinstance(errors, dict)
        assert len(errors) >= 40, f"Expected >=40 registered errors, got {len(errors)}"
        # Mutating the copy should not affect the registry
        errors["FAKE"] = None
        assert "FAKE" not in _REGISTRY

    def test_error_def_is_frozen(self):
        defn = lookup("GAIA-CORE-001")
        assert defn is not None
        with pytest.raises(AttributeError):
            defn.code = "GAIA-CORE-999"

    def test_all_codes_follow_naming_convention(self):
        import re
        pattern = re.compile(r"^GAIA-[A-Z]+-\d{3}$")
        for code in all_errors():
            assert pattern.match(code), f"Code {code} does not match GAIA-{{SERVICE}}-{{NNN}}"

    def test_category_enum_values(self):
        assert ErrorCategory.MODEL.value == "model"
        assert ErrorCategory.SAFETY.value == "safety"
        assert ErrorCategory.NETWORK.value == "network"

    def test_level_conventions(self):
        """GAIA-CORE critical errors should be in the 001-009 range."""
        for code, defn in all_errors().items():
            if not code.startswith("GAIA-CORE-"):
                continue
            num = int(code.rsplit("-", 1)[-1])
            if num <= 9:
                assert defn.level == logging.CRITICAL, (
                    f"{code} is in 001-009 range but level is {logging.getLevelName(defn.level)}, not CRITICAL"
                )
