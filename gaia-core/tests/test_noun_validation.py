import pytest
from gaia_common.utils.entity_validator import EntityValidator

def test_hard_mappings():
    validator = EntityValidator()
    text = "Asriel the architect met with Gaya."
    corrected = validator.correct_text(text)
    assert "Azrael the architect" in corrected
    assert "GAIA" in corrected

def test_fuzzy_matching():
    validator = EntityValidator(canonical_entities=["Samvega", "BlueShot"])
    text = "The system entered Samvege mode after exposure to Blueshot."
    corrected = validator.correct_text(text)
    assert "Samvega" in corrected
    assert "BlueShot" in corrected

def test_case_insensitivity_in_hard_mapping():
    validator = EntityValidator()
    text = "asriel said gaya is cool."
    corrected = validator.correct_text(text)
    assert "Azrael said GAIA is cool." in corrected

def test_no_correction_needed():
    validator = EntityValidator()
    text = "Azrael and GAIA are working on the Core."
    corrected = validator.correct_text(text)
    assert corrected == text

def test_empty_text():
    validator = EntityValidator()
    assert validator.correct_text("") == ""
    assert validator.correct_text(None) is None
