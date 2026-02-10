"""Tests for the PersonaData and BasePersonaManager."""

import pytest
from gaia_common.base.persona import PersonaData, BasePersonaManager


class TestPersonaData:
    """Test PersonaData creation and serialization."""

    def test_from_dict_minimal(self):
        data = {"name": "TestBot"}
        persona = PersonaData.from_dict(data)
        assert persona.name == "TestBot"
        assert persona.identity_id == "GAIA"
        assert persona.role == "Default"
        assert persona.enabled is True

    def test_from_dict_full(self):
        data = {
            "name": "Scholar",
            "identity_id": "GAIA-v2",
            "role": "Researcher",
            "tone_hint": "academic",
            "safety_profile_id": "strict",
            "traits": {"curiosity": 0.9},
            "knowledge_base_name": "research_kb",
            "description": "A scholarly persona",
            "system_prompt_additions": "Cite sources when possible.",
            "enabled": False,
        }
        persona = PersonaData.from_dict(data)
        assert persona.name == "Scholar"
        assert persona.tone_hint == "academic"
        assert persona.traits["curiosity"] == 0.9
        assert persona.enabled is False

    def test_to_dict_roundtrip(self):
        original = PersonaData(
            name="TestBot",
            identity_id="GAIA",
            role="Helper",
            tone_hint="friendly",
        )
        d = original.to_dict()
        restored = PersonaData.from_dict(d)
        assert restored.name == original.name
        assert restored.role == original.role
        assert restored.tone_hint == original.tone_hint

    def test_defaults(self):
        persona = PersonaData(name="Minimal")
        assert persona.identity_id == "GAIA"
        assert persona.traits == {}
        assert persona.knowledge_base_name is None


class TestBasePersonaManager:
    """Test the abstract base class contract."""

    def test_get_persona_returns_persona_data(self):
        class TestManager(BasePersonaManager):
            def load_persona_data(self, name):
                if name == "test":
                    return {"name": "test", "role": "Tester"}
                return None

            def list_personas(self):
                return ["test"]

        mgr = TestManager()
        persona = mgr.get_persona("test")
        assert isinstance(persona, PersonaData)
        assert persona.role == "Tester"

    def test_get_persona_returns_none_for_missing(self):
        class EmptyManager(BasePersonaManager):
            def load_persona_data(self, name):
                return None

            def list_personas(self):
                return []

        mgr = EmptyManager()
        assert mgr.get_persona("nonexistent") is None
