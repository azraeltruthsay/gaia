import pytest
import json
from unittest.mock import MagicMock, patch
from gaia_core.cognition.agent_core import AgentCore
from gaia_common.protocols.cognition_packet import CognitionPacket

@pytest.fixture
def mock_agent_core():
    config = MagicMock()
    config.get_persona_instructions.return_value = "Test Persona"
    # Ensure all numeric limits are real ints, not MagicMocks
    config.max_reflection_tokens = 500
    config.max_history_len = 20
    config.max_tokens = 2048
    config.context_length = 8192
    config._slim_prompt = False
    config.cheat_sheet_path = "/tmp/cheat.json"
    config.identity_file_path = "/tmp/id.json"
    
    config.constants = {
        "KNOWLEDGE_BASES": {},
        "COGNITIVE_AUDIT": {"enabled": False},
        "MODEL_CONFIGS": {"observer": {"enabled": False}},
        "LOOP_DETECTION_ENABLED": False,
        "MODELS": {
            "lite": {"name": "lite"},
            "gpu_prime": {"name": "prime"}
        }
    }
    
    with patch('gaia_common.utils.entity_validator.EntityValidator'), \
         patch('gaia_core.memory.session_manager.SessionManager'), \
         patch('gaia_core.main.AIManagerShim'):
        
        core = AgentCore(config)
        core.logger = MagicMock()
        core.model_pool = MagicMock()
        core.model_pool.models = {"lite": MagicMock(), "gpu_prime": MagicMock()}
        core.model_pool.get_idle_model.return_value = "gpu_prime"
        core.model_pool._gpu_released = False
        
        # Explicitly mock identity_guardian to PASS
        core.identity_guardian = MagicMock()
        core.identity_guardian.verify_prompt_stack.return_value = (True, [])
        core.identity_guardian.validate_reflex.return_value = True
        
        return core

def test_stream_integrity_no_double_posting(mock_agent_core):
    """Verify that tokens are unique and not repeated in a final block."""
    session_id = "test_integrity"
    user_input = "Hello"
    
    mock_llm = MagicMock()
    mock_agent_core.model_pool.acquire_model_for_role.return_value = mock_llm
    
    # We patch ExternalVoice.stream_response to yield our tokens
    with patch('gaia_core.cognition.agent_core.build_from_packet', return_value=[]), \
         patch('gaia_core.cognition.agent_core.route_output', return_value={"council_messages": [], "response_to_user": "Hello world"}), \
         patch('gaia_core.cognition.agent_core.reflect_and_refine', return_value="Refined plan"), \
         patch('gaia_core.cognition.agent_core.run_cognitive_self_audit'), \
         patch('gaia_core.cognition.agent_core.ExternalVoice') as MockVoice:
        
        # Setup the voice mock to return a generator of tokens
        instance = MockVoice.return_value
        instance.stream_response.return_value = iter(["Hello", " world"])
        
        # run_turn is a generator
        events = list(mock_agent_core.run_turn(user_input, session_id))
        
        tokens = [e["value"] for e in events if e.get("type") == "token"]
        full_streamed_text = "".join(tokens)
        
        # ASSERTIONS
        # 1. We should have the streamed content
        assert "Hello world" in full_streamed_text
        
        # 2. DOUBLE-POSTING CHECK:
        # The core should NOT yield the consolidated "Hello world" again at the end
        # because it was already streamed bit-by-bit.
        assert full_streamed_text.count("Hello world") == 1
        
        # 3. Exactly one packet yielded at the end
        packets = [e for e in events if e.get("type") == "packet"]
        assert len(packets) == 1
        
        # 4. Flush event present
        flushes = [e for e in events if e.get("type") == "flush"]
        assert len(flushes) >= 1

def test_stream_integrity_flush_sequencing(mock_agent_core):
    """Verify that flush events appear correctly after content."""
    session_id = "test_flush"
    user_input = "Hello"
    
    mock_llm = MagicMock()
    mock_agent_core.model_pool.acquire_model_for_role.return_value = mock_llm
    
    with patch('gaia_core.cognition.agent_core.build_from_packet', return_value=[]), \
         patch('gaia_core.cognition.agent_core.route_output', return_value={"council_messages": [], "response_to_user": "Part 1"}), \
         patch('gaia_core.cognition.agent_core.reflect_and_refine', return_value="Refined plan"), \
         patch('gaia_core.cognition.agent_core.run_cognitive_self_audit'), \
         patch('gaia_core.cognition.agent_core.ExternalVoice') as MockVoice:
        
        instance = MockVoice.return_value
        instance.stream_response.return_value = iter(["Part 1"])
        
        events = list(mock_agent_core.run_turn(user_input, session_id))
        
        token_idx = next(i for i, e in enumerate(events) if e.get("type") == "token")
        flush_idx = next(i for i, e in enumerate(events) if e.get("type") == "flush")
        
        # Flush MUST come after tokens
        assert flush_idx > token_idx
