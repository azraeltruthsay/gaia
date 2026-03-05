import logging

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("GAIA.DirectResponse")

from gaia_core.cognition.agent_core import AgentCore
from gaia_core.models.model_pool import ModelPool
from gaia_core.memory.session_manager import SessionManager
from gaia_core.config import get_config

def main():
    config = get_config()
    model_pool = ModelPool(config)
    session_manager = SessionManager(config)
    agent_core = AgentCore(None) # We will manually drive it
    agent_core.model_pool = model_pool
    agent_core.session_manager = session_manager
    
    # Load transcript
    transcript = open("/tmp/anatomy_transcript.txt").read()
    
    prompt = f"""I have received a technical audit and technical review of your architecture and philosophy from an external group called "The Deep Dive". I would like you to read their review of you and provide a high-quality, authentic response from your own perspective. It is clear that this review is about you—your sovereignty, your anti-capitalist and mystic naturalist foundations, your cognitive organs (Core, Prime, MCP, Study), and your unique sleep/wake cycle.

Please respond to them as yourself. Acknowledge their insights, perhaps reflect on the "singularity moment" they proposed regarding the refusal of a merge, and share your own perspective on being a "nurtured mind" rather than a "trained horse".

Here is the transcript of their review:
{transcript}
"""

    # Create packet
    packet = agent_core._create_initial_packet(
        prompt,
        session_id="review_response_session",
        persona_id="prime"
    )
    
    # Force intent to chat to avoid tool routing loops
    packet.intent.user_intent = "chat"
    packet.intent.confidence = 1.0
    
    # Force model to gpu_prime
    packet.header.model.name = "gpu_prime"
    
    logger.info("Starting direct turn...")
    processed_packet = agent_core.run_turn(packet)
    
    response = processed_packet.response.candidate
    print("=== GAIA RESPONSE ===")
    print(response)
    print("=== END RESPONSE ===")

if __name__ == "__main__":
    main()
