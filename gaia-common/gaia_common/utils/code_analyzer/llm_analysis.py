import logging
from typing import List, Dict
from app.cognition.agent_core import AgentCore
from app.config import Config

logger = logging.getLogger("GAIA.LLMAnalysis")

def summarize_chunks(chunks: List[Dict], llm, reflect: bool = True) -> str:
    """
    Generate a natural language summary of code chunks using the provided LLM.
    """
    try:
        cfg = Config()
        # This is a placeholder for a more robust AI Manager instance
        ai_manager = type('obj', (object,), {'model_pool': llm, 'config': cfg, 'active_persona': type('obj', (object,), {'get_full_instructions': lambda: ""})()})()
        agent_core = AgentCore(ai_manager)

        messages = [chunk["content"] for chunk in chunks if chunk["type"] in ("function", "class")]
        payload = "\n\n".join(messages).strip()

        if not payload:
            logger.warning("LLMAnalysis received an empty code chunk payload. Skipping summarization.")
            return "(Empty prompt — no summary generated.)"

        # We need to adapt the agent_core's run_turn method to this use case.
        # For now, we'll just pass the payload as the user input.
        response_generator = agent_core.run_turn(payload, session_id="code_summary")
        summary = "".join([event["value"] for event in response_generator if event["type"] == "token"])

        return summary

    except Exception as e:
        logger.error(f"❌ LLM summarization failed: {e}", exc_info=True)
        return "(Failed to summarize code.)"