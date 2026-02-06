# app/core/pipeline.py

"""
GAIA Cognitive Pipeline (Production)
- Stateless orchestration for GAIA's end-to-end reasoning cycle.
- Handles: intent detection, primitive routing, context/persona assembly, model-powered thought, observer streaming, and self-reflection.
- Extensible for any frontend (CLI, web, REST, agent).
"""

import logging
from typing import Any, Callable, Dict, Optional

# TODO: [GAIA-REFACTOR] intent_detection.py module not yet migrated.
# from app.cognition.nlu.intent_detection import detect_intent
from .manager import route_primitive, get_context
from gaia_core.behavior.persona_manager import load_persona
from gaia_core.behavior.persona_adapter import adapt_persona
# TODO: [GAIA-REFACTOR] agent_core.py is not yet refactored.
# from app.cognition.agent_core import AgentCore
# TODO: [GAIA-REFACTOR] stream_observer.py module not yet migrated.
# from app.utils.stream_observer import stream_observer
from gaia_core.cognition.self_reflection import run_self_reflection

logger = logging.getLogger("GAIA.Pipeline")

def gaia_pipeline(
    user_input: str,
    state: Any,
    config: Any,
    responder_name: str = "Prime",
    stream_output: bool = False,
    interrupt_check: Optional[Callable[[], bool]] = None,
    stream_callback: Optional[Callable[[str], None]] = None,
    observer_callback: Optional[Callable[[str, Dict], str]] = None,
    **kwargs
) -> str:
    """
    Core GAIA pipeline: runs full cognitive cycle and returns model output (optionally streamed).

    Args:
        user_input (str): The user's input or message.
        state (GAIAState): Current GAIA state (memory, persona, model_pool, etc.).
        config (Config): Loaded GAIA config object.
        responder_name (str): Model name to use for main response ("Prime", etc.).
        stream_output (bool): Whether to stream output via callback.
        interrupt_check (callable): Function to check for interrupt condition (optional).
        stream_callback (callable): Token callback for streaming output (optional).
        observer_callback (callable): Observer/interrupter callback (optional).
        **kwargs: Any extra pipeline options.

    Returns:
        str: The generated response (full or last chunk if streaming).
    """
    logger.info("=== GAIA Pipeline: Input received ===")
    state.memory.append({"user": user_input})
    persona_data = load_persona(state.persona)
    context = get_context(state)
    context = adapt_persona(context, persona_data)

    # === Intent Detection & Primitive Routing ===
    # TODO: [GAIA-REFACTOR] intent_detection.py module not yet migrated.
    # intent = detect_intent(
    #     user_input,
    #     config,
    #     lite_llm=state.model_pool.get_model_for_role("lite"),
    #     full_llm=state.model_pool.get_model_for_role("prime")
    # )
    intent = None # Placeholder

    # Handle core primitives before monologue/LLM step
    primitive_result = route_primitive(intent, user_input)
    if primitive_result is not None:
        logger.info(f"[Primitive Routed]: {intent}")
        state.memory[-1]["gaia"] = primitive_result
        return primitive_result

    # === Main Model Pipeline ===
    logger.info("🧠 [Pipeline] Running inner monologue (model-powered)...")
    # TODO: [GAIA-REFACTOR] agent_core.py is not yet refactored.
    # monologue = run_turn(
    #     user_input,
    #     config=config,
    #     llm=state.model_pool.get_model_for_role(responder_name),
    #     stream_output=stream_output,
    #     log_tokens=True,
    #     interrupt_check=interrupt_check,
    #     stream_callback=stream_callback,
    # )
    monologue = "" # Placeholder
    context["monologue"] = monologue

    # Observer/streaming hook (can extend for council/observer in future)
    def default_observer(buffer, ctx=context):
        # TODO: [GAIA-REFACTOR] stream_observer.py module not yet migrated.
        # return stream_observer(buffer, ctx, lambda buf, ctx: "continue")
        return "continue"
    observer_fn = observer_callback or default_observer

    if stream_output and stream_callback:
        # If streaming, monologue is handled by stream_callback
        final_output = None  # Streaming already handled, nothing to return
    else:
        final_output = monologue

    # Self-reflection, append to memory
    logger.info("🔄 [Pipeline] Running self-reflection...")
    reflection = run_self_reflection(context, final_output or monologue)
    if reflection:
        state.memory[-1]["reflection"] = reflection

    state.memory[-1]["gaia"] = final_output or monologue
    logger.info("=== GAIA Pipeline: Response complete ===")
    return final_output or monologue
