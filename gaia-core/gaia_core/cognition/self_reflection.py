"""
Self Reflection Processor (model-powered, robust pipeline)
- Calls LLM for post-generation analysis and hallucination/error detection.
- Integrates config-driven safety and can fallback to rule-based checks.
- Used for post-response reflection in manager.py or external_voice.py pipeline.
"""

import logging
import time
import json
import re
import copy
from gaia_core.config import Config, get_config
from gaia_core.memory.conversation.summarizer import ConversationSummarizer
from gaia_core.utils.gaia_rescue_helper import sketch, show_sketchpad, clear_sketchpad
from gaia_common.utils.thoughtstream import write as ts_write

# [GCP v0.3] Import the new packet structure
from gaia_common.protocols.cognition_packet import CognitionPacket, Persona, PersonaRole, ReflectionLog
from gaia_core.utils.prompt_builder import build_from_packet, count_tokens
from gaia_core.utils.packet_builder import build_packet_snapshot

logger = logging.getLogger("GAIA.SelfReflection")
# File logging setup for self-reflection module
import os
log_dir = os.path.join(os.getcwd(), "logs")
os.makedirs(log_dir, exist_ok=True)
file_handler = logging.FileHandler(os.path.join(log_dir, "self_reflection.log"), mode="a")
file_handler.setLevel(logging.INFO)
file_handler.setFormatter(logging.Formatter("%(asctime)s %(name)s %(levelname)s %(message)s"))
logger.addHandler(file_handler)
logger.setLevel(logging.INFO)
logger.propagate = False

def reflect_and_refine(packet: CognitionPacket, output: str, config, llm, ethical_sentinel) -> str:
    """
    Iteratively reflect on the given output using the LLM and ethical sentinel,
    summarizing large outputs and respecting token budgets.
    Returns the final refined thought or reflection.
    """
    # --- Intent detection step ---
    try:
        if hasattr(config, 'model_pool') and hasattr(config.model_pool, 'get_model_for_role'):
            intent_model = config.model_pool.get_model_for_role('intent')
            if intent_model:
                intent_result = intent_model.create_chat_completion(
                    messages=[{'role': 'user', 'content': packet.content.original_prompt}],
                    max_tokens=32,
                    temperature=0.0
                )
                packet.context.intent = intent_result['choices'][0]['message']['content'].strip()
                logger.info(f"SelfReflection: intent detected: {packet.context.intent}")
    except Exception as e:
        logger.warning(f"SelfReflection: intent detection failed: {e}")

    # -- Reflection budget handling --
    MAX_REFLECTION_TOKENS = getattr(config, 'max_reflection_tokens', 500)
    output_tokens = count_tokens(output)
    if output_tokens > MAX_REFLECTION_TOKENS:
        logger.info(f"SelfReflection: summarizing output of {output_tokens} tokens (threshold {MAX_REFLECTION_TOKENS})")
    # Pass the configured llm (if provided) to the summarizer; do not pass the Config object as llm
    summarizer = ConversationSummarizer(llm=llm)
    try:
        # Build a reflection packet snapshot to ground the summarization
        try:
            reflection_packet_snapshot = build_packet_snapshot(session_id=getattr(config, 'session_id', 'reflection'), persona_id='Reflector', original_prompt=output, history=[{'role':'assistant','content':output}])
        except Exception:
            reflection_packet_snapshot = None
        summary = summarizer.generate_summary([{'role': 'assistant', 'content': output}], packet=reflection_packet_snapshot)
        output = summary
    except Exception as e:
        logger.error(f"SelfReflection: summarization failed: {e}", exc_info=True)

    # -- Build reflection prompt using a temporary, reflection-focused packet --
    reflection_packet = copy.deepcopy(packet)
    reflection_packet.header.persona = Persona(
        identity_id="Prime",
        persona_id="Reflector",
        role=PersonaRole.ANALYST, # Using Analyst role for critical thinking
        tone_hint="Analytical, critical, and focused on providing actionable feedback."
    )
    reflection_packet.content.original_prompt = output

    messages = build_from_packet(reflection_packet, task_instruction_key="refinement")

    final_thought = ""
    iterations = getattr(config, 'max_reflection_iterations', 3)
    threshold = getattr(config, 'reflection_threshold', 0.9)
    max_tokens = getattr(config, 'reflection_max_tokens', 256)


    # Iterative observation/self-reflection loop (configurable)
    final_thought = ""
    for i in range(iterations):
        t_iter_start = time.perf_counter()
        try:
            raw = llm.create_chat_completion(
                messages=messages,
                max_tokens=max_tokens,
                temperature=getattr(config, 'reflection_temperature', 0.3)
            )
        except Exception as e:
            logger.error(f"SelfReflection: LLM call failed on iteration {i}: {e}", exc_info=True)
            raw = None
        t_iter_end = time.perf_counter()
        logger.info(f"SelfReflection: LLM call iteration {i} took {t_iter_end - t_iter_start:.2f}s")

        text = ""
        if isinstance(raw, dict):
            choices = raw.get("choices", [])
            if choices and isinstance(choices[0], dict):
                message_content = choices[0].get("message", {}).get("content")
                if message_content:
                    text = message_content
        text = (text or "").strip()
        logger.info(f"SelfReflection: raw response (iter {i}): {text[:200]}")

        # Parse confidence
        score_match = re.search(r"Confidence:\s*([0-1]\.\d*)", text, re.IGNORECASE)
        score = float(score_match.group(1)) if score_match else 0.5

        logger.info(f"SelfReflection: confidence (iter {i}) {score}")
        sketch(f"Reflection iter {i} (Score: {score:.2f})", text)

        # Append to packet reasoning log so chain-of-thought is persisted
        try:
            packet.reasoning.reflection_log.append(ReflectionLog(step=f"reflection_iter_{i}", summary=text, confidence=score))
        except Exception:
            logger.debug("SelfReflection: failed to append reflection log to packet")

        # Telemetry: write a compact reflection event to the thoughtstream
        try:
            model_name = None
            try:
                model_name = getattr(llm, 'name', None) or llm.__class__.__name__
            except Exception:
                model_name = None
            ts_write({
                "type": "reflection_iteration",
                "packet_id": getattr(packet.header, 'packet_id', None),
                "iteration": i,
                "confidence": score,
                "summary": (text or "")[:500],
                "model": model_name,
            }, getattr(packet.header, 'session_id', getattr(packet, 'session_id', 'unknown')))
        except Exception:
            logger.debug("SelfReflection: failed to write reflection telemetry", exc_info=True)

        final_thought = text

        # Stop early if confidence threshold reached
        try:
            if score >= threshold:
                logger.info(f"SelfReflection: confidence threshold reached (iter {i})")
                break
        except Exception:
            pass

        # Stop early if the model has produced a concrete plan
        if "PLAN:" in final_thought:
            logger.info(f"SelfReflection: PLAN found, breaking reflection loop (iter {i})")
            break

    logger.info(f"SelfReflection: completed up to {iterations} iterations; final confidence approx {score}")

    # -- Final safety check --
    try:
        # Call EthicalSentinel with expected parameter names; it may call the identity guardian internally
        safe = True
        if ethical_sentinel:
            try:
                # Prepare persona_traits (best-effort): look for explicit traits or fallback to empty dict
                persona_traits = {}
                try:
                    persona_traits = getattr(packet.header.persona, 'traits', {}) or {}
                except Exception:
                    persona_traits = {}

                # Handle cases where reflection_log entries are deserialized as dicts instead of objects
                instructions = [
                    r.get("summary") if isinstance(r, dict) else getattr(r, "summary", "")
                    for r in packet.reasoning.reflection_log
                ] or []
                safe = ethical_sentinel.run_full_safety_check(
                    persona_traits=persona_traits,
                    instructions=instructions,
                    prompt=final_thought,
                )
            except Exception as e:
                logger.error(f"SelfReflection: safety check error: {e}", exc_info=True)
                safe = False

        if safe:
            logger.info("SelfReflection: passed final safety check")
        else:
            logger.warning("SelfReflection: final safety check failed")
    except Exception as e:
        logger.error(f"SelfReflection: unexpected safety check error: {e}", exc_info=True)

    clear_sketchpad()

    if not isinstance(final_thought, str):
        final_thought = ""

    plan_match = re.search(r"PLAN:(.*)", final_thought, re.DOTALL)
    if plan_match:
        return plan_match.group(1).strip()
    else:
        return final_thought


# This function is largely superseded by the logic in AgentCore but is kept for potential standalone use.
def run_self_reflection(packet: CognitionPacket, output: str, config=None, llm=None):
    """
    Main entrypoint for model-powered self-reflection.
    """
    if config is None:
        config = Config()
    if llm is None and hasattr(config, "model_pool"):
        llm = config.model_pool.get_model_for_role("prime")

    MAX_REFLECTION_TOKENS = getattr(config, 'max_reflection_tokens', 500)
    output_tokens = count_tokens(output)
    if output_tokens > MAX_REFLECTION_TOKENS:
        logger.info(f"SelfReflection: summarizing output of {output_tokens} tokens")
    summarizer = ConversationSummarizer(llm=llm)
    try:
        summary = summarizer.generate_summary([{'role': 'assistant', 'content': output}])
        output = summary
    except Exception as e:
        logger.error(f"SelfReflection: summarization failed (fallback): {e}", exc_info=True)

    # Build prompt using a temporary packet
    reflection_packet = copy.deepcopy(packet)
    reflection_packet.header.persona.role = PersonaRole.ANALYST
    reflection_packet.content.original_prompt = output
    messages = build_from_packet(reflection_packet, task_instruction_key="self_reflection_task")

    if llm:
        try:
            result = llm.create_chat_completion(
                messages=messages,
                temperature=0.3,
                top_p=0.7,
                max_tokens=256,
                stream=False
            )
            reflection = result["choices"][0]["message"]["content"].strip()
            logger.info(f"🪞 Model-powered self-reflection: {reflection[:100]}...")
            return reflection
        except Exception as e:
            logger.error(f"Self-reflection LLM error: {e}")

    # Rule-based fallback
    # ... (existing fallback logic) ...
    return None