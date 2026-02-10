"""
Output Router - Central hub for parsing and dispatching all directives from LLM output.

Handles:
- Parsing LLM output into CognitionPacket structures
- Safety gate checking before execution
- Sidecar action dispatch
- Destination routing
"""
import json
import logging
import re
from typing import Dict, Any, Optional

# [GCP v0.3] Import new packet structure and safety gate
from gaia_common.protocols import CognitionPacket, PacketState, OutputDestination
from gaia_common.utils.packet_utils import is_execution_safe

# TODO: [GAIA-REFACTOR] thought_seed.py module not yet migrated.
# from gaia_core.cognition.thought_seed import save_thought_seed
def save_thought_seed(*args, **kwargs):
    """Placeholder until thought_seed module is migrated."""
    pass

from gaia_core.utils.mcp_client import dispatch_sidecar_actions

logger = logging.getLogger("GAIA.OutputRouter")


def _strip_think_tags_robust(text: str) -> str:
    """
    Robustly strip <think>...</think> and <thinking>...</thinking> blocks
    from model output.

    Handles edge cases that the basic regex misses:
    - Properly closed tags: <think>content</think>
    - Unclosed tags: <think>content (no closing tag)
    - Multiple/nested tags
    - Truncated closing tags: </think (missing >)
    - Both <think> and <thinking> tags.
    - Reflection/reasoning blocks with other tag formats.

    Returns the content AFTER all think blocks, or empty string if
    the entire response is inside think tags.
    """
    if not text:
        return text

    result = text

    # Regex to match both <think>...</think> and <thinking>...</thinking> blocks
    think_pattern = re.compile(r'<(?:think|thinking)>.*?</(?:think|thinking)>\s*', re.DOTALL)
    result = think_pattern.sub('', result)

    # Handle UNCLOSED opening tags: <think>content... (to end of string)
    # This catches cases where the model starts thinking but never closes
    unclosed_pattern = re.compile(r'<(?:think|thinking)>.*$', re.DOTALL)
    result = unclosed_pattern.sub('', result)

    # Handle truncated/malformed tags
    result = re.sub(r'</?(?:think|thinking)[^>]*>', '', result)

    # Strip other common reflection tags that shouldn't be user-facing
    # These include <reflection>, <reasoning>, <internal>, <scratchpad>, etc.
    other_tags = ['reflection', 'reasoning', 'internal', 'scratchpad', 'planning', 'analysis']
    for tag in other_tags:
        # Closed tags
        result = re.sub(rf'<{tag}>.*?</{tag}>\s*', '', result, flags=re.DOTALL | re.IGNORECASE)
        # Unclosed tags
        result = re.sub(rf'<{tag}>.*$', '', result, flags=re.DOTALL | re.IGNORECASE)
        # Malformed tags
        result = re.sub(rf'</?{tag}[^>]*>', '', result, flags=re.IGNORECASE)

    # If we're left with just whitespace, return empty
    return result.strip()

# Lazy import to avoid circular dependencies
_destination_registry = None

def _get_destination_registry():
    """Lazy-load the destination registry to avoid circular imports."""
    global _destination_registry
    if _destination_registry is None:
        try:
            # TODO: [GAIA-REFACTOR] destination_registry.py module not yet migrated.
            # from gaia_core.utils.destination_registry import get_registry
            # _destination_registry = get_registry()
            _destination_registry = None  # Placeholder
        except Exception as e:
            logger.warning(f"Could not load destination registry: {e}")
    return _destination_registry

def route_output(response_text: str, packet: CognitionPacket, ai_manager, session_id: str, destination: str) -> Dict[str, Any]:
    """
    The central hub for parsing and dispatching all directives from LLM output for v0.3 packets.
    It relies on the structured `sidecar_actions` field and the `is_execution_safe` gate.
    Legacy text parsing is minimized.
    """
    config = ai_manager.config
    execution_results = []
    response_to_user = ""

    # The LLM's raw output might contain a candidate response and proposed sidecar actions.
    # We assume a previous step has already parsed this text and populated the packet's
    # `response.candidate` and `response.sidecar_actions` fields.
    # For now, we'll do a simple parse here.
    _parse_llm_output_into_packet(response_text, packet)

    # 1. Check for authoritative blocks from the observer or other safety systems.
    if packet.status.state == PacketState.ABORTED:
        logger.warning(f"Routing aborted; packet state is {packet.status.state}. Reason: {packet.status.next_steps}")
        return {
            "response_to_user": "My apologies, but I cannot proceed. The current operation was aborted.",
            "execution_results": []
        }

    # 2. Check for actions and run safety gate before execution.
    if packet.response.sidecar_actions:
        logger.info(f"Packet contains {len(packet.response.sidecar_actions)} sidecar actions to evaluate.")
        if is_execution_safe(packet):
            logger.info("Safety gate passed. Proceeding with execution.")
            # The executor should be responsible for running the actions.
            # This is a conceptual change from the old router.
            execution_results = dispatch_sidecar_actions(packet, config)
        else:
            logger.warning("Safety gate FAILED. Execution of sidecar actions is denied.")
            denied_actions = ", ".join([action.action_type for action in packet.response.sidecar_actions])
            packet.status.state = PacketState.ABORTED
            packet.status.next_steps.append(f"Execution denied for actions: {denied_actions}")
            response_to_user = "I understand the proposed next steps, but I cannot execute them without proper authorization. The operation has been halted."
            return {"response_to_user": response_to_user, "execution_results": []}

    # 3. Determine the final response to the user.
    # The primary source should be the `candidate` field in the packet.
    if packet.response.candidate:
        response_to_user = packet.response.candidate
    elif execution_results:
        # If actions were taken but no explicit RESPONSE, generate a summary.
        actions_summary = []
        for res in execution_results:
            if res.get("ok"):
                actions_summary.append(f"Successfully executed: {res.get('op')}")
        if actions_summary:
            response_to_user = "I have completed the requested actions:\n- " + "\n- ".join(actions_summary)
        else:
            response_to_user = "I attempted to perform the requested actions but encountered an issue."
    else:
        # Fallback if no candidate and no actions.
        response_to_user = "I have processed the request."

    # Handle THOUGHT_SEED as a special legacy case for now.
    thought_seed_match = re.search(r"THOUGHT_SEED:\s*(.*)", response_text, re.DOTALL)
    if thought_seed_match:
        seed_text = thought_seed_match.group(1).strip()
        logger.info(f"Routing THOUGHT_SEED directive: {seed_text[:80]}...")
        save_thought_seed(seed_text, packet, config)

    # Strip any <think> tags from the response before sending to user
    # This handles cases where the model's reasoning blocks weren't properly stripped
    original_len = len(response_to_user)
    response_to_user = _strip_think_tags_robust(response_to_user)
    if len(response_to_user) != original_len:
        logger.info(f"Stripped think tags from response: {original_len} -> {len(response_to_user)} chars")

    # If stripping left us with nothing, provide a fallback
    if not response_to_user.strip():
        logger.warning("Response was empty after stripping think tags - model may have generated only reasoning")
        response_to_user = "I apologize, but I encountered an issue generating a response. Please try rephrasing your question."

    # 4. Route to destinations via the spinal column
    destination_results = {}
    try:
        registry = _get_destination_registry()
        if registry:
            # Map legacy destination string to OutputDestination enum
            override_dest = _legacy_destination_to_enum(destination)
            destination_results = registry.route(response_to_user, packet, override_dest)
            logger.info(f"Routed to destinations: {destination_results}")
    except Exception as e:
        logger.warning(f"Destination routing failed: {e}")

    return {
        "response_to_user": response_to_user,
        "execution_results": execution_results,
        "destination_results": destination_results
    }


def _legacy_destination_to_enum(destination: str) -> Optional[OutputDestination]:
    """Convert legacy destination strings to OutputDestination enum."""
    if not destination:
        return None

    mapping = {
        "cli_chat": OutputDestination.CLI,
        "cli": OutputDestination.CLI,
        "web": OutputDestination.WEB,
        "web_chat": OutputDestination.WEB,
        "discord": OutputDestination.DISCORD,
        "api": OutputDestination.API,
        "log": OutputDestination.LOG,
    }
    return mapping.get(destination.lower())

def _strip_gcp_metadata(text: str) -> str:
    """
    Strip GAIA Cognition Packet metadata that the LLM may have echoed back.
    This includes [HEADER], [ROUTING], [GOVERNANCE], [METRICS], [STATUS], [REASONING],
    [CONTEXT], [INTENT], [CONTENT], [MODEL] sections and their contents.
    """
    # Remove GCP section blocks: [SECTION_NAME] followed by indented content until next section or end
    # Pattern matches: [SECTION] followed by lines that start with whitespace or specific markers
    gcp_sections = [
        'HEADER', 'ROUTING', 'MODEL', 'CONTEXT', 'INTENT', 'CONTENT',
        'REASONING', 'GOVERNANCE', 'METRICS', 'STATUS', 'RESPONSE'
    ]

    # Remove entire GCP sections (e.g., [GOVERNANCE]\n  execution_allowed: False\n  dry_run: True)
    for section in gcp_sections:
        # Match [SECTION] followed by indented lines until next [SECTION] or end of text
        pattern = rf'\[{section}\]\s*\n(?:[ \t]+[^\n]*\n?)*'
        text = re.sub(pattern, '', text, flags=re.IGNORECASE)

    # Remove "GAIA COGNITION PACKET" header line
    text = re.sub(r'^GAIA COGNITION PACKET\s*\n?', '', text, flags=re.MULTILINE | re.IGNORECASE)

    # Remove inline GCP-style metadata (e.g., "confidence=0.5", "step=initial_plan")
    # These appear when the LLM echoes back reflection log entries
    text = re.sub(r',?\s*confidence=[\d.]+', '', text)
    text = re.sub(r'-\s*step=\w+,\s*summary=', '', text)

    # Remove repeated <think> blocks that got concatenated
    # Keep only the first substantive response after </think>
    think_end_match = re.search(r'</think>\s*\n?\s*(?!</think>)', text)
    if think_end_match:
        # Find content after the last </think> tag
        last_think_end = text.rfind('</think>')
        if last_think_end != -1:
            text = text[last_think_end + len('</think>'):].strip()

    return text.strip()


def _parse_llm_output_into_packet(response_text: str, packet: CognitionPacket):
    """
    A temporary parser to populate the v0.3 packet from raw LLM text.
    In a mature system, the LLM would ideally return structured JSON.
    This function handles RESPONSE: and a simplified form of sidecar actions.
    """
    from gaia_common.protocols.cognition_packet import SidecarAction

    # First, strip any GCP metadata the LLM may have echoed back
    cleaned_response = _strip_gcp_metadata(response_text)

    # Extract RESPONSE
    response_match = re.search(r"RESPONSE:\s*(.*)", cleaned_response, re.DOTALL)
    if response_match:
        packet.response.candidate = response_match.group(1).strip()
    else:
        # If no explicit RESPONSE, use the whole text, minus directives.
        clean_text = re.sub(r"^(THOUGHT_SEED:|EXECUTE:|RESPONSE:|<<<|>>>).*\n?", "", cleaned_response, flags=re.MULTILINE).strip()
        packet.response.candidate = clean_text

    # Extract EXECUTE directives and convert to SidecarActions.
    # Supports two formats:
    #   Structured: EXECUTE: write_file {"path": "/knowledge/doc.txt", "content": "..."}
    #   Legacy:     EXECUTE: run_shell ls -la /knowledge
    execute_matches = re.findall(r"EXECUTE:\s*(.*)", response_text)
    for cmd_str in execute_matches:
        parts = cmd_str.strip().split(None, 1)  # Split into tool_name + rest
        action_type = parts[0]
        raw_args = parts[1] if len(parts) > 1 else ""

        # Try JSON params first (structured format)
        params = {}
        if raw_args.lstrip().startswith("{"):
            try:
                params = json.loads(raw_args)
            except json.JSONDecodeError:
                params = {"command": raw_args}
        else:
            params = {"command": raw_args} if raw_args else {}

        packet.response.sidecar_actions.append(SidecarAction(action_type=action_type, params=params))
