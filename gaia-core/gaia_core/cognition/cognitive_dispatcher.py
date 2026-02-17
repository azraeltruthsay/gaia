import logging
from gaia_common.protocols import CognitionPacket

logger = logging.getLogger("GAIA.CognitiveDispatcher")

def process_execution_results(execution_results, session_manager, session_id, packet: CognitionPacket):
    if not execution_results:
        return

    for result in execution_results:
        # FIX: The 'result' dict from the executor has 'op', 'label', 'ok', 'detail', 'raw' keys, not 'command' or 'result'.
        # This change correctly and safely extracts the information.
        command = result.get("raw") or f"ai.{result.get('op')}('{result.get('label')}')"
        outcome = {
            "returncode": 0 if result.get("ok") else 1,
            "stdout": result.get("detail", "") if result.get("ok") else "",
            "stderr": "" if result.get("ok") else result.get("detail", "Execution failed")
        }

        # Format the result for the conversation history
        formatted_result = f"Executed command: {command}\n"
        formatted_result += f"Exit Code: {outcome['returncode']}\n"
        if outcome['stdout']:
            formatted_result += f"Output:\n{outcome['stdout']}\n"
        if outcome['stderr']:
            formatted_result += f"Errors:\n{outcome['stderr']}\n"

        packet.append_thought(f"Execution Result:\n{formatted_result}")
        session_manager.add_message(session_id, "assistant", formatted_result)
