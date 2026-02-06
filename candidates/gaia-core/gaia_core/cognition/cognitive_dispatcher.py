import json
import logging
import re
from gaia_core.models.model_pool import model_pool
from gaia_core.config import get_config
from gaia_common.protocols import CognitionPacket

# Get constants from config
_config = get_config()
constants = getattr(_config, 'constants', {})

logger = logging.getLogger("GAIA.CognitiveDispatcher")

def dispatch(packet) -> dict:
    """
    Analyzes the prompt and dispatches it to the appropriate model with a dynamic context window.
    Acquires the selected model and returns it. The caller is responsible for releasing the model.

    Returns:
        A dictionary containing the selected model name, the acquired model instance, token budget, and persona instructions.
    """
    prompt = packet.prompt
    persona_instructions = packet.contextual_instructions
    lite_model = model_pool.acquire_model_for_role("lite")
    if not lite_model:
        logger.error("Could not acquire Lite model for initial analysis.")
        return None

    try:
        # Improved, more explicit analysis prompt to encourage valid JSON output
        analysis_prompt = (
            "You are a JSON-only API. Your only purpose is to analyze a user's prompt and return a single, valid JSON object. "
            "Do not include any text, explanation, or commentary before or after the JSON object. "
            "The JSON object must have two keys: 'complexity' (string) and 'required_context' (string). "
            "The value for 'complexity' must be one of: 'simple', 'moderate', or 'complex'. "
            "The value for 'required_context' must be one of: 'minimal', 'medium', or 'full'. "
            "Example response: {\"complexity\": \"moderate\", \"required_context\": \"medium\"}. "
            f"Analyze the following prompt:\n\n---\n\n{prompt}"
        )
        
        analysis_response_raw = lite_model.create_completion(prompt=analysis_prompt, max_tokens=128)
        analysis_response_text = analysis_response_raw["choices"][0]["text"].strip()

        try:
            # Enhanced JSON extraction to find the first valid JSON object
            json_match = re.search(r"(?s)\{.*?\}", analysis_response_text)
            if json_match:
                analysis = json.loads(json_match.group(0))
            else:
                raise json.JSONDecodeError("No JSON object found in response", analysis_response_text, 0)
        except (json.JSONDecodeError, IndexError) as e:
            logger.warning(f"Could not decode JSON from lite model. Error: {e}. Response: {analysis_response_text}")
            analysis = {"complexity": "simple", "required_context": "minimal"}

        complexity = analysis.get("complexity", "simple")
        required_context = analysis.get("required_context", "minimal")

        token_budgets = constants.get("TOKEN_BUDGETS", {
            "minimal": 1024,
            "medium": 2048,
            "full": 4096
        })
        token_budget = token_budgets.get(required_context, 1024)

        selected_model_name = "lite"
        selected_model = lite_model

        if complexity != "simple":
            # Complex task: switch to prime model
            prime_model = model_pool.acquire_model_for_role("prime")
            if prime_model:
                # Successfully acquired prime, so we can release lite
                model_pool.release_model_for_role("lite")
                selected_model_name = "prime"
                selected_model = prime_model
            else:
                # Failed to get prime, so we'll just use the lite model
                logger.warning("Could not acquire Prime model, falling back to Lite model for complex task.")
        
        return {
            "model_name": selected_model_name,
            "model": selected_model,
            "token_budget": token_budget,
            "persona_instructions": persona_instructions
        }

    except Exception as e:
        logger.error(f"Error in cognitive dispatch: {e}", exc_info=True)
        # Release any models we might have acquired before the error
        model_pool.release_model_for_role("lite")
        model_pool.release_model_for_role("prime") # It might have been acquired
        return None

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
