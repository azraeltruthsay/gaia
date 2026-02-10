"""
GAIA Core Manager
- Orchestrates prompt/context assembly, persona adaptation, cognitive pipeline (model-powered inner monologue, streaming, self-reflection, observer), and council/collaborative model logic.
- Integrates all mature config, persona, and logging features.
"""
import logging
from .primitives import read, vector_query, shell
from gaia_core.config import Config
# TODO: [GAIA-REFACTOR] intent_detection.py module not yet migrated.
# from app.cognition.nlu.intent_detection import detect_intent
# Only import AgentCore inside functions that actually use it, not at the top level
# TODO: [GAIA-REFACTOR] These imports are used in planned pipeline stages
# from gaia_core.cognition.self_reflection import run_self_reflection
# from gaia_core.behavior.persona_manager import load_persona
# from gaia_core.utils.prompt_builder import build_prompt
# TODO: [GAIA-REFACTOR] stream_bus.py module not yet migrated.
# from app.utils.stream_bus import publish_stream  # Streaming, observer support
# TODO: [GAIA-REFACTOR] stream_observer.py module not yet migrated.
# from app.utils.stream_observer import StreamObserver


logger = logging.getLogger("GAIA.Manager")

class GAIAState:
    def __init__(self, config=None):
        self.current_project = "default"
        self.current_session = "main"
        self.current_user = "local"
        self.persona = "default"
        self.instructions = ""
        self.memory = []
        # Model pool: model_name -> callable (Prime, Lite, etc.)
        self.model_pool = {
            "Prime": None,  # Inject real callable in app/bootstrap
            "Lite": None,
        }
        self.currently_busy = set()
        self.config = config or Config()

    def reset(self):
        self.memory = []
        logger.info("GAIA session memory reset.")

# REMOVED BY PATCH: state = GAIAState()

def get_context(state):
    """
    Assemble prompt context from all config, persona, and session state.
    Restores robust persona/context layering from legacy.
    """
    cfg = state.config  # now uses passed-in state
    identity = cfg.get("identity", "GAIA - Artisanal Intelligence")
    identity_intro = cfg.get("identity_intro", "")
    persona = state.persona
    instructions = state.instructions or cfg.get("persona_defaults", {}).get("instructions", "Assist with integrity and care.")
    primitives = cfg.get("primitives", [])
    constraints = cfg.get("reflection_guidelines", [])
    temperature = cfg.get("temperature", 0.7)
    max_tokens = cfg.get("max_tokens", 512)
    top_p = cfg.get("top_p", 0.95)
    project = state.current_project
    session = state.current_session
    user = state.current_user
    history = state.memory[-8:]
    return {
        "identity": identity,
        "identity_intro": identity_intro,
        "persona": persona,
        "instructions": instructions,
        "primitives": primitives,
        "constraints": constraints,
        "temperature": temperature,
        "max_tokens": max_tokens,
        "top_p": top_p,
        "project": project,
        "session": session,
        "user": user,
        "history": history,
    }

def get_idle_observer(state, responder="Prime"):
    """
    Assign idle observer model; fallback to responder if all are busy.
    """
    available = [m for m in state.model_pool if m != responder and m not in state.currently_busy]
    return available[0] if available else responder

def route_primitive(intent, user_input):
    """
    Routes detected intent to the right primitive and returns result.
    Robust intent/primitive routing (legacy logic restored).
    """
    if intent == "read_file":
        import re
        match = re.search(r"([\w\-/]+\.[\w]+)", user_input)
        if match:
            filename = match.group(1)
            return read(filename)
        else:
            return "⚠️ No filename detected."
    elif intent == "write_file":
        return "Write functionality: please specify edit content."
    elif intent == "vector_query":
        return vector_query(user_input)
    elif intent == "shell":
        return shell(user_input)
    return None

#def chat_loop(input_fn=input, output_fn=print, state=None, responder_name="Prime", interrupt_check=None, stream_callback=None):
#    if state is None:
#        # Fallback to global 'ai' if present
#        try:
#            state = ai  # rescue shell style
#        except NameError:
#            raise RuntimeError("No GAIA state provided and no global 'ai' found.")
#    """
#    Main chat loop for GAIA core (fully restored, feature-rich).
#    Handles:
#      - persona/context loading
#      - intent detection and primitive routing (reflex/LLM)
#      - model-powered inner monologue (for non-reflex)
#      - streaming (with interrupt)
#      - self-reflection
#      - robust memory logging
#      - multi-model council hooks
#    """
#    output_fn("GAIA is ready. (type 'exit' to quit)")
#    if not state.model_pool.get("Prime"):
#        output_fn("❌ ERROR: No model loaded in model pool for 'Prime'. GAIA cannot operate.")
#        logger.error("No model loaded for 'Prime'. Exiting chat loop.")
#        return
#    while True:
#        user_input = input_fn("You> ")
#        if user_input.lower() in ("exit", "quit"):
#            output_fn("Exiting chat loop.")
#            break
#
#        state.memory.append({"user": user_input})
#        persona_data = load_persona(state.persona)
#        context = get_context(state)
#        context = adapt_persona(context, persona_data)
#
#        # === Intent Detection & Primitive Routing ===
#        intent = detect_intent(
#            user_input,
#            state.config,
#            lite_llm=state.model_pool.get("Lite"),
#            full_llm=state.model_pool.get("Prime")
#        )
#
#        # Reflexes
#        if intent == "exit":
#            output_fn("Exiting chat loop.")
#            break
#        elif intent == "help":
#            output_fn("Help: You can type a question, 'exit' to quit, or use a core primitive.")
#            continue
#        elif intent == "shell":
#            shell_result = shell(user_input)
#            output_fn(shell_result)
#            state.memory[-1]["gaia"] = shell_result
#            continue
#        primitive_result = route_primitive(intent, user_input)
#        if primitive_result is not None:
#            output_fn(primitive_result)
#            state.memory[-1]["gaia"] = primitive_result
#            continue
#
#        # === Main Cognitive Pipeline ===
#        logger.info("🧠 Running inner monologue (model-powered)...")
#        monologue = generate_inner_monologue(
#            user_input,
#            config=state.config,
#            llm=state.model_pool[responder_name],
#            stream_output=False,  # Streamable if needed
#            log_tokens=True,
#            interrupt_check=interrupt_check,
#            stream_callback=stream_callback
#        )
#        context["monologue"] = monologue
#        logger.info(f"🧠 Monologue generated: {monologue[:100]}...")
#
#        # Prompt Assembly (robust config, persona, context, monologue)
#        prompt = build_prompt(context)
#        responder_fn = state.model_pool.get(responder_name)
#        if responder_fn is None:
#            response = f"[Error: Model '{responder_name}' not available]"
#        else:
#            # LLM call (streaming supported, with observer/council hooks)
#            logger.info("🤖 Generating LLM response (streaming supported)...")
#            response = responder_fn(
#                prompt,
#                config=state.config,
#                stream_output=True,
#                interrupt_check=interrupt_check,
#                stream_callback=stream_callback
#            )
#
#        # Observer logic: assign observer, allow interrupt/feedback
#        observer = get_idle_observer(state, responder_name)
#        logger.info(f"👀 Assigned observer: {observer}")
#
#        def observer_fn(buffer, ctx=context):
#            # Can call any model (council) or rule-based observer
#            return stream_observer(buffer, ctx, lambda buf, ctx: "continue")
#
#        output_buffer = []
#        def streaming_output(chunk, end="", flush=True):
#            output_buffer.append(chunk)
#            output_fn(chunk, end=end, flush=flush)
#
#        final_output = publish_stream(
#            response,
#            streaming_output,
#            observer_fn
#        )
#
#        # Self-Reflection (model-powered, robust config-driven)
#        logger.info("🔄 Running self-reflection...")
#        reflection = run_self_reflection(context, final_output)
#        if reflection:
#            output_fn(reflection)
#            state.memory[-1]["reflection"] = reflection
#        state.memory[-1]["gaia"] = final_output
#
#def chat_loop_cli():
#    from .bootstrap import bootstrap_gaia
#    ai = bootstrap_gaia()
#    chat_loop(state=ai)
#
def load_models(config, model_pool_dict):
    """
    Loads models using the singleton ModelPool and maps them to model_pool dict for legacy compatibility.

    NOTE: Uses the singleton model_pool instance to avoid double-loading models.
    The models are loaded once and held in memory by the singleton.
    """
    from gaia_core.models.model_pool import model_pool as pool

    # Only call load_models if not already loaded (the pool has internal guard too)
    if not getattr(pool, '_models_loaded', False):
        pool.load_models()

    # Map ModelPool keys to GAIA's model_pool dict ('Prime', 'Lite') for legacy compatibility
    if pool.get('prime'):
        model_pool_dict["Prime"] = pool.get('prime')
    if pool.get('lite'):
        model_pool_dict["Lite"] = pool.get('lite')
