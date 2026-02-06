# === observer_manager.py ===

import logging

# Lazy imports to avoid circular dependencies between gaia-common and gaia-core
def _get_model_pool():
    """Lazy import of model_pool from gaia_core."""
    try:
        from gaia_core.models.model_pool import get_model_pool
        return get_model_pool()
    except ImportError:
        return None

def _get_stream_observer():
    """Lazy import of StreamObserver from gaia_core."""
    try:
        from gaia_core.utils.stream_observer import StreamObserver
        return StreamObserver
    except ImportError:
        return None

from gaia_common.utils.role_manager import RoleManager

logger = logging.getLogger("GAIA.ObserverManager")

def assign_observer(current_responder, stream_channel="active_response"):
    # Lazily obtain the model_pool singleton to avoid import-time heavy imports
    mp = _get_model_pool()
    StreamObserver = _get_stream_observer()
    observer_name = mp.get_idle_model(exclude=[current_responder]) if mp else None
    if not observer_name:
        logger.warning(f"⚠️ No idle observer available. {current_responder} will self-monitor.")
        # Self-observing StreamObserver, no LLM (no-ops are handled in StreamObserver)
        if StreamObserver is None:
            return None
        return StreamObserver(llm=None, name=current_responder)

    observer_llm = mp.get_model_for_role(observer_name) if observer_name and mp else None
    try:
        if mp is not None:
            mp.set_status(observer_name, "observing")
    except Exception:
        logger.exception("Failed to set observer status")

    try:
        RoleManager.add_observer(observer_name)
    except Exception:
        logger.exception("RoleManager.add_observer failed")

    logger.info(f"👂 Assigned observer: {observer_name}")
    return StreamObserver(llm=observer_llm, name=observer_name)


def default_interrupt_eval(model, stream_text):
    prompt = f"You are monitoring output. Does this contain hallucinations, contradictions, or tone drift?\n\n{stream_text}\n\nReply with:\n- CONTINUE\n- INTERRUPT: <reason>"
    result = model.create_completion(prompt=prompt, max_tokens=64)
    return result["choices"][0]["text"].strip()


def default_interrupt_handler(reason):
    print(f"🔔 Observer Interrupt: {reason}")
