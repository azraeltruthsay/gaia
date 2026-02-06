# app/utils/context.py
# Utility: get_context_for_task(task_type, config=None)
# Generates lightweight runtime context strings for LLM prompts.

import uuid
from datetime import datetime

def get_context_for_task(task_type: str, config=None) -> str:
    """
    Builds a basic runtime context string for LLM prompts.
    Optionally consults Config.task_context_templates for custom behavior.
    """
    now = datetime.utcnow().isoformat() + "Z"
    context_id = str(uuid.uuid4())

    # Check if config has predefined context templates
    if config and hasattr(config, "task_context_templates"):
        template = config.task_context_templates.get(task_type)
        if callable(template):
            return template()
        elif isinstance(template, str):
            return template.format(
                task=task_type,
                timestamp=now,
                uuid=context_id
            )

    # Fallback default context
    return f"[AutoContext] task={task_type} | time={now} | uuid={context_id}"
