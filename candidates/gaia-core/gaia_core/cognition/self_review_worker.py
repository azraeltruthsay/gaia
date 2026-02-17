"""Self-review worker: reviews thought seeds and proposes dev_matrix updates.

This worker runs in proposal-only mode: it will create a pending MCP action to
update `knowledge/system_reference/dev_matrix.json` and return the approval challenge
to the operator. The operator must respond with the reversed 5-char code to approve.
"""
from __future__ import annotations
import json
import logging
from datetime import datetime, timezone
from gaia_core.cognition import thought_seed
from gaia_core.config import Config, get_config
from gaia_core.utils import mcp_client
from pathlib import Path
from gaia_core.cognition.nlu.intent_service import detect_intent


def _get_model_pool():
    """Lazily import and return the shared model_pool singleton.

    Importing `model_pool` at module import time can create circular
    imports. Import it inside functions that need it instead.
    """
    try:
        from gaia_core.models.model_pool import model_pool as _mp
        return _mp
    except Exception:
        logger = logging.getLogger("GAIA.SelfReviewWorker")
        logger.exception("Failed to import model_pool lazily")
        return None


logger = logging.getLogger("GAIA.SelfReviewWorker")

DEV_MATRIX = Path("knowledge/system_reference/dev_matrix.json")


def _load_dev_matrix():
    if not DEV_MATRIX.exists():
        return []
    try:
        with open(DEV_MATRIX, "r", encoding="utf-8") as f:
            return json.load(f)
    except Exception as e:
        logger.error(f"Failed to load dev_matrix.json: {e}")
        return []


def _save_dev_matrix(data):
    try:
        with open(DEV_MATRIX, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return True
    except Exception as e:
        logger.error(f"Failed to save dev_matrix.json: {e}")
        return False


def run_review_once(config: Config = None):
    import os
    from gaia_core.utils.dev_matrix_utils import mark_task_complete, load_dev_matrix, diff_dev_matrix, DEV_MATRIX_PATH
    from gaia_common.protocols.cognition_packet import CognitionPacket, Persona, Origin, Routing, Model, Header, Content, Context, Constraints, PersonaRole, TargetEngine
    from gaia_core.config import Config, get_config
    from gaia_core.utils.mcp_client import request_approval_via_mcp
    import logging
    logger = logging.getLogger("GAIA.SelfReviewWorker")
    config = Config()

    seeds = thought_seed.list_unreviewed_seeds()
    if not seeds:
        logger.info("No unreviewed thought seeds found.")
        return None

    mp = _get_model_pool()
    if mp is None:
        logger.error("model_pool is unavailable; aborting self-review")
        return None

    llm = mp.get_model_for_role("prime")

    for f, data in seeds:
        # Ask the model to decide if this seed corresponds to a dev_matrix task
        messages = [
            {"role": "system", "content": "You are GAIA's self-review assistant. Determine if this thought seed corresponds to any open task in dev_matrix.json. If so, provide the task name and recommend marking it 'resolved' with a short justification."},
            {"role": "user", "content": f"Thought seed: {data['seed']}\nContext: {data['context']}\nPlease respond with a JSON object only. Example: {{\"task\": \"<task name>\", \"mark_resolved\": true_or_false, \"justification\": \"<text>\"}}. Use true/false for boolean values."}
        ]

        try:
            resp = llm.create_chat_completion(messages=messages, max_tokens=256, temperature=0.2)
            text = resp.get("choices", [])[0].get("message", {}).get("content", "")
            # Heuristic parse: look for JSON blob in response
            import re
            m = re.search(r"\{.*\}", text, re.DOTALL)
            if not m:
                logger.info(f"No actionable JSON in model response for seed {f.name}: {text[:200]}")
                # mark seed reviewed but no action
                data["reviewed"] = True
                data["reviewed_at"] = datetime.now(timezone.utc).isoformat()
                thought_seed.update_seed(f.name, data)
                continue
            payload = json.loads(m.group(0))
            task_name = payload.get("task")
            mark = bool(payload.get("mark_resolved"))
            justification = payload.get("justification", "")

            if mark and task_name:
                # Load dev matrix, find matching task, set status to resolved and add resolved timestamp
                dev = _load_dev_matrix()
                matched = False
                for item in dev:
                    if item.get("task") == task_name:
                        item["status"] = "resolved"
                        item["resolved"] = datetime.now(timezone.utc).isoformat()
                        if "audit" not in item:
                            item["audit"] = []
                        item["audit"].append({"by": "gaia_self_review", "justification": justification, "ts": datetime.now(timezone.utc).isoformat()})
                        matched = True
                        break
                if not matched:
                    # Append a new resolved task entry
                    dev.append({"task": task_name, "status": "resolved", "resolved": datetime.now(timezone.utc).isoformat(), "audit": [{"by": "gaia_self_review", "justification": justification, "ts": datetime.now(timezone.utc).isoformat()}]})

                # Instead of writing directly, request approval via MCP to perform the ai_write
                new_dev_json = json.dumps(dev, indent=2)
                # Send an absolute path so the agent performing the approved write
                # writes to the exact intended file (avoids cwd/executor mismatches).
                # Mark this request as allowed to remain pending until user approval
                params = {"path": str(DEV_MATRIX.resolve()), "content": new_dev_json, "_allow_pending": True}
                req = mcp_client.request_approval_via_mcp("ai_write", params)
                if not req.get("ok"):
                    logger.error(f"Failed to request approval for dev_matrix update: {req}")
                    continue
                # Present the challenge to operator (CLI or UI should surface this)
                action_id = req.get("action_id")
                challenge = req.get("challenge")
                created_at = req.get("created_at")
                expiry = req.get("expiry")
                proposal = req.get("proposal")
                # If server response lacked proposal or timestamps, try to fetch from MCP pending list
                if action_id and (not proposal or not created_at or not expiry):
                    try:
                        fetched = mcp_client.get_pending_action(action_id)
                        if fetched.get("ok") and fetched.get("entry"):
                            entry = fetched.get("entry")
                            proposal = proposal or entry.get("proposal")
                            created_at = created_at or entry.get("created_at")
                            expiry = expiry or entry.get("expiry")
                    except Exception:
                        pass
                proposal = req.get("proposal")
                logger.info(f"Pending dev_matrix update action created: action_id={action_id} challenge={challenge} created_at={created_at} expiry={expiry}")
                # Mark seed as reviewed and record pending action
                data["reviewed"] = True
                data["reviewed_at"] = datetime.now(timezone.utc).isoformat()
                data["pending_action"] = {"action_id": action_id, "created_at": created_at, "expiry": expiry}
                thought_seed.update_seed(f.name, data)
                # Return the challenge so the caller can present it to the human
                return {"action_id": action_id, "challenge": challenge, "created_at": created_at, "expiry": expiry, "proposal": proposal}
            else:
                data["reviewed"] = True
                data["reviewed_at"] = datetime.now(timezone.utc).isoformat()
                data["review_decision"] = payload
                thought_seed.update_seed(f.name, data)
        except Exception as e:
            logger.exception(f"Error processing thought seed {f.name}: {e}")

    return None


def run_review_with_prompt(prompt: str, task_key: str = "thought_seed_system", session_id: str = "rescue-shell", persona_id: str = "RescueOperator"):
    """
    Runs a natural-language review of dev_matrix, encourages GAIA to reason and reflect on completed tasks.
    Persona and instructions are enhanced for deeper reasoning.
    """
    from gaia_core.utils.dev_matrix_utils import mark_task_complete, load_dev_matrix, diff_dev_matrix, DEV_MATRIX_PATH
    from gaia_common.protocols.cognition_packet import CognitionPacket, Persona, Origin, Routing, Model, Header, Content, Context, Constraints, PersonaRole, TargetEngine
    from gaia_core.config import Config, get_config
    from gaia_core.utils.mcp_client import request_approval_via_mcp
    import os
    config = Config()
    persona = Persona(
        identity_id="gaia-core",
        persona_id=persona_id,
        role=PersonaRole.ANALYST,
        tone_hint="You are encouraged to reason deeply, reflect on your recent actions, and propose marking tasks complete only if you have clear evidence."
    )
    routing = Routing(target_engine=TargetEngine.LITE)
    model = Model(name="Hermes-Lite", provider="local", context_window_tokens=4096)
    header = Header(
        datetime=os.environ.get("GAIA_UTC_NOW", ""),
        session_id=session_id,
        packet_id=os.urandom(8).hex(),
        sub_id="review",
        persona=persona,
        origin=Origin.AGENT,
        routing=routing,
        model=model,
        parent_packet_id=None,
        lineage=[]
    )
    constraints = Constraints(max_tokens=2048, time_budget_ms=5000, safety_mode="strict", policies=[])
    context = Context(session_history_ref=None, cheatsheets=[], constraints=constraints)
    instruction = (
        "You are GAIA, an agentic assistant. Review the dev matrix and reason through which tasks have been completed. "
        "Reflect on your recent actions and thought seeds. Only propose marking a task complete if you have clear evidence. "
        "Explain your reasoning in detail."
    )
    content = Content(original_prompt=f"{instruction}\n\n{prompt}")
    packet = CognitionPacket(header=header, context=context, content=content)
    # Simulate LLM call: in real use, would call model with packet
    # For PoC, mark as complete if task_key found
    old, new, diff = mark_task_complete(task_key, prompt)
    print("\n--- Proposed dev_matrix.json changes ---\n")
    print(diff)
    # For the interactive helper path, request an approval and request it to remain pending
    approval_req = request_approval_via_mcp(
        method="ai_write",
        params={"path": str(DEV_MATRIX_PATH), "content": json.dumps(new, indent=2, ensure_ascii=False), "_allow_pending": True}
    )
    action_id = approval_req.get("action_id") if approval_req.get("ok") else None
    proposal = approval_req.get("proposal") if approval_req.get("ok") else None
    created_at = approval_req.get("created_at") if approval_req.get("ok") else None
    expiry = approval_req.get("expiry") if approval_req.get("ok") else None
    print(f"\nApproval required. action_id={action_id}")
    return action_id, diff, proposal, created_at, expiry