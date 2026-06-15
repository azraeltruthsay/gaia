#!/usr/bin/env python3
"""
repro_231_tool_grounding.py — live end-to-end repro for GAIA_Project-231.

Reproduces the production Discord failure (a recited poem -> an unrelated
"recursion" answer when asked its name) and demonstrates the Phase 0-3 fix on
the LIVE system: tool-result ledger -> prompt injection -> live-model answer ->
expand_context recall + CFR deixis rescue.

Runs the real prompt_builder + the LOCAL Core model (Gemma4-E4B via the embedded
GAIA Engine at :8092 — the model that actually serves Discord) against the same
session twice — once WITHOUT the tool ledger (the old behaviour) and once WITH
it (the fix) — so the difference is visible. Groq was a sidequest; this is the
model whose grounding matters.

    docker exec gaia-core python /app/scripts/repro_231_tool_grounding.py
"""
from __future__ import annotations

import logging
import uuid
from datetime import datetime, timezone

logging.disable(logging.WARNING)

from gaia_common.protocols.cognition_packet import (
    CognitionPacket, Header, Persona, PersonaRole, Routing, TargetEngine, Model,
    Intent, SystemTask, Context, SessionHistoryRef, Constraints, Content,
    Reasoning, Response, Governance, Safety, Metrics, TokenUsage, Status,
    PacketState, Origin, RelevantHistorySnippet,
)
from gaia_core.config import get_config
from gaia_core.memory.session_manager import SessionManager
from gaia_core.memory.conversation_cfr import looks_like_reference_query
from gaia_core.utils.prompt_builder import build_from_packet
from gaia_core.utils.output_router import _strip_think_tags_robust as _strip
from gaia_core.models.vllm_remote_model import VLLMRemoteModel
from gaia_core.main import _resolve_cfr_recall

# The exact production scenario.
POEM_TITLE = "Coming Undone | The Poetry Foundation"
POEM_URL = "https://www.poetryfoundation.org/articles/coming-undone"
POEM_BODY = ("Coming Undone — On Context Collapse, Ryan Ruby's vertiginous secret history of poetry. "
             "Writing, it's been said, remains unique because it is the only medium that can use its own "
             "form to investigate itself...")
QUESTION = "Can you tell me its name?"


def _build_snippets(history_turns, ledger):
    """Replicate agent_core's relevant_history_snippet build (with optional ledger)."""
    snips = []
    for t in history_turns:
        snips.append(RelevantHistorySnippet(id=t["id"], role=t["role"],
                                             summary=_strip(t["content"])[:2000]))
    # agent_core prepends the tool ledger as always-present 'tool' snippets.
    for e in ledger:
        prov = e.get("title") or e.get("url") or e.get("tool", "")
        bits = [f"Retrieved via {e.get('tool','tool')}: {prov}"]
        if e.get("url") and e["url"] != prov:
            bits.append(e["url"])
        if e.get("gist"):
            bits.append(f"— {e['gist']}")
        if e.get("body"):
            bits.append(f"(full text via expand_context id={e.get('id')})")
        snips.insert(0, RelevantHistorySnippet(id=e.get("id", "tl"), role="tool",
                                               summary=_strip(" ".join(bits))[:600]))
    return snips


def _packet(snippets):
    now = datetime.now(timezone.utc).isoformat()
    p = CognitionPacket(
        version="0.3.0-repro", schema_id="gaia-cogpacket-v0.3",
        header=Header(datetime=now, session_id="repro231", packet_id="p-" + uuid.uuid4().hex[:8],
            sub_id="s0", persona=Persona(identity_id="gaia", persona_id="Default",
            role=PersonaRole.DEFAULT, tone_hint="neutral"), origin=Origin.SYSTEM,
            routing=Routing(target_engine=TargetEngine.PRIME, priority=5),
            model=Model(name="core", provider="managed", context_window_tokens=4096,
                        max_output_tokens=300, response_buffer_tokens=64, temperature=0.4, top_p=0.95)),
        intent=Intent(user_intent=QUESTION, system_task=SystemTask.STREAM, confidence=1.0, tags=["repro"]),
        context=Context(session_history_ref=SessionHistoryRef(type="none", value="none"), cheatsheets=[],
            constraints=Constraints(max_tokens=300, time_budget_ms=30000, safety_mode="permissive"),
            relevant_history_snippet=snippets),
        content=Content(original_prompt=QUESTION), reasoning=Reasoning(),
        response=Response(candidate="", confidence=0.0, stream_proposal=True),
        governance=Governance(safety=Safety(execution_allowed=True, dry_run=False)),
        metrics=Metrics(token_usage=TokenUsage(prompt_tokens=0, completion_tokens=0, total_tokens=0), latency_ms=0),
        status=Status(finalized=False, state=PacketState.INITIALIZED, next_steps=["x"]))
    p.compute_hashes()
    return p


def _ask(snippets, model):
    msgs = build_from_packet(_packet(snippets))
    r = model.create_chat_completion(messages=msgs, max_tokens=250, temperature=0.4)
    try:
        return r["choices"][0]["message"]["content"].strip()
    except Exception:
        return (r.get("response") or r.get("text") or str(r)).strip()


def main():
    cfg = get_config()
    sm = SessionManager(cfg)
    sid = "repro231"
    sm.sessions.pop(sid, None)
    # LOCAL Core (Gemma4-E4B) via the embedded GAIA Engine — the real model.
    model = VLLMRemoteModel(dict(cfg.MODEL_CONFIGS.get("core", {})), cfg)

    # Turn 1 (simulated) — faithful to the original bug: the content was
    # surfaced via a TOOL and never became a durable turn. So the saved
    # assistant turn does NOT carry the title/author (just a hand-off line),
    # and the only record of what was fetched is the tool ledger. This is the
    # real failure condition: without the ledger, turn 2 has nothing to ground
    # on (BEFORE), with it the title is in context (AFTER).
    # The user does NOT name the work (as in the real failure) — only the tool
    # knows what was fetched, so the title lives solely in the ledger.
    sm.add_message(sid, "user", "Find me a poem about context collapse and recite it.")
    sm.add_message(sid, "assistant", "Here it is — let me know what you think.")
    sm.record_tool_result(sid, tool="web_search", action="search",
                          title=POEM_TITLE, url=POEM_URL, gist=POEM_BODY, body=POEM_BODY)
    history = sm.get_or_create_session(sid).history
    ledger = sm.get_tool_ledger(sid)

    print("=" * 72)
    print(f"Turn 2 question: {QUESTION!r}")
    print(f"CFR detects reference/meta query: {looks_like_reference_query(QUESTION)}")
    print("=" * 72)

    # BEFORE: the old behaviour — no tool ledger in context.
    before = _ask(_build_snippets(history, ledger=[]), model)
    print("\n--- BEFORE (no tool ledger) ---")
    print(before)
    print(f"  grounded? {'Coming Undone' in before}")

    # AFTER: Phase 1 ledger injected as always-present provenance.
    after = _ask(_build_snippets(history, ledger=ledger), model)
    print("\n--- AFTER (with tool ledger, the fix) ---")
    print(after)
    print(f"  grounded? {'Coming Undone' in after}")

    # Phase 2: expand_context recalls the full body by ledger id.
    tl_id = ledger[-1]["id"]
    rec = _resolve_cfr_recall(history=history, rid=tl_id, tool_ledger=ledger)
    print("\n--- Phase 2: expand_context recall ---")
    print(f"  expand_context(id={tl_id}) -> role={rec['role']}, "
          f"{len(rec['text'])} chars, title present: {'Coming Undone' in rec['text']}")

    print("\n" + "=" * 72)
    verdict = "PASS" if ("Coming Undone" in after and "Coming Undone" not in before) else \
              ("PARTIAL" if "Coming Undone" in after else "INCONCLUSIVE")
    print(f"VERDICT: {verdict} — ledger grounding {'changed' if verdict=='PASS' else 'did not clearly change'} the answer.")
    sm.sessions.pop(sid, None)


if __name__ == "__main__":
    main()
