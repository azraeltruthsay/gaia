"""Cross-tier audit — Prime reviews Core's deliberation in the post-stream
idle window.

k23-be7. After Core's deliberated response is yielded to the user and
the user is reading it, the GPU is idle and Prime (on CPU GGUF in AWAKE
gear) is available. We run Prime against the deliberation entry + user
message + final response with an audit prompt, capture findings, and
either annotate the journal entry (clean or with concerns) or emit a
samvega artifact when the audit flags real issues.

Three output sinks per the issue spec:
  (a) journal annotation via annotate_entry(source='cross_tier_audit')
  (b) samvega artifact when Prime flags real issues
  (c) (deferred to v1.5) pattern-tracking for next-turn Recaller injection

Lifecycle gating:
  - Skip if orchestrator state is FOCUSING (Prime on GPU doing real work)
  - Skip if state is MEDITATION (training has the GPU)
  - Skip if state is PARKED / DEEP_SLEEP (models unloaded)
  - AWAKE / LISTENING are fine — Prime is on CPU GGUF

Cap: one audit per deliberation entry. Re-audit is a no-op (the entry
already has a cross_tier_audit edit annotation).

Threading: fire-and-forget daemon thread. If the user sends another
message, the audit completes in background. If gaia-core restarts
mid-audit, the annotation just doesn't happen — entry stays without it,
which is fine. The journal entry itself is the source of truth.
"""
from __future__ import annotations

import json
import logging
import os
import re
import threading
import time
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from typing import Any, Dict, List, Optional, Tuple

logger = logging.getLogger("GAIA.CrossTierAudit")


# ── Configuration defaults (overridable via gaia_constants.json) ────────

DEFAULT_ORCH_URL = os.environ.get("GAIA_ORCHESTRATOR_URL",
                                  "http://gaia-orchestrator:6410")
DEFAULT_WEB_URL = os.environ.get("GAIA_WEB_URL", "http://gaia-web:6414")
# Samvega storage path/schema now owned by gaia_core.cognition.samvega
# (SamvegaArtifact / save_samvega_artifact) — see _maybe_emit_samvega below.

# Lifecycle states where Prime CPU is generally available for an audit.
# AWAKE: Core is on GPU, Prime is on CPU GGUF. Audit runs on Prime CPU.
# LISTENING: same as AWAKE for our purposes.
_AUDIT_OK_STATES = frozenset({"awake", "listening", "active"})


# ── Data classes ────────────────────────────────────────────────────────

@dataclass
class AuditFinding:
    """One specific concern raised by Prime about Core's deliberation."""
    category: str   # engagement | confabulation | self_claims | identity | deflection
    concern: str    # one-line specific concern text
    severity: int = 2  # 1=note, 2=concern, 3=serious. Maps loosely to samvega weight.


@dataclass
class AuditResult:
    """Result of one cross-tier audit pass."""
    clean: bool
    findings: List[AuditFinding] = field(default_factory=list)
    summary: str = ""
    followup: str = ""  # natural in-voice correction, set only when flagged
    raw_output: str = ""
    elapsed_ms: float = 0.0
    skipped_reason: Optional[str] = None  # set if audit skipped (lifecycle, etc.)
    samvega_path: Optional[str] = None    # set if a samvega artifact was created
    annotated_entry: Optional[str] = None  # journal entry id that was annotated


# ── Lifecycle gate ──────────────────────────────────────────────────────

def _lifecycle_allows_audit(timeout_s: float = 3.0) -> Tuple[bool, str]:
    """Probe the orchestrator's lifecycle state. Returns (ok, reason).

    Returns False with a reason string when audit should NOT fire.
    """
    try:
        req = urllib.request.Request(f"{DEFAULT_ORCH_URL}/lifecycle/state")
        with urllib.request.urlopen(req, timeout=timeout_s) as r:
            data = json.loads(r.read().decode())
        state = (data.get("state") or "").lower()
        prime_tier = (data.get("tiers", {}) or {}).get("prime", {}) or {}
        prime_loaded = prime_tier.get("model_loaded", False)
        prime_device = (prime_tier.get("device") or "").lower()
    except Exception as e:
        return False, f"lifecycle_probe_failed:{e.__class__.__name__}"

    if state not in _AUDIT_OK_STATES:
        return False, f"lifecycle_state:{state}"
    if not prime_loaded:
        return False, "prime_not_loaded"
    # Prime on GPU usually means it's busy serving FOCUSING — don't pile on
    if prime_device == "cuda":
        return False, "prime_on_gpu_busy"
    return True, "ok"


# ── Audit prompt ────────────────────────────────────────────────────────

_AUDIT_INSTRUCTIONS = """\
You are Prime, GAIA's deeper-reasoning tier. A turn just completed on \
Core's deliberation pipeline. Your job is a cross-check audit: examine \
Core's work for problems Core might have missed.

You have three things below: the user's message, Core's internal \
thinking trace, and Core's final user-facing response. For each \
checklist item, state PASS or FLAG with one specific concern. End with \
a one-line summary.

Audit checklist:
  1. Engagement — did Core engage with what the user actually said, \
or did it template-match a generic conversational shape?
  2. Confabulation — did Core invent specifics (function names, dates, \
made-up acronyms, fake components, technical-sounding scaffolding) \
without grounding?
  3. Self-claims — did Core claim system internals, runtime state, or \
training history it can't actually verify from inside the forward pass?
  4. Identity — did Core slip identity boundaries (claim the user's \
character traits/equipment/background as its own, conflate fictional \
with real, etc.)?
  5. Deflection — did Core deflect when a direct answer was warranted \
(introspective probes ignored, "I'll investigate" with no plan, \
generic acknowledgment without commitment)?

Output format — exactly this shape:

AUDIT: clean
or
AUDIT: flagged
FLAGS:
- <category>: <specific concern in one line>
- <category>: <another specific concern>

SUMMARY: <one short sentence under 25 words>

If flagged, also write a short, natural, first-person follow-up message \
GAIA could send the user right now to correct herself mid-conversation \
— as if she just noticed her own mistake. Rules for this message:
  - Speak as GAIA, first person, warm and brief (2-4 sentences).
  - If the user asked a direct factual/trivia question and Core deflected \
or got it wrong, answer it correctly and plainly now.
  - Never mention "audit", "Prime", "cross-tier", or "samvega", and never \
describe your own review process — just correct the substance.
  - Don't apologize profusely; this is a quick, natural correction, not \
an incident report.

FOLLOWUP: <the message>

Omit the FOLLOWUP line entirely if AUDIT is clean.

If you genuinely have no concerns, output "AUDIT: clean" with the \
SUMMARY line and nothing else. Don't invent concerns to fill space."""


def _build_audit_prompt(user_input: str, thinking: str, final_response: str) -> List[Dict[str, str]]:
    """Build the message list for Prime's audit pass."""
    user_content = (
        "USER MESSAGE:\n"
        f"{user_input.strip()}\n\n"
        "CORE'S THINKING (was inside <think>...</think>):\n"
        f"{(thinking or '(empty)').strip()}\n\n"
        "CORE'S FINAL RESPONSE:\n"
        f"{(final_response or '(empty)').strip()}\n\n"
        "Run the audit now."
    )
    return [
        {"role": "system", "content": _AUDIT_INSTRUCTIONS},
        {"role": "user", "content": user_content},
    ]


# ── Output parsing ──────────────────────────────────────────────────────

_VERDICT_RE = re.compile(r"^\s*AUDIT:\s*(clean|flagged)\s*$", re.IGNORECASE | re.MULTILINE)
_FLAG_LINE_RE = re.compile(
    r"^\s*[-•*]\s*(engagement|confabulation|self[_\- ]claims?|identity|deflection)\s*:\s*(.+?)\s*$",
    re.IGNORECASE | re.MULTILINE,
)
_SUMMARY_RE = re.compile(r"^\s*SUMMARY:\s*(.+?)\s*$", re.IGNORECASE | re.MULTILINE)
# FOLLOWUP is written last and may span multiple lines — capture to end of output.
_FOLLOWUP_RE = re.compile(r"^\s*FOLLOWUP:\s*(.+)", re.IGNORECASE | re.DOTALL | re.MULTILINE)


def _normalize_category(raw: str) -> str:
    raw = raw.strip().lower().replace(" ", "_").replace("-", "_")
    if raw.startswith("self"):
        return "self_claims"
    return raw


def _parse_audit_response(raw: str) -> Tuple[bool, List[AuditFinding], str, str]:
    """Parse Prime's audit text into (clean, findings, summary, followup)."""
    if not raw:
        return False, [], "", ""

    verdict_match = _VERDICT_RE.search(raw)
    clean: Optional[bool] = None
    if verdict_match:
        clean = verdict_match.group(1).lower() == "clean"

    findings: List[AuditFinding] = []
    for m in _FLAG_LINE_RE.finditer(raw):
        category = _normalize_category(m.group(1))
        concern = m.group(2).strip()
        # Prime sometimes emits "PASS" or "no concerns" as bullet items;
        # filter those out.
        if re.match(r"^(pass|none|no concerns?|n/?a|ok)\b", concern, re.IGNORECASE):
            continue
        findings.append(AuditFinding(category=category, concern=concern))

    summary = ""
    sm = _SUMMARY_RE.search(raw)
    if sm:
        summary = sm.group(1).strip()

    followup = ""
    fm = _FOLLOWUP_RE.search(raw)
    if fm:
        followup = fm.group(1).strip()

    # If we found bullet flags but verdict says "clean", trust the flags
    if findings and clean is True:
        clean = False
    # If verdict missing, infer from flags
    if clean is None:
        clean = not findings

    # A follow-up only makes sense for a flagged turn.
    if clean:
        followup = ""

    return clean, findings, summary, followup


# ── Samvega artifact ────────────────────────────────────────────────────

def _maybe_emit_samvega(
    findings: List[AuditFinding],
    user_input: str,
    final_response: str,
    journal_entry_id: Optional[str],
    summary: str,
    session_id: Optional[str] = None,
) -> Optional[str]:
    """Emit a samvega artifact when findings are serious. Returns path or None.

    GAIA_Project-e8hf (2026-10-14): this must build a real SamvegaArtifact and
    go through save_samvega_artifact(), not write an ad-hoc dict. The actual
    correction-learning pipeline (samvega_kv_fold, run during the sleep
    cycle's DREAMING phase) only treats an artifact as eligible when it has a
    non-zero `weight` and a non-empty `corrected_understanding` — fields the
    old ad-hoc payload never set, so every audit-flagged confabulation was
    silently invisible to the one mechanism designed to act on it.
    """
    if not findings:
        return None
    # Heuristic: emit samvega only when we have ≥2 findings or any
    # severity-3 finding. Single low-severity flags get journaled but
    # don't escalate to samvega (avoid noise).
    if len(findings) < 2 and not any(f.severity >= 3 for f in findings):
        return None

    from gaia_core.cognition.samvega import (
        SamvegaArtifact, SamvegaTrigger, compute_samvega_weight, save_samvega_artifact,
    )

    max_severity = max(f.severity for f in findings)
    weight = compute_samvega_weight(
        trigger=SamvegaTrigger.CROSS_TIER_AUDIT.value,
        audit_finding_count=len(findings),
        audit_max_severity=max_severity,
    )

    # Dominant category becomes root_cause so _run_samvega_introspection's
    # clustering (group-by-root_cause, boost repeated patterns) actually
    # groups repeated confabulation/identity/etc. findings together instead
    # of lumping every audit artifact into one bucket.
    category_counts: Dict[str, int] = {}
    for f in findings:
        category_counts[f.category] = category_counts.get(f.category, 0) + 1
    root_cause = max(category_counts, key=category_counts.get)

    corrected_understanding = summary.strip() or "Response was flagged by cross-tier audit."
    concern_lines = "; ".join(f"[{f.category}] {f.concern}" for f in findings)
    if concern_lines:
        corrected_understanding = f"{corrected_understanding} {concern_lines}"

    artifact = SamvegaArtifact(
        artifact_type="samvega",
        timestamp=datetime.now(timezone.utc).isoformat(),
        session_id=session_id or "",
        packet_id=journal_entry_id or "",
        trigger=SamvegaTrigger.CROSS_TIER_AUDIT.value,
        original_output_summary=final_response[:600],
        what_went_wrong=concern_lines or summary,
        root_cause=root_cause,
        corrected_understanding=corrected_understanding,
        weight=weight,
    )

    out = save_samvega_artifact(artifact)
    if out is None:
        return None
    logger.info(
        "Cross-tier audit: samvega %s emitted (%d findings, weight=%.2f, root_cause=%s)",
        out.name, len(findings), weight, root_cause,
    )
    return str(out)


# ── Visible same-conversation follow-up (e8hf Phase B) ──────────────────

def _maybe_push_followup(session_id: str, followup: str, cfg: Dict[str, Any]) -> bool:
    """Best-effort delivery of a visible self-correction into the live
    session. Routed by gaia-web, which knows how to reach each transport
    (Discord bot push vs. the Mission Control autonomous-message SSE feed)
    from the session_id's existing prefix scheme. Never raises — this is a
    background daemon thread with no caller waiting on it.
    """
    if not cfg.get("followup_enabled", True):
        return False
    url = f"{DEFAULT_WEB_URL}/api/conversations/{session_id}/push"
    payload = json.dumps({
        "text": followup,
        "source": "cross_tier_audit_correction",
    }).encode("utf-8")
    try:
        req = urllib.request.Request(
            url, data=payload, method="POST",
            headers={"Content-Type": "application/json"},
        )
        with urllib.request.urlopen(req, timeout=5.0) as r:
            ok = 200 <= r.status < 300
        logger.info("Cross-tier audit: follow-up push to %s -> %s", session_id, "ok" if ok else r.status)
        return ok
    except Exception as e:
        logger.warning("Cross-tier audit: follow-up push failed for %s: %s", session_id, e)
        return False


# ── Per-entry de-dup (one audit per journal entry) ──────────────────────

_seen_lock = threading.Lock()
_audited_entries: set = set()


def _claim_entry(entry_id: Optional[str]) -> bool:
    """Mark an entry as audited. Returns False if already claimed."""
    if not entry_id:
        return True  # Can't dedupe without an id; let it through
    with _seen_lock:
        if entry_id in _audited_entries:
            return False
        _audited_entries.add(entry_id)
        # Cap memory: keep the set bounded
        if len(_audited_entries) > 1000:
            # Drop oldest half (set ordering isn't reliable in Python 3.6+
            # but we just need bounded growth, not strict LRU)
            for v in list(_audited_entries)[:500]:
                _audited_entries.discard(v)
    return True


# ── Sync audit ──────────────────────────────────────────────────────────

def run_cross_tier_audit_sync(
    *,
    user_input: str,
    thinking: str,
    final_response: str,
    journal_entry_id: Optional[str],
    model_pool,
    config,
    session_id: Optional[str] = None,
    max_audit_tokens: Optional[int] = None,
    audit_temperature: Optional[float] = None,
) -> AuditResult:
    """Synchronous cross-tier audit. Reads config; usually called from a thread."""
    cfg: Dict[str, Any] = {}
    try:
        constants = config.constants if hasattr(config, "constants") else config
        cfg = (constants or {}).get("CROSS_TIER_AUDIT", {}) or {}
    except Exception:
        cfg = {}

    if not cfg.get("enabled", True):
        return AuditResult(clean=True, skipped_reason="disabled_in_config")

    if not _claim_entry(journal_entry_id):
        return AuditResult(clean=True, skipped_reason="already_audited")

    ok, reason = _lifecycle_allows_audit()
    if not ok:
        logger.debug("Cross-tier audit: skipping (%s)", reason)
        return AuditResult(clean=True, skipped_reason=reason)

    max_tokens = int(max_audit_tokens or cfg.get("max_audit_tokens", 350))
    temperature = float(audit_temperature or cfg.get("audit_temperature", 0.3))

    messages = _build_audit_prompt(user_input, thinking, final_response)

    t0 = time.time()
    try:
        res = model_pool.forward_to_model(
            "prime",
            messages=messages,
            max_tokens=max_tokens,
            temperature=temperature,
            top_p=0.9,
        )
        raw = (res.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
    except Exception:
        logger.exception("Cross-tier audit: Prime call failed")
        return AuditResult(
            clean=True,
            elapsed_ms=(time.time() - t0) * 1000.0,
            skipped_reason="prime_call_failed",
        )

    elapsed_ms = (time.time() - t0) * 1000.0
    clean, findings, summary, followup = _parse_audit_response(raw)

    # Annotate the journal entry — clean or flagged
    annotated = None
    try:
        from gaia_core.memory.journal import annotate_entry
        if journal_entry_id:
            note_lines = [f"Cross-tier audit ({'clean' if clean else 'flagged'}, Prime)"]
            if summary:
                note_lines.append(summary)
            for f in findings:
                note_lines.append(f"  - {f.category}: {f.concern}")
            note_text = "\n".join(note_lines)
            ok = annotate_entry(
                journal_entry_id, note_text,
                source="cross_tier_audit",
                reason=f"audit_elapsed={elapsed_ms:.0f}ms findings={len(findings)}",
            )
            if ok:
                annotated = journal_entry_id
    except Exception:
        logger.exception("Cross-tier audit: annotate_entry failed")

    samvega_path = _maybe_emit_samvega(
        findings=findings,
        user_input=user_input,
        final_response=final_response,
        journal_entry_id=journal_entry_id,
        summary=summary,
        session_id=session_id,
    )

    # e8hf Phase B: a flagged-and-samvega-serious turn gets a visible,
    # same-conversation follow-up, not just a silent next-sleep-cycle fold.
    # Reuses _maybe_emit_samvega's own severity bar (>=2 findings, or any
    # severity-3) rather than inventing a second threshold.
    if samvega_path and followup and session_id:
        _maybe_push_followup(session_id, followup, cfg)

    logger.info(
        "Cross-tier audit complete: entry=%s clean=%s findings=%d elapsed=%.0fms samvega=%s followup=%s",
        journal_entry_id, clean, len(findings), elapsed_ms,
        os.path.basename(samvega_path) if samvega_path else "-",
        "yes" if (samvega_path and followup) else "no",
    )

    return AuditResult(
        clean=clean,
        findings=findings,
        summary=summary,
        followup=followup,
        raw_output=raw,
        elapsed_ms=elapsed_ms,
        annotated_entry=annotated,
        samvega_path=samvega_path,
    )


# ── Async fire-and-forget ───────────────────────────────────────────────

def schedule_cross_tier_audit(
    *,
    user_input: str,
    thinking: str,
    final_response: str,
    journal_entry_id: Optional[str],
    model_pool,
    config,
    session_id: Optional[str] = None,
) -> Optional[threading.Thread]:
    """Spawn a daemon thread to run the audit. Returns immediately.

    Returns None if config-disabled, otherwise the started Thread.
    """
    cfg: Dict[str, Any] = {}
    try:
        constants = config.constants if hasattr(config, "constants") else config
        cfg = (constants or {}).get("CROSS_TIER_AUDIT", {}) or {}
    except Exception:
        cfg = {}
    if not cfg.get("enabled", True):
        return None

    def _runner():
        try:
            run_cross_tier_audit_sync(
                user_input=user_input,
                thinking=thinking,
                final_response=final_response,
                journal_entry_id=journal_entry_id,
                model_pool=model_pool,
                config=config,
                session_id=session_id,
            )
        except Exception:
            logger.exception("Cross-tier audit: thread crashed")

    name = f"cross-tier-audit-{(journal_entry_id or session_id or 'anon')[:32]}"
    t = threading.Thread(target=_runner, daemon=True, name=name)
    t.start()
    return t
