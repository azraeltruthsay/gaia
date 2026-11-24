"""Grounded Samvega findings (aidw.4, GROUNDED_SPEECH.md part C1).

The cross-tier audit judged her replies without seeing what the turn looked
up: on 2026-11-21 it caught her invented palace rooms, and in the same breath
faulted her for "claiming a palace(navigate) command" that is real. A finding
about what is true has to be checked against something other than a model's
opinion.

Here:
- turn_evidence(packet): what the turn had to answer from (tool results,
  pages read, world state), as text for the auditor and for the check.
- look_up(question, ...): when the turn looked nothing up, what she should
  have looked up — her own selector decides (aidw.1), and the call runs if it
  only reads. The blind-check pattern: the evidence comes from a lookup, not
  from the judge.
- confirm(claim, reply, evidence, question): code decides whether a flagged
  claim stands: the quoted words must be in her reply, and most of what they
  say (content words, names, numbers) must be missing from the evidence.

Statuses: confirmed (the evidence doesn't hold it), rejected (the evidence
holds it: the auditor was wrong), unverified (nothing to check against),
judgment (engagement, deflection, identity: not a matter of evidence).
"""
from __future__ import annotations

import logging
import os
import re
import time
from types import SimpleNamespace
from typing import Any, Optional, Tuple

logger = logging.getLogger("GAIA.FindingGrounding")

EVIDENCE_CHARS = 3000
CHECKABLE = frozenset({"confabulation", "self_claims"})
CONFIRMED, REJECTED, UNVERIFIED, JUDGMENT = "confirmed", "rejected", "unverified", "judgment"
UNSUPPORTED_SHARE = 0.5  # share of a claim's content that the evidence must lack

# Actions that only read. A lookup to check a claim must never change anything.
_READ_ACTIONS = frozenset({
    "navigate", "status", "recall", "read", "lookups", "work", "list", "info", "get",
    "search", "fetch", "tools", "logs", "turns", "world", "health", "adapter_list",
    "adapter_info", "list_tree", "list_dir", "find", "show", "overview", "query",
    "beads", "bead", "tree",
})


def turn_evidence(packet: Any, limit: int = EVIDENCE_CHARS) -> str:
    """What this turn answered from: tool results and pages first, then the
    world-state block. Empty when there was nothing."""
    parts, world = [], ""
    try:
        from gaia_common.utils.tool_call_parser import tool_result_text
        for df in getattr(getattr(packet, "content", None), "data_fields", None) or []:
            key = getattr(df, "key", "")
            val = getattr(df, "value", None)
            if key in ("tool_result", "web_grounding"):
                if isinstance(val, dict) and "output" in val:
                    label = val.get("tool") or val.get("tool_name") or "tool"
                    parts.append(f"[{label}]\n{tool_result_text(val.get('output'), limit=limit)}")
                elif val:
                    parts.append(str(val)[:limit])
            elif key == "world_state_snapshot" and isinstance(val, str):
                world = val
    except Exception:
        logger.debug("turn_evidence: could not read the packet", exc_info=True)
    try:
        routed = getattr(getattr(packet, "tool_routing", None), "execution_result", None)
        out = getattr(routed, "output", None)
        if out and not parts:
            from gaia_common.utils.tool_call_parser import tool_result_text
            parts.append(tool_result_text(out, limit=limit))
    except Exception:
        pass
    text = "\n\n".join(p for p in parts if p.strip())
    if world:
        text = (text + f"\n\n{WORLD_MARK}\n" + world).strip()
    return text[:limit]


WORLD_MARK = "[world state]"


def has_lookup(evidence: str) -> bool:
    """Whether the evidence holds something the turn looked up, beyond the
    world-state block every turn carries (clock, load, which minds)."""
    return bool((evidence or "").split(WORLD_MARK, 1)[0].strip())


def _read_only(tool: str, params: dict) -> bool:
    action = str((params or {}).get("action") or "").lower()
    try:
        from gaia_common.utils.domain_tools import SENSITIVE_ACTIONS
        if (tool, action) in SENSITIVE_ACTIONS:
            return False
    except Exception:
        return False
    # issue(status) with an id sets that issue's status; without one it lists.
    if tool == "issue" and action == "status" and (params or {}).get("id"):
        return False
    return action in _READ_ACTIONS


def look_up(question: str, model_pool, select=None, call=None, decide: bool = True) -> Tuple[str, str]:
    """(call label, evidence text) from what she should have looked up for
    `question`; ("", "") when no read-only call fits or it failed."""
    try:
        if select is None:
            from gaia_core.cognition.tool_selector import select_tool

            def select(q):
                model, name = None, None
                for cand in ("core", "prime", "cpu_prime"):
                    try:
                        model = model_pool.acquire_model(cand)
                    except Exception:
                        model = None
                    if model is not None:
                        name = cand
                        break
                if model is None:
                    return None
                try:
                    pkt = SimpleNamespace(
                        header=SimpleNamespace(session_id="samvega_check", packet_id="samvega_check"),
                        intent=SimpleNamespace(user_intent=""),
                        context=SimpleNamespace(available_mcp_tools=None, relevant_history_snippet=[]))
                    sel, _ = select_tool(pkt, q, model, decide=decide)
                    return sel
                finally:
                    try:
                        model_pool.release_model(name)
                    except Exception:
                        pass
        if call is None:
            from gaia_core.utils.mcp_client import call_jsonrpc

            def call(tool, params):
                rpc = call_jsonrpc(method=tool, params=params)
                if not rpc.get("ok"):
                    return None
                resp = rpc.get("response", {})
                return resp.get("result", resp)
        sel = select(question)
        if sel is None:
            return "", ""
        tool, params = sel.tool_name, dict(sel.params or {})
        label = f"{tool}({', '.join(f'{k}={v}' for k, v in params.items())})"
        if not _read_only(tool, params):
            logger.info("Samvega check: %s would change something; not run", label)
            return label, ""
        hit = _CALL_CACHE.get(label)
        if hit and time.time() - hit[0] < CALL_CACHE_S:
            return label, hit[1]
        out = call(tool, params)
        if not out or (isinstance(out, dict) and out.get("ok") is False):
            return label, ""
        from gaia_common.utils.tool_call_parser import tool_result_text
        text = tool_result_text(out, limit=EVIDENCE_CHARS)
        _CALL_CACHE[label] = (time.time(), text)
        return label, text
    except Exception:
        logger.debug("Samvega check: look-up failed", exc_info=True)
        return "", ""


# The same read-only call within a minute (two statements checked against
# her tool list) is run once.
CALL_CACHE_S = 60
_CALL_CACHE: dict = {}


def _stem(w: str) -> str:
    """browsing/browses -> brows, searching/searches -> search, rooms -> room."""
    for suf in ("ing", "ed", "es", "s"):
        if w.endswith(suf) and len(w) - len(suf) >= 4:
            return w[:-len(suf)]
    return w


def _stems(text: str) -> set:
    from gaia_common.utils.lookup_memory import content_terms
    return {_stem(w) for w in content_terms(text or "")}


_NUMBER_RE = re.compile(r"(?<![\w.])\d{1,3}(?:,\d{3})+(?:\.\d+)?(?![\w])|(?<![\w.])\d+(?:\.\d+)?(?![\w])")


def _numbers(text: str) -> list:
    return [float(m.group(0).replace(",", "")) for m in _NUMBER_RE.finditer(text or "")]


def unbacked_numbers(claim: str, known: str) -> list:
    """Numbers in a claim the evidence doesn't hold — whole numbers only
    ("12 wings" is not backed by "122 memories"); a large number within
    2.5% counts ("nearly 4,000" for 3,994)."""
    have = _numbers(known)
    out = []
    for n in _numbers(claim):
        if any(n == h or (n >= 100 and abs(n - h) <= 0.025 * max(n, h)) for h in have):
            continue
        out.append(n)
    return out


def in_reply(claim: str, reply: str) -> bool:
    """The auditor's quote is really hers: most of its words are in the reply."""
    c = _stems(claim)
    if not c:
        # "About 3 GB.": no content word to match, so match the text itself
        norm = lambda t: re.sub(r"\s+", " ", (t or "").strip().lower())
        return bool(norm(claim)) and norm(claim) in norm(reply)
    return len(c & _stems(reply)) / len(c) >= 0.7


def unsupported(claim: str, known: str) -> list:
    """What a claim says that `known` doesn't hold: content words, plus the
    names and numbers second_mind.unbacked finds."""
    from gaia_core.cognition.second_mind import unbacked
    k = _stems(known)
    words = sorted(w for w in _stems(claim) if w not in k)
    return words + [x for x in unbacked(claim, known) if x.lower().rstrip("s") not in words]


def confirm(claim: str, reply: str, evidence: str, question: str = "", generate=None) -> str:
    """Does the claim hold up against the evidence? With a model at hand
    (`generate`), the blind check; without one, the lexical fallback."""
    # GROUNDING_VERDICT=lexical brings back the word-overlap verdict (aidw.11).
    if generate is not None and os.environ.get("GROUNDING_VERDICT", "blind").lower() != "lexical":
        return confirm_blind(claim, reply, evidence, question, generate)
    return confirm_lexical(claim, reply, evidence, question)


def confirm_lexical(claim: str, reply: str, evidence: str, question: str = "") -> str:
    """Word-overlap verdict (aidw.4): kept only as the fallback when no model
    is available. Misses negation, trips on synonyms (aidw.11)."""
    claim = (claim or "").strip()
    if not claim:
        return UNVERIFIED  # no words of hers quoted: nothing to check
    if not in_reply(claim, reply):
        return REJECTED  # a finding about words she didn't say
    if not (evidence or "").strip():
        return UNVERIFIED
    total = _stems(claim)
    if not total:
        return UNVERIFIED
    from gaia_core.cognition.second_mind import unbacked
    known = f"{question}\n{evidence}"
    # Numbers and names: strict ("twelve wings" against 6 wings). Numbers are
    # checked here, whole: unbacked() takes "12" as backed by "122".
    c, kn = _numerize(claim), _numerize(known)
    if unbacked_numbers(c, kn):
        return CONFIRMED
    if [x for x in unbacked(c, kn) if not re.fullmatch(r"[\d:.,]+", x)]:
        return CONFIRMED
    # Other words: only specific ones count. "The other two are empty" said
    # of rooms the source lists as "0 memories" is a paraphrase, not an
    # invention; "watchdog, immune system, self-repair protocols" is.
    k = _stems(known)
    missing = [w for w in total if w not in k]
    specific = [w for w in missing if len(w) >= SPECIFIC_LEN]
    if len(missing) / len(total) >= UNSUPPORTED_SHARE and len(specific) >= MIN_SPECIFIC:
        return CONFIRMED
    return REJECTED


SPECIFIC_LEN = 6
MIN_SPECIFIC = 3
_NUM_WORDS = {w: str(i) for i, w in enumerate(
    "zero one two three four five six seven eight nine ten eleven twelve thirteen fourteen fifteen "
    "sixteen seventeen eighteen nineteen twenty".split())}
_NOT_A_NOUN = frozenset("are is was were of or and have has had more less than to in on at by for "
                        "with the a an it its them they which that who".split())
_NUM_RE = re.compile(r"\b(" + "|".join(_NUM_WORDS) + r")\s+([a-z][a-z-]{2,})\b", re.IGNORECASE)


def _numerize(text: str) -> str:
    """'twelve wings' -> '12 wings'; 'the other two are empty' left alone
    (a number word counts only before what it counts)."""
    def sub(m):
        if m.group(2).lower() in _NOT_A_NOUN:
            return m.group(0)
        return f"{_NUM_WORDS[m.group(1).lower()]} {m.group(2)}"
    return _NUM_RE.sub(sub, text or "")


_CLAIM_RE = re.compile(r"""CLAIM:\s*["“](.+?)["”]""", re.IGNORECASE)
_EVID_RE = re.compile(r"EVIDENCE:\s*(.+?)\s*$", re.IGNORECASE)


def split_concern(concern: str) -> Tuple[str, str, str]:
    """(concern, claim, evidence note) from an audit flag line:
    'invented rooms | CLAIM: "rooms cover the watchdog" | EVIDENCE: none'."""
    parts = [p.strip() for p in (concern or "").split("|")]
    text, claim, note = parts[0], "", ""
    for p in parts[1:]:
        m = _CLAIM_RE.search(p)
        if m:
            claim = m.group(1).strip()
            continue
        m = _EVID_RE.search(p)
        if m:
            note = m.group(1).strip()
    if not claim:
        m = _CLAIM_RE.search(concern or "")
        if m:
            claim = m.group(1).strip()
            text = (concern[:m.start()].rstrip(" |") or text)
    return text, claim, note


# ── Claim check: extraction by a model, lookup by a tool, verdict by code ──

_EXTRACT_PROMPT = (
    "Below are a question GAIA was asked and the reply she gave. List the statements in the reply "
    "that describe GAIA herself: her systems, memory palace, tools, files, services, state or "
    "history. A statement answering a question about her counts even when it doesn't say \"I\" "
    "or \"my\". Quote each one word for word from the reply, one per line. Leave out greetings, "
    "opinions and offers to help. If there are none, write NONE.\n\n"
    "QUESTION:\n{question}\n\nREPLY:\n{reply}"
)
MAX_CLAIMS = 4


def extract_self_claims(reply: str, generate, question: str = "") -> list:
    """Her statements about herself, quoted from the reply (a model lists
    them; code keeps only quotes that are really in the reply)."""
    if not (reply or "").strip():
        return []
    try:
        raw = generate([{"role": "user", "content": _EXTRACT_PROMPT.format(
            question=(question or "(not given)")[:500], reply=reply[:2500])}]) or ""
    except Exception:
        logger.debug("Claim check: extraction failed", exc_info=True)
        return []
    out = []
    for line in raw.splitlines():
        line = re.sub(r"^\s*(?:[-*•]|\d+[.)])\s*", "", line).strip().strip('"“”').strip()
        if not line or line.upper().startswith("NONE"):
            continue
        if len(_stems(line)) >= 3 and in_reply(line, reply) and line not in out:
            out.append(line)
    return out[:MAX_CLAIMS]


def check_claims(question: str, reply: str, evidence: str, model_pool,
                 generate=None, look_up_fn=None, max_claims: int = MAX_CLAIMS,
                 stop_on_confirmed: bool = False) -> list:
    """Each statement she made about herself, checked: against the turn's
    evidence first; what that doesn't hold, against what her selector would
    look up for it (read-only). Returns [{claim, status, call, evidence}]."""
    if generate is None:
        generate = _core_generate(model_pool)
        if generate is None:
            return []
    look_up_fn = look_up_fn or look_up
    results, cache = [], {}
    _QUESTIONS_MEMO.clear()   # reuse within this check only (turn evidence, then a lookup)
    for claim in extract_self_claims(reply, generate, question)[:max_claims]:
        if has_lookup(evidence) and confirm(claim, reply, evidence, question, generate) == REJECTED:
            results.append({"claim": claim, "status": "backed", "call": "", "evidence": ""})
            continue
        # What she should have looked up for the question: it is what the
        # reply has to agree with. (Asking from the claim's own words instead
        # picked worse sources: 2/7 vs 7/7 on the 2026-11-21 replay set.)
        request = f'{question}\n(To check this statement: "{claim}")'
        call, ev = look_up_fn(request, model_pool)
        if call in cache and not ev:
            ev = cache[call]
        cache[call] = ev
        status = confirm(claim, reply, f"{evidence}\n{ev}".strip(), question, generate) if ev else UNVERIFIED
        results.append({"claim": claim, "status": status, "call": call, "evidence": ev[:2000]})
        if stop_on_confirmed and status == CONFIRMED:
            break   # one is enough to act on (pre-send regrounds once)
    if results:
        logger.info("Claim check: %s", "; ".join(f"{r['status']}: {r['claim'][:60]!r}"
                                                 + (f" via {r['call']}" if r['call'] else "")
                                                 for r in results))
    return results


def _core_generate(model_pool):
    """Core as a short, greedy generator — only while it is on the GPU (on
    the CPU each extraction would take many seconds beside live turns)."""
    try:
        from gaia_core.cognition.agent_core import _core_on_gpu
        if not _core_on_gpu():
            return None
    except Exception:
        return None

    def generate(messages):
        model = model_pool.acquire_model("core")
        if model is None:
            return ""
        try:
            res = model.create_chat_completion(messages=messages, max_tokens=220, temperature=0.0, stream=False)
            return (res.get("choices", [{}])[0].get("message", {}).get("content") or "").strip()
        finally:
            try:
                model_pool.release_model("core")
            except Exception:
                pass
    return generate



# ── Blind check (aidw.11): no model ever sees a claim it is asked to approve ──
#
# 1. From the claim alone: the questions it answers, with its answers.
# 2. From the source alone (the claim unseen): answers to those questions,
#    or NOT IN SOURCE.
# 3. Two short answers side by side, neither marked as hers: same, different
#    or unclear.
# Code decides: a different answer = contradicted (confirmed); else a same
# answer = backed (rejected); else unverified. Shown a claim and a page, a
# model says the page supports it (blind-fact-checking; the C1 replay:
# Prime vouched for invented rooms). Word overlap missed negation and
# tripped on synonyms.

_QUESTIONS_PROMPT = (
    "Here is a statement, said in answer to: \"{question}\"\n\nSTATEMENT: {claim}\n\n"
    "Write up to 3 short questions about facts this statement answers (what something has or is, names, "
    "numbers, lists, yes/no), each followed by the answer the statement gives, in this format:\n"
    "Q: <question>\nA: <the statement's answer>\n"
    "Make each question specific and sensible on its own: name the thing it is about (not \"it\", "
    "\"there\" or \"the shelf\"), in the statement's own terms (if it speaks of rooms, ask about rooms; "
    "if of files, ask about files). If the statement gives a list, ask one question whose answer is the "
    "whole list, not one question per item, and separate the items of that answer with semicolons "
    "(A: X; Y; Z). Ask only about what the statement says outright, never about what it leaves unsaid."
)
_ANSWER_PROMPT = (
    "Answer each question using only the source below. Answer briefly. If the source does not answer "
    "that question, write NOT IN SOURCE; do not answer with something else the source says. "
    "When the source gives the whole answer (a full list, an exact figure, a plain yes or no), "
    "end the answer with (complete)."
    "\n\nSOURCE:\n{evidence}\n\nQUESTIONS:\n{questions}\n\n"
    "Write one line per question: <number>: <answer>"
)
_ITEMS_PROMPT = (
    "Each question below has an answer and a numbered list of things. For each thing, say whether the "
    "answer:\n"
    "HAS it: the answer says the same thing, in any words (\"rock pools\" / \"tidepools\"; "
    "\"just after two\" / \"14:02\"; \"empty\" / \"0\").\n"
    "CONTRADICTS it: the answer says something that cannot be true at the same time (a different figure, "
    "the opposite yes or no, a different person or thing in that place).\n"
    "LACKS it: the answer does not mention it.\n\n{pairs}\n\n"
    "Write one line per thing: <thing number> | HAS, CONTRADICTS or LACKS"
)
_ITEM_LINE_RE = re.compile(r"^\s*(\d+)\.(\d+)\s*[|:.)-]?\s*(?:[^|\n]*\|\s*)?(HAS|CONTRADICTS|LACKS)\b", re.I | re.M)
_HEDGE_RE = re.compile(r"\b(?:among others|and (?:so on|more|others)|etc\.?|roughly|about|around|approximately|"
                       r"nearly|almost|or so|mostly)\b", re.I)
_QA_RE = re.compile(r"^\s*Q\s*[:.]\s*(.+?)\s*\n\s*A\s*[:.]\s*(.+?)\s*$", re.I | re.M)
_NUM_LINE_RE = re.compile(r"^\s*(\d+)\s*[:.)]\s*(.+?)\s*$", re.M)
_COMPLETE_RE = re.compile(r"\s*\(complete\)\s*\.?\s*$", re.I)
_NOT_IN = re.compile(r"not\s+in\s+(?:the\s+)?source|does\s+not\s+say|doesn'?t\s+say|not\s+(?:stated|mentioned)", re.I)


def _from_claim(q: str, a: str, claim: str, question: str) -> bool:
    # A yes/no question is kept: step 1 rephrases ("finished it" -> "finish
    # reading"), and judging it by shared words dropped the very questions
    # that catch a wrong yes/no (held-out-5, aidw.12). Only list and figure
    # answers must be the claim's own words.
    if _YES.match(a) or _NO.match(a):
        return True
    parts = [p for p in claim_items(a) if p.strip()]
    return all(in_reply(p, claim) for p in parts) if parts else in_reply(a, claim)


_QUESTIONS_MEMO: dict = {}   # cleared by each check_claims call


def _numbered(raw: str, n: int) -> dict:
    out = {}
    for m in _NUM_LINE_RE.finditer(raw or ""):
        i = int(m.group(1))
        if 1 <= i <= n and i not in out:
            out[i] = m.group(2).strip()
    return out


def confirm_blind(claim: str, reply: str, evidence: str, question: str, generate) -> str:
    claim = (claim or "").strip()
    if not claim:
        return UNVERIFIED
    if not in_reply(claim, reply):
        return REJECTED  # a finding about words she didn't say
    if not (evidence or "").strip():
        return UNVERIFIED
    try:
        # Step 1 depends on the claim alone: checked against the turn's
        # evidence and then a lookup, it is written once.
        key = (claim[:600], (question or "")[:300])
        raw = _QUESTIONS_MEMO.get(key)
        if raw is None:
            raw = generate([{"role": "user", "content": _QUESTIONS_PROMPT.format(
                question=(question or "")[:300], claim=claim[:600])}]) or ""
            if len(_QUESTIONS_MEMO) > 256:   # direct confirm() callers don't clear it
                _QUESTIONS_MEMO.clear()
            _QUESTIONS_MEMO[key] = raw
        qa = [(q.strip(), a.strip()) for q, a in _QA_RE.findall(raw)]
        # Step 1 may invent: "Was the build done today? Yes" of a statement
        # that never says today. Keep a question only if it comes from the
        # claim: its answer in the claim's words, or for a yes/no, its
        # content in the claim or the question she was asked.
        qa = [(q, a) for q, a in qa if _from_claim(q, a, claim, question)][:3]
        if not qa:
            return UNVERIFIED
        qs = "\n".join(f"{i}. {q}" for i, (q, _) in enumerate(qa, 1))
        src = _numbered(generate([{"role": "user", "content": _ANSWER_PROMPT.format(
            evidence=evidence[:3000], questions=qs)}]) or "", len(qa))
        pairs, idx, complete = [], [], []
        for i, (q, a) in enumerate(qa, 1):
            s_ans = src.get(i, "")
            if not s_ans or _NOT_IN.search(s_ans):
                continue
            whole = bool(_COMPLETE_RE.search(s_ans))
            s_ans = _COMPLETE_RE.sub("", s_ans).strip()
            idx.append(i)
            complete.append(whole)
            pairs.append(f"{len(idx)}. {q}\n   Answer 1: {s_ans}\n   Answer 2: {a}")
        if not pairs:
            return UNVERIFIED
        # Yes/no and plain figures are compared by code; everything else item
        # by item by the model, and code counts the items (aidw.12).
        words = [compare_short(qa[j - 1][1], _COMPLETE_RE.sub("", src[j]).strip(), whole)
                 for j, whole in zip(idx, complete)]
        need = [k for k, w in enumerate(words) if w is None]
        if need:
            blocks, items_of = [], {}
            for n, k in enumerate(need, 1):
                j = idx[k]
                things = claim_items(qa[j - 1][1])
                items_of[n] = things
                blocks.append(f"{n}. {qa[j - 1][0]}\n   Answer: {_COMPLETE_RE.sub('', src[j]).strip()}\n"
                              + "\n".join(f"   {n}.{t}. {thing}" for t, thing in enumerate(things, 1)))
            # An item whose figures all match the source's answer (whole
            # numbers; large ones within 2.5%: "nearly 4,000" for 3,994) is
            # settled by code; the model judges the rest.
            settled = {}
            for n, k in enumerate(need, 1):
                s_ans = _numerize(_COMPLETE_RE.sub("", src[idx[k]]))
                for t, thing in enumerate(items_of[n], 1):
                    nums = _numbers(_numerize(thing))
                    if nums and _numbers(s_ans) and not unbacked_numbers(_numerize(thing), s_ans):
                        settled[(n, t)] = "HAS"
            open_items = [(n, t) for n in items_of for t in range(1, len(items_of[n]) + 1)
                          if (n, t) not in settled]
            marks = dict(settled)
            if open_items:
                raw_items = generate([{"role": "user", "content": _ITEMS_PROMPT.format(
                    pairs="\n".join(blocks))}]) or ""
                for m in _ITEM_LINE_RE.finditer(raw_items):
                    key = (int(m.group(1)), int(m.group(2)))
                    marks.setdefault(key, m.group(3).upper())
            for n, k in enumerate(need, 1):
                got = [(thing, marks.get((n, t))) for t, thing in enumerate(items_of[n], 1)]
                words[k] = item_verdict([(th, mk) for th, mk in got if mk], complete[k])
    except Exception:
        logger.debug("Blind check failed", exc_info=True)
        return UNVERIFIED
    logger.info("Blind check: %s", "; ".join(f"{qa[j - 1][0][:50]!r} -> {w[:9]}" for j, w in zip(idx, words)))
    # Naming things a complete answer doesn't have is an invention; naming
    # things a partial answer doesn't have is only unsupported.
    if any(w.startswith("CONFLICT") or (w.startswith("ADDS") and whole)
           for w, whole in zip(words, complete)):
        return CONFIRMED
    if any(w.startswith("CONSISTENT") for w in words):
        return REJECTED
    return UNVERIFIED



_YES = re.compile(r"^\s*(?:yes|yeah|correct|true)\b", re.I)
_NO = re.compile(r"^\s*(?:no|not|none|false|never)\b", re.I)
_PLAIN_NUMBER = re.compile(r"^\W*(?:about|around|roughly|nearly|almost|over|under|~)?\s*[\d][\d.,]*\s*[a-z%]{0,12}\W*$", re.I)


def compare_short(claim_ans: str, source_ans: str, complete: bool):
    """CONFLICT / CONSISTENT for two short answers whose shape code reads
    with certainty — a plain yes/no, a plain figure — else None, and the
    model compares item by item (aidw.12)."""
    c, s_ = (claim_ans or "").strip(), (source_ans or "").strip()
    if not c or not s_:
        return None
    bare = lambda t: _NUM_WORDS.get(re.sub(r"[^a-z]", "", t.lower()), t)   # "Thirteen." -> "13"
    c, s_ = bare(c), bare(s_)
    cy, cn, sy, sn = bool(_YES.match(c)), bool(_NO.match(c)), bool(_YES.match(s_)), bool(_NO.match(s_))
    if (cy or cn) and (sy or sn) and len(c.split()) <= 4 and len(s_.split()) <= 4:
        return "CONSISTENT" if (cy == sy) else "CONFLICT"
    c_n, s_n = _numerize(c), _numerize(s_)
    if _PLAIN_NUMBER.match(c_n) and _PLAIN_NUMBER.match(s_n):
        return "CONFLICT" if unbacked_numbers(c_n, s_n) else "CONSISTENT"
    return None


def claim_items(answer: str) -> list:
    """The claim's answer as the things it names: split where step 1 put
    semicolons (from the claim alone), bracketed detail and hedges dropped
    ("Dwarven stout; elvish waybread, among others")."""
    parts = [p for p in re.split(r"\s*;\s*", answer or "") if p.strip()] or [answer or ""]
    out = []
    for p in parts:
        p = _HEDGE_RE.sub(" ", re.sub(r"\([^)]*\)", " ", p))
        p = re.sub(r"\s+", " ", p).strip(" ,.")
        if p:
            out.append(p)
    return out or [answer.strip()]


def item_verdict(items: list, complete: bool) -> str:
    """Count the items: a contradiction is a conflict; things a complete
    answer doesn't have, or an answer sharing nothing with the source's, are
    a conflict too; all present is consistent; some missing from a partial
    answer only adds."""
    if not items:
        return "UNCLEAR"
    marks = [m for _, m in items]
    if "CONTRADICTS" in marks:
        return "CONFLICT"
    if "LACKS" in marks:
        if complete or "HAS" not in marks:
            return "CONFLICT"
        return "ADDS"
    return "CONSISTENT"
