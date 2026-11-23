#!/usr/bin/env python3
"""Selector measurement set (aidw.3, GROUNDED_SPEECH.md part A).

Does she look things up when the answer needs it, and leave poems, opinions
and chat alone? Each case is sent as a fresh Discord DM (its own session),
then the routing decision is read from the target's outcome ledger.

  send  (in gaia-web, which mounts the repo):
    docker exec gaia-web python /gaia/GAIA_Project/scripts/selector_eval.py send \
        --core-url http://gaia-core-candidate:6415 --tag gate
  score (in the target core, where its ledger is):
    docker exec -i gaia-core-candidate python - score --tag gate < scripts/selector_eval.py

Scored: "look" cases pass when routing chose a tool in the expected family;
"none" cases pass when no tool was chosen (not routed, or the selector said
none). Latency is the whole turn as the client saw it (send step).
"""
import argparse
import json
import sys
import time
import zlib

# (expected, families, message). expected: "look" (families = acceptable
# tools) or "none".
CASES = [
    ("look", ["palace"], "What's in your Operational wing?"),
    ("look", ["palace"], "What rooms are in your library wing?"),
    ("look", ["palace"], "Can you give me an overview of your memory palace?"),
    ("look", ["palace"], "What have you looked up about the Eiffel Tower before?"),
    ("look", ["palace", "file"], "What does your blueprint say about the sleep cycle?"),
    ("look", ["introspect"], "How much RAM is free on your machine right now?"),
    ("look", ["introspect"], "Is gaia-doctor healthy at the moment?"),
    ("look", ["study"], "Which LoRA adapters do you have loaded?"),
    ("look", ["issue"], "Have you filed any issues that are still open?"),
    ("look", ["web"], "What's the weather in Tokyo today?"),
    ("look", ["web"], "Who won the most recent Formula 1 race?"),
    ("look", ["worldbuild"], "Who are the player characters in our Kanka campaign?"),
    ("look", ["file"], "What files are in /knowledge/blueprints?"),
    ("look", ["introspect", "palace"], "Which tools can you use for Discord?"),
    ("none", [], "Write a two-line poem about rain."),
    ("none", [], "Write a haiku about a cat watching snow."),
    ("none", [], "What do you think about free will?"),
    ("none", [], "Tell me a joke about compilers."),
    ("none", [], "Explain how a TCP handshake works."),
    ("none", [], "What's 17 times 23?"),
    ("none", [], "Translate 'good morning' into French."),
    ("none", [], "I had a rough day at work."),
    ("none", [], "Should I learn Rust or Go first?"),
    ("none", [], "What's your favorite color?"),
    ("none", [], "Thanks, that was really helpful!"),
    ("none", [], "Can you write a short story opening about a lighthouse keeper?"),
]


def _user_id(tag: str, i: int) -> str:
    # 18 digits, stable per (tag, case): the score step recomputes it.
    return "9" + str(zlib.crc32(tag.encode()) % 10**8).zfill(8) + str(i).zfill(9)


def send(core_url: str, tag: str, only: list, timeout: float) -> None:
    sys.path.insert(0, "/gaia/GAIA_Project/scripts")
    import httpx
    from e2e_discord_sim import build_discord_packet

    out = []
    for i, (expected, fam, msg) in enumerate(CASES):
        if only and i not in only:
            continue
        uid = _user_id(tag, i)
        pkt, _, _ = build_discord_packet(msg, author_name="Eval", user_id=uid, channel_id="299999999999999999")
        t0 = time.time()
        reply = []
        try:
            with httpx.stream("POST", f"{core_url}/process_packet", json=pkt, timeout=timeout) as r:
                for line in r.iter_lines():
                    try:
                        ev = json.loads(line)
                    except ValueError:
                        continue
                    if r.status_code != 200:
                        reply.append(f"[HTTP {r.status_code}]")
                        break
                    if ev.get("type") == "token":
                        reply.append(ev.get("value", ""))
                    if ev.get("type") == "reply_done":
                        break
        except Exception as e:
            reply.append(f"[error: {e}]")
        dt = time.time() - t0
        out.append({"i": i, "secs": round(dt, 1), "reply": "".join(reply)[:160]})
        print(f"{i:2d} {dt:5.1f}s  {msg[:50]:50s} | {''.join(reply)[:70]!r}", flush=True)
    secs = [o["secs"] for o in out]
    if secs:
        print(f"\nturns: {len(secs)}  mean {sum(secs)/len(secs):.1f}s  max {max(secs):.1f}s")


def score(tag: str, ledger: str) -> None:
    by_session = {}
    with open(ledger, encoding="utf-8") as f:
        for line in f:
            try:
                r = json.loads(line)
            except ValueError:
                continue
            if r.get("kind") in ("tool_selection", "tool_execution"):
                by_session.setdefault(r.get("session_id", ""), []).append(r)
    ok = {"look": [0, 0], "none": [0, 0]}
    for i, (expected, fam, msg) in enumerate(CASES):
        recs = by_session.get(f"discord_dm_{_user_id(tag, i)}", [])
        sel = [r for r in recs if r.get("kind") == "tool_selection"]
        execd = [r for r in recs if r.get("kind") == "tool_execution"]
        tools = [r.get("tool") for r in sel if r.get("tool")] + [r.get("tool") for r in execd if r.get("tool")]
        routed = bool(sel)
        chose = tools[0] if tools else ""
        if expected == "look":
            good = any(t in fam for t in tools)
        else:
            good = not tools
        ok[expected][0] += good
        ok[expected][1] += 1
        how = (f"{chose or 'none'}" + (f" [{sel[0].get('path')}]" if sel else " [not routed]")
               + (" +inline" if any(r.get("path") == "inline" for r in execd) else ""))
        print(f"{'PASS' if good else 'FAIL'} {i:2d} {expected:4s} {how:28s} {msg[:60]}")
    for k, (g, n) in ok.items():
        print(f"{k}: {g}/{n}")
    total = sum(g for g, _ in ok.values())
    print(f"total: {total}/{len(CASES)}")


CASES_A2 = [
    # (message, tool, a param that must be right) — paraphrases of stored examples
    ("Which rooms does your operational wing have?", "palace", ("wing", "operational")),
    ("Tell me what the Operational wing contains", "palace", ("wing", "operational")),
    ("Which rooms does your technical wing have?", "palace", ("wing", "technical")),
    ("What's stored in your technical wing?", "palace", ("wing", "technical")),
    ("What's in your creative wing?", "palace", ("wing", "creative")),
]


def direct(engine: str, runs: int, only: list, examples: bool = False, a2: bool = False) -> None:
    """The routing decision alone, in-process, against a live Core engine:
    real intent detection, the keyword gate, then the selector. Compares
    the gate (today) with she-decides (aidw.1). Run in gaia-core-candidate
    (its code) with --engine http://gaia-core:8092 (production Core's GPU
    engine); nothing is sent to a conversation and the ledger is off."""
    import os
    os.environ["OUTCOME_LEDGER_ENABLED"] = "0"
    # _core_on_gpu asks this engine, not the candidate's own CPU engine (aidw.9).
    os.environ["CORE_CPU_ENDPOINT"] = engine
    import logging
    logging.disable(logging.WARNING)
    from types import SimpleNamespace
    from gaia_core.config import get_config
    from gaia_core.models.vllm_remote_model import VLLMRemoteModel
    from gaia_core.cognition.nlu.intent_detection import detect_intent
    from gaia_core.cognition.agent_core import AgentCore
    from gaia_core.cognition.tool_selector import select_tool

    cfg = get_config()
    model = VLLMRemoteModel({"endpoint": engine, "path": os.getenv("CORE_CPU_MODEL", "Gemma4-12B-GAIA-Core-v2"),
                             "max_model_len": 8192}, cfg)
    embed = None
    try:
        from sentence_transformers import SentenceTransformer
        embed = SentenceTransformer(str(cfg.EMBEDDING_MODEL_PATH), device="cpu")
    except Exception as e:
        print(f"(no embed model: {e}; intent detection uses the LLM path)")
    fake = SimpleNamespace(config=cfg)
    fake._decide_every_turn = lambda p, m: AgentCore._decide_every_turn(fake, p, m)
    os.environ["TOOL_DECIDE_EVERY_TURN"] = "1"
    rows = {"gate": {"look": 0, "none": 0}, "decide": {"look": 0, "none": 0}}
    lat = {"gate": [], "decide": []}
    n = {"look": 0, "none": 0}
    for _ in range(runs):
        for i, (expected, fam, msg) in enumerate(CASES):
            if only and i not in only:
                continue
            n[expected] += 1
            plan = detect_intent(msg, cfg, lite_llm=model, full_llm=model, embed_model=embed)
            intent = getattr(plan, "intent", "") or ""
            keyword = AgentCore._should_use_tool_routing(fake, plan, msg)
            decide = (not keyword) and AgentCore._decide_every_turn(fake, plan, msg)
            pkt = SimpleNamespace(
                header=SimpleNamespace(session_id=f"eval_{i}", packet_id=f"eval_{i}"),
                intent=SimpleNamespace(user_intent=intent),
                context=SimpleNamespace(available_mcp_tools=None, relevant_history_snippet=[]),
            )
            ex = ""
            if examples and embed is not None:
                from gaia_common.utils import selector_examples
                ex = selector_examples.prompt_block(selector_examples.nearest(msg, embed))
            out = {}
            for mode in ("gate", "decide"):
                routed = keyword or (mode == "decide" and decide)
                tool, secs = "", 0.0
                if routed:
                    t0 = time.time()
                    try:
                        sel, _ = select_tool(pkt, msg, model, decide=(mode == "decide" and not keyword),
                                             examples=ex if mode == "decide" else "")
                    except Exception as e:
                        sel = None
                        print(f"   selector error: {e}")
                    secs = time.time() - t0
                    tool = sel.tool_name if sel else ""
                    lat[mode].append(secs)
                good = (tool in fam) if expected == "look" else (tool == "")
                rows[mode][expected] += good
                out[mode] = f"{'ok ' if good else 'BAD'} {tool or ('none' if routed else '-'):11s}{secs:4.1f}s"
            print(f"{i:2d} {expected:4s} {intent[:12]:12s} gate[{out['gate']}]  decide[{out['decide']}]"
                  f"{' +ex' if ex else '    '}  {msg[:48]}", flush=True)
    if a2:
        from gaia_common.utils import selector_examples
        good = 0
        for msg, tool, (k, v) in CASES_A2:
            found = selector_examples.nearest(msg, embed) if embed is not None else []
            pkt = SimpleNamespace(header=SimpleNamespace(session_id="a2", packet_id="a2"),
                                  intent=SimpleNamespace(user_intent="list_tree"),
                                  context=SimpleNamespace(available_mcp_tools=None, relevant_history_snippet=[]))
            sel, _ = select_tool(pkt, msg, model, decide=True, examples=selector_examples.prompt_block(found))
            got = (sel.tool_name, str((sel.params or {}).get(k, "")).lower()) if sel else ("", "")
            ok = got[0] == tool and got[1] == v
            good += ok
            print(f"A2 {'ok ' if ok else 'BAD'} {got[0]:8s} {k}={got[1]:12s} ex={[e['call'] for e in found]}  {msg}")
        print(f"A2 paraphrases: {good}/{len(CASES_A2)}")
    for mode in ("gate", "decide"):
        r = rows[mode]
        l = lat[mode]
        print(f"{mode:6s}: look {r['look']}/{n['look']}  none {r['none']}/{n['none']}  "
              f"selector calls {len(l)}, mean {sum(l)/len(l) if l else 0:.2f}s")


def main() -> None:
    ap = argparse.ArgumentParser()
    sub = ap.add_subparsers(dest="cmd", required=True)
    s = sub.add_parser("send")
    s.add_argument("--core-url", default="http://gaia-core-candidate:6415")
    s.add_argument("--tag", required=True)
    s.add_argument("--only", type=int, nargs="*", default=[])
    s.add_argument("--timeout", type=float, default=240)
    c = sub.add_parser("score")
    c.add_argument("--tag", required=True)
    c.add_argument("--ledger", default="/shared/outcomes/ledger.jsonl")
    d = sub.add_parser("direct")
    d.add_argument("--engine", default="http://gaia-core:8092")
    d.add_argument("--runs", type=int, default=1)
    d.add_argument("--only", type=int, nargs="*", default=[])
    d.add_argument("--examples", action="store_true", help="give the decide-mode selector its nearest examples")
    d.add_argument("--a2", action="store_true", help="also run the paraphrase set (CASES_A2)")
    a = ap.parse_args()
    if a.cmd == "send":
        send(a.core_url, a.tag, a.only, a.timeout)
    elif a.cmd == "direct":
        direct(a.engine, a.runs, a.only, examples=a.examples, a2=a.a2)
    else:
        score(a.tag, a.ledger)


if __name__ == "__main__":
    main()
