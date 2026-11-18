"""Teacher data for the thinking adapters: a stock model's native thought traces
on GAIA's real prompts.

For each real user message, build the prompt GAIA would assemble
(prompt_builder.build_from_packet + tool schema when offered), format it in the
engine's turn format for the tier, and keep the teacher's thought and the
answer after it (both are trained on — see train_core_thinking.py).

  --tier core  (h1ok core_thinking_v1; w6in core_thinking_v2 for the 12B)
      Teacher: stock google/gemma-4-12B-it (GGUF), asked in Google's turn
      format. <|think|> opens the system turn; the thought comes back as
      <|channel>thought ...<channel|>. Packets are named 'core' so the prompt
      is the one live Core deliberation gets (v1 used the default header).
  --tier prime (isep, prime_thinking_v1)
      Teacher: stock Qwen/Qwen3-VL-8B-Thinking (Q8 GGUF). Packets are named
      'prime' so prompt_builder emits Prime's real prompt; ChatML turns; the
      assistant turn is prefilled with <think>\\n as Qwen's Thinking template
      does, and the thought ends at </think>. Intents come from the live
      classifier (embedding stage, as agent_core runs it), so identity
      questions get identity prompts — the keyword heuristic labelled nearly
      everything 'chat', and casual prompts dropped Prime's architecture fact
      before 6rfx (round 1's teacher answered identity questions without it).

Teacher server: llama-server on :8199 inside gaia-core with the teacher GGUF on
the GPU (GAIA in maintenance + deep-sleep), e.g.
    llama-server -m /shared/tmp_h1ok/<teacher>.gguf --port 8199 -c 32768 -np 4 -ngl 99
Inputs: /shared/tmp_h1ok/turns.json (scripts/build_thinking_turns.py: real turns
with session and history), then /shared/tmp_h1ok/user_inputs.json — distinct
real user messages and untruncated deliberation-journal inputs — as first messages.
Usage: python gen_thinking_teacher.py [--tier core|prime] [limit]   (in gaia-core)
"""
import argparse, json, logging, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/app")
logging.disable(logging.WARNING)
from gaia_common.protocols.cognition_packet import (CognitionPacket, Content, DataField, Intent,
                                                      RelevantHistorySnippet)
from gaia_core.utils import prompt_builder
from gaia_common.utils.tool_call_parser import build_tool_schema_injection

URL = "http://localhost:8199/completion"
GREET = re.compile(r"^\s*(hi|hello|hey|good (morning|afternoon|evening|night)|morning|evening|how are you|how's it going|thanks|thank you)\b", re.I)

TIERS = {
    "core": {
        # w6in: Core's own header ('core'), so prompt_builder gives the prompt
        # live Core deliberation gets; v1 trained on the default (Prime's).
        "out": "/shared/tmp_h1ok/teacher_core12b.jsonl",
        "header_name": "core",
        "live_intents": True,
        "think_in_system": "<|think|>\n",
        "prefill": "",
        # Teacher: stock gemma-4-12B-it, which never answers in the house
        # format (prompt_format.py), so it is asked in Google's own.
        "teacher_format": "stock",
        "thought_re": re.compile(r"\s*<\|channel>thought\n(.*?)<channel\|>(.*)", re.S),
        "stop": ["<turn|>", "<|turn>"],
        "n_predict": 1100, "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 64},
    },
    "prime": {
        "out": "/shared/tmp_h1ok/teacher_prime_v2b.jsonl",
        "header_name": "prime",
        "live_intents": True,
        "think_in_system": "",
        "prefill": "<think>\n",
        "thought_re": re.compile(r"(.*?)</think>(.*)", re.S),
        "stop": ["<|im_end|>", "<|im_start|>"],
        # Qwen's recommended sampling for its Thinking models
        "n_predict": 3000, "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 20},
    },
}


def intent_for(text):
    return "greeting" if GREET.search(text) and len(text) < 80 else "chat"


_LIVE = {}


def live_intent_for(text):
    """agent_core's intent detection without an LLM: the embedding classifier
    (the same MiniLM the model pool serves) plus its heuristics."""
    if not _LIVE:
        from sentence_transformers import SentenceTransformer
        from gaia_core.config import Config
        from gaia_core.cognition.nlu.intent_detection import detect_intent
        _LIVE.update(embed=SentenceTransformer("/models/all-MiniLM-L6-v2", device="cpu"),
                     config=Config(), detect=detect_intent)
    try:
        return _LIVE["detect"](text, _LIVE["config"], embed_model=_LIVE["embed"]).intent or "chat"
    except Exception:
        return intent_for(text)


def render(tier, messages):
    """The turn format the tier's teacher is asked in: the engine's house
    format, or Google's own for the stock 12B (it thinks in no other)."""
    if TIERS[tier].get("teacher_format") == "stock":
        out = []
        for i, m in enumerate(messages):
            role = "model" if m["role"] == "assistant" else m["role"]
            out.append(f"<|turn>{role}\n{m['content']}<turn|>\n")
        return "".join(out) + "<|turn>model\n"
    if tier == "core":
        parts = [f"<|turn>{m['role']}<turn|>\n{m['content']}" for m in messages]
        return "\n".join(parts) + "\n<|turn>assistant<turn|>\n"
    parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages]
    return "\n".join(parts) + "\n<|im_start|>assistant\n"


def _grounding(text):
    """What live grounding attaches for a question she looked up before
    (0vuz: remembered lookups, as agent_core attaches them). No web."""
    try:
        from gaia_core.cognition.knowledge_router import _query_lookup_memory
        hits = [r for r in _query_lookup_memory(text, top_k=2) if not r.stale]
    except Exception:
        return None
    if not hits:
        return None
    return DataField(key="web_grounding", type="json", value={text[:50]: {"results": [
        {"title": r.title or r.source, "snippet": r.content, "url": r.url, "remembered": True,
         "retrieved_at": r.retrieved_at, "stale": r.stale, "memory_id": r.memory_id,
         "trust": r.trust_tier} for r in hits]}})


def build(tier, turn):
    """The deliberation prompt for a real turn: `turn` is a user message, or
    a dict from build_thinking_turns.py with its session and history (b9mp:
    live THINKER prompts carry history, speaker, persona and grounding)."""
    cfg = TIERS[tier]
    if isinstance(turn, str):
        turn = {"user": turn}
    text = turn["user"]
    intent = live_intent_for(text) if cfg["live_intents"] else intent_for(text)
    pkt = CognitionPacket(content=Content(original_prompt=text), intent=Intent(user_intent=intent))
    if cfg["header_name"]:
        pkt.header.model.name = cfg["header_name"]
    if turn.get("session_id"):
        pkt.header.session_id = turn["session_id"]
    if turn.get("persona"):
        pkt.header.persona.persona_id = turn["persona"]
    pkt.context.relevant_history_snippet = [
        RelevantHistorySnippet(id=f"h{i}", role=h["role"], summary=h["content"])
        for i, h in enumerate(turn.get("history") or [])]
    if intent not in ("greeting", "farewell", "gratitude", "smalltalk", "chat"):
        g = _grounding(text)
        if g is not None:
            pkt.content.data_fields.append(g)
    msgs = [dict(m) for m in prompt_builder.build_from_packet(pkt)]
    msgs = [m for m in msgs if isinstance(m.get("content"), str)]
    if prompt_builder.offers_tools(intent):
        schema = build_tool_schema_injection()
        if schema and msgs and msgs[0]["role"] == "system":
            msgs[0]["content"] = msgs[0]["content"].rstrip() + "\n\n---\n" + schema
    if cfg["think_in_system"]:
        if msgs and msgs[0]["role"] == "system":
            msgs[0]["content"] = cfg["think_in_system"] + msgs[0]["content"]
        else:
            msgs.insert(0, {"role": "system", "content": cfg["think_in_system"].strip()})
    if not msgs or msgs[-1]["role"] != "user":
        msgs.append({"role": "user", "content": text})
    return intent, msgs


def one(tier, turn):
    cfg = TIERS[tier]
    text = turn["user"] if isinstance(turn, dict) else turn
    try:
        intent, msgs = build(tier, turn)
        body = {"prompt": render(tier, msgs) + cfg["prefill"], "n_predict": cfg["n_predict"],
                "special": True, "cache_prompt": True, "stop": cfg["stop"], **cfg["sampling"]}
        t = time.time()
        r = json.load(urllib.request.urlopen(urllib.request.Request(
            URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=1800))
        out = r["content"]
        m = cfg["thought_re"].match(out)
        return {"user": text, "intent": intent, "messages": msgs,
                "thought": m.group(1).strip() if m else None,
                "answer": m.group(2).strip() if m else out.strip(),
                "closed": bool(m), "n": r.get("tokens_predicted"), "s": round(time.time() - t, 1)}
    except Exception as e:
        return {"user": text, "error": str(e)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--tier", choices=sorted(TIERS), default="core")
    ap.add_argument("--parallel", type=int, default=4)
    ap.add_argument("limit", nargs="?", type=int)
    args = ap.parse_args()
    # Real turns with their context first, then the journal's inputs as
    # first messages (no history), which live prompts also are.
    turns = json.load(open("/shared/tmp_h1ok/turns.json"))
    have = {t["user"].strip().lower() for t in turns}
    inputs = turns + [t for t in json.load(open("/shared/tmp_h1ok/user_inputs.json"))
                      if t.strip().lower() not in have]
    lim = args.limit or len(inputs)
    done = closed = 0
    t0 = time.time()
    with open(TIERS[args.tier]["out"], "w") as f, ThreadPoolExecutor(args.parallel) as ex:
        for rec in ex.map(lambda t: one(args.tier, t), inputs[:lim]):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done += 1
            closed += bool(rec.get("closed"))
            if done % 25 == 0 or done == lim:
                print(f"{done}/{lim} closed={closed} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
