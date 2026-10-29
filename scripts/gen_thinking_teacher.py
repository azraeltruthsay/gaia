"""Teacher data for the thinking adapters: a stock model's native thought traces
on GAIA's real prompts.

For each real user message, build the prompt GAIA would assemble
(prompt_builder.build_from_packet + tool schema when offered), format it in the
engine's turn format for the tier, and keep the teacher's thought and the
answer after it (both are trained on — see train_core_thinking.py).

  --tier core  (h1ok, core_thinking_v1)
      Teacher: stock google/gemma-4-E4b-it (Q8 GGUF). <|think|> opens the system
      turn; the thought comes back as <|channel>thought ...<channel|>. Packets
      keep the default header, as the shipped adapter was trained (see bead
      w6in: live Core deliberation gets the compact 'core' prompt instead).
  --tier prime (isep, prime_thinking_v1)
      Teacher: stock Qwen/Qwen3-VL-8B-Thinking (Q8 GGUF). Packets are named
      'prime' so prompt_builder emits Prime's real prompt; ChatML turns; the
      assistant turn is prefilled with <think>\\n as Qwen's Thinking template
      does, and the thought ends at </think>.

Teacher server: llama-server on :8199 inside gaia-core with the teacher GGUF on
the GPU (GAIA in maintenance + deep-sleep), e.g.
    llama-server -m /shared/tmp_h1ok/<teacher>.gguf --port 8199 -c 32768 -np 4 -ngl 99
Inputs: /shared/tmp_h1ok/user_inputs.json — distinct real user messages from
/shared/sessions*.json plus untruncated deliberation-journal inputs.
Usage: python gen_thinking_teacher.py [--tier core|prime] [limit]   (in gaia-core)
"""
import argparse, json, logging, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/app")
logging.disable(logging.WARNING)
from gaia_common.protocols.cognition_packet import CognitionPacket, Content, Intent
from gaia_core.utils import prompt_builder
from gaia_common.utils.tool_call_parser import build_tool_schema_injection

URL = "http://localhost:8199/completion"
GREET = re.compile(r"^\s*(hi|hello|hey|good (morning|afternoon|evening|night)|morning|evening|how are you|how's it going|thanks|thank you)\b", re.I)

TIERS = {
    "core": {
        "out": "/shared/tmp_h1ok/teacher.jsonl",
        "header_name": None,            # default header, as core_thinking_v1 was trained
        "think_in_system": "<|think|>\n",
        "prefill": "",
        "thought_re": re.compile(r"\s*<\|channel>thought\n(.*?)<channel\|>(.*)", re.S),
        "stop": ["<turn|>", "<|turn>"],
        "n_predict": 1100, "sampling": {"temperature": 0.6, "top_p": 0.95, "top_k": 64},
    },
    "prime": {
        "out": "/shared/tmp_h1ok/teacher_prime.jsonl",
        "header_name": "prime",
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


def render(tier, messages):
    """The engine's turn format for the tier (ChatFormatter / gaia_cpp)."""
    if tier == "core":
        parts = [f"<|turn>{m['role']}<turn|>\n{m['content']}" for m in messages]
        return "\n".join(parts) + "\n<|turn>assistant<turn|>\n"
    parts = [f"<|im_start|>{m['role']}\n{m['content']}<|im_end|>" for m in messages]
    return "\n".join(parts) + "\n<|im_start|>assistant\n"


def build(tier, text):
    cfg = TIERS[tier]
    intent = intent_for(text)
    pkt = CognitionPacket(content=Content(original_prompt=text), intent=Intent(user_intent=intent))
    if cfg["header_name"]:
        pkt.header.model.name = cfg["header_name"]
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


def one(tier, text):
    cfg = TIERS[tier]
    try:
        intent, msgs = build(tier, text)
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
    inputs = json.load(open("/shared/tmp_h1ok/user_inputs.json"))
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
