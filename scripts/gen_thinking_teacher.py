"""h1ok teacher data: stock Gemma 4 E4B-it native thought traces on GAIA's real prompts.

For each real user message, build the prompt GAIA would assemble
(prompt_builder.build_from_packet + tool schema when offered), put <|think|> at
the top of the system turn, format it in the engine's house dialect, and keep
the teacher's <|channel>thought ...<channel|> trace and the answer after it
(both are trained on — see train_core_thinking.py).

Teacher: stock google/gemma-4-E4b-it as a Q8 GGUF, served by llama-server on
:8199 inside gaia-core (GAIA in maintenance + deep-sleep so the GPU is free):
    convert_hf_to_gguf.py /models/google/gemma-4-E4b-it --outtype q8_0 \
        --outfile /shared/tmp_h1ok/gemma4-e4b-it-q8.gguf          # in gaia-study
    llama-server -m /shared/tmp_h1ok/gemma4-e4b-it-q8.gguf --port 8199 \
        -c 32768 -np 4 -ngl 99                                      # in gaia-core
Inputs: /shared/tmp_h1ok/user_inputs.json — distinct real user messages from
/shared/sessions*.json plus untruncated deliberation-journal inputs.
Usage: python gen_thinking_teacher.py [limit]   (in gaia-core)
"""
import json, logging, re, sys, time, urllib.request
from concurrent.futures import ThreadPoolExecutor
sys.path.insert(0, "/app")
logging.disable(logging.WARNING)
from gaia_common.protocols.cognition_packet import CognitionPacket, Content, Intent
from gaia_core.utils import prompt_builder
from gaia_common.utils.tool_call_parser import build_tool_schema_injection

URL = "http://localhost:8199/completion"
OUT = "/shared/tmp_h1ok/teacher.jsonl"
GREET = re.compile(r"^\s*(hi|hello|hey|good (morning|afternoon|evening|night)|morning|evening|how are you|how's it going|thanks|thank you)\b", re.I)

def intent_for(text):
    return "greeting" if GREET.search(text) and len(text) < 80 else "chat"

def house(messages):
    parts = [f"<|turn>{m['role']}<turn|>\n{m['content']}" for m in messages]
    return "\n".join(parts) + "\n<|turn>assistant<turn|>\n"

def build(text):
    intent = intent_for(text)
    pkt = CognitionPacket(content=Content(original_prompt=text), intent=Intent(user_intent=intent))
    msgs = [dict(m) for m in prompt_builder.build_from_packet(pkt)]
    msgs = [m for m in msgs if isinstance(m.get("content"), str)]
    if prompt_builder.offers_tools(intent):
        schema = build_tool_schema_injection()
        if schema and msgs and msgs[0]["role"] == "system":
            msgs[0]["content"] = msgs[0]["content"].rstrip() + "\n\n---\n" + schema
    if msgs and msgs[0]["role"] == "system":
        msgs[0]["content"] = "<|think|>\n" + msgs[0]["content"]
    else:
        msgs.insert(0, {"role": "system", "content": "<|think|>"})
    if not msgs or msgs[-1]["role"] != "user":
        msgs.append({"role": "user", "content": text})
    return intent, msgs

def one(text):
    try:
        intent, msgs = build(text)
        body = {"prompt": house(msgs), "n_predict": 1100, "temperature": 0.6, "top_p": 0.95,
                "top_k": 64, "special": True, "cache_prompt": True, "stop": ["<turn|>", "<|turn>"]}
        t = time.time()
        r = json.load(urllib.request.urlopen(urllib.request.Request(
            URL, json.dumps(body).encode(), {"Content-Type": "application/json"}), timeout=900))
        out = r["content"]
        m = re.match(r"\s*<\|channel>thought\n(.*?)<channel\|>(.*)", out, re.S)
        return {"user": text, "intent": intent, "messages": msgs,
                "thought": m.group(1).strip() if m else None,
                "answer": m.group(2).strip() if m else out.strip(),
                "closed": bool(m), "n": r.get("tokens_predicted"), "s": round(time.time() - t, 1)}
    except Exception as e:
        return {"user": text, "error": str(e)}

def main():
    inputs = json.load(open("/shared/tmp_h1ok/user_inputs.json"))
    lim = int(sys.argv[1]) if len(sys.argv) > 1 else len(inputs)
    done = closed = 0
    t0 = time.time()
    with open(OUT, "w") as f, ThreadPoolExecutor(4) as ex:
        for rec in ex.map(one, inputs[:lim]):
            f.write(json.dumps(rec) + "\n")
            f.flush()
            done += 1
            closed += bool(rec.get("closed"))
            if done % 25 == 0 or done == lim:
                print(f"{done}/{lim} closed={closed} {time.time()-t0:.0f}s", flush=True)


if __name__ == "__main__":
    main()
