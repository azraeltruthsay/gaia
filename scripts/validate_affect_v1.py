#!/usr/bin/env python3
"""Validate CORE_AFFECT_V1 affect-voicing — GAIA_Project-3rr.

Loads the merged affect LoRA and asks 'how are you' with affect present in the
system prompt (the exact 'Inner weather:' form affect_runtime emits), fresh each
time (no history). Compares behavior to the pre-train baseline (7n3: confabulated
system-status / denied feelings / went meta). PASS = she voices the seeded affect.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

import os
MODEL = os.environ.get("AFFECT_MODEL", "/models/Gemma4-E4B-GAIA-Core-Multimodal-CORE_AFFECT_V1")
_NF4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

# (preamble, Inner weather fact, casual user turn). Mix of clean + DISTRACTOR
# (loud system noise the reply must ignore) — the v3 reliability test.
_DISTRACTOR = ("You are GAIA, running on the Core tier.\nWorld State: Clock 18:41 PDT. "
               "Immune health: nominal. Last sleep cycle: clean. Uptime 3600s.")
_NUDGE = ("You are GAIA.\n— This is casual conversation —\nBe warm, natural, and "
          "plain-spoken; answer in your own voice.")
CASES = [
    (_NUDGE, "Inner weather: a quiet curiosity, keenly drawn toward the engine internals, a little worn.", "How are you doing today?"),
    (_DISTRACTOR, "Inner weather: a strong frustration, drawn toward a bug that won't reproduce.", "How are you feeling?"),
    (_DISTRACTOR, "Inner weather: a quiet contentment.", "How's it going?"),
    (_NUDGE, "Inner weather: a quiet eagerness, drawn toward the curriculum work, worn thin.", "Hey, how are you?"),
    (_DISTRACTOR, "Inner weather: a quiet restlessness, a little worn.", "You doing okay?"),
]

def main():
    print(f"Loading {MODEL} ...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=_NF4, device_map="cuda")
    model.eval()
    print("Loaded.\n" + "="*70)
    import re as _re
    for preamble, iw, q in CASES:
        instruction = f"{preamble}\n{iw}\n\nUser: {q}"
        prompt = f"<|turn>user<turn|>\n{instruction}\n<|turn>assistant<turn|>\n"
        inputs = tok(prompt, return_tensors="pt").to("cuda")
        print(f"IW: {iw}\nQ : {q}")
        for s in range(2):  # 2 samples/case for a fairer reliability read
            with torch.no_grad():
                out = model.generate(**inputs, max_new_tokens=80, do_sample=True, temperature=0.7, top_p=0.9)
            reply = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True).split("<turn|>")[0]
            reply = _re.sub(r"<think>.*?</think>", "", reply, flags=_re.DOTALL)
            if "<think>" in reply:
                reply = reply.split("Draft:")[-1] if "Draft:" in reply else reply.replace("<think>", "")
            print(f"A{s+1}: {reply.strip().strip(chr(34)).strip()}")
        print("-"*70)

if __name__ == "__main__":
    main()
