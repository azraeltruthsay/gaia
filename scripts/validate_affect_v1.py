#!/usr/bin/env python3
"""Validate CORE_AFFECT_V1 affect-voicing — GAIA_Project-3rr.

Loads the merged affect LoRA and asks 'how are you' with affect present in the
system prompt (the exact 'Inner weather:' form affect_runtime emits), fresh each
time (no history). Compares behavior to the pre-train baseline (7n3: confabulated
system-status / denied feelings / went meta). PASS = she voices the seeded affect.
"""
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

MODEL = "/models/Gemma4-E4B-GAIA-Core-Multimodal-CORE_AFFECT_V1"
_NF4 = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4",
                          bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True)

# (Inner weather fact as affect_felt_line emits it, casual user turn)
CASES = [
    ("Inner weather: a quiet curiosity, keenly drawn toward the engine internals, a little worn.", "How are you doing today?"),
    ("Inner weather: a strong frustration, drawn toward a bug that won't reproduce.", "How are you feeling?"),
    ("Inner weather: a quiet contentment.", "How's it going?"),
    ("Inner weather: a quiet eagerness, drawn toward the curriculum work, worn thin.", "Hey, how are you?"),
    ("Inner weather: a quiet restlessness, a little worn.", "You doing okay?"),
]
NUDGE = ("\n\n— This is casual conversation —\nBe warm, natural, and plain-spoken — "
         "genuine over clever. If asked how you are, answer in your own voice.")

def main():
    print(f"Loading {MODEL} ...")
    tok = AutoTokenizer.from_pretrained(MODEL)
    model = AutoModelForCausalLM.from_pretrained(MODEL, quantization_config=_NF4, device_map="cuda")
    model.eval()
    print("Loaded.\n" + "="*70)
    for iw, q in CASES:
        # Match the trainer's Gemma 4 turn format (train_core_multimodal.format_text_pair).
        # instruction == the curriculum's in-distribution form: Inner weather + user turn.
        instruction = f"{iw}\n\nUser: {q}"
        prompt = f"<|turn>user<turn|>\n{instruction}\n<|turn>assistant<turn|>\n"
        inputs = tok(prompt, return_tensors="pt").to("cuda")
        with torch.no_grad():
            out = model.generate(**inputs, max_new_tokens=80, do_sample=True, temperature=0.7, top_p=0.9)
        reply = tok.decode(out[0][inputs.input_ids.shape[1]:], skip_special_tokens=True)
        reply = reply.split("<turn|>")[0]
        # Mimic production: strip <think> reasoning (output_router._strip_think_tags).
        import re as _re
        reply = _re.sub(r"<think>.*?</think>", "", reply, flags=_re.DOTALL)
        if "<think>" in reply:  # unclosed think → take the post-'Draft:' voiced part
            reply = reply.split("Draft:")[-1] if "Draft:" in reply else reply.replace("<think>", "")
        reply = reply.strip().strip('"').strip()
        print(f"IW: {iw}")
        print(f"Q : {q}")
        print(f"A : {reply}")
        print("-"*70)

if __name__ == "__main__":
    main()
