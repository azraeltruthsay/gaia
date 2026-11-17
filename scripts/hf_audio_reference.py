#!/usr/bin/env python3
"""HF reference for Core's hearing: the same clips through transformers, not gaia_cpp.

Is a wrong answer the model's or our llama.cpp audio path's? This loads Core's NF4
weights (/models/core) with the stock chat template and asks each clip the battery's
question, greedy. Run it where the GPU has ~9 GB free (gear 2: Prime ~4.6 GB), e.g.

    docker exec gaia-study python /gaia/GAIA_Project/scripts/hf_audio_reference.py \\
        --clips /gaia/GAIA_Project/candidates/gaia-doctor/test_assets/audio --out /shared/hf_audio_ref.json

Prints one line per clip; --out keeps them for comparing with media_gate.py --dump.
"""
import argparse
import json
import os
import time

QUESTIONS = {
    "esc_": "What is this sound? Be brief.",
    "podcast_": "Who is speaking in this clip, and what do they say? Be brief.",
}
DEFAULT_Q = "Listen to this audio and describe what you hear. Be brief."


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", default="/models/core")
    p.add_argument("--clips", required=True)
    p.add_argument("--only", default="", help="comma list of filename prefixes (default: all .wav)")
    p.add_argument("--out", default="")
    p.add_argument("--max-new-tokens", type=int, default=80)
    p.add_argument("--nf4", action="store_true", help="quantize to NF4 on load (for full-precision weights)")
    a = p.parse_args()

    import torch
    from transformers import AutoModelForImageTextToText, AutoProcessor

    t0 = time.time()
    proc = AutoProcessor.from_pretrained(a.model)
    kw = {}
    if a.nf4:
        from transformers import BitsAndBytesConfig
        kw["quantization_config"] = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
            llm_int8_skip_modules=["audio_tower", "vision_tower", "embed_audio", "embed_vision", "lm_head"])
    model = AutoModelForImageTextToText.from_pretrained(a.model, device_map="cuda:0", dtype=torch.bfloat16, **kw)
    model.eval()
    print(f"loaded {a.model} in {time.time() - t0:.0f}s, "
          f"{torch.cuda.memory_allocated() / 2**30:.1f} GiB", flush=True)

    prefixes = [x for x in a.only.split(",") if x]
    files = sorted(f for f in os.listdir(a.clips) if f.endswith(".wav")
                   and (not prefixes or any(f.startswith(x) for x in prefixes)))
    out = {}
    for f in files:
        q = next((v for k, v in QUESTIONS.items() if f.startswith(k)), DEFAULT_Q)
        msgs = [{"role": "user", "content": [
            {"type": "audio", "audio": os.path.join(a.clips, f)},
            {"type": "text", "text": q}]}]
        inputs = proc.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True,
                                          return_dict=True, return_tensors="pt").to(model.device)
        with torch.no_grad():
            gen = model.generate(**inputs, max_new_tokens=a.max_new_tokens, do_sample=False)
        reply = proc.decode(gen[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        out[f] = {"question": q, "reply": reply}
        print(f"{f:28} -> {reply[:110]!r}", flush=True)
    if a.out:
        with open(a.out, "w") as fh:
            json.dump(out, fh, indent=1, ensure_ascii=False)


if __name__ == "__main__":
    main()
