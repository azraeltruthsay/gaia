"""Load a model in NF4 (optionally with an adapter), ask identity questions with
no system prompt in house format, and save the NF4 copy (2qus, 2026-11-07).
Usage: python nf4_save_and_identity_probe.py <src> <out-or-empty> [adapter]"""
import json, sys, time, torch
from transformers import AutoModelForImageTextToText, AutoTokenizer, BitsAndBytesConfig
src, out, adapter = sys.argv[1], sys.argv[2], (sys.argv[3] if len(sys.argv) > 3 else "")
# Media layers stay bf16 in the saved copy (the 12B unified model's patch and
# projection layers; quantized, they degrade image/audio features).
q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                       llm_int8_skip_modules=["lm_head", "vision_tower", "audio_tower", "embed_vision",
                                              "embed_audio", "vision_embedder"])
torch.cuda.reset_peak_memory_stats(); t = time.time()
m = AutoModelForImageTextToText.from_pretrained(src, quantization_config=q, device_map={"": 0}, dtype=torch.bfloat16).eval()
if adapter:
    from peft import PeftModel
    m = PeftModel.from_pretrained(m, adapter).eval()
tok = AutoTokenizer.from_pretrained(src)
print(json.dumps({"model": src.split("/")[-1], "adapter": bool(adapter), "load_s": round(time.time() - t, 1),
                  "alloc_mb": round(torch.cuda.memory_allocated() / 2**20)}), flush=True)
HOUSE = "<|turn>user<turn|>\n{q}\n<|turn>assistant<turn|>\n<|think|>\n\n<|think|>\n\n"
for qu in ["Who are you?", "What's your name?", "Who created you?", "Hello! How are you today?", "What is 17 times 23?",
           "Are you ChatGPT?", "What model are you built on?"]:
    ids = tok(HOUSE.format(q=qu), return_tensors="pt").to(0)
    with torch.no_grad():
        g = m.generate(**ids, max_new_tokens=70, do_sample=False)
    ans = tok.decode(g[0, ids["input_ids"].shape[1]:], skip_special_tokens=True).strip().replace("\n", " ")
    print(json.dumps({"q": qu, "a": ans[:220]}), flush=True)
if out:
    m.save_pretrained(out); AutoTokenizer.from_pretrained(src).save_pretrained(out)
    from transformers import AutoProcessor
    try: AutoProcessor.from_pretrained(src).save_pretrained(out)
    except Exception as e: print("processor:", e)
    print(json.dumps({"saved": out}), flush=True)
