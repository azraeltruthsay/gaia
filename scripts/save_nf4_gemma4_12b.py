import time, torch, json
from transformers import AutoModelForImageTextToText, AutoProcessor, BitsAndBytesConfig
src, out = "/models/gemma-4-12B-it", "/models/gemma-4-12B-it-NF4"
q = BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_quant_type="nf4", bnb_4bit_compute_dtype=torch.bfloat16,
                       bnb_4bit_use_double_quant=False)
t = time.time(); torch.cuda.reset_peak_memory_stats()
m = AutoModelForImageTextToText.from_pretrained(src, quantization_config=q, device_map={"": 0}, dtype=torch.bfloat16)
m.eval()
mb = lambda x: round(x / 2**20)
print(json.dumps({"load_s": round(time.time() - t, 1), "peak_alloc_mb": mb(torch.cuda.max_memory_allocated()),
                  "alloc_mb": mb(torch.cuda.memory_allocated()), "reserved_mb": mb(torch.cuda.memory_reserved())}), flush=True)
p = AutoProcessor.from_pretrained(src)
msgs = [{"role": "user", "content": [{"type": "text", "text": "In two sentences, why is the sky blue?"}]}]
ids = p.apply_chat_template(msgs, add_generation_prompt=True, tokenize=True, return_dict=True, return_tensors="pt").to(0)
torch.cuda.reset_peak_memory_stats(); t = time.time()
with torch.no_grad():
    g = m.generate(**ids, max_new_tokens=60, do_sample=False)
n = g.shape[1] - ids["input_ids"].shape[1]
print(json.dumps({"gen_tok": int(n), "tok_s": round(n / (time.time() - t), 1), "gen_peak_mb": mb(torch.cuda.max_memory_allocated()),
                  "text": p.decode(g[0, ids["input_ids"].shape[1]:], skip_special_tokens=True)[:200]}), flush=True)
t = time.time(); m.save_pretrained(out); p.save_pretrained(out)
print(json.dumps({"saved": out, "save_s": round(time.time() - t, 1)}), flush=True)
