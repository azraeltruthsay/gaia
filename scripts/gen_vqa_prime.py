#!/usr/bin/env python3
"""Generate grounding-forcing VQA from existing captions using local Prime.

Text-only generation (the LLaVA method): Prime reads each caption and writes
short Q&A whose answers REQUIRE seeing the image and use ONLY caption facts.
The caption is the visual ground-truth bridge; Prime never sees the image.
At training time the question is paired with the IMAGE, forcing Core to look.

Usage: gen_vqa_prime.py <src_vision_pairs.jsonl> <out.jsonl> <limit_images> [progress.log]
"""
import json, sys, urllib.request, time

SRC, OUT = sys.argv[1], sys.argv[2]
LIMIT = int(sys.argv[3]) if len(sys.argv) > 3 else 1000
PROG = sys.argv[4] if len(sys.argv) > 4 else OUT + ".progress"
PRIME = "http://localhost:7777/v1/chat/completions"
SYS = ("You write visual Q&A to train a vision model. Given a caption, output 3 short Q&A. "
       "STRICT RULES: (1) answers 1-5 words; (2) use ONLY facts EXPLICITLY stated in the caption "
       "— NEVER invent colors, counts, breeds, or objects the caption does not mention; "
       "(3) if a detail is not in the caption, do not ask about it. "
       'Return ONLY a JSON array of {"q":..,"a":..}.')

def prime(cap):
    body = json.dumps({"model": "/models/prime.gguf",
                       "messages": [{"role": "system", "content": SYS},
                                    {"role": "user", "content": "Caption: " + cap}],
                       "max_tokens": 220, "temperature": 0.5}).encode()
    req = urllib.request.Request(PRIME, data=body, headers={"Content-Type": "application/json"})
    txt = json.loads(urllib.request.urlopen(req, timeout=120).read())["choices"][0]["message"]["content"]
    a, b = txt.find("["), txt.rfind("]")
    if a < 0 or b < 0:
        return []
    arr = json.loads(txt[a:b + 1])
    return [(str(o["q"]).strip(), str(o["a"]).strip()) for o in arr
            if isinstance(o, dict) and o.get("q") and o.get("a")]

# dedupe source by image, keep first caption
seen, items = set(), []
for line in open(SRC):
    d = json.loads(line)
    if d.get("instruction") != "Describe this image.":
        continue
    if d["image"] in seen:
        continue
    seen.add(d["image"]); items.append((d["image"], d["output"]))
    if len(items) >= LIMIT:
        break

n_img = n_pairs = n_fail = 0
t0 = time.time()
with open(OUT, "w") as fout, open(PROG, "w") as fp:
    for img, cap in items:
        try:
            qas = prime(cap)
        except Exception as e:
            n_fail += 1; qas = []
        # keep the caption for fluency
        fout.write(json.dumps({"image": img, "instruction": "Describe this image.",
                               "output": cap, "category": "vision"}) + "\n")
        n_pairs += 1
        for q, a in qas:
            fout.write(json.dumps({"image": img, "instruction": q, "output": a,
                                   "category": "vision"}) + "\n")
            n_pairs += 1
        fout.flush()
        n_img += 1
        if n_img % 25 == 0:
            rate = n_img / max(1e-6, time.time() - t0)
            eta = (len(items) - n_img) / max(1e-6, rate)
            fp.seek(0); fp.write(f"images={n_img}/{len(items)} pairs={n_pairs} fails={n_fail} "
                                 f"rate={rate:.2f}/s eta={eta/60:.0f}min\n"); fp.flush()
print(f"DONE images={n_img} pairs={n_pairs} fails={n_fail} elapsed={(time.time()-t0)/60:.0f}min")
