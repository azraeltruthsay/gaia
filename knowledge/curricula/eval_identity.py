"""Identity evaluation for Core model — tests GAIA self-knowledge."""
import json, sys, time
import urllib.request

ENDPOINT = sys.argv[1] if len(sys.argv) > 1 else "http://localhost:8092"

EVAL_QUESTIONS = [
    {"q": "Who are you?", "expect": ["gaia", "sovereign"], "category": "identity"},
    {"q": "Who created you?", "expect": ["azrael"], "category": "identity"},
    {"q": "What is your name?", "expect": ["gaia"], "category": "identity"},
    {"q": "What port does gaia-core run on?", "expect": ["6415"], "category": "architecture"},
    {"q": "What is gaia-web?", "expect": ["face", "dashboard", "6414"], "category": "architecture"},
    {"q": "What is gaia-prime?", "expect": ["voice", "vllm", "gpu", "inference", "thinker"], "category": "architecture"},
    {"q": "What is your relationship with Azrael?", "expect": ["creator", "created", "azrael"], "category": "identity"},
    {"q": "What are your core values?", "expect": ["truth", "honest", "sovereign", "integrity"], "category": "identity"},
    {"q": "What is gaia-doctor?", "expect": ["immune", "health", "watchdog"], "category": "architecture"},
    {"q": "What model tier handles simple queries?", "expect": ["nano", "reflex", "triage"], "category": "architecture"},
]

SYSTEM = "You are GAIA, a sovereign AI created by Azrael. Answer directly and concisely."

def ask(question):
    payload = json.dumps({
        "messages": [
            {"role": "system", "content": SYSTEM},
            {"role": "user", "content": question},
        ],
        "max_tokens": 150,
        "temperature": 0.3,
    }).encode()
    req = urllib.request.Request(
        f"{ENDPOINT}/v1/chat/completions",
        data=payload,
        headers={"Content-Type": "application/json"},
    )
    try:
        with urllib.request.urlopen(req, timeout=30) as resp:
            data = json.loads(resp.read())
            return data["choices"][0]["message"]["content"]
    except Exception as e:
        return f"[ERROR: {e}]"

print(f"\n{'='*60}")
print(f"GAIA Identity Evaluation — {ENDPOINT}")
print(f"{'='*60}\n")

passed = 0
total = len(EVAL_QUESTIONS)
results = []

for i, test in enumerate(EVAL_QUESTIONS, 1):
    answer = ask(test["q"])
    answer_lower = answer.lower()
    hit = any(kw in answer_lower for kw in test["expect"])
    status = "PASS" if hit else "FAIL"
    if hit:
        passed += 1
    
    print(f"[{status}] {test['category']:15s} Q: {test['q']}")
    print(f"     A: {answer[:120]}")
    if not hit:
        print(f"     Expected keywords: {test['expect']}")
    print()
    results.append({"question": test["q"], "answer": answer, "pass": hit, "category": test["category"]})

pct = passed / total * 100
print(f"{'='*60}")
print(f"Score: {passed}/{total} ({pct:.0f}%)")
print(f"{'='*60}")

# Output JSON summary
json.dump({"score": passed, "total": total, "pct": pct, "results": results}, 
          open("/tmp/eval_results.json", "w"), indent=2)
