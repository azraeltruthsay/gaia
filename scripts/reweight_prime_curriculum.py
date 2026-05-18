#!/usr/bin/env python3
"""Reweight Prime tier curriculum for Qwen3-VL training.

Starts from core_v2x_prime/text.jsonl (the V11 mirror) and produces
a Prime-specific weighting:

  - deliberation:  4x   — Prime's signature capability (was 367, → ~1500)
  - multiturn:     2x   — Prime should excel at longer dialogues
  - tool_routing:  1.5x — robust tool emission (closes V11 gap)
  - gaia_identity: 1x   — dissociation+termination patches are fine as-is
  - alpaca:        0.5x — subsample; general capability is well-covered

Also generates and inlines `tool_synthesis` samples (h73 — V11 couldn't
synthesize from tool_result; Prime gets explicit training in:
  prompt + tool_call + tool_result → natural-language summary).

Text-only output (no vision/audio symlinks). Vision is Core's job;
Prime is a deep-reasoning text tier.
"""
import json
import random
from collections import Counter
from pathlib import Path


SRC = Path("/gaia/GAIA_Project/knowledge/curricula/core_v2x_prime/text.jsonl")
DST_DIR = Path("/gaia/GAIA_Project/knowledge/curricula/core_v2x_prime_weighted")

# Per-category weights. >1 = repeat-with-shuffle to upweight; <1 = subsample.
# These are MULTIPLIERS on the original count, not target counts.
WEIGHTS: dict[str, float] = {
    "deliberation":  4.0,   # Prime's signature — biggest jump
    "multiturn":     2.5,   # Prime handles longer dialogues
    "tool_routing":  1.5,   # robust tool emission
    "gaia_identity": 0.4,   # dissociation is useful but not core to Prime
    "alpaca":        0.25,  # general capability — covered, don't over-rely
}


# ── Tool-result synthesis samples (h73 — train tool_result → summary) ─────
# These teach Prime to read a tool_result and produce a natural-language
# answer. Without this, V11-style training causes the model to either
# re-emit the tool_call or hallucinate. We provide the user prompt + the
# model's prior tool_call + the tool_result, and the expected answer is
# the synthesis.

def make_synthesis_samples(rng: random.Random) -> list[dict]:
    """Generate ~250 tool_result synthesis samples covering the same
    13 MCP tool families as the tool_routing curriculum."""
    out: list[dict] = []

    # Each entry: (user_prompt, ack, tool_call_json, tool_result_json, synthesis)
    file_cases = [
        ("Read /shared/CLAUDE.md briefly.",
         "Reading the file.",
         '{"tool":"file","action":"read","path":"/shared/CLAUDE.md"}',
         '{"ok":true,"path":"/shared/CLAUDE.md","content":"# GAIA Project — Claude Code Instructions\\n\\n> Last updated: 2026-04-14 | Era: Sovereign Duality | Services: 12"}',
         "The file starts with the GAIA Project Claude Code Instructions header, last updated April 14 2026, in the Sovereign Duality era. It tracks 12 services."),
        ("What files are in /shared?",
         "Listing files.",
         '{"tool":"file","action":"list","path":"/shared/"}',
         '{"ok":true,"path":"/shared","entries":["atlas","audio","kvcache","logs","sessions.json","training_runs"]}',
         "There are 6 entries in /shared/: atlas, audio, kvcache, logs, sessions.json, and training_runs."),
        ("Show me the tree under /gaia/GAIA_Project/scripts.",
         "Getting the tree.",
         '{"tool":"file","action":"tree","path":"/gaia/GAIA_Project/scripts"}',
         '{"ok":true,"entries":["audit_curriculum.py","build_core_v2x_curriculum.py","train_core_multimodal.py","merge_and_save_adapter.py"]}',
         "Four scripts under /gaia/GAIA_Project/scripts: audit_curriculum.py, build_core_v2x_curriculum.py, train_core_multimodal.py, and merge_and_save_adapter.py."),
        ("Find Python files under /shared/training_runs.",
         "Searching.",
         '{"tool":"file","action":"find","path":"/shared/training_runs","pattern":"*.py"}',
         '{"ok":true,"matches":[]}',
         "No Python files in /shared/training_runs/ — that directory holds checkpoints and metrics, not source."),
    ]
    web_cases = [
        ("Look up the latest Gemma 4 release notes.",
         "Searching.",
         '{"tool":"web","action":"search","query":"latest Gemma 4 release notes","max_results":5}',
         '{"ok":true,"results":[{"title":"Gemma 4 E4B model card","snippet":"Multimodal LLM with native vision + audio towers, 8B parameters."}]}',
         "The top result describes Gemma 4 E4B as Google's 8-billion-parameter multimodal model with native vision and audio towers."),
        ("Fetch huggingface.co/google/gemma-4-E4B and summarize.",
         "Fetching.",
         '{"tool":"web","action":"fetch","url":"https://huggingface.co/google/gemma-4-E4B"}',
         '{"ok":true,"content":"Gemma 4 E4B — open-weight multimodal transformer."}',
         "The page describes Gemma 4 E4B as an open-weight multimodal transformer from Google."),
    ]
    shell_cases = [
        ("Run `nvidia-smi --query-gpu=memory.used,memory.free --format=csv`.",
         "Running.",
         '{"tool":"shell","action":"run","command":"nvidia-smi --query-gpu=memory.used,memory.free --format=csv"}',
         '{"ok":true,"stdout":"memory.used, memory.free\\n9501 MiB, 6332 MiB\\n"}',
         "GPU has 9501 MiB used and 6332 MiB free — Core is loaded and there's a bit over 6 GB headroom."),
        ("Execute: df -h /shared",
         "Running.",
         '{"tool":"shell","action":"run","command":"df -h /shared"}',
         '{"ok":true,"stdout":"Filesystem Size Used Avail Use% Mounted on\\ngaia 480G 331G 125G 73% /shared\\n"}',
         "/shared is on a 480 GB filesystem; 331 GB used, 125 GB free, 73 percent utilization."),
    ]
    knowledge_cases = [
        ("What does the knowledge base say about the consciousness matrix?",
         "Querying.",
         '{"tool":"knowledge","action":"query","query":"consciousness matrix"}',
         '{"ok":true,"hits":[{"snippet":"Three states: Conscious (GPU), Subconscious (CPU/GGUF), Unconscious (unloaded)."}]}',
         "The consciousness matrix has three states per tier: Conscious (GPU), Subconscious (CPU/GGUF), and Unconscious (unloaded)."),
        ("Recall what we know about training run V11.",
         "Recalling.",
         '{"tool":"knowledge","action":"memory","query":"V11 training run"}',
         '{"ok":true,"hits":[{"snippet":"V11: r=32 LoRA, 6000 steps, Gemma 4 E4B base, final loss 21.26."}]}',
         "V11 was the rank-32 LoRA training run on the Gemma 4 E4B base, 6000 steps total with a final loss of 21.26."),
    ]
    palace_cases = [
        ("Store this in palace memory: the V8-V11 lesson — identity is system-prompt state, not weights.",
         "Storing.",
         '{"tool":"palace","action":"store","topic":"V8-V11 identity lesson"}',
         '{"ok":true,"stored":true,"id":"palace-2026-05-18-001"}',
         "Stored — palace entry id palace-2026-05-18-001."),
        ("Recall everything about Azrael's preferences.",
         "Recalling.",
         '{"tool":"palace","action":"recall","topic":"Azrael preferences"}',
         '{"ok":true,"entries":["Prefers terse responses","Match the pace of 70+ commit sessions","Tests in Docker, never on host"]}',
         "Three relevant preferences: terse responses, fast session pacing, and Docker-only testing — never on host."),
    ]
    introspect_cases = [
        ("What tools do I have access to?",
         "Listing tools.",
         '{"tool":"introspect","action":"tools"}',
         '{"ok":true,"tools":["file","shell","web","knowledge","palace","audio","study","introspect","worldbuild","notebook","context","browser","manage","fabric"]}',
         "Fourteen tool families available: file, shell, web, knowledge, palace, audio, study, introspect, worldbuild, notebook, context, browser, manage, and fabric."),
        ("Show recent system logs.",
         "Fetching logs.",
         '{"tool":"introspect","action":"logs"}',
         '{"ok":true,"recent":["05:43 conversation Hello GAIA","04:25 lifecycle parked -> awake"]}',
         "Recent events: a 'Hello GAIA' conversation at 05:43 and a lifecycle transition from parked to awake at 04:25."),
    ]
    error_cases = [
        ("Read the file at /shared/nonexistent.txt.",
         "Reading.",
         '{"tool":"file","action":"read","path":"/shared/nonexistent.txt"}',
         '{"ok":false,"error":"FileNotFoundError: /shared/nonexistent.txt"}',
         "That file doesn't exist — /shared/nonexistent.txt isn't there. Want me to list /shared to find what you meant?"),
        ("Search the web for 'made-up-thing-that-returns-nothing'.",
         "Searching.",
         '{"tool":"web","action":"search","query":"made-up-thing-that-returns-nothing"}',
         '{"ok":true,"results":[]}',
         "No results came back — that phrase doesn't match anything indexed. Want to try a different query?"),
        ("Run `non_existent_command`.",
         "Running.",
         '{"tool":"shell","action":"run","command":"non_existent_command"}',
         '{"ok":false,"error":"command not found: non_existent_command","stderr":"bash: non_existent_command: command not found"}',
         "The shell reported `command not found` for non_existent_command. Did you mean something else, or want me to search for the right tool?"),
    ]

    all_cases = (
        file_cases + web_cases + shell_cases + knowledge_cases
        + palace_cases + introspect_cases + error_cases
    )

    for prompt, ack, tool_call_json, tool_result_json, synthesis in all_cases:
        # Multi-turn format: instruction includes the user prompt + the
        # previous assistant turn (ack + tool_call) + the tool_result;
        # the output is the synthesis (natural-language summary).
        instruction = (
            f"{prompt}\n\n"
            f"[previous assistant turn: {ack}\n"
            f"<tool_call>{tool_call_json}</tool_call>]\n\n"
            f"[tool_result: {tool_result_json}]\n\n"
            f"Respond to the user with a natural-language summary of the result."
        )
        out.append({
            "instruction": instruction,
            "output": synthesis,
            "category": "tool_synthesis",
        })

    # Shuffle to mix tool families and error cases, then repeat enough
    # to give the synthesis pattern real gradient weight. ~500 samples
    # (~3.5% of curriculum) so the model sees each unique case ~25 times
    # across a 6000-step run — sufficient to learn the pattern without
    # so much duplication that it memorizes specific result-strings.
    rng.shuffle(out)
    SYNTHESIS_TARGET = 500
    reps = max(1, SYNTHESIS_TARGET // len(out))
    return out * reps


def main() -> int:
    if not SRC.exists():
        print(f"ERROR: {SRC} not found — run build_prime_v11_curriculum.py first")
        return 1

    DST_DIR.mkdir(parents=True, exist_ok=True)
    rng = random.Random(42)

    # Bucket source samples by category
    buckets: dict[str, list[dict]] = {}
    with open(SRC) as f:
        for line in f:
            d = json.loads(line)
            buckets.setdefault(d.get("category", "?"), []).append(d)

    print("Source counts:")
    for c, ss in sorted(buckets.items()):
        print(f"  {c:25s} {len(ss):6d}")

    # Apply weights
    weighted: list[dict] = []
    print("\nApplying weights:")
    for cat, samples in sorted(buckets.items()):
        w = WEIGHTS.get(cat, 1.0)
        if w >= 1.0:
            # Repeat-with-shuffle to upweight
            target = int(len(samples) * w)
            # Cycle through the bucket; the spiral builder will reshuffle later
            picks = []
            while len(picks) < target:
                rng.shuffle(samples)
                picks.extend(samples)
            picks = picks[:target]
        else:
            # Subsample
            target = int(len(samples) * w)
            picks = rng.sample(samples, target)
        weighted.extend(picks)
        print(f"  {cat:25s} x{w:4.2f} → {len(picks):6d}")

    # Add synthesis samples (h73 fix)
    syn = make_synthesis_samples(rng)
    weighted.extend(syn)
    print(f"  {'tool_synthesis':25s} (new) → {len(syn):6d}")

    rng.shuffle(weighted)
    dst = DST_DIR / "text.jsonl"
    with open(dst, "w") as f:
        for r in weighted:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")

    final_cats = Counter(r.get("category", "?") for r in weighted)
    print(f"\nWrote {len(weighted)} samples → {dst}")
    print("\nFinal per-category distribution:")
    for c, n in sorted(final_cats.items()):
        print(f"  {c:25s} {n:6d}  ({100*n/len(weighted):.1f}%)")

    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
