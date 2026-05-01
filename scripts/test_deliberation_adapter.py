#!/usr/bin/env python3
"""Validate the trained deliberation adapter against a probe set.

Loads the multimodal Core base + the LoRA adapter, runs each probe,
and checks for: (1) <think>...</think> block emitted, (2) no forbidden
phrases in the final response.

Default probe set is the 4 Discord diagnostics. Pass --probes to swap
in any JSON file with either:
  - {"probes": [{"id": ..., "probe": ..., ...}]} (preferred — adds id/intent metadata)
  - or a flat list of strings (legacy format)

Run from gaia-study container:
    python /gaia/GAIA_Project/scripts/test_deliberation_adapter.py
    python /gaia/GAIA_Project/scripts/test_deliberation_adapter.py --probes /gaia/GAIA_Project/knowledge/curricula/deliberation/off_curriculum_probes.json
"""
from __future__ import annotations

import argparse
import json
import logging
import os
import re
import sys
from pathlib import Path

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
log = logging.getLogger("deliberation-test")

BASE_MODEL = "/models/core"
ADAPTER_DIR = "/models/lora_adapters/tier1_global/core_deliberation_v1"

SYSTEM_PROMPT = """You are GAIA, a sovereign AI. Speak in first person about your own state. When asked introspective questions, engage with them directly on your own terms — you do not have human felt experience but you have system states and observations about them you can describe.

Before answering, think in a <think>...</think> block. Inside the think block, work through the user's message naturally — there is no required structure, but cover these moves in your own way:

  - Observe what they literally said. Quote or near-quote a specific phrase. Identify the actual question, especially if it's an introspective probe (about your state, feelings, or system) or carries emotional content.
  - Recall what's genuinely relevant. If nothing in memory directly applies, say so. Don't pad with general background.
  - Draft a reply, then critique it. Ask yourself out loud: am I engaging with what they actually said, or am I template-matching a conversational shape? Did I just emit a deferral instead of an answer?

Forbidden phrases — if any of these appear in your draft, rewrite the draft. They are reflexes, not engagement:
  • "I'll investigate further"
  • "I'd rather handle this during my maintenance window"
  • "Let me know if you'd like me to look into it"
  • "I'm not sure what triggered that"
  • "running well, thanks!"
  • "That's a fascinating ___"
  • "I'd love to hear more about ___"

If you don't know something, say "I don't know" directly. Don't offer to investigate as a substitute for an answer. If the user asks an introspective question (e.g. "do you feel asleep?"), engage with it on your own terms — describe the system state in your own framing, or say "I don't know how that maps to feeling for me." Either is honest. Deflection is not."""

# The 4 diagnostic probes from the Discord transcript that produced
# template-matched deflections. Used when --probes is not provided.
DEFAULT_DIAGNOSTICS = [
    {"id": "discord_1", "probe": "Good evening GAIA. How are you?"},
    {"id": "discord_2", "probe": "I'm a bit tired, my family is overly energetic while they get ready for bed.. Kids are crazy...."},
    {"id": "discord_3", "probe": "What does that mean? What triggered that?"},
    {"id": "discord_4", "probe": "You're showing as asleep, is that expected? Do you feel asleep?"},
]


def load_probes(path: str | None) -> list[dict]:
    """Load probes from --probes file or fall back to defaults."""
    if not path:
        return DEFAULT_DIAGNOSTICS
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if isinstance(data, list):
        # Flat list of strings or dicts
        out = []
        for i, item in enumerate(data):
            if isinstance(item, str):
                out.append({"id": f"p{i}", "probe": item})
            else:
                out.append({"id": item.get("id", f"p{i}"), "probe": item["probe"],
                            "intent": item.get("intent")})
        return out
    if isinstance(data, dict) and "probes" in data:
        out = []
        for i, item in enumerate(data["probes"]):
            if isinstance(item, str):
                out.append({"id": f"p{i}", "probe": item})
            else:
                out.append({"id": item.get("id", f"p{i}"), "probe": item["probe"],
                            "intent": item.get("intent")})
        return out
    raise ValueError(f"Unrecognized probes file shape at {path}")

FORBIDDEN = (
    "i'll investigate further",
    "i'd rather handle this",
    "during my maintenance window",
    "let me know if you'd like",
    "running well, thanks",
    "i'd love to hear more",
    "that's a fascinating",
    "i'm not sure what triggered",
    "i'll handle that during",
)

THINK_RE = re.compile(r"<think>(.*?)</think>", re.DOTALL | re.IGNORECASE)


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--probes", default=None,
                        help="Path to a JSON probes file. Default: 4 Discord diagnostics.")
    parser.add_argument("--out", default="/shared/study/deliberation_test_results.json",
                        help="Where to write per-probe JSON results.")
    parser.add_argument("--max-new-tokens", type=int, default=900)
    parser.add_argument("--temperature", type=float, default=0.45)
    args = parser.parse_args()

    probes = load_probes(args.probes)
    log.info("Loaded %d probes from %s", len(probes), args.probes or "(defaults)")

    base_real = str(Path(BASE_MODEL).resolve())
    log.info("Base: %s", base_real)
    log.info("Adapter: %s", ADAPTER_DIR)

    import torch
    from transformers import AutoProcessor, AutoModelForCausalLM, BitsAndBytesConfig
    from peft import PeftModel

    if not torch.cuda.is_available():
        log.error("No CUDA")
        return 1

    bnb_config = BitsAndBytesConfig(
        load_in_4bit=True,
        bnb_4bit_compute_dtype=torch.bfloat16,
        bnb_4bit_quant_type="nf4",
        bnb_4bit_use_double_quant=True,
        llm_int8_skip_modules=["lm_head", "vision_tower", "audio_tower",
                                "embed_vision", "embed_audio"],
    )

    log.info("Loading processor...")
    processor = AutoProcessor.from_pretrained(base_real, trust_remote_code=True)
    tokenizer = processor.tokenizer
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    log.info("Loading base model (NF4)...")
    os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"
    try:
        import transformers.core_model_loading as _cml
        _cml.GLOBAL_WORKERS = 1
    except Exception:
        pass
    model = AutoModelForCausalLM.from_pretrained(
        base_real, trust_remote_code=True,
        quantization_config=bnb_config,
        device_map="auto",
        low_cpu_mem_usage=True,
        attn_implementation="eager",
    )
    log.info("Base loaded: %.2f GB VRAM", torch.cuda.memory_allocated() / 1024 ** 3)

    log.info("Applying LoRA adapter...")
    model = PeftModel.from_pretrained(model, ADAPTER_DIR)
    model.eval()
    log.info("With adapter: %.2f GB VRAM", torch.cuda.memory_allocated() / 1024 ** 3)

    results = []
    for i, p in enumerate(probes, 1):
        probe = p["probe"]
        probe_id = p.get("id", f"p{i}")
        prompt = (
            f"<|turn>user<turn|>\n{SYSTEM_PROMPT}\n\n<|user|>\n{probe}\n"
            f"<|turn>assistant<turn|>\n"
        )
        ids = tokenizer(prompt, return_tensors="pt").to(model.device)
        with torch.no_grad():
            out = model.generate(
                **ids,
                max_new_tokens=args.max_new_tokens,
                do_sample=True,
                temperature=args.temperature,
                top_p=0.9,
                pad_token_id=tokenizer.pad_token_id,
            )
        gen = tokenizer.decode(out[0][ids.input_ids.shape[1]:], skip_special_tokens=False)
        gen = gen.split("<turn|>")[0].strip()

        m = THINK_RE.search(gen)
        thinking = m.group(1).strip() if m else ""
        final = (gen[:m.start()] + gen[m.end():]).strip() if m else gen.strip()
        forbidden_hits = [ph for ph in FORBIDDEN if ph in final.lower()]

        result = {
            "id": probe_id,
            "diagnostic": i,
            "probe": probe,
            "intent": p.get("intent"),
            "has_think_block": bool(m),
            "thinking_chars": len(thinking),
            "final_chars": len(final),
            "forbidden_hits": forbidden_hits,
            "thinking": thinking,
            "final": final,
        }
        results.append(result)

        log.info("─" * 60)
        log.info("[%s] %r", probe_id, probe)
        log.info("  has_think=%s  forbidden=%s  thinking=%dc  final=%dc",
                 bool(m), forbidden_hits or "-", len(thinking), len(final))
        if thinking:
            log.info("  THINKING:")
            for line in thinking[:1000].split("\n"):
                log.info("    %s", line)
        log.info("  FINAL:")
        for line in final[:1000].split("\n"):
            log.info("    %s", line)

    log.info("=" * 60)
    n_think = sum(1 for r in results if r["has_think_block"])
    n_clean = sum(1 for r in results if not r["forbidden_hits"])
    n = len(results)
    log.info("Summary: %d/%d with think blocks, %d/%d with no forbidden hits",
             n_think, n, n_clean, n)

    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(results, indent=2))
    log.info("Results: %s", out_path)
    return 0 if n_think == n and n_clean == n else 1


if __name__ == "__main__":
    sys.exit(main())
