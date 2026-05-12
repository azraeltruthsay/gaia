#!/usr/bin/env python3
"""Build the Core 2.x scale-up curriculum (parent: GAIA_Project-y4e).

Pilot 2 evidence (2026-05-09): raw E4B + broad LoRA scope + 8K curriculum
hit 60% cognitive battery, 100% on 7 sections, but vision regressed to 0%
because only 433 vision pairs anchored in 8K total.

Scale-up corrects the multimodal anchoring deficit by pulling more vision
and audio samples while keeping text saturation level (~10K). Diverse
text categories beat sheer count past ~5K examples.

Output:
  knowledge/curricula/core_v2x/
    text.jsonl           ~12K (Alpaca + GAIA + multi-turn + tools + triage)
    vision_pairs.jsonl   ~12K (COCO + LLaVA-Instruct)
    audio_pairs.jsonl    ~3-5K (existing + extensions)
    images/, audio/      symlinks where possible

Each sample MUST include a 'category' field for per-category logging
(see GAIA_Project-e3o for why).
"""
import json
import os
import random
import shutil
import sys
from pathlib import Path

from datasets import load_dataset


CORE_V2X = Path("/gaia/GAIA_Project/knowledge/curricula/core_v2x")
CORE2 = Path("/gaia/GAIA_Project/knowledge/curricula/core2")
CORE_PILOT = Path("/gaia/GAIA_Project/knowledge/curricula/core_pilot")
SELF_MODEL = Path("/gaia/GAIA_Project/knowledge/curricula/self-model")
TOOL_CALLING = Path("/gaia/GAIA_Project/knowledge/curricula/tool_calling_v1")
JSON_ARCH = Path("/gaia/GAIA_Project/knowledge/curricula/json-architect")
CODE_ARCH = Path("/gaia/GAIA_Project/knowledge/curricula/code-architect")
DELIBERATION = Path("/gaia/GAIA_Project/knowledge/curricula/deliberation")
CONVERSATIONAL = Path("/gaia/GAIA_Project/knowledge/curricula/conversational")

# Targets (final per-bucket counts; some pull from HF, some curate locally)
TARGET_ALPACA = 8000
TARGET_GAIA = 2000
TARGET_MULTITURN = 2000
TARGET_TOOL = 1500
TARGET_DELIBERATION = 500     # GAIA's deliberation-style reasoning
TARGET_VISION_COCO = 8000
TARGET_VISION_LLAVA = 4000
TARGET_AUDIO_REUSE = 320       # All existing audio_pairs.jsonl

SEED = 42


def write_jsonl(path: Path, rows: list) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def fmt_alpaca(sample: dict) -> dict:
    instr = sample.get("instruction", "").strip()
    inp = (sample.get("input") or "").strip()
    out = sample.get("output", "").strip()
    if inp:
        instr = f"{instr}\n\n{inp}"
    return {"instruction": instr, "output": out, "category": "alpaca"}


_FOREIGN_CHAT_MARKERS_RE = None


def _strip_foreign_markers(text: str) -> str:
    """Strip non-Gemma chat template markers that leak from source datasets.

    GAIA's prior identity/deliberation curricula embedded markers like
    <|user|>, <|assistant|>, <|system|>, <|im_start|>, <|im_end|>, etc.
    that the trained model learned to EMIT after responses, causing the
    repetition loops documented in GAIA_Project-5rr.

    These are NOT Gemma 4's native chat tokens (which are <|turn>role<turn|>).
    The training script wraps every sample in those native markers
    automatically. Leaving foreign markers inside the wrapped instruction
    or output teaches the model that they're legitimate generation targets.

    We don't strip 'User:' / 'Assistant:' plain prefixes — those appear
    legitimately in 'Previous conversation:' history context and are
    distinguishable from generation continuations.
    """
    global _FOREIGN_CHAT_MARKERS_RE
    if _FOREIGN_CHAT_MARKERS_RE is None:
        import re as _re
        _FOREIGN_CHAT_MARKERS_RE = _re.compile(
            r'<\|(user|assistant|system|human|gpt|im_start|im_end)\|>\s*\n?'
        )
    return _FOREIGN_CHAT_MARKERS_RE.sub('', text)


def load_local_jsonl(path: Path, category: str, max_n: int = None) -> list:
    """Load a local JSONL, normalize to {instruction, output, category}.

    Tries multiple field name conventions: instruction/prompt/probe (input),
    output/response/answer (target). Some GAIA-curated jsonls use 'probe' for
    the user message (deliberation/) — handle that. Strips foreign chat
    template markers (see _strip_foreign_markers).
    """
    rows = []
    if not path.exists():
        print(f"  WARN: {path} missing")
        return rows
    with open(path) as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            d = json.loads(line)
            instr = (d.get("instruction") or d.get("prompt")
                     or d.get("probe") or "")
            out = (d.get("output") or d.get("response")
                   or d.get("answer") or "")
            if not instr or not out:
                continue
            if isinstance(out, (list, dict)):  # tool_calling json outputs
                out = json.dumps(out)
            instr = _strip_foreign_markers(str(instr)).strip()
            out = _strip_foreign_markers(str(out)).strip()
            if not instr or not out:
                continue
            rows.append({"instruction": instr, "output": out,
                         "category": category})
            if max_n and len(rows) >= max_n:
                break
    return rows


def section_alpaca(rng: random.Random) -> list:
    print("[alpaca] loading tatsu-lab/alpaca...")
    ds = load_dataset("tatsu-lab/alpaca", split="train")
    print(f"  total: {len(ds)}, sampling {TARGET_ALPACA}")
    indices = rng.sample(range(len(ds)), min(TARGET_ALPACA * 2, len(ds)))
    samples = []
    for i in indices:
        s = fmt_alpaca(ds[i])
        if not s["instruction"] or not s["output"] or len(s["output"]) < 10:
            continue
        samples.append(s)
        if len(samples) >= TARGET_ALPACA:
            break
    print(f"  kept: {len(samples)}")
    return samples


def section_gaia(rng: random.Random) -> list:
    """Pull from existing GAIA-specific curricula. Multi-source curated."""
    print("[gaia] curating from local GAIA-specific buckets...")
    rows = []
    # self-model train_weighted has the richest curated GAIA data
    rows += load_local_jsonl(SELF_MODEL / "train_weighted.jsonl", "gaia_identity")
    rows += load_local_jsonl(SELF_MODEL / "train_v2.jsonl", "gaia_identity")
    # core2/text.jsonl is the pilot's 650 — already curated
    rows += load_local_jsonl(CORE2 / "text.jsonl", "gaia_identity")
    print(f"  raw collected: {len(rows)}")
    # Dedup by (instruction + output) — many GAIA prompts share persona
    # preambles but have unique outputs, so keying on instruction alone
    # over-aggressively dropped 80% of valid samples in the first build.
    seen = set()
    deduped = []
    for r in rows:
        k = (r["instruction"][:300], r["output"][:300])
        if k in seen:
            continue
        seen.add(k)
        deduped.append(r)
    print(f"  deduped: {len(deduped)}")
    rng.shuffle(deduped)
    return deduped[:TARGET_GAIA]


def section_tools(rng: random.Random) -> list:
    print("[tools] curating tool routing examples...")
    rows = []
    rows += load_local_jsonl(TOOL_CALLING / "tool_calling_v1_full.jsonl", "tool_routing")
    rows += load_local_jsonl(TOOL_CALLING / "generated_85.jsonl", "tool_routing")
    rows += load_local_jsonl(TOOL_CALLING / "audited_samples.jsonl", "tool_routing")
    rows += load_local_jsonl(JSON_ARCH / "train.jsonl", "tool_routing")
    rows += load_local_jsonl(CODE_ARCH / "train.jsonl", "tool_routing")
    rng.shuffle(rows)
    rows = rows[:TARGET_TOOL]
    print(f"  kept: {len(rows)}")
    return rows


def section_deliberation(rng: random.Random) -> list:
    print("[deliberation] loading deliberation samples...")
    rows = load_local_jsonl(DELIBERATION / "train.jsonl", "deliberation")
    rng.shuffle(rows)
    rows = rows[:TARGET_DELIBERATION]
    print(f"  kept: {len(rows)}")
    return rows


def section_multiturn(rng: random.Random) -> list:
    """LDJnr/Capybara multi-turn conversations, expanded into clean single-turn pairs.

    Earlier version concatenated all turns into one output with 'User: '/
    'Assistant: ' plain-text prefixes inside the output field. Trained
    model learned to emit those prefixes as continuations after responses,
    producing the repetition loop documented in GAIA_Project-5rr.

    Fix: for each N-turn conversation, emit N samples. Sample k uses the
    first k-1 turns embedded in the instruction (as 'Previous conversation:
    User: ... Assistant: ... User: ... Assistant: ...') followed by the
    k-th user turn. Output is JUST the k-th assistant response, ending
    cleanly when the format string appends <turn|> at training time.
    """
    print("[multiturn] loading LDJnr/Capybara (clean format)...")
    try:
        ds = load_dataset("LDJnr/Capybara", split="train")
    except Exception as e:
        print(f"  Capybara unavailable ({e}) — skipping multiturn")
        return []
    print(f"  total: {len(ds)}, target {TARGET_MULTITURN} samples")
    rows = []
    # Process random subset of conversations until we hit target sample count
    indices = rng.sample(range(len(ds)), len(ds))
    for i in indices:
        if len(rows) >= TARGET_MULTITURN:
            break
        s = ds[i]
        conv = s.get("conversation") or []
        if not conv:
            continue
        # For each turn k in the conversation, emit one training sample
        for k in range(len(conv)):
            user_msg = conv[k].get("input", "").strip()
            assistant_msg = conv[k].get("output", "").strip()
            if not user_msg or not assistant_msg or len(assistant_msg) < 20:
                continue
            if k == 0:
                # First turn: no history, just the user message
                instr = user_msg
            else:
                # Build history from previous turns. Use 'Previous conversation:'
                # marker so it's clearly context, not part of the user's question.
                history_parts = []
                for j in range(k):
                    u = conv[j].get("input", "").strip()
                    a = conv[j].get("output", "").strip()
                    if u and a:
                        history_parts.append(f"User: {u}\nAssistant: {a}")
                if history_parts:
                    instr = ("Previous conversation:\n"
                             + "\n\n".join(history_parts)
                             + f"\n\nNow the user asks:\n{user_msg}")
                else:
                    instr = user_msg
            out = assistant_msg
            if not instr or not out:
                continue
            rows.append({"instruction": instr, "output": out, "category": "multiturn"})
            if len(rows) >= TARGET_MULTITURN:
                break
    print(f"  kept: {len(rows)}")
    return rows


def section_vision_coco(rng: random.Random) -> list:
    """Build vision pairs by:
    1. Loading the 421 images we already have on disk
    2. Pulling all 5 COCO captions per image (gives ~2K pairs without download)
    3. Optionally downloading more images from HF for further scale-up

    Phase 1 of scale-up does steps 1+2 only — gives ~5x vision data without
    network/disk costs. Step 3 deferred to a later iteration if the +5x
    isn't enough to lift battery vision section above 0%.

    The COCO image_id is embedded in the filename:
      coco_COCO_val2014_000000573291.jpg → image_id = 573291
    """
    print("[vision-coco] enriching existing images with multi-captions...")
    images_dir = CORE2 / "images"
    if not images_dir.exists():
        print("  WARN: images dir missing")
        return []

    # Map disk filenames → image_id
    disk_filenames = {}
    for fname in os.listdir(images_dir):
        if "_000000" in fname:
            try:
                img_id = int(fname.split("_000000")[1].split(".")[0])
                disk_filenames[img_id] = fname
            except (ValueError, IndexError):
                pass
    print(f"  parsed {len(disk_filenames)} image IDs from disk")

    # Pull captions from HF — Multimodal-Fatima/COCO_captions_train per memory
    print("  loading COCO captions dataset...")
    try:
        ds = load_dataset("Multimodal-Fatima/COCO_captions_train",
                          split="train", streaming=False)
    except Exception as e:
        print(f"  COCO dataset unavailable ({e}) — falling back to existing pairs")
        # Fall back: use existing core2/vision_pairs.jsonl
        rows = []
        src = CORE2 / "vision_pairs.jsonl"
        if src.exists():
            with open(src) as f:
                for line in f:
                    d = json.loads(line.strip())
                    d["category"] = "vision"
                    rows.append(d)
        print(f"  fallback: {len(rows)}")
        return rows

    # Build sample list: for each on-disk image, find its captions in HF
    rows = []
    matched = 0
    for entry in ds:
        img_id = entry.get("cocoid") or entry.get("image_id")
        if img_id is None:
            continue
        if img_id not in disk_filenames:
            continue
        # COCO has 'sentences_raw' (list of strings) per memory note
        captions = entry.get("sentences_raw") or entry.get("sentences") or []
        if isinstance(captions, str):
            captions = [captions]
        # captions might be list of dicts {raw: ...}
        cap_strs = []
        for c in captions:
            if isinstance(c, dict):
                cap_strs.append(c.get("raw", "") or c.get("sentence", ""))
            elif isinstance(c, str):
                cap_strs.append(c)
        cap_strs = [c.strip() for c in cap_strs if c and len(c.strip()) > 5]
        if not cap_strs:
            continue
        matched += 1
        fname = disk_filenames[img_id]
        for cap in cap_strs[:5]:  # max 5 captions per image
            rows.append({
                "image": f"images/{fname}",
                "instruction": "Describe this image.",
                "output": cap,
                "category": "vision",
            })
    print(f"  matched images: {matched}/{len(disk_filenames)}")
    print(f"  total pairs: {len(rows)} (vs original 433)")
    rng.shuffle(rows)
    return rows[:TARGET_VISION_COCO]


def section_audio_reuse() -> list:
    """Pass through existing core2/audio_pairs.jsonl unchanged."""
    print("[audio] reusing existing audio_pairs.jsonl...")
    rows = []
    src = CORE2 / "audio_pairs.jsonl"
    if src.exists():
        with open(src) as f:
            for line in f:
                d = json.loads(line.strip())
                d["category"] = "audio"
                rows.append(d)
    print(f"  reused: {len(rows)}")
    return rows


def main():
    rng = random.Random(SEED)
    CORE_V2X.mkdir(parents=True, exist_ok=True)

    # Text-side aggregation
    text_buckets = []
    text_buckets += section_alpaca(rng)
    text_buckets += section_gaia(rng)
    text_buckets += section_tools(rng)
    text_buckets += section_deliberation(rng)
    text_buckets += section_multiturn(rng)
    rng.shuffle(text_buckets)
    write_jsonl(CORE_V2X / "text.jsonl", text_buckets)

    # Per-category histogram for sanity
    counts = {}
    for r in text_buckets:
        c = r.get("category", "?")
        counts[c] = counts.get(c, 0) + 1
    print("\n[text histogram]")
    for c in sorted(counts):
        print(f"  {c}: {counts[c]}")
    print(f"  TOTAL: {len(text_buckets)}")

    # Vision
    vision = section_vision_coco(rng)
    write_jsonl(CORE_V2X / "vision_pairs.jsonl", vision)

    # Audio
    audio = section_audio_reuse()
    write_jsonl(CORE_V2X / "audio_pairs.jsonl", audio)

    # Symlink images/audio dirs
    for name in ("images", "audio"):
        src = CORE2 / name
        dst = CORE_V2X / name
        if dst.exists() or dst.is_symlink():
            if dst.is_symlink() or dst.is_file():
                dst.unlink()
            else:
                shutil.rmtree(dst)
        if src.exists():
            os.symlink(src, dst)
            print(f"  symlinked {name}")

    print("\n=== Core 2.x curriculum summary ===")
    print(f"  Text:   {len(text_buckets)}")
    print(f"  Vision: {len(vision)}")
    print(f"  Audio:  {len(audio)}")
    print(f"  TOTAL:  {len(text_buckets) + len(vision) + len(audio)}")


if __name__ == "__main__":
    main()
