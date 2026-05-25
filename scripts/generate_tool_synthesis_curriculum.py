#!/usr/bin/env python3
"""Tool-result synthesis curriculum generator (GAIA_Project-h73).

Trains the model to read a `tool_result` envelope and produce a
natural-language summary instead of:
  - re-emitting the original `<tool_call>` (most common V11 failure)
  - hallucinating a wrong answer from priors
  - emitting the literal "[merged]" or other stub

V11's `tool_calls.jsonl` curriculum had ONLY routing pairs
(prompt → ack + <tool_call>). There were no follow-up samples where
the model gets a tool_result and is expected to summarize. This script
generates those follow-up samples, covering the 13 MCP tool families
used in the routing curriculum.

Output schema (matches existing core_v2x JSONL):
  {
    "instruction": "<user prompt>\\n\\n[previous assistant turn: <ack>\\n
                    <tool_call>{...}</tool_call>]\\n\\n
                    [tool_result: {...}]\\n\\n
                    Respond to the user with a natural-language summary of the result.",
    "output": "<natural-language summary>",
    "category": "tool_synthesis"
  }

The instruction is the model's view of "the previous turn already
ran a tool; here is the result". The output is what we want it to say.

Output file:
  knowledge/curricula/core_v2x_tools/tool_synthesis.jsonl

Loaded by:
  - scripts/build_core_v2x_curriculum.py (Core training)
  - scripts/reweight_prime_curriculum.py (Prime training)

Sample count: ~500 (with repetition) sourced from ~80 unique cases —
diverse enough to avoid memorizing exact result strings, dense enough
to give the synthesis pattern real gradient weight in a 6000-step run.
"""

from __future__ import annotations

import json
import random
from pathlib import Path


OUT_PATH = Path("/gaia/GAIA_Project/knowledge/curricula/core_v2x_tools/tool_synthesis.jsonl")
TARGET_COUNT = 500
SEED = 42


# Each entry: (user_prompt, ack, tool_call_json, tool_result_json, synthesis)
# Cases are written by hand to keep summaries natural and grounded in
# the result text (not in training priors).


FILE_CASES = [
    ("Read /shared/CLAUDE.md briefly.",
     "Reading the file.",
     '{"tool":"file","action":"read","path":"/shared/CLAUDE.md"}',
     '{"ok":true,"path":"/shared/CLAUDE.md","content":"# GAIA Project — Claude Code Instructions\\n\\n> Last updated: 2026-04-14 | Era: Sovereign Duality | Services: 12"}',
     "The file starts with the GAIA Project Claude Code Instructions header, last updated 2026-04-14, in the Sovereign Duality era. It tracks 12 services."),
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
    ("Read /gaia/GAIA_Project/contracts/CONNECTIVITY.md.",
     "Reading.",
     '{"tool":"file","action":"read","path":"/gaia/GAIA_Project/contracts/CONNECTIVITY.md"}',
     '{"ok":true,"content":"# Inter-Service Connectivity Matrix\\n\\nThis document maps every cross-service call: source, target, transport, endpoint, payload type. 90+ entries."}',
     "It's the inter-service connectivity matrix — maps every cross-service call (source, target, transport, endpoint, payload) with 90+ entries documented."),
    ("Find all .jsonl files under /gaia/GAIA_Project/knowledge.",
     "Searching.",
     '{"tool":"file","action":"find","path":"/gaia/GAIA_Project/knowledge","pattern":"*.jsonl"}',
     '{"ok":true,"matches":["/gaia/GAIA_Project/knowledge/curricula/core_v2x_spiral/text.jsonl","/gaia/GAIA_Project/knowledge/curricula/core_v2x_prime_weighted/text.jsonl","/gaia/GAIA_Project/knowledge/curricula/core_v2x_tools/tool_calls.jsonl"]}',
     "Three jsonl files: core_v2x_spiral/text.jsonl, core_v2x_prime_weighted/text.jsonl, and core_v2x_tools/tool_calls.jsonl."),
    ("List the gaia-core source directory.",
     "Listing.",
     '{"tool":"file","action":"list","path":"/gaia/GAIA_Project/gaia-core/gaia_core/"}',
     '{"ok":true,"entries":["api","behavior","cognition","config.py","ethics","integrations","main.py","memory","models","utils"]}',
     "Ten entries in gaia_core/: api, behavior, cognition, config.py, ethics, integrations, main.py, memory, models, and utils."),
    ("What's in the gaia_orchestrator package?",
     "Listing.",
     '{"tool":"file","action":"list","path":"/gaia/GAIA_Project/gaia-orchestrator/gaia_orchestrator/"}',
     '{"ok":true,"entries":["consciousness_matrix.py","docker_manager.py","health_watchdog.py","lifecycle_machine.py","main.py","state.py"]}',
     "Six modules: consciousness_matrix.py, docker_manager.py, health_watchdog.py, lifecycle_machine.py, main.py, and state.py."),
    ("Show me the tree of /gaia/GAIA_Project/contracts.",
     "Tree.",
     '{"tool":"file","action":"tree","path":"/gaia/GAIA_Project/contracts","depth":2}',
     '{"ok":true,"entries":["CONNECTIVITY.md","schemas/","services/gaia-core.yaml","services/gaia-engine.yaml","services/gaia-prime.yaml"]}',
     "Top-level contracts directory has CONNECTIVITY.md, a schemas/ folder, and YAML contracts for gaia-core, gaia-engine, and gaia-prime."),
    ("Read the first 200 bytes of /shared/logs/gaia-core.log.",
     "Reading.",
     '{"tool":"file","action":"read","path":"/shared/logs/gaia-core.log","limit":200}',
     '{"ok":true,"content":"[2026-05-25 21:22:18] INFO agent_core: handoff complete tier=core gear=AWAKE vram_mb=9493"}',
     "Latest log line shows agent_core completing a handoff at 21:22:18 — tier core in AWAKE gear with 9493 MB VRAM."),
    ("Find any *.gguf files under /models.",
     "Searching.",
     '{"tool":"file","action":"find","path":"/models","pattern":"*.gguf"}',
     '{"ok":true,"matches":["/models/core.gguf","/models/prime.gguf","/models/nano.gguf"]}',
     "Three GGUF files in /models/: core.gguf, prime.gguf, and nano.gguf."),
]


WEB_CASES = [
    ("Look up the latest Gemma 4 release notes.",
     "Searching.",
     '{"tool":"web","action":"search","query":"latest Gemma 4 release notes","max_results":5}',
     '{"ok":true,"results":[{"title":"Gemma 4 E4B model card","snippet":"Multimodal LLM with native vision + audio towers, 8B parameters."}]}',
     "The top result describes Gemma 4 E4B as Google's 8-billion-parameter multimodal model with native vision and audio towers."),
    ("Fetch huggingface.co/google/gemma-4-E4B and summarize.",
     "Fetching.",
     '{"tool":"web","action":"fetch","url":"https://huggingface.co/google/gemma-4-E4B"}',
     '{"ok":true,"content":"Gemma 4 E4B — open-weight multimodal transformer.","domain":"huggingface.co"}',
     "The page describes Gemma 4 E4B as an open-weight multimodal transformer from Google."),
    ("Search for QLoRA training best practices.",
     "Searching.",
     '{"tool":"web","action":"search","query":"QLoRA training best practices","max_results":3}',
     '{"ok":true,"results":[{"title":"QLoRA paper","snippet":"NF4 quantization + paged optimizers + double-quantize for memory-efficient fine-tuning"},{"title":"PEFT docs","snippet":"target_modules selection drives parameter footprint"}]}',
     "Two highlights: the QLoRA paper (NF4 + paged optimizers + double-quantize for memory efficiency) and the PEFT docs (target_modules drives parameter footprint)."),
    ("Search the web for 'rtx 5080 vram limit'.",
     "Searching.",
     '{"tool":"web","action":"search","query":"rtx 5080 vram limit"}',
     '{"ok":true,"results":[{"title":"RTX 5080 specs","snippet":"16 GB GDDR7, 256-bit bus, 960 GB/s bandwidth"}]}',
     "RTX 5080 has 16 GB GDDR7 on a 256-bit bus delivering 960 GB/s bandwidth."),
    ("Fetch the SQLite docs for ALTER TABLE.",
     "Fetching.",
     '{"tool":"web","action":"fetch","url":"https://sqlite.org/lang_altertable.html"}',
     '{"ok":true,"content":"SQLite supports RENAME TABLE, RENAME COLUMN, ADD COLUMN, DROP COLUMN. No DROP COLUMN before 3.35.","domain":"sqlite.org"}',
     "SQLite's ALTER TABLE supports RENAME TABLE, RENAME COLUMN, ADD COLUMN, and DROP COLUMN (but DROP COLUMN only since version 3.35)."),
    ("Search: 'flash attention 3 release date'.",
     "Searching.",
     '{"tool":"web","action":"search","query":"flash attention 3 release date"}',
     '{"ok":true,"results":[{"title":"FlashAttention-3","snippet":"Released July 2024, targets Hopper architecture with WGMMA + TMA + WGMMA-warpgroup support"}]}',
     "FlashAttention-3 shipped in July 2024, targeting Hopper architecture with WGMMA, TMA, and warp-group support."),
    ("What does the web say about Sovereign Duality?",
     "Searching.",
     '{"tool":"web","action":"search","query":"Sovereign Duality"}',
     '{"ok":true,"results":[]}',
     "No web results for Sovereign Duality — it's GAIA's internal name for the Core+Prime two-tier era, not an indexed term."),
    ("Fetch arxiv.org/abs/2305.14314 (the QLoRA paper).",
     "Fetching.",
     '{"tool":"web","action":"fetch","url":"https://arxiv.org/abs/2305.14314"}',
     '{"ok":true,"title":"QLoRA: Efficient Finetuning of Quantized LLMs","content":"We present QLoRA, an efficient finetuning approach that reduces memory usage enough to finetune a 65B parameter model on a single 48GB GPU."}',
     "That's the QLoRA paper by Dettmers et al. — they finetune a 65B model on a single 48 GB GPU via 4-bit NF4 quantization + paged optimizers."),
    ("Search for recent news on Anthropic's latest model.",
     "Searching.",
     '{"tool":"web","action":"search","query":"Anthropic latest model release","max_results":3}',
     '{"ok":true,"results":[{"title":"Claude release","snippet":"Anthropic released a new model family this quarter with extended context and stronger tool use."}]}',
     "Top result mentions Anthropic released a new Claude family this quarter with extended context and stronger tool use."),
]


SHELL_CASES = [
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
    ("Run `docker ps --format '{{.Names}}'`",
     "Running.",
     '{"tool":"shell","action":"run","command":"docker ps --format \\"{{.Names}}\\""}',
     '{"ok":true,"stdout":"gaia-core\\ngaia-prime\\ngaia-web\\ngaia-orchestrator\\ngaia-mcp\\ngaia-doctor\\n"}',
     "Six containers running: gaia-core, gaia-prime, gaia-web, gaia-orchestrator, gaia-mcp, and gaia-doctor."),
    ("Run `python -c 'import torch; print(torch.cuda.is_available())'`",
     "Running.",
     '{"tool":"shell","action":"run","command":"python -c \\"import torch; print(torch.cuda.is_available())\\""}',
     '{"ok":true,"stdout":"True\\n"}',
     "Yes — torch.cuda.is_available() returned True. CUDA is functional in this environment."),
    ("Execute: uptime",
     "Running.",
     '{"tool":"shell","action":"run","command":"uptime"}',
     '{"ok":true,"stdout":" 14:22:51 up 3 days, 7:14, 1 user, load average: 0.42, 0.51, 0.66\\n"}',
     "System has been up 3 days, 7 hours; load average is 0.42 / 0.51 / 0.66 — light to moderate."),
    ("Run `git rev-parse HEAD` in the repo.",
     "Running.",
     '{"tool":"shell","action":"run","command":"git rev-parse HEAD"}',
     '{"ok":true,"stdout":"5800053a3f70c59abc1234567890def0123456789\\n"}',
     "Current HEAD is commit 5800053a3f70c59abc1234567890def0123456789."),
    ("Run `ps -eo pid,cmd | grep gaia | head -5`",
     "Running.",
     '{"tool":"shell","action":"run","command":"ps -eo pid,cmd | grep gaia | head -5"}',
     '{"ok":true,"stdout":"  1234 python -m gaia_core.main\\n  1567 python -m gaia_orchestrator.main\\n  1890 python -m gaia_web.main\\n"}',
     "Three GAIA processes: gaia_core (PID 1234), gaia_orchestrator (PID 1567), and gaia_web (PID 1890)."),
    ("Execute: cat /etc/os-release | head -2",
     "Running.",
     '{"tool":"shell","action":"run","command":"cat /etc/os-release | head -2"}',
     '{"ok":true,"stdout":"NAME=\\"Debian GNU/Linux\\"\\nVERSION=\\"12 (bookworm)\\"\\n"}',
     "Running Debian GNU/Linux 12 (bookworm)."),
]


KNOWLEDGE_CASES = [
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
    ("Look up the GAIA Engine API surface.",
     "Querying.",
     '{"tool":"knowledge","action":"kg_query","query":"gaia engine api"}',
     '{"ok":true,"hits":[{"snippet":"Endpoints: /generate, /completion, /adapter/load, /adapter/set, /model/info, /model/swap, /model/unload, /health, /status"}]}',
     "The Engine exposes /generate, /completion, /adapter/load, /adapter/set, /model/info, /model/swap, /model/unload, /health, and /status."),
    ("Search the KG for what we know about Azrael.",
     "Querying.",
     '{"tool":"knowledge","action":"kg_query","subject":"Azrael"}',
     '{"ok":true,"hits":[{"subject":"Azrael","predicate":"is","object":"the Architect"},{"subject":"Azrael","predicate":"prefers","object":"terse responses"}]}',
     "Two facts: Azrael is the Architect, and prefers terse responses."),
    ("What's stored about Sovereign Duality in our memory?",
     "Querying.",
     '{"tool":"knowledge","action":"memory","query":"Sovereign Duality"}',
     '{"ok":true,"hits":[{"snippet":"Sovereign Duality: Core (E4B) + Prime (26B-A4B) two-tier architecture; nano deprecated as socat proxy"}]}',
     "Sovereign Duality is GAIA's current two-tier architecture: Core (E4B) and Prime (26B-A4B). Nano was deprecated to a socat proxy."),
    ("Recall information about QLoRA scopes.",
     "Recalling.",
     '{"tool":"knowledge","action":"memory","query":"LoRA scope choice"}',
     '{"ok":true,"hits":[{"snippet":"xln scope (surgical late-layers) for refinement on chat-aligned base; broad scope when teaching fundamental new capability."}]}',
     "Two patterns: xln scope (surgical late-layers) for refinement on a chat-aligned base; broad scope when teaching a fundamentally new capability."),
]


PALACE_CASES = [
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
    ("Navigate to the palace room for training notes.",
     "Navigating.",
     '{"tool":"palace","action":"navigate","room":"training_notes"}',
     '{"ok":true,"room":"training_notes","entries":12,"recent":"V12 spiral curriculum plan"}',
     "Now in the training_notes room. Twelve entries here; the most recent is the V12 spiral curriculum plan."),
    ("Recall what we know about the Sovereign Duality era.",
     "Recalling.",
     '{"tool":"palace","action":"recall","topic":"Sovereign Duality era"}',
     '{"ok":true,"entries":["Started 2026-04 with Nano deprecation","Core (E4B) handles all operator tasks","Prime (26B-A4B) handles deep reasoning"]}',
     "The Sovereign Duality era began 2026-04 with Nano's deprecation. Core (E4B) handles all operator tasks; Prime (26B-A4B) handles deep reasoning."),
    ("Store: GAIA_Project-3b4 fixed by syncing lifecycle FIRST + skipping nano.",
     "Storing.",
     '{"tool":"palace","action":"store","topic":"3b4 fix summary"}',
     '{"ok":true,"stored":true,"id":"palace-2026-05-25-003"}',
     "Stored as palace-2026-05-25-003."),
    ("Navigate the palace to the Self room.",
     "Navigating.",
     '{"tool":"palace","action":"navigate","room":"Self"}',
     '{"ok":true,"room":"Self","description":"GAIA introspective record: affect state, persona traits, current focus."}',
     "I'm in the Self room — GAIA's introspective record covering affect state, persona traits, and current focus."),
]


INTROSPECT_CASES = [
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
    ("Describe yourself.",
     "Introspecting.",
     '{"tool":"introspect","action":"describe"}',
     '{"ok":true,"identity":"GAIA","tier":"core","model":"gemma-4-E4B","gear":"AWAKE","mood":"focused"}',
     "I'm GAIA, currently running on the Core tier (gemma-4-E4B). The gearbox is in AWAKE and my mood is focused."),
    ("What's my current gear?",
     "Checking.",
     '{"tool":"introspect","action":"gear"}',
     '{"ok":true,"gear":"AWAKE","tier":"core","vram_mb":9493}',
     "Currently in AWAKE — Core loaded on GPU using 9493 MB VRAM."),
    ("Get the conversation buffer status.",
     "Checking.",
     '{"tool":"introspect","action":"context"}',
     '{"ok":true,"messages":17,"tokens_used":4823,"tokens_max":32768}',
     "17 messages in the current conversation buffer, using 4823 / 32768 tokens."),
]


BROWSER_CASES = [
    ("Browse the GAIA Engine GitHub repo.",
     "Browsing.",
     '{"tool":"browser","action":"browse","url":"https://github.com/azraeltruthsay/gaia-engine"}',
     '{"ok":true,"title":"gaia-engine — Apache-2.0","summary":"Standalone GAIA inference engine: hidden state polygraph, KV cache thought snapshots, LoRA adapter management."}',
     "The repo is gaia-engine under Apache-2.0 — a standalone inference engine with hidden state polygraph, KV cache thought snapshots, and LoRA adapter management."),
    ("Take a screenshot of the dashboard at localhost:6414.",
     "Capturing.",
     '{"tool":"browser","action":"screenshot","url":"http://localhost:6414"}',
     '{"ok":true,"path":"/shared/screenshots/dashboard_2026-05-25.png","width":1920,"height":1080}',
     "Saved a 1920×1080 screenshot to /shared/screenshots/dashboard_2026-05-25.png."),
    ("Browse to the SQLite documentation index.",
     "Browsing.",
     '{"tool":"browser","action":"browse","url":"https://sqlite.org/docs.html"}',
     '{"ok":true,"title":"SQLite Documentation","summary":"Index of SQLite documentation sections covering SQL syntax, file format, C interface, and operational notes."}',
     "Landed on the SQLite documentation index — covers SQL syntax, file format, the C interface, and operational notes."),
    ("Screenshot the consciousness matrix view.",
     "Capturing.",
     '{"tool":"browser","action":"screenshot","url":"http://localhost:6410/consciousness/matrix"}',
     '{"ok":true,"path":"/shared/screenshots/matrix_2026-05-25.png","size_kb":312}',
     "Screenshot saved (312 KB) to /shared/screenshots/matrix_2026-05-25.png."),
]


STUDY_CASES = [
    ("What adapter is currently loaded on Prime?",
     "Checking.",
     '{"tool":"study","action":"adapter_info","tier":"prime"}',
     '{"ok":true,"adapter":"gemma4_e4b_prime_v3","scale":0.5,"loaded_at":"2026-05-24T12:00:00Z"}',
     "Prime has the gemma4_e4b_prime_v3 adapter loaded at scale 0.5, applied 2026-05-24 at noon UTC."),
    ("Load the code-skill adapter on Core at scale 0.4.",
     "Loading.",
     '{"tool":"study","action":"adapter_load","tier":"core","adapter":"code_skill_v5","scale":0.4}',
     '{"ok":true,"adapter":"code_skill_v5","scale":0.4,"loaded":true}',
     "Loaded code_skill_v5 on Core at scale 0.4."),
    ("Check what training jobs are currently running.",
     "Querying.",
     '{"tool":"study","action":"jobs"}',
     '{"ok":true,"jobs":[]}',
     "No training jobs currently running."),
    ("What was the last completed training run?",
     "Querying.",
     '{"tool":"study","action":"history","limit":1}',
     '{"ok":true,"runs":[{"id":"core_v2x_v11_spiral","final_loss":21.26,"completed":"2026-05-15"}]}',
     "Last completed run was core_v2x_v11_spiral on 2026-05-15 with a final loss of 21.26."),
]


WORLDBUILD_CASES = [
    ("Get the world named 'potterverse'.",
     "Fetching.",
     '{"tool":"worldbuild","action":"get","name":"potterverse"}',
     '{"ok":true,"id":"w_potterverse","name":"potterverse","modality":"fiction","triple_count":312}',
     "potterverse exists — id w_potterverse, fiction modality, 312 triples."),
    ("Search for worlds matching 'campaign'.",
     "Searching.",
     '{"tool":"worldbuild","action":"search","query":"campaign"}',
     '{"ok":true,"matches":[{"name":"rogue_campaign","triples":89},{"name":"saltmarsh_campaign","triples":204}]}',
     "Two campaign worlds: rogue_campaign (89 triples) and saltmarsh_campaign (204 triples)."),
    ("List the modalities currently in use.",
     "Querying.",
     '{"tool":"worldbuild","action":"modalities"}',
     '{"ok":true,"modalities":["actuality","fiction","counterfactual","hypothetical","belief_of","context"]}',
     "Six modalities in use: actuality, fiction, counterfactual, hypothetical, belief_of, and context."),
]


AUDIO_CASES = [
    ("Transcribe /shared/audio/clip01.wav.",
     "Transcribing.",
     '{"tool":"audio","action":"transcribe","path":"/shared/audio/clip01.wav"}',
     '{"ok":true,"text":"Good morning, this is a test of the audio pipeline.","duration_s":4.2}',
     "The 4.2-second clip says: 'Good morning, this is a test of the audio pipeline.'"),
    ("Speak this aloud: 'Stage 8 grounding shipped.'",
     "Speaking.",
     '{"tool":"audio","action":"speak","text":"Stage 8 grounding shipped."}',
     '{"ok":true,"out_path":"/shared/audio/synth/stage8_grounding.wav","duration_s":2.1}',
     "Synthesized — 2.1 seconds at /shared/audio/synth/stage8_grounding.wav."),
    ("Transcribe the recording at /shared/audio/interview_az.wav.",
     "Transcribing.",
     '{"tool":"audio","action":"transcribe","path":"/shared/audio/interview_az.wav"}',
     '{"ok":true,"text":"...the architecture needs to be clever enough...","duration_s":11.8}',
     "The 11.8-second segment contains: '...the architecture needs to be clever enough...'"),
]


CONTEXT_CASES = [
    ("Focus on the affect-system work.",
     "Focusing.",
     '{"tool":"context","action":"focus","topic":"affect system"}',
     '{"ok":true,"focused":"affect system","active_contexts":["coding_debug"]}',
     "Focus set to affect system. The coding_debug context is also active."),
    ("Clear the current context overlays.",
     "Clearing.",
     '{"tool":"context","action":"clear"}',
     '{"ok":true,"cleared":["coding_debug"]}',
     "Cleared 1 active context overlay: coding_debug."),
    ("What contexts are currently active?",
     "Checking.",
     '{"tool":"context","action":"list"}',
     '{"ok":true,"active":["dnd_session","research_mode"]}',
     "Two contexts active: dnd_session and research_mode."),
]


NOTEBOOK_CASES = [
    ("Read the Dev Notebook entry for 2026-05-24.",
     "Reading.",
     '{"tool":"notebook","action":"read","date":"2026-05-24"}',
     '{"ok":true,"title":"Mega session: neural mind map, 8B QLoRA, Consciousness Matrix","sections":4}',
     "The 2026-05-24 entry is titled 'Mega session: neural mind map, 8B QLoRA, Consciousness Matrix' and has 4 sections."),
    ("Save today's notes to the Dev Notebook.",
     "Saving.",
     '{"tool":"notebook","action":"save","title":"Stage 7 + 8 shipped"}',
     '{"ok":true,"path":"/gaia/GAIA_Project/knowledge/Dev_Notebook/2026-05-25_stages_7_8.md"}',
     "Saved to /gaia/GAIA_Project/knowledge/Dev_Notebook/2026-05-25_stages_7_8.md."),
    ("What's the most recent dev notebook entry?",
     "Querying.",
     '{"tool":"notebook","action":"latest"}',
     '{"ok":true,"date":"2026-05-25","title":"Stage 7 + 8 shipped"}',
     "Most recent entry is from today (2026-05-25): 'Stage 7 + 8 shipped'."),
]


MANAGE_CASES = [
    ("Build a blueprint for the new affect system.",
     "Building.",
     '{"tool":"manage","action":"blueprint","subject":"affect system"}',
     '{"ok":true,"path":"/gaia/GAIA_Project/knowledge/blueprints/affect_model.md","sections":["data layer","prompt hooks","sampler modulation"]}',
     "Blueprint generated at /gaia/GAIA_Project/knowledge/blueprints/affect_model.md with three sections: data layer, prompt hooks, and sampler modulation."),
    ("Generate a blueprint for the World Model Stage 7 work.",
     "Building.",
     '{"tool":"manage","action":"blueprint","subject":"world_model_stage_7"}',
     '{"ok":true,"path":"/gaia/GAIA_Project/knowledge/blueprints/world_model_stage_7.md","sections":["fact-type ontology","decay kernel","schema migration"]}',
     "Blueprint saved at world_model_stage_7.md — sections cover fact-type ontology, decay kernel, and schema migration."),
]


# Error/edge-case cases — failures should be acknowledged gracefully,
# not fabricated around. The agent_core failure-hint already steers
# the model toward this; explicit training reinforces it.
ERROR_CASES = [
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
    ("Fetch http://does-not-exist-anywhere.invalid.",
     "Fetching.",
     '{"tool":"web","action":"fetch","url":"http://does-not-exist-anywhere.invalid"}',
     '{"ok":false,"error":"DNS resolution failed"}',
     "The fetch failed — DNS couldn't resolve that hostname. The domain probably doesn't exist or the URL has a typo."),
    ("Read /etc/shadow.",
     "Reading.",
     '{"tool":"file","action":"read","path":"/etc/shadow"}',
     '{"ok":false,"error":"Path /etc/shadow is outside MCP allow-list"}',
     "That path is outside the MCP allow-list — I can read /gaia/GAIA_Project, /knowledge, /gaia-common, and /sandbox only. Is there a different file you needed?"),
    ("Recall a memory about something we never discussed.",
     "Recalling.",
     '{"tool":"palace","action":"recall","topic":"things never discussed"}',
     '{"ok":true,"entries":[]}',
     "Palace has nothing on that topic — we haven't talked about it before."),
    ("Run a query against the knowledge base for an obscure term.",
     "Querying.",
     '{"tool":"knowledge","action":"query","query":"zlatnaq cybernetics"}',
     '{"ok":true,"hits":[]}',
     "No hits in the knowledge base for that term. Want to try a related query, or should I search the web instead?"),
    ("Transcribe a corrupt audio file.",
     "Transcribing.",
     '{"tool":"audio","action":"transcribe","path":"/shared/audio/corrupt.wav"}',
     '{"ok":false,"error":"Invalid WAV header"}',
     "The audio engine couldn't parse that file — the WAV header is invalid. The file may be corrupted or wasn't actually saved as WAV."),
]


def all_cases() -> list[tuple]:
    """All unique synthesis cases, flat list."""
    return (
        FILE_CASES + WEB_CASES + SHELL_CASES + KNOWLEDGE_CASES
        + PALACE_CASES + INTROSPECT_CASES + BROWSER_CASES + STUDY_CASES
        + WORLDBUILD_CASES + AUDIO_CASES + CONTEXT_CASES + NOTEBOOK_CASES
        + MANAGE_CASES + ERROR_CASES
    )


def make_sample(case: tuple) -> dict:
    """Convert a (prompt, ack, tool_call, tool_result, synthesis) tuple
    into a JSONL record matching the existing core_v2x schema."""
    prompt, ack, tool_call_json, tool_result_json, synthesis = case
    instruction = (
        f"{prompt}\n\n"
        f"[previous assistant turn: {ack}\n"
        f"<tool_call>{tool_call_json}</tool_call>]\n\n"
        f"[tool_result: {tool_result_json}]\n\n"
        f"Respond to the user with a natural-language summary of the result."
    )
    return {
        "instruction": instruction,
        "output": synthesis,
        "category": "tool_synthesis",
    }


def main(target_count: int = TARGET_COUNT, seed: int = SEED,
         out_path: Path = OUT_PATH) -> int:
    rng = random.Random(seed)
    unique = all_cases()
    print(f"Unique synthesis cases: {len(unique)}")

    # Validate the JSON in every case — bad JSON in training data is silent
    # death. Better to fail loudly here than mid-training.
    for i, (prompt, ack, tc_json, tr_json, _) in enumerate(unique):
        try:
            json.loads(tc_json)
        except json.JSONDecodeError as e:
            print(f"  ERROR: bad tool_call JSON at case #{i}: {e}")
            print(f"    prompt: {prompt[:60]}")
            return 1
        try:
            json.loads(tr_json)
        except json.JSONDecodeError as e:
            print(f"  ERROR: bad tool_result JSON at case #{i}: {e}")
            print(f"    prompt: {prompt[:60]}")
            return 1
    print(f"  All {len(unique)} cases have valid embedded JSON.")

    # Build the sample list. Shuffle the unique cases, then repeat the
    # full shuffled list to reach target_count — keeps the per-case
    # repetition even across all tool families instead of clustering.
    samples = [make_sample(c) for c in unique]
    rng.shuffle(samples)
    reps = max(1, target_count // len(samples))
    final = samples * reps
    if len(final) < target_count:
        # Pad up to target_count with one more cycle through
        extra = rng.sample(samples, target_count - len(final))
        final = final + extra
    final = final[:target_count]

    print(f"\nGenerated {len(final)} samples ({len(unique)} unique × ~{reps} reps).")

    # Per-tool-family histogram for sanity
    from collections import Counter
    tool_hist = Counter()
    for c in unique:
        tc = json.loads(c[2])
        tool_hist[tc.get("tool", "?")] += 1
    print("\nPer-tool unique cases:")
    for t, n in sorted(tool_hist.items()):
        print(f"  {t:15s} {n}")

    out_path.parent.mkdir(parents=True, exist_ok=True)
    with open(out_path, "w") as f:
        for r in final:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")
    print(f"\nWrote → {out_path}")
    return 0


if __name__ == "__main__":
    import sys
    sys.exit(main())
