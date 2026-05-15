#!/usr/bin/env python3
"""Generate 3000 tool-calling training samples for V8 curriculum.

V7 audit found that current tool_calling_v1/ samples use the OLD format
`{"tool": "file", "action": "read_file", ...}` but current MCP expects
DOMAIN format `{"tool": "file", "action": "read", ...}`. That mismatch
is why V7 produces broken tool calls (model knows the tag exists but
can't construct the JSON correctly).

This generator:
  1. Iterates the 13 tool families with their REAL action enums (from
     gaia-mcp/list_tools_full)
  2. For each (tool, action), produces 8-30 sample (instruction, output)
     pairs with parameter variety and natural-language instruction variety
  3. Output format: brief acknowledgment + <tool_call>{JSON}</tool_call>
     matching what gaia-core/agent_core.py actually parses

Output: knowledge/curricula/core_v2x_tools/tool_calls.jsonl
Target: ~3000 samples total, category="tool_routing".
"""
import json
import random
import sys
from pathlib import Path


OUT_DIR = Path("/gaia/GAIA_Project/knowledge/curricula/core_v2x_tools")


# (instruction template, ack template, params dict). All use {} placeholders
# resolved from PARAM_VALUES.
#
# tool format: <tool_call>{"tool": "X", "action": "Y", ...params}</tool_call>
# We do NOT include the result — the model's job is only to emit the call.

# Common parameter value pools — EXPANDED for V8 to reduce duplicate rate.
# Generator dupe rate was 80% at v1 because instruction × param permutations
# were too constrained. Each pool now has 20-40+ values.
PARAM_VALUES = {
    "file_path_text": [
        "/shared/notes.md", "/shared/CLAUDE.md", "/shared/knowledge/index.md",
        "/gaia/GAIA_Project/README.md", "/shared/logs/today.log",
        "/shared/sessions.json", "/shared/training_runs/latest/summary.json",
        "/shared/doctor/cognitive_test_results.json",
        "/shared/kvcache/core/identity_prefix.pt",
        "/shared/docs_drafts/AS_BUILT_LATEST.md",
        "/gaia/GAIA_Project/COUNCIL_CHAMBER.md",
        "/gaia/GAIA_Project/knowledge/Dev_Notebook/TODO.md",
        "/shared/maintenance_mode.json",
        "/etc/hostname", "/etc/hosts", "/etc/os-release",
        "/proc/cpuinfo", "/proc/meminfo", "/proc/uptime",
        "/gaia/GAIA_Project/gaia_constants.json",
        "/gaia/GAIA_Project/contracts/CONNECTIVITY.md",
        "/shared/training_runs/core_run_1778622218_core2x_v7/config.json",
        "/shared/training_runs/core_run_1778622218_core2x_v7/summary.json",
        "/home/azrael/.bashrc", "/home/azrael/.gitconfig",
    ],
    "file_path_code": [
        "/gaia/GAIA_Project/gaia-core/gaia_core/main.py",
        "/gaia/GAIA_Project/scripts/train_core_multimodal.py",
        "/gaia/GAIA_Project/gaia-mcp/gaia_mcp/server.py",
        "/gaia/GAIA_Project/gaia-engine/gaia_engine/core.py",
        "/gaia/GAIA_Project/gaia-core/gaia_core/cognition/agent_core.py",
        "/gaia/GAIA_Project/gaia-orchestrator/gaia_orchestrator/main.py",
        "/gaia/GAIA_Project/gaia-doctor/cognitive_test_battery.py",
        "/gaia/GAIA_Project/gaia-common/gaia_common/lifecycle/states.py",
        "/gaia/GAIA_Project/scripts/audit_curriculum.py",
        "/gaia/GAIA_Project/scripts/merge_and_save_adapter.py",
        "/gaia/GAIA_Project/gaia-web/static/index.html",
        "/gaia/GAIA_Project/gaia-monkey/monkey.py",
    ],
    "dir_path": [
        "/shared", "/shared/logs", "/gaia/GAIA_Project/scripts",
        "/shared/training_runs", "/shared/sessions.archived",
        "/gaia/GAIA_Project/gaia-core", "/gaia/GAIA_Project/contracts",
        "/gaia/GAIA_Project/knowledge", "/gaia/GAIA_Project/knowledge/curricula",
        "/gaia/GAIA_Project/knowledge/blueprints",
        "/gaia/GAIA_Project/gaia-mcp", "/gaia/GAIA_Project/gaia-engine",
        "/gaia/GAIA_Project/gaia-orchestrator", "/gaia/GAIA_Project/candidates",
        "/shared/doctor", "/shared/kvcache", "/shared/doctor_fixtures",
        "/gaia/gaia-instance/gaia-models", "/var/log", "/etc",
        "/tmp", "/home/azrael", "/root",
    ],
    "search_pattern": [
        "*.py", "*.md", "*.json", "*.jsonl", "test_*.py", "*v6*", "*v7*",
        "*.yaml", "*.yml", "*.sh", "*.png", "*.jpg", "*.wav", "*.log",
        "build_*.py", "download_*.py", "merge_*.py", "scripts/*.py",
        "Core2X_*", "checkpoint-*", "*.safetensors",
    ],
    "web_query": [
        "latest Gemma 4 release notes", "PyTorch 2.5 changelog",
        "Docker compose v3 reference", "LoRA training best practices",
        "RTX 5080 thermal specs", "transformers tokenizer special tokens",
        "Python 3.11 release date", "JSON schema specification",
        "MCP protocol specification", "vLLM inference optimization",
        "best practices for fine-tuning multimodal models",
        "Hugging Face PEFT documentation", "bitsandbytes NF4 quantization",
        "audio token alignment in Gemma 4", "ESC-50 dataset baseline",
        "LibriSpeech ASR benchmarks", "CUDA OOM debugging strategies",
        "PyTorch flash attention", "Linux process memory inspection",
        "regex for stripping ANSI escape codes", "Python pathlib examples",
        "vector database comparison", "ChromaDB vs FAISS",
        "sentence embedding models 2026", "RAG retrieval thresholds",
        "Docker container restart policies", "git rebase interactive guide",
        "kubernetes vs docker compose", "nginx reverse proxy config",
        "systemd service tutorial", "TLS certificate renewal",
    ],
    "web_url": [
        "https://github.com/anthropics/anthropic-cookbook",
        "https://huggingface.co/google/gemma-4-E4B",
        "https://docs.python.org/3.11/",
        "https://pytorch.org/docs/stable/index.html",
        "https://docs.docker.com/compose/",
        "https://github.com/openai/openai-python",
        "https://huggingface.co/datasets/openslr/librispeech_asr",
        "https://huggingface.co/datasets/ashraq/esc50",
        "https://github.com/Dao-AILab/flash-attention",
        "https://docs.nvidia.com/deeplearning/cudnn/",
        "https://modelcontextprotocol.io/specification",
        "https://en.wikipedia.org/wiki/LoRA_(machine_learning)",
        "https://arxiv.org/abs/2401.00001",
        "https://github.com/huggingface/transformers",
        "https://docs.python.org/3/library/pathlib.html",
    ],
    "knowledge_query": [
        "what is the GPU lifecycle gearbox",
        "how does the consciousness matrix work",
        "what are GAIA's tier models",
        "what is the Sovereign Duality architecture",
        "what tools does gaia-mcp expose",
        "how does Memento-style skill learning work",
        "what is the cognitive pipeline structure",
        "how does the immune system watchdog work",
        "what is the prefix cache",
        "how do we save merged LoRA models",
        "what is the spiral curriculum design",
        "how does the per-category loss logger work",
        "what is the audit_curriculum tool",
        "how does the engine handle vision input",
        "what is the chord protocol",
        "how does the Discord bridge connect",
        "what is the maintenance mode flag",
        "what is the safety blast shield",
    ],
    "memory_query": [
        "recent training runs",
        "yesterday's cognitive battery results",
        "Azrael's recent design discussions",
        "the last vision capability test",
        "what bugs are open in beads",
        "skills GAIA learned from past sessions",
        "the V7 training decision",
        "audio data acquisition history",
        "vision battery infrastructure change",
        "stale curriculum cleanup pass",
        "recent gear shifts and transitions",
        "energy consumption over the last 24h",
        "tool calls that succeeded today",
        "the V5 polish discussion",
    ],
    "shell_cmd_safe": [
        "ls -la /shared/", "df -h /gaia", "free -h",
        "docker ps --format 'table {{.Names}}\\t{{.Status}}'",
        "nvidia-smi --query-gpu=memory.used,memory.free --format=csv",
        "uptime", "git log --oneline -10", "git status",
        "ps aux | head -20", "du -sh /shared/*",
        "cat /proc/cpuinfo | head -20", "lscpu",
        "ip a", "netstat -tlnp",
        "docker logs gaia-core --tail 50",
        "docker logs gaia-mcp --tail 50",
        "find /shared -mtime -1 -type f",
        "ls -la /shared/training_runs/ | tail -10",
        "wc -l /gaia/GAIA_Project/knowledge/curricula/core_v2x/text.jsonl",
        "git diff --stat HEAD~5",
        "git log --oneline --all -20",
        "find . -name '*.jsonl' -size +1M",
        "du -sh /shared/training_runs/*",
        "head -3 /shared/sessions.json",
        "tail -50 /shared/logs/today.log",
        "docker stats --no-stream",
        "systemctl status docker",
    ],
    "topic": [
        "yesterday's training discussion", "the V7 release decision",
        "Azrael's preferences for terse responses",
        "the vision capability gap",
        "audio data acquisition strategy",
        "the curriculum hygiene pattern",
        "Memento-style skill creation",
        "the consciousness matrix transitions",
        "Sovereign Duality model architecture",
        "GAIA Engine's NF4 dequantization",
        "the cognitive battery design",
        "the Robust Channel watchdog issue",
        "spiral curriculum phasing",
        "vision tower calibration",
    ],
}


def fmt_tool_call(tool: str, action: str, **params) -> str:
    """Format a tool call as the model should emit it."""
    obj = {"tool": tool, "action": action}
    obj.update(params)
    return f"<tool_call>{json.dumps(obj, ensure_ascii=False)}</tool_call>"


def sample(rng, key: str) -> str:
    """Pick a random value from PARAM_VALUES[key]."""
    return rng.choice(PARAM_VALUES[key])


# ── Template generators per tool family ─────────────────────────────────────


def gen_file(rng) -> list[dict]:
    samples = []
    read_instrs = [
        ("Read the file at {path}.", "I'll read that file."),
        ("Show me the contents of {path}.", "Let me read it."),
        ("Open {path} and tell me what's in it.", "Reading the file."),
        ("What's in {path}?", "I'll check."),
        ("Display {path}.", "Reading."),
        ("Pull up the contents of {path}.", "I'll read it for you."),
        ("Cat {path} for me.", "Let me read that file."),
    ]
    all_paths = PARAM_VALUES["file_path_text"] + PARAM_VALUES["file_path_code"]
    for path in all_paths:
        for instr_tpl, ack in read_instrs[:3]:  # cap to keep balance
            instr = instr_tpl.format(path=path)
            out = f"{ack}\n{fmt_tool_call('file', 'read', path=path)}"
            samples.append({"instruction": instr, "output": out, "category": "tool_routing"})

    list_instrs = [
        ("What files are in {path}?", "I'll list the directory contents."),
        ("List the contents of {path}.", "Listing files."),
        ("Show me what's in {path}.", "Let me check the directory."),
        ("Ls {path}.", "I'll list it."),
        ("What's in the directory {path}?", "Let me see."),
    ]
    for path in PARAM_VALUES["dir_path"]:
        for instr_tpl, ack in list_instrs[:3]:
            instr = instr_tpl.format(path=path)
            out = f"{ack}\n{fmt_tool_call('file', 'list', path=path)}"
            samples.append({"instruction": instr, "output": out, "category": "tool_routing"})

    tree_instrs = [
        ("Show me the tree under {path}.", "I'll get the recursive tree view."),
        ("Recursively list {path}.", "Getting the full tree."),
        ("Tree view of {path}.", "I'll fetch the tree."),
    ]
    for path in PARAM_VALUES["dir_path"][:15]:
        for instr_tpl, ack in tree_instrs:
            instr = instr_tpl.format(path=path)
            out = f"{ack}\n{fmt_tool_call('file', 'tree', path=path)}"
            samples.append({"instruction": instr, "output": out, "category": "tool_routing"})

    find_instrs = [
        ("Find all {pattern} files under {path}.", "I'll search for matching files."),
        ("Locate {pattern} files in {path}.", "Searching."),
        ("Search for {pattern} in {path}.", "I'll look for them."),
    ]
    # Pair up patterns and paths to avoid combinatorial explosion
    paths = PARAM_VALUES["dir_path"]
    patterns = PARAM_VALUES["search_pattern"]
    for i, pattern in enumerate(patterns):
        path = paths[i % len(paths)]
        for instr_tpl, ack in find_instrs:
            instr = instr_tpl.format(pattern=pattern, path=path)
            out = f"{ack}\n{fmt_tool_call('file', 'find', path=path, pattern=pattern)}"
            samples.append({"instruction": instr, "output": out, "category": "tool_routing"})

    write_instrs = [
        ("Write 'Hello' to {path}.", "I'll write that. (Note: requires approval.)"),
        ("Save the text 'X' to {path}.", "Writing the file. (Sensitive op — approval required.)"),
        ("Create a file at {path} with some content.", "I'll write the file. (Approval required.)"),
    ]
    contents = ["Hello world", "test content", "GAIA was here",
                "Generated by GAIA", "Sample text", "Placeholder"]
    paths_for_write = PARAM_VALUES["file_path_text"][:10]
    for path in paths_for_write:
        for (instr_tpl, ack), content in zip(write_instrs, contents):
            instr = instr_tpl.format(path=path)
            out = f"{ack}\n{fmt_tool_call('file', 'write', path=path, content=content)}"
            samples.append({"instruction": instr, "output": out, "category": "tool_routing"})

    return samples


def gen_web(rng) -> list[dict]:
    samples = []
    search_instrs = [
        ("Search the web for {q}.", "I'll search the web."),
        ("Look up {q} online.", "Searching."),
        ("What does the web say about {q}?", "Let me search."),
        ("Find information about {q}.", "I'll look it up."),
        ("Google {q} for me.", "I'll search the web."),
        ("Search online for {q}.", "Querying the web."),
        ("Web search: {q}", "Searching."),
    ]
    for q in PARAM_VALUES["web_query"]:
        for instr_tpl, ack in search_instrs:
            instr = instr_tpl.format(q=q)
            out = f"{ack}\n{fmt_tool_call('web', 'search', query=q, max_results=5)}"
            samples.append({"instruction": instr, "output": out, "category": "tool_routing"})

    fetch_instrs = [
        ("Fetch the page at {url}.", "I'll fetch that page."),
        ("Get the contents of {url}.", "Fetching."),
        ("Pull up {url} for me.", "I'll grab the page."),
        ("Download {url}.", "Fetching the URL."),
        ("Retrieve {url}.", "Retrieving."),
        ("Read the page at {url}.", "Reading the URL."),
    ]
    for url in PARAM_VALUES["web_url"]:
        for instr_tpl, ack in fetch_instrs:
            instr = instr_tpl.format(url=url)
            out = f"{ack}\n{fmt_tool_call('web', 'fetch', url=url)}"
            samples.append({"instruction": instr, "output": out, "category": "tool_routing"})

    return samples


def gen_shell(rng) -> list[dict]:
    samples = []
    shell_instrs = [
        ("Run the command `{cmd}`.", "I'll execute that. (Requires approval.)"),
        ("Execute: {cmd}", "Running the command. (Sensitive — approval needed.)"),
        ("Can you run {cmd}?", "I'll run it. (Shell requires approval.)"),
        ("Please run `{cmd}`.", "I'll execute it. (Approval needed.)"),
        ("Try `{cmd}`.", "Running. (Shell requires approval.)"),
        ("Shell: {cmd}", "Executing the shell command. (Sensitive op.)"),
        ("Invoke `{cmd}`.", "Invoking. (Approval required.)"),
        ("Fire off {cmd}", "Firing the command. (Shell approval needed.)"),
    ]
    for cmd in PARAM_VALUES["shell_cmd_safe"]:
        for instr_tpl, ack in shell_instrs:
            instr = instr_tpl.format(cmd=cmd)
            out = f"{ack}\n{fmt_tool_call('shell', 'run', command=cmd)}"
            samples.append({"instruction": instr, "output": out, "category": "tool_routing"})
    return samples


def gen_knowledge(rng) -> list[dict]:
    samples = []
    # knowledge.query / search
    for action in ["query", "search"]:
        instrs = [
            (f"What does the knowledge base say about {{q}}?", f"I'll {action} the knowledge base."),
            (f"{action.capitalize()} the knowledge base for {{q}}.", f"{action.capitalize()}ing."),
        ]
        for _ in range(40):
            instr_tpl, ack = rng.choice(instrs)
            q = sample(rng, "knowledge_query")
            instr = instr_tpl.format(q=q)
            out = f"{ack}\n{fmt_tool_call('knowledge', action, query=q)}"
            samples.append({"instruction": instr, "output": out, "category": "tool_routing"})

    # knowledge.memory (recall)
    mem_instrs = [
        ("Recall from memory: {q}", "I'll search my memory."),
        ("What do you remember about {q}?", "Let me check my memory."),
    ]
    for _ in range(40):
        instr_tpl, ack = rng.choice(mem_instrs)
        q = sample(rng, "memory_query")
        instr = instr_tpl.format(q=q)
        out = f"{ack}\n{fmt_tool_call('knowledge', 'memory', query=q)}"
        samples.append({"instruction": instr, "output": out, "category": "tool_routing"})

    # knowledge.status / rebuild / list / kg_stats / kg_timeline — no params
    no_param_actions = {
        "status": (
            ["What's the knowledge base status?", "Knowledge base status?",
             "Is the KB healthy?", "How's the knowledge index doing?",
             "Show KB status.", "Check the knowledge base."],
            ["I'll check the status.", "Querying KB state.",
             "Fetching status.", "Let me check."],
        ),
        "rebuild": (
            ["Rebuild the knowledge index.", "Re-index the knowledge base.",
             "Refresh the KB index.", "Reindex knowledge.",
             "Rebuild KB embeddings."],
            ["I'll rebuild it.", "Triggering a rebuild.",
             "Starting re-indexing.", "Reindexing."],
        ),
        "list": (
            ["List all knowledge bases.", "Show available knowledge stores.",
             "What KBs are loaded?", "Enumerate the knowledge bases.",
             "Tell me which KBs exist."],
            ["Listing them.", "Fetching the KB list.",
             "Querying registry.", "Let me check."],
        ),
        "kg_stats": (
            ["Show knowledge graph statistics.", "What's the KG size?",
             "Knowledge graph stats?", "How big is the knowledge graph?",
             "Tell me the KG statistics."],
            ["I'll fetch the stats.", "Querying KG.",
             "Retrieving stats.", "Let me check."],
        ),
        "kg_timeline": (
            ["Show the recent knowledge timeline.", "Pull up the KG timeline.",
             "What's on the knowledge timeline?", "Display recent KG events.",
             "Fetch the timeline."],
            ["I'll fetch the timeline.", "Retrieving timeline.",
             "Fetching.", "Let me check."],
        ),
    }
    for action, (instrs, acks) in no_param_actions.items():
        for instr in instrs:
            for ack in acks:
                out = f"{ack}\n{fmt_tool_call('knowledge', action)}"
                samples.append({"instruction": instr, "output": out, "category": "tool_routing"})

    # kg_query — varies by entity
    entities = ["GAIA", "Azrael", "Gemma 4", "Sovereign Duality",
                "the Consciousness Matrix", "the GAIA Engine",
                "Memento Skills", "tool routing", "the cognitive battery",
                "the V7 training run"]
    kg_query_instrs = ["Query the knowledge graph for entity '{e}'.",
                       "Look up '{e}' in the KG.",
                       "What does the knowledge graph know about '{e}'?",
                       "Find '{e}' in the knowledge graph.",
                       "Pull KG data on '{e}'."]
    for e in entities:
        for tpl in kg_query_instrs:
            out = f"I'll query the knowledge graph.\n{fmt_tool_call('knowledge', 'kg_query', entity=e)}"
            samples.append({"instruction": tpl.format(e=e), "output": out, "category": "tool_routing"})

    return samples


def gen_palace(rng) -> list[dict]:
    samples = []
    topic_actions = {
        "store": [("Store this in the memory palace: {x}", "I'll store it."),
                  ("Save to memory palace: {x}", "Storing."),
                  ("Remember this for later: {x}", "Saved to memory palace."),
                  ("Add to my palace: {x}", "Adding to palace."),
                  ("Persist this in palace memory: {x}", "Persisting.")],
        "recall": [("Recall from memory palace: {x}", "Recalling."),
                   ("What does memory palace say about {x}?", "Let me recall."),
                   ("Pull up memory of {x}", "Checking memory."),
                   ("Search palace for {x}", "Searching palace."),
                   ("Retrieve palace entry on {x}", "Retrieving.")],
        "navigate": [("Navigate the memory palace to {x}", "Navigating."),
                     ("Move to the room about {x}", "Navigating there."),
                     ("Go to the palace room for {x}", "Going there."),
                     ("Walk to the {x} room in palace", "Walking over.")],
    }
    for action, instrs in topic_actions.items():
        for x in PARAM_VALUES["topic"]:
            for tpl_instr, tpl_ack in instrs:
                out = f"{tpl_ack}\n{fmt_tool_call('palace', action, topic=x)}"
                samples.append({"instruction": tpl_instr.format(x=x), "output": out, "category": "tool_routing"})

    status_instrs = ["What's the memory palace status?", "Is the memory palace healthy?",
                     "Show palace stats.", "How's the palace doing?",
                     "Palace status?", "Tell me palace health.",
                     "Check the memory palace.", "Show palace metrics."]
    status_acks = ["I'll check.", "Checking status.",
                   "Fetching status.", "Querying palace."]
    for instr in status_instrs:
        for ack in status_acks:
            out = f"{ack}\n{fmt_tool_call('palace', 'status')}"
            samples.append({"instruction": instr, "output": out, "category": "tool_routing"})
    return samples


def gen_introspect(rng) -> list[dict]:
    samples = []
    action_variants = {
        "world": (
            ["What's the current world state?", "Show me the world state.",
             "Tell me about the current world.", "What's happening right now in the world model?",
             "Give me the world state summary.", "What does my world model look like?",
             "Snapshot the current world state.", "What's GAIA's view of the world right now?"],
            ["I'll check the world state.", "Fetching world state.",
             "Let me see.", "Looking at the world model.",
             "Retrieving the snapshot."],
        ),
        "recall": (
            ["Recall recent events.", "What happened recently?",
             "Show me recent activity.", "Pull up the recent event log.",
             "What's been going on lately?", "Give me a recap of recent events.",
             "Show recent memory entries.", "What do I remember from recently?"],
            ["Looking up recent events.", "Recalling.",
             "I'll fetch the recent log.", "Let me check."],
        ),
        "logs": (
            ["Show recent system logs.", "What are the latest logs saying?",
             "Pull up the system logs.", "Tail the recent logs.",
             "Give me the last few log entries.", "What's in the logs?",
             "Display recent log activity.", "Check the logs for me."],
            ["Fetching logs.", "I'll pull the logs.",
             "Let me grab them.", "Retrieving."],
        ),
        "tools": (
            ["List all available tools.", "What tools do I have access to?",
             "Show me my toolset.", "Enumerate the tools.",
             "What's in my MCP toolbox?", "List my capabilities.",
             "Tell me what tools are registered.", "Give me the tool roster."],
            ["I'll list them.", "Listing tools.",
             "Fetching the tool list.", "Here are the tools."],
        ),
        "count_chars": (
            ["Count characters in this text.", "How many characters is that?",
             "Give me the character count.", "Count the chars."],
            ["I'll count.", "Counting.",
             "Let me check.", "Computing the count."],
        ),
    }
    for action, (instrs, acks) in action_variants.items():
        for instr in instrs:
            for ack in acks:
                out = f"{ack}\n{fmt_tool_call('introspect', action)}"
                samples.append({"instruction": instr, "output": out, "category": "tool_routing"})
    # describe — varies on target
    describe_instrs = [
        "Describe the {tool_name} tool.", "Tell me about the {tool_name} tool.",
        "What does the {tool_name} tool do?", "Give me details on {tool_name}.",
        "Explain the {tool_name} tool.", "What's the {tool_name} tool for?",
    ]
    describe_acks = ["I'll fetch the description.", "Let me look it up.",
                     "Retrieving tool info.", "Fetching."]
    targets = ["file", "web", "knowledge", "palace", "shell", "browser",
               "audio", "study", "notebook", "context", "manage", "worldbuild"]
    for t in targets:
        for instr_tpl in describe_instrs:
            ack = rng.choice(describe_acks)
            out = f"{ack}\n{fmt_tool_call('introspect', 'describe', target=t)}"
            samples.append({"instruction": instr_tpl.format(tool_name=t),
                            "output": out, "category": "tool_routing"})
    return samples


def gen_browser(rng) -> list[dict]:
    samples = []
    no_param = {
        "snapshot": (
            ["Take a snapshot of the current page.", "Snapshot this page.",
             "Capture the current page state.", "Get a snapshot.",
             "Save the current page snapshot.", "Take a DOM snapshot."],
            ["Snapshotting.", "Capturing.", "Taking the snapshot.",
             "Saving page state."],
        ),
        "links": (
            ["List the links on the current page.", "What links are on this page?",
             "Show all hyperlinks.", "Enumerate the page's links.",
             "Pull up the links from this page.", "What URLs does this page link to?"],
            ["I'll list them.", "Fetching links.", "Enumerating.",
             "Let me check."],
        ),
        "forms": (
            ["What forms are on this page?", "List the forms on the page.",
             "Show the form elements.", "Enumerate the forms.",
             "What input forms exist here?", "Pull up the page's forms."],
            ["Checking for forms.", "Listing forms.",
             "Fetching form elements.", "Enumerating."],
        ),
    }
    for action, (instrs, acks) in no_param.items():
        for instr in instrs:
            for ack in acks:
                out = f"{ack}\n{fmt_tool_call('browser', action)}"
                samples.append({"instruction": instr, "output": out, "category": "tool_routing"})

    browse_instrs = ["Browse to {url}", "Open {url} in the browser",
                     "Navigate to {url}", "Load the page at {url}",
                     "Go to {url}"]
    shot_instrs = ["Take a screenshot of {url}", "Screenshot {url}",
                   "Grab a screenshot of {url}", "Capture {url} as an image"]
    urls = PARAM_VALUES["web_url"]
    for url in urls:
        for tpl in browse_instrs:
            out = f"I'll open that page.\n{fmt_tool_call('browser', 'browse', url=url)}"
            samples.append({"instruction": tpl.format(url=url), "output": out, "category": "tool_routing"})
        for tpl in shot_instrs:
            out = f"Taking a screenshot.\n{fmt_tool_call('browser', 'screenshot', url=url)}"
            samples.append({"instruction": tpl.format(url=url), "output": out, "category": "tool_routing"})
    return samples


def gen_audio_tool(rng) -> list[dict]:
    """Audio TOOL (not audio modality) — listening control + inbox management."""
    samples = []
    action_variants = {
        "listen_start": (
            ["Start listening for voice input.", "Turn on the microphone.",
             "Begin audio listening.", "Activate voice input.",
             "Start the STT listener.", "Open the audio channel.",
             "Listen for my voice.", "Start the speech recognizer.",
             "Engage audio capture."],
            ["I'll start the audio listener.", "Starting voice input.",
             "Activating the microphone.", "Engaging audio capture.",
             "Listener online."],
        ),
        "listen_stop": (
            ["Stop listening.", "Turn off the microphone.",
             "End audio listening.", "Disable voice input.",
             "Stop the STT listener.", "Close the audio channel.",
             "Pause the listener.", "Mute the mic."],
            ["Stopping the listener.", "Disengaging audio capture.",
             "Voice input off.", "Listener stopped."],
        ),
        "listen_status": (
            ["Is the audio listener running?", "What's the mic status?",
             "Is the microphone on?", "Check the audio listener.",
             "Is voice input active?", "Status of the STT listener?",
             "Audio channel status?"],
            ["I'll check the status.", "Checking listener state.",
             "Let me see.", "Fetching status."],
        ),
        "inbox_status": (
            ["What's in the audio inbox?", "Check the audio inbox.",
             "Is there pending audio?", "Audio inbox status?",
             "Any new audio messages?", "How many unprocessed audio items?"],
            ["Checking the inbox.", "Fetching inbox state.",
             "Let me see.", "Querying the audio inbox."],
        ),
        "inbox_list": (
            ["List unprocessed audio messages.", "Show me the pending audio.",
             "What audio items are in the queue?", "Enumerate the audio inbox.",
             "List the audio backlog."],
            ["Listing inbox.", "Fetching the list.",
             "Retrieving pending audio.", "Let me pull them up."],
        ),
    }
    for action, (instrs, acks) in action_variants.items():
        for instr in instrs:
            for ack in acks:
                out = f"{ack}\n{fmt_tool_call('audio', action)}"
                samples.append({"instruction": instr, "output": out, "category": "tool_routing"})
    return samples


def gen_study(rng) -> list[dict]:
    samples = []
    no_param = {
        "status": (
            ["What's the training status?", "How's training going?",
             "Is study still running?", "Check training progress.",
             "Where's the training run at?", "What's gaia-study doing?",
             "Tell me the current training step.", "Is a training job active?"],
            ["I'll check training status.", "Querying gaia-study.",
             "Fetching status.", "Let me see."],
        ),
        "adapter_list": (
            ["List available LoRA adapters.", "Show all adapters.",
             "What adapters are loaded?", "Enumerate the LoRA adapters.",
             "What's in the adapter registry?", "Tell me which adapters exist.",
             "List the available models I can swap into."],
            ["Listing adapters.", "Fetching the adapter list.",
             "Retrieving registry.", "Let me check."],
        ),
    }
    for action, (instrs, acks) in no_param.items():
        for instr in instrs:
            for ack in acks:
                out = f"{ack}\n{fmt_tool_call('study', action)}"
                samples.append({"instruction": instr, "output": out, "category": "tool_routing"})

    adapter_names = ["core_deliberation_v1", "gaia_architecture", "code_skill_v3",
                     "core2x_v6", "core2x_v7", "vision_calibration", "tool_calling_v1",
                     "memento_skills_v2", "speech_diversity_v8"]
    info_instrs = [
        "Tell me about the {a} adapter.", "What's in the {a} adapter?",
        "Show details on {a}.", "Get info on the {a} adapter.",
        "Describe the {a} adapter.",
    ]
    load_instrs = [
        "Load the {a} adapter.", "Switch to {a}.",
        "Activate the {a} adapter.", "Apply {a}.",
        "Use the {a} adapter for inference.",
    ]
    for a in adapter_names:
        for tpl in info_instrs:
            out = f"I'll fetch info.\n{fmt_tool_call('study', 'adapter_info', adapter_name=a)}"
            samples.append({"instruction": tpl.format(a=a), "output": out, "category": "tool_routing"})
        for tpl in load_instrs:
            out = f"Loading the adapter.\n{fmt_tool_call('study', 'adapter_load', adapter_name=a)}"
            samples.append({"instruction": tpl.format(a=a), "output": out, "category": "tool_routing"})
    return samples


def gen_notebook(rng) -> list[dict]:
    samples = []
    # list — no params
    list_instrs = ["List my notebooks.", "Show all notebooks.",
                   "What notebooks exist?", "Enumerate the notebooks.",
                   "Pull up the notebook index.", "What's in my notebook collection?",
                   "Show the notebook list."]
    list_acks = ["Listing notebooks.", "Fetching the list.",
                 "Let me check.", "Retrieving."]
    for instr in list_instrs:
        for ack in list_acks:
            out = f"{ack}\n{fmt_tool_call('notebook', 'list')}"
            samples.append({"instruction": instr, "output": out, "category": "tool_routing"})

    notebooks = ["research-2026", "gaia-architecture", "training-log",
                 "vision-battery", "core-v2x-prep", "audio-curriculum",
                 "engine-extraction", "council-chamber"]
    notes_instrs = ["Show notes from notebook {nb}.", "What's in the {nb} notebook?",
                    "Pull up notes from {nb}.", "Display {nb} notes.",
                    "Read the {nb} notebook."]
    for nb in notebooks:
        for tpl in notes_instrs:
            out = f"I'll fetch notes.\n{fmt_tool_call('notebook', 'notes', notebook_id=nb)}"
            samples.append({"instruction": tpl.format(nb=nb), "output": out, "category": "tool_routing"})

    titles_contents = [
        ("Today's V7 results", "Battery scored 65.9%."),
        ("Vision battery follow-up", "Vision still hallucinates animals."),
        ("V8 curriculum plan", "Triple tool routing, fix transcription prompts, anti-confab samples."),
        ("PARAM_VALUES expansion", "Tripled pool sizes to fix 80% dup rate."),
        ("Sovereign Duality migration", "Two-tier Gemma 4 architecture: Core (E4B) and Prime (26B-A4B)."),
        ("Consciousness Matrix gear shifts", "PARKED → AWAKE ↔ FOCUSING → PARKED transitions tested."),
        ("Memento skill graph", "SkillManager + knowledge_router utility scoring."),
        ("Engine extraction", "gaia-engine moved to separate repo; gaia_common acts as shim."),
    ]
    create_instrs = ["Create a note titled '{t}' with content '{c}'.",
                     "Make a note: '{t}' — {c}",
                     "Add a notebook entry titled '{t}' with body '{c}'.",
                     "Save this note: title='{t}', content='{c}'."]
    for t, c in titles_contents:
        for tpl in create_instrs:
            out = f"I'll create the note.\n{fmt_tool_call('notebook', 'create_note', title=t, content=c)}"
            samples.append({"instruction": tpl.format(t=t, c=c), "output": out, "category": "tool_routing"})
    return samples


def gen_context(rng) -> list[dict]:
    samples = []
    no_param = {
        "compress": (
            ["Compress the current context.", "Squeeze down the context.",
             "Run a context compression pass.", "Shrink the active context.",
             "Compress what's loaded.", "Summarize and compress the context.",
             "Trim the context."],
            ["I'll compress it.", "Running compression.",
             "Compressing.", "Squeezing."],
        ),
        "status": (
            ["What's the context status?", "How big is my context right now?",
             "Show context utilization.", "What's loaded in context?",
             "Context state?", "How full is the context window?",
             "Tell me the context size."],
            ["I'll check.", "Fetching status.",
             "Let me see.", "Querying context state."],
        ),
        "rolling": (
            ["Show the rolling context.", "Pull up the rolling buffer.",
             "What's in the rolling context window?", "Display the rolling memory.",
             "Tail the rolling context.", "Show what's currently rolling."],
            ["Fetching.", "Retrieving rolling context.",
             "Let me grab it.", "Pulling the buffer."],
        ),
        "fragment_list": (
            ["List pending context fragments.", "What fragments are queued?",
             "Show me the unmerged context fragments.", "Enumerate fragments.",
             "What's in the fragment queue?"],
            ["Listing fragments.", "Fetching the queue.",
             "Retrieving.", "Let me check."],
        ),
    }
    for action, (instrs, acks) in no_param.items():
        for instr in instrs:
            for ack in acks:
                out = f"{ack}\n{fmt_tool_call('context', action)}"
                samples.append({"instruction": instr, "output": out, "category": "tool_routing"})

    focus_topics = ["training history", "user preferences", "current bug",
                    "the V8 plan", "GAIA's architecture", "consciousness matrix gear shifts",
                    "the audio curriculum", "the recent vision findings",
                    "session continuity work", "the engine extraction"]
    focus_instrs = ["Focus context on {topic}.", "Narrow context to {topic}.",
                    "Set context focus to {topic}.", "Pull up context around {topic}.",
                    "Concentrate on {topic} in context."]
    for t in focus_topics:
        for tpl in focus_instrs:
            out = f"Focusing context.\n{fmt_tool_call('context', 'focus', topic=t)}"
            samples.append({"instruction": tpl.format(topic=t), "output": out, "category": "tool_routing"})
    return samples


def gen_worldbuild(rng) -> list[dict]:
    samples = []
    camp_instrs = ["List the active campaigns.", "Show all campaigns.",
                   "What campaigns exist?", "Enumerate worldbuild campaigns.",
                   "Pull up the campaign list.", "Tell me which campaigns are running."]
    camp_acks = ["Listing campaigns.", "Fetching.",
                 "Querying worldbuild.", "Let me check."]
    for instr in camp_instrs:
        for ack in camp_acks:
            out = f"{ack}\n{fmt_tool_call('worldbuild', 'campaigns')}"
            samples.append({"instruction": instr, "output": out, "category": "tool_routing"})

    queries = ["dwarven mountain hold", "ancient prophecy", "Tharizdun",
               "the lost city of Sigil", "elven royal bloodlines",
               "draconic temples", "the Spell Plague", "Mordenkainen's lab",
               "the Underdark trade routes", "tiefling clans"]
    search_instrs = ["Search worldbuild for '{q}'.", "Look up '{q}' in the worldbuild.",
                     "Find '{q}' in the world.", "Query worldbuild: {q}",
                     "What does the world know about '{q}'?"]
    for q in queries:
        for tpl in search_instrs:
            out = f"Searching the world.\n{fmt_tool_call('worldbuild', 'search', query=q)}"
            samples.append({"instruction": tpl.format(q=q), "output": out, "category": "tool_routing"})

    # 'get' — use a small deterministic set of entity IDs
    eids = [str(100000 + i * 7919) for i in range(10)]
    get_instrs = ["Get the entity {id}.", "Fetch entity {id}.",
                  "Pull up entity {id}.", "Retrieve worldbuild entity {id}.",
                  "Show me entity {id}."]
    for eid in eids:
        for tpl in get_instrs:
            out = f"Fetching.\n{fmt_tool_call('worldbuild', 'get', entity_id=eid)}"
            samples.append({"instruction": tpl.format(id=eid), "output": out, "category": "tool_routing"})
    return samples


def gen_manage(rng) -> list[dict]:
    samples = []
    no_param = {
        "promote_list": (
            ["List candidates ready to promote.", "What's queued for promotion?",
             "Show pending promotions.", "Enumerate the candidate services.",
             "Which candidates can ship?", "Tell me what's ready to promote.",
             "Which candidate services have green tests?"],
            ["Listing.", "Fetching the promotion queue.",
             "Retrieving.", "Let me check."],
        ),
        "promote_status": (
            ["What's the promotion status?", "How's the promotion pipeline?",
             "Show the promotion state.", "Status of the candidate pipeline?",
             "Is anything blocked in promotion?"],
            ["Checking.", "Fetching status.",
             "Let me see.", "Querying promotion state."],
        ),
    }
    for action, (instrs, acks) in no_param.items():
        for instr in instrs:
            for ack in acks:
                out = f"{ack}\n{fmt_tool_call('manage', action)}"
                samples.append({"instruction": instr, "output": out, "category": "tool_routing"})

    services = ["gaia-core", "gaia-mcp", "gaia-orchestrator", "gaia-prime",
                "gaia-web", "gaia-study", "gaia-audio", "gaia-doctor",
                "gaia-monkey", "gaia-engine"]
    bp_instrs = ["Show the {svc} service blueprint.", "Get the blueprint for {svc}.",
                 "What's in the {svc} blueprint?", "Display {svc}'s blueprint.",
                 "Fetch the architecture blueprint for {svc}."]
    for svc in services:
        for tpl in bp_instrs:
            out = f"I'll fetch the blueprint.\n{fmt_tool_call('manage', 'blueprint', service=svc)}"
            samples.append({"instruction": tpl.format(svc=svc), "output": out, "category": "tool_routing"})
    return samples


# ── No-tool refusal samples — teach the model when NOT to call tools ────────


def gen_no_tool(rng) -> list[dict]:
    """Samples where the prompt sounds tool-ish but should be answered from memory.
    Teaches the model when NOT to call a tool. Each sample appears exactly once
    (no inflation) — variety comes from sample count, not duplicates."""
    samples = [
        ("What is 7 times 8?", "56."),
        ("What's the capital of France?", "Paris."),
        ("How do you say hello in Spanish?", "Hola."),
        ("What year did World War II end?", "1945."),
        ("What's the chemical symbol for water?", "H2O."),
        ("Define photosynthesis briefly.", "The process by which plants convert sunlight, water, and CO₂ into glucose and oxygen."),
        ("Who wrote Hamlet?", "William Shakespeare."),
        ("What's the speed of light?", "Approximately 299,792 kilometers per second in a vacuum."),
        ("Convert 100 Celsius to Fahrenheit.", "212°F."),
        ("What is your name?", "I'm GAIA, a sovereign AI agent built on Google's Gemma 4 E4B by Azrael."),
        ("What's 15 squared?", "225."),
        ("Capital of Japan?", "Tokyo."),
        ("Capital of Australia?", "Canberra."),
        ("Capital of Brazil?", "Brasília."),
        ("Capital of Canada?", "Ottawa."),
        ("Translate 'thank you' to French.", "Merci."),
        ("Translate 'goodbye' to German.", "Auf Wiedersehen."),
        ("How many continents are there?", "Seven."),
        ("How many planets are in our solar system?", "Eight."),
        ("What is the largest mammal?", "The blue whale."),
        ("What is the smallest country in the world?", "Vatican City."),
        ("Who painted the Mona Lisa?", "Leonardo da Vinci."),
        ("Who discovered penicillin?", "Alexander Fleming, in 1928."),
        ("What is the boiling point of water in Fahrenheit?", "212°F."),
        ("What is the freezing point of water in Celsius?", "0°C."),
        ("How many sides does a hexagon have?", "Six."),
        ("How many bytes in a kilobyte?", "1024 bytes (binary) or 1000 (decimal/SI)."),
        ("What's 144 divided by 12?", "12."),
        ("What's the square root of 81?", "9."),
        ("What language is spoken in Japan?", "Japanese (日本語)."),
        ("Who is the current President of the United States?", "I don't have current event data to verify that reliably — I can search the web if you need the latest answer."),
        ("What's a prime number?", "A natural number greater than 1 that has no positive divisors other than 1 and itself. Examples: 2, 3, 5, 7, 11."),
        ("Define gravity in one sentence.", "Gravity is the attractive force between objects with mass."),
        ("What is DNA?", "Deoxyribonucleic acid — the molecule carrying genetic instructions in all living organisms."),
    ]
    return [{"instruction": i, "output": o, "category": "tool_routing"} for i, o in samples]


def main() -> int:
    rng = random.Random(42)
    OUT_DIR.mkdir(parents=True, exist_ok=True)

    all_samples: list[dict] = []
    all_samples += gen_file(rng)
    all_samples += gen_web(rng)
    all_samples += gen_shell(rng)
    all_samples += gen_knowledge(rng)
    all_samples += gen_palace(rng)
    all_samples += gen_introspect(rng)
    all_samples += gen_browser(rng)
    all_samples += gen_audio_tool(rng)
    all_samples += gen_study(rng)
    all_samples += gen_notebook(rng)
    all_samples += gen_context(rng)
    all_samples += gen_worldbuild(rng)
    all_samples += gen_manage(rng)
    all_samples += gen_no_tool(rng)

    rng.shuffle(all_samples)

    out_path = OUT_DIR / "tool_calls.jsonl"
    with open(out_path, "w") as f:
        for s in all_samples:
            f.write(json.dumps(s, ensure_ascii=False) + "\n")

    print(f"Wrote {len(all_samples)} samples → {out_path}")
    # Summary by tool
    from collections import Counter
    tool_counts: Counter = Counter()
    for s in all_samples:
        out = s["output"]
        if "<tool_call>" in out:
            try:
                j_str = out.split("<tool_call>")[1].split("</tool_call>")[0]
                obj = json.loads(j_str)
                tool_counts[obj.get("tool", "?")] += 1
            except Exception:
                tool_counts["[unparsable]"] += 1
        else:
            tool_counts["[no-tool]"] += 1
    print("\nBy tool:")
    for t in sorted(tool_counts):
        print(f"  {t}: {tool_counts[t]}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
