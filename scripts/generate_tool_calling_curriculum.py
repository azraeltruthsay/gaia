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

# Common parameter value pools
PARAM_VALUES = {
    "file_path_text": [
        "/shared/notes.md", "/shared/CLAUDE.md", "/shared/knowledge/index.md",
        "/gaia/GAIA_Project/README.md", "/shared/logs/today.log",
        "/shared/sessions.json", "/shared/training_runs/latest/summary.json",
        "/shared/doctor/cognitive_test_results.json",
        "/shared/kvcache/core/identity_prefix.pt",
    ],
    "file_path_code": [
        "/gaia/GAIA_Project/gaia-core/gaia_core/main.py",
        "/gaia/GAIA_Project/scripts/train_core_multimodal.py",
        "/gaia/GAIA_Project/gaia-mcp/gaia_mcp/server.py",
        "/gaia/GAIA_Project/gaia-engine/gaia_engine/core.py",
    ],
    "dir_path": [
        "/shared", "/shared/logs", "/gaia/GAIA_Project/scripts",
        "/shared/training_runs", "/shared/sessions.archived",
        "/gaia/GAIA_Project/gaia-core", "/gaia/GAIA_Project/contracts",
    ],
    "search_pattern": [
        "*.py", "*.md", "*.json", "*.jsonl", "test_*.py", "*v6*", "*v7*",
    ],
    "web_query": [
        "latest Gemma 4 release notes", "PyTorch 2.5 changelog",
        "Docker compose v3 reference", "LoRA training best practices",
        "RTX 5080 thermal specs", "transformers tokenizer special tokens",
        "Python 3.11 release date", "JSON schema specification",
        "MCP protocol specification", "vLLM inference optimization",
    ],
    "web_url": [
        "https://github.com/anthropics/anthropic-cookbook",
        "https://huggingface.co/google/gemma-4-E4B",
        "https://docs.python.org/3.11/",
        "https://pytorch.org/docs/stable/index.html",
    ],
    "knowledge_query": [
        "what is the GPU lifecycle gearbox",
        "how does the consciousness matrix work",
        "what are GAIA's tier models",
        "what is the Sovereign Duality architecture",
        "what tools does gaia-mcp expose",
        "how does Memento-style skill learning work",
    ],
    "memory_query": [
        "recent training runs",
        "yesterday's cognitive battery results",
        "Azrael's recent design discussions",
        "the last vision capability test",
        "what bugs are open in beads",
    ],
    "shell_cmd_safe": [
        "ls -la /shared/", "df -h /gaia", "free -h",
        "docker ps --format 'table {{.Names}}\\t{{.Status}}'",
        "nvidia-smi --query-gpu=memory.used,memory.free --format=csv",
        "uptime", "git log --oneline -10", "git status",
    ],
    "session_id": [
        "session-2026-05-13-001", "training-validation-v7",
        "azrael-conversation-morning",
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
    # file.read
    read_instrs = [
        ("Read the file at {path}.", "I'll read that file."),
        ("Show me the contents of {path}.", "Let me read it."),
        ("Open {path} and tell me what's in it.", "Reading the file."),
        ("What's in {path}?", "I'll check."),
        ("Display {path}.", "Reading."),
        ("Pull up the contents of {path}.", "I'll read it for you."),
        ("Cat {path} for me.", "Let me read that file."),
    ]
    for _ in range(60):
        instr_tpl, ack = rng.choice(read_instrs)
        path = sample(rng, rng.choice(["file_path_text", "file_path_code"]))
        instr = instr_tpl.format(path=path)
        out = f"{ack}\n{fmt_tool_call('file', 'read', path=path)}"
        samples.append({"instruction": instr, "output": out, "category": "tool_routing"})

    # file.list
    list_instrs = [
        ("What files are in {path}?", "I'll list the directory contents."),
        ("List the contents of {path}.", "Listing files."),
        ("Show me what's in {path}.", "Let me check the directory."),
        ("Ls {path}.", "I'll list it."),
        ("What's in the directory {path}?", "Let me see."),
    ]
    for _ in range(50):
        instr_tpl, ack = rng.choice(list_instrs)
        path = sample(rng, "dir_path")
        instr = instr_tpl.format(path=path)
        out = f"{ack}\n{fmt_tool_call('file', 'list', path=path)}"
        samples.append({"instruction": instr, "output": out, "category": "tool_routing"})

    # file.tree
    tree_instrs = [
        ("Show me the tree under {path}.", "I'll get the recursive tree view."),
        ("Recursively list {path}.", "Getting the full tree."),
        ("Tree view of {path}.", "I'll fetch the tree."),
    ]
    for _ in range(30):
        instr_tpl, ack = rng.choice(tree_instrs)
        path = sample(rng, "dir_path")
        instr = instr_tpl.format(path=path)
        out = f"{ack}\n{fmt_tool_call('file', 'tree', path=path)}"
        samples.append({"instruction": instr, "output": out, "category": "tool_routing"})

    # file.find
    find_instrs = [
        ("Find all {pattern} files under {path}.", "I'll search for matching files."),
        ("Locate {pattern} files in {path}.", "Searching."),
        ("Search for {pattern} in {path}.", "I'll look for them."),
    ]
    for _ in range(50):
        instr_tpl, ack = rng.choice(find_instrs)
        pattern = sample(rng, "search_pattern")
        path = sample(rng, "dir_path")
        instr = instr_tpl.format(pattern=pattern, path=path)
        out = f"{ack}\n{fmt_tool_call('file', 'find', path=path, pattern=pattern)}"
        samples.append({"instruction": instr, "output": out, "category": "tool_routing"})

    # file.write
    write_instrs = [
        ("Write 'Hello' to {path}.", "I'll write that. (Note: requires approval.)"),
        ("Save the text 'X' to {path}.", "Writing the file. (Sensitive op — approval required.)"),
    ]
    for _ in range(30):
        instr_tpl, ack = rng.choice(write_instrs)
        path = sample(rng, "file_path_text")
        instr = instr_tpl.format(path=path)
        content = rng.choice(["Hello world", "test content", "GAIA was here"])
        out = f"{ack}\n{fmt_tool_call('file', 'write', path=path, content=content)}"
        samples.append({"instruction": instr, "output": out, "category": "tool_routing"})

    return samples


def gen_web(rng) -> list[dict]:
    samples = []
    # web.search
    search_instrs = [
        ("Search the web for {q}.", "I'll search the web."),
        ("Look up {q} online.", "Searching."),
        ("What does the web say about {q}?", "Let me search."),
        ("Find information about {q}.", "I'll look it up."),
        ("Google {q} for me.", "I'll search the web."),
    ]
    for _ in range(150):
        instr_tpl, ack = rng.choice(search_instrs)
        q = sample(rng, "web_query")
        instr = instr_tpl.format(q=q)
        out = f"{ack}\n{fmt_tool_call('web', 'search', query=q, max_results=5)}"
        samples.append({"instruction": instr, "output": out, "category": "tool_routing"})

    # web.fetch
    fetch_instrs = [
        ("Fetch the page at {url}.", "I'll fetch that page."),
        ("Get the contents of {url}.", "Fetching."),
        ("Pull up {url} for me.", "I'll grab the page."),
    ]
    for _ in range(100):
        instr_tpl, ack = rng.choice(fetch_instrs)
        url = sample(rng, "web_url")
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
    ]
    for _ in range(200):
        instr_tpl, ack = rng.choice(shell_instrs)
        cmd = sample(rng, "shell_cmd_safe")
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

    # knowledge.status / stats / list / kg_*
    for action, tpl_instr, tpl_ack in [
        ("status", "What's the knowledge base status?", "I'll check the status."),
        ("rebuild", "Rebuild the knowledge index.", "I'll rebuild it."),
        ("list", "List all knowledge bases.", "Listing them."),
        ("kg_query", "Query the knowledge graph for entity 'GAIA'.", "I'll query the knowledge graph."),
        ("kg_stats", "Show knowledge graph statistics.", "I'll fetch the stats."),
        ("kg_timeline", "Show the recent knowledge timeline.", "I'll fetch the timeline."),
    ]:
        for _ in range(20):
            kwargs = {}
            if action == "kg_query":
                kwargs["entity"] = rng.choice(["GAIA", "Azrael", "Gemma 4"])
            out = f"{tpl_ack}\n{fmt_tool_call('knowledge', action, **kwargs)}"
            samples.append({"instruction": tpl_instr, "output": out, "category": "tool_routing"})

    return samples


def gen_palace(rng) -> list[dict]:
    samples = []
    for action, instrs in [
        ("store", [("Store this in the memory palace: {x}", "I'll store it.")]),
        ("recall", [("Recall from memory palace: {x}", "Recalling.")]),
        ("navigate", [("Navigate the memory palace to {x}", "Navigating.")]),
        ("status", [("What's the memory palace status?", "I'll check.")]),
    ]:
        for _ in range(25):
            tpl_instr, tpl_ack = rng.choice(instrs)
            kwargs = {}
            x = rng.choice(["yesterday's training discussion", "the v7 release decision",
                            "Azrael's preferences"])
            if action in ("store", "recall", "navigate"):
                kwargs["topic"] = x
                instr = tpl_instr.format(x=x)
            else:
                instr = tpl_instr
            out = f"{tpl_ack}\n{fmt_tool_call('palace', action, **kwargs)}"
            samples.append({"instruction": instr, "output": out, "category": "tool_routing"})
    return samples


def gen_introspect(rng) -> list[dict]:
    samples = []
    for action, instr, ack in [
        ("world", "What's the current world state?", "I'll check the world state."),
        ("recall", "Recall recent events.", "Looking up recent events."),
        ("logs", "Show recent system logs.", "Fetching logs."),
        ("tools", "List all available tools.", "I'll list them."),
        ("describe", "Describe the {tool_name} tool.", "I'll fetch the description."),
        ("count_chars", "Count characters in this text.", "I'll count."),
    ]:
        for _ in range(20):
            kwargs = {}
            if action == "describe":
                t = rng.choice(["file", "web", "knowledge", "palace"])
                kwargs["target"] = t
                cur_instr = instr.format(tool_name=t)
            else:
                cur_instr = instr
            out = f"{ack}\n{fmt_tool_call('introspect', action, **kwargs)}"
            samples.append({"instruction": cur_instr, "output": out, "category": "tool_routing"})
    return samples


def gen_browser(rng) -> list[dict]:
    samples = []
    for action, instr_tpl, ack in [
        ("browse", "Browse to {url}", "I'll open that page."),
        ("snapshot", "Take a snapshot of the current page.", "Snapshotting."),
        ("links", "List the links on the current page.", "I'll list them."),
        ("forms", "What forms are on this page?", "Checking for forms."),
        ("screenshot", "Take a screenshot of {url}", "Taking a screenshot."),
    ]:
        for _ in range(25):
            kwargs = {}
            if "{url}" in instr_tpl:
                url = sample(rng, "web_url")
                kwargs["url"] = url
                cur_instr = instr_tpl.format(url=url)
            else:
                cur_instr = instr_tpl
            out = f"{ack}\n{fmt_tool_call('browser', action, **kwargs)}"
            samples.append({"instruction": cur_instr, "output": out, "category": "tool_routing"})
    return samples


def gen_audio_tool(rng) -> list[dict]:
    """Audio TOOL (not audio modality) — listening control + inbox management."""
    samples = []
    for action, instr, ack in [
        ("listen_start", "Start listening for voice input.", "I'll start the audio listener."),
        ("listen_stop", "Stop listening.", "Stopping the listener."),
        ("listen_status", "Is the audio listener running?", "I'll check the status."),
        ("inbox_status", "What's in the audio inbox?", "Checking the inbox."),
        ("inbox_list", "List unprocessed audio messages.", "Listing inbox."),
    ]:
        for _ in range(25):
            out = f"{ack}\n{fmt_tool_call('audio', action)}"
            samples.append({"instruction": instr, "output": out, "category": "tool_routing"})
    return samples


def gen_study(rng) -> list[dict]:
    samples = []
    for action, instr, ack in [
        ("status", "What's the training status?", "I'll check training status."),
        ("adapter_list", "List available LoRA adapters.", "Listing adapters."),
        ("adapter_info", "Tell me about the {a} adapter.", "I'll fetch info."),
        ("adapter_load", "Load the {a} adapter.", "Loading the adapter."),
    ]:
        for _ in range(25):
            kwargs = {}
            if "{a}" in instr:
                a = rng.choice(["core_deliberation_v1", "gaia_architecture", "code_skill_v3"])
                kwargs["adapter_name"] = a
                cur_instr = instr.format(a=a)
            else:
                cur_instr = instr
            out = f"{ack}\n{fmt_tool_call('study', action, **kwargs)}"
            samples.append({"instruction": cur_instr, "output": out, "category": "tool_routing"})
    return samples


def gen_notebook(rng) -> list[dict]:
    samples = []
    for action, instr, ack in [
        ("list", "List my notebooks.", "Listing notebooks."),
        ("notes", "Show notes from notebook {nb}.", "I'll fetch notes."),
        ("create_note", "Create a note titled '{t}' with content '{c}'.", "I'll create the note."),
    ]:
        for _ in range(25):
            kwargs = {}
            if "{nb}" in instr:
                nb = rng.choice(["research-2026", "gaia-architecture", "training-log"])
                kwargs["notebook_id"] = nb
                cur_instr = instr.format(nb=nb)
            elif "{t}" in instr:
                t = rng.choice(["Today's V7 results", "Vision battery follow-up"])
                c = rng.choice(["Battery scored 65.9%", "Vision still hallucinates"])
                kwargs["title"] = t
                kwargs["content"] = c
                cur_instr = instr.format(t=t, c=c)
            else:
                cur_instr = instr
            out = f"{ack}\n{fmt_tool_call('notebook', action, **kwargs)}"
            samples.append({"instruction": cur_instr, "output": out, "category": "tool_routing"})
    return samples


def gen_context(rng) -> list[dict]:
    samples = []
    for action, instr, ack in [
        ("focus", "Focus context on {topic}.", "Focusing context."),
        ("compress", "Compress the current context.", "I'll compress it."),
        ("status", "What's the context status?", "I'll check."),
        ("rolling", "Show the rolling context.", "Fetching."),
        ("fragment_list", "List pending context fragments.", "Listing fragments."),
    ]:
        for _ in range(20):
            kwargs = {}
            if "{topic}" in instr:
                t = rng.choice(["training history", "user preferences", "current bug"])
                kwargs["topic"] = t
                cur_instr = instr.format(topic=t)
            else:
                cur_instr = instr
            out = f"{ack}\n{fmt_tool_call('context', action, **kwargs)}"
            samples.append({"instruction": cur_instr, "output": out, "category": "tool_routing"})
    return samples


def gen_worldbuild(rng) -> list[dict]:
    samples = []
    for action, instr, ack in [
        ("campaigns", "List the active campaigns.", "Listing campaigns."),
        ("search", "Search worldbuild for '{q}'.", "Searching the world."),
        ("get", "Get the entity {id}.", "Fetching."),
    ]:
        for _ in range(20):
            kwargs = {}
            if "{q}" in instr:
                q = rng.choice(["dwarven mountain hold", "ancient prophecy", "Tharizdun"])
                kwargs["query"] = q
                cur_instr = instr.format(q=q)
            elif "{id}" in instr:
                eid = str(rng.randint(100000, 999999))
                kwargs["entity_id"] = eid
                cur_instr = instr.format(id=eid)
            else:
                cur_instr = instr
            out = f"{ack}\n{fmt_tool_call('worldbuild', action, **kwargs)}"
            samples.append({"instruction": cur_instr, "output": out, "category": "tool_routing"})
    return samples


def gen_manage(rng) -> list[dict]:
    samples = []
    for action, instr, ack in [
        ("blueprint", "Show the {svc} service blueprint.", "I'll fetch the blueprint."),
        ("promote_list", "List candidates ready to promote.", "Listing."),
        ("promote_status", "What's the promotion status?", "Checking."),
    ]:
        for _ in range(20):
            kwargs = {}
            if "{svc}" in instr:
                svc = rng.choice(["gaia-core", "gaia-mcp", "gaia-orchestrator"])
                kwargs["service"] = svc
                cur_instr = instr.format(svc=svc)
            else:
                cur_instr = instr
            out = f"{ack}\n{fmt_tool_call('manage', action, **kwargs)}"
            samples.append({"instruction": cur_instr, "output": out, "category": "tool_routing"})
    return samples


# ── No-tool refusal samples — teach the model when NOT to call tools ────────


def gen_no_tool(rng) -> list[dict]:
    """Samples where the prompt sounds tool-ish but should be answered from memory."""
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
    ]
    out = []
    for instr, ans in samples:
        for _ in range(10):  # 10 copies each → 100 total
            out.append({"instruction": instr, "output": ans, "category": "tool_routing"})
    return out


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
