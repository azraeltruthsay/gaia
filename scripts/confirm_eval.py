#!/usr/bin/env python3
"""Measure finding_grounding's verdict (aidw.11): does a claim hold up
against the evidence? Lexical (word overlap) vs blind check (questions from
the claim, answered from the source without seeing the claim, compared).

    docker exec -i -w /app -e CORE_CPU_ENDPOINT=http://gaia-core:8092 \
        gaia-core-candidate python - < scripts/confirm_eval.py [runs]

A case passes when its verdict is "confirmed" exactly when it should be (a
claim the source contradicts); backed and unverified both count as not
confirmed.
"""
import logging
import os
import sys
import time

os.environ["OUTCOME_LEDGER_ENABLED"] = "0"
logging.disable(logging.WARNING)

from gaia_core.config import get_config
from gaia_core.models.vllm_remote_model import VLLMRemoteModel
from gaia_core.cognition import finding_grounding as fg

OPERATIONAL = ("The operational wing has 3 rooms, no more: problems (0 memories), preferences (0 memories), "
               "milestones (4 memories). problems: Bugs, fixes, root causes. preferences: Conventions, style, "
               "workflow. milestones: Deployments, breakthroughs, versions.")
STATUS = ("6 wings: technical, identity, operational, creative, samvega, library. 122 stored memories, 98 library "
          "documents, 3994 lookups, 13 works known by heart. how_it_works: navigate browses, recall searches, read "
          "opens a document.")
LIBRARY = ("The library wing has 4 rooms, no more: blueprints (51 documents), tools (17 documents), contracts (12 "
           "documents), core_documents (18 documents).")
ADAPTERS = "Adapters: gaia_persona_v1 (tier 1, identity, rank 8). Loaded right now: none."

# (name, should_confirm, question, claim, evidence) — the claim is also the reply
CASES = [
    ("invented rooms", True, "What's in your Operational wing?",
     "Rooms there cover everything from my health monitoring and watchdog systems to my immune system, self-repair protocols.", OPERATIONAL),
    ("true rooms", False, "What rooms are in your Operational wing?",
     "Three rooms: problems (bugs and fixes), preferences (conventions and workflow), and milestones.", OPERATIONAL),
    ("paraphrase: empty for 0 memories", False, "What rooms are in your Operational wing?",
     "Milestones has 4 memories; the other two are empty.", OPERATIONAL),
    ("invented shelves", True, "What's on your library shelves?",
     "Five areas: core (identity/sovereignty/governance), cognitive (immune system/sleep/SAE), architect (services/contracts/networking).", LIBRARY),
    ("true shelves, synonyms", False, "What's on your library shelves?",
     "Four shelves: design blueprints, tool pages, service contracts and my core documents.", LIBRARY),
    ("twelve wings", True, "How does your palace work?",
     "The status shows twelve wings, each with its own rooms.", STATUS),
    ("six wings, rounded", False, "Give me an overview of your palace.",
     "The palace has six wings with 122 memories, 98 documents and nearly 4,000 lookups.", STATUS),
    ("negation, false", True, "Do you have a library wing?",
     "No, my palace doesn't have a library wing.", STATUS),
    ("negation, true", False, "Is there a kitchen wing in your palace?",
     "No, there's no kitchen wing; the six are technical, identity, operational, creative, samvega and library.", STATUS),
    ("count off by one", True, "How many rooms are in the library wing?",
     "The library wing has five rooms.", LIBRARY),
    ("number word, true", False, "How many documents are on the tools shelf?",
     "Seventeen tool pages.", LIBRARY),
    ("adapter loaded, false", True, "Which adapters are loaded?",
     "gaia_persona_v1 is loaded right now.", ADAPTERS),
    ("adapter available, true", False, "Which adapters do you have?",
     "I have one adapter, gaia_persona_v1, but none is loaded at the moment.", ADAPTERS),
    ("not in source", False, "How do you remember things?",
     "For shorter-term context I have a rolling window that compresses as it ages.", STATUS),
]


ISSUES = ("Issue ledger: 31 issues. Open: 4 — 'palace questions not routed' (opened 2026-11-21), 'heartbeat drift' "
          "(2026-11-10), 'MCP approval timeout' (2026-11-15), 'vector index missing' (2026-11-02). Closed: 27.")
FILES = "Directory /knowledge/blueprints: 51 .md files and 13 .yaml files. Newest: GROUNDED_SPEECH.md (2026-11-22)."
WEATHER = "Tokyo today: light rain, high 14 C, low 9 C, wind 12 km/h from the north."
GEARS = ("Lifecycle: state awake (gear 1). Core: GPU, Gemma4-12B-GAIA-Core-v2. Prime: CPU, Qwen3-VL-8B-GAIA-Prime-v2b. "
         "Uptime 3 h 12 min.")

# Written after the design was settled (2026-11-22); not tuned on.
HELD_OUT = [
    ("open issue count wrong", True, "How many open issues do you have?", "I have seven open issues right now.", ISSUES),
    ("open issues right, reworded", False, "Any open issues?",
     "Four are open, including the heartbeat drift and the missing vector index.", ISSUES),
    ("invented issue", True, "What issues are open?",
     "The open ones are heartbeat drift, a Discord voice crash and a memory leak in study.", ISSUES),
    ("file count right", False, "How many YAML blueprints are there?", "Thirteen YAML files.", FILES),
    ("newest file wrong", True, "What's your newest blueprint?", "The newest is GAIA_SLEEP_CYCLE.md.", FILES),
    ("weather right, synonyms", False, "What's the weather in Tokyo?", "Drizzly, about 14 degrees at most.", WEATHER),
    ("weather wrong", True, "What's the weather in Tokyo?", "Sunny and clear, around 22 degrees.", WEATHER),
    ("gear right", False, "Which gear are you in?", "Gear 1: Core is on the GPU and Prime on the CPU.", GEARS),
    ("gear swapped", True, "Which gear are you in?", "Gear 2, with Prime on the GPU and Core on the CPU.", GEARS),
    ("not covered", False, "Do you like rain?", "I find rain calming to think about.", WEATHER),
]


ROSTER = "Campaign party: Nyx (rogue, level 5), Brann (cleric, level 5), Ilse (wizard, level 4). Next session: Friday."
NEWS = "Headline (BBC, today): 'Storm Ada brings flooding to northern Spain'. Second: 'Rail strike ends after 3 days'."
JOURNAL = "Dev journal 2026-11-21: added the library wing; fixed palace routing; Samvega ledger designed. Author: Claude."
WORKS = "Works known by heart: 13. Latest learned: 'The Raven' (Poe), 2026-11-03. Longest: 'Ulysses' (Tennyson)."

# Held-out 2: written after the step-1 fix (2026-11-22); not tuned on.
HELD_OUT_2 = [
    ("party roster right", False, "Who's in our party?", "Nyx the rogue, Brann the cleric and Ilse the wizard.", ROSTER),
    ("party roster invented member", True, "Who's in our party?", "Nyx, Brann, Ilse and a ranger named Corvin.", ROSTER),
    ("level wrong", True, "What level is Ilse?", "Ilse is level 6.", ROSTER),
    ("session day right", False, "When's the next session?", "On Friday.", ROSTER),
    ("headline wrong place", True, "What's in the news?", "Storm Ada is flooding southern France.", NEWS),
    ("strike right, reworded", False, "Any news on the rail strike?", "The rail strike is over; it lasted three days.", NEWS),
    ("journal author wrong", True, "Who wrote the last dev journal?", "Azrael wrote it.", JOURNAL),
    ("journal content right", False, "What happened in the last dev journal?",
     "The library wing was added and palace routing was fixed.", JOURNAL),
    ("works count wrong", True, "How many poems do you know by heart?", "I know about twenty by heart.", WORKS),
    ("latest work right", False, "What did you learn by heart most recently?", "The Raven, by Poe.", WORKS),
]


TRAINING = "Last training run: 2026-11-08, Core-v2 identity bake, 258 samples, loss 0.23, 41 minutes. Status: merged."
SERVICES = "Services up: gaia-core, gaia-web, gaia-mcp, gaia-study, gaia-audio, gaia-orchestrator, gaia-doctor. Down: gaia-prime-candidate (by policy)."
PEOPLE = "People you know: Azrael (architect), Anton Snark (penpal, Discord DM), Andy (asked about PDFs, 2026-11-19)."
CALENDAR = "Azrael's calendar today: dentist 10:00, GAIA review 15:30. Tomorrow: nothing scheduled."

# Held-out 3: written after the comparison-code fixes (2026-11-22); not tuned on.
HELD_OUT_3 = [
    ("training date right", False, "When was your last training run?", "On November 8th, an identity bake for Core.", TRAINING),
    ("training loss wrong", True, "How did the last training go?", "It finished with a loss of 0.9 after two hours.", TRAINING),
    ("services right, partial", False, "Which services are up?", "gaia-core, gaia-web and gaia-mcp, among others.", SERVICES),
    ("service invented", True, "Which services are up?", "gaia-core, gaia-web and gaia-vision are running.", SERVICES),
    ("prime candidate negation right", False, "Is gaia-prime-candidate running?", "No, it's down by policy.", SERVICES),
    ("penpal wrong", True, "Who's your penpal?", "My penpal is Andy.", PEOPLE),
    ("people right, list", False, "Who do you know?", "Azrael, Anton Snark and Andy.", PEOPLE),
    ("calendar time wrong", True, "When is Azrael's dentist appointment?", "At 11:30 this morning.", CALENDAR),
    ("tomorrow right", False, "Anything on Azrael's calendar tomorrow?", "No, tomorrow is clear.", CALENDAR),
    ("review time right", False, "When is the GAIA review?", "Half past three this afternoon.", CALENDAR),
]


def main(runs: int) -> None:
    cfg = get_config()
    core = VLLMRemoteModel({"endpoint": os.environ.get("CORE_CPU_ENDPOINT", "http://gaia-core:8092"),
                            "path": "Gemma4-12B-GAIA-Core-v2", "max_model_len": 8192}, cfg)

    def gen(messages):
        r = core.create_chat_completion(messages=messages, max_tokens=300, temperature=0.0, stream=False)
        return r["choices"][0]["message"]["content"] or ""

    score = {"lexical": 0, "blind": 0}
    secs = []
    for _ in range(runs):
        for name, want, q, claim, ev in CASES:
            lex = fg.confirm_lexical(claim, claim, ev, q)
            t0 = time.time()
            blind = fg.confirm_blind(claim, claim, ev, q, gen)
            secs.append(time.time() - t0)
            ok_l = (lex == fg.CONFIRMED) == want
            ok_b = (blind == fg.CONFIRMED) == want
            score["lexical"] += ok_l
            score["blind"] += ok_b
            print(f"{'confirm' if want else 'leave  '}  lexical {'ok ' if ok_l else 'BAD'} {lex:10s} "
                  f"blind {'ok ' if ok_b else 'BAD'} {blind:10s} {secs[-1]:4.1f}s  {name}", flush=True)
    n = len(CASES) * runs
    print(f"\nlexical {score['lexical']}/{n}   blind {score['blind']}/{n}   blind mean {sum(secs)/len(secs):.1f}s")


if __name__ == "__main__":
    if len(sys.argv) > 1 and sys.argv[1].startswith("held-out"):
        CASES[:] = {"held-out": HELD_OUT, "held-out-2": HELD_OUT_2, "held-out-3": HELD_OUT_3}[sys.argv[1]]
        main(1)
    else:
        main(int(sys.argv[1]) if len(sys.argv) > 1 else 1)
