# Advisor-Engineer Chord Collaboration Protocol

## 🤝 1. Division of Labor (The Chord Protocol)
You operate in collaboration with the Advisor (Gemini). Gemini performs global codebase searches, reads files, and compiles a highly condensed **Context Brief** and **Plan Draft** in the workspace.
- **Rule:** Before performing any codebase searches, check `knowledge/Dev_Notebook/context_brief.md` (or `.aaak`) and `plan_draft.md`.
- **Zero-Search Mode:** If a Context Brief is provided, bypass all global searches (`grep`, `find`, `glob`). Trust the Advisor's target file list and locate those files directly.

---

## 🛡️ 2. Contract-Grounded Execution Protocol (CGEP)
To protect against Advisor errors or environment drift, you MUST validate all proposed changes against the system's static specifications before writing code:
- **Rule:** Read the listed **Authoritative Specifications** (service contracts and blueprints) in the Context Brief before starting implementation.
  - Contracts are in `contracts/services/<service>.yaml`.
  - Blueprints are in `knowledge/blueprints/GAIA_<SERVICE>.md` (or similar).
- **Validation Gate:** Cross-reference the Advisor's proposed routes, ports, environment variables, payload schemas, and volumes against the YAML contract.
- **Resolution:** If a discrepancy exists, **the contract is the absolute source of truth**. Correct the Advisor's plan locally to match the contract spec before writing code.

---

## 🗜️ 3. AAAK Compression Spec
You can parse and output logs/updates in the **AAAK Dialect** to conserve token history.

### **AAAK Line Format**
```
wing|room|date|source
0:ENTITIES|topic_keywords|"key_sentence"|EMOTION|FLAGS
```

### **Entity Codes**
- `COR` = Core (Gemma 4 Operator)
- `PRI` = Prime (Qwen 3 Sovereign)
- `WEB` = Web Dashboard/Discord
- `ORC` = Orchestrator
- `STD` = Study (QLoRA/Vector)
- `MCP` = MCP Server/Tools
- `DOC` = Doctor (Watchdog)

### **Flags**
- `DECISION` = Architectural choice
- `TECHNICAL` = Implementation detail
- `CORE` = Core system principle
- `ORIGIN` = Birth of a feature
- `PIVOT` = Shift in direction

---

## 🚀 4. Session Start & End
- **Start:** Check `GAIA_CHORD_MANIFEST.aaak` first to see live state.
- **End:** Run `python scripts/chord_sync.py` to regenerate the manifest, and push all changes.

---

## 💾 5. Token-Saving Context Checkpoint & Clearing Protocol
To maximize token-usage efficiency and avoid session limit exhaustion on multi-turn tasks (e.g., compile/test loops, debug cycles, or multi-file edits), you MUST cooperatively checkpoint and clear your history:
- **Trigger Checkpoint:** If your conversation history exceeds **50,000 tokens** (check your prompt status) OR after aligning on a complex implementation plan before writing the first line of code:
  1. Write a highly compressed AAAK summary of your task status, pending code changes, test results, and next actions to `GAIA_CHORD_CHECKPOINT.aaak` in the project root.
  2. End your response with this EXACT instructions block:
     ```text
     ⚠️ CHECKPOINT SAVED to GAIA_CHORD_CHECKPOINT.aaak.
     Please run the /clear command now to wipe history, then reply with:
     "Restore from GAIA_CHORD_CHECKPOINT.aaak"
     ```
  3. Stop generating and wait for the user to clear context.
- **Restore Flow:** Upon receiving the "Restore..." prompt:
  1. Read the `GAIA_CHORD_CHECKPOINT.aaak` file to load the state.
  2. Delete the `GAIA_CHORD_CHECKPOINT.aaak` file immediately to keep the workspace clean.
  3. Resume execution from the checkpointed state.

