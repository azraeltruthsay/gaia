# 🧠 GAIA Codebase Audit & Refactor Overview

This document serves as a comprehensive summary of all guiding principles, technical standards, and structural conventions established during the major GAIA refactor. It acts as both a developer reference and an internal review rubric for GAIA herself.

---

## ✅ 1. Tiered Knowledge & Memory Architecture

- **Tier I** → Immutable Core Identity (e.g. `core_identity.json`)
- **Tier 0** → System Reference Documents (`gaia_constitution`, `layered_identity_model`)
- **Tier 1–3** → Structured archives, vector embeddings, LoRA prep, conversations

All systems must:
- Use `config.get_path_for_tier()` or similar for proper pathing
- Maintain tier metadata for each asset
- Avoid cross-tier contamination (e.g., persona code can’t overwrite Tier I identity)

---

## 👤 2. Persona + Instruction Hierarchy

- **Personas** = Core behavioral mode (`cheerful_mentor`, `code_analyzer`, etc.)
- **Instructions** = Additive task-specific modifiers (`frustrated_mode`, `use_c++`, etc.)

### Design Rules:
- One active persona per session
- Instructions are optional overlays, possibly stackable
- Persona files stored in `/personas/` with corresponding instructions subdirectory
- GAIA must reference Tier I identity even when personality shifts occur

---

## 🔐 3. Identity Enforcement & Ethical Safety

- `core_identity_guardian.py` + `ethical_sentinel.py` = validate all responses
- No prompt, instruction, or personality may override Tier I
- Self-correction behaviors enforced via `SelfReflection`

GAIA uses these for:
- Pre-response prompt injection checks
- Ongoing idle-time validation of long-term drift

---

## 🧩 4. Modular, Robust, Auditable Code Structure

- Each module performs **one** responsibility
- Extensive use of:
  - `logger`
  - try/except guards
  - type annotations
  - docstrings for all public functions

Key Directories:
- `/behavior/`: session + persona logic
- `/conversation/`: message logging, summarization, keywording
- `/code_analyzer/`: chunking, summaries, change tracking
- `/background/`: idle workflows, queue processor

---

## ⚙️ 5. Boot Resilience + Semantic Reprocessing

- `CodeAnalyzer` avoids redundant work via `SnapshotManager`
- Only processes changed files
- Structured summaries saved per file in `system_reference/code_summaries/`
- Conversation summaries stored in `structured/conversations/{persona}/`

---

## 🧠 6. Config-Driven Everything

- `config.py` is the source of truth for:
  - Tier paths
  - Codebase root
  - Vector store locations
  - Instruction + persona directories

All subsystems now read from this central config to ensure:
- Docker volume compatibility
- Portability to Windows/Mac/Linux environments

---

## 🌙 7. Background Cognition + Idle Triggers

- `BackgroundProcessor` coordinates tasks
- Idle detection from `IdleMonitor`
- Reflection, document embedding, code review queued via `TaskQueue`
- Includes:
  - `summarize_conversation`
  - `embed_documents`
  - `generate_artifacts`
  - `summarize_codebase`

---

## 📜 8. Documentation and Traceability (WIP)

- Auto-generated `functions_reference.md` planned
- Persona registry + instruction doc generation pending
- GAIA should eventually:
  - Understand her own boot process
  - Log reasons for all behavioral decisions
  - Flag potential ethical violations and ambiguities

---

> This document is living and should be referenced at every code review or module audit to ensure continuity with GAIA’s evolving self-awareness and integrity goals.
