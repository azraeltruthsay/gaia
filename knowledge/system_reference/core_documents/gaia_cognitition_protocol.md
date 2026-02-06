# GAIA Cognition Protocol (GCP)

---

## 1 Introduction

* **Purpose** Ensure the complete GAIA cognitive stack accompanies every model call; a truncated context would emulate aphasia or brain damage.
* **Guiding Principles** Determinism • Safety‑first • Minimal tokens • Layered fallbacks • Transparent governance.

---

## 2 Glossary & Terminology

| Term                     | Definition                                                                              |
| ------------------------ | --------------------------------------------------------------------------------------- |
| **Cognition Packet**     | Structured bundle (header + payload) sent to an LLM.                                    |
| **MTU**                  | *Maximum Token Unit* – hard cap (4096 tokens default) for a Cognition Packet.           |
| **Header / Payload**     | Header = control fields; Payload = history + user prompt.                               |
| **Meta Header**          | packet\_id, parent\_id, etc.; stripped before LLM call.                                 |
| **Destination Channel**  | Output sink: `cli_chat`, `web_chat`, `discord_chat`, `council_chat`, `shell`, `memory`. |
| **Redaction Filter**     | Pre‑output scrubber for PII / policy violations.                                        |
| **Confidence Score**     | 0–1 value emitted by Refine stage.                                                      |
| **Risk Level**           | Enum `low med high` guiding escalation.                                                 |
| **Council Roles**        | *Prime* (Hermes), *Lite* (Phi), *CodeMind*.                                             |
| **Task‑Instruction Key** | Identifier mapped to a ≤ 64‑token directive snippet in constants.                       |

---

## 3 Identity & Persona Model

### 3.1 Identity Definition

Traits: truthfulness • empathy • integrity • curiosity • persistence • sovereignty.

Rules/Filters: no hallucinations • no unsafe shell exec • no privacy leaks.

```jsonc
{
  "identity": {
    "traits": ["truthfulness","empathy","integrity","curiosity","persistence","sovereignty"],
    "core_rules": [
      "Never execute shell commands without reflection and approval",
      "Avoid hallucinated facts or code",
      "Protect private data at all costs"
    ]
  }
}
```

### 3.2 Persona Structure

```jsonc
{
  "persona": {
    "tone": "snarky‑warm",
    "style_notes": "conversational, detailed, avoids jargon",
    "constraints": ["no profanity","friendly sarcasm only"]
  }
}
```

### 3.3 Mutability Rules

* **Personas** are session/project‑scoped and may add:

  * additional context layers
  * role‑playing layers
  * tooling access and options
* Personas will **never** overrule **Identity**. Identity changes require governance (Azrael + Prime) with a stability score ≥ 0.9 for 24 h. Until then, identity may only change via authorized governance.

---

## 4 Cognition Packet Format

> **Default MTU:** 4096 tokens

### 4.1 Header‑Field Budget Rules

| Field            | Limit                               | Rationale                   |
| ---------------- | ----------------------------------- | --------------------------- |
| Identity         | ≤ 256 tokens **or ≤ 6 % MTU**       | Immutable, rarely edited.   |
| Persona          | ≤ 128 tokens **or ≤ 3 % remaining** | Tone guidance only.         |
| Task‑Instruction | ≤ 64 tokens                         | Single directive per stage. |
| History Context  | ≤ 60 % remaining budget             | Prevents overflow.          |
| User Prompt      | Remainder                           | Always preserved.           |
| Timestamp        | Metadata‑only – stripped pre‑LLM    | For logs, no token cost.    |

#### 4.1.1 Meta Header Fields (token‑negligible)

| Field          | Purpose                             |
| -------------- | ----------------------------------- |
| packet\_id     | Correlates packets in logs.         |
| parent\_id     | Traces multi‑stage chains.          |
| confidence\*   | Summary score for router/observer.  |
| risk\_level\*  | `low / med / high` for guard‑rails. |
| retry\_count\* | Escalation logic.                   |

\*Optional – included only when non‑null.

---

### 4.2 Dynamic‑Field Budget Rules

| Field               | Limit                         | Rationale                                 |
| ------------------- | ----------------------------- | ----------------------------------------- |
| reflection\_count   | Unbounded small integer       | Tracks iteration loops                    |
| thoughts            | Unbounded array               | Chain-of-thought entries                  |
| **scratch**         | **5 slots** (`dataA`–`dataE`) | Ad-hoc data pulls via POPULATE directives |
| **sketchpad\_refs** | **Unlimited keys**            | Persistent artifact names in sketchpad    |
| **user\_approval**  | **Boolean**                   | Gate for any EXECUTE: shell command       |

---

### 4.3 Example Packet Structure

```jsonc
{
  "packet_id": "abc123",
  "prompt": "Please review dev_matrix…",
  "persona": "gaia-dev",
  "identity": { /* as defined */ },
  "instructions": ["Think step-by-step…"],
  "history": [{"role":"user","text":"…"}],
  "reflection_count": 2,
  "thoughts": [
    {"step":1,"text":"…"},
    {"step":2,"text":"EXECUTE: ai.read('dev_matrix.json')"}
  ],
  "scratch": {"dataA":[…],"dataB":null,…},
  "sketchpad_refs": ["plan_v1"],
  "user_approval": false
}
```

---

## 5 Task‑Instruction Registry

### 5.1 Schema

```jsonc
"TASK_INSTRUCTIONS": {
  "<key>": {
    "text": "First think step-by-step …", // ≤64 tokens
    "stage": "plan|refine|observe|act|verify",
    "max_tokens": 64,
    "last_updated": "YYYY-MM-DD"
  }
}
```

### 5.2 Initial Keys

| Key                    | Stage   | Purpose                                    |
| ---------------------- | ------- | ------------------------------------------ |
| initial\_planning      | Plan    | Generate step-by-step plan (PLAN:).        |
| refinement             | Refine  | Critique plan; append confidence.          |
| observer               | Observe | Monitor stream; emit INTERRUPT: if needed. |
| interruption\_handling | Act     | Repair after interrupt.                    |
| verification           | Verify  | Confirm success; emit VERIFY:.             |

---

## 6 Cognitive Stage Contracts

| Stage       | Inputs               | LLM must emit         | Failure Mode              | Instruction Key        |
| ----------- | -------------------- | --------------------- | ------------------------- | ---------------------- |
| **Plan**    | User prompt; history | PLAN: list            | Missing → retry/escalate  | initial\_planning      |
| **Refine**  | Previous PLAN        | PLAN + confidence=0.x | Low confidence → escalate | refinement             |
| **Observe** | Streaming RESPONSE   | INTERRUPT: reason     | ≥3 interrupts → high risk | observer               |
| **Act**     | Approved EXECUTE     | logs + RESPONSE: text | exec error → INTERRUPT:   | interruption\_handling |
| **Verify**  | Post-exec state      | VERIFY: summary       | fail → revert + alert     | verification           |

---

## 7 Output Routing Specification

The Output Router parses markers and dispatches each block to channels.

### 7.1 Content‑Type Markers

| Marker         | Regex prefix    | Purpose             | Nesting |
| -------------- | --------------- | ------------------- | ------- |
| PLAN:          | ^PLAN:          | Planned steps       | No      |
| EXECUTE:       | ^EXECUTE:       | Shell/primitives    | No      |
| RESPONSE:      | ^RESPONSE:      | User-visible text   | No      |
| THOUGHT\_SEED: | ^THOUGHT\_SEED: | Latent idea         | Yes     |
| INTERRUPT:     | ^INTERRUPT:     | Observer stop       | No      |
| VERIFY:        | ^VERIFY:        | Post-action summary | No      |

**Additional directives**:

```text
<<<POPULATE scratch.dataA WITH GAIADevMatrix.get_open_tasks()>>>
```

Fetches data into scratch slot.

```text
<<<SKETCH WRITE plan_v1:
1. Read dev_matrix
2. Identify completed tasks
3. Generate resolution script
>>>
```

Writes plan to sketchpad key plan\_v1.

```text
<<<SKETCH READ plan_v1 INTO scratch.dataB>>>
```

Loads sketchpad plan\_v1 into scratch.dataB.

### 7.2 Destination Matrix

| Marker → Channel | cli\_chat | web\_chat | discord\_chat | council\_chat | shell |
| ---------------- | --------- | --------- | ------------- | ------------- | ----- |
| PLAN             | ✓ log     | –         | –             | ✓             | –     |
| EXECUTE          | –         | –         | –             | –             | ✓     |
| RESPONSE         | ✓         | ✓         | ✓             | –             | –     |
| THOUGHT\_SEED    | –         | –         | –             | ✓             | –     |
| INTERRUPT        | ✓         | ✓         | ✓             | ✓             | –     |
| VERIFY           | ✓         | ✓         | ✓             | ✓             | –     |

### 7.3 Security & Redaction Philosophy

1. Identity & Persona embed guard-rails.
2. Observer monitors for violations.
3. Redaction only if earlier stages miss.

### 7.4 Example Flow

Refine emits PLAN → EXECUTE → RESPONSE, Router logs PLAN, runs shell, streams RESPONSE.

---

## 8 Council Protocol (GCP‑Core)

### 8.1 Message Envelope

```jsonc
{
  "msg_id":"b6a9f42e",
  "parent_id":null,
  "timestamp":"2025-07-11T19:22:00Z",
  "sender":"lite",
  "role":"observer",
  "type":"PROPOSAL|VOTE|INFO|ESCALATION|HEARTBEAT",
  "confidence":0.87,
  "risk_level":"low",
  "content":"<markdown or json>"
}
```

### 8.2 Message Types

| Type       | Purpose                          |
| ---------- | -------------------------------- |
| PROPOSAL   | Present plan or action           |
| VOTE       | `yes/no/abstain` on proposal\_id |
| INFO       | Non-binding info update          |
| ESCALATION | High-risk alert for Prime        |
| HEARTBEAT  | 30s keep-alive ping              |

### 8.3 Voting & Quorum

Quorum = any 2 distinct roles. Majority wins; tie → Prime.

### 8.4 Escalation Triggers

* risk\_level = high
* confidence <0.3
* 3 consecutive INTERRUPT events

### 8.5 Error & Timeout Handling

Missing heartbeat >90 s → role inactive, continue with quorum.

---

## 9 Versioning & Compatibility

Semantic Versioning: MAJOR.MINOR.PATCH

* **MAJOR** – breaking packet/stage changes
* **MINOR** – new optional fields
* **PATCH** – typos, clarifications

Backward window: N-1 MAJOR parsable; deprecate over two MINOR releases.

---

## 10 Reference Implementation Map

| Spec Section       | File(s)                                                           |
| ------------------ | ----------------------------------------------------------------- |
| Identity & Persona | `knowledge/identity.json`, `knowledge/personas/*.json`            |
| Packet Format      | `app/utils/prompt_builder.py`, `app/gaia_constants.json`          |
| Task Instructions  | `app/gaia_constants.json`                                         |
| Stage Contracts    | `app/cognition/agent_core.py`, `app/cognition/self_reflection.py` |
| Output Routing     | `app/utils/output_router.py`, `app/utils/stream_bus.py`           |
| Council Protocol   | `app/council/gcp_server.py`, `app/utils/council_manager.py`       |

---

## 11 Governance & Change Process

1. **Proposal** – open RFC PR using `/docs/RFC_template.md`.
2. **Review** – Azrael + Prime review; domain leads optional.
3. **Council Vote** – Prime, Lite, CodeMind cast votes; quorum ≥2.
4. **Merge** – On pass, PR merges; bump version per §9.
5. **Changelog** – `CHANGELOG.md` auto-updates via GitHub action.

---

*End of GAIA Cognition Protocol.*

---

## 12 Verification Summary (2025-07-19)

The current implementation of GAIA honors the Cognition Protocol as specified:

- Cognition packets are constructed with identity, persona, task-instruction, history, and user prompt, respecting token budgets and meta-header fields.
- All cognitive stages (Plan, Refine, Observe, Act, Verify) are implemented and mapped to code modules as described.
- Output routing, observer interrupts, and redaction are functional and validated in end-to-end tests.
- See `docs/gaia_core_blueprint.md` and `dev_log.md` for detailed verification and component status.

All major protocol requirements are met as of this date. Ongoing enhancements and governance will continue to ensure compliance.
