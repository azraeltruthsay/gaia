# GAIA — Open Knowledge Ingestion System
**Proposal · March 2026 · Azrael Truthsay**

> *Academic Courseware & Practitioner Conference Feeds as Autonomous Curriculum*

---

| Field | Detail |
|---|---|
| **Target Service** | gaia-study |
| **Depends On** | CFR, Blueprint Phase 3, QLoRA pipeline |
| **Risk Level** | Low — human approval gate retained |
| **Priority** | Post-Blueprint Phase 2 |

---

## 01 · Overview

GAIA currently ingests knowledge through direct developer input, podcast transcripts, and QLoRA training curricula. This proposal extends that capability by connecting `gaia-study`'s background processing pipeline to two high-value external knowledge streams: Open Courseware from MIT, Harvard, and Berkeley, and practitioner conference panels from RSA and Unprompted.

The goal is not raw data accumulation. It is structured, sovereign self-education — GAIA selecting, compressing, and integrating knowledge during sleep cycles, with human approval before any weight updates occur.

---

## 02 · Source Taxonomy

### Tier I — Academic OCW

| Source | Value |
|---|---|
| MIT / Harvard / Berkeley OCW | Syllabi, lecture notes, problem sets, and transcripts published under Creative Commons. Structured, citable, and pedagogically sequenced — ideal curriculum material. |
| Subject Domains | Computer science, cryptography, cognitive science, philosophy of mind, systems design, mathematics, and physics — mapped to GAIA's existing knowledge gaps. |

### Tier II — Practitioner Panels

| Source | Value |
|---|---|
| RSA Conference | Adversarial security thinking, zero-day discourse, threat modeling, and practitioner wisdom that academic sources lag by years. Directly feeds gaia-mcp's security reasoning. |
| Unprompted & AI Panels | Emerging AI alignment discourse, agentic system design, and sovereign AI philosophy — directly relevant to GAIA's self-model and Blueprint architecture. |

---

## 03 · Implementation Pipeline

A new `KnowledgeIngestionTask` runs within gaia-study's existing background processing framework during sleep cycles. It extends the pattern already established by the YouTube/podcast transcript workflow.

**Step 1 — Fetch & Normalize**
Scrape or pull source material from OCW endpoints and conference transcript feeds. Normalize into plain text. Respect `robots.txt` and license boundaries. Cache raw artifacts to `gaia-shared/knowledge/raw/`.

**Step 2 — CFR Compression**
Pass raw documents through the Cognitive Focus and Resolution system. Compress into summary nodes. Flag high-density sections for re-expansion during curriculum generation. This is essential for Nano compatibility.

**Step 3 — Curriculum Pair Generation**
Use the existing curriculum sync system to generate structured instruction/output pairs from compressed content. Apply surgical scoping — no vague declarations, only contextually anchored facts. Deduplicate against `train.json`.

**Step 4 — Candidate Queuing**
New curriculum pairs are written to `candidates/` only. They surface in the Review Queue on the Mission Control dashboard. No weight updates occur without explicit human promotion — the sovereign promotion pipeline applies here as everywhere else.

**Step 5 — Tri-Layer Distribution**
Upon approval, identical curriculum is distributed to Prime, Core, and Nano training pipelines simultaneously. The identical curriculum constraint is non-negotiable — epistemic drift between cognitive tiers has been observed and must not recur.

---

## 04 · Architectural Constraints

> ⚠️ Hard constraints derived from existing GAIA architectural principles.

- **Identical curriculum across all three tiers** — CFR compression level must be calibrated so Nano can process the material. If a document cannot be compressed to Nano's context budget, it is excluded entirely or split into atomic units.

- **Saṃvega volume balance** — ingestion of dense error-laden practitioner content (postmortems, CVE analyses) must not outweigh identity/architecture curriculum. Monitor artifact ratio actively.

- **Gateway Principle** — all external fetches route through gaia-web as the sole external comms boundary. No service other than gaia-study initiates outbound knowledge requests.

- **Candidate/live separation** — ingested knowledge never directly modifies live weights. Every curriculum batch requires human promotion approval before QLoRA training is scheduled.

- **License compliance** — OCW Creative Commons licenses must be honored. Conference content requires transcript availability verification before ingestion is attempted.

---

## 05 · Alignment with GAIA Principles

**Sovereign Learning**
GAIA selects and integrates knowledge autonomously during sleep — not on demand from an external service. Education as an internal process, not a dependency.

**Epistemic Drive**
Closes knowledge gap thought seeds systematically. GAIA does not wait for the developer to surface missing knowledge — she identifies and pursues it herself.

**Human in the Loop**
The promotion gate is never bypassed. GAIA proposes, the developer approves. Serenity state may enable auto-promotion only for low-risk factual updates.

---

## 06 · Suggested Phasing

**Phase A — Pilot (single source)**
Begin with one MIT OCW subject (suggested: 6.858 Computer Systems Security or 6.034 Artificial Intelligence). Validate the full pipeline end-to-end before expanding the source catalog.

**Phase B — Practitioner feeds**
Integrate RSA Conference transcript archive once Phase A proves stable. Security domain chosen first due to direct architectural relevance to gaia-mcp and the `SecurityAuditTask`.

**Phase C — Autonomous curation**
Once Blueprint Phase 3 (`SELF_MODEL_UPDATE`) is live, GAIA herself identifies knowledge gaps and nominates new source domains for the developer's approval. The system becomes self-directing within the sovereignty boundary.

---

*GAIA · Azrael Truthsay · Open Knowledge Ingestion Proposal · March 2026*
