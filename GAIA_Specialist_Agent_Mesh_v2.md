  
**GAIA Development Toolchain**

Specialist Agent Mesh

with Agent-Local Knowledge Directories & Pattern Libraries

Plan Proposal v2  |  February 2026

**Status:**  Draft Proposal — Supersedes v1

**Scope:**  External development toolchain only — zero changes to GAIA runtime architecture

**Nature:**  Bootstrap-phase scaffolding, retired as GAIA achieves autonomous capability

**Authors:**  Azrael & Claude (Sonnet 4.6) via GAIA Project Chat

# **What's New in v2**

Version 1 of this proposal established the Specialist Agent Mesh — a set of purpose-built Claude Code subagents replacing generic reviewer invocations, each operating in an isolated session with a narrow, domain-specific context. v2 builds on that foundation with two significant extensions.

The first extension is Agent-Local Knowledge Directories: a formalised file-system structure inside each agent's directory that houses curated reference documents, known-pattern lists, and architectural excerpts. Rather than the agent deriving context from scratch or the orchestrator injecting it ad hoc, each agent carries its own permanent knowledge base that travels with it.

The second — and more architecturally interesting — extension is Pattern Libraries for creative and compositional agents. When an agent's job involves designing or constructing something (a UI page, a service scaffold, a blueprint YAML), it should not re-invent from first principles on every invocation. A local pattern library of canonical templates, component primitives, and design tokens gives the agent a stable foundation to compose from, ensuring consistency, reducing tokens, and encoding the project's aesthetic and structural preferences as durable artifacts rather than ephemeral prompts.

| Core Insight Templates shift the agent's cognitive task from synthesis to composition. Synthesis is expensive, inconsistent, and prone to drift. Composition from known primitives is cheap, consistent, and drift-resistant. The pattern library is the difference between asking an agent to invent GAIA's design language on every call and asking it to faithfully apply a design language it can read directly from disk. |
| :---- |

# **1\. Executive Summary**

This proposal defines the complete architecture for purpose-built Claude Code specialist agents in the GAIA development toolchain, incorporating agent-local knowledge directories and pattern libraries as first-class components of each agent's structure.

Each agent is a self-contained unit: its identity and instructions live in a CLAUDE.md, its curated reference knowledge lives in a context/ subdirectory, and — for agents that design or construct artifacts — its canonical templates and component primitives live in a templates/ subdirectory. The agent loads exactly what it needs and nothing more.

The result is a toolchain that is efficient by construction: low token cost, high consistency, and a natural feedback loop where good solutions are promoted from one-off outputs into the shared pattern library, progressively improving every future invocation.

As with v1, this is entirely bootstrap-phase scaffolding. It will be retired service by service as GAIA develops the internal capabilities to replace each external agent.

# **2\. The Two-Layer Knowledge Model**

Every agent directory houses two distinct layers of local knowledge, serving different purposes and evolving at different rates.

## **2.1  Layer 1: Contextual References (context/)**

Reference documents are curated, human-authored extracts of architectural truth. They do not change often — they reflect stable facts about the system: how services communicate, what security invariants must hold, what the CognitionPacket schema looks like. They exist to give the agent reliable ground truth without requiring it to read the entire codebase.

Critically, these are extracts, not copies. A Sentinel agent's security\_patterns.md is not a dump of the codebase — it is a distilled list of the 20 patterns that matter for security review. The compression ratio is high, and the signal-to-noise ratio is maximised. An agent loading 200 lines of dense, curated reference gets more useful context than one loading 10,000 lines of source.

## **2.2  Layer 2: Pattern Libraries (templates/)**

Pattern libraries serve a fundamentally different purpose. Where reference documents answer the question 'what is true about this system', pattern libraries answer the question 'what does a correct artifact look like in this domain'. They are starting points, not descriptions.

A template is canonical. When the UX Designer agent builds a new console page, it begins from base.html — not from its own inference about what GAIA's pages look like. The template encodes decisions that have already been made: the grid system, the color tokens, the component structure, the motion curve. The agent's creative energy goes into solving the new problem, not re-solving already-solved ones.

Pattern libraries grow organically through a promotion pipeline. A solution designed for one task, if sufficiently general and well-reviewed, is extracted and added to the library. Over time the library becomes a living design system, grown from real solutions rather than designed speculatively up front.

| Candidate / Live — Applied to Design The same promotion philosophy that governs GAIA's code artifacts applies to design patterns. A new component begins as a one-off solution. If it proves robust and reusable, it is promoted to templates/components/. From that point, every agent invocation inherits it. Bad patterns are never promoted. The library only improves. |
| :---- |

# **3\. Full Directory Architecture**

## **3.1  Root Structure**

| .claude/   agents/     codemind/     sentinel/     alignment/     blueprint/     study/     ux-designer/       \<- pattern library agent     service-scaffold/  \<- pattern library agent   shared/     architectural-overview.md   \<- loaded by ALL agents     cognition-packet-v03.md     \<- CognitionPacket spec     container-topology.md       \<- service map   slash-commands/     codemind-review.sh     sentinel-review.sh     ux-design.sh     ... |
| :---- |

The shared/ directory holds facts that every agent needs regardless of domain. Rather than duplicating them into every agent's context/, the CLAUDE.md for each agent references shared/ explicitly. If the CognitionPacket spec changes, it is updated once and all agents immediately have the correct version.

## **3.2  Review Agent Structure (CodeMind as canonical example)**

| .claude/agents/codemind/   CLAUDE.md   context/     coding-idioms.md         \<- GAIA naming conventions, patterns, anti-patterns     blueprint-schema.md      \<- BlueprintModel Pydantic schema excerpt     known-drift-patterns.md  \<- historical divergence patterns to watch for     interface-contracts.md   \<- service boundary definitions   examples/     good-review.json         \<- a high-quality review result for calibration     bad-review.json          \<- a poor review showing what to avoid |
| :---- |

## **3.3  Security Agent Structure (Sentinel)**

| .claude/agents/sentinel/   CLAUDE.md   context/     security-patterns.md     \<- distilled from Feb 2026 audit: H1/H2 findings     mcp-threat-model.md      \<- MCP sandbox boundary rules, capability constraints     injection-history.md     \<- known injection vectors found in GAIA codebase     container-boundaries.md  \<- what each service is/is not permitted to do |
| :---- |

## **3.4  UX Designer Agent Structure (Pattern Library Agent)**

| .claude/agents/ux-designer/   CLAUDE.md   context/     design-principles.md     \<- GAIA console aesthetic philosophy     accessibility-rules.md   \<- contrast, keyboard nav, ARIA requirements   templates/     base.html                \<- standard page scaffold (doctype, head, shell)     layout.css               \<- grid system, spacing scale, breakpoints     components/       card.html       sidebar.html       status-badge.html       data-table.html       modal.html       toast.html     patterns/       color-tokens.css       \<- \--gaia-primary, \--gaia-accent, \--gaia-surface...       typography.css         \<- heading scale, font stack, line-height system       motion.css             \<- transition timing, easing curves, durations       icons.md               \<- icon library reference and usage rules   examples/     blueprint-panel/         \<- complete finished page: HTML \+ CSS \+ JS     system-state/            \<- complete finished page as second reference |
| :---- |

## **3.5  Service Scaffold Agent Structure (Pattern Library Agent)**

| .claude/agents/service-scaffold/   CLAUDE.md   context/     service-conventions.md   \<- naming, logging, error handling standards     docker-patterns.md       \<- standard Dockerfile and compose patterns   templates/     service/       main.py                \<- standard service entrypoint       config.py              \<- config loading pattern       health.py              \<- health check endpoint       Dockerfile       docker-compose.yml     \<- service compose fragment     common-patterns/       packet-handler.py      \<- CognitionPacket receive/respond pattern       vector-client.py       \<- read-only VectorClient usage pattern       sleep-hook.py          \<- sleep cycle integration pattern   examples/     gaia-study-scaffold/     \<- a real service used as reference implementation |
| :---- |

# **4\. The CLAUDE.md Contract**

The CLAUDE.md is the agent's constitution. It must do three things clearly: establish the agent's identity and cognitive mode, declare the authority of its local files, and specify the output contract. The authority declaration is the part most often underspecified — and the most important.

## **4.1  Authority Declaration (Pattern Library Agents)**

For agents that have a templates/ directory, the CLAUDE.md must explicitly state that templates are canonical and deviations must be declared. A suggested formulation:

| Template Authority Clause (CLAUDE.md excerpt) Templates in templates/ are canonical. Do not introduce CSS properties, color values, layout structures, or component patterns not present in these files unless the task explicitly requires it. New components must be composed from existing primitives where possible. When deviation from a template is necessary, state the deviation and its justification in the summary field of your AgentReviewResult. Deviations are candidates for promotion — not license for drift. |
| :---- |

This framing matters: deviations are candidates, not failures. It creates a healthy relationship between the agent and the pattern library — the agent can innovate, but it must account for that innovation in a way that makes it reviewable and promotable.

## **4.2  Shared Context Loading**

Each CLAUDE.md should explicitly reference the shared/ directory in addition to its own context/:

| \# Always load on invocation: @../../shared/architectural-overview.md @../../shared/cognition-packet-v03.md \# Load from local context: @context/coding-idioms.md @context/blueprint-schema.md \# Load templates (UX Designer only): @templates/base.html @templates/patterns/color-tokens.css @templates/patterns/typography.css |
| :---- |

# **5\. The Promotion Pipeline**

The pattern library is not static. It grows through a lightweight promotion process that mirrors the candidate/live pipeline already governing GAIA's code artifacts.

## **5.1  For Design Patterns**

* Agent produces a new component or layout solution as part of a task output.

* If the solution is sufficiently general (usable beyond its immediate context), it is flagged in the AgentReviewResult summary as a promotion candidate.

* Human reviews the candidate. If approved, it is added to templates/components/ or templates/patterns/.

* It becomes available to every subsequent agent invocation immediately.

## **5.2  For Context References**

* New architectural truth is discovered during development (new service contract, new security invariant, new coding convention).

* It is added to the relevant agent's context/ file — or to shared/ if it applies broadly.

* Context file updates are treated as part of the same PR process that updates the relevant code, keeping context and reality in sync.

## **5.3  Drift Detection (Future — gaia-study)**

During sleep cycles, gaia-study already has access to the codebase and produces reflective notes. A natural extension is to have it flag when agent context files appear to have drifted from the current implementation — for example, when a documented interface contract no longer matches the actual code. This closes the loop between the living codebase and the static reference files, without requiring manual auditing.

| Phase 2 Hook gaia-study generating context-drift alerts is a natural Phase 2 capability. It does not need to be built now — the directory structure and promotion process work without it. But the architecture accommodates it without any structural change. |
| :---- |

# **6\. Complete Agent Roster**

| Agent | Type | context/ | templates/ | Primary Output |
| :---- | :---- | :---- | :---- | :---- |
| **CodeMind** | **Review** | Idioms, blueprint schema, drift patterns, contracts | — | Structural findings, divergence score |
| **Sentinel** | **Review** | Security patterns, MCP threat model, injection history | — | Security findings, severity classification |
| **AlignmentAgent** | **Review** | Service contracts, CognitionPacket spec, v0.3 API | — | Alignment score, semantic gap findings |
| **BlueprintAgent** | **Review** | BlueprintModel schema, candidate/live pipeline rules | — | Validated/corrected blueprint YAML |
| **StudyAgent** | **Review** | QLoRA curriculum structure, corpus metadata | — | Training suitability verdict |
| **UX Designer** | **Creative** | Design principles, accessibility rules | base.html, layout.css, components/, color-tokens.css | Composed page HTML/CSS/JS \+ deviation notes |
| **Service Scaffold** | **Creative** | Service conventions, Docker patterns | main.py, Dockerfile, packet-handler.py, sleep-hook.py | New service scaffold \+ CLAUDE.md |

# **7\. AgentReviewResult Schema**

The output schema is unchanged from v1 and applies to all agents — both review and creative. For creative agents, the findings list captures template deviations rather than code issues, and the summary field carries the deviation justifications required by the CLAUDE.md template authority clause.

| Field | Type | Notes |
| :---- | :---- | :---- |
| agent\_id | str | Identifies the specialist: sentinel, codemind, ux-designer, etc. |
| schema\_version | str | For forward compatibility |
| target | str | File, service, or page under review or construction |
| timestamp | datetime | ISO 8601 — audit trail and corpus ordering |
| verdict | Literal | approve | approve\_with\_notes | reject |
| findings | list | Typed findings — code issues for review agents, template deviations for creative agents |
| metrics | dict | Agent-specific: divergence\_score, vuln\_count, template\_deviation\_count, etc. |
| summary | str | Written last, derived from findings. Always consistent with structured data. Human-readable. |

# **8\. Implementation Plan**

## **Phase 1 — Foundation & Review Agents (Week 1\)**

* Create .claude/ directory structure: agents/, shared/, slash-commands/.

* Populate shared/ with architectural-overview.md, cognition-packet-v03.md, and container-topology.md — extracts, not raw source.

* Define AgentReviewResult and Finding Pydantic models in gaia-common/gaia\_common/models/review.py.

* Port codemind.json persona into a full CodeMind CLAUDE.md with context/ populated from the Feb 2026 audit journal and existing architectural docs.

* Author Sentinel CLAUDE.md and context/ from the audit's H1/H2 security findings.

* Extend prepare\_review.sh to route artifact bundles per-agent.

## **Phase 2 — Remaining Review Agents (Week 2\)**

* Author AlignmentAgent, BlueprintAgent, and StudyAgent CLAUDE.md files with their context/ directories.

* Write parse\_review\_result.py: validates AgentReviewResult JSON, routes by verdict (reject, approve\_with\_notes, approve).

* Codify the precedence rule: any reject verdict blocks promotion regardless of other agent verdicts.

## **Phase 3 — UX Designer Agent & Pattern Library Seed (Week 3\)**

* Author UX Designer CLAUDE.md with template authority clause.

* Seed templates/ by extracting canonical patterns from the two most complete existing console pages — these become the initial examples/.

* Extract color tokens, typography scale, and motion curves from existing CSS into patterns/ files — the design language becomes explicit for the first time.

* Author component templates from existing UI elements: card, sidebar, status-badge, data-table.

* Establish the promotion review process: document the criteria for a solution to graduate from one-off to templates/components/.

## **Phase 4 — Service Scaffold Agent & Validation (Week 4\)**

* Author Service Scaffold CLAUDE.md with templates/ seeded from existing services.

* Extract canonical service patterns: entrypoint structure, config loading, health check, CognitionPacket handler, sleep hook.

* Run all agents against a retroactive corpus of existing live services to validate context quality and finding accuracy.

* Compare agent findings against the Feb 2026 manually-identified issues for precision/recall calibration.

* Document retirement criteria for each agent.

# **9\. Token Economy**

v2 adds a new dimension to the token savings from v1. The agent-local knowledge model reduces context not just by isolating sessions but by pre-compressing domain knowledge into curated extracts. Pattern libraries add a further reduction for creative agents: a well-structured template carries more design intent per token than any amount of prose instruction.

| Source of Saving | Mechanism | Relative Impact |
| :---- | :---- | :---- |
| Session isolation | Review agents start cold with no generation-session baggage | **High** |
| Curated context extracts | 200 lines of dense reference vs 10,000 lines of source | **High** |
| Shared/ deduplication | Common facts loaded once, not duplicated across agent directories | **Medium** |
| Template composition | Agent composes from primitives rather than synthesising from scratch | **High (creative agents)** |
| Elimination of re-derivation | Design language never re-inferred — read directly from token files | **High (creative agents)** |
| Fewer regeneration cycles | Higher first-pass quality means fewer reject-and-retry loops | **Compounding** |

| Sustainability Fewer tokens is less compute, which is less energy. This is not incidental — it is a design goal. An agent that reads 200 lines of curated context and one canonical template to produce a consistent, correct output is doing more with less in the most literal sense. The Specialist Agent Mesh is efficient by construction, and that efficiency compounds over hundreds of development cycles. |
| :---- |

# **10\. Symmetry with GAIA Architecture**

The agent-local knowledge model is structurally isomorphic to GAIA's own architectural principles. This is not coincidence — it is the same set of design values applied at a different level of the system.

| GAIA Principle | GAIA Implementation | Toolchain Mirror |
| :---- | :---- | :---- |
| **Separation of concerns** | gaia-core (brain) vs gaia-web (interface) | Review agents vs creative agents — different cognitive modes, different directory structures |
| **Gateway Principle** | gaia-web as sole external communications boundary | Each agent's CLAUDE.md as sole authority declaration for its domain |
| **Semantic Codex** | Only relevant context loaded per cognitive turn | Curated context/ extracts — not raw source, not everything |
| **Candidate/live pipeline** | Human-approved promotion from candidates/ to live graph | Template promotion pipeline — solutions earn their way into the pattern library |
| **Self-model via blueprint** | GAIA describes herself in blueprint.yaml | Agents describe their knowledge domain in CLAUDE.md and context/ |
| **Sleep-cycle reflection** | gaia-study writes prime.md notes during idle time | gaia-study flags context drift (Phase 2\) — pattern library stays current automatically |

# **11\. Open Questions**

* How granular should color-tokens.css be at seed time? Err toward fewer, well-named tokens initially — the library should reflect decisions made, not aspirational completeness.

* Should the UX Designer agent have write access to templates/ (to propose new components directly), or should that always be a human-mediated promotion step? Recommend human-mediated for now — maintains the oversight model.

* Template versioning: should templates/ be versioned in git separately from the codebase? They are part of the toolchain configuration surface and should be versioned, but may warrant a separate commit convention to distinguish template evolution from code changes.

* How does the Service Scaffold agent handle services that deviate significantly from the standard scaffold (e.g. gaia-prime with its vLLM specifics)? Context/ should include a note on legitimate scaffold exceptions.

# **12\. Retirement Criteria**

Each agent is retired when the equivalent GAIA-internal capability demonstrates sufficient fidelity. The pattern library agents have a slightly different retirement path than review agents — their templates/ content doesn't disappear when they are retired, it migrates into GAIA's own design system knowledge.

* Review agents (CodeMind, Sentinel, AlignmentAgent, BlueprintAgent, StudyAgent): retired when gaia-prime achieves comparable finding quality on the retroactive validation corpus, as assessed by gaia-study.

* UX Designer: retired when GAIA can compose new console pages from her own internal design system representation, maintaining the same consistency guarantees the templates/ currently provide. The templates/ content migrates into gaia-study's knowledge base.

* Service Scaffold: retired when the Builder Panel generates scaffolds from blueprint seeds with sufficient fidelity that external scaffolding adds no value. The templates/ content migrates into the Builder Panel's generation context.

Retirement is the success condition, not a failure. The agents exist to build the capability that replaces them, and the artifacts they produce — reviewed code, promoted templates, training data — are the means by which that replacement happens.

# **13\. Summary**

The Specialist Agent Mesh v2 is a coherent, layered system. At its foundation are isolated specialist agents, each carrying only the context relevant to its domain. On top of that, for agents that design or construct, sits a local pattern library that encodes the project's established solutions as canonical starting points. Above that, a lightweight promotion pipeline ensures the library evolves from real solutions rather than speculation. And threading through all of it, the same candidate/live promotion philosophy and separation-of-concerns principles that govern GAIA herself.

The result is a development toolchain that is efficient, consistent, self-improving, and architecturally coherent with the system it is building. The scaffolding reflects the building. That kind of coherence is rare, and worth preserving deliberately as the system grows.

End of Proposal  —  GAIA Development Toolchain  —  Specialist Agent Mesh v2