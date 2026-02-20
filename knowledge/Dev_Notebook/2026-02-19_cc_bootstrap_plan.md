# GAIA Code Generation Bootstrap Plan
## CC-Assisted Code Quality, Self-Review, and QLoRA Training Loop

**Document status:** Implementation specification (Revision 7)
**Intended audience:** Claude Code (CC) for autonomous implementation
**Date:** 2026-02-19
**Author:** Seumas & Claude (Sonnet 4.6) via GAIA Project Chat
**Revised by:** Claude (Opus 4.6) via Claude Code — Rev 2-7 (see revision log)

---

## 1. Background & Motivation

### The Problem

GAIA's primary inference model (gaia-prime, currently Qwen3-4B-Instruct, 8K context window) is a capable general reasoner but has no inherent knowledge of the GAIA codebase, service contracts, or architectural idioms. When used to generate code — especially through the planned Builder Panel — it will produce output that is syntactically reasonable but semantically adrift from the project's conventions.

We have already observed this problem in miniature: prime's JSON output required a formatter extension to become reliable. Code is orders of magnitude more complex than JSON. The question is not whether prime will drift, but how we detect it, correct it, and close the loop so it drifts less over time.

### The Solution Architecture

This plan implements a three-layer approach:

1. **Mechanical quality gates** — ruff, mypy, AST validation, blueprint divergence scoring. Fast, deterministic, runs on every candidate.
2. **CC as bootstrap reviewer** — Claude Code acts as an independent expert reviewer during the early phase, producing structured critiques that become training data.
3. **QLoRA self-training loop** — GAIA studies her own reviewed code, converging toward CC-level fidelity on her own architecture without requiring CC indefinitely.

The key insight driving the design: we are not training GAIA to be a great Python programmer in general. We are training her to faithfully implement *her own blueprints* in *her own idioms*. That is a narrow and tractable target.

---

## 2. What CC Needs to Know: The GAIA Codebase

Before implementing anything, CC must understand several existing systems this plan builds on.

### 2.1 The Blueprint System

Every GAIA service is described by a **BlueprintModel** (Pydantic v2 schema in `gaia-common/gaia_common/models/blueprint.py`). Blueprints are YAML files in `knowledge/blueprints/` with two states:

- **CANDIDATE** (`knowledge/blueprints/candidates/{service}.yaml`) — prescriptive or unvalidated
- **LIVE** (`knowledge/blueprints/{service}.yaml`) — validated, rendered in the graph

Key blueprint sections relevant to code review:
- `intent` — purpose, cognitive_role, design_decisions, open_questions
- `interfaces` — exposed endpoints (typed via discriminated union: http_rest, websocket, sse, event, direct_call, mcp, grpc)
- `dependencies` — declared service APIs and volumes this service may call
- `failure_modes` — documented failure conditions and expected responses
- `source_files` — canonical list of source files with roles
- `meta.confidence` — per-section confidence scores (runtime/contract/dependencies/failure_modes/intent)

**Currently live blueprints (6):** gaia-core, gaia-web, gaia-mcp, gaia-orchestrator, gaia-prime, gaia-study.

Blueprint YAML is the **source of truth for design intent**. The code review task is always: "does this code faithfully implement what the blueprint claims?"

### 2.2 The Promotion Pipeline

`scripts/promote_pipeline.sh` orchestrates candidate-to-live promotion through 9 stages. Relevant stages for this plan:

- **Stage 3 (Validation):** ruff lint, mypy type check, pytest unit tests
- **Stage 4 (Smoke Tests):** 16-test cognitive battery against candidate service
- **Stage 6 (Journal):** Auto-generates `{DATE}_promotion_journal.md`

The promotion journal is the primary source of structured ground-truth data for training corpus construction. Every successful promotion is a (blueprint, code, validation outcome) tuple.

### 2.3 Existing Blueprint Validation (Sleep Task)

`candidates/gaia-core/gaia_core/cognition/sleep_task_scheduler.py` already implements mechanical blueprint-vs-code validation during sleep cycles. It uses regex/AST extraction to check that enum members, endpoints, and constants declared in blueprints are present in source files. This is the foundation we extend — it catches *structural* staleness but not *semantic* divergence.

The validation logic is currently embedded in the `_run_blueprint_validation` handler method. This plan requires factoring it out into a reusable function (Section 3.5) so the review prompt builder can invoke it independently to produce mechanical pre-check annotations for the LLM reviewer.

### 2.4 Existing Code Analyzer Infrastructure

`gaia-common/gaia_common/utils/code_analyzer/` contains a comprehensive analysis pipeline that the AST summarizer should build on rather than duplicate:

- **`structure_extractor.py`** — AST-based extraction of function and class definitions with line numbers
- **`docstring_extractor.py`** — Module, function, and class docstring extraction
- **`chunk_creator.py`** — Code chunking for vector indexing
- **`base_analyzer.py` (CodeAnalyzer)** — Full pipeline orchestration

**What already works:** AST parsing, function/class name extraction, docstring extraction, line-number tracking.
**What's missing for our needs:** Full parameter signatures with type annotations, return type annotations, decorator extraction (especially `@router.*`), enum member extraction, module-level constant extraction, and gaia-package import filtering.

The AST summarizer (Section 3.1) should extend `structure_extractor` rather than creating a parallel module.

### 2.5 QLoRA Training Infrastructure

`gaia-study` has a complete QLoRA training pipeline (PEFT + bitsandbytes, 4-bit quantization). Adapters are trained via `POST /study/start`, validated via `validate_adapter.py`, and hot-loaded into gaia-prime via vLLM's `--enable-lora`. The `json-architect` adapter is the first example — it was trained on synthetic tool-selection pairs to improve prime's JSON output quality.

The new adapter this plan creates is called **`code-architect`**.

### 2.6 The Independent Review Principle

CC should not review code it generated in the same context window. The reviewer must start cold from the artifact — blueprint + generated code + no memory of the generation session. This mirrors the "developer cannot approve their own PR" principle and ensures the critique is genuinely adversarial.

---

## 3. Phase 0: Infrastructure Prerequisites

These components must exist before the generation/review loop can run. Most are small and can be built quickly. None require architectural changes.

### 3.1 AST Summarizer Utility

**Location:** `gaia-common/gaia_common/utils/ast_summarizer.py`

**Implementation note:** This module should import and extend `structure_extractor` from the existing `code_analyzer/` package rather than reimplementing AST parsing from scratch. The existing extractor handles function/class discovery; the summarizer adds the richer extraction layer on top.

**Purpose:** Reduce a Python source file to a compact, structured summary suitable for LLM context. Raw source files are 300-800 lines; summaries should be 30-80 lines. This is the primary mechanism for keeping review prompts within context budget.

**What the summary must include:**
- Module-level docstring (first 200 chars)
- All class names with their base classes and class-level docstring (first 100 chars)
- All method/function signatures: name, parameters with type annotations, return type annotation
- All `@router.{method}` decorators with their path strings (endpoint declarations)
- All `Enum` subclass members with their values
- All module-level `UPPER_CASE` constants with their values (string/int/bool only; truncate long strings)
- All `import` statements from gaia packages (i.e., `from gaia_common...`, `from gaia_core...`, etc.)
- **Targeted body extractions** (see Section 3.1.1)

**What the summary must NOT include:**
- Full function bodies
- Inline comments
- Local variables
- Any line that is purely implementation detail (beyond the targeted extractions)

**Output format:** A structured dict serializable to JSON, with keys: `module_docstring`, `classes`, `functions`, `endpoints`, `enums`, `constants`, `gaia_imports`, `error_handlers`, `http_calls`. Also provide a `to_prompt_text()` method that renders this as a human-readable block suitable for inclusion in an LLM prompt.

**Testing:** `gaia-common/tests/utils/test_ast_summarizer.py` — verify against `gaia_core/cognition/run_turn.py` (a known complex file) and assert summary is < 15% of original line count.

#### 3.1.1 Targeted Body Extraction

Pure AST signature-level summaries have a blind spot: failure mode coverage (dimension 3) and dependency correctness (dimension 2) live *inside* function bodies — a `try/except` block wrapping an `httpx.get()` call won't appear in signatures alone.

To address this without including full function bodies, the summarizer must also extract:

- **Exception handlers:** For each `try/except` block, record the exception type(s) caught and any HTTP status code returned (e.g., `"handles: httpx.TimeoutException → 504"`)
- **HTTP client calls:** For each call matching `httpx.*`, `requests.*`, or `self.client.*`, record the method, target URL/path pattern, and enclosing function name (e.g., `"POST /mcp/invoke in call_mcp_tool()"`)
- **gaia-common client utility calls:** For each call to known client utilities (e.g., `ServiceClient.call()`, any function from `gaia_common.clients`), record the target service and method

These are added as `error_handlers` and `http_calls` keys in the summary dict. They provide heuristic evidence for the LLM reviewer on dimensions 2 and 3.

**Known limitations of targeted extraction alone:** These heuristics will miss conditional guards that aren't `try/except` blocks, retry/circuit-breaker semantics, implicit dependencies via environment variables, failure handling delegated to utility functions in other files, middleware-level handlers, and handlers defined in base classes. The false-negative rate for failure mode detection via targeted extraction alone is estimated at ~30-40%.

This is why targeted extraction is complemented by mechanical pre-check annotations (Section 3.5). The pre-checks run the existing sleep-task blueprint validator against full source code and provide deterministic structural completeness results. Together, the AST summary (including targeted extractions) gives the LLM *what* the code does, while the pre-check gives it *what's structurally present or missing*. The LLM's role shifts from "find what's missing" to "assess whether what's present actually implements the blueprint's intent."

### 3.2 Blueprint-Anchored Review Prompt Builder

**Location:** `gaia-common/gaia_common/utils/review_prompt_builder.py`

**Purpose:** Given a blueprint YAML, one or more AST summaries, and mechanical pre-check results, construct a structured review prompt that asks an LLM to identify specific, blueprint-anchored discrepancies. The prompt combines three information sources: the blueprint (what should exist), the AST summaries (what the code looks like), and the pre-check results (what's structurally present or missing).

**Prompt structure:**

```
SYSTEM:
You are a code reviewer for the GAIA AI system. Your task is to verify that
the provided source code faithfully implements its blueprint specification.

You are NOT evaluating general code quality. You are evaluating blueprint
fidelity across five dimensions:

1. CONTRACT FIDELITY — The mechanical pre-check below shows which endpoints
   are structurally present or missing. For [FOUND] endpoints, verify from
   the AST summaries that the implementation signature (parameters, return
   type) matches the blueprint's schema. For [MISSING] endpoints, confirm
   they are genuinely absent or flag if implemented under a different path.

2. DEPENDENCY CORRECTNESS — The pre-check confirms which declared dependencies
   appear in imports. Your task: verify from the AST summaries that dependency
   calls use correct paths/methods, and flag any UNDECLARED external calls the
   pre-check may have missed. Note: gaia-common imports are universally
   available and should NOT be flagged as undeclared dependencies.

3. FAILURE MODE COVERAGE — The pre-check shows which failure modes have
   matching handlers. For [FOUND] handlers, assess from the AST summary
   whether the handling logic matches the blueprint's documented response
   (not just that a handler exists). For [MISSING] handlers, confirm absence
   or flag if handled via a non-standard pattern.

4. INTENT COHERENCE — Does the code's overall structure reflect the blueprint's
   stated purpose and cognitive_role? This dimension is NOT covered by
   mechanical pre-checks — it requires your semantic judgment. Flag obvious
   divergences.

5. OPEN QUESTIONS — Does the code reveal answers to any open_questions in the
   blueprint? Or does it raise new ones? Also NOT covered by pre-checks.

Respond ONLY with a structured JSON object matching the ReviewResult schema below.

USER:
## Blueprint: {service_id}

### Intent
{bp.intent.purpose}
Cognitive role: {bp.intent.cognitive_role}

### Interfaces
{formatted interfaces table}

### Dependencies
{formatted dependency list}

### Failure Modes
{formatted failure modes table}

### Open Questions
{formatted open questions list}

### Confidence Scores
{confidence table}

---

## Mechanical Pre-Check Results

{precheck_result.to_prompt_text()}

---

## Source Files Under Review

{for each file: filename header + ast_summary.to_prompt_text()}

---

## Review Task

The mechanical pre-check above shows structural completeness — what is present
or missing at a syntactic level. Your task is to assess SEMANTIC fidelity:

- For items the pre-check marked [FOUND]: does the implementation actually
  fulfill the blueprint's intent, or is it a superficial match?
- For items the pre-check marked [MISSING]: is this genuinely absent, or
  implemented in a way the pre-check couldn't detect?
- For dimensions 4-5 (intent coherence, open questions): apply your own
  judgment — these have no mechanical coverage.

Be specific: cite the blueprint claim and the contradicting (or absent)
code evidence.
```

**ReviewResult schema** (Pydantic model, also in this module):

```python
class DiscrepancyItem(BaseModel):
    dimension: Literal["contract", "dependencies", "failure_modes", "intent", "open_questions"]
    severity: Literal["critical", "major", "minor", "observation"]
    blueprint_claim: str          # exact text from blueprint
    code_evidence: str            # what the AST summary shows (or "not found")
    recommendation: str           # specific fix
    affected_file: Optional[str]  # which source file

class OpenQuestionUpdate(BaseModel):
    question: str                 # existing or new open question text
    status: Literal["answered", "new", "escalated"]
    evidence: str

class ReviewResult(BaseModel):
    service_id: str
    reviewer: str                 # "cc" | "gaia-study" | "human"
    review_direction: Literal["forward", "reverse"]  # forward: blueprint is truth; reverse: code is truth
    review_timestamp: datetime
    overall_fidelity_score: float # 0.0–1.0
    discrepancies: List[DiscrepancyItem]
    open_question_updates: List[OpenQuestionUpdate]
    promotion_recommendation: Literal["approve", "approve_with_notes", "reject"]
    summary_note: str             # 2-3 sentence human-readable summary
```

#### 3.2.1 Token Budget Guard

The assembled review prompt for a multi-file service can be large. For CC review sessions (Phase 2), this is manageable — Claude has ample context. But for Phase 5 sleep-cycle reviews where gaia-prime (Qwen3-4B, 8K context window, ~7424 usable tokens after response buffer) runs autonomously, prompts must fit within the model's effective context.

The review prompt builder must accept a `max_prompt_tokens` parameter (default: `None` for CC, e.g. `3072` for gaia-prime) and enforce it via progressive truncation:

**Truncation priority (drop lowest-priority sections first):**

1. **Open questions** — most expendable, least impact on structural review
2. **Intent section** — can be reduced to `purpose` only (drop `design_decisions`)
3. **AST summaries** — truncate per-file summaries to signatures-only (drop `error_handlers`, `http_calls`, `gaia_imports`)
4. **Pre-check results** — reduce to summary line only (drop per-item detail)
5. **Interfaces/failure modes** — NEVER truncated, these are the review target

The builder should log a warning when truncation is applied, including which sections were reduced and the final token count.

**Token estimation:** Use a simple `len(text) / 4` heuristic for token count (conservative for English text). Do not add a tokenizer dependency just for this — the guard is a safety rail, not a precise budget.

**Testing:** `gaia-common/tests/utils/test_review_prompt_builder.py` — verify prompt renders correctly from a seed blueprint + mock AST summaries; verify ReviewResult parses from a known CC response. Test that truncation activates correctly when `max_prompt_tokens` is set below the natural prompt size, and that critical sections survive truncation.

### 3.3 ruff Format Post-Processing Hook

**Location:** `scripts/format_candidate.sh`

**Purpose:** Apply `ruff format` and `ruff check --fix` to all Python files in a candidate service directory before validation runs. This is a one-liner wrapper but should be explicit and logged.

```bash
#!/bin/bash
# format_candidate.sh <service_name>
# Applies ruff format + auto-fixable lint rules to candidate source
SERVICE=$1
CANDIDATE_DIR="/gaia/GAIA_Project/candidates/$SERVICE"
echo "Formatting $CANDIDATE_DIR..."
ruff format "$CANDIDATE_DIR"
ruff check --fix --select I "$CANDIDATE_DIR"   # import sorting only for auto-fix
echo "Format complete."
```

Wire this into `promote_pipeline.sh` Stage 3 (before ruff lint, not instead of it).

### 3.4 mypy Promotion to Error

In `promote_candidate.sh`, change mypy result from `warn` to `fail` for gaia-core, gaia-mcp, and gaia-study. gaia-common already has the most complete type coverage and should be the model. Add a `# type: ignore` count check — compare the candidate's ignore count against the live version of the same file (not the entire branch). Fail if any individual file's ignore count increases. Implementation: a small Python helper script (`scripts/check_type_ignores.py`) that takes two directories and diffs per-file counts, called from `promote_candidate.sh`.

### 3.5 Mechanical Pre-Check Validator (Refactored)

**Extracted from:** `candidates/gaia-core/gaia_core/cognition/sleep_task_scheduler.py` (`_run_blueprint_validation` handler)
**New location:** `gaia-common/gaia_common/utils/blueprint_precheck.py`

**Purpose:** Factor the existing mechanical blueprint-vs-code validation logic out of the sleep-task handler into a standalone, importable function. This serves two consumers:

1. **The sleep-task handler** — calls it as before during sleep cycles (no behavior change)
2. **The review prompt builder** — calls it to produce structured pre-check annotations that are included in the LLM review prompt

**Function signature:**

```python
@dataclass
class PreCheckItem:
    category: Literal["endpoint", "enum_member", "constant", "failure_mode", "dependency"]
    blueprint_claim: str          # what the blueprint declares
    status: Literal["found", "missing", "diverged"]
    source_file: Optional[str]    # where it was found (or expected)
    detail: str                   # e.g., "GET /health found at router.py:42"

@dataclass
class PreCheckResult:
    service_id: str
    timestamp: datetime
    items: List[PreCheckItem]
    summary: PreCheckSummary       # counts by category and status

    def to_prompt_text(self) -> str:
        """Render as a concise block for inclusion in an LLM review prompt."""

def run_blueprint_precheck(
    blueprint_path: str,
    source_dir: str,
    *,
    categories: Optional[List[str]] = None,  # filter to specific categories
) -> PreCheckResult:
    """
    Run mechanical blueprint-vs-code validation.

    Checks structural presence of blueprint-declared items in source code
    using regex and AST extraction (NOT LLM inference). Fast and deterministic.

    This is the same logic currently in sleep_task_scheduler._run_blueprint_validation,
    refactored for reuse.
    """
```

**What the pre-check covers:**

| Category | What it checks | Method |
|----------|----------------|--------|
| `endpoint` | Each interface endpoint path + HTTP method exists as a `@router.{method}("{path}")` or `@app.websocket("{path}")` decorator | Regex scan of source files |
| `enum_member` | Each enum declared in blueprint has matching `class(Enum)` with expected members | AST extraction |
| `constant` | Each constant declared in blueprint exists as `UPPER_CASE = value` at module level | AST extraction |
| `failure_mode` | Each failure mode ID has a matching exception handler or status code return | Regex scan for exception types + HTTP status codes |
| `dependency` | Each declared dependency service appears in import statements or client calls | Regex scan for service names in imports and URL patterns |

**Per-dimension pre-check reliability:**

| Category | Estimated accuracy | Why |
|----------|-------------------|-----|
| `endpoint` | ~95% | REST decorator patterns are highly regular; WebSocket endpoints matched via `@app.websocket()` or `@router.websocket()` patterns |
| `dependency` | ~85% | Misses env-var-based URLs and indirect client construction |
| `failure_mode` | ~50-60% | Misses middleware handlers, base class handlers, delegation to gaia-common utilities, retry/circuit-breaker patterns, conditional guards that aren't try/except |
| `enum_member` | ~95% | AST extraction is deterministic for enum classes |
| `constant` | ~90% | Misses constants defined in config files or env vars |

The LLM reviewer carries the majority of dimension 3 (failure mode coverage) load. The pre-check's `[FOUND]`/`[MISSING]` for failure modes should be treated as a *hint*, not ground truth. The review prompt instructions reflect this — the LLM is told to check for "non-standard patterns" on [MISSING] items, which is where the pre-check's blind spots concentrate.

**What the pre-check does NOT cover** (this is the LLM reviewer's domain):
- Whether an endpoint's *behavior* matches the blueprint's description
- Whether a failure mode handler actually implements the *correct recovery strategy*
- Whether the dependency calls use the *right parameters*
- Whether the code's overall architecture reflects the blueprint's *intent*
- Whether open questions have been answered or new ones raised

**Pre-check output in the review prompt:**

The `to_prompt_text()` method renders a concise block like:

```
## Mechanical Pre-Check Results: gaia-core

### Endpoints (7/8 found)
  [FOUND] GET  /health                     → main.py:15
  [FOUND] POST /turn                       → run_turn.py:45
  [FOUND] GET  /sessions                   → sessions.py:22
  [FOUND] POST /sessions/{id}/message      → sessions.py:67
  [FOUND] GET  /cognition/status           → cognition_manager.py:31
  [FOUND] POST /cognition/sleep            → cognition_manager.py:89
  [FOUND] GET  /blueprints                 → blueprint_router.py:14
  [MISSING] DELETE /sessions/{id}          → expected in sessions.py

### Failure Modes (4/5 matched)
  [FOUND] mcp_timeout           → httpx.TimeoutException handler in mcp_client.py:112
  [FOUND] study_unavailable     → ConnectionError handler in study_client.py:45
  [FOUND] invalid_session       → ValueError guard in sessions.py:71
  [FOUND] token_limit_exceeded  → conditional return in run_turn.py:89
  [MISSING] blueprint_parse_error → no matching handler found

### Dependencies (3/3 confirmed)
  [FOUND] gaia-mcp              → import in mcp_client.py
  [FOUND] gaia-study            → import in study_client.py
  [FOUND] gaia-prime (via vLLM) → httpx call in inference.py:33

### Summary
  Total checks: 16 | Found: 14 | Missing: 2 | Diverged: 0
  Structural completeness: 87.5%
```

This gives the LLM reviewer a **cheat sheet** for dimensions 1-3. The reviewer doesn't need to independently discover whether a `try/except` exists — the pre-check already answered that. The reviewer's job shifts to:
- **For [FOUND] items:** Does the implementation actually match the blueprint's *semantic intent*, not just its surface structure?
- **For [MISSING] items:** Is this genuinely missing, or is it implemented in a non-standard way the pre-check couldn't detect?
- **For dimensions 4-5 (intent coherence, open questions):** These remain pure LLM judgment calls, unaffected by pre-checks.

**Testing:** `gaia-common/tests/utils/test_blueprint_precheck.py` — verify pre-check output against a known service (gaia-core) with at least one [FOUND] and one [MISSING] item seeded by a modified test blueprint.

**Migration:** After extracting the logic, update `sleep_task_scheduler.py` to import and call `run_blueprint_precheck()` instead of its inline implementation. This is a pure refactor with no behavior change for the sleep cycle.

---

## 4. Phase 1: CC Generation Workflow

This phase establishes the pattern by which CC generates candidate code from a blueprint seed. It does not require any new GAIA services — CC runs this directly via the shell.

### 4.1 The Generation Context

When CC is asked to generate a new GAIA service, it must receive:

1. **The blueprint seed YAML** — the prescriptive design intent
2. **All live blueprints** — so it understands the existing service graph and available dependencies
3. **Three reference implementations** — the most recently promoted services (from dev journal dates) as style exemplars
4. **gaia-common source** — so it knows the exact types, protocols, and utilities available
5. **The generation rubric** — explicit criteria CC must satisfy (see 4.2)

CC should NOT receive: the full source of every existing service (context budget), any previous generation attempts for this service (independence), or instructions that override the blueprint.

### 4.2 Generation Rubric

CC must produce code that satisfies these criteria, verified before handing off to the reviewer:

**Structural requirements:**
- Every endpoint declared in `blueprint.interfaces` (http_rest type) is implemented with the exact path and HTTP method
- Every dependency in `blueprint.dependencies.services` is called only through gaia-common client utilities (never raw `httpx` or `requests` directly)
- Every failure mode in `blueprint.failure_modes` has a corresponding `try/except` or conditional guard with the documented response
- No imports from services not in `blueprint.dependencies` (gaia-common is universally available and exempt from this rule)
- All public functions have type annotations on parameters and return values

**Style requirements (GAIA idioms):**
- FastAPI router pattern matching gaia-core/gaia-mcp
- Pydantic models for all request/response bodies
- Structured logging via `logger = logging.getLogger(__name__)`
- Constants in `UPPER_CASE` at module level, not inline
- Health check endpoint at `GET /health` returning `{"status": "healthy", "service": "{service_id}"}`

**Documentation requirements:**
- Module-level docstring explaining the service's cognitive role
- Docstring on every class and every public function
- Inline comments only where non-obvious logic exists

### 4.3 Generation Output Structure

CC should produce a complete candidate directory at:

```
candidates/{service_id}/
├── {service_id}/
│   ├── __init__.py
│   ├── main.py               # FastAPI app + router registration
│   ├── {module_a}.py         # as declared in blueprint.source_files
│   └── {module_b}.py
├── tests/
│   └── test_{service_id}_smoke.py   # basic smoke tests
├── requirements.txt
└── Dockerfile
```

The blueprint seed YAML remains at its canonical location (`knowledge/blueprints/candidates/{service_id}.yaml`), NOT copied into the candidate directory. The review and generation scripts reference it from there.

CC should also run `scripts/format_candidate.sh {service_id}` on its own output before marking generation complete.

### 4.4 Reverse Blueprint Generation (Code → Blueprint)

The standard flow is blueprint → code → review. But for existing services that were developed without a blueprint (e.g., gaia-audio), or for services whose code has drifted significantly from an outdated blueprint, we need the reverse: **code → blueprint → review**.

This is a fundamentally different task from forward generation. It requires reading existing code and synthesizing a prescriptive specification that accurately captures what the code *does* and *intends*.

#### 4.4.1 When to Use Reverse Generation

- A candidate service exists with no blueprint (gaia-audio)
- A live service's blueprint has drifted beyond repair (blueprint `confidence` scores all below 0.4)
- A new service was organically developed (not from a blueprint seed) and needs to join the blueprint system

#### 4.4.2 Reverse Generation Workflow

| Step | Agent | Input | Output |
|------|-------|-------|--------|
| 1. Mechanical extraction | AST summarizer + pre-check | Full source code | Structured summary: endpoints, models, imports, error handlers, constants |
| 2. Blueprint draft | CC (Claude Code) | AST summaries + BlueprintModel schema + 2-3 existing blueprints as exemplars | Candidate blueprint YAML |
| 3. Structural validation | Pre-check validator | Draft blueprint + source code | PreCheckResult: which claims are structurally confirmed |
| 4. Semantic review | CC (cold session) | Draft blueprint + AST summaries + pre-check results | ReviewResult: does the blueprint accurately describe the code? |
| 5. Blueprint refinement | CC or human | ReviewResult discrepancies | Updated blueprint YAML addressing discrepancies |

**Key difference from forward review:** In forward review, the blueprint is the source of truth and the code is evaluated against it. In reverse review, the *code* is the source of truth and the *blueprint* is evaluated against it. The review prompt must be modified for this inversion — the question is "does this blueprint accurately describe this code?" not "does this code implement this blueprint?"

#### 4.4.3 Reverse Review Prompt Variant

The review prompt builder (Section 3.2) must accept a `review_direction` parameter:

- `review_direction="forward"` (default) — "Does the code implement the blueprint?"
- `review_direction="reverse"` — "Does the blueprint accurately describe the code?"

For reverse reviews, the prompt's SYSTEM instruction changes:

```
You are verifying that a DRAFT blueprint accurately captures the behavior of
existing, working code. The code is the source of truth. Your task is to identify
claims in the blueprint that are:
- MISSING: behavior present in the code but absent from the blueprint
- INACCURATE: claims that don't match the code's actual behavior
- INCOMPLETE: claims that are directionally correct but lack specificity

Do NOT evaluate code quality. The code works. Evaluate blueprint accuracy.
```

#### 4.4.4 Training Pair Generation from Reverse Reviews

Reverse reviews produce training pairs with `pair_type: "reverse"` (distinct from "forward" and "retroactive"). These pairs teach a different skill: code comprehension and specification synthesis. They are valuable for a potential future `code-reviewer` adapter but are also useful for `code-architect` because they demonstrate the mapping between code patterns and blueprint specifications — the same mapping the architect needs to implement in the forward direction.

Include reverse pairs in the training corpus with `pair_type: "reverse"`. They count toward the total corpus size but NOT toward the forward-pair minimum (Section 8.1).

#### 4.4.5 GAIA's Role in Reverse Generation

gaia-prime (Qwen3-4B, 8K context) cannot currently author a complete blueprint from scratch — the task requires seeing the full service source (1000+ lines across multiple files) plus the BlueprintModel schema plus exemplar blueprints, which exceeds the context window.

However, GAIA can participate in **step 3 (structural validation)** and in **scoped section review** during step 4 — given a single AST summary file + the corresponding blueprint section, she can verify accuracy within her context budget. This iterative, section-at-a-time pattern is a realistic near-term capability.

After the code-architect adapter is trained (Phase 4), GAIA's reverse generation capability should be re-evaluated — the adapter may provide enough architectural priors to compensate for the context limitation.

---

## 5. Phase 2: CC Review Workflow (Independent Session)

This is the core of the bootstrap mechanism. It must be a genuinely independent CC invocation — not a continuation of the generation session.

### 5.1 Review Session Setup

The review session receives:

1. **The blueprint seed** — the design specification
2. **AST summaries of all generated files** — produced by running `ast_summarizer.py` on the candidate directory (not raw source)
3. **Mechanical pre-check results** — produced by running `run_blueprint_precheck()` against the candidate source (deterministic structural completeness check)
4. **The ReviewResult schema** — so CC knows exactly what to produce
5. **Explicit instruction:** "You did not write this code. The mechanical pre-check shows structural completeness. Assess semantic fidelity beyond what the pre-check covers."

CC must not receive: the full source files, any generation session context, or any human commentary on the code quality.

### 5.2 Review Session Prompt Template

**Location:** `scripts/review_templates/cc_review_prompt.txt`

```
You are performing a blueprint fidelity review for the GAIA AI system.

You will be given:
1. A blueprint specification (YAML)
2. Mechanical pre-check results (deterministic structural completeness)
3. AST summaries of the candidate source files

The mechanical pre-check has already verified structural presence of endpoints,
failure mode handlers, dependencies, enums, and constants. Your task is to
assess SEMANTIC fidelity — whether the structurally-present elements actually
implement the blueprint's intent correctly.

Produce a ReviewResult JSON object identifying all discrepancies.

IMPORTANT:
- You are reviewing for blueprint fidelity ONLY, not general code quality
- Use the pre-check results as your starting point for dimensions 1-3
- For [FOUND] items: verify semantic correctness from AST summaries
- For [MISSING] items: confirm absence or identify non-standard implementations
- Dimensions 4-5 (intent coherence, open questions) have no pre-check coverage
- Your output must be valid JSON matching the ReviewResult schema exactly

ReviewResult schema:
{schema_json}

---

BLUEPRINT:
{blueprint_yaml}

---

MECHANICAL PRE-CHECK:
{precheck_text}

---

AST SUMMARIES:
{ast_summaries}

---

Produce your ReviewResult JSON now:
```

### 5.3 Review Invocation Script

**Location:** `scripts/cc_review_candidate.sh`

```bash
#!/bin/bash
# cc_review_candidate.sh <service_id> [--live]
# Generates AST summaries and prepares CC review prompt for a candidate
# (or live) service.
#
# --live: review a live service instead of a candidate (for retroactive corpus)
#
# Outputs:
#   candidates/{service_id}/review/review_prompt_{timestamp}.txt
#   candidates/{service_id}/review/ast_summaries_{timestamp}.json

SERVICE=$1
LIVE_MODE=false
if [ "$2" == "--live" ]; then
    LIVE_MODE=true
fi

TIMESTAMP=$(date +%Y%m%dT%H%M%S)

if [ "$LIVE_MODE" == "true" ]; then
    # For retroactive corpus: review live service source against live blueprint
    SOURCE_DIR="/gaia/GAIA_Project/${SERVICE}"
    BLUEPRINT="/gaia/GAIA_Project/knowledge/blueprints/${SERVICE}.yaml"
    REVIEW_DIR="/gaia/GAIA_Project/knowledge/curricula/code-architect/retroactive/${SERVICE}"
else
    SOURCE_DIR="/gaia/GAIA_Project/candidates/$SERVICE"
    BLUEPRINT="/gaia/GAIA_Project/knowledge/blueprints/candidates/${SERVICE}.yaml"
    REVIEW_DIR="$SOURCE_DIR/review"
fi

mkdir -p "$REVIEW_DIR"

# Step 1: Generate AST summaries
python3 scripts/generate_ast_summaries.py \
    --source-dir "$SOURCE_DIR" \
    --output "$REVIEW_DIR/ast_summaries_${TIMESTAMP}.json"

# Step 2: Run mechanical pre-check
python3 scripts/run_blueprint_precheck.py \
    --blueprint "$BLUEPRINT" \
    --source-dir "$SOURCE_DIR" \
    --output "$REVIEW_DIR/precheck_${TIMESTAMP}.json"

# Step 3: Build review prompt (combines blueprint + pre-check + AST summaries)
python3 scripts/build_review_prompt.py \
    --blueprint "$BLUEPRINT" \
    --ast-summaries "$REVIEW_DIR/ast_summaries_${TIMESTAMP}.json" \
    --precheck "$REVIEW_DIR/precheck_${TIMESTAMP}.json" \
    --output "$REVIEW_DIR/review_prompt_${TIMESTAMP}.txt"

# Step 3: Instructions for CC invocation
echo ""
echo "Review prompt written to: $REVIEW_DIR/review_prompt_${TIMESTAMP}.txt"
echo ""
echo "To invoke CC for review, start a NEW Claude Code session and provide"
echo "the review prompt file as context. CC must not have prior context about"
echo "this service from a generation session."
echo ""
echo "Save CC output to: $REVIEW_DIR/cc_review_${TIMESTAMP}.json"
```

**Note on CC invocation:** The actual mechanism for invoking CC in a cold session depends on the available integration (Claude Code CLI, Anthropic API, or manual copy-paste). The critical requirement is session independence — the reviewer CC must have no memory of the generation session. The script prepares all artifacts; the invocation step is intentionally left flexible.

### 5.4 Review Result Handling

After CC produces a `ReviewResult` JSON:

1. **Parse and validate** against the Pydantic schema
2. **If `promotion_recommendation == "reject"`:** log to dev journal, return code to generation phase with discrepancy list attached
3. **If `promotion_recommendation == "approve_with_notes"`:** log discrepancies as `open_questions` in the blueprint seed, continue to promotion pipeline
4. **If `promotion_recommendation == "approve"`:** continue directly to promotion pipeline

In all cases, the ReviewResult is written to:
- `candidates/{service_id}/review/cc_review_{timestamp}.json` (or `retroactive/{service_id}/` for live reviews)
- `knowledge/curricula/code-architect/reviews/{service_id}_{timestamp}.json` — training corpus

---

## 6. Phase 2.5: Retroactive Corpus Bootstrapping

### The Bottleneck

Phase 3 (Section 7) requires 50 training pairs before QLoRA training can start. If each pair requires generating a *new* GAIA service, reviewing it, fixing it, and promoting it — that's 50 full development cycles. At current pace, that would take months.

### The Solution

We already have **6 live promoted services** with validated blueprints and production source code that passed the promotion pipeline. These are pre-existing (blueprint, code, validation outcome) tuples. By running them through the review infrastructure built in Phases 0-2, we can bootstrap the training corpus immediately.

### 6.1 Retroactive Review Process

For each live service with a LIVE blueprint:

1. **Filter by confidence:** Only include services where `blueprint.meta.confidence` averages ≥ 0.6 across all sections. Low-confidence blueprints produce noisy training signal.

2. **Run AST summarizer** on the live service's production source files.

3. **Build review prompts.** Two granularities:
   - **Per-service review:** Full blueprint + all file summaries → one ReviewResult
   - **Per-file reviews:** For each source file listed in `blueprint.source_files`, scope the blueprint context to the relevant subset (the interfaces that file implements, the failure modes it handles, the dependencies it uses). This produces more granular training pairs.

4. **Have CC review in cold sessions.** The independent review principle (Section 2.6) applies — each review must be a fresh CC session with no prior context about the service.

5. **Record results** to `knowledge/curricula/code-architect/retroactive/{service_id}/`.

### 6.2 Per-File Blueprint Scoping

Per-file training pairs require scoping the blueprint to what each file is responsible for. Use the `source_files` section's role tags to determine which blueprint sections are relevant:

- A file tagged as `router` or `api` → scope to the interfaces it declares + related failure modes
- A file tagged as `model` or `schema` → scope to the data models referenced by interfaces
- A file tagged as `client` → scope to the dependencies section
- A file tagged as `core` or `logic` → scope to intent + failure modes
- **Any file with no role tag, multiple role tags, or a role that doesn't map cleanly** → **use the full blueprint**. Many GAIA service files are composite — a handler that both exposes an endpoint AND calls gaia-mcp, or a core module that also defines models. Silently dropping relevant blueprint sections for these files would produce training pairs with incomplete context. The full-blueprint default is the safe choice; the scoped variants are an optimization for clearly single-responsibility files.

The review prompt builder (Section 3.2) should accept an optional `scope_to_file` parameter that filters the blueprint context accordingly. When the scoping logic cannot confidently reduce the blueprint (composite files, ambiguous roles), it must pass `scope_to_file=None` and include the full blueprint.

### 6.3 Retroactive Review Result Handling

Retroactive reviews follow a different result handling path from forward reviews (Section 5.4), because the code under review is already live and promoted.

**If `promotion_recommendation == "approve"` or `"approve_with_notes"`:**
- Record the ReviewResult as a training pair (standard path)
- For `approve_with_notes`: append discrepancies to the live blueprint's `open_questions` for future awareness, but do NOT trigger remediation — the service is already running and passed promotion

**If `promotion_recommendation == "reject"`:**
This means CC found a significant blueprint fidelity problem in a live promoted service. This is a different outcome from a forward-pair rejection — we cannot "return to generation" because the code is already in production. Instead:

1. **Record the ReviewResult as a training pair** — rejections are high-value training signal regardless of source
2. **Append all `critical` and `major` discrepancies to the live blueprint's `open_questions`** with a tag: `source: retroactive_review`
3. **Create a Review Queue item** (if the Web UI Review Queue exists, per Section 9.2) flagging the service for human attention
4. **Log to `prime.md`:** *"Retroactive review of {service_id} found {n} critical/{m} major discrepancies. Blueprint fidelity concern on a live service — human review recommended."*
5. **Do NOT automatically trigger a remediation cycle.** The service passed promotion, meaning it satisfies mechanical gates and smoke tests. A CC review rejection on a live service most likely indicates either (a) blueprint drift since promotion, or (b) a blueprint that was incomplete when the service was promoted. Both require human judgment about whether the *code* needs fixing or the *blueprint* needs updating.

This path ensures retroactive rejections surface as actionable information without triggering automated code changes to running services.

### 6.5 Training Signal Characteristics

Retroactive pairs have different training signal from forward pairs:

| | Retroactive Pairs | Forward Pairs |
|---|---|---|
| **Code quality** | Already promoted, high fidelity | Pre-review, may have discrepancies |
| **Training signal** | "What good looks like" | "What mistakes to avoid + how to fix" |
| **CC review tone** | Mostly approvals, minor observations | May include rejections, major issues |
| **Volume** | Bounded by existing services (~50-75) | Unbounded, grows with development |

Both signals are complementary. The training pair schema (Section 7.1) includes a `pair_type` field to distinguish them so their effectiveness can be analyzed separately during validation.

### 6.6 Corpus Size Estimate (All Sources)

| Source | Type | Estimated pairs |
|--------|------|----------------|
| 6 live services, per-service reviews | retroactive | 6 |
| ~8-12 source files per service, per-file reviews | retroactive | 48-72 |
| 32 dev journals, patterns A+C only | journal | 40-70 |
| gaia-audio reverse blueprint (per-service + per-file) | reverse | ~8 |
| gaia-web GUI features (audio + sleep log, per sub-task with GAIA) | forward | ~6-10 |
| **Total estimated corpus** | | **~108-166** |

This **substantially exceeds** the 50-pair training threshold. The three non-forward sources (retroactive + journal + reverse) provide volume, while the pilot project's forward pairs provide the error-correction signal the forward-pair minimum requires.

**Forward-pair ratio concern:** With ~6-10 forward pairs out of ~108-166 total, the initial forward ratio is only ~4-6% — well below the 15% minimum (Section 8.1). This means the first QLoRA training cycle can proceed with the loss weight multiplier compensating for the imbalance (Section 8.1), but the 15% threshold for unweighted training will only be met after additional forward pairs accumulate from ongoing development. This is acceptable: the first training cycle is explicitly a calibration run (see calibration note in Section 8.1).

### 6.7 Retroactive Corpus Generation Script

**Location:** `scripts/generate_retroactive_corpus.sh`

```bash
#!/bin/bash
# generate_retroactive_corpus.sh [--dry-run]
# Enumerates all live services with qualifying blueprints and generates
# review prompts for retroactive corpus bootstrapping.
#
# Prerequisite: Phase 0 infrastructure (ast_summarizer, review_prompt_builder)
#
# Outputs: knowledge/curricula/code-architect/retroactive/{service_id}/

DRY_RUN=false
if [ "$1" == "--dry-run" ]; then
    DRY_RUN=true
fi

BLUEPRINT_DIR="/gaia/GAIA_Project/knowledge/blueprints"
OUTPUT_DIR="/gaia/GAIA_Project/knowledge/curricula/code-architect/retroactive"

mkdir -p "$OUTPUT_DIR"

for bp_file in "$BLUEPRINT_DIR"/*.yaml; do
    SERVICE=$(basename "$bp_file" .yaml)

    # Check confidence threshold
    AVG_CONFIDENCE=$(python3 -c "
import yaml
with open('$bp_file') as f:
    bp = yaml.safe_load(f)
conf = bp.get('meta', {}).get('confidence', {})
vals = [v for v in conf.values() if isinstance(v, (int, float))]
print(sum(vals) / len(vals) if vals else 0)
")

    if (( $(echo "$AVG_CONFIDENCE < 0.6" | bc -l) )); then
        echo "SKIP $SERVICE (avg confidence $AVG_CONFIDENCE < 0.6)"
        continue
    fi

    echo "QUALIFY $SERVICE (avg confidence $AVG_CONFIDENCE)"

    if [ "$DRY_RUN" == "true" ]; then
        continue
    fi

    # Generate per-service review
    bash scripts/cc_review_candidate.sh "$SERVICE" --live

    # Generate per-file reviews
    python3 scripts/generate_ast_summaries.py \
        --source-dir "/gaia/GAIA_Project/${SERVICE}" \
        --per-file \
        --blueprint "$bp_file" \
        --output-dir "$OUTPUT_DIR/$SERVICE/per_file/"
done

echo ""
echo "Retroactive corpus preparation complete."
echo "Next: invoke CC in cold sessions for each review prompt."
```

### 6.8 Retroactive Corpus Storage

```
knowledge/curricula/code-architect/
├── retroactive/                          # NEW
│   ├── {service_id}/
│   │   ├── ast_summaries_{ts}.json       # per-service summaries
│   │   ├── review_prompt_{ts}.txt        # per-service review prompt
│   │   ├── cc_review_{ts}.json           # CC review result (after invocation)
│   │   └── per_file/
│   │       ├── {filename}_summary.json
│   │       ├── {filename}_prompt.txt
│   │       └── {filename}_review.json
│   └── corpus_manifest.json              # tracks which services/files reviewed
├── reviews/                              # forward pair reviews (existing)
├── pairs/                                # processed training pairs (existing)
└── ...
```

### 6.9 Dev Journal Corpus Extraction (Historical Training Data)

The retroactive approach (Section 6.1-6.8) generates training pairs by having CC review existing live code against existing blueprints. But the GAIA project has a second, richer source of structured training data: **32 dev journals** spanning January-February 2026, each following a consistent template and linked to specific git commits.

#### 6.9.1 Why Dev Journals Are Valuable

Dev journals contain information that retroactive reviews cannot capture:

| Signal Type | Retroactive Review | Dev Journal |
|---|---|---|
| Final code state | Yes | Yes |
| Blueprint alignment | Yes | Partial (journals reference blueprint sections) |
| **Problem → fix narrative** | No | Yes (debugging journals) |
| **Architectural rationale** | No | Yes (design decisions sections) |
| **Test-driven cycles** | No | Yes (test summaries with pass/fail) |
| **Promotion validation** | No | Yes (8 promotion journals with stage results) |
| **Bug pattern recognition** | No | Yes (root cause → fix → verify tuples) |

Retroactive pairs teach "what correct code looks like." Dev journal pairs teach "how to reason about code problems" — a different and complementary training signal.

#### 6.9.2 Extractable Training Tuple Patterns

**Included in code-architect training (generation-oriented):**

**Pattern A: Blueprint-Anchored Change**
- Input: Blueprint section + "the problem" from journal context
- Output: The code change (from git diff via commit hash)
- Signal: Promotion outcome (passed/failed/modified)
- Example: Temporal awareness journal → `lite_journal.py` 306 lines → "20/20 tests PASS"
- **Why included:** Directly teaches blueprint → code generation in a real-world context

**Pattern C: Audit Finding → Remediation**
- Input: Security/quality finding + severity
- Output: Specific code change
- Signal: Finding status (fixed/deferred/acknowledged)
- Example: "Shell injection in mcp_client.py" → `shell=False` + `shlex.split()` → promoted
- **Why included:** Teaches corrective code generation given a structured deficiency report — similar to the review → fix cycle in forward pairs

**Deferred to future `code-reviewer` adapter:**

**Pattern B: Bug Diagnosis → Fix** — This is a reasoning/diagnosis task, not a generation task. The input is a symptom and the expected output is a root cause analysis. Valuable for a reviewer/debugger adapter but doesn't fit code-architect's generation objective.

**Pattern D: Promotion Sequence** — Operational knowledge about pipeline orchestration. Valuable for a future DevOps-aware adapter but orthogonal to blueprint-faithful code generation.

#### 6.9.3 Extraction Script

**Location:** `scripts/extract_journal_corpus.py`

This script:
1. Scans `knowledge/Dev_Notebook/*.md` for date-stamped dev journals
2. Parses each journal's structured sections (Context, Implementation, Test Summary, Files Summary, Validation)
3. Extracts commit hashes from journals and retrieves corresponding `git diff` data
4. Cross-references blueprint sections mentioned in journal text
5. Constructs training tuples with `pair_type: "journal"` and `journal_pattern: "A|B|C|D"`
6. Outputs to `knowledge/curricula/code-architect/journal/` directory

**Extraction is best-effort.** Journal formatting varies slightly across sessions. The script should parse what it can, log what it skips, and produce a manifest showing coverage. Human review of the manifest confirms pair quality before training inclusion.

#### 6.9.4 Journal Pair Characteristics

Journal pairs are distinct from both retroactive and forward pairs:

| | Retroactive | Forward | Journal | Reverse |
|---|---|---|---|---|
| **Source** | Live service review | New code generation | Historical dev records | Code → blueprint |
| **Signal** | "What good looks like" | "Mistakes + fixes" | "How to reason about changes" | "Code comprehension" |
| **Volume** | ~54-78 | Unbounded | ~40-70 (patterns A+C from 32 journals) | Per new service |
| **Quality** | Uniform (all promoted) | Variable (pre-review) | High (includes validation) | High (working code) |

Journal pairs count toward the total corpus size but NOT toward the forward-pair minimum (Section 8.1). Like retroactive pairs, they represent already-validated work. The forward-pair minimum specifically targets error-correction learning that only comes from generating new code and having it reviewed.

#### 6.9.5 Corpus Storage

```
knowledge/curricula/code-architect/
├── journal/                              # NEW
│   ├── {date}_{topic}/
│   │   ├── journal_context.json          # parsed journal sections
│   │   ├── git_diff.patch                # corresponding code changes
│   │   ├── blueprint_context.json        # referenced blueprint sections
│   │   └── training_tuples.json          # extracted (input, output, signal) tuples
│   └── journal_manifest.json             # coverage tracking
├── retroactive/                          # existing
├── reviews/                              # existing
├── pairs/                                # existing
└── ...
```

---

## 7. Phase 3: Training Corpus Construction

Every CC review — whether from forward generation (Phase 2), retroactive bootstrapping (Phase 2.5), or journal extraction (Phase 2.5, Section 6.9) — produces a training example. The corpus accumulates from all sources.

### 7.1 Training Pair Schema

Each training pair captures a review cycle:

```json
{
  "pair_id": "uuid",
  "pair_type": "forward | retroactive | reverse | journal",
  "granularity": "service | file",
  "service_id": "gaia-example",
  "file_scope": "router.py | null",
  "created_at": "ISO datetime",
  "blueprint_yaml": "...",
  "blueprint_scoped": "... (subset if per-file, full if per-service)",
  "ast_summaries": { "filename": { ... }, ... },
  "cc_review": { "ReviewResult object" },
  "promotion_outcome": "passed | failed | modified",
  "modifications_before_promotion": [
    {
      "file": "example/router.py",
      "change_type": "added | removed | modified",
      "description": "Added missing error handler for MCP timeout"
    }
  ],
  "divergence_score_final": 0.12,
  "ground_truth_fidelity": 0.88
}
```

The `ground_truth_fidelity` is computed as: `1.0 - (critical_count * 0.3 + major_count * 0.15 + minor_count * 0.05)`, clamped to [0.0, 1.0].

**Note on the fidelity formula:** This does not normalize by total blueprint surface area (number of endpoints, failure modes, etc.). A service with 2 endpoints and 1 critical issue scores the same as a service with 20 endpoints and 1 critical issue. Consider normalizing by `total_checkpoints` after seeing the first batch of reviews — the current formula is a starting point optimized for consistency over precision. Track `total_checkpoints` in the pair metadata so normalization can be applied retroactively.

### 7.2 Corpus Storage

**Location:** `knowledge/curricula/code-architect/`

```
knowledge/curricula/code-architect/
├── curriculum.json              # adapter spec (see 7.3)
├── retroactive/                 # retroactive reviews (Phase 2.5)
├── reviews/                     # forward CC review JSON files
│   └── {service_id}_{ts}.json
├── pairs/                       # processed training pairs (both types)
│   └── {pair_id}.json
├── train.jsonl                  # assembled by generate_pairs.py
├── validation.jsonl
└── generation_metadata.json
```

### 7.3 Adapter Curriculum Spec

**Location:** `knowledge/curricula/code-architect/curriculum.json`

```json
{
  "adapter_name": "code-architect",
  "adapter_version": 1,
  "tier": 1,
  "pillar": "cognition",
  "priority": "high",
  "description": "Blueprint-faithful Python code generation for GAIA services",
  "base_model": "gaia-prime",
  "hyperparameters": {
    "rank": 32,
    "alpha": 64,
    "dropout": 0.05,
    "target_modules": ["q_proj", "k_proj", "v_proj", "o_proj", "gate_proj", "up_proj"],
    "batch_size": 1,
    "gradient_accumulation_steps": 8,
    "learning_rate": 1e-4,
    "max_steps": 500,
    "max_sequence_length": 4096
  },
  "chat_template": "auto",
  "training_objective": "Given a blueprint specification and context, generate Python source code that achieves maximum blueprint fidelity (low divergence score).",
  "minimum_corpus_size": 50,
  "validation_split": 0.15,
  "activation_triggers": ["builder_panel", "gaia_initiated"],
  "routing_keywords": ["generate service", "implement blueprint", "write the code for"],
  "versioning": {
    "strategy": "incremental",
    "retention_count": 3,
    "rollback_supported": true
  }
}
```

Notes:
- Rank 32 (vs json-architect's 16) because code generation requires more expressive capacity than JSON formatting.
- `chat_template: "auto"` — derive the instruction format from the base model's tokenizer config rather than hardcoding `[INST]`/`<</SYS>>` tags, which are Llama-2-specific and may not match Qwen3-4B's expected format.
- `adapter_version` tracks training iterations. When retraining with an expanded corpus, increment the version. The `versioning.retention_count` keeps the last 3 adapter checkpoints for rollback.

### 7.4 Training Pair Generation Script

**Location:** `knowledge/curricula/code-architect/generate_pairs.py`

This script:
1. Reads all `reviews/*.json` and `retroactive/**/cc_review_*.json` files
2. For each review, loads the corresponding blueprint and AST summaries
3. Tags each pair with `pair_type` (forward/retroactive) and `granularity` (service/file)
4. Constructs instruction/output pairs using the base model's chat template (loaded from tokenizer config, NOT hardcoded)
5. Outputs to `train.jsonl` and `validation.jsonl` with stratified split (ensure both pair types are represented in validation)

#### 7.4.1 Reference Implementation Selection

The training instruction includes AST summaries from 3 existing live services as style exemplars. The selection mechanism must be deterministic and documented so training pairs have consistent context.

**Selection criteria (applied in order):**

1. **Dependency overlap** (primary): Select services that share the most declared dependencies with the target blueprint. A service that calls gaia-mcp and gaia-study is most similar to another service that calls gaia-mcp and gaia-study, because they face the same client patterns and failure modes.

2. **Interface type overlap** (secondary): Among tied candidates, prefer services with the same interface types (http_rest, websocket, mcp, etc.). A REST API service learns more from another REST API service than from a WebSocket service.

3. **Recency** (tiebreaker): Among still-tied candidates, prefer the most recently promoted (from dev journal dates), as they represent the latest idioms.

**Implementation:** A `select_reference_services(target_blueprint, all_blueprints, n=3)` function in `generate_pairs.py` that scores each live blueprint on dependency overlap (Jaccard similarity of dependency service sets) + interface type overlap (Jaccard of interface types), and returns the top `n`. This function must be used consistently for both training pair generation AND the live generation rubric (Section 4.1), so the model trains on the same reference selection logic it will encounter at inference time.

**Instruction format for each pair:**

```
{chat_template.system_prefix}
You are GAIA's code-architect. Generate Python source code that faithfully
implements the following blueprint. Your output must satisfy all contract,
dependency, failure mode, and intent specifications exactly.
{chat_template.system_suffix}

{chat_template.user_prefix}
BLUEPRINT:
{blueprint_yaml}

AVAILABLE GAIA IDIOMS (from reference implementations):
{3 reference AST summaries, selected by dependency + interface overlap}

Generate the implementation for: {service_id}
{chat_template.user_suffix}
```

**Output format for each pair:**
The correct code (post-review, post-modification, as-promoted) for each source file, structured as:
```
## FILE: {service_id}/{filename}
{source_code}
```

---

## 8. Phase 4: QLoRA Training & Validation

Once the corpus reaches `minimum_corpus_size` (50 pairs), trigger training.

### 8.1 Training Trigger

Add corpus size monitoring to the blueprint_validation sleep task in `sleep_task_scheduler.py`. Training readiness requires BOTH:

- **Total corpus size:** `len(pairs/*.json) >= minimum_corpus_size` (50)
- **Forward pair minimum:** At least 15% of pairs must be `pair_type: "forward"` (i.e., ≥ 8 forward pairs)

The forward-pair minimum prevents training on a corpus dominated by retroactive approvals. Retroactive pairs teach "what correct code looks like" but not "error → correction" patterns. Without forward pairs showing mistakes and fixes, the model may learn to produce superficially correct-looking code that doesn't handle the edge cases that real generation cycles reveal.

When both conditions are met and no `code-architect` adapter exists (or the current adapter version is stale):

1. Log a note to `prime.md`: *"code-architect corpus has reached training threshold ({n} pairs: {r} retroactive, {f} forward). Forward pair ratio: {f/n:.0%}. Recommend triggering training via promote_pipeline.sh --qlora --adapter code-architect"*
2. Add a `high_priority` flag to the blueprint's `open_questions`

If total corpus ≥ 50 but forward pairs < 15%, log a different note: *"code-architect corpus size sufficient ({n} pairs) but forward pair ratio too low ({f/n:.0%} < 15%). Need {needed} more forward pairs before training."*

Additionally, when constructing `train.jsonl`, apply a **loss weight multiplier of 1.5x** to forward pairs if they constitute < 30% of the corpus. This compensates for retroactive-pair dominance in early training cycles without requiring strict parity. The multiplier is applied in `generate_pairs.py` as a `weight` field in the JSONL records, consumed by `qlora_trainer.py`'s data loader.

**Calibration note:** The 15% minimum ratio, the 30% weighting threshold, and the 1.5x multiplier are initial estimates, not empirically derived. After the first training cycle, compare validation loss on the forward-pair subset vs. the retroactive-pair subset. If the model performs significantly worse on forward pairs (error-correction patterns) than retroactive pairs (approval patterns), increase the multiplier or the minimum ratio. If performance is comparable, the current values are adequate. Log the per-subset validation metrics in the dev journal alongside the composite score so these thresholds can be tuned with data.

Training is deliberately not triggered automatically — it requires a conscious human decision given the GPU time involved.

### 8.2 Validation Dimensions

The code-architect adapter requires a fundamentally different validation pipeline from json-architect's JSON validity scoring. It needs to shell out to ruff, run AST extraction pipelines, and compare against blueprints — too much logic to co-locate with the JSON validator.

**Location:** `candidates/gaia-study/scripts/validators/blueprint_fidelity.py`

Implement `BlueprintFidelityValidator` as a standalone module. Update `validate_adapter.py` to act as a router: `--validator json_schema` dispatches to the existing logic, `--validator blueprint_fidelity` dispatches to the new module. This keeps each validator self-contained and follows the extend-don't-duplicate principle.

Validation dimensions:

| Dimension | Weight | Measurement |
|-----------|--------|-------------|
| Contract completeness | 30% | % of blueprint endpoints present in generated code (via AST extraction) |
| Dependency correctness | 25% | % of API calls to declared dependencies only (via http_calls extraction) |
| Failure mode coverage | 25% | % of failure modes with observable handling (via error_handlers extraction) |
| Syntactic validity | 10% | Does `ruff check` pass with zero errors? (shell out to ruff) |
| Type annotation coverage | 10% | % of public functions with complete type annotations (via AST) |

**Composite score threshold for promotion:** 0.75 (higher than json-architect's 0.6 because code errors have higher blast radius than JSON formatting errors).

### 8.3 Graduation Criteria

Graduation means CC steps back from mandatory review of every generated candidate. There are two distinct capabilities to assess:

**Generation quality** (can code-architect produce good code?):

1. **Divergence score convergence:** Average divergence score of last 10 code-architect generations ≤ 0.15 (vs. CC-reviewed baseline of ≈ 0.12)
2. **First-pass promotion rate:** ≥ 80% of generated candidates pass the full promotion pipeline (mechanical gates + smoke tests) without modification

**Review quality** is a separate concern that applies to Phase 5 (sleep-cycle autonomous review), NOT to generation graduation. The code-architect adapter is trained for *generation*. Phase 5's sleep-task reviewer uses it with a review prompt, but the adapter wasn't specifically trained on the review task. If Phase 5 review quality needs to match CC, that requires a separate `code-reviewer` adapter or a review-specific fine-tuning pass — a future extension, not a graduation gate.

**CC spot-check** (is the generation quality durable?):

3. **CC audit agreement:** On a random sample of 5 generated candidates, CC reviews them cold and finds no `critical` discrepancies and ≤ 2 `major` discrepancies total across all 5. This validates that the mechanical gates and divergence scores aren't masking semantic issues that only an LLM reviewer would catch.

These metrics should be tracked in the dev journal. When all three are met for two consecutive training cycles, CC moves from mandatory reviewer to periodic auditor (every 5th generation rather than every generation).

---

## 9. Phase 5: gaia-study SELF_MODEL_UPDATE Integration

Once the code-architect adapter exists and is validated, integrate blueprint-anchored code review into the sleep cycle as a `SELF_MODEL_UPDATE` task variant.

### 9.1 New Task: code_review

Add to `sleep_task_scheduler.py` alongside `blueprint_validation`:

```python
SleepTask(
    task_id="code_review",
    task_type="SELF_MODEL_UPDATE",
    priority=4,              # lower than blueprint_validation (3)
    interruptible=True,
    estimated_duration_seconds=120,
    handler=self._run_code_review,
)
```

**Handler behavior:**
1. For each live service blueprint, run `ast_summarizer` on the live source files
2. Query gaia-prime with the `code-architect` adapter using the review prompt
3. Parse the ReviewResult
4. For any `critical` or `major` discrepancies, append to that service's blueprint `open_questions`
5. Write a summary to `prime.md`: *"Code review cycle complete. {n} discrepancies found across {m} services. {service_x}: {top discrepancy summary}."*

### 9.2 Review Queue Integration

Critical discrepancies from the sleep-cycle code review should surface in the Web UI's Review Queue (Phase 2 of the Blueprint System plan). Each item should show:
- Service ID
- Discrepancy dimension and severity
- Blueprint claim vs. code evidence
- CC recommendation

Human approval clears the item from the queue. Approval without fix closes the discrepancy as acknowledged. Approval with fix triggers a new promotion cycle.

---

## 10. Implementation Order & Dependencies

```
Phase 0 (Infrastructure)
├── 3.1  AST Summarizer (extend code_analyzer)     ← implement first, no deps
├── 3.2  Review Prompt Builder + schema             ← depends on 3.1, 3.5
├── 3.3  ruff format hook                           ← no dependencies, trivial
├── 3.4  mypy promotion to error + ignore check     ← no dependencies, small script
└── 3.5  Mechanical Pre-Check Validator (refactor)  ← no deps (extracts existing code)

Phase 1 (CC Generation Workflow)
├── 4.x  Generation rubric + context                ← depends on 3.1, 3.2
├── 4.3  Candidate directory structure               ← depends on generation rubric
└── 4.4  Reverse blueprint generation workflow       ← depends on 3.1, 3.2, 3.5

Phase 2 (CC Review Workflow)
├── 5.2  Review prompt template (+ reverse variant)  ← depends on 3.2, 3.5
├── 5.3  cc_review_candidate.sh (+ --live flag)      ← depends on 3.1, 3.5, 5.2
└── 5.4  Result handling + routing                   ← depends on 5.3

Phase 2.5 (Corpus Bootstrapping — 3 sources)
├── 6.1  Retroactive: generate_retroactive_corpus.sh ← depends on Phase 0 + 2
├── 6.2  Per-file blueprint scoping logic            ← depends on 3.2
├── 6.3  CC reviews of all live services (batch)     ← depends on 6.1
├── 6.4  Quality filtering (confidence ≥ 0.6)        ← depends on 6.3
├── 6.9  Journal: extract_journal_corpus.py          ← depends on 7.1 schema
├── 10.1 Pilot: gaia-audio reverse blueprint         ← depends on 4.4, Phase 0
├── 10.1 Pilot: gaia-web GUI feature (forward pair)  ← depends on 10.1 blueprint
└── 6.5  If corpus ≥ 50: Phase 4 unblocked           ← depends on 6.4 + 6.9

Phase 3 (Forward Corpus Construction)
├── 7.1  Training pair schema (+ pair_type tag)      ← depends on 5.4
├── 7.3  Curriculum spec (+ versioning, auto template) ← depends on 7.1
└── 7.4  generate_pairs.py (all pair types)          ← depends on 7.1, 7.3

Phase 4 (Training & Validation)
├── 8.1  Corpus size monitoring                      ← depends on Phase 3
├── 8.2  BlueprintFidelityValidator class             ← depends on 7.3
└── 8.3  Graduation tracking                         ← depends on 8.2

Phase 5 (Sleep Cycle Integration)
├── 9.1  code_review sleep task                      ← depends on Phase 4 adapter
└── 9.2  Review Queue integration                    ← depends on Blueprint System Phase 2
```

**Critical path acceleration:** Three corpus sources run in parallel during Phase 2.5:
1. Retroactive reviews of 6 live services (~54-78 pairs)
2. Dev journal extraction — patterns A+C from 32 journals (~40-70 pairs)
3. Pilot project forward + reverse pairs from gaia-audio + gaia-web GUI features (~14-18 pairs)

Combined, these should yield **~108-166 training pairs** — well above the 50-pair minimum, with the journal and pilot sources providing richer signal than retroactive reviews alone. Forward pairs (Phase 3) continue accumulating organically.

### 10.1 Pilot Project: gaia-audio Promotion + gaia-web GUI Features

The bootstrap plan needs a concrete, bounded target to validate against before scaling. The pilot project serves this purpose: **promote gaia-audio to production** (exercising reverse blueprint generation) and **add audio service awareness to gaia-web** (exercising forward code generation with GAIA participation).

#### 10.1.1 Pilot Part 1: gaia-audio Blueprint & Promotion

gaia-audio is a complete candidate service (1,233 lines, 7 source files, 30 tests) with no blueprint. This makes it the ideal first exercise of Section 4.4 (Reverse Blueprint Generation).

**Pilot workflow:**

1. **Generate blueprint** — CC reads gaia-audio source and produces candidate blueprint YAML, covering:
   - 12 HTTP endpoints (including WebSocket `/status/ws`)
   - Dependencies: gaia-core (wake signal), gaia-orchestrator (service registration). Note: gaia-web calls gaia-audio (not the reverse) — gaia-web is a consumer, not a dependency
   - Failure modes: GPU OOM, model load failure, muted-state rejection, audio format errors, cloud API fallback
   - Intent: "GAIA's ears and mouth" — sensory perception and voice synthesis
   - Source files: 7 files with role tags

2. **Run Phase 0 infrastructure** — Use gaia-audio as the test target:
   - AST summarizer → summarize all 7 source files (validates Section 3.1 against real code)
   - Blueprint pre-check → validate code against the new blueprint (validates Section 3.5)
   - Review prompt builder → assemble review prompt (validates Section 3.2)

3. **CC review (cold session)** — Independent CC reviews code against blueprint:
   - Uses reverse review variant (Section 4.4.3): "does this blueprint accurately describe this code?"
   - Produces ReviewResult JSON → training pair (`pair_type: "reverse"`)

4. **Fix discrepancies** — For reverse reviews, fix the blueprint (not the code — the code is working)

5. **Promote** — Standard `promote_pipeline.sh` for gaia-audio

**Special considerations for gaia-audio:**
- First service using both `http_rest` AND `websocket` interface types → validates pre-check handling of WebSocket endpoints
- GPU resource management → exercises failure mode coverage dimension beyond typical REST services
- Half-duplex model swapping → complex state machine that tests intent coherence dimension

**Training pair output:** 1 reverse pair (per-service) + 7 reverse pairs (per-file) = **~8 training pairs**

#### 10.1.2 Pilot Part 2: gaia-web GUI Features (GAIA-Guided Forward Generation)

With the gaia-audio blueprint now live, the next step is a **forward generation exercise** — adding audio service awareness to the web dashboard. This is where GAIA participates directly, making it a true test of the bootstrap plan's training loop.

**Features to implement (scoped precisely):**

**Feature Group A: Audio Service Awareness**

| Feature | Backend Change | Frontend Change |
|---------|---------------|-----------------|
| gaia-audio in service registry | Add entry to `_SERVICE_REGISTRY` in `main.py` | Add `'gaia-audio': 'Audio'` to `SERVICE_LABELS` |
| Audio proxy routes | Add `GET /api/hooks/audio/status`, `POST /api/hooks/audio/mute`, `POST /api/hooks/audio/unmute` to `hooks.py` | Update `audioWidget()` URLs to use proxy routes |
| TTS/STT model status | (Data already exposed by gaia-audio `/status`) | Add `sttModel`, `ttsModel` fields to `audioWidget()` + HTML display |
| Audio sleep/wake buttons | Add `POST /api/hooks/audio/sleep`, `POST /api/hooks/audio/wake` to `hooks.py` | Add Audio Control group to `hooksPanel()` with sleep/wake/mute/unmute buttons |

**Feature Group B: Sleep Task Log Viewer**

| Feature | Backend Change | Frontend Change |
|---------|---------------|-----------------|
| Task scheduler status endpoint | Add `GET /sleep/tasks` to `sleep_endpoints.py` — returns `SleepTaskScheduler.get_status()` (run counts, last run, errors per task) | — |
| Task execution history endpoint | Add `GET /sleep/task-history?limit=50` to `sleep_endpoints.py` — queries TimelineStore for `task_exec` events | — |
| Proxy routes for task data | Add `GET /api/hooks/sleep/tasks`, `GET /api/hooks/sleep/task-history` to `hooks.py` | — |
| Sleep Task Log widget | — | New component in Commands view showing task list with priority, run count, last execution time, duration, and pass/fail status |

**Why this belongs in the pilot:** Phase 5 (Section 9) adds a `code_review` sleep task. Without visibility into sleep task execution from the dashboard, there's no way to observe whether the bootstrap plan's autonomous review cycle is running, succeeding, or failing. The log viewer is operational infrastructure for the plan itself.

**Data sources (already exist, just not surfaced):**
- `SleepTaskScheduler.get_status()` — per-task metadata: `task_id`, `priority`, `run_count`, `last_run`, `last_error`
- TimelineStore `task_exec` events — per-execution records: `task_id`, `duration_s`, `success`, `error` (written to `/shared/timeline/gaia_timeline_*.jsonl`)

**GAIA's participation model:**

The full feature set exceeds gaia-prime's context window. But individual tasks are small enough:

1. **Service registry addition** (~5 lines of context) — GAIA can generate this
2. **Single proxy endpoint** (~15 lines of code + 20 lines of context) — GAIA can generate each endpoint individually
3. **Alpine.js field additions** (~10 lines) — GAIA can generate with the existing widget as exemplar
4. **HTML template additions** (~20 lines) — GAIA can generate with existing hooks panel as exemplar
5. **Sleep task status endpoint** (~20 lines of code) — GAIA can generate given the existing sleep endpoint file as exemplar
6. **Sleep task log widget** (~30 lines of JS) — GAIA can generate with the hooks panel as exemplar

For each sub-task:
- CC provides the scoped blueprint section as context
- GAIA generates the code
- CC reviews the output (cold session) → produces a forward training pair
- If GAIA's output needs correction, the correction itself becomes high-value training signal

**Training pair output:** ~6-10 forward pairs (one per sub-task that GAIA attempts)

#### 10.1.3 Pilot Training Pair Summary

| Source | Type | Estimated Count |
|--------|------|----------------|
| gaia-audio blueprint (per-service + per-file) | reverse | ~8 |
| gaia-web GUI features — audio awareness (per sub-task) | forward | ~4-6 |
| gaia-web GUI features — sleep task log (per sub-task) | forward | ~2-4 |
| **Pilot total** | | **~14-18** |

These are high-quality pairs: the reverse pairs come from a real working service, and the forward pairs capture GAIA's actual generation patterns (including mistakes). Combined with retroactive reviews (~54-78) and journal extraction (~40-70), the pilot alone doesn't need to hit the 50-pair threshold — it's one of three parallel corpus sources.

#### 10.1.4 Pilot Success Criteria

The pilot is successful if:
1. gaia-audio has a live blueprint that passes pre-check validation with ≥ 85% structural completeness
2. gaia-audio is promoted to production via the standard pipeline
3. gaia-web displays gaia-audio in the service health grid with correct status
4. Audio model status (STT/TTS model names) is visible in the dashboard
5. Sleep/wake/mute/unmute buttons for gaia-audio work from the hooks panel
6. Sleep task log viewer shows task run history in the Commands view
7. At least 12 training pairs are generated and stored in `knowledge/curricula/code-architect/`
8. GAIA participated in at least 4 generation sub-tasks (regardless of output quality)

---

## 11. Files to Create / Modify Summary

### New Files

| File | Purpose | Phase |
|------|---------|-------|
| `gaia-common/gaia_common/utils/ast_summarizer.py` | AST-based source file summarizer (extends code_analyzer) | 0 |
| `gaia-common/gaia_common/utils/blueprint_precheck.py` | Refactored mechanical pre-check validator | 0 |
| `gaia-common/gaia_common/utils/review_prompt_builder.py` | Prompt construction + ReviewResult schema | 0 |
| `gaia-common/tests/utils/test_ast_summarizer.py` | AST summarizer tests | 0 |
| `gaia-common/tests/utils/test_blueprint_precheck.py` | Pre-check validator tests | 0 |
| `gaia-common/tests/utils/test_review_prompt_builder.py` | Review builder tests | 0 |
| `scripts/format_candidate.sh` | ruff format wrapper | 0 |
| `scripts/check_type_ignores.py` | Per-file type-ignore count diffing | 0 |
| `scripts/run_blueprint_precheck.py` | CLI wrapper for mechanical pre-check | 2 |
| `scripts/cc_review_candidate.sh` | CC review invocation helper (+ --live) | 2 |
| `scripts/generate_ast_summaries.py` | CLI wrapper for batch AST summarization | 2 |
| `scripts/build_review_prompt.py` | CLI wrapper for prompt construction | 2 |
| `scripts/review_templates/cc_review_prompt.txt` | Review prompt template | 2 |
| `scripts/generate_retroactive_corpus.sh` | Retroactive corpus bootstrapping | 2.5 |
| `scripts/extract_journal_corpus.py` | Dev journal training data extraction | 2.5 |
| `scripts/review_templates/cc_reverse_review_prompt.txt` | Reverse review prompt variant (code → blueprint) | 1 |
| `knowledge/blueprints/candidates/gaia-audio.yaml` | gaia-audio blueprint (reverse-generated) | Pilot |
| `candidates/gaia-study/scripts/validators/blueprint_fidelity.py` | BlueprintFidelityValidator (standalone module) | 4 |
| `knowledge/curricula/code-architect/curriculum.json` | Adapter training spec (versioned) | 3 |
| `knowledge/curricula/code-architect/generate_pairs.py` | Training pair generator (all types) | 3 |

### Modified Files

| File | Change | Phase |
|------|--------|-------|
| `scripts/promote_pipeline.sh` | Wire in format_candidate.sh before Stage 3 | 0 |
| `scripts/promote_candidate.sh` | mypy warn→error + type-ignore check | 0 |
| `gaia-common/gaia_common/utils/code_analyzer/structure_extractor.py` | Add signature/decorator/enum extraction | 0 |
| `candidates/gaia-core/gaia_core/cognition/sleep_task_scheduler.py` | Refactor `_run_blueprint_validation` to call `blueprint_precheck.run_blueprint_precheck()` | 0 |
| `candidates/gaia-study/scripts/validate_adapter.py` | Add `--validator` routing to dispatch json_schema vs blueprint_fidelity | 4 |
| `candidates/gaia-core/gaia_core/cognition/sleep_task_scheduler.py` | Add code_review task (uses pre-check + code-architect adapter) | 5 |
| `knowledge/blueprints/QLORA_SELF_STUDY.md` | Add code-architect adapter section | 3 |
| `candidates/gaia-web/gaia_web/main.py` | Add gaia-audio to `_SERVICE_REGISTRY` | Pilot |
| `candidates/gaia-web/gaia_web/routes/hooks.py` | Add audio proxy routes (status, mute, unmute, sleep, wake) + sleep task proxy routes (tasks, task-history) | Pilot |
| `candidates/gaia-web/static/app.js` | Add `SERVICE_LABELS` entry, update `audioWidget()` model display, add `hooksPanel()` audio controls + sleep task log widget | Pilot |
| `candidates/gaia-web/static/index.html` | Add model status row to audio widget, add Audio Control hook group, add Sleep Task Log section | Pilot |
| `candidates/gaia-core/gaia_core/api/sleep_endpoints.py` | Add `GET /sleep/tasks` and `GET /sleep/task-history` endpoints | Pilot |

---

## 12. Key Design Principles (Do Not Violate)

These are the architectural invariants CC must respect throughout implementation:

1. **CC reviews in a cold context.** Never review code you generated in the same session. The independence is what makes the review meaningful.

2. **Blueprints are the spec, not the code** (with one exception). In forward reviews, the review flows from blueprint → code. If the code doesn't match the blueprint, fix the code. In reverse reviews (Section 4.4), the code is the source of truth and the blueprint is evaluated against it — if the blueprint doesn't match the code, fix the blueprint. The direction must be explicitly set via `review_direction` and never mixed in a single review.

3. **Mechanical pre-checks for structural completeness, LLM for semantic fidelity.** The refactored blueprint pre-check (Section 3.5) runs on full source, is fast and deterministic, and answers "is it there?" The LLM reviewer receives pre-check results + AST summaries and answers "does it do the right thing?" Never feed full source to the LLM reviewer. The targeted body extractions (Section 3.1.1) provide heuristic evidence; the pre-check provides ground truth for structural presence. Together they give the LLM reviewer near-complete coverage without context budget explosion.

4. **gaia-study is the sole writer.** All training data, adapter weights, and vector store writes go through gaia-study. Nothing in this plan bypasses that. CC generates artifacts to `candidates/` and `knowledge/curricula/` but does not invoke training directly.

5. **Human approval gates training.** The corpus size monitor triggers a `prime.md` note, not an automatic training run. The QLoRA cycle requires conscious initiation via `promote_pipeline.sh --qlora --adapter code-architect`.

6. **Graduation is earned, not declared.** The three graduation criteria in Section 8.3 must all be met across two consecutive training cycles before CC review becomes optional. Track the metrics in the dev journal.

7. **Extend, don't duplicate.** New infrastructure should build on existing modules (code_analyzer, validate_adapter.py, promote_pipeline.sh) rather than creating parallel implementations. The AST summarizer extends structure_extractor; the fidelity validator is a new class in validate_adapter.py; the review script integrates with the promotion pipeline.

8. **Chat templates are model-derived.** Never hardcode instruction format tokens (`[INST]`, `<</SYS>>`, etc.). Always derive from the base model's tokenizer configuration. This ensures training pairs are correctly formatted if the base model changes.

---

*End of implementation specification.*
*Storage: `knowledge/Dev_Notebook/2026-02-19_cc_bootstrap_plan.md`*

### Revision Log

| Date | Author | Changes |
|------|--------|---------|
| 2026-02-19 | Seumas & Claude (Sonnet 4.6) | Original specification |
| 2026-02-19 | Claude (Opus 4.6) via CC | Rev 2: Added Phase 2.5 (retroactive corpus bootstrapping), Section 3.1.1 (targeted body extraction), Section 2.4 (code_analyzer reuse), adapter versioning, chat template parameterization, per-file blueprint scoping, gaia-common import exemption, blueprint path fix, fidelity formula normalization note, validate_adapter.py rework clarification, type-ignore diffing script, Design Principle 7 (extend don't duplicate) and 8 (model-derived templates) |
| 2026-02-19 | Claude (Opus 4.6) via CC | Rev 3: Added Section 3.5 (mechanical pre-check validator — refactored from sleep_task_scheduler). Adopted hybrid review architecture: pre-checks provide structural ground truth, LLM assesses semantic fidelity. Updated review prompt (3.2), prompt template (5.2), review script (5.3), session setup (5.1) to incorporate pre-check results. Reframed LLM reviewer role from "find what's missing" to "assess whether what's present implements intent." Updated Design Principle 3. Added blueprint_precheck.py, test_blueprint_precheck.py, run_blueprint_precheck.py to file manifest. |
| 2026-02-19 | Claude (Opus 4.6) via CC | Rev 4: Calibrated pre-check accuracy estimates (failure_mode ~50-60%, honest per-dimension table). Added Section 3.2.1 token budget guard with progressive truncation priority for gaia-prime context limits. Pinned reference implementation selection (Section 7.4.1) to dependency+interface Jaccard overlap. Separated graduation criteria into generation quality vs review quality (Section 8.3) — review quality deferred to Phase 5 maturity. Added 15% minimum forward-pair ratio + 1.5x loss weight for forward pairs in skewed corpora (Section 8.1). Moved BlueprintFidelityValidator to standalone `validators/blueprint_fidelity.py` module. |
| 2026-02-19 | Claude (Opus 4.6) via CC | Rev 5: Added calibration caveat to forward-pair ratio/weight thresholds — explicitly marked as initial estimates requiring post-first-cycle tuning (Section 8.1). Fixed per-file blueprint scoping (Section 6.2) to default to full blueprint for composite files, ambiguous roles, or missing role tags. Added Section 6.3 (retroactive review rejection handling) — defines the distinct result path for live services where rejection surfaces as blueprint open_questions + Review Queue items rather than looping back to generation. |
| 2026-02-19 | Claude (Opus 4.6) via CC | Rev 6: Added Section 4.4 (Reverse Blueprint Generation), Section 6.9 (Dev Journal Corpus Extraction), Section 10.1 (Pilot Project). Updated model reference Nanbeige4-3B → Qwen3-4B-Instruct. Updated Design Principle 2 for reverse reviews. Added `reverse` and `journal` pair types. Updated corpus estimates, implementation order, file manifest. |
| 2026-02-20 | Claude (Opus 4.6) via CC | Rev 7: Scoped journal pairs to patterns A+C only (generation-oriented), deferred B+D to future code-reviewer adapter. Added WebSocket decorator pattern (`@app.websocket()`) to pre-check validator. Fixed gaia-audio dependency direction (gaia-web is consumer, not dependency). Added `review_direction` field to ReviewResult schema. Added Sleep Task Log Viewer to pilot (Feature Group B) — surfaces existing TimelineStore `task_exec` events and `SleepTaskScheduler.get_status()` in dashboard Commands view. Added `sleep_endpoints.py` to modified files. Updated corpus estimates to ~108-166 total (~6-10 forward pairs). Updated pilot success criteria (8 items, ≥12 training pairs, ≥4 GAIA sub-tasks). Section numbering fix: 6.4 reference was cosmetic (confidence filter is inline in 6.1). |
