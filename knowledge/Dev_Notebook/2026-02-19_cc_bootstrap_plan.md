# GAIA Code Generation Bootstrap Plan
## CC-Assisted Code Quality, Self-Review, and QLoRA Training Loop

**Document status:** Implementation specification (Revision 2)
**Intended audience:** Claude Code (CC) for autonomous implementation
**Date:** 2026-02-19
**Author:** Seumas & Claude (Sonnet 4.6) via GAIA Project Chat
**Revised by:** Claude (Opus 4.6) via Claude Code — incorporated retroactive corpus bootstrapping, AST summary refinements, code_analyzer reuse, and practical implementation fixes

---

## 1. Background & Motivation

### The Problem

GAIA's primary inference model (gaia-prime, currently Nanbeige4-3B) is a capable general reasoner but has no inherent knowledge of the GAIA codebase, service contracts, or architectural idioms. When used to generate code — especially through the planned Builder Panel — it will produce output that is syntactically reasonable but semantically adrift from the project's conventions.

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

These are added as `error_handlers` and `http_calls` keys in the summary dict. They allow the reviewer to assess failure mode coverage and dependency correctness without reading full function bodies.

### 3.2 Blueprint-Anchored Review Prompt Builder

**Location:** `gaia-common/gaia_common/utils/review_prompt_builder.py`

**Purpose:** Given a blueprint YAML and one or more AST summaries, construct a structured review prompt that asks an LLM to identify specific, blueprint-anchored discrepancies.

**Prompt structure:**

```
SYSTEM:
You are a code reviewer for the GAIA AI system. Your task is to verify that
the provided source code faithfully implements its blueprint specification.

You are NOT evaluating general code quality. You are evaluating blueprint
fidelity across five dimensions:

1. CONTRACT FIDELITY — Do the exposed endpoints match the blueprint's
   interfaces section exactly? Check: paths, HTTP methods, request/response
   schema field names.

2. DEPENDENCY CORRECTNESS — Does the code only call services and volumes
   declared in the blueprint's dependencies section? Flag any undeclared
   external calls. Note: gaia-common imports are universally available and
   should NOT be flagged as undeclared dependencies.

3. FAILURE MODE COVERAGE — For each failure mode in the blueprint, is there
   corresponding handling code? Check the error_handlers and http_calls
   sections of the AST summaries for evidence. Flag failure modes with no
   observable implementation.

4. INTENT COHERENCE — Does the code's overall structure reflect the blueprint's
   stated purpose and cognitive_role? Flag obvious divergences.

5. OPEN QUESTIONS — Does the code reveal answers to any open_questions in the
   blueprint? Or does it raise new ones?

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

## Source Files Under Review

{for each file: filename header + ast_summary.to_prompt_text()}

---

## Review Task

Identify all discrepancies between the blueprint specification above and the
source code summaries. Be specific: cite the blueprint claim and the contradicting
(or absent) code evidence.
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
    review_timestamp: datetime
    overall_fidelity_score: float # 0.0–1.0
    discrepancies: List[DiscrepancyItem]
    open_question_updates: List[OpenQuestionUpdate]
    promotion_recommendation: Literal["approve", "approve_with_notes", "reject"]
    summary_note: str             # 2-3 sentence human-readable summary
```

**Testing:** `gaia-common/tests/utils/test_review_prompt_builder.py` — verify prompt renders correctly from a seed blueprint + mock AST summaries; verify ReviewResult parses from a known CC response.

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

---

## 5. Phase 2: CC Review Workflow (Independent Session)

This is the core of the bootstrap mechanism. It must be a genuinely independent CC invocation — not a continuation of the generation session.

### 5.1 Review Session Setup

The review session receives:

1. **The blueprint seed** — the design specification
2. **AST summaries of all generated files** — produced by running `ast_summarizer.py` on the candidate directory (not raw source)
3. **The ReviewResult schema** — so CC knows exactly what to produce
4. **Explicit instruction:** "You did not write this code. Review it against the blueprint without reference to any generative intent."

CC must not receive: the full source files, any generation session context, or any human commentary on the code quality.

### 5.2 Review Session Prompt Template

**Location:** `scripts/review_templates/cc_review_prompt.txt`

```
You are performing a blueprint fidelity review for the GAIA AI system.

You will be given:
1. A blueprint specification (YAML)
2. AST summaries of the candidate source files

Your task is to produce a ReviewResult JSON object identifying all discrepancies
between the blueprint's claims and the code's actual structure.

IMPORTANT:
- You are reviewing for blueprint fidelity ONLY, not general code quality
- Base your assessment exclusively on what the AST summaries show
- If something is absent from the summaries, note it as "not found in summary"
- Your output must be valid JSON matching the ReviewResult schema exactly

ReviewResult schema:
{schema_json}

---

BLUEPRINT:
{blueprint_yaml}

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

# Step 2: Build review prompt
python3 scripts/build_review_prompt.py \
    --blueprint "$BLUEPRINT" \
    --ast-summaries "$REVIEW_DIR/ast_summaries_${TIMESTAMP}.json" \
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

The review prompt builder (Section 3.2) should accept an optional `scope_to_file` parameter that filters the blueprint context accordingly.

### 6.3 Training Signal Characteristics

Retroactive pairs have different training signal from forward pairs:

| | Retroactive Pairs | Forward Pairs |
|---|---|---|
| **Code quality** | Already promoted, high fidelity | Pre-review, may have discrepancies |
| **Training signal** | "What good looks like" | "What mistakes to avoid + how to fix" |
| **CC review tone** | Mostly approvals, minor observations | May include rejections, major issues |
| **Volume** | Bounded by existing services (~50-75) | Unbounded, grows with development |

Both signals are complementary. The training pair schema (Section 7.1) includes a `pair_type` field to distinguish them so their effectiveness can be analyzed separately during validation.

### 6.4 Corpus Size Estimate

| Source | Estimated pairs |
|--------|----------------|
| 6 live services, per-service reviews | 6 |
| ~8-12 source files per service, per-file reviews | 48-72 |
| **Total retroactive** | **~54-78** |

This is likely enough to **hit the 50-pair training threshold** from Phase 0-2 infrastructure alone, unblocking Phase 4 (QLoRA training) without waiting for months of organic forward-pair accumulation.

### 6.5 Retroactive Corpus Generation Script

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

### 6.6 Retroactive Corpus Storage

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

---

## 7. Phase 3: Training Corpus Construction

Every CC review — whether from forward generation (Phase 2) or retroactive bootstrapping (Phase 2.5) — produces a training example. The corpus accumulates from both sources.

### 7.1 Training Pair Schema

Each training pair captures a review cycle:

```json
{
  "pair_id": "uuid",
  "pair_type": "forward | retroactive",
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
    "max_sequence_length": 2048
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
- `chat_template: "auto"` — derive the instruction format from the base model's tokenizer config rather than hardcoding `[INST]`/`<</SYS>>` tags, which are Llama-2-specific and may not match Nanbeige4-3B's expected format.
- `adapter_version` tracks training iterations. When retraining with an expanded corpus, increment the version. The `versioning.retention_count` keeps the last 3 adapter checkpoints for rollback.

### 7.4 Training Pair Generation Script

**Location:** `knowledge/curricula/code-architect/generate_pairs.py`

This script:
1. Reads all `reviews/*.json` and `retroactive/**/cc_review_*.json` files
2. For each review, loads the corresponding blueprint and AST summaries
3. Tags each pair with `pair_type` (forward/retroactive) and `granularity` (service/file)
4. Constructs instruction/output pairs using the base model's chat template (loaded from tokenizer config, NOT hardcoded)
5. Outputs to `train.jsonl` and `validation.jsonl` with stratified split (ensure both pair types are represented in validation)

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
{3 most relevant AST summaries from existing live services}

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

Add corpus size monitoring to the blueprint_validation sleep task in `sleep_task_scheduler.py`. When `len(pairs/*.json) >= minimum_corpus_size` and no `code-architect` adapter exists (or the current adapter version is stale):

1. Log a note to `prime.md`: *"code-architect corpus has reached training threshold ({n} pairs: {r} retroactive, {f} forward). Recommend triggering training via promote_pipeline.sh --qlora --adapter code-architect"*
2. Add a `high_priority` flag to the blueprint's `open_questions`

Training is deliberately not triggered automatically — it requires a conscious human decision given the GPU time involved.

### 8.2 Validation Dimensions

The `validate_adapter.py` script needs a new validation pipeline for code-architect. This is not a simple mode flag — the dimensions require fundamentally different evaluation logic from json-architect's JSON validity scoring. Implement as a new class `BlueprintFidelityValidator` alongside the existing `JSONSchemaValidator`.

Add `--validator blueprint_fidelity` with these dimensions:

| Dimension | Weight | Measurement |
|-----------|--------|-------------|
| Contract completeness | 30% | % of blueprint endpoints present in generated code (via AST extraction) |
| Dependency correctness | 25% | % of API calls to declared dependencies only (via http_calls extraction) |
| Failure mode coverage | 25% | % of failure modes with observable handling (via error_handlers extraction) |
| Syntactic validity | 10% | Does `ruff check` pass with zero errors? (shell out to ruff) |
| Type annotation coverage | 10% | % of public functions with complete type annotations (via AST) |

**Composite score threshold for promotion:** 0.75 (higher than json-architect's 0.6 because code errors have higher blast radius than JSON formatting errors).

### 8.3 Graduation Criteria

The goal is for GAIA's `code-architect` adapter to match CC's review quality on the project's own codebase. Graduation — meaning CC steps back from mandatory review — is reached when:

1. **Divergence score convergence:** Average divergence score of last 10 code-architect generations ≤ 0.15 (vs. CC-reviewed baseline of ≈ 0.12)
2. **First-pass promotion rate:** ≥ 80% of generated candidates pass the promotion pipeline without modification
3. **CC spot-check agreement:** On a random sample of 5 generations, CC review agrees with code-architect's self-assessment in ≥ 90% of discrepancy classifications

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
├── 3.1  AST Summarizer (extend code_analyzer)  ← implement first, no deps
├── 3.2  Review Prompt Builder + schema          ← depends on 3.1
├── 3.3  ruff format hook                        ← no dependencies, trivial
└── 3.4  mypy promotion to error + ignore check  ← no dependencies, small script

Phase 1 (CC Generation Workflow)
├── 4.x  Generation rubric + context             ← depends on 3.1, 3.2
└── 4.3  Candidate directory structure            ← depends on generation rubric

Phase 2 (CC Review Workflow)
├── 5.2  Review prompt template                   ← depends on 3.2
├── 5.3  cc_review_candidate.sh (+ --live flag)   ← depends on 3.1, 5.2
└── 5.4  Result handling + routing                ← depends on 5.3

Phase 2.5 (Retroactive Corpus Bootstrapping)     ← NEW
├── 6.1  generate_retroactive_corpus.sh           ← depends on Phase 0 + 2
├── 6.2  Per-file blueprint scoping logic         ← depends on 3.2
├── 6.3  CC reviews of all live services (batch)  ← depends on 6.1
├── 6.4  Quality filtering (confidence ≥ 0.6)     ← depends on 6.3
└── 6.5  If corpus ≥ 50: Phase 4 unblocked        ← depends on 6.4

Phase 3 (Forward Corpus Construction)
├── 7.1  Training pair schema (+ pair_type tag)   ← depends on 5.4
├── 7.3  Curriculum spec (+ versioning, auto template) ← depends on 7.1
└── 7.4  generate_pairs.py (both pair types)      ← depends on 7.1, 7.3

Phase 4 (Training & Validation)
├── 8.1  Corpus size monitoring                   ← depends on Phase 3
├── 8.2  BlueprintFidelityValidator class          ← depends on 7.3
└── 8.3  Graduation tracking                      ← depends on 8.2

Phase 5 (Sleep Cycle Integration)
├── 9.1  code_review sleep task                   ← depends on Phase 4 adapter
└── 9.2  Review Queue integration                 ← depends on Blueprint System Phase 2
```

**Critical path acceleration:** Phases 0 → 2 → 2.5 can be executed as a focused sprint. If retroactive bootstrapping yields ≥ 50 pairs, Phase 4 is immediately unblocked without waiting for forward pair accumulation. Forward pairs (Phase 3) continue accumulating in parallel and are folded into subsequent training cycles.

---

## 11. Files to Create / Modify Summary

### New Files

| File | Purpose | Phase |
|------|---------|-------|
| `gaia-common/gaia_common/utils/ast_summarizer.py` | AST-based source file summarizer (extends code_analyzer) | 0 |
| `gaia-common/gaia_common/utils/review_prompt_builder.py` | Prompt construction + ReviewResult schema | 0 |
| `gaia-common/tests/utils/test_ast_summarizer.py` | AST summarizer tests | 0 |
| `gaia-common/tests/utils/test_review_prompt_builder.py` | Review builder tests | 0 |
| `scripts/format_candidate.sh` | ruff format wrapper | 0 |
| `scripts/check_type_ignores.py` | Per-file type-ignore count diffing | 0 |
| `scripts/cc_review_candidate.sh` | CC review invocation helper (+ --live) | 2 |
| `scripts/generate_ast_summaries.py` | CLI wrapper for batch AST summarization | 2 |
| `scripts/build_review_prompt.py` | CLI wrapper for prompt construction | 2 |
| `scripts/review_templates/cc_review_prompt.txt` | Review prompt template | 2 |
| `scripts/generate_retroactive_corpus.sh` | Retroactive corpus bootstrapping | 2.5 |
| `knowledge/curricula/code-architect/curriculum.json` | Adapter training spec (versioned) | 3 |
| `knowledge/curricula/code-architect/generate_pairs.py` | Training pair generator (both types) | 3 |

### Modified Files

| File | Change | Phase |
|------|--------|-------|
| `scripts/promote_pipeline.sh` | Wire in format_candidate.sh before Stage 3 | 0 |
| `scripts/promote_candidate.sh` | mypy warn→error + type-ignore check | 0 |
| `gaia-common/gaia_common/utils/code_analyzer/structure_extractor.py` | Add signature/decorator/enum extraction | 0 |
| `candidates/gaia-study/scripts/validate_adapter.py` | Add BlueprintFidelityValidator class | 4 |
| `candidates/gaia-core/gaia_core/cognition/sleep_task_scheduler.py` | Add code_review task | 5 |
| `knowledge/blueprints/QLORA_SELF_STUDY.md` | Add code-architect adapter section | 3 |

---

## 12. Key Design Principles (Do Not Violate)

These are the architectural invariants CC must respect throughout implementation:

1. **CC reviews in a cold context.** Never review code you generated in the same session. The independence is what makes the review meaningful.

2. **Blueprints are the spec, not the code.** The review always flows from blueprint → code, never from code → inferred intent. If the blueprint is wrong, fix the blueprint; don't bend the review.

3. **AST summaries for LLM context, source for mechanical validation.** The regex/AST extraction in `sleep_task_scheduler.py` works on full source and is fast. LLM review works on summaries and is expensive. Never feed full source to the LLM reviewer. The targeted body extractions (Section 3.1.1) are a controlled exception — they extract structured signals, not raw bodies.

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
