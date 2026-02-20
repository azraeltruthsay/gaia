#!/usr/bin/env python3
"""
Blueprint Fidelity Validator for code-architect adapter.

Validates generated code against blueprint specifications using five dimensions:

| Dimension                | Weight | Measurement                                      |
|--------------------------|--------|--------------------------------------------------|
| Contract completeness    |   30%  | % of blueprint endpoints present in generated code|
| Dependency correctness   |   25%  | % of API calls to declared dependencies only      |
| Failure mode coverage    |   25%  | % of failure modes with observable handling        |
| Syntactic validity       |   10%  | Does ruff check pass with zero errors?             |
| Type annotation coverage |   10%  | % of public functions with complete type annotations|

Composite score threshold for promotion: 0.75

Usage (standalone):
    python blueprint_fidelity.py \
        --adapter code-architect \
        --validation-file knowledge/curricula/code-architect/validation.jsonl \
        --blueprints /knowledge/blueprints \
        [--endpoint http://localhost:7777] \
        [--threshold 0.75] \
        [--dry-run]

Usage (via validate_adapter.py router):
    python validate_adapter.py \
        --adapter code-architect \
        --validator blueprint_fidelity \
        --validation-file knowledge/curricula/code-architect/validation.jsonl \
        --blueprints /knowledge/blueprints
"""
from __future__ import annotations

import argparse
import ast
import json
import re
import subprocess
import sys
import tempfile
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple


# ── Result data structures ───────────────────────────────────────────────────

@dataclass
class DimensionScore:
    """Score for a single validation dimension."""
    name: str
    weight: float
    score: float  # 0.0 to 1.0
    detail: str = ""


@dataclass
class FidelityResult:
    """Result of validating one generated service against its blueprint."""
    index: int
    service_id: str
    dimensions: List[DimensionScore] = field(default_factory=list)
    raw_output: str = ""
    latency_ms: float = 0.0
    error: str = ""

    @property
    def composite_score(self) -> float:
        if not self.dimensions:
            return 0.0
        return sum(d.weight * d.score for d in self.dimensions)

    def to_dict(self) -> dict:
        return {
            "index": self.index,
            "service_id": self.service_id,
            "composite_score": round(self.composite_score, 4),
            "dimensions": {d.name: {"score": round(d.score, 4), "detail": d.detail} for d in self.dimensions},
            "error": self.error,
        }


@dataclass
class FidelityReport:
    """Aggregate validation report for blueprint fidelity."""
    adapter_name: str
    total_examples: int = 0
    results: List[FidelityResult] = field(default_factory=list)
    duration_seconds: float = 0.0

    @property
    def avg_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.composite_score for r in self.results) / len(self.results)

    @property
    def dimension_averages(self) -> Dict[str, float]:
        if not self.results:
            return {}
        dim_scores: Dict[str, List[float]] = {}
        for r in self.results:
            for d in r.dimensions:
                dim_scores.setdefault(d.name, []).append(d.score)
        return {name: sum(scores) / len(scores) for name, scores in dim_scores.items()}


# ── Scoring functions ────────────────────────────────────────────────────────

def _parse_generated_files(raw_output: str) -> Dict[str, str]:
    """
    Parse model output into filename -> source code mapping.

    Expected format:
        ## FILE: service_id/filename.py
        <source code>
    """
    files: dict[str, str] = {}
    current_file: Optional[str] = None
    current_lines: list[str] = []

    for line in raw_output.split("\n"):
        match = re.match(r"^## FILE:\s*(.+)$", line)
        if match:
            if current_file is not None:
                files[current_file] = "\n".join(current_lines)
            current_file = match.group(1).strip()
            # Normalize: extract just the filename part
            if "/" in current_file:
                current_file = current_file.split("/", 1)[1]
            current_lines = []
        elif current_file is not None:
            current_lines.append(line)

    if current_file is not None:
        files[current_file] = "\n".join(current_lines)

    return files


def _extract_endpoints_from_code(source: str) -> List[str]:
    """
    Extract endpoint paths from Python source code via regex.

    Looks for @router.{method}("path") and @app.{method}("path") patterns.
    """
    endpoints: list[str] = []
    pattern = re.compile(
        r'@(?:router|app)\.(get|post|put|delete|patch|websocket)\s*\(\s*["\']([^"\']+)["\']',
        re.IGNORECASE,
    )
    for match in pattern.finditer(source):
        method = match.group(1).upper()
        path = match.group(2)
        endpoints.append(f"{method} {path}")
    return endpoints


def _extract_blueprint_endpoints(instruction: str) -> List[str]:
    """
    Extract expected endpoints from the blueprint YAML in the instruction.

    Looks for interface entries with http_rest transport type.
    """
    endpoints: list[str] = []
    # Look for patterns like: http_rest GET /path or method: GET\n  path: /path
    # Simple heuristic: find all "method: X" + "path: /Y" pairs
    method_pattern = re.compile(r"method:\s*(GET|POST|PUT|DELETE|PATCH)", re.IGNORECASE)
    path_pattern = re.compile(r"path:\s*['\"]?(/[^\s'\"]+)")

    lines = instruction.split("\n")
    for i, line in enumerate(lines):
        method_match = method_pattern.search(line)
        if method_match:
            method = method_match.group(1).upper()
            # Look for path in surrounding lines
            for j in range(max(0, i - 3), min(len(lines), i + 4)):
                path_match = path_pattern.search(lines[j])
                if path_match:
                    endpoints.append(f"{method} {path_match.group(1)}")
                    break

    return endpoints


def score_contract_completeness(instruction: str, generated_files: Dict[str, str]) -> DimensionScore:
    """
    Dimension 1: Contract completeness (30%).
    % of blueprint endpoints present in generated code.
    """
    expected = _extract_blueprint_endpoints(instruction)
    if not expected:
        return DimensionScore(
            name="contract_completeness", weight=0.30, score=1.0,
            detail="No endpoints declared in blueprint"
        )

    all_code = "\n".join(generated_files.values())
    found = _extract_endpoints_from_code(all_code)
    found_paths = {e.split(" ", 1)[1] if " " in e else e for e in found}

    matched = 0
    for ep in expected:
        ep_path = ep.split(" ", 1)[1] if " " in ep else ep
        if ep_path in found_paths:
            matched += 1

    score = matched / len(expected) if expected else 1.0
    return DimensionScore(
        name="contract_completeness", weight=0.30, score=score,
        detail=f"{matched}/{len(expected)} endpoints found"
    )


def _extract_blueprint_dependencies(instruction: str) -> List[str]:
    """Extract declared dependency service IDs from the blueprint in the instruction."""
    deps: list[str] = []
    # Look for service dependency entries: "id: gaia-xxx"
    in_deps_section = False
    for line in instruction.split("\n"):
        if "dependencies:" in line.lower() or "services:" in line.lower():
            in_deps_section = True
            continue
        if in_deps_section:
            match = re.match(r"\s*-?\s*id:\s*['\"]?(gaia-\w+)", line)
            if match:
                deps.append(match.group(1))
            elif line.strip() and not line.strip().startswith(("-", "id:", "role:", "required:", "fallback:")):
                in_deps_section = False
    return deps


def score_dependency_correctness(instruction: str, generated_files: Dict[str, str]) -> DimensionScore:
    """
    Dimension 2: Dependency correctness (25%).
    % of API calls to declared dependencies only.
    """
    declared = set(_extract_blueprint_dependencies(instruction))
    # gaia-common is always allowed
    declared.add("gaia-common")

    all_code = "\n".join(generated_files.values())

    # Find all gaia service imports
    import_pattern = re.compile(r"from\s+(gaia[_-]\w+)")
    called_services: set[str] = set()
    for match in import_pattern.finditer(all_code):
        svc = match.group(1).replace("_", "-")
        called_services.add(svc)

    if not called_services:
        return DimensionScore(
            name="dependency_correctness", weight=0.25, score=1.0,
            detail="No service imports detected"
        )

    undeclared = called_services - declared
    if undeclared:
        score = 1.0 - len(undeclared) / len(called_services)
        return DimensionScore(
            name="dependency_correctness", weight=0.25, score=max(0.0, score),
            detail=f"Undeclared: {sorted(undeclared)}"
        )

    return DimensionScore(
        name="dependency_correctness", weight=0.25, score=1.0,
        detail=f"All {len(called_services)} imports declared"
    )


def _extract_blueprint_failure_modes(instruction: str) -> List[str]:
    """Extract failure mode conditions from the blueprint in the instruction."""
    modes: list[str] = []
    # Look for failure_modes section with condition entries
    in_fm_section = False
    for line in instruction.split("\n"):
        if "failure_modes:" in line.lower():
            in_fm_section = True
            continue
        if in_fm_section:
            match = re.match(r"\s*-?\s*condition:\s*(.+)", line)
            if match:
                modes.append(match.group(1).strip().strip("'\""))
            elif line.strip() and not line.strip().startswith(("-", "condition:", "response:", "severity:")):
                if not any(kw in line for kw in ("response:", "severity:", "condition:")):
                    in_fm_section = False
    return modes


def score_failure_mode_coverage(instruction: str, generated_files: Dict[str, str]) -> DimensionScore:
    """
    Dimension 3: Failure mode coverage (25%).
    % of failure modes with observable handling (try/except or status code returns).
    """
    expected_modes = _extract_blueprint_failure_modes(instruction)
    if not expected_modes:
        return DimensionScore(
            name="failure_mode_coverage", weight=0.25, score=1.0,
            detail="No failure modes declared in blueprint"
        )

    all_code = "\n".join(generated_files.values())

    # Count try/except blocks and error-returning patterns
    except_pattern = re.compile(r"except\s+(\w+)")
    status_pattern = re.compile(r"status_code\s*=\s*(\d+)")

    handlers = set()
    for match in except_pattern.finditer(all_code):
        handlers.add(match.group(1).lower())
    for match in status_pattern.finditer(all_code):
        code = int(match.group(1))
        if code >= 400:
            handlers.add(f"http_{code}")

    # Heuristic match: check if each failure mode has a plausible handler
    covered = 0
    for mode in expected_modes:
        mode_lower = mode.lower()
        # Check for exception handler keywords overlapping with mode description
        if any(h in mode_lower or mode_lower in h for h in handlers):
            covered += 1
        elif "timeout" in mode_lower and ("timeout" in all_code.lower()):
            covered += 1
        elif "error" in mode_lower and ("except" in all_code):
            covered += 1
        elif "unavailable" in mode_lower and ("connectionerror" in all_code.lower() or "503" in all_code):
            covered += 1

    score = covered / len(expected_modes)
    return DimensionScore(
        name="failure_mode_coverage", weight=0.25, score=score,
        detail=f"{covered}/{len(expected_modes)} modes covered"
    )


def score_syntactic_validity(generated_files: Dict[str, str]) -> DimensionScore:
    """
    Dimension 4: Syntactic validity (10%).
    Does the code parse without errors? Uses ast.parse and optionally ruff.
    """
    total_files = len(generated_files)
    if total_files == 0:
        return DimensionScore(
            name="syntactic_validity", weight=0.10, score=0.0,
            detail="No files generated"
        )

    valid_files = 0
    errors: list[str] = []

    for filename, source in generated_files.items():
        try:
            ast.parse(source)
            valid_files += 1
        except SyntaxError as e:
            errors.append(f"{filename}:{e.lineno}: {e.msg}")

    # Try ruff if available
    ruff_score = 1.0
    try:
        with tempfile.TemporaryDirectory() as tmpdir:
            for filename, source in generated_files.items():
                fpath = Path(tmpdir) / filename
                fpath.parent.mkdir(parents=True, exist_ok=True)
                fpath.write_text(source, encoding="utf-8")

            result = subprocess.run(
                ["ruff", "check", tmpdir, "--select=E,F", "--quiet"],
                capture_output=True, text=True, timeout=30,
            )
            if result.returncode != 0:
                error_count = len(result.stdout.strip().split("\n")) if result.stdout.strip() else 0
                # Deduct per error, floor at 0
                ruff_score = max(0.0, 1.0 - error_count * 0.1)
    except (FileNotFoundError, subprocess.TimeoutExpired):
        pass  # ruff not available, skip

    ast_score = valid_files / total_files
    # Composite: 70% AST parse + 30% ruff
    combined = 0.7 * ast_score + 0.3 * ruff_score

    detail = f"{valid_files}/{total_files} files parse"
    if errors:
        detail += f"; errors: {errors[:3]}"

    return DimensionScore(
        name="syntactic_validity", weight=0.10, score=combined,
        detail=detail,
    )


def score_type_annotation_coverage(generated_files: Dict[str, str]) -> DimensionScore:
    """
    Dimension 5: Type annotation coverage (10%).
    % of public functions with complete type annotations.
    """
    total_public = 0
    annotated = 0

    for _filename, source in generated_files.items():
        try:
            tree = ast.parse(source)
        except SyntaxError:
            continue

        for node in ast.walk(tree):
            if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                # Skip private/dunder methods except __init__
                if node.name.startswith("_") and node.name != "__init__":
                    continue

                total_public += 1

                # Check: all params have annotations + return type
                has_return = node.returns is not None
                params_annotated = True
                for arg in node.args.args:
                    if arg.arg == "self" or arg.arg == "cls":
                        continue
                    if arg.annotation is None:
                        params_annotated = False
                        break

                if has_return and params_annotated:
                    annotated += 1

    if total_public == 0:
        return DimensionScore(
            name="type_annotation_coverage", weight=0.10, score=1.0,
            detail="No public functions"
        )

    score = annotated / total_public
    return DimensionScore(
        name="type_annotation_coverage", weight=0.10, score=score,
        detail=f"{annotated}/{total_public} functions fully annotated"
    )


# ── Main validation logic ────────────────────────────────────────────────────

def validate_one_fidelity(
    index: int,
    example: Dict,
    endpoint: Optional[str],
    model: Optional[str],
    dry_run: bool,
) -> FidelityResult:
    """Validate a single code-architect example."""
    service_id = example.get("service_id", "unknown")
    instruction = example.get("instruction", "")
    expected_output = example.get("output", "")

    result = FidelityResult(index=index, service_id=service_id)

    if dry_run:
        # Use expected output as the "generated" code
        raw_output = expected_output
        result.raw_output = raw_output
    else:
        # Call the model
        raw_output, latency = _call_model(endpoint, instruction, model=model)
        result.raw_output = raw_output
        result.latency_ms = latency

        if raw_output.startswith("ERROR:"):
            result.error = raw_output
            return result

    # Parse generated files
    generated_files = _parse_generated_files(raw_output)
    if not generated_files:
        result.error = "No parseable files in output"
        # Give zero scores
        result.dimensions = [
            DimensionScore("contract_completeness", 0.30, 0.0, "No output"),
            DimensionScore("dependency_correctness", 0.25, 0.0, "No output"),
            DimensionScore("failure_mode_coverage", 0.25, 0.0, "No output"),
            DimensionScore("syntactic_validity", 0.10, 0.0, "No output"),
            DimensionScore("type_annotation_coverage", 0.10, 0.0, "No output"),
        ]
        return result

    # Score each dimension
    result.dimensions = [
        score_contract_completeness(instruction, generated_files),
        score_dependency_correctness(instruction, generated_files),
        score_failure_mode_coverage(instruction, generated_files),
        score_syntactic_validity(generated_files),
        score_type_annotation_coverage(generated_files),
    ]

    return result


def _call_model(endpoint: str, instruction: str, model: Optional[str] = None,
                timeout: int = 60) -> Tuple[str, float]:
    """Call vLLM endpoint. Returns (raw_output, latency_ms)."""
    try:
        import requests
    except ImportError:
        return "ERROR: requests not available", 0.0

    url = f"{endpoint}/v1/chat/completions"
    payload = {
        "model": model or "default",
        "messages": [
            {"role": "system", "content": "You are GAIA's code-architect. Generate Python source code that faithfully implements blueprints."},
            {"role": "user", "content": instruction},
        ],
        "max_tokens": 4096,
        "temperature": 0.0,
    }

    start = time.monotonic()
    try:
        resp = requests.post(url, json=payload, timeout=timeout)
        latency = (time.monotonic() - start) * 1000
        resp.raise_for_status()
        data = resp.json()
        text = data["choices"][0]["message"]["content"].strip()
        return text, latency
    except Exception as e:
        latency = (time.monotonic() - start) * 1000
        return f"ERROR: {e}", latency


def run_fidelity_validation(
    adapter_name: str,
    validation_file: str,
    endpoint: Optional[str] = None,
    model: Optional[str] = None,
    max_examples: int = 0,
    dry_run: bool = False,
) -> FidelityReport:
    """Run blueprint fidelity validation across all examples."""
    report = FidelityReport(adapter_name=adapter_name)

    val_path = Path(validation_file)
    if not val_path.exists():
        print(f"ERROR: Validation file not found: {val_path}", file=sys.stderr)
        sys.exit(1)

    examples = []
    with open(val_path, "r") as f:
        for line in f:
            line = line.strip()
            if line:
                examples.append(json.loads(line))

    if max_examples > 0:
        examples = examples[:max_examples]

    report.total_examples = len(examples)
    print(f"Validating {len(examples)} examples for blueprint fidelity{'  [dry-run]' if dry_run else ''}...")

    start = time.monotonic()
    for i, example in enumerate(examples):
        result = validate_one_fidelity(i, example, endpoint, model, dry_run)
        report.results.append(result)

        if (i + 1) % 5 == 0 or i == len(examples) - 1:
            print(f"  [{i+1}/{len(examples)}] avg_score={report.avg_score:.3f}")

    report.duration_seconds = time.monotonic() - start
    return report


def print_fidelity_report(report: FidelityReport, verbose: bool = False) -> None:
    """Print formatted fidelity validation report."""
    print(f"\n{'='*60}")
    print(f"  Blueprint Fidelity: {report.adapter_name}")
    print(f"{'='*60}")
    print(f"  Examples:         {report.total_examples}")
    print(f"  Duration:         {report.duration_seconds:.1f}s")
    print(f"  Composite score:  {report.avg_score:.3f}")

    print(f"\n  Per-dimension averages:")
    for name, avg in sorted(report.dimension_averages.items()):
        print(f"    {name:.<35s} {avg:.3f}")

    if verbose:
        print(f"\n  Per-example scores:")
        for r in report.results:
            print(f"    [{r.index}] {r.service_id}: {r.composite_score:.3f}", end="")
            if r.error:
                print(f"  ERROR: {r.error}", end="")
            print()


def generate_fidelity_json_report(report: FidelityReport) -> dict:
    """Generate machine-readable JSON report."""
    return {
        "adapter_name": report.adapter_name,
        "validator": "blueprint_fidelity",
        "total_examples": report.total_examples,
        "duration_seconds": round(report.duration_seconds, 2),
        "composite_score": round(report.avg_score, 4),
        "dimension_averages": {k: round(v, 4) for k, v in report.dimension_averages.items()},
        "results": [r.to_dict() for r in report.results],
    }


# ── CLI ──────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(description="Blueprint fidelity validator for code-architect")
    parser.add_argument("--adapter", default="code-architect", help="Adapter name")
    parser.add_argument("--validation-file", required=True, help="Path to validation JSONL")
    parser.add_argument("--endpoint", default="http://localhost:7777", help="vLLM endpoint")
    parser.add_argument("--threshold", type=float, default=0.75, help="Minimum composite score")
    parser.add_argument("--max-examples", type=int, default=0, help="Limit examples (0=all)")
    parser.add_argument("--dry-run", action="store_true", help="Use expected output instead of model")
    parser.add_argument("--verbose", "-v", action="store_true")
    parser.add_argument("--json-report", type=str, default="", help="Write JSON report to path")
    args = parser.parse_args()

    report = run_fidelity_validation(
        adapter_name=args.adapter,
        validation_file=args.validation_file,
        endpoint=args.endpoint,
        model=args.adapter,
        max_examples=args.max_examples,
        dry_run=args.dry_run,
    )
    print_fidelity_report(report, verbose=args.verbose)

    if args.json_report:
        report_data = generate_fidelity_json_report(report)
        report_path = Path(args.json_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2)
        print(f"\nJSON report written to {report_path}")

    passed = report.avg_score >= args.threshold
    status = "PASS" if passed else "FAIL"
    print(f"\n{'='*60}")
    print(f"  RESULT: {status}  (score={report.avg_score:.3f}, threshold={args.threshold})")
    print(f"{'='*60}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
