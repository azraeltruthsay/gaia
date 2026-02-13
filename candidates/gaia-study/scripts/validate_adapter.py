#!/usr/bin/env python3
"""
QLoRA Adapter Validator for GAIA

Validates a trained adapter by running held-out validation examples through
it and scoring the outputs against expected results.

Scoring dimensions:
  1. JSON validity  — Is the output parseable JSON?
  2. Schema match   — Does the JSON have required keys?
  3. Value accuracy  — Are key field values correct (selected_tool, params, etc.)?
  4. Confidence cal. — Is the confidence score in a reasonable range?

Usage:
    python validate_adapter.py \
        --adapter json-architect \
        --validation-file knowledge/curricula/json-architect/validation.jsonl \
        --endpoint http://localhost:7777 \
        [--baseline]          # Also score the base model for comparison
        [--max-examples 50]   # Cap validation examples
        [--threshold 0.6]     # Minimum pass score (0.0-1.0)
        [--dry-run]           # Parse + report without calling model
"""

import argparse
import json
import sys
import time
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional


@dataclass
class ExampleResult:
    """Result of validating one example."""
    index: int
    category: str  # tool_selection, tool_review, confidence_assessment, null_selection
    json_valid: bool = False
    schema_match: bool = False
    value_accuracy: float = 0.0
    expected: Optional[Dict] = None
    actual: Optional[Dict] = None
    raw_output: str = ""
    latency_ms: float = 0.0
    error: str = ""

    @property
    def score(self) -> float:
        """Composite score: 40% json_valid, 30% schema_match, 30% value_accuracy."""
        return (
            (0.4 if self.json_valid else 0.0)
            + (0.3 if self.schema_match else 0.0)
            + 0.3 * self.value_accuracy
        )


@dataclass
class ValidationReport:
    """Aggregate validation report."""
    adapter_name: str
    total_examples: int = 0
    results: List[ExampleResult] = field(default_factory=list)
    duration_seconds: float = 0.0
    is_baseline: bool = False

    @property
    def json_valid_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.json_valid) / len(self.results)

    @property
    def schema_match_rate(self) -> float:
        if not self.results:
            return 0.0
        return sum(1 for r in self.results if r.schema_match) / len(self.results)

    @property
    def avg_value_accuracy(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.value_accuracy for r in self.results) / len(self.results)

    @property
    def avg_score(self) -> float:
        if not self.results:
            return 0.0
        return sum(r.score for r in self.results) / len(self.results)

    @property
    def category_scores(self) -> Dict[str, float]:
        categories: Dict[str, List[float]] = {}
        for r in self.results:
            categories.setdefault(r.category, []).append(r.score)
        return {cat: sum(scores) / len(scores) for cat, scores in categories.items()}


def detect_category(instruction: str) -> str:
    """Detect the category of a validation example from its instruction text."""
    lower = instruction.lower()
    if "confidence assessor" in lower or "assess your confidence" in lower:
        return "confidence_assessment"
    if "review this tool selection" in lower or "careful reviewer" in lower:
        return "tool_review"
    if '"selected_tool": null' in instruction or '"selected_tool":null' in instruction:
        return "null_selection"
    return "tool_selection"


def get_required_keys(category: str) -> List[str]:
    """Get the required JSON keys for a given category."""
    if category == "tool_selection":
        return ["selected_tool", "reasoning", "confidence"]
    elif category == "tool_review":
        return ["approved", "confidence", "reasoning"]
    elif category == "confidence_assessment":
        return ["confidence", "reasoning"]
    elif category == "null_selection":
        return ["selected_tool", "reasoning", "confidence"]
    return []


def score_value_accuracy(expected: Dict, actual: Dict, category: str) -> float:
    """Score value-level accuracy between expected and actual outputs."""
    if not expected or not actual:
        return 0.0

    checks = []

    if category in ("tool_selection", "null_selection"):
        # Check selected_tool
        exp_tool = expected.get("selected_tool")
        act_tool = actual.get("selected_tool")
        checks.append(1.0 if exp_tool == act_tool else 0.0)

        # Check params (if tool selected)
        if exp_tool is not None:
            exp_params = expected.get("params", {})
            act_params = actual.get("params", {})
            if exp_params and act_params:
                # Score by key overlap
                exp_keys = set(exp_params.keys())
                act_keys = set(act_params.keys())
                if exp_keys:
                    key_overlap = len(exp_keys & act_keys) / len(exp_keys)
                    checks.append(key_overlap)
            elif not exp_params and not act_params:
                checks.append(1.0)

        # Confidence should be in reasonable range
        exp_conf = expected.get("confidence", 0)
        act_conf = actual.get("confidence", 0)
        if isinstance(act_conf, (int, float)):
            # Within 0.3 of expected = full credit, linear decay beyond
            conf_diff = abs(float(exp_conf) - float(act_conf))
            checks.append(max(0.0, 1.0 - conf_diff / 0.5))

    elif category == "tool_review":
        # Check approved boolean
        checks.append(1.0 if expected.get("approved") == actual.get("approved") else 0.0)
        # Confidence within range
        exp_conf = expected.get("confidence", 0)
        act_conf = actual.get("confidence", 0)
        if isinstance(act_conf, (int, float)):
            conf_diff = abs(float(exp_conf) - float(act_conf))
            checks.append(max(0.0, 1.0 - conf_diff / 0.5))

    elif category == "confidence_assessment":
        exp_conf = expected.get("confidence", 0)
        act_conf = actual.get("confidence", 0)
        if isinstance(act_conf, (int, float)):
            conf_diff = abs(float(exp_conf) - float(act_conf))
            checks.append(max(0.0, 1.0 - conf_diff / 0.3))

    return sum(checks) / max(len(checks), 1)


def call_model(endpoint: str, instruction: str, model: Optional[str] = None,
               timeout: int = 30) -> tuple:
    """
    Call the vLLM or gaia-core endpoint with a prompt.

    Returns (raw_output_text, latency_ms).
    """
    import requests

    # vLLM OpenAI-compatible chat completions
    url = f"{endpoint}/v1/chat/completions"

    payload = {
        "model": model or "default",
        "messages": [{"role": "user", "content": instruction}],
        "max_tokens": 512,
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
    except requests.exceptions.ConnectionError:
        latency = (time.monotonic() - start) * 1000
        return "", latency
    except Exception as e:
        latency = (time.monotonic() - start) * 1000
        return f"ERROR: {e}", latency


def validate_one(index: int, example: Dict, endpoint: Optional[str],
                 model: Optional[str], dry_run: bool) -> ExampleResult:
    """Validate a single example."""
    instruction = example.get("instruction", "")
    expected_raw = example.get("output", "{}")
    category = detect_category(instruction)

    result = ExampleResult(index=index, category=category)

    # Parse expected output
    try:
        result.expected = json.loads(expected_raw)
    except json.JSONDecodeError:
        result.error = "Could not parse expected output as JSON"
        return result

    if dry_run:
        # In dry-run mode, simulate perfect results
        result.json_valid = True
        result.schema_match = True
        result.value_accuracy = 1.0
        result.raw_output = expected_raw
        result.actual = result.expected
        return result

    # Call the model
    raw_output, latency = call_model(endpoint, instruction, model=model)
    result.raw_output = raw_output
    result.latency_ms = latency

    if raw_output.startswith("ERROR:"):
        result.error = raw_output
        return result

    # Extract JSON from output (model may wrap in markdown fences)
    json_text = raw_output.strip()
    if json_text.startswith("```"):
        # Strip markdown code fences
        lines = json_text.split("\n")
        lines = [l for l in lines if not l.strip().startswith("```")]
        json_text = "\n".join(lines).strip()

    # Score 1: JSON validity
    try:
        result.actual = json.loads(json_text)
        result.json_valid = True
    except json.JSONDecodeError:
        result.error = "Model output is not valid JSON"
        return result

    # Score 2: Schema match (required keys present)
    required = get_required_keys(category)
    result.schema_match = all(k in result.actual for k in required)

    # Score 3: Value accuracy
    result.value_accuracy = score_value_accuracy(result.expected, result.actual, category)

    return result


def run_validation(
    adapter_name: str,
    validation_file: str,
    endpoint: Optional[str] = None,
    model: Optional[str] = None,
    max_examples: int = 0,
    dry_run: bool = False,
    is_baseline: bool = False,
) -> ValidationReport:
    """Run validation across all examples."""
    report = ValidationReport(adapter_name=adapter_name, is_baseline=is_baseline)

    # Load validation data
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
    print(f"Validating {len(examples)} examples against {adapter_name}{'  [dry-run]' if dry_run else ''}...")

    start = time.monotonic()
    for i, example in enumerate(examples):
        result = validate_one(i, example, endpoint, model, dry_run)
        report.results.append(result)

        # Progress indicator every 10 examples
        if (i + 1) % 10 == 0 or i == len(examples) - 1:
            print(f"  [{i+1}/{len(examples)}] avg_score={report.avg_score:.3f}")

    report.duration_seconds = time.monotonic() - start
    return report


def print_report(report: ValidationReport, verbose: bool = False):
    """Print a formatted validation report."""
    label = "BASELINE" if report.is_baseline else "ADAPTER"
    print(f"\n{'='*60}")
    print(f"  {label}: {report.adapter_name}")
    print(f"{'='*60}")
    print(f"  Examples:         {report.total_examples}")
    print(f"  Duration:         {report.duration_seconds:.1f}s")
    print(f"  JSON valid rate:  {report.json_valid_rate:.1%}")
    print(f"  Schema match:     {report.schema_match_rate:.1%}")
    print(f"  Value accuracy:   {report.avg_value_accuracy:.1%}")
    print(f"  Composite score:  {report.avg_score:.3f}")

    print(f"\n  Per-category scores:")
    for cat, score in sorted(report.category_scores.items()):
        count = sum(1 for r in report.results if r.category == cat)
        print(f"    {cat:.<30s} {score:.3f}  (n={count})")

    if verbose:
        # Show failures
        failures = [r for r in report.results if r.score < 0.5]
        if failures:
            print(f"\n  Low-scoring examples ({len(failures)}):")
            for r in failures[:10]:
                print(f"    [{r.index}] {r.category} score={r.score:.3f} err={r.error or 'low accuracy'}")


def print_comparison(adapter_report: ValidationReport, baseline_report: ValidationReport):
    """Print a comparison between adapter and baseline results."""
    print(f"\n{'='*60}")
    print("  COMPARISON: Adapter vs Baseline")
    print(f"{'='*60}")

    metrics = [
        ("JSON valid rate", adapter_report.json_valid_rate, baseline_report.json_valid_rate),
        ("Schema match", adapter_report.schema_match_rate, baseline_report.schema_match_rate),
        ("Value accuracy", adapter_report.avg_value_accuracy, baseline_report.avg_value_accuracy),
        ("Composite score", adapter_report.avg_score, baseline_report.avg_score),
    ]

    for name, adapter_val, baseline_val in metrics:
        delta = adapter_val - baseline_val
        arrow = "+" if delta > 0 else ""
        print(f"  {name:.<25s} adapter={adapter_val:.3f}  base={baseline_val:.3f}  ({arrow}{delta:.3f})")


def generate_json_report(report: ValidationReport,
                         baseline: Optional[ValidationReport] = None) -> Dict:
    """Generate a machine-readable JSON report."""
    result = {
        "adapter_name": report.adapter_name,
        "total_examples": report.total_examples,
        "duration_seconds": round(report.duration_seconds, 2),
        "json_valid_rate": round(report.json_valid_rate, 4),
        "schema_match_rate": round(report.schema_match_rate, 4),
        "avg_value_accuracy": round(report.avg_value_accuracy, 4),
        "composite_score": round(report.avg_score, 4),
        "category_scores": {k: round(v, 4) for k, v in report.category_scores.items()},
    }

    if baseline:
        result["baseline"] = {
            "composite_score": round(baseline.avg_score, 4),
            "json_valid_rate": round(baseline.json_valid_rate, 4),
        }
        result["improvement"] = round(report.avg_score - baseline.avg_score, 4)

    return result


def main():
    parser = argparse.ArgumentParser(description="Validate a QLoRA adapter")
    parser.add_argument("--adapter", required=True, help="Adapter name")
    parser.add_argument("--validation-file", required=True, help="Path to validation JSONL")
    parser.add_argument("--endpoint", default="http://localhost:7777",
                        help="vLLM or gaia-core endpoint (default: http://localhost:7777)")
    parser.add_argument("--baseline", action="store_true",
                        help="Also run validation against base model for comparison")
    parser.add_argument("--max-examples", type=int, default=0,
                        help="Limit number of examples (0=all)")
    parser.add_argument("--threshold", type=float, default=0.6,
                        help="Minimum composite score to pass (default: 0.6)")
    parser.add_argument("--dry-run", action="store_true",
                        help="Parse and report without calling model")
    parser.add_argument("--verbose", "-v", action="store_true",
                        help="Show detailed failure information")
    parser.add_argument("--json-report", type=str, default="",
                        help="Write JSON report to this path")
    args = parser.parse_args()

    # Run adapter validation
    adapter_report = run_validation(
        adapter_name=args.adapter,
        validation_file=args.validation_file,
        endpoint=args.endpoint,
        model=args.adapter,
        max_examples=args.max_examples,
        dry_run=args.dry_run,
    )
    print_report(adapter_report, verbose=args.verbose)

    baseline_report = None
    if args.baseline and not args.dry_run:
        baseline_report = run_validation(
            adapter_name="base-model",
            validation_file=args.validation_file,
            endpoint=args.endpoint,
            model=None,  # Default model = base
            max_examples=args.max_examples,
            is_baseline=True,
        )
        print_report(baseline_report, verbose=args.verbose)
        print_comparison(adapter_report, baseline_report)

    # Write JSON report if requested
    if args.json_report:
        report_data = generate_json_report(adapter_report, baseline_report)
        report_path = Path(args.json_report)
        report_path.parent.mkdir(parents=True, exist_ok=True)
        with open(report_path, "w") as f:
            json.dump(report_data, f, indent=2)
        print(f"\nJSON report written to {report_path}")

    # Pass/fail decision
    passed = adapter_report.avg_score >= args.threshold
    status = "PASS" if passed else "FAIL"
    print(f"\n{'='*60}")
    print(f"  RESULT: {status}  (score={adapter_report.avg_score:.3f}, threshold={args.threshold})")
    print(f"{'='*60}")

    sys.exit(0 if passed else 1)


if __name__ == "__main__":
    main()
