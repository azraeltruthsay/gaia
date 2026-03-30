#!/usr/bin/env python3
"""
build_review_prompt.py — Assemble a review prompt from blueprint + AST summaries + pre-check.

Usage (inside Docker container):
    python - --blueprint /knowledge/blueprints/gaia-core.yaml \
             --ast-summaries /knowledge/tmp/ast.json \
             --precheck /knowledge/tmp/precheck.json \
        < scripts/build_review_prompt.py

Outputs the assembled review prompt to stdout.
"""
from __future__ import annotations

import argparse
import json
import sys
from datetime import datetime, timezone
from pathlib import Path

from gaia_common.models.blueprint import BlueprintModel
from gaia_common.utils.ast_summarizer import (
    ASTSummary,
    ClassInfo,
    ConstantInfo,
    EndpointInfo,
    EnumInfo,
    ErrorHandlerInfo,
    FunctionInfo,
    HttpCallInfo,
)
from gaia_common.utils.blueprint_precheck import PreCheckItem, PreCheckResult, PreCheckSummary
from gaia_common.utils.review_prompt_builder import ReviewResult, build_review_prompt


def _load_blueprint(path: str) -> BlueprintModel:
    """Load a blueprint from YAML file."""
    try:
        import yaml
    except ImportError:
        print("Error: PyYAML not available", file=sys.stderr)
        sys.exit(1)

    bp_path = Path(path)
    if not bp_path.exists():
        print(f"Error: Blueprint not found: {bp_path}", file=sys.stderr)
        sys.exit(1)

    raw = yaml.safe_load(bp_path.read_text(encoding="utf-8"))
    return BlueprintModel.model_validate(raw)


def _load_ast_summaries(path: str) -> dict[str, ASTSummary]:
    """Load AST summaries from JSON file (as produced by generate_ast_summaries.py)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    summaries: dict[str, ASTSummary] = {}

    for filename, d in data.items():
        # Reconstruct ClassInfo with nested FunctionInfo for methods
        classes = []
        for c in d.get("classes", []):
            methods = [FunctionInfo(**m) for m in c.get("methods", [])]
            classes.append(ClassInfo(
                name=c["name"],
                bases=c.get("bases", []),
                docstring=c.get("docstring"),
                methods=methods,
                line=c.get("line", 0),
            ))

        # Reconstruct EnumInfo with members (tuples from JSON lists)
        enums = []
        for e in d.get("enums", []):
            members = [tuple(m) if isinstance(m, list) else m for m in e.get("members", [])]
            enums.append(EnumInfo(
                name=e["name"],
                members=members,
                line=e.get("line", 0),
            ))

        summaries[filename] = ASTSummary(
            module_docstring=d.get("module_docstring"),
            classes=classes,
            functions=[FunctionInfo(**f) for f in d.get("functions", [])],
            endpoints=[EndpointInfo(**e) for e in d.get("endpoints", [])],
            enums=enums,
            constants=[ConstantInfo(**c) for c in d.get("constants", [])],
            gaia_imports=d.get("gaia_imports", []),
            error_handlers=[ErrorHandlerInfo(**h) for h in d.get("error_handlers", [])],
            http_calls=[HttpCallInfo(**h) for h in d.get("http_calls", [])],
            filename=d.get("filename", filename),
        )

    return summaries


def _load_precheck(path: str) -> PreCheckResult:
    """Load pre-check results from JSON file (as produced by run_blueprint_precheck.py)."""
    data = json.loads(Path(path).read_text(encoding="utf-8"))

    items = [
        PreCheckItem(
            category=item["category"],
            blueprint_claim=item["blueprint_claim"],
            status=item["status"],
            source_file=item.get("source_file"),
            detail=item.get("detail", ""),
        )
        for item in data.get("items", [])
    ]

    summary_data = data.get("summary", {})
    summary = PreCheckSummary(
        total=summary_data.get("total", 0),
        found=summary_data.get("found", 0),
        missing=summary_data.get("missing", 0),
        diverged=summary_data.get("diverged", 0),
    )

    return PreCheckResult(
        service_id=data.get("service_id", "unknown"),
        timestamp=datetime.fromisoformat(data["timestamp"]) if "timestamp" in data else datetime.now(timezone.utc),
        items=items,
        summary=summary,
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build a review prompt from blueprint + AST + pre-check")
    parser.add_argument("--blueprint", required=True, help="Path to blueprint YAML file")
    parser.add_argument("--ast-summaries", required=True, help="Path to AST summaries JSON file")
    parser.add_argument("--precheck", required=True, help="Path to pre-check results JSON file")
    parser.add_argument("--output", help="Output file path (default: stdout)")
    parser.add_argument("--direction", choices=["forward", "reverse"], default="forward",
                        help="Review direction (default: forward)")
    parser.add_argument("--max-tokens", type=int, default=None,
                        help="Maximum prompt tokens (triggers truncation if exceeded)")
    parser.add_argument("--schema", action="store_true",
                        help="Also emit the ReviewResult JSON schema to stderr")
    args = parser.parse_args()

    blueprint = _load_blueprint(args.blueprint)
    ast_summaries = _load_ast_summaries(args.ast_summaries)
    precheck_result = _load_precheck(args.precheck)

    prompt = build_review_prompt(
        blueprint,
        ast_summaries,
        precheck_result,
        review_direction=args.direction,
        max_prompt_tokens=args.max_tokens,
    )

    if args.output:
        Path(args.output).parent.mkdir(parents=True, exist_ok=True)
        Path(args.output).write_text(prompt, encoding="utf-8")
        print(f"Wrote review prompt to {args.output}", file=sys.stderr)
    else:
        print(prompt)

    token_est = len(prompt) // 4
    print(f"Prompt: {len(prompt)} chars (~{token_est} tokens), {len(ast_summaries)} files", file=sys.stderr)

    if args.schema:
        schema = ReviewResult.model_json_schema()
        print(json.dumps(schema, indent=2), file=sys.stderr)


if __name__ == "__main__":
    main()
