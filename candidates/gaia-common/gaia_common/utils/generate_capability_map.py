"""generate_capability_map.py – Build a concise capability manifest for GAIA.

This utility supports two collection modes – static (import‑graph) and runtime
(trace‑based) – and lets you *filter* the final list so only actionable
primitives reach the AI.

Typical usage inside the Docker container
----------------------------------------
    # Ground‑truth scan, keep only ai.* helpers and GAIADevMatrix methods
    python app/utils/generate_capability_map.py \
        --runtime --include-methods --filter-regex '^ai\\.|^GAIADevMatrix'

CLI flags
~~~~~~~~~
--entry            Entry script (default: gaia_rescue.py)
--runtime          Execute entry under trace; otherwise static analysis.
--include-funcs    Add top‑level public functions.
--include-methods  Add Class.method names for public methods.
--filter-regex     Regex; keep capability names that *match* (post‑collect).
--out              Output JSON path (default: knowledge/system_reference/...)
"""

from __future__ import annotations

# -------------------------------------------------------------
#  Imports                                                     #
# -------------------------------------------------------------
import argparse
import ast
import json
import os
import re
import runpy
import sys
import trace
from pathlib import Path
from typing import List, Set

# -------------------------------------------------------------
#  Helpers                                                     #
# -------------------------------------------------------------

def discover_repo_root() -> Path:
    """Return repo root by walking **two** levels up (utils → app → <root>)."""
    return Path(__file__).resolve().parents[2]


def is_public_name(name: str) -> bool:
    """Return *True* for non‑private, non‑dunder names."""
    return not name.startswith("_") and not (name.startswith("__") and name.endswith("__"))


def module_to_path(module: str, repo_root: Path) -> Path:
    """Best‑effort convert *package.sub.module* → filesystem path inside repo."""
    candidate = repo_root.joinpath(*module.split(".")).with_suffix(".py")
    if candidate.exists():
        return candidate
    candidate_init = candidate.with_name("__init__.py")
    return candidate_init if candidate_init.exists() else Path()

# -------------------------------------------------------------
#  AST extraction utilities                                    #
# -------------------------------------------------------------

def _add_class_caps(tree: ast.AST, caps: Set[str]):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and is_public_name(node.name):
            caps.add(node.name)


def _add_function_caps(tree: ast.AST, caps: Set[str]):
    for node in ast.walk(tree):
        if isinstance(node, ast.FunctionDef) and is_public_name(node.name):
            caps.add(node.name)


def _add_method_caps(tree: ast.AST, caps: Set[str]):
    for node in ast.walk(tree):
        if isinstance(node, ast.ClassDef) and is_public_name(node.name):
            for body_item in node.body:
                if isinstance(body_item, ast.FunctionDef) and is_public_name(body_item.name):
                    caps.add(f"{node.name}.{body_item.name}")


def extract_caps_from_file(py_file: Path, caps: Set[str], *, include_funcs: bool, include_methods: bool):
    """Update *caps* with public symbols from *py_file*."""
    try:
        tree = ast.parse(py_file.read_text("utf‑8"), filename=str(py_file))
    except Exception as err:  # pragma: no cover – skip unparsable files
        print(f"Warning: Failed to parse {py_file}: {err}", file=sys.stderr)
        return

    _add_class_caps(tree, caps)
    if include_funcs:
        _add_function_caps(tree, caps)
    if include_methods:
        _add_method_caps(tree, caps)

# -------------------------------------------------------------
#  Static collection                                           #
# -------------------------------------------------------------

def static_collect(entry: Path, repo_root: Path, *, include_funcs: bool, include_methods: bool) -> Set[str]:
    visited: Set[Path] = set()
    queue: List[Path] = [entry]
    caps: Set[str] = set()

    while queue:
        current = queue.pop()
        if current in visited or not current.is_file():
            continue
        visited.add(current)
        extract_caps_from_file(current, caps, include_funcs=include_funcs, include_methods=include_methods)

        # Follow imports
        try:
            tree = ast.parse(current.read_text("utf‑8"))
        except Exception:
            continue
        for node in ast.walk(tree):
            mod = None
            if isinstance(node, ast.Import):
                mod = node.names[0].name  # first alias
            elif isinstance(node, ast.ImportFrom) and node.level == 0:
                mod = node.module
            if mod:
                target = module_to_path(mod, repo_root)
                if target and target not in visited:
                    queue.append(target)
    return caps

# -------------------------------------------------------------
#  Runtime collection                                          #
# -------------------------------------------------------------

def runtime_collect(entry: Path, repo_root: Path, *, include_funcs: bool, include_methods: bool) -> Set[str]:
    """Execute *entry* under `trace` and gather executed python files."""
    os.environ.setdefault("GAIA_CAPABILITY_SNAPSHOT", "1")
    tracer = trace.Trace(trace=False, count=True)

    def _exec():
        original_argv = sys.argv[:]
        sys.argv = [str(entry)]  # strip our flags
        try:
            runpy.run_path(str(entry), run_name="__main__")
        finally:
            sys.argv = original_argv

    try:
        tracer.runfunc(_exec)
    except SystemExit:
        pass  # entry may call sys.exit()
    except Exception as e:
        print(f"Warning: runtime trace raised {e.__class__.__name__}: {e}", file=sys.stderr)

    executed = set()
    for key in tracer.results().counts.keys():
        fname = key[0] if isinstance(key, tuple) else key
        if isinstance(fname, str) and fname.endswith(".py"):
            executed.add(Path(fname).resolve())

    caps: Set[str] = set()
    for py in executed:
        if not str(py).startswith(str(repo_root)):
            continue  # skip site‑packages
        extract_caps_from_file(py, caps, include_funcs=include_funcs, include_methods=include_methods)
    return caps

# -------------------------------------------------------------
#  Manifest writer                                             #
# -------------------------------------------------------------

def write_manifest(caps: Set[str], outfile: Path):
    outfile.parent.mkdir(parents=True, exist_ok=True)
    outfile.write_text(json.dumps(sorted(caps), indent=2), "utf‑8")
    print(f"Capability map generated at {outfile} with {len(caps)} capabilities.")

# -------------------------------------------------------------
#  CLI                                                         #
# -------------------------------------------------------------


def main() -> None:  # pragma: no cover
    repo_root = discover_repo_root()

    p = argparse.ArgumentParser(prog="generate_capability_map", description="Produce GAIA capability manifest (JSON)")
    p.add_argument("--entry", default="gaia_rescue.py", help="Entry script relative to repo root")
    p.add_argument("--out", default="knowledge/system_reference/capabilities.json", help="Output JSON path relative to repo root")
    p.add_argument("--runtime", action="store_true", help="Run entry under trace to capture executed files")
    p.add_argument("--include-funcs", action="store_true", help="Include public top‑level functions")
    p.add_argument("--include-methods", action="store_true", help="Include public methods as Class.method")
    p.add_argument("--filter-regex", help="Regex to *keep* names that match (e.g. '^GAIA|^ai\\.')")
    args = p.parse_args()

    entry_path = repo_root / args.entry
    if not entry_path.exists():
        sys.exit(f"Error: entry file {entry_path} not found.")

    collect = runtime_collect if args.runtime else static_collect
    caps = collect(entry_path, repo_root, include_funcs=args.include_funcs, include_methods=args.include_methods)

    # Optional regex filter
    if args.filter_regex:
        try:
            rx = re.compile(args.filter_regex)
            caps = {c for c in caps if rx.search(c)}
        except re.error as err:
            sys.exit(f"Invalid --filter-regex: {err}")

    write_manifest(caps, repo_root / args.out)


if __name__ == "__main__":
    main()
