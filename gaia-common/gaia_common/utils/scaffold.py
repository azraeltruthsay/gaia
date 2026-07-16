"""Scaffold library — parameterized code templates for constrained generation (s4r2).

Azrael's premise: generation costs more than reading. Instead of asking a
small model to free-generate a whole module (where syntax errors and
confabulated APIs come from — and post-kmcb every such error costs a
deploy-rollback cycle), ship fill-in-the-blank skeletons with the
invariants pre-written. The model generates only the blanks.

Templates live in gaia_common/scaffolds/*.tmpl (shipped into every
container by the existing gaia-common mount). Format: a header comment
block parsed for metadata, then a `string.Template` body using ${var}
placeholders.

Header grammar (line-oriented, inside the leading comment block):

    # scaffold: <name>
    # description: <one line>
    # output: python | json
    # var <name>: <description> | example: <example value>

Rendered Python output is ast.parse-validated; JSON output is
json.loads-validated. Unfilled placeholders raise — a scaffold never
emits a half-filled artifact.
"""

from __future__ import annotations

import ast
import json
import logging
import re
from dataclasses import dataclass, field
from pathlib import Path
from string import Template
from typing import Dict, List, Optional

logger = logging.getLogger("GAIA.Scaffold")

SCAFFOLDS_DIR = Path(__file__).resolve().parent.parent / "scaffolds"

# 'example: ' captures raw after one space so examples can carry the
# leading indentation their placeholder position requires.
_VAR_RE = re.compile(r"^#\s*var\s+(\w+):\s*(.*?)\s*\|\s*example: ?(.*)$")
_META_RE = re.compile(r"^#\s*(scaffold|description|output):\s*(.*)$")


@dataclass(frozen=True)
class ScaffoldVar:
    name: str
    description: str
    example: str


@dataclass(frozen=True)
class Scaffold:
    name: str
    description: str
    output: str                      # "python" | "json"
    variables: List[ScaffoldVar] = field(default_factory=list)
    body: str = ""
    path: Optional[Path] = None


class ScaffoldError(ValueError):
    """Raised for unknown scaffolds, missing variables, or invalid output."""


def _parse_template(path: Path) -> Optional[Scaffold]:
    try:
        text = path.read_text(encoding="utf-8")
    except OSError:
        logger.warning("Cannot read scaffold %s", path)
        return None

    meta: Dict[str, str] = {}
    variables: List[ScaffoldVar] = []
    body_lines: List[str] = []
    in_header = True
    for line in text.splitlines():
        if in_header:
            vm = _VAR_RE.match(line)
            if vm:
                variables.append(ScaffoldVar(vm.group(1), vm.group(2), vm.group(3)))
                continue
            mm = _META_RE.match(line)
            if mm:
                meta[mm.group(1)] = mm.group(2).strip()
                continue
            if line.startswith("#") or not line.strip():
                continue
            in_header = False
        body_lines.append(line)

    if "scaffold" not in meta:
        logger.warning("Scaffold %s has no '# scaffold:' header — skipping", path.name)
        return None
    return Scaffold(
        name=meta["scaffold"],
        description=meta.get("description", ""),
        output=meta.get("output", "python"),
        variables=variables,
        body="\n".join(body_lines).strip() + "\n",
        path=path,
    )


def list_scaffolds(scaffolds_dir: Optional[Path] = None) -> List[Scaffold]:
    """All parseable scaffolds, sorted by name."""
    d = scaffolds_dir or SCAFFOLDS_DIR
    if not d.is_dir():
        return []
    out = []
    for p in sorted(d.glob("*.tmpl")):
        s = _parse_template(p)
        if s:
            out.append(s)
    return out


def get_scaffold(name: str, scaffolds_dir: Optional[Path] = None) -> Scaffold:
    for s in list_scaffolds(scaffolds_dir):
        if s.name == name:
            return s
    raise ScaffoldError(f"unknown scaffold: {name!r}")


def render(name: str, variables: Dict[str, str],
           scaffolds_dir: Optional[Path] = None) -> str:
    """Instantiate a scaffold. Raises ScaffoldError on missing variables
    or invalid output — never returns a half-filled artifact."""
    s = get_scaffold(name, scaffolds_dir)
    missing = [v.name for v in s.variables if v.name not in variables]
    if missing:
        raise ScaffoldError(f"scaffold {name!r}: missing variables {missing}")
    try:
        content = Template(s.body).substitute(variables)
    except (KeyError, ValueError) as e:
        raise ScaffoldError(f"scaffold {name!r}: substitution failed: {e}") from e

    if s.output == "python":
        try:
            ast.parse(content)
        except SyntaxError as e:
            raise ScaffoldError(
                f"scaffold {name!r}: rendered output is not valid Python "
                f"(line {e.lineno}: {e.msg}) — check variable values") from e
    elif s.output == "json":
        try:
            json.loads(content)
        except json.JSONDecodeError as e:
            raise ScaffoldError(
                f"scaffold {name!r}: rendered output is not valid JSON ({e})") from e
    return content


def scaffold_prompt_block(scaffolds_dir: Optional[Path] = None) -> str:
    """Compact scaffold inventory for injection into generation prompts."""
    scaffolds = list_scaffolds(scaffolds_dir)
    if not scaffolds:
        return ""
    lines = ["AVAILABLE SCAFFOLDS (prefer instantiating one over free-writing new code;",
             "name the scaffold and provide ONLY the variable values):"]
    for s in scaffolds:
        vars_desc = ", ".join(v.name for v in s.variables)
        lines.append(f"- {s.name}: {s.description} (vars: {vars_desc})")
    return "\n".join(lines)
