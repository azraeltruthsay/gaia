"""
gaia_common/utils/ast_summarizer.py

Reduce a Python source file to a compact, structured summary suitable for
LLM context. Raw source files are 300-800 lines; summaries should be 30-80
lines. This is the primary mechanism for keeping review prompts within
context budget.

The summarizer extracts:
  - Module-level docstring
  - Class definitions with bases and docstrings
  - Function/method signatures with type annotations
  - FastAPI router endpoint decorators
  - Enum subclass members
  - Module-level UPPER_CASE constants
  - gaia-package imports
  - Exception handlers with HTTP status codes (targeted body extraction)
  - HTTP client calls (targeted body extraction)

Uses dataclasses (not Pydantic) to keep gaia-common's dependency footprint light.
"""

from __future__ import annotations

import ast
import re
from dataclasses import dataclass, field
from typing import List, Optional


# ── Data classes ─────────────────────────────────────────────────────────────

@dataclass
class ClassInfo:
    name: str
    bases: List[str]
    docstring: Optional[str]
    methods: List[FunctionInfo]
    line: int


@dataclass
class FunctionInfo:
    name: str
    params: List[str]
    return_type: Optional[str]
    decorators: List[str]
    is_async: bool
    line: int


@dataclass
class EndpointInfo:
    method: str  # GET, POST, PUT, DELETE, PATCH, WEBSOCKET
    path: str
    function_name: str
    line: int


@dataclass
class EnumInfo:
    name: str
    members: List[tuple[str, str]]  # (name, value)
    line: int


@dataclass
class ConstantInfo:
    name: str
    value: str
    line: int


@dataclass
class ErrorHandlerInfo:
    exception_types: List[str]
    status_code: Optional[int]
    enclosing_function: Optional[str]
    line: int


@dataclass
class HttpCallInfo:
    call_method: str  # get, post, put, delete, etc.
    url_or_path: Optional[str]
    enclosing_function: Optional[str]
    line: int


@dataclass
class ASTSummary:
    module_docstring: Optional[str]
    classes: List[ClassInfo] = field(default_factory=list)
    functions: List[FunctionInfo] = field(default_factory=list)
    endpoints: List[EndpointInfo] = field(default_factory=list)
    enums: List[EnumInfo] = field(default_factory=list)
    constants: List[ConstantInfo] = field(default_factory=list)
    gaia_imports: List[str] = field(default_factory=list)
    error_handlers: List[ErrorHandlerInfo] = field(default_factory=list)
    http_calls: List[HttpCallInfo] = field(default_factory=list)
    filename: str = "<unknown>"

    def to_dict(self) -> dict:
        """Serialize to a plain dict for JSON output."""
        return {
            "module_docstring": self.module_docstring,
            "classes": [
                {
                    "name": c.name,
                    "bases": c.bases,
                    "docstring": c.docstring,
                    "methods": [
                        {
                            "name": m.name,
                            "params": m.params,
                            "return_type": m.return_type,
                            "decorators": m.decorators,
                            "is_async": m.is_async,
                            "line": m.line,
                        }
                        for m in c.methods
                    ],
                    "line": c.line,
                }
                for c in self.classes
            ],
            "functions": [
                {
                    "name": f.name,
                    "params": f.params,
                    "return_type": f.return_type,
                    "decorators": f.decorators,
                    "is_async": f.is_async,
                    "line": f.line,
                }
                for f in self.functions
            ],
            "endpoints": [
                {"method": e.method, "path": e.path, "function_name": e.function_name, "line": e.line}
                for e in self.endpoints
            ],
            "enums": [
                {"name": e.name, "members": e.members, "line": e.line}
                for e in self.enums
            ],
            "constants": [
                {"name": c.name, "value": c.value, "line": c.line}
                for c in self.constants
            ],
            "gaia_imports": self.gaia_imports,
            "error_handlers": [
                {
                    "exception_types": h.exception_types,
                    "status_code": h.status_code,
                    "enclosing_function": h.enclosing_function,
                    "line": h.line,
                }
                for h in self.error_handlers
            ],
            "http_calls": [
                {
                    "call_method": h.call_method,
                    "url_or_path": h.url_or_path,
                    "enclosing_function": h.enclosing_function,
                    "line": h.line,
                }
                for h in self.http_calls
            ],
        }

    def to_prompt_text(self) -> str:
        """Render as a human-readable block suitable for LLM prompt inclusion."""
        lines: list[str] = []
        lines.append(f"### File: {self.filename}")
        lines.append("")

        if self.module_docstring:
            lines.append(f"**Module:** {self.module_docstring}")
            lines.append("")

        if self.gaia_imports:
            lines.append("**GAIA Imports:**")
            for imp in self.gaia_imports:
                lines.append(f"  {imp}")
            lines.append("")

        if self.constants:
            lines.append("**Constants:**")
            for c in self.constants:
                lines.append(f"  {c.name} = {c.value}  (line {c.line})")
            lines.append("")

        if self.enums:
            lines.append("**Enums:**")
            for e in self.enums:
                members_str = ", ".join(f"{n}={v}" for n, v in e.members)
                lines.append(f"  {e.name}: {members_str}  (line {e.line})")
            lines.append("")

        if self.endpoints:
            lines.append("**Endpoints:**")
            for ep in self.endpoints:
                lines.append(f"  {ep.method} {ep.path} -> {ep.function_name}()  (line {ep.line})")
            lines.append("")

        # Top-level functions (not class methods)
        if self.functions:
            lines.append("**Functions:**")
            for f in self.functions:
                async_prefix = "async " if f.is_async else ""
                params_str = ", ".join(f.params)
                ret = f" -> {f.return_type}" if f.return_type else ""
                deco_str = ""
                if f.decorators:
                    deco_str = "  [" + ", ".join(f.decorators) + "]"
                lines.append(f"  {async_prefix}def {f.name}({params_str}){ret}{deco_str}  (line {f.line})")
            lines.append("")

        if self.classes:
            lines.append("**Classes:**")
            for c in self.classes:
                bases_str = f"({', '.join(c.bases)})" if c.bases else ""
                lines.append(f"  class {c.name}{bases_str}  (line {c.line})")
                if c.docstring:
                    lines.append(f"    \"{c.docstring}\"")
                for m in c.methods:
                    async_prefix = "async " if m.is_async else ""
                    params_str = ", ".join(m.params)
                    ret = f" -> {m.return_type}" if m.return_type else ""
                    lines.append(f"    {async_prefix}def {m.name}({params_str}){ret}  (line {m.line})")
            lines.append("")

        if self.error_handlers:
            lines.append("**Error Handlers:**")
            for h in self.error_handlers:
                exc_str = ", ".join(h.exception_types) if h.exception_types else "bare except"
                status = f" -> {h.status_code}" if h.status_code else ""
                fn = f" in {h.enclosing_function}()" if h.enclosing_function else ""
                lines.append(f"  handles: {exc_str}{status}{fn}  (line {h.line})")
            lines.append("")

        if self.http_calls:
            lines.append("**HTTP Calls:**")
            for h in self.http_calls:
                url = h.url_or_path or "?"
                fn = f" in {h.enclosing_function}()" if h.enclosing_function else ""
                lines.append(f"  {h.call_method.upper()} {url}{fn}  (line {h.line})")
            lines.append("")

        return "\n".join(lines)


# ── Constants ────────────────────────────────────────────────────────────────

_UPPER_CASE_RE = re.compile(r"^[A-Z][A-Z_]{2,}$")

_ROUTER_METHODS = {"get", "post", "put", "delete", "patch", "websocket"}

_HTTP_CLIENT_PATTERNS = {"httpx", "requests"}


# ── Main entry point ─────────────────────────────────────────────────────────

def summarize_file(source: str, filename: str = "<unknown>") -> ASTSummary:
    """Parse Python source and return structured summary."""
    tree = ast.parse(source, filename=filename)

    summary = ASTSummary(
        module_docstring=_extract_module_docstring(tree),
        filename=filename,
    )

    # Extract gaia imports
    summary.gaia_imports = _extract_gaia_imports(tree)

    # Extract module-level constants
    summary.constants = _extract_constants(tree)

    # Build function-enclosure map for targeted body extractions
    _func_map = _build_function_map(tree)

    # Walk tree for classes, functions, endpoints, enums, error handlers, http calls
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ClassDef):
            if _is_enum_class(node):
                summary.enums.append(_extract_enum(node))
            else:
                class_info = _extract_class(node)
                summary.classes.append(class_info)
                # Extract endpoints from class methods
                for method_node in ast.walk(node):
                    if isinstance(method_node, (ast.FunctionDef, ast.AsyncFunctionDef)):
                        eps = _extract_endpoints_from_decorators(method_node)
                        summary.endpoints.extend(eps)

        elif isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            summary.functions.append(_extract_function(node))
            eps = _extract_endpoints_from_decorators(node)
            summary.endpoints.extend(eps)

    # Targeted body extractions: error handlers and HTTP calls
    summary.error_handlers = _extract_error_handlers(tree, _func_map)
    summary.http_calls = _extract_http_calls(tree, _func_map)

    return summary


# ── Extraction helpers ───────────────────────────────────────────────────────

def _extract_module_docstring(tree: ast.Module) -> Optional[str]:
    doc = ast.get_docstring(tree)
    if doc:
        return doc[:200]
    return None


def _extract_gaia_imports(tree: ast.Module) -> list[str]:
    imports: list[str] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.ImportFrom) and node.module and node.module.startswith("gaia_"):
            names = ", ".join(alias.name for alias in node.names)
            imports.append(f"from {node.module} import {names}")
        elif isinstance(node, ast.Import):
            for alias in node.names:
                if alias.name.startswith("gaia_"):
                    imports.append(f"import {alias.name}")
    return imports


def _extract_constants(tree: ast.Module) -> list[ConstantInfo]:
    constants: list[ConstantInfo] = []
    for node in ast.iter_child_nodes(tree):
        if isinstance(node, ast.Assign):
            for target in node.targets:
                if isinstance(target, ast.Name) and _UPPER_CASE_RE.match(target.id):
                    value = _safe_constant_value(node.value)
                    if value is not None:
                        constants.append(ConstantInfo(
                            name=target.id,
                            value=value,
                            line=node.lineno,
                        ))
    return constants


def _safe_constant_value(node: ast.expr) -> Optional[str]:
    """Extract value for str/int/bool/float constants. Truncate long strings."""
    if isinstance(node, ast.Constant):
        if isinstance(node.value, str):
            val = node.value
            if len(val) > 80:
                val = val[:77] + "..."
            return repr(val)
        if isinstance(node.value, (int, float, bool)):
            return repr(node.value)
    return None


def _is_enum_class(node: ast.ClassDef) -> bool:
    for base in node.bases:
        base_name = _get_name(base)
        if base_name and "Enum" in base_name:
            return True
    return False


def _extract_enum(node: ast.ClassDef) -> EnumInfo:
    members: list[tuple[str, str]] = []
    for item in node.body:
        if isinstance(item, ast.Assign):
            for target in item.targets:
                if isinstance(target, ast.Name):
                    val = _safe_constant_value(item.value)
                    members.append((target.id, val or "?"))
    return EnumInfo(name=node.name, members=members, line=node.lineno)


def _extract_class(node: ast.ClassDef) -> ClassInfo:
    bases = [_get_name(b) or "?" for b in node.bases]
    docstring = ast.get_docstring(node)
    if docstring:
        docstring = docstring[:100]

    methods: list[FunctionInfo] = []
    for item in node.body:
        if isinstance(item, (ast.FunctionDef, ast.AsyncFunctionDef)):
            methods.append(_extract_function(item))

    return ClassInfo(
        name=node.name,
        bases=bases,
        docstring=docstring,
        methods=methods,
        line=node.lineno,
    )


def _extract_function(node: ast.FunctionDef | ast.AsyncFunctionDef) -> FunctionInfo:
    params = _extract_params(node)
    return_type = ast.unparse(node.returns) if node.returns else None
    decorators = [ast.unparse(d) for d in node.decorator_list]
    is_async = isinstance(node, ast.AsyncFunctionDef)

    return FunctionInfo(
        name=node.name,
        params=params,
        return_type=return_type,
        decorators=decorators,
        is_async=is_async,
        line=node.lineno,
    )


def _extract_params(node: ast.FunctionDef | ast.AsyncFunctionDef) -> list[str]:
    """Extract parameter signatures with type annotations."""
    params: list[str] = []
    args = node.args

    # Calculate default offset (defaults align with the last N positional args)
    num_defaults = len(args.defaults)
    num_args = len(args.args)
    default_offset = num_args - num_defaults

    for i, arg in enumerate(args.args):
        param = arg.arg
        if arg.annotation:
            param += f": {ast.unparse(arg.annotation)}"
        # Check if this arg has a default
        default_idx = i - default_offset
        if default_idx >= 0 and default_idx < len(args.defaults):
            default_val = ast.unparse(args.defaults[default_idx])
            if len(default_val) > 30:
                default_val = "..."
            param += f" = {default_val}"
        params.append(param)

    if args.vararg:
        p = f"*{args.vararg.arg}"
        if args.vararg.annotation:
            p += f": {ast.unparse(args.vararg.annotation)}"
        params.append(p)
    elif args.kwonlyargs:
        params.append("*")

    for i, arg in enumerate(args.kwonlyargs):
        param = arg.arg
        if arg.annotation:
            param += f": {ast.unparse(arg.annotation)}"
        if i < len(args.kw_defaults) and args.kw_defaults[i] is not None:
            default_val = ast.unparse(args.kw_defaults[i])
            if len(default_val) > 30:
                default_val = "..."
            param += f" = {default_val}"
        params.append(param)

    if args.kwarg:
        p = f"**{args.kwarg.arg}"
        if args.kwarg.annotation:
            p += f": {ast.unparse(args.kwarg.annotation)}"
        params.append(p)

    return params


def _extract_endpoints_from_decorators(
    node: ast.FunctionDef | ast.AsyncFunctionDef,
) -> list[EndpointInfo]:
    """Extract FastAPI endpoint info from @router.get(...) style decorators."""
    endpoints: list[EndpointInfo] = []
    for deco in node.decorator_list:
        if not isinstance(deco, ast.Call):
            continue
        func = deco.func
        if not isinstance(func, ast.Attribute):
            continue
        if func.attr not in _ROUTER_METHODS:
            continue
        # Check that the object is router or app (or similar)
        obj_name = _get_name(func.value)
        if obj_name not in ("router", "app", "self.router", "self.app"):
            continue
        # Extract path from first positional arg
        path = "?"
        if deco.args and isinstance(deco.args[0], ast.Constant) and isinstance(deco.args[0].value, str):
            path = deco.args[0].value
        method = func.attr.upper()
        endpoints.append(EndpointInfo(
            method=method,
            path=path,
            function_name=node.name,
            line=node.lineno,
        ))
    return endpoints


def _build_function_map(tree: ast.Module) -> dict[int, str]:
    """Map line numbers to enclosing function names for targeted extraction."""
    func_ranges: list[tuple[int, int, str]] = []
    for node in ast.walk(tree):
        if isinstance(node, (ast.FunctionDef, ast.AsyncFunctionDef)):
            end = getattr(node, "end_lineno", node.lineno + 100)
            func_ranges.append((node.lineno, end, node.name))
    # Sort by start line
    func_ranges.sort(key=lambda x: x[0])

    line_map: dict[int, str] = {}
    for start, end, name in func_ranges:
        for line in range(start, end + 1):
            line_map[line] = name
    return line_map


def _extract_error_handlers(
    tree: ast.Module, func_map: dict[int, str]
) -> list[ErrorHandlerInfo]:
    """Extract try/except blocks with exception types and HTTP status codes."""
    handlers: list[ErrorHandlerInfo] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.ExceptHandler):
            continue
        exc_types: list[str] = []
        if node.type:
            if isinstance(node.type, ast.Tuple):
                exc_types = [_get_name(e) or "?" for e in node.type.elts]
            else:
                name = _get_name(node.type)
                if name:
                    exc_types = [name]

        # Look for status_code in the handler body
        status_code = _find_status_code_in_body(node.body)
        enclosing = func_map.get(node.lineno)

        handlers.append(ErrorHandlerInfo(
            exception_types=exc_types,
            status_code=status_code,
            enclosing_function=enclosing,
            line=node.lineno,
        ))
    return handlers


def _find_status_code_in_body(body: list[ast.stmt]) -> Optional[int]:
    """Scan handler body for HTTP status codes (e.g., status_code=504)."""
    for node in ast.walk(ast.Module(body=body, type_ignores=[])):
        if isinstance(node, ast.keyword) and node.arg == "status_code":
            if isinstance(node.value, ast.Constant) and isinstance(node.value.value, int):
                return node.value.value
        # Also check for plain integer returns like `return JSONResponse(status_code=504, ...)`
        if isinstance(node, ast.Constant) and isinstance(node.value, int):
            val = node.value
            if 400 <= val <= 599:
                return val
    return None


def _extract_http_calls(
    tree: ast.Module, func_map: dict[int, str]
) -> list[HttpCallInfo]:
    """Extract HTTP client calls (httpx.*, requests.*, self.client.*)."""
    calls: list[HttpCallInfo] = []
    for node in ast.walk(tree):
        if not isinstance(node, ast.Call):
            continue
        if not isinstance(node.func, ast.Attribute):
            continue

        attr = node.func
        method_name = attr.attr
        obj_name = _get_name(attr.value)

        if obj_name is None:
            continue

        is_http = False
        if obj_name in _HTTP_CLIENT_PATTERNS:
            is_http = True
        elif obj_name.startswith("self.client") or obj_name.startswith("self._client"):
            is_http = True
        elif "client" in obj_name.lower() and method_name in ("get", "post", "put", "delete", "patch", "request"):
            is_http = True

        if not is_http:
            continue
        if method_name.startswith("_"):
            continue

        # Extract URL from first positional arg
        url: Optional[str] = None
        if node.args:
            first_arg = node.args[0]
            if isinstance(first_arg, ast.Constant) and isinstance(first_arg.value, str):
                url = first_arg.value
            elif isinstance(first_arg, ast.JoinedStr):
                url = "<f-string>"
            else:
                url = ast.unparse(first_arg)
                if len(url) > 60:
                    url = url[:57] + "..."

        enclosing = func_map.get(node.lineno)
        calls.append(HttpCallInfo(
            call_method=method_name,
            url_or_path=url,
            enclosing_function=enclosing,
            line=node.lineno,
        ))
    return calls


def _get_name(node: ast.expr) -> Optional[str]:
    """Recursively resolve dotted names from AST nodes."""
    if isinstance(node, ast.Name):
        return node.id
    if isinstance(node, ast.Attribute):
        parent = _get_name(node.value)
        if parent:
            return f"{parent}.{node.attr}"
        return node.attr
    return None
