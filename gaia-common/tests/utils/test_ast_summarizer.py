"""
Tests for gaia_common.utils.ast_summarizer

Uses synthetic fixture code that exercises all extraction paths.
"""

import pytest
from gaia_common.utils.ast_summarizer import summarize_file, ASTSummary


# ── Synthetic fixture: a realistic multi-feature Python module ───────────────

FIXTURE_SOURCE = '''
"""
Module for managing cognitive sessions.

Provides REST endpoints for session CRUD operations, WebSocket streaming,
and enum-based status tracking with failure mode handling.
"""

from __future__ import annotations

import logging
from typing import Dict, List, Optional

from fastapi import APIRouter, WebSocket, HTTPException
from pydantic import BaseModel

from gaia_common.models.blueprint import BlueprintModel
from gaia_core.cognition.session import SessionManager
import gaia_common.utils.helpers

logger = logging.getLogger(__name__)

router = APIRouter(prefix="/sessions")
app = None  # placeholder for websocket

MAX_SESSIONS = 100
DEFAULT_TIMEOUT_MS = 5000
SESSION_PREFIX = "gaia-session"


class SessionStatus(Enum):
    """Status of a cognitive session."""
    ACTIVE = "active"
    IDLE = "idle"
    TERMINATED = "terminated"


class SessionRequest(BaseModel):
    """Request model for creating a session."""
    user_id: str
    context: Optional[Dict[str, str]] = None


class SessionResponse(BaseModel):
    """Response model for session data."""
    session_id: str
    status: str
    created_at: str


class SessionManager:
    """Manages cognitive session lifecycle."""

    def __init__(self, max_sessions: int = MAX_SESSIONS):
        self._sessions: Dict[str, dict] = {}
        self._max = max_sessions

    def create_session(self, user_id: str) -> str:
        """Create a new cognitive session."""
        pass

    async def terminate_session(self, session_id: str) -> bool:
        """Terminate a session by ID."""
        pass

    def get_session(self, session_id: str) -> Optional[dict]:
        """Get session data."""
        pass


@router.get("/health")
async def health_check() -> dict:
    return {"status": "healthy", "service": "session-manager"}


@router.get("/{session_id}")
async def get_session(session_id: str) -> SessionResponse:
    try:
        result = httpx.get(f"http://gaia-core:8000/internal/sessions/{session_id}")
        return SessionResponse(**result.json())
    except httpx.TimeoutException as e:
        raise HTTPException(status_code=504, detail="Core service timeout")
    except httpx.ConnectError:
        raise HTTPException(status_code=502, detail="Core service unavailable")


@router.post("/")
async def create_session(request: SessionRequest) -> SessionResponse:
    try:
        result = httpx.post("http://gaia-core:8000/internal/sessions", json=request.dict())
        return SessionResponse(**result.json())
    except ValueError as e:
        raise HTTPException(status_code=400, detail=str(e))


@router.delete("/{session_id}")
async def delete_session(session_id: str) -> dict:
    return {"deleted": True}


@router.websocket("/ws/stream")
async def websocket_stream(websocket: WebSocket):
    await websocket.accept()
    try:
        while True:
            data = await websocket.receive_text()
            await websocket.send_text(f"echo: {data}")
    except Exception:
        pass


def _internal_helper(x: int) -> str:
    """Not a public function."""
    return str(x)


async def process_batch(
    items: List[str],
    *,
    timeout: int = DEFAULT_TIMEOUT_MS,
    callback: Optional[callable] = None,
) -> List[dict]:
    """Process a batch of items with optional callback."""
    results = []
    try:
        response = httpx.post(
            "http://gaia-mcp:8001/mcp/invoke",
            json={"items": items},
        )
        results = response.json()
    except httpx.ReadTimeout:
        logger.warning("MCP call timed out")
    return results
'''


class TestSummarizeFile:
    """Test the main summarize_file function."""

    @pytest.fixture
    def summary(self) -> ASTSummary:
        return summarize_file(FIXTURE_SOURCE, filename="test_sessions.py")

    def test_module_docstring_extracted(self, summary: ASTSummary):
        assert summary.module_docstring is not None
        assert "cognitive sessions" in summary.module_docstring

    def test_module_docstring_truncated(self):
        long_doc = '"""' + "x" * 300 + '"""'
        s = summarize_file(long_doc)
        assert s.module_docstring is not None
        assert len(s.module_docstring) <= 200

    def test_gaia_imports_extracted(self, summary: ASTSummary):
        assert len(summary.gaia_imports) >= 2
        import_strs = " ".join(summary.gaia_imports)
        assert "gaia_common.models.blueprint" in import_strs
        assert "gaia_core.cognition.session" in import_strs
        assert "gaia_common.utils.helpers" in import_strs

    def test_non_gaia_imports_excluded(self, summary: ASTSummary):
        import_strs = " ".join(summary.gaia_imports)
        assert "fastapi" not in import_strs
        assert "pydantic" not in import_strs
        assert "logging" not in import_strs

    def test_constants_extracted(self, summary: ASTSummary):
        const_names = {c.name for c in summary.constants}
        assert "MAX_SESSIONS" in const_names
        assert "DEFAULT_TIMEOUT_MS" in const_names
        assert "SESSION_PREFIX" in const_names
        # logger is not UPPER_CASE
        assert "logger" not in const_names

    def test_constant_values(self, summary: ASTSummary):
        by_name = {c.name: c for c in summary.constants}
        assert by_name["MAX_SESSIONS"].value == "100"
        assert by_name["DEFAULT_TIMEOUT_MS"].value == "5000"
        assert "gaia-session" in by_name["SESSION_PREFIX"].value

    def test_enum_extracted(self, summary: ASTSummary):
        assert len(summary.enums) == 1
        e = summary.enums[0]
        assert e.name == "SessionStatus"
        member_names = [m[0] for m in e.members]
        assert "ACTIVE" in member_names
        assert "IDLE" in member_names
        assert "TERMINATED" in member_names

    def test_enum_values(self, summary: ASTSummary):
        e = summary.enums[0]
        member_dict = dict(e.members)
        assert member_dict["ACTIVE"] == "'active'"

    def test_classes_extracted(self, summary: ASTSummary):
        class_names = {c.name for c in summary.classes}
        assert "SessionRequest" in class_names
        assert "SessionResponse" in class_names
        assert "SessionManager" in class_names
        # Enum class should NOT be in classes
        assert "SessionStatus" not in class_names

    def test_class_bases(self, summary: ASTSummary):
        by_name = {c.name: c for c in summary.classes}
        assert "BaseModel" in by_name["SessionRequest"].bases
        assert "BaseModel" in by_name["SessionResponse"].bases

    def test_class_docstring(self, summary: ASTSummary):
        by_name = {c.name: c for c in summary.classes}
        assert by_name["SessionManager"].docstring is not None
        assert "cognitive session" in by_name["SessionManager"].docstring.lower()

    def test_class_methods(self, summary: ASTSummary):
        by_name = {c.name: c for c in summary.classes}
        manager = by_name["SessionManager"]
        method_names = {m.name for m in manager.methods}
        assert "create_session" in method_names
        assert "terminate_session" in method_names
        assert "get_session" in method_names

    def test_async_method_detection(self, summary: ASTSummary):
        by_name = {c.name: c for c in summary.classes}
        manager = by_name["SessionManager"]
        methods_by_name = {m.name: m for m in manager.methods}
        assert methods_by_name["terminate_session"].is_async is True
        assert methods_by_name["create_session"].is_async is False

    def test_endpoints_extracted(self, summary: ASTSummary):
        ep_set = {(e.method, e.path) for e in summary.endpoints}
        assert ("GET", "/health") in ep_set
        assert ("GET", "/{session_id}") in ep_set
        assert ("POST", "/") in ep_set
        assert ("DELETE", "/{session_id}") in ep_set
        assert ("WEBSOCKET", "/ws/stream") in ep_set

    def test_endpoint_count(self, summary: ASTSummary):
        assert len(summary.endpoints) == 5

    def test_endpoint_function_names(self, summary: ASTSummary):
        by_path = {e.path: e for e in summary.endpoints}
        assert by_path["/health"].function_name == "health_check"
        assert by_path["/ws/stream"].function_name == "websocket_stream"

    def test_top_level_functions(self, summary: ASTSummary):
        func_names = {f.name for f in summary.functions}
        assert "health_check" in func_names
        assert "process_batch" in func_names
        assert "_internal_helper" in func_names

    def test_function_params(self, summary: ASTSummary):
        by_name = {f.name: f for f in summary.functions}
        batch = by_name["process_batch"]
        param_str = " ".join(batch.params)
        assert "items: List[str]" in param_str
        assert "timeout: int" in param_str

    def test_function_return_types(self, summary: ASTSummary):
        by_name = {f.name: f for f in summary.functions}
        assert by_name["_internal_helper"].return_type == "str"
        assert by_name["health_check"].return_type == "dict"

    def test_error_handlers_extracted(self, summary: ASTSummary):
        assert len(summary.error_handlers) >= 3
        exc_types_flat = []
        for h in summary.error_handlers:
            exc_types_flat.extend(h.exception_types)
        assert "httpx.TimeoutException" in exc_types_flat
        assert "httpx.ConnectError" in exc_types_flat
        assert "ValueError" in exc_types_flat

    def test_error_handler_status_codes(self, summary: ASTSummary):
        handlers_with_status = [h for h in summary.error_handlers if h.status_code is not None]
        status_codes = {h.status_code for h in handlers_with_status}
        assert 504 in status_codes
        assert 502 in status_codes

    def test_error_handler_enclosing_function(self, summary: ASTSummary):
        for h in summary.error_handlers:
            if "httpx.TimeoutException" in h.exception_types:
                assert h.enclosing_function == "get_session"

    def test_http_calls_extracted(self, summary: ASTSummary):
        assert len(summary.http_calls) >= 3
        methods = {h.call_method for h in summary.http_calls}
        assert "get" in methods
        assert "post" in methods

    def test_http_call_urls(self, summary: ASTSummary):
        urls = {h.url_or_path for h in summary.http_calls if h.url_or_path}
        # f-strings should be captured
        has_internal = any("gaia-core" in (u or "") or "f-string" in (u or "").lower() for u in urls)
        has_mcp = any("gaia-mcp" in (u or "") for u in urls)
        assert has_internal or has_mcp

    def test_http_call_enclosing_function(self, summary: ASTSummary):
        for h in summary.http_calls:
            if h.url_or_path and "mcp" in h.url_or_path:
                assert h.enclosing_function == "process_batch"


class TestASTSummaryOutput:
    """Test serialization and prompt rendering."""

    @pytest.fixture
    def summary(self) -> ASTSummary:
        return summarize_file(FIXTURE_SOURCE, filename="test_sessions.py")

    def test_to_dict_has_all_keys(self, summary: ASTSummary):
        d = summary.to_dict()
        expected_keys = {
            "module_docstring", "classes", "functions", "endpoints",
            "enums", "constants", "gaia_imports", "error_handlers", "http_calls",
        }
        assert set(d.keys()) == expected_keys

    def test_to_dict_serializable(self, summary: ASTSummary):
        import json
        d = summary.to_dict()
        # Should not raise
        json_str = json.dumps(d)
        assert len(json_str) > 0

    def test_to_prompt_text_not_empty(self, summary: ASTSummary):
        text = summary.to_prompt_text()
        assert len(text) > 0
        assert "### File: test_sessions.py" in text

    def test_to_prompt_text_contains_sections(self, summary: ASTSummary):
        text = summary.to_prompt_text()
        assert "**Module:**" in text
        assert "**Constants:**" in text
        assert "**Enums:**" in text
        assert "**Endpoints:**" in text
        assert "**Functions:**" in text
        assert "**Classes:**" in text
        assert "**Error Handlers:**" in text
        assert "**HTTP Calls:**" in text
        assert "**GAIA Imports:**" in text

    def test_summary_compactness(self, summary: ASTSummary):
        """Summary should be significantly shorter than source."""
        source_lines = FIXTURE_SOURCE.strip().count("\n") + 1
        prompt_lines = summary.to_prompt_text().strip().count("\n") + 1
        ratio = prompt_lines / source_lines
        assert ratio < 0.70, f"Summary is {ratio:.0%} of source — expected < 70%"


class TestEdgeCases:
    """Test edge cases and error handling."""

    def test_empty_source(self):
        s = summarize_file("")
        assert s.module_docstring is None
        assert s.classes == []
        assert s.functions == []

    def test_syntax_error_raises(self):
        with pytest.raises(SyntaxError):
            summarize_file("def broken(")

    def test_no_endpoints(self):
        source = """
def plain_function(x: int) -> int:
    return x + 1
"""
        s = summarize_file(source)
        assert s.endpoints == []
        assert len(s.functions) == 1

    def test_nested_class(self):
        source = """
class Outer:
    class Inner:
        pass
    def method(self) -> None:
        pass
"""
        s = summarize_file(source)
        assert len(s.classes) == 1  # Only Outer at module level
        assert s.classes[0].name == "Outer"

    def test_long_constant_string_truncated(self):
        source = f'LONG_STRING = "{"x" * 200}"'
        s = summarize_file(source)
        assert len(s.constants) == 1
        assert len(s.constants[0].value) < 100

    def test_filename_preserved(self):
        s = summarize_file("x = 1", filename="my_module.py")
        assert s.filename == "my_module.py"
