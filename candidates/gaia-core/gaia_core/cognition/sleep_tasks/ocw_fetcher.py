"""
OCW Fetcher — MIT OpenCourseWare content ingestion for GAIA sleep cycles.

Fetches and normalizes MIT OCW course pages via MCP web_fetch (JSON-RPC).
Designed to run as a sleep task, respecting rate limits and license compliance.

All content fetched is CC BY-NC-SA 4.0 licensed (MIT OCW standard license).
"""

import json
import logging
import re
from typing import Any, Dict, List
from urllib.request import Request, urlopen

logger = logging.getLogger("GAIA.OCWFetcher")

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------

OCW_BASE = "https://ocw.mit.edu"

PHASE_A_COURSES = [
    {
        "course_id": "6-858-computer-systems-security-fall-2014",
        "department": "6",
        "title": "Computer Systems Security",
        "license": "CC BY-NC-SA 4.0",
        "pages": ["syllabus", "lecture-notes", "assignments"],
    },
]

_DEFAULT_MCP_ENDPOINT = "http://gaia-mcp:8765/jsonrpc"


# ---------------------------------------------------------------------------
# MCP JSON-RPC helper
# ---------------------------------------------------------------------------

def _mcp_call(
    method: str,
    params: dict,
    endpoint: str = _DEFAULT_MCP_ENDPOINT,
) -> dict:
    """Make a JSON-RPC 2.0 call to gaia-mcp."""
    body = json.dumps({
        "jsonrpc": "2.0",
        "method": method,
        "params": params,
        "id": 1,
    }).encode()
    req = Request(endpoint, data=body, method="POST")
    req.add_header("Content-Type", "application/json")
    with urlopen(req, timeout=30) as resp:
        result = json.loads(resp.read())
    if "error" in result:
        raise RuntimeError(
            f"MCP error: {result['error'].get('message', result['error'])}"
        )
    return result.get("result", result)


# ---------------------------------------------------------------------------
# License compliance
# ---------------------------------------------------------------------------

_ACCEPTED_LICENSES = {
    "CC BY-NC-SA 4.0",
    "CC BY-NC-SA 3.0",
    "CC BY-SA 4.0",
    "CC BY 4.0",
}


def check_license_compliance(course_metadata: dict) -> Dict[str, Any]:
    """Verify that the course has an acceptable Creative Commons license.

    Args:
        course_metadata: Dict with at least a 'license' key.

    Returns:
        Dict with keys: compliant (bool), license (str), reason (str).
    """
    license_str = (course_metadata.get("license") or "").strip()
    if not license_str:
        return {
            "compliant": False,
            "license": "",
            "reason": "No license information found in course metadata.",
        }
    if license_str in _ACCEPTED_LICENSES:
        return {
            "compliant": True,
            "license": license_str,
            "reason": f"License '{license_str}' is in the accepted list.",
        }
    # Partial match for variations like "CC BY-NC-SA 4.0 International"
    for accepted in _ACCEPTED_LICENSES:
        if accepted in license_str:
            return {
                "compliant": True,
                "license": license_str,
                "reason": f"License contains accepted variant '{accepted}'.",
            }
    return {
        "compliant": False,
        "license": license_str,
        "reason": f"License '{license_str}' is not in the accepted list.",
    }


# ---------------------------------------------------------------------------
# Course manifest
# ---------------------------------------------------------------------------

def get_course_manifest(course_id: str) -> List[Dict[str, str]]:
    """Return list of fetchable pages for a course from PHASE_A_COURSES.

    Each entry has keys: course_id, page, url.
    Falls back to a default page list if the course is not in PHASE_A_COURSES.
    """
    default_pages = ["syllabus", "lecture-notes", "assignments"]
    pages = default_pages

    for course in PHASE_A_COURSES:
        if course["course_id"] == course_id:
            pages = course.get("pages", default_pages)
            break

    return [
        {
            "course_id": course_id,
            "page": page,
            "url": f"{OCW_BASE}/courses/{course_id}/pages/{page}",
        }
        for page in pages
    ]


# ---------------------------------------------------------------------------
# Content normalization
# ---------------------------------------------------------------------------

# Patterns that match OCW navigation/chrome elements
_OCW_CHROME_PATTERNS = [
    # Navigation breadcrumbs
    re.compile(r"MIT OpenCourseWare\s*[»>].*?(?=\n)", re.IGNORECASE),
    # Course info sidebar patterns
    re.compile(r"As Taught In:.*?\n", re.IGNORECASE),
    re.compile(r"Level:\s*(Undergraduate|Graduate).*?\n", re.IGNORECASE),
    # Footer/legal boilerplate
    re.compile(
        r"(MIT OpenCourseWare makes the materials|"
        r"Your use of the MIT OpenCourseWare|"
        r"No enrollment or registration|"
        r"Freely browse and use|"
        r"Made for sharing).*?(?=\n\n|\Z)",
        re.IGNORECASE | re.DOTALL,
    ),
    # "Send feedback" / "Cite this" / "Download" links
    re.compile(r"(Send feedback|Cite this|Download .+?)\s*\n", re.IGNORECASE),
    # Repeated whitespace / separator lines
    re.compile(r"\n{3,}"),
]


def normalize_ocw_content(raw_text: str) -> str:
    """Strip OCW navigation chrome and extract main educational content.

    Args:
        raw_text: Raw text extracted from an OCW page (via web_fetch).

    Returns:
        Cleaned content string with navigation, footers, and chrome removed.
    """
    if not raw_text:
        return ""

    text = raw_text

    # Apply chrome-stripping patterns
    for pattern in _OCW_CHROME_PATTERNS:
        text = pattern.sub("", text)

    # Collapse multiple blank lines to double newline
    text = re.sub(r"\n{3,}", "\n\n", text)

    # Strip leading/trailing whitespace
    text = text.strip()

    return text


# ---------------------------------------------------------------------------
# Page fetcher
# ---------------------------------------------------------------------------

def fetch_course_page(
    course_id: str,
    page: str,
    mcp_endpoint: str = _DEFAULT_MCP_ENDPOINT,
) -> Dict[str, Any]:
    """Fetch a single OCW course page via MCP web_fetch.

    Args:
        course_id: OCW course identifier (e.g. '6-858-computer-systems-security-fall-2014').
        page: Page slug (e.g. 'syllabus', 'lecture-notes', 'assignments').
        mcp_endpoint: JSON-RPC endpoint for gaia-mcp.

    Returns:
        Dict with keys: ok (bool), title (str), content (str), url (str),
        license (str), error (str, only on failure).
    """
    url = f"{OCW_BASE}/courses/{course_id}/pages/{page}"

    # Look up course metadata for license info
    license_str = ""
    for course in PHASE_A_COURSES:
        if course["course_id"] == course_id:
            license_str = course.get("license", "")
            # Verify license compliance before fetching
            compliance = check_license_compliance(course)
            if not compliance["compliant"]:
                return {
                    "ok": False,
                    "url": url,
                    "license": license_str,
                    "error": f"License non-compliant: {compliance['reason']}",
                }
            break

    # Call MCP web_fetch via JSON-RPC
    try:
        result = _mcp_call(
            method="web_fetch",
            params={"url": url},
            endpoint=mcp_endpoint,
        )
    except Exception as exc:
        logger.warning("OCW fetch failed for %s/%s: %s", course_id, page, exc)
        return {
            "ok": False,
            "url": url,
            "license": license_str,
            "error": f"MCP web_fetch failed: {exc}",
        }

    # Handle MCP-level errors (domain not allowed, rate limit, etc.)
    if isinstance(result, dict) and not result.get("ok", True):
        return {
            "ok": False,
            "url": url,
            "license": license_str,
            "error": result.get("error", "Unknown fetch error"),
        }

    # Extract and normalize content
    raw_content = result.get("content", "") if isinstance(result, dict) else ""
    title = result.get("title", "") if isinstance(result, dict) else ""
    content = normalize_ocw_content(raw_content)

    if not content:
        return {
            "ok": False,
            "url": url,
            "license": license_str,
            "error": "No content extracted after normalization.",
        }

    return {
        "ok": True,
        "title": title,
        "content": content,
        "url": url,
        "license": license_str,
    }
