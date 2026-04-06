"""Browser Tools — Web interaction via Playwright CDP and httpx fallback.

Provides GAIA with browser control capabilities inspired by OpenClaw's
methodology but implemented within gaia-mcp's sandboxed environment with
approval workflows.

Three tiers of capability:
  1. HTTP fetch + parse (always available via httpx + BeautifulSoup)
  2. Playwright headless (when installed — full browser control)
  3. CDP direct (when connected to an existing browser instance)

Tool actions:
  - browse: Navigate to a URL and return page content/snapshot
  - click: Click an element by selector or text
  - type: Type text into an input field
  - snapshot: Get accessibility tree or DOM snapshot
  - screenshot: Capture page screenshot
  - evaluate: Run JavaScript in page context

Security:
  - All actions go through MCP approval workflow
  - URL allowlist/blocklist enforced
  - No access to local file:// URLs
  - JavaScript evaluation sandboxed to page context
  - Browser runs headless in container (no display access)
"""

import logging
import re
from typing import Dict, List

logger = logging.getLogger("GAIA.BrowserTools")

# URL security
_BLOCKED_SCHEMES = {"file", "ftp", "data", "javascript"}
_BLOCKED_HOSTS = {"localhost", "127.0.0.1", "0.0.0.0", "169.254.169.254"}  # metadata service
_ALLOWED_SCHEMES = {"http", "https"}


def _validate_url(url: str) -> str:
    """Validate and sanitize a URL. Raises ValueError if unsafe."""
    from urllib.parse import urlparse
    parsed = urlparse(url)

    if parsed.scheme not in _ALLOWED_SCHEMES:
        raise ValueError(f"Blocked URL scheme: {parsed.scheme}")

    if parsed.hostname and parsed.hostname in _BLOCKED_HOSTS:
        raise ValueError(f"Blocked host: {parsed.hostname}")

    # Block internal Docker network
    if parsed.hostname and (
        parsed.hostname.startswith("gaia-") or
        parsed.hostname.endswith(".internal")
    ):
        raise ValueError(f"Blocked internal host: {parsed.hostname}")

    return url


# ── Tier 1: HTTP Fetch + Parse (always available) ─────────────────────

def _http_browse(url: str, extract: str = "text") -> Dict:
    """Fetch a URL via httpx and parse with BeautifulSoup.

    Args:
        url: Target URL
        extract: "text" (readable content), "links" (all links),
                 "forms" (form elements), "meta" (title, description)

    Returns:
        Dict with content, title, links, forms as appropriate.
    """
    import httpx
    from bs4 import BeautifulSoup

    url = _validate_url(url)

    try:
        resp = httpx.get(url, timeout=15, follow_redirects=True, headers={
            "User-Agent": "GAIA/1.0 (Sovereign AI Research Assistant)",
        })
        resp.raise_for_status()
    except httpx.HTTPError as e:
        return {"ok": False, "error": f"HTTP error: {e}"}

    soup = BeautifulSoup(resp.text, "html.parser")

    # Remove script/style noise
    for tag in soup(["script", "style", "nav", "footer", "header"]):
        tag.decompose()

    result = {
        "ok": True,
        "url": str(resp.url),
        "status": resp.status_code,
        "title": soup.title.string.strip() if soup.title else "",
    }

    if extract == "text":
        text = soup.get_text(separator="\n", strip=True)
        # Collapse multiple newlines
        text = re.sub(r'\n{3,}', '\n\n', text)
        result["content"] = text[:8000]
        result["content_length"] = len(text)
        result["truncated"] = len(text) > 8000

    elif extract == "links":
        links = []
        for a in soup.find_all("a", href=True):
            href = a["href"]
            text = a.get_text(strip=True)
            if href.startswith("http") and text:
                links.append({"text": text[:100], "href": href})
        result["links"] = links[:50]

    elif extract == "forms":
        forms = []
        for form in soup.find_all("form"):
            fields = []
            for inp in form.find_all(["input", "textarea", "select"]):
                fields.append({
                    "type": inp.get("type", "text"),
                    "name": inp.get("name", ""),
                    "id": inp.get("id", ""),
                    "placeholder": inp.get("placeholder", ""),
                })
            forms.append({
                "action": form.get("action", ""),
                "method": form.get("method", "GET"),
                "fields": fields,
            })
        result["forms"] = forms[:10]

    elif extract == "meta":
        meta = {}
        for tag in soup.find_all("meta"):
            name = tag.get("name", tag.get("property", ""))
            content = tag.get("content", "")
            if name and content:
                meta[name] = content[:200]
        result["meta"] = meta

    elif extract == "accessibility":
        # Lightweight accessibility tree — like OpenClaw's snapshot mode
        result["tree"] = _extract_accessibility_tree(soup)

    return result


def _extract_accessibility_tree(soup) -> List[Dict]:
    """Extract a simplified accessibility tree from parsed HTML.

    Similar to OpenClaw's aria-ref system — gives the LLM a structured
    view of interactive elements without raw HTML noise.
    """
    elements = []
    idx = 0

    # Interactive elements
    for tag in soup.find_all(["a", "button", "input", "textarea", "select", "label", "h1", "h2", "h3", "h4", "img"]):
        elem = {
            "ref": idx,
            "tag": tag.name,
            "text": tag.get_text(strip=True)[:100],
        }

        if tag.name == "a":
            elem["href"] = tag.get("href", "")[:200]
            elem["role"] = "link"
        elif tag.name == "button":
            elem["role"] = "button"
        elif tag.name == "input":
            elem["role"] = "input"
            elem["type"] = tag.get("type", "text")
            elem["name"] = tag.get("name", "")
            elem["value"] = tag.get("value", "")[:50]
            elem["placeholder"] = tag.get("placeholder", "")
        elif tag.name == "textarea":
            elem["role"] = "textarea"
            elem["name"] = tag.get("name", "")
        elif tag.name == "select":
            elem["role"] = "dropdown"
            elem["name"] = tag.get("name", "")
            elem["options"] = [o.get_text(strip=True) for o in tag.find_all("option")][:10]
        elif tag.name in ("h1", "h2", "h3", "h4"):
            elem["role"] = "heading"
            elem["level"] = int(tag.name[1])
        elif tag.name == "img":
            elem["role"] = "image"
            elem["alt"] = tag.get("alt", "")
            elem["src"] = tag.get("src", "")[:200]

        # ARIA attributes
        for attr in ["aria-label", "aria-role", "aria-expanded", "aria-checked"]:
            val = tag.get(attr)
            if val:
                elem[attr.replace("aria-", "")] = val

        elements.append(elem)
        idx += 1

    return elements


# ── Tier 2: Playwright (when available) ────────────────────────────────

_playwright_available = False
try:
    from playwright.sync_api import sync_playwright
    _playwright_available = True
except ImportError:
    pass

_browser_context = None


def _get_browser():
    """Get or create a persistent browser context."""
    global _browser_context
    if _browser_context is not None:
        return _browser_context

    if not _playwright_available:
        return None

    pw = sync_playwright().start()
    browser = pw.chromium.launch(
        headless=True,
        args=[
            "--no-sandbox",
            "--disable-dev-shm-usage",
            "--disable-gpu",
            "--disable-extensions",
        ],
    )
    _browser_context = browser.new_context(
        user_agent="GAIA/1.0 (Sovereign AI Research Assistant)",
        viewport={"width": 1280, "height": 720},
    )
    logger.info("Playwright browser context created (headless Chromium)")
    return _browser_context


def _playwright_browse(url: str, extract: str = "text") -> Dict:
    """Full browser navigation via Playwright."""
    ctx = _get_browser()
    if not ctx:
        return {"ok": False, "error": "Playwright not available"}

    url = _validate_url(url)
    page = ctx.new_page()

    try:
        page.goto(url, wait_until="domcontentloaded", timeout=15000)

        result = {
            "ok": True,
            "url": page.url,
            "title": page.title(),
        }

        if extract == "text":
            # Get visible text content
            text = page.evaluate("() => document.body.innerText")
            result["content"] = text[:8000]
            result["truncated"] = len(text) > 8000

        elif extract == "accessibility":
            # Playwright's built-in accessibility tree
            tree = page.accessibility.snapshot()
            result["tree"] = tree

        elif extract == "screenshot":
            screenshot = page.screenshot(type="png")
            import base64
            result["screenshot_base64"] = base64.b64encode(screenshot).decode()
            result["screenshot_size"] = len(screenshot)

        return result

    except Exception as e:
        return {"ok": False, "error": f"Browser error: {e}"}
    finally:
        page.close()


def _playwright_click(url: str, selector: str = "", text: str = "") -> Dict:
    """Click an element on a page."""
    ctx = _get_browser()
    if not ctx:
        return {"ok": False, "error": "Playwright not available"}

    url = _validate_url(url)
    page = ctx.new_page()

    try:
        page.goto(url, wait_until="domcontentloaded", timeout=15000)

        if text:
            page.get_by_text(text, exact=False).first.click(timeout=5000)
        elif selector:
            page.click(selector, timeout=5000)
        else:
            return {"ok": False, "error": "Must provide 'selector' or 'text' to click"}

        page.wait_for_load_state("domcontentloaded", timeout=5000)

        return {
            "ok": True,
            "url": page.url,
            "title": page.title(),
            "content": page.evaluate("() => document.body.innerText")[:4000],
        }
    except Exception as e:
        return {"ok": False, "error": f"Click failed: {e}"}
    finally:
        page.close()


def _playwright_type(url: str, selector: str, text: str) -> Dict:
    """Type text into an input field."""
    ctx = _get_browser()
    if not ctx:
        return {"ok": False, "error": "Playwright not available"}

    url = _validate_url(url)
    page = ctx.new_page()

    try:
        page.goto(url, wait_until="domcontentloaded", timeout=15000)
        page.fill(selector, text, timeout=5000)

        return {"ok": True, "url": page.url, "typed": text[:100]}
    except Exception as e:
        return {"ok": False, "error": f"Type failed: {e}"}
    finally:
        page.close()


# ── Public Tool Interface ──────────────────────────────────────────────

def browser_tool(params: dict) -> dict:
    """Main entry point for the browser domain tool.

    Actions:
        browse:     Navigate to URL, return content/snapshot
        click:      Click element by selector or text
        type:       Type into input field
        snapshot:   Get accessibility tree of a page
        screenshot: Capture page screenshot (Playwright only)

    Params vary by action — see each action's docstring.
    """
    action = params.get("action", "browse")
    url = params.get("url", "")

    if not url:
        return {"ok": False, "error": "url is required"}

    try:
        _validate_url(url)
    except ValueError as e:
        return {"ok": False, "error": str(e)}

    # Route to appropriate tier
    use_playwright = _playwright_available and params.get("full_browser", False)

    if action == "browse":
        extract = params.get("extract", "text")
        if use_playwright:
            return _playwright_browse(url, extract)
        return _http_browse(url, extract)

    elif action == "snapshot":
        if use_playwright:
            return _playwright_browse(url, "accessibility")
        return _http_browse(url, "accessibility")

    elif action == "click":
        if not use_playwright:
            return {"ok": False, "error": "Click requires Playwright (full_browser=true)"}
        return _playwright_click(url,
                                  selector=params.get("selector", ""),
                                  text=params.get("text", ""))

    elif action == "type":
        if not use_playwright:
            return {"ok": False, "error": "Type requires Playwright (full_browser=true)"}
        return _playwright_type(url,
                                 selector=params.get("selector", ""),
                                 text=params.get("text", ""))

    elif action == "screenshot":
        if not use_playwright:
            return {"ok": False, "error": "Screenshot requires Playwright (full_browser=true)"}
        return _playwright_browse(url, "screenshot")

    elif action == "links":
        return _http_browse(url, "links")

    elif action == "forms":
        return _http_browse(url, "forms")

    elif action == "meta":
        return _http_browse(url, "meta")

    else:
        return {"ok": False, "error": f"Unknown action: {action}"}
