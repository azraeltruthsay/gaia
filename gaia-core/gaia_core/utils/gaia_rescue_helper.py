"""
GAIA Rescue Helper (production copy)
------------------------------------
- Central, Config-safe utilities used by the router and rescue shell
- Provides GAIARescueHelper class (expected by output_router.py)
- Adds legacy function shims so older code keeps working
"""
from __future__ import annotations
import os
import json
import logging
from pathlib import Path
from datetime import datetime
from typing import Dict, Any, Optional, List
import subprocess
from gaia_core.utils.mcp_client import ai_execute as mcp_ai_execute, ai_read as mcp_ai_read, ai_write as mcp_ai_write

try:
    import yaml  # optional; used only if present
except Exception:
    yaml = None

logger = logging.getLogger("GAIA.Helper")

# Core GAIA imports
from gaia_core.config import Config, get_config
from gaia_core.memory.status_tracker import GAIAStatus
try:
    from gaia_core.memory.dev_matrix import GAIADevMatrix  # optional
except Exception:
    GAIADevMatrix = None

# ----------------------------
# small internal utilities
# ----------------------------
def _safe_read_text(path: Path) -> str:
    try:
        with open(path, "r", encoding="utf-8") as f:
            return f.read()
    except Exception as e:
        return f"[IO ERROR: {e}]"

def _find_first_with_ext(base: Path, stem: str, exts: tuple[str, ...]) -> Optional[Path]:
    for ext in exts:
        p = base / f"{stem}{ext}"
        if p.exists():
            return p
    return None

# -------------------------------------------
# GAIARescueHelper (router-facing class)
# -------------------------------------------
class GAIARescueHelper:
    """
    Config-aware façade for:
      - Blueprints / Cheatsheets under knowledge/system_reference/...
      - Sketchpad JSON log
      - Safe shell execution (whitelist)
      - Code reads / spans / symbol searches / stub summaries
    """
    def __init__(self, config: Config, llm: Optional[Any] = None):
        self.config = config
        self.llm = llm  # Optional 'lite' model for summarization
        self.knowledge_root: Path = Path(config.KNOWLEDGE_DIR)
        sysref = self.knowledge_root / "system_reference"
        # Read-only reference directories (from /knowledge)
        self.blueprints_dir: Path  = sysref / "blueprints"
        self.cheatsheets_dir: Path = sysref / "cheatsheets"
        # Writable state files go to SHARED_DIR (mounted as rw)
        shared_dir = Path(getattr(config, 'SHARED_DIR', None) or os.environ.get('SHARED_DIR', '/shared'))
        gaia_state = shared_dir / "gaia_state"
        self.sketchpad_path: Path  = gaia_state / "sketchpad.json"
        self.fragments_path: Path  = gaia_state / "response_fragments.json"
        self.seeds_path: Path      = gaia_state / "thought_seeds" / "queued_reflections.json"
        self.memory_store_path: Path = gaia_state / "agent_memory.json"
        # Ensure writable dirs exist (read-only dirs should already exist)
        try:
            gaia_state.mkdir(parents=True, exist_ok=True)
            self.seeds_path.parent.mkdir(parents=True, exist_ok=True)
        except OSError as e:
            logger.warning(f"Could not create writable state directories: {e}")
        # Allowed file extensions
        self._bp_exts = (".md", ".txt", ".yaml", ".yml", ".json")
        self._cs_exts = (".md", ".txt", ".json")

    # -------------------------
    # Blueprints / Cheatsheets
    # -------------------------
    def load_blueprint(self, blueprint_id: str) -> str:
        blueprint_id = str(blueprint_id).strip()
        p = _find_first_with_ext(self.blueprints_dir, blueprint_id, self._bp_exts)
        if p:
            return _safe_read_text(p)
        return f"[BLUEPRINT NOT FOUND: {blueprint_id}]"

    def load_cheatsheet(self, cheatsheet_id: str) -> str:
        cheatsheet_id = str(cheatsheet_id).strip()
        p = _find_first_with_ext(self.cheatsheets_dir, cheatsheet_id, self._cs_exts)
        if p:
            return _safe_read_text(p)
        # Fallback to the in-memory cheatsheet from Config if available
        try:
            if getattr(self.config, "cheat_sheet", None):
                return json.dumps(self.config.cheat_sheet, indent=2)
        except Exception:
            pass
        return f"[CHEATSHEET NOT FOUND: {cheatsheet_id}]"

    # ---------
    # Sketchpad
    # ---------
    def sketchpad_write(self, title: str, content: str) -> str:
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "title": str(title).strip(),
            "content": str(content),
        }
        # Try MCP read/write for auditing. Fallback to local file I/O if MCP is not available.
        try:
            existing = {"sketchpad": []}
            if self.sketchpad_path.exists():
                r = mcp_ai_read(str(self.sketchpad_path))
                if r.get("ok"):
                    existing = json.loads(r.get("content") or "{}") or {"sketchpad": []}
            existing.setdefault("sketchpad", []).append(entry)
            mcp_res = mcp_ai_write(str(self.sketchpad_path), json.dumps(existing, indent=2))
            if mcp_res.get("ok"):
                return f"✅ Sketch '{entry['title']}' saved."
        except Exception:
            pass

        # Fallback to local write
        data = {"sketchpad": []}
        if self.sketchpad_path.exists():
            try:
                with open(self.sketchpad_path, "r", encoding="utf-8") as f:
                    data = json.load(f) or {"sketchpad": []}
            except Exception:
                data = {"sketchpad": []}
        data.setdefault("sketchpad", []).append(entry)
        with open(self.sketchpad_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2)
        return f"✅ Sketch '{entry['title']}' saved."

    def sketchpad_read(self, key: str = "") -> str:
        if not self.sketchpad_path.exists():
            return "📝 Sketchpad is empty."
        try:
            with open(self.sketchpad_path, "r", encoding="utf-8") as f:
                data = json.load(f)
            sketches = data.get("sketchpad", [])
            if not sketches:
                return "📝 Sketchpad is empty."
            key = str(key or "").strip()
            if not key:
                return "\n".join([f"- [{s['timestamp']}] {s.get('title','')}" for s in sketches[-20:]])
            for s in reversed(sketches):
                if s.get("title", "") == key:
                    return f"[{s['timestamp']}] {s['title']}\n{s.get('content','')}"
            return f"[No sketch titled '{key}']"
        except Exception as e:
            return f"❌ Failed to load sketchpad: {e}"

    def sketchpad_clear(self) -> str:
        # Backup and clear
        try:
            if self.sketchpad_path.exists():
                ts = datetime.utcnow().strftime("%Y%m%d_%H%M%S")
                backup = self.sketchpad_path.with_suffix(self.sketchpad_path.suffix + f".bak.{ts}")
                # Try to copy via MCP
                try:
                    r = mcp_ai_read(str(self.sketchpad_path))
                    if r.get("ok"):
                        mcp_ai_write(str(backup), r.get("content") or "")
                    else:
                        self.sketchpad_path.replace(backup)
                except Exception:
                    try:
                        self.sketchpad_path.replace(backup)
                    except Exception:
                        pass
            # Attempt MCP write for clearing the file
            try:
                mcp_ai_write(str(self.sketchpad_path), json.dumps({"sketchpad": []}, indent=2))
                return "🧹 Sketchpad cleared."
            except Exception:
                with open(self.sketchpad_path, "w", encoding="utf-8") as f:
                    json.dump({"sketchpad": []}, f, indent=2)
                return "🧹 Sketchpad cleared."
        except Exception as e:
            return f"❌ Failed to clear sketchpad: {e}"

    # ----------------------
    # Response Fragmentation
    # ----------------------
    def _load_fragments_store(self) -> Dict[str, Any]:
        """Load the fragments store from disk."""
        data = {"fragments": {}, "pending": []}
        if self.fragments_path.exists():
            try:
                with open(self.fragments_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f)
                    if isinstance(loaded, dict):
                        loaded.setdefault("fragments", {})
                        loaded.setdefault("pending", [])
                        return loaded
            except Exception:
                pass
        return data

    def _save_fragments_store(self, data: Dict[str, Any]) -> bool:
        """Save the fragments store to disk."""
        try:
            with open(self.fragments_path, "w", encoding="utf-8") as f:
                json.dump(data, f, indent=2)
            return True
        except Exception as e:
            logger.error(f"Failed to save fragments store: {e}")
            return False

    def fragment_write(self, parent_request_id: str, sequence: int, content: str,
                       continuation_hint: str = "", is_complete: bool = False,
                       token_count: int = 0) -> Dict[str, Any]:
        """
        Store a response fragment for later assembly.

        Args:
            parent_request_id: UUID linking all fragments from the same request
            sequence: Fragment ordering (0, 1, 2, ...)
            content: The actual text content of this fragment
            continuation_hint: Context for continuation (e.g., "The Raven stanza 10/18")
            is_complete: True if this is the final fragment
            token_count: Approximate token count for this fragment

        Returns:
            Dict with fragment_id and status
        """
        import uuid
        fragment_id = str(uuid.uuid4())

        fragment = {
            "fragment_id": fragment_id,
            "parent_request_id": parent_request_id,
            "sequence": sequence,
            "content": content,
            "continuation_hint": continuation_hint,
            "is_complete": is_complete,
            "token_count": token_count,
            "created_at": datetime.utcnow().isoformat()
        }

        store = self._load_fragments_store()

        # Store by parent_request_id for easy retrieval
        if parent_request_id not in store["fragments"]:
            store["fragments"][parent_request_id] = []
        store["fragments"][parent_request_id].append(fragment)

        # Track pending (incomplete) requests
        if not is_complete and parent_request_id not in store["pending"]:
            store["pending"].append(parent_request_id)
        elif is_complete and parent_request_id in store["pending"]:
            store["pending"].remove(parent_request_id)

        if self._save_fragments_store(store):
            return {"ok": True, "fragment_id": fragment_id, "sequence": sequence}
        return {"ok": False, "error": "Failed to save fragment"}

    def fragment_read(self, parent_request_id: str) -> List[Dict[str, Any]]:
        """
        Retrieve all fragments for a given request, sorted by sequence.

        Args:
            parent_request_id: The UUID linking fragments

        Returns:
            List of fragment dicts, sorted by sequence number
        """
        store = self._load_fragments_store()
        fragments = store["fragments"].get(parent_request_id, [])
        return sorted(fragments, key=lambda f: f.get("sequence", 0))

    def fragment_assemble(self, parent_request_id: str,
                          seam_overlap_check: bool = True) -> Dict[str, Any]:
        """
        Assemble fragments into a complete response.

        Args:
            parent_request_id: The UUID linking fragments
            seam_overlap_check: If True, attempt to detect/remove duplicate text at seams

        Returns:
            Dict with assembled content and metadata
        """
        fragments = self.fragment_read(parent_request_id)

        if not fragments:
            return {"ok": False, "error": "No fragments found", "content": ""}

        # Check completeness
        is_complete = any(f.get("is_complete", False) for f in fragments)

        # Assemble content
        assembled_parts = []
        for i, frag in enumerate(fragments):
            content = frag.get("content", "")

            # Seam overlap detection: check if end of previous fragment
            # overlaps with start of current fragment
            if seam_overlap_check and i > 0 and assembled_parts:
                prev_content = assembled_parts[-1]
                # Look for overlap in last 100 chars of prev and first 100 of current
                overlap_window = 100
                prev_tail = prev_content[-overlap_window:] if len(prev_content) > overlap_window else prev_content
                curr_head = content[:overlap_window] if len(content) > overlap_window else content

                # Simple overlap detection: find if prev_tail ends with something curr_head starts with
                for overlap_len in range(min(len(prev_tail), len(curr_head)), 10, -1):
                    if prev_tail.endswith(curr_head[:overlap_len]):
                        # Remove the overlapping portion from current content
                        content = content[overlap_len:]
                        logger.info(f"Fragment seam: removed {overlap_len} char overlap")
                        break

            assembled_parts.append(content)

        assembled = "".join(assembled_parts)
        total_tokens = sum(f.get("token_count", 0) for f in fragments)

        return {
            "ok": True,
            "content": assembled,
            "fragment_count": len(fragments),
            "total_tokens": total_tokens,
            "is_complete": is_complete,
            "parent_request_id": parent_request_id
        }

    def fragment_clear(self, parent_request_id: Optional[str] = None) -> str:
        """
        Clear fragments. If parent_request_id is provided, clear only that request's fragments.
        Otherwise, clear all fragments.
        """
        store = self._load_fragments_store()

        if parent_request_id:
            if parent_request_id in store["fragments"]:
                del store["fragments"][parent_request_id]
            if parent_request_id in store["pending"]:
                store["pending"].remove(parent_request_id)
            self._save_fragments_store(store)
            return f"🧹 Cleared fragments for request {parent_request_id[:8]}..."
        else:
            store = {"fragments": {}, "pending": []}
            self._save_fragments_store(store)
            return "🧹 Cleared all response fragments."

    def fragment_list_pending(self) -> List[str]:
        """List all pending (incomplete) fragment requests."""
        store = self._load_fragments_store()
        return store.get("pending", [])

    # ---------------
    # Memory helpers
    # ---------------
    def _load_memory_store(self) -> Dict[str, Any]:
        data = {"facts": []}
        if self.memory_store_path.exists():
            # MCP read first for auditability
            try:
                resp = mcp_ai_read(str(self.memory_store_path))
                if resp.get("ok"):
                    loaded = json.loads(resp.get("content") or "{}")
                    if isinstance(loaded, dict):
                        loaded.setdefault("facts", [])
                        return loaded
            except Exception:
                pass
            try:
                with open(self.memory_store_path, "r", encoding="utf-8") as f:
                    loaded = json.load(f) or {"facts": []}
                    loaded.setdefault("facts", [])
                    return loaded
            except Exception:
                pass
        return data

    def _write_memory_store(self, data: Dict[str, Any]) -> None:
        payload = json.dumps(data, indent=2)
        try:
            mcp_ai_write(str(self.memory_store_path), payload)
            return
        except Exception:
            pass
        with open(self.memory_store_path, "w", encoding="utf-8") as f:
            f.write(payload)

    def remember_fact(self, key: str, value: str, note: str = "") -> str:
        key = str(key or "").strip()
        if not key:
            return "⚠️ 'key' is required when calling remember_fact."
        entry = {
            "timestamp": datetime.utcnow().isoformat(),
            "key": key,
            "value": str(value or ""),
            "note": str(note or ""),
        }
        data = self._load_memory_store()
        data.setdefault("facts", []).append(entry)
        self._write_memory_store(data)
        return f"🧠 Stored fact '{key}'."

    def recall_fact(self, key: str = "", limit: int = 5) -> str:
        data = self._load_memory_store()
        facts = data.get("facts") or []
        if not facts:
            return "🧠 Memory store is empty."
        key = str(key or "").strip().lower()
        matches = facts
        if key:
            matches = [f for f in facts if key in f.get("key", "").lower()]
            if not matches:
                return f"🧠 No stored facts matching '{key}'."
        matches = matches[-max(1, int(limit)) :]
        lines = []
        for fact in reversed(matches):
            line = f"[{fact.get('timestamp','?')}] {fact.get('key','(key)')}: {fact.get('value','')}"
            note = fact.get("note")
            if note:
                line += f"\n  note: {note}"
            lines.append(line)
        return "\n".join(lines)

    def get_recent_facts(self, limit: int = 5) -> List[Dict[str, str]]:
        """Return the most recent stored facts as structured data for prompts/UI."""
        limit = max(1, int(limit or 1))
        data = self._load_memory_store()
        facts = data.get("facts") or []
        if not facts:
            return []
        recent = facts[-limit:]
        cleaned: List[Dict[str, str]] = []
        for fact in recent:
            cleaned.append({
                "timestamp": str(fact.get("timestamp", "")),
                "key": str(fact.get("key", "")),
                "value": str(fact.get("value", "")),
                "note": str(fact.get("note", "")),
            })
        return cleaned

    # ------------------
    # Thought seed queue
    # ------------------
    def queue_thought_seed(self, prompt: str, note: str = "", priority: str = "normal") -> str:
        seed = {
            "prompt": str(prompt).strip(),
            "note": note or "Queued from shell",
            "priority": (priority or "normal").lower(),
            "timestamp": datetime.utcnow().isoformat(),
            "source": "shell",
        }
        # Load existing list
        seeds: list = []
        if self.seeds_path.exists():
            try:
                with open(self.seeds_path, "r", encoding="utf-8") as f:
                    seeds = json.load(f)
                if not isinstance(seeds, list):
                    seeds = []
            except Exception as e:
                logger.warning(f"⚠️ Could not load existing seeds: {e}")
                seeds = []
        seeds.append(seed)
        try:
            # Prefer MCP write for seeds
            try:
                mcp_ai_write(str(self.seeds_path), json.dumps(seeds, indent=2))
                logger.info(f"🌱 Thought seed queued: {note or seed['prompt'][:60]}")
                return f"🌱 Thought seed queued for reflection: {note or seed['prompt'][:60]}"
            except Exception:
                with open(self.seeds_path, "w", encoding="utf-8") as f:
                    json.dump(seeds, f, indent=2)
                logger.info(f"🌱 Thought seed queued: {note or seed['prompt'][:60]}")
                return f"🌱 Thought seed queued for reflection: {note or seed['prompt'][:60]}"
        except Exception as e:
            logger.error(f"❌ Failed to save thought seeds: {e}")
            return f"❌ Failed to queue thought seed: {e}"

    # -------------------
    # Safe shell routines
    # -------------------
    def run_shell_safe(self, command: str) -> str:
        """Check first token against whitelist in Config.SAFE_EXECUTE_FUNCTIONS."""
        safe_cmds = set(self.config.SAFE_EXECUTE_FUNCTIONS or [])
        parts = (command or "").strip().split()
        if not parts:
            return "❌ Shell error: Empty command."
        if parts[0] not in safe_cmds:
            return f"❌ Shell error: '{parts[0]}' not in SAFE_EXECUTE_FUNCTIONS."
        try:
            # Prefer MCP execution so sidecar can audit or enforce dry-run policies
            try:
                r = mcp_ai_execute(command, timeout=10, shell=True, dry_run=False)
                if r.get("ok"):
                    return (r.get("stdout") or "").strip() or (r.get("stderr") or "").strip()
            except Exception:
                pass

            res = subprocess.run(
                command, shell=True, check=True,
                capture_output=True, text=True, timeout=10
            )
            return res.stdout.strip() or res.stderr.strip()
        except Exception as e:
            return f"❌ Shell error: {e}"

    def buffer_and_execute_shell(self, content: str) -> None:
        """Find an EXECUTE block, verify whitelist, run, log result, sketch it."""
        import re
        m = re.search(r"EXECUTE:\s*(?:```(?:bash|python)?\s*)?(.+?)(?:```)?$", str(content).strip(), re.DOTALL)
        if not m:
            logger.warning("🛑 EXECUTE block not found or malformed.")
            return
        command = m.group(1).strip()
        safe_cmds = set(self.config.SAFE_EXECUTE_FUNCTIONS or [])
        if not any(command.startswith(func) for func in safe_cmds):
            logger.warning(f"⛔ Unsafe command blocked: {command}")
            return
        stdout = stderr = ""
        try:
            r = mcp_ai_execute(command, timeout=30, shell=True, dry_run=False)
            if r.get("ok"):
                stdout = r.get("stdout") or ""
                stderr = r.get("stderr") or ""
            else:
                from subprocess import Popen, PIPE
                proc = Popen(command, shell=True, stdout=PIPE, stderr=PIPE, text=True)
                stdout, stderr = proc.communicate()
        except Exception:
            from subprocess import Popen, PIPE
            proc = Popen(command, shell=True, stdout=PIPE, stderr=PIPE, text=True)
            stdout, stderr = proc.communicate()
        result = (stdout or "").strip() or (stderr or "").strip()
        GAIAStatus.set("last_command_output", result)
        self.sketchpad_write("ShellCommand", f"EXECUTE: {command}\n\n{result}")

    # -------------
    # Code helpers
    # -------------
    def code_read(self, path: str) -> Dict[str, Any]:
        try:
            rp = os.path.realpath(path)
            if not os.path.exists(rp):
                return {"kind": "code", "path": path, "error": "not found"}
            with open(rp, "r", encoding="utf-8") as f:
                txt = f.read()
            return {"kind": "code", "path": path, "content": txt}
        except Exception as e:
            return {"kind": "code", "path": path, "error": str(e)}

    def code_span(self, path: str, start: int, end: int) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            start = max(1, int(start)); end = min(len(lines), int(end))
            snippet = "".join(f"{i+1:04d}: {lines[i]}" for i in range(start - 1, end))
            return {"kind": "span", "path": path, "start": start, "end": end, "content": snippet}
        except Exception as e:
            return {"kind": "span", "path": path, "error": str(e)}

    def code_symbol(self, path: str, symbol: str) -> Dict[str, Any]:
        try:
            with open(path, "r", encoding="utf-8") as f:
                lines = f.readlines()
            idx = next((i for i, l in enumerate(lines) if str(symbol) in l), -1)
            if idx < 0:
                return {"kind": "symbol", "path": path, "symbol": symbol, "error": "not found"}
            lo = max(0, idx - 5); hi = min(len(lines), idx + 20)
            snippet = "".join(f"{i+1:04d}: {lines[i]}" for i in range(lo, hi))
            return {"kind": "symbol", "path": path, "symbol": symbol, "content": snippet}
        except Exception as e:
            return {"kind": "symbol", "path": path, "symbol": symbol, "error": str(e)}

    def code_summarize(self, src: Dict[str, Any], max_tokens: int = 256) -> Dict[str, Any]:
        """Summarizes a code snippet, using an LLM if available."""
        txt = src.get("content", "")
        if not txt:
            return {"kind": "summary", "content": "[empty]"}

        # If an LLM is available, use it for a proper summary.
        if self.llm:
            try:
                prompt = f"Summarize the following code snippet in a few sentences, focusing on its purpose and primary functions:\n\n```\n{txt[:4000]}\n```"
                res = self.llm.create_chat_completion(
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=max_tokens,
                    temperature=0.3,
                )
                summary = res["choices"][0]["message"]["content"].strip()
                return {"kind": "summary", "content": summary}
            except Exception as e:
                logger.warning(f"⚠️ LLM code summary failed: {e}. Falling back to truncation.")

        # Fallback to simple truncation
        N = max(200, int(max_tokens) * 4)  # rough char proxy
        return {"kind": "summary", "content": f"[TRUNCATED]\n{txt[:N]}..."}

# ------------------------------------------------
# Legacy-style top-level shims (keep older code ok)
# ------------------------------------------------
_SINGLETON: Optional[GAIARescueHelper] = None
def _helper() -> GAIARescueHelper:
    global _SINGLETON
    if _SINGLETON is None:
        _SINGLETON = GAIARescueHelper(get_config())
    return _SINGLETON

# Sketchpad shims
def sketch(title: str, content: str) -> str:
    return _helper().sketchpad_write(title, content)
def sketchpad_write(title: str, content: str) -> str:
    """Alias for sketch() - stores content to sketchpad."""
    return _helper().sketchpad_write(title, content)
def show_sketchpad(key: str = "") -> str:
    return _helper().sketchpad_read(key)
def sketchpad_read(key: str = "") -> str:
    """Alias for show_sketchpad() - reads from sketchpad."""
    return _helper().sketchpad_read(key)
def clear_sketchpad() -> str:
    return _helper().sketchpad_clear()
def sketchpad_clear() -> str:
    """Alias for clear_sketchpad()."""
    return _helper().sketchpad_clear()

# Blueprint/Cheatsheet shims
def load_blueprint(blueprint_id: str) -> str:
    return _helper().load_blueprint(blueprint_id)
def load_cheatsheet(cheatsheet_id: str) -> str:
    return _helper().load_cheatsheet(cheatsheet_id)

# Thought seeds
def queue_thought_seed(prompt: str, note: str = "", priority: str = "normal") -> str:
    return _helper().queue_thought_seed(prompt, note, priority)

# Safe shell shims
def run_shell_safe(command: str) -> str:
    return _helper().run_shell_safe(command)
def buffer_and_execute_shell(content: str) -> None:
    return _helper().buffer_and_execute_shell(content)

# Code shims
def code_read(path: str) -> Dict[str, Any]:
    return _helper().code_read(path)
def code_span(path: str, start: int, end: int) -> Dict[str, Any]:
    return _helper().code_span(path, start, end)
def code_symbol(path: str, symbol: str) -> Dict[str, Any]:
    return _helper().code_symbol(path, symbol)
def code_summarize(src: Dict[str, Any], max_tokens: int = 256) -> Dict[str, Any]:
    return _helper().code_summarize(src, max_tokens=max_tokens)

# Memory shims exposed to the rescue shell / prompt builder
def remember_fact(key: str, value: str, note: str = "") -> str:
    return _helper().remember_fact(key, value, note)

def recall_fact(key: str = "", limit: int = 5) -> str:
    return _helper().recall_fact(key, limit=limit)

def get_recent_facts(limit: int = 5) -> List[Dict[str, str]]:
    return _helper().get_recent_facts(limit=limit)
