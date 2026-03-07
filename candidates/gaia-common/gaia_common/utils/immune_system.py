"""
GAIA Digital Immune System (SIEM-lite) - Smart Edition.

Tracks, consolidates, and triages system errors to provide
a systemic health overview without "alert fatigue."
Also performs proactive "MRI" diagnostics for structural integrity.
"""

from __future__ import annotations

import logging
import re
import math
import json
import py_compile
import subprocess
import threading
import time
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger("GAIA.ImmuneSystem")

# Grit Mode — module-level flag to push past cosmetic irritation
_grit_mode_active = False


def enable_grit_mode():
    global _grit_mode_active
    _grit_mode_active = True
    logger.info("🦷 Grit Mode ENABLED — pushing past irritation for this turn.")


def clear_grit_mode():
    global _grit_mode_active
    _grit_mode_active = False


# Resolved status file path (handles container vs host)
def _get_status_file() -> Path:
    p = Path("/logs/immune_status.json")
    if p.parent.exists():
        return p
    return Path("./logs/immune_status.json")

class ImmuneSystem:
    """
    Analyzes system logs and performs proactive diagnostics (MRI) 
    with semantic consolidation and triage.
    """
    
    SERVICES = ["gaia-core", "gaia-web", "gaia-mcp", "gaia-study", "gaia-audio", "gaia-orchestrator"]
    
    # Priority weighting for known error patterns
    PRIORITY_MAP = {
        r"ModuleNotFoundError": 3.0, # CRITICAL: Missing dependency
        r"NameResolutionError": 0.5,  # High volume but often transient/expected
        r"ConnectionError": 0.5,
        r"Permission denied": 2.0,    # Real structural issue
        r"not found in configuration": 1.0, # Configuration gap
        r"Model path does not exist": 2.5, # CRITICAL: Missing model file
        r"SyntaxError": 4.0,           # CRITICAL: Code compilation failure
        r"LintError": 2.0,             # Structural logic failure (Ruff)
        r"Root not allowed": 0.2,     # Security gate working as intended (Noise)
        r"timeout": 0.8,
        r"uid not found": 2.0,        # Show-stopper
        r"cpuinfo": 0.1,              # Minor dependency noise
    }

    def __init__(self, log_dir: str = "/logs"):
        self.log_dir = Path(log_dir)
        self.syntax_cache_file = self.log_dir / "syntax_cache.json"
        
    def get_health_summary(self, fast: bool = False) -> str:
        """
        Returns a smart summary of system health, including proactive diagnostics.
        If fast=True, attempts to read the background-generated status file first.
        """
        status_file = _get_status_file()
        if fast and status_file.exists():
            try:
                data = json.loads(status_file.read_text())
                # Ensure the data isn't too stale (older than 10 mins)
                if time.time() - data.get("timestamp", 0) < 600:
                    return data.get("summary", "Immune System: STALE")
            except Exception:
                pass

        try:
            # 1. Proactive Diagnostic (MRI)
            diagnostic_issues = self._run_diagnostics()
            
            # 2. Log-based Triage
            log_stats = self._scan_and_triage()
            
            if not log_stats and not diagnostic_issues:
                return "Immune System: UNKNOWN (diagnostics/logs inaccessible)"
            
            total_unique_issues = sum(len(s["issues"]) for s in log_stats.values()) + len(diagnostic_issues)
            total_raw_events = sum(s["raw_count"] for s in log_stats.values()) + len(diagnostic_issues)
            
            # Calculate systemic score
            systemic_score = sum(s["weighted_score"] for s in log_stats.values())
            # Add scores for diagnostic issues (proactive detection is high priority)
            for issue in diagnostic_issues:
                systemic_score += self._get_priority(issue) * 3.0 

            if total_raw_events == 0 and not diagnostic_issues:
                summary = "Immune System: STABLE. No active irritants."
                self._save_status(summary, systemic_score, diagnostic_issues)
                return summary
            
            # Determine "Irritation" level based on systemic score
            if systemic_score > 25:
                state = "CRITICAL"
            elif systemic_score > 8:
                state = "IRRITATED"
            elif systemic_score > 2:
                state = "MINOR NOISE"
            else:
                state = "STABLE"
                
            summary_parts = [f"Immune System: {state} (Score: {systemic_score:.1f})"]
            
            # Include diagnostic issues (Proactive MRI) - ALWAYS if they exist
            if diagnostic_issues:
                # Summary for the text line (limit length)
                mri_summary = "; ".join([i.splitlines()[0] for i in diagnostic_issues[:2]])
                if len(diagnostic_issues) > 2:
                    mri_summary += f" (+{len(diagnostic_issues)-2} more)"
                summary_parts.append(f"MRI: {mri_summary}")
            
            if total_raw_events > 0:
                summary_parts.append(f"{total_unique_issues} unique issues across {total_raw_events} events")
                
                # Add top service issues
                for service, data in log_stats.items():
                    if data["raw_count"] > 0:
                        top_issue = max(data["issues"].items(), key=lambda x: x[1])[0]
                        short_issue = top_issue[:40] + "..." if len(top_issue) > 40 else top_issue
                        summary_parts.append(f"{service}: {len(data['issues'])} issues ({short_issue})")
            else:
                summary_parts.append(f"{len(diagnostic_issues)} proactive structural issues")
            
            summary = " | ".join(summary_parts)
            self._save_status(summary, systemic_score, diagnostic_issues)
            return summary
            
        except Exception as e:
            logger.error(f"Failed to generate smart immune summary: {e}")
            return "Immune System: ERROR (triage recommended)"

    def _save_status(self, summary: str, score: float, diagnostics: List[str]):
        """Save the summary and detailed MRI to a shared status file."""
        try:
            status_file = _get_status_file()
            status_file.parent.mkdir(parents=True, exist_ok=True)
            status_file.write_text(json.dumps({
                "summary": summary,
                "score": score,
                "diagnostics": diagnostics,
                "timestamp": time.time()
            }))
        except Exception:
            pass

    def get_detailed_mri(self) -> List[str]:
        """Returns the full list of diagnostic issues from the latest scan."""
        status_file = _get_status_file()
        if status_file.exists():
            try:
                data = json.loads(status_file.read_text())
                return data.get("diagnostics", [])
            except Exception:
                pass
        return []

    def get_dissonance_report(self) -> Dict[str, List[str]]:
        """
        Detects 'Cognitive Dissonance' - modules where the Candidate
        has diverged from the Live stack.
        """
        dissonance = {}
        target_services = ["gaia-core", "gaia-web", "gaia-mcp", "gaia-study", "gaia-audio"]
        
        project_root = Path("/gaia/GAIA_Project")
        for svc in target_services:
            prod_path = project_root / svc
            cand_path = project_root / "candidates" / svc
            
            if not prod_path.exists() or not cand_path.exists():
                continue
                
            svc_dissonance = []
            # Scan all python files in PROD
            for prod_file in prod_path.rglob("*.py"):
                if any(p in str(prod_file) for p in ["venv", "__pycache__", ".pytest_cache", ".ruff_cache"]):
                    continue
                    
                try:
                    relative_path = prod_file.relative_to(prod_path)
                    corresponding_cand = cand_path / relative_path
                    
                    if not corresponding_cand.exists():
                        svc_dissonance.append(f"Missing in CAND: {relative_path}")
                        continue
                        
                    # Compare content hashes
                    import hashlib
                    prod_hash = hashlib.sha256(prod_file.read_bytes()).hexdigest()
                    cand_hash = hashlib.sha256(corresponding_cand.read_bytes()).hexdigest()
                    
                    if prod_hash != cand_hash:
                        svc_dissonance.append(f"Diverged: {relative_path}")
                except Exception:
                    pass
            
            if svc_dissonance:
                dissonance[svc] = svc_dissonance
                
        return dissonance

    def _run_diagnostics(self) -> List[str]:
        """Proactive MRI-like checks for common structural failures."""
        issues = []
        
        # 1. Dependency Checks (Module MRI)
        required_modules = ["llama_cpp", "pydantic", "fastapi", "psutil", "dataclasses_json"]
        for mod in required_modules:
            try:
                __import__(mod)
            except ImportError:
                issues.append(f"ModuleNotFoundError: '{mod}' missing")
            except Exception as e:
                issues.append(f"ModuleLoadError: '{mod}' ({str(e)[:30]})")
        
        # 2. Model File Checks (Artifact MRI)
        try:
            from gaia_common.config import get_config
            cfg = get_config()
            model_configs = getattr(cfg, "MODEL_CONFIGS", {})
            for name, mcfg in model_configs.items():
                if not mcfg.get("enabled", True):
                    continue
                path_val = mcfg.get("path")
                if path_val:
                    p = Path(path_val)
                    if not p.exists():
                        issues.append(f"Model path does not exist: {name} ({p.name})")
        except Exception:
            pass

        # 3. Continuous Syntax & Logic Checks
        try:
            syntax_issues = self._run_syntax_checks()
            issues.extend(syntax_issues)
        except Exception as e:
            logger.debug(f"Syntax checks failed: {e}")

        return issues

    def _run_syntax_checks(self) -> List[str]:
        """Checks Python files for syntax errors and logic errors."""
        cache_file = Path("/logs/syntax_cache.json")
        if not cache_file.parent.exists():
            cache_file = Path("./logs/syntax_cache.json")
            
        cache = {}
        if cache_file.exists():
            try:
                cache = json.loads(cache_file.read_text())
            except Exception:
                pass

        # We keep track of ALL current issues in a separate persistent file
        issues_file = cache_file.parent / "current_structural_issues.json"
        current_issues = {}
        if issues_file.exists():
            try:
                current_issues = json.loads(issues_file.read_text())
            except Exception:
                pass

        project_root = Path("/gaia/GAIA_Project")
        if not project_root.exists():
            project_root = Path(".")

        # Targeted scan of GAIA services only (avoid 33k+ files in SDKs/archives)
        search_dirs = []
        target_services = ["gaia-core", "gaia-web", "gaia-mcp", "gaia-study", "gaia-audio", "gaia-common", "gaia-orchestrator", "gaia-doctor"]
        for svc in target_services:
            svc_path = project_root / svc
            if svc_path.exists():
                search_dirs.append((svc_path, "[PROD]"))
            # Candidates are dev-only and should not affect live health score
            # cand_path = project_root / "candidates" / svc
            # if cand_path.exists():
            #     search_dirs.append((cand_path, "[CAND]"))

        cache_updated = False
        
        for base_dir, env_tag in search_dirs:
            for py_file in base_dir.rglob("*.py"):
                # Strict exclusion of non-source and giant library folders
                parts = [p.lower() for p in py_file.parts]
                if any(p in parts for p in [".venv", "venv", "__pycache__", ".git", "archive", "google-cloud-sdk", "artifacts"]):
                    continue
                # Also skip venv_notebooklm specifically
                if "venv_notebooklm" in parts:
                    continue
                
                try:
                    file_key = str(py_file)
                    mtime = py_file.stat().st_mtime
                    
                    if cache.get(file_key) != mtime:
                        file_issues = []
                        
                        # 1. Bytecode compilation (Syntax check)
                        try:
                            py_compile.compile(str(py_file), doraise=True)
                        except py_compile.PyCompileError as e:
                            file_issues.append(f"{env_tag} SyntaxError in {py_file} (Line ?): {str(e)[:100]}")
                        
                        # 2. Ruff check (Logic/Reference check)
                        if not file_issues:
                            try:
                                res = subprocess.run(
                                    ["ruff", "check", "--select", "F", "--output-format", "json", str(py_file)],
                                    capture_output=True, text=True, timeout=5
                                )
                                if res.returncode != 0:
                                    try:
                                        ruff_data = json.loads(res.stdout)
                                        for entry in ruff_data:
                                            msg = entry.get("message", "Logic error")
                                            line = entry.get("location", {}).get("row", "?")
                                            code = entry.get("code", "F")
                                            
                                            snippet = ""
                                            try:
                                                lines = py_file.read_text().splitlines()
                                                idx = int(line) - 1
                                                start = max(0, idx - 1)
                                                end = min(len(lines), idx + 2)
                                                snippet = "\n".join([f"L{i+1}: {lines[i]}" for i in range(start, end)])
                                            except Exception: pass
                                                
                                            file_issues.append(
                                                f"LintError[{code}] in {py_file} (Line {line}): {msg}\nSnippet:\n{snippet}"
                                            )
                                    except Exception:
                                        file_issues.append(f"LintError in {py_file}: Logic error detected")
                            except Exception: pass

                        if file_issues:
                            current_issues[file_key] = file_issues
                        else:
                            current_issues.pop(file_key, None)
                            
                        cache[file_key] = mtime
                        cache_updated = True
                except Exception:
                    pass

        if cache_updated:
            try:
                cache_file.write_text(json.dumps(cache))
                issues_file.write_text(json.dumps(current_issues))
            except Exception:
                pass
                
        all_flattened_issues = []
        for file_issues in current_issues.values():
            all_flattened_issues.extend(file_issues)
            
        return all_flattened_issues

    def _normalize_message(self, message: str) -> str:
        """Strips timestamps, IDs, and addresses."""
        msg = re.sub(r'\[?\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}[^\]]*\]?', '', message)
        msg = re.sub(r'0x[0-9a-fA-F]+', '0x...', msg)
        msg = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', 'UUID', msg)
        return msg.strip()

    def _get_priority(self, message: str) -> float:
        """Calculates weight for a message based on PRIORITY_MAP."""
        for pattern, weight in self.PRIORITY_MAP.items():
            if re.search(pattern, message, re.IGNORECASE):
                return weight
        return 1.0

    def _scan_and_triage(self) -> Dict[str, Any]:
        stats = {}
        for service in self.SERVICES:
            log_path = self.log_dir / service / "error.log"
            if not log_path.exists():
                log_path = self.log_dir / f"{service}.log"
            if not log_path.exists():
                continue
            try:
                lines = log_path.read_text(errors="replace").splitlines()[-500:]
                error_lines = [l for l in lines if "ERROR" in l or "CRITICAL" in l]
                service_issues = Counter()
                service_score = 0.0
                for line in error_lines:
                    msg = line.split(":")[-1].strip() if ":" in line else line
                    normalized = self._normalize_message(msg)
                    service_issues[normalized] += 1
                for issue, count in service_issues.items():
                    service_score += self._get_priority(issue) * math.log10(count + 9) 
                stats[service] = {
                    "raw_count": len(error_lines),
                    "issues": dict(service_issues),
                    "weighted_score": service_score
                }
            except Exception: pass
        return stats

class BackgroundImmuneSystem:
    """Daemon that monitors system health with a dynamic frequency."""
    MIN_INTERVAL = 30.0
    MAX_INTERVAL = 300.0
    def __init__(self, log_dir: str = "/logs"):
        self.immune_system = ImmuneSystem(log_dir)
        self._stop_event = threading.Event()
        self._thread: Optional[threading.Thread] = None
        self.last_score = 0.0
    def start(self):
        if self._thread is not None: return
        self._thread = threading.Thread(target=self._run, daemon=True, name="ImmuneSystemDaemon")
        self._thread.start()
        logger.info("Background Immune System started.")
    def stop(self):
        self._stop_event.set()
        if self._thread: self._thread.join(timeout=5)
        self._thread = None
    def _run(self):
        while not self._stop_event.is_set():
            try:
                self.immune_system.get_health_summary(fast=False)
                status_file = _get_status_file()
                if status_file.exists():
                    try:
                        data = json.loads(status_file.read_text())
                        self.last_score = data.get("score", 0.0)
                    except Exception: pass
                interval = max(self.MIN_INTERVAL, self.MAX_INTERVAL / (self.last_score + 1))
                if self.last_score > 8:
                    logger.warning("Immune system IRRITATED (score=%.1f). Next check in %.1fs", self.last_score, interval)
                for _ in range(int(interval)):
                    if self._stop_event.is_set(): break
                    time.sleep(1)
            except Exception as e:
                logger.error("Error in background immune system loop: %s", e)
                time.sleep(30)

_bg_immune_system = None
def start_background_immune_system(log_dir: str = "/logs"):
    global _bg_immune_system
    if _bg_immune_system is None:
        _bg_immune_system = BackgroundImmuneSystem(log_dir)
        _bg_immune_system.start()
def get_immune_summary(log_dir: str = "/logs", fast: bool = True) -> str:
    return ImmuneSystem(log_dir).get_health_summary(fast=fast)

def get_detailed_mri(log_dir: str = "/logs") -> List[str]:
    """Module-level helper to get the latest detailed MRI report."""
    return ImmuneSystem(log_dir).get_detailed_mri()

def is_system_irritated(threshold: float = 8.0) -> bool:
    """Returns True if the systemic irritation score is above the threshold."""
    if _grit_mode_active:
        try:
            status_file = _get_status_file()
            if status_file.exists():
                data = json.loads(status_file.read_text())
                score = data.get("score", 0.0)
                logger.info("🦷 Grit Mode active — reporting NOT irritated (actual score: %.1f)", score)
        except Exception:
            pass
        return False
    try:
        status_file = _get_status_file()
        if status_file.exists():
            data = json.loads(status_file.read_text())
            # Data must be reasonably fresh (10 mins)
            if time.time() - data.get("timestamp", 0) < 600:
                return data.get("score", 0.0) >= threshold
    except Exception:
        pass
    return False

