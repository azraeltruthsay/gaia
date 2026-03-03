\"\"\"
GAIA Digital Immune System (SIEM-lite) - Smart Edition.

Tracks, consolidates, and triages system errors to provide
a systemic health overview without \"alert fatigue.\"
Also performs proactive \"MRI\" diagnostics for structural integrity.
\"\"\"

from __future__ import annotations

import logging
import re
import hashlib
import math
from collections import Counter
from pathlib import Path
from typing import Dict, List, Any, Optional

logger = logging.getLogger(\"GAIA.ImmuneSystem\")

class ImmuneSystem:
    \"\"\"
    Analyzes system logs and performs proactive diagnostics (MRI) 
    with semantic consolidation and triage.
    \"\"\"
    
    SERVICES = [\"gaia-core\", \"gaia-web\", \"gaia-mcp\", \"gaia-study\", \"gaia-audio\", \"gaia-orchestrator\"]
    
    # Priority weighting for known error patterns
    PRIORITY_MAP = {
        r\"ModuleNotFoundError\": 3.0, # CRITICAL: Missing dependency
        r\"NameResolutionError\": 0.5,  # High volume but often transient/expected
        r\"ConnectionError\": 0.5,
        r\"Permission denied\": 2.0,    # Real structural issue
        r\"not found in configuration\": 1.0, # Configuration gap
        r\"Model path does not exist\": 2.5, # CRITICAL: Missing model file
        r\"Root not allowed\": 0.2,     # Security gate working as intended (Noise)
        r\"timeout\": 0.8,
        r\"uid not found\": 2.0,        # Show-stopper
        r\"cpuinfo\": 0.1,              # Minor dependency noise
    }

    def __init__(self, log_dir: str = \"/logs\"):
        self.log_dir = Path(log_dir)
        
    def get_health_summary(self) -> str:
        \"\"\"
        Returns a smart summary of system health, including proactive diagnostics.
        \"\"\"
        try:
            # 1. Proactive Diagnostic (MRI)
            diagnostic_issues = self._run_diagnostics()
            
            # 2. Log-based Triage
            log_stats = self._scan_and_triage()
            
            if not log_stats and not diagnostic_issues:
                return \"Immune System: UNKNOWN (diagnostics/logs inaccessible)\"
            
            total_unique_issues = sum(len(s[\"issues\"]) for s in log_stats.values()) + len(diagnostic_issues)
            total_raw_events = sum(s[\"raw_count\"] for s in log_stats.values()) + len(diagnostic_issues)
            
            # Calculate systemic score
            systemic_score = sum(s[\"weighted_score\"] for s in log_stats.values())
            # Add scores for diagnostic issues (proactive detection is high priority)
            for issue in diagnostic_issues:
                systemic_score += self._get_priority(issue) * 3.0 

            if total_raw_events == 0:
                return \"Immune System: STABLE. No active irritants.\"
            
            # Determine \"Irritation\" level based on systemic score
            if systemic_score > 25:
                state = \"CRITICAL\"
            elif systemic_score > 8:
                state = \"IRRITATED\"
            elif systemic_score > 2:
                state = \"MINOR NOISE\"
            else:
                state = \"STABLE\"
                
            summary_parts = [f\"Immune System: {state} (Score: {systemic_score:.1f})\"]
            
            # Include diagnostic issues first (Proactive MRI)
            if diagnostic_issues:
                mri_summary = \"; \".join(diagnostic_issues[:3])
                if len(diagnostic_issues) > 3:
                    mri_summary += f\" (+{len(diagnostic_issues)-3} more)\"
                summary_parts.append(f\"MRI: {mri_summary}\")
            
            summary_parts.append(f\"{total_unique_issues} unique issues across {total_raw_events} events\")
            
            # Add top service issues
            for service, data in log_stats.items():
                if data[\"raw_count\"] > 0:
                    top_issue = max(data[\"issues\"].items(), key=lambda x: x[1])[0]
                    short_issue = top_issue[:40] + \"...\" if len(top_issue) > 40 else top_issue
                    summary_parts.append(f\"{service}: {len(data['issues'])} issues ({short_issue})\")
            
            return \" | \".join(summary_parts)
            
        except Exception as e:
            logger.error(f\"Failed to generate smart immune summary: {e}\")
            return \"Immune System: ERROR (triage recommended)\"

    def _run_diagnostics(self) -> List[str]:
        \"\"\"Proactive MRI-like checks for common structural failures.\"\"\"
        issues = []
        
        # 1. Dependency Checks (Module MRI)
        # We check common failure points like llama_cpp, pydantic, etc.
        # This identifies issues before they manifest as cryptic errors deep in the loop.
        required_modules = [\"llama_cpp\", \"pydantic\", \"fastapi\", \"psutil\", \"dataclasses_json\"]
        for mod in required_modules:
            try:
                # Use __import__ to check existence without heavy loading if possible
                __import__(mod)
            except ImportError:
                issues.append(f\"ModuleNotFoundError: '{mod}' missing\")
            except Exception as e:
                issues.append(f\"ModuleLoadError: '{mod}' ({str(e)[:30]})\")
        
        # 2. Model File Checks (Artifact MRI)
        # Verify that models defined in configuration actually exist on disk.
        try:
            from gaia_common.config import get_config
            cfg = get_config()
            model_configs = getattr(cfg, \"MODEL_CONFIGS\", {})
            for name, mcfg in model_configs.items():
                if not mcfg.get(\"enabled\", True):
                    continue
                path_val = mcfg.get(\"path\")
                if path_val:
                    p = Path(path_val)
                    if not p.exists():
                        issues.append(f\"Model path does not exist: {name} ({p.name})\")
        except Exception:
            pass # Best effort, avoids circular imports or boot-time loops

        return issues

    def _normalize_message(self, message: str) -> str:
        \"\"\"
        Strips timestamps, IDs, and addresses to consolidate similar errors.
        \"\"\"
        # Remove typical timestamp patterns [2026-03-02...]
        msg = re.sub(r'\[?\d{4}-\d{2}-\d{2}[T\s]\d{2}:\d{2}:\d{2}[^\]]*\]?', '', message)
        # Remove memory addresses 0x...
        msg = re.sub(r'0x[0-9a-fA-F]+', '0x...', msg)
        # Remove UUIDs
        msg = re.sub(r'[0-9a-f]{8}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{4}-[0-9a-f]{12}', 'UUID', msg)
        # Strip trailing \"...\" and whitespace
        return msg.strip()

    def _get_priority(self, message: str) -> float:
        \"\"\"Calculates weight for a message based on PRIORITY_MAP.\"\"\"
        for pattern, weight in self.PRIORITY_MAP.items():
            if re.search(pattern, message, re.IGNORECASE):
                return weight
        return 1.0 # Default weight

    def _scan_and_triage(self) -> Dict[str, Any]:
        stats = {}
        for service in self.SERVICES:
            log_path = self.log_dir / service / \"error.log\"
            if not log_path.exists():
                log_path = self.log_dir / f\"{service}.log\"
                
            if not log_path.exists():
                continue
                
            try:
                # Read last 500 lines to identify recent persistent issues
                lines = log_path.read_text(errors=\"replace\").splitlines()[-500:]
                error_lines = [l for l in lines if \"ERROR\" in l or \"CRITICAL\" in l]
                
                service_issues = Counter()
                service_score = 0.0
                
                for line in error_lines:
                    # Extract the message part
                    msg = line.split(\":\")[-1].strip() if \":\" in line else line
                    normalized = self._normalize_message(msg)
                    service_issues[normalized] += 1
                
                # Calculate weighted score (unique issues * their priority)
                for issue, count in service_issues.items():
                    # We weight unique issues more than repetitions to avoid count-bloat
                    # score = priority * log(count + 9) -> ensures base weight for single occurrences
                    service_score += self._get_priority(issue) * math.log10(count + 9) 

                stats[service] = {
                    \"raw_count\": len(error_lines),
                    \"issues\": dict(service_issues),
                    \"weighted_score\": service_score
                }
            except Exception:
                pass
                
        return stats

def get_immune_summary(log_dir: str = \"/logs\") -> str:
    return ImmuneSystem(log_dir).get_health_summary()
