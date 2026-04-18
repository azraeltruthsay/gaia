"""
Skill Gateway — routes the 5 meta-verbs to existing GAIA infrastructure.

Meta-verbs:
  search(query)              → Knowledge Router + skill discovery
  do(skill, input)           → KNOWLEDGE or PLAYBOOK skill execution
  learn(task, result, success) → Utility scoring + skill creation trigger
  remember(fact)             → MemPalace storage
  ask(question)              → Human input request

This is the single entry point for all skill operations, inspired by
Memento-Skills' SkillGateway pattern. It wraps the existing SkillManager,
execute_limb, and MemPalace systems behind a clean 5-verb interface.
"""

import json
import logging
import time
from pathlib import Path
from typing import Any, Dict, Optional

logger = logging.getLogger("GAIA.SkillGateway")

# Lazy imports to avoid circular dependencies at module level
_gateway_instance: Optional["SkillGateway"] = None


def get_gateway() -> "SkillGateway":
    """Get or create the singleton SkillGateway."""
    global _gateway_instance
    if _gateway_instance is None:
        _gateway_instance = SkillGateway()
    return _gateway_instance


class SkillGateway:
    """Routes meta-verb calls to existing GAIA infrastructure."""

    META_VERBS = {"search", "do", "learn", "remember", "ask"}

    def __init__(self, skills_dir: Optional[Path] = None):
        from gaia_mcp.skill_package import load_all_packages

        self._skills_dir = skills_dir or Path("/knowledge/skills")
        self._packages = load_all_packages(self._skills_dir)
        logger.info("SkillGateway initialized with %d skill packages", len(self._packages))

    def reload_packages(self):
        """Reload all skill packages from disk."""
        from gaia_mcp.skill_package import load_all_packages
        self._packages = load_all_packages(self._skills_dir)
        logger.info("SkillGateway reloaded: %d packages", len(self._packages))

    async def route(self, verb: str, params: Dict[str, Any]) -> Dict[str, Any]:
        """Route a meta-verb call to the appropriate handler.

        Args:
            verb: One of search, do, learn, remember, ask
            params: Verb-specific parameters

        Returns:
            Result dict with at least {"ok": bool}
        """
        handlers = {
            "search": self._handle_search,
            "do": self._handle_do,
            "learn": self._handle_learn,
            "remember": self._handle_remember,
            "ask": self._handle_ask,
        }
        handler = handlers.get(verb)
        if not handler:
            return {"ok": False, "error": f"Unknown verb: {verb}"}

        t0 = time.time()
        try:
            result = await handler(params)
            elapsed = (time.time() - t0) * 1000
            logger.info("SkillGateway: %s completed in %.0fms", verb, elapsed)
            return result
        except Exception as e:
            logger.exception("SkillGateway: %s failed", verb)
            return {"ok": False, "error": f"{type(e).__name__}: {e}"}

    # ── search(query) ────────────────────────────────────────────────

    async def _handle_search(self, params: Dict) -> Dict:
        """Unified retrieval: system sensors + MemPalace + KG + skills + web.

        Uses the Knowledge Router for knowledge retrieval and adds
        skill discovery on top.
        """
        query = params.get("query", "")
        if not query:
            return {"ok": False, "error": "search requires a 'query' parameter"}

        results = []

        # 1. Knowledge Router (system sensors, MemPalace, KG, web)
        try:
            from gaia_core.cognition.knowledge_router import ground_query
            ctx = ground_query(query=query, intent="other", max_total_ms=2000)
            for r in ctx.results:
                results.append({
                    "source": r.source,
                    "trust": r.trust_tier,
                    "content": r.content,
                    "url": r.url,
                    "type": "knowledge",
                })
        except ImportError:
            # gaia-core not available in MCP context — try MCP calls
            try:
                from gaia_mcp.tools import execute_limb
                from gaia_mcp.approval import ApprovalStore
                _store = ApprovalStore()

                # System clock check
                q = query.lower()
                if any(s in q for s in ["time", "clock", "date", "day"]):
                    import os
                    from datetime import datetime, timezone, timedelta
                    try:
                        tz_off = int(os.environ.get("LOCAL_TZ_OFFSET", "-7"))
                        tz = timezone(timedelta(hours=tz_off))
                        now = datetime.now(tz)
                        display = now.strftime("%-I:%M %p %Z, %A %B %d, %Y")
                        results.append({
                            "source": "system_clock",
                            "trust": "verified_local",
                            "content": f"The current time is {display}.",
                            "type": "knowledge",
                        })
                    except Exception:
                        pass

                # MemPalace recall
                try:
                    palace_result = await execute_limb("palace_recall", {"query": query, "top_k": 2}, _store, pre_approved=True)
                    if isinstance(palace_result, dict):
                        for mem in palace_result.get("results", []):
                            content = mem.get("content", "") if isinstance(mem, dict) else str(mem)
                            if content:
                                results.append({
                                    "source": "mempalace",
                                    "trust": "verified_local",
                                    "content": content[:300],
                                    "type": "knowledge",
                                })
                except Exception:
                    pass
            except Exception:
                logger.debug("Knowledge retrieval failed in MCP context", exc_info=True)

        # 2. Skill discovery — find matching skills by name/description
        matching_skills = self._find_matching_skills(query)
        for name, score in matching_skills[:3]:
            pkg = self._packages[name]
            results.append({
                "source": f"skill:{name}",
                "trust": "skill",
                "content": f"Skill '{name}': {pkg.description}",
                "match_score": score,
                "type": "skill",
                "execution_mode": pkg.execution_mode,
            })

        return {
            "ok": True,
            "query": query,
            "results": results,
            "result_count": len(results),
        }

    def _find_matching_skills(self, query: str, top_k: int = 5) -> list:
        """Simple keyword matching for skill discovery.

        Returns list of (skill_name, score) tuples.
        Phase 2 replaces this with embedding similarity.
        """
        query_words = set(query.lower().split())
        scores = []
        for name, pkg in self._packages.items():
            name_words = set(name.lower().replace("-", " ").replace("_", " ").split())
            desc_words = set(pkg.description.lower().split())
            all_words = name_words | desc_words

            # Word overlap score
            overlap = len(query_words & all_words)
            if overlap > 0:
                score = overlap / max(len(query_words), 1)
                scores.append((name, score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores[:top_k]

    # ── do(skill, input) ─────────────────────────────────────────────

    async def _handle_do(self, params: Dict) -> Dict:
        """Execute a skill by name.

        KNOWLEDGE skills: inject body as system prompt, send input to LLM.
        PLAYBOOK skills: route to legacy tool or SkillManager.
        """
        skill_name = params.get("skill", "")
        skill_input = params.get("input", "")

        if not skill_name:
            return {"ok": False, "error": "do() requires a 'skill' parameter"}

        # Normalize name
        skill_name = skill_name.lower().replace(" ", "-").replace("_", "-")
        pkg = self._packages.get(skill_name)

        # Also try underscore variant
        if not pkg:
            pkg = self._packages.get(skill_name.replace("-", "_"))
        if not pkg:
            # Try partial match
            candidates = [n for n in self._packages if skill_name in n]
            if len(candidates) == 1:
                pkg = self._packages[candidates[0]]

        if not pkg:
            available = ", ".join(sorted(self._packages.keys())[:10])
            return {
                "ok": False,
                "error": f"Skill '{skill_name}' not found. Available: {available}...",
            }

        # Approval check
        if pkg.sensitive:
            return {
                "ok": False,
                "error": f"Skill '{pkg.name}' requires approval. Use the approval workflow.",
                "requires_approval": True,
            }

        if pkg.is_knowledge:
            return await self._execute_knowledge_skill(pkg, skill_input)
        else:
            return await self._execute_playbook_skill(pkg, skill_input, params)

    async def _execute_knowledge_skill(self, pkg, input_text: str) -> Dict:
        """Execute a KNOWLEDGE-mode skill (prompt injection)."""
        try:
            import httpx
            import os
            core_url = os.getenv("CORE_ENDPOINT", "http://gaia-core:6415")

            async with httpx.AsyncClient(timeout=120.0) as client:
                resp = await client.post(f"{core_url}/api/cognitive/query", json={
                    "system_prompt": pkg.body,
                    "user_input": input_text,
                    "target": "core",
                    "max_tokens": 2048,
                    "no_think": True,
                })
                if resp.status_code == 200:
                    data = resp.json()
                    return {
                        "ok": True,
                        "content": data.get("content", data.get("response", str(data))),
                        "skill": pkg.name,
                        "mode": "KNOWLEDGE",
                    }
                return {"ok": False, "error": f"Core returned {resp.status_code}"}
        except Exception as e:
            return {"ok": False, "error": f"KNOWLEDGE skill execution failed: {e}"}

    async def _execute_playbook_skill(self, pkg, input_text: str, full_params: Dict) -> Dict:
        """Execute a PLAYBOOK-mode skill (code execution)."""
        from gaia_mcp.tools import execute_limb
        from gaia_mcp.approval import ApprovalStore

        _store = ApprovalStore()

        # Route through legacy tool if mapped
        if pkg.legacy_maps_to:
            # Build params from skill parameters + input
            tool_params = {}
            if pkg.legacy_action:
                tool_params["action"] = pkg.legacy_action

            # Map the input to the first required parameter
            if pkg.parameters:
                first_param = pkg.parameters[0].name
                tool_params[first_param] = input_text
            else:
                tool_params["input"] = input_text

            # Merge any extra params from the caller
            for k, v in full_params.items():
                if k not in ("skill", "input"):
                    tool_params[k] = v

            logger.info("PLAYBOOK skill '%s' → legacy tool '%s' with params %s",
                        pkg.name, pkg.legacy_maps_to, list(tool_params.keys()))
            result = await execute_limb(pkg.legacy_maps_to, tool_params, _store, pre_approved=True)
            return {
                "ok": True,
                "content": result if isinstance(result, str) else json.dumps(result, default=str)[:2000],
                "skill": pkg.name,
                "mode": "PLAYBOOK",
                "raw": result,
            }

        # Otherwise try SkillManager
        try:
            from gaia_mcp.skill_manager import SkillManager
            sm = SkillManager()
            result = await sm.execute_limb(pkg.name, {"input": input_text, **full_params})
            return {
                "ok": True,
                "content": result if isinstance(result, str) else json.dumps(result, default=str)[:2000],
                "skill": pkg.name,
                "mode": "PLAYBOOK",
                "raw": result,
            }
        except Exception as e:
            return {"ok": False, "error": f"PLAYBOOK skill execution failed: {e}"}

    # ── learn(task, result, success) ─────────────────────────────────

    async def _handle_learn(self, params: Dict) -> Dict:
        """Record outcome for utility scoring and skill learning."""
        task = params.get("task", "")
        result = params.get("result", "")
        success = params.get("success", True)

        if isinstance(success, str):
            success = success.lower() in ("true", "1", "yes")

        # Record utility outcome
        try:
            from gaia_core.cognition.knowledge_router import record_outcome, save_learned_knowledge
            domain = params.get("domain", "general")
            source = params.get("source", "model")
            record_outcome(domain, source, success)

            if success and result:
                save_learned_knowledge(
                    query=task, answer=result,
                    source=source, success=True, domain=domain,
                )
        except ImportError:
            logger.debug("Knowledge router not available in MCP context")

        return {"ok": True, "recorded": True, "task": task[:80], "success": success}

    # ── remember(fact) ───────────────────────────────────────────────

    async def _handle_remember(self, params: Dict) -> Dict:
        """Persist a fact to MemPalace + KG."""
        fact = params.get("fact", "")
        if not fact:
            return {"ok": False, "error": "remember() requires a 'fact' parameter"}

        try:
            from gaia_mcp.tools import execute_limb
            from gaia_mcp.approval import ApprovalStore
            _store = ApprovalStore()

            result = await execute_limb("palace_store", {
                "text": fact,
                "source": params.get("source", "model"),
            }, _store, pre_approved=True)

            return {
                "ok": True,
                "stored": True,
                "fact": fact[:200],
                "palace_result": result if isinstance(result, dict) else str(result)[:500],
            }
        except Exception as e:
            return {"ok": False, "error": f"MemPalace store failed: {e}"}

    # ── ask(question) ────────────────────────────────────────────────

    async def _handle_ask(self, params: Dict) -> Dict:
        """Request human input. Returns a marker for the conversation system."""
        question = params.get("question", "")
        if not question:
            return {"ok": False, "error": "ask() requires a 'question' parameter"}

        return {
            "ok": True,
            "type": "ask_user",
            "question": question,
            "awaiting_response": True,
        }
