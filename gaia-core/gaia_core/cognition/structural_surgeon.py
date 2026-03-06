import ast
import logging
import re
from typing import List, Optional

from gaia_core.config import Config
from gaia_core.models.model_pool import ModelPool

logger = logging.getLogger("GAIA.StructuralSurgeon")


class StructuralSurgeon:
    """Windowed LLM repair for Python syntax/structural failures."""

    def __init__(self, config: Config, model_pool: ModelPool):
        self.config = config
        self.model_pool = model_pool

    def repair_structural_failure(self, service: str, broken_code: str, error_msg: str) -> Optional[str]:
        """
        HA Surgeon: Windowed repair that targets ALL error-bearing lines.

        1. Collects every error line from ast.parse + error_msg mentions.
        2. Builds a window that covers all of them with padding.
        3. Prompts the LLM to find and remove ALL visible stray syntax.
        4. Splices the fixed snippet back into the full file.
        """
        logger.info("StructuralSurgeon: Initiating HA surgery for %s...", service)

        error_lines = self._collect_error_lines(broken_code, error_msg)
        if not error_lines:
            error_lines = [1]
        logger.info("StructuralSurgeon: Error lines detected: %s", error_lines)

        lines = broken_code.splitlines()
        start = max(0, min(error_lines) - 40)
        end = min(len(lines), max(error_lines) + 40)

        # Snap end forward to the next blank line so we never cut mid-docstring
        # or mid-multiline-string.
        while end < len(lines) and lines[end].strip() != "":
            end += 1
        end = min(end, len(lines))

        window_code = "\n".join(lines[start:end])

        model = self.model_pool.acquire_model_for_role("thinker")
        if not model:
            logger.error("StructuralSurgeon: Thinker model unavailable.")
            return None

        try:
            error_summary = ", ".join(f"line {l}" for l in error_lines[:5])

            system_prompt = (
                "You are a Python code surgeon. Your only job is to remove injected syntax "
                "errors (stray parentheses, extra brackets, mismatched delimiters) from Python "
                "snippets. Output ONLY the corrected snippet. No markdown fences, no explanations."
            )

            user_prompt = (
                f"REPORTED ERRORS AT: {error_summary}\n"
                f"ERROR DETAIL: {error_msg.strip()[:400]}\n\n"
                f"SNIPPET (lines {start + 1}–{end}):\n{window_code}\n\n"
                "TASK: Carefully examine EVERY LINE for stray characters, extra parentheses, "
                "or mismatched brackets injected into the code. Remove ALL of them. "
                "Do not change any other logic.\n\n"
                "FIXED SNIPPET:"
            )

            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt},
            ]

            res = model.create_chat_completion(messages=messages, temperature=0.0, max_tokens=2048)

            fixed_snippet = ""
            if isinstance(res, dict):
                fixed_snippet = res.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                fixed_snippet = "".join([str(chunk) for m in res for chunk in m])

            fixed_snippet = re.sub(
                r'^```python\s*|^```\s*|```$', '', fixed_snippet, flags=re.MULTILINE
            ).strip()

            if not fixed_snippet:
                logger.warning("StructuralSurgeon: Empty response from LLM.")
                return None

            lines[start:end] = fixed_snippet.splitlines()
            reassembled = "\n".join(lines)
            logger.info("StructuralSurgeon: Surgery complete for %s (len=%d)", service, len(reassembled))
            return reassembled

        except Exception:
            logger.exception("StructuralSurgeon: HA surgery failed")
            return None
        finally:
            self.model_pool.release_model_for_role("thinker")

    def _collect_error_lines(self, code: str, error_msg: str) -> List[int]:
        """
        Gather all probable error line numbers.

        Uses ast.parse as the authoritative source (lineno + any line number
        embedded in the exception message), then supplements with error_msg
        mentions that are within 100 lines of the primary error — filtering
        out spurious numbers from file paths and unrelated text.
        """
        found: set = set()
        primary_line: int = 0

        try:
            ast.parse(code)
        except SyntaxError as e:
            if e.lineno:
                found.add(e.lineno)
                primary_line = e.lineno
            if e.end_lineno and e.end_lineno != e.lineno:
                found.add(e.end_lineno)
            # The SyntaxError message itself often says "on line N" for the
            # mismatching bracket — extract it directly from the structured msg.
            if e.msg:
                for m in re.finditer(r'\bline\s+(\d+)', e.msg, re.IGNORECASE):
                    found.add(int(m.group(1)))
        except Exception:
            pass

        # Supplement with error_msg string mentions, but only near the primary error
        # to avoid noise from file paths or unrelated content.
        if primary_line:
            for m in re.finditer(r'\bline\s+(\d+)', error_msg, re.IGNORECASE):
                ln = int(m.group(1))
                if abs(ln - primary_line) <= 100:
                    found.add(ln)

        return sorted(found)
