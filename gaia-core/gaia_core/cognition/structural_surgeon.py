import logging
import re
from typing import Optional
from gaia_core.models.model_pool import ModelPool
from gaia_core.config import Config

logger = logging.getLogger("GAIA.StructuralSurgeon")

class StructuralSurgeon:
    """Specialized utility for fixing IndentationError and SyntaxError via LLM."""
    
    def __init__(self, config: Config, model_pool: ModelPool):
        self.config = config
        self.model_pool = model_pool

    def repair_snippet(self, broken_code: str, error_msg: str) -> Optional[str]:
        """Send broken code to Lite model for structural repair."""
        logger.info("StructuralSurgeon: Initiating cognitive repair for structural error...")
        
        # 1. Acquire Lite model (System 2 responder)
        model = self.model_pool.acquire_model_for_role("lite")
        if not model:
            logger.error("StructuralSurgeon: Lite model unavailable for repair.")
            return None
            
        try:
            # 2. Assemble repair prompt
            system_prompt = (
                "You are GAIA's Structural Surgeon. Your sole purpose is to fix Python SyntaxError and IndentationError. "
                "You must PRESERVE the logic, comments, and strings exactly as they are. "
                "Only correct the indentation and structural syntax (brackets, colons, etc.). "
                "Output ONLY the corrected Python code. No preamble, no markdown blocks, no explanation."
            )
            
            user_prompt = (
                f"ERROR MESSAGE:\n{error_msg}\n\n"
                f"BROKEN CODE SNIPPET:\n{broken_code}\n\n"
                "FIXED CODE:"
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # 3. Request repair
            res = model.create_chat_completion(
                messages=messages,
                temperature=0.1, # Low temperature for structural stability
                max_tokens=2048
            )
            
            # Handle both stream and non-stream responses
            if hasattr(res, "__iter__") and not isinstance(res, (dict, list, str)):
                fixed_code = "".join([str(chunk) for m in res for chunk in m])
            elif isinstance(res, dict):
                fixed_code = res.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                fixed_code = str(res)
                
            # Scrub markdown blocks if the model ignored instructions
            fixed_code = re.sub(r'^```python\n', '', fixed_code)
            fixed_code = re.sub(r'^```\n', '', fixed_code)
            fixed_code = re.sub(r'\n```$', '', fixed_code)
            
            logger.info("StructuralSurgeon: Repair completed (len=%d chars)", len(fixed_code))
            return fixed_code.strip()
            
        except Exception:
            logger.exception("StructuralSurgeon: Cognitive repair turn failed")
            return None
        finally:
            self.model_pool.release_model_for_role("lite")
