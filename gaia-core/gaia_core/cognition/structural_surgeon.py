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

    def repair_structural_failure(self, service: str, broken_code: str, error_msg: str) -> Optional[str]:
        """
        HA Surgeon: Perform deep analysis of a structural failure.
        Injects the latest CognitionPacket v0.3 blueprint for grounding.
        """
        logger.info(f"StructuralSurgeon: Initiating HA surgery for {service}...")
        
        # 1. Acquire Thinker model (System 2 high-reasoning)
        model = self.model_pool.acquire_model_for_role("thinker")
        if not model:
            logger.error("StructuralSurgeon: Thinker model unavailable for surgery.")
            return None
            
        try:
            # 2. Load the latest blueprint for grounding
            blueprint_path = "/knowledge/blueprints/GAIA_COMMON.md"
            blueprint_content = ""
            try:
                with open(blueprint_path, "r") as f:
                    blueprint_content = f.read()
            except Exception:
                logger.warning("StructuralSurgeon: Could not load blueprint for grounding.")

            # 3. Assemble surgical prompt
            system_prompt = (
                "You are the GAIA HA Surgeon. Your purpose is to repair fatal Python errors (SyntaxError, IndentationError, KeyError, NameError) "
                "in the Candidate stack by leveraging your stable Production knowledge. "
                "GROUNDING CONTEXT: You must ensure all protocol changes align with the CognitionPacket v0.3 blueprint provided below. "
                "RULES:\n"
                "1. PRESERVE all logic, comments, and strings.\n"
                "2. Correct ONLY the structural flaw identified in the error message.\n"
                "3. If a field is missing in a constructor, add it with a safe default factory based on the blueprint.\n"
                "4. Output ONLY the fully corrected Python code. No explanation, no markdown blocks."
            )
            
            user_prompt = (
                f"SERVICE: {service}\n"
                f"LATEST BLUEPRINT (v0.3):\n{blueprint_content}\n\n"
                f"ERROR MESSAGE:\n{error_msg}\n\n"
                f"BROKEN SOURCE CODE:\n{broken_code}\n\n"
                "FIXED SOURCE CODE:"
            )
            
            messages = [
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": user_prompt}
            ]
            
            # 4. Request surgical fix
            res = model.create_chat_completion(
                messages=messages,
                temperature=0.0, # Deterministic repair
                max_tokens=8192  # Support full files
            )
            
            fixed_code = ""
            if isinstance(res, dict):
                fixed_code = res.get("choices", [{}])[0].get("message", {}).get("content", "")
            else:
                # Handle stream
                fixed_code = "".join([str(chunk) for m in res for chunk in m])
                
            # Clean up output
            fixed_code = re.sub(r'^```python\n', '', fixed_code)
            fixed_code = re.sub(r'^```\n', '', fixed_code)
            fixed_code = re.sub(r'\n```$', '', fixed_code).strip()
            
            if fixed_code:
                logger.info(f"StructuralSurgeon: Surgery complete for {service} (len={len(fixed_code)})")
                return fixed_code
            return None
            
        except Exception:
            logger.exception("StructuralSurgeon: HA surgery failed")
            return None
        finally:
            self.model_pool.release_model_for_role("thinker")
