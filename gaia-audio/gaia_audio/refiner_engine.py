"""
Nano-Refiner engine wrapping llama-cpp-python.

Provides high-speed text refinement (spelling, diarization, formatting)
using a small 0.5B model on the CPU.
"""

import logging
from llama_cpp import Llama
from typing import Optional

logger = logging.getLogger("GAIA.Audio.Refiner")

class RefinerEngine:
    def __init__(self, model_path: str, context_window: int = 4096):
        self.model_path = model_path
        self.context_window = context_window
        self.llm: Optional[Llama] = None

    def load(self):
        """Load the nano model into memory (CPU)."""
        if self.llm is not None:
            return
        
        logger.info(f"Loading Nano-Refiner model: {self.model_path}")
        try:
            self.llm = Llama(
                model_path=self.model_path,
                n_ctx=self.context_window,
                n_threads=4, # Use 4 threads for high-speed CPU inference
                n_gpu_layers=0, # Force CPU
                verbose=False
            )
            logger.info("Nano-Refiner model loaded successfully")
        except Exception as e:
            logger.error(f"Failed to load Nano-Refiner model: {e}")
            raise

    def refine(self, prompt: str, max_tokens: int = 2048) -> str:
        """Run refinement on the provided text."""
        if self.llm is None:
            self.load()
        
        try:
            # Use create_chat_completion for reliable instruction following
            response = self.llm.create_chat_completion(
                messages=[
                    {"role": "system", "content": "You are GAIA's Transcript Refiner. Clean up raw audio transcripts, correct proper names like 'Azrael', and perform semantic diarization. Be concise and return ONLY the refined text."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=max_tokens,
                temperature=0.1
            )
            
            return response["choices"][0]["message"]["content"].strip()
        except Exception as e:
            logger.error(f"Refinement failed: {e}")
            return f"Error: {e}"

    def unload(self):
        """Free memory."""
        if self.llm is not None:
            del self.llm
            self.llm = None
            logger.info("Nano-Refiner model unloaded")
