from typing import List, Dict, Any, Optional
import logging

logger = logging.getLogger("GAIA.HFModel")
from gaia_common.utils.hf_prompting import build_hf_prompt, default_stop_tokens

class HFModel:
    """A thin wrapper around Hugging Face transformers for text generation.

    Provides a create_completion(prompt, ...) and create_chat_completion(messages, ...)
    compatible interface used by ModelPool.forward_to_model.
    """
    logger = logging.getLogger("GAIA.HFModel")

    def __init__(
        self,
        model_ref: str,
        local_path: str = None,
        device_map: Optional[str] = "auto",
        torch_dtype=None,
        prompt_config: Optional[Dict] = None,
    ):
        try:
            from transformers import AutoModelForCausalLM, AutoTokenizer, pipeline
            import torch
        except Exception as e:
            raise RuntimeError(f"transformers or torch not available: {e}")

        self.model_ref = local_path or model_ref
        self.prompt_config = prompt_config or {}

        # allow an environment override to force conservative CPU-only mode
        import os
        if os.getenv("GAIA_FORCE_CPU", "0") == "1":
            device_map = None
            torch_dtype = None

        # Attempt an accelerate/device_map GPU-aware load, but fall back to CPU if it fails
        tried_device_map = device_map
        last_exc = None
        for attempt in (0, 1):
            try:
                # Load tokenizer and model; allow local paths or HF repo ids
                self.tokenizer = AutoTokenizer.from_pretrained(self.model_ref, use_fast=True)
                # Prefer float16 on CUDA devices when using device_map to reduce memory
                preferred_dtype = torch_dtype
                if preferred_dtype is None and device_map is not None:
                    try:
                        import torch as _torch
                        preferred_dtype = _torch.float16
                    except Exception:
                        preferred_dtype = None

                # encourage accelerate to keep weights on GPU: construct an explicit max_memory mapping
                load_kwargs = {
                    "trust_remote_code": True,
                }
                # Try to reduce peak CPU memory during model deserialization
                # Transformers/Accelerate support `low_cpu_mem_usage=True` which
                # uses a streaming initialization to avoid building giant tensors
                # in CPU RAM before moving them to GPU.
                try:
                    load_kwargs["low_cpu_mem_usage"] = True
                except Exception:
                    pass
                if device_map:
                    load_kwargs["device_map"] = device_map
                    # prefer GPU: allow user to override GAIA_MAX_MEMORY_PER_GPU
                    max_mem_env = os.getenv("GAIA_MAX_MEMORY_PER_GPU")
                    mm = None
                    if max_mem_env:
                        try:
                            # Accept '14000MB' or integer values representing MB
                            if isinstance(max_mem_env, str) and max_mem_env.lower().endswith('mb'):
                                mm = max_mem_env
                            else:
                                mm = f"{int(max_mem_env)}MB"
                        except Exception:
                            mm = None

                    # If user didn't provide a cap, try to compute a conservative default
                    # using torch.cuda total memory (80% of total) to avoid OOM in many setups.
                    if mm is None:
                        try:
                            import torch as _torch
                            if _torch.cuda.is_available():
                                # total_memory is bytes
                                props = _torch.cuda.get_device_properties(0)
                                total_mb = int(props.total_memory / (1024 * 1024))
                                # conservatively use 65% of total VRAM to leave headroom
                                cap_mb = int(total_mb * 0.65)
                                mm = f"{cap_mb}MB"
                                self.logger.info(f"GAIA_MAX_MEMORY_PER_GPU not set; using conservative default {mm}")
                        except Exception:
                            mm = None

                    if mm:
                        try:
                            # map device index 0 to the provided memory cap
                            load_kwargs["max_memory"] = {"0": mm}
                        except Exception:
                            pass
                    # If an offload directory is provided via env, pass it through
                    offload_dir = os.getenv("GAIA_OFFLOAD_DIR")
                    if offload_dir:
                        try:
                            load_kwargs["offload_folder"] = offload_dir
                        except Exception:
                            pass
                if preferred_dtype is not None:
                    # use the modern 'dtype' argument instead of deprecated 'torch_dtype'
                    load_kwargs["dtype"] = preferred_dtype

                self.logger.info(f"AutoModel.from_pretrained called with: {load_kwargs}")
                self.model = AutoModelForCausalLM.from_pretrained(self.model_ref, **load_kwargs)
                # build a lightweight pipeline wrapper around the model for non-stream usage
                self.pipeline = pipeline(
                    "text-generation",
                    model=self.model,
                    tokenizer=self.tokenizer,
                )
                last_exc = None
                break
            except Exception as e:
                logger.warning(f"HFModel init attempt failed (device_map={device_map}): {e}")
                last_exc = e
                # On first failure try a CPU-only fallback (clear device_map and dtype)
                device_map = None
                torch_dtype = None

        if last_exc:
            logger.error(f"Failed to initialize HFModel for {self.model_ref}: {last_exc}")
            raise last_exc

        # Best-effort device info: model.device may not exist for sharded/accelerate loads
        try:
            self._device = next(self.model.parameters()).device
        except Exception:
            self._device = None

    def _messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
        # Use hf_prompting templating to build model-appropriate prompts.
        template = self.prompt_config.get("template", "short_answer")
        return build_hf_prompt(messages, template_name=template, prompt_config=self.prompt_config)

    def create_completion(self, prompt: str, max_tokens: int = 128, temperature: float = 0.2, **kwargs):
        # Support both streaming and non-streaming via kwargs. If stream=True is passed
        # we return a generator that yields dicts compatible with ExternalVoice's
        # streaming format: {"choices":[{"delta": {"content": "..."}}]}
        stream = kwargs.pop("stream", False)

        # Allow caller to pass stop tokens via kwargs or fall back to defaults
        stop_tokens = kwargs.pop("stop", None)
        if stop_tokens is None:
            stop_tokens = default_stop_tokens(self.prompt_config)

        if not stream:
            # Use inference mode to reduce CPU/GPU overhead during generation
            try:
                import torch
                with torch.inference_mode():
                    gen = self.pipeline(
                        prompt,
                        max_new_tokens=max_tokens,
                        do_sample=(temperature > 0.0),
                        temperature=float(temperature),
                        **kwargs,
                    )
            except Exception:
                gen = self.pipeline(
                    prompt,
                    max_new_tokens=max_tokens,
                    do_sample=(temperature > 0.0),
                    temperature=float(temperature),
                    **kwargs,
                )
            text = gen[0]["generated_text"]
            # Apply simple stop-token truncation if any
            for t in stop_tokens:
                if t and t in text:
                    text = text.split(t)[0]
                    break
            return {"choices": [{"text": text}]}

        # Streaming path using TextIteratorStreamer
        try:
            from transformers import TextIteratorStreamer
            import torch
        except Exception as e:
            raise RuntimeError(f"streaming generation requires transformers.TextIteratorStreamer: {e}")

        # Tokenize input; prefer returning PyTorch tensors to leverage device map
        inputs = self.tokenizer(prompt, return_tensors="pt")

        streamer = TextIteratorStreamer(self.tokenizer, skip_prompt=True, skip_special_tokens=True)

        def _generate():
            try:
                # move inputs to device if we can
                try:
                    device = next(self.model.parameters()).device
                    inputs = self.tokenizer(prompt, return_tensors="pt").to(device)
                except Exception:
                    inputs = self.tokenizer(prompt, return_tensors="pt")
                # run generation under no_grad/inference mode for better performance
                try:
                    import torch
                    with torch.inference_mode():
                        self.model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature, do_sample=(temperature>0.0), streamer=streamer, **kwargs)
                except Exception:
                    self.model.generate(**inputs, max_new_tokens=max_tokens, temperature=temperature, do_sample=(temperature>0.0), streamer=streamer, **kwargs)
            except Exception as e:
                logger.error(f"HFModel streaming generation error: {e}")

        import threading
        thread = threading.Thread(target=_generate, daemon=True)
        thread.start()

        # Yield token deltas as they arrive but enforce stop tokens
        buffer = ""
        for chunk in streamer:
            buffer += chunk
            # If any stop token appears in buffer, yield up to it and stop
            truncated = False
            for t in stop_tokens:
                if t and t in buffer:
                    idx = buffer.index(t)
                    piece = buffer[:idx]
                    if piece:
                        yield {"choices": [{"delta": {"content": piece}}]}
                    truncated = True
                    break
            if truncated:
                break
            else:
                yield {"choices": [{"delta": {"content": chunk}}]}

    def create_chat_completion(self, messages: List[Dict[str, Any]], max_tokens: int = 128, temperature: float = 0.2, **kwargs):
        prompt = self._messages_to_prompt(messages)
        # If caller requests streaming via stream=True, forward-through the generator
        stream = kwargs.get("stream", False)
        if stream:
            # create_completion will yield generator items in streaming mode
            for item in self.create_completion(prompt, max_tokens=max_tokens, temperature=temperature, **kwargs):
                yield item
            return

        result = self.create_completion(prompt, max_tokens=max_tokens, temperature=temperature, **kwargs)
        # If result is a generator for any reason, consume it like a stream and assemble text
        try:
            import types
            if isinstance(result, types.GeneratorType):
                pieces = []
                for item in result:
                    try:
                        delta = item.get("choices", [{}])[0].get("delta", {})
                        text = delta.get("content")
                        if text:
                            pieces.append(text)
                            continue
                    except Exception:
                        pass
                    try:
                        text = item.get("choices", [{}])[0].get("message", {}).get("content")
                        if text:
                            pieces.append(text)
                    except Exception:
                        pass
                assembled = "".join(pieces).strip()
                return {"choices": [{"message": {"content": assembled}}]}
        except Exception:
            logger.exception("Failed while assembling generator result in create_chat_completion")
            # If assembling a generator-style result fails, return a safe empty response
            # so callers don't hit a TypeError when indexing the result.
            return {"choices": [{"message": {"content": ""}}]}

        # Return a compat structure: choices[0].message.content
        text = result["choices"][0].get("text") or result["choices"][0].get("message", {}).get("content", "")
        return {"choices": [{"message": {"content": text}}]}
