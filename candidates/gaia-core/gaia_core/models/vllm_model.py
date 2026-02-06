import json
import logging
import os
from collections import defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, Generator, Iterable, Iterator, List, Optional, Tuple
import inspect

# Defer importing vLLM until after we set the worker-method env var so the
# vLLM library can initialize multiprocessing with the desired start method.
LLM = None
SamplingParams = None
LoRARequest = None

from gaia_common.utils.hf_prompting import build_hf_prompt, default_stop_tokens

logger = logging.getLogger("GAIA.VLLM")


@dataclass
class LoRAAdapterInfo:
    """Metadata for a loaded LoRA adapter."""
    name: str
    path: str
    tier: int  # 1=global, 2=user, 3=session
    rank: int = 8
    loaded: bool = False
    lora_id: int = 0  # vLLM internal ID
    metadata: Dict[str, Any] = field(default_factory=dict)


class VLLMChatModel:
	"""
	Thin wrapper around vLLM that exposes GAIA's create_chat_completion interface.

	Parameters are sourced from MODEL_CONFIGS entries and GAIA_VLLM_* environment
	variables so GPU memory usage can be tuned without code changes.
	"""

	def __init__(
		self,
		model_config: Dict[str, Any],
		global_config,
		gpu_info: Optional[Tuple[Optional[int], Optional[int]]] = None,
	):
		# Bind the module-level aliases to the imported vLLM classes. Without
		# this `global` declaration, Python would treat the import assignments as
		# locals, leaving the outer `SamplingParams` as None and causing
		# "'NoneType' object is not callable" when building sampling params.
		global LLM, SamplingParams
		# Allow override of the multiprocessing start method used by vLLM workers.
		# Default to 'spawn' which is safer for CUDA, but allow processes launched
		# from interactive shells (rescue) to opt into 'fork' via env var
		# GAIA_VLLM_WORKER_METHOD if necessary to avoid runpy/run-from-stdin issues.
		method = os.getenv("GAIA_VLLM_WORKER_METHOD")
		if not method:
			# If running interactively, prefer 'fork' to avoid spawn/runpy problems
			try:
				import sys
				interactive = hasattr(sys, 'ps1') or sys.stdin is not None and sys.stdin.isatty()
			except Exception:
				interactive = False
			method = "fork" if interactive else "spawn"
		# Export env var before importing vllm so it picks up the requested method.
		os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", method)
		# Try to set the Python multiprocessing start method early so child
		# processes created by vLLM use the requested method. Use force=True to
		# override if another method was already set (safe on Linux; avoids
		# spawn/runpy errors when running inside interactive shells).
		try:
			import multiprocessing
			multiprocessing.set_start_method(method, force=True)
		except Exception as _:
			# If this fails, continue — vLLM may still work depending on context.
			logger.debug("Could not set multiprocessing start method to %s", method)
		# Now import vllm (done here so the library respects VLLM_WORKER_MULTIPROC_METHOD)
		try:
			from vllm import LLM as VLLM_LLM, SamplingParams as VLLM_SamplingParams  # type: ignore
			LLM = VLLM_LLM
			SamplingParams = VLLM_SamplingParams
		except Exception as exc:  # pragma: no cover - handled at runtime when optional dep missing
			logger.warning("vLLM import failed: %s", exc)
			raise RuntimeError("vLLM is not installed; cannot load vLLM-backed model")
		self.config = global_config
		self.model_config = model_config or {}
		self.prompt_config = self.model_config.get("prompt_config") or getattr(global_config, "prompt_config", {})
		self.model_path = (
			self.model_config.get("path")
			or self.model_config.get("model")
			or os.getenv("GAIA_VLLM_MODEL_PATH")
		)
		if not self.model_path:
			raise ValueError("VLLM model requires 'path' or 'model' field")

		# Allow override of the multiprocessing start method used by vLLM workers.
		# Default to 'spawn' which is safer for CUDA, but allow processes launched
		# from interactive shells (rescue) to opt into 'fork' via env var
		# GAIA_VLLM_WORKER_METHOD if necessary to avoid runpy/run-from-stdin issues.
		method = os.getenv("GAIA_VLLM_WORKER_METHOD")
		if not method:
			# If running interactively, prefer 'fork' to avoid spawn/runpy problems
			try:
				import sys
				interactive = hasattr(sys, 'ps1') or sys.stdin is not None and sys.stdin.isatty()
			except Exception:
				interactive = False
			method = "fork" if interactive else "spawn"
		os.environ.setdefault("VLLM_WORKER_MULTIPROC_METHOD", method)
		self._free_bytes, self._total_bytes = gpu_info if gpu_info else (None, None)
		self._gpu_util = self._resolve_gpu_utilization()
		self._supports_native_stream = False

		max_model_len = self._int_from_config("max_model_len", os.getenv("GAIA_VLLM_MAX_MODEL_LEN"), 8192)
		max_num_seqs = self._int_from_config("max_num_seqs", None, 1)
		max_batched = self._int_from_config("max_num_batched_tokens", None, 1024)

		llm_kwargs: Dict[str, Any] = {
			"model": self.model_path,
			"max_model_len": max_model_len,
			"max_num_seqs": max_num_seqs,
			"max_num_batched_tokens": max_batched,
			"gpu_memory_utilization": self._gpu_util,
		}
		for key in (
			"tensor_parallel_size",
			"trust_remote_code",
			"download_dir",
			"revision",
			"tokenizer",
		):
			if key in self.model_config:
				llm_kwargs[key] = self.model_config[key]

		if str(os.getenv("GAIA_VLLM_SAFE_MODE", "0")) == "1":
			llm_kwargs.setdefault("enforce_eager", True)
			llm_kwargs.setdefault("disable_log_stats", True)
			llm_kwargs["gpu_memory_utilization"] = min(self._gpu_util, 0.4)

		logger.info(
			"Initializing vLLM model at %s with max_len=%s, gpu_util=%.3f, max_batched=%s",
			self.model_path,
			max_model_len,
			llm_kwargs["gpu_memory_utilization"],
			max_batched,
		)
		llm_kwargs["disable_log_stats"] = False

		# LoRA configuration
		self._lora_enabled = False
		self._loaded_adapters: Dict[str, LoRAAdapterInfo] = {}
		self._next_lora_id = 1
		lora_config = self.model_config.get("lora_config") or {}
		if lora_config.get("enabled", False):
			self._lora_enabled = True
			llm_kwargs["enable_lora"] = True
			llm_kwargs["max_loras"] = lora_config.get("max_loras", 4)
			llm_kwargs["max_lora_rank"] = lora_config.get("max_lora_rank", 64)
			logger.info("LoRA support enabled: max_loras=%s, max_rank=%s",
				llm_kwargs["max_loras"], llm_kwargs["max_lora_rank"])

		self.llm = LLM(**llm_kwargs)

		# Import LoRARequest if LoRA is enabled
		if self._lora_enabled:
			try:
				from vllm.lora.request import LoRARequest as VLLM_LoRARequest
				global LoRARequest
				LoRARequest = VLLM_LoRARequest
				logger.info("LoRARequest imported successfully")
			except ImportError as e:
				logger.warning("Could not import LoRARequest: %s", e)
				self._lora_enabled = False

		try:
			sig = inspect.signature(self.llm.generate)
			self._supports_native_stream = "stream" in sig.parameters
		except Exception:
			self._supports_native_stream = False
		self._request_cache: Dict[str, str] = {}

	def _int_from_config(self, key: str, env_override: Optional[str], default: int) -> int:
		try:
			if env_override is not None:
				return int(env_override)
		except Exception:
			pass
		try:
			if key in self.model_config:
				return int(self.model_config.get(key))
		except Exception:
			pass
		return default

	def _resolve_gpu_utilization(self) -> float:
		def _parse_env(name: str) -> Optional[float]:
			try:
				val = os.getenv(name)
				return float(val) if val is not None else None
			except Exception:
				return None

		# Priority: env var > model_config > default
		target = _parse_env("GAIA_VLLM_GPU_MEMORY_UTILIZATION")
		if target is None:
			target = _parse_env("GAIA_VLLM_GPU_UTIL")
		if target is None:
			# Check model_config for gpu_memory_utilization
			try:
				cfg_val = self.model_config.get("gpu_memory_utilization")
				if cfg_val is not None:
					target = float(cfg_val)
			except Exception:
				pass
		if target is None:
			# Default to 0.85 - the Claude model is ~7.4 GiB and needs room for KV cache
			target = 0.85

		min_util = _parse_env("GAIA_VLLM_MIN_UTIL") or 0.4
		max_util = _parse_env("GAIA_VLLM_MAX_UTIL") or 0.9
		target = max(min(target, max_util), min_util)

		reserve_mb = _parse_env("GAIA_VLLM_RESERVE_MB") or 256
		reserve_frac = _parse_env("GAIA_VLLM_RESERVE_FRAC") or 0.02
		if self._total_bytes:
			reserve_bytes = max(
				int(reserve_mb * 1024 * 1024),
				int(self._total_bytes * reserve_frac),
			)
			available = max(self._total_bytes - reserve_bytes, 0)
			if self._total_bytes > 0:
				max_allowed = available / float(self._total_bytes)
				target = min(target, max(max_allowed, min_util))
		if str(os.getenv("GAIA_VLLM_SAFE_MODE", "0")) == "1":
			target = min(target, 0.4)
		return float(target)

	def create_completion(
		self,
		prompt: str,
		max_tokens: int = 128,
		temperature: float = 0.2,
		top_p: float = 0.9,
		presence_penalty: float = 0.0,
		stream: bool = False,
		stop: Optional[Iterable[str]] = None,
		**kwargs,
	):
		# Heuristic: allow longer outputs and anti-repetition for clearly long-form asks (poems, stories, recitals).
		longform = False
		try:
			plower = prompt.lower()
			# Detect long-form content requests (poems, stories, recitations)
			longform_markers = ("jabberwocky", "poem", "recite", "story", "long-form", "raven", "verse", "lyrics", "ballad", "sonnet")
			if any(key in plower for key in longform_markers):
				longform = True
				# For poetry/recitation, need much more tokens (The Raven is ~1000 tokens)
				max_tokens = max(max_tokens, 2048)
		except Exception:
			pass
		# Presence penalty to reduce loops; bump it for long-form recitations.
		pp = presence_penalty
		# Base repetition penalty from env or default
		rp = float(os.getenv("GAIA_VLLM_REPETITION_PENALTY", "1.15"))
		if longform:
			pp = max(pp, 0.8)  # Higher presence penalty for long-form
			rp = max(rp, 1.25)  # Stronger repetition penalty to prevent loops
			# Lower temperature slightly for more faithful recitation
			temperature = min(temperature, 0.5)
			# Avoid over-stopping long-form outputs
			if stop is not None and len(stop) == 0:
				stop = None
		sampling_params = self._build_sampling_params(max_tokens, temperature, top_p, stop, presence_penalty=pp, repetition_penalty=rp)
		if stream:
			return self._stream_text([prompt], sampling_params)
		logger.debug("VLLMChatModel.create_completion prompt preview: %s", prompt[:400])
		outputs = self.llm.generate(prompts=[prompt], sampling_params=sampling_params)
		text = self._extract_text(outputs)
		# Filter out tool-noise completions that sometimes appear during self-checks.
		tool_noise = {"read_file", "write_file", "read_file, write_file", "read_file, write_file,"}
		if text.strip().lower() in tool_noise:
			logger.warning("VLLMChatModel.create_completion filtered tool-like text: %r", text)
			text = ""
		logger.warning("VLLMChatModel.create_completion raw outputs: %s", self._summarize_outputs(outputs))
		return {"choices": [{"message": {"content": text}}]}

	def _create_chat_completion_simple(
		self,
		messages: List[Dict[str, Any]],
		max_tokens: int = 2048,
		temperature: float = 0.7,
		top_p: float = 0.95,
		**kwargs,
	):
		try:
			prompt = self._messages_to_prompt(messages)
			logger.warning("VLLMChatModel._create_chat_completion_simple prompt: %s", prompt)
		except Exception as e:
			logger.exception("Error in _messages_to_prompt: %s", e)
			prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

		# Use _build_sampling_params to include proper stop tokens (especially <|im_end|>)
		sampling_params = self._build_sampling_params(
			max_tokens=max_tokens,
			temperature=temperature,
			top_p=top_p,
			stop=None,  # Will use default_stop_tokens
		)

		logger.debug("VLLMChatModel._create_chat_completion_simple prompt preview: %s", prompt[:500])
		outputs = self.llm.generate(prompts=[prompt], sampling_params=sampling_params)

		text = ""
		if outputs and outputs[0].outputs:
			text = outputs[0].outputs[0].text or ""
			logger.debug("VLLMChatModel._create_chat_completion_simple output: %r", text[:200])
		else:
			logger.warning("VLLMChatModel._create_chat_completion_simple: empty output, raw=%s", outputs)

		return {"choices": [{"message": {"content": text}}]}

	def create_chat_completion(
		self,
		messages: List[Dict[str, Any]],
		max_tokens: int = 128,
		temperature: float = 0.2,
		top_p: float = 0.9,
		stream: bool = False,
		stop: Optional[Iterable[str]] = None,
		**kwargs,
	):
		# prompt = self._messages_to_prompt(messages)
		# Just use the last user message for now
		prompt = messages[-1]["content"]

		# Detect long-form content requests for better generation params
		longform = False
		pp = 0.0  # presence penalty
		rp = float(os.getenv("GAIA_VLLM_REPETITION_PENALTY", "1.15"))
		try:
			plower = prompt.lower()
			longform_markers = ("jabberwocky", "poem", "recite", "story", "long-form", "raven", "verse", "lyrics", "ballad", "sonnet")
			if any(key in plower for key in longform_markers):
				longform = True
				max_tokens = max(max_tokens, 2048)
				pp = 0.8  # Higher presence penalty
				rp = max(rp, 1.25)  # Stronger repetition penalty
				temperature = min(temperature, 0.5)  # Lower temp for faithful recitation
		except Exception:
			pass

		sampling_params = self._build_sampling_params(max_tokens, temperature, top_p, stop, presence_penalty=pp, repetition_penalty=rp)
		if stream:
			return self._stream_text([prompt], sampling_params)

		return self._create_chat_completion_simple(
			messages,
			max_tokens=max_tokens,
			temperature=temperature,
			top_p=top_p,
			**kwargs,
		)

	def _messages_to_prompt(self, messages: List[Dict[str, Any]]) -> str:
		template = self.model_config.get("template") or self.prompt_config.get("default_template") or "raw"
		return build_hf_prompt(messages, template_name=template, prompt_config=self.prompt_config)

	def _build_sampling_params(
		self,
		max_tokens: int,
		temperature: float,
		top_p: float,
		stop: Optional[Iterable[str]],
		presence_penalty: float = 0.0,
		repetition_penalty: float = 1.0,
		frequency_penalty: float = 0.0,
	) -> SamplingParams:
		if stop is None:
			stop = default_stop_tokens(self.prompt_config)
		elif isinstance(stop, str):
			stop = [stop]
		else:
			stop = list(stop)
		stop = [s for s in stop if s]
		params = {
			"temperature": float(temperature),
			"top_p": float(top_p),
			"max_tokens": int(max_tokens),
			"stop": stop,
			"presence_penalty": float(presence_penalty),
			"repetition_penalty": float(repetition_penalty),
		}
		# Add frequency_penalty if non-zero (vLLM >= 0.4.0 supports it)
		if frequency_penalty > 0:
			params["frequency_penalty"] = float(frequency_penalty)
		return SamplingParams(**params)

	def shutdown(self) -> None:
		"""Best-effort shutdown for vLLM engine."""
		try:
			if hasattr(self, "llm") and hasattr(self.llm, "shutdown") and callable(self.llm.shutdown):
				self.llm.shutdown()
			elif hasattr(self, "llm") and hasattr(self.llm, "close") and callable(self.llm.close):
				self.llm.close()
			self.llm = None
			logger.info("VLLMChatModel shutdown complete")
		except Exception:
			logger.debug("VLLMChatModel shutdown failed", exc_info=True)

	def _stream_text(self, prompts: List[str], sampling_params: SamplingParams) -> Generator[Dict[str, Any], None, None]:
		if self._supports_native_stream:
			yield from self._native_stream_text(prompts, sampling_params)
			return
		outputs = self.llm.generate(prompts=prompts, sampling_params=sampling_params)
		text = self._extract_text(outputs)
		for delta in self._chunk_text(text):
			yield {"choices": [{"delta": {"content": delta}}]}

	def _native_stream_text(self, prompts: List[str], sampling_params: SamplingParams) -> Iterator[Dict[str, Any]]:
		cache: Dict[str, str] = defaultdict(str)
		any_delta = False
		try:
			generator = self.llm.generate(prompts=prompts, sampling_params=sampling_params, stream=True)
		except TypeError:
			# Some builds don't expose the stream kwarg even though signature advertised it.
			self._supports_native_stream = False
			yield from self._stream_text(prompts, sampling_params)
			return
		for chunk in generator:
			if not getattr(chunk, "outputs", None):
				continue
			text = getattr(chunk.outputs[0], "text", "") or ""
			req_id = getattr(chunk, "request_id", "default")
			prev = cache.get(req_id, "")
			delta = text[len(prev) :] if text.startswith(prev) else text
			logger.warning(f"VLLM NATIVE STREAM CHUNK: {chunk}")
			logger.warning(f"VLLM NATIVE STREAM TEXT: {text}")
			logger.warning(f"VLLM NATIVE STREAM PREV: {prev}")
			logger.warning(f"VLLM NATIVE STREAM DELTA: {delta}")
			cache[req_id] = text
			if not delta:
				continue
			any_delta = True
			try:
				logger.info("VLLM stream chunk: req=%s len=%s preview=%r", req_id, len(delta), delta[:120])
			except Exception:
				pass
			yield {"choices": [{"delta": {"content": delta}}]}
		if not any_delta:
			logger.info("VLLMChatModel.stream yielded no deltas; cache=%s", cache)
		else:
			logger.debug(
				"VLLMChatModel.stream completed for %s prompts; total_chars=%s",
				len(prompts),
				sum(len(v) for v in cache.values()),
			)

	def _chunk_text(self, text: str, chunk_size: int = 96) -> Iterator[str]:
		text = (text or "").strip()
		if not text:
			return iter(())

		words = text.split()
		if not words:
			return iter(())

		def _chunk_gen() -> Iterator[str]:
			current: List[str] = []
			current_len = 0
			for word in words:
				word_len = len(word)
				if current and current_len + word_len + 1 > chunk_size:
					yield " ".join(current)
					current = [word]
					current_len = word_len
				else:
					current.append(word)
					current_len += (word_len + (1 if current_len > 0 else 0))
			if current:
				yield " ".join(current)

		return _chunk_gen()

	def _extract_text(self, outputs) -> str:
		try:
			first = outputs[0]
			if hasattr(first, "outputs") and first.outputs:
				return first.outputs[0].text or ""
			if isinstance(first, dict):
				return first.get("text") or ""
		except Exception:
			pass
		return ""

	def _summarize_outputs(self, outputs) -> str:
		try:
			first = outputs[0]
			if hasattr(first, "outputs") and first.outputs:
				return repr(first.outputs[0])[:4000]
			return repr(first)[:4000]
		except Exception as exc:
			return f"<failed to summarize outputs: {exc}>"

	# -------------------------------------------------------------------------
	# LoRA Adapter Management
	# -------------------------------------------------------------------------

	def lora_enabled(self) -> bool:
		"""Check if LoRA support is enabled."""
		return self._lora_enabled

	def load_adapter(self, name: str, path: str, tier: int = 3) -> bool:
		"""
		Load a LoRA adapter for use with this model.

		Args:
			name: Unique identifier for the adapter
			path: Path to the adapter directory (containing adapter_config.json)
			tier: Tier level (1=global, 2=user, 3=session)

		Returns:
			True if adapter was loaded successfully
		"""
		if not self._lora_enabled:
			logger.warning("Cannot load adapter '%s': LoRA not enabled", name)
			return False

		if name in self._loaded_adapters:
			logger.info("Adapter '%s' already loaded", name)
			return True

		adapter_path = Path(path)
		if not adapter_path.exists():
			logger.error("Adapter path does not exist: %s", path)
			return False

		# Read adapter metadata if available
		metadata = {}
		metadata_file = adapter_path / "metadata.json"
		if metadata_file.exists():
			try:
				with open(metadata_file) as f:
					metadata = json.load(f)
			except Exception as e:
				logger.warning("Could not read adapter metadata: %s", e)

		# Read adapter config to get rank
		rank = 8  # default
		config_file = adapter_path / "adapter_config.json"
		if config_file.exists():
			try:
				with open(config_file) as f:
					adapter_config = json.load(f)
					rank = adapter_config.get("r", adapter_config.get("lora_r", 8))
			except Exception as e:
				logger.warning("Could not read adapter config: %s", e)

		# Assign a unique LoRA ID
		lora_id = self._next_lora_id
		self._next_lora_id += 1

		adapter_info = LoRAAdapterInfo(
			name=name,
			path=str(adapter_path),
			tier=tier,
			rank=rank,
			loaded=True,
			lora_id=lora_id,
			metadata=metadata
		)
		self._loaded_adapters[name] = adapter_info

		logger.info("Loaded LoRA adapter '%s' (id=%d, tier=%d, rank=%d) from %s",
			name, lora_id, tier, rank, path)
		return True

	def unload_adapter(self, name: str) -> bool:
		"""
		Unload a LoRA adapter.

		Args:
			name: Name of the adapter to unload

		Returns:
			True if adapter was unloaded successfully
		"""
		if name not in self._loaded_adapters:
			logger.warning("Adapter '%s' not loaded", name)
			return False

		adapter_info = self._loaded_adapters.pop(name)
		logger.info("Unloaded LoRA adapter '%s' (id=%d)", name, adapter_info.lora_id)
		return True

	def get_loaded_adapters(self) -> List[LoRAAdapterInfo]:
		"""Get list of currently loaded adapters."""
		return list(self._loaded_adapters.values())

	def get_adapter(self, name: str) -> Optional[LoRAAdapterInfo]:
		"""Get info for a specific adapter."""
		return self._loaded_adapters.get(name)

	def create_lora_request(self, adapter_name: str) -> Optional[Any]:
		"""
		Create a LoRARequest object for use with vLLM generation.

		Args:
			adapter_name: Name of the loaded adapter

		Returns:
			LoRARequest object or None if adapter not found
		"""
		if not self._lora_enabled or LoRARequest is None:
			return None

		adapter = self._loaded_adapters.get(adapter_name)
		if not adapter:
			logger.warning("Adapter '%s' not loaded", adapter_name)
			return None

		return LoRARequest(
			lora_name=adapter.name,
			lora_int_id=adapter.lora_id,
			lora_path=adapter.path,
		)

	def generate_with_adapter(
		self,
		prompts: List[str],
		adapter_name: str,
		sampling_params: Optional[Any] = None,
		max_tokens: int = 2048,
		temperature: float = 0.7,
		top_p: float = 0.95,
		**kwargs
	) -> List[Any]:
		"""
		Generate completions using a specific LoRA adapter.

		Args:
			prompts: List of prompts to generate from
			adapter_name: Name of the adapter to use
			sampling_params: Optional pre-built SamplingParams
			max_tokens: Maximum tokens to generate
			temperature: Sampling temperature
			top_p: Top-p sampling parameter

		Returns:
			List of generation outputs
		"""
		if sampling_params is None:
			sampling_params = self._build_sampling_params(
				max_tokens=max_tokens,
				temperature=temperature,
				top_p=top_p,
				stop=None
			)

		lora_request = self.create_lora_request(adapter_name)
		if lora_request is None:
			logger.warning("Generating without adapter (adapter '%s' not available)", adapter_name)
			return self.llm.generate(prompts=prompts, sampling_params=sampling_params)

		logger.info("Generating with LoRA adapter '%s'", adapter_name)
		return self.llm.generate(
			prompts=prompts,
			sampling_params=sampling_params,
			lora_request=lora_request
		)

	def create_chat_completion_with_adapter(
		self,
		messages: List[Dict[str, Any]],
		adapter_name: str,
		max_tokens: int = 2048,
		temperature: float = 0.7,
		top_p: float = 0.95,
		**kwargs
	) -> Dict[str, Any]:
		"""
		Create a chat completion using a specific LoRA adapter.

		Args:
			messages: Chat messages
			adapter_name: Name of the adapter to use
			max_tokens: Maximum tokens to generate
			temperature: Sampling temperature
			top_p: Top-p sampling parameter

		Returns:
			Completion result dict
		"""
		try:
			prompt = self._messages_to_prompt(messages)
		except Exception as e:
			logger.exception("Error in _messages_to_prompt: %s", e)
			prompt = "\n".join([f"{m['role']}: {m['content']}" for m in messages])

		sampling_params = self._build_sampling_params(
			max_tokens=max_tokens,
			temperature=temperature,
			top_p=top_p,
			stop=None
		)

		outputs = self.generate_with_adapter(
			prompts=[prompt],
			adapter_name=adapter_name,
			sampling_params=sampling_params
		)

		text = self._extract_text(outputs)
		return {"choices": [{"message": {"content": text}}]}
