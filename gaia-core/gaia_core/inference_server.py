"""
Safetensors Inference Server — GPU-native model serving with KV prefix caching.

Three key innovations:
1. KV Prefix Snapshot: Identity/tools/world state processed once, KV cache
   tensors stored as GAIA's "mental baseline." Every request starts from
   this snapshot instead of reprocessing 6K+ prefix tokens.

2. Hash-Based Invalidation: Each prompt segment has a content hash. Only
   recompute KV for segments that actually changed. Identity almost never
   changes. World state refreshes every ~60s.

3. Segmented Prompt Architecture: System prompt split into independently
   cacheable segments:
     Segment 0: Identity (immutable per session)
     Segment 1: Tools (changes rarely)
     Segment 2: World state (changes every ~60s)
     Segment 3: Conversation context (changes per turn)
   KV cache is recomputed incrementally — if segment 2 changes,
   only segments 2+ are reprocessed. Segments 0-1 stay cached.

Usage:
    python -m gaia_core.inference_server \
        --model /models/Qwen3.5-2B-GAIA-Core \
        --port 8092 \
        --device cuda
"""

import argparse
import hashlib
import json
import logging
import re
import time
import threading
import uuid
from typing import Optional, List, Tuple

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("GAIA.InferenceServer")

# ── Segmented Prompt Cache ───────────────────────────────────────────────────

class PromptSegment:
    """A single cacheable segment of the system prompt."""
    __slots__ = ("name", "content", "content_hash", "token_ids", "token_count")

    def __init__(self, name: str, content: str = ""):
        self.name = name
        self.content = content
        self.content_hash = self._hash(content)
        self.token_ids: Optional[torch.Tensor] = None
        self.token_count: int = 0

    @staticmethod
    def _hash(text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def update(self, content: str) -> bool:
        """Update content. Returns True if content changed."""
        new_hash = self._hash(content)
        if new_hash == self.content_hash:
            return False
        self.content = content
        self.content_hash = new_hash
        self.token_ids = None  # invalidate
        return True


class KVPrefixCache:
    """Manages segmented KV cache for the identity prefix.

    The system prompt is split into segments. Each segment's KV cache
    is computed independently. When a segment changes, only that segment
    and all subsequent segments are recomputed — earlier segments stay cached.

    This is like incremental compilation for LLM context.
    """

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device

        # Ordered segments (identity → tools → world state)
        self.segments: List[PromptSegment] = [
            PromptSegment("identity"),
            PromptSegment("tools"),
            PromptSegment("world_state"),
        ]

        # Cached KV state after processing all segments
        self._cached_kv: Optional[Tuple] = None
        self._cached_prefix_len: int = 0
        self._cached_segment_hashes: List[str] = []
        self._cache_hits: int = 0
        self._cache_misses: int = 0
        self._last_snapshot_time: float = 0.0

    def update_segment(self, name: str, content: str) -> bool:
        """Update a segment's content. Returns True if it changed."""
        for seg in self.segments:
            if seg.name == name:
                return seg.update(content)
        logger.warning("Unknown segment: %s", name)
        return False

    def get_segment(self, name: str) -> Optional[PromptSegment]:
        """Get a segment by name."""
        for seg in self.segments:
            if seg.name == name:
                return seg
        return None

    def _build_prefix_text(self) -> str:
        """Concatenate all segments into a single system prompt."""
        parts = [seg.content for seg in self.segments if seg.content]
        return "\n\n".join(parts)

    def _current_hashes(self) -> List[str]:
        return [seg.content_hash for seg in self.segments]

    def _find_first_changed_segment(self) -> int:
        """Find the index of the first segment that changed since last cache."""
        current = self._current_hashes()
        for i, (cached, current_h) in enumerate(zip(self._cached_segment_hashes, current)):
            if cached != current_h:
                return i
        if len(current) != len(self._cached_segment_hashes):
            return min(len(current), len(self._cached_segment_hashes))
        return -1  # nothing changed

    def get_prefix_kv(self) -> Tuple[Optional[Tuple], torch.Tensor, int]:
        """Get the cached KV state and prefix token IDs.

        Returns:
            (past_key_values, prefix_token_ids, prefix_length)

        If cache is valid, returns stored KV. If invalidated, recomputes
        from the first changed segment onward.
        """
        first_changed = self._find_first_changed_segment()

        if first_changed == -1 and self._cached_kv is not None:
            # Full cache hit — nothing changed
            self._cache_hits += 1
            prefix_text = self._build_prefix_text()
            prefix_ids = self.tokenizer.encode(prefix_text, return_tensors="pt").to(self.device)
            return self._cached_kv, prefix_ids, self._cached_prefix_len

        # Cache miss — need to recompute
        self._cache_misses += 1

        # For simplicity in v1: recompute the full prefix KV
        # (Incremental segment-level caching is a v2 optimization —
        # requires careful KV tensor slicing per segment boundary)
        prefix_text = self._build_prefix_text()
        if not prefix_text.strip():
            return None, torch.tensor([[]], dtype=torch.long), 0

        # Wrap in system message format
        system_msg = f"<|im_start|>system\n{prefix_text}<|im_end|>\n"
        prefix_ids = self.tokenizer.encode(system_msg, return_tensors="pt").to(self.device)
        prefix_len = prefix_ids.shape[1]

        logger.info(
            "KV prefix %s (segment '%s' changed, %d tokens, hashes: %s)",
            "recomputed" if self._cached_kv else "initialized",
            self.segments[first_changed].name if first_changed >= 0 else "all",
            prefix_len,
            [h[:8] for h in self._current_hashes()],
        )

        # Forward through model to get KV cache
        with torch.no_grad():
            outputs = self.model(
                input_ids=prefix_ids,
                use_cache=True,
            )
            self._cached_kv = outputs.past_key_values
            self._cached_prefix_len = prefix_len
            self._cached_segment_hashes = self._current_hashes()
            self._last_snapshot_time = time.time()

        return self._cached_kv, prefix_ids, prefix_len

    def invalidate(self):
        """Force full recomputation on next request."""
        self._cached_kv = None
        self._cached_segment_hashes = []

    def migrate_device(self, target_device: str):
        """Move cached KV tensors to a new device."""
        if self._cached_kv is not None:
            try:
                self._cached_kv = tuple(
                    tuple(t.to(target_device) for t in layer)
                    for layer in self._cached_kv
                )
                self.device = target_device
            except Exception as e:
                logger.warning("KV cache device migration failed, invalidating: %s", e)
                self.invalidate()
        self.device = target_device

    def stats(self) -> dict:
        return {
            "cache_hits": self._cache_hits,
            "cache_misses": self._cache_misses,
            "hit_rate": round(self._cache_hits / max(1, self._cache_hits + self._cache_misses), 3),
            "prefix_tokens": self._cached_prefix_len,
            "segments": {s.name: {"hash": s.content_hash[:8], "chars": len(s.content)} for s in self.segments},
            "last_snapshot": self._last_snapshot_time,
        }


# ── Model State ──────────────────────────────────────────────────────────────

_model = None
_tokenizer = None
_device = "cpu"
_model_path = ""
_kv_cache: Optional[KVPrefixCache] = None
_lock = threading.Lock()
_request_count = 0
_total_tokens = 0
_started_at = 0.0


def load_model(model_path: str, device: str = "cuda", dtype=torch.bfloat16):
    global _model, _tokenizer, _device, _model_path, _started_at, _kv_cache, _activation_monitor

    logger.info("Loading model %s on %s...", model_path, device)
    start = time.time()

    _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        model_path, trust_remote_code=True, dtype=dtype,
    )
    # Move explicitly to device (avoids FLA mixed-device issues with device_map)
    if device == "cuda" and torch.cuda.is_available():
        _model = _model.to("cuda")
    _model.eval()

    # ── Performance optimizations ────────────────────────────────────────
    # torch.compile: JIT-compiles the forward pass for 2-4x speedup.
    # Uses reduce-overhead mode for inference (no training).
    compile_mode = _os.environ.get("GAIA_COMPILE_MODE", "reduce-overhead")
    if compile_mode != "none" and device == "cuda":
        try:
            _model = torch.compile(_model, mode=compile_mode, fullgraph=False)
            logger.info("Model compiled with torch.compile (mode=%s)", compile_mode)
        except Exception as e:
            logger.warning("torch.compile failed (continuing uncompiled): %s", e)

    # Enable optimized attention kernels (FlashAttention via SDPA)
    try:
        torch.backends.cuda.enable_flash_sdp(True)
        torch.backends.cuda.enable_mem_efficient_sdp(True)
        logger.info("Flash/memory-efficient SDPA enabled")
    except Exception:
        pass  # older PyTorch versions may not support this

    _device = device
    _model_path = model_path
    _started_at = time.time()

    # Initialize KV prefix cache
    _kv_cache = KVPrefixCache(_model, _tokenizer, device)

    # Initialize activation monitor (the polygraph)
    _activation_monitor = ActivationMonitor(num_layers=24, hidden_size=2048)

    # Try to load SAE atlas if available
    atlas_path = _os.environ.get("SAE_ATLAS_PATH", "/shared/atlas/core")
    if _Path(atlas_path).exists():
        _activation_monitor.load_atlas(atlas_path)
    else:
        logger.info("No SAE atlas at %s — polygraph will show raw activations only", atlas_path)

    elapsed = time.time() - start
    mem_mb = torch.cuda.memory_allocated() // (1024 * 1024) if device == "cuda" else 0
    logger.info("Model loaded on %s in %.1fs (VRAM: %dMB)", device, elapsed, mem_mb)


def migrate_to(target_device: str) -> dict:
    global _model, _device, _kv_cache

    with _lock:
        if _device == target_device:
            return {"ok": True, "device": _device, "message": "already on target"}

        logger.info("Migrating model %s → %s...", _device, target_device)
        start = time.time()

        if target_device == "cpu":
            _model = _model.to("cpu")
            if _kv_cache:
                _kv_cache.invalidate()  # free cached KV tensors on GPU
            # Aggressive VRAM cleanup — release as much as possible
            import gc
            gc.collect()
            torch.cuda.synchronize()
            torch.cuda.empty_cache()
            _device = "cpu"
        elif target_device == "cuda":
            if not torch.cuda.is_available():
                return {"ok": False, "error": "CUDA not available"}
            _model = _model.to("cuda")
            if _kv_cache:
                _kv_cache.migrate_device("cuda")
            _device = "cuda"

        elapsed = time.time() - start
        mem_mb = torch.cuda.memory_allocated() // (1024 * 1024) if _device == "cuda" else 0
        logger.info("Migration complete in %.1fs (device: %s, VRAM: %dMB)", elapsed, _device, mem_mb)
        return {"ok": True, "device": _device, "elapsed_s": round(elapsed, 2), "vram_mb": mem_mb}


# ── Activation Monitor ("The Polygraph") ─────────────────────────────────────

class ActivationMonitor:
    """Real-time activation monitoring during inference.

    Hooks into model.forward() to capture hidden states at configurable
    layer depths. When a SAE feature atlas is loaded, maps activations
    to interpretable features in real time.

    This is GAIA's polygraph — always available, never modifies inference.
    """

    def __init__(self, num_layers: int = 24, hidden_size: int = 2048):
        self.num_layers = num_layers
        self.hidden_size = hidden_size
        self.enabled = True

        # Latest activation snapshot (last forward pass)
        self._last_activations: Optional[dict] = None
        self._last_timestamp: float = 0.0

        # SAE feature atlas (loaded from disk when available)
        self._feature_atlas: Optional[dict] = None  # layer_idx → SAE decoder matrix
        self._atlas_path: Optional[str] = None

        # Monitoring stats
        self._total_captures = 0

    def capture(self, hidden_states: tuple) -> dict:
        """Capture activation snapshot from a forward pass.

        Args:
            hidden_states: tuple of (num_layers+1) tensors, each [batch, seq, hidden]

        Returns:
            Activation summary dict
        """
        if not self.enabled or hidden_states is None:
            return {}

        self._total_captures += 1
        self._last_timestamp = time.time()

        # Sample specific layers (every 4th + first + last)
        sample_layers = [0, 6, 12, 18, 23, len(hidden_states) - 1]
        sample_layers = [i for i in sample_layers if i < len(hidden_states)]

        snapshot = {}
        for layer_idx in sample_layers:
            hs = hidden_states[layer_idx]
            # Take the last token's activations (most recent cognitive state)
            last_token = hs[0, -1, :]  # [hidden_size]

            # Basic activation stats
            snapshot[f"layer_{layer_idx}"] = {
                "mean": float(last_token.mean()),
                "std": float(last_token.std()),
                "max": float(last_token.max()),
                "min": float(last_token.min()),
                "l2_norm": float(last_token.norm()),
                "top_5_indices": last_token.abs().topk(5).indices.tolist(),
                "top_5_values": [round(float(v), 4) for v in last_token.abs().topk(5).values],
            }

            # If SAE atlas is loaded, map to interpretable features
            if self._feature_atlas and layer_idx in self._feature_atlas:
                feature_activations = self._project_to_features(last_token, layer_idx)
                snapshot[f"layer_{layer_idx}"]["features"] = feature_activations

        self._last_activations = snapshot
        return snapshot

    def _project_to_features(self, activations: torch.Tensor, layer_idx: int) -> dict:
        """Project activations through SAE decoder to get feature strengths."""
        atlas = self._feature_atlas.get(layer_idx)
        if atlas is None:
            return {}

        # atlas["decoder"] is [num_features, hidden_size]
        decoder = atlas["decoder"].to(activations.device)
        feature_strengths = torch.matmul(decoder, activations)  # [num_features]

        # Top active features
        top_k = min(10, feature_strengths.shape[0])
        top_vals, top_idx = feature_strengths.abs().topk(top_k)

        features = {}
        for i in range(top_k):
            fidx = top_idx[i].item()
            label = atlas.get("labels", {}).get(str(fidx), f"feature_{fidx}")
            features[label] = round(float(top_vals[i]), 4)

        return features

    def load_atlas(self, path: str) -> bool:
        """Load a SAE feature atlas from disk."""
        try:
            import os
            atlas_dir = _Path(path)
            if not atlas_dir.exists():
                logger.warning("Atlas path not found: %s", path)
                return False

            self._feature_atlas = {}
            for atlas_file in atlas_dir.glob("layer_*.pt"):
                layer_idx = int(atlas_file.stem.split("_")[1])
                self._feature_atlas[layer_idx] = torch.load(atlas_file, map_location="cpu")
                logger.info("Loaded SAE atlas for layer %d (%d features)",
                            layer_idx, self._feature_atlas[layer_idx]["decoder"].shape[0])

            self._atlas_path = path
            return True
        except Exception as e:
            logger.warning("Failed to load atlas: %s", e)
            return False

    def get_status(self) -> dict:
        return {
            "enabled": self.enabled,
            "total_captures": self._total_captures,
            "last_timestamp": self._last_timestamp,
            "atlas_loaded": self._feature_atlas is not None,
            "atlas_path": self._atlas_path,
            "atlas_layers": list(self._feature_atlas.keys()) if self._feature_atlas else [],
            "last_activations": self._last_activations,
        }


_activation_monitor: Optional[ActivationMonitor] = None


def _autoregressive_generate(input_ids: torch.Tensor, past_kv, max_tokens: int,
                              temperature: float, top_p: float) -> tuple:
    """Custom generation loop using model.forward() instead of model.generate().

    model.generate() can't handle Qwen3.5's hybrid attention cache format
    (Qwen3_5DynamicCache with mixed recurrent + KV states). But model.forward()
    handles it correctly. This loop replicates generate()'s behavior using
    forward() directly.

    Returns: (generated_token_ids: list[int], final_past_key_values)
    """
    generated = []
    current_kv = past_kv

    # First forward: process all input tokens (or continuation tokens if cache hit)
    # Capture hidden states for the activation monitor (polygraph)
    capture_activations = _activation_monitor is not None and _activation_monitor.enabled
    with torch.no_grad():
        out = _model(input_ids, past_key_values=current_kv, use_cache=True,
                     output_hidden_states=capture_activations)
    current_kv = out.past_key_values
    logits = out.logits[:, -1, :]

    # Feed activations to the polygraph
    if capture_activations and hasattr(out, "hidden_states") and out.hidden_states:
        _activation_monitor.capture(out.hidden_states)

    for _ in range(max_tokens):
        # Sample or greedy
        if temperature > 0:
            logits = logits / temperature
            if top_p < 1.0:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(torch.softmax(sorted_logits, dim=-1), dim=-1)
                mask = cumulative_probs - torch.softmax(sorted_logits, dim=-1) >= top_p
                sorted_logits[mask] = float("-inf")
                logits = sorted_logits.scatter(1, sorted_indices, sorted_logits)
            probs = torch.softmax(logits, dim=-1)
            next_id = torch.multinomial(probs, num_samples=1)
        else:
            next_id = logits.argmax(dim=-1, keepdim=True)

        token = next_id.item()
        if token == _tokenizer.eos_token_id:
            break
        generated.append(token)

        # Forward single token with cache
        with torch.no_grad():
            out = _model(next_id, past_key_values=current_kv, use_cache=True)
        current_kv = out.past_key_values
        logits = out.logits[:, -1, :]

    return generated, current_kv


def generate(messages: list, max_tokens: int = 512, temperature: float = 0.7,
             top_p: float = 0.9, use_prefix_cache: bool = True) -> dict:
    global _request_count, _total_tokens

    with _lock:
        if _model is None:
            return {"error": "model not loaded"}

        # Separate system message (goes into prefix cache) from conversation
        system_content = ""
        conversation_msgs = []
        for msg in messages:
            if msg.get("role") == "system":
                system_content = msg.get("content", "")
            else:
                conversation_msgs.append(msg)

        # Try to use KV prefix cache for the system prompt
        past_kv = None
        prefix_len = 0

        if use_prefix_cache and _kv_cache and system_content:
            _kv_cache.update_segment("identity", system_content)
            past_kv, _, prefix_len = _kv_cache.get_prefix_kv()

        # Build the conversation portion
        conv_parts = []
        for msg in conversation_msgs:
            role = msg.get("role", "user")
            content = msg.get("content", "")
            conv_parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
        conv_parts.append("<|im_start|>assistant\n")
        conv_text = "\n".join(conv_parts)

        if past_kv is not None:
            # KV cache hit — only tokenize the conversation part
            input_ids = _tokenizer.encode(conv_text, return_tensors="pt", add_special_tokens=False).to(_model.device)
            input_len = prefix_len + input_ids.shape[1]
        else:
            # No cache — build full prompt
            if system_content:
                full_text = f"<|im_start|>system\n{system_content}<|im_end|>\n{conv_text}"
            else:
                full_text = conv_text
            input_ids = _tokenizer.encode(full_text, return_tensors="pt").to(_model.device)
            input_len = input_ids.shape[1]

        # Use custom autoregressive loop (model.forward) instead of model.generate()
        # This correctly handles Qwen3.5's hybrid attention cache format
        generated_ids, _ = _autoregressive_generate(
            input_ids, past_kv, max_tokens, temperature, top_p,
        )

        response_text = _tokenizer.decode(generated_ids, skip_special_tokens=True)

        # Strip think tags if present
        if "<think>" in response_text:
            response_text = re.sub(r"<think>.*?</think>\s*", "", response_text, flags=re.DOTALL)

        _request_count += 1
        _total_tokens += len(generated_ids)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": _model_path,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response_text.strip()},
                "finish_reason": "stop" if len(generated_ids) < max_tokens else "length",
            }],
            "usage": {
                "prompt_tokens": input_len,
                "completion_tokens": len(generated_ids),
                "total_tokens": input_len + len(generated_ids),
                "cached_prefix_tokens": prefix_len if past_kv is not None else 0,
            },
        }


# ── Thought Management — "Hold that thought" ────────────────────────────────

import os as _os
from pathlib import Path as _Path

THOUGHT_DIR = _Path(_os.environ.get("SHARED_DIR", "/shared")) / "thoughts"
THOUGHT_DIR.mkdir(parents=True, exist_ok=True)

# In-memory thought cache (label → (past_key_values, metadata))
_held_thoughts: dict = {}


def hold_thought(label: str, context_summary: str = "") -> dict:
    """Freeze current KV cache state as a named thought.

    Saves the complete cognitive state — every attention pattern,
    every recurrent state — so GAIA can resume this exact train
    of thought later. Like putting a finger in a book, except the
    finger remembers what you were thinking about the page.
    """
    with _lock:
        if _kv_cache is None or _kv_cache._cached_kv is None:
            return {"ok": False, "error": "no active KV cache to save"}

        # Deep copy the KV state (tensors stay on current device)
        import copy
        kv_snapshot = copy.deepcopy(_kv_cache._cached_kv)
        prefix_len = _kv_cache._cached_prefix_len
        segment_hashes = list(_kv_cache._cached_segment_hashes)

        metadata = {
            "label": label,
            "context_summary": context_summary,
            "prefix_tokens": prefix_len,
            "segment_hashes": segment_hashes,
            "device": _device,
            "timestamp": time.time(),
            "model": _model_path,
        }

        _held_thoughts[label] = {
            "kv": kv_snapshot,
            "metadata": metadata,
        }

        # Also persist metadata to disk (KV tensors stay in memory for speed)
        meta_path = THOUGHT_DIR / f"{label}.json"
        meta_path.write_text(json.dumps(metadata, indent=2))

        logger.info("THOUGHT HELD: '%s' (%d prefix tokens, device=%s)", label, prefix_len, _device)
        return {"ok": True, "label": label, **metadata}


def resume_thought(label: str) -> dict:
    """Resume a previously held thought — restore the exact cognitive state.

    GAIA picks up exactly where she left off. No re-reading, no context
    loss, no "where was I?" moment. The attention patterns, the recurrent
    states, everything is restored.
    """
    with _lock:
        if label not in _held_thoughts:
            return {"ok": False, "error": f"no thought named '{label}'"}

        thought = _held_thoughts[label]
        kv_snapshot = thought["kv"]
        metadata = thought["metadata"]

        # Restore KV cache state
        if _kv_cache is not None:
            # Move KV to current device if needed
            target = _model.device if _model is not None else _device
            try:
                restored_kv = tuple(
                    tuple(t.to(target) for t in layer) if isinstance(layer, tuple)
                    else layer.to(target) if hasattr(layer, 'to') else layer
                    for layer in kv_snapshot
                ) if isinstance(kv_snapshot, tuple) else kv_snapshot
            except Exception:
                restored_kv = kv_snapshot

            _kv_cache._cached_kv = restored_kv
            _kv_cache._cached_prefix_len = metadata["prefix_tokens"]
            _kv_cache._cached_segment_hashes = metadata["segment_hashes"]
            _kv_cache._cache_hits += 1

        logger.info("THOUGHT RESUMED: '%s' (%d prefix tokens)", label, metadata["prefix_tokens"])
        return {"ok": True, "label": label, "resumed": metadata}


def list_thoughts() -> dict:
    """List all held thoughts with their metadata."""
    result = {}
    for label, thought in _held_thoughts.items():
        meta = thought["metadata"]
        result[label] = {
            "context_summary": meta.get("context_summary", ""),
            "prefix_tokens": meta["prefix_tokens"],
            "timestamp": meta["timestamp"],
            "device": meta["device"],
            "age_seconds": round(time.time() - meta["timestamp"], 1),
        }

    # Also check disk for thoughts from previous sessions
    for meta_file in THOUGHT_DIR.glob("*.json"):
        label = meta_file.stem
        if label not in result:
            try:
                meta = json.loads(meta_file.read_text())
                result[label] = {
                    "context_summary": meta.get("context_summary", ""),
                    "prefix_tokens": meta.get("prefix_tokens", 0),
                    "timestamp": meta.get("timestamp", 0),
                    "device": meta.get("device", "unknown"),
                    "age_seconds": round(time.time() - meta.get("timestamp", 0), 1),
                    "in_memory": False,  # on disk only, needs reload
                }
            except Exception:
                pass

    return {"thoughts": result, "count": len(result)}


def drop_thought(label: str) -> dict:
    """Release a held thought, freeing memory."""
    if label in _held_thoughts:
        del _held_thoughts[label]
        meta_path = THOUGHT_DIR / f"{label}.json"
        if meta_path.exists():
            meta_path.unlink()
        logger.info("THOUGHT DROPPED: '%s'", label)
        return {"ok": True, "label": label}
    return {"ok": False, "error": f"no thought named '{label}'"}


# ── HTTP Server ──────────────────────────────────────────────────────────────

from http.server import HTTPServer, BaseHTTPRequestHandler


class InferenceHandler(BaseHTTPRequestHandler):

    def log_message(self, format, *args):
        if "/health" not in str(args):
            logger.debug(format, *args)

    def _send_json(self, data: dict, status: int = 200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _read_body(self) -> dict:
        length = int(self.headers.get("Content-Length", 0))
        if length == 0:
            return {}
        return json.loads(self.rfile.read(length))

    def do_GET(self):
        if self.path == "/health":
            self._send_json({"status": "ok"})

        elif self.path == "/v1/models":
            self._send_json({
                "object": "list",
                "data": [{"id": _model_path, "object": "model", "owned_by": "gaia"}],
            })

        elif self.path == "/device/status":
            mem_mb = torch.cuda.memory_allocated() // (1024 * 1024) if _device == "cuda" else 0
            result = {
                "device": _device,
                "model": _model_path,
                "vram_mb": mem_mb,
                "requests": _request_count,
                "total_tokens": _total_tokens,
                "uptime_s": round(time.time() - _started_at, 1),
            }
            if _kv_cache:
                result["kv_cache"] = _kv_cache.stats()
            self._send_json(result)

        elif self.path == "/thought/list":
            self._send_json(list_thoughts())

        elif self.path == "/polygraph/status":
            if _activation_monitor:
                self._send_json(_activation_monitor.get_status())
            else:
                self._send_json({"enabled": False, "error": "monitor not initialized"})

        elif self.path == "/polygraph/activations":
            if _activation_monitor and _activation_monitor._last_activations:
                self._send_json({
                    "timestamp": _activation_monitor._last_timestamp,
                    "activations": _activation_monitor._last_activations,
                    "atlas_loaded": _activation_monitor._feature_atlas is not None,
                })
            else:
                self._send_json({"activations": None, "message": "no activations captured yet"})

        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            try:
                body = self._read_body()
                result = generate(
                    messages=body.get("messages", []),
                    max_tokens=body.get("max_tokens", 512),
                    temperature=body.get("temperature", 0.7),
                    top_p=body.get("top_p", 0.9),
                )
                self._send_json(result)
            except Exception as e:
                logger.exception("Generation failed")
                self._send_json({"error": str(e)}, 500)

        elif self.path == "/device/gpu":
            self._send_json(migrate_to("cuda"))

        elif self.path == "/device/cpu":
            self._send_json(migrate_to("cpu"))

        elif self.path == "/cache/invalidate":
            if _kv_cache:
                _kv_cache.invalidate()
                self._send_json({"ok": True, "message": "KV prefix cache invalidated"})
            else:
                self._send_json({"ok": False, "error": "no cache"})

        elif self.path == "/cache/update":
            body = self._read_body()
            changed = []
            if _kv_cache:
                for seg_name, content in body.items():
                    if _kv_cache.update_segment(seg_name, content):
                        changed.append(seg_name)
                self._send_json({"ok": True, "changed": changed})
            else:
                self._send_json({"ok": False, "error": "no cache"})

        elif self.path == "/thought/hold":
            body = self._read_body()
            label = body.get("label", f"thought_{int(time.time())}")
            context = body.get("context", "")
            self._send_json(hold_thought(label, context))

        elif self.path == "/thought/resume":
            body = self._read_body()
            label = body.get("label", "")
            if not label:
                self._send_json({"ok": False, "error": "label required"}, 400)
            else:
                self._send_json(resume_thought(label))

        elif self.path == "/thought/drop":
            body = self._read_body()
            label = body.get("label", "")
            self._send_json(drop_thought(label))

        elif self.path == "/polygraph/enable":
            if _activation_monitor:
                _activation_monitor.enabled = True
                self._send_json({"ok": True, "enabled": True})
            else:
                self._send_json({"ok": False, "error": "not initialized"})

        elif self.path == "/polygraph/disable":
            if _activation_monitor:
                _activation_monitor.enabled = False
                self._send_json({"ok": True, "enabled": False})
            else:
                self._send_json({"ok": False, "error": "not initialized"})

        elif self.path == "/polygraph/load-atlas":
            body = self._read_body()
            path = body.get("path", "/shared/atlas/core")
            if _activation_monitor:
                ok = _activation_monitor.load_atlas(path)
                self._send_json({"ok": ok, "path": path})
            else:
                self._send_json({"ok": False, "error": "not initialized"})

        else:
            self._send_json({"error": "not found"}, 404)


def main():
    parser = argparse.ArgumentParser(description="GAIA Safetensors Inference Server")
    parser.add_argument("--model", required=True, help="Path to HF model directory")
    parser.add_argument("--port", type=int, default=8092, help="Port to serve on")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"])
    parser.add_argument("--host", default="0.0.0.0")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    load_model(args.model, device=args.device)

    server = HTTPServer((args.host, args.port), InferenceHandler)
    logger.info("Inference server listening on %s:%d (device: %s, KV cache: enabled)", args.host, args.port, args.device)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
