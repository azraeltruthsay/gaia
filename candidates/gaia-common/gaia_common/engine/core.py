"""
GAIA Inference Engine — purpose-built inference for self-aware AI.

Not a general-purpose inference server. Optimized for:
- Single user, single GPU
- Sub-100ms latency on cached requests
- Hidden state access at every layer (polygraph)
- KV cache management with thought snapshots
- GPU↔CPU device migration
- Speculative decoding (Nano drafts, Core/Prime verifies)

Performance stack:
- torch.compile with reduce-overhead mode
- FlashAttention via SDPA
- Static KV cache pre-allocation
- Fused generation loop (minimize Python overhead)
- Optional: speculative decoding across tiers
"""

import gc
import hashlib
import json
import logging
import os
import re
import time
import threading
import uuid
from pathlib import Path
from typing import Optional, List, Tuple, Dict

import torch
import torch.nn.functional as F
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("GAIA.Engine")

# ── Static KV Cache ──────────────────────────────────────────────────────────

class StaticKVCache:
    """Pre-allocated KV cache to avoid dynamic allocation per token.

    Standard generation allocates new tensors for each token's key/value
    states. This pre-allocates a fixed buffer and writes into it,
    eliminating allocation overhead during generation.
    """

    def __init__(self, model, max_seq_len: int = 4096, device: str = "cuda"):
        self.max_seq_len = max_seq_len
        self.device = device
        self.position = 0

        # Extract model config
        config = model.config
        if hasattr(config, 'text_config'):
            config = config.text_config

        self.num_layers = getattr(config, 'num_hidden_layers', 24)
        self.num_kv_heads = getattr(config, 'num_key_value_heads', getattr(config, 'num_attention_heads', 8))
        self.head_dim = getattr(config, 'head_dim', getattr(config, 'hidden_size', 2048) // getattr(config, 'num_attention_heads', 8))

        # Pre-allocate buffers (only for standard attention models — Qwen3)
        # For Qwen3.5 hybrid, we fall back to dynamic cache
        self.is_standard_attention = not hasattr(config, 'layer_types')

        if self.is_standard_attention:
            self._k_cache = torch.zeros(
                self.num_layers, 1, self.num_kv_heads, max_seq_len, self.head_dim,
                dtype=torch.bfloat16, device=device,
            )
            self._v_cache = torch.zeros(
                self.num_layers, 1, self.num_kv_heads, max_seq_len, self.head_dim,
                dtype=torch.bfloat16, device=device,
            )
            mem_mb = (self._k_cache.nelement() + self._v_cache.nelement()) * 2 / (1024 * 1024)
            logger.info("Static KV cache allocated: %d layers × %d heads × %d seq × %d dim (%.0fMB)",
                        self.num_layers, self.num_kv_heads, max_seq_len, self.head_dim, mem_mb)
        else:
            logger.info("Hybrid attention detected — using dynamic KV cache")

    def reset(self):
        self.position = 0


# ── Activation Monitor ───────────────────────────────────────────────────────

class ActivationMonitor:
    """Real-time activation monitoring during inference."""

    def __init__(self):
        self.enabled = True
        self._last_snapshot: Optional[dict] = None
        self._last_timestamp: float = 0.0
        self._captures: int = 0

    def capture(self, hidden_states: tuple, sample_every: int = 4) -> dict:
        if not self.enabled or hidden_states is None:
            return {}

        self._captures += 1
        self._last_timestamp = time.time()
        num_layers = len(hidden_states)

        snapshot = {}
        sample_layers = [0] + list(range(sample_every, num_layers - 1, sample_every)) + [num_layers - 1]

        for idx in sample_layers:
            if idx >= num_layers:
                continue
            last_token = hidden_states[idx][0, -1, :]
            snapshot[f"layer_{idx}"] = {
                "mean": float(last_token.mean()),
                "std": float(last_token.std()),
                "l2_norm": float(last_token.norm()),
                "top_5_indices": last_token.abs().topk(5).indices.tolist(),
                "top_5_values": [round(float(v), 4) for v in last_token.abs().topk(5).values],
            }

        self._last_snapshot = snapshot
        return snapshot

    def stats(self) -> dict:
        return {
            "enabled": self.enabled,
            "captures": self._captures,
            "last_timestamp": self._last_timestamp,
            "last_snapshot": self._last_snapshot,
        }


# ── Thought Manager ──────────────────────────────────────────────────────────

class ThoughtManager:
    """Manages cognitive state snapshots via KV cache."""

    def __init__(self, storage_dir: str = "/shared/thoughts"):
        self._thoughts: Dict[str, dict] = {}
        self._dir = Path(storage_dir)
        self._dir.mkdir(parents=True, exist_ok=True)

    def hold(self, label: str, kv_state, prefix_len: int,
             segment_hashes: list, context: str = "") -> dict:
        import copy
        metadata = {
            "label": label, "context": context,
            "prefix_tokens": prefix_len,
            "segment_hashes": segment_hashes,
            "timestamp": time.time(),
        }
        self._thoughts[label] = {"kv": copy.deepcopy(kv_state), "meta": metadata}
        (self._dir / f"{label}.json").write_text(json.dumps(metadata, indent=2))
        logger.info("THOUGHT HELD: '%s' (%d tokens)", label, prefix_len)
        return {"ok": True, **metadata}

    def resume(self, label: str) -> Optional[dict]:
        if label not in self._thoughts:
            return None
        return self._thoughts[label]

    def list_all(self) -> dict:
        result = {}
        for label, thought in self._thoughts.items():
            m = thought["meta"]
            result[label] = {
                "context": m.get("context", ""),
                "prefix_tokens": m["prefix_tokens"],
                "age_s": round(time.time() - m["timestamp"], 1),
            }
        return {"thoughts": result, "count": len(result)}

    def drop(self, label: str) -> bool:
        if label in self._thoughts:
            del self._thoughts[label]
            p = self._dir / f"{label}.json"
            if p.exists():
                p.unlink()
            return True
        return False


# ── KV Prefix Cache ──────────────────────────────────────────────────────────

class PrefixCache:
    """Segmented KV prefix cache with hash-based invalidation."""

    def __init__(self, model, tokenizer, device: str = "cuda"):
        self.model = model
        self.tokenizer = tokenizer
        self.device = device
        self.segments = {"identity": "", "tools": "", "world_state": ""}
        self._hashes: Dict[str, str] = {}
        self._cached_kv = None
        self._cached_len = 0
        self._hits = 0
        self._misses = 0

    def _hash(self, text: str) -> str:
        return hashlib.sha256(text.encode()).hexdigest()[:16]

    def update_segment(self, name: str, content: str) -> bool:
        h = self._hash(content)
        if self._hashes.get(name) == h:
            return False
        self.segments[name] = content
        self._hashes[name] = h
        self._cached_kv = None  # invalidate
        return True

    def get_kv(self):
        current_hashes = {k: self._hash(v) for k, v in self.segments.items()}
        if self._cached_kv is not None and current_hashes == self._hashes:
            self._hits += 1
            return self._cached_kv, self._cached_len

        # Recompute
        self._misses += 1
        prefix = "\n\n".join(v for v in self.segments.values() if v)
        if not prefix.strip():
            return None, 0

        text = f"<|im_start|>system\n{prefix}<|im_end|>\n"
        ids = self.tokenizer.encode(text, return_tensors="pt").to(self.device)

        with torch.no_grad():
            out = self.model(ids, use_cache=True)
            self._cached_kv = out.past_key_values
            self._cached_len = ids.shape[1]
            self._hashes = current_hashes

        logger.info("KV prefix recomputed (%d tokens, segments: %s)",
                     self._cached_len, list(self._hashes.keys()))
        return self._cached_kv, self._cached_len

    def invalidate(self):
        self._cached_kv = None
        self._hashes = {}

    def stats(self) -> dict:
        return {
            "hits": self._hits, "misses": self._misses,
            "hit_rate": round(self._hits / max(1, self._hits + self._misses), 3),
            "prefix_tokens": self._cached_len,
            "segments": {k: len(v) for k, v in self.segments.items()},
        }


# ── GAIA Engine ──────────────────────────────────────────────────────────────

class GAIAEngine:
    """The GAIA Inference Engine — self-aware inference for a sovereign AI.

    Combines optimized generation with introspection capabilities that
    no general-purpose inference server provides.
    """

    def __init__(self, model_path: str, device: str = "cuda",
                 dtype=torch.bfloat16, compile_mode: str = "reduce-overhead"):
        self.model_path = model_path
        self.device = device
        self.dtype = dtype
        self._lock = threading.Lock()
        self._request_count = 0
        self._total_tokens = 0
        self._started_at = time.time()

        logger.info("GAIA Engine initializing: %s on %s", model_path, device)
        start = time.time()

        # Load tokenizer
        self.tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        # Load model
        self.model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, dtype=dtype,
        )
        if device == "cuda" and torch.cuda.is_available():
            self.model = self.model.to("cuda")
        self.model.eval()

        # Compile for speed — disable CUDA graphs to avoid conflicts
        # with dynamic KV cache sizes in autoregressive generation
        if compile_mode != "none" and device == "cuda":
            try:
                torch._dynamo.config.suppress_errors = True
                self.model = torch.compile(
                    self.model, mode=compile_mode, fullgraph=False,
                    options={"triton.cudagraphs": False},
                )
                logger.info("Model compiled (mode=%s, cudagraphs=off)", compile_mode)
            except Exception as e:
                logger.warning("torch.compile failed: %s", e)

        # Enable optimized attention
        try:
            torch.backends.cuda.enable_flash_sdp(True)
            torch.backends.cuda.enable_mem_efficient_sdp(True)
        except Exception:
            pass

        # Initialize subsystems
        self.prefix_cache = PrefixCache(self.model, self.tokenizer, device)
        self.monitor = ActivationMonitor()
        self.thoughts = ThoughtManager()

        # Initialize dynamic awareness
        try:
            from gaia_common.engine.awareness import AwarenessManager
            self.awareness = AwarenessManager()
            logger.info("Dynamic awareness initialized (%d packages)", len(self.awareness.packages))
        except Exception as e:
            logger.warning("Awareness system not available: %s", e)
            self.awareness = None

        elapsed = time.time() - start
        mem_mb = torch.cuda.memory_allocated() // (1024 * 1024) if device == "cuda" else 0
        logger.info("GAIA Engine ready in %.1fs (VRAM: %dMB)", elapsed, mem_mb)

    def generate(self, messages: list, max_tokens: int = 512,
                 temperature: float = 0.7, top_p: float = 0.9) -> dict:
        """Generate a chat completion with full introspection."""
        with self._lock:
            # Separate system message for prefix cache
            system = ""
            conversation = []
            for msg in messages:
                if msg.get("role") == "system":
                    system = msg.get("content", "")
                else:
                    conversation.append(msg)

            # CogPacket compression — skip sections already in KV cache or weights
            if system and len(system) > 500:
                try:
                    from gaia_common.engine.cogpacket_compressor import compress_system_prompt
                    system = compress_system_prompt(
                        system,
                        kv_cache=self.prefix_cache,
                        awareness=self.awareness,
                        sae_confident_topics=["identity"],  # SAE confirmed identity is weight-baked
                    )
                except Exception as e:
                    logger.debug("CogPacket compression failed (using full prompt): %s", e)

            # KV prefix cache — identity + situational awareness
            past_kv = None
            prefix_len = 0
            if system:
                self.prefix_cache.update_segment("identity", system)

                # Inject situational awareness if available
                if self.awareness:
                    user_text = " ".join(m.get("content", "") for m in conversation)
                    # Boost operational context when question is about architecture/services
                    boosts = {}
                    operational_signals = ['port', 'service', 'gaia-', 'tier', 'gpu', 'model', 'architecture']
                    if any(sig in user_text.lower() for sig in operational_signals):
                        boosts = {"operational": 0.5}
                    awareness_text = self.awareness.compose_awareness_text(
                        context=user_text, boost_categories=boosts,
                    )
                    if awareness_text:
                        self.prefix_cache.update_segment("world_state", awareness_text)

                past_kv, prefix_len = self.prefix_cache.get_kv()

            # Build conversation tokens
            parts = []
            for msg in conversation:
                parts.append(f"<|im_start|>{msg['role']}\n{msg['content']}<|im_end|>")
            parts.append("<|im_start|>assistant\n")
            conv_text = "\n".join(parts)

            if past_kv is not None:
                input_ids = self.tokenizer.encode(conv_text, return_tensors="pt",
                                                   add_special_tokens=False).to(self.model.device)
                total_input = prefix_len + input_ids.shape[1]
            else:
                full = f"<|im_start|>system\n{system}<|im_end|>\n{conv_text}" if system else conv_text
                input_ids = self.tokenizer.encode(full, return_tensors="pt").to(self.model.device)
                total_input = input_ids.shape[1]

            # ── Fused generation loop ────────────────────────────────────
            generated = []
            current_kv = past_kv

            # First forward — process input + capture activations
            capture = self.monitor.enabled
            with torch.no_grad():
                out = self.model(input_ids, past_key_values=current_kv,
                                  use_cache=True, output_hidden_states=capture)
            current_kv = out.past_key_values
            logits = out.logits[:, -1, :]

            if capture and hasattr(out, "hidden_states") and out.hidden_states:
                self.monitor.capture(out.hidden_states)

            # Autoregressive loop — minimal overhead
            eos_id = self.tokenizer.eos_token_id
            for _ in range(max_tokens):
                # Sample
                if temperature > 0:
                    scaled = logits / temperature
                    if top_p < 1.0:
                        sorted_logits, sorted_idx = torch.sort(scaled, descending=True)
                        cumprobs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                        mask = (cumprobs - F.softmax(sorted_logits, dim=-1)) >= top_p
                        sorted_logits[mask] = float("-inf")
                        logits = sorted_logits.scatter(1, sorted_idx, sorted_logits)
                    next_id = torch.multinomial(F.softmax(logits, dim=-1), 1)
                else:
                    next_id = logits.argmax(dim=-1, keepdim=True)

                token = next_id.item()
                if token == eos_id:
                    break
                generated.append(token)

                # Forward single token (no hidden states capture for speed)
                with torch.no_grad():
                    out = self.model(next_id, past_key_values=current_kv, use_cache=True)
                current_kv = out.past_key_values
                logits = out.logits[:, -1, :]

            # Decode
            text = self.tokenizer.decode(generated, skip_special_tokens=True)
            if "<think>" in text:
                text = re.sub(r"<think>.*?</think>\s*", "", text, flags=re.DOTALL)

            self._request_count += 1
            self._total_tokens += len(generated)

            return {
                "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
                "object": "chat.completion",
                "created": int(time.time()),
                "model": self.model_path,
                "choices": [{"index": 0, "message": {"role": "assistant", "content": text.strip()},
                             "finish_reason": "stop" if len(generated) < max_tokens else "length"}],
                "usage": {
                    "prompt_tokens": total_input,
                    "completion_tokens": len(generated),
                    "total_tokens": total_input + len(generated),
                    "cached_prefix_tokens": prefix_len if past_kv is not None else 0,
                },
            }

    def migrate_to(self, target: str) -> dict:
        """Migrate model between GPU and CPU."""
        with self._lock:
            if self.device == target:
                return {"ok": True, "device": self.device, "message": "already there"}

            start = time.time()
            if target == "cpu":
                self.model = self.model.to("cpu")
                self.prefix_cache.invalidate()
                gc.collect()
                torch.cuda.synchronize()
                torch.cuda.empty_cache()
            elif target == "cuda":
                self.model = self.model.to("cuda")

            self.device = target
            elapsed = time.time() - start
            mem = torch.cuda.memory_allocated() // (1024**2) if target == "cuda" else 0
            return {"ok": True, "device": target, "elapsed_s": round(elapsed, 2), "vram_mb": mem}

    def status(self) -> dict:
        mem = torch.cuda.memory_allocated() // (1024**2) if self.device == "cuda" else 0
        return {
            "model": self.model_path,
            "device": self.device,
            "vram_mb": mem,
            "requests": self._request_count,
            "total_tokens": self._total_tokens,
            "uptime_s": round(time.time() - self._started_at, 1),
            "compiled": hasattr(self.model, "_orig_mod"),
            "kv_cache": self.prefix_cache.stats(),
            "polygraph": self.monitor.stats(),
            "thoughts": self.thoughts.list_all(),
        }


# ── HTTP Server ──────────────────────────────────────────────────────────────

from http.server import HTTPServer, BaseHTTPRequestHandler

_engine: Optional[GAIAEngine] = None


class EngineHandler(BaseHTTPRequestHandler):

    def log_message(self, fmt, *args):
        if "/health" not in str(args):
            logger.debug(fmt, *args)

    def _json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get("Content-Length", 0))
        return json.loads(self.rfile.read(n)) if n else {}

    def do_GET(self):
        if self.path == "/health":
            self._json({"status": "ok", "engine": "gaia"})
        elif self.path == "/v1/models":
            self._json({"object": "list", "data": [{"id": _engine.model_path, "object": "model", "owned_by": "gaia"}]})
        elif self.path == "/status":
            self._json(_engine.status())
        elif self.path == "/polygraph/activations":
            self._json({"activations": _engine.monitor._last_snapshot, "timestamp": _engine.monitor._last_timestamp})
        elif self.path == "/thought/list":
            self._json(_engine.thoughts.list_all())
        elif self.path == "/awareness/status":
            if _engine.awareness:
                self._json(_engine.awareness.status())
            else:
                self._json({"error": "awareness not available"})
        elif self.path == "/awareness/curiosity":
            if _engine.awareness:
                self._json({"signals": _engine.awareness.get_curiosity_signals()})
            else:
                self._json({"signals": []})
        elif self.path == "/compression/stats":
            # Show what WOULD be compressed from a typical system prompt
            from gaia_common.engine.cogpacket_compressor import get_compression_stats
            # Use the last cached system prompt or a representative one
            sample = "You are GAIA, a sovereign AI. EPISTEMIC HONESTY rules apply. World State: Clock 2026. Reference Cheatsheets available."
            self._json(get_compression_stats(
                sample, _engine.prefix_cache, _engine.awareness,
                sae_confident_topics=["identity"],
            ))
        else:
            self._json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            try:
                b = self._body()
                self._json(_engine.generate(b.get("messages", []), b.get("max_tokens", 512),
                                             b.get("temperature", 0.7), b.get("top_p", 0.9)))
            except Exception as e:
                logger.exception("Generation failed")
                self._json({"error": str(e)}, 500)
        elif self.path == "/device/gpu":
            self._json(_engine.migrate_to("cuda"))
        elif self.path == "/device/cpu":
            self._json(_engine.migrate_to("cpu"))
        elif self.path == "/cache/update":
            b = self._body()
            changed = [k for k, v in b.items() if _engine.prefix_cache.update_segment(k, v)]
            self._json({"ok": True, "changed": changed})
        elif self.path == "/cache/invalidate":
            _engine.prefix_cache.invalidate()
            self._json({"ok": True})
        elif self.path == "/thought/hold":
            b = self._body()
            pc = _engine.prefix_cache
            self._json(_engine.thoughts.hold(
                b.get("label", f"t_{int(time.time())}"), pc._cached_kv,
                pc._cached_len, list(pc._hashes.values()), b.get("context", "")))
        elif self.path == "/thought/resume":
            b = self._body()
            t = _engine.thoughts.resume(b.get("label", ""))
            if t:
                _engine.prefix_cache._cached_kv = t["kv"]
                _engine.prefix_cache._cached_len = t["meta"]["prefix_tokens"]
                self._json({"ok": True, "resumed": t["meta"]})
            else:
                self._json({"ok": False, "error": "not found"}, 404)
        elif self.path == "/thought/drop":
            self._json({"ok": _engine.thoughts.drop(self._body().get("label", ""))})
        elif self.path == "/thought/compose":
            # Compose two held thoughts into a unified cognitive state
            # {"primary": "label_a", "secondary": "label_b", "shared_prefix": 14}
            b = self._body()
            primary = _engine.thoughts.resume(b.get("primary", ""))
            secondary = _engine.thoughts.resume(b.get("secondary", ""))
            if not primary or not secondary:
                self._json({"ok": False, "error": "one or both thoughts not found"}, 404)
            else:
                from gaia_common.engine.thought_composer import compose_thoughts, estimate_composed_size
                shared = b.get("shared_prefix", 0)
                pw = b.get("primary_weight", 0.6)
                sw = b.get("secondary_weight", 0.4)
                est = estimate_composed_size(primary["kv"], secondary["kv"], shared)
                composed_kv = compose_thoughts(primary["kv"], secondary["kv"], shared, pw, sw)
                # Install composed state as active KV cache
                _engine.prefix_cache._cached_kv = composed_kv
                _engine.prefix_cache._cached_len = est["composed_tokens"]
                self._json({"ok": True, "estimate": est,
                            "primary": b.get("primary"), "secondary": b.get("secondary")})
        elif self.path == "/thought/estimate-compose":
            b = self._body()
            primary = _engine.thoughts.resume(b.get("primary", ""))
            secondary = _engine.thoughts.resume(b.get("secondary", ""))
            if not primary or not secondary:
                self._json({"ok": False, "error": "not found"}, 404)
            else:
                from gaia_common.engine.thought_composer import estimate_composed_size
                est = estimate_composed_size(primary["kv"], secondary["kv"], b.get("shared_prefix", 0))
                self._json({"ok": True, "estimate": est})
        elif self.path == "/polygraph/enable":
            _engine.monitor.enabled = True
            self._json({"ok": True})
        elif self.path == "/polygraph/disable":
            _engine.monitor.enabled = False
            self._json({"ok": True})
        else:
            self._json({"error": "not found"}, 404)


def serve(model_path: str, port: int = 8092, device: str = "cuda",
          compile_mode: str = "reduce-overhead", host: str = "0.0.0.0"):
    """Start the GAIA Inference Engine."""
    global _engine

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    _engine = GAIAEngine(model_path, device=device, compile_mode=compile_mode)

    server = HTTPServer((host, port), EngineHandler)
    logger.info("GAIA Inference Engine serving on %s:%d", host, port)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="GAIA Inference Engine")
    p.add_argument("--model", required=True)
    p.add_argument("--port", type=int, default=8092)
    p.add_argument("--device", default="cuda")
    p.add_argument("--compile", default="reduce-overhead", choices=["reduce-overhead", "max-autotune", "none"])
    p.add_argument("--host", default="0.0.0.0")
    args = p.parse_args()
    serve(args.model, args.port, args.device, args.compile, args.host)
