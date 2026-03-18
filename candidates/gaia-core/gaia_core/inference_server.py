"""
Safetensors Inference Server — GPU-native model serving with live device migration.

Serves a HuggingFace transformers model via an OpenAI-compatible API.
Supports live GPU↔CPU migration for the GPU watch rotation protocol.

This replaces llama-server for models that need to stay in safetensors
format (for SAE analysis, ROME editing, steering vectors).

Usage:
    python -m gaia_core.inference_server \
        --model /models/Qwen3.5-2B-GAIA-Core \
        --port 8092 \
        --device cuda \
        --ctx-size 8192
"""

import argparse
import json
import logging
import time
import threading
import uuid
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

logger = logging.getLogger("GAIA.InferenceServer")

# Global model state
_model = None
_tokenizer = None
_device = "cpu"
_model_path = ""
_lock = threading.Lock()
_request_count = 0
_total_tokens = 0
_started_at = 0.0


def load_model(model_path: str, device: str = "cuda", dtype=torch.bfloat16):
    """Load model onto the specified device."""
    global _model, _tokenizer, _device, _model_path, _started_at

    logger.info("Loading model %s on %s...", model_path, device)
    start = time.time()

    _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    _model = AutoModelForCausalLM.from_pretrained(
        model_path,
        trust_remote_code=True,
        dtype=dtype,
        device_map=device if device == "cuda" else "cpu",
    )
    _model.eval()

    _device = device
    _model_path = model_path
    _started_at = time.time()
    elapsed = time.time() - start

    mem_mb = torch.cuda.memory_allocated() // (1024 * 1024) if device == "cuda" else 0
    logger.info("Model loaded on %s in %.1fs (VRAM: %dMB)", device, elapsed, mem_mb)


def migrate_to(target_device: str) -> dict:
    """Migrate model between GPU and CPU. Thread-safe."""
    global _model, _device

    with _lock:
        if _device == target_device:
            return {"ok": True, "device": _device, "message": "already on target"}

        logger.info("Migrating model %s → %s...", _device, target_device)
        start = time.time()

        if target_device == "cpu":
            _model = _model.to("cpu")
            torch.cuda.empty_cache()
            _device = "cpu"
        elif target_device == "cuda":
            if not torch.cuda.is_available():
                return {"ok": False, "error": "CUDA not available"}
            _model = _model.to("cuda")
            _device = "cuda"

        elapsed = time.time() - start
        mem_mb = torch.cuda.memory_allocated() // (1024 * 1024) if _device == "cuda" else 0
        logger.info("Migration complete in %.1fs (device: %s, VRAM: %dMB)", elapsed, _device, mem_mb)
        return {"ok": True, "device": _device, "elapsed_s": round(elapsed, 2), "vram_mb": mem_mb}


def generate(messages: list, max_tokens: int = 512, temperature: float = 0.7,
             top_p: float = 0.9, stop: Optional[list] = None) -> dict:
    """Generate a chat completion. Thread-safe."""
    global _request_count, _total_tokens

    with _lock:
        if _model is None:
            return {"error": "model not loaded"}

        # Build prompt from messages using the tokenizer's chat template
        try:
            text = _tokenizer.apply_chat_template(
                messages, tokenize=False, add_generation_prompt=True,
            )
        except Exception:
            # Fallback: manual ChatML format
            parts = []
            for msg in messages:
                role = msg.get("role", "user")
                content = msg.get("content", "")
                parts.append(f"<|im_start|>{role}\n{content}<|im_end|>")
            parts.append("<|im_start|>assistant\n")
            text = "\n".join(parts)

        inputs = _tokenizer(text, return_tensors="pt").to(_model.device)
        input_len = inputs["input_ids"].shape[1]

        gen_kwargs = {
            "max_new_tokens": max_tokens,
            "do_sample": temperature > 0,
            "pad_token_id": _tokenizer.pad_token_id,
        }
        if temperature > 0:
            gen_kwargs["temperature"] = temperature
            gen_kwargs["top_p"] = top_p

        with torch.no_grad():
            output = _model.generate(**inputs, **gen_kwargs)

        new_tokens = output[0][input_len:]
        response_text = _tokenizer.decode(new_tokens, skip_special_tokens=True)

        # Strip think tags if present
        if "<think>" in response_text:
            import re
            response_text = re.sub(r"<think>.*?</think>\s*", "", response_text, flags=re.DOTALL)

        _request_count += 1
        _total_tokens += len(new_tokens)

        return {
            "id": f"chatcmpl-{uuid.uuid4().hex[:8]}",
            "object": "chat.completion",
            "created": int(time.time()),
            "model": _model_path,
            "choices": [{
                "index": 0,
                "message": {"role": "assistant", "content": response_text.strip()},
                "finish_reason": "stop",
            }],
            "usage": {
                "prompt_tokens": input_len,
                "completion_tokens": len(new_tokens),
                "total_tokens": input_len + len(new_tokens),
            },
        }


# ── HTTP Server (stdlib, no Flask/FastAPI dependency) ────────────────────────

from http.server import HTTPServer, BaseHTTPRequestHandler


class InferenceHandler(BaseHTTPRequestHandler):
    """Minimal OpenAI-compatible HTTP handler."""

    def log_message(self, format, *args):
        # Suppress default access logs for health checks
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
                "data": [{
                    "id": _model_path,
                    "object": "model",
                    "owned_by": "gaia",
                }],
            })

        elif self.path == "/device/status":
            mem_mb = torch.cuda.memory_allocated() // (1024 * 1024) if _device == "cuda" else 0
            self._send_json({
                "device": _device,
                "model": _model_path,
                "vram_mb": mem_mb,
                "requests": _request_count,
                "total_tokens": _total_tokens,
                "uptime_s": round(time.time() - _started_at, 1),
            })

        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/v1/chat/completions":
            try:
                body = self._read_body()
                messages = body.get("messages", [])
                max_tokens = body.get("max_tokens", 512)
                temperature = body.get("temperature", 0.7)
                top_p = body.get("top_p", 0.9)

                result = generate(messages, max_tokens, temperature, top_p)
                self._send_json(result)
            except Exception as e:
                logger.exception("Generation failed")
                self._send_json({"error": str(e)}, 500)

        elif self.path == "/device/gpu":
            result = migrate_to("cuda")
            self._send_json(result)

        elif self.path == "/device/cpu":
            result = migrate_to("cpu")
            self._send_json(result)

        else:
            self._send_json({"error": "not found"}, 404)


def main():
    parser = argparse.ArgumentParser(description="GAIA Safetensors Inference Server")
    parser.add_argument("--model", required=True, help="Path to HF model directory")
    parser.add_argument("--port", type=int, default=8092, help="Port to serve on")
    parser.add_argument("--device", default="cuda", choices=["cuda", "cpu"], help="Initial device")
    parser.add_argument("--host", default="0.0.0.0", help="Host to bind to")
    args = parser.parse_args()

    logging.basicConfig(
        level=logging.INFO,
        format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    )

    load_model(args.model, device=args.device)

    server = HTTPServer((args.host, args.port), InferenceHandler)
    logger.info("Inference server listening on %s:%d (device: %s)", args.host, args.port, args.device)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        logger.info("Shutting down...")
        server.shutdown()


if __name__ == "__main__":
    main()
