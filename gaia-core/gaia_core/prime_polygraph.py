"""
Prime Polygraph — activation monitoring for the Thinker tier.

Loads Prime's safetensors in NF4 quantization (~4.5GB VRAM) for
activation analysis. Runs alongside Core/Nano during IDLE state.

Since Prime is Qwen3 (standard transformer, not hybrid Qwen3.5),
hidden states work cleanly with model.forward() and model.generate().

Usage:
    python -m gaia_core.prime_polygraph \
        --model /models/Huihui-Qwen3-8B-abliterated-v2-merged \
        --port 8094
"""

import argparse
import gc
import json
import logging
import time
import threading
import uuid
from typing import Optional

import torch
from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

logger = logging.getLogger("GAIA.PrimePolygraph")

_model = None
_tokenizer = None
_device = "cuda"
_model_path = ""
_lock = threading.Lock()
_request_count = 0
_last_activations = None
_last_timestamp = 0.0


def load_model(model_path: str, device: str = "auto"):
    """Load Prime for polygraph analysis.

    Tries NF4 quantization first (~4.5GB VRAM). Falls back to bf16 on
    CPU if bitsandbytes isn't available (~16GB RAM, slower but works).
    """
    global _model, _tokenizer, _model_path, _device

    logger.info("Loading Prime for polygraph analysis: %s", model_path)
    start = time.time()

    _tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)
    if _tokenizer.pad_token is None:
        _tokenizer.pad_token = _tokenizer.eos_token

    # Try NF4 quantization (GPU, ~4.5GB)
    try:
        from transformers import BitsAndBytesConfig
        bnb_config = BitsAndBytesConfig(
            load_in_4bit=True, bnb_4bit_quant_type="nf4",
            bnb_4bit_compute_dtype=torch.bfloat16, bnb_4bit_use_double_quant=True,
        )
        _model = AutoModelForCausalLM.from_pretrained(
            model_path, quantization_config=bnb_config, device_map="auto",
            trust_remote_code=True, dtype=torch.bfloat16,
        )
        _device = "cuda"
        logger.info("Prime loaded in NF4 on GPU")
    except (ImportError, Exception) as e:
        logger.info("NF4 not available (%s), loading bf16 on CPU", e)
        _model = AutoModelForCausalLM.from_pretrained(
            model_path, trust_remote_code=True, dtype=torch.bfloat16,
        )
        _device = "cpu"
        logger.info("Prime loaded in bf16 on CPU")

    _model.eval()
    _model_path = model_path

    elapsed = time.time() - start
    mem_mb = torch.cuda.memory_allocated() // (1024 * 1024)
    logger.info("Prime ready in %.1fs (VRAM: %dMB, device: %s)", elapsed, mem_mb, _device)


def unload_model():
    """Unload Prime to free VRAM."""
    global _model, _tokenizer
    _model = None
    _tokenizer = None
    gc.collect()
    torch.cuda.empty_cache()
    logger.info("Prime unloaded, VRAM freed")


def analyze(prompt: str, system: str = "", max_tokens: int = 50) -> dict:
    """Run a prompt through Prime and capture activations at all layers."""
    global _request_count, _last_activations, _last_timestamp

    with _lock:
        if _model is None:
            return {"error": "Prime not loaded"}

        # Build prompt
        parts = []
        if system:
            parts.append(f"<|im_start|>system\n{system}<|im_end|>")
        parts.append(f"<|im_start|>user\n{prompt}<|im_end|>")
        parts.append("<|im_start|>assistant\n")
        full_text = "\n".join(parts)

        input_ids = _tokenizer.encode(full_text, return_tensors="pt").to(_model.device)

        with torch.no_grad():
            out = _model(input_ids, output_hidden_states=True)

        hidden_states = out.hidden_states
        num_layers = len(hidden_states)

        # Capture activations at sampled layers
        sample_layers = [0] + list(range(4, num_layers - 1, 4)) + [num_layers - 1]
        activations = {}

        for layer_idx in sample_layers:
            if layer_idx >= num_layers:
                continue
            hs = hidden_states[layer_idx]
            last_token = hs[0, -1, :]  # last token activations

            activations[f"layer_{layer_idx}"] = {
                "mean": float(last_token.mean()),
                "std": float(last_token.std()),
                "max": float(last_token.max()),
                "min": float(last_token.min()),
                "l2_norm": float(last_token.norm()),
                "top_10_indices": last_token.abs().topk(10).indices.tolist(),
                "top_10_values": [round(float(v), 4) for v in last_token.abs().topk(10).values],
            }

        # Also generate a short response for context
        gen_ids = _model.generate(
            input_ids, max_new_tokens=max_tokens,
            do_sample=False, pad_token_id=_tokenizer.pad_token_id,
        )
        response = _tokenizer.decode(gen_ids[0][input_ids.shape[1]:], skip_special_tokens=True)

        _request_count += 1
        _last_activations = activations
        _last_timestamp = time.time()

        return {
            "model": _model_path,
            "prompt": prompt[:100],
            "response": response.strip()[:200],
            "num_layers": num_layers,
            "activations": activations,
            "vram_mb": torch.cuda.memory_allocated() // (1024 * 1024),
        }


# ── HTTP Server ──────────────────────────────────────────────────────────────

from http.server import HTTPServer, BaseHTTPRequestHandler


class PrimePolygraphHandler(BaseHTTPRequestHandler):

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
            self._send_json({"status": "ok", "model_loaded": _model is not None})

        elif self.path == "/status":
            self._send_json({
                "model_loaded": _model is not None,
                "model": _model_path,
                "requests": _request_count,
                "vram_mb": torch.cuda.memory_allocated() // (1024 * 1024) if _model else 0,
                "last_timestamp": _last_timestamp,
            })

        elif self.path == "/activations":
            self._send_json({
                "activations": _last_activations,
                "timestamp": _last_timestamp,
            })

        else:
            self._send_json({"error": "not found"}, 404)

    def do_POST(self):
        if self.path == "/analyze":
            body = self._read_body()
            prompt = body.get("prompt", "Who are you?")
            system = body.get("system", "You are GAIA, a sovereign AI.")
            max_tokens = body.get("max_tokens", 50)
            result = analyze(prompt, system, max_tokens)
            self._send_json(result)

        elif self.path == "/load":
            body = self._read_body()
            path = body.get("model", "/models/Huihui-Qwen3-8B-abliterated-v2-merged")
            try:
                load_model(path)
                self._send_json({"ok": True, "vram_mb": torch.cuda.memory_allocated() // (1024 * 1024)})
            except Exception as e:
                self._send_json({"ok": False, "error": str(e)}, 500)

        elif self.path == "/unload":
            unload_model()
            self._send_json({"ok": True})

        else:
            self._send_json({"error": "not found"}, 404)


def main():
    parser = argparse.ArgumentParser(description="Prime Polygraph — activation analysis server")
    parser.add_argument("--model", default="/models/Huihui-Qwen3-8B-abliterated-v2-merged")
    parser.add_argument("--port", type=int, default=8094)
    parser.add_argument("--no-autoload", action="store_true", help="Don't load model on startup")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(name)s: %(message)s")

    if not args.no_autoload:
        load_model(args.model)

    server = HTTPServer(("0.0.0.0", args.port), PrimePolygraphHandler)
    logger.info("Prime Polygraph listening on port %d", args.port)

    try:
        server.serve_forever()
    except KeyboardInterrupt:
        server.shutdown()


if __name__ == "__main__":
    main()
