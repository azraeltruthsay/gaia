#!/bin/bash
# gaia-prime entrypoint: GAIA Engine for the Thinker tier
#
# Starts the engine in "standby" mode — server runs but no model loaded.
# The orchestrator loads the model via POST /model/load when FOCUSING.
# Unloads via POST /model/unload when returning to IDLE.
#
# If PRIME_AUTOLOAD=1, loads the model on startup (for standalone use).

set -euo pipefail

PORT="${PRIME_PORT:-7777}"
MODEL_PATH="${PRIME_MODEL_PATH:-/models/Huihui-Qwen3-8B-GAIA-Prime-adaptive}"
DEVICE="${PRIME_DEVICE:-cuda}"
AUTOLOAD="${PRIME_AUTOLOAD:-0}"

mkdir -p /shared/thoughts 2>/dev/null || true

if [ "$AUTOLOAD" = "1" ] && [ -d "$MODEL_PATH" ]; then
    echo "[prime-entrypoint] Starting GAIA Engine with model auto-load..."
    echo "[prime-entrypoint] Model: $MODEL_PATH"
    exec python -c "from gaia_common.engine import serve; serve('$MODEL_PATH', $PORT, '$DEVICE')"
else
    echo "[prime-entrypoint] Starting GAIA Engine in standby mode (no model loaded)..."
    echo "[prime-entrypoint] Model will be loaded by orchestrator via POST /model/load"
    # Run engine with no model — serves health endpoint but returns errors on inference
    exec python -c "
from gaia_common.engine.core import GAIAEngine, serve
from http.server import HTTPServer, BaseHTTPRequestHandler
import json, logging, os

logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('GAIA.Prime')

_engine = None

class StandbyHandler(BaseHTTPRequestHandler):
    def log_message(self, fmt, *args):
        if '/health' not in str(args):
            logger.debug(fmt, *args)

    def _json(self, data, status=200):
        body = json.dumps(data).encode()
        self.send_response(status)
        self.send_header('Content-Type', 'application/json')
        self.send_header('Content-Length', str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _body(self):
        n = int(self.headers.get('Content-Length', 0))
        return json.loads(self.rfile.read(n)) if n else {}

    def do_GET(self):
        global _engine
        if self.path == '/health':
            self._json({'status': 'ok', 'model_loaded': _engine is not None, 'mode': 'active' if _engine else 'standby'})
        elif self.path == '/status':
            if _engine:
                self._json(_engine.status())
            else:
                self._json({'mode': 'standby', 'model_loaded': False})
        elif self.path == '/v1/models':
            if _engine:
                self._json({'object': 'list', 'data': [{'id': _engine.model_path, 'object': 'model'}]})
            else:
                self._json({'object': 'list', 'data': []})
        elif self.path == '/polygraph/activations':
            if _engine:
                self._json({'activations': _engine.monitor._last_snapshot, 'timestamp': _engine.monitor._last_timestamp})
            else:
                self._json({'activations': None, 'message': 'standby'})
        elif self.path == '/adapter/status':
            if _engine and hasattr(_engine, 'adapter_status'):
                self._json(_engine.adapter_status())
            else:
                self._json({'active': None, 'loaded': [], 'base_model': 'standby'})
        else:
            self._json({'error': 'not found'}, 404)

    def do_POST(self):
        global _engine
        if self.path == '/model/load':
            b = self._body()
            model_path = b.get('model', '$MODEL_PATH')
            device = b.get('device', '$DEVICE')
            quantize = b.get('quantize', 'int8')
            try:
                if quantize == 'int8':
                    logger.info('Loading Prime (int8 quanto): %s', model_path)
                    import torch
                    from transformers import AutoModelForCausalLM, AutoTokenizer
                    from optimum.quanto import quantize as q_quantize, freeze, qint8

                    model = AutoModelForCausalLM.from_pretrained(model_path, trust_remote_code=True, dtype=torch.bfloat16, device_map='cpu')
                    q_quantize(model, weights=qint8); freeze(model)
                    if device == 'cuda':
                        model = model.to('cuda')
                    tokenizer = AutoTokenizer.from_pretrained(model_path, trust_remote_code=True)

                    _engine = GAIAEngine.__new__(GAIAEngine)
                    _engine.model = model
                    _engine.model.eval()
                    _engine.tokenizer = tokenizer
                    _engine.model_path = model_path
                    _engine.device = device
                    _engine._lock = __import__('threading').Lock()
                    _engine._request_count = 0
                    _engine._total_tokens = 0
                    _engine._started_at = __import__('time').time()
                    from gaia_common.engine.core import PrefixCache, ActivationMonitor, ThoughtManager
                    _engine.prefix_cache = PrefixCache(model, tokenizer, device)
                    _engine.monitor = ActivationMonitor()
                    _engine.thoughts = ThoughtManager()
                    _engine.awareness = None

                    vram = torch.cuda.memory_allocated() // (1024**2) if device == 'cuda' else 0
                    logger.info('Prime loaded: %dMB VRAM', vram)
                    self._json({'ok': True, 'vram_mb': vram, 'model': model_path})
                else:
                    _engine = GAIAEngine(model_path, device=device)
                    self._json({'ok': True, 'model': model_path})
            except Exception as e:
                logger.exception('Model load failed')
                self._json({'ok': False, 'error': str(e)}, 500)

        elif self.path == '/model/unload':
            if _engine:
                import gc, torch
                _engine.model = None
                _engine.tokenizer = None
                _engine = None
                gc.collect()
                if torch.cuda.is_available():
                    torch.cuda.empty_cache()
                logger.info('Prime unloaded')
                self._json({'ok': True})
            else:
                self._json({'ok': True, 'message': 'already unloaded'})

        elif self.path == '/v1/chat/completions':
            if _engine:
                b = self._body()
                self._json(_engine.generate(b.get('messages', []), b.get('max_tokens', 512), b.get('temperature', 0.7), b.get('top_p', 0.9)))
            else:
                self._json({'error': 'model not loaded — Prime is in standby'}, 503)

        elif self.path in ('/device/gpu', '/device/cpu'):
            if _engine:
                target = 'cuda' if 'gpu' in self.path else 'cpu'
                self._json(_engine.migrate_to(target))
            else:
                self._json({'error': 'no model loaded'}, 503)

        elif self.path == '/thought/hold':
            if _engine:
                b = self._body()
                pc = _engine.prefix_cache
                self._json(_engine.thoughts.hold(b.get('label', 't'), pc._cached_kv, pc._cached_len, list(pc._hashes.values()), b.get('context', '')))
            else:
                self._json({'error': 'standby'}, 503)

        elif self.path == '/thought/resume':
            if _engine:
                b = self._body()
                t = _engine.thoughts.resume(b.get('label', ''))
                if t:
                    _engine.prefix_cache._cached_kv = t['kv']
                    _engine.prefix_cache._cached_len = t['meta']['prefix_tokens']
                    self._json({'ok': True, 'resumed': t['meta']})
                else:
                    self._json({'ok': False, 'error': 'not found'}, 404)
            else:
                self._json({'error': 'standby'}, 503)

        elif _engine and self.path in ('/atlas/record', '/polygraph/enable', '/polygraph/disable', '/cache/update', '/cache/invalidate', '/adapter/load', '/adapter/unload', '/adapter/set'):
            # Delegate to the real EngineHandler when model is loaded
            from gaia_common.engine.core import EngineHandler, _engine as _core_engine
            import gaia_common.engine.core as _core_mod
            # Inject our engine into the module so EngineHandler can find it
            _core_mod._engine = _engine
            handler = EngineHandler.__new__(EngineHandler)
            handler.rfile = self.rfile
            handler.wfile = self.wfile
            handler.headers = self.headers
            handler.path = self.path
            handler.requestline = self.requestline
            handler.client_address = self.client_address
            handler.server = self.server
            handler.command = self.command
            handler._json = self._json
            handler._body = self._body
            handler.do_POST()

        else:
            self._json({'error': 'not found'}, 404)

server = HTTPServer(('0.0.0.0', $PORT), StandbyHandler)
logger.info('GAIA Prime Engine (standby) on port $PORT')
server.serve_forever()
"
fi
