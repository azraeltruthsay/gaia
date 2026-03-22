#!/bin/bash
# gaia-prime entrypoint: GAIA Engine for the Thinker tier
#
# Starts the engine in managed "standby" mode — zero GPU, subprocess isolation.
# The orchestrator loads the model via POST /model/load when FOCUSING.
# Unloads via POST /model/unload when returning to IDLE — kills subprocess,
# frees ALL GPU memory (no zombie CUDA contexts).
#
# If PRIME_AUTOLOAD=1, loads the model on startup (for standalone use).
# If GAIA_ENGINE_DIRECT=1, uses legacy in-process mode (no subprocess isolation).

set -euo pipefail

PORT="${PRIME_PORT:-7777}"
MODEL_PATH="${PRIME_MODEL_PATH:-/models/Huihui-Qwen3-8B-GAIA-Prime-adaptive}"
DEVICE="${PRIME_DEVICE:-cuda}"
AUTOLOAD="${PRIME_AUTOLOAD:-0}"
GAIA_ENGINE_DIRECT="${GAIA_ENGINE_DIRECT:-0}"

mkdir -p /shared/thoughts 2>/dev/null || true

# ── Direct mode (legacy) ─────────────────────────────────────────────────────
if [ "$GAIA_ENGINE_DIRECT" = "1" ]; then
    if [ "$AUTOLOAD" = "1" ] && [ -d "$MODEL_PATH" ]; then
        echo "[prime-entrypoint] Starting GAIA Engine DIRECT with model auto-load..."
        echo "[prime-entrypoint] Model: $MODEL_PATH"
        exec python -c "from gaia_common.engine import serve; serve('$MODEL_PATH', $PORT, '$DEVICE')"
    else
        echo "[prime-entrypoint] Starting GAIA Engine DIRECT in standby (legacy inline handler)..."
        # Fall back to the old inline Python standby handler
        exec python -c "
from gaia_common.engine.core import GAIAEngine, serve
from http.server import HTTPServer, BaseHTTPRequestHandler
import json, logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s [%(levelname)s] %(name)s: %(message)s')
logger = logging.getLogger('GAIA.Prime')
_engine = None
class H(BaseHTTPRequestHandler):
    def log_message(self, fmt, *a): pass
    def _json(self, d, s=200):
        b = json.dumps(d).encode()
        self.send_response(s); self.send_header('Content-Type','application/json'); self.send_header('Content-Length',str(len(b))); self.end_headers(); self.wfile.write(b)
    def do_GET(self):
        if self.path=='/health': self._json({'status':'ok','model_loaded':_engine is not None,'mode':'active' if _engine else 'standby'})
        else: self._json({'error':'not found'},404)
    def do_POST(self):
        global _engine
        if self.path=='/model/load':
            n=int(self.headers.get('Content-Length',0)); b=json.loads(self.rfile.read(n)) if n else {}
            try:
                _engine=GAIAEngine(b.get('model','$MODEL_PATH'),device=b.get('device','$DEVICE'))
                self._json({'ok':True})
            except Exception as e: self._json({'ok':False,'error':str(e)},500)
        elif self.path=='/model/unload':
            import gc,torch; _engine=None; gc.collect(); torch.cuda.empty_cache() if torch.cuda.is_available() else None
            self._json({'ok':True})
        else: self._json({'error':'not found'},404)
server=HTTPServer(('0.0.0.0',$PORT),H); logger.info('Prime standby (legacy) on port $PORT'); server.serve_forever()
"
    fi
    exit 0
fi

# ── Managed mode (default) — zero-GPU subprocess isolation ───────────────────
echo "[prime-entrypoint] Starting GAIA Engine Manager (zero-GPU standby) on port $PORT..."
echo "[prime-entrypoint] Model will be loaded by orchestrator via POST /model/load"

python -m gaia_common.engine --managed --port "$PORT" &
MANAGER_PID=$!

# Wait for manager to be ready (fast — no model loading needed)
for i in $(seq 1 15); do
    if curl -sf "http://localhost:$PORT/health" > /dev/null 2>&1; then
        echo "[prime-entrypoint] Engine Manager healthy after ${i}s"
        break
    fi
    if ! kill -0 "$MANAGER_PID" 2>/dev/null; then
        echo "[prime-entrypoint] WARNING: Engine Manager exited prematurely"
        break
    fi
    sleep 1
done

# Auto-load if requested
if [ "$AUTOLOAD" = "1" ] && [ -d "$MODEL_PATH" ]; then
    echo "[prime-entrypoint] Auto-loading model: $MODEL_PATH (device=$DEVICE)"
    LOAD_RESULT=$(curl -sf -X POST "http://localhost:$PORT/model/load" \
        -H "Content-Type: application/json" \
        -d "{\"model\":\"$MODEL_PATH\",\"device\":\"$DEVICE\"}" \
        2>&1) || true
    echo "[prime-entrypoint] Model load result: $LOAD_RESULT"
fi

# Wait for manager process (keep container alive)
wait "$MANAGER_PID"
