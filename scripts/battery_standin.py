#!/usr/bin/env python3
"""A stand-in for gaia-core's /api/cognitive/query, pointed at any engine (2qus).

The cognitive battery's direct mode posts each test to gaia-core, which adds
its framing (default system prompt, live system state, architecture facts)
and asks the tier's engine. To test a model that isn't loaded in gaia-core
(a candidate Core in a throwaway engine), this adds the same framing and
forwards to --engine. Point it at today's Core as well for a like-for-like
comparison.

Run where gaia_common is importable (gaia-core-candidate), then:
    python scripts/battery_standin.py --engine http://core12b-test:8099 --port 8199
    python gaia-doctor/cognitive_test_battery.py --endpoint http://localhost:8199 --target core --no-think
"""
import argparse
import json
import os
import urllib.request
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

DEFAULT_SYSTEM = "You are GAIA, a sovereign AI agent. Answer concisely and accurately."


def framed_system(system: str, no_think: bool) -> str:
    """The system prompt gaia-core's /api/cognitive/query builds (main.py)."""
    content = system or DEFAULT_SYSTEM
    try:
        from gaia_common.utils.world_state import format_world_state_snapshot
        ws = format_world_state_snapshot(max_lines=6)
        if ws:
            content += f"\n\nCurrent System State:\n{ws}"
    except Exception:
        pass
    facts_path = Path(os.environ.get("KNOWLEDGE_DIR", "/knowledge")) / "awareness" / "operational" / "architecture_facts.md"
    if facts_path.exists():
        facts = facts_path.read_text(encoding="utf-8").strip()
        content += "\n\nArchitecture Reference:\n" + (facts[:1200] + "\n..." if len(facts) > 1200 else facts)
    return content + (" /no_think" if no_think else "")


def make_handler(engine: str, timeout: float):
    class Handler(BaseHTTPRequestHandler):
        def log_message(self, *a):
            pass

        def _send(self, obj, code=200):
            body = json.dumps(obj).encode()
            self.send_response(code)
            self.send_header("Content-Type", "application/json")
            self.send_header("Content-Length", str(len(body)))
            self.end_headers()
            self.wfile.write(body)

        def do_GET(self):
            self._send({"status": "healthy", "service": "battery-standin", "engine": engine})

        def do_POST(self):
            if self.path != "/api/cognitive/query":
                return self._send({"error": "not found"}, 404)
            req = json.loads(self.rfile.read(int(self.headers.get("Content-Length", 0))) or b"{}")
            no_think = bool(req.get("no_think"))
            payload = {"messages": [{"role": "system", "content": framed_system(req.get("system", DEFAULT_SYSTEM), no_think)},
                                    {"role": "user", "content": req.get("prompt", "")}],
                       "max_tokens": req.get("max_tokens", 512), "temperature": req.get("temperature", 0.3)}
            if no_think:
                payload["chat_template_kwargs"] = {"enable_thinking": False}
            try:
                r = urllib.request.Request(f"{engine.rstrip('/')}/v1/chat/completions", json.dumps(payload).encode(),
                                           {"Content-Type": "application/json"})
                data = json.load(urllib.request.urlopen(r, timeout=timeout))
                text = data["choices"][0]["message"].get("content") or ""
                self._send({"content": text.strip(), "target": req.get("target", "core"), "tool_call": False})
            except Exception as e:
                self._send({"content": "", "error": str(e), "target": req.get("target", "core")})
    return Handler


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--engine", required=True, help="engine base URL, e.g. http://core12b-test:8099")
    ap.add_argument("--port", type=int, default=8199)
    ap.add_argument("--timeout", type=float, default=180)
    a = ap.parse_args()
    ThreadingHTTPServer(("0.0.0.0", a.port), make_handler(a.engine, a.timeout)).serve_forever()
