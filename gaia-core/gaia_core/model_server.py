"""
model_server.py — Embedded llama-server lifecycle manager.

Manages the llama-server subprocess inside gaia-core. The entrypoint.sh
starts it initially and writes the PID to /tmp/llama_server.pid; this
module takes over management at runtime for release/reload operations
triggered by the self-awareness training pipeline.
"""

import logging
import os
import signal
import subprocess
import time
from pathlib import Path

logger = logging.getLogger("GAIA.Core.ModelServer")


class ModelServer:
    """Manages the embedded llama-server subprocess lifecycle."""

    PID_FILE = "/tmp/llama_server.pid"

    def __init__(self):
        self._process: subprocess.Popen | None = None
        self.model_path: str | None = None
        self._pid: int | None = None
        self._started_at: float | None = None
        self._read_pid_file()

    def _read_pid_file(self):
        """Read PID from file written by entrypoint.sh, or scan /proc."""
        try:
            pid_path = Path(self.PID_FILE)
            if pid_path.exists():
                pid = int(pid_path.read_text().strip())
                os.kill(pid, 0)
                self._pid = pid
                self._started_at = pid_path.stat().st_mtime
                self.model_path = os.environ.get(
                    "CORE_CPU_MODEL_PATH",
                    "/models/Qwen3.5-4B-Abliterated-Q4_K_M.gguf",
                )
                logger.info("Adopted llama-server PID %d from PID file", pid)
                return
        except (ValueError, OSError, FileNotFoundError):
            pass

        # Fallback: scan /proc for llama-server (covers pre-PID-file containers)
        self._find_llama_server_proc()

    def _find_llama_server_proc(self):
        """Scan /proc to find a running llama-server process."""
        proc = Path("/proc")
        if not proc.exists():
            return
        for entry in proc.iterdir():
            if not entry.name.isdigit():
                continue
            try:
                cmdline = (entry / "cmdline").read_bytes().decode("utf-8", errors="replace")
                if "llama-server" in cmdline and "--port" in cmdline:
                    pid = int(entry.name)
                    os.kill(pid, 0)
                    self._pid = pid
                    self._started_at = (entry / "stat").stat().st_mtime
                    self.model_path = os.environ.get(
                        "CORE_CPU_MODEL_PATH",
                        "/models/Qwen3.5-4B-Abliterated-Q4_K_M.gguf",
                    )
                    # Write PID file for future use
                    try:
                        Path(self.PID_FILE).write_text(str(pid))
                    except OSError:
                        pass
                    logger.info("Found llama-server via /proc scan (PID %d)", pid)
                    return
            except (OSError, PermissionError):
                continue

    def _is_running(self) -> bool:
        """Check if the llama-server process is alive."""
        if self._process is not None:
            return self._process.poll() is None
        if self._pid is not None:
            try:
                os.kill(self._pid, 0)
                return True
            except OSError:
                self._pid = None
                return False
        return False

    def _get_pid(self) -> int | None:
        """Return the active PID."""
        if self._process is not None and self._process.poll() is None:
            return self._process.pid
        if self._pid is not None:
            try:
                os.kill(self._pid, 0)
                return self._pid
            except OSError:
                self._pid = None
        return None

    def release(self) -> dict:
        """Stop llama-server and free RAM.

        Returns a status dict describing the outcome.
        """
        pid = self._get_pid()
        if pid is None:
            return {"ok": True, "message": "llama-server not running"}

        logger.info("Releasing llama-server (PID %d)...", pid)

        # Try graceful SIGTERM first
        try:
            os.kill(pid, signal.SIGTERM)
        except OSError as e:
            return {"ok": False, "error": f"Failed to send SIGTERM: {e}"}

        # Wait up to 10s for graceful shutdown
        for _ in range(100):
            try:
                os.kill(pid, 0)
                time.sleep(0.1)
            except OSError:
                break
        else:
            # Force kill if still alive
            logger.warning("llama-server did not exit gracefully, sending SIGKILL")
            try:
                os.kill(pid, signal.SIGKILL)
                time.sleep(0.5)
            except OSError:
                pass

        # Clean up state
        self._process = None
        self._pid = None
        old_model = self.model_path
        self.model_path = None
        self._started_at = None

        # Remove PID file
        try:
            Path(self.PID_FILE).unlink(missing_ok=True)
        except OSError:
            pass

        logger.info("llama-server released (was serving %s)", old_model)
        return {"ok": True, "message": f"llama-server stopped (was PID {pid})", "previous_model": old_model}

    def reload(self, model_path: str | None = None) -> dict:
        """Start a new llama-server, optionally with a different GGUF.

        If already running, stops the current instance first.
        Returns a status dict with the outcome.
        """
        # Kill existing if running
        if self._is_running():
            result = self.release()
            if not result.get("ok"):
                return result

        # Read config from environment
        port = os.environ.get("CORE_CPU_PORT", "8092")
        model = model_path or os.environ.get(
            "CORE_CPU_MODEL_PATH",
            "/models/Qwen3.5-4B-Abliterated-Q4_K_M.gguf",
        )
        ctx_size = os.environ.get("CORE_CPU_CTX", "4096")
        threads = os.environ.get("CORE_CPU_THREADS", "8")
        slot_save_path = os.environ.get("CORE_CPU_SLOT_SAVE_PATH", "/shared/kvcache/core")

        if not Path(model).exists():
            return {"ok": False, "error": f"Model not found: {model}"}

        # Ensure slot save path exists
        Path(slot_save_path).mkdir(parents=True, exist_ok=True)

        logger.info("Starting llama-server: model=%s port=%s", model, port)

        cmd = [
            "llama-server",
            "--host", "0.0.0.0",
            "--port", port,
            "--model", model,
            "--ctx-size", ctx_size,
            "--threads", threads,
            "--n-gpu-layers", "0",
            "--chat-template", "chatml",
            "--slot-save-path", slot_save_path,
        ]

        try:
            self._process = subprocess.Popen(
                cmd,
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
            )
        except FileNotFoundError:
            return {"ok": False, "error": "llama-server binary not found"}

        # Write PID file
        pid = self._process.pid
        self._pid = pid
        self.model_path = model
        self._started_at = time.time()
        try:
            Path(self.PID_FILE).write_text(str(pid))
        except OSError:
            pass

        # Poll /health up to 120s
        import urllib.request
        import urllib.error

        health_url = f"http://localhost:{port}/health"
        for i in range(120):
            if self._process.poll() is not None:
                return {"ok": False, "error": "llama-server exited during startup"}
            try:
                resp = urllib.request.urlopen(health_url, timeout=2)
                if resp.status == 200:
                    logger.info("llama-server healthy after %ds (PID %d)", i + 1, pid)
                    return {
                        "ok": True,
                        "pid": pid,
                        "model_path": model,
                        "startup_seconds": i + 1,
                    }
            except (urllib.error.URLError, OSError):
                pass
            time.sleep(1)

        # Timeout — kill the process
        logger.error("llama-server failed to become healthy in 120s")
        self.release()
        return {"ok": False, "error": "llama-server health check timed out after 120s"}

    def status(self) -> dict:
        """Return current llama-server status."""
        pid = self._get_pid()
        running = pid is not None
        uptime = None
        if running and self._started_at:
            uptime = round(time.time() - self._started_at, 1)

        return {
            "running": running,
            "pid": pid,
            "model_path": self.model_path,
            "uptime_seconds": uptime,
        }


# Singleton
_model_server: ModelServer | None = None


def get_model_server() -> ModelServer:
    """Get the singleton ModelServer instance."""
    global _model_server
    if _model_server is None:
        _model_server = ModelServer()
    return _model_server
